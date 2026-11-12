"""Evaluation runner: generate, score and record predictions, resumable across sessions.

Outputs in ``<output_dir>/<run_name>/``:

* ``config.yaml``: the resolved config. A resumed run must match it, apart from paths and W&B
  settings.
* ``predictions.jsonl``: one record per example, appended and flushed after every batch.
* ``generation_settings.json``: model, versions and decoding settings, as reported by the generator.
* ``metrics.json``: the summary, with ``complete: false`` until every example has a record.

Resuming: rerun the same command. Records already in ``predictions.jsonl`` are kept and skipped. A
partial last line (the session died mid-write) is dropped. Pending examples are batched longest
prompt first, so an out-of-memory error shows up in the first batch. Their order is deterministic,
so a resumed run builds the same batches the uninterrupted run would have.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from joinery.config import EvalConfig, config_to_dict, save_config
from joinery.data import SpiderExample, database_path, load_examples, require_databases
from joinery.generation import Generation, Generator
from joinery.metrics import summarize
from joinery.prompts import build_messages
from joinery.rewards import tiered_reward
from joinery.scoring import Score, execute_gold, score_completion

logger = logging.getLogger(__name__)

PREDICTIONS_FILE = "predictions.jsonl"
METRICS_FILE = "metrics.json"
CONFIG_FILE = "config.yaml"
GENERATION_SETTINGS_FILE = "generation_settings.json"

# Config keys that may differ between a run and its resumption (machine-specific paths, tracking).
_RESUME_IGNORED_KEYS = (("output_dir",), ("data", "spider_dir"), ("data", "database_dir"), ("wandb",))

_GENERATION_KEYS = ("prompt", "prompt_tokens", "output", "output_tokens", "truncated")
_SCORE_KEYS = (
    "extracted_sql",
    "extraction_status",
    "parses",
    "exec_error_type",
    "exec_error_message",
    "pred_row_count",
    "gold_row_count",
    "gold_ordered",
    "correct",
    "correct_strict",
    "error_type",
    "reward",
)


@dataclass(frozen=True)
class _Job:
    example: SpiderExample
    prompt: str
    prompt_tokens: int


def error_type(score: Score) -> str | None:
    """One label per example: None if correct, else the first stage that went wrong.

    Labels are an extraction status (``no_block``, ``empty_block``), an execution error type
    (``syntax``, ``schema``, ``timeout``, ...) or ``wrong_result``.
    """
    if score.correct:
        return None
    if score.pred is None:
        return score.extraction.status.value
    if score.pred.error_type is not None:
        return score.pred.error_type.value
    return "wrong_result"


def score_fields(output: str, gold_sql: str, db_path: Path, config: EvalConfig) -> dict[str, Any]:
    """Score one model output and return the scoring fields of its record."""
    score = score_completion(output, gold_sql, db_path, config.execution, config.comparison)
    pred = score.pred
    return {
        "extracted_sql": score.extraction.sql,
        "extraction_status": score.extraction.status.value,
        "parses": score.parses,
        "exec_error_type": pred.error_type.value if pred is not None and pred.error_type else None,
        "exec_error_message": pred.error_message if pred is not None else None,
        "pred_row_count": len(pred.rows) if pred is not None and pred.rows is not None else None,
        "gold_row_count": len(score.gold.rows or ()),
        "gold_ordered": score.gold_ordered,
        "correct": score.correct,
        "correct_strict": score.correct_strict,
        "error_type": error_type(score),
        "reward": tiered_reward(score, config.reward),
    }


def _example_fields(example: SpiderExample) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "db_id": example.db_id,
        "difficulty": example.difficulty,
        "question": example.question,
        "gold_sql": example.gold_sql,
    }


def _excluded_record(example: SpiderExample, reason: str) -> dict[str, Any]:
    record = _example_fields(example)
    record.update(dict.fromkeys(_GENERATION_KEYS + _SCORE_KEYS))
    record.update(excluded=True, exclusion_reason=reason)
    return record


def _generated_record(job: _Job, generation: Generation, db_path: Path, config: EvalConfig) -> dict[str, Any]:
    record = _example_fields(job.example)
    record.update(
        prompt=job.prompt,
        prompt_tokens=job.prompt_tokens,
        output=generation.text,
        output_tokens=generation.num_tokens,
        truncated=generation.truncated,
    )
    record.update(score_fields(generation.text, job.example.gold_sql, db_path, config))
    record.update(excluded=False, exclusion_reason=None)
    return record


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a predictions file, dropping a partial last line left by an interrupted write.

    Raises:
        ValueError: If a line other than the last is not valid JSON.
    """
    if not path.is_file():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if index != len(lines) - 1:
                raise ValueError(f"{path}: corrupt record on line {index + 1}") from exc
            logger.warning("Dropping partial last record in %s (interrupted write).", path)
    return records


def write_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Write records to a JSONL file, replacing its contents."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _comparable(config_dict: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(config_dict))
    for keys in _RESUME_IGNORED_KEYS:
        node = result
        for key in keys[:-1]:
            node = node.get(key, {})
        node.pop(keys[-1], None)
    return result


def _check_resume_config(run_dir: Path, config: EvalConfig) -> None:
    path = run_dir / CONFIG_FILE
    if not path.is_file():
        return
    with open(path, encoding="utf-8") as f:
        previous = yaml.safe_load(f)
    if _comparable(previous) != _comparable(config_to_dict(config)):
        raise ValueError(
            f"{path} differs from the current config. Results from different configs must not be "
            "mixed: use a new run_name (or delete the old run directory deliberately)."
        )


def _log_wandb(config: EvalConfig, metrics: dict[str, Any]) -> None:
    import wandb

    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.run_name,
        mode=config.wandb.mode,
        config=config_to_dict(config),
        job_type="eval",
    )
    flat = {
        "execution_accuracy": metrics["execution_accuracy"],
        "execution_accuracy_strict": metrics["execution_accuracy_strict"],
        "execution_accuracy_incl_excluded": metrics["execution_accuracy_incl_excluded"],
        "mean_reward": metrics["mean_reward"],
        "n_scored": metrics["n_scored"],
        "n_excluded": metrics["n_excluded"],
    }
    for level, bucket in metrics["by_difficulty"].items():
        flat[f"accuracy/{level}"] = bucket["accuracy"]
        flat[f"accuracy_strict/{level}"] = bucket["accuracy_strict"]
    run.summary.update(flat)
    run.finish()


def finalize(run_dir: Path, examples: Sequence[SpiderExample], config: EvalConfig) -> dict[str, Any]:
    """Compute and write ``metrics.json`` from the predictions on disk."""
    records = read_records(run_dir / PREDICTIONS_FILE)
    expected = {ex.example_id for ex in examples}
    found = {r["example_id"] for r in records}
    metrics: dict[str, Any] = {
        "run_name": config.run_name,
        "model": config.model.name_or_path,
        "split": config.data.split,
        "n_expected": len(expected),
        "complete": found == expected,
    }
    metrics.update(summarize(records, config.prompt.warn_prompt_tokens))
    with open(run_dir / METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def run_eval(config: EvalConfig, make_generator: Callable[[], Generator]) -> dict[str, Any]:
    """Run (or resume) an evaluation and return its metrics.

    Args:
        config: The run configuration.
        make_generator: Called at most once, and only if there are examples left to generate, so
            a finished run never loads the model.
    """
    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    _check_resume_config(run_dir, config)
    save_config(config, run_dir / CONFIG_FILE)

    examples = load_examples(config.data)
    database_dir = config.data.resolved_database_dir()
    require_databases(examples, database_dir)

    predictions_path = run_dir / PREDICTIONS_FILE
    records = read_records(predictions_path)
    known = {ex.example_id for ex in examples}
    done = {r["example_id"] for r in records}
    if len(done) != len(records) or not done <= known:
        raise ValueError(f"{predictions_path} has duplicate or unknown example ids")
    write_records(predictions_path, records)
    pending = [ex for ex in examples if ex.example_id not in done]
    logger.info("%d/%d examples already done, %d pending", len(done), len(examples), len(pending))

    generator: Generator | None = None
    jobs: list[_Job] = []
    with open(predictions_path, "a", encoding="utf-8") as out:
        for ex in pending:
            gold = execute_gold(ex.gold_sql, str(database_path(database_dir, ex.db_id)), config.execution)
            if not gold.ok:
                reason = f"gold {gold.error_type.value if gold.error_type else 'error'}: {gold.error_message}"
                out.write(json.dumps(_excluded_record(ex, reason), ensure_ascii=False) + "\n")
                continue
            if generator is None:
                generator = make_generator()
                with open(run_dir / GENERATION_SETTINGS_FILE, "w", encoding="utf-8") as f:
                    json.dump(generator.describe(), f, indent=2, default=str)
            messages = build_messages(database_path(database_dir, ex.db_id), ex.question, config.prompt)
            prompt = generator.render_prompt(messages)
            jobs.append(_Job(ex, prompt, generator.count_tokens(prompt)))
        out.flush()

        jobs.sort(key=lambda job: (-job.prompt_tokens, job.example.example_id))
        batch_size = config.generation.batch_size
        start = time.perf_counter()
        for offset in range(0, len(jobs), batch_size):
            assert generator is not None
            batch = jobs[offset : offset + batch_size]
            generations = generator.generate([job.prompt for job in batch])
            if len(generations) != len(batch):
                raise RuntimeError("generator returned a different number of outputs than prompts")
            for job, generation in zip(batch, generations):
                db_path = database_path(database_dir, job.example.db_id)
                record = _generated_record(job, generation, db_path, config)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            finished = offset + len(batch)
            elapsed = time.perf_counter() - start
            logger.info(
                "generated %d/%d (%.1fs elapsed, %.2f examples/s)",
                finished,
                len(jobs),
                elapsed,
                finished / elapsed if elapsed > 0 else float("nan"),
            )

    metrics = finalize(run_dir, examples, config)
    if metrics["complete"] and config.wandb.mode != "disabled":
        _log_wandb(config, metrics)
    return metrics


def rescore(predictions_path: Path, config: EvalConfig, out_dir: Path) -> dict[str, Any]:
    """Re-score saved outputs with the current harness (no model needed).

    Use this after changing extraction, execution or comparison rules. Excluded examples are
    checked again. If a gold query that used to fail now succeeds, the example has no output to
    score and this raises.
    """
    examples = load_examples(config.data)
    by_id = {ex.example_id: ex for ex in examples}
    database_dir = config.data.resolved_database_dir()
    rescored = []
    for record in read_records(predictions_path):
        ex = by_id[record["example_id"]]
        db_path = database_path(database_dir, ex.db_id)
        gold = execute_gold(ex.gold_sql, str(db_path), config.execution)
        if record["excluded"]:
            if gold.ok:
                raise ValueError(f"{ex.example_id} was excluded but its gold query now runs; regenerate it")
            rescored.append(record)
            continue
        if not gold.ok:
            reason = f"gold {gold.error_type.value if gold.error_type else 'error'}: {gold.error_message}"
            excluded = _excluded_record(ex, reason)
            excluded.update({k: record[k] for k in _GENERATION_KEYS})
            rescored.append(excluded)
            continue
        rescored.append({**record, **score_fields(record["output"], ex.gold_sql, db_path, config)})
    out_dir.mkdir(parents=True, exist_ok=True)
    write_records(out_dir / PREDICTIONS_FILE, rescored)
    save_config(config, out_dir / CONFIG_FILE)
    return finalize(out_dir, examples, config)
