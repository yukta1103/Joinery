"""End-to-end tests of the evaluation runner with a fake generator (no model, no GPU)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import MINI_DEV

from joinery.config import EvalConfig, build_dataclass
from joinery.eval import CONFIG_FILE, METRICS_FILE, PREDICTIONS_FILE, read_records, rescore, run_eval
from joinery.generation import Generation
from joinery.prompts import Message

# Canned outputs keyed by question: correct, wrong result, schema error, no block, truncated.
OUTPUTS = {
    MINI_DEV[0][0]: "Count rows.\n```sql\nSELECT count(*) FROM singer;\n```",
    MINI_DEV[1][0]: "Order by age.\n```sql\nSELECT name FROM singer ORDER BY age DESC\n```",
    MINI_DEV[2][0]: "```sql\nSELECT DISTINCT countries FROM singer\n```",
    MINI_DEV[3][0]: "SELECT venue FROM concert WHERE year = 2015",
    MINI_DEV[4][0]: "Average.\n```sql\nSELECT avg(net_worth) FROM singer",
}
TRUNCATED = {MINI_DEV[4][0]}


class FakeGenerator:
    """Answers from ``OUTPUTS``. Can be told to crash after a number of generate calls."""

    def __init__(self, crash_after: int | None = None) -> None:
        self.crash_after = crash_after
        self.generated: list[str] = []
        self.calls = 0

    def render_prompt(self, messages: list[Message]) -> str:
        return "\n".join(m["content"] for m in messages)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompts: Sequence[str]) -> list[Generation]:
        if self.crash_after is not None and self.calls >= self.crash_after:
            raise RuntimeError("simulated session timeout")
        self.calls += 1
        results = []
        for prompt in prompts:
            question = prompt.rsplit("Question: ", 1)[1]
            self.generated.append(question)
            text = OUTPUTS[question]
            results.append(Generation(text, len(text.split()), truncated=question in TRUNCATED))
        return results

    def describe(self) -> dict[str, Any]:
        return {"model": "fake"}


def make_config(spider_dir: Path, output_dir: Path, **overrides: Any) -> EvalConfig:
    data: dict[str, Any] = {
        "run_name": "fake_run",
        "seed": 0,
        "output_dir": str(output_dir),
        "data": {"spider_dir": str(spider_dir), "split": "dev"},
        "model": {"name_or_path": "fake"},
        "generation": {"batch_size": 2, "max_new_tokens": 64},
        "prompt": {"sample_rows": 2, "max_value_chars": 30, "warn_prompt_tokens": 50},
        "execution": {"timeout_s": 2.0, "max_rows": 1000},
        "comparison": {"float_digits": 6},
        "reward": {"parse_reward": 0.1, "execute_reward": 0.3, "match_reward": 1.0},
    }
    for key, value in overrides.items():
        section, _, field = key.partition("__")
        if field:
            data[section][field] = value
        else:
            data[section] = value
    return build_dataclass(EvalConfig, data)


def by_question(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["question"]: r for r in records}


def test_full_run(spider_dir: Path, tmp_path: Path) -> None:
    cfg = make_config(spider_dir, tmp_path / "results")
    generator = FakeGenerator()
    metrics = run_eval(cfg, lambda: generator)

    run_dir = tmp_path / "results" / "fake_run"
    records = by_question(read_records(run_dir / PREDICTIONS_FILE))
    assert len(records) == len(MINI_DEV)

    assert records[MINI_DEV[0][0]]["correct"] and records[MINI_DEV[0][0]]["error_type"] is None
    wrong_order = records[MINI_DEV[1][0]]
    assert wrong_order["gold_ordered"] and wrong_order["error_type"] == "wrong_result"
    assert wrong_order["reward"] == 0.3
    assert records[MINI_DEV[2][0]]["error_type"] == "schema"
    assert records[MINI_DEV[3][0]]["error_type"] == "no_block"
    truncated = records[MINI_DEV[4][0]]
    assert truncated["correct"] and truncated["extraction_status"] == "unclosed_fence"
    assert truncated["truncated"]
    excluded = records[MINI_DEV[5][0]]
    assert excluded["excluded"] and "no such table" in excluded["exclusion_reason"]
    assert excluded["output"] is None

    # The excluded example is never sent to the model.
    assert MINI_DEV[5][0] not in generator.generated

    assert metrics["complete"]
    assert metrics["n_scored"] == 5 and metrics["n_excluded"] == 1 and metrics["n_correct"] == 2
    assert metrics["execution_accuracy"] == pytest.approx(2 / 5)
    assert metrics["execution_accuracy_incl_excluded"] == pytest.approx(2 / 6)
    assert metrics["by_difficulty"]["easy"] == {"n": 5, "correct": 2, "accuracy": pytest.approx(0.4)}
    assert metrics["error_types"] == {"correct": 2, "wrong_result": 1, "schema": 1, "no_block": 1}
    assert metrics["n_truncated"] == 1
    assert json.loads((run_dir / METRICS_FILE).read_text()) == metrics
    assert yaml.safe_load((run_dir / CONFIG_FILE).read_text())["run_name"] == "fake_run"


def test_record_has_all_fields(spider_dir: Path, tmp_path: Path) -> None:
    cfg = make_config(spider_dir, tmp_path / "results")
    run_eval(cfg, FakeGenerator)
    records = read_records(cfg.run_dir() / PREDICTIONS_FILE)
    keys = {frozenset(r) for r in records}
    assert len(keys) == 1, "excluded and generated records must share one schema"
    expected = {
        "example_id", "db_id", "difficulty", "question", "gold_sql", "prompt", "prompt_tokens",
        "output", "output_tokens", "truncated", "extracted_sql", "extraction_status", "parses",
        "exec_error_type", "exec_error_message", "pred_row_count", "gold_row_count",
        "gold_ordered", "correct", "error_type", "reward", "excluded", "exclusion_reason",
    }
    assert set(next(iter(keys))) == expected


def test_resume_after_crash(spider_dir: Path, tmp_path: Path) -> None:
    cfg = make_config(spider_dir, tmp_path / "results")
    reference = run_eval(make_config(spider_dir, tmp_path / "reference"), FakeGenerator)

    crashing = FakeGenerator(crash_after=1)
    with pytest.raises(RuntimeError, match="simulated"):
        run_eval(cfg, lambda: crashing)
    partial = read_records(cfg.run_dir() / PREDICTIONS_FILE)
    assert len(partial) == 3  # the excluded example plus the first batch of two

    resumed = FakeGenerator()
    metrics = run_eval(cfg, lambda: resumed)
    assert len(resumed.generated) == 3
    assert not set(resumed.generated) & set(crashing.generated)
    assert metrics == reference


def test_partial_last_line_is_dropped(spider_dir: Path, tmp_path: Path) -> None:
    cfg = make_config(spider_dir, tmp_path / "results")
    run_eval(cfg, FakeGenerator)
    path = cfg.run_dir() / PREDICTIONS_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:20], encoding="utf-8")

    resumed = FakeGenerator()
    metrics = run_eval(cfg, lambda: resumed)
    assert len(resumed.generated) == 1
    assert metrics["complete"] and metrics["n_records"] == len(MINI_DEV)


def test_corrupt_middle_line_raises(tmp_path: Path) -> None:
    path = tmp_path / PREDICTIONS_FILE
    path.write_text('{"a": 1}\nnot json\n{"b": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_records(path)


def test_finished_run_does_not_load_model(spider_dir: Path, tmp_path: Path) -> None:
    cfg = make_config(spider_dir, tmp_path / "results")
    run_eval(cfg, FakeGenerator)

    def must_not_load() -> FakeGenerator:
        raise AssertionError("model loaded for a finished run")

    assert run_eval(cfg, must_not_load)["complete"]


def test_config_change_refuses_resume(spider_dir: Path, tmp_path: Path) -> None:
    run_eval(make_config(spider_dir, tmp_path / "results"), FakeGenerator)
    changed = make_config(spider_dir, tmp_path / "results", prompt__sample_rows=0)
    with pytest.raises(ValueError, match="differs"):
        run_eval(changed, FakeGenerator)


def test_path_change_allows_resume(spider_dir: Path, tmp_path: Path) -> None:
    cfg = make_config(spider_dir, tmp_path / "results")
    run_eval(cfg, FakeGenerator)
    moved = make_config(
        spider_dir, tmp_path / "results", data={"spider_dir": str(spider_dir), "database_dir": str(spider_dir / "database"), "split": "dev"}
    )
    assert run_eval(moved, FakeGenerator)["complete"]


def test_limit_and_incomplete_metrics(spider_dir: Path, tmp_path: Path) -> None:
    cfg = make_config(spider_dir, tmp_path / "results", data={"spider_dir": str(spider_dir), "limit": 2})
    metrics = run_eval(cfg, FakeGenerator)
    assert metrics["complete"] and metrics["n_records"] == 2


def test_batches_are_longest_first(spider_dir: Path, tmp_path: Path) -> None:
    cfg = make_config(spider_dir, tmp_path / "results")
    generator = FakeGenerator()
    run_eval(cfg, lambda: generator)
    records = by_question(read_records(cfg.run_dir() / PREDICTIONS_FILE))
    lengths = [records[q]["prompt_tokens"] for q in generator.generated]
    assert lengths == sorted(lengths, reverse=True)


def test_rescore_reproduces_metrics(spider_dir: Path, tmp_path: Path) -> None:
    cfg = make_config(spider_dir, tmp_path / "results")
    original = run_eval(cfg, FakeGenerator)
    rescored = rescore(cfg.run_dir() / PREDICTIONS_FILE, cfg, tmp_path / "rescored")
    assert rescored == original
    assert read_records(tmp_path / "rescored" / PREDICTIONS_FILE) == read_records(
        cfg.run_dir() / PREDICTIONS_FILE
    )
