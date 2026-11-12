"""Aggregation of per-example prediction records into accuracy reports.

Two execution accuracies are reported side by side:

* ``execution_accuracy`` (the default metric): a prediction is correct if some permutation of its
  columns matches the gold result, as in test-suite-sql-eval;
* ``execution_accuracy_strict``: the columns must also be in the gold order.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from joinery.spider_hardness import DIFFICULTY_LEVELS

Record = Mapping[str, Any]


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _bucket(records: Sequence[Record]) -> dict[str, Any]:
    correct = sum(bool(r["correct"]) for r in records)
    correct_strict = sum(bool(r["correct_strict"]) for r in records)
    return {
        "n": len(records),
        "correct": correct,
        "accuracy": _rate(correct, len(records)),
        "correct_strict": correct_strict,
        "accuracy_strict": _rate(correct_strict, len(records)),
    }


def summarize(records: Sequence[Record], warn_prompt_tokens: int) -> dict[str, Any]:
    """Summarize prediction records.

    Examples whose gold query failed are *excluded*. ``execution_accuracy`` is computed over the
    scored examples only. ``execution_accuracy_incl_excluded`` counts excluded examples as wrong,
    which gives a lower bound comparable to evaluations that never filter.
    """
    scored = [r for r in records if not r["excluded"]]
    n_correct = sum(bool(r["correct"]) for r in scored)
    n_correct_strict = sum(bool(r["correct_strict"]) for r in scored)
    rewards = [r["reward"] for r in scored]
    return {
        "n_records": len(records),
        "n_excluded": len(records) - len(scored),
        "n_scored": len(scored),
        "n_correct": n_correct,
        "n_correct_strict": n_correct_strict,
        "execution_accuracy": _rate(n_correct, len(scored)),
        "execution_accuracy_strict": _rate(n_correct_strict, len(scored)),
        "execution_accuracy_incl_excluded": _rate(n_correct, len(records)),
        "by_difficulty": {
            level: _bucket([r for r in scored if r["difficulty"] == level]) for level in DIFFICULTY_LEVELS
        },
        "gold_empty": _bucket([r for r in scored if r["gold_row_count"] == 0]),
        "gold_nonempty": _bucket([r for r in scored if r["gold_row_count"] != 0]),
        "error_types": dict(Counter(r["error_type"] or "correct" for r in scored).most_common()),
        "extraction_status": dict(Counter(r["extraction_status"] for r in scored).most_common()),
        "n_truncated": sum(bool(r["truncated"]) for r in scored),
        "n_prompt_over_warn": sum(r["prompt_tokens"] > warn_prompt_tokens for r in scored),
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def format_headline(metrics: Mapping[str, Any]) -> str:
    """Human-readable accuracy table (default and strict metric, overall and by difficulty)."""
    lines = [
        f"{'':<10}{'n':>6}{'exec acc':>11}{'strict':>9}",
        f"{'overall':<10}{metrics['n_scored']:>6}{_percent(metrics['execution_accuracy']):>11}"
        f"{_percent(metrics['execution_accuracy_strict']):>9}",
    ]
    for level, bucket in metrics["by_difficulty"].items():
        lines.append(
            f"{level:<10}{bucket['n']:>6}{_percent(bucket['accuracy']):>11}"
            f"{_percent(bucket['accuracy_strict']):>9}"
        )
    if not metrics["complete"]:
        lines.append(f"INCOMPLETE: {metrics['n_records']}/{metrics['n_expected']} examples have records")
    if metrics["n_excluded"]:
        lines.append(f"{metrics['n_excluded']} example(s) excluded because their gold query failed")
    return "\n".join(lines)
