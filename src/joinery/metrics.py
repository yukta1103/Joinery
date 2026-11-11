"""Aggregation of per-example prediction records into accuracy reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from joinery.spider_hardness import DIFFICULTY_LEVELS

Record = Mapping[str, Any]


def _bucket(records: Sequence[Record]) -> dict[str, Any]:
    correct = sum(bool(r["correct"]) for r in records)
    return {"n": len(records), "correct": correct, "accuracy": correct / len(records) if records else None}


def summarize(records: Sequence[Record], warn_prompt_tokens: int) -> dict[str, Any]:
    """Summarize prediction records.

    Examples whose gold query failed are *excluded*. ``execution_accuracy`` is computed over the
    scored examples only. ``execution_accuracy_incl_excluded`` counts excluded examples as wrong,
    which gives a lower bound comparable to evaluations that never filter.
    """
    scored = [r for r in records if not r["excluded"]]
    n_correct = sum(bool(r["correct"]) for r in scored)
    rewards = [r["reward"] for r in scored]
    return {
        "n_records": len(records),
        "n_excluded": len(records) - len(scored),
        "n_scored": len(scored),
        "n_correct": n_correct,
        "execution_accuracy": n_correct / len(scored) if scored else None,
        "execution_accuracy_incl_excluded": n_correct / len(records) if records else None,
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
