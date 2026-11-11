"""Tiered execution reward for GRPO (drafted in Phase 1, used in Phase 3).

Tiers, highest one reached wins:

* ``match_reward``: the predicted rows match the gold rows (same decision as evaluation);
* ``execute_reward``: the query runs without error but the rows differ;
* ``parse_reward``: SQL was extracted and parses as a query, but execution failed;
* ``0``: no SQL could be extracted, or it does not parse and does not run.
"""

from __future__ import annotations

from pathlib import Path

from joinery.config import ComparisonConfig, ExecutionConfig, RewardConfig
from joinery.scoring import Score, score_completion


def tiered_reward(score: Score, config: RewardConfig) -> float:
    """Map a :class:`~joinery.scoring.Score` to a scalar reward."""
    if score.correct:
        return config.match_reward
    if score.pred is not None and score.pred.ok:
        return config.execute_reward
    if score.parses:
        return config.parse_reward
    return 0.0


def execution_reward(
    completion: str,
    gold_sql: str,
    db_path: str | Path,
    exec_config: ExecutionConfig,
    cmp_config: ComparisonConfig,
    reward_config: RewardConfig,
) -> float:
    """Reward one completion. Deterministic and free of side effects except the gold cache.

    Raises:
        joinery.scoring.GoldExecutionError: If the gold query fails. Training data must be
            filtered so this never happens.
    """
    score = score_completion(completion, gold_sql, db_path, exec_config, cmp_config)
    return tiered_reward(score, reward_config)
