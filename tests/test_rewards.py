from __future__ import annotations

from pathlib import Path

import pytest

from joinery.config import ComparisonConfig, ExecutionConfig, RewardConfig
from joinery.execution import ExecErrorType
from joinery.extraction import ExtractionStatus
from joinery.rewards import execution_reward, tiered_reward
from joinery.scoring import GoldExecutionError, parses_as_query, score_completion

GOLD = "SELECT name FROM singer WHERE country = 'USA'"
ORDERED_GOLD = "SELECT name FROM singer ORDER BY age DESC, name"


def block(sql: str) -> str:
    return f"Some reasoning.\n```sql\n{sql}\n```"


@pytest.fixture
def reward(db_path: Path, exec_cfg: ExecutionConfig, cmp_cfg: ComparisonConfig, reward_cfg: RewardConfig):
    def _reward(completion: str, gold: str = GOLD) -> float:
        return execution_reward(completion, gold, db_path, exec_cfg, cmp_cfg, reward_cfg)

    return _reward


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        (block("SELECT name FROM singer WHERE country = 'USA'"), 1.0),
        (block("SELECT T1.name FROM singer T1 WHERE T1.country = 'USA' ORDER BY 1"), 1.0),
        (block("SELECT name FROM singer"), 0.3),
        (block("SELECT name, country FROM singer WHERE country = 'USA'"), 0.3),
        (block("SELECT name FROM singers WHERE country = 'USA'"), 0.1),
        (block("SELEC name FROM singer"), 0.0),
        (block("hello world"), 0.0),
        ("SELECT name FROM singer WHERE country = 'USA'", 0.0),
        ("", 0.0),
        (block(""), 0.0),
        (block("DELETE FROM singer"), 0.0),
        (block("SELECT 1; SELECT 2"), 0.0),
    ],
)
def test_reward_tiers(reward, completion: str, expected: float) -> None:
    assert reward(completion) == expected


def test_ordered_gold_requires_order(reward) -> None:
    assert reward(block("SELECT name FROM singer ORDER BY age DESC, name"), ORDERED_GOLD) == 1.0
    assert reward(block("SELECT name FROM singer ORDER BY age ASC, name"), ORDERED_GOLD) == 0.3


def test_timeout_gets_parse_tier_only(db_path: Path, cmp_cfg: ComparisonConfig, reward_cfg: RewardConfig) -> None:
    cfg = ExecutionConfig(timeout_s=0.1, max_rows=1000)
    runaway = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    score = score_completion(block(runaway), GOLD, db_path, cfg, cmp_cfg)
    assert score.pred is not None and score.pred.error_type is ExecErrorType.TIMEOUT
    assert tiered_reward(score, reward_cfg) == reward_cfg.parse_reward


def test_unclosed_block_can_still_be_correct(reward) -> None:
    completion = "Reasoning\n```sql\nSELECT name FROM singer WHERE country = 'USA'"
    assert reward(completion) == 1.0


def test_reward_is_deterministic(reward) -> None:
    completion = block("SELECT name FROM singer WHERE age = 30")
    assert len({reward(completion) for _ in range(5)}) == 1


def test_match_tier_iff_scored_correct(
    db_path: Path, exec_cfg: ExecutionConfig, cmp_cfg: ComparisonConfig, reward_cfg: RewardConfig
) -> None:
    candidates = [
        "SELECT name FROM singer WHERE country = 'USA'",
        "SELECT name FROM singer",
        "SELECT name FROM singer WHERE country = 'UK'",
        "SELECT nme FROM singer",
        "SELECT DISTINCT name FROM singer WHERE country IN ('USA')",
    ]
    for sql in candidates:
        score = score_completion(block(sql), GOLD, db_path, exec_cfg, cmp_cfg)
        assert (tiered_reward(score, reward_cfg) == reward_cfg.match_reward) == score.correct


def test_score_fields(db_path: Path, exec_cfg: ExecutionConfig, cmp_cfg: ComparisonConfig) -> None:
    score = score_completion(block("SELECT name FROM singer"), GOLD, db_path, exec_cfg, cmp_cfg)
    assert score.extraction.status is ExtractionStatus.OK
    assert score.parses
    assert score.pred is not None and score.pred.ok
    assert score.gold.rows == (("Joe",), ("Bo",))
    assert not score.gold_ordered
    assert not score.correct


def test_no_sql_score(db_path: Path, exec_cfg: ExecutionConfig, cmp_cfg: ComparisonConfig) -> None:
    score = score_completion("no idea", GOLD, db_path, exec_cfg, cmp_cfg)
    assert score.pred is None and not score.parses and not score.correct


def test_gold_failure_raises(db_path: Path, exec_cfg: ExecutionConfig, cmp_cfg: ComparisonConfig) -> None:
    with pytest.raises(GoldExecutionError):
        score_completion(block("SELECT 1"), "SELECT * FROM missing", db_path, exec_cfg, cmp_cfg)


def test_empty_gold_matches_empty_prediction(reward) -> None:
    empty_gold = "SELECT name FROM singer WHERE age > 100"
    assert reward(block("SELECT name FROM singer WHERE age < 0"), empty_gold) == 1.0


def test_custom_reward_values(db_path: Path, exec_cfg: ExecutionConfig, cmp_cfg: ComparisonConfig) -> None:
    cfg = RewardConfig(parse_reward=0.0, execute_reward=0.0, match_reward=1.0)
    assert execution_reward(block("SELECT name FROM singer"), GOLD, db_path, exec_cfg, cmp_cfg, cfg) == 0.0


@pytest.mark.parametrize(
    ("parse", "execute", "match"),
    [(-0.1, 0.3, 1.0), (0.5, 0.3, 1.0), (0.1, 1.5, 1.0)],
)
def test_reward_config_validation(parse: float, execute: float, match: float) -> None:
    with pytest.raises(ValueError):
        RewardConfig(parse_reward=parse, execute_reward=execute, match_reward=match)


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT 1", True),
        ("SELECT a FROM t UNION SELECT b FROM u", True),
        ("WITH x AS (SELECT 1) SELECT * FROM x", True),
        ("hello world", False),
        ("SELEC a FROM t", False),
        ("SELECT 1; SELECT 2", False),
        ("DELETE FROM t", False),
        ("", False),
    ],
)
def test_parses_as_query(sql: str, expected: bool) -> None:
    assert parses_as_query(sql) is expected
