"""Scoring of one completion against a gold query.

This is the single place where "correct" is decided. Evaluation and the reward function both call
:func:`score_completion`, so a reward at the match tier always means the evaluation counts the
completion as correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from joinery.comparison import MatchOutcome, compare_results, gold_is_ordered
from joinery.config import ComparisonConfig, ExecutionConfig
from joinery.execution import ExecResult, execute_sql
from joinery.extraction import Extraction, extract_sql


class GoldExecutionError(RuntimeError):
    """Raised when the gold query itself fails. Such examples cannot be scored."""


@dataclass(frozen=True)
class Score:
    """Everything learned while scoring one completion.

    ``correct`` is the default metric (any column permutation may match) and drives the reward.
    ``correct_strict`` additionally requires the gold column order and is reported separately.
    ``permutation_limit_hit`` means the permutation search ran out of budget. ``correct`` is then
    False because no match was found, not because a mismatch was proven.
    """

    extraction: Extraction
    parses: bool
    pred: ExecResult | None
    gold: ExecResult
    gold_ordered: bool
    correct: bool
    correct_strict: bool
    permutation_limit_hit: bool


def parses_as_query(sql: str) -> bool:
    """True if sqlglot parses ``sql`` (SQLite dialect) as exactly one query expression.

    The parser is lenient (``"hello world"`` parses as an alias), so the result must also be a
    query node such as SELECT or UNION. This only feeds the reward's "parses" tier. Correctness is
    always decided by execution.
    """
    try:
        trees = sqlglot.parse(sql, read="sqlite")
    except SqlglotError:
        return False
    return len(trees) == 1 and isinstance(trees[0], exp.Query)


@lru_cache(maxsize=16_384)
def execute_gold(gold_sql: str, db_path: str, config: ExecutionConfig) -> ExecResult:
    """Execute a gold query, cached by (query, database, limits). Gold results never change."""
    return execute_sql(gold_sql, db_path, config)


def score_completion(
    completion: str,
    gold_sql: str,
    db_path: str | Path,
    exec_config: ExecutionConfig,
    cmp_config: ComparisonConfig,
) -> Score:
    """Extract, execute and compare one completion against the gold query.

    Raises:
        GoldExecutionError: If the gold query fails to execute.
    """
    gold = execute_gold(gold_sql, str(db_path), exec_config)
    if not gold.ok or gold.rows is None:
        raise GoldExecutionError(f"gold query failed ({gold.error_type}): {gold.error_message}")
    ordered = gold_is_ordered(gold_sql)
    extraction = extract_sql(completion)
    if extraction.sql is None:
        return Score(extraction, False, None, gold, ordered, False, False, False)
    pred = execute_sql(extraction.sql, db_path, exec_config)
    strict = default = MatchOutcome.NO_MATCH
    if pred.rows is not None:
        rules = {
            "ordered": ordered,
            "float_digits": cmp_config.float_digits,
            "search_budget": cmp_config.permutation_search_budget,
        }
        strict = compare_results(pred.rows, gold.rows, permute_columns=False, **rules)
        default = (
            strict
            if strict is MatchOutcome.MATCH
            else compare_results(pred.rows, gold.rows, permute_columns=True, **rules)
        )
    return Score(
        extraction=extraction,
        parses=parses_as_query(extraction.sql),
        pred=pred,
        gold=gold,
        gold_ordered=ordered,
        correct=default is MatchOutcome.MATCH,
        correct_strict=strict is MatchOutcome.MATCH,
        permutation_limit_hit=default is MatchOutcome.SEARCH_LIMIT,
    )
