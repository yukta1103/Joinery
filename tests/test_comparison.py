from __future__ import annotations

from pathlib import Path

import pytest

from joinery.comparison import gold_is_ordered, normalize_value, results_match
from joinery.config import ExecutionConfig
from joinery.execution import execute_sql


def match(pred: list[tuple], gold: list[tuple], ordered: bool = False, digits: int = 6) -> bool:
    return results_match(pred, gold, ordered=ordered, float_digits=digits)


def test_identical_results_match() -> None:
    assert match([(1, "a"), (2, "b")], [(1, "a"), (2, "b")])


def test_unordered_ignores_row_order() -> None:
    assert match([(2, "b"), (1, "a")], [(1, "a"), (2, "b")])


def test_ordered_respects_row_order() -> None:
    assert not match([(2, "b"), (1, "a")], [(1, "a"), (2, "b")], ordered=True)
    assert match([(1, "a"), (2, "b")], [(1, "a"), (2, "b")], ordered=True)


def test_duplicates_count_in_multiset() -> None:
    assert not match([(1,), (1,), (2,)], [(1,), (2,)])
    assert not match([(1,), (2,)], [(1,), (1,), (2,)])
    assert match([(1,), (2,), (1,)], [(1,), (1,), (2,)])


def test_duplicates_count_when_ordered() -> None:
    assert not match([(1,), (1,)], [(1,)], ordered=True)


def test_nulls_equal_each_other() -> None:
    assert match([(None, 1)], [(None, 1)])
    assert not match([(None,)], [(0,)])
    assert not match([(None,)], [("",)])


def test_int_float_equivalence() -> None:
    assert match([(1,)], [(1.0,)])
    assert match([(3.0, 2)], [(3, 2.0)])


def test_float_rounding() -> None:
    assert match([(0.1 + 0.2,)], [(0.3,)])
    assert match([(2.0000001,)], [(2,)])
    assert not match([(2.001,)], [(2.0,)])
    assert not match([(1.0000004,)], [(1.0,)], digits=7)


def test_float_rounding_boundary_is_a_known_limitation() -> None:
    # Values 1e-6 apart that straddle a rounding boundary do not match; documented in the README.
    assert not match([(2.5000006,)], [(2.4999996,)])


def test_non_finite_floats() -> None:
    assert match([(float("inf"),)], [(float("inf"),)])
    assert not match([(float("inf"),)], [(float("-inf"),)])


def test_text_number_not_equal_to_number() -> None:
    assert not match([("1",)], [(1,)])


def test_strings_case_and_whitespace_sensitive() -> None:
    assert not match([("USA",)], [("usa",)])
    assert not match([("USA ",)], [("USA",)])


def test_column_order_matters() -> None:
    assert not match([("Joe", 1)], [(1, "Joe")])


def test_column_count_must_match() -> None:
    assert not match([(1, "Joe")], [(1,)])


def test_empty_results() -> None:
    assert match([], [])
    assert match([], [], ordered=True)
    assert not match([(1,)], [])
    assert not match([], [(1,)])


def test_bytes_values() -> None:
    assert match([(b"\x00\x01",)], [(b"\x00\x01",)])
    assert not match([(b"\x00",)], [("\x00",)])


def test_normalize_value_only_touches_floats() -> None:
    assert normalize_value(1, 2) == 1 and isinstance(normalize_value(1, 2), int)
    assert normalize_value("1.23456", 2) == "1.23456"
    assert normalize_value(None, 2) is None
    assert normalize_value(1.23456, 2) == 1.23


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT name FROM singer ORDER BY age", True),
        ("SELECT name FROM singer", False),
        ("SELECT name FROM singer WHERE age = (SELECT max(age) FROM singer ORDER BY age)", False),
    ],
)
def test_gold_is_ordered(sql: str, expected: bool) -> None:
    assert gold_is_ordered(sql) is expected


def test_end_to_end_equivalent_queries(db_path: Path, exec_cfg: ExecutionConfig) -> None:
    gold = execute_sql("SELECT name FROM singer WHERE country = 'USA'", db_path, exec_cfg)
    pred = execute_sql(
        "SELECT T1.name FROM singer AS T1 WHERE T1.country LIKE 'USA' ORDER BY name DESC",
        db_path,
        exec_cfg,
    )
    assert gold.rows is not None and pred.rows is not None
    assert results_match(pred.rows, gold.rows, ordered=False, float_digits=6)


def test_end_to_end_avg_vs_manual(db_path: Path, exec_cfg: ExecutionConfig) -> None:
    gold = execute_sql("SELECT avg(age) FROM singer", db_path, exec_cfg)
    pred = execute_sql("SELECT sum(age) * 1.0 / count(*) FROM singer", db_path, exec_cfg)
    assert gold.rows is not None and pred.rows is not None
    assert results_match(pred.rows, gold.rows, ordered=False, float_digits=6)
