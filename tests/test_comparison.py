from __future__ import annotations

import itertools
import time
from pathlib import Path

import pytest

from joinery.comparison import MatchOutcome, compare_results, gold_is_ordered, normalize_value
from joinery.config import ExecutionConfig
from joinery.execution import execute_sql


BUDGET = 1_000_000


def outcome(
    pred: list[tuple],
    gold: list[tuple],
    ordered: bool = False,
    digits: int = 6,
    permute: bool = True,
    budget: int = BUDGET,
) -> MatchOutcome:
    return compare_results(
        pred, gold, ordered=ordered, float_digits=digits, permute_columns=permute, search_budget=budget
    )


def match(
    pred: list[tuple], gold: list[tuple], ordered: bool = False, digits: int = 6, permute: bool = True
) -> bool:
    return outcome(pred, gold, ordered, digits, permute) is MatchOutcome.MATCH


def strict(pred: list[tuple], gold: list[tuple], ordered: bool = False) -> bool:
    return match(pred, gold, ordered=ordered, permute=False)


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


def test_swapped_columns_match_by_default_but_not_strict() -> None:
    pred = [("Joe", 1), ("Ann", 2)]
    gold = [(1, "Joe"), (2, "Ann")]
    assert match(pred, gold)
    assert not strict(pred, gold)


def test_swapped_columns_with_unordered_rows() -> None:
    assert match([("Ann", 2), ("Joe", 1)], [(1, "Joe"), (2, "Ann")])


def test_swapped_columns_with_ordered_rows() -> None:
    gold = [(1, "Joe"), (2, "Ann")]
    assert match([("Joe", 1), ("Ann", 2)], gold, ordered=True)
    assert not match([("Ann", 2), ("Joe", 1)], gold, ordered=True)
    assert not strict([("Joe", 1), ("Ann", 2)], gold, ordered=True)


def test_three_column_rotation() -> None:
    gold = [(1, "a", 1.5), (2, "b", None)]
    pred = [("a", 1.5, 1), ("b", None, 2)]
    assert match(pred, gold)
    assert not strict(pred, gold)


def test_permutation_must_be_consistent_across_rows() -> None:
    # Each column holds the right values, but no single permutation pairs them up correctly.
    assert not match([(1, "b"), (2, "a")], [(1, "a"), (2, "b")])
    assert not match([(1, 1), (2, 2)], [(1, 2), (2, 1)])


def test_duplicate_column_values() -> None:
    gold = [(1, 1, 2), (3, 3, 4)]
    assert match([(1, 2, 1), (3, 4, 3)], gold)
    assert match([(2, 1, 1), (4, 3, 3)], gold)
    assert not match([(1, 2, 2), (3, 4, 4)], gold)


def test_identical_columns_whose_values_differ_per_row() -> None:
    gold = [(1, 2, 2), (5, 6, 7)]
    assert match([(2, 1, 2), (6, 5, 7)], gold)
    assert match([(2, 2, 1), (7, 6, 5)], gold)
    assert not match([(2, 2, 1), (6, 6, 5)], gold)


def test_permutation_with_duplicate_rows() -> None:
    gold = [(1, "a"), (1, "a"), (2, "b")]
    assert match([("a", 1), ("b", 2), ("a", 1)], gold)
    assert not match([("a", 1), ("b", 2), ("b", 2)], gold)


def test_permutation_with_floats_and_nulls() -> None:
    gold = [(0.3, None), (1, "x")]
    assert match([(None, 0.1 + 0.2), ("x", 1.0)], gold)


@pytest.mark.parametrize("permute", [True, False])
def test_column_count_must_match(permute: bool) -> None:
    assert not match([(1, "Joe")], [(1,)], permute=permute)
    assert not match([(1,)], [(1, "Joe")], permute=permute)
    assert not match([(1, "Joe", "Joe")], [(1, "Joe")], permute=permute)


@pytest.mark.parametrize("permute", [True, False])
def test_row_count_must_match(permute: bool) -> None:
    assert not match([(1, "a")], [(1, "a"), (1, "a")], permute=permute)


def test_empty_results_match_regardless_of_columns() -> None:
    assert match([], [])
    assert strict([], [])


def test_permutation_search_handles_wide_identical_columns() -> None:
    # 10 identical columns: 10! permutations, but interchangeable columns are tried only once.
    gold = [tuple([i] * 10) for i in range(50)]
    pred = [row[::-1] for row in gold]
    assert match(pred, gold)
    assert not match([row[:-1] + (-1,) for row in gold], gold)


def test_permutation_search_handles_wide_same_multiset_columns() -> None:
    # 9 columns with identical value multisets but distinct vectors: partial-row pruning keeps
    # the search small instead of trying 9! permutations.
    gold = [tuple((r + c) % 9 for c in range(9)) for r in range(9)]
    assert match([row[::-1] for row in gold], gold)
    assert match([row[1:] + row[:1] for row in gold], gold)
    assert not match([row[:-1] + (99,) for row in gold], gold)
    # Every column still holds 0..8, but rows 0 and 1 swap their last values: no permutation fits.
    tampered = [list(row) for row in gold]
    tampered[0][8], tampered[1][8] = tampered[1][8], tampered[0][8]
    assert not match([tuple(row) for row in tampered], gold)


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
    assert match(list(pred.rows), list(gold.rows))


def test_end_to_end_avg_vs_manual(db_path: Path, exec_cfg: ExecutionConfig) -> None:
    gold = execute_sql("SELECT avg(age) FROM singer", db_path, exec_cfg)
    pred = execute_sql("SELECT sum(age) * 1.0 / count(*) FROM singer", db_path, exec_cfg)
    assert gold.rows is not None and pred.rows is not None
    assert match(list(pred.rows), list(gold.rows))


def parity_tables(n: int) -> tuple[list[tuple], list[tuple]]:
    """Adversarial input for the permutation search: all even- vs all odd-parity n-bit rows.

    Every column holds the same multiset in both tables, and every projection onto fewer than n
    columns is the full set of shorter bit tuples in both. No partial check can prune before the
    last column, so the search must try every permutation, and none matches.
    """
    rows = list(itertools.product((0, 1), repeat=n))
    gold = [r for r in rows if sum(r) % 2 == 0]
    pred = [r for r in rows if sum(r) % 2 == 1]
    return pred, gold


def test_adversarial_search_proves_no_match_within_budget() -> None:
    pred, gold = parity_tables(6)
    assert outcome(pred, gold) is MatchOutcome.NO_MATCH


@pytest.mark.parametrize("n", [8, 10, 12])
def test_adversarial_search_hits_budget_quickly(n: int) -> None:
    # Without the budget, n=8 took ~13 s and n=9 ~255 s on a laptop; the cap bounds every width.
    pred, gold = parity_tables(n)
    start = time.perf_counter()
    assert outcome(pred, gold) is MatchOutcome.SEARCH_LIMIT
    assert time.perf_counter() - start < 5.0


def test_search_limit_is_deterministic() -> None:
    pred, gold = parity_tables(8)
    assert {outcome(pred, gold) for _ in range(3)} == {MatchOutcome.SEARCH_LIMIT}


def test_tiny_budget_reports_limit_not_mismatch() -> None:
    # A real match that needs the search: with too little budget it must say "limit", not "no".
    pred = [("Joe", 1), ("Ann", 2)]
    gold = [(1, "Joe"), (2, "Ann")]
    assert outcome(pred, gold, budget=1) is MatchOutcome.SEARCH_LIMIT
    assert outcome(pred, gold, budget=100) is MatchOutcome.MATCH


def test_budget_not_spent_when_no_search_needed() -> None:
    gold = [(1, "Joe"), (2, "Ann")]
    assert outcome(list(gold), gold, budget=1) is MatchOutcome.MATCH
    assert outcome([(1, "x")], gold, budget=1) is MatchOutcome.NO_MATCH  # row count differs
    assert outcome([("x", 1), ("y", 2)], gold, budget=1) is MatchOutcome.NO_MATCH  # column values differ


def test_strict_never_searches() -> None:
    pred, gold = parity_tables(10)
    assert outcome(pred, gold, permute=False, budget=1) is MatchOutcome.NO_MATCH


def test_ordered_search_is_linear_even_when_wide() -> None:
    # Ordered rows: equal column vectors decide everything, so no budget is consumed.
    width = 200
    gold = [tuple(r * width + c for c in range(width)) for r in range(20)]
    pred = [row[::-1] for row in gold]
    assert outcome(pred, gold, ordered=True, budget=1) is MatchOutcome.MATCH


def test_wide_result_with_identical_columns_is_cheap() -> None:
    # 200 identical columns over 1000 rows: the identical-vector rule tries one candidate per step.
    gold = [tuple([i] * 200) for i in range(1000)]
    pred = list(reversed(gold))
    start = time.perf_counter()
    assert outcome(pred, gold) is MatchOutcome.MATCH
    assert time.perf_counter() - start < 5.0
