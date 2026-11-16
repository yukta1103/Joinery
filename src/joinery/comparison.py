"""Comparison of query results for execution accuracy.

Rules (see the README's "Comparison rules" section for the reasoning):

* Rows are compared as a **multiset** (duplicates count) unless the gold query has a top-level
  ORDER BY, in which case they are compared as an ordered list.
* Columns: by default a prediction matches if **some permutation of its columns** matches the gold
  result, as in test-suite-sql-eval. The row rule above is applied after permuting. A strict mode
  (``permute_columns=False``) requires the same column order. Column counts must be equal, and
  column names are always ignored. The permutation search has a deterministic work budget. When
  the budget runs out the outcome is ``SEARCH_LIMIT``, which never counts as correct.
* NULL equals NULL.
* Integers and floats compare by value (``1 == 1.0``). Floats are rounded to a fixed number of
  decimal places first, so ``0.1 + 0.2`` matches ``0.3``.
* Strings are compared exactly (case and whitespace sensitive). A text ``'1'`` does not equal the
  integer ``1``.
* Two empty results match, whatever their columns.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from enum import Enum
from typing import Any

from joinery.execution import Row
from joinery.sqltext import has_top_level_order_by


def normalize_value(value: Any, float_digits: int) -> Any:
    """Normalize one SQLite value for comparison. Only finite floats change (they are rounded)."""
    if isinstance(value, float) and math.isfinite(value):
        return round(value, float_digits)
    return value


def normalize_rows(rows: Sequence[Row], float_digits: int) -> list[Row]:
    """Apply :func:`normalize_value` to every cell."""
    return [tuple(normalize_value(v, float_digits) for v in row) for row in rows]


def _rows_equal(pred: list[Row], gold: list[Row], ordered: bool) -> bool:
    if ordered:
        return pred == gold
    return Counter(pred) == Counter(gold)


class MatchOutcome(str, Enum):
    """Result of comparing two result sets.

    ``SEARCH_LIMIT`` means the column-permutation search used up its work budget before finding a
    match or proving there is none. Callers must treat it as *not* correct and report it apart
    from an ordinary mismatch.
    """

    MATCH = "match"
    NO_MATCH = "no_match"
    SEARCH_LIMIT = "search_limit"


class _BudgetExhausted(Exception):
    pass


def _column_key(column: tuple[Any, ...], ordered: bool) -> Hashable:
    """A value every matching column must share: the column itself if ordered, else its multiset."""
    return column if ordered else frozenset(Counter(column).items())


def _search_permutation(pred: list[Row], gold: list[Row], ordered: bool, budget: int) -> MatchOutcome:
    """Search for a mapping of gold columns to distinct pred columns under which the rows match.

    The search assigns gold columns left to right, with these prunings:

    * a pred column is a candidate only if it holds the same values as the gold column (same
      sequence when ordered, same multiset otherwise), and the two tables must have the same
      collection of such column keys;
    * pred columns that are identical vectors are interchangeable, so only one of them is tried at
      each step;
    * when unordered, the multiset of partial rows over the assigned columns must already match.

    **Ordered** rows need no budget: equal column vectors imply equal rows, and the identical-
    vector rule leaves at most one candidate per step, so the search is O(width²) key comparisons.

    **Unordered** rows have no polynomial bound. There are tables where every projection onto
    fewer than all columns agrees, so nothing can be pruned before the last column (e.g. all even-
    parity bit rows vs all odd-parity bit rows). The search then visits every permutation. So each
    O(rows) step (building one level of partial rows, or one gold prefix multiset) spends
    ``len(rows)`` units of ``budget``, and the search returns ``SEARCH_LIMIT`` once the budget is
    gone. Counting work instead of wall-clock time keeps the outcome the same on every machine.
    """
    width = len(gold[0])
    pred_cols = [tuple(row[j] for row in pred) for j in range(width)]
    gold_cols = [tuple(row[i] for row in gold) for i in range(width)]
    # Small integer ids for column keys and column vectors keep each candidate check O(1), so the
    # only O(rows) steps inside the search are the ones charged to the budget.
    key_ids: dict[Hashable, int] = {}
    pred_keys = [key_ids.setdefault(_column_key(col, ordered), len(key_ids)) for col in pred_cols]
    gold_keys = [key_ids.setdefault(_column_key(col, ordered), len(key_ids)) for col in gold_cols]
    if Counter(pred_keys) != Counter(gold_keys):
        return MatchOutcome.NO_MATCH
    vector_ids: dict[tuple[Any, ...], int] = {}
    pred_vectors = [vector_ids.setdefault(col, len(vector_ids)) for col in pred_cols]

    remaining = budget
    gold_prefixes: list[Counter[Row]] = []

    def spend() -> None:
        nonlocal remaining
        remaining -= len(gold)
        if remaining < 0:
            raise _BudgetExhausted

    def gold_prefix(i: int) -> Counter[Row]:
        while len(gold_prefixes) <= i:
            spend()
            depth = len(gold_prefixes) + 1
            gold_prefixes.append(Counter(row[:depth] for row in gold))
        return gold_prefixes[i]

    used = [False] * width
    # partial[i] holds each pred row projected onto the pred columns assigned to gold 0..i-1.
    partial: list[list[Row]] = [[() for _ in pred]]

    def search(i: int) -> bool:
        if i == width:
            return True
        tried: set[int] = set()
        for j in range(width):
            if used[j] or pred_keys[j] != gold_keys[i] or pred_vectors[j] in tried:
                continue
            tried.add(pred_vectors[j])
            if not ordered:
                spend()
                extended = [p + (v,) for p, v in zip(partial[i], pred_cols[j])]
                if Counter(extended) != gold_prefix(i):
                    continue
                partial.append(extended)
            used[j] = True
            if search(i + 1):
                return True
            used[j] = False
            if not ordered:
                partial.pop()
        return False

    try:
        return MatchOutcome.MATCH if search(0) else MatchOutcome.NO_MATCH
    except _BudgetExhausted:
        return MatchOutcome.SEARCH_LIMIT


def compare_results(
    pred_rows: Sequence[Row],
    gold_rows: Sequence[Row],
    *,
    ordered: bool,
    float_digits: int,
    permute_columns: bool,
    search_budget: int,
) -> MatchOutcome:
    """Compare predicted rows with gold rows.

    Args:
        pred_rows: Rows returned by the predicted query.
        gold_rows: Rows returned by the gold query.
        ordered: Compare rows as ordered lists (gold has a top-level ORDER BY) instead of
            multisets.
        float_digits: Decimal places floats are rounded to before comparison.
        permute_columns: Accept any permutation of the predicted columns (the default metric). If
            False, column order must match exactly (the strict metric). Strict comparison never
            searches, so it never returns ``SEARCH_LIMIT``.
        search_budget: Work budget for the permutation search, in row operations (see
            :func:`_search_permutation`).
    """
    pred = normalize_rows(pred_rows, float_digits)
    gold = normalize_rows(gold_rows, float_digits)
    if len(pred) != len(gold):
        return MatchOutcome.NO_MATCH
    if not gold:
        return MatchOutcome.MATCH
    if len(pred[0]) != len(gold[0]):
        return MatchOutcome.NO_MATCH
    if _rows_equal(pred, gold, ordered):
        return MatchOutcome.MATCH
    if not permute_columns:
        return MatchOutcome.NO_MATCH
    return _search_permutation(pred, gold, ordered, search_budget)


def gold_is_ordered(gold_sql: str) -> bool:
    """Whether the gold query's result order is part of its meaning."""
    return has_top_level_order_by(gold_sql)
