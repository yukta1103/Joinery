"""Comparison of query results for execution accuracy.

Rules (see the README's "Comparison rules" section for the reasoning):

* Rows are compared as a **multiset** (duplicates count) unless the gold query has a top-level
  ORDER BY, in which case they are compared as an ordered list.
* Columns: by default a prediction matches if **some permutation of its columns** matches the gold
  result, as in test-suite-sql-eval. The row rule above is applied after permuting. A strict mode
  (``permute_columns=False``) requires the same column order. Column counts must be equal, and
  column names are always ignored.
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


def _column_key(column: tuple[Any, ...], ordered: bool) -> Hashable:
    """A value every matching column must share: the column itself if ordered, else its multiset."""
    return column if ordered else frozenset(Counter(column).items())


def _some_permutation_matches(pred: list[Row], gold: list[Row], ordered: bool) -> bool:
    """Search for a mapping of gold columns to distinct pred columns under which the rows match.

    The search assigns gold columns left to right, with three prunings:

    * a pred column is a candidate only if it holds the same values as the gold column (same
      sequence when ordered, same multiset otherwise);
    * pred columns that are identical vectors are interchangeable, so only one of them is tried at
      each step;
    * when unordered, the multiset of partial rows over the assigned columns must already match.
      When ordered, equal column vectors already imply equal rows.
    """
    width = len(gold[0])
    pred_cols = [tuple(row[j] for row in pred) for j in range(width)]
    gold_cols = [tuple(row[i] for row in gold) for i in range(width)]
    pred_keys = [_column_key(col, ordered) for col in pred_cols]
    gold_keys = [_column_key(col, ordered) for col in gold_cols]
    gold_prefixes = [Counter(row[: i + 1] for row in gold) for i in range(width)]
    assignment: list[int] = []
    used = [False] * width

    def search(i: int) -> bool:
        if i == width:
            return True
        tried: set[tuple[Any, ...]] = set()
        for j in range(width):
            if used[j] or pred_keys[j] != gold_keys[i] or pred_cols[j] in tried:
                continue
            tried.add(pred_cols[j])
            assignment.append(j)
            if ordered or Counter(tuple(row[k] for k in assignment) for row in pred) == gold_prefixes[i]:
                used[j] = True
                if search(i + 1):
                    return True
                used[j] = False
            assignment.pop()
        return False

    return search(0)


def results_match(
    pred_rows: Sequence[Row],
    gold_rows: Sequence[Row],
    *,
    ordered: bool,
    float_digits: int,
    permute_columns: bool,
) -> bool:
    """Return True if the predicted rows match the gold rows.

    Args:
        pred_rows: Rows returned by the predicted query.
        gold_rows: Rows returned by the gold query.
        ordered: Compare rows as ordered lists (gold has a top-level ORDER BY) instead of
            multisets.
        float_digits: Decimal places floats are rounded to before comparison.
        permute_columns: Accept any permutation of the predicted columns (the default metric). If
            False, column order must match exactly (the strict metric).
    """
    pred = normalize_rows(pred_rows, float_digits)
    gold = normalize_rows(gold_rows, float_digits)
    if len(pred) != len(gold):
        return False
    if not gold:
        return True
    if len(pred[0]) != len(gold[0]):
        return False
    if _rows_equal(pred, gold, ordered):
        return True
    return permute_columns and _some_permutation_matches(pred, gold, ordered)


def gold_is_ordered(gold_sql: str) -> bool:
    """Whether the gold query's result order is part of its meaning."""
    return has_top_level_order_by(gold_sql)
