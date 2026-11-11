"""Comparison of query results for execution accuracy.

Rules (see the README's "Comparison rules" section for the reasoning):

* Rows are compared as a **multiset** (duplicates count) unless the gold query has a top-level
  ORDER BY, in which case they are compared as an ordered list.
* Column **order** matters and column **names** do not. Rows are positional tuples.
* NULL equals NULL.
* Integers and floats compare by value (``1 == 1.0``). Floats are rounded to a fixed number of
  decimal places first, so ``0.1 + 0.2`` matches ``0.3``.
* Strings are compared exactly (case and whitespace sensitive). A text ``'1'`` does not equal the
  integer ``1``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
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


def results_match(
    pred_rows: Sequence[Row], gold_rows: Sequence[Row], *, ordered: bool, float_digits: int
) -> bool:
    """Return True if the predicted rows match the gold rows.

    Args:
        pred_rows: Rows returned by the predicted query.
        gold_rows: Rows returned by the gold query.
        ordered: Compare as ordered lists (gold has a top-level ORDER BY) instead of multisets.
        float_digits: Decimal places floats are rounded to before comparison.
    """
    pred = normalize_rows(pred_rows, float_digits)
    gold = normalize_rows(gold_rows, float_digits)
    if ordered:
        return pred == gold
    return Counter(pred) == Counter(gold)


def gold_is_ordered(gold_sql: str) -> bool:
    """Whether the gold query's result order is part of its meaning."""
    return has_top_level_order_by(gold_sql)
