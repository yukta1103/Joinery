"""Difficulty labels on hand-built Spider SQL dicts, worked out by hand from the official rules."""

from __future__ import annotations

import copy
from typing import Any

from conftest import easy_sql_dict

from joinery.spider_hardness import eval_hardness

# Spider encodings: AGG_OPS index 3 = count, 5 = avg; WHERE_OPS index 3 = '>', 9 = 'like'.
COUNT, AVG = 3, 5
GT, LIKE = 3, 9


def col(col_id: int, agg: int = 0) -> list[Any]:
    return [agg, col_id, False]


def val(col_id: int) -> list[Any]:
    return [0, col(col_id), None]


def cond(col_id: int, op: int, value: Any) -> list[Any]:
    return [False, op, val(col_id), value, None]


def base(select_cols: int = 1) -> dict[str, Any]:
    sql = easy_sql_dict()
    sql["select"] = [False, [[0, val(i)] for i in range(select_cols)]]
    return sql


def test_easy_count() -> None:
    # SELECT count(*) FROM singer
    assert eval_hardness(easy_sql_dict()) == "easy"


def test_easy_single_where() -> None:
    # SELECT name FROM singer WHERE age > 20
    sql = base()
    sql["where"] = [cond(1, GT, 20.0)]
    assert eval_hardness(sql) == "easy"


def test_medium_two_columns_and_where() -> None:
    # SELECT name, country FROM singer WHERE age > 20
    sql = base(select_cols=2)
    sql["where"] = [cond(1, GT, 20.0)]
    assert eval_hardness(sql) == "medium"


def test_medium_join() -> None:
    # SELECT T1.name FROM singer T1 JOIN concert T2 ON ... WHERE T2.year > 2014
    sql = base()
    sql["from"] = {"table_units": [["table_unit", 0], ["table_unit", 1]], "conds": [cond(0, 2, col(5))]}
    sql["where"] = [cond(6, GT, 2014.0)]
    assert eval_hardness(sql) == "medium"


def test_hard_nested_subquery() -> None:
    # SELECT name FROM singer WHERE age > (SELECT avg(age) FROM singer)
    inner = base()
    inner["select"] = [False, [[AVG, val(1)]]]
    sql = base()
    sql["where"] = [cond(1, GT, inner)]
    assert eval_hardness(sql) == "hard"


def test_hard_many_components() -> None:
    # WHERE + GROUP BY + ORDER BY -> component1 = 3 -> hard.
    # Making the WHERE a LIKE and adding LIMIT -> component1 = 5 -> extra.
    sql = base()
    sql["where"] = [cond(1, GT, 20.0)]
    sql["groupBy"] = [col(2)]
    sql["orderBy"] = ["desc", [val(1)]]
    assert eval_hardness(sql) == "hard"
    sql2 = copy.deepcopy(sql)
    sql2["where"] = [cond(1, LIKE, "'%a%'")]
    sql2["limit"] = 1
    assert eval_hardness(sql2) == "extra"


def test_extra_union_plus_nested() -> None:
    inner = base()
    sql = base()
    sql["where"] = [cond(1, GT, inner)]
    sql["union"] = base()
    assert eval_hardness(sql) == "extra"
