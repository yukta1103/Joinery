"""Spider difficulty levels (easy / medium / hard / extra).

Adapted from the official Spider evaluation script, ``evaluation.py`` in
https://github.com/taoyds/spider (Apache License 2.0, Copyright 2024 XLANG NLP Lab). The logic is
kept identical, including its quirks, so the labels match the official ones. One quirk: in
``count_others``, WHERE condition units and the raw HAVING list (which includes 'and'/'or' tokens)
go through ``has_agg``. The function takes the parsed SQL dict stored in the ``sql`` field of
Spider's JSON files.

Only names were changed (to snake_case and module-private).
"""

from __future__ import annotations

from typing import Any

SqlDict = dict[str, Any]

DIFFICULTY_LEVELS = ("easy", "medium", "hard", "extra")

_WHERE_OPS = ("not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists")
_AGG_OPS = ("none", "max", "min", "count", "sum", "avg")


def _has_agg(unit: Any) -> bool:
    return unit[0] != _AGG_OPS.index("none")


def _count_agg(units: list[Any]) -> int:
    return len([unit for unit in units if _has_agg(unit)])


def _get_nested_sql(sql: SqlDict) -> list[SqlDict]:
    nested = []
    for cond_unit in sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql["intersect"] is not None:
        nested.append(sql["intersect"])
    if sql["except"] is not None:
        nested.append(sql["except"])
    if sql["union"] is not None:
        nested.append(sql["union"])
    return nested


def _count_component1(sql: SqlDict) -> int:
    count = 0
    if len(sql["where"]) > 0:
        count += 1
    if len(sql["groupBy"]) > 0:
        count += 1
    if len(sql["orderBy"]) > 0:
        count += 1
    if sql["limit"] is not None:
        count += 1
    if len(sql["from"]["table_units"]) > 0:  # JOIN
        count += len(sql["from"]["table_units"]) - 1

    ao = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    count += len([token for token in ao if token == "or"])
    cond_units = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == _WHERE_OPS.index("like")])
    return count


def _count_component2(sql: SqlDict) -> int:
    return len(_get_nested_sql(sql))


def _count_others(sql: SqlDict) -> int:
    count = 0
    # number of aggregation
    agg_count = _count_agg(sql["select"][1])
    agg_count += _count_agg(sql["where"][::2])
    agg_count += _count_agg(sql["groupBy"])
    if len(sql["orderBy"]) > 0:
        agg_count += _count_agg(
            [unit[1] for unit in sql["orderBy"][1] if unit[1]]
            + [unit[2] for unit in sql["orderBy"][1] if unit[2]]
        )
    agg_count += _count_agg(sql["having"])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql["select"][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql["where"]) > 1:
        count += 1

    # number of group by clauses
    if len(sql["groupBy"]) > 1:
        count += 1
    return count


def eval_hardness(sql: SqlDict) -> str:
    """Return the Spider difficulty level of a parsed SQL dict."""
    count_comp1 = _count_component1(sql)
    count_comp2 = _count_component2(sql)
    count_others = _count_others(sql)

    if count_comp1 <= 1 and count_others == 0 and count_comp2 == 0:
        return "easy"
    elif (count_others <= 2 and count_comp1 <= 1 and count_comp2 == 0) or (
        count_comp1 <= 2 and count_others < 2 and count_comp2 == 0
    ):
        return "medium"
    elif (
        (count_others > 2 and count_comp1 <= 2 and count_comp2 == 0)
        or (2 < count_comp1 <= 3 and count_others <= 2 and count_comp2 == 0)
        or (count_comp1 <= 1 and count_others == 0 and count_comp2 <= 1)
    ):
        return "hard"
    else:
        return "extra"
