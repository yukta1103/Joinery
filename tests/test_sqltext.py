from __future__ import annotations

import pytest

from joinery.sqltext import (
    SegmentKind,
    count_statements,
    has_top_level_order_by,
    mask_quoted,
    segments,
    strip_comments,
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 'a;b' FROM t -- trailing\n",
        "SELECT \"x\"\"y\", `a``b`, [col name] /* c */ FROM t",
        "SELECT 'unterminated",
        "/* unterminated block",
        "",
    ],
)
def test_segments_round_trip(sql: str) -> None:
    assert "".join(text for _, text in segments(sql)) == sql


def test_segments_kinds() -> None:
    kinds = [kind for kind, _ in segments("SELECT 'x' -- c\nFROM t")]
    assert kinds == [
        SegmentKind.CODE,
        SegmentKind.QUOTED,
        SegmentKind.CODE,
        SegmentKind.COMMENT,
        SegmentKind.CODE,
    ]


def test_doubled_quote_stays_inside_string() -> None:
    assert mask_quoted("SELECT 'it''s; fine' FROM t") == "SELECT _ FROM t"


def test_strip_comments_keeps_dashes_inside_strings() -> None:
    assert strip_comments("SELECT '--not a comment' -- real") == "SELECT '--not a comment'  "


def test_strip_block_comment() -> None:
    assert strip_comments("SELECT /* hi */ 1") == "SELECT   1"


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT 1", 1),
        ("SELECT 1;", 1),
        ("SELECT 1; -- done", 1),
        ("SELECT 1; SELECT 2", 2),
        ("SELECT ';' FROM t", 1),
        (";;", 0),
        ("", 0),
        ("-- only a comment", 0),
    ],
)
def test_count_statements(sql: str, expected: int) -> None:
    assert count_statements(sql) == expected


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT a FROM t ORDER BY a", True),
        ("select a from t order\n  by a desc limit 3", True),
        ("SELECT a FROM (SELECT a FROM t ORDER BY a)", False),
        ("SELECT a FROM t WHERE a IN (SELECT a FROM u ORDER BY a LIMIT 1)", False),
        ("SELECT row_number() OVER (ORDER BY a) FROM t", False),
        ("SELECT group_concat(a ORDER BY b) FROM t", False),
        ("SELECT a FROM t UNION SELECT b FROM u ORDER BY 1", True),
        ("SELECT 'order by' FROM t", False),
        ("SELECT a FROM t -- order by a", False),
        ('SELECT "order by" FROM t', False),
        ("SELECT a FROM t GROUP BY a", False),
        ("SELECT border_by FROM t", False),
    ],
)
def test_has_top_level_order_by(sql: str, expected: bool) -> None:
    assert has_top_level_order_by(sql) is expected
