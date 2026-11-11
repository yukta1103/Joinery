"""Lexical helpers that split SQL text into code, quoted spans and comments.

These helpers do not parse SQL. They only find quoted literals and comments, so that later steps
can find semicolons, parentheses and keywords without being misled by text inside a string.
SQLite quoting rules apply: '...' strings, "..." and `...` quoted names (a doubled quote escapes
itself), and [...] bracketed names. Comments are ``-- ...`` to end of line and ``/* ... */``.
"""

from __future__ import annotations

import re
from enum import Enum


class SegmentKind(Enum):
    """Kind of a lexical segment."""

    CODE = "code"
    QUOTED = "quoted"
    COMMENT = "comment"


Segment = tuple[SegmentKind, str]

_QUOTE_CLOSERS = {"'": "'", '"': '"', "`": "`", "[": "]"}


def _end_of_quoted(sql: str, start: int, closer: str) -> int:
    """Return the index just past the quoted span opened at ``start`` (or ``len(sql)``)."""
    i = start + 1
    while i < len(sql):
        if sql[i] == closer:
            if closer != "]" and sql.startswith(closer, i + 1):
                i += 2
                continue
            return i + 1
        i += 1
    return len(sql)


def segments(sql: str) -> list[Segment]:
    """Split ``sql`` into consecutive code, quoted and comment segments.

    Concatenating the segment texts reproduces ``sql`` exactly. An unclosed quote or block
    comment extends to the end of the text.
    """
    out: list[Segment] = []
    code_start = 0
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch in _QUOTE_CLOSERS:
            end = _end_of_quoted(sql, i, _QUOTE_CLOSERS[ch])
            kind = SegmentKind.QUOTED
        elif sql.startswith("--", i):
            newline = sql.find("\n", i)
            end = n if newline == -1 else newline
            kind = SegmentKind.COMMENT
        elif sql.startswith("/*", i):
            close = sql.find("*/", i + 2)
            end = n if close == -1 else close + 2
            kind = SegmentKind.COMMENT
        else:
            i += 1
            continue
        if i > code_start:
            out.append((SegmentKind.CODE, sql[code_start:i]))
        out.append((kind, sql[i:end]))
        i = code_start = end
    if code_start < n:
        out.append((SegmentKind.CODE, sql[code_start:]))
    return out


def strip_comments(sql: str) -> str:
    """Replace every comment with a single space, leaving quoted text untouched."""
    return "".join(" " if kind is SegmentKind.COMMENT else text for kind, text in segments(sql))


def mask_quoted(sql: str) -> str:
    """Return the code with comments removed and each quoted span replaced by ``_``."""
    parts = []
    for kind, text in segments(sql):
        if kind is SegmentKind.CODE:
            parts.append(text)
        elif kind is SegmentKind.QUOTED:
            parts.append("_")
        else:
            parts.append(" ")
    return "".join(parts)


def count_statements(sql: str) -> int:
    """Count non-empty statements separated by top-level semicolons.

    Comments and whitespace alone do not form a statement, so ``"SELECT 1; -- done"`` is one
    statement and ``";;"`` is zero.
    """
    return sum(1 for part in mask_quoted(sql).split(";") if part.strip())


_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def has_top_level_order_by(sql: str) -> bool:
    """Return True if ``sql`` has an ORDER BY outside every parenthesis.

    An ORDER BY inside a subquery, a window ``OVER (...)`` clause or an aggregate's argument list
    does not order the final result, so it is ignored. An ORDER BY after a compound query
    (``... UNION SELECT ... ORDER BY x``) is top level and counts.
    """
    depth = 0
    top_level = []
    for ch in mask_quoted(sql):
        if ch == "(":
            depth += 1
            top_level.append(" ")
        elif ch == ")":
            depth = max(depth - 1, 0)
            top_level.append(" ")
        elif depth == 0:
            top_level.append(ch)
    return _ORDER_BY.search("".join(top_level)) is not None
