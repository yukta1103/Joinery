"""Extraction of the final SQL query from a model completion.

The prompt asks for brief reasoning followed by the query in a ```sql code block. Extraction rules,
in priority order:

1. If the completion ends inside an opened ```sql block that never closes (usually because
   generation hit the token limit), take everything after the opening fence. This counts as the
   last answer because it comes after every closed block.
2. Otherwise take the **last** closed block tagged ``sql`` or ``sqlite`` (any case).
3. Otherwise take the last closed block with no language tag.
4. Otherwise there is no SQL. Bare SQL in the reasoning is deliberately not scraped, so the
   output format is enforced.

The extracted text has comments removed and trailing semicolons stripped. A block holding more
than one statement keeps its text but is flagged, and execution then rejects it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from joinery.sqltext import count_statements, strip_comments

_SQL_TAGS = frozenset({"sql", "sqlite"})
_CLOSED_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_OPEN_SQL_FENCE = re.compile(r"```[ \t]*(?:sql|sqlite)[ \t]*\n", re.IGNORECASE)
_TRAILING_SEMICOLONS = re.compile(r"[;\s]+$")


class ExtractionStatus(str, Enum):
    """How (or why not) SQL was extracted."""

    OK = "ok"
    UNTAGGED_FENCE = "untagged_fence"
    UNCLOSED_FENCE = "unclosed_fence"
    MULTIPLE_STATEMENTS = "multiple_statements"
    EMPTY_BLOCK = "empty_block"
    NO_BLOCK = "no_block"


@dataclass(frozen=True)
class Extraction:
    """Extracted SQL, or None when no usable block was found."""

    sql: str | None
    status: ExtractionStatus


def clean_sql(raw: str) -> str:
    """Strip comments, surrounding whitespace and trailing semicolons."""
    return _TRAILING_SEMICOLONS.sub("", strip_comments(raw)).strip()


def _from_block(body: str, status: ExtractionStatus) -> Extraction:
    sql = clean_sql(body)
    if not sql:
        return Extraction(None, ExtractionStatus.EMPTY_BLOCK)
    if count_statements(sql) > 1:
        return Extraction(sql, ExtractionStatus.MULTIPLE_STATEMENTS)
    return Extraction(sql, status)


def extract_sql(completion: str) -> Extraction:
    """Extract the final SQL query from ``completion`` following the module rules."""
    text = completion.replace("\r\n", "\n").replace("\r", "\n")
    closed = list(_CLOSED_FENCE.finditer(text))
    tail_start = closed[-1].end() if closed else 0
    unclosed = list(_OPEN_SQL_FENCE.finditer(text, tail_start))
    if unclosed:
        return _from_block(text[unclosed[-1].end():], ExtractionStatus.UNCLOSED_FENCE)

    tagged = [m for m in closed if m.group(1).strip().lower() in _SQL_TAGS]
    if tagged:
        return _from_block(tagged[-1].group(2), ExtractionStatus.OK)
    untagged = [m for m in closed if not m.group(1).strip()]
    if untagged:
        return _from_block(untagged[-1].group(2), ExtractionStatus.UNTAGGED_FENCE)
    return Extraction(None, ExtractionStatus.NO_BLOCK)
