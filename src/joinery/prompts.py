"""Prompt construction: database schema plus question, with a fixed output format.

The schema is the database's own CREATE TABLE statements (read from ``sqlite_master``). Each table
can be followed by a few sample rows in a SQL comment. The system prompt asks for brief reasoning
and then the final query in a ```sql block, which :mod:`joinery.extraction` relies on.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from joinery.config import PromptConfig
from joinery.execution import open_readonly

SYSTEM_PROMPT = (
    "You are an expert SQLite developer. Given a database schema and a question, write one "
    "SQLite query that answers the question.\n"
    "Respond in exactly this format:\n"
    "1. Brief reasoning: at most a few sentences about which tables, joins and conditions are "
    "needed.\n"
    "2. The final query in a single ```sql code block containing exactly one SQL statement."
)

Message = dict[str, str]


def quote_identifier(name: str) -> str:
    """Quote a table or column name for SQLite."""
    return '"' + name.replace('"', '""') + '"'


def format_value(value: Any, max_chars: int) -> str:
    """Render one sample value on a single line, truncated to ``max_chars`` characters."""
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return f"<blob {len(value)} bytes>"
    text = " ".join(str(value).split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _clean_ddl(ddl: str) -> str:
    lines = [line.rstrip() for line in ddl.strip().splitlines() if line.strip()]
    text = "\n".join(lines)
    return text if text.endswith(";") else text + ";"


@lru_cache(maxsize=512)
def schema_text(db_path: str, sample_rows: int, max_value_chars: int) -> str:
    """Return the schema of ``db_path`` as CREATE TABLE statements, with optional sample rows.

    Tables come in creation order. Sample rows are the first ``sample_rows`` rows in storage
    order, which is deterministic for a given database file.
    """
    blocks = []
    with open_readonly(Path(db_path)) as conn:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        ).fetchall()
        for name, ddl in tables:
            block = _clean_ddl(ddl)
            if sample_rows > 0:
                cursor = conn.execute(f"SELECT * FROM {quote_identifier(name)} LIMIT ?", (sample_rows,))
                rows = cursor.fetchall()
                if rows:
                    header = " | ".join(d[0] for d in cursor.description)
                    body = "\n".join(
                        " | ".join(format_value(v, max_value_chars) for v in row) for row in rows
                    )
                    block += f"\n/* Sample rows from {name}:\n{header}\n{body}\n*/"
            blocks.append(block)
    return "\n\n".join(blocks)


def build_messages(db_path: str | Path, question: str, config: PromptConfig) -> list[Message]:
    """Build chat messages (system and user) for one question."""
    schema = schema_text(str(db_path), config.sample_rows, config.max_value_chars)
    user = f"Database schema:\n{schema}\n\nQuestion: {question.strip()}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
