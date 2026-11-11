"""Safe, read-only execution of SQL against SQLite databases.

Every query runs on a fresh connection that is:

* opened read-only and immutable (``mode=ro&immutable=1``), which also works on read-only
  filesystems such as Kaggle's ``/kaggle/input``;
* set to ``PRAGMA query_only``;
* guarded by an authorizer that allows only reads, SELECT, function calls and recursive CTEs, so
  ATTACH, PRAGMA, writes, schema changes, transactions and VACUUM are rejected at compile time;
* stopped by a progress handler once the wall-clock budget is used up;
* limited in how many rows it may return.

Text that is not valid UTF-8 (a few Spider databases contain some) is decoded with replacement
characters rather than raising.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from joinery.config import ExecutionConfig

Row = tuple[Any, ...]

# Authorizer action codes, see https://www.sqlite.org/c3ref/c_alter_table.html
_SQLITE_READ = 20
_SQLITE_SELECT = 21
_SQLITE_FUNCTION = 31
_SQLITE_RECURSIVE = 33
_ALLOWED_ACTIONS = frozenset({_SQLITE_READ, _SQLITE_SELECT, _SQLITE_FUNCTION, _SQLITE_RECURSIVE})

# The progress handler runs every this many SQLite virtual-machine instructions.
_PROGRESS_INTERVAL = 1_000


class ExecErrorType(str, Enum):
    """Why an execution did not produce a usable result."""

    EMPTY_QUERY = "empty_query"
    SYNTAX = "syntax"
    SCHEMA = "schema"
    NOT_ALLOWED = "not_allowed"
    MULTIPLE_STATEMENTS = "multiple_statements"
    TIMEOUT = "timeout"
    TOO_MANY_ROWS = "too_many_rows"
    OTHER = "other"


@dataclass(frozen=True)
class ExecResult:
    """Outcome of one execution. ``rows`` and ``columns`` are None exactly when it failed."""

    rows: tuple[Row, ...] | None
    columns: tuple[str, ...] | None
    error_type: ExecErrorType | None
    error_message: str | None
    elapsed_s: float

    @property
    def ok(self) -> bool:
        """True if the query ran to completion within the limits."""
        return self.error_type is None


_ERROR_PATTERNS: tuple[tuple[ExecErrorType, tuple[str, ...]], ...] = (
    (ExecErrorType.MULTIPLE_STATEMENTS, ("one statement at a time",)),
    (ExecErrorType.NOT_ALLOWED, ("not authorized", "authorization denied", "readonly", "attempt to write")),
    (ExecErrorType.SCHEMA, ("no such table", "no such column", "ambiguous column name")),
    (ExecErrorType.SYNTAX, ("syntax error", "incomplete input", "unrecognized token")),
)


def classify_error(message: str) -> ExecErrorType:
    """Map a SQLite error message to an :class:`ExecErrorType`."""
    lowered = message.lower()
    for error_type, needles in _ERROR_PATTERNS:
        if any(needle in lowered for needle in needles):
            return error_type
    return ExecErrorType.OTHER


def _authorize(action: int, *_: Any) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _decode_text(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


@contextmanager
def open_readonly(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open ``db_path`` read-only with the authorizer installed.

    Raises:
        FileNotFoundError: If the database file does not exist. That is a setup problem, not a
            query failure, so it is raised instead of reported as an execution error.
    """
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        conn.text_factory = _decode_text
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_authorize)
        yield conn
    finally:
        conn.close()


def _failure(error_type: ExecErrorType, message: str, start: float) -> ExecResult:
    return ExecResult(None, None, error_type, message, time.perf_counter() - start)


def execute_sql(sql: str, db_path: str | Path, config: ExecutionConfig) -> ExecResult:
    """Execute one SQL query read-only and capture its rows or its error.

    Args:
        sql: A single SQL statement. A trailing semicolon is allowed.
        db_path: Path to the SQLite database file.
        config: Timeout and row limit.

    Returns:
        An :class:`ExecResult`. SQL errors, timeouts and limit violations are reported in the
        result and never raised.
    """
    start = time.perf_counter()
    if not sql.strip():
        return _failure(ExecErrorType.EMPTY_QUERY, "empty SQL", start)
    deadline = start + config.timeout_s
    timed_out = False

    def check_deadline() -> int:
        nonlocal timed_out
        if time.perf_counter() > deadline:
            timed_out = True
            return 1
        return 0

    with open_readonly(db_path) as conn:
        conn.set_progress_handler(check_deadline, _PROGRESS_INTERVAL)
        try:
            cursor = conn.execute(sql)
            if cursor.description is None:
                return _failure(ExecErrorType.NOT_ALLOWED, "statement returns no rows", start)
            columns = tuple(d[0] for d in cursor.description)
            rows = cursor.fetchmany(config.max_rows + 1)
        except (sqlite3.Error, sqlite3.Warning, ValueError) as exc:
            error_type = ExecErrorType.TIMEOUT if timed_out else classify_error(str(exc))
            return _failure(error_type, str(exc), start)
    if len(rows) > config.max_rows:
        return _failure(
            ExecErrorType.TOO_MANY_ROWS, f"more than {config.max_rows} rows returned", start
        )
    return ExecResult(tuple(rows), columns, None, None, time.perf_counter() - start)
