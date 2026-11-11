from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from joinery.config import ExecutionConfig
from joinery.execution import ExecErrorType, classify_error, execute_sql

INFINITE_QUERY = (
    "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_simple_select(db_path: Path, exec_cfg: ExecutionConfig) -> None:
    result = execute_sql("SELECT singer_id, name FROM singer ORDER BY singer_id", db_path, exec_cfg)
    assert result.ok
    assert result.columns == ("singer_id", "name")
    assert result.rows == ((1, "Joe"), (2, "Ann"), (3, "Bo"), (4, "Cy"))
    assert result.error_message is None


def test_trailing_semicolon_allowed(db_path: Path, exec_cfg: ExecutionConfig) -> None:
    assert execute_sql("SELECT count(*) FROM singer;", db_path, exec_cfg).rows == ((4,),)


def test_null_and_real_values(db_path: Path, exec_cfg: ExecutionConfig) -> None:
    result = execute_sql("SELECT net_worth FROM singer ORDER BY singer_id", db_path, exec_cfg)
    assert result.rows == ((1.5,), (None,), (2.25,), (0.1,))


def test_empty_result_is_ok(db_path: Path, exec_cfg: ExecutionConfig) -> None:
    result = execute_sql("SELECT name FROM singer WHERE age > 100", db_path, exec_cfg)
    assert result.ok
    assert result.rows == ()
    assert result.columns == ("name",)


def test_invalid_utf8_text_is_decoded(db_path: Path, exec_cfg: ExecutionConfig) -> None:
    result = execute_sql("SELECT venue FROM concert WHERE concert_id = 5", db_path, exec_cfg)
    assert result.ok
    assert result.rows == (("�(",),)


def test_recursive_cte_allowed(db_path: Path, exec_cfg: ExecutionConfig) -> None:
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 5) SELECT sum(x) FROM c"
    assert execute_sql(sql, db_path, exec_cfg).rows == ((15,),)


@pytest.mark.parametrize("sql", ["", "   ", "\n\t"])
def test_empty_query(db_path: Path, exec_cfg: ExecutionConfig, sql: str) -> None:
    assert execute_sql(sql, db_path, exec_cfg).error_type is ExecErrorType.EMPTY_QUERY


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELEC name FROM singer", ExecErrorType.SYNTAX),
        ("SELECT name FROM singer WHERE", ExecErrorType.SYNTAX),
        ("SELECT name FROM singer WHERE name = 'unterminated", ExecErrorType.SYNTAX),
        ("SELECT name FROM singers", ExecErrorType.SCHEMA),
        ("SELECT nme FROM singer", ExecErrorType.SCHEMA),
        (
            "SELECT singer_id FROM singer JOIN concert ON singer.singer_id = concert.singer_id",
            ExecErrorType.SCHEMA,
        ),
        ("SELECT 1; SELECT 2", ExecErrorType.MULTIPLE_STATEMENTS),
        ("SELECT name FROM singer GROUP BY count(*)", ExecErrorType.OTHER),
        ("SELECT no_such_function(name) FROM singer", ExecErrorType.OTHER),
    ],
)
def test_error_types(db_path: Path, exec_cfg: ExecutionConfig, sql: str, expected: ExecErrorType) -> None:
    result = execute_sql(sql, db_path, exec_cfg)
    assert result.error_type is expected
    assert result.rows is None and result.columns is None
    assert result.error_message


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM singer",
        "UPDATE singer SET age = 0",
        "INSERT INTO singer VALUES (9, 'X', 'Y', 1, 1.0)",
        "DROP TABLE singer",
        "CREATE TABLE t (x INTEGER)",
        "ALTER TABLE singer ADD COLUMN y INTEGER",
        "PRAGMA query_only = OFF",
        "PRAGMA table_info(singer)",
        "BEGIN",
        "VACUUM",
        "REPLACE INTO singer VALUES (1, 'Z', 'Z', 1, 1.0)",
        "WITH x AS (SELECT 1) DELETE FROM singer",
    ],
)
def test_writes_and_admin_statements_rejected(
    db_path: Path, exec_cfg: ExecutionConfig, sql: str
) -> None:
    before = _digest(db_path)
    result = execute_sql(sql, db_path, exec_cfg)
    assert result.error_type is ExecErrorType.NOT_ALLOWED
    assert _digest(db_path) == before


def test_attach_rejected_and_creates_no_file(
    db_path: Path, exec_cfg: ExecutionConfig, tmp_path: Path
) -> None:
    target = tmp_path / "attached.sqlite"
    result = execute_sql(f"ATTACH DATABASE '{target}' AS other", db_path, exec_cfg)
    assert result.error_type is ExecErrorType.NOT_ALLOWED
    assert not target.exists()


def test_vacuum_into_rejected(db_path: Path, exec_cfg: ExecutionConfig, tmp_path: Path) -> None:
    target = tmp_path / "copy.sqlite"
    result = execute_sql(f"VACUUM INTO '{target}'", db_path, exec_cfg)
    assert result.error_type is ExecErrorType.NOT_ALLOWED
    assert not target.exists()


def test_timeout_stops_runaway_query(db_path: Path) -> None:
    cfg = ExecutionConfig(timeout_s=0.2, max_rows=1_000)
    result = execute_sql(INFINITE_QUERY, db_path, cfg)
    assert result.error_type is ExecErrorType.TIMEOUT
    assert result.elapsed_s < 5.0


def test_timeout_during_cross_join(db_path: Path) -> None:
    cfg = ExecutionConfig(timeout_s=0.2, max_rows=1_000)
    sql = (
        "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 100000) "
        "SELECT count(*) FROM n a, n b"
    )
    assert execute_sql(sql, db_path, cfg).error_type is ExecErrorType.TIMEOUT


def test_row_limit(db_path: Path) -> None:
    cfg = ExecutionConfig(timeout_s=2.0, max_rows=10)
    sql = "SELECT * FROM singer a, singer b"
    result = execute_sql(sql, db_path, cfg)
    assert result.error_type is ExecErrorType.TOO_MANY_ROWS
    exact = execute_sql("SELECT * FROM singer a, concert b LIMIT 10", db_path, cfg)
    assert exact.ok and len(exact.rows or ()) == 10


def test_missing_database_raises(tmp_path: Path, exec_cfg: ExecutionConfig) -> None:
    with pytest.raises(FileNotFoundError):
        execute_sql("SELECT 1", tmp_path / "missing.sqlite", exec_cfg)


def test_connection_state_does_not_leak(db_path: Path, exec_cfg: ExecutionConfig) -> None:
    execute_sql(INFINITE_QUERY, db_path, ExecutionConfig(timeout_s=0.1, max_rows=10))
    assert execute_sql("SELECT count(*) FROM singer", db_path, exec_cfg).rows == ((4,),)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("near \"SELEC\": syntax error", ExecErrorType.SYNTAX),
        ("no such table: x", ExecErrorType.SCHEMA),
        ("ambiguous column name: id", ExecErrorType.SCHEMA),
        ("not authorized", ExecErrorType.NOT_ALLOWED),
        ("misuse of aggregate: count()", ExecErrorType.OTHER),
    ],
)
def test_classify_error(message: str, expected: ExecErrorType) -> None:
    assert classify_error(message) is expected
