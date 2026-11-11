"""Shared fixtures: a tiny SQLite database and a miniature Spider release built on the fly."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from joinery.config import ComparisonConfig, ExecutionConfig, RewardConfig

FIXTURE_SCHEMA = """
CREATE TABLE singer (
    singer_id INTEGER PRIMARY KEY,
    name TEXT,
    country TEXT,
    age INTEGER,
    net_worth REAL
);
CREATE TABLE concert (
    concert_id INTEGER PRIMARY KEY,
    singer_id INTEGER REFERENCES singer(singer_id),
    year INTEGER,
    venue TEXT
);
INSERT INTO singer VALUES
    (1, 'Joe', 'USA', 30, 1.5),
    (2, 'Ann', 'UK', 25, NULL),
    (3, 'Bo', 'USA', 30, 2.25),
    (4, 'Cy', NULL, 41, 0.1);
INSERT INTO concert VALUES
    (1, 1, 2014, 'Hall A'),
    (2, 1, 2015, 'Hall B'),
    (3, 3, 2015, 'Hall A'),
    (4, 2, 2016, 'Hall C');
"""


def build_fixture_db(path: Path) -> Path:
    """Create the fixture database at ``path`` (including one row of invalid UTF-8 text)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(FIXTURE_SCHEMA)
    conn.execute("INSERT INTO concert VALUES (5, 4, 2017, CAST(X'C328' AS TEXT))")
    conn.commit()
    conn.close()
    return path


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Path to the fixture database."""
    return build_fixture_db(tmp_path_factory.mktemp("db") / "concert_singer.sqlite")


@pytest.fixture
def exec_cfg() -> ExecutionConfig:
    return ExecutionConfig(timeout_s=2.0, max_rows=1_000)


@pytest.fixture
def cmp_cfg() -> ComparisonConfig:
    return ComparisonConfig(float_digits=6)


@pytest.fixture
def reward_cfg() -> RewardConfig:
    return RewardConfig(parse_reward=0.1, execute_reward=0.3, match_reward=1.0)


def easy_sql_dict() -> dict[str, Any]:
    """A Spider-style parsed SQL dict for ``SELECT count(*) FROM singer`` (difficulty: easy)."""
    return {
        "select": [False, [[3, [0, [0, "__all__", False], None]]]],
        "from": {"table_units": [["table_unit", 0]], "conds": []},
        "where": [],
        "groupBy": [],
        "having": [],
        "orderBy": [],
        "limit": None,
        "intersect": None,
        "union": None,
        "except": None,
    }


# (question, gold SQL) pairs for the mini Spider dev split. The last gold query is broken on
# purpose so the runner's exclusion path is exercised.
MINI_DEV = [
    ("How many singers are there?", "SELECT count(*) FROM singer"),
    ("List singer names ordered by age.", "SELECT name FROM singer ORDER BY age, name"),
    ("Which countries have singers?", "SELECT DISTINCT country FROM singer"),
    ("What venues hosted concerts in 2015?", "SELECT venue FROM concert WHERE year = 2015"),
    ("What is the average net worth?", "SELECT avg(net_worth) FROM singer"),
    ("Broken gold example.", "SELECT * FROM no_such_table"),
]


@pytest.fixture
def spider_dir(tmp_path: Path) -> Path:
    """A miniature Spider release: ``dev.json``, train files and one database."""
    root = tmp_path / "spider_data"
    build_fixture_db(root / "database" / "concert_singer" / "concert_singer.sqlite")
    dev = [
        {"db_id": "concert_singer", "question": q, "query": sql, "sql": easy_sql_dict()}
        for q, sql in MINI_DEV
    ]
    (root / "dev.json").write_text(json.dumps(dev), encoding="utf-8")
    (root / "train_spider.json").write_text(json.dumps(dev[:2]), encoding="utf-8")
    (root / "train_others.json").write_text(json.dumps(dev[2:3]), encoding="utf-8")
    return root
