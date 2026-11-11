"""Loading Spider examples and locating their SQLite databases.

Spider has to be downloaded by hand (see the README's "Data setup" section). The extracted folder
must contain ``dev.json``, ``train_spider.json``, ``train_others.json`` and
``database/<db_id>/<db_id>.sqlite``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from joinery.comparison import gold_is_ordered
from joinery.config import DataConfig, ExecutionConfig
from joinery.execution import execute_sql
from joinery.spider_hardness import DIFFICULTY_LEVELS, eval_hardness

SPLIT_FILES: dict[str, tuple[str, ...]] = {
    "dev": ("dev.json",),
    "train": ("train_spider.json", "train_others.json"),
}

_SETUP_HINT = (
    "Download Spider as described in the README ('Data setup') and point data.spider_dir "
    "(and data.database_dir, if the databases live elsewhere) at the extracted folder."
)


@dataclass(frozen=True)
class SpiderExample:
    """One Spider question with its gold query.

    ``example_id`` is ``<source file stem>_<index>`` (e.g. ``dev_00042``), which stays stable as
    long as the Spider release doesn't change.
    """

    example_id: str
    db_id: str
    question: str
    gold_sql: str
    difficulty: str


def _read_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Spider file not found: {path}. {_SETUP_HINT}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_examples(config: DataConfig) -> list[SpiderExample]:
    """Load the configured split, with difficulty computed from each example's parsed SQL."""
    spider_dir = Path(config.spider_dir)
    examples = []
    for filename in SPLIT_FILES[config.split]:
        stem = Path(filename).stem
        for index, raw in enumerate(_read_json(spider_dir / filename)):
            examples.append(
                SpiderExample(
                    example_id=f"{stem}_{index:05d}",
                    db_id=raw["db_id"],
                    question=raw["question"],
                    gold_sql=raw["query"],
                    difficulty=eval_hardness(raw["sql"]),
                )
            )
    return examples[: config.limit] if config.limit is not None else examples


def database_path(database_dir: Path, db_id: str) -> Path:
    """Path of the SQLite file for ``db_id``."""
    return database_dir / db_id / f"{db_id}.sqlite"


def require_databases(examples: Iterable[SpiderExample], database_dir: Path) -> None:
    """Check that every example's database file exists.

    Raises:
        FileNotFoundError: Listing every missing database id.
    """
    missing = sorted({ex.db_id for ex in examples if not database_path(database_dir, ex.db_id).is_file()})
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise FileNotFoundError(
            f"{len(missing)} database(s) missing under {database_dir}: {shown}. {_SETUP_HINT}"
        )


def audit_gold(
    examples: Iterable[SpiderExample], database_dir: Path, config: ExecutionConfig
) -> dict[str, Any]:
    """Execute every gold query and summarize the results.

    Returns a JSON-serializable report: counts by difficulty, gold queries that fail (and why),
    how many return no rows and how many are order-sensitive.
    """
    examples = list(examples)
    failures = []
    empty = 0
    ordered = 0
    for ex in examples:
        result = execute_sql(ex.gold_sql, database_path(database_dir, ex.db_id), config)
        if not result.ok:
            failures.append(
                {
                    "example_id": ex.example_id,
                    "db_id": ex.db_id,
                    "error_type": result.error_type.value if result.error_type else None,
                    "error_message": result.error_message,
                }
            )
            continue
        empty += not result.rows
        ordered += gold_is_ordered(ex.gold_sql)
    by_difficulty = Counter(ex.difficulty for ex in examples)
    return {
        "n_examples": len(examples),
        "n_databases": len({ex.db_id for ex in examples}),
        "by_difficulty": {level: by_difficulty.get(level, 0) for level in DIFFICULTY_LEVELS},
        "n_gold_failed": len(failures),
        "n_gold_empty": empty,
        "n_gold_ordered": ordered,
        "gold_failures": failures,
    }
