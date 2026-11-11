from __future__ import annotations

from pathlib import Path

import pytest
from conftest import MINI_DEV

from joinery.config import DataConfig, ExecutionConfig
from joinery.data import audit_gold, database_path, load_examples, require_databases


def test_load_dev(spider_dir: Path) -> None:
    examples = load_examples(DataConfig(spider_dir=str(spider_dir)))
    assert [ex.question for ex in examples] == [q for q, _ in MINI_DEV]
    assert examples[0].example_id == "dev_00000"
    assert examples[0].gold_sql == MINI_DEV[0][1]
    assert examples[0].difficulty == "easy"


def test_load_train_concatenates_files(spider_dir: Path) -> None:
    examples = load_examples(DataConfig(spider_dir=str(spider_dir), split="train"))
    assert [ex.example_id for ex in examples] == ["train_spider_00000", "train_spider_00001", "train_others_00000"]


def test_limit(spider_dir: Path) -> None:
    assert len(load_examples(DataConfig(spider_dir=str(spider_dir), limit=2))) == 2


def test_missing_spider_dir_gives_setup_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Data setup"):
        load_examples(DataConfig(spider_dir=str(tmp_path / "nope")))


def test_database_dir_defaults_and_override(spider_dir: Path) -> None:
    assert DataConfig(spider_dir=str(spider_dir)).resolved_database_dir() == spider_dir / "database"
    assert DataConfig(spider_dir="x", database_dir="/db").resolved_database_dir() == Path("/db")


def test_require_databases(spider_dir: Path, tmp_path: Path) -> None:
    examples = load_examples(DataConfig(spider_dir=str(spider_dir)))
    require_databases(examples, spider_dir / "database")
    with pytest.raises(FileNotFoundError, match="concert_singer"):
        require_databases(examples, tmp_path)


def test_database_path(spider_dir: Path) -> None:
    assert database_path(spider_dir / "database", "concert_singer").is_file()


def test_audit_gold(spider_dir: Path) -> None:
    examples = load_examples(DataConfig(spider_dir=str(spider_dir)))
    report = audit_gold(examples, spider_dir / "database", ExecutionConfig())
    assert report["n_examples"] == len(MINI_DEV)
    assert report["n_databases"] == 1
    assert report["n_gold_failed"] == 1
    assert report["gold_failures"][0]["error_type"] == "schema"
    assert report["n_gold_ordered"] == 1
    assert report["n_gold_empty"] == 0
    assert report["by_difficulty"] == {"easy": len(MINI_DEV), "medium": 0, "hard": 0, "extra": 0}


@pytest.mark.parametrize("bad", [{"split": "test"}, {"limit": 0}])
def test_data_config_validation(bad: dict) -> None:
    with pytest.raises(ValueError):
        DataConfig(spider_dir="x", **bad)
