from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from conftest import MINI_DEV

from joinery.config import DataConfig, ExecutionConfig
from joinery.data import audit_gold, database_path, load_examples, require_databases

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_audit_counts_gold_over_max_rows(spider_dir: Path) -> None:
    # MINI_DEV golds return 1, 4, 3, 2, 1, (broken), 4 rows. With max_rows=3 two golds overflow.
    examples = load_examples(DataConfig(spider_dir=str(spider_dir)))
    report = audit_gold(examples, spider_dir / "database", ExecutionConfig(timeout_s=2.0, max_rows=3))
    assert report["n_gold_too_many_rows"] == 2
    assert report["max_rows_limit"] == 3
    assert report["max_gold_rows"] == 3
    over = {f["example_id"] for f in report["gold_failures"] if f["error_type"] == "too_many_rows"}
    assert over == {"dev_00001", "dev_00006"}


def test_audit_reports_largest_gold_under_default_limit(spider_dir: Path) -> None:
    examples = load_examples(DataConfig(spider_dir=str(spider_dir)))
    report = audit_gold(examples, spider_dir / "database", ExecutionConfig())
    assert report["n_gold_too_many_rows"] == 0
    assert report["max_gold_rows"] == 4


def _run_check_data(spider_dir: Path, tmp_path: Path, max_rows: int) -> subprocess.CompletedProcess:
    config = REPO_ROOT / "configs" / "eval_qwen15b_instruct.yaml"
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "check_data.py"),
            "--config", str(config),
            "--set", f"data.spider_dir={spider_dir}",
            "--set", f"output_dir={tmp_path / 'results'}",
            "--set", f"execution.max_rows={max_rows}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_check_data_fails_when_gold_exceeds_max_rows(spider_dir: Path, tmp_path: Path) -> None:
    result = _run_check_data(spider_dir, tmp_path, max_rows=3)
    assert result.returncode != 0
    assert "Gold queries returning more than execution.max_rows=3 rows: 2" in result.stderr


def test_check_data_passes_when_gold_fits(spider_dir: Path, tmp_path: Path) -> None:
    result = _run_check_data(spider_dir, tmp_path, max_rows=100_000)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "results" / "data_check_dev.json").is_file()
