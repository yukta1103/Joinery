"""Offline tests of scripts/download_spider.py using locally built zips (no network)."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_spider.py"
_spec = importlib.util.spec_from_file_location("download_spider", SCRIPT)
assert _spec and _spec.loader
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)


def make_zip(path: Path, members: dict[str, bytes]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spider_members(root: str = "spider_data") -> dict[str, bytes]:
    members = {f"{root}/{name}": b"[]" for name in ("dev.json", "train_spider.json", "train_others.json", "tables.json")}
    members[f"{root}/database/db1/db1.sqlite"] = b"sqlite"
    return members


def no_download(*_: object) -> None:
    raise AssertionError("must not download")


def test_extracts_kept_zip_and_deletes_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl, "download", no_download)
    sha = make_zip(tmp_path / dl.ZIP_NAME, spider_members())
    out = dl.run(tmp_path, keep_zip=False, file_id="x", expected_sha256=sha)
    assert (out / "database" / "db1" / "db1.sqlite").read_bytes() == b"sqlite"
    assert (out / dl.MARKER).read_text().strip() == sha
    assert not (tmp_path / dl.ZIP_NAME).exists()
    assert not (tmp_path / "spider_data.extracting").exists()


def test_keep_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl, "download", no_download)
    sha = make_zip(tmp_path / dl.ZIP_NAME, spider_members())
    dl.run(tmp_path, keep_zip=True, file_id="x", expected_sha256=sha)
    assert (tmp_path / dl.ZIP_NAME).is_file()


def test_verified_extract_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = make_zip(tmp_path / dl.ZIP_NAME, spider_members())
    monkeypatch.setattr(dl, "download", no_download)
    dl.run(tmp_path, keep_zip=False, file_id="x", expected_sha256=sha)
    dl.run(tmp_path, keep_zip=False, file_id="x", expected_sha256=sha)  # no zip, no download


def test_hash_mismatch_keeps_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl, "download", no_download)
    sha = make_zip(tmp_path / dl.ZIP_NAME, spider_members())
    with pytest.raises(dl.DownloadError, match=f"actual   {sha}"):
        dl.run(tmp_path, keep_zip=False, file_id="x", expected_sha256="0" * 64)
    assert (tmp_path / dl.ZIP_NAME).is_file()
    assert not (tmp_path / dl.DATASET_DIR).exists()


def test_unverified_existing_folder_is_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / dl.DATASET_DIR).mkdir()
    with pytest.raises(dl.DownloadError, match="not a verified extract"):
        dl.run(tmp_path, keep_zip=False, file_id="x", expected_sha256="0" * 64)


@pytest.mark.parametrize("bad", ["../evil.txt", "/abs.txt", "spider_data/../../evil.txt"])
def test_unsafe_paths_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setattr(dl, "download", no_download)
    sha = make_zip(tmp_path / dl.ZIP_NAME, {**spider_members(), bad: b"x"})
    with pytest.raises(dl.DownloadError, match="unsafe path"):
        dl.run(tmp_path, keep_zip=False, file_id="x", expected_sha256=sha)
    assert not (tmp_path.parent / "evil.txt").exists()


def test_incomplete_zip_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl, "download", no_download)
    members = spider_members()
    del members["spider_data/dev.json"]
    sha = make_zip(tmp_path / dl.ZIP_NAME, members)
    with pytest.raises(dl.DownloadError, match="dev.json"):
        dl.run(tmp_path, keep_zip=False, file_id="x", expected_sha256=sha)
    assert not (tmp_path / dl.DATASET_DIR).exists()


def test_non_zip_response_points_to_manual_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGdown:
        @staticmethod
        def download(id: str, output: str, quiet: bool, retries: int) -> str:
            Path(output).write_text("<html>Google Drive - quota exceeded</html>")
            return output

    monkeypatch.setitem(sys.modules, "gdown", FakeGdown)
    with pytest.raises(dl.DownloadError, match="Manual download"):
        dl.run(tmp_path, keep_zip=False, file_id="x", expected_sha256="0" * 64)
    assert not (tmp_path / dl.ZIP_NAME).exists()


def test_gdown_exception_points_to_manual_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGdown:
        @staticmethod
        def download(**_: object) -> str:
            raise RuntimeError("Too many users have viewed or downloaded this file recently")

    monkeypatch.setitem(sys.modules, "gdown", FakeGdown)
    with pytest.raises(dl.DownloadError, match="refused the download.*Manual download"):
        dl.run(tmp_path, keep_zip=False, file_id="x", expected_sha256="0" * 64)
