"""Download, verify and extract the Spider dataset into data/spider_data/.

The official spider_data.zip is fetched from Google Drive with gdown, checked against a pinned
SHA256 and extracted with path checks. Re-running is safe:

* if data/spider_data/ was already extracted from a verified zip, nothing is downloaded;
* if a kept zip with the right hash is present, it is extracted without downloading again.

On a hash mismatch nothing is deleted. The actual hash is printed so you can decide what to do.
If Google Drive refuses the download (quota exceeded, confirmation page), the script stops and
points to the manual steps in the README. It never retries.

Example:
    python scripts/download_spider.py
    python scripts/download_spider.py --keep-zip
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

DRIVE_FILE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
EXPECTED_SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
ZIP_NAME = "spider_data.zip"
DATASET_DIR = "spider_data"
MARKER = ".verified_sha256"
REQUIRED = ("dev.json", "train_spider.json", "train_others.json", "tables.json", "database")
MANUAL_HINT = "Follow the manual steps in README.md ('Data setup' -> 'Manual download')."


class DownloadError(RuntimeError):
    """A failure the user has to resolve, with a message saying how."""


def sha256_of(path: Path) -> str:
    """SHA256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_verified_extract(dataset_dir: Path, expected_sha256: str) -> bool:
    """True if ``dataset_dir`` holds a complete extract of the zip with ``expected_sha256``."""
    marker = dataset_dir / MARKER
    return (
        marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == expected_sha256
        and all((dataset_dir / name).exists() for name in REQUIRED)
    )


def download(file_id: str, zip_path: Path) -> None:
    """Fetch the Drive file once with gdown and check it really is a zip."""
    import gdown

    partial = zip_path.with_name(zip_path.name + ".part")
    try:
        result = gdown.download(id=file_id, output=str(partial), quiet=False, retries=0)
    except Exception as exc:  # gdown raises several types for quota/permission/HTML responses
        raise DownloadError(f"Google Drive refused the download ({exc}). {MANUAL_HINT}") from exc
    if result is None or not partial.is_file():
        raise DownloadError(f"Google Drive returned no file. {MANUAL_HINT}")
    if not zipfile.is_zipfile(partial):
        raise DownloadError(
            f"Google Drive returned something that is not a zip (likely a quota or confirmation "
            f"page); it was left at {partial} for inspection. {MANUAL_HINT}"
        )
    partial.replace(zip_path)


def _zip_root(archive: zipfile.ZipFile) -> str:
    """The single top-level directory the zip must contain, after checking every member path."""
    roots = set()
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
            raise DownloadError(f"refusing to extract unsafe path in zip: {info.filename!r}")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise DownloadError(f"refusing to extract symlink in zip: {info.filename!r}")
        roots.add(path.parts[0])
    roots.discard("__MACOSX")
    if len(roots) != 1:
        raise DownloadError(f"expected one top-level folder in the zip, found {sorted(roots)}")
    return roots.pop()


def extract(zip_path: Path, dataset_dir: Path, sha256: str) -> None:
    """Extract into a staging folder, check the layout, then move it into place."""
    staging = dataset_dir.with_name(dataset_dir.name + ".extracting")
    if staging.exists():
        shutil.rmtree(staging)
    with zipfile.ZipFile(zip_path) as archive:
        root = _zip_root(archive)
        members = [m for m in archive.namelist() if not m.startswith("__MACOSX")]
        archive.extractall(staging, members=members)
    extracted = staging / root
    missing = [name for name in REQUIRED if not (extracted / name).exists()]
    if missing:
        raise DownloadError(f"zip is missing expected entries {missing}; staging left at {staging}")
    (extracted / MARKER).write_text(sha256 + "\n", encoding="utf-8")
    extracted.replace(dataset_dir)
    shutil.rmtree(staging)


def run(data_dir: Path, keep_zip: bool, file_id: str, expected_sha256: str) -> Path:
    """Make ``data_dir/spider_data`` a verified extract and return its path."""
    dataset_dir = data_dir / DATASET_DIR
    zip_path = data_dir / ZIP_NAME
    if is_verified_extract(dataset_dir, expected_sha256):
        print(f"Verified Spider already present at {dataset_dir}; nothing to do.")
        return dataset_dir
    if dataset_dir.exists():
        raise DownloadError(
            f"{dataset_dir} exists but is not a verified extract. Move or delete it yourself, "
            "then rerun."
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    if zip_path.is_file():
        print(f"Using existing {zip_path}")
    else:
        download(file_id, zip_path)

    actual = sha256_of(zip_path)
    if actual != expected_sha256:
        raise DownloadError(
            f"SHA256 mismatch for {zip_path}\n  expected {expected_sha256}\n  actual   {actual}\n"
            "Nothing was deleted or extracted. Check the file, then rerun with --sha256 <hash> if "
            "you decide to accept it."
        )
    print(f"SHA256 verified: {actual}")
    extract(zip_path, dataset_dir, actual)
    if keep_zip:
        print(f"Kept {zip_path}")
    else:
        zip_path.unlink()
    print(f"Spider extracted to {dataset_dir}")
    return dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="parent folder (default: data)")
    parser.add_argument("--keep-zip", action="store_true", help="keep spider_data.zip after extracting")
    parser.add_argument("--file-id", default=DRIVE_FILE_ID, help="Google Drive file id")
    parser.add_argument("--sha256", default=EXPECTED_SHA256, help="expected SHA256 of the zip")
    args = parser.parse_args()
    try:
        run(args.data_dir, args.keep_zip, args.file_id, args.sha256)
    except DownloadError as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
