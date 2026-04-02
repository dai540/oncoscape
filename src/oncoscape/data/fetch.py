from __future__ import annotations

import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml

from oncoscape.core import ensure_directory, write_json


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}


def _load_downloads(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return list(data.get("downloads", []))


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    with urllib.request.urlopen(req) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _extract_archive(archive_path: Path, extract_dir: Path, cleanup_archive: bool = False) -> list[str]:
    extracted: list[str] = []
    extract_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path) as tar:
            tar.extractall(extract_dir)
            extracted.extend(str((extract_dir / member.name).resolve()) for member in tar.getmembers() if member.name)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
            extracted.extend(str((extract_dir / member).resolve()) for member in zf.namelist() if member)
    else:
        raise ValueError(f"unsupported archive format for extraction: {archive_path}")
    if cleanup_archive:
        archive_path.unlink(missing_ok=True)
    return extracted


def _expand_nested_archives(root: Path, cleanup_archive: bool = False) -> list[str]:
    extracted: list[str] = []
    patterns = ("*.tar.gz", "*.tgz", "*.tar", "*.zip")
    archives = []
    for pattern in patterns:
        archives.extend(root.rglob(pattern))
    for archive in sorted(path for path in archives if path.is_file()):
        target_name = archive.name
        if target_name.endswith(".tar.gz"):
            folder_name = target_name[:-7]
        elif target_name.endswith(".tgz"):
            folder_name = target_name[:-4]
        else:
            folder_name = archive.stem
        extract_dir = archive.parent / folder_name
        if extract_dir.exists() and any(extract_dir.iterdir()):
            continue
        extracted.extend(_extract_archive(archive, extract_dir, cleanup_archive=cleanup_archive))
    return extracted


def fetch_public_data(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    fetch_cfg = config.get("fetch", {})
    catalog_path = fetch_cfg.get("catalog_path")
    if not catalog_path and "paths" in config and "run_root" in config["paths"]:
        catalog_path = str(Path(config["paths"]["run_root"]) / "config" / "breast_downloads.yaml")
    if not catalog_path:
        raise ValueError("config.fetch.catalog_path is required for public data fetching")
    downloads = _load_downloads(catalog_path)
    default_output_root = config.get("paths", {}).get("data_root", ".")
    out_root = ensure_directory(fetch_cfg.get("output_root", default_output_root))

    planned = []
    fetched = []
    extracted = []
    skip_existing = bool(fetch_cfg.get("skip_existing", True))
    cleanup_archive = bool(fetch_cfg.get("cleanup_archive_after_extract", False))
    for item in downloads:
        rel_path = item["target_relpath"]
        url = item["url"]
        dest = out_root / rel_path
        extract = bool(item.get("extract", False))
        extract_relpath = item.get("extract_relpath")
        extract_dir = out_root / extract_relpath if extract_relpath else dest.parent
        planned.append(
            {
                "name": item.get("name", dest.name),
                "url": url,
                "target": str(dest.resolve()),
                "extract": extract,
                "extract_dir": str(extract_dir.resolve()) if extract else "",
            }
        )
        if dry_run:
            continue
        if skip_existing and dest.exists():
            fetched.append(str(dest.resolve()))
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            _download(url, dest)
            fetched.append(str(dest.resolve()))
        if extract:
            extracted.extend(_extract_archive(dest, extract_dir, cleanup_archive=cleanup_archive))
            if bool(item.get("expand_nested_archives", False)):
                extracted.extend(_expand_nested_archives(extract_dir, cleanup_archive=cleanup_archive))

    summary = {
        "catalog_path": str(Path(catalog_path).resolve()),
        "output_root": str(out_root.resolve()),
        "num_files": len(downloads),
        "planned": planned,
        "fetched": fetched,
        "extracted": extracted,
        "dry_run": dry_run,
    }
    if not dry_run:
        write_json(out_root / "fetch_summary.json", summary)
    return summary
