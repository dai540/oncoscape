from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .paths import ensure_parent


def _git_commit(cwd: str | Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _config_hash(config: dict[str, Any]) -> str:
    dumped = yaml.safe_dump(config, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def write_provenance(path: str | Path, config: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    payload = {
        "git_commit": _git_commit(),
        "config_hash": _config_hash(config),
    }
    if extra:
        payload.update(extra)
    path = ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return Path(path)
