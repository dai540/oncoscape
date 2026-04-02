from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import ensure_parent


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return Path(path)
