from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SourceManifestEntry:
    name: str
    source_type: str
    platform: str
    patient_id: str
    slide_id: str
    image_path: str
    counts_path: str
    annotation_path: str = ""
    coord_path: str = ""
    mpp_x: float | None = None
    mpp_y: float | None = None
    coord_unit: str = "micron"
    split: str = ""
    sample_id: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(path: str | Path) -> list[SourceManifestEntry]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    entries = []
    for item in data.get("sources", []):
        entries.append(SourceManifestEntry(**item))
    return entries
