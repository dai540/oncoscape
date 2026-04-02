from __future__ import annotations

from pathlib import Path
from typing import Any

from oncoscape.core import ensure_directory, ensure_parent, write_json


def extract_patches_and_graphs(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    cfg = config["patch_extraction"]
    patches_dir = ensure_directory(cfg["patches_dir"]) if not dry_run else Path(cfg["patches_dir"])
    graphs_dir = ensure_directory(cfg["graphs_dir"]) if not dry_run else Path(cfg["graphs_dir"])
    tile_dataset_path = Path(cfg["tile_dataset_path"])
    outputs = {
        "patches_dir": str(patches_dir.resolve()),
        "graphs_dir": str(graphs_dir.resolve()),
        "tile_dataset_path": str(tile_dataset_path.resolve()),
        "dry_run": dry_run,
        "status": "scaffold_only",
    }
    if not dry_run:
        ensure_parent(tile_dataset_path).write_text("", encoding="utf-8")
        write_json(tile_dataset_path.with_suffix(".summary.json"), outputs)
    return outputs
