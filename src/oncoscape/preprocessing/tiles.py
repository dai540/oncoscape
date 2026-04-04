from __future__ import annotations

from typing import Any


def extract_tiles_plan(config: dict[str, Any]) -> dict[str, Any]:
    tiling = config["tiling"]
    return {
        "step": "04_extract_tiles",
        "goal": "extract H&E tiles and build spatial graphs",
        "tiling": tiling,
        "outputs": {
            "tile_dir": config["outputs"]["tile_dir"],
            "tile_dataset": f"{config['outputs']['tile_dir']}/tile_dataset.parquet",
            "graph_dir": f"{config['outputs']['tile_dir']}/graphs",
        },
    }
