from __future__ import annotations

from typing import Any


def rendering_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "maps": [
            "compartment_map",
            "broad_tme_map",
            "program_map",
            "uncertainty_map",
        ],
        "output_root": config["outputs"]["prediction_dir"],
    }
