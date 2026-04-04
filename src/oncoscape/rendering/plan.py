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
        "summaries": [
            "slide_level_summary",
            "candidate_validation_regions",
        ],
        "output_root": config["outputs"]["prediction_dir"],
    }
