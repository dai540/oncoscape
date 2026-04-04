from __future__ import annotations

from typing import Any


def reporting_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal": "prepare pathology and wet-lab review summaries",
        "summary_outputs": [
            "slide-level summary",
            "hotspot list",
            "candidate validation regions",
        ],
        "report_dir": config["outputs"]["report_dir"],
    }
