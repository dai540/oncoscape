from __future__ import annotations

from typing import Any

from oncoscape.reporting.feature_table import biomarker_feature_table_spec


def reporting_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal": "prepare pathology and wet-lab review summaries and export biomarker features",
        "summary_outputs": [
            "slide-level summary",
            "hotspot list",
            "candidate validation regions",
        ],
        "biomarker_feature_table": biomarker_feature_table_spec(config),
        "report_dir": config["outputs"]["report_dir"],
    }
