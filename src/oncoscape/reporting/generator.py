from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from oncoscape.core import ensure_directory, write_json


def generate_reports(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    reporting = config["reporting"]
    report_dir = Path(config["render"]["report_dir"])
    summary = {
        "report_dir": str(report_dir.resolve()),
        "dry_run": dry_run,
    }
    if dry_run:
        return summary

    ensure_directory(report_dir)
    test_metrics_path = report_dir / "test_metrics.json"
    per_slide_path = report_dir / "per_slide_metrics.csv"
    slide_metrics = pd.read_csv(per_slide_path) if per_slide_path.exists() else pd.DataFrame()

    executive = {
        "project": config["project_name"],
        "test_metrics_path": str(test_metrics_path.resolve()),
        "num_slides": int(slide_metrics["slide_id"].nunique()) if not slide_metrics.empty else 0,
    }
    wet_lab = {
        "message": "Predictions are estimates for research triage and require wet-lab validation.",
        "prediction_root": str(Path(config["render"]["predictions_dir"]).resolve()),
    }
    developer = {
        "config": {
            "register": config["registration"]["slides_csv_path"],
            "reference": config["reference"]["output_h5ad_path"],
            "teachers": config["teachers"]["output_h5ad_path"],
            "tiles": config["patch_extraction"]["tile_dataset_path"],
            "checkpoint": config["render"]["checkpoint_path"],
        }
    }
    write_json(reporting["executive_summary_path"], executive)
    write_json(reporting["wet_lab_summary_path"], wet_lab)
    write_json(reporting["developer_summary_path"], developer)
    return summary | {"num_slides": executive["num_slides"]}
