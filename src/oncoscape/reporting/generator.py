from __future__ import annotations

from pathlib import Path
from typing import Any

from oncoscape.core import ensure_directory, write_json


def generate_reports(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    report_dir = ensure_directory(config["render"]["report_dir"]) if not dry_run else Path(config["render"]["report_dir"])
    summary = {
        "report_dir": str(report_dir.resolve()),
        "dry_run": dry_run,
        "status": "scaffold_only",
    }
    if not dry_run:
        write_json(report_dir / "executive_summary.json", summary)
        write_json(report_dir / "wet_lab_summary.json", summary)
    return summary
