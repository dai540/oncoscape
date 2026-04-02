from __future__ import annotations

from pathlib import Path
from typing import Any

from oncoscape.core import ensure_directory, write_json


def evaluate_and_render(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    render = config["render"]
    predictions_dir = ensure_directory(render["predictions_dir"]) if not dry_run else Path(render["predictions_dir"])
    report_dir = ensure_directory(render["report_dir"]) if not dry_run else Path(render["report_dir"])
    summary = {
        "checkpoint_path": str(Path(render["checkpoint_path"]).resolve()),
        "predictions_dir": str(predictions_dir.resolve()),
        "report_dir": str(report_dir.resolve()),
        "split": config["evaluation"]["split"],
        "dry_run": dry_run,
        "status": "scaffold_only",
    }
    if not dry_run:
        write_json(Path(report_dir) / "test_metrics.json", summary)
        Path(report_dir, "per_slide_metrics.csv").write_text("slide_id,metric\n", encoding="utf-8")
    return summary
