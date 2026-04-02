from __future__ import annotations

from pathlib import Path
from typing import Any

from oncoscape.core import ensure_directory, write_json
from oncoscape.models import build_model_spec


def train_breast_model(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    cfg = config["train_run"]
    training = config["training"]
    ensure_directory(cfg["checkpoint_dir"]) if not dry_run else None
    ensure_directory(cfg["report_dir"]) if not dry_run else None

    summary = {
        "checkpoint_dir": str(Path(cfg["checkpoint_dir"]).resolve()),
        "report_dir": str(Path(cfg["report_dir"]).resolve()),
        "epochs": int(training["epochs"]),
        "device": str(training["device"]),
        "model_spec": build_model_spec(config),
        "dry_run": dry_run,
        "status": "scaffold_only",
    }
    if not dry_run:
        Path(cfg["checkpoint_dir"], "best.pt").touch()
        Path(cfg["checkpoint_dir"], "last.pt").touch()
        Path(cfg["report_dir"], "train_metrics.csv").write_text("epoch,loss\n", encoding="utf-8")
        Path(cfg["report_dir"], "val_metrics.csv").write_text("epoch,loss\n", encoding="utf-8")
        write_json(Path(cfg["report_dir"]) / "train_summary.json", summary)
    return summary
