from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from oncoscape.core import ensure_directory


DEFAULT_SELECTION_WEIGHTS: dict[str, float] = {
    "compartment_macro_f1": 1.0,
    "compartment_balanced_accuracy": 0.5,
    "composition_mean_pearson": 1.0,
    "program_mean_pearson": 1.0,
    "composition_js_divergence": -1.0,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_val_metrics(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    record = frame.iloc[-1].to_dict()
    return {str(key): float(value) for key, value in record.items() if key != "epoch"}


def compute_selection_score(metrics: dict[str, float], weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_SELECTION_WEIGHTS
    score = 0.0
    for key, weight in weights.items():
        score += float(weight) * float(metrics.get(key, 0.0))
    return float(score)


def _copy_tree_contents(src: Path, dst: Path) -> None:
    ensure_directory(dst)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def _copy_if_different(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def summarize_seed_sweep(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    selection_cfg = config["selection"]
    seed_sweep_dir = Path(selection_cfg["seed_sweep_dir"])
    report_dir = Path(config["render"]["report_dir"])
    summary_json = Path(selection_cfg["summary_json"])
    summary_csv = Path(selection_cfg["summary_csv"])
    canonical_checkpoint_dir = Path(config["train_run"]["checkpoint_dir"])
    canonical_predictions_dir = Path(config["render"]["predictions_dir"])
    summary = {
        "seed_sweep_dir": str(seed_sweep_dir.resolve()),
        "summary_json": str(summary_json.resolve()),
        "summary_csv": str(summary_csv.resolve()),
        "dry_run": dry_run,
    }
    if dry_run:
        return summary

    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(seed_sweep_dir.glob("seed_*")):
        val_path = seed_dir / "reports" / "val_metrics.csv"
        test_path = seed_dir / "reports" / "test_metrics.json"
        if not val_path.exists() or not test_path.exists():
            continue
        val_metrics = _load_val_metrics(val_path)
        test_metrics = _load_json(test_path)
        row = {
            "seed": seed_dir.name.replace("seed_", ""),
            "seed_dir": str(seed_dir.resolve()),
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        row["selection_score"] = compute_selection_score(val_metrics, selection_cfg.get("metric_weights"))
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No seed runs with reports found in {seed_sweep_dir}")

    frame = pd.DataFrame(rows).sort_values(["selection_score", "seed"], ascending=[False, True]).reset_index(drop=True)
    best = frame.iloc[0].to_dict()
    best_seed_dir = Path(str(best["seed_dir"]))

    ensure_directory(summary_json.parent)
    ensure_directory(summary_csv.parent)
    frame.to_csv(summary_csv, index=False)
    summary_payload = {
        "selection_method": "validation_weighted_score",
        "metric_weights": selection_cfg.get("metric_weights", DEFAULT_SELECTION_WEIGHTS),
        "best_seed": best["seed"],
        "best_seed_dir": str(best_seed_dir.resolve()),
        "best_selection_score": float(best["selection_score"]),
        "num_seed_runs": int(len(frame)),
        "results": frame.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    if selection_cfg.get("promote_best", True):
        _copy_tree_contents(best_seed_dir / "checkpoints", canonical_checkpoint_dir)
        _copy_tree_contents(best_seed_dir / "predictions", canonical_predictions_dir)
        for name in ["test_metrics.json", "per_slide_metrics.csv", "train_metrics.csv", "val_metrics.csv", "train_summary.json"]:
            src = best_seed_dir / "reports" / name
            if src.exists():
                ensure_directory(report_dir)
                _copy_if_different(src, report_dir / name)
        _copy_if_different(summary_json, report_dir / summary_json.name)
        _copy_if_different(summary_csv, report_dir / summary_csv.name)

    summary.update(
        {
            "best_seed": str(best["seed"]),
            "best_seed_dir": str(best_seed_dir.resolve()),
            "selection_score": float(best["selection_score"]),
            "num_seed_runs": int(len(frame)),
            "promoted": bool(selection_cfg.get("promote_best", True)),
        }
    )
    return summary
