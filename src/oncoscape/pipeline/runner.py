from __future__ import annotations

from typing import Any

from oncoscape.data import build_registry
from oncoscape.evaluation import evaluate_and_render
from oncoscape.labels import build_teachers
from oncoscape.preprocessing import extract_patches_and_graphs
from oncoscape.reference import build_reference_atlas
from oncoscape.reporting import generate_reports
from oncoscape.training import train_breast_model
from oncoscape.validation import run_preflight


def run_pipeline(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    stages: list[tuple[str, Any]] = [
        ("preflight", run_preflight),
        ("register", build_registry),
        ("build_reference", build_reference_atlas),
        ("make_teachers", build_teachers),
        ("extract_patches", extract_patches_and_graphs),
        ("train", train_breast_model),
        ("eval_render", evaluate_and_render),
        ("report", generate_reports),
    ]
    results: dict[str, Any] = {}
    for name, fn in stages:
        result = fn(config=config, dry_run=dry_run)
        results[name] = result
        if name == "preflight" and not result.get("ok", False) and not dry_run:
            raise RuntimeError("preflight failed; fix config or data paths before running the pipeline")
    return results
