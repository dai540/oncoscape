from __future__ import annotations

from typing import Any


def evaluate_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": "06_eval_and_render",
        "goal": "run strict holdout evaluation for compartment and broad TME maps",
        "metrics": config["evaluation"]["metrics"],
        "outputs": {
            "prediction_dir": config["outputs"]["prediction_dir"],
            "report_dir": config["outputs"]["report_dir"],
        },
    }
