from __future__ import annotations

from typing import Any

from oncoscape.models import model_spec


def train_model_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": "05_train_model",
        "goal": "train a compartment-first and broad-TME-first model",
        "model": model_spec(config),
        "outputs": {
            "checkpoint_dir": config["outputs"]["checkpoint_dir"],
            "best_checkpoint": f"{config['outputs']['checkpoint_dir']}/best.pt",
        },
    }
