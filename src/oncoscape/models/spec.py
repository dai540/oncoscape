from __future__ import annotations

from typing import Any


def model_spec(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy": config["model"]["strategy"],
        "encoder": config["model"]["encoder"],
        "spatial_module": config["model"]["spatial_module"],
        "primary_outputs": config["targets"]["primary"],
        "secondary_outputs": config["targets"]["secondary"],
    }
