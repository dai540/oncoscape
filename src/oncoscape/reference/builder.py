from __future__ import annotations

from typing import Any


def build_reference_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": "02_build_reference",
        "goal": "build a breast broad-cell-type reference with scvi-tools",
        "inputs": config["data"]["required_sources"][-2:],
        "outputs": {
            "reference_dir": config["outputs"]["reference_dir"],
            "atlas": f"{config['outputs']['reference_dir']}/reference_atlas.h5ad",
        },
    }
