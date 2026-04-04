from __future__ import annotations

from typing import Any


def build_teacher_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": "03_build_teachers",
        "goal": "build breast-specific compartment and broad-TME teachers",
        "teacher_sources": {
            "compartment": ["Wu pathology-reviewed metadata"],
            "broad_tme": ["10x Xenium breast", "Visium + cell2location"],
            "programs": ["breast program panel"],
        },
        "outputs": {
            "teacher_dir": config["outputs"]["teacher_dir"],
            "teacher_labels": f"{config['outputs']['teacher_dir']}/teacher_labels.h5ad",
        },
    }
