from __future__ import annotations

from typing import Any


def preflight_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "required_tools": [
            "HEST",
            "scvi-tools",
            "cell2location",
            "OpenSlide",
            "pathology foundation encoder",
        ],
        "primary_outputs": config["targets"]["primary"],
        "secondary_outputs": config["targets"]["secondary"],
    }
