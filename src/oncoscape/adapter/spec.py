from __future__ import annotations

from typing import Any


def biomarker_adapter_spec(config: dict[str, Any]) -> dict[str, Any]:
    adapter = config["adapter"]
    return {
        "role": adapter["role"],
        "supported_framework_families": adapter["supported_framework_families"],
        "required_inputs": adapter["required_inputs"],
        "upstream_handoff": config["outputs"]["biomarker_feature_table"],
        "outputs": adapter["outputs"],
        "output_dir": config["outputs"]["adapter_dir"],
    }
