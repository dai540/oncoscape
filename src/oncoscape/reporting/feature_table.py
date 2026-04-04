from __future__ import annotations

from typing import Any


def biomarker_feature_table_spec(config: dict[str, Any]) -> dict[str, Any]:
    feature_table = config["feature_table"]
    return {
        "unit": feature_table["unit"],
        "output_path": config["outputs"]["biomarker_feature_table"],
        "required_id_columns": feature_table["required_id_columns"],
        "feature_groups": {
            "compartment": feature_table["compartment_features"],
            "composition": feature_table["composition_features"],
            "programs": feature_table["program_features"],
            "topology": feature_table["topology_features"],
            "hotspots": feature_table["hotspot_features"],
            "uncertainty": feature_table["uncertainty_features"],
        },
    }
