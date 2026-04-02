from __future__ import annotations

from typing import Any


def build_model_spec(config: dict[str, Any]) -> dict[str, Any]:
    train = config["training"]
    tasks = config["tasks"]
    return {
        "model_family": train.get("model_family", "deep_spatial_multitask"),
        "encoder_name": train["encoder_name"],
        "encoder_pretrained": bool(train["encoder_pretrained"]),
        "encoder_out_dim": int(train["encoder_out_dim"]),
        "hidden_dim": int(train["hidden_dim"]),
        "spatial_num_layers": int(train["spatial_num_layers"]),
        "dropout": float(train["dropout"]),
        "encoder_batch_tiles": int(train["encoder_batch_tiles"]),
        "num_compartments": int(len(tasks["compartment_classes"])),
        "num_composition": int(len(tasks["composition_classes"])),
        "num_programs": int(len(tasks["programs"])),
    }
