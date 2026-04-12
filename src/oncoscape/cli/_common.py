from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import yaml


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_step(description: str, handler: Callable[[dict], dict]) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to breast_hpc.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    print(json.dumps(handler(config), indent=2))
