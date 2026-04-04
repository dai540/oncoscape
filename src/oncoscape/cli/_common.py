from __future__ import annotations

import argparse
import json
from typing import Callable

from oncoscape.core import load_yaml


def run_step(description: str, handler: Callable[[dict], dict]) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to breast_hpc.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    print(json.dumps(handler(config), indent=2))
