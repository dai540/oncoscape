from __future__ import annotations

import argparse
import json

from oncoscape.config import load_config


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and emit planned outputs")
    return parser


def run_cli(description: str, handler):
    parser = build_parser(description)
    args = parser.parse_args()
    config = load_config(args.config)
    result = handler(config=config, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
