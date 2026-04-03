from __future__ import annotations

import argparse
import json

from oncoscape.bootstrap import initialize_hpc_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize an HPC project layout and generate a resolved breast config.")
    parser.add_argument("--project-root", required=True, help="Project root on the target filesystem, e.g. /project")
    parser.add_argument("--code-root", required=True, help="Path to the cloned oncoscape repository")
    parser.add_argument("--data-root", default=None, help="Optional data root; defaults to <project-root>/data")
    parser.add_argument("--run-root", default=None, help="Optional run root; defaults to <project-root>/run")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without writing files")
    args = parser.parse_args()
    result = initialize_hpc_project(
        project_root=args.project_root,
        code_root=args.code_root,
        data_root=args.data_root,
        run_root=args.run_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
