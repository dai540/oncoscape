from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from oncoscape.config import load_config
from oncoscape.core import ensure_directory, write_json


def _normalize_path(value: str | Path) -> str:
    return str(Path(value).resolve())


def _replace_paths(node: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(node, dict):
        return {key: _replace_paths(value, replacements) for key, value in node.items()}
    if isinstance(node, list):
        return [_replace_paths(value, replacements) for value in node]
    if isinstance(node, str):
        updated = node
        for source, dest in replacements:
            updated = updated.replace(source, dest)
        return updated
    return node


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def initialize_hpc_project(
    project_root: str | Path,
    code_root: str | Path,
    data_root: str | Path | None = None,
    run_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    code_root = Path(code_root).resolve()
    data_root = Path(data_root).resolve() if data_root is not None else (project_root / "data").resolve()
    run_root = Path(run_root).resolve() if run_root is not None else (project_root / "run").resolve()
    config_root = run_root / "config"

    base_config = load_config(code_root / "configs" / "breast_hpc.yaml")
    generated = deepcopy(base_config)
    replacements = [
        ("/project/code/oncoscape", _normalize_path(code_root)),
        ("/project/data", _normalize_path(data_root)),
        ("/project/run", _normalize_path(run_root)),
        ("/project", _normalize_path(project_root)),
    ]
    generated = _replace_paths(generated, replacements)
    generated["paths"] = {
        "project_root": _normalize_path(project_root),
        "code_root": _normalize_path(code_root),
        "data_root": _normalize_path(data_root),
        "run_root": _normalize_path(run_root),
    }

    generated_config = config_root / "breast_hpc.yaml"
    downloads_config = config_root / "breast_downloads.yaml"
    sources_template = config_root / "breast_sources.template.yaml"

    planned_dirs = [
        config_root,
        run_root / "data_interim" / "adata",
        run_root / "data_interim" / "labels",
        run_root / "data_interim" / "patches",
        run_root / "data_interim" / "graphs",
        run_root / "outputs" / "checkpoints",
        run_root / "outputs" / "reports",
        run_root / "outputs" / "predictions",
    ]

    if not dry_run:
        for directory in planned_dirs:
            ensure_directory(directory)
        _write_yaml(generated_config, generated)
        downloads_config.write_text((code_root / "configs" / "breast_downloads.template.yaml").read_text(encoding="utf-8"), encoding="utf-8")
        sources_template.write_text((code_root / "configs" / "breast_sources.template.yaml").read_text(encoding="utf-8"), encoding="utf-8")
        write_json(
            config_root / "init_summary.json",
            {
                "project_root": _normalize_path(project_root),
                "code_root": _normalize_path(code_root),
                "data_root": _normalize_path(data_root),
                "run_root": _normalize_path(run_root),
                "generated_config": str(generated_config.resolve()),
                "downloads_config": str(downloads_config.resolve()),
                "sources_template": str(sources_template.resolve()),
            },
        )

    return {
        "project_root": _normalize_path(project_root),
        "code_root": _normalize_path(code_root),
        "data_root": _normalize_path(data_root),
        "run_root": _normalize_path(run_root),
        "generated_config": str(generated_config.resolve()),
        "downloads_config": str(downloads_config.resolve()),
        "sources_template": str(sources_template.resolve()),
        "created_dirs": [str(path.resolve()) for path in planned_dirs],
        "dry_run": dry_run,
    }
