from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from oncoscape.data import load_manifest


REQUIRED_MANIFEST_FIELDS = {
    "name",
    "source_type",
    "platform",
    "patient_id",
    "slide_id",
    "image_path",
    "counts_path",
    "annotation_path",
    "coord_path",
    "coord_unit",
    "split",
}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_preflight(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    manifest_path = Path(config["registration"]["manifest_path"])
    checks: list[dict[str, Any]] = []

    checks.append({"name": "config_exists", "ok": True, "detail": "config loaded"})
    checks.append({"name": "manifest_exists", "ok": manifest_path.exists(), "detail": str(manifest_path)})

    entries = load_manifest(manifest_path) if manifest_path.exists() else []
    checks.append({"name": "manifest_has_entries", "ok": len(entries) > 0, "detail": len(entries)})

    required_modules = ["anndata", "pandas", "numpy", "PIL", "yaml"]
    optional_modules = ["pyarrow", "torch", "openslide", "scvi", "cell2location"]
    for module in required_modules:
        checks.append({"name": f"module::{module}", "ok": _module_available(module), "detail": "required"})
    for module in optional_modules:
        checks.append({"name": f"module::{module}", "ok": _module_available(module), "detail": "optional"})

    output_paths = [
        Path(config["registration"]["output_dir"]),
        Path(config["train_run"]["checkpoint_dir"]),
        Path(config["train_run"]["report_dir"]),
        Path(config["render"]["predictions_dir"]),
    ]
    for path in output_paths:
        parent = path if path.suffix == "" else path.parent
        checks.append({"name": f"writable::{parent}", "ok": parent.exists() or parent.parent.exists(), "detail": str(parent)})

    missing_files = []
    for entry in entries:
        row = entry.to_row()
        missing = []
        for field in REQUIRED_MANIFEST_FIELDS:
            if field not in row:
                missing.append(f"missing-field:{field}")
        for field in ["image_path", "counts_path"]:
            value = row.get(field, "")
            if value and not Path(value).exists():
                missing.append(f"missing-file:{field}")
        if missing:
            missing_files.append({"slide_id": entry.slide_id, "issues": missing})
    checks.append({"name": "manifest_paths", "ok": len(missing_files) == 0, "detail": missing_files})

    all_ok = all(item["ok"] for item in checks if item["detail"] != "optional")
    return {"ok": all_ok, "checks": checks, "dry_run": dry_run}
