from __future__ import annotations

from pathlib import Path
from typing import Any

from oncoscape.core import ensure_parent, write_json


def build_teachers(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    teachers = config["teachers"]
    outputs = {
        "output_h5ad_path": str(Path(teachers["output_h5ad_path"]).resolve()),
        "output_ontology_json": str(Path(teachers["output_ontology_json"]).resolve()),
        "output_programs_json": str(Path(teachers["output_programs_json"]).resolve()),
        "use_cell2location": bool(teachers.get("use_cell2location", False)),
        "use_scvi_label_transfer": bool(teachers.get("use_scvi_label_transfer", False)),
        "dry_run": dry_run,
        "status": "scaffold_only",
    }
    if not dry_run:
        ensure_parent(teachers["output_h5ad_path"]).touch()
        write_json(teachers["output_ontology_json"], {"status": "placeholder"})
        write_json(teachers["output_programs_json"], {"status": "placeholder"})
        write_json(Path(teachers["output_h5ad_path"]).with_suffix(".summary.json"), outputs)
    return outputs
