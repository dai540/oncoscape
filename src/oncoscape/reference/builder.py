from __future__ import annotations

from pathlib import Path
from typing import Any

from oncoscape.core import ensure_parent, write_json


def build_reference_atlas(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    ref = config["reference"]
    outputs = {
        "output_h5ad_path": str(Path(ref["output_h5ad_path"]).resolve()),
        "output_markers_path": str(Path(ref["output_markers_path"]).resolve()),
        "output_qc_report_path": str(Path(ref["output_qc_report_path"]).resolve()),
        "num_inputs": int(len(ref.get("input_h5ad_paths", []))),
        "dry_run": dry_run,
        "status": "scaffold_only",
    }
    if not dry_run:
        ensure_parent(ref["output_h5ad_path"]).touch()
        ensure_parent(ref["output_markers_path"]).write_text("group,rank,gene\n", encoding="utf-8")
        ensure_parent(ref["output_qc_report_path"]).write_text(
            "<html><body><h1>oncoscape reference QC scaffold</h1></body></html>",
            encoding="utf-8",
        )
        write_json(Path(ref["output_h5ad_path"]).with_suffix(".summary.json"), outputs)
    return outputs
