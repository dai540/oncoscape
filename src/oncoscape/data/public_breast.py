from __future__ import annotations

from pathlib import Path

import yaml

from oncoscape.core import ensure_parent, write_json


def _first_existing(path: Path, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        full = path / candidate
        if full.exists():
            return full
    return None


def _discover_scrna_raw_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    sample_dirs = [path for path in root.iterdir() if path.is_dir() and _first_existing(path, ["count_matrix_sparse.mtx"]) is not None]
    if sample_dirs:
        return sorted(sample_dirs)
    return sorted(path for path in root.rglob("*") if path.is_dir() and _first_existing(path, ["count_matrix_sparse.mtx"]) is not None)


def build_public_breast_manifest(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    data_root = Path(config.get("fetch", {}).get("output_root") or config.get("paths", {}).get("data_root", "."))
    manifest_path = Path(config["registration"]["manifest_path"])
    sources: list[dict[str, Any]] = []

    ff_specs = [
        ("V1_Breast_Cancer_Block_A_Section_1", "tenx_visium_ff_block_a_section_1", "train"),
        ("V1_Breast_Cancer_Block_A_Section_2", "tenx_visium_ff_block_a_section_2", "val"),
    ]
    for dirname, slide_id, split in ff_specs:
        ff_root = data_root / "breast" / "tenx_visium_ff" / dirname
        ff_counts = _first_existing(ff_root, [f"{dirname}_filtered_feature_bc_matrix.h5", "filtered_feature_bc_matrix.h5"])
        ff_spatial = _first_existing(ff_root, ["spatial"])
        ff_image = _first_existing(ff_spatial, ["tissue_hires_image.png"]) if ff_spatial else None
        if ff_counts and ff_spatial and ff_image:
            sources.append(
                {
                    "name": slide_id,
                    "source_type": "visium",
                    "platform": "visium_fresh_frozen",
                    "patient_id": dirname.lower(),
                    "slide_id": slide_id,
                    "image_path": str(ff_image.resolve()),
                    "counts_path": str(ff_counts.resolve()),
                    "annotation_path": "",
                    "coord_path": str(ff_spatial.resolve()),
                    "mpp_x": 0.5,
                    "mpp_y": 0.5,
                    "coord_unit": "pixel",
                    "split": split,
                }
            )

    ffpe_root = data_root / "breast" / "tenx_visium_ffpe" / "CytAssist_FFPE_Human_Breast_Cancer"
    ffpe_counts = _first_existing(ffpe_root, ["CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5", "filtered_feature_bc_matrix.h5"])
    ffpe_spatial = _first_existing(ffpe_root, ["spatial"])
    ffpe_image = _first_existing(ffpe_spatial, ["tissue_hires_image.png"]) if ffpe_spatial else None
    if ffpe_counts and ffpe_spatial and ffpe_image:
        sources.append(
            {
                "name": "tenx_visium_ffpe_human_breast_cancer",
                "source_type": "visium",
                "platform": "visium_ffpe",
                "patient_id": "tenx_ffpe_human_breast",
                "slide_id": "tenx_visium_ffpe_human_breast_cancer",
                "image_path": str(ffpe_image.resolve()),
                "counts_path": str(ffpe_counts.resolve()),
                "annotation_path": "",
                "coord_path": str(ffpe_spatial.resolve()),
                "mpp_x": 0.5,
                "mpp_y": 0.5,
                "coord_unit": "pixel",
                "split": "test",
            }
        )

    xenium_root = data_root / "breast" / "tenx_xenium_breast" / "Xenium_FFPE_Human_Breast_Cancer_Rep1" / "outs"
    xenium_counts = _first_existing(xenium_root, ["cell_feature_matrix.h5"])
    xenium_cells = _first_existing(xenium_root, ["cells.csv.gz", "cells.csv"])
    xenium_image = _first_existing(xenium_root, ["morphology_mip.ome.tif", "he_image.ome.tif"])
    if xenium_counts and xenium_cells and xenium_image:
        sources.append(
            {
                "name": "tenx_xenium_ffpe_human_breast_cancer_rep1",
                "source_type": "xenium",
                "platform": "xenium",
                "patient_id": "xenium_breast_rep1",
                "slide_id": "tenx_xenium_ffpe_human_breast_cancer_rep1",
                "image_path": str(xenium_image.resolve()),
                "counts_path": str(xenium_counts.resolve()),
                "annotation_path": str(xenium_cells.resolve()),
                "coord_path": str(xenium_cells.resolve()),
                "mpp_x": 0.5,
                "mpp_y": 0.5,
                "coord_unit": "micron",
                "split": "train",
            }
        )

    for label, root_name in [
        ("wu_scrna", "wu_scrna/GSE176078_RAW"),
        ("gse161529", "gse161529/GSE161529_RAW"),
        ("gse235326", "gse235326/GSE235326_RAW"),
    ]:
        scrna_root = data_root / "breast" / root_name
        for sample_dir in _discover_scrna_raw_dirs(scrna_root):
            sample_id = sample_dir.name
            metadata_path = _first_existing(sample_dir, ["metadata.csv"])
            sources.append(
                {
                    "name": f"{label}_{sample_id}",
                    "source_type": "scrna",
                    "platform": "geo_scrna",
                    "patient_id": sample_id,
                    "slide_id": f"{sample_id}_reference",
                    "sample_id": sample_id,
                    "image_path": "",
                    "counts_path": str(sample_dir.resolve()),
                    "annotation_path": str(metadata_path.resolve()) if metadata_path else "",
                    "coord_path": "",
                    "mpp_x": 0.5,
                    "mpp_y": 0.5,
                    "coord_unit": "micron",
                    "split": "train",
                }
            )

    payload = {"sources": sources}
    if dry_run:
        return {
            "manifest_path": str(manifest_path.resolve()),
            "num_sources": len(sources),
            "dry_run": True,
        }

    ensure_parent(manifest_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    summary = {
        "manifest_path": str(manifest_path.resolve()),
        "num_sources": len(sources),
        "source_names": [item["name"] for item in sources],
        "dry_run": False,
    }
    write_json(manifest_path.with_suffix(".summary.json"), summary)
    return summary
