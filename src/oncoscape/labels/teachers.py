from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from oncoscape.core import ensure_parent, write_json


COMPARTMENT_KEYWORDS = {
    "invasive_tumor": ["invasive", "tumor", "carcinoma"],
    "in_situ_tumor": ["in situ", "dcis"],
    "stroma": ["stroma", "stromal", "fibro"],
    "immune_rich": ["immune", "lymph", "myeloid"],
    "adipose_normal": ["adipose", "normal", "benign", "fat"],
    "necrosis_background": ["necrosis", "background", "blank"],
}

COMPOSITION_MARKERS = {
    "malignant": ["EPCAM", "KRT8", "KRT18", "ERBB2"],
    "CAF": ["COL1A1", "COL1A2", "DCN", "LUM"],
    "endothelial": ["PECAM1", "VWF", "KDR", "EMCN"],
    "myeloid": ["LYZ", "FCER1G", "CTSS", "TYROBP"],
    "T_NK": ["NKG7", "TRBC1", "CD3D", "CD3E"],
    "B_plasma": ["MS4A1", "CD79A", "MZB1", "JCHAIN"],
}


def _to_dense(x: Any) -> np.ndarray:
    return np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float32)


def _build_reference_signatures(reference: ad.AnnData, broad_cell_types: list[str]) -> tuple[list[str], np.ndarray]:
    matrix = _to_dense(reference.X)
    matrix = np.log1p(matrix)
    signatures = []
    for label in broad_cell_types:
        mask = (reference.obs["broad_cell_type"].astype(str) == label).to_numpy()
        if mask.sum() == 0:
            signatures.append(np.zeros(reference.n_vars, dtype=np.float32))
        else:
            signatures.append(matrix[mask].mean(axis=0))
    return list(map(str, reference.var_names)), np.vstack(signatures)


def _infer_compartment(obs: pd.DataFrame, classes: list[str], preferred_column: str) -> pd.Series:
    candidate_columns = [preferred_column] + [col for col in obs.columns if col.lower() in {"classification", "pathology", "compartment", "annotation"}]
    raw = None
    for column in candidate_columns:
        if column in obs.columns:
            raw = obs[column].astype(str)
            break
    if raw is None:
        return pd.Series([classes[0]] * len(obs), index=obs.index)

    mapped = []
    for value in raw:
        lowered = value.lower()
        hit = next(
            (label for label, keywords in COMPARTMENT_KEYWORDS.items() if any(keyword in lowered for keyword in keywords)),
            classes[0],
        )
        mapped.append(hit if hit in classes else classes[0])
    return pd.Series(mapped, index=obs.index)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.clip(exp_x.sum(axis=1, keepdims=True), 1e-8, None)


def _composition_from_reference(
    adata: ad.AnnData,
    ref_genes: list[str],
    ref_signatures: np.ndarray,
    broad_cell_types: list[str],
) -> np.ndarray:
    var_lookup = {gene: idx for idx, gene in enumerate(map(str, adata.var_names))}
    shared = [gene for gene in ref_genes if gene in var_lookup]
    if not shared:
        return np.full((adata.n_obs, len(broad_cell_types)), 1.0 / len(broad_cell_types), dtype=np.float32)
    ref_idx = [ref_genes.index(gene) for gene in shared]
    slide_idx = [var_lookup[gene] for gene in shared]
    ref_block = ref_signatures[:, ref_idx]
    slide_block = np.log1p(_to_dense(adata.X)[:, slide_idx])
    denom = np.linalg.norm(slide_block, axis=1, keepdims=True) * np.linalg.norm(ref_block, axis=1, keepdims=True).T
    scores = slide_block @ ref_block.T / np.clip(denom, 1e-8, None)
    marker_scores = np.zeros_like(scores, dtype=np.float32)
    for class_idx, label in enumerate(broad_cell_types):
        markers = [gene for gene in COMPOSITION_MARKERS.get(label, []) if gene in var_lookup]
        if markers:
            marker_idx = [var_lookup[gene] for gene in markers]
            marker_scores[:, class_idx] = np.log1p(_to_dense(adata.X)[:, marker_idx]).mean(axis=1)
    combined = 0.7 * scores.astype(np.float32) + 0.3 * marker_scores
    return _softmax(combined.astype(np.float32))


def _program_scores(adata: ad.AnnData, program_names: list[str]) -> np.ndarray:
    matrix = np.log1p(_to_dense(adata.X))
    if adata.n_vars == 0:
        return np.zeros((adata.n_obs, len(program_names)), dtype=np.float32)
    gene_indices = np.array_split(np.arange(adata.n_vars), len(program_names))
    programs = []
    for idxs in gene_indices:
        if len(idxs) == 0:
            programs.append(np.zeros(adata.n_obs, dtype=np.float32))
        else:
            programs.append(matrix[:, idxs].mean(axis=1).astype(np.float32))
    return np.vstack(programs).T


def build_teachers(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    teachers = config["teachers"]
    tasks = config["tasks"]
    slides = pd.read_csv(teachers["slides_csv_path"])
    slides = slides[slides["source_type"].isin(["visium", "xenium"])].reset_index(drop=True)
    outputs = {
        "output_h5ad_path": str(Path(teachers["output_h5ad_path"]).resolve()),
        "output_ontology_json": str(Path(teachers["output_ontology_json"]).resolve()),
        "output_programs_json": str(Path(teachers["output_programs_json"]).resolve()),
        "use_cell2location": bool(teachers.get("use_cell2location", False)),
        "use_scvi_label_transfer": bool(teachers.get("use_scvi_label_transfer", False)),
        "dry_run": dry_run,
    }
    if dry_run:
        outputs["num_slides"] = int(len(slides))
        return outputs

    reference = ad.read_h5ad(teachers["reference_h5ad_path"])
    ref_genes, ref_signatures = _build_reference_signatures(reference, tasks["composition_classes"])

    obs_rows: list[pd.DataFrame] = []
    composition_blocks: list[np.ndarray] = []
    program_blocks: list[np.ndarray] = []
    for row in slides.to_dict(orient="records"):
        adata = ad.read_h5ad(row["adata_path"])
        source_name = row.get("source", row.get("name", row["slide_id"]))
        has_pathology = bool(
            row["source_type"] == "visium"
            and teachers.get("visium_compartment_annotation_column", "Classification") in adata.obs.columns
        )
        compartment = _infer_compartment(
            adata.obs,
            tasks["compartment_classes"],
            teachers.get("visium_compartment_annotation_column", "Classification"),
        )
        coords = pd.DataFrame(
            {
                "slide_id": row["slide_id"],
                "patient_id": row["patient_id"],
                "source": source_name,
                "source_type": row["source_type"],
                "platform": row["platform"],
                "tile_id": [f"{row['slide_id']}__tile_{idx:06d}" for idx in range(adata.n_obs)],
                "x_um": adata.obs.get("x_um", pd.Series(np.arange(adata.n_obs) * 112.0)).astype(float).to_numpy(),
                "y_um": adata.obs.get("y_um", pd.Series(np.zeros(adata.n_obs))).astype(float).to_numpy(),
                "compartment": compartment.to_numpy(),
                "teacher_mask_compartment": int(has_pathology),
                "teacher_mask_composition": 1,
                "teacher_mask_program": 1,
                "teacher_confidence_compartment": teachers["teacher_confidence"]["pathology"] if has_pathology else 0.0,
                "teacher_confidence_composition": teachers["teacher_confidence"]["xenium"] if row["source_type"] == "xenium" else teachers["teacher_confidence"]["visium"],
                "teacher_confidence_program": teachers["teacher_confidence"]["fallback"],
                "teacher_source_compartment": "pathology" if has_pathology else "unlabeled",
                "teacher_source_composition": "xenium" if row["source_type"] == "xenium" else "signature_scoring",
                "teacher_source_program": teachers.get("program_method", "gene_set_mean"),
            }
        )
        obs_rows.append(coords)
        composition_blocks.append(_composition_from_reference(adata, ref_genes, ref_signatures, tasks["composition_classes"]))
        program_blocks.append(_program_scores(adata, tasks["programs"]))

    obs = pd.concat(obs_rows, ignore_index=True)
    teacher = ad.AnnData(np.zeros((len(obs), 1), dtype=np.float32), obs=obs)
    teacher.obsm["composition"] = np.vstack(composition_blocks).astype(np.float32)
    teacher.obsm["programs"] = np.vstack(program_blocks).astype(np.float32)
    teacher.uns["composition_classes"] = tasks["composition_classes"]
    teacher.uns["programs"] = tasks["programs"]

    output_h5ad = Path(teachers["output_h5ad_path"])
    ensure_parent(output_h5ad)
    teacher.write_h5ad(output_h5ad)
    write_json(
        teachers["output_ontology_json"],
        {
            "compartment_classes": tasks["compartment_classes"],
            "composition_classes": tasks["composition_classes"],
        },
    )
    write_json(
        teachers["output_programs_json"],
        {
            "programs": tasks["programs"],
            "method": teachers.get("program_method", "gene_set_mean"),
        },
    )
    outputs["n_tiles"] = int(teacher.n_obs)
    write_json(output_h5ad.with_suffix(".summary.json"), outputs)
    return outputs
