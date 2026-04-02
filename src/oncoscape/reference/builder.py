from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from oncoscape.core import ensure_parent, write_json


CELL_TYPE_MARKERS = {
    "malignant": ["EPCAM", "KRT8", "KRT18", "ERBB2"],
    "CAF": ["COL1A1", "COL1A2", "DCN", "LUM"],
    "endothelial": ["PECAM1", "VWF", "KDR", "EMCN"],
    "myeloid": ["LYZ", "FCER1G", "CTSS", "TYROBP"],
    "T_NK": ["NKG7", "TRBC1", "CD3D", "CD3E"],
    "B_plasma": ["MS4A1", "CD79A", "MZB1", "JCHAIN"],
}


def _resolve_label_column(obs: pd.DataFrame, candidates: list[str]) -> str | None:
    obs_columns = {col.lower(): col for col in obs.columns}
    for candidate in candidates:
        if candidate.lower() in obs_columns:
            return obs_columns[candidate.lower()]
    return None


def _load_reference_inputs(paths: list[str], max_obs_per_input: int) -> list[ad.AnnData]:
    adatas: list[ad.AnnData] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        adata = ad.read_h5ad(path)
        if adata.n_obs > max_obs_per_input:
            adata = adata[:max_obs_per_input].copy()
        adatas.append(adata)
    return adatas


def _resolve_reference_input_paths(config: dict[str, Any]) -> list[str]:
    ref = config["reference"]
    explicit = [str(path) for path in ref.get("input_h5ad_paths", []) if str(path).strip()]
    if explicit:
        return explicit
    slides_csv_path = ref.get("slides_csv_path") or config.get("registration", {}).get("slides_csv_path") or str(
        Path(config["registration"]["output_dir"]) / "slides.csv"
    )
    slides_csv = Path(slides_csv_path)
    if not slides_csv.exists():
        return []
    slides = pd.read_csv(slides_csv)
    scrna = slides[slides["source_type"] == "scrna"].copy()
    return [str(path) for path in scrna["adata_path"].dropna().astype(str).tolist()]


def _intersect_and_concat(adatas: list[ad.AnnData]) -> ad.AnnData:
    if not adatas:
        raise ValueError("no readable scRNA inputs found for reference atlas")
    common = set(map(str, adatas[0].var_names))
    for adata in adatas[1:]:
        common &= set(map(str, adata.var_names))
    genes = sorted(common) if common else sorted(map(str, adatas[0].var_names))
    trimmed = [adata[:, genes].copy() if genes else adata.copy() for adata in adatas]
    return ad.concat(trimmed, join="inner", label="source_atlas", keys=[f"atlas_{i}" for i in range(len(trimmed))])


def _compute_latent(matrix: np.ndarray, latent_dim: int) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0, latent_dim), dtype=np.float32)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    k = int(min(latent_dim, centered.shape[0], centered.shape[1]))
    if k == 0:
        return np.zeros((centered.shape[0], latent_dim), dtype=np.float32)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    latent = u[:, :k] * s[:k]
    if k < latent_dim:
        latent = np.pad(latent, ((0, 0), (0, latent_dim - k)))
    return latent.astype(np.float32)


def _marker_score_labels(adata: ad.AnnData, broad_cell_types: list[str]) -> list[str]:
    var_lookup = {str(gene).upper(): idx for idx, gene in enumerate(adata.var_names)}
    matrix = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X, dtype=np.float32)
    matrix = np.log1p(matrix)
    labels = []
    for row in matrix:
        scores = []
        for label in broad_cell_types:
            markers = [marker for marker in CELL_TYPE_MARKERS.get(label, []) if marker in var_lookup]
            if not markers:
                scores.append(-1e6)
            else:
                idx = [var_lookup[m] for m in markers]
                scores.append(float(row[idx].mean()))
        labels.append(broad_cell_types[int(np.argmax(scores))])
    return labels


def build_reference_atlas(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    ref = config["reference"]
    input_paths = _resolve_reference_input_paths(config)
    outputs = {
        "output_h5ad_path": str(Path(ref["output_h5ad_path"]).resolve()),
        "output_markers_path": str(Path(ref["output_markers_path"]).resolve()),
        "output_qc_report_path": str(Path(ref["output_qc_report_path"]).resolve()),
        "num_inputs": int(len(input_paths)),
        "dry_run": dry_run,
    }
    if dry_run:
        return outputs

    adatas = _load_reference_inputs(input_paths, int(ref.get("max_obs_per_input", 50000)))
    merged = _intersect_and_concat(adatas)
    merged.obs_names_make_unique()
    label_column = _resolve_label_column(merged.obs, list(ref.get("label_candidates", [])))
    if label_column is None:
        merged.obs["broad_cell_type"] = _marker_score_labels(merged, ref["broad_cell_types"])
    else:
        values = merged.obs[label_column].astype(str)
        broad_classes = ref["broad_cell_types"]
        assigned = []
        for value in values:
            lowered = value.lower()
            mapped = next((label for label in broad_classes if label.lower() in lowered), broad_classes[0])
            assigned.append(mapped)
        unresolved = [label == broad_classes[0] and broad_classes[0].lower() not in str(v).lower() for label, v in zip(assigned, values)]
        marker_labels = _marker_score_labels(merged, broad_classes)
        merged.obs["broad_cell_type"] = [marker_labels[i] if unresolved[i] else assigned[i] for i in range(len(assigned))]

    matrix = np.asarray(merged.X.todense() if hasattr(merged.X, "todense") else merged.X, dtype=np.float32)
    matrix = np.log1p(matrix)
    merged.obsm["X_scvi"] = _compute_latent(matrix, int(ref.get("latent_dim", 32)))

    markers = []
    gene_frame = pd.DataFrame(matrix, columns=merged.var_names)
    for label in ref["broad_cell_types"]:
        mask = merged.obs["broad_cell_type"] == label
        if mask.sum() == 0:
            continue
        class_mean = gene_frame.loc[mask.to_numpy()].mean(axis=0)
        top = class_mean.sort_values(ascending=False).head(int(ref.get("marker_top_k", 20)))
        for rank, (gene, score) in enumerate(top.items(), start=1):
            markers.append({"group": label, "rank": rank, "gene": gene, "score": float(score)})

    output_h5ad = Path(ref["output_h5ad_path"])
    ensure_parent(output_h5ad)
    merged.write_h5ad(output_h5ad)
    pd.DataFrame(markers).to_csv(ref["output_markers_path"], index=False)
    ensure_parent(ref["output_qc_report_path"]).write_text(
        (
            "<html><body><h1>oncoscape reference QC</h1>"
            f"<p>cells={merged.n_obs}</p><p>genes={merged.n_vars}</p>"
            f"<p>broad_cell_types={merged.obs['broad_cell_type'].nunique()}</p></body></html>"
        ),
        encoding="utf-8",
    )

    outputs.update(
        {
            "num_loaded_inputs": len(adatas),
            "n_obs": int(merged.n_obs),
            "n_vars": int(merged.n_vars),
            "latent_key": "X_scvi",
        }
    )
    write_json(output_h5ad.with_suffix(".summary.json"), outputs)
    return outputs
