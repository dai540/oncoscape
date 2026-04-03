from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from oncoscape.core import ensure_parent, write_json


CELL_TYPE_MARKERS = {
    "malignant": ["EPCAM", "KRT8", "KRT18", "ERBB2"],
    "CAF": ["COL1A1", "COL1A2", "DCN", "LUM"],
    "endothelial": ["PECAM1", "VWF", "KDR", "EMCN"],
    "myeloid": ["LYZ", "FCER1G", "CTSS", "TYROBP"],
    "T_NK": ["NKG7", "TRBC1", "CD3D", "CD3E"],
    "B_plasma": ["MS4A1", "CD79A", "MZB1", "JCHAIN"],
}

BROAD_CLASS_KEYWORDS = {
    "malignant": [
        "malignant",
        "tumor",
        "tumour",
        "cancer",
        "carcinoma",
        "epithelial",
        "luminal",
        "basal",
        "myoepithelial",
        "her2",
        "triple negative",
        "tnbc",
        "pvl",
    ],
    "CAF": ["caf", "fibro", "stroma", "stromal", "myofibro", "pericyte"],
    "endothelial": ["endothelial", "vascular", "vessel", "lymphatic"],
    "myeloid": ["myeloid", "macrophage", "monocyte", "dc", "dendritic", "neutroph", "mast"],
    "T_NK": ["t-cell", "t cells", "t-cells", "cd4", "cd8", "nk", "nkt", "lymphocyte"],
    "B_plasma": ["b-cell", "b cells", "b-cells", "plasma", "plasmablast"],
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


def _to_dense(x: Any) -> np.ndarray:
    return np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=np.float32)


def _gene_subset_for_latent(adata: ad.AnnData, max_genes: int) -> ad.AnnData:
    if max_genes <= 0 or adata.n_vars <= max_genes:
        return adata.copy()
    matrix = adata.X
    if sparse.issparse(matrix):
        gene_means = np.asarray(matrix.mean(axis=0)).ravel()
        squared = matrix.copy()
        squared.data **= 2
        gene_sq_means = np.asarray(squared.mean(axis=0)).ravel()
        gene_vars = np.maximum(gene_sq_means - gene_means**2, 0.0)
    else:
        dense = np.asarray(matrix, dtype=np.float32)
        gene_vars = dense.var(axis=0)
    top_idx = np.argsort(gene_vars)[::-1][:max_genes]
    return adata[:, np.sort(top_idx)].copy()


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


def _compute_scvi_latent(adata: ad.AnnData, ref_cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
        import scvi
        from scvi.model import SCVI
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(f"scvi import failed: {exc}") from exc

    scvi_cfg = ref_cfg.get("scvi", {})
    scvi.settings.seed = int(scvi_cfg.get("seed", ref_cfg.get("_runtime_seed", 17)))
    working = adata.copy()
    batch_key = "source_atlas" if "source_atlas" in working.obs.columns else None
    SCVI.setup_anndata(working, batch_key=batch_key)
    model = SCVI(
        working,
        n_latent=int(scvi_cfg.get("n_latent", ref_cfg.get("latent_dim", 32))),
        n_layers=int(scvi_cfg.get("n_layers", 2)),
        n_hidden=int(scvi_cfg.get("n_hidden", 128)),
        dropout_rate=float(scvi_cfg.get("dropout_rate", 0.1)),
        gene_likelihood=str(scvi_cfg.get("gene_likelihood", "nb")),
    )
    accelerator = str(scvi_cfg.get("accelerator", "gpu" if torch.cuda.is_available() else "cpu"))
    devices = scvi_cfg.get("devices", 1 if accelerator != "gpu" else "auto")
    train_kwargs = {
        "max_epochs": int(scvi_cfg.get("max_epochs", 30)),
        "accelerator": accelerator,
        "devices": devices,
        "batch_size": int(scvi_cfg.get("batch_size", 1024)),
        "train_size": float(scvi_cfg.get("train_size", 0.9)),
        "validation_size": float(scvi_cfg.get("validation_size", 0.1)),
        "early_stopping": bool(scvi_cfg.get("early_stopping", True)),
        "check_val_every_n_epoch": int(scvi_cfg.get("check_val_every_n_epoch", 5)),
        "enable_checkpointing": False,
        "logger": False,
        "enable_progress_bar": bool(scvi_cfg.get("enable_progress_bar", False)),
    }
    lr = scvi_cfg.get("lr")
    if lr is not None:
        train_kwargs["plan_kwargs"] = {"lr": float(lr)}
    model.train(**train_kwargs)
    latent = model.get_latent_representation(working).astype(np.float32)
    return latent, {
        "latent_method": "scvi",
        "latent_dim": int(latent.shape[1]) if latent.ndim == 2 else 0,
        "accelerator": accelerator,
    }


def _map_label_to_broad_class(value: str, broad_cell_types: list[str]) -> str | None:
    lowered = value.lower()
    for label in broad_cell_types:
        if label.lower() in lowered:
            return label
    for label in broad_cell_types:
        if any(keyword in lowered for keyword in BROAD_CLASS_KEYWORDS.get(label, [])):
            return label
    return None


def _marker_score_labels(adata: ad.AnnData, broad_cell_types: list[str]) -> list[str]:
    var_lookup = {str(gene).upper(): idx for idx, gene in enumerate(adata.var_names)}
    marker_genes = sorted({marker for label in broad_cell_types for marker in CELL_TYPE_MARKERS.get(label, []) if marker in var_lookup})
    if not marker_genes:
        return [broad_cell_types[0]] * adata.n_obs
    marker_idx = [var_lookup[marker] for marker in marker_genes]
    marker_matrix = np.log1p(_to_dense(adata[:, marker_idx].X))
    local_lookup = {gene: idx for idx, gene in enumerate(marker_genes)}
    labels = []
    for row in marker_matrix:
        scores = []
        for label in broad_cell_types:
            markers = [marker for marker in CELL_TYPE_MARKERS.get(label, []) if marker in local_lookup]
            if not markers:
                scores.append(-1e6)
            else:
                idx = [local_lookup[m] for m in markers]
                scores.append(float(row[idx].mean()))
        labels.append(broad_cell_types[int(np.argmax(scores))])
    return labels


def _assign_broad_labels(adata: ad.AnnData, broad_cell_types: list[str], label_column: str | None) -> list[str]:
    marker_labels = _marker_score_labels(adata, broad_cell_types)
    if label_column is None:
        return marker_labels
    values = adata.obs[label_column].astype(str).tolist()
    assigned = []
    for idx, value in enumerate(values):
        mapped = _map_label_to_broad_class(value, broad_cell_types)
        assigned.append(mapped if mapped is not None else marker_labels[idx])
    return assigned


def _compute_marker_table(adata: ad.AnnData, broad_cell_types: list[str], marker_top_k: int) -> list[dict[str, Any]]:
    markers = []
    matrix = adata.X
    for label in broad_cell_types:
        mask = (adata.obs["broad_cell_type"].astype(str) == label).to_numpy()
        if not mask.any():
            continue
        class_block = matrix[mask]
        class_mean = np.asarray(class_block.mean(axis=0)).ravel()
        top_idx = np.argsort(class_mean)[::-1][:marker_top_k]
        for rank, idx in enumerate(top_idx, start=1):
            markers.append(
                {
                    "group": label,
                    "rank": rank,
                    "gene": str(adata.var_names[idx]),
                    "score": float(class_mean[idx]),
                }
            )
    return markers


def build_reference_atlas(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    ref = config["reference"]
    ref = dict(ref)
    ref["_runtime_seed"] = int(config.get("runtime", {}).get("seed", 17))
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
    broad_classes = list(ref["broad_cell_types"])
    merged.obs["broad_cell_type"] = _assign_broad_labels(merged, broad_classes, label_column)

    latent_input = _gene_subset_for_latent(merged, int(ref.get("max_genes_for_latent", 4000)))
    latent_summary = {"latent_method": "svd_fallback", "latent_dim": int(ref.get("latent_dim", 32))}
    if bool(ref.get("use_scvi", False)):
        try:
            latent, latent_summary = _compute_scvi_latent(latent_input, ref)
        except Exception:
            matrix = np.log1p(_to_dense(latent_input.X))
            latent = _compute_latent(matrix, int(ref.get("latent_dim", 32)))
            latent_summary = {"latent_method": "svd_fallback", "latent_dim": int(latent.shape[1]) if latent.ndim == 2 else 0}
    else:
        matrix = np.log1p(_to_dense(latent_input.X))
        latent = _compute_latent(matrix, int(ref.get("latent_dim", 32)))
    merged.obsm["X_scvi"] = latent.astype(np.float32)
    merged.uns["latent_method"] = latent_summary["latent_method"]

    markers = _compute_marker_table(merged, broad_classes, int(ref.get("marker_top_k", 20)))

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
            **latent_summary,
        }
    )
    write_json(output_h5ad.with_suffix(".summary.json"), outputs)
    return outputs
