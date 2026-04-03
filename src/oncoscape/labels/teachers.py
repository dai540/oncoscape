from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

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

PROGRAM_GENE_SETS = {
    "proliferation": ["MKI67", "TOP2A", "PCNA", "TYMS", "UBE2C", "BIRC5"],
    "EMT": ["VIM", "FN1", "COL1A1", "COL1A2", "SNAI1", "TAGLN"],
    "hypoxia": ["CA9", "VEGFA", "SLC2A1", "LDHA", "ENO1", "BNIP3"],
    "IFN_gamma": ["STAT1", "CXCL9", "CXCL10", "IFIT3", "IRF1", "GBP1"],
    "inflammatory": ["IL6", "CXCL8", "NFKBIA", "CCL2", "PTGS2", "TNFAIP3"],
    "angiogenesis": ["PECAM1", "VWF", "KDR", "EMCN", "FLT1", "ENG"],
    "TGFb_like": ["TGFB1", "TGFBI", "SERPINE1", "COL3A1", "THBS1", "ACTA2"],
    "antigen_presentation": ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "PSMB8"],
}


def _to_dense(x: Any) -> np.ndarray:
    return np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=np.float32)


def _build_reference_signatures(reference: ad.AnnData, broad_cell_types: list[str]) -> tuple[list[str], np.ndarray]:
    signatures = []
    for label in broad_cell_types:
        mask = (reference.obs["broad_cell_type"].astype(str) == label).to_numpy()
        if mask.sum() == 0:
            signatures.append(np.zeros(reference.n_vars, dtype=np.float32))
        else:
            class_mean = np.asarray(reference.X[mask].mean(axis=0)).ravel().astype(np.float32)
            signatures.append(np.log1p(class_mean))
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
    if adata.n_vars == 0 or adata.n_obs == 0:
        return np.zeros((adata.n_obs, len(program_names)), dtype=np.float32)
    var_lookup = {str(gene).upper(): idx for idx, gene in enumerate(adata.var_names)}
    fallback_blocks = np.array_split(np.arange(adata.n_vars), len(program_names))
    programs = []
    for name, fallback in zip(program_names, fallback_blocks):
        genes = [gene for gene in PROGRAM_GENE_SETS.get(name, []) if gene in var_lookup]
        if not genes:
            idxs = np.asarray(fallback, dtype=int)
            programs.append(matrix[:, idxs].mean(axis=1).astype(np.float32) if len(idxs) else np.zeros(adata.n_obs, dtype=np.float32))
        else:
            idxs = [var_lookup[gene] for gene in genes]
            programs.append(matrix[:, idxs].mean(axis=1).astype(np.float32))
    return np.vstack(programs).T


def _subsample_reference_for_cell2location(reference: ad.AnnData, broad_cell_types: list[str], max_cells: int) -> ad.AnnData:
    if max_cells <= 0 or reference.n_obs <= max_cells:
        return reference.copy()
    keep = []
    per_class = max(1, max_cells // max(len(broad_cell_types), 1))
    labels = reference.obs["broad_cell_type"].astype(str)
    for label in broad_cell_types:
        idx = np.flatnonzero((labels == label).to_numpy())
        if len(idx) == 0:
            continue
        chosen = idx if len(idx) <= per_class else idx[np.linspace(0, len(idx) - 1, per_class, dtype=int)]
        keep.extend(chosen.tolist())
    if len(keep) < max_cells:
        remainder = [idx for idx in range(reference.n_obs) if idx not in set(keep)]
        extra = remainder[: max(0, max_cells - len(keep))]
        keep.extend(extra)
    keep = np.unique(np.asarray(keep, dtype=int))
    return reference[keep].copy()


def _shared_gene_subset(reference: ad.AnnData, query: ad.AnnData, max_genes: int) -> tuple[ad.AnnData, ad.AnnData]:
    shared = sorted(set(map(str, reference.var_names)) & set(map(str, query.var_names)))
    if not shared:
        raise ValueError("no shared genes between reference and query")
    if max_genes > 0 and len(shared) > max_genes:
        ref_idx = [list(map(str, reference.var_names)).index(gene) for gene in shared]
        mean_expr = np.asarray(reference[:, ref_idx].X.mean(axis=0)).ravel()
        top = np.argsort(mean_expr)[::-1][:max_genes]
        shared = [shared[idx] for idx in top]
    return reference[:, shared].copy(), query[:, shared].copy()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.clip(matrix, 0.0, None)
    return matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-8, None)


def _blend_composition(primary: np.ndarray, secondary: np.ndarray, secondary_weight: float) -> np.ndarray:
    weight = float(np.clip(secondary_weight, 0.0, 1.0))
    if weight <= 0.0:
        return _normalize_rows(primary)
    return _normalize_rows((1.0 - weight) * primary + weight * secondary)


def _canonicalize_factor_columns(frame: pd.DataFrame, broad_cell_types: list[str]) -> pd.DataFrame:
    rename_map = {}
    for column in frame.columns:
        name = str(column)
        canonical = name
        for prefix in [
            "means_per_cluster_mu_fg_",
            "cell_abundance_w_sf_",
            "means_cell_abundance_w_sf_",
            "meanscell_abundance_w_sf_",
        ]:
            if canonical.startswith(prefix):
                canonical = canonical[len(prefix) :]
        rename_map[column] = canonical
    out = frame.rename(columns=rename_map)
    return out.reindex(columns=broad_cell_types, fill_value=0.0)


def _build_cell2location_signatures(
    reference: ad.AnnData,
    broad_cell_types: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame | None, str]:
    try:
        import scvi
        from cell2location.models import RegressionModel
    except Exception as exc:  # pragma: no cover - optional dependency path
        return None, f"cell2location_unavailable:{exc}"

    scvi.settings.seed = int(config.get("seed", 17))
    work = _subsample_reference_for_cell2location(
        reference,
        broad_cell_types,
        int(config.get("max_reference_cells", 20000)),
    )
    work.obs["broad_cell_type"] = work.obs["broad_cell_type"].astype(str)
    RegressionModel.setup_anndata(work, labels_key="broad_cell_type")
    model = RegressionModel(work)
    train_kwargs = {
        "max_epochs": int(config.get("regression_max_epochs", config.get("max_epochs", 250))),
        "batch_size": int(config.get("batch_size", 2048)),
        "train_size": float(config.get("regression_train_size", 1.0)),
        "lr": float(config.get("lr", 0.002)),
    }
    model.train(**train_kwargs)
    posterior_kwargs = {
        "num_samples": int(config.get("posterior_samples", 64)),
        "batch_size": int(config.get("batch_size", 2048)),
    }
    work = model.export_posterior(work, sample_kwargs=posterior_kwargs)
    signature_df = work.varm["means_per_cluster_mu_fg"]
    if not isinstance(signature_df, pd.DataFrame):
        signature_df = pd.DataFrame(signature_df, index=work.var_names)
    signature_df = signature_df.reindex(index=work.var_names)
    signature_df = _canonicalize_factor_columns(signature_df, broad_cell_types)
    return signature_df, "cell2location"


def _composition_from_cell2location(
    adata: ad.AnnData,
    signature_df: pd.DataFrame,
    broad_cell_types: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, str]:
    try:
        import scvi
        from cell2location.models import Cell2location
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(f"cell2location unavailable: {exc}") from exc

    scvi.settings.seed = int(config.get("seed", 17))
    shared = [gene for gene in map(str, adata.var_names) if gene in signature_df.index]
    if not shared:
        raise ValueError("cell2location: no shared genes with signatures")
    spatial = adata[:, shared].copy()
    cell_state_df = signature_df.loc[shared, broad_cell_types].copy()
    Cell2location.setup_anndata(spatial)
    model = Cell2location(
        spatial,
        cell_state_df=cell_state_df,
        N_cells_per_location=int(config.get("n_cells_per_location", 8)),
        detection_alpha=float(config.get("detection_alpha", 20.0)),
    )
    train_kwargs = {
        "max_epochs": int(config.get("max_epochs", 250)),
        "batch_size": int(min(int(config.get("batch_size", 1024)), max(spatial.n_obs, 1))),
        "train_size": float(config.get("train_size", 1.0)),
        "lr": float(config.get("lr", 0.002)),
    }
    model.train(**train_kwargs)
    posterior_kwargs = {
        "num_samples": int(config.get("posterior_samples", 64)),
        "batch_size": int(min(int(config.get("batch_size", 1024)), max(spatial.n_obs, 1))),
    }
    spatial = model.export_posterior(spatial, sample_kwargs=posterior_kwargs)
    abundance = spatial.obsm["means_cell_abundance_w_sf"]
    if isinstance(abundance, pd.DataFrame):
        abundance_df = abundance
    else:
        abundance_df = pd.DataFrame(abundance, index=spatial.obs_names)
    abundance_df = _canonicalize_factor_columns(abundance_df, broad_cell_types)
    return _normalize_rows(abundance_df.to_numpy(dtype=np.float32)), "cell2location"


def _composition_from_scvi_label_transfer(
    query: ad.AnnData,
    reference: ad.AnnData,
    broad_cell_types: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, str]:
    try:
        import torch
        import scvi
        from scvi.model import SCVI
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(f"scvi unavailable: {exc}") from exc

    scvi.settings.seed = int(config.get("seed", 17))
    ref_aligned, query_aligned = _shared_gene_subset(reference, query, int(config.get("max_shared_genes", 4000)))
    ref_aligned.obs["_dataset"] = "reference"
    query_aligned.obs["_dataset"] = "query"
    combined = ad.concat([ref_aligned, query_aligned], join="inner")
    SCVI.setup_anndata(combined, batch_key="_dataset")
    model = SCVI(
        combined,
        n_latent=int(config.get("n_latent", 32)),
        n_layers=int(config.get("n_layers", 2)),
        n_hidden=int(config.get("n_hidden", 128)),
        dropout_rate=float(config.get("dropout_rate", 0.1)),
        gene_likelihood=str(config.get("gene_likelihood", "nb")),
    )
    accelerator = "gpu" if torch.cuda.is_available() and bool(config.get("use_gpu", False)) else "cpu"
    model.train(
        max_epochs=int(config.get("max_epochs", 30)),
        batch_size=int(config.get("batch_size", 1024)),
        train_size=float(config.get("train_size", 0.9)),
        validation_size=float(config.get("validation_size", 0.1)),
        accelerator=accelerator,
        devices=1,
        early_stopping=bool(config.get("early_stopping", True)),
        enable_progress_bar=bool(config.get("enable_progress_bar", False)),
        logger=False,
        enable_checkpointing=False,
    )
    latent = model.get_latent_representation(combined).astype(np.float32)
    ref_n = ref_aligned.n_obs
    ref_latent = latent[:ref_n]
    query_latent = latent[ref_n:]
    ref_labels = ref_aligned.obs["broad_cell_type"].astype(str).to_numpy()
    centroids = []
    for label in broad_cell_types:
        mask = ref_labels == label
        centroids.append(ref_latent[mask].mean(axis=0) if mask.any() else np.zeros(ref_latent.shape[1], dtype=np.float32))
    centroid_matrix = np.vstack(centroids).astype(np.float32)
    denom = np.linalg.norm(query_latent, axis=1, keepdims=True) * np.linalg.norm(centroid_matrix, axis=1, keepdims=True).T
    scores = query_latent @ centroid_matrix.T / np.clip(denom, 1e-8, None)
    return _softmax(scores.astype(np.float32)), "scvi_label_transfer"


def build_teachers(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    teachers = config["teachers"]
    tasks = config["tasks"]
    outputs = {
        "output_h5ad_path": str(Path(teachers["output_h5ad_path"]).resolve()),
        "output_ontology_json": str(Path(teachers["output_ontology_json"]).resolve()),
        "output_programs_json": str(Path(teachers["output_programs_json"]).resolve()),
        "use_cell2location": bool(teachers.get("use_cell2location", False)),
        "use_scvi_label_transfer": bool(teachers.get("use_scvi_label_transfer", False)),
        "dry_run": dry_run,
    }
    if dry_run:
        outputs["expected_slide_registry"] = str(Path(teachers["slides_csv_path"]).resolve())
        outputs["reference_h5ad_path"] = str(Path(teachers["reference_h5ad_path"]).resolve())
        return outputs

    slides = pd.read_csv(teachers["slides_csv_path"])
    slides = slides[slides["source_type"].isin(["visium", "xenium"])].reset_index(drop=True)

    reference = ad.read_h5ad(teachers["reference_h5ad_path"])
    ref_genes, ref_signatures = _build_reference_signatures(reference, tasks["composition_classes"])
    methods_used = {"composition": set(), "programs": set([teachers.get("program_method", "gene_set_mean")])}

    signature_df = None
    if bool(teachers.get("use_cell2location", False)):
        try:
            signature_df, method_name = _build_cell2location_signatures(
                reference,
                tasks["composition_classes"],
                dict(teachers.get("cell2location", {})) | {"seed": int(config.get("runtime", {}).get("seed", 17))},
            )
            if signature_df is not None:
                methods_used["composition"].add(method_name)
        except Exception:
            signature_df = None

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
                "teacher_source_composition": "",
                "teacher_source_program": teachers.get("program_method", "gene_set_mean"),
            }
        )
        composition_method = "signature_scoring"
        if row["source_type"] == "visium" and signature_df is not None:
            try:
                comp, composition_method = _composition_from_cell2location(
                    adata,
                    signature_df,
                    tasks["composition_classes"],
                    dict(teachers.get("cell2location", {})) | {"seed": int(config.get("runtime", {}).get("seed", 17))},
                )
            except Exception:
                comp = _composition_from_reference(adata, ref_genes, ref_signatures, tasks["composition_classes"])
                composition_method = "signature_scoring"
        elif row["source_type"] == "xenium" and bool(teachers.get("use_scvi_label_transfer", False)):
            try:
                comp, composition_method = _composition_from_scvi_label_transfer(
                    adata,
                    reference,
                    tasks["composition_classes"],
                    dict(teachers.get("scvi", {})) | {"seed": int(config.get("runtime", {}).get("seed", 17))},
                )
            except Exception:
                comp = _composition_from_reference(adata, ref_genes, ref_signatures, tasks["composition_classes"])
                composition_method = "signature_scoring"
        else:
            comp = _composition_from_reference(adata, ref_genes, ref_signatures, tasks["composition_classes"])
        coords["teacher_source_composition"] = composition_method
        methods_used["composition"].add(composition_method)
        obs_rows.append(coords)
        if composition_method == "cell2location":
            fallback_comp = _composition_from_reference(adata, ref_genes, ref_signatures, tasks["composition_classes"])
            comp = _blend_composition(
                comp,
                fallback_comp,
                float(teachers.get("cell2location", {}).get("blend_signature_weight", 0.25)),
            )
        composition_blocks.append(comp)
        program_blocks.append(_program_scores(adata, tasks["programs"]))

    obs = pd.concat(obs_rows, ignore_index=True)
    teacher = ad.AnnData(np.zeros((len(obs), 1), dtype=np.float32), obs=obs)
    teacher.obsm["composition"] = np.vstack(composition_blocks).astype(np.float32)
    teacher.obsm["programs"] = np.vstack(program_blocks).astype(np.float32)
    teacher.uns["composition_classes"] = tasks["composition_classes"]
    teacher.uns["programs"] = tasks["programs"]
    teacher.uns["methods_used"] = {key: sorted(values) for key, values in methods_used.items()}

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
    outputs["methods_used"] = {key: sorted(values) for key, values in methods_used.items()}
    write_json(output_h5ad.with_suffix(".summary.json"), outputs)
    return outputs
