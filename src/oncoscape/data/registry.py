from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import sparse

from oncoscape.core import ensure_directory, read_table, write_json
from .manifest import SourceManifestEntry, load_manifest
from .splits import assign_deterministic_split, build_split_tables


def _decode(values: Any) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _load_10x_h5(path: Path) -> ad.AnnData:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        data = np.asarray(matrix["data"])
        indices = np.asarray(matrix["indices"])
        indptr = np.asarray(matrix["indptr"])
        shape = tuple(np.asarray(matrix["shape"]).tolist())
        barcodes = _decode(matrix["barcodes"][()])
        features_group = matrix["features"]
        if "name" in features_group:
            genes = _decode(features_group["name"][()])
        elif "gene_names" in features_group:
            genes = _decode(features_group["gene_names"][()])
        elif "_all_tag_keys" in features_group:
            genes = _decode(features_group["id"][()])
        else:
            genes = _decode(features_group["id"][()])
    matrix_csc = sparse.csc_matrix((data, indices, indptr), shape=shape)
    adata = ad.AnnData(matrix_csc.T.tocsr())
    adata.obs_names = pd.Index(barcodes)
    adata.var_names = pd.Index(genes)
    return adata


def _find_existing(path: Path, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        full = path / candidate
        if full.exists():
            return full
    return None


def _load_mtx_directory(path: Path) -> ad.AnnData:
    matrix_path = _find_existing(path, ["matrix.mtx", "matrix.mtx.gz", "count_matrix_sparse.mtx"])
    features_path = _find_existing(path, ["features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz", "count_matrix_genes.tsv"])
    barcodes_path = _find_existing(path, ["barcodes.tsv", "barcodes.tsv.gz", "count_matrix_barcodes.tsv"])
    if matrix_path is None or features_path is None or barcodes_path is None:
        raise FileNotFoundError(f"could not locate 10x/mtx trio under {path}")
    matrix = spio.mmread(matrix_path).tocsr().T
    features = pd.read_csv(features_path, sep="\t", header=None, compression="gzip" if features_path.name.endswith(".gz") else None)
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None, compression="gzip" if barcodes_path.name.endswith(".gz") else None)
    feature_names = features.iloc[:, 1] if features.shape[1] >= 2 else features.iloc[:, 0]
    barcode_names = barcodes.iloc[:, 0]
    adata = ad.AnnData(matrix)
    adata.obs_names = pd.Index(barcode_names.astype(str).tolist())
    adata.var_names = pd.Index(feature_names.astype(str).tolist())
    return adata


def _load_counts_matrix(entry: SourceManifestEntry) -> ad.AnnData:
    counts_path = Path(entry.counts_path)
    if counts_path.is_dir():
        return _load_mtx_directory(counts_path)
    suffix = counts_path.suffix.lower()
    name = counts_path.name.lower()
    if suffix == ".h5ad" and counts_path.exists():
        return ad.read_h5ad(counts_path)
    if suffix in {".h5", ".hdf5"} and counts_path.exists():
        return _load_10x_h5(counts_path)
    if suffix in {".csv", ".tsv", ".txt", ".parquet"} or name.endswith((".csv.gz", ".tsv.gz", ".txt.gz")):
        frame = read_table(counts_path)
        if frame.empty:
            frame = pd.DataFrame([[0.0]], columns=["placeholder_gene"])
        if "cell_id" in frame.columns:
            frame = frame.set_index("cell_id")
        adata = ad.AnnData(frame.astype(float))
        return adata
    adata = ad.AnnData(np.zeros((1, 1), dtype=np.float32))
    adata.var_names = ["placeholder_gene"]
    adata.obs_names = [f"{entry.slide_id}_obs_0"]
    return adata


def _normalize_visium_positions(frame: pd.DataFrame) -> pd.DataFrame:
    if set(frame.columns) >= {"barcode", "pxl_col_in_fullres", "pxl_row_in_fullres"}:
        return frame.copy()
    if frame.shape[1] >= 6:
        raw = frame.iloc[:, :6].copy()
        raw.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]
        return raw
    raise ValueError("unsupported Visium tissue positions format")


def _read_visium_coordinates(entry: SourceManifestEntry, barcodes: pd.Index) -> pd.DataFrame | None:
    coord_path = Path(entry.coord_path) if entry.coord_path else None
    if coord_path is None or not coord_path.exists():
        return None
    if coord_path.is_dir():
        positions_path = _find_existing(coord_path, ["tissue_positions.csv", "tissue_positions_list.csv", "tissue_positions.parquet"])
        if positions_path is None:
            return None
        if positions_path.suffix.lower() == ".parquet":
            coords = pd.read_parquet(positions_path)
        elif positions_path.name.endswith(".csv"):
            try:
                coords = pd.read_csv(positions_path)
                if "barcode" not in coords.columns and coords.shape[1] >= 6:
                    coords = pd.read_csv(positions_path, header=None)
            except Exception:
                coords = pd.read_csv(positions_path, header=None)
        else:
            coords = read_table(positions_path)
    else:
        coords = read_table(coord_path)
    coords = _normalize_visium_positions(coords)
    coords["barcode"] = coords["barcode"].astype(str)
    coords = coords.drop_duplicates("barcode").set_index("barcode")
    joined = pd.DataFrame(index=barcodes.astype(str))
    joined["x_um"] = coords.reindex(joined.index)["pxl_col_in_fullres"].astype(float).to_numpy() * float(entry.mpp_x or 1.0)
    joined["y_um"] = coords.reindex(joined.index)["pxl_row_in_fullres"].astype(float).to_numpy() * float(entry.mpp_y or 1.0)
    return joined.reset_index(drop=True)


def _read_generic_coordinates(entry: SourceManifestEntry, barcodes: pd.Index) -> pd.DataFrame | None:
    coord_path = Path(entry.coord_path) if entry.coord_path else None
    if coord_path is None or not coord_path.exists():
        return None
    if coord_path.suffix.lower() == ".parquet":
        coords = pd.read_parquet(coord_path)
    else:
        coords = read_table(coord_path)
    lookup_cols = {col.lower(): col for col in coords.columns}
    id_col = next((lookup_cols[key] for key in ["barcode", "cell_id", "obs_id"] if key in lookup_cols), None)
    x_col = next(
        (
            lookup_cols[key]
            for key in ["x_um", "x", "x_centroid", "pxl_col_in_fullres", "x_location"]
            if key in lookup_cols
        ),
        None,
    )
    y_col = next(
        (
            lookup_cols[key]
            for key in ["y_um", "y", "y_centroid", "pxl_row_in_fullres", "y_location"]
            if key in lookup_cols
        ),
        None,
    )
    if x_col is None or y_col is None:
        return None
    if id_col is not None:
        coords = coords.drop_duplicates(id_col).set_index(coords[id_col].astype(str))
        joined = pd.DataFrame(index=barcodes.astype(str))
        x_values = coords.reindex(joined.index)[x_col].astype(float).to_numpy()
        y_values = coords.reindex(joined.index)[y_col].astype(float).to_numpy()
    else:
        coords = coords[[x_col, y_col]].copy().reset_index(drop=True)
        if len(coords) < len(barcodes):
            repeat = int(np.ceil(len(barcodes) / max(len(coords), 1)))
            coords = pd.concat([coords] * repeat, ignore_index=True)
        x_values = coords.iloc[: len(barcodes)][x_col].astype(float).to_numpy()
        y_values = coords.iloc[: len(barcodes)][y_col].astype(float).to_numpy()
    if str(entry.coord_unit).lower().startswith("pixel"):
        x_values = x_values * float(entry.mpp_x or 1.0)
        y_values = y_values * float(entry.mpp_y or 1.0)
    return pd.DataFrame({"x_um": x_values, "y_um": y_values})


def _apply_coordinates(adata: ad.AnnData, entry: SourceManifestEntry) -> None:
    coords = None
    coord_path = Path(entry.coord_path) if entry.coord_path else None
    if entry.source_type == "visium" and coord_path is not None and coord_path.exists() and coord_path.is_dir():
        coords = _read_visium_coordinates(entry, adata.obs_names)
    if coords is None:
        coords = _read_generic_coordinates(entry, adata.obs_names)
    if coords is None:
        coords = pd.DataFrame({"x_um": np.arange(adata.n_obs, dtype=float) * 112.0, "y_um": 0.0})
    adata.obs["x_um"] = coords["x_um"].astype(float).to_numpy()
    adata.obs["y_um"] = coords["y_um"].astype(float).to_numpy()


def _apply_annotations(adata: ad.AnnData, entry: SourceManifestEntry) -> None:
    annotation_path = Path(entry.annotation_path) if entry.annotation_path else None
    if annotation_path is None or not annotation_path.exists():
        return
    annotations = read_table(annotation_path) if annotation_path.suffix.lower() != ".parquet" else pd.read_parquet(annotation_path)
    if annotations.empty:
        return
    id_lookup = {col.lower(): col for col in annotations.columns}
    id_col = next((id_lookup[key] for key in ["barcode", "cell_id", "obs_id"] if key in id_lookup), None)
    if id_col is not None:
        annotations = annotations.drop_duplicates(id_col).set_index(annotations[id_col].astype(str))
        joined = annotations.reindex(adata.obs_names.astype(str))
        for column in joined.columns[: min(joined.shape[1], 16)]:
            adata.obs[column] = joined[column].to_numpy()
        return
    for column in annotations.columns[: min(annotations.shape[1], 16)]:
        values = annotations[column].reset_index(drop=True)
        if len(values) < adata.n_obs:
            repeat = int(np.ceil(adata.n_obs / max(len(values), 1)))
            values = pd.concat([values] * repeat, ignore_index=True)
        adata.obs[column] = values.iloc[: adata.n_obs].to_numpy()


def _normalize_adata(adata: ad.AnnData, entry: SourceManifestEntry) -> ad.AnnData:
    adata = adata.copy()
    if adata.var_names.empty:
        adata.var_names = [f"gene_{idx}" for idx in range(adata.n_vars)]
    if adata.obs_names.empty:
        adata.obs_names = [f"{entry.slide_id}_obs_{idx}" for idx in range(adata.n_obs)]
    adata.obs_names = pd.Index([str(name).strip() or f"{entry.slide_id}_obs_{idx}" for idx, name in enumerate(adata.obs_names)])
    adata.var_names = pd.Index([str(name).strip() or f"gene_{idx}" for idx, name in enumerate(adata.var_names)])
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    adata.obs["slide_id"] = entry.slide_id
    adata.obs["patient_id"] = entry.patient_id
    adata.obs["sample_id"] = entry.sample_id or entry.slide_id
    adata.obs["source"] = entry.name
    adata.obs["source_type"] = entry.source_type
    adata.obs["platform"] = entry.platform
    adata.obs["mpp_x"] = float(entry.mpp_x or 0.5)
    adata.obs["mpp_y"] = float(entry.mpp_y or 0.5)
    _apply_coordinates(adata, entry)
    _apply_annotations(adata, entry)
    return adata


def build_registry(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    registration = config["registration"]
    manifest_path = registration["manifest_path"]
    output_dir = Path(registration["output_dir"])
    adata_dir = output_dir / "adata"
    manifest_exists = Path(manifest_path).exists()
    entries = load_manifest(manifest_path) if manifest_exists else []

    slides = pd.DataFrame([entry.to_row() for entry in entries])
    if slides.empty and not dry_run:
        raise ValueError("manifest has no source entries")

    if dry_run:
        return {
            "num_sources": int(len(slides)),
            "manifest_exists": bool(manifest_exists),
            "slides_csv": str((output_dir / "slides.csv").resolve()),
            "gene_symbol_map_csv": str((output_dir / "gene_symbol_map.csv").resolve()),
            "adata_dir": str(adata_dir.resolve()),
            "dry_run": True,
        }

    ensure_directory(output_dir)
    ensure_directory(adata_dir)

    gene_symbols: set[str] = set()
    saved_adata_paths: list[str] = []
    for entry in entries:
        adata = _normalize_adata(_load_counts_matrix(entry), entry)
        out_path = adata_dir / f"{entry.slide_id}.h5ad"
        adata.write_h5ad(out_path)
        saved_adata_paths.append(str(out_path.resolve()))
        gene_symbols.update(map(str, adata.var_names))

    slides["adata_path"] = saved_adata_paths
    if "split" not in slides.columns:
        slides["split"] = ""
    slides["split"] = slides.apply(
        lambda row: row["split"] if str(row.get("split", "")).strip() else assign_deterministic_split(str(row["patient_id"])),
        axis=1,
    )
    slides.to_csv(output_dir / "slides.csv", index=False)
    pd.DataFrame({"gene_symbol": sorted(gene_symbols)}).to_csv(output_dir / "gene_symbol_map.csv", index=False)
    split_summary = build_split_tables(slides, output_dir / "splits")

    summary = {
        "num_sources": int(len(slides)),
        "slides_csv": str((output_dir / "slides.csv").resolve()),
        "gene_symbol_map_csv": str((output_dir / "gene_symbol_map.csv").resolve()),
        "adata_dir": str(adata_dir.resolve()),
        "saved_adata_paths": saved_adata_paths,
        **split_summary,
        "dry_run": False,
    }
    write_json(output_dir / "register_summary.json", summary)
    return summary
