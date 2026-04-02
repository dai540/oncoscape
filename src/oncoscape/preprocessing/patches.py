from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pickle
from PIL import Image

from oncoscape.core import ensure_directory, write_frame, write_json


def _crop_patch(
    image: Image.Image,
    x_um: float,
    y_um: float,
    tile_size_um: float,
    patch_size_px: int,
    source_mpp_x: float,
    source_mpp_y: float,
) -> Image.Image:
    cx = int(round(x_um / max(source_mpp_x, 1e-6)))
    cy = int(round(y_um / max(source_mpp_y, 1e-6)))
    half_w = max(1, int(round((tile_size_um / max(source_mpp_x, 1e-6)) / 2.0)))
    half_h = max(1, int(round((tile_size_um / max(source_mpp_y, 1e-6)) / 2.0)))
    box = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    cropped = image.crop(box)
    if cropped.size != (patch_size_px, patch_size_px):
        cropped = cropped.resize((patch_size_px, patch_size_px), Image.BILINEAR)
    return cropped


def _tissue_fraction(array: np.ndarray) -> float:
    gray = array.mean(axis=2) / 255.0
    return float((gray < 0.95).mean())


def _knn_edges(coords: np.ndarray, k: int) -> np.ndarray:
    if len(coords) <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, : min(k, len(coords) - 1)]
    src = np.repeat(np.arange(len(coords)), neighbors.shape[1])
    dst = neighbors.reshape(-1)
    return np.vstack([src, dst]).astype(np.int64)


def _subsample_frame(frame: pd.DataFrame, max_tiles: int | None) -> pd.DataFrame:
    if max_tiles is None or max_tiles <= 0 or len(frame) <= max_tiles:
        return frame.reset_index(drop=True)
    idx = np.linspace(0, len(frame) - 1, max_tiles, dtype=int)
    return frame.iloc[idx].reset_index(drop=True)


def extract_patches_and_graphs(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    cfg = config["patch_extraction"]
    slides = pd.read_csv(cfg["slides_csv_path"])
    slides = slides[slides["source_type"].isin(["visium", "xenium"]) & slides["image_path"].fillna("").ne("")].reset_index(drop=True)
    teachers = ad.read_h5ad(cfg["teacher_labels_path"])
    patches_dir = Path(cfg["patches_dir"])
    graphs_dir = Path(cfg["graphs_dir"])
    tile_dataset_path = Path(cfg["tile_dataset_path"])
    outputs = {
        "patches_dir": str(patches_dir.resolve()),
        "graphs_dir": str(graphs_dir.resolve()),
        "tile_dataset_path": str(tile_dataset_path.resolve()),
        "dry_run": dry_run,
    }
    if dry_run:
        outputs["num_slides"] = int(slides["slide_id"].nunique())
        outputs["n_teacher_tiles"] = int(teachers.n_obs)
        return outputs

    ensure_directory(patches_dir)
    ensure_directory(graphs_dir)
    patch_rows: list[dict[str, Any]] = []
    patch_size_px = int(cfg["patch_size_px"])
    target_mpp = float(cfg["target_mpp"])
    tile_size_um = float(cfg.get("tile_size_um", patch_size_px * target_mpp))
    graph_k = int(config["graph"]["k"])
    max_tiles_per_slide = cfg.get("max_tiles_per_slide")
    max_tiles_per_slide = int(max_tiles_per_slide) if max_tiles_per_slide not in (None, "") else None

    teacher_obs = teachers.obs.copy().reset_index(drop=True)
    composition = pd.DataFrame(
        teachers.obsm["composition"],
        columns=config["tasks"]["composition_classes"],
        index=teacher_obs.index,
    )
    programs = pd.DataFrame(
        teachers.obsm["programs"],
        columns=config["tasks"]["programs"],
        index=teacher_obs.index,
    )

    for slide in slides.to_dict(orient="records"):
        slide_tiles = teacher_obs[teacher_obs["slide_id"] == slide["slide_id"]].copy()
        if slide_tiles.empty:
            continue
        slide_tiles = _subsample_frame(slide_tiles, max_tiles_per_slide)
        image = Image.open(slide["image_path"]).convert("RGB")
        slide_patch_dir = ensure_directory(patches_dir / slide["slide_id"])
        coords = slide_tiles[["x_um", "y_um"]].to_numpy(dtype=float)
        edge_index = _knn_edges(coords, graph_k)
        with Path(graphs_dir / f"{slide['slide_id']}.pt").open("wb") as handle:
            pickle.dump({"edge_index": edge_index, "coords_um": coords}, handle)

        for row_id, tile in slide_tiles.iterrows():
            patch = _crop_patch(
                image,
                float(tile["x_um"]),
                float(tile["y_um"]),
                tile_size_um=tile_size_um,
                patch_size_px=patch_size_px,
                source_mpp_x=float(slide.get("mpp_x", target_mpp) or target_mpp),
                source_mpp_y=float(slide.get("mpp_y", target_mpp) or target_mpp),
            )
            patch_array = np.asarray(patch, dtype=np.uint8)
            tissue_fraction = _tissue_fraction(patch_array)
            patch_mean = float(patch_array.mean() / 255.0)
            patch_std = float(patch_array.std() / 255.0)
            qc_pass = (
                tissue_fraction >= float(cfg["tissue_fraction_min"])
                and patch_mean >= float(cfg["patch_mean_min"])
                and patch_std >= float(cfg["patch_std_min"])
            )
            patch_path = slide_patch_dir / f"{tile['tile_id']}.png"
            patch.save(patch_path)
            comp = composition.loc[row_id].to_numpy(dtype=np.float32)
            prog = programs.loc[row_id].to_numpy(dtype=np.float32)
            patch_rows.append(
                {
                    "tile_id": tile["tile_id"],
                    "slide_id": slide["slide_id"],
                    "patient_id": slide["patient_id"],
                    "source": slide.get("source", slide.get("name", slide["slide_id"])),
                    "split": slide.get("split", ""),
                    "x_um": float(tile["x_um"]),
                    "y_um": float(tile["y_um"]),
                    "patch_path": str(patch_path.resolve()),
                    "graph_path": str((graphs_dir / f"{slide['slide_id']}.pt").resolve()),
                    "compartment_target": tile["compartment"],
                    "composition_target": comp.tolist(),
                    "program_target": prog.tolist(),
                    "teacher_mask_compartment": int(tile["teacher_mask_compartment"]),
                    "teacher_mask_composition": int(tile["teacher_mask_composition"]),
                    "teacher_mask_program": int(tile["teacher_mask_program"]),
                    "teacher_confidence_compartment": float(tile["teacher_confidence_compartment"]),
                    "teacher_confidence_composition": float(tile["teacher_confidence_composition"]),
                    "teacher_confidence_program": float(tile["teacher_confidence_program"]),
                    "teacher_source_compartment": str(tile.get("teacher_source_compartment", "")),
                    "teacher_source_composition": str(tile.get("teacher_source_composition", "")),
                    "teacher_source_program": str(tile.get("teacher_source_program", "")),
                    "tissue_fraction": tissue_fraction,
                    "patch_mean": patch_mean,
                    "patch_std": patch_std,
                    "qc_pass": bool(qc_pass),
                }
            )

    frame = pd.DataFrame(patch_rows)
    written_path = write_frame(frame, tile_dataset_path)
    outputs["n_tiles"] = int(len(frame))
    outputs["n_qc_pass"] = int(frame["qc_pass"].sum()) if not frame.empty else 0
    outputs["tile_dataset_written_path"] = str(written_path.resolve())
    write_json(tile_dataset_path.with_suffix(".summary.json"), outputs)
    return outputs
