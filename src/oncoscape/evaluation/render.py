from __future__ import annotations

import ast
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from oncoscape.core import ensure_directory, read_frame, write_frame, write_json
from oncoscape.evaluation.metrics import balanced_accuracy, js_divergence, macro_f1, pearson_mean
from oncoscape.models import DeepSpatialMultiTaskModel
from oncoscape.training.trainer import (
    _feature_matrix,
    _parse_vector,
    _predict_centroid_classifier,
    _predict_linear_classifier,
    _predict_regression,
    _prepare_slide_batch,
    _smooth_compartments,
)


COMPARTMENT_COLORS = {
    "invasive_tumor": (220, 20, 60),
    "in_situ_tumor": (255, 140, 0),
    "stroma": (46, 139, 87),
    "immune_rich": (65, 105, 225),
    "adipose_normal": (238, 221, 130),
    "necrosis_background": (119, 136, 153),
}


def _predict_compartment_classical(x: np.ndarray, model_name: str, model: dict[str, Any]) -> np.ndarray:
    if model_name == "constant":
        return np.asarray([str(model["constant"])] * len(x))
    if model_name in {"nearest_centroid", "gaussian_diag"}:
        return _predict_centroid_classifier(x, model, model_name)
    return _predict_linear_classifier(x, model)


def _render_class_map(frame: pd.DataFrame, value_column: str, out_path: Path, tile_px: int) -> None:
    width = int((frame["x_um"].max() - frame["x_um"].min()) / 112.0 + 2) * tile_px
    height = int((frame["y_um"].max() - frame["y_um"].min()) / 112.0 + 2) * tile_px
    image = Image.new("RGB", (max(width, tile_px), max(height, tile_px)), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    x0 = frame["x_um"].min()
    y0 = frame["y_um"].min()
    for row in frame.to_dict(orient="records"):
        x = int(round((row["x_um"] - x0) / 112.0)) * tile_px
        y = int(round((row["y_um"] - y0) / 112.0)) * tile_px
        color = COMPARTMENT_COLORS.get(str(row[value_column]), (0, 0, 0))
        draw.rectangle([x, y, x + tile_px - 1, y + tile_px - 1], fill=color)
    image.save(out_path)


def _render_score_map(frame: pd.DataFrame, score_column: str, out_path: Path, tile_px: int) -> None:
    width = int((frame["x_um"].max() - frame["x_um"].min()) / 112.0 + 2) * tile_px
    height = int((frame["y_um"].max() - frame["y_um"].min()) / 112.0 + 2) * tile_px
    image = Image.new("RGB", (max(width, tile_px), max(height, tile_px)), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    x0 = frame["x_um"].min()
    y0 = frame["y_um"].min()
    max_score = max(float(frame[score_column].max()), 1e-8)
    for row in frame.to_dict(orient="records"):
        x = int(round((row["x_um"] - x0) / 112.0)) * tile_px
        y = int(round((row["y_um"] - y0) / 112.0)) * tile_px
        intensity = int(np.clip((float(row[score_column]) / max_score) * 255, 0, 255))
        draw.rectangle([x, y, x + tile_px - 1, y + tile_px - 1], fill=(255, 255 - intensity, 255 - intensity))
    image.save(out_path)


def _evaluate_classical(model: dict[str, Any], eval_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    raw_x = _feature_matrix(eval_frame)
    mean = np.asarray(model["feature_mean"], dtype=np.float32)
    std = np.clip(np.asarray(model["feature_std"], dtype=np.float32), 1e-6, None)
    x_eval = (raw_x - mean) / std
    coords = x_eval[:, -5:-3]
    smoothing_radius = float(model.get("spatial_smoothing_radius_um", 160.0))
    eval_frame = eval_frame.copy()
    eval_frame["compartment_pred"] = _smooth_compartments(
        coords,
        _predict_compartment_classical(x_eval, model["compartment_model_name"], model["compartment_model"]),
        radius=smoothing_radius,
    )
    composition_pred = _predict_regression(x_eval, model["composition_model"], "composition", smoothing_radius=smoothing_radius)
    program_pred = _predict_regression(x_eval, model["program_model"], "program", smoothing_radius=smoothing_radius)
    return _attach_predictions(eval_frame, model, composition_pred, program_pred)


def _evaluate_deep(model_payload: dict[str, Any], eval_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("deep_spatial_multitask evaluation requires torch") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    smoothing_radius = float(model_payload.get("spatial_smoothing_radius_um", 160.0))
    model = DeepSpatialMultiTaskModel(model_payload["model_spec"]).to(device)
    model.load_state_dict(model_payload["state_dict"])
    model.eval()
    compartments = model_payload["compartment_classes"]
    slide_tables = []
    with torch.no_grad():
        for _, slide_frame in eval_frame.groupby("slide_id"):
            batch = _prepare_slide_batch(slide_frame.reset_index(drop=True), compartments, torch, device)
            outputs = model(batch["patches"], batch["coords"], batch["edge_index"])
            pred_comp_idx = outputs["compartment_logits"].argmax(dim=1).cpu().numpy()
            pred_comp = np.asarray([compartments[idx] for idx in pred_comp_idx])
            pred_comp = _smooth_compartments(batch["coords_np"], pred_comp, radius=smoothing_radius)
            pred_mix = outputs["composition_logits"].softmax(dim=1).cpu().numpy()
            pred_prog = outputs["program_values"].cpu().numpy()
            slide_pred = slide_frame.copy()
            slide_pred["compartment_pred"] = pred_comp
            for col_idx, name in enumerate(model_payload["composition_classes"]):
                slide_pred[f"composition_pred__{name}"] = pred_mix[:, col_idx]
            for prog_idx, name in enumerate(model_payload["programs"]):
                slide_pred[f"program_pred__{name}"] = pred_prog[:, prog_idx]
            slide_tables.append(slide_pred)
    merged = pd.concat(slide_tables, ignore_index=True) if slide_tables else eval_frame.copy()
    return _metrics_from_table(merged)


def _attach_predictions(eval_frame: pd.DataFrame, model: dict[str, Any], composition_pred: np.ndarray, program_pred: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    eval_frame = eval_frame.copy()
    for col_idx, name in enumerate(model["composition_classes"]):
        eval_frame[f"composition_pred__{name}"] = composition_pred[:, col_idx]
    for prog_idx, name in enumerate(model["programs"]):
        eval_frame[f"program_pred__{name}"] = program_pred[:, prog_idx]
    return _metrics_from_table(eval_frame)


def _metrics_from_table(eval_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    y_comp = eval_frame["compartment_target"].astype(str).to_numpy()
    y_mix = np.vstack([_parse_vector(v) for v in eval_frame["composition_target"]])
    y_prog = np.vstack([_parse_vector(v) for v in eval_frame["program_target"]])
    comp_mask = eval_frame.get("teacher_mask_compartment", pd.Series(np.ones(len(eval_frame), dtype=int))).to_numpy(dtype=np.float32) > 0
    mix_mask = eval_frame.get("teacher_mask_composition", pd.Series(np.ones(len(eval_frame), dtype=int))).to_numpy(dtype=np.float32) > 0
    prog_mask = eval_frame.get("teacher_mask_program", pd.Series(np.ones(len(eval_frame), dtype=int))).to_numpy(dtype=np.float32) > 0
    composition_cols = [col for col in eval_frame.columns if col.startswith("composition_pred__")]
    program_cols = [col for col in eval_frame.columns if col.startswith("program_pred__")]
    pred_mix = eval_frame[composition_cols].to_numpy(dtype=np.float32)
    pred_prog = eval_frame[program_cols].to_numpy(dtype=np.float32)
    metrics = {
        "compartment_macro_f1": macro_f1(y_comp[comp_mask], eval_frame["compartment_pred"].astype(str).to_numpy()[comp_mask]) if comp_mask.any() else 0.0,
        "compartment_balanced_accuracy": balanced_accuracy(y_comp[comp_mask], eval_frame["compartment_pred"].astype(str).to_numpy()[comp_mask]) if comp_mask.any() else 0.0,
        "composition_mean_pearson": pearson_mean(y_mix[mix_mask], pred_mix[mix_mask]) if mix_mask.any() else 0.0,
        "composition_js_divergence": js_divergence(y_mix[mix_mask], pred_mix[mix_mask]) if mix_mask.any() else 0.0,
        "program_mean_pearson": pearson_mean(y_prog[prog_mask], pred_prog[prog_mask]) if prog_mask.any() else 0.0,
        "n_eval_tiles": int(len(eval_frame)),
        "n_compartment_eval_tiles": int(comp_mask.sum()),
        "n_composition_eval_tiles": int(mix_mask.sum()),
        "n_program_eval_tiles": int(prog_mask.sum()),
    }
    return eval_frame, metrics


def evaluate_and_render(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    render = config["render"]
    predictions_dir = Path(render["predictions_dir"])
    report_dir = Path(render["report_dir"])
    summary = {
        "checkpoint_path": str(Path(render["checkpoint_path"]).resolve()),
        "predictions_dir": str(predictions_dir.resolve()),
        "report_dir": str(report_dir.resolve()),
        "split": config["evaluation"]["split"],
        "dry_run": dry_run,
    }
    if dry_run:
        return summary

    ensure_directory(predictions_dir)
    ensure_directory(report_dir)
    frame = read_frame(config["train_run"]["tile_dataset_path"])
    target_split = config["evaluation"]["split"]
    eval_frame = frame[frame["split"] == target_split].copy()
    if eval_frame.empty:
        eval_frame = frame[frame["split"] == "val"].copy()
    if eval_frame.empty:
        eval_frame = frame.copy()

    checkpoint_path = Path(render["checkpoint_path"])
    model_payload = None
    try:
        import torch

        model_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        with checkpoint_path.open("rb") as handle:
            model_payload = pickle.load(handle)

    model_family = model_payload.get("model_family", "enhanced_classical_ensemble")
    if model_family == "deep_spatial_multitask":
        eval_table, metrics = _evaluate_deep(model_payload, eval_frame)
    else:
        eval_table, metrics = _evaluate_classical(model_payload, eval_frame)

    write_json(report_dir / "test_metrics.json", metrics)
    per_slide_rows = []
    tile_px = int(render.get("render_tile_px", 24))
    composition_names = model_payload["composition_classes"]
    program_names = model_payload["programs"]
    for slide_id, slide_frame in eval_table.groupby("slide_id"):
        slide_dir = ensure_directory(predictions_dir / str(slide_id))
        write_frame(slide_frame, slide_dir / "tile_predictions.parquet")
        _render_class_map(slide_frame, "compartment_pred", slide_dir / "compartment_map.png", tile_px)
        composition_dir = ensure_directory(slide_dir / "composition_maps")
        for name in composition_names:
            _render_score_map(slide_frame, f"composition_pred__{name}", composition_dir / f"{name}.png", tile_px)
        program_dir = ensure_directory(slide_dir / "program_maps")
        for name in program_names:
            _render_score_map(slide_frame, f"program_pred__{name}", program_dir / f"{name}.png", tile_px)
        per_slide_rows.append(
            {
                "slide_id": slide_id,
                "n_tiles": int(len(slide_frame)),
                "compartment_macro_f1": (
                    macro_f1(
                        slide_frame.loc[slide_frame["teacher_mask_compartment"] > 0, "compartment_target"].astype(str).to_numpy(),
                        slide_frame.loc[slide_frame["teacher_mask_compartment"] > 0, "compartment_pred"].astype(str).to_numpy(),
                    )
                    if (slide_frame["teacher_mask_compartment"] > 0).any()
                    else 0.0
                ),
            }
        )
    pd.DataFrame(per_slide_rows).to_csv(report_dir / "per_slide_metrics.csv", index=False)
    return summary | metrics | {"model_family": model_family}
