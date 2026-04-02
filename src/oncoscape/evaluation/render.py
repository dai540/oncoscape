from __future__ import annotations

import ast
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from oncoscape.core import ensure_directory, read_frame, write_frame, write_json
from oncoscape.training.trainer import _feature_matrix


COMPARTMENT_COLORS = {
    "invasive_tumor": (220, 20, 60),
    "in_situ_tumor": (255, 140, 0),
    "stroma": (46, 139, 87),
    "immune_rich": (65, 105, 225),
    "adipose_normal": (238, 221, 130),
    "necrosis_background": (119, 136, 153),
}


def _parse_vector(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=np.float32)
    raise TypeError(f"unsupported vector payload: {type(value)}")


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.clip(std, 1e-6, None)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    vals = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        vals.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(vals)) if vals else 0.0


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for label in sorted(pd.Series(y_true).unique()):
        mask = y_true == label
        if mask.sum():
            scores.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(scores)) if scores else 0.0


def _pearson_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for idx in range(y_true.shape[1]):
        a = y_true[:, idx]
        b = y_pred[:, idx]
        a_center = a - a.mean()
        b_center = b - b.mean()
        denom = float(np.sqrt((a_center * a_center).sum()) * np.sqrt((b_center * b_center).sum()))
        if denom < 1e-8:
            vals.append(0.0)
        else:
            vals.append(float((a_center * b_center).sum() / denom))
    return float(np.mean(vals)) if vals else 0.0


def _rbf_features(centers: np.ndarray, x: np.ndarray, gamma: float) -> np.ndarray:
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * d2).astype(np.float32)


def _smooth_composition(coords: np.ndarray, values: np.ndarray, radius: float = 160.0) -> np.ndarray:
    if len(values) <= 1:
        return values
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    neighbors = dist <= radius
    out = np.zeros_like(values)
    for i in range(len(values)):
        out[i] = values[neighbors[i]].mean(axis=0)
    out = np.clip(out, 0.0, None)
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-8, None)


def _smooth_programs(coords: np.ndarray, values: np.ndarray, radius: float = 160.0) -> np.ndarray:
    if len(values) <= 1:
        return values
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    neighbors = dist <= radius
    out = np.zeros_like(values)
    for i in range(len(values)):
        out[i] = values[neighbors[i]].mean(axis=0)
    return out


def _smooth_compartments(coords: np.ndarray, labels: np.ndarray, radius: float = 160.0) -> np.ndarray:
    if len(labels) <= 1:
        return labels
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    neighbors = dist <= radius
    out = []
    for i in range(len(labels)):
        vals, counts = np.unique(labels[neighbors[i]], return_counts=True)
        out.append(vals[counts.argmax()])
    return np.asarray(out)


def _predict_compartment(x: np.ndarray, model_name: str, model: dict[str, Any]) -> np.ndarray:
    if model_name in {"nearest_centroid", "gaussian_diag"}:
        labels = []
        for row in x:
            scores = []
            for label in model["classes"]:
                mu = model["centroids"][label]
                if model_name == "nearest_centroid":
                    score = -float(((row - mu) ** 2).sum())
                else:
                    var = model["variances"][label]
                    prior = max(model["priors"][label], 1e-8)
                    score = -0.5 * float((((row - mu) ** 2) / var).sum() + np.log(var).sum()) + np.log(prior)
                scores.append(score)
            labels.append(model["classes"][int(np.argmax(scores))])
        return np.asarray(labels)
    scores = x @ model["weights"]
    idx = scores.argmax(axis=1)
    return np.asarray([model["classes"][i] for i in idx])


def _predict_regression(x: np.ndarray, model: dict[str, Any], task: str) -> np.ndarray:
    if model["kind"] == "linear":
        pred = x @ model["weights"]
    else:
        pred = _rbf_features(model["centers"], x, model["gamma"]) @ model["weights"]
    if task == "composition":
        pred = np.clip(pred, 0.0, None)
        pred = pred / np.clip(pred.sum(axis=1, keepdims=True), 1e-8, None)
        pred = _smooth_composition(x[:, -5:-3], pred)
    else:
        pred = _smooth_programs(x[:, -5:-3], pred)
    return pred


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

    with Path(render["checkpoint_path"]).open("rb") as handle:
        model = pickle.load(handle)
    raw_x = _feature_matrix(eval_frame)
    x_eval = _standardize(raw_x, model["feature_mean"], model["feature_std"])
    coords = x_eval[:, -5:-3]

    eval_frame["compartment_pred"] = _smooth_compartments(
        coords,
        _predict_compartment(x_eval, model["compartment_model_name"], model["compartment_model"]),
    )
    composition_pred = _predict_regression(x_eval, model["composition_model"], "composition")
    program_pred = _predict_regression(x_eval, model["program_model"], "program")

    y_comp = eval_frame["compartment_target"].astype(str).to_numpy()
    y_mix = np.vstack([_parse_vector(v) for v in eval_frame["composition_target"]])
    y_prog = np.vstack([_parse_vector(v) for v in eval_frame["program_target"]])
    metrics = {
        "compartment_macro_f1": _macro_f1(y_comp, eval_frame["compartment_pred"].to_numpy()),
        "compartment_balanced_accuracy": _balanced_accuracy(y_comp, eval_frame["compartment_pred"].to_numpy()),
        "composition_mean_pearson": _pearson_mean(y_mix, composition_pred),
        "program_mean_pearson": _pearson_mean(y_prog, program_pred),
        "n_eval_tiles": int(len(eval_frame)),
    }
    write_json(report_dir / "test_metrics.json", metrics)

    per_slide_rows = []
    tile_px = int(render.get("render_tile_px", 24))
    for slide_id, slide_frame in eval_frame.groupby("slide_id"):
        slide_dir = ensure_directory(predictions_dir / str(slide_id))
        row_positions = eval_frame.index.get_indexer(slide_frame.index)
        slide_table = slide_frame.copy()
        for col_idx, name in enumerate(model["composition_classes"]):
            slide_table[f"composition_pred__{name}"] = composition_pred[row_positions, col_idx]
        for prog_idx, name in enumerate(model["programs"]):
            slide_table[f"program_pred__{name}"] = program_pred[row_positions, prog_idx]
        write_frame(slide_table, slide_dir / "tile_predictions.parquet")
        _render_class_map(slide_table, "compartment_pred", slide_dir / "compartment_map.png", tile_px)
        composition_dir = ensure_directory(slide_dir / "composition_maps")
        for name in model["composition_classes"]:
            _render_score_map(slide_table, f"composition_pred__{name}", composition_dir / f"{name}.png", tile_px)
        program_dir = ensure_directory(slide_dir / "program_maps")
        for name in model["programs"]:
            _render_score_map(slide_table, f"program_pred__{name}", program_dir / f"{name}.png", tile_px)
        per_slide_rows.append(
            {
                "slide_id": slide_id,
                "n_tiles": int(len(slide_frame)),
                "compartment_macro_f1": _macro_f1(
                    slide_table["compartment_target"].astype(str).to_numpy(),
                    slide_table["compartment_pred"].astype(str).to_numpy(),
                ),
            }
        )

    pd.DataFrame(per_slide_rows).to_csv(report_dir / "per_slide_metrics.csv", index=False)
    return summary | metrics
