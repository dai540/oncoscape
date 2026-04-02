from __future__ import annotations

import ast
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from oncoscape.core import ensure_directory, read_frame, write_json


def _parse_vector(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=np.float32)
    raise TypeError(f"unsupported vector payload: {type(value)}")


def _rgb_hist(arr: np.ndarray, bins: int = 8) -> np.ndarray:
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0.0, 1.0), density=True)
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats)


def _gradient_features(gray: np.ndarray) -> np.ndarray:
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    mag = np.sqrt(gx * gx + gy * gy)
    return np.asarray(
        [
            float(gx.mean()),
            float(gx.std()),
            float(gy.mean()),
            float(gy.std()),
            float(mag.mean()),
            float(mag.std()),
        ],
        dtype=np.float32,
    )


def _coarse_grid(gray: np.ndarray, grid: int = 8) -> np.ndarray:
    h, w = gray.shape
    crop_h = (h // grid) * grid
    crop_w = (w // grid) * grid
    gray = gray[:crop_h, :crop_w]
    pooled = gray.reshape(grid, crop_h // grid, grid, crop_w // grid).mean(axis=(1, 3))
    return pooled.reshape(-1).astype(np.float32)


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    features = []
    for row in frame.to_dict(orient="records"):
        image = Image.open(row["patch_path"]).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        gray = arr.mean(axis=2)
        rgb_mean = arr.mean(axis=(0, 1))
        rgb_std = arr.std(axis=(0, 1))
        gray_stats = np.asarray([float(gray.mean()), float(gray.std())], dtype=np.float32)
        feature = np.concatenate(
            [
                rgb_mean,
                rgb_std,
                gray_stats,
                _rgb_hist(arr, bins=8),
                _gradient_features(gray),
                _coarse_grid(gray, grid=8),
                np.asarray(
                    [
                        float(row["x_um"]) / 10000.0,
                        float(row["y_um"]) / 10000.0,
                        float(row["tissue_fraction"]),
                        float(row["patch_mean"]),
                        float(row["patch_std"]),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        features.append(feature)
    return np.vstack(features).astype(np.float32)


def _standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = np.clip(train_x.std(axis=0, keepdims=True), 1e-6, None)
    return (train_x - mean) / std, (other_x - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for label in sorted(pd.Series(y_true).unique()):
        mask = y_true == label
        if mask.sum():
            scores.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(scores)) if scores else 0.0


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


def _fit_ridge(x: np.ndarray, y: np.ndarray, weight: np.ndarray, alpha: float) -> np.ndarray:
    w = np.sqrt(np.clip(weight, 1e-8, None))[:, None]
    xw = x * w
    yw = y * w
    eye = np.eye(x.shape[1], dtype=np.float32)
    return np.linalg.solve(xw.T @ xw + alpha * eye, xw.T @ yw)


def _fit_centroid_classifier(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    classes = sorted(map(str, pd.Series(y).unique()))
    centroids = {}
    variances = {}
    priors = {}
    for label in classes:
        mask = y == label
        block = x[mask]
        centroids[label] = block.mean(axis=0) if mask.any() else np.zeros(x.shape[1], dtype=np.float32)
        variances[label] = np.clip(block.var(axis=0), 1e-4, None) if mask.any() else np.ones(x.shape[1], dtype=np.float32)
        priors[label] = float(mask.mean())
    return {"classes": classes, "centroids": centroids, "variances": variances, "priors": priors}


def _predict_centroid_classifier(x: np.ndarray, model: dict[str, Any], mode: str) -> np.ndarray:
    labels = []
    for row in x:
        scores = []
        for label in model["classes"]:
            mu = model["centroids"][label]
            if mode == "nearest_centroid":
                score = -float(((row - mu) ** 2).sum())
            else:
                var = model["variances"][label]
                prior = max(model["priors"][label], 1e-8)
                score = -0.5 * float((((row - mu) ** 2) / var).sum() + np.log(var).sum()) + np.log(prior)
            scores.append(score)
        labels.append(model["classes"][int(np.argmax(scores))])
    return np.asarray(labels)


def _one_hot(y: np.ndarray, classes: list[str]) -> np.ndarray:
    mat = np.zeros((len(y), len(classes)), dtype=np.float32)
    lookup = {label: idx for idx, label in enumerate(classes)}
    for idx, label in enumerate(y):
        mat[idx, lookup[str(label)]] = 1.0
    return mat


def _fit_linear_classifier(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    classes = sorted(map(str, pd.Series(y).unique()))
    target = _one_hot(y, classes)
    weight = np.ones(len(y), dtype=np.float32)
    weights = _fit_ridge(x, target, weight, alpha)
    return {"classes": classes, "weights": weights}


def _predict_linear_classifier(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    scores = x @ model["weights"]
    idx = scores.argmax(axis=1)
    return np.asarray([model["classes"][i] for i in idx])


def _rbf_features(train_x: np.ndarray, x: np.ndarray, gamma: float, max_centers: int = 256) -> tuple[np.ndarray, np.ndarray]:
    centers = train_x if len(train_x) <= max_centers else train_x[np.linspace(0, len(train_x) - 1, max_centers, dtype=int)]
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    feats = np.exp(-gamma * d2)
    return feats.astype(np.float32), centers.astype(np.float32)


def _smooth_composition(coords: np.ndarray, values: np.ndarray, radius: float = 160.0) -> np.ndarray:
    if len(values) <= 1:
        return values
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    neighbors = dist <= radius
    smoothed = np.zeros_like(values)
    for i in range(len(values)):
        mask = neighbors[i]
        block = values[mask]
        smoothed[i] = block.mean(axis=0)
    smoothed = np.clip(smoothed, 0.0, None)
    return smoothed / np.clip(smoothed.sum(axis=1, keepdims=True), 1e-8, None)


def _smooth_programs(coords: np.ndarray, values: np.ndarray, radius: float = 160.0) -> np.ndarray:
    if len(values) <= 1:
        return values
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    neighbors = dist <= radius
    smoothed = np.zeros_like(values)
    for i in range(len(values)):
        smoothed[i] = values[neighbors[i]].mean(axis=0)
    return smoothed


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


def _split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["split"].isin(["", "train"])].copy()
    val = frame[frame["split"] == "val"].copy()
    if val.empty:
        holdout_slides = train["slide_id"].drop_duplicates().sort_values().tail(max(1, train["slide_id"].nunique() // 5))
        val = train[train["slide_id"].isin(holdout_slides)].copy()
        train = train[~train["slide_id"].isin(holdout_slides)].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)


def _select_compartment_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> tuple[dict[str, Any], dict[str, float]]:
    candidates = []
    centroid_model = _fit_centroid_classifier(x_train, y_train)
    candidates.append(("nearest_centroid", centroid_model, _predict_centroid_classifier(x_val, centroid_model, "nearest_centroid")))
    candidates.append(("gaussian_diag", centroid_model, _predict_centroid_classifier(x_val, centroid_model, "gaussian_diag")))
    for alpha in (1e-3, 1e-2, 1e-1, 1.0):
        linear_model = _fit_linear_classifier(x_train, y_train, alpha)
        candidates.append((f"ridge_ovr_{alpha}", linear_model, _predict_linear_classifier(x_val, linear_model)))

    best = None
    best_metrics = None
    for name, model, pred in candidates:
        pred = _smooth_compartments(x_val[:, -5:-3], pred)
        metrics = {
            "macro_f1": _macro_f1(y_val, pred),
            "balanced_accuracy": _balanced_accuracy(y_val, pred),
        }
        if best is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best = {"name": name, "model": model}
            best_metrics = metrics
    return best, best_metrics


def _select_regression_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weight: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    task: str,
) -> tuple[dict[str, Any], float]:
    candidates = []
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        weights = _fit_ridge(x_train, y_train, weight, alpha)
        pred = x_val @ weights
        candidates.append((f"linear_ridge_{alpha}", {"kind": "linear", "weights": weights, "alpha": alpha}, pred))
    for gamma in (0.01, 0.05, 0.1):
        train_phi, centers = _rbf_features(x_train, x_train, gamma)
        val_phi, _ = _rbf_features(x_train, x_val, gamma)
        for alpha in (1e-3, 1e-2, 1e-1):
            weights = _fit_ridge(train_phi, y_train, weight, alpha)
            pred = val_phi @ weights
            candidates.append((f"rbf_ridge_g{gamma}_a{alpha}", {"kind": "rbf", "weights": weights, "gamma": gamma, "centers": centers, "alpha": alpha}, pred))

    best = None
    best_score = None
    for name, model, pred in candidates:
        if task == "composition":
            pred = np.clip(pred, 0.0, None)
            pred = pred / np.clip(pred.sum(axis=1, keepdims=True), 1e-8, None)
            pred = _smooth_composition(x_val[:, -5:-3], pred)
        else:
            pred = _smooth_programs(x_val[:, -5:-3], pred)
        score = _pearson_mean(y_val, pred)
        if best is None or score > best_score:
            best = {"name": name, "model": model}
            best_score = score
    return best, float(best_score)


def _predict_regression(x: np.ndarray, model: dict[str, Any], task: str) -> np.ndarray:
    if model["kind"] == "linear":
        pred = x @ model["weights"]
    else:
        phi, _ = _rbf_features(model["centers"], x, model["gamma"], max_centers=len(model["centers"]))
        pred = phi @ model["weights"]
    if task == "composition":
        pred = np.clip(pred, 0.0, None)
        pred = pred / np.clip(pred.sum(axis=1, keepdims=True), 1e-8, None)
        pred = _smooth_composition(x[:, -5:-3], pred)
    else:
        pred = _smooth_programs(x[:, -5:-3], pred)
    return pred


def train_breast_model(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    cfg = config["train_run"]
    training = config["training"]
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    report_dir = Path(cfg["report_dir"])
    summary = {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "report_dir": str(report_dir.resolve()),
        "epochs": int(training["epochs"]),
        "device": str(training["device"]),
        "dry_run": dry_run,
    }
    if dry_run:
        return summary

    ensure_directory(checkpoint_dir)
    ensure_directory(report_dir)

    frame = read_frame(cfg["tile_dataset_path"])
    frame = frame[frame["qc_pass"]].reset_index(drop=True)
    train_frame, val_frame = _split_frame(frame)
    if train_frame.empty or val_frame.empty:
        raise ValueError("training requires non-empty train and val splits")

    raw_x_train = _feature_matrix(train_frame)
    raw_x_val = _feature_matrix(val_frame)
    x_train, x_val, mean, std = _standardize(raw_x_train, raw_x_val)
    y_comp_train = train_frame["compartment_target"].astype(str).to_numpy()
    y_comp_val = val_frame["compartment_target"].astype(str).to_numpy()
    y_mix_train = np.vstack([_parse_vector(v) for v in train_frame["composition_target"]])
    y_mix_val = np.vstack([_parse_vector(v) for v in val_frame["composition_target"]])
    y_prog_train = np.vstack([_parse_vector(v) for v in train_frame["program_target"]])
    y_prog_val = np.vstack([_parse_vector(v) for v in val_frame["program_target"]])

    best_compartment, comp_metrics = _select_compartment_model(x_train, y_comp_train, x_val, y_comp_val)
    best_composition, compo_score = _select_regression_model(
        x_train,
        y_mix_train,
        train_frame["teacher_confidence_composition"].to_numpy(dtype=np.float32),
        x_val,
        y_mix_val,
        "composition",
    )
    best_program, prog_score = _select_regression_model(
        x_train,
        y_prog_train,
        train_frame["teacher_confidence_program"].to_numpy(dtype=np.float32),
        x_val,
        y_prog_val,
        "program",
    )

    comp_val_pred = (
        _smooth_compartments(x_val[:, -5:-3], _predict_centroid_classifier(x_val, best_compartment["model"], "nearest_centroid"))
        if best_compartment["name"] == "nearest_centroid"
        else _smooth_compartments(
            x_val[:, -5:-3],
            _predict_centroid_classifier(x_val, best_compartment["model"], "gaussian_diag")
            if best_compartment["name"] == "gaussian_diag"
            else _predict_linear_classifier(x_val, best_compartment["model"]),
        )
    )
    mix_val_pred = _predict_regression(x_val, best_composition["model"], "composition")
    prog_val_pred = _predict_regression(x_val, best_program["model"], "program")

    val_metrics = {
        "macro_f1": _macro_f1(y_comp_val, comp_val_pred),
        "balanced_accuracy": _balanced_accuracy(y_comp_val, comp_val_pred),
        "composition_mean_pearson": _pearson_mean(y_mix_val, mix_val_pred),
        "program_mean_pearson": _pearson_mean(y_prog_val, prog_val_pred),
    }
    train_metrics = {
        "epoch": 1,
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "compartment_model": best_compartment["name"],
        "composition_model": best_composition["name"],
        "program_model": best_program["name"],
        "composition_val_score": compo_score,
        "program_val_score": prog_score,
    }

    model_payload = {
        "feature_names": [
            "rgb_mean_std_hist_texture_grid_spatial",
        ],
        "feature_mean": mean,
        "feature_std": std,
        "compartment_model_name": best_compartment["name"],
        "compartment_model": best_compartment["model"],
        "composition_model_name": best_composition["name"],
        "composition_model": best_composition["model"],
        "program_model_name": best_program["name"],
        "program_model": best_program["model"],
        "composition_classes": config["tasks"]["composition_classes"],
        "programs": config["tasks"]["programs"],
        "val_metrics": val_metrics,
    }
    with Path(checkpoint_dir / "best.pt").open("wb") as handle:
        pickle.dump(model_payload, handle)
    with Path(checkpoint_dir / "last.pt").open("wb") as handle:
        pickle.dump(model_payload, handle)
    pd.DataFrame([train_metrics]).to_csv(report_dir / "train_metrics.csv", index=False)
    pd.DataFrame([{"epoch": 1, **val_metrics}]).to_csv(report_dir / "val_metrics.csv", index=False)
    summary.update({"train_rows": int(len(train_frame)), "val_rows": int(len(val_frame)), "val_metrics": val_metrics})
    write_json(report_dir / "train_summary.json", summary)
    return summary
