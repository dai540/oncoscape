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


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    features = []
    for row in frame.to_dict(orient="records"):
        image = Image.open(row["patch_path"]).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        rgb_mean = arr.mean(axis=(0, 1))
        rgb_std = arr.std(axis=(0, 1))
        feature = np.concatenate(
            [
                rgb_mean,
                rgb_std,
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


def _fit_ridge(x: np.ndarray, y: np.ndarray, weight: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    w = np.sqrt(np.clip(weight, 1e-8, None))[:, None]
    xw = x * w
    yw = y * w
    eye = np.eye(x.shape[1], dtype=np.float32)
    return np.linalg.solve(xw.T @ xw + alpha * eye, xw.T @ yw)


def _fit_compartment_centroids(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    classes = sorted(map(str, pd.Series(y).unique()))
    centroids = {}
    for label in classes:
        mask = y == label
        centroids[label] = x[mask].mean(axis=0) if mask.any() else np.zeros(x.shape[1], dtype=np.float32)
    return {"classes": classes, "centroids": centroids}


def _predict_compartment(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    centroids = np.vstack([model["centroids"][label] for label in model["classes"]])
    distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.asarray([model["classes"][idx] for idx in distances.argmin(axis=1)])


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for label in sorted(pd.Series(y_true).unique()):
        mask = y_true == label
        if mask.sum() == 0:
            continue
        scores.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(scores)) if scores else 0.0


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    f1s = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def _pearson_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for idx in range(y_true.shape[1]):
        a = y_true[:, idx]
        b = y_pred[:, idx]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            scores.append(0.0)
        else:
            scores.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(scores)) if scores else 0.0


def _split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["split"].isin(["", "train"])].copy()
    val = frame[frame["split"] == "val"].copy()
    if val.empty:
        holdout_slides = train["slide_id"].drop_duplicates().sort_values().tail(max(1, train["slide_id"].nunique() // 5))
        val = train[train["slide_id"].isin(holdout_slides)].copy()
        train = train[~train["slide_id"].isin(holdout_slides)].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)


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

    x_train = _feature_matrix(train_frame)
    x_val = _feature_matrix(val_frame)
    y_comp_train = train_frame["compartment_target"].astype(str).to_numpy()
    y_comp_val = val_frame["compartment_target"].astype(str).to_numpy()
    y_mix_train = np.vstack([_parse_vector(v) for v in train_frame["composition_target"]])
    y_mix_val = np.vstack([_parse_vector(v) for v in val_frame["composition_target"]])
    y_prog_train = np.vstack([_parse_vector(v) for v in train_frame["program_target"]])
    y_prog_val = np.vstack([_parse_vector(v) for v in val_frame["program_target"]])

    compartment_model = _fit_compartment_centroids(x_train, y_comp_train)
    mix_weights = _fit_ridge(x_train, y_mix_train, train_frame["teacher_confidence_composition"].to_numpy(dtype=np.float32))
    prog_weights = _fit_ridge(x_train, y_prog_train, train_frame["teacher_confidence_program"].to_numpy(dtype=np.float32))

    comp_val_pred = _predict_compartment(x_val, compartment_model)
    mix_val_pred = np.clip(x_val @ mix_weights, 0.0, None)
    mix_val_pred = mix_val_pred / np.clip(mix_val_pred.sum(axis=1, keepdims=True), 1e-8, None)
    prog_val_pred = x_val @ prog_weights

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
    }

    model_payload = {
        "compartment_model": compartment_model,
        "composition_weights": mix_weights,
        "program_weights": prog_weights,
        "feature_names": ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std", "x_scaled", "y_scaled", "tissue_fraction", "patch_mean", "patch_std"],
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
