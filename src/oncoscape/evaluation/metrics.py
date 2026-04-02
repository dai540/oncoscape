from __future__ import annotations

import numpy as np
import pandas as pd


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for label in sorted(pd.Series(y_true).unique()):
        mask = y_true == label
        if mask.sum():
            scores.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(scores)) if scores else 0.0


def pearson_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def js_divergence(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.clip(y_true, eps, None)
    y_pred = np.clip(y_pred, eps, None)
    y_true = y_true / y_true.sum(axis=1, keepdims=True)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    mean = 0.5 * (y_true + y_pred)
    kl_true = (y_true * (np.log(y_true) - np.log(mean))).sum(axis=1)
    kl_pred = (y_pred * (np.log(y_pred) - np.log(mean))).sum(axis=1)
    return float((0.5 * (kl_true + kl_pred)).mean())
