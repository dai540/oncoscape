from __future__ import annotations

import ast
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from oncoscape.core import ensure_directory, read_frame, write_json
from oncoscape.evaluation.metrics import balanced_accuracy, js_divergence, macro_f1, pearson_mean
from oncoscape.models import DeepSpatialMultiTaskModel, build_model_spec


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
        smoothed[i] = values[neighbors[i]].mean(axis=0)
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


def _select_compartment_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    smoothing_radius: float = 160.0,
) -> tuple[dict[str, Any], dict[str, float]]:
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
        pred = _smooth_compartments(x_val[:, -5:-3], pred, radius=smoothing_radius)
        metrics = {"macro_f1": macro_f1(y_val, pred), "balanced_accuracy": balanced_accuracy(y_val, pred)}
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
    smoothing_radius: float = 160.0,
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
            pred = _smooth_composition(x_val[:, -5:-3], pred, radius=smoothing_radius)
        else:
            pred = _smooth_programs(x_val[:, -5:-3], pred, radius=smoothing_radius)
        score = pearson_mean(y_val, pred)
        if best is None or score > best_score:
            best = {"name": name, "model": model}
            best_score = score
    return best, float(best_score)


def _predict_regression(x: np.ndarray, model: dict[str, Any], task: str, smoothing_radius: float = 160.0) -> np.ndarray:
    if model["kind"] == "linear":
        pred = x @ model["weights"]
    else:
        phi, _ = _rbf_features(model["centers"], x, model["gamma"], max_centers=len(model["centers"]))
        pred = phi @ model["weights"]
    if task == "composition":
        pred = np.clip(pred, 0.0, None)
        pred = pred / np.clip(pred.sum(axis=1, keepdims=True), 1e-8, None)
        pred = _smooth_composition(x[:, -5:-3], pred, radius=smoothing_radius)
    else:
        pred = _smooth_programs(x[:, -5:-3], pred, radius=smoothing_radius)
    return pred


def _save_pickle(path: Path, payload: dict[str, Any]) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _load_graph(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("rb") as handle:
        graph = pickle.load(handle)
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    coords = np.asarray(graph["coords_um"], dtype=np.float32)
    return edge_index, coords


def _load_patch_tensor(path: str | Path, torch_module) -> Any:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch_module.from_numpy(arr.transpose(2, 0, 1)).float()


def _prepare_slide_batch(frame: pd.DataFrame, compartments: list[str], torch_module, device: str) -> dict[str, Any]:
    graph_path = frame["graph_path"].iloc[0]
    edge_index_np, coords_np = _load_graph(graph_path)
    patches = torch_module.stack([_load_patch_tensor(path, torch_module) for path in frame["patch_path"].tolist()]).to(device)
    coords = torch_module.tensor(frame[["x_um", "y_um"]].to_numpy(dtype=np.float32), device=device)
    edge_index = torch_module.tensor(edge_index_np, dtype=torch_module.long, device=device)
    comp_lookup = {label: idx for idx, label in enumerate(compartments)}
    compartment_target = torch_module.tensor(
        [comp_lookup[str(value)] for value in frame["compartment_target"].tolist()],
        dtype=torch_module.long,
        device=device,
    )
    composition_target = torch_module.tensor(
        np.vstack([_parse_vector(v) for v in frame["composition_target"]]),
        dtype=torch_module.float32,
        device=device,
    )
    program_target = torch_module.tensor(
        np.vstack([_parse_vector(v) for v in frame["program_target"]]),
        dtype=torch_module.float32,
        device=device,
    )
    return {
        "slide_id": frame["slide_id"].iloc[0],
        "patches": patches,
        "coords": coords,
        "edge_index": edge_index,
        "compartment_target": compartment_target,
        "composition_target": composition_target,
        "program_target": program_target,
        "compartment_mask": torch_module.tensor(frame["teacher_mask_compartment"].to_numpy(dtype=np.float32), device=device),
        "composition_mask": torch_module.tensor(frame["teacher_mask_composition"].to_numpy(dtype=np.float32), device=device),
        "program_mask": torch_module.tensor(frame["teacher_mask_program"].to_numpy(dtype=np.float32), device=device),
        "compartment_weight": torch_module.tensor(frame["teacher_confidence_compartment"].to_numpy(dtype=np.float32), device=device),
        "composition_weight": torch_module.tensor(frame["teacher_confidence_composition"].to_numpy(dtype=np.float32), device=device),
        "program_weight": torch_module.tensor(frame["teacher_confidence_program"].to_numpy(dtype=np.float32), device=device),
        "coords_np": coords_np,
    }


def _deep_losses(outputs: dict[str, Any], batch: dict[str, Any], weights: dict[str, float], torch_module) -> tuple[Any, dict[str, float]]:
    import torch.nn.functional as F  # type: ignore

    comp_weight = batch["compartment_weight"] * batch["compartment_mask"]
    comp_loss_raw = F.cross_entropy(outputs["compartment_logits"], batch["compartment_target"], reduction="none")
    comp_denom = comp_weight.sum().clamp_min(1e-8)
    comp_loss = (comp_loss_raw * comp_weight).sum() / comp_denom

    pred_comp = outputs["composition_logits"].softmax(dim=1).clamp_min(1e-8)
    true_comp = batch["composition_target"].clamp_min(1e-8)
    true_comp = true_comp / true_comp.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mean = 0.5 * (pred_comp + true_comp)
    js = 0.5 * (
        (true_comp * (true_comp.log() - mean.log())).sum(dim=1)
        + (pred_comp * (pred_comp.log() - mean.log())).sum(dim=1)
    )
    composition_weight = batch["composition_weight"] * batch["composition_mask"]
    composition_loss = (js * composition_weight).sum() / composition_weight.sum().clamp_min(1e-8)

    huber = F.huber_loss(outputs["program_values"], batch["program_target"], reduction="none")
    program_weight = batch["program_weight"] * batch["program_mask"]
    program_loss = (huber.mean(dim=1) * program_weight).sum() / program_weight.sum().clamp_min(1e-8)

    if outputs["latent"].shape[0] > 1 and batch["edge_index"].numel() > 0:
        src, dst = batch["edge_index"]
        smooth = ((outputs["latent"][src] - outputs["latent"][dst]) ** 2).mean()
    else:
        smooth = outputs["latent"].new_tensor(0.0)

    total = (
        weights["compartment"] * comp_loss
        + weights["composition"] * composition_loss
        + weights["program"] * program_loss
        + weights["smooth"] * smooth
    )
    return total, {
        "compartment_loss": float(comp_loss.detach().cpu()),
        "composition_loss": float(composition_loss.detach().cpu()),
        "program_loss": float(program_loss.detach().cpu()),
        "smooth_loss": float(smooth.detach().cpu()),
    }


def _evaluate_deep(model, slide_batches: list[dict[str, Any]], tasks: dict[str, Any], torch_module) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    comp_true = []
    comp_pred = []
    mix_true = []
    mix_pred = []
    prog_true = []
    prog_pred = []
    per_slide = []
    with torch_module.no_grad():
        for batch in slide_batches:
            outputs = model(batch["patches"], batch["coords"], batch["edge_index"])
            pred_comp = outputs["compartment_logits"].argmax(dim=1).cpu().numpy()
            pred_mix = outputs["composition_logits"].softmax(dim=1).cpu().numpy()
            pred_prog = outputs["program_values"].cpu().numpy()
            true_comp = batch["compartment_target"].cpu().numpy()
            true_mix = batch["composition_target"].cpu().numpy()
            true_prog = batch["program_target"].cpu().numpy()
            comp_mask = batch["compartment_mask"].cpu().numpy() > 0
            mix_mask = batch["composition_mask"].cpu().numpy() > 0
            prog_mask = batch["program_mask"].cpu().numpy() > 0
            if comp_mask.any():
                comp_true.append(true_comp[comp_mask])
                comp_pred.append(pred_comp[comp_mask])
            if mix_mask.any():
                mix_true.append(true_mix[mix_mask])
                mix_pred.append(pred_mix[mix_mask])
            if prog_mask.any():
                prog_true.append(true_prog[prog_mask])
                prog_pred.append(pred_prog[prog_mask])
            per_slide.append(
                {
                    "slide_id": batch["slide_id"],
                    "n_tiles": int(len(true_comp)),
                    "compartment_macro_f1": macro_f1(true_comp[comp_mask], pred_comp[comp_mask]) if comp_mask.any() else 0.0,
                }
            )
    comp_true_all = np.concatenate(comp_true) if comp_true else np.asarray([])
    comp_pred_all = np.concatenate(comp_pred) if comp_pred else np.asarray([])
    mix_true_all = np.vstack(mix_true) if mix_true else np.zeros((0, len(tasks["composition_classes"])))
    mix_pred_all = np.vstack(mix_pred) if mix_pred else np.zeros((0, len(tasks["composition_classes"])))
    prog_true_all = np.vstack(prog_true) if prog_true else np.zeros((0, len(tasks["programs"])))
    prog_pred_all = np.vstack(prog_pred) if prog_pred else np.zeros((0, len(tasks["programs"])))
    metrics = {
        "macro_f1": macro_f1(comp_true_all, comp_pred_all) if len(comp_true_all) else 0.0,
        "balanced_accuracy": balanced_accuracy(comp_true_all, comp_pred_all) if len(comp_true_all) else 0.0,
        "composition_mean_pearson": pearson_mean(mix_true_all, mix_pred_all) if len(mix_true_all) else 0.0,
        "composition_js_divergence": js_divergence(mix_true_all, mix_pred_all) if len(mix_true_all) else 0.0,
        "program_mean_pearson": pearson_mean(prog_true_all, prog_pred_all) if len(prog_true_all) else 0.0,
    }
    return metrics, per_slide


def _train_deep_model(config: dict[str, Any], frame: pd.DataFrame, checkpoint_dir: Path, report_dir: Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - explicit runtime guard
        raise RuntimeError("deep_spatial_multitask requires torch to be installed") from exc

    train_frame, val_frame = _split_frame(frame)
    if train_frame.empty or val_frame.empty:
        raise ValueError("training requires non-empty train and val splits")

    device = config["training"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    tasks = config["tasks"]
    spec = build_model_spec(config) | {"model_family": config["training"].get("model_family", "deep_spatial_multitask")}
    model = DeepSpatialMultiTaskModel(spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loss_weights = {key: float(value) for key, value in config["training"]["loss_weights"].items()}
    smoothing_radius = float(config["training"].get("spatial_smoothing_radius_um", 160.0))

    train_batches = [
        _prepare_slide_batch(slide_frame.reset_index(drop=True), tasks["compartment_classes"], torch, device)
        for _, slide_frame in train_frame.groupby("slide_id")
    ]
    val_batches = [
        _prepare_slide_batch(slide_frame.reset_index(drop=True), tasks["compartment_classes"], torch, device)
        for _, slide_frame in val_frame.groupby("slide_id")
    ]

    best_metrics = None
    best_state = None
    train_history = []
    val_history = []
    epochs = int(config["training"]["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_batches:
            optimizer.zero_grad()
            outputs = model(batch["patches"], batch["coords"], batch["edge_index"])
            loss, parts = _deep_losses(outputs, batch, loss_weights, torch)
            loss.backward()
            if float(config["training"]["grad_clip_norm"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip_norm"]))
            optimizer.step()
            epoch_losses.append({"loss": float(loss.detach().cpu()), **parts})
        avg_train_loss = float(np.mean([item["loss"] for item in epoch_losses])) if epoch_losses else 0.0
        val_metrics, per_slide = _evaluate_deep(model, val_batches, tasks, torch)
        score = val_metrics["macro_f1"] + 0.5 * val_metrics["composition_mean_pearson"] + 0.25 * val_metrics["program_mean_pearson"]
        train_history.append({"epoch": epoch, "train_loss": avg_train_loss})
        val_history.append({"epoch": epoch, **val_metrics})
        if best_metrics is None or score > best_metrics["selection_score"]:
            best_metrics = {"selection_score": score, **val_metrics}
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_per_slide = per_slide

    payload = {
        "model_family": "deep_spatial_multitask",
        "model_spec": spec,
        "state_dict": best_state,
        "composition_classes": tasks["composition_classes"],
        "programs": tasks["programs"],
        "compartment_classes": tasks["compartment_classes"],
        "val_metrics": best_metrics,
        "spatial_smoothing_radius_um": smoothing_radius,
    }
    torch.save(payload, checkpoint_dir / "best.pt")
    torch.save(payload, checkpoint_dir / "last.pt")
    pd.DataFrame(train_history).to_csv(report_dir / "train_metrics.csv", index=False)
    pd.DataFrame(val_history).to_csv(report_dir / "val_metrics.csv", index=False)
    summary = {
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "model_family": "deep_spatial_multitask",
        "val_metrics": best_metrics,
        "per_slide_val_metrics": best_per_slide,
        "device": device,
    }
    write_json(report_dir / "train_summary.json", summary)
    return summary


def _train_classical_model(config: dict[str, Any], frame: pd.DataFrame, checkpoint_dir: Path, report_dir: Path) -> dict[str, Any]:
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
    comp_mask_train = train_frame["teacher_mask_compartment"].to_numpy(dtype=np.float32) > 0
    comp_mask_val = val_frame["teacher_mask_compartment"].to_numpy(dtype=np.float32) > 0
    mix_mask_train = train_frame["teacher_mask_composition"].to_numpy(dtype=np.float32) > 0
    mix_mask_val = val_frame["teacher_mask_composition"].to_numpy(dtype=np.float32) > 0
    prog_mask_train = train_frame["teacher_mask_program"].to_numpy(dtype=np.float32) > 0
    prog_mask_val = val_frame["teacher_mask_program"].to_numpy(dtype=np.float32) > 0
    smoothing_radius = float(config["training"].get("spatial_smoothing_radius_um", 160.0))

    if comp_mask_train.any() and comp_mask_val.any():
        best_compartment, comp_metrics = _select_compartment_model(
            x_train[comp_mask_train],
            y_comp_train[comp_mask_train],
            x_val[comp_mask_val],
            y_comp_val[comp_mask_val],
            smoothing_radius,
        )
    else:
        default_label = str(y_comp_train[0]) if len(y_comp_train) else config["tasks"]["compartment_classes"][0]
        best_compartment = {"name": "constant", "model": {"constant": default_label}}
        comp_metrics = {"macro_f1": 0.0, "balanced_accuracy": 0.0}
    best_composition, compo_score = _select_regression_model(
        x_train[mix_mask_train],
        y_mix_train[mix_mask_train],
        train_frame.loc[mix_mask_train, "teacher_confidence_composition"].to_numpy(dtype=np.float32),
        x_val[mix_mask_val],
        y_mix_val[mix_mask_val],
        "composition",
        smoothing_radius,
    )
    best_program, prog_score = _select_regression_model(
        x_train[prog_mask_train],
        y_prog_train[prog_mask_train],
        train_frame.loc[prog_mask_train, "teacher_confidence_program"].to_numpy(dtype=np.float32),
        x_val[prog_mask_val],
        y_prog_val[prog_mask_val],
        "program",
        smoothing_radius,
    )

    comp_val_pred = (
        _smooth_compartments(x_val[:, -5:-3], _predict_centroid_classifier(x_val, best_compartment["model"], "nearest_centroid"), radius=smoothing_radius)
        if best_compartment["name"] == "nearest_centroid"
        else _smooth_compartments(
            x_val[:, -5:-3],
            _predict_centroid_classifier(x_val, best_compartment["model"], "gaussian_diag")
            if best_compartment["name"] == "gaussian_diag"
            else (
                np.asarray([best_compartment["model"]["constant"]] * len(x_val))
                if best_compartment["name"] == "constant"
                else _predict_linear_classifier(x_val, best_compartment["model"])
            ),
            radius=smoothing_radius,
        )
    )
    mix_val_pred = _predict_regression(x_val, best_composition["model"], "composition", smoothing_radius=smoothing_radius)
    prog_val_pred = _predict_regression(x_val, best_program["model"], "program", smoothing_radius=smoothing_radius)

    val_metrics = {
        "macro_f1": macro_f1(y_comp_val[comp_mask_val], comp_val_pred[comp_mask_val]) if comp_mask_val.any() else 0.0,
        "balanced_accuracy": balanced_accuracy(y_comp_val[comp_mask_val], comp_val_pred[comp_mask_val]) if comp_mask_val.any() else 0.0,
        "composition_mean_pearson": pearson_mean(y_mix_val[mix_mask_val], mix_val_pred[mix_mask_val]) if mix_mask_val.any() else 0.0,
        "composition_js_divergence": js_divergence(y_mix_val[mix_mask_val], mix_val_pred[mix_mask_val]) if mix_mask_val.any() else 0.0,
        "program_mean_pearson": pearson_mean(y_prog_val[prog_mask_val], prog_val_pred[prog_mask_val]) if prog_mask_val.any() else 0.0,
    }
    model_payload = {
        "model_family": "enhanced_classical_ensemble",
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
        "compartment_classes": config["tasks"]["compartment_classes"],
        "val_metrics": val_metrics,
        "spatial_smoothing_radius_um": smoothing_radius,
    }
    _save_pickle(checkpoint_dir / "best.pt", model_payload)
    _save_pickle(checkpoint_dir / "last.pt", model_payload)
    pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_rows": int(len(train_frame)),
                "val_rows": int(len(val_frame)),
                "compartment_model": best_compartment["name"],
                "composition_model": best_composition["name"],
                "program_model": best_program["name"],
                "composition_val_score": compo_score,
                "program_val_score": prog_score,
            }
        ]
    ).to_csv(report_dir / "train_metrics.csv", index=False)
    pd.DataFrame([{"epoch": 1, **val_metrics}]).to_csv(report_dir / "val_metrics.csv", index=False)
    summary = {
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "model_family": "enhanced_classical_ensemble",
        "val_metrics": val_metrics,
    }
    write_json(report_dir / "train_summary.json", summary)
    return summary


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
        "model_family": str(training.get("model_family", "deep_spatial_multitask")),
        "dry_run": dry_run,
    }
    if dry_run:
        return summary

    ensure_directory(checkpoint_dir)
    ensure_directory(report_dir)
    frame = read_frame(cfg["tile_dataset_path"])
    frame = frame[frame["qc_pass"]].reset_index(drop=True)

    model_family = str(training.get("model_family", "deep_spatial_multitask"))
    if model_family == "deep_spatial_multitask":
        result = _train_deep_model(config, frame, checkpoint_dir, report_dir)
    else:
        result = _train_classical_model(config, frame, checkpoint_dir, report_dir)
    summary.update(result)
    return summary
