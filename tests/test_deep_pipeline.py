from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from oncoscape.data import build_registry
from oncoscape.evaluation import evaluate_and_render
from oncoscape.labels import build_teachers
from oncoscape.preprocessing import extract_patches_and_graphs
from oncoscape.reference import build_reference_atlas
from oncoscape.training import train_breast_model


CLASS_LAYOUT = [
    ("invasive_tumor", "malignant", (220, 30, 40)),
    ("stroma", "CAF", (40, 180, 90)),
    ("immune_rich", "T_NK", (50, 100, 220)),
]
GENES = ["EPCAM", "COL1A1", "PECAM1", "LYZ", "NKG7", "MS4A1", "MKI67", "VIM"]


def _counts_for_class(cell_type: str) -> list[float]:
    base = np.ones(len(GENES), dtype=np.float32)
    if cell_type == "malignant":
        base[[0, 6]] = 20
    elif cell_type == "CAF":
        base[[1, 7]] = 20
    elif cell_type == "T_NK":
        base[[4, 3]] = 20
    return base.tolist()


def _make_slide(root: Path, slide_id: str, patient_id: str, split: str) -> dict[str, str]:
    slide_dir = root / "data" / slide_id
    slide_dir.mkdir(parents=True, exist_ok=True)
    coords = []
    ann = []
    expr_rows = []
    image = Image.new("RGB", (1400, 1400), "white")
    draw = ImageDraw.Draw(image)
    tile_index = 0
    for row_idx in range(2):
        for col_idx, (compartment, cell_type, color) in enumerate(CLASS_LAYOUT):
            x_um = float(56 + col_idx * 112)
            y_um = float(56 + row_idx * 112)
            px = int(round(x_um / 0.5)) - 112
            py = int(round(y_um / 0.5)) - 112
            draw.rectangle([px, py, px + 223, py + 223], fill=color)
            coords.append({"x_um": x_um, "y_um": y_um})
            ann.append({"Classification": compartment, "broad_cell_type": cell_type})
            expr_rows.append(_counts_for_class(cell_type))
            tile_index += 1
    image_path = slide_dir / "image.png"
    image.save(image_path)
    pd.DataFrame(expr_rows, columns=GENES).to_csv(slide_dir / "counts.csv", index=False)
    pd.DataFrame(coords).to_csv(slide_dir / "coords.csv", index=False)
    pd.DataFrame(ann).to_csv(slide_dir / "ann.csv", index=False)
    return {
        "name": slide_id,
        "source_type": "visium",
        "platform": "visium_ffpe",
        "patient_id": patient_id,
        "slide_id": slide_id,
        "image_path": str(image_path),
        "counts_path": str(slide_dir / "counts.csv"),
        "annotation_path": str(slide_dir / "ann.csv"),
        "coord_path": str(slide_dir / "coords.csv"),
        "mpp_x": 0.5,
        "mpp_y": 0.5,
        "coord_unit": "micron",
        "split": split,
    }


class DeepPipelineTest(unittest.TestCase):
    def test_deep_pipeline_reaches_strong_synthetic_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "run"
            (run_dir / "config").mkdir(parents=True, exist_ok=True)
            sources = {
                "sources": [
                    _make_slide(root, "train_slide_1", "train_patient_1", "train"),
                    _make_slide(root, "train_slide_2", "train_patient_2", "train"),
                    _make_slide(root, "val_slide_1", "val_patient_1", "val"),
                    _make_slide(root, "test_slide_1", "test_patient_1", "test"),
                ]
            }
            import yaml

            manifest_path = run_dir / "config" / "breast_sources.yaml"
            with manifest_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(sources, handle, sort_keys=False)

            config = {
                "project_name": "oncoscape-test",
                "tasks": {
                    "compartment_classes": [
                        "invasive_tumor",
                        "in_situ_tumor",
                        "stroma",
                        "immune_rich",
                        "adipose_normal",
                        "necrosis_background",
                    ],
                    "composition_classes": ["malignant", "CAF", "endothelial", "myeloid", "T_NK", "B_plasma"],
                    "programs": ["proliferation", "EMT", "hypoxia", "IFN_gamma", "inflammatory", "angiogenesis", "TGFb_like", "antigen_presentation"],
                },
                "graph": {"method": "knn", "k": 2},
                "registration": {
                    "output_dir": str(run_dir / "data_interim"),
                    "manifest_path": str(manifest_path),
                },
                "reference": {
                    "input_h5ad_paths": [
                        str(run_dir / "data_interim" / "adata" / "train_slide_1.h5ad"),
                        str(run_dir / "data_interim" / "adata" / "train_slide_2.h5ad"),
                        str(run_dir / "data_interim" / "adata" / "val_slide_1.h5ad"),
                    ],
                    "output_h5ad_path": str(run_dir / "outputs" / "reference_atlas.h5ad"),
                    "output_markers_path": str(run_dir / "outputs" / "reference_markers.csv"),
                    "output_qc_report_path": str(run_dir / "outputs" / "reference_qc_report.html"),
                    "max_obs_per_input": 1000,
                    "latent_dim": 8,
                    "marker_top_k": 5,
                    "broad_cell_types": ["malignant", "CAF", "endothelial", "myeloid", "T_NK", "B_plasma"],
                    "label_candidates": ["broad_cell_type"],
                },
                "teachers": {
                    "slides_csv_path": str(run_dir / "data_interim" / "slides.csv"),
                    "input_adata_dir": str(run_dir / "data_interim" / "adata"),
                    "reference_h5ad_path": str(run_dir / "outputs" / "reference_atlas.h5ad"),
                    "output_h5ad_path": str(run_dir / "data_interim" / "labels" / "teacher_labels.h5ad"),
                    "output_ontology_json": str(run_dir / "data_interim" / "labels" / "ontology.json"),
                    "output_programs_json": str(run_dir / "data_interim" / "labels" / "program_definitions.json"),
                    "visium_compartment_annotation_column": "Classification",
                    "teacher_confidence": {"pathology": 1.0, "xenium": 1.0, "visium": 1.0, "fallback": 0.5},
                    "use_cell2location": True,
                    "use_scvi_label_transfer": True,
                    "program_method": "gene_set_mean",
                },
                "patch_extraction": {
                    "slides_csv_path": str(run_dir / "data_interim" / "slides.csv"),
                    "teacher_labels_path": str(run_dir / "data_interim" / "labels" / "teacher_labels.h5ad"),
                    "patches_dir": str(run_dir / "data_interim" / "patches"),
                    "graphs_dir": str(run_dir / "data_interim" / "graphs"),
                    "tile_dataset_path": str(run_dir / "data_interim" / "tile_dataset.parquet"),
                    "tile_size_um": 112,
                    "target_mpp": 0.5,
                    "patch_size_px": 224,
                    "tissue_fraction_min": 0.01,
                    "patch_mean_min": 0.0,
                    "patch_std_min": 0.0,
                },
                "train_run": {
                    "tile_dataset_path": str(run_dir / "data_interim" / "tile_dataset.parquet"),
                    "checkpoint_dir": str(run_dir / "outputs" / "checkpoints"),
                    "report_dir": str(run_dir / "outputs" / "reports"),
                },
                "training": {
                    "model_family": "deep_spatial_multitask",
                    "epochs": 30,
                    "lr": 2.0e-3,
                    "weight_decay": 1.0e-4,
                    "grad_clip_norm": 1.0,
                    "device": "cpu",
                    "encoder_name": "simple_cnn",
                    "encoder_pretrained": False,
                    "encoder_out_dim": 64,
                    "hidden_dim": 64,
                    "spatial_num_layers": 2,
                    "dropout": 0.0,
                    "encoder_batch_tiles": 64,
                    "spatial_smoothing_radius_um": 40.0,
                    "loss_weights": {"compartment": 2.0, "composition": 2.0, "program": 1.0, "smooth": 0.05},
                },
                "evaluation": {"split": "test"},
                "render": {
                    "checkpoint_path": str(run_dir / "outputs" / "checkpoints" / "best.pt"),
                    "predictions_dir": str(run_dir / "outputs" / "predictions"),
                    "report_dir": str(run_dir / "outputs" / "reports"),
                    "render_tile_px": 18,
                },
                "reporting": {
                    "executive_summary_path": str(run_dir / "outputs" / "reports" / "executive_summary.json"),
                    "wet_lab_summary_path": str(run_dir / "outputs" / "reports" / "wet_lab_summary.json"),
                    "developer_summary_path": str(run_dir / "outputs" / "reports" / "developer_summary.json"),
                },
            }

            build_registry(config)
            build_reference_atlas(config)
            build_teachers(config)
            extract_patches_and_graphs(config)
            train_breast_model(config)
            result = evaluate_and_render(config)

            self.assertGreaterEqual(result["compartment_macro_f1"], 0.80)
            self.assertGreaterEqual(result["composition_mean_pearson"], 0.70)
            self.assertGreaterEqual(result["program_mean_pearson"], 0.70)

            metrics_path = run_dir / "outputs" / "reports" / "test_metrics.json"
            self.assertTrue(metrics_path.exists())
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            self.assertGreaterEqual(metrics["compartment_macro_f1"], 0.80)


if __name__ == "__main__":
    unittest.main()
