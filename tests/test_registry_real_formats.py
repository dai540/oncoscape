from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from oncoscape.data import build_public_breast_manifest, build_registry
from oncoscape.core import read_frame
from oncoscape.labels import build_teachers
from oncoscape.preprocessing import extract_patches_and_graphs
from oncoscape.reference import build_reference_atlas


def _write_fake_10x_h5(path: Path) -> None:
    data = np.asarray([1, 2, 3, 4], dtype=np.int64)
    indices = np.asarray([0, 2, 1, 2], dtype=np.int64)
    indptr = np.asarray([0, 2, 4], dtype=np.int64)
    shape = np.asarray([3, 2], dtype=np.int64)
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("data", data=data)
        matrix.create_dataset("indices", data=indices)
        matrix.create_dataset("indptr", data=indptr)
        matrix.create_dataset("shape", data=shape)
        matrix.create_dataset("barcodes", data=np.asarray([b"bc1", b"bc2"]))
        features = matrix.create_group("features")
        features.create_dataset("name", data=np.asarray([b"EPCAM", b"COL1A1", b"NKG7"]))
        features.create_dataset("id", data=np.asarray([b"EPCAM", b"COL1A1", b"NKG7"]))


class RegistryRealFormatsTest(unittest.TestCase):
    def test_visium_h5_and_spatial_folder_are_registered_and_scRNA_is_excluded_from_patches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_root = root / "data"
            run_root = root / "run"
            visium_root = data_root / "breast" / "tenx_visium_ff" / "V1_Breast_Cancer_Block_A_Section_1"
            spatial_root = visium_root / "spatial"
            spatial_root.mkdir(parents=True, exist_ok=True)
            _write_fake_10x_h5(visium_root / "V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5")
            pd.DataFrame(
                [
                    ["bc1", 1, 0, 0, 100, 200],
                    ["bc2", 1, 0, 1, 300, 400],
                ]
            ).to_csv(spatial_root / "tissue_positions_list.csv", index=False, header=False)
            Image.new("RGB", (600, 600), color=(255, 255, 255)).save(spatial_root / "tissue_hires_image.png")

            scrna_root = data_root / "breast" / "wu_scrna" / "GSE176078_RAW" / "CID3586"
            scrna_root.mkdir(parents=True, exist_ok=True)
            (scrna_root / "count_matrix_genes.tsv").write_text("EPCAM\nCOL1A1\nNKG7\n", encoding="utf-8")
            (scrna_root / "count_matrix_barcodes.tsv").write_text("cell1\ncell2\n", encoding="utf-8")
            (scrna_root / "count_matrix_sparse.mtx").write_text(
                "%%MatrixMarket matrix coordinate integer general\n%\n3 2 4\n1 1 5\n2 1 1\n2 2 4\n3 2 6\n",
                encoding="utf-8",
            )
            pd.DataFrame({"broad_cell_type": ["malignant", "T_NK"]}).to_csv(scrna_root / "metadata.csv", index=False)

            manifest_path = run_root / "config" / "breast_sources.yaml"
            config = {
                "paths": {"data_root": str(data_root), "run_root": str(run_root)},
                "fetch": {"output_root": str(data_root)},
                "registration": {"manifest_path": str(manifest_path), "output_dir": str(run_root / "data_interim")},
                "reference": {
                    "input_h5ad_paths": [str(run_root / "data_interim" / "adata" / "CID3586_reference.h5ad")],
                    "output_h5ad_path": str(run_root / "outputs" / "reference_atlas.h5ad"),
                    "output_markers_path": str(run_root / "outputs" / "reference_markers.csv"),
                    "output_qc_report_path": str(run_root / "outputs" / "reference_qc_report.html"),
                    "max_obs_per_input": 100,
                    "latent_dim": 4,
                    "marker_top_k": 3,
                    "broad_cell_types": ["malignant", "CAF", "endothelial", "myeloid", "T_NK", "B_plasma"],
                    "label_candidates": ["broad_cell_type"],
                },
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
                "teachers": {
                    "slides_csv_path": str(run_root / "data_interim" / "slides.csv"),
                    "reference_h5ad_path": str(run_root / "outputs" / "reference_atlas.h5ad"),
                    "output_h5ad_path": str(run_root / "data_interim" / "labels" / "teacher_labels.h5ad"),
                    "output_ontology_json": str(run_root / "data_interim" / "labels" / "ontology.json"),
                    "output_programs_json": str(run_root / "data_interim" / "labels" / "program_definitions.json"),
                    "visium_compartment_annotation_column": "Classification",
                    "teacher_confidence": {"pathology": 1.0, "xenium": 1.0, "visium": 1.0, "fallback": 0.5},
                },
                "graph": {"k": 1},
                "patch_extraction": {
                    "slides_csv_path": str(run_root / "data_interim" / "slides.csv"),
                    "teacher_labels_path": str(run_root / "data_interim" / "labels" / "teacher_labels.h5ad"),
                    "patches_dir": str(run_root / "data_interim" / "patches"),
                    "graphs_dir": str(run_root / "data_interim" / "graphs"),
                    "tile_dataset_path": str(run_root / "data_interim" / "tile_dataset.parquet"),
                    "tile_size_um": 112,
                    "target_mpp": 0.5,
                    "patch_size_px": 224,
                    "tissue_fraction_min": 0.0,
                    "patch_mean_min": 0.0,
                    "patch_std_min": 0.0,
                },
            }

            build_public_breast_manifest(config)
            build_registry(config)

            slides = pd.read_csv(run_root / "data_interim" / "slides.csv")
            self.assertEqual(set(slides["source_type"]), {"visium", "scrna"})
            visium_path = run_root / "data_interim" / "adata" / "tenx_visium_ff_block_a_section_1.h5ad"
            visium = ad.read_h5ad(visium_path)
            self.assertEqual(visium.n_obs, 2)
            self.assertAlmostEqual(float(visium.obs["x_um"].iloc[0]), 100.0, places=3)
            self.assertAlmostEqual(float(visium.obs["y_um"].iloc[0]), 50.0, places=3)

            build_reference_atlas(config)
            build_teachers(config)
            extract_patches_and_graphs(config)
            tiles = read_frame(run_root / "data_interim" / "tile_dataset.parquet")
            self.assertEqual(set(tiles["slide_id"]), {"tenx_visium_ff_block_a_section_1"})


if __name__ == "__main__":
    unittest.main()
