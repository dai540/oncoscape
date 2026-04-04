from __future__ import annotations

from typing import Any


def fetch_and_manifest_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": "00_fetch_and_manifest",
        "goal": "download public breast datasets and build a source manifest",
        "inputs": {
            "data_root": config["paths"]["data_root"],
            "source_template": "configs/breast_sources.template.yaml",
        },
        "external_dependencies": [
            "HEST",
            "public breast datasets",
        ],
        "outputs": {
            "manifest": f"{config['paths']['run_root']}/breast_sources.yaml",
        },
    }


def register_data_plan(config: dict[str, Any]) -> dict[str, Any]:
    registry_dir = config["outputs"]["registry_dir"]
    return {
        "step": "01_register_data",
        "goal": "normalize source files into a HEST-compatible breast registry",
        "inputs": {
            "manifest": f"{config['paths']['run_root']}/breast_sources.yaml",
        },
        "outputs": {
            "slides_csv": f"{registry_dir}/slides.csv",
            "gene_symbol_map": f"{registry_dir}/gene_symbol_map.csv",
            "adata_dir": f"{registry_dir}/adata",
        },
    }


def build_reference_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": "02_build_reference",
        "goal": "build a breast broad-cell-type reference with scvi-tools",
        "inputs": [
            "gse176078",
            "gse161529",
        ],
        "outputs": {
            "reference_dir": config["outputs"]["reference_dir"],
            "atlas": f"{config['outputs']['reference_dir']}/reference_atlas.h5ad",
        },
    }


def build_teacher_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": "03_build_teachers",
        "goal": "build breast-specific compartment and broad-TME teachers",
        "teacher_sources": {
            "compartment": ["Wu pathology-reviewed metadata"],
            "broad_tme": ["10x Xenium breast", "Visium + cell2location"],
            "programs": ["breast program panel"],
        },
        "outputs": {
            "teacher_dir": config["outputs"]["teacher_dir"],
            "teacher_labels": f"{config['outputs']['teacher_dir']}/teacher_labels.h5ad",
        },
    }


def extract_tiles_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": "04_extract_tiles",
        "goal": "extract H&E tiles and build spatial graphs",
        "tiling": config["tiling"],
        "outputs": {
            "tile_dir": config["outputs"]["tile_dir"],
            "tile_dataset": f"{config['outputs']['tile_dir']}/tile_dataset.parquet",
            "graph_dir": f"{config['outputs']['tile_dir']}/graphs",
        },
    }


def train_model_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": "05_train_model",
        "goal": "declare the planned training contract for the H&E-to-spatial-state model",
        "model": {
            "strategy": config["model"]["strategy"],
            "encoder": config["model"]["encoder"],
            "spatial_module": config["model"]["spatial_module"],
            "primary_outputs": config["targets"]["primary"],
            "secondary_outputs": config["targets"]["secondary"],
        },
        "outputs": {
            "checkpoint_dir": config["outputs"]["checkpoint_dir"],
            "best_checkpoint": f"{config['outputs']['checkpoint_dir']}/best.pt",
        },
    }


def evaluate_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": "06_eval_and_render",
        "goal": "declare the planned evaluation, rendering, and feature export contract",
        "metrics": config["evaluation"]["metrics"],
        "maps": [
            "compartment_map",
            "broad_tme_map",
            "program_map",
            "uncertainty_map",
        ],
        "outputs": {
            "prediction_dir": config["outputs"]["prediction_dir"],
            "report_dir": config["outputs"]["report_dir"],
            "biomarker_feature_table": config["outputs"]["biomarker_feature_table"],
        },
    }


def biomarker_feature_table_spec(config: dict[str, Any]) -> dict[str, Any]:
    feature_table = config["feature_table"]
    return {
        "unit": feature_table["unit"],
        "output_path": config["outputs"]["biomarker_feature_table"],
        "required_id_columns": feature_table["required_id_columns"],
        "feature_groups": {
            "compartment": feature_table["compartment_features"],
            "composition": feature_table["composition_features"],
            "programs": feature_table["program_features"],
            "topology": feature_table["topology_features"],
            "hotspots": feature_table["hotspot_features"],
            "uncertainty": feature_table["uncertainty_features"],
        },
    }


def biomarker_adapter_spec(config: dict[str, Any]) -> dict[str, Any]:
    adapter = config["adapter"]
    return {
        "step": "07_run_biomarker_adapter",
        "goal": "declare the downstream predictive biomarker handoff contract",
        "role": adapter["role"],
        "supported_framework_families": adapter["supported_framework_families"],
        "required_inputs": adapter["required_inputs"],
        "upstream_handoff": config["outputs"]["biomarker_feature_table"],
        "outputs": adapter["outputs"],
        "output_dir": config["outputs"]["adapter_dir"],
    }
