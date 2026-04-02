# oncoscape

`oncoscape` is an HPC-first research pipeline for building breast cancer spatial biology models from public H&E whole-slide images and public spatial / single-cell references.

This repository is designed for:

- public-data-only research
- breast cancer H&E WSI input
- HPC or GPU server execution
- reproducible end-to-end pipeline execution
- spatial prediction of:
  - compartment
  - composition
  - programs

This repository is not intended for full-scale local-PC model training.

## Project Goal

Build a reproducible HPC pipeline that:

1. downloads and registers public breast datasets on cluster storage
2. builds a breast scRNA reference atlas
3. builds weak teachers from Visium, Xenium, and pathology metadata
4. extracts 112 um image tiles and spatial graphs
5. trains a multi-task model
6. renders WSI-level prediction maps and reports

## Repository Layout

```text
oncoscape/
  README.md
  pyproject.toml
  environment.yml
  .gitignore
  configs/
  ontology/
  scripts/
    cluster/
  src/oncoscape/
```

## Standard Pipeline

### 1. Register public sources

```bash
python scripts/01_download_and_register.py --config configs/breast_hpc.yaml
```

Outputs:

- `run/data_interim/slides.csv`
- `run/data_interim/gene_symbol_map.csv`
- `run/data_interim/adata/*.h5ad`

### 2. Build breast reference atlas

```bash
python scripts/02_build_reference_atlas.py --config configs/breast_hpc.yaml
```

Outputs:

- `run/outputs/reference_atlas.h5ad`
- `run/outputs/reference_markers.csv`
- `run/outputs/reference_qc_report.html`

### 3. Build teachers

```bash
python scripts/03_make_teachers.py --config configs/breast_hpc.yaml
```

Outputs:

- `run/data_interim/labels/teacher_labels.h5ad`
- `run/data_interim/labels/ontology.json`
- `run/data_interim/labels/program_definitions.json`

### 4. Extract patches and graphs

```bash
python scripts/04_extract_patches_and_graphs.py --config configs/breast_hpc.yaml
```

Outputs:

- `run/data_interim/patches/*.png`
- `run/data_interim/graphs/*.pt`
- `run/data_interim/tile_dataset.parquet`

### 5. Train

```bash
python scripts/05_train_breast_model.py --config configs/breast_hpc.yaml
```

Outputs:

- `run/outputs/checkpoints/best.pt`
- `run/outputs/checkpoints/last.pt`
- `run/outputs/reports/train_metrics.csv`
- `run/outputs/reports/val_metrics.csv`
- `run/outputs/reports/train_summary.json`

### 6. Evaluate and render

```bash
python scripts/06_eval_and_render.py --config configs/breast_hpc.yaml
```

Outputs:

- `run/outputs/reports/test_metrics.json`
- `run/outputs/reports/per_slide_metrics.csv`
- `run/outputs/predictions/<slide_id>/tile_predictions.parquet`
- `run/outputs/predictions/<slide_id>/compartment_map.png`
- `run/outputs/predictions/<slide_id>/composition_maps/*.png`
- `run/outputs/predictions/<slide_id>/program_maps/*.png`

## Public Data Requirements

Required public datasets:

- Wu 2021 breast spatial atlas
- 10x Visium FFPE human breast cancer
- 10x Visium fresh frozen human breast cancer
- 10x Xenium FFPE human breast
- GSE176078
- GSE161529

Recommended addition:

- GSE235326

All public data should be downloaded directly to HPC storage, not committed to Git.

## HPC Directory Convention

```text
/project/
  code/oncoscape/
  data/
    breast/
      wu2021_visium/
      tenx_visium_ffpe/
      tenx_visium_ff/
      tenx_xenium_breast/
      wu_scrna/
      gse161529/
      gse235326/
  run/
    config/
    data_interim/
      adata/
      labels/
      patches/
      graphs/
    outputs/
      checkpoints/
      reports/
      predictions/
```

## Highest-Accuracy Guidance

To maximize accuracy, prioritize improvements in this order:

1. better pathology-reviewed teachers
2. stricter evaluation design
3. stronger training setup

Recommended training conditions:

- pretrained encoder
- GPU execution
- patient-level and source-level holdout
- larger breast scRNA reference atlas
- Visium plus Xenium teachers
- no tutorial-scale downsampling unless needed for debugging

## Environment

Recommended:

- Python 3.11
- CUDA-capable PyTorch
- 24 GB or larger GPU
- 16+ CPU cores
- 128+ GB RAM
- fast scratch storage

Install:

```bash
conda env create -f environment.yml
conda activate oncoscape
pip install -e .
```

## Cluster Launch

Before running the pipeline, prepare the two required config files:

- use `configs/breast_hpc.yaml` as the HPC pipeline template
- copy `configs/breast_sources.template.yaml` to `/project/run/config/breast_sources.yaml`
- replace every placeholder path in `/project/run/config/breast_sources.yaml` with the real public-data locations on your cluster

The default `breast_hpc.yaml` expects:

- code at `/project/code/oncoscape`
- public data under `/project/data/breast`
- run outputs under `/project/run`

Edit:

- `configs/breast_hpc.yaml`
- `scripts/cluster/run_breast_pipeline.slurm`

Then submit:

```bash
sbatch scripts/cluster/run_breast_pipeline.slurm
```

## Current Repository Status

This repository is a clean rebuild scaffold aligned to the v2.0 requirements document.

It provides:

- project structure
- config system
- manifest schema
- CLI entry points for the 6-step pipeline
- HPC execution template

It is intended as the new base for full implementation and cluster execution.
