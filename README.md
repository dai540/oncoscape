# OncoScape

`OncoScape` is an HPC-oriented script set for breast cancer spatial pathology.

It is built around one short idea:

- use strong existing tools such as `HEST`, `scvi-tools`, and `cell2location`
- define a clear breast-specific contract for spatial map generation
- hand off a `biomarker_feature_table` to downstream predictive biomarker workflows

`OncoScape` is **not** an end-to-end model implementation in this repository. It is a **script-first, specification-first HPC scaffold**.

## What it is

`OncoScape core` is an upstream H&E-to-spatial-state layer. It defines how to:

- fetch and register public breast spatial data
- build breast-specific references and teachers
- define planned tile extraction, training, and evaluation contracts
- export `biomarker_feature_table`

`OncoScape adapter` is a downstream handoff contract. It defines how to:

- take `biomarker_feature_table`
- join clinical metadata and endpoints
- prepare inputs for PBMF-like or SELECT/ENLIGHT-like biomarker models

## What it is not

- full transcriptome prediction code
- pan-cancer foundation model training code
- treatment recommendation software
- diagnosis software
- a local-PC training package

## Required data

Mandatory public datasets:

- Wu 2021 breast spatial atlas
- 10x Visium FFPE breast cancer
- 10x Visium fresh frozen breast cancer
- 10x Xenium breast cancer
- GSE176078
- GSE161529

Recommended:

- GSE235326

## Required external tools

- `HEST`
- `scvi-tools`
- `cell2location`
- `OpenSlide`
- a pathology foundation encoder

## Installation

Create an environment and install the lightweight scaffold dependencies:

```bash
conda env create -f environment.yml
conda activate oncoscape
pip install -e .
```

This repository intentionally keeps Python dependencies light because the real heavy lifting is expected to happen through external tools on HPC.

To build the documentation:

```bash
pip install -e .[docs]
sphinx-build -b html docs/source docs/_build/html
```

## Quick start

Run the numbered scripts in order:

```bash
python scripts/00_fetch_and_manifest.py --config configs/breast_hpc.yaml
python scripts/01_register_data.py --config configs/breast_hpc.yaml
python scripts/02_build_reference.py --config configs/breast_hpc.yaml
python scripts/03_build_teachers.py --config configs/breast_hpc.yaml
python scripts/04_extract_tiles.py --config configs/breast_hpc.yaml
python scripts/05_train_model.py --config configs/breast_hpc.yaml
python scripts/06_eval_and_render.py --config configs/breast_hpc.yaml
python scripts/07_run_biomarker_adapter.py --config configs/breast_hpc.yaml
```

Each script currently prints the **planned contract** for that stage. That keeps the repository honest about its current status while fixing the HPC execution interface.

## SLURM example

```bash
sbatch scripts/submit_oncoscape.slurm
```

Edit the config path, environment name, and project paths before use.

## Main outputs

Planned `OncoScape core` outputs:

- `compartment_map`
- `broad_tme_map`
- `program_map`
- `uncertainty_map`
- `slide_summary`
- `biomarker_feature_table`

Planned `OncoScape adapter` outputs:

- `biomarker_score`
- `response_probability`
- `subgroup_report`
- `utility_metrics`

## biomarker_feature_table

The `biomarker_feature_table` is the formal handoff from spatial modeling to downstream biomarker modeling.

Minimum columns:

- `slide_id`
- `patient_id`
- `source`
- `split`
- `n_tiles_total`
- `n_tiles_qc_pass`

Minimum feature groups:

- compartment fractions
- broad TME means
- program burdens
- spatial interaction features
- hotspot features
- uncertainty-aware features

## Repository layout

```text
oncoscape/
  README.md
  pyproject.toml
  environment.yml
  configs/
    breast_hpc.yaml
    breast_sources.template.yaml
  ontology/
    breast.yaml
    breast_programs.yaml
  scripts/
    00_fetch_and_manifest.py
    01_register_data.py
    02_build_reference.py
    03_build_teachers.py
    04_extract_tiles.py
    05_train_model.py
    06_eval_and_render.py
    07_run_biomarker_adapter.py
    submit_oncoscape.slurm
  src/oncoscape/
    __init__.py
    plans.py
    cli/
  docs/
    source/
```

## Current status

This repository is a **script set for HPC execution planning**, not a finished spatial modeling engine.

What is already fixed:

- breast-specific scope
- required datasets
- core versus adapter split
- `biomarker_feature_table` contract
- numbered step-by-step scripts

What is intentionally not included here:

- the full model implementation
- the full training engine
- the full evaluation engine

## Author

`OncoScape` is maintained by Dai.
