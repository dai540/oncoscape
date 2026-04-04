# OncoScape

**HEST-compatible breast spatial state generator for H&E whole-slide images**

OncoScape is a breast-focused, HPC-first research layer built on top of existing spatial pathology tooling. It is **not** a replacement for HEST, and it is **not** a treatment-selection model by itself.

Its role is simpler and narrower:

- take breast cancer H&E WSI as input
- integrate public spatial and single-cell references during training
- generate high-confidence spatial state maps
- export structured slide-level features that can be passed to downstream predictive biomarker models

In other words, **OncoScape is an upstream spatial state generator**.

## What OncoScape does

OncoScape produces the spatial outputs that are most realistic to estimate with reasonable confidence in a first release:

1. `compartment map`
   - invasive_tumor
   - in_situ_tumor
   - stroma
   - immune_rich
   - adipose_normal
   - necrosis_background
2. `broad cell-type / TME map`
   - malignant
   - CAF
   - endothelial
   - myeloid
   - T_NK
   - B_plasma
3. `program/activity map`
   - proliferation
   - hypoxia
   - EMT
   - IFN_gamma
   - angiogenesis
   - inflammatory
4. `uncertainty map`
5. `slide-level summary`
6. `biomarker_feature_table`
   - a structured feature table derived from the spatial outputs
   - intended for downstream predictive biomarker modeling

## What OncoScape does not do

OncoScape does not directly perform:

- full transcriptome prediction
- pan-cancer foundation model training from scratch
- diagnosis support
- treatment recommendation
- response prediction as a final clinical model

These belong to downstream models, not to OncoScape core.

## Design principle

OncoScape follows a simple rule:

**Use existing strong tools whenever possible, and only implement the missing breast-specific integration layer.**

The project is designed around:

- `HEST` for compatible task structure and benchmarking concepts
- `scvi-tools` for reference integration and transfer
- `cell2location` for Visium-based composition teachers
- `OpenSlide` for WSI access
- a pathology foundation encoder for image features

## Why this project exists

Existing tools already cover many important pieces of the pipeline, but they do not directly provide a breast-specific, HPC-oriented layer that combines:

- public breast spatial data
- pathology-reviewed compartment teachers
- Xenium-enhanced broad TME teachers
- breast-specific ontology
- wet-lab and pathology-oriented outputs
- exportable biomarker features for downstream predictive modeling

That is the role of OncoScape.

## Core idea

OncoScape should be used like this:

- **OncoScape core**
  - generates spatial maps and structured spatial state features from H&E
- **Downstream biomarker adapter**
  - converts those features into input for predictive biomarker models such as PBMF-like or SELECT/ENLIGHT-like frameworks

This separation is intentional. It keeps the spatial modeling problem distinct from the treatment-benefit modeling problem.

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

## Minimal repository layout

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
  src/oncoscape/
    bootstrap/
    cli/
    core/
    data/
    validation/
    reference/
    labels/
    preprocessing/
    models/
    training/
    evaluation/
    rendering/
    reporting/
    adapter/
```

## Core steps

### 00 Fetch and manifest

Download public breast datasets and assemble a source manifest.

### 01 Register data

Create `slides.csv`, `gene_symbol_map.csv`, and registered `h5ad` files.

### 02 Build reference

Build a breast-specific broad-cell-type reference atlas from scRNA data.

### 03 Build teachers

Build pathology-reviewed compartment labels and spatial composition teachers.

### 04 Extract tiles

Extract 112 um tiles, run QC, and construct spatial graphs.

### 05 Train model

Train a compartment-first, broad-TME-first model using a pathology foundation encoder.

### 06 Evaluate and render

Run strict holdout evaluation, render maps, and export `biomarker_feature_table`.

### 07 Run biomarker adapter

Transform `biomarker_feature_table` into downstream predictive biomarker model inputs.

## `biomarker_feature_table`

This table is the formal handoff from OncoScape core to downstream biomarker modeling.

Minimum output path:

- `outputs/reports/biomarker_feature_table.parquet`

Minimum key columns:

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

## Responsibility split

### OncoScape core

Owns:

- data fetch
- registry
- reference atlas building
- teacher building
- tile extraction
- spatial modeling
- map generation
- uncertainty estimation
- slide summary generation
- biomarker feature table generation

Does not own:

- treatment benefit prediction
- therapy response classification
- clinical thresholding
- deployable assay lock-down

### Biomarker adapter

Owns:

- consuming `biomarker_feature_table`
- joining clinical metadata
- preparing treated/control or responder/non-responder labels
- fitting downstream predictive biomarker models
- clinical utility evaluation

## Scope of this repository

This repository is a **HEST-compatible breast specialization layer** and an **upstream spatial feature generator**. It should remain narrow, composable, and easy to connect to existing downstream biomarker pipelines.
