# OncoScape

**HEST-compatible breast specialization layer for high-confidence spatial maps from H&E WSI**

OncoScape is a simple HPC-first proposal for breast cancer spatial pathology. It does **not** try to replace HEST. Instead, it adds breast-specific teacher integration, ontology, rendering, and reporting on top of a HEST-compatible workflow.

## Goal

Use breast cancer H&E WSI as input and produce only the spatial maps that are most likely to be accurate in a first release:

1. `compartment map`
2. `broad cell-type / TME map`
3. `program/activity map`
4. `uncertainty map`
5. `slide-level summary`

## What v3 deliberately does not do

- full transcriptome prediction
- pan-cancer foundation model training from scratch
- diagnosis support
- local-PC full training

## Required data

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
```

## Minimal steps

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

Run strict holdout evaluation and render maps for pathology and wet-lab review.

## Scope of this repository

This repository is currently a **v3 scaffold and specification-first implementation target**. The code is intentionally kept minimal while the design is narrowed to the highest-confidence maps first.
