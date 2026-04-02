# oncoscape

`oncoscape` is an HPC-first research pipeline for building breast cancer spatial biology models from public H&E whole-slide images and public spatial / single-cell references, with a Sphinx documentation site and a deep multitask training path.

The project documentation is now maintained as a Sphinx site under [`docs/`](C:\Users\daiki\Desktop\codex\oncoscape\docs). Start there for:

- installation
- public-data download setup
- source manifest preparation
- end-to-end pipeline execution
- CLI reference
- package API reference

## Quick Start

```bash
conda env create -f environment.yml
conda activate oncoscape
pip install -e .
pip install sphinx
python scripts/00_fetch_public_data.py --config configs/breast_hpc.yaml
python scripts/00_make_public_breast_manifest.py --config configs/breast_hpc.yaml
python scripts/00_preflight.py --config configs/breast_hpc.yaml
python scripts/08_run_pipeline.py --config configs/breast_hpc.yaml
```

The default HPC config now targets the `deep_spatial_multitask` training path.

For public breast runs, the intended order is:

1. fetch the public datasets defined in `configs/breast_downloads.template.yaml`
2. assemble a source manifest from the downloaded directory layout
3. run preflight and the full pipeline

The real-data ingestion path now supports:

- 10x Visium HDF5 matrices
- 10x / GEO sparse matrix directories
- Visium `spatial/` folders with `tissue_positions*.csv`
- Wu/GEO-style scRNA raw sample directories

## Validate The Pipeline

Run the lightweight accuracy-oriented regression tests before launching a large HPC job:

```bash
python -m unittest discover -s tests -v
```

The deep pipeline test exercises the real `register -> reference -> teachers -> patches -> train -> eval`
path on a synthetic dataset and requires strong held-out accuracy thresholds.

## Build Sphinx Documentation

```bash
sphinx-build -b html docs/source docs/_build/html
```

Open:

- `docs/_build/html/index.html`

## Repository Layout

```text
oncoscape/
  README.md
  docs/
  configs/
  ontology/
  scripts/
  src/oncoscape/
```
