# oncoscape

`oncoscape` is an HPC-first research pipeline for building breast cancer spatial biology models from public H&E whole-slide images and public spatial / single-cell references.

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
python scripts/00_preflight.py --config configs/breast_hpc.yaml
python scripts/08_run_pipeline.py --config configs/breast_hpc.yaml
```

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
