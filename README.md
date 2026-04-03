# oncoscape

`oncoscape` is an HPC-first research pipeline for building breast cancer spatial biology models from public H&E whole-slide images and public spatial / single-cell references, with a Sphinx documentation site and a deep multitask training path.

The project documentation is now maintained as a Sphinx site under [`docs/`](C:\Users\daiki\Desktop\codex\oncoscape\docs). Start there for:

- installation
- public-data download setup
- source manifest preparation
- end-to-end pipeline execution
- CLI reference
- package API reference

## HPC Quick Start

```bash
git clone https://github.com/dai540/oncoscape.git /project/code/oncoscape
cd /project/code/oncoscape
conda env create -f environment.yml
conda activate oncoscape
pip install -e .
pip install sphinx
python scripts/00_init_hpc_project.py --project-root /project --code-root /project/code/oncoscape
python scripts/00_fetch_public_data.py --config /project/run/config/breast_hpc.yaml
python scripts/00_make_public_breast_manifest.py --config /project/run/config/breast_hpc.yaml
python scripts/00_preflight.py --config /project/run/config/breast_hpc.yaml
python scripts/08_run_pipeline.py --config /project/run/config/breast_hpc.yaml
python scripts/09_select_best_seed.py --config /project/run/config/breast_hpc.yaml
```

`00_init_hpc_project.py` creates the run-time config and directories under `/project/run`. The generated
`/project/run/config/breast_hpc.yaml` is the config you should use for jobs.

The default HPC config targets the `deep_spatial_multitask` training path.

For strict holdout experiments, run multiple seeds into a shared `outputs/seed_sweep/` root, then use
`09_select_best_seed.py` to choose the canonical model by validation score and promote its checkpoints,
reports, and predictions into the standard output locations.

For public breast runs, the intended order is:

1. initialize the project layout and generated config under `/project/run/config`
2. fetch the public datasets defined in `/project/run/config/breast_downloads.yaml`
3. assemble a source manifest from the downloaded directory layout
4. run preflight and the full pipeline

If you already placed public datasets on the cluster, set `--data-root` when running
`00_init_hpc_project.py` and skip the fetch step.

## Clean Clone Validation

Run these checks on a fresh clone before submitting a long HPC job:

```bash
python scripts/00_init_hpc_project.py --project-root /project --code-root /project/code/oncoscape
python scripts/00_make_public_breast_manifest.py --config /project/run/config/breast_hpc.yaml
python scripts/00_preflight.py --config /project/run/config/breast_hpc.yaml
python scripts/08_run_pipeline.py --config /project/run/config/breast_hpc.yaml --dry-run
```

If `00_preflight.py` fails, fix the generated config or the public-data layout before launching the full run.

Launch the full HPC job with:

```bash
sbatch scripts/cluster/run_breast_pipeline.slurm
```

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
