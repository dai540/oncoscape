#!/bin/bash
#SBATCH --job-name=oncoscape
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/breast_hpc.yaml}"

python scripts/00_fetch_and_manifest.py --config "${CONFIG_PATH}"
python scripts/01_register_data.py --config "${CONFIG_PATH}"
python scripts/02_build_reference.py --config "${CONFIG_PATH}"
python scripts/03_build_teachers.py --config "${CONFIG_PATH}"
python scripts/04_extract_tiles.py --config "${CONFIG_PATH}"
python scripts/05_train_model.py --config "${CONFIG_PATH}"
python scripts/06_eval_and_render.py --config "${CONFIG_PATH}"
python scripts/07_run_biomarker_adapter.py --config "${CONFIG_PATH}"
