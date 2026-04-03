#!/bin/bash
#SBATCH --job-name=oncoscape-breast
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /project/code/oncoscape
source ~/.bashrc
conda activate oncoscape
mkdir -p logs

python scripts/00_init_hpc_project.py --project-root /project --code-root /project/code/oncoscape
python scripts/00_fetch_public_data.py --config /project/run/config/breast_hpc.yaml
python scripts/00_make_public_breast_manifest.py --config /project/run/config/breast_hpc.yaml
python scripts/00_preflight.py --config /project/run/config/breast_hpc.yaml
python scripts/08_run_pipeline.py --config /project/run/config/breast_hpc.yaml
python scripts/09_select_best_seed.py --config /project/run/config/breast_hpc.yaml
