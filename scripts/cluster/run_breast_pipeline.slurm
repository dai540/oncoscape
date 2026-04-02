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

python scripts/01_download_and_register.py --config configs/breast_hpc.yaml
python scripts/02_build_reference_atlas.py --config configs/breast_hpc.yaml
python scripts/03_make_teachers.py --config configs/breast_hpc.yaml
python scripts/04_extract_patches_and_graphs.py --config configs/breast_hpc.yaml
python scripts/05_train_breast_model.py --config configs/breast_hpc.yaml
python scripts/06_eval_and_render.py --config configs/breast_hpc.yaml
python scripts/07_generate_reports.py --config configs/breast_hpc.yaml
