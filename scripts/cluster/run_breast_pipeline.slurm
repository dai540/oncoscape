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

python scripts/00_preflight.py --config configs/breast_hpc.yaml
python scripts/08_run_pipeline.py --config configs/breast_hpc.yaml
