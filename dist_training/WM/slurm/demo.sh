#!/bin/bash
#SBATCH --job-name=wm_demo
#SBATCH --account=aip-zaiane
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/masod110/World_Model_SSMs_Video_IMGGen/dist_training/WM/logs/%x_%j.out
# Offline demos of a trained run:  sbatch slurm/demo.sh --run r1_S_1gpu
set -euo pipefail
WM_SRC=${SLURM_SUBMIT_DIR:-$(pwd)}
source "$WM_SRC/slurm/env.sh"
cd "$WM_SRC"
python demo.py "$@"
