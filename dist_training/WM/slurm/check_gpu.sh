#!/bin/bash
#SBATCH --job-name=wm_check_gpu
#SBATCH --account=aip-zaiane
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/masod110/World_Model_SSMs_Video_IMGGen/dist_training/WM/logs/%x_%j.out
# Step 1b: the same smoke test on a full node (CUDA, bf16, SDPA, 4-GPU NCCL all-reduce).
# Submit from dist_training/WM:  sbatch slurm/check_gpu.sh
set -euo pipefail
WM_SRC=${SLURM_SUBMIT_DIR:-$(pwd)}
source "$WM_SRC/slurm/env.sh"

nvidia-smi
nvidia-smi topo -m
torchrun --standalone --nproc_per_node=4 "$WM_SRC/smoke_test.py"
