#!/bin/bash
#SBATCH --job-name=wm_train
#SBATCH --account=aip-zaiane
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/masod110/World_Model_SSMs_Video_IMGGen/dist_training/WM/logs/%x_%j.out
# Single-node BC training; scale by overriding the sbatch flags, e.g. 4 GPUs:
#   sbatch --gres=gpu:l40s:4 --cpus-per-task=32 --mem=160G --time=12:00:00 \
#          slurm/train.sh --run s_ddp4 --size S --max_minutes 700
# Arguments after the script name go to train_bc.py. Resubmitting the same --run resumes.
set -euo pipefail
WM_SRC=${SLURM_SUBMIT_DIR:-$(pwd)}
source "$WM_SRC/slurm/env.sh"
pip install --no-index -q -r "$WM_SRC/requirements.txt"   # no-op once installed

NGPU=$(nvidia-smi -L | wc -l)
export OMP_NUM_THREADS=4
cd "$WM_SRC"
torchrun --standalone --nproc_per_node="$NGPU" train_bc.py \
    --workers $(( SLURM_CPUS_PER_TASK / NGPU - 1 )) "$@"
