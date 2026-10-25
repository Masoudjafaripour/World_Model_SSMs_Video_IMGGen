#!/bin/bash
#SBATCH --job-name=wm_libero_download
#SBATCH --account=aip-zaiane
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/masod110/World_Model_SSMs_Video_IMGGen/dist_training/WM/logs/%x_%j.out
# Step 2a: download LIBERO (~35 GB) + the SigLIP encoder via the cluster proxy.
# Step 2b (encode) is submitted with GPU flags on top of this script:
#   J=$(sbatch --parsable slurm/prepare_libero.sh)
#   sbatch --dependency=afterok:$J --job-name=wm_libero_encode --gres=gpu:l40s:1 \
#          --cpus-per-task=16 --mem=48G --time=02:00:00 slurm/prepare_libero.sh encode
set -euo pipefail
WM_SRC=${SLURM_SUBMIT_DIR:-$(pwd)}
source "$WM_SRC/slurm/env.sh"
python "$WM_SRC/prepare_libero.py" --stage "${1:-download}"
