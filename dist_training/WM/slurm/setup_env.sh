#!/bin/bash
#SBATCH --job-name=wm_setup_env
#SBATCH --account=aip-zaiane
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/masod110/World_Model_SSMs_Video_IMGGen/dist_training/WM/logs/%x_%j.out
# Step 1: build the venv on $SCRATCH from the wheelhouse, then a CPU smoke test.
# Submit from dist_training/WM:  sbatch slurm/setup_env.sh
set -euo pipefail
WM_SRC=${SLURM_SUBMIT_DIR:-$(pwd)}
source "$WM_SRC/slurm/env.sh"

if [ ! -f "$WM_VENV/bin/activate" ]; then
    virtualenv --no-download "$WM_VENV"
fi
source "$WM_VENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$WM_SRC/requirements.txt"
pip freeze > "$WM_ROOT/logs/pip_freeze_${SLURM_JOB_ID}.txt"

python "$WM_SRC/smoke_test.py"
