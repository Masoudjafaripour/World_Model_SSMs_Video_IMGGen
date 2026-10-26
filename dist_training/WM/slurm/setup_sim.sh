#!/bin/bash
#SBATCH --job-name=wm_setup_sim
#SBATCH --account=aip-zaiane
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/masod110/World_Model_SSMs_Video_IMGGen/dist_training/WM/logs/%x_%j.out
# Install the LIBERO simulator stack into the WM venv, then render one episode
# headless on a GPU (sim_test.py). robosuite is not in the wheelhouse, so it comes
# from PyPI via the proxy, without dependencies; its deps come from the wheelhouse.
set -euo pipefail
WM_SRC=${SLURM_SUBMIT_DIR:-$(pwd)}
source "$WM_SRC/slurm/sim_env.sh"
module list 2>&1 | tail -4

pip install --no-index numba scipy termcolor bddl easydict future cloudpickle imageio
pip install --no-deps robosuite==1.4.1 pynput
python "$WM_SRC/sim_test.py"
