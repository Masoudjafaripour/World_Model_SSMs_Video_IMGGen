#!/bin/bash
# Sourced by every WM job: one module stack + venv, and all heavy paths on $SCRATCH.
module --force purge
module load StdEnv/2023 python/3.12.4 arrow/25.0.0

export WM_ROOT=$SCRATCH/World_Model_SSMs_Video_IMGGen/dist_training/WM
export WM_DATA=$WM_ROOT/data
export WM_RUNS=$WM_ROOT/runs
export WM_VENV=$SCRATCH/World_Model_SSMs_Video_IMGGen/wm_venv

export HF_HOME=$SCRATCH/hf_cache
export XDG_CACHE_HOME=$SCRATCH/cache
export WANDB_DIR=$WM_RUNS
export PIP_CACHE_DIR=$SCRATCH/cache/pip

if [ -f "$WM_VENV/bin/activate" ]; then
    source "$WM_VENV/bin/activate"
fi
