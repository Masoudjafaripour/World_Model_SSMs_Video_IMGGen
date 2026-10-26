#!/bin/bash
# env.sh + the LIBERO simulator: MuJoCo from the module tree, headless EGL rendering,
# LIBERO from a clone on $SCRATCH (its config file lives there too, not in $HOME).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
module load gcc/12.3 opencv/4.11.0 mujoco/3.3.0   # 3.1.6 has no python3.12 build

export LIBERO_SRC=$SCRATCH/World_Model_SSMs_Video_IMGGen/LIBERO
export LIBERO_CONFIG_PATH=$SCRATCH/World_Model_SSMs_Video_IMGGen/libero_config
export PYTHONPATH=$LIBERO_SRC${PYTHONPATH:+:$PYTHONPATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
