#!/bin/bash
#SBATCH --job-name=wm_multinode
#SBATCH --account=aip-zaiane
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/masod110/World_Model_SSMs_Video_IMGGen/dist_training/WM/logs/%x_%j.out
# Multi-node DDP: one torchrun per node, 4 GPUs each, rendezvous on the first node.
#   sbatch slurm/multinode.sh smoke_test.py                       # interconnect + busbw check
#   sbatch --nodes=2 --time=12:00:00 slurm/multinode.sh train_bc.py --run b_2n --size B
set -euo pipefail
WM_SRC=${SLURM_SUBMIT_DIR:-$(pwd)}
source "$WM_SRC/slurm/env.sh"
SCRIPT=$1; shift

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export OMP_NUM_THREADS=4
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

echo "nodes: $SLURM_JOB_NODELIST  master: $MASTER_ADDR:$MASTER_PORT"
# what the fabric is: InfiniBand devices (if any) and network links on each node
srun --ntasks-per-node=1 bash -c 'echo "== $(hostname): ib=[$(ls /sys/class/infiniband 2>/dev/null | tr "\n" " ")] links=[$(ip -br link | awk "{print \$1}" | tr "\n" " ")]"'

cd "$WM_SRC"
srun --ntasks-per-node=1 torchrun \
    --nnodes="$SLURM_NNODES" --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$SCRIPT" "$@"
