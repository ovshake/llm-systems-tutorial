#!/bin/bash
#SBATCH --job-name=multinode-transformer
#SBATCH --partition=defq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

# ===========================================================================
# 2 nodes x 8 H100 = 16-way DDP of a real ~1.3B transformer on synthetic data.
# Same launch mechanics as multinode.slurm (one torchrun/node, c10d rendezvous
# on the head node) — only the entrypoint and step count differ.
#
# Submit FROM THIS DIRECTORY:   sbatch transformer.slurm
# ===========================================================================

set -euo pipefail

REPO_ROOT=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / GPUS_PER_NODE > 0 ? ${SLURM_CPUS_PER_TASK:-8} / GPUS_PER_NODE : 1 ))
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # cut allocator fragmentation
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
# If NCCL grabs the wrong NIC for the cross-node hop, pin it (names via `ip -br link`):
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5

echo "job=$SLURM_JOB_ID nodes=$SLURM_NNODES nodelist=$SLURM_JOB_NODELIST"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT gpus/node=$GPUS_PER_NODE omp=$OMP_NUM_THREADS"

# Default model is ~1.3B (24L / 2048d / 16h / seq 2048). Scale via the flags below;
# add --grad-checkpoint if you push the model big enough to pressure 80GB.
srun --kill-on-bad-exit=1 torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --max-restarts=0 \
    train_transformer.py --steps 100
