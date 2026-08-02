#!/bin/bash
#SBATCH --job-name=multinode-demo
#SBATCH --partition=defq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
# (stderr is merged into the same file since --error is not set)

# ===========================================================================
# 2 nodes x 8 H100 = 16-rank DDP smoke test.
#
# One `torchrun` per node (--ntasks-per-node=1). Each torchrun fans out to 8
# local workers, and the c10d rendezvous hosted on the head node stitches the
# two nodes into a single 16-rank process group. torchrun is what populates
# RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT for each worker.
#
# Submit FROM THIS DIRECTORY:   sbatch multinode.slurm
# Watch:                        tail -f logs/multinode-demo-<jobid>.out
# ===========================================================================

set -euo pipefail

# Activate the repo venv (this script lives in <repo>/multinode-demo/).
REPO_ROOT=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

# --- Rendezvous endpoint: head node + a job-derived port (avoids collisions) ---
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
# One worker process per GPU; split the task's cores evenly so OMP doesn't oversubscribe.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / GPUS_PER_NODE > 0 ? ${SLURM_CPUS_PER_TASK:-8} / GPUS_PER_NODE : 1 ))
export PYTHONUNBUFFERED=1

# --- NCCL: fail fast, and show which transport it picks for the inter-node hop ---
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1     # don't hang forever if a rank dies
export NCCL_DEBUG=INFO                        # logs IB vs SOCKET, the rings/trees it builds
export NCCL_DEBUG_SUBSYS=INIT,NET             # ...trimmed to the useful subsystems
# If NCCL grabs the wrong NIC for the cross-node hop, pin it (names via `ip -br link`):
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5
# Last-resort fallbacks if the fabric is misconfigured (slow, but proves correctness):
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

echo "job=$SLURM_JOB_ID nodes=$SLURM_NNODES nodelist=$SLURM_JOB_NODELIST"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT gpus/node=$GPUS_PER_NODE omp=$OMP_NUM_THREADS"

srun --kill-on-bad-exit=1 torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --max-restarts=0 \
    train.py --steps 200
