# Multi-node training smoke test (2 × 8×H100 via SLURM)

A minimal, self-contained demo whose only job is to answer one question:
**does data-parallel training actually work across two nodes?**

There is no dataset and no real model worth keeping. A small MLP *student* learns to mimic a
frozen random-MLP *teacher* on synthetic random inputs, with each of the 16 ranks (2 nodes ×
8 GPUs) drawing its own random batch. The value is in the four explicit checks it prints:

1. **Rank join** — all 16 ranks form one NCCL group; it lists which ranks landed on which host.
2. **Cross-node collective** — an `all_reduce` over all ranks returns the correct sum.
3. **DDP grad-sync** — gathers a gradient checksum from every rank and asserts they're identical
   (this is the bit that actually crosses the node boundary every step).
4. **Loss falls** — forward/backward/optimizer step works end-to-end.

If you see `MULTI-NODE TRAINING OK` at the bottom of the log, multi-node works.

## Files

| File              | What it is                                                        |
|-------------------|-------------------------------------------------------------------|
| `train.py`              | Toy-MLP DDP trainer + the four checks. Backend-agnostic (nccl/gloo). |
| `multinode.slurm`       | SLURM script for `train.py`: 2 nodes, `srun` → one `torchrun` per node. |
| `transformer.py`        | Self-contained ~1.3B decoder (RMSNorm + RoPE + FlashAttention + SwiGLU + tied embeds). |
| `train_transformer.py`  | Same four checks, but trains the real transformer + reports tokens/s and MFU. |
| `transformer.slurm`     | SLURM script for the transformer run.                             |
| `logs/`                 | `--output` lands here (`logs/<jobname>-<jobid>.out`).             |

## Run it

From **this directory**, with the repo venv built (`uv pip install -e .` at the repo root):

```bash
cd multinode-demo
sbatch multinode.slurm
squeue --me                                   # wait for state R
tail -f logs/multinode-demo-*.out             # watch it
```

The job grabs 2 nodes on `defq`, 8 GPUs each, and runs 200 steps (~seconds of compute once
scheduled). Edit `--steps`/sizes on the last line of `multinode.slurm`, or the `#SBATCH` lines
to change node count / partition / time.

### Test the logic first without burning an allocation

`train.py` falls back to the `gloo` backend on CPU and runs under a local rendezvous, so you can
validate everything except NCCL/GPU on a login node:

```bash
torchrun --standalone --nproc-per-node=2 train.py \
    --steps 12 --batch 64 --in-dim 64 --out-dim 64 --hidden 128 --layers 3 --log-every 3
```

(It will correctly print the "only ONE host" warning — that branch is *expected* on a single box.)

## Expected output (real 2-node run)

```
=== world_size=16 across 2 node(s) ===
  dgx-120: ranks 0..7  (8 gpus)
  dgx-149: ranks 8..15  (8 gpus)
  OK: 2 distinct hosts — genuine multi-node.
  OK: cross-node all_reduce(sum of ranks) = 120 (expected 120)
student=42.0M params | per-rank batch=4096 | global batch=65536 | steps=200
  OK: DDP grad-sync — first-layer grad-sum is identical across all 16 ranks (spread=0.00e+00)
step     0 | loss  ...
...
throughput: ... samples/s (199 timed steps, ...s, 16 gpus)
====================================================== 
 MULTI-NODE TRAINING OK — ranks joined, grads synced, loss fell.
======================================================
```

With `NCCL_DEBUG=INFO` you'll also see NCCL log lines like `NET/IB` (good — the inter-node hop is
on InfiniBand) or `NET/Socket` (it fell back to TCP — slower, but still correct).

## How the launch works

`--ntasks-per-node=1` means `srun` starts exactly **one `torchrun` per node**. Each `torchrun`
spawns 8 local workers (one per GPU) and they all join a single **c10d rendezvous** hosted on the
head node at `MASTER_ADDR:MASTER_PORT` → one 16-rank process group. `torchrun` is what sets
`RANK` / `LOCAL_RANK` / `WORLD_SIZE` / `MASTER_ADDR` / `MASTER_PORT`, which `init_dist()` reads.

```
sbatch ──> node A: srun task ──> torchrun ──> 8 workers (ranks 0..7)
       └─> node B: srun task ──> torchrun ──> 8 workers (ranks 8..15)
                                    └── c10d rendezvous @ head node ──> world_size=16
```

## Real transformer (~1.3B) across the 2 nodes

Same launch mechanics, real model:

```bash
sbatch transformer.slurm                          # 24L / 2048d / 16h, seq 1024, 16-way DDP
tail -f logs/multinode-transformer-*.out
```

Verified run (job 273293, dgx-[120,149], 100 steps):

```
model=1.34B params | seq_len=1024 | micro_batch=4/rank | global tokens/step=65,536
  OK: DDP grad-sync — a parameter's grad-sum is identical across all 16 ranks (spread=0.00e+00)
step    0 | loss  11.231 | step_ms 2401.6 | tok/s      27,289 | mfu  1.5%   <- warmup
step   20 | loss   1.932 | step_ms  180.9 | tok/s     362,200 | mfu 19.7%
step   99 | loss   0.054 | step_ms  181.1 | tok/s     361,790 | mfu 19.7%
avg over 99 steps: 326,105 tok/s | mfu 17.8% | loss 11.231 -> 0.054 | peak GPU mem 38.0 GB / ~80
```

Loss starts at ≈ln(vocab) and crashes toward 0 — the model learns the synthetic LCG next-token
rule, so the full forward/backward/all-reduce/step loop is proven across both nodes.

**Sizing note — DDP replicates the whole model on every GPU.** Each rank holds the full 1.3B
params + grads + Adam states (~21 GB) *plus* activations and the `(B, T, vocab)` logits, all on
one 80 GB card. The defaults peak at ~38 GB, leaving headroom. To push utilization (MFU ~20% here
is low because seq/batch are small and the 50k-vocab head is a big non-matmul chunk):

- raise `--micro-batch` / `--seq-len` in `transformer.slurm` (watch the printed peak mem),
- add `--grad-checkpoint` to trade compute for activation memory on bigger models,
- or move to **FSDP** (shards params/grads/optimizer across all 16 ranks) when one replica no
  longer fits — that's the next step up from this DDP demo.

Local logic smoke test (CPU/gloo, tiny — MFU will read 0%):

```bash
torchrun --standalone --nproc-per-node=2 train_transformer.py \
    --steps 8 --layers 2 --hidden 128 --heads 4 --intermediate 256 \
    --vocab 512 --seq-len 64 --micro-batch 4 --log-every 2
```

## Troubleshooting

- **Hangs at startup, no NCCL log past `INIT`** — almost always the inter-node NIC. Uncomment
  `NCCL_SOCKET_IFNAME` (set it to your fabric interface from `ip -br link`) and/or `NCCL_IB_HCA`
  in `multinode.slurm`. To prove the rest works while you debug the fabric, set `NCCL_IB_DISABLE=1`
  (forces TCP — slow but correct).
- **`only ONE host detected`** on the real run — you got 2 tasks on the same node; check
  `--nodes=2` and that the partition has 2 free nodes (`sinfo -p defq -t idle`).
- **Address already in use** — two jobs picked the same port; `MASTER_PORT` is derived from the
  job id so this is rare, but you can change the formula in the script.
- **`torchrun: command not found`** — the venv didn't activate; check the `REPO_ROOT` path at the
  top of `multinode.slurm` resolves to your repo's `.venv`.
- Want clean throughput numbers (no neighbors sharing the node's CPUs)? Add `#SBATCH --exclusive`.

> Note: the course (`COURSE.md`) otherwise treats multi-node as out of scope and targets a single
> 8×H100 node. This demo is the exception — a deliberate cross-node sanity check.
