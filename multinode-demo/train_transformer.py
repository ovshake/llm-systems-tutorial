"""Multi-node DDP training of a REAL transformer (~1.3B by default) on synthetic data.

Same cross-node verification scaffold as train.py (rank-join map, cross-node all_reduce,
DDP grad-sync proof) but the model is a genuine decoder transformer (transformer.py:
RMSNorm + RoPE + FlashAttention + SwiGLU + tied embeddings) and the metrics are the ones a
systems engineer actually watches: loss, step time, tokens/s, and MFU.

No dataset: every sequence is a deterministic LCG chain  x_{t+1} = (A*x_t + C) mod V  seeded
by a random start token, drawn per-rank. The next-token rule is learnable, so the loss falls
from ~ln(V) toward 0 — proving the full forward/backward/step loop works end to end. The LCG
is just "random numbers with structure"; MFU (a compute metric) is meaningful regardless.

Launch on 2 nodes x 8 H100 via SLURM:   sbatch transformer.slurm
Local logic smoke test (CPU/gloo, tiny):
    torchrun --standalone --nproc-per-node=2 train_transformer.py \
        --steps 8 --layers 2 --hidden 128 --heads 4 --intermediate 256 \
        --vocab 512 --seq-len 64 --micro-batch 4 --log-every 2
"""

from __future__ import annotations

import argparse
import os
import socket
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from shared.utils.mfu import estimate_mfu
from shared.utils.nccl import init_dist
from transformer import GPT, GPTConfig, num_params

# Classic LCG constants (glibc). next = (A * cur + C) mod vocab.
_LCG_A, _LCG_C = 1103515245, 12345


def ensure_env_defaults() -> None:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-node DDP transformer smoke test (synthetic).")
    ap.add_argument("--steps", type=int, default=100)
    # DDP replicates the whole model on every GPU (params+grads+Adam ~21GB for 1.3B), so the
    # per-rank batch/seq must leave room for activations + the (B,T,vocab) logits on one 80GB
    # card. These defaults peak around ~35GB; raise them (or add --grad-checkpoint) to fill more.
    ap.add_argument("--micro-batch", type=int, default=4, help="per-rank sequences per step")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=50304)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=None)
    ap.add_argument("--intermediate", type=int, default=5632)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--bucket-cap-mb", type=int, default=40)
    ap.add_argument("--grad-checkpoint", action="store_true")
    args = ap.parse_args()

    ensure_env_defaults()
    use_cuda = torch.cuda.is_available()
    rank, world, local = init_dist(backend="nccl" if use_cuda else "gloo")
    device = torch.device("cuda", local) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.set_float32_matmul_precision("high")
    host = socket.gethostname()

    # ---- 1. Topology: how many distinct physical nodes are we on? ----
    gathered = [None] * world
    dist.all_gather_object(gathered, (rank, local, host))
    if rank == 0:
        by_host: dict[str, list[int]] = {}
        for r, _lr, h in sorted(gathered):
            by_host.setdefault(h, []).append(r)
        print(f"\n=== world_size={world} across {len(by_host)} node(s) ===", flush=True)
        for h, ranks in by_host.items():
            print(f"  {h}: ranks {ranks[0]}..{ranks[-1]}  ({len(ranks)} gpus)", flush=True)
        msg = (
            f"OK: {len(by_host)} distinct hosts — genuine multi-node."
            if len(by_host) >= 2
            else "WARNING: only ONE host — this is NOT a multi-node run."
        )
        print(f"  {msg}", flush=True)

    # ---- 2. Pre-flight cross-node collective ----
    probe = torch.tensor([float(rank)], device=device)
    dist.all_reduce(probe)
    expected = float(world * (world - 1) // 2)
    assert abs(probe.item() - expected) < 1e-3, f"all_reduce got {probe.item()}, want {expected}"
    if rank == 0:
        print(f"  OK: cross-node all_reduce(sum of ranks) = {probe.item():.0f} "
              f"(expected {expected:.0f})\n", flush=True)

    # ---- 3. Build the real transformer + DDP ----
    torch.manual_seed(args.seed)  # identical init on every rank (DDP also broadcasts at ctor)
    cfg = GPTConfig(
        vocab_size=args.vocab, n_layers=args.layers, n_heads=args.heads,
        n_kv_heads=args.kv_heads, hidden_size=args.hidden,
        intermediate_size=args.intermediate, context_length=args.seq_len,
        use_grad_checkpoint=args.grad_checkpoint,
    )
    model = GPT(cfg).to(device)
    n_params = num_params(model)
    model = DDP(
        model,
        device_ids=[local] if use_cuda else None,
        bucket_cap_mb=args.bucket_cap_mb,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=use_cuda)

    if rank == 0:
        gtok = args.micro_batch * args.seq_len * world
        print(f"model={n_params / 1e9:.2f}B params | seq_len={args.seq_len} | "
              f"micro_batch={args.micro_batch}/rank | global tokens/step={gtok:,} | "
              f"steps={args.steps}\n", flush=True)

    # ---- Per-rank synthetic LCG token stream (random start, deterministic rule) ----
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 1 + rank)
    # Precompute the next-token lookup table once: nxt[v] = (A*v + C) mod vocab.
    nxt = (_LCG_A * torch.arange(args.vocab, device=device) + _LCG_C) % args.vocab

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        B, T = args.micro_batch, args.seq_len
        seq = torch.empty(B, T + 1, dtype=torch.long, device=device)
        seq[:, 0] = torch.randint(0, args.vocab, (B,), device=device, generator=gen)
        for t in range(T):
            seq[:, t + 1] = nxt[seq[:, t]]  # cheap gather; LCG is a fixed map
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

    grad_verified = False
    timed_steps = 0
    timed_elapsed = 0.0
    first_loss: float | None = None
    last_loss = float("nan")
    tokens_per_step = args.micro_batch * args.seq_len * world

    for step in range(args.steps):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        x, y = make_batch()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            _, loss = model(x, y)
        loss.backward()

        # ---- 4. One-time grad-sync proof across all ranks (incl. the cross-node hop) ----
        if not grad_verified:
            g = next(p for p in model.parameters() if p.grad is not None)
            local_sum = g.grad.detach().double().sum().reshape(1)
            sums = torch.empty(world, dtype=torch.float64, device=device)
            dist.all_gather_into_tensor(sums, local_sum)
            spread = (sums.max() - sums.min()).item()
            assert spread < 1e-2, (
                f"grads differ across ranks (spread={spread:.3e}) — DDP did not sync all ranks!"
            )
            if rank == 0:
                print(f"  OK: DDP grad-sync — a parameter's grad-sum is identical across all "
                      f"{world} ranks (spread={spread:.2e})\n", flush=True)
            grad_verified = True

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        if use_cuda:
            torch.cuda.synchronize()
        step_time = time.perf_counter() - t0
        if step >= 1:  # step 0 is warmup (NCCL ring build, Flash/cuDNN autotune, allocator)
            timed_steps += 1
            timed_elapsed += step_time

        if step % args.log_every == 0 or step == args.steps - 1:
            lt = loss.detach().clone()
            dist.all_reduce(lt)
            mean_loss = lt.item() / world
            last_loss = mean_loss
            if first_loss is None:
                first_loss = mean_loss
            if rank == 0:
                tps = tokens_per_step / step_time
                mfu = estimate_mfu(tps, n_params, args.seq_len, args.layers, args.hidden, world)
                print(f"step {step:>4d} | loss {mean_loss:>7.3f} | "
                      f"step_ms {step_time * 1000:>7.1f} | "
                      f"tok/s {tps:>11,.0f} | mfu {mfu * 100:>4.1f}%", flush=True)

    if rank == 0 and timed_steps > 0:
        tps = (tokens_per_step * timed_steps) / timed_elapsed
        mfu = estimate_mfu(tps, n_params, args.seq_len, args.layers, args.hidden, world)
        peak = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
        print(f"\navg over {timed_steps} steps: {tps:,.0f} tok/s | mfu {mfu * 100:.1f}% | "
              f"loss {first_loss:.3f} -> {last_loss:.3f} | peak GPU mem {peak:.1f} GB / ~80",
              flush=True)
        bar = "=" * 60
        print(f"{bar}\n MULTI-NODE TRANSFORMER OK — {world} ranks across nodes, grads synced.\n{bar}",
              flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
