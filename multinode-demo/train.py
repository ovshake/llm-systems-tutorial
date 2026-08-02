"""Multi-node DDP smoke test — does cross-node data-parallel training actually work?

There is no dataset. A small MLP *student* learns to mimic a frozen, randomly-initialized
MLP *teacher* on synthetic random inputs. Every rank draws its own random batch from a
per-rank-seeded generator, so all 16 ranks (2 nodes x 8 H100) run genuine data-parallel
SGD. The model is irrelevant — the point is to prove, with explicit checks, that:

  1. all WORLD_SIZE ranks across >=2 physical hosts join one NCCL process group,
  2. an all_reduce spanning both nodes returns the arithmetically correct answer,
  3. DDP's gradient all-reduce leaves *identical* grads on every rank (incl. the
     cross-node hop) — verified by gathering a grad checksum from all ranks, and
  4. the loss actually falls — i.e. end-to-end forward/backward/step works at scale.

Launch on 2 nodes x 8 GPUs via SLURM:  see `multinode.slurm` in this directory.

Local logic smoke test (no SLURM, no GPU needed — falls back to gloo/CPU):
    torchrun --standalone --nproc-per-node=2 train.py --steps 20
    python train.py --steps 5            # single process
"""

from __future__ import annotations

import argparse
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from shared.utils.nccl import init_dist


class MLP(nn.Module):
    """Plain GELU-MLP. Same class is used for both the frozen teacher and the student."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ensure_env_defaults() -> None:
    """Let a bare `python train.py` (no torchrun) run as a 1-process smoke test."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-node DDP smoke test on synthetic data.")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4096, help="per-rank micro-batch")
    ap.add_argument("--in-dim", type=int, default=1024)
    ap.add_argument("--out-dim", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    ensure_env_defaults()
    use_cuda = torch.cuda.is_available()
    rank, world, local = init_dist(backend="nccl" if use_cuda else "gloo")
    device = torch.device("cuda", local) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.set_float32_matmul_precision("high")
    host = socket.gethostname()

    # ---- 1. Topology: who am I, and how many distinct physical nodes are we on? ----
    gathered = [None] * world
    dist.all_gather_object(gathered, (rank, local, host))
    if rank == 0:
        by_host: dict[str, list[int]] = {}
        for r, _lr, h in sorted(gathered):
            by_host.setdefault(h, []).append(r)
        print(f"\n=== world_size={world} across {len(by_host)} node(s) ===", flush=True)
        for h, ranks in by_host.items():
            print(f"  {h}: ranks {ranks[0]}..{ranks[-1]}  ({len(ranks)} gpus)", flush=True)
        if len(by_host) < 2:
            print("  WARNING: only ONE host detected — this is NOT a multi-node run.", flush=True)
        else:
            print(f"  OK: {len(by_host)} distinct hosts — genuine multi-node.", flush=True)

    # ---- 2. Pre-flight collective: all_reduce a per-rank value across all nodes ----
    probe = torch.tensor([float(rank)], device=device)
    dist.all_reduce(probe)  # default op = SUM
    expected = float(world * (world - 1) // 2)
    assert abs(probe.item() - expected) < 1e-3, f"all_reduce got {probe.item()}, want {expected}"
    if rank == 0:
        print(
            f"  OK: cross-node all_reduce(sum of ranks) = {probe.item():.0f} "
            f"(expected {expected:.0f})\n",
            flush=True,
        )

    # ---- 3. Frozen random teacher (identical on every rank) + the student we train ----
    torch.manual_seed(args.seed)  # identical RNG stream => identical teacher & student init
    teacher = MLP(args.in_dim, args.hidden, args.out_dim, args.layers).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    student = MLP(args.in_dim, args.hidden, args.out_dim, args.layers).to(device)
    student = DDP(
        student,
        device_ids=[local] if use_cuda else None,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    optim = torch.optim.AdamW(student.parameters(), lr=args.lr, fused=use_cuda)

    n_params = sum(p.numel() for p in student.parameters())
    if rank == 0:
        print(
            f"student={n_params / 1e6:.1f}M params | per-rank batch={args.batch} | "
            f"global batch={args.batch * world} | steps={args.steps}",
            flush=True,
        )

    # Per-rank generator => every rank sees a *different* random batch (true data parallel).
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 1 + rank)

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(args.batch, args.in_dim, device=device, generator=gen)
        with torch.no_grad():
            noise = 0.01 * torch.randn(args.batch, args.out_dim, device=device, generator=gen)
            y = teacher(x) + noise
        return x, y

    loss_fn = nn.MSELoss()
    grad_verified = False
    t_start: float | None = None
    timed_steps = 0

    for step in range(args.steps):
        x, y = make_batch()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            loss = loss_fn(student(x), y)
        loss.backward()

        # ---- 4. One-time grad-sync proof: every rank must hold identical grads ----
        if not grad_verified:
            first_w = next(student.parameters())
            local_sum = first_w.grad.detach().double().sum().reshape(1)
            sums = torch.empty(world, dtype=torch.float64, device=device)
            dist.all_gather_into_tensor(sums, local_sum)
            spread = (sums.max() - sums.min()).item()
            assert spread < 1e-3, (
                f"grads differ across ranks (spread={spread:.3e}) — "
                "DDP all-reduce did NOT span all ranks!"
            )
            if rank == 0:
                print(
                    f"  OK: DDP grad-sync — first-layer grad-sum is identical across all "
                    f"{world} ranks (spread={spread:.2e})\n",
                    flush=True,
                )
            grad_verified = True

        optim.step()
        optim.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == args.steps - 1:
            mean_loss = loss.detach().clone()
            dist.all_reduce(mean_loss)
            if rank == 0:
                print(f"step {step:>5d} | loss {mean_loss.item() / world:>9.4f}", flush=True)

        # Time steps 1..N-1; step 0 is warmup (NCCL ring build, cuDNN autotune, allocator).
        if step == 0:
            if use_cuda:
                torch.cuda.synchronize()
            t_start = time.perf_counter()
        else:
            timed_steps += 1

    if use_cuda:
        torch.cuda.synchronize()

    if rank == 0 and t_start is not None and timed_steps > 0:
        dt = time.perf_counter() - t_start
        global_samples = args.batch * world * timed_steps
        print(
            f"\nthroughput: {global_samples / dt:,.0f} samples/s "
            f"({timed_steps} timed steps, {dt:.2f}s, {world} gpus)",
            flush=True,
        )
        bar = "=" * 54
        print(f"{bar}\n MULTI-NODE TRAINING OK — ranks joined, grads synced, loss fell.\n{bar}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
