"""Hand-rolled DDP. No overlap — gradients are reduced serially after backward.

Launch:
    torchrun --standalone --nproc_per_node=8 ddp_from_scratch.py --config configs/small.yaml

The point of this script is to make the gradient all-reduce visible. The "real" DDP
trainer is train_ddp.py.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, "../01-transformer-from-scratch")
from data import TokenizedBinaryDataset  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from shared.utils.mfu import estimate_mfu  # noqa: E402
from shared.utils.nccl import init_dist  # noqa: E402


def manual_all_reduce_grads(model: torch.nn.Module, world: int) -> None:
    """All-reduce every gradient. No bucketing, no overlap. The clearest possible DDP."""
    for p in model.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    rank, world, local = init_dist()
    device = torch.device("cuda", local)
    torch.set_float32_matmul_precision("high")

    # Seed: weights must be identical across ranks. Data should be different.
    torch.manual_seed(cfg["seed"])
    model_cfg = GPTConfig(**cfg["model"])
    model = GPT(model_cfg).to(device)

    # Broadcast rank 0's weights to every rank to guarantee equality.
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], fused=True)

    # Each rank seeds its data sampler differently.
    torch.manual_seed(cfg["seed"] + 1000 * rank)
    train_ds = TokenizedBinaryDataset(cfg["data"]["train_path"], cfg["seq_len"])

    n_params = sum(p.numel() for p in model.parameters())

    for step in range(cfg["total_steps"]):
        t0 = time.perf_counter()
        x, y = train_ds.sample_batch(cfg["micro_batch"], device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()

        # All-reduce gradients across ranks.
        manual_all_reduce_grads(model, world)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if rank == 0 and step % 10 == 0:
            # global tokens/sec across ALL ranks
            global_tokens = cfg["micro_batch"] * cfg["seq_len"] * world
            tps = global_tokens / dt
            mfu = estimate_mfu(
                tokens_per_second=tps,
                n_params=n_params,
                seq_len=cfg["seq_len"],
                n_layers=model_cfg.n_layers,
                hidden_size=model_cfg.hidden_size,
                n_gpus=world,
            )
            print(
                f"step {step:>5d} | loss {loss.item():>7.3f} | "
                f"step_ms {dt * 1000:>6.1f} | tok/s {tps:>9.0f} | mfu {mfu * 100:>4.1f}%"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
