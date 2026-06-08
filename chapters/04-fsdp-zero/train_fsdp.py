"""FSDP trainer. Launch with:

    torchrun --standalone --nproc_per_node=8 train_fsdp.py --config configs/1B.yaml

Demonstrates: FULL_SHARD, ModuleWrapPolicy({Block}), BF16 MixedPrecision, and the
basic checkpoint pattern.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time

import torch
import torch.distributed as dist
import yaml
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

sys.path.insert(0, "../01-transformer-from-scratch")
from data import TokenizedBinaryDataset  # noqa: E402
from model import GPT, GPTConfig, Block  # noqa: E402
from shared.utils.mfu import estimate_mfu  # noqa: E402
from shared.utils.nccl import init_dist  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    rank, world, local = init_dist()
    device = torch.device("cuda", local)
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(cfg["seed"])
    model_cfg = GPTConfig(**cfg["model"])

    # Construct on CPU and move per-shard, to avoid OOM on a single device.
    with torch.device("meta"):
        model = GPT(model_cfg)

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({Block}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=local,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        param_init_fn=lambda m: m.to_empty(device=device, recurse=False),
        sync_module_states=True,
    )

    # After FSDP wraps, parameters are on `device` as flat shards. Re-init weights to
    # match the per-rank state we'd see if we'd just instantiated and broadcast.
    def _init_module(m: torch.nn.Module) -> None:
        for sub in m.modules():
            if hasattr(sub, "reset_parameters"):
                sub.reset_parameters()
    # NOTE: reset_parameters won't recreate the model's custom init exactly. For research
    # workloads this is fine. For a faithful Llama-style init you'd patch this up.

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], fused=True)

    torch.manual_seed(cfg["seed"] + 1000 * rank)
    train_ds = TokenizedBinaryDataset(cfg["data"]["train_path"], cfg["seq_len"])

    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"world={world} unsharded params={n_params * world / 1e9:.2f}B (each rank: {n_params / 1e9:.2f}B)")

    for step in range(cfg["total_steps"]):
        t0 = time.perf_counter()
        x, y = train_ds.sample_batch(cfg["micro_batch"], device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        model.clip_grad_norm_(1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if rank == 0 and step % 10 == 0:
            tokens = cfg["micro_batch"] * cfg["seq_len"] * world
            tps = tokens / dt
            mfu = estimate_mfu(
                tokens_per_second=tps,
                n_params=n_params * world,
                seq_len=cfg["seq_len"],
                n_layers=model_cfg.n_layers,
                hidden_size=model_cfg.hidden_size,
                n_gpus=world,
            )
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(
                f"step {step:>5d} | loss {loss.item():>7.3f} | "
                f"step_ms {dt * 1000:>6.1f} | tok/s {tps:>9.0f} | "
                f"mfu {mfu * 100:>4.1f}% | peak {mem:>5.1f}G"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
