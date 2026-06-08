"""Production-shaped DDP trainer with bucketed gradient overlap.

Launch:
    torchrun --standalone --nproc_per_node=8 train_ddp.py --config configs/small.yaml

This is the recommended pattern for any real DDP training. Diff against ddp_from_scratch.py
to see exactly what `DistributedDataParallel` is buying you — bucketing, overlap, an
optimized backward hook, gradient-as-bucket-view, etc.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, "../01-transformer-from-scratch")
from data import TokenizedBinaryDataset  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from shared.utils.mfu import estimate_mfu  # noqa: E402
from shared.utils.nccl import init_dist  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bucket-cap-mb", type=int, default=25)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    rank, world, local = init_dist()
    device = torch.device("cuda", local)
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(cfg["seed"])
    model_cfg = GPTConfig(**cfg["model"])
    model = GPT(model_cfg).to(device)
    model = DDP(
        model,
        device_ids=[local],
        bucket_cap_mb=args.bucket_cap_mb,
        gradient_as_bucket_view=True,        # avoid an extra copy when reducing grads
        static_graph=True,                   # allow more optimizations; safe for fixed-shape transformers
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], fused=True)

    torch.manual_seed(cfg["seed"] + 1000 * rank)
    train_ds = TokenizedBinaryDataset(cfg["data"]["train_path"], cfg["seq_len"])

    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(
            f"world={world} model params={n_params / 1e6:.1f}M bucket_cap_mb={args.bucket_cap_mb}"
        )

    accum = cfg.get("accum_steps", 1)

    for step in range(cfg["total_steps"]):
        t0 = time.perf_counter()
        loss_sum = 0.0
        for ms in range(accum):
            x, y = train_ds.sample_batch(cfg["micro_batch"], device)
            # `no_sync` skips the gradient all-reduce on non-last accumulation steps.
            cm = model.no_sync() if ms < accum - 1 else _noop_cm()
            with cm:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                (loss / accum).backward()
            loss_sum += loss.item() / accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if rank == 0 and step % 10 == 0:
            tokens = cfg["micro_batch"] * cfg["seq_len"] * world * accum
            tps = tokens / dt
            mfu = estimate_mfu(
                tokens_per_second=tps,
                n_params=n_params,
                seq_len=cfg["seq_len"],
                n_layers=model_cfg.n_layers,
                hidden_size=model_cfg.hidden_size,
                n_gpus=world,
            )
            print(
                f"step {step:>5d} | loss {loss_sum:>7.3f} | "
                f"step_ms {dt * 1000:>6.1f} | tok/s {tps:>9.0f} | mfu {mfu * 100:>4.1f}%"
            )

    dist.destroy_process_group()


class _noop_cm:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


if __name__ == "__main__":
    main()
