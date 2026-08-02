"""DeepSpeed ZeRO trainer — the ZeRO-{1,2,3} counterpart to train_fsdp.py (Exercise 5).

Install DeepSpeed first (it's an optional dep, not in the base environment):

    uv pip install deepspeed

Launch with DeepSpeed's own launcher (it sets RANK / LOCAL_RANK / WORLD_SIZE):

    deepspeed --num_gpus=8 train_zero.py --config configs/1B.yaml --deepspeed_config configs/zero1.json
    deepspeed --num_gpus=8 train_zero.py --config configs/1B.yaml --deepspeed_config configs/zero2.json
    deepspeed --num_gpus=8 train_zero.py --config configs/1B.yaml --deepspeed_config configs/zero3.json

Division of labour: the YAML owns the model shape, data path, micro_batch and seq_len;
the JSON owns the ZeRO stage, precision, and optimizer. We patch micro_batch / accum /
optimizer-lr from the YAML into the JSON at runtime so the global batch can't silently
drift from a stale config. Compare the printed peak / tok-s against FSDP FULL_SHARD.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import yaml

try:
    import deepspeed
except ImportError:
    sys.exit("DeepSpeed not installed. Run: uv pip install deepspeed")

sys.path.insert(0, "../01-transformer-from-scratch")
from data import TokenizedBinaryDataset  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from shared.utils.mfu import estimate_mfu  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML: model + data + micro_batch + seq_len")
    ap.add_argument("--deepspeed_config", required=True, help="JSON: ZeRO stage / precision / optimizer")
    ap.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds_config = json.load(open(args.deepspeed_config))

    # YAML is the single source of truth for batch geometry + lr; patch it into the DS
    # config so a stale JSON can't change the global batch out from under us.
    ds_config["train_micro_batch_size_per_gpu"] = cfg["micro_batch"]
    ds_config["gradient_accumulation_steps"] = cfg.get("accum_steps", 1)
    # Drop train_batch_size so DeepSpeed derives it (= micro * accum * world). The "auto"
    # keyword is a HuggingFace-integration feature, NOT core DeepSpeed: passed to a raw
    # deepspeed.initialize() it survives as the string "auto" and crashes the batch
    # assertion ('>' not supported between str and int).
    ds_config.pop("train_batch_size", None)
    if "optimizer" in ds_config:
        ds_config["optimizer"]["params"]["lr"] = cfg["lr"]

    deepspeed.init_distributed()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))

    torch.manual_seed(cfg["seed"])
    model_cfg = GPTConfig(**cfg["model"])
    model = GPT(model_cfg)  # construct on CPU; DeepSpeed shards onto each GPU in initialize()
    full_params = sum(p.numel() for p in model.parameters())  # full count, before sharding

    # DeepSpeed builds the optimizer from the JSON "optimizer" block and wraps it with the
    # ZeRO partitioner. bf16.enabled in the config means the engine runs the forward in
    # bf16 directly, so there's no torch.amp.autocast in the loop below.
    model_engine, _optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    device = model_engine.device

    torch.manual_seed(cfg["seed"] + 1000 * rank)
    train_ds = TokenizedBinaryDataset(cfg["data"]["train_path"], cfg["seq_len"])

    stage = ds_config.get("zero_optimization", {}).get("stage", 0)
    if rank == 0:
        print(
            f"world={world} ZeRO-stage={stage} params={full_params / 1e9:.2f}B "
            f"micro_batch={cfg['micro_batch']} seq_len={cfg['seq_len']}"
        )

    for step in range(cfg["total_steps"]):
        t0 = time.perf_counter()
        x, y = train_ds.sample_batch(cfg["micro_batch"], device)
        _, loss = model_engine(x, y)
        model_engine.backward(loss)   # ZeRO reduce-scatter + grad accumulation handled here
        model_engine.step()           # optimizer step + zero_grad + gradient clipping (from config)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if rank == 0 and step % 10 == 0:
            tokens = cfg["micro_batch"] * cfg["seq_len"] * world
            tps = tokens / dt
            mfu = estimate_mfu(
                tokens_per_second=tps,
                n_params=full_params,
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

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
