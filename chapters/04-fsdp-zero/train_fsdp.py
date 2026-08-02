"""FSDP trainer — the workhorse for Exercises 2, 3, 4, and 6.

    # Ex 2/3: FULL_SHARD, per-Block wrapping, BF16 mixed precision (the recommended path)
    torchrun --standalone --nproc_per_node=8 train_fsdp.py --config configs/small.yaml
    torchrun --standalone --nproc_per_node=8 train_fsdp.py --config configs/1B.yaml

    # Ex 4: wrapping-policy ablation
    torchrun --standalone --nproc_per_node=8 train_fsdp.py --config configs/1B.yaml --wrap-policy none
    torchrun --standalone --nproc_per_node=8 train_fsdp.py --config configs/1B.yaml --wrap-policy size

    # Ex 6: activation checkpointing (raise micro_batch until the no-ckpt run OOMs)
    torchrun --standalone --nproc_per_node=8 train_fsdp.py --config configs/3B.yaml --activation-checkpoint

Demonstrates: FULL_SHARD, ModuleWrapPolicy({Block}), BF16 MixedPrecision, the wrapping-
policy knob, and FSDP-level activation checkpointing.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import yaml
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy

sys.path.insert(0, "../01-transformer-from-scratch")
from data import TokenizedBinaryDataset  # noqa: E402
from model import GPT, GPTConfig, Block  # noqa: E402
from shared.utils.mfu import estimate_mfu  # noqa: E402
from shared.utils.nccl import init_dist  # noqa: E402


def build_wrap_policy(kind: str):
    """Map --wrap-policy to an FSDP auto_wrap_policy (Exercise 4)."""
    if kind == "block":
        return ModuleWrapPolicy({Block})  # one Block = one all-gather unit (recommended)
    if kind == "size":
        return functools.partial(size_based_auto_wrap_policy, min_num_params=int(1e6))
    return None  # "none": only the root is a unit -> the whole model is all-gathered at once


def save_sharded_checkpoint(model, optim, ckpt_dir: str, tag: str, step: int) -> None:
    """Collective SHARDED_STATE_DICT save via PyTorch Distributed Checkpoint (DCP).

    EVERY rank must call this together — it is a collective. get_state_dict() puts FSDP into
    sharded mode and returns DTensor-based shards; dcp.save writes one shard per rank under
    ckpt_dir/tag. The step counter is kept OUT of the DCP state_dict and written to a tiny
    rank-0 sidecar (meta.json): a plain key inside the state_dict must also be present on load,
    which would break resuming any checkpoint that lacks it (e.g. one saved by older code).
    """
    model_sd, optim_sd = get_state_dict(model, optim)
    ckpt_path = f"{ckpt_dir}/{tag}"
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=ckpt_path)
    if dist.get_rank() == 0:
        (Path(ckpt_path) / "meta.json").write_text(json.dumps({"step": step}))


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    """Path of the most recently written DCP checkpoint under ckpt_dir, or None if there is
    none. A DCP checkpoint is a directory containing a `.metadata` file; we pick the newest."""
    base = Path(ckpt_dir)
    if not base.is_dir():
        return None
    metas = list(base.glob("*/.metadata"))
    if not metas:
        return None
    return str(max(metas, key=lambda m: m.stat().st_mtime).parent)


def load_sharded_checkpoint(model, optim, ckpt_path: str) -> int:
    """Collective DCP load. Refills the (resharded) model + optimizer shards in place and
    returns the step to resume from. dcp.load reshards automatically, so the world size that
    wrote the checkpoint need not match the current one — the Exercise 7 reshard win. The step
    comes from the meta.json sidecar (0 if absent, e.g. a checkpoint saved by older code).
    """
    model_sd, optim_sd = get_state_dict(model, optim)  # empty template, shaped for THIS world
    sd = {"model": model_sd, "optim": optim_sd}
    dcp.load(sd, checkpoint_id=ckpt_path)
    set_state_dict(model, optim, model_state_dict=sd["model"], optim_state_dict=sd["optim"])
    meta = Path(ckpt_path) / "meta.json"
    return int(json.loads(meta.read_text())["step"]) if meta.exists() else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--wrap-policy", choices=["block", "none", "size"], default="block")
    ap.add_argument("--sharding-strategy", choices=["full", "grad_op", "hybrid", "no_shard"], default="full")
    ap.add_argument(
        "--activation-checkpoint",
        action="store_true",
        help="non-reentrant activation checkpointing per Block, applied after FSDP (Ex 6)",
    )
    ap.add_argument("--cpu-offload", action="store_true", help="offload sharded params to CPU")
    ap.add_argument(
        "--ckpt-dir",
        default="/scratch/abhishek.maiti/llm-systems/tutorial/checkpoint/v1",
        help="directory for the sharded DCP checkpoint (Ex 7)",
    )
    ap.add_argument("--save-every", type=int, default=0, help="also save every N steps (0 = only at the end)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    rank, world, local = init_dist()
    device = torch.device("cuda", local)
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(cfg["seed"])
    model_cfg = GPTConfig(**cfg["model"])
    # Build + initialize on CPU — this also fills the RoPE cos/sin buffer correctly — then
    # let FSDP shard onto each GPU. Fine for <=7B on a fat-RAM node; for much larger models
    # you'd construct on the `meta` device and pass a param_init_fn instead.
    model = GPT(model_cfg)
    full_params = sum(p.numel() for p in model.parameters())

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    strategy = {
        "no_shard": ShardingStrategy.NO_SHARD,
        "full": ShardingStrategy.FULL_SHARD,
        "grad_op": ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2 equivalent (params replicated)
        "hybrid": ShardingStrategy.HYBRID_SHARD,    # == FULL_SHARD on a single node
    }[args.sharding_strategy]

    model = FSDP(
        model,
        auto_wrap_policy=build_wrap_policy(args.wrap_policy),
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        device_id=local,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=True) if args.cpu_offload else None,
        sync_module_states=True,  # broadcast rank-0 weights so every shard starts in agreement
    )

    if args.activation_checkpoint:
        # Must run AFTER FSDP wrapping: recompute each Block's activations in backward
        # instead of stashing them (Ex 6). FSDP shards params/grads/opt but NOT activations,
        # so this is the only lever for activation memory.
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda submodule: isinstance(submodule, Block),
        )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], fused=True)

    # Auto-resume: if the checkpoint dir already holds a checkpoint, load model + optimizer +
    # step from the newest one. dcp.load reshards on the fly, so a run saved at e.g. 4 ranks
    # resumes fine here at 2 or 8. Collective — every rank participates.
    start_step = 0
    if args.ckpt_dir:
        latest = find_latest_checkpoint(args.ckpt_dir)
        if latest is not None:
            start_step = load_sharded_checkpoint(model, optim, latest)
            if rank == 0:
                print(f"resuming from {latest} at step {start_step}")
        elif rank == 0:
            print(f"no checkpoint under {args.ckpt_dir} — starting fresh")

    torch.manual_seed(cfg["seed"] + 1000 * rank)
    train_ds = TokenizedBinaryDataset(cfg["data"]["train_path"], cfg["seq_len"])

    if rank == 0:
        print(
            f"world={world} params={full_params / 1e9:.2f}B "
            f"| wrap={args.wrap_policy} shard={args.sharding_strategy} "
            f"ckpt={args.activation_checkpoint} offload={args.cpu_offload}"
        )

    for step in range(start_step, cfg["total_steps"]):
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
        # DCP save is a COLLECTIVE — every rank must call it together, so it lives OUTSIDE
        # the `if rank == 0` block above. Under `if rank == 0` it deadlocks ranks 1..N.
        if args.ckpt_dir and args.save_every and (step + 1) % args.save_every == 0:
            if rank == 0:
                print(f"saving sharded checkpoint @ step {step + 1} ...")
            save_sharded_checkpoint(model, optim, args.ckpt_dir, f"step-{step + 1}", step + 1)

    if args.ckpt_dir:
        if rank == 0:
            print("saving final sharded checkpoint ...")
        save_sharded_checkpoint(model, optim, args.ckpt_dir, "final", cfg["total_steps"])

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
