"""Single-GPU training loop with BF16, grad accumulation, activation checkpointing,
and checkpoint save/resume.

Run:
    python train.py --config configs/tiny.yaml

To resume:
    python train.py --config configs/tiny.yaml --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from rich.console import Console

from data import TokenizedBinaryDataset
from model import GPT, GPTConfig, num_params
from shared.utils.memory import gpu_memory_summary
from shared.utils.mfu import estimate_mfu
from torch.profiler import ProfilerActivity, profile, schedule, record_function


console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", default='/scratch/abhishek.maiti/llm-systems-tutorial/runs/')
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_lr(step: int, total: int, base: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return base * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(args.out or cfg.get("out_dir", "/scratch/abhishek.maiti/llm-systems-tutorial/runs/default"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.dump(cfg))

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")  # tf32 for fp32 ops

    # Data
    train_ds = TokenizedBinaryDataset(cfg["data"]["train_path"], cfg["seq_len"])
    val_ds = TokenizedBinaryDataset(cfg["data"]["val_path"], cfg["seq_len"])

    # Model
    model_cfg = GPTConfig(**cfg["model"])
    model = GPT(model_cfg).to(device)
    if cfg.get("compile", False):
        model = torch.compile(model)

    n_p = num_params(model)
    console.print(f"[bold]model params:[/] {n_p / 1e6:.2f} M")

    # Optimizer. AdamW with FP32 master weights via fused implementation when available.
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=tuple(cfg.get("betas", (0.9, 0.95))),
        weight_decay=cfg.get("weight_decay", 0.1),
        fused=True,
    )

    # Resume
    start_step = 0
    rng_state = None
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        console.print(f"[yellow]resuming from {ckpt_path}[/]")
        # Load to CPU: load_state_dict copies into the already-on-GPU model/optim
        # params, and the RNG ByteTensors must stay on CPU for set_rng_state.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        start_step = state["step"]
        rng_state = state.get("rng_state")
    if rng_state is not None:
        torch.random.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    total_steps = cfg["total_steps"]
    accum_steps = cfg.get("accum_steps", 1)
    log_every = cfg.get("log_every", 10)
    save_every = cfg.get("save_every", 500)

    autocast_dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float32

    tokens_seen = 0
    t_step = time.perf_counter()

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            schedule=schedule(skip_first=10, wait=1, warmup=1, active=3, repeat=1),
            record_shapes=True, 
            with_stack=False,
        ) as prof:

        for step in range(start_step, total_steps):
            lr = cosine_lr(
                step,
                total_steps,
                cfg["lr"],
                cfg.get("min_lr", cfg["lr"] * 0.1),
                cfg.get("warmup_steps", 100),
            )
            for g in optim.param_groups:
                g["lr"] = lr

            loss_accum = 0.0
            with record_function("forward_pass"):
                for _ in range(accum_steps):
                    x, y = train_ds.sample_batch(cfg["micro_batch"], device)
                    with torch.amp.autocast("cuda", dtype=autocast_dtype):
                        _, loss = model(x, y)
                    (loss / accum_steps).backward()
                    loss_accum += loss.item() / accum_steps
                    tokens_seen += x.numel()
            

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))

            with record_function("backward_pass"):
                optim.step()
                optim.zero_grad(set_to_none=True)

            if step % log_every == 0:
                torch.cuda.synchronize()
                dt = time.perf_counter() - t_step
                t_step = time.perf_counter()
                tokens_per_sec = (cfg["micro_batch"] * cfg["seq_len"] * accum_steps * log_every) / dt
                mfu = estimate_mfu(
                    tokens_per_second=tokens_per_sec,
                    n_params=n_p,
                    seq_len=cfg["seq_len"],
                    n_layers=model_cfg.n_layers,
                    hidden_size=model_cfg.hidden_size,
                    n_gpus=1,
                )
                console.print(
                    f"step {step:>6d} | loss {loss_accum:>7.3f} | lr {lr:>.2e} | "
                    f"|g| {grad_norm:>6.2f} | tok/s {tokens_per_sec:>9.0f} | "
                    f"mfu {mfu * 100:>4.1f}% | mem {gpu_memory_summary()}"
                )

            if step > 0 and step % save_every == 0:
                console.print(f"[green]saving ckpt step={step}[/]")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                        "config": cfg,
                        "rng_state": {
                            "torch": torch.random.get_rng_state(),
                            "torch_cuda": torch.cuda.get_rng_state_all(),
                        },
                    },
                    ckpt_path,
                )

            prof.step()
            if step >= start_step + (10 + 1 + 1 + 3):
                break
    os.makedirs(str(out_dir / f'profiler_step'), exist_ok=True) 
    prof.export_chrome_trace(str(out_dir / f'profiler_step' / 'trace.json'))
    print('profiler saved at ', str(out_dir / f'profiler_step' / 'trace.json'))

    console.print("[bold green]done[/]")


if __name__ == "__main__":
    main()
