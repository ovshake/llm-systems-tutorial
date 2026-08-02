"""Sweep batch_size × seq_len for the `small` preset and report tokens/sec + MFU.

Useful for finding the configuration that best utilizes the GPU before you scale out.
Also emits a chrome trace for a single configuration's steady-state step (see
`profile_cell`), kept separate from the timed sweep so the profiler's overhead doesn't
pollute the tok/s numbers.
"""

from __future__ import annotations

import sys
import time

import torch

from shared.utils.mfu import estimate_mfu
from shared.utils.model_presets import PRESETS
from torch.profiler import ProfilerActivity, profile, record_function, schedule

sys.path.insert(0, "../01-transformer-from-scratch")
from model import GPT, GPTConfig  # noqa: E402


def build_cfg(preset, context_length: int = 4096) -> GPTConfig:
    return GPTConfig(
        vocab_size=preset.vocab_size,
        n_layers=preset.n_layers,
        n_heads=preset.n_heads,
        n_kv_heads=preset.n_kv_heads,
        hidden_size=preset.hidden_size,
        intermediate_size=preset.intermediate_size,
        context_length=context_length,
    )


def labeled_step(model, opt, x, y) -> torch.Tensor:
    """One train step with record_function regions, for a readable trace."""
    with record_function("forward"):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
    with record_function("backward"):
        loss.backward()
    with record_function("optim"):
        opt.step()
        opt.zero_grad(set_to_none=True)
    return loss


def profile_cell(cfg, B: int, S: int, out_path: str = "trace.json") -> None:
    """Capture a chrome trace of one (B, S) cell's steady-state step.

    The schedule below does nothing for `wait` steps, warms up the profiler for
    `warmup`, then records `active`. It needs exactly wait+warmup+active `prof.step()`
    calls to run one cycle and flush the trace, so the loop runs that many iterations
    and calls `prof.step()` once per iteration (work first, then step). A few untimed
    iterations beforehand push the model into steady state (allocator warm, cuDNN
    autotuned) so the recorded steps are representative rather than first-run outliers.
    """
    model = GPT(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    x = torch.randint(0, cfg.vocab_size, (B, S), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (B, S), device="cuda")

    for _ in range(3):  # steady-state warmup, outside the profiler window
        labeled_step(model, opt, x, y)
    torch.cuda.synchronize()

    wait, warmup, active = 1, 1, 3
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(wait + warmup + active):
            labeled_step(model, opt, x, y)
            prof.step()

    prof.export_chrome_trace(out_path)
    print(f"chrome trace ({active} steady-state steps at B={B}, S={S}) -> {out_path}")
    del model, opt, x, y
    torch.cuda.empty_cache()


def main() -> None:
    preset = PRESETS["small"]
    cfg = build_cfg(preset, context_length=4096)

    batches = [8]  # [4, 8, 16, 32, 64]
    seqs = [4096]  # [512, 1024, 2048, 4096]

    print(f"{'batch':>6s} {'seq':>6s} {'tok/s':>10s} {'MFU':>6s} {'peak':>10s}")
    print("-" * 44)

    for B in batches:
        for S in seqs:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                model = GPT(cfg).cuda()
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
                x = torch.randint(0, cfg.vocab_size, (B, S), device="cuda")
                y = torch.randint(0, cfg.vocab_size, (B, S), device="cuda")

                # warmup
                for _ in range(3):
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()

                # timed (no profiler here — keep tok/s clean)
                iters = 5
                t0 = time.perf_counter()
                for _ in range(iters):
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0

                tps = (B * S * iters) / dt
                mfu = estimate_mfu(
                    tokens_per_second=tps,
                    n_params=preset.approx_params,
                    seq_len=S,
                    n_layers=preset.n_layers,
                    hidden_size=preset.hidden_size,
                )
                peak = torch.cuda.max_memory_allocated() / 2**30
                print(f"{B:>6d} {S:>6d} {tps:>10.0f} {mfu * 100:>5.1f}% {peak:>9.1f}G")
                del model, opt, x, y, loss
            except torch.cuda.OutOfMemoryError:
                print(f"{B:>6d} {S:>6d} {'OOM':>10s} {'-':>6s} {'-':>10s}")
                torch.cuda.empty_cache()

    # Trace a single representative cell, separate from the timing above.
    profile_cell(cfg, B=batches[0], S=seqs[0])


if __name__ == "__main__":
    main()
