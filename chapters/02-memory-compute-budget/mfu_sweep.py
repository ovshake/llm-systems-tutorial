"""Sweep batch_size × seq_len for the `small` preset and report tokens/sec + MFU.

Useful for finding the configuration that best utilizes the GPU before you scale out.
"""

from __future__ import annotations

import sys
import time

import torch

from shared.utils.mfu import estimate_mfu
from shared.utils.model_presets import PRESETS

sys.path.insert(0, "../01-transformer-from-scratch")
from model import GPT, GPTConfig  # noqa: E402


def main() -> None:
    preset = PRESETS["small"]
    cfg = GPTConfig(
        vocab_size=preset.vocab_size,
        n_layers=preset.n_layers,
        n_heads=preset.n_heads,
        n_kv_heads=preset.n_kv_heads,
        hidden_size=preset.hidden_size,
        intermediate_size=preset.intermediate_size,
        context_length=4096,
    )

    batches = [4, 8, 16, 32, 64]
    seqs = [512, 1024, 2048, 4096]

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

                # timed
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


if __name__ == "__main__":
    main()
