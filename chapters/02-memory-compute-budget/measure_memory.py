"""Measure actual VRAM use for each preset; compare to predict_memory.

Skips presets that don't fit on the GPU. Prints predicted vs measured.
"""

from __future__ import annotations

import sys

import torch

from predict_memory import predict_memory
from shared.utils.memory import bytes_human
from shared.utils.model_presets import PRESETS

sys.path.insert(0, "../01-transformer-from-scratch")
from model import GPT, GPTConfig  # noqa: E402


def cfg_from_preset(p) -> GPTConfig:
    return GPTConfig(
        vocab_size=p.vocab_size,
        n_layers=p.n_layers,
        n_heads=p.n_heads,
        n_kv_heads=p.n_kv_heads,
        hidden_size=p.hidden_size,
        intermediate_size=p.intermediate_size,
        context_length=p.context_length,
    )


def measure(preset_name: str, batch_size: int, seq_len: int) -> int:
    """Return peak allocated bytes for a single forward+backward at (B, S)."""
    preset = PRESETS[preset_name]
    cfg = cfg_from_preset(preset)
    cfg.context_length = max(cfg.context_length, seq_len)
    torch.cuda.reset_peak_memory_stats()
    model = GPT(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del model, opt, x, y, loss
    torch.cuda.empty_cache()
    return peak


def main() -> None:
    print(f"{'preset':<8s} {'predicted':>12s} {'measured':>12s} {'delta':>8s}")
    print("-" * 44)
    for name, preset in PRESETS.items():
        if preset.approx_params > 4e9:
            # too big to fit unsharded on a single 80GB GPU; skip
            print(f"{name:<8s} {'too large':>12s} {'skip':>12s} {'-':>8s}")
            continue
        try:
            measured = measure(name, batch_size=4, seq_len=2048)
            predicted = predict_memory(preset, batch_size=4, seq_len=2048)["total"]
            delta = (measured - predicted) / predicted * 100
            print(
                f"{name:<8s} {bytes_human(predicted):>12s} {bytes_human(measured):>12s} "
                f"{delta:>+7.1f}%"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{name:<8s} {'-':>12s} {'OOM':>12s} {'-':>8s}")


if __name__ == "__main__":
    main()
