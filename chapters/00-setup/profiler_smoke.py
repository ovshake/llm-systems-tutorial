"""Smoke test for `torch.profiler`. Drops a Chrome trace at out/trace.json that you can
open in chrome://tracing or in Perfetto (https://ui.perfetto.dev).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule


def main() -> None:
    device = torch.device("cuda")
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(6):
            _ = a @ b
            torch.cuda.synchronize()
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace(str(out_dir / "trace.json"))
    print(f"\nWrote {out_dir / 'trace.json'}. Open in chrome://tracing or ui.perfetto.dev.")


if __name__ == "__main__":
    main()
