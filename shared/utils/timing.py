"""Timing helpers built on CUDA events.

Why CUDA events instead of `time.perf_counter()`:

Kernel launches are asynchronous. `time.perf_counter()` measures CPU wall time around the
launch, not GPU execution. For anything beyond the most coarse-grained benchmarking, that
gives misleading numbers — especially when overlapping comm and compute, where wall-clock
is dominated by the *slower* of the two and the CPU does not stall on it.

CUDA events are recorded *into* the GPU stream. `event.elapsed_time(other_event)` then
returns the wall time between the two events as the GPU saw them, in milliseconds.

`CudaTimer` is a context manager so the call sites read cleanly:

    with CudaTimer() as t:
        run_thing()
    print(t.ms)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class CudaTimer:
    """Measure GPU time between __enter__ and __exit__ via CUDA events."""

    device: torch.device | int | None = None
    ms: float = 0.0
    _start: torch.cuda.Event | None = None
    _end: torch.cuda.Event | None = None

    def __enter__(self) -> "CudaTimer":
        if not torch.cuda.is_available():
            raise RuntimeError("CudaTimer requires CUDA. Use cuda_sync_time on CPU.")
        if self.device is not None:
            torch.cuda.set_device(self.device)
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    def __exit__(self, *exc) -> None:
        assert self._end is not None and self._start is not None
        self._end.record()
        torch.cuda.synchronize()
        self.ms = self._start.elapsed_time(self._end)


@contextmanager
def cuda_sync_time(label: str = ""):
    """Lightweight `with`-style timer that calls `cuda.synchronize` and uses CPU wall time.

    Use when you don't have a CUDA device, or when you want to measure CPU+GPU end-to-end
    including post-processing on the host. For pure GPU work prefer `CudaTimer`.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if label:
        print(f"[{label}] {dt * 1000:.2f} ms")


def format_throughput(tokens: int, seconds: float) -> str:
    """Pretty-print tokens/sec with k/M units."""
    if seconds <= 0:
        return "n/a"
    tps = tokens / seconds
    if tps >= 1e6:
        return f"{tps / 1e6:.2f} Mtok/s"
    if tps >= 1e3:
        return f"{tps / 1e3:.2f} ktok/s"
    return f"{tps:.0f} tok/s"
