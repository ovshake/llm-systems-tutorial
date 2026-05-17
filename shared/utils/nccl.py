"""NCCL helpers: bandwidth probes, deterministic init, and rank discovery.

Used by Chapter 00 to measure the actual link bandwidth your node delivers, and reused
throughout Chapters 03–08 to translate collective costs into wall-clock numbers.

NCCL bandwidth is reported in two flavors:
- *algbw* (algorithm bandwidth): bytes moved / time.
- *busbw* (bus bandwidth): per-link bandwidth corrected for the ring/tree topology.
  For an all-reduce with N GPUs: busbw = algbw * 2 * (N-1) / N. For all-gather and
  reduce-scatter: busbw = algbw * (N-1) / N.

Both numbers are useful; busbw is what you compare against the link's hardware spec
(NVLink ~ 300 GB/s on A100, PCIe ~ 25 GB/s).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


def init_dist(backend: str = "nccl") -> tuple[int, int, int]:
    """Initialize `torch.distributed`. Returns (rank, world_size, local_rank).

    Works under `torchrun`. Sets the CUDA device to `local_rank` automatically.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main(rank: int | None = None) -> bool:
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    return rank == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


@dataclass
class BandwidthResult:
    op: str
    n_bytes: int
    seconds: float
    algbw_gbps: float
    busbw_gbps: float

    def __str__(self) -> str:
        return (
            f"{self.op:<14s} size={self.n_bytes / 2**20:>8.1f} MiB  "
            f"t={self.seconds * 1000:>7.3f} ms  "
            f"algbw={self.algbw_gbps:>6.1f} GB/s  busbw={self.busbw_gbps:>6.1f} GB/s"
        )


def benchmark_collective(
    op: str,
    n_bytes: int,
    iters: int = 20,
    warmup: int = 5,
    dtype: torch.dtype = torch.float32,
    device: torch.device | int = 0,
) -> BandwidthResult:
    """Microbenchmark an NCCL collective. `op` ∈ {"all_reduce", "all_gather", "reduce_scatter"}.

    Returns measured algbw and busbw in GB/s (powers-of-2 GiB/s, sometimes shown as GB/s
    in NCCL docs).
    """
    assert dist.is_initialized(), "Call init_dist() first."
    world_size = dist.get_world_size()
    elem_size = torch.tensor([], dtype=dtype).element_size()
    n_elem = n_bytes // elem_size
    if op == "all_reduce":
        buf = torch.ones(n_elem, dtype=dtype, device=device)

        def run():
            dist.all_reduce(buf)

        bus_factor = 2.0 * (world_size - 1) / world_size
    elif op == "all_gather":
        shard = n_elem // world_size
        out = torch.empty(n_elem, dtype=dtype, device=device)
        chunk = torch.ones(shard, dtype=dtype, device=device)

        def run():
            dist.all_gather_into_tensor(out, chunk)

        bus_factor = (world_size - 1) / world_size
        n_bytes = shard * elem_size * world_size  # be precise
    elif op == "reduce_scatter":
        shard = n_elem // world_size
        out = torch.empty(shard, dtype=dtype, device=device)
        full = torch.ones(n_elem, dtype=dtype, device=device)

        def run():
            dist.reduce_scatter_tensor(out, full)

        bus_factor = (world_size - 1) / world_size
    else:
        raise ValueError(f"unknown op {op}")

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / 1000.0 / iters

    algbw_gbps = n_bytes / t / 1e9
    busbw_gbps = algbw_gbps * bus_factor
    return BandwidthResult(op, n_bytes, t, algbw_gbps, busbw_gbps)
