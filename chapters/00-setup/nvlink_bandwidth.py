"""Measure peer-to-peer copy bandwidth between every GPU pair.

This is a *unidirectional* P2P copy benchmark — for each (src, dst) pair we copy a 1 GiB
tensor from src to dst and report GB/s. NVLink should show 250–500 GB/s depending on the
GPU and NVSwitch topology; PCIe Gen4 shows 15–25 GB/s.

If you see a row/column with surprisingly low bandwidth, that pair is on PCIe rather than
NVLink. Plan your tensor-parallel and pipeline-parallel placement to avoid it.
"""

from __future__ import annotations

import torch


def bench_p2p(src: int, dst: int, size_bytes: int = 1 << 30, iters: int = 10) -> float:
    """Return GB/s for src -> dst P2P copy."""
    src_dev = torch.device("cuda", src)
    dst_dev = torch.device("cuda", dst)
    a = torch.empty(size_bytes // 4, dtype=torch.float32, device=src_dev)
    b = torch.empty_like(a, device=dst_dev)
    # Warmup
    for _ in range(2):
        b.copy_(a, non_blocking=True)
    torch.cuda.synchronize(src_dev)
    torch.cuda.synchronize(dst_dev)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.set_device(src_dev)
    start.record()
    for _ in range(iters):
        b.copy_(a, non_blocking=True)
    end.record()
    torch.cuda.synchronize(src_dev)
    torch.cuda.synchronize(dst_dev)
    t = start.elapsed_time(end) / 1000.0 / iters
    return size_bytes / t / 1e9


def main() -> None:
    n = torch.cuda.device_count()
    if n < 2:
        print("Need at least 2 GPUs.")
        return

    print(f"P2P copy bandwidth, {n} GPUs, 1 GiB transfer, GB/s\n")
    print("       " + "".join(f"  dst={j:<6d}" for j in range(n)))
    for src in range(n):
        row = [f"src={src} "]
        for dst in range(n):
            if src == dst:
                row.append(f"  {'-':>7s}")
                continue
            try:
                gbps = bench_p2p(src, dst)
                row.append(f"  {gbps:>7.1f}")
            except Exception as e:  # noqa: BLE001
                row.append(f"  {'FAIL':>7s}")
                print(f"  ({src}->{dst}) error: {e}")
        print("".join(row))


if __name__ == "__main__":
    main()
