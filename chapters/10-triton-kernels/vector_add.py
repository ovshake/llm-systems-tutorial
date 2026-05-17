"""Triton vector-add. The 'hello world' of Triton.

Run:
    python vector_add.py
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape and x.is_cuda and y.is_cuda
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def bench(n: int) -> tuple[float, float]:
    x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    # Torch
    for _ in range(3):
        _ = x + y
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        _ = x + y
    end.record()
    torch.cuda.synchronize()
    t_torch = start.elapsed_time(end) / 100 / 1000.0

    # Triton
    for _ in range(3):
        _ = add(x, y)
    torch.cuda.synchronize()
    start.record()
    for _ in range(100):
        _ = add(x, y)
    end.record()
    torch.cuda.synchronize()
    t_triton = start.elapsed_time(end) / 100 / 1000.0

    # Bytes moved: read x + read y + write out = 3 * n * 2 bytes (BF16)
    bytes_moved = 3 * n * 2
    return bytes_moved / t_torch / 1e9, bytes_moved / t_triton / 1e9


def main() -> None:
    print(f"{'n':>10s} {'torch GB/s':>12s} {'triton GB/s':>12s}")
    for log_n in (16, 20, 22, 24, 26, 28):
        n = 1 << log_n
        bw_torch, bw_triton = bench(n)
        print(f"{n:>10d} {bw_torch:>12.1f} {bw_triton:>12.1f}")


if __name__ == "__main__":
    main()
