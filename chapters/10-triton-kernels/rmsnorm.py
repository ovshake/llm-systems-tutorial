"""Triton RMSNorm forward.

Skeleton — you implement the kernel body. The driver, autograd integration, and benchmark
are wired up.

Run:
    python rmsnorm.py
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, w_ptr, y_ptr,
    stride_x_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One Triton program per row. The whole row fits in registers/SRAM for typical hidden sizes."""
    # TODO (Exercise 2): implement RMSNorm in Triton.
    # Pseudocode:
    #   row = tl.program_id(0)
    #   cols = tl.arange(0, BLOCK_SIZE)
    #   mask = cols < n_cols
    #   x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask).to(tl.float32)
    #   ms = tl.sum(x * x, axis=0) / n_cols
    #   rms = tl.sqrt(ms + eps)
    #   y = (x / rms).to(x_ptr.dtype.element_ty)
    #   w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    #   y = y * w.to(x_ptr.dtype.element_ty)
    #   tl.store(y_ptr + row * stride_x_row + cols, y, mask=mask)
    pass


def rmsnorm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    assert x.shape[-1] == weight.shape[0]
    y = torch.empty_like(x)
    n_cols = x.shape[-1]
    rows = x.numel() // n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    rmsnorm_fwd_kernel[(rows,)](
        x.view(-1, n_cols), weight, y.view(-1, n_cols),
        x.view(-1, n_cols).stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return y


def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.to(torch.float32)
    ms = x32.pow(2).mean(dim=-1, keepdim=True)
    return (x32 * torch.rsqrt(ms + eps)).to(x.dtype) * weight


def bench(B: int, S: int, H: int) -> None:
    x = torch.randn(B, S, H, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(H, device="cuda", dtype=torch.bfloat16)

    # Correctness (run only if kernel implemented)
    y_triton = rmsnorm_fwd(x, w)
    y_torch = torch_rmsnorm(x, w)
    if y_triton.norm() > 0:
        assert torch.allclose(y_triton, y_torch, atol=1e-2, rtol=1e-2), "Triton vs torch disagree"

    # Timing
    def time_fn(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / 50 / 1000

    t_torch = time_fn(lambda: torch_rmsnorm(x, w))
    t_triton = time_fn(lambda: rmsnorm_fwd(x, w))
    bytes_moved = 2 * B * S * H * 2 + H * 2
    print(
        f"(B,S,H)=({B},{S},{H})  "
        f"torch {t_torch * 1e6:>7.1f} us ({bytes_moved / t_torch / 1e9:>5.1f} GB/s)  "
        f"triton {t_triton * 1e6:>7.1f} us ({bytes_moved / t_triton / 1e9:>5.1f} GB/s)  "
        f"speedup {t_torch / t_triton:>4.2f}x"
    )


def main() -> None:
    for shape in [(4, 1024, 768), (4, 2048, 4096), (1, 8192, 4096)]:
        bench(*shape)


if __name__ == "__main__":
    main()
