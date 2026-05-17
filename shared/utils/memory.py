"""Memory accounting helpers.

The single most useful skill in large-model training is being able to predict, before
launch, where your VRAM goes. These helpers give you the parts of the equation.

For a dense transformer the memory budget on each GPU is roughly:

    weights + gradients + optimizer states + activations + KV cache (inference only) + overhead

For BF16 training with AdamW:
- weights:           2 bytes/param
- gradients:         2 bytes/param   (held in BF16 with PyTorch FSDP default)
- optimizer states:  8 bytes/param   (FP32 m + FP32 v)
- master weights:    4 bytes/param   (FP32 copy if you keep one; some setups skip)

Activations are the wildcard — they scale with batch * seq * hidden * layers and depend
on activation checkpointing policy. We don't try to estimate those here; profile instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def bytes_human(n: int | float) -> str:
    """Format bytes as a human-readable string with KiB/MiB/GiB units."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def param_count(module: nn.Module, trainable_only: bool = True) -> int:
    """Total parameter count of a module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


@dataclass
class MemoryEstimate:
    weights_bytes: int
    gradients_bytes: int
    optimizer_bytes: int
    master_weights_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.gradients_bytes
            + self.optimizer_bytes
            + self.master_weights_bytes
        )

    def __str__(self) -> str:
        return (
            f"weights      {bytes_human(self.weights_bytes)}\n"
            f"gradients    {bytes_human(self.gradients_bytes)}\n"
            f"optimizer    {bytes_human(self.optimizer_bytes)}\n"
            f"master       {bytes_human(self.master_weights_bytes)}\n"
            f"total        {bytes_human(self.total_bytes)}"
        )


def estimate_param_memory(
    n_params: int,
    param_dtype_bytes: int = 2,
    grad_dtype_bytes: int = 2,
) -> tuple[int, int]:
    return n_params * param_dtype_bytes, n_params * grad_dtype_bytes


def estimate_optimizer_memory(
    n_params: int,
    states: int = 2,
    state_dtype_bytes: int = 4,
    keep_master_weights: bool = True,
) -> MemoryEstimate:
    """Estimate per-GPU memory for params + grads + optimizer + master copy.

    Defaults are for AdamW in BF16 mixed precision with a master FP32 copy:
        - weights BF16: 2 B / param
        - grads    BF16: 2 B / param
        - optimizer (m + v) FP32: 8 B / param
        - master FP32 weights:     4 B / param

    For ZeRO-{1,2,3} the *sharded* counterpart divides each entry by the shard count.
    """
    w_bytes, g_bytes = estimate_param_memory(n_params)
    opt_bytes = n_params * states * state_dtype_bytes
    master_bytes = n_params * 4 if keep_master_weights else 0
    return MemoryEstimate(
        weights_bytes=w_bytes,
        gradients_bytes=g_bytes,
        optimizer_bytes=opt_bytes,
        master_weights_bytes=master_bytes,
    )


def gpu_memory_summary(device: torch.device | int = 0) -> str:
    """One-line summary of current GPU memory usage. Cheap to call every step."""
    if not torch.cuda.is_available():
        return "no cuda"
    alloc = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)
    return (
        f"alloc={bytes_human(alloc)} "
        f"reserved={bytes_human(reserved)} "
        f"peak={bytes_human(peak)}"
    )
