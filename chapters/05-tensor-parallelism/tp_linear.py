"""Column- and row-parallel linear layers, from scratch.

These are the Megatron building blocks. Implementation deliberately uses raw NCCL
collectives so you see the comm pattern explicitly. In production you'd use PyTorch's
DTensor or Megatron's official implementations, but the math here is identical.

Conventions:
- "column-parallel" splits the *output* dimension across ranks. Input replicated, output sharded.
- "row-parallel" splits the *input* dimension across ranks. Input sharded, output replicated.

Backward for both is the dual operation, which `torch.autograd` handles for us once we use
the `_AllReduce` / `_AllGather` `Function` shims below.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


# ── Autograd-aware collective shims ──────────────────────────────────────────


class _AllReduce(torch.autograd.Function):
    """all_reduce in forward, identity in backward (since grad of sum-reduce is replicate)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        return dy


class _CopyToTPRegion(torch.autograd.Function):
    """Identity in forward, all_reduce in backward.

    Used at the *input* of a column-parallel layer: the input is replicated. On backward,
    each rank produces its slice of the gradient, and we need them summed to get the
    full input gradient.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(dy, op=dist.ReduceOp.SUM)
        return dy


# ── Column-parallel linear ───────────────────────────────────────────────────


class ColumnParallelLinear(nn.Module):
    """y = x W^T where W is split along its row (output) dim.

    Input: (..., in_features) — replicated across TP ranks.
    Output: (..., out_features / tp_size) — sharded along last dim.

    If `gather_output=True`, gathers the output so it's replicated again (paid: one all_gather).
    Default False — the next layer is usually row-parallel and expects the sharded form.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        bias: bool = False,
        gather_output: bool = False,
    ):
        super().__init__()
        assert out_features % tp_size == 0, (
            f"out_features ({out_features}) must be divisible by tp_size ({tp_size})"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.gather_output = gather_output
        self.out_per_rank = out_features // tp_size

        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        nn.init.normal_(self.weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(self.out_per_rank)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _CopyToTPRegion.apply(x)            # identity fwd, all_reduce bwd
        y = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.gather_output and self.tp_size > 1:
            gathered = [torch.empty_like(y) for _ in range(self.tp_size)]
            dist.all_gather(gathered, y)
            y = torch.cat(gathered, dim=-1)
        return y


# ── Row-parallel linear ──────────────────────────────────────────────────────


class RowParallelLinear(nn.Module):
    """y = x W^T where W is split along its column (input) dim.

    Input: (..., in_features / tp_size) — sharded.
    Output: (..., out_features) — replicated, via all_reduce of partial outputs.

    If `input_is_parallel=False`, expects a replicated input and shards it before compute.
    Default True since the previous layer is usually column-parallel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        bias: bool = False,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        assert in_features % tp_size == 0
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.input_is_parallel = input_is_parallel
        self.in_per_rank = in_features // tp_size

        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        nn.init.normal_(self.weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            # Split the input across ranks (each rank takes its slice).
            shard = x.shape[-1] // self.tp_size
            x = x[..., self.tp_rank * shard : (self.tp_rank + 1) * shard]
        y = torch.nn.functional.linear(x, self.weight, None)  # partial output
        if self.tp_size > 1:
            y = _AllReduce.apply(y)
        if self.bias is not None:
            y = y + self.bias
        return y
