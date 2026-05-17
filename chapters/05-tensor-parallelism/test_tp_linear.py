"""Numerical equivalence tests for ColumnParallelLinear / RowParallelLinear.

Run:
    torchrun --standalone --nproc_per_node=4 -m pytest test_tp_linear.py -v

These tests construct a reference `nn.Linear`, copy the relevant weight slice into each
TP rank's shard, and assert the output matches.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from shared.utils.nccl import init_dist
from tp_linear import ColumnParallelLinear, RowParallelLinear


def _ensure_dist():
    if not dist.is_initialized():
        init_dist()


@pytest.fixture(scope="module")
def world():
    _ensure_dist()
    return dist.get_world_size(), dist.get_rank()


def test_column_parallel_matches_linear(world):
    tp, rank = world
    torch.manual_seed(0)
    in_f, out_f = 64, 128
    full = torch.randn(out_f, in_f, device="cuda")  # same on every rank, same seed
    x = torch.randn(2, 8, in_f, device="cuda")

    ref = torch.nn.functional.linear(x, full)

    col = ColumnParallelLinear(in_f, out_f, tp, rank, bias=False, gather_output=True).cuda()
    shard = full[rank * (out_f // tp) : (rank + 1) * (out_f // tp), :]
    col.weight.data.copy_(shard)
    out = col(x)
    assert torch.allclose(out, ref, atol=1e-4)


def test_row_parallel_matches_linear(world):
    tp, rank = world
    torch.manual_seed(0)
    in_f, out_f = 128, 64
    full = torch.randn(out_f, in_f, device="cuda")
    x = torch.randn(2, 8, in_f, device="cuda")

    ref = torch.nn.functional.linear(x, full)

    row = RowParallelLinear(in_f, out_f, tp, rank, bias=False, input_is_parallel=False).cuda()
    shard = full[:, rank * (in_f // tp) : (rank + 1) * (in_f // tp)]
    row.weight.data.copy_(shard)
    out = row(x)
    assert torch.allclose(out, ref, atol=1e-4)


def test_column_then_row_round_trip(world):
    """Stack column-parallel into row-parallel — the canonical TP MLP shape."""
    tp, rank = world
    torch.manual_seed(0)
    in_f, mid_f, out_f = 64, 256, 64
    W1 = torch.randn(mid_f, in_f, device="cuda")
    W2 = torch.randn(out_f, mid_f, device="cuda")
    x = torch.randn(2, 8, in_f, device="cuda")

    ref = torch.nn.functional.linear(torch.nn.functional.linear(x, W1), W2)

    col = ColumnParallelLinear(in_f, mid_f, tp, rank, bias=False, gather_output=False).cuda()
    row = RowParallelLinear(mid_f, out_f, tp, rank, bias=False, input_is_parallel=True).cuda()
    col.weight.data.copy_(W1[rank * (mid_f // tp) : (rank + 1) * (mid_f // tp), :])
    row.weight.data.copy_(W2[:, rank * (mid_f // tp) : (rank + 1) * (mid_f // tp)])

    out = row(col(x))
    assert torch.allclose(out, ref, atol=1e-3)
