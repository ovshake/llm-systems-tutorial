"""Contiguous KV cache for inference. Pre-allocates per-request buffers up to max_seq_len.

The contiguous-layout cache is what every inference engine started with. Chapter 13
generalizes to a *paged* layout, which is what production engines use.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KVCacheConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    max_seq_len: int
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"


class ContiguousKVCache:
    """One contiguous K and V tensor per layer, pre-allocated to max_seq_len.

    Memory per request:
        2 * n_layers * max_seq_len * n_kv_heads * head_dim * dtype_size

    This is the simplest possible layout — fast to read, easy to reason about, but it
    wastes memory for requests that don't fill `max_seq_len`. Real engines use paged
    memory (see Chapter 13).
    """

    def __init__(self, cfg: KVCacheConfig, batch_size: int = 1):
        self.cfg = cfg
        # K and V layout: (n_layers, batch_size, max_seq_len, n_kv_heads, head_dim)
        self.k = torch.zeros(
            cfg.n_layers,
            batch_size,
            cfg.max_seq_len,
            cfg.n_kv_heads,
            cfg.head_dim,
            dtype=cfg.dtype,
            device=cfg.device,
        )
        self.v = torch.zeros_like(self.k)
        self.lengths = torch.zeros(batch_size, dtype=torch.int32, device=cfg.device)

    def append(
        self,
        layer_idx: int,
        k_new: torch.Tensor,  # (B, T_new, n_kv_heads, head_dim)
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K, V to the per-request buffers. Returns the full K, V slices."""
        B, T_new = k_new.shape[:2]
        assert B == self.k.shape[1]
        for b in range(B):
            start = int(self.lengths[b].item())
            self.k[layer_idx, b, start : start + T_new] = k_new[b]
            self.v[layer_idx, b, start : start + T_new] = v_new[b]
        if layer_idx == self.cfg.n_layers - 1:
            # only advance lengths once per token (after last layer)
            self.lengths.add_(T_new)
        # The caller wants slices up to the current length. We assume same-length batch
        # for simplicity — generalize this later.
        L = int(self.lengths.max().item()) + (T_new if layer_idx < self.cfg.n_layers - 1 else 0)
        return self.k[layer_idx, :, :L], self.v[layer_idx, :, :L]

    def reset(self) -> None:
        self.lengths.zero_()
