"""Page-style block manager for KV cache memory.

Treats KV memory as a pool of fixed-size blocks (typically 16 or 32 tokens each). Each
request gets a "block table" — an ordered list of block IDs that hold its tokens. When
the request grows past a block boundary, the manager allocates another block.

This is the foundation of PagedAttention. See README.md.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class PagedRequest:
    request_id: int
    block_table: list[int] = field(default_factory=list)
    seq_len: int = 0   # tokens currently stored in this request's KV

    def n_blocks(self, block_size: int) -> int:
        return (self.seq_len + block_size - 1) // block_size


class OutOfBlocksError(RuntimeError):
    pass


class BlockManager:
    """Allocator + reverse lookup. No GPU memory is owned here — just the bookkeeping.

    Callers pair this with a KV pool tensor of shape:
        (n_blocks, block_size, n_kv_heads, head_dim)
    or per-layer:
        (n_layers, n_blocks, block_size, n_kv_heads, head_dim)
    """

    def __init__(self, n_blocks: int, block_size: int):
        self.n_blocks = n_blocks
        self.block_size = block_size
        self._free: deque[int] = deque(range(n_blocks))

    def free_blocks_count(self) -> int:
        return len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise OutOfBlocksError(f"Asked for {n} blocks; only {len(self._free)} free.")
        return [self._free.popleft() for _ in range(n)]

    def free(self, block_ids: list[int]) -> None:
        for b in block_ids:
            self._free.append(b)

    # Convenience helpers ----------------------------------------------------------

    def grow_request_by(self, req: PagedRequest, n_new_tokens: int) -> None:
        """Allocate enough blocks for `n_new_tokens` more tokens on `req`."""
        new_total = req.seq_len + n_new_tokens
        new_n_blocks = (new_total + self.block_size - 1) // self.block_size
        cur_n_blocks = len(req.block_table)
        if new_n_blocks > cur_n_blocks:
            extra = self.allocate(new_n_blocks - cur_n_blocks)
            req.block_table.extend(extra)
        req.seq_len = new_total

    def release_request(self, req: PagedRequest) -> None:
        self.free(req.block_table)
        req.block_table.clear()
        req.seq_len = 0
