"""Dataset and dataloader for the tokenized binary files produced by prepare_data.py.

We use memory-mapped numpy arrays — much cheaper than loading multi-GB datasets into RAM,
and lets the OS page in only the chunks we need.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenizedBinaryDataset:
    """Random-window sampler over a memory-mapped uint16 token stream."""

    def __init__(self, path: str | Path, seq_len: int):
        self.path = Path(path)
        self.seq_len = seq_len
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.data) < seq_len + 1:
            raise ValueError(
                f"Dataset {self.path} has {len(self.data)} tokens, need at least {seq_len + 1}."
            )

    def __len__(self) -> int:
        # Effectively unlimited; we randomly sample windows.
        return 10**9

    def sample_batch(
        self, batch_size: int, device: torch.device | str = "cuda"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ix = np.random.randint(0, len(self.data) - self.seq_len - 1, size=batch_size)
        x = np.stack([self.data[i : i + self.seq_len] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + 1 + self.seq_len] for i in ix]).astype(np.int64)
        x_t = torch.from_numpy(x).to(device, non_blocking=True)
        y_t = torch.from_numpy(y).to(device, non_blocking=True)
        return x_t, y_t
