"""Distributed hello world. Run with:

    torchrun --standalone --nproc_per_node=8 hello_distributed.py

Each rank prints its identity, then we exchange a single all_reduce to confirm NCCL works.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Print in rank order using a barrier-stagger to avoid interleaving.
    for r in range(world_size):
        if rank == r:
            print(
                f"[rank {rank:>2d} / {world_size:>2d}] "
                f"local_rank={local_rank} device={device} "
                f"hostname={os.uname().nodename}"
            )
        dist.barrier()

    x = torch.tensor([1.0], device=device)
    dist.all_reduce(x)
    expected = float(world_size)
    assert x.item() == expected, f"all_reduce got {x.item()}, expected {expected}"
    if rank == 0:
        print(f"all_reduce(1)*world_size: got {x.item()} (expected {expected}) ✓")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
