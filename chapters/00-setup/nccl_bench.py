"""NCCL collective sweep. Run with:

    torchrun --standalone --nproc_per_node=4 nccl_bench.py

Reports algbw and busbw for all_reduce / all_gather / reduce_scatter at sizes from 1 MiB
to 1 GiB. This is the script that tells you, for your specific node, the *effective*
bandwidth your model's gradients/parameters will see during distributed training.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from shared.utils.nccl import benchmark_collective, init_dist


SIZES = [1 << s for s in range(20, 31, 1)]  # 1 MiB → 1 GiB
OPS = ["all_reduce", "all_gather", "reduce_scatter"]


def main() -> None:
    rank, world, local = init_dist()
    device = torch.device("cuda", local)

    if rank == 0:
        print(f"World size: {world}\n")
        header = (
            f"{'op':<14s} {'size(MiB)':>10s} {'time(ms)':>10s} "
            f"{'algbw(GB/s)':>12s} {'busbw(GB/s)':>12s}"
        )
        print(header)
        print("-" * len(header))

    for op in OPS:
        for nb in SIZES:
            try:
                res = benchmark_collective(op, nb, dtype=torch.bfloat16, device=device)
            except Exception as e:  # noqa: BLE001
                if rank == 0:
                    print(f"{op:<14s} size={nb / 2**20:>8.1f} MiB  error: {e}")
                continue
            if rank == 0:
                print(
                    f"{op:<14s} {nb / 2**20:>10.1f} {res.seconds * 1000:>10.3f} "
                    f"{res.algbw_gbps:>12.1f} {res.busbw_gbps:>12.1f}"
                )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
