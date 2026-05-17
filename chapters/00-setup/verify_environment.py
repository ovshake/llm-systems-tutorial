"""Environment audit: driver, CUDA, PyTorch, NCCL, Triton, and per-GPU info."""

from __future__ import annotations

import os
import platform
import subprocess
import sys

import torch


def shell(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}: {e}>"


def main() -> int:
    print("=" * 72)
    print("Environment audit")
    print("=" * 72)
    print(f"Python                : {sys.version.split()[0]}  ({platform.machine()})")
    print(f"PyTorch               : {torch.__version__}")
    print(f"PyTorch CUDA build    : {torch.version.cuda}")
    print(f"PyTorch CUDNN         : {torch.backends.cudnn.version()}")
    print(f"NCCL (linked)         : {torch.cuda.nccl.version()}")

    try:
        import triton

        print(f"Triton                : {triton.__version__}")
    except ImportError:
        print("Triton                : not installed (ok for early chapters)")

    print()
    print("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
    print(
        shell(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                "--format=csv",
            ]
        )
    )
    print()
    print("nvidia-smi topo -m")
    print(shell(["nvidia-smi", "topo", "-m"]))
    print()

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False. Fix this first.")
        return 1

    print(f"GPUs visible to PyTorch: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"  [{i}] {props.name}  "
            f"cap={props.major}.{props.minor}  "
            f"mem={props.total_memory / 2**30:.1f} GiB  "
            f"SMs={props.multi_processor_count}"
        )

    # BF16 support check
    try:
        a = torch.randn(8, 8, dtype=torch.bfloat16, device="cuda")
        b = a @ a
        _ = b.float().sum().item()
        print("BF16 matmul           : ok")
    except Exception as e:  # noqa: BLE001
        print(f"BF16 matmul           : FAILED: {e}")
        return 1

    # FP8 sanity (Hopper+ only — A100 does not have FP8, expect failure on A100)
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is not None:
        try:
            x = torch.zeros(8, dtype=fp8_dtype, device="cuda")
            _ = x + 0  # operations are limited
            print("FP8 dtype            : present (does not imply hardware support)")
        except Exception:
            print("FP8 dtype            : present but unusable on this device (expected on A100)")
    else:
        print("FP8 dtype            : not in this PyTorch")

    print()
    print("Relevant env vars:")
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
        "NCCL_IB_DISABLE",
        "TORCH_NCCL_BLOCKING_WAIT",
        "OMP_NUM_THREADS",
    ):
        print(f"  {var:<28s}= {os.environ.get(var, '<unset>')}")

    print()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
