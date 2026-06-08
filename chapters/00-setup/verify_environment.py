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

    # FP8 sanity. Hopper (H100, sm_90) has native FP8 tensor cores, so on this course's
    # target hardware FP8 should be present AND usable. A failure here on an H100 means a
    # stale PyTorch / CUDA build, not an expected hardware limitation. (The FP8 GEMM entry
    # point you'll use in Chapters 11/14 is torch._scaled_mm.)
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    major = torch.cuda.get_device_capability()[0]
    if fp8_dtype is None:
        print("FP8 dtype            : not in this PyTorch (upgrade to a Hopper-aware build)")
    else:
        try:
            x = torch.zeros(8, dtype=fp8_dtype, device="cuda")
            _ = x.float().sum().item()
            if major >= 9:
                print("FP8 dtype            : present and usable (Hopper native FP8) ok")
            else:
                print(f"FP8 dtype            : present, but sm_{major}0 has no native FP8 tensor cores")
        except Exception as e:  # noqa: BLE001
            print(f"FP8 dtype            : present but cast FAILED: {e}")

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
