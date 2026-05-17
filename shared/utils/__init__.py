"""Cross-chapter helpers.

Imported from any chapter via `from shared.utils import ...` after `uv pip install -e .`
from the repo root. Keep this surface small — anything chapter-specific lives in the chapter.
"""

from .timing import CudaTimer, cuda_sync_time, format_throughput
from .memory import (
    bytes_human,
    gpu_memory_summary,
    param_count,
    estimate_param_memory,
    estimate_optimizer_memory,
)
from .mfu import a100_bf16_peak_tflops, estimate_mfu, transformer_flops_per_token
from .model_presets import PRESETS, ModelPreset

__all__ = [
    "CudaTimer",
    "cuda_sync_time",
    "format_throughput",
    "bytes_human",
    "gpu_memory_summary",
    "param_count",
    "estimate_param_memory",
    "estimate_optimizer_memory",
    "a100_bf16_peak_tflops",
    "estimate_mfu",
    "transformer_flops_per_token",
    "PRESETS",
    "ModelPreset",
]
