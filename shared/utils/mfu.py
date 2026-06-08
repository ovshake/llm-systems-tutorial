"""Model FLOPs Utilization (MFU).

`MFU = achieved_flops_per_second / peak_flops_per_second_of_the_GPU`

It's the most useful single number for "is my training utilising the hardware?" Anything
under ~30% on an H100 means you have a bottleneck (data loading, comm, kernel choice...).
Production training of dense models often runs at 40–55% MFU.

The transformer FLOPs estimate below uses the standard Hoffmann et al. (Chinchilla)
approximation: ~6 * params * tokens for the forward+backward, plus the attention term
(quadratic in sequence length). For short sequences the 6N approximation dominates; for
long sequences attention starts to matter and you should add the 12*L*H*S^2 term per layer
where L is layers, H is hidden size, S is sequence length. We expose both.
"""

from __future__ import annotations


def h100_bf16_peak_tflops() -> float:
    """H100 SXM5 BF16/FP16 tensor core peak (dense, no sparsity): 989 TFLOPS.

    NVIDIA's datasheet headline of 1,979 TFLOPS includes 2:4 structured sparsity;
    use the dense 989 number as the MFU denominator. (For reference, the A100 SXM4
    this course used to target was 312 TFLOPS.)
    """
    return 989.0


def transformer_flops_per_token(
    n_params: int,
    seq_len: int,
    n_layers: int,
    hidden_size: int,
    include_attention_quadratic: bool = True,
) -> float:
    """Approximate forward+backward FLOPs per token.

    The 6N term is the Kaplan/Hoffmann approximation (2N per token forward + 4N backward).
    The attention-quadratic term (12 * L * H * S) accounts for the QK^T and attention*V
    matmuls scaling with sequence length squared.
    """
    base = 6.0 * n_params
    if not include_attention_quadratic:
        return base
    # 12 * L * H * S per token (the S^2 term divided by S to get per-token)
    attn = 12.0 * n_layers * hidden_size * seq_len
    return base + attn


def estimate_mfu(
    tokens_per_second: float,
    n_params: int,
    seq_len: int,
    n_layers: int,
    hidden_size: int,
    n_gpus: int = 1,
    peak_tflops_per_gpu: float = 989.0,
    include_attention_quadratic: bool = True,
) -> float:
    """Return MFU as a fraction in [0, 1+].

    `tokens_per_second` is the aggregate throughput across all GPUs.
    `peak_tflops_per_gpu` defaults to H100 BF16 peak (dense, 989 TFLOPS).
    """
    flops_per_token = transformer_flops_per_token(
        n_params, seq_len, n_layers, hidden_size, include_attention_quadratic
    )
    achieved_flops_per_second = flops_per_token * tokens_per_second
    peak_flops_per_second = peak_tflops_per_gpu * 1e12 * n_gpus
    return achieved_flops_per_second / peak_flops_per_second
