"""Closed-form memory estimate for the dense transformer presets.

This is the *planning* tool — what you'd run before launching a training job to
estimate VRAM per GPU.

For dense BF16 + AdamW with master weights, per parameter:
    weights        2 B
    gradients      2 B
    optimizer m+v  8 B   (FP32 each)
    master         4 B   (FP32, optional)
                  ────
                  16 B per param

Activation memory uses a simplified Korthikanti-style formula. The constant 10 (per layer
per token per element) is calibrated to a SwiGLU + RMSNorm decoder with FlashAttention.
Tune it for your own architecture in Exercise 1.
"""

from __future__ import annotations

from shared.utils.memory import bytes_human, estimate_optimizer_memory
from shared.utils.model_presets import PRESETS, ModelPreset


# Per-layer, per-token, per-hidden-dim activation BYTES for forward+backward stash.
# With FlashAttention this is roughly 10 bytes (residual + attn out + ffn intermediate
# in BF16). Without FlashAttention you'd add ~4 bytes per layer per token per S dim for
# the attention scratch — but we always use Flash here.
ACTIVATION_BYTES_PER_LAYER_TOKEN_H = 10.0


def predict_memory(
    preset: ModelPreset,
    batch_size: int,
    seq_len: int,
    use_grad_checkpoint: bool = False,
    keep_master_weights: bool = True,
) -> dict[str, int]:
    n = preset.approx_params
    mem = estimate_optimizer_memory(n, keep_master_weights=keep_master_weights)

    act_bytes_per_layer = ACTIVATION_BYTES_PER_LAYER_TOKEN_H * seq_len * batch_size * preset.hidden_size
    n_stashed_layers = 1 if use_grad_checkpoint else preset.n_layers
    activations = int(act_bytes_per_layer * n_stashed_layers)
    # In ckpt mode we still hold ~1 layer of activations in flight; the rest get recomputed.

    return {
        "weights": mem.weights_bytes,
        "gradients": mem.gradients_bytes,
        "optimizer_states": mem.optimizer_bytes,
        "master_weights": mem.master_weights_bytes,
        "activations": activations,
        "total": mem.total_bytes + activations,
    }


def main() -> None:
    header = (
        f"{'preset':<8s} {'params':>8s} "
        f"{'weights':>10s} {'grads':>10s} {'optim':>10s} {'master':>10s} "
        f"{'acts':>10s} {'total':>10s}  fits 80GB?"
    )
    print(header)
    print("-" * len(header))
    for name, preset in PRESETS.items():
        m = predict_memory(preset, batch_size=4, seq_len=2048, use_grad_checkpoint=False)
        fits = "yes" if m["total"] < 80 * 2**30 else " NO"
        print(
            f"{name:<8s} {preset.approx_params / 1e6:>7.0f}M "
            f"{bytes_human(m['weights']):>10s} {bytes_human(m['gradients']):>10s} "
            f"{bytes_human(m['optimizer_states']):>10s} {bytes_human(m['master_weights']):>10s} "
            f"{bytes_human(m['activations']):>10s} {bytes_human(m['total']):>10s}  {fits}"
        )

    print("\nWith activation checkpointing:")
    print(header)
    print("-" * len(header))
    for name, preset in PRESETS.items():
        m = predict_memory(preset, batch_size=4, seq_len=2048, use_grad_checkpoint=True)
        fits = "yes" if m["total"] < 80 * 2**30 else " NO"
        print(
            f"{name:<8s} {preset.approx_params / 1e6:>7.0f}M "
            f"{bytes_human(m['weights']):>10s} {bytes_human(m['gradients']):>10s} "
            f"{bytes_human(m['optimizer_states']):>10s} {bytes_human(m['master_weights']):>10s} "
            f"{bytes_human(m['activations']):>10s} {bytes_human(m['total']):>10s}  {fits}"
        )


if __name__ == "__main__":
    main()
