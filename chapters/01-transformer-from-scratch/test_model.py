"""Tests that pin the *contract* of model.py without leaking the implementation.

If these pass, you've implemented the model correctly enough to start training. Run with:
    pytest test_model.py -v
"""

from __future__ import annotations

import torch

from model import GPT, GPTConfig, RMSNorm, apply_rope, precompute_rope_freqs


# ── RMSNorm ──────────────────────────────────────────────────────────────────


def test_rmsnorm_shape_and_dtype():
    rn = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    y = rn(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_rmsnorm_unit_input():
    """For a constant input, rms = |x|, so output = weight * sign(x)."""
    rn = RMSNorm(64)
    x = torch.full((1, 1, 64), 3.0)
    y = rn(x)
    assert torch.allclose(y, torch.ones_like(y), atol=1e-4)


# ── RoPE ─────────────────────────────────────────────────────────────────────


def test_rope_zero_position_is_identity():
    """RoPE at position 0 (angle 0) leaves the tensor unchanged."""
    head_dim = 16
    freqs = precompute_rope_freqs(head_dim, 4)
    x = torch.randn(2, 4, 1, head_dim)  # (B, T=4 → take first), here just first pos
    out = apply_rope(x[:, :1], freqs[:1])
    assert torch.allclose(out, x[:, :1], atol=1e-5)


def test_rope_norm_preserved():
    """RoPE is a rotation: it preserves the L2 norm of every head_dim vector."""
    head_dim = 16
    freqs = precompute_rope_freqs(head_dim, 8)
    x = torch.randn(2, 4, 8, head_dim)
    out = apply_rope(x, freqs[: x.size(-2)])
    in_norm = x.norm(dim=-1)
    out_norm = out.norm(dim=-1)
    assert torch.allclose(in_norm, out_norm, atol=1e-4)


# ── Full model ────────────────────────────────────────────────────────────────


def small_cfg() -> GPTConfig:
    return GPTConfig(
        vocab_size=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        hidden_size=64,
        intermediate_size=128,
        context_length=32,
    )


def test_forward_shapes():
    cfg = small_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 16))
    logits, loss = model(x)
    assert logits.shape == (3, 16, cfg.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_loss():
    cfg = small_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 16))
    y = torch.randint(0, cfg.vocab_size, (3, 16))
    logits, loss = model(x, y)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0


def test_causal_mask():
    """Token t's output must not depend on tokens t+1, t+2, ..."""
    cfg = small_cfg()
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits_a, _ = model(x)

        # Perturb only the last token.
        x_b = x.clone()
        x_b[0, -1] = (x_b[0, -1] + 1) % cfg.vocab_size
        logits_b, _ = model(x_b)

    # Logits for positions 0..6 should be unchanged; position 7 may change.
    assert torch.allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-4), (
        "Causal mask is broken — earlier positions see later tokens."
    )


def test_grad_checkpoint_matches_eager():
    """Activation checkpointing must give the same logits as no checkpointing."""
    cfg = small_cfg()
    cfg.use_grad_checkpoint = False
    torch.manual_seed(0)
    model_a = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    logits_a, _ = model_a(x)

    cfg.use_grad_checkpoint = True
    torch.manual_seed(0)
    model_b = GPT(cfg)
    # Copy weights so the two models match.
    model_b.load_state_dict(model_a.state_dict())
    logits_b, _ = model_b(x)
    assert torch.allclose(logits_a, logits_b, atol=1e-5)
