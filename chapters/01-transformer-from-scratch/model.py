"""Tiny GPT in the modern decoder style. Pre-norm, RMSNorm, SwiGLU, RoPE.

This file is the central artifact of Chapter 01. Several methods raise NotImplementedError
on purpose — go fill them in. The shape comments and the tests in test_model.py specify
exactly what each piece should do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int | None = None  # None -> MHA; equal heads. Otherwise GQA.
    hidden_size: int = 768
    intermediate_size: int = 3072
    context_length: int = 1024
    rope_base: float = 10000.0
    dropout: float = 0.0
    use_grad_checkpoint: bool = False

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.hidden_size % self.n_heads == 0, "hidden_size must be divisible by n_heads"


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # TODO Exercise 1: implement RMSNorm.
        # Steps:
        #   rms = sqrt(mean(x * x, dim=-1) + eps)
        #   return self.weight * x / rms
        # Cast to fp32 for the norm computation, then back to x.dtype for the output.
        # This matches the standard Llama implementation and is numerically stable in BF16.
        raise NotImplementedError("Implement RMSNorm.forward (Exercise 1.2)")


# ─────────────────────────────────────────────────────────────────────────────
# Rotary positional embedding
# ─────────────────────────────────────────────────────────────────────────────


def precompute_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute (cos, sin) frequencies for RoPE. Shape: (max_seq_len, head_dim/2, 2)."""
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.einsum("i,j->ij", t, freqs)  # (seq, head_dim/2)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)  # (seq, head_dim/2, 2)


def apply_rope(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (B, H, T, head_dim).

    cos_sin has shape (T, head_dim/2, 2) — the slice for these T positions.

    Treat consecutive pairs (x_{2k}, x_{2k+1}) as a 2-vector and rotate by angle θ_{t,k}:
        x'_{2k}   = x_{2k} * cos - x_{2k+1} * sin
        x'_{2k+1} = x_{2k} * sin + x_{2k+1} * cos
    """
    # TODO Exercise 1: implement rotary embedding.
    raise NotImplementedError("Implement apply_rope (Exercise 1.1)")


# ─────────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────────


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.hidden_size // cfg.n_heads
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # how many Q-heads share a KV-head (GQA)

        self.wq = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        # TODO Exercise 1: implement attention.
        # Steps:
        #   1. Compute q, k, v with the three linear projections.
        #   2. Reshape to (B, n_heads/n_kv_heads, T, head_dim).
        #   3. Apply RoPE to q and k (not v).
        #   4. If GQA (n_kv_heads < n_heads), repeat k and v along the head dimension
        #      so they broadcast against q.
        #   5. Use F.scaled_dot_product_attention with is_causal=True. On H100 this
        #      dispatches to FlashAttention; on older GPUs it falls back to the math kernel.
        #   6. Reshape back to (B, T, C), apply the output projection.
        raise NotImplementedError("Implement Attention.forward (Exercise 1.3)")


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU MLP
# ─────────────────────────────────────────────────────────────────────────────


class SwiGLUMLP(nn.Module):
    """SwiGLU: y = W3(Swish(W1 x) * W2 x). Three linear projections, no bias.

    Note on sizing: the canonical SwiGLU MLP uses intermediate ~ 8/3 * hidden to roughly
    match the parameter count of a 4 * hidden GeLU MLP. Choose `cfg.intermediate_size`
    accordingly; we don't enforce it.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w3 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO Exercise 1: implement SwiGLU.
        raise NotImplementedError("Implement SwiGLUMLP.forward (Exercise 1.4)")


# ─────────────────────────────────────────────────────────────────────────────
# Block
# ─────────────────────────────────────────────────────────────────────────────


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos_sin)
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# GPT
# ─────────────────────────────────────────────────────────────────────────────


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Tie weights — saves ~vocab*hidden parameters and tends to help small LMs.
        self.lm_head.weight = self.tok_embed.weight

        self.register_buffer(
            "cos_sin",
            precompute_rope_freqs(cfg.hidden_size // cfg.n_heads, cfg.context_length, cfg.rope_base),
            persistent=False,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        input_ids: (B, T) int64
        targets:   (B, T) int64 — shifted by one for next-token prediction. Optional.

        Returns: (logits, loss). loss is None if targets is None.
        """
        # TODO Exercise 1: implement the full forward.
        # Steps:
        #   1. Embed input_ids -> (B, T, C).
        #   2. Slice cos_sin for the current T.
        #   3. Pass through every block. If cfg.use_grad_checkpoint, wrap each block call
        #      in torch.utils.checkpoint.checkpoint with use_reentrant=False.
        #   4. norm_f, then lm_head to get logits (B, T, V).
        #   5. If targets is provided, compute cross-entropy with F.cross_entropy. Reshape
        #      logits to (B*T, V) and targets to (B*T) for the call.
        raise NotImplementedError("Implement GPT.forward (Exercise 1.5)")

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Naive autoregressive generation — recomputes from scratch every step.

        Chapter 12 builds a real KV-cache-driven inference loop. This one is just for
        sanity-checking that the trained model produces text.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_length :]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
