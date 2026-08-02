"""A small, fully-implemented modern decoder transformer — self-contained for the demo.

Pre-norm, RMSNorm, RoPE, causal scaled-dot-product attention (FlashAttention on H100),
SwiGLU MLP, tied embeddings. This is deliberately a *standalone* implementation rather than
an import of `chapters/01-transformer-from-scratch/model.py`, because that file ships as a
teaching skeleton (its forward methods raise NotImplementedError until you complete the
chapter). The demo must run regardless of how far you are in the course.

The architecture mirrors chapter 01's intended design, so it's a faithful stand-in.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304          # 50257 rounded up to a multiple of 64 (tensor-core friendly)
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int | None = None    # None -> MHA; < n_heads -> GQA
    hidden_size: int = 2048
    intermediate_size: int = 5632    # ~8/3 * hidden, rounded
    context_length: int = 2048
    rope_base: float = 10000.0
    use_grad_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.hidden_size % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        assert (self.hidden_size // self.n_heads) % 2 == 0, "head_dim must be even for RoPE"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def _rope_tables(head_dim: int, seq_len: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)            # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (T, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (T, D)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        self.head_dim = cfg.hidden_size // cfg.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.wq = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # FlashAttention on H100
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(o)


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w3 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight  # tie

        cos, sin = _rope_tables(cfg.hidden_size // cfg.n_heads, cfg.context_length, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = input_ids.shape
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.tok_embed(input_ids)
        for blk in self.blocks:
            if self.cfg.use_grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
