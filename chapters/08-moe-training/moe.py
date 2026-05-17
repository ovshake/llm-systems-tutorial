"""Toy mixture-of-experts (single GPU). Top-k routing, optional load-balance loss,
optional capacity-factor token-drop.

Designed to be read end-to-end before you tackle the EP version. Doesn't use any
specialized kernels — the per-expert MLP is just a regular SwiGLU. Drop in to your
Chapter 01 model by replacing SwiGLUMLP with this.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """SwiGLU MLP. One per expert. Identical to Chapter 01's MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MoE(nn.Module):
    """Single-GPU MoE block with top-k routing.

    forward returns (output, aux_loss) where `aux_loss` is the load-balance auxiliary
    (already weighted by `self.aux_loss_weight`). Caller adds it to the main loss.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight

        self.router = nn.Linear(hidden_size, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(hidden_size, intermediate_size) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, S, H). Returns (y, aux_loss)."""
        # TODO Exercise 1: implement the MoE forward.
        # Steps:
        #   1. Flatten x to (N, H) where N = B*S, for routing convenience.
        #   2. router_logits = self.router(x_flat) -> (N, E)
        #   3. probs = softmax(router_logits, dim=-1) -> (N, E)
        #   4. top_k_probs, top_k_idx = probs.topk(k=top_k, dim=-1) -> (N, k), (N, k)
        #      Renormalize top_k_probs so they sum to 1 along k (some implementations do; some don't).
        #   5. Capacity: each expert can take at most `capacity = ceil(N * top_k * capacity_factor / E)` tokens.
        #   6. For each expert e:
        #         indices = (top_k_idx == e).any(dim=-1).nonzero() — tokens for expert e
        #         take the first `capacity` of them (drop the rest)
        #         output_e = experts[e](x_flat[indices])
        #         weight   = top_k_probs masked to those tokens, summed across k axis
        #         scatter weight * output_e back into y_flat[indices]
        #   7. Reshape y_flat to (B, S, H).
        #   8. Compute aux_loss with self._load_balance_loss(...).
        raise NotImplementedError("Implement MoE.forward (Exercise 1)")

    def _load_balance_loss(
        self, probs: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """Switch-Transformer auxiliary loss.

        probs:           (N, E) — softmax output of router
        expert_indices:  (N, k) — top-k expert IDs per token

        Returns a scalar:
            L_aux = E * sum_e f_e * P_e
        where f_e is the fraction of tokens routed to e (top-1), P_e is mean router prob for e.
        """
        E = probs.size(-1)
        # f_e: fraction of tokens that chose e as their TOP-1.
        top1 = expert_indices[..., 0]
        f = F.one_hot(top1, num_classes=E).float().mean(dim=0)
        # P_e: mean router probability for e.
        P = probs.mean(dim=0)
        return float(E) * (f * P).sum()
