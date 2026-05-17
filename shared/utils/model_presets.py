"""Standard transformer presets used across the course.

Sizes are chosen so each one fits the corresponding hardware envelope with realistic
batch size and sequence length on A100-80GB. They roughly match Pythia / Llama-family
shapes, scaled down where convenient for fast iteration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    name: str
    n_layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int          # for MQA / GQA. Equal to n_heads => MHA.
    intermediate_size: int   # MLP up-projection size (usually 4 * hidden for GeLU MLPs, ~8/3 * hidden for SwiGLU)
    vocab_size: int = 50257  # GPT-2 BPE
    rope_base: float = 10000.0
    context_length: int = 2048

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    @property
    def approx_params(self) -> int:
        # rough estimate, ignoring biases / norms which are tiny
        embed = self.vocab_size * self.hidden_size
        attn = 4 * self.hidden_size * self.hidden_size * self.n_layers
        mlp = 2 * self.hidden_size * self.intermediate_size * self.n_layers
        return embed + attn + mlp


PRESETS: dict[str, ModelPreset] = {
    "tiny": ModelPreset(
        name="tiny", n_layers=6, hidden_size=384, n_heads=6, n_kv_heads=6,
        intermediate_size=1536, context_length=512,
    ),  # ~10M params. Single GPU, fast iteration.
    "small": ModelPreset(
        name="small", n_layers=12, hidden_size=768, n_heads=12, n_kv_heads=12,
        intermediate_size=3072, context_length=1024,
    ),  # ~125M params (~GPT-2 small). Single GPU.
    "medium": ModelPreset(
        name="medium", n_layers=24, hidden_size=1024, n_heads=16, n_kv_heads=16,
        intermediate_size=4096, context_length=2048,
    ),  # ~350M params (~GPT-2 medium). Single A100 with grad checkpointing.
    "1B": ModelPreset(
        name="1B", n_layers=24, hidden_size=2048, n_heads=16, n_kv_heads=16,
        intermediate_size=5632, context_length=2048,
    ),  # ~1.1B params. Single A100-80GB with care; comfortable on 2 with DDP.
    "3B": ModelPreset(
        name="3B", n_layers=32, hidden_size=2560, n_heads=20, n_kv_heads=20,
        intermediate_size=6912, context_length=2048,
    ),  # ~2.8B params. Needs FSDP or TP on 4 GPUs.
    "7B": ModelPreset(
        name="7B", n_layers=32, hidden_size=4096, n_heads=32, n_kv_heads=8,
        intermediate_size=11008, context_length=2048,
    ),  # ~7B params (~Llama-2 7B). FSDP or TP=4 required on 4xA100-80GB.
}
