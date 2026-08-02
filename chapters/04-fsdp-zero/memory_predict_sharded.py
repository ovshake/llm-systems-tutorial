"""Per-GPU memory under ZeRO-{1,2,3} and FSDP FULL_SHARD (Exercise 1).

Extends Chapter 02's `predict_memory.py`: same closed-form model-state + activation
accounting, but now each of the four states (weights / grads / optimizer m+v / FP32
master) is *sharded* across `world_size` ranks according to the ZeRO stage.

    Stage / strategy          weights  grads   opt+master
    DDP                        repl     repl    repl
    ZeRO-1                     repl     repl    shard
    ZeRO-2                     repl     shard   shard
    ZeRO-3 / FSDP FULL_SHARD   shard    shard   shard

Activations are NOT sharded — they're set by per-rank batch*seq and are identical on
every rank. That's exactly why activation checkpointing still matters under FSDP.

This is a planning estimate. It ignores the transient all-gather buffer (~one wrap
unit), NCCL workspace, and allocator fragmentation, so the measured peak runs higher —
compare against the `peak` column printed by train_fsdp.py.

    python memory_predict_sharded.py                 # world=8, B=4, S=2048
    python memory_predict_sharded.py --world 4 --batch 8 --seq 4096
    python memory_predict_sharded.py --no-master     # BF16 AdamW, no FP32 master copy
"""

from __future__ import annotations

import argparse
import sys

from shared.utils.memory import bytes_human, estimate_optimizer_memory
from shared.utils.model_presets import PRESETS, ModelPreset

# Reuse Chapter 02's calibrated activation constant so the two predictors never drift.
sys.path.insert(0, "../02-memory-compute-budget")
from predict_memory import ACTIVATION_BYTES_PER_LAYER_TOKEN_H  # noqa: E402

GIB = 2**30
LIMIT = 80 * GIB  # H100-80GB

# Which of the four states each scheme shards across the world (True => divide by world).
# "opt" is Adam m+v (8 B/param); "master" is the FP32 master copy (4 B/param). Both are
# owned by the optimizer and get sharded together starting at ZeRO-1.
SCHEMES = ["DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3/FSDP"]
SHARDED = {
    "DDP":         {"weights": False, "grads": False, "opt": False, "master": False},
    "ZeRO-1":      {"weights": False, "grads": False, "opt": True,  "master": True},
    "ZeRO-2":      {"weights": False, "grads": True,  "opt": True,  "master": True},
    "ZeRO-3/FSDP": {"weights": True,  "grads": True,  "opt": True,  "master": True},
}


def predict_sharded(
    preset: ModelPreset,
    scheme: str,
    world_size: int,
    batch_size: int,
    seq_len: int,
    use_grad_checkpoint: bool = False,
    keep_master_weights: bool = True,
) -> dict[str, float]:
    """Per-rank memory breakdown (bytes) for one preset under one ZeRO/FSDP scheme."""
    n = preset.approx_params
    full = estimate_optimizer_memory(n, keep_master_weights=keep_master_weights)
    plan = SHARDED[scheme]

    def shard(nbytes: int, key: str) -> float:
        return nbytes / world_size if plan[key] else float(nbytes)

    weights = shard(full.weights_bytes, "weights")
    grads = shard(full.gradients_bytes, "grads")
    opt = shard(full.optimizer_bytes, "opt")
    master = shard(full.master_weights_bytes, "master")

    act_per_layer = ACTIVATION_BYTES_PER_LAYER_TOKEN_H * seq_len * batch_size * preset.hidden_size
    activations = act_per_layer * (1 if use_grad_checkpoint else preset.n_layers)

    return {
        "weights": weights,
        "grads": grads,
        "opt": opt,
        "master": master,
        "activations": activations,
        "total": weights + grads + opt + master + activations,
    }


def _fit(total: float) -> str:
    return bytes_human(total) + ("!" if total >= LIMIT else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--ckpt", action="store_true", help="stash only 1 layer (activation checkpointing)")
    ap.add_argument("--no-master", dest="master", action="store_false", help="drop the FP32 master copy")
    ap.add_argument("--detail", default="7B", choices=list(PRESETS), help="preset to break down by component")
    args = ap.parse_args()

    print(
        f"Per-GPU memory  (world={args.world}, B={args.batch}, S={args.seq}, "
        f"master={'on' if args.master else 'off'}, ckpt={'on' if args.ckpt else 'off'})    "
        f"'!' = exceeds {bytes_human(LIMIT)}/GPU"
    )

    # ---- summary: per-rank total for every preset under every scheme ----
    head = f"{'preset':<8s} {'params':>8s} " + " ".join(f"{s:>13s}" for s in SCHEMES)
    print("\n" + head)
    print("-" * len(head))
    for name, preset in PRESETS.items():
        cells = []
        for scheme in SCHEMES:
            m = predict_sharded(preset, scheme, args.world, args.batch, args.seq, args.ckpt, args.master)
            cells.append(f"{_fit(m['total']):>13s}")
        print(f"{name:<8s} {preset.approx_params / 1e6:>7.0f}M " + " ".join(cells))

    # ---- component breakdown for one preset (default 7B: the headline verification) ----
    preset = PRESETS[args.detail]
    print(f"\nComponent breakdown — {args.detail} preset ({preset.approx_params / 1e9:.1f}B params)")
    cols = ["weights", "grads", "opt", "master", "activations", "total"]
    head = f"{'scheme':<12s} " + " ".join(f"{c:>12s}" for c in cols) + "  fit"
    print(head)
    print("-" * len(head))
    for scheme in SCHEMES:
        m = predict_sharded(preset, scheme, args.world, args.batch, args.seq, args.ckpt, args.master)
        row = " ".join(f"{bytes_human(m[c]):>12s}" for c in cols)
        print(f"{scheme:<12s} {row}  {'yes' if m['total'] < LIMIT else 'OOM'}")


if __name__ == "__main__":
    main()
