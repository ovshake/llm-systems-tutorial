# Chapter 04 — FSDP & ZeRO (Memory Sharding)

**Hardware target:** 8× H100-80GB. Many exercises will fit on 2 GPUs but you lose the comparison.
**Time budget:** 2–3 weeks.
**Prerequisites:** Chapter 03 — DDP scaling works and you can read its profile.

---

## Why this chapter exists

DDP keeps a full copy of the model + grads + optimizer on every GPU. That's wasted memory: in DDP all eight ranks each store the exact same FP32 Adam states. Fixing this is "memory sharding" — split the model state across GPUs, then gather/scatter pieces as needed.

There are two equivalent framings of the same idea:

- **DeepSpeed ZeRO** (Rajbhandari et al.) — invented the concept. Three stages: optimizer-state sharding (ZeRO-1), gradient sharding (ZeRO-2), parameter sharding (ZeRO-3).
- **PyTorch FSDP** (Fully Sharded Data Parallel) — PyTorch-native implementation, essentially equivalent to ZeRO-3 with finer-grained options.

You can train a 7B model on your 8×H100-80GB node with FSDP. You cannot with DDP. That's the punchline.

By the end of this chapter you will:

1. Understand the memory math for ZeRO-{1,2,3} and FSDP's `ShardingStrategy.{FULL_SHARD, HYBRID_SHARD, SHARD_GRAD_OP}`.
2. Have trained a 1B–3B model with FSDP that wouldn't fit with DDP.
3. Be able to read FSDP's wrapping policy, mixed precision, and CPU offload settings, and explain the *cost* of each.
4. Know FSDP's failure modes: too-small wrapping units = excessive comm; too-large = OOM; activation checkpointing interactions.

---

## Read first

- **ZeRO paper** (Rajbhandari et al., 2019).
- **PyTorch FSDP tutorial** — `pytorch.org/tutorials → fsdp_tutorial.html`. Includes the wrapping policy section.
- **FSDP getting started** — official docs page in `torch.distributed.fsdp`.
- **DeepSpeed ZeRO docs** — at least the config reference page; you'll write a config in Exercise 5.
- **Ultra-Scale Playbook chapter on ZeRO/FSDP.**
- Optional: **FSDP2** preview — the newer, more compositional FSDP design landing in PyTorch ≥ 2.4 (`torch.distributed._composable.fsdp`). The mental model is the same; we'll mostly use FSDP1 because the docs are more mature.

---

## Concepts

### 1. The four memory states

For a parameter `p`:
- **Parameter** (BF16): used in forward and backward.
- **Gradient** (BF16): produced by backward.
- **Optimizer states** (Adam: m, v in FP32): consumed by optimizer step.
- **Master weight** (FP32): the FP32 source-of-truth that the optimizer updates and that gets cast to BF16 each step.

ZeRO/FSDP shard *each of these independently*:

| Stage / strategy            | Param | Grad | Opt state |
|-----------------------------|-------|------|-----------|
| DDP                         | repl  | repl | repl      |
| ZeRO-1                      | repl  | repl | shard     |
| ZeRO-2                      | repl  | shard| shard     |
| ZeRO-3 / FSDP `FULL_SHARD`  | shard | shard| shard     |

"Shard" means each rank holds 1/N of the state; "repl" means each rank holds a full copy.

### 2. The cost of sharding: communication

You don't get sharded memory for free. ZeRO-3 / FSDP adds two collectives *per forward step* and one per backward step, per FSDP unit:

- Forward: `all_gather` the shard from all ranks → reconstructed full parameter for compute → discard the gathered copy after.
- Backward: same `all_gather` to get the parameter; after backward, `reduce_scatter` the gradient.

The all-gather time scales with parameter bytes; for a 7B model in BF16 that's ~14 GB of all-gather per forward pass (split across FSDP units). Even on NVLink that's not free.

The *win* is that you can fit larger models. The *cost* is comm; if comm > the compute it overlaps, you regress vs DDP. FSDP is faster when:

- The model is large enough that DDP OOMs.
- Or: the model is large enough that the all-reduce in DDP is huge (and FSDP's reduce-scatter is only half the bytes).

### 3. Wrapping policy

FSDP shards at the granularity of "units." A unit is what gets all-gathered together. For a transformer the natural choice is "one block = one unit":

```python
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
policy = ModuleWrapPolicy({Block})   # your Block class from Chapter 01
```

Why per-block?

- Each all-gather is the size of one block's parameters → manageable.
- Forward can overlap: while block N runs compute, FSDP can prefetch block N+1's parameters in the background.
- After block N's backward, its gradient gets reduce-scattered while block N-1's backward runs.

Too-large wrap (e.g., wrap the whole model) defeats the point — you all-gather every parameter on the first forward, killing the memory win. Too-small wrap (wrap each linear) means a flood of tiny all-gathers, each of which has fixed launch overhead — comm-bound.

### 4. Mixed precision policy

```python
MixedPrecision(
    param_dtype=torch.bfloat16,        # how params live during compute
    reduce_dtype=torch.bfloat16,       # how grads get reduced
    buffer_dtype=torch.bfloat16,
)
```

If `reduce_dtype=torch.float32`, you get FP32 accuracy on gradient reductions but 2× the comm. For BF16 training this is usually unnecessary; the BF16 reduction is fine.

### 5. `HYBRID_SHARD`

`HYBRID_SHARD` shards across one dimension (intra-node), replicates across another (across nodes). On your 8-GPU single node it's identical to `FULL_SHARD`; on a multi-node cluster it's the default for the right reason — full shard across the slow inter-node link is brutal. We'll discuss in Chapter 07.

### 6. CPU offload

`offload_params=True` offloads sharded parameters to CPU when not in use. Lets you fit even bigger models by paying massive H2D/D2H traffic. Mostly useful for fine-tuning at home with one GPU; production almost never uses it because it tanks throughput. We touch it briefly.

### 7. FSDP checkpointing is its own thing

There are three styles:

- `FULL_STATE_DICT`: gather everything to rank 0, save one file. Simple, but the whole model briefly lives on rank 0 → OOM risk for big models.
- `SHARDED_STATE_DICT`: each rank saves its shard. Fastest, but resume must use the same world size unless you reshard.
- `LOCAL_STATE_DICT`: each rank saves a flat parameter buffer; nearly impossible to use elsewhere.

For research code, `FULL_STATE_DICT` is fine. For production-scale training, `SHARDED_STATE_DICT` (plus an offline rescatter tool) is standard.

---

## Hands-on

### Exercise 1 — Memory accounting prediction

In [`memory_predict_sharded.py`](memory_predict_sharded.py), extend Chapter 02's predictor to model ZeRO stages and FSDP `FULL_SHARD`. For each preset at `(B=4, S=2048)` and `world_size=8`, print per-GPU memory under:

- DDP
- ZeRO-1
- ZeRO-2
- ZeRO-3 / FSDP `FULL_SHARD`

**Deliverable:** Table in `NOTES.md`. Verify that ZeRO-3 lets the 7B preset fit on 80GB per rank with sane batch.

### Exercise 2 — FSDP wired up

In [`train_fsdp.py`](train_fsdp.py) (provided skeleton), wrap the model in FSDP with:

- `ShardingStrategy.FULL_SHARD`
- `auto_wrap_policy=ModuleWrapPolicy({Block})`
- `mixed_precision=MixedPrecision(param_dtype=bf16, reduce_dtype=bf16, buffer_dtype=bf16)`
- `device_id=local_rank`

Train the `small` preset on 8 GPUs. Confirm it converges to roughly the same loss as the DDP version.

```bash
torchrun --standalone --nproc_per_node=8 train_fsdp.py --config configs/small.yaml
```

**Deliverable:** Loss curve comparison (DDP vs FSDP) at the same global batch.

### Exercise 3 — Bigger model

Train the `1B` preset (1.1B params) under FSDP. Try DDP first — it should OOM. Then FSDP `FULL_SHARD` — it should fit.

Run `train_fsdp.py --config configs/1B.yaml`. Confirm:

- Peak memory per rank is roughly `(weights + grads + opt) / world_size + activations + overhead`.
- Tokens/sec is lower than DDP-equivalent of a smaller model (you pay for comm).

**Deliverable:** Three runs in `NOTES.md`: (small, DDP), (small, FSDP), (1B, FSDP). For each: peak memory per rank, step time, tokens/sec.

### Exercise 4 — Wrapping policy ablation

Run the `1B` preset three times:

1. `ModuleWrapPolicy({Block})` — recommended.
2. No wrap policy (wraps only the root module) — should OOM or be very slow.
3. `size_based_auto_wrap_policy(min_num_params=int(1e6))` — wraps any module with ≥ 1M params.

Compare peak memory and tokens/sec.

**Deliverable:** Table. A paragraph on why "wrap the whole model" defeats the point.

### Exercise 5 — DeepSpeed ZeRO comparison

Install DeepSpeed (`uv pip install deepspeed`). In [`train_zero.py`](train_zero.py) provide the same training loop driven by DeepSpeed, with three configs:

- `configs/zero1.json`: stage 1.
- `configs/zero2.json`: stage 2.
- `configs/zero3.json`: stage 3.

Run all three on the `1B` preset. Compare memory and tokens/sec to FSDP `FULL_SHARD`.

**Deliverable:** 4-row table. For each (ZeRO-1, ZeRO-2, ZeRO-3, FSDP): per-rank memory, tokens/sec, brief notes on configuration complexity.

### Exercise 6 — Activation checkpointing under FSDP

Enable activation checkpointing under FSDP. The integration is fiddly — see `torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing`. Wrap your `Block` instances *after* FSDP wrapping. Confirm it shrinks activation memory.

Train the `3B` preset with FSDP on 8 GPUs. Push `(batch, seq)` up until it OOMs *without* activation checkpointing, then show it fits *with* it. Note FSDP shards params/grads/optimizer but **not** activations — so this OOM crossover is set by your per-GPU batch×seq, not by the GPU count (going from 4 to 8 GPUs frees optimizer-state memory, not activation memory).

**Deliverable:** Table at your chosen `(batch, seq)`: (3B without ckpt → OOM), (3B with ckpt → fits).

### Exercise 7 — `SHARDED_STATE_DICT` round-trip

Save a sharded checkpoint mid-training. Re-launch the trainer at a different rank count (e.g., 4 → 2). Without a reshard tool, this is *hard*. Read PyTorch's `_shard.checkpoint` docs to see how it's done in practice. You don't need to write a full reshard tool; reading the docs and writing 1 paragraph on the design is enough.

**Deliverable:** Paragraph in `NOTES.md` describing the resharding problem.

---

## Self-check

1. ZeRO-1, ZeRO-2, ZeRO-3 — what does each shard, and what's the extra communication for each?
2. For a 13B model in BF16+AdamW (no master weights), how much per-GPU memory under FSDP `FULL_SHARD` at world size 4 (params/grads/opt only)?
3. Why does the FSDP wrapping policy matter for both memory and comm?
4. When would you use `HYBRID_SHARD` over `FULL_SHARD`?
5. Why is CPU offload almost always slow in production?
6. What's the failure mode of `FULL_STATE_DICT` checkpoints for very large models?

---

## Deliverables before moving on

- `memory_predict_sharded.py` matches measured per-rank memory.
- `train_fsdp.py` working at small and 1B presets.
- `train_zero.py` working at three ZeRO stages.
- 3B with FSDP + activation checkpointing fits.
- `NOTES.md` covers wrapping policy ablation, mixed precision choices, and a 1-paragraph on checkpoint resharding.
