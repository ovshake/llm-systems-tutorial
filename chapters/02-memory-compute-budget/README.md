# Chapter 02 — Memory & Compute Budgeting

**Hardware target:** 1 GPU (the more memory the better — H100-80GB makes the explorations cleaner).
**Time budget:** 1–2 weeks.
**Prerequisites:** Chapter 01 model trains.

---

## Why this chapter exists

In Chapter 01 you noticed sequence length 4096 OOMs and activation checkpointing fixes it. This chapter makes that observation *quantitative*: given a model spec and a hardware spec, *predict the memory and throughput before you launch*. This is the most directly interview-relevant skill in the course, and the foundation for every distributed-training decision later.

By the end you should be able to fill in this table from a napkin for any dense transformer:

| Component       | Bytes |
|-----------------|-------|
| Weights         |       |
| Gradients       |       |
| Optimizer state |       |
| Activations     |       |
| Workspace / overhead |       |
| **Total**       |       |

…and match it to within ~10% of what `torch.cuda.max_memory_allocated()` reports.

---

## Read first

- **Reducing Activation Recomputation in Large Transformer Models** (Korthikanti et al., 2022) — the canonical activation memory paper. Memorize the activation formula.
- **FlashAttention** (Dao et al., 2022) — the *memory* contribution matters as much as the speed. See Figure 1: vanilla attention keeps an `O(S²)` mask in HBM, FlashAttention does not.
- **FlashAttention-2** and **FlashAttention-3** — successive engineering improvements; the high-level story is the same.
- **NVIDIA H100 (Hopper) whitepaper** — peak TFLOPS for BF16 (989 dense) and FP8 (1,979 dense), memory bandwidth (3.35 TB/s HBM3), L2 cache size (50 MB).
- **PyTorch docs: `torch.cuda` memory tools** — `memory_allocated`, `max_memory_allocated`, `memory_summary`, `memory_history`.

---

## Concepts

### 1. The "6N" forward+backward FLOP count

For a dense transformer (ignoring attention's quadratic-in-S term), per *token*, training one step costs ≈ 6N FLOPs, where N = parameter count:

- Forward: ~2N (matmuls dominate).
- Backward: ~4N (input gradient + weight gradient).

So total FLOPs per step ≈ 6 * N * batch * seq_len. With `tokens_per_second` measured and `peak_flops_per_second` known, MFU = (6N * tps) / peak. This is the same formula `shared.utils.mfu` computes.

For long context (S > ~8k) the attention quadratic term `12 * L * H * S * S` (per layer) starts mattering. Add it in.

### 2. Memory: weights + grads + optimizer states

Steady-state per-parameter memory for BF16 mixed precision + AdamW:

```
weights        2 B
gradients      2 B
m (Adam, FP32) 4 B
v (Adam, FP32) 4 B
master (FP32)  4 B    ← present in classic mixed precision; omitted in pure-BF16 training
                       (PyTorch FSDP's default is to keep it; DeepSpeed has options)
total         16 B per param (with master) or 12 B (without)
```

So a 7B model in BF16 + AdamW with master weights needs `7e9 * 16 ≈ 112 GB` *just for the static state*. Won't fit on one H100 (80 GB). You either need ZeRO-2+ (shard optimizer states) or tensor parallel.

### 3. Activation memory is the lever you actually pull

Activation memory per layer scales as `S * B * H * (some constants)` and dominates everything else once `S` gets long. Korthikanti et al. give the formula explicitly. The intuition:

- Each transformer block stashes O(S * B * H) activations.
- With L layers, total = O(L * S * B * H).
- The quadratic-in-S attention scratch (mask + softmax intermediate) was O(L * S² * B) — *eliminated* by FlashAttention.

For a 7B model (L=32, H=4096), S=2048, B=4, BF16:
```
activations ≈ 32 * 2048 * 4 * 4096 * 2 B ≈ 2.1 GB without checkpointing
```
That sounds harmless. But add overhead, multiple intermediates per block, and grow S — at S=8192 you're at ~8 GB just for activations. By S=32768, you're out of memory.

### 4. The four levers for memory

When you can't fit:

| Lever | What it costs | What it saves |
|---|---|---|
| **Reduce batch size** | tokens/sec (and effective batch — needs grad accum) | activations linearly |
| **Activation checkpointing** | ~30% extra compute | activations by ~L× (per-block) |
| **Gradient checkpointing of attention** | even more compute | activation peak around attention |
| **Lower precision (FP8)** | quality risk; calibration | ~2× for weights/grads (native Hopper FP8) |
| **ZeRO / FSDP** | comm | weights/grads/optimizer linearly in world size |
| **Tensor parallel** | comm | weights/activations linearly in TP size |

Master this table — it's the answer to "model doesn't fit, what do I do?"

### 5. Bandwidth-bound vs compute-bound

For a kernel doing `f` FLOPs on `b` bytes of memory traffic, *arithmetic intensity* = f/b. The roofline says:

```
achievable_flops_per_second = min(peak_flops, peak_bandwidth * arithmetic_intensity)
```

On H100 BF16: peak 989 TFLOPS, peak HBM BW 3.35 TB/s. Crossover intensity ≈ 295 FLOPs/B.

- Matmul on big square matrices: intensity scales like the matrix side. Easily 200+. Compute-bound.
- Element-wise ops (layer norm, residuals, activations): intensity ≈ 1–2. Bandwidth-bound.
- Softmax / attention: depends.

The implication: when you're trying to speed something up, the first question is always "is this thing compute-bound or bandwidth-bound?" Kernel fusion helps bandwidth-bound ops; better tiling helps compute-bound ones.

### 6. FlashAttention in one paragraph

Vanilla attention computes `softmax(Q K^T / √d) V` by *materializing* `Q K^T` and the softmax output, each of size `S × S` per head per layer. For long S this is huge. FlashAttention tiles the computation along S, keeps the running max and sum for the online softmax, and never writes the full `S × S` intermediate to HBM. Speedup comes mostly from avoiding HBM traffic; memory savings are dramatic.

On H100, `F.scaled_dot_product_attention(..., is_causal=True)` dispatches to FlashAttention 2 automatically (provided `q, k, v` shapes are right). You generally don't need to pip-install `flash-attn` until later chapters; the Hopper-tuned FlashAttention 3 lives in `flash-attn`'s `hopper/` build, which you'll meet in Chapter 11.

---

## Hands-on

### Exercise 1 — Predict memory

In [`predict_memory.py`](predict_memory.py) write a function:

```python
def predict_memory(cfg: GPTConfig, batch_size: int, seq_len: int,
                   use_grad_checkpoint: bool,
                   keep_master_weights: bool = True) -> dict[str, int]:
    """Return a dict of bytes for each of: weights, gradients, optimizer_states,
    master_weights, activations_estimate."""
```

For activations, use the Korthikanti formula (simplified, ignoring attention quadratic when FlashAttention is in use):
```
A_per_layer ≈ S * B * H * (10 + 24 * (1 if not checkpointed else 0))    # rough
A_total ≈ L * A_per_layer
```
(The constants vary; calibrate to your model. The *shape* of the formula is what matters.)

Run it for the `tiny`, `small`, `medium`, `1B`, `3B`, `7B` presets at `(B, S) = (4, 2048)`. Print the table.

**Deliverable:** `predict_memory.py` output saved to `NOTES.md`, alongside what fits on 80 GB.

### Exercise 2 — Measure memory and compare

Write [`measure_memory.py`](measure_memory.py): instantiate each preset on GPU, run forward+backward on a `(B=4, S=2048)` batch, print:

```
preset  predicted_total  measured_peak  delta_pct
```

Your prediction should land within ±15% of the measured number. If it's wildly off, your activation constant is wrong — calibrate.

**Deliverable:** Table in `NOTES.md`. Write one paragraph on what surprised you.

### Exercise 3 — MFU sweep

In [`mfu_sweep.py`](mfu_sweep.py), for the `small` preset, sweep `(batch_size ∈ {4, 8, 16, 32, 64}) × (seq_len ∈ {512, 1024, 2048, 4096})`. For each cell, print tokens/sec and MFU. Skip configurations that OOM.

You should see MFU rise with batch size (kernel launch overhead amortizes) and rise then plateau with seq length (FlashAttention keeps things compute-bound). The "best" cell on an H100 for this model is usually around `(B=32, S=2048)` → 25–35% MFU (a 125M model in BF16 leaves H100's tensor cores partly idle — you need bigger models, or FP8, to push the percentage higher).

**Deliverable:** Heatmap in `mfu_heatmap.png`. Write a paragraph identifying the dominant bottleneck for each regime: small B → launch overhead; small S → low arithmetic intensity; large B+S → memory bandwidth.

### Exercise 4 — Step-time breakdown

For the `small` preset at the best MFU configuration, use the PyTorch profiler to break the step into:

- Data loading (CPU + H2D).
- Forward pass — attention vs MLP vs norms vs residuals.
- Loss + backward — same decomposition.
- Optimizer step.
- Other (`barrier`, kernel launch gaps).

Use `with record_function("...")` blocks around each. The profiler's "key averages" table gives you cumulative time per label.

**Deliverable:** Stacked bar chart in `step_breakdown.png`, plus a 1-paragraph reading: where is your time going?

### Exercise 5 — Activation checkpointing memory math, validated

For `small`, predict and then measure the activation memory savings from activation checkpointing. Predicted: roughly `activations / L * (sqrt(L))` if you use sqrt-L checkpointing, or `activations / L` if you checkpoint every block. Measured: take `max_memory_allocated()` before vs after toggling `cfg.use_grad_checkpoint=True`.

**Deliverable:** Two-row table: (no-ckpt, with-ckpt). Predicted vs measured.

### Exercise 6 — FlashAttention vs eager

Force the math kernel by wrapping the attention in `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])`. Run with seq_len=4096, batch=2. Measure peak memory and time. Compare to default (FlashAttention) on the same shape.

You should see FlashAttention use much less memory (no S×S materialized) and run faster.

**Deliverable:** Table in `NOTES.md`. Note where the math kernel OOMs that Flash does not.

---

## Self-check

1. For a 13B model in BF16 + AdamW with master weights, how many GB of static state per GPU with no sharding?
2. Why is the FlashAttention speedup mostly a memory-traffic story, not a compute story?
3. What is the arithmetic intensity of a layer norm? Why is it bandwidth-bound?
4. What's the MFU formula for a transformer at long context, and where does the attention term come from?
5. Activation checkpointing: how do you decide *which* blocks to checkpoint when you can't afford to checkpoint all of them?
6. Given a 7B model and 8× H100-80GB, sketch a parallelism strategy. Argue your TP/PP/DP degrees.

---

## Deliverables before moving on

- `predict_memory.py`, `measure_memory.py`, `mfu_sweep.py` all runnable.
- `NOTES.md` with all tables and plots above.
- A one-page writeup titled "How to choose batch size and seq length for a transformer" — *your own words*, no copy-paste. This is interview prep.
