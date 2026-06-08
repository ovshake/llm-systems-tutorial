# Chapter 01 — Transformer from Scratch (Single GPU)

**Hardware target:** 1 GPU (any H100; A10 / 3090 / 4090 also fine for the "tiny" preset).
**Time budget:** 2 weeks.
**Prerequisites:** Chapter 00 passed all checks.

---

## Why this chapter exists

You cannot debug a distributed transformer if you cannot debug a single-GPU transformer. And you cannot reason about memory and throughput tradeoffs in Chapter 02 if you don't have a model whose internals you wrote yourself.

The deliverable here is a small but *real* GPT-style model trained end-to-end on TinyShakespeare or TinyStories, with all the production-relevant knobs wired up:

- Pre-norm decoder transformer with RMSNorm, SwiGLU MLP, RoPE.
- BF16 mixed precision with explicit master-weight bookkeeping.
- Gradient accumulation for arbitrary effective batch sizes.
- Activation checkpointing (per-block).
- Robust checkpoint save / resume.
- Per-step throughput, memory, and grad-norm logging.

You'll spend half this chapter writing the model and half profiling it. Don't skip the profiling.

---

## Read first

- **"Attention Is All You Need"** (Vaswani et al., 2017) — origins of the architecture.
- **GPT-2 paper** (Radford et al., 2019) — the modern decoder-only pattern.
- **Llama / Llama-2 papers** (Touvron et al.) — RMSNorm, SwiGLU, RoPE choices we use here.
- **RoPE paper** (Su et al., 2021) — rotary position embeddings.
- **FlashAttention** (Dao et al., 2022) — we won't reimplement, but read it now.
- **Andrej Karpathy's nanoGPT** — the cleanest single-file reference for a tiny GPT. We deliberately diverge in a few places (RMSNorm + SwiGLU + RoPE instead of LayerNorm + GeLU + learned PE) to match modern decoder LLMs.
- **PyTorch `torch.amp` docs** — `autocast`, `GradScaler` (for FP16; not needed for BF16).

---

## Concepts

### 1. Modern decoder transformer = nanoGPT + 3 swaps

The decoder block we'll implement is:

```
x = x + Attention(RMSNorm(x))
x = x + SwiGLU_MLP(RMSNorm(x))
```

Differences from the original GPT block:

- **RMSNorm instead of LayerNorm** — drops the mean-subtraction and bias. Same expressiveness in practice, slightly cheaper, slightly more stable in BF16.
- **SwiGLU instead of GeLU MLP** — `SwiGLU(x) = Swish(W1 x) ⊙ (W2 x); W3 (·)`. Two up-projections fused; the gating multiplies in an extra learnable interaction. Standard in Llama / Mistral.
- **RoPE instead of learned positional embedding** — rotary embedding applied to Q and K. Better length generalization, zero parameters.

### 2. Why pre-norm, not post-norm

Original GPT was post-norm: `x = LayerNorm(x + Attention(x))`. This trains *worse* for deep models — gradients through the residual path are scaled by the norm parameters, causing instability. Pre-norm (`x = x + Attention(LayerNorm(x))`) leaves the residual stream untouched and lets gradients flow cleanly. All modern models are pre-norm.

### 3. BF16 vs FP16 vs FP32

- **FP32:** classical training dtype. ~32-bit precision, slow, big.
- **FP16:** half precision. Same range issues as FP32 at small values; needs `GradScaler` to prevent gradient underflow.
- **BF16:** "brain float". Same *range* as FP32 (8-bit exponent), reduced *precision* (7-bit mantissa). No grad scaler needed. Standard for modern LLM training. A100 / H100 / RTX 30+ all have native BF16 tensor cores.

For pure BF16 training, the right pattern is:
- Master weights in FP32 (the optimizer step uses these).
- Cast to BF16 for forward/backward.
- Gradients accumulate in BF16 (or FP32 if you can afford it).
- Optimizer states in FP32.

PyTorch's `torch.amp.autocast(dtype=torch.bfloat16)` handles forward/backward casting. The optimizer step naturally runs in FP32 because the master weights are FP32. You don't need a `GradScaler` for BF16.

### 4. Gradient accumulation

If you want global batch size `B` but only `b` fits in memory, accumulate gradients over `B/b` micro-batches, then step the optimizer once:

```python
for micro in micro_batches:
    loss = compute_loss(micro) / accumulation_steps  # rescale!
    loss.backward()                                   # grads accumulate in .grad
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

The `set_to_none=True` is a small perf win — it avoids re-zeroing the gradient tensors in place, which is a memory-bandwidth-bound op.

### 5. Activation checkpointing

For a transformer block, the forward stashes activations (typically: input to attention, attention output, MLP input, MLP intermediate) that backward needs. For long sequences, these activations are huge — often the dominant memory cost.

Activation checkpointing trades compute for memory: instead of saving activations, save just the block's input. On backward, *re-run* the block's forward to regenerate activations on the fly. Roughly +33% compute, ~1/N memory where N is the number of checkpointed blocks.

PyTorch ships `torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)`. Always use `use_reentrant=False` — reentrant mode has subtle bugs with FSDP and is being deprecated.

### 6. Checkpoint save/resume

Three things must round-trip exactly:
- Model state dict.
- Optimizer state dict.
- LR scheduler state (`scheduler.state_dict()`).
- (And the RNG state if you care about deterministic resume.)

The biggest single mistake is forgetting that `state_dict()` for tensors is *not* a copy of the data — it shares storage. If you keep a reference to the dict and continue training, you'll silently mutate it. Always serialize immediately.

For distributed training (later chapters), there are *two* problems:
- Each rank holds a shard. Naïve `torch.save` per rank → N files; reconstruction is non-trivial.
- Optimizer states are sharded with FSDP/ZeRO; pulling them together requires `gather` or `state_dict_type=FULL_STATE_DICT`.

We use single-GPU checkpoints in this chapter and revisit in Chapter 04.

---

## Hands-on

### Exercise 1 — Implement the model

Stub model is in [`model.py`](model.py). It compiles and produces output, but the implementation is missing several pieces — they raise `NotImplementedError`. Fill them in:

1. `apply_rope(x, freqs)` — apply rotary embeddings to Q and K.
2. `RMSNorm.forward` — root-mean-square normalization.
3. `Attention.forward` — multi-head self-attention with causal mask. Use `F.scaled_dot_product_attention` (which dispatches to FlashAttention on H100).
4. `SwiGLUMLP.forward` — the gated MLP.
5. `GPT.forward` — assemble the blocks, return logits.

Tests live in [`test_model.py`](test_model.py). Run `pytest test_model.py`. They check shapes, that the causal mask works (token *t* can't see token *t+1*), and that RoPE is positional-equivariant.

**Deliverable:** All tests passing.

### Exercise 2 — Train tiny preset on TinyShakespeare

```bash
python prepare_data.py --dataset tinyshakespeare
python train.py --config configs/tiny.yaml
```

This trains the `tiny` preset (10M params) for ~1000 steps in BF16. You should see:

- Loss falling from ~10.8 (random) to ~3.5 within 500 steps.
- ~50–70k tokens/sec on an H100.
- Peak memory under 4 GiB.

Inspect what the training script prints: step time, tokens/sec, peak alloc, MFU. The MFU should be ~10–20% on a single H100 for the tiny preset — small models don't saturate the GPU, and a 3× faster GPU is *even harder* to saturate with a toy model (throughput rises modestly while the peak you divide by triples vs. the old A100 target).

**Deliverable:** Training log committed as `runs/tiny/log.txt`. Loss curve plot as `runs/tiny/loss.png`.

### Exercise 3 — Scale up and break things

Run the `small` preset (125M, GPT-2 small shape):

```bash
python train.py --config configs/small.yaml
```

Now:
1. Watch peak memory. With sequence 1024, batch 32, you should fit comfortably on 80GB.
2. Double the sequence length to 2048. Note new peak memory.
3. Try sequence 4096. You'll likely OOM or get close. *Do not* fix this yet — note where it broke. Chapter 02 covers why.

**Deliverable:** A 3-row table in `NOTES.md`: (seq_len, batch, peak GiB, tokens/sec).

### Exercise 4 — Add activation checkpointing

In `model.py`, wrap each transformer block in `torch.utils.checkpoint.checkpoint(...)` when `cfg.use_grad_checkpoint=True`. Add `use_grad_checkpoint: true` to `configs/small.yaml`, rerun seq=4096.

You should now fit, with ~30% lower tokens/sec.

**Deliverable:** Table from Exercise 3 with two more rows: (seq_len=4096, no ckpt), (seq_len=4096, with ckpt). Note the memory drop and throughput hit.

### Exercise 5 — Gradient accumulation

Add `accum_steps` to the config. Train with `(micro_batch=8, accum=4)` and compare to `(micro_batch=32, accum=1)`. The loss curves should be (statistically) identical; tokens/sec should be very close; memory should be ~4× smaller for the accumulated run.

**Deliverable:** Side-by-side loss curves in `NOTES.md`.

### Exercise 6 — Checkpoint save/resume

Make sure the script saves a checkpoint every N steps and resumes seamlessly. Verify by:

1. Training to step 200, killing the process.
2. Re-running with `--resume`. The loss at step 201 should match the value from the original run at step 201, within numerical noise (≤ 1e-3 relative).

If it doesn't match, the RNG state or scheduler state isn't being restored correctly.

**Deliverable:** A short paragraph in `NOTES.md` documenting that resume works bit-for-bit (or, more honestly, "to 1e-3 relative").

### Exercise 7 — Profile a step

Use `torch.profiler` (as in Chapter 00) to capture 3 training steps. Dump the trace, open it, and identify:

1. Where does the forward pass start and end?
2. Where does backward start? (Hint: look for `aten::*_backward` kernels.)
3. What fraction of the step is in matmuls vs everything else?
4. Is there CPU overhead between kernels? (Look for gaps on the CUDA stream.)

**Deliverable:** A bullet list in `NOTES.md` with the answers.

---

## Self-check

1. In BF16 training, why don't we need `GradScaler`?
2. What does `set_to_none=True` save you, and what's the gotcha (loss of explicit "zero" state)?
3. Why is `use_reentrant=False` the right choice for `torch.utils.checkpoint`?
4. What does activation checkpointing trade for memory?
5. Why is post-norm worse than pre-norm for deep models?
6. RoPE applies a rotation to Q and K. Why not to V?
7. SwiGLU has *three* linear projections per MLP. Why is the parameter count similar to a standard 4*H GeLU MLP rather than 1.5×?

---

## Deliverables before moving on

- Working `model.py`, `train.py`, `prepare_data.py`.
- All tests in `test_model.py` passing.
- `runs/tiny/log.txt` and `runs/small/log.txt`.
- `NOTES.md` with seq-length/memory table, profile findings, and resume verification.

Once these are committed, you're ready for Chapter 02, which makes the *memory and throughput* knobs you just played with quantitative.
