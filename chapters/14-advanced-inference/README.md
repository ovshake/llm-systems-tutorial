# Chapter 14 — Advanced Inference

**Hardware target:** 8× H100-80GB (for tensor-parallel inference and bigger models).
**Time budget:** 4 weeks.
**Prerequisites:** Chapter 13 — your engine has paged KV and prefix caching.

---

## Why this chapter exists

Your mini-engine from Chapters 12–13 is competitive on small models but loses to vLLM/SGLang on real workloads. Why? Because production engines layer on a stack of optimizations:

- **Speculative decoding** — propose multiple tokens with a small draft model, verify with the big one. 2–3× throughput win for many workloads.
- **Tensor-parallel inference** — split the model across GPUs (same as Chapter 05's training TP, but with a different optimal config for decode).
- **Quantization** — INT8 / FP8 / INT4 weights, optional FP8 activations. ~2× throughput, ~4× memory savings.
- **Disaggregated prefill/decode** — separate GPU pools for the two phases.
- **Multi-LoRA serving** — serve multiple LoRA adapters on the same base model.
- **CUDA graphs for decode** — kill kernel-launch overhead.

This chapter goes through each. You will not implement *all* of these to production quality (that would be vLLM's whole codebase), but you'll implement enough to understand the tradeoffs.

By the end you will:

1. Have implemented speculative decoding with a draft model.
2. Have run your engine with TP=4 and benchmarked it.
3. Have quantized your model to INT8 weights and measured the win.
4. Have used CUDA graphs to speed up decode.
5. Be able to articulate when to enable disaggregated prefill/decode and when not to.
6. Have implemented (or read in detail) multi-LoRA serving.

---

## Read first

- **"Fast Inference from Transformers via Speculative Decoding"** (Leviathan et al., 2022).
- **"Accelerating LLM Inference with Staged Speculative Decoding"** (Spector & Re, 2023).
- **EAGLE** / **Medusa** papers — modern speculative decoding variants.
- **GPTQ** (Frantar et al., 2022) — INT4 weight quantization.
- **AWQ** (Lin et al., 2023) — activation-aware INT4.
- **SmoothQuant** (Xiao et al., 2022) — INT8 activations.
- **FP8 GEMM** — native on your H100 via `torch._scaled_mm` (and the FP8 paths in vLLM/CUTLASS); this is the production default for FP8 inference, not just a reference.
- **"Punica: Multi-Tenant LoRA Serving"** (Chen et al., 2023).
- **vLLM docs on speculative decoding, quantization, multi-LoRA.**

---

## Concepts

### 1. Speculative decoding

Big model `M_big` produces high-quality tokens but is slow. Small model `M_draft` is fast but lower quality. The decode step:

1. `M_draft` greedily generates K next tokens (e.g., K=4).
2. Run `M_big` on the prompt + K draft tokens *in parallel* — one forward pass over K+1 positions.
3. Compare token by token. Accept as many as match (or, with stochastic acceptance, accept under a probability ratio).
4. The first rejected token (or all matched + 1 sampled from big model) is the output.

If `M_draft` matches `M_big` on average for ~3/4 tokens, you get ~3× throughput. The big model now uses its capacity for *verifying* multiple tokens in one batch-K forward (which is decode-shape × K, almost as cheap as one decode step).

Variants:
- **EAGLE**: train a draft head on the big model itself for higher acceptance.
- **Medusa**: parallel decoding heads.
- **Lookahead decoding**: no separate draft model.

### 2. Tensor-parallel inference

Same column/row parallel pattern as training (Chapter 05), but with one major difference: **decode is bandwidth-bound, not compute-bound**. The TP all-reduce in row-parallel costs you a meaningful fraction of the step, and there's less compute to overlap it with.

Optimal TP degree for inference is often *smaller* than for training:
- 7B model: TP=1 fits easily on H100-80GB. TP=2 sometimes helps if it doubles available memory bandwidth.
- 30B model: TP=2 or TP=4.
- 70B model: TP=4 or TP=8.

The sweet spot is "smallest TP that lets the model fit with decent batch."

### 3. Quantization

Three independent axes:

- **Weights**: BF16 baseline. INT8 (cheap), INT4 (GPTQ/AWQ), FP8 (Hopper+).
- **Activations**: usually keep in BF16. INT8 activations (SmoothQuant) requires careful calibration. FP8 activations on Hopper.
- **KV cache**: BF16 → FP8 or INT8. Saves ~2-4× KV memory. Tiny quality hit.

Weight-only INT8 with BF16 activations is the easiest win and what most production engines default to. The kernel does:

```
W_int8: int8[N, K]
W_scale: bf16[N]    # per-channel
x_bf16: bf16[B, K]

# Inside kernel:
W_bf16 = (W_int8 - zero_point) * W_scale   # dequant on the fly
out = x @ W_bf16^T                          # BF16 GEMM
```

Memory: half. Compute: same (you dequant in registers; the GEMM is still BF16). Throughput win comes from less HBM traffic.

### 4. Disaggregated prefill/decode

Prefill is compute-bound; decode is bandwidth-bound. Different hardware is optimal:
- H100 (what you have) is fine for both.
- For prefill at scale: the highest-FLOPs GPUs (H100/H200, Blackwell).
- For decode: high-bandwidth, lots of memory (e.g. H200's 141 GB HBM3e).

Disagg means: send prefill to one pool, then transfer the KV cache to a decode pool, continue. Pros: independent scaling per phase, better utilization. Cons: KV transfer between pools (which is just bytes over RDMA) adds latency, complexity is huge.

Worth it for very high-volume workloads. Not worth it for hobby setups. Read about it; don't implement.

### 5. Multi-LoRA serving

For LoRA-fine-tuned models, each LoRA is small (e.g., 1–10MB). You can serve many LoRA adapters on the same base model in the same engine:

- Hold base weights once.
- Hold N LoRAs in GPU memory.
- Per request, select the right LoRA and apply its delta (`Wx + (B A)x = (W + BA)x`).

The trick is *batching* requests that use different LoRAs. Punica / S-LoRA handle this with grouped GEMMs: each row of the batch is multiplied by a different LoRA pair, fused into one kernel.

### 6. CUDA graphs for decode

Each decode step is ~10–50 kernel launches × ~5μs launch overhead = ~50–250 μs of CPU. For a 5 ms decode, that's a meaningful chunk. CUDA graphs collapse it to one launch.

You can't just record once and replay — input shapes change as the batch fills up. Solution: record one graph per batch size you care about. Replay the right one based on the current iteration's batch. vLLM does this with `capture-mode` graphs for ~16 batch sizes.

### 7. The optimization order

If you're optimizing serving from scratch, the rough payoff order:

1. Paged KV cache (already done in Chapter 13).
2. Continuous batching (Chapter 12).
3. INT8 / FP8 weight quantization.
4. CUDA graphs for decode.
5. Speculative decoding.
6. Prefix caching (Chapter 13).
7. Tensor-parallel inference at large model sizes.
8. Chunked prefill.
9. (Maybe) disaggregated serving.

The first five gets you most of the way.

---

## Hands-on

### Exercise 1 — Speculative decoding

In [`speculative.py`](speculative.py), implement greedy speculative decoding:

1. Train (or load) a small draft model (the `tiny` preset from Chapter 01).
2. The verifier is your `small` preset.
3. Per step: draft proposes K=4 tokens; verifier runs on prompt + K drafts in one forward pass; accept matching prefix; sample one new token from the verifier's distribution at the first mismatch.

Measure:
- Acceptance rate (mean tokens accepted per round).
- Throughput vs. plain decode.

**Deliverable:** Table of acceptance rate and tokens/sec for K=2, 4, 8.

### Exercise 2 — Tensor-parallel inference

Adapt your mini-engine to use TP=4. The TP path you wrote in Chapter 05 needs slight changes for inference:

- The KV cache is sharded along the head dim (each TP rank holds 1/TP heads).
- The output projection's all-reduce happens once per decode step.

Benchmark TP=1 vs TP=2 vs TP=4 on the same model. For your small models, TP=1 will win on throughput; TP=4 is interesting when the model doesn't fit on one GPU.

Try also TP=4 on the `7B` preset (you couldn't run it on TP=1 anyway).

**Deliverable:** Throughput table. A paragraph on when TP wins for inference vs training.

### Exercise 3 — INT8 weight quantization

In [`int8_quant.py`](int8_quant.py):

1. For each `nn.Linear` in your model, quantize the weight to INT8 with per-channel scales: `w_int8 = round(w / scale).clamp(-128, 127); scale = w.abs().max(dim=-1) / 127`.
2. Replace `nn.Linear` with `QuantLinear` that stores `w_int8, scale` and dequantizes on-the-fly inside the matmul.
3. Use either Triton or `torch._int_mm` for the actual INT8 GEMM (note: still requires BF16 outputs; the dequant is in the kernel).

Measure:
- Quality (perplexity on val set).
- Throughput.
- Memory.

**Deliverable:** Three-row table: BF16 / INT8 weight-only / (optional) INT8-INT8.

### Exercise 4 — CUDA graphs for decode

Capture a CUDA graph for the decode step at fixed batch size B. Replay it for N steps. Measure tokens/sec.

Repeat for several batch sizes (1, 4, 16). You'll need one graph per batch size.

**Deliverable:** Throughput before/after for each batch size.

### Exercise 5 — Multi-LoRA serving (lighter exercise)

Train 3 LoRAs on different (tiny) downstream datasets — instruction-following, code, chat — using PEFT. Hold all 3 in your inference engine. Per request, the LoRA ID is part of the request.

Implement two batching policies:
- **Per-LoRA batching**: group requests by LoRA, run separate forwards. Simple, no fusion.
- **Mixed batching**: one fused forward, with the LoRA delta applied per row. Use a grouped-GEMM approach (per-row LoRA).

Compare throughput.

**Deliverable:** Two-row table. Note: production implementations like Punica/S-LoRA are much more sophisticated; this is a stepping-stone.

### Exercise 6 — Chunked prefill revisit

You implemented chunked prefill in Chapter 13. Now sweep the chunk size (256, 512, 1024, 2048) and measure ITL p99 in a workload with a mix of long prefills and decode requests.

**Deliverable:** Plot. There's a sweet spot — chunks too small mean too many iterations; chunks too big mean ITL spikes.

### Exercise 7 — KV cache quantization

Quantize your KV cache to FP8 — native on your H100 (E5M2 gives more exponent headroom; E4M3 more precision), no INT8 simulation needed. Measure quality and throughput.

**Deliverable:** Two-row table.

### Exercise 8 — End-to-end engine benchmark

Take your mini-engine with paged KV, continuous batching, prefix caching, chunked prefill, speculative decoding, INT8 weights, CUDA graphs. Run the chat-workload benchmark from Chapter 12 Ex. 6.

Compare to vLLM with the same workload.

You'll still lose to vLLM, but by less. Where do you still lose? Identify *one specific thing* to address.

**Deliverable:** Throughput / TTFT / ITL p99 table. One paragraph on the remaining gap.

---

## Self-check

1. Why does speculative decoding speed up decode without changing the model's output distribution?
2. What's the acceptance-rate × cost-ratio formula that determines spec decoding's break-even?
3. When does TP help for inference, and when does it hurt?
4. Why is INT8 weight-only the easiest quantization win?
5. Why are CUDA graphs huge for decode but small for prefill?
6. What's the trade-off in chunk size for chunked prefill?

---

## Deliverables before moving on

- Working speculative decoder with measured acceptance + speedup.
- TP=4 inference working.
- INT8 weight-only quantization with perplexity & throughput comparison.
- CUDA graphs decode benchmark.
- Multi-LoRA naive vs grouped comparison.
- End-to-end engine vs vLLM table.
