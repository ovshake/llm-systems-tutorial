# Chapter 12 — Inference Engine Fundamentals

**Hardware target:** 1 GPU is sufficient; 4 makes TP-inference exercises meaningful in later chapters.
**Time budget:** 3 weeks.
**Prerequisites:** Chapter 09 — you can profile.

---

## Why this chapter exists

You've trained models for 11 chapters. Now we turn around and *serve* them. The inference workload has different physics:

- **Training**: large batch, fixed shapes, compute-bound, mostly forward+backward+optimizer in lock-step.
- **Inference**: small batch (especially decode), highly variable shapes, *memory-bandwidth-bound* during decode, many concurrent requests.

This chapter builds a "mini-vLLM" from scratch: greedy and sampled generation, naive batching, then continuous batching, all driven by a KV cache. Subsequent chapters generalize the KV cache to paged (13), then add the advanced features that production engines have (14), then we read the real engines (15, 16).

By the end you will:

1. Have an end-to-end inference loop with a KV cache, naive batching, continuous batching, and a basic request scheduler.
2. Be able to measure TTFT, ITL, throughput, p50/p95/p99 latency.
3. Understand the *prefill* vs *decode* split and why their compute profiles are different.
4. Have benchmarked against `transformers.generate()` and started a comparison against vLLM/SGLang.

---

## Read first

- **vLLM paper** ("Efficient Memory Management for Large Language Model Serving with PagedAttention" — Kwon et al., 2023). Even if you skip the paged-cache part for now, read sections on scheduling and continuous batching.
- **Orca paper** (Yu et al., 2022) — origin of continuous batching ("iteration-level scheduling").
- **vLLM documentation** — "Scheduler" and "KV cache" sections.
- **SGLang technical blog** posts on scheduling and RadixAttention.

---

## Concepts

### 1. Prefill vs decode

A request goes through two phases:

- **Prefill**: process the prompt (`P` tokens). Compute: `O(P * params)` for the forward pass. This is *compute-bound* — same physics as training a batch of size 1 with sequence P.
- **Decode**: generate one new token at a time. Each step processes 1 new token, but must attend over the full context (P + already-generated). Compute: `O(1 * params)` per step. This is *bandwidth-bound* — you read all the weights for 1 token of work.

For a 7B model, prefill of a 2000-token prompt is one big batch-1 forward (~25 ms on A100). Decode for 200 generated tokens is 200 separate forward passes, each ~5 ms, totaling ~1 second. **The decode phase is where most of the wall-clock time goes** for a typical chat-length response.

### 2. The KV cache

During decode, the attention block reads K and V for *all* past tokens at every step. Recomputing K, V from scratch every step is `O(seq²)` total work. Instead, *cache* K and V:

- After prefill, store `K, V` for every layer, every head, for all `P` prompt tokens.
- On each decode step, compute K, V for the *single new token*, append to the cache, then attend over the full cache.

KV cache size for one request: `2 * L * S * H_kv * head_dim * 2 bytes (BF16)`. For 7B with GQA-8: `2 * 32 * 2048 * 8 * 128 * 2 = 256 MB per request`. For 32 concurrent requests with 4k contexts that's 8 GB just for KV cache. This becomes the dominant memory cost during serving.

### 3. Naive batching

Batch N requests by padding everyone to the longest sequence. Run forward, slice out per-request outputs. Simple, terrible:

- Requests finish at different points → you keep computing for finished requests, wasting compute.
- New requests can't join — they wait for the current batch to finish.
- Memory is sized for the *worst-case* sequence length.

### 4. Continuous batching (Orca / vLLM)

Instead of batching at the request level, batch at the *token* level:

- Maintain a queue of in-flight requests.
- Each iteration, gather the "next token to compute" for each request — this is one token for decode-mode requests, P tokens for a new request entering prefill, or some intermediate count for chunked prefill.
- Run a single forward pass over this dynamic batch.
- Update each request's state (KV cache, generated tokens).
- Finished requests leave the batch; new requests enter on the next iteration.

The win: you never waste compute on padding, and new requests start prefill *immediately* without waiting for the batch to finish. Typical throughput gain over naive batching: 5–20×.

### 5. Scheduling

The scheduler decides which requests to include in the next iteration's batch. Constraints:

- **Memory budget** — total KV cache pages, available HBM.
- **Sequence budget** — max tokens per iteration (you can't blow up the forward).
- **Fairness / SLA** — sort by waiting time, priority, etc.

vLLM has different scheduling policies; SGLang has its own (with RadixAttention). The basic version:

```
1. Drain any requests whose generation finished last step.
2. For each running request, queue 1 token of decode.
3. For each waiting request, if memory permits, allocate KV cache and queue its prefill.
4. Run one forward over the gathered batch.
5. Update KV caches, output buffers, return any finished tokens.
```

### 6. TTFT, ITL, throughput, p99

Standard inference metrics:

- **TTFT** (Time To First Token) — wall-clock from request arrival to first decoded token. Dominated by prefill cost + queueing time.
- **ITL** (Inter-Token Latency, sometimes TPOT — Time Per Output Token) — average wall-clock between successive output tokens for one request. Dominated by decode-step latency + batch interference.
- **Throughput** — tokens/sec across all requests.
- **p95 / p99 TTFT / ITL** — tail latency. Production SLAs are tail-bound.

There's a *fundamental trade-off* between throughput and latency: bigger batches → higher throughput, longer ITL. Production serving usually has separate SLOs for TTFT, ITL, and total tokens served.

### 7. Sampling

Greedy decode (argmax) is the baseline. Real serving needs:

- **Temperature** — divide logits by T before softmax.
- **Top-k** — keep top-k logits, zero out the rest.
- **Top-p (nucleus)** — keep the smallest set whose cumulative probability ≥ p.
- **Repetition penalty** — reduce probability of recently-seen tokens.
- **Logit bias** — explicit per-token offsets (used for grammar-constrained decoding).

All of these are O(vocab) on the logits tensor each step. Optimize them as one kernel.

---

## Hands-on

### Exercise 1 — Generate without a KV cache (baseline)

In [`naive_generate.py`](naive_generate.py), implement greedy generation that *recomputes everything from scratch each step* — exactly what your Chapter 01 `generate()` does. Measure tokens/sec for prompt length 100, generating 200 tokens on the `small` preset.

**Deliverable:** Tokens/sec number. Note: this is your worst-case baseline.

### Exercise 2 — KV cache

In [`kv_cache.py`](kv_cache.py), modify your model's attention to *cache K, V* and only compute K, V for the new token during decode:

```python
class KVCache:
    """Per-layer K, V buffers preallocated to max_seq_len."""
    def __init__(self, n_layers, n_kv_heads, head_dim, max_seq_len, dtype, device): ...
    def append(self, layer_idx, k_new, v_new) -> tuple[torch.Tensor, torch.Tensor]: ...

# In Attention.forward:
def forward(self, x, cos_sin, kv_cache=None, position=0):
    q, k_new, v_new = ...
    if kv_cache is not None:
        k, v = kv_cache.append(layer_idx, k_new, v_new)
    else:
        k, v = k_new, v_new
    ...
```

Compare tokens/sec to Exercise 1.

**Deliverable:** Speedup table. Should be ~50-200× depending on prompt length.

### Exercise 3 — Naive batched generation

In [`batched_generate.py`](batched_generate.py), batch N requests with the same prompt length, generate the same number of tokens. Measure aggregate throughput.

Now make the prompts *different* lengths. Pad to the max. Note how throughput drops compared to equal-length.

**Deliverable:** Throughput table for batch sizes 1, 4, 16, 64 with mixed prompt lengths.

### Exercise 4 — Continuous batching

In [`continuous_batching.py`](continuous_batching.py), implement the Orca-style scheduler. Critical pieces:

- A request queue with each request's state (prompt, generated, KV slot, position).
- Per-iteration: collect "next token to do" for each active request.
- A forward pass that accepts a *flat* batch with `cu_seqlens` (cumulative sequence lengths) so each request's logical position is clear.
- A scheduler that admits new requests when memory permits.

You will need a *flattened* attention path: instead of `(batch, seq, hidden)`, use `(total_tokens, hidden)` with a `cu_seqlens` array, and use `F.scaled_dot_product_attention` with a manually-constructed attention mask, or use the `varlen` API from FlashAttention.

**Deliverable:** Throughput comparison: naive batching at batch 16 vs continuous batching with the same admit policy.

### Exercise 5 — Sampling kernel

In [`sampling.py`](sampling.py), implement temperature, top-k, top-p, repetition penalty. Then write a fused version using a single Triton kernel that takes raw logits and returns sampled token IDs.

Benchmark fused vs unfused.

**Deliverable:** Speedup table.

### Exercise 6 — Metrics: TTFT, ITL, p99

Add metrics collection. Simulate Poisson-arriving requests with realistic prompt-length and output-length distributions (e.g., chat: 200-token prompts, 100-token outputs; long-form: 4000-token prompts, 500-token outputs). Compute:

- TTFT p50/p95/p99.
- ITL p50/p95/p99.
- Aggregate throughput.

**Deliverable:** Two tables (chat workload, long-form workload). A short writeup of which knob (batch admit policy, max-concurrent-requests, scheduler priority) controls each metric.

### Exercise 7 — Compare to a reference

Install `transformers` if not already, and run `model.generate(...)` on the same prompts. Compare tokens/sec.

Then install vLLM (`uv pip install vllm`), wrap your same prompts through `vllm.LLM.generate`, and compare. vLLM with default settings should be 2–5× faster than your continuous-batched implementation on the same model.

**Deliverable:** Three-row throughput table: yours / transformers / vLLM.

---

## Self-check

1. Why is prefill compute-bound and decode bandwidth-bound?
2. What's the KV cache size formula for a 13B GQA model with 8 KV heads?
3. What's the difference between naive batching and continuous batching, in one sentence?
4. Why does continuous batching help even when all requests have the same length?
5. What goes wrong if you let the batch grow unbounded?
6. What's the fundamental tradeoff between TTFT and throughput?

---

## Deliverables before moving on

- Working `naive_generate`, `kv_cache`, `batched_generate`, `continuous_batching` implementations.
- Sampling kernel (fused).
- Metrics for both chat and long-form workloads.
- Comparison vs vLLM/transformers in `NOTES.md`.
- A short writeup: "Where my mini-engine loses to vLLM" — your educated guess, before Chapter 13 covers paged attention.
