# Chapter 16 — SGLang Deep Dive & Contributions

**Hardware target:** 4× A100-80GB; some experiments degrade gracefully to fewer.
**Time budget:** 4 weeks.
**Prerequisites:** Chapter 15 — you've done a vLLM source tour.

---

## Why this chapter exists

SGLang is the second major open-source serving framework. It comes from a different angle than vLLM:

- vLLM's killer feature is PagedAttention. SGLang's killer feature is **RadixAttention** — generalized prefix caching as a radix tree.
- vLLM is C++-Python heavy. SGLang has a Python frontend (DSL for structured generation) plus a Rust + CUDA + Python runtime.
- SGLang invests heavily in *grammar-constrained decoding*, *structured outputs*, and *batched JSON / regex / function-calling*.

For someone targeting LLM systems engineering, knowing both engines well is a major signal. Compare/contrast questions are common.

By the end you will:

1. Have read enough SGLang source to map it to its vLLM equivalent.
2. Have run SGLang and benchmarked it on the same workloads as vLLM.
3. Have read Mini-SGLang — the deliberately-simplified educational version of the runtime.
4. Have understood RadixAttention end-to-end and reproduced its prefix-reuse benefits.
5. Have planned a contribution.

---

## Read first

- **SGLang docs** — start with `Getting Started`, `Server Args`, `Performance Tuning`, `Custom Models`.
- **SGLang paper** — `SGLang: Efficient Execution of Structured Language Model Programs`.
- **RadixAttention paper / blog** — the key contribution.
- **SGLang contributing guide** — points new contributors to Mini-SGLang, code walkthroughs, profiling guides.
- **Mini-SGLang** — a simplified educational version of the runtime. Read it carefully; it's designed for what you're about to do.

---

## The SGLang source map

```
sglang/
├── python/sglang/
│   ├── srt/                              # the runtime (server-side)
│   │   ├── managers/
│   │   │   ├── scheduler.py              # the scheduler (analog of vllm/core/scheduler)
│   │   │   ├── tokenizer_manager.py      # tokenization, detokenization
│   │   │   └── tp_worker.py              # per-TP-rank worker
│   │   ├── mem_cache/
│   │   │   ├── radix_cache.py            # the radix tree for prefix reuse
│   │   │   ├── memory_pool.py            # KV pool allocator
│   │   │   └── chunk_cache.py            # chunked prefill / chunk reuse
│   │   ├── layers/                       # model layers, attention, fused kernels
│   │   ├── models/                       # model definitions
│   │   └── server.py / server_args.py    # entrypoint
│   ├── lang/                             # the structured-generation DSL
│   └── api.py
├── sgl-router/                           # Rust-based load-balancer / router
├── sgl-kernel/                           # CUDA kernels (Python + CUDA + C++)
└── benchmark/
```

The runtime split is similar to vLLM: scheduler → worker → model → attention. RadixAttention is a *replacement* for the per-request block table — the radix tree generalizes the "request owns its blocks" model to "tree of token sequences owns blocks."

---

## Concepts

### 1. RadixAttention end-to-end

```
Prefix tree:
                ┌── "what is the capital of France?" (cached KV blocks)
"You are a"  ──┤
helpful AI"     ├── "explain quantum entanglement in simple terms" (cached blocks)
                └── "summarize this article: ..." (cached blocks)
```

- Each *edge* in the tree corresponds to a token segment.
- Each *node* holds the KV blocks for the path from root to here.
- A new request with prompt `"You are a helpful AI. Translate to French: ..."` walks the tree:
  - Token-by-token until it diverges.
  - Diverges at "Translate" — branches a new leaf, allocates fresh KV blocks for "Translate to French: ...".
  - Reuses all KV from the shared parent path.

The data structure is a *radix tree* (compressed trie). LRU eviction of leaves keeps it bounded.

Trade-offs:
- Massive win for repeated prefixes (chatbots, instruction templates, function-calling preludes).
- Some bookkeeping overhead per token.
- More complex than vLLM's simpler block manager.

### 2. The scheduler

`python/sglang/srt/managers/scheduler.py`. Analogous to vLLM's, but:

- Token-level admission decisions integrate with the radix tree (we admit a request "up to the matching prefix", then schedule the unmatched-tail prefill).
- Chunked prefill is first-class.
- Structured-decoding constraints (grammar, regex, JSON) can override sampling.

### 3. Grammar / regex / JSON-constrained decoding

SGLang lets you generate text constrained by a grammar (e.g., regex, BNF, JSON schema). The implementation:

- At each step, compute the mask of *legal next tokens* given the current state of the grammar automaton.
- Set illegal tokens' logits to -inf before sampling.

Implementing this efficiently is non-trivial — the per-state legal-tokens computation can be the bottleneck. SGLang uses `outlines`-style automata and precomputes masks.

### 4. The Python DSL (frontend)

`python/sglang/lang/`. Lets you write programs like:

```python
@function
def question_answer(s, document, question):
    s += "Document: " + document + "\n"
    s += "Question: " + question + "\n"
    s += "Answer: "
    s += gen("answer", max_tokens=100)
```

Compiled into a server-side execution graph. Includes constructs like `select` (constrained choice) and `for ... gen` for batched generation.

You don't need to master the DSL to contribute to the runtime, but knowing it exists clarifies a lot of design decisions in `srt/`.

### 5. Performance optimizations

SGLang ships:
- FlashInfer attention backend (the modern recommended choice on Hopper and Ampere).
- Their own paged KV.
- CUDA graphs (very aggressively).
- A Rust router for multi-instance load balancing.
- Custom kernels in `sgl-kernel/`.

The result is generally similar throughput to vLLM in raw decode, with significant wins on prefix-heavy workloads.

---

## Hands-on

### Exercise 1 — Install and serve

```bash
uv pip install "sglang[all]"
python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --tp 4
```

Hit it from a client. Verify it works.

**Deliverable:** Captured response in `NOTES.md`.

### Exercise 2 — Tour the runtime

Spend a day reading `python/sglang/srt/`. Cross-reference each subsystem to its vLLM equivalent. In `NOTES.md`:

- Map scheduler.py (SGLang) ↔ scheduler.py (vLLM): what's the same, what's different.
- Same for memory_pool / block_manager.
- Find the radix tree implementation and read it carefully — explain it in your own words.

### Exercise 3 — Read Mini-SGLang

Find the Mini-SGLang code in the repo (`python/sglang/test/` or similar — check current location). Read it end-to-end. It's designed to be educational.

**Deliverable:** A 1-page writeup in `NOTES.md` summarizing the runtime in the simplest possible terms.

### Exercise 4 — Reproduce the RadixAttention benefit

Run two workloads:

- **Chatbot workload**: 100 requests, all share a 500-token system prompt + few-shot examples, then differ.
- **Random workload**: 100 requests with no shared prefix.

Run each on:
- vLLM with prefix caching off.
- vLLM with prefix caching on.
- SGLang.

Measure TTFT and throughput.

**Deliverable:** Six-cell table. SGLang should be roughly at parity with vLLM-prefix-on for the shared-prefix workload, and at parity with vLLM-off for the random workload (or slightly better for both).

### Exercise 5 — Profile SGLang under load

Same profiling exercise as Chapter 15 Exercise 5, applied to SGLang. Find where its time goes. Compare to vLLM's profile.

**Deliverable:** Annotated profiles. A 1-paragraph comparison.

### Exercise 6 — Structured output: JSON-constrained generation

Use SGLang's `regex=...` or `json_schema=...` constraint to force the model to emit a JSON object matching a schema. Try a few schemas.

Measure: what's the per-token overhead of grammar masking? Profile to see.

**Deliverable:** Benchmark with and without grammar constraint.

### Exercise 7 — Plan a PR

Same exercise as Chapter 15 Exercise 7, applied to SGLang. Pick something small and well-scoped:

- Doc clarification.
- Test addition.
- Small kernel improvement.
- Benchmark suite expansion.
- A new model architecture's plumbing.

**Deliverable:** PR plan in `NOTES.md`. Open the PR if you're ready.

### Exercise 8 — Compare/contrast writeup

Write a 1–2 page document comparing vLLM and SGLang along these axes:

- Architecture (frontend vs runtime split, language choice).
- Scheduler design.
- Memory management (paged vs radix).
- Optimization sets (which features each has).
- Performance tradeoffs.
- Contribution accessibility (issue triage, code quality, mentor presence).

This is the kind of writeup that lands in an interview presentation. Polish it.

**Deliverable:** Document committed as `vllm_vs_sglang.md`.

---

## Self-check

1. What does RadixAttention generalize beyond simple system-prompt prefix caching?
2. How does SGLang's scheduler differ from vLLM's?
3. Where does SGLang implement grammar-constrained decoding, and what's the per-step cost?
4. What's Mini-SGLang and why does it exist?
5. When is SGLang the better choice over vLLM in production?

---

## Deliverables before moving on

- SGLang serving works on 4×A100.
- Runtime tour and Mini-SGLang writeup.
- RadixAttention reproduction with comparison to vLLM.
- Profile screenshots.
- Structured-output benchmark.
- vLLM vs SGLang comparison writeup.
- PR plan (and ideally a PR).
