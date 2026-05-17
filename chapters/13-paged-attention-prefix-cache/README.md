# Chapter 13 — Paged Attention & Prefix Caching

**Hardware target:** 1 GPU enough; 4 lets you see scaling effects.
**Time budget:** 3 weeks.
**Prerequisites:** Chapter 12 — your mini-engine has continuous batching.

---

## Why this chapter exists

The contiguous KV cache from Chapter 12 wastes a *lot* of memory. A request that asks for a 4k-context response but only uses 200 tokens has paid for 4k slots. With 32 requests, that's `(4096 - 200) * 32 = ~125k slots` of wasted KV space.

PagedAttention (vLLM, Kwon et al. 2023) fixes this. It treats KV cache memory the way an OS treats RAM: virtual pages, physical pages, a page table per request. The "block size" is typically 16 or 32 tokens.

Add **prefix caching** on top: many requests share a common prefix (system prompt, chat history). If you can detect and reuse the KV cache for shared prefixes, you skip recomputing it. SGLang's RadixAttention generalizes this to a radix tree of all prefixes seen so far.

By the end you will:

1. Have implemented a paged KV cache with a block manager.
2. Have integrated it into your mini-engine and measured the memory win.
3. Have implemented prefix caching for the simplest case (system-prompt sharing).
4. Have read RadixAttention's design and described it in your own words.
5. Have implemented chunked prefill.

---

## Read first

- **vLLM paper (PagedAttention)** — Kwon et al., 2023. Section 3 is the meat.
- **SGLang RadixAttention paper / blog** — `SGLang: Efficient Execution of Structured Language Model Programs`. Read the RadixAttention section in detail.
- **vLLM source** — `vllm/core/block_manager_v2.py`, `vllm/worker/cache_engine.py`, the `csrc/attention/` CUDA kernels. Skim now, deep dive in Chapter 15.
- **SGLang source** — `python/sglang/srt/managers/scheduler.py`, `python/sglang/srt/mem_cache/memory_pool.py`.

---

## Concepts

### 1. Why paged?

Contiguous cache: 32 requests × 4096-token slots × ~256 KB/token (for 7B GQA-8) = `32 GB`, even if every request actually uses only 500 tokens.

Paged cache with 16-token blocks: each request gets only the blocks it needs. 32 requests × ~32 blocks (for 500 tokens) × ~4 KB/block = `~4 MB`. 8000× less, *and* you can grow on demand.

The cost: an extra level of indirection on every KV read. Each attention op needs to look up which physical block holds each token. This is what `PagedAttention` CUDA kernels do — they take the page table and the physical KV pool, and gather the right blocks before computing attention.

### 2. The block manager

Memory pool = `(n_total_blocks, block_size, n_kv_heads, head_dim)` allocated once. The block manager:

- `allocate(n_blocks) -> List[BlockId]`: grab `n_blocks` free blocks.
- `free(block_ids)`: return blocks to the free pool.
- Per-request "block table": ordered list of `BlockId` pointing into the pool.

When a request grows by one token and needs a new block, the manager allocates one and appends it to the request's block table.

When the GPU runs out of blocks, the scheduler must *preempt* some request — either swap its KV out to CPU/disk or *recompute* it. Production engines do recompute by default (cheaper than swap).

### 3. PagedAttention kernel

For each query (1 token in decode, N in prefill), the kernel needs to attend over the full KV history. With paged storage, it:

1. Reads the block table for the request.
2. For each block, gathers the K, V from the pool into shared memory.
3. Computes Q @ K^T, softmax, attention @ V — incremental like FlashAttention.

vLLM ships custom CUDA kernels for this. You can also build it in Triton — `attention_paged.py` in vLLM has both paths.

### 4. Prefix caching, simplest version

When multiple requests share a prefix (e.g., a system prompt prepended to every chat):

1. After computing KV for the system prompt once, save its blocks as "shared."
2. New requests that detect the same prefix re-use those blocks (read-only) without recomputing.
3. They allocate fresh blocks only for tokens past the shared prefix.

Wins are huge for chatbots — the system prompt + few-shot examples can be 1000+ tokens. Skipping that prefill is a 5–10× TTFT improvement.

### 5. RadixAttention (SGLang)

Generalizes prefix caching to *any* shared prefix. Maintains a radix tree (trie) over all prefixes seen:

- Each node represents a token sequence.
- Each node holds the KV blocks for its segment.
- A new request walks the tree by tokens, reusing as much as is matches, then branches.

The eviction policy (LRU on subtrees) prevents the tree from blowing up.

This is the single biggest difference between vLLM and SGLang as of 2024–2025: SGLang has more flexible prefix sharing. You'll read SGLang's source in Chapter 16.

### 6. Chunked prefill

Prefill cost for a long prompt is large (4k tokens × full forward = expensive). If you do it in one shot, no decode for any *other* request happens during that time → ITL of running requests spikes.

**Chunked prefill**: split the prefill into chunks (say 512 tokens), and interleave them with decode work in the same batch. Each iteration processes some decode tokens + a 512-token prefill chunk. New requests start producing tokens within ~50ms.

This trades total prefill latency (slightly worse for the long request) for ITL smoothness across all requests. Production engines enable it by default.

### 7. Disaggregated prefill/decode

The newest pattern (2024+): run prefill on one set of GPUs and decode on another, with a KV-cache handoff over the network. Lets you size each pool independently. Used in some production engines, very actively researched. We'll discuss in Chapter 14.

---

## Hands-on

### Exercise 1 — Block manager

In [`block_manager.py`](block_manager.py), implement:

```python
class BlockManager:
    def __init__(self, n_blocks, block_size): ...
    def allocate(self, n: int) -> list[int]: ...
    def free(self, block_ids: list[int]) -> None: ...
    def free_blocks_count(self) -> int: ...

class PagedRequest:
    block_table: list[int]   # ordered blocks holding this request's tokens
    seq_len: int             # tokens written so far
```

Plus a unit test that verifies free/allocate is balanced.

**Deliverable:** Tests pass. `pytest test_block_manager.py`.

### Exercise 2 — Paged KV cache (Python-only, slow)

In [`paged_kv_cache.py`](paged_kv_cache.py), implement a Python-driven paged read. On each attention step:

1. For each request, gather its blocks from the pool into a contiguous K, V tensor.
2. Compute attention normally.

This is slow (gather is expensive in Python), but lets you verify *correctness* before optimizing.

Test: same prompts and outputs as Chapter 12's contiguous cache.

**Deliverable:** Numerical equivalence test passes.

### Exercise 3 — Triton paged attention

In [`paged_attention.py`](paged_attention.py), write a Triton kernel that takes (Q, KV pool, block table, sequence length) and computes attention by directly reading from the pool.

This is genuinely advanced. Use vLLM's `paged_attention.cu` as a reference for the layout (don't copy the CUDA — read it). Start with a one-head, no-batching version and scale up.

**Deliverable:** Numerical match with Exercise 2; speed comparison.

### Exercise 4 — Integrate into mini-engine

Replace the contiguous KV cache from Chapter 12 with your paged cache. Re-run the chat-workload benchmark from Chapter 12 Ex. 6 (Poisson-arriving requests).

**Deliverable:** Memory peak comparison (contiguous vs paged at the same concurrency). Throughput comparison.

### Exercise 5 — System-prompt prefix caching

For the chat workload, prepend a 500-token system prompt to every request. Implement a simple prefix cache that:

1. Hashes the first N tokens of each request.
2. If the hash matches a previously-cached KV, reuses its blocks (read-only).
3. Allocates fresh blocks only past the shared prefix.

Measure TTFT improvement.

**Deliverable:** TTFT p50 / p95 / p99 with and without prefix caching. Should improve 3–10×.

### Exercise 6 — Chunked prefill

Modify your scheduler so prefills are chunked to at most 512 tokens per iteration. New prefill requests can join existing decode batches.

Measure ITL p99 across running requests in a workload where one request has a 4k-token prefill.

**Deliverable:** ITL p99 before and after chunked prefill.

### Exercise 7 — Read RadixAttention

Read SGLang's RadixAttention source (`mem_cache/radix_cache.py` or similar). Write a 1-page summary in your own words:

- Data structure.
- How matches happen.
- Eviction policy.
- Where it lives in the inference loop.

**Deliverable:** Writeup in `NOTES.md`.

### Exercise 8 — Compare to vLLM

For the same workload, compare your engine's metrics to vLLM (with default settings: paged KV, prefix caching off and on, chunked prefill on).

**Deliverable:** Comparison table. Make notes on the gap — it's where your engine's optimization opportunities lie.

---

## Self-check

1. What's the memory savings of paged vs contiguous for a workload with highly variable output lengths?
2. What does the block manager's "free blocks count" determine in the scheduler?
3. What's the indirection cost of paged attention, and how is it mitigated?
4. Why does prefix caching matter so much for chat workloads?
5. What's RadixAttention's data structure, and what does it generalize beyond simple prefix caching?
6. When does chunked prefill *hurt* throughput?

---

## Deliverables before moving on

- Working `block_manager.py`, `paged_kv_cache.py`.
- Triton paged attention kernel that matches reference numerically.
- Engine running with paged cache; metrics report better memory than Chapter 12.
- Prefix-cache TTFT plot.
- Chunked prefill ITL plot.
- RadixAttention writeup.
- vLLM comparison in `NOTES.md`.
