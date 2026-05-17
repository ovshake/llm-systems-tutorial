# Chapter 15 — vLLM Deep Dive & Contributions

**Hardware target:** 4× A100-80GB for the multi-GPU exercises; 1 GPU is enough for code reading.
**Time budget:** 4–6 weeks.
**Prerequisites:** Chapter 14 — your mini-engine has the major optimizations.

---

## Why this chapter exists

vLLM is the most-used open-source LLM serving engine. Knowing it well — being able to read its source, reproduce its issues, and ship small PRs — is the most direct path to "LLM systems engineer" credibility.

You've now built a mini version of vLLM from scratch (Chapters 12–14). This chapter inverts: now you read the real thing, with enough context to actually understand it. Then you reproduce a real performance issue and plan a real contribution.

By the end you will:

1. Know the vLLM directory layout and the major subsystems.
2. Have run vLLM with the key features enabled (paged attention, prefix caching, speculative decoding, TP, quantization).
3. Have profiled vLLM and explained where its time goes.
4. Have reproduced an open performance issue or a documented benchmark.
5. Have a plan for a small PR you could realistically land.

---

## Read first

- **vLLM docs** — read the entire user-facing docs first. Quickstart, distributed inference, PagedAttention, prefix caching, speculative decoding, structured outputs, performance tuning.
- **vLLM paper** (Kwon et al., 2023) — re-read with your Chapter 13 context.
- **vLLM contributing guide** — `vllm/CONTRIBUTING.md`. Understand the PR process, CI, and the "good first issue" label.
- **Recent vLLM design docs / RFCs** — search for "RFC" in the issues and PRs.
- **vLLM source map** — see below.

---

## The vLLM source map

Approximate, accurate as of recent versions. Layout may shift; orient yourself with the actual code.

```
vllm/
├── engine/                          # User-facing entrypoints
│   ├── llm_engine.py                # the LLMEngine class — orchestrates everything
│   ├── async_llm_engine.py          # AsyncLLMEngine — for OpenAI-compatible API
│   └── arg_utils.py                 # config parsing
├── core/
│   ├── scheduler.py                 # token-level scheduling, admit policy
│   ├── block_manager.py             # paged KV blocks
│   ├── block.py                     # block, block table
│   └── policy.py                    # FCFS / priority policies
├── worker/                          # per-GPU worker process
│   ├── worker.py                    # one Worker per GPU
│   ├── cache_engine.py              # KV cache allocator on the worker side
│   └── model_runner.py              # forward pass orchestration
├── model_executor/
│   ├── models/                      # per-model definitions (llama, mistral, qwen, ...)
│   ├── layers/                      # attention backends, linear, quantization
│   └── parallel_utils/              # TP, PP utilities (DTensor-based)
├── attention/
│   ├── backends/                    # FlashAttention 2/3, paged attention, ...
│   └── selector.py
├── distributed/                     # NCCL setup, process groups
├── csrc/                            # custom CUDA kernels (paged attention, quantization, ...)
└── entrypoints/
    └── openai/                      # OpenAI-compatible API server
```

The hot path for a single decode iteration:

```
AsyncLLMEngine.add_request
    ↓
LLMEngine.step()
    ↓
Scheduler.schedule()        # decides what runs this iteration
    ↓
Worker.execute_model()
    ↓
ModelRunner.execute_model() # builds the input tensors
    ↓
Model.forward()             # the LLaMA / etc. model
    ↓
Attention backend (PagedAttention)
    ↓
Sampler.sample()            # logits → token IDs
    ↓
returns SamplerOutput, which propagates back to the scheduler.
```

You should be able to walk this trace in the source, end-to-end.

---

## Concepts (deep dive)

### 1. The scheduler

`vllm/core/scheduler.py`. Each step:

1. Move requests that finished last step out of `running` into `swapped` or `completed`.
2. For each request in `running`, queue 1 decode token. If it needs another KV block, allocate.
3. Pull from `waiting` (admission): for each, allocate KV for its prompt prefill. If you can't, the request stays in `waiting`. If memory is tight, *preempt* a running request (recompute or swap).
4. The scheduled batch is one `ScheduledSequenceGroup` per request, with the token counts each contributes (decode = 1; new request = prefill_len; chunked = chunk_size).

The scheduler is the single most consequential file. Read it 3 times.

### 2. The block manager

`vllm/core/block_manager.py`. Wraps the block allocator and the block tables. Implements:

- `can_allocate(seq)`: does the request fit?
- `allocate(seq)`: assign blocks.
- `append_slot(seq, position)`: extend by one more token.
- `swap_in` / `swap_out`: for preemption.
- Prefix caching machinery (in newer versions; see `block_manager_v2`).

### 3. The attention backend

`vllm/attention/backends/`. Each backend exposes a uniform interface: given Q, KV pool, block tables, sequence lengths, scale, return attention output. Backends:

- **FlashAttention2** — the workhorse on Ampere.
- **FlashAttention3** — for Hopper.
- **XFormers** — fall-back.
- **PagedAttentionV1/V2** — custom CUDA kernels in `csrc/attention/`.

You're not expected to write a new backend; reading one is enough.

### 4. The model runner

`vllm/worker/model_runner.py`. Handles:

- Building the input tensor (variable-length, with cumulative-seqlen arrays).
- Position IDs.
- Slot mapping (which KV slot does each new token go to?).
- Calling the model's `forward` with the right metadata.
- Sampling.

This is where many of the "make it fast" details live.

### 5. Quantization

`vllm/model_executor/layers/quantization/`. Each method (GPTQ, AWQ, FP8, INT8) has its own subclass of `QuantizationConfig`. The `linear.py` wraps `nn.Linear` to call the right quantized kernel.

### 6. TP and PP

vLLM's TP is mostly transparent — the model definitions in `model_executor/models/` use `ColumnParallelLinear` and friends. PP is more recent and uses `pipelining` from PyTorch.

---

## Hands-on

### Exercise 1 — Install and serve

Install vLLM with CUDA 12.x. Serve `Llama-3-8B` (or `Llama-3.1-8B-Instruct` if accessible) on TP=4:

```bash
uv pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4
```

Hit it from a client with `openai` python lib. Verify it works.

**Deliverable:** A captured response in `NOTES.md`.

### Exercise 2 — Tour the source

Spend a full day reading vLLM. Goal: be able to draw the engine-step diagram from memory, and find any specific subsystem within 30 seconds.

In `NOTES.md` write:
- 3 architectural choices that surprised you.
- 1 piece of code you thought was elegant.
- 1 piece you thought was confusing.

### Exercise 3 — Benchmark with `benchmark_serving.py`

vLLM ships a benchmarking script. Run it for two workloads:

- **ShareGPT** workload (mixed conversation lengths).
- **Long-context** workload (e.g., LongBench, or synthetic 16k prompts).

Compare TTFT / ITL / throughput / p99 to your numbers from Chapter 14.

**Deliverable:** Comparison table.

### Exercise 4 — Reproduce an open performance issue

Go to vLLM's issues. Filter by `performance`. Pick one you can plausibly reproduce on 4×A100-80GB. Run the scenario, confirm the regression or bottleneck.

**Deliverable:** Issue link + your reproduction steps + numbers, in `NOTES.md`.

### Exercise 5 — Profile vLLM under load

Run vLLM under load. Use `py-spy` for CPU work and `nsys` for GPU work. Find:

- Where the scheduler spends its time.
- Where the model runner spends time.
- Whether the GPU has idle gaps (CPU bottleneck).

**Deliverable:** Annotated profile screenshots. A 1-paragraph reading of where time goes.

### Exercise 6 — Speculative decoding in vLLM

Enable speculative decoding with a draft model. Measure throughput vs base.

Read the speculative-decoding implementation in `vllm/spec_decode/`. Note the rejection-sampling code.

**Deliverable:** Throughput comparison + a 1-paragraph summary of how vLLM's spec decoding differs from yours.

### Exercise 7 — Plan a PR

Pick a small, well-scoped contribution. Examples:

- Fix a documentation gap you found during your reading.
- Add a benchmark for a niche case.
- Add an error message clarification.
- Add a test for a feature that's currently untested.
- Fix a flaky test.
- Add support for a small model variant that's missing.

Write a 1-paragraph PR plan: what you'll change, why, what tests you'll add, how you'll benchmark.

**Deliverable:** PR plan in `NOTES.md`. If you have time, actually open the PR.

### Exercise 8 (stretch) — Add a feature flag for an experimental optimization

Pick a documented but disabled-by-default optimization (e.g., a new attention backend, a new scheduler policy). Turn it on, run the benchmark suite, write up the results. This is the kind of thing maintainers love because it advances the design discussion.

**Deliverable:** Comparison table + 1-page writeup. Optional PR.

---

## Self-check

1. Walk through what happens when a single new request hits `AsyncLLMEngine.add_request`.
2. How does the scheduler decide between admitting a new request and continuing existing ones?
3. What does the model runner do that the model itself doesn't?
4. Where is the paged-attention CUDA kernel, and what does it take as input?
5. What's the difference between `block_manager_v1` and `block_manager_v2`?
6. When would you write a new attention backend vs use FlashAttention?

---

## Deliverables before moving on

- vLLM serving works on 4×A100.
- Source-tour notes in `NOTES.md`.
- Two benchmark workloads run and compared to Chapter 14.
- Reproduced an open issue.
- Profile screenshots and reading.
- Speculative decoding analysis.
- PR plan (and ideally an actual PR).
