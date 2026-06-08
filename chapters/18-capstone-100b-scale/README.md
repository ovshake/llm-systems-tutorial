# Chapter 18 — Capstone: 100B-Scale Training Plans + Portfolio

**Hardware target:** None — these are mostly paper exercises.
**Time budget:** Ongoing; finish during or shortly after Chapter 17.
**Prerequisites:** Everything previous.

---

## Why this chapter exists

You're not going to train a 100B model on your 8-GPU box. But the *plans* you'd make if you had the GPUs are exactly what you'd be asked to do in a serious interview, in a design doc, or in the first month at a pretraining shop. The capstone is where you write those plans down — properly, with math you can defend, citing the lessons from earlier chapters.

This is also where the portfolio takes its final shape. Until now you have notes, scripts, and benchmark results scattered across chapter folders. The capstone collects them into a coherent narrative.

By the end you will have:

1. Two written training plans (dense 100B, MoE) with memory, comm, MFU, and hardware sizing.
2. A serving plan for those models.
3. A polished portfolio you'd hand a hiring manager.

---

## Read first

Re-read these with end-to-end context, in order:

- **Ultra-Scale Playbook** — all of it.
- **Megatron-LM scaling paper** (Narayanan et al., 2021) — the canonical 3D-parallelism reference.
- **Llama-3 technical report** — production setup for a 405B run.
- **DeepSeek-V3 technical report** — production setup for a 671B-parameter MoE.
- **GShard / Switch Transformer** — origin of large MoE.
- **vLLM paper, FlashAttention 3 paper, MegaBlocks paper** — the implementation references.

---

## The capstone deliverables

You'll write three documents and assemble one portfolio.

### Deliverable 1 — 100B dense training plan

A 5-page document covering, for a hypothetical 100B parameter dense decoder transformer trained from scratch on `~15T tokens` to a comparable quality to Llama-3-70B:

1. **Model spec.** Layer count, hidden, heads, KV heads, FFN size, vocab size, context length. Justify each.
2. **Cluster spec.** Pick your hardware (e.g., 1024×H100-80GB across 128 nodes, NVLink intra-node, NDR-400 InfiniBand inter-node, or whatever you assume). Justify.
3. **Parallelism strategy.** (TP, PP, DP, SP, CP) — with the math for each:
   - Memory per rank (weights, grads, opt, activations, overhead).
   - Comm per step (TP all-reduces × layers, PP send/recv × microbatches, DP all-reduce, etc.).
   - Estimated MFU (use Chapter 02's tools).
4. **Optimization choices.** AdamW vs Lion, beta values, weight decay, LR schedule, warmup, gradient clipping. Justify in 2 sentences each.
5. **Activation policy.** Full vs selective vs none. Per layer, per stage.
6. **Checkpointing strategy.** Frequency, format (full vs sharded), where it goes, recovery time math.
7. **Failure modes.** What you expect to break: NCCL deadlocks, OOMs on the tail microbatch, optimizer NaN, etc. For each, your detection plan.
8. **Schedule.** Tokens/sec target, total training days, with caveats.

This is a *document you would put in front of a hiring manager*. Polish accordingly.

### Deliverable 2 — 100B MoE training plan

A 4-page document for a hypothetical 100B-active / 600B-total parameter MoE model (e.g., 8 experts top-2, or 32 experts top-4). Same structure as Deliverable 1, with these extra sections:

1. **Expert parallelism choice.** EP degree, intra-node vs inter-node placement.
2. **All-to-all budget.** Estimated comm per block, total per step.
3. **Imbalance handling.** Capacity factor or dropless (MegaBlocks)? Auxiliary loss weight?
4. **Composition with TP / PP / DP.** Where do the experts live in the parallelism cube?

### Deliverable 3 — Serving plan for either model

A 3-page document covering:

1. **Target workload.** Chat (latency-sensitive) vs batch-job (throughput-only) vs long-context (16k+).
2. **Hardware.** Per-instance GPU count and memory.
3. **Engine choice.** vLLM, SGLang, Triton-server, custom. Justify.
4. **Parallelism for inference.** TP and PP degrees (likely smaller than training).
5. **Quantization.** Weight bits, activation bits, KV cache bits.
6. **Capacity planning.** Tokens/sec per replica, replicas needed, autoscaling.
7. **Tail latency strategy.** Chunked prefill, speculative decoding, prefix caching policies.
8. **What you'd profile day 1.** The first three Nsight Systems / Nsight Compute traces you'd capture.

### The portfolio repo

Assemble a separate repo (`yourname/llm-systems-portfolio` or similar) with:

- The three plans from above.
- Distributed training scaling report (Chapters 03, 04, 07).
- MoE bottleneck report (Chapter 08).
- Triton kernel benchmarks + roofline (Chapter 10).
- Mini inference engine code + benchmarks vs vLLM/SGLang (Chapters 12–14).
- OSS contribution log (Chapters 15–16 PR plans, plus any actually-merged PRs).
- Post-training writeup (Chapter 17).

Each as a top-level Markdown file with embedded plots / numbers. Curate aggressively — three excellent reports beat ten mediocre ones.

---

## Exercises

### Exercise 1 — Pre-mortem

Before you write the plans, do a "pre-mortem": imagine each plan has *failed*. List 10 specific failure modes you would not want to have to debug. Pin those to the wall, and make sure the plan addresses each.

**Deliverable:** Pre-mortem list per plan.

### Exercise 2 — Write the dense plan

Write Deliverable 1. Get all the math right. Cite the chapters where each technique was learned.

**Deliverable:** `plans/dense_100b.md`.

### Exercise 3 — Write the MoE plan

Write Deliverable 2.

**Deliverable:** `plans/moe_100b.md`.

### Exercise 4 — Write the serving plan

Write Deliverable 3.

**Deliverable:** `plans/serving.md`.

### Exercise 5 — Assemble portfolio

Move (or symlink) the chapter reports into a portfolio repo. Read each as if you were a hiring manager. Cut anything that's weak.

**Deliverable:** `portfolio/` repo, public on GitHub.

### Exercise 6 — Mock interview

Find a friend in the field. Walk them through one plan and one project. Note the questions you couldn't answer well, then fix the gap.

**Deliverable:** Updated plan / project addressing the gaps.

---

## Self-check

1. Why did you pick the (TP, PP, DP) you did?
2. What's the dominant comm cost per step? How did you estimate it?
3. What's your MFU target? Why is it not 50%+?
4. What's the single likeliest failure mode of your dense 100B training plan?
5. For the MoE plan, why is all-to-all not your dominant cost? (Or is it?)
6. What's your serving plan's biggest assumption?

---

## When you're "done"

You're done with this course when:

- All three plans exist and you could defend any of them in a 30-minute conversation.
- The portfolio repo is public.
- You've opened (and ideally merged) at least one vLLM or SGLang PR.
- You've taken at least one mock interview from a relevant person and acted on the feedback.

This isn't a course you finish in 12 months and forget. It's a course that gets you to the *starting* line as an LLM systems engineer. The actual job starts the day you stop.

Good luck.
