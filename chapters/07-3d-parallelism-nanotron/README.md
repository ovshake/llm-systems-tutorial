# Chapter 07 — 3D Parallelism with Nanotron / Megatron-LM

**Hardware target:** 4× A100-80GB.
**Time budget:** 3–4 weeks.
**Prerequisites:** Chapters 03, 04, 05, 06.

---

## Why this chapter exists

You've now built every distributed-training scheme by hand: DDP, FSDP, TP, PP. In real pretraining you combine them. The standard topology is *3D parallelism*: DP × TP × PP, with their product equal to the total GPU count.

Choosing the right (TP, PP, DP) for a given model and cluster is the most consequential decision a pretraining engineer makes. This chapter teaches you to do that *with real frameworks* — Nanotron and Megatron-LM — rather than from scratch.

Why two frameworks? Nanotron is a clean, modern, PyTorch-native reference that the Ultra-Scale Playbook uses for production-style code. Megatron-LM is the elder, the most-cited reference, and what NVIDIA / many labs run in production. Reading both shows you the genuinely-important patterns (and which idioms are accidental).

By the end you will:

1. Have trained a small dense LM with Nanotron in a (TP, PP, DP) config and tuned it.
2. Have read enough of the Megatron-LM `megatron/core` source to recognize the equivalent components.
3. Be able to take a model spec + GPU count and propose a 3D config with memory + comm justifications.
4. Know what context parallelism and expert parallelism *are*, even if you don't implement them here (Chapter 08 covers expert parallel).

---

## Read first

- **Megatron-LM "Efficient large-scale" paper** (Narayanan et al., 2021) — the canonical 3D parallelism reference.
- **Nanotron README + `examples/`** — `huggingface/nanotron`. Especially the distributed config and training launch scripts.
- **Megatron-LM source** — `megatron/core/parallel_state.py`, `megatron/core/transformer/transformer_block.py`, `megatron/core/distributed/`.
- **Ultra-Scale Playbook** — the 3D parallelism chapter is where this all comes together.

---

## Concepts

### 1. Process groups: the data structure

Each GPU belongs to several process groups simultaneously:

- A *TP group*: GPUs that share a slice of a layer (size = TP degree).
- A *PP group*: GPUs that share a slice of layers (size = PP degree).
- A *DP group*: GPUs that hold replicas of the same TP×PP slice (size = DP degree).

For a 4×A100 node with TP=2, PP=2, DP=1:

```
rank 0: TP group {0,1}, PP group {0,2}, DP group {0}
rank 1: TP group {0,1}, PP group {1,3}, DP group {1}
rank 2: TP group {2,3}, PP group {0,2}, DP group {2}
rank 3: TP group {2,3}, PP group {1,3}, DP group {3}
```

Nanotron's `ParallelContext` and Megatron-LM's `parallel_state` both create these groups at startup. Every collective in the training loop is scoped to one group.

### 2. Choosing (TP, PP, DP)

Rules of thumb:

- **TP** — set to the smallest power of 2 that makes the model + activations fit on one *node*. Within a node: NVLink is fast, TP is cheap. Across nodes: TP is expensive, avoid.
- **PP** — set to fill out the remaining inter-node dimension after TP. PP boundaries are tolerant of slower links.
- **DP** — set to the leftover. DP is the most efficient form of scaling (each rank has full ownership of compute for its mini-batch).

So for a 64-GPU cluster of 8 DGX-A100 (8 GPUs each), one common layout: TP=8, PP=8, DP=1. For a 7B model on 4 GPUs: TP=4, PP=1, DP=1 (you're not even doing DP — you're just doing TP for memory). For a 13B model on 8 GPUs: TP=2, PP=2, DP=2. Etc.

The Ultra-Scale Playbook has a flowchart for this; internalize it.

### 3. Sequence parallelism + context parallelism

- **Sequence parallelism (SP)** — already covered in Chapter 05. Shards activations along the sequence dim *within* the TP group.
- **Context parallelism (CP)** — for *very* long contexts (≥ 32k), shard the sequence across a separate process group. Each GPU sees a chunk of the sequence and exchanges KV across the CP group for attention. This is *separate from* SP and adds one more dimension to the parallelism cube.

We won't implement CP here, but we'll discuss its memory savings.

### 4. Expert parallelism (preview for Chapter 08)

For MoE models, the experts are split across an *expert parallel* group. Token routing requires all-to-all between the EP group. This makes the parallelism cube 4D: DP × TP × PP × EP. Chapter 08.

### 5. Mixed precision and gradient buffers in 3D

A subtle thing: under TP, gradient buffers exist on each TP rank for that rank's *slice* of the parameters. Under DP, those gradient buffers must be reduced *across DP only* (not TP). Under FSDP within DP, reduce-scatter still works because the FSDP shard is orthogonal to the TP shard. Frameworks handle all this for you — but it's worth understanding so you don't accidentally re-reduce.

### 6. Activation checkpointing in 3D

Standard: checkpoint every transformer block. With TP+SP, you're checkpointing the sequence-sharded form. With PP, you're checkpointing *per pipeline stage's set of blocks*. With DP, no effect.

Selective activation checkpointing — checkpoint only attention but not MLP, or only the SwiGLU intermediate — saves more memory than full checkpointing at much lower compute cost. Megatron's `--recompute-granularity selective` is the canonical knob.

### 7. The communication budget

For a 3D run, the comm cost is roughly:

```
comm_per_step ≈ TP_comm_per_block * n_layers / PP_stages_per_rank +
                PP_send_recv * (PP - 1) * microbatches +
                DP_all_reduce_grad_step
```

When you propose a config, sum these up and compare to the available bandwidth. If TP comm > NVLink time × number of blocks, you'll be TP-bound. If DP all-reduce dominates the step, you're DP-bound (rare unless your model is small).

---

## Hands-on

### Exercise 1 — Install Nanotron, run a tutorial

Clone Nanotron next to this repo. Follow its quickstart. Train a tiny config on 1 GPU first; confirm the loss falls.

Then run the multi-GPU example with the smallest model that fits — Nanotron exposes `tp`, `pp`, `dp` in its YAML config. Try TP=2, PP=1, DP=2 on your 4-GPU box.

**Deliverable:** Loss curve, tokens/sec, peak memory in `NOTES.md`.

### Exercise 2 — Read the Nanotron `ParallelContext`

In Nanotron, find the file that defines the parallel context (process groups) — usually `nanotron/parallel/__init__.py` or similar. Read it carefully. Write a short summary in `NOTES.md` answering:

- How are TP, PP, DP groups constructed?
- How is each rank's role in each group determined?
- Where in the training loop are these groups passed to communication ops?

You should be able to draw the rank-to-group mapping for a (TP=2, PP=2, DP=2) on 8 GPUs from memory after this.

### Exercise 3 — Tune the (TP, PP, DP) for a `medium` model

Using Nanotron, try four configs at world=4:

1. TP=1, PP=1, DP=4 (= DDP).
2. TP=4, PP=1, DP=1.
3. TP=2, PP=2, DP=1.
4. TP=1, PP=2, DP=2.

For each: tokens/sec, peak memory per rank, MFU. Hold global batch constant.

**Deliverable:** 4-row table. A paragraph identifying the best config and *why*.

### Exercise 4 — Try Megatron-LM

Clone Megatron-LM. Run the GPT pretraining script on a small config. Megatron-LM is heavier to set up than Nanotron — be patient. The setup itself is part of the lesson.

Read `megatron/core/parallel_state.py` and find:
- Where TP groups are created.
- The PP scheduling code (look for `forward_backward_pipelining_with_interleaving`).
- The DP gradient reduction.

**Deliverable:** A short writeup in `NOTES.md`: 3 things Megatron does that Nanotron does differently, and your read on which design is cleaner.

### Exercise 5 — Memory and comm prediction

Write [`predict_3d.py`](predict_3d.py): given (model_cfg, TP, PP, DP, B_micro, S), predict per-rank memory and total comm bytes per step. Validate against your Nanotron run from Exercise 3.

**Deliverable:** Validation table — predicted vs measured (memory and step time) for the four configs.

### Exercise 6 — Selective activation checkpointing

In Megatron, run with `--recompute-granularity selective` and `--recompute-granularity full` on the same config. Compare memory and tokens/sec.

**Deliverable:** Two-row table. Selective should give you most of the memory savings at a fraction of the compute cost.

### Exercise 7 — Plan a 100B run (paper exercise)

Suppose you have 256 H100-80GB across 32 nodes (8 GPUs/node, NVLink intra-node, InfiniBand inter-node at 400 Gbps). Propose a 3D config for:

- Llama-2-70B (dense).
- A hypothetical 175B dense model.

Include:
- (TP, PP, DP) values.
- Memory accounting per rank.
- Where you'd activation-checkpoint.
- Expected MFU range.
- The first thing you'd profile if MFU < 30%.

This is a classic interview question. Write it up properly.

**Deliverable:** A 1–2 page writeup in `NOTES.md`.

---

## Self-check

1. For a 4-GPU node, when does TP=4 beat TP=2, PP=2?
2. Why do PP boundaries tolerate slower links than TP boundaries?
3. What is the rank-to-group mapping for (TP=2, PP=2, DP=2) on 8 GPUs?
4. Why doesn't FSDP-within-DP conflict with TP?
5. Where does the all-reduce for DDP-style gradient sync happen in a 3D config — across all ranks or just the DP group?
6. Selective vs full activation checkpointing: what's selective skipping, and why is it usually a better default?

---

## Deliverables before moving on

- Nanotron quickstart run done.
- 4-config TP/PP/DP table.
- One Megatron-LM training run completed.
- `predict_3d.py` validates within 15%.
- "Plan a 100B run" writeup committed.
