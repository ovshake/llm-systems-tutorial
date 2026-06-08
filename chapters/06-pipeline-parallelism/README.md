# Chapter 06 — Pipeline Parallelism

**Hardware target:** 2–8 GPUs (you'll feel the bubbles best at 8 — more stages, bigger bubble).
**Time budget:** 2–3 weeks.
**Prerequisites:** Chapter 05 — TP works.

---

## Why this chapter exists

Pipeline parallelism (PP) splits the model *along its depth*: GPU 0 holds layers 0–N/4, GPU 1 holds N/4–N/2, and so on. A batch flows through the pipeline: GPU 0 forwards layer 0, hands activations to GPU 1, which forwards layer N/4, and so on. Backward flows in reverse.

The reason PP exists: TP scales to the size of one node, FSDP comm grows with model size, and at very large scale you eventually want to split *layers* across machines so each machine holds only a few. PP is the scheme for that.

The reason PP is *annoying*: naïve PP keeps almost every GPU idle most of the time (the "pipeline bubble"). The whole field of PP is about clever scheduling that shrinks the bubble.

By the end you will:

1. Have implemented a hand-rolled PP loop with GPipe and 1F1B schedules.
2. Understand the bubble and the analytic relationship between bubble fraction, microbatch count, and depth.
3. Have used PyTorch's `torch.distributed.pipelining` (the modern API replacing `torch.distributed.pipeline.sync.Pipe`) on a real model.
4. Know when *not* to use PP (intra-node, small model — usually wrong).

---

## Read first

- **GPipe paper** (Huang et al., 2019) — microbatching.
- **PipeDream paper** (Narayanan et al., 2019) — non-synchronous 1F1B (you can skip "PipeDream-2BW" first time).
- **Megatron-LM PP paper** (Narayanan et al., 2021, "Efficient large-scale language model training") — 1F1B, interleaved schedule.
- **PyTorch `torch.distributed.pipelining`** docs (PiPPy / pipelining v2).
- **Ultra-Scale Playbook PP chapter.**

---

## Concepts

### 1. The naïve pipeline (and its bubble)

```
GPU0:  F0 F1 F2 F3                            B3 B2 B1 B0
GPU1:     F0 F1 F2 F3                      B3 B2 B1 B0
GPU2:        F0 F1 F2 F3                B3 B2 B1 B0
GPU3:           F0 F1 F2 F3          B3 B2 B1 B0
       ^^^^^^^^^^^^                ^^^^^^^^^^^^^^^
        forward fill                backward drain
```

With M microbatches and P stages, the bubble is `(P-1) / (M + P - 1)`. So to keep the bubble below 10% with P=4, you need M ≥ 36.

**Microbatch count is the lever.** More microbatches → smaller bubble, but each microbatch is smaller → worse arithmetic intensity. Tune.

### 2. 1F1B (one-forward-one-backward)

GPipe runs all forwards then all backwards. Activations from every microbatch must be stashed across the whole forward sweep → activation memory blows up.

1F1B: after the pipeline fills up, alternate. Once you've done a microbatch's backward, you can free its activations.

```
GPU0:  F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3
GPU1:     F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3
GPU2:        ...
```

Same bubble, much less activation memory (you only hold ~P microbatches' activations, not M).

### 3. Interleaved 1F1B

Megatron's contribution: instead of giving each rank one *contiguous* chunk of layers, give it *interleaved* chunks. For P=4 stages and L=24 layers, instead of {0–5, 6–11, 12–17, 18–23}, give rank 0 layers {0,1,12,13}, rank 1 {2,3,14,15}, etc. With V "virtual stages" per rank, the bubble shrinks by 1/V.

The trade: V× more activation-stashing per microbatch transition (each rank now sees more microbatches in flight).

### 4. Pipeline-flush vs no-flush

If you have N optimizer steps and each one is a "flush" (drain pipeline, step, refill), the bubble cost is paid every step. PipeDream's no-flush schedule avoids this by *delaying* the optimizer update to use stale weights for the forward pass while gradients accumulate.

In practice: GPipe and 1F1B are the standard for synchronous LM training; PipeDream's no-flush ideas live on in research but aren't the default.

### 5. PP cuts across blocks; you choose the cut

Two questions:
- *Which* layers go on which rank? Equal layer-count partition is easy but unequal layer cost wastes time. Some layers (embedding, lm_head, first/last) are imbalanced — usually treat them specially.
- *How* are activations passed? `send`/`recv` of a tensor between adjacent ranks. The tensor includes everything backward needs: the activation itself plus whatever positional info / attention masks the next stage requires.

### 6. Inter-stage comm

For each microbatch, every stage boundary requires one `send` (activations) on forward and one `recv` (gradient w.r.t. activations) on backward. Comm volume per microbatch boundary = `B_micro * S * H * 2 bytes` (BF16). For a 7B model with B_micro=4, S=2048, H=4096: ~128 MB per boundary. Times microbatches, times boundaries.

This is *much smaller* than TP comm (which is per block) but *more frequent at boundaries*. Latency-sensitive — PP wants any link, but it works on PCIe and across nodes too.

### 7. Composing PP with DP and TP (3D parallelism)

Standard layout for a 4×4×4 = 64-GPU run: TP=4 (intra-node), PP=4 (across nodes), DP=4 (replicas of the TP×PP slice). We don't have the GPUs for this, but the math is in Chapter 07.

---

## Hands-on

### Exercise 1 — Naïve GPipe on 8 GPUs

In [`gpipe.py`](gpipe.py) (skeleton), implement a synchronous GPipe scheduler:

1. Split the model into `P=world_size` equal blocks-of-blocks.
2. Use `dist.send` / `dist.recv` between adjacent ranks.
3. For M microbatches: forward all, store activations, then backward all.
4. After all microbatches, step the optimizer.

Run on the `medium` preset (~350M, 24 layers) with P=4. Print:

- Forward fill bubble (idle time on rank 0 between F_M and B_0).
- Step time per global batch.
- Estimated bubble fraction.

**Deliverable:** Plot of bubble fraction vs M ∈ {4, 8, 16, 32, 64}.

### Exercise 2 — 1F1B

In [`one_f_one_b.py`](one_f_one_b.py), implement 1F1B. Same model, same M. Compare to GPipe:

- Bubble fraction should be identical (same critical path).
- Peak activation memory should be much smaller.

**Deliverable:** Two-row table: (GPipe peak mem, 1F1B peak mem). Bubble fraction plot from Exercise 1, now with 1F1B overlaid.

### Exercise 3 — PyTorch pipelining

Replace your hand-rolled scheduler with `torch.distributed.pipelining.ScheduleGPipe` or `Schedule1F1B`. The API takes a "stage module" (your sub-model on each rank) and a microbatch generator.

Run on the same model. Confirm:

- Equivalent loss curve to your hand-rolled version.
- Equivalent or better step time (PyTorch's implementation handles `send`/`recv` overlap).

**Deliverable:** Side-by-side step time table. Note any differences in setup complexity.

### Exercise 4 — Imbalanced partition study

For the 24-layer `medium` model, P=4. Try three partitions:

- Equal: {6, 6, 6, 6} layers per stage.
- Front-heavy: {8, 6, 6, 4} (because embedding lives on stage 0).
- Back-heavy: {4, 6, 6, 8} (because lm_head + final norm live on stage P-1).

Profile each. The one matching your actual compute imbalance should be ~5–10% faster.

**Deliverable:** Three-row table.

### Exercise 5 — Compose with DP

Run `(PP=2, DP=4)` on 8 GPUs. Then `(PP=4, DP=2)`. Then `(PP=8, DP=1)`. Then `(PP=1, DP=8)`. For each, hold global batch constant and compare tokens/sec.

Spoiler: on a single node with NVLink, DP almost always beats PP for a small model. PP only wins when DP/FSDP runs out of memory.

**Deliverable:** Four-row table; conclusion in 1 paragraph.

### Exercise 6 — Interleaved schedule (stretch)

Implement (or use `torch.distributed.pipelining`'s) interleaved schedule with V=2 virtual stages per rank. Compare to plain 1F1B at the same M.

The bubble should be halved (roughly). The activation memory roughly doubles.

**Deliverable:** Table comparing 1F1B vs interleaved at the same M.

---

## Self-check

1. What's the bubble fraction formula and what dominates it?
2. Why does 1F1B reduce peak activation memory without changing the bubble?
3. What's the trade in interleaved 1F1B?
4. When would you choose PP=4, DP=1 over PP=1, DP=4 on a single node?
5. Why are embedding and lm_head layers usually handled specially in partitioning?
6. What happens if your microbatch size doesn't divide evenly into your global batch?

---

## Deliverables before moving on

- `gpipe.py` and `one_f_one_b.py` hand-rolled and working.
- One run using `torch.distributed.pipelining`.
- Bubble-fraction plot, peak-memory comparison, imbalanced-partition table, DP×PP composition table.
- `NOTES.md` with a 1-paragraph statement of when *you* would reach for PP in a real project.
