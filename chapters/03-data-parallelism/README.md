# Chapter 03 — Data Parallelism (DDP)

**Hardware target:** 2–8 GPUs. Full benefit at 8.
**Time budget:** 2 weeks.
**Prerequisites:** Chapter 02 — you can predict and measure memory and MFU.

---

## Why this chapter exists

Data parallelism is the simplest distributed-training scheme: every GPU has a full copy of the model, sees a different mini-batch, then they reconcile gradients via an all-reduce. It's also the one you will use in production *first* — DDP is the cheapest scaling, the fewest moving parts, and the workhorse under everything else.

By the end of this chapter you will:

1. Have implemented a "DDP-from-scratch" trainer (hand-rolled all-reduce on gradients) before using PyTorch's `DistributedDataParallel`.
2. Understand bucketization and why DDP is overlap-friendly.
3. Be able to measure scaling efficiency on your 8-GPU node and explain *why* it's not 100%.
4. Know the difference between micro-batch, global batch, and gradient accumulation, and be able to debug a "loss is different on different world sizes" bug.

---

## Read first

- **PyTorch DDP design doc** (`docs.pytorch.org` → distributed → DDP). Most important sections: "Bucketing", "Gradient reduction", "When the backward hook fires".
- **PyTorch `torch.distributed` reference** — get familiar with `init_process_group`, `all_reduce`, `barrier`.
- **NCCL ring vs tree algorithms** — `NCCL_ALGO=Ring` vs `Tree`. The all-reduce cost difference matters above 8 GPUs.
- **Ultra-Scale Playbook** — chapter on data parallelism.

---

## Concepts

### 1. What "data parallel" actually means

Every rank holds:
- The **same** model parameters.
- The **same** optimizer state.
- A **different** mini-batch from the same dataset.

Each step:
1. All ranks run forward and backward on their local mini-batch, producing local gradients.
2. Ranks `all_reduce` the gradients with `op=SUM`, then divide by world size — you now have the mean gradient across all ranks.
3. All ranks run the *same* optimizer step. Because parameters started equal and gradients are now equal, parameters stay equal.

If you replace "world size" with 1 and re-derive, you get standard SGD. DDP scales the *throughput*, not the optimization.

### 2. Effective batch size

```
global_batch = micro_batch * world_size * gradient_accumulation
```

This is the actual batch your optimizer sees. Three knobs, one number:

- **Bigger world_size:** faster wall-clock for the same total tokens.
- **Bigger gradient_accumulation:** lets you simulate large batches on small memory.
- **Bigger micro_batch:** better MFU (amortize kernel launches), worse memory.

A common pitfall: keep `global_batch` constant when changing `world_size` to do an apples-to-apples scaling study. Otherwise you can't tell if the loss curve changed because of scaling or because of a bigger effective batch.

### 3. The math of all-reduce

For `N` ranks and a tensor of `B` bytes:

- **Ring all-reduce:** `2B(N-1)/N` data moved per link. Optimal bandwidth for large B.
- **Tree all-reduce:** lower latency for small B, higher bandwidth for large B; uses log(N) steps. NCCL picks dynamically based on message size and topology.

For DDP, the all-reduce is the *only* communication on the forward/backward path. Compare: TP all-reduces *inside* each layer. PP exchanges activations between stages. FSDP all-gathers parameters and reduce-scatters gradients. The DDP all-reduce is *once per step*, and it's already overlapped with backward.

### 4. Bucketization

PyTorch DDP does *not* do one all-reduce at the end of backward. It buckets gradients into fixed-size groups (default 25 MB), and as soon as all gradients in a bucket are ready (i.e., the backward kernel for those parameters has run), it kicks off the all-reduce on that bucket. While that all-reduce runs, the rest of backward continues. This *overlaps* communication and computation, which is the magic.

Two knobs:
- `bucket_cap_mb` — smaller buckets = more frequent comm = less to overlap with the *last* bucket; larger = bigger leftover that doesn't get overlapped. Default 25 MB is usually fine. Tune if your model has unusual gradient shapes.
- `static_graph=True` — tells DDP your model's computational graph doesn't change across iterations, enabling more aggressive optimizations (e.g., the last bucket is fused).

### 5. Why scaling isn't 100%

You'll *not* see 8× tokens/sec on 8 GPUs. Where the loss goes:

| Cause | Symptom | Fix |
|---|---|---|
| All-reduce cost | dt(N=8) > dt(N=1) | bigger model (more compute per all-reduce byte), bigger micro-batch |
| Data loading bottleneck | low GPU utilization, dataloader pegs CPU | preload, prefetch, persistent workers, IterableDataset |
| Stragglers | one rank slower than others; barrier waits | check thermal throttling, weird PCIe links, system noise |
| Synchronizing on `loss.item()` | weird CPU↔GPU stalls | move logging off the hot path |

A healthy 8-GPU DDP run on a single node should get >80% scaling efficiency for a ~125M model with batch 32, seq 1024 (a small model has less compute to hide the all-reduce behind, so 8-way efficiency runs a bit below the 4-way number — push the model or micro-batch up and it recovers).

### 6. The "DDP from scratch" mental model

Before using `DistributedDataParallel`, do it by hand:

```python
loss = compute_loss(batch)
loss.backward()
# now every param has .grad set
for p in model.parameters():
    if p.grad is None:
        continue
    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
optimizer.step()
```

This is functionally equivalent to DDP but slower (no overlap). The point is to *see* that DDP is just gradient averaging — once you've written it, the production version stops being magic.

---

## Hands-on

### Exercise 1 — DDP from scratch

In [`ddp_from_scratch.py`](ddp_from_scratch.py) (provided as a skeleton), wire up:

1. `dist.init_process_group("nccl")` from environment.
2. Replicate the model: every rank instantiates the same model with the same seed, then `dist.broadcast` the parameters from rank 0 to confirm they're identical.
3. After every backward, for each parameter with a grad, run `dist.all_reduce` with `op=AVG` (or `SUM`+ divide by world size).
4. Confirm that a 1-rank run and an N-rank run with `micro_batch_per_rank = batch / N, accum=1` produce the same loss after the same number of *global* steps. (This won't be exactly bitwise — different reduction order in NCCL — but should match to ~1e-3 relative.)

Run:
```bash
torchrun --standalone --nproc_per_node=8 ddp_from_scratch.py --config configs/small.yaml
```

**Deliverable:** Side-by-side loss curves at world_size=1 and world_size=8 with `global_batch` held constant.

### Exercise 2 — Replace with `DistributedDataParallel`

Now port to the real thing in [`train_ddp.py`](train_ddp.py):

```python
model = GPT(cfg).cuda()
model = DDP(model, device_ids=[local_rank], bucket_cap_mb=25)
```

That's it — backward now triggers bucketed all-reduces automatically.

Compare the *step time* between your hand-rolled version and DDP. DDP should be 1.3–1.8× faster for the small preset, because of overlap. (Bigger models → bigger overlap win.)

**Deliverable:** Table: `(version, step_ms, tokens/sec, scaling_efficiency_vs_1gpu)` for hand-rolled and DDP, at world sizes 1, 2, 4.

### Exercise 3 — Scaling-efficiency study

Hold `global_batch=128` and sweep `world_size ∈ {1, 2, 4}` with `micro_batch = 128 / world_size`. Report tokens/sec and scaling efficiency (tps(N) / (N × tps(1))).

Then *vary* `micro_batch` per rank to find the most efficient point. You're looking for the smallest micro-batch that still saturates the GPU enough that all-reduce overhead amortizes.

**Deliverable:** Plot of scaling efficiency vs micro-batch per rank. Note the elbow.

### Exercise 4 — Bucket size tuning

DDP's `bucket_cap_mb` defaults to 25 MB. Try `5`, `25`, `100`. For your model size and link bandwidth, find where it stops mattering.

**Deliverable:** 3-row table. A 1-paragraph reading of why bigger isn't always better.

### Exercise 5 — Profile a step

Use `torch.profiler` with `record_shapes=True` and `with_stack=False`. Capture 3 training steps. In the trace:

1. Find the `nccl:all_reduce` events. Do they overlap with backward kernels, or sit at the end?
2. Find the *last* all-reduce. Does any backward kernel still run during it? (Probably not — it's the leftover bucket.)
3. Measure `forward_ms / backward_ms / allreduce_ms / opt_ms / total_ms` and plot a stacked bar.

**Deliverable:** Annotated screenshot in `NOTES.md`.

### Exercise 6 — Find a real bottleneck

Run the small preset with `num_workers=0` for the dataloader. Then `num_workers=4` with `pin_memory=True, persistent_workers=True, prefetch_factor=4`. Compare tokens/sec.

For larger models, data loading rarely matters; for smaller models it can dominate. You should see the difference yourself, not just hear about it.

**Deliverable:** Numbers in `NOTES.md`.

---

## Self-check

1. What's the formula for ring-all-reduce cost on N ranks and B bytes?
2. Why does PyTorch DDP bucket gradients?
3. If you double world size and keep global batch constant, what should happen to tokens/sec? What might prevent it?
4. What's the difference between `DistributedSampler` and just sampling random indices on each rank?
5. Why do you need to seed each rank differently for dropout but identically for weight init?
6. What happens if rank 0 calls `.backward()` and rank 1 calls `.eval()` — describe the deadlock.

---

## Deliverables before moving on

- `ddp_from_scratch.py` working, validated against single-GPU loss curve.
- `train_ddp.py` using `DistributedDataParallel`.
- Scaling efficiency plot.
- Bucket-tuning table.
- Profiler screenshot showing all-reduce overlap with backward.
- `NOTES.md` with 5–10 bullets on what surprised you about DDP.
