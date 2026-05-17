# Chapter 05 — Tensor Parallelism

**Hardware target:** 2–4 GPUs with NVLink. TP across PCIe is painful and not recommended.
**Time budget:** 3 weeks.
**Prerequisites:** Chapter 04 — FSDP works.

---

## Why this chapter exists

Tensor parallelism (TP) splits the *math inside each layer* across GPUs, not the data. It's the parallelism scheme that lets you fit very wide models (giant hidden sizes, gigantic vocab heads) on multiple GPUs while still running each forward pass collectively.

TP is also the most numerically subtle parallelism scheme in the toolkit:

- A wrong TP implementation gives subtly different outputs vs. single-GPU, *not* an obvious crash.
- TP-comm happens *inside* every transformer block, not once per step like DDP. That's why TP wants NVLink and almost never crosses node boundaries.
- TP composes with DP and PP — Chapter 07 combines all three.

By the end you will have implemented column- and row-parallel linear layers, sharded a transformer block (attention + MLP) across 4 GPUs, added sequence parallelism, used vocab-parallel cross-entropy, and benchmarked the trade-offs.

---

## Read first

- **Megatron-LM paper** (Shoeybi et al., 2019): the canonical TP design.
- **Sequence parallelism** section of the Megatron-LM v3 paper / Korthikanti et al.
- **PyTorch DTensor + `torch.distributed.tensor.parallel`** docs. This is the modern primitive PyTorch is moving toward.
- **Megatron-LM source** — `megatron/core/tensor_parallel/`. Especially `layers.py` (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding).
- **Ultra-Scale Playbook** TP chapter and any Nanotron TP discussion.

---

## Concepts

### 1. Column-parallel linear

For `y = x @ W^T` where `W` is `(out, in)`, *split W along its rows* (output dim) across GPUs. Each GPU computes its slice of `y`:

```
                ┌── GPU 0: W[:out/N, :]
W (out, in) ────┼── GPU 1: W[out/N:2*out/N, :]
                ├── GPU 2: ...
                └── GPU 3: W[3*out/N:, :]
```

Forward:
- Input `x` is **replicated** across GPUs (each has the full `x`).
- Each GPU computes its slice of `y`.
- Output `y` is **sharded** along the output dim (each GPU has 1/N of `y`).

Backward:
- Output gradient `dy` is sharded the same way as `y`.
- Local weight grad: `dW_local = dy_local^T @ x` — no comm needed.
- Input grad: `dx_local = dy_local @ W_local` — gives the slice of `dx`. Combine with `all_reduce` (sum) to get `dx`.

Column-parallel is the standard input layer for an attention block (Q/K/V projection) and the input of an MLP.

### 2. Row-parallel linear

Split `W` along *columns* (input dim). Each GPU has `W[:, in/N * k : in/N * (k+1)]`.

Forward:
- Input `x` is sharded along its last dim (each GPU has 1/N of `x`).
- Each GPU computes a partial `y`: `y_local = x_local @ W_local^T`.
- `y` is **replicated** after an `all_reduce(SUM)` of partial outputs.

This is the standard output layer of attention (output projection `W_o`) and of an MLP (down-projection).

### 3. The Megatron block pattern

The clever observation: chain `column → row` and the intermediate stays sharded.

For an MLP `y = x → W1 → activation → W2`:

- `W1` column-parallel: `x` replicated, hidden sharded along output (intermediate) dim.
- Activation: element-wise, runs on the sharded hidden — no comm.
- `W2` row-parallel: input sharded, output replicated — requires an all-reduce of the partial outputs.

**Cost:** one all-reduce per MLP. Same pattern for attention: column-parallel `W_Q, W_K, W_V`, attention runs on sharded heads, row-parallel `W_O` with all-reduce at the end. So a transformer block has **2 all-reduces per forward**, plus 2 on backward = 4 total.

### 4. Sequence parallelism

TP all-reduces the *full sequence × hidden* tensor every block. For long sequences this is huge. Sequence parallelism (SP) is a refinement:

- The non-TP layers (layer norms, residuals, dropout) also act on the same (S × H) tensor.
- Instead of replicating across TP ranks, *shard along the sequence dimension*.
- Replace the `all_reduce(SUM)` in row-parallel with `reduce_scatter` (gives sequence-sharded output), and convert the next column-parallel's input from sequence-sharded back to hidden-replicated with `all_gather`.

Net effect: same total bytes moved, but you avoid running layer norm / residual / dropout on the full replicated tensor. Saves a lot of activation memory at long context. The right default for any modern TP implementation.

### 5. Vocab-parallel cross-entropy

The LM head `W_lm: hidden → vocab` is *huge* — a 32k vocab × 4096 hidden = 128M params. Sharding across TP ranks halves that per GPU. But the softmax cross-entropy at the end is tricky:

- Each GPU computes logits for its vocab shard.
- To compute the global softmax, you need to all-reduce the per-vocab-shard max (for numerical stability) and the per-vocab-shard exp-sum.
- Then each GPU computes its local contribution to the cross-entropy and you `all_reduce(SUM)` the scalar at the end.

Megatron-LM's `vocab_parallel_cross_entropy` implements this. Use it; don't roll your own unless you want a multi-day numerics bug hunt.

### 6. Why TP wants NVLink

Per transformer block, TP needs (with SP):
- 1× `all_gather` (S → S/N along seq)
- 1× `reduce_scatter` (the row-parallel output)
… per block, per forward and backward.

For a 24-layer model at S=2048, H=4096, that's `48 * 2 * 2048 * 4096 * 2 bytes ≈ 6.4 GB` per step over the TP comm dim, *per direction*. On NVLink that's a few ms; on PCIe it would dominate the step.

This is why TP is almost always *intra-node*. Cross-node TP is technically possible but rarely worth the complexity.

### 7. TP degree limits

Practical TP degrees:

- **TP=2 or TP=4** on a 4-GPU node — standard.
- **TP=8** on a DGX-8 — standard for 30B+ models.
- **TP=16+** — needs careful interconnect. Usually combined with PP across the slow links.

If your TP degree doesn't divide cleanly into your model dimensions, expect padding or quality regressions.

---

## Hands-on

### Exercise 1 — Toy column- and row-parallel linear

In [`tp_linear.py`](tp_linear.py), implement:

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_size, tp_rank, bias=False): ...
    def forward(self, x): ...   # x replicated; output sharded on last dim

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_size, tp_rank, bias=False): ...
    def forward(self, x): ...   # x sharded on last dim; output replicated via all_reduce
```

Tests in [`test_tp_linear.py`](test_tp_linear.py) verify numerical equivalence to a single-GPU `nn.Linear`. Run:
```bash
torchrun --standalone --nproc_per_node=4 -m pytest test_tp_linear.py -v
```

**Deliverable:** Tests pass at tp_size ∈ {1, 2, 4}.

### Exercise 2 — Sharded attention

Replace the `Attention` in your Chapter 01 model with [`TPAttention`](tp_block.py):

- Q, K, V projections are column-parallel. Heads are split across TP ranks.
- Output projection is row-parallel.
- For GQA, each TP rank still gets a whole number of KV heads (so `n_kv_heads % tp_size == 0`).

The attention math itself runs *on sharded heads* — no comm needed in the SDPA call.

**Deliverable:** A unit test that asserts `TPAttention` with `tp_size=4` produces the same output (within float tolerance) as `Attention` on a single GPU, given the same weights.

### Exercise 3 — Sharded MLP

Replace `SwiGLUMLP` with a sharded version. `w1` and `w2` (the two up-projections) are column-parallel; `w3` (down-projection) is row-parallel.

**Deliverable:** Numerical-equivalence test as in Exercise 2.

### Exercise 4 — Train a TP transformer

In [`train_tp.py`](train_tp.py), wire your sharded block into the model and train the `small` preset on 4 GPUs with tp=4.

Compare to (a) single GPU and (b) DDP at world=4 on the same global batch. Loss curves should overlap. Tokens/sec — discuss in `NOTES.md`. (TP is rarely faster than DDP for small models; the win is fitting bigger ones.)

**Deliverable:** Three loss curves overlayed.

### Exercise 5 — Sequence parallelism

Add sequence parallelism. The cleanest way is to use PyTorch's `torch.distributed.tensor.parallel` and `parallelize_module` with the `SequenceParallel` style, or implement by hand following the Megatron pattern (`all_gather` after RMSNorm, `reduce_scatter` after the row-parallel projection).

Train the `medium` preset with TP=4 and SP. Measure peak memory before and after enabling SP at seq_len=4096.

**Deliverable:** Two-row table: (TP=4 no SP, TP=4 with SP). Peak memory and tokens/sec.

### Exercise 6 — Vocab parallel cross-entropy

Shard the LM head. Implement a vocab-parallel cross-entropy that:

1. Computes local logits on each TP rank.
2. All-reduces the max across ranks for numerical stability.
3. Computes local exp-sum and all-reduces it.
4. Returns per-token loss (each rank has same value).

Read Megatron's `vocab_parallel_cross_entropy.py` for reference. Then write a unit test that compares to a single-GPU `F.cross_entropy`.

**Deliverable:** Unit test passes. Note the memory savings: the lm_head was `vocab × hidden` — now `(vocab / TP) × hidden`.

### Exercise 7 — Profile the comm

For the `medium` model at TP=4, use the PyTorch profiler. Look at the trace:

1. Per block, how many NCCL ops do you see in forward? In backward?
2. Which ones are `all_reduce` vs `all_gather` vs `reduce_scatter`?
3. Does any compute overlap with the all-reduce? (Hint: with SP, the `reduce_scatter` can overlap; without SP, the `all_reduce` mostly can't.)

**Deliverable:** Annotated trace screenshot in `NOTES.md`.

---

## Self-check

1. Why is the column→row pairing the natural building block?
2. For TP=4 on a transformer block, how many all-reduces per forward? Per backward?
3. How does sequence parallelism reduce activation memory without changing the total bytes communicated?
4. Why does vocab-parallel cross-entropy need *two* all-reduces, not just one?
5. Why is TP almost always intra-node?
6. What goes wrong if you set `tp_size=3` and your model has 12 attention heads?

---

## Deliverables before moving on

- `tp_linear.py` implementing column-parallel and row-parallel linear.
- `tp_block.py` implementing TPAttention and the sharded MLP.
- All TP unit tests numerically match single-GPU within ~1e-3 relative.
- `train_tp.py` trains a `small` model with TP=4.
- `medium` model with TP=4 + SP fits and trains.
- `NOTES.md` with annotated profile and memory-savings table.
