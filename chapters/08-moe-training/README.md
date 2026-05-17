# Chapter 08 — MoE Training

**Hardware target:** 4× A100-80GB.
**Time budget:** 3–4 weeks.
**Prerequisites:** Chapter 07 — you have 3D parallelism intuition.

---

## Why this chapter exists

Mixture-of-Experts models add a *fourth* parallelism dimension and a fundamentally different communication pattern. A dense transformer's parallelism is "split matmul"; an MoE's parallelism is "route tokens to the right machine, run the right expert, route back."

The systems consequences are huge:

- The dominant communication is `all_to_all`, not `all_reduce`. Latency and imbalance matter as much as bandwidth.
- Token routing is non-deterministic in output volume — expert utilization is uneven, leading to padding or token drops.
- Memory accounting is different — most parameters are "rarely active" experts.

By the end of this chapter you will:

1. Have implemented a toy MoE layer (top-k router, load balance loss, expert MLPs).
2. Have implemented expert parallelism with explicit `all_to_all` for token routing.
3. Have measured token imbalance and all-to-all cost as the dominant bottleneck.
4. Be able to read MegaBlocks-style "dropless" MoE and explain why it's preferred.
5. Know what "shared experts" and "fine-grained experts" mean (DeepSeek-V3 style).

---

## Read first

- **Sparsely-Gated MoE** (Shazeer et al., 2017) — origin.
- **GShard** (Lepikhin et al., 2020) — capacity factor, top-2 routing, the token-drop mechanism.
- **Switch Transformer** (Fedus et al., 2021) — simpler top-1 routing.
- **MegaBlocks** (Gale et al., 2022) — dropless MoE via block-sparse matmul. The key paper.
- **Mixtral 8x7B / Mixtral 8x22B technical report** (Jiang et al.) — modern production MoE.
- **DeepSeek-V3 technical report** — shared experts, fine-grained experts, auxiliary-loss-free routing.
- **Tutel** docs — Microsoft's MoE library, all-to-all optimizations.
- **DeepSpeed-MoE** docs.

---

## Concepts

### 1. The MoE block

A dense MLP block does:
```python
h = MLP(x)
```
with a single MLP applied to *all* tokens.

An MoE block does:
```python
logits = router(x)               # (B, S, E)
probs, idx = top_k(softmax(logits), k=2)
h = sum_k (probs[..., k] * MLP_{idx[..., k]}(x))
```

Where `E` is the number of experts (often 8, 16, 32, 64, 128…) and `k` is typically 1 or 2.

If you have 8 experts and use top-2 routing with 1B params per expert, your *total* parameter count is 8B but each token only sees 2 experts → ~2B of "active" compute per token. You get the representational capacity of an 8B dense model at the FLOPs cost of a 2B dense model.

### 2. Expert capacity, token drops, and padding

If you have 8 experts and a batch with 1024 tokens, ideally each expert gets ~128 tokens. In practice, the router routes some experts 200 tokens and others 50.

Two strategies:

- **Capacity factor + drop**: cap the per-expert token count at `C = ceil(N_tokens / E * capacity_factor * k)`. Tokens past the cap get *dropped* — their MLP contribution is just the residual (no expert computation). Easy to implement; quality degrades when drops are frequent.
- **Dropless / variable capacity (MegaBlocks)**: pad to the max bucket size or use block-sparse matmul that handles ragged batches efficiently. No drops, better quality, harder kernels.

Standard load balance loss (Shazeer/Switch):
```
L_aux = E * sum_e (f_e * P_e)
```
where `f_e` is the fraction of tokens routed to expert `e` and `P_e` is the average router probability for expert `e`. Minimized when `f_e = P_e = 1/E`. Often weighted by ~0.01 in the total loss.

### 3. Expert parallelism (EP)

Each expert is *placed* on a specific GPU (or shared across a TP slice of GPUs). For 8 experts on 4 GPUs (EP=4), each GPU holds 2 experts.

The routing dance:

1. Router computes top-k expert assignments per token.
2. Tokens get *permuted* by expert assignment.
3. **`all_to_all`**: every GPU sends its tokens-destined-for-other-experts to the right GPU, and receives others' tokens-destined-for-its-experts.
4. Each GPU runs its experts on the tokens it received.
5. **`all_to_all`** again to send results back.
6. Tokens get *un-permuted* to their original positions.

The two `all_to_all`s per MoE block are the dominant cost — usually 20–60% of step time for a well-tuned MoE. Optimizing all-to-all is most of what production MoE systems do (Tutel, GShard, DeepSeek's DeepEP library, etc.).

### 4. Token imbalance is the enemy

Even with a load-balance loss, in any given step some experts get more tokens than others. The slowest expert determines the wall-clock — same logic as pipeline bubble. Mitigations:

- Bigger batch sizes (more tokens to redistribute → law of large numbers).
- Capacity factor > 1.0 (extra slack at the cost of drops or padding).
- Smarter routing (e.g., DeepSeek-V3's "topology-aware" routing).

### 5. Shared experts (DeepSeek style)

Every token also goes through a "shared expert" MLP that's always active. The MoE is the *additional* sparse capacity. This stabilizes training (every token gets at least one MLP pass) and adds dense FLOPs without proportional memory cost.

### 6. Fine-grained experts (DeepSeek style)

Instead of 8 large experts, use 64 small ones. Each token routes to top-8 of 64 (still 12.5% activation rate, but more specialization). Comm volume per all-to-all roughly constant if total active dim is held constant; routing quality usually improves.

### 7. MoE composes with TP, PP, DP

In a real production layout:
- **EP** along one axis of the cluster (often within a node or two).
- **DP** for replication across the rest.
- **TP** for very large *non-MoE* layers (attention, embeddings).
- **PP** less commonly used with MoE; the routing already creates uneven compute per stage.

The Mixtral / DeepSeek codebases are good references for actual layouts.

---

## Hands-on

### Exercise 1 — Toy MoE layer (single GPU)

In [`moe.py`](moe.py) (skeleton), implement:

```python
class MoE(nn.Module):
    def __init__(self, hidden_size, intermediate_size, n_experts, top_k=2): ...
    def forward(self, x):     # x: (B, S, H)
        # 1. router: linear x → (B, S, E)
        # 2. softmax + top_k → probs (B, S, k), idx (B, S, k)
        # 3. for each expert e: gather tokens with idx == e, run MLP_e, scatter back
        # 4. weighted sum with probs
        ...
```

Plus a `load_balance_loss(router_probs, expert_indices)` that returns the Switch-Transformer auxiliary loss.

**Deliverable:** `pytest test_moe.py` passes. The toy MoE should produce the same output as the equivalent dense MLP when `n_experts=1, top_k=1` (sanity check).

### Exercise 2 — Train a tiny MoE

Replace the SwiGLU MLP in your Chapter 01 model with the MoE layer. Train on TinyShakespeare with 4 experts, top-2, on a single GPU. Confirm loss falls.

**Deliverable:** Loss curve. A log of expert utilization per step (fraction of tokens per expert).

### Exercise 3 — Expert parallelism with `all_to_all`

In [`moe_ep.py`](moe_ep.py), spread the experts across GPUs. With 4 experts on 4 GPUs (EP=4), each GPU holds 1 expert.

The forward becomes:
1. Each rank computes router output for its local tokens.
2. Permute tokens by expert.
3. `dist.all_to_all_single` to send tokens to the rank that owns the target expert.
4. Run local expert(s).
5. `all_to_all_single` back.
6. Un-permute.

Launch:
```bash
torchrun --standalone --nproc_per_node=4 train_moe_ep.py --config configs/small_moe.yaml
```

**Deliverable:** Loss curve matches single-GPU version (within stochastic noise).

### Exercise 4 — Measure token imbalance

Add logging that reports per-expert token count per step. Plot imbalance over training:

- The first 100 steps will show drift (the router is random).
- Once the load balance loss kicks in, imbalance should stabilize.
- The *worst* expert per step still gets ~1.5–2× the mean (variance doesn't go to zero).

**Deliverable:** Plot of `max(tokens_per_expert) / mean(tokens_per_expert)` over training.

### Exercise 5 — All-to-all is the bottleneck

Profile a step. Compare:

- Total step time.
- All-to-all time (sum of both directions).
- Expert MLP compute time.

For a small MoE on 4 NVLink-connected GPUs, all-to-all is typically 15–30% of step time. Try shrinking the model — the fraction grows; expert compute drops faster than all-to-all.

**Deliverable:** Stacked bar chart in `NOTES.md` showing the breakdown for 3 model sizes.

### Exercise 6 — Capacity factor and drops

Add a `capacity_factor` parameter to your MoE. With factor=1.0 (no slack) and a small batch, you'll see token drops. Increase to 1.25; drops should disappear at the cost of more padding.

Plot drop rate vs capacity factor for several batch sizes.

**Deliverable:** Plot. Note where the curves elbow.

### Exercise 7 — Dropless via padding-to-bucket

Read the MegaBlocks paper (Section 3, "Block-Sparse MoE"). You don't need to implement the full block-sparse kernel — instead, implement a "pad-to-max-bucket" version where every expert receives tokens padded to the largest expert's count for the step. This avoids drops at the cost of wasted compute on padding.

Compare loss curve and tokens/sec to your top-k-with-drops implementation.

**Deliverable:** Side-by-side loss curves. Note the tradeoff in `NOTES.md`.

### Exercise 8 — Shared expert and fine-grained experts (paper sketch)

Read DeepSeek-V3's MoE section. Sketch how you'd extend `moe.py` to add:

- A shared expert that's always active.
- Fine-grained experts: 16 experts, top-4 routing.

You don't have to fully run it — implementing the wiring change and confirming a forward pass works is enough.

**Deliverable:** Modified `moe.py` plus a short writeup in `NOTES.md` on the expected quality vs throughput tradeoffs.

---

## Self-check

1. Why is all-to-all the dominant comm in MoE training, instead of all-reduce?
2. What's the load-balance loss minimizing, and why isn't it perfect?
3. Capacity factor = 1.0 vs 1.25: which is more memory, which gives better quality?
4. Why are EP and TP often used together (not exclusively)?
5. What's the *dropless* MoE problem MegaBlocks solves?
6. Why does increasing the number of experts (with the same active fraction) tend to improve quality?

---

## Deliverables before moving on

- Working toy MoE (single GPU and EP=4).
- Token-imbalance plot.
- All-to-all-dominated profile.
- Capacity factor / drop rate study.
- Dropless via padding-to-bucket experiment.
- Writeup on shared experts + fine-grained experts.
