# Chapter 10 — Triton Kernels

**Hardware target:** 1 H100 (Triton ≥ 3.0 targets Hopper — `tl.dot` lowers to WGMMA and FP8 inputs hit native tensor cores; the lessons transfer to other GPUs).
**Time budget:** 4 weeks.
**Prerequisites:** Chapter 09 — you can read a trace and a roofline.

---

## Why this chapter exists

When you read vLLM, SGLang, Megatron, or any modern production LLM codebase, you'll find handwritten Triton kernels for the hot ops: RMSNorm, softmax, fused attention, fused activations, quantized matmuls. Knowing what these kernels look like — and being able to write a passable one yourself — is a major step toward "I can contribute to these projects."

Triton is the right starting point for two reasons:
1. **Python ergonomics.** You write Python with NumPy-like array semantics; Triton compiles to PTX. Iteration is fast.
2. **The mental model is the right one.** Triton thinks in *blocks of threads* with *block-level memory loads* — the same abstraction CUTLASS uses internally, just with less ceremony.

By the end of this chapter you will:

1. Be able to read a Triton kernel and understand what it's doing without staring.
2. Have written and benchmarked: vector add, RMSNorm, softmax, matmul, fused SwiGLU, and at least one custom variant of attention.
3. Know when Triton wins (custom shapes, fused element-wise, custom attention) and when CUTLASS/cuBLAS wins (standard square matmuls).

---

## Read first

- **Triton website + tutorials** — work through all the official tutorials in order. They are excellent and we deliberately build on them here.
- **Triton paper** (Tillet et al., 2019) — useful context, but the docs are more current.
- **Phil Tillet's talks** on Triton — search YouTube; high-bandwidth intuition.
- **OpenAI Triton GitHub** — the source for `triton.jit`, `triton.testing`, and the language extensions.
- **vLLM's `csrc/`** and Triton kernels in `vllm/model_executor/layers/` — production references.

---

## Concepts

### 1. The Triton programming model

A Triton kernel processes a *block* of data at a time, parameterized by `BLOCK_SIZE`. Each block is run by one *program* (analogous to a CUDA block). Within a program, you write Python-level ops on `tl.tensor` objects, which are vectorized across `BLOCK_SIZE`.

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

Compared to writing a CUDA kernel:
- No `threadIdx` / `blockIdx` for individual threads — Triton handles intra-block parallelism for you.
- `tl.load` / `tl.store` issue *vectorized*, optionally masked, memory operations.
- Shared memory and registers are managed by the compiler. You generally don't allocate them by hand.

### 2. Tiling

For matmul (or anything 2D), tile both axes. The classic block-matmul pattern:

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid %  tl.cdiv(N, BLOCK_N)

    # Pointer arithmetic for this (BLOCK_M, BLOCK_N) tile of C.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (offs_k + k)[None, :] * stride_ak)
        b = tl.load(b_ptr + (offs_k + k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)        # uses Tensor Cores when shapes/dtype align

    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc.to(c_ptr.dtype))
```

The `tl.dot` call is what hits Tensor Cores. For BF16 inputs on H100, you'll get many hundreds of TFLOPS — Triton lowers `tl.dot` to Hopper's warp-group MMA (WGMMA).

### 3. Autotuning

Triton has built-in autotuning over `@triton.autotune(configs=[...], key=[...])`. You provide a list of (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) configs; Triton picks the best for each input shape.

The most-common knobs:
- **num_warps**: how many warps per program (i.e., parallelism within a block). 4 or 8 usually.
- **num_stages**: how many pipeline stages for asynchronous loads (think double/triple buffering). 2–4 typical.
- **BLOCK_***: tile sizes. Bigger = more reuse but more registers/shared-memory pressure.

### 4. When Triton wins vs cuBLAS

| Scenario | Best choice |
|---|---|
| Standard square FP/BF matmul, shape divides nicely | cuBLAS (via `torch.matmul`) |
| Tall-skinny / weird shapes (e.g., decode-phase matvec) | Triton or CUTLASS |
| Fused element-wise + matmul + element-wise | Triton |
| Fused attention (causal + RoPE-on-K + dropout) | Triton or FlashAttention/CUTLASS |
| Quantized matmul (INT8, INT4, native FP8) | Triton (CUTLASS too, but Triton wins for new dtypes) |
| Softmax, layer norm, RMSNorm | Triton easily |

For "ordinary" matmul, don't try to beat cuBLAS unless you have a very specific reason.

### 5. Numerical stability

Triton lets you compute in any dtype, including FP32 inside a BF16 kernel. For softmax / norm-like ops, *always* compute reductions in FP32 even when inputs are BF16. The pattern:

```python
x = tl.load(x_ptr + offsets).to(tl.float32)
max_x = tl.max(x, axis=-1)        # FP32 reduction
exp_x = tl.exp(x - max_x[:, None]) # FP32
sum_x = tl.sum(exp_x, axis=-1)
out = (exp_x / sum_x[:, None]).to(x_ptr.dtype.element_ty)
tl.store(out_ptr + offsets, out)
```

This is the same pattern FlashAttention uses for its online softmax.

### 6. Online softmax (for attention)

When you can't materialize the full `Q K^T` (memory!), you compute the softmax in tiles, keeping a running max and running sum. The math:

```
For tile k:
    s = max(running_max, max(tile_k))
    correction = exp(running_max - s)
    running_sum = correction * running_sum + sum(exp(tile_k - s))
    running_max = s
    output_so_far = correction * output_so_far + tile_k_softmax @ V_tile
```

This is the core of FlashAttention. Section 3 of the FlashAttention paper is required reading before Exercise 7.

### 7. Persistent kernels

For decode-phase inference, you might launch hundreds of tiny matmuls per token. Each has fixed launch overhead. *Persistent* kernels run for a long time on the GPU, polling a queue for work. Hot in newer inference engines.

We won't write persistent kernels here — they're advanced — but recognize the term.

---

## Hands-on

Each exercise has its own file. Start simple and accumulate.

### Exercise 1 — Vector add (warmup)

[`vector_add.py`](vector_add.py): write the Triton vector-add. Benchmark against `torch.add` on the same shape. The Triton kernel should match or slightly beat `torch.add` for medium sizes (it's a trivial element-wise op; the gap is small).

**Deliverable:** Bandwidth (GB/s) for Triton and torch versions, in a printed table.

### Exercise 2 — RMSNorm

[`rmsnorm.py`](rmsnorm.py): RMSNorm forward (and optionally backward). Compute the RMS in FP32, then normalize. Compare to your Chapter 01 implementation.

The Triton version should beat eager PyTorch by 1.5–3× for typical shapes (B, S, H) = (4, 2048, 4096), because it fuses the reduction and the scale.

**Deliverable:** Speedup vs eager. Also bandwidth-bound check on the roofline.

### Exercise 3 — Online softmax

[`softmax.py`](softmax.py): a numerically-stable softmax kernel for the last dimension of a 2D tensor. Use the online-softmax trick: one pass to compute max + exp-sum.

Benchmark against `torch.softmax`.

**Deliverable:** Speedup table for (rows × cols) shapes (1024×1024), (4096×4096), (1×32000).

### Exercise 4 — Tiled matmul

[`matmul.py`](matmul.py): port the Triton tutorial matmul. Add autotuning.

Benchmark against `torch.matmul` at three shapes:
- (4096, 4096) × (4096, 4096) — large square.
- (8, 4096) × (4096, 4096) — tall-skinny "decode-phase" matvec.
- (4096, 7) × (7, 4096) — weird intermediate.

For the large square, cuBLAS wins. For the tall-skinny, your Triton kernel *can* win if you autotune well — this is the "weird shape" regime where Triton shines.

**Deliverable:** 3-row table. Note that cuBLAS isn't always better.

### Exercise 5 — Fused SwiGLU

[`fused_swiglu.py`](fused_swiglu.py): fuse the SwiGLU MLP forward — given `x`, weights `W1, W2, W3`, compute the final output in one kernel. (Or two: matmul + fused activation × matmul. Free choice.)

Benchmark against the eager 3-matmul implementation.

**Deliverable:** Speedup and a roofline placement.

### Exercise 6 — Quantized matmul (stretch)

[`int8_matmul.py`](int8_matmul.py): write a Triton kernel that does `int8_a @ int8_b → int32_acc → scale → bf16_out`. This is the shape of an INT8-weight-only quantized linear layer.

Compare to a BF16 matmul. INT8 should be roughly 2× faster on H100 (its INT8 peak is 2× its BF16 peak). Stretch further: redo the kernel in FP8 (`tl.float8e4nv` inputs, FP32 accumulate) — on Hopper `tl.dot` lowers this to native FP8 tensor cores, the same path production engines use for FP8 inference.

**Deliverable:** Speedup. Note this is the same kernel you'd plug into vLLM's quantized inference paths.

### Exercise 7 — A toy FlashAttention (very stretchy)

[`flash_attn_toy.py`](flash_attn_toy.py): implement a simplified FlashAttention forward — one head, causal, fixed BLOCK_M=BLOCK_N=BLOCK_K. Don't worry about backward.

You won't beat FlashAttention 2/3. The point is to see the algorithm in 200 lines of Triton instead of 1000+ of CUDA.

**Deliverable:** Numerical match against `F.scaled_dot_product_attention` (within ~1e-3 relative on BF16). Speed comparison; expect FA2 to win by a lot.

---

## Self-check

1. What does `tl.program_id` give you, and what's the analog in CUDA?
2. Why do reductions inside a Triton kernel typically use FP32 even when inputs are BF16?
3. What does `tl.dot` actually compile to on H100? (Hint: WGMMA — and how does that differ from the Ampere `mma`?)
4. When would you pick a small BLOCK_K vs large BLOCK_K?
5. What's `num_stages` and what's the trade-off when it grows?
6. Why is online softmax the right way to do attention without materializing the score matrix?

---

## Deliverables before moving on

- All Exercise 1–5 kernels written and benchmarked.
- A `kernel_bench.csv` with per-shape numbers for each kernel.
- Roofline plot for at least RMSNorm and softmax.
- A short writeup: "I would write a custom Triton kernel when…"
- (Stretch) INT8 matmul and toy FlashAttention if time allows.

This chapter is what separates "I can read kernel code" from "I can write one." Don't skip the benchmark/roofline checks — knowing *whether your kernel is good* is half the skill.
