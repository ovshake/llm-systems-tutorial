# 01 — Tensor Core Evolution

Every modern LLM article assumes you know what a tensor core is, what it can do, and how it has changed across generations. This file walks the evolution.

The headline: each generation of tensor cores **changes the shape**, **adds a dtype**, and **rebalances issue/data movement**. Reading any architecture article amounts to: what did this generation change along those three axes?

---

## What a tensor core actually computes

One instruction, one warp, one matrix-multiply-accumulate:

```
D = A * B + C
```

A, B, C, D are matrices of fixed shape (per instruction). The *MMA instruction family* is the language tensor cores speak; in PTX you'll see things like `mma.sync.aligned.m16n8k16...`.

The 'shape' notation is `m × n × k`: A is m×k, B is k×n, C and D are m×n.

A single MMA gives you `m * n * k * 2` FLOPs (multiply + add). So `m16n8k16` gives 16 * 8 * 16 * 2 = 4096 FLOPs per instruction. At one issue per cycle and ~108 SMs at ~1.4 GHz, that gets you the ~620 TFLOPs the marketing slide brags about (after correction for instruction issue rate, etc.).

---

## Volta (2017) — the first tensor cores

- Shape: 4×4×4. Two issues per warp can do an 8×8×4.
- Dtypes: FP16 inputs, FP32 accumulate.
- Used as: PyTorch `torch.cuda.amp` FP16 + GradScaler. Trained BERT, GPT-2.

Volta is the *concept* (replacement for CUDA-core matmuls with dedicated dense-MM units). The shapes were awkward and the dtype coverage thin. Useful to read about for context but you'll rarely tune for Volta today.

---

## Turing (2018) — consumer + INT

- Same FP16/FP32 as Volta.
- **Added INT8 / INT4 tensor cores** — first widespread use of integer MMA. Important for quantized inference.
- Shape additions: more m×n×k options.
- Hardware: T4, RTX 20-series.

T4 was the workhorse inference GPU for years; INT8 quantization on Turing was where production quantized inference started.

---

## Ampere (2020) — A100

- **Shapes:** `m16n8k16` is the canonical Ampere FP16/BF16 MMA. Larger options exist.
- **Dtypes added:** BF16 (a big deal — wider exponent than FP16, kills `GradScaler`), TF32 (a 19-bit format for "FP32-like" tensor core matmul), FP64.
- **2:4 structured sparsity:** if your weight matrix has at most 2 nonzeros per group of 4, hardware does the matmul in half the time. Inference quantization technique.
- **`cp.async`:** the first async memory copy instruction. Sets up the multi-stage pipeline.

Ampere is the previous generation, but everything it added — BF16, TF32, `cp.async`, 2:4 sparsity — carries forward into Hopper, so it's worth understanding even though it's no longer your target.

---

## Hopper (2022) — H100

Three substantial changes that you'll hear about constantly:

### 1. Warp-Group MMA (WGMMA)

Instead of a *warp*-level instruction (32 threads), Hopper introduces *warp-group* (128 threads = 4 warps) instructions:

```
wgmma.mma_async.sync.aligned.m64n256k16...
```

The shape is dramatically wider: typically `m64nNk16`. Why?
- Bigger MMAs amortize the (now-larger) overhead of feeding the tensor core.
- The warp-group becomes the natural unit of producer/consumer.

WGMMA is also **asynchronous**: the issue returns immediately, the warp can do other work, and a `wgmma.commit_group`/`wgmma.wait_group` pair fences when the result is needed. This enables overlapping issue with TMA loads.

### 2. Tensor Memory Accelerator (TMA)

A hardware engine for *strided multi-dimensional* loads/stores between HBM and shared memory. One instruction transfers a full tile. Releases the warp to do other things while the transfer is in flight.

The producer/consumer pattern:
- Some warps ("producers") issue `tma.load`s.
- Other warps ("consumers") issue `wgmma`s on the loaded tiles.
- They sync via `mbarrier`.

This is the structural core of FlashAttention 3.

### 3. Transformer Engine, FP8

Hopper added **FP8 tensor cores** (two formats: E4M3 and E5M2 — see `03-numeric-formats.md`). The "Transformer Engine" is a *library* that automates the per-tensor scale management for FP8 — you can't just cast to FP8 and hope; you need running scales per tensor to keep numerics sane.

For training: FP8 weights/activations + FP32 master + per-tensor scale.

### 4. CTA clusters

(Covered in `00-hardware-fluency.md`.) New scheduling level above the block: 2–16 blocks scheduled on adjacent SMs in the same GPC, with shared distributed memory and a cluster barrier.

This is what your 8×H100 box is. Most of the Triton and CUTLASS kernels you'll write target this generation — WGMMA, TMA, and FP8 are the tools you actually have.

---

## Blackwell (2024) — B100 / B200

Blackwell is what the Blackwell Tensor Memory article is about. The big changes:

### 1. TCGen05 (5th-gen tensor cores)

The new tensor core *generation*. It supersedes WGMMA. PTX gains a new instruction family `tcgen05.*`. Key differences:

- **Bigger native shapes.** TCGen05 issues a much larger MMA per instruction; you can think of it as one warp-group dispatching a much wider matmul.
- **Tensor Memory operands.** TCGen05 reads operands not from shared memory or registers directly, but from a new memory space called **Tensor Memory (TMEM)**. Operands stage into TMEM via TMA-like instructions, then the MMA executes against TMEM.
- **Block-scaled MMA.** Hardware-level support for block-scaled formats (MX) without per-element scaling overhead.
- **FP4 inputs.** With block scales (MXFP4 / NVFP4), tensor cores can do MMAs in FP4. Roughly 2× the throughput of FP8, with quality close to BF16 for inference if you scale carefully.

### 2. Tensor Memory (TMEM)

A new memory space *between* registers and shared memory, sized per-SM in the hundreds of KB. Designed specifically for tensor core operands and accumulators.

The motivation: WGMMA accumulators on Hopper had to live in registers, and the register file (~256 KB per SM) was a bottleneck when you wanted huge accumulator tiles. TMEM gives accumulators their own backing storage that:
- Is bigger than the register file.
- Is read by tensor cores directly (no register-file round trip).
- Has its own dedicated bandwidth to the tensor cores.

The Blackwell Tensor Memory article is a deep dive on exactly this. Case study in `case-studies/blackwell-tensor-memory.md`.

### 3. Async issue, more aggressive pipelines

TCGen05 instructions are async-by-default. The producer/consumer pattern from Hopper becomes even more central — TMEM is essentially a third stage in the pipeline (HBM → SMEM → TMEM → tensor core).

### 4. Dual-die packaging

B200 is two compute dies on the same package, linked by NVLink-C2C. From a software perspective there are still some quirks (e.g., NUMA-like effects across the dies), but most CUDA code treats it as one big GPU.

---

## MMA shapes you'll see in articles

A non-exhaustive cheat sheet:

| Notation | Generation | Inputs | Output |
|---|---|---|---|
| `m16n8k16` | Ampere | FP16 / BF16 | FP32 |
| `m16n8k8`  | Ampere | TF32 | FP32 |
| `m16n8k32` | Ampere | INT8 | INT32 |
| `m16n8k64` | Ampere | INT4 | INT32 |
| `m64n128k16` (WGMMA) | Hopper | BF16 | FP32 |
| `m64n256k16` (WGMMA) | Hopper | BF16 | FP32 |
| `m64n128k32` (WGMMA) | Hopper | FP8 | FP32 |
| Many wider TCGen05 shapes | Blackwell | FP8, MXFP6, MXFP4, NVFP4 | FP32 |

When an article cites a shape, that's the *instruction's* shape, not your matmul's shape. Your GEMM tiles over many MMAs.

---

## Reading the marketing numbers

NVIDIA presents tensor TFLOPs by dtype, "dense" and "sparse." When you see "Hopper: 989 TFLOPS BF16, 1979 TFLOPS FP8":

- 989 is BF16 *dense* peak. Your real MFU will be 40–70% of this.
- 1979 is FP8 *dense*. Roughly 2× BF16 because the dtype is half as wide.
- *Sparse* is 2× either of those, but only if your weights have the 2:4 structured sparsity pattern (rare in real training).

When B200 is quoted at "4500 TFLOPs FP8" or "9000 TFLOPs FP4," that's again dense peak. Real inference workloads land in the 50–70% range with careful kernel work.

---

## What to read next

- **NVIDIA H100 whitepaper, section "Hopper Streaming Multiprocessor"** — the canonical reference for WGMMA and TMA.
- **NVIDIA Blackwell whitepaper.**
- **PTX ISA documentation, section "Asynchronous Warpgroup Level Matrix Instructions"** — for WGMMA syntax.
- **PTX ISA, `tcgen05.*` section** — for TCGen05.
- **FlashAttention-3 paper and source** — production WGMMA + TMA usage.
- **CUTLASS docs `cute/atom/mma_atom.hpp` etc.** — how CUTLASS abstracts MMA instructions.
- Move on to `02-memory-spaces.md`.

---

## Self-check

1. What's the difference between an MMA and a CUDA-core multiply-add?
2. What problem does WGMMA solve that MMA doesn't?
3. Why does WGMMA being asynchronous matter?
4. What's the structural relationship between TMA, mbarrier, and WGMMA?
5. What does TCGen05 add that WGMMA didn't?
6. What does 2:4 structured sparsity require of your weights?
