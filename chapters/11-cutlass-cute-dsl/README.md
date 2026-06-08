# Chapter 11 — CUTLASS & CuTe DSL

**Hardware target:** 1 H100. You can finally exercise the Hopper-specific CUTLASS templates (WGMMA, TMA, FP8) for real instead of "read, understand, defer" — these are the templates production kernels actually use.
**Time budget:** 4–6 weeks. This is the hardest chapter.
**Prerequisites:** Chapter 10 — Triton kernels written and benchmarked.

---

## Why this chapter exists

CUTLASS is NVIDIA's C++ template library for high-performance CUDA GEMM and adjacent kernels. It powers the kernels behind `cuBLAS`, FlashAttention 3, parts of vLLM, Megatron, etc. **CuTe DSL** is NVIDIA's Python-native DSL on top of CuTe (the layout algebra inside CUTLASS), targeting the same performance with much less ceremony.

You don't *have* to master CUTLASS to contribute to vLLM or SGLang. You do have to understand the CuTe layout vocabulary — it's the language production GPU engineers speak.

By the end of this chapter you will:

1. Know what "layout algebra" means and be able to read a `Layout` printout.
2. Have implemented a simple GEMM in CuTe DSL and benchmarked it.
3. Understand the threadblock/warp/instruction tile hierarchy and what CUTLASS template parameters mean.
4. Be able to compare your Triton kernel from Chapter 10 to a CuTe-DSL equivalent and explain the trade-offs.

This is the hardest chapter. Budget time accordingly.

---

## Read first

- **CUTLASS GitHub** (`NVIDIA/cutlass`) — `media/docs/` is the documentation directory; start with `efficient_gemm.md`, `gemm_api.md`, `pipeline.md`.
- **CuTe DSL docs** — the official NVIDIA Python DSL docs. Cover: layouts, JIT, autotuning, debugging.
- **"Persistent and Cooperative Kernels for Deep Learning" (NVIDIA blog)** and other CUTLASS blog posts.
- **FlashAttention 3 paper / source** — uses CUTLASS extensively on Hopper.
- **Lei Mao's "CUTLASS Tutorial" series** — among the clearest external references.

---

## Concepts

### 1. The CUTLASS hierarchy

CUTLASS organizes a GEMM as a *tiled* computation across three nested levels:

```
Threadblock tile    (e.g., 128×256×64)
  Warp-group tile   (e.g.,  64×256×64 — on Hopper a *warp group* of 4 warps, not a single warp)
    Instruction     (e.g.,  m64nNk16 WGMMA — Hopper warp-group MMA; the Ampere unit was a 16×16×16 warp MMA)
```

You choose tile sizes; CUTLASS generates the kernel that loads the right slabs into shared memory, distributes them to warps, and feeds the MMA pipeline. The template parameters in a CUTLASS GEMM declaration are *exactly these tile choices* plus the data types.

For an H100 BF16 matmul with `(M, N, K) = (8192, 8192, 8192)`:
- Threadblock 128×256×64 with two warp groups (8 warps) is a common Hopper starting point.
- Each warp group issues `m64nNk16` WGMMAs, with TMA feeding the next K-tile asynchronously.

### 2. Pipeline stages

A `cooperative` GEMM uses **multiple pipeline stages**: while one stage's `mma` runs, the next stage's `cp.async` is loading the next tile. Modern CUTLASS uses 2–6 stages, async memory copies (`cp.async` on Ampere; TMA on Hopper), and *warp specialization* (some warps do MMA, others do loads/stores).

CUTLASS lets you tune `NumStages` as a template parameter. More stages = more shared memory used but more latency hidden.

### 3. The CuTe Layout

CuTe is the layout algebra. A `Layout` is a *shape* and a *stride*:

```
Layout< Shape <_8, _4>, Stride <_4, _1> >
```

Means an 8×4 tensor laid out as: stride 4 in the first dim, stride 1 in the second. So element (i, j) is at offset `4*i + j`. *Sounds trivial, but the algebra (composition, complement, etc.) is what lets CUTLASS express any tile mapping cleanly.*

The CuTe layout vocabulary captures:
- How a tensor lives in HBM, shared memory, or registers.
- How threads tile that tensor.
- How an MMA instruction reads/writes its operands.

Master the basics: `shape`, `stride`, `cosize`, `composition`, `complement`. The CuTe DSL docs have worked examples.

### 4. Warp-specialized GEMMs

Modern kernels assign different jobs to different warps:

- *Producer warps*: load A and B tiles from HBM to shared memory.
- *Consumer warps*: do the MMA.

A barrier coordinates them. This was the breakthrough behind FlashAttention 3 on Hopper — producer warps use TMA (Tensor Memory Accelerator) to load while consumers run MMAs. The two pipelines never block each other.

### 5. CuTe DSL gives you all this in Python

Instead of templated C++:

```cpp
using Gemm = cutlass::gemm::device::GemmUniversal<
    bfloat16_t, RowMajor,
    bfloat16_t, ColumnMajor,
    bfloat16_t, RowMajor,
    float,
    arch::OpClassTensorOp,
    arch::Sm90,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp,
    ...
>;
```

You write Python:

```python
@cute.kernel
def gemm_kernel(a, b, c, *, tile=(128, 128, 32), num_stages=3, ...):
    ...
```

…and CuTe DSL JIT-compiles to CUDA. Much faster iteration. Production codebases will keep using C++ CUTLASS for stability and tooling, but CuTe DSL is the right pedagogical entry point.

### 6. When CUTLASS beats Triton

| | Triton | CUTLASS / CuTe DSL |
|---|---|---|
| Fast iteration | yes | newer (CuTe DSL helps) |
| Custom shapes / fused ops | excellent | possible but more work |
| Last-percent perf for standard GEMM | rarely | usually |
| Hopper-specific features (WGMMA, TMA) | partial | full |
| FlashAttention-3-style warp specialization | possible but harder | the natural fit |
| Standard square BF16 matmul | competitive | slightly faster |

The mature production answer is *both*. Triton for the long tail of custom kernels; CUTLASS (or cuBLAS, which is built on it) for standard GEMMs and bleeding-edge architectural features.

### 7. The reality on H100

You are on Hopper, so the Hopper-specific machinery is exactly what you should be learning — it's what production GEMM and attention kernels actually use. The kernels you write will use:

- **TMA** (Tensor Memory Accelerator) for async, multi-dimensional HBM↔SMEM tile loads, instead of Ampere's `cp.async`.
- **WGMMA** — warp-group (128-thread) async MMAs (`m64nNk16` for BF16), instead of Ampere's 16×16×16 warp MMA.
- **Warp specialization** — dedicated producer (TMA) and consumer (WGMMA) warp groups, coordinated by `mbarrier`. This is the structure behind FlashAttention 3.
- **FP8 tensor cores** (E4M3/E5M2) for ~2× the BF16 throughput, with the `cutlass::gemm::collective` builder picking the FP8 kernel.

Build for `sm_90a` — the `a` suffix unlocks the Hopper-only WGMMA/TMA instructions. The CuTe DSL exposes all of this from Python.

---

## Hands-on

### Exercise 1 — Install CUTLASS and explore

Clone `NVIDIA/cutlass` next to this repo. Build the examples:

```bash
cd cutlass
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=90a   # Hopper (the 'a' unlocks WGMMA/TMA)
make -j install_unit_tests
```

Run `tools/profiler/cutlass_profiler` to compare CUTLASS GEMM kernels against your `torch.matmul` baseline. Record the best CUTLASS config for a `(8192, 8192, 8192)` BF16 GEMM.

**Deliverable:** TFLOPS achieved by best CUTLASS GEMM vs `torch.matmul`. Note the template parameters of the winning config.

### Exercise 2 — Install CuTe DSL and `Hello, layout`

Follow the CuTe DSL install instructions. Write a Python script that constructs a few `Layout`s, prints them, and verifies you understand their semantics:

```python
from cute import Layout, make_shape, make_stride
L = Layout(make_shape(8, 4), make_stride(4, 1))
print(L)                          # Shape and stride
print(L(0, 0), L(1, 0), L(0, 1)) # offsets
```

Then practice `composition` and `complement` from the docs.

**Deliverable:** A short Python file demonstrating ~5 layout operations with comments explaining what each one means.

### Exercise 3 — Naive GEMM in CuTe DSL

Write a single-pipeline-stage `(M, N, K) = (256, 256, 256)` BF16 GEMM in CuTe DSL. Don't worry about performance — just get the right answer:

```python
@cute.kernel
def gemm_naive(A, B, C):
    # one threadblock, one tile
    ...
```

Compare result against `A @ B` from PyTorch.

**Deliverable:** Numerical correctness within 1e-2 relative. Time it — naive will be slow. Note how slow it is.

### Exercise 4 — Tiled, pipelined GEMM

Following the CuTe DSL tutorials, build up to a full multi-stage, warp-cooperative GEMM. Tune `(BLOCK_M, BLOCK_N, BLOCK_K, NumStages)`.

Benchmark at `(8192, 8192, 8192)` against `torch.matmul`. You should get within 70–90% of cuBLAS — beating cuBLAS at standard shapes is *hard*, but matching closely is achievable.

**Deliverable:** A speed comparison table at three shapes (square, tall-skinny, wide).

### Exercise 5 — Compare with your Triton matmul from Chapter 10

For the same three shapes, compare:

- `torch.matmul` (cuBLAS).
- Your Triton matmul.
- Your CuTe DSL matmul.

Plot all three. Discuss where each wins.

**Deliverable:** Plot + 1-paragraph reading.

### Exercise 6 — Read FlashAttention 3 source

FlashAttention 3 is built on CUTLASS / CuTe with extensive warp specialization for Hopper. On your H100 you can both *build/run* it and read the source. Build the `hopper/` target of `Dao-AILab/flash-attention`, then read for an hour. Identify:

- The producer/consumer split.
- Where the online softmax lives.
- Where `cp.async`/TMA loads happen.
- The high-level pipeline.

You won't understand everything. Write down what you *do* understand.

**Deliverable:** A 1-page writeup in `NOTES.md`.

### Exercise 7 — Stretch: epilogue fusion

Write a CuTe DSL GEMM with a fused epilogue: `D = activation(alpha * A @ B + bias)`. The epilogue is where you fold in element-wise ops and reductions without re-launching kernels. Production GEMM kernels have rich epilogues; CUTLASS exposes them as templates.

Compare to `(A @ B + bias).relu()` in eager PyTorch.

**Deliverable:** Speedup. Note the kernel-count drop in nsys.

---

## Self-check

1. What are the three levels of tile hierarchy in a CUTLASS GEMM?
2. What does `cp.async` do, and why does it enable multi-stage pipelines?
3. What is a CuTe `Layout`? Describe `Layout< Shape <_4, _4>, Stride <_1, _4> >` element-wise.
4. Why do modern GEMM kernels use warp specialization?
5. When would you choose CUTLASS over Triton? Triton over CUTLASS?
6. What does "epilogue" mean in CUTLASS, and why does it exist as a first-class concept?

---

## Deliverables before moving on

- CUTLASS profiler result for `(8192, 8192, 8192)` BF16 vs `torch.matmul`.
- A small layout-algebra Python script.
- A naive CuTe DSL GEMM matches `torch.matmul` numerically.
- A tuned CuTe DSL GEMM within 70%+ of cuBLAS for standard shapes.
- Triton vs CuTe DSL vs cuBLAS comparison plot.
- FA3 source reading writeup.

After this, you can read any CUTLASS-based codebase (FA, vLLM kernels, Megatron's GEMM-fused ops) without getting lost.
