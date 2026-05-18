# 04 — PTX and SASS

When an article cites an instruction by name — `wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16.f32`, or `cp.async.bulk.tensor.2d.shared::cta.global`, or `tcgen05.mma.*` — that's PTX. To read those articles fluently you need to know what PTX *is*, how to look at it, and how it relates to SASS (the actual machine code).

You don't need to write PTX. You need to be able to *read* PTX snippets in articles and recognize what they're doing.

---

## What PTX is

**PTX (Parallel Thread eXecution)** is NVIDIA's portable virtual ISA for GPUs. Conceptually it sits between your CUDA C++ / Triton and the actual hardware:

```
CUDA C++  /  Triton  /  CUTLASS  /  cuTe DSL
       │
       ▼          (nvcc / triton / cutlass compiler)
PTX  ── portable across GPU generations (mostly)
       │
       ▼          (ptxas, driver JIT)
SASS  ── actual hardware machine code, specific to compute capability
       │
       ▼
GPU executes
```

PTX has a real spec, the *PTX ISA documentation* (a several-hundred-page PDF NVIDIA publishes per CUDA release). When someone references an instruction in a paper or article, they're naming it from this spec.

### PTX is not stable forever

A new generation often introduces a *new instruction family*. Things like `cp.async`, `mbarrier`, `cp.async.bulk.tensor` (TMA), `wgmma`, `tcgen05` — each landed in a specific PTX version tied to a specific compute capability:

```
sm_70  Volta       — mma.sync (first MMA)
sm_75  Turing      — INT8/INT4 MMA, more shapes
sm_80  Ampere      — cp.async, structured sparsity, BF16, TF32
sm_90  Hopper      — wgmma, cp.async.bulk.tensor (TMA), mbarrier, cluster
sm_100 Blackwell   — tcgen05.*, MX/NVFP4 MMA shapes
```

You target a compute capability via `-arch=sm_90a` (the `a` suffix is for Hopper-and-later features). A binary built for `sm_90` won't run on `sm_80`.

### A PTX example

A simple `add` kernel becomes something like:

```ptx
.visible .entry add_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
)
{
    .reg .pred    %p<2>;
    .reg .b32     %r<8>;
    .reg .b64     %rd<6>;

    ld.param.u64  %rd1, [a_ptr];
    ld.param.u64  %rd2, [b_ptr];
    ld.param.u64  %rd3, [out_ptr];
    ld.param.u32  %r1,  [n];

    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.s32    %r5, %r2, %r3, %r4;

    setp.ge.s32   %p1, %r5, %r1;
    @%p1 bra      L_END;

    ld.global.f32 %r6, [%rd4 + %r5*4];
    ld.global.f32 %r7, [%rd5 + %r5*4];
    add.f32       %r6, %r6, %r7;
    st.global.f32 [%rd3 + %r5*4], %r6;

L_END:
    ret;
}
```

You can usually puzzle out PTX with a bit of patience. Useful conventions:

- `%r<N>` = N register names. `.b32` = 32-bit untyped, `.u32` / `.s32` / `.f32` are typed.
- `ld.global.f32` = load FP32 from global memory.
- `st.global.f32` = store.
- `mad.lo.s32 d, a, b, c` = `d = lower32(a*b) + c` (multiply-add).
- `setp.ge.s32` = set predicate if greater-or-equal.
- `@%p1 bra L_END` = if predicate, branch.
- `%tid.x`, `%ntid.x`, `%ctaid.x` = thread/block IDs.

The interesting ones for LLM articles:

- `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3}` — an Ampere BF16 MMA. The shape suffix tells you what matrix sizes.
- `wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 ...` — a Hopper warp-group MMA.
- `cp.async.ca.shared.global [smem], [gmem], 4` — Ampere async copy.
- `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes ...` — Hopper TMA delivering to cluster shared.

### What SASS is

**SASS (Streaming Assembler)** is the actual machine code. SASS is undocumented in the same way PTX is — NVIDIA tells you the names but doesn't publish a full ISA reference. It is generation-specific, so the same PTX `mma` may lower to different SASS on Ampere vs Hopper.

You inspect SASS with the `cuobjdump` tool that ships with CUDA:

```bash
cuobjdump --dump-sass my_kernel.cubin
```

SASS is what you read when you really need to know what the hardware is doing — register usage, instruction count, predication. For most reading you stay at PTX.

---

## How to look at PTX for your own kernels

### From CUDA C++ (`nvcc`)

```bash
nvcc -ptx -arch=sm_80 my_kernel.cu -o my_kernel.ptx
# or compile and inspect:
nvcc -arch=sm_80 my_kernel.cu -o my_kernel
cuobjdump --dump-ptx my_kernel
```

### From PyTorch (compiled / Triton)

```python
import torch

@torch.compile
def f(x):
    return (x + 1).relu()

# After running f at least once, the inductor-generated kernels exist:
torch._inductor.compile_fx.compile_fx_inner  # has a debug option
# or set env vars:
#   TORCH_LOGS="inductor,output_code"
#   TORCH_COMPILE_DEBUG=1
```

The `TORCH_LOGS` and `TORCH_COMPILE_DEBUG=1` environment variables make PyTorch dump Triton source + the lowered PTX for every fused kernel.

### From Triton directly

```python
import triton

@triton.jit
def add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    ...

# Compile and inspect:
ptx = add.warmup(...).asm["ptx"]
print(ptx)
# SASS:
print(add.warmup(...).asm["sass"])
```

This is the *fastest* way to see what tensor-core instructions your Triton kernel emits.

### From CUTLASS

CUTLASS templates generate kernels at compile time; `cuobjdump` after build gives you PTX/SASS.

---

## A worked reading

Suppose an article says:

> The kernel uses `cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes`-style TMA loads delivering A and B tiles to the consumer cluster, with `wgmma.mma_async.sync.aligned.m64n256k32.f32.f8e4m3.f8e4m3.f32` running on each loaded tile.

Decoding:

- **`cp.async.bulk.tensor`** — TMA bulk copy. Multi-dim tile move.
- **`.5d`** — 5-dimensional tile descriptor (for higher-rank tensors).
- **`.shared::cluster`** — destination is cluster shared memory (DSMEM).
- **`.global`** — source is HBM.
- **`.mbarrier::complete_tx::bytes`** — completion notifies an mbarrier when N bytes have arrived.
- **`wgmma.mma_async`** — warp-group MMA, asynchronous issue.
- **`.m64n256k32`** — instruction shape: 64×256 output, K=32.
- **`.f32.f8e4m3.f8e4m3.f32`** — output FP32, inputs both FP8 E4M3, accumulator FP32.

So: "TMA loads FP8 tiles into cluster shared memory; warp-group consumer warps run FP8 GEMM that accumulates in FP32." A perfectly reasonable Hopper FP8 GEMM.

After enough of these, the syntax becomes natural.

---

## Common modifiers cheat sheet

```
.sync           explicitly synchronous within the warp (almost always present)
.aligned        operands are aligned in memory
.async          asynchronous issue, requires completion fence
.acquire / .release    memory-order semantics (relaxed/acquire/release/sc)
.shared::cta    shared memory of this block
.shared::cluster  shared memory of this cluster (DSMEM)
.global         global memory (HBM)
.const          constant memory
.local          per-thread local memory (often spilled registers — bad)
.bf16 / .f16 / .f8e4m3 / .f8e5m2  dtype suffixes
.row / .col     matrix layout for MMA operands
```

---

## When you'd actually write PTX

Most production code doesn't. The cases where you do:

- **CUTLASS internals** — when you want to use a specific new instruction the C++ template system doesn't yet expose.
- **Triton internals** — the Triton compiler emits PTX; sometimes contributors add new instruction lowerings.
- **Numerical experiments** — when you want a specific rounding mode that CUDA C++ doesn't surface.
- **Performance investigations** — when SASS shows a register spill and you need to trace it back.

For reading articles, recognizing the instructions is enough. You don't need to write them.

---

## What to read next

- **PTX ISA documentation, current version.** Available on NVIDIA's developer site as a PDF. Skim the table of contents; bookmark the sections on `mma`, `wgmma`, `tcgen05`, `cp.async.bulk.tensor`, `mbarrier`.
- **CUDA Binary Utilities documentation** — `cuobjdump`, `nvdisasm` references.
- **Lei Mao's PTX/SASS posts** — practical walkthroughs.
- **The CUTLASS source for any of `cute/atom/mma_*.hpp`** — to see how it picks the right PTX instructions.
- Move on to `05-interconnect.md`.

---

## Self-check

1. What's the difference between PTX and SASS?
2. Why does `sm_90` vs `sm_90a` matter?
3. What does `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes` do, broken into parts?
4. What does `.row.col.f32.bf16.bf16.f32` mean as a modifier suffix to `mma.sync`?
5. How would you dump the PTX a Triton kernel compiles to?
6. Why would you read SASS rather than PTX?
