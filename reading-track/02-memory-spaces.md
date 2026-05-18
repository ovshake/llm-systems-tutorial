# 02 — Memory Spaces

GPUs have *more* memory spaces than CPUs, with explicit programmer control over which space data lives in. The article you're reading will refer to them by name (shared, L1, L2, HBM, DSMEM, TMEM…). This file is the reference.

---

## The classical spaces

### Registers

- *Scope:* one thread.
- *Size:* ~256 × 32 bits per thread on A100; 32-bit fixed-size. With 1024 threads/block, that's ~1 MB per block of register state.
- *Latency:* effectively zero (1 cycle).
- *Allocated:* by the compiler. You influence this by writing more or less "register-pressure"-heavy code.

**Register spilling.** If your kernel needs more registers per thread than the SM can give, the compiler spills to *local memory*, which is actually backed by L1/L2/global. This is catastrophic for performance. NCU shows you the spill amount; aim for zero.

### Shared memory / L1 cache

- *Scope:* one CTA (block).
- *Size:* 192 KB/SM on A100, 228 KB/SM on H100, ~256 KB on B200. Split between programmer-controlled shared memory and L1.
- *Latency:* ~20 cycles.
- *Bank conflicts:* shared memory is divided into 32 banks. If 32 threads in a warp access 32 *different* banks, full speed. If two threads hit the same bank, the accesses serialize.

Shared memory is your fast scratchpad. CUTLASS, FlashAttention, and most production GEMMs stage tiles into shared memory before issuing MMAs.

### L2 cache

- *Scope:* all SMs (whole GPU).
- *Size:* 40 MB on A100, 50 MB on H100, ~60 MB on B200 (per die).
- *Latency:* ~200 cycles.
- *Behavior:* fully associative. NVIDIA exposes a "persisting L2" hint (Ampere+) for data you want to keep across multiple kernel launches.

L2 is the cache that hides HBM latency. The 40+ MB on a modern GPU is enough to hold a meaningful working set; clever kernels deliberately fit their hot data inside it.

### HBM (DRAM)

- *Scope:* all SMs.
- *Size:* 80 GB (A100/H100), 192 GB (B200).
- *Bandwidth:* 2.0 TB/s (A100), 3.35 TB/s (H100), ~8 TB/s (B200).
- *Latency:* ~400 cycles.

HBM is the only "main memory" the GPU has. Everything else is cache or scratchpad. The "memory bandwidth" you see quoted is HBM bandwidth.

### Constant memory

- *Scope:* read-only, all threads.
- *Size:* small (64 KB).
- *Use:* compiler stashes literal constants here.

Not something you manage explicitly in modern code.

---

## The modern additions

These are what distinguish a 2024 article from a 2019 one.

### Distributed Shared Memory (DSMEM) — Hopper+

When several blocks belong to the same **CTA cluster**, each block can read and write the *shared memory* of the others. This is "distributed shared memory."

```
Block A's SMEM ──┐
Block B's SMEM ──┼── all visible to Blocks A, B, C, D
Block C's SMEM ──┤   if A,B,C,D are in the same cluster
Block D's SMEM ──┘
```

Practically: with a cluster of 4 and 228 KB SMEM/SM, you have ~912 KB of logical "shared" memory available to the cluster.

PTX surface: a special generic pointer type and the `multimem` instructions. Triton 3+ exposes this via cluster primitives.

Why this exists: large GEMM tiles want a lot of SMEM, and one SM's SMEM isn't enough. Rather than spilling to L2 (slow), cluster across SMs.

### Tensor Memory Accelerator (TMA) — Hopper+

TMA is *not* a memory space — it's an **engine** that moves data between memory spaces. But understanding it requires the memory-space context, so it lives in this file.

Pre-Hopper, you copied data with normal load/store instructions:
- `ld.global` → register
- `st.shared` ← register

Hopper added TMA, a dedicated DMA-like engine that:
- Takes a multi-dim *tile descriptor* (shape, stride, dtype, layout).
- Issues *one* PTX instruction (`cp.async.bulk.tensor.*`).
- Asynchronously moves the tile between HBM and shared memory.
- Signals completion via an `mbarrier`.

The kernel can issue the TMA, then do math, then wait. This is the asynchrony that enables warp specialization.

TMA also supports the **bulk-cluster** mode that delivers to DSMEM rather than a single block's SMEM. Used in big cluster GEMMs.

### Tensor Memory (TMEM) — Blackwell

A genuinely *new memory space*, not just a new engine. On Blackwell SMs:

```
Registers (~256 KB/SM)
        │
        │   (used to feed compute)
        ▼
SMEM / L1 (~256 KB/SM)
        │
        │   (programmer-controlled scratchpad)
        ▼
TMEM (a few hundred KB/SM)        ◄── NEW
        │
        │   (dedicated to tensor core operands and accumulators)
        ▼
L2 (~60 MB/die)
        │
        ▼
HBM
```

Key properties:

- **Sized in the hundreds of KB per SM** (NVIDIA hasn't published exact specs at the time of writing; SemiAnalysis estimates a particular size; treat the article's numbers as best-known).
- **Read directly by TCGen05 MMA instructions.** The MMA operand doesn't pass through registers.
- **Has its own loads from SMEM/HBM** via TMA-like instructions.
- **The accumulator can live entirely in TMEM** — no register pressure.

Why this was needed: Hopper WGMMA accumulators lived in registers. As MMA shapes grew toward `m64n256k16` and beyond, accumulators consumed an enormous fraction of the register file. TMEM gives accumulators their own home.

The Blackwell Tensor Memory article (case study) digs much deeper. Read this file first to have the vocabulary, then read the case study.

---

## Memory ↔ tensor-core data flow, by generation

```
A100:    HBM → cp.async → SMEM → load → registers → mma → registers
H100:    HBM → TMA → SMEM → wgmma reads → registers (accumulator) → wgmma writes → registers
B200:    HBM → TMA → SMEM → tmem load → TMEM → tcgen05 reads → TMEM (accumulator) → tcgen05 writes → TMEM
```

Each new generation moves the tensor cores' operand source one step further from registers. That's the structural story.

---

## Sizing intuition

For a Hopper BF16 GEMM with `(M, N, K) = (8192, 8192, 8192)`:

- Threadblock tile: 128×256, K-tile 64.
- One A-tile: 128 × 64 × 2 B = 16 KB.
- One B-tile: 64 × 256 × 2 B = 32 KB.
- Plus 4 pipeline stages → ~200 KB SMEM used.

That's why H100's 228 KB SMEM was exactly the right size for the WGMMA shapes Hopper introduced. Hardware/software co-design.

For Blackwell, the threadblock tiles grow further (especially in FP4), and the new memory hierarchy keeps up.

---

## What this means for software

For the next few years, most of these will *not* be exposed in PyTorch or even Triton (3.x), though they're being added. They live at the PTX / CUTLASS / cuTe level.

If you want to write Hopper-or-Blackwell-aware kernels:

- CUTLASS / cuTe DSL exposes TMA, WGMMA, and TCGen05 directly.
- Triton 3.x has partial support: `tl.async_copy`, cluster primitives, etc.
- Writing PTX directly is rare but possible (CUTLASS internals do this).

For the article reader: when you see "the kernel uses TMA + WGMMA + warp specialization with `mbarrier`" you should know this is a Hopper kernel, and the key components live in this file.

---

## What to read next

- **NVIDIA H100 whitepaper** — section on TMA and DSMEM.
- **NVIDIA Blackwell whitepaper** — TMEM discussion.
- **NVIDIA blog post: "Programming the Hopper TMA"** — practical walkthrough.
- **PTX ISA documentation:** `cp.async.bulk.tensor.*` (TMA), `mbarrier.*`, `tcgen05.*`.
- **CUTLASS source:** `cutlass/include/cute/atom/copy_atom.hpp` — abstract copy operations.
- Move on to `03-numeric-formats.md`.

---

## Self-check

1. What's the difference between *shared memory* and *L1*?
2. What is a "bank conflict"?
3. What's DSMEM and what problem does it solve?
4. What is TMA, and is it a memory space or an engine?
5. What's TMEM, and why was it added?
6. Why did Hopper need a 228 KB SMEM and not the A100's 192 KB?
