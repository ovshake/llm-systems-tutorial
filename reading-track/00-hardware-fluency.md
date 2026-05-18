# 00 — Hardware Fluency: The Core Mental Model

Before any other reading-track topic, you need the vocabulary every hardware article assumes. This file is the floor. If you've gone through Chapter 09 you'll know most of this; the parts new here are the *generation-by-generation* numbers and the recent vocabulary (cluster, distributed shared memory, tensor memory).

---

## The execution hierarchy

```
GPU
└── Streaming Multiprocessor (SM)    × many (A100: 108, H100: 132, B200: ~148)
    └── (Optional) CTA cluster        × few SMs (Hopper+ only)
        └── Cooperative Thread Array (CTA, == "block")
            └── Warp                  × 32 threads, lockstep SIMT
                └── Thread            × 1 thread
```

Important: SMs do *not* share work automatically. The hardware scheduler assigns *blocks* to SMs at launch. Threads within a block share the SM's shared memory and can sync; threads in different blocks cannot, *unless* they're in the same Hopper-and-later **cluster** (more on this below).

Numbers to memorize for current-generation hardware:

| | V100 | A100 | H100 | B200 |
|---|---|---|---|---|
| Process | 12nm | 7nm | 4N (TSMC custom) | 4NP (TSMC) |
| SMs per GPU | 80 | 108 | 132 | ~148 (per die; B200 is dual-die) |
| FP32 peak (TFLOPS) | 15.7 | 19.5 | 67 | ~80 (per die) |
| BF16 / FP16 tensor (TFLOPS, dense) | 125 | 312 | 989 | ~2250 |
| FP8 tensor (TFLOPS, dense) | — | — | 1979 | ~4500 |
| FP4 tensor (TFLOPS, dense) | — | — | — | ~9000 |
| HBM capacity | 32 GB | 80 GB | 80 GB | 192 GB |
| HBM bandwidth | 0.9 TB/s | 2.0 TB/s | 3.35 TB/s | ~8 TB/s |
| NVLink BW per GPU | 300 GB/s | 600 GB/s | 900 GB/s | 1.8 TB/s |

(B200 numbers are the SemiAnalysis-quoted values circa launch; treat them as approximate.)

These numbers travel in pairs: a generation grew tensor FLOPs ~3× *and* HBM bandwidth ~2×; the ratio shifted compute-heavy. Any hardware article you read is making a claim about how a new generation rebalances these.

---

## The on-chip memory hierarchy

From smallest/fastest to largest/slowest:

| Memory space | A100 size | Latency (cycles) | Visibility |
|---|---|---|---|
| Registers (per thread) | ~255 × 4 B = ~1 KB | 1 | per thread |
| Shared memory / L1 | 192 KB per SM | ~20 | per CTA (block) |
| L2 cache | 40 MB whole GPU | ~200 | all SMs |
| HBM (DRAM) | 80 GB | ~400 | all SMs |

H100 grew shared+L1 to 228 KB/SM and L2 to 50 MB. B200 enlarged both further and *added a new memory space called Tensor Memory* — see `02-memory-spaces.md`.

Three rules of thumb:

1. **Each level is 10–100× more bandwidth than the next.** That's why kernel optimization is *mostly* about moving data up the hierarchy and reusing it.
2. **Compute-bound vs bandwidth-bound** (Chapter 09) is a hierarchy-level question: a kernel is bound by the slowest memory level it depends on.
3. **You don't control these explicitly in PyTorch or Triton.** You influence them via tile shapes (which fit in registers/SRAM) and access patterns. CUTLASS lets you control them somewhat more explicitly via layouts.

---

## SIMT, warps, and divergence

Threads in a warp execute the *same* instruction at the same time. If you have a conditional like:

```cuda
if (tid % 2 == 0) {
    a = expensive_op();
} else {
    b = different_op();
}
```

…the warp executes *both* branches serially, masking off threads that don't take that path. That's "warp divergence." When you write Triton kernels, the compiler hides this from you, but the cost is still there in the generated PTX.

Modern GPUs (Volta+) have *independent thread scheduling* — threads in a warp can be at different program counters if necessary. This makes divergence less catastrophic than on pre-Volta GPUs, but it's still measurable.

---

## Tensor cores: what they actually do

A tensor core is a fused-multiply-add unit that takes an MMA (matrix-multiply-accumulate) instruction:

```
D = A * B + C
```

where A, B, C, D are small matrices (e.g., 16×16). One instruction issues one of these per cycle per warp. Generations differ in:

- **MMA shape:** Volta started at 4×4×4; Ampere did 16×16×16; Hopper has WGMMA (warp-group MMA) shapes like 64×N×16; Blackwell's TCGen05 can do larger and more flexible shapes.
- **Input dtypes:** FP16/BF16 (A100), FP8 (H100), FP4 (B200), block-scaled (B200).
- **Sparsity:** A100+ supports 2:4 structured sparsity (skip half the multiplies if the weight matrix has that pattern). Mostly used for inference quantization.
- **Async issue:** Hopper introduced async MMA where the issue and the data movement can overlap. This is the genesis of *warp specialization*.

`01-tensor-cores.md` does this generation-by-generation.

---

## CTA clusters and distributed shared memory (Hopper+)

A new concept from Hopper that you'll see in any article newer than 2023:

A **CTA cluster** is a group of CTAs (blocks) — typically 2 to 16 — that:

- Are guaranteed to be scheduled on **physically adjacent SMs** within the same GPC (Graphics Processing Cluster).
- Can access each other's **shared memory** via a new memory space called **distributed shared memory (DSMEM)**.
- Can synchronize via cluster-level barriers.

Why? Because for very large tiles (e.g., the M=128, N=256 tiles in a Hopper GEMM), one SM's shared memory isn't enough. Clustering lets you treat 4 SMs' shared memory as one logical block of ~1 MB.

You enable a cluster with `__cluster_dims__(...)` in CUDA C++ or its equivalent in PTX. Triton 3+ has Python-level support.

---

## Async memory copies and TMA (Hopper+)

Pre-Ampere, copying from HBM to shared memory used a load instruction that bounces through registers:
```
ld.global.f32 r0, [addr];   // load to register
st.shared.f32 [smem], r0;   // store to shared
```

Ampere added `cp.async` — copy directly from global to shared, bypassing registers:
```
cp.async.ca.shared.global [smem], [addr], 4;
```

Hopper added the **Tensor Memory Accelerator (TMA)** — a hardware unit that issues *strided multi-dimensional* loads and stores via one instruction. The CPU-equivalent intuition: DMA. The kernel issues `tma_load_tile(src, dst, shape, stride)`, and the TMA engine handles the transfer asynchronously while the kernel does math.

This is the building block for the producer/consumer warp-specialization pattern: some warps issue TMA loads, others run MMAs. They sync via `mbarrier` (memory barriers).

---

## Tensor Memory (TMEM) — Blackwell's new space

Blackwell added an *additional* on-chip memory space called **Tensor Memory** (TMEM). It sits between the register file and shared memory in size and is dedicated to *tensor core operands and accumulators*.

The original problem TMEM solves: on Hopper, MMA accumulators have to live in registers, and there aren't enough registers per warp for the giant accumulator tiles modern transformers want. TMEM gives the tensor cores their own backing storage that's bigger than the register file but faster than shared memory.

The first SemiAnalysis case study (`case-studies/blackwell-tensor-memory.md`) walks through this in detail.

---

## How to read a hardware metric

When an article says something like "H100 has 67 TFLOPS of FP32," ask:

- **Dense or sparse?** (2× difference for tensor core operations with 2:4 sparsity.)
- **Tensor core or CUDA core?** (Tensor cores can be 10–30× higher.)
- **Boost clock or base?** (Boost is best-case; real workloads are 10–20% lower.)
- **Per-die or per-package?** (B200 has 2 dies; some specs are per-die.)
- **Sustained or peak?** (Peak assumes 100% utilization, which is unreachable in practice.)

A common gotcha: comparing "FP4 peak" of generation N+1 to "FP16 peak" of generation N gives a 30× number that's almost meaningless because the formats aren't comparable for the same workload.

---

## What to read next

- **NVIDIA A100 / H100 / Blackwell whitepapers.** All three are free PDFs from NVIDIA. They are *the* canonical reference and you should have at least skimmed each.
- **Hot Chips talks.** NVIDIA presents at Hot Chips every year; the slides and videos are public. The H100 Hot Chips talk especially is a goldmine.
- **NVIDIA CUDA C++ Programming Guide, "Hardware Implementation" chapter.** Authoritative on the execution model.
- **GPU Technology Conference (GTC) keynotes.** Where new generations are announced. Skim the technical sessions for the depth.
- Then move on to `01-tensor-cores.md`.

---

## Self-check

1. How many SMs does an A100 have? An H100?
2. Why is shared memory ~10× faster than L2?
3. What's the difference between a warp and a CTA?
4. What does a CTA cluster get you that plain blocks don't?
5. What's `cp.async`, and what does TMA add on top?
6. Why is "FP4 peak" not directly comparable to "FP16 peak"?
