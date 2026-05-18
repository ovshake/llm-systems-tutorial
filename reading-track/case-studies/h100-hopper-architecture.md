# Case Study — Hopper / H100 Architecture

> Sources: NVIDIA H100 whitepaper; Hot Chips 2022 talk; SemiAnalysis Hopper deep dive.

Less of a single article than a family of canonical sources for the Hopper generation. This case study is shorter than the Blackwell one — Hopper is "your toolkit's reference point" rather than something new to decode.

---

## Prerequisites

- `../00-hardware-fluency.md`
- `../01-tensor-cores.md` (the WGMMA section especially)
- `../02-memory-spaces.md` (TMA, DSMEM)

---

## What Hopper actually added

Read in this order:

1. **WGMMA — warp-group MMA.** The instruction-issue unit became 4 warps (128 threads) rather than 1 warp (32 threads). Bigger MMA shapes; asynchronous issue. Section in `01-tensor-cores.md`.
2. **TMA — Tensor Memory Accelerator.** Hardware DMA for tile copies, programmable from PTX. Section in `02-memory-spaces.md`.
3. **DSMEM and clusters.** A new scheduling layer above blocks; blocks in a cluster share memory and synchronize. Section in `02-memory-spaces.md`.
4. **FP8 tensor cores.** Two formats (E4M3, E5M2). Per-tensor scaling automated by the Transformer Engine library. Section in `../03-numeric-formats.md`.
5. **More HBM (80 GB), more L2 (50 MB), more SMEM (228 KB).**
6. **NVLink 4.0 — 900 GB/s per GPU.** Section in `../05-interconnect.md`.

These five additions together enabled the kernel patterns you'll see in:

- FlashAttention 3 (warp-specialized producer/consumer with TMA + WGMMA).
- Recent CUTLASS GEMM templates.
- Modern vLLM / SGLang attention kernels.

---

## The architectural argument

Hopper's bet was on **asynchrony** and **larger units of work**:

- Async tensor cores → producer/consumer overlap → higher utilization.
- Larger warp-group MMAs → less per-instruction overhead → bigger natural tile sizes.
- TMA → free your warps from address arithmetic for big loads.
- DSMEM → bigger logical scratchpad than one SM can supply alone.

The downside: kernel complexity jumped substantially. A well-tuned Hopper GEMM kernel is *much* harder to write than its Ampere predecessor. This is part of why CUTLASS and Triton both invested heavily in compiler infrastructure during the Hopper era.

---

## What to read

- **NVIDIA H100 whitepaper.** 70-page PDF. Required reading for anyone targeting H100. Pay special attention to the "Streaming Multiprocessor" and "Asynchronous Execution" sections.
- **Hot Chips 2022 H100 talk.** ~30 min video + slides. Great for the "why we did this" framing.
- **"Programming the NVIDIA Hopper TMA"** — NVIDIA developer blog post. Walks through the PTX.
- **FlashAttention 3 paper.** Production WGMMA + TMA usage, with measured numbers.
- **CUTLASS 3.x Hopper template tour** — `cute/atom/mma_atom.hpp` and friends. See how CUTLASS exposes the new instructions.
- **SemiAnalysis Hopper deep-dive articles** — for the analyst-interpretation layer.

---

## Self-check

1. What's the issue unit for a WGMMA, in threads?
2. What does TMA replace, and what's the asynchronous semantics?
3. What's a "cluster" in Hopper, and how is it different from a block?
4. Why are FP8 tensor cores on H100 only useful with a scaling library?
5. Where does the producer/consumer pattern live in a Hopper kernel?
