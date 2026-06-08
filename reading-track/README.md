# Reading Track — Hardware Fluency for the GPU Press

A parallel learning track that lives next to the main course. The chapters in `chapters/` teach you how to *use* and *contribute to* the GPU software stack. This track teaches you how to *read* the GPU hardware press — SemiAnalysis deep dives, NVIDIA whitepapers, Hot Chips talks, hyperscaler architecture announcements — without bouncing off the jargon.

If you've ever opened a SemiAnalysis article like ["Dissecting NVIDIA Blackwell Tensor Memory"](https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor) and stopped at "TMEM", "TCGen05", "block-scaled MXFP8", or "DSMEM cluster", this track is the prerequisite.

---

## How to use this track

- **Read alongside the main chapters.** You don't have to finish `chapters/` before starting here. A natural pairing: read one reading-track topic per week while you work through the main chapters.
- **The chapter sequence is *suggested*, not required.** Each file stands alone; cross-references link to deeper detail when needed.
- **Every topic ends with a "What to read next" list of authoritative sources** — NVIDIA whitepapers, Hot Chips talks, primary papers. The reading-track file gives you the vocabulary; the linked sources give you the depth.
- **Case studies** at the end walk through specific high-profile articles (Blackwell Tensor Memory, Hopper architecture, MI300 vs H100) decoding terms inline. After three case studies you should be able to read new ones cold.

---

## Topic map

| # | File | What you'll learn |
|---|---|---|
| 00 | [`00-hardware-fluency.md`](00-hardware-fluency.md) | The core mental model: SM, warp, block, grid; HBM, L2, L1/shared; bandwidth and latency numbers across generations. The vocabulary every other file assumes. |
| 01 | [`01-tensor-cores.md`](01-tensor-cores.md) | Tensor core evolution: Volta MMA → Ampere MMA → Hopper WGMMA → Blackwell TCGen05. What each generation added and why it matters. |
| 02 | [`02-memory-spaces.md`](02-memory-spaces.md) | All the on-chip memory spaces: register file, shared memory / L1, L2, HBM. The modern additions: TMA (Hopper), DSMEM (cluster shared memory, Hopper), TMEM (Tensor Memory, Blackwell). |
| 03 | [`03-numeric-formats.md`](03-numeric-formats.md) | FP32 / TF32 / BF16 / FP16 / FP8 (E4M3, E5M2) / block-scaled MXFP8 / MXFP6 / MXFP4 / NVFP4. When each is used and what they cost. |
| 04 | [`04-ptx-and-sass.md`](04-ptx-and-sass.md) | How to read low-level GPU code: PTX (the portable IR), SASS (the actual assembly). How to dump and inspect what a kernel becomes. |
| 05 | [`05-interconnect.md`](05-interconnect.md) | NVLink generations, NVSwitch, the scale-up/scale-out distinction, InfiniBand, Ethernet alternatives, rail-optimized vs fat-tree topologies. |
| 06 | [`06-economics.md`](06-economics.md) | TCO, $/GPU-hour, what drives "cost per token", how to read capacity-planning numbers in the press. |
| 07 | [`07-roadmap-reading.md`](07-roadmap-reading.md) | How to read NVIDIA whitepapers and decode marketing-speak. The vocabulary of "sparsity", "structured sparsity", "transformer engine", "MIG". How to follow the roadmap. |

Case studies in [`case-studies/`](case-studies/):

| File | Article it decodes |
|---|---|
| [`blackwell-tensor-memory.md`](case-studies/blackwell-tensor-memory.md) | SemiAnalysis, "Dissecting NVIDIA Blackwell Tensor Memory" |
| [`h100-hopper-architecture.md`](case-studies/h100-hopper-architecture.md) | NVIDIA H100 whitepaper; SemiAnalysis Hopper deep dive |
| [`mi300-vs-h100.md`](case-studies/mi300-vs-h100.md) | AMD MI300 launch coverage and competitive analyses |

---

## A meta-suggestion: how to read a hardware article

When you sit down with a SemiAnalysis-style piece:

1. **First pass (5 min):** skim. Read the section headers, the figures and their captions, the conclusion. *Don't* try to understand technical claims. You're building a map.
2. **Glossary pass:** open a notebook. As you skim, write down every term you can't define in one sentence. *Don't look them up yet.*
3. **Second pass (20 min):** read carefully. For each term on your list, either find the definition in the article or in this track. If it's still opaque, mark it "unresolved" and continue.
4. **Resolution pass:** look up your unresolved terms. The Reading Track is one place; the linked NVIDIA whitepapers and original papers are usually the authoritative source. (Wikipedia is occasionally fine for hardware basics but unreliable for cutting-edge specifics.)
5. **Distill:** one paragraph in your own words on what the article actually claimed and why it matters. If you can't write the paragraph, you didn't understand it yet — go back to step 3.

Most engineers who "can read hardware articles fluently" are doing this fast and silently. You'll do it slow and explicitly until the vocabulary is automatic. After ~15 articles it becomes automatic.

---

## Why this track exists separately from `chapters/`

`chapters/` is about *doing*. Most learners benefit from doing-first because the abstractions only stick once you've debugged a bug they explain. But there is a category of knowledge — microarchitectural evolution, modern instruction families, networking topology — that you mostly learn by *reading*, not by *implementing*. (You won't be implementing a Blackwell tensor core simulator on your 8-GPU node.)

This track is the reading-shaped part of the curriculum.

The two tracks reinforce each other: when Chapter 09 says "Tensor Cores read inputs from shared memory or registers," the reading track explains *which* tensor core generation, what instruction, what shapes; when the reading track says "WGMMA enabled warp specialization on Hopper," Chapter 10's Triton kernels are the closest hands-on you can get on Ampere.

---

## A note on dates

GPU hardware evolves quickly. This material is accurate as of the current course build (2026). The *vocabulary* and *concepts* are durable — SMs, MMA, memory hierarchy, scale-up vs scale-out aren't going away. The *specific generations* (Blackwell, B200, MXFP4) will be superseded. When that happens, the reading habit you build here is the lasting skill.
