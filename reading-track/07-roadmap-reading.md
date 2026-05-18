# 07 — Roadmap Reading: How to Read NVIDIA Whitepapers and Hyperscaler Announcements

The previous six topics gave you the vocabulary. This file teaches you the *meta-skill* of reading NVIDIA's primary sources and the GPT-economy press without getting fooled by marketing or drowning in jargon.

---

## The genre tree

When you sit down to a hardware article, identify which genre it is. The reading style differs:

| Genre | Examples | What to extract |
|---|---|---|
| **Vendor whitepaper** | NVIDIA H100 / Blackwell whitepapers | The *spec*. Numbers, instructions, sizes. Treat these as authoritative for capabilities; check the press for "but does it work?" |
| **Vendor blog post** | NVIDIA developer blog, AMD ROCm blog | The *intended use case*. Marketing-tinted but rich on programmer-facing detail. |
| **Hot Chips / GTC talk** | "Tesla V100 GPU: The Future of HPC" | The *architectural reasoning*. Slides and Q&A often have detail not in the whitepaper. |
| **Academic paper** | FlashAttention, vLLM, MegaBlocks papers | The *technique* + experimental validation. Read with scientific skepticism. |
| **SemiAnalysis / Dylan Patel** | Most articles | The *interpretation*. Often more honest about tradeoffs than vendor sources; sometimes speculative. Read the speculation as speculation. |
| **Mainstream press** | TechCrunch, Verge | Mostly background. Don't expect technical depth. |
| **Earnings call / capex release** | Microsoft, Meta, Google quarterly | The *capital flow*. What's getting built where, with rough numbers. |

A reading habit: most weeks, read one vendor source + one analyst piece on the same topic. The disagreements are where you learn.

---

## How to read a NVIDIA whitepaper

Whitepapers are 30–100 page PDFs that document a new architecture or feature. The good news: they're heavily structured. The bad news: NVIDIA optimizes them for "everyone gets something" — researchers, developers, IT buyers, investors. Each gets confused by the parts written for the others.

The reading order that works:

1. **Table of contents.** Decide which 3–5 sections you actually need. Skip the rest.
2. **The "architecture overview" diagram.** Always one near the front. Pin it to your wall (mentally). Note SMs, memory hierarchy levels, interconnect.
3. **The SM section.** This is where instruction-level changes live. Look for new instruction families.
4. **The memory section.** Bandwidths and sizes. Compare to the previous generation.
5. **The interconnect section.** NVLink generation, NVSwitch / NVL details.
6. **The numeric format section.** What dtypes are supported in tensor cores; new formats.
7. **Performance numbers.** Always quoted at peak. Note dense vs sparse, dtype.
8. **Use cases / SKUs.** Towards the end. Often skim-able.

Time per pass: 30 minutes for a first read, 2 hours for a deep read.

### The marketing-speak you'll encounter

- **"Sparsity": always means 2:4 structured sparsity for inference.**
- **"Transformer Engine": NVIDIA's per-tensor-scale FP8 management library.**
- **"MIG (Multi-Instance GPU)": partition one GPU into sub-GPUs for multi-tenancy.** Real feature, mostly used in cloud.
- **"NVLink C2C": chip-to-chip NVLink, used inside packaged products like Grace-Hopper or Blackwell dual-die.**
- **"DLAs / Tensor Memory": specialized accelerator or memory space.**
- **"Confidential Computing": hardware encryption of memory; matters for some enterprise.**
- **"Spectrum / Quantum": NVIDIA's Ethernet and InfiniBand product lines respectively.**
- **"DGX": NVIDIA's branded server. SuperPOD: their branded cluster reference architecture.**

When in doubt, the whitepaper's first chapter usually defines its acronyms.

---

## How to read a SemiAnalysis-style article

Independent analyst writeups (Dylan Patel's SemiAnalysis, Dwarkesh Patel's interviews, others) have a specific shape:

- A *strong claim* in the intro (often more confident than the data supports).
- A *technical body* that's usually solid on the publicly-known parts and speculative on the embargoed parts.
- A *conclusion / takeaway* that mixes well-supported and ambitious claims.

Reading habits:

- **Separate "this is in the whitepaper" from "this is sourced from industry contacts."** The former is checkable; the latter is plausible but not proven.
- **Watch for benchmark choice.** A 2× speedup at FP4 vs another vendor's BF16 is real, but apples-to-oranges. Make sure you know what's being compared.
- **Skepticism on cost numbers.** Most cost-per-token claims rely on assumptions about utilization that aren't disclosed.
- **Strong on numbers, weak on names.** SemiAnalysis is usually right on TFLOPS, bandwidth, ratios. Less reliable on "this customer bought N of these" rumors.

For the Blackwell Tensor Memory article specifically: it's in their typical genre — vendor specs + analyst interpretation. Read it as "a smart analyst's reading of NVIDIA's disclosures plus some industry sourcing." The case study below decodes it line by line.

---

## How to read Hot Chips and GTC

Hot Chips is the annual academic hardware conference where vendors present chip details. GTC is NVIDIA's own conference.

- **Hot Chips slides are usually freely posted.** The chip-talks are 25 minutes; very dense.
- **GTC keynote videos are on YouTube.** The technical sessions are gated behind registration but free.
- **Both have Q&A** that's often more informative than the prepared talk.

Useful talks for the modern era:

- Hot Chips 2017: V100.
- Hot Chips 2020: A100.
- Hot Chips 2022: H100.
- Hot Chips 2024: B100 / B200.
- GTC keynotes: every year, the highest-bandwidth update on NVIDIA's roadmap.

For each new generation, the Hot Chips talk is the closest thing to a *whitepaper-quality* technical talk you'll get from outside the whitepaper itself.

---

## Reading the competitive landscape

The non-NVIDIA pieces of the GPU landscape:

### AMD

- **MI300X** (2023): 192 GB HBM3, BF16 peak ~2.6 TFLOPS dense. AMD's H100 competitor.
- **MI325X** (2024–2025): refresh.
- **MI350 / MI400** (2025–2026): future generations.
- **ROCm**: AMD's CUDA equivalent. Improving fast; still behind on the software stack.
- **HIP**: AMD's CUDA-source-compatible API.

Reading AMD coverage, watch for:
- **Throughput at *equal precision*** (often AMD wins HBM bandwidth and loses on tensor-core utilization due to software).
- **PyTorch / vLLM support state.**

### Google TPU

- **TPU v4 / v5 / v5p / v6**: Google's in-house ASIC. Different architecture: systolic array of MMA units, smaller per-chip memory, *much* larger pod size (up to 8960 v5p chips in a single network domain).
- **Jax / XLA**: the standard software stack for TPU.

Reading TPU coverage, watch for:
- **Per-chip vs per-pod claims.** TPU pods compete with NVIDIA's largest SuperPODs.
- **Software portability.** Most production LLM work happens on PyTorch; running on TPU means rewriting in JAX (which is increasingly viable).

### AWS Trainium / Inferentia

- **Trainium2** (2024): AWS's training ASIC, competitive on $/throughput within AWS.
- **Inferentia2** / **Trainium1-inference mode**: inference-focused chips.
- **Neuron SDK**: the software stack.

### Cerebras, Groq, SambaNova, others

Wafer-scale and other exotic architectures. Read for awareness; their economics are different (often inference-only at niche scale).

### Microsoft Maia, Google Trillium (TPU v6), Meta MTIA

Hyperscaler-internal ASICs. The press coverage tells you the *strategic* story (each hyperscaler hedging on NVIDIA), even if the per-chip details are sparse.

---

## How to read capex announcements

Common shapes:

- "$N billion AI capex this year." Roughly: ~70% of this is GPUs. So $10B capex ≈ $7B in GPUs ≈ ~250k H100-equivalents.
- "We're building X megawatts of capacity." 1 MW ≈ 1000 H100-equivalents (allowing for cooling, etc.).
- "We've ordered N units of GPU X." Treat as plausible-but-unverified. NVIDIA doesn't break out customer.

When evaluating: do the numbers add up? E.g., if a company announces $20B capex but only $15B in PP&E in their 10-Q, the announcement is forward-looking or has accounting nuance.

---

## A reading routine

Most engineers I know who are fluent here keep one or two of these going:

- **Weekly:** a SemiAnalysis or Dylan Patel piece. Allocate 1 hour for reading + 30 min for any vocabulary lookups.
- **Monthly:** a fresh whitepaper section. NVIDIA publishes ~one major whitepaper per year + several minor ones.
- **Quarterly:** earnings calls of NVDA + the major hyperscalers. Skim transcripts for capex numbers.
- **Annually:** Hot Chips talks. Block 4–5 hours one weekend.

This is the *passive* reading. The *active* part is the case studies below.

---

## What to read next

- The case studies in [`case-studies/`](case-studies/). Start with `blackwell-tensor-memory.md` since you have the vocabulary now.
- Pick a Hot Chips talk for the current generation. Watch with the slides open.

---

## Self-check

1. What's the difference between a vendor whitepaper and a SemiAnalysis article in terms of how to read them?
2. What does "sparsity" almost always mean in NVIDIA marketing?
3. What's MIG? When is it relevant?
4. What's the rough order in which to read the sections of a NVIDIA architecture whitepaper?
5. How would you cross-check a capex announcement?
6. Which TPU generations would you currently compare to which NVIDIA generations?
