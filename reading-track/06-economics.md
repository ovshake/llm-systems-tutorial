# 06 — Economics: TCO, $/token, Power, and Capacity Planning

The hyperscaler press talks a lot about money. Capex, opex, TCO, $/MFU-hour, $/token-served, power-per-rack. This file is the vocabulary so when a SemiAnalysis piece says "Blackwell improves $/token by 5× at the rack level," you know which numbers are doing what work.

You will not become a TCO analyst from this file. You'll be able to *read* a TCO analysis without bouncing off the units.

---

## The cost layers

A useful decomposition:

```
        Cost per inference call
                │
        ┌───────┼───────────┐
        │       │           │
   Compute   Memory      Network/IO
        │       │           │
        ├── GPUs        ── HBM bytes-per-token
        ├── CPU           moved
        ├── Storage     ── L2 cache
        └── Cooling     ── inter-GPU traffic
                          ── inter-node traffic
```

For LLM inference, the *compute* and *memory* lines dominate. Networking matters but is usually amortized cheaper. Storage matters for cold-start; not on the hot path.

### Capex vs Opex

- **Capex (capital expenditure):** one-time cost of buying the hardware. GPUs, switches, racks, cabling, building / leasing the datacenter shell.
- **Opex (operating expenditure):** ongoing cost. Power, cooling, network bandwidth, staff, software licenses, depreciation if you bookkeep that way.

**TCO (Total Cost of Ownership)** = sum over the useful life of the cluster. For GPUs, "useful life" is usually 3–5 years before they're considered obsolete vs the latest generation.

Press articles often quote *amortized hourly TCO per GPU* — pretend the cluster runs at 100% for the depreciation period, divide total cost. Numbers like $2.50/H100-hour are typical (for an enterprise owner; cloud rentals are higher).

---

## What's expensive in a GPU cluster

Rough breakdown for a modern AI cluster:

```
GPUs:             50–60% of capex
Networking:       10–15%
Storage:          5–10%
Memory (system):  3–5%
CPUs:             3–5%
Datacenter shell: 10–15% (power, cooling, racks, real estate)
```

GPUs dominate capex but not opex — power and cooling come back to claim a lot of the opex.

### Power per GPU

| | TDP | Practical power |
|---|---|---|
| A100 SXM4 | 400 W | ~350–400 W |
| H100 SXM5 | 700 W | ~650–700 W |
| H200 SXM | 700 W | ~650–700 W |
| B200 SXM | 1000 W | ~900–1000 W |

Plus a non-trivial overhead for CPUs, NVSwitches, NICs, fans, idle power, etc. A DGX H100 system draws ~10 kW for 8 GPUs.

### Power per rack

Standard rack: ~30 kW for liquid-cooled AI. NVL72 racks push 120 kW per rack — needing direct-to-chip liquid cooling. Datacenter retrofits to support this are non-trivial.

Press articles will discuss things like:

- **PUE (Power Usage Effectiveness)**: total facility power / compute power. PUE 1.1 means 10% overhead for cooling, lights, etc. Modern AI datacenters target 1.05–1.15.
- **Power Cap**: utilities can only deliver so much. Some datacenters are "power constrained" — they have rack space but no more megawatts.
- **Stranded power**: an interesting term meaning power capacity exists but not where compute is. Drives the choice of build sites.

### Cooling

Direct-to-chip liquid cooling is now standard for AI workloads. Air cooling is impossible for NVL72 racks. Articles will discuss:

- **Cold plates** on GPU and CPU.
- **CDU (Coolant Distribution Unit)** — manifolds in the rack.
- **Facility water** — the loop that takes heat outside.

Microsoft's underwater datacenters and Google's seawater-cooled designs are at the exotic end; press articles often mention them.

---

## $/token: how it's actually computed

For LLM inference, "$/token served" decomposes as:

```
$/token = (amortized $/GPU-hour) / (tokens/GPU/hour)
```

The numerator is the (capex + opex) / (lifetime × utilization). The denominator is the model's achieved throughput.

Both terms move when you change:

- **Generation:** Blackwell costs more per GPU but produces more tokens/hour.
- **Quantization:** INT4/FP4 weights speed up inference 2–4× → 2–4× lower $/token, holding all else.
- **Batch size:** higher batch → higher throughput → lower $/token, but worse ITL.
- **TP degree:** higher TP fits bigger models but adds comm → throughput curve is non-monotone.

A SemiAnalysis-style "$/token improvement" claim is *always* qualified by these axes. Read the qualifier.

### Two reference numbers

- **Llama-3 8B** at production scale on H100 with vLLM: ~5,000-10,000 tokens/sec aggregate per H100. At $2.50/GPU-hour, that's ~$0.05–0.10 per million tokens (back-of-envelope; YMMV).
- **Llama-3 70B** at TP=8 on H100: ~500–1500 tokens/sec aggregate. At $20/8GPU-hour, ~$5–15 per million tokens.

Real serving APIs price at $0.50–$15 per million tokens depending on model size. The arbitrage between these and the underlying cost is what the inference vendor lives on (plus their software stack, plus their reliability, plus their margins).

---

## Utilization is the lever you don't see

A cluster spec is "1024 GPUs at $2.50/hour amortized." A *real* cluster runs at maybe 60% utilization across all jobs. That 40% idle is pure opex burn.

Hyperscalers spend enormous engineering effort on:

- **Job packing** (multiple training jobs sharing a node).
- **Preemptible / spot allocation**.
- **Fast scaling** (a job can grow and shrink without restart).
- **Multi-tenant inference** (multi-LoRA, vLLM's quota system, etc.).

A 70% → 85% utilization improvement is worth a *generation* of hardware improvement, easily.

### MFU revisited

In the LLM-training context, **MFU (model FLOPs utilization)** is "of the GPU's peak FLOPS, what fraction is being spent on useful model FLOPs?" 40–55% on a tuned dense pretraining is normal.

There's also **HFU (hardware FLOPs utilization)** — same idea but counting *all* FLOPs the GPU does (including recompute from activation checkpointing). HFU > MFU when checkpointing is on; MFU is the more honest number for "training efficiency."

When an article quotes MFU, check which one (most are MFU). 50% MFU on a 1024-GPU H100 cluster training a Llama-3-70B-class model is roughly state of the art at the time of writing.

---

## Where the next gains come from

For someone reading the press from a systems-engineering perspective, the durable patterns:

1. **Numeric format**: BF16 → FP8 → MXFP4 has been the throughput driver. Each step needs new methodology.
2. **Memory hierarchy**: TMEM (Blackwell), more L2, more SMEM per generation.
3. **Interconnect**: NVL72 reshaping training scaling.
4. **Software**: vLLM/SGLang/MAX have done as much for $/token over 2 years as a generation of silicon.
5. **Specialization**: dedicated inference SKUs (B100 vs B200), inference vs training topologies.

A serious systems engineer is fluent in *all five*. The Reading Track equips you for the first four; the main course's Chapters 12–16 cover the software side.

---

## What to read next

- **SemiAnalysis general articles** — pick any "X vs Y deep dive" once you've read this file.
- **Dylan Patel's posts on datacenter capacity and capex** — for the macro view.
- **Andy Bechtolsheim's keynotes on Hot Interconnects** — for the networking-economics view.
- **The Information / The Verge** for hyperscaler capex announcements; verify against company filings.
- **Public earnings transcripts** for Microsoft, Meta, Google — they specify AI capex.
- Move on to `07-roadmap-reading.md`.

---

## Self-check

1. What's the difference between capex and opex, with a concrete example for an AI cluster?
2. What's PUE, and what's a typical AI-datacenter value?
3. How does $/token decompose into hardware and software factors?
4. Why does a 70% → 85% utilization improvement matter more than a 10% throughput improvement?
5. What's the difference between MFU and HFU?
6. What's a "stranded power" site, and why would you avoid one?
