# 05 — Interconnect: NVLink, NVSwitch, IB, and Cluster Topology

GPU compute is only as fast as the interconnect feeding it. The hyperscaler press spends enormous ink on interconnect: NVLink generation, NVSwitch sizing, scale-up vs scale-out, rail-optimized topologies, in-network compute, optical vs electrical. This file is the vocabulary.

If you've done Chapter 03 you've felt NVLink's bandwidth as a developer. This file takes the cluster-architect's view.

---

## Two scales: scale-up and scale-out

The most fundamental distinction in cluster design:

- **Scale-up:** make one logical computer bigger. NVLink connects many GPUs into a single tight memory domain. Latencies sub-microsecond, bandwidths 100s of GB/s. Limited to a *pod* (sometimes called a *rack*, *NVL72*, *DGX SuperPOD building block*).
- **Scale-out:** connect many pods. InfiniBand / RoCE / Ethernet. Latencies single-digit microseconds, bandwidths 10s of GB/s. Connects thousands to hundreds of thousands of GPUs.

The choice between these matters because TP wants scale-up (low latency, high bandwidth) and DP/PP can tolerate scale-out (higher latency OK).

A modern cluster has both. The press articles you'll read are mostly about how a vendor balances the two.

---

## NVLink: the scale-up fabric

NVLink is the chip-to-chip link NVIDIA designs into its GPUs. It's not a generic bus — it's a custom switched fabric.

Per-GPU bandwidth (full duplex, sum of all lanes):

| Generation | Year | Per-GPU BW |
|---|---|---|
| NVLink 1.0 | 2016 (P100) | 160 GB/s |
| NVLink 2.0 | 2017 (V100) | 300 GB/s |
| NVLink 3.0 | 2020 (A100) | 600 GB/s |
| NVLink 4.0 | 2022 (H100) | 900 GB/s |
| NVLink 5.0 | 2024 (B200) | 1.8 TB/s |

These numbers are *full-duplex sum*, i.e., 600 GB/s on A100 means 300 GB/s each direction. Press articles often quote the sum. Be careful.

### NVSwitch

A *switch chip* that connects GPUs in a non-blocking fashion. Without NVSwitch, NVLink lanes between GPUs are point-to-point — 8 GPUs would need a ring or partial mesh, and any-to-any communication ends up oversubscribed.

NVSwitch sits in the middle. Every GPU connects all its NVLink lanes to NVSwitch ports. Any GPU can reach any other GPU at full per-GPU bandwidth.

DGX A100 had 6 NVSwitch chips. DGX H100 has 4 NVSwitch chips. Blackwell's NVL72 has 9 NVSwitch chips.

### NVL72 / Blackwell pod

The Blackwell generation introduces **NVL72** — a single scale-up domain of **72 GPUs** (18 trays × 4 GPUs / tray, or various other configurations) all connected by NVSwitch at full bandwidth.

72 GPUs sharing one NVLink fabric is the largest scale-up domain NVIDIA has shipped. Previous DGX-class systems topped out at 8 GPUs sharing NVLink (with multi-DGX requiring InfiniBand). NVL72 essentially replaces the *first* layer of InfiniBand for very large training jobs.

The press obsesses over NVL72 because:

- 72 GPUs × 192 GB ≈ 13 TB of HBM in one tight memory domain.
- A 70B-100B model fits in one pod with TP=72 (or any product TP×PP=72).
- Inter-pod communication is still InfiniBand, but more of the training fits inside one pod.

---

## InfiniBand: the scale-out fabric

InfiniBand (IB) is the high-performance scale-out fabric most NVIDIA training clusters use.

Generations and bandwidths:

| Name | Year | Per-port BW |
|---|---|---|
| EDR | 2014 | 100 Gbps  ≈ 12 GB/s |
| HDR | 2017 | 200 Gbps  ≈ 25 GB/s |
| NDR | 2022 | 400 Gbps  ≈ 50 GB/s |
| XDR | 2024+ | 800 Gbps ≈ 100 GB/s |

A modern training GPU has either *one* or *several* IB ports per server. DGX H100 has 8× NDR400 (one per GPU). Aggregate inter-node BW per GPU: 400 Gbps ≈ 50 GB/s — much slower than NVLink's 900 GB/s.

This bandwidth gap (NVLink:IB ≈ 18× on Hopper, ~36× on Blackwell+NVL72) is *why* TP fits inside one node and DP/PP straddle nodes.

### NIC: ConnectX / BlueField

The IB card you plug in. *ConnectX-7* / *ConnectX-8* are Mellanox/NVIDIA's current NICs. *BlueField-3* is the DPU (data processing unit) — a small ARM CPU + NIC + offload engine. Used for offloaded networking work in some hyperscalers.

### Topology

InfiniBand fabrics are switched. The arrangement matters:

- **Fat-tree:** classical "blocking-free" topology. Three layers of switches, each layer has more switches than the last. Any-to-any communication possible at full bandwidth. Expensive.
- **Rail-optimized:** each GPU has its own dedicated path (rail) through the fabric. GPU 0 on node A talks to GPU 0 on node B over rail 0; GPU 1 talks GPU 1 over rail 1. Used for DP-heavy workloads.

Most modern clusters are some hybrid; the press articles get specific about how exactly the topology was designed.

---

## Ethernet alternatives

The scale-out alternative people have been pushing for: Ethernet instead of InfiniBand.

**Pros:**
- Cheaper, more vendor diversity (Arista, Broadcom, NVIDIA Spectrum).
- Operations teams know how to run Ethernet.

**Cons:**
- Historically higher latency, less RDMA-friendly.
- Software stack (RoCE — RDMA over Converged Ethernet) is less mature.

Recent vendor pushes:
- **NVIDIA Spectrum-X** — NVIDIA's Ethernet-for-AI platform.
- **Ultra Ethernet** consortium — open-standard effort.
- **Microsoft's Maia / Cobalt clusters** — Ethernet-based.

For the article reader: when you see "ROCE," it's RDMA-over-Ethernet, a way to get InfiniBand-style low-latency RDMA over Ethernet wires.

---

## Optical interconnect

Both NVLink and IB are mostly electrical (copper) today, with optical reaches for longer cables. Press articles will discuss:

- **Co-packaged optics (CPO)**: optical transceivers integrated into the switch chip. Saves power vs pluggable transceivers. Industry transition over the next ~5 years.
- **DWDM / Linear Drive Pluggable Optics**: longer reach, lower-power transceivers.
- **NVIDIA's "Quantum-2" IB switches**: optical.

The interesting numbers: power per bit moved. Optical wins above ~3 meters; electrical wins below.

---

## In-network compute

The newest pattern: do *some* of the reduction inside the switch.

**SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)** — NVIDIA's IB feature where the switch performs the sum during an all-reduce. Instead of N hops through the fabric, you get a tree-like reduction in the switch. Saves bandwidth and latency.

Used by NCCL when enabled. SemiAnalysis articles often mention SHARP as an example of "the network is part of the compute."

---

## Topologies in the wild

A "DGX SuperPOD" of N GPUs looks roughly like:

```
- Node = DGX H100 = 8 GPUs + 8 ConnectX-7 NICs + 4 NVSwitch chips
  - GPUs intra-node: NVLink 4.0 via NVSwitch, 900 GB/s/GPU
  - GPUs to network: each via one NIC, 400 Gbps
- Rack = several nodes
- Pod = several racks, each with leaf IB switches
- SuperPOD = several pods, with spine IB switches
- Multi-SuperPOD = several SuperPODs, with core IB switches
```

For Blackwell:

```
- NVL72 pod = 72 GPUs (18 compute trays × 4 GPUs, 9 NVSwitch trays) all in NVLink 5
- Inter-pod = IB NDR/XDR
```

The structural change with NVL72: the *first hop* of scale-out is now between **72-GPU pods**, not between 8-GPU nodes. Lets you train a 100B-class dense model entirely inside the scale-up domain.

---

## Why bandwidth doesn't tell the whole story

When the press compares clusters by "aggregate bandwidth," remember:

- **Latency** matters as much as bandwidth for small messages. TP all-reduces are *latency*-sensitive on the tail.
- **Topology** determines actual end-to-end throughput. A fat-tree at 100% utilization is different from a rail-optimized at 100% utilization.
- **Software stack** (NCCL version, NCCL_ALGO choice, SHARP enablement) can change observed throughput by 2× on the same hardware.
- **Failure modes** matter. A 1024-GPU job with a 1-in-10000 link error rate per second sees an error every ~10 seconds. Cluster reliability is part of cluster performance.

---

## What to read next

- **NVIDIA "NVLink and NVSwitch" technical brief** — current generation.
- **NVIDIA "Scaling Generative AI with InfiniBand"** white paper.
- **Mellanox / NVIDIA SHARP technical brief.**
- **Meta's "Building Meta's GenAI Infrastructure" blog post** — describes their RoCE-based fabric.
- **OpenAI's "Scaling Kubernetes to 7500 nodes" post** — node-level operational lessons.
- **SemiAnalysis articles on Blackwell, NVL72, networking** — once you have the vocabulary above.
- Move on to `06-economics.md`.

---

## Self-check

1. What's the bandwidth gap between NVLink 4.0 and NDR IB? What does it imply for TP placement?
2. What's an NVSwitch and why doesn't a ring topology suffice?
3. What's NVL72, and what training-workload implication does it have?
4. What's the difference between InfiniBand and RoCE?
5. What does SHARP do, and when does it help?
6. Why is "aggregate cluster bandwidth" a misleading single metric?
