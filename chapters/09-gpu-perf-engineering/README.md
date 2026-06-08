# Chapter 09 — GPU Performance Engineering

**Hardware target:** 1 GPU is enough; multi-GPU adds NCCL profiling.
**Time budget:** 3–4 weeks.
**Prerequisites:** Chapters 02 and 03 — you've used the PyTorch profiler and can name kernels.

---

## Why this chapter exists

So far you've used the PyTorch profiler as a black box: "look at the trace, find the slow thing." That's enough to *spot* a bottleneck. To *fix* one, you need a deeper mental model of the GPU. This chapter builds that model.

By the end you will:

1. Be able to draw the H100 memory hierarchy from memory and label the bandwidths.
2. Use Nsight Systems for timeline analysis and Nsight Compute for kernel-level analysis.
3. Read a roofline plot and decide if a kernel is compute-bound or bandwidth-bound.
4. Know when to fuse kernels and when fusion doesn't help.
5. Use CUDA graphs to amortize launch overhead.

This is the prerequisite for Chapters 10 (Triton kernels) and 11 (CUTLASS/CuTe).

---

## Read first

- **NVIDIA H100 (Hopper) architecture whitepaper** (and the Ampere/A100 whitepaper if you want to see what changed generation-to-generation).
- **Programming Massively Parallel Processors** (Hwu, Kirk, Hajj) — Chapters 1–6 are the canonical CUDA mental model.
- **NVIDIA CUDA C++ Programming Guide** — at least the memory hierarchy and execution model sections.
- **Nsight Systems user guide** — focus on the GUI walkthrough.
- **Nsight Compute user guide** — `ncu` CLI usage, the GPU Speed Of Light section.
- **The Roofline Performance Model** — Williams, Waterman, Patterson.
- **NVIDIA developer blog posts** on the PyTorch profiler and CUDA graphs.

---

## Concepts

### 1. The execution hierarchy

```
Grid          ──▶ a collection of thread blocks (the whole launch)
  Block       ──▶ a collection of threads (max 1024 threads/block)
    Warp      ──▶ 32 threads executing in lockstep (SIMT)
      Thread  ──▶ the basic unit; has its own registers
```

A kernel launch specifies grid dimensions and block dimensions. The hardware scheduler assigns blocks to Streaming Multiprocessors (SMs). H100 has 132 SMs (A100 had 108). Each SM can hold many blocks concurrently (subject to register / shared-memory / warp budgets).

### 2. The memory hierarchy (H100 numbers)

```
Registers              : ~256 KB per SM, free latency  (per-thread; 32 bits each, ~65k per SM)
Shared memory / L1     : ~228 KB per SM, ~20 cycles
L2 cache               :  50 MB (whole GPU), ~200 cycles
HBM3                   : 80 GB, ~400 cycles, 3.35 TB/s
Host RAM (via PCIe Gen5): up to your system, ~50 GB/s
```

Implications:
- "Coalesced" memory access (consecutive threads → consecutive addresses) hits the same cache line. Uncoalesced → multiple cache lines, big slowdown.
- Shared memory is **manual L1**. You explicitly stage data from HBM to shared memory.
- Tensor Cores read inputs from shared memory or registers; output goes to registers.

### 3. The roofline

```
achievable_flops_per_second = min(peak_flops, peak_bandwidth * arithmetic_intensity)
```

For H100 BF16: peak 989 TFLOPS, peak HBM 3.35 TB/s. Crossover intensity = 989e12 / 3.35e12 ≈ 295 FLOPs/byte. (Note the crossover *rose* from ~200 on A100: H100's compute grew ~3× while its bandwidth grew only ~1.6×, so more kernels are bandwidth-bound on Hopper than were on Ampere.)

A kernel doing `f` FLOPs and reading/writing `b` bytes has intensity `f / b`. Compare against 295:

- **f/b > 295**: compute-bound. Speed up by using more cores efficiently — tiling, Tensor Cores, fewer wasted threads.
- **f/b < 295**: bandwidth-bound. Speed up by *moving less data* — fusion, in-place updates, lower precision.

You'll re-derive this in Exercise 3.

### 4. Common bottlenecks

| Bottleneck | Symptom | Tool that reveals it | Fix |
|---|---|---|---|
| Kernel launch overhead | Many tiny kernels, gaps on stream | Nsight Systems timeline | Fusion, CUDA graphs, bigger batch |
| Bandwidth-bound | Achieved < peak FLOPs; high HBM use | Nsight Compute "Speed Of Light" | Fusion, fewer reads, reuse |
| Tensor Core underutilized | Achieved BF16 FLOPS < peak | NCU "Tensor Pipe" section | Right shapes (multiples of 16), aligned, contiguous |
| Bank conflicts | Slower than expected shared mem | NCU "Shared Memory" section | Pad/swizzle indices |
| Uncoalesced loads | High global load transactions | NCU "Memory Workload" | Reshape access pattern |
| Warp divergence | Low warp execution efficiency | NCU "Scheduler Stats" | Restructure conditionals |
| Low occupancy | Few warps per SM, latency exposed | NCU "Occupancy" | Reduce register/shared-mem per block |
| CPU bottleneck | Long gaps on GPU stream, CPU at 100% | Nsight Systems CPU rows | Optimize Python / move work to GPU |

### 5. Kernel fusion

If you have:
```python
y = (x + a) * b
z = y.relu()
```
that's two kernels, each reading and writing memory. Fused: one kernel reads `x, a, b`, computes everything in registers, writes `z`. Bandwidth halved. The classic example.

`torch.compile` does many fusions automatically. Manual Triton kernels are how you go further. Chapter 10.

### 6. CUDA graphs

Each `cudaLaunchKernel` has ~10 μs of CPU overhead. For small kernels (e.g., a 256-element add), the overhead exceeds the kernel time. CUDA graphs let you record a sequence of launches once and replay them as a single graph launch.

Use cases in LLM training:
- Inference at small batch (decode phase): a few μs per kernel * many kernels = death by 1000 cuts. CUDA graphs collapse it to one launch.
- Training step is fixed-shape: record the entire step.

PyTorch exposes `torch.cuda.CUDAGraph` and `torch.cuda.graph` context manager.

### 7. Mixed precision: BF16 → FP32 carefully

Inside Tensor Core matmuls, accumulation is usually in FP32 even with BF16 inputs. You get BF16 throughput with FP32 accuracy of accumulation. But after the matmul, the output is back to BF16 — a layer norm in BF16 has different numerics than one in FP32.

The right pattern:
- Big matmuls in BF16 (Tensor Cores).
- Reduction-prone ops (loss, layer norm, softmax) often in FP32 even in "BF16 training."

`torch.amp.autocast` handles this automatically for standard ops.

---

## Hands-on

### Exercise 1 — Install Nsight Systems / Nsight Compute

On the node, install `nsight-systems` and `nsight-compute` (both are NVIDIA-provided). Confirm `nsys --version` and `ncu --version` work.

For remote use: install the GUI on your laptop, capture on the server with the CLI, copy `.nsys-rep` / `.ncu-rep` back, open locally.

**Deliverable:** Versions printed in `NOTES.md`.

### Exercise 2 — Capture a Nsight Systems trace

Run your `train.py` from Chapter 01 under nsys:

```bash
nsys profile -o run --capture-range cudaProfilerApi --capture-range-end stop -t cuda,osrt,nvtx \
    python train.py --config configs/small.yaml
```

Wrap a few training steps in `torch.cuda.profiler.cudart().cudaProfilerStart()` / `Stop` (or `nvtx.range_push/pop`). Open `run.nsys-rep` in Nsight Systems GUI.

In the trace, find:
1. The CUDA HW timeline.
2. Forward pass kernels (look for `aten::*_forward`, matmuls, `gemm`).
3. Backward pass (`*_backward`).
4. Optimizer step (`adamw_kernel`).
5. The NCCL row (if DDP).

**Deliverable:** Annotated screenshot in `NOTES.md` with each phase labeled.

### Exercise 3 — Roofline a layer norm

Write [`roofline_layernorm.py`](roofline_layernorm.py): for an input of shape `(B, S, H)` with various sizes, run a layer norm and compute:

- Bytes moved: roughly `4 * B * S * H` (read input, read weight/bias, write output, plus the reductions in BF16).
- FLOPs: roughly `8 * B * S * H` (mean, variance, normalize, affine).
- Arithmetic intensity: FLOPs / bytes.
- Achieved FLOPs/s (measured).

Plot achieved FLOPs/s vs intensity, with the roofline overlaid. Layer norm should sit on the *bandwidth* line, well below the compute roof.

**Deliverable:** `roofline.png` and a paragraph in `NOTES.md`.

### Exercise 4 — Nsight Compute on one matmul

```bash
ncu --set full -o gemm_profile python gemm_one.py
```

(`gemm_one.py`: a single `torch.matmul` of a fixed shape.) Open `gemm_profile.ncu-rep` in NCU GUI. In the "Speed Of Light" section, find:

- Compute (SM) Throughput.
- Memory Throughput.
- Tensor pipe utilization.

For a `(4096, 4096) @ (4096, 4096)` BF16 matmul on H100, you should see ~500–700 TFLOPS achieved (≈ 50–70% of peak). A 4096³ matmul is on the small side for saturating an H100, so don't be surprised if you need larger shapes (8192³) to get close to peak.

**Deliverable:** Screenshot of the SOL section; numbers in `NOTES.md`.

### Exercise 5 — Find the bottleneck in your training step

Take your `medium`-preset Chapter 01 trainer. Run it under `nsys`. In the trace:

1. Sum forward kernel time, backward, optimizer, "other."
2. Pick the *single longest kernel* in forward. Run NCU on it (use `--target-processes` and `--launch-skip` / `--launch-count` to capture a specific kernel).
3. Read the SOL: is it compute-bound or memory-bound?
4. Read the "Warp State Statistics" section. Where are warps stalled?

**Deliverable:** Writeup of the bottleneck and the suggested fix.

### Exercise 6 — CUDA graphs for inference

Take your Chapter 01 `generate()` function (single-token autoregressive). Profile it: most kernels are tiny.

Now wrap the decode step in `torch.cuda.graph`:

```python
g = torch.cuda.CUDAGraph()
# warm up to allocate buffers
for _ in range(3):
    out = step(...)
torch.cuda.synchronize()
with torch.cuda.graph(g):
    out = step(...)
# replay
for _ in range(N):
    g.replay()
```

Measure tokens/sec before and after.

**Deliverable:** Speedup table. For a small model, CUDA graphs typically yield 2–4× during decode.

### Exercise 7 — Fusion via `torch.compile`

Take a simple op chain (e.g., `(x + a) * b → relu`). Time it eager, then `torch.compile` it. Profile both with NCU; compare the kernel count and the achieved bandwidth.

`torch.compile`'s inductor backend usually fuses element-wise ops into one Triton kernel — you'll see kernel count drop from 3 to 1.

**Deliverable:** Numbers in `NOTES.md`.

---

## Self-check

1. H100 peak BF16 FLOPS? Peak HBM bandwidth? Roofline crossover?
2. Why does coalesced memory access matter, and what causes it to break?
3. What does Nsight Compute's "Speed Of Light" tell you that "kernel time" doesn't?
4. When does kernel fusion help, and when is it irrelevant?
5. Why are CUDA graphs huge for inference but only marginal for training?
6. What does "warp divergence" mean, and when does it show up in LLM code?

---

## Deliverables before moving on

- nsys trace screenshot from training.
- Roofline plot for layer norm.
- NCU SOL screenshot for one matmul.
- Bottleneck writeup for your longest-running kernel.
- CUDA graphs speedup measurement.
- `NOTES.md` with all of the above and your own summary of "where is *my* training time going."

This chapter is the gate to writing custom kernels in Chapter 10. Don't skip the profiling exercises.
