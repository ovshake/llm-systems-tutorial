# Chapter 00 — Setup & GPU Sanity Checks

**Hardware target:** 1 to 4 GPUs (works on as little as one A100; ideal is the full 4×A100-80GB).
**Time budget:** 1 week.
**Prerequisites:** Linux box with CUDA-capable GPUs and recent NVIDIA driver. You are comfortable with `bash`, `git`, and `uv`.

---

## Why this chapter exists

Half of the bugs in distributed training are environment bugs masquerading as ML bugs. A `RuntimeError: NCCL communicator init failed` after 9 minutes of debugging your model is the prototypical wasted afternoon. The fix is almost always: validate the environment *first*, with a script that doesn't touch your model, and only move on once those numbers look right.

This chapter is a checklist for that. After it, you should be able to:

- State your driver version, CUDA toolkit version, PyTorch CUDA version, and confirm they're consistent.
- Run a distributed hello world and see all 4 ranks print.
- Measure NVLink and PCIe bandwidth between each pair of GPUs and recognize an unhealthy topology.
- Measure NCCL all-reduce / all-gather / reduce-scatter bandwidth and compare to the hardware peak.
- Boot a PyTorch profiler trace and open it in `chrome://tracing`.

If any of these fail or look wrong, you'll catch it now rather than at Chapter 03.

---

## Read first

- **NVIDIA Driver / CUDA compatibility matrix** — the canonical "does my driver support this CUDA version" table. Search for "CUDA Toolkit and Compatible Driver Versions" on NVIDIA's site.
- **PyTorch installation page** — make sure you're using the right CUDA build (e.g., `cu121`).
- **NCCL user guide**, sections "Environment Variables" and "Troubleshooting". The variables you'll actually use most: `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS`, `NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE`, `NCCL_IB_DISABLE`, `NCCL_SOCKET_IFNAME`.
- **PyTorch distributed quickstart** — how `torchrun` populates `RANK`, `WORLD_SIZE`, `LOCAL_RANK`.
- **NVIDIA Nsight Systems quickstart** — even if you don't profile yet, install it now so it's not a yak-shave in Chapter 09.

---

## Concepts

### 1. The CUDA software stack (briefly)

A GPU program goes through several layers:

```
PyTorch / Triton / CUTLASS / your code      ← what you write
        │
        ▼
CUDA Runtime / Driver API                   ← libcudart / libcuda
        │
        ▼
GPU Driver (kernel module + user libs)      ← libcuda.so, nvidia kernel module
        │
        ▼
GPU hardware                                ← SMs, memory, NVLink fabric
```

A common bug: PyTorch was built against CUDA 12.1, you have driver 525 (which is CUDA 12.0). It looks fine, then explodes inside NCCL. Always check `nvidia-smi` *driver* version vs. PyTorch's reported CUDA version.

### 2. Topology and bandwidth

Within a node, GPUs can talk to each other through:

- **NVLink / NVSwitch** — high-bandwidth, direct GPU-GPU. A100 SXM4 has 12 NVLink lanes per GPU, 50 GB/s per direction per lane → ~600 GB/s aggregate per GPU. In a DGX A100 there are NVSwitches so any GPU-pair gets full bandwidth.
- **PCIe** — fall-back path. PCIe Gen4 x16 ≈ 25 GB/s each direction. ~30× slower than NVLink for collectives.
- **InfiniBand / RoCE** — between nodes. We don't use this.

Run `nvidia-smi topo -m`. If you see `NV12` or `NV4` between every GPU pair, you have NVLink. If you see `PXB` / `PHB` instead, you're on PCIe and tensor parallelism will be brutal.

### 3. NCCL collectives we'll use everywhere

| Collective       | What it does                                              | Where it shows up |
|------------------|-----------------------------------------------------------|-------------------|
| `all_reduce`     | Each rank gets the sum (or other op) of all ranks' tensors | DDP gradients, TP forward (column-parallel output) |
| `all_gather`     | Each rank gets the concatenation of all ranks' shards     | FSDP unsharding before compute |
| `reduce_scatter` | Reduce + split; each rank gets one shard of the sum       | FSDP gradient reduction |
| `all_to_all`     | Each rank sends a different chunk to every other rank     | MoE expert routing |
| `broadcast`      | One rank sends, all receive                               | Initialization, checkpoint loading |

The cost of each on a ring of N GPUs and tensor of size B bytes is roughly `2B(N-1)/N` (all-reduce) or `B(N-1)/N` (all-gather, reduce-scatter). On NVLink with 600 GB/s, an all-reduce of a 1 GB BF16 tensor on 4 GPUs takes roughly `2 * 1 GB * 3 / 4 / 600 GB/s ≈ 2.5 ms`.

We'll measure this for real.

### 4. Determinism is not free

The defaults in `torch.distributed` are *not* deterministic. NCCL uses a non-deterministic reduction order. If you need exact bitwise reproducibility (e.g., debugging an "is this regression in my code or the framework?" problem), set:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring
```

and in Python:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

There's a perf cost. Default to non-deterministic for performance work; flip on determinism only when you need to bisect.

---

## Hands-on

Install:

```bash
cd ../..                            # repo root
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
cd chapters/00-setup
```

### Exercise 1 — Environment audit

Run:

```bash
python verify_environment.py
```

Read the output. It prints driver, CUDA, PyTorch, NCCL, and Triton versions, plus per-GPU info. Confirm:

- All 4 GPUs are visible.
- BF16 is supported.
- The reported NCCL version matches the one your PyTorch was built against (warning printed if not).

**Deliverable:** Paste the output into `NOTES.md` for this chapter.

### Exercise 2 — Distributed hello world

```bash
torchrun --standalone --nproc_per_node=4 hello_distributed.py
```

Each rank should print its rank, local rank, and device. They should all see `world_size=4`. The script also does a small `all_reduce(1)` and asserts it returns 4 on every rank — a baseline that NCCL works at all.

**Deliverable:** Save the output to `NOTES.md`. Note which rank ran first.

### Exercise 3 — NVLink / PCIe bandwidth

```bash
python nvlink_bandwidth.py
```

This script copies a 1 GB tensor between every pair of GPUs and reports throughput. You'll get a 4×4 matrix.

**Deliverable:**
1. Save the matrix to `NOTES.md`.
2. Mark any pair below 200 GB/s — this means they're on PCIe rather than NVLink.
3. If you have a heterogeneous topology, plan around it: TP wants the fastest pairs.

### Exercise 4 — NCCL collectives benchmark

```bash
torchrun --standalone --nproc_per_node=4 nccl_bench.py
```

This measures `all_reduce`, `all_gather`, and `reduce_scatter` over a sweep of message sizes from 1 MiB to 1 GiB. Output looks like:

```
all_reduce     size=  256.0 MiB  t=  1.234 ms  algbw=  217.6 GB/s  busbw=  326.5 GB/s
```

**Deliverable:**
1. Save the full table to `NOTES.md`.
2. Compute the bus bandwidth as a percentage of NVLink peak (600 GB/s aggregate). You should see busbw saturate near 200–280 GB/s on a DGX-A100 — the link is shared.
3. Plot algbw vs message size in `bandwidth_plot.png` (use matplotlib). Note where the curve plateaus — this is the message size above which the link is the bottleneck rather than launch overhead.

### Exercise 5 — Profiler sanity

```bash
python profiler_smoke.py
```

This runs a few matmuls under `torch.profiler` and dumps a Chrome trace to `out/trace.json`. Open `chrome://tracing`, drag the file in. You should see:

- CPU events for `aten::mm`.
- CUDA stream events for the actual matmul kernel.
- A `cudaLaunchKernel` event preceding each kernel.

**Deliverable:** A screenshot of the trace in `NOTES.md`. You don't need to interpret it yet — Chapter 09 covers that.

---

## Self-check

You should be able to answer without looking things up:

1. What's the difference between *algbw* and *busbw*?
2. Why does `nvidia-smi topo -m` matter before you choose a TP degree?
3. What does `torchrun` populate in your environment so that `dist.init_process_group()` works without arguments?
4. Why is `time.perf_counter()` a bad way to time a CUDA kernel?
5. What's the first NCCL environment variable you set when a distributed run fails to start?

If any of these are fuzzy, re-read the "Concepts" section and run the corresponding script.

---

## Deliverables before moving on

Commit `NOTES.md` containing:
- `verify_environment.py` output.
- `hello_distributed.py` output.
- NVLink topology matrix from Exercise 3.
- NCCL bandwidth table + bandwidth plot.
- Profiler screenshot.
- A 1-paragraph "what I learned that surprised me."

If anything in those scripts errored, fix it before Chapter 01. The whole rest of the course depends on this stack working.
