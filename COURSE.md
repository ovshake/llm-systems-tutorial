# LLM Systems — A Hands-On Course

This course turns `ROADMAP.md` into 19 chapters you can work through on a single 4×A100-80GB node. Every chapter has a *read* block (a long-form README with concepts + references) and an *implement* block (a runnable baseline plus exercises that make the baseline measurably better, or expose where it breaks).

The course assumes you already know PyTorch and the math of a transformer. If "implement attention from scratch in 20 minutes" makes you sweat, do Chapter 01 first regardless of the rest of the roadmap.

---

## How to use the course

Each chapter follows the same loop, adapted from the Ultra-Scale Playbook's recommended approach:

1. **Read** the chapter README cover-to-cover before touching code.
2. **Run** the baseline that ships with the chapter and confirm the numbers it prints match what the README claims (this is your "ground state" before you change anything).
3. **Implement** the exercises in order. Each exercise has a clear deliverable: a code change, a metric, a plot, or a short writeup.
4. **Profile.** Almost every chapter ends with "run the profiler, find the bottleneck, write down what you saw." This is the *whole* point of the systems track.
5. **Write** a short `NOTES.md` per chapter (5–20 bullets) summarizing what you learned, what surprised you, and what's still fuzzy. These become your interview prep and portfolio.
6. **Move on** only after you can answer the "Self-check" questions at the bottom of the chapter without looking them up.

Anti-pattern to avoid: reading three chapters in a row without writing code. The whole reason this domain is hard to learn from books is that the numbers only become real once you measure them on hardware. **Treat the README as documentation for the implementation, not a substitute for it.**

---

## Chapter map

| # | Folder | Topic | Approx. time |
|---|---|---|---|
| 00 | [`chapters/00-setup`](chapters/00-setup) | Setup, GPU sanity checks, NCCL, NVLink/PCIe probes | 1 week |
| 01 | [`chapters/01-transformer-from-scratch`](chapters/01-transformer-from-scratch) | Tiny GPT, BF16, grad accum, activation checkpointing, ckpt save/resume | 2 weeks |
| 02 | [`chapters/02-memory-compute-budget`](chapters/02-memory-compute-budget) | Memory math, MFU, FlashAttention, step time breakdown | 1–2 weeks |
| 03 | [`chapters/03-data-parallelism`](chapters/03-data-parallelism) | DDP from scratch, all-reduce, bucketization, scaling | 2 weeks |
| 04 | [`chapters/04-fsdp-zero`](chapters/04-fsdp-zero) | PyTorch FSDP, ZeRO stages, hybrid sharding | 2–3 weeks |
| 05 | [`chapters/05-tensor-parallelism`](chapters/05-tensor-parallelism) | TP linear/MLP/attention, sequence parallelism, vocab parallel CE | 3 weeks |
| 06 | [`chapters/06-pipeline-parallelism`](chapters/06-pipeline-parallelism) | GPipe, 1F1B, interleaved, bubble analysis | 2–3 weeks |
| 07 | [`chapters/07-3d-parallelism-nanotron`](chapters/07-3d-parallelism-nanotron) | TP+PP+DP in Nanotron / Megatron-LM | 3–4 weeks |
| 08 | [`chapters/08-moe-training`](chapters/08-moe-training) | Top-k router, load balance loss, expert parallel all-to-all | 4 weeks |
| 09 | [`chapters/09-gpu-perf-engineering`](chapters/09-gpu-perf-engineering) | CUDA model, Nsight Systems/Compute, roofline, fusion | 3–4 weeks |
| 10 | [`chapters/10-triton-kernels`](chapters/10-triton-kernels) | Triton kernels: matmul, softmax, RMSNorm, SwiGLU, attention | 4 weeks |
| 11 | [`chapters/11-cutlass-cute-dsl`](chapters/11-cutlass-cute-dsl) | CUTLASS GEMM, CuTe layouts, CuTe DSL kernels | 4–6 weeks |
| 12 | [`chapters/12-inference-fundamentals`](chapters/12-inference-fundamentals) | KV cache, greedy/sampling, naive→continuous batching | 3 weeks |
| 13 | [`chapters/13-paged-attention-prefix-cache`](chapters/13-paged-attention-prefix-cache) | Paged KV cache, block manager, prefix cache, chunked prefill | 3 weeks |
| 14 | [`chapters/14-advanced-inference`](chapters/14-advanced-inference) | Speculative decoding, TP inference, quant, disagg, CUDA graphs | 4 weeks |
| 15 | [`chapters/15-vllm-deep-dive`](chapters/15-vllm-deep-dive) | vLLM architecture tour, reproduce a perf issue, plan a PR | 4–6 weeks |
| 16 | [`chapters/16-sglang-deep-dive`](chapters/16-sglang-deep-dive) | SGLang architecture, Mini-SGLang, plan a PR | 4 weeks |
| 17 | [`chapters/17-post-training-systems`](chapters/17-post-training-systems) | SFT, DPO/IPO/KTO/ORPO, RLHF/GRPO, rollout systems | 4 weeks |
| 18 | [`chapters/18-capstone-100b-scale`](chapters/18-capstone-100b-scale) | 100B-dense and MoE training plans; portfolio reports | ongoing |

Total ≈ 12 months at 12 h/week. Some chapters are explicitly longer; some shorter. Don't try to keep weekly cadence — keep *content* cadence.

---

## Hardware envelope

This course is designed for **a single 4×A100-80GB node**:

- **Total VRAM:** 320 GB. Enough for a 7B BF16 model with full DDP, a 13B with FSDP, or a 30B-ish with FSDP + activation checkpointing + heavy parallelism (8 GPUs would be more comfortable but 4 is fine for the *concepts*).
- **NVLink/NVSwitch:** 600 GB/s aggregate on a DGX-style box. This is fast enough that intra-node TP (degree 2 or 4) is realistic.
- **No InfiniBand / multi-node:** We treat multi-node as a thought experiment. The 100B-scale planning in Chapter 18 covers it on paper.
- **Local storage / network:** Assume you have somewhere to stash a few hundred GB of datasets and checkpoints. If you don't, use TinyStories / Shakespeare-scale data — the principles don't care.

For each chapter we list a "Hardware target" line. If you have fewer GPUs, you can usually still run the chapter — just drop the parallel degrees. If you have more, increase them. Don't change the *order* of chapters to chase availability.

---

## Conventions

- **Python:** ≥ 3.10, managed with `uv`. Each chapter folder is runnable directly; some pin extra deps in a chapter-local `pyproject.toml` or `requirements.txt`.
- **Launch commands:** Single GPU: `python script.py`. Multi-GPU: `torchrun --standalone --nproc_per_node=4 script.py`. Multi-node is out of scope.
- **Toy dataset:** `tinyshakespeare` (1 MB plain text). Tokenized with `tiktoken` (GPT-2 BPE) and saved as a single binary file. For chapters that need a real-ish corpus (Chapters 04, 07, 08), we use a subset of `wikitext-103` or `fineweb` streamed from Hugging Face.
- **Models:** Configurable `tiny / small / medium / large` presets in `shared/utils/model_presets.py`. "Tiny" fits on a 12 GB GPU; "Large" needs the full 4×80 GB.
- **Logging:** `rich.print` for human output, `wandb` (or `tensorboard`) for runs. Each chapter prints a structured JSON line per step so you can diff runs trivially with `jq`.
- **Naming:** Files use kebab-case for scripts (`benchmark-all-reduce.py`), snake_case for modules (`fused_kernels.py`). Folders use the `NN-topic-name` form already in use.

---

## What "done" means for the course

You are done with the course when you can do all of these without looking anything up:

1. Given a model spec + a target throughput, sketch a parallelism strategy (TP/PP/DP/EP/SP degrees, microbatch, activation checkpointing policy) on a whiteboard, with rough memory and communication math, and identify the likely first bottleneck.
2. Take a slow training step and use Nsight Systems / the PyTorch profiler to localize the bottleneck to a specific kernel or collective.
3. Write a Triton kernel for an LLM op (RMSNorm, fused SwiGLU, vocab-parallel softmax, etc.) that beats the eager PyTorch baseline on an A100.
4. Implement a working continuous-batching inference loop with a paged KV cache, plus benchmarks for TTFT / ITL / throughput / p99.
5. Read a vLLM or SGLang issue / PR, reproduce the regression locally, and propose a fix or follow-up benchmark.
6. Explain DPO, GRPO, and rollout-based RLHF in terms of their *systems* requirements (memory for ref model, KL constraint, asynchronous rollout queue, etc.), not their math.

The chapter "Self-check" questions are a more granular version of this list.

---

## Reading track (parallel)

`chapters/` teaches you to *use* the GPU stack. There's a separate track, [`reading-track/`](reading-track/), that teaches you to *read about* it — SemiAnalysis articles, NVIDIA whitepapers, Hot Chips talks. It's a parallel resource you can dip into anytime; the chapters do not depend on it.

| # | Topic |
|---|---|
| 00 | Hardware fluency — SMs, memory hierarchy, cluster, the vocabulary |
| 01 | Tensor core evolution — V100 → A100 → H100 → B200, MMA shapes |
| 02 | Memory spaces — registers, SMEM, L2, HBM, DSMEM, TMA, TMEM |
| 03 | Numeric formats — FP32 / BF16 / FP8 / MXFP4 / NVFP4 |
| 04 | PTX and SASS — how to read low-level GPU code |
| 05 | Interconnect — NVLink, NVSwitch, NVL72, InfiniBand, RoCE |
| 06 | Economics — TCO, $/token, power, capacity planning |
| 07 | Roadmap reading — how to read NVIDIA whitepapers and analyst pieces |

Plus worked case studies in [`reading-track/case-studies/`](reading-track/case-studies/): Blackwell Tensor Memory, Hopper, MI300X-vs-H100.

When a chapter README mentions things like "WGMMA", "TMA", "FP8 E4M3", "NVL72", the reading-track files are the canonical place to fill in the depth.

---

## Course-wide cross-references

Some ideas show up in many chapters. Where this happens, the README points you back to the canonical chapter so you don't relearn:

- **Memory accounting** — Chapter 02 is the canonical reference. Later chapters say "see §02" rather than redoing the math.
- **NCCL collectives** — Chapter 03 introduces them; Chapters 04, 05, 06, 08 reuse the same mental model.
- **Profiling workflow** — Chapter 09 is canonical; earlier chapters (02 onward) use a *simplified* version (PyTorch profiler + `torch.cuda.Event`) and explicitly defer Nsight Systems / Compute to 09.
- **KV cache layout** — Chapter 12 introduces it; Chapter 13 generalizes to paged; Chapters 14–16 reuse the abstraction.

This way each chapter stays focused, and you can revisit specific systems concepts via a single canonical chapter.

---

## Reading list (course-wide)

These are the resources we reference repeatedly. Pin them in browser tabs:

- **Hugging Face Ultra-Scale Playbook** — the course's anchor reference for distributed training tradeoffs.
- **Nanotron** — production-style reference code for 3D parallelism, integrated with the Playbook.
- **PyTorch Distributed docs** — DDP, FSDP, `torch.distributed`, NCCL.
- **DeepSpeed docs** — ZeRO, pipeline engine, MoE.
- **Megatron-LM / Megatron-Core** — TP, PP, SP, CP implementations.
- **Triton tutorials** — the official tutorials are excellent and we work through them in Chapter 10.
- **CUTLASS** — `docs/` and `media/`. CuTe DSL has its own dedicated docs page.
- **vLLM docs + source** — `vllm-project/vllm`. Read the docs for usage, the source for the architecture.
- **SGLang docs + source + Mini-SGLang** — `sgl-project/sglang`.
- **NCCL user guide and tuning guide** — surprisingly readable, essential for debugging weird scaling.

Plus *papers* surfaced in individual chapters: Megatron-LM (Shoeybi et al.), FSDP, GPipe / PipeDream / 1F1B, FlashAttention 1/2/3, GShard, Switch Transformer, MegaBlocks, DPO, PPO, GRPO, vLLM (Kwon et al. — PagedAttention), and others. These live in each chapter's "References" section.

---

## Portfolio outputs

By the end of the course you should have *written down* at least these:

1. **Distributed training scaling report** — DDP vs FSDP vs Nanotron on a small dense model. Tokens/sec, MFU, peak memory, scaling efficiency, comm vs compute breakdown. (Built up across Chapters 03, 04, 07; finalized in Chapter 18.)
2. **MoE bottleneck report** — top-k routing, expert parallel all-to-all costs, token imbalance, throughput vs dense baseline. (Chapter 08.)
3. **Triton kernel series** — RMSNorm, softmax, fused SwiGLU, optionally simple attention; with bandwidth/FLOPs achieved and a roofline plot. (Chapter 10.)
4. **Mini inference engine** — KV cache, continuous batching, paged KV cache, prefix cache, with TTFT/ITL/throughput vs vLLM/SGLang. (Chapters 12–14.)
5. **OSS contribution log** — list of issues triaged, PRs opened, PRs merged, with one-paragraph summaries each. (Chapters 15–16.)
6. **Post-training writeup** — one DPO run + one rollout-based RL run, with notes on memory and orchestration. (Chapter 17.)
7. **100B training plan** — Chapter 18 capstone.

Keep these as Markdown files in a separate `portfolio/` repo you'll show employers; this repo is for the learning code itself.

---

## Quickstart

```bash
# 0. Clone + cd
cd llm-systems-tutorial

# 1. Make a venv with uv
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# 2. Sanity-check GPUs / NCCL / NVLink
cd chapters/00-setup
uv pip install -r requirements.txt   # if present
python verify_environment.py
torchrun --standalone --nproc_per_node=4 hello_distributed.py
python nvlink_bandwidth.py

# 3. Start Chapter 01
cd ../01-transformer-from-scratch
python train.py --config configs/tiny.yaml
```

If `verify_environment.py` complains, fix that before touching anything else. Half of distributed bugs are env bugs in disguise.

---

## A note on AI assistance

A lot of this content can be assembled with help from an LLM. That's fine for *exploration* — bad for *learning*. The exercises in each chapter are designed so that the only way to get the numbers right is to actually run them. Don't fake the metrics; you'll feel it in interviews.
