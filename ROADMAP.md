Below is a practical roadmap from “I understand PyTorch” to “I can work on large-scale LLM training/inference systems and contribute to vLLM/SGLang-style repos.”

Your current resource, Hugging Face’s Ultra-Scale Playbook, is a very good anchor: it explicitly focuses on scaling LLM training on GPU clusters, uses Nanotron as production-oriented reference code, and discusses the real tradeoffs behind distributed training decisions.  ￼

⸻

North star

Your target role is not “ML researcher who trains models” but LLM systems / performance engineer.

That means you need four overlapping skill stacks:
	1.	Distributed training systems
Data parallelism, tensor parallelism, pipeline parallelism, expert parallelism, ZeRO/FSDP, checkpointing, NCCL, cluster failures, throughput debugging.
	2.	GPU performance engineering
CUDA mental model, memory hierarchy, roofline thinking, kernels, profiling, Triton, CUTLASS, CuTe DSL.
	3.	LLM inference engines
KV cache, paged attention, continuous batching, prefill/decode split, scheduling, quantization, speculative decoding, distributed serving.
	4.	Open-source production engineering
Reading large codebases, benchmarking rigorously, writing tests, reproducing regressions, landing small but real PRs.

⸻

Phase 0 — Prerequisites: what you should already be comfortable with

Before going deep into 100B-scale training, make sure you are solid on these.

Python / PyTorch

You should be able to implement from scratch:
	•	Transformer block: attention, MLP, RMSNorm/LayerNorm, residuals.
	•	Autograd training loop.
	•	Mixed precision training with BF16/FP16.
	•	Gradient accumulation.
	•	Activation checkpointing.
	•	Basic distributed PyTorch with torchrun.

Minimum project:

Train a tiny GPT-style model on 1 GPU, then modify it to support gradient accumulation, activation checkpointing, BF16, and checkpoint save/resume.

Systems basics

You do not need to be a kernel engineer yet, but you should understand:
	•	GPU vs CPU execution.
	•	Host memory vs device memory.
	•	PCIe vs NVLink vs InfiniBand.
	•	Latency vs bandwidth.
	•	FLOPs vs memory bandwidth.
	•	Collective communication: all-reduce, reduce-scatter, all-gather, all-to-all, broadcast.

For performance engineering, this distinction matters constantly:

Training is often dominated by compute + communication overlap.
Inference is often dominated by memory bandwidth, KV-cache movement, scheduling, and tail latency.

⸻

Phase 1 — Single-GPU LLM training deeply

Do not start with distributed training. Start by understanding the full memory and compute budget on one GPU.

Learn
	•	Transformer compute cost.
	•	Parameter memory.
	•	Optimizer state memory.
	•	Activation memory.
	•	Gradient memory.
	•	BF16 vs FP16 vs FP32.
	•	AdamW memory overhead.
	•	Gradient checkpointing.
	•	FlashAttention.
	•	Sequence length scaling.
	•	Batch size vs microbatch size.
	•	Tokens/sec as the core training metric.

Build

Train a small GPT, for example 100M–1B parameters depending on GPU access.

Instrument:
	•	Tokens/sec.
	•	GPU memory allocated/reserved.
	•	MFU: model FLOPs utilization.
	•	Step time breakdown.
	•	Data loading time.
	•	Forward/backward/optimizer time.

You should be able to answer
	•	Why does increasing sequence length hurt more than increasing hidden size in some regimes?
	•	Why does activation memory dominate for long context?
	•	Why does AdamW use so much memory?
	•	Why does FlashAttention reduce memory traffic?
	•	What is the difference between global batch, microbatch, and gradient accumulation steps?

⸻

Phase 2 — Core distributed training

Now move from 1 GPU to many GPUs.

2.1 Data parallelism

Start here.

Learn:
	•	DDP.
	•	Gradient all-reduce.
	•	Bucketization.
	•	Communication/computation overlap.
	•	Effective batch size.
	•	Global batch = microbatch × grad accumulation × data parallel size.
	•	Stragglers.
	•	NCCL basics.

Build:

Train your small GPT with PyTorch DDP on 2–8 GPUs.

Measure:
	•	Scaling efficiency.
	•	Tokens/sec per GPU.
	•	Communication time.
	•	How performance changes with batch size.

You should understand when DDP scales well and when gradient synchronization becomes the bottleneck.

⸻

2.2 ZeRO / FSDP

Next, learn memory sharding.

Learn:
	•	ZeRO-1: shard optimizer states.
	•	ZeRO-2: shard optimizer states + gradients.
	•	ZeRO-3 / FSDP: shard parameters too.
	•	All-gather before compute.
	•	Reduce-scatter after backward.
	•	Parameter flattening.
	•	Resharding.
	•	CPU offload and why it is often slow.
	•	Checkpoint formats.

Build:

Train a model that does not fit with normal DDP but fits with FSDP or DeepSpeed ZeRO.

Libraries to use:
	•	PyTorch FSDP.
	•	DeepSpeed ZeRO.
	•	Hugging Face Accelerate only as a convenience layer, not as the thing you rely on to understand the system.

You should be able to explain:
	•	Why ZeRO-3 saves memory but introduces more communication.
	•	Why FSDP wrapping policy matters.
	•	Why small layers can cause excessive communication overhead.
	•	Why checkpointing distributed models is painful.

⸻

2.3 Tensor parallelism

Tensor parallelism splits matrix multiplications across GPUs.

Learn:
	•	Column-parallel linear layers.
	•	Row-parallel linear layers.
	•	Sharded QKV projections.
	•	Sharded MLP.
	•	All-reduce in tensor parallel layers.
	•	Sequence parallelism.
	•	Vocabulary parallel cross entropy.
	•	Why tensor parallelism usually wants fast intra-node links like NVLink.

Build:

Implement a toy tensor-parallel MLP or attention block using PyTorch distributed collectives.

Then read Megatron-LM’s tensor parallel implementation.

You should understand:
	•	Why TP is useful for very large dense models.
	•	Why TP communication happens inside the layer.
	•	Why TP across slow interconnects can be painful.
	•	Why TP degree is often limited to GPUs within a node.

⸻

2.4 Pipeline parallelism

Pipeline parallelism splits layers across devices.

Learn:
	•	Model stages.
	•	Microbatches.
	•	Pipeline bubbles.
	•	GPipe schedule.
	•	1F1B schedule.
	•	Interleaved pipeline parallelism.
	•	Activation transfer between stages.
	•	Load balancing stages.

Build:

Split a transformer into pipeline stages and train it with microbatching.

You should be able to reason about:
	•	Why more microbatches reduce bubble overhead.
	•	Why too many microbatches may hurt.
	•	Why uneven layer costs hurt utilization.
	•	Why pipeline parallelism interacts strongly with activation checkpointing.

⸻

2.5 3D / 4D / 5D parallelism

This is where Ultra-Scale Playbook becomes central.

You need to combine:
	•	Data parallelism.
	•	Tensor parallelism.
	•	Pipeline parallelism.
	•	Sequence/context parallelism.
	•	Expert parallelism for MoEs.

Nanotron’s distributed training docs describe 3D parallelism using tensor, pipeline, and data parallelism, including process groups and coordination between strategies.  ￼

Your goal is to look at a model and hardware topology and decide:
	•	TP degree.
	•	PP degree.
	•	DP/FSDP degree.
	•	Microbatch size.
	•	Gradient accumulation.
	•	Activation checkpointing policy.
	•	Whether expert parallelism is needed.
	•	Whether context parallelism is needed.

⸻

Phase 3 — MoE training

MoE is a separate beast.

Learn
	•	Dense vs sparse MoE.
	•	Experts.
	•	Router/gating network.
	•	Top-1 vs top-2 routing.
	•	Expert capacity.
	•	Token dropping.
	•	Load balancing loss.
	•	Auxiliary losses.
	•	Expert parallelism.
	•	All-to-all communication.
	•	Dropless MoE.
	•	Shared experts.
	•	Fine-grained experts.
	•	MegaBlocks / block-sparse MoE ideas.
	•	Why MoE improves parameter count without proportional compute increase.

Important systems idea

Dense model parallelism is mostly about splitting large matrix multiplications.

MoE parallelism is often about:

Route tokens → all-to-all exchange → run expert MLPs → all-to-all return.

So MoE performance is extremely sensitive to:
	•	Token imbalance.
	•	All-to-all bandwidth.
	•	Expert placement.
	•	Batch size.
	•	Sequence length.
	•	Routing distribution.
	•	Overlap of communication and expert computation.

Build

Start small:
	1.	Implement a toy MoE MLP layer.
	2.	Add top-k routing.
	3.	Add load balancing loss.
	4.	Add expert parallelism across 2–8 GPUs.
	5.	Measure token imbalance and all-to-all time.

Then compare with:
	•	Megatron-LM MoE.
	•	DeepSpeed MoE.
	•	Tutel.
	•	MegaBlocks-style implementations.

⸻

Phase 4 — Open-source training frameworks

After you know the concepts, use real libraries.

Training libraries to study

PyTorch FSDP

Best for understanding modern sharded training inside PyTorch.

Learn:
	•	FSDP wrapping.
	•	Mixed precision.
	•	State dict types.
	•	Sharded optimizer state.
	•	Activation checkpointing.
	•	Hybrid sharding.

DeepSpeed

Good for ZeRO, pipeline parallelism, and practical large-model training.

Learn:
	•	ZeRO stages.
	•	Config files.
	•	CPU/NVMe offload.
	•	Pipeline engine.
	•	MoE support.

Megatron-LM / Megatron-Core

Critical for serious pretraining systems.

Learn:
	•	Tensor parallelism.
	•	Pipeline parallelism.
	•	Sequence parallelism.
	•	Context parallelism.
	•	MoE.
	•	Distributed optimizer.
	•	Fused kernels.

Nanotron

Since you are already following Ultra-Scale Playbook, use Nanotron as a readable bridge between theory and production-style distributed training. The playbook itself says it uses Nanotron implementations as production-ready reference code.  ￼

torchtitan

Useful if you want a PyTorch-native reference for large-scale training patterns.

Practical project

Train a 1B–7B dense model using at least two frameworks: FSDP and Megatron/Nanotron. Write a report comparing memory use, throughput, configuration complexity, and failure modes.

This kind of report is excellent portfolio material.

⸻

Phase 5 — 100B-scale dense model training mental model

You may not personally have access to enough GPUs to train 100B models end-to-end, but you can still learn the engineering.

For a 100B dense model, you need to understand:
	•	Parameter memory: 100B × 2 bytes = 200 GB just for BF16 weights.
	•	Gradients: another ~200 GB.
	•	Adam states: often much larger, commonly 2 FP32 states = 800 GB.
	•	Master weights may add more depending on optimizer setup.
	•	Activations depend on batch, sequence length, hidden size, layers, checkpointing.
	•	Checkpoints are huge.
	•	Network failures become normal.
	•	Data pipeline must sustain huge token throughput.

You should learn to make training plans

Given:
	•	Model size.
	•	Sequence length.
	•	Global batch size.
	•	GPU type.
	•	GPU memory.
	•	Number of GPUs.
	•	Interconnect.
	•	Target tokens.
	•	Desired training time.

You should estimate:
	•	Memory per GPU.
	•	FLOPs per token.
	•	Tokens/sec required.
	•	Number of GPUs required.
	•	Parallelism strategy.
	•	Checkpoint frequency.
	•	Expected bottleneck.

This is one of the most important skills for pretraining companies.

⸻

Phase 6 — GPU performance engineering foundations

Now move below PyTorch.

Learn CUDA mental model

You need to understand:
	•	Threads, warps, blocks, grids.
	•	SMs.
	•	Registers.
	•	Shared memory.
	•	Global memory.
	•	L1/L2 cache.
	•	Coalesced memory access.
	•	Occupancy.
	•	Warp divergence.
	•	Memory bandwidth.
	•	Tensor Cores.
	•	WMMA / MMA.
	•	Asynchronous copies.
	•	Tiling.
	•	Double buffering.

Profiling tools

Learn:
	•	NVIDIA Nsight Systems: timeline, CPU/GPU overlap, NCCL calls.
	•	NVIDIA Nsight Compute: kernel-level analysis.
	•	PyTorch profiler.
	•	NCCL debug logs.
	•	CUDA events.
	•	nvidia-smi dmon, DCGM, cluster monitoring.

Performance concepts

You should be comfortable with:
	•	Roofline model.
	•	Arithmetic intensity.
	•	Compute-bound vs memory-bound kernels.
	•	Kernel launch overhead.
	•	Fusion.
	•	Layout transformations.
	•	Quantization effects.
	•	Benchmark variance.
	•	Warmup.
	•	CUDA graphs.

⸻

Phase 7 — DSLs, Triton, CUTLASS, CuTe DSL

You said you have no idea what DSLs are, so here is the clean mental model.

What is a DSL?

A domain-specific language is a programming language designed for a specific problem domain.

Examples:
	•	SQL is a DSL for querying data.
	•	Regex is a DSL for string matching.
	•	Triton is a DSL for writing GPU kernels.
	•	CuTe DSL is a DSL for writing high-performance CUDA kernels using CUTLASS/CuTe abstractions.

The point is not “new language for fun.” The point is:

A DSL gives you higher-level concepts for a domain while still allowing the compiler/runtime to generate efficient low-level code.

Triton

Triton is a Python-based language and compiler for writing custom DNN GPU kernels aimed at high throughput on modern GPUs.  ￼

Learn Triton after you understand basic CUDA concepts.

Start with:
	•	Vector add.
	•	Matrix multiplication.
	•	LayerNorm.
	•	RMSNorm.
	•	Softmax.
	•	Fused bias + activation.
	•	Simple attention-like kernels.

Then move to:
	•	Blocked matmul.
	•	Persistent matmul.
	•	FP8 kernels.
	•	Quantized matmul.
	•	Fused MoE kernels.
	•	Paged attention kernels.

CUTLASS

CUTLASS is NVIDIA’s template library for high-performance CUDA GEMM and related kernels.

You do not need to master all of CUTLASS immediately. But if you want to be a serious GPU performance engineer, you should understand:
	•	GEMM tiling.
	•	Threadblock/warp/instruction tile hierarchy.
	•	Epilogues.
	•	Tensor Core layouts.
	•	Memory layouts.
	•	Hopper/Blackwell-specific features.

CuTe DSL

CuTe DSL is NVIDIA’s Python DSL in the CUTLASS ecosystem. NVIDIA describes CUTLASS 4 as adding Python-native DSL interfaces for writing high-performance CUDA kernels based on CUTLASS and CuTe concepts, while reducing compile times and avoiding deep C++ boilerplate.  ￼

The official CuTe DSL docs cover JIT compilation, layouts, integration with frameworks, debugging, autotuning, and AOT compilation.  ￼

A good progression:
	1.	CUDA basics.
	2.	Triton basics.
	3.	CUTLASS concepts.
	4.	CuTe layout algebra.
	5.	CuTe DSL kernels.
	6.	Compare Triton vs CuTe DSL for the same kernel.

Rule of thumb

Use this mental map:

Tool	Best for
PyTorch	Model-level experimentation
torch.compile	Graph capture and compiler-level optimization
Triton	Custom kernels with Python ergonomics
CUDA C++	Full control
CUTLASS	Production-grade GEMM-style kernels
CuTe DSL	CUTLASS/CuTe-style kernel programming with Python-native workflow
TVM / MLIR / XLA	Compiler-stack work


⸻

Phase 8 — Inference engine fundamentals

Training and inference are different systems.

For inference, learn:
	•	Prefill phase.
	•	Decode phase.
	•	KV cache.
	•	Paged KV cache.
	•	Continuous batching.
	•	Request scheduling.
	•	Token latency.
	•	Time to first token.
	•	Inter-token latency.
	•	Throughput: tokens/sec.
	•	Tail latency: p95/p99.
	•	Tensor parallel inference.
	•	Pipeline parallel inference.
	•	Expert parallel inference for MoEs.
	•	Quantization.
	•	Speculative decoding.
	•	Prefix caching.
	•	Chunked prefill.
	•	Disaggregated prefill/decode.
	•	Multi-LoRA serving.
	•	CUDA graphs.

vLLM is directly relevant here. Its docs describe it as a fast, easy-to-use LLM inference and serving library, with topics including distributed inference, PagedAttention, quantization, automatic prefix caching, speculative decoding, and performance tuning.  ￼

SGLang is also directly relevant: its docs describe it as a high-performance serving framework for large language and multimodal models, designed for low-latency and high-throughput inference from single-GPU setups to large distributed clusters.  ￼

Core inference project

Build a toy inference engine:
	1.	Load a small transformer.
	2.	Implement greedy decoding.
	3.	Add batching.
	4.	Add continuous batching.
	5.	Add KV cache.
	6.	Add paged KV cache.
	7.	Add prefix caching.
	8.	Add tensor-parallel inference.
	9.	Add benchmark harness.
	10.	Compare against vLLM/SGLang on the same model.

You will learn much more from building a bad mini-vLLM than from only reading vLLM.

⸻

Phase 9 — Read and contribute to vLLM and SGLang

vLLM contribution path

The official vLLM contributing docs say contributions can include bug reports, model support, features, documentation, how-to guides, answering questions, PR reviews, and community support.  ￼

Start with:
	1.	Run vLLM locally.
	2.	Serve a small model.
	3.	Benchmark with different batch sizes and sequence lengths.
	4.	Enable prefix caching.
	5.	Try speculative decoding.
	6.	Try tensor parallelism.
	7.	Read scheduler code.
	8.	Read KV cache manager code.
	9.	Read attention backend code.
	10.	Reproduce one open performance issue.

Good first contributions:
	•	Add docs for confusing performance flags.
	•	Add benchmark coverage.
	•	Fix a flaky test.
	•	Add support for a small model architecture.
	•	Improve error messages.
	•	Add a regression test for a scheduler/KV-cache bug.

Harder contributions:
	•	Kernel optimization.
	•	New attention backend.
	•	Better scheduling policy.
	•	Prefix caching improvement.
	•	Multi-node performance improvement.
	•	Quantization kernel support.

SGLang contribution path

SGLang’s contribution guide points new contributors to Mini-SGLang, code walkthroughs, and optimization/benchmarking/profiling materials.  ￼

SGLang’s GitHub repo is active and substantial, with Python, Rust, CUDA, C++, and docs components.  ￼

Start with:
	1.	Run SGLang locally.
	2.	Serve one model.
	3.	Run benchmarks.
	4.	Read Mini-SGLang.
	5.	Understand the frontend/runtime split.
	6.	Understand scheduling.
	7.	Understand memory pool / KV cache management.
	8.	Trace one request from API entrypoint to token output.
	9.	Reproduce a benchmark from docs.
	10.	Pick a small issue.

Good first contributions:
	•	Documentation.
	•	Benchmark scripts.
	•	Model support.
	•	Reproducible bug reports.
	•	Small scheduler fixes.
	•	Profiling writeups.

⸻

Phase 10 — Post-training systems

You also mentioned post-training companies. For those, learn:

SFT
	•	Dataset packing.
	•	Loss masking.
	•	Chat templates.
	•	Sequence packing.
	•	FSDP/ZeRO for finetuning.
	•	LoRA / QLoRA.
	•	Multi-node SFT.

Preference optimization
	•	DPO.
	•	IPO.
	•	KTO.
	•	ORPO.
	•	Reward models.
	•	Pairwise data.
	•	KL penalties.
	•	Reference model memory cost.

RLHF / RLVR systems
	•	PPO-style RLHF.
	•	GRPO-style methods.
	•	Rollout generation.
	•	Policy model.
	•	Reference model.
	•	Reward model/verifier.
	•	Inference-training loop.
	•	Asynchronous generation.
	•	Checkpoint handoff.
	•	Serving engines inside training loops.
	•	Data freshness.

Systems angle

Post-training performance engineering is often about:
	•	Generating rollouts quickly.
	•	Efficiently serving the policy model.
	•	Avoiding idle trainers.
	•	Managing many model copies.
	•	Fast checkpoint loading.
	•	Efficient LoRA or adapter training.
	•	Packing and masking correctly.
	•	Debugging instability.

This is where training and inference knowledge meet.

⸻

12-month roadmap

Assuming 10–15 focused hours/week.

Months 1–2: Single-GPU LLM + PyTorch distributed basics

Deliverables:
	•	Tiny GPT from scratch.
	•	Training loop with BF16, checkpointing, gradient accumulation.
	•	DDP training on multiple GPUs.
	•	Written notes on memory and compute breakdown.

Resources:
	•	PyTorch distributed docs.
	•	FlashAttention paper/code.
	•	Ultra-Scale Playbook first sections.  ￼

⸻

Months 3–4: FSDP, ZeRO, tensor parallelism, pipeline parallelism

Deliverables:
	•	FSDP training run.
	•	DeepSpeed ZeRO run.
	•	Toy tensor-parallel linear/MLP.
	•	Toy pipeline-parallel transformer.
	•	Scaling report.

You should be able to answer interview-style questions like:
	•	“Why does ZeRO-3 reduce memory but increase communication?”
	•	“When would you use TP instead of FSDP?”
	•	“Why is pipeline parallelism hard to tune?”
	•	“How do you choose microbatch size?”

⸻

Months 5–6: Megatron/Nanotron-style training and MoE

Deliverables:
	•	Train a small model with Nanotron or Megatron.
	•	Implement toy MoE.
	•	Implement expert parallel all-to-all.
	•	Profile MoE token imbalance.
	•	Write a “MoE training bottlenecks” report.

Focus:
	•	TP + PP + DP interactions.
	•	Sequence parallelism.
	•	Expert parallelism.
	•	All-to-all bottlenecks.
	•	Load balancing.

⸻

Months 7–8: GPU kernels and profiling

Deliverables:
	•	Triton vector add, matmul, softmax, RMSNorm.
	•	Triton fused MLP or fused activation.
	•	Nsight Systems profile of training step.
	•	Nsight Compute analysis of one kernel.
	•	Compare PyTorch eager vs compiled vs Triton kernel.

Resources:
	•	Triton docs.  ￼
	•	CUTLASS/CuTe DSL docs.  ￼

⸻

Months 9–10: Inference engine internals

Deliverables:
	•	Toy inference engine with KV cache.
	•	Add continuous batching.
	•	Add paged KV cache.
	•	Benchmark TTFT, ITL, throughput.
	•	Compare against vLLM and SGLang.

Focus:
	•	Scheduler.
	•	KV cache allocator.
	•	Attention backend.
	•	Prefix caching.
	•	Chunked prefill.
	•	Speculative decoding.
	•	Quantization.

⸻

Months 11–12: Open-source contribution + portfolio

Deliverables:
	•	2–5 merged PRs across vLLM/SGLang/Nanotron/Megatron-related repos.
	•	One serious technical blog post.
	•	One benchmark report.
	•	One kernel optimization writeup.
	•	One distributed training scaling report.

Your portfolio should show:
	•	You can profile.
	•	You can measure correctly.
	•	You can read large systems.
	•	You can improve performance.
	•	You understand training and inference tradeoffs.
	•	You can communicate results clearly.

⸻

What to build for your portfolio

Project 1: Distributed training scaling report

Train the same small transformer with:
	•	DDP.
	•	FSDP.
	•	DeepSpeed ZeRO.
	•	Megatron/Nanotron if available.

Report:
	•	Tokens/sec.
	•	Peak memory.
	•	Scaling efficiency.
	•	Communication overhead.
	•	Failure modes.
	•	Config complexity.

Project 2: Toy MoE with expert parallelism

Implement:
	•	Top-k router.
	•	Expert MLPs.
	•	Load balancing loss.
	•	All-to-all token dispatch.
	•	Expert parallelism.

Report:
	•	Token imbalance.
	•	All-to-all cost.
	•	Expert utilization.
	•	Throughput vs dense baseline.

Project 3: Mini inference engine

Implement:
	•	KV cache.
	•	Batching.
	•	Continuous batching.
	•	Paged KV cache.
	•	Prefix cache.

Report:
	•	TTFT.
	•	Inter-token latency.
	•	Throughput.
	•	Memory fragmentation.
	•	Comparison with vLLM/SGLang.

Project 4: Triton kernel series

Implement and benchmark:
	•	RMSNorm.
	•	Softmax.
	•	Matmul.
	•	Fused SwiGLU.
	•	Quantized matmul if ambitious.
	•	Simple attention kernel if very ambitious.

Report:
	•	Baseline PyTorch vs Triton.
	•	Bandwidth achieved.
	•	FLOPs achieved.
	•	Bottleneck analysis.

⸻

Interview readiness checklist

You should be able to explain these without handwaving.

Distributed training
	•	DDP vs FSDP vs ZeRO.
	•	TP vs PP vs DP.
	•	Sequence parallelism.
	•	Context parallelism.
	•	Expert parallelism.
	•	All-reduce vs reduce-scatter vs all-gather vs all-to-all.
	•	NCCL topology issues.
	•	Activation checkpointing tradeoffs.
	•	Optimizer state sharding.
	•	Distributed checkpointing.
	•	MFU and scaling efficiency.
	•	How to debug a slow training job.

MoE
	•	Why MoE is sparse compute.
	•	Router design.
	•	Load balancing.
	•	Expert capacity.
	•	Expert parallelism.
	•	All-to-all bottleneck.
	•	Dropped tokens.
	•	Shared experts.
	•	MoE inference challenges.

Inference
	•	Prefill vs decode.
	•	KV cache.
	•	Paged attention.
	•	Continuous batching.
	•	Chunked prefill.
	•	Prefix caching.
	•	Speculative decoding.
	•	Quantization.
	•	Tensor parallel inference.
	•	Disaggregated serving.
	•	TTFT vs ITL vs throughput vs p99 latency.

GPU kernels
	•	CUDA execution hierarchy.
	•	Memory coalescing.
	•	Shared memory.
	•	Tensor Cores.
	•	Tiling.
	•	Occupancy.
	•	Arithmetic intensity.
	•	Kernel fusion.
	•	Triton programming model.
	•	CUTLASS/CuTe concepts.

Production engineering
	•	Benchmark methodology.
	•	Reproducibility.
	•	Profiling.
	•	Regression testing.
	•	CI.
	•	Reading unfamiliar code.
	•	Writing minimal PRs.
	•	Communicating performance results.

⸻

Recommended learning order

Do not learn everything in parallel. Use this order:
	1.	Single-GPU transformer training.
	2.	DDP.
	3.	FSDP/ZeRO.
	4.	Tensor parallelism.
	5.	Pipeline parallelism.
	6.	Megatron/Nanotron.
	7.	MoE training.
	8.	CUDA mental model.
	9.	Triton.
	10.	CUTLASS/CuTe DSL.
	11.	Inference basics.
	12.	vLLM internals.
	13.	SGLang internals.
	14.	Open-source contributions.
	15.	Advanced post-training systems.

⸻

How to use Ultra-Scale Playbook

Use it as your spine, but not your only resource.

For each chapter:
	1.	Read the concept.
	2.	Reproduce a tiny version yourself.
	3.	Find the corresponding Nanotron code.
	4.	Run a small experiment.
	5.	Write down:
	•	What memory was saved?
	•	What communication was added?
	•	What throughput changed?
	•	What would happen at 100B scale?

The key is to avoid passive reading. Every distributed training concept should become a small experiment.

⸻

Practical weekly cadence

A good weekly loop:

Day 1–2: Read one concept.
Day 3–4: Implement or run a small experiment.
Day 5: Profile it.
Day 6: Read production code for the same concept.
Day 7: Write a short note or benchmark report.

By the end of a year, those notes become your interview prep and public portfolio.

⸻

Final target

You are “ready” for LLM performance engineering roles when you can do this:

Given a 100B dense or MoE model, a hardware cluster, and a target throughput, you can propose a parallelism strategy, estimate memory and communication costs, identify likely bottlenecks, run/propose benchmarks, profile the system, and make code-level improvements in training or inference frameworks.

That is the skillset pretraining, post-training, and inference companies care about.