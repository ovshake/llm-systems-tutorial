# LLM Systems Tutorial — Course Build Plan

This is the build plan for the hands-on course derived from `ROADMAP.md`.

## Target learner

Strong PyTorch user with an 8×H100-80GB node who wants to ship code in vLLM / SGLang / Megatron-style projects within ~12 months. Course covers:
distributed training, GPU perf engineering, kernels (Triton/CUTLASS/CuTe), inference engines, and post-training systems.

## Build checklist

- [x] Decide chapter structure (19 chapters, 00–18)
- [x] Lay out folder skeleton
- [x] Write `tasks/todo.md`
- [x] Write `tasks/lessons.md`
- [x] Write root `COURSE.md` (course-wide overview, learning loop, conventions, hardware notes)
- [x] Write root `pyproject.toml`
- [x] Write root `README.md`
- [x] Replace `.gitignore` with a Python-oriented one
- [x] Write `shared/utils/` helpers used across chapters (timing, memory, profiling, NCCL bandwidth)
- [x] Chapter 00 — Setup & GPU sanity checks
- [x] Chapter 01 — Transformer from scratch (single GPU)
- [x] Chapter 02 — Memory & compute budgeting
- [x] Chapter 03 — Data parallelism (DDP)
- [x] Chapter 04 — FSDP & ZeRO
- [x] Chapter 05 — Tensor parallelism
- [x] Chapter 06 — Pipeline parallelism
- [x] Chapter 07 — 3D parallelism with Nanotron / Megatron
- [x] Chapter 08 — MoE training
- [x] Chapter 09 — GPU performance engineering
- [x] Chapter 10 — Triton kernels
- [x] Chapter 11 — CUTLASS & CuTe DSL
- [x] Chapter 12 — Inference engine fundamentals
- [x] Chapter 13 — Paged attention & prefix caching
- [x] Chapter 14 — Advanced inference (speculative, quant, disagg)
- [x] Chapter 15 — vLLM deep dive & contributions
- [x] Chapter 16 — SGLang deep dive & contributions
- [x] Chapter 17 — Post-training systems (SFT / DPO / RLHF / GRPO)
- [x] Chapter 18 — Capstone: 100B-scale plan + portfolio writeups

## Chapter template

Every chapter folder has:

1. `README.md` — *reading* block (concepts, references, deliverables, exercises, self-check, stretch goals).
2. Starter Python files where it makes sense (`model.py`, `train.py`, etc.). These are *naive but runnable baselines* the learner enhances.
3. `exercises.md` (optional) for chapters with many sub-exercises.
4. Scripts for benchmarks/profiling specific to the chapter.
5. Each chapter ends with a "Deliverables" section that defines what to commit before moving on.

## Pacing

12 months at ~12 h/week ≈ 600 hours total. Rough budget per chapter:
- Foundational (00–02): 1–2 weeks each
- Distributed training (03–07): 3–4 weeks each
- MoE (08): 3–4 weeks
- Kernels (09–11): 4–6 weeks each
- Inference (12–14): 3–4 weeks each
- OSS deep dives (15–16): 4–6 weeks each
- Post-training (17): 4 weeks
- Capstone (18): ongoing, write as you go

## Review section

Course v1 is built. Summary:

- **19 chapter folders** (00–18) under `chapters/`, each with a substantive `README.md`.
- **Starter Python code** for the foundational implementation chapters (00, 01, 02, 03, 04, 05, 08, 10, 12, 13). Later chapters lean on the learner using real frameworks (Nanotron, Megatron, vLLM, SGLang, TRL) rather than reimplementing.
- **Shared utilities** in `shared/utils/`: timing (CUDA-event-based), memory accounting, MFU estimation, NCCL collectives benchmark, model presets (tiny → 7B).
- **Root files**: `COURSE.md` (course-wide overview), `README.md` (one-paragraph onboarding), `pyproject.toml` (uv-managed deps), `.gitignore` (Python + training artifacts).

What the learner should do next:
1. Read `COURSE.md`.
2. Set up environment, run `chapters/00-setup/verify_environment.py`.
3. Work through chapters in order. The order matters; later chapters explicitly defer concepts to earlier "canonical" chapters.
4. Write a `NOTES.md` per chapter. Keep `tasks/lessons.md` updated.

Areas for future iteration:
- Add starter code for Chapters 06, 07, 09 (left as exercises so learners do their own scaffolding).
- Add `requirements.txt` per chapter that needs special deps (DeepSpeed for 04, Nanotron for 07, flash-attn for 02+, Triton for 10+).
- Add `NOTES_TEMPLATE.md` showing the expected format for chapter notes.
- The MoE chapter (08) could use full EP scaffolding once the learner has done the toy version.
