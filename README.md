# llm-systems-tutorial

A hands-on course taking a competent PyTorch user to *LLM systems / performance engineer* — comfortable contributing to vLLM, SGLang, Megatron-style projects.

- **Start here:** [`COURSE.md`](COURSE.md) — full chapter map, learning loop, conventions.
- **Source roadmap:** [`ROADMAP.md`](ROADMAP.md) — the unstructured notes the course was built from.
- **Chapters:** [`chapters/`](chapters/) — 19 numbered folders, each self-contained.
- **Reading track:** [`reading-track/`](reading-track/) — hardware-fluency material for reading SemiAnalysis / NVIDIA whitepapers / Hot Chips talks. Read alongside the chapters; not a prerequisite for them.
- **Shared utilities:** [`shared/utils/`](shared/utils/) — timing, memory, NCCL bandwidth helpers reused across chapters.
- **Your scratchpad:** [`tasks/lessons.md`](tasks/lessons.md) — write down every correction so the same mistake doesn't ship twice.

Target hardware: **4×A100-80GB, single node, NVLink/NVSwitch interconnect.** Most chapters degrade gracefully on smaller setups.

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .
cd chapters/00-setup && python verify_environment.py
```
