# 03 — Numeric Formats

Modern LLM articles throw around "FP8", "BF16", "MXFP4", "NVFP4" as if everyone knows what they mean. They are not the same family of trade-offs. This file walks each in turn.

The throughline: each new format trades **precision** and/or **range** for **bandwidth** and **compute**. The reason you see so many is that LLMs tolerate astonishingly aggressive precision reduction *if you scale properly*.

---

## The IEEE family

```
              total  | sign | exponent | mantissa | dynamic range (powers of 2)
─────────────────────────────────────────────────────────────────────────
FP64 (double)   64  |  1   |   11     |   52     | ~2^-1022 .. 2^1024
FP32 (single)   32  |  1   |    8     |   23     | ~2^-126  .. 2^128
FP16 (half)     16  |  1   |    5     |   10     | ~2^-14   .. 2^16
BF16             16  |  1   |    8     |    7     | ~2^-126  .. 2^128  (FP32 range, FP16 width)
TF32             19  |  1   |    8     |   10     | (uses FP32 layout, with 23-bit mantissa truncated)
```

**FP16 problem:** narrow range. Activations and gradients in FP16 routinely underflow. Hence `GradScaler`.

**BF16:** keeps FP32's exponent (range), drops most of the mantissa. Less precise, but no underflow. Workhorse of modern LLM training.

**TF32:** Ampere-only "internal" format. Tensor cores read FP32 inputs, internally truncate the mantissa to 10 bits (BF16-precision, FP32-range), accumulate in FP32. Lets you keep an FP32-typed API with tensor-core throughput. Used by `torch.set_float32_matmul_precision("high")`.

---

## FP8: two variants

H100+ supports two FP8 formats:

```
                    | sign | exp | mantissa | range          | use case
────────────────────────────────────────────────────────────────────────────
E4M3 (range/precision balance) | 1  |  4  |  3       | ~2^-9 .. 2^9   | activations, weights
E5M2 (more range)              | 1  |  5  |  2       | ~2^-16 .. 2^15 | gradients
```

You usually want **E4M3 for forward** (activations and weights) and **E5M2 for backward** (gradients, which need range more than precision).

**Per-tensor scaling.** FP8 has only 256 distinct values. You can't blindly cast a BF16 activation tensor to FP8 and hope. You track a *per-tensor scale*:

```
x_fp8 = round(x_bf16 / scale)              // store FP8
x_used = x_fp8 * scale                     // dequant during compute
```

The scale floats with the tensor's magnitude. NVIDIA's "Transformer Engine" library automates this — keeps an exponential moving average of `max(|x|)` and updates `scale` per step.

**Throughput.** FP8 tensor cores are ~2× FP16/BF16 because the operands are half as wide.

**Quality.** Done right (Transformer Engine), FP8 training gives quality within ~1% of BF16 for typical LLMs. Done wrong (no scaling), you lose far more.

---

## Block-scaled formats (MX)

The Microscaling (MX) family — standardized in 2023 by an industry consortium (NVIDIA + AMD + Intel + Arm + Microsoft + Meta + Qualcomm) — generalizes per-tensor scaling to **per-block scaling**:

- Each block of (usually) 32 consecutive elements gets its **own** shared exponent (a single FP8 E8M0 scale).
- The 32 elements themselves are stored in some sub-byte format.

The two important variants:

```
MXFP8    : 32 elements × FP8 (E4M3) + 1 shared E8M0 scale     | ~9 bits/elem
MXFP6    : 32 elements × FP6 (E2M3 or E3M2) + 1 scale          | ~7 bits/elem
MXFP4    : 32 elements × FP4 (E2M1) + 1 shared E8M0 scale      | ~5 bits/elem
NVFP4    : NVIDIA variant of MXFP4 with smaller block (~16)    | similar
```

**Why MXFP wins:** per-block scales handle outliers far better than per-tensor scales. A weight matrix has wide regions of small values and rare large outliers; per-block scaling adapts to both.

**Hardware support:** Blackwell's TCGen05 reads MX operands directly — no software-side dequant. Each block's scale is fetched alongside the data, applied in the MMA unit. This is most of why Blackwell's FP4 numbers look the way they do.

**NVFP4 vs MXFP4:** NVIDIA's "NVFP4" uses a smaller block (16 elements vs 32) for finer-grained scaling. Marginally better quality, slightly more scale overhead. Press articles often blur the two.

---

## INT formats

```
            | range            | use case
─────────────────────────────────────────────────────
INT8         -128 .. 127        weight-only or full quantized inference (SmoothQuant, etc.)
INT4         -8 .. 7            GPTQ, AWQ weight-only quantization
```

Integer formats don't have a "shared exponent" in the IEEE sense. Quantization frameworks supply their own scale + zero-point per channel or per block.

INT8 tensor cores are widely supported (Turing+). INT4 is more niche; on Ampere you typically dequantize INT4 weights to BF16 inside the kernel and do a BF16 GEMM.

---

## Sparsity, briefly

Orthogonal to dtype: **2:4 structured sparsity** (Ampere+) requires that in every group of 4 consecutive weights along an axis, at most 2 are nonzero. Hardware skips the zero multiplies. 2× speedup on supported dtypes.

Mostly an inference optimization. Training to enforce this pattern requires masked training; doable but not common.

---

## When to use what

| Workload | Recommended dtype |
|---|---|
| LLM pretraining (default) | BF16 mixed precision + FP32 master + FP32 optimizer |
| LLM pretraining (aggressive) | FP8 (E4M3 fwd, E5M2 bwd) with Transformer Engine; FP32 master |
| LLM SFT | BF16 |
| LoRA training | BF16 for LoRA, QLoRA can use 4-bit base |
| Inference, quality-sensitive | BF16 weights, BF16 activations |
| Inference, throughput | INT8 weight-only, BF16 activations |
| Inference, aggressive | INT4 (GPTQ/AWQ) weights, BF16 activations |
| Inference, Blackwell, most aggressive | MXFP4 / NVFP4 weights, MXFP8 or BF16 activations |
| KV cache | FP8 (E5M2 for headroom) on Hopper+; INT8 on Ampere |

---

## "FLOPS" by dtype is not a fair comparison

A common press confusion. When NVIDIA says:

> Blackwell: 9000 TFLOPS FP4

This is the *peak rate at which Blackwell can issue FP4 multiply-adds*. It is **not** comparable to "Hopper: 989 TFLOPS BF16" in terms of equivalent ML work, because:

- FP4 is 1/4 the bits per operand.
- FP4 needs careful per-block scaling, which limits batch / sequence lengths in some regimes.
- A model that converges at BF16 may *not* converge at FP4 without careful methodology.

The right comparison is "tokens/sec at equivalent quality" — which is what production engineers care about.

---

## What to read next

- **OCP MX format specification** — the standard for MXFP8/6/4. Free PDF.
- **Transformer Engine docs (NVIDIA)** — practical FP8 training.
- **Llama-3.1-405B-Instruct technical report** — discusses FP8 inference quality.
- **DeepSeek-V3 report** — heavy FP8 training discussion.
- **AWQ paper** (Lin et al., 2023) — INT4 quantization with activation awareness.
- Move on to `04-ptx-and-sass.md`.

---

## Self-check

1. Why is BF16 the LLM training default and not FP16?
2. What's the difference between E4M3 and E5M2 FP8, and when to use each?
3. What does "per-block scale" mean in MXFP4, and why does it beat per-tensor scaling?
4. Why isn't "9000 TFLOPS FP4" 9× faster than "989 TFLOPS BF16" for real workloads?
5. What's the difference between MXFP4 and NVFP4?
6. When does INT4 weight-only quantization stop helping (i.e., when does the bottleneck move elsewhere)?
