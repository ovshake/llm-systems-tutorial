# Chapter 17 — Post-Training Systems (SFT, DPO, RLHF / GRPO)

**Hardware target:** 8× H100-80GB.
**Time budget:** 4 weeks.
**Prerequisites:** Chapter 04 (FSDP) and Chapter 12 (inference engine). You'll combine them.

---

## Why this chapter exists

Pretraining produces a base model. Post-training turns it into a useful model. The systems work in post-training is *different* from pretraining:

- **SFT** is just supervised training, but with packing, masking, and chat-template subtleties.
- **DPO / IPO / KTO / ORPO** keep a *reference model* in memory at all times — your memory footprint roughly doubles. Different optimizer / loss math but same training shape.
- **RLHF / GRPO** is the most systems-interesting: it combines *training* and *inference* in the same loop, with rollout generation, reward / verifier evaluation, and policy gradient updates. The dominant cost is usually rollout generation, so you have to be able to run an inference engine inside your training loop.

Post-training is where training and inference systems meet. Many of the most lucrative post-training companies (Surge, Scale, Anthropic's post-training team, etc.) need engineers who can both serve and train.

By the end you will:

1. Have run SFT with proper packing / masking on a chat dataset.
2. Have run DPO on a preference dataset with an explicit reference model.
3. Have implemented GRPO-style rollout generation inside a training loop.
4. Understand the system architecture of "rollout-trainer-verifier" pipelines.

---

## Read first

- **InstructGPT paper** (Ouyang et al., 2022) — RLHF.
- **DPO paper** (Rafailov et al., 2023).
- **IPO** (Azar et al., 2023), **KTO** (Ethayarajh et al., 2024), **ORPO** (Hong et al., 2024).
- **GRPO** (DeepSeekMath / DeepSeek-R1 papers).
- **Open Instruct (AllenAI)** and **TRL (HuggingFace)** source code — production-quality SFT / DPO / RLHF training in Python.
- **Open RLHF**, **OpenRLHF**, **veRL**, **NeMo-Aligner** — RL training frameworks.
- **Tülu** papers — for what a state-of-the-art post-training pipeline looks like end-to-end.

---

## Concepts

### 1. SFT

Supervised Fine-Tuning. Take a base model, train on (prompt, response) pairs with cross-entropy on response tokens *only* (mask out prompt tokens from the loss).

**Packing**: To avoid padding waste, concatenate multiple short examples into one sequence with a separator token, attention mask, and per-example position resets. Memory and FLOPs are now used on real tokens, not padding. Most production SFT runs use packing.

**Chat templates**: The model's tokenizer ships with a chat template (e.g., Llama-3 uses `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n...`). Apply it consistently or your model won't learn to follow the template.

**FSDP for SFT**: Use the same FSDP setup as pretraining. Often LoRA / QLoRA for cheaper fine-tuning.

### 2. LoRA / QLoRA

Low-Rank Adaptation: train only a low-rank update to each linear layer:

```
W_new x = W_base x + B(A x)        # A: (r, in), B: (out, r), with r << min(in, out)
```

Trains far fewer parameters (~0.1–1% of base). QLoRA quantizes the base to 4-bit and trains the LoRA in BF16 — a 7B model + LoRA fits on a single 24 GB GPU.

For production-quality SFT, full FT often beats LoRA. For experimentation / data efficiency, LoRA is the right starting point.

### 3. DPO and friends

Direct Preference Optimization. Train on (prompt, preferred_response, rejected_response) tuples. Loss:

```
L = -log σ(β * [log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)])
```

where `π` is the policy (being trained) and `π_ref` is the reference (frozen). The interpretation: maximize the log-ratio between policy and reference on the preferred response, minus the same on the rejected response. β is a temperature.

**System implications**: Two forward passes per step (policy + reference). Two model copies in memory. With FSDP for the policy and the reference, your memory budget doubles. Tricks:

- Reference can be quantized (it's only used for forward).
- Reference can be on CPU (still doubles memory but not VRAM).
- For LoRA-SFT base models: reference = base (no LoRA), policy = base + LoRA. Same weights, swap the adapter in and out. The cheapest version.

IPO / KTO / ORPO have different losses but the same memory profile (with one exception — ORPO doesn't need a reference, so it's leaner).

### 4. PPO-style RLHF

The classical RLHF pipeline:

```
1. Initialize policy π_θ (= SFT model) and reference π_ref (= SFT, frozen).
2. Train a reward model R (separately, on preferences).
3. For each iteration:
   a. Sample prompts.
   b. Roll out responses from π_θ. (← This is inference!)
   c. Score with R: r = R(prompt, response).
   d. Compute PPO-style update on π_θ using r, with a KL penalty against π_ref.
```

The systems pattern: rollouts dominate the wall-clock (usually 60–90% of total time). So *the rollout engine needs to be a real inference engine*. Training tooling that doesn't acknowledge this (e.g., uses HuggingFace's `model.generate()`) is unusable at scale.

### 5. GRPO

Group Relative Policy Optimization (DeepSeek). Simpler than PPO:

- For each prompt, sample G responses (a "group").
- Score each response with a reward / verifier.
- Compute advantage as `(r_i - mean(r_group)) / std(r_group)`.
- Update policy with a policy-gradient term + KL penalty.

No value model needed (which PPO requires). Much simpler. But still rollout-dominated.

### 6. The asynchronous rollout pattern

Naive: trainer generates rollouts synchronously, then steps optimizer. Inefficient because the trainer GPUs are idle during rollouts and vice versa.

Asynchronous: dedicated rollout engine pulls policy weights from trainer at regular intervals, generates rollouts continuously, pushes them to a queue. Trainer consumes from the queue.

The challenges:
- **Staleness**: rollouts were generated with a slightly older policy. Standard practice tolerates 1–5 policy versions of staleness; more starts hurting.
- **Weight transfer**: NCCL broadcast from trainer to rollout group works, but is bandwidth-heavy. Several systems use offline checkpoints + reload.
- **Verifier latency**: code-execution verifiers, search-based verifiers, etc., can take seconds per response. Need their own queue.

### 7. End-to-end pipeline shapes

```
Trainer ──policy weights──▶ Rollout engine ──responses──▶ Verifier ──rewards──▶ Trainer
   ▲                                                                                │
   └────────────────────────────  policy update ──────────────────────────────────┘
```

Different frameworks make different choices:
- **TRL** — synchronous, simple, HF-flavored. Use for prototyping.
- **OpenRLHF / Open Instruct** — async patterns.
- **veRL** — newer, more aggressively async, vLLM as the rollout backend.
- **NeMo-Aligner** — NVIDIA's, Megatron-based.

---

## Hands-on

### Exercise 1 — SFT with packing and masking

In [`sft_packing.py`](sft_packing.py):

1. Load a small chat dataset (e.g., `databricks/databricks-dolly-15k`).
2. Apply the model's chat template.
3. Pack multiple examples into 4k-token sequences using `EOS` separator + position-id resets.
4. Train your `small` or `medium` preset for a few hundred steps with cross-entropy on response tokens only (prompt tokens have `label = -100` and contribute zero loss).

Run on 8 GPUs with FSDP from Chapter 04.

**Deliverable:** Loss curve. Generate a sample response after training — should follow the chat template.

### Exercise 2 — LoRA on top of SFT

Adapt `sft_packing.py` to train a LoRA instead of full FT. Use `peft.LoraConfig` for simplicity, or implement the low-rank update by hand. Aim for ~1% of parameters trainable.

Compare quality (qualitative) and training time vs full FT.

**Deliverable:** Loss curve + memory comparison.

### Exercise 3 — DPO

In [`dpo.py`](dpo.py):

1. Load a preference dataset (`Anthropic/hh-rlhf` or `HuggingFaceH4/ultrafeedback_binarized`).
2. SFT model = policy initialization = reference (frozen).
3. Run the DPO loss with β=0.1 for a few hundred steps.

Two forward passes per step. Memory roughly doubles.

**Deliverable:** Loss curve. Compare policy responses to reference responses on a holdout prompt.

### Exercise 4 — DPO with LoRA (memory-light)

Modify Exercise 3 so:
- The base model is shared between policy and reference.
- The "policy" is base + active LoRA.
- The "reference" is base alone (the LoRA is disabled / zeroed during the reference forward).

Memory now only adds a small LoRA delta, not a full second model.

**Deliverable:** Memory comparison vs Exercise 3.

### Exercise 5 — Rollout engine integration

Hook up your Chapter 12–14 inference engine as the rollout engine for an RL training loop. Architecture:

```
[Trainer process on GPUs 0–3, FSDP]
   policy.state_dict()  →  serialize  →  shared file or NCCL bcast

[Rollout process on the same GPUs OR a subset]
   inference engine loads new weights periodically
   pulls prompts from a queue, generates responses, pushes to results queue
```

For this exercise, run synchronously: trainer step → trainer pushes weights → trainer triggers rollouts → trainer waits → rollouts done → trainer does next step.

This is the simplest possible setup. It works; it's slow.

**Deliverable:** End-to-end pipeline runs. Show one rollout batch + verifier scores + policy update.

### Exercise 6 — GRPO loss

Implement the GRPO update:

```python
def grpo_loss(logprobs_policy, logprobs_old, advantages, kl_coeff, ref_logprobs):
    ratios = (logprobs_policy - logprobs_old).exp()
    policy_loss = -(advantages * ratios).mean()
    kl = (logprobs_policy - ref_logprobs).mean()
    return policy_loss + kl_coeff * kl
```

(Real GRPO has clipping; the DeepSeek paper has the full form.)

Run GRPO on a small math dataset where the verifier is "check if the final answer equals the gold answer."

**Deliverable:** Reward curve over training. Even with a small model, you should see reward go up.

### Exercise 7 — Asynchronous rollouts (stretch)

Refactor Exercise 5 so the rollout engine runs continuously in its own process and the trainer just consumes results. Use `multiprocessing.Queue` or a Redis-style queue.

Add a "policy version" tag to every rollout. The trainer's loss has to use the `logprobs` computed with the *rollout's* policy version (`logprobs_old`), not the current policy.

**Deliverable:** Throughput comparison: synchronous vs asynchronous. Discuss the staleness tradeoff.

### Exercise 8 — Read a real RLHF codebase

Pick one of: OpenRLHF, veRL, NeMo-Aligner. Read enough to write a 1-page summary of:

- How weight sync between trainer and rollout works.
- How the verifier queue is set up.
- What gets parallelized and what doesn't.

**Deliverable:** 1-page writeup in `NOTES.md`.

---

## Self-check

1. Why does SFT use loss masking on prompt tokens?
2. Why does DPO need a reference model? Can it be omitted?
3. What's the memory cost difference between DPO and ORPO?
4. What's the dominant cost in GRPO training, and how do production systems address it?
5. What does "staleness" mean in async RLHF, and how do you bound it?
6. Why is having an inference engine inside your training loop systems-different from any other RL?

---

## Deliverables before moving on

- SFT with packing + masking works.
- LoRA-SFT works.
- DPO works (full and LoRA versions).
- GRPO with a verifier on a simple task works.
- Rollout-engine integration runs end-to-end (synchronous at minimum).
- Production codebase reading writeup.
