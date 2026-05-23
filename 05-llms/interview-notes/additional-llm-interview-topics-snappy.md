---
module: Llms
topic: Interview Notes
subtopic: Additional Llm Interview Topics Snappy
status: unread
tags: [llms, ml, interview-notes-additional-llm]
---
# Additional LLM interview topics — bonus round (high signal)

This is the “you seem senior” sheet.

---

# Q1: What are scaling laws, and what does “Chinchilla-optimal” training mean?
- **Direct answer:** Scaling laws relate loss to compute/data/model size; Chinchilla-style results suggest many models were under-trained on data for their size (more data, fewer params can be better at fixed compute).

---

# Q2: RLHF, PPO, and DPO — what problem does each solve?
- **RLHF:** align behavior to preferences/safety.
- **PPO:** classic RL algorithm used in RLHF (but finicky).
- **DPO:** simpler preference optimization on chosen vs rejected pairs (often more stable).

---

# Q3: What is MoE, and why use it?
- **Direct answer:** Sparse experts route tokens so you get large capacity with less per-token compute.

---

# Q4: Extending context length beyond training (RoPE scaling, YaRN, etc.)?
- **Direct answer:** modify positional encoding behavior (RoPE scaling/interpolation) and do light uptraining; still watch lost-in-the-middle.

---

# Q5: What causes hallucinations, and how reduce in production?
- **Direct answer:** next-token generation fills gaps; reduce via grounding (RAG), constraints, verifiers, and better evals.

---

# Q6: What is structured output (JSON mode, grammars, constrained decoding)?
- **Direct answer:** enforce a schema so invalid tokens are blocked; reduces retries and parser breakage.

---

# Q7: LLM-as-a-judge — pitfalls?
- **Pitfalls:** judge bias, prompt sensitivity, preference drift, self-judging.
- **Mitigate:** calibration sets, multiple judges, human spot checks.

---

# Q8: Knowledge distillation for LLMs?
- **Direct answer:** train a smaller model to mimic a larger one (soft targets/behaviors) for cheaper serving.

---

# Q9: Prompt injection vs jailbreaking?
- **Injection:** untrusted text tries to override instructions (often via retrieved content).
- **Jailbreak:** user tries to bypass safety policies.

---

# Q10: Prefix / prompt caching to reduce cost and latency?
- **Direct answer:** reuse computation/KV for repeated prefixes (system prompts, templates) to reduce TTFT and tokens.

---

# Q11: AWQ vs GPTQ (interview level)?
- **GPTQ:** quantize with error compensation; heavier preparation.
- **AWQ:** activation-aware protection of salient weights; popular for GPU serving.

---

# Q12: What should a model card / release checklist cover?
- **Include:** intended use, limitations, evals, safety notes, data provenance, privacy, monitoring, rollback plan.

## Flashcards

**Direct answer?** #flashcard
Scaling laws relate loss to compute/data/model size; Chinchilla-style results suggest many models were under-trained on data for their size (more data, fewer params can be better at fixed compute).

**RLHF?** #flashcard
align behavior to preferences/safety.

**PPO?** #flashcard
classic RL algorithm used in RLHF (but finicky).

**DPO?** #flashcard
simpler preference optimization on chosen vs rejected pairs (often more stable).

**Direct answer?** #flashcard
Sparse experts route tokens so you get large capacity with less per-token compute.

**Direct answer?** #flashcard
modify positional encoding behavior (RoPE scaling/interpolation) and do light uptraining; still watch lost-in-the-middle.

**Direct answer?** #flashcard
next-token generation fills gaps; reduce via grounding (RAG), constraints, verifiers, and better evals.

**Direct answer?** #flashcard
enforce a schema so invalid tokens are blocked; reduces retries and parser breakage.

**Pitfalls?** #flashcard
judge bias, prompt sensitivity, preference drift, self-judging.

**Mitigate?** #flashcard
calibration sets, multiple judges, human spot checks.

**Direct answer?** #flashcard
train a smaller model to mimic a larger one (soft targets/behaviors) for cheaper serving.

**Injection?** #flashcard
untrusted text tries to override instructions (often via retrieved content).

**Jailbreak?** #flashcard
user tries to bypass safety policies.

**Direct answer?** #flashcard
reuse computation/KV for repeated prefixes (system prompts, templates) to reduce TTFT and tokens.

**GPTQ?** #flashcard
quantize with error compensation; heavier preparation.

**AWQ?** #flashcard
activation-aware protection of salient weights; popular for GPU serving.

**Include?** #flashcard
intended use, limitations, evals, safety notes, data provenance, privacy, monitoring, rollback plan.
