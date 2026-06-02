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

## Rapid Recall

### Direct answer
- Direct Answer: Scaling laws relate loss to compute/data/model size; Chinchilla-style results suggest many models were under-trained on data for their size (more data, fewer params can be better at fixed compute).
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Scaling laws relate loss to compute/data/model size; Chinchilla-style results suggest many models were under-trained on data for their size (more data, fewer params can be better…

### RLHF
- Direct Answer: align behavior to preferences/safety.
- Why: This matters because it tells you how to reason about rlhf.
- Pitfall: Don't answer "RLHF" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: align behavior to preferences/safety.

### PPO
- Direct Answer: classic RL algorithm used in RLHF (but finicky).
- Why: This matters because it tells you how to reason about ppo.
- Pitfall: Don't answer "PPO" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: classic RL algorithm used in RLHF (but finicky).

### DPO
- Direct Answer: simpler preference optimization on chosen vs rejected pairs (often more stable).
- Why: This matters because it tells you how to reason about dpo.
- Pitfall: Don't answer "DPO" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: simpler preference optimization on chosen vs rejected pairs (often more stable).

### Direct answer
- Direct Answer: Sparse experts route tokens so you get large capacity with less per-token compute.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Sparse experts route tokens so you get large capacity with less per-token compute.

### Direct answer
- Direct Answer: modify positional encoding behavior (RoPE scaling/interpolation) and do light uptraining; still watch lost-in-the-middle.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: modify positional encoding behavior (RoPE scaling/interpolation) and do light uptraining; still watch lost-in-the-middle.

### Direct answer
- Direct Answer: next-token generation fills gaps; reduce via grounding (RAG), constraints, verifiers, and better evals.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: next-token generation fills gaps; reduce via grounding (RAG), constraints, verifiers, and better evals.

### Direct answer
- Direct Answer: enforce a schema so invalid tokens are blocked; reduces retries and parser breakage.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: enforce a schema so invalid tokens are blocked; reduces retries and parser breakage.

### Pitfalls
- Direct Answer: judge bias, prompt sensitivity, preference drift, self-judging.
- Why: This matters because it tells you how to reason about pitfalls.
- Pitfall: Don't answer "Pitfalls" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: judge bias, prompt sensitivity, preference drift, self-judging.

### Mitigate
- Direct Answer: calibration sets, multiple judges, human spot checks.
- Why: This matters because it tells you how to reason about mitigate.
- Pitfall: Don't answer "Mitigate" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: calibration sets, multiple judges, human spot checks.

### Direct answer
- Direct Answer: train a smaller model to mimic a larger one (soft targets/behaviors) for cheaper serving.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: train a smaller model to mimic a larger one (soft targets/behaviors) for cheaper serving.

### Injection
- Direct Answer: untrusted text tries to override instructions (often via retrieved content).
- Why: This matters because it tells you how to reason about injection.
- Pitfall: Don't answer "Injection" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: untrusted text tries to override instructions (often via retrieved content).

### Jailbreak
- Direct Answer: user tries to bypass safety policies.
- Why: This matters because it tells you how to reason about jailbreak.
- Pitfall: Don't answer "Jailbreak" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: user tries to bypass safety policies.

### Direct answer
- Direct Answer: reuse computation/KV for repeated prefixes (system prompts, templates) to reduce TTFT and tokens.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: reuse computation/KV for repeated prefixes (system prompts, templates) to reduce TTFT and tokens.

### GPTQ
- Direct Answer: quantize with error compensation; heavier preparation.
- Why: This matters because it tells you how to reason about gptq.
- Pitfall: Don't answer "GPTQ" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: quantize with error compensation; heavier preparation.

### AWQ
- Direct Answer: activation-aware protection of salient weights; popular for GPU serving.
- Why: This matters because it tells you how to reason about awq.
- Pitfall: Don't answer "AWQ" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: activation-aware protection of salient weights; popular for GPU serving.

### Include
- Direct Answer: intended use, limitations, evals, safety notes, data provenance, privacy, monitoring, rollback plan.
- Why: This matters because it tells you how to reason about include.
- Pitfall: Don't answer "Include" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: intended use, limitations, evals, safety notes, data provenance, privacy, monitoring, rollback plan.
