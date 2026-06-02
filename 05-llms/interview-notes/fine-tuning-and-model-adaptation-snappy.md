---
module: Llms
topic: Interview Notes
subtopic: Fine Tuning And Model Adaptation Snappy
status: unread
tags: [llms, ml, interview-notes-fine-tuning-an]
---
# Fine-tuning & model adaptation — the “remaster, not rewrite” playbook

Fine-tuning is how you change **behavior**. RAG is how you change **knowledge at runtime**. Prompting is how you change **instructions**.

**One-line:** Prompting = config. RAG = read-through cache. Fine-tuning = shipping a patch.

---

# Q1: What is fine-tuning, and when should you fine-tune an LLM?
- **Direct answer:** Fine-tuning updates model parameters (fully or partially) so it behaves differently on your task.
- **Use it when:** you need consistent style/format, domain behaviors, tool reliability, or latency/cost improvements vs huge prompts.
- **Avoid when:** you just need fresh facts → use RAG.

---

# Q2: Full fine-tuning vs PEFT?
- **Full FT:** updates all weights → most flexible, most expensive, risk of forgetting.
- **PEFT:** train a small set of params (LoRA/adapters/prefix) → cheaper, safer, easier multi-tenant.
- **DevOps bridge:** full FT = fork the whole service; PEFT = ship a lightweight plugin.

---

# Q3: What is LoRA and how does it work?
- **Direct answer:** Freeze base weights; learn low-rank update \(\Delta W pprox BA\) with tiny matrices.
- **Analogy (music):** You don’t re-record the track—you add an EQ/remix layer.

---

# Q4: What is QLoRA?
- **Direct answer:** Quantize base weights (e.g., 4-bit NF4) during training while keeping LoRA adapters in 16-bit.
- **Why it matters:** makes big-model tuning feasible on small GPUs.

---

# Q5: Prefix tuning vs prompt tuning vs LoRA?
- **Prompt tuning:** learn soft prompt vectors.
- **Prefix tuning:** learn soft “prefix” vectors injected into attention.
- **LoRA:** modifies linear layers via low-rank updates; often stronger and widely used.

---

# Q6: Adapter-based fine-tuning?
- **Direct answer:** Insert small trainable modules between layers while freezing the base.
- **Note:** can add inference overhead unless merged/optimized.

---

# Q7: What is RLHF and how is it used to align LLMs?
- **Direct answer:** Use preference feedback to push outputs toward helpful/safe behavior (often via reward model + PPO historically).
- **DevOps bridge:** policy training + compliance gates for a system that would otherwise “do whatever works.”

---

# Q8: What is instruction tuning?
- **Direct answer:** SFT on instruction→response pairs to make base models follow commands/chat format reliably.

---

# Q9: How do you prepare a dataset for fine-tuning?
- **Checklist:** clear objective, clean templates, dedupe, balance, red-team examples, train/val split, PII scrubbing.
- **Mini prompt:** What kills you fastest? → leakage (train examples repeated in eval).

---

# Q10: Catastrophic forgetting — what is it and how prevent?
- **Direct answer:** Model overwrites general skills while learning new domain.
- **Fixes:** PEFT, mix in general data, lower LR, fewer epochs, regularization, rehearsal.

---

# Q11: Fine-tuning vs RAG vs prompt engineering?
- **Prompting:** fastest iteration; best for formatting/instructions.
- **RAG:** best for changing/adding facts + citations.
- **Fine-tuning:** best for stable behavior, tone, domain style, tool reliability, latency/cost.

---

# Q12: How do you evaluate a fine-tuned model?
- **Direct answer:** task success + format validity + safety + regression tests.
- **DevOps bridge:** treat evals like CI checks; block promotion if regressions.

---

# Q13: Synthetic data generation for fine-tuning?
- **Direct answer:** generate training pairs using a strong model/humans; filter and validate.
- **Risk:** model learns your generator’s quirks; must diversify and audit.

---

# Q14: Key hyperparameters (LR, epochs, batch size, LoRA rank)?
- **LR:** too high = forget/instability; too low = no learning.
- **Epochs:** too many = memorization.
- **Batch size:** affects stability; use grad accumulation.
- **LoRA rank:** small ranks often saturate; higher ranks can overfit.

---

# Q15: Fine-tune for legal/medical/finance domains?
- **Direct answer:** domain templates + terminology + safety constraints + eval sets.
- **Best practice:** pair with RAG for latest policies/guidelines.

---

# Q16: What is continual pre-training?
- **Direct answer:** continue pre-training on large unlabeled domain text to shift the base distribution.
- **When:** you need domain language modeling improvements, not just instruction format.

---

# Q17: How do you merge multiple LoRA adapters?
- **Options:** merge sequentially (risk interference), route by task, or compose via weighted merges.
- **DevOps bridge:** multiple plugins need versioning + compatibility tests.

---

# Q18: SFT vs alignment training?
- **SFT:** learn “ideal answers.”
- **Alignment:** learn preferences/safety trade-offs (RLHF/DPO/RLAIF).

---

# Q19: What is RLAIF?
- **Direct answer:** reinforcement learning from **AI** feedback instead of humans (model judges with a constitution/rubric).

---

# Q20: Distillation for fine-tuning + legal considerations?
- **Direct answer:** compress behavior from teacher to student.
- **Legal note:** licensing/data provenance matters; don’t distill proprietary outputs into redistributable weights without rights.

---

# Q21: Fine-tuned model outputs wrong facts due to bad data. Fix?
- **Fixes:** clean data, dedupe, add retrieval grounding, strengthen evals, add counterexamples, reduce epochs.

---

# Q22: Decide LoRA vs full fine-tuning?
- **Choose LoRA/PEFT:** limited GPUs, multi-tenant adapters, lower risk.
- **Choose full FT:** you need deep capability shift and can afford the cost.

---

# Q23: Model memorized training data verbatim (overfitting). Fix?
- **Fixes:** more/cleaner data, stronger regularization, early stopping, lower LR, fewer epochs, better splits.

---

# Q24: Model forgot general capabilities after domain FT. Fix?
- **Fixes:** PEFT, rehearsal mix, lower LR, fewer steps, add general eval gates.

---

# Q25: RLHF preference data has low annotator agreement. Ensure quality?
- **Fixes:** clearer rubric, training raters, adjudication, calibration tasks, remove ambiguous prompts, measure inter-annotator agreement.

## Rapid Recall

### Direct answer
- Direct Answer: Fine-tuning updates model parameters (fully or partially) so it behaves differently on your task.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Fine-tuning updates model parameters (fully or partially) so it behaves differently on your task.

### Use it when
- Direct Answer: you need consistent style/format, domain behaviors, tool reliability, or latency/cost improvements vs huge prompts.
- Why: This matters because it tells you how to reason about use it when.
- Pitfall: Don't answer "Use it when" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: you need consistent style/format, domain behaviors, tool reliability, or latency/cost improvements vs huge prompts.

### Avoid when
- Direct Answer: you just need fresh facts → use RAG.
- Why: This matters because it tells you how to reason about avoid when.
- Pitfall: Don't answer "Avoid when" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: you just need fresh facts → use RAG.

### Full FT
- Direct Answer: updates all weights → most flexible, most expensive, risk of forgetting.
- Why: This matters because it tells you how to reason about full ft.
- Pitfall: Don't answer "Full FT" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: updates all weights → most flexible, most expensive, risk of forgetting.

### PEFT
- Direct Answer: train a small set of params (LoRA/adapters/prefix) → cheaper, safer, easier multi-tenant.
- Why: This matters because it tells you how to reason about peft.
- Pitfall: Don't answer "PEFT" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: train a small set of params (LoRA/adapters/prefix) → cheaper, safer, easier multi-tenant.

### DevOps bridge
- Direct Answer: full FT = fork the whole service; PEFT = ship a lightweight plugin.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: full FT = fork the whole service; PEFT = ship a lightweight plugin.

### Direct answer
- Direct Answer: Freeze base weights; learn low-rank update \(\Delta W pprox BA\) with tiny matrices.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Freeze base weights; learn low-rank update \(\Delta W pprox BA\) with tiny matrices.

### Analogy (music)
- Direct Answer: You don’t re-record the track—you add an EQ/remix layer.
- Why: This matters because it tells you how to reason about analogy (music).
- Pitfall: Don't answer "Analogy (music)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: You don’t re-record the track—you add an EQ/remix layer.

### Direct answer
- Direct Answer: Quantize base weights (e.g., 4-bit NF4) during training while keeping LoRA adapters in 16-bit.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Quantize base weights (e.g., 4-bit NF4) during training while keeping LoRA adapters in 16-bit.

### Why it matters
- Direct Answer: makes big-model tuning feasible on small GPUs.
- Why: This matters because it tells you how to reason about why it matters.
- Pitfall: Don't answer "Why it matters" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: makes big-model tuning feasible on small GPUs.

### Prompt tuning
- Direct Answer: learn soft prompt vectors.
- Why: This matters because it tells you how to reason about prompt tuning.
- Pitfall: Don't answer "Prompt tuning" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: learn soft prompt vectors.

### Prefix tuning
- Direct Answer: learn soft “prefix” vectors injected into attention.
- Why: This matters because it tells you how to reason about prefix tuning.
- Pitfall: Don't answer "Prefix tuning" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: learn soft “prefix” vectors injected into attention.

### LoRA
- Direct Answer: modifies linear layers via low-rank updates; often stronger and widely used.
- Why: This matters because it tells you how to reason about lora.
- Pitfall: Don't answer "LoRA" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: modifies linear layers via low-rank updates; often stronger and widely used.

### Direct answer
- Direct Answer: Insert small trainable modules between layers while freezing the base.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Insert small trainable modules between layers while freezing the base.

### Note
- Direct Answer: can add inference overhead unless merged/optimized.
- Why: This matters because it tells you how to reason about note.
- Pitfall: Don't answer "Note" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: can add inference overhead unless merged/optimized.

### Direct answer
- Direct Answer: Use preference feedback to push outputs toward helpful/safe behavior (often via reward model + PPO historically).
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Use preference feedback to push outputs toward helpful/safe behavior (often via reward model + PPO historically).

### DevOps bridge
- Direct Answer: policy training + compliance gates for a system that would otherwise “do whatever works.”
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: policy training + compliance gates for a system that would otherwise “do whatever works.”

### Direct answer
- Direct Answer: SFT on instruction→response pairs to make base models follow commands/chat format reliably.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: SFT on instruction→response pairs to make base models follow commands/chat format reliably.

### Checklist
- Direct Answer: clear objective, clean templates, dedupe, balance, red-team examples, train/val split, PII scrubbing.
- Why: This matters because it tells you how to reason about checklist.
- Pitfall: Don't answer "Checklist" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: clear objective, clean templates, dedupe, balance, red-team examples, train/val split, PII scrubbing.

### Mini prompt
- Direct Answer: What kills you fastest? → leakage (train examples repeated in eval).
- Why: This matters because it tells you how to reason about mini prompt.
- Pitfall: Don't answer "Mini prompt" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: What kills you fastest? → leakage (train examples repeated in eval).

### Direct answer
- Direct Answer: Model overwrites general skills while learning new domain.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Model overwrites general skills while learning new domain.

### Fixes
- Direct Answer: PEFT, mix in general data, lower LR, fewer epochs, regularization, rehearsal.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: PEFT, mix in general data, lower LR, fewer epochs, regularization, rehearsal.

### Prompting
- Direct Answer: fastest iteration; best for formatting/instructions.
- Why: This matters because it tells you how to reason about prompting.
- Pitfall: Don't answer "Prompting" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: fastest iteration; best for formatting/instructions.

### RAG
- Direct Answer: best for changing/adding facts + citations.
- Why: This matters because it tells you how to reason about rag.
- Pitfall: Don't answer "RAG" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: best for changing/adding facts + citations.

### Fine-tuning
- Direct Answer: best for stable behavior, tone, domain style, tool reliability, latency/cost.
- Why: This matters because it tells you how to reason about fine-tuning.
- Pitfall: Don't answer "Fine-tuning" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: best for stable behavior, tone, domain style, tool reliability, latency/cost.

### Direct answer
- Direct Answer: task success + format validity + safety + regression tests.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: task success + format validity + safety + regression tests.

### DevOps bridge
- Direct Answer: treat evals like CI checks; block promotion if regressions.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: treat evals like CI checks; block promotion if regressions.

### Direct answer
- Direct Answer: generate training pairs using a strong model/humans; filter and validate.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: generate training pairs using a strong model/humans; filter and validate.

### Risk
- Direct Answer: model learns your generator’s quirks; must diversify and audit.
- Why: This matters because it tells you how to reason about risk.
- Pitfall: Don't answer "Risk" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: model learns your generator’s quirks; must diversify and audit.

### LR
- Direct Answer: too high = forget/instability; too low = no learning.
- Why: This matters because it tells you how to reason about lr.
- Pitfall: Don't answer "LR" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: too high = forget/instability; too low = no learning.

### Epochs
- Direct Answer: too many = memorization.
- Why: This matters because it tells you how to reason about epochs.
- Pitfall: Don't answer "Epochs" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: too many = memorization.

### Batch size
- Direct Answer: affects stability; use grad accumulation.
- Why: This matters because it tells you how to reason about batch size.
- Pitfall: Don't answer "Batch size" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: affects stability; use grad accumulation.

### LoRA rank
- Direct Answer: small ranks often saturate; higher ranks can overfit.
- Why: This matters because it tells you how to reason about lora rank.
- Pitfall: Don't answer "LoRA rank" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: small ranks often saturate; higher ranks can overfit.

### Direct answer
- Direct Answer: domain templates + terminology + safety constraints + eval sets.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: domain templates + terminology + safety constraints + eval sets.

### Best practice
- Direct Answer: pair with RAG for latest policies/guidelines.
- Why: This matters because it tells you how to reason about best practice.
- Pitfall: Don't answer "Best practice" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: pair with RAG for latest policies/guidelines.

### Direct answer
- Direct Answer: continue pre-training on large unlabeled domain text to shift the base distribution.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: continue pre-training on large unlabeled domain text to shift the base distribution.

### When
- Direct Answer: you need domain language modeling improvements, not just instruction format.
- Why: This matters because it tells you how to reason about when.
- Pitfall: Don't answer "When" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: you need domain language modeling improvements, not just instruction format.

### Options
- Direct Answer: merge sequentially (risk interference), route by task, or compose via weighted merges.
- Why: This matters because it tells you how to reason about options.
- Pitfall: Don't answer "Options" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: merge sequentially (risk interference), route by task, or compose via weighted merges.

### DevOps bridge
- Direct Answer: multiple plugins need versioning + compatibility tests.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: multiple plugins need versioning + compatibility tests.

### SFT
- Direct Answer: learn “ideal answers.”
- Why: This matters because it tells you how to reason about sft.
- Pitfall: Don't answer "SFT" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: learn “ideal answers.”

### Alignment
- Direct Answer: learn preferences/safety trade-offs (RLHF/DPO/RLAIF).
- Why: This matters because it tells you how to reason about alignment.
- Pitfall: Don't answer "Alignment" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: learn preferences/safety trade-offs (RLHF/DPO/RLAIF).

### Direct answer
- Direct Answer: reinforcement learning from AI feedback instead of humans (model judges with a constitution/rubric).
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: reinforcement learning from AI feedback instead of humans (model judges with a constitution/rubric).

### Direct answer
- Direct Answer: compress behavior from teacher to student.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: compress behavior from teacher to student.

### Legal note
- Direct Answer: licensing/data provenance matters; don’t distill proprietary outputs into redistributable weights without rights.
- Why: This matters because it tells you how to reason about legal note.
- Pitfall: Don't answer "Legal note" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: licensing/data provenance matters; don’t distill proprietary outputs into redistributable weights without rights.

### Fixes
- Direct Answer: clean data, dedupe, add retrieval grounding, strengthen evals, add counterexamples, reduce epochs.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: clean data, dedupe, add retrieval grounding, strengthen evals, add counterexamples, reduce epochs.

### Choose LoRA/PEFT
- Direct Answer: limited GPUs, multi-tenant adapters, lower risk.
- Why: This matters because it tells you how to reason about choose lora/peft.
- Pitfall: Don't answer "Choose LoRA/PEFT" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: limited GPUs, multi-tenant adapters, lower risk.

### Choose full FT
- Direct Answer: you need deep capability shift and can afford the cost.
- Why: This matters because it tells you how to reason about choose full ft.
- Pitfall: Don't answer "Choose full FT" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: you need deep capability shift and can afford the cost.

### Fixes
- Direct Answer: more/cleaner data, stronger regularization, early stopping, lower LR, fewer epochs, better splits.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: more/cleaner data, stronger regularization, early stopping, lower LR, fewer epochs, better splits.

### Fixes
- Direct Answer: PEFT, rehearsal mix, lower LR, fewer steps, add general eval gates.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: PEFT, rehearsal mix, lower LR, fewer steps, add general eval gates.

### Fixes
- Direct Answer: clearer rubric, training raters, adjudication, calibration tasks, remove ambiguous prompts, measure inter-annotator agreement.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: clearer rubric, training raters, adjudication, calibration tasks, remove ambiguous prompts, measure inter-annotator agreement.
