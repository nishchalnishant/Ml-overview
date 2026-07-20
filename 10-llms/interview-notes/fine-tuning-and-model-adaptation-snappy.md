---
module: Llms
topic: Interview Notes
subtopic: Fine Tuning And Model Adaptation Snappy
status: unread
tags: [llms, ml, interview-notes-fine-tuning-an]
---

> _Quick-recall companion. For the full deep-dive, see [fine-tuning-and-model-adaptation.md](05-fine-tuning-and-model-adaptation.md)._

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
