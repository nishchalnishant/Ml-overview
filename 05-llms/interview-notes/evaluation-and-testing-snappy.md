---
module: Llms
topic: Interview Notes
subtopic: Evaluation And Testing Snappy
status: unread
tags: [llms, ml, interview-notes-evaluation-and]
---
# Evaluation & testing — CI/CD for LLMs

LLMs are non-deterministic services. So “it worked once” is not a victory—**it’s a coincidence**.

**One-line:** Evals are your **quality gates**: they stop prompt tweaks from becoming incidents.

---

# Q1: What is evaluation-driven development for AI applications?
- **Direct answer:** Build evals first, iterate with metrics, and ship only when gates pass.
- **DevOps bridge:** exactly like test-driven development + CI checks.

---

# Q2: How do you evaluate LLM outputs? What metrics do you use?
- **Common buckets:** correctness, faithfulness, safety, format validity, latency/cost.
- **Mini prompt:** If this is a SQL generator, what’s the best metric? → execution accuracy.

---

# Q3: BLEU, ROUGE, BERTScore — when use?
- **BLEU/ROUGE:** n-gram overlap (good for constrained extraction, not free-form).
- **BERTScore:** semantic similarity via embeddings.

---

# Q4: What is G-Eval?
- **Direct answer:** rubric-based evaluation using an LLM to score outputs.
- **Risk:** judge bias; mitigate with calibration and consistency checks.

---

# Q5: LLM-as-a-judge + limitations?
- **Pros:** scalable, captures nuance.
- **Limits:** bias, contamination, preference drift, prompt sensitivity.

---

# Q6: Human evaluation?
- **Direct answer:** curated prompts + trained raters + rubric + inter-annotator agreement.

---

# Q7: Red teaming?
- **Direct answer:** systematically attack your system with adversarial prompts to find failures.
- **DevOps bridge:** security testing + chaos engineering for language.

---

# Q8: Detect/measure hallucinations?
- **Patterns:** grounding checks, citation verification, claim extraction + evidence matching.

---

# Q9: Adversarial testing?
- **Direct answer:** test robustness against prompt injection, jailbreaks, edge inputs, multilingual attacks.

---

# Q10: Regression test suite for AI apps?
- **Direct answer:** fixed eval set + expected outputs/criteria; run on every prompt/model change.

---

# Q11: Benchmarks (MMLU/HumanEval/GSM8K) — interpret?
- **Use:** directional signal.
- **Caution:** contamination and “teaching to the test.”

---

# Q12: Evaluate a RAG system end-to-end?
- **Retrieval:** context precision/recall, recall@k.
- **Generation:** faithfulness, answer relevance.
- **System:** latency and cost.

---

# Q13: Evaluate AI agents?
- **Metrics:** task success, tool correctness, safety violations, loop rate, cost, time.

---

# Q14: Offline vs online evaluation?
- **Offline:** curated datasets, repeatable.
- **Online:** production signals, A/B tests, user feedback.

---

# Q15: Measure factual consistency?
- **Techniques:** claim checking against sources, citation audits, retrieval-based verification.

---

# Q16: Multi-turn conversation quality?
- **Metrics:** goal completion, consistency, memory correctness, tone/safety.

---

# Q17: Golden datasets?
- **Direct answer:** high-quality labeled prompts used as regression anchors.

---

# Q18: Continuous evaluation in production?
- **Direct answer:** sample traffic, run shadow evals, alert on drift/regressions.
- **DevOps bridge:** continuous testing + observability.

---

# Q19: Evaluate bias?
- **Direct answer:** subgroup analysis, counterfactual tests, fairness metrics + qualitative review.

---

# Q20: Compare models/prompts rigorously?
- **Direct answer:** paired A/B tests, significance testing, control for prompt mix.

---

# Q21: Robustness across input variations?
- **Direct answer:** fuzzing with paraphrases, typos, dialects, adversarial templates.

---

# Q22: Traditional ML eval vs LLM eval?
- **Traditional ML:** fixed labels, deterministic metrics.
- **LLMs:** open-ended outputs; need rubrics, semantic/functional checks.

---

# Q23: Set up an eval framework from scratch?
- **Steps:** define success → collect prompts → define rubrics → automate scoring → set gates → monitor.

---

# Q24: Conflicting fairness metrics?
- **Direct answer:** align with product risk/cost, document trade-offs, pick primary metric + constraints.

---

# Q25: Model biased after 6 months — monitor?
- **Direct answer:** continuous eval + drift checks + periodic audits.

---

# Q26: Auditor can’t reproduce results — ensure reproducibility?
- **Controls:** version prompts/models/data, log configs, deterministic settings, store eval datasets.

---

# Q27: Red team a chatbot before launch?
- **Plan:** injection tests, jailbreaks, PII extraction, tool abuse, refusal quality, rate limit tests.

---

# Q28: Red team multimodal models?
- **Direct answer:** test cross-modal attacks (image text, hidden instructions), OCR paths, combined prompts.

## Rapid Recall

### Direct answer
- Direct Answer: Build evals first, iterate with metrics, and ship only when gates pass.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Build evals first, iterate with metrics, and ship only when gates pass.

### DevOps bridge
- Direct Answer: exactly like test-driven development + CI checks.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: exactly like test-driven development + CI checks.

### Common buckets
- Direct Answer: correctness, faithfulness, safety, format validity, latency/cost.
- Why: This matters because it tells you how to reason about common buckets.
- Pitfall: Don't answer "Common buckets" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: correctness, faithfulness, safety, format validity, latency/cost.

### Mini prompt
- Direct Answer: If this is a SQL generator, what’s the best metric? → execution accuracy.
- Why: This matters because it tells you how to reason about mini prompt.
- Pitfall: Don't answer "Mini prompt" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: If this is a SQL generator, what’s the best metric? → execution accuracy.

### BLEU/ROUGE
- Direct Answer: n-gram overlap (good for constrained extraction, not free-form).
- Why: This matters because it tells you how to reason about bleu/rouge.
- Pitfall: Don't answer "BLEU/ROUGE" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: n-gram overlap (good for constrained extraction, not free-form).

### BERTScore
- Direct Answer: semantic similarity via embeddings.
- Why: This matters because it tells you how to reason about bertscore.
- Pitfall: Don't answer "BERTScore" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: semantic similarity via embeddings.

### Direct answer
- Direct Answer: rubric-based evaluation using an LLM to score outputs.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: rubric-based evaluation using an LLM to score outputs.

### Risk
- Direct Answer: judge bias; mitigate with calibration and consistency checks.
- Why: This matters because it tells you how to reason about risk.
- Pitfall: Don't answer "Risk" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: judge bias; mitigate with calibration and consistency checks.

### Pros
- Direct Answer: scalable, captures nuance.
- Why: This matters because it tells you how to reason about pros.
- Pitfall: Don't answer "Pros" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: scalable, captures nuance.

### Limits
- Direct Answer: bias, contamination, preference drift, prompt sensitivity.
- Why: This matters because it tells you how to reason about limits.
- Pitfall: Don't answer "Limits" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: bias, contamination, preference drift, prompt sensitivity.

### Direct answer
- Direct Answer: curated prompts + trained raters + rubric + inter-annotator agreement.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: curated prompts + trained raters + rubric + inter-annotator agreement.

### Direct answer
- Direct Answer: systematically attack your system with adversarial prompts to find failures.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: systematically attack your system with adversarial prompts to find failures.

### DevOps bridge
- Direct Answer: security testing + chaos engineering for language.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: security testing + chaos engineering for language.

### Patterns
- Direct Answer: grounding checks, citation verification, claim extraction + evidence matching.
- Why: This matters because it tells you how to reason about patterns.
- Pitfall: Don't answer "Patterns" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: grounding checks, citation verification, claim extraction + evidence matching.

### Direct answer
- Direct Answer: test robustness against prompt injection, jailbreaks, edge inputs, multilingual attacks.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: test robustness against prompt injection, jailbreaks, edge inputs, multilingual attacks.

### Direct answer
- Direct Answer: fixed eval set + expected outputs/criteria; run on every prompt/model change.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: fixed eval set + expected outputs/criteria; run on every prompt/model change.

### Use
- Direct Answer: directional signal.
- Why: This matters because it tells you how to reason about use.
- Pitfall: Don't answer "Use" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: directional signal.

### Caution
- Direct Answer: contamination and “teaching to the test.”
- Why: This matters because it tells you how to reason about caution.
- Pitfall: Don't answer "Caution" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: contamination and “teaching to the test.”

### Retrieval
- Direct Answer: context precision/recall, recall@k.
- Why: This matters because it tells you how to reason about retrieval.
- Pitfall: Don't answer "Retrieval" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: context precision/recall, recall@k.

### Generation
- Direct Answer: faithfulness, answer relevance.
- Why: This matters because it tells you how to reason about generation.
- Pitfall: Don't answer "Generation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: faithfulness, answer relevance.

### System
- Direct Answer: latency and cost.
- Why: This matters because it tells you how to reason about system.
- Pitfall: Don't answer "System" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: latency and cost.

### Metrics
- Direct Answer: task success, tool correctness, safety violations, loop rate, cost, time.
- Why: This matters because it tells you how to reason about metrics.
- Pitfall: Don't answer "Metrics" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: task success, tool correctness, safety violations, loop rate, cost, time.

### Offline
- Direct Answer: curated datasets, repeatable.
- Why: This matters because it tells you how to reason about offline.
- Pitfall: Don't answer "Offline" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: curated datasets, repeatable.

### Online
- Direct Answer: production signals, A/B tests, user feedback.
- Why: This matters because it tells you how to reason about online.
- Pitfall: Don't answer "Online" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: production signals, A/B tests, user feedback.

### Techniques
- Direct Answer: claim checking against sources, citation audits, retrieval-based verification.
- Why: This matters because it tells you how to reason about techniques.
- Pitfall: Don't answer "Techniques" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: claim checking against sources, citation audits, retrieval-based verification.

### Metrics
- Direct Answer: goal completion, consistency, memory correctness, tone/safety.
- Why: This matters because it tells you how to reason about metrics.
- Pitfall: Don't answer "Metrics" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: goal completion, consistency, memory correctness, tone/safety.

### Direct answer
- Direct Answer: high-quality labeled prompts used as regression anchors.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: high-quality labeled prompts used as regression anchors.

### Direct answer
- Direct Answer: sample traffic, run shadow evals, alert on drift/regressions.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: sample traffic, run shadow evals, alert on drift/regressions.

### DevOps bridge
- Direct Answer: continuous testing + observability.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: continuous testing + observability.

### Direct answer
- Direct Answer: subgroup analysis, counterfactual tests, fairness metrics + qualitative review.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: subgroup analysis, counterfactual tests, fairness metrics + qualitative review.

### Direct answer
- Direct Answer: paired A/B tests, significance testing, control for prompt mix.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: paired A/B tests, significance testing, control for prompt mix.

### Direct answer
- Direct Answer: fuzzing with paraphrases, typos, dialects, adversarial templates.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: fuzzing with paraphrases, typos, dialects, adversarial templates.

### Traditional ML
- Direct Answer: fixed labels, deterministic metrics.
- Why: This matters because it tells you how to reason about traditional ml.
- Pitfall: Don't answer "Traditional ML" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: fixed labels, deterministic metrics.

### LLMs
- Direct Answer: open-ended outputs; need rubrics, semantic/functional checks.
- Why: This matters because it tells you how to reason about llms.
- Pitfall: Don't answer "LLMs" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: open-ended outputs; need rubrics, semantic/functional checks.

### Steps
- Direct Answer: define success → collect prompts → define rubrics → automate scoring → set gates → monitor.
- Why: This matters because it tells you how to reason about steps.
- Pitfall: Don't answer "Steps" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: define success → collect prompts → define rubrics → automate scoring → set gates → monitor.

### Direct answer
- Direct Answer: align with product risk/cost, document trade-offs, pick primary metric + constraints.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: align with product risk/cost, document trade-offs, pick primary metric + constraints.

### Direct answer
- Direct Answer: continuous eval + drift checks + periodic audits.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: continuous eval + drift checks + periodic audits.

### Controls
- Direct Answer: version prompts/models/data, log configs, deterministic settings, store eval datasets.
- Why: This matters because it tells you how to reason about controls.
- Pitfall: Don't answer "Controls" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: version prompts/models/data, log configs, deterministic settings, store eval datasets.

### Plan
- Direct Answer: injection tests, jailbreaks, PII extraction, tool abuse, refusal quality, rate limit tests.
- Why: This matters because it tells you how to reason about plan.
- Pitfall: Don't answer "Plan" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: injection tests, jailbreaks, PII extraction, tool abuse, refusal quality, rate limit tests.

### Direct answer
- Direct Answer: test cross-modal attacks (image text, hidden instructions), OCR paths, combined prompts.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: test cross-modal attacks (image text, hidden instructions), OCR paths, combined prompts.
