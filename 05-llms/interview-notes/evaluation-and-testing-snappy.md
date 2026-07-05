---
module: Llms
topic: Interview Notes
subtopic: Evaluation And Testing Snappy
status: unread
tags: [llms, ml, interview-notes-evaluation-and]
---

> _Quick-recall companion. For the full deep-dive, see [evaluation-and-testing.md](09-evaluation-and-testing.md)._

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
