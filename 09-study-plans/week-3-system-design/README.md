# Week 3 (Days 22-25): System Design

**Goal:** Translate ML knowledge into production-scale thinking. This week is about design patterns, trade-off reasoning, and communicating end-to-end system architecture clearly under pressure.

---

## What This Week Covers

| Days   | Topic                              | Key Concepts                                                        |
|--------|------------------------------------|---------------------------------------------------------------------|
| 22-23  | ML System Design Patterns          | Feature stores, serving infrastructure, drift detection, pipelines  |
| 24     | Case Studies                       | Ranking, recommendation, fraud detection, search                    |
| 25     | Behavioral Questions               | Failure stories, trade-off decisions, cross-functional collaboration |

---

## Focus Areas

- **End-to-end design:** Practice narrating a full ML system from data ingestion to monitoring. Interviewers want to see you hold the whole stack in your head.
- **Retrieval-ranking systems:** Understand two-stage retrieval (candidate gen + ranking) and why it exists at scale.
- **Drift detection:** Know the difference between data drift, concept drift, and model degradation — and how to monitor for each.
- **Inference optimization:** Quantization, distillation, batching, caching — know the tradeoffs.
- **Behavioral framing:** Structure answers using situation-action-result. Tie outcomes to measurable impact.

---

## Daily Study Pattern

1. Pick one case study domain (e.g., recommendation) and sketch a full system design on paper.
2. Identify two places in your design where an ML failure would cascade and explain mitigations.
3. For behavioral prep: write out one story per day using the SAR format.

---

## Linked Resources

- [Production ML Overview](../06-production-ml/README.md)
- [ML System Design Interview Guide](../06-production-ml/system-design/machine-learning-design-interview.md)
- [System Design & MLOps (interview notes)](../07-interview-prep/ml/system-design-and-mlops.md)
- [Scenario-Based LLM Questions](../07-interview-prep/llm/scenario-based-questions.md)

---

## End-of-Week Check

- Can you design a real-time recommendation system end-to-end, including the retraining loop?
- Can you explain how you would detect and respond to model drift in production?
- Can you describe a project failure and what you would change — without sounding defensive?
- Can you contrast online vs. batch inference and when you would choose each?
