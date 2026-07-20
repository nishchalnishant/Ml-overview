---
module: Study Plans
topic: Week 3 (Days 22-25): System Design
subtopic: ""
status: unread
tags: [studyplans, ml, week-3-days-22-25-system-design]
---
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
4. For each design, write the monitoring spec: what metrics would you alert on and at what threshold?

---

## Linked Resources

- [Production ML Overview](README.md)
- [ML System Design Interview Guide](../../../15-system-design/01-design-interview-framework.md)
- [System Design & MLOps (interview notes)](../../../13-production-ml/11-system-design-and-mlops.md)
- [Scenario-Based LLM Questions](../../08-scenario-based-questions.md)
- Day files in this folder: 01-day-22, 02-day-23, 03-day-24-25

---

## Projects for This Week

**Day 22-23 Project: System Design Write-Up**

Choose one of these prompts and write a 1-2 page design document (not slides — prose with diagrams):

Option A: Design a spam detection system for a messaging app with 10M daily messages.
- Specify: latency requirements, model choice and justification, feature pipeline, monitoring, retraining strategy
- Address: cold start (new users/accounts), adversarial adaptation (spammers evolve), and rollback procedure

Option B: Design a "similar items" feature for an e-commerce site.
- Specify: how you build item embeddings, how you handle new item cold start, the ANN index choice
- Address: what happens when the catalog updates (item added/removed), catalog freshness vs. latency tradeoff

The deliverable is a document you could hand to an engineering team. It should include: system diagram, data flow, key components with their technology choices, failure modes and mitigations, and monitoring plan.

**Day 24 Project: Case Study Deconstruction**

Pick the YouTube recommendation system (the 2016 deep learning paper is freely available). Read the paper and answer in writing:
1. What was the metric they optimized, and why is it not the metric they wanted to maximize in principle?
2. Draw the two-stage architecture. For each stage, name the primary design constraint (latency, accuracy, scale).
3. How did they handle the cold-start problem for new users and new videos?
4. What is one design decision in the paper you would question or change, and why?

**Day 25 Project: STAR Story Bank**

Write 5 STAR stories covering:
1. A technical project where you made a design choice under constraint — and can defend it
2. A time you identified and fixed a data quality or pipeline bug in production (or a personal project)
3. A time you communicated a technical tradeoff to a non-technical stakeholder
4. A failure — a model that did not work as expected — and what you would do differently
5. A time you learned from a colleague's approach and changed your own

For each story: Situation (2-3 sentences max), Task (1 sentence), Action (5-8 sentences, all "I"), Result (quantified if possible).

---

## Milestone Checkpoints

**After Day 22:** Given a latency budget of 50ms for a fraud detection system, can you design a cascade architecture, specify the model at each stage, and explain why each stage's recall matters more than its precision?

**After Day 23:** Can you explain why the YouTube paper uses weighted sampling by watch time rather than random sampling for training? Can you name two failure modes of the two-tower model and how to detect them?

**After Day 25:** Do you have at least one story where you quantified the outcome? Can you deliver any STAR story in under 2 minutes without exceeding 30 seconds on the Situation?

---

## End-of-Week Check

- Can you design a real-time recommendation system end-to-end, including the retraining loop?
- Can you explain how you would detect and respond to model drift (data drift vs. concept drift) in production?
- Can you describe a project failure and what you would change — without sounding defensive?
- Can you contrast online vs. batch inference and when you would choose each?
- Can you explain train-serve skew with a concrete example and describe the architectural solution?
