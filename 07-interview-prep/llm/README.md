---
module: Interview Prep
topic: General ML Interview Portal
subtopic: ""
status: unread
tags: [interviewprep, ml, general-ml-interview-portal]
---
# General ML Interview Portal

> **Quick routing:**
> - **30 min** → `02-ml-revision.md` + your STAR stories
> - **2 hours** → `01-top-ml-interview-questions.md` + `04-ml-system-design.md` + `06-scenario-based-questions.md`
> - **Full day** → work through the level-based path below
>
> **Deep-dive hubs** (richer content):
> [Interview Hub](../README.md) | [ML Interview Notes](../ml/) | [LLM Interview Notes](../../05-llms/interview-notes/) | [Production ML](../../06-production-ml/)

---

This folder is your general interview war room.

Not the sleepy textbook version.
The useful one.

If you already think in **Azure**, **DevOps**, **pipelines**, **rollouts**, and **production reliability**, this folder should feel natural.

Because good ML interview answers are rarely just:

- "here is the algorithm"

They are usually:

- "here is the tradeoff"
- "here is the deployment implication"
- "here is how I would ship it safely"

---

## Start Here by Level

### Junior / L3

Start with:

1. `01-top-ml-interview-questions.md`
2. `02-ml-revision.md`
3. `03-statistics-probability.md`

### Mid / L4

Add:

4. `04-ml-system-design.md`
5. `05-ml-coding-patterns.md`
6. `06-scenario-based-questions.md`

### Senior / L5+

Add:

7. `07-math-derivations.md`
8. `08-dl-architectures.md`
9. `09-nlp-transformers.md`
10. `10-machine-learning-interviews.md`

---

## What Each File Is For

- `01-top-ml-interview-questions.md`
  The fast question bank

- `02-ml-revision.md`
  Last-minute revision sprint

- `03-statistics-probability.md`
  The part where uncertainty starts asking follow-up questions

- `05-ml-coding-patterns.md`
  Clean implementation patterns plus coding-round instincts

- `04-ml-system-design.md`
  Architecture, retrieval-ranking, drift, latency, rollout

- `06-scenario-based-questions.md`
  Practical troubleshooting and decision-making

- `07-math-derivations.md`
  Whiteboard survival

- `08-dl-architectures.md`
  CNNs, RNNs, Transformers, GANs, core deep-learning comparisons

- `09-nlp-transformers.md`
  Tokenization, embeddings, attention, BERT, GPT

- `10-machine-learning-interviews.md`
  The meta-strategy: how to frame answers, guide system-design rounds, and sound senior. Includes company-specific FAANG prep notes (§11) and take-home case study tips (§12).

## LLM-Specific Route

If the question is specifically about RAG, agents, prompt engineering, alignment, or LLMOps, switch to the dedicated LLM notes in [`../../05-llms/interview-notes/`](../../05-llms/interview-notes/).

---

## Emergency Study Plan

### 6 hours left

- `01-top-ml-interview-questions.md`
- `04-ml-system-design.md`
- `06-scenario-based-questions.md`

### 3 hours left

- `02-ml-revision.md`
- `03-statistics-probability.md`
- `08-dl-architectures.md`

### 1 hour left

- `02-ml-revision.md`
- `07-math-derivations.md`

### 30 minutes left

- your STAR stories
- top metrics
- top system-design framework
- one failure story

---

## Quick Thought Experiment

If someone says:

> "I know the algorithm well."

your interview brain should ask:

- Can you choose the right metric?
- Can you explain tradeoffs?
- Can you detect leakage?
- Can you ship it?
- Can you debug it when production gets weird?

If yes, now we are talking.
