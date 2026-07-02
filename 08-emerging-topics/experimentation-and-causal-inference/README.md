---
module: Emerging Topics
topic: Experimentation and Causal Inference
subtopic: ""
status: unread
tags: [emergingtopics, ml, experimentation-and-causal-inference]
---
# Experimentation and Causal Inference

This folder covers how to measure whether a change actually caused an effect — from online A/B tests to observational causal inference when you can't randomize.

## Files in This Folder

| File | What it covers |
| :--- | :--- |
| [experimentation-and-causal-inference.md](experimentation-and-causal-inference.md) | A/B testing mechanics, what breaks (peeking, multiple testing, Simpson's paradox), Bayesian testing, bandits, propensity score matching, diff-in-diff, RDD, instrumental variables, causal DAGs, uplift modeling, and experimentation in ML systems (model A/B tests, interleaving) |

> For the infrastructure side of A/B testing (assignment services, logging, experiment platforms), see [06-production-ml/system-design/ab-testing-experimentation.md](../../06-production-ml/system-design/ab-testing-experimentation.md) — that file focuses on system design, this one on the statistics.

---

## How To Read It

Read top to bottom if you're new to causal inference — each section builds on the counterfactual framing introduced in §2. If you already know A/B testing, jump straight to §7 onward (observational methods: PSM, diff-in-diff, RDD, IV, DAGs).

---

## Back to top

[Emerging Topics README](../README.md)
