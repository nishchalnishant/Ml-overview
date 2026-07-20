---
module: ML Overview
topic: AI & ML — Senior Prep, Without the Yawn
subtopic: ""
status: unread
tags: [mloverviewroot, ml, ai-ml-senior-prep-without-the-yawn]
---
# AI & ML — Senior Prep, Without the Yawn

You already ship software: pipelines, environments, rollouts, observability. **Machine learning is that same muscle** — except the "binary" you deploy is a model, and the bugs can drift in silently like a change in crowd noise at Wankhede.

This repo is your **two-track studio album**: fast interview-ready takes, and deeper "liner notes" when you want the full mix.

---

## Repo Structure

```
01-foundations/          Intro to AI/ML systems, glossary, flashcards
02-classical-ml/         Supervised, unsupervised, preprocessing, evaluation, calibration
03-deep-learning/        Components + methods (NLP, CV, time series) + transfer learning + PyTorch
05-llms/                 Architecture, training, applications, interview notes
06-production-ml/        MLOps, system design, case studies
07-interview-prep/       ML + DL + LLM interview Q&As, scenarios, coding
09-study-plans/          35-day week-by-week study plan (includes LLM deep-dive week)
12-projects/             Runnable end-to-end projects: tabular ML, RAG, LoRA, multi-agent A2A+MCP
13-interview-prep/       EA-specific SDE-2 AI engineer mock interviews (30 scenarios)
14-ea-ai-system-design-playbook/  23 full system design chapters (46 sections each)
MIND-MAP.md              Full topic mind map + gap analysis
FLOWCHARTS.md            Decision flowcharts across topics
```

---

## Track 1 — Interview Blueprint (high tempo)

**Where to start tonight**

- **[AI/ML systems & application](00-meta/01-ai-ml-systems-and-application.md)** — Golden rules, architectures, infra patterns, gotchas.
- **[35-day roadmap](16-interview-prep/study-plans/README.md)** — Structured reps from zero to "I can hold the room."
- **[Math derivations hub](01-math-foundations/03-math-derivations.md)** — Chain rule to attention, with your pen.
- **[Interview hub](16-interview-prep/README.md)** — canonical entry point for interview prep.
- **[ML interview notes](README.md)** — full classical + DL Q&A bank.
- **[LLM deep dives](10-llms/interview-notes/README.md)** — RAG, agents, alignment, serving.
- **[Pre-interview checklist](16-interview-prep/03-pre-interview-checklist.md)** — 48h → 24h → morning-of: what to review, verify, and do before walking in.

**10-Minute Revision Cards** (skim before any topic or interview)

| Section | Quick-review card |
|---------|--------------|
| Foundations | [flashcards.md](01-math-foundations/_flashcards.md) |
| Classical ML | [classical-ml-flashcards.md](03-classical-ml/_flashcards.md) |
| Deep Learning | [deep-learning-cheatsheet.md](05-deep-learning-core/_cheatsheet.md) · [flashcards](05-deep-learning-core/_flashcards.md) |
| LLMs | [REVISION.md](10-llms/_revision.md) |
| Production ML | [REVISION.md](13-production-ml/_revision.md) |

**Engineering bridge:** *Training is your build job; the model artifact is your release candidate; inference is the always-on service; MLOps is CI/CD when the "code" and the "data" both change.*

---

## Track 2 — Deep-Dive Library (studio sessions)

| Section | What's inside |
|---|---|
| [Foundations](01-foundations/) | Intro to AI, glossary, revision guide |
| [Classical ML](02-classical-ml/) | Bias–variance, trees, calibration, when classical beats DL |
| [Deep Learning](03-deep-learning/) | Activations, backprop, attention, PyTorch |
| [Computer Vision](07-domains/04-computer-vision.md) | CNNs, detection, ViT, CLIP, self-supervised |
| [Time Series](07-domains/06-time-series.md) | ARIMA through Transformers, forecasting, anomaly detection |
| [LLMs](10-llms/README.md) | Architecture, training, scaling, evaluation |
| [LLM Applications](README.md) | RAG, agents, tuning, inference optimization |
| [Multimodal AI](11-llm-applications/09-multimodal.md) | CLIP, VLMs, fusion architectures, audio, video, deployment |
| [Speculative Decoding](11-llm-applications/07-speculative-decoding.md) | Medusa, Eagle, standard SD, production trade-offs |
| [LLM Training Stability](10-llms/05-training-stability.md) | Loss spikes, mixed precision, RLHF failure modes |
| [Production ML](13-production-ml/README.md) | MLOps, CI/CD for ML, deployment |
| [Model Governance](13-production-ml/03-model-governance.md) | Model registry, audit trails, GDPR, champion-challenger |
| [ML System Design](06-production-ml/system-design/) | Design patterns, case studies, engineering |
| [Model Compression](12-systems-and-scale/01-model-compression.md) | Quantization, distillation, pruning |
| [Transfer Learning & Domain Adaptation](05-deep-learning-core/10-transfer-learning.md) | Fine-tuning, DANN, few-shot, MAML, zero-shot |
| [Hands-On Projects](17-projects/README.md) | Runnable tabular ML pipeline, RAG system, and LoRA fine-tuning — not just prose |
| [EA Mock Interviews](16-interview-prep/ea/sde2-handbook/README.md) | 30 structured SDE-2 AI engineer interview scenarios |
| [EA System Design Playbook](16-interview-prep/ea/system-design-playbook/README.md) | 23 full system design chapters with 46-section template |

---

## How These Notes Are Written

Each topic aims for a **senior answer in three beats**:

1. **Direct line** — What you say in the first ten seconds.
2. **Intuition** — An analogy that sticks.
3. **Production** — Latency, cost, scale, and what breaks first.

---

> **Cold open:** Production ML is a lot of engineering with a little bit of "magic." This repo leans into the engineering — because that's what keeps models out of the **Incident** channel.
