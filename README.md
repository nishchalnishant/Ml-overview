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
01-foundations/          Intro to AI/ML systems, math & theory, glossary, flashcards
02-classical-ml/         Supervised, unsupervised, preprocessing, anomaly detection, active learning, Bayesian methods
03-deep-learning/        Components + methods (NLP, CV, time series) + transfer learning + PyTorch
05-llms/                 Architecture, training, applications, interview notes
06-production-ml/        MLOps, system design
07-interview-prep/       ML + LLM interview Q&As, scenarios, coding
08-emerging-topics/      Emerging trends, XAI, causal inference, privacy-preserving ML, continual learning
09-study-plans/          30-day week-by-week study plan
10-references/           Book notes, research papers, datasets, tools, benchmarks, notation
11-data-scientist/       SQL, statistics, A/B testing, metrics, EDA, causal inference
12-projects/              Runnable end-to-end projects: tabular ML pipeline, RAG, LoRA fine-tuning
MIND-MAP.md              Full topic mind map + gap analysis
FLOWCHARTS.md            Decision flowcharts across topics
```

---

## Track 1 — Interview Blueprint (high tempo)

**Where to start tonight**

- **[AI/ML systems & application](01-foundations/01-ai-ml-systems-and-application.md)** — Golden rules, architectures, infra patterns, gotchas.
- **[Math & theory foundations](01-foundations/02-math-and-theory-foundations.md)** — The math you can whiteboard.
- **[30-day roadmap](09-study-plans/README.md)** — Structured reps from zero to "I can hold the room."
- **[Math derivations hub](07-interview-prep/ml/18-math-derivations.md)** — Chain rule to attention, with your pen.
- **[Interview hub](07-interview-prep/README.md)** — canonical entry point for interview prep.
- **[ML interview notes](07-interview-prep/ml/README.md)** — full classical + DL Q&A bank.
- **[LLM deep dives](05-llms/interview-notes/README.md)** — RAG, agents, alignment, serving.
- **[Pre-interview checklist](07-interview-prep/PRE-INTERVIEW-CHECKLIST.md)** — 48h → 24h → morning-of: what to review, verify, and do before walking in.

**10-Minute Revision Cards** (skim before any topic or interview)

| Section | Quick-review card |
|---------|--------------|
| Foundations | [flashcards.md](01-foundations/flashcards.md) |
| Classical ML | [classical-ml-flashcards.md](02-classical-ml/classical-ml-flashcards.md) |
| Deep Learning | [deep-learning-cheatsheet.md](03-deep-learning/deep-learning-cheatsheet.md) · [flashcards](03-deep-learning/deep-learning-flashcards.md) |
| LLMs | [REVISION.md](05-llms/REVISION.md) |
| Production ML | [REVISION.md](06-production-ml/REVISION.md) |

**Azure / DevOps bridge:** *Training is your build job; the model artifact is your release candidate; inference is the always-on service; MLOps is CI/CD when the "code" and the "data" both change.*

---

## Track 2 — Deep-Dive Library (studio sessions)

| Section | What's inside |
|---|---|
| [Foundations](01-foundations/) | Intro to AI, glossary, revision guide |
| [Classical ML](02-classical-ml/) | Bias–variance, trees, calibration, when classical beats DL |
| [Deep Learning](03-deep-learning/) | Activations, backprop, attention, PyTorch, MCP |
| [Computer Vision](03-deep-learning/methods/03-computer-vision.md) | CNNs, detection, ViT, CLIP, self-supervised |
| [Time Series](03-deep-learning/methods/09-time-series.md) | ARIMA through Transformers, forecasting, anomaly detection |
| [LLMs](05-llms/README.md) | Architecture, training, scaling, evaluation |
| [LLM Applications](05-llms/applications/README.md) | RAG, agents, tuning, inference optimization |
| [Multimodal AI](05-llms/applications/09-multimodal.md) | CLIP, VLMs, fusion architectures, audio, video, deployment |
| [Speculative Decoding](05-llms/applications/07-speculative-decoding.md) | Medusa, Eagle, standard SD, production trade-offs |
| [LLM Training Stability](05-llms/05-training-stability.md) | Loss spikes, mixed precision, RLHF failure modes |
| [Production ML](06-production-ml/README.md) | MLOps, CI/CD for ML, deployment |
| [Model Governance](06-production-ml/03-model-governance.md) | Model registry, audit trails, GDPR, champion-challenger |
| [ML System Design](06-production-ml/system-design/) | Design patterns, case studies, engineering |
| [Causal Inference](08-emerging-topics/experimentation-and-causal-inference/README.md) | A/B testing, causal graphs, uplift modeling |
| [Model Compression](03-deep-learning/components/16-model-compression.md) | Quantization, distillation, pruning |
| [Research Papers](10-references/research-papers/README.md) | 25 foundational LLM papers every ML interview cares about |
| [Transfer Learning & Domain Adaptation](03-deep-learning/02-transfer-learning.md) | Fine-tuning, DANN, few-shot, MAML, zero-shot |
| [Data Scientist Reference](11-data-scientist/README.md) | SQL, statistics & probability, A/B testing, business metrics, EDA |
| [Hands-On Projects](12-projects/README.md) | Runnable tabular ML pipeline, RAG system, and LoRA fine-tuning — not just prose |

---

## How These Notes Are Written

Each topic aims for a **senior answer in three beats**:

1. **Direct line** — What you say in the first ten seconds.
2. **Intuition** — An analogy that sticks.
3. **Production** — Latency, cost, scale, and what breaks first.

---

> **Cold open:** Production ML is a lot of engineering with a little bit of "magic." This repo leans into the engineering — because that's what keeps models out of the **Incident** channel.
