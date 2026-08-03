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
00-meta/                    How to use this repo, applied AI/ML systems guide
01-math-foundations/        Probability, stats, optimization theory, learning theory
02-data/                    Data tooling, EDA, preprocessing, feature engineering, data engineering
03-classical-ml/            Supervised/unsupervised, ensembles, time series, HPO, anomaly detection
04-evaluation/              Metrics, calibration, interpretation, A/B testing, causal inference
05-deep-learning-core/      Backprop, activations, losses, optimizers, normalization, PyTorch
06-architectures/           RNN/LSTM/GRU, attention, transformers, GNN, DL architecture selection
07-domains/                 NLP, computer vision, time series (domain-specific applications)
08-generative/              Autoencoders, GANs, diffusion, flows, frontier generative models
09-reinforcement-learning/  MDPs, bandits, value/policy methods, offline & deep RL
10-llms/                    Architecture, training, scaling, evaluation, interview notes
11-llm-applications/        RAG, agents, prompt optimization, inference optimization, MCP
12-systems-and-scale/       Compression, quantization, serving, distributed training, cost
13-production-ml/           MLOps, deployment, feature stores, model registry, monitoring
14-responsible-ai/          Privacy, fairness, adversarial robustness, LLM safety/governance
15-system-design/           Design frameworks, patterns, case studies
16-interview-prep/          Communication, playbooks, tradeoffs, coding, EA mock interviews
17-projects/                Runnable end-to-end projects: tabular ML, RAG, LoRA, MCP agent system
18-resources/               Reading lists, glossaries, external references
MIND-MAP.md                 Full topic mind map + gap analysis
FLOWCHARTS.md               Decision flowcharts across topics
SUMMARY.md                  Complete file-by-file table of contents
```

---

## Track 1 — Interview Blueprint (high tempo)

**Where to start tonight**

- **[AI/ML systems & application](00-meta/01-ai-ml-systems-and-application.md)** — Golden rules, architectures, infra patterns, gotchas.
- **[Interview hub](16-interview-prep/README.md)** — canonical entry point for interview prep.
- **[Math derivations hub](01-math-foundations/03-math-derivations.md)** — Chain rule to attention, with your pen.
- **[LLM interview notes](16-interview-prep/llm-interview-notes/README.md)** — RAG, agents, alignment, serving.
- **[LLM applications](11-llm-applications/README.md)** — retrieval, prompting, agents, tuning, inference optimization.
- **[Systems and scale](12-systems-and-scale/README.md)** — distributed training, serving, compression, cost.
- **[Pre-interview checklist](16-interview-prep/03-pre-interview-checklist.md)** — 48h → 24h → morning-of: what to review, verify, and do before walking in.
- **[EA mock interview handbook](16-interview-prep/ea/sde2-handbook/README.md)** — 30 structured SDE-2 AI engineer scenarios.

**10-Minute Revision Cards** (skim before any topic or interview)

| Section | Quick-review card |
|---------|--------------|
| Math Foundations | [_flashcards.md](01-math-foundations/_flashcards.md) · [_rapid-fire.md](01-math-foundations/_rapid-fire.md) |
| Classical ML | [_flashcards.md](03-classical-ml/_flashcards.md) · [_comparison.md](03-classical-ml/_comparison.md) |
| Deep Learning | [_cheatsheet.md](05-deep-learning-core/_cheatsheet.md) · [_flashcards.md](05-deep-learning-core/_flashcards.md) |
| LLMs | [_revision.md](10-llms/_revision.md) · [_comparison.md](10-llms/_comparison.md) |
| Production ML | [_revision.md](13-production-ml/_revision.md) · [_comparison.md](13-production-ml/_comparison.md) |
| Interview Prep | [_revision-cheatsheet.md](16-interview-prep/_revision-cheatsheet.md) |

**Engineering bridge:** *Training is your build job; the model artifact is your release candidate; inference is the always-on service; MLOps is CI/CD when the "code" and the "data" both change.*

---

## Track 2 — Deep-Dive Library (studio sessions)

| Section | What's inside |
|---|---|
| [Math Foundations](01-math-foundations/README.md) | Probability, optimization theory, statistical learning theory |
| [Data](02-data/README.md) | EDA, preprocessing, feature engineering, imbalanced data, data engineering |
| [Classical ML](03-classical-ml/README.md) | Trees, ensembles, clustering, time series, HPO |
| [Evaluation](04-evaluation/README.md) | Metrics, calibration, SHAP/LIME, A/B testing, causal inference |
| [Deep Learning Core](05-deep-learning-core/README.md) | Backprop, activations, losses, optimizers, PyTorch |
| [Architectures](06-architectures/README.md) | RNN/LSTM/GRU, attention, transformers, GNN |
| [Domains](07-domains/README.md) | NLP, computer vision, time series applications |
| [Generative Models](08-generative/README.md) | Autoencoders, GANs, diffusion, flow matching, consistency models |
| [Reinforcement Learning](09-reinforcement-learning/README.md) | MDPs, bandits, policy gradients, offline/deep RL |
| [LLMs](10-llms/README.md) | Architecture, training, scaling, evaluation, interview notes |
| [LLM Applications](11-llm-applications/README.md) | RAG, agents, tuning, inference optimization, MCP |
| [Systems & Scale](12-systems-and-scale/README.md) | Compression, serving, distributed training, cost optimization |
| [Production ML](13-production-ml/README.md) | MLOps, deployment, feature stores, registry, monitoring |
| [Responsible AI](14-responsible-ai/README.md) | Privacy, fairness, robustness, safety, governance |
| [ML System Design](15-system-design/README.md) | Design patterns, case studies, infra tradeoffs |
| [Transfer Learning & Domain Adaptation](05-deep-learning-core/10-transfer-learning.md) | Fine-tuning, DANN, few-shot, MAML, zero-shot |
| [Hands-On Projects](17-projects/README.md) | Runnable tabular ML pipeline, RAG system, and LoRA fine-tuning — not just prose |
| [EA Mock Interviews](16-interview-prep/ea/sde2-handbook/README.md) | 30 structured SDE-2 AI engineer interview scenarios |
| [EA System Design Playbook](16-interview-prep/ea/system-design-playbook/README.md) | Full system design chapters with worked scenarios |

---

## How These Notes Are Written

Each topic aims for a **senior answer in three beats**:

1. **Direct line** — What you say in the first ten seconds.
2. **Intuition** — An analogy that sticks.
3. **Production** — Latency, cost, scale, and what breaks first.

---

> **Cold open:** Production ML is a lot of engineering with a little bit of "magic." This repo leans into the engineering — because that's what keeps models out of the **Incident** channel.
