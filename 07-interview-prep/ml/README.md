---
module: Interview Prep
topic: ML Interview Notes
subtopic: ""
status: unread
tags: [interviewprep, ml, ml-interview-notes]
---
# ML Interview Notes

> **Quick routing:**
> - **30 min before interview** → `01-fundamentals-of-machine-learning.md` + `02-model-evaluation.md`
> - **2 hours** → add `08-algorithms.md` + `06-system-design-and-mlops.md`
> - **Full day** → read the full list below in order
>
> **Deep-dive hubs** (richer content, same topics):
> [Classical ML](../../02-classical-ml/) | [Deep Learning](../../03-deep-learning/) | [Production ML](../../06-production-ml/) | [LLMs](../../05-llms/)
>
> **Sibling folders in this hub:** [Deep Learning interview notes](../dl/README.md) | [LLM interview notes](../llm/README.md)

---

Welcome to the version of ML notes that does **not** sound like a sleepy university PDF.

If you come from **Azure + DevOps**, here is the cheat code:

- **Data prep** = your ingestion + validation stage
- **Training** = the build pipeline
- **Evaluation** = quality gates
- **Inference** = runtime service
- **MLOps** = CI/CD + observability + rollback, but for models

Think of the ML lifecycle like an **Azure DevOps release flow**, except the artifact is not just code. It is:

- code
- data
- features
- model weights
- infra
- monitoring logic

If one of those drifts, your "deployment" can still go live and still fail.

That is pure DevOps heartbreak.

---

## Start Here First

If you want the **most important notes first**, begin in this order:

1. `01-fundamentals-of-machine-learning.md`
2. `02-model-evaluation.md`
3. `03-optimization.md`
4. `08-algorithms.md`
5. `06-system-design-and-mlops.md`
6. `22-top-ml-interview-questions.md`

That set gives you:

- the language of ML
- how models learn
- how to tell if they are actually good
- how to ship them without drama
- a rapid-fire pass across the most commonly asked questions

For deep learning and LLM fundamentals, start in [dl/README.md](../dl/README.md) and [llm/README.md](../llm/README.md) instead.

---

## What's In This Folder

| File | Why it matters |
|------|----------------|
| `01-fundamentals-of-machine-learning.md` | The IPL powerplay of ML. Must-haves. |
| `02-model-evaluation.md` | Where "99% accuracy" gets exposed. |
| `03-optimization.md` | Training knobs, optimizer behavior, tuning instincts. |
| `04-optimization-theory.md` | Deeper optimization intuition and derivations. |
| `06-system-design-and-mlops.md` | Azure/DevOps bridge. This should feel like home. |
| `08-algorithms.md` | Trees, boosting, clustering, SVM, PCA. |
| `09-data-preprocessing-and-feature-engineering.md` | Feature work, leakage, scaling, encoding. |
| `10-probability-and-statistics.md` | Interview math without the pain. |
| `11-canonical-stats-questions.md` | Extra drill-down on statistics and probability edge cases. |
| `13-nlp.md` | Classical NLP foundations (BoW, TF-IDF, embeddings, RNNs). See [llm/02-nlp-transformers.md](../llm/02-nlp-transformers.md) for transformer-based NLP. |
| `15-coding.md` | Implementations plus what to say while coding. |
| `16-practical-ml-scenarios.md` | Situation-based prompts and production judgment. |
| `17-behavioral-and-scenario-based-questions.md` | The human round. Still important. |
| `18-math-derivations.md` | Worked math derivations behind common ML/DL results. |
| `19-maths.md` | Linear algebra and information theory refresher. |
| `20-privacy-and-fairness.md` | Differential privacy, fairness metrics, bias mitigation. |
| `21-additional-ml-interview-topics.md` | Data leakage, SMOTE, BatchNorm vs LayerNorm, and other grab-bag topics. |
| `22-top-ml-interview-questions.md` | Rapid-fire bank of the most commonly asked ML interview questions. |
| `23-ml-revision-cheatsheet.md` | Compressed revision cheatsheet across all ML topics. |
| `24-statistics-probability-rapid-fire.md` | Fast-recall statistics and probability drill. |
| `25-ml-system-design.md` | ML system design patterns and worked examples (cross-links to LLM serving in `llm/`). |
| `26-ml-coding-patterns.md` | Common coding patterns asked in ML interviews. |
| `27-scenario-based-questions.md` | Open-ended scenario and judgment questions. |
| `28-ml-interview-meta-strategy.md` | Meta-strategy for approaching the ML interview loop as a whole. |

Deep learning and LLM content that used to live here has moved to [dl/](../dl/README.md) and [llm/](../llm/README.md).

---

## How To Read These Notes

Each strong answer should do four things fast:

1. **Say the idea cleanly**
2. **Explain why it works**
3. **Call out where it breaks**
4. **Connect it to shipping**

That last part matters.

Because an ML model without deployment thinking is like a couture outfit with no stitching.
Looks fabulous on paper. Falls apart on first movement.

---

## Quick Thought Experiment

If someone says:

> "We trained a great model."

Your DevOps brain should instantly ask:

- On what data?
- With what versioned features?
- Evaluated using which metric?
- Deployed how?
- Monitored where?
- Rolled back how?

If those answers are fuzzy, the model is not "great."
It is just unemployed.
