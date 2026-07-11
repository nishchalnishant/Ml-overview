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
4. `05-deep-learning.md`
5. `06-system-design-and-mlops.md`
6. `07-large-language-model.md`

That set gives you:

- the language of ML
- how models learn
- how to tell if they are actually good
- how to ship them without drama
- how modern AI systems are built

---

## What's In This Folder

| File | Why it matters |
|------|----------------|
| `01-fundamentals-of-machine-learning.md` | The IPL powerplay of ML. Must-haves. |
| `02-model-evaluation.md` | Where "99% accuracy" gets exposed. |
| `03-optimization.md` | Training knobs, optimizer behavior, tuning instincts. |
| `05-deep-learning.md` | Neural nets, CNNs, RNNs, Transformers. |
| `06-system-design-and-mlops.md` | Azure/DevOps bridge. This should feel like home. |
| `07-large-language-model.md` | LLM basics, prompt vs RAG vs fine-tune. |
| `08-algorithms.md` | Trees, boosting, clustering, SVM, PCA. |
| `09-data-preprocessing-and-feature-engineering.md` | Feature work, leakage, scaling, encoding. |
| `10-probability-and-statistics.md` | Interview math without the pain. |
| `11-canonical-stats-questions.md` | Extra drill-down on statistics and probability edge cases. |
| `04-optimization-theory.md` | Deeper optimization intuition and derivations. |
| `13-nlp.md` | Classical NLP to Transformers. |
| `14-computer-vision.md` | Detection, segmentation, OCR, ViTs. |
| `15-coding.md` | Implementations plus what to say while coding. |
| `16-practical-ml-scenarios.md` | Situation-based prompts and production judgment. |
| `17-behavioral-and-scenario-based-questions.md` | The human round. Still important. |
| `18-math-derivations.md` | Worked math derivations behind common ML/DL results. |
| `19-maths.md` | Linear algebra and information theory refresher. |
| `20-privacy-and-fairness.md` | Differential privacy, fairness metrics, bias mitigation. |
| `21-additional-ml-interview-topics.md` | Data leakage, SMOTE, BatchNorm vs LayerNorm, and other grab-bag topics. |

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
