# ML Interview Notes

> **Quick routing:**
> - **30 min before interview** → `fundamentals-of-machine-learning.md` + `model-evaluation.md`
> - **2 hours** → add `algorithms.md` + `system-design-and-mlops.md`
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

1. `fundamentals-of-machine-learning.md`
2. `model-evaluation.md`
3. `optimization.md`
4. `deep-learning.md`
5. `system-design-and-mlops.md`
6. `large-language-model.md`

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
| `fundamentals-of-machine-learning.md` | The IPL powerplay of ML. Must-haves. |
| `model-evaluation.md` | Where "99% accuracy" gets exposed. |
| `optimization.md` | Training knobs, optimizer behavior, tuning instincts. |
| `deep-learning.md` | Neural nets, CNNs, RNNs, Transformers. |
| `system-design-and-mlops.md` | Azure/DevOps bridge. This should feel like home. |
| `large-language-model.md` | LLM basics, prompt vs RAG vs fine-tune. |
| `algorithms.md` | Trees, boosting, clustering, SVM, PCA. |
| `data-preprocessing-and-feature-engineering.md` | Feature work, leakage, scaling, encoding. |
| `probability-and-statistics.md` | Interview math without the pain. |
| `canonical-stats-questions.md` | Extra drill-down on statistics and probability edge cases. |
| `optimization-theory.md` | Deeper optimization intuition and derivations. |
| `probabilistic-graphical-models.md` | Structured probability, graphical models, and inference. |
| `nlp.md` | Classical NLP to Transformers. |
| `computer-vision.md` | Detection, segmentation, OCR, ViTs. |
| `coding.md` | Implementations plus what to say while coding. |
| `practical-ml-scenarios.md` | Situation-based prompts and production judgment. |
| `behavioral-and-scenario-based-questions.md` | The human round. Still important. |

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
