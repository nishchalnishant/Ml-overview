---
module: Projects
topic: Projects
subtopic: ""
status: unread
tags: [projects, hands-on]
---
# Hands-On Projects

Everything else in this repo is reference material — theory, math, interview Q&A. This folder is the exception: runnable code. Each project is self-contained (own `requirements.txt`, own README, own data generation script) and has been executed end-to-end to confirm it works, not just written.

## Projects

| Project | What it demonstrates |
| :--- | :--- |
| [01-tabular-ml-pipeline/](01-tabular-ml-pipeline/) | Leakage-safe `sklearn` pipeline on a synthetic churn dataset: EDA-informed preprocessing, `ColumnTransformer`, baseline vs. XGBoost comparison via cross-validation, held-out evaluation, model serialization. |
| [02-rag-pipeline/](02-rag-pipeline/) | Full retrieval-augmented generation system: chunking with overlap, local embeddings, two-stage dense + lexical-rerank retrieval, citation-forced generation with a pluggable LLM backend. |
| [03-llm-finetuning/](03-llm-finetuning/) | LoRA parameter-efficient fine-tuning loop: adapter injection via `peft`, `Trainer`-based training, adapter-only save, before/after generation comparison. |

## Why these three

They correspond to the three most-referenced-but-never-built milestones across this repo's study plans: a classical tabular ML pipeline, an LLM RAG system, and LLM fine-tuning. Each is sized to run on a laptop in minutes as a correctness smoke test, with a documented path to scale up (bigger dataset, real base model, GPU) for a production-grade result.

## Where to Next

- **Concepts behind project 1** → [01-foundations/04-data-processing-and-eda.md](../01-foundations/04-data-processing-and-eda.md), [02-classical-ml/](../02-classical-ml/)
- **Concepts behind project 2** → [05-llms/](../05-llms/)
- **Concepts behind project 3** → [05-llms/](../05-llms/)
