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
| [04-multi-agent-a2a-mcp/](04-multi-agent-a2a-mcp/) | Three independent agent processes (Research, Analysis, Report), each with its own restricted MCP client(s) wired to real MCP servers (stdio), handing work to each other over a real A2A (Agent2Agent) protocol implementation (Agent Cards, Tasks, JSON-over-HTTP). |
| [09-mcp-agent-system/](09-mcp-agent-system/) | Broader MCP surface-area tour (custom servers, HITL approval, remote SSE servers, MCP Resources) as a design doc/spec — see project 04 for the runnable A2A-protocol counterpart. |

## Why these

They correspond to the most-referenced-but-never-built milestones across this repo's study plans: a classical tabular ML pipeline, an LLM RAG system, LLM fine-tuning, and a concrete multi-agent/MCP/A2A system. Each is sized to run on a laptop in minutes as a correctness smoke test, with a documented path to scale up (bigger dataset, real base model, GPU, real external APIs) for a production-grade result.

## Where to Next

- **Concepts behind project 1** → [01-foundations/04-data-processing-and-eda.md](../02-data/02-data-processing-and-eda.md), [02-classical-ml/](../02-classical-ml/)
- **Concepts behind project 2** → [05-llms/](../05-llms/)
- **Concepts behind project 3** → [05-llms/](../05-llms/)
- **Concepts behind project 4** → [05-llms/applications/01-agentic-workflows.md](../11-llm-applications/01-agentic-workflows.md)
