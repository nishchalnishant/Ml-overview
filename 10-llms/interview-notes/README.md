---
module: LLMs
topic: LLM Interview Notes — GenAI without the fog
subtopic: ""
status: unread
tags: [llms, ml, llm-interview-notes-genai-without-the-fog]
---
# LLM Interview Notes — *GenAI without the fog*

> **Primary interview hub:** [Interview Hub](../../16-interview-prep/README.md)
>
> Use this folder as the deeper GenAI supplement when you need more than the general ML interview path.

> **Priority order** (read the "snappy" version of each first):
> 1. `llm-fundamentals` — non-negotiable foundation
> 2. `prompt-engineering` — asked in almost every LLM round
> 3. `retrieval-augmented-generation-rag` — the most common system design
> 4. `ai-system-design` — the senior test
> 5. `fine-tuning-and-model-adaptation` — LoRA/RLHF/DPO
> 6. `llmops-and-production-ai` — for ML engineer roles
>
> **Deep dives** (when the interview panel is specialist):
> [Architecture](../03-architecture-deep-dive.md) | [Training](../01-training-process.md) | [Inference optimization](../../11-llm-applications/06-inference-optimization.md)

LLMs are **services** that eat tokens and return distributions. Your job in an interview is to sound like you’ve **operated** them: context, retrieval, safety, cost, evals — not just read the brochure.

**DevOps translation:** Prompting is **config**. RAG is **read-through cache + citations**. Fine-tuning is **shipping a patch** to behavior. Agents are **orchestration** with tools — same story as your favorite workflow engine, just with a stochastic brain.

---

## What’s in this folder

You now have **two versions** of most notes:

- **Deep**: the original, more in-depth version (keeps the classic structure).
- **Snappy**: the rewritten, punchy version for fast reading (`*-snappy.md`).

| Topic | Deep | Snappy | What you’ll grab from it |
|------|------|--------|--------------------------|
| Fundamentals | `01-llm-fundamentals.md` | `llm-fundamentals-snappy.md` | Transformers, attention, context, limits |
| Prompting | `02-prompt-engineering.md` | `prompt-engineering-snappy.md` | Reliability, structure, failure modes |
| RAG | `03-retrieval-augmented-generation-rag.md` | `retrieval-augmented-generation-rag-snappy.md` | Chunking, retrieval, faithfulness, **RAGAS**-style thinking |
| Agents | `04-ai-agents-and-agentic-systems.md` | `ai-agents-and-agentic-systems-snappy.md` | Loops, tools, guardrails — **observe → act → verify** |
| Fine-tuning | `05-fine-tuning-and-model-adaptation.md` | `fine-tuning-and-model-adaptation-snappy.md` | LoRA, adapters, when **not** to fine-tune |
| Vector DBs | `06-vector-databases-and-embeddings.md` | `vector-databases-and-embeddings-snappy.md` | Embeddings, ANN/HNSW — **search at scale** |
| System design | `07-ai-system-design.md` | `ai-system-design-snappy.md` | End-to-end GenAI systems — **the system design round** |
| LLMOps | `08-llmops-and-production-ai.md` | `llmops-and-production-ai-snappy.md` | Monitoring, cost, release — **LLMOps** |
| Evals | `09-evaluation-and-testing.md` | `evaluation-and-testing-snappy.md` | Benchmarks, judge models, regression — **quality gates** |
| Safety / RAI | `10-ai-safety-ethics-and-responsible-ai-what.md` | `ai-safety-ethics-and-responsible-ai-what-snappy.md` | Bias, policy, misuse — **the grown-up table** |
| Multimodal | `11-multi-modal-ai.md` | `multi-modal-ai-snappy.md` | Vision + language — **more modalities, same infra** |
| Infra | `12-ai-infrastructure-and-scalability.md` | `ai-infrastructure-and-scalability-snappy.md` | GPUs, batching, parallelism — **when the bill arrives** |
| Coding | `13-coding-and-practical-implementation.md` | `coding-and-practical-implementation-snappy.md` | Patterns you can code live |
| Behavioral | `14-behavioral-and-scenario-based-questions.md` | `behavioral-and-scenario-based-questions-snappy.md` | Stories that don’t sound rehearsed |
| Extras | `15-additional-llm-interview-topics.md` | `additional-llm-interview-topics-snappy.md` | Scaling laws, RLHF/DPO, MoE, long context, hallucinations, distillation, AWQ/GPTQ, **prompt injection** |
| Efficient deployment | `16-efficient-llm-deployment.md` | `efficient-llm-deployment-snappy.md` | Quantization, caching, batching, routing — serving playbook |
| Alignment & reasoning | `17-advanced-alignment-and-reasoning.md` | `advanced-alignment-and-reasoning-snappy.md` | RLHF/DPO, ReAct, agents — the “senior” layer |
| Production failures | [18-production-alignment-failures.md](18-production-alignment-failures.md) | *(no snappy version)* | Real-world alignment/safety incidents and postmortems |

---

**Gulzar-adjacent intuition:** A good LLM answer isn’t a single word — it’s **what the last hundred tokens did to the next one**. That’s **context** and **attention** in one sentence.

**Mini prompt for you:** *Sketch RAG on a whiteboard as three boxes: ingest, index, serve. Where does your “pipeline” analogy break?* (Hint: **freshness** and **citation** trust.)

---

Use this alongside the main question banks in the repo when you want **depth** without losing the thread.
