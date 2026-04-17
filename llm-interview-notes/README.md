# LLM interview notes — *GenAI without the fog*

LLMs are **services** that eat tokens and return distributions. Your job in an interview is to sound like you’ve **operated** them: context, retrieval, safety, cost, evals — not just read the brochure.

**DevOps translation:** Prompting is **config**. RAG is **read-through cache + citations**. Fine-tuning is **shipping a patch** to behavior. Agents are **orchestration** with tools — same story as your favorite workflow engine, just with a stochastic brain.

---

## What’s in this folder

| File | What you’ll grab from it |
|------|--------------------------|
| `llm-fundamentals.md` | Transformers, attention, context, limits |
| `prompt-engineering.md` | Reliability, structure, failure modes |
| `retrieval-augmented-generation-rag.md` | Chunking, retrieval, faithfulness, **RAGAS**-style thinking |
| `ai-agents-and-agentic-systems.md` | Loops, tools, guardrails — **observe → act → verify** |
| `fine-tuning-and-model-adaptation.md` | LoRA, adapters, when **not** to fine-tune |
| `vector-databases-and-embeddings.md` | Embeddings, ANN, HNSW — **search at scale** |
| `ai-system-design.md` | End-to-end GenAI systems — **the system design round** |
| `llmops-and-production-ai.md` | Monitoring, cost, release — **LLMOps** |
| `evaluation-and-testing.md` | Benchmarks, judge models, regression — **quality gates** |
| `ai-safety-ethics-and-responsible-ai-what.md` | Bias, policy, misuse — **the grown-up table** |
| `multi-modal-ai.md` | Vision + language — **more modalities, same infra** |
| `ai-infrastructure-and-scalability.md` | GPUs, batching, parallelism — **when the bill arrives** |
| `coding-and-practical-implementation.md` | Patterns you can code live |
| `behavioral-and-scenario-based-questions.md` | Stories that don’t sound rehearsed |
| **`additional-llm-interview-topics.md`** | Scaling laws, RLHF/DPO, MoE, long context, hallucinations, distillation, AWQ/GPTQ, **prompt injection** — extra tracks |

---

**Gulzar-adjacent intuition:** A good LLM answer isn’t a single word — it’s **what the last hundred tokens did to the next one**. That’s **context** and **attention** in one sentence.

**Mini prompt for you:** *Sketch RAG on a whiteboard as three boxes: ingest, index, serve. Where does your “pipeline” analogy break?* (Hint: **freshness** and **citation** trust.)

---

Use this alongside the main question banks in the repo when you want **depth** without losing the thread.
