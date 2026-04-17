# LLM interview notes — *GenAI without the fog*

LLMs are **services** that eat tokens and return distributions. Your job in an interview is to sound like you’ve **operated** them: context, retrieval, safety, cost, evals — not just read the brochure.

**DevOps translation:** Prompting is **config**. RAG is **read-through cache + citations**. Fine-tuning is **shipping a patch** to behavior. Agents are **orchestration** with tools — same story as your favorite workflow engine, just with a stochastic brain.

---

## What’s in this folder

You now have **two versions** of most notes:

- **Deep**: the original, more in-depth version (keeps the classic structure).
- **Snappy**: the rewritten, punchy version for fast reading (`*-snappy.md`).

| Topic | Deep | Snappy | What you’ll grab from it |
|------|------|--------|--------------------------|
| Fundamentals | `llm-fundamentals.md` | `llm-fundamentals-snappy.md` | Transformers, attention, context, limits |
| Prompting | `prompt-engineering.md` | `prompt-engineering-snappy.md` | Reliability, structure, failure modes |
| RAG | `retrieval-augmented-generation-rag.md` | `retrieval-augmented-generation-rag-snappy.md` | Chunking, retrieval, faithfulness, **RAGAS**-style thinking |
| Agents | `ai-agents-and-agentic-systems.md` | `ai-agents-and-agentic-systems-snappy.md` | Loops, tools, guardrails — **observe → act → verify** |
| Fine-tuning | `fine-tuning-and-model-adaptation.md` | `fine-tuning-and-model-adaptation-snappy.md` | LoRA, adapters, when **not** to fine-tune |
| Vector DBs | `vector-databases-and-embeddings.md` | `vector-databases-and-embeddings-snappy.md` | Embeddings, ANN/HNSW — **search at scale** |
| System design | `ai-system-design.md` | `ai-system-design-snappy.md` | End-to-end GenAI systems — **the system design round** |
| LLMOps | `llmops-and-production-ai.md` | `llmops-and-production-ai-snappy.md` | Monitoring, cost, release — **LLMOps** |
| Evals | `evaluation-and-testing.md` | `evaluation-and-testing-snappy.md` | Benchmarks, judge models, regression — **quality gates** |
| Safety / RAI | `ai-safety-ethics-and-responsible-ai-what.md` | `ai-safety-ethics-and-responsible-ai-what-snappy.md` | Bias, policy, misuse — **the grown-up table** |
| Multimodal | `multi-modal-ai.md` | `multi-modal-ai-snappy.md` | Vision + language — **more modalities, same infra** |
| Infra | `ai-infrastructure-and-scalability.md` | `ai-infrastructure-and-scalability-snappy.md` | GPUs, batching, parallelism — **when the bill arrives** |
| Coding | `coding-and-practical-implementation.md` | `coding-and-practical-implementation-snappy.md` | Patterns you can code live |
| Behavioral | `behavioral-and-scenario-based-questions.md` | `behavioral-and-scenario-based-questions-snappy.md` | Stories that don’t sound rehearsed |
| Extras | `additional-llm-interview-topics.md` | `additional-llm-interview-topics-snappy.md` | Scaling laws, RLHF/DPO, MoE, long context, hallucinations, distillation, AWQ/GPTQ, **prompt injection** |
| Efficient deployment | `efficient-llm-deployment.md` | `efficient-llm-deployment-snappy.md` | Quantization, caching, batching, routing — serving playbook |
| Alignment & reasoning | `advanced-alignment-and-reasoning.md` | `advanced-alignment-and-reasoning-snappy.md` | RLHF/DPO, ReAct, agents — the “senior” layer |

---

**Gulzar-adjacent intuition:** A good LLM answer isn’t a single word — it’s **what the last hundred tokens did to the next one**. That’s **context** and **attention** in one sentence.

**Mini prompt for you:** *Sketch RAG on a whiteboard as three boxes: ingest, index, serve. Where does your “pipeline” analogy break?* (Hint: **freshness** and **citation** trust.)

---

Use this alongside the main question banks in the repo when you want **depth** without losing the thread.
