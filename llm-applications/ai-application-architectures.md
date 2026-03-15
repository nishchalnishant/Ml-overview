# AI Application Architectures

End-to-end patterns for **conversational AI**, **AI copilots**, **document search**, **recommendations**, and **agents with tools**. Focus: data flow, model interaction, latency, and scalability.

---

## Conversational AI

- **Flow**: User message → (optional) intent/entity detection → build prompt (system + history + message) → LLM → response → update history.
- **State**: Keep conversation history (and optionally summarization) within context limit; truncate or summarize when long.
- **Latency**: Dominated by LLM; use streaming and smaller/faster models where acceptable. **Scale**: Replicas and load balancing; rate limits and queues.
- **Variants**: Open-domain chat; task-oriented (booking, support) with slots and APIs; hybrid (LLM + rules or small classifiers).

---

## AI copilots

- **Flow**: User action (e.g. edit, selection) → context gathered (file, selection, cursor) → prompt built → LLM suggests completion, edit, or explanation → user accepts or rejects.
- **Data**: Editor buffer, selection, language, repo context (e.g. retrieved code). Keep prompt small enough for low latency.
- **Latency**: Critical (sub-second for inline completion). Use small/fast models, caching, and speculative decoding where applicable.
- **Scale**: Per-user or per-org models; batch only when acceptable; prioritize TTFT.

---

## Document search systems

- **Flow**: User query → **embed** query → **vector search** (and/or keyword) → optional **rerank** → top-k chunks. For “answer from docs”: inject chunks into prompt → LLM → answer + citations (RAG).
- **Data**: Documents chunked and embedded; vector DB (and optionally keyword index). Incremental indexing for updates.
- **Latency**: Embedding + retrieval + optional rerank + LLM. Cache embeddings for repeated queries; tune k and reranker.
- **Scale**: Vector DB scaling (replicas, sharding); batch embedding jobs; scale LLM separately. See [RAG](rag.md) and [Vector databases](vector-databases.md).

---

## Recommendation systems

- **Collaborative**: User/item embeddings (from matrix factorization or neural retrieval); recommend items closest to user or to items they liked. Vector DB for approximate nearest neighbor over item embeddings.
- **Content-based**: Embed item features (or descriptions); recommend by similarity to items the user liked.
- **Hybrid**: Combine collaborative and content-based; optionally use LLM to explain or rank candidates.
- **Latency**: Retrieval (vector + filters) is usually fast; LLM for explanation adds latency. **Scale**: Precompute embeddings; vector DB and filters; A/B test ranking models.

---

## AI agents with tools

- **Flow**: User goal → **agent loop**: LLM reasons → tool call (search, code, API, RAG) → observation → repeat until answer or max steps. See [AGENTIC_AI](../AGENTIC_AI/README.md).
- **Data**: Tools (schemas + backends), conversation + tool results in context; optional long-term memory (vector store of facts).
- **Latency**: Multiple LLM calls + tool calls; dominate by reducing steps or using faster models. **Scale**: Replicas for LLM; rate limits and timeouts for tools; cache tool results when safe.
- **Bottlenecks**: Context length (truncate or summarize); tool failures (retry, fallback); cost (limit steps, cache).

---

## Cross-cutting concerns

- **Security**: Validate tool inputs; sandbox code execution; don’t leak secrets in prompts or logs.
- **Observability**: Log prompts (redacted), token counts, latency, tool calls; monitor cost and error rates.
- **Evaluation**: A/B tests; offline metrics (retrieval recall, answer correctness); human review for quality and safety.

---

## Quick revision

- **Conversational**: history in context; streaming; scale with replicas. **Copilots**: low latency; small context; fast/small models.
- **Document search**: embed + retrieve + optional RAG + LLM; scale vector DB and indexing. **Recommendations**: user/item embeddings + vector search; optional LLM for explanation.
- **Agents**: loop (reason → tool → observe); tools and memory; scale and limit steps to control cost and latency.
