# Interview 12 — Customer Support RAG Assistant (Condensed)

CS agents waste 60% of time searching Confluence/Jira for fixes. Build a RAG system that answers agent questions with citations, respecting per-agent document permissions and near-real-time freshness.

## Clarifying Questions to Ask
- ACL on documents? → Yes, Tier1 agents can't see unannounced-game docs; must filter per-agent.
- Index freshness SLA? → New Confluence page must be searchable within 15 min.
- Hallucination tolerance? → Zero. Must say "I don't know" rather than invent policy (e.g. refunds).
- Source volume? → ~50k Confluence pages + 2M resolved Jira tickets.
- Document formats to support? → Confluence HTML, Jira JSON, PDFs (drives chunking strategy).
- Evaluation method agreed upfront? → Establishes need for Ragas/TruLens style eval, not vibes.

## Core Architecture
```
Confluence/Jira Webhooks → Semantic (header-based) Chunking → Embed
        → Vector DB (Qdrant/Pinecone) w/ ACL metadata {allowed_roles}
Agent Query → Auth→roles → Embed → Hybrid Search (vector + BM25) w/ role filter
        → Top-K chunks → Strict prompt (context-only, cite sources) → LLM → Answer + citations
```
- Vector DB does ACL as a **pre-filter** (not post-filter) — preserves correct top_k.
- Hybrid search (dense + BM25): dense misses exact error codes like "0x887A0005"; BM25 catches them.
- temperature=0, system prompt forces "context insufficient" fallback — kills hallucination risk.
- Ingestion via webhook→queue→worker (not nightly batch) to hit the 15-min freshness bar.

## Talking Points That Signal Seniority
- State explicitly: ACL filtering must happen inside the vector query, not as a post-processing step, or top_k gets silently broken.
- Propose Hybrid Search (BM25 + vectors) unprompted for exact product/error codes.
- Mention a Cross-Encoder re-ranker (e.g. bge-reranker or Cohere Rerank): retrieve 50 broad, rerank to top 3 for precision.
- Propose semantic caching (Redis + cheap embedding, similarity >0.98) to cut LLM cost 40-60% on repeat questions.
- Flag "lost in the middle" — keep top_k small or rerank rather than dumping 20 chunks at the LLM.
- Raise recency bias in scoring (`final_score = similarity + f(doc_date)`) to resolve conflicting document versions.
- Note that stale vectors need deterministic upsert IDs (`hash(url+chunk_idx)`) plus orphan deletion on doc update.
- If asked about player-facing exposure: separate collection, prompt-injection guardrails, PII scrubbing, low-confidence fallback to human — as a checklist, unprompted.

## Top 3 Tradeoffs
- **Header/semantic chunking vs fixed-token chunking** — token chunking is simple but cuts sentences mid-thought, wrecking retrieval quality; semantic chunking costs parsing complexity but is worth it.
- **Dense vs hybrid (dense+BM25) search** — pure dense search misses exact error codes/product IDs that tech support constantly references; hybrid costs more infra but is non-negotiable here.
- **RAG vs fine-tuning** — fine-tuning teaches style, not facts, and still hallucinates; RAG is the only sound approach for a fact-grounded support bot.

## Toughest Follow-ups
**Q: Costs $50k/mo in OpenAI fees at 100k queries/day — cut cost without swapping model.**
A: Semantic caching — embed incoming queries, vector-search against a Redis cache of previously answered questions, return cached response above ~0.98 similarity. Skips both Qdrant and the LLM call entirely for repeat questions, typically 40-60% cost reduction since support questions repeat heavily.

**Q: Retrieved chunks conflict — one from a stale 2021 doc, one from 2024.**
A: Add recency biasing to the retrieval score (blend vector similarity with document age) so newer docs rank higher, and inject document dates into the prompt with an explicit instruction to prefer the most recent source when sources conflict.

**Q: Expose this bot directly to players, not just internal agents — what's strictly required first?**
A: Segregate data into a public-only collection (no internal Jira/unannounced content), add input guardrails against prompt injection, scrub PII from chat, and force a fallback to a human agent whenever retrieval confidence is low — hallucinating to a paying customer is a different risk tier than to an internal agent.

## Biggest Pitfall
Proposing fine-tuning on Confluence/Jira data as a substitute for RAG — it signals a fundamental confusion between LLM memorization and hallucination and is close to an automatic No Hire.
