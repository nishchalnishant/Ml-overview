---
module: Llms
topic: Interview Notes
subtopic: Retrieval Augmented Generation Rag Snappy
status: unread
tags: [llms, ml, interview-notes-retrieval-augm]
---

> _Quick-recall companion. For the full deep-dive, see [retrieval-augmented-generation-rag.md](retrieval-augmented-generation-rag.md)._

# RAG (Retrieval-Augmented Generation) — the Azure/DevOps-fluent version

RAG is the moment you stop asking an LLM to “remember everything” and start treating it like a service that can **retrieve evidence**.

**One-line mental model:** RAG = **read-through cache** for knowledge + **citations** + **grounding**.

---

# Q1: What is RAG, and why is it important?
- **Direct answer:** RAG retrieves relevant external context (docs/search) and injects it into the prompt so answers are **grounded** and up-to-date.
- **Azure/DevOps bridge:** LLM weights = baked container image; RAG = pulling config/artifacts at runtime.
- **Analogy:** Classic romance track remaster: the voice is the same (model), but you bring in clean backing instruments (fresh facts).
- **Mini prompt:** When is RAG better than fine-tuning? → when facts change and you need citations.

---

# Q2: Explain the architecture of a basic RAG system.
- **Direct answer:** Two pipelines:
  - **Offline ingestion:** load → chunk → embed → store.
  - **Online retrieval:** embed query → search → prompt → generate.
- **DevOps bridge:** ingestion is your nightly pipeline; retrieval is your online API path.

```mermaid
graph LR
  subgraph Offline[Offline: ingestion]
    Doc[Docs] --> Chunk[Chunk]
    Chunk --> Embed[Embed]
    Embed --> VDB[(Vector DB)]
  end

  subgraph Online[Online: retrieval]
    Q[User query] --> QE[Embed query]
    QE -->|kNN| VDB
    VDB --> Ctx[Top-k chunks]
    Ctx --> LLM[LLM]
    LLM --> Ans[Answer + citations]
  end
```

---

# Q3: What are the key components of a RAG pipeline?
- **Ingestion:** parsers, chunker, embedding model, vector store, metadata.
- **Retrieval:** query embedding, search (vector/hybrid), reranker, prompt template.
- **Generation:** LLM, output contract (JSON/markdown), citation formatting.
- **Ops:** eval suite, monitoring (latency + quality), feedback loop.

---

# Q4: Chunking strategies — how choose chunk size?
- **Direct answer:** Chunk size is a trade-off between **retrieval precision** and **context completeness**.
- **Rules of thumb:**
  - start ~300–800 tokens; add **overlap** (30–150 tokens)
  - chunk by structure first (headings/sections) not raw characters
- **Fashion analogy:** don’t cut fabric mid-seam; cut at natural stitch lines.
- **Mini prompt:** What happens if chunks are huge? → relevance dilution + wasted tokens.

---

# Q5: Fixed-size vs semantic vs recursive chunking?
- **Fixed-size:** fast, dumb, often fine for clean docs.
- **Semantic:** split by meaning (sentences/sections); better faithfulness.
- **Recursive:** try paragraph → sentence → character; practical default.

---

# Q6: What are embedding models?
- **Direct answer:** Models that map text into dense vectors where semantic similarity becomes distance (cosine / dot product).
- **DevOps bridge:** embeddings are your **index format**; you can’t mix coordinate systems casually.

---

# Q7: How do you choose an embedding model?
- **Criteria:** domain fit, multilingual needs, vector dimension (cost), speed, license.
- **Mini prompt:** What’s the cardinal sin? → indexing with one embedder and querying with another.

---

# Q8: Explain Agentic RAG.
- **Direct answer:** The model iteratively decides what to retrieve, queries tools/search multiple times, verifies, then answers.
- **DevOps bridge:** it’s an orchestrated workflow (observe → act → verify) with audit logs.
- **MI analogy:** captain adjusts field after each ball instead of locking strategy for 20 overs.

---

# Q9: What is hybrid search, and why better than pure vector?
- **Direct answer:** Combine semantic retrieval (vectors) with lexical retrieval (BM25/keyword).
- **When it matters:** IDs, codes, exact phrases, numbers (“invoice #12345”).
- **Mini prompt:** Vector search for numbers—good or risky? → risky; use hybrid.

---

# Q10: What is re-ranking?
- **Direct answer:** A second model (cross-encoder/LLM) re-scores retrieved candidates to improve ordering/precision.
- **Trade-off:** better quality, extra latency.

---

# Q11: Multi-document / multi-hop questions?
- **Direct answer:** Use retrieval + synthesis across multiple chunks; often needs query decomposition.
- **Patterns:**
  - decompose question → retrieve per sub-question → synthesize
  - iterative/agentic retrieval until confidence threshold

---

# Q12: Lost-in-the-middle in RAG?
- **Direct answer:** Even if you retrieve many chunks, the model may ignore middle context.
- **Fixes:** keep top-k small, rerank hard, place best evidence at top/bottom, summarize evidence.

---

# Q13: How do you evaluate a RAG system? (faithfulness, relevance, context precision/recall)
- **Faithfulness:** answer supported by retrieved context (no invented claims).
- **Answer relevance:** answers the question asked.
- **Context precision/recall:** did we retrieve the right stuff (and not too much junk)?
- **DevOps bridge:** build a regression suite; treat prompt/retrieval changes like releases.

---

# Q14: Explain Self-RAG.
- **Direct answer:** The model learns/decides when to retrieve vs answer from parametric knowledge, often with self-critique signals.
- **Mini prompt:** Why retrieve at all if the model “knows”? → freshness + citations + reduced hallucination.

---

# Q15: What is Graph RAG?
- **Direct answer:** Retrieval over a graph of entities/relations (plus text) for better multi-hop reasoning.
- **When to use:** knowledge bases, org charts, dependency graphs, “how are A and B connected?”

---

# Q16: Structured data (tables/SQL) in RAG?
- **Direct answer:** Don’t embed raw tables blindly. Use:
  - schema-aware retrieval (table/column metadata)
  - text summaries of tables
  - tool use: SQL generation + execution + return rows as evidence
- **Azure/DevOps prompt:** Where do you validate? → SQL sandbox + row limits + allow-listed queries.
