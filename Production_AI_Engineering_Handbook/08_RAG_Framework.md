# PART 8: RAG DECISION FRAMEWORK

## Goal
To teach candidates how to design, optimize, and debug Retrieval-Augmented Generation (RAG) systems that work reliably in production.

## Mental Model
**RAG = a search engine that feeds an LLM. Fix the search, fix the hallucinations.**
Most RAG failures are *retrieval failures*, not *generation failures*. Before tuning the LLM, verify that the retrieved context is accurate and relevant.

---

## 8.1 RAG vs. Fine-tuning Decision Tree

```text
What do you need the LLM to do?
├── Access proprietary/changing knowledge → RAG.
│   └── Knowledge base updates without retraining.
├── Learn a specific format/style/behavior → Fine-tuning (LoRA).
├── Both knowledge AND style → Fine-tuning + RAG.
└── Simple factual retrieval → Consider keyword search before RAG.
```

---

## 8.2 Chunking Strategy

### Decision Tree
```text
What is the document structure?
├── Fixed-size documents (articles, emails) → Fixed-size chunking (512-1024 tokens).
├── Structured documents (manuals, FAQs) → Semantic chunking (split at section boundaries).
├── Long documents with hierarchical structure → Parent-Child chunking.
│   └── Small chunks (256 tokens) for retrieval, larger parent chunk (2048 tokens) for context.
└── Code → Chunk by function/class boundaries, not arbitrary token count.
```

| Chunk Size | Pros | Cons |
| :--- | :--- | :--- |
| **Small (128-256 tokens)** | High precision, targeted retrieval | Loses surrounding context |
| **Medium (512-1024 tokens)** | Balanced (default choice) | May split mid-thought |
| **Large (2048+ tokens)** | Rich context | Noisy retrieval, higher LLM cost |

### Chunk Overlap
- Use **10-20% overlap** (e.g., 50-100 tokens) to prevent critical information from being cut at chunk boundaries.
- Too much overlap → redundant retrieval, inflated storage cost.

---

## 8.3 Embedding Model Selection

### Decision Tree
```text
What language/domain?
├── English, general purpose → OpenAI text-embedding-3-large OR open-source BAAI/bge-large-en.
├── Multilingual → intfloat/multilingual-e5-large.
├── Code → CodeBERT or OpenAI embedding with code-specific prompts.
├── Biomedical → BioLinkBERT, MedCPT.
└── Latency-critical → gte-small, all-MiniLM-L6-v2 (smaller, faster).
```

| Model | Dimension | Speed | Quality | Cost |
| :--- | :--- | :--- | :--- |:--- |
| **text-embedding-3-large** | 3072 | Fast (API) | Excellent | $$$ |
| **BAAI/bge-large-en-v1.5** | 1024 | Fast (local) | Very Good | Free |
| **all-MiniLM-L6-v2** | 384 | Very Fast | Good | Free |
| **intfloat/e5-large-v2** | 1024 | Fast | Very Good | Free |

---

## 8.4 Retrieval Strategy

### Decision Tree
```text
What type of queries are you handling?
├── Keyword-heavy queries ("What is the patch version on 2024-01-15?") → BM25 (Keyword).
├── Semantic queries ("Find documents about player frustration") → Dense Vector Search.
└── Mix of both → Hybrid Search (BM25 + Dense, combined with RRF).
```

| Strategy | Precision | Recall | Best For |
| :--- | :--- | :--- | :--- |
| **Dense Vector Search** | High semantic | Medium exact | Open-ended Q&A |
| **BM25 (Keyword)** | High exact | Medium semantic | Specific terms, IDs |
| **Hybrid (RRF)** | High | High | Production default |
| **Graph RAG** | Very High | Medium | Multi-hop reasoning, relationships |

### Reciprocal Rank Fusion (RRF)
Combine rankings from BM25 and vector search:
```text
RRF_score(doc) = Σ 1 / (k + rank_i)
k = 60 (constant, prevents division by zero)
```
Effectively boosts documents that rank highly in both searches.

---

## 8.5 Re-ranking

### When to Use
After initial retrieval (top 100 documents), apply a **cross-encoder re-ranker** (e.g., `cross-encoder/ms-marco-MiniLM-L6-v2`) to re-score documents with full attention between query and document.

```text
[Query] + [Top 100 retrieved chunks] → Cross-Encoder → [Top 5 re-ranked chunks] → LLM
```

**Why not use the cross-encoder for all retrieval?**
Cross-encoders cannot be pre-computed for all document pairs (O(n²) complexity). Use a fast bi-encoder to retrieve, then a slow cross-encoder to rerank.

---

## 8.6 Metadata Filtering

### Framework
Before semantic search, filter by metadata to reduce the search space and improve precision.

```text
Example:
User: "What are the patch notes for game version 2.4?"

Without metadata filter: Search all 500k documents.
With metadata filter: Filter to documents where metadata.version == "2.4" (1k documents), then semantic search.
```

Always index metadata fields (game version, date, category) in the vector DB as scalar filters.

---

## 8.7 Vector Database Selection

| Database | Deployment | Strengths | Use Case |
| :--- | :--- | :--- | :--- |
| **Pinecone** | Managed cloud | Zero ops, fast | Quick prototyping, startups |
| **Weaviate** | Hybrid (cloud/self) | GraphQL, hybrid search | Production with metadata |
| **Qdrant** | Self-hosted | Rust-fast, payload filtering | High-perf self-hosted |
| **Chroma** | Local/embedded | Simplest API | Local development |
| **pgvector** | Postgres extension | SQL familiarity, no new infra | Small scale, already using Postgres |
| **Milvus** | Self-hosted | Enterprise-scale | Large-scale production |

---

## 8.8 Hallucination Reduction Strategies

### In-order of implementation
1. **Ground in retrieved context:** Prompt: *"Answer only using the following context. If the answer is not in the context, say 'I don't know'."*
2. **Faithfulness scoring:** Use RAGAS `faithfulness` metric to flag answers not supported by context.
3. **Citation enforcement:** Force the LLM to cite the specific chunk used (structured JSON output).
4. **Confidence thresholds:** If retrieval similarity scores are below a threshold (e.g., < 0.7), refuse to answer.
5. **Self-consistency check:** Generate the same answer 3 times; flag if they disagree.

---

## 8.9 RAG Evaluation (RAGAS Framework)

| Metric | Question Answered |
| :--- | :--- |
| **Context Precision** | Is the retrieved context relevant to the question? |
| **Context Recall** | Was all the necessary information retrieved? |
| **Faithfulness** | Does the answer stick to the retrieved context? |
| **Answer Relevance** | Does the generated answer actually answer the question? |

**Production tip:** Run RAGAS evaluations on a golden test set (100–500 curated Q&A pairs) after every change to the RAG pipeline.

---

## 8.10 RAG Cost Optimization

```text
Cost = (Embedding cost) + (Vector DB query cost) + (LLM token cost)

Reduce embedding cost:
→ Use a smaller, self-hosted embedding model (BAAI/bge-small-en).
→ Cache embeddings for static documents (never re-embed unchanged documents).

Reduce LLM token cost:
→ Context compression: Summarize/compress retrieved chunks before feeding to LLM.
→ Reduce top-K: Send top 3 chunks instead of top 10.
→ Use a smaller LLM (GPT-4o-mini instead of GPT-4o) for simpler queries.
→ Route simple factual queries to keyword search + templates (no LLM needed).
```

---

## 8.11 RAG Failure Modes & Debug Framework

```text
Bad Answer?
├── Is it HALLUCINATED (not in any document)?
│   └── Fix: Improve faithfulness prompting. Add retrieval confidence thresholds.
├── Is the CONTEXT wrong (wrong documents retrieved)?
│   └── Fix: Improve chunking, switch to hybrid search, add metadata filtering.
├── Is the context RIGHT but ANSWER still wrong?
│   └── Fix: Upgrade LLM, improve system prompt, add chain-of-thought.
└── Is it TOO SLOW?
    └── Fix: Cache frequent queries, reduce top-K, use smaller embedding model.
```

---

## Engineering Checklist

- [ ] Have I established a golden evaluation set before building the RAG system?
- [ ] Am I using hybrid search (BM25 + vector) by default?
- [ ] Have I tuned chunk size against the evaluation set?
- [ ] Am I caching embeddings for static documents?
- [ ] Am I filtering by metadata before vector search?
- [ ] Have I added re-ranking for precision-critical applications?
- [ ] Have I measured faithfulness (not just answer quality)?

## Production Considerations

- **Index Freshness:** For dynamic knowledge bases, design an ingestion pipeline that embeds and upserts new documents as they arrive (e.g., Kafka → embedding service → vector DB).
- **Observability:** Log every retrieval query with the top-K results and similarity scores. This is the single most important debugging tool for a RAG system.
- **Multi-tenancy:** If different users should see different documents, use namespace isolation or metadata-based ACL filters in the vector DB.

## Interview Follow-up Questions & Best Answers

**Q: "Your RAG system keeps hallucinating even when the answer is in the documents. What do you check?"**
*Best Answer:* "I isolate the problem by testing retrieval separately from generation. I first check: 'If I give the LLM the correct context directly, does it answer correctly?' If yes, the problem is retrieval. I then inspect the retrieved chunks for the failing queries. Common causes: the chunk boundary cuts off the key sentence, the answer spans multiple chunks that aren't retrieved together (fix: parent-child chunking), or the query is ambiguous and the embedding model retrieves semantically similar but irrelevant chunks (fix: hybrid search + metadata filtering)."
