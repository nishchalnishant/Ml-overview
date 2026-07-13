# Interview 15 — Production RAG Pipeline (Condensed)

Build a production RAG assistant over EA's internal developer docs (Markdown + Confluence HTML, heavy C++ code blocks) so developers get accurate, cited answers instead of hallucinated API calls.

## Core Architecture
```
Docs → Markdown-header-aware chunking (preserves headers + code blocks)
     → Embeddings → Vector DB (Qdrant, dense + sparse vectors)

Query → Hybrid Search (dense cosine + BM25) → RRF fusion
      → Cross-encoder re-ranker (top 20 → top 5)
      → LLM generation with citation-forced prompt → streamed answer + sources
```
- Chunking: `MarkdownHeaderTextSplitter` — keeps a header + its code block atomic; naive char-splitting breaks C++ functions.
- Hybrid retrieval: dense catches concepts ("add gravity"), BM25 catches exact symbols (`btRigidBody`) — mandatory for technical docs.
- Re-ranker (cross-encoder, e.g. bge-reranker): cheap dense recall at k=50, precise LLM-facing top 3-5.
- Generator: strict system prompt, contexts labeled with IDs, forced inline citations, "I don't know" fallback.

## Talking Points That Signal Seniority
- Proactively flags that fixed-length chunking will slice C++ functions in half — proposes Markdown/AST-aware splitting unprompted.
- Argues hybrid search (dense + BM25 via RRF) is not optional for code/error-code search, not just an enhancement.
- Brings up re-ranking or Map-Reduce summarization to solve context-window limits before being asked.
- Mentions "lost in the middle" — sorts retrieved chunks so top hits sit at the start/end of context.
- Proposes post-generation verification (regex-extract `fb::` symbols, check they exist in retrieved context) to catch hallucinated APIs.
- Calls out Ragas-style automated eval (Context Precision, Faithfulness) as a CI gate on doc updates, not just a demo metric.
- Mentions streaming (`stream=True`) to cut time-to-first-token from ~5s to ~500ms.
- Uses async LLM/embedding calls so FastAPI's event loop isn't blocked during multi-second API calls.

## Top 3 Tradeoffs
- Naive fixed-length chunking vs. semantic/Markdown chunking: naive is simple but destroys code logic; semantic preserves logic at the cost of uneven chunk sizes for vector matching.
- Dense vector search vs. BM25: vectors generalize ("gravity" ≈ "things falling") but blur exact symbol names; BM25 nails exact matches but misses paraphrase — hybrid is the only real answer for technical docs.
- Stuffing more chunks into the LLM vs. re-ranking down to fewer: more context raises recall but costs latency/money and triggers lost-in-the-middle; re-ranking trades a small recall risk for precision and speed.

## Biggest Pitfall
Proposing naive fixed-size chunking (or worse, fine-tuning the LLM on the docs instead of doing retrieval at all) and not recognizing why it breaks code/API accuracy — this is the fastest path from Hire to No Hire.
