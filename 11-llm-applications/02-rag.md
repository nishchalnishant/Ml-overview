---
module: LLM Applications
topic: Applications
subtopic: Rag
status: unread
tags: [llms, ml, applications-rag]
---
# Retrieval-Augmented Generation (RAG)

---

## The Core Problem

**The problem:** a language model's knowledge is frozen at training time. It cannot access facts that postdate its training cutoff, it cannot access private or proprietary documents, and even for facts it did know at training time, it has no mechanism to verify whether a recalled fact is accurate or confabulated — both look identical to the model's generation process. Asking the model to "remember" a fact is asking it to reconstruct a specific piece of information from billions of compressed parameters, with no error signal if the reconstruction is wrong.

**The core insight:** ground the generation in retrieved documents the model must attend to. Now there is a verifiable source, and if the context contains the answer, the model is much more likely to return it accurately — because it is attending to an explicit source rather than reconstructing from distributed weights. The source is also updatable without retraining.

**When to use RAG:**
- Facts change frequently and the model needs current information
- Answers must be attributed to specific source documents
- The domain is proprietary or too narrow to justify retraining
- Knowledge needs to be updatable without GPU budget

**When not to use RAG:**
- The problem is behavioral (tone, format, style) — fine-tune instead
- The latency budget cannot absorb retrieval overhead (50–300ms)
- The knowledge is static and already well-covered in pretraining data

---

## Document Ingestion

**The problem:** documents come in at arbitrary length. A PDF chapter cannot be retrieved as a unit — it is too long to fit in a prompt, and its retrieval score would be diluted by all the parts irrelevant to the query. The document must be split into retrievable units.

**The core insight:** the size of the retrievable unit determines the precision-context tradeoff. Smaller chunks retrieve more precisely (the retrieved chunk is more likely to be entirely relevant) but provide less context (the answer may span chunk boundaries). Larger chunks provide more context but score lower on retrieval because most of the chunk is off-topic for any given query.

**The mechanics — chunking strategies:**

| Strategy | When to use | Tradeoff |
|:---|:---|:---|
| Fixed-size (token count) | Simple baseline | Splits mid-sentence |
| Sentence splitting | Coherent retrievable units | Short chunks, less context |
| Recursive character | General-purpose default | Respects paragraph → sentence hierarchy |
| Semantic chunking | High-quality retrieval | Expensive (embed each chunk boundary candidate) |
| Structure-aware | Markdown, PDFs with headers | Preserves logical sections |

Chunk size guidance: 256–512 tokens for precise retrieval; 512–1024 tokens as a general default; 1024–2048 tokens when answers often span multiple paragraphs. Overlap of 50–100 tokens between adjacent chunks prevents answers from falling in the gap between chunks.

Attach metadata to each chunk before indexing: source filename, section header, date, document type. This enables filtered retrieval — "only search documents from Q4 2024" — reducing noise without semantic search.

**What breaks:** poor chunking is the single most common cause of RAG failures. If the answer spans the boundary between two chunks and neither chunk contains the full answer, retrieval will return both chunks at lower scores — or miss the answer entirely. Chunk boundaries should respect semantic units (paragraphs, sections), not arbitrary token counts.

---

## Embedding and Retrieval

**The problem:** the query is text. The chunks are text. To find which chunks are relevant to the query, they must be comparable. Keyword matching (BM25) works for exact terms but fails for paraphrase, synonymy, and domain-specific language. Dense embedding-based retrieval handles semantic similarity but misses exact-match cases that keyword search handles trivially.

**The core insight:** map both queries and chunks to dense vectors in a shared semantic space, then find nearest neighbors. Semantically similar content will be geometrically close even if the words differ. For cases where exact-match matters (proper nouns, technical identifiers), combine dense and sparse retrieval via Reciprocal Rank Fusion.

**The mechanics — three retrieval modes:**

Dense retrieval: embed query and chunks with the same model, retrieve by cosine similarity. Handles paraphrase and semantics.

Sparse retrieval (BM25): keyword-based scoring. Fast and strong for exact-match queries, technical terms, and proper nouns.
```
BM25(q, d) = Σ_{t∈q} IDF(t) · f(t,d)·(k₁+1) / (f(t,d) + k₁(1-b+b·|d|/avgdl))
```
where k₁ ≈ 1.5 and b ≈ 0.75 are tuning parameters.

Hybrid search — Reciprocal Rank Fusion (RRF):
```
RRF(d) = Σ_{r∈rankers} 1 / (k + r(d))
```
where k=60 is a common default and r(d) is the rank of document d in each ranker. RRF combines rank lists from dense and sparse retrieval without requiring calibrated scores from each.

**Embedding model selection:** use the same model for indexing and querying. Model updates require full re-indexing. Strong open-source options: BAAI/bge-m3 (multilingual, 8K context), sentence-transformers/all-mpnet-base-v2 (strong English, small). OpenAI text-embedding-3-large for highest quality at cost.

**What breaks:** embedding model quality is the ceiling on retrieval quality. The semantic space may not align with the domain vocabulary of the retrieved documents — a medical embedding model will outperform a general-purpose one for medical retrieval. Model updates require full re-indexing, which is expensive at scale.

---

## Vector Databases and ANN Search

**The problem:** finding exact nearest neighbors in a database of millions of 1536-dimensional vectors requires computing cosine similarity against every vector — O(n) per query. At millions of documents, this is too slow for real-time use.

**The core insight:** approximate nearest neighbor (ANN) algorithms sacrifice exact correctness for logarithmic query time by building an index structure over the vector space.

**The mechanics — two main algorithms:**

HNSW (Hierarchical Navigable Small World): builds a multi-layer graph where higher layers have fewer, longer-range connections. Search starts at the top layer and greedily navigates toward the query, descending to finer layers. O(log n) query time. The `ef_construction` parameter controls build quality; `ef` controls query-time recall vs. speed.

IVF (Inverted File Index): clusters vectors into k Voronoi cells, searches only the nearest cells. IVFPQ combines IVF with Product Quantization for compressed storage at the cost of recall.

**Vector database selection:**

| Database | Deployment | Strengths |
|:---|:---|:---|
| Chroma | Local / self-hosted | Simple, dev-friendly, good for prototyping |
| Pinecone | Managed cloud | Zero-ops, scalable, paid |
| Weaviate | Self-hosted / cloud | Native hybrid search (BM25 + vector) |
| Qdrant | Self-hosted / cloud | Fast, Rust-based, good filter performance |
| pgvector | PostgreSQL extension | Keeps data in existing database |
| FAISS | In-memory library | Fastest for batch search, not a full DB |

**What breaks:** ANN is approximate — some true nearest neighbors will be missed. The recall/speed tradeoff is controlled by index parameters (ef, nprobe) that must be tuned per workload. All-vector memory storage becomes a bottleneck at billions of documents — quantized indexes (IVFPQ) reduce memory at the cost of recall.

---

## Query Optimization

**The problem:** user queries are often short, ambiguous, or framed differently from how documents express the same information. A query like "AAPL Q3 revenue" may fail to retrieve a chunk saying "Apple reported third-quarter earnings of $..." because the lexical and semantic overlap is weak.

### Multi-Query Expansion

**The core insight:** generate multiple reformulations of the original query to cover different phrasings of the same information need. Retrieve for each, then deduplicate results. Higher recall at the cost of more retrieval calls.

### HyDE (Hypothetical Document Embeddings)

**The core insight:** the embedding of a hypothetical answer to the query is closer in embedding space to actual answer-containing documents than the embedding of the short query itself. Generate a plausible answer first, embed it, retrieve using the richer embedding.

```
query → LLM generates hypothetical answer → embed hypothetical answer → retrieve
```

Works best when the user query is short and vague and the relevant documents are detailed and specific.

### Step-Back Prompting

**The core insight:** rewrite the query at a higher level of abstraction before retrieval. A specific question may not match document language, but its generalized form will.

```
User: "What are the side effects of metformin in elderly patients?"
Step-back: "What are general considerations for metformin use?"
```

**What breaks:** query optimization adds latency (one or more LLM calls before retrieval) and cost. Multi-query expansion multiplies retrieval calls by the number of variants. HyDE can generate a confidently wrong hypothetical answer that retrieves the wrong documents — the LLM hallucination problem now infects the retrieval step.

---

## Reranking

**The problem:** ANN retrieval returns candidates quickly but imprecisely — it optimizes for speed, not exact relevance. The top-10 ANN results may include irrelevant chunks that happen to be geometrically close to the query embedding.

**The core insight:** a cross-encoder model that jointly processes the query and each candidate document can compute much more accurate relevance scores than embedding cosine similarity alone — at the cost of running a full model inference per candidate. Use ANN for broad recall, cross-encoder for precise ranking.

**The mechanics — two-stage retrieval:**
1. ANN retrieval: retrieve top 20 candidates fast.
2. Cross-encoder reranking: score each of the 20 against the query, keep top 5.

Bi-encoder (retrieval): embed query and document independently, dot product. Fast, approximate.
Cross-encoder (reranking): concatenate query + document, run through transformer, output relevance score. Slow, accurate.

**What breaks:** reranking adds latency proportional to the number of candidates × cross-encoder inference time. For low-latency applications, this step may be prohibitive. Cross-encoders are fixed-domain — a cross-encoder trained on MS-MARCO may not rank domain-specific technical documents well.

---

## Context Assembly

**The problem:** retrieved chunks must be assembled into a prompt, but their order affects how well the LLM uses them. LLMs have a U-shaped recall pattern over long contexts — information at the beginning and end is recalled better than information in the middle (lost-in-the-middle phenomenon).

**The core insight:** place the most relevant retrieved chunk first in the assembled context. Avoid putting critical information in the middle of a long list of documents.

**The mechanics — prompt structure:**
```
System: Answer using ONLY the provided context.
        If the answer is not in the context, say so.
        Cite sources using [Doc N].

Context:
[1] Source: quarterly-report-q3.pdf
Apple reported third-quarter earnings of $89.5 billion...

[2] Source: analyst-note-2024.pdf
Revenue growth was driven primarily by services...

Question: What was Apple's Q3 revenue?
```

**What breaks:** lost-in-the-middle — the LLM ignores chunks in the middle of a long context. Conflicting retrieved passages cause hedging or arbitrary selection. A strict "answer only from context" instruction can make the model refuse to answer even when it could derive the answer with minimal inference.

---

## Advanced Patterns

### Self-RAG

Fine-tunes the model to emit reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) so retrieval-need, relevance, and answer quality become learned model behaviors rather than external orchestration logic. Requires a specially fine-tuned model — not replicable with prompting alone. Full token taxonomy and interview framing: see [interview-notes/03-retrieval-augmented-generation-rag.md](../10-llms/interview-notes/03-retrieval-augmented-generation-rag.md#self-rag).

### CRAG (Corrective RAG)

**The core insight:** evaluate the quality of retrieved passages. If retrieval quality is low (none of the top-k chunks are relevant), fall back to web search. If quality is ambiguous, combine retrieved docs and web results. Prevents confident hallucination caused by irrelevant retrieved context.

**What breaks:** the retrieval quality evaluator is itself imperfect. The web search fallback adds latency and introduces unverified external content.

### GraphRAG

Build a knowledge graph from documents (entity-relationship triples) so multi-hop and corpus-wide summarization queries can be answered via graph traversal instead of flat chunk retrieval — expensive to construct, wins only when relationships or global themes matter. Local/global search split and tool comparison: see [interview-notes/03-retrieval-augmented-generation-rag.md](../10-llms/interview-notes/03-retrieval-augmented-generation-rag.md#graph-rag).

---

## Evaluation

**The problem:** RAG systems have multiple failure modes at multiple stages: wrong document retrieved, right document retrieved but answer not extracted, answer extracted but unfaithfully stated. A single end-to-end metric cannot distinguish which component failed.

**The core insight:** decompose evaluation into component metrics. Faithfulness measures whether the generated answer is grounded in the retrieved context. Context recall measures whether retrieval surfaced the necessary information. These require different ground truth and different correction strategies.

**RAGAS metrics:**

| Metric | What it measures | How |
|:---|:---|:---|
| Faithfulness | Is the answer grounded in retrieved context? | Decompose answer into atomic claims; check each against context |
| Answer relevancy | Does the answer address the question? | Cosine similarity between question embedding and answer embedding |
| Context precision | Are retrieved chunks relevant to the question? | LLM judges each chunk for relevance |
| Context recall | Did retrieval surface all necessary information? | Check if ground-truth answer is derivable from context |

**What breaks:** RAGAS metrics use an LLM as the judge — the judge can be wrong. Faithfulness evaluation decomposes answers into claims and checks each against context; if the decomposition fails, the score is unreliable. Context recall requires ground-truth answers, which are expensive to collect.

---

## Production Considerations

**Debugging RAG failures:** wrong answers usually trace to one of four causes, in order of likelihood:
1. Stale or missing documents — the answer is not in the index
2. Poor chunking — the answer spans a chunk boundary and neither chunk scores highly
3. Retrieval returning irrelevant chunks — embedding model mismatch with domain vocabulary
4. Prompt not enforcing groundedness — model ignores context in favor of parametric memory

Fix in this order before touching the generation model.

| Challenge | Solution |
|:---|:---|
| Index freshness | Scheduled re-ingestion + delta updates |
| Embedding model changes | Version-pin embeddings; re-index when upgrading |
| Multi-tenant isolation | Namespace/partition by tenant in vector DB |
| Latency | Cache embeddings of frequent queries; tune ANN ef parameter |
| Context overflow | Hard limit on retrieved chunks; summarize if needed |
| Hallucination despite context | Stricter system prompt; faithfulness filter on output |
| Security | Metadata filtering to scope retrieval to user's accessible documents |

*Related: [Hallucination Mitigation](05-hallucination-mitigation.md) | [Agentic Workflows](01-agentic-workflows.md) | [Tuning and Optimization](10-tuning-optimization.md)*

---

## Canonical Interview Q&As

**Q: Design a RAG system for a 10M-document enterprise knowledge base. Walk through every architectural decision.**
A: (1) **Chunking**: split documents into 512-token chunks with 64-token overlap (overlap preserves context at boundaries); use semantic chunking (split at paragraph/section boundaries) over fixed-size to avoid cutting mid-sentence. (2) **Embedding**: use a retrieval-tuned encoder (e5-large, BGE, or domain-fine-tuned model); embed chunks offline into a vector store — chunk embedding is asymmetric (query encoder may differ from document encoder in bi-encoders). (3) **Vector store**: at 10M chunks × 1024 dims × 4 bytes = ~40GB — use FAISS IVF with 10K centroids for sub-100ms retrieval, or a managed store (Pinecone, Weaviate) with HNSW index. (4) **Hybrid retrieval**: combine dense ANN (semantic recall) + BM25 (exact keyword recall) via RRF — neither alone is sufficient; "API v2.3" won't match semantically but BM25 catches it exactly. (5) **Re-ranking**: retrieve top-100 from hybrid, re-rank with a cross-encoder (slower but higher precision) to get top-5. (6) **Generation**: pass top-5 chunks as context with source citations; instruct model to say "I don't know" if documents don't contain the answer. (7) **Eval**: RAGAS framework — context recall (are relevant chunks retrieved?), faithfulness (does answer stay grounded in context?), answer relevance.

**Q: What are the failure modes of RAG and how do you address each?**
A: (1) **Retrieval miss** (correct document not retrieved): fix with hybrid retrieval (BM25+dense), query expansion (generate 3 paraphrases of the query and retrieve for all), and HyDE (generate a hypothetical answer, embed it, retrieve documents similar to that embedding). (2) **Context too long** (retrieved chunks don't fit in context): use re-ranking to select top-3 most relevant chunks rather than top-10; use context compression (extract relevant sentences from each chunk). (3) **Faithfulness failure** (model ignores retrieved context and hallucinates): use explicit grounding instructions ("answer ONLY from provided documents"); add citation requirement ("cite the document section"); use Chain-of-Verification post-generation. (4) **Stale retrieval** (documents updated but embeddings not): implement incremental index updates with document versioning; track document timestamps and re-embed on update; for time-sensitive queries, add recency filter to retrieval. (5) **Multi-hop failure** (answer requires combining info from multiple documents): iterative retrieval — generate partial answer, identify missing info, retrieve again; or use agentic RAG where the model decides when to retrieve.

**Q: What is the difference between RAG and fine-tuning for injecting knowledge, and when do you use each?**
A: **RAG**: retrieves relevant context at inference time. Strengths: knowledge is always up-to-date (add documents to the index); source-attributable (can cite which document); no training cost; can handle large knowledge bases (billions of tokens). Weaknesses: retrieval latency (~100-200ms); retrieval can fail; context window limits how much knowledge fits; requires a vector store in production. **Fine-tuning**: bakes knowledge into weights during training. Strengths: zero retrieval latency; knowledge is always available; can learn domain-specific reasoning patterns (not just facts). Weaknesses: knowledge becomes stale as the world changes; hallucinations increase for tail knowledge (fine-tuning on rare facts causes the model to confabulate similar-sounding facts); expensive to update — must retrain; can't cite sources. **Decision rule**: use RAG when (a) knowledge changes frequently, (b) you need source attribution, (c) the knowledge base is large, or (d) you need to add knowledge post-deployment. Use fine-tuning when (a) you need to change behavior/tone/format, (b) you need to teach reasoning patterns not just facts, (c) latency is critical. In practice: combine both — fine-tune for behavior and domain adaptation, RAG for up-to-date factual grounding.


## Flashcards

**When should you use RAG vs. fine-tuning to give a model new knowledge?** #flashcard
Use RAG when facts change frequently, answers must cite sources, the domain is proprietary/narrow, or knowledge must update without retraining. Use fine-tuning instead when the problem is behavioral (tone, format, style) rather than factual, or when the knowledge is static and already well-covered in pretraining. Avoid RAG if the latency budget can't absorb the 50-300ms retrieval overhead.

**Why does chunk size create a precision/context tradeoff in RAG?** #flashcard
Smaller chunks retrieve more precisely because nearly all their content is relevant to a matching query, but they provide less surrounding context and answers spanning chunk boundaries get missed. Larger chunks preserve more context but score lower on retrieval since most of the chunk is off-topic for any given query. General default is 512-1024 tokens with 50-100 token overlap to avoid losing answers at the boundary.

**Why combine dense and sparse (BM25) retrieval instead of using embeddings alone?** #flashcard
Dense retrieval captures semantic similarity/paraphrase but can miss exact-match terms like proper nouns, part numbers, or technical identifiers. BM25 handles exact matches well but misses paraphrase. Reciprocal Rank Fusion combines both rank lists (RRF(d) = Σ 1/(k + rank)) without needing calibrated scores from either ranker.

**Why use ANN (approximate nearest neighbor) search instead of exact nearest neighbor for retrieval?** #flashcard
Exact nearest neighbor requires computing similarity against every vector — O(n) per query, too slow at millions of documents. ANN algorithms like HNSW (multi-layer navigable graph, O(log n) query time) or IVF (cluster into Voronoi cells, search only nearest clusters) trade a small amount of recall for large speed gains.

**What is HyDE and when does it help retrieval?** #flashcard
HyDE (Hypothetical Document Embeddings) generates a plausible LLM answer to the query first, then embeds and retrieves using that hypothetical answer instead of the raw query — because a full answer's embedding is closer in vector space to real answer-containing documents than a short vague query's embedding is. Risk: if the LLM hallucinates a wrong hypothetical answer, it retrieves the wrong documents, injecting the hallucination problem into retrieval itself.

**Why is reranking a separate stage from initial retrieval, rather than just using better embeddings?** #flashcard
Bi-encoders (used for ANN retrieval) embed query and document independently for speed, but can't model query-document interaction, so they optimize for speed over precision. Cross-encoders process query and document together and score relevance much more accurately, but are too slow to run against the full corpus. So the pattern is: ANN retrieves top-N candidates fast, cross-encoder reranks those N down to a precise top-k.

**What is "lost-in-the-middle" and how does it affect RAG prompt construction?** #flashcard
LLMs recall information at the start and end of a long context better than in the middle (U-shaped recall). This means the most relevant retrieved chunk should be placed first (or last) in the assembled context, not buried among lower-relevance chunks, or the model may effectively ignore it.

**What do the four RAGAS metrics each isolate, and why can't one end-to-end score do this?** #flashcard
RAG can fail at multiple independent stages, so a single score can't localize the failure: faithfulness (is the answer grounded in retrieved context, checked by decomposing into atomic claims), answer relevancy (does the answer address the question), context precision (are retrieved chunks actually relevant), and context recall (did retrieval surface everything needed). Each requires different ground truth and points to a different fix.

**When debugging a RAG system that gives wrong answers, what's the correct order to check causes?** #flashcard
Check in this order before touching the generation model: (1) stale/missing documents — answer isn't in the index at all, (2) poor chunking — answer spans a boundary and no single chunk scores highly, (3) irrelevant retrieval — embedding model mismatched to domain vocabulary, (4) prompt not enforcing groundedness — model ignores context in favor of parametric memory. Most RAG failures are retrieval/chunking problems, not generation problems.
