---
module: Emerging Topics
topic: Emerging Trends
subtopic: Advanced Rag And Memory
status: unread
tags: [emergingtopics, ml, emerging-trends-advanced-rag-a]
---
# Advanced RAG and Memory Systems

How naive RAG fails in production and what GraphRAG, Agentic RAG, Self-RAG, and long-context architectures actually solve — with the engineering trade-offs that determine which approach to deploy.

---

## 1. Core Concept & Intuition

Naive RAG (embed → retrieve top-k → stuff into context → generate) fails on four categories of questions:

1. **Multi-hop reasoning:** "What is the revenue growth rate of the company that acquired Slack?" requires finding the acquirer (Salesforce), then finding its revenue — two retrieval steps with a dependency.
2. **Aggregation:** "Summarize all mentions of the CEO across these 50 quarterly reports" — no single chunk contains the answer; the answer is the synthesis of many chunks.
3. **Comparative:** "How does our Q3 performance compare to Q3 last year?" — requires simultaneously retrieving two documents and performing a comparison.
4. **Absence:** "Does our contract mention liability caps?" — requires confident assertion that something is NOT present, which naive retrieval cannot establish.

Each advanced RAG paradigm solves a different subset of these failure modes.

---

## 2. Architecture & Mathematics

### 2.1 Retrieval Fundamentals

**Dense retrieval:** embed query q and all documents d_i using a bi-encoder. Score = cosine similarity.

```
score(q, d_i) = (E_q(q) · E_d(d_i)) / (||E_q(q)|| · ||E_d(d_i)||)
```

**BM25 (sparse retrieval):** term-frequency based lexical matching.

```
BM25(q, d) = Σ_{t ∈ q} IDF(t) · (TF(t,d) · (k1+1)) / (TF(t,d) + k1·(1 - b + b·|d|/avgdl))

IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
k1 = 1.5 (term saturation), b = 0.75 (length normalization)
```

BM25 excels at exact keyword matching (product codes, names, rare terms). Dense retrieval excels at semantic similarity (paraphrase, conceptual questions). Neither alone is sufficient.

**Reciprocal Rank Fusion (RRF):**

```
RRF(d) = Σ_{r ∈ rankings} 1 / (k + rank_r(d)),  k=60 (damping constant)
```

Merge ranked lists from BM25 and dense retrieval by summing reciprocal ranks. Documents ranked high in both systems score highest. k=60 means rank 1 is worth 1/61 ≈ 0.016; rank 60 is worth 1/120 ≈ 0.008 — penalizes but doesn't zero out lower ranks. RRF is robust to calibration differences between systems (BM25 scores and cosine similarities are not on the same scale).

**Cross-encoder reranking:** after retrieving top-k candidates, rerank using a cross-encoder that processes (query, document) together — allowing query-document interaction in attention layers.

```
score(q, d) = CrossEncoder([CLS] q [SEP] d [SEP]) → scalar
```

More accurate than bi-encoder (full attention between q and d) but O(k) forward passes vs O(1) for bi-encoder retrieval. Use bi-encoder for recall (retrieve 100), cross-encoder for precision (rerank to top 5).

### 2.2 Chunking Strategies

The fundamental tension: small chunks → high precision retrieval (chunk is specific), poor context (too little text around the answer). Large chunks → rich context, poor retrieval precision (query matches a tiny part of a large chunk).

**Hierarchical chunking (Small-to-Big / Parent-Document Retrieval):**

```
Document → Parent chunks (512 tokens, overlapping by 64)
         → Child chunks (128 tokens, overlapping by 16)

Retrieve by child chunk similarity (precise match)
Return parent chunk as context (rich context)
```

The child chunk finds the right location; the parent chunk provides context. This directly solves the precision-context trade-off.

**Semantic chunking:** split on semantic boundaries (sentence similarity drops below threshold) rather than fixed token counts. Adjacent sentences with high cosine similarity stay in the same chunk; semantically distinct sentences create a new chunk.

```
For sentences s_1, ..., s_n:
  sim(s_i, s_{i+1}) = cosine(embed(s_i), embed(s_{i+1}))
  Split at i if sim(s_i, s_{i+1}) < θ (threshold, typically 0.75)
```

### 2.3 GraphRAG

GraphRAG (Microsoft Research, 2024) solves the aggregation failure mode by building a knowledge graph from the document corpus before any query is issued.

**Offline preprocessing:**

```
1. Chunk documents
2. LLM extracts entity-relationship triples from each chunk:
   (Entity_A, relationship, Entity_B, source_chunk_id)
   e.g., (Apple Inc., acquired, Intel Modem Business, chunk_42)
3. Resolve entity coreference: "Apple" = "Apple Inc." = "AAPL"
4. Build graph: nodes=entities, edges=relationships with chunk provenance
5. Community detection (Louvain algorithm) on the entity graph:
   groups of densely connected entities form communities
6. LLM generates a summary report for each community
```

**Query-time (two modes):**

*Local search:* find entities mentioned in query → retrieve their neighborhoods in the graph → retrieve relevant community reports → generate answer. Best for specific factual queries.

*Global search:* distribute query to all community reports → each reports a partial answer → aggregate partial answers via map-reduce → synthesize final answer. Best for aggregation queries ("what are the main themes?").

**Why GraphRAG beats naive RAG on aggregation:** A query about "main themes in Q4 earnings calls" cannot be answered by any single chunk. GraphRAG's community reports pre-summarize clusters of related information — the LLM synthesizes reports, not raw chunks.

**Cost:** GraphRAG requires O(n_chunks) LLM calls during indexing (entity extraction + community summarization). For a 10,000-chunk corpus, this means ~10K-50K LLM calls at index time. This is acceptable for a static corpus but expensive for frequently updated content.

### 2.4 Self-RAG

Self-RAG (Asai et al., 2023) trains a model to generate special reflection tokens that decide whether retrieval is needed and whether retrieved passages are relevant.

**Reflection tokens (four types):**

```
[Retrieve]: {yes, no, continue}  — should we retrieve for the next sentence?
[ISREL]: {relevant, irrelevant}  — is this retrieved passage relevant to the query?
[ISSUP]: {fully supported, partially supported, no support}  — does the passage support what was generated?
[ISUSE]: {1,2,3,4,5}           — overall utility of this generation for the query
```

**Inference algorithm:**

```
Given query q:
1. Generate [Retrieve] token
   If "yes": retrieve top-k documents {d_1,...,d_k}
             For each d_i: generate [ISREL_i]; discard irrelevant
             For each relevant d_i: generate continuation conditioned on d_i
             Generate [ISSUP] for each continuation segment
   If "no": generate directly without retrieval

2. Score each candidate generation:
   score = P([ISREL]=relevant) × P([ISSUP]=fully) × P([ISUSE]=5)
   
3. Return highest-scoring generation
```

**Training:** The model is fine-tuned on a dataset where reflection tokens are added by a teacher model (GPT-4 generates the "ground truth" reflection tokens). The student model then learns to generate these tokens as part of its output, making retrieval a first-class model decision rather than a pipeline heuristic.

**Key insight:** The model can decide NOT to retrieve. "What is 2+2?" doesn't benefit from retrieval — and a Self-RAG model learns this, reducing unnecessary retrieval latency.

### 2.5 Agentic RAG

Agentic RAG gives an LLM agent control over the retrieval strategy — deciding what to query, how many times, and how to combine results.

**ReAct-style Agentic RAG:**

```python
def agentic_rag(query: str) -> str:
    context = []
    for step in range(max_steps):
        thought_and_action = llm.generate(
            system="You are a research assistant. Think step by step. "
                   "Use retrieve(query) to get information. Answer when ready.",
            messages=context + [{"role": "user", "content": query}]
        )
        
        if "retrieve(" in thought_and_action:
            sub_query = extract_query(thought_and_action)
            retrieved = retrieval_system.search(sub_query, k=5)
            context.append({"role": "tool", "content": retrieved})
        else:
            return extract_final_answer(thought_and_action)
```

**Multi-hop decomposition pattern:**

For "What is the revenue growth of the company that acquired Slack?":
```
Step 1: retrieve("Slack acquisition") → "Salesforce acquired Slack for $27.7B in 2021"
Step 2: retrieve("Salesforce revenue 2020 2021") → "Revenue: $17.1B (2020) → $21.25B (2021)"
Step 3: calculate: (21.25 - 17.1) / 17.1 = 24.3% growth
Answer: "Salesforce, which acquired Slack, grew revenue by 24.3% in the acquisition year"
```

Single-step RAG cannot solve this — the second query depends on the result of the first.

**Adaptive retrieval:** The agent can use different retrieval strategies per step:
- Dense retrieval for semantic queries
- BM25 for exact-match queries (product codes, dates)
- SQL query for structured data
- Web search for real-time information

**Failure mode — hallucinated retrieval:** The agent generates a plausible-sounding query but the retrieval system returns irrelevant results. The agent then generates an answer conditioned on irrelevant context. Prevention: require the agent to generate an [ISREL] check before incorporating retrieval results.

### 2.6 RAPTOR: Recursive Abstractive Processing

RAPTOR (Tree-organized Retrieval, 2024) builds a hierarchical index where each level summarizes clusters from the level below:

```
Level 0: raw chunks (leaf nodes)
Level 1: cluster chunks by semantic similarity → LLM summarizes each cluster
Level 2: cluster level-1 summaries → LLM summarizes each cluster
...
Level K: global document summary (root)

Retrieval: search across all levels simultaneously
```

This allows answering at the right granularity: a specific factual question finds a leaf chunk; a thematic summary question finds a mid-level summary node; a global overview question finds the root.

**Clustering for RAPTOR:** Gaussian Mixture Model on UMAP-reduced embeddings (rather than k-means) — soft cluster membership allows chunks to appear in multiple clusters.

### 2.7 Long-Context Window vs. RAG

The fundamental tension: if a model can process 1M tokens, why not just put all documents in context?

**"Lost in the Middle" problem (Liu et al., 2023):**

```
Performance on multi-doc QA:
Position of relevant document in context:
  First 20%: 90% accuracy
  Middle 40-60%: 55% accuracy  ← sharp drop
  Last 20%: 88% accuracy

At 128K context with 100 documents:
  Documents 40-60 may be nearly invisible to the model
```

**KV cache memory cost:**

```
KV cache per token = 2 × n_layers × n_heads × d_head × 2 bytes (bf16)
For Llama-3-70B: 2 × 80 × 8 × 128 × 2 = 327 KB per token

1M context window:
  KV cache = 1M × 327 KB = 327 GB — requires ~5 A100-80GB GPUs just for KV cache
  
10 concurrent users at 1M context = 3.27 TB — 41 A100s for KV alone
```

**Prefill latency for long contexts:**

```
Prefill is O(n²) in sequence length for attention
At n=128K, prefill is 128K² / 1K² = 16,384× more expensive than 1K context
At n=1M, prefill is 1M² / 1K² = 1M× more expensive

Practical: 1M token prefill on 8×H100 = ~120 seconds — unusable for interactive applications
```

**Decision framework: when to use RAG vs long context:**

| Scenario | RAG | Long Context |
|---|---|---|
| >100K tokens of documents | ✓ | Memory/cost prohibitive |
| Real-time (latency <1s) | ✓ | Prefill latency too high |
| Frequently updated docs | ✓ | Requires re-prefill on update |
| Complex multi-hop reasoning | Agentic RAG | ✓ (if fits) |
| Precise verbatim extraction | Long context better | ✓ |
| Confidentiality (no vector DB) | | ✓ |
| 10-50K token context | Either, depends on quality | ✓ if docs are stable |

**Hybrid architecture (production recommendation):** RAG for retrieval to narrow to 5-10 relevant chunks (3-5K tokens total), then feed into a 32K-context LLM for synthesis. Get the precision of RAG and the coherence of long-context reasoning within an affordable context window.

---

## 3. Trade-offs & System Design Implications

### Latency Budget

```
Naive RAG latency breakdown:
  Embedding query:      ~10ms  (single forward pass, small model)
  ANN search:           ~5ms   (HNSW lookup, 1M docs)
  Fetch chunks:         ~5ms   (key-value lookup)
  LLM prefill (2K ctx): ~200ms
  LLM decode (300 tok): ~600ms
  Total:                ~820ms

Agentic RAG (3 hops):
  3× retrieval loop = 3 × (10+5+5+200) ms = 660ms retrieval overhead
  Plus LLM decode    = ~600ms
  Total:             ~1260ms (1.5× naive RAG)

GraphRAG (global search):
  Community report retrieval: ~20ms
  Map-reduce LLM calls:       ~2000ms (parallel to ~500ms with 4 concurrent)
  Final synthesis:            ~600ms
  Total:                      ~1100ms (with parallelism)
```

### Chunking Hyperparameters

```
Chunk size (in tokens):
  128-256:  high retrieval precision, poor context, good for FAQ
  512:      standard, good general purpose
  1024:     better for technical documents with long explanations
  2048+:    poor retrieval precision, better for narrative documents

Overlap:
  0:        hard boundaries, answers near chunk boundaries are missed
  10-15%:   good general rule (e.g., 64-token overlap for 512-token chunks)
  25%+:     redundancy overhead, index size grows
```

### Embedding Model Selection

| Model | Dim | MTEB Score | Tokens/query | Use case |
|---|---|---|---|---|
| text-embedding-3-small | 1536 | 62.3 | 8192 | Cost-sensitive |
| text-embedding-3-large | 3072 | 64.6 | 8192 | Best OpenAI |
| BGE-M3 | 1024 | 66.0 | 8192 | Open source, best hybrid |
| E5-mistral-7b | 4096 | 66.6 | 32768 | Long document retrieval |
| ColBERT v2 | 128 per token | — | — | Multi-vector, highest precision |

For production: BGE-M3 (supports dense + sparse + multi-vector in one model) or text-embedding-3-large (lowest integration cost). Deploy BGE-M3 locally to avoid per-query API costs at scale.

---

## 4. Canonical Interview Q&As

**Q1: Explain why naive RAG fails on multi-hop reasoning queries and how Agentic RAG solves this. What are the failure modes of Agentic RAG?**

Naive RAG retrieves once using the original query as the embedding. For "what is the growth rate of the Slack acquirer?" the query embeds to a vector near "Slack acquisition" — it retrieves chunks about the acquisition but likely not revenue data. Even if the acquisition chunk mentions Salesforce, the model then needs Salesforce revenue data which wasn't retrieved. The model hallucinates or gives up. Fundamentally: single-step retrieval cannot satisfy queries where the information need is only known after the first retrieval result.

Agentic RAG solves this by making retrieval a tool in a reasoning loop. After the first retrieval reveals "Salesforce acquired Slack," the agent formulates a second targeted query for Salesforce revenue. Each retrieval step is conditioned on previous results.

Failure modes: (1) Retrieval hallucination — agent generates a confident query but the corpus doesn't contain the answer; the agent then generates an answer from irrelevant context. Mitigation: require explicit "information found / not found" classification after each retrieval step. (2) Retrieval loop — agent keeps retrieving variations of the same query without convergence; add action deduplication or a max-steps budget. (3) Compounding errors — an error in hop 1 (wrong entity identified) propagates to hop 2 with the wrong query; mitigation is harder, but explicit entity linking (resolve to canonical identifiers) reduces this. (4) Cost explosion — each hop is an LLM call + retrieval; for queries requiring 5+ hops, this becomes expensive. Cap max steps and implement query planning (generate the full multi-step plan before executing).

**Q2: Describe the GraphRAG architecture and explain why it outperforms naive RAG on "global" queries. What are the practical limitations?**

GraphRAG builds a knowledge graph during an offline preprocessing phase: LLMs extract entity-relationship triples from all document chunks, entities are resolved and linked, and a community detection algorithm (Louvain) identifies clusters of densely connected entities. Each community gets an LLM-generated summary report that synthesizes the key information about that cluster of entities.

At query time, "global search" distributes the query to all community reports and runs a map-reduce: each report independently generates a partial answer, then a reduce step synthesizes partial answers into a final response. This works for aggregation queries ("what are the main themes?", "which topics appear most frequently?") because the community reports pre-aggregate information that no single chunk contains.

Naive RAG fails these queries because: top-k retrieval returns isolated chunks with local context, not synthesized overviews; and the LLM context window cannot accommodate all relevant chunks simultaneously.

Practical limitations: (1) Indexing cost — building the graph requires O(n_chunks) LLM calls for entity extraction and O(n_communities) calls for summaries; for 100K chunks, this can mean $500-2000 in API costs. (2) Update latency — adding a new document requires re-extracting entities, merging them into the graph, re-running community detection (which can globally reorganize communities), and re-generating affected community reports. Not suitable for frequently updated corpora. (3) Entity extraction quality — LLM extraction introduces hallucinated or missed entities; errors propagate to graph structure and community reports. (4) Graph size — for very large corpora, the graph can become too large for the community detection algorithm to run efficiently.

**Q3: What is the mathematical difference between BM25 and dense retrieval? When does each outperform the other, and what does RRF actually do statistically?**

BM25 is a lexical scoring function: it counts how often query terms appear in a document, normalized by document length and diminished by term frequency saturation (the k1 parameter prevents a term appearing 100 times from being 100× more relevant than appearing once). BM25 has no notion of semantics — "automobile" and "car" are completely unrelated to BM25 unless both appear in the query.

Dense retrieval maps both query and document to a continuous vector space where semantic similarity corresponds to geometric proximity. "Automobile" and "car" are nearby vectors. Dense retrieval generalizes across synonyms, paraphrases, and conceptual relationships.

BM25 outperforms dense retrieval when: the query contains rare proper nouns, product codes, or technical terms that the dense embedding model has never seen during training (OOV problem — dense models collapse rare terms to their nearest known neighbor); when exact phrase matching is critical.

Dense retrieval outperforms when: the query is semantically expressed differently from the document ("how to reduce server costs" finds documents about "cloud cost optimization"); cross-lingual retrieval; queries about abstract concepts.

RRF statistically: it's a rank aggregation method that is robust to score scale differences. BM25 scores are in [0, ~30+] depending on document length; cosine similarities are in [0, 1]. Directly combining these requires careful calibration. RRF avoids this by discarding the actual scores and working only with ranks. The k=60 constant ensures that the difference between rank 1 and rank 2 is less dramatic (1/61 vs 1/62 ≈ 0.015 vs 0.016) than the difference between rank 1 and rank 100 (1/61 vs 1/160 ≈ 0.016 vs 0.006). RRF is theoretically motivated by the Plackett-Luce model of ranking — the optimal combination of independent rankers under this model approximates RRF.

**Q4: Design a RAG system for a legal document QA product (10M pages of contracts). Walk through chunking strategy, indexing, retrieval, and generation — with specific component choices and their justifications.**

**Chunking:** Contracts have hierarchical structure (sections, clauses, sub-clauses). Use semantic chunking at the clause level: split on section headings and sentence-level semantic breaks. Target 512 tokens per chunk. Maintain parent chunk (full section, up to 2048 tokens) for context retrieval. Store metadata: document ID, section number, effective date, party names.

**Indexing:** Two-tier index. (1) BM25 index on clause text + party names + contract type. Legal documents are full of specific terms (defined terms in ALL CAPS, clause numbers) that BM25 handles well. (2) Dense index using a legal-domain embedding model (LegalBERT fine-tuned for retrieval, or BGE-M3). Embed clause text only (not metadata) to match query embedding at query time. Store in Qdrant with metadata filtering support. Total index size: 10M pages × ~50 clauses/page × 512 tokens/clause = 500M chunks × 1024-dim float32 = 2TB for dense vectors — requires sharding across multiple Qdrant nodes.

**Retrieval:** Hybrid BM25 + dense with RRF (k=60). Pre-filter by metadata: contract type, effective date range, party name (if specified in the query). Filter first (Qdrant supports pre-filter on metadata before ANN search), then retrieve top-50 candidates, then cross-encoder reranking to top-5. Use a legal-domain cross-encoder for reranking.

**Generation:** Use a 32K-context LLM. Inject the top-5 chunks (with section context from parent chunks) as system context. Use structured output (JSON schema) for extraction tasks. Add explicit "not found" handling: if no chunk has ISSUP > 0.5, return "this information was not found in the contract" rather than hallucinating.

**Critical production features:** (1) Citation tracking — every claim in the output must reference a specific chunk ID and page number. Implement by requiring the LLM to cite [chunk_id] inline. (2) Confidence scoring — legal consequences of errors are high; expose per-field confidence and flag low-confidence extractions for human review. (3) Version management — contracts are amended; the index must track document versions and support "as of date" queries.

**Q5: How does Self-RAG differ from standard RAG in its training objective, and what practical capability does this enable that standard RAG lacks?**

Standard RAG is a pipeline: retrieval and generation are separate systems with separate training objectives. The retriever is trained on retrieval metrics (NDCG, recall@k); the generator is trained on generation quality (perplexity, RLHF). They are optimized independently and may not coordinate — the generator has no mechanism to signal that it didn't find the retrieved passage useful.

Self-RAG trains a single model end-to-end to generate both content and reflection tokens. The reflection tokens ([Retrieve], [ISREL], [ISSUP], [ISUSE]) are treated as ordinary vocabulary — the model learns to generate them as part of the output. Training requires a dataset where these tokens are labeled (typically generated by GPT-4 on existing QA datasets). The training objective is standard next-token prediction over a sequence that includes both content tokens and reflection tokens.

The practical capability this enables: adaptive retrieval. The model decides per-sentence whether retrieval would help. For a factual claim that requires external knowledge, it generates [Retrieve=yes] and triggers retrieval. For a continuation that follows logically from what was already said, it generates [Retrieve=no] and continues without retrieval. This is impossible in standard RAG where retrieval is always triggered. Empirically, Self-RAG achieves higher accuracy than standard RAG on open-domain QA while generating fewer hallucinations, because the model can assert confidence in what it knows and only retrieve when genuinely uncertain.

The [ISSUP] token is particularly valuable: it provides per-sentence attribution that tells you which retrieved chunk supports which claim. This is a form of automatic citation generation and factuality checking that emerges from the training objective.

## Flashcards

**Dense retrieval for semantic queries?** #flashcard
Dense retrieval for semantic queries

**BM25 for exact-match queries (product codes, dates)?** #flashcard
BM25 for exact-match queries (product codes, dates)

**SQL query for structured data?** #flashcard
SQL query for structured data

**Web search for real-time information?** #flashcard
Web search for real-time information
