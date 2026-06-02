---
module: Llms
topic: Interview Notes
subtopic: Vector Databases And Embeddings
status: unread
tags: [llms, ml, interview-notes-vector-databas]
---
# Vector Databases and Embeddings — Interview Notes

---

## What Embeddings Are and Why They Exist

**The problem**: semantic search requires comparing the meaning of a query against a large corpus. You cannot do this with exact string matching — "refund policy" and "how do I get my money back" mean the same thing but share no words. You need a representation where similar meanings are computationally close.

**The core insight**: train a neural network to produce numerical vectors where semantically similar texts map to geometrically nearby points. Semantic similarity becomes a distance computation — fast, scalable, and language-independent.

**The mechanics**: an embedding model takes a text span as input and outputs a fixed-size vector in R^d. The model is trained with a contrastive objective: positive pairs (semantically similar texts) are pushed close together; negative pairs are pushed apart. For retrieval specifically, training uses (query, relevant passage) pairs so the embedding space directly encodes retrieval relevance.

At inference: query is embedded to q, documents are embedded to {d1, d2, ..., dN}, and similarity is computed as:
- **Cosine similarity**: (q·d) / (||q|| ||d||) — measures directional alignment; scale-invariant
- **Dot product**: q·d — equivalent to cosine if vectors are L2-normalized; faster if you skip normalization
- **L2 distance**: ||q - d||² — finds absolute proximity; valid but less common for retrieval

If an embedding model normalizes its output (most do), cosine and dot product are equivalent. Always match your similarity metric to what the model was trained with — using L2 on a model trained with cosine can degrade retrieval quality.

**What breaks**: embeddings encode what was similar in the training distribution. Out-of-domain text (specialized legal or medical jargon, rare product codes, code in a non-English context) may not embed accurately because the training distribution did not represent it well. Embeddings are also insensitive to exact string matching — "SKU-7734-B" embeds near other product descriptions, not at the exact identifier. This is why hybrid search (dense + sparse) is necessary for enterprise use cases.

**What the interviewer is testing**: do you understand that embeddings are trained representations, not a magic semantic lookup — and that their accuracy is bounded by the training distribution?

**Common traps**: assuming high cosine similarity means factual correctness (semantic proximity ≠ factual entailment); not knowing that similarity metric must match the model's training (cosine vs. dot product vs. L2); confusing token embeddings (per-token, part of the model) with document embeddings (pooled representation, produced by a separate encoder).

---

## How Embedding Models Produce Vectors

**The problem**: text is variable-length; a neural network needs fixed-size inputs. You need to transform an arbitrary-length sequence of tokens into a single vector that captures the overall meaning.

**The core insight**: run a transformer encoder over the token sequence to produce one hidden state per token (each token is contextualized by attending to all others), then aggregate (pool) those hidden states into a single vector. The pooling step is the compression from sequence to fixed-size representation.

**The mechanics**:

1. Tokenize: text → subword token IDs using the model's tokenizer.
2. Encode: pass token IDs through the transformer encoder. Output: one hidden state vector per token, each of size d_hidden.
3. Pool: aggregate token hidden states into one vector.
   - **Mean pooling**: average over all non-padding token positions. Most common.
   - **CLS token pooling**: use the hidden state of the special [CLS] token (BERT-style). Designed for classification but used for embeddings in some models.
   - **Attention pooling**: learn a weighted combination of token states. More expressive; rarely used in standard embedding models.
4. Normalize: L2-normalize the output vector so cosine similarity equals dot product.

**What breaks**: pooling over padding tokens without a mask inflates the average toward the padding token's representation. Mean pooling averages equally over all tokens — a 500-token document gives equal weight to the main claim and a boilerplate disclaimer. This can be addressed with weighted mean pooling or by using better chunking before embedding.

**What the interviewer is testing**: whether you can explain where the vector comes from — specifically what pooling is and why it matters.

**Common traps**: not knowing that different pooling strategies (mean vs. CLS) produce different representations with different quality profiles; forgetting L2 normalization (models trained with cosine loss expect normalized vectors; unnormalized vectors produce wrong similarity scores); using the same model but with different tokenizer truncation settings at ingestion vs. query time.

---

## Sparse vs. Dense Embeddings

**The problem**: there are two fundamentally different ways to represent text for retrieval. Each works well in cases where the other fails. Choosing only one means accepting systematic failures on a subset of queries.

**The core insight**: sparse representations (BM25/TF-IDF) are exact — they score documents by weighted term overlap with the query. Dense representations (neural embeddings) are semantic — they score by meaning proximity regardless of exact terms. Exact-match queries fail in dense search; paraphrase queries fail in sparse search.

**The mechanics**:

**Sparse (BM25)**: represents each document as a vector over the vocabulary where most entries are zero (only present terms are non-zero). BM25 score:
```
score(D, Q) = Σ IDF(qi) · (f(qi, D) · (k1 + 1)) / (f(qi, D) + k1 · (1 - b + b · |D|/avgdl))
```
f(qi, D) = term frequency of query term i in document D. IDF = inverse document frequency. k1 and b are tuning parameters. Implemented efficiently via inverted index. Latency is O(|vocabulary terms in query|) per query.

**Dense**: represents each document as a d-dimensional float vector. Similarity via cosine/dot product. Retrieved via ANN index (see HNSW below). Encodes semantics. Fails on rare terms, identifiers, exact strings.

**Hybrid**: retrieve from both independently; merge candidate lists (typically with RRF). Gets the best of both: exact-match recall from BM25, semantic recall from dense. Standard in production systems handling mixed query types.

**What breaks**: dense-only retrieval fails on product IDs, error codes, rare acronyms. Sparse-only retrieval fails on paraphrases, conceptual questions, cross-lingual queries. Neither failure mode is obvious from aggregate metrics — they appear as tail query failures.

**What the interviewer is testing**: whether you know the specific failure mode of each retrieval type and can justify using hybrid search for production systems.

**Common traps**: saying "dense search is always better because it understands semantics" (it systematically fails on exact-match queries); not being able to describe what BM25 does concretely; proposing hybrid search without being able to explain RRF.

---

## Vector Databases: What They Are and Why They Exist

**The problem**: you have embedded 10 million documents. A user submits a query, which you embed into a vector q. You need to find the top-K most similar vectors. Brute-force scan (compute similarity against all 10M vectors) takes seconds and is unacceptable for production latency requirements. A relational database's B-tree index is designed for exact match and range queries — it provides no benefit for nearest-neighbor queries in high-dimensional space.

**The core insight**: build a specialized index structure that partitions the vector space so that queries can be answered by examining a small fraction of the total vectors. Trade a small amount of recall (approximate nearest neighbor, not exact) for orders-of-magnitude speed improvement.

**The mechanics**: a vector database stores:
- Embedding vectors (the float arrays)
- Raw text or document IDs (references to the original content)
- Metadata (source, date, tenant_id, access level, etc.)

And provides:
- ANN index for fast similarity search (HNSW or IVF+PQ)
- Metadata filtering (combine similarity search with structured predicates)
- CRUD operations (upsert, delete, update)

Choices: Pinecone (managed, serverless), Weaviate (open-source, graph features), Qdrant (open-source, Rust-based, efficient filtering), pgvector (PostgreSQL extension, no new infrastructure), Milvus/Zilliz (high-scale, distributed).

**What breaks**: vector databases are approximate — ANN search can miss the true nearest neighbor. Metadata filtering interacts with ANN search: some DBs pre-filter before ANN search (can reduce recall), others post-filter after ANN (can reduce effective top-K). Know which your database does. At very high dimensionality (>3000), ANN index quality degrades — the "curse of dimensionality" makes all vectors approximately equidistant.

**What the interviewer is testing**: whether you understand the distinction between approximate and exact nearest neighbor search, and why approximate is necessary at scale.

**Common traps**: treating vector DB similarity search as exact (it is approximate by design); not knowing about the pre-filter vs. post-filter difference (affects recall under strict metadata filters); using a relational DB with a cosine function (works at thousands of records, catastrophically slow at millions).

---

## HNSW: How ANN Indexing Works

**The problem**: with 10 million vectors, brute-force nearest-neighbor search requires 10 million similarity computations per query. At d=768 dimensions, each computation is ~1536 floating point operations. Total: ~15 billion FLOPs per query. This is seconds, not milliseconds.

**The core insight**: organize vectors into a hierarchical graph where long-distance traversal is possible at the top level (few nodes, long edges) and precise local search is possible at the bottom level (many nodes, short edges). Search starts at the top, finds an approximate region, then refines locally.

**The mechanics — HNSW (Hierarchical Navigable Small World)**:

HNSW builds a layered proximity graph:

- **Layer 0 (base layer)**: all vectors, each connected to M nearest neighbors (dense connections)
- **Layer 1**: a random subset of vectors (~1/e of layer 0), connected to M neighbors in this sparser subgraph
- **Layer 2**: sparser still (~1/e of layer 1)
- ... up to log(N) layers

**Build**: each vector is assigned a random max layer. Starting from the top, greedily insert the vector by finding the M nearest neighbors at each layer (using greedy search within the layer) and adding bidirectional edges.

**Query**: start at the top-layer entry point. Greedily traverse toward the query vector (follow edges to the nearest neighbor at each step). At each layer, find the best entry point for the layer below. At layer 0, perform beam search with a frontier of size ef (efSearch parameter) to find the top-K candidates.

Key parameters:
- **M**: connections per node. More connections = better recall, more memory. Typical: 16–64.
- **efConstruction**: beam size during index build. Higher = better quality index, slower build. Typical: 100–400.
- **efSearch**: beam size during query. Higher = better recall, slower query. Tune to meet recall SLA.

Query complexity: approximately O(log N) expected, versus O(N) brute-force.

**What breaks**: HNSW is memory-intensive — each node stores M connection lists. For 10M vectors at d=768 (fp32), the raw vectors use ~28 GB; HNSW index adds ~5–10 GB of graph structure. Inserts are O(log N) but require locking for concurrent updates. Deletes in HNSW are typically "soft deletes" — the vector is marked deleted but the graph structure is not rebuilt.

**What the interviewer is testing**: whether you understand why ANN is necessary and the basic mechanism — coarse search at high layers, refine at low layers.

**Common traps**: thinking HNSW guarantees finding the true nearest neighbor (it's approximate); not knowing that efSearch is a tunable query-time parameter that trades recall for speed; confusing HNSW with IVF (IVF partitions the space into Voronoi cells; HNSW builds a proximity graph — different structures with different tradeoffs).

---

## Embedding Dimensionality and Quantization

**The problem**: you have 100M document embeddings at d=1536 dimensions (OpenAI ada-002 size), stored as float32. Storage: 100M × 1536 × 4 bytes = ~576 GB. This does not fit in GPU memory for fast ANN search, and the cost of storing and serving this index is significant.

**The core insight**: most of the information in a high-dimensional float32 vector is recoverable from a lower-precision representation. You can sacrifice a small amount of retrieval accuracy for large reductions in memory footprint.

**The mechanics — quantization approaches**:

**Scalar quantization (INT8)**: map each float32 dimension to an 8-bit integer by linear scaling. 4× memory reduction (fp32 → int8). Typical recall degradation: <2% at top-5. This is the standard first optimization.

**Binary quantization**: represent each dimension as a single bit (sign of the float). 32× memory reduction. Recall degradation is larger — best used with re-scoring: retrieve a large candidate set with binary vectors, then re-score with full precision. Works best for models trained with binary quantization in mind.

**Product Quantization (PQ)**: divide the d-dimensional vector into M subspaces of d/M dimensions each. Independently quantize each subspace to one of K centroids. Store centroid index (typically log2(K) bits per subspace). Enables approximate distance computation without reconstructing the full vector. The FAISS library implements IVFPQ (IVF clustering combined with PQ compression).

**Matryoshka Representation Learning (MRL)**: train the embedding model so that the first 256 dimensions alone carry most of the retrieval signal (the first 512 even more, the first 1024 more still). You can then truncate vectors to a shorter length (e.g., 256 from 1536) with modest recall loss. Used in OpenAI's text-embedding-3 models and BGE models.

**What breaks**: quantization reduces recall. Always benchmark recall@K before and after quantization on your actual query distribution — generic benchmarks may not reflect your domain. After quantizing, ANN index parameters (M, efSearch) need retuning because quantization changes the neighborhood structure.

**What the interviewer is testing**: whether you can explain the memory-recall tradeoff concretely and know which technique to apply when.

**Common traps**: applying quantization without re-evaluating recall (you may have degraded a working system); not knowing about MRL as a model-level solution to the dimensionality problem; treating INT8 and binary quantization as equivalent (binary is much more aggressive; INT8 is usually the right first step).

---

## Choosing and Evaluating an Embedding Model

**The problem**: there are dozens of embedding models. The MTEB leaderboard shows one model ranking #1. You deploy it, retrieval quality is worse than your baseline. Why? Because MTEB measures general retrieval quality across diverse benchmarks — your domain may be different.

**The core insight**: embedding quality is task- and domain-specific. The only way to select a model for your production use case is to evaluate it on your actual query distribution against your actual document corpus. MTEB rankings are a useful filter, not a selection criterion.

**The mechanics — evaluation protocol**:

1. Create a retrieval evaluation set: 100–500 (query, relevant document IDs) pairs from your domain. Sources: human annotation, click logs, LLM-generated synthetic pairs.
2. Embed all queries and all corpus documents with each candidate model.
3. Compute retrieval metrics per model:
   - **Recall@K**: fraction of queries where the relevant document appears in the top-K results. Most important for RAG.
   - **MRR (Mean Reciprocal Rank)**: 1/rank of the first relevant result, averaged over queries. Penalizes relevant documents at lower ranks.
   - **NDCG@K**: normalized discounted cumulative gain; useful when multiple relevance levels exist.
4. Run end-to-end RAG faithfulness evaluation on a subset (not just retrieval metrics — poor embedding quality can produce good retrieval metrics but poor answer quality in edge cases).

Key selection criteria:
- **Max sequence length**: must equal or exceed your chunk size. If chunks are 512 tokens, models with 256-token max length will truncate your chunks.
- **Dimensionality**: higher d → better recall ceiling, higher storage cost.
- **Latency**: self-hosted GPU vs. API (cloud API adds network latency, sends data off-premise).
- **Domain coverage**: models trained on domain-specific data (legal, biomedical, code) outperform general models in those domains.

**What breaks**: evaluating on publicly available benchmarks without domain data can select the wrong model. Selecting a model based on average recall without inspecting its failure cases (which query types does it miss?) misses systematic gaps.

**What the interviewer is testing**: whether you would run a domain-specific evaluation, not just defer to public rankings.

**Common traps**: using MTEB rankings as the selection criterion without domain validation; selecting a model with max sequence length shorter than your chunk size (silent truncation degrades embeddings); not knowing that fine-tuning the embedding model (see below) is an option when off-the-shelf models underperform.

---

## Fine-Tuning an Embedding Model for a Domain

**The problem**: your retrieval quality is poor despite good chunking and hybrid search. The embedding model simply doesn't understand what "relevant" means in your domain. Legal contract clauses, biomedical abstracts, proprietary product descriptions — the off-the-shelf model's geometry doesn't reflect domain-specific relevance.

**The core insight**: the embedding model was trained to make documents similar if they are semantically similar according to its training data. Your domain has different relevance relationships. Fine-tuning with domain-specific positive and negative pairs re-calibrates the embedding space to match your domain's definition of relevance.

**The mechanics**:

1. Construct training data: (query, relevant document, hard negative) triples.
   - Positives: known query-document relevant pairs (from human annotations, click logs, or LLM-generated synthetic queries for your documents).
   - Hard negatives: documents that are superficially similar but not the correct answer (retrieve with BM25 or the current embedding model, take top results that are not the true relevant document). Hard negatives force the model to learn fine-grained distinctions.
2. Apply contrastive loss (InfoNCE or MultipleNegativesRankingLoss):
   ```
   L = -log(exp(sim(q, d+)) / (exp(sim(q, d+)) + Σ exp(sim(q, d-))))
   ```
   This pulls (query, relevant doc) pairs together and pushes (query, hard negative) pairs apart.
3. After training: re-embed all corpus documents with the new model. Rebuild the vector index.
4. Evaluate recall@K before and after on a held-out evaluation set.

**What breaks**: fine-tuning without hard negatives trains the model only on easy positives — recall improves slightly but precision does not (the model still cannot distinguish between plausible-but-wrong documents). Re-embedding the full corpus after fine-tuning is mandatory — skipping this produces an index where query vectors live in the new space and document vectors live in the old space, making retrieval worse than before.

**What the interviewer is testing**: whether you understand the role of hard negatives and know that re-indexing after fine-tuning is not optional.

**Common traps**: fine-tuning without hard negatives (the most common mistake); not rebuilding the index after fine-tuning (query and document spaces no longer align); treating embedding fine-tuning as the first optimization step (it should come after chunking, hybrid search, and re-ranking improvements — those have lower cost).

---

## Embedding Drift and Index Migration

**The problem**: you update your embedding model (better performance on the new MTEB leaderboard, or domain fine-tuning improved recall). You update the query-time embedding. Retrieval quality crashes overnight. Why?

**The core insight**: the embedding model defines a coordinate system. Every document in the index is represented in the coordinates of the old model. The query is now expressed in the coordinates of the new model. Different coordinate systems — the nearest neighbors in the new space are not the same as the nearest neighbors in the old space.

**The mechanics — migration protocol**:

1. Do not update the query embedding model without first rebuilding the index. These two changes must be atomic.
2. Build the new index offline: re-embed all corpus documents using the new model; build a new ANN index; validate recall@K on the evaluation set.
3. If the new model has different dimensionality: you must rebuild — there is no way to project old vectors to a different dimension without a learned mapping (and learned projections degrade quality).
4. Dual-run migration: route a fraction of traffic to the new index while keeping the old index live. Compare recall@K, answer faithfulness, and latency. Switch fully when the new index is validated.
5. Rollback procedure: keep the old index live until the new index has been validated in production.

**Monitoring for drift**: track recall@K proxies (e.g., fraction of queries where the model's answer cites a retrieved chunk) and answer faithfulness on sampled queries. A sudden drop in these metrics often indicates an embedding version mismatch before it reaches incident severity.

**What breaks**: in-place partial updates (re-embedding a subset of documents with the new model while the rest remain in old model coordinates) produce a mixed-space index that degrades retrieval for all queries. Always re-embed all documents atomically.

**What the interviewer is testing**: whether you understand that the index and the query encoder must always be in the same embedding space — and that migration requires rebuilding, not patching.

**Common traps**: updating the query encoder without rebuilding the index (most common production incident); attempting to project old-dimensional vectors into a new-dimensional space without a learned projection; not having a rollback plan before migrating.

---

## Multi-Tenant Vector Search and Access Control

**The problem**: you have a single vector index serving multiple enterprise customers. Customer A's documents must not be retrievable by Customer B's queries, even if they are semantically similar. This is a hard security requirement that cannot be satisfied by prompting the LLM to "not reveal" information from other tenants.

**The core insight**: access control must be enforced at the retrieval layer, not the generation layer. The LLM is not a security boundary. If unauthorized documents enter the context window, the model may use them regardless of instructions. Prevent unauthorized documents from being retrieved in the first place.

**The mechanics**:

1. **Metadata filtering**: store `tenant_id` and `access_level` with every document vector at ingestion time. At query time, pass the user's tenant and permission level as metadata filters — the vector DB enforces these before returning results.
   ```python
   chunks = vector_db.search(qv, top_k=8, filter={"tenant_id": tenant_id, "access_level": {"$in": user_permitted_levels}})
   ```
2. **Sharding by tenant**: for large enterprise deployments, give each tenant their own index shard. Queries are routed to the correct shard. No cross-tenant vectors in the same query scope.
3. **Permission invalidation**: when a user's permissions change (role change, document reclassification), update the metadata in the vector DB and invalidate any cached results for that user. Do not cache retrieval results across permission changes.

**What breaks**: pre-filter semantics vary across vector DB implementations. Some DBs first filter by metadata (may reduce recall if the filtered subset is small), then search by similarity within that subset. Others search by similarity first, then filter (may produce fewer than top-K results if many are filtered out). Know your DB's behavior and tune top-K accordingly.

**What the interviewer is testing**: whether you know that access control must be enforced at the retrieval layer and can describe how metadata filtering works.

**Common traps**: relying on the LLM to keep tenant data separate (the model is not a security boundary); not knowing that permission changes require cache invalidation; using a single index for all tenants without metadata filtering (every user gets access to all documents).

---

## Large-Scale Vector Search (Billions of Vectors)

**The problem**: HNSW works well at 10–100M vectors. At 1B+ vectors, a single node cannot hold the entire index in RAM. Query latency increases and index build time becomes days.

**The core insight**: at billion scale, the ANN index must be distributed. Partition the vector space across multiple nodes (shards), query each shard in parallel, and merge results. Add quantization to reduce per-node memory footprint.

**The mechanics — system design levers**:

**Sharding**: partition vectors across shards by tenant, domain, or random hash. Each query fans out to all shards, retrieves the shard-local top-K, and a merge step produces the global top-K. Sharding by tenant/domain also reduces search scope (tenant A's query only fans out to tenant A's shard).

**Quantization**: INT8 reduces memory 4×; binary reduces 32×; PQ can reduce 16–32× depending on settings. This allows more vectors per node, reducing the number of nodes needed.

**Replication**: each shard has replicas for read throughput and availability. Queries are routed to any replica.

**Index parameter tuning**: increase efSearch to improve recall, but monitor latency per percentile. ANN index parameters must be tuned post-quantization — quantization changes the neighborhood structure.

**Caching**: cache embeddings of frequent queries (not just results, because the same query text produces the same embedding). For high-volume production systems, 20–40% of queries may hit the embedding cache.

**What breaks**: sharding requires a scatter-gather fan-out: more shards = more parallel requests, more network latency, more aggregation cost. Shard imbalance (hot shards vs. cold shards) creates latency outliers. ANN recall per shard must be high enough that global merging recovers the true global top-K.

**What the interviewer is testing**: whether you can describe a distributed vector search architecture from first principles — not just name tools.

**Common traps**: describing only quantization without addressing sharding (quantization alone doesn't solve the single-node capacity problem); not knowing that per-shard top-K must be tuned relative to global top-K; treating replication as primarily a fault-tolerance mechanism (it is also the read throughput mechanism).

---

## Short Query Failures and Query Expansion

**The problem**: a user submits the query "pricing." The embedding of "pricing" is a general vector that matches many documents about prices, costs, fees, and rates. It does not strongly match the specific document about your product's pricing tier structure that the user actually wants. Short queries produce low-variance embeddings that retrieve broadly rather than precisely.

**The core insight**: the embedding of the query should represent what a relevant document looks like, not what the query text looks like. A short query underspecifies the embedding; expanding the query to a fuller description better approximates the embedding of the relevant document.

**The mechanics — query expansion techniques**:

**HyDE (Hypothetical Document Embedding)**: ask the LLM to generate a hypothetical answer to the query. Embed the hypothetical answer (not the query). The hypothetical answer tends to share vocabulary and structure with relevant documents, so its embedding is more similar to relevant documents than the embedding of the short query.

```python
hypothesis = llm.generate(f"Write a short passage that would answer: {query}")
qv = embed_model.encode([hypothesis])[0]
```

**Step-back prompting**: ask the LLM "What is the underlying concept or entity that this query is about?" and use the expanded concept as the query.

**Hybrid retrieval as a hedge**: BM25 handles the exact terms in short queries well; dense retrieval handles semantic paraphrases. For very short queries (1–3 words), BM25 often outperforms dense search.

**What breaks**: HyDE can hallucinate content in the hypothetical document that drifts the embedding away from the true relevant document. This is particularly dangerous in factual domains — the hypothetical answer might describe the wrong product feature, retrieving documents about a different thing. Always test HyDE improvement on your evaluation set; it does not universally help.

**What the interviewer is testing**: whether you can diagnose *why* short queries fail (underspecified embedding) and propose a principled fix (expand toward what a relevant document looks like).

**Common traps**: applying HyDE universally without testing (it degrades quality on some query distributions); not knowing that BM25 often outperforms dense search for very short exact-term queries; over-expanding queries in ways that introduce intent drift.

## Rapid Recall

### Cosine similarity: (q·d) / (||q|| ||d||)
- Direct Answer: measures directional alignment; scale-invariant
- Why: This matters because it tells you how to reason about cosine similarity: (q·d) / (||q|| ||d||).
- Pitfall: Don't answer "Cosine similarity: (q·d) / (||q|| ||d||)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: measures directional alignment; scale-invariant

### Dot product: q·d
- Direct Answer: equivalent to cosine if vectors are L2-normalized; faster if you skip normalization
- Why: This matters because it tells you how to reason about dot product: q·d.
- Pitfall: Don't answer "Dot product: q·d" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: equivalent to cosine if vectors are L2-normalized; faster if you skip normalization

### L2 distance: ||q - d||²
- Direct Answer: finds absolute proximity; valid but less common for retrieval
- Why: This matters because it tells you how to reason about l2 distance: ||q - d||².
- Pitfall: Don't answer "L2 distance: ||q - d||²" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: finds absolute proximity; valid but less common for retrieval

### Mean pooling
- Direct Answer: average over all non-padding token positions. Most common.
- Why: This matters because it tells you how to reason about mean pooling.
- Pitfall: Don't answer "Mean pooling" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: average over all non-padding token positions. Most common.

### CLS token pooling
- Direct Answer: use the hidden state of the special [CLS] token (BERT-style). Designed for classification but used for embeddings in some models.
- Why: This matters because it tells you how to reason about cls token pooling.
- Pitfall: Don't answer "CLS token pooling" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: use the hidden state of the special [CLS] token (BERT-style). Designed for classification but used for embeddings in some models.

### Attention pooling
- Direct Answer: learn a weighted combination of token states. More expressive; rarely used in standard embedding models.
- Why: This matters because it tells you how to reason about attention pooling.
- Pitfall: Don't answer "Attention pooling" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: learn a weighted combination of token states. More expressive; rarely used in standard embedding models.

### Embedding vectors (the float arrays)
- Direct Answer: Embedding vectors (the float arrays)
- Why: This matters because it tells you how to reason about embedding vectors (the float arrays).
- Pitfall: Don't answer "Embedding vectors (the float arrays)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Embedding vectors (the float arrays)

### Raw text or document IDs (references to the original content)
- Direct Answer: Raw text or document IDs (references to the original content)
- Why: This matters because it tells you how to reason about raw text or document ids (references to the original content).
- Pitfall: Don't answer "Raw text or document IDs (references to the original content)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Raw text or document IDs (references to the original content)

### Metadata (source, date, tenant_id, access level, etc.)
- Direct Answer: Metadata (source, date, tenant_id, access level, etc.)
- Why: This matters because it tells you how to reason about metadata (source, date, tenant_id, access level, etc.).
- Pitfall: Don't answer "Metadata (source, date, tenant_id, access level, etc.)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Metadata (source, date, tenant_id, access level, etc.)

### ANN index for fast similarity search (HNSW or IVF+PQ)
- Direct Answer: ANN index for fast similarity search (HNSW or IVF+PQ)
- Why: This matters because it tells you how to reason about ann index for fast similarity search (hnsw or ivf+pq).
- Pitfall: Don't answer "ANN index for fast similarity search (HNSW or IVF+PQ)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: ANN index for fast similarity search (HNSW or IVF+PQ)

### Metadata filtering (combine similarity search with structured predicates)
- Direct Answer: Metadata filtering (combine similarity search with structured predicates)
- Why: This matters because it tells you how to reason about metadata filtering (combine similarity search with structured predicates).
- Pitfall: Don't answer "Metadata filtering (combine similarity search with structured predicates)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Metadata filtering (combine similarity search with structured predicates)

### CRUD operations (upsert, delete, update)
- Direct Answer: CRUD operations (upsert, delete, update)
- Why: This matters because it tells you how to reason about crud operations (upsert, delete, update).
- Pitfall: Don't answer "CRUD operations (upsert, delete, update)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: CRUD operations (upsert, delete, update)

### Layer 0 (base layer)
- Direct Answer: all vectors, each connected to M nearest neighbors (dense connections)
- Why: This matters because it tells you how to reason about layer 0 (base layer).
- Pitfall: Don't answer "Layer 0 (base layer)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: all vectors, each connected to M nearest neighbors (dense connections)

### Layer 1
- Direct Answer: a random subset of vectors (~1/e of layer 0), connected to M neighbors in this sparser subgraph
- Why: This matters because it tells you how to reason about layer 1.
- Pitfall: Don't answer "Layer 1" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: a random subset of vectors (~1/e of layer 0), connected to M neighbors in this sparser subgraph

### Layer 2
- Direct Answer: sparser still (~1/e of layer 1)
- Why: This matters because it tells you how to reason about layer 2.
- Pitfall: Don't answer "Layer 2" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: sparser still (~1/e of layer 1)

### ... up to log(N) layers
- Direct Answer: ... up to log(N) layers
- Why: This matters because it tells you how to reason about ... up to log(n) layers.
- Pitfall: Don't answer "... up to log(N) layers" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: ... up to log(N) layers

### M
- Direct Answer: connections per node. More connections = better recall, more memory. Typical: 16–64.
- Why: This matters because it tells you how to reason about m.
- Pitfall: Don't answer "M" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: connections per node. More connections = better recall, more memory. Typical: 16–64.

### efConstruction
- Direct Answer: beam size during index build. Higher = better quality index, slower build. Typical: 100–400.
- Why: This matters because it tells you how to reason about efconstruction.
- Pitfall: Don't answer "efConstruction" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: beam size during index build. Higher = better quality index, slower build. Typical: 100–400.

### efSearch
- Direct Answer: beam size during query. Higher = better recall, slower query. Tune to meet recall SLA.
- Why: This matters because it tells you how to reason about efsearch.
- Pitfall: Don't answer "efSearch" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: beam size during query. Higher = better recall, slower query. Tune to meet recall SLA.

### Recall@K
- Direct Answer: fraction of queries where the relevant document appears in the top-K results. Most important for RAG.
- Why: This matters because it tells you how to reason about recall@k.
- Pitfall: Don't answer "Recall@K" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: fraction of queries where the relevant document appears in the top-K results. Most important for RAG.

### MRR (Mean Reciprocal Rank)
- Direct Answer: 1/rank of the first relevant result, averaged over queries. Penalizes relevant documents at lower ranks.
- Why: This matters because it tells you how to reason about mrr (mean reciprocal rank).
- Pitfall: Don't answer "MRR (Mean Reciprocal Rank)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: 1/rank of the first relevant result, averaged over queries. Penalizes relevant documents at lower ranks.

### NDCG@K
- Direct Answer: normalized discounted cumulative gain; useful when multiple relevance levels exist.
- Why: This matters because it tells you how to reason about ndcg@k.
- Pitfall: Don't answer "NDCG@K" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: normalized discounted cumulative gain; useful when multiple relevance levels exist.

### Max sequence length
- Direct Answer: must equal or exceed your chunk size. If chunks are 512 tokens, models with 256-token max length will truncate your chunks.
- Why: This matters because it tells you how to reason about max sequence length.
- Pitfall: Don't answer "Max sequence length" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: must equal or exceed your chunk size. If chunks are 512 tokens, models with 256-token max length will truncate your chunks.

### Dimensionality
- Direct Answer: higher d → better recall ceiling, higher storage cost.
- Why: This matters because it tells you how to reason about dimensionality.
- Pitfall: Don't answer "Dimensionality" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: higher d → better recall ceiling, higher storage cost.

### Latency
- Direct Answer: self-hosted GPU vs. API (cloud API adds network latency, sends data off-premise).
- Why: This matters because it tells you how to reason about latency.
- Pitfall: Don't answer "Latency" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: self-hosted GPU vs. API (cloud API adds network latency, sends data off-premise).

### Domain coverage
- Direct Answer: models trained on domain-specific data (legal, biomedical, code) outperform general models in those domains.
- Why: This matters because it tells you how to reason about domain coverage.
- Pitfall: Don't answer "Domain coverage" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: models trained on domain-specific data (legal, biomedical, code) outperform general models in those domains.

### Positives
- Direct Answer: known query-document relevant pairs (from human annotations, click logs, or LLM-generated synthetic queries for your documents).
- Why: This matters because it tells you how to reason about positives.
- Pitfall: Don't answer "Positives" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: known query-document relevant pairs (from human annotations, click logs, or LLM-generated synthetic queries for your documents).

### Hard negatives
- Direct Answer: documents that are superficially similar but not the correct answer (retrieve with BM25 or the current embedding model, take top results that are not the true relevant document). Hard negatives force the model to learn fine-grained distinctions.
- Why: This matters because it tells you how to reason about hard negatives.
- Pitfall: Don't answer "Hard negatives" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: documents that are superficially similar but not the correct answer (retrieve with BM25 or the current embedding model, take top results that are not the true relevant document).…
