---
module: Llms
topic: Interview Notes
subtopic: Vector Databases And Embeddings Snappy
status: unread
tags: [llms, ml, interview-notes-vector-databas]
---
# Vector databases & embeddings — semantic search infra (Azure/DevOps-fluent)

Embeddings turn messy language into **geometry**. Vector databases turn that geometry into **fast retrieval**.

**One-line:** embeddings = feature vectors; vector DB = ANN index + metadata + ops.

---

# Q1: What are embeddings in AI engineering?
- **Direct answer:** Dense vectors that represent meaning so similarity becomes distance.
- **Fashion analogy:** turning an outfit into a feature vector: fabric, cut, silhouette, vibe.

---

# Q2: How do embedding models convert text to vectors?
- **Direct answer:** tokenize → encoder → pooled hidden state → vector; trained so semantically similar texts map nearby.

---

# Q3: Sparse vs dense embeddings?
- **Sparse:** keyword-like (BM25/TF-IDF-ish signals).
- **Dense:** semantic meaning in continuous space.
- **Hybrid is common** in production.

---

# Q4: Cosine vs dot vs Euclidean distance?
- **Cosine:** angle similarity (scale-invariant).
- **Dot product:** similar to cosine if vectors are normalized; faster sometimes.
- **Euclidean:** distance in space; sensitive to scale.

---

# Q5: What is a vector database?
- **Direct answer:** Stores vectors + runs approximate nearest neighbor search (ANN) efficiently.
- **DevOps bridge:** classic DBs answer exact queries; vector DBs answer “closest neighbors.”

---

# Q6: Choosing the right embedding model?
- **Criteria:** domain fit, multilingual, cost/latency, dimension size, licensing.
- **Mini prompt:** What breaks instantly? → indexing with one model, querying with another.

---

# Q7: Embedding dimensionality — effect on cost and quality?
- Higher dim can improve nuance but increases storage, memory bandwidth, index size.
- Lower dim is cheaper and often good enough.

---

# Q8: Embedding drift when updating embedder?
- **Direct answer:** new model changes the coordinate space.
- **Safe rollout:** dual-write embeddings, backfill, A/B evaluate retrieval, then cut over.

---

# Q9: Multi-modal embeddings?
- **Direct answer:** map text+image (etc.) into a shared space (e.g., CLIP).

---

# Q10: Multi-tenant indexing/querying?
- **Patterns:** namespace per tenant, metadata filters, per-tenant keys, shard hot tenants.
- **DevOps bridge:** it’s isolation + RBAC + noisy-neighbor control.

---

# Q11: Quantization of embeddings?
- **Direct answer:** compress vectors (float16/int8/product quantization) to cut storage.
- **Trade-off:** some recall loss.

---

# Q12: Benchmark embedding quality?
- **Metrics:** recall@k, MRR, nDCG; human relevance judgments.
- **Production:** evaluate end-to-end (retrieval + answer quality for RAG).

---

# Q13: Role of metadata in vector DBs?
- **Direct answer:** filtering, access control, citations (source/page), time-based routing.

---

# Q14: Scale to billions of vectors?
- **Patterns:** sharding, HNSW/IVF/PQ, caching hot queries, batch ingestion, tiered storage.
- **Azure hint:** treat it like search infra (index build + query SLA).

---

# Q15: Hybrid search?
- **Direct answer:** combine keyword (BM25) + vector similarity.
- **Why:** exact IDs/numbers + semantic meaning.

---

# Q16: Fine-tune an embedding model?
- **Direct answer:** contrastive training on domain pairs (query, relevant doc) and hard negatives.

---

# Q17: Vector DB uses too much memory — reduce it.
- **Fixes:** lower dimension, quantize, prune old docs, smaller top-k, better chunking.

---

# Q18: Can’t scale to millions — bottleneck fixes?
- **Fixes:** ANN index choice, shard/replicate, optimize ingestion, move to distributed index.

---

# Q19: New embedding model has different dimension — mismatch?
- **Fix:** you can’t compare vectors across dims/spaces. Re-embed + rebuild index, or run dual indexes during migration.

---

# Q20: Irrelevant results despite high similarity?
- **Causes:** bad chunking, wrong embedder, no metadata filters, semantic gap, no reranking.
- **Fix:** reranker + hybrid search + better chunk boundaries.

---

# Q21: Search quality crashed after new embedder — handle drift?
- **Process:** rollback, compare metrics, inspect queries with failures, recalibrate thresholds, ensure backfill complete.

---

# Q22: Short queries fail — improve?
- **Fixes:** query expansion (LLM rewrite), hybrid search, user intent classification, add synonyms/metadata boosting.

## Rapid Recall

### Direct answer
- Direct Answer: Dense vectors that represent meaning so similarity becomes distance.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Dense vectors that represent meaning so similarity becomes distance.

### Fashion analogy
- Direct Answer: turning an outfit into a feature vector: fabric, cut, silhouette, vibe.
- Why: This matters because it tells you how to reason about fashion analogy.
- Pitfall: Don't answer "Fashion analogy" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: turning an outfit into a feature vector: fabric, cut, silhouette, vibe.

### Direct answer
- Direct Answer: tokenize → encoder → pooled hidden state → vector; trained so semantically similar texts map nearby.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: tokenize → encoder → pooled hidden state → vector; trained so semantically similar texts map nearby.

### Sparse
- Direct Answer: keyword-like (BM25/TF-IDF-ish signals).
- Why: This matters because it tells you how to reason about sparse.
- Pitfall: Don't answer "Sparse" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: keyword-like (BM25/TF-IDF-ish signals).

### Dense
- Direct Answer: semantic meaning in continuous space.
- Why: This matters because it tells you how to reason about dense.
- Pitfall: Don't answer "Dense" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: semantic meaning in continuous space.

### Hybrid is common in production.
- Direct Answer: Hybrid is common in production.
- Why: This matters because it tells you how to reason about hybrid is common in production..
- Pitfall: Don't answer "Hybrid is common in production." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Hybrid is common in production.

### Cosine
- Direct Answer: angle similarity (scale-invariant).
- Why: This matters because it tells you how to reason about cosine.
- Pitfall: Don't answer "Cosine" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: angle similarity (scale-invariant).

### Dot product
- Direct Answer: similar to cosine if vectors are normalized; faster sometimes.
- Why: This matters because it tells you how to reason about dot product.
- Pitfall: Don't answer "Dot product" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: similar to cosine if vectors are normalized; faster sometimes.

### Euclidean
- Direct Answer: distance in space; sensitive to scale.
- Why: This matters because it tells you how to reason about euclidean.
- Pitfall: Don't answer "Euclidean" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: distance in space; sensitive to scale.

### Direct answer
- Direct Answer: Stores vectors + runs approximate nearest neighbor search (ANN) efficiently.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Stores vectors + runs approximate nearest neighbor search (ANN) efficiently.

### DevOps bridge
- Direct Answer: classic DBs answer exact queries; vector DBs answer “closest neighbors.”
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: classic DBs answer exact queries; vector DBs answer “closest neighbors.”

### Criteria
- Direct Answer: domain fit, multilingual, cost/latency, dimension size, licensing.
- Why: This matters because it tells you how to reason about criteria.
- Pitfall: Don't answer "Criteria" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: domain fit, multilingual, cost/latency, dimension size, licensing.

### Mini prompt
- Direct Answer: What breaks instantly? → indexing with one model, querying with another.
- Why: This matters because it tells you how to reason about mini prompt.
- Pitfall: Don't answer "Mini prompt" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: What breaks instantly? → indexing with one model, querying with another.

### Higher dim can improve nuance but increases storage, memory bandwidth, index size.
- Direct Answer: Higher dim can improve nuance but increases storage, memory bandwidth, index size.
- Why: This matters because it tells you how to reason about higher dim can improve nuance but increases storage, memory bandwidth, index size..
- Pitfall: Don't answer "Higher dim can improve nuance but increases storage, memory bandwidth, index size." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Higher dim can improve nuance but increases storage, memory bandwidth, index size.

### Lower dim is cheaper and often good enough.
- Direct Answer: Lower dim is cheaper and often good enough.
- Why: This matters because it tells you how to reason about lower dim is cheaper and often good enough..
- Pitfall: Don't answer "Lower dim is cheaper and often good enough." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Lower dim is cheaper and often good enough.

### Direct answer
- Direct Answer: new model changes the coordinate space.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: new model changes the coordinate space.

### Safe rollout
- Direct Answer: dual-write embeddings, backfill, A/B evaluate retrieval, then cut over.
- Why: This matters because it tells you how to reason about safe rollout.
- Pitfall: Don't answer "Safe rollout" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: dual-write embeddings, backfill, A/B evaluate retrieval, then cut over.

### Direct answer
- Direct Answer: map text+image (etc.) into a shared space (e.g., CLIP).
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: map text+image (etc.) into a shared space (e.g., CLIP).

### Patterns
- Direct Answer: namespace per tenant, metadata filters, per-tenant keys, shard hot tenants.
- Why: This matters because it tells you how to reason about patterns.
- Pitfall: Don't answer "Patterns" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: namespace per tenant, metadata filters, per-tenant keys, shard hot tenants.

### DevOps bridge
- Direct Answer: it’s isolation + RBAC + noisy-neighbor control.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: it’s isolation + RBAC + noisy-neighbor control.

### Direct answer
- Direct Answer: compress vectors (float16/int8/product quantization) to cut storage.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: compress vectors (float16/int8/product quantization) to cut storage.

### Trade-off
- Direct Answer: some recall loss.
- Why: This matters because it tells you how to reason about trade-off.
- Pitfall: Don't answer "Trade-off" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: some recall loss.

### Metrics
- Direct Answer: recall@k, MRR, nDCG; human relevance judgments.
- Why: This matters because it tells you how to reason about metrics.
- Pitfall: Don't answer "Metrics" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: recall@k, MRR, nDCG; human relevance judgments.

### Production
- Direct Answer: evaluate end-to-end (retrieval + answer quality for RAG).
- Why: This matters because it tells you how to reason about production.
- Pitfall: Don't answer "Production" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: evaluate end-to-end (retrieval + answer quality for RAG).

### Direct answer
- Direct Answer: filtering, access control, citations (source/page), time-based routing.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: filtering, access control, citations (source/page), time-based routing.

### Patterns
- Direct Answer: sharding, HNSW/IVF/PQ, caching hot queries, batch ingestion, tiered storage.
- Why: This matters because it tells you how to reason about patterns.
- Pitfall: Don't answer "Patterns" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: sharding, HNSW/IVF/PQ, caching hot queries, batch ingestion, tiered storage.

### Azure hint
- Direct Answer: treat it like search infra (index build + query SLA).
- Why: This matters because it tells you how to reason about azure hint.
- Pitfall: Don't answer "Azure hint" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: treat it like search infra (index build + query SLA).

### Direct answer
- Direct Answer: combine keyword (BM25) + vector similarity.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: combine keyword (BM25) + vector similarity.

### Why
- Direct Answer: exact IDs/numbers + semantic meaning.
- Why: This matters because it tells you how to reason about why.
- Pitfall: Don't answer "Why" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: exact IDs/numbers + semantic meaning.

### Direct answer
- Direct Answer: contrastive training on domain pairs (query, relevant doc) and hard negatives.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: contrastive training on domain pairs (query, relevant doc) and hard negatives.

### Fixes
- Direct Answer: lower dimension, quantize, prune old docs, smaller top-k, better chunking.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: lower dimension, quantize, prune old docs, smaller top-k, better chunking.

### Fixes
- Direct Answer: ANN index choice, shard/replicate, optimize ingestion, move to distributed index.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: ANN index choice, shard/replicate, optimize ingestion, move to distributed index.

### Fix
- Direct Answer: you can’t compare vectors across dims/spaces. Re-embed + rebuild index, or run dual indexes during migration.
- Why: This matters because it tells you how to reason about fix.
- Pitfall: Don't answer "Fix" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: you can’t compare vectors across dims/spaces. Re-embed + rebuild index, or run dual indexes during migration.

### Causes
- Direct Answer: bad chunking, wrong embedder, no metadata filters, semantic gap, no reranking.
- Why: This matters because it tells you how to reason about causes.
- Pitfall: Don't answer "Causes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: bad chunking, wrong embedder, no metadata filters, semantic gap, no reranking.

### Fix
- Direct Answer: reranker + hybrid search + better chunk boundaries.
- Why: This matters because it tells you how to reason about fix.
- Pitfall: Don't answer "Fix" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: reranker + hybrid search + better chunk boundaries.

### Process
- Direct Answer: rollback, compare metrics, inspect queries with failures, recalibrate thresholds, ensure backfill complete.
- Why: This matters because it tells you how to reason about process.
- Pitfall: Don't answer "Process" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: rollback, compare metrics, inspect queries with failures, recalibrate thresholds, ensure backfill complete.

### Fixes
- Direct Answer: query expansion (LLM rewrite), hybrid search, user intent classification, add synonyms/metadata boosting.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: query expansion (LLM rewrite), hybrid search, user intent classification, add synonyms/metadata boosting.
