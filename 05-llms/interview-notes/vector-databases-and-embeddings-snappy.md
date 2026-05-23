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

## Flashcards

**Direct answer?** #flashcard
Dense vectors that represent meaning so similarity becomes distance.

**Fashion analogy?** #flashcard
turning an outfit into a feature vector: fabric, cut, silhouette, vibe.

**Direct answer?** #flashcard
tokenize → encoder → pooled hidden state → vector; trained so semantically similar texts map nearby.

**Sparse?** #flashcard
keyword-like (BM25/TF-IDF-ish signals).

**Dense?** #flashcard
semantic meaning in continuous space.

**Hybrid is common in production.?** #flashcard
Hybrid is common in production.

**Cosine?** #flashcard
angle similarity (scale-invariant).

**Dot product?** #flashcard
similar to cosine if vectors are normalized; faster sometimes.

**Euclidean?** #flashcard
distance in space; sensitive to scale.

**Direct answer?** #flashcard
Stores vectors + runs approximate nearest neighbor search (ANN) efficiently.

**DevOps bridge?** #flashcard
classic DBs answer exact queries; vector DBs answer “closest neighbors.”

**Criteria?** #flashcard
domain fit, multilingual, cost/latency, dimension size, licensing.

**Mini prompt?** #flashcard
What breaks instantly? → indexing with one model, querying with another.

**Higher dim can improve nuance but increases storage, memory bandwidth, index size.?** #flashcard
Higher dim can improve nuance but increases storage, memory bandwidth, index size.

**Lower dim is cheaper and often good enough.?** #flashcard
Lower dim is cheaper and often good enough.

**Direct answer?** #flashcard
new model changes the coordinate space.

**Safe rollout?** #flashcard
dual-write embeddings, backfill, A/B evaluate retrieval, then cut over.

**Direct answer?** #flashcard
map text+image (etc.) into a shared space (e.g., CLIP).

**Patterns?** #flashcard
namespace per tenant, metadata filters, per-tenant keys, shard hot tenants.

**DevOps bridge?** #flashcard
it’s isolation + RBAC + noisy-neighbor control.

**Direct answer?** #flashcard
compress vectors (float16/int8/product quantization) to cut storage.

**Trade-off?** #flashcard
some recall loss.

**Metrics?** #flashcard
recall@k, MRR, nDCG; human relevance judgments.

**Production?** #flashcard
evaluate end-to-end (retrieval + answer quality for RAG).

**Direct answer?** #flashcard
filtering, access control, citations (source/page), time-based routing.

**Patterns?** #flashcard
sharding, HNSW/IVF/PQ, caching hot queries, batch ingestion, tiered storage.

**Azure hint?** #flashcard
treat it like search infra (index build + query SLA).

**Direct answer?** #flashcard
combine keyword (BM25) + vector similarity.

**Why?** #flashcard
exact IDs/numbers + semantic meaning.

**Direct answer?** #flashcard
contrastive training on domain pairs (query, relevant doc) and hard negatives.

**Fixes?** #flashcard
lower dimension, quantize, prune old docs, smaller top-k, better chunking.

**Fixes?** #flashcard
ANN index choice, shard/replicate, optimize ingestion, move to distributed index.

**Fix?** #flashcard
you can’t compare vectors across dims/spaces. Re-embed + rebuild index, or run dual indexes during migration.

**Causes?** #flashcard
bad chunking, wrong embedder, no metadata filters, semantic gap, no reranking.

**Fix?** #flashcard
reranker + hybrid search + better chunk boundaries.

**Process?** #flashcard
rollback, compare metrics, inspect queries with failures, recalibrate thresholds, ensure backfill complete.

**Fixes?** #flashcard
query expansion (LLM rewrite), hybrid search, user intent classification, add synonyms/metadata boosting.
