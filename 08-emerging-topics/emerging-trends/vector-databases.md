# Vector Databases

How semantic search works under the hood — from embedding storage to approximate nearest neighbor indexes, distance metrics, and production deployment patterns. Critical for RAG systems, recommendation engines, and any system that needs to find "similar" things at scale.

---

## 1. What Problem Vector Databases Solve

Traditional databases answer exact queries: "give me all rows where user_id = 42." They cannot answer semantic queries: "give me the 10 documents most similar in meaning to this question." Relational databases have no concept of "similar" — similarity requires a distance metric over high-dimensional vectors, and B-tree/hash indexes are useless for that.

A vector database stores high-dimensional embedding vectors and answers queries of the form: **"given query vector q, return the k vectors in the database that are nearest to q."** This is called approximate nearest neighbor (ANN) search.

The "approximate" is intentional and important — exact nearest neighbor search in high dimensions is O(n·d) per query (brute force), which is too slow at billions of vectors. ANN indexes trade a small amount of recall for orders-of-magnitude speedup.

---

## 2. How Embeddings Work

An embedding is a dense vector representation of an object (text, image, audio, code) in a space where geometric proximity ≈ semantic similarity. Two sentences with the same meaning should be close in embedding space even if they share no words.

```
"The cat sat on the mat"     →  [0.12, -0.34, 0.89, ..., 0.23]  (1536 dims)
"A feline rested on a rug"   →  [0.14, -0.31, 0.91, ..., 0.21]  ← very close
"Quantum field theory notes" →  [-0.88, 0.71, -0.12, ..., 0.55]  ← far away
```

Common embedding models:
| Model | Dims | Best for |
|---|---|---|
| text-embedding-3-large (OpenAI) | 3072 | General text, English |
| e5-large-v2 | 1024 | Retrieval-tuned, asymmetric |
| BGE-M3 | 1024 | Multilingual, hybrid dense+sparse |
| CLIP ViT-L/14 | 768 | Text-image joint embedding |
| Nomic Embed | 768 | Long documents, open-source |

---

## 3. Distance Metrics

The choice of distance metric must match how the embedding model was trained.

### Cosine Similarity
$$\text{cos}(u, v) = \frac{u \cdot v}{||u|| \cdot ||v||}$$

Measures the angle between vectors, ignoring magnitude. Range: [-1, 1]. Most text embedding models are trained with cosine similarity. If vectors are L2-normalized (||v|| = 1), cosine similarity = dot product, making it the fastest to compute.

### Dot Product (Inner Product)
$$\text{dot}(u, v) = u \cdot v = \sum_i u_i v_i$$

When vectors are normalized: identical to cosine. When not normalized: biased toward high-magnitude vectors. Used in two-tower retrieval models and recommendation systems where magnitude encodes relevance score.

### Euclidean Distance (L2)
$$d(u, v) = \sqrt{\sum_i (u_i - v_i)^2}$$

Measures straight-line distance. Less common for text (magnitude matters), more common for image embeddings and clustering. FAISS IVFFlat typically uses L2.

**Rule**: always check which metric your embedding model was trained with and use the same at query time. Using L2 on cosine-trained embeddings gives wrong results.

---

## 4. ANN Indexes — How They Actually Work

### 4.1 Flat (Brute Force)
Search every vector. 100% recall but O(n·d) per query.
- Use when: n < 100K, or recall is non-negotiable (re-ranking stage)
- FAISS: `IndexFlatIP`, `IndexFlatL2`

### 4.2 IVF (Inverted File Index)
**Idea**: cluster vectors into K cells (K-means). At query time, search only the nearest C cells instead of all cells.

```
Build time:
  1. Run K-means on all vectors → K centroids
  2. Assign each vector to its nearest centroid
  3. Store inverted list: centroid_id → [vector_ids]

Query time:
  1. Find top C nearest centroids to query
  2. Search only vectors in those C cells
  3. Return top-k from those candidates
```

- `nlist` (K) = number of clusters, typically √n to n/10
- `nprobe` (C) = cells to search at query time, trade-off recall vs speed
- nprobe=1: fast, lower recall; nprobe=32: slower, higher recall
- At 100M vectors with nlist=4096, nprobe=8: ~100× speedup vs brute force

### 4.3 HNSW (Hierarchical Navigable Small World)
Graph-based index. The best general-purpose ANN algorithm for recall/speed trade-off.

**Construction**: build a multi-layer graph where each node (vector) connects to its M nearest neighbors. Higher layers have fewer nodes (exponentially sparser), like a skip list.

```
Layer 2 (sparse):    ●----------●---------●
Layer 1 (medium):  ●--●-----●--●-------●--●
Layer 0 (dense):  ●-●-●--●-●-●-●---●-●-●-●-●
```

**Query**: enter at the top (sparse) layer, greedily navigate toward the query vector, then descend to the dense layer for fine-grained search. Complexity: O(log n) per query.

Key parameters:
- `M` = number of neighbors per node (16-64). Higher = better recall, more memory, slower build
- `ef_construction` = beam width during index build (100-500). Higher = better recall at cost of build time
- `ef_search` = beam width during query. Can be set at query time; higher = better recall, slower

**When to use**: best recall/latency for up to ~100M vectors on a single machine. Memory-heavy: O(M·n·4 bytes) just for the graph.

### 4.4 PQ (Product Quantization)
**Compression technique** that reduces memory 8-32× by quantizing vectors.

Split d-dimensional vector into m sub-vectors of d/m dimensions each. Quantize each sub-vector to one of 256 centroids (1 byte). A 1024-dim float32 vector (4KB) → 128 bytes (32×compression).

```
Original vector (1024 dims, 4 bytes each) = 4096 bytes
PQ-128:  split into 128 sub-vectors of 8 dims → 128 bytes (32× compression)
```

Distance approximation: ADC (Asymmetric Distance Computation) precomputes distances from the query to all sub-codebooks and sums them. Very fast but approximate.

**IVFPQ**: IVF for coarse search + PQ for compressed storage within each cell — used for billion-scale indexes where memory is the bottleneck.

---

## 5. Major Vector Database Systems

| System | Index types | Filtering | Hosted | Best for |
|---|---|---|---|---|
| **FAISS** | Flat, IVF, HNSW, PQ | No native filter | No (library) | Research, offline batch |
| **Pinecone** | HNSW + proprietary | Metadata filters | Yes (managed) | Startups, fast setup |
| **Weaviate** | HNSW | GraphQL filters | Self/managed | Hybrid search + graph |
| **Qdrant** | HNSW + scalar quant | JSON payload filters | Self/managed | Production, Rust-based |
| **Chroma** | HNSW | Metadata filters | Self | Development, prototyping |
| **pgvector** | IVFFlat, HNSW | Full SQL | With Postgres | Existing Postgres infra |
| **Milvus** | IVF, HNSW, DISKANN | Scalar/vector | Self/managed | Large scale, cloud-native |

**DISKANN**: Microsoft's graph-based index designed for disk-resident vectors. Allows billion-scale indexes that don't fit in RAM by streaming from SSD. Memory: O(1) resident, disk: O(n·d).

---

## 6. Hybrid Search

Pure dense ANN fails for exact keyword matching ("API v2.3", product SKUs, names). Pure BM25 fails for semantic paraphrase. Hybrid search combines both.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, NamedSparseVector

# Hybrid query: dense + sparse (BM25-like)
results = client.query_points(
    collection_name="docs",
    prefetch=[
        # Dense: semantic
        models.Prefetch(query=dense_embedding, using="dense", limit=100),
        # Sparse: keyword
        models.Prefetch(
            query=SparseVector(indices=bm25_indices, values=bm25_weights),
            using="sparse",
            limit=100,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),  # Reciprocal Rank Fusion
    limit=10,
)
```

**Reciprocal Rank Fusion (RRF)**: score(d) = Σ 1/(k + rank(d)) where k=60. Combines ranked lists without needing to calibrate score scales.

---

## 7. Metadata Filtering

Production systems need to filter by attributes (date, user_id, category) AND do ANN search. Naive approach: run ANN first, then filter — problem: after filtering, you may not have enough results (low-recall with strict filters).

**Pre-filtering (correct approach)**: filter first, then do ANN within the filtered subset. Requires index to support filtered ANN. HNSW-based systems (Qdrant, Weaviate, Pinecone) handle this with payload-aware graph traversal.

```python
# Qdrant filtered search — finds nearest vectors WHERE created_at > X AND category = "tech"
results = client.search(
    collection_name="articles",
    query_vector=query_embedding,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(key="category", match=models.MatchValue(value="tech")),
            models.FieldCondition(key="created_at", range=models.DatetimeRange(gte="2024-01-01")),
        ]
    ),
    limit=10,
)
```

---

## 8. Capacity Planning

For a 10M document RAG system with 1536-dim embeddings:

```
Storage:
  Raw vectors: 10M × 1536 × 4 bytes = 61.4 GB (float32)
  HNSW graph (M=16): ~10M × 16 × 8 bytes = 1.3 GB extra
  Total in RAM: ~63 GB → needs 2×80GB nodes for replication

With PQ compression (32×):
  Compressed vectors: 10M × 48 bytes = 0.48 GB
  Fits in RAM easily, but recall drops ~5%

Query throughput:
  HNSW on GPU: ~10K QPS at ef_search=64
  HNSW on CPU (32 cores): ~2-5K QPS
  IVFFlat GPU (FAISS): ~100K QPS (lower recall)
```

---

## 9. Multi-Vector and Late Interaction

ColBERT model encodes each query token and document token separately, computing MaxSim across all token pairs — more accurate than single-vector but requires storing one vector per token.

```
Single-vector (bi-encoder): one 768-dim vector per document
ColBERT: 128 token vectors per document → 128× more storage, but higher recall
```

**PLAID (ColBERT v2 indexing)**: compress token vectors with centroid-based compression, reducing ColBERT storage to 2-4× dense while keeping most of the accuracy gain. Used in production at scale.

---

## Canonical Interview Q&As

**Q: What is HNSW and why is it preferred over IVF for most production use cases?**
A: HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm that builds a multi-layer graph where each vector connects to its M nearest neighbors. At query time, it enters at the top (sparse) layer and greedily navigates toward the query, descending layers for increasingly fine-grained search — analogous to highway→street navigation. Query complexity is O(log n). IVF clusters vectors with K-means and searches only the C nearest clusters. HNSW outperforms IVF in recall/latency trade-off for in-memory indexes because: (1) graph traversal adapts to local density, while K-means clusters can be uneven; (2) HNSW's `ef_search` can be tuned at query time without rebuilding the index; (3) HNSW achieves >95% recall at <1ms latency for 10M vectors. IVF wins when memory is constrained or for GPU-accelerated batch search (IVF maps naturally to GPU parallelism). For billion-scale disk-resident data: DISKANN is preferred.

**Q: How do you handle metadata filtering in a vector database without sacrificing recall?**
A: Naive post-filtering (ANN first, then filter) fails with strict filters — you might retrieve 100 vectors but only 2 pass the filter, giving far fewer results than requested. The correct approach is pre-filtering: apply the metadata filter first, then do ANN within the filtered subset. This requires the index to support filtered graph traversal (HNSW-based systems traverse only nodes whose payload matches the filter). For very selective filters (<1% of the collection), pre-filtering can be slow because there are few valid nodes to traverse — in this case, fall back to brute-force search on the filtered subset (which is small and therefore fast). Qdrant automatically selects the strategy based on the estimated filter selectivity. For complex filters, segment the collection (one HNSW index per segment/shard) and merge results.

**Q: Design the vector storage layer for a 100M-document RAG system with sub-100ms P99 query latency.**
A: (1) Embedding: use a 768-dim model (BGE-M3 or e5-large) — 768 dims hits the sweet spot of quality vs storage. 100M × 768 × 4 bytes = 307 GB for raw vectors. (2) Index: HNSW with M=32, ef_construction=200, stored in RAM. With HNSW graph overhead: ~330 GB total — requires a cluster of 4-5 nodes with 128GB RAM each, with 2 replicas per shard. (3) Quantization: apply scalar quantization (float32→int8, 4× compression) to bring RAM to ~82 GB, fitting on 2 nodes with margin. Recall drop: ~1-2%. (4) Query: ef_search=64 gives >98% recall at ~3-5ms per query — well within 100ms budget. (5) Metadata: store full document text + metadata in a Postgres/S3 store, indexed by vector ID; return IDs from ANN, fetch payload from Postgres. (6) Updates: for new documents, insert immediately (HNSW supports online insertion). For deletions: soft-delete with metadata flag, compact index periodically.
