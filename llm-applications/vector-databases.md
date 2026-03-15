# Vector Databases and Embedding Search

Vector databases store **embeddings** (dense vectors) and support **similarity search**: find items whose vectors are closest to a query vector. They are the backbone of RAG, semantic search, and recommendation systems.

---

## Embeddings

- **Embedding**: A dense vector representation of a token, sentence, or document produced by an embedding model (e.g. from transformers, sentence-transformers, or API like OpenAI `text-embedding-3`).
- **Semantic similarity**: Similar meaning → vectors close in distance (e.g. cosine similarity or L2). Enables “find text similar to this” without exact keyword match.

---

## Similarity search

- **Query**: User text (or item) is embedded to a vector **q**.
- **Search**: Find stored vectors **v** such that distance **d(q, v)** is smallest (or similarity largest).
- **Exact** search: compare **q** to every **v** — O(n) per query, fine for small n.
- **Approximate** search: use indexes (HNSW, IVF, etc.) to avoid scanning all vectors — sublinear time, slight loss in recall.

---

## Nearest neighbor and ANN

- **k-NN (k nearest neighbors)**: Return the k vectors closest to **q** (exact: linear scan).
- **ANN (approximate nearest neighbor)**: Return k vectors that are *likely* among the true nearest; trade recall for speed.
- **Metrics**: L2 (Euclidean), inner product, cosine (often on normalized vectors so cosine ≈ inner product).

---

## Indexing strategies

| Method | Idea | Pros / cons |
|--------|------|-------------|
| **Flat / brute force** | Store all vectors; scan all | Exact, simple; slow for large n |
| **HNSW (Hierarchical Navigable Small World)** | Graph with layers; greedy search | Good recall/speed; used by many vector DBs |
| **IVF (Inverted File)** | Cluster vectors; search only in nearest cluster(s) | Fast; needs tuning (nlist, nprobe) |
| **PQ (Product Quantization)** | Compress vectors; approximate distance | Saves memory; some accuracy loss |
| **FAISS** | Library (Facebook); IVF + PQ, GPU | Very fast at scale; in-memory or memory-mapped |

---

## HNSW in short

- **Graph**: Each vector is a node; edges connect to “close” neighbors. Multiple layers: bottom = full graph, top = long-range shortcuts.
- **Search**: Start at top layer, greedy walk to nearest neighbor; drop to lower layer; repeat until bottom. Returns k nearest.
- **Parameters**: `ef_construction` (build-time search size), `ef` (query-time search size); higher → better recall, slower.

---

## FAISS

- **FAISS** (Facebook AI Similarity Search): Library for efficient similarity search and clustering.
- **Index types**: IndexFlatL2 (exact), IndexIVFFlat, IndexHNSWFlat, IndexIVFPQ, etc.
- **GPU**: Use GPU indices for large-scale batch search.
- **Usage**: Fit on a matrix of vectors (n × d); then `index.search(query_vectors, k)` returns k nearest IDs and distances.

---

## Vector store architectures

- **Managed services**: Pinecone, Weaviate, Qdrant, Milvus, pgvector (Postgres), Atlas (MongoDB). Handle replication, filtering, and sometimes hybrid (keyword + vector).
- **Self-hosted**: Chroma, FAISS on your own infra, Vespa.
- **Embedded**: Chroma, LanceDB, SQLite with vector extension — good for dev or small deployments.

---

## Real-world usage

- **RAG**: Embed documents (or chunks); at query time embed the question and retrieve top-k chunks; pass to LLM as context.
- **Semantic search**: Replace or augment keyword search with vector similarity over product descriptions, docs, or tickets.
- **Deduplication**: Embed items; merge or flag near-duplicates (small distance).
- **Recommendations**: Embed users and items; recommend items closest to user vector or to items they liked.

---

## Quick revision

- **Embeddings** = dense vectors from text (or items); **similarity search** = find nearest vectors to a query.
- **ANN** = approximate nearest neighbor for speed at scale; **HNSW** and **IVF/PQ** are common index types.
- **FAISS** = fast library for similarity search; **vector DBs** (Pinecone, Weaviate, pgvector, etc.) add persistence, filtering, and APIs.
- **RAG** uses vector search to retrieve relevant chunks before generation.
