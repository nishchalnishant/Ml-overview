# Build an Embedding Search System

**Embed** items (e.g. documents or products), **index** vectors, then **search** by embedding a query and finding k nearest vectors.

---

## Pipeline

```
Items (text) → Embedding model → Vectors → Index (e.g. FAISS)
                                                    ↑
Query (text) → Embedding model → Query vector ──────┼→ Search → Top-k items
```

---

## 1. Embed

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = ["First document...", "Second document..."]
embeddings = model.encode(texts)  # shape: (n, 384)
```

---

## 2. Index (FAISS)

```python
import faiss
import numpy as np

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)   # exact L2; or IndexFlatIP with normalized for cosine
index.add(embeddings.astype(np.float32))
# Keep a list: id_to_text[i] = texts[i]
```

---

## 3. Search

```python
def search(query: str, k: int = 5):
    q = model.encode([query]).astype(np.float32)
    distances, ids = index.search(q, k)
    return [id_to_text[i] for i in ids[0]]
```

---

## Approximate search (HNSW with FAISS)

For large corpora, use an approximate index:

```python
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_L2)  # 32 neighbors per node
index.hnsw.efConstruction = 40
index.add(embeddings.astype(np.float32))
index.hnsw.efSearch = 16
# search same as above; faster, slight recall tradeoff
```

---

## Cosine similarity (normalize + inner product)

```python
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(d)  # inner product = cosine when normalized
index.add(embeddings)
# same for query: normalize then search
faiss.normalize_L2(q)
scores, ids = index.search(q, k)
```

---

## Minimal end-to-end

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = load_documents()
embeddings = model.encode(texts)
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

def search(query: str, k: int = 5):
    q = model.encode([query])
    faiss.normalize_L2(q)
    _, ids = index.search(q, k)
    return [texts[i] for i in ids[0]]
```

See [Vector databases](../llm-applications/vector-databases.md) for HNSW, scaling, and production options.
