# Build a RAG System

End-to-end pipeline: **chunk documents → embed → store → retrieve → augment prompt → generate**.

---

## Pipeline diagram

```
Documents → Chunk → Embed → Vector store
                              ↑
User query → Embed ──────────┼→ Retrieve top-k → Build prompt → LLM → Answer
```

---

## 1. Chunking

Split text into overlapping or boundary-aligned segments (e.g. 256–512 tokens, 50–100 token overlap).

```python
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    # Simplified: split by tokens or sentences; use overlap
    tokens = text.split()  # or use a tokenizer
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks
```

---

## 2. Embedding

Use an embedding model (sentence-transformers, OpenAI, etc.) to get vectors.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunks)
# chunk_embeddings shape: (n_chunks, dim)
```

---

## 3. Vector store

Store (chunk_id, text, embedding). Example with FAISS:

```python
import faiss
import numpy as np

dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # inner product (normalize for cosine)
faiss.normalize_L2(chunk_embeddings)
index.add(chunk_embeddings)
# Store chunk texts in a list: texts[i] corresponds to index i
```

---

## 4. Retrieval

Embed query, search top-k, get chunk texts.

```python
def retrieve(query: str, index, texts: list, model, k: int = 5):
    q_emb = model.encode([query])
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, k)
    return [texts[i] for i in ids[0]]
```

---

## 5. Augment and generate

Build prompt with retrieved context; call LLM.

```python
def rag_prompt(query: str, retrieved: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(retrieved))
    return f"""Use the following context to answer. Cite [1], [2], etc. when relevant.

Context:
{context}

Question: {query}

Answer:"""

# Then: response = llm.create(messages=[{"role": "user", "content": rag_prompt(query, retrieved)}])
```

---

## Full flow (pseudocode)

```python
# Offline
docs = load_documents()
chunks = [chunk(d) for d in docs]
chunks_flat = [c for sub in chunks for c in sub]
embeddings = embed_model.encode(chunks_flat)
index.add(embeddings)
store_texts(chunks_flat)

# Online
def answer(query: str) -> str:
    retrieved = retrieve(query, index, stored_texts, embed_model, k=5)
    prompt = rag_prompt(query, retrieved)
    return llm.generate(prompt)
```

---

## Extensions

- **Reranker**: Retrieve top-20, rerank with cross-encoder, take top-5.
- **Hybrid**: Run vector + BM25; merge with RRF.
- **Evaluation**: Measure recall@k and answer faithfulness on a labeled set.

See [RAG](../llm-applications/rag.md) and [Vector databases](../llm-applications/vector-databases.md) for more detail.
