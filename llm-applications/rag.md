# RAG: Retrieval-Augmented Generation

## Executive Summary
| Stage | Technical Goal | Components |
|-------|----------------|------------|
| **Indexing** | Knowledge Representation | Chunking, Embedding Models, Vector DBs |
| **Retrieval** | Semantic Search | Cosine Similarity, HNSW Indexing |
| **Augmentation** | Context Injection | Prompt Engineering, Re-ranking |
| **Evaluation** | Quality Control | RAGAS (Faithfulness, Relevance) |

---

## 1. The RAG Pipeline

### Precision Chunking
The quality of RAG depends on the granularity of data.
- **Fixed Size**: $512$ tokens with $50$ overlap.
- **Semantic**: Split by headers or logical paragraphs.
- **Recursive**: Tries fixed sizes but respects sentence/paragraph boundaries.

### Vector Databases & HNSW
Storing embeddings in a way that allows for sub-millisecond search among millions of documents.
- **HNSW (Hierarchical Navigable Small Worlds)**: A graph-based index for fast Approximate Nearest Neighbor ($ANN$) search.

---

## 2. Advanced RAG Patterns

### Re-Ranking
Standard retrieval might return the top 10 documents by vector similarity, but those might not be the most "helpful".
- **Cross-Encoders**: A heavy model that takes (Query, Document) pairs and scores them accurately. Used to refine the top-K results from initial retrieval.

### Query Transformation
1. **Multi-Query**: Expand one query into 3-5 variations to capture more context.
2. **HyDE (Hypothetical Document Embeddings)**: Model generates a "fake" answer first, then uses that fake answer to search for real documents.

### GraphRAG
Instead of just vector proximity, it uses **Knowledge Graphs** (Nodes = Entities, Edges = Relationships).
- **Benefit**: Captures global themes and non-obvious connections that vector search misses.

---

## 3. Evaluation: The RAGAS Framework
Don't just "vibe check". Use the **RAG Triad**:
1. **Context Relevance**: Is the retrieved context useful for the query?
2. **Faithfulness (Groundedness)**: Is the answer derived *only* from the context?
3. **Answer Relevance**: Does the answer actually address the user's intent?

---

## Interview Questions

**1. "What is the difference between RAG and Fine-tuning?"**
> **RAG** is like giving a student an open book; it's great for facts, low cost, and no training needed. **Fine-tuning** is like the student studying and internalizing knowledge; it's better for style, tone, and specific patterns but expensive and prone to hallucinations on facts.

**2. "How would you handle 'Lost in the Middle' in RAG?"**
> LLMs tend to process information better at the start and end of a context window. **Solution**: Use **Re-ranking** to place the most relevant documents at the very top of the prompt or reduce the total number of retrieved documents.

**3. "Why use a Vector DB instead of a standard SQL DB with 'LIKE' queries?"**
> SQL 'LIKE' looks for exact keyword matches. Vector search looks for **semantic similarity**. (e.g., "fast cars" and "quick vehicles" would match in a vector DB but not in SQL).

---

## Code: Simple Retrieval Pattern (LangChain logic)
```python
# Semantic Search Flow
query_vector = embed_model.embed(query)
docs = vector_db.similarity_search(query_vector, k=5)
reranked_docs = cross_encoder.rank(query, docs)[:3]
final_prompt = f"Context: {reranked_docs}\n\nQuestion: {query}"
```
