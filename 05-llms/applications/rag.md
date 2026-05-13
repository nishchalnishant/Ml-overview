# Retrieval-Augmented Generation (RAG)

RAG grounds LLM responses in external documents retrieved at query time. Instead of relying on knowledge baked into model weights during training, the system fetches relevant context and passes it to the model as part of the prompt.

**When to use RAG:**
- Facts change frequently (news, product docs, internal wikis)
- Answers must be attributed to specific sources
- Domain is too narrow or proprietary to justify retraining
- Knowledge needs to be updateable without GPU budget

**When NOT to use RAG:**
- The problem is behavioral (tone, format, style) — use fine-tuning
- Latency budget cannot absorb retrieval overhead
- The knowledge is static and already in pretraining data

---

## 1. Architecture

```
User Query
    │
    ▼
Query Rewriting (optional)
    │
    ▼
Embedding Model → Query Vector
    │
    ▼
Vector Store ──── ANN Search ────► Top-K Chunks
    │                                    │
    │                               Reranker (optional)
    │                                    │
    └─────────────────────────────► Context Assembly
                                         │
                                         ▼
                                   Prompt + Context
                                         │
                                         ▼
                                        LLM
                                         │
                                         ▼
                                   Final Answer + Citations
```

---

## 2. Document Ingestion

### Chunking Strategies

| Strategy | When to use | Tradeoff |
| :--- | :--- | :--- |
| **Fixed-size (tokens)** | Simple, fast baseline | Splits mid-sentence |
| **Sentence splitting** | Coherent retrievable units | Short chunks, less context |
| **Recursive character** | General-purpose (LangChain default) | Respects paragraph → sentence → word hierarchy |
| **Semantic chunking** | High-quality retrieval | Expensive embedding per chunk |
| **Document-structure-aware** | Markdown, PDFs with headers | Preserves logical sections |

**Chunk size guidance:**
- 256–512 tokens: precise retrieval, may lack context
- 512–1024 tokens: good default for most use cases
- 1024–2048 tokens: more context per chunk, noisier retrieval

**Overlap:** typically 50–100 token overlap between chunks to avoid splitting context across chunk boundaries.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)
```

### Metadata Enrichment

Attach metadata to each chunk before indexing. This enables filtered retrieval:

```python
for chunk in chunks:
    chunk.metadata.update({
        "source": doc.metadata["source"],
        "section": extract_section_header(chunk.page_content),
        "date": doc.metadata.get("last_modified"),
        "doc_type": classify_document(doc),
    })
```

---

## 3. Embedding Models

Embeddings map text to dense vectors in a semantic space. Retrieval finds the nearest vectors by cosine similarity.

### Model Selection

| Model | Dimensions | Strengths | Context window |
| :--- | :--- | :--- | :--- |
| **text-embedding-3-small** | 1536 | Fast, cheap | 8191 tokens |
| **text-embedding-3-large** | 3072 | Best OpenAI quality | 8191 tokens |
| **BAAI/bge-m3** | 1024 | Multilingual, open-source | 8192 tokens |
| **sentence-transformers/all-mpnet-base-v2** | 768 | Strong English, small | 512 tokens |
| **Cohere embed-v3** | 1024 | Strong retrieval performance | 512 tokens |

**Important:** use the same embedding model for indexing and querying. Model updates require re-indexing.

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"}
)
```

---

## 4. Vector Databases

| Database | Deployment | Strengths | Notes |
| :--- | :--- | :--- | :--- |
| **Chroma** | Local / self-hosted | Simple, dev-friendly | Good for prototyping |
| **Pinecone** | Managed cloud | Zero-ops, scalable | Paid |
| **Weaviate** | Self-hosted / cloud | Hybrid search (BM25 + vector) | Good for production |
| **Qdrant** | Self-hosted / cloud | Fast, Rust-based | Good filter performance |
| **pgvector** | PostgreSQL extension | Keeps data in existing DB | Good for small-medium scale |
| **FAISS** | In-memory library | Fastest for batch search | Not a full DB |

### ANN Algorithms

**HNSW (Hierarchical Navigable Small World):** most common in production vector DBs.
- Builds a multi-layer graph where higher layers have fewer, longer-range connections
- Search: start at top layer, greedily navigate toward query, descend to finer layers
- **Complexity:** $O(\log n)$ build, $O(\log n)$ query
- **Tradeoff:** `ef_construction` (build quality) vs `ef` (query quality) vs speed

**IVF (Inverted File Index):** clusters vectors, searches only nearest clusters.
- **IVFPQ:** combines IVF with Product Quantization for compressed storage

```python
import faiss
import numpy as np

d = 1536  # embedding dimension
n = 100000  # number of vectors

# HNSW index
index = faiss.IndexHNSWFlat(d, 32)  # M=32 neighbors per layer
index.hnsw.efConstruction = 200
index.add(embeddings_matrix)

# Search
query = np.array([query_embedding])
distances, indices = index.search(query, k=10)
```

---

## 5. Retrieval Strategies

### Dense Retrieval (Semantic)

Standard embedding similarity search. Finds semantically similar chunks even with different wording.

### Sparse Retrieval (BM25 / TF-IDF)

Keyword-based. Fast and surprisingly strong for exact-match queries, technical terms, and proper nouns.

**BM25 scoring:**

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1+1)}{f(t,d) + k_1(1-b+b\frac{|d|}{\text{avgdl}})}$$

where $k_1 \approx 1.2–2.0$ and $b \approx 0.75$ are tuning parameters.

### Hybrid Search

Combine dense and sparse retrieval scores using Reciprocal Rank Fusion (RRF):

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + r(d)}$$

where $k=60$ is a common default and $r(d)$ is the rank of document $d$ in each ranker.

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(chunks, k=10)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)
```

---

## 6. Query Optimization

### Multi-Query Expansion

Generate multiple reformulations of the query to improve recall:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
# Generates 3 variants of the user query, retrieves for each, deduplicates
```

### HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer first, embed it, then retrieve based on the richer semantic signal:

```python
def hyde_retrieve(query, vectorstore, llm, k=5):
    hypothetical_doc = llm.complete(
        f"Write a concise, factual answer to: {query}\n\nAnswer:"
    )
    # Embed the hypothetical answer instead of the raw query
    hyp_embedding = embeddings.embed_query(hypothetical_doc)
    return vectorstore.similarity_search_by_vector(hyp_embedding, k=k)
```

Works especially well when the user query is short and the relevant documents are longer.

### Step-Back Prompting

Rewrite the query at a higher level of abstraction before retrieval:

```
User: "What are the side effects of metformin in elderly patients?"
Step-back: "What are general considerations for metformin use?"
```

---

## 7. Reranking

Initial retrieval returns candidates quickly. A **reranker** (cross-encoder) scores each candidate against the query more accurately, at the cost of more compute.

**Bi-encoder (retrieval):** embed query and documents independently, dot product similarity. Fast but approximate.

**Cross-encoder (reranking):** concatenate query + document, run through transformer, output relevance score. Slow but accurate.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Score all candidates
pairs = [(query, chunk.page_content) for chunk in candidates]
scores = reranker.predict(pairs)

# Sort by score, keep top k
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
top_k = [chunk for chunk, score in reranked[:5]]
```

**Two-stage pattern:** retrieve top 20 with ANN → rerank → pass top 5 to LLM. Balances speed and precision.

---

## 8. Context Assembly and Prompt Design

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer questions using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."
Always cite your sources using [Source: filename]."""),
    ("human", """Context:
{context}

Question: {question}

Answer:""")
])

def format_context(chunks):
    formatted = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        formatted.append(f"[{i+1}] Source: {source}\n{chunk.page_content}")
    return "\n\n".join(formatted)
```

**"Lost in the Middle" problem:** LLMs pay more attention to context at the beginning and end than the middle. When retrieving multiple chunks:
- Put the most relevant chunk first
- Or use RAG-Fusion: generate multiple hypothetical answers, rerank, present the top result prominently

---

## 9. Advanced Patterns

### Self-RAG

The model decides when to retrieve, what to retrieve, and evaluates whether retrieved passages are relevant/grounded. Adds `[Retrieve]`, `[Relevant]`, `[Supported]` tokens to the generation.

### CRAG (Corrective RAG)

Evaluates retrieval quality. If retrieval quality is low, falls back to web search. If quality is ambiguous, uses both retrieved docs and web results.

### GraphRAG

Extracts entities and relationships from documents, builds a knowledge graph, and answers multi-hop queries using graph traversal + LLM synthesis. Best for questions requiring reasoning across multiple facts: "How does entity A relate to entity B through entity C?"

---

## 10. Evaluation

Use automated evaluation to score at scale; human evaluation for ground truth.

### RAGAS Metrics

| Metric | Measures | How |
| :--- | :--- | :--- |
| **Faithfulness** | Is the answer grounded in the retrieved context? | LLM decomposes answer into claims, checks each against context |
| **Answer Relevancy** | Does the answer address the question? | Embed question vs generated answer, cosine similarity |
| **Context Precision** | Are retrieved chunks relevant to the question? | LLM judges each chunk for relevance |
| **Context Recall** | Did retrieval surface all necessary information? | Checks if ground-truth answer is derivable from context |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

eval_data = {
    "question": questions,
    "answer": answers,
    "contexts": [[c.page_content for c in retrieved] for retrieved in all_retrieved],
    "ground_truth": ground_truths,
}

result = evaluate(
    Dataset.from_dict(eval_data),
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print(result)
```

---

## 11. Production Considerations

| Challenge | Solution |
| :--- | :--- |
| **Index freshness** | Scheduled re-ingestion + delta updates |
| **Embedding model changes** | Version-pin embeddings; plan full re-index when upgrading |
| **Multi-tenant isolation** | Namespace/partition by tenant in vector DB |
| **Latency** | Cache embeddings of frequent queries; use HNSW with tuned `ef` |
| **Context window overflow** | Hard limit on retrieved chunks; summarize if needed |
| **Hallucination despite context** | Stricter system prompt; faithfulness filter on output |
| **Security** | Metadata filtering to scope retrieval to user's accessible docs |

> [!TIP]
> **Debugging RAG failures:** trace the pipeline — wrong answer usually traces back to (1) stale/missing documents, (2) poor chunking losing key context, (3) retrieval returning irrelevant chunks, (4) prompt not enforcing groundedness. Fix in this order before touching the generation model.
