# Interview 15 — Production RAG Pipeline with LLMs
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer building an internal knowledge-base assistant for EA Game Developers. Developers want to ask questions like: *"How do I implement custom physics materials in Frostbite?"* and get an accurate answer based on the internal developer documentation.

Your task is to **build a production-ready Retrieval-Augmented Generation (RAG) pipeline** from scratch. 

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Ingestion scale (How many documents? PDF vs HTML vs Markdown?)
- Chunking strategy (How do we preserve code blocks?)
- Retrieval quality (How do we handle exact keyword searches vs semantic concepts?)
- Hallucination prevention (What if the LLM invents a C++ function?)
- Evaluation (How do we measure if the pipeline is actually improving?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"What formats are the documents in?"**
   → *Answer: Mostly Markdown files from GitHub, and some Confluence HTML pages. They contain a lot of C++ code blocks.*

2. **"How do we handle code blocks during chunking?"**
   → *Answer: Good question. We cannot split a code block in half. You need to propose a strategy.*

3. **"Is the system purely vector-based, or do we need hybrid search?"**
   → *Answer: Developers often search for exact function names (e.g., `fb::PhysicsMaterial`). We probably need hybrid search.*

4. **"How will we evaluate the RAG system?"**
   → *Answer: We have a test set of 100 QA pairs. We need a way to automatically evaluate retrieval and generation quality.*

---

## Part 4 — Expected Assumptions

- **Architecture:** Document Loaders ➔ Semantic Chunking ➔ Embedding (e.g., OpenAI `text-embedding-3`) ➔ Vector DB (Qdrant/Milvus) ➔ Retriever ➔ LLM Generator (GPT-4o/Claude).
- **Code Handling:** Standard character-based text splitters will ruin code. Must use a Markdown-aware or AST-aware splitter.
- **Serving:** FastAPI backend returning streaming responses with citations.

---

## Part 5 — High-Level Solution

```
  [Offline Ingestion]
  Markdown Docs ➔ MarkdownHeaderTextSplitter (Keeps headers + code blocks intact)
       │
       ▼
  Embedding Model ➔ Vector DB (Stores chunk, embedding, and metadata: filename/header)
  
       =========================================================

  [Online RAG API]
  User Query: "How to compile Frostbite physics?"
       │
       ▼
  1. Query Rewriter: (Optional) Expands query for better retrieval.
  2. Hybrid Search: Dense (Vector) + Sparse (BM25) search in Vector DB.
  3. Re-ranker: Pass Top 20 results to a Cross-Encoder to get Top 5.
  4. Generator: LLM constructs answer using Top 5 contexts, with strict citation prompt.
```

**Core ML Component:** A pipeline focused heavily on **Retrieval Quality**. If retrieval fails, generation fails. Implementing Hybrid Search + Re-ranking is the industry standard for high-quality RAG.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Intelligent Chunking
- Use LangChain's `MarkdownHeaderTextSplitter`. It splits text based on `#`, `##`, `###`.
- This guarantees that a section titled `## Custom Materials` and its associated C++ code block stay in the same chunk.

### Step 2: Hybrid Retrieval
- Vector search (Cosine Similarity) finds conceptual matches ("how to add gravity").
- BM25 (Sparse keyword search) finds exact matches ("btRigidBody").
- Qdrant supports querying both simultaneously and fusing the scores via Reciprocal Rank Fusion (RRF).

### Step 3: Generation & Citations
- The Prompt must clearly label contexts (e.g., `<context id="doc_1">...</context>`).
- Instruct the LLM to output answers with citations (e.g., "Use the physics engine [doc_1].").

---

## Part 7 — Complete Python Code

```python
"""
production_rag.py - Advanced RAG pipeline with LangChain & Qdrant
"""
import logging
from typing import List, Dict
import openai
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_text_splitters import MarkdownHeaderTextSplitter
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Developer RAG API")

# Setup clients
openai_client = openai.AsyncOpenAI(api_key="internal-key")
qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION = "dev_docs"

# ---------------------------------------------------------------------------
# 1. Ingestion Pipeline (Offline)
# ---------------------------------------------------------------------------
def ingest_markdown_doc(markdown_text: str, source_url: str):
    """Chunks markdown intelligently and pushes to Vector DB."""
    
    # Split by headers to preserve semantic structure and code blocks
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = markdown_splitter.split_text(markdown_text)
    
    logger.info(f"Split document into {len(chunks)} chunks.")
    
    # In production, batch embed these
    # For simplicity, mocked insertion...
    # qdrant.upsert(...)

# ---------------------------------------------------------------------------
# 2. Retrieval Pipeline (Online)
# ---------------------------------------------------------------------------
async def embed_query(query: str) -> List[float]:
    resp = await openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    return resp.data[0].embedding

async def retrieve_documents(query: str, top_k: int = 5) -> List[Dict]:
    """Retrieves documents. (Assumes Qdrant is configured for Hybrid Search)."""
    
    query_vector = await embed_query(query)
    
    # Execute search
    search_result = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )
    
    results = []
    for hit in search_result:
        results.append({
            "content": hit.payload["text"],
            "source": hit.payload["source_url"],
            "score": hit.score
        })
    return results

# ---------------------------------------------------------------------------
# 3. Generation Pipeline (Online)
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str

@app.post("/v1/rag/ask")
async def ask_rag(req: QueryRequest):
    # 1. Retrieve
    contexts = await retrieve_documents(req.query, top_k=5)
    
    if not contexts:
        return {"answer": "No relevant documentation found.", "sources": []}
        
    # 2. Build Prompt
    system_prompt = (
        "You are an expert EA C++ Engine Developer. Answer the question based ONLY on the provided contexts.\n"
        "If you do not know the answer based on the context, say 'I don't know'.\n"
        "Cite your sources inline using [Source_URL]."
    )
    
    context_str = "\n\n".join(
        [f"--- Context ---\nSource: {c['source']}\n{c['content']}" for c in contexts]
    )
    
    user_prompt = f"Contexts:\n{context_str}\n\nQuestion: {req.query}"
    
    # 3. Generate
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    
    return {
        "answer": response.choices[0].message.content,
        "retrieved_sources": [c["source"] for c in contexts]
    }
```

---

## Part 8 — Deployment

### Vector Database
- Deploy Qdrant in a StatefulSet on Kubernetes.
- Enable `sparse_vectors` alongside dense vectors to support BM25 exact keyword search.

### Application Backend
- FastAPI running on standard CPU nodes.
- Use asynchronous API calls (`AsyncOpenAI`) so the FastAPI event loop isn't blocked while waiting for the LLM API to respond (which can take 5+ seconds).

---

## Part 9 — Unit Testing

```python
import pytest
from langchain_text_splitters import MarkdownHeaderTextSplitter

def test_markdown_chunking():
    # Verify that code blocks aren't destroyed
    md_text = """
## Physics Setup
Here is how to do it:
```cpp
void setup() {
    fb::Physics p = new fb::Physics();
}
```
    """
    
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "H2")])
    chunks = splitter.split_text(md_text)
    
    assert len(chunks) == 1
    # The code block should remain perfectly intact inside the chunk
    assert "void setup()" in chunks[0].page_content
    # Metadata should capture the header
    assert chunks[0].metadata["H2"] == "Physics Setup"
```

---

## Part 10 — Integration Testing

- **RAG Evaluation (Ragas framework):**
  - Offline CI/CD step. When documentation is updated, trigger an evaluation script.
  - Run 50 test questions through the pipeline.
  - Assert that `Context Precision` > 0.8 (Did the retriever find the right docs?).
  - Assert that `Faithfulness` > 0.9 (Did the LLM answer match the docs, or did it hallucinate?).

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Latency vs Context Size** | Feeding 20 chunks (10k tokens) into GPT-4o takes time and costs money. Use a Re-ranker (like `bge-reranker`). Retrieve 50 chunks from Qdrant, pass them through the local Re-ranker (cross-encoder), and only send the absolute best 3 chunks to the LLM. |
| **Token Limits** | Code blocks can be huge. If a chunk exceeds the embedding model's context window (e.g., 8191 tokens for OpenAI), the embedding will truncate. Add a recursive character fallback splitter after the Markdown splitter. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Naive Chunking vs Semantic Chunking | Chunking by exactly 1000 characters is easy, but splits C++ functions in half, destroying logic. Semantic/Markdown chunking preserves logic but creates chunks of vastly different sizes, which can make vector matching slightly uneven. |
| BM25 vs Vector Search | Vector search understands "how to make things fall" = "gravity". BM25 fails this. But BM25 understands `btRigidBody` perfectly, while Vector search might blur it with other physics terms. Hybrid search is mandatory for technical docs. |

---

## Part 13 — Alternative Approaches

1. **HyDE (Hypothetical Document Embeddings):** If user queries are very short (e.g., "physics error 12"), vector matching fails because queries don't look like documents. Use an LLM to generate a hypothetical answer first, then embed that answer to search the DB.
2. **Graph RAG:** Parse the codebase into an Abstract Syntax Tree (AST) graph. Query the graph to understand relationships (e.g., "Function A calls Function B").

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Lost in the Middle | The LLM ignores the correct answer because it was placed in the middle of a massive context block. | Sort the retrieved chunks so the highest-scoring chunks are placed at the beginning and the end of the context window, leaving lower-scoring chunks in the middle. |
| Hallucinating APIs | The LLM invents a C++ function that looks real but doesn't exist in the engine. | Severe penalty in system prompt. Post-generation verification: Use a fast regex to extract any `fb::` namespaces from the answer and verify they exist in the retrieved context. |

---

## Part 15 — Debugging

**Symptom:** A developer searches for "how to initialize audio", and the bot returns documents about "how to initialize rendering". The audio documents exist in the database.

**Debugging steps:**
1. Check the vector search results. Is the audio document in the Top 5?
2. If No: The embedding model thinks "initialize rendering" is closer to "initialize audio" than the actual audio document (often happens if the audio document doesn't use the word 'initialize'). 
3. **Fix:** Enable Hybrid Search. The keyword "audio" will force BM25 to boost the audio documents to the top, overcoming the embedding's failure.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `openai_api_latency_s` | > 8s → Model is degraded, switch to fallback model or stream response to UI. |
| `retrieval_empty_result_rate` | > 5% → Users are asking about undocumented features. |
| `user_thumbs_down_rate` | > 10% → Review query logs for poor retrieval performance. |

---

## Part 17 — Production Improvements

1. **Streaming Responses:** Use `stream=True` in the OpenAI API and `StreamingResponse` in FastAPI. This reduces Time-to-First-Token (TTFT) from 5 seconds to 500ms, vastly improving user experience.
2. **Conversational Memory:** Pass the previous chat history into the prompt so developers can ask follow-up questions ("Can you explain line 5 of that code block?").
3. **Citation Formatting:** Have the LLM output structured JSON citations, and build a UI component that highlights the exact paragraph in the source document when the user clicks the citation.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"The Markdown splitter works great, but some of our C++ files are just massive 5,000-line raw text files with no markdown headers. How do you chunk these so we don't split functions in half?"**
2. **"We notice the LLM often says 'Based on the context, here is the answer...' which is annoying. How do we force it to just give the answer directly, but still refuse to answer if the context is missing?"**
3. **"Our vector database is returning excellent results, but the LLM is running into context window limits. We can only fit 3 chunks, but we need information from 10 different chunks to answer a complex architecture question. How do you solve this?"**

---

## Part 19 — Ideal Answers

**Q1 (Chunking Raw Code):**
> "We should use an AST-based (Abstract Syntax Tree) text splitter, like LangChain's `Language.CPP` splitter, or Tree-sitter. These tools parse the actual syntax of the code and guarantee that they split the text at valid boundaries—like the end of a class or function definition—ensuring we never slice a function down the middle."

**Q2 (Prompt Engineering for Tone):**
> "We update the system prompt with strict negative constraints and few-shot examples. Constraint: 'Do not use introductory filler phrases like "Based on the context". Answer directly.' We also provide a Few-Shot example in the prompt showing a correct, direct answer, and a correct refusal format."

**Q3 (Context Limits / Multi-Doc Synthesis):**
> "We implement a Map-Reduce RAG pipeline. 
> 1. **Map:** We pass all 10 chunks individually (in parallel) to a fast/cheap LLM (like Claude 3 Haiku) with the prompt: 'Extract information relevant to the user query from this chunk.'
> 2. **Reduce:** We take the 10 summarized extractions, combine them, and pass them into the main GPT-4o model for the final synthesis. This bypasses the context limit and filters out irrelevant noise from the raw chunks."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Anticipates the need for specialized chunking (Markdown/AST) for technical documents.
- Understands why Hybrid Search (Dense + Sparse) is strictly necessary for code/error-code search.
- Mentions Map-Reduce or Re-ranking to handle context limits.
- Writes robust async Python code.

### Hire
- Sets up a standard Vector DB + OpenAI architecture.
- Understands the core tenets of RAG (Retrieval vs Generation).
- Knows how to structure a system prompt to prevent hallucination.
- Might need a hint regarding how standard chunking breaks code blocks.

### Lean Hire
- Understands RAG conceptually but relies purely on theoretical LangChain abstractions without knowing how the DB or Embedding APIs actually work under the hood.
- Fails the integration testing/evaluation questions.

### Lean No Hire
- Proposes a naive chunking strategy (`text[:1000]`) and refuses to acknowledge why it breaks code.
- Cannot explain how to mitigate LLM hallucinations.

### No Hire
- Cannot write code to call an LLM API.
- Suggests fine-tuning the LLM on the docs instead of using RAG.
