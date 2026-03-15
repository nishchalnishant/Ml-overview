# RAG

## Overview

* What is RAG?
  *
* Types of RAG
  * Traditional RAG
  * Graph RAG
* Traditional RAG VS Graph RAG
* Summary of RAG eco-system

\-----------------------------------------------------------------------------------------

### What is RAG?

<figure><img src="../.gitbook/assets/image (4).png" alt=""><figcaption></figcaption></figure>

* LLM + external resources which were not included in their training
  * i.e. Any LLM (gpt-4o, Gemini, llama) + internal document, service now ticket etc
* RAG is GenAI design technique that enhances LLM with external knowledge, thus improving the LLMS with&#x20;
  * Proprietary knowledge - It includes proprietary info which wasn't initially used to train the LLMs such as emails, documentation etc. &#x20;
  * Up to date information - RAG application supply LLMs with info from updated data resources.
  * Citing resources - RAG enables LLMs to cite specific resources thus allowing users to verify the factual accuracy of responses.
* RAG includes --
  * Indexing --
    * Data preparation step where data on which retrieval(next step) is performed is extracted and cleaned from data sources and converted into plain text.
    * Ex - If you want to create a RAG for Rx trouble shooting using the email conversation between the end user and CODE orange. You can't pass the entire email conversation to LLM since it might exceeds the context window of the LLM.\
      Hence we break the entire content into smaller and managable pieces called chunks , this process is called chunking.\
      These are then transformed into high dimensional vectors with help of embedding models which gives us a list of chiunk pairs of the data source.
  * Retrieval&#x20;
    * Users request is used to query an outside data store such as Vector store, SQL DB etc. ( here we can use differnet type of DB which leads to different types of DB like tradional rag,Graph based RAG etc.
    * The goal is to get supporting data for LLM's response.
  * Augmentation&#x20;
    * The retrieved data is combined with user's request using a template with additional formatting and instructions to create a prompt.
    * This augmentation can be of three types. Iterative, recursice and Adaptive
    *

        <figure><img src="../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>
  * Generation&#x20;
    * The prompt is passed to the LLMs which then generates a response to the query
* Types of RAG
  * Traditional RAG
  * Graph RAG
* Traditional RAG VS Graph RAG



Summary of RAG ecosystem

<figure><img src="../.gitbook/assets/image (3) (1).png" alt=""><figcaption></figcaption></figure>

---

## Why RAG?

- LLMs have a **knowledge cutoff** and no access to private or live data.
- **RAG** injects relevant external content into the prompt so the model can answer from that context and cite sources.
- Benefits: **proprietary knowledge**, **up-to-date info**, **citations**, and reduced hallucination when the answer is grounded in retrieved text.

---

## RAG architecture (detailed)

1. **Indexing (offline)**  
   - Ingest documents → clean and normalize text → **chunk** → **embed** chunks → store (chunk text + vector) in a **vector store** (and optionally keyword index).

2. **Retrieval (online)**  
   - User query → **embed** query → **similarity search** (e.g. k-NN in vector DB) → optionally **rerank** → top-k chunks.

3. **Augmentation**  
   - Build prompt: system instruction + **retrieved chunks** (formatted) + user question. Can be **iterative** (multiple retrieval steps), **recursive** (chunk then sub-chunk), or **adaptive** (decide how much to retrieve).

4. **Generation**  
   - Send prompt to LLM → get response; optionally ask model to cite chunk IDs.

---

## Document chunking

- **Why**: Context window is limited; chunks must be small enough to fit many in the prompt but large enough to be self-contained.
- **Strategies**: Fixed size (e.g. 512 tokens) with overlap; sentence or paragraph boundaries; semantic chunking (split at topic changes). Overlap reduces broken context at boundaries.
- **Best practice**: Match chunk size to embedding model and LLM context; test retrieval quality (recall) vs chunk size.

---

## Retrieval pipelines

- **Single-stage**: Embed query → vector search → take top-k. Simple and fast.
- **Two-stage**: Vector search → **reranker** (cross-encoder or small model) over candidates to get final top-k. Better precision, more latency.
- **Hybrid**: Combine **vector** (semantic) and **keyword** (BM25) search: e.g. reciprocal rank fusion (RRF) or weighted sum of normalized scores. Helps when exact terms matter (names, IDs).

---

## Reranking

- **Reranker**: Model that scores (query, chunk) pairs (e.g. cross-encoder). Takes top-N from first-stage retrieval, reranks, returns top-k.
- **Effect**: Improves relevance and reduces noise in the context; adds latency and cost. Often worth it for critical apps.

---

## Hybrid search

- **Vector**: semantic similarity (embeddings).  
- **Keyword**: BM25 or similar over tokenized text.  
- **Combination**: Run both; merge results with RRF or learned weights. Useful for mixed queries (conceptual + exact match).

---

## Context injection

- **Template**: Placeholder in the prompt for “retrieved context”, e.g. “Use the following excerpts to answer. Cite [id] when relevant.”  
- **Order**: Usually put context right before the user question.  
- **Length**: Stay under context limit; if many chunks, truncate or summarize.  
- **Formatting**: Clear section labels and chunk IDs help the model cite correctly.

---

## RAG evaluation

- **Retrieval**: Recall@k (is the gold chunk in top-k?), MRR, NDCG; or downstream **answer correctness** (e.g. LLM-as-judge or exact match).
- **Generation**: Faithfulness (answer grounded in context?), relevance (answers the question?), citation accuracy.
- **End-to-end**: QA benchmarks with your docs; A/B test with and without RAG or with different chunking/retrieval settings.

---

## Latency considerations

- **Embedding** query + **vector search** + optional **rerank** + **LLM** call. Each step adds latency.
- **Optimizations**: Cache embeddings for repeated queries; use faster/smaller reranker; reduce k or chunk size; use faster vector index (e.g. HNSW with lower ef); batch embedding calls if many queries.

---

## Scaling RAG systems

- **Indexing**: Batch embed and upsert; use distributed vector DB or sharding for very large corpora.
- **Retrieval**: Vector DB scaling (replicas, partitioning); consider approximate search (ANN) vs exact for huge collections.
- **Generation**: Scale LLM serving (batching, multiple replicas); consider smaller/faster models for simple queries.
- **Freshness**: Incremental indexing and periodic full rebuild; TTL or versioning if documents change often.

---

## Quick revision

- **RAG** = retrieve relevant chunks → inject into prompt → LLM generates answer. Enables proprietary, up-to-date, citable answers.
- **Chunking**: balance size and overlap; semantic or boundary-based splits.
- **Retrieval**: vector (semantic), keyword (BM25), or **hybrid**; **rerank** for better precision.
- **Evaluate**: retrieval recall, faithfulness, relevance, citations; tune chunk size, k, and reranker.
- **Scale**: batch indexing, vector DB scaling, LLM serving; optimize latency (cache, smaller k, faster index).

