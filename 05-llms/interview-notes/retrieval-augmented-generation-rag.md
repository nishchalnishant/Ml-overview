See also: [RAG deep dive](../applications/rag.md)

# RAG — Interview Q&A

---

**Q1. What is RAG and why does it exist?**

RAG is an architectural pattern that grounds LLM outputs in external, verifiable data by retrieving relevant context for every query. It exists because LLMs freeze knowledge at training time. When facts change weekly — internal docs, policy updates, live pricing — you cannot retrain the model each time. RAG decouples the reasoning engine (the LLM) from the knowledge store (a vector database), letting you update data without touching model weights.

The one-line interview hook: RAG gives the model better books without changing how the model thinks.

---

**Q2. Walk me through the two pipelines in a basic RAG system.**

**Ingestion (offline):** documents are loaded, chunked, embedded via an embedding model, and stored in a vector DB along with metadata (source, page, date).

**Retrieval (online):** the user query is embedded with the same model, a similarity search returns the top-K chunks, those chunks are injected into a prompt, and the LLM generates a grounded answer.

Critical rule: the same embedding model must handle both ingestion and query time. Different models live in incompatible vector spaces.

---

**Q3. What is chunking and why does it matter more than most teams realize?**

Chunking is splitting documents into segments before embedding. Get it wrong and retrieval silently fails regardless of model quality.

- Chunks too small: they lose the subject of a sentence, retrieval misses relevant content.
- Chunks too large: the embedding averages over too much text, diluting the signal; the LLM prompt bloats.

Industry starting point: 512 tokens with 10–20% overlap. The overlap prevents facts from being severed at chunk boundaries.

Three main strategies:
- **Fixed-size:** fast, ignores structure, breaks sentences arbitrarily.
- **Recursive:** splits on paragraph → sentence → word hierarchy. Default for 95% of production systems.
- **Semantic:** embeds every sentence, splits where cosine similarity drops below a threshold. High quality, expensive.

---

**Q4. How do you choose an embedding model?**

Check the MTEB leaderboard for your task type (retrieval, semantic similarity). Key tradeoffs:

| Factor | Implication |
|---|---|
| Max sequence length | Must match or exceed your chunk size |
| Dimensionality | Higher = better accuracy, more disk/RAM, slower search |
| Latency | API vs. self-hosted GPU |
| Privacy | Cloud APIs send data off-premise |

Matryoshka embeddings (some OpenAI and BGE models) let you truncate 1536-dim vectors to 256 without proportional quality loss — useful for cost-sensitive setups.

---

**Q5. What is hybrid search and when do you need it?**

Hybrid search combines dense (vector) retrieval with sparse (BM25/keyword) retrieval. Pure vector search fails on exact-match queries: product IDs, error codes like `0xc0000005`, rare acronyms, model serial numbers. The embedding model pushes rare tokens toward general meanings from training data.

Merge strategy: **Reciprocal Rank Fusion (RRF)** combines ranked lists without needing score normalization.

```
score(d) = Σ 1 / (k + rank(d))   where k = 60
```

A document ranked #1 in both lists wins decisively. One ranked #2 in vector and #15 in BM25 still beats one ranked #5 in only one list.

Use hybrid search for: e-commerce, code search, technical documentation, medical IDs.

---

**Q6. What is re-ranking and how does it differ from first-stage retrieval?**

First-stage retrieval uses bi-encoders: query and document are embedded separately, similarity is a dot product. Fast, O(N), but the query and document never "see" each other during scoring.

Re-ranking uses cross-encoders: query and document are concatenated and processed together. Attention heads can directly compare tokens across both. Much higher precision, O(K²) over the candidate set.

Workflow: retrieve 50–100 candidates with the bi-encoder, re-rank with a cross-encoder (Cohere Rerank, BGE-Reranker), pass only the top 3–5 to the LLM.

Cost: adds 100–300ms latency. Do not re-rank more than 50 chunks or latency spikes unacceptably.

---

**Q7. What is the "lost in the middle" problem?**

LLMs show a U-shaped recall curve over long prompts: high accuracy at the beginning and end of the context, degraded accuracy for content buried in the middle. Stanford research (2023) showed accuracy drops from ~90% to ~40-50% for the middle chunk of a 20-document context.

Practical fixes:
1. Re-rank hard, pass only 3–5 chunks. Fewer high-quality chunks beats 20 mediocre ones.
2. Place the most relevant chunk first, second-most relevant last, noise in the middle.
3. Use the LangChain `LongContextReorder` utility.

Do not blindly increase top-K thinking "more context is safer." It usually lowers quality.

---

**Q8. How do you evaluate a RAG system?**

The RAG triad (RAGAS framework):

- **Faithfulness (groundedness):** does every claim in the answer appear in the retrieved context? Extract claims, check entailment against chunks. Measures hallucination.
- **Answer relevance:** does the answer address the actual user question? An answer can be faithful but off-topic.
- **Context precision/recall:** did the retriever fetch the right chunks? If the gold answer contains facts not present in any retrieved chunk, recall is low — no amount of generation quality fixes that.

Retrieval-only metrics: Hit Rate (did the correct doc appear in top-K?), MRR (Mean Reciprocal Rank).

Always evaluate the retriever independently. If retrieval is weak, generation is built on sand.

---

**Q9. What is agentic RAG?**

Standard RAG is a single retrieve-then-generate pass. Agentic RAG wraps the retrieval process in a reasoning loop.

Capabilities an agent adds:
- **Routing:** choose which database or tool to query (finance DB vs. HR DB vs. web search).
- **Query decomposition:** split "compare 2022 vs 2023 revenue" into two sub-queries.
- **Multi-hop retrieval:** result from query A informs query B.
- **Self-correction:** if retrieved chunks score low on relevance, retry with a different query.

Cost: multiple LLM calls per request, 5–15s latency vs 1–2s for standard RAG. Only add agentic layers when the information is provably fragmented across sources.

---

**Q10. What is Self-RAG?**

Self-RAG fine-tunes the LLM itself to output special reflection tokens that control its own retrieval behavior:

- `[Retrieve]` — "I need external information before continuing."
- `[IsRel]` — "Is this chunk actually relevant?"
- `[IsSup]` — "Does this chunk support my claim?"
- `[IsUse]` — "Is my final answer actually good?"

Unlike agentic RAG (a software loop around the model), Self-RAG is a behavior baked into the model weights. It avoids retrieval on simple factual or conversational queries, reducing latency and token cost. True Self-RAG requires a specially fine-tuned model; you can approximate it with prompt engineering but lose the mathematical optimization.

---

**Q11. How do you handle structured data (tables, SQL) in a RAG pipeline?**

Do not embed raw CSVs or spreadsheets as text blobs. Embeddings cannot represent column relationships or do arithmetic.

The correct approach — **Text-to-SQL:**
1. Inject the DB schema (`CREATE TABLE` statements) into the prompt.
2. LLM generates a SQL query.
3. Execute the query against a read-only DB connection.
4. Convert result rows to a markdown table and pass that as retrieved context.
5. LLM generates a human-readable answer.

Security: use a read-only DB user, sanitize inputs, never allow `DROP` or `UPDATE` in generated SQL.

For small tables: serialize each row as `"Product: iPhone, Revenue: $1M, Date: 2023-01-01"` and embed those strings. Works at small scale, breaks at thousands of rows.

---

**Q12. What is Graph RAG and when does it beat vector RAG?**

Graph RAG stores entities and relationships in a knowledge graph (Neo4j, FalkorDB) extracted by an LLM during ingestion. Retrieval traverses graph edges rather than searching for nearest vectors.

When it wins:
- **Global summarization:** "What are the main themes across all 5,000 pages?" Vector RAG sees only top-K chunks. Graph RAG uses community summaries that aggregate mentions across the entire corpus.
- **Multi-hop reasoning:** "Is Person A connected to Patent Z?" No single document mentions both. Graph RAG follows `A → works_at → Company → developed → Z`.

When it loses: for simple single-document Q&A, graph construction is expensive overhead that hurts more than it helps. Also, adding new documents requires graph rebuilding or careful node-merging logic.

Hybrid Graph RAG: embed graph nodes for vector search to find the starting node, then traverse edges. Best of both worlds.

---

**Q13. How do multi-hop questions break standard RAG and what do you do about it?**

Multi-hop: "What is the climate of the city where the Eiffel Tower is located?" requires hop 1 (Eiffel Tower → Paris) then hop 2 (Paris → climate). A single vector search for "Eiffel Tower climate" may retrieve nothing.

Solutions:
- **Query decomposition:** LLM breaks the query into ordered sub-queries. Execute sub-query 1, use the result to construct sub-query 2.
- **RAG-Fusion:** generate 5 paraphrases of the query, run 5 parallel searches, merge with RRF. Increases recall for edge-case phrasings.
- **Iterative retrieval:** retrieved chunk mentions an entity, trigger another search on that entity.

Risk: error compounds across hops. If hop 1 fails, hop 2 fails. Self-RAG or explicit verification steps between hops mitigate this.

---

**Q14. What are common production failure modes in RAG and how do you debug them?**

When a RAG system hallucinates, the root cause is almost never "the LLM is bad." Work the chain backward:

| Symptom | Likely cause | Fix |
|---|---|---|
| Answer uses facts not in context | Retrieval returned wrong docs | Improve chunking, switch to hybrid search, add re-ranker |
| Retrieval returns right docs but answer is wrong | Prompt does not enforce groundedness | Add "answer only from context" instruction; add faithfulness check |
| Retrieval misses relevant content | Chunk boundary cuts the key sentence | Increase overlap; check tokenizer vs character count |
| Correct answer buried in 20-chunk context | Lost in the middle | Reduce top-K, use re-ranker, reorder chunks |
| Outdated answer | Stale index | Add document freshness metadata, trigger re-indexing pipeline |

Always evaluate retrieval and generation independently. A RAGAS eval that shows low context recall tells you the retriever is broken — no prompt engineering will fix that.

---

**Q15. RAG vs fine-tuning — how do you decide?**

This comes up in every senior interview. The clean framing:

| Problem | Solution |
|---|---|
| Facts change frequently | RAG |
| Facts are internal / private documents | RAG |
| Citations and verifiability matter | RAG |
| Model answers in the wrong format | Fine-tune |
| Model needs to sound like a domain expert | Fine-tune |
| Model needs to learn a specialized reasoning pattern | Fine-tune |
| Both fresh knowledge and consistent style needed | RAG + fine-tune |

RAG changes what the model knows at inference time. Fine-tuning changes how the model behaves. They are complementary, not competing. A medical assistant might be fine-tuned to write in clinical note format but still use RAG to retrieve the specific patient's records.
