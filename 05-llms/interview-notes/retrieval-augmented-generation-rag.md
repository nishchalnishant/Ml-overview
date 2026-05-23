---
module: Llms
topic: Interview Notes
subtopic: Retrieval Augmented Generation Rag
status: unread
tags: [llms, ml, interview-notes-retrieval-augm]
---
See also: [RAG deep dive](../applications/rag.md)

# Retrieval-Augmented Generation (RAG) — Interview Notes

---

## Why RAG Exists

**The problem**: LLMs encode knowledge into weights during training. Once trained, those weights are frozen. Factual knowledge about the world changes continuously — policy documents are updated, new research is published, internal records are modified. You cannot retrain a model every time a fact changes. You also cannot include every private enterprise document in a public model's training data.

**The core insight**: separate the reasoning engine (the LLM) from the knowledge store (a retrieval system). The model's job is to read evidence and reason about it. The knowledge store's job is to provide current, relevant evidence at query time. Updating knowledge means updating the store, not retraining the model.

**The mechanics**: two independent pipelines:

**Ingestion** (runs offline): documents → chunk → embed (using an embedding model) → store (vector DB with metadata). The output is an indexed collection of text segments with their vector representations.

**Retrieval + generation** (runs at query time): user query → embed (same model) → similarity search → top-K chunks → inject into prompt → LLM generates grounded answer.

The critical constraint: ingestion and query-time embedding must use the same model. Different models produce different vector spaces — a query vector from model B cannot retrieve documents embedded with model A.

**What breaks**: RAG reduces knowledge-gap hallucination but does not eliminate context infidelity (the model ignoring the retrieved context and answering from its weights). RAG adds latency (embedding + search + generation) and operational complexity (maintaining a separate retrieval system). Knowledge in the store can itself be wrong or outdated if ingestion is not maintained.

**What the interviewer is testing**: whether you understand why RAG and fine-tuning are complementary, not competing. RAG changes what the model knows at inference time. Fine-tuning changes how the model behaves. A medical assistant might be fine-tuned to write in clinical note format but still use RAG to retrieve current drug interactions.

**Common traps**: saying "RAG prevents hallucination" (it reduces one type; context infidelity hallucination is a separate failure mode); proposing RAG for tasks that require stable behavioral change (fine-tuning is better for format/style consistency); not knowing that the embedding model must be the same at ingestion and query time.

---

## Chunking

**The problem**: you cannot embed an entire 200-page document as one vector. A single vector represents the average of all the document's content — specific facts get buried. Retrieving the whole document to answer one question also blows up the LLM context window. Chunking solves both: it creates units small enough to embed meaningfully and pass to the LLM.

**The core insight**: the chunk is the unit of retrieval. If the answer to a query spans a chunk boundary, it will not be retrieved. If the chunk is too large, the embedding averages over irrelevant content, degrading retrieval precision. The chunk size determines whether retrieval succeeds or silently fails, regardless of model quality.

**The mechanics — three main strategies**:

- **Fixed-size**: split at fixed token count (e.g., 512 tokens) with N% overlap (~10–20%). Overlap prevents severing a fact that straddles a boundary. Fast, simple, ignores document structure. Breaks sentences arbitrarily.
- **Recursive**: split on hierarchical delimiters: paragraph break → sentence boundary → word boundary. Preserves semantic units. Default for ~95% of production RAG systems.
- **Semantic**: embed each sentence, split where cosine similarity between adjacent sentences drops below a threshold. Produces semantically coherent chunks. High quality, high compute cost. Use when document structure is irregular.

Industry starting point: 512 tokens with 10–20% overlap for most structured documents. Validate by checking whether your gold-standard answers fall entirely within single chunks — if not, your chunking is too small.

**Parent-child chunking**: index small chunks (256 tokens) for retrieval precision; return the parent chunk (1024 tokens) for LLM context. This resolves the tension between retrieval precision and answer context richness.

**What breaks**: chunks too small lose the grammatical subject of a sentence; retrieval scores semantically empty text. Chunks too large embed over too much irrelevant content; the query vector matches text about one subsection but retrieves the whole page. Wrong chunk sizes are the silent killer — the eval metrics look reasonable but the system is missing answers it could retrieve.

**What the interviewer is testing**: whether you can diagnose chunking problems from symptoms (retrieval misses, context bloat) and know which chunking strategy fits which document type.

**Common traps**: treating 512 tokens as universally correct without validating on your documents; not using overlap (boundary facts are frequently split and lost); not knowing about parent-child chunking as the solution to precision vs. context tradeoff.

---

## Hybrid Search

**The problem**: dense vector search is excellent for semantic queries ("what is the refund policy?") but fails on exact-match queries: product SKUs, error codes like `0xc0000005`, model serial numbers, rare proper nouns. The embedding model pushes rare tokens toward generic representations from training data, obscuring exact identifiers.

**The core insight**: exact-match failures and semantic-match failures have different root causes that require different retrieval mechanisms. Dense retrieval cannot be made to reliably match exact strings (you cannot retrain the embedding model for every rare identifier). BM25/keyword search cannot be made to understand paraphrases. Use both.

**The mechanics**: retrieve independently from both the dense index (vector DB, top-K by cosine similarity) and the sparse index (BM25/Elasticsearch, top-K by TF-IDF weighted keyword score). Merge using Reciprocal Rank Fusion (RRF):

```
RRF_score(d) = Σ 1 / (k + rank_i(d))   where k = 60
```

A document ranked #1 in both retrieval systems wins decisively. A document ranked in only one system can still surface if it ranks highly there. RRF does not require score normalization — you are combining ranks, not scores, so incompatible score scales do not matter.

**What breaks**: hybrid search adds a second retrieval path, increasing ingestion complexity (separate BM25 index) and query latency. RRF does not have a semantic interpretation — it is a fusion heuristic that works empirically but does not optimize for a ground-truth relevance measure. Tuning the fusion weight (α in weighted hybrid) requires a labeled evaluation set.

**What the interviewer is testing**: whether you know the specific failure mode of dense-only retrieval (exact-match queries) and can describe RRF concretely.

**Common traps**: saying hybrid search "always outperforms" dense-only (it does on mixed-query distributions, but for purely semantic queries BM25 can hurt); not knowing RRF or thinking you need to normalize scores before merging; not knowing that keyword search is still necessary in 2024 despite advances in embedding models.

---

## Re-Ranking

**The problem**: bi-encoder (dense) retrieval embeds query and document independently, then scores by dot product. The query and document never "see each other" during scoring. This is fast but imprecise — two texts can have similar embeddings without actually addressing the same question.

**The core insight**: cross-encoders process query and document jointly, allowing attention heads to directly compare tokens across both. This is much more accurate than bi-encoder scoring. But cross-encoders are too slow to score every document in a large corpus — they scale as O(N×D) where D is document encoding time. The solution is a two-stage pipeline: fast bi-encoder for candidate recall, slow cross-encoder for precision.

**The mechanics**: retrieve 50–100 candidates with the bi-encoder (O(1) per query via ANN search). Re-rank those candidates with a cross-encoder re-ranker (Cohere Rerank, BGE-Reranker, or a fine-tuned BERT-based model). Pass only the top 3–5 to the LLM.

Why 50–100 candidates and not 20? The re-ranker often promotes a document from rank 40 to rank 1 — it is not just polishing the top results. If you pass only 20 candidates, you constrain the re-ranker's ability to recover missed relevant content.

**What breaks**: re-ranking adds 100–300ms latency depending on candidate count. Do not re-rank more than 50–100 chunks — latency spikes are unacceptable. The re-ranker itself is a model that can be wrong. A re-ranker trained on general-domain pairs may not perform well in your domain — domain-specific fine-tuning helps.

**What the interviewer is testing**: whether you can articulate the bi-encoder vs. cross-encoder tradeoff and why the two-stage design is necessary.

**Common traps**: saying the re-ranker is the "main" retrieval step (it is the refinement step; recall is established by the bi-encoder); not knowing the candidate count tradeoff (too few candidates means the re-ranker cannot recover missed relevant docs); proposing re-ranking as a replacement for hybrid search (they address different problems).

---

## The Lost-in-the-Middle Problem

**The problem**: you increase top-K to 20 to give the model more context. Accuracy goes down. This seems counterintuitive — more relevant information should produce better answers.

**The core insight**: LLMs exhibit a U-shaped recall curve over context position. Stanford research (2023) showed accuracy at ~90% for context at the beginning and end of the window, dropping to ~40–50% for context in the middle of a 20-document prompt. More chunks do not mean more usable context — they mean more noise in the positions the model reliably attends to.

**The mechanics — mitigations**:

1. **Re-rank hard, reduce aggressively**: pass only 3–5 high-quality chunks. The gain from having the relevant chunk present far outweighs the loss from excluding borderline-relevant chunks.
2. **Ordering by relevance**: most relevant chunk first (or last). Second-most relevant last (or first). Noise in the middle.
3. **Parent-child chunking**: reduces the number of separate context segments by returning a parent block instead of many small chunks.
4. **Explicit labeling**: number chunks and include "Cite the evidence by number" instructions, which forces the model to actively locate the relevant chunk.

**What breaks**: aggressively reducing top-K increases the risk of excluding the relevant chunk entirely if retrieval recall is imperfect. The mitigation is investing in better retrieval (re-ranker, hybrid search) not increasing top-K.

**What the interviewer is testing**: whether you know that more context can hurt quality and understand why — it is an attention positional bias, not a capacity limit.

**Common traps**: recommending "just increase the context window" (the problem is positional bias, not window size); not knowing the U-shaped recall finding; increasing top-K without also adding a re-ranker.

---

## RAG Evaluation: The RAGAS Triad

**The problem**: you cannot evaluate a RAG system by testing the LLM in isolation. The LLM may be performing correctly given bad context, or may be ignoring correct context. The retriever may be working but the generator is hallucinating. These failure modes are invisible when you evaluate only end-to-end answer quality.

**The core insight**: evaluate the retriever and the generator independently, then evaluate the full system. If retrieval recall is low, no generation improvement will fix the system. If faithfulness is low, no retrieval improvement will fix the system. Each component has independent failure modes.

**The mechanics — RAGAS framework**:

**Faithfulness (groundedness)**: does every claim in the answer appear in the retrieved context? Decompose the answer into individual claims; check each claim for entailment against the retrieved chunks. Score = (entailed claims) / (total claims). Measures hallucination.

**Answer relevance**: does the answer address the user's actual question? An answer can be perfectly faithful to the context and still be off-topic. Score can be measured by generating "what question would this answer?" and checking similarity to the original question.

**Context precision**: are the retrieved chunks relevant to the question? Measures retrieval precision. Score = (relevant retrieved chunks) / (total retrieved chunks).

**Context recall**: is all information needed to answer the question present in the retrieved chunks? If the gold answer requires a fact not in any retrieved chunk, recall is zero — generation quality is irrelevant. Score = (answer claims supported by retrieved chunks) / (total answer claims).

**Retrieval-only metrics**: Hit Rate (does the relevant document appear in top-K?), MRR (Mean Reciprocal Rank — at what rank does the relevant document first appear?).

**What breaks**: faithfulness measurement requires an NLI (natural language inference) model or an LLM-as-judge. Both can be wrong. RAGAS scores are noisy on small evaluation sets — you need at least 100–200 examples for stable metrics. Context recall requires knowing the gold answer in advance, which means maintaining a labeled evaluation set.

**What the interviewer is testing**: whether you know to evaluate retrieval and generation independently, and can name specific metrics for each.

**Common traps**: evaluating only end-to-end accuracy (masks whether the retriever or generator is failing); using ROUGE/BLEU for RAG evaluation (they measure textual similarity, not factual grounding); not having a labeled evaluation set (you cannot detect regressions without baseline numbers).

---

## Agentic RAG

**The problem**: standard RAG is a single retrieve-then-generate pass. It fails for: (1) queries that require information from multiple sources; (2) queries that are ambiguous and need clarification before retrieval; (3) queries that require intermediate reasoning steps where the answer to one step informs the next retrieval.

**The core insight**: wrap the retrieval process in a reasoning loop. The model decides whether it needs more information, formulates a more specific query, retrieves, evaluates the result, and either answers or retrieves again. This is RAG where the model controls its own retrieval strategy.

**The mechanics — capabilities added**:

- **Routing**: the model selects which knowledge source (finance DB, HR docs, web search) to query, rather than searching a single index.
- **Query decomposition**: "Compare 2022 and 2023 revenue" becomes two sub-queries executed in sequence.
- **Multi-hop retrieval**: result from sub-query A contains an entity; sub-query B is generated using that entity.
- **Self-correction**: if retrieved chunks score low on relevance (checked by the model), retry with a rephrased query.

**What breaks**: multiple LLM calls per request — latency increases from 1–2s to 5–15s. Errors compound across hops — if hop 1 is wrong, hop 2 is built on wrong premises. Without a step budget, the agent can loop indefinitely. Each additional LLM call increases cost.

**What the interviewer is testing**: whether you know when agentic RAG is justified (fragmented information across sources, multi-hop queries) vs. when it is over-engineering (single-source, single-hop queries).

**Common traps**: proposing agentic RAG as the default architecture (it adds latency and complexity that most applications do not need); not knowing that error propagation across hops is the core reliability challenge; thinking agentic RAG is always better than standard RAG.

---

## Self-RAG

**The problem**: standard RAG retrieves on every query, even when the question is conversational ("thanks, got it") or answerable from general knowledge without retrieval. Unnecessary retrieval wastes latency and cost. But heuristics for when to retrieve (question length, keyword presence) are brittle.

**The core insight**: train the model itself to decide whether retrieval is needed, and to self-evaluate the quality of retrieved content. These meta-decisions become model behaviors rather than external orchestration logic.

**The mechanics**: Self-RAG fine-tunes the LLM to output special reflection tokens:

- `[Retrieve]` — "I need external information before I can continue."
- `[IsRel]` — "Is this retrieved chunk relevant to my query?" (yes/no)
- `[IsSup]` — "Does this chunk support the claim I'm about to make?" (yes/no/partial)
- `[IsUse]` — "Is my generated answer actually useful?" (score 1–5)

These tokens allow the model to skip retrieval for simple queries, check relevance of retrieved content, and verify its own output quality. Unlike an agent loop (which is external code wrapping the model), Self-RAG bakes these behaviors into the model weights via fine-tuning on a specially constructed training set.

**What breaks**: true Self-RAG requires a specially fine-tuned model — you cannot replicate it exactly with prompt engineering on a standard model. The reflection tokens must appear in the tokenizer vocabulary and the model must be trained to generate them reliably. Prompt-engineered approximations (telling the model to assess relevance) are less reliable because the behavior is not baked into the weights.

**What the interviewer is testing**: whether you can distinguish Self-RAG (a model behavior baked into weights) from standard agentic RAG (external orchestration logic wrapping a standard model).

**Common traps**: saying you can implement Self-RAG with a standard model and a prompt (you can approximate it, not implement it); confusing the reflection tokens with regular CoT output; not knowing that Self-RAG's main contribution is conditional retrieval (not retrieving when unnecessary).

---

## Handling Structured Data: Text-to-SQL

**The problem**: you have data in tables, and users ask questions that require computation — aggregations, filtering, joins. Embedding raw CSV rows as text loses column relationships and arithmetic meaning. A vector search for "average revenue by region Q3 2023" cannot compute the average — it can only retrieve rows that mention "average revenue."

**The core insight**: the query against structured data is a program (SQL), not a vector search. The LLM's job is to write the SQL program; the database executes it. This separates the natural language understanding problem (LLM) from the computation problem (database).

**The mechanics — Text-to-SQL pipeline**:

1. Inject the database schema (`CREATE TABLE` statements + sample rows) into the LLM prompt.
2. LLM generates a SQL query.
3. Execute the query against a read-only database connection.
4. Convert result rows to a markdown table or structured text.
5. Pass the formatted result as context to the LLM for the final natural-language answer.

Security requirements: read-only DB user; no `DROP`, `UPDATE`, `INSERT` in generated SQL (enforce at the execution layer, not just in the prompt); sanitize inputs before SQL execution.

**What breaks**: the LLM may generate syntactically valid but semantically wrong SQL (joins on the wrong columns, wrong aggregation, wrong date range). Complex schemas with many similar tables require schema simplification or schema linking (first identify which tables are relevant) before SQL generation. Long schema injections push SQL generation instructions into the "lost in the middle" zone.

**What the interviewer is testing**: whether you know not to embed tabular data as raw text blobs and can describe the Text-to-SQL pattern correctly.

**Common traps**: proposing to embed CSV rows as text chunks (breaks for computation queries); not knowing that security must be enforced at the execution layer, not only in the prompt; thinking schema injection alone handles large databases with hundreds of tables.

---

## Graph RAG

**The problem**: vector RAG retrieves semantically similar chunks. For two types of queries, it fails structurally: (1) global summarization ("what are the main themes across all 5,000 pages?") — no single chunk captures the corpus-wide pattern; (2) multi-hop reasoning ("is person A connected to patent Z?") — no single document mentions both entities.

**The core insight**: some queries are not about semantically similar text — they are about relationships. A knowledge graph makes relationships first-class objects that can be traversed. Graph traversal reaches answers that vector search cannot.

**The mechanics**: during ingestion, run an LLM over documents to extract entity-relationship triples: (Entity A, relation, Entity B). Store these in a knowledge graph (Neo4j, FalkorDB). Generate community summaries by clustering entities and summarizing their interconnections.

At query time:
- **Local search**: embed the query, find the nearest entities via vector search on graph nodes, then traverse outward via edges.
- **Global search**: use precomputed community summaries rather than traversing the full graph; answers corpus-wide themes without exhaustive traversal.

**When graph beats vector**: multi-hop questions requiring relationship traversal; global summarization; tasks where entity co-occurrence across documents matters. **When vector beats graph**: single-document Q&A; queries where semantic similarity is the right matching signal; construction of a high-quality knowledge graph would require expensive LLM extraction over the full corpus.

**What breaks**: graph construction is expensive — extracting entity-relation triples from 5,000 pages requires thousands of LLM calls. Adding new documents requires either graph rebuilding or careful node-merging logic to avoid duplicate entities. Graph quality is limited by extraction quality; missed relationships are permanently absent.

**What the interviewer is testing**: whether you can identify when graph RAG is the right tool versus over-engineering, and what the construction cost tradeoff is.

**Common traps**: proposing graph RAG for simple single-document Q&A (unnecessary overhead); not knowing that graph construction requires LLM extraction (not just embedding); treating graph and vector RAG as mutually exclusive (hybrid graph RAG — embed graph nodes for vector search, traverse edges for retrieval — is often the right architecture).

---

## Multi-Hop Queries

**The problem**: "What is the climate of the city where the Eiffel Tower is located?" requires two facts from two different documents: Eiffel Tower → Paris (hop 1), Paris climate (hop 2). A single vector search for "Eiffel Tower climate" may retrieve nothing because no document discusses both together. Standard RAG fails structurally, not because of retrieval quality.

**The core insight**: decompose the multi-hop query into ordered sub-queries. The answer to sub-query 1 becomes an input to sub-query 2. Each sub-query is individually answerable by single-document retrieval.

**The mechanics — solutions**:

**Query decomposition**: LLM breaks the original query into a sequence of sub-queries. Execute sub-query 1 → extract the key entity from the result → construct sub-query 2 using that entity → execute sub-query 2 → synthesize final answer.

**RAG-Fusion**: generate 5 paraphrases of the query; run 5 parallel searches; merge candidate sets with RRF. Increases recall for queries that are phrased in ways the bi-encoder does not map well to the relevant documents.

**Iterative retrieval**: after retrieving for the original query, scan retrieved chunks for named entities not in the original query; trigger additional searches on those entities. Continue until no new relevant entities are found or a step budget is reached.

**What breaks**: errors propagate across hops — if hop 1 retrieves the wrong entity, hop 2 searches for the wrong thing and the final answer is confidently wrong. Each hop is a separate LLM call, multiplying latency. Self-RAG or explicit verification prompts between hops mitigate error propagation but add cost.

**What the interviewer is testing**: whether you understand why multi-hop fails for standard RAG and can describe a concrete solution.

**Common traps**: saying "just use a bigger context window and retrieve more chunks" (the problem is structural — no single chunk has both answers); not knowing about RAG-Fusion as an alternative to sequential decomposition; not accounting for hop error propagation when designing the pipeline.

---

## RAG vs. Fine-Tuning

**The problem**: an LLM is performing poorly on your task. You have two investment options: build a RAG pipeline, or fine-tune the model. These are not fungible — using the wrong one wastes significant time and money.

**The core insight**: RAG changes what the model knows at inference time (provides facts). Fine-tuning changes how the model behaves (teaches formats, styles, patterns). The right question is not "which is better?" but "what is actually broken?"

**The mechanics — decision framework**:

| What is broken | Right lever |
|---|---|
| Facts are stale or private | RAG |
| Citations and verifiability are required | RAG |
| Output format is inconsistent | Fine-tune (SFT) |
| Model needs domain-specific tone or vocabulary | Fine-tune |
| Model needs a specialized reasoning pattern | Fine-tune |
| Knowledge changes frequently AND style is wrong | RAG + fine-tune |

A medical assistant might be fine-tuned on clinical note format but use RAG to retrieve current drug interaction data. These are orthogonal improvements: fine-tuning affects all queries (format, style); RAG affects queries where the retrieval is relevant.

**What breaks when you use the wrong lever**: fine-tuning to inject facts into weights fails because facts baked into weights go stale, and the model can still hallucinate when uncertain about a fine-tuned fact. RAG to fix format inconsistency fails because no amount of retrieved context teaches the model a new output format — that requires weight updates.

**What the interviewer is testing**: whether you understand the mechanism of each approach, not just that both exist. The interviewer wants to see the clean framing: RAG = inference-time knowledge; fine-tuning = permanent behavior change.

**Common traps**: proposing fine-tuning when the problem is knowledge staleness (you will retrain every time data changes); proposing RAG when the problem is inconsistent output format (retrieved context does not teach format); not knowing that RAG + fine-tuning is the right answer when both problems exist simultaneously.

---

## Production Failure Modes and Debugging

**The problem**: your RAG system is producing wrong or hallucinated answers. The naive assumption is "the LLM is wrong." This is almost always incorrect. Most RAG failures trace to the retriever, not the generator. Debugging by adjusting prompts when the retriever is broken wastes time.

**The core insight**: work the chain backward from symptom to component. Evaluate retrieval and generation independently to localize the failure before applying any fix.

**The mechanics — symptom-to-cause mapping**:

| Symptom | Likely cause | Fix |
|---|---|---|
| Answer uses facts not in any retrieved chunk | Retrieval failure | Improve chunking; switch to hybrid search; add re-ranker |
| Retrieval returns right chunks but answer is wrong | Context infidelity (model ignoring context) | Add "answer only from context" instruction; add faithfulness post-check |
| Relevant chunk exists but is not retrieved | Chunk boundary cut the key sentence | Increase overlap; check tokenizer count vs. character count |
| Correct answer buried in long context | Lost in the middle | Reduce top-K; use re-ranker; reorder by relevance |
| Answer is factually correct but outdated | Stale index | Add document freshness metadata; schedule re-indexing |
| Answer is hallucinated but retrieval was correct | The LLM | Add RAG faithfulness checks; consider fine-tuning or different model |

**Debugging protocol**: (1) Run the query; inspect retrieved chunks manually — is the answer there? (2) If yes: the generator is ignoring context. Fix the prompt. (3) If no: the retriever is failing. Fix chunking, embedding, or hybrid search. (4) Run RAGAS evaluation on a representative set to quantify retrieval recall before spending time on generation improvements.

**What breaks**: retrieving without logging makes debugging impossible. Always log retrieved chunks, similarity scores, and the final prompt at query time. Production RAG systems need observability as a first-class feature.

**What the interviewer is testing**: whether you would start debugging from retrieval evaluation, not prompt adjustment. This is the key diagnostic insight most candidates miss.

**Common traps**: adjusting the prompt when the retrieval recall is low (the model cannot answer a question whose answer is not in the context, regardless of prompt quality); increasing top-K when the problem is lost-in-the-middle (this makes the problem worse); treating all RAG failures as "LLM hallucination" without evaluating the retriever.

## Flashcards

**Fixed-size?** #flashcard
split at fixed token count (e.g., 512 tokens) with N% overlap (~10–20%). Overlap prevents severing a fact that straddles a boundary. Fast, simple, ignores document structure. Breaks sentences arbitrarily.

**Recursive?** #flashcard
split on hierarchical delimiters: paragraph break → sentence boundary → word boundary. Preserves semantic units. Default for ~95% of production RAG systems.

**Semantic?** #flashcard
embed each sentence, split where cosine similarity between adjacent sentences drops below a threshold. Produces semantically coherent chunks. High quality, high compute cost. Use when document structure is irregular.

**Routing?** #flashcard
the model selects which knowledge source (finance DB, HR docs, web search) to query, rather than searching a single index.

**Query decomposition?** #flashcard
"Compare 2022 and 2023 revenue" becomes two sub-queries executed in sequence.

**Multi-hop retrieval?** #flashcard
result from sub-query A contains an entity; sub-query B is generated using that entity.

**Self-correction?** #flashcard
if retrieved chunks score low on relevance (checked by the model), retry with a rephrased query.

**[Retrieve]?** #flashcard
"I need external information before I can continue."

**[IsRel]?** #flashcard
"Is this retrieved chunk relevant to my query?" (yes/no)

**[IsSup]?** #flashcard
"Does this chunk support the claim I'm about to make?" (yes/no/partial)

**[IsUse]?** #flashcard
"Is my generated answer actually useful?" (score 1–5)

**Local search?** #flashcard
embed the query, find the nearest entities via vector search on graph nodes, then traverse outward via edges.

**Global search?** #flashcard
use precomputed community summaries rather than traversing the full graph; answers corpus-wide themes without exhaustive traversal.
