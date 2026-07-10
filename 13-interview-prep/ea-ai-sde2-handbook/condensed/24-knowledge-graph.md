# Interview 24 — Knowledge Graph Construction for Lore & Story (Condensed)

BioWare narrative team needs an automated pipeline to turn 20 years of unstructured lore (wikis, GDDs, scripts, novels) into a queryable Knowledge Graph, so writers stop creating plot holes. Query example: "Who are all the enemies of Commander Shepard born on Earth?"

## Clarifying Questions to Ask
- Fixed ontology or dynamic discovery? → Loose node schema (Character/Faction/Location/Item), but relationship predicates should be discovered dynamically.
- How is entity resolution handled? → Not solved yet — "The Illusive Man" and "Jack Harper" must resolve to one node; candidate must design this.
- Who queries the graph — writers via Cypher/SQL? → No, narrative writers need a natural-language interface.
- Is the graph a one-time build or continuously updated as new docs land? (signals ingestion/versioning awareness)
- Do we need to preserve provenance/conflicting accounts (lies, retcons) or just one "truth"? (sets up the epistemic-tracking follow-up)

## Core Architecture
```
Raw docs → LLM Information Extraction (function calling, strict JSON schema)
        → Entity Resolution (embed + cosine-sim clustering, blocked by type)
        → Neo4j Graph DB (MERGE nodes/edges)
        → Text-to-Cypher QA Agent (NL question → Cypher → execute → summarize)
```
- LLM-based zero-shot IE instead of custom Spacy NER — no labeled training data needed, trades off cost/latency/hallucination risk.
- Entity resolution via embedding similarity (threshold ~0.90) — the piece that keeps the graph from becoming duplicate-node soup.
- Neo4j chosen over relational DB — native multi-hop traversal for queries like "enemies of friends of X."
- GraphCypherQAChain (or equivalent) for NL→Cypher→answer round trip.

## Talking Points That Signal Seniority
- Proactively flags entity resolution/deduplication as a required design piece, not an afterthought.
- Names "predicate explosion" (KILLED/MURDERED/ASSASSINATED all meaning the same thing) and proposes constraining the LLM to an allowed predicate taxonomy via the function-calling schema.
- Recommends full-text/fuzzy search or embedding lookup for node retrieval instead of exact-string Cypher `MATCH` (handles casing/typos).
- Raises coreference resolution ("He shot the alien") as a pre-processing step before extraction, not something to catch later.
- Proposes blocking (compare only same-type/same-prefix entities) to make entity resolution sub-O(N²) at scale.
- Suggests temporal edges (`start_time`/`end_time`) so relationships that change across games (allies → enemies) don't produce contradictory answers.
- Suggests source-citation properties on edges so answers are auditable back to the source document.
- Raises the "conflicting lore / lies" problem unprompted — reifying edges with `stated_by`/`reliability` rather than treating every extraction as universal truth.

## Top 3 Tradeoffs
- LLM extraction vs custom NLP (Spacy NER/RE): LLM is zero-shot, no labeled data, but expensive, slow, and can hallucinate nodes; Spacy is fast/free/deterministic but needs thousands of labeled examples.
- Graph RAG vs vector RAG: vector RAG fails on multi-hop questions ("enemies of friends of X"); Graph RAG handles multi-hop but costs a much heavier ETL/graph-build pipeline upfront.
- Exact-match Cypher vs fuzzy/full-text retrieval: exact `MATCH` breaks the moment the LLM's casing/spelling doesn't match ingestion normalization — full-text index trades a little precision for robustness.

## Toughest Follow-ups
**Q: Two extracted facts conflict — "Reapers destroyed Earth" vs a lying character's claim "Turians destroyed Earth." How does the graph handle this?**
A: Don't store relationships as universal truth — reify each edge with `stated_by` and `reliability` metadata (epistemic tracking). Writers can then filter for objective lore vs. subjective in-character claims instead of getting one poisoned answer.

**Q: Switching from GPT-4o to a small local model (Llama-3 8B) to cut cost — what breaks?**
A: Small models are unreliable at strict JSON/function-calling — they drop brackets or invent schema keys. Fix with constrained decoding (Outlines/Guidance) that forces token-level adherence to the JSON schema, making even an 8B model produce guaranteed-parsable output.

**Q: Graph is dense — "Reapers" connects to 500 characters, and visualization becomes a hairball. What do you do?**
A: Run PageRank/betweenness centrality over the graph and only surface the top-K most important relationships per node in the UI; don't try to render the full adjacency.

## Biggest Pitfall
Proposing standard vector-chunk RAG instead of an actual structured graph — it ignores the explicit requirement and fails outright on multi-hop relationship queries, which is an automatic drop to No Hire territory.
