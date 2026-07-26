---
module: System Design
topic: System Design
subtopic: LLM Product Assistant Full System
status: unread
tags: [productionml, ml, system-design-llm-product, llm]
---
# LLM-Powered Product Assistant — System Design

End-to-end design for an LLM-based assistant embedded in a product (e.g., "chat with your data" for a SaaS app, an internal knowledge assistant, or a customer-facing support bot). Canonical system design question at any company shipping generative AI features (OpenAI, Anthropic, Notion, Intercom, Salesforce).

**Scale:** 1M+ queries/day, p50 first-token latency <500ms, p99 full-response <8s, cost ceiling ~$0.01–0.05/query, hallucination rate <2% on grounded queries.

---

## 1. Problem Framing

### Clarifying Questions
- **Closed-domain or open-domain?** Answering only from company docs (grounded) vs general knowledge — changes the entire architecture.
- **Synchronous chat or async/agentic?** Single-turn Q&A vs multi-step tool-calling changes latency budget and failure surface.
- **What's the cost of a wrong answer?** Internal tool (low stakes, fast iteration) vs customer-facing support (wrong answer → refund/liability) vs medical/legal (high stakes) — this decides how much guardrailing and human-in-the-loop is needed.
- **Freshness requirement?** Static knowledge base vs content that changes hourly (pricing, inventory) — decides RAG index refresh cadence vs live tool calls.
- **Multi-tenant?** Each customer's data must stay isolated — decides whether retrieval is per-tenant-indexed or filtered at query time.

### Metric Trade-offs

**The grounding-latency-cost triangle is the central problem:**

| Approach | Groundedness | Latency | Cost |
|---|---|---|---|
| No retrieval, raw LLM | Low (hallucination-prone) | Fastest | Cheapest |
| RAG, single retrieval pass | Medium-high | +100-300ms | + embedding + retrieval cost |
| RAG + reranking + multi-hop | Highest | +500ms-2s | 2-3x retrieval cost |
| Agentic (tool calls, multi-turn) | Highest (verifiable) | Slowest, unbounded | Highest, unbounded |

The staff-level answer is never "always use the most grounded approach" — it's picking the point on this triangle that matches the query's stakes, and routing different query types to different points.

---

## 2. System Architecture

```
User Query
    │
    ▼
┌──────────────┐
│ Query Router │──── Classify: needs retrieval? needs tool call? safety-sensitive?
│ (small/fast  │      <20ms
│  classifier) │
└──────┬───────┘
       │
       ├─── (no retrieval needed) ──────────────────┐
       │                                             │
       ▼                                             │
┌──────────────┐    Vector DB / Hybrid Search        │
│  Retrieval   │◄─── (embeddings + BM25)              │
│  (RAG)       │      per-tenant filtered             │
└──────┬───────┘                                      │
       │                                              │
       ▼                                              │
┌──────────────┐                                      │
│  Reranker    │──── Cross-encoder, top-k → top-n      │
│  (optional)  │      +50-150ms                        │
└──────┬───────┘                                      │
       │                                              │
       ▼                                              ▼
┌─────────────────────────────────────────────────────┐
│              Context Assembly                        │
│  system prompt + retrieved chunks + chat history      │
│  + tool schemas, trimmed to context budget            │
└──────────────────────┬────────────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │   LLM Inference    │──── streaming, KV cache, speculative decoding
              │  (serving layer)   │
              └─────────┬──────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Post-processing  │──── citation extraction, PII redaction,
              │  & Guardrails     │      safety classifier, format validation
              └─────────┬──────────┘
                        │
                        ▼
                 Streamed Response
                        │
                        ▼
              ┌──────────────────┐
              │  Feedback Loop    │──── thumbs up/down, edit-and-resend,
              │                   │      escalation-to-human rate → eval set
              └──────────────────┘
```

---

## 3. Retrieval Design (RAG)

### Chunking Strategy

| Strategy | When to use | Trade-off |
|---|---|---|
| Fixed-size (500 tokens, overlap 50) | Uniform docs, quick baseline | Splits mid-thought, loses structure |
| Semantic/structure-aware (by heading, paragraph) | Structured docs (docs, wikis) | Better coherence, uneven chunk sizes |
| Recursive with parent-doc lookup | Long docs needing broader context | Retrieve small chunk, expand to parent for generation |

### Hybrid Retrieval

Pure dense (embedding) retrieval misses exact-match cases (part numbers, error codes, proper nouns unseen in training). Pure sparse (BM25) misses semantic paraphrase. Production systems combine both:

```python
def hybrid_retrieve(query: str, top_k: int = 20):
    dense_results = vector_db.search(embed(query), top_k=top_k)
    sparse_results = bm25_index.search(query, top_k=top_k)

    # Reciprocal Rank Fusion — combine two rankings without needing
    # calibrated scores from either system
    fused_scores = {}
    for rank, doc in enumerate(dense_results):
        fused_scores[doc.id] = fused_scores.get(doc.id, 0) + 1 / (60 + rank)
    for rank, doc in enumerate(sparse_results):
        fused_scores[doc.id] = fused_scores.get(doc.id, 0) + 1 / (60 + rank)

    return sorted(fused_scores.items(), key=lambda x: -x[1])[:top_k]
```

### Reranking

Retrieval optimizes for recall over a large candidate pool cheaply; a cross-encoder reranker then trades latency for precision on a small candidate set:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query: str, candidates: list[str], top_n: int = 5):
    pairs = [[query, c] for c in candidates]
    scores = reranker.predict(pairs)  # joint query-doc attention, not independent embeddings
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [c for c, s in ranked[:top_n]]
```

Bi-encoder (embedding) retrieval encodes query and doc independently — fast, indexable, but can't model query-doc interaction. Cross-encoder reranking sees both jointly — much more accurate, but can't be precomputed, so it's only run on the top-k candidates.

---

## 4. Context Assembly and Prompt Design

### Context Budget

```
total_context_window = 128K tokens (example)
  - system prompt:         ~500 tokens (fixed)
  - tool schemas:          ~1000 tokens (if agentic)
  - retrieved chunks:      ~4000 tokens (top-5 reranked, ~800 tokens each)
  - chat history:          ~2000 tokens (last N turns, summarized if long)
  - response reserve:      ~2000 tokens
  ────────────────────────────────
  used: ~9500 / 128K — retrieval quality matters far more than window size
```

**Lost-in-the-middle**: models attend more reliably to content at the start and end of context than the middle. Place the most load-bearing retrieved chunk first or last, not buried in position 3 of 5.

### Long Conversation History

Naive approach (keep full history) blows the context budget and re-pays attention cost every turn. Production pattern:

```python
def assemble_history(turns: list[dict], max_tokens: int = 2000):
    # Keep the last few turns verbatim (recent context matters most)
    recent = turns[-3:]
    older = turns[:-3]

    if not older:
        return recent

    # Summarize older turns into a compact running summary
    summary = summarize(older)  # cheap/fast model, not the main LLM
    return [{"role": "system", "content": f"Earlier conversation summary: {summary}"}] + recent
```

---

## 5. Grounding and Hallucination Mitigation

**The core failure mode**: the model generates plausible-sounding text not supported by retrieved context, especially when the retrieved context doesn't actually answer the question.

**Mitigations, layered:**

1. **Prompt-level**: explicit instruction to answer only from provided context, and to say "I don't know" when context is insufficient — necessary but not sufficient, models still hallucinate under instruction.
2. **Citation-forcing**: require the model to tag claims with source chunk IDs; claims without a valid citation are flagged or stripped post-hoc.
3. **Retrieval confidence gating**: if top retrieval score is below a threshold, route to "I don't have enough information" or escalate to human, rather than letting the LLM generate from weak context.
4. **Post-hoc verification (NLI-style)**: a smaller, cheaper model checks whether each generated claim is entailed by the retrieved context; unsupported claims get flagged before the response ships.

```python
def verify_groundedness(response_claims: list[str], retrieved_context: str, nli_model) -> list[bool]:
    """Each claim must be entailed by the context, not just topically related."""
    results = []
    for claim in response_claims:
        label = nli_model.predict(premise=retrieved_context, hypothesis=claim)
        results.append(label == "entailment")
    return results
```

**Trade-off**: verification adds latency and cost proportional to claim count. For low-stakes internal tools, skip it. For customer-facing or regulated domains, it's close to mandatory.

---

## 6. Serving and Latency

### Where latency goes (typical breakdown for a grounded, streaming response)

| Stage | Latency | Optimization lever |
|---|---|---|
| Query routing/classification | 10-20ms | Small distilled classifier, not the main LLM |
| Embedding the query | 10-30ms | Batch if possible, cache repeated queries |
| Vector search | 20-100ms | ANN index (HNSW), approximate not exact |
| Reranking | 50-150ms | Skip for low-stakes queries, cap candidate count |
| Time to first token | 200-500ms | KV cache reuse, prefix caching for shared system prompt |
| Full generation | 2-8s | Streaming (perceived latency ≠ actual latency), speculative decoding |

**Prefix/prompt caching**: the system prompt and tool schemas are identical across nearly every request. Caching their KV state avoids recomputing attention over thousands of fixed tokens every single query — this is one of the highest-leverage latency optimizations for RAG systems with a large fixed system prompt.

**Streaming**: users perceive latency to *first token*, not total completion time. Stream tokens as generated; this alone can cut perceived latency by 5-10x even when total generation time is unchanged.

---

## 7. Cost Management

```
cost_per_query ≈ (input_tokens × input_price) + (output_tokens × output_price)
              + embedding_cost (query + any new docs indexed)
              + reranker_cost (if using a hosted cross-encoder)
```

**Levers, cheapest to most impactful:**
- Route simple queries to a smaller/cheaper model; reserve the large model for complex or ambiguous queries (a router classifier decides).
- Cache full responses for repeated/common queries (FAQ-shaped traffic is highly duplicated).
- Trim retrieved context aggressively — more chunks isn't free, and past a point it degrades quality (lost-in-the-middle) while raising cost.
- Prompt caching (see above) — repeated system-prompt tokens billed once per cache window, not per request, on providers that support it.

---

## 8. Evaluation

**Offline eval set**: curated (query, expected-answer-properties) pairs, not just (query, exact-answer) — LLM outputs are non-deterministic in phrasing, so exact-match is the wrong comparison.

```python
def evaluate_rag_response(query, response, retrieved_context, reference_answer, judge_model):
    """LLM-as-judge for properties that are hard to check programmatically."""
    return {
        "groundedness": judge_model.score(  # is every claim supported by context?
            f"Context: {retrieved_context}\nResponse: {response}\nIs every claim in the response supported by the context? Score 1-5."
        ),
        "relevance": judge_model.score(     # does it actually answer the question?
            f"Query: {query}\nResponse: {response}\nDoes this fully answer the query? Score 1-5."
        ),
        "correctness": semantic_similarity(response, reference_answer),  # vs known-good answer
    }
```

**LLM-as-judge caveats**: judge models have their own biases (favor longer responses, favor responses stylistically similar to their own outputs) — validate judge scores against a sample of human ratings before trusting them as the primary signal, and re-validate periodically as the underlying model changes.

**Online metrics**: thumbs up/down rate, edit-and-resend rate (user rewording the query = implicit failure signal), escalation-to-human rate, session abandonment after a response.

**Retrieval-specific eval**: measure retrieval quality independent of generation — recall@k against a labeled (query, relevant-doc) set — so regressions can be isolated to "retrieval got worse" vs "generation got worse" rather than only seeing an end-to-end score drop.

---

## 9. Guardrails and Safety

- **Input guardrails**: prompt injection detection (user input trying to override system instructions), PII detection before it reaches the LLM or logs.
- **Output guardrails**: safety classifier on generated response before it streams to the user; for customer-facing bots, a hard block-list for topics requiring human handling (legal threats, self-harm, refund disputes above a $ threshold).
- **Tool-calling guardrails** (if agentic): allow-list of callable tools/actions, confirmation step before any state-mutating action (never let the model directly execute a destructive action without a checkpoint).

---

## Interview Angles

### Q: Your RAG system's retrieval looks correct (the right chunks show up in top-5), but the LLM still gives a wrong answer. What do you check? [Medium]
A: Isolate retrieval from generation before assuming a model problem. Check: (1) Is the correct chunk actually reaching the prompt, or getting dropped/truncated by the context budget? (2) Position — is it buried in the middle of a long context (lost-in-the-middle)? (3) Is the system prompt instructing the model to prioritize its own knowledge over retrieved context in a conflict? (4) Run the same context through a groundedness-verification pass — if the claim genuinely isn't in the chunk despite being topically adjacent, it's a retrieval-precision problem disguised as a generation problem, not a real generation failure.

**Cross-questions to expect:**

- *You move the correct chunk to the front of the context and the answer becomes correct. What does that tell you, and what do you fix?* -> It confirms lost-in-the-middle, not a retrieval problem — the model had the right information but didn't attend to it reliably at that position. Fix at the assembly layer (reorder by importance, not retrieval rank) rather than retuning the retriever, which was already working.
- *Groundedness verification says the claim isn't supported by the chunk, but a human reading the chunk says it clearly is. What's your NLI checker missing?* -> Off-the-shelf NLI models are trained on short, generic sentence pairs and struggle with domain-specific phrasing, negation, or claims that require combining two chunks. Treat verification-model disagreement as evidence the verifier needs domain fine-tuning or a higher-context-window formulation, not as ground truth by default.

**Trap:** immediately retraining or swapping the LLM when a wrong answer appears. Most "hallucination" bugs in production RAG systems are retrieval or context-assembly bugs, not model capability problems — diagnose before changing the expensive component.

### Q: How do you decide whether a query needs retrieval at all? [Medium]
A: A lightweight router classifier (much cheaper/faster than the main LLM) trained or prompted to distinguish query types: general knowledge/chit-chat (skip retrieval), needs-company-specific-data (retrieve), needs-live-data/tool-call (route to a tool, not RAG). Always-retrieve wastes latency and money on queries that don't need it and can actively hurt quality by injecting irrelevant context that the model then has to reason around. Never-retrieve leaves the model hallucinating on anything domain-specific. The router itself needs its own eval set and periodic review, since misrouting either direction has a distinct failure mode.

**Cross-questions to expect:**

- *Your router is a small fast model and is itself sometimes wrong. A no-retrieval-needed query gets routed to RAG. What actually breaks?* -> Usually nothing catastrophic — retrieval runs, finds nothing relevant, and a decent prompt still lets the model answer from general knowledge; the cost is wasted latency/spend, not necessarily a wrong answer. The more dangerous misroute is the opposite direction.
- *A needs-retrieval query gets routed as no-retrieval-needed. Why is this the dangerous direction, and how do you catch it in production, not just eval?* -> The model confidently answers from parametric knowledge on a domain-specific question it had no business answering ungrounded — this is a silent hallucination, not an error, so it won't show up as a crash or timeout. Catch it via the online feedback loop (thumbs-down rate segmented by router decision) and by periodically auditing a sample of no-retrieval-routed queries against the retrieval corpus offline.

**Trap:** treating the router as a one-time classifier that ships and stays fixed. Query distribution shifts as the product evolves (new features, new user questions), so router precision decays the same way any classifier does under distribution shift — it needs the same monitoring as the main model.

### Q: How would you evaluate whether a new LLM version (upgrading the base model) is safe to roll out? [Hard]
A: Never ship on vibes or a handful of manual spot-checks. Run the full offline eval set (groundedness, relevance, correctness) against both old and new model on identical inputs — regression on any dimension blocks the rollout even if the average score improves. Shadow the new model in production (serve real traffic, log both outputs, don't show the new model's response to users) to catch distribution-specific failures the offline set doesn't cover. Then canary to a small live traffic percentage, watching online metrics (thumbs-down rate, escalation rate) before full rollout. Critically: re-validate prompts — a prompt tuned against the old model's quirks (verbosity, instruction-following style) may need retuning for the new model, since prompts aren't portable across model versions without verification.

**Cross-questions to expect:**

- *Offline eval scores improve across the board, but the canary shows a higher thumbs-down rate. What's the likely explanation?* -> Offline eval sets are a fixed, curated snapshot and can miss shifts in style or format that real users react to even when correctness improves — e.g., the new model is more verbose or restructures answers in a way that reads as less helpful despite being more accurate. Reconcile by expanding the eval set with the specific query patterns showing thumbs-down in the canary, not by trusting whichever number is more convenient.
- *The new model is measurably better on your eval set but costs 3x more per token. How do you decide whether the quality gain justifies it?* -> Translate the quality delta into a business metric the cost can be weighed against — reduced escalation-to-human rate, reduced session abandonment — rather than comparing eval score deltas to dollar cost directly. If the quality gain doesn't move a downstream business metric, it may not be worth the cost regardless of the eval score.

**Trap:** treating "the eval score went up" as sufficient justification alone. Eval sets are proxies; the rollout decision needs an online signal tied to real user behavior before it's trusted at scale.

## Flashcards

**Why does hybrid (dense + sparse) retrieval outperform pure dense retrieval in production RAG?** #flashcard
Dense/embedding retrieval captures semantic similarity but misses exact-match cases like part numbers, error codes, or proper nouns poorly represented in the embedding space; sparse (BM25) retrieval catches exact matches but misses paraphrase. Combining both (e.g., via Reciprocal Rank Fusion) covers both failure modes.

**Why use a cross-encoder reranker after retrieval instead of just retrieving more candidates with the bi-encoder?** #flashcard
Bi-encoders embed query and document independently for speed and indexability, but can't model query-document interaction, capping their precision. Cross-encoders jointly attend over the (query, doc) pair for much higher precision, but can't be precomputed — so they're only run on the small top-k candidate set from the cheap bi-encoder retrieval pass.

**What is "lost-in-the-middle" and why does it matter for context assembly?** #flashcard
LLMs attend more reliably to content at the start and end of a long context window than content buried in the middle. The fix is at the assembly layer — order retrieved chunks by importance rather than raw retrieval rank, not by trying to retune the retriever.

**Why does prompt/prefix caching matter specifically for RAG systems?** #flashcard
The system prompt and tool schemas are identical across nearly every request; caching their KV state avoids recomputing attention over those fixed tokens every query. Since RAG systems often carry a large fixed system prompt (instructions, tool schemas), this is one of the highest-leverage latency and cost optimizations available.

**Why is streaming a latency optimization even when total generation time doesn't change?** #flashcard
Users perceive latency to time-to-first-token, not total completion time. Streaming tokens as they're generated can cut perceived latency by 5-10x versus waiting for the full response, even though the underlying generation time is identical.

**Why can't exact-match be used to evaluate LLM response quality offline?** #flashcard
LLM outputs are non-deterministic in phrasing even when semantically correct, so exact-match against a reference answer produces false negatives constantly. Evaluation needs semantic similarity and/or LLM-as-judge scoring against defined quality properties (groundedness, relevance, correctness), not string matching.

**Why does groundedness verification need to check individual claims, not the whole response at once?** #flashcard
A response can be mostly grounded with one fabricated detail mixed in — checking the whole response for topical relevance to the context misses this, since the fabricated claim is often topically adjacent even when unsupported. Per-claim entailment checking (NLI-style) catches partial hallucination that whole-response checks miss.

**Why is misrouting a query to "no retrieval needed" more dangerous than misrouting it to "retrieval needed" when it didn't need it?** #flashcard
Unnecessary retrieval wastes latency and cost but the model can usually still answer correctly from general knowledge. Skipping necessary retrieval produces a confident, ungrounded answer on a domain-specific question — a silent hallucination that won't surface as an error or timeout, only as a wrong answer the user may not catch.

**Why must a new LLM version be shadow-tested and canaried, not just evaluated offline, before full rollout?** #flashcard
Offline eval sets are fixed, curated snapshots that can miss shifts in response style, verbosity, or formatting that real users react to even when correctness improves on the eval set. Shadowing and canarying expose the new model to live traffic distribution and surface online-metric regressions (thumbs-down rate, escalation rate) that offline eval doesn't capture.

---

## Connections

This file covers the *LLM-product application* layer — retrieval, context assembly, grounding, and product-facing evaluation. For the underlying *model architecture and inference internals* (attention, KV cache, decoding strategies), see [10-llms/03-architecture-deep-dive.md](../../10-llms/03-architecture-deep-dive.md). For the *serving/infra* side (batching, tensor parallelism, PagedAttention), see [12-systems-and-scale/03-llm-serving.md](../../12-systems-and-scale/03-llm-serving.md). For RAG-specific application patterns beyond this case study, see [11-llm-applications/02-rag.md](../../11-llm-applications/02-rag.md).
