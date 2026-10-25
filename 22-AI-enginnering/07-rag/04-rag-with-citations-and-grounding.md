---
module: AI Engineering
topic: RAG and Retrieval
subtopic: RAG with Citations and Grounding
status: unread
source: original (lesson paywalled; written from the course outline)
tags: [citations, grounding, attribution, faithfulness, nli, abstention, confidence, hallucination]
---
# RAG with Citations and Grounding

```mermaid
flowchart LR
    R[Reranked chunks<br/>with ids] --> G[Generate answer<br/>claims + citation ids]
    G --> V1[Validate ids exist]
    V1 --> V2[Claim-level support check<br/>NLI / LLM judge]
    V2 --> C{Supported?}
    C -->|all| OUT[Answer + sources]
    C -->|some| FIX[Drop / hedge / regenerate<br/>unsupported claims]
    C -->|none or weak evidence| ABS[Abstain:<br/>"not in the docs"]
    FIX --> OUT
```

Retrieval alone doesn't make RAG trustworthy. Users need to know **which evidence supports which claim**, what is uncertain, and when there isn't enough evidence to answer.
- **Citation**: a link from a claim to a specific source unit (passage, page, table row, timestamp, record).
- **Grounding / verification**: checking that the cited source **actually supports** the claim.

Citations *point to* evidence. Verification checks that the evidence *supports* the claim. Grounding reduces unsupported answers but never fully eliminates them.

## 1. Citation Failure Modes
| Failure | Example | Detection |
|---|---|---|
| Fabricated citation | `[doc-17]` not in the retrieved set | Id validation (deterministic) |
| Wrong source | Claim is true but cites an unrelated chunk | Claim–source NLI |
| Overstatement | Source: "may reduce latency"; answer: "eliminates latency" | NLI (neutral/contradiction), LLM judge |
| Related but insufficient | Source discusses the feature but not the asked limit | Support check per claim |
| Missed contradiction | Two sources disagree; answer picks one silently | Conflict detection across sources |
| Uncited claim | Factual sentence with no citation | Sentence-level coverage check |
| Stale source | Cites the 2023 policy when a 2025 one exists | Metadata (`updated_at`, superseded flag) |

## 2. The Citation Pipeline
1. **Retrieve and label**: every chunk gets a short stable id (`S1…Sn`) mapped to `{doc_id, chunk_id, url, page, span}`.
2. **Generate with structure**: ask for claims + citation ids as JSON (not free-form brackets).
3. **Validate**: ids exist; each claim has ≥ 1 citation.
4. **Verify support**: per (claim, cited passage) pair.
5. **Decide**: keep, hedge, drop, regenerate, or abstain.
6. **Render**: inline markers → clickable source cards with highlighted spans.

### Structured output
```python
from pydantic import BaseModel, Field

class Claim(BaseModel):
    text: str
    source_ids: list[str] = Field(min_length=1)
    quote: str | None = None            # optional exact supporting span

class GroundedAnswer(BaseModel):
    answerable: bool
    claims: list[Claim]
    caveats: list[str] = []

SYSTEM = """Answer ONLY from the sources. Each claim must cite source ids like S2.
Copy a short exact quote that supports each claim when possible.
If the sources don't answer the question, set answerable=false and return no claims.
If sources conflict, say so in caveats and cite both."""
```
Asking for an **exact quote** makes verification cheap: check that the quote is a substring (after normalization) of the cited chunk.

### Citation granularity
| Granularity | Pros | Cons |
|---|---|---|
| Document | Simple | Users can't find the evidence |
| Chunk / passage | Default | Chunk may contain mixed content |
| Sentence / span (char offsets) | Precise highlighting | Needs span extraction |
| Page (PDF) / timestamp (A/V) / row (tables) | Matches how users verify | Needs layout-aware ingestion |

Provider-native options: Anthropic Citations API (returns cited spans of supplied documents), Gemini grounding metadata, Cohere RAG citations. They reduce prompt engineering but still need verification for high-stakes uses.

## 3. Verifying Support
```python
def verify(ans: GroundedAnswer, sources: dict[str, str]) -> list[dict]:
    results = []
    for c in ans.claims:
        if any(sid not in sources for sid in c.source_ids):
            results.append({"claim": c.text, "status": "invalid_id"}); continue
        if c.quote and not any(norm(c.quote) in norm(sources[s]) for s in c.source_ids):
            results.append({"claim": c.text, "status": "quote_not_found"}); continue
        evidence = "\n".join(sources[s] for s in c.source_ids)
        label = judge(premise=evidence, hypothesis=c.text)   # entailed | neutral | contradicted
        results.append({"claim": c.text, "status": label})
    return results
```
| Verifier | Cost | Notes |
|---|---|---|
| Substring / fuzzy quote match | ~0 | Catches fabricated quotes only |
| NLI cross-encoder (DeBERTa-MNLI, AlignScore, MiniCheck) | Low (ms) | Good claim-level entailment; limited context length |
| LLM judge (small model, strict rubric) | Medium | Handles paraphrase, multi-source synthesis |
| Specialized (HHEM, Lynx, Patronus, Bespoke-MiniCheck) | Low–medium | Trained hallucination detectors |

Decompose long answers into **atomic claims** first (one fact per claim); verification on whole paragraphs is unreliable.

## 4. Confidence Estimation
No single signal is reliable; combine them:
| Signal | Meaning |
|---|---|
| Max / mean rerank score of the top-k | Is there any strong evidence at all? |
| Fraction of claims entailed | Answer-level groundedness |
| Agreement across sources | Multiple independent sources → higher confidence |
| Self-consistency (sample N answers, compare) | Disagreement → uncertainty |
| Model-verbalized confidence | Weak alone; poorly calibrated |
| Token logprobs | Useful for short answers / classification |

Calibrate thresholds on a labeled set: plot the accuracy of answers shown vs the abstention rate and pick an operating point.

## 5. Abstention
Saying "I don't know" is a feature. Abstain when:
- The top rerank score is below a threshold (retrieval found nothing relevant).
- The model returns `answerable=false`.
- After verification, too few claims are supported.
- The question is out of scope (classified before retrieval).

Good abstention UX: say what was searched, show the nearest sources, offer rephrasing or escalation, and **log it as a content gap**.

| Policy | Coverage | Accuracy on answered | Use |
|---|---|---|---|
| Never abstain | 100% | Lowest | Low-stakes chat |
| Threshold on retrieval | ~85–95% | Higher | Default |
| Retrieval + verification gating | Lower | Highest | Legal, medical, finance, policy |

## 6. Handling Unsupported Claims
| Situation | Action |
|---|---|
| One claim unsupported, rest fine | Drop it or add a hedge ("The docs don't state…") |
| Many unsupported | Regenerate with feedback ("Claims X, Y were not supported; use only the sources") |
| Contradicted claim | Remove; surface the conflict |
| Streaming UX | Stream, then attach verification badges; or buffer high-risk answers |

## 7. Conflicting and Stale Sources
- Include `updated_at`, version and authority in the source header; instruct: prefer the newest **authoritative** source and mention conflicts.
- Mark superseded docs at ingest (`superseded_by`) and filter them out of retrieval.
- For policy/legal content, answer **per source** ("Policy v3 (2025) says…; v2 said…").

## 8. Rendering Citations
- Inline numbered markers `[1]` → hover/click shows the snippet with the supporting span highlighted, title, date, and a deep link (`#page=4`, `?t=132s`, anchor).
- Deduplicate sources; order by first mention.
- Never render a link the model invented; only links from the source map.

## 9. Measuring
- **Citation precision**: fraction of citations that support their claim.
- **Citation recall**: fraction of claims that carry a supporting citation.
- **Faithfulness / groundedness**: fraction of entailed claims.
- **Abstention quality**: correct abstentions on unanswerable questions vs false abstentions on answerable ones.
- Benchmarks: ALCE (citation quality), RAGTruth, FRAMES; build an in-domain set with unanswerable questions included.

## Takeaways
- Generate citations with structured output that ties each claim to source IDs (and spans).
- Verify support with NLI or a judge; drop or flag unsupported claims.
- Abstain when evidence is missing; surface conflicting and stale sources rather than picking silently.
- Measure citation precision and recall, and the abstention rate.

## References
- Gao et al., *ALCE: Enabling LLMs to Generate Text with Citations* (2023)
- Tang et al., *MiniCheck* (2024) · Zha et al., *AlignScore* (2023) · Vectara HHEM
- Anthropic Citations API docs

## Related
- [Production RAG pipeline](02-building-a-production-rag-pipeline.md) · [Evaluating RAG](05-evaluating-rag-systems.md)
- [Structured output](../03-first-ai-application/03-structured-output-from-llms.md)
