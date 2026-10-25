---
module: AI Engineering
topic: Cheatsheet
subtopic: ""
status: unread
tags: [ai-engineering, cheatsheet, numbers, defaults, decision-tables]
---
# AI Engineering — Cheatsheet

Numbers, defaults, and decision tables on one page. For the facts per section see the [revision sheet](REVISION.md); for decision flows see [flowcharts](FLOWCHARTS.md); for terms see the [glossary](GLOSSARY.md).

## 1. Rules of Thumb (numbers)
| Quantity | Value | Use it for |
|---|---|---|
| Tokens per English word | ~1.3 (≈ 4 chars/token) | Cost and context estimates |
| Tokens per page of prose | ~500–700 | Document ingestion sizing |
| Output vs. input price | Output 3–5× input per token | Shorten outputs before inputs |
| Prompt cache hit discount | ~50–90% off cached input | Put stable content first |
| Batch API discount | ~50% | Offline/async workloads |
| Human conversational gap | ~300–500 ms | Voice agent latency target (< 800 ms voice-to-voice) |
| Autocomplete latency | p50 < 200 ms | Small self-hosted model |
| Chat time-to-first-token | < 1–1.5 s | Streaming UX |
| Embedding storage | dims × 4 bytes (float32) | 1M × 1024-d ≈ 4 GB; int8 ÷4, binary ÷32 |
| KV cache per token | 2 × layers × kv_heads × head_dim × bytes | Concurrency and context limits when self-hosting |
| Model weights memory | params × bytes (fp16 = 2, int8 = 1, int4 = 0.5) | 70B fp16 ≈ 140 GB; int4 ≈ 35–40 GB |
| LoRA trainable params | ~0.1–1% of the model | Fine-tuning cost |
| Eval set size | 50–100 to start, 200–1,000 for decisions | Detecting ~5-point differences needs hundreds |
| Judge–human agreement | Target Cohen's κ ≥ 0.6–0.7 | Trusting LLM-as-judge |

## 2. Cost Formula
```
cost/request = in_tokens × p_in + out_tokens × p_out  − cached_in × p_in × cache_discount
monthly      = cost/request × requests/day × 30
agent run    ≈ Σ steps (context grows each step → cost ~ quadratic in steps without caching/compaction)
```
Levers in order of effort: shorter outputs → prompt caching → smaller model / routing → batch API → semantic cache → fine-tune/distill → self-host ([costs](13-llm-optimization/01-understanding-ai-application-costs.md)).

## 3. Sampling Defaults
| Task | temperature | Other |
|---|---|---|
| Extraction, classification, code | 0–0.2 | Structured outputs / JSON schema |
| Q&A, RAG answers | 0.2–0.5 | Cite sources |
| Creative writing, brainstorming | 0.7–1.0 | top_p 0.9–0.95 |
| Self-consistency voting | 0.7 × N samples | Majority vote |
| Reasoning models | Provider default | Control with reasoning effort/budget, not temperature |

## 4. Chunking and Retrieval Defaults
| Knob | Default | Tune when |
|---|---|---|
| Chunk size | 300–800 tokens, structure-aware | Answers span chunks → larger or parent-doc retrieval |
| Overlap | 10–20% | Boundary losses |
| Contextual header | Title + breadcrumb prepended | Always cheap win |
| Retrieve k | 20–50 (hybrid), rerank to 5–10 | Recall@k low → raise k / add BM25 |
| Fusion | Reciprocal Rank Fusion (k=60) | Mixing dense + sparse |
| Reranker | Cross-encoder | Precision low at top ranks |
| Similarity | Cosine on normalized vectors | — |
| Index | HNSW (M 16–32, ef_search 64–256) | Recall vs. latency trade-off |

## 5. Technique Selection
| Symptom | Reach for |
|---|---|
| Model doesn't know private/fresh facts | RAG |
| Model knows but formats/behaves wrong | Prompting → few-shot → fine-tune (SFT) |
| Needs to act or fetch live data | Tool use / function calling |
| Multi-step with known steps | Workflow (chain, router, parallel) |
| Open-ended, steps unknown | Agent loop with budgets |
| Too slow/expensive at volume | Routing, caching, distillation, smaller model |
| Tone/preference alignment | DPO / preference tuning |
| Need machine-readable output | Structured outputs (JSON schema, constrained decoding) |

## 6. RAG Failure → Fix
| Symptom | Likely cause | Fix |
|---|---|---|
| Right doc never retrieved | Embedding mismatch, jargon, IDs | Hybrid (BM25), query rewrite, HyDE, domain embeddings |
| Retrieved but ranked low | Weak ranking | Cross-encoder rerank |
| Retrieved, answer ignores it | Lost in the middle, prompt | Fewer/better chunks, put best first/last, instructions |
| Answer adds unsupported facts | Hallucination | Cite-or-abstain prompt, citation verification, NLI check |
| Partial answers | Chunk too small | Parent-document / sentence-window retrieval |
| Stale answers | Index lag | Event-driven ingestion, recency boost |
| Multi-hop questions fail | Single retrieval | Query decomposition, agentic RAG, GraphRAG |

## 7. Agent Design Defaults
- Start with a **workflow**; use an agent only when the path can't be predetermined.
- Max steps (8–20), token budget, wall-clock timeout, per-tool timeouts.
- Tools: ≤ ~10–20 per agent, clear names, typed args, errors returned as data, idempotency keys on writes.
- Authorization in code: identity comes from the session, never from model arguments.
- HITL on irreversible or high-value actions; read-back confirmation in voice.
- Trace every step (inputs, tool calls, outputs, cost, latency).

## 8. Evaluation Toolkit
| Need | Method |
|---|---|
| Deterministic correctness | Exact match, regex, schema validation, unit tests, execution |
| Retrieval quality | Recall@k, MRR, nDCG |
| RAG answer quality | Faithfulness, answer relevance, context precision/recall (RAGAS-style) |
| Open-ended quality | LLM-as-judge (pairwise or rubric, with position swap) calibrated to humans |
| Agent quality | Task success on end state, trajectory checks, steps/cost per task |
| Safety | Attack success rate (ASR) + over-refusal rate |
| Online | A/B with guardrail metrics, thumbs, retention, task completion, cost/latency |

## 9. Latency Levers
| Component | Lever |
|---|---|
| Time-to-first-token | Shorter prompt, prompt caching, smaller/faster model, region colocation |
| Tokens/sec | Smaller model, speculative decoding, quantization, better hardware |
| Total | Streaming, parallel calls, fewer round-trips, shorter outputs, early exit |
| Perceived | Streaming UI, progress/status, optimistic UI, filler in voice |

## 10. Fine-Tuning Quick Reference
| Method | Data | When |
|---|---|---|
| SFT (full) | 1k–100k+ input→output pairs | Max quality, have GPUs |
| LoRA | Same, rank 8–64, alpha ≈ 2×rank | Default PEFT |
| QLoRA | Same; 4-bit base | Single GPU, large model |
| DPO | Chosen/rejected pairs (1k–10k) | Preferences, style, refusals |
| RL (GRPO/PPO) with verifiers | Prompts + reward function | Verifiable tasks (math, code) |
| Distillation | Teacher outputs on real traffic | Cut cost/latency of a working prompt |

LR: full FT ~1e-5; LoRA ~1e-4–2e-4. 1–3 epochs. Always hold out an eval split and compare against the prompted baseline.

## 11. Production Checklist
- [ ] Eval set + CI regression gate before every prompt/model change
- [ ] Pinned model versions; prompts versioned in code
- [ ] AI gateway: retries with backoff, fallback provider, timeouts, rate limits, per-tenant budgets
- [ ] Input/output rails; PII redaction in logs; retention policy
- [ ] Tracing (OpenTelemetry GenAI conventions), cost and latency dashboards, alerts
- [ ] Feature flags / kill switch; shadow → canary → full rollout
- [ ] Feedback capture (thumbs, edits) feeding the eval set
- [ ] Red-team pass (OWASP LLM Top 10) before launch

## 12. Security Quick Reference
| Threat | Control |
|---|---|
| Direct prompt injection | Instruction hierarchy, input classifier, least privilege |
| Indirect injection (docs, web, email) | Treat retrieved content as data, no write tools in same context, spotlighting |
| Lethal trifecta (private data + untrusted input + exfiltration) | Remove one leg: no external egress, or no private data, or HITL |
| Data leakage across users | ACL-filtered retrieval, per-tenant caches, scoped tools |
| Excessive agency | Scoped tools, approvals, sandboxes, budgets |
| Unbounded consumption | Rate limits, token caps, step budgets |

## Related
- [Index](README.md) · [Revision](REVISION.md) · [Flashcards](FLASHCARDS.md) · [Flowcharts](FLOWCHARTS.md) · [Glossary](GLOSSARY.md)
