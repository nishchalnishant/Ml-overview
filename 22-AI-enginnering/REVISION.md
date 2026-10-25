---
module: AI Engineering
topic: Revision
subtopic: ""
status: unread
tags: [ai-engineering, revision, cheatsheet]
---
# AI Engineering — Revision Sheet

One block per section: what to be able to say from memory. Deep dives are in the [index](README.md).

## 01 — Introduction
- AI engineering = building products on **foundation models**: adapt (prompt → RAG → tools → fine-tune), evaluate, ship, operate. ML engineering trains models; AI engineering composes them.
- Three layers: **application** (prompts, RAG, agents, UX, evals) → **model** (training, fine-tuning, inference optimization) → **infrastructure** (GPUs, serving, data, monitoring).
- Escalation ladder: prompt engineering → few-shot → RAG → tool use/agents → fine-tuning. Climb only when evals show you need to.

## 02 — Python for AI
- Pydantic models are the contract for LLM I/O: validate, coerce, and generate JSON schema.
- Generators/streaming for tokens; decorators for retries, timing, caching.
- `asyncio.gather` + a `Semaphore` for concurrent LLM calls under rate limits; `tenacity` for backoff with jitter.

## 03 — First AI Application
- Parameters: `temperature` (randomness), `top_p` (nucleus), `max_tokens` (output cap and cost), `stop`, `seed`. Use low temperature for extraction and higher for ideation.
- Structured output: native JSON schema / `responses.parse` > tool-call forcing > "reply in JSON" + validate + retry.
- Streaming cuts **perceived** latency (TTFT). Conversation state = you resend history, so trim/summarize to fit the context and the budget.

## 04 — How LLMs Work
- Tokens ≈ ¾ of an English word; BPE/subword. Cost, limits, and odd failures (spelling, arithmetic) are token-level.
- Transformer = embeddings + positional info (RoPE) + stacked **self-attention** + MLP blocks; decoder-only, causal mask.
- Generation is autoregressive: logits → sampling. **Prefill** (parallel, compute-bound) vs. **decode** (sequential, memory-bound); the KV cache stores past keys/values.
- Training pipeline: pretraining (next-token) → SFT (instructions) → preference tuning (RLHF/DPO) → RL on verifiable rewards (reasoning).

## 05 — Prompt Engineering
- Anatomy: role/context → task → constraints → examples → output format. Be explicit; put long documents first and the question last.
- Techniques: few-shot, chain-of-thought (or reasoning models), decomposition/prompt chaining, self-consistency, reflection.
- **Context engineering** = choosing what goes into the window (retrieval, memory, tool results, compaction). The window is a budget.
- Templates are versioned artifacts with tests. Injection defense is layered: separate data from instructions, least-privilege tools, output filtering, human approval. There's no complete fix.
- DSPy: declare signatures and modules, then let an optimizer tune prompts and demos against a metric.

## 06 — Embeddings and Vector Search
- Embedding = a vector where semantic similarity is geometric closeness (cosine). Pick a model by MTEB **on your domain**, dimension and cost, max tokens, and multilingual support. Matryoshka embeddings allow truncation.
- ANN indexes: **HNSW** (fast, RAM-heavy), IVF(-PQ) (compressed, large scale), DiskANN. Recall/latency tuned via `ef_search` / `nprobe`.
- Hybrid search (BM25 + dense, fused with RRF) beats either alone. Scale with sharding, quantization (int8/binary), metadata filtering, and re-embedding plans.

## 07 — RAG
- Pipeline: ingest → parse → **chunk** (structure-aware, 200–800 tokens, overlap) → embed → index; query → (rewrite) → hybrid retrieve → **re-rank** → assemble → generate with citations.
- Advanced: HyDE, multi-query, parent-document/small-to-big, contextual retrieval, metadata filters.
- Evaluate retrieval (recall@k, MRR, nDCG) **separately** from generation (faithfulness, answer relevance). Use RAGAS-style LLM judges calibrated against humans.
- Variants: conversational (condense the question with history), GraphRAG (entities and communities for global questions), multimodal (page images, ColPali), agentic (the model decides when and what to retrieve).

## 08 — Tool Use
- The model **proposes** a call (name + JSON args); your code executes it and returns the result; the loop runs until a final answer.
- Schemas: clear names, descriptions that say *when* to use the tool, enums, required fields, few tools per turn. Errors returned as content let the model recover.
- Parallel calls for independent tools; deterministic workflows (code-driven) when the steps are known.

## 09 — MCP
- Client–server protocol (JSON-RPC) that standardizes **tools, resources, and prompts**; it solves the M×N integration problem.
- Transports: stdio (local) and **Streamable HTTP** (remote) with OAuth 2.1.
- Production: auth per user, scoped tools, rate limits, audit logs, treat tool output as untrusted, beware tool poisoning and rug pulls.

## 10 — Agents
- Agent = LLM + tools + loop + memory, with autonomy over the control flow. Prefer **workflows** (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer) until open-endedness requires an agent.
- Memory: working (context), episodic, semantic (facts), procedural; plus write, retrieve, and forget policies.
- Multi-agent pays off for parallelizable breadth; it costs tokens and coordination failures.
- Reliability: step and budget limits, structured outputs, retries, checkpoints, tracing. Evaluate on trajectories and outcomes, with pass^k for consistency.
- Human-in-the-loop: approve risky actions, escalate on low confidence, review queues.

## 11 — AI System Architecture
- Patterns: single call → chain → RAG → agent. Put an **AI gateway** in front of providers (routing, fallbacks, keys, budgets, caching, logging, guardrails).
- Reliability: timeouts, retries with jitter, circuit breakers, provider fallback, graceful degradation, idempotency.
- Caching: exact (hash) → **semantic** (embedding threshold) → **prompt/prefix caching** (provider-side, stable prefix first).
- Rate limit on **tokens**, not just requests, per tenant (token bucket). Queue async/batch work.
- UX: stream, show sources, allow editing and undo, show uncertainty.

## 12 — Production Deployment
- Containerize with pinned dependencies, slim images, health checks, and secrets from a vault. Separate GPU images for self-hosted models.
- CI/CD adds **eval gates**: a golden set with regression thresholds on every prompt or model change. Treat prompts and model versions as deployable config.
- Observability: traces per request (spans for retrieval, LLM, tools), tokens, cost, latency (TTFT/TPOT), feedback, quality sampling with LLM judges (OpenTelemetry GenAI conventions, Langfuse/LangSmith/Phoenix).
- Incidents: provider outage → fallback; quality regression → roll back the prompt or model; cost spike → budgets and alerts. Run blameless postmortems.
- Iterate: shadow → canary → A/B; feedback loops into evals. The lifecycle runs scope → prototype → eval → harden → ship → monitor → improve.

## 13 — LLM Optimization
- Cost = input tokens × price + output tokens × price (output costs 3–5× more). Levers: shorter prompts, caching, smaller models, batch APIs (~50% off), output limits.
- **Routing/cascades**: a cheap model first, escalating on low confidence or hard classes. Validate the router on evals.
- Latency: stream, cut output tokens, parallelize calls, prompt caching, smaller or faster models, speculative decoding (when self-hosted), region co-location.
- Self-host when volume, privacy, or customization justify it: vLLM/SGLang, quantization (AWQ/GPTQ/FP8), continuous batching, paged attention. Count the ops cost.

## 14 — Agentic AI
- Computer use: screenshot → action loop. DOM/accessibility tree when available, pixels when not. Sandbox the browser and gate irreversible actions.
- Coding agents: an edit-run-observe loop. Tests are the reward signal; context comes from repo maps and search; work in small diffs. Evaluated with SWE-bench and Terminal-Bench.
- Deep research: plan → parallel search → read → evidence ledger → synthesize with citations and confidence. Contradictions are surfaced, not hidden.
- Protocols: **MCP** = agent↔tools, **A2A** = agent↔agent (Agent Cards, tasks with states).
- Long-running: decouple from the request; durable execution (Temporal, LangGraph checkpoints), idempotency keys, compaction, progress events, budgets.
- Security: the **lethal trifecta** (private data + untrusted content + exfiltration channel). Mitigate with least privilege, sandboxes (gVisor/Firecracker), egress allowlists, taint tracking, approvals, and a kill switch.

## 15 — Multimodal
- VLM = vision encoder → projector → LLM. Image resolution becomes tokens, which become cost. Crop, pick the detail level, validate counts and totals in code.
- Image generation: latent diffusion, CFG, DiT + flow matching. Use versioned style presets, composite real product photos, moderate, and add C2PA provenance.
- Speech: batch for accuracy, streaming for interactivity. Measure WER plus entity accuracy. Voice agent latency budget is under ~800 ms, and turn detection is the biggest lever. Handle barge-in.
- Video = frames + audio + time. Frame sampling is the cost lever; answer with timestamps.
- Multimodal apps: normalize every input into evidence with references; shared embedding spaces (CLIP/SigLIP/ColPali); route by modality and precision.

## 16 — Evaluation
- **Eval-driven development**: write success criteria and an eval set before changing anything; every prompt, model, or retrieval change is gated on evals in CI.
- Datasets: start with 50–100 real examples (golden + edge + adversarial), grow from production failures. **Error analysis** (read traces, cluster failures, count) decides what to fix next.
- Graders, cheapest first: code checks (exact match, regex, schema, execution) → LLM-as-judge → humans.
- **LLM-as-judge**: binary or rubric criteria, pairwise with position swap, calibrate against human labels (κ ≥ 0.6). Watch for position, verbosity, and self-preference bias.
- Online: A/B tests with guardrail metrics, implicit feedback (edits, retries, abandonment), shadow and canary rollouts. Offline wins must be confirmed online.

## 17 — Fine-Tuning
- Fine-tune for **behaviour** (format, style, skill, cost/latency), not knowledge. Try prompting and RAG first; compare against the prompted baseline.
- Data quality beats quantity: ~500–5k clean, diverse, deduplicated examples. Decontaminate against the eval set. Synthesize with a teacher, then filter.
- **LoRA**: train low-rank adapters (~0.1–1% of params; rank 8–64, LR ~1e-4). **QLoRA**: LoRA over a 4-bit base for a single GPU.
- **DPO** on chosen/rejected pairs for preferences; **RL with verifiable rewards** (GRPO) for checkable tasks like math and code.
- **Distillation**: teacher outputs on real traffic → smaller student. Serve many adapters on one base (multi-LoRA). Pin versions and watch for regressions.

## 18 — Safety and Governance
- Guardrails at four points: **input** (injection, PII, topic), **retrieval** (ACL, injection scan), **tool** (authz in code, approvals, limits), **output** (PII, policy, grounding, schema).
- Hallucination: ground with retrieval, cite or abstain, verify claims (NLI/judge), use sampling-consistency signals (semantic entropy). Report hallucination rate together with abstention rate.
- Red teaming: threat model → attack library (OWASP LLM Top 10, indirect injection, jailbreaks) → automated + manual attacks → measure ASR and over-refusal → regression suite.
- **Lethal trifecta** (private data + untrusted content + exfiltration): remove one leg.
- Compliance: EU AI Act risk tiers, NIST AI RMF, data retention and residency, model/system cards, audit logs, human oversight for high-risk use.

## 19 — AI System Design Cases
- Interview flow: clarify → estimate (QPS, tokens, cost, latency) → high-level design → deep dive on the hardest part → evaluation → failure modes → cost and iteration.
- Support agent: retrieval over KB + scoped tools with authz from the session, HITL for refunds, deflection and CSAT metrics, escalation with a summary.
- Enterprise RAG: ACL filtering inside the query, incremental connectors, hybrid + rerank, citations, per-tenant isolation.
- Coding assistant: FIM autocomplete under ~200 ms on a small model; repo context via retrieval and symbols; agent mode in a sandbox with tests as the verifier.
- Document extraction: schema + validation rules + confidence → auto-post or human review; corrections become training data; batch APIs.
- Voice agent: cascaded streaming under ~800 ms, turn detection, barge-in, read-back confirmation, identity verification in code.
