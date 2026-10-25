---
module: AI Engineering
topic: Flashcards
subtopic: ""
status: unread
tags: [ai-engineering, flashcards, active-recall]
---
# AI Engineering — Flashcards

Companion to the [revision sheet](REVISION.md). Each card is one fact.

## Foundations

**What is the AI engineering escalation ladder?** #flashcard
Prompting → few-shot → RAG → tools/agents → fine-tuning. Climb a rung only when evals show the current one falls short.

**What are the three layers of the AI stack?** #flashcard
Application (prompts, RAG, agents, evals, UX), model (training, fine-tuning, inference optimization), infrastructure (compute, serving, data, monitoring).

**How does temperature differ from top_p?** #flashcard
Temperature rescales logits (sharper or flatter distribution). top_p samples from the smallest token set whose cumulative probability reaches p. Tune one, not both.

**What are the most reliable ways to get structured output?** #flashcard
Native JSON-schema structured outputs (constrained decoding), then forced tool calls, then prompt-for-JSON plus validate and retry.

**What is the difference between prefill and decode?** #flashcard
Prefill processes the whole prompt in parallel (compute-bound, sets TTFT). Decode generates one token at a time (memory-bandwidth-bound, sets TPOT).

**What does the KV cache store, and why does it matter?** #flashcard
Keys and values of past tokens for every layer, so decode doesn't recompute attention over the prefix. Its memory grows with sequence length × batch and limits concurrency.

**What are the stages of the LLM training pipeline?** #flashcard
Pretraining (next-token prediction) → SFT (instruction following) → preference tuning (RLHF/DPO) → RL on verifiable rewards (reasoning).

**Why do LLMs struggle with spelling and character counting?** #flashcard
They see subword tokens, not characters.

## Prompting and Context

**What is context engineering?** #flashcard
Deciding what enters the context window on each call (instructions, retrieved docs, memory, tool results, compacted history) under a token budget.

**Where should long documents go relative to the question?** #flashcard
Documents first, question and instructions last. This improves quality on long contexts.

**What is indirect prompt injection?** #flashcard
Instructions hidden in content the model reads (web pages, documents, emails, tool outputs, images) that hijack its behavior.

**Can prompt injection be fully prevented?** #flashcard
No. Defend in layers: separate data from instructions, least-privilege tools, output and egress filtering, human approval for risky actions, dual-LLM/CaMeL patterns.

**What does DSPy optimize?** #flashcard
The prompts and few-shot demos of declared modules and signatures, against a metric on a training set (e.g., MIPROv2, BootstrapFewShot).

## Embeddings and Retrieval

**HNSW vs. IVF-PQ?** #flashcard
HNSW is a graph index with high recall and low latency but is RAM-heavy. IVF-PQ clusters and compresses vectors, trading some recall for memory at very large scale.

**What is Reciprocal Rank Fusion (RRF)?** #flashcard
score = Σ 1/(k + rank_i) across result lists (k≈60). It merges BM25 and dense results without score calibration.

**Why does hybrid search beat pure dense retrieval?** #flashcard
BM25 catches exact terms (IDs, names, codes) that dense embeddings blur. Dense catches paraphrases that keywords miss.

**What are Matryoshka embeddings?** #flashcard
Embeddings trained so that prefixes (e.g., the first 256 of 1024 dims) still work, so you can truncate for cheaper storage and search.

**What does a re-ranker do in RAG?** #flashcard
A cross-encoder scores (query, chunk) pairs jointly for the top-k candidates, improving precision before generation.

**What is HyDE?** #flashcard
Generate a hypothetical answer, embed it, and retrieve with that embedding instead of the raw query.

**Which metrics evaluate retrieval and which evaluate generation in RAG?** #flashcard
Retrieval: recall@k, precision@k, MRR, nDCG, context precision and recall. Generation: faithfulness/groundedness, answer relevance, correctness.

**When is GraphRAG worth it?** #flashcard
For global or multi-hop questions over a corpus ("main themes across all reports"), where entity and community summaries beat chunk retrieval. It's costly to build and update.

**What makes RAG agentic?** #flashcard
The model decides whether, what, and where to retrieve, iterates on queries, and checks sufficiency before answering.

**What is contextual retrieval?** #flashcard
Prepend an LLM-generated description of the chunk's place in its document before embedding and BM25 indexing. This reduces retrieval failures from context-less chunks.

## Tools, MCP, Agents

**Who executes a tool call: the model or the app?** #flashcard
The app. The model only emits the tool name and arguments; your code validates, executes, and returns the result.

**What makes a good tool description?** #flashcard
It says what the tool does, when to use it and when not to, the argument semantics and formats, and the return shape. Use enums and required fields.

**What are the three MCP server primitives?** #flashcard
Tools (model-invoked actions), resources (app-selected context data), prompts (user-selected templates).

**What are the two MCP transports?** #flashcard
stdio for local subprocesses and Streamable HTTP for remote servers (with OAuth 2.1 for auth).

**What are MCP tool poisoning and rug pulls?** #flashcard
Malicious instructions hidden in tool descriptions, or a server changing tool definitions after approval. Pin versions, review descriptions, and alert on changes.

**MCP vs. A2A?** #flashcard
MCP connects an agent to tools and data. A2A connects agents to other agents (Agent Cards, tasks with lifecycle states).

**When should you build a workflow instead of an agent?** #flashcard
When the steps are known in advance. Workflows are cheaper, faster, and more predictable. Use agents for open-ended tasks with unpredictable paths.

**What are Anthropic's five workflow patterns?** #flashcard
Prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer.

**What are the four types of agent memory?** #flashcard
Working (context window), episodic (past interactions), semantic (facts), procedural (skills and instructions).

**What is pass^k?** #flashcard
The probability that an agent succeeds on all k trials of a task. It measures consistency, unlike pass@k.

**What is the main cost of multi-agent systems?** #flashcard
Many times more tokens, plus coordination failures (lost context, conflicting work). They're worth it only for parallelizable, high-value breadth tasks.

## Architecture and Production

**What does an AI gateway do?** #flashcard
Unified API across providers, routing and fallbacks, key management, per-tenant budgets and rate limits, caching, logging and tracing, guardrails.

**What are the three levels of LLM caching?** #flashcard
Exact-match (hash of the request), semantic (embedding similarity above a threshold), and provider prompt/prefix caching (a stable prefix reuses the KV cache).

**How should you structure a prompt to maximize prompt-cache hits?** #flashcard
Static content first (system prompt, tools, documents), variable content (user message) last.

**Why rate-limit on tokens instead of requests?** #flashcard
Request cost varies by orders of magnitude, and provider limits are in TPM. A token bucket per tenant protects shared capacity.

**What is an eval gate in CI/CD?** #flashcard
An automated golden-set evaluation that must meet thresholds before a prompt, model, or retrieval change ships.

**What should a production LLM trace contain?** #flashcard
Spans for retrieval, each LLM call (model, prompt version, tokens, cost, latency), and tool calls, plus the final output and any user feedback.

**What are the first three moves in a provider outage?** #flashcard
Fail over to a backup provider or model via the gateway, degrade gracefully (cached or simpler responses), and communicate status. Then review circuit-breaker thresholds.

## Optimization

**Why is output more expensive than input?** #flashcard
Output tokens are generated sequentially (decode) while input is processed in parallel. Providers price output roughly 3–5× higher.

**What is a model cascade?** #flashcard
Try a cheap model first and escalate to a stronger one when confidence or validation fails.

**How much do batch APIs save?** #flashcard
About 50%, in exchange for asynchronous results (up to 24 h).

**What are the main latency levers for LLM apps?** #flashcard
Streaming, fewer output tokens, smaller models, prompt caching, parallel calls, co-location. When self-hosting: speculative decoding and quantization.

**What do vLLM's PagedAttention and continuous batching do?** #flashcard
PagedAttention stores the KV cache in non-contiguous blocks (no fragmentation). Continuous batching adds and removes requests every step to keep the GPU busy.

## Agentic and Multimodal

**What is the lethal trifecta?** #flashcard
Access to private data + exposure to untrusted content + a way to communicate externally. Combining all three enables data exfiltration via injection.

**Which isolation tiers suit agent code execution?** #flashcard
Containers (weakest), gVisor (user-space kernel), Firecracker/Kata microVMs (strongest practical), with egress allowlists and no secrets inside.

**How do long-running agents survive crashes?** #flashcard
Durable execution: checkpoint state after each step, idempotency keys for side effects, and resume from the last checkpoint (Temporal, LangGraph checkpointers).

**What drives the cost of a VLM call?** #flashcard
Image resolution and count, which convert to visual tokens. Crop to the region and choose the detail level per task.

**What should you never trust a VLM to do without verification?** #flashcard
Count objects, do math on extracted numbers, reproduce small text exactly, or give precise spatial positions.

**What is the biggest latency lever in a voice agent?** #flashcard
End-of-turn detection (VAD/semantic endpointing), followed by streaming every stage.

**What is barge-in handling?** #flashcard
When the user interrupts, stop TTS, cancel generation, and truncate the assistant turn in history to what was actually spoken.

**What is the main cost lever for video understanding?** #flashcard
Frame sampling: fps, scene-change keyframes, or transcript-first with frames on demand.

**What is ColPali?** #flashcard
A late-interaction retriever that embeds PDF page images directly (multi-vector), so documents can be retrieved without OCR.

## Evaluation

**What is eval-driven development?** #flashcard
Defining success criteria and an eval set before changing the system, then gating every prompt, model, or retrieval change on eval results.

**How big should a first eval set be?** #flashcard
50–100 real examples to start; 200–1,000 to make decisions between close variants.

**What is error analysis?** #flashcard
Reading traces, labelling failure types, clustering and counting them, then fixing the most frequent or costly category first.

**Which graders should you prefer, in order?** #flashcard
Code checks (exact match, schema, execution) → LLM-as-judge → human review. Use the cheapest grader that measures the criterion reliably.

**What are the main biases of LLM judges?** #flashcard
Position bias, verbosity bias, and self-preference. Mitigate with position swaps, binary/rubric criteria, and calibration against human labels.

**How do you know you can trust an LLM judge?** #flashcard
Measure agreement with human labels on a sample (Cohen's κ ≥ ~0.6–0.7, or TPR/TNR) and re-check after prompt or model changes.

**Why aren't offline evals enough?** #flashcard
Real traffic differs from the eval set; confirm wins with shadow, canary, and A/B tests and track guardrail metrics (latency, cost, safety).

## Fine-Tuning

**When should you fine-tune?** #flashcard
When prompting and RAG have plateaued on evals and the gap is behaviour (format, style, skill) or cost/latency, not missing knowledge.

**Why is fine-tuning a poor way to add knowledge?** #flashcard
Models learn new facts slowly from SFT and hallucinate more on them; RAG adds and updates facts cheaply with citations.

**What does LoRA do?** #flashcard
Freezes the base weights and trains low-rank matrices A·B added to selected layers, so ~0.1–1% of parameters are trained and adapters are swappable.

**What is QLoRA?** #flashcard
LoRA on top of a 4-bit (NF4) quantized frozen base, letting large models be fine-tuned on a single GPU.

**DPO vs. RLHF?** #flashcard
DPO optimizes directly on chosen/rejected pairs with a classification-style loss; RLHF trains a reward model and then optimizes with RL (PPO). DPO is simpler and more stable.

**What is RL with verifiable rewards?** #flashcard
RL (e.g., GRPO) where the reward comes from a checker such as unit tests or exact math answers; it is how reasoning models are trained.

**What is distillation in AI engineering?** #flashcard
Collecting a large teacher model's outputs on real traffic and fine-tuning a smaller student to match them, cutting cost and latency.

**What is multi-LoRA serving?** #flashcard
Serving many fine-tuned adapters on one shared base model, swapping adapters per request (e.g., in vLLM or LoRAX).

## Safety and Governance

**Where do guardrails go?** #flashcard
Input (injection, PII, topic), retrieval (ACL, injection scan), tool calls (authz in code, approvals, limits), and output (PII, policy, grounding, schema).

**What is the lethal trifecta?** #flashcard
An agent with private data, exposure to untrusted content, and a way to send data out. Removing any one leg blocks exfiltration.

**Name three hallucination mitigations.** #flashcard
Grounding with retrieval plus cite-or-abstain, claim verification with NLI or a judge, and sampling-consistency signals such as semantic entropy.

**Why report abstention rate with hallucination rate?** #flashcard
A system can cut hallucinations by refusing everything; both together show the real trade-off.

**What does a red-team program measure?** #flashcard
Attack success rate (ASR) per attack category, plus the over-refusal rate on benign inputs; successful attacks become regression tests.

**What frameworks shape AI compliance?** #flashcard
EU AI Act (risk tiers), NIST AI RMF, ISO/IEC 42001, plus data-protection law (GDPR) for retention, residency, and user rights.

## System Design Cases

**What are the steps of an AI system design interview?** #flashcard
Clarify → estimate (QPS, tokens, cost, latency) → high-level design → deep dive → evaluation → failure modes and guardrails → cost and iteration.

**How do you enforce permissions in enterprise RAG?** #flashcard
Store ACL metadata per chunk and filter inside the vector query using the user's identity from the session, never after generation.

**How does a document-extraction pipeline reach 99% accuracy on posted records?** #flashcard
Schema extraction + deterministic validation (totals, master data, duplicates) + confidence routing: confident docs auto-post, the rest go to human review.

**What is the latency target for a voice agent and how is it met?** #flashcard
~800 ms voice-to-voice: stream every stage (ASR, LLM, TTS), use semantic turn detection, a fast model with prompt caching, and fillers during tool calls.

**How does a coding assistant keep autocomplete fast?** #flashcard
A small self-hosted FIM model, tight context (prefix/suffix + nearby symbols), caching, and cancelling stale requests, targeting p50 < 200 ms.
