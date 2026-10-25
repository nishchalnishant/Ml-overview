---
module: AI Engineering
topic: Glossary
subtopic: ""
status: unread
tags: [ai-engineering, glossary, terminology]
---
# AI Engineering — Glossary

One-line definitions, alphabetical. The section number points to the lesson folder with the detail.

| Term | Definition | § |
|---|---|---|
| **A2A (Agent2Agent)** | Protocol for agents from different vendors to discover each other (agent cards) and exchange tasks | 14 |
| **Abstention rate** | Share of queries where the system declines to answer; report alongside hallucination rate | 18 |
| **ACL filtering** | Restricting retrieval to documents the user may access, applied inside the vector query | 07, 19 |
| **Agent** | LLM in a loop that chooses tools and next steps until a goal or budget is reached | 10 |
| **Agentic RAG** | Retrieval driven by an agent that decides when, what, and how often to retrieve | 07 |
| **AI gateway** | Proxy in front of model providers handling auth, routing, fallback, rate limits, caching, logging | 11 |
| **Attack success rate (ASR)** | Fraction of red-team attacks that achieve their goal | 18 |
| **Barge-in** | User interrupts a voice agent; TTS stops and history is truncated to what was spoken | 15, 19 |
| **BM25** | Classic sparse lexical ranking function; complements dense retrieval for exact terms and IDs | 06, 07 |
| **Chain-of-thought (CoT)** | Prompting the model to reason step by step before answering | 05 |
| **Chunking** | Splitting documents into retrievable units (fixed, recursive, structure-aware, semantic) | 07 |
| **Citation precision** | Share of cited sources that actually support the claim they're attached to | 07 |
| **Constrained decoding** | Restricting next-token choices to a grammar or JSON schema so output is always valid | 03 |
| **Context engineering** | Deciding what goes into the context window: instructions, retrieved data, memory, tool results, history | 05 |
| **Context window** | Maximum tokens (input + output) a model processes in one call | 04 |
| **Continuous batching** | Server-side batching that adds and removes requests per decoding step; core of vLLM-style throughput | 13 |
| **Contextual retrieval** | Prepending chunk-specific context (doc title, summary) before embedding/indexing | 07 |
| **Cross-encoder** | Model scoring (query, document) jointly; slower than bi-encoders, used for reranking | 07 |
| **DPO** | Direct Preference Optimization: fine-tunes on chosen/rejected pairs without a reward model | 17 |
| **Distillation** | Training a smaller student model on a larger teacher's outputs | 17 |
| **DSPy** | Framework that compiles declarative LLM programs and optimizes prompts/few-shot examples against a metric | 05 |
| **Embedding** | Dense vector representing meaning; similar texts lie close together | 06 |
| **Eval-driven development** | Writing evals before changes and gating changes on eval results | 16 |
| **Faithfulness** | Whether an answer is supported by the provided context (groundedness) | 07, 18 |
| **Few-shot prompting** | Including worked examples in the prompt | 05 |
| **FIM (fill-in-the-middle)** | Code-completion format using prefix and suffix around the cursor | 19 |
| **Function calling / tool use** | Model emits structured calls to developer-defined functions; the app executes them | 08 |
| **Guardrails / rails** | Checks around a model call: input, retrieval, tool, output | 18 |
| **GraphRAG** | RAG over an extracted knowledge graph and community summaries for multi-hop and global questions | 07 |
| **Hallucination** | Fluent output not supported by sources or reality (intrinsic vs. extrinsic) | 18 |
| **HITL (human-in-the-loop)** | Human approval, review, or correction at defined points in a workflow | 10 |
| **HNSW** | Graph-based approximate nearest-neighbour index; default in most vector DBs | 06 |
| **Hybrid search** | Combining dense (vector) and sparse (BM25) retrieval, usually fused with RRF | 07 |
| **HyDE** | Hypothetical Document Embeddings: embed a generated answer instead of the raw query | 07 |
| **Idempotency key** | Unique key so retried write operations execute only once | 08, 19 |
| **Indirect prompt injection** | Malicious instructions hidden in data the model reads (web pages, docs, emails) | 05, 18 |
| **KV cache** | Cached attention keys/values of previous tokens; makes decoding fast and drives memory use | 04, 13 |
| **Lethal trifecta** | Private data + untrusted content + ability to exfiltrate in one agent → data theft risk | 18 |
| **LLM-as-a-judge** | Using a model to grade outputs with a rubric or pairwise comparison | 16 |
| **LoRA / QLoRA** | Low-rank adapter fine-tuning; QLoRA does it over a 4-bit quantized base | 17 |
| **Lost in the middle** | Models attend less to information in the middle of long contexts | 05, 07 |
| **MCP (Model Context Protocol)** | Open protocol for exposing tools, resources, and prompts to LLM clients via servers | 09 |
| **Memory (agent)** | Short-term (context), working (scratchpad), long-term (episodic, semantic, procedural) stores | 10 |
| **MRR / nDCG / Recall@k** | Retrieval metrics: first-hit rank, graded ranking quality, share of relevant docs in top k | 06, 07 |
| **Multi-agent system** | Several specialised agents coordinated (orchestrator–workers, handoffs, debate) | 10 |
| **OWASP LLM Top 10** | Standard list of LLM application risks (prompt injection, excessive agency, etc.) | 18 |
| **PEFT** | Parameter-efficient fine-tuning (LoRA, adapters, prefix tuning) | 17 |
| **Prompt caching** | Provider reuses computation for an identical prompt prefix; cheaper and faster | 13 |
| **Prompt injection** | Input that overrides the developer's instructions | 05, 18 |
| **Quantization** | Lower-precision weights/activations (int8, int4, fp8) to cut memory and cost | 13 |
| **RAG** | Retrieval-Augmented Generation: retrieve relevant context and generate from it | 07 |
| **ReAct** | Interleaving reasoning and actions (tool calls) with observations | 10 |
| **Reasoning model** | Model trained to think at length before answering; controlled by reasoning effort/budget | 04 |
| **Red teaming** | Adversarial testing to find safety and security failures before attackers do | 18 |
| **Reranking** | Second-stage ordering of retrieved candidates with a stronger model | 07 |
| **RLHF / RLVR** | RL from human feedback / from verifiable rewards | 04, 17 |
| **Router** | Component sending each request to a model, chain, or agent based on intent or difficulty | 11, 13 |
| **RRF (Reciprocal Rank Fusion)** | Fusing ranked lists by summing 1/(k + rank) | 07 |
| **Sandbox** | Isolated execution environment (container, microVM) for agent code and tools | 14 |
| **Self-consistency** | Sample several reasoning paths and take the majority answer | 05 |
| **Semantic cache** | Cache keyed on embedding similarity of queries rather than exact match | 11 |
| **Semantic entropy** | Hallucination signal: disagreement in meaning across sampled answers | 18 |
| **SFT** | Supervised fine-tuning on input→output examples | 17 |
| **Shadow mode** | New system runs on real traffic without its output reaching users | 12 |
| **Speculative decoding** | Draft model proposes tokens, target model verifies in parallel; faster with same output | 13 |
| **Streaming** | Returning tokens as generated (SSE/WebSocket) to cut perceived latency | 03 |
| **Structured outputs** | Model output guaranteed to match a JSON schema | 03 |
| **System prompt** | Developer instructions setting role, rules, and format for a conversation | 05 |
| **Temperature / top-p** | Sampling controls: flatten/sharpen the distribution; restrict to a probability mass | 03 |
| **Tokenization** | Splitting text into subword tokens (BPE); determines cost and context use | 04 |
| **TTFT / TPOT** | Time to first token / time per output token | 13 |
| **Vector database** | Store with ANN indexing, metadata filtering, and CRUD for embeddings | 06 |
| **VLM** | Vision-language model that takes images (and text) as input | 15 |
| **Workflow** | Predefined orchestration of LLM calls and code (chain, router, parallel, evaluator-optimizer) | 10 |

## Related
- [Index](README.md) · [Cheatsheet](CHEATSHEET.md) · [Flashcards](FLASHCARDS.md)
