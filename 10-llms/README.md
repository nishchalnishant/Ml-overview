---
module: LLMs
topic: LLM Core: Theory, Training & Research
subtopic: ""
status: unread
tags: [llms, ml, llm-core-theory-training-research]
---
# LLM Core: Theory, Training & Research

This directory serves as the theoretical bedrock for Large Language Models (LLMs). It covers the fundamental research, architectural innovations, and training methodologies that power models like GPT-4, Llama 3, and Claude.

> **Frontier model roster:** the specific-model examples in this folder's deep-dives (GPT-4o, Claude 3, o3, DeepSeek-R1, Qwen3, etc.) are illustrative, not exhaustive — they're not refreshed every model release.

## Theoretical Roadmap

### 1. [Training Lifecycle](01-training-process.md)
*From trillions of tokens to a helpful assistant.*
- Pre-training (Objectives, Data Mixture).
- SFT (Instruction Tuning).
- Alignment (RLHF vs DPO).

### 2. [Scaling Laws & Data](02-scaling-and-data.md)
*The physics of LLM performance.*
- Compute Optimality (Chinchilla Laws).
- Data Quality & Deduplication.
- Synthetic Data generation.

### 3. [Advanced Architecture](03-architecture-deep-dive.md)
*State-of-the-art model internals.*
- Mixture of Experts (MoE).
- KV-Cache & Attention Optimization (GQA, MQA).
- Flash Attention & RoPE.

### 4. [Evaluation & Benchmarks](04-evaluation-benchmarks.md)
*Measuring intelligence objectively.*
- MMLU, GSM8K, and HumanEval.
- Model-as-a-Judge.
- LMSYS Chatbot Arena.

---

## Related Directories

This folder is the *theory/training* layer. Applied and systems treatments of the same topics live next door:
- **[11-llm-applications/](../11-llm-applications/)** — building with LLMs: RAG, agentic workflows, prompt optimization, multimodal, inference optimization, MCP.
- **[12-systems-and-scale/](../12-systems-and-scale/)** — the serving/systems layer: model compression, LLM serving, inference ops.
- **[interview-notes/](interview-notes/)** — topic-by-topic interview Q&A in deep + snappy modalities.

---

## [Interview Question Bank](_interview-questions.md)
*Direct answers to high-signal LLM interview questions.*
- Architecture deep-dives.
- Training strategies.
- Production trade-offs.

> This is a small, first-principles-reasoning-focused set (why the trap exists, how to reason to the answer) — for a much larger topic-by-topic Q&A bank (deep + snappy versions, RAG/agents/fine-tuning/evals/system design), see [interview-notes/README.md](interview-notes/README.md).

**Applied companion:** [_interview-questions-applied.md](_interview-questions-applied.md) — a 37-question bank concentrated on RAG pipelines, agent patterns (ReAct, MCP, prompt injection), serving optimizations (PagedAttention, continuous batching, prefix caching), quantization, and LLMOps/eval design.

There is also [interviewquestions.md](../16-interview-prep/study-plans/interviewquestions.md) — a rapid-fire Easy/Medium/Hard tiered Q&A bank spanning the whole folder (tokenization through RLHF math), calibrated to interview round depth rather than organized by topic.

---

## [Recommended Reading](README.md)
*Deep-dive books and resources on LLM development.*
