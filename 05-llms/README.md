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

## [Interview Question Bank](11-interview-questions.md)
*Direct answers to high-signal LLM interview questions.*
- Architecture deep-dives.
- Training strategies.
- Production trade-offs.

> This is a small, first-principles-reasoning-focused set (why the trap exists, how to reason to the answer) — for a much larger topic-by-topic Q&A bank (deep + snappy versions, RAG/agents/fine-tuning/evals/system design), see [interview-notes/README.md](interview-notes/README.md).

---

## [Recommended Reading](books/README.md)
*Deep-dive books and resources on LLM development.*
