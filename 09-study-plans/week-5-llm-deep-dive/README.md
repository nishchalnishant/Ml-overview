---
module: Study Plans
topic: Week 5 (Days 31-35): LLM Deep Dive
subtopic: ""
status: unread
tags: [studyplans, ml, week-5-days-31-35-llm-deep-dive]
---
# Week 5 (Days 31-35): LLM Deep Dive

**Goal:** Extend the core 30-day sprint with the LLM-specific depth that modern ML interviews increasingly require — architecture internals, efficient training/inference, alignment, and agentic systems. Weeks 1-4 cover classical ML, algorithms, system design, and final prep; this week fills the gap where LLM coverage was previously limited to two days.

---

## What This Week Covers

| Days | Topic | Key Concepts |
|------|-------|--------------|
| 31 | Architecture & Scaling | Transformer internals, scaling laws, training stability, MoE routing |
| 32 | Efficient Training & Inference | LoRA/QLoRA, KV cache, MQA/GQA, quantization, speculative decoding |
| 33 | Alignment & Safety | RLHF, PPO vs. DPO, reward hacking, AI safety and responsible AI |
| 34 | Agentic Systems & RAG | Agentic workflows, MCP, RAG architecture, hallucination mitigation |
| 35 | Applied & Production LLMs | Multimodal, context window extension, LLMOps, mock interview |

---

## Focus Areas

- **Architecture internals:** Be able to derive attention complexity, explain why scale by $\sqrt{d_k}$, and reason about MoE routing and load balancing.
- **Efficiency tradeoffs:** Know why KV cache/MQA/GQA/quantization/speculative decoding exist — each solves a specific cost or latency constraint, not just "makes it faster."
- **Alignment mechanics:** Understand RLHF's reward-model + PPO pipeline versus DPO's direct optimization, and where each can fail (reward hacking, preference data quality).
- **Agentic & retrieval systems:** Be able to design a RAG pipeline end-to-end and reason about agent/tool-use failure modes (including MCP as the emerging tool-integration standard).
- **Production concerns:** LLMOps practices — prompt versioning, eval harnesses, drift monitoring for generative outputs — differ meaningfully from classical MLOps.

---

## Daily Study Pattern

1. Read the day's primary sources, then close the laptop and explain the mechanism out loud from memory.
2. For every technique, answer: "what specific constraint or failure does this solve?" (e.g., KV cache solves recomputation cost; paged attention solves memory fragmentation).
3. End each day by writing 3 interview questions you could be asked on the topic, then answering them cold.

---

## Linked Resources

### Architecture & Scaling
- [Architecture Deep Dive](../../05-llms/03-architecture-deep-dive.md)
- [Scaling and Data](../../05-llms/02-scaling-and-data.md)
- [Training Stability](../../05-llms/05-training-stability.md)
- [MoE Advanced and Routing](../../05-llms/09-moe-advanced-and-routing.md)

### Efficient Training & Inference
- [Fine-Tuning at Scale](../../05-llms/06-fine-tuning-at-scale.md)
- [KV Cache and MQA/GQA](../../05-llms/08-kv-cache-and-mqa-gqa.md)
- [Inference Optimization](../../05-llms/applications/06-inference-optimization.md)
- [Speculative Decoding](../../05-llms/applications/07-speculative-decoding.md)
- [Efficient LLM Deployment](../../05-llms/interview-notes/16-efficient-llm-deployment.md)

### Alignment & Safety
- [Advanced Alignment and Reasoning](../../05-llms/interview-notes/17-advanced-alignment-and-reasoning.md)
- [Production Alignment Failures](../../05-llms/interview-notes/18-production-alignment-failures.md)
- [AI Safety, Ethics, and Responsible AI](../../05-llms/interview-notes/10-ai-safety-ethics-and-responsible-ai-what.md)

### Agentic Systems & RAG
- [Agentic Workflows](../../05-llms/applications/01-agentic-workflows.md)
- [MCP](../../05-llms/applications/12-mcp.md)
- [Retrieval-Augmented Generation (RAG)](../../05-llms/interview-notes/03-retrieval-augmented-generation-rag.md)
- [Hallucination Mitigation](../../05-llms/applications/05-hallucination-mitigation.md)
- [AI Agents and Agentic Systems](../../05-llms/interview-notes/04-ai-agents-and-agentic-systems.md)

### Applied & Production LLMs
- [Multimodal](../../05-llms/applications/09-multimodal.md)
- [Context Window Extension](../../05-llms/07-context-window-extension.md)
- [LLMOps and Production AI](../../05-llms/interview-notes/08-llmops-and-production-ai.md)
- [AI System Design](../../05-llms/interview-notes/07-ai-system-design.md)

Day files in this folder: 01-day-31-32, 02-day-33-34, 03-day-35

---

## Projects for This Week

**Day 31-32 Project: Efficiency Tradeoff Table**

Build a one-page table comparing KV cache, MQA/GQA, quantization (int8/int4), and speculative decoding: what each optimizes (memory vs. latency vs. throughput), what it costs (accuracy, complexity), and when you'd reach for it. Present it as if explaining it to a teammate deciding how to cut inference cost by 50%.

**Day 33-34 Project: Design a RAG System with Guardrails**

Design a RAG-based support assistant for a product with a large, frequently-updated knowledge base. Specify: chunking strategy, embedding model choice, vector DB, reranking, hallucination mitigation (citation-grounding, refusal when retrieval confidence is low), and how you'd evaluate answer quality offline and online.

**Day 35 Project: Mock Interview — LLM System Design**

Run a 25-minute mock interview on: "Design an AI coding assistant that reads a codebase and answers questions with citations, using tools (search, run tests) via MCP." Cover architecture, latency budget, failure modes, and evaluation.

---

## Milestone Checkpoints

**After Day 32:** Can you explain why paged attention exists (what specific problem in naive KV cache it solves) and derive the parameter savings of LoRA vs. full fine-tuning?

**After Day 34:** Can you design a RAG pipeline end-to-end and name the failure mode at each stage (embedding, retrieval, reranking, generation)? Can you explain what MCP standardizes and why it matters for agentic tool use?

**After Day 35:** Can you contrast RLHF and DPO in under 90 seconds, covering both mechanism and failure mode (reward hacking vs. loss of reward/policy separation)?

---

## End-of-Week Check

- Can you explain the constraint each of KV cache, MQA/GQA, and speculative decoding was built to solve?
- Can you design a RAG system end-to-end, including where and why you'd add a reranker?
- Can you contrast RLHF and DPO, including a concrete failure mode for each?
- Can you explain MoE routing and the load-balancing problem it introduces?
- Can you describe how LLMOps differs from classical MLOps (prompt versioning, eval harnesses, generative-output drift)?
