---
module: LLMs
topic: LLM Applications
subtopic: ""
status: unread
tags: [llms, ml, llm-applications]
---
# LLM Applications

An LLM app is rarely "just a model" — it's usually a stack of retrieval, orchestration, tool calling, validation, and monitoring. That's why LLM engineering feels close to platform engineering.

Mental model:
- **base model** = the shared platform image
- **prompt** = runtime configuration
- **RAG** = dependency injection for knowledge at runtime
- **fine-tuning** = building a customized artifact
- **agent workflow** = an orchestrated multi-step pipeline with retries, tools, and guardrails

## Start Here

Read these in order:

1. `11-how-to-train-your-dragon-llm.md`
2. `02-rag.md`
3. `10-tuning-optimization.md`
4. `01-agentic-workflows.md`

That journey gives you the clean progression:

- how LLMs work
- how they use external knowledge
- how they are adapted
- how they become useful systems

## What Each File Helps You Do

- `11-how-to-train-your-dragon-llm.md`
  Understand attention, tokens, inference, scaling, and why LLMs behave the way they do.

- `02-rag.md`
  Learn how to ground answers in fresh documents instead of hoping the model memorized everything.

- `03-prompt-optimization-and-versioning.md`
  Treat prompts as versioned, testable artifacts instead of throwaway strings.

- `04-synthetic-data.md`
  Generate and filter training data with LLMs instead of relying purely on human-labeled sets.

- `05-hallucination-mitigation.md`
  Reduce ungrounded, confidently-wrong outputs with grounding, decoding, and verification techniques.

- `06-inference-optimization.md`
  Speed up serving with KV cache, PagedAttention, FlashAttention, and batching strategies.

- `07-speculative-decoding.md`
  Get multi-token-per-step throughput by pairing a small draft model with the target model.

- `08-model-merging.md`
  Combine multiple fine-tuned models into one without retraining.

- `09-multimodal.md`
  Extend LLMs to handle vision, audio, and other modalities alongside text.

- `10-tuning-optimization.md`
  Decide when to prompt, when to retrieve, when to fine-tune, and when to save GPU money with smarter adaptation.

- `01-agentic-workflows.md`
  Design multi-step LLM systems that can think, call tools, recover, and not spiral into chaos.

- `12-mcp.md`
  Understand the Model Context Protocol — the standard interface that makes tools reusable across agent frameworks.

## Debugging Mental Model

Chatbot gives a wrong answer about company policy — where do you look first: prompt, retrieval, document freshness, model choice, or post-processing guardrails?

Answer: **the whole chain**. Treat it as a pipeline, not a single black box, and inspect each stage.
