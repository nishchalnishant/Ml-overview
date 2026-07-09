---
module: LLMs
topic: LLM Applications
subtopic: ""
status: unread
tags: [llms, ml, llm-applications]
---
# LLM Applications

This folder is where LLM theory stops posing for LinkedIn and starts shipping features.

If Azure DevOps is your comfort zone, this whole folder will click faster if you use one simple mental model:

- **base model** = the shared platform image
- **prompt** = runtime configuration
- **RAG** = dependency injection for knowledge at runtime
- **fine-tuning** = building a customized artifact
- **agent workflow** = an orchestrated multi-step pipeline with retries, tools, and guardrails

In short: an LLM app is rarely "just a model."

It is usually a stack of:

- retrieval
- orchestration
- tool calling
- validation
- monitoring

That is why good LLM engineering feels surprisingly close to platform engineering.

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

## Azure / DevOps Bridge

Here is the fastest translation layer:

- **training** is like building a heavy artifact in a costly pipeline
- **inference** is the runtime service path where latency matters
- **evaluation** is your release gate
- **guardrails** are policy checks and runtime protections
- **model registry** is your artifact registry
- **prompt versioning** is config versioning with real production impact

Once you see that, LLM applications stop feeling mystical and start feeling operational.

## Fashion Analogy

A base model is like a beautifully stitched couture jacket off the runway.

Useful? Yes.

Ready for your exact event, climate, styling goal, and audience? Not always.

RAG is the styling layer that brings in the right scarf, shoes, and accessories for tonight.
Fine-tuning is tailoring the jacket itself.

Both matter. They solve different problems.

## Quick Thought Experiment

Your chatbot gives the wrong answer about a company policy.

Where do you inspect first?

- the prompt?
- retrieval?
- document freshness?
- model choice?
- post-processing guardrails?

Best answer: **the whole chain**.

That is how strong LLM engineers think.
