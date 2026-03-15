# LLM System Design

Production LLM systems involve **prompt engineering**, **context and tokenization**, **inference pipelines**, and **serving** (batching, caching). This note summarizes key concepts.

---

## Prompt engineering

- **System prompt**: Sets role, rules, and format (e.g. “You are a helpful assistant. Cite sources.”). Often fixed per application.
- **User / assistant turns**: Conversation history and current user message. Model is trained to continue the conversation.
- **Few-shot**: Include 1–3 (input, output) examples in the prompt to steer behavior without fine-tuning.
- **Chain-of-thought (CoT)**: Ask for “step-by-step” reasoning to improve accuracy on math and reasoning tasks.
- **Structured output**: Request JSON or a schema (e.g. tool calls, parsed fields) and optionally validate or parse with a grammar.

---

## Context windows

- **Context window**: Maximum input tokens the model accepts (e.g. 8K, 128K). Includes system + history + current message + any RAG context.
- **Usage**: Stay under the limit; reserve space for RAG chunks and response. For long docs, summarize or retrieve only relevant parts (RAG).
- **Long context**: Newer models support 100K+ tokens; still expensive and sometimes less accurate in the middle; retrieval often better than stuffing everything.

---

## Tokenization

- **Token**: Subword unit (BPE, WordPiece, or similar). One token ≈ 0.75 words (English); code and other languages vary.
- **Impact**: Length limits are in **tokens**; cost and latency scale with token count. Count tokens before calling the API (e.g. tiktoken for OpenAI).
- **Chunking for RAG**: Chunk by token count (or sentence/paragraph) so you know how many chunks fit in context.

---

## Inference pipelines

- **Single call**: One request → one response. Simple; used for single-turn QA or classification.
- **Multi-turn**: Append each user and assistant message; send full history each time. Context grows; may need summarization or truncation.
- **Streaming**: Server sends tokens as they are generated; reduces time-to-first-token and improves perceived latency.
- **Tool use**: Model returns tool calls; orchestrator runs tools and appends results; model continues. See [AGENTIC_AI](../AGENTIC_AI/README.md).

---

## Batching and caching

- **Batching**: Group multiple requests and run one forward pass (e.g. for embedding or small LLM). Improves GPU utilization and throughput.
- **Caching**: Cache identical or similar prompts (e.g. prefix cache for repeated system + history). Reduces recomputation and cost.
- **KV-cache**: Store key/value from previous tokens so generation only computes new tokens; essential for long sequences.

---

## Model serving

- **Deployment**: Run the model on GPU (single node or multi-node); use a serving framework (vLLM, TGI, TensorRT-LLM, or cloud APIs).
- **Latency**: Time to first token (TTFT) and time per output token; affected by batch size, context length, and hardware.
- **Throughput**: Requests per second or tokens per second; increased by batching and multiple replicas.
- **Cost**: Function of input + output tokens and model size; optimize with smaller models, caching, and efficient batching.

---

## Quick revision

- **Prompts**: system + history + user; few-shot and CoT improve behavior. **Context**: stay within token limit; long context has tradeoffs.
- **Tokenization**: subword units; count tokens for cost and chunking. **Inference**: single/multi-turn, streaming, tool use.
- **Serving**: batching, KV-cache, prefix cache; balance latency and throughput. **Scale**: replicas, caching, and smaller/faster models where possible.
