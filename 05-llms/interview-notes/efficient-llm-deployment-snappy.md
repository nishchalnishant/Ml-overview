---
module: Llms
topic: Interview Notes
subtopic: Efficient Llm Deployment Snappy
status: unread
tags: [llms, ml, interview-notes-efficient-llm-]
---
# Efficient LLM deployment & optimization — “keep it fast, keep it sane”

If you’ve ever tuned a Kubernetes service under load, this will feel familiar: **latency, throughput, memory, cost, and safety**.

**One-line:** Deployment efficiency = (better kernels + smarter batching + smaller weights) + (tight SLIs + guardrails).

---

## 1) Quantization & compression (make it fit)

### Q1: What is quantization in LLMs? (FP16 → INT8/INT4)
- **Direct answer:** Lower numeric precision for weights/activations so models use less memory and run faster.
- **Music analogy:** like compressing a classic track—8-bit is near-lossless, 4-bit is “still beautiful,” 2-bit often sounds underwater.
- **Practical:** FP16/BF16 (training), 8-bit (near-lossless), 4-bit (common local/cheap), AWQ/GPTQ (popular PTQ flavors).

**Mini pop quiz:** Quantization mostly speeds up LLMs because of what? → **memory bandwidth** (LLMs are memory-bound).

---

### Q2: How do you estimate VRAM for serving a model?
- **Direct answer:** You pay for **weights + KV cache + overhead**.
- **Rule of thumb:**
  - weights: \(P \times \text{bytes/param}\)
  - KV cache grows with **(context × concurrency × layers × hidden size)**

**Quick thought experiment:** You moved from 8k to 64k context and p95 latency exploded—what likely grew? → **KV cache**.

---

## 2) Serving optimizations (make it fast)

### Q3: What is KV caching and why is it essential?
- **Direct answer:** Cache past Keys/Values so decoding doesn’t recompute history every token.
- **DevOps bridge:** it’s memoization + locality. Great for speed, expensive for memory.

---

### Q4: FlashAttention & paged attention (vLLM)
- **FlashAttention:** faster attention via fused kernels and better GPU memory IO.
- **Paged attention:** manages KV cache like virtual memory pages → fewer fragmentation issues and higher concurrency.

---

### Q5: Speculative decoding
- **Direct answer:** draft model proposes tokens; target model verifies in chunks.
- **MI analogy:** a junior fielder predicts the shot, captain confirms the placement—fewer wasted moves.

---

## 3) Throughput, batching, and scheduling (make it scale)

### Q6: Continuous batching
- **Direct answer:** dynamically pack incoming requests so the GPU stays busy.
- **Trade-off:** better throughput, slightly more queueing delay.

### Q7: Prefill vs decode — why you care
- **Prefill:** process the prompt (heavy compute once).
- **Decode:** generate tokens one-by-one (where KV cache helps).

### Q8: Load balancing for LLMs
- **Direct answer:** route based on **model residency**, GPU memory headroom, and queue depth—not just CPU.

---

## 4) Edge / local deployment (make it portable)

### Q9: How do you run LLMs on edge/mobile?
- **Levers:** smaller models, distillation, aggressive quantization, limited context, on-device caching.
- **Reality:** correctness requirements must be lower or narrowly scoped.

---

## 5) Safety + reliability (make it production)

### Q10: What are the “don’t page me at 2am” guardrails?
- **Hard caps:** max tokens, max tool calls, timeouts.
- **Validation:** schema/JSON enforcement, allow-listed tools, safe parameter ranges.
- **Fallbacks:** smaller model, cached responses, “insufficient data.”

---

## 6) Azure deployment sketch (how you’d actually ship this)

### Q11: How would you deploy an LLM service using Azure + DevOps?
- **Build:** containerize inference server (vLLM/TGI/ONNX runtime).
- **Provision:** AKS GPU node pool (or managed endpoints), autoscaling rules.
- **Release:** blue/green or canary; monitor TTFT + tokens/sec + p95.
- **Observe:** App Insights + custom metrics; alert on error rate/latency + quality regressions.

**Mini prompt:** What metric tells you users *feel* it’s slow? → **TTFT** (time-to-first-token).

## Flashcards

**Direct answer?** #flashcard
Lower numeric precision for weights/activations so models use less memory and run faster.

**Music analogy?** #flashcard
like compressing a classic track—8-bit is near-lossless, 4-bit is “still beautiful,” 2-bit often sounds underwater.

**Practical?** #flashcard
FP16/BF16 (training), 8-bit (near-lossless), 4-bit (common local/cheap), AWQ/GPTQ (popular PTQ flavors).

**Direct answer?** #flashcard
You pay for weights + KV cache + overhead.

**Rule of thumb:?** #flashcard
Rule of thumb:

**weights?** #flashcard
\(P \times \text{bytes/param}\)

**KV cache grows with (context × concurrency × layers × hidden size)?** #flashcard
KV cache grows with (context × concurrency × layers × hidden size)

**Direct answer?** #flashcard
Cache past Keys/Values so decoding doesn’t recompute history every token.

**DevOps bridge?** #flashcard
it’s memoization + locality. Great for speed, expensive for memory.

**FlashAttention?** #flashcard
faster attention via fused kernels and better GPU memory IO.

**Paged attention?** #flashcard
manages KV cache like virtual memory pages → fewer fragmentation issues and higher concurrency.

**Direct answer?** #flashcard
draft model proposes tokens; target model verifies in chunks.

**MI analogy?** #flashcard
a junior fielder predicts the shot, captain confirms the placement—fewer wasted moves.

**Direct answer?** #flashcard
dynamically pack incoming requests so the GPU stays busy.

**Trade-off?** #flashcard
better throughput, slightly more queueing delay.

**Prefill?** #flashcard
process the prompt (heavy compute once).

**Decode?** #flashcard
generate tokens one-by-one (where KV cache helps).

**Direct answer?** #flashcard
route based on model residency, GPU memory headroom, and queue depth—not just CPU.

**Levers?** #flashcard
smaller models, distillation, aggressive quantization, limited context, on-device caching.

**Reality?** #flashcard
correctness requirements must be lower or narrowly scoped.

**Hard caps?** #flashcard
max tokens, max tool calls, timeouts.

**Validation?** #flashcard
schema/JSON enforcement, allow-listed tools, safe parameter ranges.

**Fallbacks?** #flashcard
smaller model, cached responses, “insufficient data.”

**Build?** #flashcard
containerize inference server (vLLM/TGI/ONNX runtime).

**Provision?** #flashcard
AKS GPU node pool (or managed endpoints), autoscaling rules.

**Release?** #flashcard
blue/green or canary; monitor TTFT + tokens/sec + p95.

**Observe?** #flashcard
App Insights + custom metrics; alert on error rate/latency + quality regressions.
