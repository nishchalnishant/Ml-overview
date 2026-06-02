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

## Rapid Recall

### Direct answer
- Direct Answer: Lower numeric precision for weights/activations so models use less memory and run faster.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Lower numeric precision for weights/activations so models use less memory and run faster.

### Music analogy
- Direct Answer: like compressing a classic track—8-bit is near-lossless, 4-bit is “still beautiful,” 2-bit often sounds underwater.
- Why: This matters because it tells you how to reason about music analogy.
- Pitfall: Don't answer "Music analogy" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: like compressing a classic track—8-bit is near-lossless, 4-bit is “still beautiful,” 2-bit often sounds underwater.

### Practical
- Direct Answer: FP16/BF16 (training), 8-bit (near-lossless), 4-bit (common local/cheap), AWQ/GPTQ (popular PTQ flavors).
- Why: This matters because it tells you how to reason about practical.
- Pitfall: Don't answer "Practical" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: FP16/BF16 (training), 8-bit (near-lossless), 4-bit (common local/cheap), AWQ/GPTQ (popular PTQ flavors).

### Direct answer
- Direct Answer: You pay for weights + KV cache + overhead.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: You pay for weights + KV cache + overhead.

### Rule of thumb:
- Direct Answer: Rule of thumb:
- Why: This matters because it tells you how to reason about rule of thumb:.
- Pitfall: Don't answer "Rule of thumb:" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Rule of thumb:

### weights
- Direct Answer: \(P \times \text{bytes/param}\)
- Why: This matters because it tells you how to reason about weights.
- Pitfall: Don't answer "weights" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: \(P \times \text{bytes/param}\)

### KV cache grows with (context × concurrency × layers × hidden size)
- Direct Answer: KV cache grows with (context × concurrency × layers × hidden size)
- Why: This matters because it tells you how to reason about kv cache grows with (context × concurrency × layers × hidden size).
- Pitfall: Don't answer "KV cache grows with (context × concurrency × layers × hidden size)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: KV cache grows with (context × concurrency × layers × hidden size)

### Direct answer
- Direct Answer: Cache past Keys/Values so decoding doesn’t recompute history every token.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Cache past Keys/Values so decoding doesn’t recompute history every token.

### DevOps bridge
- Direct Answer: it’s memoization + locality. Great for speed, expensive for memory.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: it’s memoization + locality. Great for speed, expensive for memory.

### FlashAttention
- Direct Answer: faster attention via fused kernels and better GPU memory IO.
- Why: This matters because it tells you how to reason about flashattention.
- Pitfall: Don't answer "FlashAttention" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: faster attention via fused kernels and better GPU memory IO.

### Paged attention
- Direct Answer: manages KV cache like virtual memory pages → fewer fragmentation issues and higher concurrency.
- Why: This matters because it tells you how to reason about paged attention.
- Pitfall: Don't answer "Paged attention" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: manages KV cache like virtual memory pages → fewer fragmentation issues and higher concurrency.

### Direct answer
- Direct Answer: draft model proposes tokens; target model verifies in chunks.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: draft model proposes tokens; target model verifies in chunks.

### MI analogy
- Direct Answer: a junior fielder predicts the shot, captain confirms the placement—fewer wasted moves.
- Why: This matters because it tells you how to reason about mi analogy.
- Pitfall: Don't answer "MI analogy" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: a junior fielder predicts the shot, captain confirms the placement—fewer wasted moves.

### Direct answer
- Direct Answer: dynamically pack incoming requests so the GPU stays busy.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: dynamically pack incoming requests so the GPU stays busy.

### Trade-off
- Direct Answer: better throughput, slightly more queueing delay.
- Why: This matters because it tells you how to reason about trade-off.
- Pitfall: Don't answer "Trade-off" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: better throughput, slightly more queueing delay.

### Prefill
- Direct Answer: process the prompt (heavy compute once).
- Why: This matters because it tells you how to reason about prefill.
- Pitfall: Don't answer "Prefill" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: process the prompt (heavy compute once).

### Decode
- Direct Answer: generate tokens one-by-one (where KV cache helps).
- Why: This matters because it tells you how to reason about decode.
- Pitfall: Don't answer "Decode" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: generate tokens one-by-one (where KV cache helps).

### Direct answer
- Direct Answer: route based on model residency, GPU memory headroom, and queue depth—not just CPU.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: route based on model residency, GPU memory headroom, and queue depth—not just CPU.

### Levers
- Direct Answer: smaller models, distillation, aggressive quantization, limited context, on-device caching.
- Why: This matters because it tells you how to reason about levers.
- Pitfall: Don't answer "Levers" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: smaller models, distillation, aggressive quantization, limited context, on-device caching.

### Reality
- Direct Answer: correctness requirements must be lower or narrowly scoped.
- Why: This matters because it tells you how to reason about reality.
- Pitfall: Don't answer "Reality" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: correctness requirements must be lower or narrowly scoped.

### Hard caps
- Direct Answer: max tokens, max tool calls, timeouts.
- Why: This matters because it tells you how to reason about hard caps.
- Pitfall: Don't answer "Hard caps" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: max tokens, max tool calls, timeouts.

### Validation
- Direct Answer: schema/JSON enforcement, allow-listed tools, safe parameter ranges.
- Why: This matters because it tells you how to reason about validation.
- Pitfall: Don't answer "Validation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: schema/JSON enforcement, allow-listed tools, safe parameter ranges.

### Fallbacks
- Direct Answer: smaller model, cached responses, “insufficient data.”
- Why: This matters because it tells you how to reason about fallbacks.
- Pitfall: Don't answer "Fallbacks" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: smaller model, cached responses, “insufficient data.”

### Build
- Direct Answer: containerize inference server (vLLM/TGI/ONNX runtime).
- Why: This matters because it tells you how to reason about build.
- Pitfall: Don't answer "Build" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: containerize inference server (vLLM/TGI/ONNX runtime).

### Provision
- Direct Answer: AKS GPU node pool (or managed endpoints), autoscaling rules.
- Why: This matters because it tells you how to reason about provision.
- Pitfall: Don't answer "Provision" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: AKS GPU node pool (or managed endpoints), autoscaling rules.

### Release
- Direct Answer: blue/green or canary; monitor TTFT + tokens/sec + p95.
- Why: This matters because it tells you how to reason about release.
- Pitfall: Don't answer "Release" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: blue/green or canary; monitor TTFT + tokens/sec + p95.

### Observe
- Direct Answer: App Insights + custom metrics; alert on error rate/latency + quality regressions.
- Why: This matters because it tells you how to reason about observe.
- Pitfall: Don't answer "Observe" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: App Insights + custom metrics; alert on error rate/latency + quality regressions.
