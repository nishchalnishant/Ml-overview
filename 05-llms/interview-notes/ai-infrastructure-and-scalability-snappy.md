---
module: Llms
topic: Interview Notes
subtopic: Ai Infrastructure And Scalability Snappy
status: unread
tags: [llms, ml, interview-notes-ai-infrastruct]
---
# AI infrastructure & scalability — run LLMs like real services

This is the “why is my GPU bill screaming?” section.

**One-line:** Serving LLMs is mostly **memory + batching + scheduling + caching**.

---

# Q1: LLM optimization techniques
- **Direct answer:** The big levers: batching, KV cache, quantization, speculative decoding, faster attention kernels, routing.
- **MI analogy:** you win by field placement + shot selection, not by hoping every ball is a Yorker.

---

# Q2: How do you select GPUs for LLM inference?
- **Criteria:** VRAM (model + KV cache), bandwidth, tensor cores, concurrency, cost.
- **Rule:** if it OOMs, it doesn’t matter how fast the FLOPs are.

---

# Q3: Model parallelism vs data parallelism in distributed training?
- **Data parallel:** split batch across GPUs.
- **Model parallel:** split model weights across GPUs.

---

# Q4: What is tensor parallelism?
- **Direct answer:** Split tensor operations (matrix mults) across GPUs; needed when a single GPU can’t hold compute/weights efficiently.

---

# Q5: What is pipeline parallelism?
- **Direct answer:** Split layers into stages across GPUs; micro-batch to keep pipeline busy.

---

# Q6: Continuous batching for throughput?
- **Direct answer:** Merge requests dynamically so GPUs run large batches even with staggered arrivals.
- **DevOps bridge:** like request coalescing + queue-based workers.

---

# Q7: Speculative decoding?
- **Direct answer:** Small “draft” model proposes tokens; big model verifies in chunks → faster perceived speed.

---

# Q8: KV cache — manage memory?
- **Direct answer:** KV cache accelerates decode but consumes VRAM proportional to context and concurrency.
- **Fixes:** shorter contexts, GQA/MQA, paged KV, cache quantization (careful), routing.

---

# Q9: Paged Attention and vLLM?
- **Direct answer:** Manage KV cache like virtual memory pages to reduce fragmentation and improve concurrency.

---

# Q10: Optimize inference for edge/mobile?
- **Levers:** smaller models, aggressive quantization, distillation, on-device batching, prune features.

---

# Q11: Quantization (INT8/INT4/FP16/BF16) and quality?
- **Direct answer:** lower precision saves memory/bandwidth; 8-bit near-lossless; 4-bit common; below that risky.

---

# Q12: Auto-scaling for AI workloads?
- **Signals:** queue depth, GPU util, TTFT, p95 latency, error rate.
- **Caution:** scaling too fast can thrash cold starts.

---

# Q13: Load balancing in AI serving?
- **Direct answer:** route to healthy GPUs, account for model residency and KV cache locality.

---

# Q14: GPU memory for serving multiple models?
- **Patterns:** model pooling, adapter swapping (LoRA), eviction policies, multi-model servers.

---

# Q15: Model sharding — when?
- **Direct answer:** when model weights don’t fit on one GPU or you need multi-GPU throughput.

---

# Q16: Request queuing and priority scheduling?
- **Direct answer:** queues protect GPUs; priorities support VIP traffic and background jobs.

---

# Q17: Self-hosted vs API-based inference cost trade-offs?
- **API:** faster to ship, predictable-ish billing, less ops.
- **Self-host:** control + potentially cheaper at scale, but you own reliability and capacity planning.

---

# Q18: Cold start latency for serverless?
- **Fixes:** warm pools, provisioned concurrency, model snapshotting, smaller models for first response.

---

# Q19: Model caching to reduce redundant compute?
- **Direct answer:** prompt caching / prefix caching; reuse KV for repeated prefixes.

---

# Q20: Sync vs async inference?
- **Sync:** chat/interactive.
- **Async:** batch jobs, long runs; better utilization.

---

# Q21: FSDP vs DeepSpeed ZeRO?
- **Direct answer:** both shard optimizer/grad/params to scale training; different implementations and trade-offs.

---

# Q22: Monitor/profile LLM inference (TTFT, inter-token latency, GPU util)?
- **Key metrics:** TTFT, tokens/sec, p95 latency, queue time, GPU memory/util, error rate.
- **DevOps bridge:** treat them like SLIs + error budgets.

---

# Q23: Model routing — route by complexity/cost?
- **Direct answer:** send easy tasks to cheap models; hard tasks to expensive models; enforce budgets.
- **Pattern:** classifier/router + fallbacks + eval gates.

## Rapid Recall

### Direct answer
- Direct Answer: The big levers: batching, KV cache, quantization, speculative decoding, faster attention kernels, routing.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: The big levers: batching, KV cache, quantization, speculative decoding, faster attention kernels, routing.

### MI analogy
- Direct Answer: you win by field placement + shot selection, not by hoping every ball is a Yorker.
- Why: This matters because it tells you how to reason about mi analogy.
- Pitfall: Don't answer "MI analogy" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: you win by field placement + shot selection, not by hoping every ball is a Yorker.

### Criteria
- Direct Answer: VRAM (model + KV cache), bandwidth, tensor cores, concurrency, cost.
- Why: This matters because it tells you how to reason about criteria.
- Pitfall: Don't answer "Criteria" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: VRAM (model + KV cache), bandwidth, tensor cores, concurrency, cost.

### Rule
- Direct Answer: if it OOMs, it doesn’t matter how fast the FLOPs are.
- Why: This matters because it tells you how to reason about rule.
- Pitfall: Don't answer "Rule" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: if it OOMs, it doesn’t matter how fast the FLOPs are.

### Data parallel
- Direct Answer: split batch across GPUs.
- Why: This matters because it tells you how to reason about data parallel.
- Pitfall: Don't answer "Data parallel" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: split batch across GPUs.

### Model parallel
- Direct Answer: split model weights across GPUs.
- Why: This matters because it tells you how to reason about model parallel.
- Pitfall: Don't answer "Model parallel" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: split model weights across GPUs.

### Direct answer
- Direct Answer: Split tensor operations (matrix mults) across GPUs; needed when a single GPU can’t hold compute/weights efficiently.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Split tensor operations (matrix mults) across GPUs; needed when a single GPU can’t hold compute/weights efficiently.

### Direct answer
- Direct Answer: Split layers into stages across GPUs; micro-batch to keep pipeline busy.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Split layers into stages across GPUs; micro-batch to keep pipeline busy.

### Direct answer
- Direct Answer: Merge requests dynamically so GPUs run large batches even with staggered arrivals.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Merge requests dynamically so GPUs run large batches even with staggered arrivals.

### DevOps bridge
- Direct Answer: like request coalescing + queue-based workers.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: like request coalescing + queue-based workers.

### Direct answer
- Direct Answer: Small “draft” model proposes tokens; big model verifies in chunks → faster perceived speed.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Small “draft” model proposes tokens; big model verifies in chunks → faster perceived speed.

### Direct answer
- Direct Answer: KV cache accelerates decode but consumes VRAM proportional to context and concurrency.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: KV cache accelerates decode but consumes VRAM proportional to context and concurrency.

### Fixes
- Direct Answer: shorter contexts, GQA/MQA, paged KV, cache quantization (careful), routing.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: shorter contexts, GQA/MQA, paged KV, cache quantization (careful), routing.

### Direct answer
- Direct Answer: Manage KV cache like virtual memory pages to reduce fragmentation and improve concurrency.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Manage KV cache like virtual memory pages to reduce fragmentation and improve concurrency.

### Levers
- Direct Answer: smaller models, aggressive quantization, distillation, on-device batching, prune features.
- Why: This matters because it tells you how to reason about levers.
- Pitfall: Don't answer "Levers" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: smaller models, aggressive quantization, distillation, on-device batching, prune features.

### Direct answer
- Direct Answer: lower precision saves memory/bandwidth; 8-bit near-lossless; 4-bit common; below that risky.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: lower precision saves memory/bandwidth; 8-bit near-lossless; 4-bit common; below that risky.

### Signals
- Direct Answer: queue depth, GPU util, TTFT, p95 latency, error rate.
- Why: This matters because it tells you how to reason about signals.
- Pitfall: Don't answer "Signals" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: queue depth, GPU util, TTFT, p95 latency, error rate.

### Caution
- Direct Answer: scaling too fast can thrash cold starts.
- Why: This matters because it tells you how to reason about caution.
- Pitfall: Don't answer "Caution" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: scaling too fast can thrash cold starts.

### Direct answer
- Direct Answer: route to healthy GPUs, account for model residency and KV cache locality.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: route to healthy GPUs, account for model residency and KV cache locality.

### Patterns
- Direct Answer: model pooling, adapter swapping (LoRA), eviction policies, multi-model servers.
- Why: This matters because it tells you how to reason about patterns.
- Pitfall: Don't answer "Patterns" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: model pooling, adapter swapping (LoRA), eviction policies, multi-model servers.

### Direct answer
- Direct Answer: when model weights don’t fit on one GPU or you need multi-GPU throughput.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: when model weights don’t fit on one GPU or you need multi-GPU throughput.

### Direct answer
- Direct Answer: queues protect GPUs; priorities support VIP traffic and background jobs.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: queues protect GPUs; priorities support VIP traffic and background jobs.

### API
- Direct Answer: faster to ship, predictable-ish billing, less ops.
- Why: This matters because it tells you how to reason about api.
- Pitfall: Don't answer "API" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: faster to ship, predictable-ish billing, less ops.

### Self-host
- Direct Answer: control + potentially cheaper at scale, but you own reliability and capacity planning.
- Why: This matters because it tells you how to reason about self-host.
- Pitfall: Don't answer "Self-host" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: control + potentially cheaper at scale, but you own reliability and capacity planning.

### Fixes
- Direct Answer: warm pools, provisioned concurrency, model snapshotting, smaller models for first response.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: warm pools, provisioned concurrency, model snapshotting, smaller models for first response.

### Direct answer
- Direct Answer: prompt caching / prefix caching; reuse KV for repeated prefixes.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: prompt caching / prefix caching; reuse KV for repeated prefixes.

### Sync
- Direct Answer: chat/interactive.
- Why: This matters because it tells you how to reason about sync.
- Pitfall: Don't answer "Sync" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: chat/interactive.

### Async
- Direct Answer: batch jobs, long runs; better utilization.
- Why: This matters because it tells you how to reason about async.
- Pitfall: Don't answer "Async" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: batch jobs, long runs; better utilization.

### Direct answer
- Direct Answer: both shard optimizer/grad/params to scale training; different implementations and trade-offs.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: both shard optimizer/grad/params to scale training; different implementations and trade-offs.

### Key metrics
- Direct Answer: TTFT, tokens/sec, p95 latency, queue time, GPU memory/util, error rate.
- Why: This matters because it tells you how to reason about key metrics.
- Pitfall: Don't answer "Key metrics" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: TTFT, tokens/sec, p95 latency, queue time, GPU memory/util, error rate.

### DevOps bridge
- Direct Answer: treat them like SLIs + error budgets.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: treat them like SLIs + error budgets.

### Direct answer
- Direct Answer: send easy tasks to cheap models; hard tasks to expensive models; enforce budgets.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: send easy tasks to cheap models; hard tasks to expensive models; enforce budgets.

### Pattern
- Direct Answer: classifier/router + fallbacks + eval gates.
- Why: This matters because it tells you how to reason about pattern.
- Pitfall: Don't answer "Pattern" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: classifier/router + fallbacks + eval gates.
