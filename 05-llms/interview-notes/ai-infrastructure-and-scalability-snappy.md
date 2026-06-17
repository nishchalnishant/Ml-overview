---
module: Llms
topic: Interview Notes
subtopic: Ai Infrastructure And Scalability Snappy
status: unread
tags: [llms, ml, interview-notes-ai-infrastruct]
---

> _Quick-recall companion. For the full deep-dive, see [ai-infrastructure-and-scalability.md](ai-infrastructure-and-scalability.md)._

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
