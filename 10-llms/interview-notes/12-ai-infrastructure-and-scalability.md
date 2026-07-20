---
module: LLMs
topic: Interview Notes
subtopic: Ai Infrastructure And Scalability
status: unread
tags: [llms, ml, interview-notes-ai-infrastruct]
---

> _Full study notes. For the quick-recall version, see [ai-infrastructure-and-scalability-snappy.md](ai-infrastructure-and-scalability-snappy.md)._

# AI Infrastructure & Scalability

## The Scenario That Drives Every Topic Here

Your model takes 3 seconds per request. You have 10,000 concurrent users. What fails first?

Not quality. Not the model. **Memory**. The GPU runs out of space to hold the intermediate state for all those in-flight sequences. Then throughput collapses — the GPU sits mostly idle because it's serializing requests it could batch. Then latency spirals because batched requests queue behind long ones.

Every technique in this file exists to fix a specific failure in that scenario. Learn which bottleneck each one addresses, and the "what should I do first?" question answers itself.

---

## Q1: LLM optimization techniques — the full map

**The problem:** "Optimize our LLM serving" is too vague to act on. Different techniques fix different bottlenecks. Without knowing what's failing, you optimize the wrong thing.

**The core insight:** LLM inference has exactly two bottlenecks — memory and compute — and they're felt differently depending on what you're measuring. Latency-sensitive paths feel the prefill bottleneck (compute). Memory-sensitive paths feel the KV cache and weight-loading bottleneck. Throughput-sensitive paths feel idle GPU time from poor batching.

**The mechanics:** Map every technique to its bottleneck:

| What's bottlenecked | Technique | Effect |
|---|---|---|
| Memory (weights) | Quantization | Fit larger model or more batch |
| Memory (KV cache) | PagedAttention, prefix caching | More concurrent requests |
| Compute (prefill) | Flash Attention, smaller models | Faster TTFT |
| Throughput (idle GPU) | Continuous batching | Higher utilization |
| Decode speed | Speculative decoding | Faster token generation |
| Scale beyond 1 GPU | Tensor / pipeline parallelism | Fit and serve huge models |

**What breaks:** Applying techniques without profiling. Using batching improvements when the actual bottleneck is memory fragmentation. Applying quantization when the task is sensitivity-critical without evaluating task metrics.

**What the interviewer is testing:** Whether you think "bottleneck first" or just list techniques. They want to hear you profile, identify the constraint, then choose a tool.

**Common traps:**
- Listing only training tricks when the question is about serving
- Treating quantization and batching as alternatives rather than orthogonal
- Not distinguishing TTFT (prefill-bound) from TPOT (decode-bound)

---

## Q2: How do you select GPUs for LLM inference?

**The problem:** You buy the GPU with the most FLOPs, deploy your model, and still get bad latency. Why? Because LLM decode is not compute-bound — it's memory-bandwidth-bound. The GPU is mostly waiting for weights to load from DRAM, not performing arithmetic.

**The core insight:** For LLM inference, pick GPUs by VRAM capacity (does the model fit?), memory bandwidth (how fast can you load weights each decode step?), and interconnect speed (can you tensor-parallelize?). FLOPs matter for prefill, not for decode.

**The mechanics:**
```
VRAM needed ≈ (N_params × bytes/param × 1.25) + KV_cache(batch, ctx, layers, heads, dim)
Max decode throughput (single request) = memory_bandwidth / weight_bytes
```

For an A100 80GB with BF16 70B model: 2 TB/s ÷ 140 GB ≈ 14 tokens/sec per request. That's the ceiling. No amount of FLOPs changes it.

For multi-GPU inference: NVLink (~600 GB/s) vs PCIe (~64 GB/s) determines whether tensor parallelism is practical. Without NVLink, the AllReduce communication overhead can exceed the compute benefit.

**What breaks:**
- Buying "most FLOPs" GPU when decode is memory-bandwidth-limited
- Under-provisioning VRAM by forgetting KV cache grows with batch × context
- Using PCIe-only topology for high tensor-parallelism degrees — communication dominates

**What the interviewer is testing:** Whether you know decode is bandwidth-bound, and that VRAM planning requires both weights and KV cache.

**Common traps:**
- Quoting only peak FLOPS without memory bandwidth
- Forgetting that KV cache is dynamic — it grows with every request in the batch
- Spec-sheet comparisons without profiling on representative workloads

---

## Q3: Continuous batching — why static batching leaves GPUs idle

**The problem:** In the 10,000-concurrent-users scenario, requests have wildly different lengths. Request A generates 10 tokens. Request B generates 500. Static batching waits for all requests in a batch to finish before admitting new ones. When Request A finishes at step 10, its GPU slot sits empty until Request B finishes at step 500. For 490 steps, you're using 1/batch_size of your GPU capacity.

**The core insight:** Autoregressive generation is iteration-level, not request-level. Each "step" is a full forward pass over the current batch. There's nothing stopping you from swapping a finished sequence out and a new one in at any step boundary.

**The mechanics — iteration-level scheduling:** at every step boundary, check which sequences finished, free their KV blocks, and admit waiting requests up to memory capacity — rather than waiting for the whole batch to drain. Throughput improvement over static batching: 5–10× for typical chat workloads with mixed-length responses. For the step-by-step scheduling diagram, see [Continuous Batching](16-efficient-llm-deployment.md#5-continuous-batching-never-let-the-gpu-wait).

**What breaks:**
- Without PagedAttention: admitting new requests requires contiguous KV memory that may not be available even if total free memory is sufficient (fragmentation)
- Long requests can still monopolize batch slots — need preemption or length-based admission
- p99 tail latency: short requests behind a long queue may still wait if admission control isn't length-aware

**What the interviewer is testing:** That you understand why static batching wastes GPU time and how iteration-level scheduling fixes it. That you connect it to PagedAttention as the enabling memory mechanism.

**Common traps:**
- Thinking batching is only a training concern
- Not knowing that continuous batching requires solving KV memory allocation (PagedAttention)
- Claiming continuous batching eliminates tail latency — it improves throughput, not worst-case latency

---

## Q4: Speculative decoding — parallelizing the sequential bottleneck

**The problem:** Autoregressive decoding is fundamentally sequential. Token N cannot be generated until token N-1 is complete. Each step requires a full forward pass through the 70B model. For long outputs, this is the dominant cost. You cannot simply "run it in parallel" — each token depends on the last.

**The core insight:** You can guess the next several tokens cheaply with a small model, then verify all guesses simultaneously with one forward pass of the large model. Verification is parallelizable because the large model can evaluate all positions at once given the draft tokens.

**The mechanics:** a small draft model generates γ candidate tokens autoregressively, then the large target model verifies all γ positions in one parallel forward pass, accepting each token with probability min(1, p_target/p_draft) and resampling from the residual at the first rejection. The output distribution is identical to sampling from the target model alone — no quality loss. Speedup depends on acceptance rate α (with α≈0.8, γ=4, throughput gain is 2–3×). For the full accept/reject derivation, see [Speculative Decoding](16-efficient-llm-deployment.md#4-speculative-decoding-parallelizing-what-was-sequential).

**What breaks:**
- Low acceptance rate: if draft and target disagree often, you run the large model more than unspeculative decoding
- Draft model must run on same hardware — adds memory pressure
- Doesn't help with prefill (only decode is the bottleneck)
- Not beneficial when the bottleneck is memory, not compute (small batch, bandwidth-limited)

**What the interviewer is testing:** The mechanism (draft + parallel verify), the correctness guarantee (same distribution), and the condition for speedup (high acceptance rate).

**Common traps:**
- Claiming quality degrades — it doesn't, by construction of the acceptance criterion
- Not knowing when it fails (low acceptance rate or memory-bound regime)
- Confusing Medusa/Eagle variants (self-speculative, no separate draft model) with standard speculative decoding

---

## Q5: KV cache — why you can't regenerate past context every step

**The problem:** At decode step N, the model needs to compute attention between the new token (query) and every previous token (keys and values). Recomputing K and V for all previous tokens at every step costs O(N²) time per request. With sequence length 4096, that's 16M operations per step, repeated 4096 times — 64B total operations just for attention.

**The core insight:** K and V for past tokens never change after they're computed. Cache them. Each new token only needs to compute its own Q, K, V, then attend to the cached K and V from all prior tokens.

**The mechanics:** KV memory scales as `2 × L × H_kv × d_h × T × bytes_per_element` — for LLaMA 3 70B (GQA, 8 KV heads) at 4096 tokens that's 1.34 GB per sequence, so KV cache dominates VRAM at any real batch size (42.8 GB at batch=32). GQA reduces this 8× versus full multi-head attention. Mitigation hierarchy: cap max_seq_len/batch_size, PagedAttention (allocate on demand), KV quantization, prefix caching. Full derivation and code in [KV Cache Mechanics](16-efficient-llm-deployment.md#3-kv-cache-mechanics-the-state-you-must-keep).

**What breaks:**
- Forgetting KV cache in capacity planning: you can fit the model but not serve any requests
- Pre-allocating contiguous max-length KV for every request: catastrophic fragmentation
- Caching with PII in shared prefix cache: data leakage risk

**What the interviewer is testing:** That you can compute KV memory, understand why it's often larger than weights at scale, and know the mitigation hierarchy.

**Common traps:**
- Accounting for weights but not KV in VRAM planning
- Not knowing that GQA dramatically reduces KV cache size
- Treating "KV cache" and "response cache" as the same thing (they're not)

---

## Q6: PagedAttention and vLLM — virtual memory for KV cache

**The problem:** Naive KV cache pre-allocates a contiguous block of memory for each request sized at max_seq_len. If max_seq_len = 8192, you reserve 8192-token worth of KV memory even for a request that generates 50 tokens. For a batch of 64 requests, 60–80% of KV memory is wasted. This fragmentation is the primary reason throughput collapses at scale.

**The core insight:** OS virtual memory solved this exact problem for RAM decades ago. Pages. Allocate fixed-size blocks (pages) only when needed. Maintain a page table mapping logical KV positions to physical memory blocks. Requests that finish release their pages immediately for reuse.

**The mechanics:** PagedAttention divides KV cache into fixed-size blocks (e.g. 16 tokens) with a page table mapping logical position → physical block, allocating on demand and letting sequences sharing a prefix share physical blocks (copy-on-write). Result: 3–4× higher batch capacity than naive contiguous allocation, 24× higher throughput than naive HF Transformers batching. vLLM combines PagedAttention + continuous batching + an OpenAI-compatible API server — a serving runtime, not a model. Full block-table diagram in [PagedAttention (vLLM)](16-efficient-llm-deployment.md#3-kv-cache-mechanics-the-state-you-must-keep).

**What breaks:**
- Very small block sizes (e.g., 4 tokens): too many page table lookups, kernel overhead
- Very large block sizes (e.g., 256 tokens): loses the fragmentation benefit
- Prefix sharing with dynamic prefixes: copy-on-write overhead for sequences that diverge quickly

**What the interviewer is testing:** The analogy to OS virtual memory, why fragmentation was the problem, and that vLLM is a system (not just a model wrapper).

**Common traps:**
- Thinking vLLM is primarily about the model — it's about the memory scheduler and serving stack
- Not knowing the block size trade-off
- Confusing prefix caching (reuse computed KV) with response caching (reuse final output)

---

## Q7: Edge and mobile deployment — everything changes at the constraint boundary

**The problem:** Your serving infrastructure assumes 80GB VRAM, NVLink, and 2TB/s memory bandwidth. A phone has 6–12GB unified memory, a Neural Engine with 2–4 TOPS, and needs to complete inference in under 200ms before the user notices. The entire serving stack is irrelevant. You need a different model and a different execution path.

**The core insight:** Edge deployment is a constraint satisfaction problem, not an optimization problem. The model must fit. The model must run fast enough. Battery life and thermal throttling are hard limits. You trade model capability for constraint satisfaction.

**The mechanics — in order of priority:**

1. Model size first: distillation or smaller architecture (1B–7B range). Quantize to INT4/INT8 via GGUF or Core ML.
2. Runtime selection: CoreML (iOS/macOS), NNAPI (Android), ONNX Runtime, llama.cpp. Each has different kernel support — benchmark on target device, not simulator.
3. Context cap: limit to 512–2048 tokens to control KV memory and inference time.
4. Batch size 1: no concurrent users on device.
5. Hybrid design: on-device for latency-sensitive/private tasks; cloud for heavy tasks.

Privacy benefit: on-device inference means user data never leaves the device. For healthcare, finance, and enterprise, this is often the primary driver.

**What breaks:**
- Assuming desktop benchmark numbers transfer to phone — Neural Engine throughput profiles differently from GPU
- Thermal throttling: sustained inference degrades after 30–60 seconds on mobile
- GGUF Q4_K_M quality is acceptable for general chat but can hurt specialized tasks (math, code)

**What the interviewer is testing:** Awareness that edge is a different problem domain, not just "smaller."

**Common traps:**
- Applying datacenter optimization thinking to edge
- Ignoring battery/thermal as constraints
- Not knowing platform-specific runtimes (CoreML, NNAPI)

---

## Q8: Quantization — trading precision for memory and speed

**The problem:** LLaMA 3 70B in FP16 requires 140 GB — two A100 80GB GPUs just for weights. INT4 quantization reduces this to 35 GB, fitting on one GPU with room for KV cache. But you're now representing each weight with 4 bits instead of 16. When does that cause accuracy problems, and when is it acceptable?

**The core insight:** Most of a model's weights lie in a smooth distribution — they're "close" to their quantized values. The exceptions are outlier weights with large magnitudes. The entire game of good quantization is: protect the important weights (large activations, sensitive layers) from precision loss.

**The mechanics:** PTQ symmetric INT8 needs no retraining (`q = round(w/s), s = max(|w|)/127`). GPTQ quantizes layer-by-layer using inverse-Hessian curvature to propagate quantization error into later columns. AWQ scales high-activation-magnitude weight channels before quantizing to protect them, and generally beats GPTQ. NF4 (bitsandbytes) uses a normal-float 4-bit format matching the bell-curve distribution of transformer weights — the default for QLoRA. Full derivations, code (`auto_gptq`, `awq`, `bitsandbytes`), and the extended tradeoff table are in [Quantization](16-efficient-llm-deployment.md#2-quantization-fitting-more-into-the-same-hardware).

**What breaks:**
- Quantizing with a calibration dataset that doesn't match production distribution
- Quantizing sensitive layers (first/last layers often degrade more) at the same bit width as others
- Claiming "no quality loss" without task-specific evaluation (perplexity is a proxy)
- BF16 vs FP16: BF16 has wider exponent range (better for training stability); FP16 has higher precision in mantissa

**What the interviewer is testing:** That you know the quality-size trade-off, when each method is appropriate, and that "evaluate on task metrics" is mandatory.

**Common traps:**
- Claiming quantization has no quality impact without eval
- Not distinguishing PTQ (no retraining) from QAT (quantization-aware training)
- Confusing weight quantization with activation quantization (activation quantization is harder and less commonly applied)

---

## Q9: Auto-scaling for AI workloads — GPUs are not CPUs

**The problem:** Standard CPU autoscaling triggers on CPU utilization. GPU workloads have different saturation signals. A GPU can be 90% utilized while requests queue at 10× the acceptable latency. Or it can show 50% "utilization" while a memory allocation failure serializes everything. CPU-based autoscaling will under-scale until the system is already in trouble.

**The core insight:** Scale on the signal that predicts degradation, not the signal that reports it after the fact. For LLM serving, that's queue depth, pending token backlog, and p95 TTFT — not GPU utilization alone.

**The mechanics:**

Metrics to scale on (in priority order):
1. Request queue depth > threshold → add replicas
2. p95 TTFT > SLO → add replicas
3. Pending tokens (sum of prompt + max_output tokens in queue) > capacity × 0.8 → add replicas
4. GPU memory utilization > 90% → add replicas (fragmentation risk)

Scaling challenges:
- GPU cold start: provisioning an A100, loading model weights (140 GB over NVMe/network), compiling CUDA graphs — 3–10 minutes. This is not acceptable for burst handling.
- Mitigation: min_replicas ≥ 1, warm pools with loaded weights, pre-pulled container images, compiled model artifacts cached

Scale-down hysteresis: don't scale down immediately when load drops. LLM serving has variable request sizes — brief lulls between bursts shouldn't trigger teardown.

**What breaks:**
- Pure CPU-based HPA for GPU workloads
- Setting min_replicas=0 for interactive SLOs — cold starts are unacceptably slow
- Not separating interactive (p95 latency SLO) and batch (throughput SLO) pools

**What the interviewer is testing:** That you know GPU autoscaling is different from CPU autoscaling, and that cold start is the hardest operational challenge.

**Common traps:**
- Scaling only on CPU when GPU is the bottleneck
- Not knowing that model load time (not container start) dominates GPU cold start
- Ignoring batch vs interactive traffic separation

---

## Q10: Load balancing for AI serving — request-aware routing

**The problem:** Round-robin load balancing sends request #1 to server A, #2 to B, #3 to C. Request #1 has a 10,000-token prompt. Requests #2 and #3 have 50-token prompts. Server A is pinned for 10 seconds processing the long prompt while B and C are idle after 200ms. Naive round-robin creates systematic imbalance for variable-length LLM requests.

**The core insight:** Route based on current server load, not request order. For LLMs, load is measured in pending tokens (sum of context + max_output for all in-flight requests), not request count.

**The mechanics:**

Strategies (in order of effectiveness):
1. **Least pending tokens:** route to replica with lowest pending token count. Requires token counting at the gateway, but gives the best load distribution.
2. **Least active connections:** proxies for load when token counting is unavailable.
3. **Weighted round-robin:** adjust weights based on replica capacity (e.g., 2× weight for A100 vs L40S).
4. **Geography-aware:** route to nearest datacenter for latency; fallback on overload.

Gateway responsibilities: health checks (not just TCP — test generation endpoint), circuit breakers (stop sending to a replica that's returning errors), rate limiting (token-bucket per tenant), and TLS termination.

**What breaks:**
- Pure round-robin with variable-length requests (systematic load imbalance)
- No health checks: routing to a replica that loaded the wrong model or has OOM errors
- Sticky sessions for stateless serving: usually wrong; state lives in the request, not the server

**What the interviewer is testing:** That you know round-robin is inappropriate for LLMs and can describe a better routing strategy.

**Common traps:**
- Assuming round-robin is fine because "it's stateless"
- Not thinking about long prompts tying up one replica
- Conflating model routing (which model to use) with load balancing (which replica of the same model)

---

## Q11: GPU memory management for multiple models

**The problem:** You have 5 fine-tuned models for different use cases and 4 GPUs. All 5 models don't fit simultaneously. If you load/unload models per request, each swap costs 30–60 seconds for a 7B model over NVMe. The system becomes unusable.

**The core insight:** Static footprint (weights) and dynamic footprint (KV cache) are independent. Separate the planning. For weights: identify which models are "hot" (high QPS) and pin them. For KV: it's dynamic — plan per-request based on context length.

**The mechanics:**

Memory allocation hierarchy:
1. **Dedicated replicas per hot model:** pin the most-used model(s) to specific GPUs. No swapping.
2. **Shared GPU for cold models:** use LRU eviction. Load model on first request, hold in VRAM until memory pressure forces eviction.
3. **Quantization:** smaller models free memory for coexistence. 7B INT4 = 3.5 GB, enabling 10+ models on a single 80 GB GPU (weights only; KV cache is still dynamic).
4. **MIG (Multi-Instance GPU):** partition an A100 into up to 7 isolated instances. Each instance gets dedicated VRAM, compute, and bandwidth. Useful for strict isolation between tenants.

Routing must know which model is loaded: gateway tracks model placement and routes requests to the appropriate GPU pool.

**What breaks:**
- Forgetting KV cache when colocating models — the models fit in VRAM at zero requests, but KV pressure evicts models mid-request
- LRU eviction without request-ahead loading — first request after eviction is slow
- MIG partitions are fixed at setup — can't be resized per traffic pattern dynamically

**What the interviewer is testing:** The weights/KV separation, the strategy for hot vs cold models, and that swapping has a real performance cost.

**Common traps:**
- Ignoring KV footprint when calculating "fits on the GPU"
- Assuming MIG is always appropriate — it adds operational complexity and fixed capacity allocation
- Not having a routing layer that knows model placement

---

## Q12: Model sharding — when and which kind

**The problem:** Training a new 100B model. Adam optimizer stores 2 copies of gradients and 2 of optimizer states for every parameter. A 100B model at FP32 = 400 GB just for weights. Add optimizer states: 1.6 TB. No single GPU cluster node has this. You need to distribute not just parameters but the entire training state.

**The core insight:** Sharding means each device holds only a slice of the state. The question is which state to shard and how to reconstruct what's needed for each forward/backward pass.

**The mechanics — the training sharding family:** FSDP shards parameters, gradients, and optimizer states across DP ranks (AllGather before each forward, ReduceScatter gradients in backward), giving O(model_size/num_gpus) memory per GPU. DeepSpeed ZeRO shards progressively more state per stage (1: optimizer state, 2: +gradients, 3: +parameters ≈ FSDP), with ZeRO-Infinity offloading to CPU/NVMe. TP/PP (tensor/pipeline parallelism) are the inference-serving analogs — FSDP/ZeRO are primarily training strategies. Full mechanics for TP/PP (AllReduce math, pipeline bubble formula) are in [Model Parallelism](16-efficient-llm-deployment.md#7-model-parallelism-when-one-gpu-isnt-enough).

**What breaks:**
- High sharding degree with slow interconnect: AllGather/ReduceScatter communication dominates training time
- ZeRO-3 with many small modules: excessive communication granularity
- Mixing FSDP sharding with non-FSDP modules: gradient/parameter misalignment

**What the interviewer is testing:** That you can distinguish TP, PP, FSDP, and ZeRO, and know when each applies.

**Common traps:**
- Using "sharding" without specifying which type
- Claiming FSDP and ZeRO-3 are the same (they're similar but different implementations)
- Applying training sharding strategies to inference serving

---

## Q13: Request queuing and priority scheduling

**The problem:** You have 10,000 concurrent users but can only process 50 requests at a time. 9,950 requests must queue. A FIFO queue means a 100-token "what's the weather?" request waits behind a 10,000-token document analysis job. That user's experience is terrible. Meanwhile your paid enterprise customer is waiting behind free-tier batch jobs.

**The core insight:** Queues are not just buffers — they're scheduling decisions. Every queuing policy embeds a priority model. FIFO says "arrival time is the only priority." You almost never actually want that. Design the priority model explicitly.

**The mechanics:**

Priority classes (example):
1. Streaming interactive (latency SLO < 200ms TTFT)
2. Synchronous API calls (latency SLO < 2s)
3. Batch processing (throughput SLO, no latency SLO)

Per class: token-bucket rate limiting per tenant. A free-tier tenant gets 100K tokens/minute. An enterprise tenant gets 10M tokens/minute. Exceeding budget → queue, not fail.

Admission control:
- When queue depth > max_queue_depth: reject new requests with 429 + Retry-After
- When pending tokens > GPU_memory × 0.9: backpressure — stop admitting until KV memory clears
- Never unbounded queues: they hide memory pressure and cause OOM under burst

Scheduling within a priority class:
- Shortest-job-first (predict output length): minimizes mean wait time
- Weighted fair queuing: prevent starvation of low-priority tenants

**What breaks:**
- Unbounded queue: requests accumulate, memory usage grows, OOM kills the server
- Pure FIFO with long requests: short interactive requests starve
- No backpressure: load balancer keeps sending requests that can't be served, cascading failure

**What the interviewer is testing:** That you know queuing is a design decision, not a default. That backpressure and admission control are necessary for stability.

**Common traps:**
- Implementing FIFO and calling it "fair"
- Not limiting queue depth (unbounded queue = OOM vulnerability)
- Not distinguishing rate limiting (per-tenant) from admission control (global capacity)

---

## Q14: Self-hosted vs API — the TCO decision

**The problem:** You're spending $50K/month on OpenAI API calls. A colleague says "we could run this ourselves for $20K/month in GPU costs." Is that true? What did they miss?

**The core insight:** API cost is per-token. Self-hosting cost is fixed (GPU + ops + engineering) plus variable (electricity, cloud compute). The break-even depends on volume, utilization, and how much you value engineer time and operational risk.

**The mechanics:**

Full TCO comparison:
```
TCO_api = (tokens_in × rate_in + tokens_out × rate_out) × months + integration_eng

TCO_self = GPU_cost_or_rental + (infra_eng_FTE × salary) + (on_call_burden)
         + data_egress + model_ops + incident_cost + opportunity_cost_of_eng_time
```

Break-even volume: typically at $200–500K/year API spend, self-hosting becomes competitive if you have GPU utilization > 60% and operational expertise.

Critical factors often missed:
- GPU utilization: idle GPUs burn money. A 40% utilized GPU cluster costs 2.5× per token vs a busy one.
- Engineering time: 1–2 engineers full-time maintaining GPU infra vs zero for API.
- On-call burden: GPU failures, CUDA OOM, model crashes at 3am.
- Flexibility: self-hosted locks you to one model; API lets you switch cheaply.

Decision framework:
- Data residency / compliance requirements → self-host (no choice)
- High volume + stable workload + GPU ops expertise → evaluate self-host
- Rapid iteration / uncertain volume / small team → API + gateway
- Best of both: API for general traffic, self-host for private/high-volume paths

**What breaks:**
- Comparing API price × volume vs GPU hardware cost only (ignoring ops)
- Assuming 100% GPU utilization in self-host projections
- Ignoring vendor lock-in risk in API-first projections

**What the interviewer is testing:** That you can do a realistic TCO including engineering and ops burden.

**Common traps:**
- Recommending self-host without accounting for staffing cost
- Not mentioning hybrid as a common real-world answer

---

## Q15: Cold start latency in serverless AI

**The problem:** Serverless scales to zero when idle, then spins up instances on demand. For a web server, cold start is 100–500ms (container start + process init). For an LLM serving node, cold start is 3–10 minutes: provision GPU instance + pull container image (10+ GB) + load model weights (35–140 GB from NVMe/network) + compile CUDA kernels. This is completely incompatible with interactive latency SLOs.

**The core insight:** The cold start problem for LLMs is fundamentally about model load time, not container startup. You need to either prevent scale-to-zero (min_instances > 0) or pre-stage everything that takes time.

**The mechanics:**

Component breakdown of LLM cold start:
```
GPU instance provision: 60–300s (cloud-dependent)
Docker image pull: 30–120s (10–20 GB image)
Model weights load to VRAM: 30–120s (35–140 GB over NVMe or network)
CUDA kernel compilation: 30–60s (first run after weight load)
Total: 3–10 minutes
```

Mitigation strategies:
1. **min_instances ≥ 1:** never scale to zero for interactive SLOs. Accept baseline cost.
2. **Pre-warmed pools:** keep N instances ready with weights loaded, not serving.
3. **NVMe-cached weights:** store model on local NVMe (not network), eliminate network load time.
4. **CUDA graph compilation cache:** pre-compile and cache, avoid recompilation on startup.
5. **Smaller quantized models:** INT4 weights load in 25% of the time of FP16.
6. **Async first request:** accept the request, queue it, complete cold start, serve. Acceptable for async workloads.

**What breaks:**
- Scale-to-zero for interactive chat: users see 3–10 minute first-response latency
- Over-provisioning warm pools: expensive for low-traffic applications
- Not caching compiled CUDA graphs: repeated compilation on every restart

**What the interviewer is testing:** Understanding that LLM cold start is model-load-dominated, not container-start-dominated.

**Common traps:**
- Assuming serverless = zero cold start with the right settings
- Not knowing that CUDA graph compilation adds substantial startup time
- Treating cold start as a solvable problem rather than a managed trade-off

---

## Q16: Model caching to eliminate redundant computation

**The problem:** 80% of your requests start with the same 2000-token system prompt. You're running a 70B model and the prefill cost for 2000 tokens is significant. At 1000 requests/hour, you're computing the same KV tensors 1000 times/hour. That's pure waste.

**The core insight:** Computation that produces the same result from the same input can be cached at any level: the final response, the KV state, the embedding, or the compiled graph.

**The mechanics — the cache hierarchy:**

1. **Exact response cache:** hash(model_ver + prompt_ver + normalized_input + decoder_params) → stored response. Works when input determinism is acceptable. Key must include temperature/max_tokens or you get wrong cache hits.

2. **Prefix KV cache (vLLM, SGLang RadixAttention):** store computed K/V tensors for common prefixes. Subsequent requests that share the prefix skip prefill for those tokens. Especially effective for: system prompts, few-shot examples, RAG context templates.

3. **Embedding cache:** store embedding vectors for repeated query strings. Saves embedding model inference.

4. **Semantic cache:** embed queries, find nearest cached query (cosine similarity > threshold). Return cached response if similar enough. Risk: semantically similar ≠ identical intent.

Cache key discipline: include every parameter that affects output — model version, prompt version, all decoder params (temperature, top_p, max_tokens, stop sequences). A missing key element causes stale or incorrect cache hits.

Privacy: never cache PII in shared caches. Use per-user partitioning or encrypt cache keys.

**What breaks:**
- Cache key missing temperature: deterministic (temp=0) and stochastic (temp=0.7) responses share a key — wrong
- Semantic cache false positives: "What's Python?" and "What's COBOL?" might be too similar by embedding
- Prefix KV cache invalidation: any change to the system prompt invalidates all cached KV blocks

**What the interviewer is testing:** The hierarchy (response → KV → embedding → semantic) and the cache key discipline.

**Common traps:**
- Treating all caches as equivalent (response cache and KV cache are very different)
- Missing decoder params in cache keys
- Caching PII without access controls

---

## Q17: Synchronous vs asynchronous inference

**The problem:** Your LLM generates long documents — 2000–5000 tokens. At 50 tokens/second, that's 40–100 seconds of generation. HTTP timeouts at gateways and clients are typically 30–60 seconds. Synchronous serving for long jobs breaks before they complete.

**The core insight:** When expected latency exceeds client timeout budget, the request model must decouple submission from completion. Asynchronous inference accepts the work and returns a reference; the client polls or is notified when ready.

**The mechanics:**

Synchronous (appropriate for interactive, < 30s):
```
POST /generate → [GPU runs, client waits] → 200 + full response body
```

Streaming (synchronous connection, incremental delivery):
```
POST /generate → [GPU runs] → SSE/WebSocket stream of tokens → final token + 200
```
Streaming is still synchronous in connection terms but dramatically improves perceived latency. Users see the first token within TTFT instead of waiting for full completion.

Asynchronous (appropriate for long generation, batch jobs, queue-based processing):
```
POST /jobs → 202 + {job_id: "abc123"}
GET /jobs/abc123 → 202 + {status: "running"}
GET /jobs/abc123 → 200 + {status: "complete", result: ...}
```
Or webhook: POST /jobs → 202; POST to callback_url when complete.

**What breaks:**
- Sync long jobs with 30s gateway timeouts: requests fail mid-generation
- Async without job cleanup: job state accumulates, storage fills
- Streaming without backpressure: if client disconnects, GPU continues generating wasted tokens

**What the interviewer is testing:** Understanding when streaming vs async is appropriate, and the operational implications of each.

**Common traps:**
- Treating streaming as "async" — it's still a synchronous HTTP connection
- Using sync for document generation without timeout planning
- No TTL on async job state

---

## Q18: Monitoring and profiling LLM inference — separating the phases

**The problem:** Your p95 latency is 4 seconds. Without knowing where time is spent, you don't know what to fix. Is it queue wait? Prefill? Decode? KV memory pressure? Each has a different solution. Profiling total latency obscures the bottleneck.

**The core insight:** LLM inference has two fundamentally different computational phases with different bottlenecks. Prefill is compute-bound (runs once, processes all prompt tokens in parallel). Decode is memory-bandwidth-bound (runs N times, loads all weights once per token). Measure them separately.

**The mechanics:**

Metrics hierarchy (from outermost to innermost):
```
end_to_end_latency = queue_wait + prefill_time + (decode_steps × tpot) + output_parse_time

where:
- queue_wait = time from request arrival to GPU processing start
- prefill_time = time to process all prompt tokens (compute-bound)
- tpot = time per output token during decode (memory-bandwidth-bound)
- TTFT = queue_wait + prefill_time
```

What each metric indicates:
- High queue_wait → request backlog, need more replicas or better load balancing
- High prefill_time → long prompts or compute bottleneck; consider prompt compression
- High TPOT → memory bandwidth saturated; consider quantization or larger batch
- KV cache pressure → high fragmentation; check PagedAttention block allocation stats

GPU-level metrics:
- DCGM: GPU utilization, memory used, memory bandwidth utilization, PCIe/NVLink bandwidth
- PyTorch Profiler: per-operation timing, kernel launch overhead, CUDA stream serialization
- Nsight Systems: GPU timeline, kernel concurrency, memory copy overhead

Dashboard targets:
- TTFT p50/p95/p99
- TPOT p50/p95/p99 (inter-token latency)
- Tokens/sec throughput
- GPU utilization (target > 80%)
- KV cache block utilization
- Queue depth

**What breaks:**
- Measuring only total latency: hides whether prefill or decode is the bottleneck
- Alerting only on mean latency: p99 can be 5× mean under realistic load distributions
- No per-request tracing: can't reproduce failures or attribute latency spikes

**What the interviewer is testing:** The prefill/decode distinction, TTFT vs TPOT, and that effective profiling requires instrumenting the pipeline stages separately.

**Common traps:**
- Measuring only total latency without phase breakdown
- Conflating p50 and p99 — LLM latency distributions have heavy tails
- Not tracking KV cache pressure as a leading indicator of throughput collapse

---

## Q19: Model routing at the infrastructure level

**The problem:** You have a single endpoint that routes everything to your most capable (and expensive) model. 40% of requests are simple one-turn factual lookups. 30% are complex multi-step reasoning. 20% are code generation. 10% are safety-critical compliance checks. Routing everything to the large model costs 3× what it needs to, and the simple factual lookups are competing for capacity with the complex tasks.

**The core insight:** Not every request has the same capability requirement. A routing layer that matches requests to the minimum-capable model that satisfies the requirement reduces cost and improves throughput without degrading quality for requests that don't need it.

**The mechanics:**

Router signals (in order of reliability):
1. **Explicit task type:** client sends task classification (most reliable, no inference needed)
2. **Input features:** token count, presence of code blocks, detected language
3. **Small classifier model:** lightweight model predicts complexity or task type from input
4. **Confidence-based cascade:** route to small model; if output confidence < threshold, re-route to large model

Cascade routing:
```python
def route(request):
    if request.task_type == "factual_lookup" and request.token_count < 200:
        return "small_fast_model"
    if request.requires_code_execution:
        return "code_specialized_model"
    if request.safety_classification == "high_risk":
        return "safety_tuned_model"
    return "general_large_model"
```

Cost impact: routing 40% of requests to a model 10× cheaper reduces average cost by 36% with no quality regression on those requests — if the classification is accurate.

**What breaks:**
- Classifier latency adds to TTFT — must be sub-10ms or route async
- Misclassification: routing a complex reasoning task to the small model, then having to re-run → 2× cost
- Quality regression monitoring: must track quality per route, not just average

**What the interviewer is testing:** That routing is a cost optimization with quality implications, requiring careful evaluation of the routing policy.

**Common traps:**
- Routing only by user tier or request metadata, ignoring content complexity
- Not A/B testing the routing policy against a baseline
- Forgetting to monitor quality separately per route (degradation is invisible in aggregate metrics)
