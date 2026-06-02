---
module: Llms
topic: Interview Notes
subtopic: Ai Infrastructure And Scalability
status: unread
tags: [llms, ml, interview-notes-ai-infrastruct]
---
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

## Q3: Data parallelism vs model parallelism — why the distinction matters

**The problem:** Your model is 70B parameters. An A100 80GB holds 35 GB of INT4 weights. After KV cache, you have no room for activations. You can't fit even one replica on one GPU. Data parallelism — which assumes each worker holds a complete model — is not an option.

**The core insight:** Data parallelism (DP) copies the model to every device and splits data. It only works when one device can hold the full model. Model parallelism splits the model itself across devices — required when the model is too large for any single device.

**The mechanics:**

Data parallelism:
- Each GPU holds identical full model weights
- Each GPU processes a different batch of data
- Gradients are all-reduced (averaged) across replicas
- Memory per GPU = full model weights + activations for local batch

Model parallelism — two forms:
- **Tensor parallelism (TP):** Split individual weight matrices across GPUs. GPU 0 holds columns 0..d/2, GPU 1 holds columns d/2..d. AllReduce after each layer.
- **Pipeline parallelism (PP):** Assign consecutive layers to different GPUs. GPU 0 does layers 0–19, GPU 1 does layers 20–39. Activations pass between stages.

Production LLMs: DP × TP × PP ("3D parallelism"). Example: 8-node cluster, each node has 8 GPUs — TP=8 within a node, DP×PP across nodes.

**What breaks:**
- TP communication dominates at small batch sizes (AllReduce per layer adds latency)
- PP creates pipeline bubbles — GPU 0 is idle while downstream GPUs process
- Combining both naively leads to communication storms on slow interconnects

**What the interviewer is testing:** That you understand when each strategy is appropriate and that production systems combine them.

**Common traps:**
- Using "model parallelism" to mean only TP (PP is also model parallel)
- Assuming TP always helps — at TP degree > 8 on NVLink, communication often hurts
- Forgetting that DP requires each device to hold a complete model replica

---

## Q4: Tensor parallelism — splitting within a layer

**The problem:** A single attention layer in a 70B model has QKV projection matrices of shape [8192 × 8192]. In FP16, that's 128 MB per matrix. With 80 such layers, just the QKV projections are 10 GB. This must fit on one GPU before you've counted anything else.

**The core insight:** Matrix multiplication is splittable. If W = [W₁ | W₂], then XW = [XW₁ | XW₂]. GPU 0 can hold W₁ and compute XW₁; GPU 1 holds W₂ and computes XW₂. The results are concatenated via AllReduce. Each GPU does half the work and holds half the memory.

**The mechanics (Megatron-style):**

For a linear layer Y = XW:
- Column-split: GPU i holds W[:, i*k:(i+1)*k]. Compute local XWᵢ. AllGather outputs.
- Row-split: GPU i holds W[i*k:(i+1)*k, :]. Compute local XᵢW. AllReduce sums.

For attention heads: distribute heads across GPUs. GPU 0 computes heads 0–15, GPU 1 computes heads 16–31. No communication until after head concatenation.

Communication cost: one AllReduce per Transformer block (2 total: after attention and after MLP). At TP=8 with NVLink, this is ~1–2ms per layer. On PCIe: 8–15ms — often slower than sequential execution.

**What breaks:**
- TP=8 on PCIe is usually slower than TP=2 due to communication overhead
- TP requires all GPUs to participate in every forward pass — one slow GPU stalls the rest
- Higher TP degrees reduce per-GPU memory but increase communication frequency

**What the interviewer is testing:** The communication-memory trade-off. Why TP is only practical with high-bandwidth interconnects (NVLink 600 GB/s vs PCIe 64 GB/s).

**Common traps:**
- Claiming TP always speeds up inference
- Confusing TP (splits tensors within a layer) with PP (splits layers)
- Not knowing the AllReduce cost at different batch sizes

---

## Q5: Pipeline parallelism — splitting across layers

**The problem:** Your model has 80 layers but you have 4 GPUs. TP would require splitting every matrix by 4×, with 4× AllReduce communication overhead per layer. An alternative: assign layers 0–19 to GPU 0, layers 20–39 to GPU 1, etc. Each GPU holds 25% of the model parameters and does 25% of the computation. No communication within a layer — only activations pass between stages.

**The core insight:** Depth is independent. The output of layer 19 is just a tensor — it can be passed from GPU 0 to GPU 1 as a memory copy. This enables much lower communication bandwidth requirements than TP, at the cost of pipeline bubbles.

**The mechanics:**

Forward pass:
```
GPU 0: process layers 0–19 → send activation to GPU 1
GPU 1: process layers 20–39 → send activation to GPU 2
...
```

The bubble problem: while GPU 1 is processing, GPU 0 is idle. With 4 stages, the bubble fraction is approximately (P-1)/P = 75% idle for a single microbatch.

GPipe/PipeDream fix: microbatching. Split the batch into M microbatches. While GPU 1 processes microbatch 1, GPU 0 processes microbatch 2. Bubble fraction becomes (P-1)/(M+P-1), which approaches 0 as M increases.

**What breaks:**
- Small batch sizes amplify bubble time — PP is much less efficient at low concurrency
- More stages = more pipeline latency even when GPUs are busy
- Load imbalance: if layer 40 is twice as expensive as layer 20, GPU 2 becomes the bottleneck

**What the interviewer is testing:** Bubble awareness. That you know PP reduces memory per device but adds latency and requires careful scheduling.

**Common traps:**
- Claiming PP always reduces latency — it often increases it for single requests
- Confusing PP (splits by layer depth) with TP (splits tensor width)
- Not mentioning microbatching as the mitigation for bubbles

---

## Q6: Continuous batching — why static batching leaves GPUs idle

**The problem:** In the 10,000-concurrent-users scenario, requests have wildly different lengths. Request A generates 10 tokens. Request B generates 500. Static batching waits for all requests in a batch to finish before admitting new ones. When Request A finishes at step 10, its GPU slot sits empty until Request B finishes at step 500. For 490 steps, you're using 1/batch_size of your GPU capacity.

**The core insight:** Autoregressive generation is iteration-level, not request-level. Each "step" is a full forward pass over the current batch. There's nothing stopping you from swapping a finished sequence out and a new one in at any step boundary.

**The mechanics — iteration-level scheduling:**

```
Step  1: [A: tok1] [B: tok1] [C: tok1] [D: tok1]
Step  2: [A: tok2] [B: tok2] [C: tok2] [D: tok2]
Step 10: [A: DONE] [B:tok10] [C: tok10] [D: tok10]
          └→ admit E immediately
Step 11: [E: tok1] [B: tok11] [C: tok11] [D: tok11]
```

Each step: check if any sequences finished; free their KV blocks; admit waiting requests up to memory capacity; run one forward pass over all active sequences.

Throughput improvement over static batching: 5–10× for typical chat workloads with mixed-length responses.

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

## Q7: Speculative decoding — parallelizing the sequential bottleneck

**The problem:** Autoregressive decoding is fundamentally sequential. Token N cannot be generated until token N-1 is complete. Each step requires a full forward pass through the 70B model. For long outputs, this is the dominant cost. You cannot simply "run it in parallel" — each token depends on the last.

**The core insight:** You can guess the next several tokens cheaply with a small model, then verify all guesses simultaneously with one forward pass of the large model. Verification is parallelizable because the large model can evaluate all positions at once given the draft tokens.

**The mechanics:**

1. Draft model (e.g., 7B) generates γ candidate tokens autoregressively: t₁, t₂, ..., tᵧ
2. Target model (70B) evaluates all γ positions in one parallel forward pass
3. Accept token tᵢ with probability min(1, p_target(tᵢ) / p_draft(tᵢ))
4. At the first rejection, resample from p_target(t) - p_draft(t) (normalized)
5. Repeat

Key property: the output distribution is **identical to sampling from the target model alone**. No quality loss.

Speedup depends on acceptance rate α. With α ≈ 0.8 and γ = 4, expected accepted tokens per large-model forward pass ≈ 1/(1-α) ≈ 3.2. Throughput gain: 2–3×.

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

## Q8: KV cache — why you can't regenerate past context every step

**The problem:** At decode step N, the model needs to compute attention between the new token (query) and every previous token (keys and values). Recomputing K and V for all previous tokens at every step costs O(N²) time per request. With sequence length 4096, that's 16M operations per step, repeated 4096 times — 64B total operations just for attention.

**The core insight:** K and V for past tokens never change after they're computed. Cache them. Each new token only needs to compute its own Q, K, V, then attend to the cached K and V from all prior tokens.

**The mechanics:**

Memory cost per sequence:
```
KV_bytes = 2 × L × H_kv × d_h × T × bytes_per_element
```
For LLaMA 3 70B (80 layers, 8 KV heads via GQA, 128 head dim), BF16, 4096 tokens:
```
= 2 × 80 × 8 × 128 × 4096 × 2 = 1.34 GB per sequence
```

At batch=16: 21.4 GB. At batch=32: 42.8 GB. KV cache dominates VRAM at scale.

GQA (Grouped Query Attention): LLaMA 3 uses 8 KV heads vs 64 query heads. KV cache is 8× smaller than if using full multi-head attention.

Management strategies:
- **Limit max_seq_len and batch_size:** hard cap on KV memory
- **PagedAttention:** don't pre-allocate for max length; allocate blocks on demand
- **KV quantization (INT8/FP8):** halve or quarter KV memory at some quality cost
- **Prefix caching:** if many requests share a system prompt, compute its KV once and reuse

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

## Q9: PagedAttention and vLLM — virtual memory for KV cache

**The problem:** Naive KV cache pre-allocates a contiguous block of memory for each request sized at max_seq_len. If max_seq_len = 8192, you reserve 8192-token worth of KV memory even for a request that generates 50 tokens. For a batch of 64 requests, 60–80% of KV memory is wasted. This fragmentation is the primary reason throughput collapses at scale.

**The core insight:** OS virtual memory solved this exact problem for RAM decades ago. Pages. Allocate fixed-size blocks (pages) only when needed. Maintain a page table mapping logical KV positions to physical memory blocks. Requests that finish release their pages immediately for reuse.

**The mechanics:**

PagedAttention (vLLM):
- Divide KV cache into fixed-size blocks, e.g., 16 tokens per block
- Each sequence has a page table: logical token position → physical block index
- Allocate one block at a time as generation proceeds
- On request completion, mark blocks as free for reuse
- Sequences sharing a prefix (same system prompt) can share physical blocks (copy-on-write)

```
Sequence A: [block_42][block_7][block_31]    (tokens 0–15, 16–31, 32–47)
Sequence B: [block_42][block_19][block_8]    (shared prefix in block_42)
```

Result: 3–4× higher batch capacity vs naive contiguous allocation, 24× higher throughput vs HuggingFace Transformers naive batching on equivalent hardware.

vLLM combines PagedAttention + continuous batching + an OpenAI-compatible API server. It's a serving runtime, not a model.

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

## Q10: Edge and mobile deployment — everything changes at the constraint boundary

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

## Q11: Quantization — trading precision for memory and speed

**The problem:** LLaMA 3 70B in FP16 requires 140 GB — two A100 80GB GPUs just for weights. INT4 quantization reduces this to 35 GB, fitting on one GPU with room for KV cache. But you're now representing each weight with 4 bits instead of 16. When does that cause accuracy problems, and when is it acceptable?

**The core insight:** Most of a model's weights lie in a smooth distribution — they're "close" to their quantized values. The exceptions are outlier weights with large magnitudes. The entire game of good quantization is: protect the important weights (large activations, sensitive layers) from precision loss.

**The mechanics:**

PTQ (Post-Training Quantization) — no retraining:

Symmetric INT8: q = round(w / s), s = max(|w|) / 127. Dequantize: ŵ = q × s

GPTQ: quantize layer by layer using second-order curvature (inverse Hessian) to propagate and minimize quantization error. Works well at 4-bit for most transformer weights.

AWQ (Activation-Aware Weight Quantization): observe which weights correspond to high-magnitude activations (important for output), scale those weights before quantizing to protect them. Better quality than GPTQ on average.

NF4 (bitsandbytes): uses a normal float 4-bit format that better matches the bell-curve distribution of transformer weights. Best for 4-bit with QLoRA/fine-tuning scenarios.

Quality vs memory trade-off:
| Format | VRAM (70B) | Quality cost |
|---|---|---|
| FP16 | 140 GB | Reference |
| INT8 | 70 GB | ~0.2 perplexity increase |
| AWQ 4-bit | 35 GB | ~0.4 perplexity increase |
| INT4 naive | 35 GB | 1–2+ perplexity increase |

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

## Q12: Auto-scaling for AI workloads — GPUs are not CPUs

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

## Q13: Load balancing for AI serving — request-aware routing

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

## Q14: GPU memory management for multiple models

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

## Q15: Model sharding — when and which kind

**The problem:** Training a new 100B model. Adam optimizer stores 2 copies of gradients and 2 of optimizer states for every parameter. A 100B model at FP32 = 400 GB just for weights. Add optimizer states: 1.6 TB. No single GPU cluster node has this. You need to distribute not just parameters but the entire training state.

**The core insight:** Sharding means each device holds only a slice of the state. The question is which state to shard and how to reconstruct what's needed for each forward/backward pass.

**The mechanics — the training sharding family:**

FSDP (PyTorch Fully Sharded Data Parallel):
- Shards parameters, gradients, and optimizer states across DP ranks
- AllGather parameters before each layer's forward pass, free after
- ReduceScatter gradients during backward
- Memory per GPU: O(model_size / num_gpus) instead of O(model_size)

DeepSpeed ZeRO (Zero Redundancy Optimizer):
- Stage 1: shard optimizer states only. Each GPU holds full weights, sharded optimizer state.
- Stage 2: + shard gradients. Reduced gradient memory.
- Stage 3: + shard parameters (equivalent to FSDP). Minimum memory.
- ZeRO-Infinity: offload to CPU/NVMe for extreme scale.

For inference: TP/PP as described in Q4/Q5. FSDP and ZeRO are primarily training strategies.

```
Choosing shard topology:
- Training, PyTorch ecosystem → FSDP
- Training, want CPU/NVMe offload → DeepSpeed ZeRO-3 / ZeRO-Infinity
- Inference, single node → TP (fast interconnect required)
- Inference, multi-node → TP within node + PP across nodes
```

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

## Q16: Request queuing and priority scheduling

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

## Q17: Self-hosted vs API — the TCO decision

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

## Q18: Cold start latency in serverless AI

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

## Q19: Model caching to eliminate redundant computation

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

## Q20: Synchronous vs asynchronous inference

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

## Q21: FSDP vs DeepSpeed ZeRO — the training memory optimizers

**The problem:** Training a 70B model. Adam optimizer requires 4× model size for optimizer states (FP32 master weights + momentum + variance). That's 4 × 280 GB = 1.12 TB of optimizer state alone. This must be distributed across GPUs. DP doesn't help — each replica would need the full 1.12 TB. You need to shard the optimizer state.

**The core insight:** ZeRO (Zero Redundancy Optimizer) eliminates the memory redundancy of data parallelism by sharding optimizer state, gradients, and parameters across data-parallel ranks. Each GPU holds 1/N of the total state. FSDP is PyTorch's native implementation of the same idea.

**The mechanics:**

ZeRO stages:
- Stage 1: shard optimizer states across DP ranks. Each GPU holds full weights and gradients, but only 1/N of optimizer state. Memory: ~60% of full DP.
- Stage 2: + shard gradients. Memory: ~33% of full DP.
- Stage 3: + shard parameters (equivalent to FSDP). Memory: O(1/N) per GPU.

Communication pattern for ZeRO-3 / FSDP:
- Forward pass: AllGather parameters before each layer, free after layer completes
- Backward pass: ReduceScatter gradients after each layer
- Optimizer step: each rank updates its local shard; AllGather to reconstruct for next forward pass

FSDP vs DeepSpeed ZeRO-3:
- FSDP: native PyTorch 2.0+, tight integration with `torch.compile`, better for HuggingFace-based training
- DeepSpeed ZeRO-3: more features (CPU/NVMe offload, ZeRO-Infinity), custom launcher, more configuration options
- Both achieve similar memory reduction at equivalent sharding levels

**What breaks:**
- High ZeRO/FSDP sharding with slow interconnect: AllGather for each layer dominates training time
- ZeRO-3 with PyTorch compile: extra work needed for compatibility
- Not using activation checkpointing alongside sharding: activations can still be a memory bottleneck

**What the interviewer is testing:** The staged sharding concept, which state is sharded at each stage, and practical ecosystem tradeoffs.

**Common traps:**
- Saying FSDP is not related to ZeRO (FSDP ≈ ZeRO-3)
- Not knowing CPU offload exists (ZeRO-Infinity)
- Conflating training sharding with inference serving strategies

---

## Q22: Monitoring and profiling LLM inference — separating the phases

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

## Q23: Model routing at the infrastructure level

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

## Rapid Recall

### Listing only training tricks when the question is about serving
- Direct Answer: Listing only training tricks when the question is about serving
- Why: This matters because it tells you how to reason about listing only training tricks when the question is about serving.
- Pitfall: Don't answer "Listing only training tricks when the question is about serving" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Listing only training tricks when the question is about serving

### Treating quantization and batching as alternatives rather than orthogonal
- Direct Answer: Treating quantization and batching as alternatives rather than orthogonal
- Why: This matters because it tells you how to reason about treating quantization and batching as alternatives rather than orthogonal.
- Pitfall: Don't answer "Treating quantization and batching as alternatives rather than orthogonal" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Treating quantization and batching as alternatives rather than orthogonal

### Not distinguishing TTFT (prefill-bound) from TPOT (decode-bound)
- Direct Answer: Not distinguishing TTFT (prefill-bound) from TPOT (decode-bound)
- Why: This matters because it tells you how to reason about not distinguishing ttft (prefill-bound) from tpot (decode-bound).
- Pitfall: Don't answer "Not distinguishing TTFT (prefill-bound) from TPOT (decode-bound)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not distinguishing TTFT (prefill-bound) from TPOT (decode-bound)

### Buying "most FLOPs" GPU when decode is memory-bandwidth-limited
- Direct Answer: Buying "most FLOPs" GPU when decode is memory-bandwidth-limited
- Why: This matters because it tells you how to reason about buying "most flops" gpu when decode is memory-bandwidth-limited.
- Pitfall: Don't answer "Buying "most FLOPs" GPU when decode is memory-bandwidth-limited" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Buying "most FLOPs" GPU when decode is memory-bandwidth-limited

### Under-provisioning VRAM by forgetting KV cache grows with batch × context
- Direct Answer: Under-provisioning VRAM by forgetting KV cache grows with batch × context
- Why: This matters because it tells you how to reason about under-provisioning vram by forgetting kv cache grows with batch × context.
- Pitfall: Don't answer "Under-provisioning VRAM by forgetting KV cache grows with batch × context" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Under-provisioning VRAM by forgetting KV cache grows with batch × context

### Using PCIe-only topology for high tensor-parallelism degrees
- Direct Answer: communication dominates
- Why: This matters because it tells you how to reason about using pcie-only topology for high tensor-parallelism degrees.
- Pitfall: Don't answer "Using PCIe-only topology for high tensor-parallelism degrees" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: communication dominates

### Quoting only peak FLOPS without memory bandwidth
- Direct Answer: Quoting only peak FLOPS without memory bandwidth
- Why: This matters because it tells you how to reason about quoting only peak flops without memory bandwidth.
- Pitfall: Don't answer "Quoting only peak FLOPS without memory bandwidth" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Quoting only peak FLOPS without memory bandwidth

### Forgetting that KV cache is dynamic
- Direct Answer: it grows with every request in the batch
- Why: This matters because it tells you how to reason about forgetting that kv cache is dynamic.
- Pitfall: Don't answer "Forgetting that KV cache is dynamic" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: it grows with every request in the batch

### Spec-sheet comparisons without profiling on representative workloads
- Direct Answer: Spec-sheet comparisons without profiling on representative workloads
- Why: This matters because it tells you how to reason about spec-sheet comparisons without profiling on representative workloads.
- Pitfall: Don't answer "Spec-sheet comparisons without profiling on representative workloads" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Spec-sheet comparisons without profiling on representative workloads

### Each GPU holds identical full model weights
- Direct Answer: Each GPU holds identical full model weights
- Why: This matters because it tells you how to reason about each gpu holds identical full model weights.
- Pitfall: Don't answer "Each GPU holds identical full model weights" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Each GPU holds identical full model weights

### Each GPU processes a different batch of data
- Direct Answer: Each GPU processes a different batch of data
- Why: This matters because it tells you how to reason about each gpu processes a different batch of data.
- Pitfall: Don't answer "Each GPU processes a different batch of data" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Each GPU processes a different batch of data

### Gradients are all-reduced (averaged) across replicas
- Direct Answer: Gradients are all-reduced (averaged) across replicas
- Why: This matters because it tells you how to reason about gradients are all-reduced (averaged) across replicas.
- Pitfall: Don't answer "Gradients are all-reduced (averaged) across replicas" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Gradients are all-reduced (averaged) across replicas

### Memory per GPU = full model weights + activations for local batch
- Direct Answer: Memory per GPU = full model weights + activations for local batch
- Why: This matters because it tells you how to reason about memory per gpu = full model weights + activations for local batch.
- Pitfall: Don't answer "Memory per GPU = full model weights + activations for local batch" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Memory per GPU = full model weights + activations for local batch

### Tensor parallelism (TP)
- Direct Answer: Split individual weight matrices across GPUs. GPU 0 holds columns 0..d/2, GPU 1 holds columns d/2..d. AllReduce after each layer.
- Why: This matters because it tells you how to reason about tensor parallelism (tp).
- Pitfall: Don't answer "Tensor parallelism (TP)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Split individual weight matrices across GPUs. GPU 0 holds columns 0..d/2, GPU 1 holds columns d/2..d. AllReduce after each layer.

### Pipeline parallelism (PP)
- Direct Answer: Assign consecutive layers to different GPUs. GPU 0 does layers 0–19, GPU 1 does layers 20–39. Activations pass between stages.
- Why: This matters because it tells you how to reason about pipeline parallelism (pp).
- Pitfall: Don't answer "Pipeline parallelism (PP)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Assign consecutive layers to different GPUs. GPU 0 does layers 0–19, GPU 1 does layers 20–39. Activations pass between stages.

### TP communication dominates at small batch sizes (AllReduce per layer adds latency)
- Direct Answer: TP communication dominates at small batch sizes (AllReduce per layer adds latency)
- Why: This matters because it tells you how to reason about tp communication dominates at small batch sizes (allreduce per layer adds latency).
- Pitfall: Don't answer "TP communication dominates at small batch sizes (AllReduce per layer adds latency)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: TP communication dominates at small batch sizes (AllReduce per layer adds latency)

### PP creates pipeline bubbles
- Direct Answer: GPU 0 is idle while downstream GPUs process
- Why: This matters because it tells you how to reason about pp creates pipeline bubbles.
- Pitfall: Don't answer "PP creates pipeline bubbles" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: GPU 0 is idle while downstream GPUs process

### Combining both naively leads to communication storms on slow interconnects
- Direct Answer: Combining both naively leads to communication storms on slow interconnects
- Why: This matters because it tells you how to reason about combining both naively leads to communication storms on slow interconnects.
- Pitfall: Don't answer "Combining both naively leads to communication storms on slow interconnects" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Combining both naively leads to communication storms on slow interconnects

### Using "model parallelism" to mean only TP (PP is also model parallel)
- Direct Answer: Using "model parallelism" to mean only TP (PP is also model parallel)
- Why: This matters because it tells you how to reason about using "model parallelism" to mean only tp (pp is also model parallel).
- Pitfall: Don't answer "Using "model parallelism" to mean only TP (PP is also model parallel)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Using "model parallelism" to mean only TP (PP is also model parallel)

### Assuming TP always helps
- Direct Answer: at TP degree > 8 on NVLink, communication often hurts
- Why: This matters because it tells you how to reason about assuming tp always helps.
- Pitfall: Don't answer "Assuming TP always helps" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: at TP degree > 8 on NVLink, communication often hurts

### Forgetting that DP requires each device to hold a complete model replica
- Direct Answer: Forgetting that DP requires each device to hold a complete model replica
- Why: This matters because it tells you how to reason about forgetting that dp requires each device to hold a complete model replica.
- Pitfall: Don't answer "Forgetting that DP requires each device to hold a complete model replica" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Forgetting that DP requires each device to hold a complete model replica

### Column-split
- Direct Answer: GPU i holds W[:, ik:(i+1)k]. Compute local XWᵢ. AllGather outputs.
- Why: This matters because it tells you how to reason about column-split.
- Pitfall: Don't answer "Column-split" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: GPU i holds W[:, ik:(i+1)k]. Compute local XWᵢ. AllGather outputs.

### Row-split
- Direct Answer: GPU i holds W[ik:(i+1)k, :]. Compute local XᵢW. AllReduce sums.
- Why: This matters because it tells you how to reason about row-split.
- Pitfall: Don't answer "Row-split" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: GPU i holds W[ik:(i+1)k, :]. Compute local XᵢW. AllReduce sums.

### TP=8 on PCIe is usually slower than TP=2 due to communication overhead
- Direct Answer: TP=8 on PCIe is usually slower than TP=2 due to communication overhead
- Why: This matters because it tells you how to reason about tp=8 on pcie is usually slower than tp=2 due to communication overhead.
- Pitfall: Don't answer "TP=8 on PCIe is usually slower than TP=2 due to communication overhead" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: TP=8 on PCIe is usually slower than TP=2 due to communication overhead

### TP requires all GPUs to participate in every forward pass
- Direct Answer: one slow GPU stalls the rest
- Why: This matters because it tells you how to reason about tp requires all gpus to participate in every forward pass.
- Pitfall: Don't answer "TP requires all GPUs to participate in every forward pass" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: one slow GPU stalls the rest

### Higher TP degrees reduce per-GPU memory but increase communication frequency
- Direct Answer: Higher TP degrees reduce per-GPU memory but increase communication frequency
- Why: This matters because it tells you how to reason about higher tp degrees reduce per-gpu memory but increase communication frequency.
- Pitfall: Don't answer "Higher TP degrees reduce per-GPU memory but increase communication frequency" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Higher TP degrees reduce per-GPU memory but increase communication frequency

### Claiming TP always speeds up inference
- Direct Answer: Claiming TP always speeds up inference
- Why: This matters because it tells you how to reason about claiming tp always speeds up inference.
- Pitfall: Don't answer "Claiming TP always speeds up inference" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Claiming TP always speeds up inference

### Confusing TP (splits tensors within a layer) with PP (splits layers)
- Direct Answer: Confusing TP (splits tensors within a layer) with PP (splits layers)
- Why: This matters because it tells you how to reason about confusing tp (splits tensors within a layer) with pp (splits layers).
- Pitfall: Don't answer "Confusing TP (splits tensors within a layer) with PP (splits layers)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Confusing TP (splits tensors within a layer) with PP (splits layers)

### Not knowing the AllReduce cost at different batch sizes
- Direct Answer: Not knowing the AllReduce cost at different batch sizes
- Why: This matters because it tells you how to reason about not knowing the allreduce cost at different batch sizes.
- Pitfall: Don't answer "Not knowing the AllReduce cost at different batch sizes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not knowing the AllReduce cost at different batch sizes

### Small batch sizes amplify bubble time
- Direct Answer: PP is much less efficient at low concurrency
- Why: This matters because it tells you how to reason about small batch sizes amplify bubble time.
- Pitfall: Don't answer "Small batch sizes amplify bubble time" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: PP is much less efficient at low concurrency

### More stages = more pipeline latency even when GPUs are busy
- Direct Answer: More stages = more pipeline latency even when GPUs are busy
- Why: This matters because it tells you how to reason about more stages = more pipeline latency even when gpus are busy.
- Pitfall: Don't answer "More stages = more pipeline latency even when GPUs are busy" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: More stages = more pipeline latency even when GPUs are busy

### Load imbalance
- Direct Answer: if layer 40 is twice as expensive as layer 20, GPU 2 becomes the bottleneck
- Why: This matters because it tells you how to reason about load imbalance.
- Pitfall: Don't answer "Load imbalance" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: if layer 40 is twice as expensive as layer 20, GPU 2 becomes the bottleneck

### Claiming PP always reduces latency
- Direct Answer: it often increases it for single requests
- Why: This matters because it tells you how to reason about claiming pp always reduces latency.
- Pitfall: Don't answer "Claiming PP always reduces latency" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: it often increases it for single requests

### Confusing PP (splits by layer depth) with TP (splits tensor width)
- Direct Answer: Confusing PP (splits by layer depth) with TP (splits tensor width)
- Why: This matters because it tells you how to reason about confusing pp (splits by layer depth) with tp (splits tensor width).
- Pitfall: Don't answer "Confusing PP (splits by layer depth) with TP (splits tensor width)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Confusing PP (splits by layer depth) with TP (splits tensor width)

### Not mentioning microbatching as the mitigation for bubbles
- Direct Answer: Not mentioning microbatching as the mitigation for bubbles
- Why: This matters because it tells you how to reason about not mentioning microbatching as the mitigation for bubbles.
- Pitfall: Don't answer "Not mentioning microbatching as the mitigation for bubbles" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not mentioning microbatching as the mitigation for bubbles

### Without PagedAttention
- Direct Answer: admitting new requests requires contiguous KV memory that may not be available even if total free memory is sufficient (fragmentation)
- Why: This matters because it tells you how to reason about without pagedattention.
- Pitfall: Don't answer "Without PagedAttention" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: admitting new requests requires contiguous KV memory that may not be available even if total free memory is sufficient (fragmentation)

### Long requests can still monopolize batch slots
- Direct Answer: need preemption or length-based admission
- Why: This matters because it tells you how to reason about long requests can still monopolize batch slots.
- Pitfall: Don't answer "Long requests can still monopolize batch slots" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: need preemption or length-based admission

### p99 tail latency
- Direct Answer: short requests behind a long queue may still wait if admission control isn't length-aware
- Why: This matters because it tells you how to reason about p99 tail latency.
- Pitfall: Don't answer "p99 tail latency" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: short requests behind a long queue may still wait if admission control isn't length-aware

### Thinking batching is only a training concern
- Direct Answer: Thinking batching is only a training concern
- Why: This matters because it tells you how to reason about thinking batching is only a training concern.
- Pitfall: Don't answer "Thinking batching is only a training concern" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Thinking batching is only a training concern

### Not knowing that continuous batching requires solving KV memory allocation (PagedAttention)
- Direct Answer: Not knowing that continuous batching requires solving KV memory allocation (PagedAttention)
- Why: This matters because it tells you how to reason about not knowing that continuous batching requires solving kv memory allocation (pagedattention).
- Pitfall: Don't answer "Not knowing that continuous batching requires solving KV memory allocation (PagedAttention)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not knowing that continuous batching requires solving KV memory allocation (PagedAttention)

### Claiming continuous batching eliminates tail latency
- Direct Answer: it improves throughput, not worst-case latency
- Why: This matters because it tells you how to reason about claiming continuous batching eliminates tail latency.
- Pitfall: Don't answer "Claiming continuous batching eliminates tail latency" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: it improves throughput, not worst-case latency

### Low acceptance rate
- Direct Answer: if draft and target disagree often, you run the large model more than unspeculative decoding
- Why: This matters because it tells you how to reason about low acceptance rate.
- Pitfall: Don't answer "Low acceptance rate" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: if draft and target disagree often, you run the large model more than unspeculative decoding

### Draft model must run on same hardware
- Direct Answer: adds memory pressure
- Why: This matters because it tells you how to reason about draft model must run on same hardware.
- Pitfall: Don't answer "Draft model must run on same hardware" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: adds memory pressure

### Doesn't help with prefill (only decode is the bottleneck)
- Direct Answer: Doesn't help with prefill (only decode is the bottleneck)
- Why: This matters because it tells you how to reason about doesn't help with prefill (only decode is the bottleneck).
- Pitfall: Don't answer "Doesn't help with prefill (only decode is the bottleneck)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Doesn't help with prefill (only decode is the bottleneck)

### Not beneficial when the bottleneck is memory, not compute (small batch, bandwidth-limited)
- Direct Answer: Not beneficial when the bottleneck is memory, not compute (small batch, bandwidth-limited)
- Why: This matters because it tells you how to reason about not beneficial when the bottleneck is memory, not compute (small batch, bandwidth-limited).
- Pitfall: Don't answer "Not beneficial when the bottleneck is memory, not compute (small batch, bandwidth-limited)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not beneficial when the bottleneck is memory, not compute (small batch, bandwidth-limited)

### Claiming quality degrades
- Direct Answer: it doesn't, by construction of the acceptance criterion
- Why: This matters because it tells you how to reason about claiming quality degrades.
- Pitfall: Don't answer "Claiming quality degrades" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: it doesn't, by construction of the acceptance criterion

### Not knowing when it fails (low acceptance rate or memory-bound regime)
- Direct Answer: Not knowing when it fails (low acceptance rate or memory-bound regime)
- Why: This matters because it tells you how to reason about not knowing when it fails (low acceptance rate or memory-bound regime).
- Pitfall: Don't answer "Not knowing when it fails (low acceptance rate or memory-bound regime)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not knowing when it fails (low acceptance rate or memory-bound regime)

### Confusing Medusa/Eagle variants (self-speculative, no separate draft model) with standard speculative decoding
- Direct Answer: Confusing Medusa/Eagle variants (self-speculative, no separate draft model) with standard speculative decoding
- Why: This matters because it tells you how to reason about confusing medusa/eagle variants (self-speculative, no separate draft model) with standard speculative decoding.
- Pitfall: Don't answer "Confusing Medusa/Eagle variants (self-speculative, no separate draft model) with standard speculative decoding" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Confusing Medusa/Eagle variants (self-speculative, no separate draft model) with standard speculative decoding

### Limit max_seq_len and batch_size
- Direct Answer: hard cap on KV memory
- Why: This matters because it tells you how to reason about limit max_seq_len and batch_size.
- Pitfall: Don't answer "Limit max_seq_len and batch_size" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: hard cap on KV memory

### PagedAttention
- Direct Answer: don't pre-allocate for max length; allocate blocks on demand
- Why: This matters because it tells you how to reason about pagedattention.
- Pitfall: Don't answer "PagedAttention" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: don't pre-allocate for max length; allocate blocks on demand

### KV quantization (INT8/FP8)
- Direct Answer: halve or quarter KV memory at some quality cost
- Why: This matters because it tells you how to reason about kv quantization (int8/fp8).
- Pitfall: Don't answer "KV quantization (INT8/FP8)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: halve or quarter KV memory at some quality cost

### Prefix caching
- Direct Answer: if many requests share a system prompt, compute its KV once and reuse
- Why: This matters because it tells you how to reason about prefix caching.
- Pitfall: Don't answer "Prefix caching" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: if many requests share a system prompt, compute its KV once and reuse

### Forgetting KV cache in capacity planning
- Direct Answer: you can fit the model but not serve any requests
- Why: This matters because it tells you how to reason about forgetting kv cache in capacity planning.
- Pitfall: Don't answer "Forgetting KV cache in capacity planning" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: you can fit the model but not serve any requests

### Pre-allocating contiguous max-length KV for every request
- Direct Answer: catastrophic fragmentation
- Why: This matters because it tells you how to reason about pre-allocating contiguous max-length kv for every request.
- Pitfall: Don't answer "Pre-allocating contiguous max-length KV for every request" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: catastrophic fragmentation

### Caching with PII in shared prefix cache
- Direct Answer: data leakage risk
- Why: This matters because it tells you how to reason about caching with pii in shared prefix cache.
- Pitfall: Don't answer "Caching with PII in shared prefix cache" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: data leakage risk

### Accounting for weights but not KV in VRAM planning
- Direct Answer: Accounting for weights but not KV in VRAM planning
- Why: This matters because it tells you how to reason about accounting for weights but not kv in vram planning.
- Pitfall: Don't answer "Accounting for weights but not KV in VRAM planning" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Accounting for weights but not KV in VRAM planning

### Not knowing that GQA dramatically reduces KV cache size
- Direct Answer: Not knowing that GQA dramatically reduces KV cache size
- Why: This matters because it tells you how to reason about not knowing that gqa dramatically reduces kv cache size.
- Pitfall: Don't answer "Not knowing that GQA dramatically reduces KV cache size" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not knowing that GQA dramatically reduces KV cache size

### Treating "KV cache" and "response cache" as the same thing (they're not)
- Direct Answer: Treating "KV cache" and "response cache" as the same thing (they're not)
- Why: This matters because it tells you how to reason about treating "kv cache" and "response cache" as the same thing (they're not).
- Pitfall: Don't answer "Treating "KV cache" and "response cache" as the same thing (they're not)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Treating "KV cache" and "response cache" as the same thing (they're not)

### Divide KV cache into fixed-size blocks, e.g., 16 tokens per block
- Direct Answer: Divide KV cache into fixed-size blocks, e.g., 16 tokens per block
- Why: This matters because it tells you how to reason about divide kv cache into fixed-size blocks, e.g., 16 tokens per block.
- Pitfall: Don't answer "Divide KV cache into fixed-size blocks, e.g., 16 tokens per block" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Divide KV cache into fixed-size blocks, e.g., 16 tokens per block

### Each sequence has a page table
- Direct Answer: logical token position → physical block index
- Why: This matters because it tells you how to reason about each sequence has a page table.
- Pitfall: Don't answer "Each sequence has a page table" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: logical token position → physical block index

### Allocate one block at a time as generation proceeds
- Direct Answer: Allocate one block at a time as generation proceeds
- Why: This matters because it tells you how to reason about allocate one block at a time as generation proceeds.
- Pitfall: Don't answer "Allocate one block at a time as generation proceeds" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Allocate one block at a time as generation proceeds

### On request completion, mark blocks as free for reuse
- Direct Answer: On request completion, mark blocks as free for reuse
- Why: This matters because it tells you how to reason about on request completion, mark blocks as free for reuse.
- Pitfall: Don't answer "On request completion, mark blocks as free for reuse" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: On request completion, mark blocks as free for reuse

### Sequences sharing a prefix (same system prompt) can share physical blocks (copy-on-write)
- Direct Answer: Sequences sharing a prefix (same system prompt) can share physical blocks (copy-on-write)
- Why: This matters because it tells you how to reason about sequences sharing a prefix (same system prompt) can share physical blocks (copy-on-write).
- Pitfall: Don't answer "Sequences sharing a prefix (same system prompt) can share physical blocks (copy-on-write)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Sequences sharing a prefix (same system prompt) can share physical blocks (copy-on-write)

### Very small block sizes (e.g., 4 tokens)
- Direct Answer: too many page table lookups, kernel overhead
- Why: This matters because it tells you how to reason about very small block sizes (e.g., 4 tokens).
- Pitfall: Don't answer "Very small block sizes (e.g., 4 tokens)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: too many page table lookups, kernel overhead

### Very large block sizes (e.g., 256 tokens)
- Direct Answer: loses the fragmentation benefit
- Why: This matters because it tells you how to reason about very large block sizes (e.g., 256 tokens).
- Pitfall: Don't answer "Very large block sizes (e.g., 256 tokens)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: loses the fragmentation benefit

### Prefix sharing with dynamic prefixes
- Direct Answer: copy-on-write overhead for sequences that diverge quickly
- Why: This matters because it tells you how to reason about prefix sharing with dynamic prefixes.
- Pitfall: Don't answer "Prefix sharing with dynamic prefixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: copy-on-write overhead for sequences that diverge quickly

### Thinking vLLM is primarily about the model
- Direct Answer: it's about the memory scheduler and serving stack
- Why: This matters because it tells you how to reason about thinking vllm is primarily about the model.
- Pitfall: Don't answer "Thinking vLLM is primarily about the model" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: it's about the memory scheduler and serving stack

### Not knowing the block size trade-off
- Direct Answer: Not knowing the block size trade-off
- Why: This matters because it tells you how to reason about not knowing the block size trade-off.
- Pitfall: Don't answer "Not knowing the block size trade-off" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not knowing the block size trade-off

### Confusing prefix caching (reuse computed KV) with response caching (reuse final output)
- Direct Answer: Confusing prefix caching (reuse computed KV) with response caching (reuse final output)
- Why: This matters because it tells you how to reason about confusing prefix caching (reuse computed kv) with response caching (reuse final output).
- Pitfall: Don't answer "Confusing prefix caching (reuse computed KV) with response caching (reuse final output)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Confusing prefix caching (reuse computed KV) with response caching (reuse final output)

### Assuming desktop benchmark numbers transfer to phone
- Direct Answer: Neural Engine throughput profiles differently from GPU
- Why: This matters because it tells you how to reason about assuming desktop benchmark numbers transfer to phone.
- Pitfall: Don't answer "Assuming desktop benchmark numbers transfer to phone" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Neural Engine throughput profiles differently from GPU

### Thermal throttling
- Direct Answer: sustained inference degrades after 30–60 seconds on mobile
- Why: This matters because it tells you how to reason about thermal throttling.
- Pitfall: Don't answer "Thermal throttling" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: sustained inference degrades after 30–60 seconds on mobile

### GGUF Q4_K_M quality is acceptable for general chat but can hurt specialized tasks (math, code)
- Direct Answer: GGUF Q4_K_M quality is acceptable for general chat but can hurt specialized tasks (math, code)
- Why: This matters because it tells you how to reason about gguf q4_k_m quality is acceptable for general chat but can hurt specialized tasks (math, code).
- Pitfall: Don't answer "GGUF Q4_K_M quality is acceptable for general chat but can hurt specialized tasks (math, code)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: GGUF Q4_K_M quality is acceptable for general chat but can hurt specialized tasks (math, code)

### Applying datacenter optimization thinking to edge
- Direct Answer: Applying datacenter optimization thinking to edge
- Why: This matters because it tells you how to reason about applying datacenter optimization thinking to edge.
- Pitfall: Don't answer "Applying datacenter optimization thinking to edge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Applying datacenter optimization thinking to edge

### Ignoring battery/thermal as constraints
- Direct Answer: Ignoring battery/thermal as constraints
- Why: This matters because it tells you how to reason about ignoring battery/thermal as constraints.
- Pitfall: Don't answer "Ignoring battery/thermal as constraints" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Ignoring battery/thermal as constraints

### Not knowing platform-specific runtimes (CoreML, NNAPI)
- Direct Answer: Not knowing platform-specific runtimes (CoreML, NNAPI)
- Why: This matters because it tells you how to reason about not knowing platform-specific runtimes (coreml, nnapi).
- Pitfall: Don't answer "Not knowing platform-specific runtimes (CoreML, NNAPI)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not knowing platform-specific runtimes (CoreML, NNAPI)

### Quantizing with a calibration dataset that doesn't match production distribution
- Direct Answer: Quantizing with a calibration dataset that doesn't match production distribution
- Why: This matters because it tells you how to reason about quantizing with a calibration dataset that doesn't match production distribution.
- Pitfall: Don't answer "Quantizing with a calibration dataset that doesn't match production distribution" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Quantizing with a calibration dataset that doesn't match production distribution

### Quantizing sensitive layers (first/last layers often degrade more) at the same bit width as others
- Direct Answer: Quantizing sensitive layers (first/last layers often degrade more) at the same bit width as others
- Why: This matters because it tells you how to reason about quantizing sensitive layers (first/last layers often degrade more) at the same bit width as others.
- Pitfall: Don't answer "Quantizing sensitive layers (first/last layers often degrade more) at the same bit width as others" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Quantizing sensitive layers (first/last layers often degrade more) at the same bit width as others

### Claiming "no quality loss" without task-specific evaluation (perplexity is a proxy)
- Direct Answer: Claiming "no quality loss" without task-specific evaluation (perplexity is a proxy)
- Why: This matters because it tells you how to reason about claiming "no quality loss" without task-specific evaluation (perplexity is a proxy).
- Pitfall: Don't answer "Claiming "no quality loss" without task-specific evaluation (perplexity is a proxy)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Claiming "no quality loss" without task-specific evaluation (perplexity is a proxy)

### BF16 vs FP16
- Direct Answer: BF16 has wider exponent range (better for training stability); FP16 has higher precision in mantissa
- Why: This matters because it tells you how to reason about bf16 vs fp16.
- Pitfall: Don't answer "BF16 vs FP16" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: BF16 has wider exponent range (better for training stability); FP16 has higher precision in mantissa

### Claiming quantization has no quality impact without eval
- Direct Answer: Claiming quantization has no quality impact without eval
- Why: This matters because it tells you how to reason about claiming quantization has no quality impact without eval.
- Pitfall: Don't answer "Claiming quantization has no quality impact without eval" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Claiming quantization has no quality impact without eval

### Not distinguishing PTQ (no retraining) from QAT (quantization-aware training)
- Direct Answer: Not distinguishing PTQ (no retraining) from QAT (quantization-aware training)
- Why: This matters because it tells you how to reason about not distinguishing ptq (no retraining) from qat (quantization-aware training).
- Pitfall: Don't answer "Not distinguishing PTQ (no retraining) from QAT (quantization-aware training)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not distinguishing PTQ (no retraining) from QAT (quantization-aware training)

### Confusing weight quantization with activation quantization (activation quantization is harder and less commonly applied)
- Direct Answer: Confusing weight quantization with activation quantization (activation quantization is harder and less commonly applied)
- Why: This matters because it tells you how to reason about confusing weight quantization with activation quantization (activation quantization is harder and less commonly applied).
- Pitfall: Don't answer "Confusing weight quantization with activation quantization (activation quantization is harder and less commonly applied)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Confusing weight quantization with activation quantization (activation quantization is harder and less commonly applied)

### GPU cold start: provisioning an A100, loading model weights (140 GB over NVMe/network), compiling CUDA graphs
- Direct Answer: 3–10 minutes. This is not acceptable for burst handling.
- Why: This matters because it tells you how to reason about gpu cold start: provisioning an a100, loading model weights (140 gb over nvme/network), compiling cuda graphs.
- Pitfall: Don't answer "GPU cold start: provisioning an A100, loading model weights (140 GB over NVMe/network), compiling CUDA graphs" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: 3–10 minutes. This is not acceptable for burst handling.

### Mitigation
- Direct Answer: min_replicas ≥ 1, warm pools with loaded weights, pre-pulled container images, compiled model artifacts cached
- Why: This matters because it tells you how to reason about mitigation.
- Pitfall: Don't answer "Mitigation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: min_replicas ≥ 1, warm pools with loaded weights, pre-pulled container images, compiled model artifacts cached

### Pure CPU-based HPA for GPU workloads
- Direct Answer: Pure CPU-based HPA for GPU workloads
- Why: This matters because it tells you how to reason about pure cpu-based hpa for gpu workloads.
- Pitfall: Don't answer "Pure CPU-based HPA for GPU workloads" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Pure CPU-based HPA for GPU workloads

### Setting min_replicas=0 for interactive SLOs
- Direct Answer: cold starts are unacceptably slow
- Why: This matters because it tells you how to reason about setting min_replicas=0 for interactive slos.
- Pitfall: Don't answer "Setting min_replicas=0 for interactive SLOs" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: cold starts are unacceptably slow

### Not separating interactive (p95 latency SLO) and batch (throughput SLO) pools
- Direct Answer: Not separating interactive (p95 latency SLO) and batch (throughput SLO) pools
- Why: This matters because it tells you how to reason about not separating interactive (p95 latency slo) and batch (throughput slo) pools.
- Pitfall: Don't answer "Not separating interactive (p95 latency SLO) and batch (throughput SLO) pools" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not separating interactive (p95 latency SLO) and batch (throughput SLO) pools

### Scaling only on CPU when GPU is the bottleneck
- Direct Answer: Scaling only on CPU when GPU is the bottleneck
- Why: This matters because it tells you how to reason about scaling only on cpu when gpu is the bottleneck.
- Pitfall: Don't answer "Scaling only on CPU when GPU is the bottleneck" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Scaling only on CPU when GPU is the bottleneck

### Not knowing that model load time (not container start) dominates GPU cold start
- Direct Answer: Not knowing that model load time (not container start) dominates GPU cold start
- Why: This matters because it tells you how to reason about not knowing that model load time (not container start) dominates gpu cold start.
- Pitfall: Don't answer "Not knowing that model load time (not container start) dominates GPU cold start" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not knowing that model load time (not container start) dominates GPU cold start

### Ignoring batch vs interactive traffic separation
- Direct Answer: Ignoring batch vs interactive traffic separation
- Why: This matters because it tells you how to reason about ignoring batch vs interactive traffic separation.
- Pitfall: Don't answer "Ignoring batch vs interactive traffic separation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Ignoring batch vs interactive traffic separation

### Pure round-robin with variable-length requests (systematic load imbalance)
- Direct Answer: Pure round-robin with variable-length requests (systematic load imbalance)
- Why: This matters because it tells you how to reason about pure round-robin with variable-length requests (systematic load imbalance).
- Pitfall: Don't answer "Pure round-robin with variable-length requests (systematic load imbalance)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Pure round-robin with variable-length requests (systematic load imbalance)

### No health checks
- Direct Answer: routing to a replica that loaded the wrong model or has OOM errors
- Why: This matters because it tells you how to reason about no health checks.
- Pitfall: Don't answer "No health checks" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: routing to a replica that loaded the wrong model or has OOM errors

### Sticky sessions for stateless serving
- Direct Answer: usually wrong; state lives in the request, not the server
- Why: This matters because it tells you how to reason about sticky sessions for stateless serving.
- Pitfall: Don't answer "Sticky sessions for stateless serving" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: usually wrong; state lives in the request, not the server

### Assuming round-robin is fine because "it's stateless"
- Direct Answer: Assuming round-robin is fine because "it's stateless"
- Why: This matters because it tells you how to reason about assuming round-robin is fine because "it's stateless".
- Pitfall: Don't answer "Assuming round-robin is fine because "it's stateless"" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Assuming round-robin is fine because "it's stateless"

### Not thinking about long prompts tying up one replica
- Direct Answer: Not thinking about long prompts tying up one replica
- Why: This matters because it tells you how to reason about not thinking about long prompts tying up one replica.
- Pitfall: Don't answer "Not thinking about long prompts tying up one replica" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not thinking about long prompts tying up one replica

### Conflating model routing (which model to use) with load balancing (which replica of the same model)
- Direct Answer: Conflating model routing (which model to use) with load balancing (which replica of the same model)
- Why: This matters because it tells you how to reason about conflating model routing (which model to use) with load balancing (which replica of the same model).
- Pitfall: Don't answer "Conflating model routing (which model to use) with load balancing (which replica of the same model)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Conflating model routing (which model to use) with load balancing (which replica of the same model)

### Forgetting KV cache when colocating models
- Direct Answer: the models fit in VRAM at zero requests, but KV pressure evicts models mid-request
- Why: This matters because it tells you how to reason about forgetting kv cache when colocating models.
- Pitfall: Don't answer "Forgetting KV cache when colocating models" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: the models fit in VRAM at zero requests, but KV pressure evicts models mid-request

### LRU eviction without request-ahead loading
- Direct Answer: first request after eviction is slow
- Why: This matters because it tells you how to reason about lru eviction without request-ahead loading.
- Pitfall: Don't answer "LRU eviction without request-ahead loading" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: first request after eviction is slow

### MIG partitions are fixed at setup
- Direct Answer: can't be resized per traffic pattern dynamically
- Why: This matters because it tells you how to reason about mig partitions are fixed at setup.
- Pitfall: Don't answer "MIG partitions are fixed at setup" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: can't be resized per traffic pattern dynamically

### Ignoring KV footprint when calculating "fits on the GPU"
- Direct Answer: Ignoring KV footprint when calculating "fits on the GPU"
- Why: This matters because it tells you how to reason about ignoring kv footprint when calculating "fits on the gpu".
- Pitfall: Don't answer "Ignoring KV footprint when calculating "fits on the GPU"" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Ignoring KV footprint when calculating "fits on the GPU"

### Assuming MIG is always appropriate
- Direct Answer: it adds operational complexity and fixed capacity allocation
- Why: This matters because it tells you how to reason about assuming mig is always appropriate.
- Pitfall: Don't answer "Assuming MIG is always appropriate" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: it adds operational complexity and fixed capacity allocation

### Not having a routing layer that knows model placement
- Direct Answer: Not having a routing layer that knows model placement
- Why: This matters because it tells you how to reason about not having a routing layer that knows model placement.
- Pitfall: Don't answer "Not having a routing layer that knows model placement" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not having a routing layer that knows model placement

### Shards parameters, gradients, and optimizer states across DP ranks
- Direct Answer: Shards parameters, gradients, and optimizer states across DP ranks
- Why: This matters because it tells you how to reason about shards parameters, gradients, and optimizer states across dp ranks.
- Pitfall: Don't answer "Shards parameters, gradients, and optimizer states across DP ranks" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Shards parameters, gradients, and optimizer states across DP ranks

### AllGather parameters before each layer's forward pass, free after
- Direct Answer: AllGather parameters before each layer's forward pass, free after
- Why: This matters because it tells you how to reason about allgather parameters before each layer's forward pass, free after.
- Pitfall: Don't answer "AllGather parameters before each layer's forward pass, free after" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: AllGather parameters before each layer's forward pass, free after

### ReduceScatter gradients during backward
- Direct Answer: ReduceScatter gradients during backward
- Why: This matters because it tells you how to reason about reducescatter gradients during backward.
- Pitfall: Don't answer "ReduceScatter gradients during backward" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: ReduceScatter gradients during backward

### Memory per GPU
- Direct Answer: O(model_size / num_gpus) instead of O(model_size)
- Why: This matters because it tells you how to reason about memory per gpu.
- Pitfall: Don't answer "Memory per GPU" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: O(model_size / num_gpus) instead of O(model_size)

### Stage 1
- Direct Answer: shard optimizer states only. Each GPU holds full weights, sharded optimizer state.
- Why: This matters because it tells you how to reason about stage 1.
- Pitfall: Don't answer "Stage 1" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: shard optimizer states only. Each GPU holds full weights, sharded optimizer state.

### Stage 2
- Direct Answer: + shard gradients. Reduced gradient memory.
- Why: This matters because it tells you how to reason about stage 2.
- Pitfall: Don't answer "Stage 2" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: + shard gradients. Reduced gradient memory.

### Stage 3
- Direct Answer: + shard parameters (equivalent to FSDP). Minimum memory.
- Why: This matters because it tells you how to reason about stage 3.
- Pitfall: Don't answer "Stage 3" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: + shard parameters (equivalent to FSDP). Minimum memory.

### ZeRO-Infinity
- Direct Answer: offload to CPU/NVMe for extreme scale.
- Why: This matters because it tells you how to reason about zero-infinity.
- Pitfall: Don't answer "ZeRO-Infinity" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: offload to CPU/NVMe for extreme scale.

### Training, PyTorch ecosystem → FSDP
- Direct Answer: Training, PyTorch ecosystem → FSDP
- Why: This matters because it tells you how to reason about training, pytorch ecosystem → fsdp.
- Pitfall: Don't answer "Training, PyTorch ecosystem → FSDP" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Training, PyTorch ecosystem → FSDP

### Training, want CPU/NVMe offload → DeepSpeed ZeRO-3 / ZeRO-Infinity
- Direct Answer: Training, want CPU/NVMe offload → DeepSpeed ZeRO-3 / ZeRO-Infinity
- Why: This matters because it tells you how to reason about training, want cpu/nvme offload → deepspeed zero-3 / zero-infinity.
- Pitfall: Don't answer "Training, want CPU/NVMe offload → DeepSpeed ZeRO-3 / ZeRO-Infinity" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Training, want CPU/NVMe offload → DeepSpeed ZeRO-3 / ZeRO-Infinity

### Inference, single node → TP (fast interconnect required)
- Direct Answer: Inference, single node → TP (fast interconnect required)
- Why: This matters because it tells you how to reason about inference, single node → tp (fast interconnect required).
- Pitfall: Don't answer "Inference, single node → TP (fast interconnect required)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Inference, single node → TP (fast interconnect required)

### Inference, multi-node → TP within node + PP across nodes
- Direct Answer: Inference, multi-node → TP within node + PP across nodes
- Why: This matters because it tells you how to reason about inference, multi-node → tp within node + pp across nodes.
- Pitfall: Don't answer "Inference, multi-node → TP within node + PP across nodes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Inference, multi-node → TP within node + PP across nodes

### High sharding degree with slow interconnect
- Direct Answer: AllGather/ReduceScatter communication dominates training time
- Why: This matters because it tells you how to reason about high sharding degree with slow interconnect.
- Pitfall: Don't answer "High sharding degree with slow interconnect" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: AllGather/ReduceScatter communication dominates training time

### ZeRO-3 with many small modules
- Direct Answer: excessive communication granularity
- Why: This matters because it tells you how to reason about zero-3 with many small modules.
- Pitfall: Don't answer "ZeRO-3 with many small modules" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: excessive communication granularity

### Mixing FSDP sharding with non-FSDP modules
- Direct Answer: gradient/parameter misalignment
- Why: This matters because it tells you how to reason about mixing fsdp sharding with non-fsdp modules.
- Pitfall: Don't answer "Mixing FSDP sharding with non-FSDP modules" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: gradient/parameter misalignment

### Using "sharding" without specifying which type
- Direct Answer: Using "sharding" without specifying which type
- Why: This matters because it tells you how to reason about using "sharding" without specifying which type.
- Pitfall: Don't answer "Using "sharding" without specifying which type" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Using "sharding" without specifying which type

### Claiming FSDP and ZeRO-3 are the same (they're similar but different implementations)
- Direct Answer: Claiming FSDP and ZeRO-3 are the same (they're similar but different implementations)
- Why: This matters because it tells you how to reason about claiming fsdp and zero-3 are the same (they're similar but different implementations).
- Pitfall: Don't answer "Claiming FSDP and ZeRO-3 are the same (they're similar but different implementations)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Claiming FSDP and ZeRO-3 are the same (they're similar but different implementations)

### Applying training sharding strategies to inference serving
- Direct Answer: Applying training sharding strategies to inference serving
- Why: This matters because it tells you how to reason about applying training sharding strategies to inference serving.
- Pitfall: Don't answer "Applying training sharding strategies to inference serving" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Applying training sharding strategies to inference serving

### When queue depth > max_queue_depth
- Direct Answer: reject new requests with 429 + Retry-After
- Why: This matters because it tells you how to reason about when queue depth > max_queue_depth.
- Pitfall: Don't answer "When queue depth > max_queue_depth" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: reject new requests with 429 + Retry-After

### When pending tokens > GPU_memory × 0.9: backpressure
- Direct Answer: stop admitting until KV memory clears
- Why: This matters because it tells you how to reason about when pending tokens > gpu_memory × 0.9: backpressure.
- Pitfall: Don't answer "When pending tokens > GPU_memory × 0.9: backpressure" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: stop admitting until KV memory clears

### Never unbounded queues
- Direct Answer: they hide memory pressure and cause OOM under burst
- Why: This matters because it tells you how to reason about never unbounded queues.
- Pitfall: Don't answer "Never unbounded queues" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: they hide memory pressure and cause OOM under burst

### Shortest-job-first (predict output length)
- Direct Answer: minimizes mean wait time
- Why: This matters because it tells you how to reason about shortest-job-first (predict output length).
- Pitfall: Don't answer "Shortest-job-first (predict output length)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: minimizes mean wait time

### Weighted fair queuing
- Direct Answer: prevent starvation of low-priority tenants
- Why: This matters because it tells you how to reason about weighted fair queuing.
- Pitfall: Don't answer "Weighted fair queuing" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: prevent starvation of low-priority tenants

### Unbounded queue
- Direct Answer: requests accumulate, memory usage grows, OOM kills the server
- Why: This matters because it tells you how to reason about unbounded queue.
- Pitfall: Don't answer "Unbounded queue" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: requests accumulate, memory usage grows, OOM kills the server

### Pure FIFO with long requests
- Direct Answer: short interactive requests starve
- Why: This matters because it tells you how to reason about pure fifo with long requests.
- Pitfall: Don't answer "Pure FIFO with long requests" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: short interactive requests starve

### No backpressure
- Direct Answer: load balancer keeps sending requests that can't be served, cascading failure
- Why: This matters because it tells you how to reason about no backpressure.
- Pitfall: Don't answer "No backpressure" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: load balancer keeps sending requests that can't be served, cascading failure

### Implementing FIFO and calling it "fair"
- Direct Answer: Implementing FIFO and calling it "fair"
- Why: This matters because it tells you how to reason about implementing fifo and calling it "fair".
- Pitfall: Don't answer "Implementing FIFO and calling it "fair"" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Implementing FIFO and calling it "fair"

### Not limiting queue depth (unbounded queue = OOM vulnerability)
- Direct Answer: Not limiting queue depth (unbounded queue = OOM vulnerability)
- Why: This matters because it tells you how to reason about not limiting queue depth (unbounded queue = oom vulnerability).
- Pitfall: Don't answer "Not limiting queue depth (unbounded queue = OOM vulnerability)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not limiting queue depth (unbounded queue = OOM vulnerability)

### Not distinguishing rate limiting (per-tenant) from admission control (global capacity)
- Direct Answer: Not distinguishing rate limiting (per-tenant) from admission control (global capacity)
- Why: This matters because it tells you how to reason about not distinguishing rate limiting (per-tenant) from admission control (global capacity).
- Pitfall: Don't answer "Not distinguishing rate limiting (per-tenant) from admission control (global capacity)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not distinguishing rate limiting (per-tenant) from admission control (global capacity)

### GPU utilization
- Direct Answer: idle GPUs burn money. A 40% utilized GPU cluster costs 2.5× per token vs a busy one.
- Why: This matters because it tells you how to reason about gpu utilization.
- Pitfall: Don't answer "GPU utilization" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: idle GPUs burn money. A 40% utilized GPU cluster costs 2.5× per token vs a busy one.

### Engineering time
- Direct Answer: 1–2 engineers full-time maintaining GPU infra vs zero for API.
- Why: This matters because it tells you how to reason about engineering time.
- Pitfall: Don't answer "Engineering time" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: 1–2 engineers full-time maintaining GPU infra vs zero for API.

### On-call burden
- Direct Answer: GPU failures, CUDA OOM, model crashes at 3am.
- Why: This matters because it tells you how to reason about on-call burden.
- Pitfall: Don't answer "On-call burden" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: GPU failures, CUDA OOM, model crashes at 3am.

### Flexibility
- Direct Answer: self-hosted locks you to one model; API lets you switch cheaply.
- Why: This matters because it tells you how to reason about flexibility.
- Pitfall: Don't answer "Flexibility" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: self-hosted locks you to one model; API lets you switch cheaply.

### Data residency / compliance requirements → self-host (no choice)
- Direct Answer: Data residency / compliance requirements → self-host (no choice)
- Why: This matters because it tells you how to reason about data residency / compliance requirements → self-host (no choice).
- Pitfall: Don't answer "Data residency / compliance requirements → self-host (no choice)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Data residency / compliance requirements → self-host (no choice)

### High volume + stable workload + GPU ops expertise → evaluate self-host
- Direct Answer: High volume + stable workload + GPU ops expertise → evaluate self-host
- Why: This matters because it tells you how to reason about high volume + stable workload + gpu ops expertise → evaluate self-host.
- Pitfall: Don't answer "High volume + stable workload + GPU ops expertise → evaluate self-host" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: High volume + stable workload + GPU ops expertise → evaluate self-host

### Rapid iteration / uncertain volume / small team → API + gateway
- Direct Answer: Rapid iteration / uncertain volume / small team → API + gateway
- Why: This matters because it tells you how to reason about rapid iteration / uncertain volume / small team → api + gateway.
- Pitfall: Don't answer "Rapid iteration / uncertain volume / small team → API + gateway" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Rapid iteration / uncertain volume / small team → API + gateway

### Best of both
- Direct Answer: API for general traffic, self-host for private/high-volume paths
- Why: This matters because it tells you how to reason about best of both.
- Pitfall: Don't answer "Best of both" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: API for general traffic, self-host for private/high-volume paths

### Comparing API price × volume vs GPU hardware cost only (ignoring ops)
- Direct Answer: Comparing API price × volume vs GPU hardware cost only (ignoring ops)
- Why: This matters because it tells you how to reason about comparing api price × volume vs gpu hardware cost only (ignoring ops).
- Pitfall: Don't answer "Comparing API price × volume vs GPU hardware cost only (ignoring ops)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Comparing API price × volume vs GPU hardware cost only (ignoring ops)

### Assuming 100% GPU utilization in self-host projections
- Direct Answer: Assuming 100% GPU utilization in self-host projections
- Why: This matters because it tells you how to reason about assuming 100% gpu utilization in self-host projections.
- Pitfall: Don't answer "Assuming 100% GPU utilization in self-host projections" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Assuming 100% GPU utilization in self-host projections

### Ignoring vendor lock-in risk in API-first projections
- Direct Answer: Ignoring vendor lock-in risk in API-first projections
- Why: This matters because it tells you how to reason about ignoring vendor lock-in risk in api-first projections.
- Pitfall: Don't answer "Ignoring vendor lock-in risk in API-first projections" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Ignoring vendor lock-in risk in API-first projections

### Recommending self-host without accounting for staffing cost
- Direct Answer: Recommending self-host without accounting for staffing cost
- Why: This matters because it tells you how to reason about recommending self-host without accounting for staffing cost.
- Pitfall: Don't answer "Recommending self-host without accounting for staffing cost" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Recommending self-host without accounting for staffing cost

### Not mentioning hybrid as a common real-world answer
- Direct Answer: Not mentioning hybrid as a common real-world answer
- Why: This matters because it tells you how to reason about not mentioning hybrid as a common real-world answer.
- Pitfall: Don't answer "Not mentioning hybrid as a common real-world answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not mentioning hybrid as a common real-world answer

### Scale-to-zero for interactive chat
- Direct Answer: users see 3–10 minute first-response latency
- Why: This matters because it tells you how to reason about scale-to-zero for interactive chat.
- Pitfall: Don't answer "Scale-to-zero for interactive chat" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: users see 3–10 minute first-response latency

### Over-provisioning warm pools
- Direct Answer: expensive for low-traffic applications
- Why: This matters because it tells you how to reason about over-provisioning warm pools.
- Pitfall: Don't answer "Over-provisioning warm pools" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: expensive for low-traffic applications

### Not caching compiled CUDA graphs
- Direct Answer: repeated compilation on every restart
- Why: This matters because it tells you how to reason about not caching compiled cuda graphs.
- Pitfall: Don't answer "Not caching compiled CUDA graphs" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: repeated compilation on every restart

### Assuming serverless = zero cold start with the right settings
- Direct Answer: Assuming serverless = zero cold start with the right settings
- Why: This matters because it tells you how to reason about assuming serverless = zero cold start with the right settings.
- Pitfall: Don't answer "Assuming serverless = zero cold start with the right settings" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Assuming serverless = zero cold start with the right settings

### Not knowing that CUDA graph compilation adds substantial startup time
- Direct Answer: Not knowing that CUDA graph compilation adds substantial startup time
- Why: This matters because it tells you how to reason about not knowing that cuda graph compilation adds substantial startup time.
- Pitfall: Don't answer "Not knowing that CUDA graph compilation adds substantial startup time" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not knowing that CUDA graph compilation adds substantial startup time

### Treating cold start as a solvable problem rather than a managed trade-off
- Direct Answer: Treating cold start as a solvable problem rather than a managed trade-off
- Why: This matters because it tells you how to reason about treating cold start as a solvable problem rather than a managed trade-off.
- Pitfall: Don't answer "Treating cold start as a solvable problem rather than a managed trade-off" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Treating cold start as a solvable problem rather than a managed trade-off

### Cache key missing temperature: deterministic (temp=0) and stochastic (temp=0.7) responses share a key
- Direct Answer: wrong
- Why: This matters because it tells you how to reason about cache key missing temperature: deterministic (temp=0) and stochastic (temp=0.7) responses share a key.
- Pitfall: Don't answer "Cache key missing temperature: deterministic (temp=0) and stochastic (temp=0.7) responses share a key" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: wrong

### Semantic cache false positives
- Direct Answer: "What's Python?" and "What's COBOL?" might be too similar by embedding
- Why: This matters because it tells you how to reason about semantic cache false positives.
- Pitfall: Don't answer "Semantic cache false positives" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: "What's Python?" and "What's COBOL?" might be too similar by embedding

### Prefix KV cache invalidation
- Direct Answer: any change to the system prompt invalidates all cached KV blocks
- Why: This matters because it tells you how to reason about prefix kv cache invalidation.
- Pitfall: Don't answer "Prefix KV cache invalidation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: any change to the system prompt invalidates all cached KV blocks

### Treating all caches as equivalent (response cache and KV cache are very different)
- Direct Answer: Treating all caches as equivalent (response cache and KV cache are very different)
- Why: This matters because it tells you how to reason about treating all caches as equivalent (response cache and kv cache are very different).
- Pitfall: Don't answer "Treating all caches as equivalent (response cache and KV cache are very different)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Treating all caches as equivalent (response cache and KV cache are very different)

### Missing decoder params in cache keys
- Direct Answer: Missing decoder params in cache keys
- Why: This matters because it tells you how to reason about missing decoder params in cache keys.
- Pitfall: Don't answer "Missing decoder params in cache keys" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Missing decoder params in cache keys

### Caching PII without access controls
- Direct Answer: Caching PII without access controls
- Why: This matters because it tells you how to reason about caching pii without access controls.
- Pitfall: Don't answer "Caching PII without access controls" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Caching PII without access controls

### Sync long jobs with 30s gateway timeouts
- Direct Answer: requests fail mid-generation
- Why: This matters because it tells you how to reason about sync long jobs with 30s gateway timeouts.
- Pitfall: Don't answer "Sync long jobs with 30s gateway timeouts" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: requests fail mid-generation

### Async without job cleanup
- Direct Answer: job state accumulates, storage fills
- Why: This matters because it tells you how to reason about async without job cleanup.
- Pitfall: Don't answer "Async without job cleanup" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: job state accumulates, storage fills

### Streaming without backpressure
- Direct Answer: if client disconnects, GPU continues generating wasted tokens
- Why: This matters because it tells you how to reason about streaming without backpressure.
- Pitfall: Don't answer "Streaming without backpressure" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: if client disconnects, GPU continues generating wasted tokens

### Treating streaming as "async"
- Direct Answer: it's still a synchronous HTTP connection
- Why: This matters because it tells you how to reason about treating streaming as "async".
- Pitfall: Don't answer "Treating streaming as "async"" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: it's still a synchronous HTTP connection

### Using sync for document generation without timeout planning
- Direct Answer: Using sync for document generation without timeout planning
- Why: This matters because it tells you how to reason about using sync for document generation without timeout planning.
- Pitfall: Don't answer "Using sync for document generation without timeout planning" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Using sync for document generation without timeout planning

### No TTL on async job state
- Direct Answer: No TTL on async job state
- Why: This matters because it tells you how to reason about no ttl on async job state.
- Pitfall: Don't answer "No TTL on async job state" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: No TTL on async job state

### Stage 1
- Direct Answer: shard optimizer states across DP ranks. Each GPU holds full weights and gradients, but only 1/N of optimizer state. Memory: ~60% of full DP.
- Why: This matters because it tells you how to reason about stage 1.
- Pitfall: Don't answer "Stage 1" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: shard optimizer states across DP ranks. Each GPU holds full weights and gradients, but only 1/N of optimizer state. Memory: ~60% of full DP.

### Stage 2
- Direct Answer: + shard gradients. Memory: ~33% of full DP.
- Why: This matters because it tells you how to reason about stage 2.
- Pitfall: Don't answer "Stage 2" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: + shard gradients. Memory: ~33% of full DP.

### Stage 3
- Direct Answer: + shard parameters (equivalent to FSDP). Memory: O(1/N) per GPU.
- Why: This matters because it tells you how to reason about stage 3.
- Pitfall: Don't answer "Stage 3" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: + shard parameters (equivalent to FSDP). Memory: O(1/N) per GPU.

### Forward pass
- Direct Answer: AllGather parameters before each layer, free after layer completes
- Why: This matters because it tells you how to reason about forward pass.
- Pitfall: Don't answer "Forward pass" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: AllGather parameters before each layer, free after layer completes

### Backward pass
- Direct Answer: ReduceScatter gradients after each layer
- Why: This matters because it tells you how to reason about backward pass.
- Pitfall: Don't answer "Backward pass" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: ReduceScatter gradients after each layer

### Optimizer step
- Direct Answer: each rank updates its local shard; AllGather to reconstruct for next forward pass
- Why: This matters because it tells you how to reason about optimizer step.
- Pitfall: Don't answer "Optimizer step" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: each rank updates its local shard; AllGather to reconstruct for next forward pass

### FSDP
- Direct Answer: native PyTorch 2.0+, tight integration with torch.compile, better for HuggingFace-based training
- Why: This matters because it tells you how to reason about fsdp.
- Pitfall: Don't answer "FSDP" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: native PyTorch 2.0+, tight integration with torch.compile, better for HuggingFace-based training

### DeepSpeed ZeRO-3
- Direct Answer: more features (CPU/NVMe offload, ZeRO-Infinity), custom launcher, more configuration options
- Why: This matters because it tells you how to reason about deepspeed zero-3.
- Pitfall: Don't answer "DeepSpeed ZeRO-3" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: more features (CPU/NVMe offload, ZeRO-Infinity), custom launcher, more configuration options

### Both achieve similar memory reduction at equivalent sharding levels
- Direct Answer: Both achieve similar memory reduction at equivalent sharding levels
- Why: This matters because it tells you how to reason about both achieve similar memory reduction at equivalent sharding levels.
- Pitfall: Don't answer "Both achieve similar memory reduction at equivalent sharding levels" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Both achieve similar memory reduction at equivalent sharding levels

### High ZeRO/FSDP sharding with slow interconnect
- Direct Answer: AllGather for each layer dominates training time
- Why: This matters because it tells you how to reason about high zero/fsdp sharding with slow interconnect.
- Pitfall: Don't answer "High ZeRO/FSDP sharding with slow interconnect" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: AllGather for each layer dominates training time

### ZeRO-3 with PyTorch compile
- Direct Answer: extra work needed for compatibility
- Why: This matters because it tells you how to reason about zero-3 with pytorch compile.
- Pitfall: Don't answer "ZeRO-3 with PyTorch compile" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: extra work needed for compatibility

### Not using activation checkpointing alongside sharding
- Direct Answer: activations can still be a memory bottleneck
- Why: This matters because it tells you how to reason about not using activation checkpointing alongside sharding.
- Pitfall: Don't answer "Not using activation checkpointing alongside sharding" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: activations can still be a memory bottleneck

### Saying FSDP is not related to ZeRO (FSDP ≈ ZeRO-3)
- Direct Answer: Saying FSDP is not related to ZeRO (FSDP ≈ ZeRO-3)
- Why: This matters because it tells you how to reason about saying fsdp is not related to zero (fsdp ≈ zero-3).
- Pitfall: Don't answer "Saying FSDP is not related to ZeRO (FSDP ≈ ZeRO-3)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Saying FSDP is not related to ZeRO (FSDP ≈ ZeRO-3)

### Not knowing CPU offload exists (ZeRO-Infinity)
- Direct Answer: Not knowing CPU offload exists (ZeRO-Infinity)
- Why: This matters because it tells you how to reason about not knowing cpu offload exists (zero-infinity).
- Pitfall: Don't answer "Not knowing CPU offload exists (ZeRO-Infinity)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not knowing CPU offload exists (ZeRO-Infinity)

### Conflating training sharding with inference serving strategies
- Direct Answer: Conflating training sharding with inference serving strategies
- Why: This matters because it tells you how to reason about conflating training sharding with inference serving strategies.
- Pitfall: Don't answer "Conflating training sharding with inference serving strategies" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Conflating training sharding with inference serving strategies

### queue_wait = time from request arrival to GPU processing start
- Direct Answer: queue_wait = time from request arrival to GPU processing start
- Why: This matters because it tells you how to reason about queue_wait = time from request arrival to gpu processing start.
- Pitfall: Don't answer "queue_wait = time from request arrival to GPU processing start" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: queue_wait = time from request arrival to GPU processing start

### prefill_time = time to process all prompt tokens (compute-bound)
- Direct Answer: prefill_time = time to process all prompt tokens (compute-bound)
- Why: This matters because it tells you how to reason about prefill_time = time to process all prompt tokens (compute-bound).
- Pitfall: Don't answer "prefill_time = time to process all prompt tokens (compute-bound)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: prefill_time = time to process all prompt tokens (compute-bound)

### tpot = time per output token during decode (memory-bandwidth-bound)
- Direct Answer: tpot = time per output token during decode (memory-bandwidth-bound)
- Why: This matters because it tells you how to reason about tpot = time per output token during decode (memory-bandwidth-bound).
- Pitfall: Don't answer "tpot = time per output token during decode (memory-bandwidth-bound)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: tpot = time per output token during decode (memory-bandwidth-bound)

### TTFT = queue_wait + prefill_time
- Direct Answer: TTFT = queue_wait + prefill_time
- Why: This matters because it tells you how to reason about ttft = queue_wait + prefill_time.
- Pitfall: Don't answer "TTFT = queue_wait + prefill_time" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: TTFT = queue_wait + prefill_time

### High queue_wait → request backlog, need more replicas or better load balancing
- Direct Answer: High queue_wait → request backlog, need more replicas or better load balancing
- Why: This matters because it tells you how to reason about high queue_wait → request backlog, need more replicas or better load balancing.
- Pitfall: Don't answer "High queue_wait → request backlog, need more replicas or better load balancing" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: High queue_wait → request backlog, need more replicas or better load balancing

### High prefill_time → long prompts or compute bottleneck; consider prompt compression
- Direct Answer: High prefill_time → long prompts or compute bottleneck; consider prompt compression
- Why: This matters because it tells you how to reason about high prefill_time → long prompts or compute bottleneck; consider prompt compression.
- Pitfall: Don't answer "High prefill_time → long prompts or compute bottleneck; consider prompt compression" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: High prefill_time → long prompts or compute bottleneck; consider prompt compression

### High TPOT → memory bandwidth saturated; consider quantization or larger batch
- Direct Answer: High TPOT → memory bandwidth saturated; consider quantization or larger batch
- Why: This matters because it tells you how to reason about high tpot → memory bandwidth saturated; consider quantization or larger batch.
- Pitfall: Don't answer "High TPOT → memory bandwidth saturated; consider quantization or larger batch" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: High TPOT → memory bandwidth saturated; consider quantization or larger batch

### KV cache pressure → high fragmentation; check PagedAttention block allocation stats
- Direct Answer: KV cache pressure → high fragmentation; check PagedAttention block allocation stats
- Why: This matters because it tells you how to reason about kv cache pressure → high fragmentation; check pagedattention block allocation stats.
- Pitfall: Don't answer "KV cache pressure → high fragmentation; check PagedAttention block allocation stats" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: KV cache pressure → high fragmentation; check PagedAttention block allocation stats

### DCGM
- Direct Answer: GPU utilization, memory used, memory bandwidth utilization, PCIe/NVLink bandwidth
- Why: This matters because it tells you how to reason about dcgm.
- Pitfall: Don't answer "DCGM" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: GPU utilization, memory used, memory bandwidth utilization, PCIe/NVLink bandwidth

### PyTorch Profiler
- Direct Answer: per-operation timing, kernel launch overhead, CUDA stream serialization
- Why: This matters because it tells you how to reason about pytorch profiler.
- Pitfall: Don't answer "PyTorch Profiler" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: per-operation timing, kernel launch overhead, CUDA stream serialization

### Nsight Systems
- Direct Answer: GPU timeline, kernel concurrency, memory copy overhead
- Why: This matters because it tells you how to reason about nsight systems.
- Pitfall: Don't answer "Nsight Systems" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: GPU timeline, kernel concurrency, memory copy overhead

### TTFT p50/p95/p99
- Direct Answer: TTFT p50/p95/p99
- Why: This matters because it tells you how to reason about ttft p50/p95/p99.
- Pitfall: Don't answer "TTFT p50/p95/p99" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: TTFT p50/p95/p99

### TPOT p50/p95/p99 (inter-token latency)
- Direct Answer: TPOT p50/p95/p99 (inter-token latency)
- Why: This matters because it tells you how to reason about tpot p50/p95/p99 (inter-token latency).
- Pitfall: Don't answer "TPOT p50/p95/p99 (inter-token latency)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: TPOT p50/p95/p99 (inter-token latency)

### Tokens/sec throughput
- Direct Answer: Tokens/sec throughput
- Why: This matters because it tells you how to reason about tokens/sec throughput.
- Pitfall: Don't answer "Tokens/sec throughput" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Tokens/sec throughput

### GPU utilization (target > 80%)
- Direct Answer: GPU utilization (target > 80%)
- Why: This matters because it tells you how to reason about gpu utilization (target > 80%).
- Pitfall: Don't answer "GPU utilization (target > 80%)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: GPU utilization (target > 80%)

### KV cache block utilization
- Direct Answer: KV cache block utilization
- Why: This matters because it tells you how to reason about kv cache block utilization.
- Pitfall: Don't answer "KV cache block utilization" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: KV cache block utilization

### Queue depth
- Direct Answer: Queue depth
- Why: This matters because it tells you how to reason about queue depth.
- Pitfall: Don't answer "Queue depth" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Queue depth

### Measuring only total latency
- Direct Answer: hides whether prefill or decode is the bottleneck
- Why: This matters because it tells you how to reason about measuring only total latency.
- Pitfall: Don't answer "Measuring only total latency" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: hides whether prefill or decode is the bottleneck

### Alerting only on mean latency
- Direct Answer: p99 can be 5× mean under realistic load distributions
- Why: This matters because it tells you how to reason about alerting only on mean latency.
- Pitfall: Don't answer "Alerting only on mean latency" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: p99 can be 5× mean under realistic load distributions

### No per-request tracing
- Direct Answer: can't reproduce failures or attribute latency spikes
- Why: This matters because it tells you how to reason about no per-request tracing.
- Pitfall: Don't answer "No per-request tracing" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: can't reproduce failures or attribute latency spikes

### Measuring only total latency without phase breakdown
- Direct Answer: Measuring only total latency without phase breakdown
- Why: This matters because it tells you how to reason about measuring only total latency without phase breakdown.
- Pitfall: Don't answer "Measuring only total latency without phase breakdown" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Measuring only total latency without phase breakdown

### Conflating p50 and p99
- Direct Answer: LLM latency distributions have heavy tails
- Why: This matters because it tells you how to reason about conflating p50 and p99.
- Pitfall: Don't answer "Conflating p50 and p99" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: LLM latency distributions have heavy tails

### Not tracking KV cache pressure as a leading indicator of throughput collapse
- Direct Answer: Not tracking KV cache pressure as a leading indicator of throughput collapse
- Why: This matters because it tells you how to reason about not tracking kv cache pressure as a leading indicator of throughput collapse.
- Pitfall: Don't answer "Not tracking KV cache pressure as a leading indicator of throughput collapse" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not tracking KV cache pressure as a leading indicator of throughput collapse

### Classifier latency adds to TTFT
- Direct Answer: must be sub-10ms or route async
- Why: This matters because it tells you how to reason about classifier latency adds to ttft.
- Pitfall: Don't answer "Classifier latency adds to TTFT" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: must be sub-10ms or route async

### Misclassification
- Direct Answer: routing a complex reasoning task to the small model, then having to re-run → 2× cost
- Why: This matters because it tells you how to reason about misclassification.
- Pitfall: Don't answer "Misclassification" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: routing a complex reasoning task to the small model, then having to re-run → 2× cost

### Quality regression monitoring
- Direct Answer: must track quality per route, not just average
- Why: This matters because it tells you how to reason about quality regression monitoring.
- Pitfall: Don't answer "Quality regression monitoring" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: must track quality per route, not just average

### Routing only by user tier or request metadata, ignoring content complexity
- Direct Answer: Routing only by user tier or request metadata, ignoring content complexity
- Why: This matters because it tells you how to reason about routing only by user tier or request metadata, ignoring content complexity.
- Pitfall: Don't answer "Routing only by user tier or request metadata, ignoring content complexity" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Routing only by user tier or request metadata, ignoring content complexity

### Not A/B testing the routing policy against a baseline
- Direct Answer: Not A/B testing the routing policy against a baseline
- Why: This matters because it tells you how to reason about not a/b testing the routing policy against a baseline.
- Pitfall: Don't answer "Not A/B testing the routing policy against a baseline" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Not A/B testing the routing policy against a baseline

### Forgetting to monitor quality separately per route (degradation is invisible in aggregate metrics)
- Direct Answer: Forgetting to monitor quality separately per route (degradation is invisible in aggregate metrics)
- Why: This matters because it tells you how to reason about forgetting to monitor quality separately per route (degradation is invisible in aggregate metrics).
- Pitfall: Don't answer "Forgetting to monitor quality separately per route (degradation is invisible in aggregate metrics)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Forgetting to monitor quality separately per route (degradation is invisible in aggregate metrics)
