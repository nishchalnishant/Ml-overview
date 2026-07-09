---
module: Llms
topic: Applications
subtopic: Inference Optimization
status: unread
tags: [llms, ml, applications-inference-optimiz]
---
# Inference Optimization

---

## The Asymmetry Between Training and Inference

**The problem:** training an LLM is a one-time cost. Inference is the ongoing bill — every user query, every API call, every second the product is alive. A model that takes 1 second per 100 tokens may be acceptable for an internal tool but unacceptable for a consumer product requiring sub-200ms time-to-first-token. A model costing $50/hour in GPU time may be fine for a single user but untenable at 10,000 concurrent users.

The root issue is that autoregressive generation is inherently sequential. Each output token requires a full forward pass through all transformer layers. These forward passes cannot be parallelized across output tokens. The model produces one token at a time, and every optimization must work within this constraint.

**Before optimizing, ask:** Is the bottleneck in prefill (processing the prompt) or decode (generating tokens)? Is the bottleneck memory bandwidth or compute? Is the deployment latency-bound (interactive, single user) or throughput-bound (batch processing, many users)? The answer changes which lever to pull.

---

## The KV Cache

**The problem:** in a naive implementation, every new decoding step recomputes the Key and Value matrices for every token in the current sequence — including all previously generated tokens. For a sequence of length n after generating t tokens, that is O((n+t)²) attention computation at each new step. As the sequence grows, each step becomes more expensive.

**The core insight:** the Key and Value matrices for any token depend only on that token's embedding and its position, which are fixed once generated. They do not change when new tokens are appended. There is no reason to recompute them.

**The mechanics:** store K and V for every layer and every previously generated token. On the next step, compute K and V only for the new token, append them to the cache, and compute attention using all cached K/V pairs. Cost per step drops from O(n²) to O(n) — but this introduces a memory cost.

**KV cache memory footprint:**
```
per request = 2 × num_layers × num_heads × head_dim × sequence_length × bytes_per_element

For a 7B model (32 layers, 32 heads, 128 head_dim, fp16):
2 × 32 × 32 × 128 × S × 2 bytes = 0.5 MB per token

At 4096 tokens: ~2 GB per request
At batch of 16: ~32 GB — fills a full A100
```

The KV cache is the primary reason batching LLM inference is hard: the GPU runs out of KV cache memory before it runs out of compute.

**What breaks:** KV cache memory scales linearly with sequence length and batch size. Long context or large batches cause out-of-memory errors. Naive memory allocation wastes GPU RAM through fragmentation — pre-allocating maximum sequence length per request leaves most of the allocation empty until tokens are generated.

---

## PagedAttention

**The problem:** naive KV cache allocation gives each request a contiguous pre-allocated block sized to its maximum possible sequence length. On a busy server with hundreds of concurrent requests of varying lengths, most of each block is empty. This internal fragmentation wastes 50–80% of GPU memory and severely limits how many requests can be served simultaneously.

**The core insight:** operating systems solved this problem for RAM decades ago with virtual memory and paging. Apply the same idea to the KV cache. Instead of one contiguous block per request, divide the KV cache into fixed-size "pages" and allocate them on demand as tokens are actually generated. Map pages non-contiguously through a page table — exactly as OS virtual memory works.

**The mechanics:** each block holds the KV entries for a fixed number of tokens (e.g., 16). Requests start with no allocated blocks. As each token is generated, the system allocates the next block only when the current one is full. The attention kernel uses the page table to find where each block lives in physical memory.

**What breaks:** the indirection through a page table adds a small overhead to the attention kernel. More complex memory management code. Requires a serving framework that implements PagedAttention — vLLM pioneered this; it is now standard in most production LLM servers.

---

## FlashAttention

**The problem:** standard attention writes the full n×n score matrix to GPU high-bandwidth memory (HBM) to compute softmax, then reads it back for the weighted sum of values. At n=8192 in float16, this is ~256 MB of HBM traffic per attention layer. HBM bandwidth (~2 TB/s on A100) is the bottleneck — the arithmetic units idle while waiting for memory transfers.

**The core insight:** the n×n matrix is a transient intermediate result. It does not need to exist as a complete tensor in HBM. It can be computed tile by tile in SRAM (fast on-chip cache, ~192 KB on A100), with softmax numerics tracked via a running max and running sum (online softmax). The output accumulates incrementally. The full n×n matrix is never written to or read from HBM.

**The mechanics:**
1. Tile Q, K, V into blocks that fit in SRAM.
2. For each Q tile, iterate over K and V tiles.
3. Maintain running softmax statistics to combine results across tiles.
4. Accumulate output — the n×n matrix is never materialized.

Memory: O(n) instead of O(n²). FLOPs are unchanged (still O(n²) arithmetic). Speedup: 2–4× wall-clock time for typical sequence lengths. HBM reads/writes drop from O(n²) to O(n²/M) where M is SRAM size.

FlashAttention-2 improved parallelism across the sequence dimension. FlashAttention-3 adds pipeline overlap for H100 hardware. For short-context decode-heavy workloads, FlashAttention is less impactful because KV cache memory reads dominate, not the attention matrix.

**What breaks:** requires custom CUDA kernels — cannot be expressed as naive PyTorch. Hardware-specific: tiling must fit SRAM, which differs across GPU generations. The backward pass for training recomputes attention tiles, adding compute overhead but saving memory.

---

## Quantization

**The problem:** a 70B parameter model in float32 weighs 280 GB. In float16 or bfloat16, 140 GB. Loading 140 GB of weights per inference at HBM bandwidth limits how fast tokens can be generated. Smaller precision means smaller model, faster memory transfer, potentially faster compute — but accuracy must not degrade unacceptably.

**The core insight:** most model weights and activations do not need 16-bit precision for accurate inference. The information required for accurate next-token prediction can often be maintained with 4–8 bits, especially with calibration-aware quantization that carefully maps the weight distribution to the reduced precision range.

**The mechanics:**

*W4A16 (weight-only 4-bit):* store weights in 4-bit; dequantize to fp16 before matrix multiply. Memory footprint drops 4×. Throughput gain comes from reduced memory bandwidth pressure. No compute speedup from INT4 tensor cores because dequantization happens before compute.

*W8A8 (weight + activation 8-bit):* store and compute in 8-bit. Requires hardware support (NVIDIA A100+ INT8 tensor cores). Provides real compute speedup (~2×) in addition to memory savings.

*GPTQ:* post-training quantization that minimizes weight error layer by layer using the Hessian of the loss. Better quality than naive rounding at INT4.

*AWQ:* identifies "salient" weights — those whose magnitudes are amplified by large activations — and protects them from aggressive quantization. Often better than GPTQ at INT4.

*SmoothQuant:* activations have more extreme outliers than weights, making them harder to quantize. SmoothQuant migrates quantization difficulty from activations to weights via a per-channel smoothing factor, enabling accurate W8A8 quantization.

**What breaks:** quantization degrades non-uniformly across tasks. Arithmetic reasoning, code generation, and factual recall are more sensitive to precision loss than creative generation or summarization. Always evaluate on the target task — perplexity improvements do not guarantee task-specific quality. On consumer GPUs without INT8 tensor cores, W4A16 reduces memory but may not speed up compute.

---

## Batching Strategies

**The problem:** a single inference request rarely saturates a GPU. At batch size 1, GPU utilization on a 70B model is typically 1–5% of theoretical FLOPs — the GPU idles waiting for memory transfers. Combining multiple requests amortizes the cost of loading weights and improves utilization.

### Static Batching

All requests in the batch start and end together. The batch is padded to the longest sequence. Simple but wasteful: fast requests block waiting for slow ones (head-of-line blocking), and padding tokens consume compute for no reason.

### Dynamic Batching

Group requests arriving within a time window. Better utilization than static. Still suffers from head-of-line blocking within a batch.

### Continuous Batching (Iteration-Level Scheduling)

**The problem:** in static and dynamic batching, the GPU idles when some requests in a batch finish early while others continue generating. The early-finishing slots sit empty for the rest of the batch.

**The core insight:** batch at the token generation step level rather than the request level. After each decode step, completed sequences are evicted and new requests are immediately inserted into their slots. The GPU always has a full batch of active requests.

```
Step 1: [Request A: token 12] [Request B: token 7] [Request C: token 3]
Step 2: [Request A: token 13] [Request B: token 8] [Request D: token 1]  ← C finished, D inserted
Step 3: [Request A: token 14] [Request E: token 1] [Request D: token 2]  ← B finished, E inserted
```

This eliminates head-of-line blocking. For mixed-length workloads, continuous batching gives 2–10× throughput improvement over static batching. PagedAttention is what makes this feasible — flexible KV cache allocation per request allows dynamic insert and evict.

---

## Speculative Decoding

Autoregressive decode is memory-bandwidth bound (~70ms/token on a 140GB model at 2TB/s HBM), not compute bound — a small draft model proposes K tokens, the target model verifies all K in one parallel pass, and rejection sampling guarantees the output distribution is unchanged. Speedup depends entirely on the draft model's acceptance rate.

> Full mechanics, acceptance-rate math, and failure modes: [07-speculative-decoding.md](07-speculative-decoding.md).

---

## Serving Frameworks

### The Problem They Solve

Running a raw model with naive HuggingFace inference is 10–100× less efficient than production serving frameworks. Raw inference allocates memory poorly, batches naively, and does not implement PagedAttention, FlashAttention, or continuous batching. Production frameworks provide all of these.

### vLLM

**Best for:** multi-user API servers on NVIDIA GPUs with variable-length interactive workloads.

Key innovations: PagedAttention for KV cache management, continuous batching, OpenAI-compatible API server, wide model support. The standard starting point for production serving.

**What breaks:** NVIDIA-only (AMD ROCm support is experimental). Higher memory overhead than llama.cpp for single-user local deployment. Slow startup for very large models.

### TensorRT-LLM

**Best for:** maximum throughput on NVIDIA hardware when the model and hardware are fixed.

Key innovations: ahead-of-time compiled engines with fused CUDA kernels optimized for a specific GPU + model + precision combination. Often 2–4× faster than vLLM for batch inference on A100/H100 for the same model.

**What breaks:** engines are hardware-specific — an A100 engine will not run on V100. Compilation takes 20–60 minutes per model/configuration. Updating the model requires recompilation. Not suitable for rapid iteration or mixed model deployments.

### llama.cpp

**Best for:** CPU inference, Mac (Apple Metal), local development, quantized inference on consumer hardware.

Key innovations: pure C/C++ with minimal dependencies, GGUF quantization format (Q2 through Q8), runs 7B models in 8 GB RAM with Q4_K_M quantization. Streaming generation, OpenAI-compatible server mode.

**What breaks:** significantly slower than GPU inference at high throughput. No continuous batching. Not suitable for production multi-user serving at scale.

### Decision Table

| Constraint | Recommendation |
|:---|:---|
| Multi-user API server, NVIDIA GPU | vLLM |
| Maximum throughput, fixed NVIDIA hardware, stable model | TensorRT-LLM |
| Local development, Mac, CPU, consumer GPU | llama.cpp |
| Rapid experimentation with quantization | llama.cpp or vLLM with bitsandbytes |
| A100/H100 cluster, production team | TensorRT-LLM or vLLM |

---

## Debugging Slow Inference

**The problem:** "the model is slow" is not actionable. Slow prefill and slow decode have different causes and different fixes.

**The core insight:** isolate the phase first, then identify the bottleneck within that phase.

- **Slow time-to-first-token (prefill):** the bottleneck is processing the prompt. Long prompts → FlashAttention-2, chunked prefill, prefix caching. Same system prompt repeated across requests → prefix caching eliminates re-encoding cost.
- **Slow token generation (decode):** check GPU utilization. If utilization is low, the GPU is memory-bandwidth-bound — it is waiting for weight reads from HBM, not doing compute. Solutions: reduce model size (quantize), improve batching (continuous batching), or add GPUs (tensor parallelism).
- **High latency for a single user but acceptable throughput:** speculative decoding reduces per-token latency for a single request.

Always profile before optimizing: nvtop, nsys, vLLM's built-in metrics endpoint. Guessing the bottleneck without measurement leads to optimizing the wrong thing.

---

## The Memory Wall

**The problem:** modern GPUs have vastly more compute than memory bandwidth. An A100 has 312 TFLOPS (fp16) but only 2 TB/s HBM bandwidth. For small batch sizes, LLM inference is almost entirely memory-bandwidth-bound: the GPU's arithmetic units idle while waiting for weight tensors to arrive from HBM.

**The core insight:** for memory-bandwidth-bound workloads, the bottleneck is how fast data moves, not how fast it is computed. Optimizations that reduce data movement (FlashAttention, KV cache, quantization) are more effective than optimizations that reduce FLOPs.

Larger batches help by amortizing the weight-load cost across many requests — reading the same weight matrix once and using it for 32 requests instead of 1. This is why throughput scales nearly linearly with batch size up to the compute-bound regime.

*Related: [Speculative Decoding](07-speculative-decoding.md) | [Context Window Extension](../07-context-window-extension.md) | [Tuning and Optimization](10-tuning-optimization.md)*

---

## Canonical Interview Q&As

**Q: Explain the KV cache — what exactly is cached and why is it the central bottleneck in LLM serving?**
A: During autoregressive generation, at each step the model computes attention over all previous tokens. For a sequence of length t, computing attention for the new token requires K and V projections for all t previous tokens. Without caching, this is O(t²) total compute for a sequence of length t. The KV cache stores the K and V tensors for all previous tokens — on each new step, we only compute K, V for the new token and append it. Compute drops from O(t²) to O(t) total. Memory cost: for a model with L layers, H heads, d_kv dimension, the KV cache per token = 2 × L × H × d_kv × 2 bytes (bf16). For Llama-3 70B (80 layers, 8 KV heads, d_kv=128): 2 × 80 × 8 × 128 × 2 = 327KB per token. At 32K context with batch=32: 327KB × 32K × 32 = 335GB — exceeding a single A100's 80GB. KV cache memory is the primary constraint on context length and batch size in production, which is why GQA (8 vs 64 KV heads = 8× reduction) and quantized KV caches (INT8 KV) are critical.

**Q: What is speculative decoding and when does it provide a speedup?**
A: Speculative decoding uses a small "draft" model to propose K tokens at once, then the large "target" model verifies all K in parallel in a single forward pass (since verification is parallelizable, unlike generation). If the target model accepts the draft token, it's free; if it rejects at position i, it corrects token i and discards tokens i+1..K. The speedup depends on the acceptance rate α — if α is high (draft and target agree often), you get ~K× throughput improvement with identical output distribution. The target model runs at the same quality as without speculative decoding because rejections are handled by sampling from a corrected distribution. Best case: for coding tasks where the draft model guesses common patterns correctly, α ≈ 0.8+, yielding 2-3× throughput. Worst case: for creative/diverse tasks where the draft model diverges, α ≈ 0.3, overhead from running the draft model erases the benefit. Key constraint: the draft model must run much faster than the target — typically 7B draft for 70B target (10× parameter ratio). Self-speculative decoding uses early exit from the target model itself as the draft, avoiding a separate model.

**Q: Compare continuous batching vs static batching for LLM serving — why is continuous batching essential at scale?**
A: **Static batching**: group N requests into a batch, run prefill + decode together, release the batch only when all sequences finish. Problem: sequences have different lengths — the batch must wait for the longest sequence before returning any results. A batch of 32 where one sequence generates 2000 tokens and others generate 50 tokens wastes 97.5% of compute on the short sequences while they wait. GPU utilization is low because slots are idle after short sequences finish. **Continuous batching** (iteration-level scheduling): after each decode step, evict finished sequences from the batch and immediately insert new waiting requests. The batch composition changes every step. Throughput improvement: 5-10× in practice because GPU utilization stays high. Complexity: need to manage variable batch sizes and KV cache memory dynamically. **PagedAttention** (vLLM): extends continuous batching by managing KV cache in fixed-size pages (like virtual memory), allowing non-contiguous KV cache allocation and enabling near-zero fragmentation. Without PagedAttention, KV cache fragmentation wastes 20-40% of GPU memory. Production: all serious LLM serving (vLLM, TGI, TensorRT-LLM) uses continuous batching + PagedAttention. Static batching is only viable for batch inference jobs where latency doesn't matter.

## Flashcards

**Slow time-to-first-token (prefill)?** #flashcard
the bottleneck is processing the prompt. Long prompts → FlashAttention-2, chunked prefill, prefix caching. Same system prompt repeated across requests → prefix caching eliminates re-encoding cost.

**Slow token generation (decode): check GPU utilization. If utilization is low, the GPU is memory-bandwidth-bound?** #flashcard
it is waiting for weight reads from HBM, not doing compute. Solutions: reduce model size (quantize), improve batching (continuous batching), or add GPUs (tensor parallelism).

**High latency for a single user but acceptable throughput?** #flashcard
speculative decoding reduces per-token latency for a single request.
