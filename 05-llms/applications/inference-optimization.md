# LLM Inference Optimization

Running an LLM in production is not the same problem as training one.

Training is a one-time cost you amortize over the model's lifetime. Inference is the ongoing bill — every query, every user, every second your product is alive. Getting inference wrong means you either spend too much or serve too slowly. Usually both.

This file is about making inference fast and cheap without breaking the model.

## The Real Question

Before reaching for any optimization, ask:

- Is the bottleneck **memory bandwidth** or **compute**?
- Is the bottleneck in **prefill** (processing the prompt) or **decode** (generating tokens)?
- Is the system **latency-bound** (interactive, must respond in <2s) or **throughput-bound** (batch, maximize tokens/sec)?

The answer changes which lever you pull.

---

## Cost Modeling

Before you optimize, you have to know what you are optimizing.

The key unit is **tokens**. Everything else derives from it.

### Primary cost drivers

| Driver | What it means |
|---|---|
| Tokens per second (TPS) | How fast the model generates. Higher = lower latency or more capacity. |
| GPU-hours | Total compute consumed. Drives the cloud bill. |
| $/1M tokens | The normalized cost metric that makes different deployments comparable. |

### A back-of-envelope for $/1M tokens

```
cost = (GPU hourly rate × latency_per_request × requests_per_hour) / tokens_per_request
```

For a 7B model on an A10G ($1.50/hr), generating 200 tokens at 40 tok/s per request with 60 concurrent requests:

- Time per request: 200/40 = 5s
- Requests/hr per GPU: 3600/5 = 720
- Tokens/hr: 720 × 200 = 144,000
- Cost/1M tokens: ($1.50 / 144,000) × 1,000,000 ≈ $10.40

That number gets you a baseline. Then optimization is about pulling that number down.

### Where cost hides

- **KV cache memory** forces smaller batch sizes, limiting throughput.
- **Prefill latency** dominates for RAG workloads with long contexts.
- **Decode latency** dominates for chat workloads with long outputs.
- **Cold starts** matter for serverless deployments — a model that takes 30s to load cannot serve interactive traffic.

---

## KV Cache

This is the most important optimization in autoregressive inference. Almost everything else builds on it.

### What it stores

During the forward pass, each transformer layer computes Key (K) and Value (V) matrices from the input tokens. For previously seen tokens, these computations do not change — they only depend on that token's content and position, which are fixed once generated.

The KV cache stores K and V matrices for every layer, for every token already in the context window. On the next generation step, only the new token's Q is computed fresh. K and V for all prior tokens are read from cache.

This converts attention from O(n²) per step to O(n) per step — but it moves the bottleneck to **memory bandwidth**, not compute.

### Memory footprint

For a single request with sequence length S:

```
KV cache size = 2 × num_layers × num_heads × head_dim × S × bytes_per_element
```

For a 7B model (32 layers, 32 heads, 128 head dim, fp16):

```
2 × 32 × 32 × 128 × S × 2 bytes = 524,288 × S bytes ≈ 0.5 MB per token
```

A 4,096 token context: ~2 GB just for KV cache. For a batch of 16 requests: ~32 GB. This is why batching LLM inference is hard — the GPU runs out of KV cache memory before it runs out of compute.

### PagedAttention

PagedAttention (vLLM, 2023) applies the OS virtual memory idea to KV cache management.

Instead of allocating a contiguous block of memory per request (which wastes memory on over-allocation and fragments GPU RAM), PagedAttention divides the KV cache into fixed-size "pages" and maps them non-contiguously. A request only holds the pages it actually needs, and pages are allocated on demand.

Results: near-zero memory waste from fragmentation, 2–4× higher throughput compared to naive allocation because more requests fit in GPU memory simultaneously.

This is the core reason vLLM outperforms naive Hugging Face inference at scale.

---

## Flash Attention

Transformers have two bottlenecks: compute (matmul) and IO (reading/writing to HBM).

Standard attention reads Q, K, V from GPU High-Bandwidth Memory (HBM), computes the attention matrix (n × n), writes it back to HBM, reads it again, applies softmax, multiplies by V, writes the output back. This is O(n²) in IO even though the useful compute is O(n²) FLOPS.

### Flash Attention (Dao et al., 2022)

Flash Attention reorganizes the attention computation to stay in SRAM (fast on-chip memory) using **tiling** — it processes the attention matrix in blocks that fit in SRAM, never materializing the full n × n matrix in HBM.

IO complexity drops from O(n²) to O(n²/block_size), which is effectively O(n) for realistic sequence lengths.

Wall-clock speedup:
- 2–4× faster attention computation
- 10–20% end-to-end training speedup
- Memory footprint of attention drops from O(n²) to O(n)

### Flash Attention 2 (Dao, 2023)

Flash Attention 2 adds:
- Better parallelization across the sequence dimension (not just batch/head)
- Fewer non-matmul FLOPs (removes unnecessary operations in the softmax)
- Support for causal masking with no extra cost

Practical gains: 2× over FA1, ~50-73% of theoretical peak A100 MFU (Model FLOP Utilization).

### When it matters

Flash Attention matters most for:
- Long context (>2,048 tokens) — the IO savings compound with sequence length
- Training — memory reduction allows larger batch sizes, better GPU utilization
- Inference with long prompts — prefill phase benefits most

For short-context decode-heavy workloads (32-token outputs), Flash Attention is less impactful because attention is not the bottleneck — memory bandwidth for KV cache reads dominates.

---

## Speculative Decoding

Autoregressive generation is inherently sequential — each token requires a forward pass of the full model. This is hard to parallelize.

Speculative decoding (Leviathan et al., 2022; Chen et al., 2023) breaks this by using a small **draft model** to propose multiple tokens, then verifying them in parallel with the **target model**.

### How it works

1. **Draft**: A small model (e.g., 7B drafts for a 70B target) generates K candidate tokens autoregressively. This is fast because the draft model is small.
2. **Verify**: The target model processes all K tokens in a single forward pass (parallel, not sequential). It checks: for each draft token, would the target model have generated that token?
3. **Accept/Reject**: Accept all tokens up to the first mismatch. If the draft was wrong at position i, reject from position i onward and use the target's prediction at that position.
4. **Repeat**.

### Why it works without changing output distribution

The verification step uses a carefully designed rejection sampling scheme that guarantees the accepted output distribution is identical to the target model's distribution. This is not an approximation — the output is mathematically identical to running the target model alone.

### Speedup

- Wall-clock speedup of 2–4× on tasks where the draft model's acceptance rate is high (>70%).
- Best results on: repetitive text, code generation, structured outputs, templated formats.
- Worst results on: creative generation, highly uncertain tasks, heavily sampled (temperature > 1) outputs.

### Practical considerations

- Draft model must share the same tokenizer and vocabulary as the target.
- Draft model must be fast enough that the overhead of generating K draft tokens is less than the time saved on verification. Rule of thumb: draft model should be 10× smaller than target.
- K is tunable: K=4-8 is typical. Too high and rejection rate increases and wastes compute; too low and you capture fewer speedup opportunities.

---

## Quantization at Inference

Quantization reduces numerical precision to save memory and increase throughput.

### Precision options

| Format | Bits | Memory vs FP32 | Typical quality loss |
|---|---|---|---|
| FP32 | 32 | 1× | Baseline |
| BF16 / FP16 | 16 | 0.5× | Negligible |
| INT8 | 8 | 0.25× | Small (< 1% on most benchmarks) |
| INT4 / NF4 | 4 | 0.125× | Noticeable on precision tasks |
| INT2 / binary | 2-1 | Extreme | Significant degradation |

### What gets quantized

- **Weights only** (W4A16, W8A16): Weights are stored in lower precision but dequantized to fp16 for compute. Reduces memory footprint, limited compute speedup.
- **Weights and activations** (W8A8): Both stored and computed in low precision. Requires specific hardware support (NVIDIA A100+ has INT8 tensor cores). Real throughput speedup.

### Methods

**GPTQ** (post-training quantization): Quantizes layer by layer using Hessian-based information to minimize weight error. Good quality at INT4, popular for deployment.

**AWQ (Activation-aware Weight Quantization)**: Identifies important weights by looking at activation magnitudes, protects them from aggressive quantization. Often better than GPTQ at INT4.

**SmoothQuant**: Migrates quantization difficulty from activations to weights (activations are harder to quantize than weights). Enables W8A8 quantization with minimal quality loss.

**bitsandbytes (NF4)**: Used in QLoRA. Optimal 4-bit quantization for normally distributed weights. Good for fine-tuning; not necessarily optimal for inference throughput.

### What to watch for

Quantization degrades non-uniformly across tasks:
- Arithmetic reasoning, coding, factual recall: more sensitive to precision loss
- Creative generation, summarization: less sensitive
- Always evaluate on your actual task, not just perplexity

Quantization affects latency differently depending on hardware:
- On NVIDIA Ampere (A100), INT8 tensor cores give real speedup for W8A8
- On consumer GPUs (RTX 3090), INT4 weight-only reduces memory but may not speed up compute (no INT4 tensor cores)

---

## Batching Strategies

A single user request rarely saturates a GPU. Batching combines multiple requests to increase hardware utilization.

### Static batching

All requests in a batch must complete before any are returned. The batch is padded to the longest sequence.

- Simple to implement
- Wastes compute on padding tokens
- Fast requests block behind slow ones (head-of-line blocking)
- Acceptable for offline batch processing, not for interactive use

### Dynamic batching

Requests arriving within a time window (e.g., 5ms) are grouped into a batch. The window is a latency/throughput knob.

- Better GPU utilization than static
- Still has head-of-line blocking within a batch
- Latency is bounded by the window size + the slowest request in the batch

### Continuous batching (iteration-level scheduling)

Pioneered by Orca (Yu et al., 2022), adopted by vLLM, TGI.

Instead of batching at the request level, the scheduler operates at the **token generation step** level. After each decode step, completed sequences are evicted and new requests are inserted into the batch immediately.

```
Step 1: [Request A token 1] [Request B token 1] [Request C token 1]
Step 2: [Request A token 2] [Request B token 2] [Request D token 1]  <- C finished, D inserted
Step 3: [Request A token 3] [Request E token 1] [Request D token 2]  <- B finished, E inserted
```

This eliminates head-of-line blocking and keeps GPU utilization high throughout.

Results: 2–10× throughput improvement over static batching for mixed-length workloads.

The downside: more complex scheduling logic. PagedAttention makes this feasible by allowing flexible KV cache allocation per request.

---

## Deployment Frameworks: vLLM vs TensorRT-LLM vs llama.cpp

No single framework wins everywhere. Choose based on constraints.

### vLLM

**Best for:** Production serving on NVIDIA GPUs with variable-length interactive workloads.

Key features:
- PagedAttention for efficient KV cache management
- Continuous batching built-in
- OpenAI-compatible API server out of the box
- Supports tensor parallelism across multiple GPUs
- Wide model support (Llama, Mistral, Falcon, Mixtral, etc.)

Limitations:
- NVIDIA-only (AMD ROCm support is experimental)
- Higher memory overhead than llama.cpp for small single-user deployments
- Startup time is slow for large models

Use when: you are running a multi-user inference service and need maximum throughput.

### TensorRT-LLM

**Best for:** Maximum throughput on NVIDIA hardware when you control the deployment environment.

Key features:
- Compiled engines optimized for specific GPU + model + precision combinations
- Fused CUDA kernels, INT8/FP8 support, in-flight batching
- Often 2–4× faster than vLLM for batch inference on A100/H100
- Supports multi-GPU tensor parallelism and pipeline parallelism

Limitations:
- Engines are hardware-specific: a compiled engine for A100 will not run on V100
- Compilation takes 20–60 minutes per model/configuration
- Harder to update models quickly (recompilation required)
- Less flexible for rapid experimentation

Use when: you have fixed hardware, a stable model, and need to squeeze every last token/sec out of the GPU.

### llama.cpp

**Best for:** CPU inference, edge deployment, local development, and quantized inference on consumer hardware.

Key features:
- Pure C/C++ with minimal dependencies — runs on Mac (Metal), CPU, Raspberry Pi
- GGUF format with multiple quantization levels (Q2 to Q8)
- 4-bit quantization (Q4_K_M) runs 7B models on 8GB RAM
- Streaming generation, OpenAI-compatible server mode

Limitations:
- Significantly slower than GPU inference for high-throughput serving
- No continuous batching (single-user focus)
- Less suitable for production multi-user serving

Use when: you are developing locally, running on consumer hardware, or deploying to edge devices without GPUs.

### Decision table

| Constraint | Recommendation |
|---|---|
| Multi-user API server, NVIDIA GPU | vLLM |
| Maximum throughput, fixed NVIDIA hardware, stable model | TensorRT-LLM |
| Local development, Mac, CPU, edge | llama.cpp |
| Experimenting with quantization quickly | llama.cpp or vLLM with bitsandbytes |
| A100/H100 cluster, production ML team | TensorRT-LLM or vLLM |

---

## Azure / DevOps Bridge

Inference optimization is a deployment concern, not just a model concern.

Think of inference configuration like any other service configuration:

- **Quantization level** = a versioned artifact parameter. Change it, validate it, roll it back if quality drops.
- **Batch size / continuous batching** = a scaling parameter. Tune it per load profile, not once and forgotten.
- **KV cache capacity** = a capacity planning parameter. Size it based on max concurrent users × average context length.
- **Model version** = a deployment artifact. Keep the previous version warm for fast rollback.

A strong LLM inference pipeline validates:

- Token throughput and latency P50/P95/P99 under load
- Output quality on a regression suite (do not deploy a quantized model without quality gate)
- Memory headroom under peak batch sizes
- Graceful degradation when GPU memory is exhausted (queue or reject, not crash)

---

## Quick Decision Tree

```
Is the model too slow?
├── Is it slow on the first token (prefill)?
│   ├── Yes → Long prompt? → Use Flash Attention 2, prefix caching, chunked prefill
│   └── Yes → Short prompt? → Profile — may be cold start or tokenization overhead
└── Is it slow on generation (decode)?
    ├── Low throughput (many users)? → Continuous batching, PagedAttention, larger batch size
    └── High latency (single user)? → Speculative decoding, smaller model, quantization

Is the model too expensive?
├── Memory too high? → Quantization (W4A16 or W8A8), reduce KV cache with shorter context
└── Compute too high? → Flash Attention, better batching, smaller model for the task
```

---

## Interview Q&A

**Q1. What is the KV cache and why does it matter for inference?**

The KV cache stores the Key and Value matrices from the attention mechanism for all previously generated tokens. On each new decode step, only the new token's Query is computed; K and V for prior tokens are read from cache. Without caching, attention would be O(n²) per step (full recomputation). With caching, it is O(n) per step. The tradeoff: cache size grows linearly with sequence length × batch size × num_layers, which is the primary reason GPU memory fills up quickly during inference on long contexts.

---

**Q2. What is PagedAttention and what problem does it solve?**

PagedAttention solves KV cache memory fragmentation. Naively, each request gets a contiguous memory block sized to its maximum possible sequence length — most of which sits empty until tokens are generated. On a busy server, this fragmentation means the GPU runs out of memory despite much of it being technically unused.

PagedAttention maps KV cache memory as fixed-size non-contiguous pages, allocated on demand as each token is generated. This brings memory utilization close to 100% and allows far more concurrent requests to share the GPU. The throughput improvement is typically 2–4×.

---

**Q3. How does speculative decoding work and when should you use it?**

A small draft model generates K candidate tokens. The large target model verifies all K in one parallel forward pass. Tokens are accepted up to the first mismatch; the target's prediction at the mismatch position is used instead. The output distribution is mathematically identical to running the target alone.

Use it when: the draft model's acceptance rate is high (>70%), which happens on structured outputs, code, templated formats, or tasks with low entropy (predictable next tokens). Avoid it on creative generation or high-temperature sampling, where the draft's predictions will be wrong too often and the overhead of running two models dominates.

---

**Q4. Explain Flash Attention — what problem does it solve and how?**

Standard attention writes the full n × n attention matrix to HBM (GPU main memory), reads it back for softmax, reads it again for the weighted sum. This is IO-bound — the math is fast but memory transfers are slow.

Flash Attention keeps the attention computation in SRAM (fast on-chip memory) using tiling: it processes the attention matrix in blocks that fit in SRAM, never writing the full n × n matrix to HBM. IO complexity drops from O(n²) to effectively O(n). Speedup is 2–4× for the attention operation, with memory usage also dropping from O(n²) to O(n).

Flash Attention 2 adds better parallelism across the sequence dimension and fewer non-matmul operations, reaching ~70% of peak A100 throughput.

---

**Q5. What is the difference between W4A16 and W8A8 quantization?**

W4A16: weights stored in 4-bit, activations computed in fp16. This reduces model memory footprint by 4× but does not directly speed up matrix multiply (activations must be dequantized back to fp16 before compute). Throughput gain comes mainly from reduced memory bandwidth pressure.

W8A8: both weights and activations stored and computed in 8-bit. Requires hardware with INT8 tensor cores (NVIDIA A100+). This gives real compute speedup in addition to memory savings — INT8 tensor cores are roughly 2× faster than FP16 on A100.

For most consumer or older GPU deployments, W4A16 is the practical choice. For A100/H100 production deployments where throughput matters, W8A8 (via SmoothQuant or similar) is worth evaluating.

---

**Q6. What is continuous batching and why is it better than static batching?**

Static batching holds a batch together until all requests complete. If one request generates 500 tokens and another generates 10, the GPU sits partially idle while waiting for the slow request.

Continuous batching (iteration-level scheduling) evicts completed sequences after each decode step and inserts new requests immediately. The GPU always has a full batch of active requests. For mixed-length workloads (typical in production), this gives 2–10× throughput improvement over static batching.

---

**Q7. When would you choose vLLM over TensorRT-LLM?**

vLLM when: you need flexibility — quick model updates, mixed model types, or teams that cannot afford 60-minute compilation cycles. Also when you have variable-length interactive traffic that benefits most from continuous batching and PagedAttention.

TensorRT-LLM when: you have fixed hardware (A100/H100), a stable model that changes infrequently, and throughput is the primary KPI. Compiled TensorRT engines can be 2–4× faster than vLLM for pure batch throughput on the same hardware. The compilation cost is paid once; the throughput gain is permanent.

---

**Q8. Your LLM inference is slow. Walk through your debugging process.**

First, isolate the phase. Measure time-to-first-token (prefill latency) separately from token generation throughput (decode).

If prefill is slow: the bottleneck is processing the prompt. Check: is the context long? Flash Attention 2 helps. Are you re-processing the same system prompt every request? Prefix caching eliminates that.

If decode is slow (low tokens/sec): check GPU utilization. If utilization is low, the bottleneck is memory bandwidth — the model is too large for the GPU, or batching is poor. Solutions: reduce model size (quantize), improve batching (continuous batching), or add GPUs (tensor parallelism).

If latency is high for a single user but throughput per GPU looks okay: consider speculative decoding for interactive workloads.

Always profile with tools (nsys, nvtop, or vLLM's built-in metrics) before guessing.

---

**Q9. What is the "memory wall" problem in LLM inference?**

Modern GPUs have far more compute (FLOPS) than memory bandwidth. A100 has 312 TFLOPS (fp16) but only 2 TB/s memory bandwidth. For small batch sizes, the attention and weight-loading operations are memory-bandwidth-limited — the GPU's compute sits idle waiting for data to arrive from HBM.

This is why batching helps: larger batch sizes reuse the same weights across many requests, amortizing the memory transfer cost. It also explains why quantization helps even without faster compute: smaller weights transfer faster through the same bandwidth-limited bus.

The KV cache makes this worse because it adds more memory traffic proportional to sequence length. PagedAttention and Flash Attention both address aspects of this memory wall.

---

**Q10. A team wants to serve a 70B model on 2× A100 80GB GPUs. What do you consider?**

First, memory budget: a 70B model in fp16 needs ~140 GB. Two A100s give 160 GB total — tight. After model weights, there is ~20 GB for KV cache and activations. At ~0.5 MB per token per request, this supports roughly 40 concurrent requests with 1,000 token contexts. That may be insufficient for production.

Options: (1) Use INT4 quantization (AWQ/GPTQ) to bring the model to ~35 GB, freeing ~125 GB for KV cache. (2) Use tensor parallelism across both GPUs, splitting attention heads and FFN layers. vLLM and TensorRT-LLM both support this natively.

For the serving framework: if throughput is primary, TensorRT-LLM compiled for A100. If flexibility and faster iteration are needed, vLLM with tensor-parallel=2.

Monitor: KV cache utilization, batch size under load, time-to-first-token P95. If TTFT is too high, consider chunked prefill (process long prompts in chunks, interleaved with decode steps, so decode requests are not blocked).
