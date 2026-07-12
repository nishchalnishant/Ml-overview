---
module: Deep Learning
topic: LLM Serving
subtopic: Production Inference
status: unread
tags: [llm, serving, vllm, tensorrt, triton, inference, throughput, latency, quantization]
---
# LLM Serving — Production Inference Systems

**The problem:** Training a 70B parameter model takes weeks and $1M+. Serving it to millions of users is a different engineering challenge entirely — one where the bottlenecks are GPU memory capacity, memory bandwidth, and request scheduling, not compute.

**What you'll learn:**
- Why LLM inference is fundamentally different from training
- vLLM and PagedAttention — the key innovation in modern LLM serving
- Continuous batching — how to maximize GPU utilization
- Quantization for serving (AWQ, GPTQ, GGUF)
- TensorRT-LLM and Triton Inference Server
- Throughput vs. latency tradeoffs
- Production deployment patterns

---

## Table of Contents
1. [LLM Inference Fundamentals](#1-llm-inference-fundamentals)
2. [The KV Cache Problem](#2-the-kv-cache-problem)
3. [vLLM and PagedAttention](#3-vllm-and-pagedattention)
4. [Continuous Batching](#4-continuous-batching)
5. [Quantization for Serving](#5-quantization-for-serving)
6. [TensorRT-LLM](#6-tensorrt-llm)
7. [Triton Inference Server](#7-triton-inference-server)
8. [Tensor Parallelism for Serving](#8-tensor-parallelism-for-serving)
9. [Throughput vs. Latency Tradeoffs](#9-throughput-vs-latency-tradeoffs)
10. [Production Deployment Patterns](#10-production-deployment-patterns)
11. [Interview Questions](#11-interview-questions)

---

## 1. LLM Inference Fundamentals

### Two Phases of Autoregressive Generation

```
Prefill Phase:                   Decode Phase:
─────────────────                ─────────────────────────────────
"What is machine learning?"   →  "Machine" "learning" "is" "the"...
  [all at once, parallel]         [one token at a time, sequential]
  
  Compute-bound                   Memory-bandwidth-bound
  High arithmetic intensity        Low arithmetic intensity
  Efficient GPU utilization        GPU mostly idle (memory-bound)
```

**Prefill:** Process the full prompt in parallel (one forward pass). Fast, compute-intensive. Time-to-first-token (TTFT) depends on prompt length.

**Decode:** Autoregressively generate tokens one at a time. For each new token, the full model runs a forward pass — but only computing attention for the **one new token** against all cached past keys and values (KV cache). This is memory-bandwidth-bound: for a 70B model, just loading the weights takes ~140 GB of memory bandwidth per step.

**Memory bandwidth rule of thumb:** For a model with P parameters at BF16 (2 bytes/param), each decode step requires ~2P bytes of memory bandwidth. A100 SXM (2 TB/s bandwidth): max decode throughput ≈ 2 × 10¹² / (2 × 70 × 10⁹) ≈ 14 tokens/second per sample — regardless of GPU compute.

### Roofline Analysis

```
Operational Intensity = FLOPs / Bytes accessed

Prefill (sequence len N):
  FLOPs = O(N × params), Bytes = O(params + N × KV_size)
  → Compute-bound for large N

Decode (single token):
  FLOPs = O(params), Bytes = O(params + context × KV_size)
  → Always memory-bandwidth-bound

Implication: Batching multiple decode requests together is crucial
to amortize the weight loading cost across many users.
```

---

## 2. The KV Cache Problem

For autoregressive generation, keys and values for past tokens are cached to avoid recomputation:

$$\text{KV cache per token} = 2 \times n_\text{layers} \times n_\text{heads} \times d_k \times \text{bytes/element}$$

**LLaMA 2 70B (BF16) example:**
- 80 layers, 64 heads, 128 head_dim, 2 bytes/BF16
- Per token: 2 × 80 × 64 × 128 × 2 = **2.6 MB per token**
- For 4K context: ~10 GB per sequence
- For batch of 10 sequences: 100 GB — exceeds most single-GPU VRAM

With GQA (8 KV heads instead of 64): 2 × 80 × 8 × 128 × 2 = **0.33 MB per token** — 8× reduction.

### The Memory Fragmentation Problem

**Before vLLM:** Each serving system pre-allocated a fixed contiguous memory block for each request's KV cache. Problems:
- **Internal fragmentation:** Requests rarely use their full pre-allocated block (most generation stops earlier than max_len)
- **External fragmentation:** Memory holes between requests cannot be used
- Result: 20-40% of GPU memory wasted; can't batch many requests simultaneously

---

## 3. vLLM and PagedAttention

**vLLM** (Kwon et al., 2023, UC Berkeley) solves KV cache memory fragmentation by borrowing OS virtual memory concepts.

### PagedAttention

**Insight:** Treat KV cache like OS virtual memory. Divide KV cache into fixed-size **blocks** (e.g., 16 tokens per block). Maintain a **block table** mapping logical sequence positions to physical memory blocks — just like a page table.

```
Logical KV Cache (per sequence):           Physical GPU Memory (shared pool):
┌─────────────────────────────┐           ┌────────────────────────────────────┐
│ Request A: 256 tokens       │           │ Block 0  │ Block 1  │ Block 2  │...│
│ [Block 0→5] [Block 1→8]     │──────────▶│ Req A    │ Req B    │ Req A    │...│
│ [Block 2→12][Block 3→...]   │           │ tok 1-16 │ tok 1-16 │ tok 17-32│...│
│                             │           └────────────────────────────────────┘
│ Request B: 128 tokens       │
│ [Block 0→1] [Block 1→3]     │
└─────────────────────────────┘

Block table (per request):
Request A: {logical_block_0: physical_5, logical_block_1: physical_8, ...}
Request B: {logical_block_0: physical_1, logical_block_1: physical_3, ...}
```

**Benefits:**
- Blocks only allocated when needed → nearly zero wasted memory
- Non-contiguous physical blocks are fine (block table handles mapping)
- Blocks can be **shared across sequences** (e.g., system prompt shared between users)
- Copy-on-write for parallel sampling (beam search, best-of-N)

### vLLM Throughput vs. Existing Systems

In the original paper, vLLM achieved:
- **2-4× higher throughput** than HuggingFace text-generation-inference
- **Near-zero KV cache memory waste** (<4% vs 20-40% before)
- Better prefix caching (system prompt reuse across requests)

### Using vLLM

```python
from vllm import LLM, SamplingParams

# Load model with tensor parallelism across multiple GPUs
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,           # split model across 2 GPUs
    gpu_memory_utilization=0.90,      # use 90% of GPU memory for KV cache
    max_model_len=8192,               # max context length
    quantization="awq",               # use AWQ 4-bit quantization
    dtype="bfloat16",
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    stop=["<|eot_id|>"],
)

# Batch inference (most efficient)
prompts = [
    "Explain attention mechanisms in transformers.",
    "What is the difference between RLHF and DPO?",
    "How does Flash Attention reduce memory usage?",
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)

# OpenAI-compatible server (for production)
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --tensor-parallel-size 2
```

---

## 4. Continuous Batching

### The Problem with Static Batching

Traditional static batching pads all sequences in a batch to the same length and runs them together. Problem: a request that finishes early holds its GPU slot until the entire batch completes — causing GPU bubble time and reducing throughput.

```
Static batch (4 requests, pad to longest):
Request A: [tok1][tok2][tok3][pad][pad][pad]  → finishes at step 3
Request B: [tok1][tok2][tok3][tok4][tok5][tok6]  → finishes at step 6
Request C: [tok1][tok2][pad][pad][pad][pad]  → finishes at step 2
Request D: [tok1][tok2][tok3][tok4][pad][pad]  → finishes at step 4

GPU steps: [ABCD][ABCD][ABCD][BD__][BD__][B___]
                              ↑ GPU slots A,C,D wasted here
```

### Continuous Batching (Iteration-Level Batching)

**Core idea:** After each token generation step (iteration), remove finished sequences and insert new pending requests into the batch. The batch composition changes every step.

```
Continuous batching (same 4 requests + 2 new ones in queue):
Step 1: [A B C D]
Step 2: [A B C D]
Step 3: [A B _ D]  → C finishes; insert E
Step 3: [A B E D]  → new request E joins immediately
Step 4: [A B E _]  → D finishes; insert F
Step 4: [A B E F]
...

GPU is always fully utilized. No bubble time.
```

Continuous batching (Orca, Yu et al. 2022) is now the default in all serious serving systems (vLLM, TGI, TensorRT-LLM).

---

## 5. Quantization for Serving

Quantization reduces model size and increases throughput by lowering precision. For inference-only (no training), we can use more aggressive schemes than training allows.

### Post-Training Quantization (PTQ) — Serving Methods

| Method | Precision | Quantizes | Quality | Speedup | Notes |
|---|---|---|---|---|---|
| **GPTQ** | INT4/INT8 | Weights only | Good | 2-4× | Layer-by-layer reconstruction |
| **AWQ** | INT4 | Weights only | Better | 2-4× | Protects salient weights |
| **GGUF** | 2-8 bit | Weights | Good | CPU-friendly | llama.cpp format |
| **FP8** (H100) | FP8 | Weights + Activations | Best | 1.5-2× vs BF16 | Native H100 support |
| **SmoothQuant** | INT8 | Both | Good | ~1.5× | Migrates outliers to weights |

**AWQ (Activation-aware Weight Quantization):**

Key insight: not all weights are equally important. A small fraction of weights (those corresponding to high-activation channels) have disproportionate impact on output quality. AWQ protects these by finding a per-channel scale that minimizes quantization error for the salient channels.

```python
# Using AWQ with vLLM (pre-quantized model)
from vllm import LLM

# Load an AWQ pre-quantized model
llm = LLM(
    model="TheBloke/Llama-2-70B-Chat-AWQ",
    quantization="awq",
    dtype="float16",
    gpu_memory_utilization=0.9,
)

# Or quantize on the fly with autoawq
# pip install autoawq
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("llama3-8b-awq-4bit")
```

**GGUF (llama.cpp format) — CPU and consumer GPU inference:**

```python
# Using llama-cpp-python for GGUF models (works on CPU + Apple Silicon)
from llama_cpp import Llama

llm = Llama(
    model_path="llama-3.1-8b-q4_k_m.gguf",  # Q4_K_M is best quality/size ratio
    n_ctx=8192,          # context window
    n_gpu_layers=35,     # layers to offload to GPU (0 = all CPU)
    n_threads=8,
)

output = llm(
    "Q: What is machine learning?\nA:",
    max_tokens=256,
    stop=["Q:"],
    echo=False,
)
print(output["choices"][0]["text"])
```

---

## 6. TensorRT-LLM

**NVIDIA TensorRT-LLM** is NVIDIA's production LLM inference library — it compiles models to optimized TensorRT engines with kernels specifically tuned for each GPU.

### Key Optimizations

1. **Kernel fusion:** Fuse multiple operations (attention, layer norm, residual) into a single CUDA kernel — fewer memory round-trips
2. **In-flight batching:** Equivalent to continuous batching, built in
3. **Paged KV cache:** Similar to vLLM's PagedAttention
4. **FP8 quantization:** Native on H100/H200, 2× throughput vs BF16
5. **Speculative decoding:** Built-in support for draft-target decoding
6. **Multi-GPU tensor parallelism:** Optimized all-reduce operations

```python
# TensorRT-LLM usage (simplified)
import tensorrt_llm
from tensorrt_llm import LLM as TRTLLM

# Load and convert model (or use pre-compiled engine)
llm = TRTLLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
)

# Batch inference
from tensorrt_llm.hlapi import SamplingParams

output = llm.generate(
    ["Explain transformer architecture in one paragraph."],
    sampling_params=SamplingParams(max_new_tokens=200),
)
print(output[0].outputs[0].text)
```

### vLLM vs TensorRT-LLM

| Aspect | vLLM | TensorRT-LLM |
|---|---|---|
| **Flexibility** | Very high — most models supported | Lower — NVIDIA-specific |
| **GPU support** | NVIDIA + AMD (ROCm) | NVIDIA only |
| **Peak throughput** | High | Highest (10-20% over vLLM at large scale) |
| **Setup complexity** | Easy (pip install) | Complex (compilation required) |
| **Quantization** | AWQ, GPTQ, GGUF | INT8, FP8, INT4 |
| **Best for** | Most production deployments | Maximum throughput on NVIDIA H100/H200 |

---

## 7. Triton Inference Server

**NVIDIA Triton** is an open-source serving framework that handles the **serving infrastructure** (HTTP/gRPC API, health checks, metrics, batching control) while backends (TensorRT-LLM, vLLM, PyTorch, ONNX) handle the model execution.

```
Client (HTTP/gRPC)
       ↓
Triton Inference Server
├── Model Management (load/unload/version)
├── Dynamic Batching (aggregate requests)
├── Ensemble Pipelines (chain models)
└── Backend Selection
    ├── TensorRT-LLM backend (LLMs)
    ├── PyTorch backend (custom models)
    ├── ONNX Runtime backend
    └── Python backend (custom pre/post-processing)
       ↓
GPU / CPU Execution
```

Triton is the choice when you need:
- Multi-model serving from one server
- Complex request pipelines (pre-processing → model → post-processing)
- Mixed hardware (some models on GPU, some on CPU)
- Kubernetes-native deployment with model versioning

---

## 8. Tensor Parallelism for Serving

When a single model doesn't fit on one GPU, **tensor parallelism** shards the model across multiple GPUs. For LLM inference:

### Megatron-style Column/Row Parallelism

**Column-parallel linear (attention projections W_Q, W_K, W_V):**
- Split weight matrix columnwise across $p$ GPUs
- Each GPU computes a partition of the output
- Gather with All-Reduce at the end

**Row-parallel linear (output projection W_O):**
- Split weight matrix rowwise across $p$ GPUs
- Each GPU gets a partition of the input
- Sum with All-Reduce at the end

```
70B model (BF16, 140 GB) across 2 × A100 (80 GB each):
  Each GPU holds ~70 GB (140 GB / 2 = 70 GB per GPU)
  Communication overhead: All-Reduce at each attention+FFN layer
  Typical efficiency: ~85-90% of 2× speedup

Tensor parallel = TP (model split across GPUs)
Pipeline parallel = PP (layers split across GPUs)
Data parallel = DP (batch split across GPUs, same model on each)
```

vLLM handles tensor parallelism transparently:

```python
from vllm import LLM

# Automatically shards 70B model across 4 GPUs
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,    # TP=4
    gpu_memory_utilization=0.9,
)
```

---

## 9. Throughput vs. Latency Tradeoffs

```
              ┌──────────────────────────────────────────┐
High          │                              ●  ●  ●      │
Throughput    │                    ●  ●  ●               │
(tokens/s)    │          ●  ●  ●                         │
              │  ●  ●  ●                                  │
              └──────────────────────────────────────────┘
                Low Latency (ms/token)         High Latency
```

**Key tradeoffs:**

| Knob | Effect on Throughput | Effect on Latency |
|---|---|---|
| **Larger batch size** | ↑↑ | ↑ (more queuing) |
| **Quantization (INT4)** | ↑↑ (2-4×) | ↓ (faster compute) |
| **Tensor parallelism** | ~ (communication overhead) | ↓ (parallel compute) |
| **Longer context** | ↓↓ (larger KV cache, less batch) | ↑ (more prefill) |
| **Speculative decoding** | ~ (overhead for easy tokens) | ↓↓ for hard reasoning |

**Production SLAs:**
- **Interactive (chatbot):** Latency < 200ms TTFT, < 50ms/token
- **Batch processing:** Maximize throughput, latency can be seconds
- **Streaming:** TTFT matters most; tokens should arrive at human reading speed (~5-10 tok/s)

### Speculative Decoding Integration

Run a small "draft" model to propose $k$ tokens, verify with the large "target" model in parallel:

```python
# vLLM speculative decoding
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    num_speculative_tokens=5,    # propose 5 tokens, verify in 1 target pass
    tensor_parallel_size=4,
)
```

Typical speedup: 2-3× for reasoning/coding tasks (high acceptance rate), near-neutral for creative generation (low acceptance).

---

## 10. Production Deployment Patterns

### Single-Node Deployment (≤ 8 GPUs)

```
┌─────────────────────────────────────────────┐
│  Load Balancer (nginx/Kubernetes Service)    │
└───────────────┬─────────────────────────────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌───────┐  ┌───────┐  ┌───────┐
│ vLLM  │  │ vLLM  │  │ vLLM  │
│ GPU 0 │  │ GPU 2 │  │ GPU 4 │
│ GPU 1 │  │ GPU 3 │  │ GPU 5 │
│ TP=2  │  │ TP=2  │  │ TP=2  │
└───────┘  └───────┘  └───────┘
 3 replicas of 7B model on 6 GPUs
```

### Multi-Node Deployment (Larger Models)

```
Node 0 (8x H100):                Node 1 (8x H100):
PP stage 0 (layers 0-39)         PP stage 1 (layers 40-79)
TP=8 within node                 TP=8 within node
        │ inter-node (InfiniBand) │
        └──────────────────────── ┘
       Pipeline parallel across nodes
```

### Kubernetes with Ray Serve

```python
# ray serve deployment (production-grade)
from ray import serve
from vllm.entrypoints.openai.api_server import router as vllm_router

@serve.deployment(
    ray_actor_options={"num_gpus": 2},
    num_replicas=3,
    max_concurrent_queries=100,
)
class VLLMDeployment:
    def __init__(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            tensor_parallel_size=2,
        )
    
    async def __call__(self, request):
        params = await request.json()
        outputs = self.llm.generate(
            params["prompt"],
            SamplingParams(**params.get("sampling_params", {}))
        )
        return {"text": outputs[0].outputs[0].text}
```

---

## 11. Interview Questions

**Q: What is the KV cache and why does it dominate LLM serving memory?**  
*A:* The KV cache stores the key and value tensors for all past tokens to avoid recomputing them on each generation step. Size = 2 × layers × heads × head_dim × ctx_len × bytes. For LLaMA 2 70B at 4K context (BF16): ~10 GB per sequence. This means with 80 GB GPU, you can only serve ~8 concurrent long-context sequences — the main serving bottleneck. Solutions: GQA (8-32× reduction), quantized KV cache (FP8 or INT8), PagedAttention (eliminate fragmentation).

**Q: Explain PagedAttention and how it improves serving throughput.**  
*A:* Before vLLM, each request pre-allocated a contiguous memory block of max_len tokens. Most requests finish before max_len, wasting 20-40% of GPU memory. PagedAttention (borrowed from OS virtual memory) divides KV cache into small fixed-size blocks (e.g., 16 tokens). A block table maps logical positions to physical blocks. Blocks are allocated on demand and can be non-contiguous. Benefits: near-zero memory waste, allows sharing system prompt blocks across users (copy-on-write), enables fitting 3-4× more sequences into the same GPU memory → 2-4× throughput improvement.

**Q: What is continuous batching and why does it matter?**  
*A:* Static batching waits for all requests in a batch to finish before starting new ones, causing GPU idle time when short requests finish early. Continuous batching inserts new requests into the batch immediately after each generation step (token by token), so the GPU is always maximally utilized. This reduces mean latency and increases throughput — particularly important for request streams with high variance in output length.

**Q: How does tensor parallelism differ from pipeline parallelism for serving?**  
*A:* Tensor parallelism (TP) splits each layer's weight matrices across GPUs — each GPU handles a shard of every attention head and FFN row/column. Communication: All-Reduce after each sublayer (frequent, small messages). Best for minimizing latency on a single request. Pipeline parallelism (PP) assigns different layers to different GPUs/nodes — data flows through stages. Communication: activation tensors between stages (less frequent, larger messages). Best for very large models that don't fit even with TP. In practice: TP within a node (fast NVLink), PP across nodes (slower InfiniBand).

**Q: What are the tradeoffs between AWQ and GPTQ quantization for inference?**  
*A:* Both are weight-only INT4 post-training quantization. GPTQ uses layer-by-layer reconstruction — minimize quantization error by adjusting each layer's weights to compensate for errors from quantizing previous layers. AWQ analyzes activation patterns to identify "salient" weight channels (those multiplied by large activations) and applies per-channel scaling to protect them. In practice: AWQ has ~0.5-1 perplexity point better than GPTQ for the same bit-width, because protecting salient weights matters more than layer-by-layer reconstruction. Both give ~4× memory reduction and 2-3× throughput improvement over BF16 at near-lossless quality (0.5-1% accuracy drop on most benchmarks).

---

## Where to Next

- **Speculative decoding algorithm** → [05-llms/applications/07-speculative-decoding.md](../../05-llms/applications/07-speculative-decoding.md)
- **KV cache and GQA math** → [05-llms/08-kv-cache-and-mqa-gqa.md](../../05-llms/08-kv-cache-and-mqa-gqa.md)
- **Model quantization (GPTQ, AWQ, GGUF)** → [quantization-pruning-detailed.md](./17-quantization-pruning-detailed.md)
- **System design for LLM inference ops** → [06-production-ml/system-design/12-llm-inference-ops.md](../../06-production-ml/system-design/12-llm-inference-ops.md)
