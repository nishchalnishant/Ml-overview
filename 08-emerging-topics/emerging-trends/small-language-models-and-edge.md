---
module: Emerging Topics
topic: Emerging Trends
subtopic: Small Language Models And Edge
status: unread
tags: [emergingtopics, ml, emerging-trends-small-language]
---
# Small Language Models and Edge AI

How SLMs achieve near-frontier performance at 1-7B parameters, what quantization formats actually do at the bit level, and the engineering constraints that govern on-device AI deployment.

---

## 1. Core Concept & Intuition

The assumption that "bigger is always better" breaks down at inference time. A 70B parameter model generating 30 tokens/second on a $30,000 server cluster is strictly worse than a 7B model generating 60 tokens/second on a $500 device — if the 7B model achieves 90% of the task accuracy. The question shifts from "what is the best model?" to "what is the best model for this hardware budget, latency requirement, and accuracy threshold?"

**What changed to make SLMs viable:**

1. **Knowledge distillation from frontier teachers:** Phi-3, Gemma, and Qwen were trained on data curated or synthesized by GPT-4-class models. A 3.8B model trained on 3.3T tokens of GPT-4-filtered data can outperform a 70B model trained on raw Common Crawl. Quality of data > quantity of parameters.

2. **Architectural efficiency:** Modern SLMs use GQA (Grouped Query Attention), SwiGLU activations, and rotary embeddings — the same techniques as frontier models. The parameter count is smaller but the architecture is not inferior.

3. **Quantization:** A 7B fp16 model at 14GB of GPU memory becomes a 4.4GB Q4 model that runs entirely in the RAM of a high-end phone. The quality loss is 1-3% on standard benchmarks.

---

## 2. Architecture & Mathematics

### 2.1 What Makes an SLM Efficient

**Grouped Query Attention (GQA):**

Standard MHA: each query head has its own K, V matrices.

```
MHA: Q=[H×d_k], K=[H×d_k], V=[H×d_v]  → H independent attention computations
Parameters per layer: 4 × H × d_model² / H = 4 × d_model²

GQA: Q=[H×d_k], K=[G×d_k], V=[G×d_v]  → G groups, each shared by H/G query heads
Parameters per layer: d_model² + d_model×d_k×G×2/H_q + d_model²  (much smaller KV)
```

For H=32 query heads and G=8 KV groups (Llama 3.1-8B config):
- KV cache per token = 2 × n_layers × G × d_head × 2 bytes
- Llama 3.1-8B: 2 × 32 × 8 × 128 × 2 = 131 KB/token (vs 327 KB for 70B with GQA-8)
- At G=1 (MQA — Multi-Query Attention): 2 × 32 × 1 × 128 × 2 = 16.4 KB/token

KV cache reduction enables longer context at the same memory budget and higher throughput batching.

**SwiGLU FFN (used in Llama, Phi, Gemma):**

```
Standard FFN: h = W_2 · max(0, W_1 · x)                    # 2 weight matrices
SwiGLU FFN:  h = W_2 · (SiLU(W_1 · x) ⊙ (W_3 · x))       # 3 weight matrices

SiLU(x) = x · σ(x)  (smooth approximation of ReLU)
```

SwiGLU uses 3 weight matrices instead of 2, but each matrix is smaller (2/3 × d_ffn). Total parameters roughly equivalent to standard FFN. Quality improvement: the gating mechanism (⊙ with W_3·x) allows the FFN to selectively pass information, improving representational quality per parameter. SwiGLU is the standard FFN for all modern SLMs.

### 2.2 Knowledge Distillation

**Standard distillation:**

```
L_distill = α · L_CE(student_logits, hard_labels) 
           + (1-α) · KL(softmax(teacher_logits/T), softmax(student_logits/T))

where T = temperature (2-5 softens the teacher distribution, revealing inter-class structure)
```

The soft labels carry more information than hard labels: a teacher giving "cat: 70%, dog: 20%, tiger: 10%" reveals that "this cat looks slightly dog-like" — the student learns the teacher's uncertainty structure, not just the correct class.

**Data distillation (the Phi approach):**

Rather than distilling the model's logits, distill the teacher's *data generation capability*. Use GPT-4 to generate high-quality synthetic training data on topics selected for educational value. Train the student on this synthetic data from scratch.

```
Dataset: D_synthetic = {GPT-4("write a Python tutorial on recursion"), 
                         GPT-4("explain gradient descent to a 10-year-old"),
                         GPT-4("give a step-by-step proof of the Pythagorean theorem"),
                         ...}

Student objective: standard next-token prediction on D_synthetic
L = -Σ log P_student(x_t | x_{<t})
```

This is not traditional distillation (no teacher logits during student training), but it leverages frontier model quality at data-generation time. Phi-3-mini (3.8B) trained this way on 3.3T "textbook-quality" tokens matches Mistral-7B across most benchmarks.

### 2.3 Quantization: The Bit-Level Math

**Uniform quantization (INT8, INT4):**

Map a floating-point range [x_min, x_max] to integers [0, 2^b - 1]:

```
scale = (x_max - x_min) / (2^b - 1)
zero_point = round(-x_min / scale)

Quantize:  x_q = round(x / scale) + zero_point         (b-bit integer)
Dequantize: x̂  = scale · (x_q - zero_point)            (float approximation)

Error: ||x - x̂||² is bounded by (scale/2)² — smaller scale → smaller error
       scale is determined by the range; wider ranges → larger error
```

**Absmax quantization (common for activations):**

```
scale = max(|x|) / 127   (for INT8)
x_q = round(x / scale)   (symmetric: zero_point = 0)
```

**Block-wise quantization:** Rather than quantizing the entire weight matrix with one scale, divide into blocks of 64-128 elements, each with its own scale. Reduces quantization error because outlier weights in one block don't inflate the scale for all other blocks.

```
# Naive: one scale for entire row (10,000 elements)
# Block: one scale per 64 elements → 156 scales per row
# Error reduction: proportional to ratio of outlier influence
```

### 2.4 GGUF / llama.cpp Quantization Types

GGUF (GPT-Generated Unified Format) is the container format used by llama.cpp — the primary CPU/Metal/CUDA inference engine for local models.

**Q4_K_M (the standard "good" quantization):**

```
Q4_K_M breakdown:
  - Most weights: 4-bit quantized in blocks of 256
  - Super-blocks: 8-bit scales stored for blocks of 8 sub-blocks (256×8=2048 weights)
  - Attention and FFN output projections: 6-bit (higher precision for sensitive layers)
  - Embedding and output layer: 6-8 bit
  
Memory: ~4.5 bits per weight average (vs 16 bits fp16)
7B model: 7B × 4.5/8 bytes = ~3.9 GB (fits on high-end mobile, or CPU RAM)
```

**Naming convention:**

```
Q[bits]_[variant]_[size_qualifier]
Q4_K_M: 4-bit, K-quants method, Medium (balance of speed and quality)
Q4_K_S: 4-bit, K-quants, Small (smaller but lower quality)
Q5_K_M: 5-bit, Medium (better quality, larger)
Q8_0:   8-bit, simple (nearly lossless, 2× larger than Q4)
IQ4_XS: Importance-Weighted 4-bit, Extra Small (newer, better quality/size)
```

**K-quants key innovation:** Use the importance of each weight (estimated by activation statistics) to decide quantization precision. Weights that multiply large activations get higher precision bits; weights that multiply near-zero activations get lower precision.

### 2.5 AWQ (Activation-Aware Weight Quantization)

AWQ identifies which weights are most important by looking at the input activations they multiply:

```
For weight W (row vector):
  importance(W_j) = E[|x_j|]  (expected magnitude of the input that multiplies W_j)
  
Salient weights: top 1% of importance scores
  → These are NOT quantized (kept in fp16)
  → The remaining 99% are quantized to INT4

But 1% of weights in fp16 with 99% in INT4 creates a mixed-precision format that's
hard to implement efficiently. AWQ instead scales:

  Equivalent transformation:
  W·x = (W · diag(s)) · (diag(s)^{-1} · x)
      = W̃ · x̃
  
  Choose s such that salient channels in x̃ are smaller (easier to quantize W̃)
  s_j = E[|x_j|]^α, α ≈ 0.5
  
  Now quantize W̃ uniformly — the scaling pre-absorbed the importance weighting
```

AWQ achieves better quality than naive INT4 without the mixed-precision complexity. It's the standard quantization method for GPU-accelerated inference (vLLM, TGI).

### 2.6 EXL2 (ExLlamaV2) Format

EXL2 extends AWQ with per-row mixed-bit quantization — different rows of the weight matrix get different bit widths based on their measured quantization error:

```
For each row r of weight matrix W:
  Measure quantization error: e_r = ||W_r - dequantize(quantize(W_r, b))||²
  If e_r > threshold: assign higher bit depth (5 or 6 bits)
  Else: assign lower bit depth (2 or 3 bits)

Target average bits: user specifies (e.g., 4.0 bpw)
Optimizer: finds the per-row bit assignment that minimizes total error subject to
           the average bit constraint (knapsack-style optimization)
```

EXL2 provides better perplexity at the same average bits-per-weight compared to uniform quantization. Primarily used for local inference with ExLlamaV2 on consumer GPUs (24GB VRAM models at very low bpw).

### 2.7 Apple Intelligence Architecture

Apple Intelligence (2024) deploys ML across three tiers:

```
Tier 1: On-device (iPhone A17/M-series)
  Model: 3B parameter foundation model, Apple's custom architecture
  Quantization: 2-bit weights with mixed 4-bit for critical layers
  Memory: 3B × 2/8 = 750 MB weight memory (fits within 8GB device RAM)
  Latency: ~30 tokens/sec for language tasks
  Use cases: Writing tools, summarization, Reply suggestions, local Siri tasks

Tier 2: Private Cloud Compute (Apple Silicon servers)
  Model: Larger models (est. 70B class) running on Apple Silicon
  Privacy: Requests processed without Apple retaining or accessing the data
           Cryptographically verifiable privacy: third-party auditing of server software
  Use cases: More complex reasoning, knowledge queries

Tier 3: OpenAI (GPT-4o via opt-in)
  User consent required per-query
  Use cases: frontier-level tasks beyond on-device/PCC capability
```

**On-device model specializations:**
Apple's on-device model is not a general-purpose LLM — it has task-specific adapter heads trained for: summarization, Smart Reply, Proofreading, Rewriting, Priority classification (email), OCR understanding.

**Neural Engine hardware:** Apple's Neural Engine (ANE) is a dedicated matrix multiplication accelerator. A17 Pro ANE: 35 TOPS. The model is compiled to Core ML format (`.mlpackage`) using ANE-optimal computation graphs — fused attention, INT4 weight kernels, and memory bandwidth-optimized layouts. The compilation step happens on-device during model download.

### 2.8 ONNX and Cross-Platform Edge Deployment

```
Training (PyTorch) → Export to ONNX → Runtime optimization

ONNX export:
  torch.onnx.export(model, dummy_input, "model.onnx",
                    opset_version=17,
                    dynamic_axes={"input": {0: "batch", 1: "seq_len"}})

ONNX Runtime backends:
  CPU:    MLAS (Microsoft Linear Algebra Subprograms), AVX-512 optimized
  GPU:    CUDA/TensorRT execution provider
  Mobile: NNAPI (Android), CoreML (iOS)
  NPU:    Qualcomm QNN, Samsung Eden
```

INT8 quantization in ONNX Runtime uses calibration datasets to determine per-tensor scales:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    weight_type=QuantType.QInt8  # weights quantized; activations dynamic
)
```

---

## 3. Trade-offs & System Design Implications

### SLM vs Large Model Decision Matrix

| Constraint | Choose SLM | Choose Large Model |
|---|---|---|
| Latency < 500ms | ✓ | Only with speculative decoding |
| Privacy / no internet | ✓ | Requires expensive self-hosting |
| Cost < $0.001/query | ✓ | Typically 10-100× more expensive |
| Multi-step complex reasoning | Only with chain-of-thought | ✓ |
| Long-form generation > 2K tokens | Quality degrades | ✓ |
| Domain-specific + fine-tunable | ✓ (fast fine-tuning) | Expensive to fine-tune |
| Batch inference, not real-time | Either | ✓ (better quality) |

### Quantization Quality vs. Size Trade-off

```
Model: Llama 3.1-8B  (baseline: fp16 perplexity on WikiText-2 = 8.24)

Format    | Size (GB) | Perplexity | Quality Loss
----------|-----------|------------|-------------
fp16      | 16.0      | 8.24       | 0%
Q8_0      | 8.5       | 8.25       | 0.1%
Q6_K      | 6.6       | 8.28       | 0.5%
Q5_K_M    | 5.3       | 8.32       | 1.0%
Q4_K_M    | 4.9       | 8.41       | 2.1%
Q4_K_S    | 4.7       | 8.49       | 3.0%
Q3_K_M    | 3.5       | 8.84       | 7.3%
Q2_K      | 2.9       | 10.4       | 26%  ← significant degradation

Recommendation: Q4_K_M is the practical sweet spot (4.9 GB, 2% quality loss)
```

### Inference Speed on Different Hardware

```
7B Q4_K_M model (Llama 3.1-8B):
Hardware                     | Tokens/sec (generation)
-----------------------------|------------------------
MacBook Pro M3 Pro (36GB)    | 60-80 t/s
RTX 4090 (24GB)              | 150-200 t/s
iPhone 15 Pro (A17 Pro)      | 10-20 t/s
Raspberry Pi 5 (8GB)         | 1-3 t/s
Intel Core i9 (64GB RAM)     | 8-15 t/s

Memory bandwidth is the bottleneck for autoregressive generation:
  Generation is memory-bandwidth-bound (weights loaded per token, not compute-bound)
  RTX 4090: 1008 GB/s bandwidth → 1008 / (4.9 GB × 1 token) ≈ 206 tokens/s
  M3 Pro unified memory: ~300 GB/s → 300 / 4.9 ≈ 61 tokens/s
```

The formula: `max_tokens_per_second ≈ memory_bandwidth_GB_s / model_size_GB`

### Fine-tuning SLMs for Domain-Specific Tasks

**LoRA (Low-Rank Adaptation):** inject trainable rank-r matrices into attention projections:

```
W_adapted = W_pretrained + ΔW = W + A·B
where A ∈ [d_model × r], B ∈ [r × d_model], r << d_model

Trainable parameters: 2 × d_model × r (vs d_model² for full fine-tuning)
For d_model=4096, r=16: 2 × 4096 × 16 = 131K params per layer (vs 16.7M for full)
```

QLoRA: quantize the base model to INT4, apply LoRA adapters in fp16/bf16. Fine-tune a 7B model on a single 24GB GPU. Effectively enables SLM fine-tuning on consumer hardware.

---

## 4. Canonical Interview Q&As

**Q1: Explain why a 3.8B model trained on "textbook-quality" synthetic data (Phi-3) can outperform a 70B model trained on raw web data. What are the implications for scaling laws?**

The classical scaling laws (Chinchilla) establish optimal compute allocation between model size and training tokens for models trained on web data. They assume a fixed-quality dataset. The key insight Phi-3 exploits: scaling laws hold for a given data distribution, but if you can dramatically shift the data distribution toward higher information density, you change the effective trade-off curve.

Web data is dominated by low-information-density content: marketing copy, repetitive forum posts, SEO spam. A model trained on this data spends most of its capacity modeling this low-value content. GPT-4-generated "textbook" data is maximally information-dense: every sentence teaches a concept, every example is pedagogically designed to clarify a principle.

Empirically: training a 3.8B model on 3.3T tokens of GPT-4-curated data is more parameter-efficient than training a 70B model on 1T tokens of web data for coding and reasoning benchmarks. The smaller model "learns more per parameter" from higher quality data.

Implication: the scaling laws need a data quality term. The original Chinchilla results are a lower bound on what's achievable at a given compute budget if data quality is optimized. This suggests the SLM paradigm is sustainable — continued investment in synthetic data generation and quality filtering can extend the performance of smaller models further, narrowing the gap with frontier models without scaling parameters.

**Q2: Walk through the mathematics of INT4 quantization. Why does block-wise quantization reduce error, and why does 2-bit quantization degrade so severely?**

INT4 quantization maps 16-bit floats to 4-bit integers (16 possible values). The mapping is:

```
x_q = clamp(round((x - x_min) / scale), 0, 15)
scale = (x_max - x_min) / 15
```

The quantization error for a single value x is bounded by scale/2. The scale is determined by the range [x_min, x_max]. If one weight is an outlier (e.g., x=50 while most weights are in [-1, 1]), the scale becomes 50/15 ≈ 3.3, meaning every "normal" weight in [-1, 1] is quantized with an error up to ±1.65 — larger than the weight itself for values near zero.

Block-wise quantization fixes this by computing a separate scale per 64-element block:
```
Block 1 (64 weights, range [-0.8, 0.9]): scale = 1.7/15 = 0.113
Block 2 (64 weights, range [-0.5, 50.0]): scale = 50.5/15 = 3.37  ← only this block degrades
```

The outlier only inflates the scale for its own block of 64 weights, not the entire weight matrix. For a typical transformer weight matrix (4096×4096), block-wise quantization isolates outlier damage to 0.38% of the weights.

INT2 degradation: 2 bits = only 4 levels. Weights in [-1, 1] get quantized to {-1, -0.33, 0.33, 1}. Maximum error = 0.33. For a typical transformer with millions of weight-input multiplications per forward pass, this error accumulates multiplicatively through layers. Each layer's error becomes the next layer's input error — and with 32 layers, the accumulated distortion renders the model's output distribution nearly random. This is why Q2_K shows 26% perplexity increase while Q4_K_M shows only 2%.

**Q3: What hardware constraints govern on-device LLM deployment on smartphones, and how do quantization formats like GGUF address each constraint?**

Three constraints dominate smartphone deployment:

(1) **DRAM bandwidth (not compute):** Autoregressive generation requires loading all model weights per generated token. An A17 Pro has ~68 GB/s DRAM bandwidth. A 7B fp16 model (14 GB) can generate at most 68/14 ≈ 4.9 tokens/second — unusable. At Q4_K_M (4.9 GB): 68/4.9 ≈ 13.9 tokens/second — marginal. At Q2_K (2.9 GB): 68/2.9 ≈ 23 tokens/second — faster but quality degraded. The practical solution is to stay at Q4 and accept 10-15 t/s.

(2) **DRAM capacity:** iPhone 15 Pro has 8 GB total RAM shared between OS, other apps, and the model. The model must fit within ~3-4 GB to coexist with the OS (typically 2-3 GB) and leave headroom. GGUF Q4_K_M of a 3B model ≈ 1.9 GB — fits. Of a 7B model ≈ 4.9 GB — too large for most phones. This is why Apple's on-device model is 3B.

(3) **Power envelope:** A smartphone thermal design power (TDP) is ~5-8W. Running intensive matrix operations for extended periods triggers thermal throttling, reducing performance 50-70% after 30-60 seconds. Apple's ANE is specifically designed to be energy-efficient for matrix multiply workloads (~2W for typical model inference vs ~6W on the GPU).

GGUF addresses these by: (a) providing carefully tuned quantization that minimizes quality loss within a given memory budget; (b) supporting layer-wise mixed quantization (embedding layers kept at higher bits to preserve token representation quality); (c) including memory mapping (mmap) of weight files so the OS manages which weight pages are in RAM — models can be larger than free RAM if generation is slow enough that the OS can page in needed weights.

**Q4: Compare LoRA, QLoRA, and full fine-tuning for adapting a 7B SLM to a domain-specific task. When would you choose each, and what are the theoretical trade-offs?**

**Full fine-tuning:** update all 7B parameters. Requires 7B × 4 bytes = 28 GB for weights + 28 GB for gradients + optimizer state (Adam: 2× gradients) = ~112 GB minimum — requires 2×80GB A100s. Risk: catastrophic forgetting of general capabilities. Benefits: highest expressive capacity; can change any behavior.

**LoRA (r=16):** inject A·B matrices at attention Q, K, V, O projections and FFN layers. For a 7B model with d_model=4096, 8 projection matrices: 8 × 2 × 4096 × 16 = 1.05M trainable parameters (vs 7B). Memory: the base model stays in fp16 (14 GB) + 1.05M × 4 bytes for LoRA weights + proportional gradient/optimizer state ≈ 24 GB total — fits on a single 24GB RTX 4090. During inference, merge LoRA weights: W_merged = W + A·B (zero extra inference cost). Trade-off: LoRA can only change directions within the span of A·B — it cannot represent arbitrary weight changes. For tasks requiring genuinely new knowledge (new language, domain with different token distribution), full fine-tuning outperforms.

**QLoRA:** quantize base model to NF4 (4-bit Normalized Float), then apply LoRA in bf16. NF4 is designed for normally distributed weights:

```
NF4 levels: optimally placed quantization points for N(0,1) distribution
            (more levels near zero, fewer at extremes — matches weight distribution)
QLoRA memory: 7B × 0.5 bytes (4-bit) + 1.05M × 2 bytes (bf16 LoRA) ≈ 3.7 GB
              + gradient/optimizer state for LoRA only ≈ 4-5 GB total
              → fits on a single 8GB GPU (RTX 3070, etc.)
```

QLoRA quality is nearly identical to LoRA (NF4 is near-lossless for normally distributed weights) at dramatically lower memory. Choose QLoRA for fine-tuning on consumer hardware. One subtlety: QLoRA backward pass must dequantize base model weights per layer for gradient computation — this is slower than LoRA (2-3× training time) but enables fine-tuning with 4-8GB GPU.

**Choice guide:** full fine-tuning → highest quality, new domain from scratch, have A100s; LoRA → standard domain adaptation, have 24GB GPU; QLoRA → same use case as LoRA but budget hardware (8-16GB GPU); adapter layers (Prefix-tuning, Prompt-tuning) → want to preserve base model exactly, adapting to simple task distribution shifts.

**Q5: What is the memory bandwidth utilization efficiency of LLM inference on different hardware backends, and why does this matter more than FLOP counts for autoregressive generation?**

Autoregressive generation is memory-bandwidth-bound, not compute-bound. At each decode step, the model must: (1) load every weight matrix from memory to compute the forward pass for the single new token; (2) the KV cache grows by one row per layer per token. The arithmetic intensity (FLOPs per byte loaded) is approximately:

```
Arithmetic intensity (decode) ≈ 2 × batch_size / bytes_per_parameter

At batch_size=1, INT4: 2 × 1 / 0.5 = 4 FLOP/byte
At batch_size=32, INT4: 2 × 32 / 0.5 = 128 FLOP/byte

Hardware peak FLOP/byte (roofline):
  RTX 4090: 82.6 TFLOPS / 1008 GB/s = 82 FLOP/byte
  H100 SXM: 1979 TFLOPS / 3350 GB/s = 591 FLOP/byte
  M3 Pro Neural Engine: 18 TOPS / 100 GB/s = 180 FLOP/byte (est.)
```

At batch_size=1, arithmetic intensity=4 is far below every hardware's roofline — the GPU is spending most of its time waiting for weights to arrive from HBM, not computing. This means:

- **FLOPs per second is nearly irrelevant** for single-user latency
- **Memory bandwidth determines tokens/sec**: `tokens/sec ≈ bandwidth / (model_size_bytes)`
- To reach compute-bound operation: batch_size must exceed `hardware_FLOP_byte × bytes_per_param / 2`
  - For RTX 4090 INT4: batch ≥ 82 × 0.5 / 2 = 20.5 → batch_size > 20 is compute-bound

Practical implication: for batch_size=1 deployment (interactive apps), buy hardware with high memory bandwidth (H100 NVL has 7.2 TB/s; A100 SXM has 2 TB/s). For batch inference (offline), buy hardware with high FLOP/s and use large batches. The RTX 4090's high memory bandwidth (relative to cost) makes it excellent for small-batch consumer inference — a better buy for a startup than a single A100 at 5× the price.

## Flashcards

**KV cache per token = 2 × n_layers × G × d_head × 2 bytes?** #flashcard
KV cache per token = 2 × n_layers × G × d_head × 2 bytes

**Llama 3.1-8B?** #flashcard
2 × 32 × 8 × 128 × 2 = 131 KB/token (vs 327 KB for 70B with GQA-8)

**At G=1 (MQA?** #flashcard
Multi-Query Attention): 2 × 32 × 1 × 128 × 2 = 16.4 KB/token

**Most weights?** #flashcard
4-bit quantized in blocks of 256

**Super-blocks?** #flashcard
8-bit scales stored for blocks of 8 sub-blocks (256×8=2048 weights)

**Attention and FFN output projections?** #flashcard
6-bit (higher precision for sensitive layers)

**Embedding and output layer?** #flashcard
6-8 bit

**FLOPs per second is nearly irrelevant for single-user latency?** #flashcard
FLOPs per second is nearly irrelevant for single-user latency

**Memory bandwidth determines tokens/sec?** #flashcard
tokens/sec ≈ bandwidth / (model_size_bytes)

**To reach compute-bound operation?** #flashcard
batch_size must exceed hardware_FLOP_byte × bytes_per_param / 2

**For RTX 4090 INT4?** #flashcard
batch ≥ 82 × 0.5 / 2 = 20.5 → batch_size > 20 is compute-bound
