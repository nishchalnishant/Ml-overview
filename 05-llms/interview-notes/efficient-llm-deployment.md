# Efficient LLM Deployment & Optimization

Production LLM serving is an engineering discipline in its own right. The gap between a model that runs in a research notebook and one that handles 10k concurrent requests at <200ms TTFT is filled by quantization, KV cache management, batching strategies, parallelism schemes, and hardware-aware algorithm design. This file covers the full stack with the quantitative specifics engineers need to know.

---

## 1. Memory Arithmetic

Before optimizing, measure what you're working with.

### Model Weight Memory

$$\text{Memory (bytes)} = N_{\text{params}} \times \text{bytes per param}$$

| Format | Bytes/param | 7B model | 70B model | Notes |
| :--- | :--- | :--- | :--- | :--- |
| FP32 | 4 | 28 GB | 280 GB | Training master weights |
| FP16 / BF16 | 2 | 14 GB | 140 GB | Standard serving |
| INT8 | 1 | 7 GB | 70 GB | Post-training quantization |
| INT4 | 0.5 | 3.5 GB | 35 GB | Aggressive quantization |
| INT2 | 0.25 | 1.75 GB | 17.5 GB | Extreme (quality degradation) |

With overhead (activations, CUDA context, framework): multiply by ~1.2–1.3.

**Rule:** $\text{GPU VRAM (GB)} \approx N_{\text{params (B)}} \times \text{bytes/param} \times 1.25$

### KV Cache Memory

$$\text{KV Cache (bytes)} = 2 \times L \times H \times d_h \times T \times \text{bytes/element} \times B$$

For LLaMA 3 70B (80 layers, 8 KV heads after GQA, 128 head dim) in BF16, batch=1, 4096 tokens:

$$= 2 \times 80 \times 8 \times 128 \times 4096 \times 2 \approx 1.34 \text{ GB}$$

With batch size 32: ~43 GB — KV cache often dominates VRAM at high batch sizes.

**GQA reduces KV cache by $n_{\text{heads}} / n_{\text{kv\_heads}}$:** LLaMA 3 70B uses 8 KV heads vs 64 query heads — 8× KV cache reduction versus MHA.

---

## 2. Quantization

Quantization maps floating-point weights to lower-precision integers.

### Post-Training Quantization (PTQ)

No retraining required. Calibrate on a small dataset.

**Symmetric INT8:**
$$q = \text{round}\left(\frac{w}{s}\right), \quad s = \frac{\max(|w|)}{127}$$

Dequantize at inference: $\hat{w} = q \times s$

**Asymmetric INT4:**
$$q = \text{round}\left(\frac{w - z}{s}\right), \quad s = \frac{\max(w) - \min(w)}{15}, \quad z = -\text{round}\left(\frac{\min(w)}{s}\right)$$

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit NF4 quantization (bitsandbytes)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",          # NF4 better than INT4 for normal-distributed weights
    bnb_4bit_use_double_quant=True,     # quantize the quantization constants too
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-70B",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### GPTQ (Group-wise PTQ)

Quantize layer-by-layer using second-order information (inverse Hessian):

1. Process columns of $W$ one at a time
2. Quantize a column: $q_j = \text{round}(w_j / s)$
3. Absorb the quantization error into remaining unquantized columns using the inverse Hessian: $\delta W = -\frac{e_j}{[H^{-1}]_{jj}} [H^{-1}]_{:,j}$

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,    # quantize in groups of 128 columns
    desc_act=False,    # activation ordering — disable for speed
)

model = AutoGPTQForCausalLM.from_pretrained(model_name_or_path, quantize_config)
model.quantize(calibration_dataset)
model.save_quantized("./gptq-model")
```

### AWQ (Activation-Aware Weight Quantization)

Key insight: weights that correspond to large-magnitude activations matter more. Scale these weights before quantization to protect them.

$$s_j = \frac{\max(|X_j|)^{\alpha}}{\max(|W_j|)}, \quad \tilde{W}_j = W_j / s_j, \quad \tilde{X}_j = X_j \cdot s_j$$

Then quantize $\tilde{W}_j$ — the important channels get more bits effectively.

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(model_path)
model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
model.save_quantized("./awq-model")
```

### Quantization Quality vs. Speed

| Method | VRAM reduction | Perplexity delta | Throughput | Notes |
| :--- | :--- | :--- | :--- | :--- |
| FP16 baseline | 1× | 0 | 1× | Reference |
| INT8 (bitsandbytes) | ~2× | +0.1–0.3 | 1.3× | Safe for most tasks |
| GPTQ 4-bit | ~4× | +0.3–0.8 | 2–3× | Good balance |
| AWQ 4-bit | ~4× | +0.2–0.6 | 2–3× | Better than GPTQ on average |
| GGUF Q4_K_M | ~4× | +0.4 | CPU viable | llama.cpp format |

---

## 3. KV Cache Mechanics and Optimization

### Standard KV Cache

During autoregressive generation, store $K$ and $V$ for all past tokens. Each new token only computes its own $Q$, $K$, $V$ and queries the cached $K$, $V$.

```python
class KVCache:
    def __init__(self, max_seq_len: int, n_layers: int, n_kv_heads: int, 
                 head_dim: int, dtype=torch.float16, device="cuda"):
        shape = (n_layers, 2, max_seq_len, n_kv_heads, head_dim)
        self.cache = torch.zeros(shape, dtype=dtype, device=device)
        self.current_pos = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        seq_len = k.shape[1]
        self.cache[layer_idx, 0, self.current_pos:self.current_pos + seq_len] = k
        self.cache[layer_idx, 1, self.current_pos:self.current_pos + seq_len] = v
        self.current_pos += seq_len

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.cache[layer_idx, 0, :self.current_pos],
                self.cache[layer_idx, 1, :self.current_pos])
```

### PagedAttention (vLLM)

**Problem:** naive KV cache pre-allocates memory for max sequence length. With varying request lengths (some 100 tokens, some 4000), most of this memory is wasted — **up to 60–80% fragmentation**.

**PagedAttention** treats KV cache like OS virtual memory:
- Divide KV cache into fixed-size **pages** (e.g., 16 tokens per page)
- Maintain a page table mapping logical → physical pages
- Allocate pages on demand; release when request completes
- Share physical pages across requests with the same prefix (prompt caching)

```
Logical KV blocks:   [Prompt tokens][Gen token 1][Gen token 2]...
                          ↓                ↓              ↓
Physical pages:      [Page 42]         [Page 7]       [Page 31]   (non-contiguous)
```

**Result:** 24× higher throughput than HuggingFace Transformers naive batching at same hardware.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,    # fraction of GPU VRAM for KV cache pages
    max_model_len=8192,
    block_size=16,                  # tokens per KV page
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(prompts, sampling_params)
```

### Prefix Caching (Prompt Caching)

When many requests share the same system prompt, compute its KV entries once and reuse across all requests. vLLM implements this with copy-on-write page sharing.

**TTFT reduction:** for a 1000-token system prompt, prefix caching eliminates the prefill computation for every subsequent request.

---

## 4. Speculative Decoding

**Bottleneck:** autoregressive generation is sequential — each token requires a full LLM forward pass. A 70B model at 1 token/step is slow.

**Speculative decoding:** use a small draft model to generate $\gamma$ candidate tokens, then verify all $\gamma$ with a single target model forward pass (parallel evaluation).

### Algorithm

1. Draft model generates tokens $t_1, t_2, \ldots, t_\gamma$ autoregressively
2. Target model evaluates all $\gamma$ positions in one parallel forward pass, producing $p_\text{target}(t | x)$
3. Accept token $t_i$ with probability $\min\left(1, \frac{p_\text{target}(t_i)}{p_\text{draft}(t_i)}\right)$
4. At first rejection, resample from adjusted distribution: $p_\text{target}(t) - p_\text{draft}(t)$ (normalized)
5. Repeat

**Guarantee:** output distribution is identical to sampling from target model alone (no quality loss).

**Speedup:** depends on draft acceptance rate. When draft and target agree on ~80% of tokens, speedup ≈ 2–3× with $\gamma = 4$.

```python
def speculative_decode(
    target_model,
    draft_model,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    gamma: int = 4,
) -> torch.Tensor:
    generated = prompt_tokens.clone()
    
    while generated.shape[1] - prompt_tokens.shape[1] < max_new_tokens:
        # Draft: generate gamma tokens
        draft_tokens = []
        draft_probs = []
        input_seq = generated.clone()
        
        for _ in range(gamma):
            with torch.no_grad():
                draft_logits = draft_model(input_seq).logits[:, -1, :]
            draft_prob = torch.softmax(draft_logits, dim=-1)
            next_token = torch.multinomial(draft_prob, 1)
            draft_tokens.append(next_token)
            draft_probs.append(draft_prob)
            input_seq = torch.cat([input_seq, next_token], dim=1)
        
        # Target: verify all draft tokens in one pass
        verify_input = torch.cat([generated] + draft_tokens, dim=1)
        with torch.no_grad():
            target_logits = target_model(verify_input).logits
        
        # Accept/reject each token
        n_accepted = 0
        for i, (dt, dp) in enumerate(zip(draft_tokens, draft_probs)):
            target_prob = torch.softmax(target_logits[:, generated.shape[1] + i - 1, :], dim=-1)
            acceptance_prob = torch.min(torch.ones(1), target_prob[:, dt] / dp[:, dt])
            
            if torch.rand(1) < acceptance_prob:
                generated = torch.cat([generated, dt], dim=1)
                n_accepted += 1
            else:
                # Reject: sample from corrected distribution
                corrected = torch.clamp(target_prob - dp, min=0)
                corrected /= corrected.sum()
                fallback = torch.multinomial(corrected, 1)
                generated = torch.cat([generated, fallback], dim=1)
                break
        
        if n_accepted == gamma:
            # All accepted: sample one more from target
            bonus = torch.multinomial(torch.softmax(target_logits[:, -1, :], dim=-1), 1)
            generated = torch.cat([generated, bonus], dim=1)
    
    return generated
```

### Self-Speculative Decoding (Medusa, Eagle)

Instead of a separate draft model, add multiple heads to the target model itself:

- **Medusa:** adds $k$ extra parallel LM heads at the final layer, each predicting $k$ steps ahead. No separate model needed.
- **Eagle:** trains a draft model that reuses the target's hidden states — higher acceptance rate.

---

## 5. Continuous Batching

**Static batching** waits for a fixed batch to fill before running inference. Long requests hold up short ones.

**Continuous batching** (iteration-level scheduling): process each token generation step across all active requests. When a request finishes, immediately insert a new one — no waiting.

```
Time step 1: [Req A: token 1] [Req B: token 1] [Req C: token 1]
Time step 2: [Req A: token 2] [Req B: token 2] [Req C: token 2]
Time step 3: [Req A: token 3] [Req B: DONE   ] [Req D: token 1] ← D starts immediately
Time step 4: [Req A: token 4] [Req D: token 2] [Req E: token 1] ← E inserted
```

**Throughput improvement:** 5–10× over static batching for mixed short/long request distributions.

---

## 6. Flash Attention

**Problem:** standard attention materializes the $N \times N$ attention matrix in GPU HBM (high-bandwidth memory). For sequence length 4096: $4096^2 \times 2$ bytes = 32 MB per head per layer. With 80 layers × 64 heads: ~160 GB just for attention matrices.

**Flash Attention** avoids materializing the full matrix by tiling:

1. Divide $Q, K, V$ into blocks that fit in SRAM (~20 MB on A100)
2. Compute attention for each tile without writing intermediate results to HBM
3. Track the softmax normalization factor across tiles using the log-sum-exp trick

$$\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j$$

Accumulate per block: $m_{\text{new}} = \max(m_{\text{old}}, m_{\text{block}})$, $\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \ell_{\text{old}} + \sum_j e^{x_j - m_{\text{new}}}$

**Result:**
- Memory: $O(N)$ instead of $O(N^2)$
- Speed: 2–4× faster than standard attention for long sequences
- Output: **mathematically identical** — no approximation

```python
import torch
import torch.nn.functional as F

# Flash Attention via PyTorch 2.0+ SDPA
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,    # causal masking built-in
    )
```

---

## 7. Model Parallelism

When the model doesn't fit on a single GPU.

### Tensor Parallelism (Megatron-style)

Split weight matrices across GPUs along a single dimension:

$$Y = XW \rightarrow Y = X[W_1 | W_2] = [XW_1 | XW_2]$$

Each GPU holds a shard of $W$ and computes part of $Y$. An AllReduce synchronizes after each layer.

```
GPU 0: W[:, :d/2]    GPU 1: W[:, d/2:]
              ↓ AllReduce ↓
          Full output Y
```

**Communication overhead:** one AllReduce per layer (2 per Transformer block). Requires high-bandwidth interconnect (NVLink, 600 GB/s vs PCIe 64 GB/s).

### Pipeline Parallelism

Assign consecutive layers to different GPUs:

```
GPU 0: layers 0–19     GPU 1: layers 20–39    GPU 2: layers 40–59    GPU 3: layers 60–79
```

**Pipeline bubbles:** GPU 0 is idle while GPUs 1–3 process its output. Micro-batching reduces this: split each batch into micro-batches that fill the pipeline.

### Context Parallelism (Ring Attention)

For very long sequences: split the sequence across GPUs. Each GPU computes attention over its local tokens, communicates KV with neighbors in a ring pattern.

**Use case:** context lengths > 128k tokens.

### Recommended Parallelism Strategy

| Model size | GPUs | Strategy |
| :--- | :--- | :--- |
| ≤13B | 1–2 × 80GB | Single GPU or pipeline |
| 70B | 4–8 × 80GB | Tensor parallel TP=4–8 |
| 405B | 16–64 × 80GB | TP=8 + PP=4, or TP=8 + CP for long ctx |
| >405B | 64+ | 3D parallelism (TP + PP + DP) |

---

## 8. VRAM Calculation for Serving

Full equation for a single serving instance:

$$\text{VRAM} = \underbrace{N_{\text{params}} \times b_w}_{\text{weights}} + \underbrace{2 \times L \times H_{\text{kv}} \times d_h \times T_{\text{max}} \times B \times b_{\text{kv}}}_{\text{KV cache}} + \underbrace{O(1)}_{\text{framework overhead (~2GB)}}$$

**Example:** LLaMA 3 70B, INT4 weights, BF16 KV cache, batch=16, max_seq=8192:
- Weights: $70 \times 10^9 \times 0.5 = 35$ GB
- KV cache: $2 \times 80 \times 8 \times 128 \times 8192 \times 16 \times 2 = 42.9$ GB
- **Total: ~80 GB → requires 2× A100 80GB with tensor parallelism**

---

## 9. Inference Benchmarking Metrics

| Metric | Definition | Target | Bottleneck |
| :--- | :--- | :--- | :--- |
| **TTFT** | Time from request to first output token | <200ms | Prefill compute |
| **TPOT** | Time per output token after first | <50ms | KV cache bandwidth |
| **Throughput** | Output tokens/sec across all requests | Maximize | GPU compute utilization |
| **Latency P99** | 99th percentile request latency | <2s | KV cache memory |
| **GPU utilization** | Fraction of time doing useful compute | >80% | Batching efficiency |

**Prefill vs decode:**
- **Prefill** (processing the prompt): compute-bound. Scales with $O(L \times T_{\text{prompt}})$.
- **Decode** (generating tokens): memory-bandwidth-bound. Each step reads all weights once. Effective throughput: $\text{bytes/weight} / \text{memory bandwidth}$.

For A100 80GB (2 TB/s memory bandwidth), 70B BF16 model (140 GB weights):
$$\text{Max decode throughput} = \frac{2 \times 10^{12}}{140 \times 10^9} \approx 14 \text{ tokens/sec (single request)}$$

With batching and quantization, this scales linearly with batch size until compute-bound.

---

## 10. Serving Stack Comparison

| Framework | Best for | Key features |
| :--- | :--- | :--- |
| **vLLM** | High-throughput serving | PagedAttention, continuous batching, AWQ/GPTQ |
| **TensorRT-LLM** | NVIDIA-optimized production | Kernel fusion, INT8/INT4, tensor parallelism |
| **llama.cpp** | CPU + quantized serving | GGUF format, MPS/Metal, runs on MacBook |
| **HuggingFace TGI** | Easy deployment | Flash Attention, continuous batching, Docker |
| **Ollama** | Local/dev serving | GGUF, simple CLI, automatic model management |
| **SGLang** | Complex workflows (multi-turn, LoRA) | RadixAttention (prefix sharing), structured output |

```python
# vLLM with AWQ quantization
from vllm import LLM, SamplingParams

llm = LLM(
    model="TheBloke/Llama-2-70B-AWQ",
    quantization="awq",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    enable_prefix_caching=True,    # reuse KV for shared prefixes
)

outputs = llm.generate(
    ["Explain gradient descent in one paragraph."],
    SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256),
)
```

---

## 11. Production Deployment Checklist

**Memory planning:**
- [ ] Compute weight memory + KV cache at target batch size and max sequence length
- [ ] Account for quantization format — INT4 is ~4× smaller but has quality tradeoffs
- [ ] Leave 10–15% VRAM headroom for activations and framework

**Latency optimization:**
- [ ] Flash Attention enabled (PyTorch 2.0+ SDPA or xformers)
- [ ] Continuous batching (vLLM, TGI) for multi-request serving
- [ ] KV cache compressed with GQA (check model architecture)
- [ ] Prefix caching for common system prompts

**Throughput optimization:**
- [ ] Tensor parallelism for large models (TP=4+ for 70B+)
- [ ] Speculative decoding for latency-sensitive paths (2–3× speedup)
- [ ] Appropriate quantization: AWQ 4-bit for 70B+ models serving at scale

**Quality verification:**
- [ ] Run perplexity delta check before/after quantization
- [ ] Benchmark on task-specific eval set (quantization hurts some tasks more than others)
- [ ] Test edge cases: very long inputs, special tokens, multilingual content

> [!TIP]
> **Interview structure:** LLM deployment = memory arithmetic (weights + KV cache) → quantization (GPTQ/AWQ for 4-bit) → serving optimizations (Flash Attention, continuous batching, PagedAttention) → parallelism strategy (tensor for width, pipeline for depth). The key insight is that decode is memory-bandwidth-bound, not compute-bound — this drives all optimization decisions: smaller models, quantization, batching, speculative decoding to amortize the memory cost.
