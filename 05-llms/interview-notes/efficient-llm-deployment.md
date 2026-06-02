---
module: Llms
topic: Interview Notes
subtopic: Efficient Llm Deployment
status: unread
tags: [llms, ml, interview-notes-efficient-llm-]
---
# Efficient LLM Deployment & Optimization

## The Bottleneck That Drives Every Technique Here

Your model takes 3 seconds per request. You have 10,000 concurrent users. What fails first?

Not the model's quality. Memory. Specifically, GPU memory runs out of space to hold the working state for all those in-flight sequences. When memory fills, requests queue. Queue depth grows. Latency spikes. The system falls over.

But there's a more specific failure hidden inside this: even before memory runs out, you notice the GPU is only doing useful compute 15% of the time. The rest is stalled waiting for data to move between HBM and compute cores. Your expensive A100 is mostly idle — not because the problem is hard, but because **token generation is memory-bandwidth-bound, not compute-bound**.

Every technique in this file — quantization, KV cache optimization, Flash Attention, speculative decoding, continuous batching, model parallelism — is an answer to one or both of these problems:
1. We don't have enough memory to hold all the state we need
2. We're not moving data efficiently enough to keep compute busy

Keep this in your head as you read. The techniques only make sense in that context.

---

## 1. Memory Arithmetic: Before You Optimize, Know What You're Paying

### The Problem

You want to serve LLaMA 3 70B. You have four A100 80GB GPUs. Will it fit? You don't know, because you haven't accounted for what "serving a model" actually requires in memory.

### The Core Insight

Memory cost has two independent components that each scale with different things: **weights** scale with model size, **KV cache** scales with batch size × sequence length. Both must fit simultaneously. Engineers consistently underestimate the KV cache.

### The Mechanics

**Weight memory:**

$$\text{Memory (bytes)} = N_{\text{params}} \times \text{bytes per param}$$

| Format | Bytes/param | 7B model | 70B model |
| :--- | :--- | :--- | :--- |
| FP32 | 4 | 28 GB | 280 GB |
| FP16 / BF16 | 2 | 14 GB | 140 GB |
| INT8 | 1 | 7 GB | 70 GB |
| INT4 | 0.5 | 3.5 GB | 35 GB |

With framework overhead (activations, CUDA context): multiply by ~1.25.

**KV cache memory:**

$$\text{KV Cache (bytes)} = 2 \times L \times H_{\text{kv}} \times d_h \times T \times \text{bytes/element} \times B$$

Where: L = layers, H_kv = KV heads, d_h = head dim, T = sequence length, B = batch size.

For LLaMA 3 70B (80 layers, 8 KV heads after GQA, head_dim=128) in BF16, batch=1, 4096 tokens:

$$= 2 \times 80 \times 8 \times 128 \times 4096 \times 2 \approx 1.34 \text{ GB per request}$$

At batch=32: ~43 GB. **KV cache dominates VRAM at high batch sizes.** This is why scaling batch size is not free.

**Full VRAM estimate:**

$$\text{VRAM} = \underbrace{N_{\text{params}} \times b_w \times 1.25}_{\text{weights + overhead}} + \underbrace{2 \times L \times H_{\text{kv}} \times d_h \times T_{\text{max}} \times B \times b_{\text{kv}}}_{\text{KV cache}}$$

**Worked example:** LLaMA 3 70B, INT4 weights, BF16 KV, batch=16, max_seq=8192:
- Weights: 70B × 0.5B × 1.25 = ~44 GB
- KV cache: 2 × 80 × 8 × 128 × 8192 × 16 × 2 = ~43 GB
- **Total: ~87 GB → requires 2× A100 80GB**

### What Breaks

Underestimate KV cache → OOM at serving time, not load time. The model loads fine, then crashes when actual traffic arrives with real batch sizes. This is a common production failure.

### What the Interviewer Is Testing

Can you do the arithmetic? Do you know the difference between weight memory and KV cache memory? Do you know GQA reduces KV cache (not weights)?

### Common Traps

Forgetting the 1.25× overhead multiplier. Forgetting that KV cache scales with batch size. Assuming INT4 weights means INT4 KV cache — they're independent choices.

---

## 2. Quantization: Fitting More into the Same Hardware

### The Problem

LLaMA 3 70B in BF16 requires 140 GB of GPU memory just for weights. Four A100 80GB cards hold 320 GB total — fine. But most teams have smaller budgets. You need the same model to fit on two A100s, or one, or a consumer GPU. The question is: how much precision can you give up before quality degrades?

### The Core Insight

Neural network weights are not uniformly important. Some columns correspond to activations that barely move; others correspond to activations with huge variance that the model depends on. Uniform quantization treats all weights the same and wastes bits on unimportant ones. The key is to protect the weights that matter and aggressively compress the ones that don't.

### The Mechanics

**Post-Training Quantization (PTQ) — symmetric INT8:**

$$q = \text{round}\left(\frac{w}{s}\right), \quad s = \frac{\max(|w|)}{127}, \quad \hat{w} = q \times s$$

No retraining required. Quality loss is small at INT8.

**GPTQ — second-order layer-wise PTQ:**

Quantize weights layer-by-layer using inverse Hessian to compensate for each column's quantization error:

1. Quantize column j: $q_j = \text{round}(w_j / s)$
2. Compute error: $e_j = w_j - q_j \times s$
3. Distribute error to remaining columns: $\delta W = -\frac{e_j}{[H^{-1}]_{jj}} [H^{-1}]_{:,j}$

This means early quantization errors are absorbed by later columns. Result: much better quality at 4-bit than naive round-to-nearest.

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)
model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
model.quantize(calibration_dataset)
model.save_quantized("./gptq-model")
```

**AWQ — activation-aware weight quantization:**

Key insight: weights paired with large-magnitude activations cause disproportionate output error when quantized. Scale those channels before quantization so they get effective higher precision:

$$s_j = \frac{\max(|X_j|)^{\alpha}}{\max(|W_j|)}, \quad \tilde{W}_j = W_j / s_j, \quad \tilde{X}_j = X_j \cdot s_j$$

Quantize $\tilde{W}_j$. The important channels are scaled down before quantization (smaller range = more effective bits), then compensated at inference by scaling activations up.

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(model_path)
model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
model.save_quantized("./awq-model")
```

**NF4 (Normal Float 4):** instead of uniform INT4 levels, use levels optimally spaced for normally distributed weights (which most pretrained model weights are). Used in bitsandbytes QLoRA.

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,  # quantize the scale factors too
)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
```

**Quality vs compression tradeoff:**

| Method | VRAM reduction | Perplexity delta | Notes |
| :--- | :--- | :--- | :--- |
| FP16 baseline | 1× | 0 | Reference |
| INT8 (bitsandbytes) | ~2× | +0.1–0.3 | Safe for most tasks |
| GPTQ 4-bit | ~4× | +0.3–0.8 | Good balance |
| AWQ 4-bit | ~4× | +0.2–0.6 | Generally better than GPTQ |
| GGUF Q4_K_M | ~4× | +0.4 | llama.cpp CPU serving |

### What Breaks

Quality degradation is not uniform across tasks. A model that passes general perplexity checks after quantization can fail badly on structured output tasks (code, JSON), arithmetic, or low-resource languages — because these require precise probability differences that get crushed by quantization noise.

Calibration dataset mismatch: GPTQ and AWQ calibrate on a dataset that approximates your production distribution. If your users send very different inputs (code, medical text), calibrate on representative data.

### What the Interviewer Is Testing

Understanding of why quantization error is non-uniform, and what GPTQ/AWQ do differently than naive rounding. The ability to choose the right method given constraints (speed to deploy, quality bar, hardware budget). Awareness that quality testing after quantization is not optional.

### Common Traps

Assuming INT4 is always acceptable because perplexity looks fine. Not understanding that NF4 is specifically for normally distributed weights (good for pretrained models, potentially poor for fine-tuned adapters). Confusing weight quantization with activation quantization (W4A16 vs W8A8).

---

## 3. KV Cache Mechanics: The State You Must Keep

### The Problem

During autoregressive generation, at every decode step you need the key and value vectors for every token that came before. Without caching, you'd recompute all previous K/V at every step — O(T²) total compute. With naive caching, you pre-allocate enough memory for the maximum sequence length for every in-flight request. With 100 concurrent requests and a 4096-token limit, most of that memory is empty most of the time.

### The Core Insight

KV cache is unavoidable — you cannot regenerate it cheaply. But the way you allocate it is a choice. Pre-allocating the maximum possible length for every request is like renting a warehouse for a grocery bag. You need virtual memory — allocate space as you actually use it.

### The Mechanics

**Standard KV cache:**

Store K and V for all past tokens. At each decode step, fetch them all for attention, append the new K/V, advance the position counter.

```python
class KVCache:
    def __init__(self, max_seq_len, n_layers, n_kv_heads, head_dim, dtype=torch.float16):
        shape = (n_layers, 2, max_seq_len, n_kv_heads, head_dim)
        self.cache = torch.zeros(shape, dtype=dtype, device="cuda")
        self.current_pos = 0

    def update(self, layer_idx, k, v):
        seq_len = k.shape[1]
        self.cache[layer_idx, 0, self.current_pos:self.current_pos + seq_len] = k
        self.cache[layer_idx, 1, self.current_pos:self.current_pos + seq_len] = v
        self.current_pos += seq_len

    def get(self, layer_idx):
        return (self.cache[layer_idx, 0, :self.current_pos],
                self.cache[layer_idx, 1, :self.current_pos])
```

**GQA (Grouped Query Attention):** LLaMA 3 70B uses 8 KV heads vs 64 query heads. KV cache scales with KV heads, not query heads. That's an 8× reduction in KV cache memory versus standard MHA with the same number of attention heads. This is why GQA adoption was rapid — no quality loss, 8× KV savings.

**PagedAttention (vLLM):**

Naive pre-allocation wastes 60–80% of KV memory through fragmentation. PagedAttention applies OS virtual memory paging to KV cache:

- KV cache divided into fixed-size **pages** (e.g., 16 tokens per page)
- Page table maps logical sequence positions to physical pages
- Allocate pages on demand; release immediately when request finishes
- Share physical pages across requests with identical prefixes (copy-on-write)

```
Logical sequence:  [0..15][16..31][32..47]...
                      ↓       ↓       ↓
Physical pages:   [Page 42][Page 7][Page 31]  ← non-contiguous, allocated on demand
```

Result: 24× higher throughput than naive HuggingFace Transformers batching on the same hardware.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    gpu_memory_utilization=0.90,  # fraction reserved for KV pages
    max_model_len=8192,
    block_size=16,                # tokens per KV page
    enable_prefix_caching=True,   # reuse KV for shared system prompts
)
```

**Prefix caching:** when many requests share the same system prompt (1000 tokens), compute its KV entries once and reuse across all requests. The prefill cost for those 1000 tokens is paid once, not per-request. Directly reduces TTFT.

### What Breaks

PagedAttention still has overhead: maintaining page tables, handling page faults, managing copy-on-write for prefix sharing. For very short requests (< 1 block), page management overhead dominates. For very long requests with unique prefixes, there's no sharing benefit.

KV cache at high batch size becomes the memory bottleneck even with quantized weights. With LLaMA 3 70B INT4 (35 GB weights), batch=32 at 4096 tokens gives 43 GB KV — the KV now exceeds the weight memory.

### What the Interviewer Is Testing

Whether you understand that KV cache is not constant — it grows with sequence length and batch size. Whether you can derive the memory formula. Whether you understand why PagedAttention was a step change in throughput.

### Common Traps

Thinking GQA reduces model quality (it doesn't in practice; it reduces KV cache size). Forgetting that prefix caching only helps when prefixes are actually shared — it does nothing for unique prompts. Not accounting for KV cache in VRAM estimates.

---

## 4. Speculative Decoding: Parallelizing What Was Sequential

### The Problem

Autoregressive generation is fundamentally sequential: you cannot generate token T+1 until you have token T. A 70B model generates ~14 tokens/second on a single A100 (memory-bandwidth-bound). For a 256-token response, that's 18 seconds — unusable for interactive applications.

You could use a smaller model (faster, but lower quality). You could use a larger batch (higher throughput, but not lower per-request latency). Neither solves the latency problem for a single user waiting for a response.

### The Core Insight

The target model does one forward pass per token. But a forward pass over a sequence of γ tokens takes almost the same time as a forward pass over one token (for the decode phase — it's memory-bandwidth-bound on weights, not sequence length). If you can cheaply *guess* the next γ tokens and verify them all at once, you get γ tokens worth of output per target model forward pass.

The trick: use a small draft model to guess, use the target model to verify in parallel. The output distribution is mathematically guaranteed to be identical to the target model — no approximation.

### The Mechanics

1. Draft model autoregressively generates γ candidate tokens: $t_1, t_2, \ldots, t_\gamma$
2. Target model evaluates all γ positions in one parallel forward pass
3. Accept token $t_i$ with probability:
$$\alpha_i = \min\left(1, \frac{p_\text{target}(t_i \mid x)}{p_\text{draft}(t_i \mid x)}\right)$$
4. At first rejection, resample from the corrected distribution:
$$p_\text{corrected}(t) = \text{normalize}\left(\max(0,\; p_\text{target}(t) - p_\text{draft}(t))\right)$$
5. Guaranteed: if you marginalize over all possible draft sequences, the output distribution equals the target distribution exactly.

```python
def speculative_decode(target_model, draft_model, prompt_tokens, max_new_tokens, gamma=4):
    generated = prompt_tokens.clone()

    while generated.shape[1] - prompt_tokens.shape[1] < max_new_tokens:
        # Draft: generate gamma tokens
        draft_tokens, draft_probs = [], []
        input_seq = generated.clone()
        for _ in range(gamma):
            with torch.no_grad():
                logits = draft_model(input_seq).logits[:, -1, :]
            prob = torch.softmax(logits, dim=-1)
            token = torch.multinomial(prob, 1)
            draft_tokens.append(token)
            draft_probs.append(prob)
            input_seq = torch.cat([input_seq, token], dim=1)

        # Target: verify all gamma tokens in one pass
        verify_input = torch.cat([generated] + draft_tokens, dim=1)
        with torch.no_grad():
            target_logits = target_model(verify_input).logits

        # Accept/reject
        n_accepted = 0
        for i, (dt, dp) in enumerate(zip(draft_tokens, draft_probs)):
            tp = torch.softmax(target_logits[:, generated.shape[1] + i - 1, :], dim=-1)
            accept_prob = torch.min(torch.ones(1), tp[:, dt] / dp[:, dt])
            if torch.rand(1) < accept_prob:
                generated = torch.cat([generated, dt], dim=1)
                n_accepted += 1
            else:
                corrected = torch.clamp(tp - dp, min=0)
                corrected /= corrected.sum()
                fallback = torch.multinomial(corrected, 1)
                generated = torch.cat([generated, fallback], dim=1)
                break

        if n_accepted == gamma:
            # All accepted: bonus token from target
            bonus = torch.multinomial(torch.softmax(target_logits[:, -1, :], dim=-1), 1)
            generated = torch.cat([generated, bonus], dim=1)

    return generated
```

**Speedup analysis:** if the draft model agrees with the target on fraction α of tokens, expected tokens per target forward pass = $\frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$. At α=0.8, γ=4: ~3.4 tokens per pass versus 1 without speculation.

**Self-speculative variants:**
- **Medusa:** extra parallel LM heads at the final layer, each predicting k steps ahead. No separate draft model.
- **Eagle:** a small draft model that reuses the target's hidden states, achieving higher acceptance rates than an independent draft model.

### What Breaks

Acceptance rate falls for out-of-distribution content (the draft model was trained on different data), or for high-temperature / diverse sampling (the target and draft distributions diverge). When acceptance rate drops below ~0.5, speculative decoding is slower than direct decoding due to wasted draft compute.

Speculative decoding adds latency in the P99 case because rejected tokens require a rejection-resample cycle.

### What the Interviewer Is Testing

Whether you understand *why* the output distribution is unchanged (the acceptance-rejection argument). Whether you know speculative decoding is a latency optimization for single-request P50, not a throughput optimization. Whether you understand when it's harmful (low acceptance rate, high-temperature sampling).

### Common Traps

Claiming speculative decoding increases throughput (it doesn't necessarily — it reduces per-request latency). Forgetting the output distribution guarantee requires the corrected rejection-resample, not just stopping at rejection. Not knowing that γ is a hyperparameter that needs tuning per workload.

---

## 5. Continuous Batching: Never Let the GPU Wait

### The Problem

A serving system runs a batch of requests. Request A needs 50 tokens, Request B needs 500 tokens. With static batching, you wait for the entire batch to finish before accepting new requests. While Request B grinds through tokens 51–500, the GPU slot that Request A vacated sits empty. GPU utilization collapses for mixed-length request distributions.

### The Core Insight

You don't need to wait for a whole batch to finish before inserting new work. Every decode step is a matrix multiply over the current batch. You can swap finished requests out and new requests in *between steps* — iteration-level scheduling instead of request-level scheduling.

### The Mechanics

At each decode step:
1. Run one decode step for all active requests in the current batch
2. Check which requests just generated an EOS token or reached max_tokens
3. Remove those requests from the batch, release their KV cache pages
4. Fill vacant slots with the highest-priority waiting requests (after their prefill)
5. Repeat

```
Step 1: [Req A: tok 1] [Req B: tok 1] [Req C: tok 1]
Step 2: [Req A: tok 2] [Req B: tok 2] [Req C: tok 2]
Step 3: [Req A: tok 3] [Req B: DONE ] [Req D: tok 1]  ← D inserted immediately
Step 4: [Req A: tok 4] [Req D: tok 2] [Req E: tok 1]  ← E inserted
```

**Throughput improvement:** 5–10× over static batching for typical mixed-length distributions. Combined with PagedAttention (which enables fine-grained KV memory management), this is the basis for vLLM's throughput advantage.

**Prefill-decode separation:** prefill (processing the input prompt) is compute-intensive; decode (generating one token at a time) is memory-bandwidth-bound. Some systems route them to separate GPU pools to prevent long prefills from stalling decode for other requests.

### What Breaks

Continuous batching increases batch variability — different requests are at different positions in their generation. This complicates tensor shapes (ragged batches) and requires careful KV cache indexing. Systems like vLLM handle this transparently.

If all requests happen to be long (batch is always full), continuous batching provides no benefit over static batching. It helps most when request length variance is high.

### What the Interviewer Is Testing

Whether you understand the distinction between static (request-level) and continuous (iteration-level) scheduling. Whether you can articulate why it improves utilization for mixed-length workloads.

### Common Traps

Confusing continuous batching with dynamic batching (which adjusts batch size at request boundaries, not step boundaries). Not knowing that prefill and decode are different phases with different bottlenecks.

---

## 6. Flash Attention: Moving Data Smarter

### The Problem

Standard attention materializes the full N × N attention weight matrix in GPU HBM. For sequence length 4096 with 64 heads: $4096^2 \times 4 \text{ bytes} \times 64 = 4.3$ GB per layer. With 80 layers: over 340 GB just for attention intermediates. This exceeds the available VRAM for a single forward pass. Even if it didn't, reading and writing that much data to/from HBM is the primary latency bottleneck.

### The Core Insight

You never actually need the full N × N matrix in memory simultaneously. You only need it to compute a weighted sum of V vectors. If you tile the computation into SRAM-sized blocks and track the softmax normalization factor across tiles, you can compute the exact same output without ever writing the full matrix to HBM.

### The Mechanics

The online softmax trick: maintain a running maximum m and normalizer ℓ as you process tiles.

$$m_{\text{new}} = \max(m_{\text{old}},\; m_{\text{block}}), \quad \ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \ell_{\text{old}} + \sum_j e^{x_j - m_{\text{new}}}$$

At each tile, load Q_tile, K_tile, V_tile into SRAM (~20 MB on A100). Compute local attention scores, update running normalizer and output accumulator. Write only the final output to HBM.

**Result:**
- Memory: O(N) instead of O(N²) — attention matrices never materialize
- Speed: 2–4× faster for long sequences (less HBM traffic)
- Output: mathematically identical — not an approximation

```python
import torch.nn.functional as F

# Flash Attention via PyTorch 2.0+ SDPA
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )
```

### What Breaks

Flash Attention requires Q, K, V to fit in SRAM per tile. If head dimension is very large or the GPU has small SRAM, tiling becomes inefficient. For very short sequences (< 512 tokens), standard attention is faster because the overhead of the online softmax tracking dominates.

Flash Attention 2 and 3 add additional optimizations: better parallelization across query heads (FA2), specialized Tensor Core utilization (FA3). You should know which version is in your inference stack.

### What the Interviewer Is Testing

Whether you know Flash Attention is an IO-aware algorithm, not an approximation. Whether you understand why memory complexity drops from O(N²) to O(N). Whether you can explain the online softmax trick conceptually.

### Common Traps

Claiming Flash Attention changes the output (it doesn't). Thinking it helps equally at all sequence lengths (it helps most for long sequences where N² HBM traffic dominates). Not knowing that PyTorch 2.0 SDPA uses Flash Attention automatically unless you've disabled it.

---

## 7. Model Parallelism: When One GPU Isn't Enough

### The Problem

LLaMA 3 405B in BF16 requires ~810 GB just for weights. No single GPU holds that. You need the model split across multiple GPUs. But *how* you split it determines whether communication overhead kills your throughput.

### The Core Insight

There are two fundamentally different ways to split a model across GPUs, and they have different communication patterns:
- **Tensor parallelism:** split a single layer across GPUs (wide split). High communication frequency, low latency per communication.
- **Pipeline parallelism:** split layers across GPUs (depth split). Low communication frequency, but GPUs can be idle waiting for the previous stage.

The right choice depends on interconnect bandwidth and model size.

### The Mechanics

**Tensor Parallelism (Megatron-style):**

Split weight matrices column-wise across GPUs, row-wise on the projection back:

$$Y = XW \rightarrow Y = X[W_1 \mid W_2] = [XW_1 \mid XW_2]$$

Each GPU computes a partial output. AllReduce sums the partials after each layer. Requires one AllReduce per Transformer block (2 per layer in practice).

```
GPU 0: W[:, :d/2]    → partial_Y_0 ↘
                                      AllReduce → full Y
GPU 1: W[:, d/2:]    → partial_Y_1 ↗
```

**Communication requirement:** AllReduce bandwidth = 2 × (B × T × d_model × bytes) per layer. At 600 GB/s NVLink vs 64 GB/s PCIe, NVLink makes TP viable; PCIe makes TP prohibitively slow.

**Pipeline Parallelism:**

Assign consecutive layer blocks to each GPU. Data flows forward through the pipeline.

```
GPU 0: layers  0–19  →  GPU 1: layers 20–39  →  GPU 2: layers 40–59  →  GPU 3: layers 60–79
```

**Pipeline bubbles:** GPU 0 is idle waiting for gradients (training) or the next batch (inference). Bubble fraction = (P-1)/(M+P-1) where P = pipeline stages, M = microbatches. With M=8 microbatches and P=4 stages, bubble = 3/11 ≈ 27%. Microbatching amortizes this.

**Context Parallelism (Ring Attention):** for sequences > 128k tokens, split the sequence across GPUs. Each GPU computes local attention and passes KV to neighbors in a ring. Communication scales linearly with sequence length.

**Recommended strategies:**

| Model size | GPUs | Strategy |
| :--- | :--- | :--- |
| ≤13B | 1–2 × 80GB | Single GPU or TP=2 |
| 70B | 4–8 × 80GB | TP=4 to TP=8 |
| 405B | 16–64 × 80GB | TP=8 + PP=4 |
| >405B | 64+ | 3D (TP + PP + DP) |

### What Breaks

TP requires NVLink — on PCIe-only clusters, AllReduce latency exceeds the savings from parallelism. PP has pipeline bubbles that reduce GPU utilization; at small batch sizes, bubbles dominate. Context parallelism adds ring communication proportional to sequence count, which caps throughput for very long sequences.

### What the Interviewer Is Testing

Whether you can explain the two split dimensions (tensor vs pipeline) and why they have different communication patterns. Whether you know that TP requires NVLink to be effective. Whether you know the bubble fraction formula and how microbatching addresses it.

### Common Traps

Conflating TP (splits within a layer) with PP (splits across layers). Not knowing that FSDP (for training) and TP (for inference) are different approaches. Assuming you can just use TP=N for any N without checking interconnect bandwidth.

---

## 8. VRAM Calculation for Serving

### The Problem

You need to provision hardware for a serving job. Under-provision and you get OOM. Over-provision and you waste budget. You need a formula.

### The Core Insight

Three things live in VRAM simultaneously: weights, KV cache for all active requests, and framework overhead. All three must fit. Budget them independently before combining.

### The Mechanics

$$\text{VRAM} = \underbrace{N_{\text{params}} \times b_w}_{\text{weights}} \times 1.25 \;\;+\;\; \underbrace{2 \times L \times H_{\text{kv}} \times d_h \times T_{\text{max}} \times B \times b_{\text{kv}}}_{\text{KV cache}}$$

**Full worked example — LLaMA 3 70B production serving:**
- Config: INT4 weights, BF16 KV cache, batch=16, max_seq=8192
- Weights: 70 × 10⁹ × 0.5 bytes × 1.25 = 43.75 GB
- KV cache: 2 × 80 × 8 × 128 × 8192 × 16 × 2 bytes = 42.9 GB
- **Total: ~87 GB → 2× A100 80GB with TP=2**

**Sensitivity analysis:** doubling batch size doubles KV cache. Doubling max_seq doubles KV cache. Switching from BF16 KV to INT8 KV halves KV cache. These are the levers you pull when you're near memory limits.

### What the Interviewer Is Testing

Whether you can derive the VRAM estimate from first principles, not just guess "a few A100s." Whether you account for both weight and KV memory. Whether you know which knobs to turn to fit within a budget.

### Common Traps

Forgetting the 1.25× overhead multiplier. Assuming INT4 quantization means INT4 KV cache. Not knowing that max_seq in the VRAM formula is the *allocated* max, not the average actual sequence length.

---

## 9. Inference Benchmarking Metrics: What to Measure

### The Problem

Your system "feels slow" in testing. But "slow" means different things for different users: a user waiting for the first word to appear experiences latency differently than a system calculating cost per million tokens. You need precise metrics to know what's actually broken.

### The Core Insight

LLM inference has two distinct phases with different bottlenecks and different user impacts:
- **Prefill** (processing the input prompt): compute-bound, scales with prompt length
- **Decode** (generating output tokens): memory-bandwidth-bound, scales with model size

TTFT measures the prefill phase. TPOT measures the decode phase. They can fail independently.

### The Mechanics

| Metric | Definition | Primary bottleneck |
| :--- | :--- | :--- |
| **TTFT** | Time from request submission to first output token | Prefill compute (scales with prompt tokens) |
| **TPOT** | Time per output token after the first | KV cache memory bandwidth (scales with model size) |
| **Throughput** | Output tokens/sec across all requests | Batch size × GPU utilization |
| **P99 latency** | 99th percentile end-to-end request time | KV cache memory, queue depth |
| **GPU utilization** | Fraction of wall time doing useful compute | Batching efficiency |

**Decode bandwidth ceiling:**

For A100 80GB (2 TB/s HBM bandwidth), 70B BF16 model (140 GB weights):
$$\text{Max decode throughput (single request)} = \frac{2 \times 10^{12}}{140 \times 10^9} \approx 14 \text{ tokens/sec}$$

This is a hard ceiling per request, independent of hardware count, because every decode step must read all weights. Quantization (INT4: 35 GB) pushes this to:
$$\frac{2 \times 10^{12}}{35 \times 10^9} \approx 57 \text{ tokens/sec}$$

Batching amortizes weight reads across requests until you become compute-bound.

### What Breaks

Optimizing TTFT when users are complaining about slow completion (need to optimize TPOT). Running benchmarks at batch=1 that don't reflect production batch sizes. Reporting mean latency when P99 failures are causing user-visible errors.

### What the Interviewer Is Testing

Whether you know TTFT and TPOT are distinct metrics with distinct root causes. Whether you can derive the decode bandwidth ceiling. Whether you know that latency and throughput can be optimized somewhat independently.

### Common Traps

Treating latency and throughput as synonymous. Not knowing that prefill and decode have different computational bottlenecks. Reporting only mean latency (P99 is what SLAs are written against).

---

## 10. Serving Stack Comparison

### The Problem

You need to pick a serving framework. Each has different performance profiles, supported features, and operational complexity. The wrong choice means either leaving performance on the table or building unnecessary infrastructure.

### The Core Insight

The right framework depends on your primary constraint: throughput for scale (vLLM), maximum NVIDIA optimization (TensorRT-LLM), CPU/consumer hardware (llama.cpp), ease of deployment (TGI/Ollama), or complex multi-turn workflows with prefix sharing (SGLang).

### The Mechanics

| Framework | Best for | Key features |
| :--- | :--- | :--- |
| **vLLM** | High-throughput API serving | PagedAttention, continuous batching, AWQ/GPTQ, prefix caching |
| **TensorRT-LLM** | NVIDIA-optimized production | Kernel fusion, INT8/INT4, TP/PP, maximum throughput on A100/H100 |
| **llama.cpp** | CPU + quantized edge serving | GGUF format, Metal/MPS support, runs on MacBook |
| **HuggingFace TGI** | Easy deployment | Flash Attention, continuous batching, Docker-native |
| **Ollama** | Local/developer serving | GGUF, simple CLI, automatic model management |
| **SGLang** | Complex multi-turn/LoRA workflows | RadixAttention (advanced prefix sharing), structured output engine |

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="TheBloke/Llama-2-70B-AWQ",
    quantization="awq",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    enable_prefix_caching=True,
)

outputs = llm.generate(
    ["Explain gradient descent in one paragraph."],
    SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256),
)
```

### What Breaks

TensorRT-LLM requires NVIDIA-specific compilation and doesn't run on AMD/CPU. vLLM doesn't match TensorRT-LLM peak throughput for NVIDIA workloads. llama.cpp trades throughput for portability. Framework lock-in is real — switching serving frameworks requires re-benchmarking, re-validating quality, and potentially re-deploying client integrations.

### What the Interviewer Is Testing

Whether you know the landscape and can make a justified recommendation given constraints. Whether you can articulate why you'd pick vLLM over TGI (or vice versa) for a specific scenario.

### Common Traps

Recommending TensorRT-LLM for a startup that doesn't have NVIDIA enterprise support. Recommending vLLM without checking whether your model architecture is supported. Not knowing that GGUF is llama.cpp's format and not directly compatible with vLLM.

---

## The Through-Line

Every technique in this file is answering the same two questions:

1. **Can we fit more model/context into the same VRAM?** → Quantization (weights), GQA (KV size), PagedAttention (KV allocation efficiency), prefix caching (KV reuse)

2. **Can we get more useful tokens per second from the hardware we have?** → Flash Attention (reduce HBM traffic), continuous batching (reduce GPU idle time), speculative decoding (parallelize sequential generation), tensor parallelism (use more GPUs)

The key insight that ties everything together: **decode is memory-bandwidth-bound**. You're not waiting for compute — you're waiting for weights to move from HBM to compute cores. That's why smaller models (quantization), less data movement (Flash Attention), and better hardware utilization (batching) all help.

## Rapid Recall

### Weights
- Direct Answer: 70B × 0.5B × 1.25 = ~44 GB
- Why: This matters because it tells you how to reason about weights.
- Pitfall: Don't answer "Weights" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: 70B × 0.5B × 1.25 = ~44 GB

### KV cache
- Direct Answer: 2 × 80 × 8 × 128 × 8192 × 16 × 2 = ~43 GB
- Why: This matters because it tells you how to reason about kv cache.
- Pitfall: Don't answer "KV cache" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: 2 × 80 × 8 × 128 × 8192 × 16 × 2 = ~43 GB

### Total
- Direct Answer: ~87 GB → requires 2× A100 80GB
- Why: This matters because it tells you how to reason about total.
- Pitfall: Don't answer "Total" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: ~87 GB → requires 2× A100 80GB

### KV cache divided into fixed-size pages (e.g., 16 tokens per page)
- Direct Answer: KV cache divided into fixed-size pages (e.g., 16 tokens per page)
- Why: This matters because it tells you how to reason about kv cache divided into fixed-size pages (e.g., 16 tokens per page).
- Pitfall: Don't answer "KV cache divided into fixed-size pages (e.g., 16 tokens per page)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: KV cache divided into fixed-size pages (e.g., 16 tokens per page)

### Page table maps logical sequence positions to physical pages
- Direct Answer: Page table maps logical sequence positions to physical pages
- Why: This matters because it tells you how to reason about page table maps logical sequence positions to physical pages.
- Pitfall: Don't answer "Page table maps logical sequence positions to physical pages" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Page table maps logical sequence positions to physical pages

### Allocate pages on demand; release immediately when request finishes
- Direct Answer: Allocate pages on demand; release immediately when request finishes
- Why: This matters because it tells you how to reason about allocate pages on demand; release immediately when request finishes.
- Pitfall: Don't answer "Allocate pages on demand; release immediately when request finishes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Allocate pages on demand; release immediately when request finishes

### Share physical pages across requests with identical prefixes (copy-on-write)
- Direct Answer: Share physical pages across requests with identical prefixes (copy-on-write)
- Why: This matters because it tells you how to reason about share physical pages across requests with identical prefixes (copy-on-write).
- Pitfall: Don't answer "Share physical pages across requests with identical prefixes (copy-on-write)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Share physical pages across requests with identical prefixes (copy-on-write)

### Medusa
- Direct Answer: extra parallel LM heads at the final layer, each predicting k steps ahead. No separate draft model.
- Why: This matters because it tells you how to reason about medusa.
- Pitfall: Don't answer "Medusa" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: extra parallel LM heads at the final layer, each predicting k steps ahead. No separate draft model.

### Eagle
- Direct Answer: a small draft model that reuses the target's hidden states, achieving higher acceptance rates than an independent draft model.
- Why: This matters because it tells you how to reason about eagle.
- Pitfall: Don't answer "Eagle" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: a small draft model that reuses the target's hidden states, achieving higher acceptance rates than an independent draft model.

### Memory: O(N) instead of O(N²)
- Direct Answer: attention matrices never materialize
- Why: This matters because it tells you how to reason about memory: o(n) instead of o(n²).
- Pitfall: Don't answer "Memory: O(N) instead of O(N²)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: attention matrices never materialize

### Speed
- Direct Answer: 2–4× faster for long sequences (less HBM traffic)
- Why: This matters because it tells you how to reason about speed.
- Pitfall: Don't answer "Speed" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: 2–4× faster for long sequences (less HBM traffic)

### Output: mathematically identical
- Direct Answer: not an approximation
- Why: This matters because it tells you how to reason about output: mathematically identical.
- Pitfall: Don't answer "Output: mathematically identical" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: not an approximation

### Tensor parallelism
- Direct Answer: split a single layer across GPUs (wide split). High communication frequency, low latency per communication.
- Why: This matters because it tells you how to reason about tensor parallelism.
- Pitfall: Don't answer "Tensor parallelism" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: split a single layer across GPUs (wide split). High communication frequency, low latency per communication.

### Pipeline parallelism
- Direct Answer: split layers across GPUs (depth split). Low communication frequency, but GPUs can be idle waiting for the previous stage.
- Why: This matters because it tells you how to reason about pipeline parallelism.
- Pitfall: Don't answer "Pipeline parallelism" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: split layers across GPUs (depth split). Low communication frequency, but GPUs can be idle waiting for the previous stage.

### Config
- Direct Answer: INT4 weights, BF16 KV cache, batch=16, max_seq=8192
- Why: This matters because it tells you how to reason about config.
- Pitfall: Don't answer "Config" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: INT4 weights, BF16 KV cache, batch=16, max_seq=8192

### Weights
- Direct Answer: 70 × 10⁹ × 0.5 bytes × 1.25 = 43.75 GB
- Why: This matters because it tells you how to reason about weights.
- Pitfall: Don't answer "Weights" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: 70 × 10⁹ × 0.5 bytes × 1.25 = 43.75 GB

### KV cache
- Direct Answer: 2 × 80 × 8 × 128 × 8192 × 16 × 2 bytes = 42.9 GB
- Why: This matters because it tells you how to reason about kv cache.
- Pitfall: Don't answer "KV cache" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: 2 × 80 × 8 × 128 × 8192 × 16 × 2 bytes = 42.9 GB

### Total
- Direct Answer: ~87 GB → 2× A100 80GB with TP=2
- Why: This matters because it tells you how to reason about total.
- Pitfall: Don't answer "Total" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: ~87 GB → 2× A100 80GB with TP=2

### Prefill (processing the input prompt)
- Direct Answer: compute-bound, scales with prompt length
- Why: This matters because it tells you how to reason about prefill (processing the input prompt).
- Pitfall: Don't answer "Prefill (processing the input prompt)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: compute-bound, scales with prompt length

### Decode (generating output tokens)
- Direct Answer: memory-bandwidth-bound, scales with model size
- Why: This matters because it tells you how to reason about decode (generating output tokens).
- Pitfall: Don't answer "Decode (generating output tokens)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: memory-bandwidth-bound, scales with model size
