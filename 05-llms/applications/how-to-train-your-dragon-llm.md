# How LLMs Actually Work

A modern LLM is a decoder-only Transformer trained to predict the next token given all preceding tokens. Everything else — reasoning, code generation, instruction following — emerges from doing this well at scale.

---

## 1. The Transformer Architecture (Decoder-Only)

A decoder-only Transformer (GPT-style) stacks $L$ identical blocks. Each block contains:

1. **Masked Multi-Head Self-Attention** — attends to all previous tokens (causally masked)
2. **Feed-Forward Network (FFN)** — independent per-token processing
3. **LayerNorm + Residual connections** — stabilize training

```
Input tokens → Token Embedding + Positional Encoding
                        │
               ┌────────▼─────────┐
               │  Decoder Block   │ × L
               │  ┌────────────┐  │
               │  │ LayerNorm  │  │
               │  │ Causal MHA │  │
               │  │ Residual   │  │
               │  └────────────┘  │
               │  ┌────────────┐  │
               │  │ LayerNorm  │  │
               │  │    FFN     │  │
               │  │ Residual   │  │
               │  └────────────┘  │
               └──────────────────┘
                        │
                   LayerNorm
                        │
                   LM Head (Linear → Softmax)
                        │
                  Token probabilities
```

---

## 2. Tokenization

The model does not read words — it reads **tokens**. Most modern LLMs use Byte-Pair Encoding (BPE).

**BPE algorithm:**
1. Start with a character-level vocabulary
2. Repeatedly merge the most frequent adjacent pair into a new token
3. Stop when vocabulary size is reached (typically 32k–128k tokens)

**Practical implications:**

| Input | Tokens | Notes |
| :--- | :--- | :--- |
| common English word "the" | 1 token | Frequent → own token |
| "unbelievable" | 3–4 tokens | Split into subwords |
| "```python" | 2 tokens | Code-specific patterns |
| "2023-11-15" | 4–6 tokens | Dates are expensive |
| Chinese character | 1–3 tokens | Non-Latin costly |

**Token count affects:** cost (API billing), latency (proportional to tokens generated), and context window usage.

---

## 3. Attention Mechanism

Scaled dot-product attention computes how much each token should attend to every other (earlier) token:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- $Q = XW_Q$ — what this token is looking for
- $K = XW_K$ — what tokens are offering
- $V = XW_V$ — what information they contribute
- $\sqrt{d_k}$ — scaling to prevent softmax saturation in high dimensions

**Causal mask:** in decoder-only models, the attention mask prevents token $i$ from attending to token $j > i$. This is what makes left-to-right autoregressive generation possible.

**Multi-head attention:** run $h$ attention heads in parallel on projected subspaces, concatenate outputs:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Each head can specialize in different types of relationships (syntax, coreference, distance, etc.).

---

## 4. Positional Encoding

Transformers process all tokens in parallel — they have no inherent notion of position. Positional encoding injects order information.

### Absolute Sinusoidal (original Transformer)

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

Fixed, not learned. Does not generalize to longer sequences than seen in training.

### Learned Positional Embeddings (GPT-2, GPT-3)

A learnable embedding table indexed by position. Simple but cannot extrapolate beyond training length.

### RoPE (Rotary Position Embedding) — LLaMA, Mistral, GPT-NeoX

Applies a rotation to Q and K vectors based on position. Relative position is captured naturally:

$$\text{RoPE}(q, m) = q \cdot e^{im\theta}$$

Benefits:
- Relative position information flows into attention naturally
- Better length generalization than absolute embeddings
- YaRN / LongRoPE extend RoPE to longer contexts by adjusting the base frequency

### ALiBi (Attention with Linear Biases) — MPT, BLOOM

Adds a linear penalty to attention scores based on token distance:

$$\text{score}(i,j) = q_i k_j^T / \sqrt{d_k} - m \cdot (i-j)$$

Cheap, effective, generalizes beyond training length.

---

## 5. Feed-Forward Network (FFN)

After attention, each token passes through an FFN independently:

$$\text{FFN}(x) = W_2 \cdot \text{activation}(W_1 x + b_1) + b_2$$

The FFN is typically 4× wider than the model dimension. For a model with $d=4096$:
- $W_1 \in \mathbb{R}^{4096 \times 16384}$, $W_2 \in \mathbb{R}^{16384 \times 4096}$

**SwiGLU activation** (LLaMA, PaLM) — outperforms ReLU and GELU empirically:

$$\text{SwiGLU}(x) = \text{Swish}(W_1 x) \odot (W_2 x)$$

The FFN is where most model parameters live. It is thought to store factual associations (key-value memories).

---

## 6. Mixture of Experts (MoE)

Instead of a dense FFN, MoE has $N$ expert FFNs. Each token is routed to only $k$ experts (typically $k=2$):

$$y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x), \quad G(x) = \text{TopK}(\text{softmax}(W_g x))$$

**Benefits:** large parameter count at fraction of compute (active parameters per token = $k/N$ of total).

**Examples:** Mixtral 8×7B (8 experts, top-2 routing = 12.9B active params out of 46.7B total), GPT-4 (speculated).

**Challenge:** load balancing — the router must distribute tokens across experts fairly. Auxiliary load-balancing loss is added during training.

---

## 7. Training

### Pretraining Objective

Standard autoregressive language modeling:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

Trained on internet text, books, code, scientific papers. The model learns language, facts, and reasoning patterns from next-token prediction alone.

### Scaling Laws (Chinchilla)

Optimal training given compute budget $C$:

$$N_{opt} \approx \sqrt{C / 6}, \quad D_{opt} \approx \sqrt{6C}$$

Simplification: **~20 tokens of training data per model parameter** for a compute-optimal run.

| Model | Parameters | Training tokens |
| :--- | :--- | :--- |
| LLaMA 2 7B | 7B | 2T (over-trained for inference) |
| LLaMA 2 70B | 70B | 2T |
| GPT-3 | 175B | 300B (under-trained by Chinchilla) |

**Key insight:** smaller models trained on more data can outperform larger models trained on less data, especially for inference efficiency (fixed inference cost per token).

### Instruction Tuning and RLHF

1. **Pretraining:** predict next token on large corpus → broad capabilities
2. **Supervised Fine-Tuning (SFT):** fine-tune on high-quality instruction-response pairs → follows instructions
3. **RLHF / DPO:** align to human preferences for helpfulness and safety

See [tuning-optimization.md](tuning-optimization.md) for details on SFT, LoRA, RLHF, DPO.

---

## 8. Inference

### Autoregressive Generation

At each step:
1. Compute forward pass on current token sequence
2. Sample from output distribution over vocabulary
3. Append sampled token to sequence
4. Repeat until `<eos>` token or max length

This is **sequential** — cannot be parallelized across output tokens.

### KV Cache

Without caching, every step recomputes all key-value pairs for the entire history. Cost: $O(T^2 d)$ for a sequence of length $T$.

**With KV cache:** store $K$ and $V$ tensors for all past tokens. At each new step, only compute the new token's $Q$, $K$, $V$ and append to cache.

Cost per step: $O(Td)$ — linear in sequence length.

**Memory cost:** $2 \times L \times H \times d_h \times T \times \text{bytes per element}$

For LLaMA 70B (80 layers, 64 heads, 128 head dim) at BF16: ~80GB for 4096 tokens. This limits batch size and context length.

### Speculative Decoding

Use a small draft model to generate candidate tokens, verify multiple tokens per LLM forward pass:

1. Draft model generates $\gamma$ candidate tokens autoregressively (cheap)
2. Target LLM evaluates all $\gamma$ tokens in one parallel forward pass
3. Accept tokens that match the target distribution; reject and resample from the first mismatch

**Speedup:** 2–4× with identical output distribution. Works best when draft and target share vocabulary.

### Flash Attention

Standard attention computes and materializes the $N \times N$ attention matrix — $O(N^2)$ memory.

**Flash Attention** reorders computation to avoid materializing the full matrix by tiling and fusing operations into a single CUDA kernel. Achieves exact attention in $O(N)$ memory with significant throughput improvement.

Flash Attention 2 and 3 are standard in modern training and inference frameworks.

---

## 9. Context Window and Long Context

| Model | Context window |
| :--- | :--- |
| GPT-3 | 4,096 tokens |
| Claude 3.5 Sonnet | 200,000 tokens |
| Gemini 1.5 Pro | 1,000,000 tokens |

**Challenges with very long context:**
- KV cache memory grows linearly with context length
- "Lost in the middle" — attention pays less to middle of long context
- Positional encodings must generalize beyond training length

**Solutions:**
- YaRN, LongRoPE, RoPE scaling for position extrapolation
- Ring attention for multi-GPU context parallelism
- Selective state space models (Mamba) for $O(N)$ long-context processing

---

## 10. Inference Architectures

| Approach | How | Latency | Throughput |
| :--- | :--- | :--- | :--- |
| **Single GPU** | Load full model | Low | Low |
| **Tensor parallelism** | Split weight matrices across GPUs | Medium | High |
| **Pipeline parallelism** | Split layers across GPUs | High (pipeline bubbles) | High |
| **Continuous batching** | Dynamic batching of requests | Low per-token | High |
| **vLLM / PagedAttention** | OS-inspired KV cache memory management | Low | Very high |

**PagedAttention (vLLM):** treats KV cache like virtual memory pages. Non-contiguous physical memory blocks are mapped to contiguous logical blocks. Eliminates KV cache memory fragmentation — enables 24× higher throughput than HuggingFace Transformers at same memory.

---

## 11. Key Metrics

| Metric | Definition | Typical value |
| :--- | :--- | :--- |
| **TTFT** (Time to First Token) | Latency until first output token | 100ms–2s |
| **TPOT** (Time Per Output Token) | Latency per generated token after first | 20–100ms |
| **Throughput** | Tokens per second across all requests | Depends on hardware |
| **Perplexity** | $\exp(-\frac{1}{N}\sum \log P(w_i \mid w_{<i}))$ | Lower = better LM |

> [!TIP]
> **Architecture interview summary:** LLM = decoder-only Transformer with causal attention + RoPE positional encoding + SwiGLU FFN, pretrained with next-token prediction. Inference generates autoregressively with KV cache. The gap between "it works" and "it ships" is filled by quantization, speculative decoding, continuous batching, and PagedAttention.
