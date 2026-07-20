---
module: Architectures
topic: Components
subtopic: Transformers
status: unread
tags: [deeplearning, ml, components-transformers]
---
**Primary reference:** [LLM Architecture Deep Dive](../10-llms/03-architecture-deep-dive.md) | [Interview Q&As](../10-llms/interview-notes/01-llm-fundamentals.md)

# Transformers

---

## Why the Sequential Bottleneck Had to Go

**The problem**: RNNs process sequences one token at a time. To learn that a verb at position 1 agrees with its subject at position 20, the relevant information must survive intact through 18 hidden state updates. Gradients flowing back from position 20 to position 1 are multiplied by 18 weight matrices and 18 activation derivatives — they vanish. Long-range dependencies are hard to learn.

The sequential processing also prevents parallelism: you cannot compute hidden state $h_t$ until $h_{t-1}$ is done. Training on long sequences is slow even on hardware that could parallelize.

**The core insight**: instead of threading context through a bottleneck hidden state, let every token directly look at every other token in a single step. Context is computed in parallel, and the gradient path from position 20 to position 1 is just one attention operation — no vanishing chain of multiplications.

---

## The Transformer Block

A single Transformer layer applies two sub-layers, both wrapped in a residual connection and layer normalization:

```
Input x
  │
  ├─ LayerNorm (pre-norm)
  │     ↓
  │  Multi-Head Self-Attention
  │     ↓
  ├─ Residual: x = x + Attention(LN(x))
  │
  ├─ LayerNorm (pre-norm)
  │     ↓
  │  Feed-Forward Network (FFN)
  │     ↓
  └─ Residual: x = x + FFN(LN(x))
```

This block is stacked $L$ times. Each block refines the token representations; the final representations are fed to a task head.

The residual connections ensure that gradients flow directly from the output head to early layers without any multiplication — the skip path has gradient 1 regardless of what the sublayer does.

---

## Self-Attention and Multi-Head Attention

> Full mechanics of scaled dot-product attention, multi-head attention, causal masking, Q/K/V projections, MQA, and GQA are covered in [attention.md](02-attention.md). This section summarizes what matters for understanding the Transformer block structure.

Self-attention lets every token directly look at every other token in a single step:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Multi-head attention runs $h$ attention computations in parallel with independent projections, then concatenates:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

For autoregressive decoders, a causal mask zeros out future positions. Computational cost is $O(n^2 d)$ — the bottleneck for long sequences.

---

## Feed-Forward Network (FFN)

**The problem**: attention layers aggregate information across positions but are limited in what non-linear transformation they can apply to individual token representations. The model needs per-token computation that does not require cross-token interaction.

**The core insight**: after attention has updated each token representation with context, apply a per-position feedforward network. This is where the model processes and transforms individual token representations — it is position-wise (independent across tokens) and adds the model's main non-linear capacity.

**The mechanics**:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Width typically $4 \times d_\text{model}$. LLaMA-style models replace this with SwiGLU:

$$\text{FFN}(x) = (\text{Swish}(xW_1) \odot xW_2) W_3$$

Three projections instead of two; hidden dimension $\frac{8}{3}d_\text{model}$ to keep parameter count matched.

**What breaks**: the FFN is the largest part of the model by parameter count — roughly two thirds of all parameters in a standard Transformer are in FFN layers. Making it more efficient (narrower, sparser, or using mixture-of-experts) is a major target for scaling.

### Mixture-of-Experts (MoE) FFN

**The problem**: making the FFN wider increases model capacity but also increases the compute cost of *every* forward pass — a dense FFN activates 100% of its parameters for every single token, so parameter count and FLOPs scale together. Frontier models want trillions of parameters worth of capacity without paying trillions of FLOPs per token.

**The core insight**: replace the single FFN with $N$ parallel "expert" FFNs of the same shape, plus a lightweight router that selects only the top-$k$ experts (typically $k=1$ or $2$) per token. Total parameter count scales with $N$, but compute per token only scales with $k$ — decoupling capacity from per-token cost.

$$y = \sum_{i \in \text{Top-}k(\text{router}(x))} g_i(x) \cdot \text{FFN}_i(x)$$

where $\text{router}(x) = \text{softmax}(xW_r)$ produces per-expert scores, $\text{Top-}k$ selects the $k$ highest-scoring experts, and $g_i(x)$ is the (renormalized) gate weight for expert $i$. Everything else in the Transformer block (attention, norm, residuals) stays dense — only the FFN sublayer becomes sparse.

**What breaks**: naive top-$k$ routing collapses — the router quickly learns to send most tokens to a handful of "popular" experts, starving the rest of gradient signal (a rich-get-richer dynamic). This is fixed with an auxiliary load-balancing loss that penalizes uneven routing, and at inference, a fixed per-expert capacity that drops or reroutes tokens once an expert's buffer is full. Routing also adds a genuinely new failure surface at serving time — expert-parallel deployments must shard experts across devices and route tokens with all-to-all communication, which is why MoE inference infrastructure (batching, expert placement, capacity tuning) is nontrivial compared to dense models.

> This is only the architectural sketch. Router variants (softmax vs. expert-choice), the load-balancing loss derivations, fine-grained expert segmentation (DeepSeek MoE), token-dropping/capacity-factor mechanics, expert-parallelism sharding, and what experts empirically specialize in are all covered in depth in [05-llms/09-moe-advanced-and-routing.md](../10-llms/09-moe-advanced-and-routing.md) — start there for anything beyond "how does the FFN become sparse."

---

## Positional Encodings

> Full treatment of sinusoidal, RoPE, ALiBi, and learned encodings is in [attention.md](02-attention.md). Summary below.

Attention is permutation-invariant — positional encodings inject order information. Three main families:

- **Sinusoidal** (original Transformer): fixed frequencies, no parameters, limited length generalization
- **RoPE** (LLaMA, Mistral, most modern LLMs): rotates Q/K vectors so relative distance appears directly in attention dot products. Better length generalization, extendable via YaRN/LongRoPE
- **ALiBi** (MPT, BLOOM): subtracts a linear bias proportional to distance from attention scores. No parameters, strong extrapolation

---

## Pre-Norm vs Post-Norm

**The problem**: where does the layer normalization go — before or after the sublayer?

**Post-Norm** (original "Attention is All You Need"):

$$x = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Gradients must flow through the normalization on the path back through the residual stream. This adds an extra scaling step in the gradient path, making very deep networks unstable.

**Pre-Norm** (modern default — GPT-2+, LLaMA, most LLMs):

$$x = x + \text{Sublayer}(\text{LayerNorm}(x))$$

The normalization happens only inside the sublayer. The residual stream $x$ is updated by direct addition — the gradient path through the skip connection has gradient magnitude 1, completely bypassing any normalization scaling.

**What breaks** with Pre-Norm: the final residual stream values are unnormalized. A final LayerNorm before the output head is always required in Pre-Norm architectures. Without it, logit magnitudes vary unpredictably.

---

## Encoder-Only, Decoder-Only, Encoder-Decoder

**The problem**: different tasks have different informational requirements. Understanding a sentence (classification) benefits from seeing the full context. Generating text requires not seeing future tokens. Translating requires encoding a source and generating into a target space.

**Encoder-Only** — bidirectional attention; every token attends to every other:

**What it's for**: tasks where the full input is known upfront — classification, named entity recognition, extractive QA, embeddings.

Examples: BERT, RoBERTa, DeBERTa.

**Decoder-Only** — causal attention; each token attends only to past tokens:

**What it's for**: generation tasks — text completion, instruction following, agents.

Examples: GPT-2/3/4, LLaMA, Mistral, Claude.

**Encoder-Decoder** — encoder uses bidirectional attention; decoder uses causal self-attention + cross-attention to encoder output:

**What it's for**: transformation tasks where input and output are different sequences — translation, summarization, speech-to-text.

Examples: T5, BART, Whisper, original Transformer.

---

## KV Cache

> Full mechanics of KV caching, MQA, and GQA are in [attention.md](02-attention.md).

During autoregressive generation, keys and values for past tokens do not change — cache them after the first computation. Cache size grows linearly:

$$\text{cache size} = 2 \times n_\text{layers} \times n_\text{heads} \times d_k \times n_\text{ctx}$$

**MQA**: all query heads share one K/V projection — cache divided by $n_\text{heads}$.
**GQA**: groups of heads share K/V — used in LLaMA 3. Intermediate quality/memory tradeoff between MHA and MQA.

---

## Scaling Laws

**The problem**: given a compute budget, how should you split it between model size and training data? Should you train a very large model on less data, or a smaller model on more data?

**The core insight (Chinchilla, 2022)**: before Chinchilla, most large models were trained on too little data for their size. The optimal compute split roughly requires training tokens $D \approx 20N$ for a model with $N$ parameters.

$$L(N, D) \approx \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$$

Loss decreases as a power law in both model size and data. They contribute approximately equally to loss reduction per unit of compute.

**What breaks**: the Chinchilla compute-optimal ratio applies to training compute. At inference, smaller models with longer training are more efficient to serve. A 7B model trained on 1T tokens may outperform a 70B model trained on 100B tokens and serve $10\times$ faster. The tradeoffs between training cost and serving cost must be considered together.

---

## Attention Complexity

| Method | Time | Memory | Notes |
| :--- | :--- | :--- | :--- |
| **Standard Attention** | $O(n^2 d)$ | $O(n^2)$ | Baseline |
| **Flash Attention** | $O(n^2 d)$ | $O(n)$ | IO-aware tiling, same output, 2–4× faster |
| **Sparse Attention** | $O(n\sqrt{n}\cdot d)$ | $O(n\sqrt{n})$ | Longformer, BigBird |
| **Linear Attention** | $O(nd^2)$ | $O(d^2)$ | Approximate, quality trade-off |

---

## Canonical Interview Q&As

**Q: Walk through the full forward pass of a transformer encoder layer, specifying what each component does.**  
A: Input tokens are embedded + positional encoding added → shape [batch, seq_len, d_model]. (1) Layer norm applied to input. (2) Multi-head attention: project input to Q, K, V for each head (W_Q, W_K, W_V projections); compute scaled dot-product attention: softmax(QKᵀ/√d_k)V for each head; concatenate and project with W_O — output shape same as input. (3) Residual connection: input + attention_output. (4) Layer norm again. (5) FFN: two linear layers with activation: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2, where hidden dim is typically 4×d_model. (6) Residual: previous + FFN output. The residual connections are critical — they allow gradients to flow directly to earlier layers, enabling deep transformers to train. Pre-norm (norm before sublayer, as above) is more stable than post-norm for large models.

**Q: Why do transformers use positional encoding, and what are the trade-offs between learned and sinusoidal encodings?**  
A: Self-attention is permutation-equivariant — without positional encoding, "cat ate fish" and "fish ate cat" produce identical representations. Positional encoding injects order information. Sinusoidal encoding: PE(pos, 2i) = sin(pos/10000^{2i/d_model}), PE(pos, 2i+1) = cos(...). Advantages: fixed, no parameters, can generalize to positions beyond training length through the periodic structure. Learned absolute encodings (e.g., original BERT): each position has a learned embedding vector. Better performance at training lengths but no extrapolation. In modern LLMs, both are replaced by relative encodings (RoPE, ALiBi) because they handle long contexts better. RoPE encodes position by rotating Q, K vectors, making relative positions implicit in the dot product — length generalization is better, though not perfect without fine-tuning on longer sequences.

**Q: What architectural change makes LLaMA/Mistral different from the original transformer, and why does it matter?**  
A: Several key changes: (1) RMSNorm instead of LayerNorm — removes mean subtraction, ~10% faster with no quality loss; (2) SwiGLU activation instead of ReLU in FFN — SwiGLU(x, W, V, W2) = (xW ⊙ SiLU(xV))W2; requires 3 weight matrices instead of 2, so FFN hidden dim is scaled down to keep params constant, but empirically converges 15-20% faster; (3) RoPE instead of absolute position encodings — handles long contexts better; (4) GQA instead of MHA — 4-8 KV heads instead of 32-64, reduces KV cache size by 8-16× which is the primary serving bottleneck; (5) Pre-normalization throughout. Together these changes make models faster to train (SwiGLU, RMSNorm), much cheaper to serve at long contexts (GQA), and better at long-context tasks (RoPE). At interview level: GQA's KV cache reduction is the most practically impactful change for production systems.

---

## Flash Attention — I/O-Aware Attention

### Why Standard Attention Is Memory-Bound

Standard attention materializes the full N×N attention matrix in GPU HBM (High Bandwidth Memory). For N=8192 tokens, BF16, per batch element: 8192² × 2 ≈ 134 MB per layer. The bottleneck isn't FLOPs — it's **memory bandwidth**. GPU SMs sit idle waiting for HBM reads/writes.

Flash Attention (Dao et al., 2022) observes that GPU on-chip SRAM (~19 TB/s) is ~6× faster than HBM (~3.35 TB/s). The solution: **tile attention computation to stay in SRAM**, never materializing the full N×N matrix.

**Online softmax** makes this possible: using a running max m and running sum ℓ updated per tile, the algorithm produces numerically identical output to full softmax — tile by tile, without the full matrix.

**Memory complexity:** O(N²) → **O(N)**.  
**Training:** Recomputes attention in backward pass instead of storing it (saves ~4× memory at ~1.25× FLOP cost).

| Version | Speedup | Key improvement |
|---|---|---|
| FA-1 (2022) | 2–4× vs standard | Tiling + online softmax |
| FA-2 (2023) | 2× over FA-1 | Better warp-level parallelism, reduced overhead |
| FA-3 (2024) | ~2× over FA-2 | FP8 support, overlapped GEMM+softmax for H100 |

```python
import torch
import torch.nn.functional as F

# PyTorch 2.0+ dispatches to Flash Attention automatically on CUDA
def attention(q, k, v, is_causal=True):
    """q, k, v: (batch, heads, seq_len, head_dim)"""
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

print(torch.backends.cuda.flash_sdp_enabled())  # True if Flash Attention available
```

---

## RoPE — Rotary Position Embedding

**Problem with absolute encodings:** attention score q_m · k_n depends on absolute positions m, n — not relative distance m−n. Models fail to extrapolate beyond training length.

**RoPE's insight:** Encode position as a rotation so that q_m · k_n depends only on m−n.

For each dimension pair (x_2i, x_2i+1), rotate by angle m·θ_i where θ_i = 1/10000^(2i/d):

The dot product identity (R_m q)^T(R_n k) = q^T R_{m−n} k guarantees relative-only dependence. Low-frequency dimension pairs respond to long-range context; high-frequency to local. This is why RoPE generalizes better than learned positions.

**Long-context extension methods:**

| Method | Approach | Notes |
|---|---|---|
| NTK scaling | Scale base: 10000 → 10000 × λ^(d/(d-2)) | Simple; often no fine-tuning needed |
| YaRN | Non-uniform per-dimension scaling | High-freq unchanged; low-freq interpolated |
| LongRoPE (2024) | Search optimal per-dim scale factors | Used in Phi-3-mini-128K |

---

## ALiBi — Attention with Linear Biases

No positional encoding. Subtract a linear penalty: score_ij = q_i·k_j − m_h·|i−j| where m_h is a per-head constant. Zero parameters. Strong extrapolation (train 1K → inference 2K+ without fine-tuning). Hardcoded local bias limits long-range tasks. Used in MPT, BLOOM.

---

## Architecture Comparison Table

| Architecture | Position | Normalization | FFN Activation | Attention | Notes |
|---|---|---|---|---|---|
| Original Transformer (2017) | Sinusoidal | Post-LayerNorm | ReLU, 4×d | MHA | Machine translation |
| BERT (2019) | Learned | Post-LayerNorm | GELU, 4×d | MHA | Bidirectional |
| GPT-2 (2019) | Learned | Pre-LayerNorm | GELU, 4×d | MHA | Causal generation |
| T5 (2020) | Relative bias | Pre-LayerNorm | ReLU, 4×d | MHA | Encoder-decoder |
| PaLM (2022) | RoPE | Pre-LayerNorm | SwiGLU | MHA (parallel) | Parallel FFN+Attn |
| LLaMA 2 (2023) | RoPE | Pre-RMSNorm | SwiGLU, 8/3×d | GQA | Modern baseline |
| Mistral 7B (2023) | RoPE | Pre-RMSNorm | SwiGLU | GQA + SWA | Sliding window |
| LLaMA 3.x (2024) | RoPE | Pre-RMSNorm | SwiGLU | GQA (8 KV heads) | Llama3/3.1/3.3 |
| Gemma 2 (2024) | RoPE | Pre+Post-RMSNorm | GEGLU | GQA+local+global | Alternating attn |
| Llama 4 (2025) | RoPE | Pre-RMSNorm | SwiGLU | GQA (MoE FFN) | MoE layers |
| Qwen3 (2025) | RoPE | Pre-RMSNorm | SwiGLU | GQA | Dense + MoE variants |

**Takeaways:**
1. **RoPE + Pre-RMSNorm + SwiGLU + GQA** is the 2024-2025 default for frontier open models
2. The biggest serving optimization: MHA → GQA (8-32× KV cache reduction per sequence)
3. MoE FFN (DeepSeek, Llama 4) enables trillion-parameter models at 1/10th the FLOPs
4. Flash Attention is now baseline infrastructure — every serious training framework uses it

---

## Where to Next

- **Attention mechanics, KV cache, MQA, GQA** → [attention.md](02-attention.md)
- **Full LLM architecture from first principles** → [05-llms/03-architecture-deep-dive.md](../10-llms/03-architecture-deep-dive.md)
- **Production serving: vLLM, PagedAttention** → [llm-serving.md](../12-systems-and-scale/03-llm-serving.md)
