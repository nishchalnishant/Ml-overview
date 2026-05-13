# Attention

Attention is the mechanism that let deep learning stop reading sequences like a forgetful intern and start understanding context more flexibly.

It is one of the biggest reasons modern NLP and LLMs look the way they do.

---

# 1. What Attention Does

Attention lets one token decide which other tokens matter more.

Instead of compressing everything into one hidden state, the model dynamically pulls the most relevant context when needed.

---

# 2. Query, Key, Value

- **Query ($Q$):** what this token is looking for
- **Key ($K$):** what other tokens advertise for matching
- **Value ($V$):** the information they provide if matched

Think of it like a search engine: $Q$ is the search query, $K$ are the document summaries, $V$ are the actual documents.

---

# 3. Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Why scale by $\sqrt{d_k}$?**

The dot product $QK^T$ grows with dimension $d_k$ (variance scales as $d_k$ for random vectors). Without scaling, large $d_k$ pushes softmax into near-zero gradient regions. Dividing by $\sqrt{d_k}$ stabilizes variance at 1.

**Dimensions:**
- $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$
- Attention weights: $\mathbb{R}^{n \times m}$ — for each query token, a weight over all key tokens
- Output: $\mathbb{R}^{n \times d_v}$

**Computational complexity:** $O(n^2 \cdot d)$ — quadratic in sequence length $n$. This is the main scaling bottleneck.

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights
```

---

# 4. Self-Attention vs Cross-Attention

## Self-Attention

$Q$, $K$, $V$ all come from the **same sequence**.

Used for:
- context building within a sequence
- encoder layers, decoder layers

## Cross-Attention

$Q$ comes from one sequence; $K$, $V$ come from another.

Used for:
- encoder-decoder setups (decoder attends to encoder output)
- multimodal (text attends to image patches)
- retrieval-conditioned generation

---

# 5. Causal (Masked) Self-Attention

In decoder-only models (GPT-style), each token must only attend to **past tokens** (autoregressive generation).

Achieved by masking future positions to $-\infty$ before softmax:

```python
# Causal mask: lower-triangular matrix
seq_len = 10
mask = torch.tril(torch.ones(seq_len, seq_len))
# Positions where mask==0 become -inf → softmax gives weight ≈ 0
```

---

# 6. Multi-Head Attention

Instead of one attention pattern, run $h$ attention heads in parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

Each head has its own projection matrices $W_i^Q \in \mathbb{R}^{d \times d_k}$, $W_i^K \in \mathbb{R}^{d \times d_k}$, $W_i^V \in \mathbb{R}^{d \times d_v}$, with $d_k = d_v = d / h$.

**Why multiple heads?** Different heads learn different types of relationships simultaneously:
- local structure vs long-range dependencies
- syntactic relations vs semantic similarity
- coreference vs topic coherence

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        B, T, D = Q.shape
        
        # Project and split into heads: (B, T, D) -> (B, h, T, d_k)
        Q = self.W_q(Q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
        
        # Recombine heads: (B, h, T, d_k) -> (B, T, D)
        out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)
```

---

# 7. Positional Encodings

Attention is **permutation invariant** — it has no inherent notion of order. Positional encodings inject sequence position.

## Sinusoidal (Original Transformer)

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Fixed, not learned. Generalizes to unseen lengths.

## Learned Positional Embeddings (BERT, GPT)

Learnable embedding table, one vector per position. Limited to training context length.

## RoPE (Rotary Position Embedding)

Used in LLaMA, GPT-NeoX, Mistral. Encodes relative position by rotating Q and K vectors:

$$\text{RoPE}(q, pos) = q \cdot e^{i \cdot pos \cdot \theta}$$

**Advantage:** naturally captures relative distances; can be extended beyond training length.

## ALiBi (Attention with Linear Biases)

Add a position-dependent bias to attention scores: $-m \cdot (i - j)$ where $m$ is a per-head slope. No extra parameters; strong length generalization.

---

# 8. KV Cache (Inference Optimization)

During autoregressive generation, previously computed $K$ and $V$ tensors can be cached.

Without cache: recompute full attention for all previous tokens at each step → $O(n^2)$ per token.

With KV cache: only compute $Q$, $K$, $V$ for the new token; reuse past $K$, $V$ → $O(n)$ per token.

**Memory cost:** each token requires $2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_k$ floats. At long contexts, KV cache becomes the memory bottleneck. Solutions: grouped-query attention (GQA), multi-query attention (MQA).

---

# 9. Flash Attention

Standard attention materializes the $n \times n$ attention matrix in memory — $O(n^2)$ memory.

Flash Attention (Dao et al., 2022) computes attention in tiles without materializing the full matrix:
- IO-aware: minimizes HBM reads/writes on GPU
- Same mathematical result as standard attention
- Memory: $O(n)$ instead of $O(n^2)$
- **2-4× faster** in practice; enabled training on longer contexts

Available as `F.scaled_dot_product_attention` in PyTorch 2.0+ with `enable_flash_sdp(True)`.

---

# 10. Complexity Summary

| Method | Time | Memory | Notes |
| :--- | :--- | :--- | :--- |
| **Standard Attention** | $O(n^2 d)$ | $O(n^2)$ | Baseline |
| **Flash Attention** | $O(n^2 d)$ | $O(n)$ | IO-efficient, same result |
| **Linear Attention** | $O(nd^2)$ | $O(d^2)$ | Approximate, quality trade-off |
| **Sparse Attention** | $O(n \sqrt{n} d)$ | $O(n \sqrt{n})$ | Longformer/BigBird |
