# Advanced LLM Architecture

The base Transformer from "Attention is All You Need" is now the floor, not the ceiling. Production frontier models add Grouped Query Attention, RoPE, SwiGLU, Mixture of Experts, and sliding window attention. This file covers each optimization with the math and the tradeoff.

---

## 1. Grouped Query Attention (GQA)

### The Problem: KV Cache Memory at Scale

In standard Multi-Head Attention (MHA), every head has its own $K$ and $V$ projections. For a model with $H$ heads, the KV cache size is:

$$\text{KV cache per token} = 2 \times H \times d_h \times \text{bytes per element}$$

For LLaMA 2 70B (64 heads, 128 head dim, BF16): $2 \times 64 \times 128 \times 2 = 32$KB per token. At 4096 context: 128MB per sequence. This limits batch size and context length.

### Multi-Query Attention (MQA)

All query heads share a single K and V projection:
- KV cache reduction: $H \times$ smaller
- Quality loss: significant — each head cannot specialize K/V representations

### Grouped Query Attention (GQA)

Divide $H$ query heads into $G$ groups; each group shares one K and V head:

$$Q_i \in \mathbb{R}^{d_h}, \quad K_g \in \mathbb{R}^{d_h}, \quad V_g \in \mathbb{R}^{d_h}$$

where group $g = \lfloor i \cdot G / H \rfloor$ for query head $i$.

**Memory reduction:** $H/G \times$ vs MHA. With $G=8$ and $H=64$: 8× smaller KV cache with minimal quality loss.

| Variant | K/V heads | KV cache size | Quality |
| :--- | :--- | :--- | :--- |
| MHA | $H$ | $2Hd_h$ per token | Highest |
| GQA | $G \ll H$ | $2Gd_h$ per token | Near-MHA |
| MQA | 1 | $2d_h$ per token | Lower |

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # heads per KV group
        self.d_head = d_model // n_heads
        
        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(n_heads * self.d_head, d_model, bias=False)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        
        # Expand K and V to match number of query heads
        k = k.repeat_interleave(self.n_rep, dim=1)  # (B, n_heads, T, d_head)
        v = v.repeat_interleave(self.n_rep, dim=1)
        
        scale = self.d_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
```

---

## 2. Rotary Positional Embeddings (RoPE)

### Why Absolute Positional Embeddings Fail at Scale

Sinusoidal and learned absolute embeddings cannot generalize to sequence lengths longer than those seen in training. The model has no basis for predicting position 8192 if it was only trained on 4096.

### RoPE Mechanism

RoPE encodes position by rotating Q and K vectors. For a 2D pair at position $m$:

$$\text{RoPE}(q, m) = q \cdot e^{im\theta}$$

In practice, split the head dimension into pairs $(q_{2i}, q_{2i+1})$ and apply:

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

where $\theta_i = 10000^{-2i/d}$ (same base as sinusoidal PE).

**Key property:** the dot product $q_m \cdot k_n$ depends only on $q$, $k$, and the *relative* position $(m-n)$, not absolute positions. This makes attention naturally relative.

```python
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, heads, T, d_head), cos/sin: (T, d_head/2)"""
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    # Rotate: x1' = x1*cos - x2*sin, x2' = x1*sin + x2*cos
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

def precompute_rope_freqs(d_head: int, max_seq: int, base: float = 10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_seq)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)
```

### RoPE Extensions for Long Context

| Method | Approach | Context extension |
| :--- | :--- | :--- |
| **Linear scaling** | Scale position indices: $m \to m/s$ | Works up to ~2× before degrading |
| **YaRN** | Non-uniform scaling + attention temperature correction | 16×-32× extrapolation |
| **LongRoPE** | Evolutionary search for per-dimension scaling factors | 2M tokens (claimed) |
| **NTK-aware scaling** | Scale base frequency: $\theta_i = (10000 \cdot s^{d/(d-2)})^{-2i/d}$ | Smoother interpolation |

---

## 3. Mixture of Experts (MoE)

### Architecture

Replace each dense FFN with $N$ expert FFNs and a router:

$$\text{MoE}(x) = \sum_{i=1}^{N} G_i(x) \cdot E_i(x)$$

$$G(x) = \text{TopK}\left(\text{softmax}(W_g x), k\right)$$

Only $k$ of $N$ experts are activated per token. For Mixtral 8×7B: $N=8$, $k=2$, giving 12.9B active parameters out of 46.7B total.

**Compute savings:** proportional to $k/N$ for the FFN component. Attention remains dense.

### Load Balancing

Without regularization, the router learns to route all tokens to a few experts ("expert collapse"):

$$\mathcal{L}_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the average routing probability for expert $i$. This auxiliary loss encourages uniform load.

```python
class MoELayer(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int, ffn_dim: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ffn_dim),
                nn.SiLU(),
                nn.Linear(ffn_dim, d_model)
            ) for _ in range(n_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.view(-1, D)                        # (B*T, D)
        
        logits = self.router(x_flat)                   # (B*T, N)
        probs = F.softmax(logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(x_flat)
        for i in range(self.n_experts):
            # Tokens routed to expert i
            mask = (top_k_idx == i).any(dim=-1)
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = self.experts[i](expert_input)
                weight = top_k_probs[mask][top_k_idx[mask] == i]
                output[mask] += weight.unsqueeze(-1) * expert_output
        
        # Load balancing loss
        f_i = (top_k_idx.view(-1) == torch.arange(self.n_experts).unsqueeze(1)).float().mean(dim=1)
        P_i = probs.mean(dim=0)
        balance_loss = self.n_experts * (f_i * P_i).sum()
        
        return output.view(B, T, D), balance_loss
```

---

## 4. Flash Attention

### Standard Attention: The Memory Wall

Standard attention materializes the full $N \times N$ attention matrix:

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{N \times N}$$

For $N = 8192$ and FP16: $8192^2 \times 2 = 128$MB per attention layer per batch element. At 32 layers, 40GB — far exceeding GPU SRAM capacity.

Every read/write to GPU HBM (high-bandwidth memory) is expensive. Standard attention makes $O(N^2)$ HBM reads/writes.

### Flash Attention: IO-Aware Computation

Flash Attention (Dao et al. 2022) reorders computation to never materialize the full $N \times N$ matrix:

1. Tile $Q$, $K$, $V$ into blocks that fit in SRAM
2. Compute partial softmax statistics incrementally using the log-sum-exp trick
3. Accumulate output without HBM round-trips

**Memory:** $O(N)$ instead of $O(N^2)$. **IO:** $O(N^2/M)$ HBM accesses where $M$ is SRAM size, vs. $O(N^2)$ for standard attention.

**Throughput improvement:** 2-4× faster than standard attention on A100. Flash Attention 2 achieves ~73% theoretical FLOP utilization (vs ~25% standard).

```python
# PyTorch 2.0+ includes Flash Attention natively via scaled_dot_product_attention
import torch.nn.functional as F

# Automatically uses Flash Attention if available on the hardware
output = F.scaled_dot_product_attention(
    query,   # (B, H, T, d_head)
    key,
    value,
    attn_mask=causal_mask,
    dropout_p=0.0,
    is_causal=True  # enables causal masking without explicit mask
)
```

Flash Attention 3 (2024) adds asynchronous execution of GEMM and softmax stages, achieving >75% utilization on H100.

---

## 5. Sliding Window Attention (SWA)

Used in Mistral 7B. Each token attends only to the $W$ most recent tokens instead of the full context:

$$\text{score}(i, j) = q_i \cdot k_j^T / \sqrt{d_k}, \quad j \in [i-W, i]$$

**Per-layer receptive field:** $W$. But stacking $L$ layers gives effective receptive field of $L \times W$.

For Mistral: $W = 4096$, $L = 32$ → effective receptive field of 131k tokens.

**Memory:** KV cache is constant at $W$ tokens per layer rather than growing with sequence length.

**Tradeoff:** information from tokens beyond $W$ must be compressed through multiple layers, which works for information that appears repeatedly in context but loses one-shot references.

---

## 6. Architecture Comparison

| Model | Attention | Position | FFN | Context |
| :--- | :--- | :--- | :--- | :--- |
| GPT-3 | MHA (96h) | Learned absolute | Dense GELU | 4,096 |
| LLaMA 2 7B | GQA (32h, 4kv) | RoPE | SwiGLU | 4,096 |
| LLaMA 3 70B | GQA (64h, 8kv) | RoPE | SwiGLU | 8,192 → 128k fine-tuned |
| Mistral 7B | GQA + SWA | RoPE | SwiGLU | 32,768 |
| Mixtral 8×7B | GQA + SWA | RoPE | MoE (8 experts, top-2) | 32,768 |
| Gemma 2 27B | GQA (16h) | RoPE | GeGLU | 8,192 |

---

## 7. KV Cache and Paged Attention

### KV Cache Mechanics

During autoregressive generation, avoid recomputing all prior K and V:

```python
class KVCache:
    def __init__(self, max_batch: int, max_seq: int, n_layers: int, n_heads: int, d_head: int):
        self.k = torch.zeros(n_layers, max_batch, n_heads, max_seq, d_head)
        self.v = torch.zeros(n_layers, max_batch, n_heads, max_seq, d_head)
        self.length = 0
    
    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor):
        self.k[layer, :, :, self.length] = k_new
        self.v[layer, :, :, self.length] = v_new
    
    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[layer, :, :, :self.length+1], self.v[layer, :, :, :self.length+1]
```

**Memory cost for LLaMA 70B** (80 layers, 8 KV heads, 128 head dim, BF16):
$$2 \times 80 \times 8 \times 128 \times T \times 2 = 327.7 \text{ KB/token}$$

At 4096 tokens: ~1.3 GB per sequence. Limits concurrent requests significantly.

### PagedAttention (vLLM)

Inspired by OS virtual memory: KV cache is divided into fixed-size pages (blocks) of $B$ tokens each. A page table maps logical positions to physical blocks.

**Benefits:**
- No fragmentation from variable-length sequences
- Enables sharing KV cache across parallel beam search paths
- 24× higher throughput than HuggingFace Transformers at same memory

```
Logical KV cache (sequence view):    |  0  |  1  |  2  |  3  | ... | T-1 |
                                           ↓      ↓      ↓      ↓           ↓
Physical blocks (page table):          block_2  block_7  block_1  block_9  ...
```

---

## 8. Normalization and Activation

### Pre-LN vs Post-LN

**Post-LN** (original Transformer): LayerNorm after residual connection. Unstable at large scale — requires careful warmup and gradient clipping.

**Pre-LN** (modern LLMs): LayerNorm before sublayer. More stable training; used in GPT-3, LLaMA, Mistral.

**RMSNorm** (used in LLaMA): removes the centering step — only scales by RMS. Cheaper and equally effective:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

### SwiGLU FFN

Standard FFN: $\text{FFN}(x) = \text{GELU}(xW_1)W_2$

SwiGLU (LLaMA, PaLM): introduces a gating mechanism:

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$$

$$\text{Swish}(x) = x \cdot \sigma(x)$$

Three weight matrices instead of two ($W_1$, $W_2$, $W_3$ for the gate). To keep parameter count equal, the intermediate dimension is reduced from $4d$ to $\frac{8d}{3}$ (rounded to a multiple of 64).

Empirically outperforms RELU and GELU on language modeling loss at all scales tested.

> [!TIP]
> **Interview pattern:** For any architecture question, structure as: (1) what problem it solves, (2) the mechanism, (3) the tradeoff. For GQA: solves KV cache memory, mechanism = shared K/V across query groups, tradeoff = slight quality loss vs. MHA. For MoE: solves compute cost of large models, mechanism = sparse routing, tradeoff = load balancing complexity and all-expert memory footprint.
