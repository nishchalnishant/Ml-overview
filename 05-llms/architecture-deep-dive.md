---
module: Llms
topic: Architecture Deep Dive
subtopic: ""
status: unread
tags: [llms, ml, architecture-deep-dive]
---
# LLM Architecture: From First Principles

---

## 1. Self-Attention: Why It Exists

**The problem**: An RNN processes tokens one at a time, left to right, compressing the entire history into a fixed-size hidden state. When you're at token 1,000, the hidden state is a lossy summary of everything that came before it. Information from token 1 has been through 999 compression steps — most of it is gone. Training is also fundamentally serial: step $t$ depends on step $t-1$, so you can't parallelize across sequence length.

**The core insight**: Instead of compressing history into a single vector, let every token look directly at every other token. The model learns *which* token-to-token relationships matter. No sequential bottleneck, no forgetting.

**The mechanics**: Given a sequence of token embeddings packed into a matrix $X \in \mathbb{R}^{T \times d}$, project into three matrices:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

$Q$ (query): what this token is looking for. $K$ (key): what this token offers to others. $V$ (value): what this token actually passes along if attended to.

Compute attention scores and outputs:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The $\sqrt{d_k}$ scale factor prevents the dot products from growing large enough to push the softmax into saturation (near-zero gradients).

**What breaks**: full attention is $O(T^2)$ in memory and compute. For $T = 8192$ tokens and FP16, the attention matrix alone is $8192^2 \times 2 \approx 128$ MB per layer per batch element. This is the wall that all subsequent optimizations are trying to climb over.

---

## 2. Multi-Head Attention: Why One Set of Q/K/V Isn't Enough

**The problem**: a single attention pattern is a single relational function. But text has multiple simultaneous relational structures — syntactic agreement, coreference, semantic similarity, positional proximity. A single head learns one weighted combination of all of these, which is a bottleneck.

**The core insight**: run $H$ independent attention operations with different learned projections, then concatenate. Each head can specialize in a different type of relationship.

**The mechanics**: for each head $i$, project to lower-dimensional queries, keys, values:

$$\text{head}_i = \text{Attention}(QW_i^Q,\, KW_i^K,\, VW_i^V)$$

Concatenate and project back:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)W^O$$

With $H=8$ heads and $d=512$: each head operates in dimension 64, total compute is unchanged from a single $d=512$ attention.

**What breaks**: with standard multi-head attention (MHA), every head has its own $K$ and $V$ projections. At inference time, these are cached — see the KV cache problem below.

---

## 3. Grouped Query Attention (GQA): Why the KV Cache Becomes a Crisis

**The problem**: during autoregressive generation, you compute Q, K, V for the new token and need to attend back to all previous tokens. To avoid recomputing, you cache the K and V tensors for all past tokens. With MHA, each head has its own K and V:

$$\text{KV cache per token} = 2 \times H \times d_h \times \text{bytes per element}$$

For LLaMA 2 70B (64 heads, 128 head dim, BF16): $2 \times 64 \times 128 \times 2 = 32$ KB per token. At 4096 tokens: 128 MB per sequence. At a batch of 32 sequences: 4 GB, just for the KV cache. This destroys throughput.

**The core insight**: query heads need to be distinct — different heads are attending to different things. But do the keys and values need to be equally distinct? Empirically, no: a small number of KV heads, shared across groups of query heads, loses almost no quality while dramatically cutting cache size.

**The mechanics**: divide $H$ query heads into $G$ groups. Each group shares one K head and one V head:

$$\text{group}(i) = \left\lfloor i \cdot G / H \right\rfloor \quad \text{for query head } i$$

Memory reduction: $H/G \times$ versus MHA. With $G = 8$ and $H = 64$: 8× smaller KV cache.

| Variant | K/V heads | KV cache per token | Quality |
| :--- | :--- | :--- | :--- |
| MHA | $H$ | $2Hd_h$ | Highest |
| GQA | $G \ll H$ | $2Gd_h$ | Near-MHA |
| MQA | 1 | $2d_h$ | Noticeably lower |

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
        self.n_rep = n_heads // n_kv_heads
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

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        scale = self.d_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))
```

**What breaks**: GQA with very few KV groups ($G = 1$, i.e., MQA) degrades quality noticeably on tasks requiring fine-grained retrieval. The quality cliff is real below about $G = 4$.

---

## 4. Positional Encodings: Why the Model Doesn't Know Word Order

**The problem**: the self-attention mechanism as described treats the input as a *set*, not a sequence. The attention score between token $i$ and token $j$ is the same regardless of whether they are adjacent or 500 positions apart. A model without position information cannot distinguish "the dog bit the man" from "the man bit the dog."

**The original solution and why it fails at scale**: the original Transformer used sinusoidal absolute encodings — a fixed function of position is added to each token embedding before any attention. This works, but the model never sees position 5000 during training on 4096-length sequences. At inference, positions beyond the training horizon are out of distribution and performance collapses.

**The core insight for RoPE**: instead of encoding position as an additive offset to the embedding, encode *relative* position directly in the attention score. If the dot product $q_m \cdot k_n$ is a function only of the token features *and* of the offset $(m - n)$, then the model learns relationships like "4 tokens before me" rather than "at absolute position 104."

**The mechanics**: rotate each query and key vector by an angle proportional to its position. For a 2D pair at position $m$ with frequency $\theta_i$:

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

where $\theta_i = 10000^{-2i/d}$. The rotation is applied to both $Q$ and $K$ before computing attention scores. The dot product of the rotated vectors depends only on the *difference* in positions, not the absolute values.

```python
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

def precompute_rope_freqs(d_head: int, max_seq: int, base: float = 10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_seq)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)
```

**What breaks — extrapolation**: RoPE handles positions it has seen during training with no problem. But at positions beyond the training length, the rotation angles are out of distribution and attention scores become unreliable. Extending context requires either fine-tuning on longer sequences or remapping the position indices.

| Extension method | Approach | Effective range |
| :--- | :--- | :--- |
| Linear scaling | $m \to m/s$ (stretch positions) | Up to ~2× before degrading |
| YaRN | Non-uniform scaling + temperature fix | 16×–32× |
| NTK-aware | Scale base frequency: $\theta \to \theta \cdot s^{d/(d-2)}$ | Smoother interpolation |
| LongRoPE | Per-dimension scaling via search | Claimed 2M tokens |

---

## 5. Layer Normalization: Keeping Activations Sane Through 96 Layers

**The problem**: during training, the distribution of activations entering each layer shifts continuously as upstream weights update. A layer trained when its inputs had mean 0.1 and std 1.0 suddenly receives inputs with mean 2.3 and std 5.0. The effective learning rate, gradient magnitudes, and even the meaning of the weights have changed — the layer is partly untrained for its new input regime. This is called internal covariate shift.

**The core insight**: normalize the activations at each layer so that the distribution each layer sees is stable, regardless of what upstream layers are doing.

**Why LayerNorm instead of BatchNorm for transformers**: BatchNorm normalizes across the batch dimension using batch statistics. For variable-length sequences or small batch sizes (both common in LLM training), the batch statistics are noisy or meaningless. LayerNorm normalizes across the feature dimension for each token independently — no batch dependency, no sequence-length dependency.

**The mechanics**: for each token's embedding vector $x$ of dimension $d$: compute mean $\mu$ and std $\sigma$ over $d$, then normalize and rescale:

$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

where $\gamma$ and $\beta$ are learned per-dimension scale and shift. RMSNorm (used in LLaMA) drops the centering step — only scales by the root mean square:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

Cheaper to compute, empirically equivalent or better on LLM tasks.

**What breaks — pre-LN vs post-LN placement**: in the original Transformer, LayerNorm came *after* the residual connection (post-LN). In deep networks (>24 layers), this causes gradient vanishing: the residual stream grows large relative to the normalized sublayer output, and gradients through the sublayer become negligible. Modern LLMs use pre-LN — normalize before the sublayer — which keeps gradients flowing. The tradeoff is a slightly different effective learning rate that requires re-tuning.

---

## 6. Flash Attention: The Memory Wall Hidden Inside O(T²)

**The problem**: standard attention materializes the full $T \times T$ attention score matrix in GPU memory (HBM). At $T = 8192$ and FP16: $8192^2 \times 2 = 128$ MB per layer per batch element. On an A100 with 80 GB HBM, a 32-layer model at batch size 4 uses $128 \times 32 \times 4 = 16$ GB just for attention matrices — and every read/write of this matrix to HBM is slow. Standard attention makes $O(T^2)$ HBM reads/writes. The GPU's arithmetic units sit idle waiting for memory.

**The core insight**: the $T \times T$ matrix is never *used* as a whole — it's immediately multiplied into $V$ to produce the $T \times d$ output. You can tile the computation into blocks that fit in SRAM (fast on-chip memory), compute partial softmax results incrementally, and accumulate the output without the large HBM round-trips.

**The mechanics — Flash Attention (Dao et al. 2022)**:
1. Tile $Q$, $K$, $V$ into blocks sized to fit in SRAM
2. For each block of $Q$, loop over blocks of $K$ and $V$
3. Maintain running statistics for the log-sum-exp trick to compute the correct softmax across tiles
4. Accumulate the output $O$ without ever writing the $T \times T$ matrix

Memory: $O(T)$ instead of $O(T^2)$. HBM accesses: $O(T^2/M)$ where $M$ is SRAM size, versus $O(T^2)$ for standard attention. Throughput on A100: 2–4× faster than standard attention, ~73% theoretical FLOP utilization vs ~25%.

```python
# PyTorch 2.0+ includes Flash Attention via scaled_dot_product_attention
import torch.nn.functional as F

output = F.scaled_dot_product_attention(
    query,   # (B, H, T, d_head)
    key,
    value,
    is_causal=True  # enables causal masking without materializing the mask
)
```

**What breaks**: Flash Attention trades FLOP count for memory efficiency — it recomputes some partial results during the backward pass rather than storing them. This means the backward pass uses slightly more FLOPs than naive attention. For most workloads this is irrelevant since memory was the bottleneck.

---

## 7. Sliding Window Attention: Constant Memory for Arbitrary Contexts

**The problem**: even with Flash Attention, the memory and compute for a full attention over $T$ tokens grow with $T$. For very long contexts — 100K or 1M tokens — full attention is intractable.

**The core insight**: most tokens don't need to attend to all of history. Local context (the last few thousand tokens) is usually sufficient for generation, and global information can percolate through layers. Limit each token's attention to a window of $W$ recent tokens.

**The mechanics**:

$$\text{score}(i, j) = q_i \cdot k_j^T / \sqrt{d_k}, \quad j \in [\max(0,\, i-W),\, i]$$

The KV cache is now constant at $W$ tokens per layer rather than growing with sequence length. Stacking $L$ layers gives an effective receptive field of $L \times W$: information from token $i - kW$ can reach the current token through $k$ layers of local windows.

For Mistral 7B: $W = 4096$, $L = 32$ → effective receptive field of ~131K tokens.

**What breaks**: the multi-layer receptive field argument assumes information propagates reliably. In practice, one-shot references — a fact mentioned once at the beginning of a very long document — can be lost because they fall out of every layer's window before influencing the final output. Full attention on selected tokens (Longformer, BigBird) or retrieval-augmented approaches address this.

---

## 8. Mixture of Experts (MoE): Scale Without Proportional Compute Cost

**The problem**: doubling a dense model's parameter count doubles both training compute and inference compute. But a large fraction of parameters may be irrelevant to any given token. A 70B model processing "def fibonacci(n):" activates all 70B parameters, even the ones that encode French grammar.

**The core insight**: replace each dense FFN layer with a collection of $N$ expert FFNs, and route each token to only $k$ of them. Total parameters scale with $N$, but active parameters per token scale with $k$. You get a large model's capacity at a fraction of the compute.

**The mechanics**: a learned router selects the top-$k$ experts per token:

$$G(x) = \text{TopK}(\text{softmax}(W_g x),\, k)$$
$$\text{MoE}(x) = \sum_{i \in \text{TopK}} G_i(x) \cdot E_i(x)$$

For Mixtral 8×7B: $N = 8$ experts, $k = 2$. Each token activates 2 of 8 experts: 12.9B active parameters out of 46.7B total. Training compute matches a ~12B dense model; capacity matches ~47B.

**What breaks — expert collapse**: without regularization, the router quickly converges to routing everything through one or two experts. The other experts receive no gradient and stay randomly initialized. Fix with a load-balancing auxiliary loss:

$$\mathcal{L}_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the mean routing probability for expert $i$. This loss penalizes skew.

**What else breaks**: all experts must fit in memory simultaneously — MoE saves compute, not memory. For distributed inference, different experts may live on different devices, requiring all-to-all communication for routing. Communication overhead can negate compute savings at small batch sizes.

```python
class MoELayer(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int, ffn_dim: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, ffn_dim), nn.SiLU(), nn.Linear(ffn_dim, d_model))
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        probs = F.softmax(self.router(x_flat), dim=-1)
        top_k_probs, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for i in range(self.n_experts):
            mask = (top_k_idx == i).any(dim=-1)
            if mask.any():
                expert_out = self.experts[i](x_flat[mask])
                weight = top_k_probs[mask][top_k_idx[mask] == i]
                output[mask] += weight.unsqueeze(-1) * expert_out

        f_i = (top_k_idx.view(-1) == torch.arange(self.n_experts).unsqueeze(1)).float().mean(dim=1)
        P_i = probs.mean(dim=0)
        balance_loss = self.n_experts * (f_i * P_i).sum()
        return output.view(B, T, D), balance_loss
```

---

## 9. SwiGLU: Why the FFN Activation Function Matters

**The problem**: the FFN in a transformer block is $\text{FFN}(x) = \text{GELU}(xW_1)W_2$. GELU is a smooth approximation to ReLU that helps gradient flow. But empirically, it consistently underperforms on language modeling compared to a gated variant.

**The core insight**: gating mechanisms — where one branch controls the magnitude of another — give the network a multiplicative interaction that a purely additive function cannot express. This acts like a learned, input-dependent activation function.

**The mechanics**:

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2) \cdot W_3$$
$$\text{Swish}(x) = x \cdot \sigma(x)$$

Three weight matrices instead of two. To keep total parameters equal to a standard FFN, the intermediate dimension is reduced from $4d$ to $\frac{8d}{3}$ (rounded to a multiple of 64). The gating branch ($xW_2$) acts as a filter on the activated branch ($\text{Swish}(xW_1)$).

**What breaks**: the three-matrix structure can't be expressed as a single linear layer followed by an activation. This slightly complicates efficient kernel fusion but is well-supported in modern deep learning frameworks. Used by LLaMA, PaLM, and Gemma.

---

## 10. The KV Cache and PagedAttention: Memory Fragmentation at Scale

**The problem**: the KV cache approach of storing past keys and values is sound in principle but disastrous in practice for a serving system. Every new request needs its own KV cache allocation. If the model can handle 4096-token sequences, you allocate 4096 tokens of KV cache upfront — even if the request only uses 200 tokens. For LLaMA 70B: ~1.3 GB of KV cache per fully allocated sequence. Pre-allocating for peak length wastes most of the memory on every request.

**The core insight**: borrow OS virtual memory. Don't allocate a contiguous block per sequence — maintain a page table that maps logical token positions to physical memory blocks. Allocate physical blocks on demand as the sequence grows.

**The mechanics — PagedAttention (vLLM)**: the KV cache is divided into fixed-size blocks of $B$ tokens each. A page table maps each sequence's logical token indices to physical block IDs. When attention is computed, the system looks up physical block IDs and gathers the relevant keys and values.

```
Logical KV (sequence view):   token 0, token 1, ..., token T-1
                                  ↓        ↓              ↓
Physical blocks (page table): block_2   block_7   ...  block_9
```

Benefits: no pre-allocation waste, no fragmentation between variable-length sequences, copy-on-write sharing of KV cache across beam search paths. Result: 24× higher throughput than HuggingFace Transformers in a memory-equivalent setup.

---

## 11. Architecture Comparison

| Model | Attention | Position | FFN | Context |
| :--- | :--- | :--- | :--- | :--- |
| GPT-3 | MHA (96h) | Learned absolute | Dense GELU | 4,096 |
| LLaMA 2 7B | GQA (32h, 4kv) | RoPE | SwiGLU | 4,096 |
| LLaMA 3 70B | GQA (64h, 8kv) | RoPE | SwiGLU | 8,192 → 128k (fine-tuned) |
| Mistral 7B | GQA + SWA | RoPE | SwiGLU | 32,768 |
| Mixtral 8×7B | GQA + SWA | RoPE | MoE (8 experts, top-2) | 32,768 |
| Gemma 2 27B | GQA (16h) | RoPE | GeGLU | 8,192 |
