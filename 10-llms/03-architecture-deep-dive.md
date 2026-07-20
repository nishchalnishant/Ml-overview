---
module: LLMs
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

> For full KV cache mechanics, MLA (Multi-head Latent Attention, DeepSeek), PagedAttention, prefix caching, and KV quantization, see [kv-cache-and-mqa-gqa.md](08-kv-cache-and-mqa-gqa.md).

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

> For advanced MoE topics — token dropping, expert capacity, DeepSeek fine-grained expert segmentation (160 experts, top-6 routing), expert parallelism, load balancing without auxiliary loss (DeepSeek V3), and analysis of Mixtral/Llama 4 routing behavior — see [moe-advanced-and-routing.md](09-moe-advanced-and-routing.md).

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

## 10.5 In-Context Learning (ICL)

**What it is**: In-context learning is the ability of LLMs to learn new tasks from examples provided in the prompt, without any gradient updates. You provide $k$ labeled examples as text, then a query, and the model generalizes from the examples to answer the query. No fine-tuning occurs — the model's weights are unchanged.

```
# Zero-shot: no examples
Translate to French: "the cat is on the mat"
→ Le chat est sur le tapis.

# Few-shot: k=3 examples in context
English: "the dog barked"  French: "le chien a aboyé"
English: "she opened the door"  French: "elle a ouvert la porte"
English: "we need more time"  French: "nous avons besoin de plus de temps"
English: "the cat is on the mat"  → Le chat est sur le tapis.
```

**Why ICL works — the induction heads hypothesis**: Mechanistic interpretability research identified *induction heads* — pairs of attention heads that together implement a "copy with offset" operation. Head A (the "previous token head") attends to the token immediately before the current token. Head B (the "induction head") attends to the position where Head A's output matches the current token, then copies the following token. This circuit can generalize: if the context shows "A→B" patterns, the model predicts B when it sees A again, even for novel patterns not seen during pretraining. ICL is essentially the induction heads circuit operating at scale over structured demonstrations.

**What determines ICL quality**:

1. **Label format matters, but less than expected**: flipping labels (marking positive sentiment examples as "negative") in few-shot examples degrades performance only ~10–20% on some tasks. The format and distribution of inputs matters more than the labels themselves.
2. **Example selection**: randomly sampled examples work, but retrieving examples similar to the test query (using a dense retriever over a demonstration pool) consistently outperforms random — ICL with kNN-retrieved demonstrations is one of the strongest prompt engineering techniques.
3. **Example ordering**: later examples in the prompt have more influence (recency bias). Put the most informative examples last.
4. **Number of examples**: performance improves with k, but with diminishing returns past ~8–16 examples (bounded by context length and the quadratic attention cost).

**ICL vs fine-tuning — when each wins**:

| Dimension | ICL | Fine-tuning |
|---|---|---|
| Setup cost | Zero (just edit prompt) | GPU hours |
| Knowledge source | Must be in context | Baked into weights |
| Generalization | Bounded by context window | Unlimited |
| Speed | Slower (long prompts) | Standard inference |
| Best for | Few-shot new tasks, rapid prototyping | Consistent behavior on well-defined task |
| Fails when | Task requires > context window | Task is poorly specified or data is noisy |

**Chain-of-thought ICL**: adding step-by-step reasoning examples in the few-shot demonstrations dramatically improves performance on multi-step reasoning tasks (math, code, logic). The reasoning chain in the example teaches the model *how* to think, not just what the answer should be. Zero-shot chain-of-thought ("Think step by step.") works because the model has seen this reasoning style in pretraining data. See [applications/01-agentic-workflows.md](../11-llm-applications/01-agentic-workflows.md) for deeper coverage of CoT variants.

**The meta-learning interpretation**: ICL works because LLM pretraining on diverse text implicitly trains a meta-learner. The model has seen countless (task description, example, solution) patterns across many domains. At inference, the few-shot prompt provides a new task description and examples; the model applies learned meta-learning patterns from pretraining to complete the query. This interpretation explains why larger models exhibit stronger ICL — they've seen more varied pattern types and generalize more robustly.

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

---

## 12. Major Model Families

*(niche — useful for context/trivia, but exact version-by-version specs and benchmark numbers are rarely quizzed directly; know the general architectural trends per family, not every number)*

### GPT Family (OpenAI)

**GPT-3 (2020)**: 175B parameters, 96 layers, 96 heads, 12,288 embedding dim. Trained on ~300B tokens (WebText2 + Books + Wikipedia). Standard MHA with learned absolute positional embeddings. Context: 2,048 tokens. Severely undertrained by Chinchilla standards (1.7 tokens/param vs. Chinchilla's 20). Demonstrated emergent few-shot learning at scale — no task-specific fine-tuning, just examples in the prompt.

**GPT-3.5 / InstructGPT (2022)**: 175B + RLHF alignment. The key result from the InstructGPT paper: a 1.3B model aligned with RLHF was preferred by human raters over a 175B unaligned model 71% of the time. Showed that alignment quality matters more than raw scale for real-world utility.

**GPT-4 (2023)**: Architecture not publicly disclosed; figures below are unconfirmed industry estimates, not official numbers. Widely rumored to be a Mixture of Experts (~16 experts, top-2 routing, ~1.8T total params, ~220B active params per forward pass). First OpenAI model with multimodal input (image + text). Context: initially 8,192 tokens, extended to 128K (GPT-4 Turbo). Evaluated on bar exam (90th percentile), GRE, AMC.

**GPT-4o (2024)**: "o" = omni. Native multimodal — processes text, audio, and images in a single model without separate encoder/decoder per modality. Latency: ~320ms audio response (vs. 2.8s for GPT-4V + speech pipeline). Text token stream approach for audio rather than spectrogram-to-spectrogram. Enables end-to-end latency improvements and richer voice interaction.

**o1 / o3 (2024–2025)**: "thinking" models — extended chain-of-thought reasoning before producing final answer. o1 improved from 13% to 83% on AIME 2024 math competition. Trained via RL on verifiable tasks (math proofs, code tests). o3 reached ~96% on AIME 2025. Key architectural insight: separating "thinking tokens" (not visible to user) from "output tokens" allows the model to reason at greater depth without inflating visible response length.

---

### LLaMA Family (Meta)

**LLaMA 1 (2023)**: Released as open-weights research model. Four sizes: 7B, 13B, 33B, 65B. Trained on ~1T–1.4T tokens (CommonCrawl, C4, GitHub, Books, Wikipedia, StackExchange). Key architectural changes from GPT-3: RoPE positional encodings, SwiGLU FFN activation, RMSNorm instead of LayerNorm. The 13B LLaMA 1 matched GPT-3 on most benchmarks — demonstrating that training efficiency (more tokens per parameter) dominates raw parameter count.

**LLaMA 2 (2023)**: Sizes: 7B, 13B, 34B, 70B. Extended context from 2K → 4K tokens. Introduced GQA in the 70B model (32 KV heads instead of 64 full heads). Trained on 2T tokens. Chat versions fine-tuned with SFT + RLHF. First Meta model with commercial license (not for use by companies with >700M monthly active users).

**LLaMA 3 (2024)**: Sizes: 8B, 70B (initially). Context 8K, extended to 128K via continued pre-training with RoPE scaling (YaRN-style). Trained on 15T tokens (~190 tokens/param for 70B — deliberately overtrained for inference efficiency). Vocabulary expanded to 128K tokens. GQA in all sizes (8 KV heads). MMLU: 82% for 70B — competitive with GPT-3.5.

**LLaMA 3.1 (2024)**: Added 405B parameter model. 128K context natively. 405B used tensor+pipeline parallelism across 16K GPUs during training. Three sizes: 8B, 70B, 405B. Synthetic data generation from LLaMA 3 70B used to improve instruction tuning quality.

**LLaMA 3.2 (2024)**: Multimodal variants (11B and 90B vision models). 1B and 3B edge/mobile models. Cross-attention for vision following Flamingo-style architecture.

**LLaMA 3.3 (2024)**: 70B instruction model with improved reasoning and coding. Comparable to LLaMA 3.1 405B on many benchmarks at 1/6th the inference cost — benefit of continued training improvements.

**LLaMA 4 (2025)**: Scout (17B active / 109B total MoE), Maverick (17B active / 400B total MoE), Behemoth (288B active / ~2T total MoE). Early fusion multimodal — vision and text share the same transformer, no separate cross-attention bridge. 10M token context window (Scout). Mixture of experts with 16 routed experts in Maverick. iRoPE: alternating attention layers without positional encoding (NoPE layers) for extended context.

---

### Mistral / Mixtral Family (Mistral AI)

**Mistral 7B (2023)**: Demonstrated that careful architecture and data choices can produce a 7B model that outperforms LLaMA 2 13B on most benchmarks. Two key architectural choices:
- **Sliding Window Attention (SWA)**: each token attends to only the nearest $w=4096$ tokens rather than all previous tokens, keeping attention memory O(w) instead of O(T). Effective receptive field still grows with depth (layer $k$ can reach tokens $k \times w$ positions back).
- **GQA**: 8 KV heads, 32 Q heads. Combined with SWA, enables very efficient long-context inference.

**Mixtral 8×7B (2023)**: First widely-adopted open-source Mixture of Experts LLM. Architecture: 8 experts per layer, top-2 routing. Total parameters: 46.7B, but only 12.9B (2 × 7B / 2 expert FF layers) are active per token — comparable compute cost to a 12.9B dense model with the knowledge capacity of 46.7B. Outperforms LLaMA 2 70B on most benchmarks at 1/5th the inference compute.

**Mixtral 8×22B (2024)**: Scaled-up MoE with 8 experts, 39B active out of 141B total. 64K context. Strong coding and math performance.

**Mistral Large / Nemo (2024)**: 123B dense model (Mistral Large 2). Mistral Nemo: 12B model with 128K context, joint development with NVIDIA.

---

### Gemini Family (Google DeepMind)

**Gemini 1.0 (2023)**: Three sizes: Ultra (largest), Pro (mid), Nano (edge). First Google model to outperform GPT-4 on a subset of benchmarks. Natively multimodal — trained jointly on text, image, audio, and video from scratch. Architecture uses specialized per-modality tokenizers (pixels → patches; audio → log-mel spectrograms; text → SentencePiece subwords) with a shared transformer backbone.

**Gemini 1.5 Pro (2024)**: Extended context to 1M tokens (10M in research preview). Uses Multi-Query Attention variants and sparse attention mechanisms to handle 1M context within memory constraints. "Needle in a haystack" evaluation: >99% recall at 1M context length. Key capability: analyze entire codebases, long videos, and books in a single context window. Architecture: MoE (not publicly disclosed but strongly implied by latency/cost profile).

**Gemini 2.0 (2024–2025)**: Flash (fast, low-cost), Pro (capability frontier), Ultra (research). 2.0 Flash Thinking: extended reasoning similar to o1. Multimodal output — can generate images and audio natively, not just process them. Sub-100ms TTFT at serving scale.

**Gemma (2024)**: Open-weights models from Google. Gemma 2B and 7B. Architecture based on Gemini principles: logit soft-capping (instead of scaling, cap logits with $\tanh$), GeGLU activations, alternating GQA/MHA. Gemma 2 (9B, 27B): added sliding window attention in alternating layers and knowledge distillation from a larger teacher model during training.

---

### Claude Family (Anthropic)

**Claude 1/2 (2022–2023)**: First public Anthropic models. Constitutional AI training — model critiques its own outputs against a set of principles, generates revised responses, and those revisions form the preference dataset for RLHF. Claude 2 extended context to 100K tokens — at the time the longest context of any production LLM. Claude 2.1: reduced hallucination via improved calibration training.

**Claude 3 (2024)**: Three-tier model family:
- **Haiku**: smallest, fastest. Designed for real-time applications, cost-efficient at scale.
- **Sonnet**: mid-size, balanced capability and speed. Most widely deployed.
- **Opus**: largest, highest capability. Scored 86.8% on MMLU, competitive with or ahead of GPT-4 on most published benchmarks at time of release.
All Claude 3 models are natively multimodal (text + image). Context: 200K tokens.

**Claude 3.5 Sonnet (2024)**: Outperformed Claude 3 Opus while being faster and cheaper — demonstrating post-training efficiency gains. Strong coding performance: top scores on SWE-Bench (real GitHub issues). Introduced artifacts (rich document creation in the UI).

**Claude 3.5 Haiku (2024)**: Fastest model in the Claude 3.5 generation, comparable to Claude 3 Opus on benchmarks at fraction of cost.

**Claude 3.7 Sonnet (2025)**: First Claude model with extended thinking — can "think" before responding using a chain-of-thought process hidden from the user. Hybrid reasoning: can operate in standard mode or extended thinking mode based on task complexity. Best-in-class on coding benchmarks (SWE-Bench Verified: 70.3%). Context: 200K tokens.

**Anthropic's training philosophy**: Constitutional AI and RLAIF (Reinforcement Learning from AI Feedback) instead of human annotators for preference data at scale. The "constitutional" principles include helpfulness, harmlessness, and honesty. Emphasis on interpretability research alongside capability development.

---

### DeepSeek Family (DeepSeek AI)

**DeepSeek V2 (2024)**: 236B total parameters, 21B active (MoE). Key innovation: Multi-head Latent Attention (MLA) — compresses KV cache by projecting K and V through a low-rank bottleneck, dramatically reducing KV cache memory (5–13× less than standard GQA). 128K context. Fine-grained expert segmentation: 160 experts per layer vs. Mixtral's 8, with top-6 routing — more fine-grained specialization per token.

**DeepSeek V3 (2024)**: 671B total, 37B active. Trained for $2.788M in compute on 14.8T tokens — remarkable cost efficiency. Multi-Token Prediction (MTP): auxiliary training heads predict multiple future tokens simultaneously, improving training signal density. Load balancing without auxiliary loss: sequence-level balance instead of token-level loss, reducing interference with main training objective.

**DeepSeek R1 (2025)**: Reasoning-focused model trained with Group Relative Policy Optimization (GRPO) — a PPO variant that removes the value network, using group statistics as baseline instead. "Aha moment" emergent behavior: model spontaneously learned to use extended reasoning chains (re-evaluation, backtracking) without explicit chain-of-thought supervision. Comparable to o1 on math/reasoning benchmarks, fully open-weights.

---

### Qwen Family (Alibaba)

**Qwen2.5 (2024)**: 0.5B to 72B. Strong multilingual performance, especially Chinese + English. 128K context. Key datasets: 18T tokens with high code and math proportion. Qwen2.5-Coder and Qwen2.5-Math specialized variants.

**Qwen3 (2025)**: Hybrid reasoning model with thinking and non-thinking modes. iRoPE: every 4th attention layer uses NoPE (no positional encoding) while others use RoPE — the NoPE layers learn to perform global cross-position comparisons without positional bias, while RoPE layers handle local context. 0.6B to 235B (MoE variant: 22B active / 235B total). 32K base context, extendable to 128K.

---

### Model Scale Reference

| Model | Release | Params (active) | Params (total) | Training tokens | Context |
|---|---|---|---|---|---|
| GPT-3 | 2020 | 175B | 175B | 300B | 4K |
| LLaMA 2 70B | 2023 | 70B | 70B | 2T | 4K |
| Mistral 7B | 2023 | 7B | 7B | ~1T | 32K |
| Mixtral 8×7B | 2023 | 12.9B | 46.7B | ~1T | 32K |
| LLaMA 3 70B | 2024 | 70B | 70B | 15T | 128K |
| Gemini 1.5 Pro | 2024 | ~100B est. | MoE (undisclosed) | Undisclosed | 1M |
| DeepSeek V3 | 2024 | 37B | 671B | 14.8T | 128K |
| LLaMA 3.1 405B | 2024 | 405B | 405B | 15T | 128K |
| Claude 3.7 Sonnet | 2025 | Undisclosed | Undisclosed | Undisclosed | 200K |
| DeepSeek R1 | 2025 | 37B | 671B | Undisclosed | 128K |
| LLaMA 4 Maverick | 2025 | 17B | 400B | Undisclosed | 1M |
| Qwen3-235B-A22B | 2025 | 22B | 235B | Undisclosed | 128K |
