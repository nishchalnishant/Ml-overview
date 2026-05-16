# Context Window Extension

Techniques for scaling transformer context beyond native training length — from position encoding tricks to sparse attention, memory-efficient kernels, and architectural alternatives.

---

## 1. The Problem

Transformers compute attention over all pairs of tokens: for a sequence of length n, the attention matrix is n×n, giving **O(n²) time and memory complexity**. At n=128k, that matrix alone is ~65 billion entries.

**Where limits bite:**
- Long documents (legal, scientific, books)
- Multi-turn dialogue accumulated over many exchanges
- Code repositories where cross-file context matters
- RAG pipelines that want to stuff many retrieved chunks in-context

**Two separate bottlenecks:**

| Bottleneck | Cause | Consequence |
|---|---|---|
| Compute | O(n²) FLOPs in attention | Slow inference / training |
| Memory (HBM) | Full n×n attention matrix materialized | OOM at long context |
| Generalization | Model trained up to length L, tested at L' > L | Positional OOD, degraded output |

All three must be addressed independently. A model can have efficient attention but still fail at lengths beyond its training distribution.

---

## 2. Position Encoding Strategies

### 2.1 RoPE — Rotary Position Embedding

**Core idea:** Instead of adding a positional vector, *rotate* the query and key vectors in 2D subspaces by an angle proportional to position. The dot-product between Q and K then depends only on their *relative* position.

For position m, dimension pair (2i, 2i+1), the rotation matrix is:

```
R(m, θᵢ) = [[cos(mθᵢ), -sin(mθᵢ)],
             [sin(mθᵢ),  cos(mθᵢ)]]

where θᵢ = 10000^(-2i/d)
```

**Key properties:**
- Relative position appears naturally in Q·K: `(R(m)q)ᵀ(R(n)k) = qᵀR(n-m)k`
- **Long-term decay**: high-frequency dimensions (small θᵢ) oscillate fast and cancel for large |m-n|; low-frequency dimensions carry coarse structure. Attention score naturally decays with distance.
- No learned parameters — pure geometric construction
- Used in: LLaMA family, Mistral, GPT-NeoX, Falcon, Gemma

```python
import torch
import torch.nn as nn

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """Precompute complex exponentials for RoPE."""
    # Frequency for each dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # Position indices
    t = torch.arange(seq_len)
    # Outer product: (seq_len, dim/2)
    freqs = torch.outer(t, freqs)
    # Convert to complex form: cos + i*sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor):
    """Apply rotary embeddings to queries and keys.

    Args:
        xq: (batch, seq_len, n_heads, head_dim)
        xk: (batch, seq_len, n_kv_heads, head_dim)
        freqs_cis: (seq_len, head_dim/2) complex tensor
    """
    # View real tensor as complex: pairs (2i, 2i+1) become one complex number
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast freqs_cis over batch and heads: (1, seq_len, 1, head_dim/2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Element-wise complex multiplication = rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# Usage
dim = 128          # head_dim
seq_len = 4096
freqs_cis = precompute_freqs_cis(dim, seq_len)

batch, n_heads = 2, 8
xq = torch.randn(batch, seq_len, n_heads, dim)
xk = torch.randn(batch, seq_len, n_heads, dim)
xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis)
print(xq_rot.shape)  # (2, 4096, 8, 128)
```

### 2.2 ALiBi — Attention with Linear Biases

**Core idea:** Remove positional embeddings entirely. Instead, subtract a linear bias proportional to distance directly in the attention score:

```
Attention(Q, K, V) = softmax(QKᵀ/√d  -  m · |i - j|) · V
```

Where `m` is a head-specific slope (geometric sequence: 2^(-8/n_heads), 2^(-16/n_heads), ...).

**Why it works for extrapolation:**
- The model sees biases during training; at test time with longer sequences, the same bias formula applies — no OOD position indices
- Steeper slopes (larger m) encode local attention; shallower slopes encode longer-range
- No parameters added

**Limitation:** ALiBi shows degraded performance at very long contexts compared to RoPE-based methods, and the linear decay assumption doesn't always match what tasks need.

Used in: BLOOM, MPT, Baichuan

### 2.3 Position Interpolation

When extending a RoPE-trained model from length L to L', the naive approach applies positions 0..L'-1 — but the model was only trained on 0..L-1, so positions L..L'-1 are out of distribution.

**Linear interpolation:** Scale all position indices by L/L':

```
m' = m · (L / L')
```

This squeezes the new longer sequence into the trained position range. Requires brief fine-tuning (~1000 steps) to adapt.

**NTK-aware interpolation:** Linear interpolation uniformly compresses all frequency components, including high-frequency ones that the model uses for local structure. NTK-aware interpolation modifies the base θ instead:

```
θ'ᵢ = θᵢ · (L'/L)^(d/(d-2))
```

This leaves high-frequency dimensions mostly unchanged (local structure preserved) and compresses only low-frequency dimensions. Works with *zero* fine-tuning for moderate extension ratios.

---

## 3. RoPE Scaling Methods

### 3.1 Linear Scaling

Simplest approach: divide all position indices by a scale factor s = L'/L.

```python
# In model config or at inference time
def apply_linear_scaling(freqs_cis, scale_factor):
    """Scale down position angles to stay within training range."""
    # freqs_cis shape: (seq_len, head_dim/2)
    # Reconstructing angles, scaling, recomputing
    angles = torch.angle(freqs_cis)
    scaled_freqs_cis = torch.polar(torch.ones_like(angles), angles / scale_factor)
    return scaled_freqs_cis
```

Works without fine-tuning up to ~2x extension. Degrades at higher ratios because squished positions create confusion between nearby tokens.

### 3.2 Dynamic NTK Scaling

Apply NTK-aware base scaling dynamically at runtime, adjusting based on actual sequence length seen:

```python
def get_ntk_alpha(original_max_position: int, current_length: int, head_dim: int):
    """Compute NTK scaling alpha given current sequence length."""
    if current_length <= original_max_position:
        return 1.0
    ratio = current_length / original_max_position
    alpha = ratio ** (head_dim / (head_dim - 2))
    return alpha

def precompute_freqs_cis_ntk(dim: int, seq_len: int,
                               original_max_len: int = 4096,
                               theta: float = 10000.0):
    alpha = get_ntk_alpha(original_max_len, seq_len, dim)
    # Scale the base frequency
    theta_scaled = theta * alpha
    freqs = 1.0 / (theta_scaled ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)
```

### 3.3 YaRN — Yet Another RoPE extensioN

YaRN (Peng et al., 2023) combines three ideas:

1. **NTK-by-parts interpolation**: Different frequency bands get different treatment
   - High frequency (small wavelength): no interpolation, left unchanged
   - Low frequency (large wavelength): pure linear interpolation
   - Medium frequency: interpolate proportional to wavelength

2. **Attention temperature correction**: Longer contexts increase the entropy of attention distributions, making them more uniform. YaRN corrects with a scaling factor `√(1/t)` applied to the attention scores, where t is derived from the extension ratio.

3. **Fine-tuning on long sequences**: A small amount of continued pretraining on long documents (~400 steps) to consolidate the adjustments.

YaRN achieves strong perplexity at 128k context on LLaMA with minimal compute. Used in Mistral's long-context variants.

### 3.4 LongRoPE

LongRoPE (Ding et al., 2024) finds non-uniform rescaling factors per dimension via evolutionary search, then applies a two-stage extension:
1. Extend to 256k using found factors
2. Fine-tune at 256k, then recover short-context performance by using original RoPE for sequences ≤ original_max_len

Key insight: different RoPE dimensions have different sensitivity to length extension — one uniform scale factor is suboptimal.

---

## 4. Sparse Attention Patterns

Full attention is O(n²). Sparse attention restricts which tokens can attend to which, reducing to O(n·w) for window size w.

### 4.1 Longformer

**Three attention pattern types:**

| Pattern | Which tokens | Cost |
|---|---|---|
| Sliding window | Every token attends to ±w/2 neighbors | O(n·w) |
| Dilated window | Every token attends to ±w/2 at stride d | O(n·w) |
| Global tokens | Attend to/from ALL tokens | O(n·g) where g = #global |

Global tokens are placed on [CLS], [SEP], or task-specific tokens. They aggregate global information that the window attention cannot carry across long spans.

```python
from transformers import LongformerModel, LongformerTokenizer
import torch

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")

# Longformer can handle up to 4096 tokens
text = "Your very long document here..." * 100  # simulate long input
encoding = tokenizer(
    text,
    return_tensors="pt",
    max_length=4096,
    truncation=True,
)

input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

# Global attention mask: 0 = local window, 1 = global
# Put global attention on first token ([CLS])
global_attention_mask = torch.zeros_like(input_ids)
global_attention_mask[:, 0] = 1  # CLS attends globally

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        global_attention_mask=global_attention_mask,
    )

# outputs.last_hidden_state: (batch, seq_len, hidden_size)
cls_embedding = outputs.last_hidden_state[:, 0, :]
print(cls_embedding.shape)  # (1, 768)

# For classification tasks, use LongformerForSequenceClassification
from transformers import LongformerForSequenceClassification

clf_model = LongformerForSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096",
    num_labels=2,
)
logits = clf_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    global_attention_mask=global_attention_mask,
).logits
```

### 4.2 BigBird

BigBird (Zaheer et al., 2020) proves that sparse attention with three components is Turing-complete and can approximate full attention:

1. **Random attention**: each query attends to r randomly selected keys (long-range links)
2. **Window attention**: sliding window of size w (local structure)
3. **Global tokens**: g tokens attend to/from all (information hubs)

Total complexity: O(n · (r + w + g)) which is O(n) for fixed r, w, g.

**Theoretical result**: BigBird can simulate any function computable by full attention, given sufficient global tokens. Random edges create small-world graph properties — diameter O(log n) instead of O(n).

### 4.3 Streaming LLM

**StreamingLLM** (Xiao et al., 2023) addresses a different problem: *infinite-length* inference with a fixed KV cache budget, without re-encoding past context.

**Observation**: Attention sinks — the first few tokens (often [BOS]) accumulate disproportionate attention mass regardless of content. Removing them from the KV cache catastrophically degrades performance, but their *values* matter less than their *presence* as attention anchors.

**Solution:** Keep two sets of KV cache entries:
- **Sink tokens**: first 4 tokens (hardcoded attention sinks)
- **Recent window**: last W tokens (sliding window)

```
KV cache = [sink_tokens | recent_window]
           [    4       |      W        ]
```

When the window overflows, evict the oldest non-sink token. This allows indefinitely long generation with O(1) memory growth (bounded by 4 + W).

**Limitation:** The model cannot recall tokens that fell out of the window — it is not a memory mechanism, just a stable inference mechanism.

---

## 5. Memory-Efficient Attention

### 5.1 FlashAttention

**The memory bottleneck:** Standard attention materializes the full n×n matrix in HBM (GPU high-bandwidth memory) to compute softmax. At n=8192, that's 256MB per layer per head in float16.

**FlashAttention** (Dao et al., 2022) restructures the computation to avoid ever materializing the full matrix:

**Tiling algorithm:**
1. Divide Q, K, V into blocks that fit in SRAM (fast on-chip memory)
2. For each block of Q, iterate over all blocks of K, V
3. Maintain running statistics (max, sum) for numerically stable online softmax
4. Accumulate output block without writing intermediate attention matrix to HBM

```
Memory:  O(n)    instead of O(n²)
FLOPs:   same    (still O(n²) math, but done in SRAM — much faster)
Speedup: 2–4×    wall-clock on A100 for typical lengths
```

The key insight: HBM bandwidth is the bottleneck, not raw FLOPs. Keeping the n×n intermediate in SRAM avoids expensive HBM reads/writes.

**IO complexity:** FlashAttention performs O(n²/M) HBM accesses where M is SRAM size, versus O(n²) for standard attention.

### 5.2 FlashAttention-2

Improvements over v1:
- Better work partitioning across warps within a thread block (less synchronization)
- Separate forward/backward parallelization strategy
- Supports causal masking, multi-query attention (MQA), grouped-query attention (GQA)
- ~2× faster than FlashAttention-1 on A100

```python
# FlashAttention-2 via the flash_attn package
from flash_attn import flash_attn_func
import torch

batch, seq_len, n_heads, head_dim = 2, 8192, 32, 128
q = torch.randn(batch, seq_len, n_heads, head_dim, dtype=torch.float16, device="cuda")
k = torch.randn(batch, seq_len, n_heads, head_dim, dtype=torch.float16, device="cuda")
v = torch.randn(batch, seq_len, n_heads, head_dim, dtype=torch.float16, device="cuda")

# causal=True for autoregressive models
out = flash_attn_func(q, k, v, causal=True)
print(out.shape)  # (2, 8192, 32, 128)

# Compared to standard attention (for illustration):
# Standard would do: scores = (q @ k.transpose(-2,-1)) / sqrt(head_dim)
# This creates a (2, 32, 8192, 8192) tensor in HBM — ~4GB in float16
# FlashAttention never creates this tensor
```

### 5.3 Ring Attention

For sequences longer than a single GPU's memory allows, **Ring Attention** (Liu et al., 2023) distributes the sequence across devices:

- Each device holds a chunk of Q and a chunk of K, V
- Devices form a ring; K, V blocks rotate around the ring
- Each device computes its contribution to the output using the FlashAttention online softmax trick across all received K, V blocks

This allows sequence lengths proportional to `n_devices × per_device_memory`, achieving nearly linear scaling of context with GPU count. Compute-communication overlap hides most of the ring communication cost.

---

## 6. Retrieval-Augmented Approaches

Instead of fitting everything into the context window, retrieve only what is needed.

### 6.1 KNN-LM

**kNN-LM** (Khandelwal et al., 2019): At inference, maintain a datastore of (hidden_state, next_token) pairs from a large corpus. For each query, find the k nearest neighbors by hidden state distance, interpolate their next-token distributions with the model's prediction:

```
P_final = λ · P_kNN + (1 - λ) · P_LM
```

No retraining required — plugs into any frozen LM. Effective for domain adaptation (build a domain-specific datastore).

### 6.2 RETRO

**RETRO** (Borgeaud et al., 2022): Retrieval during *training*, not just inference. The model architecture includes cross-attention layers that attend to retrieved chunks:

- Input is split into chunks of 64 tokens
- For each chunk, retrieve k similar chunks from a 2 trillion token corpus via approximate nearest neighbor search
- Special RETRO cross-attention layers inject retrieved context
- At 7B parameters, matches performance of 25× larger models without retrieval

### 6.3 RAG as Alternative to Long Context

For tasks like "answer questions about a 500-page PDF," RAG is often preferable to long context even when long context is available:

| Dimension | Long Context | RAG |
|---|---|---|
| Latency | High (full attention over all tokens) | Lower (only retrieved chunks in context) |
| Cost | Scales with full doc length | Scales with retrieved chunk size |
| Recall | Perfect — nothing missed | Depends on retrieval quality |
| Lost-in-middle | Affected | Not affected |
| Dynamic updates | Re-encode entire doc | Update index incrementally |

**When to use long context:** When retrieval recall is critical, document structure matters (tables, cross-references), or the task requires synthesizing across many parts simultaneously.

---

## 7. Recurrent and State-Space Alternatives

These architectures achieve **O(n) training complexity** and **O(1) inference state** (fixed recurrent state), trading some expressiveness for efficiency.

### 7.1 Mamba (Selective State Space Models)

**S4 / Mamba** (Gu & Dao, 2023) frames sequence modeling as a continuous-time state space system:

```
h'(t) = A·h(t) + B·x(t)
y(t)  = C·h(t)
```

Discretized for sequences:
```
hₜ = Ā·hₜ₋₁ + B̄·xₜ
yₜ = C·hₜ
```

**Selective SSM (Mamba's key innovation):** Make B, C, and Δ (discretization step) *input-dependent*. This lets the model selectively remember or forget based on content — something fixed-parameter RNNs cannot do.

- Training: parallel scan (efficient on GPUs), O(n) time
- Inference: pure recurrence, O(1) state update per token
- Scales to very long sequences without attention
- Weakness: fundamentally harder to do "in-context lookup" — retrieving a specific earlier token requires it to be preserved in the compressed state

### 7.2 RWKV

RWKV reformulates attention as a linear recurrence, making it RNN-like at inference but trainable like a transformer. The key-value interaction is:

```
WKV_t = (Σ exp(w·(t-i) + kᵢ) · vᵢ) / (Σ exp(w·(t-i) + kᵢ))
```

Where w is a learnable per-channel decay. This is computable as a recurrence in O(1) state per step. Used in RWKV-4/5/6 with competitive performance to similarly-sized transformers.

### 7.3 RetNet

RetNet (Microsoft, 2023) replaces softmax attention with a retention mechanism that has three equivalent representations:
- **Parallel** (training): matrix form, O(n²) like attention but with a decay mask
- **Recurrent** (inference): O(1) state, constant memory
- **Chunkwise** (training efficiency): process chunks in parallel, propagate state between chunks

**Retention vs Attention:** Retention applies an explicit exponential decay γ^(m-n) instead of softmax normalization. This loses the ability to attend uniformly to distant tokens, but gains stable recurrent inference.

---

## 8. Context Compression

Instead of extending the window or retrieving, *compress* the context itself.

### 8.1 AutoCompressor

**AutoCompressor** (Chevalier et al., 2023): Fine-tune an LLM to recursively summarize its own context. The model processes a segment and produces "summary tokens" — soft tokens in embedding space that summarize that segment. These summary tokens are prepended to the next segment.

```
Segment 1 → [summary_tokens_1]
Segment 2 + summary_tokens_1 → [summary_tokens_12]
Segment 3 + summary_tokens_12 → answer
```

Compression ratio ~20x (1000 tokens → 50 summary tokens). Requires fine-tuning.

### 8.2 Gist Tokens

**Gist tokens** (Mu et al., 2023): Train a model to compress a long prompt (e.g., system prompt or instructions) into a small number of learned "gist" tokens. At inference, cache the gist token KV representations and reuse across many queries sharing the same prefix.

- Trained with a masking strategy: the model sees gist tokens instead of the full prompt during supervised fine-tuning
- Achieves ~26x compression of instruction prompts with minimal task performance degradation
- Directly reduces KV cache size for repeated prompt prefixes

### 8.3 LLMLingua

**LLMLingua** (Jiang et al., 2023): Token-level prompt compression using a small proxy LM to identify low-perplexity (redundant) tokens for removal.

**Algorithm:**
1. Use a small LM (e.g., LLaMA-7B) to compute perplexity of each token conditioned on the prompt so far
2. Tokens with low perplexity (highly predictable) are redundant — prune them
3. Apply a budget-aware allocation across prompt segments (instructions get higher budget than examples)

```python
# LLMLingua (conceptual usage — requires llmlingua package)
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="NousResearch/Llama-2-7b-hf",
    device_map="cuda",
)

long_prompt = "..." * 500  # very long prompt

compressed = compressor.compress_prompt(
    long_prompt,
    instruction="Answer the question.",
    question="What is the main conclusion?",
    ratio=0.5,           # target 50% compression
    rank_method="llmlingua",
)

print(f"Original tokens: {compressed['origin_tokens']}")
print(f"Compressed tokens: {compressed['compressed_tokens']}")
print(f"Ratio: {compressed['ratio']}")
print(compressed['compressed_prompt'])
```

**LongLLMLingua** extends this to emphasize tokens most relevant to the question (coarse-to-fine compression) — up to 5x compression with <5% performance drop on NLP benchmarks.

---

## 9. Evaluation

### 9.1 RULER Benchmark

**RULER** (Hsieh et al., 2024) is a synthetic benchmark specifically designed to measure long-context capability with configurable difficulty:

- **NIAH (Needle In A Haystack)**: Single/multi-hop fact retrieval from injected sentences buried in a long document
- **Variable tracking**: Follow a chain of variable assignments across many tokens
- **Aggregation**: Count occurrences of a word across a long document
- **QA**: Answer questions requiring synthesis across the full document

RULER revealed that models claiming 128k context windows often degrade sharply past 32k in practice.

### 9.2 Needle-in-a-Haystack Test

**NIAH test** directly measures whether a model can retrieve a specific fact injected at an arbitrary position in a long context:

```
"The magic word is PINEAPPLE."  ← injected at position p% through document
[... long filler text ...]
"What is the magic word?"
```

Results are plotted as a heatmap over (document length, injection depth). Most models show:
- High retrieval accuracy at the beginning and end (primacy/recency effects)
- Degraded retrieval in the middle
- Abrupt failure cliff at some length threshold

### 9.3 Lost-in-the-Middle Phenomenon

**Lost-in-the-Middle** (Liu et al., 2023): Even with sufficient context length, LLMs perform significantly worse when relevant information appears in the middle of the context versus the beginning or end.

Tested on multi-document QA: performance follows a U-shaped curve over position. The effect persists across GPT-3.5, GPT-4, Claude, and open-source models.

**Implications:**
- For RAG: put the most relevant retrieved chunks first or last, not in the middle
- For long-context fine-tuning: training examples where the answer is in the middle are underrepresented; explicit position augmentation helps
- Cause: attention patterns during pretraining create implicit primacy/recency bias that survives instruction tuning

---

## 10. Key Interview Points

**On the core tradeoff:**
> "Long context vs retrieval vs compression are three strategies for the same goal — getting relevant information to the model. Long context has perfect recall but quadratic cost; RAG has sublinear cost but imperfect retrieval; compression has bounded cost but lossy information. In practice, you combine them."

**On RoPE vs ALiBi:**
> "RoPE encodes relative position through geometry — dot products naturally reflect distance. ALiBi encodes it through explicit linear bias. ALiBi extrapolates more cleanly without fine-tuning but tends to plateau in quality; RoPE with NTK scaling or YaRN achieves better perplexity at very long contexts with light fine-tuning."

**On FlashAttention:**
> "FlashAttention doesn't reduce FLOPs — it's still O(n²) math. What it reduces is HBM memory traffic by keeping the attention computation in SRAM and using tiling. The result is 2–4× wall-clock speedup and O(n) memory instead of O(n²)."

**On sparse attention:**
> "Longformer and BigBird reduce complexity to O(n·w) through sliding window attention. The key design choice is global tokens — without them, information can't flow between distant parts of the document. StreamingLLM is different: it's not about training, it's about stable KV cache eviction for unbounded inference."

**On SSMs vs attention:**
> "Mamba achieves O(n) training and O(1) inference via selective state space models. The tradeoff is expressiveness — transformers can do exact copying and in-context lookup trivially; SSMs must compress history into a fixed state, making that harder. Hybrid models (Mamba + attention layers) try to get the best of both."

**On the lost-in-the-middle problem:**
> "Extending context length doesn't automatically solve long-document understanding. Models exhibit primacy/recency bias: facts at the start and end of context are recalled better than facts in the middle. RULER and NIAH tests expose this. Fine-tuning with position-diverse examples and attention-modified architectures (e.g., YaRN's temperature correction) partially address it."

**On context compression:**
> "LLMLingua and Gist tokens approach compression differently. LLMLingua prunes tokens at the input level using a proxy model's perplexity — interpretable but lossy. Gist tokens compress prompts into continuous embeddings — not human-readable but potentially more information-dense. Both reduce KV cache pressure downstream."

**On extending a deployed model:**
> "For extending an existing RoPE model without training, NTK-aware interpolation is the first thing to try — zero fine-tuning, works up to ~4× extension with minimal degradation. For production quality at 8–16× extension, YaRN with ~400 steps of long-context fine-tuning is the current best practice. LongRoPE adds per-dimension optimization on top."

---

*Related: [RAG](rag.md) | [Inference Optimization](inference-optimization.md) | [Speculative Decoding](speculative-decoding.md)*
