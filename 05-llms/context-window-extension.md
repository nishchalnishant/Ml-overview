---
module: Llms
topic: Context Window Extension
subtopic: ""
status: unread
tags: [llms, ml, context-window-extension]
---
# Context Window Extension

How to extend Transformer context beyond training length. Critical topic for 2024/2025 LLM interviews — every major model (Llama 3, Qwen3, Gemini, GPT-4) uses these techniques.

---

## Why Context Extension Is Hard

Standard RoPE embeds position via rotation matrix applied to Q and K:

$$\mathbf{q}_m = R_m \mathbf{q}, \quad \mathbf{k}_n = R_n \mathbf{k}$$

$$R_m = \begin{pmatrix} \cos(m\theta_1) & -\sin(m\theta_1) & \cdots \\ \sin(m\theta_1) & \cos(m\theta_1) & \cdots \\ \vdots & & \ddots \end{pmatrix}$$

where $\theta_i = 10000^{-2i/d}$ (base = 10000).

**The problem:** At position m > L_train, the model has never seen rotations with those frequencies during training. The attention pattern breaks — effective perplexity spikes sharply at the training length boundary.

**Why not just train longer?** Quadratic attention cost: O(L²) FLOPs. 32K context costs 4× more compute per forward pass than 8K. Pre-training at 128K is prohibitively expensive.

---

## Method 1: Linear (Position) Interpolation

**Idea:** Rescale all positions by factor L_train / L_target.

$$m' = m \cdot \frac{L_{train}}{L_{target}}$$

A position at m = 50,000 in a 4K-trained model (target 128K) becomes m' = 50,000 × (4000/128000) ≈ 1562 — within training range.

**Works but degrades:** positions are now fractional frequencies, model wasn't trained to handle fine-grained position distinctions at low values. Requires fine-tuning (~1000 steps on long documents) to recover quality.

**Code:**
```python
def linear_rope_scaling(position_ids, scale_factor):
    """scale_factor = L_target / L_train"""
    return position_ids / scale_factor

# In HuggingFace: rope_scaling = {"type": "linear", "factor": 8.0}
```

---

## Method 2: NTK-Aware Scaling

**Insight from Neural Tangent Kernel theory:** Rather than uniformly scaling positions, change the RoPE base. A higher base increases wavelengths of all frequency components proportionally — the model can represent longer-range dependencies without fractional positions.

**New base formula:**
$$\theta_i^{new} = \left(\alpha \cdot b\right)^{-2i/d}$$

where $\alpha = \left(\frac{L_{target}}{L_{train}}\right)^{d/(d-2)}$ and b = 10000 (original base).

Equivalently, set new_base = base × scale_factor^(d/(d-2)).

**Key advantage:** Preserves high-frequency (short-range) position sensitivity while expanding low-frequency (long-range) range. No fine-tuning required in some variants.

```python
def ntk_rope_base(original_base, scale_factor, head_dim):
    """Compute NTK-aware RoPE base for context extension."""
    return original_base * (scale_factor ** (head_dim / (head_dim - 2)))

# Llama-3 8B uses base=500000 for 128K context
# Original Llama-2 used base=10000 for 4K context
# 500000 / 10000 = 50x base increase for ~32x context extension
```

**Base values across models:**
| Model | Context | RoPE base |
|---|---|---|
| Llama-2 | 4K | 10,000 |
| Llama-3 | 8K (pretrain) | 500,000 |
| Qwen2.5 | 128K | 1,000,000 |
| Qwen3 | 32K–131K | 1,000,000 (+ iRoPE) |
| Gemma 3 | 128K | — |

---

## Method 3: YaRN (Yet another RoPE extensioN)

YaRN (Peng et al. 2023) combines interpolation + NTK with a per-frequency ramp function.

**Key observation:** Different frequency components have different extrapolation behavior:
- **High-frequency (short-range):** very sensitive, should not be scaled (keep original)
- **Low-frequency (long-range):** less sensitive, can be linearly interpolated

**Ramp function:**
$$\gamma(r) = \begin{cases} 0 & r < r_{low} \\ 1 & r > r_{high} \\ \frac{r - r_{low}}{r_{high} - r_{low}} & \text{otherwise} \end{cases}$$

where r = 2i/d is the normalized frequency index.

**Mixed interpolation:**
$$h_i = (1 - \gamma(r_i)) \cdot \frac{m}{s} + \gamma(r_i) \cdot m'_{ntk,i}$$

High frequencies (γ=1): use NTK scaling. Low frequencies (γ=0): use linear interpolation.

**Attention temperature correction (entropy compensation):**

Extended context → attention entropy increases → attention becomes diffuse. YaRN applies a scale factor:
$$\sqrt{s} = 0.1 \cdot \ln(s) + 1.0$$

Applied as: attention_score *= 1/sqrt(s) before softmax. This sharpens attention back to training distribution.

```python
# YaRN in HuggingFace
rope_scaling = {
    "type": "yarn",
    "factor": 8.0,
    "original_max_position_embeddings": 4096,
    "attention_factor": 0.1 * math.log(8.0) + 1.0
}
```

---

## Method 4: LongRoPE

**Problem with all above:** They use a single global scale factor. Optimal interpolation factor differs per attention head and per layer.

**LongRoPE approach:** Evolutionary search to find per-dimension rescaling factors that minimize perplexity on long documents.

Two-phase: 
1. Extend to 2× target length (e.g., 512K) with non-uniform factors
2. Fine-tune, then apply rescaled factors at inference for 256K

**Result:** Achieves 2M token context with significantly less fine-tuning than YaRN.

---

## Method 5: ALiBi (Attention with Linear Biases)

Instead of encoding position in Q/K via rotation, add a linear bias to attention logits:

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} - m \cdot |i - j|\right) V$$

where m is a head-specific slope (m = 2^{-8/n_heads} for each head).

**Properties:**
- No positional encoding during training
- Linear extrapolation is built-in (linear penalty for distance)
- Handles arbitrarily long sequences without fine-tuning
- **Used in:** BLOOM, MPT

**Trade-off vs RoPE:**
| | RoPE | ALiBi |
|---|---|---|
| Short context (<4K) | Better | Slightly worse |
| Long context extension | Needs rescaling | Extrapolates natively |
| Relative position | Implicit via rotation | Explicit via bias |
| Pre-trained model adaptation | Needs fine-tuning | Needs retraining |

---

## Method 6: Streaming LLM (StreamingLLM)

**Problem:** Even with long-context support, KV cache grows linearly with sequence length → OOM for infinite streams.

**Key insight (Xiao et al. 2023):** LLMs place heavy attention on "sink tokens" (first few tokens, especially BOS). These anchors stabilize attention distribution.

**StreamingLLM:** Keep only:
1. Attention sinks: first K tokens (K=4 typically)
2. Recent window: last W tokens (sliding window)

```
[sink_1, sink_2, sink_3, sink_4] + [...sliding window of W tokens...]
                                               ↑ new tokens appended here
                                            old tokens evicted from left
```

**Result:** Constant KV cache memory O(K + W), handles infinite streams. Tested up to 4M token generation.

**Trade-off:** Cannot attend to tokens outside the window (memory is truly gone). Not suitable for RAG or tasks requiring recall of arbitrary past tokens.

---

## Lost-in-the-Middle: U-Shaped Recall

**Empirical finding (Liu et al. 2023):** LLM performance degrades significantly when the relevant information is in the middle of a long context.

```
Performance
    │  ▲            ▲
    │   \          /
    │    \        /
    │     \      /
    │      \    /
    │       \  /
    │        \/
    └─────────────────────
    start   middle   end
         Position in context
```

**Implication:** Place the most important information at the start or end of context. For RAG, most relevant chunks should be at the beginning.

**Why it happens:** Attention has positional recency bias (recent tokens easier to attend to) + primacy bias (early tokens are attention sinks). Middle tokens compete poorly for attention.

---

## Qwen3 iRoPE

**Innovation:** Every 4th layer in Qwen3 uses no positional encoding at all (position-agnostic attention).

**Motivation:** Full attention layers can model global dependencies. With NoPE layers interspersed, the model learns position-agnostic representations for some transformations while maintaining positional grounding in other layers.

**Result:** Enables extending to 10M+ context with minimal fine-tuning. The NoPE layers "reset" positional context, preventing position OOF issues from cascading through all layers.

```
Layer 0: RoPE (positional)
Layer 1: RoPE (positional)
Layer 2: RoPE (positional)
Layer 3: NoPE (position-agnostic) ← every 4th
Layer 4: RoPE (positional)
...
```

---

## Practical Extension Recipe (e.g., fine-tune Llama to 128K)

```python
# Step 1: Change RoPE base (NTK-aware)
model.config.rope_scaling = {
    "type": "yarn",
    "factor": 32.0,  # 4K → 128K
    "original_max_position_embeddings": 4096
}

# Step 2: Fine-tune on long documents
# - Use ~1000–5000 steps
# - Mix long documents (>32K tokens) with short documents
# - Learning rate: 2e-5 (small, avoid catastrophic forgetting)
# - Use gradient checkpointing (saves memory at long contexts)
# - Enable FlashAttention 2 (O(n) memory instead of O(n²))

# Step 3: Evaluate with RULER benchmark or "needle in haystack"
# Needle: insert a random fact at position p, ask model to retrieve it
# Sweep p across 0%–100% of context → measure recall per position
```

**Memory for 128K context (7B model, fp16):**
$$\text{KV cache} = 2 \times L \times H \times d_k \times \text{context} \times \text{dtype\_bytes}$$
$$= 2 \times 32 \times 32 \times 128 \times 131072 \times 2 \approx 68 \text{GB}$$

At 128K context, KV cache alone exceeds a single A100 (80GB). Requires:
- Chunked prefill (process prompt in 4K chunks)
- PagedAttention (virtual memory for KV cache)
- Multi-GPU with tensor parallelism

---

## Benchmark: Needle in a Haystack

Standard evaluation for long-context capability:
1. Fill context with distractor text (e.g., Paul Graham essays)
2. Insert "secret fact" at random position (needle)
3. Ask model to retrieve needle
4. Measure recall@position across 16 positions × 8 context lengths = 128 test cases

```python
def needle_in_haystack_eval(model, tokenizer, context_len, needle_position):
    needle = "The special password is: XK9-ALPHA-7"
    question = "What is the special password?"
    
    # Build context
    haystack = load_paul_graham_essays()[:context_len - 200]
    insert_pos = int(len(haystack) * needle_position)
    full_context = haystack[:insert_pos] + needle + haystack[insert_pos:]
    
    # Query
    response = model.generate(full_context + "\n\n" + question)
    return "XK9-ALPHA-7" in response
```

---

## Comparison Table

| Method | Fine-tuning needed | Quality | Memory | Notes |
|---|---|---|---|---|
| Linear interpolation | Yes (~1K steps) | Good | Same | Simple, widely used |
| NTK-aware | No (some need FT) | Good | Same | Higher base = longer wavelengths |
| YaRN | Yes (~400 steps) | Best | Same | Per-frequency ramp + entropy fix |
| LongRoPE | Yes | Best at extreme | Same | Evolutionary search for factors |
| ALiBi | Retrain | Good | Same | Extrapolates naturally |
| StreamingLLM | No | Limited (no recall) | O(K+W) | Infinite stream only |

---

## Canonical Interview Q&As

**Q: Why does naive RoPE fail beyond the training context length?**  
A: RoPE encodes position m via rotation matrix with frequencies θᵢ = base^(-2i/d). At m > L_train, the rotation angles are out-of-distribution — the model has never seen these values during training. The query-key dot products produce attention patterns that don't correspond to any learned position relationship. Empirically, perplexity spikes sharply at the training length boundary. The fix is to rescale positions or adjust base frequencies so in-distribution rotations cover the desired context length.

**Q: What's the difference between NTK-aware scaling and YaRN?**  
A: NTK-aware globally increases the RoPE base to extend the period of all frequency components proportionally. YaRN is more nuanced: it applies different scaling to different frequency dimensions — high frequencies (short-range dependencies) are left largely unchanged, while low frequencies (long-range) are interpolated. YaRN also corrects attention entropy (attention becomes too uniform at long contexts) via a learned temperature factor. YaRN generally outperforms NTK at the same context extension factor but requires ~400 fine-tuning steps.

**Q: You're building an LLM application that needs 500K context. What architecture and deployment choices do you make?**  
A: (1) Model choice: Qwen3 (10M context) or Gemini 1.5 Pro (1M) — both use techniques specifically designed for extreme context; (2) serving: need chunked prefill to avoid OOM on long prompts, PagedAttention for KV cache management, multi-GPU with tensor parallelism; (3) KV cache: at 500K context, 7B fp16 model needs ~265GB KV cache — must use quantized KV (int8 saves 2×), or use MLA (DeepSeek-style) which compresses KV to low-rank representation; (4) architecture decision: consider whether all 500K tokens actually need attention to each other, or if a sliding window + retrieved context pattern suffices.

**Q: What is the "lost in the middle" problem and how do you mitigate it?**  
A: LLMs recall information at the beginning and end of context much better than the middle — U-shaped recall curve (Liu et al. 2023). Caused by attention primacy (early tokens are attention sinks) and recency bias. For RAG: place most relevant retrieved chunks at the start of context, not the middle. For long-document summarization: use recursive summarization (chunk → summarize → summarize summaries). For retrieval: use a reranker to ensure the top-1 result is placed first. Some research uses position-aware loss during fine-tuning to reduce this bias.

**Q: ALiBi vs RoPE — when would you choose ALiBi for a new model?**  
A: ALiBi natively extrapolates to longer sequences without any rescaling — if you're training a model where inference context will vary widely and you don't want to engineer context extension, ALiBi is simpler. Trade-off: ALiBi slightly underperforms RoPE on short-context tasks (the linear bias is suboptimal for close-range attention patterns). For a model where you know the context length upfront and want maximum quality, RoPE + NTK/YaRN is better. ALiBi is rarely chosen for new foundation models since 2023 — RoPE with high base has largely superseded it.

## Flashcards

**High-frequency (short-range)?** #flashcard
very sensitive, should not be scaled (keep original)

**Low-frequency (long-range)?** #flashcard
less sensitive, can be linearly interpolated

**No positional encoding during training?** #flashcard
No positional encoding during training

**Linear extrapolation is built-in (linear penalty for distance)?** #flashcard
Linear extrapolation is built-in (linear penalty for distance)

**Handles arbitrarily long sequences without fine-tuning?** #flashcard
Handles arbitrarily long sequences without fine-tuning

**Used in?** #flashcard
BLOOM, MPT

**Chunked prefill (process prompt in 4K chunks)?** #flashcard
Chunked prefill (process prompt in 4K chunks)

**PagedAttention (virtual memory for KV cache)?** #flashcard
PagedAttention (virtual memory for KV cache)

**Multi-GPU with tensor parallelism?** #flashcard
Multi-GPU with tensor parallelism
