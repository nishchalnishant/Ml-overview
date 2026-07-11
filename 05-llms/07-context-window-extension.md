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

## Method 4: LongRoPE (brief)

LongRoPE uses evolutionary search to find per-dimension (rather than one global) rescaling factors, achieving extreme context (millions of tokens) with less fine-tuning than YaRN. Research/Staff-depth topic — know it exists and why (single global scale factor is suboptimal per-head/per-layer), but implementation detail is out of scope here.

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

## Qwen3 iRoPE (brief)

Qwen3 interleaves layers with no positional encoding (NoPE) every 4th layer among standard RoPE layers, enabling extension to 10M+ context with minimal fine-tuning. Architecture-research-depth — worth knowing as an example of "not every layer needs explicit position info," not something an SDE2 needs to implement.

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

## Sparse Attention Patterns

Full attention is O(n²). For n=128K, the attention matrix is 16B entries — too large to materialize. Sparse attention patterns approximate full attention by attending to a structured subset of token pairs.

### Longformer

Replace full O(n²) attention with two complementary patterns:
1. **Local window**: each token attends to its ±w/2 nearest neighbors — captures local syntax and nearby coreference.
2. **Global tokens**: a small set of designated tokens (CLS token, task-specific markers) attend to and from *all* tokens — propagate long-range information.

Cost: O(n·w) instead of O(n²). Information from distant parts of the document can only reach other parts via the global tokens, so global token placement matters.

**Limitation**: Longformer is a pretrained architecture, not a plug-in modification. You can't simply apply it to an existing model.

### BigBird

Extends Longformer by adding **random attention**: each token also attends to a random subset of all tokens. This creates a small-world graph structure:
- Local edges from the sliding window
- Global edges from global tokens
- Random long-range edges

BigBird proves this combination is theoretically equivalent to full attention (Turing-complete). In practice, random edges help with tasks requiring unpredictable long-range dependencies.

**Limitation**: random attention is non-deterministic — different runs may sample different edges. For precise long-range recall tasks, the random edges may miss the critical token.

---

## Context Compression

When most of a long context is irrelevant to the current query, compress before generation.

### LLMLingua

A small proxy LM computes per-token perplexity given surrounding context. Predictable (low-perplexity) tokens are redundant — the reader can reconstruct them. Remove the redundant tokens before passing to the expensive target LM.

Pipeline:
1. Use a small LM to compute perplexity of each token in the prompt
2. Rank tokens: low perplexity = remove, high perplexity = keep
3. Budget-aware allocation: instructions and the question get higher token budgets than background context
4. **LongLLMLingua** also weights tokens by relevance to the specific query (not just global perplexity)

**Limitation**: the proxy model may disagree with the target model about which tokens matter. Compression can remove causally necessary tokens.

### Gist Tokens

For workloads where many queries share the same long system prompt, fine-tune the model to compress the prompt into a small set of learned "gist" token embeddings. Cache the gist KV representations and reuse across all queries.

During fine-tuning: mask out the original prompt tokens; train the model to answer questions using only the gist tokens. At inference: compute gist once, cache, reuse.

**Limitation**: gist tokens are not human-readable. If the system prompt changes, gist tokens must be recomputed. Compression is lossy.

---

## State-Space Model Alternative: Mamba (brief)

Mamba is a selective state-space model offering O(n) training and O(1)-memory inference instead of attention's O(n²), by compressing history into a fixed-size recurrent state rather than attending to every past token. Tradeoff: exact long-range retrieval degrades since the state is bounded (vs. transformers' exact lookup). Hybrid Mamba+attention models (Jamba, Zamba) are the current practical compromise. Research/architecture-depth — good to know it exists as an alternative to RoPE scaling, not expected in SDE2-level depth.

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
