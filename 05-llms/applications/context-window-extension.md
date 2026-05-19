# Context Window Extension

---

## The Core Problem

**The problem:** transformer attention computes every pair of token interactions. For a sequence of length n that produces n² dot-products per layer. At n=8192 that is 67 million per layer; at n=128k it is 16 billion. This manifests as:

- **Memory:** the attention matrix must be materialized in GPU high-bandwidth memory — at n=32k in float16, roughly 2 GB per layer.
- **Compute:** FLOPs scale quadratically, making long-context inference slow even when memory holds.

But there is a third problem independent of both: a model trained up to length L will fail at length L' > L — not because of memory or compute, but because the **position encoding values seen at test time are out of the training distribution**. These three problems (memory, compute, generalization) require independent solutions.

---

## Positional Encodings

### The Problem They Solve

A transformer has no inherent notion of token order — attention is a set operation. Without positional information, "the cat sat on the mat" and "the mat sat on the cat" produce identical attention patterns. Positional encodings inject order so the model can distinguish position 3 from position 103. The challenge is making encodings that generalize beyond training lengths.

---

### RoPE (Rotary Position Embedding)

**The problem:** absolute positional embeddings assign a fixed vector to each position index. A model trained to position 4096 has no learned representation for position 4097 — that embedding was never updated. Absolute position is also less useful than relative position for most reasoning: whether token i is 5 positions before token j matters more than what their absolute positions are.

**The core insight:** instead of adding a positional vector to token embeddings, rotate the query and key vectors in 2D subspaces by an angle proportional to position. The dot product between a rotated Q and a rotated K then depends only on the relative offset between them — not their absolute positions. This follows from the geometry of rotation: `(R(m)q)ᵀ(R(n)k) = qᵀR(n-m)k`.

**The mechanics:** for position m and dimension pair (2i, 2i+1), the rotation angle is `mθᵢ` where `θᵢ = 10000^(-2i/d)`. High-frequency dimensions (small θ) encode fine-grained local position; low-frequency dimensions encode coarse long-range structure. Attention scores naturally decay with distance because high-frequency terms cancel for large position gaps.

**What breaks:** the model is still trained up to some maximum length. Positions beyond that produce rotation angles the model has never seen — positionally out-of-distribution even though the formula still applies. This is what the length extension methods below address.

---

### ALiBi (Attention with Linear Biases)

**The problem:** even with RoPE, extending beyond the training length requires either fine-tuning or interpolation. Position indices out of the training range remain a problem.

**The core insight:** remove positional embeddings entirely. Instead, subtract a fixed linear penalty proportional to token distance directly from the attention score before softmax:

```
score(i, j) = QᵢKⱼ/√d − m·(i−j)
```

where `m` is a head-specific slope. The penalty grows linearly with distance. Because the penalty formula involves only the distance `(i−j)` and is identical at any sequence length, the model sees nothing "new" when the sequence grows. There are no out-of-distribution position indices.

**What breaks:** the linear decay assumption may not match task requirements. A fact 10,000 tokens ago is penalized proportionally to its distance even if highly relevant. ALiBi also plateaus in quality at very long contexts compared to RoPE-based methods with good scaling.

---

## RoPE Length Extension

These methods address the generalization problem: how to use a RoPE-trained model at lengths beyond its training distribution.

### Linear Scaling (Position Interpolation)

**The problem:** a model trained on positions 0..L-1, when given positions L..L'-1, encounters values it has never seen. Naive application degrades output.

**The core insight:** compress the longer sequence's positions back into the trained range. Scale every position index by `L/L'` so the longest sequence still uses only positions 0..L-1.

**The mechanics:** at inference, replace position m with `m × (L/L')`. Requires ~1000 steps of fine-tuning to adapt the model to the new spacing. Works to roughly 2–4× extension.

**What breaks:** uniform compression treats all frequency bands equally. High-frequency RoPE dimensions that encode local syntax are compressed as aggressively as low-frequency ones that encode long-range structure — degrading local representations.

---

### NTK-Aware Interpolation

**The problem:** linear scaling breaks high-frequency dimensions disproportionately. The model uses high-frequency RoPE components to distinguish nearby tokens; squishing those frequencies confuses local structure.

**The core insight:** instead of scaling positions, scale the base frequency θ. Changing θ affects low-frequency dimensions strongly (their wavelength is proportional to θ) while high-frequency dimensions are nearly unchanged. Local structure is preserved while range is extended.

**The mechanics:** replace `θᵢ = 10000^(-2i/d)` with `θ'ᵢ = (10000 · (L'/L)^(d/(d-2)))^(-2i/d)`. Requires **zero fine-tuning** for moderate extension ratios (up to ~4×) and minimal fine-tuning for larger ones.

**What breaks:** at large extension ratios the approximation breaks down. NTK-aware scaling without fine-tuning degrades noticeably past ~8× extension.

---

### YaRN

**The problem:** NTK-aware scaling applies a single global modification. Different frequency bands have different sensitivities to length extension — one scalar cannot optimally handle all of them.

**The core insight:** treat the frequency spectrum in three bands. High-frequency dimensions (local structure): leave completely unchanged. Low-frequency dimensions (global structure): apply linear interpolation. Medium-frequency: interpolate proportionally to wavelength. Additionally, longer contexts increase softmax entropy (attention spreads more uniformly), so apply a temperature correction to re-sharpen attention.

**What breaks:** requires ~400 steps of fine-tuning on long documents to consolidate. The hyperparameters (band boundaries, temperature correction factor) require tuning per model family.

---

### LongRoPE

**The problem:** the frequency band boundaries in YaRN are set by hand. Optimal per-dimension rescaling factors cannot be derived analytically for a specific trained model.

**The core insight:** search for optimal per-dimension rescaling factors using an evolutionary algorithm, then apply a two-stage extension — first to 256k tokens, then fine-tune, then restore original RoPE for sequences shorter than the original max length so short-context quality is not degraded.

**What breaks:** the evolutionary search is expensive to run. The resulting factors are model-specific and do not transfer to other model families.

---

## Sparse Attention

### The Problem They Solve

Even with perfect positional encodings, full attention is O(n²) in both memory and compute. At n=128k, the attention matrix is 16 billion entries — past the limits of any current GPU for a single layer. The question is whether something close to full attention can be computed without materializing the full n×n matrix.

---

### Longformer

**The problem:** for most tokens, nearly all of the n×n attention matrix is low-information noise — distant, unrelated tokens. Only nearby tokens and a small set of globally important tokens receive meaningful attention weight.

**The core insight:** replace full attention with two complementary patterns. Every token attends to its ±w/2 nearest neighbors (capturing local syntax and coreference). A small set of designated global tokens (CLS, task tokens) attend to and from all tokens (propagating information across the document despite the window restriction).

Cost: O(n·w) instead of O(n²). Global tokens are the bottleneck through which long-range information flows.

**What breaks:** information can only flow from distant parts of the document through global tokens. If global tokens are not positioned or trained to aggregate the right information, distant cross-document reasoning fails. Longformer is a pretrained architecture, not a plug-in for existing models.

---

### BigBird

**The core insight:** add random attention (each token attends to a random subset of all tokens) on top of window + global attention. This creates a small-world graph structure: local edges from the window, global edges from global tokens, random long-range edges. BigBird proves this combination is Turing-complete and can approximate full attention.

**What breaks:** random attention is non-deterministic — different runs sample different edges. For tasks requiring precise long-range recall, random edges may miss the critical token. Theoretical guarantees require sufficient global tokens.

---

### StreamingLLM

**The problem:** standard models are trained with a fixed context window. In streaming settings — indefinitely long conversations, live transcription — the context grows without bound. Re-encoding the full context on each new token is intractable.

**The core insight:** two empirical observations make stable streaming possible. First, most relevant context for current generation is recent. Second, the very first tokens in a sequence accumulate disproportionately large attention mass regardless of their content — they act as attention sinks, serving as numerical anchors for the softmax. Removing them from the KV cache catastrophically destabilizes generation even though they carry no semantic importance.

**The mechanics:** keep exactly two sets of KV cache entries — the first 4 tokens (attention sinks) and the most recent W tokens (sliding window). When the window is full, evict the oldest non-sink token.

**What breaks:** tokens that slide out of the window are permanently lost. The model cannot recall anything more than W tokens ago. This is a stable inference mechanism, not a long-term memory mechanism.

---

## Memory-Efficient Attention

### FlashAttention

**The problem:** standard attention writes the full n×n score matrix to GPU high-bandwidth memory (HBM) for softmax, then reads it back for the weighted sum of values. At n=8192 in float16, this is ~256 MB of HBM traffic per attention layer per head. HBM bandwidth (~2 TB/s on A100) is the bottleneck — the arithmetic units idle while waiting for memory transfers.

**The core insight:** the n×n matrix is a transient intermediate result. It does not need to exist as a complete tensor in HBM. It can be computed tile by tile in SRAM (fast on-chip memory, ~192 KB on A100), with softmax maintained via running max and running sum statistics (online softmax). The output is accumulated without ever writing the full matrix to HBM.

**The mechanics (tiling):**
1. Divide Q, K, V into blocks that fit in SRAM.
2. For each Q block, iterate over all K and V blocks.
3. Maintain running softmax statistics to combine results across blocks.
4. Accumulate the output — the full n×n matrix is never in HBM.

Memory: O(n) instead of O(n²). FLOPs are unchanged (still O(n²) arithmetic). Speedup comes from reduced HBM reads/writes: IO complexity drops from O(n²) to O(n²/M) where M is SRAM size.

**What breaks:** requires custom CUDA kernels — cannot be expressed as standard PyTorch operations. Hardware-specific: tiling must fit SRAM, which differs across GPU generations. FlashAttention-2 improved work partitioning across warps. FlashAttention-3 adds pipeline overlap for H100's asynchronous memory units.

---

### Ring Attention

**The problem:** even FlashAttention is bounded by a single GPU's SRAM. For sequences longer than a single GPU's memory can handle even with tiling, the sequence must be distributed across devices.

**The core insight:** each GPU holds a chunk of Q and a chunk of K, V. The K, V blocks rotate around a ring of GPUs. Each GPU computes its part of the output using the online softmax trick across all K, V blocks it receives. No device ever needs the full sequence — each device sees one K, V chunk at a time.

**What breaks:** ring communication adds latency. Compute-communication overlap mitigates this but requires careful implementation. Sequence length must be divisible by the number of devices.

---

## Context Compression

### The Problem They Solve

Long context is expensive to process: O(n²) attention over a 100k token prompt is slow. For many tasks, most of that context is irrelevant to the current query — only a fraction matters. Compression removes the irrelevant parts before generation.

---

### LLMLingua

**The problem:** the full prompt must be processed even if most of it is redundant for the specific query.

**The core insight:** a small proxy language model can measure how predictable (low perplexity) each token is given surrounding context. Predictable tokens are redundant — if the reader can predict them, they carry little new information. Remove the low-perplexity tokens.

**The mechanics:**
1. Use a small LM to compute perplexity of each token.
2. Rank tokens by perplexity — low perplexity = remove, high perplexity = keep.
3. Apply budget-aware allocation: instructions and question tokens get higher budgets than background tokens.
4. LongLLMLingua extends this to weight tokens by relevance to the specific question.

**What breaks:** the proxy model may not agree with the target model about which tokens matter. Compression can remove tokens that are low perplexity but causally necessary for the answer. Quality must be evaluated on the specific task, not just on prompt perplexity.

---

### Gist Tokens

**The problem:** many queries share the same long system prompt or instruction prefix. Re-encoding that prefix for every query wastes compute.

**The core insight:** fine-tune the model to compress a long prompt into a small set of learned "gist" token embeddings that capture the prompt's semantic content. Cache the gist tokens' KV representations and reuse them across all queries sharing the prefix.

**The mechanics:** during fine-tuning, mask out the original prompt tokens and train the model to perform tasks conditioned only on gist tokens. At inference, compute gist tokens once, cache their KV pairs, and never re-encode the original prompt.

**What breaks:** gist tokens are not human-readable. If the original prompt changes, gist tokens must be recomputed. The compression is lossy — nuance in the original prompt may be lost.

---

## Evaluation

### Needle-in-a-Haystack (NIAH)

**The problem:** a model may claim to support 128k context but fail to retrieve a specific fact buried in the middle of a 128k document. Perplexity on long documents does not capture this — a model can have low perplexity while missing retrievable facts.

**The core insight:** inject a single specific fact ("the magic word is PINEAPPLE") at a known position in a long document. Ask the model to retrieve it. Plot retrieval accuracy as a heatmap over (document length, injection position). The model either retrieves it or it does not.

**What breaks:** NIAH tests only single-fact retrieval. A model can ace NIAH while failing at multi-hop reasoning over long contexts. RULER extends NIAH to variable tracking, aggregation, and multi-hop QA to expose these failures.

---

### Lost-in-the-Middle

**The problem:** even models that pass NIAH often fail when relevant information is in the middle of the context rather than at the beginning or end. This U-shaped recall pattern is documented across GPT-3.5, GPT-4, Claude, and open-source models.

**The core insight:** attention patterns during pretraining create an implicit primacy/recency bias — the first and last tokens are consistently attended to more than middle tokens. This survives instruction tuning because it is baked into learned attention weights, not just surface behavior.

**Practical implication:** for RAG systems, put the most relevant retrieved chunks first or last, not in the middle. For long-context fine-tuning, oversample examples where the answer appears in the middle of the context.

---

## Recurrent and State-Space Alternatives

### Mamba (Selective State Space Models)

**The problem:** even with all the above techniques, transformers are fundamentally O(n²) in training due to full attention. For sequences of millions of tokens, this is prohibitive regardless of engineering.

**The core insight:** model sequences as a continuous-time state space system: `h'(t) = A·h(t) + B·x(t)`, `y(t) = C·h(t)`. Discretized to: `hₜ = Ā·hₜ₋₁ + B̄·xₜ`, `yₜ = C·hₜ`. Training parallelizes via parallel scan — O(n) on GPUs. Inference is pure recurrence — O(1) state update per new token.

Mamba's key innovation: make B, C, and the discretization step Δ input-dependent. This lets the model choose what to retain in state based on content, which fixed-parameter RNNs cannot do.

**What breaks:** the fixed-size state is a bottleneck. Transformers can do exact in-context lookup by attending to any earlier token. Mamba must compress all history into a fixed state vector. A critical fact processed 50,000 tokens ago may not survive many subsequent state updates. Hybrid models (alternating Mamba and attention layers) are the current practical compromise.

---

*Related: [RAG](rag.md) | [Inference Optimization](inference-optimization.md) | [Speculative Decoding](speculative-decoding.md)*
