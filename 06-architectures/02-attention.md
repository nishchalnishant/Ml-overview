---
module: Deep Learning
topic: Components
subtopic: Attention
status: unread
tags: [deeplearning, ml, components-attention]
---
# Attention

---

## The Fixed-Context Bottleneck

**The problem**: RNNs compress an entire input sequence into a single fixed-size hidden state before the decoder can generate output. For a long sentence, the encoder must somehow squeeze all semantically relevant information into a vector of a few hundred dimensions. By the time the decoder reads the 30th output word, the context from the 1st input word has been overwritten or diluted. Translation quality degrades sharply for long sentences.

**The core insight**: instead of compressing everything upfront, let the decoder directly look back at all encoder hidden states, and decide which ones are most relevant at each decoding step. The model learns what to look at — this is attention.

---

## Query, Key, Value

**The problem**: "look at relevant encoder states" is vague. How does the decoder specify what it's looking for? How does each encoder state say what it contains?

**The core insight**: split the problem into three roles. The decoder poses a *query* — a vector representing what it currently needs. Each encoder state broadcasts a *key* — what it has to offer. The *value* is the actual content that gets returned if the match is good. Compute similarity between query and all keys, normalize to a distribution, and return a weighted sum of values.

**The mechanics**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $Q \in \mathbb{R}^{n \times d_k}$: queries (what each token is looking for)
- $K \in \mathbb{R}^{m \times d_k}$: keys (what each token advertises)
- $V \in \mathbb{R}^{m \times d_v}$: values (what each token actually returns)
- Output: $\mathbb{R}^{n \times d_v}$ — for each query, a weighted blend of values

**Why divide by $\sqrt{d_k}$**: the dot product $QK^T$ has variance proportional to $d_k$ for random unit vectors. At high dimension, dot products become large, pushing softmax into near-zero gradient regions (the distribution becomes near one-hot). Dividing by $\sqrt{d_k}$ restores variance to 1.

**What breaks**: computing $QK^T$ is $O(n^2)$ in both time and memory. For a sequence of length 4096, this matrix has 16 million entries. At long contexts, this is the dominant bottleneck.

---

## Self-Attention vs Cross-Attention

**The problem**: the attention mechanism needs $Q$, $K$, $V$ inputs, but a Transformer has no separate encoder-decoder pair in many modern uses (e.g., GPT). How does a single sequence attend to itself? And how does one modality attend to another (e.g., text attending to image)?

**Self-attention**: $Q$, $K$, $V$ all come from the same sequence. Each token asks: "which other tokens in this sequence matter for my representation?" This is how Transformers build contextual meaning — the word "bank" updates its representation by attending to whether "river" or "money" appeared nearby.

**Cross-attention**: $Q$ comes from one sequence, $K$ and $V$ from another. Used in:
- Encoder-decoder Transformers (decoder attends to encoder output)
- Multimodal models (text tokens attending to image patch embeddings)
- Retrieval-augmented generation (query tokens attending to retrieved document tokens)

---

## Causal Masking

**The problem**: during autoregressive generation, the model must predict token $t$ using only tokens $1, \ldots, t-1$. If token $t$ can attend to tokens $t+1, t+2, \ldots$, the model sees the answer before it is asked — no learning happens.

**The core insight**: before applying softmax, mask the attention scores for future positions to $-\infty$. After softmax, those positions have weight $\approx 0$ and contribute nothing to the output.

**The mechanics**:

```python
seq_len = 10
mask = torch.tril(torch.ones(seq_len, seq_len))  # lower-triangular matrix
scores = scores.masked_fill(mask == 0, float('-inf'))
weights = F.softmax(scores, dim=-1)
```

**What breaks**: causal masking makes the model unidirectional — each token only sees the past. This is correct for generation, but wrong for understanding tasks. BERT-style encoders use bidirectional attention (no mask) because they process the whole sequence at once.

---

## Multi-Head Attention

**The problem**: a single attention pattern can only capture one type of relationship at a time — e.g., it might learn to track syntactic dependencies, but then lose semantic similarity. Language has many simultaneous structure types.

**The core insight**: run $h$ separate attention computations in parallel, each with its own learned projection matrices. Different heads learn different relationship types. Concatenate the results.

**The mechanics**:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

Each head projects to $d_k = d_v = d_\text{model} / h$. Total parameters: same as one large attention (the dimension per head is smaller). Total compute: the same $O(n^2 d)$ — just split across heads.

**What breaks**: with many heads, each head's subspace is small ($d_k = 64$ for a 1024-dim model with 16 heads). If the model is shallow and the task requires broad representations, small-head dimensions can be limiting. Some heads in trained models appear nearly inert — they learn near-uniform attention weights and contribute little.

---

## Positional Encodings

**The problem**: attention is permutation-invariant. Swapping two tokens in the input produces the same set of key-value pairs — only their positions in the output differ. The model cannot distinguish "dog bites man" from "man bites dog" without explicit position information.

**The core insight**: add a position-dependent signal to each token's embedding before feeding it into the Transformer.

### Sinusoidal (original Transformer)

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Fixed, not learned. Different frequencies encode different scales of position. Can generalize to sequences longer than those seen in training, in principle.

**What breaks**: absolute positions are encoded independently — the model must learn to infer relative distance from two absolute positions. This is indirect and does not generalize well to very long sequences.

### Learned Positional Embeddings (BERT, GPT-2)

An embedding table of shape $[\text{max\_seq\_len}, d]$. Simple and effective within the training context window.

**What breaks**: hard limit at training context length. Position 2049 has no embedding if training used 2048.

### RoPE and ALiBi

RoPE rotates query/key vectors by position so their dot product depends only on relative offset; ALiBi adds a linear distance penalty to attention scores instead of modifying vectors. Both extrapolate better to longer sequences than absolute/learned encodings. Full derivation, NTK/YaRN/LongRoPE scaling comparison: see [11-transformers.md](03-transformers.md#rope--rotary-position-embedding).

---

## KV Cache

**The problem**: during autoregressive generation, producing token $t$ requires computing attention over all $t-1$ previous tokens. Naively, this means recomputing keys and values for every past token at every step — $O(n^2)$ total compute for a sequence of length $n$.

**The core insight**: the keys and values for past tokens do not change between steps (assuming causal masking). Compute them once and cache them.

**The mechanics**: at each generation step, compute $K_t$ and $V_t$ for only the new token. Append to the cached $K_{1:t-1}$ and $V_{1:t-1}$. Run attention once over the full key-value cache. Total compute per step: $O(n)$ instead of $O(n^2)$.

**What breaks**: the KV cache grows linearly with sequence length and is proportional to:

$$2 \times n_\text{layers} \times n_\text{heads} \times d_k \times n_\text{ctx} \text{ floats}$$

At long contexts (100k+ tokens), the cache dominates GPU memory. Solutions:

- **MQA (Multi-Query Attention)**: all heads share a single $K$, $V$ projection. Cache size divided by $n_\text{heads}$. Some quality loss.
- **GQA (Grouped-Query Attention)**: groups of $g$ heads share one $K$, $V$ pair. Cache reduced by factor $n_\text{heads}/g$. Better quality-memory tradeoff than MQA. Used in LLaMA 3.

---

## Flash Attention

IO-aware tiled computation that never materializes the full $n \times n$ attention matrix in HBM — loads $Q$, $K$, $V$ blocks into SRAM, accumulates via online softmax. Same output as standard attention, $O(n)$ memory instead of $O(n^2)$. Full derivation, FA-1/2/3 comparison, and code: see [11-transformers.md](03-transformers.md#flash-attention--io-aware-attention).

---

## Complexity Summary

| Method | Time | Memory | Notes |
| :--- | :--- | :--- | :--- |
| **Standard Attention** | $O(n^2 d)$ | $O(n^2)$ | Baseline |
| **Flash Attention** | $O(n^2 d)$ | $O(n)$ | IO-efficient, same output |
| **Sparse Attention** | $O(n \sqrt{n} \cdot d)$ | $O(n \sqrt{n})$ | Longformer, BigBird |
| **Linear Attention** | $O(nd^2)$ | $O(d^2)$ | Approximate; quality trade-off |

---

## Canonical Interview Q&As

**Q: Derive the complexity of self-attention and explain why it's a bottleneck at long contexts.**  
A: For a sequence of length n with model dimension d, computing Q, K, V projections costs O(nd²). The attention matrix QKᵀ is n×n, costing O(n²d) to compute and O(n²) memory. Applying softmax and multiplying by V costs another O(n²d). Total: O(n²d) time, O(n²) memory. At n=32K (GPT-4's context), the n² term is 10⁹ — each attention layer requires ~8GB for the attention matrix alone in fp16. This is why Flash Attention is critical: it tiles the computation to avoid materializing the n×n matrix in HBM, reducing memory to O(n) while maintaining identical output. The compute is still O(n²d) — Flash Attention is IO-efficient, not compute-efficient.

**Q: What is the difference between MHA, MQA, and GQA, and when would you use each?**  
A: MHA (Multi-Head Attention) uses separate K, V projections per head — max expressiveness, max KV cache memory. MQA (Multi-Query Attention) shares a single K, V across all heads — reduces KV cache by n_heads×, with some quality degradation. GQA (Grouped-Query Attention) groups heads to share K, V within each group — interpolates between MHA and MQA. For a 70B model with 64 heads, MHA KV cache ≈ 64× MQA KV cache per token. At 32K context, MHA requires ~83GB for KV cache alone; GQA with 8 KV heads reduces this to ~10GB. Production choice: GQA is the standard (Llama-3, Gemma, Mistral) — empirically < 1% quality loss vs MHA with 4–8× memory reduction.

**Q: Why does softmax in attention cause issues at long contexts, and how do RoPE and ALiBi address position encoding differently?**  
A: At long contexts, dot products Q·Kᵀ scale with d_k^0.5 but with many keys, some will be much larger than others — softmax becomes peaky, attending to a few tokens and ignoring the rest. This "attention sink" phenomenon means distant relevant tokens get near-zero weight. Additionally, absolute position encodings learned at training time don't generalize to unseen positions. RoPE encodes position by rotating Q, K vectors by position-dependent angles — only relative positions matter, and the rotation is mathematically smooth for positions beyond training length (though quality still degrades without fine-tuning). ALiBi adds a linear bias proportional to distance: score_{ij} -= m|i-j|, where m is head-specific. ALiBi requires no re-training for length extrapolation and handles very long contexts better, but slightly underperforms RoPE at training lengths.
