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

### RoPE (Rotary Position Embedding)

**The problem**: absolute position encodings embed position as part of the token vector, so position information mixes with content information in ways that are hard to disentangle.

**The core insight**: encode position by rotating the query and key vectors in attention. The dot product between two rotated vectors depends only on the relative offset between their positions, not their absolute positions.

$$q'_m = q_m e^{im\theta}, \quad k'_n = k_n e^{in\theta}$$

$$q'_m \cdot k'_n = \text{Re}\left[q_m k_n^* e^{i(m-n)\theta}\right]$$

Relative position $(m-n)$ appears naturally in the attention score. Extrapolates better to longer sequences. Extended via YaRN and LongRoPE to $>100\text{k}$ tokens.

Used in LLaMA, Mistral, GPT-NeoX.

### ALiBi (Attention with Linear Biases)

Add a linear penalty to attention scores based on distance: score $-= m \cdot (i - j)$ where $m$ is a per-head slope. No added parameters; forces the model to prefer nearby tokens unless attending far is worth the penalty. Strong length generalization.

**What breaks**: the linear penalty may not match the actual distance dependency of all attention patterns, potentially harming tasks requiring very long-range dependencies.

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

**The problem**: standard attention materializes the full $n \times n$ attention matrix in GPU high-bandwidth memory (HBM). Reading and writing this matrix dominates runtime for long sequences — the bottleneck is memory bandwidth, not compute.

**The core insight**: compute attention in tiles. Load small blocks of $Q$, $K$, $V$ into fast SRAM (on-chip), compute partial attention results, accumulate using the online softmax trick, and write only the final output back to HBM. Never materialize the full $n \times n$ matrix.

**The mechanics**: IO-aware tiled computation. Same mathematical result as standard attention. Memory: $O(n)$ instead of $O(n^2)$. Wall-clock speed: 2–4x faster in practice on A100-class hardware.

Available in PyTorch 2.0+ as `F.scaled_dot_product_attention` with Flash Attention enabled by default.

**What breaks**: Flash Attention requires the full sequence to fit in a specific tiling pattern. Custom masking patterns (non-causal, non-rectangular) require careful implementation. Backward pass is also custom (recomputes attention on the fly to save memory).

---

## Complexity Summary

| Method | Time | Memory | Notes |
| :--- | :--- | :--- | :--- |
| **Standard Attention** | $O(n^2 d)$ | $O(n^2)$ | Baseline |
| **Flash Attention** | $O(n^2 d)$ | $O(n)$ | IO-efficient, same output |
| **Sparse Attention** | $O(n \sqrt{n} \cdot d)$ | $O(n \sqrt{n})$ | Longformer, BigBird |
| **Linear Attention** | $O(nd^2)$ | $O(d^2)$ | Approximate; quality trade-off |
