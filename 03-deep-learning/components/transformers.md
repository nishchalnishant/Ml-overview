**Primary reference:** [LLM Architecture Deep Dive](../../05-llms/architecture-deep-dive.md) | [Interview Q&As](../../05-llms/interview-notes/llm-fundamentals.md)

# Transformers

Transformers are what happened when deep learning decided it was done pretending sequential bottlenecks are charming.

They became dominant because they scale beautifully and handle context far better than older sequence models.

---

# 1. Why Transformers Won

RNNs process tokens one by one — creating two problems:
- slower training (sequential bottleneck)
- weaker long-range memory (gradient path length proportional to sequence length)

Transformers solve this with attention:
- better parallelization → faster training on modern hardware
- stronger context handling → every token directly attends to every other token
- easier scaling with data and compute → scaling laws hold cleanly

---

# 2. Full Architecture Walkthrough

A standard Transformer block (one layer):

```
Input x
  │
  ├─ LayerNorm (pre-norm)
  │     ↓
  │  Multi-Head Self-Attention
  │     ↓
  ├─ Residual: x = x + Attention(LN(x))
  │
  ├─ LayerNorm (pre-norm)
  │     ↓
  │  Feed-Forward Network (FFN)
  │     ↓
  └─ Residual: x = x + FFN(LN(x))
```

Stacked $L$ times. Output head (linear + softmax) for generation.

---

# 3. Three Core Ideas

## Self-Attention

Tokens look across the sequence to build contextual meaning.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Complexity: $O(n^2 d)$ in time, $O(n^2)$ in memory. The dominant bottleneck for long sequences.

## Multi-Head Attention

Multiple attention patterns run in parallel:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Each head can specialize in different relationship types (local, long-range, syntactic, coreference).

## Feed-Forward Network (FFN)

Applied independently to each token position:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Width is typically $4 \times d_{\text{model}}$. In LLaMA-style models, replaced with SwiGLU:

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$$

which uses 3 linear projections but achieves better quality.

---

# 4. Positional Encodings

Attention is permutation-invariant — without position info, the model cannot distinguish word order.

## Sinusoidal (original Transformer)

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Added to token embeddings. Different frequencies for different dimensions. Can extrapolate to unseen lengths.

## Learned Positional Embeddings

Learnable embedding table indexed by position. Simple, but limited to training context length. Used in BERT, GPT-2.

## RoPE (Rotary Position Embedding)

Used in LLaMA, Mistral, GPT-NeoX. Encodes **relative position** by rotating Q and K in attention:

$$q'_m = q_m e^{im\theta}, \quad k'_n = k_n e^{in\theta}$$

Dot product $q'_m \cdot k'_n$ depends on $(m-n)$ (relative offset). Better length generalization; extended via YaRN and LongRoPE to $>100\text{k}$ tokens.

## ALiBi

Linear bias added to attention scores: $-m \cdot |i-j|$ where $m$ is a per-head slope. No extra parameters. Strong length extrapolation.

---

# 5. Layer Normalization Placement

## Post-Norm (Original "Attention is All You Need")

$$x = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Harder to train deep networks — gradients must flow through the norm.

## Pre-Norm (Modern Default)

$$x = x + \text{Sublayer}(\text{LayerNorm}(x))$$

More stable training; dominant in GPT-3, LLaMA, modern LLMs. Clean gradient highway through skip connection.

---

# 6. Encoder-Only vs Decoder-Only vs Encoder-Decoder

## Encoder-Only

Best for **understanding** tasks. All tokens attend to all tokens (bidirectional attention).

Examples: BERT, RoBERTa, DeBERTa.

Use for: text classification, NER, question answering (extractive), embeddings.

## Decoder-Only

Best for **generation**. Causal (left-to-right) masking — each token attends only to past tokens.

Examples: GPT-2/3/4, LLaMA, Mistral, Claude.

Use for: text generation, completion, instruction following, agents.

## Encoder-Decoder

Best for **transformation** tasks. Encoder processes the full input; decoder attends to encoder output via cross-attention.

Examples: T5, BART, original Transformer (translation), Whisper (speech-to-text).

Use for: translation, summarization, seq2seq generation.

---

# 7. KV Cache

During autoregressive generation, past keys and values are reused:

- **Without cache:** recompute $K$, $V$ for all previous tokens at each step → $O(n^2)$ total
- **With KV cache:** compute $K$, $V$ only for new token; cache the rest → $O(n)$ per step

**Memory cost:** $2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_k \times n_{\text{ctx}}$ floats per sequence.

For 30B parameter model with 2k context: KV cache ≈ 2GB per sequence in FP16.

**Techniques to reduce KV cache:**
- **MQA (Multi-Query Attention):** all heads share one K, V projection (GPT-J style)
- **GQA (Grouped-Query Attention):** groups of heads share K, V (LLaMA 3 style, balance quality vs memory)

---

# 8. Scaling Laws

From Chinchilla (DeepMind, 2022):

$$L(N, D) \approx \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$$

- $N$: number of parameters
- $D$: number of training tokens
- **Optimal compute allocation:** scale model and data equally (roughly $D \approx 20N$ tokens for $N$ params)

Key insight: most models before Chinchilla were **undertrained** (too large for their training data budget). Smaller models trained on more data can outperform larger models trained on fewer data.

**Inference-time scaling (o1 / R1 paradigm):** more compute at inference (chain-of-thought, search, verification) can substitute for training compute. Different scaling axis entirely.

---

# 9. Transformer Tradeoffs

**Why they are great:**
- Scalable — performance improves reliably with more params + data + compute
- Strong context modeling — every token directly attends to every other
- Dominant in modern NLP, LLMs, vision (ViT), multimodal

**Why they are painful:**
- Attention cost grows as $O(n^2)$ with sequence length
- Large models are expensive to serve
- Often need huge data or strong pretraining
- Training instability at scale (solved with pre-norm, careful init, learning rate warmup)

The right answer is not "Transformers are best."

It is:

> "Transformers are powerful, but their compute and data demands are a major design consideration. For short sequences on limited compute, RNNs/SSMs (Mamba) can be competitive."
