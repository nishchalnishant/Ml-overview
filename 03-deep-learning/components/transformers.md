---
module: Deep Learning
topic: Components
subtopic: Transformers
status: unread
tags: [deeplearning, ml, components-transformers]
---
**Primary reference:** [LLM Architecture Deep Dive](../../05-llms/architecture-deep-dive.md) | [Interview Q&As](../../05-llms/interview-notes/llm-fundamentals.md)

# Transformers

---

## Why the Sequential Bottleneck Had to Go

**The problem**: RNNs process sequences one token at a time. To learn that a verb at position 1 agrees with its subject at position 20, the relevant information must survive intact through 18 hidden state updates. Gradients flowing back from position 20 to position 1 are multiplied by 18 weight matrices and 18 activation derivatives — they vanish. Long-range dependencies are hard to learn.

The sequential processing also prevents parallelism: you cannot compute hidden state $h_t$ until $h_{t-1}$ is done. Training on long sequences is slow even on hardware that could parallelize.

**The core insight**: instead of threading context through a bottleneck hidden state, let every token directly look at every other token in a single step. Context is computed in parallel, and the gradient path from position 20 to position 1 is just one attention operation — no vanishing chain of multiplications.

---

## The Transformer Block

A single Transformer layer applies two sub-layers, both wrapped in a residual connection and layer normalization:

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

This block is stacked $L$ times. Each block refines the token representations; the final representations are fed to a task head.

The residual connections ensure that gradients flow directly from the output head to early layers without any multiplication — the skip path has gradient 1 regardless of what the sublayer does.

---

## Self-Attention and Multi-Head Attention

> Full mechanics of scaled dot-product attention, multi-head attention, causal masking, Q/K/V projections, MQA, and GQA are covered in [attention.md](./attention.md). This section summarizes what matters for understanding the Transformer block structure.

Self-attention lets every token directly look at every other token in a single step:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Multi-head attention runs $h$ attention computations in parallel with independent projections, then concatenates:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

For autoregressive decoders, a causal mask zeros out future positions. Computational cost is $O(n^2 d)$ — the bottleneck for long sequences.

---

## Feed-Forward Network (FFN)

**The problem**: attention layers aggregate information across positions but are limited in what non-linear transformation they can apply to individual token representations. The model needs per-token computation that does not require cross-token interaction.

**The core insight**: after attention has updated each token representation with context, apply a per-position feedforward network. This is where the model processes and transforms individual token representations — it is position-wise (independent across tokens) and adds the model's main non-linear capacity.

**The mechanics**:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Width typically $4 \times d_\text{model}$. LLaMA-style models replace this with SwiGLU:

$$\text{FFN}(x) = (\text{Swish}(xW_1) \odot xW_2) W_3$$

Three projections instead of two; hidden dimension $\frac{8}{3}d_\text{model}$ to keep parameter count matched.

**What breaks**: the FFN is the largest part of the model by parameter count — roughly two thirds of all parameters in a standard Transformer are in FFN layers. Making it more efficient (narrower, sparser, or using mixture-of-experts) is a major target for scaling.

---

## Positional Encodings

> Full treatment of sinusoidal, RoPE, ALiBi, and learned encodings is in [attention.md](./attention.md). Summary below.

Attention is permutation-invariant — positional encodings inject order information. Three main families:

- **Sinusoidal** (original Transformer): fixed frequencies, no parameters, limited length generalization
- **RoPE** (LLaMA, Mistral, most modern LLMs): rotates Q/K vectors so relative distance appears directly in attention dot products. Better length generalization, extendable via YaRN/LongRoPE
- **ALiBi** (MPT, BLOOM): subtracts a linear bias proportional to distance from attention scores. No parameters, strong extrapolation

---

## Pre-Norm vs Post-Norm

**The problem**: where does the layer normalization go — before or after the sublayer?

**Post-Norm** (original "Attention is All You Need"):

$$x = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Gradients must flow through the normalization on the path back through the residual stream. This adds an extra scaling step in the gradient path, making very deep networks unstable.

**Pre-Norm** (modern default — GPT-2+, LLaMA, most LLMs):

$$x = x + \text{Sublayer}(\text{LayerNorm}(x))$$

The normalization happens only inside the sublayer. The residual stream $x$ is updated by direct addition — the gradient path through the skip connection has gradient magnitude 1, completely bypassing any normalization scaling.

**What breaks** with Pre-Norm: the final residual stream values are unnormalized. A final LayerNorm before the output head is always required in Pre-Norm architectures. Without it, logit magnitudes vary unpredictably.

---

## Encoder-Only, Decoder-Only, Encoder-Decoder

**The problem**: different tasks have different informational requirements. Understanding a sentence (classification) benefits from seeing the full context. Generating text requires not seeing future tokens. Translating requires encoding a source and generating into a target space.

**Encoder-Only** — bidirectional attention; every token attends to every other:

**What it's for**: tasks where the full input is known upfront — classification, named entity recognition, extractive QA, embeddings.

Examples: BERT, RoBERTa, DeBERTa.

**Decoder-Only** — causal attention; each token attends only to past tokens:

**What it's for**: generation tasks — text completion, instruction following, agents.

Examples: GPT-2/3/4, LLaMA, Mistral, Claude.

**Encoder-Decoder** — encoder uses bidirectional attention; decoder uses causal self-attention + cross-attention to encoder output:

**What it's for**: transformation tasks where input and output are different sequences — translation, summarization, speech-to-text.

Examples: T5, BART, Whisper, original Transformer.

---

## KV Cache

> Full mechanics of KV caching, MQA, and GQA are in [attention.md](./attention.md).

During autoregressive generation, keys and values for past tokens do not change — cache them after the first computation. Cache size grows linearly:

$$\text{cache size} = 2 \times n_\text{layers} \times n_\text{heads} \times d_k \times n_\text{ctx}$$

**MQA**: all query heads share one K/V projection — cache divided by $n_\text{heads}$.
**GQA**: groups of heads share K/V — used in LLaMA 3. Intermediate quality/memory tradeoff between MHA and MQA.

---

## Scaling Laws

**The problem**: given a compute budget, how should you split it between model size and training data? Should you train a very large model on less data, or a smaller model on more data?

**The core insight (Chinchilla, 2022)**: before Chinchilla, most large models were trained on too little data for their size. The optimal compute split roughly requires training tokens $D \approx 20N$ for a model with $N$ parameters.

$$L(N, D) \approx \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$$

Loss decreases as a power law in both model size and data. They contribute approximately equally to loss reduction per unit of compute.

**What breaks**: the Chinchilla compute-optimal ratio applies to training compute. At inference, smaller models with longer training are more efficient to serve. A 7B model trained on 1T tokens may outperform a 70B model trained on 100B tokens and serve $10\times$ faster. The tradeoffs between training cost and serving cost must be considered together.

---

## Attention Complexity

| Method | Time | Memory | Notes |
| :--- | :--- | :--- | :--- |
| **Standard Attention** | $O(n^2 d)$ | $O(n^2)$ | Baseline |
| **Flash Attention** | $O(n^2 d)$ | $O(n)$ | IO-aware tiling, same output, 2–4× faster |
| **Sparse Attention** | $O(n\sqrt{n}\cdot d)$ | $O(n\sqrt{n})$ | Longformer, BigBird |
| **Linear Attention** | $O(nd^2)$ | $O(d^2)$ | Approximate, quality trade-off |

---

## Canonical Interview Q&As

**Q: Walk through the full forward pass of a transformer encoder layer, specifying what each component does.**  
A: Input tokens are embedded + positional encoding added → shape [batch, seq_len, d_model]. (1) Layer norm applied to input. (2) Multi-head attention: project input to Q, K, V for each head (W_Q, W_K, W_V projections); compute scaled dot-product attention: softmax(QKᵀ/√d_k)V for each head; concatenate and project with W_O — output shape same as input. (3) Residual connection: input + attention_output. (4) Layer norm again. (5) FFN: two linear layers with activation: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2, where hidden dim is typically 4×d_model. (6) Residual: previous + FFN output. The residual connections are critical — they allow gradients to flow directly to earlier layers, enabling deep transformers to train. Pre-norm (norm before sublayer, as above) is more stable than post-norm for large models.

**Q: Why do transformers use positional encoding, and what are the trade-offs between learned and sinusoidal encodings?**  
A: Self-attention is permutation-equivariant — without positional encoding, "cat ate fish" and "fish ate cat" produce identical representations. Positional encoding injects order information. Sinusoidal encoding: PE(pos, 2i) = sin(pos/10000^{2i/d_model}), PE(pos, 2i+1) = cos(...). Advantages: fixed, no parameters, can generalize to positions beyond training length through the periodic structure. Learned absolute encodings (e.g., original BERT): each position has a learned embedding vector. Better performance at training lengths but no extrapolation. In modern LLMs, both are replaced by relative encodings (RoPE, ALiBi) because they handle long contexts better. RoPE encodes position by rotating Q, K vectors, making relative positions implicit in the dot product — length generalization is better, though not perfect without fine-tuning on longer sequences.

**Q: What architectural change makes LLaMA/Mistral different from the original transformer, and why does it matter?**  
A: Several key changes: (1) RMSNorm instead of LayerNorm — removes mean subtraction, ~10% faster with no quality loss; (2) SwiGLU activation instead of ReLU in FFN — SwiGLU(x, W, V, W2) = (xW ⊙ SiLU(xV))W2; requires 3 weight matrices instead of 2, so FFN hidden dim is scaled down to keep params constant, but empirically converges 15-20% faster; (3) RoPE instead of absolute position encodings — handles long contexts better; (4) GQA instead of MHA — 4-8 KV heads instead of 32-64, reduces KV cache size by 8-16× which is the primary serving bottleneck; (5) Pre-normalization throughout. Together these changes make models faster to train (SwiGLU, RMSNorm), much cheaper to serve at long contexts (GQA), and better at long-context tasks (RoPE). At interview level: GQA's KV cache reduction is the most practically impactful change for production systems.
