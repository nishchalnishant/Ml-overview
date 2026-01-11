# LLM Mechanics: From Transformers to KV-Cache

## Executive Summary
| Concept | Technical Essence | Interview Key |
|---------|-------------------|---------------|
| **Attention** | $Softmax(\frac{QK^T}{\sqrt{d_k}})V$ | Quadratic complexity $O(L^2)$ |
| **Scaling Laws** | Performance $\propto$ Compute, Data, Params | Chinchilla optimality ($20$ tokens/param) |
| **Decoding** | Greedy, Beam, Top-P, Top-K | Precision vs. Diversity trade-off |
| **KV-Cache** | Reuse past Keys/Values during inference | Reduces $O(L)$ to $O(1)$ per token |

---

## 1. The Transformer Architecture
The core of modern LLMs (GPT, Llama, Claude).

### The Attention Mechanism
Allows the model to focus on relevant context $K$ for a given query $Q$.
- **Queries ($Q$)**: What I am looking for.
- **Keys ($K$)**: What I contain.
- **Values ($V$)**: The information I contribute.
- **Formula**: $Attention(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- **Why $\sqrt{d_k}$?**: To prevent the dot product from growing too large, which would push the Softmax into regions with near-zero gradients.

### Positional Encodings
Since Transformers process tokens in parallel (unlike RNNs), they have no inherent sense of order. We add **Sinusoidal** or **Rotary ($RoPE$)** encodings to the input embeddings.

---

## 2. Training at Scale

### Chinchilla Scaling Laws
DeepMind discovered that most models are "under-trained". 
- **Rule of Thumb**: For optimal performance, you should scale training data and parameters equally. 
- **Ratio**: Approximately **20 tokens per parameter** (e.g., a 7B model needs ~140B tokens).

### Tokenization: BPE & WordPiece
LLMs don't read words; they read **tokens**.
- **BPE (Byte Pair Encoding)**: Iteratively merges the most frequent pairs of characters.
- **Advantage**: Handles "out-of-vocabulary" words by breaking them into sub-units (e.g., `un-happi-ness`).

---

## 3. Inference Optimization: KV-Caching
In autoregressive generation, each new token requires re-computing the attention for all previous tokens.
- **The Problem**: $O(L^2)$ complexity.
- **The Solution**: Store the **Keys ($K$)** and **Values ($V$)** of previous tokens in memory. 
- **The Result**: Only compute $K$ and $V$ for the *newest* token, reducing the per-token computational cost significantly.

---

## Interview Questions

**1. "Explain the difference between Encoder-only, Decoder-only, and Encoder-Decoder architectures."**
> - **Encoder-only (BERT)**: Sees whole sentence at once. Best for NLU (Sentiment, NER).
> - **Decoder-only (GPT)**: Causal masking (sees only past). Best for generation.
> - **Encoder-Decoder (T5)**: Encoder for input, Decoder for output. Best for Translation or Summarization.

**2. "What is Perplexity ($PPL$) and how does it relate to Cross-Entropy?"**
> Perplexity is the exponentiated cross-entropy loss: $PPL = e^{H(p,q)}$. It represents the "weighted branching factor" of the model. A lower PPL means the model is less "confused" and more certain in its predictions.

**3. "Why is Softmax temperature used in decoding?"**
> $P_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$. 
> - **$T < 1$**: Makes the distribution "sharper" (more deterministic).
> - **$T > 1$**: Makes the distribution "flatter" (more creative/diverse, but higher risk of hallucinations).

---

## Code Snippet: Minimal Softmax with Temperature
```python
import numpy as np

def softmax(logits, temperature=1.0):
    logits = logits / temperature
    exp_logits = np.exp(logits - np.max(logits)) # stability trick
    return exp_logits / np.sum(exp_logits)
```
