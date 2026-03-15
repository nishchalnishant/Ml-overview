# Attention

Attention is a mechanism that lets a model focus on different parts of the input with different weights. It is the core of modern transformers and LLMs.

---

## Why attention?

- **Problem**: Fixed-size representations (e.g. last hidden state of an RNN) bottleneck long sequences; distant dependencies are hard to capture.
- **Idea**: For each output position, compute a *context* as a weighted sum of all input (or previous) positions. The weights are learned and data-dependent.

---

## Attention in one equation

For a query vector **q**, key vectors **k₁,…,kₙ**, and value vectors **v₁,…,vₙ**:

\[
\text{Attention}(\mathbf{q}, \mathbf{K}, \mathbf{V}) = \sum_i \alpha_i \mathbf{v}_i, \quad \alpha_i = \frac{\exp(\mathbf{q}^\top \mathbf{k}_i / \sqrt{d_k})}{\sum_j \exp(\mathbf{q}^\top \mathbf{k}_j / \sqrt{d_k})}
\]

- **Query (q)**: “what am I looking for?” (e.g. current decoder position).
- **Keys (K)**: “what do I offer?” (e.g. encoder positions).
- **Values (V)**: “what do I output?” (e.g. encoder hidden states).
- **αᵢ**: softmax over scores **q**ᵀ**k**ᵢ; scaling by √dₖ stabilizes gradients.

So the model *attends* to inputs where **q** matches **k**, and uses the corresponding **v** to form the output.

---

## Scaled dot-product attention (matrix form)

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

- **Q**: (batch, seq_q, d_k)  
- **K**: (batch, seq_k, d_k)  
- **V**: (batch, seq_k, d_v)  
- Scores: **QK**ᵀ / √dₖ → softmax → weights; then multiply by **V**.

---

## Multi-head attention

Run several attention “heads” in parallel with different linear projections, then concatenate and project:

\[
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h) W^O
\]
\[
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
\]

- Different heads can learn to focus on different aspects (e.g. syntax, coreference, position).
- Used in transformer encoder and decoder (and decoder cross-attention).

---

## Types of attention in transformers

| Type | Query source | Key/Value source | Use |
|------|----------------|-------------------|-----|
| **Self-attention (encoder)** | Same sequence | Same sequence | Each token attends to full input |
| **Self-attention (decoder)** | Same sequence (causal mask) | Same sequence | Each token attends only to past |
| **Cross-attention (decoder)** | Decoder | Encoder output | Decoder attends to encoder |

---

## Visual intuition

```
Input tokens  →  Q, K, V  →  Scores (QK^T / √d_k)  →  Softmax  →  Weighted sum of V  →  Output
                     ↑
              (learned projections)
```

---

## Strengths and weaknesses

**Strengths**

- Captures long-range dependencies in O(1) “hops” (vs RNN’s sequential steps).
- Parallelizable over sequence length.
- Interpretable: attention weights show where the model looked.

**Weaknesses**

- Cost is O(n²) in sequence length (n) for self-attention.
- Needs large data and careful training; sensitive to scale (hence √dₖ).

---

## Connection to modern AI

- **Transformers** are built on (multi-head) self-attention and feed-forward layers.
- **LLMs** (GPT, LLaMA, etc.) use causal self-attention in the decoder.
- **Encoder–decoder** models (e.g. T5, BART) add cross-attention from decoder to encoder for tasks like summarization and translation.

---

## Quick revision

- **Attention** = weighted sum of values, with weights from query–key similarity (softmax over qᵀk/√dₖ).
- **Multi-head**: multiple such mechanisms in parallel; concatenate and project.
- **Self** = Q,K,V from same sequence; **cross** = Q from decoder, K,V from encoder.
- **Causal mask** in decoder restricts each position to past only.
- Core building block of transformers and all modern LLMs.
