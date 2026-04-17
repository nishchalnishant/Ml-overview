# The Attention Mechanism (Deep-Dive)

Attention allowed neural networks to break free from the "sequential bottleneck" of RNNs. It enables models to focus on the most relevant parts of an input sequence, regardless of distance.

---

# 1. 🔹 The QKV Paradigm

## Q1: Explain Query, Key, and Value.

### 🔹 Direct Answer
The Attention mechanism uses three vectors for every input token:
1. **Query (Q):** "What am I looking for?"
2. **Key (K):** "What do I contain?"
3. **Value (V):** "What information do I give if targeted?"

### 🔹 The Logic
The model computes the dot product of the **Query** and all **Keys** to find the "Attention Score" (similarity). This score determines how much of each **Value** should be passed to the next layer.

---

# 2. 🔹 Scaled Dot-Product Attention

## Q2: Why divide by $\sqrt{d_k}$?

### 🔹 Scaled formula
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 🔹 Direct Answer
For large embedding dimensions $(d_k)$, the dot products can grow extremely large. Large values push the Softmax function into its flat, "saturated" regions where gradients are near-zero (Vanishing Gradient). Dividing by $\sqrt{d_k}$ keeps the values in a stable range.

---

# 3. 🔹 Self-Attention vs. Cross-Attention

- **Self-Attention:** $Q, K, \text{ and } V$ all come from the same sequence (e.g., inside an encoder). The model relates different parts of the same sentence.
- **Cross-Attention:** $Q$ comes from the decoder, while $K \text{ and } V$ come from the encoder. This allows a translator to "look" at the source sentence while generating the target.

---

# 4. 🔹 Multi-Head Attention (MHA)

## Q3: Why use multiple heads?

### 🔹 Direct Answer
A single attention head can only attend to one relationship at a time (e.g., "subject-verb" agreement). Multi-head attention allows the model to simultaneously attend to different aspects of the text (e.g., grammar, facts, style) in parallel.

---

> [!TIP]
> **Learning Tip:** For the architectural implementation of these components into a full model, see the [Transformers Hub](../ml-interview-notes/nlp.md).
