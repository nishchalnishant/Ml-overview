# AI & ML Interview Master Revision Guide (Cheat Sheet)

This is a high-yield revision hub designed for the **last 24 hours before an interview**. It condenses the most critical "Gold Standard" patterns into a single-page rapid-fire reference.

---

## 🚀 1. The "Golden Rules" of ML Logic
- **Overfitting?** -> High Variance. Add data, L1/L2 Regularization, Dropout, Early Stopping, or simplify architecture.
- **Underfitting?** -> High Bias. Increase model capacity, feature engineering, reduce regularization.
- **Data Imbalance?** -> Use PR-AUC or F1 (never Accuracy). SMOTE, Class Weights, or Focal Loss.
- **Vanishing Gradients?** -> Use ReLU/Leaky ReLU, Batch Norm, Residual Connections, or switch to LSTM/Transformer.
- **Exploding Gradients?** -> Gradient Clipping or Weight Regularization.

---

## 🏗️ 2. Architectural Patterns (The "How it Works")

| Topic | Key Mechanism | Why it Matters |
| :--- | :--- | :--- |
| **Transformer** | Self-Attention ($O(N^2)$) | Parallelism + Global Context. |
| **RAG** | Retrieval + Augmented Prompting | Fixes Hallucination & Knowledge Cutoff. |
| **LoRA** | Low-Rank (A $\times$ B) Adaptation | Fine-tune 100B models on consumer GPUs. |
| **RLHF/DPO** | Preference Alignment | Makes models helpful/safe vs. just "next-token." |
| **Agents** | Loop: Perceive -> Plan -> Tool Call | Autonomous problem solving. |

---

## 📐 3. Math & Derivations (Whiteboard Ready)

- **Backprop:** $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$ (Chain Rule).
- **Attention:** $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ (Scaling by $\sqrt{d_k}$ prevents vanishing gradients).
- **Sigmoid Derivative:** $\sigma(z)(1 - \sigma(z))$ (Max value is 0.25, leading to signal decay).
- **Adam:** Combines momentum (mean) & scaling by variance (RMSProp) + Bias Correction.
- **Cross-Entropy Gradient:** $\hat{y} - y$ (Linear error signal makes it ideal for classification).

---

## 🏛️ 4. System Design & Infrastructure

- **3D Parallelism:** DP (Data) + TP (Tensor) + PP (Pipeline). Required for models > 1 GPU memory.
- **Quantization:** FP16 -> INT8/INT4. Reduces VRAM $2\times$ to $4\times$.
- **Inference Speed:** Use KV-Caching, Speculative Decoding, and Flash Attention 2.
- **Evaluation:** G-Eval (LLM-as-a-judge), RAGAS (faithfulness), and standardized benchmarks (MMLU).

---

## 🚩 5. Common Interview "Gotchas"

1. **RAG vs. Fine-tuning?** RAG for facts/knowledge; Fine-tuning for behavior/style.
2. **Why $\sqrt{d_k}$ in Attention?** Prevents the dot product from exploding, which would push softmax into the flat region where gradients vanish.
3. **Difference between Batch Norm & Layer Norm?** BN works across samples (bad for small batches); LN works across features (standard for Transformers).
4. **Data Leakage via ID?** Sequential IDs often correlate with target. Always drop IDs.
5. **Precision vs. Recall?** Precision = quality (avoid false positives); Recall = quantity (avoid false negatives).

---

## 🔗 6. Deep Dive Specialized Hubs

- [🧠 LLM Fundamentals](llm-interview-notes/llm-fundamentals.md)
- [🏗️ AI System Design](llm-interview-notes/ai-system-design.md)
- [📐 Math Derivations](ml-interview-notes/math-derivations.md)
- [🛠️ Practical Scenarios](ml-interview-notes/practical-ml-scenarios.md)
- [🛡️ AI Safety & Ethics](llm-interview-notes/ai-safety-ethics-and-responsible-ai-what.md)

---

> [!IMPORTANT]
> **The Senior Answer Frame:**
> *"The direct answer is [Fact]. The intuition is [Analogy]. However, in real-world production, the tradeoff is usually [Cost/Latency/Scale]."*
