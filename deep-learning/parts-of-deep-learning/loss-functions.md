# Loss Functions (Deep-Dive)

The loss function (also called cost function) translates the model's error into a single scalar value that the optimizer can minimize. Choosing the right loss is critical for convergence.

---

# 1. 🔹 Classification Losses

## Q1: Why Cross-Entropy for Classification?

### 🔹 Direct Answer
Cross-Entropy measures the distance between two probability distributions (the ground truth and the model's prediction). Unlike MSE, it provides a very strong gradient signal when the model is confidently wrong, leading to much faster convergence for classification tasks.

### 🔹 Comparison Table

| Loss Function | Use Case | Mathematical Key |
| :--- | :--- | :--- |
| **Binary Cross-Entropy** | 2-class classification. | $-[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$ |
| **Categorical CE** | Multi-class (One-hot). | $-\sum y_i \log(\hat{y}_i)$ |
| **Sparse Categorical CE** | Multi-class (Integer labels). | Memory efficient for many categories. |
| **Focal Loss** | Imbalanced Data. | Down-weights easy samples to focus on hard ones. |

---

# 2. 🔹 Regression Losses

## Q2: MSE vs. MAE (L2 vs. L1).

### 🔹 Direct Answer
- **MSE (Mean Squared Error):** Squares the errors. It is highly sensitive to outliers because large errors are penalized exponentially.
- **MAE (Mean Absolute Error):** Takes the absolute difference. It is robust to outliers but its derivative is non-continuous at zero, which can complicate optimization.
- **Huber Loss:** The best of both worlds. It behaves like MSE near zero and like MAE for large errors.

---

# 3. 🔹 Knowledge Embedding Losses

- **Contrastive Loss (Triplet Loss):** Used in Face Recognition (Siamese Networks). It pulls "Positive" pairs together and pushes "Negative" pairs apart.
- **KL Divergence:** Used in VAEs and LLM Alignment (RLHF) to measure how much one probability distribution diverges from a baseline.

---

> [!TIP]
> **Implementation Note:** In PyTorch, `nn.CrossEntropyLoss` combines `nn.LogSoftmax` and `nn.NLLLoss`. Do NOT add a Softmax layer to your model if you are using this loss function.
