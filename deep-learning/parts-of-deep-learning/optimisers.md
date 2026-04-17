# Optimization Algorithms (Deep-Dive)

Optimizers are the engine of a neural network. They determine how to update the weights based on the loss gradient to reach a global minimum efficiently.

---

# 1. 🔹 Gradient Descent Families

## Q1: Why use Mini-batch SGD instead of Full Batch or Stochastic (1 sample)?

### 🔹 Direct Answer
- **Batch GD:** Computationally expensive; doesn't fit in memory for large data; often gets stuck in local minima.
- **Stochastic GD (1 Sample):** Noisy updates; can't leverage GPU parallelism.
- **Mini-batch SGD:** The industry standard. It balances efficiency (GPU vectorization) and stability (averaging gradients over a small batch).

---

# 2. 🔹 Adaptive Optimizers

## Q2: RMSProp vs. Adam vs. AdamW.

### 🔹 Direct Answer
- **RMSProp:** Solves the vanishing/exploding gradient problem by normalizing the gradient by a moving average of its magnitude.
- **Adam:** Combines **Momentum** (mean of gradients) and **RMSProp** (variance of gradients). It is the default "set-and-forget" optimizer for most tasks.
- **AdamW:** Fixes a flaw in Adam where weight decay was incorrectly implemented. It is the gold standard for Transformers and large scale models.

### 🔹 Comparison Table

| Optimizer | Mechanism | Best For |
| :--- | :--- | :--- |
| **SGD+Momentum** | Adds "velocity" to go through ravines. | Fine-tuning, simple CNNs. |
| **Adam** | Adaptive Learning Rates + Momentum. | General NLP, CV, Feed-forward. |
| **AdamW** | Decoupled Weight Decay + Adam. | Transformers (BERT, Llama). |
| **Adagrad** | Large updates for rare features. | Sparse data, embeddings. |

---

# 3. 🔹 Training Stability

- **Learning Rate Warm-up:** Linearly increasing the LR for the first few thousand steps to prevent the model from "diverging" during cold starts.
- **Cosine Annealing:** Gradually decaying the LR according to a cosine curve to reach a finer global minimum at the end of training.
- **Gradient Clipping:** Hard-capping the gradient magnitude to prevent **Exploding Gradients** in deep RNNs or large Transformers.

---

> [!TIP]
> **Production Choice:** Always start with **AdamW** with a learning rate of $1e-4$ or $5e-5$. If generalization is poor after convergence, switch to **SGD with Momentum** and a learning rate scheduler.
