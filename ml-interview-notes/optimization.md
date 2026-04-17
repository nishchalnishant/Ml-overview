# Optimization & Training Dynamics

These notes follow the **Gold Standard** for interview preparation: providing direct answers, geometric intuition, and the "why" behind modern optimizer choices.

---

# 1. Gradient Descent Fundamentals

## Q1: Explain Stochastic Gradient Descent (SGD) vs. Mini-batch GD.

### 🔹 Direct Answer
- **Batch GD:** Computes gradients using the entire dataset. Perfectly accurate but computationally impossible for large data.
- **SGD (Stochastic):** Computes gradients using a single random sample ($batch size = 1$). Very fast but extremely noisy.
- **Mini-batch GD:** Computes gradients using a small subset (e.g., 32 or 512). It is the industry standard because it balances the efficiency of GPUs with the stability of larger gradients.

### 🔹 Intuition
Imagine you are trying to find the bottom of a dark valley.
- **Batch GD:** You wait for a satellite map of the entire mountain range before taking one step. (Accurate but slow).
- **SGD:** You just feel the slope under your left foot and jump. (Fast but you might trip).
- **Mini-batch:** You use a flashlight to see 10 feet ahead. (Efficient and steady).

---

# 2. Adaptive Optimizers

## Q2: How does the Adam Optimizer work?

### 🔹 Direct Answer
**Adam** (Adaptive Moment Estimation) combines the benefits of **Momentum** (speeding up in consistent directions) and **RMSprop** (adjusting the learning rate based on how frequently a feature changes). It maintains moving averages of both the gradient (1st moment) and the squared gradient (2nd moment).

### 🔹 High-Yield Comparison Table

| Optimizer | Main Idea | Best for... |
| :--- | :--- | :--- |
| **Vanilla SGD** | Simple step. | Simpler models, stable data. |
| **SGD + Momentum** | Adds "velocity" to go fast. | Most CNNs / CV tasks. |
| **RMSprop** | Divides by a running avg of grads. | RNNs / Complex sequences. |
| **Adam** | Momentum + Adaptive Scaling. | LLMs, Transformers, Default starting point. |
| **AdamW** | Adam with Decoupled Weight Decay. | Modern training (prevents L2 coupling). |

### 🔹 Deep Dive: Why AdamW?
In standard Adam, L2 regularization (weight decay) is added to the gradient. Because Adam scales gradients adaptively, the weight decay effect is inadvertently scaled too. **AdamW** separates (decouples) the weight decay from the gradient update, leading to better generalization.

---

# 3. Training Stability

## Q3: Explain Vanishing and Exploding Gradients.

### 🔹 Direct Answer
- **Vanishing Gradients:** Gradients become near-zero during backprop, so the model stops learning (Common with Sigmoid in deep nets).
- **Exploding Gradients:** Gradients grow exponentially large, causing weights to become `NaN` and the model to diverge.

### 🔹 Fixes Table

| Problem | Fixes |
| :--- | :--- |
| **Vanishing** | ReLU activation, Residual/Skip connections, BatchNorm, Xaiver/He Initialization. |
| **Exploding** | Gradient Clipping, Weight Regularization, Batch Normalization. |

### 🔹 Code Snippet (Gradient Clipping in PyTorch)
```python
# Prevents gradients from exceeding a threshold
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

# 4. Hyperparameter Optimization (HPO)

## Q4: Grid Search vs. Random Search vs. Bayesian Optimization

### 🔹 Direct Answer
- **Grid Search:** Exhaustive search over a fixed set. Good only for 1-2 parameters.
- **Random Search:** Samples randomly. Statistically superior to Grid Search in high-dimensional spaces because not all parameters are equally important.
- **Bayesian Optimization:** Uses a surrogate model (e.g., Gaussian Process) to "reason" where the best parameters might be, balancing exploration and exploitation.

### 🔹 Intuition
- **Grid:** Mowing a lawn in perfect rows.
- **Random:** Throwing darts at a board.
- **Bayesian:** Playing "Hot or Cold." You move toward where you found a "warm" spot in the previous turn.
