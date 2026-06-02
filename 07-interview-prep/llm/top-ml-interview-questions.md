---
module: Interview Prep
topic: Llm
subtopic: Top Ml Interview Questions
status: unread
tags: [interviewprep, ml, llm-top-ml-interview-questions]
---
# Top ML Interview Questions — Reference Answers

Concept + intuition + tradeoff for each question.

---

## 1. What is Machine Learning?

Learning a function $f: X \to Y$ from data $(x_i, y_i)$ rather than hand-coding rules.

**Types:**
- **Supervised:** labeled $(x, y)$ pairs — classification, regression
- **Unsupervised:** unlabeled $x$ — clustering, density estimation, representation learning
- **Reinforcement:** agent learns by interacting with environment, maximizing cumulative reward

---

## 2. Bias-Variance Tradeoff

$$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{\text{Bias}^2[\hat{f}]}_{\text{systematic error}} + \underbrace{\text{Var}[\hat{f}]}_{\text{sensitivity to data}} + \sigma^2$$

- **High bias (underfitting):** model too simple, misses patterns. Fix: more capacity, better features
- **High variance (overfitting):** model too complex, memorizes noise. Fix: regularization, more data
- **Goal:** minimize total error, not just training error

---

## 3. Overfitting vs Underfitting Diagnosis

```
Training loss \ Validation loss:
  Both high          → Underfitting (model too simple or features too weak)
  Train low, val high → Overfitting (memorizing training data)
  Both low, close    → Generalizing well
```

```python
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=5)
# Converging gap = diminishing overfitting with more data
# Persistent gap at large n = high variance problem
```

---

## 4. Train / Validation / Test Split

| Split | Purpose | When to touch |
| :--- | :--- | :--- |
| Train | Learn parameters | Every training run |
| Validation | Tune hyperparameters, select model | During development |
| Test | Final unbiased evaluation | Once — at the end |

**Common mistakes:** tuning hyperparameters on the test set (optimistic bias), using future data in train for time series (leakage).

For time series: always use temporal splits. Random splits allow future data to leak into training.

---

## 5. Hyperparameter vs Parameter

| | Parameters | Hyperparameters |
| :--- | :--- | :--- |
| **Learned from** | Data (gradient descent) | Set before training |
| **Examples** | Weights, biases | Learning rate, depth, regularization |
| **Tuning method** | Optimization | Grid search, random search, Bayesian optimization |

---

## 6. Feature Scaling

**Why it matters for some algorithms:**

| Algorithm | Needs scaling? | Reason |
| :--- | :--- | :--- |
| KNN | Yes | Distance-based — unscaled features dominate |
| SVM | Yes | Kernel distances, margin computation |
| Logistic regression | Yes | Gradient magnitudes depend on feature scale |
| Neural nets | Yes | Exploding/vanishing gradients |
| Decision trees / XGBoost | No | Split-based — only feature rank matters |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: zero mean, unit variance — good for algorithms assuming Gaussian
scaler = StandardScaler()  # z = (x - mean) / std

# MinMaxScaler: [0, 1] range — good for neural nets, image pixels
scaler = MinMaxScaler()    # z = (x - min) / (max - min)
```

---

## 7. Logistic Regression

$$P(y=1 \mid x) = \sigma(w^T x + b) = \frac{1}{1+e^{-(w^Tx+b)}}$$

**Loss:** binary cross-entropy = $-\frac{1}{n}\sum[y\log\hat{p} + (1-y)\log(1-\hat{p})]$

**Decision boundary:** linear in feature space. For non-linear boundaries: add polynomial features or use a neural net.

**Regularization:** C in sklearn = inverse lambda. Low C = more regularization. L1 → sparse coefficients. L2 → all coefficients non-zero.

---

## 8. Precision vs Recall

$$\text{Precision} = \frac{TP}{TP+FP} \quad \text{Recall} = \frac{TP}{TP+FN} \quad F1 = \frac{2PR}{P+R}$$

- **High precision needed:** spam filter (don't block legitimate email)
- **High recall needed:** cancer screening (don't miss cases)
- **F1:** balanced. PR-AUC better than ROC-AUC for rare positive classes.

---

## 9. Bagging vs Boosting

| | Bagging | Boosting |
| :--- | :--- | :--- |
| **Training** | Independent parallel models | Sequential, each fixes previous errors |
| **Reduces** | Variance | Bias |
| **Examples** | Random Forest | XGBoost, LightGBM, AdaBoost |
| **Overfitting risk** | Lower | Higher (if too many rounds) |
| **Speed** | Fast (parallelizable) | Slower (sequential dependency) |

**Random Forest:** builds $T$ trees, each on a bootstrap sample with a random feature subset. Averages predictions.

**XGBoost:** fits additive models $F_m(x) = F_{m-1}(x) + \alpha h_m(x)$ where each $h_m$ is a shallow tree fitted to residuals, with L1/L2 regularization on tree weights.

---

## 10. Backpropagation

Chain rule applied layer-by-layer from loss to parameters:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}$$

Forward pass: compute and cache activations. Backward pass: compute gradients in reverse using cached activations.

**Why cache activations?** Computing $\partial z / \partial W$ requires $a^{(l-1)}$ — must store it during forward pass.

---

## 11. Why BatchNorm?

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

- Reduces **internal covariate shift**: distribution of inputs to each layer stays stable
- Allows higher learning rates (gradients stay well-scaled)
- Acts as regularizer (noise from batch statistics)

**LayerNorm vs BatchNorm:** LayerNorm normalizes over features per sample (independent of batch size). Standard for Transformers. BatchNorm normalizes over the batch dimension — problematic for variable-length sequences and small batches.

---

## 12. Why ReLU?

$$\text{ReLU}(x) = \max(0, x), \quad \frac{d}{dx}\text{ReLU} = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

- Non-saturating for positive values → no vanishing gradient problem
- Computationally cheap (threshold operation)
- Sparse activations (half the neurons off) → implicit regularization

**Dying ReLU problem:** if a neuron always receives negative input, gradient is always 0 — neuron never updates. Fix: LeakyReLU, ELU, or careful initialization.

---

## 13. Transformer vs LSTM

| | LSTM | Transformer |
| :--- | :--- | :--- |
| **Processing** | Sequential — $O(T)$ depth | Parallel — $O(1)$ depth |
| **Long-range memory** | Degrades with distance | Full attention to all positions |
| **Training** | Cannot parallelize across time | Fully parallelizable |
| **Complexity** | $O(T \cdot d^2)$ | $O(T^2 \cdot d)$ — quadratic in sequence |
| **Use today** | Embedded systems, short sequences | Everything at scale |

LSTM is still used in low-latency edge settings where a small model must process streaming data without the quadratic attention cost.

---

## 14. Class Imbalance

**Problem:** minority class accuracy is drowned by majority class.

**Solutions:**

| Method | When to use |
| :--- | :--- |
| `class_weight="balanced"` | Always try first — free |
| SMOTE (synthetic oversampling) | Tabular data with low minority count |
| Undersampling | Large dataset where majority is very abundant |
| Focal loss | Deep learning — down-weights easy examples |
| Threshold tuning | When you have a probability score — use cost matrix |

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(weights))
# Pass to model: RandomForestClassifier(class_weight=class_weight_dict)
```

---

## 15. Model Drift

| Type | Cause | Detection |
| :--- | :--- | :--- |
| Feature drift | $P(X)$ changes | KS test, PSI on features |
| Concept drift | $P(Y \mid X)$ changes | Monitor prediction accuracy on labeled slice |
| Label drift | $P(Y)$ changes | Monitor score distribution |

**PSI thresholds:** < 0.1 = stable, 0.1–0.25 = monitoring needed, > 0.25 = retrain.

---

## 16. Candidate Generation vs Ranking (Two-Stage Pattern)

**Why two stages?** Running a heavy model on 10M items per query is too expensive.

```
Millions of items
    → Stage 1: Fast retrieval (ANN, BM25, collaborative filtering)
    → ~1000 candidates
    → Stage 2: Heavy ranking (neural net with full features)
    → Top 10–20 results
```

**Stage 1 tools:** FAISS (approximate nearest neighbor), Elasticsearch (BM25), two-tower embeddings.  
**Stage 2 tools:** LightGBM LambdaMART, DCN, DIN.

---

## 17. Why Cross-Validate?

A single train/val split can overfit to the particular split. Cross-validation averages over multiple splits:

$$\text{CV score} = \frac{1}{K}\sum_{k=1}^K \mathcal{L}(f_{-k}, D_k)$$

where $f_{-k}$ is trained on all folds except $k$, evaluated on fold $k$.

**Stratified K-Fold** (for classification): preserves class ratio in each fold. Use `StratifiedKFold` instead of `KFold` when class imbalance exists.

**Time-series CV:** cannot use random folds — use `TimeSeriesSplit` (rolling window forward).

---

## 18. Gradient Descent Variants

| Variant | Batch size | Pros | Cons |
| :--- | :--- | :--- | :--- |
| Batch GD | Full dataset | Stable, exact gradient | Slow per step, needs full data in memory |
| SGD | 1 sample | Fast updates, escapes local minima | Noisy, high variance |
| Mini-batch SGD | 32–512 | Best of both — GPU-efficient | Requires tuning batch size |
| Adam | Mini-batch | Adaptive LR, fast convergence | Can generalize worse than SGD+momentum |

---

## 19. Regularization Techniques

| Method | Effect | Formula |
| :--- | :--- | :--- |
| L1 (Lasso) | Sparse weights — some exactly 0 | $+\lambda\sum\|w_i\|$ |
| L2 (Ridge/weight decay) | Small weights — none exactly 0 | $+\lambda\sum w_i^2$ |
| Dropout | Random deactivation at training | Keep prob $p$; scale by $1/p$ at test |
| Early stopping | Stop before val loss increases | No added compute |
| Data augmentation | Increase effective dataset size | Domain-specific transforms |

---

## 20. Key Numbers to Know

| Fact | Value |
| :--- | :--- |
| Chinchilla optimal tokens/param | ~20 |
| GPT-3 params | 175B |
| LLaMA 3 training tokens | 15T |
| Standard attention complexity | $O(n^2 d)$ |
| LoRA trainable params | ~0.06% of base model |
| BF16 memory | 2 bytes/param |
| FP32 memory | 4 bytes/param |
| 70B model BF16 VRAM | ~140GB |

## Rapid Recall

### Supervised: labeled $(x, y)$ pairs
- Direct Answer: classification, regression
- Why: This matters because it tells you how to reason about supervised: labeled $(x, y)$ pairs.
- Pitfall: Don't answer "Supervised: labeled $(x, y)$ pairs" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: classification, regression

### Unsupervised: unlabeled $x$
- Direct Answer: clustering, density estimation, representation learning
- Why: This matters because it tells you how to reason about unsupervised: unlabeled $x$.
- Pitfall: Don't answer "Unsupervised: unlabeled $x$" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: clustering, density estimation, representation learning

### Reinforcement
- Direct Answer: agent learns by interacting with environment, maximizing cumulative reward
- Why: This matters because it tells you how to reason about reinforcement.
- Pitfall: Don't answer "Reinforcement" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: agent learns by interacting with environment, maximizing cumulative reward

### High bias (underfitting)
- Direct Answer: model too simple, misses patterns. Fix: more capacity, better features
- Why: This matters because it tells you how to reason about high bias (underfitting).
- Pitfall: Don't answer "High bias (underfitting)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: model too simple, misses patterns. Fix: more capacity, better features

### High variance (overfitting)
- Direct Answer: model too complex, memorizes noise. Fix: regularization, more data
- Why: This matters because it tells you how to reason about high variance (overfitting).
- Pitfall: Don't answer "High variance (overfitting)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: model too complex, memorizes noise. Fix: regularization, more data

### Goal
- Direct Answer: minimize total error, not just training error
- Why: This matters because it tells you how to reason about goal.
- Pitfall: Don't answer "Goal" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: minimize total error, not just training error

### High precision needed
- Direct Answer: spam filter (don't block legitimate email)
- Why: This matters because it tells you how to reason about high precision needed.
- Pitfall: Don't answer "High precision needed" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: spam filter (don't block legitimate email)

### High recall needed
- Direct Answer: cancer screening (don't miss cases)
- Why: This matters because it tells you how to reason about high recall needed.
- Pitfall: Don't answer "High recall needed" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: cancer screening (don't miss cases)

### F1
- Direct Answer: balanced. PR-AUC better than ROC-AUC for rare positive classes.
- Why: This matters because it tells you how to reason about f1.
- Pitfall: Don't answer "F1" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: balanced. PR-AUC better than ROC-AUC for rare positive classes.

### Reduces internal covariate shift
- Direct Answer: distribution of inputs to each layer stays stable
- Why: This matters because it tells you how to reason about reduces internal covariate shift.
- Pitfall: Don't answer "Reduces internal covariate shift" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: distribution of inputs to each layer stays stable

### Allows higher learning rates (gradients stay well-scaled)
- Direct Answer: Allows higher learning rates (gradients stay well-scaled)
- Why: This matters because it tells you how to reason about allows higher learning rates (gradients stay well-scaled).
- Pitfall: Don't answer "Allows higher learning rates (gradients stay well-scaled)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Allows higher learning rates (gradients stay well-scaled)

### Acts as regularizer (noise from batch statistics)
- Direct Answer: Acts as regularizer (noise from batch statistics)
- Why: This matters because it tells you how to reason about acts as regularizer (noise from batch statistics).
- Pitfall: Don't answer "Acts as regularizer (noise from batch statistics)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Acts as regularizer (noise from batch statistics)

### Non-saturating for positive values → no vanishing gradient problem
- Direct Answer: Non-saturating for positive values → no vanishing gradient problem
- Why: This matters because it tells you how to reason about non-saturating for positive values → no vanishing gradient problem.
- Pitfall: Don't answer "Non-saturating for positive values → no vanishing gradient problem" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Non-saturating for positive values → no vanishing gradient problem

### Computationally cheap (threshold operation)
- Direct Answer: Computationally cheap (threshold operation)
- Why: This matters because it tells you how to reason about computationally cheap (threshold operation).
- Pitfall: Don't answer "Computationally cheap (threshold operation)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Computationally cheap (threshold operation)

### Sparse activations (half the neurons off) → implicit regularization
- Direct Answer: Sparse activations (half the neurons off) → implicit regularization
- Why: This matters because it tells you how to reason about sparse activations (half the neurons off) → implicit regularization.
- Pitfall: Don't answer "Sparse activations (half the neurons off) → implicit regularization" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Sparse activations (half the neurons off) → implicit regularization
