# Loss Functions

---

## Why the Loss Function Choice Matters

**The problem**: gradient descent can only optimize what the loss measures. If the loss is a poor proxy for what you actually care about (correct predictions, calibrated probabilities, robust embeddings), the model optimizes hard for the wrong thing. A well-trained model on the wrong loss is a well-trained failure.

**The core insight**: pick the loss that encodes the probabilistic model you actually want. For binary classification you want calibrated probabilities — cross-entropy is the log-likelihood of a Bernoulli model. For regression you want to minimize squared residuals — MSE is the negative log-likelihood of a Gaussian. The loss is not arbitrary arithmetic; it encodes your beliefs about the data-generating process.

---

## Binary Cross-Entropy (BCE)

**The problem**: for binary classification, you need the model to output a probability $\hat{y} \in (0,1)$ and you need a loss that penalizes wrong confident predictions harshly.

**The core insight**: use log-likelihood. The model asserts $P(y=1) = \hat{y}$. The log-probability of the observed label under this model is $y \log \hat{y} + (1-y) \log(1-\hat{y})$. Maximizing this (minimizing its negation) is maximum likelihood estimation for a Bernoulli model.

**The mechanics**:

$$L = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]$$

When the model predicts $\hat{y} \approx 0$ for a true label of 1, $\log(\hat{y}) \to -\infty$. The loss is unbounded for confident wrong predictions. This strong penalty drives the model away from confident errors.

**What breaks**: if you compute `log(sigmoid(logits))` directly, numerical underflow occurs when logits are very negative. Use `BCEWithLogitsLoss`, which combines sigmoid and log in a numerically stable form:

```python
F.binary_cross_entropy_with_logits(logits, targets)  # stable
F.binary_cross_entropy(torch.sigmoid(logits), targets)  # unstable at extremes
```

---

## Multiclass Cross-Entropy

**The problem**: multiple mutually exclusive classes. The model outputs one probability per class; they must sum to 1. You need a loss that penalizes assigning low probability to the correct class.

**The core insight**: same maximum likelihood idea. The model asserts a categorical distribution over $K$ classes. The log-likelihood of the correct class under this distribution is $\log \hat{y}_k$ where $k$ is the true class. Minimize the negative.

**The mechanics**:

$$L = -\frac{1}{m} \sum_{i=1}^{m} \log \hat{y}_{i,k_i}$$

where $k_i$ is the true class index for example $i$. With one-hot labels: $-\sum_k y_{ik} \log \hat{y}_{ik}$ (only the correct class contributes).

PyTorch's `CrossEntropyLoss` takes raw logits and applies log-softmax internally:

```python
ce = nn.CrossEntropyLoss()
loss = ce(logits, targets)  # targets are class indices, NOT one-hot, NOT softmax outputs
```

**What breaks**: never apply softmax before `CrossEntropyLoss`. It applies log-softmax internally and expects logits. Passing softmax outputs leads to `log(softmax(softmax(x)))` — double-squashing and incorrect gradients.

---

## Weighted Cross-Entropy and Focal Loss

**The problem**: class imbalance. If 95% of examples are class 0, the model minimizes loss by predicting class 0 for everything. The 5% minority class is ignored.

**Weighted cross-entropy**: assign higher loss weight to underrepresented classes:

$$L = -\frac{1}{m} \sum_{i} w_{k_i} \log \hat{y}_{i,k_i}, \quad w_c = \frac{N}{K \cdot N_c}$$

Scales each example's contribution inversely with class frequency. Easy examples (majority class) and hard examples (minority class) get re-weighted at the class level.

**What breaks**: class weights correct for label frequency but not for prediction difficulty. Easy majority-class examples that the model already predicts correctly still contribute loss.

**Focal Loss**: addresses this — down-weights easy examples regardless of class, up-weights hard ones:

$$L_\text{focal} = -\frac{1}{m} \sum_i \alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $p_t = \hat{y}$ if $y=1$, else $1-\hat{y}$. The factor $(1-p_t)^\gamma$ is near zero when $p_t$ is high (easy, already correct) and near one when $p_t$ is low (hard, wrong). $\gamma = 2$ is typical.

**What breaks**: focal loss requires tuning $\gamma$. Too high and the model ignores well-classified examples entirely, potentially harming calibration. It also makes training noisier because confident-wrong hard examples dominate gradients.

---

## Label Smoothing

**The problem**: a model trained with hard one-hot labels is incentivized to push logits to $\pm\infty$ — the correct class's logit should be as large as possible. This produces overconfident models that generalize poorly and are badly calibrated.

**The core insight**: soften the target distribution slightly. Instead of 100% probability on the correct class, assign $1-\epsilon$ to the correct class and $\epsilon/K$ to every other class. The model can never fully satisfy the target, preventing overconfidence.

**The mechanics**:

$$y_\text{smooth} = (1-\epsilon) \cdot y_\text{one-hot} + \frac{\epsilon}{K}$$

Typical $\epsilon = 0.1$.

```python
ce = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**What breaks**: label smoothing improves calibration but can slightly hurt accuracy on well-separated tasks where the model should be confident. It also makes it harder to identify which examples were truly ambiguous.

---

## MSE

**The problem**: for regression, you want to penalize predictions that are far from the true value. The loss should be zero when prediction is exact and grow as prediction moves away.

**The core insight**: squared error is the negative log-likelihood of a Gaussian noise model. If you believe the true value is your prediction plus Gaussian noise, MSE is the principled loss.

**The mechanics**:

$$L = \frac{1}{m} \sum_i (y_i - \hat{y}_i)^2$$

Gradient scales linearly with error: $\partial L / \partial \hat{y}_i = -2(y_i - \hat{y}_i)$. Large errors produce large gradients — the optimizer works hardest on the worst predictions.

**What breaks**: large errors produce quadratically large loss. A single outlier with $|y - \hat{y}| = 10$ contributes 100 to the loss — potentially dominating the entire batch. If your data has heavy-tailed noise or outliers, MSE trains a model that fits outliers at the expense of typical examples.

---

## MAE

**The problem**: MSE is too sensitive to outliers — a few bad examples dominate training.

**The core insight**: use absolute error. Linear penalty is robust to large errors — a $10\times$ larger error contributes only $10\times$ more loss, not $100\times$.

**The mechanics**:

$$L = \frac{1}{m} \sum_i |y_i - \hat{y}_i|$$

**What breaks**: the gradient of MAE is constant ($\pm 1$) regardless of error magnitude. Near the optimum, where errors are small, the gradient never shrinks — updates remain large even when the model is nearly correct. This causes oscillation near convergence and slow final-stage training.

---

## Huber Loss

**The problem**: MSE is too sensitive to outliers; MAE oscillates near convergence. You want both.

**The core insight**: use MSE for small errors (where its smooth gradient helps convergence) and MAE for large errors (where its linear penalty ignores outliers).

**The mechanics**:

$$L_\delta = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & |y - \hat{y}| \leq \delta \\ \delta|y - \hat{y}| - \frac{1}{2}\delta^2 & |y - \hat{y}| > \delta \end{cases}$$

Smooth at the transition point $\delta$. Default $\delta = 1$.

**What breaks**: $\delta$ must be tuned to match the scale of your residuals. If $\delta$ is too small relative to typical errors, Huber behaves like MAE everywhere. If too large, it behaves like MSE and inherits its outlier sensitivity.

| Loss | Gradient | Outlier sensitivity | Best use |
| :--- | :--- | :--- | :--- |
| **MSE** | Linear in error | High | Clean data, Gaussian noise |
| **MAE** | Constant $\pm 1$ | Low | Outlier-heavy data |
| **Huber** | Smooth hybrid | Medium | Default regression choice |

---

## Triplet Loss

**The problem**: for metric learning (face recognition, image retrieval, search ranking), you want embeddings where similar items are close and dissimilar items are far apart. Classification losses cannot express this because they only care about discrete class membership, not the geometry of the embedding space.

**The core insight**: for each training example (anchor $a$), pick a similar example (positive $p$, same identity) and a dissimilar one (negative $n$, different identity). Require the anchor-positive distance to be smaller than the anchor-negative distance by at least a margin.

**The mechanics**:

$$L = \max(0, \; d(a, p) - d(a, n) + \text{margin})$$

Loss is zero when the positive is already close enough relative to the negative. The margin prevents trivial satisfaction where $d(a,p) \approx d(a,n) \approx 0$.

**What breaks**: with random triplet selection, most triplets become easy (the negative is already far enough) after a few epochs, and the loss is zero — no gradient flows and training stalls. Hard negative mining selects the closest negatives, providing maximum learning signal, but very hard negatives can be mislabeled outliers that corrupt training. Semi-hard negatives (farther than positive but within margin) are more stable.

---

## Contrastive Loss (InfoNCE / SimCLR)

**The problem**: triplet loss requires explicit positive/negative pairs, which must be curated. In self-supervised learning, you want to learn representations without labels.

**The core insight**: create two augmented views of the same input; they should be similar. All other examples in the batch are negatives. Maximize similarity between the two views of the same input while minimizing similarity to all others.

**The mechanics**:

$$L = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

Temperature $\tau$ controls sharpness. Low $\tau$ makes the distribution peaked (hard comparisons); high $\tau$ flattens it (soft comparisons).

**What breaks**: requires a large batch to have enough negatives. With a small batch, the model can learn trivial solutions (e.g., mapping all inputs to the same point). Larger batches are essential or you need memory banks of past embeddings.

---

## KL Divergence

**The problem**: you have two probability distributions $P$ and $Q$ and want to measure how different they are — not as a distance metric, but as a measure of "extra information" needed to represent $P$ using $Q$.

**The mechanics**:

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

Zero when $P = Q$. Not symmetric: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$.

Used in:
- VAE loss: penalize encoder distribution from deviating from Gaussian prior
- Knowledge distillation: train student to match teacher's output distribution
- RLHF: KL penalty keeping policy close to reference model

**What breaks**: $D_{KL}(P \| Q) \to \infty$ if $Q(x) = 0$ for some $x$ where $P(x) > 0$. The $Q$ distribution must have support wherever $P$ does, or the divergence is undefined. In practice, add small $\epsilon$ to avoid log(0).

---

## Task-to-Loss Reference

| Task | Output activation | Loss |
| :--- | :--- | :--- |
| Binary classification | Sigmoid | `BCEWithLogitsLoss` |
| Multiclass (exclusive) | Softmax | `CrossEntropyLoss` |
| Multi-label | Sigmoid per class | `BCEWithLogitsLoss` |
| Imbalanced classification | Sigmoid | Focal Loss |
| Regression | None | MSE / Huber |
| Outlier-robust regression | None | MAE / Huber |
| Metric learning | None | Triplet / InfoNCE |
| Generative (VAE) | Sigmoid | BCE + KL divergence |
| Distillation | Softmax | KL divergence to teacher |

---

## Canonical Interview Q&As

**Q: Derive cross-entropy loss and explain its connection to KL divergence.**  
A: For a classification problem with C classes, cross-entropy loss is H(p, q) = -Σ_c p(c)·log(q(c)), where p is the true distribution (one-hot) and q is the model's predicted distribution. For one-hot p, this simplifies to -log(q(y)) — the negative log-probability of the correct class. Connection to KL divergence: KL(p||q) = Σ p·log(p/q) = Σ p·log(p) - Σ p·log(q) = H(p) - H(p,q). So cross-entropy = entropy of true labels + KL divergence. Since H(p) is constant (doesn't depend on model parameters), minimizing cross-entropy is equivalent to minimizing KL(p||q). This means cross-entropy training is MLE (for log-loss) and simultaneously minimizes the distributional distance between the model's predictions and the true label distribution. Label smoothing replaces one-hot targets with (1-ε)·one_hot + ε/C — prevents overconfident predictions by adding entropy, effectively regularizing the model to not push logits to ±∞.

**Q: When should you use MSE vs MAE as a regression loss, and what is Huber loss?**  
A: MSE = (1/n)Σ(y_i - ŷ_i)² penalizes large errors quadratically — gradients scale with error magnitude, so large outliers dominate the loss and pull the model toward fitting them. MAE = (1/n)Σ|y_i - ŷ_i| penalizes errors linearly — more robust to outliers, but gradient is constant (sign(error)) regardless of error magnitude, which makes convergence slower near the optimum (gradient doesn't decrease as prediction improves). Huber loss combines both: δ² for |error| ≤ δ (MSE regime near the optimum) and δ·(|error| - δ/2) for |error| > δ (MAE regime for large errors). The δ hyperparameter separates "small" errors (use MSE, smooth gradients) from "large" errors (use MAE, robust to outliers). Use MSE when: errors are Gaussian, no outliers expected, and the squared error is the right business metric. Use MAE when: outliers are real signal you want to ignore, median is a better target than mean. Use Huber when: mix of both — most errors are small but occasional large outliers shouldn't dominate.

**Q: What loss function would you use for a multi-label classification problem where samples can belong to multiple classes, and why?**  
A: Binary cross-entropy (BCE) applied independently per class. Unlike softmax + categorical cross-entropy (which assumes exactly one class is correct), multi-label problems require each class head to output an independent probability. Apply sigmoid (not softmax) to each logit: σ(z_c) = p(label c = 1). Loss: L = -(1/C)Σ_c [y_c·log(σ(z_c)) + (1-y_c)·log(1-σ(z_c))]. Each class is a binary classification problem. Key considerations: (1) Class imbalance is common in multi-label settings (most classes absent in most samples) — use pos_weight parameter in BCEWithLogitsLoss to upweight positive examples; (2) Label correlation — if labels are correlated (e.g., "cat" and "kitten" often co-occur), structured prediction losses (label-conditional BCE) can help; (3) Evaluation: use mean average precision (mAP) across classes, not accuracy (dominated by true negatives since most classes are absent). For extreme class imbalance (e.g., fine-grained attributes with 1% prevalence), combine focal loss with BCE: -(1-p_t)^γ·log(p_t).
