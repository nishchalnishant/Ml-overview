# Loss Functions

Loss functions tell the model what "wrong" means.

The optimizer can only improve what the loss actually measures. Choose poorly and you train the wrong thing.

---

# 1. Classification Losses

## Binary Cross-Entropy (BCE)

$$L = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

- Use with sigmoid output
- Penalizes confident-wrong predictions heavily (log goes to $-\infty$ as probability → 0)
- For multi-label tasks: apply BCE independently per class

## Multiclass Cross-Entropy (Categorical CE)

$$L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

- Use with softmax output
- Only the correct class contributes to the loss (one-hot $y_{ik}$ zeros out other terms)
- PyTorch `nn.CrossEntropyLoss` = log-softmax + NLLLoss (takes raw logits, not softmax outputs)

```python
import torch
import torch.nn as nn

# Binary
bce = nn.BCEWithLogitsLoss()          # numerically stable, takes logits
loss = bce(logits, targets.float())

# Multiclass — DO NOT apply softmax before this
ce = nn.CrossEntropyLoss()
loss = ce(logits, targets)            # targets are class indices, not one-hot
```

## Weighted Cross-Entropy

Assign higher loss weight to underrepresented classes:

$$L = -\frac{1}{m} \sum_{i=1}^{m} w_{y_i} \log(\hat{y}_{i, y_i})$$

where $w_c$ is the weight for class $c$. Common choice: $w_c = 1/\text{freq}(c)$ or $w_c = N / (K \cdot N_c)$.

```python
# Inverse frequency weighting
class_counts = torch.tensor([800, 150, 50], dtype=torch.float)
weights = 1.0 / class_counts
weights = weights / weights.sum()   # normalize
ce = nn.CrossEntropyLoss(weight=weights)
```

## Focal Loss

Designed for severe class imbalance (e.g., object detection where background >> objects):

$$L_{\text{focal}} = -\frac{1}{m} \sum_i \alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $p_t = \hat{y}$ if $y=1$ else $1 - \hat{y}$.

- $(1 - p_t)^\gamma$: **modulating factor** — down-weights easy examples (high $p_t$), up-weights hard ones
- $\gamma = 0$: reduces to standard BCE; $\gamma = 2$ is typical
- $\alpha_t$: class weighting for additional imbalance correction

## Label Smoothing

Prevents overconfidence by softening hard targets:

$$y_{\text{smooth}} = (1 - \epsilon) \cdot y_{\text{one-hot}} + \frac{\epsilon}{K}$$

Typical $\epsilon = 0.1$. Regularizes output distribution; improves calibration.

```python
ce_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

# 2. Regression Losses

## Mean Squared Error (MSE)

$$L = \frac{1}{m} \sum_i (y_i - \hat{y}_i)^2$$

- Heavily penalizes large errors ($O(e^2)$)
- Gradient is linear in error → nice optimization properties
- Sensitive to outliers

## Mean Absolute Error (MAE)

$$L = \frac{1}{m} \sum_i |y_i - \hat{y}_i|$$

- Robust to outliers (linear penalty)
- Gradient is constant ($\pm 1$) — can slow convergence near optimum
- Better when outliers are real signal that should not dominate training

## Huber Loss

$$L_\delta = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & |y - \hat{y}| \leq \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & |y - \hat{y}| > \delta \end{cases}$$

- MSE for small errors, MAE for large ones
- Smooth at the transition point
- $\delta$ controls the boundary (default $\delta = 1$)

## Comparison

| Loss | Gradient | Outlier sensitivity | Use case |
| :--- | :--- | :--- | :--- |
| **MSE** | Linear in error | High | Clean data, large errors matter |
| **MAE** | Constant $\pm 1$ | Low | Noisy/outlier-heavy data |
| **Huber** | Smooth hybrid | Medium | Default regression choice |

---

# 3. Contrastive and Embedding Losses

## Triplet Loss

Learn embeddings where similar items are close, dissimilar items are far:

$$L = \max\left(0, d(a, p) - d(a, n) + \text{margin}\right)$$

where $a$ = anchor, $p$ = positive (same class), $n$ = negative (different class), $d$ = distance.

where $a$ = anchor, $p$ = positive (same class), $n$ = negative (different class), $d$ = Euclidean distance or $1 - \cos$.

**Margin guidance:**
- Margin too small → trivial triplets dominate, little learning signal
- Margin too large → loss never reaches zero, noisy updates
- Typical: 0.2–1.0 for L2 distance, 0.1–0.3 for cosine distance
- Use **hard negative mining**: select negatives that are closest to the anchor (hardest triplets) → faster convergence but training instability risk. Semi-hard negatives (farther than positive but within margin) are more stable.

```python
triplet_loss = nn.TripletMarginLoss(margin=0.5, p=2)  # p=2: Euclidean
loss = triplet_loss(anchor, positive, negative)
```

Used in: face recognition, metric learning, search ranking.

## Contrastive Loss (SimCLR-style)

$$L = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

where $\tau$ is temperature (controls sharpness of distribution). Self-supervised learning standard.

## Cosine Embedding Loss

$$L = \begin{cases} 1 - \cos(x_1, x_2) & y = 1 \text{ (similar)} \\ \max(0, \cos(x_1, x_2) - \text{margin}) & y = -1 \text{ (dissimilar)} \end{cases}$$

---

# 4. KL Divergence

Measures how one probability distribution $P$ differs from reference $Q$:

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

Not symmetric: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$.

Used in:
- VAE loss (penalize deviation of latent distribution from standard Normal)
- Knowledge distillation (match student to teacher distribution)
- RL policy optimization (KL penalty for staying close to reference policy)

---

# 5. Task → Loss Matching

| Task | Output activation | Loss |
| :--- | :--- | :--- |
| Binary classification | Sigmoid | BCE / BCEWithLogitsLoss |
| Multiclass (exclusive) | Softmax | CrossEntropyLoss |
| Multi-label | Sigmoid per class | BCEWithLogitsLoss |
| Imbalanced classification | Sigmoid | Focal Loss |
| Regression | None | MSE / Huber |
| Robust regression | None | MAE / Huber |
| Metric learning | None | Triplet / Contrastive |
| Generative (VAE) | Sigmoid (reconstruction) | BCE + KL divergence |

If the loss and output activation do not match the task, training quality suffers fast.

---

# 6. Numerical Stability

**Never compute `log(softmax(x))` directly** — underflow when probabilities are tiny.

Use `log_softmax` instead (subtracts max for stability):

```python
# Unstable
loss = -torch.log(torch.softmax(logits, dim=-1)[range(B), targets])

# Stable
loss = F.cross_entropy(logits, targets)         # PyTorch handles this internally
# or equivalently
log_probs = F.log_softmax(logits, dim=-1)
loss = F.nll_loss(log_probs, targets)
```

**BCE with logits vs BCE with probs:**
```python
# Stable (preferred)
F.binary_cross_entropy_with_logits(logits, targets)

# Unstable when probabilities are near 0 or 1
F.binary_cross_entropy(torch.sigmoid(logits), targets)
```
