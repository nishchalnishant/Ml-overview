---
module: Foundations
topic: Information Theory
subtopic: ""
status: unread
tags: [foundations, information-theory, entropy, kl-divergence, mutual-information, cross-entropy, ml-math]
---
# Information Theory for Machine Learning

**Why this matters:** Every loss function in ML is secretly an information-theoretic quantity. Cross-entropy loss IS KL divergence. Maximum likelihood IS minimizing cross-entropy. The ELBO in VAEs IS mutual information minus divergence. Understanding this unifies all of ML.

**What you'll learn:**
- Shannon entropy — the foundation of everything
- KL divergence — the "distance" between distributions
- Cross-entropy — why it's the standard ML loss
- Mutual information — measuring statistical dependence
- How information theory connects to ML training objectives

---

## Table of Contents
1. [Shannon Entropy](#1-shannon-entropy)
2. [Joint, Conditional, and Marginal Entropy](#2-joint-conditional-and-marginal-entropy)
3. [KL Divergence](#3-kl-divergence)
4. [Cross-Entropy Loss — Derived from First Principles](#4-cross-entropy-loss)
5. [Mutual Information](#5-mutual-information)
6. [Information Theory in ML — The Unified View](#6-information-theory-in-ml)
7. [Bits vs. Nats vs. Hartleys](#7-bits-vs-nats)
8. [Interview Questions](#8-interview-questions)
9. [Quick Reference Table](#9-quick-reference)

---

## 1. Shannon Entropy

### What is it?

**Entropy measures the average surprise (uncertainty) in a distribution.**

If you always roll 6 on a die, there's no surprise — entropy = 0. If the die is fair, every outcome surprises you equally — entropy is maximum.

**Formal definition:**

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x)$$

For continuous distributions (differential entropy):

$$H(X) = -\int p(x) \log p(x)\, dx$$

**Convention:** when using $\log_2$, entropy is in **bits**. When using $\ln$, entropy is in **nats**.

### Intuition: Why the Negative Log?

The "surprise" of event $x$ with probability $p(x)$ should be:
- **Large** when $p(x)$ is small (a rare event surprises us)
- **Zero** when $p(x) = 1$ (a certain event surprises no one)
- **Additive** for independent events

Only $-\log p(x)$ satisfies all three. Entropy is the **expected surprise**:

$$H(X) = \mathbb{E}_{x \sim p}[-\log p(x)]$$

### Example — Coin Flips

```
Fair coin (p=0.5):
H = -(0.5 log₂ 0.5 + 0.5 log₂ 0.5) = -(-0.5 - 0.5) = 1 bit

Biased coin (p=0.9 heads):
H = -(0.9 log₂ 0.9 + 0.1 log₂ 0.1)
  = -(0.9 × -0.152 + 0.1 × -3.32)
  = -(−0.137 − 0.332)
  ≈ 0.469 bits

Always heads (p=1.0):
H = -(1.0 × log₂ 1.0) = 0 bits (no surprise)
```

Entropy is **maximized** when the distribution is uniform (maximum uncertainty).

### Properties

| Property | Statement | Why it matters |
|---|---|---|
| **Non-negative** | $H(X) \geq 0$ | Uncertainty can't be negative |
| **Maximum** | $H$ is maximized by the uniform distribution | Uniform = most uncertain |
| **Additive** | $H(X, Y) = H(X) + H(Y)$ if $X \perp Y$ | Independent sources add uncertainty |
| **Chain rule** | $H(X, Y) = H(X) + H(Y|X)$ | Decompose joint entropy |

### Code: Computing Entropy

```python
import numpy as np
from scipy.stats import entropy

def shannon_entropy(probs, base=2):
    """
    Compute Shannon entropy of a discrete distribution.
    
    Args:
        probs: array of probabilities (must sum to 1)
        base: 2 for bits, np.e for nats
    Returns:
        H(X) in the specified units
    """
    probs = np.array(probs)
    # Filter out zero probabilities (0 * log(0) = 0 by convention)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs) / np.log(base))

# Examples
print(shannon_entropy([0.5, 0.5]))      # 1.0 bit (fair coin)
print(shannon_entropy([0.9, 0.1]))      # ~0.469 bits
print(shannon_entropy([1.0]))           # 0.0 bits (certain)
print(shannon_entropy([0.25]*4))        # 2.0 bits (uniform over 4 outcomes)

# Using scipy (nats by default, pass qk for cross-entropy)
entropy([0.5, 0.5], base=2)             # same as above
```

---

## 2. Joint, Conditional, and Marginal Entropy

### Joint Entropy

The entropy of two random variables together:

$$H(X, Y) = -\sum_{x,y} p(x, y) \log p(x, y)$$

**Intuition:** Total uncertainty in the joint system.

### Conditional Entropy

The entropy of $Y$ given that we know $X$:

$$H(Y|X) = -\sum_{x,y} p(x, y) \log p(y|x) = \mathbb{E}_{x \sim p(x)}[H(Y|X=x)]$$

**Intuition:** How much uncertainty remains in $Y$ after observing $X$? If $X$ perfectly predicts $Y$, then $H(Y|X) = 0$.

### Chain Rule of Entropy

$$H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$

This is the entropy version of the chain rule of probability. It says: joint entropy = entropy of one variable + remaining entropy of the other given the first.

### Diagram

```
H(X, Y)  ┌─────────────────────────┐
          │  H(X)     │  H(Y|X)    │
          │           │            │
          │  H(X|Y)   │  H(Y)     │
          └─────────────────────────┘
              ↑ overlap = I(X;Y) = mutual information
```

---

## 3. KL Divergence

### What is it?

**KL divergence measures how different distribution $Q$ is from a reference distribution $P$.**

$$D_{KL}(P \| Q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim P}\left[\log \frac{p(x)}{q(x)}\right]$$

For continuous distributions:

$$D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)}\, dx$$

### Intuition

Think of $P$ as the **true distribution** and $Q$ as your **model's distribution**. KL divergence tells you: *how many extra bits per sample do you need if you encode data from $P$ using a code optimized for $Q$?*

If $P = Q$: $D_{KL}(P \| Q) = 0$ — perfect model, no wasted bits.  
If $P \neq Q$: $D_{KL}(P \| Q) > 0$ — some bits are wasted.

### Critical Properties

| Property | Explanation |
|---|---|
| **Non-negative** | $D_{KL}(P\|Q) \geq 0$, with equality iff $P = Q$ (Gibbs' inequality) |
| **Not symmetric** | $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ — it's NOT a distance metric |
| **Zero-avoiding** | $D_{KL}(P\|Q) = \infty$ if $Q(x) = 0$ but $P(x) > 0$ |
| **Forward vs reverse** | Forward KL: $D_{KL}(P\|Q)$ → mean-seeking. Reverse: $D_{KL}(Q\|P)$ → mode-seeking |

### Forward vs. Reverse KL — Why It Matters

```
Data distribution P: bimodal (two peaks)
Model distribution Q: unimodal (one peak)

Forward KL D_KL(P||Q):    Reverse KL D_KL(Q||P):
Q tries to cover all of P  Q picks ONE mode of P
→ places mass between       → mode-seeking, zero-avoiding
  modes (mean-seeking)      → used in VAEs (approximate posterior)
→ used in MLE training
```

**Forward KL** (minimized in MLE) → model is "inclusive," puts probability mass everywhere $P$ has mass. Can produce blurry/mean predictions.

**Reverse KL** (minimized in variational inference) → model is "exclusive," sharply covers one region. Can miss modes.

### Code: Computing KL Divergence

```python
import numpy as np
from scipy.stats import entropy as scipy_kl

def kl_divergence(p, q, eps=1e-10):
    """
    Forward KL: D_KL(P || Q).
    
    Args:
        p: true distribution (reference)
        q: approximate distribution (model)
        eps: small constant to avoid log(0)
    """
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    # Clip to avoid log(0)
    q = np.clip(q, eps, None)
    p = np.clip(p, eps, None)
    # Only sum where p > 0 (0 * log(0/q) = 0 by convention)
    return np.sum(p * np.log(p / q))

# Example: two distributions
p = [0.4, 0.4, 0.1, 0.1]  # true
q = [0.25, 0.25, 0.25, 0.25]  # uniform approximation

print(f"KL(P||Q) = {kl_divergence(p, q):.4f}")  # > 0
print(f"KL(Q||P) = {kl_divergence(q, p):.4f}")  # different! asymmetric
print(f"KL(P||P) = {kl_divergence(p, p):.4f}")  # = 0

# scipy equivalent (nats)
scipy_kl(p, q)  # same as D_KL(P||Q)
```

---

## 4. Cross-Entropy Loss

### Derivation from First Principles

**This is the most important derivation in ML.**

**Setting:** We have true labels $y$ (empirical distribution $P$) and model predictions $\hat{y}$ (model distribution $Q$). We want to train the model.

**Step 1: Maximum Likelihood Estimation**

Maximum likelihood says: find the model parameters $\theta$ that maximize the probability of observed data:

$$\theta^* = \arg\max_\theta \prod_{i=1}^{N} Q_\theta(y_i)$$

**Step 2: Convert to log-likelihood (log makes products into sums)**

$$\theta^* = \arg\max_\theta \sum_{i=1}^{N} \log Q_\theta(y_i)$$

**Step 3: Recognize this as cross-entropy**

The negative average log-likelihood over N samples:

$$-\frac{1}{N} \sum_{i=1}^{N} \log Q_\theta(y_i) = \mathbb{E}_{x \sim P}[-\log Q_\theta(x)] = H(P, Q)$$

This is the **cross-entropy** between the empirical distribution $P$ and the model $Q$.

**Step 4: Connect to KL divergence**

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

Since $H(P)$ (entropy of the true labels) is constant w.r.t. $\theta$:

$$\arg\min_\theta H(P, Q) = \arg\min_\theta D_{KL}(P \| Q)$$

**Conclusion: Minimizing cross-entropy loss = minimizing KL divergence from the model to the data = maximum likelihood estimation. All three are the same thing.**

### Cross-Entropy for Classification

For binary classification with label $y \in \{0, 1\}$ and predicted probability $\hat{p}$:

$$\mathcal{L}_{BCE} = -[y \log \hat{p} + (1-y) \log(1-\hat{p})]$$

For multiclass with $K$ classes and one-hot label $y$, model softmax $\hat{p}$:

$$\mathcal{L}_{CE} = -\sum_{k=1}^{K} y_k \log \hat{p}_k$$

Since only the true class $k^*$ has $y_{k^*} = 1$, this simplifies to:

$$\mathcal{L}_{CE} = -\log \hat{p}_{k^*}$$

**Intuition:** Penalize the model proportionally to the surprise of the correct answer.

### Code: Cross-Entropy Loss

```python
import numpy as np
import torch
import torch.nn as nn

def cross_entropy_from_scratch(y_true, y_pred_probs, eps=1e-10):
    """
    Cross-entropy for multiclass classification.
    
    Args:
        y_true: integer class labels (batch_size,)
        y_pred_probs: softmax probabilities (batch_size, num_classes)
    """
    n = len(y_true)
    # Index the probability of the correct class for each sample
    correct_probs = y_pred_probs[np.arange(n), y_true]
    # Clip to avoid log(0)
    correct_probs = np.clip(correct_probs, eps, 1.0)
    return -np.mean(np.log(correct_probs))

# Example
y_true = np.array([0, 1, 2])
y_pred = np.array([
    [0.9, 0.05, 0.05],   # confident and correct → low loss
    [0.1, 0.8, 0.1],     # fairly confident and correct
    [0.3, 0.3, 0.4],     # barely correct → higher loss
])
print(cross_entropy_from_scratch(y_true, y_pred))  # ~0.39

# PyTorch equivalent (takes logits, not probs)
criterion = nn.CrossEntropyLoss()
logits = torch.tensor([[2.2, -2., -2.],
                       [-1., 1.6, -1.],
                       [-0.4, -0.4, 0.5]])
labels = torch.tensor([0, 1, 2])
loss = criterion(logits, labels)
print(loss.item())  # similar to above
```

---

## 5. Mutual Information

### What is it?

**Mutual information $I(X; Y)$ measures how much knowing $X$ reduces uncertainty about $Y$ — and vice versa.**

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

It can also be written as:

$$I(X; Y) = D_{KL}(p(x,y) \| p(x)p(y)) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

If $X$ and $Y$ are independent, $p(x,y) = p(x)p(y)$, so $I(X;Y) = 0$ (knowing $X$ tells you nothing about $Y$).

### Venn Diagram of Information Measures

```
        H(X)         H(Y)
    ┌──────────┐   ┌──────────┐
    │          │   │          │
    │  H(X|Y)  │I(X│Y)  H(Y|X)│
    │          │   │          │
    └──────────┴───┴──────────┘
                ↑
          I(X; Y) = overlap = shared information
```

$$H(X, Y) = H(X) + H(Y) - I(X; Y)$$

### Applications in ML

| Application | How MI is used |
|---|---|
| **Feature selection** | Select features $X_i$ that maximize $I(X_i; Y)$ with the label |
| **InfoNCE / contrastive loss** | Maximize lower bound on $I(X; X^+)$ (same image, different augment) |
| **Information Bottleneck** | Learn representation $Z$ that maximizes $I(Z; Y)$ while minimizing $I(X; Z)$ |
| **VAE ELBO** | ELBO ≥ $I(X; Z)$ — encoder maximizes info in the latent code |
| **MINE** | Neural MI estimator for high-dimensional spaces |

### Code: Mutual Information Estimation

```python
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_iris
import numpy as np

# MI between features and labels (classification)
X, y = load_iris(return_X_y=True)
mi_scores = mutual_info_classif(X, y, random_state=42)

for feature, score in zip(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], mi_scores):
    print(f"{feature}: MI = {score:.3f}")

# petal_length and petal_width will have highest MI with species
# sepal_width will have lowest (least discriminative)
```

---

## 6. Information Theory in ML — The Unified View

```
Training Objective              Information Theory View
─────────────────────────────────────────────────────────
Cross-Entropy Loss          →   H(P_data, P_model)
Maximum Likelihood          →   Minimize D_KL(P_data || P_model)
KL term in VAE ELBO         →   D_KL(q(z|x) || p(z))
ELBO (Evidence Lower Bound) →   -D_KL(q||p) + E[log p(x|z)]
InfoNCE (contrastive loss)  →   Lower bound on I(X; X+)
Information Bottleneck      →   max I(Z;Y) s.t. min I(X;Z)
Label Smoothing             →   Add entropy to target distribution
Temperature in softmax      →   Scales the "confidence" / entropy
Perplexity of a LM          →   exp(H(P_data, P_model)) in bits
```

### Perplexity

Perplexity is the standard evaluation metric for language models:

$$\text{Perplexity}(P, Q) = 2^{H(P, Q)} = 2^{-\frac{1}{N}\sum_i \log_2 Q(x_i)}$$

**Intuition:** "How many equally likely words could the model be choosing from at each step?" Lower = better. A perplexity of 10 means the model is as uncertain as if it were choosing uniformly from 10 words at each step.

For a bigram model on English, perplexity is ~240. Modern LLMs achieve <10 on held-out text.

### The ELBO Derivation (VAE)

The ELBO (Evidence Lower BOund) is:

$$\log p(x) \geq \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{reconstruction}} - \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{regularization}}$$

- **Reconstruction term:** Cross-entropy between generated and real data
- **KL term:** Keep the latent space close to the prior (Gaussian)
- Together: maximize mutual information $I(X; Z)$ while regularizing the prior

---

## 7. Bits vs. Nats

| Unit | Log base | Used in |
|---|---|---|
| **Bits** (binary digits) | $\log_2$ | Compression, data coding, perplexity |
| **Nats** (natural) | $\ln$ | ML training losses (PyTorch/TF use ln) |
| **Hartleys** (digits) | $\log_{10}$ | Rare in ML |

**Conversion:** 1 nat = $\log_2(e) \approx 1.443$ bits

PyTorch's `nn.CrossEntropyLoss` uses natural log (nats). When you see "perplexity" reported, it's usually $e^{\text{loss}}$ (from nats) or $2^{\text{loss}}$ (from bits). Always check which convention is used.

---

## 8. Interview Questions

**Q: Why do we use cross-entropy loss instead of MSE for classification?**  
*A:* Cross-entropy is derived from maximum likelihood for categorical distributions. MSE assumes Gaussian noise (correct for regression). For classification: (1) CE naturally handles probabilities summing to 1, (2) CE gradient is proportional to the prediction error (nice property), (3) MSE saturates when predictions are near 0 or 1, causing vanishing gradients. CE gradients remain strong even for wrong confident predictions.

**Q: What is the relationship between KL divergence and cross-entropy?**  
*A:* $H(P, Q) = H(P) + D_{KL}(P \| Q)$. Cross-entropy = entropy of the true distribution + the KL divergence between true and model. Since $H(P)$ is fixed during training, minimizing cross-entropy is equivalent to minimizing KL divergence. In MLE, we minimize cross-entropy because we want the model distribution to match the data distribution — which is exactly what minimizing $D_{KL}(P_{data} \| P_{model})$ achieves.

**Q: Why is KL divergence not symmetric? When does this matter?**  
*A:* $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ because: forward KL penalizes placing zero probability where P has mass (mean-seeking: the model tries to cover all modes). Reverse KL penalizes placing probability where P has no mass (mode-seeking: the model picks one mode). In VAEs, we minimize $D_{KL}(q(z|x) \| p(z))$ (reverse KL in latent space) to prevent the approximate posterior from extending into low-prior-probability regions.

**Q: What is mutual information and how is it used in self-supervised learning?**  
*A:* MI measures how much knowing one variable reduces uncertainty about another. In contrastive learning (SimCLR, MoCo), InfoNCE loss maximizes a lower bound on $I(X; X^+)$ where $X^+$ is an augmented view of $X$. The model learns representations where augmentations of the same image share high mutual information, i.e., are close in embedding space.

**Q: What is the entropy of a model's output distribution and why does it matter?**  
*A:* A model with low-entropy outputs is confident (peaked distribution). High-entropy = uncertain/calibrated. In practice: (1) Label smoothing adds a small amount of entropy to targets, preventing overconfident predictions; (2) Temperature scaling at inference controls entropy — lower temperature → lower entropy → sharper predictions; (3) In RLHF, a KL penalty between the policy and reference model prevents the policy's output distribution from collapsing (maintaining healthy entropy).

**Q: Why does perplexity matter for LLM evaluation?**  
*A:* Perplexity is $\exp(H(P_{data}, P_{model}))$ — the exponentiated cross-entropy. It measures how "surprised" the model is, on average, by held-out text. Lower is better. Key caveat: perplexity is not directly comparable across models with different tokenizers (different vocabulary sizes change the scale). BPE with 50K tokens vs 100K tokens will give different perplexities even for identical models. Always compare perplexity on the same tokenization.

---

## 9. Quick Reference Table

| Quantity | Formula | Intuition | Used for |
|---|---|---|---|
| $H(X)$ | $-\sum p \log p$ | Average surprise | Entropy of a dist. |
| $H(X\|Y)$ | $H(X,Y) - H(Y)$ | Surprise after knowing Y | Conditional uncertainty |
| $H(P, Q)$ | $-\sum p \log q$ | Bits to encode P using Q | **Cross-entropy loss** |
| $D_{KL}(P\|Q)$ | $\sum p \log(p/q)$ | Extra bits vs optimal | **Training objective** |
| $I(X;Y)$ | $H(X) - H(X\|Y)$ | Shared information | **Feature selection, contrastive** |
| Perplexity | $\exp(H(P,Q))$ | Avg branching factor | **LLM evaluation** |

**Key identity (memorize this):**

$$\boxed{H(P, Q) = H(P) + D_{KL}(P \| Q)}$$

This is why minimizing cross-entropy = minimizing KL divergence = maximum likelihood. Three names for the same optimization.

---

## Further Reading

- **Shannon (1948)** — "A Mathematical Theory of Communication" — the original paper
- **Cover & Thomas** — "Elements of Information Theory" — the definitive textbook
- **Goodfellow et al.** — "Deep Learning" Ch. 3 — probability + information theory for ML
- → [Math and Theory Foundations](02-math-and-theory-foundations.md) — linear algebra, calculus, probability
- → [Loss Functions](../03-deep-learning/components/05-loss-functions.md) — cross-entropy, focal loss, others in practice
- → [Bayesian Methods](../02-classical-ml/15-bayesian-methods.md) — KL divergence in variational inference
