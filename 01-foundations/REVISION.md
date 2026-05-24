---
module: Foundations
topic: Revision Card
subtopic: ""
status: unread
tags: [foundations, math, revision, cheatsheet]
---
# Foundations — 10-Minute Revision Card

**Read this before any other section.** These are the primitives everything else builds on.

---

## Linear Algebra Essentials

| Concept | Key fact | Why it matters |
|---------|----------|----------------|
| Matrix multiply | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$ | Neural net forward pass is matrix multiplication |
| Transpose | $(AB)^T = B^T A^T$ | Backprop gradient derivations |
| Eigendecomposition | $Av = \lambda v$ | PCA, understanding covariance |
| SVD | $A = U\Sigma V^T$ | Dimensionality reduction, LoRA |
| Dot product | $a \cdot b = \|a\|\|b\|\cos\theta$ | Attention similarity, cosine similarity |

**Gotcha:** Matrix multiplication is not commutative — $AB \neq BA$ in general.

---

## Calculus & Gradients

**Chain rule (the engine of backprop):**
$$\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}$$

**Gradient:** vector of partial derivatives. Points in direction of steepest ascent. Gradient descent steps opposite: $\theta \leftarrow \theta - \alpha \nabla_\theta L$

**Partial derivative intuition:** how much does the output change if I wiggle only this one input? Gradient = all partial derivatives simultaneously.

**Why it matters:** every training step computes the gradient of loss with respect to all parameters and steps in the negative direction.

---

## Probability Essentials

| Concept | Definition | Key use |
|---------|-----------|---------|
| Conditional probability | $P(A|B) = \frac{P(A \cap B)}{P(B)}$ | Naive Bayes, Bayes' theorem |
| Bayes' theorem | $P(H|E) = \frac{P(E|H)P(H)}{P(E)}$ | Bayesian inference, priors |
| Expectation | $\mathbb{E}[X] = \sum_x x \cdot P(X=x)$ | Loss functions, risk |
| Variance | $\text{Var}(X) = \mathbb{E}[(X - \mu)^2]$ | Bias-variance tradeoff |
| KL Divergence | $D_{KL}(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}$ | Cross-entropy loss, RLHF KL penalty |

**Cross-entropy:** $H(P,Q) = -\sum_x P(x)\log Q(x)$ = $H(P) + D_{KL}(P\|Q)$. Minimizing cross-entropy ≡ minimizing KL divergence from true distribution.

---

## Statistics Fast Reference

**Distributions to know:**

| Distribution | Params | Use in ML |
|-------------|--------|----------|
| Normal $\mathcal{N}(\mu,\sigma^2)$ | mean, variance | Weight init, noise modeling |
| Bernoulli | p | Binary classification output |
| Categorical | $p_1,...,p_k$ | Softmax output |
| Uniform | a, b | Random initialization range |

**Central Limit Theorem:** sample means of any distribution converge to normal as n→∞. Why it matters: justifies Gaussian assumptions in many ML methods.

**p-value:** probability of seeing this result (or more extreme) if null hypothesis is true. Threshold (0.05) is arbitrary. Low p-value = strong evidence against null, not proof of effect size.

**Confidence interval:** if repeated infinitely, 95% of computed CIs would contain true parameter. Does NOT mean 95% probability that this specific interval contains it.

---

## Optimization Fundamentals

**Gradient descent:**
$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(\theta_t)$$

**Variants:**
- **Batch GD:** full dataset per step — accurate gradient, slow per update
- **Stochastic GD (SGD):** one example per step — noisy gradient, fast, can escape local minima
- **Mini-batch GD:** k examples per step — balance of accuracy and speed; standard in practice

**Learning rate too high:** loss oscillates or diverges. Too low: very slow convergence. **Warmup:** start low, ramp up, then decay (cosine schedule).

**Convex vs non-convex:** convex loss → GD finds global minimum. Neural nets are non-convex → GD finds good local minimum (in practice, sufficient).

---

## Information Theory in 2 Minutes

**Entropy:** $H(X) = -\sum_x P(x)\log P(x)$ — expected bits to encode a message. Higher entropy = more uncertainty.

**Cross-entropy loss:** $\mathcal{L} = -\sum_c y_c \log \hat{y}_c$ — how many bits needed to encode true labels with predicted distribution. Minimizing it aligns predictions with truth.

**KL divergence:** non-symmetric distance between distributions. $D_{KL}(P\|Q) \geq 0$, equals 0 only when P=Q. Used in VAEs, RLHF KL penalty, knowledge distillation.

---

## Core ML Concepts Map

```
Problem Definition
    ↓
Data (collection, cleaning, features)
    ↓
Model (hypothesis class)
    ↓
Loss Function (what to minimize)
    ↓
Optimizer (how to minimize it)
    ↓
Evaluation (does it generalize?)
    ↓
Deployment (does it work in the real world?)
```

**Every ML algorithm is an instantiation of this loop.**

---

## Key Terminology Quick Reference

| Term | One-line definition |
|------|-------------------|
| Hypothesis class | Set of all functions the model can represent |
| Generalization | Performance on unseen data from same distribution |
| Distribution shift | Test data comes from different distribution than training |
| Inductive bias | Assumptions baked into model architecture |
| Expressivity | How complex a function the model can approximate |
| Sample complexity | How many examples needed to learn a concept |
| No free lunch | No algorithm works best on all problems |

---

## "Explain to a 5-year-old" Templates

**What is a neural network?**
→ Layers of linear transformations alternated with non-linearities. Each layer learns to detect patterns that the next layer combines. The whole thing is trained end-to-end by gradient descent.

**What is gradient descent?**
→ Compute how wrong you are (loss). Compute which direction makes it less wrong (gradient). Take a small step in that direction. Repeat millions of times.

**What is overfitting?**
→ The model memorized the training set including its noise, rather than learning the underlying pattern. It performs well on training data, poorly on new data.

**What is the curse of dimensionality?**
→ In high dimensions, all points are far apart. Distance metrics lose meaning. You need exponentially more data to cover the space. Feature selection and dimensionality reduction fight this.
