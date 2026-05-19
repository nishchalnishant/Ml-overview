# Day 1-2: Introduction to Machine Learning

## Why This Topic Comes First

Before you can study any algorithm, you need a shared vocabulary for what ML is trying to do and why it works at all. Every technique in the next 28 days is either a supervised, unsupervised, or reinforcement approach — the labels are used constantly. More importantly, the bias-variance framework and the idea of a loss function underlie every model evaluation conversation you will have. If you don't have this mental model locked in on day 1, later material will feel like a list of unrelated tricks rather than a coherent field.

---

## Executive Summary

| Topic | Core Concept | Key Algorithms | Evaluation |
|-------|--------------|----------------|------------|
| **Supervised** | Labeled data mapping $X \rightarrow y$ | Linear/Logistic Reg, Decision Trees, SVM | MSE, Accuracy, F1-Score |
| **Unsupervised** | Finding structure in $X$ | K-Means, PCA, GMM | Silhouette, Inertia, Variance |
| **Reinforcement** | Learning from environment feedback | Q-Learning, PPO | Cumulative Reward |

---

## Core Learning Paradigms

### 1. Supervised Learning

**Why this is the most important paradigm to understand first:** The entire chain — features go in, a loss is computed, weights are updated — is the template for almost everything else. Once you understand it for linear regression, neural networks are just a more complex version of the same loop.

The model learns from a trainer (labeled data). The objective is to minimize a loss function $J(\theta)$.
- **Regression**: Output is continuous. $y \in \mathbb{R}$.
- **Classification**: Output is discrete/categorical. $y \in \{0, 1, \dots, K\}$.

**Key insight:** The label $y$ is not "the truth" — it is a noisy sample from an unknown true function. The model will never recover the true function perfectly. The question is: how do we make the error on *unseen* data small, not just on training data?

**How to verify understanding:** Take a linear regression model. Without using the words "training" or "testing," explain why minimizing training MSE to zero is not the goal.

**What trips people up:** Conflating "the model learned the data" with "the model learned the pattern." A model that memorizes training labels has zero training error and potentially maximum test error. Recognizing this is the entire motivation for everything from regularization to cross-validation.

### 2. Unsupervised Learning

**Why this comes in the same session as supervised:** You need the contrast. Supervised learning has a ground truth signal to optimize against. Unsupervised learning does not — it finds structure that is only meaningful if it aligns with some downstream task. This distinction shapes how you evaluate and trust each paradigm.

- **Clustering**: Grouping similar instances.
- **Dimensionality Reduction**: Projecting high-dim data to low-dim space while preserving maximum information.

**Key insight:** There is no single correct answer in unsupervised learning. Two equally valid clusterings of the same data can exist. The "best" one depends entirely on what you're trying to use the clusters for.

**How to verify understanding:** If you run K-Means twice with different random seeds and get different cluster assignments, are both results wrong? Explain what it would mean for one to be "better."

**What trips people up:** Treating silhouette score or inertia as an objective truth. These are internal quality measures — they tell you how consistent the clustering is with its own geometry, not whether the clusters are useful for your problem.

---

## Mathematical Foundations

### 1. Linear Regression

The hypothesis $h_\theta(x)$ is a linear combination of features:
$$h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n$$
- **Cost Function (MSE)**: $J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$
- **Optimization**: Gradient Descent $\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$

**Key insight:** Gradient descent is not a special algorithm for linear regression. It is the universal procedure: compute which direction in weight space makes the loss go down, take a small step in that direction, repeat. Every optimizer you will study (SGD, Adam, RMSProp) is a variation on this one idea.

**How to verify understanding:** Derive the gradient $\frac{\partial J}{\partial \theta_j}$ for MSE from scratch. If you can do this, you understand what backpropagation is doing conceptually.

**What trips people up:** Thinking gradient descent finds the exact minimum of linear regression. It does — but only because MSE for linear regression is convex. For non-convex losses (neural networks), gradient descent finds *a* minimum, not necessarily the global one.

### 2. Logistic Regression

Predicts probabilities using the **Sigmoid Function**:
$$g(z) = \frac{1}{1 + e^{-z}}$$
Hypothesis: $h_\theta(x) = g(\theta^T x) = P(y=1|x;\theta)$
- **Loss (Binary Cross-Entropy)**: $L(y, \hat{y}) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$

**Key insight:** Logistic regression is still a linear model. The sigmoid squashes the output into $[0, 1]$, but the decision boundary (where prediction flips from 0 to 1) is still a hyperplane in feature space. If you need a curved boundary, logistic regression cannot learn it from raw features — you must engineer non-linear features first.

**How to verify understanding:** Draw a 2D dataset where logistic regression would fail no matter how you tune its weights. Then describe what feature transformation would make it solvable.

**What trips people up:** Thinking logistic regression is a classification algorithm. It is a probability estimator. The decision threshold (default 0.5) is a separate choice — and changing it changes the precision-recall tradeoff, not the model itself.

---

## Bias-Variance Tradeoff

**Why this concept belongs on day 1, not later:** Every model selection decision for the rest of the 30 days can be framed as a bias-variance question. Understanding this now means the later chapters on regularization, ensembles, and cross-validation are not separate topics — they are all answers to the same underlying problem.

**Bias** is error due to overly simplistic assumptions (underfitting). **Variance** is error due to over-sensitivity to training noise (overfitting). Linear models usually have high bias/low variance, while deep trees have low bias/high variance.

**Key insight:** The goal is not to minimize training error. It is to minimize *expected test error*, which equals bias² + variance + irreducible noise. These two components move in opposite directions as model complexity increases. There is no free lunch.

**How to verify understanding:** You have a model with 100% training accuracy and 60% test accuracy. A colleague says "just make the model more complex." Is this the right move? Explain why or why not.

**What trips people up:** Treating bias and variance as properties of the model alone. They are properties of the model *given a dataset size*. The same architecture can overfit with 100 samples and underfit with 1 million. This matters when someone asks you how much data you need.

---

## Interview Questions

**1. "What is the difference between Bias and Variance?"**
> **Bias** is error due to overly simplistic assumptions (underfitting). **Variance** is error due to over-sensitivity to training noise (overfitting). Linear models usually have high bias/low variance, while deep trees have low bias/high variance.

**2. "Why use Log-Loss instead of MSE for Logistic Regression?"**
> MSE for logistic regression results in a non-convex cost function, leading to multiple local minima. Log-loss is convex, ensuring gradient descent converges to the global minimum.

**3. "Can Linear Regression be used for classification?"**
> Technically yes, by thresholding the output, but it's risky because linear regression is sensitive to outliers and doesn't output probabilities bounded by $[0, 1]$.

---

## Implementation Quick-Lookup

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# Linear Regression
reg = LinearRegression().fit(X_train, y_train)
preds = reg.predict(X_test)

# Logistic Regression
clf = LogisticRegression().fit(X_train, y_train)
probs = clf.predict_proba(X_test)
```
