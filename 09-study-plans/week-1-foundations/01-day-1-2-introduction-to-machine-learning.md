---
module: Study Plans
topic: Week 1 Foundations
subtopic: Day 1 2 Introduction To Machine Learning
status: unread
tags: [studyplans, ml, week-1-foundations-day-1-2-int]
---
# Day 1-2: Introduction to Machine Learning

## Why This Topic Comes First

Every technique in this plan is supervised, unsupervised, or reinforcement learning — these labels are used constantly. The bias-variance framework and loss functions underlie every model evaluation conversation. Get this mental model right on day 1 or later material will feel like unrelated tricks.

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

The model learns from labeled data by minimizing a loss function $J(\theta)$. Features go in, a loss is computed, weights are updated — this is the template for almost everything else, including neural networks.
- **Regression**: Output is continuous. $y \in \mathbb{R}$.
- **Classification**: Output is discrete/categorical. $y \in \{0, 1, \dots, K\}$.

**Key insight:** The label $y$ is a noisy sample from an unknown true function, not "the truth." The model will never recover that function perfectly — the goal is to make error on *unseen* data small, not just training data.

**How to verify understanding:** Explain why minimizing training MSE to zero is not the goal, without using the words "training" or "testing."

**What trips people up:** Conflating "the model learned the data" with "the model learned the pattern." A model that memorizes training labels has zero training error and potentially maximum test error — the motivation behind regularization and cross-validation.

### 2. Unsupervised Learning

Supervised learning optimizes against a ground truth signal; unsupervised learning does not — it finds structure that's only meaningful if it aligns with a downstream task.

- **Clustering**: Grouping similar instances.
- **Dimensionality Reduction**: Projecting high-dim data to low-dim space while preserving maximum information.

**Key insight:** There is no single correct answer in unsupervised learning. Two equally valid clusterings of the same data can exist — the "best" one depends on what you're using the clusters for.

**How to verify understanding:** If K-Means with different random seeds gives different cluster assignments, are both results wrong? What would make one "better"?

**What trips people up:** Treating silhouette score or inertia as objective truth. These measure how consistent a clustering is with its own geometry, not whether the clusters are useful for your problem.

### 3. Reinforcement Learning

RL is the third paradigm, mentioned here to complete the landscape — not to study in depth yet. It becomes directly relevant in week 4 (RLHF, the training method that turns a pre-trained LLM into a helpful assistant).

An **agent** interacts with an **environment** by taking **actions** and observing **rewards**. There are no labeled examples — the signal is the scalar reward after each action.

- **Policy**: The agent's strategy — maps states to actions.
- **Value function**: Estimates the expected future reward from a state.
- **Exploration vs. Exploitation**: Try new actions vs. stick with known good ones.

**Key insight:** Feedback in RL is delayed and sparse, unlike supervised learning. A sequence of 100 actions might yield one reward at the end, and the agent must figure out which actions were responsible (credit assignment problem).

**How to verify understanding:** A chess RL agent gets +1 for winning, -1 for losing, 0 otherwise, over a 40-move game. Describe the credit assignment problem here.

**What trips people up:** Thinking RL is only for games. It also appears in RLHF for LLMs, recommendation systems (contextual bandits), robotics, ad bidding, and network routing — any sequential decision problem with delayed feedback.

---

## Mathematical Foundations

### 1. Linear Regression

The hypothesis $h_\theta(x)$ is a linear combination of features:
$$h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n$$
- **Cost Function (MSE)**: $J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$
- **Optimization**: Gradient Descent $\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$

**Key insight:** Gradient descent is the universal procedure — compute which direction in weight space reduces loss, take a small step, repeat. Every optimizer (SGD, Adam, RMSProp) is a variation on this idea.

**How to verify understanding:** Derive the gradient $\frac{\partial J}{\partial \theta_j}$ for MSE from scratch — this is what backpropagation does conceptually.

**What trips people up:** Assuming gradient descent finds the exact minimum. It does for linear regression only because MSE is convex there. For non-convex losses (neural networks), it finds *a* minimum, not necessarily the global one.

### 2. Logistic Regression

Predicts probabilities using the **Sigmoid Function**:
$$g(z) = \frac{1}{1 + e^{-z}}$$
Hypothesis: $h_\theta(x) = g(\theta^T x) = P(y=1|x;\theta)$
- **Loss (Binary Cross-Entropy)**: $L(y, \hat{y}) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$

**Key insight:** Logistic regression is still a linear model — the sigmoid squashes output into $[0, 1]$, but the decision boundary stays a hyperplane in feature space. A curved boundary requires engineered non-linear features first.

**How to verify understanding:** Draw a 2D dataset where logistic regression fails no matter how you tune its weights, then describe what feature transformation fixes it.

**What trips people up:** Thinking logistic regression is a classifier. It's a probability estimator — the decision threshold (default 0.5) is a separate choice, and changing it shifts the precision-recall tradeoff, not the model itself.

---

## Bias-Variance Tradeoff

Every model selection decision in this plan can be framed as a bias-variance question — regularization, ensembles, and cross-validation are all answers to it.

**Bias** is error from overly simplistic assumptions (underfitting). **Variance** is error from over-sensitivity to training noise (overfitting). Linear models usually have high bias/low variance; deep trees have low bias/high variance.

**Key insight:** The goal is to minimize *expected test error* (bias² + variance + irreducible noise), not training error. Bias and variance move in opposite directions as model complexity increases.

**How to verify understanding:** A model has 100% training accuracy and 60% test accuracy. A colleague says "just make the model more complex." Right move or not — why?

**What trips people up:** Treating bias and variance as properties of the model alone. They depend on dataset size too — the same architecture can overfit with 100 samples and underfit with 1 million.

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
