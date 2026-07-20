---
module: Study Plans
topic: Week 2 Algorithms
subtopic: Day 8 9 Supervised Learning Algorithms
status: unread
tags: [studyplans, ml, week-2-algorithms-day-8-9-supe]
---
# Day 8-9: Supervised Learning (Ensembles & Kernels)

## Why This Topic Comes Here

Week 1 established the vocabulary: what learning is, how data should look before feeding it to a model, and what structure the data has. Now it is time to study the specific algorithms that solve supervised problems. This session focuses on the methods most commonly discussed in ML interviews: SVMs, k-NN, and ensemble methods. They are grouped together not because they are similar, but because they illustrate fundamentally different answers to the same question — how do you generalize from training examples? k-NN says "look at nearby training examples." SVM says "find the boundary with the largest safety margin." Ensembles say "use many weak models together." Understanding these contrasting philosophies is more durable than memorizing any one algorithm.

---

## Executive Summary

| Algorithm | Logic | Strength | Weakness |
|-----------|-------|----------|----------|
| **k-NN** | Proximity ($k$ neighbors) | Lazy learner, simple | High memory, slow inference |
| **Decision Tree** | Greedy split on impurity | Interpretable, handles mixed types | High variance, prone to overfitting |
| **SVM** | Wide-margin separation | Kernel trick, non-linear | Hard to tune, slow training |
| **Random Forest** | Bagging (Parallel Trees) | Low variance, robust | High memory, black box |
| **XGBoost/GBM** | Boosting (Sequential) | State-of-the-art accuracy | Prone to overfitting |

---

## 1. k-Nearest Neighbors (k-NN)

**Why this algorithm comes first in the sequence:** k-NN is the most intuitive non-parametric approach — it makes no assumptions about functional form, stores the entire training set, and defers all "learning" to inference time. This makes it a perfect contrast to linear models (which commit to a functional form at training time) and sets up the tradeoff between parametric and non-parametric methods.

**Key insight:** k-NN has no training phase — all computation happens at inference. This means its "model" is the entire training dataset. This is why it scales terribly: prediction requires searching $n$ training examples. More importantly, in high dimensions, the concept of "nearest" breaks down — all points become approximately equidistant (the curse of dimensionality). k-NN is only reliable when $n \gg d$.

**How to verify understanding:** You have 100,000 training examples with 500 features. Describe two specific problems that will arise when using k-NN with raw features, and what you would do about each.

**What trips people up:** Choosing $k=1$, which memorizes the training set perfectly (zero training error, high variance). The choice of $k$ is the entire bias-variance tradeoff for k-NN: small $k$ = low bias, high variance; large $k$ = high bias, low variance.

---

## 2. Decision Trees

**Why decision trees come before SVMs and ensembles:** Decision trees are the base learner for the most practically important ensemble methods (Random Forest and XGBoost). You cannot understand why random feature selection decorrelates trees, or why boosting focuses on residuals, without first understanding how a single tree splits.

A decision tree partitions the feature space by greedily finding the split at each node that most reduces an impurity measure.

**Impurity Measures:**

- **Gini Impurity**: $G = 1 - \sum_k p_k^2$ — measures the probability of misclassifying a randomly chosen element if it were labeled randomly according to the class distribution. Range: [0, 0.5] for binary classification.
- **Entropy**: $H = -\sum_k p_k \log_2 p_k$ — information-theoretic measure of disorder. Range: [0, 1] for binary classification.

**Key insight:** Gini and entropy almost always produce the same tree in practice. They differ slightly in how they weight extreme distributions: entropy penalizes impure nodes more at the extremes. The practical difference in final accuracy is negligible — the more important question is how you control tree depth through `max_depth`, `min_samples_leaf`, and `min_impurity_decrease`.

**How to verify understanding:** A node has 80 positive and 20 negative examples. Compute its Gini impurity and entropy. A candidate split produces two children: [70+, 5-] and [10+, 15-]. Does this split improve Gini? Walk through the weighted calculation.

**What trips people up:** Thinking that decision trees are "bad" because they overfit. An unpruned deep tree has near-zero training error — but this property is exactly why they are useful as high-variance base learners for ensemble methods. The goal of a single tree and the goal of a tree in a Random Forest are different. For a single tree, control depth. For a Forest, let trees be deep and control ensemble diversity.

**Pruning:**
- **Pre-pruning (early stopping)**: Set `max_depth`, `min_samples_split`, `min_samples_leaf` to stop splitting early. Adds bias to reduce variance.
- **Post-pruning (cost-complexity pruning)**: Grow the full tree, then remove subtrees that don't improve generalization on a validation set. Sklearn's `ccp_alpha` implements this.

---

## 3. Support Vector Machines (SVM)

**Why SVM illustrates a fundamental insight other algorithms don't:** Linear models find *any* separating hyperplane. SVM finds the *maximum-margin* hyperplane. This is a different inductive bias — SVM argues that the hyperplane furthest from both classes is the one most likely to generalize. This geometric intuition is worth internalizing regardless of whether you use SVMs in practice.

The goal is to find a hyperplane that maximizes the **margin** between classes.
- **Hard Margin**: Assumes linear separability.
- **Soft Margin**: Allows some misclassifications (controlled by $C$).
- **Kernel Trick**: Maps data to high-dim space where it *is* linearly separable.
  - **RBF Kernel**: $K(x, x') = \exp(-\gamma ||x - x'||^2)$

**Key insight:** The kernel trick allows SVM to operate in an implicitly infinite-dimensional space without ever computing the coordinates in that space. The computation only ever uses dot products between examples in the original space. This is powerful but also why SVMs are slow on large datasets — they must compare every training point to every other.

**How to verify understanding:** Explain why scaling features is mandatory for SVM but optional for a decision tree. What would happen to the margin if you scaled one feature by a factor of 1000?

**What trips people up:** Thinking the kernel trick "transforms the data." It does not — no new representation is ever computed. It computes similarity between points as if they were in a higher-dimensional space, using only their original-space values. The transformation is implicit in the kernel function.

---

## 4. Ensemble Methods: The "Wisdom of the Crowd"

**Why ensembles are the dominant approach in production:** Single models overfit or underfit. Combining many models reduces one or both types of error. The key is that two sources of error — variance and bias — respond to different ensemble strategies. Knowing which your model suffers from determines which ensemble approach to use.

### Bagging (Bootstrap Aggregating)

- **Concept**: Train $M$ models on $M$ random subsets (with replacement). Average their predictions.
- **Example**: **Random Forest**.
- **Impact**: Reduces **Variance** by averaging predictions.

**Key insight:** Averaging multiple independent predictions reduces variance by a factor of $M$ — but only if the models are uncorrelated. Random Forest introduces two sources of randomness (bootstrap sampling of rows, random feature subset at each split) specifically to decorrelate the trees. Without feature randomization, all trees would use the strongest features at the root and become nearly identical — and averaging identical models does nothing for variance.

**How to verify understanding:** Why does adding more trees to a Random Forest beyond a certain point give diminishing returns? What does this tell you about the practical limit of variance reduction through averaging?

**What trips people up:** Thinking that bagging can reduce bias. It cannot. The average of many high-bias models is still a high-bias model. If your individual decision trees are consistently wrong in the same direction, averaging them does not fix that.

### Boosting

- **Concept**: Train models sequentially. Model $i+1$ corrects the errors of model $i$.
- **Examples**: AdaBoost, Gradient Boosting, XGBoost.
- **Impact**: Reduces **Bias** (and often Variance, though it can overfit).

**Key insight:** Gradient Boosting fits each new tree to the *residuals* of the current ensemble. This is equivalent to gradient descent in function space — each new model is a step in the direction that reduces the loss most. The learning rate in gradient boosting plays the same role as step size in parameter-space gradient descent: too large and it overshoots, too small and it converges slowly.

**How to verify understanding:** XGBoost with 1000 trees and a learning rate of 0.1 overfits. You have three ways to reduce overfitting: reduce trees, reduce learning rate, or add regularization. Explain the effect of each and which you would try first.

**What trips people up:** Thinking boosting is always better than bagging. Boosting is more sensitive to noisy labels and outliers because it specifically focuses more weight on hard-to-fit examples — which in noisy datasets includes the mislabeled ones. If label quality is poor, bagging is often more robust.

---

## Interview Questions

**1. "What is the difference between Bagging and Boosting?"**
> Bagging focuses on reducing variance by parallelizing independent models (e.g., Random Forest). Boosting focuses on reducing bias by sequentially building models that learn from predecessor mistakes (e.g., XGBoost).

**2. "Why does Random Forest use random feature selection at each split?"**
> To **decorrelate** the trees. If all trees used the most important feature first, they would be highly correlated, and averaging them wouldn't reduce variance as effectively.

**3. "When would you prefer k-NN over a model like Logistic Regression?"**
> When the decision boundary is highly non-linear and you have enough memory to store the dataset. LR assumes a linear boundary, while k-NN is non-parametric.

---

## Code Comparison

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Random Forest: Focus on 'n_estimators' and 'max_depth'
rf = RandomForestClassifier(n_estimators=100, max_depth=None)

# XGBoost: Focus on 'learning_rate' (eta) and 'early_stopping'
xgb = XGBClassifier(learning_rate=0.1, n_estimators=1000, early_stopping_rounds=50)
```
