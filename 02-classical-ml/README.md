# Machine Learning Foundations

Welcome to the Machine Learning Foundations library. This track provides a deep-dive, first-principles exploration of algorithms, from their mathematical derivation to their practical implementation.

You now have **two versions** of these notes:

- **Deep**: the original, more in-depth version on the original filenames.
- **Snappy**: the punchier rewrite for faster reading on the `*-snappy.md` files.

## Quick access

| Topic | Deep | Snappy |
| :--- | :--- | :--- |
| Overview | `README.md` | `README-snappy.md` |
| Supervised learning | `supervised-learning.md` | `supervised-learning-snappy.md` |
| Unsupervised learning | `unsupervised-learning.md` | `unsupervised-learning-snappy.md` |
| Data preprocessing | `data-preprocessing.md` | `data-preprocessing-snappy.md` |

---

# 1. 🔹 Algorithm Classification Framework

Understanding where an algorithm fits mathematically is critical for choosing the right tool and explaining tradeoffs.

### 🔹 Summary Table

| Category | Math Approach | Key Algorithms | Best For |
| :--- | :--- | :--- | :--- |
| **Linear Methods** | Hyperplanes / Boundaries | Linear/Log Reg, SVM, PCA | Interpretable trends, High-dim data. |
| **Tree/Graph** | Non-linear Partitioning | Decision Trees, RF, XGBoost | Tabular data, non-linear patterns. |
| **Probabilistic** | Bayesian Inference | Naive Bayes, GMM, HMM | Uncertainty, Text, Sequence data. |
| **Metric-based** | Distance / Proximity | KNN, K-Means, DBSCAN | Small data, complex cluster shapes. |

---

# 2. 🔹 Linear Methods (Boundary-based)

## Q1: How do Linearly-based models differ from Tree-based models?

### 🔹 Direct Answer
Linear models find a continuous decision boundary (hyperplane) that minimizes a global loss function. Tree-based models recursively partition the feature space into axis-aligned boxes, making them non-parametric and scale-invariant.

### 🔹 Deep Dive: Linear Regression
**Goal:** Predict a continuous target $\hat{y}$ by finding weights $w$ and bias $b$:
$$\hat{y} = w^T x + b$$
**Loss Function (MSE):** $J(w, b) = \frac{1}{n} \sum (y - \hat{y})^2$
**Key Assumptions:**
1. **Linearity:** Relationship between X and y is linear.
2. **Homoscedasticity:** Constant variance of errors across all levels of features.
3. **No Multicollinearity:** Independent variables are not highly correlated.

---

# 3. 🔹 Tree-Based Methods (Logic-based)

## Q2: Why are Ensembles usually better than single trees?

### 🔹 Direct Answer
Ensembles reduce the **High Variance** associated with deep decision trees.
- **Bagging (Random Forest):** Trains trees independently on subsets of data/features and averages them to reduce error.
- **Boosting (XGBoost):** Trains trees sequentially, where each new tree corrects the residual errors of the previous ones.

---

# 4. 🔹 The Bias-Variance Tradeoff

## Q3: Explain the decomposition of Expected Error.

### 🔹 Mathematical Foundation
$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \sigma^2 \text{ (Irreducible Error)}$

- **High Bias (Underfitting):** Model is too simple for the data (e.g., Linear Reg on complex data).
- **High Variance (Overfitting):** Model memorizes noise in the training data (e.g., Deep Tree with no pruning).

---

## 🔹 Training Strategies & Regularization

To balance the tradeoff, we use regularization:
- **L1 (Lasso):** Shrinks weights to zero (Feature Selection).
- **L2 (Ridge):** Shrinks weights but keeps them small (Prevents Overfitting).
- **Elastic Net:** A hybrid of L1 and L2.

---

> [!TIP]
> **Learning Tip:** For a rapid interview revision, use the [AI & ML Revision Hub](../01-foundations/AI_ML_REVISION_GUIDE.md). For deep mathematical derivations of these algorithms, see the [Math Derivations Hub](../07-interview-prep/ml/math-derivations.md).
