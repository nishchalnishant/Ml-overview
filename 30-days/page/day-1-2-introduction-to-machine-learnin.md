# Day 1-2: Introduction to Machine Learning

##  Executive Summary
| Topic | Core Concept | Key Algorithms | Evaluation |
|-------|--------------|----------------|------------|
| **Supervised** | Labeled data mapping $X \rightarrow y$ | Linear/Logistic Reg, Decision Trees, SVM | MSE, Accuracy, F1-Score |
| **Unsupervised** | Finding structure in $X$ | K-Means, PCA, GMM | Silhouette, Inertia, Variance |
| **Reinforcement** | Learning from environment feedback | Q-Learning, PPO | Cumulative Reward |

---

## 🔬 Core Learning Paradigms

### 1. Supervised Learning
The model learns from a trainer (labeled data). The objective is to minimize a loss function $J(\theta)$.
- **Regression**: Output is continuous. $y \in \mathbb{R}$.
- **Classification**: Output is discrete/categorical. $y \in \{0, 1, \dots, K\}$.

### 2. Unsupervised Learning
The model finds "hidden" structures.
- **Clustering**: Grouping similar instances.
- **Dimensionality Reduction**: Projecting high-dim data to low-dim space while preserving maximum information.

---

##  Mathematical Foundations

### 1. Linear Regression
The hypothesis $h_\theta(x)$ is a linear combination of features:
$$h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n$$
- **Cost Function (MSE)**: $J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$
- **Optimization**: Gradient Descent $\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$

### 2. Logistic Regression
Predicts probabilities using the **Sigmoid Function**:
$$g(z) = \frac{1}{1 + e^{-z}}$$
Hypothesis: $h_\theta(x) = g(\theta^T x) = P(y=1|x;\theta)$
- **Loss (Binary Cross-Entropy)**: $L(y, \hat{y}) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$

---

##  Interview Questions

**1. "What is the difference between Bias and Variance in the context of Days 1-2?"**
> **Bias** is error due to overly simplistic assumptions (underfitting). **Variance** is error due to over-sensitivity to training noise (overfitting). Linear models usually have high bias/low variance, while deep trees have low bias/high variance.

**2. "Why use the Log-Loss instead of MSE for Logistic Regression?"**
> MSE for logistic regression results in a non-convex cost function, leading to multiple local minima. Log-loss is convex, ensuring gradient descent converges to the global minimum.

**3. "Can Linear Regression be used for classification?"**
> Technically yes, by thresholding the output, but it's risky because linear regression is sensitive to outliers and doesn't output probabilities bounded by $[0, 1]$.

---

##  Implementation Quick-Lookup
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
