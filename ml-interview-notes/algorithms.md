# ML Algorithms: From Trees to Ensembles

These notes follow the **Gold Standard** for interview preparation: providing direct answers, geometric intuition, and the critical tradeoffs that set senior candidates apart.

---

# 1. Tree-Based Models

## Q1: How does a Decision Tree work?

### 🔹 Direct Answer
A Decision Tree recursively partitions the feature space into regions by choosing the split that maximizes **Information Gain** (or minimizes impurity). At each node, it selects a feature and threshold that best separates the target classes (Gini/Entropy for classification) or reduces variance (MSE for regression).

### 🔹 Intuition
Imagine playing "20 Questions." You want to ask the most informative question first (e.g., "Is it alive?") to rule out the most possibilities. The tree does exactly this: it finds the "questions" (feature thresholds) that resolve the most uncertainty about the target.

### 🔹 Deep Dive: Gini vs. Entropy
- **Gini Impurity:** $1 - \sum p_i^2$. Measures the probability of a random element being misclassified if labeled according to the distribution. It is slightly faster to compute.
- **Entropy:** $-\sum p_i \log_2(p_i)$. Measures the average "information" or "surprise." It is more sensitive to changes in probability but slightly more computationally expensive.

## Q2: Random Forest vs. Gradient Boosting (XGBoost)

### 🔹 Comparison Table

| Feature | Random Forest (Bagging) | XGBoost (Boosting) |
| :--- | :--- | :--- |
| **Strategy** | Parallel; independent trees. | Sequential; trees correct past errors. |
| **Goal** | Reduces **Variance**. | Reduces **Bias** (and variance via reg). |
| **Hyperparameters** | Hard to overfit; "plug and play". | Highly sensitive; requires tuning. |
| **Handling Noise** | Robust to outliers and noise. | Can overfit to noise if not regularized. |

### 🔹 Deep Dive: Why "Random" Forest?
1. **Bagging (Bootstrap Aggregation):** Each tree sees a random subset of *rows*.
2. **Feature Subsampling:** At each node split, each tree sees a random subset of *features* (usually $\sqrt{d}$).
**Result:** These two mechanisms decorrelate the trees. Averaging decorrelated high-variance models is the mathematical secret to variance reduction.

---

# 2. Linear & Discriminative Models

## Q3: Explain Logistic Regression (Geometric & Probabilistic)

### 🔹 Direct Answer
Despite its name, Logistic Regression is a **linear classification** model. It assumes the **log-odds** of the probability are a linear combination of the features: $\log(\frac{p}{1-p}) = w^Tx + b$.

### 🔹 Intuition
- **Geometric:** It finds the hyperplane that best separates two classes.
- **Probabilistic:** It models the probability of class membership. If the score is $>0$ (log-odds), the probability is $>0.5$.

### 🔹 Code Snippet
```python
import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Prediction
p = sigmoid(np.dot(X, w) + b)
```

## Q4: How does SVM handle non-linear data? (The Kernel Trick)

### 🔹 Direct Answer
Support Vector Machines (SVMs) find the "maximum margin hyperplane." When data isn't linearly separable, they use the **Kernel Trick** to map data into a higher-dimensional space where it *is* separable, without actually computing the coordinates in that space.

### 🔹 Deep Dive: Common Kernels
- **Linear:** Fast, no extra params.
- **RBF (Gaussian):** Maps to infinite-dimensional space. The $\gamma$ parameter controls the "reach" of a single training point.

---

# 3. Dimensionality Reduction & Clustering

## Q5: How does K-Means work, and how do you choose 'K'?

### 🔹 Direct Answer
K-Means is an iterative algorithm that partitions data into $K$ clusters by:
1. Assigning points to the nearest centroid.
2. Updating centroids to be the mean of assigned points.

### 🔹 How to choose K?
1. **Elbow Method:** Plot Inertia (Within-Cluster Sum of Squares) vs. K; look for the "inflection point."
2. **Silhouette Score:** Measures how similar a point is to its own cluster compared to others. Range (-1, 1).

## Q6: PCA (Principal Component Analysis)

### 🔹 Direct Answer
PCA finds the orthogonal directions (principal components) that capture the maximum variance in the data. It is an unsupervised technique used for compression, denoising, and visualization.

### 🔹 Implementation Nuance
You **must** center and scale your data before PCA. If one feature is measured in "kilometers" and another in "millimeters," the "millimeters" feature will dominate the variance calculation solely due to scale.
