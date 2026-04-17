# Supervised Learning Mastery (Deep-Dive)

This track provides a comprehensive exploration of supervised learning algorithms, covering the mathematical "why" and the production "how."

---

# 1. 🔹 Algorithm Choice Blueprint

| Task | Category | Key Algorithm | Best For |
| :--- | :--- | :--- | :--- |
| **Regression** | Linear | Linear Regression | Interpretability, Baseline. |
| **Regression** | Non-Linear | Random Forest / XGBoost | Complex patterns, Tabular data. |
| **Classification** | Probabilistic | Logistic Regression | Probability estimation, baselines. |
| **Classification** | High-Margin | SVM | High-dim data, clear separation. |
| **Classification** | Fast Baseline | Naive Bayes | Text data, small datasets. |

---

# 2. 🔹 Linear & Logistic Models

## Q1: Why is Log-Loss used for Logistic Regression instead of MSE?

### 🔹 Direct Answer
MSE for classification is **non-convex**, meaning gradient descent can get stuck in local minima. **Log-Loss (Cross-Entropy)** is convex, ensuring that gradient descent finds the global optimum. Additionally, Log-Loss penalizes confident-but-wrong predictions much more aggressively than MSE.

### 🔹 Mathematical Foundation: Logistic Regression
The probability is mapped via the **Sigmoid** function:
$$P(y=1|x) = \frac{1}{1 + e^{-(w^T x + b)}}$$
**Loss Function (Maximum Likelihood):**
$$J(w) = -\frac{1}{m} \sum [y \ln(\hat{y}) + (1-y) \ln(1-\hat{y})]$$

---

# 3. 🔹 Support Vector Machines (SVM)

## Q2: How does the "Kernel Trick" work without adding features?

### 🔹 Direct Answer
The Kernel Trick computes the **dot product** of two vectors in a high-dimensional space without ever explicitly mapping them to that space. It uses a similarity function $K(x, y)$ that corresponds to a dot product in a some feature space.

### 🔹 Intuition
Imagine two groups of people that can only be separated by a circle (non-linear). Instead of adding new coordinates, we change the way we measure "distance" between them (the kernel), which allows the model to find a linear boundary in that "spectral" world.

---

# 4. 🔹 Tree-Based Ensembles

## Q3: Random Forest vs. Gradient Boosting.

### 🔹 Comparison Table

| Feature | Random Forest | Gradient Boosting (XGBoost) |
| :--- | :--- | :--- |
| **Method** | Bagging (Parallel) | Boosting (Sequential) |
| **Primary Goal** | Reduce Variance | Reduce Bias |
| **Overfitting** | Hard to overfit | Prone to overfitting |
| **Complexity** | Simple, robust | High, needs tuning |

### 🔹 Feature Importance Logic
- **Gini Importance:** Sum of total reduction of Gini impurity provided by a feature.
- **Permutation Importance:** Measures how much the model score drops when a feature's values are randomly shuffled.

---

# 5. 🔹 Evaluation Metrics

## Q4: Accuracy vs. Precision vs. Recall.

### 🔹 The Logic
- **Precision:** "Of all we predicted positive, how many were actually positive?" (Minimize False Positives).
- **Recall:** "Of all that were actually positive, how many did we find?" (Minimize False Negatives).
- **F1-Score:** The Harmonic Mean of Precision and Recall.

---

> [!TIP]
> **Production Recommendation:** For tabular data with high cardinality, always start with **XGBoost** or **LightGBM**. For image or sequential data, skip traditional ML and move to the [Deep Learning Foundation](../deep-learning/README.md).
