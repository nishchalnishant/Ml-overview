# Data Preprocessing & Engineering (Deep-Dive)

This module explores the critical transition from raw data to model-ready features. In production, 80% of model performance is often determined by the quality of this stage.

---

# 1. 🔹 Prevention of Data Leakage

## Q1: Why is leakage the "silent killer" of production models?

### 🔹 Direct Answer
Data leakage occurs when information from the future (the target) or information from the test set "leaks" into the training process. This leads to artificially high offline metrics that crash in production.

### 🔹 Practical Workflow
1. **Split First:** Always split into `X_train` and `X_test` BEFORE any processing.
2. **Fit on Train Only:** Scalers, Imputers, and Encoders must `fit()` on the training set.
3. **Transform Both:** Use the fitted parameters to `transform()` both sets.

---

# 2. 🔹 Handling Missing Data

## Q2: When is deletion acceptable vs. imputation?

### 🔹 Comparison Table

| Method | Technique | When to Use | Risk |
| :--- | :--- | :--- | :--- |
| **Deletion** | Drop rows/cols | Missing >50% or non-MCAR data. | Can introduce bias if not careful. |
| **Imputation** | Mean/Median/Mode | Low-to-medium missingness. | Median is safer for skewed data. |
| **Advanced** | MICE / KNN | Complex feature dependencies. | Computationally expensive. |
| **Indicator** | Missingness Flag | If missingness itself is a signal. | Adds feature dimensionality. |

---

# 3. 🔹 Numerical Transformations

## Q3: Standardization vs. Normalization.

### 🔹 Direct Answer
- **Standardization (Z-Score):** Centers data at 0 with unit variance. Best for models assuming Gaussian distribution (SVM, LogReg, PCA).
- **Normalization (Min-Max):** Scales data to [0, 1]. Best for Neural Networks and distance-based models (KNN) where output bounds matter.

### 🔹 Handling Outliers (The IQR Method)
- **IQR** = Q3 - Q1
- **Bounds:** $[Q1 - 1.5 \times IQR, \quad Q3 + 1.5 \times IQR]$
- **Strategy:** Capping (Winsorization) is often better than deletion as it preserves the sample size while neutralizing the outlier impact.

---

# 4. 🔹 Categorical Encoding

## Q4: How do you handle High Cardinality (e.g., 10,000 cities)?

### 🔹 Strategies
1. **One-Hot Encoding:** Creates a sparse matrix. Good for low cardinality ($< 50$ categories).
2. **Target Encoding:** Replaces category with the mean target value. 
    - *Danger:* Extreme overfitting. Use **Laplace Smoothing**.
3. **Feature Hashing:** Uses a hash function to map categories to fixed-size indices. Good for online learning.

---

# 5. 🔹 Production Realities: Data Drift

## Q5: How do you know when to retraining?

### 🔹 Direct Answer
By monitoring **Covariate Shift**—when the distribution of input features $(P(X))$ changes over time.
- **Detection:** Population Stability Index (PSI) or Kolmogorov-Smirnov (K-S) tests.
- **Action:** If the test set distribution deviates significantly from the training distribution, the preprocessing pipeline or model needs a refresh.

---

> [!TIP]
> **Implementation Note:** For Python pipeline code (Sklearn `Pipeline` and `ColumnTransformer`), see the [ML Coding Patterns Hub](../ml-interview-notes/coding.md).
