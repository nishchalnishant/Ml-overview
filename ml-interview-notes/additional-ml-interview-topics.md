# Additional ML Interview Topics

This hub covers high-yield questions that often show up when an interviewer wants to see whether you can go beyond definitions and talk about failure modes, tradeoffs, and production behavior.

---

# 1. 🔹 Data Quality & Leakage

## Q1: What is Data Leakage, and how do you detect/prevent it?

### 🔹 Direct Answer
**Data Leakage** occurs when information from outside the training dataset (specifically from the "future" or the "target") is used to create the model. This leads to unrealistically high performance during offline testing that disappears in production.

### 🔹 Intuition
Imagine a student who accidentally finds the answer key to their final exam while studying. They score 100% on the exam, but they haven't actually learned the material. When they go to a real job (Production), they fail because the "answer key" is no longer available.

### 🔹 Deep Dive: Types of Leakage
1. **Temporal Leakage:** Using data recorded *after* the prediction event (e.g., using "Closing Price at 5 PM" to predict "Price at 10 AM").
2. **Target Leakage:** Features that are proxies for the label (e.g., "Monthly Interest Paid" used to predict "Loan Approval").
3. **Pipeline Leakage:** Fitting scalers or imputers on the whole dataset rather than just the training split.

---

# 2. 🔹 Imbalanced Learning

## Q2: Explain SMOTE. When does it help, and what are the risks?

### 🔹 Direct Answer
**SMOTE (Synthetic Minority Over-sampling Technique)** creates synthetic examples for the minority class by interpolating between existing nearby minority points.

### 🔹 Intuition
Instead of just duplicating the same few minority examples (Random Oversampling), SMOTE creates "informed" new data points along the lines connecting existing ones. It's like filling in the gaps in a sparsely populated neighborhood.

### 🔹 Deep Dive: Risks
- **Noisy Boundaries:** If classes overlap, SMOTE can create synthetic points that fall into the majority class's region, blurring the decision boundary.
- **High-Dimensional Sparsity:** In high dimensions, the "line" between two points may not represent a realistic data point.
- **Metric Distortion:** SMOTE changes the class distribution, which can mislead you during threshold selection if you don't account for it.

---

# 3. 🔹 Normalization Strategies

## Q3: Batch Normalization vs. Layer Normalization.

### 🔹 Comparison Table

| Feature | Batch Normalization (BN) | Layer Normalization (LN) |
| :--- | :--- | :--- |
| **Normalizes across** | The Batch (Samples). | The Layer (Features). |
| **Dependency** | Sensitive to batch size. | Independent of batch size. |
| **Best For** | CNNs, Computer Vision. | Transformers, NLP, RNNs. |
| **Test Time** | Uses running mean/variance. | Same calculation as training. |

### 🔹 Why LayerNorm for Transformers?
Transformers process variable-length sequences. In such cases, the "Batch" statistics are unstable. LayerNorm normalizes each word embedding independently of the others in the batch, making it much more stable for NLP.

---

# 4. 🔹 Tree-Based Libraries

## Q4: Compare XGBoost, LightGBM, and CatBoost.

### 🔹 Direct Answer
While all are Gradient Boosting implementations, they differ in their engineering optimizations:
- **XGBoost:** The "classic." Uses level-wise growth and strong regularization. Highly reliable baseline.
- **LightGBM:** Uses **Leaf-wise growth** and Histogram-based learning. It is **significantly faster** and handles massive datasets better, but prone to overfitting on small data.
- **CatBoost:** Native support for **categorical features**. Uses "ordered boosting" to prevent target leakage during encoding.

---

# 5. 🔹 Advanced Concepts

## Q5: What is Model Calibration?

### 🔹 Direct Answer
A model is **calibrated** if its predicted probabilities match empirical reality. If a model says "80% chance of fraud," then 80 out of 100 such transactions should actually be fraud.

### 🔹 Practical Perspective
Ranking models (like AUC-optimized ones) often give good order but poor probabilities (e.g., all scores are between 0.49 and 0.51).
- **Fix:** Use **Platt Scaling** (fitting a logistic regression to outputs) or **Isotonic Regression**.

---

## 🔹 Difficulty Tag: 🔴 Hard
