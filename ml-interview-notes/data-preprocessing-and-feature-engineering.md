# Data Preprocessing & Feature Engineering

This hub covers the foundational "data-first" steps of the ML lifecycle. Senior-level mastery involves understanding not just *how* to transform data, but the mathematical assumptions, deployment robustness, and leakage prevention required for production.

---

# 1. 🔹 Encoding Categorical Features

## Q1: One-Hot vs. Target Encoding - When to use what?

### 🔹 Direct Answer
- **One-Hot Encoding:** Creates binary columns for each category. Best for **low-cardinality** nominal features (e.g., Color: Red, Blue, Green).
- **Target (Mean) Encoding:** Replaces each category with the average target value for that category. Best for **high-cardinality** features (e.g., City, Zip Code) where one-hot would create too many columns (the "Curse of Dimensionality").

### 🔹 Deep Dive: The Risks of Target Encoding
Target encoding introduces a high risk of **Data Leakage** because you are using the target (labels) to create features. If category "A" only appears twice and happens to have high target values, the model will overfit to those specific instances.
- **Solution:** Use **K-Fold Target Encoding** (calculate mean on $K-1$ folds and apply to the $K^{th}$) or add **Smoothing** (blending the category mean with the global average).

---

# 2. 🔹 Handling Missing Data

## Q2: How do you handle missing values in production vs. training?

### 🔹 Direct Answer
First, identify the missingness mechanism:
1. **MCAR (Missing Completely at Random):** Simple imputation (Mean/Median) is often safe.
2. **MAR (Missing at Random):** Signal is in other features; use model-based imputation (**MICE** or **KNN Imputer**).
3. **MNAR (Missing Not at Random):** The "missingness" itself is the signal (e.g., wealthy people skipping the "Income" question).

### 🔹 Intuition (Indicator Variables)
In production, the fact that a field is missing is often as important as the value itself.
- **Strategy:** Always add a binary **Indicator Variable** `is_missing_X` alongside the imputed value. This allows the model to learn different behaviors for "Missing" vs. "Present."

### 🔹 Deep Dive: MICE (Multivariate Imputation by Chained Equations)
MICE treats each variable with missing values as a target of a "mini-regression" model using all other variables as predictors. It iterates through the variables multiple times until the imputed values stabilize.

---

# 3. 🔹 Scaling & Normalization

## Q3: Standard Scaler vs. Robust Scaler vs. Min-Max.

### 🔹 Comparison Table

| Scaler | Formula | Sensitivity to Outliers | Use Case |
| :--- | :--- | :--- | :--- |
| **Min-Max** | $\frac{x - \min}{\max - \min}$ | **High** | Neural Nets (Image data 0-255), algorithms that rely on bounded ranges. |
| **Standard** | $\frac{x - \mu}{\sigma}$ | **Medium** | Most Linear models, SVMs, and Logistic Regression. |
| **Robust** | $\frac{x - Q2}{Q3 - Q1}$ | **Very Low** | Data with extreme outliers (uses Median and IQR). |

### 🔹 Deep Dive: Why Scale?
Algorithms that rely on **Distance** (KNN, SVM, K-Means) or **Gradient Descent** (Neural Nets, Logistic Reg) are highly sensitive to scale. A feature with a range of $[0, 1,000,000]$ will dominate the gradients over a feature with a range of $[0, 1]$, leading to slow convergence or incorrect results.

---

# 4. 🔹 Advanced Transformations

## Q4: How do you handle highly skewed data? (Power Transforms)

### 🔹 Direct Answer
Highly skewed data can bias models (especially linear ones). Use **Power Transformations** to make the distribution more Gaussian.
- **Log Transform:** $y = \log(x+1)$. Good for right-skewed data with positive values.
- **Box-Cox:** A generalized power transform that finding the best $\lambda$ to normalize the data (requires all $x > 0$).
- **Yeo-Johnson:** A version of Box-Cox that works with negative numbers.

### 🔹 Intuition
Imagine a distribution where most people earn $50k but a few earn $50M. This "Long Tail" makes it hard for the model to see the signal in the $50k range. Squashing the values into log-space brings the $50M closer to the $50k, making the variations in the majority of data more visible to the model.

---

# 5. 🔹 Feature Crossing & Interactions

## Q5: What are Feature Crosses, and why are they used?

### 🔹 Direct Answer
A **Feature Cross** is a synthetic feature formed by multiplying or combining two or more individual features. It allows linear models to learn **non-linear dependencies** between features.

### 🔹 Example
- Feature A: `is_weekend` (Binary)
- Feature B: `is_location_beach` (Binary)
- **Feature Cross:** `is_weekend_AND_at_beach`.
A user might be at the beach often, but they only purchase ice cream if it's the *weekend AND they are at the beach*. A linear model cannot learn this "AND" relationship without a cross feature.

---

# 6. 🔹 Practical Perspective: Data Leakage

## Q6: What is the most common way data leakage happens in preprocessing?

### 🔹 Direct Answer
The most common mistake is **Fitting Transformers on the whole dataset** before splitting.
- **Example:** Calculating the global "Mean" using the entire dataset and then using that mean to scale the training set. Information from the "future" (test set) has leaked into the training scaling parameters.
- **The Correct Workflow:** 
  1. `Split` into Train/Test.
  2. `fit()` transformer ONLY on `X_train`.
  3. `transform()` both `X_train` and `X_test` using the learned parameters.

---

## 🔹 Difficulty Tag: 🟢 Easy to 🟡 Medium
