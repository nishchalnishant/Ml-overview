# Day 3-4: Data Preprocessing Techniques

## 📋 Executive Summary
| Stage | Problem | Techniques | Goal |
|-------|---------|------------|------|
| **Cleaning** | Missing values, Noise | Imputation, Outlier removal | Data Integrity |
| **Scaling** | Feature range mismatch | Standardization, Min-Max | Convergence Speed |
| **Encoding** | Strings/Categories | One-Hot, Target, Label | Numerical compatibility |
| **Selection** | Irrelevant features | PCA, Lasso (L1), RFE | Complexity reduction |

---

## 🛠️ 1. Data Cleaning: The "Garbage In, Garbage Out" Rule

### Handling Missing Values
1. **Mean/Median/Mode**: Simple but reduces variance.
2. **KNN Imputation**: Uses nearest neighbors to predict missing values (more accurate but slower).
3. **Iterative Imputation**: Models each feature as a function of the others.

### Outlier Detection
- **Z-Score**: $z = \frac{x - \mu}{\sigma}$. Typically $|z| > 3$ is an outlier.
- **IQR**: Values outside $[Q1 - 1.5IQR, Q3 + 1.5IQR]$.

---

## 📐 2. Feature Scaling

### Standardization (Z-Score)
Centers data around 0 with unit variance.
$$x_{std} = \frac{x - \mu}{\sigma}$$
- **When to use**: Algorithms assuming Gaussian distributions (SVM, Linear Reg, PCA).

### Min-Max Scaling
Rescales to $[0, 1]$.
$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
- **When to use**: Neural Networks, KNN, algorithms that don't make Gaussian assumptions.

---

## 🧬 3. Categorical Encoding

### One-Hot Encoding
Creates binary columns for each category.
- **Problem**: "Dummy Variable Trap" (perfect multicollinearity). Always drop one column ($n-1$) in linear models.

### Target (Mean) Encoding
Replaces the category with the average target value for that category.
- **Warning**: High risk of **Data Leakage**. Always use K-fold cross-validation during encoding.

---

## ❓ Interview Questions

**1. "Why should you perform the train-test split before scaling?"**
> To prevent **Data Leakage**. Scaling factors ($\mu, \sigma$) should be calculated ONLY on the training set and then applied to the test set to simulate real-world unseen data.

**2. "Explain the difference between Normalization (Min-Max) and Standardization (Z-score)."**
> Normalization squashes data into a fixed range ($[0,1]$), which is sensitive to outliers. Standardization doesn't have a fixed range and is more robust to outliers as it preserves the distribution shape.

**3. "What happens if you don't scale features for Gradient Descent?"**
> The contour plots of the cost function will be highly elongated (ovals), causing the gradient to oscillate and take much longer to reach the global minimum.

---

## 💻 Code Snippet
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Recommended: Use Pipelines to avoid leakage
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
```
