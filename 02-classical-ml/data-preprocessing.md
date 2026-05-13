# Data Preprocessing & Engineering (Deep-Dive)

This module explores the critical transition from raw data to model-ready features. In production, 80% of model performance is often determined by the quality of this stage.

---

# 1. Prevention of Data Leakage

## Q: Why is leakage the "silent killer" of production models?

Data leakage occurs when information from the future (the target) or from the test set "leaks" into training. This causes artificially high offline metrics that crash in production.

**Types of leakage:**
- **Target leakage:** features that encode information about $y$ (e.g., `loan_approved` as a feature when predicting `default`)
- **Train/test contamination:** fitting scalers/encoders on the entire dataset before splitting
- **Temporal leakage:** using future timestamps to predict past events
- **ID leakage:** IDs that carry cohort or time signal

**Golden rule:** split first, then fit all preprocessing on train only.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Pipeline prevents leakage automatically — fit() on train, transform() on val
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

# cross_val_score refits the pipeline on each fold — no leakage
scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
```

---

# 2. Handling Missing Data

## Missing Data Mechanisms

- **MCAR (Missing Completely At Random):** missingness is unrelated to any variable. Safe to delete.
- **MAR (Missing At Random):** missingness depends on observed variables, not the missing value itself. Imputation is appropriate.
- **MNAR (Missing Not At Random):** missingness depends on the missing value itself. Most dangerous — deletion and naive imputation both introduce bias.

## Strategy Comparison

| Method | Technique | When to Use | Risk |
| :--- | :--- | :--- | :--- |
| **Deletion** | Drop rows/cols | MCAR, missing >50% | Bias if MAR/MNAR |
| **Simple Imputation** | Mean/Median/Mode | Low missingness | Underestimates variance |
| **KNN Imputation** | Similar rows fill in | Complex feature dependencies | Expensive, scales poorly |
| **Iterative (MICE)** | Multiple rounds of regression | Complex, many features | Very expensive |
| **Indicator flag** | Add `is_missing` column | Missingness itself is a signal | Increases dimensionality |

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Indicator + imputation combo
from sklearn.impute import MissingIndicator

# Simple approach
num_imputer = SimpleImputer(strategy='median')   # median safer for skewed data
cat_imputer = SimpleImputer(strategy='most_frequent')

# KNN for when features are correlated
knn_imputer = KNNImputer(n_neighbors=5)
```

---

# 3. Numerical Transformations

## Standardization vs Normalization

**Standardization (Z-score):**

$$z = \frac{x - \mu}{\sigma}$$

Centers at 0, unit variance. Best for: SVM, LogReg, PCA, neural networks with BatchNorm.

**Min-Max Normalization:**

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Scales to $[0, 1]$. Best for: KNN, distance-based models, neural nets without BatchNorm.

**Robust Scaler:** uses median and IQR instead of mean/std. Best for: data with significant outliers.

$$x' = \frac{x - \text{median}}{\text{IQR}}$$

## Outlier Handling

**IQR method:**
- IQR $= Q3 - Q1$
- Bounds: $[Q1 - 1.5 \times \text{IQR},\quad Q3 + 1.5 \times \text{IQR}]$

**Winsorization** (capping) is usually better than deletion — preserves sample size.

**Log transform:** compresses right-skewed distributions. Use $\log(x+1)$ to handle zeros.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import numpy as np

# Log transform skewed features
X['revenue_log'] = np.log1p(X['revenue'])

# PowerTransformer (Yeo-Johnson) handles negative values, makes data more Gaussian
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X[numeric_cols])
```

---

# 4. Categorical Encoding

## Strategy Selection

| Cardinality | Strategy | Notes |
| :--- | :--- | :--- |
| Low (<15) | One-Hot Encoding | Clean, interpretable |
| Ordinal | Ordinal Encoding | Must specify order explicitly |
| High (15–1000) | Target Encoding | Leak-prone; use with smoothing |
| Very High (>1000) | Feature Hashing | Fixed-size output, some collisions |
| Text categories | Embedding | Use for NLP or deep tabular models |

**One-Hot Encoding:**
```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
```

**Ordinal Encoding (when order matters):**
```python
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
```

**Target Encoding with Laplace Smoothing (prevents overfitting):**

$$\hat{\mu}_k = \frac{n_k \bar{y}_k + m \bar{y}}{n_k + m}$$

where $n_k$ = count in category $k$, $\bar{y}_k$ = mean target in category $k$, $\bar{y}$ = global mean, $m$ = smoothing factor.

Always fit target encoder inside CV folds to avoid leakage.

```python
import category_encoders as ce

encoder = ce.TargetEncoder(smoothing=10.0)
X_train['city_encoded'] = encoder.fit_transform(X_train['city'], y_train)
X_test['city_encoded'] = encoder.transform(X_test['city'])
```

---

# 5. Full Pipeline with ColumnTransformer

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

numeric_features = ['age', 'income', 'tenure']
categorical_features = ['city', 'product_type']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=200))
])

model.fit(X_train, y_train)
# All preprocessing is encapsulated — safe for cross-validation and production
```

---

# 6. Feature Engineering

**Date/time features:**
```python
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['days_since_event'] = (df['timestamp'] - df['event_date']).dt.days
```

**Interaction features:**
```python
df['income_per_age'] = df['income'] / (df['age'] + 1)
df['tenure_x_product'] = df['tenure'] * df['product_count']
```

**Polynomial features (for linear models):**
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)
```

**Feature selection after engineering:**
```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold='median')
X_selected = selector.fit_transform(X_train, y_train)
```

---

# 7. Production Realities: Data Drift

## Q: How do you know when to retrain?

By monitoring **covariate shift** — when $P(X)$ changes over time — and **concept drift** — when $P(y|X)$ changes.

| Drift Type | Definition | Detection Method |
| :--- | :--- | :--- |
| **Covariate shift** | $P(X)$ changes | PSI, KS test, histogram comparison |
| **Label shift** | $P(y)$ changes | Monitor target distribution |
| **Concept drift** | $P(y|X)$ changes | Monitor model performance metrics |

**PSI (Population Stability Index):**

$$\text{PSI} = \sum_{i} (A_i - E_i) \cdot \ln\left(\frac{A_i}{E_i}\right)$$

- PSI < 0.1: stable
- 0.1–0.25: moderate shift, investigate
- > 0.25: major shift, retrain

**KS test:** compares CDFs of two distributions. `p < 0.05` indicates significant drift.

```python
from scipy.stats import ks_2samp

for col in numeric_features:
    stat, p_val = ks_2samp(X_train[col], X_production[col])
    if p_val < 0.05:
        print(f"DRIFT DETECTED in {col}: KS={stat:.3f}, p={p_val:.4f}")
```

---

> [!TIP]
> **Implementation Note:** Always use sklearn `Pipeline` + `ColumnTransformer` to encapsulate preprocessing. This ensures: (1) no leakage in cross-validation, (2) consistent transformation at inference, (3) easy serialization with `joblib`.
