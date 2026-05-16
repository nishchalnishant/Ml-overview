# Conformal Prediction

Conformal prediction produces prediction **sets** (classification) or **intervals** (regression) with guaranteed coverage — the true label is inside the set with at least 1-α probability, regardless of the underlying data distribution. No distributional assumptions required.

---

## Core Guarantee

For exchangeable data: `P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 - α`

This is a frequentist guarantee over random draws, not a Bayesian credible interval. It holds for any black-box model.

---

## Inductive (Split) Conformal Prediction

The most practical variant:

1. Split labeled data into training set and calibration set
2. Train model on training set
3. Compute **nonconformity scores** on calibration set
4. At inference, output the set of labels that are "not surprising" given the calibration scores

```python
import numpy as np

# Step 1-2: Train model on train set
# Step 3: Compute nonconformity scores on calibration set
# For classification: score = 1 - P(true class | x)
cal_scores = 1 - model.predict_proba(X_cal)[np.arange(len(y_cal)), y_cal]

# Step 4: Find quantile threshold
alpha = 0.1  # target: 90% coverage
n = len(y_cal)
q_level = np.ceil((n + 1) * (1 - alpha)) / n
q_hat = np.quantile(cal_scores, q_level, method='higher')

# Inference: include all classes with score <= q_hat
def predict_set(x):
    probs = model.predict_proba(x.reshape(1, -1))[0]
    scores = 1 - probs
    return np.where(scores <= q_hat)[0]  # prediction set
```

**Coverage guarantee:** At least 90% of test examples will have true label in their prediction set.

---

## Nonconformity Scores

The score measures how unusual a sample-label pair is. Different choices lead to different efficiency.

| Task | Nonconformity Score | Notes |
|------|---------------------|-------|
| Classification | `1 - P(y | x)` | Simple; use softmax output |
| Classification (better) | Adaptive Prediction Sets (APS) | Cumulative sorted probability |
| Regression | `|y - ŷ|` | Absolute residual |
| Regression (better) | `|y - ŷ| / σ̂(x)` | Normalized by local uncertainty |

### Adaptive Prediction Sets (APS)

Accumulates sorted class probabilities until the true class is included. Produces smaller sets for easy examples, larger for hard ones.

```python
def aps_score(probs, y_true):
    sorted_idx = np.argsort(probs)[::-1]
    cumsum = 0
    for rank, cls in enumerate(sorted_idx):
        cumsum += probs[cls]
        if cls == y_true:
            return cumsum - probs[cls] * np.random.uniform()  # randomized version
```

---

## Conformal Regression

**Split conformal interval:**
```python
# Residuals on calibration set
residuals = np.abs(y_cal - model.predict(X_cal))
q_hat = np.quantile(residuals, np.ceil((n+1)*(1-alpha))/n, method='higher')

# Prediction interval at test point
y_pred = model.predict(x_test)
interval = (y_pred - q_hat, y_pred + q_hat)
```

**Limitation:** Constant-width interval — not adaptive to local uncertainty. Use **Locally Weighted CP** or **CQR (Conformalized Quantile Regression)** for adaptive intervals.

### CQR — Conformalized Quantile Regression

Train quantile regression model to predict `[q_α/2(x), q_{1-α/2}(x)]`. Calibrate residuals:
`score = max(q_α/2(x) - y, y - q_{1-α/2}(x))`

Then adjust quantile by q_hat from calibration. Interval width adapts to local difficulty.

---

## Full Conformal vs Inductive Conformal

| Property | Full Conformal | Split Conformal |
|----------|---------------|----------------|
| Uses all data for calibration | Yes (retrain per test point) | No (fixed calibration split) |
| Computationally feasible | No (n+1 models) | Yes |
| Coverage guarantee | Exact | Marginal |
| Calibration set overhead | None | ~20-30% of data |

In practice: always use split conformal (inductive CP).

---

## Cross-Conformal / CV+ (Leave-One-Out)

Uses K-fold cross-fitting to avoid wasting data on calibration. Coverage guarantee is valid but approximate.

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
residuals = np.zeros(len(X))
for train_idx, cal_idx in kf.split(X):
    model.fit(X[train_idx], y[train_idx])
    residuals[cal_idx] = np.abs(y[cal_idx] - model.predict(X[cal_idx]))

q_hat = np.quantile(residuals, np.ceil((n+1)*(1-alpha))/n)
```

---

## Conditional Coverage

The basic guarantee is **marginal** (averaged over the data distribution). For a specific subgroup or x-value, coverage may be lower. This is called **conditional coverage**.

**Approaches for better conditional coverage:**
- Locally weighted conformal prediction
- Mondrian conformal prediction (stratify by group)
- RAPS (Regularized Adaptive Prediction Sets)

---

## Practical Considerations

**Exchangeability:** Requires i.i.d. or exchangeable data. Violates for time series (use rolling calibration windows) or covariate shift (use weighted CP).

**Calibration set size:** Larger calibration = tighter threshold estimate. Rule of thumb: at least 1000 calibration samples for α = 0.05.

**Set size efficiency:** Smaller prediction sets are better (more informative). Use APS or RAPS over basic threshold for classification.

**Distribution shift:** Under shift, coverage guarantee breaks. Use **weighted conformal prediction** with importance weights `w(x) = p_test(x) / p_train(x)`.

---

## Libraries

```python
# MAPIE — Scikit-learn compatible
from mapie.classification import MapieClassifier
from mapie.regression import MapieRegressor

mapie_clf = MapieClassifier(estimator=model, method="score", cv="prefit")
mapie_clf.fit(X_cal, y_cal)
y_pred, y_pred_sets = mapie_clf.predict(X_test, alpha=0.1)

# conformal_prediction library
# nonconformist
```

---

## When to Use Conformal Prediction

| Situation | Value |
|-----------|-------|
| High-stakes decisions (medical, legal) | Guaranteed coverage without distributional assumptions |
| Model uncertainty communication | Prediction sets instead of point estimates |
| Multi-class with ambiguous samples | Sets reveal uncertainty (e.g., {cat, dog} not just "cat") |
| Compliance requirements for coverage | Provable coverage guarantee |

**Alternatives:** Bayesian credible intervals (distributional assumptions), calibrated probabilities (no set guarantee), Gaussian processes (closed-form uncertainty).

---

## Key Interview Points

- Conformal prediction gives a **distribution-free coverage guarantee** — this is the main selling point.
- "Marginal" coverage = averaged over all x; not guaranteed for any specific x or subgroup.
- Split CP wastes calibration data but is always feasible. Full CP is exact but computationally intractable.
- CQR is the go-to for adaptive regression intervals.
- APS/RAPS are preferred over threshold-based CP for classification.
- Under distribution shift, the guarantee breaks — need weighted CP.
