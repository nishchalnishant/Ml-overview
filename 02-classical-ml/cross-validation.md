# Cross-Validation Strategies

Cross-validation (CV) estimates model generalization performance on unseen data. Choosing the wrong CV strategy leads to optimistic estimates and poor production models.

---

## Why CV Matters

A single train/test split has high variance — results depend heavily on which samples land in each set. CV averages performance across multiple splits for a more reliable estimate.

**Golden rule:** Test set is untouched until final evaluation. CV operates only on training data.

---

## K-Fold Cross-Validation

Split data into K folds. Train on K-1 folds, validate on 1. Repeat K times (each fold serves as validation once). Average scores.

```python
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc')
print(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

**K = 5 or 10:** Standard choices. Higher K → less bias, more variance (more expensive).  
**K = n (Leave-One-Out):** Nearly unbiased but O(n) train runs. Rarely practical.

---

## Stratified K-Fold

Preserve class distribution in each fold. Essential for imbalanced classification.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
```

**Always use for classification** unless you have a strong reason not to.

---

## Group K-Fold

Prevent data leakage when samples are not independent — e.g., multiple records from the same user, patient, or geographic region.

```python
from sklearn.model_selection import GroupKFold

groups = df['user_id'].values   # ensure same user never in both train and val
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=gkf, groups=groups)
```

**Use when:** Same entity appears multiple times (patients in medical data, sessions per user, frames from the same video clip).

---

## Stratified Group K-Fold

Combines group constraints and class balance. Important for grouped imbalanced datasets.

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5)
for train_idx, val_idx in sgkf.split(X, y, groups):
    ...
```

---

## Time Series Cross-Validation

**Never shuffle time series.** The future cannot inform the past. Use forward-chaining (expanding window) or sliding window splits.

### TimeSeriesSplit (Expanding Window)

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=0)
for train_idx, val_idx in tscv.split(X):
    # train_idx always precedes val_idx in time
    X_tr, X_val = X[train_idx], X[val_idx]
```

```
Split 1: [=====] [===]
Split 2: [========] [===]
Split 3: [===========] [===]
Split 4: [==============] [===]
Split 5: [=================] [===]
         ↑ train            ↑ val
```

### Sliding Window

Fix train window size. Move forward in time.

```python
def sliding_window_cv(X, y, train_size, val_size, step=1):
    for start in range(0, len(X) - train_size - val_size + 1, step):
        train_idx = range(start, start + train_size)
        val_idx = range(start + train_size, start + train_size + val_size)
        yield list(train_idx), list(val_idx)
```

**Gap parameter:** Add a gap between train and validation to prevent look-ahead from feature engineering (e.g., rolling averages that bleed future information).

```python
tscv = TimeSeriesSplit(n_splits=5, gap=7)   # 7-day gap between train end and val start
```

---

## Repeated K-Fold

Run K-fold multiple times with different random splits. Reduces variance of the CV estimate.

```python
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=rskf)
print(f"Repeated CV: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Cost:** 50 model fits (5 folds × 10 repeats). Worth it for small datasets where variance is high.

---

## Nested Cross-Validation

Separate HP tuning from performance estimation. Without nesting, CV score is optimistic (selected best HP configuration for this specific data).

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

gs = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={'n_estimators': [100, 300], 'max_depth': [3, 5]},
    cv=inner_cv,
    scoring='roc_auc',
)
nested_scores = cross_val_score(gs, X, y, cv=outer_cv, scoring='roc_auc')
print(f"Unbiased estimate: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
```

**When to use:** When you want to report a performance estimate, not just select a model. The "best model" for deployment is retrained on all data with HPs from a separate tuning run.

---

## Getting Full CV Output

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    model, X, y,
    cv=StratifiedKFold(5),
    scoring=['roc_auc', 'f1', 'precision', 'recall'],
    return_train_score=True,
    return_estimator=True,   # keep fitted models
)
print(results['test_roc_auc'])
print(results['train_roc_auc'])   # check for overfitting: large train-test gap
```

---

## Common Mistakes

| Mistake | Consequence |
|---------|------------|
| Preprocessing inside CV with fit on all data | Data leakage → optimistic estimate |
| Using test set for model selection | Optimistic performance; fails on real new data |
| Random split on time series | Future leaks into training; extremely optimistic |
| Ignoring groups (e.g., per-patient data) | Same entity in train and val → over-optimistic |
| Comparing models with different CV splits | Not a fair comparison |

**Always put preprocessing in a Pipeline:**

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectFromModel(LassoCV())),
    ('clf', RandomForestClassifier()),
])
# Pipeline is CV-safe: fit() on train fold only at each split
scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5))
```

---

## Key Interview Points

- Stratified K-Fold is the default for classification — preserves class ratios per fold.
- Group K-Fold prevents leakage when samples share an identity (user, patient, video).
- Time series CV must be forward-chaining — no shuffling, and add a gap to prevent feature bleed.
- Nested CV is the only unbiased way to simultaneously tune HPs and report performance.
- Put all preprocessing inside a sklearn Pipeline to prevent leakage in CV.
- Repeated K-Fold reduces variance of the estimate at the cost of more training runs.
