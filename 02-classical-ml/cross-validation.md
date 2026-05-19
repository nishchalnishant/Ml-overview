# Cross-Validation Strategies

---

## The Problem CV Solves

**The problem**: You train a model on 80% of your data and evaluate on the remaining 20%. The result is one number from one specific split. If your dataset has 1,000 rows, whether a difficult case lands in train or test can swing performance by 5%. You report 0.87 AUC — but was that because the model is genuinely good, or because this particular split happened to put easy cases in the test set?

**The core insight**: A single holdout estimate has high variance because it depends heavily on which samples happened to land in the test set. Cross-validation averages the estimate across many different train/test splits, reducing variance. The test set remains completely untouched — CV operates only on training data.

**The mechanics**: Split training data into K folds. Train on K-1 folds, evaluate on the held-out fold. Rotate which fold is held out. After K rounds, average the K scores. Each sample served as a test point exactly once.

```python
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc')
print(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

K=5 or K=10 are standard. Higher K reduces bias (each training set is larger) but increases variance (more overlap between folds) and compute cost.

**What breaks**: The variance of the CV estimate does not decrease as 1/K — folds overlap, making estimates correlated. More folds do not always mean a better estimate. Leave-one-out CV (K=n) has near-zero bias but very high variance and costs n training runs.

---

## Stratified K-Fold

**The problem**: In a dataset with 5% positive examples, a fold created by random assignment might contain 1% positives by chance. The model trains on an unusual class distribution, and the evaluation sees a different one. Estimates vary widely across folds not because of true model variance, but because the class ratio fluctuated.

**The core insight**: Each fold should mirror the original class distribution. If 5% of the full dataset is positive, 5% of every fold should be positive.

**The mechanics**: Sort samples by class. Distribute samples from each class proportionally across all K folds. Now every fold has (approximately) the same class distribution as the full dataset.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
```

**What breaks**: With extreme imbalance (0.1% positives), stratification still can't guarantee every fold has positive examples when K is large relative to the number of minority samples. If K × (minority count) < K, some folds will have zero minority samples.

---

## Group K-Fold

**The problem**: A medical dataset has 5 records per patient. Random K-Fold assigns some records from patient 47 to the training set and others to the validation set. The model has seen patient 47's baseline characteristics — it effectively memorizes the patient, and validation performance is inflated. In production, the model will never have seen the patient before.

**The core insight**: Samples from the same group are not independent. Splitting them across train and validation introduces a form of leakage — the model trains on information it shouldn't have when evaluated on a truly new entity.

**The mechanics**: Ensure all records from a given group (patient, user, session, geographic region) appear in the same fold. A fold is held out as a group, not as individual samples.

```python
from sklearn.model_selection import GroupKFold

groups = df['user_id'].values
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=gkf, groups=groups)
```

**What breaks**: GroupKFold doesn't guarantee class balance across folds (since groups may have skewed class distributions). Use `StratifiedGroupKFold` when both group integrity and class balance matter.

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5)
for train_idx, val_idx in sgkf.split(X, y, groups):
    ...
```

---

## Time Series Cross-Validation

**The problem**: You shuffle a time series before splitting into folds. The validation set for fold 1 contains records from January, but the training set contains records from March. The model was trained on the future to predict the past. In production, it must predict without seeing the future. You've estimated performance on a task that doesn't exist.

**The core insight**: Time is a directed dependency. Training data must always precede validation data. Shuffling destroys this order and makes the estimate meaningless.

### Expanding Window (TimeSeriesSplit)

Train on all data up to a cutoff, validate on the window immediately after. Expand the training window with each fold.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=0)
for train_idx, val_idx in tscv.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
```

```
Fold 1: [===== train =====] [= val =]
Fold 2: [========= train =========] [= val =]
Fold 3: [============ train ============] [= val =]
```

### Sliding Window

Keeps the training window at a fixed size, moving forward in time. Simulates a deployment scenario where the model is periodically retrained on a fixed lookback period.

```python
def sliding_window_cv(X, y, train_size, val_size, step=1):
    for start in range(0, len(X) - train_size - val_size + 1, step):
        train_idx = range(start, start + train_size)
        val_idx   = range(start + train_size, start + train_size + val_size)
        yield list(train_idx), list(val_idx)
```

### Gap Parameter

Feature engineering that uses rolling windows can bleed future information into the training set — a 7-day rolling average computed for a training sample at t=100 uses data through t=107, which may overlap with the validation window. Add a gap between the last training timestamp and the first validation timestamp.

```python
tscv = TimeSeriesSplit(n_splits=5, gap=7)   # 7-step gap prevents feature bleed
```

**What breaks**: Even with proper time ordering, if your features use rolling windows longer than the gap, you still have leakage. The gap must be at least as long as the longest lookback window in your feature engineering.

---

## Repeated K-Fold

**The problem**: With a small dataset, a single 5-fold CV estimate has high variance — the 5-fold arrangement you happened to use could be unusually favorable or unfavorable. Two runs of 5-fold CV on the same data with different random seeds give noticeably different estimates.

**The core insight**: Average over many different K-fold arrangements. Each repetition uses a different random shuffle, producing a different partitioning of the data. The average over many repetitions has lower variance than any single one.

**The mechanics**: Run K-fold CV with n_repeats different random shuffles. Compute the mean and standard deviation over all (n_repeats × K) fold scores.

```python
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=rskf)
print(f"Repeated CV: {scores.mean():.4f} ± {scores.std():.4f}")
```

Cost: 50 model fits (5 folds × 10 repeats). Worth it for small datasets; wasteful for large ones.

**What breaks**: The fold scores are not independent (they share training samples between repeats), so the standard deviation underestimates the true uncertainty of the estimate. Use it as a diagnostic, not as a rigorous confidence interval.

---

## Nested Cross-Validation

**The problem**: You use 5-fold CV to compare 20 hyperparameter configurations and pick the best one. You then report its CV score as the model's performance. This is optimistic — you selected the configuration that happened to score best on *this particular* CV split. The reported score is biased upward.

**The core insight**: HP tuning and performance estimation are two separate problems. They need separate data to be answered without bias. The inner loop does HP selection. The outer loop estimates how well the best-selected model generalizes to truly new data.

**The mechanics**: Outer loop K-Fold splits data into outer-train and outer-test. On each outer-train fold, an inner CV loop tunes HPs. The best HPs from the inner loop are used to train a model on the full outer-train; it is evaluated on outer-test. The outer loop scores give an unbiased estimate of the model's generalization.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

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

The model you actually deploy is retrained on all training data using the best HPs from a separate, non-nested tuning run. The nested CV score is only for honest performance estimation.

**What breaks**: Nested CV is expensive (inner_K × outer_K × n_candidates evaluations). For reporting purposes it is necessary. For production model selection, most teams run a single inner CV loop and accept that the reported score is slightly optimistic.

---

## Getting Full CV Diagnostics

**The problem**: `cross_val_score` gives you test scores. But to diagnose overfitting you also need training scores. And sometimes you want the fitted models themselves.

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    model, X, y,
    cv=StratifiedKFold(5),
    scoring=['roc_auc', 'f1', 'precision', 'recall'],
    return_train_score=True,
    return_estimator=True,
)
print(results['test_roc_auc'])
print(results['train_roc_auc'])   # Large train-test gap = overfitting
```

---

## Preprocessing Must Live Inside the Pipeline

**The problem**: You fit a StandardScaler on all training data before cross-validation. The scaler's mean and std were computed using the validation fold's data. Every fold's validation estimate is computed on data that already influenced the scaler — a subtle but real form of leakage.

**The core insight**: Every preprocessing step that has a `fit` operation must re-fit on each fold's training portion. The validation portion must be transformed using only statistics from the training portion of that fold.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('scaler',    StandardScaler()),
    ('selector',  SelectFromModel(LassoCV())),
    ('clf',       RandomForestClassifier()),
])
# cross_val_score re-fits the entire pipeline (including scaler and selector)
# on each training fold — no leakage possible
scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5))
```

---

## Common Mistakes

| Mistake | Why it's wrong |
|---|---|
| Fitting preprocessor on full data before CV | Validation fold influenced the preprocessing statistics — leakage |
| Using the test set for model selection | Test set is no longer unseen — reported performance is optimistic |
| Shuffling time series | Trains on future to predict the past — invalid evaluation |
| Ignoring groups (per-patient, per-user data) | Same entity in train and val inflates performance |
| Comparing models from different random CV splits | Different splits may favor different models by chance |
| Reporting nested CV score as the "deployment model" score | The best HP configuration from a nested run was not retrained on all data |
