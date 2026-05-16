# Ensemble Methods

Ensembles combine multiple models to produce predictions better than any individual model. Key principle: errors of individual models should be uncorrelated — diverse models cancel each other's mistakes.

---

## Why Ensembles Work

**Bias-Variance decomposition:** Expected error = Bias² + Variance + Irreducible noise.

- **Bagging** reduces variance (averages out noise)
- **Boosting** reduces bias (focuses on hard examples)
- **Stacking** can reduce both by learning optimal combination

---

## Bagging (Bootstrap Aggregating)

Train B models on bootstrap samples (sampling with replacement). Combine predictions by averaging (regression) or majority vote (classification).

```python
from sklearn.ensemble import BaggingClassifier

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None),
    n_estimators=100,
    max_samples=0.8,      # fraction of samples per bag
    max_features=0.8,     # fraction of features per bag
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)
bag.fit(X_train, y_train)
```

**Random Forest** = Bagging + random feature subsets at each split (extra randomization reduces correlation between trees).

**Out-of-bag (OOB) score:** Each tree is trained on ~63% of data (bootstrap); the remaining 37% serves as a free validation set.

```python
rf = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
print(f"OOB score: {rf.oob_score_:.4f}")  # unbiased estimate without separate val set
```

---

## Boosting

Train models sequentially. Each model focuses on examples that previous models got wrong. Combine with learned weights.

### AdaBoost

Weight training examples by their classification difficulty. Mis-classified examples get higher weight at next iteration.

```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # "stumps"
    n_estimators=200,
    learning_rate=0.5,
    algorithm='SAMME.R',
    random_state=42,
)
```

**Model weight:** `α_m = ½ ln((1-ε_m)/ε_m)` where `ε_m` is weighted error. Better models get higher vote.

### Gradient Boosting

Fit each new tree to the **negative gradient** (pseudo-residuals) of the loss function. Works for any differentiable loss.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,        # stochastic gradient boosting
    min_samples_leaf=20,
    random_state=42,
)
```

**XGBoost / LightGBM / CatBoost** are optimized variants — see `supervised-learning.md` for full comparison.

---

## Voting Ensemble

Combine diverse models with equal or learned weights. No retraining — just aggregate predictions.

```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=200)),
        ('xgb', XGBClassifier(n_estimators=200)),
    ],
    voting='soft',   # average probabilities (better than hard vote)
    weights=[1, 2, 2],
)
voting.fit(X_train, y_train)
```

- `voting='hard'`: majority class vote
- `voting='soft'`: average probabilities (requires calibrated models)

**Key:** Models should be diverse and individually strong. Combining two identical models gains nothing.

---

## Stacking (Stacked Generalization)

Train a **meta-learner** that learns how to combine base model predictions.

```
Level 0 (base learners): Train diverse models (LR, RF, XGB, SVM, NN)
Level 1 (meta-learner): Train on out-of-fold predictions of Level 0
```

### Cross-validated stacking (to avoid leakage)

```python
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression

base_models = [
    RandomForestClassifier(n_estimators=200, random_state=42),
    XGBClassifier(n_estimators=200, random_state=42),
    LGBMClassifier(n_estimators=200, random_state=42),
]

# Generate OOF (out-of-fold) predictions — no data leakage
meta_features_train = np.column_stack([
    cross_val_predict(m, X_train, y_train, cv=5, method='predict_proba')[:, 1]
    for m in base_models
])

# Train base models on full train set; predict on test
for m in base_models:
    m.fit(X_train, y_train)
meta_features_test = np.column_stack([
    m.predict_proba(X_test)[:, 1] for m in base_models
])

# Train meta-learner on OOF predictions
meta_model = LogisticRegression()
meta_model.fit(meta_features_train, y_train)
final_pred = meta_model.predict_proba(meta_features_test)[:, 1]
```

**Meta-learner choice:** Logistic regression is standard (simple, low variance). Can add original features to meta-features for richer input.

**Multi-level stacking:** Stack of stacks — Level 0 → Level 1 → Level 2. Diminishing returns; rarely go past 2 levels.

---

## Blending

Simpler variant of stacking: hold out a fixed validation set for generating meta-features (instead of OOF cross-validation).

```
Train base models on X_train
Predict on X_val → meta_features_val
Train meta-learner on meta_features_val, y_val
```

**Difference from stacking:** Blending uses a single held-out set (less data efficient, higher variance). Stacking uses cross-validation (more robust). Use stacking in practice.

---

## Snapshot Ensembles

For deep learning: save model checkpoints at different points in training (e.g., when LR is at minima in cyclic schedule). Ensemble these snapshots.

- Zero extra training cost
- Works well with cosine annealing with restarts (SGD with warm restarts)

---

## Multi-Seed Ensemble

Train the same model architecture multiple times with different random seeds. Average predictions. Reduces variance from random initialization. Simple and effective.

---

## Calibration Before Ensembling

Individual models must output **calibrated probabilities** before soft voting or stacking. Uncalibrated models (e.g., raw tree outputs) produce poor combined probabilities.

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_rf = CalibratedClassifierCV(rf, cv='prefit', method='isotonic')
calibrated_rf.fit(X_cal, y_cal)   # fit calibrator on held-out calibration set
```

---

## When to Use Which

| Method | Reduces | Best for | Cost |
|--------|---------|---------|------|
| Bagging / Random Forest | Variance | High-variance base models (deep trees) | Low |
| AdaBoost | Bias | Weak learners (stumps) | Low |
| Gradient Boosting | Bias + Variance | Tabular data, competitions | Medium |
| Voting | Both | Diverse strong models | Negligible |
| Stacking | Both | Competitions, maximum accuracy | Medium |
| Snapshot / Multi-seed | Variance | Deep learning | Negligible |

---

## Key Interview Points

- Bagging reduces variance (parallel); boosting reduces bias (sequential).
- Random Forest adds random feature selection on top of bagging — key to diversity.
- OOB score provides free validation without a held-out set.
- Stacking must use OOF predictions — fitting on base model predictions trained on the same data leaks labels.
- Soft voting requires calibrated probabilities; hard voting does not.
- Meta-learner in stacking is usually simple (logistic regression) — complex meta-learner overfits.
- In practice: XGBoost/LightGBM alone often matches stacking on tabular data. Stacking matters most in competitions.
