---
module: Classical ML
topic: Revision Card
subtopic: ""
status: unread
tags: [classical-ml, revision, cheatsheet]
---
# Classical ML — 10-Minute Revision Card

**Use:** Skim before any ML interview or before diving into a specific file. Every row is a decision, not a definition.

---

## Algorithm Selector

| Signal | Go-to |
|--------|-------|
| Need interpretability | Logistic Regression, Decision Tree |
| High-dimensional, sparse | Logistic Regression + L1, Linear SVM |
| Non-linear boundary, tabular | Gradient Boosting (XGBoost/LightGBM) |
| Small data, structured | SVM with RBF kernel |
| Clustering, no labels | K-Means (convex clusters), DBSCAN (arbitrary shape) |
| Dimensionality reduction | PCA (linear), UMAP (non-linear) |
| Fast baseline | Logistic Regression always first |

---

## Bias-Variance in 30 Seconds

**Total Error = Bias² + Variance + Irreducible Noise**

- **High Bias (underfitting):** train error high, val error high → model too simple → add complexity, add features, reduce regularization
- **High Variance (overfitting):** train error low, val error high → model memorizes noise → more data, regularization, simpler model, early stopping

**Gotcha:** More data reliably reduces variance; it does not fix bias.

---

## Regularization

| Type | Effect | Use when |
|------|--------|----------|
| L1 (Lasso) | Shrinks weights to exactly zero → sparse | Feature selection, many irrelevant features |
| L2 (Ridge) | Shrinks weights smoothly, none to zero | Most cases; correlated features |
| ElasticNet | Mix of L1 + L2 | Correlated features + sparsity needed |
| Dropout | Randomly zero activations during training | Neural nets only |

**Gotcha:** L1 adds a non-differentiable kink at zero — requires subgradient. L2 is smooth everywhere.

---

## Linear Regression

- Closed-form: $w = (X^T X)^{-1} X^T y$ — exact but $O(n^3)$, use gradient descent if features > ~1000
- Assumptions: linearity, homoscedasticity, no multicollinearity, independent errors
- When assumptions break: add polynomial features, log-transform skewed variables, use Ridge for multicollinearity

---

## Logistic Regression

- Sigmoid: $\sigma(z) = \frac{1}{1+e^{-z}}$ → squashes to $[0,1]$
- Loss: cross-entropy, convex → global minimum guaranteed
- Fails when: decision boundary is non-linear (need features) or classes perfectly separable (weights diverge to ±∞)

---

## SVM

- Maximizes margin between classes; only support vectors matter
- **Kernel trick:** implicitly maps to high-dimensional space without computing coordinates
  - Linear: fast, high-dimensional data; RBF: non-linear, low-dimensional; Polynomial: rare
- **C parameter:** small C = wide margin, more misclassifications; large C = narrow margin, fewer misclassifications
- **Gotcha:** Requires feature scaling. Doesn't output calibrated probabilities by default.

---

## Ensemble Methods

**Why ensembles work:** uncorrelated errors cancel when averaged.

| Method | Mechanism | Reduces | Use |
|--------|-----------|---------|-----|
| Bagging | Train on bootstrap samples, average | Variance | Random Forest |
| Random Forest | Bagging + random feature subsets | Variance + correlation | Robust baseline |
| Boosting | Sequential: each model corrects residuals | Bias | XGBoost, LightGBM |
| Stacking | Meta-learner on top of base predictions | Both | Competition setups |

**Key facts:**
- Bootstrap sample ≈ 63% unique points → ~37% OOB, free validation
- Random Forest cannot extrapolate beyond training range
- XGBoost uses 2nd-order gradients (Hessian) → faster convergence than AdaBoost

---

## Evaluation Metrics — Pick the Right One

**Classification:**

| Metric | When to use | Gotcha |
|--------|-------------|--------|
| Accuracy | Balanced classes only | Useless on 99/1 split |
| F1 | Imbalanced, both precision & recall matter | Ignores TN |
| PR-AUC | Positive prevalence < 10% | Baseline = prevalence, not 0.5 |
| ROC-AUC | General ranking quality | Optimistic on imbalance (TN inflation) |
| MCC | Any class balance | Range [-1,1]; all-negative predicts MCC=0 |

**Regression:** RMSE penalizes large errors more (squared); MAE is robust to outliers; MAPE fails near zero.

**30-second rule:** If positive class is rare → PR-AUC. If balanced → ROC-AUC or F1. Always look at the confusion matrix.

---

## Imbalanced Data

1. **First:** change your metric (F1, PR-AUC, MCC)
2. **Then:** class weights in loss (mathematically equivalent to oversampling)
3. **If still needed:** SMOTE (synthetic minority oversampling), threshold tuning
4. **Never:** report accuracy on imbalanced data

---

## Cross-Validation

- **k-Fold:** split data into k folds, train on k-1, test on 1, rotate. k=5 or 10 standard.
- **Stratified:** preserve class ratio in each fold — mandatory for classification
- **Time series:** always forward-chaining (train on past, test on future) — never shuffle
- **Gotcha:** tune hyperparameters on the validation fold of CV, not on test set

---

## Hyperparameter Tuning

| Method | How | Use when |
|--------|-----|---------|
| Grid Search | Exhaustive over grid | Small search space |
| Random Search | Random samples | Medium space; often beats grid |
| Bayesian Optimization | Model of loss surface, sample intelligently | Expensive evaluations |
| Early Stopping | Stop when val loss stops improving | Neural nets, GBMs |

**Rule of thumb:** Random search finds good hyperparameters faster than grid in high dimensions because most hyperparameters don't matter.

---

## Interview Quick-Draws

**"How do you handle missing data?"**
→ Understand why it's missing first. If MCAR: mean/median impute. If MAR: model-based imputation. If MNAR: feature-encode the missingness.

**"Decision tree vs logistic regression?"**
→ Tree: non-linear boundaries, interpretable splits, no scaling needed, high variance solo. LR: linear boundary, calibrated probabilities, fast, works well with L1/L2.

**"Why use gradient boosting over random forest?"**
→ Boosting reduces bias sequentially; better on structured tabular data empirically. RF is faster to train, more robust to hyperparameters. GBM wins competitions; RF wins production baselines.

**"What's the difference between L1 and L2?"**
→ L1 drives weights to exactly zero (sparse), useful for feature selection. L2 shrinks weights smoothly toward zero, handles correlated features better.
