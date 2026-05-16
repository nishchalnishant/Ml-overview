# Feature Selection

Feature selection reduces dimensionality by removing irrelevant or redundant features. Benefits: reduces overfitting, speeds up training, improves interpretability, reduces data collection cost.

---

## Three Families

```
Feature Selection
├── Filter Methods    — score features independently of the model
├── Wrapper Methods   — use a model to evaluate feature subsets
└── Embedded Methods  — selection happens during model training
```

---

## Filter Methods

Fast and model-agnostic. Score each feature independently, then select top-k.

### Variance Threshold

Remove features with near-zero variance — they carry no information.

```python
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.01)   # remove features with var < 0.01
X_sel = sel.fit_transform(X)
```

### Correlation / Pearson

Remove features highly correlated with other features (redundancy).

```python
import pandas as pd
import numpy as np

corr_matrix = pd.DataFrame(X).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X_reduced = pd.DataFrame(X).drop(columns=to_drop)
```

### Mutual Information

Measures statistical dependence between feature and target. Works for non-linear relationships; captures more than Pearson.

```python
from sklearn.feature_selection import mutual_info_classif, SelectKBest

sel = SelectKBest(mutual_info_classif, k=20)
X_sel = sel.fit_transform(X_train, y_train)
```

For regression: `mutual_info_regression`.

### Chi-Squared Test

For classification with non-negative features. Tests independence between each feature and the target class.

```python
from sklearn.feature_selection import chi2, SelectPercentile

sel = SelectPercentile(chi2, percentile=50)
X_sel = sel.fit_transform(X_train, y_train)
```

### ANOVA F-statistic

Measures linear relationship between continuous feature and categorical target.

```python
from sklearn.feature_selection import f_classif

scores, pvalues = f_classif(X_train, y_train)
mask = pvalues < 0.05
X_sel = X_train[:, mask]
```

**Limitation of filter methods:** Evaluate features independently — miss interaction effects between features.

---

## Wrapper Methods

Use a model's performance to evaluate feature subsets. Accurate but computationally expensive.

### Recursive Feature Elimination (RFE)

Train model, rank features by importance, remove weakest, repeat.

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression

# Fixed n_features
rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=10)
rfe.fit(X_train, y_train)
print(rfe.support_)   # boolean mask of selected features
print(rfe.ranking_)   # 1 = selected, higher = eliminated earlier

# Cross-validated: automatically finds optimal n_features
rfecv = RFECV(estimator=LogisticRegression(), cv=5, scoring='roc_auc')
rfecv.fit(X_train, y_train)
print(f"Optimal features: {rfecv.n_features_}")
```

### Sequential Feature Selection (Forward/Backward)

- **Forward:** Start empty, greedily add feature that most improves CV score
- **Backward:** Start full, greedily remove feature with least impact

```python
from sklearn.feature_selection import SequentialFeatureSelector

sfs = SequentialFeatureSelector(RandomForestClassifier(), n_features_to_select=10, 
                                  direction='forward', cv=5, n_jobs=-1)
sfs.fit(X_train, y_train)
X_sel = sfs.transform(X_train)
```

**Cost:** O(n × k) model fits for forward selection (n features, select k). Expensive for large n.

### Exhaustive Search

For very small feature sets (<20), try all `2^n` subsets. Use `mlxtend.feature_selection.ExhaustiveFeatureSelector`.

---

## Embedded Methods

Feature selection baked into the model training — efficient, no separate step.

### L1 Regularization (Lasso)

Lasso drives irrelevant feature weights to exactly zero.

```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)
selected = np.where(lasso.coef_ != 0)[0]
X_sel = X_train[:, selected]
```

For classification: `LogisticRegression(penalty='l1', solver='liblinear')`.

**Elastic Net:** Combines L1 + L2 — better when features are correlated.

### Tree-Based Feature Importance

Decision trees and ensembles compute impurity-based feature importance (mean decrease impurity / Gini importance).

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=feature_names)
top_features = importances.nlargest(20).index
X_sel = X_train[top_features]
```

**Caution:** MDI importance is biased toward high-cardinality features. Use **permutation importance** instead:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=42)
perm_imp = pd.Series(result.importances_mean, index=feature_names)
```

### SelectFromModel

Automatic threshold-based selection using any estimator with `feature_importances_` or `coef_`.

```python
from sklearn.feature_selection import SelectFromModel

sel = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold='median')
sel.fit(X_train, y_train)
X_sel = sel.transform(X_train)
```

---

## SHAP-Based Selection

Use SHAP values (model-agnostic, accounts for feature interactions) to rank features globally.

```python
import shap

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)
# For multi-class, average across classes
mean_shap = np.abs(shap_values).mean(axis=0)
top_features = np.argsort(mean_shap)[-20:]
```

**Advantages over MDI:** Considers interaction effects, not biased by cardinality, consistent with model behavior.

---

## Dimensionality Reduction vs Feature Selection

| | Feature Selection | Dimensionality Reduction |
|---|---|---|
| Interpretability | High (original features) | Low (new axes) |
| Information loss | Possible (removes features) | Minimized (projects) |
| Examples | RFE, Lasso, MI | PCA, UMAP, t-SNE |
| Use case | Need original feature names | Purely for model input |

---

## Practical Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_sel', SelectFromModel(LassoCV(cv=5))),   # embedded
    ('model', RandomForestClassifier()),
])
pipe.fit(X_train, y_train)
```

Always fit the selector on training data only; transform both train and test using the fitted selector.

---

## Key Interview Points

- Filter methods are fast but ignore feature interactions; good for initial pruning.
- RFE with cross-validation (RFECV) is reliable but expensive — use with fast models (linear, shallow trees).
- Lasso is the standard embedded method for linear models; sets irrelevant coefficients to exactly zero.
- Tree importance (MDI) is biased; prefer permutation importance or SHAP.
- Always apply feature selection inside a CV fold — fitting the selector on all training data before CV leaks information.
- For high-dimensional sparse data (text): chi-squared or mutual information filter, then Lasso.
