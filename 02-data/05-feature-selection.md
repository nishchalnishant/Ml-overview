---
module: Classical ML
topic: Feature Selection
subtopic: ""
status: unread
tags: [classicalml, ml, feature-selection]
---
# Feature Selection

---

## The Problem Feature Selection Solves

**The problem**: You have 500 features. Some are causal. Some are correlated with the target by coincidence in this dataset but won't generalize. Some are redundant copies of each other. Some are pure noise. A model trained on all 500 will fit noise, overfit, and generalize worse than a model trained on the 30 features that actually matter. It will also be slower, harder to debug, and impossible to explain.

**The core insight**: More features is not always better. Every irrelevant feature the model must ignore is a chance for the model to find a spurious pattern. Reducing to the features that carry genuine signal — and pruning redundant, noisy, or causally downstream features — improves generalization, training speed, and interpretability simultaneously.

**Three families of approach**: Filter methods score features independently of the model. Wrapper methods use a model's actual performance to evaluate feature subsets. Embedded methods perform selection as part of model training.

---

## Filter Methods

**The problem they solve**: You have thousands of features and need to quickly prune the obvious dead weight before spending compute on model training.

**The core insight**: A feature that carries no variance, no correlation with the target, and no statistical dependence on the label can be removed without loss, regardless of what model you use.

### Variance Threshold

**The problem**: A feature with near-zero variance is constant — it carries no discriminative information. It wastes a model weight.

**The core insight**: If a feature barely changes across samples, it cannot contribute to predicting a label that does change.

**The mechanics**: Compute the variance of each feature across the training set. Drop any feature whose variance falls below a threshold.

```python
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.01)
X_sel = sel.fit_transform(X)
```

**What breaks**: Binary features with 99% of samples in one class have variance = 0.01 × 0.99 ≈ 0.01 — they might be dropped even if that 1% is highly predictive. Set the threshold conservatively for binary features.

---

### Correlation-Based Redundancy Removal

**The problem**: Feature A and feature B are 0.97 correlated. Including both gives the model no new information but doubles the noise and can cause multicollinearity in linear models.

**The core insight**: Among a group of highly correlated features, keep one representative and drop the rest. They are measuring roughly the same underlying signal.

**The mechanics**: Compute the pairwise Pearson correlation matrix among features. Identify pairs with |r| > threshold. For each such pair, drop one.

```python
import pandas as pd
import numpy as np

corr_matrix = pd.DataFrame(X).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X_reduced = pd.DataFrame(X).drop(columns=to_drop)
```

**What breaks**: Pearson correlation only detects linear relationships. Two features can be highly redundant non-linearly (e.g., one is the square of the other) while having zero Pearson correlation. Mutual information handles this.

---

### Mutual Information

**The problem**: A feature and the target are non-linearly related. Pearson correlation is zero. A naive filter would remove the feature. But it carries genuine predictive signal.

**The core insight**: Mutual information measures statistical dependence in any form — not just linear. It asks: how much does knowing this feature reduce uncertainty about the label?

**The mechanics**: Estimate mutual information $I(X_i; Y) = \sum P(x, y) \log \frac{P(x,y)}{P(x)P(y)}$ using k-nearest-neighbor density estimation. Rank features by their MI score with the target; select top-k.

```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest

sel = SelectKBest(mutual_info_classif, k=20)
X_sel = sel.fit_transform(X_train, y_train)
# For regression targets:
# sel = SelectKBest(mutual_info_regression, k=20)
```

**What breaks**: MI scores are estimated from finite data — high-variance estimates for small datasets. MI does not account for *interactions between features*: a feature might carry no individual MI with the target, but together with another feature it becomes highly predictive. Filter methods are blind to this.

---

### Chi-Squared Test

**The problem**: You have non-negative features and a classification target. You want to test whether each feature is statistically independent of the class label.

**The core insight**: If a feature and the target label are independent, the cross-tabulation of their values should look like what you'd expect by chance. Chi-squared measures the deviation from that expected table.

**The mechanics**: Bin the feature into categories (or use directly if already categorical). Compute the chi-squared statistic testing independence between the feature and the label. Low p-value → dependent → likely informative.

```python
from sklearn.feature_selection import chi2, SelectPercentile

sel = SelectPercentile(chi2, percentile=50)
X_sel = sel.fit_transform(X_train, y_train)   # X must be non-negative
```

**What breaks**: Requires non-negative features (fails on z-scored data). Only valid for classification targets. Assumes the label is discrete — doesn't extend to regression without binning.

---

### ANOVA F-Statistic

**The problem**: You have continuous features and want to measure the linear association between each feature and a categorical target.

**The core insight**: If a feature has a different mean across classes, it carries discriminative signal. ANOVA tests whether the between-class variance is large relative to within-class variance.

**The mechanics**: For each feature, compute the F-statistic = (between-group variance) / (within-group variance). High F → feature means differ significantly across classes.

```python
from sklearn.feature_selection import f_classif

scores, pvalues = f_classif(X_train, y_train)
mask = pvalues < 0.05
X_sel = X_train[:, mask]
```

**What breaks**: ANOVA is a linear test. A feature that perfectly separates classes with a U-shaped relationship (high values in both extremes) will show a near-zero F-statistic. Use mutual information when non-linear relationships are expected.

---

## Wrapper Methods

**The problem filter methods ignore**: A feature that is individually useless might be highly informative in combination with another feature. Filter methods evaluate features in isolation — they can't detect this.

**The core insight**: Let the model tell you which features are valuable by actually training on subsets and measuring generalization performance. This is more expensive but makes selection a function of the actual learning algorithm.

### Recursive Feature Elimination (RFE)

**The problem**: You want to find the best subset of features for a specific model, removing the least useful features iteratively.

**The core insight**: Train the model, identify the least important feature using its coefficient or importance, remove it, repeat. At each step, the model re-ranks the remaining features — importance estimates sharpen as irrelevant features are removed.

**The mechanics**: Train on all features, rank by |coefficient| or importance, remove the bottom feature (or bottom fraction), retrain on the remainder. Repeat until the target number of features is reached. With cross-validation (RFECV), the target number is chosen automatically.

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression

rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=10)
rfe.fit(X_train, y_train)
print(rfe.support_)   # boolean mask of selected features

rfecv = RFECV(estimator=LogisticRegression(), cv=5, scoring='roc_auc')
rfecv.fit(X_train, y_train)
print(f"Optimal features: {rfecv.n_features_}")
```

**What breaks**: RFE runs one full model training per elimination step — O(n_features) fits minimum. Slow with slow models. Use fast linear models (logistic regression, linear SVM) as the estimator for large feature sets. MDI-based RFE on random forests can be biased toward high-cardinality features.

---

### Sequential Feature Selection

**The problem**: You want to find the best feature subset using actual CV performance as the criterion, not importance scores.

**The core insight**: Greedy search over subsets. Forward selection starts empty and adds the feature that most improves CV score. Backward elimination starts with all features and removes the least damaging one.

**The mechanics**: At each step, evaluate all possible single additions (forward) or removals (backward) via cross-validation. Keep the change that most improves the score. Stop when the target number of features is reached.

```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

sfs = SequentialFeatureSelector(
    RandomForestClassifier(), n_features_to_select=10,
    direction='forward', cv=5, n_jobs=-1
)
sfs.fit(X_train, y_train)
X_sel = sfs.transform(X_train)
```

**What breaks**: O(n_features × k) model fits for forward selection of k features from n. With 200 features and a slow model, this is 200 fits just for the first step. Use with fast estimators. Does not find globally optimal subsets — greedy choices can miss synergistic feature combinations.

---

## Embedded Methods

**The problem**: Wrapper methods separate feature selection from model training — they train the model many times on different subsets. Embedded methods bake selection into training itself, selecting features as a byproduct of fitting the model — one pass, no extra cost.

### L1 Regularization (Lasso)

**The problem**: You have 100 features in a linear model and want to identify which ones matter. With L2 regularization, all weights shrink toward zero but none reach exactly zero — you can't interpret the result as feature selection.

**The core insight**: The L1 penalty (sum of absolute values of weights) has a geometric property that L2 lacks: its constraint region has corners aligned with the axes. Optimization lands on a corner where many weights are *exactly* zero. The model selects its own features during training.

**The mechanics**: Minimize $\text{Loss} + \lambda \sum_j |w_j|$. Features with weak signal get driven to exactly zero. Features with strong signal survive. $\lambda$ controls sparsity — larger $\lambda$ removes more features.

```python
from sklearn.linear_model import LassoCV
import numpy as np

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)
selected = np.where(lasso.coef_ != 0)[0]
X_sel = X_train[:, selected]
```

For classification: `LogisticRegression(penalty='l1', solver='liblinear', C=...)`.

**Elastic Net**: Combines L1 (sparsity) and L2 (grouped selection). Better when many correlated features should be selected together — Lasso tends to pick one arbitrarily from a correlated group.

**What breaks**: When features are highly correlated, Lasso picks one from the group arbitrarily — the chosen feature depends on the random seed. The coefficient path is not unique. Lasso also assumes the true model is sparse; if many features contribute small effects, it will underselect.

---

### Tree-Based Feature Importance

**The problem**: You trained a random forest and want to know which features it relied on. You also want to use those importance scores to prune irrelevant features.

**The core insight**: Decision trees quantify how much each feature reduces impurity across all splits where it was used. Summing this over all trees in a forest produces a global importance ranking.

**The mechanics — Mean Decrease Impurity (MDI)**: For each feature, sum the weighted impurity decrease across all nodes in all trees where that feature was used to split. Normalize to sum to 1.

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=feature_names)
top_features = importances.nlargest(20).index
```

MDI has a known bias: **high-cardinality features get higher importance scores** even when they contribute no genuine signal. A random ID column with thousands of unique values will rank near the top.

**Permutation importance** is the bias-free alternative: shuffle one feature at a time, measure the drop in validation performance.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=42)
perm_imp = pd.Series(result.importances_mean, index=feature_names)
```

**What breaks**: Permutation importance is measured on validation performance — correlated features protect each other. If feature A and feature B carry the same signal, permuting A doesn't hurt much because B compensates. Both may show low permutation importance despite being jointly critical.

---

### SelectFromModel

**The problem**: You want to apply any model's importance scores as an automatic feature selection step inside a sklearn pipeline.

**The core insight**: Wrap any estimator that has `feature_importances_` or `coef_` attributes. Features below a threshold are dropped.

```python
from sklearn.feature_selection import SelectFromModel

sel = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold='median')
sel.fit(X_train, y_train)
X_sel = sel.transform(X_train)
```

**What breaks**: Threshold='median' drops exactly half the features by definition — not because the bottom half are uninformative, but because median was chosen. Use a meaningful threshold (e.g., mean importance, or a value where the importance distribution shows a natural break).

---

## SHAP-Based Selection

**The problem**: MDI importance is biased. Permutation importance is better but misses redundancy. You want feature importances that are consistent with the model's actual predictions, account for feature interactions, and are not biased by cardinality.

**The core insight**: SHAP (SHapley Additive exPlanations) assigns each feature a contribution to each prediction by fairly distributing the prediction over all possible feature orderings. The mean absolute SHAP value across all samples gives a global importance that is grounded in the model's actual behavior.

**The mechanics**: For tree models, TreeSHAP computes exact SHAP values in polynomial time. Average |SHAP value| across all samples for each feature to get global importance.

```python
import shap
import numpy as np

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)
mean_shap = np.abs(shap_values).mean(axis=0)
top_features = np.argsort(mean_shap)[-20:]
```

**What breaks**: SHAP values sum to the prediction minus the baseline — they are consistent and locally accurate, but they can still be confounded by correlated features. High correlated features split SHAP credit between them, making each look less important than the combination truly is.

---

## Feature Selection vs Dimensionality Reduction

These solve related but different problems:

| | Feature Selection | Dimensionality Reduction |
|---|---|---|
| Output | Subset of original features | New transformed features (linear combinations, embeddings) |
| Interpretability | High — original feature names preserved | Low — components have no direct meaning |
| Information loss | Drops features entirely | Projects into lower-dimensional space minimizing loss |
| Examples | RFE, Lasso, MI, SHAP | PCA, UMAP, t-SNE |
| Use when | Original feature names needed for explanation | Pure model input compression, visualization |

---

## Pipeline Integration

**The problem**: Feature selection must be fit on training data only. If you fit a selector on the full dataset before cross-validation, the validation fold's information has influenced which features were selected — information leakage.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('scaler',       StandardScaler()),
    ('feature_sel',  SelectFromModel(LassoCV(cv=5))),
    ('model',        RandomForestClassifier()),
])
# cross_val_score will re-fit the selector on each training fold
scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
```

**What breaks**: Not using a pipeline — fitting the feature selector outside CV, then passing pre-selected features into the CV loop. The selector has seen all folds; its selection is biased toward features that look good in the validation set.
