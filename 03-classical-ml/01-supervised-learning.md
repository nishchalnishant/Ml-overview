---
module: Classical ML
topic: Supervised Learning
subtopic: ""
status: unread
tags: [classicalml, ml, supervised-learning]
---
# Supervised Learning

---

## Executive Summary & Cheatsheet

### Algorithm Table

| Algorithm | Best for | Key hyperparameters | Watch out for |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Continuous target, interpretability, baseline | Regularization α (Ridge/Lasso) | Linearity + homoscedasticity assumptions; multicollinearity |
| **Logistic Regression** | Binary/multi-class, calibrated probs, strong baseline | C (inverse reg), solver, max_iter | Use log-loss not MSE; won't fit non-linear boundaries |
| **SVM** | High-dim, small-medium N, clear margin needed | C, kernel (rbf/poly/linear), γ | Normalize features; slow O(N²–N³) training; kernel matters |
| **Decision Tree** | Interpretable rules, mixed feature types | max_depth, min_samples_leaf | Overfits without pruning; unstable |
| **Random Forest** | Tabular data, robust out-of-box, parallel | n_estimators, max_depth | Slow predict at scale; harder to interpret |
| **Gradient Boosting** | Best tabular accuracy (XGBoost/LightGBM) | learning_rate, n_estimators, max_depth | Overfits if over-tuned; sequential |

### Key Distinctions
- **Why log-loss for logistic regression?** MSE on probabilities is non-convex. Cross-entropy is convex and brutally punishes confident wrong predictions.
- **Kernel trick (SVM):** Computes similarity in high-D space without explicitly building coordinates.
- **Bagging (Random Forest):** Parallel trees, reduces variance, targets original labels.
- **Boosting (XGBoost):** Sequential trees, reduces bias, targets residuals.

### Evaluation Metrics
| Metric | Formula | Use when |
| :--- | :--- | :--- |
| **Accuracy** | (TP+TN)/N | Balanced classes only |
| **Precision** | TP/(TP+FP) | False alarms are costly |
| **Recall** | TP/(TP+FN) | Misses are costly (fraud, cancer) |
| **F1** | 2·P·R/(P+R) | Imbalanced classes, want balance |
| **ROC-AUC** | Area under TPR vs FPR | Ranking quality across thresholds |
| **MSE / RMSE** | mean((y−ŷ)²) | Regression; penalizes large errors |

### When to reach for what
- **CSV, unknown shape, 1 hour** → Logistic/linear baseline → Gradient boosting with CV
- **Text, fast inference** → Naive Bayes or Logistic with TF-IDF
- **High-dim, small N, clear margin** → SVM
- **Interpretability required** → Single decision tree or logistic regression
- **Best accuracy on tabular** → XGBoost / LightGBM

---

## Deep Dive

## Linear Regression

**The problem**: find the simplest model that explains a continuous target from inputs, without memorizing the training data.

**The core insight**: assume the relationship is linear and minimize squared error. Squared (not absolute) error is differentiable everywhere and penalizes large mistakes more.

**The mechanics**: minimize the sum of squared residuals (OLS):

$$J(w) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

Closed-form solution (convex quadratic in $w$):

$$w = (X^T X)^{-1} X^T y$$

Use gradient descent when $n$ (features) is large — matrix inversion is $O(n^3)$.

**What breaks**: linearity, homoscedasticity, no multicollinearity, i.i.d. errors. Fix: transform features (log, polynomial), add regularization (Ridge/Lasso).

---

## Logistic Regression

**The problem**: linear regression outputs any real number, but probabilities must lie in $[0, 1]$. Using MSE for classification also gives a non-convex loss surface.

**The core insight**: squash the linear output through a sigmoid to get probabilities, then use log-loss (cross-entropy), which is convex and penalizes confident-but-wrong predictions much harder than uncertain ones.

**The mechanics**:

Sigmoid:
$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = w^T x + b$$

Log-Loss (Binary Cross-Entropy):
$$J(w) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]$$

**Key hyperparameters**: regularization strength `C` (inverse of λ), solver (`lbfgs` for small data, `saga` for large/sparse).

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]
```

**What breaks**: non-linear decision boundaries (need polynomial features or a different model), severe class imbalance (use class weights or PR-AUC), unscaled features.

---

## Support Vector Machines (SVM)

**The problem**: many boundaries can separate two classes. Which generalizes best? A boundary that barely clears training points is fragile.

**The core insight**: the boundary with the largest margin to the nearest points from each class is most robust. Maximizing the margin is equivalent to minimizing $\|w\|$, which is why SVMs work well in high dimensions.

**The mechanics**:

Hard-margin SVM — maximize margin $\frac{2}{\|w\|}$ subject to:
$$y^{(i)}(w^T x^{(i)} + b) \geq 1 \quad \forall i$$

Soft-margin (practical) — introduce slack $\xi_i \geq 0$ to allow some misclassification:
$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C \sum_i \xi_i$$

Kernel trick: when classes aren't linearly separable, map inputs to a higher-dimensional space where they are. The kernel $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$ computes this inner product without ever constructing $\phi(x)$ explicitly.

| Kernel | Formula | Use case |
| :--- | :--- | :--- |
| **Linear** | $K(x,z) = x^T z$ | Linearly separable, high-dim (text) |
| **RBF/Gaussian** | $K(x,z) = \exp(-\gamma \|x-z\|^2)$ | Non-linear, low-dim |
| **Polynomial** | $K(x,z) = (x^T z + c)^d$ | Degree-d boundaries |

**Key hyperparameters**: `C` (margin vs misclassification penalty), `gamma` (RBF width). Use grid search with cross-validation.

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm = SVC(kernel='rbf', probability=True)
grid = GridSearchCV(svm, {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}, cv=5)
grid.fit(X_train, y_train)
```

**What breaks**: SVMs don't scale — training is $O(n^2)$ to $O(n^3)$, and prediction evaluates kernels against all support vectors. No native probability output (needs sigmoid calibration). Large `C` overfits; large `gamma` with RBF makes each point its own island.

---

## Decision Trees

**The problem**: linear models can't capture thresholds — e.g., a value that's dangerous only above a cutoff. You need a model that partitions the input space into regions.

**The core insight**: recursively split on the feature and threshold that most reduces impurity (how mixed the class labels are) in the resulting groups.

**The mechanics**:

Gini impurity at a node:
$$G = 1 - \sum_{k} p_k^2$$

Entropy-based information gain:
$$IG = H(\text{parent}) - \sum_{\text{child}} \frac{n_{\text{child}}}{n} H(\text{child}), \quad H = -\sum_k p_k \log_2 p_k$$

The split that maximizes information gain (equivalently minimizes weighted child impurity) is chosen at each node. Repeat recursively until a stopping criterion is met.

**What breaks**: trees overfit aggressively — a fully grown tree memorizes every training example. Pruning via `max_depth`, `min_samples_leaf`, and `min_samples_split` limits depth but introduces a different problem: now the tree may be too shallow to capture real patterns. A single tree also has high variance — small changes in training data can produce completely different trees. This is what motivates ensembles.

---

## Tree-Based Ensembles

### Random Forest vs Gradient Boosting

| Feature | Random Forest | Gradient Boosting (XGBoost) |
| :--- | :--- | :--- |
| **Method** | Bagging (parallel) | Boosting (sequential) |
| **Primary Goal** | Reduce variance | Reduce bias |
| **Overfitting** | Hard to overfit | Prone to overfitting |
| **Speed** | Fast (parallelizable) | Slower, but LightGBM speeds this up |
| **Hyperparameter sensitivity** | Low | High |
| **Best for** | Robust baseline, noisy data | Winning competitions, clean tabular data |

**Random Forest** — the problem is that one decision tree is brittle. The insight is that averaging many noisy but independent estimates cancels error. Bootstrap sampling (random rows) plus random feature subsets at each split (random columns) forces trees to be different enough to cancel each other's mistakes — deliberately degrading each individual tree to make the ensemble stronger:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

**Gradient Boosting** — the problem is that shallow models (stumps) are too biased to capture complex patterns. The insight is that you can correct a model's mistakes sequentially: after each tree, fit the next one to the residual error left by the current ensemble. Each new tree $h_m$ is fit to the negative gradient of the loss (pseudo-residuals):

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

### Feature Importance Methods

- **Gini Importance**: total reduction of Gini impurity provided by a feature across all trees. Fast but biased toward high-cardinality features.
- **Permutation Importance**: model score drop when a feature's values are randomly shuffled. Slower but more reliable and model-agnostic.
- **SHAP values**: game-theoretically fair attribution. Best choice when explanation quality matters.

```python
import xgboost as xgb
import shap

model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

---

## K-Nearest Neighbors (KNN)

**The problem**: you want a classifier (or regressor) that makes no assumptions about the shape of the decision boundary, requires no training phase, and naturally handles multi-class problems. But you need it to work well in practice — not just in theory.

**The core insight**: a new point is probably the same class as its nearest neighbors. Skip learning a model entirely. At prediction time, find the $k$ closest training points and let them vote. The decision boundary adapts to whatever shape the data requires, because it is never explicitly computed — it emerges from the training set.

**The mechanics**: for a query point $x$, find the $k$ training points with the smallest distance, then predict:
- **Classification**: majority vote among the $k$ neighbors (weighted or unweighted)
- **Regression**: mean (or weighted mean) of the $k$ neighbors' target values

Distance metrics:
| Metric | Formula | Use when |
|--------|---------|---------|
| **Euclidean** | $\sqrt{\sum_i (x_i - z_i)^2}$ | Continuous features, low-dimensional |
| **Manhattan** | $\sum_i \|x_i - z_i\|$ | Robust to outliers; sparse high-dim features |
| **Minkowski** | $(\sum_i \|x_i - z_i\|^p)^{1/p}$ | Generalization (p=1: Manhattan, p=2: Euclidean) |
| **Cosine** | $1 - \frac{x \cdot z}{\|x\|\|z\|}$ | Text/embeddings; direction matters, not magnitude |
| **Hamming** | Fraction of differing positions | Categorical / binary features |

**Choosing $k$**:
- $k=1$: zero training error but extremely high variance (overfits to noise)
- Large $k$: smoother boundary, higher bias, lower variance
- Rule of thumb: start at $k = \sqrt{n}$; tune via cross-validation
- Odd $k$ for binary classification avoids ties

**Weighted KNN**: weight each neighbor's vote by $1/d^2$ (inverse squared distance). Closer neighbors contribute more — reduces sensitivity to the exact choice of $k$ and improves performance near class boundaries.

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Must normalize — KNN is distance-based
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'))
])
pipe.fit(X_train, y_train)

# Tune k
from sklearn.model_selection import cross_val_score
for k in [3, 5, 7, 11, 15]:
    cv_score = cross_val_score(
        KNeighborsClassifier(n_neighbors=k, weights='distance'), X_train, y_train, cv=5
    ).mean()
    print(f"k={k}: {cv_score:.4f}")
```

**KNN for regression**: the average of the $k$ neighbors' target values. Naturally adapts to local structure — in a region where the target is complex, nearby examples directly encode that complexity, whereas a global linear model would average it away.

**Curse of dimensionality**: KNN's core assumption is that nearby points (in distance) are similar in behavior. In high dimensions, this breaks down catastrophically. The ratio of the volume of a thin shell to the volume of the full sphere approaches 1 as dimensionality grows — meaning almost all points are roughly equidistant from any query point, and "nearest neighbors" are no longer meaningfully close. Beyond ~20 features, KNN degrades sharply unless you apply dimensionality reduction first.

**What breaks**:
- **Scale sensitivity**: always standardize features before KNN — a feature on a 0–10,000 range will dominate the distance calculation over a 0–1 feature
- **Memory**: no training phase means the entire training set must be stored and searched at prediction time — $O(n \cdot d)$ per query
- **Speed**: naive KNN is $O(n \cdot d)$ per query; use KD-trees ($O(d \log n)$ for low-dimensional data) or ball trees for acceleration; approximate nearest neighbor (ANN) libraries (FAISS, HNSW) scale to millions of points
- **High dimensionality**: PCA or UMAP before KNN is essential for $d > 30$
- **Imbalanced classes**: the majority class dominates voting; use class-weighted voting or oversample

**When to use KNN**:
- Small datasets where memorizing the training set is feasible
- Non-parametric baseline — if KNN fails, the data has no local structure to exploit
- Recommendation systems (user-based collaborative filtering is KNN in user space)
- Anomaly detection (a point with no near neighbors is anomalous)

---

## Naive Bayes

**The problem**: to classify a document, you want to ask "what class is most probable given these words?" But jointly modeling all possible word combinations is intractable — the number of combinations explodes exponentially with vocabulary size.

**The core insight**: if you assume features are conditionally independent given the class label (the "naive" assumption), the joint probability decomposes into a product of individual feature likelihoods. This is almost always false in practice — words co-occur, pixels are correlated — yet the classifier still works because it only needs to rank classes, not compute accurate probabilities.

**The mechanics**: apply Bayes' theorem and drop the denominator (constant across classes):

$$P(y | x_1, \ldots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i | y)$$

| Variant | Assumption | Use Case |
| :--- | :--- | :--- |
| **GaussianNB** | $P(x_i|y) \sim \mathcal{N}(\mu, \sigma^2)$ | Continuous features |
| **MultinomialNB** | Count-based features | Text classification (bag-of-words) |
| **BernoulliNB** | Binary features | Document classification |

**What breaks**: the independence assumption is violated in nearly every real domain. When features are highly correlated, the model effectively double-counts evidence and becomes overconfident in its predictions. Calibrated probabilities are usually poor. The assumption also means it cannot model interactions between features, so any signal that only exists when two features are seen together is invisible to it.

---

## Regularization

**The problem**: a model with enough parameters can memorize the training set perfectly, achieving zero training loss. This memorization is useless for generalization — the model has learned the noise in the training data, not the signal.

**The core insight**: add a penalty term to the loss that grows with model complexity. The optimizer now has two competing objectives: fit the data and keep weights small. This tension prevents the model from committing too strongly to individual training examples.

### L1 (Lasso) and L2 (Ridge)

$$J_{L1}(w) = \text{loss} + \lambda \sum_i |w_i| \quad \text{(Lasso)}$$
$$J_{L2}(w) = \text{loss} + \lambda \sum_i w_i^2 \quad \text{(Ridge)}$$

| Property | L1 (Lasso) | L2 (Ridge) |
|----------|-----------|-----------|
| Effect on weights | Drives many to exactly zero (sparse) | Shrinks all weights, none to exactly zero |
| Geometric interpretation | Diamond constraint → corners are zeros | Sphere constraint → smooth shrinkage |
| Gradient at zero | Subgradient (not differentiable) | Zero → smooth everywhere |
| Feature selection | Yes — zero weights = excluded features | No — all features kept |
| Correlated features | Picks one arbitrarily, zeros others | Distributes weight equally |
| Best for | High-dimensional, sparse signal | Correlated features, no sparsity needed |

**ElasticNet** combines both: $\lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2$. Use when you want sparsity but have correlated features (Lasso is unstable with correlated features — the choice of which to keep is arbitrary).

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Ridge(alpha=1.0)                          # alpha = λ
lasso = Lasso(alpha=0.01)                         # small α for mild sparsity
enet  = ElasticNet(alpha=0.01, l1_ratio=0.5)      # l1_ratio: fraction of L1
```

**Bayesian interpretation**: Ridge = MAP estimation with a Gaussian prior on weights ($P(w) \propto e^{-\lambda\|w\|^2}$). Lasso = MAP with a Laplace prior ($P(w) \propto e^{-\lambda|w|}$). Regularization strength λ corresponds to the inverse variance of the prior — strong prior (large λ) → weights pulled strongly toward zero.

**Choosing λ**: use cross-validation. `RidgeCV` and `LassoCV` in sklearn do this efficiently. Log-scale grid is standard: `[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]`.

---

## XGBoost, LightGBM, and CatBoost — Deep Comparison

All three are gradient-boosted tree libraries. Each differs in how trees are grown, how they handle categorical features, and their computational tradeoffs.

### Tree Growth Strategy

| Property | XGBoost | LightGBM | CatBoost |
|----------|---------|----------|----------|
| Growth strategy | Level-wise (breadth-first) | Leaf-wise (best-leaf-first) | Symmetric (oblivious) trees |
| Depth | Controlled by `max_depth` | Controlled by `num_leaves` | Controlled by `depth` |
| Speed | Baseline | 3-10× faster than XGBoost | Slower than LightGBM, competitive with XGBoost |
| Memory | Moderate | Efficient (histogram binning) | Moderate |

**Level-wise (XGBoost)**: expands all leaves at a given depth before going deeper. Produces balanced trees. Avoids overfitting on rare branches.

**Leaf-wise (LightGBM)**: at each step, grows the leaf with the maximum loss reduction, regardless of depth. Produces deeper, more asymmetric trees. Finds the same accuracy with fewer leaves. Can overfit on small datasets — use `min_data_in_leaf` as a safeguard.

**Symmetric/Oblivious trees (CatBoost)**: all nodes at a given depth use the same split (same feature and threshold). This sounds restrictive but has two major benefits: (1) much faster prediction (can be evaluated as a table lookup), (2) more robust to overfitting because the tree structure is heavily regularized.

### Categorical Feature Handling

**XGBoost**: requires manual encoding (OHE, ordinal, target encoding). No native categorical support in the standard API.

**LightGBM**: native categorical support via `categorical_feature` parameter. Internally uses "many-vs-many" splits — partitions categories into two groups to minimize the objective. Much better than OHE for high-cardinality features.

**CatBoost**: best-in-class categorical handling via **Ordered Target Statistics** (also called ordered boosting). For each example, the target encoding is computed using only the examples seen before it (in a random permutation), which prevents target leakage from the current example. This is mathematically rigorous and avoids the target leakage that naive target encoding suffers from.

```python
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# XGBoost — manual categorical encoding required
model_xgb = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='logloss'
)

# LightGBM — native categoricals
model_lgb = lgb.LGBMClassifier(
    n_estimators=500, num_leaves=63, learning_rate=0.05,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8
)
model_lgb.fit(X_train, y_train, categorical_feature=['col_a', 'col_b'])

# CatBoost — pass column indices directly
model_cat = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.05,
    cat_features=[0, 3, 7],  # indices of categorical columns
    verbose=0
)
model_cat.fit(X_train, y_train, eval_set=(X_val, y_val))
```

### Gradient and Hessian Computation

All three use second-order (Newton boosting): the tree at step $m$ is fit to both the gradient $g_i$ and Hessian $h_i$ of the loss:

$$\text{Leaf score} = -\frac{\sum_i g_i}{\sum_i h_i + \lambda}$$

The Hessian term $h_i$ effectively provides a per-example learning rate — examples where the loss is already small (low $h_i$) get smaller updates. This is strictly better than first-order gradient boosting (AdaBoost).

**XGBoost innovation**: regularization terms on leaf scores and number of leaves directly in the objective:
$$L = \sum_i \ell(y_i, \hat{y}_i) + \gamma T + \frac{1}{2}\lambda \sum_j w_j^2$$

where $T$ = number of leaves, $w_j$ = leaf scores. This penalizes tree complexity analytically.

### Key Hyperparameters

| Hyperparameter | XGBoost | LightGBM | Effect |
|---|---|---|---|
| Tree complexity | `max_depth` | `num_leaves` | Capacity of each tree |
| Learning rate | `learning_rate` | `learning_rate` | Step size per tree; lower → need more trees |
| Row sampling | `subsample` | `bagging_fraction` | Reduces variance, speeds training |
| Column sampling | `colsample_bytree` | `feature_fraction` | Reduces correlation between trees |
| Min leaf size | `min_child_weight` | `min_child_samples` | Prevents overfitting on rare splits |
| L2 regularization | `reg_lambda` | `lambda_l2` | Weight shrinkage |
| L1 regularization | `reg_alpha` | `lambda_l1` | Weight sparsity |
| Leaf penalty | `gamma` | — | Minimum gain to make a split |

### When to Use Which

| Situation | Recommendation |
|---|---|
| Many high-cardinality categoricals | CatBoost (ordered target statistics, no leakage) |
| Large dataset (> 1M rows), speed matters | LightGBM (histogram binning, leaf-wise) |
| Need maximum control, research use | XGBoost (most mature API, most tuning options) |
| Small dataset (< 10K) | XGBoost with heavy regularization; all three overfit similarly |
| Fast serving latency | CatBoost (symmetric trees → table lookup prediction) |
| Missing values in data | XGBoost natively handles missing (learns default branch direction) |

**Practical rule**: Start with LightGBM for speed. Switch to CatBoost if you have many categoricals or XGBoost if you need more explicit control.

---

## Evaluation Metrics

> This section covers the core metrics used daily. For comprehensive coverage — MCC, Brier score, ECE, NDCG, MAP, MRR, calibration curves, and statistical significance testing — see **[ml-evaluation-metrics.md](../04-evaluation/01-ml-evaluation-metrics.md)**.

### Classification

**The problem**: accuracy is useless when classes are imbalanced. A model that always predicts "not fraud" on a 1%-fraud dataset achieves 99% accuracy while being worthless.

**The core insight**: precision and recall separately measure two different failure modes — predicting positive when you shouldn't (false positives) vs missing true positives (false negatives). The right metric depends on which failure mode is more costly.

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}, \quad F_\beta = \frac{(1+\beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

Use $\beta > 1$ when recall matters more (e.g., medical diagnosis); $\beta < 1$ when precision matters more.

**ROC-AUC**: threshold-agnostic, measures the model's ability to rank positive examples above negative ones. Optimistic with imbalanced data because it weights both classes equally.

**PR-AUC**: area under the Precision-Recall curve. Better for imbalanced datasets — directly measures quality on the minority class.

### Regression

| Metric | Formula | Notes |
| :--- | :--- | :--- |
| **MAE** | $\frac{1}{m}\sum|y - \hat{y}|$ | Robust to outliers, interpretable |
| **MSE** | $\frac{1}{m}\sum(y - \hat{y})^2$ | Penalizes large errors heavily |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same units as target |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained |

```python
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
```

---

## Common Failure Modes

| Problem | Symptom | Fix |
| :--- | :--- | :--- |
| **Overfitting** | High train accuracy, low val accuracy | Regularization, more data, simpler model |
| **Underfitting** | Low train and val accuracy | More capacity, feature engineering, less regularization |
| **Data leakage** | Suspiciously high CV score | Check preprocessing happens inside CV fold |
| **Class imbalance** | High accuracy, poor F1 on minority | Resampling (SMOTE), class weights, PR-AUC |
| **Multicollinearity** | Unstable coefficients in linear models | VIF analysis, Ridge regularization, PCA |
| **Feature scale issues** | SVM/KNN performs poorly | Standardize/normalize before fitting |

---

> [!TIP]
> **Production Recommendation:** For tabular data with high cardinality, always start with **XGBoost** or **LightGBM**. For image or sequential data, skip traditional ML and move to the [Deep Learning Foundation](../05-deep-learning-core/README.md).

---

## Canonical Interview Q&As

**Q: Derive the logistic regression gradient update and explain the connection to cross-entropy loss.**  
A: Logistic regression models p(y=1|x) = σ(wᵀx) = 1/(1 + e^{-wᵀx}). The negative log-likelihood (cross-entropy loss) for n samples is L(w) = -Σ[y_i·log(σ(wᵀx_i)) + (1-y_i)·log(1-σ(wᵀx_i))]. Taking the gradient: ∂L/∂w = Σ(σ(wᵀx_i) - y_i)·x_i. This has a beautiful form: the gradient is simply the sum of (prediction - label) weighted by the input features. The prediction error directly tells you how to update each weight. Connection to cross-entropy: the log-likelihood of a Bernoulli distribution is exactly cross-entropy. So logistic regression is MLE under a Bernoulli likelihood assumption. The sigmoid function arises naturally from assuming log-odds are linear: log(p/(1-p)) = wᵀx → p = σ(wᵀx). When output is multi-class, generalize to softmax + categorical cross-entropy — same structure, multinomial MLE.

**Q: How do you handle class imbalance at the algorithm level vs the data level?**  
A: See [11-imbalanced-data.md](../02-data/06-imbalanced-data.md) for the full algorithm-level (class weights, threshold tuning, focal loss), data-level (SMOTE, undersampling), and metric-level (PR-AUC over ROC-AUC) breakdown, plus imbalance-ratio rules of thumb.

**Q: A classification model has 95% accuracy but the product team says it's useless. Why might this be, and how do you debug it?**  
A: Classic class imbalance trap. If 95% of examples are negative class, a model that always predicts negative achieves 95% accuracy without learning anything. Debug steps: (1) check class distribution — if majority class > 90%, accuracy is misleading; (2) compute confusion matrix — if precision or recall on the minority class is near zero, the model is failing on the important cases; (3) switch to appropriate metric: F1-score for balanced precision/recall, PR-AUC for overall performance across thresholds, or a business metric (e.g., revenue recovered for fraud detection); (4) check if the model learned a trivial solution — inspect prediction distribution: if all predictions are >0.9 or <0.1, the model isn't discriminating; (5) inspect feature importances — if the top feature is a proxy for the label or a data leakage feature, the accuracy is inflated. Fix: use class_weight='balanced', tune classification threshold to maximize the business metric, retrain with proper evaluation on held-out stratified splits.


