---
module: Classical Ml
topic: Supervised Learning
subtopic: ""
status: unread
tags: [classicalml, ml, supervised-learning]
---
# Supervised Learning

---

## Algorithm Choice Blueprint

| Task | Category | Key Algorithm | Best For |
| :--- | :--- | :--- | :--- |
| **Regression** | Linear | Linear Regression | Interpretability, Baseline |
| **Regression** | Non-Linear | Random Forest / XGBoost | Complex patterns, Tabular data |
| **Classification** | Probabilistic | Logistic Regression | Probability estimation, baselines |
| **Classification** | High-Margin | SVM | High-dim data, clear separation |
| **Classification** | Fast Baseline | Naive Bayes | Text data, small datasets |
| **Classification/Regression** | Tree Ensemble | Gradient Boosting | Winning on most tabular benchmarks |

---

## Linear Regression

**The problem**: you have a target variable and a set of inputs, and you want the simplest possible model that explains the relationship. Without any structure imposed, fitting any arbitrary curve to your data would perfectly memorize it and predict nothing on new inputs.

**The core insight**: if you assume the relationship is linear and penalize the model by how far its predictions are from the truth, the optimal line is the one that minimizes the total squared error. Squared (not absolute) error because it is differentiable everywhere and assigns disproportionately more penalty to large mistakes.

**The mechanics**: minimize the sum of squared residuals (OLS):

$$J(w) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

This has a closed-form solution because $J$ is a convex quadratic in $w$:

$$w = (X^T X)^{-1} X^T y$$

Use gradient descent when $n$ (features) is large — matrix inversion is $O(n^3)$.

**What breaks**: linearity, homoscedasticity, no multicollinearity, i.i.d. errors. When these fail: transform features (log, polynomial), add regularization (Ridge/Lasso).

---

## Logistic Regression

**The problem**: linear regression can output any real number, but probabilities must lie in $[0, 1]$. Worse, using MSE as the loss for classification produces a non-convex surface with many local minima — gradient descent gets stuck.

**The core insight**: squash the linear output through a sigmoid to get probabilities, then define a loss that is convex in the weights. Log-Loss (cross-entropy) is that loss: it is convex and also penalizes confident-but-wrong predictions exponentially harder than uncertain ones.

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

**What breaks**: when the decision boundary is non-linear (features need polynomial expansion or a different model), when classes are severely imbalanced (use class weights or PR-AUC), when features are on very different scales without normalization.

---

## Support Vector Machines (SVM)

**The problem**: many classifiers find *a* boundary that separates classes, but there are infinitely many such boundaries. Which one generalizes best to unseen data? A boundary that barely clears training points is fragile — small perturbations in new inputs will cross it.

**The core insight**: the boundary with the largest gap (margin) between itself and the nearest training points from each class is the most robust. Maximizing this margin is equivalent to minimizing $\|w\|$, which constrains model complexity independently of the number of features — this is why SVMs work well in high dimensions.

**The mechanics**:

Hard-margin SVM — maximize margin $\frac{2}{\|w\|}$ subject to:
$$y^{(i)}(w^T x^{(i)} + b) \geq 1 \quad \forall i$$

Soft-margin (practical) — introduce slack $\xi_i \geq 0$ to allow some misclassification:
$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C \sum_i \xi_i$$

Kernel trick: when classes are not linearly separable, map inputs to a higher-dimensional space where they are. The kernel $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$ computes this inner product without ever constructing $\phi(x)$ explicitly — the optimization only needs dot products, not the mapped coordinates.

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

**What breaks**: SVMs do not scale — training is $O(n^2)$ to $O(n^3)$ in the number of samples, and prediction requires evaluating kernels against all support vectors. They also have no native probability output (sigmoid calibration is needed). When `C` is too large, you overfit; when `gamma` is too large with RBF, each training point becomes its own island.

---

## Decision Trees

**The problem**: linear models cannot capture thresholds — if a patient's blood pressure is dangerous above a certain value, no linear combination of features will cleanly express that. You need a model that can partition the input space into arbitrary rectangular regions.

**The core insight**: recursively split the data on the single feature and threshold that most reduces the uncertainty (impurity) of the resulting groups. Uncertainty is measured by how mixed the class labels are within each resulting partition.

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

## Evaluation Metrics

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
> **Production Recommendation:** For tabular data with high cardinality, always start with **XGBoost** or **LightGBM**. For image or sequential data, skip traditional ML and move to the [Deep Learning Foundation](../03-deep-learning/README.md).

---

## Canonical Interview Q&As

**Q: Derive the logistic regression gradient update and explain the connection to cross-entropy loss.**  
A: Logistic regression models p(y=1|x) = σ(wᵀx) = 1/(1 + e^{-wᵀx}). The negative log-likelihood (cross-entropy loss) for n samples is L(w) = -Σ[y_i·log(σ(wᵀx_i)) + (1-y_i)·log(1-σ(wᵀx_i))]. Taking the gradient: ∂L/∂w = Σ(σ(wᵀx_i) - y_i)·x_i. This has a beautiful form: the gradient is simply the sum of (prediction - label) weighted by the input features. The prediction error directly tells you how to update each weight. Connection to cross-entropy: the log-likelihood of a Bernoulli distribution is exactly cross-entropy. So logistic regression is MLE under a Bernoulli likelihood assumption. The sigmoid function arises naturally from assuming log-odds are linear: log(p/(1-p)) = wᵀx → p = σ(wᵀx). When output is multi-class, generalize to softmax + categorical cross-entropy — same structure, multinomial MLE.

**Q: How do you handle class imbalance at the algorithm level vs the data level?**  
A: **Algorithm level** (preferred, doesn't discard data): (1) class_weight='balanced' in sklearn scales the loss by N/(2·n_k) per class — minority class errors count more, equivalent to up-weighting those samples; (2) for trees: adjust sample_weight in fit(); (3) threshold tuning — train with default threshold but choose threshold to optimize F1 or business metric on validation set; (4) focal loss (for neural nets): (1-p_t)^γ scaling downweights easy majority class examples. **Data level**: oversampling minority (SMOTE generates synthetic samples by interpolating in feature space — helps for continuous features, hurts for categorical), undersampling majority (data loss), class-balanced mini-batch sampling. **Metric level** — often overlooked but critical: always evaluate with PR-AUC instead of ROC-AUC for severe imbalance (ROC-AUC is optimistic because it normalizes by true negatives, which are abundant). Rule of thumb: for imbalance < 10:1, class_weight='balanced' is sufficient; for > 100:1, combine algorithm-level with threshold tuning; for > 1000:1 (fraud), use PR-AUC + focal loss + business-metric-optimized threshold.

**Q: A classification model has 95% accuracy but the product team says it's useless. Why might this be, and how do you debug it?**  
A: Classic class imbalance trap. If 95% of examples are negative class, a model that always predicts negative achieves 95% accuracy without learning anything. Debug steps: (1) check class distribution — if majority class > 90%, accuracy is misleading; (2) compute confusion matrix — if precision or recall on the minority class is near zero, the model is failing on the important cases; (3) switch to appropriate metric: F1-score for balanced precision/recall, PR-AUC for overall performance across thresholds, or a business metric (e.g., revenue recovered for fraud detection); (4) check if the model learned a trivial solution — inspect prediction distribution: if all predictions are >0.9 or <0.1, the model isn't discriminating; (5) inspect feature importances — if the top feature is a proxy for the label or a data leakage feature, the accuracy is inflated. Fix: use class_weight='balanced', tune classification threshold to maximize the business metric, retrain with proper evaluation on held-out stratified splits.

## Flashcards

**Gini Importance?** #flashcard
total reduction of Gini impurity provided by a feature across all trees. Fast but biased toward high-cardinality features.

**Permutation Importance?** #flashcard
model score drop when a feature's values are randomly shuffled. Slower but more reliable and model-agnostic.

**SHAP values?** #flashcard
game-theoretically fair attribution. Best choice when explanation quality matters.
