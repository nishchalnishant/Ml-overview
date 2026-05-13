# Supervised Learning Mastery (Deep-Dive)

This track provides a comprehensive exploration of supervised learning algorithms, covering the mathematical "why" and the production "how."

---

# 1. Algorithm Choice Blueprint

| Task | Category | Key Algorithm | Best For |
| :--- | :--- | :--- | :--- |
| **Regression** | Linear | Linear Regression | Interpretability, Baseline |
| **Regression** | Non-Linear | Random Forest / XGBoost | Complex patterns, Tabular data |
| **Classification** | Probabilistic | Logistic Regression | Probability estimation, baselines |
| **Classification** | High-Margin | SVM | High-dim data, clear separation |
| **Classification** | Fast Baseline | Naive Bayes | Text data, small datasets |
| **Classification/Regression** | Tree Ensemble | Gradient Boosting | Winning on most tabular benchmarks |

---

# 2. Linear Models

## Linear Regression

**Objective:** Minimize sum of squared residuals (Ordinary Least Squares):

$$J(w) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

**Closed-form solution (OLS):**

$$w = (X^T X)^{-1} X^T y$$

Use gradient descent when $n$ (features) is large — matrix inversion is $O(n^3)$.

**Assumptions:** linearity, homoscedasticity, no multicollinearity, errors are i.i.d.

When assumptions break: transform features (log, polynomial), add regularization (Ridge/Lasso).

## Logistic Regression

### Q: Why is Log-Loss used instead of MSE?

MSE for classification is **non-convex** — gradient descent can get stuck. **Log-Loss (Cross-Entropy)** is convex, ensuring a global optimum. It also penalizes confident-but-wrong predictions far more aggressively.

**Sigmoid function:**

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = w^T x + b$$

**Log-Loss (Binary Cross-Entropy):**

$$J(w) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]$$

**Key hyperparameters:** regularization strength `C` (inverse of λ), solver (`lbfgs` for small data, `saga` for large/sparse).

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]
```

---

# 3. Support Vector Machines (SVM)

### Q: How does the kernel trick work?

**Hard-margin SVM objective** — maximize the margin $\frac{2}{\|w\|}$ subject to:

$$y^{(i)}(w^T x^{(i)} + b) \geq 1 \quad \forall i$$

**Soft-margin (practical):** introduce slack variables $\xi_i \geq 0$:

$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C \sum_i \xi_i$$

**Kernel trick:** instead of explicitly mapping $x \to \phi(x)$, compute $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$ directly.

Common kernels:

| Kernel | Formula | Use case |
| :--- | :--- | :--- |
| **Linear** | $K(x,z) = x^T z$ | Linearly separable, high-dim (text) |
| **RBF/Gaussian** | $K(x,z) = \exp(-\gamma \|x-z\|^2)$ | Non-linear, low-dim |
| **Polynomial** | $K(x,z) = (x^T z + c)^d$ | Degree-d boundaries |

**Key hyperparameters:** `C` (margin vs misclassification penalty), `gamma` (RBF width). Use grid search with cross-validation.

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm = SVC(kernel='rbf', probability=True)
grid = GridSearchCV(svm, {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}, cv=5)
grid.fit(X_train, y_train)
```

---

# 4. Decision Trees

**Split criterion — Gini impurity:**

$$G = 1 - \sum_{k} p_k^2$$

**Information gain (Entropy-based):**

$$IG = H(\text{parent}) - \sum_{\text{child}} \frac{n_{\text{child}}}{n} H(\text{child})$$

$$H = -\sum_k p_k \log_2 p_k$$

A split is chosen that maximizes information gain (or equivalently minimizes weighted child impurity).

**Common failure modes:** trees easily overfit — prune with `max_depth`, `min_samples_leaf`, `min_samples_split`.

---

# 5. Tree-Based Ensembles

## Random Forest vs Gradient Boosting

| Feature | Random Forest | Gradient Boosting (XGBoost) |
| :--- | :--- | :--- |
| **Method** | Bagging (parallel) | Boosting (sequential) |
| **Primary Goal** | Reduce variance | Reduce bias |
| **Overfitting** | Hard to overfit | Prone to overfitting |
| **Speed** | Fast (parallelizable) | Slower, but LightGBM speeds this up |
| **Hyperparameter sensitivity** | Low | High |
| **Best for** | Robust baseline, noisy data | Winning competitions, clean tabular data |

**Random Forest:** trains $B$ trees on bootstrap samples, aggregates predictions by voting/averaging.

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

Each tree sees a random subset of features at each split (reduces correlation between trees).

**Gradient Boosting:** each new tree fits the **residuals** of the previous ensemble:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

where $h_m$ is fit to $-\nabla_{F} L$ (negative gradient of loss).

### Feature Importance Methods

- **Gini Importance:** total reduction of Gini impurity provided by a feature across all trees. Fast but biased toward high-cardinality features.
- **Permutation Importance:** model score drop when a feature's values are randomly shuffled. Slower but more reliable and model-agnostic.
- **SHAP values:** game-theoretically fair attribution. Best choice when explanation quality matters.

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

# 6. Naive Bayes

**Bayes' theorem:**

$$P(y | x_1, \ldots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i | y)$$

The "naive" assumption: features are conditionally independent given the class.

| Variant | Assumption | Use Case |
| :--- | :--- | :--- |
| **GaussianNB** | $P(x_i|y) \sim \mathcal{N}(\mu, \sigma^2)$ | Continuous features |
| **MultinomialNB** | Count-based features | Text classification (bag-of-words) |
| **BernoulliNB** | Binary features | Document classification |

Despite "naive" assumption, it often works well on text and is very fast to train.

---

# 7. Evaluation Metrics

## Classification

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$F_\beta = \frac{(1+\beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

Use $\beta > 1$ when recall matters more (e.g., medical diagnosis); $\beta < 1$ when precision matters more.

**ROC-AUC:** area under the Receiver Operating Characteristic curve. Threshold-agnostic, but optimistic with imbalanced data.

**PR-AUC:** area under the Precision-Recall curve. Better for imbalanced datasets — directly measures quality on the minority class.

## Regression

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

# 8. Common Failure Modes

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
