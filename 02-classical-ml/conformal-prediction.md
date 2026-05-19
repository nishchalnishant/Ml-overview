# Conformal Prediction

---

## The Problem Conformal Prediction Solves

**The problem**: A model predicts class 3 with probability 0.72. What does that actually guarantee? Nothing provable. The model was trained to minimize cross-entropy, and its softmax outputs are calibrated to some degree — but no distributional assumption was made, and there is no mathematical statement bounding the probability that the true label is class 3.

In safety-critical settings — medical triage, autonomous systems, legal decision support — "approximately 72% likely" is insufficient. You need a statement like: "This prediction set contains the true label with at least 90% probability, and that guarantee holds regardless of the data distribution, model architecture, or training procedure."

**The core insight**: Instead of asking "what is the probability that y = 3?" ask "which labels are *not surprising* given this input, according to a calibration set of labeled examples?" If the model's score for a label on a new input is more extreme than 90% of the scores it assigned to the correct labels on the calibration set, include that label in the prediction set. The calibration set empirically defines "surprising" — and by construction, the coverage guarantee follows.

**The guarantee**: For exchangeable data:

$$P(Y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha$$

This is a *frequentist* guarantee over random draws of the data — not a Bayesian credible interval. It requires no distributional assumptions beyond exchangeability. It holds for any black-box model.

---

## Split Conformal Prediction (Inductive CP)

**The problem**: The ideal conformal procedure would retrain the model on every possible label assignment for every test point — computationally intractable. You need a practical approximation.

**The core insight**: Use a fixed calibration set. Train the model once on the training set. Measure how "nonconforming" (surprising) each calibration example is under the trained model. At test time, include in the prediction set all labels whose nonconformity score is no worse than the $(1-\alpha)$ quantile of the calibration scores.

**The mechanics**:

1. Split data into: training set, calibration set, and test set.
2. Train model on the training set.
3. For each calibration example $(x_i, y_i)$, compute the nonconformity score: how poorly the model's output fits the true label.
4. Compute the $(1-\alpha)(1 + 1/n_{cal})$ quantile of calibration scores: $\hat{q}$.
5. At inference on $x_{new}$: include label $y$ in the prediction set if its nonconformity score is $\leq \hat{q}$.

```python
import numpy as np

# Step 3: nonconformity scores on calibration set
# Simple classification score: 1 - P(true class | x)
cal_probs  = model.predict_proba(X_cal)
cal_scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]

# Step 4: threshold at (1 - alpha) coverage level
alpha  = 0.1   # target 90% coverage
n      = len(cal_scores)
q_level = np.ceil((n + 1) * (1 - alpha)) / n
q_hat  = np.quantile(cal_scores, q_level, method='higher')

# Step 5: prediction set at inference
def predict_set(x):
    probs  = model.predict_proba(x.reshape(1, -1))[0]
    scores = 1 - probs
    return np.where(scores <= q_hat)[0].tolist()
```

**Coverage guarantee**: At least 90% of test examples will have their true label in the prediction set. This is marginal over the test distribution — not per-example.

**What breaks**: The calibration set is consumed for calibration and not available for training. With limited data, this means either smaller training sets or smaller calibration sets. The guarantee also requires that the calibration set and test set are exchangeable — drawn from the same distribution. Under distribution shift, the guarantee breaks.

---

## Nonconformity Scores

**The problem**: The simple score $1 - P(y | x)$ works, but it produces prediction sets of uniform size. Easy examples (high model confidence) get small sets, but not as small as they could be. Hard examples get large sets, but not as adaptively large as they should be.

**The core insight**: A good nonconformity score captures how surprising the true label is, in a way that naturally produces small sets for easy inputs and large sets for hard inputs. More informative scores produce smaller, more informative prediction sets at the same coverage level.

| Task | Score | Notes |
|---|---|---|
| Classification (basic) | $1 - P(y \mid x)$ | Simple; works with any softmax model |
| Classification (adaptive) | Adaptive Prediction Sets (APS) | Cumulative sorted probability; adaptive set sizes |
| Regression (basic) | $|y - \hat{y}|$ | Absolute residual; constant-width intervals |
| Regression (adaptive) | $|y - \hat{y}| / \hat{\sigma}(x)$ | Normalized by local uncertainty estimate |

### Adaptive Prediction Sets (APS)

**The problem**: Threshold-based classification CP gives the same-width sets to all examples at a given confidence level. APS adapts: examples where the true label requires many probability mass units to reach get large sets; examples where it's in the top of the distribution get small sets.

**The mechanics**: For a given example and true class $y$, sort all classes by predicted probability (descending). Accumulate probability until you reach $y$. The APS score is the cumulative probability mass up to and including $y$.

```python
def aps_score(probs, y_true):
    sorted_idx = np.argsort(probs)[::-1]
    cumsum = 0.0
    for cls in sorted_idx:
        cumsum += probs[cls]
        if cls == y_true:
            return cumsum   # how much probability mass was needed to include the true class
```

Higher APS score = the true class was further down the ranked list = more surprising = harder example.

**What breaks**: APS scores are not symmetric — the quantile threshold found on the calibration set is in probability-mass units, not a direct probability. Prediction sets formed this way are always internally ranked by probability, which is desirable but can include unintuitive class combinations.

---

## Conformal Regression

**The problem**: You want prediction intervals for regression with a guaranteed coverage probability, without making assumptions about the noise distribution.

**The core insight**: Compute residuals on the calibration set. The calibration residual distribution tells you how wrong the model typically is. A test example gets an interval of ±$\hat{q}$ where $\hat{q}$ is the $(1-\alpha)$ quantile of calibration residuals.

```python
# Calibration residuals
residuals = np.abs(y_cal - model.predict(X_cal))
n = len(residuals)
q_hat = np.quantile(residuals, np.ceil((n + 1) * (1 - alpha)) / n, method='higher')

# Prediction interval at test point
y_pred = model.predict(x_test)
interval = (y_pred - q_hat, y_pred + q_hat)
```

**What breaks**: This produces constant-width intervals — the same ±$\hat{q}$ everywhere. A high-confidence prediction and a low-confidence prediction get the same interval width. This is wasteful and uninformative.

---

## Conformalized Quantile Regression (CQR)

**The problem**: Constant-width intervals from basic conformal regression are too wide for easy inputs and potentially too narrow for hard inputs. You want intervals that adapt to local uncertainty.

**The core insight**: Train a quantile regression model to predict the $\alpha/2$ and $1-\alpha/2$ quantiles of the target distribution. These are model-predicted bounds. Conformal calibration then corrects for any systematic underestimation or overestimation of those bounds, with a provable coverage guarantee on the corrected intervals.

**The mechanics**:
1. Train a quantile regression model predicting $[\hat{q}_{\alpha/2}(x), \hat{q}_{1-\alpha/2}(x)]$.
2. On the calibration set, compute the score: $s_i = \max(\hat{q}_{\alpha/2}(x_i) - y_i,\; y_i - \hat{q}_{1-\alpha/2}(x_i))$. This measures how much the true label exceeds the predicted interval on either side.
3. Compute $\hat{q} = $ the $(1-\alpha)$ quantile of $\{s_i\}$.
4. At test time: interval = $[\hat{q}_{\alpha/2}(x) - \hat{q},\; \hat{q}_{1-\alpha/2}(x) + \hat{q}]$.

The resulting intervals adapt to local difficulty — wide where the quantile model predicts wide uncertainty, narrow where it predicts tight uncertainty — with coverage guaranteed by the conformal adjustment.

**What breaks**: CQR depends on the quality of the quantile regression model. If the quantile model is poorly fit, the adaptive widths are wrong and the conformal correction must compensate heavily, producing near-constant-width intervals again.

---

## Cross-Conformal / CV+

**The problem**: Split conformal wastes calibration data on calibration. With small datasets, giving up 20–30% to calibration meaningfully reduces training set size.

**The core insight**: Use K-fold cross-fitting. Train K models, each leaving out one fold for calibration. Use each model to score its held-out fold. The combined scores from all folds form the calibration set, without any data being explicitly "wasted."

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
residuals = np.zeros(len(X))
for train_idx, cal_idx in kf.split(X):
    model.fit(X[train_idx], y[train_idx])
    residuals[cal_idx] = np.abs(y[cal_idx] - model.predict(X[cal_idx]))

q_hat = np.quantile(residuals, np.ceil((len(X) + 1) * (1 - alpha)) / len(X))
```

**What breaks**: The coverage guarantee for CV+ is approximate (not exact). The K models used for calibration score computation are not identical to the final model trained on all data — there is a train/score mismatch. In practice the approximation is tight for K ≥ 5.

---

## Conditional Coverage

**The problem**: The marginal guarantee says 90% of *all* test examples will have correct coverage. But within a specific subgroup — say, elderly patients or low-income applicants — coverage might be only 70%. The guarantee doesn't protect subgroups.

**The core insight**: True conditional coverage $P(Y \in C(X) | X = x) \geq 1-\alpha$ requires a different calibration for each input. This is much harder to guarantee without distributional assumptions.

**Approaches**:
- **Mondrian conformal prediction**: Stratify by group, compute a separate $\hat{q}$ for each group from group-specific calibration examples.
- **Locally weighted CP**: Weight calibration examples by their similarity to the test point, giving more influence to nearby calibration examples.
- **RAPS (Regularized Adaptive Prediction Sets)**: Adds a regularization term to the APS score that penalizes large prediction sets, improving set efficiency without sacrificing coverage.

---

## Full Conformal vs Split Conformal

| | Full Conformal | Split Conformal |
|---|---|---|
| Coverage | Exact | Marginal |
| Computation | Retrain model for every test point × every candidate label | Train once |
| Practical | No — O(n × n_classes) retraining | Yes |
| Data efficiency | All data used for training and calibration | 20–30% used only for calibration |

Always use split conformal in practice. Full conformal is the theoretical foundation.

---

## Distribution Shift

**The problem**: Exchangeability requires that calibration data and test data come from the same distribution. Under covariate shift — when $P(X)$ changes between calibration and deployment — the calibration quantile is wrong and coverage breaks.

**The core insight**: Reweight calibration examples by their importance under the test distribution. Examples that look more like test inputs get higher weight in the quantile computation.

**The mechanics** (weighted conformal prediction):
$$\hat{q} = \text{weighted quantile of } \{s_i\}, \quad w_i \propto \frac{p_{\text{test}}(x_i)}{p_{\text{cal}}(x_i)}$$

Importance weights $w_i$ can be estimated by training a classifier to distinguish calibration from test examples, or using density ratio estimation.

**What breaks**: Importance weights require either labeled test data (at calibration time) or a density ratio estimator, which introduces additional approximation error. With severe distribution shift, importance weights become very large or very small, making the weighted quantile unstable.

---

## Practical Use with MAPIE

```python
from mapie.classification import MapieClassifier
from mapie.regression    import MapieRegressor

# Classification
mapie_clf = MapieClassifier(estimator=model, method='score', cv='prefit')
mapie_clf.fit(X_cal, y_cal)
y_pred, y_pred_sets = mapie_clf.predict(X_test, alpha=0.1)
# y_pred_sets: shape (n_test, n_classes) bool array — True = in prediction set

# Regression (CQR)
from mapie.regression import MapieRegressor
mapie_reg = MapieRegressor(estimator=quantile_model, method='quantile', cv='prefit')
mapie_reg.fit(X_cal, y_cal)
y_pred, y_pred_intervals = mapie_reg.predict(X_test, alpha=0.1)
```

---

## When to Use Conformal Prediction

**Use it when**:
- You need a provable coverage guarantee (not just a well-calibrated probability).
- The deployment context is safety-critical — medical, legal, financial — and a point prediction is insufficient.
- The model is ambiguous and you want to communicate that ambiguity as a set rather than hiding it behind a point estimate. A prediction set {cat, dog} is more honest than "cat (confidence 0.55)."
- Regulatory compliance requires demonstrable coverage properties.

**It does not replace**:
- Calibration (conformal prediction says nothing about whether the probabilities are calibrated).
- Uncertainty decomposition (does not separate aleatoric from epistemic uncertainty).
- Bayesian inference (provides frequentist guarantees, not Bayesian posteriors — different conceptual framework).

**Calibration set size**: Rule of thumb: at least 1000 calibration examples for $\alpha = 0.05$ to get a stable quantile estimate. For smaller calibration sets, the quantile estimate is noisier and the effective coverage may deviate from the nominal level.
