---
module: Classical Ml
topic: Ensemble Methods
subtopic: ""
status: unread
tags: [classicalml, ml, ensemble-methods]
---
# Ensemble Methods

---

## Why Ensembles Work

**The problem**: any single model makes a particular kind of mistake — it has blind spots determined by its inductive biases, its training data, and the randomness in its initialization. There is no free lunch: a model that is strong in one region of input space tends to be weaker elsewhere.

**The core insight**: if the mistakes of multiple models are *uncorrelated*, combining their predictions causes errors to cancel. When model A is wrong, models B and C are right, and the average pushes toward the truth. The requirement is independence, not individual accuracy — a committee of diverse mediocre models often beats a single brilliant one.

Bias-Variance decomposition frames this precisely: Expected error = Bias² + Variance + Irreducible noise.
- **Bagging** reduces variance (averages out noise across models)
- **Boosting** reduces bias (focuses on hard examples the current ensemble gets wrong)
- **Stacking** can reduce both by learning how to combine models optimally

---

## Bagging (Bootstrap Aggregating)

**The problem**: a single decision tree trained to high accuracy is brittle — it memorizes training data and collapses on new inputs (overfitting). The source of this instability is high variance: small changes in training data produce completely different trees.

**The core insight**: variance drops when you average independent estimates. Bootstrap sampling creates artificial independence — each tree sees a different random subset of the training data (sampled with replacement), so their errors are partially decorrelated. Average the predictions, and errors cancel.

**The mechanics**: train $B$ models on bootstrap samples. Combine predictions by averaging (regression) or majority vote (classification).

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

**Random Forest = Bagging + random feature subsets at each split**: bootstrap sampling alone is not enough — trees trained on correlated features still make correlated errors. Randomly restricting which features are available at each split forces different trees to focus on different signals, breaking residual correlation.

**Out-of-bag (OOB) score**: each bootstrap sample contains ~63% of training points. The remaining ~37% ("out-of-bag" samples) were never seen during training for that tree — they serve as a free validation set.

```python
rf = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
print(f"OOB score: {rf.oob_score_:.4f}")  # unbiased estimate without separate val set
```

**What breaks**: when features are strongly correlated, random subsets often pick the same dominant feature — trees remain correlated and the variance reduction is limited. Bootstrap underrepresents rare events (minority classes, rare patterns), so the ensemble inherits this blind spot. Random forests also cannot extrapolate — all predictions are bounded by the range of values seen during training.

---

## Boosting

**The problem**: bagging parallelizes many weak models and averages them, but averaging cancels errors — it does not fix systematic biases. If every tree makes the same mistake on a certain kind of input (because the signal is complex and shallow trees can't capture it), averaging changes nothing.

**The core insight**: instead of building models independently, build them sequentially — each new model explicitly corrects the mistakes of the current ensemble. The models become experts at the cases the ensemble currently fails on.

### AdaBoost

**The problem**: boosting needs a way to "tell" each new model which examples to focus on, without changing the training algorithm itself.

**The core insight**: re-weight the training examples. After each round, upweight misclassified examples so the next model is forced to focus on them. Mis-classified examples get higher weight; correctly classified ones get lower weight.

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

Model weight: `α_m = ½ ln((1-ε_m)/ε_m)` where `ε_m` is the weighted error. Better models get higher vote in the final ensemble.

**What breaks**: AdaBoost is sensitive to noisy labels. Mislabeled examples get upweighted on every round because they can never be correctly classified — the algorithm obsessively focuses on them, eventually fitting pure noise.

---

### Gradient Boosting

**The problem**: AdaBoost re-weights examples, which only works for classification. You want a boosting framework that works for any loss function — regression, ranking, custom objectives.

**The core insight**: "correcting mistakes" is just fitting the residual error. The gradient of the loss with respect to the current predictions tells you exactly how each prediction needs to change. Fit the next tree to those gradients (pseudo-residuals). This is gradient descent in function space — each tree is a step in the direction that most decreases the loss.

**The mechanics**: each new tree $h_m$ is fit to $-\nabla_{F} L$ (negative gradient of loss):

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

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

**What breaks**: gradient boosting is prone to overfitting — it keeps correcting residuals until it memorizes training noise. Early stopping (via a validation set) is essential. The sequential nature makes it slow and unparallelizable across trees. It is also sensitive to hyperparameters: learning rate, tree depth, and subsampling fraction all interact.

---

## Voting Ensemble

**The problem**: you have several strong, diverse models — a logistic regression that is great on linear patterns, a random forest that handles non-linearities, and an XGBoost tuned on your dataset. None of them dominates. How do you combine them?

**The core insight**: each model has different strengths and blind spots. When they agree, you can be confident. When they disagree, averaging their probability estimates smooths over individual errors. No retraining needed — just aggregate.

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
- `voting='soft'`: average probabilities — requires calibrated models but is almost always better

**What breaks**: combining two identical models gains nothing — diversity is the source of the gain. Soft voting requires calibrated probabilities; uncalibrated models (e.g., raw tree outputs) produce unreliable averaged probabilities.

---

## Stacking (Stacked Generalization)

**The problem**: voting assigns fixed weights to each model, but the optimal combination might be context-dependent — logistic regression might be better in certain regions of input space, while XGBoost dominates elsewhere. Fixed weights cannot capture this.

**The core insight**: learn the combination weights from data. Train a meta-learner that takes the base models' predictions as inputs and learns how to best combine them. The meta-learner sees each base model's track record and learns which to trust.

**The mechanics**:

```
Level 0 (base learners): Train diverse models (LR, RF, XGB, SVM, NN)
Level 1 (meta-learner): Train on out-of-fold predictions of Level 0
```

The critical implementation detail: base model predictions for the meta-learner must come from examples the base model was *not* trained on — otherwise the meta-learner sees inflated base-model confidence and overfits.

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

**Meta-learner choice**: logistic regression is standard — it is simple, low variance, and interpretable. A complex meta-learner overfits the OOF predictions.

**Multi-level stacking**: Level 0 → Level 1 → Level 2. Diminishing returns after two levels; rarely worth going past two.

**What breaks**: if base model predictions for the meta-learner come from examples those models were trained on (no OOF), the meta-learner sees artificially perfect predictions and learns nonsense. The whole benefit of stacking comes from honest estimates of out-of-sample performance.

---

## Blending

**The problem**: stacking with cross-validation is expensive — you train each base model $k$ times (once per fold). For very slow base models, this is prohibitive.

**The core insight**: hold out a single fixed validation set instead of running cross-validation. Use base model predictions on this held-out set as meta-features, and train the meta-learner on them.

```
Train base models on X_train
Predict on X_val → meta_features_val
Train meta-learner on meta_features_val, y_val
```

**Difference from stacking**: blending uses a single held-out set (less data efficient, higher variance in meta-features). Stacking uses cross-validation (more robust). Use stacking in practice; blending only when base models are too slow to train $k$ times.

---

## Snapshot Ensembles

**The problem**: training $N$ independent models for an ensemble requires $N\times$ training compute — expensive for deep networks.

**The core insight**: a single training run traverses many different loss landscape regions. If you use a cyclic learning rate schedule, the model repeatedly converges to different local minima. Save checkpoints at each convergence point and ensemble them — $N$ diverse models for the cost of one training run.

**The mechanics**: use cosine annealing with warm restarts (SGD with warm restarts). Save model checkpoints when the learning rate reaches its minimum. Ensemble these snapshots at inference.

**What breaks**: snapshots from the same training run are more correlated than independently trained models — they share the same architecture, same data, and the same initialization trajectory. The diversity benefit is real but smaller than training from scratch.

---

## Multi-Seed Ensemble

**The problem**: a single training run of the same model architecture produces one set of weights — one particular solution to the optimization problem. The variance from random initialization means this specific solution might have idiosyncratic weaknesses.

**The core insight**: run the same training procedure multiple times with different random seeds. Each run finds a different local optimum with different strengths. Average the predictions. The gain is purely from sampling different solutions to the same problem.

**What breaks**: diminishing returns quickly — most of the benefit comes from 2–5 seeds. The models are highly correlated (same architecture, same data), so you are averaging similar models, not diverse ones. This primarily reduces variance from initialization noise, not from model bias.

---

## Calibration Before Ensembling

**The problem**: soft voting and stacking combine probability outputs from multiple models. If those probabilities are not calibrated — if a model that outputs 0.9 is actually right only 60% of the time — averaging them produces meaningless numbers.

**The core insight**: before combining models, calibrate each one so its output probability matches empirical frequency. Then the combined probability means something.

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_rf = CalibratedClassifierCV(rf, cv='prefit', method='isotonic')
calibrated_rf.fit(X_cal, y_cal)   # fit calibrator on held-out calibration set
```

**What breaks**: calibration on the same training data used to train the model overfits — the calibrator sees the model's most confident predictions on data it memorized. Always calibrate on a held-out set.

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

---

## Canonical Interview Q&As

**Q: Explain the bias-variance trade-off for bagging vs boosting and when each is preferred.**  
A: Bagging (Random Forest): trains many high-variance, low-bias models (deep trees) in parallel on bootstrap samples, then averages. Averaging reduces variance without increasing bias — the ensemble has the bias of a single tree but lower variance. Best when: individual models overfit (high variance), you have enough data for independent samples, and you need fast parallel training. Boosting (Gradient Boosting): trains shallow trees sequentially, each one correcting the residuals of the previous — primarily reduces bias. Each tree is a low-variance, high-bias model; the ensemble achieves low bias by combining many biased models additively. Best when: underfitting is the primary issue, data has complex interactions that individual trees miss, and you can afford sequential training time. In practice for tabular data: GBT (XGBoost/LightGBM) usually outperforms Random Forest because tabular data typically has complex feature interactions that benefit from bias reduction more than variance reduction. Exception: when data is noisy or small (< 1K samples), Random Forest is more robust.

**Q: What is gradient boosting mathematically — what exactly is each tree fitting?**  
A: At step m, we have a model F_m(x). For a loss L(y, F(x)), the pseudo-residuals are r_im = -[∂L(y_i, F(x_i))/∂F(x_i)]. Each new tree h_m is fit to these pseudo-residuals (which are the negative gradient of the loss). For MSE loss, ∂L/∂F = F(x) - y, so residuals are literally the prediction errors. For log-loss, residuals are y - p̂ (actual minus predicted probability). The tree is then added: F_{m+1}(x) = F_m(x) + ν·h_m(x), where ν is the learning rate (shrinkage). This is gradient descent in function space — each tree is a "step" in the direction that most reduces loss, parameterized in the space of tree functions. The tree depth controls what interactions can be captured (depth=1 trees are stumps that only capture main effects; depth=3-6 captures up to 3-way interactions). LightGBM adds leaf-wise growth (grow the leaf with highest gain at each step) vs level-wise (XGBoost default), making it faster but requiring max_depth to avoid overfitting.

**Q: How do you tune a XGBoost/LightGBM model for a competition vs for production?**  
A: **Competition** (optimize validation metric): start with learning_rate=0.05, n_estimators=2000, early stopping on validation; then tune depth (3-8), subsample/colsample_bytree (0.6-0.9), min_child_weight/min_data_in_leaf; use Optuna for Bayesian optimization; try multiple seeds and average predictions. Time is not constrained. **Production** (optimize inference latency + monitoring + robustness): restrict tree depth (≤5), number of trees (≤200), to meet latency SLA; disable features that will drift (time-sensitive features without monitoring); use monotonicity constraints on features where the direction is known (e.g., higher credit score → higher approval probability must hold); add feature importance monitoring to detect when top features shift in production; prefer fewer, interpretable trees over many shallow ones for audit purposes. Key difference: competition models chase last 0.1% of accuracy; production models trade accuracy for reliability, interpretability, and operational stability.

For active-recall drilling on these terms, see [classical-ml-flashcards.md](classical-ml-flashcards.md).
