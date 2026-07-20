---
module: Evaluation
topic: Calibration And Uncertainty
subtopic: ""
status: unread
tags: [classicalml, ml, calibration-and-uncertainty]
---
# Calibration and Uncertainty Quantification

---

## The Problem Calibration Solves

**The problem**: A credit model outputs `P(default) = 0.85` for an applicant. The underwriter approves or rejects based on this number. If the model is correct, about 85 out of every 100 applicants scored at 0.85 will default. But if the model is systematically overconfident, applicants scored at 0.85 might only default 40% of the time — and every downstream decision based on that score is wrong in a predictable, systematic direction.

Accuracy metrics don't catch this. A model can be highly accurate while being completely miscalibrated — its predicted probabilities don't reflect reality, even when its hard classifications (above/below threshold) are often correct.

**The core insight**: Calibration asks a different question than accuracy. Not "was the prediction correct?" but "when the model says 70%, does the event occur 70% of the time?" A well-calibrated model's confidence is trustworthy as a probability.

**The mechanics**: A classifier is perfectly calibrated if:

$$P(Y = 1 \mid \hat{p}(X) = p) = p \quad \forall p \in [0, 1]$$

Among all predictions where the model outputs probability $p$, the fraction of actual positives should be $p$.

**What breaks immediately**: Most models are not calibrated by default. SVMs don't optimize a probabilistic objective at all — their outputs are not probabilities. Neural networks are trained with cross-entropy (which is a proper scoring rule) but modern deep networks with batch normalization and weight decay are consistently *overconfident* — the optimization landscape encourages probabilities near 0 or 1. Random forests average tree votes, compressing probabilities toward the middle but with non-uniform bias.

| Model | Typical miscalibration | Cause |
|---|---|---|
| Logistic Regression | Well-calibrated | Optimizes log-loss, a proper scoring rule |
| Random Forest | S-shaped curve | Vote averaging compresses probabilities away from extremes |
| SVM | Severely miscalibrated | Optimizes margin, not likelihood |
| XGBoost | Moderately overconfident | Boosting concentrates mass at class boundaries |
| Deep Networks | Overconfident | Over-parameterization, batch normalization, weight decay |

---

## Reliability Diagrams

**The problem**: You need to see whether a model is calibrated without reducing everything to a single number — calibration can be wrong in different ways in different probability ranges.

**The core insight**: Bin predictions by predicted probability. Within each bin, compare the mean predicted probability to the actual fraction of positives. If the model is calibrated, these should be equal — points should fall on the diagonal.

**The mechanics**: Sort predictions into M equal-width or equal-frequency bins. For each bin, compute mean predicted probability and fraction of true positives. Plot them.

- Points above the diagonal: model is underconfident — the true positive rate exceeds the predicted probability. The model is more right than it thinks.
- Points below the diagonal: model is overconfident — the true positive rate is lower than the predicted probability. The model is more wrong than it thinks.
- S-shaped curve (above for low $p$, below for high $p$): classic random forest pattern.

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_test, y_scores, n_bins=10, strategy='uniform')

plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Reliability Diagram')
```

Two bin strategies:
- `strategy='uniform'`: equal-width bins in probability space. Can have sparse bins at extremes.
- `strategy='quantile'`: equal-count bins. Better for skewed score distributions.

**What breaks**: Reliability diagrams are sensitive to bin count and strategy — the same model can look differently calibrated with 5 vs 20 bins. Always report the binning parameters. Sparse bins at extremes produce noisy, unreliable points — treat them skeptically.

---

## ECE and MCE: Scalar Calibration Metrics

**The problem**: You want a single number to compare calibration between models, track calibration over time in production, and set alert thresholds.

**The core insight**: Average the absolute gap between predicted probability and empirical accuracy across bins, weighted by bin population.

**The mechanics**:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

MCE (Maximum Calibration Error) takes the worst bin rather than the weighted average — captures worst-case miscalibration, which matters in high-stakes decisions.

$$\text{MCE} = \max_{m} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

```python
import numpy as np

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(acc - conf)
    return ece

def compute_mce(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    gaps = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        gaps.append(abs(y_true[mask].mean() - y_prob[mask].mean()))
    return max(gaps) if gaps else 0.0
```

| ECE value | Interpretation |
|---|---|
| < 0.01 | Excellent |
| 0.01–0.05 | Acceptable for most production use |
| 0.05–0.10 | Moderate; apply post-hoc calibration |
| > 0.10 | Severe; don't use raw probabilities for downstream decisions |

**What breaks**: ECE depends on bin count. A model with ECE = 0.04 using 10 bins and ECE = 0.06 using 20 bins is the same model — only compare ECE across models with identical binning settings.

---

## Platt Scaling

**The problem**: An SVM or XGBoost model outputs scores (decision values or raw boosting scores) that are not probabilities. You need to convert them to probabilities that can be trusted as calibrated.

**The core insight**: Fit a logistic regression on top of the raw scores to map them to [0, 1] in a way that aligns with empirical positive frequencies. The logistic regression acts as a parametric calibration layer.

**The mechanics**: Hold out a calibration set (separate from both training and test data). Get raw model scores on the calibration set. Fit logistic regression $P(Y=1) = \sigma(A \cdot f(x) + B)$ where $f(x)$ is the raw score. Parameters $A$ and $B$ are learned from the calibration data.

```python
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Method 1: sklearn wrapper
svm = SVC(kernel='rbf')
calibrated_svm = CalibratedClassifierCV(svm, method='sigmoid', cv=5)
calibrated_svm.fit(X_train, y_train)
probs = calibrated_svm.predict_proba(X_test)

# Method 2: manual Platt scaling on a held-out calibration set
svm = SVC(kernel='rbf').fit(X_train, y_train)
raw_scores = svm.decision_function(X_cal)
platt = LogisticRegression().fit(raw_scores.reshape(-1, 1), y_cal)
calibrated_probs = platt.predict_proba(svm.decision_function(X_test).reshape(-1, 1))[:, 1]
```

**What breaks**: Platt scaling assumes the miscalibration is monotone and roughly sigmoid-shaped — raw scores are linearly related to true log-odds after the logistic transform. If the calibration curve is non-monotone or has a complex shape, the parametric fit will be wrong. Requires a separate calibration set — if you use the test set for calibration, that is data leakage.

---

## Isotonic Regression Calibration

**The problem**: Platt scaling imposes a specific functional form (sigmoid) on the calibration mapping. When the true mapping is non-linear or the model has a complex, non-monotone miscalibration pattern, the parametric assumption is violated.

**The core insight**: Fit a non-parametric, non-decreasing step function from raw scores to probabilities. No assumed functional form — just the constraint that higher scores should map to higher probabilities (monotonicity).

**The mechanics**: Pool Adjacent Violators algorithm finds the best monotone fit: group adjacent points that violate monotonicity and set them all to their pooled mean. The result is a piecewise-constant non-decreasing function.

```python
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(raw_scores_cal, y_cal)
calibrated_probs = iso.predict(raw_scores_test)

# Or combined with CV to avoid wasting a separate calibration set:
calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
calibrated.fit(X_train, y_train)
```

| | Platt Scaling | Isotonic Regression |
|---|---|---|
| Flexibility | Parametric sigmoid | Non-parametric step function |
| Data requirement | Works with < 1k examples | Needs > 1k for stability |
| Overfitting risk | Low | High with small calibration set |
| Best for | Sigmoid-shaped miscalibration | Complex, non-linear miscalibration |

**What breaks**: With small calibration sets (< 500 examples), isotonic regression overfits to the calibration data — it memorizes the empirical frequencies of individual score bins rather than learning a smooth mapping. Use Platt scaling for small calibration sets.

---

## Temperature Scaling

**The problem**: Neural networks trained with modern techniques (batch normalization, weight decay, deep architectures) are systematically overconfident. They produce high-confidence softmax probabilities even on examples near the decision boundary. This is one of the most consistent findings in deep learning calibration (Guo et al., ICML 2017).

**The core insight**: Divide all logits by a single scalar $T$ before the softmax. This flattens the probability distribution — reduces overconfidence — without changing any predictions (argmax is unchanged). The scalar $T$ is fit by minimizing NLL on a held-out calibration set.

**The mechanics**:
- $T > 1$: softer distribution, less confident
- $T < 1$: sharper distribution, more confident (rarely needed)
- $T = 1$: unchanged

$$\hat{q}_i = \max_k \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$

```python
import torch
import torch.nn as nn

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits_cal, labels_cal):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        nll = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            loss = nll(logits_cal / self.temperature, labels_cal)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return self

ts = TemperatureScaling().fit(val_logits, val_labels)
calibrated_probs = torch.softmax(ts(test_logits), dim=-1)
```

**What breaks**: A single scalar cannot fix class-specific miscalibration — if class A is overconfident and class B is underconfident, one temperature makes one worse while fixing the other. Vector scaling (one $T$ per class) or matrix scaling (full linear transform of logits) address this at the cost of more parameters. All variants require a held-out calibration set.

---

## Aleatoric vs Epistemic Uncertainty

**The problem**: A model outputs `P(positive) = 0.6`. That uncertainty could mean two very different things. The model might be uncertain because this input type is genuinely ambiguous — no amount of additional data will make it clearer. Or the model might be uncertain because it has never seen this type of input before — with more data from this region, it would become confident.

**The core insight**: These two sources of uncertainty demand different responses. Irreducible uncertainty (aleatoric) should be communicated to the downstream decision-maker. Reducible uncertainty (epistemic) should trigger human review, active learning, or a flag that the input is out-of-distribution.

**The mechanics**:
- **Aleatoric uncertainty**: Variance of the label given the input, $\text{Var}[Y | X = x]$. Comes from noise in the data generating process. Cannot be reduced by collecting more data.
- **Epistemic uncertainty**: Uncertainty about the model parameters given finite training data. Decreases with more data in the uncertain region.

Bayesian decomposition:
$$\underbrace{\mathbb{E}_\theta[\text{Var}[Y | X, \theta]]}_{\text{aleatoric}} + \underbrace{\text{Var}_\theta[\mathbb{E}[Y | X, \theta]]}_{\text{epistemic}} = \text{Total variance}$$

In practice: run multiple stochastic forward passes (MC Dropout or deep ensemble). Aleatoric uncertainty is the average entropy of individual predictions. Epistemic uncertainty is the variance of predictions across forward passes.

---

## Monte Carlo Dropout

**The problem**: You have a single deterministic neural network. You want an estimate of how uncertain the network is about each prediction, without retraining an ensemble.

**The core insight**: Gal & Ghahramani (2016) showed that dropout during inference — keeping the same dropout masks active as during training — approximates sampling from the posterior over network weights. Multiple forward passes with different dropout masks produce a distribution over predictions.

**The mechanics**: Set the model to `train()` mode at inference time to keep dropout active. Run N forward passes. Compute statistics over the predictions.

```python
import torch

class MCDropoutModel(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.fc1     = torch.nn.Linear(input_dim, 256)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc2     = torch.nn.Linear(256, n_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)      # active during inference when model.train() is called
        return self.fc2(x)

def mc_dropout_predict(model, x, n_samples=50):
    model.train()   # enables dropout
    predictions = torch.stack([
        torch.softmax(model(x), dim=-1) for _ in range(n_samples)
    ])
    mean_pred  = predictions.mean(dim=0)
    epistemic  = predictions.var(dim=0)
    aleatoric  = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=-1)
    return mean_pred, epistemic, aleatoric
```

**What breaks**: N=50 forward passes multiplies inference latency by 50 — batch your predictions. MC Dropout underestimates epistemic uncertainty compared to deep ensembles — it samples from a restricted variational posterior, not the full Bayesian posterior. Deep ensembles (5 independently trained models) consistently outperform MC Dropout for uncertainty quality but cost 5x training and inference.

---

## Conformal Prediction (Overview)

**The problem**: You want a coverage guarantee — not just a well-calibrated probability, but a provable statement: "The true label is in this prediction set with at least 90% probability." No distributional assumptions. Works with any model.

See the dedicated `conformal-prediction.md` file for full treatment. Summary of the key ideas here:

**The core insight**: A calibration set of labeled examples tells you how nonconforming (surprising) a prediction needs to be before you should distrust it. If a new prediction is more nonconforming than $(1-\alpha)$ of the calibration examples, exclude it from the prediction set.

**The guarantee**: $P(Y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha$, guaranteed under exchangeability alone.

```python
import numpy as np

# Calibration: compute how "wrong" the model is on each calibration example
cal_probs = model.predict_proba(X_cal)
nonconformity_scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]

alpha = 0.1   # want 90% coverage
n     = len(nonconformity_scores)
q_hat = np.quantile(nonconformity_scores, np.ceil((1 - alpha) * (n + 1)) / n)

# Inference: include all classes that are not too nonconforming
def predict_set(probs, q_hat):
    return np.where(1 - probs <= q_hat)[0].tolist()
```

**What breaks**: The guarantee is *marginal* — averaged over all test points. Conditional coverage (within a subgroup) is not guaranteed. Under distribution shift, exchangeability breaks and so does the coverage guarantee.

---

## When Calibration Is Non-Negotiable

**Medical diagnosis**: A radiologist using an AI tool to triage chest X-rays needs `P(malignancy) = 0.73` to mean 73% of similar cases are malignant — not 40%. Calibration errors translate directly to over- or under-triage decisions.

**Fraud scoring**: Thresholds for auto-block, manual review, and auto-approve are set based on score distributions. If calibration drifts, the thresholds set at deployment no longer correspond to the intended false positive rates.

**Insurance pricing**: Predicted claim frequencies must match actual frequencies within each rating cell. Miscalibration is both a financial and a regulatory risk.

**Credit risk under Basel III/IV**: Probability of Default (PD) models must be calibrated — banks demonstrate to regulators that loans assigned PD = 0.5% default at approximately that rate over a long-run horizon.

In all these cases: monitor ECE on rolling windows of scored-then-resolved cases. When ECE exceeds a threshold, trigger recalibration (fit new isotonic regression or Platt scaler on recent labeled data). A reliability diagram per monthly cohort is the minimum acceptable monitoring.

---

## Canonical Interview Q&As

**Q: What is calibration and how do you measure and fix it in a classifier?**
A: A calibrated classifier outputs probabilities that match empirical frequencies: if the model says p=0.7 for 1000 samples, ~700 of them should actually be positive. Miscalibration types: (1) overconfident — predicted probabilities cluster near 0 and 1, but accuracy at those confidence levels is lower; (2) underconfident — probabilities cluster near 0.5 regardless of true uncertainty. Measurement: reliability diagram (plot mean predicted probability vs empirical accuracy in buckets) and Expected Calibration Error (ECE) = Σ_b (|b|/n) · |acc(b) - conf(b)|. ECE < 0.05 is generally acceptable. Root causes: neural networks are overconfident due to training with hard cross-entropy targets; tree ensembles are often poorly calibrated because leaf values come from small sample counts. Fixes: Platt scaling — fit a logistic regression on a validation set mapping raw scores to probabilities; temperature scaling — a single parameter T: p_calibrated = softmax(logits/T); T > 1 softens the distribution (reduces overconfidence). Temperature scaling is preferred for neural networks because it has one parameter and doesn't overfit.

**Q: What is the difference between aleatoric and epistemic uncertainty, and how do you quantify each?**
A: **Aleatoric uncertainty** (data uncertainty): inherent noise in the data that cannot be reduced even with infinite data — e.g., a coin flip, label noise, sensor measurement error. Quantified by the model's output distribution (entropy of the predicted distribution). **Epistemic uncertainty** (model uncertainty): uncertainty from not having seen enough data — the model could be wrong because it lacks evidence for this region of input space. Reduces with more data. Quantified by variance across multiple model predictions. Methods: (1) Deep Ensembles — train N independent models; predict with all N; variance across predictions estimates epistemic uncertainty; (2) MC Dropout — keep dropout active at test time; run N forward passes; variance estimates uncertainty; (3) Conformal prediction — gives guaranteed coverage without distributional assumptions. Practical application: high epistemic uncertainty → sample needs human review (the model hasn't seen similar inputs); high aleatoric uncertainty → the label is inherently ambiguous (nothing you can do). In production, flag inputs where epistemic uncertainty > threshold for human-in-the-loop.

**Q: A model has 0.92 ROC-AUC but your product team is unhappy with its predictions. What calibration and threshold issues might explain this?**
A: Several issues: (1) **Miscalibration** — ROC-AUC measures ranking quality (whether positive scores higher than negative), not probability accuracy. A model can rank perfectly but output p=0.99 for cases that are only 60% likely to be positive. If the team is making threshold-based decisions (flag if p > 0.5), miscalibrated probabilities lead to unexpected false positive rates. Fix: check reliability diagram and apply temperature scaling. (2) **Wrong threshold** — the default 0.5 threshold is arbitrary. If the cost of a false negative is 10× a false positive (e.g., fraud), the optimal threshold should be much lower than 0.5. Use the cost-sensitive threshold: argmin_t [FPR(t)·C_FP + FNR(t)·C_FN]. (3) **Class imbalance** — with 1% positive rate, even a calibrated p=0.5 prediction is very rare; the team may be seeing many low-confidence positives that they expect to be high. Fix: recalibrate with the true base rate. (4) **Distribution shift** — model was calibrated on training distribution; production inputs may differ (feature drift → probabilities shift). Check PSI on input features and recalibrate periodically.

For active-recall drilling on these terms, see [classical-ml-flashcards.md](../03-classical-ml/_flashcards.md).
