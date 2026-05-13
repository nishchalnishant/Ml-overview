# Calibration and Uncertainty Quantification

## Executive Summary

Calibration and uncertainty quantification answer a question that accuracy metrics don't: "Can I trust this probability?" A model with 90% accuracy that outputs `P = 0.9` on every example it gets right has learned nothing about uncertainty. Calibration closes the gap between a model's confidence and its actual correctness rate. Uncertainty quantification goes further — distinguishing what the model can't know (aleatoric) from what the model doesn't know yet (epistemic).

| Concept | What it measures | When it matters |
|---------|-----------------|-----------------|
| Calibration | Alignment of predicted probabilities with empirical frequencies | Medical diagnosis, fraud scoring, insurance pricing |
| Reliability diagram | Visual calibration diagnostic | Model evaluation, stakeholder communication |
| ECE / MCE | Scalar calibration error metrics | Comparing models, monitoring in production |
| Aleatoric uncertainty | Irreducible noise in the data | Inform the limit of achievable accuracy |
| Epistemic uncertainty | Model's lack of knowledge | Out-of-distribution detection, active learning |
| Conformal prediction | Distribution-free coverage guarantees | Safety-critical inference |

---

## 1. What Calibration Means

### The Core Definition

A probabilistic classifier is **perfectly calibrated** if:

$$P(Y = 1 \mid \hat{p}(X) = p) = p \quad \forall p \in [0, 1]$$

In plain language: among all predictions where the model outputs probability $p$, the fraction of actual positives equals $p$.

**Concrete example:** A loan default model outputs a default probability for each applicant. If the model is calibrated:
- Among applicants scored at 0.10: approximately 10% actually default
- Among applicants scored at 0.50: approximately 50% actually default
- Among applicants scored at 0.90: approximately 90% actually default

If instead all applicants scored 0.90 default only 60% of the time, the model is **overconfident** — a dangerous miscalibration for credit decisioning.

### Why Models Are Miscalibrated

| Model type | Typical miscalibration direction | Cause |
|-----------|--------------------------------|-------|
| Logistic Regression | Well-calibrated | Log-loss is a proper scoring rule |
| Random Forest | Overconfident near 0 and 1 | Averaging trees pushes probabilities to extremes |
| SVM | Severely miscalibrated | SVM doesn't optimize a probabilistic objective |
| Neural Networks (after training) | Overconfident | Modern nets over-parameterized; softmax probabilities saturate |
| XGBoost | Moderately overconfident | Boosting concentrates probability mass at boundaries |

**Proper scoring rules:** A loss function is a proper scoring rule if it is minimized by the true probability. Log-loss (cross-entropy) is proper. Accuracy is not proper — you can maximize accuracy while being completely miscalibrated.

---

## 2. Reliability Diagrams (Calibration Curves)

### How to Read a Reliability Diagram

A reliability diagram (calibration curve) bins predictions by predicted probability and plots:
- X-axis: mean predicted probability in the bin
- Y-axis: fraction of positives in the bin (empirical frequency)
- Diagonal: perfect calibration line

**Interpretations:**

- Points **above the diagonal**: model is **underconfident** — actual positive rate is higher than the predicted probability
- Points **below the diagonal**: model is **overconfident** — actual positive rate is lower than the predicted probability
- S-shaped curve (above for low $p$, below for high $p$): characteristic of Random Forests and neural networks

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_test, y_scores, n_bins=10, strategy="uniform")

plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Reliability Diagram")
```

**Bin strategy choices:**
- `strategy="uniform"`: bins of equal width in probability space; can have sparse bins at extremes
- `strategy="quantile"`: bins of equal sample count; better for skewed score distributions

**Production use:** Run calibration curves on monthly model validation reports. A drift in calibration — even without accuracy degradation — signals population shift that will cause downstream business decisions to be miscalibrated.

---

## 3. Platt Scaling

### Logistic Regression on Top of Raw Scores

Platt scaling fits a logistic regression to transform raw model outputs (log-odds, SVM decision values, XGBoost raw scores) into calibrated probabilities.

**Procedure:**

1. Train the base model on training set
2. Get raw scores (not probabilities) on a held-out **calibration set** (separate from test set)
3. Fit logistic regression: $P(Y=1) = \sigma(A \cdot f(x) + B)$ where $f(x)$ is the raw score
4. At inference: apply base model, then apply Platt scaling

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Method 1: sklearn wrapper (uses internal calibration set via CV)
svm = SVC(kernel="rbf")
calibrated_svm = CalibratedClassifierCV(svm, method="sigmoid", cv=5)
calibrated_svm.fit(X_train, y_train)
probs = calibrated_svm.predict_proba(X_test)

# Method 2: Manual Platt scaling on a held-out calibration set
svm = SVC(kernel="rbf").fit(X_train, y_train)
raw_scores = svm.decision_function(X_cal)
platt = LogisticRegression().fit(raw_scores.reshape(-1, 1), y_cal)
calibrated_probs = platt.predict_proba(svm.decision_function(X_test).reshape(-1, 1))[:, 1]
```

**When Platt scaling works well:**
- SVM, boosting models, neural networks with non-probabilistic training objectives
- When the miscalibration is roughly sigmoid-shaped (monotone but wrong slope)
- Sufficient calibration data: > 1,000 examples recommended

**Limitations:**
- Assumes the miscalibration is parametrically sigmoidal — fails for complex, non-monotone miscalibration
- Requires a held-out calibration set (not the training or test set); using test data for calibration is data leakage

---

## 4. Isotonic Regression Calibration

### Non-Parametric Monotone Calibration

Isotonic regression fits a non-decreasing step function mapping raw scores to calibrated probabilities. No parametric form assumed.

**Algorithm:** Pool adjacent violators — find the best monotone fit by merging bins that violate monotonicity.

```python
from sklearn.isotonic import IsotonicRegression

# Fit isotonic regression on calibration set
iso_reg = IsotonicRegression(out_of_bounds="clip")
iso_reg.fit(raw_scores_cal, y_cal)

# Apply at inference
calibrated_probs = iso_reg.predict(raw_scores_test)
```

**vs Platt scaling:**

| | Platt Scaling | Isotonic Regression |
|--|--------------|---------------------|
| Flexibility | Parametric (sigmoid) | Non-parametric (step function) |
| Data requirements | < 1,000 examples OK | Needs > 1,000 for stability |
| Overfitting risk | Low | High with small calibration set |
| Best for | SVMs, moderate miscalibration | Highly non-linear miscalibration |

**Rule of thumb:** If calibration set > 5,000 examples, try isotonic regression first. If < 1,000 examples, use Platt scaling. Cross-validate the choice using ECE on a held-out set.

**Combined approach (sklearn):**

```python
calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
# cv=5 means 5-fold: trains base model on 4 folds, calibrates on 5th
# More data-efficient than a separate calibration set when total n is small
```

---

## 5. Temperature Scaling for Neural Networks

### The Modern Calibration Fix for Deep Learning

Temperature scaling is the dominant post-hoc calibration method for neural networks. It learns a single scalar $T$ (temperature) that divides logits before the softmax:

$$\hat{q}_i = \max_k \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$

- $T > 1$: softens the probability distribution (reduces overconfidence)
- $T < 1$: sharpens the distribution (rarely needed; networks are usually overconfident)
- $T = 1$: original softmax, unchanged

**Why it works:** Modern neural networks achieve high accuracy but become overconfident with depth and batch normalization. Temperature scaling cannot change predictions (argmax is unchanged) but corrects probability magnitudes.

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

# Usage
ts = TemperatureScaling().fit(val_logits, val_labels)
calibrated_logits = ts(test_logits)
calibrated_probs = torch.softmax(calibrated_logits, dim=-1)
```

**Key paper:** "On Calibration of Modern Neural Networks" (Guo et al., ICML 2017) demonstrated that deeper networks (ResNet, DenseNet) trained with batch normalization and weight decay are consistently overconfident, and temperature scaling is the simplest fix with minimal accuracy impact.

**Limitations of temperature scaling:**
- Single scalar: can't fix class-specific miscalibration (some classes overconfident, others underconfident)
- Vector scaling (one temperature per class) or matrix scaling address this at cost of more parameters
- Requires a held-out calibration set separate from both training and test

---

## 6. ECE and MCE: Scalar Calibration Metrics

### Expected Calibration Error (ECE)

ECE measures the average gap between predicted probability and empirical accuracy, weighted by bin size:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Where:
- $B_m$: the $m$-th bin of predictions
- $\text{acc}(B_m)$: fraction of correct predictions in bin $m$
- $\text{conf}(B_m)$: mean predicted confidence in bin $m$
- $n$: total number of examples
- $M$: number of bins (typically 10 or 15)

**Maximum Calibration Error (MCE):**

$$\text{MCE} = \max_{m \in \{1,\ldots,M\}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

MCE captures worst-case miscalibration — important in high-stakes decisions where a single badly calibrated probability region can cause systematic harm.

```python
import numpy as np

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
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

**Interpretation benchmarks:**

| ECE value | Interpretation |
|-----------|---------------|
| < 0.01 | Excellent calibration |
| 0.01 – 0.05 | Acceptable for most production use cases |
| 0.05 – 0.10 | Moderate miscalibration; apply post-hoc correction |
| > 0.10 | Severe miscalibration; model probabilities should not be trusted |

**Important caveat:** ECE depends on bin count and strategy. Always report along with the number of bins and strategy (uniform vs quantile). Compare ECE only across models evaluated identically.

---

## 7. Aleatoric vs Epistemic Uncertainty

### Two Fundamentally Different Sources of Uncertainty

**Aleatoric uncertainty** is irreducible — it comes from inherent randomness or noise in the data generating process.
- A patient with borderline biomarker values will have genuinely uncertain outcomes regardless of how much data you collect
- A noisy sensor introduces measurement uncertainty that no model improvement eliminates
- Formally: $\text{Var}[Y \mid X = x]$ — the variance of the outcome given the input

**Epistemic uncertainty** is reducible — it comes from the model not having enough data or the right data to make confident predictions.
- A model has never seen inputs from a particular demographic → high epistemic uncertainty on those inputs
- Collecting more data in that region would reduce epistemic uncertainty
- Formally: uncertainty in the model parameters $\theta$ given finite training data

**Why the distinction matters in production:**

| Uncertainty type | Actionable response |
|-----------------|---------------------|
| Aleatoric | Communicate uncertainty to downstream decision-maker; do not escalate to human review expecting a better answer |
| Epistemic | Flag as out-of-distribution; queue for human review or active learning |

**Mathematical decomposition (Bayesian framework):**

$$\underbrace{\mathbb{E}_\theta[\text{Var}[Y \mid X, \theta]]}_{\text{aleatoric}} + \underbrace{\text{Var}_\theta[\mathbb{E}[Y \mid X, \theta]]}_{\text{epistemic}} = \text{Total uncertainty}$$

In practice: run an ensemble or Monte Carlo Dropout and decompose the variance across forward passes.

---

## 8. Conformal Prediction

### Distribution-Free Coverage Guarantees

Conformal prediction produces prediction sets (not point predictions) with a guaranteed marginal coverage probability — without making distributional assumptions.

**Core guarantee:**

$$P(Y_{n+1} \in C_\alpha(X_{n+1})) \geq 1 - \alpha$$

Where $C_\alpha(X_{n+1})$ is the prediction set and $1 - \alpha$ is the desired coverage (e.g., 90%).

**Split conformal prediction procedure:**

1. Train model on training set
2. Compute nonconformity scores on calibration set: $s_i = 1 - \hat{p}(Y_i \mid X_i)$ (for classification)
3. Compute the $(1-\alpha)(1 + 1/n_{\text{cal}})$ quantile of calibration scores: $\hat{q}$
4. At inference: prediction set $C(x) = \{y : \hat{p}(y \mid x) \geq 1 - \hat{q}\}$

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Calibration step
cal_probs = model.predict_proba(X_cal)
cal_true_class_probs = cal_probs[np.arange(len(y_cal)), y_cal]
nonconformity_scores = 1 - cal_true_class_probs

alpha = 0.1  # target 90% coverage
n_cal = len(nonconformity_scores)
q_hat = np.quantile(nonconformity_scores, np.ceil((1 - alpha) * (n_cal + 1)) / n_cal)

# Inference step
def conformal_prediction_set(probs, q_hat):
    return np.where(probs >= 1 - q_hat)[0].tolist()

test_probs = model.predict_proba(X_test[0:1])[0]
prediction_set = conformal_prediction_set(test_probs, q_hat)
# prediction_set = [2, 4] means classes 2 and 4 are in the 90% coverage set
```

**Key properties:**

- **Marginal coverage guaranteed** regardless of model quality or data distribution (only exchangeability assumed)
- **Adaptive set sizes:** difficult inputs get larger prediction sets; easy inputs get singleton sets
- **No retraining required:** conformal wraps any existing model

**Conformal prediction in production:**

- Medical triage: if prediction set contains multiple diagnoses, escalate to specialist
- Content moderation: if label set is ambiguous, route to human reviewer
- Financial decisions: provide the set of plausible risk levels rather than a point estimate

**Limitation:** Marginal coverage is guaranteed in aggregate; conditional coverage (coverage within subgroups) is not guaranteed without additional assumptions.

---

## 9. Monte Carlo Dropout for Uncertainty

### Dropout as Approximate Bayesian Inference

Dropout during inference (not just training) can be interpreted as approximate Bayesian inference over neural network weights. Multiple stochastic forward passes yield a distribution over predictions.

**Theory (Gal & Ghahramani, 2016):** Training with dropout and weight decay approximates variational inference in a Gaussian Process. MC Dropout samples from the approximate posterior over weights.

**Implementation:**

```python
import torch
import torch.nn as nn

class MCDropoutModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Active at inference time too
        return self.fc2(x)

def mc_dropout_predict(model, x, n_samples=100):
    model.train()  # Enable dropout at inference
    predictions = torch.stack([
        torch.softmax(model(x), dim=-1)
        for _ in range(n_samples)
    ])
    mean_pred = predictions.mean(dim=0)
    uncertainty = predictions.var(dim=0).sum(dim=-1)  # Total variance
    return mean_pred, uncertainty
```

**Decomposing uncertainty with MC Dropout:**

```python
# Epistemic: variance across forward passes (model uncertainty)
epistemic = predictions.var(dim=0)

# Aleatoric: mean predictive entropy (data uncertainty)
p_bar = predictions.mean(dim=0)
aleatoric = -(p_bar * torch.log(p_bar + 1e-8)).sum(dim=-1)

# Total uncertainty: entropy of the mean prediction
total = aleatoric + epistemic.sum(dim=-1)
```

**Practical considerations:**
- $n\_samples = 30$–$50$ is usually sufficient; diminishing returns beyond 100
- MC Dropout adds latency proportional to $n\_samples$; batch predictions to amortize cost
- Calibrate the dropout rate on calibration ECE, not validation accuracy
- Deep ensembles (5 independently trained models) typically outperform MC Dropout for uncertainty quality at cost of 5x inference

---

## 10. When Calibration Matters: High-Stakes Domains

### Medical Diagnosis

A radiologist interpreting a chest X-ray AI tool needs: "This finding has a 73% probability of malignancy." If the model outputs 73% but the true rate at that score is 40%, the radiologist is systematically over-triaging. Calibration errors in medical AI translate directly to patient harm or resource misallocation.

**Requirements:** ECE < 0.03 for clinical deployment. Post-hoc calibration required after domain shift. Monitor calibration on each hospital cohort separately (batch effects create calibration drift).

### Fraud Scoring

A fraud score of 0.95 might trigger an automatic block; 0.70 might trigger a review queue; 0.30 is auto-approved. The scoring thresholds are calibrated to hit specific false positive rates with specific fraud detection rates. If the model's probabilities drift, the operational thresholds (set months ago) no longer hold.

**Requirements:** Monitor ECE weekly. Recalibrate (isotonic regression on recent data) monthly or on significant data drift events.

### Insurance Premium Pricing

Actuarial pricing requires that predicted claim frequencies match actual claim frequencies within each rating cell. A miscalibrated ML model charging the wrong premium is a regulatory violation in most jurisdictions.

**Requirements:** Calibration by segment (age band, geography, vehicle type), not just aggregate. Regulatory filings may require demonstration of calibration to the insurance commissioner.

### Risk Model Capital Allocation

Credit risk models (PD, LGD, EAD under Basel III/IV) must be calibrated. Banks must demonstrate to regulators that a loan assigned PD = 0.5% defaults at approximately that rate over a long-run horizon.

**Requirements:** Long-run average calibration, backtesting over multiple economic cycles, and stress-scenario calibration.

---

## Interview Q&A

**Q1: What does it mean for a model to be calibrated?**

> A model is calibrated if among all predictions with probability $p$, the actual positive rate is also $p$. Logistic regression is natively calibrated because it optimizes log-loss (a proper scoring rule). SVMs and neural networks are typically miscalibrated — the former produces geometric decision values rather than probabilities; the latter produces overconfident probabilities due to over-parameterization and batch normalization.

**Q2: How would you calibrate an SVM?**

> Apply Platt scaling: fit a logistic regression $\sigma(Af(x) + B)$ where $f(x)$ is the SVM decision value. Fit the logistic regression on a held-out calibration set (not training or test data). Verify the result with a reliability diagram and ECE before and after. If the calibration curve is highly non-linear, try isotonic regression instead.

**Q3: What is temperature scaling and when do you apply it?**

> Temperature scaling divides neural network logits by a learned scalar $T$ before softmax. It corrects the common over-confidence of modern deep networks without changing predictions (argmax is preserved). Apply it as a final post-training step: fit $T$ by minimizing NLL on a held-out calibration set using L-BFGS. It's the simplest calibration method with minimal accuracy impact.

**Q4: What is the difference between aleatoric and epistemic uncertainty?**

> Aleatoric uncertainty is irreducible noise in the data itself — collecting more data won't resolve it. Epistemic uncertainty reflects the model's lack of knowledge — it decreases with more data. Operationally: aleatoric uncertainty warrants communicating ambiguity to the decision-maker; epistemic uncertainty on out-of-distribution inputs warrants human escalation or active learning.

**Q5: How does conformal prediction work?**

> Conformal prediction produces prediction sets (not points) with a guaranteed marginal coverage. Given a calibration set, compute nonconformity scores (e.g., $1 - \hat{p}(y_{\text{true}})$ for each example). At a desired $1-\alpha$ coverage level, the threshold $\hat{q}$ is the $(1-\alpha)$-quantile of calibration scores. At inference, include in the prediction set all labels $y$ with score $\hat{p}(y) \geq 1 - \hat{q}$. No distributional assumptions are needed — only exchangeability.

**Q6: What is ECE and how do you compute it?**

> Expected Calibration Error bins predictions by predicted probability and computes the weighted average absolute gap between mean confidence and empirical accuracy across bins. ECE < 0.01 is excellent; > 0.05 indicates actionable miscalibration. Always report ECE with the bin count and strategy (uniform width vs quantile), as ECE values are not directly comparable across different binning schemes.

**Q7: Why does a random forest have an S-shaped calibration curve?**

> Random forests average the predictions of many trees. The averaging process compresses probabilities toward a middle range — extreme probabilities near 0 or 1 require all trees to agree. The result is that truly high-confidence cases get predicted at, say, 0.7 instead of 0.9 (underconfident), and truly low-confidence cases get predicted at 0.3 instead of 0.1 (underconfident). This creates the characteristic S-shape below the diagonal at extremes.

**Q8: How do you detect and respond to calibration drift in production?**

> Monitor ECE on monthly or weekly score vintages against actuals (e.g., fraud labels arrive 30–90 days after the transaction). Plot reliability diagrams per period. When ECE exceeds a threshold (e.g., 0.05), trigger recalibration: fit a new isotonic regression or Platt scaler on recent labeled data from the current distribution. In Azure ML, this can be automated as a pipeline step triggered by the model monitoring alert.

**Q9: What is a proper scoring rule and why does it matter?**

> A proper scoring rule is a loss function minimized by the true probability distribution. Log-loss is proper; accuracy and Brier score are also proper. Training with a proper scoring rule incentivizes the model to output true probabilities rather than to maximize a metric that can be gamed. A model trained with cross-entropy that then outputs probabilities is calibrated in expectation; a model trained with 0-1 loss has no calibration guarantee.

**Q10: How would you handle uncertainty in a medical AI deployment?**

> Layer multiple mechanisms: (1) temperature scaling post-training to correct overconfidence; (2) conformal prediction to produce prediction sets with guaranteed coverage — if the set contains multiple diagnoses, escalate to specialist; (3) MC Dropout or deep ensemble to decompose aleatoric vs epistemic uncertainty — flag high-epistemic predictions as out-of-distribution for human review; (4) monitor ECE on each clinical site separately since image acquisition differences cause calibration drift across hospitals.
