# Scenario-Based Questions — Interview Reference

Diagnostic and decision-making scenarios. These test whether you think like someone who runs systems, not just trains models.

---

## 1. Model Accuracy Dropped Overnight

**Systematic diagnosis — check in this order:**

```
1. Data pipeline
   ├── Did upstream data source change schema?
   ├── Did a feature engineering job fail silently?
   └── Did training-serving skew appear?

2. Input distribution
   ├── Plot feature histograms: current vs. 7-day baseline
   ├── Check PSI on top 10 features
   └── Look for missing value rates changing

3. Label quality
   ├── Did the labeling logic change?
   ├── Are labels delayed / did a batch job fail?
   └── Did ground truth definition drift?

4. Model
   ├── Was a deployment or config change made?
   └── Did a dependency (embedding service, feature store) change?

5. Infrastructure
   └── Partial rollout, cache issue, A/B assignment bug?
```

**Rule:** do not retrain first. Diagnose first. A retraining on bad data makes things worse.

---

## 2. Training Loss Down, Validation Loss Up (Overfitting)

**Confirmation:** plot loss curves. Overfitting = training loss continues falling, val loss rises after an inflection point.

**Fixes in order of cost:**

| Fix | When to use | Cost |
| :--- | :--- | :--- |
| Early stopping | Always | Free |
| Dropout | Neural nets | Low |
| L1/L2 regularization | All models | Low |
| Data augmentation | Vision, text | Medium |
| More training data | When data is available | Medium |
| Simpler architecture | When model is clearly too large | High |

```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring="roc_auc", train_sizes=np.linspace(0.1, 1.0, 10)
)
# Large gap at full data = overfitting. Both low = underfitting.
```

---

## 3. Both Train and Val Loss Are Poor (Underfitting)

**Causes:** model too simple, too much regularization, bad features, wrong loss function.

**Fixes:** increase model capacity, reduce regularization, engineer domain-specific features, check label quality (noisy labels look like underfitting).

---

## 4. Model Works in Test, Fails in Production

**Train-serve skew checklist:**

| Issue | Description | Fix |
| :--- | :--- | :--- |
| Feature leakage | Test set includes future info | Use proper time-based splits |
| Preprocessing skew | Scaler fit on full data | Fit transforms on train only |
| Distribution shift | Prod data differs from test | Collect prod samples, retest |
| Threshold mismatch | Threshold tuned on val, not prod | Re-calibrate on prod distribution |
| Serving code diff | Different preprocessing in serving | Use same pipeline (sklearn Pipeline, ONNX) |

```python
# Wrong: random split allows future leakage on time series
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# Right: temporal split
split_date = df["timestamp"].quantile(0.8)
train = df[df["timestamp"] < split_date]
test = df[df["timestamp"] >= split_date]
```

---

## 5. 99% Accuracy but Model Is Useless

**Classic imbalance trap:** 99% negative class → predict-all-negative = 99% accuracy.

```python
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

print(classification_report(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, y_proba))
print("PR-AUC:", average_precision_score(y_true, y_proba))  # more honest for imbalanced
```

**Imbalance fixes:** `class_weight="balanced"`, SMOTE oversampling, threshold tuning, focal loss (deep learning).

---

## 6. Fast 90% Model vs Slow 92% Model

**Decision framework:**

1. Quantify business value of 2%: if model drives $10M revenue → ~$200k/year
2. Quantify latency cost: P99 latency increase → user abandon rate → lost conversion
3. Quantify infra cost: larger model → ongoing GPU cost
4. A/B test both: measure actual business metric, not accuracy

**Answer pattern:** "I'd run an A/B test. If latency increase causes user abandonment, the accuracy gain may net negative. I'd also check if the 2% improvement is uniform or concentrated in high-value segments."

---

## 7. Stakeholder Wants to Launch Tomorrow, You Have Concerns

**Don't refuse — propose safer alternatives:**

1. Quantify the risk: "We haven't validated on segment X which is 15% of users"
2. Canary launch: 1% traffic, monitor for 24h before full rollout
3. Shadow mode: new model runs but doesn't serve results; compare offline
4. Define rollback trigger: "If conversion drops 5% in 24h, auto-rollback"

```python
import hashlib

def should_use_new_model(user_id: str, rollout_pct: float = 0.01) -> bool:
    """Deterministic canary: same user always gets same bucket."""
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return (hash_val % 100) < (rollout_pct * 100)
```

---

## 8. Limited Labeling Budget

| Strategy | Mechanism | Best for |
| :--- | :--- | :--- |
| Uncertainty sampling | Label most uncertain samples | Any classification |
| Core-set selection | Label most diverse samples | Low-diversity pool |
| Semi-supervised | Pseudo-label unlabeled data | Abundant unlabeled data |
| Weak supervision (Snorkel) | Combine imperfect labeling rules | When domain rules exist |

```python
import numpy as np

def uncertainty_sampling(model, X_unlabeled: np.ndarray, n: int = 100) -> np.ndarray:
    proba = model.predict_proba(X_unlabeled)
    entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
    return np.argsort(entropy)[-n:]  # indices of n most uncertain samples
```

---

## 9. Explainability Required for a Black-Box Model

| Method | Scope | Notes |
| :--- | :--- | :--- |
| SHAP | Global + local | Shapley values: average marginal feature contribution |
| LIME | Local | Interpretable surrogate around one prediction |
| Permutation importance | Global | Accuracy drop when feature is shuffled |
| Partial dependence plots | Global | Marginal effect of one feature |

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)      # global importance
shap.waterfall_plot(shap_values[0])         # single prediction
```

**Caveat:** SHAP/LIME explain the model's behavior, not the data-generating process. Not causal.

---

## 10. Model Is Biased Against a Demographic Group

**Diagnose by group:**

```python
from sklearn.metrics import classification_report

for group in df["demographic"].unique():
    mask = df["demographic"] == group
    print(f"\n--- {group} ---")
    print(classification_report(y_true[mask], y_pred[mask]))
```

**Sources:** historical bias in labels, underrepresentation, proxy features (ZIP code → race), feedback loops.

**Fairness metrics:**

| Metric | Definition |
| :--- | :--- |
| Demographic parity | Equal positive prediction rate across groups |
| Equalized odds | Equal TPR + FPR across groups |
| Calibration | $P(Y=1 \mid \text{score}=s)$ equal across groups |

**Mitigations:** reweight training samples, adversarial debiasing, post-processing threshold per group.

---

## 11. Data Drift in Production

**Types:**

| Type | What changes | Example |
| :--- | :--- | :--- |
| Feature drift | $P(X)$ shifts | Device type distribution changes |
| Concept drift | $P(Y \mid X)$ shifts | Economic shock changes fraud patterns |
| Label drift | $P(Y)$ shifts | Seasonal change in click-through rates |

**Detection:**

```python
from scipy.stats import ks_2samp
import numpy as np

def detect_drift(baseline: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> dict:
    stat, p_value = ks_2samp(baseline, current)
    return {"drifted": p_value < alpha, "ks_stat": stat, "p_value": p_value}

def psi(baseline_pcts, current_pcts, eps=1e-4):
    """PSI > 0.1 = some drift; > 0.25 = significant drift."""
    return sum(
        (a - e) * np.log((a + eps) / (e + eps))
        for a, e in zip(current_pcts, baseline_pcts)
    )
```

**Responses:** trigger retraining, switch to fallback model, alert on-call, increase monitoring frequency.

---

## 12. Real-Time Fraud System Design

**Key components:**

```
Transaction event
        │
        ├── Velocity features (Redis: txns in last 1h/24h)
        ├── Rule engine (hard blocks: known bad IPs)
        ├── ML model (gradient boosted tree, <10ms latency)
        └── Decision: approve / challenge / block
```

**Threshold setting:**
$$\text{Expected cost} = FP \times \text{cost}_{FP} + FN \times \text{cost}_{FN}$$

Use cost matrix to pick threshold that minimizes expected cost, not the default 0.5.

**Feedback loop:** labeling fraud after the fact feeds back into retraining. Use propensity scores to correct for survivorship bias (blocked transactions never get labels).

> [!TIP]
> **Interview structure:** scenario questions reward systematic thinking over quick answers. Always (1) diagnose before acting, (2) quantify risk before deciding, (3) propose the least disruptive intervention that still solves the problem, (4) mention monitoring and rollback. "Retrain the model" should never be the first response to a production problem.
