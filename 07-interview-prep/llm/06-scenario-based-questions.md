---
module: Interview Prep
topic: Llm
subtopic: Scenario Based Questions
status: unread
tags: [interviewprep, ml, llm-scenario-based-questions]
---
# Scenario-Based Questions — First-Principles Interview Guide

These questions test whether you think like someone who runs systems, not just trains models. The underlying competency: systematic diagnosis before action, and proportional intervention before heroic solutions.

---

## Core Principle: Diagnose Before Acting

Every scenario answer that jumps straight to "retrain the model" or "increase model complexity" is wrong — not because retraining is always wrong, but because acting before diagnosing is wrong. The correct structure:

```
1. What do the symptoms tell me? (diagnosis)
2. What's the minimal intervention that would fix this if I'm right? (proportionality)
3. How do I validate that my fix worked? (verification)
4. What do I monitor going forward? (prevention)
```

---

## 1. Model Accuracy Dropped Overnight

### What the interviewer is testing
Whether you have a systematic mental model for production degradation — and whether "retrain the model" is your first or last response.

### The reasoning structure

Overnight drops are almost always *external changes*, not model failures. The model is a frozen function — it doesn't change unless you change it. What changed is the world around it.

Check in this order, because earlier steps are cheaper and more common:

```
1. Data pipeline
   ├── Did an upstream schema change? (field renamed, type changed, new null behavior)
   ├── Did a feature job fail silently? (job succeeded, but produced stale/empty output)
   └── Did a new data source introduction change feature distributions?

2. Input distribution
   ├── Plot feature histograms: current vs. 7-day baseline
   ├── PSI on top 20 features — anything > 0.25?
   └── Are null/missing rates suddenly different?

3. Labels
   ├── Did the labeling logic change? (A/B test started, label pipeline updated)
   ├── Are labels delayed? (ground truth takes 24h to arrive — is today's label batch empty?)
   └── Did the ground truth definition drift? (fraud rules changed, annotation guidelines updated)

4. Model and serving
   ├── Was there a deployment or configuration change?
   └── Did a dependency change? (embedding service retrained, feature store schema bumped)

5. Infrastructure
   └── Partial rollout? Cache poisoning? A/B assignment bug?
```

**Rule: do not retrain on bad data.** Retraining on a corrupted or drifted data source makes things worse — the model learns the corrupted distribution. Diagnose first.

### The pattern in action

"Our fraud model's recall dropped from 82% to 61% overnight."

First question: is this real recall on ground-truth labels, or is it the monitored proxy metric? If the labeling pipeline failed, you're seeing 0 positive labels in today's evaluation window — recall is undefined, not 61%.

After confirming it's real: check the feature PSI. If `transaction_velocity_1h` has PSI = 0.7, a new feature computation job likely failed and is returning zeros. Recall drops because a key predictive feature is missing at serving time. Fix: restore the feature computation job. No retraining needed.

### Common traps

**Trap: assuming it's concept drift and retraining immediately.** True concept drift (the underlying fraud pattern changed) is rare and slow. Infrastructure failures, schema changes, and feature bugs are common and sudden. They look identical in the metrics but require opposite responses.

---

## 2. Training Loss Down, Validation Loss Up

### What the interviewer is testing
Whether you can read a loss curve, correctly identify overfitting, and prescribe fixes in order of cost and invasiveness.

### The reasoning structure

Overfitting is definitionally a gap between training and validation loss that grows during training. Two pieces of evidence confirm it:
1. Training loss continues decreasing
2. Validation loss has an inflection point after which it rises

```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring="roc_auc",
    train_sizes=np.linspace(0.1, 1.0, 10)
)
# Plot mean ± std for train and val
# Large persistent gap at max data = high variance → regularize
# Both low = underfitting → more capacity
# Gap closes with more data = will improve with more data
```

### Fixes in order of cost

| Fix | When to apply | Cost |
| :--- | :--- | :--- |
| Early stopping | Always — first thing | Free |
| Reduce regularizer penalty | If L1/L2 lambda is too low | Free |
| Increase dropout | If model is clearly over-parameterized | Low |
| Add L2 / weight decay | Neural nets — stable, always available | Low |
| Data augmentation | Vision (flips, crops, color jitter), text (back-translation) | Medium |
| More training data | If you can get it | Medium to high |
| Reduce model capacity | Last resort — removing layers is irreversible | High |

### Common traps

**Trap: applying multiple fixes simultaneously.** If you add dropout, increase L2, and reduce learning rate at the same time, and the model improves, you don't know which fix worked. Apply fixes one at a time, or at most one category at a time.

**Trap: confusing overfitting with distribution shift.** If val loss is high from the very first epoch — not rising after training, but consistently high — it's not overfitting. It's distributional mismatch between train and val sets (wrong split, temporal leakage, different population).

---

## 3. Both Train and Val Loss Are Poor

### What the interviewer is testing
Whether you can distinguish underfitting from "model capacity isn't the problem."

### The reasoning structure

When both training and validation loss are high, the model isn't learning. But the cause might not be model capacity — it might be:

**Check first:**
- **Noisy labels:** if 20% of labels are wrong, no model can achieve < 20% error — it looks like underfitting but more capacity makes it worse
- **Wrong loss function:** MSE on a classification problem produces slow convergence and instability
- **Bad feature engineering:** the features don't contain the information needed to predict the target
- **Learning rate too high or too low:** learning rate too high → loss oscillates; too low → essentially doesn't move

**After ruling out the above:**
- Increase model capacity (more layers, more hidden units)
- Reduce regularization strength
- Engineer domain-specific features that the model doesn't have to discover from scratch

---

## 4. Model Works in Testing, Fails in Production

### What the interviewer is testing
Whether you understand training-serving skew — the most common source of production-offline metric divergence.

### The reasoning structure

The model is a fixed function. If it worked in testing and fails in production, the *inputs* must be different between the two environments. Map every transformation the input goes through in testing vs production.

| Source of skew | Description | Detection | Fix |
| :--- | :--- | :--- | :--- |
| Feature leakage | Test set includes future information | Check data timestamps relative to split | Use proper temporal splits |
| Preprocessing skew | Scaler fit on full data, not just training data | Inspect where `fit()` was called | Use sklearn Pipeline; fit on train only |
| Distribution shift | Production data is from a different distribution than test | Compare feature histograms: prod vs test | Collect production samples; re-evaluate |
| Threshold mismatch | Threshold tuned on val, applied to prod with different class balance | Compare positive rate: val vs prod | Calibrate threshold on a held-out production sample |
| Serving code divergence | Preprocessing in the serving stack differs from training | End-to-end feature comparison | Export preprocessing as part of the model artifact (ONNX, TorchScript) |

**The most common one — temporal leakage:**
```python
# Wrong: random split for time-series data
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
# A random 20% of future rows land in training → model sees future data

# Right: temporal split
split_date = df["timestamp"].quantile(0.8)
train = df[df["timestamp"] < split_date]
test  = df[df["timestamp"] >= split_date]
```

**The serving code divergence pattern:** training uses `scaler.fit_transform(X_train)` in a notebook. Serving uses a separately written feature computation function that handles edge cases differently. Even a subtle difference (NaN handling, integer vs float cast) breaks the model silently. The fix: export the preprocessing pipeline as part of the model artifact and use the same code at both training and serving time.

---

## 5. 99% Accuracy but the Model Is Useless

### What the interviewer is testing
Whether you immediately recognize the imbalanced data trap and know which metrics to use instead.

### The reasoning structure

99% accuracy on a 99.5% negative-class problem means the model predicts all-negative. The "model" is the empty function. This is not a bug — it's a metric problem.

```python
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

print(classification_report(y_true, y_pred))
# Look at recall for the positive class — if it's 0.00, you have a trivial classifier

print("ROC-AUC:", roc_auc_score(y_true, y_proba))
# ROC-AUC = 0.5 → random (your model is random)
# But ROC-AUC is optimistic: TN is huge, so FPR is small → AUC looks good

print("PR-AUC:", average_precision_score(y_true, y_proba))
# PR-AUC does not involve TN — much more informative for rare classes
# A random classifier achieves PR-AUC ≈ class prevalence (~0.005 for 0.5% positive)
```

**Imbalance fixes in order of invasiveness:**
1. `class_weight="balanced"` — free, adjusts the loss to weight minority examples more
2. Threshold tuning — move threshold from 0.5 to match the positive rate
3. SMOTE — generate synthetic minority examples (tabular data only)
4. Undersampling — reduce majority class size
5. Focal loss — downweights easy (correctly classified) examples during training

---

## 6. Fast 90% Model vs Slow 92% Model

### What the interviewer is testing
Whether you quantify tradeoffs rather than defaulting to "higher accuracy is always better."

### The reasoning structure

2% absolute accuracy improvement means different things in different contexts. The decision requires business, user, and cost analysis:

**Step 1 — Quantify the business value of 2%:**
If the model drives $10M annual revenue: a 2% improvement = ~$200K/year. Is that meaningful relative to infrastructure cost?

**Step 2 — Quantify the latency cost:**
Every additional 100ms of latency causes user abandonment. Industry benchmarks: 100ms = ~1% drop in conversion rate. If P99 latency goes from 100ms to 300ms: potentially 2% fewer completions. That might wipe out the accuracy gain.

**Step 3 — Quantify ongoing infrastructure cost:**
A 2× larger model = 2× GPU memory = 2× hosting cost. Is $200K/year gain worth an extra $150K/year in serving cost?

**Step 4 — A/B test both:**
The only ground truth is the actual business metric on actual users. An offline accuracy gain doesn't guarantee an online metric gain.

**Strong answer:** "I'd run a shadow evaluation of the slower model on recent production traffic to quantify the 2% improvement per user segment. Then I'd A/B test both models at 10% traffic each, measuring P99 latency, conversion rate, and revenue/query. The decision criterion is net present value: accuracy gain × revenue per percentage point minus serving cost delta. I'd also check whether the 2% improvement is uniform or concentrated — if it's entirely from a high-value user segment, the case for the slower model is stronger."

---

## 7. Stakeholder Wants to Launch Tomorrow, You Have Concerns

### What the interviewer is testing
Whether you can manage risk proportionally rather than either blocking the launch or capitulating entirely.

### The reasoning structure

"I have concerns" must be quantified to be actionable. "We haven't tested on mobile users" is different from "We haven't tested on mobile users, who are 30% of our traffic."

**Alternatives to full block or full launch:**

1. **Quantify the risk:** "Mobile users are 30% of traffic and we haven't validated on that segment. Worst case: 30% of users see degraded recommendations for 24h."

2. **Canary launch:** start at 1% → 5% → 20% with metrics checks at each stage:
```python
import hashlib

def should_use_new_model(user_id: str, rollout_pct: float = 0.01) -> bool:
    """Deterministic: same user always gets the same bucket."""
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return (hash_val % 100) < (rollout_pct * 100)
```

3. **Shadow mode:** new model runs but results are logged, not served. Validate offline before any user sees it.

4. **Define a rollback trigger:** "If CTR drops > 5% relative to control in the first 24 hours, auto-rollback." Defining this in advance is not pessimistic — it's what makes fast launch safe.

### Common traps

**Trap: refusing without proposing an alternative.** "We're not ready" is not a complete answer. The complete answer is "here's what we need to do to be ready by tomorrow, and here's how we can launch with a safety net even if we can't do all of it."

---

## 8. Limited Labeling Budget

### What the interviewer is testing
Whether you know active learning strategies — and can reason about which sampling strategy is appropriate for which situation.

### The reasoning structure

Random sampling is wasteful: you spend labeling budget on examples the model already handles confidently. Strategic sampling concentrates the budget on examples that will most improve the model.

| Strategy | Mechanism | When to use |
| :--- | :--- | :--- |
| Uncertainty sampling | Label examples where model is most uncertain (entropy ≈ max) | Any classification task |
| Core-set / diversity | Label examples that maximize coverage of input space | When labeled data is highly clustered |
| Semi-supervised | Train on labeled + pseudo-labeled unlabeled data | When unlabeled data is abundant |
| Weak supervision (Snorkel) | Combine multiple noisy labeling functions | When domain rules exist but no ground truth |
| QBC (Query by Committee) | Label examples where an ensemble disagrees most | When you can train multiple models |

```python
import numpy as np

def uncertainty_sampling(model, X_unlabeled, n=100):
    """Select the n most uncertain samples by predictive entropy."""
    proba = model.predict_proba(X_unlabeled)
    # Entropy = -sum(p * log(p)): maximum when uniform, zero when certain
    entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
    return np.argsort(entropy)[-n:]  # indices of n highest-entropy samples

def core_set_selection(X_labeled, X_unlabeled, n=100):
    """Select n points from unlabeled that are farthest from any labeled point."""
    selected = []
    for _ in range(n):
        dists = np.min([np.linalg.norm(X_unlabeled - x, axis=1) for x in X_labeled + selected], axis=0)
        idx = np.argmax(dists)
        selected.append(X_unlabeled[idx])
    return selected
```

---

## 9. Explainability Required for a Black-Box Model

### What the interviewer is testing
Whether you know the difference between local and global explanation methods — and the critical caveat that model explanations explain model behavior, not causal relationships.

### The reasoning structure

Start by clarifying *what* needs to be explained and *to whom*:
- **Regulatory compliance:** SHAP values per feature on individual predictions (local)
- **Model debugging:** global feature importance, partial dependence plots
- **User-facing explanation:** simplified rule (LIME) or top-3 features

| Method | Scope | Speed | Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- |
| SHAP | Global + local | Fast for trees, slow for DNN | Exact for trees | Shapley values: average marginal contribution |
| LIME | Local only | Fast | Approximate (local linear model) | Can be unstable — perturb same instance twice, get different explanation |
| Permutation importance | Global | Fast | Approximate | Affected by correlated features |
| Partial dependence plots | Global | Medium | Average effect | Assumes independence |

```python
import shap

explainer = shap.TreeExplainer(model)    # exact Shapley values for tree models
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)    # global: feature importances + direction
shap.waterfall_plot(shap_values[0])       # local: single prediction explanation
shap.dependence_plot("age", shap_values.values, X_test)  # feature interaction
```

**The critical caveat:** SHAP and LIME explain the *model's behavior*, not the *data-generating process*. If the model uses ZIP code as a proxy for race (which it might learn from historical data), SHAP will correctly report that ZIP code is predictive — but this says nothing about whether ZIP code *causes* the outcome, or whether using it is appropriate. Model explanations do not validate model fairness.

---

## 10. Model Is Biased Against a Demographic Group

### What the interviewer is testing
Whether you can diagnose the source of bias (data, proxy features, feedback loop) and know the range of mitigations, including their tradeoffs.

### The reasoning structure

First, characterize the bias precisely — different types require different fixes:

**Step 1 — Diagnose by group:**
```python
from sklearn.metrics import classification_report

for group in df["demographic"].unique():
    mask = df["demographic"] == group
    print(f"\n--- {group} ---")
    print(classification_report(y_true[mask], y_pred[mask]))
```

**Step 2 — Identify source:**
- **Historical bias in labels:** training labels encode historical discrimination (e.g., loan default rates reflect historical lending discrimination, not just creditworthiness)
- **Underrepresentation:** minority group is 2% of training data — model doesn't generalize to them
- **Proxy features:** ZIP code → neighborhood → race. The feature encodes a protected attribute indirectly
- **Feedback loops:** model deprioritizes a group → less interaction data from them → worse model for them → further deprioritization

**Fairness metrics and their tradeoffs:**

| Metric | Definition | Tradeoff |
| :--- | :--- | :--- |
| Demographic parity | Equal positive prediction rate across groups | Ignores differences in base rates |
| Equalized odds | Equal TPR and FPR across groups | May require different thresholds per group |
| Calibration | $P(Y=1 | \text{score}=s)$ equal across groups | May be incompatible with equalized odds at different base rates |

**Mitigations:**
- **Pre-processing:** reweight training samples, oversample underrepresented group
- **In-processing:** adversarial debiasing — add an adversary that tries to predict the protected attribute from the model's representations; penalize if it succeeds
- **Post-processing:** calibrate different thresholds per group to equalize FPR (requires knowing the group at serving time — may be legally constrained)

---

## 11. Data Drift in Production

### What the interviewer is testing
Whether you know the three types of drift and have concrete detection methods for each.

### The three types

| Type | What changes | Example | Detection |
| :--- | :--- | :--- | :--- |
| Feature drift | $P(X)$ shifts | User age distribution changes after app targets younger users | KS test or PSI on each feature |
| Concept drift | $P(Y|X)$ shifts | Fraud patterns change after new attack vector emerges | Monitor accuracy on a recent labeled slice |
| Label drift | $P(Y)$ shifts | CTR drops seasonally | Monitor score distribution, positive rate |

**Detection code:**
```python
from scipy.stats import ks_2samp
import numpy as np

def detect_feature_drift(baseline, current, alpha=0.05):
    stat, p_value = ks_2samp(baseline, current)
    return {"drifted": p_value < alpha, "ks_stat": round(stat, 4), "p_value": round(p_value, 4)}

def psi(expected_pcts, actual_pcts, eps=1e-4):
    """PSI > 0.1 = monitor; > 0.25 = retrain."""
    return sum(
        (a - e) * np.log((a + eps) / (e + eps))
        for a, e in zip(actual_pcts, expected_pcts)
    )
```

**Responses by severity:**
- PSI 0.1–0.25: increase monitoring frequency, alert on-call
- PSI > 0.25: trigger retraining, evaluate whether to switch to fallback model
- Concept drift (accuracy on labeled slice drops): retrain immediately with fresh data

**The concept drift challenge:** concept drift is harder to detect because it requires labeled recent data. For many systems, labels arrive with delay (fraud chargebacks take 30–60 days). Before labels arrive, you can detect *proxy signals*: model score distribution shifts, feature-label correlation changes on a limited labeled sample.

---

## 12. Designing a Real-Time Fraud Scoring System

### What the interviewer is testing
Whether you can design a production ML system from first principles, with all the components that real fraud teams have learned the hard way.

### The design, and why each decision was made

```
Transaction event
        │
        ├── [Redis] Velocity feature lookup
        │          (why Redis: sub-millisecond sorted set lookups for rolling window counts)
        │          - txn_count_1h, txn_count_24h, unique_merchants_1h
        │          - amount_sum_1h, distinct_countries_7d
        │
        ├── [Rule engine] Hard blocks before ML model
        │   (why first: rules are microseconds; model is milliseconds; known-bad should be caught cheaply)
        │   - Known fraudulent card/account
        │   - Impossible geography (txn in NYC, then Tokyo, 5 minutes later)
        │   - Velocity hard limits (> 20 txns in 1 minute → always block)
        │
        ├── [ML model] Gradient boosted tree (LightGBM)
        │   (why GBT not DNN: < 5ms latency requirement; tabular features; SHAP interpretability for chargebacks)
        │   - Input: velocity features + static user/merchant features + transaction features
        │   - Output: fraud probability in [0, 1]
        │
        └── [Decision engine] Cost-aware thresholds
              ├── score < 0.2  → Auto-approve
              ├── 0.2–0.6      → 3DS challenge (friction)
              └── score > 0.6  → Auto-block
```

**Threshold selection:**
$$\text{Expected cost}(\tau) = FP(\tau) \times C_{FP} + FN(\tau) \times C_{FN}$$

Choose $\tau^* = \arg\min_\tau \text{Expected cost}(\tau)$. Typical values: $C_{FP}$ (false block) ≈ \$10–50 (support cost + churn risk); $C_{FN}$ (missed fraud) ≈ \$100–500 (chargeback + penalty). Default threshold 0.5 is almost never optimal.

**The propensity score problem for retraining:** blocked transactions never get ground truth labels. If you retrain only on approved transactions, the model has never seen the near-misses it blocked — it can't learn from them. Solution: periodically approve a small random sample of medium-risk transactions (with monitoring) to collect counterfactual labels. Weight these examples by $1/P(\text{approved})$ during retraining.

**Monitoring targets:**
- Model score distribution (daily PSI)
- FP and FN rates on labeled slice (as chargebacks arrive 30–60 days later)
- Rule engine block rate (sudden spike = possible data quality issue)
- Feature null rates (velocity features returning null = Redis failure)
