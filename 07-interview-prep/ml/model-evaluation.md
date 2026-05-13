# Model Evaluation

This file exists to save you from the most common ML interview disaster:

> "The model has 98% accuracy."

Lovely.
Now tell me whether that number means anything.

---

# 1. Evaluation = Quality Gates for Models

If training is your **build pipeline**, evaluation is your **release gate**.

A model is not "good" because it trained successfully.
A model is good if it survives the right checks:

- correct metric
- correct split
- correct threshold
- correct business context

If any one of those is wrong, the model can look brilliant and still be useless.

---

# 2. Accuracy, Precision, Recall, F1

**Confusion matrix:**

|  | Predicted Positive | Predicted Negative |
| :--- | :--- | :--- |
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

**Formulas:**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}$$
$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
$$F_\beta = \frac{(1+\beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R} \quad \text{($\beta > 1$ = recall preferred; $\beta < 1$ = precision preferred)}$$

```python
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

print(classification_report(y_true, y_pred))
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
```

**When to use:**

| Scenario | Metric |
| :--- | :--- |
| False positives expensive (spam filter) | Precision |
| False negatives expensive (cancer screening) | Recall |
| Both matter | F1 or $F_\beta$ |
| Class imbalance | PR-AUC, not accuracy |

## Accuracy

Great when classes are balanced. Dangerous when they are not.

If fraud is only 1% of transactions, a model predicting "not fraud" every time gets **99% accuracy** — technically impressive, practically useless.

---

# 3. Confusion Matrix

The confusion matrix is the scoreboard behind the metric.

It tells you: TP, FP, TN, FN — and two models can have the same accuracy with very different failure patterns.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
```

In production, **failure pattern** matters more than accuracy bragging rights.

---

# 4. ROC-AUC vs PR-AUC

## ROC-AUC

**ROC curve:** TPR (recall) vs FPR ($= FP/(FP+TN)$) across all decision thresholds.

**AUC interpretation:** probability that the model ranks a random positive higher than a random negative.

$$\text{TPR} = \frac{TP}{TP+FN} \qquad \text{FPR} = \frac{FP}{FP+TN}$$

Useful when classes are reasonably balanced. **Insensitive to class imbalance** because both axes are normalized by class counts.

## PR-AUC (Average Precision)

**PR curve:** Precision vs Recall across thresholds.

$$\text{AP} = \sum_n (R_n - R_{n-1}) P_n$$

More informative when the positive class is rare — ROC-AUC can be misleadingly high on imbalanced datasets while PR-AUC stays honest.

```python
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt

roc_auc = roc_auc_score(y_true, y_proba)
pr_auc = average_precision_score(y_true, y_proba)

fpr, tpr, _ = roc_curve(y_true, y_proba)
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
```

**Rule:** if positives are rare and important, use PR-AUC. If you need a threshold-independent rank quality measure on balanced data, use ROC-AUC.

---

# 5. Log Loss vs Accuracy

**Log loss (binary cross-entropy):**
$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \left[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \right]$$

Accuracy only cares if the final class is correct. Log loss cares about **how confident** the model was.

- mildly wrong = moderate penalty
- confidently wrong (predicted 0.99, actual 0) = severe penalty (approaches ∞)

Use when:
- ranking or sorting by probability score
- calibrated probabilities matter (fraud scoring, ads, medical)
- comparing model quality beyond threshold decisions

**Brier score** (for calibration):
$$\text{BS} = \frac{1}{n}\sum_{i=1}^n (\hat{p}_i - y_i)^2$$

---

# 6. Regression Metrics: MAE, MSE, RMSE, R²

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i| \quad \text{(robust to outliers, interpretable units)}$$
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2 \quad \text{(penalizes large errors heavily)}$$
$$\text{RMSE} = \sqrt{\text{MSE}} \quad \text{(same units as target, penalizes large errors)}$$
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} \quad \text{(fraction of variance explained; 1 = perfect, 0 = baseline mean)}$$

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
```

**When to use:** MAE when outliers are present and you want robust reporting; RMSE when large errors are especially costly; R² for communicating explained variance to stakeholders.

**Fashion analogy**

If your outfit sizing prediction is off by 1 inch, okay.
If it is off by 7 inches, chaos.

RMSE and MSE are the metrics that say:

> "Big misses deserve extra consequences."

---

# 7. Choosing the Right Metric

This is where interview answers become senior-level.

Do not just say:

> "I would use F1."

Say:

> "I would choose the metric based on business cost and operational behavior."

Ask:

- Are false positives costly?
- Are false negatives costly?
- Do we need calibrated probabilities?
- Is this a ranking problem?
- Is class imbalance severe?

**Short rule**

- balanced classification = accuracy can be okay
- rare positive class = precision/recall/PR-AUC
- probability quality matters = log loss / Brier / calibration
- regression with big-error pain = RMSE
- regression with robust interpretation = MAE

---

# 8. Calibration

Calibration means:

If the model says "80% probability," then about 80 out of 100 such cases should really be positive.

This matters a lot in:

- fraud
- medical systems
- pricing
- recommendations
- ads

A model can rank well and still be badly calibrated.

That means:

- good ordering
- bad probabilities

Both are not the same thing.

---

# 9. Cross-Validation

Cross-validation is how you reduce dependence on one lucky split.

Instead of trusting one train/validation split, you rotate across folds.

That gives you a more stable estimate.

**Azure/DevOps parallel**

It is the difference between:

- validating one deployment path once

and

- validating across multiple environments and conditions

Would you trust only one deployment test?
Exactly.

---

# 10. Class Imbalance

Imbalanced data is where bad evaluation habits go to thrive.

Common fixes:

- use better metrics
- tune threshold
- class weighting
- resampling
- focal loss
- better features

The key idea:

Do not "solve" imbalance by only changing the training data.
Often the biggest fix is choosing the right metric and threshold first.

---

# 11. Offline vs Online Evaluation

Offline metrics are necessary.
They are not enough.

Why?

Because users are messy.

A model can improve offline and still hurt:

- CTR
- retention
- revenue
- user trust

So in production you often need:

- offline evaluation
- shadow testing
- canary rollout
- A/B testing

That will feel very natural if you come from DevOps.

This is just release management with smarter artifacts.

---

# 12. Recommendation Metrics

For recommenders, we often care about the top of the ranked list.

Useful metrics:

- Precision@K
- Recall@K
- MAP
- NDCG

These matter because users do not scroll forever.
The first few items do the heavy lifting.

**Mini Pop Quiz**

If the best item is ranked 50th, is the recommendation system good?

No.

Technically relevant is not the same as practically useful.

---

# 13. A/B Testing for ML

A/B testing compares model variants in live traffic.

Use it when you want to know:

- does this model improve the real business metric?
- does it create new failure modes?
- does it affect user behavior in surprising ways?

**Key terms to sound solid in an interview**

- control vs treatment
- guardrail metrics
- sample size
- statistical significance
- ramp strategy
- rollback path

If you say "I would A/B test it" and stop there, that sounds junior.

If you say:

> "I would define the primary metric, guardrails, minimum detectable effect, and ramp plan"

that sounds like someone trusted with production.

---

# Quick Thought Experiment

You built a fraud model with:

- 99.4% accuracy
- poor recall
- decent precision

Would you ship?

Only if the business is okay missing fraud.
Which is a poetic way of saying:

No.

---

# How Would You Deploy This Using Azure Pipelines?

Imagine your evaluation gate as a release check.

Before deployment, your pipeline should verify:

- model artifact version
- feature schema match
- validation metric threshold
- drift check against recent data
- latency benchmark
- rollback-ready previous model version

That mindset will instantly make your ML answers stronger.
