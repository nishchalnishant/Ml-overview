# Imbalanced Data

Handling datasets where one class vastly outnumbers another — covering evaluation, resampling, cost-sensitive learning, threshold tuning, ensemble methods, and loss-level fixes.

---

## The Problem

Class imbalance occurs when the target distribution is skewed — e.g., 99% negative, 1% positive. A model that predicts the majority class for every sample achieves 99% accuracy while being completely useless.

**Why standard accuracy is misleading:**

| Metric | Majority-class predictor | Good detector |
|--------|--------------------------|---------------|
| Accuracy | 99% | 95% |
| Recall (minority) | 0% | 85% |
| Precision (minority) | undefined | 70% |

The model learns to ignore the minority class because misclassifying it incurs a smaller penalty in aggregate loss.

**When it matters most:**
- Fraud detection (< 0.1% fraud rate)
- Medical diagnosis (rare diseases, rare adverse events)
- Intrusion detection / cybersecurity
- Predictive maintenance (rare machine failures)
- Any rare event detection problem

---

## Evaluation Metrics for Imbalanced Data

### Why accuracy fails

With 1% positive rate, a dummy classifier scores 99% accuracy. Any useful metric must be indifferent to label proportion.

### Precision, Recall, F1

```
Precision = TP / (TP + FP)   # of predicted positives, how many are real
Recall    = TP / (TP + FN)   # of real positives, how many did we catch
F1        = 2 * P * R / (P + R)
```

### F-beta Score

Controls the trade-off between precision and recall:

```
F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
```

- `beta > 1`: recall weighted more (use when missing positives is costly — medical)
- `beta < 1`: precision weighted more (use when false alarms are costly — spam filter)
- `beta = 1`: standard F1

```python
from sklearn.metrics import fbeta_score
score = fbeta_score(y_true, y_pred, beta=2)  # recall twice as important
```

### Precision-Recall Curve vs ROC

**ROC AUC** is optimistic under high imbalance — the large number of true negatives inflates the true-negative rate, making a weak classifier look good.

**PR AUC** (Average Precision) directly measures performance on the minority class. It is the preferred summary metric when positives are rare.

```python
from sklearn.metrics import average_precision_score, roc_auc_score

ap  = average_precision_score(y_true, y_scores)   # preferred
roc = roc_auc_score(y_true, y_scores)              # optimistic under imbalance
```

### Matthews Correlation Coefficient (MCC)

Produces a balanced measure even when classes are very unequal. Ranges from -1 (inverse), 0 (random), to +1 (perfect).

```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

```python
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_true, y_pred)
```

MCC uses all four quadrants of the confusion matrix. It only gives a high score if the classifier performs well on both classes simultaneously.

### Cohen's Kappa

Measures agreement beyond chance. Accounts for class distribution:

```
Kappa = (P_o - P_e) / (1 - P_e)
```

where `P_o` is observed accuracy and `P_e` is expected agreement by chance.

```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_true, y_pred)
```

---

## Resampling Methods

The core idea: change the class distribution seen during training, never during evaluation.

### Random Oversampling / Undersampling

**Random oversampling** duplicates minority samples at random. Risk: overfitting to duplicated points.

**Random undersampling** removes majority samples at random. Risk: discards potentially useful information.

```python
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
```

---

### SMOTE — Synthetic Minority Oversampling Technique

Instead of duplicating, SMOTE generates synthetic minority samples by interpolating between existing minority instances and their k-nearest minority neighbors.

**Algorithm:**
1. For each minority sample `x`, find its k nearest minority neighbors.
2. Pick a random neighbor `x_nn`.
3. Generate: `x_new = x + lambda * (x_nn - x)` where `lambda ~ Uniform(0, 1)`.

```python
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(
    n_samples=10000, n_features=20,
    weights=[0.98, 0.02],   # 2% minority
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

smote = SMOTE(k_neighbors=5, random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"After  SMOTE: {dict(zip(*np.unique(y_res,   return_counts=True)))}")

clf = RandomForestClassifier(random_state=42)
clf.fit(X_res, y_res)
print(classification_report(y_test, clf.predict(X_test)))
```

**Limitations:**
- Synthesizes in feature space, not accounting for decision boundary
- Can generate noisy samples in overlapping regions
- Does not work directly on categorical features (use SMOTENC)

---

### ADASYN — Adaptive Synthetic Sampling

ADASYN is a density-aware extension of SMOTE. It generates more synthetic samples in regions where the minority class is harder to learn (i.e., surrounded by more majority samples), and fewer in easy regions.

```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(n_neighbors=5, random_state=42)
X_res, y_res = adasyn.fit_resample(X_train, y_train)
```

Use ADASYN when the minority class has varying local density — it focuses capacity on the decision boundary where it matters most.

---

### Tomek Links — Boundary Cleaning

A Tomek link exists between samples `(x_i, x_j)` from different classes if no other sample is closer to either. These pairs are on or near the decision boundary. Removing the majority sample from each link cleans the boundary.

```python
from imblearn.under_sampling import TomekLinks

tl = TomekLinks()
X_res, y_res = tl.fit_resample(X_train, y_train)
```

Tomek Links alone produce mild undersampling. They are most useful as a cleaning step after oversampling.

---

### Combination Methods: SMOTEENN and SMOTETomek

Combine oversampling with boundary cleaning for better-separated classes.

```python
from imblearn.combine import SMOTEENN, SMOTETomek
import numpy as np

# SMOTETomek: oversample with SMOTE, then remove Tomek links
smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X_train, y_train)

# SMOTEENN: oversample with SMOTE, then clean with Edited Nearest Neighbours
# ENN removes samples whose class differs from the majority of their k neighbors
smenn = SMOTEENN(random_state=42)
X_res, y_res = smenn.fit_resample(X_train, y_train)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_res, y_res)
print(classification_report(y_test, clf.predict(X_test)))
```

**SMOTEENN** is often more aggressive than SMOTETomek and better at removing class overlap. **SMOTETomek** is a gentler boundary cleaner.

---

## Cost-Sensitive Learning

Instead of changing the data, penalize misclassification of the minority class more heavily during training.

### class_weight='balanced' in sklearn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Automatically computes weights inversely proportional to class frequencies
lr = LogisticRegression(class_weight='balanced')
rf = RandomForestClassifier(class_weight='balanced')
```

**How sklearn computes balanced class weights:**

```
w_j = n_samples / (n_classes * n_samples_j)
```

For a 98/2 split with 10,000 samples and 2 classes:
- Majority weight: `10000 / (2 * 9800) ≈ 0.51`
- Minority weight: `10000 / (2 * 200)  = 25.0`

The minority class loss contribution is ~49x larger per sample.

### Custom class weights

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

lr = LogisticRegression(class_weight=class_weight_dict)
```

### sample_weight

Applies per-sample weights at fit time — more flexible than class_weight:

```python
from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight('balanced', y=y_train)
clf.fit(X_train, y_train, sample_weight=sample_weights)
```

Use `sample_weight` when you need instance-level control (e.g., more recent samples weighted higher in addition to class rebalancing).

### Cost-sensitive loss functions

Some frameworks allow explicit cost matrices:

```python
# XGBoost: scale_pos_weight balances positive/negative weights
import xgboost as xgb

ratio = (y_train == 0).sum() / (y_train == 1).sum()
clf = xgb.XGBClassifier(scale_pos_weight=ratio)
```

---

## Threshold Moving

The default decision threshold of 0.5 is calibrated for balanced data. For imbalanced data, moving the threshold changes the precision-recall trade-off without retraining.

**Key insight:** the model outputs probabilities; the threshold determines the operating point on the PR curve. Optimizing threshold is free — no resampling or retraining needed.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.linear_model import LogisticRegression

# Train model
clf = LogisticRegression(class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Get probabilities
y_scores = clf.predict_proba(X_test)[:, 1]

# Compute PR curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

# Find threshold that maximizes F1
f1_scores = 2 * precisions[:-1] * recalls[:-1] / (
    precisions[:-1] + recalls[:-1] + 1e-9
)
best_idx       = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1        = f1_scores[best_idx]

print(f"Best threshold: {best_threshold:.3f}")
print(f"Best F1:        {best_f1:.3f}")

# Apply custom threshold
y_pred_custom = (y_scores >= best_threshold).astype(int)
print(classification_report(y_test, y_pred_custom))

# Plot PR curve
plt.figure(figsize=(8, 5))
plt.plot(recalls[:-1], precisions[:-1], label='PR curve')
plt.scatter(recalls[best_idx], precisions[best_idx],
            color='red', zorder=5, label=f'Best F1 @ {best_threshold:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Optimal Threshold')
plt.legend()
plt.tight_layout()
plt.show()
```

### Business-cost threshold

When false negatives and false positives have asymmetric costs:

```python
# cost_fn: cost of missing a positive (false negative)
# cost_fp: cost of a false alarm (false positive)
cost_fn, cost_fp = 100, 1

total_costs = cost_fp * (1 - precisions[:-1]) * (y_scores >= thresholds[:, None]).sum(axis=1).mean() \
            + cost_fn * (1 - recalls[:-1]) * (y_test == 1).mean()

# Simplified per-threshold cost
costs = []
for t in thresholds:
    y_pred_t = (y_scores >= t).astype(int)
    fp = ((y_pred_t == 1) & (y_test == 0)).sum()
    fn = ((y_pred_t == 0) & (y_test == 1)).sum()
    costs.append(cost_fp * fp + cost_fn * fn)

best_threshold_cost = thresholds[np.argmin(costs)]
print(f"Cost-optimal threshold: {best_threshold_cost:.3f}")
```

---

## Ensemble Methods for Imbalance

Specialized ensembles combine resampling with bagging to get robust estimates.

### BalancedRandomForest

Draws a balanced bootstrap sample (equal minority/majority) for each tree:

```python
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(
    n_estimators=100,
    replacement=True,       # sample with replacement
    random_state=42
)
brf.fit(X_train, y_train)
print(classification_report(y_test, brf.predict(X_test)))
```

### BalancedBaggingClassifier

Wraps any base estimator with balanced bootstrap sampling:

```python
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bbc = BalancedBaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    sampling_strategy='auto',
    random_state=42
)
bbc.fit(X_train, y_train)
```

### EasyEnsemble

Trains an ensemble of AdaBoost classifiers, each on a different random undersampled majority subset. Multiple majority subsets are exploited without discarding information:

```python
from imblearn.ensemble import EasyEnsembleClassifier

ee = EasyEnsembleClassifier(n_estimators=10, random_state=42)
ee.fit(X_train, y_train)
print(classification_report(y_test, ee.predict(X_test)))
```

**When to prefer ensemble methods:** when you have enough data that resampling is unnecessary but still want to handle imbalance without touching the training set distribution explicitly.

---

## Focal Loss

Focal loss addresses imbalance at the loss level by down-weighting easy, well-classified examples and focusing training on hard, misclassified ones.

**Formula:**

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

- `p_t`: model's estimated probability for the true class
- `alpha_t`: class-frequency weighting term (handles prior imbalance)
- `gamma`: focusing parameter (typically 2). Higher gamma = more focus on hard examples
- `(1 - p_t)^gamma`: modulating factor — near zero for easy examples (p_t → 1), near one for hard examples (p_t → 0)

**Intuition:** when `gamma=0`, focal loss reduces to weighted cross-entropy. As gamma increases, easy examples contribute negligibly to the loss and gradients concentrate on the minority, harder-to-classify samples.

**Origin:** introduced in RetinaNet (Lin et al., 2017) for dense object detection, where the extreme foreground/background ratio (1:1000+) made standard cross-entropy fail.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    Args:
        alpha: weight for positive class. Set to class frequency ratio for imbalance.
        gamma: focusing parameter. 0 = standard cross-entropy. Typical: 2.
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N,) raw scores; targets: (N,) binary {0,1}
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )
        p_t = torch.exp(-bce_loss)                          # predicted probability of true class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# Usage
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# Simulate imbalanced batch
logits  = torch.randn(32)
targets = torch.zeros(32)
targets[:2] = 1  # 2/32 positive — heavily imbalanced

loss = criterion(logits, targets)
print(f"Focal loss: {loss.item():.4f}")
```

**Numpy/sklearn-compatible version for non-PyTorch pipelines:**

```python
import numpy as np

def focal_loss_numpy(y_true, y_prob, alpha=0.25, gamma=2.0):
    """Binary focal loss. y_prob: predicted probabilities for positive class."""
    y_true = np.asarray(y_true, dtype=float)
    p_t    = np.where(y_true == 1, y_prob, 1 - y_prob)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    loss   = -alpha_t * (1 - p_t) ** gamma * np.log(np.clip(p_t, 1e-8, 1.0))
    return loss.mean()
```

---

## Label Smoothing

Label smoothing replaces hard targets (0 and 1) with soft targets, reducing model overconfidence:

```
y_smooth = y * (1 - eps) + eps / K
```

where `K` is the number of classes and `eps` is the smoothing factor (typical: 0.05-0.1).

**Why it helps with imbalance:** severely imbalanced models learn to output very high confidence on majority class examples. Label smoothing penalizes this overconfidence, leading to better-calibrated probabilities and improved threshold tuning.

```python
import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, n_classes: int = 2):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, C); targets: (N,) class indices
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_targets = torch.full_like(log_probs, self.smoothing / (self.n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_targets * log_probs).sum(dim=-1).mean()
```

Label smoothing is complementary to other techniques — combine with class_weight or focal loss for severe imbalance.

---

## Practical Guidelines

### When to oversample vs undersample

| Scenario | Recommended approach |
|----------|---------------------|
| Large dataset (> 100k samples) | Undersample or class_weight |
| Small dataset | Oversample (SMOTE/ADASYN) |
| Extreme imbalance (< 0.5%) | SMOTE + cost-sensitive learning |
| Overlapping classes | SMOTEENN (boundary cleaning) |
| Need interpretability | class_weight only (no data modification) |
| Deep learning | Focal loss or class_weight |

### Always use stratified splits

```python
from sklearn.model_selection import train_test_split, StratifiedKFold

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42  # stratify= is critical
)

# Stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Without stratification, a fold might contain zero minority samples.

### Never resample the test set

Resampling changes the class distribution. Evaluating on a resampled test set gives an inflated and misleading estimate of real-world performance.

```python
# CORRECT: resample only training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

clf.fit(X_train_res, y_train_res)
clf.predict(X_test)          # X_test is untouched — original distribution
```

### Pipeline safety

Use `imblearn.pipeline.Pipeline` (not sklearn's) to ensure resampling only happens inside cross-validation folds:

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf',   LogisticRegression(class_weight='balanced'))
])

# SMOTE is applied per fold — no data leakage
scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5),
                         scoring='average_precision')
print(f"AP: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Checklist

- [ ] Use stratified splits at every step
- [ ] Report PR AUC / F1 / MCC — not raw accuracy
- [ ] Tune threshold after training using PR curve
- [ ] Resample only training data, never test/validation
- [ ] Use `imblearn.pipeline.Pipeline` in cross-validation
- [ ] Start simple: `class_weight='balanced'` before complex resampling

---

## Key Interview Points

**Q: Why is accuracy misleading for imbalanced data?**
A: A classifier predicting the majority class for every sample achieves high accuracy (equal to majority class proportion) while having zero recall on the minority class. Accuracy treats all errors equally regardless of class frequency.

**Q: PR AUC vs ROC AUC — when do you prefer each?**
A: ROC AUC is appropriate when the class distribution is roughly balanced. PR AUC is preferred under high imbalance because it does not factor in true negatives — the large TN count inflates ROC scores without reflecting performance on the rare class. PR AUC directly measures minority class precision and recall.

**Q: How does SMOTE work, and what are its failure modes?**
A: SMOTE interpolates between a minority sample and one of its k nearest minority neighbors to generate a synthetic sample. It fails when minority and majority classes overlap heavily — it generates synthetic samples in ambiguous regions. SMOTEENN or ADASYN are better in that case.

**Q: What does class_weight='balanced' do mechanically?**
A: It sets per-class loss weights to `n_samples / (n_classes * n_samples_j)`, making each class contribute equally to the total loss regardless of frequency. A class with 1% frequency gets ~50x higher weight than a class with 50% frequency in a two-class problem.

**Q: Why is threshold 0.5 wrong for imbalanced problems?**
A: 0.5 is calibrated for equal priors. With a 99/1 split, the model's prior for the positive class is 1%, so probabilities are naturally low. A threshold of 0.5 will miss most positives. The optimal threshold is found by scanning the PR curve for the operating point that minimizes your business cost or maximizes F1.

**Q: What is focal loss and when do you use it?**
A: Focal loss adds a modulating factor `(1 - p_t)^gamma` to cross-entropy. Easy examples (high `p_t`) contribute negligibly; hard examples dominate training. Used in dense object detection (RetinaNet) where background regions outnumber foreground by 1000:1. Also effective in any deep learning setting with severe class imbalance.

**Q: What is MCC and why is it preferred over F1 for binary imbalanced problems?**
A: Matthews Correlation Coefficient uses all four cells of the confusion matrix (TP, TN, FP, FN). F1 ignores true negatives entirely. Under severe imbalance, a model can get high F1 by aggressively predicting positives (high recall, mediocre precision), but MCC will penalize it for the corresponding explosion in false positives relative to true negatives.

**Q: Should you resample validation or test sets?**
A: Never. Resampling changes the class distribution and produces optimistic, unrealistic evaluation results. Only resample training data, and use `imblearn.pipeline.Pipeline` inside cross-validation to prevent leakage.

**Q: What is the difference between SMOTEENN and SMOTETomek?**
A: Both oversample with SMOTE then clean the boundary. SMOTETomek removes Tomek links — samples that are each other's nearest neighbor across classes. SMOTEENN applies Edited Nearest Neighbours — removes any sample whose class disagrees with the majority of its k neighbors. SMOTEENN is more aggressive and removes more samples, resulting in cleaner but smaller training sets.
