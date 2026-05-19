# Imbalanced Data

---

## The Problem

**The problem**: You train a classifier on a fraud dataset: 99% of transactions are legitimate, 1% are fraud. The model learns to predict "not fraud" for everything. It achieves 99% accuracy. It catches zero fraud cases. The loss function was minimized, but the model is completely useless.

**The core insight**: Accuracy treats all errors equally regardless of class frequency. A model can game accuracy by always predicting the majority class — paying a small per-sample penalty on 1% of examples to avoid the cost of actually learning. The minority class contributes negligibly to the total loss, so the model ignores it.

This is not just an evaluation problem — it is a training problem. The model's gradients are dominated by majority class examples. Unless the loss function, the sampling distribution, or the class weights are adjusted, the model will not learn to separate the minority class.

**When this matters most**: Fraud detection, rare disease diagnosis, intrusion detection, predictive maintenance, any rare-event classification.

---

## Evaluation: Metrics That Don't Lie

**The problem**: Accuracy is useless here. You need metrics that measure performance on the rare class, indifferent to how many majority examples you happened to classify correctly.

### Precision, Recall, F1

```
Precision = TP / (TP + FP)   # of predicted positives, what fraction are real
Recall    = TP / (TP + FN)   # of real positives, what fraction did we catch
F1        = 2 * P * R / (P + R)
```

Precision and recall are in tension — you can raise recall by predicting positive more aggressively, but this reduces precision. F1 is their harmonic mean, which penalizes extreme imbalance between them harder than the arithmetic mean.

### F-Beta Score

**The problem**: In medical screening, missing a true case (false negative) costs far more than a false alarm (false positive). F1 treats them equally. You need a metric that acknowledges the asymmetric cost.

**The mechanics**: The beta parameter controls the trade-off. $\beta > 1$ weights recall more heavily; $\beta < 1$ weights precision more heavily.

$$F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 \cdot P + R}$$

```python
from sklearn.metrics import fbeta_score
score = fbeta_score(y_true, y_pred, beta=2)   # recall weighted twice as much as precision
```

### PR AUC vs ROC AUC

**The problem**: ROC AUC looks at (FPR, TPR). With 99% negatives, even a weak classifier has an enormous pool of true negatives — the true-negative rate stays high even with many false positives. The large TN count inflates ROC AUC without reflecting whether you actually catch fraud.

**The core insight**: PR AUC (Average Precision) ignores true negatives entirely. It measures how well the model ranks positives among the examples it flags. This is exactly the question you care about when positives are rare.

```python
from sklearn.metrics import average_precision_score, roc_auc_score

ap  = average_precision_score(y_true, y_scores)   # preferred for imbalanced problems
roc = roc_auc_score(y_true, y_scores)              # optimistic under imbalance
```

### Matthews Correlation Coefficient (MCC)

**The problem**: F1 ignores true negatives. A model that aggressively predicts positive gets high recall and potentially high F1, even if it generates huge false positive volumes. You need a metric that accounts for all four quadrants of the confusion matrix simultaneously.

**The core insight**: MCC is the correlation coefficient between true and predicted labels. It gives high scores only when the model performs well on *both* classes.

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

```python
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_true, y_pred)
```

Ranges from -1 (perfect inverse) to 0 (random) to +1 (perfect). Robust to class imbalance.

---

## Resampling

**The problem**: The loss function is dominated by majority class examples. One solution: change the training distribution so the minority class appears more often (or the majority appears less).

**The core insight**: Change the class distribution the model sees during training. Never change the evaluation distribution — the test set must remain untouched at its original class ratio to give a realistic estimate of production performance.

### Random Oversampling and Undersampling

**Random oversampling**: Duplicate minority class samples at random until class balance is achieved.
**Random undersampling**: Delete majority class samples at random.

```python
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
```

**What breaks**: Random oversampling duplicates samples — the model sees identical training examples multiple times and can overfit to them. Random undersampling discards potentially informative majority examples.

---

### SMOTE — Synthetic Minority Oversampling Technique

**The problem**: Random oversampling creates duplicates. The model memorizes those exact points rather than learning the underlying minority class region.

**The core insight**: Generate *new* minority samples by interpolating between existing ones. If two minority samples are neighbors, any point between them should also be minority class. This introduces diversity rather than copies.

**The mechanics**:
1. For each minority sample $x$, find its k nearest minority-class neighbors.
2. Pick one neighbor $x_{nn}$ at random.
3. Generate: $x_{new} = x + \lambda \cdot (x_{nn} - x)$ where $\lambda \sim \text{Uniform}(0, 1)$.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(k_neighbors=5, random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
```

**What breaks**: SMOTE interpolates in feature space without awareness of the decision boundary. When minority and majority classes overlap, synthetic points are generated inside majority-class territory — they are labeled minority but look like majority. Use SMOTEENN or ADASYN in these cases. SMOTE also does not work on categorical features directly — use SMOTE-NC.

---

### ADASYN — Adaptive Synthetic Sampling

**The problem**: SMOTE generates the same density of synthetic samples everywhere in minority space. But the hard cases — minority samples surrounded by majority samples, near the decision boundary — need more coverage than the easy cases in the center of the minority cluster.

**The core insight**: Generate more synthetic samples in regions where the minority class is harder to learn (surrounded by more majority neighbors), and fewer in easy regions. Focus synthetic capacity where the model struggles most.

**The mechanics**: For each minority sample, compute the fraction of its k neighbors that belong to the majority class. This fraction becomes the sampling weight — minority samples with mostly majority neighbors get more synthetic neighbors generated around them.

```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(n_neighbors=5, random_state=42)
X_res, y_res = adasyn.fit_resample(X_train, y_train)
```

**What breaks**: The adaptive weighting amplifies noisy minority samples near the boundary — outliers in the minority class get extra synthetic samples generated around them, reinforcing the noise. ADASYN can worsen performance if the minority class has a noisy boundary.

---

### Tomek Links — Boundary Cleaning

**The problem**: After oversampling, the class boundary is cluttered with ambiguous samples from both sides that make it hard for the model to draw a clean separation.

**The core insight**: A Tomek link is a pair of samples from different classes where each is the other's nearest neighbor. These pairs sit exactly on or near the boundary. Removing the majority sample from each Tomek link cleans the boundary without removing large amounts of data.

**The mechanics**: Find all Tomek link pairs. Remove the majority-class member of each pair. The result is a marginally cleaner decision boundary.

```python
from imblearn.under_sampling import TomekLinks

tl = TomekLinks()
X_res, y_res = tl.fit_resample(X_train, y_train)
```

**What breaks**: Tomek Links alone produce minimal undersampling — only boundary-adjacent majority points are removed. Used primarily as a post-processing step after oversampling.

---

### Combination: SMOTEENN and SMOTETomek

**The problem**: Oversampling generates synthetic samples that may land in ambiguous regions. Undersampling removes some boundary noise. Using both together — oversample first, then clean — produces a better-separated training set.

```python
from imblearn.combine import SMOTEENN, SMOTETomek

smt   = SMOTETomek(random_state=42)     # SMOTE then remove Tomek links
smenn = SMOTEENN(random_state=42)       # SMOTE then Edited Nearest Neighbors

X_res, y_res = smenn.fit_resample(X_train, y_train)
```

SMOTEENN is more aggressive — ENN removes any sample whose class label disagrees with the majority of its k neighbors, affecting both classes. SMOTETomek is gentler, only removing Tomek link members.

**What breaks**: Both methods reduce the total training set size after the resample+clean cycle. Very small datasets can lose too much data in the cleaning step.

---

## Cost-Sensitive Learning

**The problem**: Resampling modifies the data — it creates synthetic samples or discards real ones. This can introduce artifacts. An alternative: keep the data as-is, but tell the loss function that misclassifying a minority-class sample costs more.

**The core insight**: The optimizer minimizes total weighted loss. If minority-class errors are assigned a higher weight, the model must learn to avoid them to reduce total loss. Same effect as resampling, no data modification.

### class_weight='balanced'

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

lr = LogisticRegression(class_weight='balanced')
rf = RandomForestClassifier(class_weight='balanced')
```

**The mechanics**: sklearn computes $w_j = \frac{n_{\text{samples}}}{n_{\text{classes}} \cdot n_j}$ for class $j$, where $n_j$ is the count of class $j$. For a 98/2 split: the minority class gets weight ≈ 25, the majority gets ≈ 0.51. The minority class contributes ~49x more loss per sample.

### Custom Class Weights

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
lr = LogisticRegression(class_weight=class_weight_dict)
```

### Per-Sample Weights

More flexible than class weights — lets you combine class rebalancing with other sample-level signals (e.g., more recent samples weighted higher).

```python
from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight('balanced', y=y_train)
clf.fit(X_train, y_train, sample_weight=sample_weights)
```

### XGBoost scale_pos_weight

```python
import xgboost as xgb

ratio = (y_train == 0).sum() / (y_train == 1).sum()
clf = xgb.XGBClassifier(scale_pos_weight=ratio, eval_metric='aucpr')
```

**What breaks**: Class weights don't change the model's predicted probabilities — they change which errors are penalized more during training. The output probabilities are still in the original prior space. If you need calibrated probabilities, apply post-hoc calibration after cost-sensitive training.

---

## Threshold Moving

**The problem**: The default decision threshold of 0.5 was calibrated for balanced data — it reflects the prior assumption that positives and negatives are equally likely. With a 99/1 split, the model's prior for the positive class is 1%, so predicted probabilities for positive cases are naturally low. Applying a 0.5 threshold misses most of them.

**The core insight**: The model outputs probabilities. The threshold is a post-hoc decision about where to draw the line between predicted classes. Adjusting the threshold changes the operating point on the precision-recall curve — at zero extra training cost.

**The mechanics**: Compute predicted probabilities. Sweep the threshold from 0 to 1. For each threshold, compute the metric you care about (F1, F-beta, cost). Pick the threshold that maximizes your criterion.

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

clf.fit(X_train, y_train)
y_scores = clf.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]

y_pred_custom = (y_scores >= best_threshold).astype(int)
```

For asymmetric business costs:
```python
cost_fn, cost_fp = 100, 1   # missing fraud costs 100x more than a false alert

costs = []
for t in thresholds:
    y_pred_t = (y_scores >= t).astype(int)
    fp = ((y_pred_t == 1) & (y_test == 0)).sum()
    fn = ((y_pred_t == 0) & (y_test == 1)).sum()
    costs.append(cost_fp * fp + cost_fn * fn)

best_threshold_cost = thresholds[np.argmin(costs)]
```

**What breaks**: Threshold tuning on the test set is leakage — use a separate validation set. Optimal threshold on validation may not generalize perfectly to production, especially with class distribution drift.

---

## Ensemble Methods for Imbalance

**The problem**: You want to handle imbalance at the ensemble level — each base learner sees a balanced view of the data — without modifying the overall training distribution or adding synthetic samples.

### BalancedRandomForest

Each tree in the forest is trained on a bootstrap sample drawn with equal class balance, rather than reflecting the original imbalanced ratio.

```python
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(n_estimators=100, replacement=True, random_state=42)
brf.fit(X_train, y_train)
```

### EasyEnsemble

Trains an ensemble of AdaBoost classifiers, each on a different random undersampled majority subset paired with all minority samples. Uses many majority subsets instead of discarding most of them.

```python
from imblearn.ensemble import EasyEnsembleClassifier

ee = EasyEnsembleClassifier(n_estimators=10, random_state=42)
ee.fit(X_train, y_train)
```

**What breaks**: Balanced bootstrap sampling changes the effective class prior the model sees — the model is trained on a different class distribution than the deployment distribution. This can cause miscalibration. Apply post-hoc calibration on the original class distribution.

---

## Focal Loss

**The problem**: In dense object detection (RetinaNet), the ratio of background to foreground is 1000:1. Standard cross-entropy is overwhelmed by easy background examples — the model converges to predicting background everywhere because that alone minimizes total loss. Class weighting helps but doesn't fully solve it: easy examples, even when weighted, still dominate the loss in aggregate.

**The core insight**: Down-weight easy examples dynamically, based on how confidently the model already classifies them. Let hard examples drive the gradient, regardless of their class.

**The mechanics**: Add a modulating factor $(1 - p_t)^\gamma$ to cross-entropy. When the model is already confident about a sample ($p_t \to 1$), the modulating factor approaches zero — that sample contributes negligibly to the gradient. When the model is uncertain ($p_t \to 0$), the factor approaches 1 — full gradient contribution.

$$FL(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

- $\gamma = 0$: reduces to weighted cross-entropy
- $\gamma = 2$ (typical): easy examples downweighted by ~100x

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss  = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        p_t       = torch.exp(-bce_loss)
        alpha_t   = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_w   = alpha_t * (1 - p_t) ** self.gamma
        loss      = focal_w * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

**What breaks**: Focal loss requires tuning both $\alpha$ and $\gamma$ — adding two more hyperparameters to an already complex training loop. It was designed for object detection (continuous prediction at every pixel location); its benefit on tabular binary classification is less consistent. Try class weighting first.

---

## Label Smoothing

**The problem**: On imbalanced problems, the model learns to output very high confidence on majority class examples — not because they are genuinely certain, but because cross-entropy loss rewards pushing the majority class probability toward 1.0. This overconfidence means the model's threshold for the minority class must be set extremely low, and probability outputs are poorly calibrated.

**The core insight**: Replace hard targets (0 and 1) with soft targets. The model is penalized for being too confident — it cannot perfectly minimize the loss by outputting probability 1.0 on any example.

**The mechanics**: $y_{\text{smooth}} = y(1 - \varepsilon) + \varepsilon/K$ where $K$ is the number of classes and $\varepsilon$ is the smoothing factor (0.05–0.1 typical).

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, n_classes: int = 2):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_targets = torch.full_like(log_probs, self.smoothing / (self.n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_targets * log_probs).sum(dim=-1).mean()
```

**What breaks**: Label smoothing reduces overconfidence on all classes — it reduces calibration error but also softens the model's signal on easy examples. It is complementary to focal loss and class weighting, not a replacement.

---

## Practical Guidelines

### Choosing an Approach

| Scenario | Recommended approach |
|---|---|
| Large dataset (> 100k) | Undersampling or `class_weight='balanced'` |
| Small dataset (< 10k) | SMOTE or ADASYN |
| Extreme imbalance (< 0.5%) | SMOTE + cost-sensitive learning combined |
| Overlapping classes | SMOTEENN (boundary cleaning) |
| Need interpretability / no data modification | `class_weight='balanced'` only |
| Deep learning | Focal loss or class weighting |

### The Pipeline Safety Rule

**The problem**: SMOTE applied before cross-validation exposes the validation fold to synthetic samples derived from its real samples — leakage.

**The core insight**: Use `imblearn.pipeline.Pipeline` (not sklearn's), which applies `fit_resample` only inside each training fold.

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf',   LogisticRegression(class_weight='balanced'))
])

scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5), scoring='average_precision')
```

### Never Resample the Test Set

Resampling changes the class distribution. Evaluating on a resampled test set produces metrics for a different problem than the one you are deploying to.

```python
# Correct: resample only training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

clf.fit(X_train_res, y_train_res)
clf.predict(X_test)   # X_test is the original untouched distribution
```

### Always Use Stratified Splits

Without stratification, a fold may contain no minority-class examples at all — making evaluation impossible and training misleading.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```
