---
module: Study Plans
topic: Week 2 Algorithms
subtopic: Day 15 16 Evaluation Metrics
status: unread
tags: [studyplans, ml, week-2-algorithms-day-15-16-ev]
---
# Day 15-16: Model Evaluation and Selection

## Why This Topic Comes Here

You have now studied a range of algorithms. The natural question is: how do you know which one to use? And once you have one — how do you know if it is actually working? Evaluation metrics are the answer to both questions, and they come at this point in the sequence because they only make sense after you understand the algorithms and the types of problems they solve. More critically, this is the topic where the most costly interview mistakes are made. Claiming a model "works" based on accuracy when the dataset is imbalanced is a red flag that signals shallow understanding. Interviewers probe this heavily.

---

## Executive Summary

| Category | Metric | Best For | Logic |
|----------|--------|----------|-------|
| **Classification** | **F1-Score** | Imbalanced Data | Harmonic mean of P & R |
| **Classification** | **PR-AUC** | Rare classes (Fraud) | Focus on Positive class |
| **Classification** | **ROC-AUC** | General performance | TPR vs FPR trade-off |
| **Regression** | **RMSE** | Penalizing large errors | Units match target |
| **Selection** | **K-Fold CV** | Reliable estimation | Average over $K$ runs |

---

## 1. Classification Metrics: Beyond Accuracy

**Why accuracy is almost always the wrong metric to report first:** Accuracy measures how often the model is right. But in most real problems, the classes are not equally represented and not equally important. Getting it wrong on the rare class is usually more costly than getting it wrong on the majority class. Accuracy hides both of these things.

### Precision and Recall

- **Precision**: $\frac{TP}{TP + FP}$ — of all predictions of "positive," how many were correct?
- **Recall**: $\frac{TP}{TP + FN}$ — of all actual positives, how many did the model find?

**Key insight:** Precision and Recall trade off against each other through the decision threshold. Raising the threshold makes the model more conservative (fewer positive predictions, higher precision, lower recall). Lowering the threshold makes it more aggressive (more positive predictions, lower precision, higher recall). This tradeoff is not specific to any algorithm — it applies to any model that outputs a probability score. Choosing a threshold is a business decision, not a modeling one.

**How to verify understanding:** A fraud detection system has 99% precision and 1% recall. Describe what this model is doing in practice and why a business might consider it useless despite the high precision.

**What trips people up:** Thinking that optimizing F1-score is always the right objective for imbalanced problems. F1 weights precision and recall equally. If false negatives are 10x more costly than false positives (e.g., cancer screening), you should optimize for recall or a weighted $F_\beta$ score where $\beta > 1$.

### The F1-Score

The harmonic mean of Precision and Recall:
$$F1 = 2 \cdot \frac{P \cdot R}{P + R}$$

**Key insight:** The harmonic mean is used specifically because it punishes extreme imbalance between precision and recall. A model with precision=1.0, recall=0.01 has F1=0.02. The arithmetic mean would give 0.505 — a misleadingly optimistic number. The harmonic mean forces both to be high.

**How to verify understanding:** Compute F1 for a model with P=0.9, R=0.9 vs. a model with P=0.99, R=0.5. Which has higher F1? Which would you prefer for a cancer screening task, and why?

**What trips people up:** Micro vs. Macro averaging in multi-class settings. **Macro** F1 computes F1 for each class separately and averages — treating all classes equally regardless of size. **Micro** F1 aggregates TP/FP/FN across all classes — treating all *instances* equally. For imbalanced multi-class problems, macro F1 is often more informative.

### ROC vs. PR Curve

- **ROC Curve**: Plots $TPR$ vs $FPR$. Ideal for balanced datasets. AUC = 0.5 is random; AUC = 1.0 is perfect.
- **PR Curve**: Plots $Precision$ vs $Recall$. Essential for heavily imbalanced datasets (e.g., 99% negative).

**Key insight:** In a dataset where 99% of examples are negative, a model that predicts "negative" for everything gets FPR = 0 — which looks great on the ROC curve. The PR curve exposes this because recall = 0 for such a model. ROC-AUC can be misleadingly high on imbalanced data; PR-AUC will not be.

**How to verify understanding:** You have a fraud detection model with ROC-AUC = 0.97 on a dataset where 0.1% of transactions are fraud. A colleague says this is a great model. What specific counter-question would you ask, and what number would you want to see?

**What trips people up:** Treating ROC-AUC as the universal "one number to rule them all" metric. It is a threshold-independent summary of model discriminability — useful for model comparison, but not for deciding what threshold to use in production, and not for imbalanced classes.

---

## 2. Regression Metrics

- **MAE** (Mean Absolute Error): Average absolute residual. Robust to outliers.
- **MSE** (Mean Squared Error): Average squared residual. Penalizes large errors heavily.
- **RMSE** (Root MSE): Same scale as the target variable. Easier to interpret.
- **R²** (Coefficient of Determination): Fraction of variance explained. Unitless.

**Key insight:** RMSE and MAE give different model rankings when large errors exist. If you care more about avoiding rare catastrophically wrong predictions (e.g., autonomous driving), optimize RMSE. If you want a model that is consistently decent on all cases (e.g., inventory forecasting), optimize MAE. These are not interchangeable choices.

**How to verify understanding:** Your regression model has MAE = 5 and RMSE = 40 on the same dataset. What does this tell you about the distribution of residuals? What specific type of error is the model making?

**What trips people up:** Using R² to compare models across different datasets or targets. R² = 0.8 in one domain says nothing about whether 0.8 is good in another domain. Always compare R² against a baseline (e.g., mean prediction), and prefer RMSE when you need to communicate the error in real units.

---

## 3. Robust Selection Techniques

**Why evaluation methodology belongs in the same session as metrics:** Choosing a metric is only half the problem. If you evaluate on data that leaked into training, any metric you compute is meaningless. The evaluation methodology determines whether the metric number corresponds to real-world performance.

### Cross-Validation (K-Fold)

Split data into $K$ folds. Train on $K-1$, validate on 1. Repeat $K$ times. Report mean ± std across folds.
- **Stratified K-Fold**: Ensures each fold has the same class distribution as the original data — critical for imbalanced classification.

**Key insight:** The standard deviation across folds is as informative as the mean. A model with mean AUC = 0.80 ± 0.02 is more reliable than one with mean AUC = 0.81 ± 0.15. High variance across folds means the model's performance depends heavily on which examples end up in which fold — a sign of either a small dataset, high model variance, or both.

**How to verify understanding:** You run 5-fold CV and get fold scores of [0.95, 0.61, 0.92, 0.89, 0.91]. Before taking the mean and reporting 0.856, what should you investigate first?

**What trips people up:** Using cross-validation metrics to select a model, then reporting those same metrics as the expected test performance. This is optimistic — you have implicitly used the validation folds to select the model. Use nested cross-validation when both selecting and evaluating a model.

### Nested Cross-Validation

- Inner loop: Hyperparameter tuning.
- Outer loop: Error estimation.

**Key insight:** Without nesting, using cross-validation for both hyperparameter selection and performance estimation gives an optimistic bias. The outer loop provides an honest estimate of how well your best model would perform on truly unseen data.

**How to verify understanding:** You use 5-fold CV to select the best model from 10 candidate models. You report the best CV score as your expected performance. Explain why this is likely overoptimistic and what you should do instead.

**What trips people up:** Thinking nested CV is only for hyperparameter tuning. It applies whenever you make any selection decision based on validation performance — including choosing between algorithms, feature sets, or preprocessing strategies.

---

## Interview Questions

**1. "When would you prefer F1-score over Accuracy?"**
> When classes are imbalanced. For example, in fraud detection (99.9% legit), a model that predicts "legit" always gets 99.9% accuracy but is useless. F1-score would be 0, correctly identifying the failure.

**2. "What is the difference between Micro and Macro averaging for F1?"**
> **Macro**: Calculates F1 for each class and averages them (treats all classes equally). **Micro**: Aggregates all TPs, FPs, FNs across classes (treats all *instances* equally).

**3. "Why use R-Squared in regression?"**
> It provides a relative measure of how much variance is explained. R-Squared of 0.8 means the model explains 80% of the variance. However, it can be misleading for non-linear models (use Adjusted R-Squared if adding many features).

---

## Monitoring in Production

```python
from sklearn.metrics import classification_report, confusion_matrix

# Get a full breakdown of P, R, F1 per class
print(classification_report(y_true, y_pred))

# Plot confusion matrix to see where the model is 'confused'
cm = confusion_matrix(y_true, y_pred)
```
