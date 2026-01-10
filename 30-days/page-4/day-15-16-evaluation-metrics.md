# Day 15-16: Model Evaluation and Selection

##  Executive Summary
| Category | Metric | Best For | Logic |
|----------|--------|----------|-------|
| **Classification** | **F1-Score** | Imbalanced Data | Harmonic mean of P & R |
| **Classification** | **PR-AUC** | Rare classes (Fraud) | Focus on Positive class |
| **Classification** | **ROC-AUC** | General performance | TPR vs FPR trade-off |
| **Regression** | **RMSE** | Penalizing large errors | Units match target |
| **Selection** | **K-Fold CV** | Reliable estimation | Average over $K$ runs |

---

##  1. Classification Metrics: Beyond Accuracy

### Precision and Recall
- **Precision**: $\frac{TP}{TP + FP}$ ("Of all predicted positive, how many were right?")
- **Recall**: $\frac{TP}{TP + FN}$ ("Of all actual positive, how many did we catch?")
- **The Trade-off**: Increasing threshold $\rightarrow$ Higher Precision, Lower Recall.

### ROC vs. PR Curve
- **ROC Curve**: Plots $TPR$ vs $FPR$. Ideal for balanced datasets.
- **PR Curve**: Plots $Precision$ vs $Recall$. Essential for heavily imbalanced datasets (e.g., 99% negative).

---

##  2. Robust Selection Techniques

### Cross-Validation (K-Fold)
Split data into $K$ folds. Train on $K-1$, validate on 1. Repeat $K$ times.
- **Stratified K-Fold**: Ensures each fold has the same class distribution as the original data (Critical for imbalance).

### Nested Cross-Validation
Used for tuning hyperparameters *and* estimating performance simultaneously.
- Inner loop: Hyperparameter tuning.
- Outer loop: Error estimation.

---

##  Interview Questions

**1. "When would you prefer F1-score over Accuracy?"**
> When classes are imbalanced. For example, in fraud detection (99.9% legit), a model that predicts "legit" always gets 99.9% accuracy but is useless. F1-score would be 0, correctly identifying the failure.

**2. "What is the difference between Micro and Macro averaging for F1?"**
> **Macro**: Calculates F1 for each class and averages them (treats all classes equally). **Micro**: Aggregates all TPs, FPs, FNs across classes (treats all *instances* equally).

**3. "Why use R-Squared in regression?"**
> It provides a relative measure of how much variance is explained. R-Squared of 0.8 means the model explains 80% of the variance. However, it can be misleading for non-linear models (use Adjusted R-Squared if adding many features).

---

##  Monitoring in Production
```python
from sklearn.metrics import classification_report, confusion_matrix

# Get a full breakdown of P, R, F1 per class
print(classification_report(y_true, y_pred))

# Plot confusion matrix to see where the model is 'confused'
cm = confusion_matrix(y_true, y_pred)
```
