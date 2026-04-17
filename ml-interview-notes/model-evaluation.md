# Model Evaluation & Metrics

This hub focuses on the "why" behind evaluation choices, connecting statistical metrics to real-world product impact. Senior candidates are expected to understand not just the definitions, but the tradeoffs and failure modes of each metric.

---

# 1. 🔹 Classification Fundamentals

## Q1: Precision vs. Recall - How do you choose the operating point?

### 🔹 Direct Answer
- **Precision (Reliability):** "When I predict positive, how often am I right?" Fix for False Positives.
- **Recall (Capture):** "Of all actual positives, how many did I find?" Fix for False Negatives.
The optimal threshold is chosen by minimizing the business cost of error. This is done by analyzing the **Precision-Recall Curve** or **F1-Score** vs. Threshold plots.

### 🔹 Intuition: The Courtroom Analogy
- **Precision (Beyond Reasonable Doubt):** You only convict if you are 100% sure. You avoid convicting innocent people (Low FP), but many guilty people go free (Low Recall).
- **Recall (Cast a Wide Net):** You arrest anyone even slightly suspicious. You catch every criminal (Low FN), but many innocent people are inconvenienced (Low Precision).

### 🔹 Deep Dive: The Precision-Recall Tradeoff
Mathematically, they are competing objectives. Lowering the classification threshold (e.g., from 0.5 to 0.1) will ALWAYS increase Recall and ALMOST ALWAYS decrease Precision. The best "unbiased" balance is the **F1-Score** (Harmonic Mean), which penalizes extreme values in either metric.

---

# 2. 🔹 Ranking & Probability

## Q2: ROC-AUC vs. PR-AUC - When is ROC misleading?

### 🔹 Comparison Table

| Metric | Focus | Use Case |
| :--- | :--- | :--- |
| **ROC-AUC** | TPR vs. FPR (All thresholds) | Balanced classes; measuring general model "separability." |
| **PR-AUC** | Precision vs. Recall | **Imbalanced classes** (e.g., 99% Negative, 1% Positive). |

### 🔹 Deep Dive: The ROC Pitfall
In highly imbalanced datasets (e.g., Fraud detection), the False Positive Rate (FPR = FP / (FP + TN)) grows very slowly because the number of True Negatives (TN) is massive. This can make the ROC curve look nearly perfect ($AUC \approx 0.99$) even if the model's actual Precision is abysmal (e.g., 0.01). **PR-AUC is the gold standard for imbalanced classification.**

---

# 3. 🔹 Multiclass Evaluation

## Q3: Macro vs. Micro vs. Weighted F1 - When to use what?

### 🔹 Direct Answer
1. **Micro-F1:** Calculates the metric globally (aggregating total TP, FP, FN). It is dominated by the most frequent class. Use this to measure **overall accuracy**.
2. **Macro-F1:** Calculates the F1 for each class independently and takes the unweighted mean. Use this if you care about **rare classes** (it treats the "Small Class" as equal to the "Huge Class").
3. **Weighted-F1:** Like Macro, but weights each class's score by its frequency (Support).

### 🔹 Implementation Note
In a 3-class problem (Cat, Dog, Bird) where "Bird" only appears 1% of the time:
- A model that fails on every single Bird might still have a high **Micro-F1**.
- That same model will have a poor **Macro-F1**, highlighting the failure.

---

# 4. 🔹 Regression Metrics (Numeric Output)

## Q4: MSE vs. MAE vs. RMSE - Which one should I optimize?

### 🔹 Comparison Table

| Metric | Calculation | Advantage | Weakness |
| :--- | :--- | :--- | :--- |
| **MSE** | $\frac{1}{n} \sum (y - \hat{y})^2$ | Differentiable (easy for GD). | Heavily penalizes outliers. |
| **RMSE** | $\sqrt{MSE}$ | Result is in the same units as $y$. | Still sensitive to outliers. |
| **MAE** | $\frac{1}{n} \sum |y - \hat{y}|$ | **Robust to outliers**. | Not differentiable at 0. |

### 🔹 Deep Dive: R-Squared ($R^2$) vs. Adjusted $R^2$
- **$R^2$:** Measures the % of variance explained by the model compared to a baseline (the mean). **Pitfall:** Adding any feature (even noise) will monotonically increase $R^2$.
- **Adjusted $R^2$:** Penalizes the model for adding features that don't add predictive value. This is the **correct metric** for feature selection in Linear Regression.

---

# 5. 🔹 Calibration & Reliability

## Q5: Why does a 90% accurate model still need Calibration?

### 🔹 Direct Answer
Accuracy only tells you if the *label* is correct. **Calibration** tells you if the *probability* is honest. If a model predicts "Loan Default" with 80% probability, then out of 100 people with that score, exactly 80 should actually default.

### 🔹 Intuition: The Weather Forecaster
If a forecaster says "80% chance of rain," and you take an umbrella 10 times, but it only rains 3 times, the forecaster is accurately predicting "Rain" (if the label is >0.5), but they are **poorly calibrated**.

### 🔹 Diagnostic: Reliability Diagram (Calibration Curve)
- X-axis: Mean predicted probability.
- Y-axis: Actual fraction of positives.
- **Perfect Calibration:** A perfectly diagonal 45-degree line.

---

# 6. 🔹 Practical Perspective: The Confusion Matrix

### **Visual Mastery**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# High-yield patterns to look for:
# 1. Diagonal: The "True Hits." Larger numbers are better.
# 2. Clusters of errors: E.g., model consistently confuses "Car" with "Truck."
#    - Solution: Better features or target-specific data augmentation.
# 3. Asymmetric errors: E.g., FN >> FP.
#    - Solution: Adjust threshold or reweight loss.
```

---

## 🔹 Difficulty Tag: 🟡 Medium
