# Day 8-9: Supervised Learning (Ensembles & Kernels)

## Executive Summary
| Algorithm | Logic | Strength | Weakness |
|-----------|-------|----------|----------|
| **k-NN** | Proximity ($k$ neighbors) | Lazy learner, simple | High memory, slow inference |
| **SVM** | Wide-margin separation | Kernel trick, non-linear | Hard to tune, slow training |
| **Random Forest** | Bagging (Parallel Trees) | Low variance, robust | High memory, black box |
| **XGBoost/GBM** | Boosting (Sequential) | State-of-the-art accuracy | Prone to overfitting |

---

## 1. Support Vector Machines (SVM)
The goal is to find a hyperplane that maximizes the **margin** between classes.
- **Hard Margin**: Assumes linear separability.
- **Soft Margin**: Allows some misclassifications (controlled by $C$).
- **Kernel Trick**: Maps data to high-dim space where it *is* linearly separable.
  - **RBF Kernel**: $K(x, x') = \exp(-\gamma ||x - x'||^2)$

---

## 2. Ensemble Methods: The "Wisdom of the Crowd"

### Bagging (Bootstrap Aggregating)
- **Concept**: Train $M$ models on $M$ random subsets (with replacement).
- **Example**: **Random Forest**.
- **Impact**: Reduces **Variance** by averaging predictions.

### Boosting
- **Concept**: Train models sequentially. Model $i+1$ corrects the errors of model $i$.
- **Examples**: AdaBoost, Gradient Boosting, XGBoost.
- **Impact**: Reduces **Bias** and Variance.

---

## Interview Questions

**1. "What is the difference between Bagging and Boosting?"**
> Bagging focuses on reducing variance by parallelizing independent models (e.g., Random Forest). Boosting focuses on reducing bias by sequentially building models that learn from predecessor mistakes (e.g., XGBoost).

**2. "Why does Random Forest use random feature selection at each split?"**
> To **decorrelate** the trees. If all trees used the most important feature first, they would be highly correlated, and averaging them wouldn't reduce variance as effectively.

**3. "When would you prefer k-NN over a model like Logistic Regression?"**
> When the decision boundary is highly non-linear and you have enough memory to store the dataset. LR assumes a linear boundary, while k-NN is non-parametric.

---

## Code Comparison
```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Random Forest: Focus on 'n_estimators' and 'max_depth'
rf = RandomForestClassifier(n_estimators=100, max_depth=None)

# XGBoost: Focus on 'learning_rate' (eta) and 'early_stopping'
xgb = XGBClassifier(learning_rate=0.1, n_estimators=1000)
```
