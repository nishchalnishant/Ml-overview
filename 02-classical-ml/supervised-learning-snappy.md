---
module: Classical Ml
topic: Supervised Learning Snappy
subtopic: ""
status: unread
tags: [classicalml, ml, supervised-learning-snappy]
---
# Supervised Learning — 1-Page Cheat Sheet

## Algorithm Table

| Algorithm | Best for | Key hyperparameters | Watch out for |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Continuous target, interpretability, baseline | Regularization α (Ridge/Lasso) | Linearity + homoscedasticity assumptions; multicollinearity |
| **Logistic Regression** | Binary/multi-class, calibrated probs, strong baseline | C (inverse reg), solver, max_iter | Use log-loss not MSE; won't fit non-linear decision boundaries |
| **SVM** | High-dim, small-medium N, clear margin needed | C, kernel (rbf/poly/linear), γ | Normalize features; slow O(N²–N³) training; kernel choice matters |
| **Decision Tree** | Interpretable rules, mixed feature types, interactions | max_depth, min_samples_leaf, criterion | Overfits without pruning; unstable (small data → different tree) |
| **Random Forest** | Tabular data, robust out-of-box, parallel | n_estimators, max_features, max_depth | Slow predict at scale; harder to interpret than single tree |
| **Gradient Boosting** | Best tabular accuracy (XGBoost/LightGBM) | learning_rate, n_estimators, subsample, max_depth | Overfits if over-tuned; sequential so slower to train |

---

## Key Concepts

**Why log-loss for logistic regression?**
MSE on class probabilities is non-convex → optimization can get stuck. Cross-entropy is convex for logistic output and brutally punishes confident wrong predictions.

**Kernel trick (SVM):**
Only inner products needed → kernel K(x,y) computes similarity in high-D space without explicitly building φ(x). RBF kernel handles non-linear class boundaries.

**Bagging vs Boosting:**

| | Random Forest | Gradient Boosting |
| :--- | :--- | :--- |
| Trees trained | In parallel | Sequentially |
| Targets | Original labels | Residuals of previous model |
| Reduces | Variance | Bias |
| Overfit risk | Lower | Higher (needs tuning) |

---

## Evaluation Metrics

| Metric | Formula | Use when |
| :--- | :--- | :--- |
| **Accuracy** | (TP+TN)/N | Balanced classes only |
| **Precision** | TP/(TP+FP) | False alarms are costly |
| **Recall** | TP/(TP+FN) | Misses are costly (fraud, cancer) |
| **F1** | 2·P·R/(P+R) | Imbalanced classes, want balance |
| **ROC-AUC** | Area under TPR vs FPR | Ranking quality across thresholds |
| **MSE / RMSE** | mean((y−ŷ)²) | Regression; penalizes large errors |

**Precision vs Recall call:** Fraud detection → optimize recall (catch all fraud). Spam filter → optimize precision (don't block real emails). Always justify with cost matrix.

---

## Feature Importance (know both)
- **Impurity/Gini importance** — fast; can favour high-cardinality features
- **Permutation importance** — shuffle feature, measure score drop; more reliable

---

## When to reach for what

- CSV, unknown shape, 1 hour → logistic/linear baseline → gradient boosting with CV
- Text, fast inference → Naive Bayes or logistic with TF-IDF
- High-dim, small N, clear margin → SVM
- Interpretability required → single decision tree or logistic regression
- Best accuracy on tabular → XGBoost / LightGBM

## Flashcards

**Impurity/Gini importance?** #flashcard
fast; can favour high-cardinality features

**Permutation importance?** #flashcard
shuffle feature, measure score drop; more reliable

**CSV, unknown shape, 1 hour → logistic/linear baseline → gradient boosting with CV?** #flashcard
CSV, unknown shape, 1 hour → logistic/linear baseline → gradient boosting with CV

**Text, fast inference → Naive Bayes or logistic with TF-IDF?** #flashcard
Text, fast inference → Naive Bayes or logistic with TF-IDF

**High-dim, small N, clear margin → SVM?** #flashcard
High-dim, small N, clear margin → SVM

**Interpretability required → single decision tree or logistic regression?** #flashcard
Interpretability required → single decision tree or logistic regression

**Best accuracy on tabular → XGBoost / LightGBM?** #flashcard
Best accuracy on tabular → XGBoost / LightGBM
