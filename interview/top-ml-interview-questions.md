# Top ML Interview Questions

This is the fast question bank.

Not every answer needs a mini-essay.
But every answer should show:

- concept
- intuition
- tradeoff

That is the pattern.

---

## 1. What is Machine Learning?

Learning patterns from data instead of hard-coding every rule manually.

---

## 2. Supervised vs Unsupervised vs Reinforcement Learning

- supervised = labeled data
- unsupervised = structure without labels
- reinforcement = learn from reward over time

---

## 3. Bias-Variance Tradeoff

Simple models underfit.
Complex models overfit.
The goal is lowest generalization error, not lowest training error.

---

## 4. Overfitting

Model memorizes noise or quirks of training data and fails on new data.

Common fixes:

- regularization
- more data
- early stopping
- simpler model

---

## 5. Train / Validation / Test Split

- train = learn
- validation = tune
- test = final honest check

---

## 6. Hyperparameter vs Parameter

- parameter = learned from data
- hyperparameter = chosen before training

---

## 7. Feature Scaling

Important for:

- KNN
- SVM
- logistic regression
- neural nets

Usually not crucial for trees.

---

## 8. Logistic Regression

Linear classifier that models class probability through sigmoid / log-odds.

---

## 9. Precision vs Recall

- precision = how trustworthy positive predictions are
- recall = how much of the positive class you catch

---

## 10. ROC-AUC vs PR-AUC

- ROC-AUC = general separability
- PR-AUC = often better for rare positive classes

---

## 11. Bagging vs Boosting

- bagging = parallel averaging, reduce variance
- boosting = sequential correction, reduce bias

---

## 12. Random Forest vs XGBoost

- Random Forest = stable, robust baseline
- XGBoost = stronger when tuned well, more sensitive

---

## 13. Backpropagation

Efficient chain-rule-based gradient computation across layers.

---

## 14. BatchNorm

Stabilizes training and often allows higher learning rates.

---

## 15. Why ReLU?

Simple, fast, and easier to optimize than sigmoid/tanh in deep networks.

---

## 16. Transformer vs LSTM

Transformer:

- more parallelizable
- better long-range context at scale

LSTM:

- more sequential
- smaller-scale sequence modeling

---

## 17. Class Imbalance

Do not rely on accuracy.

Use:

- precision
- recall
- F1
- PR-AUC
- cost-aware thresholds

---

## 18. Model Drift

Model performance degrades because data distribution or input-target relationship changes over time.

---

## 19. Candidate Generation vs Ranking

Use retrieval to narrow millions to hundreds.
Then use a heavier ranker for precision.

That is a very common large-scale pattern.

---

## 20. Why Cross-Validate?

Because one split can lie to you.

Cross-validation gives a more stable estimate of performance.
