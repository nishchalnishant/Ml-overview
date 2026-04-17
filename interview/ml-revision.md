# Quick Revision

This is the last-minute cheat sheet.

Not the full theory.
The stuff you want in your head when the interviewer says:

> "Let's move quickly."

---

## 1. Regression

- linear regression predicts continuous values
- MAE is robust to outliers
- MSE / RMSE punish large misses more
- `R^2` tells you variance explained, but not whether the model is actually useful in production

---

## 2. Classification

- logistic regression is classification, despite the name
- precision = of predicted positives, how many were right?
- recall = of actual positives, how many did we catch?
- F1 balances precision and recall
- ROC-AUC is useful, PR-AUC is often more honest for imbalanced problems

---

## 3. Trees and Ensembles

- decision tree = greedy splits
- random forest = bagging, reduces variance
- XGBoost / GBM = boosting, reduces bias
- bagging = parallel committee
- boosting = sequential correction

---

## 4. Neural Nets

- ReLU is the default hidden activation
- sigmoid for binary outputs
- softmax for multiclass outputs
- vanishing gradients -> ReLU, residuals, normalization help
- exploding gradients -> clipping helps

---

## 5. Unsupervised Learning

- K-Means needs `k`
- PCA finds directions of maximum variance
- clustering quality depends heavily on representation and scaling

---

## 6. Core Tradeoffs

- bias vs variance
- precision vs recall
- latency vs accuracy
- exploration vs exploitation
- interpretability vs complexity

---

## 7. MLOps Quick Recall

- training is build
- evaluation is quality gate
- deployment is release
- drift is post-release reality
- feature store helps reduce train-serve skew

---

## 8. Fast Compare

### L1 vs L2

- L1 -> sparsity
- L2 -> smoother shrinkage

### Generative vs Discriminative

- generative -> model data distribution
- discriminative -> model label boundary

### BERT vs GPT

- BERT -> understanding
- GPT -> generation

### BatchNorm vs LayerNorm

- BatchNorm -> batch-based, great in vision
- LayerNorm -> feature-based, great in Transformers

---

## Mini Pop Quiz

If fraud is 0.5% of data, should "99.5% accuracy" impress you?

No.

That metric can be completely useless there.
