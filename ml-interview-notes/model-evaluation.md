# Model Evaluation

---

# Q1: What are precision, recall, F1 score, and accuracy?

## 1. 🔹 Direct Answer
**Accuracy** = (TP+TN)/all. **Precision** = TP/(TP+FP) — trust in predicted positives. **Recall** = TP/(TP+FN) — coverage of actual positives. **F1** = harmonic mean of P and R — balances both when you need one number.

## 2. 🔹 Intuition
Precision: “of predicted positives, how many right?” Recall: “of real positives, how many caught?”

## 3. 🔹 Deep Dive
- **Harmonic** mean: punishes if either P or R is low.
- **Multiclass**: macro (unweighted mean per class), micro (global pool), weighted.

## 4. 🔹 Practical Perspective
Imbalanced data: **accuracy** misleading—use **PR-AUC**, **F1**, **cost-weighted** metrics.

## 5. 🔹 Code Snippet
```python
from sklearn.metrics import precision_recall_fscore_support
p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Which if FP costly? **A:** Precision—spam filter.

## 7. 🔹 Common Mistakes
Reporting accuracy on 99% negative class data.

## 8. 🔹 Comparison / Connections
ROC-AUC, PR-AUC, calibration.

## 9. 🔹 One-line Revision
Precision vs FP rate; recall vs FN rate; F1 balances; accuracy only when balanced and costs equal.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q2: What is the confusion matrix, and how do you interpret it?

## 1. 🔹 Direct Answer
A **confusion matrix** tabulates **predicted vs actual** for each class (binary: TP, TN, FP, FN). Shows **where** errors concentrate—**not** a single scalar.

## 2. 🔹 Intuition
Heatmap of “what we called A when truth was B.”

## 3. 🔹 Deep Dive
- **Normalize** by row (recall per class) or column (precision).
- **Multiclass**: k×k matrix.

## 4. 🔹 Practical Perspective
Use for **error analysis** (confused pairs), **threshold** tuning.

## 5. 🔹 Code Snippet
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Cost matrix? **A:** Weight FN vs FP differently—optimize expected cost.

## 7. 🔹 Common Mistakes
Only looking at accuracy without per-class breakdown.

## 8. 🔹 Comparison / Connections
Precision-recall tradeoff, ROC curve.

## 9. 🔹 One-line Revision
Confusion matrix decomposes errors into TP/TN/FP/FN (or multiclass cells)—foundation for P/R metrics.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q3: What are common evaluation metrics for Classification?

## 1. 🔹 Direct Answer
**Thresholded**: accuracy, precision, recall, F1, **MCC** (balanced). **Ranking**: **ROC-AUC**, **PR-AUC**, **log loss** (calibration). **Probabilistic**: **Brier score**, **ECE** for calibration. Choose per **imbalance** and **cost**.

## 2. 🔹 Intuition
Different metrics answer different questions—**ranking** vs **calibration** vs **hard** decisions.

## 3. 🔹 Deep Dive
- **Log loss** penalizes confident wrong predictions—sensitive to calibration.
- **MCC** informative for imbalanced binary.

## 4. 🔹 Practical Perspective
Report **multiple** metrics + **slice** by cohort.

## 5. 🔹 Code Snippet
```python
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
```

## 6. 🔹 Interview Follow-ups
1. **Q:** ROC vs PR? **A:** PR better for heavy imbalance—focus on positive class.

## 7. 🔹 Common Mistakes
Using ROC-AUC alone when positives are rare.

## 8. 🔹 Comparison / Connections
Fairness metrics, calibration curves.

## 9. 🔹 One-line Revision
Pick classification metrics matching business costs, imbalance, and whether you need calibrated probabilities.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q4: When would you use accuracy vs other metrics?

## 1. 🔹 Direct Answer
Use **accuracy** when classes are **balanced** and **FP/FN costs** are similar. Otherwise prefer **precision/recall/F1**, **MCC**, **ROC/PR-AUC**, or **cost-weighted** metrics.

## 2. 🔹 Intuition
Accuracy answers “overall right %” which hides **minority** failure.

## 3. 🔹 Deep Dive
With **0.99** negative rate, predicting always negative yields **99% accuracy**—useless.

## 4. 🔹 Practical Perspective
Always report **baseline** (majority class) alongside accuracy.

## 5. 🔹 Code Snippet
```text
baseline_acc = max(p, 1-p)  # single-class majority
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Balanced accuracy? **A:** `(TPR+TNR)/2`—better for imbalance.

## 7. 🔹 Common Mistakes
Optimizing accuracy for fraud detection.

## 8. 🔹 Comparison / Connections
Stratified sampling, threshold selection.

## 9. 🔹 One-line Revision
Accuracy is only safe when balanced and symmetric costs—otherwise use richer metrics.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q5: When would you use log loss vs accuracy?

## 1. 🔹 Direct Answer
**Log loss** (cross-entropy) scores **probabilities**—penalizes **confident wrong** predictions. Use when **calibration** matters (ranking, downstream decisions, cost-sensitive thresholds). **Accuracy** ignores confidence—only **argmax** correctness.

## 2. 🔹 Intuition
Log loss cares **how sure** you were wrong.

## 3. 🔹 Deep Dive
For well-calibrated models, lower log loss → better **probabilistic** predictions.

## 4. 🔹 Practical Perspective
If you only need **hard** label with fixed 0.5 threshold, accuracy may suffice—**but** log loss still better for **model comparison** during development.

## 5. 🔹 Code Snippet
```python
from sklearn.metrics import log_loss
log_loss(y_true, proba)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Log loss vs Brier? **A:** Both proper scoring rules; Brier for binary probability.

## 7. 🔹 Common Mistakes
Applying log loss to **non-probabilistic** scores without softmax.

## 8. 🔹 Comparison / Connections
Calibration, Platt scaling.

## 9. 🔹 One-line Revision
Log loss evaluates probability quality; accuracy evaluates only hard decisions—use log loss when calibration matters.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q6: What metrics would you use for a multi-class classification problem?

## 1. 🔹 Direct Answer
**Macro** F1 (unweighted per-class average), **weighted** F1 (by support), **micro** F1 (pool all), **accuracy**, **log loss**, **cohen’s kappa** for agreement. **Confusion matrix** essential. For imbalance: **macro** or **balanced** metrics.

## 2. 🔹 Intuition
Macro treats **small classes** equally; micro follows **frequency**.

## 3. 🔹 Deep Dive
- **Top-k accuracy** for large label spaces.
- **Hierarchical** metrics if labels are tree-structured.

## 4. 🔹 Practical Perspective
Report **per-class** recall to catch **neglected** classes.

## 5. 🔹 Code Snippet
```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, digits=3))
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Multilabel? **A:** Per-label metrics + sample average or macro.

## 7. 🔹 Common Mistakes
Single accuracy only—no per-class failure visibility.

## 8. 🔹 Comparison / Connections
Hierarchical classification, multi-task.

## 9. 🔹 One-line Revision
Multiclass: confusion matrix + macro vs weighted F1 depending on whether rare classes matter equally.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q7: How do you handle class imbalance in classification metrics?

## 1. 🔹 Direct Answer
Use **PR-AUC**, **F1**, **recall at precision**, **balanced accuracy**, **MCC**—not raw accuracy. **Threshold** tuning on **validation** to target precision/recall. **Cost-sensitive** loss.

## 2. 🔹 Intuition
Metrics must **reflect** minority class importance.

## 3. 🔹 Deep Dive
**ROC-AUC** can be **optimistic** under imbalance—**PR** curves focus on **positive** class performance.

## 4. 🔹 Practical Perspective
Align metric with **business** cost (e.g., fraud FN cost).

## 5. 🔹 Code Snippet
```python
from sklearn.metrics import average_precision_score
average_precision_score(y_true, scores)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Sampling + metrics? **A:** Still evaluate on **original** distribution test if possible.

## 7. 🔹 Common Mistakes
SMOTE on test set—invalid.

## 8. 🔹 Comparison / Connections
Resampling, class weights.

## 9. 🔹 One-line Revision
Pick imbalance-aware metrics (PR, F1, MCC) and tune thresholds—accuracy misleads.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q8: What is the ROC curve? What is AUC?

## 1. 🔹 Direct Answer
**ROC** plots **TPR vs FPR** as **classification threshold** varies. **AUC** (AUROC) is **area under ROC**—probability a random positive scores higher than random negative (**ranking** quality). **Invariant** to class balance in a specific sense but **not** always best for imbalance.

## 2. 🔹 Intuition
Shows **tradeoff** between catching positives and false alarms.

## 3. 🔹 Deep Dive
- **Mann-Whitney** relation to ranking.
- **Degenerate** if all scores equal.

## 4. 🔹 Practical Perspective
Compare models **threshold-free**; **choose** operating point on curve for deployment.

## 5. 🔹 Code Snippet
```python
from sklearn.metrics import roc_curve, auc
fpr, tpr, thr = roc_curve(y_true, scores)
roc_auc = auc(fpr, tpr)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** ROC vs PR? **A:** PR when positives rare—ROC can look optimistic.

## 7. 🔹 Common Mistakes
Using ROC-AUC alone without checking **PR** or **calibration**.

## 8. 🔹 Comparison / Connections
PR-AUC, lift charts.

## 9. 🔹 One-line Revision
ROC-AUC measures ranking separability of classes across thresholds—pair with PR when imbalance is severe.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q9: How do you handle imbalanced datasets?

## 1. 🔹 Direct Answer
**Data**: collect more minority, **resample** (undersample, oversample, SMOTE), **class weights**. **Model**: **cost-sensitive** loss, **threshold** tuning. **Metrics**: PR/F1. **Validation**: **stratified** splits.

## 2. 🔹 Intuition
Balance the **objective** with reality—minority class often matters more.

## 3. 🔹 Deep Dive
- **SMOTE** can blur boundaries—**clean** noisy data first.
- **Calibration** after reweighting.

## 4. 🔹 Practical Perspective
**Business** cost: sometimes **FNR** is worse than FPR.

## 5. 🔹 Code Snippet
```python
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Anomaly detection? **A:** Reframe as rare class or one-class SVM.

## 7. 🔹 Common Mistakes
Optimizing accuracy; **leakage** from SMOTE across train/val.

## 8. 🔹 Comparison / Connections
Focal loss, ensemble balancing.

## 9. 🔹 One-line Revision
Combine sampling/weights, right metrics, and threshold tuning—align with costs.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q10: What are common evaluation metrics for Regression?

## 1. 🔹 Direct Answer
**MAE** (L1), **MSE/RMSE** (L2), **R²**, **MAPE** (scale-dependent), **Huber** loss. **Quantile** loss for intervals. **Log** targets for heavy-tailed outcomes.

## 2. 🔹 Intuition
L2 punishes **large** errors more; L1 is **robust**; MAPE punishes **small** denominators.

## 3. 🔹 Deep Dive
**R²** = 1 − SS_res/SS_tot—relative to mean baseline.

## 4. 🔹 Practical Perspective
Report **error** on **business units** ($, ms) for interpretability.

## 5. 🔹 Code Snippet
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

## 6. 🔹 Interview Follow-ups
1. **Q:** RMSE vs MAE? **A:** RMSE in same units as y but squares large errors.

## 7. 🔹 Common Mistakes
MAPE with zero targets—undefined.

## 8. 🔹 Comparison / Connections
Gini for ranking, calibration in probabilistic regression.

## 9. 🔹 One-line Revision
Pick regression metric by outlier sensitivity (MAE vs RMSE) and scale (avoid MAPE pitfalls).

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q11: What's the difference between MAE, MSE, and RMSE?

## 1. 🔹 Direct Answer
**MAE** = mean |y−ŷ|. **MSE** = mean (y−ŷ)². **RMSE** = √MSE. MAE **robust**; MSE/RMSE **penalize** large errors more; RMSE **same units** as y like MAE.

## 2. 🔹 Intuition
MSE is **mean of squares**—outliers dominate.

## 3. 🔹 Deep Dive
Differentiability: MSE smooth at 0; MAE has kink.

## 4. 🔹 Practical Perspective
**RMSE** common in competitions; **MAE** for reporting typical error.

## 5. 🔹 Code Snippet
```python
import numpy as np
mae = np.mean(np.abs(y - yhat))
rmse = np.sqrt(np.mean((y - yhat)**2))
```

## 6. 🔹 Interview Follow-ups
1. **Q:** When MAE? **A:** Heavy-tailed noise, outliers.

## 7. 🔹 Common Mistakes
Comparing RMSE across differently scaled targets without normalization.

## 8. 🔹 Comparison / Connections
Huber loss, quantile loss.

## 9. 🔹 One-line Revision
MAE is L1 average error; MSE is L2; RMSE is sqrt(MSE) for interpretable units with outlier emphasis.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q12: How do you choose the right evaluation metric for a given problem?

## 1. 🔹 Direct Answer
Start from **business** objective and **costs** of errors (FP/FN asymmetry). Consider **class balance**, need for **calibrated probabilities**, **latency**, and **interpretability**. **Prototype** with simple metrics, then **align** with stakeholders.

## 2. 🔹 Intuition
The “right” metric is the one **decisions** will be judged on.

## 3. 🔹 Deep Dive
- **Ranking** problems: NDCG, MAP.
- **Segmented** fairness: metrics per group.

## 4. 🔹 Practical Perspective
Document **why** metric chosen—**revisit** after launch.

## 5. 🔹 Code Snippet
```text
metric = f(business_cost_matrix, distribution, calibration_needs)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Multiple metrics? **A:** Pareto frontier—don’t collapse to one without trade-off discussion.

## 7. 🔹 Common Mistakes
Copying Kaggle metric without matching product.

## 8. 🔹 Comparison / Connections
Decision theory, utility functions.

## 9. 🔹 One-line Revision
Derive metrics from costs, data distribution, and whether probabilities need to be meaningful.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q13: How do you compare the performance of different models?

## 1. 🔹 Direct Answer
**Same** validation protocol (**CV**), **same** splits, **multiple** metrics, **statistical** tests or **bootstrap** CIs for significance. **Ablation**—change one variable. **Complexity** vs gain (latency, size).

## 2. 🔹 Intuition
A 0.5% AUC lift may be **noise**—need confidence intervals.

## 3. 🔹 Deep Dive
**McNemar** for paired classification errors; **paired bootstrap** for general metrics.

## 4. 🔹 Practical Perspective
**Production** criteria: SLO, cost, maintainability—not just offline score.

## 5. 🔹 Code Snippet
```python
from scipy.stats import bootstrap
def stat(x): return np.mean(x)
ci = bootstrap((diffs,), stat, paired=True, confidence_level=0.95)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Multiple testing? **A:** Bonferroni or pre-register primary metric.

## 7. 🔹 Common Mistakes
Cherry-picking best fold without variance report.

## 8. 🔹 Comparison / Connections
A/B testing, experiment design.

## 9. 🔹 One-line Revision
Compare models with identical splits, uncertainty estimates, and deployment constraints—not point scores alone.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q14: Explain cross-validation and its importance.

## 1. 🔹 Direct Answer
**k-fold CV** splits data into **k** folds; train on **k−1**, validate on **1**, rotate. **Estimates** generalization with **lower variance** than single split—**data-efficient** hyperparameter tuning.

## 2. 🔹 Intuition
Every point gets to be validation—reduces **lucky split** bias.

## 3. 🔹 Deep Dive
- **Stratified** for classification.
- **Time series**: use **forward chaining**, not shuffle.

## 4. 🔹 Practical Perspective
**Nested** CV for unbiased performance with **inner** HPO.

## 5. 🔹 Code Snippet
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Leave-one-out? **A:** Low bias, high variance, expensive—rarely for large n.

## 7. 🔹 Common Mistakes
**Leakage** in preprocessing outside CV.

## 8. 🔹 Comparison / Connections
Bootstrap, holdout.

## 9. 🔹 One-line Revision
Cross-validation averages performance over multiple splits—use stratified/grouped/time-aware variants correctly.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q15: What is Hyperparameter Tuning?

## 1. 🔹 Direct Answer
**Hyperparameters** are settings **outside** training loop (LR, depth, k). **Tuning** searches for values **maximizing** validation metric—**distinct** from **parameters** learned by optimization (weights).

## 2. 🔹 Intuition
Different **architecture** and **optimization** regimes need different **knobs**.

## 3. 🔹 Deep Dive
Methods: **grid**, **random**, **Bayesian** optimization, **early stopping** as implicit HPO.

## 4. 🔹 Practical Perspective
**Budget**—diminishing returns; **default** strong baselines first.

## 5. 🔹 Code Snippet
```python
from sklearn.model_selection import RandomizedSearchCV
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs architecture search? **A:** NAS expands search to structure—more expensive.

## 7. 🔹 Common Mistakes
Tuning on **test** set.

## 8. 🔹 Comparison / Connections
AutoML, regularization path.

## 9. 🔹 One-line Revision
Hyperparameter tuning optimizes validation performance over settings not learned by gradient descent—use proper CV.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q16: How do you evaluate unsupervised learning models?

## 1. 🔹 Direct Answer
**Internal** metrics (cluster cohesion/separation: **silhouette**, **Davies-Bouldin**), **reconstruction** error (autoencoders), **likelihood** (if probabilistic). **External** if labels exist (**ARI**, **NMI**). **Downstream** task performance (retrieval, clustering as feature). **Human** evaluation for qualitative goals.

## 2. 🔹 Intuition
Without labels, “ground truth” is **implicit**—use proxies + **use case**.

## 3. 🔹 Deep Dive
Silhouette can be **misleading** for non-convex clusters—**domain** plots.

## 4. 🔹 Practical Perspective
**Stability** across runs/seed—important for k-means.

## 5. 🔹 Code Snippet
```python
from sklearn.metrics import silhouette_score
silhouette_score(X, labels)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Elbow method? **A:** Heuristic for k—subjective.

## 7. 🔹 Common Mistakes
Optimizing silhouette without checking **business** usefulness.

## 8. 🔹 Comparison / Connections
Semi-supervised evaluation, UMAP visualization.

## 9. 🔹 One-line Revision
Unsupervised eval mixes internal cluster metrics, external if labels exist, and downstream usability.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q17: How do you evaluate a clustering algorithm?

## 1. 🔹 Direct Answer
**Internal**: **silhouette**, **Calinski-Harabasz**, **Davies-Bouldin** (compact/separated). **External**: **ARI**, **NMI**, **V-measure** if true labels. **Stability**: **consensus** across subsamples. **Qualitative**: visualization.

## 2. 🔹 Intuition
Internal metrics assume **spherical** clusters—may disagree with human intuition.

## 3. 🔹 Deep Dive
**Rand index** adjusted for chance = **ARI**.

## 4. 🔹 Practical Perspective
**Business**: cluster **actionability** (campaigns), not just scores.

## 5. 🔹 Code Snippet
```python
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(y_true, y_pred_labels)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** k-means assumptions? **A:** Spherical equal-variance—use GMM/spectral if violated.

## 7. 🔹 Common Mistakes
Using **label** metrics when labels are **not** clustering goal.

## 8. 🔹 Comparison / Connections
Topic modeling eval (coherence), hierarchical clustering.

## 9. 🔹 One-line Revision
Use internal metrics + stability + ARI/NMI when labels exist—validate with domain use cases.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q18: What metrics would you use for a recommendation system?

## 1. 🔹 Direct Answer
**Ranking**: **Precision@k**, **Recall@k**, **MAP**, **NDCG** (graded relevance), **MRR** for single correct item. **Beyond accuracy**: **coverage**, **diversity**, **novelty**, **serendipity**. **Online**: **CTR**, **session time**, **long-term** engagement.

## 2. 🔹 Intuition
Users care about **top of list**—not global accuracy.

## 3. 🔹 Deep Dive
**Position bias** in logs—**inverse propensity** scoring for unbiased evaluation.

## 4. 🔹 Practical Perspective
**A/B** test for **business** KPIs; **offline** metrics approximate.

## 5. 🔹 Code Snippet
```python
def precision_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & relevant) / k
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Popularity bias? **A:** Metrics on **long-tail** items; debias training.

## 7. 🔹 Common Mistakes
High accuracy on **popular** items only—**filter bubbles**.

## 8. 🔹 Comparison / Connections
Learning to rank, multi-objective optimization.

## 9. 🔹 One-line Revision
Recsys metrics are ranking-focused (P@k, NDCG) plus diversity/coverage—validate online.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q19: What is A/B testing in the context of ML?

## 1. 🔹 Direct Answer
**Randomized** experiment: **users** split between **control** (baseline model) and **treatment** (new model); compare **KPIs** (CTR, revenue) with **statistical** tests. **Guards** against **confounding** and **selection bias** vs offline metrics.

## 2. 🔹 Intuition
Only way to measure **real** impact with **interference** and **behavior** in the loop.

## 3. 🔹 Deep Dive
- **Power** analysis for sample size; **SRM** checks (split ratio).
- **Network** effects in social systems—**cluster** randomization.

## 4. 🔹 Practical Perspective
**Canary** rollout, **gradual** traffic; **guardrails** (latency, error rate).

## 5. 🔹 Code Snippet
```text
two-proportion z-test or bootstrap for CTR delta; CUPED for variance reduction
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Interference? **A:** Users interact—SUTVA violations; cluster randomized.

## 7. 🔹 Common Mistakes
Peeking at results early without **sequential** testing correction.

## 8. 🔹 Comparison / Connections
Quasi-experiments, causal inference.

## 9. 🔹 One-line Revision
A/B testing compares models in production with randomization and proper stats—gold standard for causal impact.

## 10. 🔹 Difficulty Tag
🟣 Hard

---
