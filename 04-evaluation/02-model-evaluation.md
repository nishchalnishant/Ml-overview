---
module: Evaluation
topic: Ml
subtopic: Model Evaluation
status: unread
tags: [interviewprep, ml, ml-model-evaluation, interview-framing]
---
# Model Evaluation

---

## 1. Why Evaluation Is Not Optional

**What the interviewer is testing**: whether you understand evaluation as a decision-making framework that encodes assumptions about what "good" means — not just a reporting step at the end of training.

**The reasoning structure**: a model is not "done" when training converges. Training convergence means the optimizer stopped improving on training data. Whether the model generalizes to new data is an entirely separate question, answered only by evaluation on held-out examples the model has never seen. These are not the same question. A model can perfectly memorize training data (training loss zero) and be completely useless on new data.

The deeper issue: even correct evaluation on a clean held-out test set does not guarantee the model will work in production. The test set was drawn from a historical distribution. Production data reflects the future — it carries drift, adversarial inputs, edge cases, and population shifts that were not present at training time. Evaluation tells you how well the model performs on the specific distribution your test set came from, which is the best proxy you have for production. But it is a proxy, not reality.

Every evaluation decision — which metric, which data split, which threshold — encodes an assumption about what "good" means. Accuracy encodes the assumption that all errors have equal cost. ROC-AUC encodes the assumption that you care about ranking quality independent of class frequency. PR-AUC encodes the assumption that performance on the rare positive class is what matters. Making these assumptions explicit is the difference between rigorous evaluation and cargo-cult model validation.

**The pattern in action**: "My model achieves 97% accuracy on a medical diagnosis task. Before calling this a success, I ask: what does the class distribution look like? If 97% of patients don't have the condition, a model that predicts 'negative' for every patient achieves 97% accuracy. Accuracy is not the right metric. I need to evaluate sensitivity (recall on the positive class) and specificity, and I need to know what the consequences of false negatives are compared to false positives."

**Common traps**:
- Reporting training loss as model quality. The only interesting question is held-out performance. Training loss answers a different question entirely: "did the optimization converge?"
- Evaluating against the test set repeatedly during development. Each time you use the test set to inform a modeling decision, you are implicitly using it as a validation set. After 20 evaluation-adjust cycles, you have effectively trained on the test set through your choices. The final reported number is optimistic.

---

## 2. The Confusion Matrix First

**What the interviewer is testing**: whether you start from the full picture of errors before computing any summary metric, and whether you understand why summary metrics can obscure critically different failure modes.

**The reasoning structure**: every binary classifier makes four types of decisions: true positives (correctly predicted positive), true negatives (correctly predicted negative), false positives (predicted positive, actually negative — type I error), and false negatives (predicted negative, actually positive — type II error). A single number like accuracy aggregates all four into one. Two models with identical accuracy can have completely different error structures — one catching most positives but generating many false alarms, the other avoiding false alarms entirely but missing most actual positives.

The only way to know which failure mode your model has is to look at the full confusion matrix first, then compute the metrics that answer the specific questions your problem requires.

|  | Predicted Positive | Predicted Negative |
| :--- | :--- | :--- |
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

From this table, every classification metric can be derived. Precision is TP/(TP+FP) — of what I called positive, how many were right? Recall is TP/(TP+FN) — of all actual positives, how many did I catch? Specificity is TN/(TN+FP) — of all actual negatives, how many did I correctly ignore? Accuracy is (TP+TN)/(TP+TN+FP+FN) — overall correctness.

**The pattern in action**: "Two models each achieve 85% accuracy on a fraud dataset with 10% positive rate. Model A: TP=900, FP=500, FN=100, TN=8500. Model B: TP=400, FP=50, FN=600, TN=8950. Model A has high recall (90%) but low precision (64%) — it catches most fraud but overwhelms investigators with false alarms, 500 per day. Model B has high precision (89%) but low recall (40%) — investigators see mostly real fraud cases, but 600 fraudulent transactions per day slip through. These are completely different operational realities. The 85% accuracy headline is identical and meaningless for choosing between them."

**Common traps**:
- Evaluating a model only on aggregate metrics without investigating which examples it gets wrong. Understanding the failure pattern is more actionable than knowing the average failure rate. A model that fails on a specific demographic, or on a specific data type, needs targeted fixes, not more general regularization.

---

## 3. Accuracy, Precision, Recall, F1

**What the interviewer is testing**: whether you can connect each metric to a specific business cost structure, not just recall definitions. The interviewer wants to see you derive the right metric from the problem, not memorize a lookup table.

**The reasoning structure**: each metric answers a different question, and the question that matters depends on the asymmetry between error costs.

Accuracy answers: "overall, what fraction of predictions were correct?" It treats false positives and false negatives as equally costly and treats all classes as equally important. Both assumptions fail whenever classes are imbalanced or errors have different consequences.

Precision answers: "of all the times the model predicted positive, what fraction were actually positive?" It measures the cost of false alarms. A low-precision model wastes the time of whoever acts on its positive predictions. Optimizing precision means being conservative — only flag things you are confident about.

Recall (sensitivity) answers: "of all actual positives in the data, what fraction did the model find?" It measures the cost of missed positives. A low-recall model lets dangerous cases slip through. Optimizing recall means being aggressive — flag anything that might be positive.

F1 is the harmonic mean of precision and recall. It punishes extreme imbalance: a model with 100% precision and 0% recall (predicts nothing positive) gets F1 = 0. A model with 0% precision and 100% recall (predicts everything positive) also gets F1 = 0. F1 is only appropriate when you believe precision and recall have equal cost. When they don't, use $F_\beta$.

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}, \quad F_\beta = \frac{(1+\beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

$\beta > 1$ weights recall more heavily; $\beta < 1$ weights precision more heavily.

**The pattern in action**: Three domains with different error asymmetry:

Spam filter: false positives (blocking real email) are worse than false negatives (letting spam through). A user losing an important email is catastrophic; seeing some spam is tolerable. Optimize precision. Set $\beta < 1$ or use precision alone.

Cancer screening: false negatives (missing cancer) are far worse than false positives (an unnecessary follow-up test). A missed diagnosis is irreversible; a false alarm leads to a benign biopsy. Optimize recall aggressively. Lower the threshold so the model flags anything remotely suspicious. Use $F_2$ or recall alone.

Loan approval: both false positives (approving bad loans, losing money) and false negatives (rejecting good customers, losing revenue) have financial cost. The relative magnitude comes from loan default loss vs. foregone revenue. If defaults cost 5x more than missed revenue, use $F_{0.5}$.

**Common traps**:
- Defaulting to F1 without justifying that precision and recall have equal cost. They rarely do. Use $F_\beta$ with a $\beta$ derived from the actual cost ratio.
- Reporting precision or recall in isolation. They are trivially gameable in opposite directions. A model that predicts nothing positive achieves perfect precision (undefined or 1.0 depending on convention) and 0% recall. A model that predicts everything positive achieves 100% recall and precision equal to the base rate. Always report both together.

---

## 4. ROC-AUC vs PR-AUC

**What the interviewer is testing**: whether you understand why ROC-AUC is misleading on imbalanced data, not just that "PR-AUC is better for imbalanced problems." The explanation requires understanding what each curve plots and why the denominator of FPR hides information.

**The reasoning structure**: ROC-AUC plots true positive rate (recall) against false positive rate ($FP/(FP+TN)$) as the classification threshold varies from 0 to 1. The AUC of this curve equals the probability that the model assigns a higher score to a randomly chosen positive example than to a randomly chosen negative example. This interpretation makes ROC-AUC a ranking quality metric — it measures how well the model separates positives from negatives, independent of what threshold you choose.

The problem with ROC-AUC on imbalanced data is structural: the false positive rate is normalized by the total number of true negatives. When negatives are abundant (say 99,900 true negatives and 100 true positives), generating 1,000 false positives produces an FPR of only 1% — which looks excellent in the ROC curve. But those 1,000 false positives against 100 true positives means precision is 100/(100+1000) = 9%. For every 11 flagged cases, 10 are wrong. ROC-AUC reported this model as nearly perfect; operationally it is barely useful. ROC-AUC does not lie, but on imbalanced data it asks the wrong question — it tells you about the ratio of false positives to true negatives, when you need to know the ratio of false positives to true positives (which is what precision measures).

PR-AUC plots precision against recall across all thresholds. Both quantities have true positives in the denominator of their complement, making PR-AUC directly sensitive to the quality of positive predictions. When positives are rare, a model needs to be precise — every false alarm has a high cost relative to true detections. PR-AUC captures this.

**The pattern in action**: "My fraud model achieves ROC-AUC = 0.95 on a dataset where 0.1% of transactions are fraudulent. This sounds excellent. I also compute PR-AUC and get 0.23. When I set the threshold to achieve 60% recall, precision is only 5% — for every 20 alerts generated, 19 are false alarms. My operations team can investigate at most 200 alerts per day. At this precision, 190 of those investigations are wasted effort and 60% of real fraud still slips through. ROC-AUC reported 0.95; the model is operationally near-useless. PR-AUC = 0.23 gives an honest picture."

**Common traps**:
- Using ROC-AUC as the primary metric for fraud detection, anomaly detection, medical diagnosis, or any other setting where the positive class is rare. In these settings, the denominator of FPR (number of true negatives) is so large that even many false positives contribute negligible FPR. Use PR-AUC.
- Not plotting the full curve. A single AUC number hides the shape of the tradeoff. A model might have high PR-AUC because it achieves very high precision at low recall, or very high recall at low precision. These imply completely different operating strategies. The curve reveals which.

---

## 5. Calibration

**What the interviewer is testing**: whether you understand that ranking quality (AUC) and probability quality (calibration) are independent properties, and whether you can identify when calibration matters operationally.

**The reasoning structure**: a model is well-calibrated if its probability outputs mean what they say: when the model predicts "70% probability of fraud," approximately 70% of those predictions should be genuine fraud. A reliability diagram groups predictions into bins (0–10%, 10–20%, ..., 90–100%), plots the fraction actually positive in each bin, and compares to the diagonal. A perfectly calibrated model lies exactly on the diagonal.

A model can have perfect ranking quality (AUC = 1.0 — every positive is ranked above every negative) and be arbitrarily miscalibrated. Imagine a model whose scores for all positives cluster around 0.52 and all negatives cluster around 0.48 — perfect separation, perfect AUC, but the probabilities are meaningless as probability estimates. They are just scores. AUC only tests whether positives rank above negatives; it does not test whether the absolute probability values are correct.

Calibration matters specifically when the probability output is consumed directly as a probability estimate in a downstream decision. A credit model that computes "probability of default" and feeds it into a pricing formula that charges $p \times L$ for a loan of value $L$ must have accurate probabilities — miscalibration directly causes mispricing. A model that just ranks applicants and approves the top $k$ does not require calibration.

Systematic miscalibration takes two forms: overconfidence (probabilities too extreme — model says 0.95 when the true rate is 0.70) and underconfidence (probabilities too moderate — model says 0.55 when the true rate is 0.80). Overconfidence is more common in deep networks without explicit calibration. Fixes: temperature scaling (learn a scalar $T$ and divide logits by $T$ before softmax), Platt scaling (fit a logistic regression on top of the model's scores), or isotonic regression.

**The pattern in action**: "My fraud model has AUC = 0.93 and is integrated into a risk engine that uses the score to set transaction limits: score > 0.8 blocks the transaction, 0.5–0.8 routes to manual review, < 0.5 approves automatically. I plot a reliability diagram and find that predictions in the 0.6–0.7 range actually correspond to 90% fraud rate — the model is severely underconfident for high-risk transactions. Cases that should be blocked are being routed to manual review, overwhelming investigators. After temperature scaling (T = 0.4, which sharpens probabilities), the reliability diagram is close to diagonal and the routing logic works correctly."

**Tools**: reliability diagrams, Brier score ($\frac{1}{n}\sum_i (\hat{p}_i - y_i)^2$), Expected Calibration Error (ECE = $\sum_b \frac{|B_b|}{n} |\text{acc}(B_b) - \text{conf}(B_b)|$), `sklearn.calibration.CalibratedClassifierCV`.

**Common traps**:
- Treating a well-ranked model as automatically well-calibrated. AUC and calibration are orthogonal. Many practitioners skip calibration assessment entirely because they only look at AUC.
- Applying calibration on the training set. Calibration must be fit on a held-out validation set (or via cross-validation), otherwise the calibration curve is optimistic and the model is overfit to training data probabilities.

---

## 6. Regression Metrics

**What the interviewer is testing**: whether you understand the error structure of each metric well enough to choose based on the cost of large versus small errors, not just to name the formulas.

**The reasoning structure**: the choice between regression metrics reduces to a single question: are large errors disproportionately worse than small ones?

MAE (mean absolute error) treats all errors symmetrically in absolute scale. A 10-unit error is exactly 10 times worse than a 1-unit error. This is correct when the cost of an error scales linearly with its magnitude — being off by 10 really is 10 times as bad as being off by 1.

MSE/RMSE squares errors before averaging, making large errors quadratically more costly. A 10-unit error contributes 100 to MSE; a 1-unit error contributes 1. RMSE is the square root of MSE, returning to the original scale. RMSE is correct when large errors are catastrophic — a $500K error in a $1M house price causes the buyer to overbid catastrophically, while a $5K error is negligible.

The practical effect: a model optimized for MAE will distribute errors evenly — it minimizes the total absolute deviation, accepting some large errors if that avoids many small ones. A model optimized for RMSE will aggressively minimize large errors, sometimes at the cost of more frequent small errors.

$R^2$ (coefficient of determination) measures the fraction of target variance explained by the model. $R^2 = 1$ is perfect fit. $R^2 = 0$ means the model is no better than always predicting the mean. $R^2 < 0$ means the model is worse than predicting the mean — possible if predictions are systematically far from actuals.

$$\text{MAE} = \frac{1}{n}\sum_i|y_i - \hat{y}_i|, \quad \text{RMSE} = \sqrt{\frac{1}{n}\sum_i(y_i - \hat{y}_i)^2}, \quad R^2 = 1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2}$$

**The pattern in action**: "I build a model to predict how long surgery will take. The operating room is scheduled in 30-minute blocks. Being off by 10 minutes is acceptable — the next surgery starts late but nothing is catastrophic. Being off by 90 minutes is a crisis — the patient is still in surgery, the next scheduled case is cancelled, and an emergency case cannot proceed. Large errors are qualitatively different from small ones. I use RMSE as the training loss and primary evaluation metric. My secondary metric is the 95th percentile absolute error — I want to explicitly track how bad the worst predictions are, not just the average."

**Common traps**:
- Using RMSE when the target has outliers you don't want to dominate the metric. Outliers squared in RMSE can be 10,000x more influential than normal errors. If those outliers are rare, unusual events where the model's accuracy is less critical, use MAE or Huber loss (which transitions from quadratic to linear beyond a threshold).
- Interpreting $R^2$ as a percentage of "variance explained" without checking whether predictions fall within the data distribution. A model that always predicts a value slightly outside the range of $y$ can produce negative $R^2$, which is surprising until you understand the formula.

---

## 7. Log Loss and Brier Score

**What the interviewer is testing**: whether you understand why probability-scoring rules are richer than accuracy, and specifically why confident wrong predictions should be penalized more than uncertain wrong predictions.

**The reasoning structure**: accuracy records only whether the most probable class was correct — it treats a prediction of 0.51 the same as a prediction of 0.99 when both are right. But a model that says 0.99 when it is wrong is making a much worse mistake than a model that says 0.55 when it is wrong. The first model is confident and incorrect; the second is uncertain and incorrect. Accuracy cannot distinguish these.

Log loss (cross-entropy loss) measures the quality of probability estimates directly. It penalizes confident wrong predictions severely through the logarithm: if the model predicts probability 0.99 for class 1 but the example is class 0, log loss = $-\log(1 - 0.99) = -\log(0.01) \approx 4.6$. If the model predicts 0.6 and is wrong, log loss = $-\log(0.4) \approx 0.92$. The penalty grows without bound as the model becomes more confident in the wrong direction.

The Brier score is mean squared probability error: $\frac{1}{n}\sum_i (\hat{p}_i - y_i)^2$. It also rewards calibrated confidence — the minimum is achieved by predicting the true probability — but uses squared rather than logarithmic penalty, making it more robust to occasional very confident wrong predictions. Brier score can be decomposed into calibration and resolution components, making it analytically useful.

$$\text{Log Loss} = -\frac{1}{n}\sum_i\left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$

$$\text{Brier Score} = \frac{1}{n}\sum_i(\hat{p}_i - y_i)^2$$

Both metrics are "proper scoring rules": a probabilistic forecaster maximizes their expected score (under either metric) only by reporting their true beliefs. This is the formal property that makes these metrics correct for evaluating probability outputs.

**The pattern in action**: "My classifier achieves 94% accuracy. A naive baseline that predicts the majority class achieves 93% accuracy. These are almost identical. But log loss for my model is 0.18 versus 0.31 for the baseline — my model's probability estimates are substantially better. For a downstream system that uses these probabilities to set prices, the difference between 0.18 and 0.31 log loss translates directly to revenue."

**Common traps**:
- Comparing log loss across models trained on different problems. Log loss is not scale-free. A binary classification problem with 50/50 class balance has a different log loss scale than one with 99/1 balance. Log loss comparisons are only valid within the same problem.
- Not using log loss when the model outputs probabilities that feed downstream decisions. If the model just ranks examples and a human picks the top K, accuracy or AUC is sufficient. If the probabilities are consumed numerically in a pricing formula or risk calculation, log loss or Brier score are the right metrics.

---

## 8. Cross-Validation — When and How

**What the interviewer is testing**: whether you understand cross-validation as a variance reduction technique for performance estimation, with specific constraints for temporal data and a critical leakage trap in preprocessing.

**The reasoning structure**: a single train/validation split gives one estimate of generalization performance. That estimate is noisy — the specific examples that ended up in the validation set affect the number substantially. K-fold cross-validation computes performance across $k$ non-overlapping validation sets (each 1/k of the data) and averages. With 5-fold CV, each example participates in exactly one validation fold and four training folds. The averaged estimate has lower variance than any single split.

The variance of the CV estimate decreases with more folds, but leave-one-out CV (n-fold) has a notorious high-variance problem for small datasets: each fold trains on an extremely similar dataset, so the folds are correlated, which inflates the variance of the average in ways that are difficult to correct.

For temporal data — time series, event prediction, any problem where time matters — standard K-fold is wrong. Shuffling time means a model can train on data from 2025 and be tested on data from 2020, using future information to predict the past. This is data leakage disguised as a train/test split. The correct approach is time-based evaluation: always train on past data, always evaluate on future data. Walk-forward validation (expanding window: train on all history up to time $t$, test on time $t+\delta$) or sliding window validation (fixed-size training window) are the correct structures.

**The preprocessing inside CV rule**: any preprocessing step that learns from data (StandardScaler fitting mean/variance, imputers learning fill values, target encoders learning class-conditional statistics) must be refit inside each fold on the training portion only. Fitting on the full dataset before cross-validating creates leakage: the scaler has seen the validation fold's statistics, so the model indirectly uses validation data during training.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Correct: preprocessing is inside the pipeline, refitted for each fold
pipeline = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
scores = cross_val_score(pipeline, X, y, cv=5)

# Wrong: scaler fitted on full X before CV, validation folds' statistics already embedded
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
scores = cross_val_score(LogisticRegression(), X_scaled, y, cv=5)  # leakage
```

**The pattern in action**: "I run 5-fold CV on my churn model: fold scores [82%, 83%, 81%, 84%, 82%], mean 82.4%, std 1.1%. The low variance tells me performance is stable across different data compositions — I can trust the 82% estimate. If scores were [90%, 70%, 85%, 60%, 92%], the high variance would indicate the model is sensitive to data composition and I'd need to investigate why: possibly class imbalance varying across folds, or a data quality issue in one region."

**Common traps**:
- Preprocessing outside the cross-validation loop. This is the most common CV mistake in practice. The scaler or imputer fitted on all data embeds the validation fold's statistics into the training representation.
- Using K-fold CV on time series data. Any shuffle of temporal data allows future information to appear in training. Time-based splits are mandatory.

---

## 9. Choosing Thresholds

**What the interviewer is testing**: whether you understand that the classification threshold is a business parameter (determined by cost structure and operational capacity), not a model parameter, and that 0.5 is almost never the right default.

**The reasoning structure**: a classification model produces a score in [0, 1]. The threshold converts the score to a binary decision. The model training process has nothing to do with where the threshold should be — training optimizes the score calibration, not the operating point. The threshold is set after training based on operational requirements.

At threshold 0.5, you are implicitly assuming that false positives and false negatives have equal cost, and that the class distribution in deployment matches whatever the model was trained on. Both assumptions are almost always wrong.

The right process: train the model, plot the precision-recall curve across all thresholds (or the ROC curve, or both), then choose the threshold that satisfies the operational constraint. Common constraints: maximum false positive rate, minimum recall, maximum number of alerts per day, target precision given operational review capacity.

The Youden index ($J = \text{Sensitivity} + \text{Specificity} - 1$) provides a threshold that maximizes the sum of sensitivity and specificity, but it only makes sense if these two quantities have equal value — which requires justification.

**The pattern in action**: "My fraud model can generate at most 500 alerts per day given the team's review capacity. I evaluate the model on 10,000 daily transactions. I plot precision and alert count against threshold. At threshold 0.5: 400 alerts, precision 85%, recall 60% — within capacity. At threshold 0.3: 800 alerts, precision 70%, recall 85% — exceeds capacity. At threshold 0.7: 180 alerts, precision 93%, recall 38% — under-utilizes the team and misses 62% of fraud. I set the threshold at exactly 0.48 to generate 500 alerts with the highest recall achievable at that count: recall = 63%, precision = 82%."

**Common traps**:
- Using 0.5 as the default threshold without checking whether it makes sense for the class distribution and cost structure. For any imbalanced problem (class rate < 30%), 0.5 is almost certainly too high — it under-predicts the rare class relative to the actual optimal threshold.
- Tuning the threshold on the test set. The threshold is a hyperparameter. Selecting it based on test set performance inflates the test metric. Choose the threshold on the validation set or through cross-validation.

---

## 10. Offline vs Online Evaluation

**What the interviewer is testing**: whether you understand that offline metrics are necessary but not sufficient for production decisions, and that they can fail to capture feedback loops, behavior changes, and distribution shift.

**The reasoning structure**: offline evaluation measures model quality on a static historical dataset. It answers: "how would this model have performed on past data?" This is a useful question but not the question you ultimately care about. The question you care about is: "how will this model affect user behavior and business outcomes in the future?"

Several mechanisms cause offline metrics to diverge from online impact:

Feedback loops: a recommendation model's historical data was generated by the previous model. The new model, trained on this data, optimizes for what the old model tended to show users — not for what users would like to see. Offline NDCG might improve while user satisfaction stays flat because the model keeps recommending the same distribution of content.

Behavior change: exposing users to the new model changes their behavior. A model that identifies high-intent users and shows them purchase opportunities might increase short-term conversion but train users to respond to fewer, higher-quality signals — reducing long-term engagement in ways the offline dataset cannot capture.

Distribution shift: future data has a different distribution than historical data. Even a perfect offline evaluation cannot predict how well the model handles new events, new vocabulary, or population drift.

The evaluation progression for production:
1. Offline evaluation: development and selection among candidates
2. Shadow mode: model runs in parallel with production, predictions logged but not served — compare distribution of scores and check for pathological outputs
3. Canary release: serve to 1–5% of traffic, monitor business metrics closely with rollback readiness
4. Full rollout with ongoing guardrail metrics

**The pattern in action**: "My new ranking model improves offline NDCG@10 by 8%. In shadow mode, I notice the new model recommends a much narrower set of items — it concentrates recommendations on a small number of popular items, which have high historical click rates. In the canary, click-through rate is up 5% but session duration drops 3% — users are clicking more but leaving faster, suggesting less satisfaction. The offline metric improved by optimizing for historical clicks, but the model produces less diverse and ultimately less engaging recommendations."

**Common traps**:
- Shipping based on offline evaluation alone. Even a large, clean offline improvement can hurt production metrics due to the mechanisms above. Shadow mode costs almost nothing and catches distribution anomalies before they reach users.
- Not defining guardrail metrics before deployment. "We will watch for negative effects" is not a plan. Before deployment, define: which metrics are guardrails (must not degrade by more than X%), what the rollback trigger is, and who has authority to roll back. This turns a vague monitoring intention into an operational procedure.

---

## 11. Metrics for Ranking and Recommendation

**What the interviewer is testing**: whether you understand that standard classification metrics are structurally wrong for ranking problems because they ignore position, and whether you can explain each ranking metric from the property it captures.

**The reasoning structure**: recommendation systems return a ranked list. Users consume primarily the top of the list. A relevant item at position 1 is enormously more valuable than the same item at position 10 or 50. Standard metrics (precision, recall, AUC) treat all positions equally — a true positive at rank 1 contributes identically to one at rank 50. This makes them wrong for ranking.

Ranking metrics weight relevance by position. The further down the list, the less credit a relevant item receives.

**Precision@K**: fraction of top-K items that are relevant. Direct operational metric: "what fraction of the recommendations I show users are things they actually want?" Easy to interpret. Does not reward finding all relevant items, only the top-K.

**Recall@K**: fraction of all relevant items that appear in the top K. Answers: "what fraction of relevant items did I surface?" Useful when the goal is coverage — making sure users can find relevant content if they scroll.

**NDCG (Normalized Discounted Cumulative Gain)**: DCG weights relevance by logarithm of position: $\text{DCG@K} = \sum_{k=1}^K \frac{r_k}{\log_2(k+1)}$, where $r_k$ is the relevance of the item at position $k$. NDCG normalizes by the ideal DCG (perfect ordering of relevant items): $\text{NDCG@K} = \text{DCG@K} / \text{IDCG@K}$. Range: [0, 1]. NDCG supports graded relevance (not just 0/1), making it suitable for problems where items have degrees of relevance.

**MAP (Mean Average Precision)**: for each query, compute Average Precision (AP = average of Precision@K values at each position where a relevant item appears). MAP averages AP across all queries. MAP captures both precision and ordering: it rewards models that place relevant items early.

**The pattern in action**: "My recommender returns 10 items. The most relevant item is at position 7, and 5 of 10 items are relevant. Precision@10 = 0.5 — fine. But NDCG@10 = 0.42 because the most relevant item is buried. If I only optimize Precision@10, I could achieve 0.5 by placing all 5 relevant items anywhere in the 10 positions — positions 6–10 included — which is very different from placing the best items first. NDCG forces me to put the best items at the top."

**Common traps**:
- Using AUC for ranking problems. AUC is a threshold-independent metric measuring the probability that a random positive ranks above a random negative. It does not account for position within the list. A model with AUC = 0.95 might still place the most relevant item at the bottom of the top-10 list.
- Not accounting for diversity alongside relevance. A list of 10 nearly identical items can score perfectly on any relevance metric while providing low value to users who want variety. Diversity metrics (coverage, intra-list diversity, novelty) need to be measured separately.

---

## 12. A/B Testing for ML Systems

**What the interviewer is testing**: whether you can design a statistically rigorous experiment that isolates the causal effect of a model change on business metrics, with particular attention to validity threats specific to ML systems.

**The reasoning structure**: you want to answer: "does this new model improve the outcome I care about?" The challenge is that many factors change simultaneously in production, and users self-select into behavior patterns. A/B testing controls for both by randomly assigning users to control (old model) or treatment (new model) and comparing the business metric across groups. Randomization ensures that the only systematic difference between groups is which model they saw.

The key design decisions:

**Primary metric**: the single metric used for the ship/no-ship decision. It should be the closest proxy to the business objective you have. Using multiple primary metrics introduces multiplicity — the probability of at least one false positive increases with each additional test.

**Guardrail metrics**: metrics that must not degrade beyond a threshold. They define rollback conditions independent of the primary metric result. Examples: session duration must not drop more than 2%, error rate must not increase.

**Minimum detectable effect (MDE)**: the smallest improvement worth detecting. Determines required sample size. An MDE of 0.1% on a $10M/week revenue product is $10K/week — probably worth detecting. An MDE of 5% for a feature used by 100 users per day is probably not worth running a rigorous test for.

**Sample size formula** (per group, two-sample t-test): $n = \frac{2(z_{\alpha/2} + z_\beta)^2 \sigma^2}{\delta^2}$, where $\delta$ is the MDE and $\sigma^2$ is the metric variance. Default: $\alpha = 0.05$ (5% false positive rate), $\beta = 0.20$ (80% power).

**Duration**: must cover at least one full weekly cycle to account for day-of-week effects. Long-running experiments accumulate novelty effects (users behave differently when something is new) and selection effects (early adopters are not representative).

**The pattern in action**: "I test a new recommendation model. Primary metric: 7-day revenue per user ($\sigma = \$5$). MDE: $\$0.10$ (1% relative improvement at baseline $\$10$/week). Required sample per variant: $n = 2(1.96 + 0.84)^2 \times 25 / 0.01 \approx 39,200$ users. With 50K unique daily users split 50/50, I need ~1.6 days of data — but I run for 14 days to cover two weekly cycles and avoid novelty effects. After 14 days: treatment +\$0.12 revenue/user (p = 0.02), session duration −0.5% (within 2% guardrail), cart abandonment unchanged. Ship."

**Common traps**:
- Stopping the experiment early when a significant positive result appears. This is peeking. The false positive rate for a test that is evaluated continuously and stopped whenever p < 0.05 is far above 5% — it can reach 20–30%. Fix by committing to a fixed duration and sample size before the experiment starts, or by using sequential testing (alpha spending) methods.
- Running multiple simultaneous A/B tests without interaction analysis. Two concurrent tests with independent 2% improvements may not combine to 4% if the user populations overlap and the effects interact. Test for interaction effects or ensure tests run on independent user segments.
- Network effects: in social or marketplace systems, assigning users randomly violates the Stable Unit Treatment Value Assumption (SUTVA) — users in the control group interact with treatment users and vice versa, contaminating both groups. Use cluster randomization (randomize by social cluster or geographic area) instead.
