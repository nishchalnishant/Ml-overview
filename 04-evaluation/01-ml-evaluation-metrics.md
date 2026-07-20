---
module: Evaluation
topic: Ml Evaluation Metrics
subtopic: ""
status: unread
tags: [classicalml, ml, ml-evaluation-metrics]
---
# ML Evaluation Metrics

---

## TL;DR

- **Classification**: Accuracy fails on imbalanced data. Use F1, PR-AUC, or MCC. ROC-AUC is threshold-independent but optimistic when positives are rare.
- **Calibration**: ECE/Brier measure whether probabilities are trustworthy, not just whether hard predictions are correct.
- **Regression**: MAE is robust to outliers; MSE penalizes large errors more. MAPE breaks near zero. R² alone is not enough.
- **Ranking**: NDCG handles graded relevance with position discount; MAP/MRR for binary relevance. Precision@k vs recall@k depends on whether the corpus is bounded.
- **Generative**: Human win rate is ground truth; automated metrics (BLEU, FID) are proxies that can be gamed.
- **Offline vs Online**: Offline metrics are proxies. Proxy–outcome correlation drifts. Always set guardrail metrics before launching experiments.
- **Statistical testing**: Always run a significance test. McNemar for classifiers; paired t-test or Wilcoxon for regression. Correct for multiple comparisons.

---

## 1. Classification Metrics

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**When it fails**: On a dataset with 99% negatives, a classifier that always predicts negative achieves 99% accuracy while being useless. Accuracy is only informative when classes are roughly balanced and misclassification costs are symmetric.

### Precision, Recall, F1

| Metric | Formula | Optimizes for |
|---|---|---|
| Precision | $TP / (TP + FP)$ | Minimizing false positives |
| Recall (Sensitivity) | $TP / (TP + FN)$ | Minimizing false negatives |
| F1 | $2 \cdot P \cdot R / (P + R)$ | Harmonic mean — penalizes extreme imbalance between P and R |
| $F_\beta$ | $(1+\beta^2) \cdot P \cdot R / (\beta^2 P + R)$ | $\beta > 1$ weights recall more; $\beta < 1$ weights precision more |

**Interview hook**: F1 is the harmonic mean, not the arithmetic mean. The harmonic mean penalizes extreme values — a model with precision=1.0 and recall=0.1 gets F1=0.18, not 0.55.

### PR-AUC vs ROC-AUC

**ROC curve**: TPR (recall) vs FPR (false positive rate) across thresholds. Area under it = probability that a random positive ranks above a random negative.

**PR curve**: Precision vs recall across thresholds.

| | ROC-AUC | PR-AUC |
|---|---|---|
| Imbalanced classes | Optimistic — TN inflation suppresses FPR | Pessimistic — directly exposes low precision |
| Interpretation | P(positive ranked above negative) | Average precision across recall levels |
| Baseline (random) | 0.5 regardless of class balance | Equal to prevalence ($p$) |
| Use when | Classes are roughly balanced, TN matters | Positive class is rare, FP is costly |

**Rule of thumb**: If positive prevalence < 10%, use PR-AUC. ROC-AUC can look impressive (0.95+) while PR-AUC is mediocre (0.30) on the same model.

### Matthews Correlation Coefficient (MCC)

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

- Range: $[-1, 1]$. $+1$ is perfect, $0$ is random, $-1$ is perfectly inverted.
- Uses all four cells of the confusion matrix — works well for any class balance.
- Harder to explain to stakeholders than F1 but more informative mathematically.
- A model predicting all negatives on a 99/1 split gets MCC = 0, not 0.99.

---

## 2. Calibration Metrics

Covered in depth in `calibration-and-uncertainty.md`. Summary for this file:

### ECE and MCE

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

$$\text{MCE} = \max_{m} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

ECE = average calibration gap weighted by bin size. MCE = worst-case bin. MCE matters more in safety-critical applications where you need the model to behave reliably even at extremes.

### Brier Score

$$\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2$$

- Range: $[0, 1]$. Lower is better.
- Decompose: $\text{Brier} = \text{Calibration} + \text{Resolution} - \text{Uncertainty}$ (Murphy decomposition).
- Proper scoring rule: maximized in expectation only when $\hat{p}_i = P(y_i = 1)$.
- Skill score: $\text{BSS} = 1 - \text{Brier} / \text{Brier}_{\text{ref}}$ (reference = always predict base rate).

### Proper Scoring Rules

A scoring rule $S(\hat{p}, y)$ is **proper** if the expected score is maximized when $\hat{p} = P(y)$. It is **strictly proper** if this maximum is unique.

| Scoring rule | Formula | Proper? |
|---|---|---|
| Log loss | $-[y \log\hat{p} + (1-y)\log(1-\hat{p})]$ | Strictly proper |
| Brier score | $(y - \hat{p})^2$ | Strictly proper |
| 0-1 loss | $\mathbf{1}[\hat{y} \neq y]$ | Not proper (no gradient signal for calibration) |

**Why log loss is proper**: Minimizing $E[-\log \hat{p}]$ = minimizing KL divergence from model distribution to true distribution. Any deviation from the true probability increases expected loss.

### Reliability Diagrams

Plot fraction of positives (empirical) vs mean predicted probability per bin. Points on the diagonal = perfect calibration. Below diagonal = overconfident. Above diagonal = underconfident.

---

## 3. Regression Metrics

### MSE, MAE, Huber

| Metric | Formula | Properties |
|---|---|---|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Differentiable everywhere; heavily penalizes outliers |
| RMSE | $\sqrt{\text{MSE}}$ | Same units as target; easier to interpret |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Robust to outliers; not differentiable at 0 |
| Huber | $L_\delta(y,\hat{y})$ | MSE for small errors, MAE for large — best of both |

**Huber loss**:
$$L_\delta(a) = \begin{cases} \frac{1}{2}a^2 & |a| \leq \delta \\ \delta(|a| - \frac{1}{2}\delta) & |a| > \delta \end{cases}$$

Threshold $\delta$ is a hyperparameter — typically chosen at the 90th–95th percentile of residuals.

**When to use what**:
- MAE: median regression; outliers expected and shouldn't dominate.
- MSE: mean regression; outliers are real signal you care about.
- Huber: production systems with occasional bad labels or anomalous inputs.

### MAPE Pitfalls

$$\text{MAPE} = \frac{100}{n} \sum \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

**Pitfalls**:
1. Undefined when $y_i = 0$.
2. Asymmetric: over-predictions and under-predictions of the same magnitude are penalized differently.
3. Penalizes under-forecasting more than over-forecasting (bad for inventory/safety-critical).
4. Not meaningful when the target can be negative.

**Alternatives**: sMAPE (symmetric MAPE, but still has issues), MASE (mean absolute scaled error, normalizes by in-sample naive forecast — best for time series).

### R² and Adjusted R²

$$R^2 = 1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

$$R^2_{\text{adj}} = 1 - (1 - R^2)\frac{n-1}{n-p-1}$$

- $R^2$: fraction of variance explained. $R^2 = 1$ is perfect; $R^2 = 0$ matches a constant baseline. Can be negative for worse-than-baseline models.
- Adding features always increases $R^2$. Adjusted $R^2$ penalizes for number of predictors $p$.
- **Pitfall**: High $R^2$ does not guarantee a good model. Anscombe's quartet — four datasets with identical $R^2$ but wildly different distributions.

### Pinball Loss for Quantile Regression

$$\rho_\tau(y, \hat{q}) = \begin{cases} \tau (y - \hat{q}) & y \geq \hat{q} \\ (1 - \tau)(\hat{q} - y) & y < \hat{q} \end{cases}$$

- Minimizing pinball loss at quantile $\tau$ produces the $\tau$-th conditional quantile.
- $\tau = 0.5$ recovers MAE.
- Use for prediction intervals: fit $\tau = 0.1$ and $\tau = 0.9$ separately to get an 80% interval.
- Proper scoring rule for quantiles — unlike symmetric losses.

---

## 4. Ranking Metrics

### NDCG (Normalized Discounted Cumulative Gain)

**Motivation**: When relevance is graded (0/1/2/3) and position matters (rank 1 is more valuable than rank 10).

**Step 1 — DCG@k**:
$$\text{DCG}@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

$rel_i$ = relevance score of item at position $i$. Position discount $\log_2(i+1)$ reduces gain for lower ranks.

**Step 2 — IDCG@k**: DCG of the ideal (perfect) ranking — sort by true relevance descending, take DCG@k.

**Step 3 — NDCG@k**:
$$\text{NDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}$$

- Range: $[0, 1]$. NDCG = 1 means perfect ordering.
- Undefined if there are no relevant documents (IDCG = 0) — handle by treating as 0 or excluding.

**Example** (k=3, true relevance [3, 2, 3, 0, 1]):

System returns [3, 0, 2] → DCG = (7/1) + (0/1.585) + (3/2) = 8.5

Ideal [3, 3, 2] → IDCG = (7/1) + (7/1.585) + (3/2) = 13.915

NDCG@3 = 8.5 / 13.915 = **0.611**

### MAP (Mean Average Precision)

For binary relevance (relevant/not relevant):

$$\text{AP} = \frac{1}{R} \sum_{k=1}^{n} P@k \cdot \mathbf{1}[\text{doc}_k \text{ is relevant}]$$

$$\text{MAP} = \frac{1}{Q} \sum_{q=1}^{Q} \text{AP}(q)$$

- $R$: total number of relevant documents for the query.
- AP rewards finding relevant documents early.
- MAP averages AP across queries.
- Sensitive to missing relevant items (unlike NDCG with fixed $k$).

### MRR (Mean Reciprocal Rank)

$$\text{MRR} = \frac{1}{Q} \sum_{q=1}^{Q} \frac{1}{\text{rank}_q}$$

- $\text{rank}_q$: rank of the first relevant result for query $q$.
- Only cares about the first relevant hit — ignores subsequent ones.
- Use when there is one "right" answer (e.g., QA, spelling correction, entity lookup).

### Precision@k and Recall@k

$$P@k = \frac{\text{relevant items in top } k}{k}, \quad R@k = \frac{\text{relevant items in top } k}{\text{total relevant items}}$$

| Metric | Use when |
|---|---|
| P@k | Corpus is large/unbounded; user only sees top k |
| R@k | All relevant items matter (e.g., legal document retrieval) |
| MAP | Want to account for full recall profile |
| NDCG@k | Graded relevance + position discount |
| MRR | Only first relevant result matters |

---

## 5. Novelty and Diversity in Rankings

Standard ranking metrics optimize for relevance without regard for diversity. A system that always returns the same popular item can achieve high NDCG while providing poor user experience.

### Coverage

$$\text{Coverage} = \frac{|\text{unique items recommended across all users}|}{|\text{catalog}|}$$

Measures whether the system explores the catalog or concentrates on a few items. A system with 1% coverage is vulnerable to cold-start and fails long-tail users.

### Intra-List Diversity (ILD)

$$\text{ILD} = \frac{1}{\binom{k}{2}} \sum_{i \neq j \in L} d(i, j)$$

$d(i, j)$ = dissimilarity between items $i$ and $j$ (e.g., cosine distance in embedding space). Averaged over all pairs in the recommendation list $L$.

**Trade-off**: High ILD can hurt precision if dissimilar items are less relevant. Typical solution: re-rank for diversity after initial retrieval using maximal marginal relevance (MMR).

### Serendipity

$$\text{Serendipity} = \frac{1}{|U|} \sum_{u} \frac{|\text{relevant} \cap \text{unexpected}|}{k}$$

Unexpected = not in what a naive baseline (e.g., popularity model) would recommend.

### Intent-Aware Metrics and Exposure Fairness *(niche — specialized ranking/fairness roles only)*

**Intent-aware NDCG**: when a query has multiple interpretations (e.g., "jaguar" = car or animal), standard NDCG rewards systems that cover only one intent. IA-NDCG weights each result by subtopic probability and penalizes redundancy within a subtopic.

**Exposure fairness**: protected groups should receive proportional ranking exposure relative to their relevance in the corpus — used in hiring/lending/content platforms to prevent systematic suppression of minority groups.

---

## 6. Cost-Sensitive Evaluation

### Custom Cost Matrices

Standard accuracy treats all misclassifications equally. A cost matrix assigns different costs to each cell of the confusion matrix.

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | $C(TP)$ — benefit of correct detection | $C(FN)$ — cost of missing a positive |
| **Actual Negative** | $C(FP)$ — cost of false alarm | $C(TN)$ — benefit of correct rejection |

**Expected cost**:
$$\text{Expected Cost} = \frac{1}{n}\sum_i \left[ y_i \cdot (\hat{y}_i \cdot C(TP) + (1-\hat{y}_i) \cdot C(FN)) + (1-y_i) \cdot (\hat{y}_i \cdot C(FP) + (1-\hat{y}_i) \cdot C(TN)) \right]$$

### Optimal Threshold Under Cost

The optimal decision threshold is not always 0.5. Under asymmetric costs:

$$\hat{y} = 1 \iff \hat{p}(x) \geq \frac{C(FP) - C(TN)}{C(FP) - C(TN) + C(FN) - C(TP)}$$

**Example**: In fraud detection, $C(FN) = \$500$ (undetected fraud), $C(FP) = \$5$ (manual review). Threshold = $5/(5+500) \approx 0.01$ — flag at 1% probability.

### When Accuracy is Wrong

| Scenario | Why accuracy fails | Use instead |
|---|---|---|
| 1% fraud rate | 99% accuracy by predicting "no fraud" | F1, PR-AUC, Expected Cost |
| Rare disease screening | Missing a positive (FN) is catastrophic | Recall, $F_\beta$ with $\beta > 1$ |
| Spam filter | False positives destroy user trust | Precision, $F_\beta$ with $\beta < 1$ |
| Credit scoring | Different costs for different score ranges | Expected cost, scorecard metrics |

---

## 7. Prediction Intervals and Uncertainty

### Coverage Guarantee

A prediction interval $[\hat{l}(x), \hat{u}(x)]$ has **coverage** $1-\alpha$ if:

$$P(Y_{n+1} \in [\hat{l}(X_{n+1}), \hat{u}(X_{n+1})]) \geq 1 - \alpha$$

Standard intervals (e.g., $\hat{y} \pm 2\sigma$) assume distributional form. Conformal prediction provides distribution-free coverage guarantees.

### PICP and MPIW

| Metric | Formula | Measures |
|---|---|---|
| PICP (PI Coverage Probability) | $\frac{1}{n}\sum \mathbf{1}[y_i \in [\hat{l}_i, \hat{u}_i]]$ | Whether intervals contain actual values |
| MPIW (Mean PI Width) | $\frac{1}{n}\sum (\hat{u}_i - \hat{l}_i)$ | How wide intervals are (efficiency) |

A trivial model can achieve 100% PICP by returning $[-\infty, +\infty]$. Good intervals maximize PICP for a given MPIW — or equivalently, minimize MPIW subject to target PICP.

### Conformal Prediction

1. Fit model on training set. Compute nonconformity scores on calibration set: $s_i = |y_i - \hat{y}_i|$.
2. Compute $q = (1-\alpha)(1 + 1/|\text{cal}|)$ quantile of calibration scores.
3. Test interval: $\hat{y}_{n+1} \pm q$.

Guarantees marginal coverage $\geq 1-\alpha$ under exchangeability (no distributional assumption). The calibration set is the price.

---

## 8. Statistical Testing

Never compare two models with a point estimate difference alone. Differences on held-out sets are noisy.

### Tests for Model Comparison

| Setting | Test | Notes |
|---|---|---|
| Two classifiers, paired examples | **McNemar's test** | Tests whether disagreements are systematic |
| Two models, regression, paired | **Paired t-test** | Assumes normality of differences |
| Two models, non-normal/small n | **Wilcoxon signed-rank** | Non-parametric alternative to paired t-test |
| Multiple classifiers, multiple datasets | **Friedman test + Nemenyi post-hoc** | Rank-based, non-parametric |

### McNemar's Test

Build a $2\times2$ table of disagreements: $b$ = model A correct, B wrong; $c$ = model B correct, A wrong.

$$\chi^2 = \frac{(b - c)^2}{b + c}$$

Significant result means the models make systematically different errors — not just random noise. $p < 0.05$ is sufficient, but report effect size, not just significance.

### Multiple Hypothesis Correction

Running $k$ tests at $\alpha = 0.05$ gives expected false positives $\approx 0.05k$.

| Method | Controls | When to use |
|---|---|---|
| Bonferroni | FWER (family-wise error rate) | Few tests, conservative |
| Holm-Bonferroni | FWER | Slightly less conservative |
| Benjamini-Hochberg | FDR (false discovery rate) | Many tests, acceptable to have some false positives |

**Interview hook**: Bonferroni divides $\alpha$ by $k$ — it becomes very conservative with many comparisons. Benjamini-Hochberg is almost always preferable in ML contexts.

---

## 9. Evaluation in Generative Models

### Image Generation

| Metric | What it measures | Pitfalls |
|---|---|---|
| **FID** (Fréchet Inception Distance) | Distance between feature distributions of real vs generated images (lower = better) | Sensitive to sample size; requires 10k+ samples; Inception features may not generalize |
| **IS** (Inception Score) | Generated images should be class-conditional + diverse (higher = better) | Doesn't use real images; can be gamed; mode-collapse not always detected |
| **Precision/Recall** (P&R) | Precision = quality; Recall = diversity | Requires careful $k$-NN radius selection |

FID formula: $\text{FID} = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$

### Text Generation

| Metric | What it measures | Known failure |
|---|---|---|
| **BLEU** | n-gram overlap with reference | Rewards short, safe outputs; misses semantics |
| **ROUGE** | Recall-oriented n-gram overlap | Doesn't handle paraphrase; works for summarization |
| **BERTScore** | Contextual embedding similarity | Expensive; biased toward BERT-fluent text |
| **Win rate** | % of pairwise human preferences | Ground truth, but expensive and noisy |

**Key insight**: BLEU can be close to 0 for a perfect translation that happens to use different words than the reference. For ranking model quality in practice, human win rate or preference evaluation is the only reliable signal.

**LLM-as-judge**: Use a larger model to rate outputs. Faster than human eval but biased toward the judge model's style. Always calibrate with spot human checks.

---

## 10. Offline vs Online Metrics

### The Gap

Offline metrics are proxies for real-world impact. The correlation between offline lift and online lift is empirically often weak.

| Offline metric | Online surrogate | Known gaps |
|---|---|---|
| CTR on held-out data | Real CTR | Train-serve distribution shift; position bias |
| NDCG on labeled queries | Dwell time, task success | Users don't click on labeled "relevant" items |
| BLEU | User satisfaction | Short generations score well but are unhelpful |
| Validation accuracy | Revenue, retention | Label noise; population shift |

### Simpson's Paradox

A metric can improve in every segment but worsen in aggregate if segment sizes shift. Example: model A has better accuracy than model B on both mobile and desktop users, but lower overall accuracy because model A was tested on a harder mix of users.

**Defense**: Always stratify metrics by subgroup. Report weighted and unweighted aggregates.

### Metric Drift

When a model is deployed, the data distribution shifts toward the model's behavior (feedback loops, self-selection, behavioral change). A model that was well-calibrated at launch can become miscalibrated within weeks.

**Defense**: Continuously monitor ECE, coverage, and prediction distribution. Trigger retraining alerts when drift exceeds thresholds.

### Guardrail Metrics

Before launching an A/B experiment, define:
1. **North star metric**: the primary business objective to move.
2. **Guardrail metrics**: metrics that must not regress (e.g., latency, spam rate, unsubscribe rate).
3. **Minimum detectable effect (MDE)**: compute sample size before starting the experiment.

A guardrail metric violation stops the experiment even if the north star improves.

---

## 11. Interview Questions

**Q1: You have a highly imbalanced binary classification problem (1% positives). Your colleague says the model has 99% accuracy. What's your response?**

A model predicting all negatives achieves 99% accuracy trivially. I'd report precision, recall, F1 on the positive class, and PR-AUC. Accuracy is only useful here if we know the cost of FP equals the cost of FN, which is unlikely. I'd also check whether the training data was balanced or whether class weights were used — class imbalance during training often requires oversampling, undersampling, or re-weighting.

---

**Q2: When would you use ROC-AUC vs PR-AUC?**

ROC-AUC is appropriate when both classes matter and the cost of FP and FN are comparable. It's the probability that a random positive is ranked above a random negative. PR-AUC is better when the positive class is rare — it directly exposes the precision–recall trade-off without the true negatives inflating FPR. For 1% prevalence, a model with ROC-AUC=0.95 can have PR-AUC=0.20, which tells you it's not useful in practice.

---

**Q3: Explain NDCG and when you'd prefer it over MAP.**

NDCG handles graded relevance (0/1/2/3) and applies a logarithmic position discount. MAP assumes binary relevance and averages precision at each recall point. Use NDCG when: (a) you have multi-level relevance judgments, (b) higher-ranked positions matter more, (c) you're evaluating at fixed $k$. MAP is simpler and better when relevance is binary and you want to evaluate full recall.

---

**Q4: Your model's PR-AUC improved by 2% in offline evaluation but the A/B test showed no lift. Why?**

Several possible causes: (a) the offline evaluation was on a different distribution than production (covariate shift), (b) the held-out set had label noise or selection bias, (c) the metric doesn't correlate with the business outcome — PR-AUC measures ranking quality but users might not respond to the margin of improvement at the operating threshold, (d) the experiment was underpowered (insufficient sample or duration), (e) novelty effects masked or amplified the signal.

---

**Q5: How would you compare two models statistically when you only have a single hold-out set?**

Use McNemar's test for classifiers — it tests whether the two models make systematically different errors on the same examples. Build the $2\times2$ disagreement matrix (both correct, A correct/B wrong, A wrong/B correct, both wrong), compute $\chi^2 = (b-c)^2/(b+c)$. For regression models, use a paired t-test (or Wilcoxon if normality is questionable) on per-example losses. Never just compare point estimates — a 0.3% lift that isn't statistically significant could easily be noise.

---

**Q6: A recommendation system has very high NDCG but users are churning. What metrics might explain this?**

NDCG measures relevance but not diversity, novelty, or coverage. Possible causes: (a) the system recommends the same popular items repeatedly — measure intra-list diversity and coverage, (b) the system over-exploits user history with no serendipity — users feel stuck in a filter bubble, (c) the relevance labels used to train/evaluate are stale and don't reflect current user intent, (d) the system optimizes for clicks but not for session satisfaction or task completion. The gap between offline NDCG and online engagement is the fundamental problem.

---

**Q7: What is a proper scoring rule and why does it matter for model training?**

A proper scoring rule is one whose expected value is minimized (or maximized) when the model outputs the true probability. Log loss and Brier score are both strictly proper — any deviation from the true probability increases expected loss. This matters because: (a) optimizing a proper scoring rule incentivizes the model to output calibrated probabilities, not just correct hard decisions; (b) improper scoring rules (e.g., accuracy, hinge loss) can be minimized by confidently wrong models; (c) log loss = cross-entropy is the standard choice for neural networks because it has good gradients and is strictly proper.

For active-recall drilling on these terms, see [classical-ml-flashcards.md](../03-classical-ml/_flashcards.md).
