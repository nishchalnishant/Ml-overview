---
module: Interview Prep
topic: Ml
subtopic: Additional Ml Interview Topics
status: unread
tags: [interviewprep, ml, ml-additional-ml-interview-top]
---
# Additional ML Interview Topics

## What This File Is For

These topics appear in interviews to test whether you have production judgment — the ability to reason about failure modes, tradeoffs, and edge cases that only become visible when models ship. The structure for each:

1. What the interviewer is actually testing — the underlying competency
2. The reasoning structure — why first-principles thinkers approach it this way
3. The pattern in action — a worked example
4. Common traps — where people go wrong and why

---

# 1. Data Leakage

## What the interviewer is actually testing

Whether you understand that excellent offline metrics are suspicious, not reassuring, and whether you can construct the specific causal story of why a given leakage pattern inflates metrics. Anyone can say "leakage happens when future information gets into training." The test is whether you can diagnose it in a specific scenario.

## The reasoning structure

**The definition.** Leakage occurs when information that would not be available at prediction time in production is available during training or validation. The model learns a shortcut that works on historical data but does not generalize to actual prediction tasks.

**Three mechanisms:**

**Temporal leakage.** Features or labels are computed using information from the future relative to the prediction time. Example: a demand forecasting model trained to predict sales for Monday uses features that include Tuesday's actual inventory replenishment decisions. At training time, Tuesday is past; at prediction time, Tuesday is future.

**Target leakage.** Features are computed using information derived from the label. Example: a fraud detection model includes "account suspended within 7 days of transaction" as a feature. This feature is perfectly predictive — because it literally encodes whether the transaction was flagged as fraud. It would never be available at prediction time (the label is what you are predicting).

**Pipeline leakage.** Preprocessing steps (normalization, imputation, encoding) are fit on the entire dataset, including the test set. Example: normalizing features using mean/std computed on train + test together. The test set's statistics leak into the training distribution. The model performs well on that test set and poorly on any future data.

**How to detect leakage:**
1. Suspiciously high metrics: near-perfect AUC, F1 > 0.95 on a hard problem should trigger investigation
2. Feature importance check: if an unexpected feature has extremely high importance, trace its provenance
3. Temporal sanity check: for each feature, verify it would be available at the prediction timestamp
4. Train-test gap: if train performance >> test performance, or if test performance >> holdout performance, leakage may be present

## The pattern in action

**Credit default prediction.** Features: income, employment status, credit score, loan amount. Label: defaulted within 12 months.

Leaked feature: "missed payment date" — the date of the first missed payment on the loan. This is predictive (if they missed a payment, they likely defaulted). But the first missed payment occurs after the loan is issued. At prediction time (loan approval decision), this date does not exist.

**Result without catching it:** AUC = 0.97 on validation. **Result after removing:** AUC = 0.72 on validation. The 0.25 AUC difference represents pure leakage.

**Target encoding leakage.** A high-cardinality categorical feature (zip code, merchant ID) is target-encoded as the mean label value per category, computed across the entire training set. If the same transaction appears in both the encoding computation and the model training, the model learns the label directly. Solution: k-fold target encoding (hold out each fold from the encoding computation), or use a Bayesian smoothed prior.

## Common traps

**"Our validation metrics look great."** If the validation set was constructed with the same leakage as training, great validation metrics prove nothing. The test is whether performance holds on truly held-out future data.

**Pipeline-level leakage in preprocessing.** Fitting a StandardScaler on the full dataset before splitting is the most common mistake. Always fit preprocessing transforms on training data only, then apply to validation and test.

**Feature availability at prediction time vs. at training time.** In batch prediction jobs, some features may be computed using information from the same time window as the label. Trace every feature back to its timestamp.

---

# 2. Imbalanced Classes and SMOTE

## What the interviewer is actually testing

Whether you understand the full decision tree: when imbalance is actually a problem, what fixes are available, and why each fix has specific failure modes. Reaching for SMOTE without understanding why is a pattern that signals incomplete understanding.

## The reasoning structure

**When is imbalance actually a problem?**

Imbalance matters when the model optimizes a metric that is insensitive to minority class performance. If you minimize cross-entropy on 99%/1% data, the model can achieve 99% accuracy by predicting the majority class always. AUC is more robust to imbalance (it is invariant to class balance in the binary case), but precision-recall is often more informative for rare events.

**Fix the evaluation metric first.** If you care about rare positive cases, use precision-recall AUC or F1 on the positive class, not accuracy. Then ask whether the model actually struggles.

**Four approaches, in order of invasiveness:**

1. **Threshold adjustment.** Lower the classification threshold below 0.5. The model's probability outputs are unchanged; you change where you draw the positive/negative boundary. Best: always try this first. Cost: zero. Risk: precision-recall tradeoff.

2. **Class weighting.** Increase the loss weight for minority class examples. `class_weight='balanced'` in scikit-learn or `pos_weight` in PyTorch. Effectively the same as upsampling. Cost: zero implementation overhead. Risk: if minority class is noisy, upweighting it amplifies noise.

3. **Resampling.** Oversample minority class (duplicate examples or use SMOTE), undersample majority class. SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic minority examples by interpolating between a minority example and one of its k nearest neighbors.

   SMOTE risk: synthetic points may fall in regions that belong to the majority class (if class boundaries are complex), or in regions with no real examples (in high dimensions, interpolation may be extrapolation). Can blur decision boundaries.

4. **Algorithmic approaches.** Focal loss (from RetinaNet): downweights easy negatives, forces the model to focus on hard examples. More useful when imbalance is extreme (1:1000+).

## The pattern in action

**Fraud detection.** 99.9% legitimate transactions, 0.1% fraud.

Step 1: measure model performance with precision-recall AUC, not accuracy. Accuracy is meaningless here.

Step 2: set `class_weight={0: 1, 1: 1000}` in the classifier. Train. Evaluate on a temporally held-out test set. This is the baseline.

Step 3: if precision-recall is unacceptable, try SMOTE. Generate synthetic fraud examples. Evaluate.

In practice for fraud: SMOTE rarely helps much over class weighting. The real problem is not the training distribution — it is that fraud patterns change over time (concept drift). The minority class problem is secondary.

**High-dimensional SMOTE failure:** a fraud detection model with 500 features. SMOTE generates synthetic fraud examples by interpolating between two real fraud examples. In 500 dimensions, "interpolating" two sparse high-dimensional vectors often produces a vector that is effectively equidistant from both. The synthetic example does not belong to the fraud class boundary.

## Common traps

**Applying SMOTE before splitting into train/test.** Synthetic examples generated from the full dataset can put duplicated information in both train and test. Always apply SMOTE to training data only.

**Using SMOTE as the first step.** Try threshold adjustment and class weighting first. They have no failure modes and are free. SMOTE has failure modes and cost.

**SMOTE for concept drift.** If the minority class underperforms because it is underrepresented or its distribution is shifting over time, SMOTE on training data does not solve the temporal mismatch. Collect more recent minority examples.

---

# 3. BatchNorm vs LayerNorm

## What the interviewer is actually testing

Whether you can reason from the statistics these normalizations compute to why each is appropriate in specific architectures — not just recite "BatchNorm for CNNs, LayerNorm for Transformers."

## The reasoning structure

**What normalization solves.** Internal covariate shift: as parameters update, the distribution of each layer's inputs changes, making training unstable. Normalizing activations within each layer stabilizes training and enables higher learning rates.

**BatchNorm** normalizes across the batch dimension for each feature.

For feature j, across batch examples i:
```
μ_j = (1/B) Σ_i x_{ij}
σ²_j = (1/B) Σ_i (x_{ij} - μ_j)²
x̂_{ij} = (x_{ij} - μ_j) / √(σ²_j + ε)
y_{ij} = γ_j · x̂_{ij} + β_j
```

Statistics are computed per-feature, across examples. Requires a large enough batch to compute stable statistics (typical minimum: batch size ≥ 32).

**Where BatchNorm works:** CNNs, where the batch represents different images and feature maps at the same spatial position should have similar distributions. The spatial dimensions are treated as additional batch elements.

**Where BatchNorm fails:**
- Small batch sizes: statistics are noisy, training unstable
- Sequence models with variable-length sequences: batch statistics mix examples of different lengths
- Test-time/inference: uses running statistics from training. If test distribution differs, these statistics are stale
- Online learning or single-example inference: batch of 1 makes statistics undefined

**LayerNorm** normalizes across the feature dimension for each example.

For example i, across features j:
```
μ_i = (1/D) Σ_j x_{ij}
σ²_i = (1/D) Σ_j (x_{ij} - μ_i)²
x̂_{ij} = (x_{ij} - μ_i) / √(σ²_i + ε)
y_{ij} = γ_j · x̂_{ij} + β_j
```

Statistics are computed per-example, across features. Independent of batch size. Identical behavior at train and test time.

**Where LayerNorm works:** Transformers and sequence models. Each token's representation is normalized independently. Works with any batch size. Consistent train/test behavior.

## The pattern in action

**Why does the Vision Transformer (ViT) use LayerNorm instead of BatchNorm?**

ViT processes images as sequences of patches. The same model must handle: variable batch sizes during training, batch size 1 during inference, fine-tuning on small datasets (small batches). BatchNorm's statistics become unreliable at small batch sizes. LayerNorm normalizes each patch's representation independently, so batch size is irrelevant.

**Why does ResNet use BatchNorm instead of LayerNorm?**

ResNet processes fixed-size images with convolutional features. Large batch sizes are standard. The spatial positions in a feature map are homogeneous — the same filter response at any position in any image in the batch should be normalized together. BatchNorm across the batch + spatial positions is exactly right for this structure.

## Common traps

**Using BatchNorm with batch size 1 at inference.** The running statistics from training are used, but they may not match the single-example distribution. Performance can degrade compared to training. LayerNorm avoids this problem entirely.

**Not understanding that BatchNorm has different train/test behavior.** In PyTorch: `model.train()` uses batch statistics; `model.eval()` uses running statistics accumulated during training. Forgetting to call `model.eval()` before evaluation is a common bug that causes inflated variance in validation metrics.

---

# 4. Gradient Boosting: XGBoost vs LightGBM vs CatBoost

## What the interviewer is actually testing

Whether you can choose between boosting libraries based on data characteristics, and whether you understand the algorithmic differences that drive the performance profiles — not just "LightGBM is faster."

## The reasoning structure

**What gradient boosting is.** An ensemble of weak learners (typically shallow decision trees) trained sequentially, each one correcting the residuals of the previous ensemble. The loss is minimized in function space via gradient descent.

**The three libraries solve different bottlenecks:**

**XGBoost (Chen & Guestrin, 2016).** Level-wise tree growth: at each depth level, finds the best split across all nodes at that depth. Includes L1/L2 regularization on leaf weights. Handles sparse features efficiently. Well-established, well-studied, reliable default.

**LightGBM (Microsoft, 2017).** Leaf-wise growth: always splits the leaf with highest loss reduction regardless of depth. Produces deeper, more complex trees faster. Gradient-based One-Side Sampling (GOSS): sample full gradients for high-gradient examples, sample a fraction of low-gradient examples. Exclusive Feature Bundling (EFB): bin sparse features together. Result: 3–10× faster than XGBoost on large datasets. Risk: leaf-wise growth overfit faster; requires num_leaves tuning.

**CatBoost (Yandex, 2018).** Ordered boosting: to prevent target leakage in categorical encoding, each example is processed using a model trained only on examples that came before it (in a shuffled order). Symmetric trees: all nodes at a given depth use the same split. Fast inference, less overfitting. Built-in categorical handling: ordered target statistics without manual encoding.

**Decision criteria:**

| Scenario | Recommendation |
|----------|----------------|
| Medium dataset, baseline | XGBoost |
| Large dataset, speed matters | LightGBM |
| Many categorical features | CatBoost |
| Need careful regularization | XGBoost |
| Need fastest iteration | LightGBM |

## The pattern in action

**E-commerce click prediction.** Features include 50 categorical columns (product category, merchant ID, city) and 100 numerical columns. 50M rows.

CatBoost's ordered boosting prevents target leakage in categorical encoding. On this specific dataset, CatBoost outperforms XGBoost and LightGBM on validation AUC by 0.3–0.5 points because the categorical features are high-cardinality and high-signal.

**Training time:** LightGBM trains in 8 minutes, XGBoost in 35 minutes, CatBoost in 18 minutes on the same hardware. If you are running hundreds of hyperparameter search trials, the 4× training time difference between LightGBM and XGBoost is significant.

**LightGBM overfitting risk:** with leaf-wise growth, setting num_leaves=127 (2^7) allows trees as deep as 7 but with much more asymmetric structure. On a 5M-row dataset, this is fine. On a 5K-row dataset, this overfits severely. Tune num_leaves to control complexity.

## Common traps

**Treating LightGBM as a drop-in replacement for XGBoost with the same hyperparameters.** LightGBM's num_leaves controls complexity; XGBoost's max_depth does. They are not equivalent. A LightGBM model with num_leaves=31 is NOT the same as XGBoost with max_depth=5.

**Not encoding categoricals for XGBoost.** XGBoost requires numeric inputs. Passing string categories will silently fail or coerce to garbage. CatBoost handles strings natively.

**Ignoring class balance in early stopping.** XGBoost and LightGBM stop training based on validation metric. If validation set does not represent the real class balance (or is too small), early stopping stops at the wrong point.

---

# 5. Clustering: DBSCAN vs K-Means

## What the interviewer is actually testing

Whether you can reason from cluster shape assumptions to algorithm choice, and whether you understand the specific failure modes of each algorithm in high dimensions.

## The reasoning structure

**K-Means minimizes within-cluster variance.** This implicitly assumes:
- Clusters are roughly spherical (equal variance in all directions)
- Clusters are roughly equal in size
- k is known

**DBSCAN identifies dense regions.** It assumes:
- Clusters have arbitrary shape
- Noise/outlier points exist (they should not be assigned to any cluster)
- Clusters are defined by density, not distance to a centroid

**K-Means algorithm:**
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids to mean of assigned points
4. Repeat until convergence

Cost: O(n × k × d × iterations). Fast for large datasets. Sensitive to initialization (k-means++ addresses this).

**DBSCAN algorithm:**
- Core point: has ≥ minPts points within radius ε
- Border point: within ε of a core point but fewer than minPts neighbors
- Noise point: not within ε of any core point

Clusters: maximal connected sets of core points.

Cost: O(n log n) with spatial index; O(n²) naively. The ε and minPts parameters require tuning.

## The pattern in action

**Customer segmentation by purchase behavior.** 100K customers, 20 features. K-Means with k=5 is a natural starting point.

But: one segment is "high-value customers who buy twice a year" — a sparse, elongated cluster in purchase-frequency vs. spend space. K-Means splits this into two spherical blobs because it cannot model the elongated shape.

DBSCAN with appropriate ε detects this elongated cluster as one dense connected region.

**When K-Means is better:** Document clustering with TF-IDF features. 50K documents, want 20 topic clusters. The documents are dense in a high-dimensional space. DBSCAN's density concept degenerates in high dimensions (the "curse of dimensionality" — all pairwise distances converge). K-Means with k=20 runs in seconds and produces interpretable clusters.

**DBSCAN in high dimensions.** In d=100 dimensions, any two points have similar distances to each other (concentration of measure). ε must be tuned per dataset and is brittle. DBSCAN is most effective in d ≤ 10, or when a meaningful distance metric exists that preserves density structure.

## Common traps

**K-Means on non-normalized features.** K-Means uses Euclidean distance. A feature with range [0, 10000] will dominate features with range [0, 1]. Always normalize before K-Means.

**DBSCAN with different-density clusters.** If some clusters are dense (users in a city center) and others are sparse (rural users), a single ε value cannot detect both. HDBSCAN (hierarchical DBSCAN) handles variable-density clusters.

**Evaluating clustering with accuracy.** Clustering is unsupervised. Silhouette score, Davies-Bouldin index, and downstream task performance are appropriate evaluation signals. "Accuracy" requires ground truth labels that clustering does not assume.

---

# 6. Recommendation Systems: Collaborative Filtering vs Content-Based

## What the interviewer is actually testing

Whether you understand the cold-start problem as a structural constraint on system design, not a bug to fix with more data, and whether you can design a system that handles both returning users and new users correctly.

## The reasoning structure

**Collaborative filtering.** Learn from user-item interaction patterns. Users who interacted with similar items should receive similar recommendations.

Types:
- User-based CF: find similar users, recommend what they liked
- Item-based CF: find similar items, recommend items similar to what the user liked
- Matrix factorization (SVD, ALS): decompose the interaction matrix into user and item latent factors

Strengths: captures latent preferences, works without item metadata, discovers surprising but relevant items.
Weakness: **cold start** — new users (no interaction history) and new items (no interactions) have no latent factors. The model cannot make recommendations.

**Content-based filtering.** Learn from item attributes and user explicit preferences.

User profile: built from features of items the user interacted with. Recommendation: find items most similar to the user profile.

Strengths: works immediately for new users (ask for explicit preferences) and new items (item metadata is available at launch). Recommendations are interpretable ("recommended because of your interest in science fiction").

Weakness: **filter bubble** — users are recommended more of the same. Discovery is limited. Requires good item metadata, which is expensive to maintain.

**The cold start problem in three forms:**
1. New user cold start: no interaction history → cannot use CF
2. New item cold start: no interaction history → CF cannot surface it
3. System cold start: no interactions at all (new product launch)

## The pattern in action

**Streaming service.** New user signs up. They have not watched anything.

**Content-based onboarding:** ask the user to rate 5 genres and 5 titles they have seen elsewhere. Build an initial content-based profile. Use this for first 10 recommendations.

**Hybrid transition:** after 3 sessions (20+ interaction signals), switch to a weighted combination: 70% CF + 30% content-based. As more data accumulates, shift weight toward CF.

**New movie cold start:** a newly released movie has zero interaction data. CF cannot rank it. Solution: use content-based features (genre, director, cast) to find similar established movies. Recommend it to users who strongly prefer those similar movies. Collect interactions. As interactions accumulate, CF takes over.

**Matrix factorization at scale:** Netflix Prize winner used ALS (alternating least squares) for matrix factorization. For user u and item i, minimize:

```
min Σ_{(u,i) observed} (r_ui - p_u^T q_i)² + λ(‖p_u‖² + ‖q_i‖²)
```

This scales to millions of users and items. Unobserved entries are treated as unknowns (not zeros), which avoids the false negative problem in implicit feedback.

## Common traps

**Treating unobserved interactions as negative.** In implicit feedback (clicks, watches), you observe only positives. The absence of a click does not mean the user dislikes the item — they may never have seen it. Using zeros for unobserved entries introduces massive false negatives. Weight observed interactions higher and treat unobserved as uncertain (negative sampling, BPR loss).

**Not handling popularity bias.** CF matrices are dominated by popular items. The model learns to recommend popular items regardless of personalization. Counter with inverse popularity weighting, diversification constraints, or novelty reranking.

**Evaluating only on observed interactions.** A model that perfectly reconstructs observed ratings on the validation set may be learning popularity, not preferences. Use held-out users and held-out items in evaluation.

---

# 7. Model Calibration

## What the interviewer is actually testing

Whether you understand that probability outputs from classifiers are not calibrated by default, and whether you can reason about when calibration matters versus when ranking quality is sufficient.

## The reasoning structure

**What calibration means.** A model is calibrated if its predicted probabilities match empirical frequencies. If the model predicts P(Y=1) = 0.7 for a set of examples, 70% of those examples should actually be positive.

**Why classifiers are often miscalibrated:**
- SVMs output margin distances, not probabilities. These are not calibrated.
- Tree ensembles: boosted trees tend to be overconfident (scores near 0 or 1). Random forests tend to be underconfident (scores pulled toward 0.5 because of averaging).
- Neural networks: can be overconfident, especially on out-of-distribution examples.

**When calibration matters:**
- The predicted probability is used directly in a decision (bid price in ad auction, risk score in medical triage)
- Multiple models' scores are compared across populations (comparing fraud risk scores across geographies with different base rates)
- GDPR-style right to explanation: "why was this loan denied?" requires trustworthy probability scores

**When calibration does not matter:**
- You only care about ranking (movie recommendation: top-10 items, not their exact probabilities)
- The downstream decision only uses the argmax (classification, not scoring)

**Calibration methods:**

**Platt scaling:** fit a logistic regression on top of model scores using a held-out calibration set. Assumes sigmoid relationship between score and calibrated probability.

**Isotonic regression:** fit a monotonic step function on calibration data. More flexible than Platt scaling; requires more calibration data.

**Temperature scaling:** divide logits by a scalar T before softmax: `P(y=k) = exp(z_k / T) / Σ exp(z_j / T)`. T > 1 softens the distribution (less confident). The single parameter T is fit to minimize NLL on a calibration set. Highly effective for neural networks.

**Diagnosing calibration:** reliability diagram (observed frequency vs. predicted probability, binned). Expected Calibration Error (ECE) = weighted average of bin-level gaps.

## The pattern in action

**Ad bidding system.** The model predicts click-through rate (CTR). The auction uses the formula: bid = max_bid × CTR × conversion_rate. Calibration matters directly: an overconfident model that predicts CTR=0.8 when the true rate is 0.3 will bid 2.7× too high and lose money.

**How to detect miscalibration:** take the held-out validation set. Bin predictions into deciles. For each bin, compute the predicted mean vs. actual mean. A well-calibrated model's actual rate tracks the predicted rate. An overconfident model shows actual rates systematically below predicted rates at high confidence levels.

**Applying temperature scaling:** for a neural network CTR model, optimize T on a calibration holdout:

```
T* = argmin_T NLL(y, softmax(z / T))
```

Typical T > 1 (model is overconfident). After calibration, ECE drops from 0.08 to 0.02.

## Common traps

**Calibrating on the training set.** The calibration method (Platt scaling, isotonic regression) must be fit on a held-out calibration set, not the training set. Fitting on training data produces calibrated training metrics but uncalibrated predictions on new data.

**Using calibrated probabilities to rank.** Calibration adjusts the probability values but preserves monotonic order if it fits a monotonic function (Platt, isotonic). But if different models are calibrated separately, their absolute probabilities become comparable, which they were not before.

**Treating ranking AUC as a proxy for calibration quality.** AUC measures discriminative ability (ranking). A model can have AUC = 0.95 and be severely miscalibrated. They measure different things.

---

# 8. Time-Series Validation

## What the interviewer is actually testing

Whether you understand that temporal structure invalidates standard cross-validation, and whether you can design a validation scheme that correctly estimates production performance.

## The reasoning structure

**Why random splitting fails for time series.** If you split randomly, your training set contains examples from all time periods, and your test set contains examples from all time periods. Training on t=100 and predicting t=50 is predicting the past from the future. This leaks future information backward and produces optimistic performance estimates.

**The correct principle:** train only on the past, evaluate on the future. This mirrors the production setting, where at any prediction time t, only data from before t is available.

**Validation schemes:**

**Simple time-based split.** Sort data chronologically. Use the first 80% for training, last 20% for test.
- Problem: only one evaluation split; variance in the performance estimate is high.

**Walk-forward validation (rolling origin cross-validation).** Create multiple train/test splits, each with an expanding training window:
- Split 1: train on months 1–6, test on month 7
- Split 2: train on months 1–7, test on month 8
- Split 3: train on months 1–8, test on month 9
- ...

Average performance across splits gives a low-variance estimate of how the model will perform on future data.

**Sliding window validation.** Fixed-size training window slides forward:
- Split 1: train on months 1–6, test on month 7
- Split 2: train on months 2–7, test on month 8
- ...

Use when you want to test whether the model remains accurate as distribution drifts, and when older data is irrelevant (e.g., fraud patterns from 3 years ago may not apply today).

**Gap period.** Insert a gap between the end of the training period and the start of the evaluation period to match production latency. If predictions are made at t=0 but features are computed with 24-hour lag, insert a 1-day gap.

## The pattern in action

**Demand forecasting.** Daily sales data for 3 years. Predict next 7 days.

Wrong: random 80/20 split. The model trains on Wednesdays to predict Mondays — that is impossible in production.

Right: walk-forward validation. Train on years 1–2, predict week 1 of year 3. Train on years 1–2 plus week 1, predict week 2. Repeat for 52 weeks. Compute MAE per week-of-year to separate seasonality from true model error.

**Cross-validation with time groups.** If data has natural time groups (user sessions, monthly cohorts), GroupKFold ensures that all examples from a time group are in the same fold. Prevents within-group leakage.

## Common traps

**Not accounting for look-ahead features.** A "7-day trailing average" feature computed at prediction time t uses data from t-6 to t. This is valid. But a "7-day trailing average" computed incorrectly as a centered average (t-3 to t+3) uses future data. Time-series features must be carefully verified to only use past data.

**Comparing model performance across different evaluation windows.** Model A validated on 2020 data vs. Model B validated on 2021 data cannot be directly compared — 2020 and 2021 have different distributions. Always compare models on the same temporal holdout.

**Using the test set multiple times.** In time-series settings where the model is retrained monthly, the "test set" for month 3 cannot be used to tune hyperparameters. If it is, it becomes part of the training signal and performance on month 4+ is what you actually need to measure.

---

# 9. KL Divergence vs Cross-Entropy

## What the interviewer is actually testing

Whether you understand the information-theoretic relationship between cross-entropy loss and KL divergence, and whether you can reason about the asymmetry in KL divergence for forward vs. reverse application.

## The reasoning structure

**Cross-entropy** H(P, Q) measures the expected code length when using code Q to encode events from distribution P:

```
H(P, Q) = -Σ P(x) log Q(x)
```

**KL divergence** D_KL(P ‖ Q) measures the extra bits required when using Q instead of the optimal code P:

```
D_KL(P ‖ Q) = Σ P(x) log(P(x) / Q(x)) = H(P, Q) - H(P)
```

**The relationship:**
```
H(P, Q) = H(P) + D_KL(P ‖ Q)
```

When P is the true data distribution (fixed during training), minimizing cross-entropy H(P, Q) with respect to Q is equivalent to minimizing D_KL(P ‖ Q). The entropy H(P) is a constant.

**The asymmetry.** KL divergence is not symmetric. Forward KL D_KL(P ‖ Q) and reverse KL D_KL(Q ‖ P) have different properties:

**Forward KL** (used in MLE, cross-entropy loss):
```
D_KL(P ‖ Q) = Σ P(x) log(P(x) / Q(x))
```
The average is taken under P. Where P(x) > 0 and Q(x) ≈ 0, the term is large. Minimizing forward KL requires Q to cover all regions where P has mass — **zero-avoiding**: Q must place probability everywhere P does.

**Reverse KL** (used in variational inference, RLHF policy regularization):
```
D_KL(Q ‖ P) = Σ Q(x) log(Q(x) / P(x))
```
The average is taken under Q. Where Q(x) > 0 and P(x) ≈ 0, the term is large. Minimizing reverse KL makes Q avoid regions where P is small — **zero-forcing**: Q concentrates on a subset of P's modes and ignores others. This is mode-seeking behavior.

**Why RLHF uses reverse KL.** The KL penalty in the RLHF objective D_KL(π_φ ‖ π_SFT) is reverse KL from the policy perspective. The policy π_φ learns to stay close to the SFT distribution while optimizing reward, mode-seeking within the acceptable distribution.

## The pattern in action

**Variational autoencoder (VAE).** The ELBO objective minimizes reconstruction loss + D_KL(q(z|x) ‖ p(z)).

This is reverse KL (from q to p). The approximate posterior q(z|x) is encouraged to be close to the prior p(z). Reverse KL makes q concentrate on the regions where p is large, which is what you want — the latent codes should match the prior structure.

**Why cross-entropy works for classification.** The true label distribution P is one-hot (all mass on one class). H(P) = 0. Minimizing H(P, Q) = minimizing D_KL(P ‖ Q) = making the model distribution Q assign high probability to the correct class.

## Common traps

**"KL divergence is just a distance metric."** KL divergence is not a metric — it is not symmetric, and it does not satisfy the triangle inequality. D_KL(P ‖ Q) ≠ D_KL(Q ‖ P) in general.

**Conflating forward and reverse KL in variational inference.** The direction of the KL divergence determines the mode-seeking vs. mean-seeking behavior of the approximate posterior. Getting this wrong means misunderstanding the properties of the approximation.

---

# 10. Feature Importance: Permutation Importance vs SHAP

## What the interviewer is actually testing

Whether you understand the scope of each method's claims and the conditions under which each is valid — particularly for correlated features, where both methods can mislead.

## The reasoning structure

**The question both methods answer:** which features contribute most to the model's predictions?

**Permutation importance.** Measure prediction performance. Permute (shuffle) feature j in the test set — breaking the relationship between feature j and the label. Measure the performance drop. The drop is the feature's importance.

Properties:
- Captures the real contribution of a feature in the model's context
- Measures importance for the specific model's decision function
- Fast to compute
- Aggregates across all examples (global, not per-example)

Problem with correlated features: if features A and B are highly correlated, permuting A still leaves B as a proxy. The model can recover information about A through B. Permutation importance underestimates the true importance of both correlated features.

**SHAP (SHapley Additive exPlanations).** Based on Shapley values from cooperative game theory. The Shapley value for feature j is the average marginal contribution of feature j across all possible orderings of features.

Properties:
- Satisfies local accuracy: feature contributions sum to the prediction
- Handles correlated features by averaging over all possible subsets
- Provides per-example explanations (local interpretability)
- Computationally expensive for exact computation (O(2^d)); approximated via tree structures for tree models or kernel methods for black-box models

**Key distinction.** Permutation importance measures global importance for the model's predictions. SHAP explains individual predictions. SHAP TreeExplainer runs in O(T × d × max_depth) for tree models, making it practical for XGBoost/LightGBM.

**Neither is causal.** SHAP values explain the model's behavior, not the true causal effect of a feature. A feature can have high SHAP importance because it is correlated with the causal factor, not because it causes the outcome.

## The pattern in action

**Loan default model.** Features: income, zip code, credit score, employment duration. Income and employment duration are correlated (higher-income people tend to have longer employment histories).

Permutation importance underestimates both income and employment duration because permuting one still leaves the other as a proxy.

SHAP handles this correctly by averaging over feature coalitions. SHAP shows that income has the higher marginal contribution; employment duration contributes less once income is known.

**Using SHAP for model debugging.** A fraud model has unexpectedly high feature importance for "account age." SHAP analysis reveals that young accounts with high transaction velocity get extreme SHAP values — the model uses account age as a proxy for synthetic accounts. The feature is legitimate, but the model learned to weight it more than the domain experts expected.

## Common traps

**Treating SHAP as causal.** High SHAP importance for "zip code" does not mean zip code causes the outcome. It means the model uses zip code for its predictions. The causal claim requires external reasoning.

**Using permutation importance for feature selection with correlated features.** If features A and B are correlated and you drop A based on low permutation importance, you may hurt performance because the model was implicitly relying on both. Use SHAP or sequential feature selection instead.

---

# 11. Label Smoothing

## What the interviewer is actually testing

Whether you understand why overconfident models generalize worse, and whether you can trace the mechanism by which label smoothing improves calibration and robustness.

## The reasoning structure

**The problem with hard targets.** Standard cross-entropy trains the model to output probability 1.0 for the correct class and 0.0 for all others. Perfect training accuracy requires logits → ±∞. This drives the model to be overconfident on training examples, leading to:
- Poor calibration (probability 0.99 when correct probability is 0.8)
- Poor generalization (fitting training noise)
- Brittle decisions near the decision boundary

**Label smoothing.** Replace hard targets y ∈ {0, 1} with soft targets:
```
y_smooth = y · (1 - α) + α/K
```
where K is the number of classes and α is the smoothing parameter (typically 0.1).

For the correct class: target is 1 - α + α/K ≈ 0.9 (not 1.0)
For incorrect classes: target is α/K ≈ 0.1/K (not 0.0)

**Effect on training.** The model is never rewarded for extreme logits. The optimal logit gap for a k-class problem is log((K-1)(1-α)/α), which is finite. This prevents degenerate large-weight solutions.

**Benefits:**
- Improved calibration: model outputs are less extreme
- Better generalization: prevents memorizing noisy labels
- Improved robustness to adversarial examples (less sensitivity to input perturbations)

## The pattern in action

**Image classification (ImageNet).** Without label smoothing: model outputs 0.98 probability on training classes, 0.03 validation accuracy on ambiguous examples (an image containing both a dog and a cat labeled "dog").

With label smoothing (α=0.1): model's maximum logit is bounded. On ambiguous examples, the model distributes probability more evenly. Validation accuracy improves marginally (0.5–1%). Calibration ECE improves significantly.

**When label smoothing hurts.** Distillation: when training a student model to reproduce a teacher's soft predictions, you want the student to learn the full probability distribution from the teacher. If labels are already soft (from the teacher), adding label smoothing adds noise. Do not use label smoothing when the targets are already soft.

## Common traps

**Using label smoothing for regression.** Label smoothing is specific to classification with discrete targets. For regression, alternatives like Huber loss or uncertainty estimation serve different but related purposes.

**Setting α too high.** α = 0.3 or higher destroys the training signal on clean labels. For well-labeled datasets, α = 0.1 is standard. For noisy labels, higher α may help but requires validation.

---

# 12. Ranking Metrics: Precision@K, Recall@K, NDCG, MAP

## What the interviewer is actually testing

Whether you can choose the right ranking metric for a given system, and whether you understand why position matters in ranking evaluation — not just whether relevant items appear in the top-k list.

## The reasoning structure

**Why ranking metrics are different from classification metrics.** A classifier outputs a single prediction. A ranking system outputs an ordered list of k items. The value of a relevant item depends on where it appears in the list — users are more likely to see items at position 1 than position 10.

**Precision@K:** of the K returned items, what fraction are relevant?
```
P@K = (relevant items in top K) / K
```
Does not penalize for putting relevant items at positions 8-10 vs. 1-3.

**Recall@K:** of all relevant items in the system, what fraction appear in the top K?
```
R@K = (relevant items in top K) / (total relevant items)
```
Measures how many relevant items the system surfaces. Useful when missing relevant items has high cost (medical literature retrieval, legal discovery).

**Average Precision (AP):** compute precision at each position where a relevant item appears, then average. Accounts for the order of relevant items.
```
AP = (1/R) Σ_k P@k · rel(k)
```
where rel(k) = 1 if the item at position k is relevant, R = total relevant items.

**Mean Average Precision (MAP):** average AP across multiple queries.

**NDCG@K (Normalized Discounted Cumulative Gain):** allows graded relevance (not just binary). Discounts the gain of relevant items by their position.
```
DCG@K = Σ_{i=1}^{K} (2^{rel_i} - 1) / log_2(i + 1)
NDCG@K = DCG@K / IDCG@K
```
IDCG is the ideal DCG (if items were sorted perfectly by relevance). NDCG = 1.0 is a perfect ranking.

**When to use each:**
- P@K: when you care about precision in the top K (news headline ranking)
- R@K: when recall is critical (medical literature retrieval)
- NDCG: when relevance is graded and position matters (search ranking)
- MAP: when binary relevance across multiple queries (document retrieval benchmarks)

## The pattern in action

**E-commerce search.** Query: "running shoes." 8 relevant products in a catalog of 10,000.

Ranked list at K=5: [relevant, irrelevant, relevant, irrelevant, relevant]

P@5 = 3/5 = 0.6
R@5 = 3/8 = 0.375
AP = (1/8)(P@1·1 + P@3·1 + P@5·1) = (1/8)(1.0 + 0.667 + 0.6) = 0.283

Suppose a different ranking: [irrelevant, irrelevant, irrelevant, relevant, relevant]
P@5 = 2/5 = 0.4 (worse)
AP = (1/8)(P@4·1 + P@5·1) = (1/8)(0.25 + 0.4) = 0.081 (much worse)

MAP captures that putting relevant items lower in the list is bad, even if P@5 only drops from 0.6 to 0.4.

## Common traps

**Using accuracy for ranking.** Accuracy = (correct predictions) / n. For ranking, whether the top item is correct matters far more than whether the last item is correct. Use ranking metrics, not classification metrics, to evaluate ranked outputs.

**Not accounting for graded relevance when available.** If your relevance labels are "highly relevant / somewhat relevant / not relevant," using binary metrics collapses the distinction. NDCG with `rel_i ∈ {0, 1, 2}` captures this grading.

---

# 13. PU Learning

## What the interviewer is actually testing

Whether you can reason about training data where the absence of a label does not mean absence of the phenomenon — a common real-world scenario that standard supervised learning handles incorrectly.

## The reasoning structure

**The PU (Positive-Unlabeled) problem.** You have two types of data:
- P: labeled positive examples (confirmed true)
- U: unlabeled examples — some are positive, some are negative, but you do not know which

Standard supervised learning would treat U as negative examples. This is wrong. It introduces systematic label noise on the positive examples in U, biasing the model.

**Where PU learning appears:**
- Fraud detection: you have confirmed fraud cases (P) and unreviewed transactions (U — some are fraud, most are not)
- Medical screening: you have confirmed diagnoses (P) and unscreened patients (U)
- Citation networks: you have known positive links (P) and the absence of links, which is not confirmation of non-linkage

**Approaches:**

**Biased SVM (Liu et al., 2003).** Treat U as negative, train SVM. Use the trained model to identify "reliable negatives" (unlabeled examples far from the positive class boundary). Retrain on P + reliable negatives. Iterate.

**Unbiased risk estimator (du Plessis et al., 2014).** Construct an unbiased estimator of the risk on P and N (even without N labels) using the positive class prior π = P(Y=1):
```
R_PU(g) = π R_P(g) + R_U(g) - π R_P(g) × [correction term]
```

**nnPU (non-negative PU learning, Kiryo et al., 2017).** Addresses cases where the unbiased estimator becomes negative (which can cause training instability). Clips negative terms.

**Key assumption.** All PU methods assume the positive examples were selected uniformly from the true positive distribution (Selected Completely At Random, SCAR). If confirmed positives are not representative of all positives, the assumption fails.

## The pattern in action

**Drug-target interaction prediction.** A database contains 50,000 confirmed drug-target pairs (positive). The remaining drug-target space (millions of pairs) is unlabeled — some may interact, most do not.

Standard supervised learning: treat all unlabeled pairs as negative → the model learns that known databases are positive and everything else is negative, rather than learning the chemical interaction pattern.

PU learning approach:
1. Estimate the class prior π (fraction of drug-target pairs that actually interact) from the confirmed positives and domain knowledge
2. Use nnPU loss: train on P examples and unlabeled examples with the corrected risk estimator
3. The model learns the chemical interaction structure, not the database membership pattern

## Common traps

**Assuming unlabeled = negative.** This is the problem PU learning solves. Treating U as N produces a model biased against the phenomenon you are trying to detect.

**Not estimating the class prior.** PU methods require an estimate of π = P(Y=1). Misestimating π by 2× can significantly degrade performance. Use domain knowledge or estimate π from the confirmation rate of the labeling process.

**SCAR violation.** If confirmed positives were found by a process that preferentially finds a subset of all positives (e.g., a database that only records well-studied drug targets), the confirmed positives are not representative. Standard PU learning assumptions fail. The model will underperform on the novel positive patterns that are not in the database.

---

# Quick Diagnostics

**If a model has excellent offline metrics but fails in production:**

Consider data leakage first. Identify whether any feature uses future information relative to the prediction time, or whether preprocessing was fit on the full dataset. Construct a strict temporal train-test split and recompute metrics. If they drop substantially, leakage was the explanation.

**If asked to choose between XGBoost, LightGBM, and CatBoost:**

Ask: how large is the dataset, how many categorical features, how fast does retraining need to be? Large dataset with many categoricals → CatBoost. Large dataset needing fast iteration → LightGBM. Medium dataset needing a reliable interpretable baseline → XGBoost. For any choice, set early stopping on a held-out validation set to control overfitting.

**If a ranking model has good precision but the team reports users are unhappy with results:**

Check where relevant items appear in the list, not just whether they appear. P@K can be high while NDCG is low if relevant items appear at positions 8–10 rather than 1–3. Compute position-weighted metrics (NDCG, AP) and verify whether the user experience problem is a ranking order problem, not a retrieval problem.

## Rapid Recall

### Small batch sizes
- Direct Answer: statistics are noisy, training unstable
- Why: This matters because it tells you how to reason about small batch sizes.
- Pitfall: Don't answer "Small batch sizes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: statistics are noisy, training unstable

### Sequence models with variable-length sequences
- Direct Answer: batch statistics mix examples of different lengths
- Why: This matters because it tells you how to reason about sequence models with variable-length sequences.
- Pitfall: Don't answer "Sequence models with variable-length sequences" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: batch statistics mix examples of different lengths

### Test-time/inference
- Direct Answer: uses running statistics from training. If test distribution differs, these statistics are stale
- Why: This matters because it tells you how to reason about test-time/inference.
- Pitfall: Don't answer "Test-time/inference" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: uses running statistics from training. If test distribution differs, these statistics are stale

### Online learning or single-example inference
- Direct Answer: batch of 1 makes statistics undefined
- Why: This matters because it tells you how to reason about online learning or single-example inference.
- Pitfall: Don't answer "Online learning or single-example inference" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: batch of 1 makes statistics undefined

### Clusters are roughly spherical (equal variance in all directions)
- Direct Answer: Clusters are roughly spherical (equal variance in all directions)
- Why: This matters because it tells you how to reason about clusters are roughly spherical (equal variance in all directions).
- Pitfall: Don't answer "Clusters are roughly spherical (equal variance in all directions)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Clusters are roughly spherical (equal variance in all directions)

### Clusters are roughly equal in size
- Direct Answer: Clusters are roughly equal in size
- Why: This matters because it tells you how to reason about clusters are roughly equal in size.
- Pitfall: Don't answer "Clusters are roughly equal in size" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Clusters are roughly equal in size

### k is known
- Direct Answer: k is known
- Why: This matters because it tells you how to reason about k is known.
- Pitfall: Don't answer "k is known" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: k is known

### Clusters have arbitrary shape
- Direct Answer: Clusters have arbitrary shape
- Why: This matters because it tells you how to reason about clusters have arbitrary shape.
- Pitfall: Don't answer "Clusters have arbitrary shape" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Clusters have arbitrary shape

### Noise/outlier points exist (they should not be assigned to any cluster)
- Direct Answer: Noise/outlier points exist (they should not be assigned to any cluster)
- Why: This matters because it tells you how to reason about noise/outlier points exist (they should not be assigned to any cluster).
- Pitfall: Don't answer "Noise/outlier points exist (they should not be assigned to any cluster)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Noise/outlier points exist (they should not be assigned to any cluster)

### Clusters are defined by density, not distance to a centroid
- Direct Answer: Clusters are defined by density, not distance to a centroid
- Why: This matters because it tells you how to reason about clusters are defined by density, not distance to a centroid.
- Pitfall: Don't answer "Clusters are defined by density, not distance to a centroid" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Clusters are defined by density, not distance to a centroid

### Core point
- Direct Answer: has ≥ minPts points within radius ε
- Why: This matters because it tells you how to reason about core point.
- Pitfall: Don't answer "Core point" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: has ≥ minPts points within radius ε

### Border point
- Direct Answer: within ε of a core point but fewer than minPts neighbors
- Why: This matters because it tells you how to reason about border point.
- Pitfall: Don't answer "Border point" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: within ε of a core point but fewer than minPts neighbors

### Noise point
- Direct Answer: not within ε of any core point
- Why: This matters because it tells you how to reason about noise point.
- Pitfall: Don't answer "Noise point" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: not within ε of any core point

### User-based CF
- Direct Answer: find similar users, recommend what they liked
- Why: This matters because it tells you how to reason about user-based cf.
- Pitfall: Don't answer "User-based CF" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: find similar users, recommend what they liked

### Item-based CF
- Direct Answer: find similar items, recommend items similar to what the user liked
- Why: This matters because it tells you how to reason about item-based cf.
- Pitfall: Don't answer "Item-based CF" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: find similar items, recommend items similar to what the user liked

### Matrix factorization (SVD, ALS)
- Direct Answer: decompose the interaction matrix into user and item latent factors
- Why: This matters because it tells you how to reason about matrix factorization (svd, als).
- Pitfall: Don't answer "Matrix factorization (SVD, ALS)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: decompose the interaction matrix into user and item latent factors

### SVMs output margin distances, not probabilities. These are not calibrated.
- Direct Answer: SVMs output margin distances, not probabilities. These are not calibrated.
- Why: This matters because it tells you how to reason about svms output margin distances, not probabilities. these are not calibrated..
- Pitfall: Don't answer "SVMs output margin distances, not probabilities. These are not calibrated." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: SVMs output margin distances, not probabilities. These are not calibrated.

### Tree ensembles
- Direct Answer: boosted trees tend to be overconfident (scores near 0 or 1). Random forests tend to be underconfident (scores pulled toward 0.5 because of averaging).
- Why: This matters because it tells you how to reason about tree ensembles.
- Pitfall: Don't answer "Tree ensembles" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: boosted trees tend to be overconfident (scores near 0 or 1). Random forests tend to be underconfident (scores pulled toward 0.5 because of averaging).

### Neural networks
- Direct Answer: can be overconfident, especially on out-of-distribution examples.
- Why: This matters because it tells you how to reason about neural networks.
- Pitfall: Don't answer "Neural networks" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: can be overconfident, especially on out-of-distribution examples.

### The predicted probability is used directly in a decision (bid price in ad auction, risk score in medical triage)
- Direct Answer: The predicted probability is used directly in a decision (bid price in ad auction, risk score in medical triage)
- Why: This matters because it tells you how to reason about the predicted probability is used directly in a decision (bid price in ad auction, risk score in medical triage).
- Pitfall: Don't answer "The predicted probability is used directly in a decision (bid price in ad auction, risk score in medical triage)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: The predicted probability is used directly in a decision (bid price in ad auction, risk score in medical triage)

### Multiple models' scores are compared across populations (comparing fraud risk scores across geographies with different base rates)
- Direct Answer: Multiple models' scores are compared across populations (comparing fraud risk scores across geographies with different base rates)
- Why: This matters because it tells you how to reason about multiple models' scores are compared across populations (comparing fraud risk scores across geographies with different base rates).
- Pitfall: Don't answer "Multiple models' scores are compared across populations (comparing fraud risk scores across geographies with different base rates)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Multiple models' scores are compared across populations (comparing fraud risk scores across geographies with different base rates)

### GDPR-style right to explanation
- Direct Answer: "why was this loan denied?" requires trustworthy probability scores
- Why: This matters because it tells you how to reason about gdpr-style right to explanation.
- Pitfall: Don't answer "GDPR-style right to explanation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: "why was this loan denied?" requires trustworthy probability scores

### You only care about ranking (movie recommendation
- Direct Answer: top-10 items, not their exact probabilities)
- Why: This matters because it tells you how to reason about you only care about ranking (movie recommendation.
- Pitfall: Don't answer "You only care about ranking (movie recommendation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: top-10 items, not their exact probabilities)

### The downstream decision only uses the argmax (classification, not scoring)
- Direct Answer: The downstream decision only uses the argmax (classification, not scoring)
- Why: This matters because it tells you how to reason about the downstream decision only uses the argmax (classification, not scoring).
- Pitfall: Don't answer "The downstream decision only uses the argmax (classification, not scoring)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: The downstream decision only uses the argmax (classification, not scoring)

### Problem
- Direct Answer: only one evaluation split; variance in the performance estimate is high.
- Why: This matters because it tells you how to reason about problem.
- Pitfall: Don't answer "Problem" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: only one evaluation split; variance in the performance estimate is high.

### Split 1
- Direct Answer: train on months 1–6, test on month 7
- Why: This matters because it tells you how to reason about split 1.
- Pitfall: Don't answer "Split 1" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: train on months 1–6, test on month 7

### Split 2
- Direct Answer: train on months 1–7, test on month 8
- Why: This matters because it tells you how to reason about split 2.
- Pitfall: Don't answer "Split 2" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: train on months 1–7, test on month 8

### Split 3
- Direct Answer: train on months 1–8, test on month 9
- Why: This matters because it tells you how to reason about split 3.
- Pitfall: Don't answer "Split 3" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: train on months 1–8, test on month 9

### ...
- Direct Answer: ...
- Why: This matters because it tells you how to reason about ....
- Pitfall: Don't answer "..." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: ...

### Split 1
- Direct Answer: train on months 1–6, test on month 7
- Why: This matters because it tells you how to reason about split 1.
- Pitfall: Don't answer "Split 1" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: train on months 1–6, test on month 7

### Split 2
- Direct Answer: train on months 2–7, test on month 8
- Why: This matters because it tells you how to reason about split 2.
- Pitfall: Don't answer "Split 2" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: train on months 2–7, test on month 8

### ...
- Direct Answer: ...
- Why: This matters because it tells you how to reason about ....
- Pitfall: Don't answer "..." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: ...

### Captures the real contribution of a feature in the model's context
- Direct Answer: Captures the real contribution of a feature in the model's context
- Why: This matters because it tells you how to reason about captures the real contribution of a feature in the model's context.
- Pitfall: Don't answer "Captures the real contribution of a feature in the model's context" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Captures the real contribution of a feature in the model's context

### Measures importance for the specific model's decision function
- Direct Answer: Measures importance for the specific model's decision function
- Why: This matters because it tells you how to reason about measures importance for the specific model's decision function.
- Pitfall: Don't answer "Measures importance for the specific model's decision function" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Measures importance for the specific model's decision function

### Fast to compute
- Direct Answer: Fast to compute
- Why: This matters because it tells you how to reason about fast to compute.
- Pitfall: Don't answer "Fast to compute" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Fast to compute

### Aggregates across all examples (global, not per-example)
- Direct Answer: Aggregates across all examples (global, not per-example)
- Why: This matters because it tells you how to reason about aggregates across all examples (global, not per-example).
- Pitfall: Don't answer "Aggregates across all examples (global, not per-example)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Aggregates across all examples (global, not per-example)

### Satisfies local accuracy
- Direct Answer: feature contributions sum to the prediction
- Why: This matters because it tells you how to reason about satisfies local accuracy.
- Pitfall: Don't answer "Satisfies local accuracy" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: feature contributions sum to the prediction

### Handles correlated features by averaging over all possible subsets
- Direct Answer: Handles correlated features by averaging over all possible subsets
- Why: This matters because it tells you how to reason about handles correlated features by averaging over all possible subsets.
- Pitfall: Don't answer "Handles correlated features by averaging over all possible subsets" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Handles correlated features by averaging over all possible subsets

### Provides per-example explanations (local interpretability)
- Direct Answer: Provides per-example explanations (local interpretability)
- Why: This matters because it tells you how to reason about provides per-example explanations (local interpretability).
- Pitfall: Don't answer "Provides per-example explanations (local interpretability)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Provides per-example explanations (local interpretability)

### Computationally expensive for exact computation (O(2^d)); approximated via tree structures for tree models or kernel methods for black-box models
- Direct Answer: Computationally expensive for exact computation (O(2^d)); approximated via tree structures for tree models or kernel methods for black-box models
- Why: This matters because it tells you how to reason about computationally expensive for exact computation (o(2^d)); approximated via tree structures for tree models or kernel methods for black-box models.
- Pitfall: Don't answer "Computationally expensive for exact computation (O(2^d)); approximated via tree structures for tree models or kernel methods for black-box models" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Computationally expensive for exact computation (O(2^d)); approximated via tree structures for tree models or kernel methods for black-box models

### Poor calibration (probability 0.99 when correct probability is 0.8)
- Direct Answer: Poor calibration (probability 0.99 when correct probability is 0.8)
- Why: This matters because it tells you how to reason about poor calibration (probability 0.99 when correct probability is 0.8).
- Pitfall: Don't answer "Poor calibration (probability 0.99 when correct probability is 0.8)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Poor calibration (probability 0.99 when correct probability is 0.8)

### Poor generalization (fitting training noise)
- Direct Answer: Poor generalization (fitting training noise)
- Why: This matters because it tells you how to reason about poor generalization (fitting training noise).
- Pitfall: Don't answer "Poor generalization (fitting training noise)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Poor generalization (fitting training noise)

### Brittle decisions near the decision boundary
- Direct Answer: Brittle decisions near the decision boundary
- Why: This matters because it tells you how to reason about brittle decisions near the decision boundary.
- Pitfall: Don't answer "Brittle decisions near the decision boundary" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Brittle decisions near the decision boundary

### Improved calibration
- Direct Answer: model outputs are less extreme
- Why: This matters because it tells you how to reason about improved calibration.
- Pitfall: Don't answer "Improved calibration" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: model outputs are less extreme

### Better generalization
- Direct Answer: prevents memorizing noisy labels
- Why: This matters because it tells you how to reason about better generalization.
- Pitfall: Don't answer "Better generalization" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: prevents memorizing noisy labels

### Improved robustness to adversarial examples (less sensitivity to input perturbations)
- Direct Answer: Improved robustness to adversarial examples (less sensitivity to input perturbations)
- Why: This matters because it tells you how to reason about improved robustness to adversarial examples (less sensitivity to input perturbations).
- Pitfall: Don't answer "Improved robustness to adversarial examples (less sensitivity to input perturbations)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Improved robustness to adversarial examples (less sensitivity to input perturbations)

### P@K
- Direct Answer: when you care about precision in the top K (news headline ranking)
- Why: This matters because it tells you how to reason about p@k.
- Pitfall: Don't answer "P@K" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: when you care about precision in the top K (news headline ranking)

### R@K
- Direct Answer: when recall is critical (medical literature retrieval)
- Why: This matters because it tells you how to reason about r@k.
- Pitfall: Don't answer "R@K" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: when recall is critical (medical literature retrieval)

### NDCG
- Direct Answer: when relevance is graded and position matters (search ranking)
- Why: This matters because it tells you how to reason about ndcg.
- Pitfall: Don't answer "NDCG" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: when relevance is graded and position matters (search ranking)

### MAP
- Direct Answer: when binary relevance across multiple queries (document retrieval benchmarks)
- Why: This matters because it tells you how to reason about map.
- Pitfall: Don't answer "MAP" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: when binary relevance across multiple queries (document retrieval benchmarks)

### P
- Direct Answer: labeled positive examples (confirmed true)
- Why: This matters because it tells you how to reason about p.
- Pitfall: Don't answer "P" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: labeled positive examples (confirmed true)

### U: unlabeled examples
- Direct Answer: some are positive, some are negative, but you do not know which
- Why: This matters because it tells you how to reason about u: unlabeled examples.
- Pitfall: Don't answer "U: unlabeled examples" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: some are positive, some are negative, but you do not know which

### Fraud detection: you have confirmed fraud cases (P) and unreviewed transactions (U
- Direct Answer: some are fraud, most are not)
- Why: This matters because it tells you how to reason about fraud detection: you have confirmed fraud cases (p) and unreviewed transactions (u.
- Pitfall: Don't answer "Fraud detection: you have confirmed fraud cases (P) and unreviewed transactions (U" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: some are fraud, most are not)

### Medical screening
- Direct Answer: you have confirmed diagnoses (P) and unscreened patients (U)
- Why: This matters because it tells you how to reason about medical screening.
- Pitfall: Don't answer "Medical screening" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: you have confirmed diagnoses (P) and unscreened patients (U)

### Citation networks
- Direct Answer: you have known positive links (P) and the absence of links, which is not confirmation of non-linkage
- Why: This matters because it tells you how to reason about citation networks.
- Pitfall: Don't answer "Citation networks" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: you have known positive links (P) and the absence of links, which is not confirmation of non-linkage
