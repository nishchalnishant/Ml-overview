# Additional ML Interview Topics

These questions often show up when an interviewer wants to see whether you can go beyond definitions and talk about failure modes, tradeoffs, and production behavior.

---

# Q1: What is data leakage, and how do you detect and prevent it?

**Interview-ready answer**

Data leakage happens when information that would not be available at prediction time is used during training or validation, causing offline metrics to look unrealistically good. In practice, leakage often comes from future information, target-derived features, preprocessing done before the train-validation split, or duplicated entities across splits. The right way to answer this in an interview is to say that leakage is an evaluation bug first and a modeling issue second, because it invalidates your estimate of generalization.

**Go deeper if asked**

- Temporal leakage: using data recorded after the prediction point.
- Target leakage: features that directly encode or proxy the label.
- Split leakage: the same user, device, patient, or household appearing in both train and validation.
- Pipeline leakage: fitting imputers, scalers, or encoders on the full dataset before splitting.

**How to prevent it**

- Define the prediction timestamp and feature availability clearly.
- Build preprocessing inside the train-only pipeline.
- Use entity-aware or time-aware splits when needed.
- Sanity-check suspiciously high offline metrics and feature importance.

---

# Q2: Explain SMOTE. When does it help, and what are the risks?

**Interview-ready answer**

SMOTE creates synthetic minority-class examples by interpolating between nearby minority points. It can help when the minority class is small and the decision boundary is underrepresented, especially for tabular problems. But it is not automatically a good idea: if the minority class is noisy, overlapping, or multi-modal in a complicated way, SMOTE can create unrealistic examples and blur the boundary rather than improve it.

**When it helps**

- Minority class is genuinely under-sampled
- Feature space is meaningful under interpolation
- You apply it only to the training set

**Risks**

- Can amplify noise or outliers
- Often fails on categorical-heavy or highly sparse spaces
- Can distort class priors and mislead threshold selection

**Good interview nuance**

Say that you would compare SMOTE with class weighting, focal loss, better thresholds, and better metrics before committing to it.

---

# Q3: Batch Normalization vs Layer Normalization - when do you use each?

**Interview-ready answer**

Batch normalization normalizes activations using statistics computed across the batch, while layer normalization normalizes across features within each example. BatchNorm works especially well in CNN-style settings with reasonably large batches and has a regularizing effect, but it becomes less reliable with very small or highly variable batch sizes. LayerNorm is preferred in transformers and many sequence models because it does not depend on batch statistics and behaves consistently at training and inference time.

**Practical comparison**

- BatchNorm: common in CNNs, sensitive to batch size, keeps running statistics.
- LayerNorm: common in transformers, stable for variable-length sequences and micro-batches.
- If batches are tiny or distributed in awkward ways, LayerNorm is usually easier to reason about.

**Common pitfall**

Do not say BatchNorm is always better because it speeds up training. The right choice depends heavily on architecture and batch regime.

---

# Q4: Compare XGBoost, LightGBM, and CatBoost in one interview-ready answer.

**Interview-ready answer**

All three are gradient-boosted tree libraries, but they make different engineering tradeoffs. XGBoost is the most general and battle-tested choice, with strong regularization and good defaults for many structured-data problems. LightGBM is usually faster and more memory-efficient on very large datasets because it uses histogram-based learning and leaf-wise tree growth, but that same aggressiveness can overfit if not controlled. CatBoost is especially strong when you have many categorical features because it handles them natively and uses ordered statistics to reduce target leakage during encoding.

**When to choose which**

- XGBoost: strong all-purpose baseline
- LightGBM: very large tabular datasets, speed-sensitive training
- CatBoost: many categorical variables, minimal preprocessing, robust defaults

**Good nuance**

The best answer is rarely "library X is best." It is "for this data shape and deployment context, I would start with ..."

---

# Q5: DBSCAN vs k-means - when do you choose which?

**Interview-ready answer**

K-means assumes roughly spherical clusters and requires you to choose `k` in advance. It is fast and works well when clusters are compact, similar in scale, and the notion of Euclidean center is meaningful. DBSCAN instead groups points by density, can find arbitrarily shaped clusters, and labels low-density points as noise, which makes it attractive when outliers matter. The tradeoff is that DBSCAN is sensitive to scale and hyperparameters such as `eps` and `min_samples`, and it can struggle in high dimensions where distance becomes less informative.

**Rule of thumb**

- Use k-means for simple, scalable partitioning.
- Use DBSCAN when you care about noise detection or non-spherical structure.

**Common pitfall**

Do not use either blindly in high-dimensional raw feature spaces without thinking about scaling or representation quality first.

---

# Q6: Collaborative filtering vs content-based recommendation.

**Interview-ready answer**

Collaborative filtering recommends items based on interaction patterns across users and items, while content-based systems recommend items that are similar in attributes to what the user already liked. Collaborative filtering is powerful because it can learn subtle taste signals that are not explicitly encoded in metadata, but it suffers from cold start for new users and items. Content-based systems are easier to bootstrap and explain, but they can become narrow and over-specialized because they only recommend "more of the same."

**What strong candidates add**

- Most real systems combine both.
- Collaborative methods need enough interaction data.
- Content-based methods depend heavily on feature quality.
- Retrieval and ranking are often separate stages.

---

# Q7: What is model calibration, and how do you fix miscalibration?

**Interview-ready answer**

Calibration means the predicted probability matches empirical reality. If a model assigns a score of 0.8 to a set of examples, then roughly 80 percent of those examples should actually be positive. A model can have good ranking performance and still be poorly calibrated, which matters in domains like fraud, medical triage, and bidding systems where decisions depend on probability estimates rather than only ranking.

**How to diagnose and fix it**

- Use reliability plots, expected calibration error, and Brier score.
- Common fixes include Platt scaling, isotonic regression, and temperature scaling.
- Recheck calibration after distribution shift or retraining because it can drift over time.

**Common pitfall**

Do not assume softmax probabilities are well calibrated. They are often overconfident.

---

# Q8: How should you validate time-series models differently from i.i.d. data?

**Interview-ready answer**

Time-series validation must respect chronology. You should train on the past and validate on the future, because random splits leak future information and create overly optimistic results. A strong answer mentions rolling or expanding window backtesting, feature generation that only uses historical data, and awareness of forecast horizon, seasonality, and label delay.

**Good points to mention**

- Match the validation setup to the production forecasting setup.
- Use multiple backtest windows, not a single split, if the series is long enough.
- Be explicit about exogenous features and whether they are known at prediction time.
- Evaluate by horizon, because short-horizon and long-horizon quality can differ sharply.

---

# Q9: How are KL divergence and cross-entropy related in classification?

**Interview-ready answer**

Cross-entropy between the true distribution `p` and the model distribution `q` can be written as `H(p, q) = H(p) + KL(p || q)`. Since `H(p)` does not depend on the model, minimizing cross-entropy with respect to the model is equivalent to minimizing the KL divergence from the true distribution to the model distribution. That is why cross-entropy is the standard loss for classification: it encourages the predicted distribution to get close to the target distribution.

**Good interview nuance**

- With one-hot labels, cross-entropy reduces to penalizing the log probability assigned to the true class.
- With soft targets, label smoothing, or distillation, the full distributional view becomes more important.

**Common pitfall**

KL divergence is not symmetric, so `KL(p || q)` and `KL(q || p)` behave differently.

---

# Q10: Permutation importance vs SHAP - what's the difference?

**Interview-ready answer**

Permutation importance measures how much model performance drops when a feature is randomly shuffled, so it is a global, model-agnostic way to estimate feature usefulness. SHAP assigns contributions to individual predictions using a game-theoretic formulation, so it gives both local and global explanations. In interviews, the key distinction is that permutation importance tells you how much the model depends on a feature overall, while SHAP tries to explain how a prediction was formed for a specific example.

**Tradeoffs**

- Permutation importance is simpler and cheaper but can be unstable with correlated features.
- SHAP is richer but more computationally expensive and can still be misleading if the feature dependence assumptions are unrealistic.
- Neither replaces careful causal reasoning.

---

# Q11: What is label smoothing, and why is it used in deep classification?

**Interview-ready answer**

Label smoothing replaces hard one-hot targets like `[0, 0, 1, 0]` with slightly softened targets such as `[0.025, 0.025, 0.925, 0.025]`. The purpose is to discourage the model from becoming overly confident, which can improve generalization, calibration, and robustness. It is especially useful in large classifiers where the exact label may be noisy or the task has some ambiguity.

**What to mention if pushed**

- It acts like a regularizer on the output distribution.
- It can improve teacher-student distillation and large-vocabulary classification.
- Too much smoothing can hurt accuracy or make the model under-confident.

---

# Q12: What is the difference between precision@k, recall@k, and MAP?

**Interview-ready answer**

These are ranking metrics used when only the top portion of a ranked list matters. Precision@k asks what fraction of the top `k` results are relevant. Recall@k asks how many of the relevant results were recovered within the top `k`. Mean Average Precision, or MAP, goes further by rewarding systems that place relevant items earlier in the ranked list across queries or users, so it captures ranking quality more fully than a single cutoff metric.

**Good nuance**

- Precision@k is useful when screen space is limited.
- Recall@k matters when coverage of relevant items matters.
- MAP is more sensitive to ordering and is often preferred when there are multiple relevant items per query.

---

# Q13: What is positive-unlabeled (PU) learning?

**Interview-ready answer**

PU learning is the setting where you have confirmed positive examples and a large pool of unlabeled examples, but no reliable negative labels. The challenge is that unlabeled data is a mixture of true negatives and hidden positives. This shows up in domains like fraud, recommendation, and disease detection, where absence of a positive label does not mean absence of the condition.

**How people handle it**

- Treat unlabeled data cautiously rather than as clean negatives.
- Use biased learning, two-step methods, or class-prior estimation.
- Often combine with anomaly detection, ranking, or active review.

**Common pitfall**

If you train as though all unlabeled examples are negative, you usually build a biased model and underestimate recall.
