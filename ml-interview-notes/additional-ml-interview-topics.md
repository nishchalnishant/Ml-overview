# Additional ML Interview Topics

This is the "good, you know the basics, now let's see if you actually think like someone who ships" section.

These are the questions that separate:

- textbook confidence

from:

- production judgment

Which is exactly where you want to win.

---

# 1. Data Leakage

Leakage is one of the most expensive fake wins in ML.

It happens when the model gets access to information it would not truly have at prediction time.

That makes offline metrics look fabulous.
Then production arrives and the glamour evaporates.

## Common types

- temporal leakage
- target leakage
- pipeline leakage

## Short interview answer

Data leakage happens when future or target-derived information sneaks into training or validation, making model performance look much better than it will be in the real world.

**Azure/DevOps analogy**

It is like accidentally testing your deployment with production secrets and then acting surprised when staging looked perfect.

---

# 2. SMOTE

SMOTE creates synthetic minority-class points by interpolating between nearby minority examples.

Why people use it:

- rare class
- poor minority representation
- need more signal than plain duplication

Why you should be careful:

- it can blur class boundaries
- it can amplify noise
- it may create unrealistic points in high-dimensional spaces

**Short answer**

SMOTE can help when the minority class is underrepresented, but it should be used carefully because synthetic points can distort the true decision boundary.

---

# 3. BatchNorm vs LayerNorm

This one shows up a lot in deep learning interviews.

## BatchNorm

Normalizes across the batch.

Best for:

- CNN-heavy settings
- stable batch sizes

## LayerNorm

Normalizes across features within an example.

Best for:

- Transformers
- variable-length sequence settings
- small or unstable batch regimes

**Short answer**

BatchNorm depends on batch statistics, which works well in vision pipelines, while LayerNorm is batch-independent and therefore more stable for Transformers and sequence models.

---

# 4. XGBoost vs LightGBM vs CatBoost

These are all boosting libraries, but they have different personalities.

## XGBoost

- sturdy default
- well understood
- strong regularization

## LightGBM

- faster on large datasets
- leaf-wise growth
- great efficiency
- can overfit faster if you are careless

## CatBoost

- excellent categorical handling
- reduces target leakage issues with ordered boosting
- especially attractive when categorical features are a big part of the problem

**Short answer**

XGBoost is the classic dependable baseline, LightGBM is the speed-focused scale machine, and CatBoost is especially attractive when categorical features are central.

---

# 5. DBSCAN vs K-Means

Both are clustering methods, but they think differently.

## K-Means

- assumes roughly spherical clusters
- needs `k`
- fast and common

## DBSCAN

- density-based
- finds arbitrary cluster shapes
- identifies noise points
- does not need `k`

Why DBSCAN can struggle:

- sensitive to scale
- sensitive to parameter choice
- weak in high-dimensional spaces

**Short answer**

Use K-Means for fast partitioning with reasonably compact clusters; use DBSCAN when you care about density structure and noise detection.

---

# 6. Collaborative Filtering vs Content-Based Recommendation

## Collaborative Filtering

Uses user-item interaction behavior.

Strength:

- learns taste patterns from collective behavior

Weakness:

- cold start pain

## Content-Based

Uses item attributes.

Strength:

- easier bootstrap
- understandable recommendations

Weakness:

- can become narrow and repetitive

**Fashion analogy**

Collaborative filtering says:

> "People who loved this lehenga also liked these styles."

Content-based says:

> "This looks similar in silhouette, fabric, and vibe, so you may like it too."

Both are useful.
Most real systems mix them.

---

# 7. Model Calibration

Calibration means predicted probabilities line up with reality.

If the model says:

- 80% chance

then roughly 80 out of 100 such cases should actually be positive.

Why this matters:

- fraud scoring
- medical triage
- ad bidding
- risk systems

Good ranking is not enough if probability quality matters.

**Common fixes**

- Platt scaling
- isotonic regression
- temperature scaling

---

# 8. Time-Series Validation

Time-series validation should respect time order.

That means:

- train on past
- validate on future

Not random splitting.

Because random splitting leaks future information backward.

**Short answer**

Time-series models should be validated with rolling or forward-looking splits so the offline setup mirrors the real forecasting setup.

---

# 9. KL Divergence vs Cross-Entropy

These are closely related.

Cross-entropy can be seen as:

- true distribution entropy
- plus KL divergence between truth and model prediction

That is why minimizing cross-entropy also pushes the predicted distribution closer to the target distribution.

You do not need to overperform here.

Just say the connection clearly.

---

# 10. Permutation Importance vs SHAP

## Permutation Importance

Measures how much performance drops when a feature is shuffled.

Good for:

- simple global importance view

## SHAP

Assigns contribution values to features for individual predictions.

Good for:

- local explanation
- richer model interpretation

Caveat:

Neither one is causal truth.

That sentence is important.

---

# 11. Label Smoothing

Label smoothing softens one-hot targets a little.

Why do that?

Because it discourages extreme overconfidence and can improve:

- generalization
- calibration
- robustness

It is a small trick, but a useful one.

---

# 12. Precision@K, Recall@K, MAP

These matter in ranking and recommendation.

## Precision@K

Of the top `k` results, how many are relevant?

## Recall@K

Of all relevant items, how many appeared in the top `k`?

## MAP

Rewards not just getting relevant items in the list, but getting them high in the list.

That ordering sensitivity is the key difference.

---

# 13. PU Learning

PU means Positive-Unlabeled learning.

You know some positives.
Everything else is unlabeled, not confirmed negative.

That is common in:

- fraud
- abuse detection
- medical screening

The key challenge:

unlabeled does not mean negative.

That is the whole trick.

---

# Quick Thought Experiment

A model has amazing offline performance after target encoding a high-cardinality feature.

What should you worry about first?

- leakage
- overfitting
- validation setup

If your answer is "all three," excellent.

That is exactly the right level of suspicion.
