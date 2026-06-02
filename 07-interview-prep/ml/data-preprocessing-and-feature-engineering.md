---
module: Interview Prep
topic: Ml
subtopic: Data Preprocessing And Feature Engineering
status: unread
tags: [interviewprep, ml, ml-data-preprocessing-and-feat]
---
# Data Preprocessing and Feature Engineering

---

## 1. Why Preprocessing Comes First

**What the interviewer is testing**: whether you approach data problems by asking "what does this data look like and what could go wrong?" before choosing algorithms.

**The reasoning structure**: the model can only learn patterns that exist in the features you provide. If features are encoded incorrectly, scaled inappropriately, contaminated with future information, or missing without acknowledgment, the model is not just suboptimal — it might appear to work during training and fail silently in production.

The correct mental order is: understand the data → fix its representation → prevent leakage → engineer useful features → then choose a model. Every hour spent here saves many hours of unexplained model failures later.

The single most important question is not "how should I impute this?" but: "what is the data-generating process, and does this feature behave the same way during training as it does at inference time?" A feature that encodes information from the future relative to the prediction moment is leakage, not a feature. No amount of model tuning recovers from that.

**Common traps**:
- jumping to model selection before auditing the data — the fastest path to production is understanding why the data looks the way it does; a week of data investigation can prevent months of debugging
- treating preprocessing as a one-time step — preprocessing logic must be implemented identically in the training pipeline and the inference pipeline, version-controlled alongside the model, and monitored for drift

---

## 2. Feature Scaling

**What the interviewer is testing**: whether you know which algorithms require scaling and why, not just that "you should normalize your features."

**The reasoning structure**: scaling matters when the algorithm's loss function or decision rule is sensitive to feature magnitude. A feature measuring income (range 0–500,000) and a feature measuring age (range 0–100) are on completely different scales. Algorithms that compute distances or sum weighted feature values will treat income as much more important purely because of its larger numerical range — regardless of whether income is actually more informative.

Decision trees and their ensembles (Random Forest, XGBoost) are unaffected by scaling because splits depend only on rank order of feature values, not magnitude. Everything else — linear models, SVMs, KNN, neural networks, PCA — is affected.

**The pattern in action**: "I train KNN on unscaled features and get 72% accuracy. I apply StandardScaler and accuracy jumps to 86%. The income feature (range 0–500,000) was dominating all distance computations, making KNN effectively use only income as its decision rule. After scaling, all features contribute in proportion to their actual information content."

| Scaler | Formula | When to use |
| :--- | :--- | :--- |
| StandardScaler | $z = (x - \mu) / \sigma$ | Most cases; assumes roughly normal distribution |
| MinMaxScaler | $z = (x - x_{\min}) / (x_{\max} - x_{\min})$ | When bounded [0,1] output is needed; sensitive to outliers |
| RobustScaler | $z = (x - \text{median}) / \text{IQR}$ | When outliers are present but should not dominate |

**Critical rule**: fit scalers on training data only. Apply the fitted scaler to training, validation, and test sets.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_val_scaled   = scaler.transform(X_val)         # transform only
X_test_scaled  = scaler.transform(X_test)        # transform only
```

**Common traps**:
- fitting the scaler on the full dataset before splitting — this leaks test set distribution information (mean and variance of test features) into the training pipeline; it is a subtle form of data leakage
- applying scaling to tree-based models expecting improvement — scaling changes nothing for trees and adds unnecessary pipeline complexity

---

## 3. Encoding Categorical Features

**What the interviewer is testing**: whether you can choose an encoding strategy based on cardinality and leakage risk, not just default to "use one-hot encoding."

**The reasoning structure**: most ML algorithms require numeric inputs. The encoding must reflect the right relationships between categories. One-hot encoding says all categories are equally different from each other. Target encoding uses the category's relationship to the target. Ordinal encoding imposes an ordered numeric scale.

Cardinality determines which encoding is practical. Low cardinality (5 categories): one-hot is fine, interpretable, no leakage risk. High cardinality (10,000 unique cities): one-hot creates 10,000 sparse columns, most near-zero; target encoding produces one compact numeric column.

**The pattern in action**:
- Color (red/blue/green): one-hot → 3 binary columns
- City (10,000 unique values): target encoding with smoothing → 1 numeric column per city

**Smoothed target encoding formula**:
$$\text{enc}(c) = \lambda_c \cdot \bar{y}_c + (1 - \lambda_c) \cdot \bar{y}_{\text{global}}, \quad \lambda_c = \frac{n_c}{n_c + k}$$

When a city has few examples ($n_c$ small), $\lambda_c$ is small and the encoding shrinks toward the global mean — preventing wild estimates from rare categories. When $n_c$ is large, the category mean dominates.

**Common traps**:
- target encoding on the full dataset — the category mean is computed using the labels you are trying to predict; this is direct label leakage. Always use out-of-fold target encoding: compute statistics on training folds, apply to held-out folds
- not handling unseen categories at inference time — a city not seen during training has no stored encoding; you need a fallback (global mean, a special "unseen" token, or a default value)
- using ordinal encoding (1, 2, 3...) for nominal categories — this incorrectly encodes the assumption that the categories have a meaningful numeric ordering when they do not

---

## 4. Missing Values

**What the interviewer is testing**: whether you investigate why values are missing before deciding how to handle them — the mechanism of missingness determines the right strategy.

**The reasoning structure**: the question before imputation is: why is this value missing? The answer changes the strategy fundamentally.

- **MCAR (Missing Completely at Random)**: missingness is independent of both observed and unobserved values. Row deletion or simple imputation is usually safe.
- **MAR (Missing At Random)**: missingness depends on other observed features (e.g., older customers are less likely to fill in income). Model-based imputation using the observed features can recover the true distribution.
- **MNAR (Missing Not at Random)**: missingness depends on the unobserved value itself (e.g., high-income customers leave the income field blank). No imputation method can fully recover this — the missingness itself is informative.

For MNAR, the right response is to impute the value and also add a binary missing-indicator feature. This allows the model to learn two things separately: the value's effect when present, and the effect of the value being absent.

**The pattern in action**: "My dataset has a 'salary' field blank for 40% of users. I check: are the blanks random? No — blanks correlate with age. This is MAR. I use KNN imputation, leveraging age and other observed features to estimate salary. I also add `salary_is_missing` as a binary feature, because the act of not filling in salary may itself be predictive — it might indicate self-employment, privacy concerns, or a specific demographic."

**Common traps**:
- treating all missingness as MCAR and applying mean imputation — mean imputation is only appropriate for MCAR; for MAR and MNAR it introduces bias by ignoring the dependence between missingness and other variables
- dropping rows with missing values without checking what fraction of positive examples are lost — in imbalanced datasets, missingness may be correlated with the positive class; dropping rows can dramatically change your effective class distribution
- fitting the imputer on the full dataset before splitting — fit the imputer (its mean, its KNN model) only on training data; apply it to validation and test using training statistics only

---

## 5. Handling Skewed Data

**What the interviewer is testing**: whether you understand when skewness is a problem for specific algorithms and what the appropriate transformation is.

**The reasoning structure**: skewed distributions cause problems in two ways. First, for gradient-based models, a feature with a long tail can dominate updates — the extremely large values create disproportionately large gradients. Second, for assumptions-based methods, skewness violates normality assumptions.

Log transform is the most common fix for right-skewed data with strictly positive values (income, price, count data). It compresses the long right tail and expands the dense left region. For data with zeros or negative values, use $\log(x + 1)$ or a Yeo-Johnson transform (handles all real values).

**The pattern in action**: "My house price regression model has a severely right-skewed target — most houses are $100k–$500k but a few are $5M+. Training with raw prices means the expensive houses dominate the loss function; the model spends most of its capacity getting those right. After log-transforming the target, the distribution is approximately normal, gradients are more balanced across the price range, and overall performance improves. When reporting predictions, I exponentiate them back to dollar terms."

**Common traps**:
- log-transforming a feature and forgetting the inverse transform at prediction time — if you model log(price), predictions are in log-space; you must exponentiate before reporting to users
- applying log transform to features with zeros — $\log(0)$ is undefined; use $\log(x + 1)$ instead, or apply RobustScaler if the skewness is driven by outliers rather than the scale of the distribution

---

## 6. Outliers

**What the interviewer is testing**: whether you investigate outliers before deciding to remove them, understanding that outliers can be exactly the signal you care about.

**The reasoning structure**: outliers have two distinct origins. The first is measurement error or data corruption — a sensor reading of -9999 meaning "sensor failure," not an actual value. These should be removed or imputed. The second is rare but valid behavior — a $500,000 transaction in a fraud dataset, a 0.01% error rate in system monitoring. These may be exactly what you are trying to detect.

Automatically removing outliers without domain investigation risks removing your signal. The diagnostic question: "is this value impossible, or just rare?"

**The pattern in action**: "My transaction dataset has amounts from $0.01 to $1M. The top 0.1% of transactions exceed $50,000. If I build a fraud model and remove these as 'outliers,' I remove the examples where fraud is most common — large wire transfers. The model will then have no representation of high-value fraud. I investigate: are these data errors? No — they are legitimate high-value transactions. I keep them and use RobustScaler or log-transform to prevent them from dominating gradient computations while preserving their information content."

**Common traps**:
- using IQR-based outlier removal blindly without understanding domain — a value being statistically extreme does not mean it is wrong; the IQR test is a detection heuristic, not a ground truth for validity
- in fraud and anomaly detection, the outlier is often the target — removing statistical outliers in these domains destroys the very signal you are trying to model

---

## 7. Feature Engineering

**What the interviewer is testing**: whether you understand feature engineering as expressing domain knowledge in a form the model can use, not just as "inventing new columns."

**The reasoning structure**: the model can only discover patterns that are representable given the features you provide. If a linear model needs to learn that "weekend AND beach location" jointly predicts high demand, but you provide only `day_of_week` and `location` separately, it cannot discover the interaction. Feature engineering bridges the gap between raw features and the representation the model needs to succeed.

The most valuable feature engineering insights come from domain knowledge: "transaction amount at 3am is more suspicious than the same amount at noon" → add `transaction_hour` and an `amount × is_late_night` interaction. "The ratio of clicks to views is more predictive than either individually" → add `click_rate`.

**Feature crosses**: create explicit interaction terms. $f_1 \times f_2$ captures interactions that additive models cannot represent. Critical for linear models that cannot learn non-linear interactions automatically.

**Temporal features from timestamps**: day of week, hour, month, quarter, `is_weekend`, days-since-last-event. A raw Unix timestamp integer is almost never useful directly.

**Ratio features**: the ratio of two quantities is often more informative than either absolute value. Debt-to-income ratio, error rate, click-through rate, spend per session.

**The pattern in action**: "I have a fraud detection dataset with `transaction_amount` and `account_balance`. Neither alone is very predictive. The ratio `transaction_amount / account_balance` is highly predictive — large transactions relative to account balance are suspicious. This feature, derivable from two existing columns, substantially improves the model."

**Common traps**:
- engineering features that are not available at inference time — if you use the target variable or post-event information in feature construction, that is leakage; the feature must be computable using only information available at the moment of prediction
- not validating that engineered features are stable across time — a feature predictive in historical data may stop being predictive when underlying behavior shifts; monitor feature distributions and predictive power in production

---

## 8. Leakage: The Silent Killer of ML Models

**What the interviewer is testing**: whether you can identify subtle forms of leakage beyond the obvious "don't include the target as a feature."

**The reasoning structure**: leakage is when your training data contains information about the target that will not be available at inference time. The model learns to use this information, appears to perform well in offline evaluation, and then fails in production.

The most common forms:
1. **Temporal leakage**: using information from the future relative to the prediction time. Training a churn model using features computed after the churn event.
2. **Preprocessing leakage**: fitting the scaler/imputer/encoder on the full dataset before splitting, so the preprocessing knows the test set distribution.
3. **Target leakage**: including a feature causally downstream of the target. E.g., `days_since_last_support_call` in a churn model, when customers call support after they have already decided to churn.
4. **Train/test contamination**: any step that crosses the train/test boundary — feature selection, normalization, or encoding based on the full dataset.

**The pattern in action**: "My churn model achieves 97% AUC in offline evaluation. I check feature importances and find `last_support_resolution_time` has importance 0.8. I investigate: when is this field populated? It is populated when a customer closes a support ticket — which often happens the same day they submit a cancellation request. The model is detecting that a customer has already begun cancellation, not predicting churn before it happens. I remove the feature and AUC drops to 82% — the honest number."

**The correct preprocessing order**:
1. Split data (time-based for temporal data, stratified random otherwise)
2. Fit all preprocessing (scalers, imputers, encoders) on training data only
3. Apply fitted transforms to validation and test sets using training statistics

**Common traps**:
- using cross-validation without re-fitting preprocessing inside each fold — this leaks validation fold statistics into the preprocessing fit; the scaler has already seen the validation examples
- performing feature selection on the full dataset before splitting — the selected features are correlated with the test labels, inflating all downstream validation estimates

---

## 9. Feature Selection

**What the interviewer is testing**: whether you understand the difference between removing noise, removing redundancy, and removing features unavailable at inference.

**The reasoning structure**: adding irrelevant features hurts models in specific ways — they add noise to tree splitting decisions, create multicollinearity in linear models, increase the curse of dimensionality for distance-based methods, and increase training time with no benefit. Feature selection removes features that do not add predictive value.

Three main approaches:
- **Filter methods**: score each feature independently of the model (correlation with target, mutual information, chi-squared). Fast but ignore feature interactions.
- **Wrapper methods**: train the model on different feature subsets and select the best-performing subset. Expensive but finds the optimal subset for the specific model.
- **Embedded methods**: regularization during training selects features (L1/Lasso zeroes out unimportant features; XGBoost feature importance provides rankings).

**The pattern in action**: "I have 500 features. I apply L1 regularization and it zeroes out 480. I retrain on the remaining 20 features and performance is the same — but training is 25× faster and the model is far simpler to deploy and monitor. The L1 penalty identified that 480 features added noise but no signal."

**Common traps**:
- performing feature selection before the train/test split — any method that computes statistics on the full dataset (even univariate correlation) leaks test information into the selection process
- removing features with low individual correlation to the target without checking interactions — two individually uncorrelated features may be jointly predictive; filter methods miss interaction effects

---

## 10. Feature Extraction vs Feature Selection

**What the interviewer is testing**: whether you understand these as fundamentally different approaches to dimensionality reduction with different properties.

**The reasoning structure**: feature selection keeps a subset of the original features. The retained features are interpretable because they are original measured quantities. Feature extraction creates new features as transformations of the original features, reducing dimensionality through compression.

Feature selection preserves interpretability because each retained feature has a real-world meaning. Feature extraction (PCA, autoencoders) produces components that may not have a clear interpretation.

The choice depends on interpretability requirements. For a regulated credit model where you must explain feature contributions to regulators, feature selection is necessary. For a high-dimensional vision or text problem where interpretability is not required, feature extraction is more powerful.

**Common traps**:
- using PCA-extracted features and then trying to explain the model using feature importances — the "features" are linear combinations of original features; the importance scores correspond to PCA components, not to original measurable variables
- treating feature selection and extraction as equivalent alternatives — they have different tradeoffs and different downstream interpretability implications; choosing between them requires knowing whether interpretability is a hard requirement

---

## 11. The Train-Serve Skew Problem

**What the interviewer is testing**: whether you understand that a feature is only real if it can be computed identically in production.

**The reasoning structure**: during training, you might compute features from a complete historical database. At inference time, that same database might contain only a sliding window, have a data lag, or not exist at all in real-time. Any discrepancy between how features are computed in training and in production is train-serve skew.

Train-serve skew is especially insidious because it does not appear in offline evaluation — your validation set was constructed with the same training pipeline. It only manifests in production, where real-time feature computation differs from batch feature computation.

**The pattern in action**: "My recommendation model uses `user_last_7d_purchases` as a feature. In training, I compute this from a database with complete purchase history. In production, this feature is served from a Redis cache updated every 6 hours. For users who purchased in the last 6 hours, the cache shows stale data — a feature value different from what the model was trained to expect. This is train-serve skew. I need to either accept the lag and train with the lagged version, or engineer the feature to use the cache's actual update frequency as its definition."

**Common traps**:
- designing feature computations in training that cannot be replicated in real-time in production — "I used 30 joins in the training query" needs a production-ready equivalent that can execute within the latency budget
- not testing the inference pipeline with actual production feature values — offline validation metrics can look fine while production performance is degraded due to skew that only appears with real serving-time feature values

---

## 12. Handling Class Imbalance

**What the interviewer is testing**: whether you understand the distinction between the model bias problem (class distribution) and the metric problem (choosing the right evaluation criterion) — and that fixing the metric problem is often more important.

**The reasoning structure**: in fraud detection, cancer diagnosis, and rare event prediction, positive cases are a tiny fraction of all examples. Training on raw class proportions means the model sees very few positive examples and learns to predict the majority class. The root cause has two components:

First, the loss function optimization produces a model biased toward the majority class. Fix: class weighting (`class_weight='balanced'` in sklearn, or `pos_weight` in PyTorch), which multiplies the loss contribution of minority class examples by the class ratio.

Second — and more important — accuracy is the wrong metric. A model with 0% recall can achieve 99% accuracy on a 1% positive rate dataset. Fix: use precision, recall, F1, or PR-AUC as the primary metric.

**The pattern in action**: "My fraud dataset is 0.5% positive. I train a model and get 99.5% accuracy. I check recall: 0%. The model is predicting 'not fraud' for everything. I add `class_weight='balanced'` to my logistic regression. Accuracy drops to 92%, but recall jumps to 78%. I then tune the threshold using the precision-recall curve to match the team's daily review capacity."

**Common traps**:
- oversampling the minority class with SMOTE without careful thought — SMOTE generates synthetic interpolations between minority class points, which can introduce geometrically unrealistic examples; class weighting usually achieves the same effect more cleanly and with fewer artifacts
- choosing the threshold at 0.5 for imbalanced problems — with a 1% positive rate, a threshold of 0.5 means the model must be fairly confident before flagging anything; the right threshold should be chosen from the precision-recall curve based on operational constraints (how many alerts can be reviewed per day)

## Rapid Recall

### jumping to model selection before auditing the data
- Direct Answer: the fastest path to production is understanding why the data looks the way it does; a week of data investigation can prevent months of debugging
- Why: This matters because it tells you how to reason about jumping to model selection before auditing the data.
- Pitfall: Don't answer "jumping to model selection before auditing the data" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: the fastest path to production is understanding why the data looks the way it does; a week of data investigation can prevent months of debugging

### treating preprocessing as a one-time step
- Direct Answer: preprocessing logic must be implemented identically in the training pipeline and the inference pipeline, version-controlled alongside the model, and monitored for drift
- Why: This matters because it tells you how to reason about treating preprocessing as a one-time step.
- Pitfall: Don't answer "treating preprocessing as a one-time step" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: preprocessing logic must be implemented identically in the training pipeline and the inference pipeline, version-controlled alongside the model, and monitored for drift

### fitting the scaler on the full dataset before splitting
- Direct Answer: this leaks test set distribution information (mean and variance of test features) into the training pipeline; it is a subtle form of data leakage
- Why: This matters because it tells you how to reason about fitting the scaler on the full dataset before splitting.
- Pitfall: Don't answer "fitting the scaler on the full dataset before splitting" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: this leaks test set distribution information (mean and variance of test features) into the training pipeline; it is a subtle form of data leakage

### applying scaling to tree-based models expecting improvement
- Direct Answer: scaling changes nothing for trees and adds unnecessary pipeline complexity
- Why: This matters because it tells you how to reason about applying scaling to tree-based models expecting improvement.
- Pitfall: Don't answer "applying scaling to tree-based models expecting improvement" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: scaling changes nothing for trees and adds unnecessary pipeline complexity

### Color (red/blue/green)
- Direct Answer: one-hot → 3 binary columns
- Why: This matters because it tells you how to reason about color (red/blue/green).
- Pitfall: Don't answer "Color (red/blue/green)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: one-hot → 3 binary columns

### City (10,000 unique values)
- Direct Answer: target encoding with smoothing → 1 numeric column per city
- Why: This matters because it tells you how to reason about city (10,000 unique values).
- Pitfall: Don't answer "City (10,000 unique values)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: target encoding with smoothing → 1 numeric column per city

### target encoding on the full dataset
- Direct Answer: the category mean is computed using the labels you are trying to predict; this is direct label leakage. Always use out-of-fold target encoding: compute statistics on training folds, apply to held-out folds
- Why: This matters because it tells you how to reason about target encoding on the full dataset.
- Pitfall: Don't answer "target encoding on the full dataset" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: the category mean is computed using the labels you are trying to predict; this is direct label leakage. Always use out-of-fold target encoding: compute statistics on training fold…

### not handling unseen categories at inference time
- Direct Answer: a city not seen during training has no stored encoding; you need a fallback (global mean, a special "unseen" token, or a default value)
- Why: This matters because it tells you how to reason about not handling unseen categories at inference time.
- Pitfall: Don't answer "not handling unseen categories at inference time" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: a city not seen during training has no stored encoding; you need a fallback (global mean, a special "unseen" token, or a default value)

### using ordinal encoding (1, 2, 3...) for nominal categories
- Direct Answer: this incorrectly encodes the assumption that the categories have a meaningful numeric ordering when they do not
- Why: This matters because it tells you how to reason about using ordinal encoding (1, 2, 3...) for nominal categories.
- Pitfall: Don't answer "using ordinal encoding (1, 2, 3...) for nominal categories" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: this incorrectly encodes the assumption that the categories have a meaningful numeric ordering when they do not

### MCAR (Missing Completely at Random)
- Direct Answer: missingness is independent of both observed and unobserved values. Row deletion or simple imputation is usually safe.
- Why: This matters because it tells you how to reason about mcar (missing completely at random).
- Pitfall: Don't answer "MCAR (Missing Completely at Random)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: missingness is independent of both observed and unobserved values. Row deletion or simple imputation is usually safe.

### MAR (Missing At Random)
- Direct Answer: missingness depends on other observed features (e.g., older customers are less likely to fill in income). Model-based imputation using the observed features can recover the true distribution.
- Why: This matters because it tells you how to reason about mar (missing at random).
- Pitfall: Don't answer "MAR (Missing At Random)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: missingness depends on other observed features (e.g., older customers are less likely to fill in income). Model-based imputation using the observed features can recover the true d…

### MNAR (Missing Not at Random): missingness depends on the unobserved value itself (e.g., high-income customers leave the income field blank). No imputation method can fully recover this
- Direct Answer: the missingness itself is informative.
- Why: This matters because it tells you how to reason about mnar (missing not at random): missingness depends on the unobserved value itself (e.g., high-income customers leave the income field blank). no imputation method can fully recover this.
- Pitfall: Don't answer "MNAR (Missing Not at Random): missingness depends on the unobserved value itself (e.g., high-income customers leave the income field blank). No imputation method can fully recover this" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: the missingness itself is informative.

### treating all missingness as MCAR and applying mean imputation
- Direct Answer: mean imputation is only appropriate for MCAR; for MAR and MNAR it introduces bias by ignoring the dependence between missingness and other variables
- Why: This matters because it tells you how to reason about treating all missingness as mcar and applying mean imputation.
- Pitfall: Don't answer "treating all missingness as MCAR and applying mean imputation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: mean imputation is only appropriate for MCAR; for MAR and MNAR it introduces bias by ignoring the dependence between missingness and other variables

### dropping rows with missing values without checking what fraction of positive examples are lost
- Direct Answer: in imbalanced datasets, missingness may be correlated with the positive class; dropping rows can dramatically change your effective class distribution
- Why: This matters because it tells you how to reason about dropping rows with missing values without checking what fraction of positive examples are lost.
- Pitfall: Don't answer "dropping rows with missing values without checking what fraction of positive examples are lost" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: in imbalanced datasets, missingness may be correlated with the positive class; dropping rows can dramatically change your effective class distribution

### fitting the imputer on the full dataset before splitting
- Direct Answer: fit the imputer (its mean, its KNN model) only on training data; apply it to validation and test using training statistics only
- Why: This matters because it tells you how to reason about fitting the imputer on the full dataset before splitting.
- Pitfall: Don't answer "fitting the imputer on the full dataset before splitting" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: fit the imputer (its mean, its KNN model) only on training data; apply it to validation and test using training statistics only

### log-transforming a feature and forgetting the inverse transform at prediction time
- Direct Answer: if you model log(price), predictions are in log-space; you must exponentiate before reporting to users
- Why: This matters because it tells you how to reason about log-transforming a feature and forgetting the inverse transform at prediction time.
- Pitfall: Don't answer "log-transforming a feature and forgetting the inverse transform at prediction time" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: if you model log(price), predictions are in log-space; you must exponentiate before reporting to users

### applying log transform to features with zeros
- Direct Answer: $\log(0)$ is undefined; use $\log(x + 1)$ instead, or apply RobustScaler if the skewness is driven by outliers rather than the scale of the distribution
- Why: This matters because it tells you how to reason about applying log transform to features with zeros.
- Pitfall: Don't answer "applying log transform to features with zeros" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: $\log(0)$ is undefined; use $\log(x + 1)$ instead, or apply RobustScaler if the skewness is driven by outliers rather than the scale of the distribution

### using IQR-based outlier removal blindly without understanding domain
- Direct Answer: a value being statistically extreme does not mean it is wrong; the IQR test is a detection heuristic, not a ground truth for validity
- Why: This matters because it tells you how to reason about using iqr-based outlier removal blindly without understanding domain.
- Pitfall: Don't answer "using IQR-based outlier removal blindly without understanding domain" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: a value being statistically extreme does not mean it is wrong; the IQR test is a detection heuristic, not a ground truth for validity

### in fraud and anomaly detection, the outlier is often the target
- Direct Answer: removing statistical outliers in these domains destroys the very signal you are trying to model
- Why: This matters because it tells you how to reason about in fraud and anomaly detection, the outlier is often the target.
- Pitfall: Don't answer "in fraud and anomaly detection, the outlier is often the target" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: removing statistical outliers in these domains destroys the very signal you are trying to model

### engineering features that are not available at inference time
- Direct Answer: if you use the target variable or post-event information in feature construction, that is leakage; the feature must be computable using only information available at the moment of prediction
- Why: This matters because it tells you how to reason about engineering features that are not available at inference time.
- Pitfall: Don't answer "engineering features that are not available at inference time" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: if you use the target variable or post-event information in feature construction, that is leakage; the feature must be computable using only information available at the moment of…

### not validating that engineered features are stable across time
- Direct Answer: a feature predictive in historical data may stop being predictive when underlying behavior shifts; monitor feature distributions and predictive power in production
- Why: This matters because it tells you how to reason about not validating that engineered features are stable across time.
- Pitfall: Don't answer "not validating that engineered features are stable across time" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: a feature predictive in historical data may stop being predictive when underlying behavior shifts; monitor feature distributions and predictive power in production

### using cross-validation without re-fitting preprocessing inside each fold
- Direct Answer: this leaks validation fold statistics into the preprocessing fit; the scaler has already seen the validation examples
- Why: This matters because it tells you how to reason about using cross-validation without re-fitting preprocessing inside each fold.
- Pitfall: Don't answer "using cross-validation without re-fitting preprocessing inside each fold" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: this leaks validation fold statistics into the preprocessing fit; the scaler has already seen the validation examples

### performing feature selection on the full dataset before splitting
- Direct Answer: the selected features are correlated with the test labels, inflating all downstream validation estimates
- Why: This matters because it tells you how to reason about performing feature selection on the full dataset before splitting.
- Pitfall: Don't answer "performing feature selection on the full dataset before splitting" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: the selected features are correlated with the test labels, inflating all downstream validation estimates

### Filter methods
- Direct Answer: score each feature independently of the model (correlation with target, mutual information, chi-squared). Fast but ignore feature interactions.
- Why: This matters because it tells you how to reason about filter methods.
- Pitfall: Don't answer "Filter methods" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: score each feature independently of the model (correlation with target, mutual information, chi-squared). Fast but ignore feature interactions.

### Wrapper methods
- Direct Answer: train the model on different feature subsets and select the best-performing subset. Expensive but finds the optimal subset for the specific model.
- Why: This matters because it tells you how to reason about wrapper methods.
- Pitfall: Don't answer "Wrapper methods" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: train the model on different feature subsets and select the best-performing subset. Expensive but finds the optimal subset for the specific model.

### Embedded methods
- Direct Answer: regularization during training selects features (L1/Lasso zeroes out unimportant features; XGBoost feature importance provides rankings).
- Why: This matters because it tells you how to reason about embedded methods.
- Pitfall: Don't answer "Embedded methods" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: regularization during training selects features (L1/Lasso zeroes out unimportant features; XGBoost feature importance provides rankings).

### performing feature selection before the train/test split
- Direct Answer: any method that computes statistics on the full dataset (even univariate correlation) leaks test information into the selection process
- Why: This matters because it tells you how to reason about performing feature selection before the train/test split.
- Pitfall: Don't answer "performing feature selection before the train/test split" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: any method that computes statistics on the full dataset (even univariate correlation) leaks test information into the selection process

### removing features with low individual correlation to the target without checking interactions
- Direct Answer: two individually uncorrelated features may be jointly predictive; filter methods miss interaction effects
- Why: This matters because it tells you how to reason about removing features with low individual correlation to the target without checking interactions.
- Pitfall: Don't answer "removing features with low individual correlation to the target without checking interactions" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: two individually uncorrelated features may be jointly predictive; filter methods miss interaction effects

### using PCA-extracted features and then trying to explain the model using feature importances
- Direct Answer: the "features" are linear combinations of original features; the importance scores correspond to PCA components, not to original measurable variables
- Why: This matters because it tells you how to reason about using pca-extracted features and then trying to explain the model using feature importances.
- Pitfall: Don't answer "using PCA-extracted features and then trying to explain the model using feature importances" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: the "features" are linear combinations of original features; the importance scores correspond to PCA components, not to original measurable variables

### treating feature selection and extraction as equivalent alternatives
- Direct Answer: they have different tradeoffs and different downstream interpretability implications; choosing between them requires knowing whether interpretability is a hard requirement
- Why: This matters because it tells you how to reason about treating feature selection and extraction as equivalent alternatives.
- Pitfall: Don't answer "treating feature selection and extraction as equivalent alternatives" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: they have different tradeoffs and different downstream interpretability implications; choosing between them requires knowing whether interpretability is a hard requirement

### designing feature computations in training that cannot be replicated in real-time in production
- Direct Answer: "I used 30 joins in the training query" needs a production-ready equivalent that can execute within the latency budget
- Why: This matters because it tells you how to reason about designing feature computations in training that cannot be replicated in real-time in production.
- Pitfall: Don't answer "designing feature computations in training that cannot be replicated in real-time in production" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: "I used 30 joins in the training query" needs a production-ready equivalent that can execute within the latency budget

### not testing the inference pipeline with actual production feature values
- Direct Answer: offline validation metrics can look fine while production performance is degraded due to skew that only appears with real serving-time feature values
- Why: This matters because it tells you how to reason about not testing the inference pipeline with actual production feature values.
- Pitfall: Don't answer "not testing the inference pipeline with actual production feature values" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: offline validation metrics can look fine while production performance is degraded due to skew that only appears with real serving-time feature values

### oversampling the minority class with SMOTE without careful thought
- Direct Answer: SMOTE generates synthetic interpolations between minority class points, which can introduce geometrically unrealistic examples; class weighting usually achieves the same effect more cleanly and with fewer artifacts
- Why: This matters because it tells you how to reason about oversampling the minority class with smote without careful thought.
- Pitfall: Don't answer "oversampling the minority class with SMOTE without careful thought" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: SMOTE generates synthetic interpolations between minority class points, which can introduce geometrically unrealistic examples; class weighting usually achieves the same effect mo…

### choosing the threshold at 0.5 for imbalanced problems
- Direct Answer: with a 1% positive rate, a threshold of 0.5 means the model must be fairly confident before flagging anything; the right threshold should be chosen from the precision-recall curve based on operational constraints (how many alerts can be reviewed per day)
- Why: This matters because it tells you how to reason about choosing the threshold at 0.5 for imbalanced problems.
- Pitfall: Don't answer "choosing the threshold at 0.5 for imbalanced problems" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: with a 1% positive rate, a threshold of 0.5 means the model must be fairly confident before flagging anything; the right threshold should be chosen from the precision-recall curve…
