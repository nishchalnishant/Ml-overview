# Data Preprocessing and Feature Engineering

For many ML interviews, this topic matters more than model choice. Strong answers explain how preprocessing affects leakage, model assumptions, robustness, and production behavior.

---

# Q1: What is Feature Engineering?

**Interview-ready answer**

Feature engineering is the process of transforming raw data into inputs that make the learning problem easier for the model to solve. In practice, that means encoding domain knowledge, exposing signal more clearly, reducing noise, and aligning the representation with the model's assumptions. A good interview answer should make clear that feature engineering is not only "creating more columns"; it includes cleaning, aggregation, normalization, temporal logic, interaction features, and even choosing what information should not be included to avoid leakage.

**Why it matters**

- Better features can improve simple models more than switching to a more complex algorithm.
- Good features often improve interpretability and training stability.
- In production systems, feature definitions need to be consistent offline and online.

---

# Q2: What is one-hot encoding? When should you use it?

**Interview-ready answer**

One-hot encoding converts a categorical variable into binary indicator columns, one per category. It works well when the number of categories is modest and there is no meaningful ordinal relationship between them. It is especially appropriate for linear models and neural networks that need numeric inputs, but it can become inefficient when cardinality is very high because the representation becomes sparse and wide.

**Good nuance**

- It is a safe default for low-cardinality nominal variables.
- It is usually unnecessary for many tree-based models if they support categorical handling or can work from label-encoded splits safely.
- For high-cardinality features such as user IDs or ZIP-plus-household combinations, embeddings, hashing, or target statistics are often more practical.

---

# Q3: How do you deal with missing data?

**Interview-ready answer**

I first try to understand why the data is missing, because the treatment depends on the missingness mechanism. If values are missing completely at random, simple imputation may be sufficient. If missingness itself carries signal, then I would usually keep that information by adding a missing indicator along with the imputed value. In production, the key goal is to use a strategy that is leakage-safe, stable, and reproducible at serving time.

**Common approaches**

- Drop rows or columns only when missingness is limited and not informative.
- Impute with mean, median, mode, or a constant for simple baselines.
- Use model-based imputation when the added complexity is justified.
- Add missing flags when absence may carry meaning.

**Common pitfall**

Fitting the imputer before the train-validation split leaks information.

---

# Q4: How do you handle Outliers?

**Interview-ready answer**

I do not remove outliers automatically. First I determine whether they are data errors, rare but valid events, or the very cases the business cares about. If they are errors, I fix or exclude them. If they are valid but destabilize the model, I might use robust scaling, winsorization, log transforms, or a loss function less sensitive to extreme values. The answer should show that outlier handling depends on domain context, not just statistics.

**What to mention**

- Tree models are often less sensitive to outliers than linear or distance-based models.
- For skewed positive variables, log transforms are often better than blunt clipping.
- In fraud or anomaly settings, "outliers" may be the target behavior and should be preserved.

---

# Q5: Explain Feature Scaling. Why is it needed?

**Interview-ready answer**

Feature scaling puts numerical variables on comparable ranges so that optimization and distance computations behave sensibly. It matters for models like logistic regression, SVMs, KNN, k-means, PCA, and neural networks because these methods are sensitive to magnitude. Without scaling, a feature with a large numeric range can dominate the loss or the distance metric even if it is not the most informative feature.

**Typical choices**

- Standardization: zero mean and unit variance
- Min-max scaling: map values to a bounded interval
- Robust scaling: use median and IQR for outlier-heavy data

**Good nuance**

Tree-based methods usually do not require scaling, but pipelines may still standardize features when multiple model families are being compared.

---

# Q6: One-Hot, Label, Target, and K-Fold Target Encoding

**Interview-ready answer**

These encoding methods represent categorical variables in different ways and have very different failure modes. One-hot encoding is safest and most transparent for low-cardinality nominal features. Label encoding maps categories to integers, but unless the model treats them purely as identifiers, it can accidentally impose a false order. Target encoding replaces each category with a statistic derived from the target, which can be very powerful for high-cardinality features but is highly leakage-prone unless done carefully. K-fold target encoding reduces that leakage by computing category statistics out-of-fold rather than on the full training set.

**How to talk about tradeoffs**

- One-hot: simple, safe, sparse, does not scale well to huge cardinality
- Label encoding: fine for tree methods in some cases, dangerous for linear models
- Target encoding: compact and powerful, but requires smoothing and leakage-safe computation
- K-fold target encoding: better validation discipline, but more pipeline complexity

---

# Q7: How do you handle Categorical Features?

**Interview-ready answer**

My approach depends on cardinality, model family, and how stable the categories are in production. For low-cardinality variables, one-hot encoding is usually enough. For high-cardinality IDs or text-like categories, I consider target statistics, hashing, embeddings, or models with native categorical support such as CatBoost. I also think about unseen categories at inference time, because a preprocessing strategy that looks fine offline can break once new values appear in production.

**Good things to mention**

- Frequency-based grouping for rare categories
- Explicit handling for unknown values
- Leakage-safe target statistics
- Domain semantics, such as whether the category is nominal or ordinal

---

# Q8: Explain feature selection vs feature extraction.

**Interview-ready answer**

Feature selection keeps a subset of the original features, while feature extraction creates new features by transforming the original space. Selection is about deciding which columns to keep; extraction is about building a new representation. Selection is often preferred when interpretability matters, whereas extraction is powerful when the original features are noisy, redundant, or too high-dimensional.

**Examples**

- Feature selection: L1 regularization, mutual information, tree-based importance, recursive elimination
- Feature extraction: PCA, autoencoders, learned embeddings, topic models

**Good nuance**

These are not mutually exclusive. A common workflow is to engineer features, remove clearly weak ones, and then use a learned representation for a downstream model.

---

# Q9: How would you create new features from existing ones?

**Interview-ready answer**

I start from the prediction task and the data-generating process. Then I look for transformations that expose stable signal: ratios, interactions, lags, rolling aggregates, recency features, counts, domain thresholds, and group-level summaries. In interviews, it helps to say that I prefer features that reflect business logic or time structure rather than arbitrary polynomial expansion, because engineered features should improve both predictive power and interpretability.

**Useful examples**

- Transactions: spend in last 7, 30, and 90 days
- Recommenders: user-item interaction counts and recency
- Operations: rolling averages, volatility, and change relative to baseline
- Geography: density, distance, neighborhood aggregates

**Common pitfall**

Aggregations are a major source of leakage if they use future information or include the validation period.

---

# Q10: How do you approach a dataset with highly imbalanced classes?

**Interview-ready answer**

I treat imbalance as a modeling and evaluation issue together. First I choose metrics that reflect the minority class, such as precision, recall, F1, PR-AUC, recall at a target precision, or cost-weighted utility. Then I compare interventions such as class weighting, threshold tuning, better sampling, focal loss, or specialized objectives. If the minority class is extremely rare, I also question whether the problem is better framed as anomaly detection or ranking.

**Important detail**

Resampling should only be done on the training set. Validation and test should reflect the real deployment distribution whenever possible.

---

# Q11: How do you select features for a model?

**Interview-ready answer**

I combine domain knowledge with empirical testing. I usually start by removing features that are obviously unavailable at prediction time, duplicated, constant, or clearly leaky. Then I look at correlation, redundancy, sparsity, and missingness. After that I evaluate features through model-based experiments, ablations, and slice analysis rather than trusting a single importance score. The point is not just to reduce dimensionality, but to keep a robust, interpretable, and maintainable set of inputs.

**Good interview nuance**

- The best feature set depends on the model family.
- Correlation alone is not enough to decide what to drop.
- Feature importance should be interpreted carefully, especially with correlated features.

---

# Q12: Why and how do you split data into train, test, and validation sets?

**Interview-ready answer**

We split data so that model fitting, model selection, and final evaluation remain separate. The training set is used to learn parameters, the validation set is used for model choice and hyperparameter tuning, and the test set is held back for a final unbiased estimate. A strong interview answer should go beyond the textbook definition and say that the split strategy must match the real deployment setting: time-based splits for time series, group-based splits for repeated entities, and stratification when label balance matters.

**What strong candidates mention**

- Build preprocessing inside the training pipeline
- Avoid duplicate entities across splits
- Keep the test set untouched until the end
- Revisit the split design if the production setting changes
