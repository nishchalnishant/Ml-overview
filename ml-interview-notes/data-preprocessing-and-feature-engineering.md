# Data Preprocessing and Feature Engineering

If model building is glamorous, data preprocessing is the tailoring.

And like tailoring, nobody notices it when it is done well.
They only notice when everything fits badly.

This file is about the work that quietly decides whether the model has a chance.

---

# 1. Why Preprocessing Matters So Much

Most ML interview candidates rush to model choice.

Strong candidates slow down and ask:

- What does the data actually look like?
- What is missing?
- What leaks?
- What is categorical?
- What will exist at inference time?

That last one matters a lot.

Because a feature that exists only in training is not a feature.
It is a trap.

---

# 2. One-Hot Encoding vs Target Encoding

## One-Hot Encoding

Create a binary column for each category.

Best for:

- low-cardinality features
- nominal categories
- simple, transparent pipelines

## Target Encoding

Replace each category with a statistic derived from the target, often the category mean.

Best for:

- high-cardinality features
- situations where one-hot would explode feature count

**The danger**

Target encoding can leak label information very easily.

That is why you usually need:

- smoothing
- out-of-fold encoding
- careful validation

**Short interview answer**

One-hot is safer and more transparent; target encoding is more compact and powerful for high-cardinality features but much riskier for leakage.

---

# 3. K-Fold Target Encoding

This is the grown-up version of target encoding.

Instead of using the full dataset target mean for each category, you compute category statistics on training folds and apply them to held-out folds.

That helps reduce leakage.

**DevOps analogy**

Think of it as avoiding test-environment secrets leaking into build-time config.
Same flavor of mistake.
Different costume.

---

# 4. Handling Missing Values

This is one of the most practical preprocessing topics.

The first question is not:

> "Which imputer should I use?"

It is:

> "Why is this value missing?"

That changes everything.

## MCAR

Missing Completely at Random.

Usually easier to handle.

## MAR

Missingness depends on other observed variables.

Often manageable with smarter imputation.

## MNAR

Missingness itself carries signal.

This is where a missing-indicator feature becomes especially valuable.

---

# 5. Missing Indicators

Sometimes the absence of a value tells you more than the value itself would have.

Example:

- user did not fill salary field
- device fingerprint missing
- sensor reading absent

That is why a good approach is often:

- impute the value
- add a binary flag like `is_missing_feature`

This lets the model learn:

- value behavior
- missingness behavior

Both.

---

# 6. Mean vs Median vs Model-Based Imputation

## Mean Imputation

Simple.

Works okay for roughly symmetric numeric features.

## Median Imputation

More robust when outliers exist.

Often the safer default for skewed data.

## Model-Based Imputation

Use other features to predict the missing value.

Examples:

- KNN imputation
- MICE

More powerful.
Also more complex.

**Interview instinct**

Start simple unless there is strong evidence that smarter imputation materially helps.

---

# 7. Scaling: Standard, Min-Max, Robust

Scaling matters when the algorithm cares about:

- distance
- magnitude
- gradient behavior

## Standard Scaler

Centers to mean 0 and standard deviation 1.

Good for:

- linear models
- logistic regression
- SVMs
- neural nets

## Min-Max Scaler

Maps values into a bounded range, often 0 to 1.

Useful for:

- bounded-input pipelines
- some neural settings

## Robust Scaler

Uses median and IQR.

Good when outliers are wild and you do not want them dominating the scale.

**Mini Pop Quiz**

Which model cares more about scaling:

- K-Means
- Random Forest

Answer:

K-Means.

Because distance-based methods feel scale very directly.

---

# 8. Why Scaling Matters

Imagine one feature ranges from:

- 0 to 1

and another ranges from:

- 0 to 1,000,000

Without scaling, many algorithms will let the huge-range feature dominate everything.

That is not intelligence.
That is numeric bullying.

---

# 9. Handling Skewed Data

When data is heavily skewed, some models behave badly.

Common fixes:

- log transform
- Box-Cox
- Yeo-Johnson

Why this helps:

It compresses long tails and makes the main body of the data easier to model.

**Fashion pricing analogy**

If most dresses cost 3k to 10k and a few couture pieces cost 3 lakhs, the raw scale can distort everything.
Transforms help bring the structure back into focus.

---

# 10. Outliers

Do not auto-delete outliers just because they look inconvenient.

Ask:

- is this a data error?
- is this rare but valid behavior?
- is this exactly the thing we care about detecting?

Examples:

- in fraud, the outlier may be the target
- in industrial monitoring, the outlier may be the failure signal

That is why outlier handling must follow domain context.

Not spreadsheet panic.

---

# 11. Feature Crosses and Interactions

Feature crosses let simple models capture interactions that they would otherwise miss.

Example:

- weekend
- beach location

Individually, maybe mild signal.

Together?

Much stronger signal for ice cream demand.

That combination is the real story.

**Short answer**

Feature crosses help linear models represent non-linear interactions between variables.

---

# 12. Feature Selection vs Feature Extraction

## Feature Selection

Keep some original features.
Drop the rest.

## Feature Extraction

Create a new representation from the old features.

Examples:

- PCA
- embeddings
- autoencoders

**Easy memory trick**

- selection = choose the best outfit pieces
- extraction = redesign the silhouette

---

# 13. Leakage

This is one of the highest-value interview topics in preprocessing.

The most common leak:

Fitting transforms before the train/validation split.

Examples:

- scaler fit on full dataset
- imputer fit on full dataset
- target encoding using future label info

Correct workflow:

1. split first
2. fit preprocessing only on training data
3. apply transform to validation/test using learned training parameters

**Azure mindset**

This is the ML equivalent of reading production-only values during build-time tests and then congratulating yourself for "passing."

Absolutely not.

---

# 14. Feature Engineering in Production

Feature engineering is not just about inventing columns.

It is about building features that are:

- meaningful
- reproducible
- available at inference time
- stable under drift

That last point is very important.

A clever feature that cannot be served reliably is not a good feature.

It is technical debt with excellent presentation skills.

---

# 15. Categorical Handling Strategy in One Breath

A very solid interview answer sounds like this:

> "For low-cardinality nominal variables I usually start with one-hot encoding. For high-cardinality categories I consider target encoding, hashing, embeddings, or native categorical support, but only with strict leakage-safe validation and a plan for unseen categories in production."

That answer is clean, balanced, and credible.

---

# Quick Thought Experiment

You are building a churn model in Azure ML.

A feature called `last_support_resolution_time` looks incredibly predictive.

Before celebrating, ask:

- was that feature available at prediction time?
- was it created after the event we are trying to predict?
- is it leaking the label window?

If yes, that feature is not genius.
It is fraud with better manners.

---

# How Would You Deploy This Using Azure Pipelines?

For preprocessing-heavy models, your pipeline should validate:

- schema compatibility
- missing-column behavior
- scaler/imputer artifact version
- categorical mapping version
- unseen category fallback logic
- train-serve parity

That is where DevOps discipline quietly becomes ML advantage.
