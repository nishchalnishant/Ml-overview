---
module: Data Scientist
topic: Eda And Data Quality
subtopic: ""
status: unread
tags: [datascientist, ml, eda-and-data-quality]
---
# EDA and Data Quality

---

## 1. Why Explore Before Modeling?

You receive a dataset and want to fit a model. The fastest path seems to be: load data, split train/test, fit model, evaluate. You try it. Your model achieves 99% accuracy on the test set. You deploy. It immediately fails in production.

What happened? You didn't know that:
- 40% of rows had a target-adjacent column accidentally left in the feature set.
- The "age" column contained the default fill value 999 for nulls.
- Train and test were sampled from different time periods, so the test set looks nothing like production.

EDA is the process of discovering these problems before they waste weeks of work. It is not decorative. It is the first line of defense against modeling on corrupted or misunderstood data.

### The workflow order matters

Each step gates the next. Running distribution analysis before checking types is wasteful — if `age` is stored as `object` (string), your numeric summary is meaningless.

1. **Shape and types** — know what you have before anything else: `df.shape`, `df.dtypes`, `df.info()`. Flag columns where the inferred type contradicts what the column should be (numeric stored as object, datetimes stored as strings).
2. **Missingness** — understand which columns have gaps and whether the gaps are structured. Structured gaps can be the most predictive signal in the dataset, or the most dangerous source of bias.
3. **Univariate distributions** — look at each feature alone. Find values that are physically impossible, distributions that suggest data artifacts, and variables that are nearly constant.
4. **Outliers** — identify extreme values. Understand whether they represent real extreme cases, measurement errors, or encoding conventions.
5. **Bivariate and multivariate structure** — look at how features relate to each other and to the target. Find redundancy, nonlinearity, and potential leakage.
6. **Data quality audit** — check uniqueness, consistency, validity, timeliness. These problems do not show up in statistical summaries.

---

## 2. Univariate Analysis

### Why you cannot skip looking at each feature individually

When you look at correlations first, you learn how features relate to each other. But a feature with a massive spike at 999 (a sentinel null value) will have a meaningless correlation with anything else. Univariate analysis finds problems that corrupt all downstream analysis.

---

### The binning problem with histograms

You want to visualize the distribution of a continuous feature. The most obvious choice is a histogram. You divide the range into bins and count how many values fall in each bin.

The problem: the shape you see depends on where you put the bin edges. Shift every edge by half a bin width and you get a different picture. This is an artifact of your choice, not a property of the data. A feature that looks unimodal with one binning can look bimodal with another.

**What to do**: use histograms as a starting point, but treat the shape as tentative until confirmed by a binning-independent method. For bin width, Sturges' rule ($k = \lceil \log_2 n + 1 \rceil$) is the default in most tools; Freedman-Diaconis ($h = 2 \cdot \text{IQR} \cdot n^{-1/3}$) is better when data is skewed or has outliers because it uses the interquartile range rather than range.

**What to look for**:
- Bimodality: two humps strongly suggest two sub-populations mixed together. This changes what preprocessing and what model families make sense.
- Sharp cutoffs at round numbers: a flood of values exactly at 100 or 0 typically indicates truncation or a sensor ceiling, not a real concentration of the quantity.
- Spikes at specific values: a spike at 0, -1, 999, or 9999 almost always indicates a sentinel encoding for null rather than a real data point.
- Extreme right skew: common for monetary amounts, counts, and durations. Log-transforming makes the distribution more legible and often more useful for linear models.

---

### Why KDE over a histogram

As above, a histogram forces you to choose bin edges — an arbitrary decision. KDE removes that arbitrariness. It places a small smooth bump (a kernel, typically Gaussian) at each observed data point and sums all the bumps. The result is a smooth continuous curve that estimates the underlying density without depending on bin placement.

**What bandwidth controls**: how wide each bump is. Too narrow — the curve is spiky and overfit to individual data points, showing noise as structure. Too wide — the curve is oversmoothed, and real modes merge together.

Silverman's rule of thumb for bandwidth:

$$h = 1.06 \hat{\sigma} n^{-1/5}$$

This is optimal for Gaussian data. For bimodal or heavy-tailed data it oversmooths. In those cases, decrease bandwidth manually or use Scott's rule ($h = 1.059 \hat{\sigma} n^{-1/5}$, essentially the same) and compare multiple values.

**KDE is most useful** for overlaying multiple distributions on the same axis — for example, the distribution of a feature conditioned on the target class. Overlapping histograms become illegible; overlapping KDE curves remain readable.

**What breaks**: if the feature has a hard boundary (ages must be non-negative, percentages bounded at 0 and 100), a Gaussian KDE will bleed mass outside the valid range. In that case either transform the data or use a boundary-corrected kernel.

---

### The five-number summary is not enough: box plots and what they miss

You have a numeric feature. The mean is 50, the standard deviation is 10. That tells you almost nothing about the shape of the distribution. A box plot shows more.

A box plot displays:
- The median (horizontal line inside the box) — the center of the distribution, robust to outliers.
- Q1 and Q3 (box edges) — the middle 50% of the data.
- Whiskers extending to the most extreme point within $1.5 \times \text{IQR}$ of the quartiles — the "typical" spread.
- Individual points beyond the whiskers — flagged as potential outliers.

Box plots are compact. Their main use is **comparing the same feature across many categories** — one box per category, all on the same axis. This immediately reveals whether the feature's center or spread shifts across groups.

**What box plots miss**: shape. A unimodal symmetric distribution and a bimodal distribution can produce identical box plots if the two modes are equidistant from the median. This is why box plots are best paired with KDE for any feature you care about deeply.

---

### When your model assumes Normality and the data isn't: Q-Q plots

Linear models, linear discriminant analysis, and many statistical tests assume features (or residuals) are approximately Normal. If that assumption fails, your standard errors are wrong, your p-values are meaningless, and your model's calibration degrades.

A Q-Q plot (quantile-quantile plot) is a direct visual test of that assumption. It works as follows: sort your data, compute the empirical quantiles, and plot them against the quantiles you would expect from a perfect Normal distribution. If the data is Normal, the points fall on a straight diagonal line.

**Reading the deviations**:
- **S-curve (sigmoid shape)**: the tails are heavier than Normal. Values cluster near zero (the kurtosis is high). Common for financial returns, network latency.
- **Inverted S-curve**: the distribution is platykurtic — lighter tails and a flatter center than Normal.
- **Points curve up on the right only**: right-skewed distribution. The largest values are larger than the Normal would predict.
- **Points curve down on the left only**: left-skewed.

**What to do**: if the Q-Q plot deviates substantially from the diagonal, and your method requires Normality, apply a transformation (log, square root, Box-Cox) and check again.

---

### The binning-free alternative: ECDF

Both histograms and box plots require decisions that can obscure the data. The empirical cumulative distribution function (ECDF) requires no decisions at all.

The ECDF at a value $x$ answers: "What fraction of the data is less than or equal to $x$?"

$$\hat{F}(x) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}[X_i \leq x]$$

You plot this as a step function. Every data point is represented exactly; nothing is binned or smoothed. The full distributional shape is visible: the location of the median (where the curve crosses 0.5), the spread (how steeply the curve rises), multiple modes (visible as flat regions between steep rises), and outliers (isolated steps far from the bulk).

**Most important use of the ECDF**: comparing two distributions. Overlay the ECDF of a feature from the training set and the test set. Any horizontal gap at any quantile means those quantiles differ between the two populations. This is also the basis of the KS test statistic (see Section 9).

---

## 3. Bivariate Analysis

### You can't find nonlinearity without looking at pairs

Pearson correlation between two variables is a single number. It captures linear association only. If you have a parabolic relationship — feature increases, then decreases — the Pearson correlation can be exactly zero while the feature is perfectly predictive. Looking at the scatter plot immediately reveals this; the correlation alone does not.

**Scatter plots** are the primary tool for continuous × continuous relationships. Add a LOWESS (locally weighted scatterplot smoothing) trend line to see the conditional mean without imposing a functional form. Deviations from linearity are immediately visible.

**When scatter plots fail**: for large datasets ($n > 10^5$), individual points overlap completely (overplotting). You cannot tell whether the center of the plot has 1,000 or 100,000 points. Solutions:
- **Hex bins**: divide the plane into hexagonal cells, color by count. Density becomes visible. Use `kind='hex'` in pandas or `hexbin` in matplotlib.
- **Alpha transparency**: set point opacity to 0.1 so dense regions appear darker. Quick but less precise than hex bins.
- **2D KDE contour plot**: smooth density estimate in two dimensions; contour lines show equal-density regions.

**Violin plots** combine the shape information of KDE with the compact comparison use of box plots. For each category, a violin shows the full density shape on both sides of a central box. Use when comparing the distribution of a continuous feature across a small number of categories (up to ~10). Beyond that, the plot becomes cluttered.

**Pivot tables and grouped aggregations**: when you have a categorical feature and a numeric outcome, the first question is how the mean and spread of the outcome vary across categories. `df.groupby('category').agg({'metric': ['mean', 'median', 'std', 'count']})` answers this directly. Always include count — a category with mean 95% and n=3 is not meaningful.

**Bar plots for categorical features vs target**: always show uncertainty. A bar at height 0.4 with no error bar is uninterpretable — it could come from 10 samples or 100,000. Add standard error bars or annotate with sample size.

---

## 4. Multivariate Analysis

### The curse of looking at one feature at a time

Univariate analysis misses interactions. A feature that has no marginal relationship with the target might be strongly predictive when combined with another feature (XOR pattern). Conversely, two features might each correlate with the target, but if they are perfectly correlated with each other, one adds nothing.

**Pair plots** (`seaborn.pairplot`) show every pairwise scatter on a grid, with KDE on the diagonal. Color by class to see whether class boundaries are visible in any feature pair. This is the fastest way to identify: which pairs of features are correlated, which feature pairs show clear class separation, and whether any single feature dominates.

**Correlation heatmaps**: `df.corr()` computes pairwise Pearson correlation. Plot with `seaborn.heatmap(annot=True)`. Look for:
- Clusters of highly correlated features (e.g., all > 0.8 with each other) — these are redundant. Including all of them in a regularized model is fine; including them in a logistic regression inflates coefficient variance.
- Features with near-zero correlation with everything including the target — these are likely noise.
- Any feature with suspiciously high correlation with the target (> 0.95) — potential leakage.

Use Spearman rank correlation instead of Pearson when features are ordinal or when the relationship is monotone but nonlinear. `df.corr(method='spearman')`.

**PCA biplots**: project the data onto the first two principal components and plot. Overlay arrows showing where each original feature points in PC space. Features with long arrows that align with a PC drive variance along that direction. Features with short arrows contribute little total variance. This is a fast way to understand which features dominate and whether samples cluster.

**Parallel coordinates**: each vertical axis represents one feature, each line represents one sample, lines are colored by class. When lines of the same class cluster together on multiple axes, those features jointly separate the classes. This is most useful for datasets with fewer than ~20 features.

---

## 5. Missing Data: Why the Mechanism Matters More Than the Amount

You have a column with 20% missing values. Can you impute the mean and move on?

It depends entirely on *why* the values are missing. The same 20% missingness rate can be harmless or catastrophic depending on the mechanism. Getting the mechanism wrong leads to biased models even with sophisticated imputation.

### MCAR: the lucky case

**Missing Completely At Random** means the probability that a value is missing has no relationship with any variable in the dataset — observed or unobserved. A sensor fails randomly. A survey respondent skips a question by accident.

**Consequence**: the observed data is a random subsample of the full data. Mean imputation or complete-case analysis (simply dropping rows with missing values) does not introduce bias. It does lose statistical power, but it does not distort estimates.

**How to test**: compare the distribution of other variables between rows where the column is present vs. missing. Under MCAR, these distributions should not differ. Any systematic difference falsifies MCAR.

**What breaks if you assume MCAR when it's not**: complete-case analysis on MAR or MNAR data introduces selection bias. Your training set is no longer representative of the population you're predicting on.

---

### MAR: the fixable case

**Missing At Random** means the probability of missingness depends on other *observed* variables, but not on the missing value itself. Income is missing more often for younger respondents. The missing values are not random overall, but conditional on age (observed), the missingness is random.

**Consequence**: you can model the missing data mechanism using the observed variables. Multiple imputation by chained equations (MICE) and KNN imputation are valid here because they use observed variables to predict the missing values.

Complete-case analysis introduces bias: the rows with observed values are not representative of all rows because missingness is correlated with age, which in turn is correlated with income. Dropping them skews your estimate of the income distribution.

**MICE mechanics**: for each column with missing values, fit a regression model predicting that column from all other columns. Impute the missing values with predictions from this model (plus noise). Repeat this process cycling through all columns multiple times until the imputed values stabilize. Run this entire process $m$ times (typically 5–10) to get $m$ completed datasets; combine estimates across them using Rubin's rules.

---

### MNAR: the unfixable case

**Missing Not At Random** means the missing value itself predicts whether it's missing. High earners don't report income. Patients with severe symptoms drop out of a trial. Devices that are malfunctioning the most are the ones least likely to report data.

**Consequence**: there is no general statistical fix. The missing data carries information that by definition is not in your dataset. Imputation from observed data will systematically underestimate (or overestimate) the missing values.

**What you can do**:
- **Sensitivity analysis**: impute under different assumptions (lower bound, upper bound, mean) and check how much your conclusions change.
- **Selection models**: model the missingness mechanism explicitly alongside the outcome model.
- **Domain fill**: use external knowledge about what the missing values likely are.
- **Flag and include**: create a binary `is_missing` indicator for the column and include it as a feature. The missingness itself may be the most predictive signal.

**The `is_missing` flag rule**: regardless of mechanism, always create a binary missingness indicator when imputing. If the missingness is informative (which it often is, especially under MNAR), the model can use this flag. If missingness is MCAR, the flag has no predictive power and the model will ignore it. The cost of adding it is low; the cost of omitting it when it matters is high.

---

### Imputation strategy selection

| Mechanism | Valid strategies |
| :--- | :--- |
| MCAR | Mean/median imputation, complete-case analysis |
| MAR | MICE, KNN imputation, model-based imputation |
| MNAR | Sensitivity analysis, selection model, domain knowledge, flag + arbitrary fill |
| Categorical (any) | Mode imputation or dedicated "Unknown" category (preferred: unknown preserves the information that the value was absent) |

---

## 6. Outliers: The Question Is Always "Why?"

A value is extreme. Before you touch it, you need to answer: is this an error, or is this real?

The answer changes the action completely. Deleting a legitimate extreme value introduces bias. Keeping a typo corrupts your model.

### IQR fences

The interquartile range ($\text{IQR} = Q3 - Q1$) measures the spread of the middle half of the data. Tukey's fences define plausible range as:

$$\text{Lower fence} = Q1 - 1.5 \times \text{IQR}$$
$$\text{Upper fence} = Q3 + 1.5 \times \text{IQR}$$

Points outside these fences are flagged as potential outliers. The $1.5 \times$ factor is a convention calibrated so that for a Normal distribution, roughly 0.7% of points fall outside — about 7 per 1,000. Extend to $3 \times \text{IQR}$ for "extreme" outliers (roughly 1 per 100,000 under Normality).

**Why this is better than raw distance from mean**: IQR is based on quantiles, so it is not inflated by the outliers themselves. If you have a heavy-tailed distribution, the IQR still describes the middle 50% accurately; the mean and standard deviation do not.

---

### Z-score and why the standard z-score fails

The z-score standardizes each value: $z_i = (x_i - \bar{x}) / s$. Flag $|z| > 3$.

**The masking problem**: the mean $\bar{x}$ and standard deviation $s$ are themselves inflated by outliers. A dataset with a handful of enormous values will have a large $s$, making the z-scores of the extreme values smaller than they should be. The very outliers you are trying to detect pull the threshold toward themselves.

**Modified z-score** fixes this by using the median $\tilde{x}$ and median absolute deviation (MAD):

$$M_i = \frac{0.6745(x_i - \tilde{x})}{\text{MAD}}$$

where $\text{MAD} = \text{median}(|x_i - \tilde{x}|)$. The constant 0.6745 scales MAD to be consistent with the standard deviation for Normal data. Flag $|M_i| > 3.5$.

The median and MAD are not inflated by outliers, so the denominator stays in the range of typical variation regardless of how extreme the extreme values are.

---

### Domain rules catch what statistics miss

Statistical outlier detection only knows that a value is far from other values. It does not know that age cannot be negative, that a percentage cannot exceed 100, that a price cannot be in the millions when all others are in the hundreds, or that a timestamp cannot be in the future.

Enumerate the valid range for every feature before running statistical outlier detection. Values outside the valid domain are definitionally errors regardless of what the IQR fence says.

---

### What to do with outliers

| Diagnosis | Action |
| :--- | :--- |
| Confirmed data error (typo, sensor glitch, system default) | Remove the value (set to NaN, then handle as missing) |
| Legitimate extreme value (a hedge fund with $100B AUM) | Keep; consider log-transforming the feature |
| Extreme value with outsized influence on a linear model | Winsorize: cap at the 99th percentile. This preserves the observation but limits its influence |
| Outlier cluster suggesting a distinct sub-population | Model separately or add a binary indicator feature |

Winsorizing (capping) is not the same as removing. You keep the observation in the dataset; you just replace the extreme value with the boundary value. This is appropriate when the extreme value is plausible but you don't want it to dominate a linear model's coefficients.

---

## 7. Data Quality Dimensions: Problems Statistics Don't Catch

Distribution analysis tells you about the shape of data. It does not tell you whether the data is correct, current, unique, or self-consistent. These are the data quality dimensions.

### Completeness

**The problem it addresses**: required values are absent. A customer record missing an email address cannot receive communications. A loan application missing income cannot be properly underwritten. Missingness in required fields is a pipeline failure, not a statistical feature.

**Completeness** = the proportion of required values that are present. Failing completeness means you cannot perform the intended operation, not just that your model will be slightly off.

---

### Consistency

**The problem**: the same entity is described differently in two places. The product table says price is $50; the order table says price is $45. Both cannot be right.

Inconsistency is invisible to statistical summaries of a single table. It only appears when you join or compare. In EDA, check:
- Are primary keys consistent between joined tables?
- Do derived columns match the columns they are derived from (e.g., does `total = quantity × price` hold)?
- Are categorical codes consistent with lookup tables?

---

### Timeliness

**The problem**: the data was correct at the time it was collected but is now stale. Revenue data two months old, credit scores six months old, inventory levels from yesterday morning.

Stale data causes models to make decisions based on a world that no longer exists. In EDA, check the timestamp range of your data and compare it to what was expected. Feature stores and data pipelines often have silent staleness bugs.

---

### Accuracy

**The problem**: the value is present and in the right format but is simply wrong. Age = 999 because the ETL used 999 as a null sentinel. A categorical column containing "Male" and "male" and "M" as three distinct categories, all meaning the same thing.

Accuracy cannot be verified from the data alone without an external reference or domain knowledge. Checks:
- Do numeric ranges make physical sense?
- Do categorical values match a controlled vocabulary?
- For joins on names/addresses, are string representations normalized?

---

### Uniqueness

**The problem**: the same real-world entity appears multiple times. Duplicate customer records inflate counts, double-count revenue, and cause a single customer's behavior to be treated as two independent observations. This directly biases model training.

```python
df.duplicated().sum()                                     # exact row duplicates
df.duplicated(subset=['user_id', 'date']).sum()           # key duplicates
```

Deduplicate before any analysis, not after. A duplicate row in a join table can fan out to massive counts.

---

### Validity

**The problem**: the value is present, possibly even accurate, but does not conform to the expected format or range. A date column containing "N/A" strings. A phone number field containing email addresses. A zip code with 7 digits.

Validity checks are schema enforcement applied to actual data. Check format (regex for structured fields), range (min/max for numerics), and referential integrity (foreign keys match primary keys in referenced table).

---

## 8. Cardinality Issues

### When a categorical column is actually an ID

You encode a categorical column with one-hot encoding. The column is "product_id" with 50,000 unique values. You now have 50,000 binary features, 49,999 of which are zero for any given row.

**Why this is almost always wrong**: a column with near-unique values per row carries no generalizable signal — each level appears so rarely that the model cannot learn a reliable effect for it. High cardinality is not itself the problem; *uninformative high cardinality* is.

**The test**: if the number of unique values is close to the number of rows, it is probably an identifier, not a feature. Identifiers are used to join tables, not to train models. Including them causes the model to memorize training examples rather than learning patterns.

**Encoding alternatives when high-cardinality categories are genuinely meaningful**:
- **Target encoding**: replace each category with its mean target value (with regularization to prevent overfitting on rare categories). Leaks signal from the target if not done in a cross-validation fold; always compute within folds.
- **Embeddings**: learn a dense vector representation per category during model training. This is standard for neural networks dealing with user IDs or product IDs in recommendation systems.
- **Frequency encoding**: replace each category with its frequency of occurrence. Captures rarity without creating sparse features.

---

### Date parsing failures

Dates are among the most error-prone fields in practice. Mixed formats ("2024-01-15" vs "15/01/2024" vs "Jan 15, 2024"), timezone naivety (two timestamps subtracted across a daylight saving boundary give wrong durations), and month/day swaps (American MM/DD vs European DD/MM) are all common.

In EDA:
- Parse all date fields with explicit format strings; do not rely on auto-parsing.
- Check the range: are there dates in the future? Before the business existed? These are sentinel values or errors.
- After parsing, create numeric features (day of week, month, days since reference date) rather than passing raw datetime objects to models.

---

## 9. Distribution Shift Between Train and Test

### Why a held-out test set can still fool you

You split data 80/20 and hold out the 20% as a test set. You evaluate your model on it and get strong performance. You deploy. Performance collapses.

The test set you evaluated on looked like the training data because it came from the same time period and the same source. Production data comes from a later time period, a different user population, or runs through a slightly different feature pipeline. The distribution has shifted. Your model learned patterns that held in the past but don't hold now.

Distribution shift must be detected *before* deployment, not after.

---

### KS test: a formal test for distribution difference

The Kolmogorov-Smirnov test compares the ECDFs of a feature in two populations (e.g., train vs. test, or last month vs. this month). The test statistic is the maximum absolute difference between the two ECDFs:

$$D = \sup_x |\hat{F}_1(x) - \hat{F}_2(x)|$$

A large $D$ means the two populations are distributionally different. The associated p-value tests whether this difference could have arisen by chance under the null hypothesis that both samples come from the same distribution.

**Interpretation**: a statistically significant KS test on a feature between train and test does not necessarily mean the model will fail — it means the feature's distribution has changed, and you should investigate whether that change affects the predictions the model makes.

**Limitation**: the KS test has very high power for large samples. With $n = 100{,}000$ it will detect trivially small differences as statistically significant. Use the test statistic $D$ (not just the p-value) to gauge practical significance.

---

### PSI: a monitoring-grade metric

The Population Stability Index (PSI) was developed for monitoring deployed models and is widely used in credit scoring and financial modeling.

$$\text{PSI} = \sum_{b=1}^{B} \left( (\text{Actual\%}_b - \text{Expected\%}_b) \times \ln \frac{\text{Actual\%}_b}{\text{Expected\%}_b} \right)$$

where $b$ indexes bins of the feature, "Expected" is the distribution at training time, and "Actual" is the current distribution.

**Interpretation thresholds** (conventional, not universal):
- PSI < 0.1: distribution is stable. Model can be trusted.
- 0.1 ≤ PSI < 0.25: moderate shift. Investigate which bins changed. Monitor closely.
- PSI ≥ 0.25: major shift. The training distribution and current distribution are substantially different. The model should be retrained before continued use.

**PSI vs KS**: PSI is summary statistic computed over a fixed binning; KS uses the full empirical distribution. KS is more principled statistically; PSI is more interpretable and has established industry thresholds. Use both.

---

### Adversarial validation: the most powerful shift detector

Train a binary classifier to distinguish training set rows (label 0) from test set rows (label 1). If train and test come from the same distribution, the classifier should perform at chance (AUC ≈ 0.5). If AUC is high, the classifier has found systematic differences between train and test.

**Why this is powerful**: it detects shift in any direction and any subspace simultaneously, including joint shifts that per-feature tests miss. A classifier can discover that the combination of feature A and feature B is different between train and test even when neither A nor B shifts individually.

**What to do with a high AUC**: inspect the features with the highest importance in the adversarial classifier. Those are the features that most distinguish train from test. Investigate whether those features have pipeline bugs, changed semantics, or represent genuine population drift.

---

### Common causes of distribution shift

- **Temporal drift**: model trained on historical data, deployed on future data. Seasonal patterns, trends, and regime changes all cause the feature distribution to evolve over time.
- **Population shift**: the user base, customer base, or data source changes. A model trained on desktop users deployed to mobile users.
- **Feature pipeline bug**: training code and serving code compute a feature differently. Training uses `fillna(0)` after a join; serving code fills with the mean. The distributions diverge.
- **Label shift**: the class proportions change. If your model was trained on 10% fraud rate and fraud rises to 30%, calibration breaks even if the feature distributions are stable.

---

## 10. Leakage Detection

### The most dangerous form of model failure

Leakage produces models with spectacular in-sample performance and catastrophic real-world performance. It is more insidious than ordinary overfitting because it can be invisible in standard validation.

Leakage occurs when information from outside the prediction horizon — information that would not be available at the time a prediction must be made — enters the training data.

---

### Post-event features

You are predicting whether a customer will churn. One of your features is "number of support tickets submitted in the last 30 days." If you measure this feature at the time of evaluation (today), you are using tickets submitted *after* the churn event for customers who have already churned. The model learns that churned customers have high ticket counts — which is true — but at prediction time, for a customer you haven't lost yet, you can only see tickets up to the current date. The feature value at training time ≠ the feature value at prediction time.

**Fix**: compute every feature using only information available before the label observation time. This requires explicit temporal discipline: join features on the condition `feature_timestamp < label_timestamp`.

---

### Proxy targets

A feature is not the target itself but is derived from it or caused by it. A "diagnosis confirmation" flag in a medical dataset is a perfect proxy for the diagnosis label you are trying to predict. An "order status = returned" column predicts whether an order will be returned because it is the outcome.

Proxy targets are often not obvious. "Number of medications prescribed" is a proxy for disease severity when predicting hospitalization. "Last login was more than 90 days ago" is nearly the definition of the churn label.

**Detection**: sort features by correlation with the target. Features with correlation above 0.9 should be investigated manually. Ask: could this feature only be known after the target is determined?

---

### Time leakage in cross-validation

Standard $k$-fold cross-validation randomly assigns rows to folds. If your data has a temporal dimension, rows from the future can end up in the training fold and rows from the past in the validation fold. The model sees the future when training, which is impossible in production.

**Fix**: time-series cross-validation (forward chaining). Training folds contain only data prior to the validation fold. Fold 1: train on month 1, validate on month 2. Fold 2: train on months 1–2, validate on month 3. And so on.

---

### Target in features

The target variable or a linear transformation of it appears as a feature. This sounds too obvious to occur in practice. It does occur, often in aggregate form: a column `conversion_rate` is present in the dataset and it was computed as `conversions / sessions`, where `conversions` is functionally equivalent to the target `converted`.

**Detection**: train a model and inspect feature importances. If one feature explains almost all variance or has very high importance, verify from first principles that it could not possibly be derived from the target.

---

## 11. Pandas Patterns for EDA

Every step of EDA has a corresponding pandas idiom. The patterns below cover the full workflow.

```python
# --- Shape, types, overview ---
df.shape                                        # (rows, cols)
df.dtypes                                       # inferred dtype per column
df.info()                                       # dtype + non-null count per column
df.describe(include='all')                      # numeric and categorical summaries

# --- Missingness ---
df.isnull().sum().sort_values(ascending=False)          # count per column
df.isnull().mean().sort_values(ascending=False)         # fraction per column
df.isnull().mean()[df.isnull().mean() > 0]             # only columns with any missing

# --- Univariate distribution ---
df['col'].value_counts(dropna=False, normalize=True)    # frequency table incl. NaN
df['col'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])

# --- Bivariate: numeric vs categorical ---
df.groupby('category').agg({'metric': ['mean', 'median', 'std', 'count']})

# --- Cross-tabulation ---
pd.crosstab(df['cat1'], df['cat2'], normalize='index')  # row-normalized

# --- Correlation ---
df.corr(method='pearson')                               # linear
df.corr(method='spearman')                              # monotone, rank-based
df.corr()['target'].sort_values(ascending=False)        # all features vs target

# --- Duplicates ---
df.duplicated().sum()                                    # exact row duplicates
df.duplicated(subset=['user_id', 'date']).sum()         # key duplicates

# --- Outliers: IQR fence ---
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['col'] < Q1 - 1.5 * IQR) | (df['col'] > Q3 + 1.5 * IQR)]

# --- Modified z-score ---
median = df['col'].median()
MAD = (df['col'] - median).abs().median()
df['mod_z'] = 0.6745 * (df['col'] - median) / MAD
df[df['mod_z'].abs() > 3.5]

# --- Missingness indicator ---
df['col_missing'] = df['col'].isnull().astype(int)

# --- Cardinality audit ---
df.nunique().sort_values(ascending=False)               # unique count per column
df.nunique() / len(df)                                  # fraction; near 1.0 = likely ID

# --- Time series aggregation ---
df.set_index('date').resample('W').agg({'revenue': 'sum', 'orders': 'count'})
```

---

## 12. Profiling Tools

### The problem with manual EDA on wide datasets

Running the pandas patterns above on a dataset with 200 columns is tedious. You will miss columns. You will notice that column 47 has 30% missingness but forget to check whether it correlates with anything. Automated profiling tools run the full EDA checklist and compile results into a browsable report.

---

### ydata-profiling

```python
from ydata_profiling import ProfileReport
report = ProfileReport(df, title="EDA Report", explorative=True)
report.to_file("report.html")
```

What the report contains:
- Per-column distribution histogram, descriptive statistics, and missingness count.
- A missingness matrix showing which columns are simultaneously missing on the same rows — structure in joint missingness is a strong indicator of MAR.
- A correlation matrix across all numeric features.
- Duplicate row count.
- Automatic warnings for: constant columns (zero variance), high cardinality columns, high missingness, columns with high correlation to other columns.

**Limitation**: profiling tools tell you what is there, not whether it is correct. They flag a column with 1,000 unique values as high-cardinality; they cannot tell you whether those values are valid product IDs or erroneous free-text entries. Human judgment is still required after the report.

---

### Great Expectations: from exploration to enforcement

Discovering in EDA that `age` is always between 0 and 120 is useful. Ensuring that this property holds tomorrow, and next week, and after every pipeline run — that requires encoding it as a formal expectation and checking it automatically.

Great Expectations allows you to define **expectations** — assertions about data properties — and then validate each new batch of data against them.

```python
import great_expectations as gx

context = gx.get_context()
validator = context.sources.pandas_default.read_dataframe(df)

validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_be_between("age", min_value=0, max_value=120)
validator.expect_column_unique_value_count_to_be_between("status", min_value=2, max_value=6)
validator.expect_column_values_to_be_in_set("gender", {"M", "F", "Unknown"})
validator.expect_column_pair_values_a_to_be_greater_than_b("end_date", "start_date")

validator.save_expectation_suite()
```

Expectations are saved and re-run on every new data batch. Failures generate a structured report (data docs) showing which expectations passed and which failed, with examples of failing rows.

**Where Great Expectations fits in the workflow**: use it to productionize the findings from EDA. Every anomaly you discover manually in exploration is a candidate expectation. The transition from "I noticed during exploration that X" to "the pipeline will fail loudly if X is violated in the future" is what transforms EDA findings into durable data quality guarantees.

**What breaks**: expectations encode your understanding of the data at a point in time. If the valid range of a feature legitimately changes (a new product category is introduced), the expectation becomes wrong and will generate spurious failures. Expectations need maintenance as the system evolves.

## Visualization Code Patterns

**Purpose**: runnable matplotlib/seaborn code for the most common EDA plots. Each snippet is self-contained given a pandas DataFrame `df`.

---

### Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Consistent aesthetics
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
```

---

### Univariate: Continuous Features

```python
# Histogram + KDE overlay for a single feature
def plot_distribution(df, col, bins=50):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Histogram with KDE
    axes[0].hist(df[col].dropna(), bins=bins, edgecolor='white', alpha=0.7)
    axes[0].set_title(f'{col} — histogram')
    axes[0].set_xlabel(col)

    # Box plot (identifies outliers)
    axes[1].boxplot(df[col].dropna(), vert=True, patch_artist=True)
    axes[1].set_title(f'{col} — box plot')

    plt.tight_layout()
    plt.show()

# KDE with seaborn (continuous density; better than histogram for shape)
sns.kdeplot(df[col], fill=True)
plt.title(f'Density: {col}')
plt.show()

# Q-Q plot to test normality
fig, ax = plt.subplots(figsize=(5, 5))
stats.probplot(df[col].dropna(), dist="norm", plot=ax)
ax.set_title(f'Q-Q plot: {col}')
plt.show()
```

---

### Univariate: Categorical Features

```python
# Bar chart of value counts (sorted by frequency)
def plot_categorical(df, col, top_n=20):
    counts = df[col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 4))
    counts.plot(kind='bar', ax=ax, edgecolor='white')
    ax.set_title(f'{col} — top {top_n} values')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    # Add percentage labels
    total = len(df[col].dropna())
    for p in ax.patches:
        ax.annotate(f'{p.get_height()/total:.1%}',
                    (p.get_x() + p.get_width()/2, p.get_height()),
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.show()
```

---

### Bivariate: Continuous vs. Continuous

```python
# Scatter plot with regression line and correlation annotation
def plot_scatter(df, x_col, y_col, sample_n=5000):
    sample = df[[x_col, y_col]].dropna().sample(min(sample_n, len(df)))
    r, p = stats.pearsonr(sample[x_col], sample[y_col])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sample[x_col], sample[y_col], alpha=0.3, s=10)
    # Add regression line
    m, b = np.polyfit(sample[x_col], sample[y_col], 1)
    x_range = np.linspace(sample[x_col].min(), sample[x_col].max(), 100)
    ax.plot(x_range, m * x_range + b, 'r-', linewidth=2)
    ax.set_title(f'{x_col} vs {y_col}  |  r = {r:.3f}, p = {p:.3e}')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    plt.show()

# Hex bin (for large datasets where scatter is overplotted)
df.plot.hexbin(x=x_col, y=y_col, gridsize=30, cmap='Blues', figsize=(7, 5))
plt.title(f'{x_col} vs {y_col} — density')
plt.show()
```

---

### Bivariate: Continuous vs. Categorical (A/B comparison)

```python
# Overlapping KDEs per group
def plot_group_distributions(df, value_col, group_col):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # KDE per group
    for group, subset in df.groupby(group_col):
        sns.kdeplot(subset[value_col].dropna(), label=str(group),
                    fill=True, alpha=0.3, ax=axes[0])
    axes[0].set_title(f'{value_col} by {group_col} — KDE')
    axes[0].legend()

    # Box plot per group
    df.boxplot(column=value_col, by=group_col, ax=axes[1])
    axes[1].set_title(f'{value_col} by {group_col} — box plot')
    plt.suptitle('')  # suppress default title from boxplot
    plt.tight_layout()
    plt.show()
```

---

### Correlation Heatmap

```python
def plot_correlation_heatmap(df, method='pearson', figsize=(12, 10)):
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr(method=method)

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f',
        cmap='coolwarm', center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax
    )
    ax.set_title(f'Correlation matrix ({method})')
    plt.tight_layout()
    plt.show()

    # Flag suspiciously high correlations (possible leakage or redundancy)
    high_corr = (corr.abs() > 0.9) & (corr.abs() < 1.0)
    pairs = [(corr.index[i], corr.columns[j])
             for i, j in zip(*np.where(high_corr & ~mask))]
    if pairs:
        print("High correlation pairs (|r| > 0.9):", pairs)
```

---

### Missing Data Visualization

```python
# Missingness heatmap (rows × columns, missing = white)
def plot_missing(df, figsize=(14, 6)):
    missing_pct = df.isnull().mean().sort_values(ascending=False)
    cols_with_missing = missing_pct[missing_pct > 0].index

    if len(cols_with_missing) == 0:
        print("No missing values.")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar chart of missingness %
    missing_pct[cols_with_missing].plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Missingness by column')
    axes[0].set_ylabel('Fraction missing')
    axes[0].tick_params(axis='x', rotation=45)

    # Heatmap of missing patterns (sample 500 rows for readability)
    sample = df[cols_with_missing].sample(min(500, len(df)))
    sns.heatmap(sample.isnull(), cbar=False, yticklabels=False,
                cmap=['#34495e', '#e74c3c'], ax=axes[1])
    axes[1].set_title('Missing pattern (red = missing)')

    plt.tight_layout()
    plt.show()
```

---

### Time Series EDA

```python
def plot_time_series(df, date_col, value_col, freq='D'):
    ts = df.set_index(date_col)[value_col].resample(freq).mean()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    # Raw series
    axes[0].plot(ts.index, ts.values)
    axes[0].set_title(f'{value_col} over time ({freq} aggregation)')

    # Rolling statistics (7-period)
    window = 7
    axes[1].plot(ts.index, ts.values, alpha=0.3, label='raw')
    axes[1].plot(ts.index, ts.rolling(window).mean(), label=f'{window}-period MA')
    axes[1].fill_between(ts.index,
                          ts.rolling(window).mean() - ts.rolling(window).std(),
                          ts.rolling(window).mean() + ts.rolling(window).std(),
                          alpha=0.2, label='±1 std')
    axes[1].legend()
    axes[1].set_title('Rolling mean ± std')

    # Distribution of values (to check for seasonality-induced bimodality)
    axes[2].hist(ts.dropna().values, bins=50, edgecolor='white', alpha=0.7)
    axes[2].set_title('Distribution of values')

    plt.tight_layout()
    plt.show()
```

---

### Pairplot (multivariate overview)

```python
# seaborn pairplot: scatterplot matrix + histograms on diagonal
# Best for datasets with 3-8 numeric features
sns.pairplot(
    df[numeric_cols + [target_col]],
    hue=target_col,           # color points by target (classification)
    diag_kind='kde',
    plot_kws={'alpha': 0.4, 's': 20},
    corner=True               # show only lower triangle
)
plt.suptitle('Pairplot', y=1.02)
plt.show()
```

---

## Storytelling with Data: Communicating EDA Findings

Good analysis that is poorly communicated does not move decisions. The structure below applies to presenting EDA results to stakeholders (product managers, executives, cross-functional partners).

---

### The SCQA Framework

Structure every analytical narrative using four elements:

1. **Situation**: what context does the audience already know? Establish shared ground without re-explaining things they know.
   - "We launched the new checkout flow in Q3 across all platforms."

2. **Complication**: what changed or what problem arose? This is the tension that makes the analysis necessary.
   - "Revenue per session declined 8% in the two weeks following launch."

3. **Question**: what specific question does your analysis answer?
   - "Is the decline caused by the new checkout flow, or by external factors?"

4. **Answer**: state your conclusion upfront. Do not bury it at the end.
   - "The decline is real and caused by the flow change: a 3-step friction increase in mobile checkout accounts for 90% of the drop."

The rest of the presentation is evidence supporting the Answer, not a chronological retelling of how you did the analysis.

---

### Slide / Report Structure

Each analytical communication needs exactly these sections (adapt length to audience):

| Section | Purpose | Common mistake |
| :--- | :--- | :--- |
| **Executive summary** (1 slide) | Answer + 2-3 supporting bullets | Omitting it; burying the answer on slide 12 |
| **Context** (1-2 slides) | Data source, date range, metric definitions | Drowning in methodology before showing results |
| **Finding** (2-4 slides) | Main result with supporting evidence | Showing 20 charts; audience cannot identify the finding |
| **So what** (1 slide) | Concrete recommendation or next step | Ending with "more analysis needed" (non-answer) |
| **Appendix** | Details for people who want to go deeper | Putting appendix-level detail in the main flow |

---

### Chart Selection Rules

Match chart type to the question being answered:

| Question | Chart type |
| :--- | :--- |
| How is X distributed? | Histogram, KDE, box plot |
| How does X change over time? | Line chart (not bar chart) |
| How do X and Y relate? | Scatter plot, hex bin |
| How do groups compare on X? | Box plots side-by-side, strip plot |
| What is the composition of X? | Stacked bar (avoid pie charts) |
| Which categories are largest? | Horizontal bar, sorted descending |
| How correlated are many variables? | Heatmap |

**Rules**:
- One chart, one message. Title = the conclusion, not the variable name. "Revenue declined after launch" > "Revenue by week".
- Annotate directly on the chart (arrows, callout boxes) rather than expecting the reader to match a legend to a line.
- Always label axes with units. "Sessions (millions)" not "Sessions".
- Remove chartjunk: gridlines, 3D effects, dual y-axes (almost always misleading).
- Color is for encoding information, not decoration. Use one highlight color to draw attention to what matters.

---

### Numbers-First Communication

When speaking to a technical audience or writing a data doc:

- Lead with the number and its uncertainty: "Conversion rate declined from 4.2% to 3.8% (95% CI: [-0.6%, -0.2%], n = 2.1M sessions)."
- State the statistical test used and why: "We used Welch's t-test (not Student's) because group sizes and variances differed."
- Call out what could be wrong: "One confound we cannot rule out: the launch coincided with a competitor's sale event."
- Separate what you know from what you inferred: "The data shows X directly. We infer Y from X, which requires assuming Z."

---

### Handling Uncertainty

Stakeholders often want a definitive answer when the data is ambiguous. Productive framing:

- Present a range of scenarios: "If the decline is entirely attributable to the checkout flow, the expected revenue impact is -$2M/month. If it's 50% attributable, the impact is -$1M/month."
- Distinguish signal from noise quantitatively: report effect sizes and confidence intervals, not just p-values.
- Make the decision criteria explicit: "We should roll back if the decline persists for another week with p < 0.05 on the primary metric."
- Name the assumption that drives your recommendation: "This recommendation holds if we believe the parallel trends assumption is valid. Here is the evidence for that assumption."

---

## Flashcards

**40% of rows had a target-adjacent column accidentally left in the feature set.?** #flashcard
40% of rows had a target-adjacent column accidentally left in the feature set.

**The "age" column contained the default fill value 999 for nulls.?** #flashcard
The "age" column contained the default fill value 999 for nulls.

**Train and test were sampled from different time periods, so the test set looks nothing like production.?** #flashcard
Train and test were sampled from different time periods, so the test set looks nothing like production.

**Bimodality?** #flashcard
two humps strongly suggest two sub-populations mixed together. This changes what preprocessing and what model families make sense.

**Sharp cutoffs at round numbers?** #flashcard
a flood of values exactly at 100 or 0 typically indicates truncation or a sensor ceiling, not a real concentration of the quantity.

**Spikes at specific values?** #flashcard
a spike at 0, -1, 999, or 9999 almost always indicates a sentinel encoding for null rather than a real data point.

**Extreme right skew?** #flashcard
common for monetary amounts, counts, and durations. Log-transforming makes the distribution more legible and often more useful for linear models.

**The median (horizontal line inside the box)?** #flashcard
the center of the distribution, robust to outliers.

**Q1 and Q3 (box edges)?** #flashcard
the middle 50% of the data.

**Whiskers extending to the most extreme point within $1.5 \times \text{IQR}$ of the quartiles?** #flashcard
the "typical" spread.

**Individual points beyond the whiskers?** #flashcard
flagged as potential outliers.

**S-curve (sigmoid shape)?** #flashcard
the tails are heavier than Normal. Values cluster near zero (the kurtosis is high). Common for financial returns, network latency.

**Inverted S-curve: the distribution is platykurtic?** #flashcard
lighter tails and a flatter center than Normal.

**Points curve up on the right only?** #flashcard
right-skewed distribution. The largest values are larger than the Normal would predict.

**Points curve down on the left only?** #flashcard
left-skewed.

**Hex bins?** #flashcard
divide the plane into hexagonal cells, color by count. Density becomes visible. Use kind='hex' in pandas or hexbin in matplotlib.

**Alpha transparency?** #flashcard
set point opacity to 0.1 so dense regions appear darker. Quick but less precise than hex bins.

**2D KDE contour plot?** #flashcard
smooth density estimate in two dimensions; contour lines show equal-density regions.

**Clusters of highly correlated features (e.g., all > 0.8 with each other)?** #flashcard
these are redundant. Including all of them in a regularized model is fine; including them in a logistic regression inflates coefficient variance.

**Features with near-zero correlation with everything including the target?** #flashcard
these are likely noise.

**Any feature with suspiciously high correlation with the target (> 0.95)?** #flashcard
potential leakage.

**Sensitivity analysis?** #flashcard
impute under different assumptions (lower bound, upper bound, mean) and check how much your conclusions change.

**Selection models?** #flashcard
model the missingness mechanism explicitly alongside the outcome model.

**Domain fill?** #flashcard
use external knowledge about what the missing values likely are.

**Flag and include?** #flashcard
create a binary is_missing indicator for the column and include it as a feature. The missingness itself may be the most predictive signal.

**Are primary keys consistent between joined tables?** #flashcard
Are primary keys consistent between joined tables?

**Do derived columns match the columns they are derived from (e.g., does total = quantity × price hold)?** #flashcard
Do derived columns match the columns they are derived from (e.g., does total = quantity × price hold)?

**Are categorical codes consistent with lookup tables?** #flashcard
Are categorical codes consistent with lookup tables?

**Do numeric ranges make physical sense?** #flashcard
Do numeric ranges make physical sense?

**Do categorical values match a controlled vocabulary?** #flashcard
Do categorical values match a controlled vocabulary?

**For joins on names/addresses, are string representations normalized?** #flashcard
For joins on names/addresses, are string representations normalized?

**Target encoding?** #flashcard
replace each category with its mean target value (with regularization to prevent overfitting on rare categories). Leaks signal from the target if not done in a cross-validation fold; always compute within folds.

**Embeddings?** #flashcard
learn a dense vector representation per category during model training. This is standard for neural networks dealing with user IDs or product IDs in recommendation systems.

**Frequency encoding?** #flashcard
replace each category with its frequency of occurrence. Captures rarity without creating sparse features.

**Parse all date fields with explicit format strings; do not rely on auto-parsing.?** #flashcard
Parse all date fields with explicit format strings; do not rely on auto-parsing.

**Check the range?** #flashcard
are there dates in the future? Before the business existed? These are sentinel values or errors.

**After parsing, create numeric features (day of week, month, days since reference date) rather than passing raw datetime objects to models.?** #flashcard
After parsing, create numeric features (day of week, month, days since reference date) rather than passing raw datetime objects to models.

**PSI < 0.1?** #flashcard
distribution is stable. Model can be trusted.

**0.1 ≤ PSI < 0.25?** #flashcard
moderate shift. Investigate which bins changed. Monitor closely.

**PSI ≥ 0.25?** #flashcard
major shift. The training distribution and current distribution are substantially different. The model should be retrained before continued use.

**Temporal drift?** #flashcard
model trained on historical data, deployed on future data. Seasonal patterns, trends, and regime changes all cause the feature distribution to evolve over time.

**Population shift?** #flashcard
the user base, customer base, or data source changes. A model trained on desktop users deployed to mobile users.

**Feature pipeline bug?** #flashcard
training code and serving code compute a feature differently. Training uses fillna(0) after a join; serving code fills with the mean. The distributions diverge.

**Label shift?** #flashcard
the class proportions change. If your model was trained on 10% fraud rate and fraud rises to 30%, calibration breaks even if the feature distributions are stable.

**Per-column distribution histogram, descriptive statistics, and missingness count.?** #flashcard
Per-column distribution histogram, descriptive statistics, and missingness count.

**A missingness matrix showing which columns are simultaneously missing on the same rows?** #flashcard
structure in joint missingness is a strong indicator of MAR.

**A correlation matrix across all numeric features.?** #flashcard
A correlation matrix across all numeric features.

**Duplicate row count.?** #flashcard
Duplicate row count.

**Automatic warnings for?** #flashcard
constant columns (zero variance), high cardinality columns, high missingness, columns with high correlation to other columns.
