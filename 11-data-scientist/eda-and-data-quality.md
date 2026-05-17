# EDA and Data Quality

---

## 1. EDA Workflow

Systematic order prevents premature modeling on dirty data:

1. **Shape and types**: `df.shape`, `df.dtypes`, `df.info()` — check dimensionality, unexpected types (numeric stored as object)
2. **Missingness**: `df.isnull().sum()`, `df.isnull().mean()` — identify columns with high null rates; visualize with `missingno`
3. **Distributions**: univariate plots for all features — identify skewness, multimodality, impossible values
4. **Outliers**: IQR fences, z-scores, scatter inspection — flag, understand, decide whether to remove or cap
5. **Correlations**: correlation matrix, pairwise scatter — identify redundant features, potential leakage
6. **Target relationship**: how each feature relates to the label — scatter for continuous, box plots by category

---

## 2. Univariate Analysis

### Histograms
- Bin count matters: too few → obscures shape; too many → noisy; use Sturges ($k = \lceil\log_2 n + 1\rceil$) or Freedman-Diaconis ($h = 2 \cdot \text{IQR} \cdot n^{-1/3}$) as starting points
- Look for: bimodal (likely two sub-populations), sharp cutoffs (truncation), spikes at round numbers (measurement artifact)

### KDE (Kernel Density Estimate)
- Smoothed continuous estimate of distribution; bandwidth = smoothing parameter
- Better than histogram for overlaying multiple distributions (e.g., target=0 vs target=1)

### Box Plots
- Displays Q1, median, Q3, whiskers (1.5×IQR from quartiles), outlier points
- Compact; good for comparing distributions across categories

### Q-Q Plots
- Plots quantiles of data against theoretical Normal quantiles
- On the diagonal → Normal distribution; S-curve → heavy tails; concave/convex → skewed

### ECDF (Empirical CDF)
- $\hat{F}(x) = \frac{1}{n}\sum \mathbf{1}[X_i \leq x]$ — no binning artifact; shows full distributional shape
- Useful for comparing two distributions visually or via KS test statistic

---

## 3. Bivariate Analysis

- **Scatter plots**: continuous × continuous; add trend line (lowess) to detect non-linearity
- **Hex bins**: scatter plots for large datasets ($n > 10^5$) where overplotting obscures density
- **Violin plots**: distribution shape by group; combines box plot with KDE
- **Pivot tables / grouped aggregations**: `df.groupby(cat).agg({num: ['mean','median','std']})` — understand how numeric feature varies by category
- **Bar plots**: mean/median of outcome by categorical feature; always show confidence intervals or sample size

---

## 4. Multivariate Analysis

- **Pair plots** (`seaborn.pairplot`): scatter matrix for all feature pairs; diagonal = KDE; color by class
- **Correlation heatmap**: `df.corr()` with `seaborn.heatmap(annot=True)` — identify clusters of correlated features
- **PCA biplots**: first two PCs plotted; arrows show feature loadings — reveals which features drive variance
- **Parallel coordinates**: each axis = one feature; each line = one sample; color by class — reveals cluster separation

---

## 5. Missing Data Taxonomy

### MCAR (Missing Completely At Random)
- Missingness unrelated to any variable (observed or unobserved)
- Safe to use complete-case analysis (though loses power)
- Test: compare observed and missing group on other variables — should not differ

### MAR (Missing At Random)
- Missingness depends on observed variables but not on the missing value itself
- Example: income is missing more often for younger respondents (age is observed)
- Multiple imputation (MICE) or model-based imputation is valid
- Complete-case analysis introduces bias if missing data mechanism affects outcome

### MNAR (Missing Not At Random)
- Missingness depends on the unobserved value itself
- Example: high earners don't report income; patients with severe symptoms drop out of trial
- No general fix; requires sensitivity analysis, selection models, or domain knowledge to handle

### Imputation Strategies
| Mechanism | Strategy |
| :--- | :--- |
| MCAR | Mean/median imputation, complete case |
| MAR | MICE (multiple imputation by chained equations), KNN imputation |
| MNAR | Sensitivity analysis; model missingness explicitly; domain fill |
| Categorical | Mode imputation or "Unknown" category |

- Always create a binary `is_missing` flag feature alongside imputed values — missingness may be informative

---

## 6. Outlier Detection in EDA

### IQR Fence
- Lower fence: $Q1 - 1.5 \times \text{IQR}$; Upper fence: $Q3 + 1.5 \times \text{IQR}$
- Points outside fence = potential outliers; 3×IQR fence for "extreme" outliers
- Non-parametric; does not assume Normal distribution

### Z-Score
- $z_i = (x_i - \bar{x}) / s$; flag $|z| > 3$ as outlier
- Sensitive to outliers themselves (masking effect) — use modified Z-score with median and MAD instead:
  $M_i = \frac{0.6745 (x_i - \tilde{x})}{\text{MAD}}$, flag $|M_i| > 3.5$

### Domain Rules
- Negative age, revenue < 0, percentage > 100, timestamps in the future — always check domain plausibility
- Outliers may be valid extreme values or data errors — distinguish via domain knowledge, not just statistics

### Outlier Actions
- Remove: confirmed data errors (sensor malfunction, typos)
- Cap (winsorize): business decision, not error — preserve observation but limit influence
- Keep + transform: log-transform to reduce influence of large values
- Model separately: bimodal population where one group represents a distinct process

---

## 7. Data Quality Dimensions

| Dimension | Definition | Example Failure |
| :--- | :--- | :--- |
| **Completeness** | No missing required values | Customer record missing email |
| **Consistency** | No conflicts across systems or within a record | Same product has different prices in two tables |
| **Timeliness** | Data is current enough for use | Revenue data 2 months stale |
| **Accuracy** | Values are correct | Age = 999 due to default fill |
| **Uniqueness** | No duplicate records | Same user ID appears multiple times |
| **Validity** | Values conform to defined formats/ranges | Date field contains "N/A" strings |

---

## 8. Cardinality Issues

- **High-cardinality categoricals**: ID columns, URLs, raw text — high cardinality ≠ useful for modeling; check if column should be an ID field
- **ID leakage**: user_id, order_id, or session_id accidentally included as features; model learns to memorize IDs rather than patterns
- **Date parsing failures**: mixed formats ("2024-01-15" vs "15/01/2024"), timezone naivety, month/day swap
- **Encoding cost**: one-hot encoding a 10,000-category column → 10,000 sparse features; use target encoding or embedding instead

---

## 9. Distribution Shift Between Train and Test

### Detection
- **KS test (Kolmogorov-Smirnov)**: compares CDFs of a feature in train vs test; statistic = max absolute difference
- **PSI (Population Stability Index)**:
  $\text{PSI} = \sum \left((\text{Actual\%} - \text{Expected\%}) \times \ln\frac{\text{Actual\%}}{\text{Expected\%}}\right)$
  - PSI < 0.1: stable distribution
  - 0.1 ≤ PSI < 0.25: moderate shift, investigate
  - PSI ≥ 0.25: major shift — model likely invalid; retrain
- **Adversarial validation**: train a classifier to distinguish train vs test set; high AUC → significant distribution shift

### Causes
- Temporal drift (model trained on old data, deployed on new)
- Population change (user base shifted)
- Feature pipeline bug (different preprocessing between training and serving)
- Label shift (class proportions changed)

---

## 10. Leakage Detection

- **High correlation with target**: feature correlated > 0.95 with target is suspicious — may be a proxy or post-event feature
- **Post-event features**: features computed after the event being predicted (e.g., "order shipped" used to predict "will order")
- **Proxy targets**: feature derived from or strongly caused by the label (e.g., diagnosis code when predicting diagnosis)
- **Time leakage**: using future data in historical prediction — always enforce strict temporal train/test splits
- **Target in features**: target or close derivative accidentally included in feature set
- **Detection**: train a model, sort by feature importance; top features with unexpected high importance → investigate

---

## 11. Pandas Patterns for EDA

```python
# Overview
df.shape, df.dtypes, df.info(), df.describe(include='all')

# Missingness
df.isnull().sum().sort_values(ascending=False)
df.isnull().mean()[df.isnull().mean() > 0]

# Distribution
df['col'].value_counts(dropna=False, normalize=True)
df['col'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])

# Groupby aggregation
df.groupby('category').agg({'metric': ['mean', 'median', 'std', 'count']})

# Cross-tabulation
pd.crosstab(df['cat1'], df['cat2'], normalize='index')

# Correlation
df.corr(method='pearson')  # or 'spearman' for non-linear
df.corr()['target'].sort_values(ascending=False)

# Duplicates
df.duplicated().sum()
df.duplicated(subset=['user_id', 'date']).sum()

# Time series
df.set_index('date').resample('W').agg({'revenue': 'sum', 'orders': 'count'})
```

---

## 12. Profiling Tools

### ydata-profiling (formerly pandas-profiling)
```python
from ydata_profiling import ProfileReport
report = ProfileReport(df, title="EDA Report", explorative=True)
report.to_file("report.html")
```
- Generates: distributions, missing value matrix, correlations, duplicate detection, interactions
- Warns on: high cardinality, high missingness, constant columns, high correlation

### Great Expectations
- Framework for defining and validating **data contracts** (expectations about data properties)
- Expectations: `expect_column_values_to_not_be_null`, `expect_column_values_to_be_between`, `expect_column_unique_value_count_to_be_between`
- Generates data docs (HTML) showing expectation pass/fail per run
- Integrates into pipelines for automated data quality gates
```python
import great_expectations as gx
context = gx.get_context()
validator = context.sources.pandas_default.read_dataframe(df)
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_be_between("age", min_value=0, max_value=120)
validator.save_expectation_suite()
```
