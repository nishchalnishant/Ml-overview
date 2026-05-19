# Day 5-7: Exploratory Data Analysis (EDA)

## Why This Topic Comes Here

You now know how to build a model (days 1-2) and how to clean the inputs (days 3-4). EDA is placed between preprocessing and modeling because it answers the question: do I actually understand what I'm feeding the model? A model built without EDA is a black box applied to an unknown input — you will not know whether its predictions are sensible, whether important relationships are being captured, or whether the data itself has structural problems that no algorithm can fix. EDA is also the primary place where you decide which features matter and what transformations they need — decisions that precede and shape every modeling choice.

---

## Executive Summary

| Task | Objective | Primary Tools |
|------|-----------|---------------|
| **Univariate** | Distribution Check | Histograms, Boxplots, KDE |
| **Bivariate** | Relationship/Trend | Scatter Plots, Correlation Heatmaps |
| **Multivariate** | Complex Patterns | Pair Plots, Parallel Coordinates |
| **Statistical** | Hypothesis Testing | T-Tests, ANOVA, Chi-Squared |

---

## 1. Statistical Foundations of EDA

**Why descriptive statistics before visualizations:** The numbers tell you where to look; the plots tell you what you're looking at. Computing mean, median, and variance first lets you anticipate what a histogram will show — and catch cases where the plot is misleading (e.g., a bimodal distribution whose mean and variance describe neither peak).

### Descriptive Statistics

- **Central Tendency**: Mean (sensitive to outliers), Median (robust), Mode.
- **Dispersion**: Standard Deviation, Variance, Skewness, Kurtosis.
- **Skewness**: Measure of asymmetry. Positive skew = long tail on the right.

**Key insight:** The mean and standard deviation are sufficient statistics only if the distribution is Gaussian. For real-world data — which is often skewed, multimodal, or heavy-tailed — they can be actively misleading. Always check whether summary statistics actually summarize the shape of the data.

**How to verify understanding:** You have a feature with mean=50, median=12. Describe the likely distribution shape. What does this imply about using mean imputation for missing values in this feature?

**What trips people up:** Using the mean as the default summary statistic for all features. In income data, housing prices, or token frequencies, the mean describes almost no individual in the population. The median is almost always the more informative center for skewed distributions.

### The Power Law (Long Tail)

Many real-world distributions (wealth, city sizes, NLP token frequency) follow a Power Law rather than a Normal distribution.
- **Log-Transform**: Useful for making skewed distributions more Normal.

**Key insight:** A log-transform on a right-skewed feature does not just make a prettier histogram — it can convert a relationship that a linear model cannot learn into one it can. If income and price have a log-linear relationship, a model trained on raw income will underfit the pattern that a model trained on log(income) captures easily.

**How to verify understanding:** Plot log(x) vs y for a feature you suspect is power-law distributed. If it becomes linear, what does that tell you about how to include the feature in a linear model?

**What trips people up:** Applying log transforms without checking for zeros (log(0) is undefined) or negatives. Always check `df['feature'].min()` before transforming. A common fix is `log(x + 1)` — but this changes the interpretability of the feature.

---

## 2. Visual Inspection Strategies

**Why plotting follows statistics:** Statistics can tell you something unusual exists; visualization tells you what it looks like. A correlation coefficient of 0.3 between two variables could represent a weak linear trend, a tight nonlinear curve, or a noisy blob with one influential outlier pulling the number up. The number cannot distinguish these cases; the scatter plot can.

### Identifying Relationships

- **Pearson Correlation ($r$)**: Measures linear relationship. Range $[-1, 1]$.
- **Spearman Rank**: Measures monotonic relationship (non-linear but increasing/decreasing).
- **Caution**: Correlation $\neq$ Causation.

**Key insight:** Pearson's $r = 0$ does not mean no relationship. A variable with a perfect parabolic relationship to the target will have $r = 0$ with Pearson because the symmetric positive and negative halves cancel out. Spearman rank correlation catches monotonic nonlinear relationships that Pearson misses entirely.

**How to verify understanding:** Draw a scatter plot where Pearson $r \approx 0$ but the relationship is clearly predictive. Then explain which metric would detect it.

**What trips people up:** Using a correlation heatmap as the sole relationship-detection tool and concluding that uncorrelated features are useless. Nonlinear feature importance, partial correlations, and interaction terms are invisible to a standard correlation matrix.

### Identifying Outliers

- **Box Plot**: Shows Q1, Median, Q3, and "Whiskers". Any point beyond $1.5 \times IQR$ is a candidate for inspection.
- **Violin Plot**: Combines box plots with a kernel density estimation of the data.

**Key insight:** The box plot's "outlier" threshold of $1.5 \times IQR$ is a convention, not a law. For highly skewed distributions, this threshold will flag many normal points as outliers. For heavy-tailed distributions, it will miss genuine anomalies. Always inspect flagged points domain-first: is this a measurement error or a real edge case?

**How to verify understanding:** In a right-skewed salary distribution, will a box plot's whiskers flag more "outliers" on the left (low) or right (high) side? Why, and what does this imply for automated outlier removal?

**What trips people up:** Removing all flagged boxplot outliers automatically and re-running. The threshold should be understood and justified for each feature — not applied uniformly. In some datasets (e.g., click-through rates), the extreme values are the most important examples.

---

## 3. Missingness Patterns

**Why missingness analysis is part of EDA, not cleaning:** Understanding *why* data is missing determines how you should handle it. EDA is where you diagnose this. Cleaning is where you treat it.

- **MCAR** (Missing Completely at Random): Missingness is unrelated to any variable. Safe to impute or drop.
- **MAR** (Missing at Random): Missingness depends on other *observed* variables. Can be modeled.
- **MNAR** (Missing Not at Random): Missingness depends on the missing value itself. Dangerous — imputation will introduce systematic bias.

**Key insight:** MNAR data cannot be handled by any imputation method without additional assumptions. The act of imputing MNAR values creates a biased signal. The correct move is often to add a binary indicator column (`feature_was_missing`) that lets the model learn the pattern of missingness as a feature in its own right.

**How to verify understanding:** In a medical dataset, patients with the worst outcomes often skip the "Quality of Life" survey. Describe what happens to a model's predictions if you impute the mean for this field.

**What trips people up:** Treating all missing data as MCAR and proceeding to mean imputation. In most real-world datasets, especially those involving human behavior, data is not missing randomly. Always plot missingness rates against other variables to detect patterns.

---

## Interview Questions

**1. "You have a distribution with Mean > Median. What does this tell you?"**
> The distribution is likely **Right-Skewed** (Positive Skew). The extreme values on the right are pulling the mean away from the center.

**2. "How do you detect feature interaction during EDA?"**
> Using a **Scatter Plot Matrix (PairPlot)** or looking for non-random patterns in residuals if a baseline model is run. High interaction often requires domain knowledge to engineer.

**3. "Which plot would you use to show the distribution of a categorical variable vs. a continuous target?"**
> A **Box Plot** or **Violin Plot** is ideal for comparing distributions across different categories.

---

## Python EDA Quick-Start

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Distribution with KDE
sns.histplot(df['feature'], kde=True)

# Pairplot for multivariate relationships
sns.pairplot(df, hue='target_label')

# Missingness pattern
import missingno as msno
msno.matrix(df)  # visualize where values are missing
```
