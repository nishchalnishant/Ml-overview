# Day 5-7: Exploratory Data Analysis (EDA)

##  Executive Summary
| Task | Objective | Primary Tools |
|------|-----------|---------------|
| **Univariate** | Distribution Check | Histograms, Boxplots, KDE |
| **Bivariate** | Relationship/Trend | Scatter Plots, Correlation Heatmaps |
| **Multivariate** | Complex Patterns | Pair Plots, Parallel Coordinates |
| **Statistical** | Hypothesis Testing | T-Tests, ANOVA, Chi-Squared |

---

##  1. Statistical Foundations of EDA

### Descriptive Statistics
- **Central Tendency**: Mean (sensitive to outliers), Median (robust), Mode.
- **Dispersion**: Standard Deviation, Variance, Skewness, Kurtosis.
- **Skewness**: Measure of asymmetry. Positive skew = long tail on the right.

### The Power Law (Long Tail)
Many real-world distributions (wealth, city sizes, NLP token frequency) follow a Power Law rather than a Normal distribution.
- **Log-Transform**: Useful for making skewed distributions more Normal.

---

##  2. Visual Inspection Strategies

### Identifying Relationships
- **Pearson Correlation ($r$)**: Measures linear relationship. Range $[-1, 1]$.
- **Spearman Rank**: Measures monotonic relationship (non-linear but increasing/decreasing).
- **Caution**: Correlation $\neq$ Causation.

### Identifying Outliers
- **Box Plot**: Shows Q1, Median, Q3, and "Whiskers". Any point beyond $1.5 \times IQR$ is a candidate for inspection.
- **Violin Plot**: Combines box plots with a kernel density estimation of the data.

---

##  3. Missingness Patterns
Understanding *why* data is missing is key to EDA:
- **MCAR**: Missing Completely at Random.
- **MAR**: Missing at Random (dependent on other observed variables).
- **MNAR**: Missing Not at Random (dependent on the missing value itself).

---

##  Interview Questions

**1. "You have a distribution with Mean > Median. What does this tell you?"**
> The distribution is likely **Right-Skewed** (Positive Skew). The extreme values on the right are pulling the mean away from the center.

**2. "How do you detect feature interaction during EDA?"**
> Using a **Scatter Plot Matrix (PairPlot)** or looking for non-random patterns in residuals if a baseline model is run. High interaction often requires domain knowledge to engineer.

**3. "Which plot would you use to show the distribution of a categorical variable vs. a continuous target?"**
> A **Box Plot** or **Violin Plot** is ideal for comparing distributions across different categories.

---

##  Python EDA Quick-Start
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Distribution with KDE
sns.histplot(df['feature'], kde=True)

# Pairplot for multivariate relationships
sns.pairplot(df, hue='target_label')
```
