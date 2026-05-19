# Day 3-4: Data Preprocessing Techniques

## Why This Topic Comes Here

You just learned that ML models learn by minimizing a loss function. Data preprocessing is what ensures the input to that loss function is not corrupted before the model even starts. Preprocessing errors are invisible — the model will train without error, but the results will be wrong or misleading. This topic is placed before any modeling because the single most common source of mistakes in real ML work is not choosing the wrong model; it is contaminating the data pipeline. If you don't internalize data leakage now, you will misdiagnose every model you build for the rest of this plan.

---

## Executive Summary

| Stage | Problem | Techniques | Goal |
|-------|---------|------------|------|
| **Cleaning** | Missing values, Noise | Imputation, Outlier removal | Data Integrity |
| **Scaling** | Feature range mismatch | Standardization, Min-Max | Convergence Speed |
| **Encoding** | Strings/Categories | One-Hot, Target, Label | Numerical compatibility |
| **Selection** | Irrelevant features | PCA, Lasso (L1), RFE | Complexity reduction |

---

## 1. Data Cleaning: The "Garbage In, Garbage Out" Rule

**Why this is the first preprocessing step:** Before asking what a feature means statistically, you must ask whether it is accurate at all. Imputing missing values incorrectly or leaving outliers untreated distorts every downstream computation — means, covariances, learned weights.

### Handling Missing Values

1. **Mean/Median/Mode**: Simple but reduces variance.
2. **KNN Imputation**: Uses nearest neighbors to predict missing values (more accurate but slower).
3. **Iterative Imputation**: Models each feature as a function of the others.

**Key insight:** How you impute missing data is itself a modeling choice that should reflect *why* the data is missing. Imputing the mean when data is MNAR (Missing Not at Random) does not just introduce noise — it actively encodes a false signal. The missing pattern itself is often predictive.

**How to verify understanding:** You have a survey dataset where high-income respondents systematically skip the income field. What happens to your model if you impute median income for these records? Which direction does bias go?

**What trips people up:** Treating imputation as a data cleaning step and fitting the imputer on the whole dataset before splitting. This leaks test distribution information into training. The imputer must be fit on training data only and then applied to test data.

### Outlier Detection

- **Z-Score**: $z = \frac{x - \mu}{\sigma}$. Typically $|z| > 3$ is an outlier.
- **IQR**: Values outside $[Q1 - 1.5 \cdot IQR, Q3 + 1.5 \cdot IQR]$.

**Key insight:** Outliers are not always errors. In fraud detection, the outlier *is the signal*. The decision to remove or cap an outlier depends entirely on whether it represents a data quality problem or a real phenomenon you want the model to learn.

**How to verify understanding:** Name a domain where removing outliers would make your model systematically worse, not better. Explain the mechanism.

**What trips people up:** Applying outlier removal automatically as a cleaning step. Always ask: does this outlier represent noise or a real edge case my model needs to handle?

---

## 2. Feature Scaling

**Why this topic matters more than it appears:** Gradient descent converges faster when features are on the same scale. Distance-based algorithms (KNN, SVM, K-Means) are completely dominated by features with larger numeric ranges, making the other features irrelevant unless scaled. This is not an optimization nicety — without scaling, KNN on a dataset with one feature in [0, 1] and another in [0, 1,000,000] effectively ignores the first feature entirely.

### Standardization (Z-Score)

Centers data around 0 with unit variance.
$$x_{std} = \frac{x - \mu}{\sigma}$$
- **When to use**: Algorithms assuming Gaussian distributions (SVM, Linear Reg, PCA).

### Min-Max Scaling

Rescales to $[0, 1]$.
$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
- **When to use**: Neural Networks, KNN, algorithms that don't make Gaussian assumptions.

**Key insight:** Neither scaler is universally better. Min-Max preserves the shape of the original distribution but is severely distorted by a single extreme outlier (the max becomes that outlier). Standardization is more robust but gives no bounded range, which matters for some network initialization schemes.

**How to verify understanding:** You standardize your training data. A test point arrives with a value 10 standard deviations above the training mean. What should you do, and what does it suggest about the test set?

**What trips people up:** Fitting the scaler on the combined train+test set. This is data leakage. The scaler must be fit on training data only. In production, the training-time mean and standard deviation are saved and applied to every new inference.

---

## 3. Categorical Encoding

**Why encoding connects back to day 1:** Gradient descent and matrix operations require numerical inputs. Encoding is the bridge. But the encoding choice encodes assumptions about relationships between categories — assumptions that may be false.

### One-Hot Encoding

Creates binary columns for each category.
- **Problem**: "Dummy Variable Trap" (perfect multicollinearity). Always drop one column ($n-1$) in linear models.

**Key insight:** One-hot encoding tells the model that categories have no ordinal relationship — "red" is no closer to "blue" than it is to "green." This is correct for truly nominal categories. For high-cardinality features (zip codes, product IDs with millions of values), it creates a dimensionality explosion that kills both memory and generalization.

**How to verify understanding:** You have a `city` column with 500 unique values. You one-hot encode it and include it in a logistic regression. What problem will you likely face, and what alternative would you consider?

**What trips people up:** Using label encoding for nominal (unordered) categories in tree-based models and assuming it doesn't matter. Tree-based models can handle the arbitrary ordering to some extent, but it creates unnatural splits. For linear models, label encoding for nominal variables is almost always wrong.

### Target (Mean) Encoding

Replaces the category with the average target value for that category.
- **Warning**: High risk of **Data Leakage**. Always use K-fold cross-validation during encoding.

**Key insight:** Target encoding is powerful because it compresses high-cardinality categories into a single feature that directly encodes the signal. But without K-fold encoding, a rare category with a single training example gets encoded as exactly its label — perfectly memorizing that example into the feature.

**How to verify understanding:** Why does target encoding a category that appears once in training almost guarantee overfitting? What does K-fold target encoding specifically prevent?

**What trips people up:** Applying target encoding before the train-test split. The target values from test examples must never influence the encoding computed during training.

---

## Interview Questions

**1. "Why should you perform the train-test split before scaling?"**
> To prevent **Data Leakage**. Scaling factors ($\mu, \sigma$) should be calculated ONLY on the training set and then applied to the test set to simulate real-world unseen data.

**2. "Explain the difference between Normalization (Min-Max) and Standardization (Z-score)."**
> Normalization squashes data into a fixed range ($[0,1]$), which is sensitive to outliers. Standardization doesn't have a fixed range and is more robust to outliers as it preserves the distribution shape.

**3. "What happens if you don't scale features for Gradient Descent?"**
> The contour plots of the cost function will be highly elongated (ovals), causing the gradient to oscillate and take much longer to reach the global minimum.

---

## Code Snippet

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Recommended: Use Pipelines to avoid leakage
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
```
