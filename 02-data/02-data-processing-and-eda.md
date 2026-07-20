---
module: Foundations
topic: Data Processing and EDA
subtopic: ""
status: unread
tags: [foundations, eda, data-cleaning, missing-values, feature-engineering, data-leakage, pipelines]
---
# Data Processing and EDA

**For:** The workflow between "I have raw data" and "I have a model-ready matrix" — cleaning, exploration, missing values, encoding, and the pipeline discipline that prevents data leakage.
**Use:** Follow the workflow top to bottom on a new dataset; jump to §6 (Data Leakage) whenever a model's validation score looks suspiciously good.

---

## 1. The Workflow

```
Raw data → EDA → Cleaning → Feature Engineering → Encoding/Scaling → Pipeline → Model
              ↑___________________________________________________|
              (EDA is iterative — insights from later steps send you back)
```
Treat this as a loop, not a line. Cleaning decisions (e.g., how to impute a column) should be informed by EDA (is it missing at random?), and feature engineering choices often surface new things worth exploring.

---

## 2. Exploratory Data Analysis (EDA)

The goal of EDA is to build intuition for what's *actually* in the data before you commit modeling assumptions to it — shape, types, distributions, relationships, and anomalies.

### 2.1 First Pass Checklist
```python
df.shape                          # rows, columns
df.info()                         # dtypes, non-null counts — first missing-value signal
df.describe(include="all")        # numeric summary stats + categorical counts/unique
df.head(); df.sample(10)          # eyeball actual rows, not just summaries
df.duplicated().sum()             # exact duplicate rows
df.nunique()                      # cardinality per column — flags ID-like columns, constant columns
```

### 2.2 Univariate Analysis
```python
sns.histplot(df["feature"], kde=True)    # shape, skew, multimodality
sns.boxplot(x=df["feature"])             # outliers, IQR spread
df["category_col"].value_counts(normalize=True)  # class imbalance check
```
**What you're looking for:** skewed distributions (candidates for log-transform), unexpected ranges (age = -5, or 200), class imbalance (1% positive rate needs different metrics and possibly resampling), and constant/near-constant columns (zero information, drop them).

### 2.3 Bivariate / Multivariate Analysis
```python
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")  # linear correlation, first pass
sns.pairplot(df, hue="target")                                         # pairwise relationships + target separability
sns.boxplot(x="category_col", y="target", data=df)                     # categorical feature vs. target
pd.crosstab(df["cat_a"], df["cat_b"], normalize="index")               # relationship between two categoricals
```
**Why it matters:** correlation heatmaps catch multicollinearity (redundant features, a problem for linear models and interpretability, less so for trees) and give an early read on which features might actually predict the target — before spending compute training a model to find out.

### 2.4 Target Analysis
- **Regression target:** check distribution shape (skewed targets often benefit from a `log1p` transform, undone with `expm1` at inference) and for outliers that could dominate an MSE-based loss.
- **Classification target:** check class balance. A 99/1 split makes accuracy meaningless — plan for precision/recall/PR-AUC and possibly resampling or class weighting from the start.

---

## 3. Missing Values

### 3.1 Diagnose the Mechanism First
Before choosing how to handle missing data, understand *why* it's missing — the mechanism determines whether imputation is safe or actively harmful.

| Mechanism | Meaning | Example | Safe to impute? |
|---|---|---|---|
| **MCAR** (Missing Completely At Random) | Missingness independent of any data, observed or not | Sensor randomly drops readings | Yes, low risk |
| **MAR** (Missing At Random) | Missingness depends on *observed* data | Income missing more often for younger respondents (age is observed) | Yes, condition imputation on the related observed variable |
| **MNAR** (Missing Not At Random) | Missingness depends on the *unobserved* value itself | High earners refuse to report income | Dangerous — imputation can bias the model; consider a missingness indicator instead of a value guess |

```python
df.isnull().mean().sort_values(ascending=False)   # % missing per column
import missingno as msno
msno.matrix(df)     # visualize missingness pattern — look for correlated missingness across columns
```

### 3.2 Strategies
```python
# Drop: only when missingness is small and MCAR
df.dropna(subset=["critical_col"])

# Simple imputation
df["num_col"].fillna(df["num_col"].median(), inplace=False)   # median: robust to outliers
df["cat_col"].fillna(df["cat_col"].mode()[0], inplace=False)  # mode for categoricals

# Add a missingness indicator — often more informative than the imputed value itself
df["num_col_was_missing"] = df["num_col"].isnull().astype(int)

# Model-based imputation (fit on train, apply to val/test — see leakage section)
from sklearn.impute import KNNImputer, IterativeImputer
imputer = KNNImputer(n_neighbors=5)
```
**Why it matters:** mean/median imputation shrinks variance and can distort relationships with other features. A missingness indicator column preserves the signal ("this value was missing") that a single imputed number destroys — often the missingness itself is predictive (e.g., a skipped form field correlates with the outcome).

---

## 4. Outliers

```python
# IQR method
q1, q3 = df["x"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df[(df["x"] < lower) | (df["x"] > upper)]

# Z-score method (assumes roughly normal distribution)
z = (df["x"] - df["x"].mean()) / df["x"].std()
outliers = df[z.abs() > 3]
```
**Decision, not just detection:** an outlier isn't automatically wrong. Distinguish data errors (age = 200 → fix or drop) from genuine extreme values (a $10M transaction in a fraud dataset is exactly the signal you want) — dropping the latter can remove the most important rows in the dataset.

---

## 5. Feature Engineering

### 5.1 Scaling
| Method | Formula | When |
|---|---|---|
| **Standardization (Z-score)** | $(x - \mu) / \sigma$ | Distance-based models (KNN, SVM), gradient-based models (neural nets, linear/logistic regression) — centers data so gradients behave well |
| **Min-Max Normalization** | $(x - min) / (max - min)$ | When bounds are known and meaningful (pixel values [0,255] → [0,1]) |
| **Robust Scaling** | $(x - median) / IQR$ | Data with significant outliers — median/IQR aren't dragged around by extreme values the way mean/std are |
| **None needed** | — | Tree-based models (splits are invariant to monotonic transforms of a single feature) |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on TRAIN only
X_val_scaled = scaler.transform(X_val)           # transform only — reuse train's mean/std
```

### 5.2 Encoding Categoricals
| Method | How | When |
|---|---|---|
| **One-Hot** | New binary column per category | Low cardinality, nominal (no order), linear models |
| **Ordinal** | Integer per category, preserving order | Genuinely ordinal categories (`low`/`med`/`high`) |
| **Target Encoding** | Replace category with mean target value for that category | High cardinality; **requires** out-of-fold computation to avoid leakage (see §6) |
| **Frequency Encoding** | Replace category with its frequency count | High cardinality, no leakage risk, weaker signal than target encoding |
| **Embeddings** | Learned dense vector per category | Very high cardinality (user IDs, words) in a neural network |

```python
pd.get_dummies(df, columns=["category_col"], drop_first=True)  # drop_first avoids the dummy variable trap (perfect multicollinearity) for linear models
```

### 5.3 Transformations
```python
df["x_log"] = np.log1p(df["x"])          # right-skewed data (log1p handles x=0 safely, unlike log)
df["x_sq"] = df["x"] ** 2                # polynomial features for linear models to capture curvature
df["date"].dt.dayofweek, df["date"].dt.hour, df["date"].dt.is_month_end  # datetime decomposition
```

---

## 6. Data Leakage

**Leakage** is information from outside the training set (often, indirectly, from the target itself) leaking into the features — producing validation scores that look great and production performance that doesn't match.

### 6.1 The Canonical Failure Pattern
```python
# WRONG: scaler sees the full dataset, including validation, before the split
scaler = StandardScaler().fit(X)             # leakage: val statistics influence train transform
X_scaled = scaler.transform(X)
X_train, X_val = train_test_split(X_scaled)

# RIGHT: fit only on train, apply the same transform to val
X_train, X_val = train_test_split(X)
scaler = StandardScaler().fit(X_train)       # fit on train only
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)       # val never influences the fit
```

### 6.2 Common Leakage Sources
| Source | Example | Fix |
|---|---|---|
| **Preprocessing before split** | Fitting scaler/imputer/target-encoder on the full dataset | Fit only on train fold, `transform` (not `fit_transform`) on val/test |
| **Target leakage** | A feature that's a proxy for the target, or only known *after* the outcome (e.g. "number of follow-up calls" predicting churn) | Audit every feature: "would I know this value at prediction time, before the outcome exists?" |
| **Temporal leakage** | Random train/val split on time-series data lets the model "see the future" | Time-based split — train on past, validate on future, never shuffle across time |
| **Group leakage** | Same user/entity appears in both train and val (e.g. multiple rows per patient) | Group-based split (`GroupKFold`) keyed on the entity ID |
| **Target encoding leakage** | Encoding a category with its own target mean, computed using rows that include the row being encoded | Compute encoding out-of-fold (K-fold target encoding), or add noise/smoothing |
| **Duplicate rows across split** | Near-duplicate or exact-duplicate rows land in both train and test | Deduplicate before splitting, or split by an entity key that guarantees no overlap |

### 6.3 The Rule of Thumb
> **Fit nothing on data the model won't see in production. Split first, transform second, always with `fit` only on train.**

If a validation metric looks *too* good relative to what production later shows, leakage is the first hypothesis to rule out — before assuming the model architecture is simply excellent.

---

## 7. Validation Strategy

| Strategy | When |
|---|---|
| **Random K-Fold** | i.i.d. tabular data, no time or group structure |
| **Stratified K-Fold** | Classification with class imbalance — preserves class ratio in every fold |
| **Time Series Split** (`TimeSeriesSplit`) | Any temporal data — train on past folds, validate on the next chunk forward, never backward |
| **Group K-Fold** | Multiple rows per entity (patient, user, device) — keeps all of an entity's rows in one fold |
| **Nested CV** | Hyperparameter tuning + unbiased performance estimate simultaneously — outer loop for evaluation, inner loop for tuning |

```python
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GroupKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    ...
```

---

## 8. Pipelines

A `Pipeline` chains preprocessing and modeling steps into a single object that guarantees the same transformations are applied identically at fit time and inference time — the structural fix for most leakage bugs.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

numeric_features = ["age", "income"]
categorical_features = ["city", "device"]

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)        # preprocessing fit only on train, inside the pipeline
model.predict(X_val)               # same fitted transforms applied to val automatically
```
**Why it matters:**
- **Leakage-proof by construction** — cross-validating a `Pipeline` with `cross_val_score` refits preprocessing on each fold's train split automatically; you cannot accidentally fit the scaler on validation data.
- **Serialization** — `joblib.dump(model, "model.pkl")` saves preprocessing and model together; production inference code doesn't need to reimplement feature engineering by hand (a common source of train/serve skew).
- **`handle_unknown="ignore"`** on `OneHotEncoder` prevents a crash in production when a category unseen during training shows up.

---

## Interview Questions

1. **Walk through how you'd detect and fix data leakage in a pipeline that has a suspiciously high validation AUC.**
   Check preprocessing order (was anything fit before the split?), audit features for target-proxy signals or post-outcome timing, check for temporal or group overlap between train/val, and verify target encoding was done out-of-fold. See §6.

2. **When is mean/median imputation actively harmful, and what's a better alternative?**
   Harmful under MNAR (§3.1) — imputing masks a signal that was itself informative. Add a missingness indicator column alongside (or instead of) the imputed value so the model can still use the fact that data was missing.

3. **Why must a `StandardScaler` be fit only on the training set?**
   Fitting on the full dataset lets validation/test statistics (mean, std) leak into the train-time transform, producing an overly optimistic validation score that won't hold in production, where future data is genuinely unseen at fit time.

4. **Why do tree-based models not need feature scaling, but linear models and neural networks do?**
   Trees split on a single feature's ordering/threshold at a time — monotonic transforms don't change split points. Linear models' coefficients and neural network gradient magnitudes are scale-sensitive; unscaled features with large ranges dominate the loss/gradient.

5. **What's wrong with a random K-Fold split on time-series data?**
   It lets the model train on future data to predict the past, an information leak that never occurs in production (you can never train on data that doesn't exist yet). Use `TimeSeriesSplit` or an explicit chronological split instead.

6. **How would you target-encode a high-cardinality categorical feature without leakage?**
   Compute the encoding out-of-fold: for each row, compute the target mean for its category using only rows *outside* its own fold (K-fold target encoding), and add smoothing toward the global mean for rare categories to avoid overfitting on small counts.

---

## Where to Next

- **Python/Pandas mechanics underlying this workflow** → [03-python-and-data-tooling.md](01-python-and-data-tooling.md)
- **Classical algorithms this data feeds** → [02-classical-ml/](../02-classical-ml/)
- **Applied end-to-end walkthroughs** → [12-projects/](../12-projects/)
