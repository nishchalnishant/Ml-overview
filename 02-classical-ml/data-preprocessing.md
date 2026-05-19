# Data Preprocessing and Feature Engineering

---

## 1. Data Leakage

**The problem**: You build a model that scores 0.97 AUC in development and 0.61 AUC in production. Nothing changed except the data. The offline score was a lie — information the model couldn't have at prediction time was accidentally available during training. This is leakage.

**The core insight**: The model must only see information that would be available at the moment it makes a real prediction. Any statistic computed using future observations, test data, or the target variable — and then used to train — gives the model an unfair advantage that disappears in production.

**The mechanics**: Leakage enters through four doors:
- **Target leakage**: A feature encodes the answer. Example: `loan_approved = 1` as a feature when predicting `default`. The feature was created after the outcome was known.
- **Train/test contamination**: Fitting a scaler or encoder on the full dataset before splitting. The scaler now contains test-set statistics; test data is no longer unseen.
- **Temporal leakage**: Using a feature computed from events that happened *after* the prediction timestamp. Example: rolling 7-day average of future sales.
- **ID/proxy leakage**: An ID column or timestamp that correlates with the outcome cohort. Example: customer IDs assigned sequentially — high IDs are newer customers with different behavior.

The fix is a single discipline: **split first, then fit all preprocessing on train only, then transform train and test separately**.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Pipeline re-fits preprocessing on each training fold, transforms val fold
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
```

**What breaks**: The most dangerous leakage is invisible — models that memorize synthetic patterns from contaminated preprocessing. A 30% AUC gap between cross-validation and a holdout set is the classic symptom. If your CV score looks suspiciously good, assume leakage until proven otherwise.

---

## 2. Missing Data

**The problem**: Your model receives a row where three columns are blank. You can drop the row, fill with a number, or pretend it didn't happen. Every choice is a modeling decision with consequences, and the wrong choice introduces bias that is invisible in aggregate metrics.

**The core insight**: Why data is missing tells you how dangerous the gap is.
- **MCAR (Missing Completely At Random)**: missingness is independent of any variable. The survey system randomly dropped 5% of rows. Dropping is safe.
- **MAR (Missing At Random)**: missingness depends on *other observed* variables, not the missing value itself. Older patients skip the income field. Conditional imputation is appropriate.
- **MNAR (Missing Not At Random)**: missingness depends on the value that is missing. High-income people skip the income field. Both deletion and imputation introduce bias — missingness is itself a signal.

**The mechanics**:

| Strategy | How | When | Risk |
|---|---|---|---|
| Row deletion | Drop rows with missing values | MCAR + low missingness | Bias if MAR/MNAR |
| Column deletion | Drop feature if > 50% missing | Any type | Lose predictive signal |
| Mean/median imputation | Replace with training set statistic | Numeric, MAR, low missingness | Compresses variance; hides uncertainty |
| Most-frequent imputation | Replace with mode | Categorical, MAR | Amplifies dominant category |
| KNN imputation | Use similar rows to fill in | Features are correlated | O(n²) distance computation |
| Iterative (MICE) | Cycle regression imputation across features | Complex multivariate dependencies | Slow; can propagate errors |
| Missingness indicator | Add binary `is_missing` column alongside imputation | MNAR suspected | Doubles that feature's columns |

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute import MissingIndicator

num_imputer = SimpleImputer(strategy='median')   # median is robust to skew
cat_imputer = SimpleImputer(strategy='most_frequent')
knn_imputer = KNNImputer(n_neighbors=5)          # for correlated numeric features
```

**What breaks**: Mean imputation shrinks the variance of the imputed column — correlation estimates between imputed and other columns are attenuated. If you impute 30% of a column with its mean, 30% of rows become identical on that feature. KNN imputation on a large dataset is infeasible without approximate nearest-neighbor structures. MICE is theoretically sound but multiplies training time by many cycles.

---

## 3. Numerical Scaling

**The problem**: Gradient descent steps the same distance in every parameter direction. If `income` ranges 0–1,000,000 and `age` ranges 18–90, the gradient in the income direction is ~11,000x larger in raw magnitude. The optimizer overshoots in the income direction and barely moves in the age direction. The model never converges well, or converges to a suboptimal solution.

**The core insight**: Bring every feature onto the same numeric scale so the optimizer treats them symmetrically. The specific scale is arbitrary — what matters is that no single feature dominates gradient magnitudes.

**The mechanics**:

**Z-score standardization**: Subtract the mean, divide by standard deviation. Centers the distribution at zero, rescales spread to 1.

$$z = \frac{x - \mu}{\sigma}$$

Compute μ and σ on the training set only. Apply those same training statistics to test data — this is what prevents leakage.

**Min-Max normalization**: Maps to [0, 1] by shifting and scaling using the observed min and max.

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

**Robust scaling**: Uses median and interquartile range instead of mean and std. More resistant to outliers.

$$x' = \frac{x - \text{median}}{\text{IQR}}$$

**Log transform**: Compresses right-skewed distributions (revenue, counts). Use `log(x + 1)` to handle zeros.

**Power transform (Yeo-Johnson)**: Generalizes log transform — handles negative values, finds the best power to make the distribution approximately Gaussian.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import numpy as np

X['revenue_log'] = np.log1p(X['revenue'])

pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X[numeric_cols])
```

**What breaks**: Outliers inflate σ — a single value of 1,000,000 in a column that otherwise spans 0–100 pulls σ to hundreds, compressing all non-outlier values near zero. Use `RobustScaler` instead. Tree-based models (random forest, XGBoost) are entirely indifferent to feature scale — scaling adds no value and wastes a pipeline step. Log-transformed distributions with heavy tails still look weird after standardization — skewness is reduced but not eliminated.

---

## 4. Categorical Encoding

**The problem**: Models operate on numbers. A column with values `["New York", "London", "Tokyo"]` is meaningless to a gradient descent algorithm or a tree split. You need a numeric representation that preserves the relationships the model should learn — without accidentally implying ordinal or magnitude relationships that don't exist.

**The core insight**: The right encoding depends on the relationship you want the model to see. Ordinal encoding implies order; one-hot implies no order; target encoding implies a predictive relationship. Each choice bakes in a prior.

**The mechanics**:

| Cardinality | Strategy | Why |
|---|---|---|
| Low (< 15 unique values) | One-hot encoding | No spurious ordinal relationship; each category independent |
| Ordinal (low/medium/high) | Ordinal encoding | Explicit order exists; numbers carry that information |
| High (15–1000) | Target encoding with smoothing | OHE would create hundreds of sparse columns |
| Very high (> 1000) | Feature hashing | Fixed-size output; tolerates cardinality at cost of hash collisions |

**One-hot encoding**: Creates one binary column per category. The dropped category becomes the implicit reference.
```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
```

**Ordinal encoding**: Assigns integers 0, 1, 2, … according to an explicit category order you define.
```python
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
```

**Target encoding with Laplace smoothing**: Replaces each category with the mean target value in that category, blended toward the global mean for rare categories.

$$\hat{\mu}_k = \frac{n_k \bar{y}_k + m \bar{y}}{n_k + m}$$

where $n_k$ = number of samples with category $k$, $\bar{y}_k$ = mean target in that category, $\bar{y}$ = global mean, $m$ = smoothing strength. Large $m$ pulls rare categories toward the global mean.

```python
import category_encoders as ce

encoder = ce.TargetEncoder(smoothing=10.0)
X_train['city_enc'] = encoder.fit_transform(X_train['city'], y_train)
X_test['city_enc']  = encoder.transform(X_test['city'])
```

**What breaks**: Target encoding without smoothing memorizes the training set — a category seen 3 times with all positives gets encoded as 1.0, which is purely noise for rare categories. Target encoding must be fit inside each CV fold or it leaks the target into features. One-hot encoding on a 5000-category column creates 5000 features, most of which are nearly empty — use hashing or target encoding instead.

---

## 5. Full Preprocessing Pipeline

**The problem**: Preprocessing has many steps. If you apply them ad-hoc (scale, then encode, then impute, in different code blocks), cross-validation becomes a nightmare — you have to manually track which step has been fit on which data, and you will leak something eventually.

**The core insight**: Wrap all preprocessing in a single `Pipeline` object. The pipeline knows which steps to `fit` on training data and which to `transform` only — it enforces the correct boundary automatically in every CV fold.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

numeric_features     = ['age', 'income', 'tenure']
categorical_features = ['city', 'product_type']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer,     numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   GradientBoostingClassifier(n_estimators=200))
])

model.fit(X_train, y_train)
```

**What breaks**: Forgetting to use `ColumnTransformer` and applying different transformers to different columns outside a pipeline. After `fit_transform`, column names are lost — keep track of feature names explicitly if you need them for SHAP or feature importance later.

---

## 6. Feature Engineering

**The problem**: Raw data encodes information in a form that is difficult for linear models and even trees to exploit. If the relationship between a feature and the target is multiplicative, logarithmic, or involves the ratio of two columns, a model trained on raw columns must discover that relationship implicitly from data — a slow, data-hungry process.

**The core insight**: Making the relationship between features and the target explicit — before the model sees the data — reduces the complexity the model must learn. A domain expert who creates `debt_to_income = debt / income` gives the model a signal it might otherwise need thousands of examples to approximate.

**The mechanics**:

**Date/time features**: Cyclical patterns (hour of day, day of week) are invisible to models without encoding.
```python
df['hour']          = df['timestamp'].dt.hour
df['day_of_week']   = df['timestamp'].dt.dayofweek
df['is_weekend']    = df['day_of_week'].isin([5, 6]).astype(int)
df['days_since_event'] = (df['timestamp'] - df['event_date']).dt.days
```

**Interaction and ratio features**: Explicitly encodes relationships that would require many splits or polynomial terms to approximate.
```python
df['income_per_age']      = df['income'] / (df['age'] + 1)
df['tenure_x_products']   = df['tenure'] * df['product_count']
```

**Polynomial features**: For linear models only — adds squared terms and cross-products so the linear model can capture curvature and interactions.
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)
```

**What breaks**: Polynomial features with degree 2 on 50 raw features produce 1325 columns. Most of them are noise — follow with feature selection. Ratio features explode when the denominator is near zero — add a small constant or use log ratios. Feature engineering bakes in domain assumptions that may not hold across subpopulations.

---

## 7. Data Drift

**The problem**: You deploy a model and it degrades over months without any code change. The world changed. The distribution of inputs the model sees in production diverged from the distribution it was trained on. Without monitoring, you discover this only when business metrics fall — too late.

**The core insight**: A model assumes the future looks like the past. When inputs shift, predictions based on the old model become wrong. You need a quantitative way to detect this shift before predictions go bad.

**The mechanics**:

Three types of drift:
- **Covariate shift**: $P(X)$ changes — the input distribution shifts. The relationship $P(y|X)$ may be stable.
- **Label shift**: $P(y)$ changes — the prior probability of each class changes.
- **Concept drift**: $P(y|X)$ changes — the underlying relationship the model learned no longer holds.

**Population Stability Index (PSI)**: Compares binned distributions between a reference (training) window and a production window.

$$\text{PSI} = \sum_{i} (A_i - E_i) \cdot \ln\left(\frac{A_i}{E_i}\right)$$

where $A_i$ = actual fraction in bin $i$ (production), $E_i$ = expected fraction in bin $i$ (training).

- PSI < 0.1: stable
- PSI 0.1–0.25: moderate shift, investigate
- PSI > 0.25: major shift, retrain

**Kolmogorov-Smirnov test**: Compares the empirical CDFs of two samples. The KS statistic is the maximum difference between the CDFs; the p-value tests whether they come from the same distribution.

```python
from scipy.stats import ks_2samp

for col in numeric_features:
    stat, p_val = ks_2samp(X_train[col], X_production[col])
    if p_val < 0.05:
        print(f"Drift detected: {col}  KS={stat:.3f}  p={p_val:.4f}")
```

**What breaks**: Drift detection on individual features misses multivariate drift — two features can each look stable while their joint distribution has shifted. High-dimensional drift detection requires either dimensionality reduction or multivariate tests (MMD). PSI thresholds are rules of thumb, not guarantees — a feature can show PSI > 0.25 due to seasonal variation that does not impair model performance.
