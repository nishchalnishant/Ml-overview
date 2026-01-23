# Data preprocessing

> \[!IMPORTANT] **Executive Summary for ML Engineers**
>
> 1. **Data Leakage:** The #1 killer of production models. Never fit transformers on test data.
> 2. **Imputation:** Use **Median** for skewed data (robust). For categorical, consider if "Missingness" is itself a signal.
> 3. **Scaling:** **StandardScaler** is default. Use **RobustScaler** if you have extreme outliers.
> 4. **Encoding:** **One-Hot** for low cardinality; **Target Encoding** or **Embeddings** for high cardinality.
> 5. **Drift Monitoring:** Preprocessing isn't a one-time thing. Monitor distribution shift (K-S test) in production.

***

## <mark style="color:$danger;">1. The Cardinal Rule: Prevention of Data Leakage</mark>

Before any transformation, you **must perform the train-test split.**

**Why?**

* **Data Leakage** occurs when information from outside the training dataset is used to create the model.
* **Example:** If you standardize using the _global_ mean, your training data now "knows" something about the distribution of the test set.

**The Correct Workflow:**

1. Split into `X_train` and `X_test`.
2. `fit()` the scaler/encoder ONLY on `X_train`.
3. `transform()` both `X_train` and `X_test` using the fitted parameters.

***

## <mark style="color:red;">2. Handling Missing Data</mark>

### Strategies Comparison

| **Method**     | **Technique**         | **When to Use**                    | **Interview Insight**                                                                 |
| -------------- | --------------------- | ---------------------------------- | ------------------------------------------------------------------------------------- |
| **Deletion**   | Drop rows/cols        | Missing >50% or insignificant rows | Use sparingly; can introduce bias if data is not MCAR (Missing Completely At Random). |
| **Imputation** | Mean/Median/Mode      | Numerical/Categorical features     | Median is preferred for skewed data to avoid outlier influence.                       |
| **Prediction** | KNN/Iterative Imputer | Complex dependencies               | More accurate but computationally expensive and risk of over-fitting.                 |
| **Constant**   | Fill with "Unknown"   | Categorical features               | Preserves the fact that the data was missing, which can be a valuable signal.         |

**Python Code (Sklearn SimpleImputer):**

```python
from sklearn.impute import SimpleImputer

# Numerical: Use median to be robust to outliers
num_imputer = SimpleImputer(strategy='median')
X_train_num = num_imputer.fit_transform(X_train[['age', 'income']])

# Categorical: Use most_frequent (mode)
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_cat = cat_imputer.fit_transform(X_train[['city']])
```

***

## <mark style="color:red;">3. Numerical Data Transformations</mark>

### Scaling Techniques

| **Technique**       | **Formula**             | **When to Use**                  | **Impact**                                                    |
| ------------------- | ----------------------- | -------------------------------- | ------------------------------------------------------------- |
| **Standardization** | (x - μ) / σ             | Most models (SVM, Logistic, PCA) | Centers at 0, unit variance. Sensitive to extreme outliers.   |
| **Normalization**   | (x - min) / (max - min) | Neural Networks, Image pixels    | Bounds data between \[0, 1]. Extremely sensitive to outliers. |
| **Robust Scaling**  | (x - Q2) / (Q3 - Q1)    | Data with many outliers          | Scaled based on Interquartile Range (IQR); ignores extremes.  |

### <mark style="color:red;">Handling Outliers (The IQR Method)</mark>

**Formula:**

* Lower Bound = Q1 - 1.5 \* IQR
* Upper Bound = Q3 + 1.5 \* IQR _Where IQR = Q3 - Q1_

**Strategies:**

1. **Trimming:** Remove values outside bounds.
2. **Capping (Winsorization):** Replace outliers with the upper/lower bound values.

**Code Example (Scaling):**

```python
from sklearn.preprocessing import StandardScaler, RobustScaler

# Default choice
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_num)

# Better for outlier-heavy data
robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train_num)
```

***

\| **Hashing** | High cardinality | Fixed memory, no leakage | Collision risk, non-reversible. |

**Advanced Techniques:**

* **Target Encoding:** Replaces category with mean of target. _Danger: Overfitting._ Use smoothing or CV-folds.
* **Feature Hashing (The Hashing Trick):** Converts categories to indices using a hash function. Used in high-speed online learning (e.g., Vowpal Wabbit).

***

## <mark style="color:$danger;">5. Feature Engineering & Selection</mark>

**The process of using domain knowledge to create new features.**

* **Polynomial Features:** Creating interactions (e.g., $x\_1 \times x\_2$, $x\_1^2$) to capture non-linearities.
* **Binning:** Converting numerical features to categorical (e.g., Age 0-18 → "Child").
* **Geometric/Temporal:** Distance to landmarks, "Is Weekend?", "Time since last purchase".

**Feature Selection:**

* **Filter Methods:** Correlation, Chi-Square, Mutual Information.
* **Wrapper Methods:** Recursive Feature Elimination (RFE).
* **Embedded Methods:** L1 Regularization (Lasso) - coefficients shrink to zero.

***

## <mark style="color:$danger;">6. Preprocessing in Production: Data Drift</mark>

In real-world systems, data distributions change over time (Covariate Shift).

**Detection Strategies:**

* **Population Stability Index (PSI):** Measures shift in distribution between two time periods.
* **Kolmogorov-Smirnov (K-S) Test:** Non-parametric test for equality of 1D distributions.
* **Monitoring Tooling:** Use Prometheus/Grafana or specialized ML tools (WhyLabs, Arize).

***

## <mark style="color:$danger;">6. Preprocessing for Image & Text</mark>

### Image Data (Computer Vision)

* **Mandatory:** Resizing (all images must be same shape, e.g., 224x224).
* **Mandatory:** Scaling (Divide by 255 for \[0, 1] or use ImageNet normalization).
* **Optional:** Augmentation (Flips, rotations) - only during training.

### Text Data (NLP)

* **Cleaning:** Lowercasing, removing punctuation/special chars.
* **Normalization:** Stemming (crude) vs Lemmatization (smart/dictionary-based).
* **Vectorization:**
  * **TF-IDF:** Down-weights common words ("the", "is").
  * **Embeddings (Word2Vec):** Learns semantic meaning.

***

## <mark style="color:$danger;">Common Interview Questions</mark>

**1. "When would you choose Normalization (Min-Max) over Standardization?"**

> I choose Normalization when the distribution is not Gaussian or when the algorithm requires inputs in a fixed \[0, 1] range, such as in Neural Networks or Image Processing. I use Standardization for most other cases, especially when the algorithm assumes Gaussian data (e.g., PCA, Logistic Regression).

**2. "How do you handle categorical features with 10,000+ unique values?"**

> One-Hot encoding would create 10,000 columns, causing the "Curse of Dimensionality." Instead, I would use **Target Encoding**, **Count Encoding**, or **Entity Embeddings** (learned vectors, common in Deep Learning) to represent the classes in a lower-dimensional space.

**3. "What happens if you scale the entire dataset before splitting?"**

> This leads to **Data Leakage.** The training set's statistics would be influenced by the test set's values. The model would yield overly optimistic performance metrics that won't hold up in production.

**4. "Is it necessary to scale features for Decision Trees?"**

> **No.** Tree-based models (Random Forest, XGBoost) are scale-invariant. They split based on raw values and don't rely on distance metrics or gradient updates across features simultaneously.

***

## Python Implementation: A Full Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoding

# Define features
numeric_features = ['age', 'salary']
categorical_features = ['city', 'education']

# 1. Processing for numerical
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 2. Processing for categorical
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoding(handle_unknown='ignore'))
])

# 3. Combine into preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Usage: 
# X_train_preprocessed = preprocessor.fit_transform(X_train)
# X_test_preprocessed = preprocessor.transform(X_test)
```
