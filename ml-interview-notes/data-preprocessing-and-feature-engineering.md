# Data Preprocessing and Feature Engineering

---

# Q1: What is Feature Engineering?

## 1. 🔹 Direct Answer
**Feature engineering** is creating **inputs** (transformations, combinations, domain signals) that make learning **easier**—better than raw fields alone. It encodes **prior knowledge** and **structure**.

## 2. 🔹 Intuition
Models learn patterns faster when features **align** with the generative process (e.g., ratios capture scale invariance).

## 3. 🔹 Deep Dive
- Examples: **log1p** for skew, **interaction** terms, **binning**, **time** features (hour, cyclical sin/cos), **aggregates** per user.
- **Leakage**: future aggregates into past rows—fatal.

## 4. 🔹 Practical Perspective
- Deep nets can learn some raw features but **tabular** still benefits from engineering.
- **Document** features for reproducibility in Feature Store.

## 5. 🔹 Code Snippet
```python
df["log_price"] = np.log1p(df["price"])
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** AutoFeat vs manual? **A:** Start domain features; automate search second.

## 7. 🔹 Common Mistakes
Thousands of arbitrary interactions without regularization—overfits.

## 8. 🔹 Comparison / Connections
Representation learning, embeddings, PCA.

## 9. 🔹 One-line Revision
Feature engineering encodes domain structure into columns—watch leakage and complexity.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q2: What is one-hot encoding? When should you use it?

## 1. 🔹 Direct Answer
**One-hot** maps each category to a **binary vector** with one 1—**no ordinal** meaning. Use for **nominal** categories with **low cardinality** going into linear models / NNs expecting numeric input.

## 2. 🔹 Intuition
Each category gets its own switch—distance in one-hot space is uniform across different categories.

## 3. 🔹 Deep Dive
- **Sparse** matrices for many columns.
- **Dummy trap**: drop one column for linear models with intercept to avoid collinearity.

## 4. 🔹 Practical Perspective
- **High cardinality** → blowup (memory, overfit)—use **embeddings**, **target encoding**, **hashing**.

## 5. 🔹 Code Snippet
```python
import pandas as pd
X_oh = pd.get_dummies(df["city"], prefix="city")
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Tree models? **A:** Often label encoding OK; depends on implementation.

## 7. 🔹 Common Mistakes
One-hot **ordinal** grades (loses order)—use ordinal or scores.

## 8. 🔹 Comparison / Connections
Target encoding, embeddings, frequency encoding.

## 9. 🔹 One-line Revision
One-hot for nominal low-cardinality categories; avoid on high-cardinality without compression.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q3: How do you deal with missing data?

## 1. 🔹 Direct Answer
First ask **why** missing (MCAR/MAR/MNAR). Strategies: **drop** if few, **impute** (mean/median/mode, **KNN**, **MICE**), **model** missingness as feature, or use models handling **NaN** (XGBoost) with care.

## 2. 🔹 Intuition
Missingness often **carries signal**—“unknown income” may mean something.

## 3. 🔹 Deep Dive
- **Indicator** column `is_missing` can help MAR.
- **Don’t** leak test statistics into train without CV pipeline.

## 4. 🔹 Practical Perspective
Document imputation; **sensitivity** analysis if MNAR.

## 5. 🔹 Code Snippet
```python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median")
```

## 6. 🔹 Interview Follow-ups
1. **Q:** MNAR? **A:** Imputation biased—model missing mechanism or collect data.

## 7. 🔹 Common Mistakes
Dropping all rows with any NA without checking bias.

## 8. 🔹 Comparison / Connections
Semi-supervised learning, data quality monitoring.

## 9. 🔹 One-line Revision
Understand missingness mechanism; impute with pipelines; add indicators when informative.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q4: How do you handle Outliers?

## 1. 🔹 Direct Answer
**Detect** via z-score, IQR, isolation forest; **decide** if error (fix/remove), legitimate heavy tail (keep + **robust** model), or influential point (**winsorize**, **transform**). Avoid blind deletion.

## 2. 🔹 Intuition
Outliers skew mean-based models; sometimes they **are** the signal (fraud).

## 3. 🔹 Deep Dive
- **Robust scalers**, **Huber** loss, **quantile** transforms.
- **Multivariate** outliers: Mahalanobis (careful with estimation).

## 4. 🔹 Practical Perspective
Plot **per feature** and **joint**; slice by segment—global outlier may be normal in niche.

## 5. 🔹 Code Snippet
```python
q1, q99 = df["x"].quantile([0.01, 0.99])
df["x_clip"] = df["x"].clip(q1, q99)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Tree models? **A:** Less sensitive to monotonic transforms; still check data bugs.

## 7. 🔹 Common Mistakes
Removing all “far” points that represent real rare events.

## 8. 🔹 Comparison / Connections
Anomaly detection, robust statistics.

## 9. 🔹 One-line Revision
Treat outliers as data bugs vs real extremes—use robust losses/transforms or domain rules.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q5: Explain Feature Scaling. Why is it needed?

## 1. 🔹 Direct Answer
**Scaling** maps features to comparable ranges (**standardize** zero mean unit var, **min-max** [0,1], **robust** by median/IQR). Needed for **gradient descent**, **distance** methods (kNN, SVM, NNs), **regularization** fairness across dimensions.

## 2. 🔹 Intuition
If one feature is in millions and another in 0–1, optimizers and distances are dominated by the large scale.

## 3. 🔹 Deep Dive
- **Fit scaler on train only**—transform val/test.
- **Trees**: splitting invariant to monotone transforms—often **not** required.

## 4. 🔹 Practical Perspective
Sparse/interpretability: sometimes keep raw + separate scale per feature group.

## 5. 🔹 Code Snippet
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_val_t = scaler.transform(X_val)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Normalize vs standardize? **A:** Min-max bounded; z-score for unbounded gaussian-like.

## 7. 🔹 Common Mistakes
Fitting on full dataset before split—leakage.

## 8. 🔹 Comparison / Connections
BatchNorm in NNs (different purpose but scale normalization).

## 9. 🔹 One-line Revision
Fit scaling on training data for distance/gradient methods—trees often skip.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q6: One-Hot, Label, Target, and K-Fold Target Encoding

## 1. 🔹 Direct Answer
- **One-hot**: sparse binary columns per category.
- **Label encoding**: integer per category—**implies order** (risky for linear models).
- **Target encoding**: replace category with **mean target**—strong signal, **leakage** risk.
- **K-fold target encoding**: compute mean **out-of-fold** to reduce leakage.

## 2. 🔹 Intuition
Target encoding borrows **supervised** strength from the label; vanilla version **cheats** by using val labels.

## 3. 🔹 Deep Dive
- Add **smoothing** (Bayesian) toward global mean for rare categories.
- **Regularization**: noise, CV schemes.

## 4. 🔹 Practical Perspective
High-cardinality categoricals: **target encoding** + regularization often beats one-hot in linear/boosted models.

## 5. 🔹 Code Snippet
```text
for each fold: mean_y per category on train_fold only -> map val_fold
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Why leakage? **A:** Category mean includes target from same row in global calc.

## 7. 🔹 Common Mistakes
Plain target encode on full train before CV—optimistic scores.

## 8. 🔹 Comparison / Connections
Embeddings, hashing trick.

## 9. 🔹 One-line Revision
Use OOF target encoding or smoothing for high-cardinality cats—never leak validation labels.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q7: How do you handle Categorical Features?

## 1. 🔹 Direct Answer
Pick by **cardinality** and **model**: low → one-hot; medium → **ordinal** if true order; high → **target encoding**, **hashing**, **embeddings** (NN), **frequency** encoding. Always handle **unseen** cats at inference.

## 2. 🔹 Intuition
Categories aren’t numbers—encoding defines **geometry** in feature space.

## 3. 🔹 Deep Dive
- **Embeddings**: learn dense vectors—great for NN + large data.
- **Leakage-free** target stats via nested CV.

## 4. 🔹 Practical Perspective
**Default** or **UNK** bucket for rare; **drop** ultra-rare with care.

## 5. 🔹 Code Snippet
```python
df["cat"] = df["cat"].astype("category")
unk = "UNK"
df["cat"] = df["cat"].cat.add_categories([unk]).fillna(unk)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** String similarity? **A:** Embeddings from language models or hand features.

## 7. 🔹 Common Mistakes
Label encode 1000 categories and feed to linear regression without regularization.

## 8. 🔹 Comparison / Connections
Mixed data types, feature hashing.

## 9. 🔹 One-line Revision
Match encoding to cardinality and model; handle rare/unseen and avoid target leakage.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q8: Explain feature selection vs feature extraction.

## 1. 🔹 Direct Answer
**Selection** picks a **subset** of original features (filter/wrapper/embedded). **Extraction** builds **new** axes (PCA, autoencoders) that combine originals—**dimensionality reduction** / denoising.

## 2. 🔹 Intuition
Selection = “which columns matter?” Extraction = “what new coordinates summarize variance?”

## 3. 🔹 Deep Dive
- **Filter**: correlation, mutual information—fast.
- **Wrapper**: forward/backward—expensive.
- **Embedded**: L1, tree importance.
- **PCA**: unsupervised—may not align with label.

## 4. 🔹 Practical Perspective
Interpretability often favors **selection**; **compression** or **collinearity** → extraction.

## 5. 🔹 Code Snippet
```python
from sklearn.feature_selection import SelectKBest, f_classif
X2 = SelectKBest(f_classif, k=20).fit_transform(X, y)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** PCA before linear model? **A:** Loses interpretability; removes multicollinearity.

## 7. 🔹 Common Mistakes
PCA on full data before CV—leakage.

## 8. 🔹 Comparison / Connections
Autoencoders, mutual information, SHAP.

## 9. 🔹 One-line Revision
Selection keeps original features; extraction creates new components—choose for interpretability vs compression.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q9: How would you create new features from existing ones?

## 1. 🔹 Direct Answer
**Domain**: ratios, differences, logs, **lags**, rolling stats, **counts** per entity, **binning** continuous, **interactions**, **polynomials** (careful). Validate with **ablation** and **stability** across time.

## 2. 🔹 Intuition
Good features **linearize** relationships or expose **invariances** the model would struggle to learn from raw fields.

## 3. 🔹 Deep Dive
- **Leakage**: rolling mean including current row’s label.
- **Regularize** when exploding dimensionality.

## 4. 🔹 Practical Perspective
Start **simple** (3–5 strong features) before **kitchen sink**.

## 5. 🔹 Code Snippet
```python
df["ctr"] = df["clicks"] / (df["impressions"] + 1e-6)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Automated? **A:** Featuretools—watch cross-validation and runtime.

## 7. 🔹 Common Mistakes
Ratios without handling zero denominator.

## 8. 🔹 Comparison / Connections
Feature crosses in deep learning.

## 9. 🔹 One-line Revision
Combine domain ratios, time windows, and aggregates—validate without leakage.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q10: How do you approach a dataset with highly imbalanced classes?

## 1. 🔹 Direct Answer
**Metrics**: PR-AUC, F1, recall at precision—not accuracy. **Resampling**: undersample majority, **SMOTE**, oversample minority; **class weights** in loss; **threshold** tuning; **anomaly** framing if extreme imbalance.

## 2. 🔹 Intuition
Standard training **chases majority**—you must **pay** for minority errors.

## 3. 🔹 Deep Dive
- **SMOTE** risks **noise**—clean data first.
- **Stratified** CV and **calibration** after reweight.

## 4. 🔹 Practical Perspective
**Cost-sensitive** learning matches business FP/FN costs.

## 5. 🔹 Code Snippet
```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** SMOTE on test? **A:** Never—only train.

## 7. 🔹 Common Mistakes
Reporting accuracy on 99% negative class data.

## 8. 🔹 Comparison / Connections
Focal loss, precision-recall curve.

## 9. 🔹 One-line Revision
Change metrics, weights, sampling, and thresholds—never trust accuracy alone on imbalance.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q11: How do you select features for a model?

## 1. 🔹 Direct Answer
Combine **domain** knowledge, **univariate** screens (MI, χ²), **model-based** importance (trees, L1), **stability** across folds/time, and **ablation**—balance **performance** vs **complexity** and **latency**.

## 2. 🔹 Intuition
Fewer, stronger features **generalize** better and simplify debugging.

## 3. 🔹 Deep Dive
- **Wrapper** methods (RFE) costly but accurate.
- **Multicollinearity**: drop redundant to stabilize linear models.

## 4. 🔹 Practical Perspective
**Monitor** feature **drift** in prod—selection is not one-time.

## 5. 🔹 Code Snippet
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=10).fit(X, y)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Correlation filter threshold? **A:** Heuristic—check VIF for linear.

## 7. 🔹 Common Mistakes
Selecting on **test** set performance—double-dips.

## 8. 🔹 Comparison / Connections
SHAP, permutation importance.

## 9. 🔹 One-line Revision
Use domain + statistical screens + model importance + stability—validate with nested CV.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q12: Why and how do you split data into train, test, and validation sets?

## 1. 🔹 Direct Answer
**Train** fits parameters; **validation** tunes hyperparameters / early stopping; **test** estimates **generalization** once—**never** tune on test. **Random** split i.i.d. data; **time/entity** split for leakage-prone domains.

## 2. 🔹 Intuition
If you peek at test during model selection, you **overfit** the benchmark.

## 3. 🔹 Deep Dive
- Ratios e.g. 70/15/15 or CV for small data.
- **Group** split: same user not in train and test.

## 4. 🔹 Practical Perspective
**Stratify** classification; **multiple** seeds for stability.

## 5. 🔹 Code Snippet
```python
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Only train/val? **A:** Use nested CV or final holdout for unbiased estimate.

## 7. 🔹 Common Mistakes
Random split for time-series—future leaks into past.

## 8. 🔹 Comparison / Connections
Cross-validation, bootstrap.

## 9. 🔹 One-line Revision
Train/val/test separates fitting, tuning, and honest evaluation—match split to data dependencies.

## 10. 🔹 Difficulty Tag
🟡 Medium

---
