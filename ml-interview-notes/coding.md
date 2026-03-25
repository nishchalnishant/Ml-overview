# Coding

---

# Q1: Write a Python function to compute the mean squared error (MSE).

## 1. 🔹 Direct Answer
**MSE** = average of squared errors **(ŷ − y)²** over **n** samples. Differentiable; penalizes large errors heavily.

## 2. 🔹 Intuition
Square magnifies big mistakes—good for regression when outliers are real; sensitive to outliers otherwise.

## 3. 🔹 Deep Dive
\(\text{MSE} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2\). Gradient w.r.t. prediction: **2(ŷ − y)/n** (for mean).

## 4. 🔹 Practical Perspective
- Use: default regression loss when Gaussian noise assumption is OK.
- Not: heavy-tailed noise—consider **Huber** or **MAE**.

## 5. 🔹 Code Snippet
```python
import numpy as np
def mse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Vectorized? **A:** Yes—avoid Python loops on large arrays.
2. **Q:** sklearn? **A:** `mean_squared_error`.

## 7. 🔹 Common Mistakes
Forgetting to align shapes; using sum instead of mean without stating it.

## 8. 🔹 Comparison / Connections
RMSE (sqrt), MAE, Huber loss.

## 9. 🔹 One-line Revision
MSE is mean squared residual—smooth, outlier-sensitive regression loss.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q2: Write a Python function to compute the mean absolute error (MAE).

## 1. 🔹 Direct Answer
**MAE** = **mean |y − ŷ|**. Linear penalty; **robust** to outliers vs MSE.

## 2. 🔹 Intuition
Every error unit counts the same—no squaring blow-up.

## 3. 🔹 Deep Dive
Subgradient at zero; not differentiable at 0 in theory—fine in practice.

## 4. 🔹 Practical Perspective
Interpretable in **same units** as target (e.g., dollars). Median regression ties to MAE.

## 5. 🔹 Code Snippet
```python
import numpy as np
def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))
```

## 6. 🔹 Interview Follow-ups
1. **Q:** MSE vs MAE? **A:** MSE emphasizes large errors; MAE robust.

## 7. 🔹 Common Mistakes
Confusing with median absolute deviation (MAD).

## 8. 🔹 Comparison / Connections
Huber, quantile loss.

## 9. 🔹 One-line Revision
MAE is L1 average error—same units as y, less outlier-sensitive than MSE.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q3: Implement a simple linear regression model from scratch.

## 1. 🔹 Direct Answer
**y ≈ Xw + b**. Fit **w, b** by minimizing MSE—closed form **w = (XᵀX)⁻¹Xᵀy** (with bias via column of ones) or **gradient descent**.

## 2. 🔹 Intuition
Best-fit hyperplane; assumes approximate linearity + Gaussian noise (for MLE interpretation).

## 3. 🔹 Deep Dive
- **Normal equation** O(d³)—fine for small **d**.
- **GD** for large/sparse: learning rate, convergence checks.

## 4. 🔹 Practical Perspective
Always **standardize** features for numerical stability; **regularize** (ridge) if collinear.

## 5. 🔹 Code Snippet
```python
import numpy as np
def fit_linear_regression(X, y):
    X = np.c_[np.ones(len(X)), X]
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return w  # [b, w1, w2, ...]
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Ridge? **A:** `w = (XᵀX + λI)⁻¹Xᵀy`.
2. **Q:** Multicollinearity? **A:** Ill-conditioned XᵀX—use ridge or SVD.

## 7. 🔹 Common Mistakes
Forgetting bias term; inverting singular XᵀX without regularization.

## 8. 🔹 Comparison / Connections
Logistic regression, GLMs, SGD.

## 9. 🔹 One-line Revision
Linear regression: least squares on affine model—normal equation or GD; add ridge if unstable.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q4: Implement a simple logistic regression model from scratch.

## 1. 🔹 Direct Answer
**P(y=1|x) = σ(wᵀx + b)** with **σ(z) = 1/(1+e^(−z))**. Train by minimizing **binary cross-entropy** (log loss) via **gradient descent** or **IRLS**.

## 2. 🔹 Intuition
Maps linear score to **probability**; decision boundary is linear in **x**.

## 3. 🔹 Deep Dive
- Loss: **−[y log p + (1−y) log(1−p)]**.
- Gradient: **(p − y) x** (for one sample).

## 4. 🔹 Practical Perspective
- **Class imbalance**: class weights or focal loss variants.
- **Calibration**: often good out-of-the-box; verify with reliability diagram.

## 5. 🔹 Code Snippet
```python
import numpy as np
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def bce_grad(X, y, w, b):
    z = X @ w + b; p = sigmoid(z)
    err = (p - y) / len(y)
    return X.T @ err, err.sum()
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Multiclass? **A:** Softmax + cross-entropy (multinomial logistic).

## 7. 🔹 Common Mistakes
Using MSE for classification probabilities.

## 8. 🔹 Comparison / Connections
Linear SVM, naive Bayes, neural nets without hidden layers.

## 9. 🔹 One-line Revision
Logistic regression = linear scorer + sigmoid + cross-entropy optimization.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q5: Implement K-Nearest Neighbors (KNN).

## 1. 🔹 Direct Answer
**Predict** by majority (classification) or average (regression) of **k** nearest training points in feature space—**non-parametric**, **lazy** learning.

## 2. 🔹 Intuition
“People who look like you in feature space behave similarly.”

## 3. 🔹 Deep Dive
- Distance: **L2**, **L1**, **cosine** for high-dim text.
- **Complexity**: query O(nd) naive; **KD-tree/ball tree** for low dim.

## 4. 🔹 Practical Perspective
- **Scale features**; **choose k** by CV.
- Curse of dimensionality hurts—**PCA** or **metric learning** sometimes.

## 5. 🔹 Code Snippet
```python
import numpy as np
def knn_predict(X_train, y_train, x, k=5):
    d = np.linalg.norm(X_train - x, axis=1)
    idx = np.argpartition(d, k)[:k]
    return np.bincount(y_train[idx].astype(int)).argmax()
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Weighted? **A:** Weight votes by inverse distance.
2. **Q:** Imbalanced classes? **A:** Weighted voting or stratified k.

## 7. 🔹 Common Mistakes
Using raw features with different scales.

## 8. 🔹 Comparison / Connections
Parzen windows, kernel density, decision trees.

## 9. 🔹 One-line Revision
KNN = local vote/average by distance—simple, needs scaling and careful k/distance.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q6: Implement Sigmoid, Tanh, ReLU, LeakyReLU, and Softmax Activation Functions.

## 1. 🔹 Direct Answer
- **Sigmoid/Tanh**: squash to (0,1) or (−1,1); tanh zero-centered.
- **ReLU**: max(0,x)—sparse, fast; **LeakyReLU**: small slope for x<0.
- **Softmax**: maps vector to **simplex**—multiclass probabilities.

## 2. 🔹 Intuition
Nonlinearity enables depth; softmax turns logits into **competing** class masses.

## 3. 🔹 Deep Dive
- Softmax: **e^{z_i} / Σ e^{z_j}**; numerical stability: subtract max(z).
- ReLU dying neurons—Leaky/GELU mitigate.

## 4. 🔹 Practical Perspective
Hidden layers: ReLU/GELU; output multiclass: softmax + CE.

## 5. 🔹 Code Snippet
```python
import numpy as np
def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)
relu = lambda x: np.maximum(0, x)
leaky = lambda x, a=0.01: np.where(x > 0, x, a * x)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Softmax + CE derivative? **A:** **p − y** (clean backprop).

## 7. 🔹 Common Mistakes
Softmax without log-sum-exp stabilization → overflow.

## 8. 🔹 Comparison / Connections
GELU, Swish, log-softmax for numerical stability.

## 9. 🔹 One-line Revision
ReLU family for hidden layers; softmax for multiclass probs—stabilize softmax with max subtraction.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q7: How would you implement k-means clustering?

## 1. 🔹 Direct Answer
Alternate: (1) assign each point to **nearest centroid**, (2) **recompute centroids** as cluster means until convergence. Minimizes within-cluster sum of squares (non-convex—**random restarts**).

## 2. 🔹 Intuition
Partition space into **Voronoi** cells around centers; iterate like EM’s hard version.

## 3. 🔹 Deep Dive
- **k-means++** for smarter init.
- **Choose k**: elbow, silhouette, domain knowledge.

## 4. 🔹 Practical Perspective
- Scale features; sensitive to **outliers**—use **k-medoids** or trim.
- Use for **preprocessing** or **exploration**, not ground truth clusters always.

## 5. 🔹 Code Snippet
```python
import numpy as np
def kmeans(X, k, iters=100):
    rng = np.random.default_rng(0)
    c = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = ((X[:,None,:] - c[None,:,:])**2).sum(2).argmin(1)
        new = np.array([X[lab==j].mean(0) for j in range(k)])
        if np.allclose(c, new): break
        c = new
    return c, lab
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Empty cluster? **A:** Reinit centroid or remove k.

## 7. 🔹 Common Mistakes
Not restarting; wrong distance metric for categorical data.

## 8. 🔹 Comparison / Connections
GMM (soft), hierarchical clustering, spectral clustering.

## 9. 🔹 One-line Revision
k-means alternates assignment and centroid update—init matters; use k-means++ and restarts.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q8: Write code to perform k-fold cross-validation.

## 1. 🔹 Direct Answer
Split data into **k** folds; train on **k−1**, validate on **1**, rotate. Average scores—**lower variance** estimate than single split.

## 2. 🔹 Intuition
Every point gets to be validation once—better use of limited data.

## 3. 🔹 Deep Dive
- **Stratified** for classification—preserve class rates per fold.
- **Time series**: use **forward chaining**, not shuffle.

## 4. 🔹 Practical Perspective
- sklearn: `cross_val_score`, `KFold`, `StratifiedKFold`.
- **Leakage**: fit preprocessing **inside** each fold.

## 5. 🔹 Code Snippet
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=cv, scoring="roc_auc")
print(scores.mean(), scores.std())
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Nested CV? **A:** Outer for test estimate, inner for hyperparams.

## 7. 🔹 Common Mistakes
Fitting scaler on full data before CV—leaks validation stats.

## 8. 🔹 Comparison / Connections
Bootstrap, holdout, LOOCV.

## 9. 🔹 One-line Revision
k-fold CV rotates val folds—stratify classification; avoid leakage by fitting pipelines per fold.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q9: How would you use Pandas to load and clean data?

## 1. 🔹 Direct Answer
**read_csv** / **parquet** → inspect **dtypes**, **missing**, **duplicates** → **fill/drop** imputation, **outlier** handling, **encode** categoricals, **parse** dates, **merge** tables—**reproducible** in functions or pipelines.

## 2. 🔹 Intuition
Most model failures are **data** issues—Pandas is the workbench.

## 3. 🔹 Deep Dive
- `df.info()`, `describe()`, `isna().mean()`
- `pd.to_datetime`, `astype('category')`
- **Leakage**: drop future columns.

## 4. 🔹 Practical Perspective
For big data: **Polars**, **Dask**, or **SQL** pushdown first.

## 5. 🔹 Code Snippet
```python
import pandas as pd
df = pd.read_csv("data.csv", parse_dates=["timestamp"])
df = df.drop_duplicates(subset=["id"])
df["x"] = df["x"].fillna(df["x"].median())
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Categorical high cardinality? **A:** Target encoding with CV, embeddings.

## 7. 🔹 Common Mistakes
Mean imputation **before** train/test split without pipeline.

## 8. 🔹 Comparison / Connections
sklearn Pipeline, Feature Store.

## 9. 🔹 One-line Revision
Load with correct dtypes, profile missing/outliers, clean without leakage—often inside CV pipeline.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q10: Implement k-nearest neighbors from scratch.

## 1. 🔹 Direct Answer
Same as **Q5**—distance matrix or incremental nearest neighbor search; for production use **sklearn** or **FAISS** for scale.

## 2. 🔹 Intuition
Brute force is fine for interviews/small **n**; index structures for large **n**.

## 3. 🔹 Deep Dive
- **Vectorization** over Python loops.
- **Ball tree** when **d** moderate and structured.

## 4. 🔹 Practical Perspective
Mention **approximate** NN (ANN) for embeddings at scale.

## 5. 🔹 Code Snippet
```python
# See Q5; batch prediction:
def knn_predict_batch(X_train, y_train, X_test, k=5):
    dist = np.sqrt(((X_train[:,None,:] - X_test[None,:,:])**2).sum(2))  # n x m
    idx = np.argpartition(dist, k, axis=0)[:k,:]
    # vote per column...
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Complexity? **A:** O(n_train * n_test * d) brute force.

## 7. 🔹 Common Mistakes
O(n²) nested loops in Python without vectorization.

## 8. 🔹 Comparison / Connections
ANN indexes, LSH.

## 9. 🔹 One-line Revision
KNN from scratch = all-pairs or argpartition per query—know complexity and scaling tricks.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q11: Write code to calculate precision and recall.

## 1. 🔹 Direct Answer
**Precision** = TP / (TP+FP); **Recall** = TP / (TP+FN). For multiclass: **macro/micro** averaging.

## 2. 🔹 Intuition
Precision: “of predicted positives, how many right?” Recall: “of actual positives, how many caught?”

## 3. 🔹 Deep Dive
**F1** = harmonic mean of P and R. **Confusion matrix** first.

## 4. 🔹 Practical Perspective
Imbalanced data: **don’t use accuracy** alone.

## 5. 🔹 Code Snippet
```python
import numpy as np
def precision_recall(y_true, y_pred):
    tp = np.sum((y_true==1) & (y_pred==1))
    fp = np.sum((y_true==0) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))
    p = tp / (tp + fp) if (tp+fp) else 0
    r = tp / (tp + fn) if (tp+fn) else 0
    return p, r
```

## 6. 🔹 Interview Follow-ups
1. **Q:** sklearn? **A:** `precision_score`, `recall_score`, `classification_report`.

## 7. 🔹 Common Mistakes
Using wrong positive class definition in multiclass one-vs-rest.

## 8. 🔹 Comparison / Connections
ROC-AUC, PR-AUC, calibration.

## 9. 🔹 One-line Revision
Precision targets false alarms; recall targets misses—code from TP/FP/FN counts.

## 10. 🔹 Difficulty Tag
🟢 Easy

---
