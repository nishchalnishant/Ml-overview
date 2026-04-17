# ML Coding Patterns

This file is for coding rounds where showing structure matters more than showing off.

You want:

- clean baseline
- sane APIs
- edge-case awareness
- quick explanation

---

## 1. Pipeline Pattern

Good for preventing leakage:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression())
])
```

Why it matters:

fit transforms only on training flow, not manually all over the notebook.

---

## 2. ColumnTransformer Pattern

Good for mixed numerical and categorical features:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
```

---

## 3. Stratified K-Fold

Use when class balance matters.

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 4. Basic PyTorch Training Loop

```python
for X, y in train_loader:
    optimizer.zero_grad()
    preds = model(X)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()
```

If you can write and explain this calmly, you are already in a good place.

---

## 5. Softmax

```python
import numpy as np

def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum()
```

Subtracting max is the important stability detail.

---

## 6. Logistic Regression from Scratch

Core idea:

- linear score
- sigmoid
- BCE gradient

That is the main coding-round story.

---

## 7. K-Means from Scratch

Core loop:

- initialize centroids
- assign points
- recompute centroids
- repeat

If you can explain that while writing, it goes far.

---

## 8. Precision / Recall / F1

Always remember:

- handle divide-by-zero
- say what each metric means

That sounds much better than dropping formulas with no explanation.
