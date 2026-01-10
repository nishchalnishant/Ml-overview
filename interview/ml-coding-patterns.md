# ML Coding Patterns: Scikit-Learn, NumPy, & PyTorch

In an "ML Coding" or "Data Manipulation" round, speed and correctness matter. Here are the templates you should have in muscle memory.

---

## 1. The Scikit-Learn Pipeline (Best Practice)
Avoid data leakage by using Pipelines for scaling and modeling.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# 1. Define
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

# 2. Tune
params = {'clf__C': [0.1, 1, 10]}
grid = GridSearchCV(pipeline, params, cv=5)
grid.fit(X_train, y_train)

# 3. Predict
best_model = grid.best_estimator_
preds = best_model.predict(X_test)
```

---

## 2. NumPy Vectorization Tricks
Interviewer: "Implement this without for-loops."

**Problem:** Calculate Euclidean distance between two matrices $A$ ($N \times D$) and $B$ ($M \times D$).
```python
# (A-B)^2 = A^2 + B^2 - 2AB
A_sq = np.sum(A**2, axis=1).reshape(-1, 1)
B_sq = np.sum(B**2, axis=1)
dist = np.sqrt(A_sq + B_sq - 2 * A.dot(B.T))
```

**Problem:** Find indices of unique elements and counts.
```python
vals, counts = np.unique(arr, return_counts=True)
most_freq = vals[np.argmax(counts)]
```

---

## 3. PyTorch Training Loop (Skeletal)
You might be asked to "Write a training loop from scratch".

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    # Training Stage
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    
    loss.backward() # Computes gradients
    optimizer.step() # Updates weights
    
    # Validation Stage
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
```

---

## 4. Evaluation Slicing (Production ML)
Often asked: "How do you check if your model is biased against a certain group?"

```python
import pandas as pd

# Assume 'gender' is a feature
def calculate_slice_metrics(df, label_col, pred_col, slice_col):
    for group in df[slice_col].unique():
        subset = df[df[slice_col] == group]
        acc = (subset[label_col] == subset[pred_col]).mean()
        print(f"Group: {group}, Accuracy: {acc:.2f}")
```

---

## 5. Implement K-Means (Simplified)
Usually as a toy implementation problem.

```python
def kmeans(X, k, iters=100):
    # 1. Random Init
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(iters):
        # 2. Assign to nearest
        dists = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(dists, axis=1)
        
        # 3. Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # 4. Check convergence
        if np.all(centroids == new_centroids): break
        centroids = new_centroids
        
    return centroids, labels
```
