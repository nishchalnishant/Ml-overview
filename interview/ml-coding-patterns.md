# ML Coding Patterns: 30+ Templates & Questions

---

## Scikit-Learn Patterns

### 1. Basic Pipeline (Prevent Data Leakage)
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
```

### 2. Grid Search with Cross-Validation
```python
from sklearn.model_selection import GridSearchCV

params = {'clf__C': [0.1, 1, 10], 'clf__penalty': ['l1', 'l2']}
grid = GridSearchCV(pipeline, params, cv=5, scoring='f1')
grid.fit(X_train, y_train)
print(grid.best_params_)
```

### 3. Random Search (Faster)
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

params = {'clf__C': uniform(0.1, 10)}
search = RandomizedSearchCV(pipeline, params, n_iter=20, cv=5)
```

### 4. Column Transformer (Mixed Types)
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income']),
    ('cat', OneHotEncoder(), ['city', 'gender'])
])
```

### 5. Stratified K-Fold
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
```

### 6. Class Weight for Imbalance
```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(class_weight='balanced')
# or explicit: class_weight={0: 1, 1: 10}
```

---

## NumPy Vectorization

### 7. Euclidean Distance Matrix
```python
import numpy as np

# Between all pairs of rows in A and B
A_sq = np.sum(A**2, axis=1).reshape(-1, 1)
B_sq = np.sum(B**2, axis=1)
dist = np.sqrt(A_sq + B_sq - 2 * A.dot(B.T))
```

### 8. Normalize Vectors (L2 Norm)
```python
norm = np.linalg.norm(X, axis=1, keepdims=True)
X_normalized = X / (norm + 1e-9)  # Avoid divide by zero
```

### 9. Softmax Implementation
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

### 10. One-Hot Encoding
```python
def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]
```

### 11. Argmax with Random Tie-Breaking
```python
def argmax_random(arr):
    max_val = np.max(arr)
    indices = np.where(arr == max_val)[0]
    return np.random.choice(indices)
```

### 12. Moving Average
```python
def moving_average(x, window):
    return np.convolve(x, np.ones(window)/window, mode='valid')
```

---

## PyTorch Patterns

### 13. Basic Training Loop
```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
```

### 14. Validation Loop
```python
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    val_loss = criterion(val_outputs, y_val)
```

### 15. Save and Load Model
```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### 16. Custom Dataset
```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

loader = DataLoader(MyDataset(X, y), batch_size=32, shuffle=True)
```

### 17. Learning Rate Scheduler
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(epochs):
    train()
    scheduler.step()
```

### 18. Early Stopping
```python
best_loss = float('inf')
patience = 5
counter = 0

for epoch in range(epochs):
    val_loss = validate()
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best.pth')
    else:
        counter += 1
        if counter >= patience:
            break
```

---

## Evaluation Code

### 19. Classification Report
```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

### 20. ROC-AUC
```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

auc = roc_auc_score(y_true, y_proba)
fpr, tpr, _ = roc_curve(y_true, y_proba)
plt.plot(fpr, tpr, label=f'AUC={auc:.2f}')
```

### 21. Precision-Recall Curve
```python
from sklearn.metrics import precision_recall_curve, average_precision_score

ap = average_precision_score(y_true, y_proba)
precision, recall, _ = precision_recall_curve(y_true, y_proba)
```

### 22. Feature Importance (Tree Models)
```python
import pandas as pd

importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

---

## Implement From Scratch

### 23. K-Means Clustering
```python
def kmeans(X, k, iters=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(iters):
        dists = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([X[labels == i].mean(0) for i in range(k)])
        if np.allclose(centroids, new_centroids): break
        centroids = new_centroids
    return labels, centroids
```

### 24. Logistic Regression (Gradient Descent)
```python
def logistic_regression(X, y, lr=0.01, iters=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    for _ in range(iters):
        z = X.dot(w) + b
        pred = 1 / (1 + np.exp(-z))
        dw = (1/m) * X.T.dot(pred - y)
        db = (1/m) * np.sum(pred - y)
        w -= lr * dw
        b -= lr * db
    return w, b
```

### 25. Naive Bayes (Gaussian)
```python
def fit_gaussian_nb(X, y):
    classes = np.unique(y)
    params = {}
    for c in classes:
        X_c = X[y == c]
        params[c] = {'mean': X_c.mean(0), 'var': X_c.var(0), 'prior': len(X_c)/len(X)}
    return params
```

### 26. KNN Classifier
```python
def knn_predict(X_train, y_train, X_test, k=3):
    dists = np.linalg.norm(X_test[:, None] - X_train, axis=2)
    nearest = np.argsort(dists, axis=1)[:, :k]
    predictions = []
    for neighbors in nearest:
        labels = y_train[neighbors]
        predictions.append(np.bincount(labels).argmax())
    return np.array(predictions)
```

---

## Interview Coding Questions

**27. "How would you compute the cosine similarity between two vectors?"**
```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**28. "Implement sigmoid function."**
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

**29. "Implement binary cross-entropy loss."**
```python
def bce_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
```

**30. "Write a function to compute precision, recall, and F1."**
```python
def metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1
```
