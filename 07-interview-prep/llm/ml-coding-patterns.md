# ML Coding Patterns — Interview Reference

Common implementations you need to write from scratch or extend in coding rounds.

---

## 1. sklearn Pipeline (Feature Leakage Prevention)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

num_cols = ["age", "income", "credit_score"]
cat_cols = ["city", "job_type"]

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(C=1.0, max_iter=1000)),
])

# fit on train only — no leakage from val/test
pipe.fit(X_train, y_train)
print(pipe.score(X_val, y_val))
```

**Why pipelines prevent leakage:** `pipe.fit(X_train)` fits the scaler only on training data. `pipe.transform(X_val)` applies training statistics to val — never the reverse.

---

## 2. Cross-Validation with Stratification

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
print(f"ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

Use `StratifiedKFold` when class imbalance exists — ensures each fold has the same class ratio as the full dataset.

---

## 3. Softmax (Numerically Stable)

```python
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    # Subtract max for numerical stability: prevents overflow in exp()
    # Does NOT change output: softmax(x) = softmax(x - c) for any constant c
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

# Without stability: exp(1000) overflows to inf
# With stability: exp(1000 - 1000) = exp(0) = 1 — fine
```

---

## 4. Logistic Regression from Scratch

```python
import numpy as np

class LogisticRegression:
    def __init__(self, lr: float = 0.01, n_iter: int = 1000, lambda_: float = 0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.lambda_ = lambda_  # L2 regularization

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.n_iter):
            z = X @ self.w + self.b
            p = self._sigmoid(z)

            # Gradient of BCE + L2
            dw = (X.T @ (p - y)) / n + self.lambda_ * self.w
            db = (p - y).mean()

            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
```

---

## 5. K-Means from Scratch

```python
import numpy as np

class KMeans:
    def __init__(self, k: int, n_iter: int = 100, tol: float = 1e-4, random_state: int = 42):
        self.k = k
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        # K-Means++ initialization: probability proportional to squared distance
        idx = self.rng.integers(len(X))
        centroids = [X[idx]]
        for _ in range(self.k - 1):
            dists = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X])
            probs = dists / dists.sum()
            idx = self.rng.choice(len(X), p=probs)
            centroids.append(X[idx])
        return np.array(centroids)

    def fit(self, X: np.ndarray):
        self.centroids_ = self._init_centroids(X)

        for _ in range(self.n_iter):
            # Assignment step: each point to nearest centroid
            dists = np.linalg.norm(X[:, None] - self.centroids_[None], axis=2)  # (n, k)
            labels = dists.argmin(axis=1)

            # Update step: recompute centroids
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if (labels == j).any() else self.centroids_[j]
                for j in range(self.k)
            ])

            if np.linalg.norm(new_centroids - self.centroids_) < self.tol:
                break
            self.centroids_ = new_centroids

        self.labels_ = labels
        self.inertia_ = sum(
            np.linalg.norm(X[labels == j] - self.centroids_[j]) ** 2
            for j in range(self.k)
            if (labels == j).any()
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X[:, None] - self.centroids_[None], axis=2)
        return dists.argmin(axis=1)
```

---

## 6. Precision, Recall, F1 from Scratch

```python
import numpy as np

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def roc_auc_from_scratch(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn."""
    # Sort by score descending
    order = np.argsort(-y_scores)
    y_true_sorted = y_true[order]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    tpr = tp / n_pos
    fpr = fp / n_neg

    # Trapezoidal rule
    return np.trapz(tpr, fpr)
```

---

## 7. Complete PyTorch Training Loop

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

def train_epoch(model, loader, optimizer, criterion, device,
                scheduler=None, max_grad_norm=1.0, use_amp=True):
    model.train()
    scaler = GradScaler(enabled=use_amp)
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total
```

---

## 8. Attention from Scratch (NumPy)

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, seq, d_k)
    Returns: (batch, seq, d_v)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)  # (batch, seq_q, seq_k)

    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)

    return weights @ V  # (batch, seq_q, d_v)
```

---

## 9. Gradient Descent Variants from Scratch

```python
import numpy as np

class SGDOptimizer:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if self.weight_decay > 0:
                g = g + self.weight_decay * p
            self.velocities[i] = self.momentum * self.velocities[i] + (1 - self.momentum) * g
            p -= self.lr * self.velocities[i]


class AdamOptimizer:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            # AdamW: weight decay directly on params, not folded into gradient
            if self.weight_decay > 0:
                p *= (1 - self.lr * self.weight_decay)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

---

## 10. Common Coding Round Patterns

**Pattern: always subtract max in softmax/log-softmax** — prevents overflow.

**Pattern: use `eps` in denominators** — prevents division by zero in precision/recall/F1.

**Pattern: `optimizer.zero_grad()` before `loss.backward()`** — PyTorch accumulates gradients by default.

**Pattern: `model.eval()` + `torch.no_grad()` for inference** — disables dropout and gradient tracking.

**Pattern: `pin_memory=True, num_workers=4` in DataLoader** — overlap CPU data loading with GPU compute.

**Pattern: save optimizer state in checkpoints** — needed to resume training, not just the model weights.

```python
# Minimal but complete checkpoint
torch.save({
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler else None,
    "val_loss": val_loss,
}, "checkpoint.pt")

# Load
ckpt = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
start_epoch = ckpt["epoch"] + 1
```
