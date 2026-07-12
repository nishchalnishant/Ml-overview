---
module: Interview Prep
topic: Llm
subtopic: Ml Coding Patterns
status: unread
tags: [interviewprep, ml, llm-ml-coding-patterns]
---
# ML Coding Patterns — First-Principles Interview Reference

Each pattern starts with what breaks in the naive version, then shows the fix.

---

## 1. Numerically Stable Softmax

### What goes wrong in the naive version

```python
# Naive — breaks on large inputs
def softmax_naive(x):
    e = np.exp(x)
    return e / e.sum()
```

Test with `x = [1000, 999]`: `np.exp(1000)` overflows to `inf`. `inf / (inf + inf) = nan`. The model is dead.

### Why the pattern exists

Softmax is invariant to adding a constant: $\text{softmax}(x_i) = \text{softmax}(x_i - c)$ for any constant $c$, since the $e^{-c}$ factor cancels between numerator and denominator. Choosing $c = \max(x)$ makes the largest exponent $e^0 = 1$, so nothing overflows.

```python
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)  # shift: max element → 0, rest → negative
    e = np.exp(x)                           # e^0 = 1 is the largest value now
    return e / e.sum(axis=-1, keepdims=True)
```

**The same logic applies to log-softmax**, which is the loss for cross-entropy:
```python
def log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
    # More stable: x - (max + log(sum(exp(x - max))))
```

---

## 2. Feature Leakage via Preprocessing

### What goes wrong in the naive version

```python
# Naive — leaks test statistics into training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # fitted on ALL data including test
X_train, X_test = train_test_split(X_scaled, ...)
```

The scaler's mean/std are computed on the full dataset, including test examples — the model has effectively seen the test distribution. This produces optimistic metrics that won't hold in production, where the scaler is fit only on historical data.

### Why the pattern exists

Rule: **any statistics used to transform data must be computed only on training data.** This mirrors production: fit preprocessing on historical data, apply those fixed statistics to new data.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

num_cols = ["age", "income", "credit_score"]
cat_cols = ["city", "job_type"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat_cols),
])

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(C=1.0, max_iter=1000)),
])

pipe.fit(X_train, y_train)         # scaler.fit() sees only X_train
pipe.score(X_val, y_val)           # scaler.transform() applies train stats to val
```

The Pipeline guarantees: `fit` on train only, `transform` on val/test using train-derived statistics.

---

## 3. Logistic Regression from Scratch

### What goes wrong in the naive version

```python
# Naive — no clipping, no regularization, no numerical safety
def predict(X, w, b):
    return 1 / (1 + np.exp(-(X @ w + b)))  # crashes on large negative inputs
```

`np.exp(-(-500))` = `np.exp(500)` = `inf`. The sigmoid returns 0, the log returns `-inf`, the loss is `nan`. Training stops.

### Why the pattern exists

Clip the pre-activation before exponentiation. Also: L2 regularization prevents weight explosion, which is what causes the extreme inputs in the first place.

```python
import numpy as np

class LogisticRegression:
    def __init__(self, lr: float = 0.01, n_iter: int = 1000, lambda_: float = 0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.lambda_ = lambda_

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.n_iter):
            p = self._sigmoid(X @ self.w + self.b)

            # Gradient of BCE + L2 regularization
            # Derivation: ∂L/∂w = (p - y)^T X / n + λw
            dw = (X.T @ (p - y)) / n + self.lambda_ * self.w
            db = (p - y).mean()

            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

---

## 4. K-Means from Scratch

### What goes wrong in the naive version

Random initialization: if two initial centroids land in the same dense cluster, one of them will never "win" any points — you effectively get $k-1$ clusters. The inertia varies wildly across runs; you'd need to run many times to get a good solution.

### Why the pattern exists

K-Means++ initialization solves this probabilistically: after placing the first centroid uniformly at random, each subsequent centroid is placed with probability proportional to its squared distance from the nearest existing centroid. This spreads centroids across the data space, dramatically reducing the likelihood of bad initializations.

```python
import numpy as np

class KMeans:
    def __init__(self, k: int, n_iter: int = 100, tol: float = 1e-4, random_state: int = 42):
        self.k = k
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

    def _init_centroids(self, X):
        # K-Means++: spread centroids proportional to distance^2
        idx = self.rng.integers(len(X))
        centroids = [X[idx]]
        for _ in range(self.k - 1):
            dists = np.array([min(np.linalg.norm(x - c)**2 for c in centroids) for x in X])
            probs = dists / dists.sum()
            idx = self.rng.choice(len(X), p=probs)
            centroids.append(X[idx])
        return np.array(centroids)

    def fit(self, X):
        self.centroids_ = self._init_centroids(X)

        for _ in range(self.n_iter):
            # E-step: assign each point to the nearest centroid
            dists = np.linalg.norm(X[:, None] - self.centroids_[None], axis=2)  # (n, k)
            labels = dists.argmin(axis=1)

            # M-step: recompute centroids as cluster means
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if (labels == j).any() else self.centroids_[j]
                for j in range(self.k)
            ])

            if np.linalg.norm(new_centroids - self.centroids_) < self.tol:
                break
            self.centroids_ = new_centroids

        self.labels_ = labels
        return self

    def predict(self, X):
        dists = np.linalg.norm(X[:, None] - self.centroids_[None], axis=2)
        return dists.argmin(axis=1)
```

**The empty cluster edge case** (line `if (labels == j).any() else self.centroids_[j]`): if a centroid ends up with no assigned points — possible if initialization puts two centroids in the same cluster — leave it unchanged rather than crashing on `.mean()` of an empty array.

---

## 5. Precision, Recall, F1 — and ROC-AUC from Scratch

### What goes wrong in the naive version

Forgetting the denominators, or getting them swapped. Precision denominates on *predicted positives*; recall denominates on *actual positives*.

```python
import numpy as np

def precision_recall_f1(y_true, y_pred, eps=1e-8):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()   # called positive, was negative
    fn = ((y_pred == 0) & (y_true == 1)).sum()   # called negative, was positive

    # Precision: of what we said was positive, how many actually were?
    precision = tp / (tp + fp + eps)
    # Recall: of all actual positives, how many did we find?
    recall = tp / (tp + fn + eps)
    # F1: harmonic mean — penalizes if either is very low
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {"precision": precision, "recall": recall, "f1": f1}


def roc_auc_from_scratch(y_true, y_scores):
    """Compute ROC-AUC without sklearn — demonstrates the sorting + trapezoid logic."""
    order = np.argsort(-y_scores)          # sort by score descending
    y_sorted = y_true[order]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    # At each threshold, cumulative TP and FP counts
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    tpr = tp / n_pos
    fpr = fp / n_neg

    return np.trapz(tpr, fpr)             # trapezoidal rule on the ROC curve
```

---

## 6. Cross-Validation with Stratification

### What goes wrong in the naive version

```python
# Naive — random folds may be heavily imbalanced
scores = cross_val_score(model, X, y, cv=5)
```

With 95%/5% class imbalance, a fold might have 0% positive examples — the model can't learn and the metric is meaningless.

### Why the pattern exists

Stratified K-Fold ensures each fold preserves the class distribution of the full dataset. Each fold is a fair miniature of the real problem.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
print(f"ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

**For time series:** never use random folds. Use `TimeSeriesSplit` (each fold's test set is strictly after the training set):
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# fold 1: train on weeks 1-10, test on week 11
# fold 2: train on weeks 1-11, test on week 12
# ...
```

---

## 7. Complete PyTorch Training Loop

### What goes wrong in the naive version

Common bugs in order of occurrence:
1. Forgetting `optimizer.zero_grad()` — gradients accumulate across batches, making updates wrong
2. Forgetting `model.eval()` at inference — dropout stays active, predictions vary across calls
3. Gradient clipping missing — exploding gradients crash training silently (loss goes to `nan`)
4. Not saving optimizer state in checkpoints — resuming training resets the momentum, causing instability

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

def train_epoch(model, loader, optimizer, criterion, device,
                scheduler=None, max_grad_norm=1.0, use_amp=True):
    model.train()
    scaler = GradScaler(enabled=use_amp)
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()                          # must be first in the loop

        with autocast(dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # before step
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()                           # after optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()                                       # disables dropout, uses BN running stats
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        total_loss += loss.item()

    return total_loss / len(loader), correct / total
```

**Complete checkpoint — why optimizer state is required:**
```python
torch.save({
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),  # contains momentum buffers (m, v in Adam)
    "scheduler": scheduler.state_dict() if scheduler else None,
    "val_loss": val_loss,
}, "checkpoint.pt")

# Load and resume correctly
ckpt = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])   # restores momentum — without this, training restarts
start_epoch = ckpt["epoch"] + 1
```

---

## 8. Attention from Scratch (NumPy)

### What goes wrong in the naive version

Forgetting the mask handling: without masking, future tokens are visible during training of decoder models, causing information leakage — the model learns to cheat by attending to future tokens, then fails completely at inference when future tokens don't exist.

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, seq, d_k)
    mask: boolean array, True where attention is allowed
    Returns: (batch, seq_q, d_v)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)   # (batch, seq_q, seq_k)

    if mask is not None:
        # Positions where mask is False get -1e9 → softmax assigns ~0 weight
        scores = np.where(mask, scores, -1e9)

    # Numerically stable softmax along key dimension
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)

    return weights @ V   # (batch, seq_q, d_v)


def causal_mask(seq_len):
    """Upper triangular mask: position i can only attend to positions <= i."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))
```

---

## 9. Gradient Descent Variants from Scratch

### What goes wrong in the naive version

Plain SGD oscillates in narrow valleys — it takes large steps along the steep dimension and small steps along the shallow dimension. Momentum smooths this by accumulating velocity. Adam additionally scales by per-parameter gradient magnitude, giving each weight its own effective learning rate.

```python
import numpy as np

class AdamW:
    """AdamW: Adam with decoupled weight decay.
    Key difference from Adam: weight decay applied directly to params,
    not folded into the gradient. This prevents weight decay from interacting
    with adaptive learning rates in a harmful way."""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p) for p in params]   # first moment
        self.v = [np.zeros_like(p) for p in params]   # second moment
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            # Weight decay: shrink parameters directly (not via gradient)
            if self.weight_decay > 0:
                p *= (1 - self.lr * self.weight_decay)

            # Update biased moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2

            # Bias correction: corrects for initialization at zero
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Update: divide by sqrt of second moment (per-parameter LR scaling)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

---

## 10. Patterns to Apply Automatically

These are the invariants of correct ML code. Violations indicate a bug, not a style preference:

**Always subtract max before softmax/log-softmax.**
Reason: exp overflow. Invariance to additive constants makes this safe.

**Always add `eps` in precision/recall/F1 denominators.**
Reason: at the start of training (or on empty folds), TP+FP or TP+FN can be 0. Division by zero produces NaN, which propagates silently.

**Always call `optimizer.zero_grad()` before `loss.backward()`.**
Reason: PyTorch accumulates gradients. Not zeroing means each backward pass adds to the previous one — training with the average gradient of all past batches.

**Always call `model.eval()` AND `torch.no_grad()` for inference.**
`model.eval()` is about mode (disables dropout, changes BatchNorm behavior). `torch.no_grad()` is about memory (stops building the autograd graph). Both are needed. One without the other gives wrong results or wastes memory.

**Always use `pin_memory=True` and `num_workers > 0` in DataLoader for GPU training.**
Reason: with `pin_memory=True`, the DataLoader allocates tensors in pinned (page-locked) CPU memory, enabling faster async host-to-device transfers. Without it, each batch transfer blocks the GPU.

**Always save optimizer state in checkpoints.**
Reason: Adam's `m` and `v` buffers contain the gradient history. Loading only model weights and restarting the optimizer resets this history — the first few steps after resumption have wrong update directions and magnitudes.
