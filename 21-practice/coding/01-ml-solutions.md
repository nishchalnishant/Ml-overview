---
module: Practice
topic: Coding Drills
subtopic: ML Solutions
status: unread
tags: [practice, coding, ml, solutions]
---
# ML Implementation Problems — Solutions

Problems: [01-ml-problems.md](01-ml-problems.md). Attempt before reading.

Each solution gives working code plus **what is actually scored** — the thing the
interviewer is checking that is not "does it run."

---

## Tier 1 — Warm-ups

### 1.1 Numerically stable softmax

```python
import numpy as np

def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)                      # shift: the entire trick
    return e / np.sum(e, axis=axis, keepdims=True)
```

**Scored:** the `x - x_max` shift, and knowing *why* it is safe — subtracting a constant from
all logits leaves the softmax unchanged, because the constant factors out of numerator and
denominator. Without it, `exp(1002)` overflows to `inf` and you get `nan`. `keepdims=True` is
the second thing checked; without it the broadcast silently transposes your result.

### 1.2 Train/test split

```python
def train_test_split(X, y, test_size=0.2, seed=None):
    X, y = np.asarray(X), np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))              # permute INDICES, not arrays separately
    n_test = int(round(len(X) * test_size))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

**Scored:** shuffling a single index array rather than shuffling `X` and `y` independently —
the latter destroys the correspondence and is a genuine interview failure. `default_rng` over
the legacy `np.random.seed` global shows current NumPy practice.

### 1.3 StandardScaler

```python
class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0    # constant column -> no-op, not NaN
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

**Scored:** the zero-variance guard, and articulating the leakage rule — statistics come from
training data only, because letting the test set influence `mean_` leaks information about
data the model is supposed to have never seen, inflating your estimate of generalization.

### 1.4 Confusion matrix and metrics

```python
def binary_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=precision, recall=recall, f1=f1)
```

**Scored:** every zero-denominator handled, and having a *defensible* position on
precision-when-nothing-predicted-positive. Returning 0.0 is conventional (sklearn does, with
a warning); arguing it is undefined is also acceptable. Having no position is not.

### 1.5 One-hot with unseen categories

```python
class OneHotEncoder:
    def fit(self, values):
        self.categories_ = sorted(set(values))
        self.index_ = {c: i for i, c in enumerate(self.categories_)}
        return self

    def transform(self, values):
        out = np.zeros((len(values), len(self.categories_)))
        for row, v in enumerate(values):
            j = self.index_.get(v)
            if j is not None:
                out[row, j] = 1                # unseen -> all-zero row
        return out
```

**Scored:** not crashing on unseen input, and naming the policy. All-zeros means "none of the
known categories," which is usually right. The alternative — an explicit `__unknown__` column
reserved at fit time — is better when unseen categories are expected to be common and
predictive. Say which you chose and when you would switch.

### 1.6 Batch generator

```python
def batches(X, y, batch_size, shuffle=True, drop_last=False, seed=None):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    last = (n // batch_size) * batch_size if drop_last else n
    for start in range(0, last, batch_size):
        sel = idx[start:start + batch_size]
        yield X[sel], y[sel]
```

**Scored:** the partial final batch is handled rather than silently dropped or crashed on,
and shuffling happens per call (per epoch) rather than once at definition.

---

## Tier 2 — Core algorithms

### 2.1 Linear regression, two ways

```python
def fit_normal_equation(X, y, l2=0.0):
    X = np.c_[np.ones(len(X)), X]                        # bias column
    n_features = X.shape[1]
    A = X.T @ X + l2 * np.eye(n_features)
    A[0, 0] -= l2                                        # do not regularize the bias
    return np.linalg.solve(A, X.T @ y)                   # solve, never inv()

def fit_gradient_descent(X, y, lr=0.01, epochs=1000):
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        grad = 2 * X.T @ (X @ w - y) / len(X)
        w -= lr * grad
    return w
```

**Scored:** `np.linalg.solve` rather than `np.linalg.inv(...) @ ...` — explicitly forming the
inverse is slower and numerically worse. The crossover answer: the normal equation is
O(d³) in the number of features, so it stops being viable in the low thousands of features,
and it is unusable when `XᵀX` is singular or the data does not fit in memory. Not
regularizing the bias term is a detail that signals you have implemented this before.

### 2.2 Logistic regression with L2

```python
def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos, neg = z >= 0, z < 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[neg])                       # exp of a negative: always safe
    out[neg] = ez / (1 + ez)
    return out

def fit_logreg(X, y, lr=0.1, epochs=1000, l2=0.0):
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(X)
        reg = l2 * w
        reg[0] = 0
        w -= lr * (grad + reg)
    return w
```

**Scored:** the branched sigmoid. The naive `1/(1+np.exp(-z))` overflows for large negative
`z`. Most candidates write the naive version; writing the stable one unprompted is a strong
signal. Also scored: recognizing that the logistic gradient `Xᵀ(p − y)/n` has the same
*form* as the linear-regression gradient — that is not a coincidence, it falls out of both
being generalized linear models with their canonical link.

### 2.3 K-means with k-means++

```python
def kmeans_plusplus_init(X, k, rng):
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(len(X), 1 / len(X))
        centers.append(X[rng.choice(len(X), p=probs)])
    return np.array(centers)

def kmeans(X, k, iters=100, seed=None):
    rng = np.random.default_rng(seed)
    C = kmeans_plusplus_init(X, k, rng)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        labels = d.argmin(1)
        new_C = C.copy()
        for j in range(k):
            members = X[labels == j]
            if len(members):
                new_C[j] = members.mean(0)
            else:
                new_C[j] = X[d.min(1).argmax()]   # empty cluster -> farthest point
        if np.allclose(new_C, C):
            break                                  # converged
        C = new_C
    return C, labels
```

**Scored:** three things. The empty-cluster branch (naive code does `mean of []` → `nan` →
silent corruption of every subsequent iteration). The convergence check rather than always
burning all iterations. And being able to state what k-means++ buys: seeding proportional to
squared distance gives an O(log k) approximation guarantee in expectation, versus none at all
for random init.

### 2.4 k-NN

```python
from collections import Counter

def knn_predict(X_train, y_train, X_query, k=5, metric="euclidean"):
    if metric == "euclidean":
        d = np.sqrt(((X_query[:, None, :] - X_train[None]) ** 2).sum(-1))
    elif metric == "manhattan":
        d = np.abs(X_query[:, None, :] - X_train[None]).sum(-1)
    else:
        raise ValueError(f"unknown metric {metric}")

    preds = []
    for row in d:
        nn = np.argsort(row, kind="stable")[:k]     # stable sort -> deterministic ties
        votes = Counter(y_train[nn])
        top = max(votes.values())
        winners = sorted(c for c, v in votes.items() if v == top)
        preds.append(winners[0])                    # tie -> smallest label
    return np.array(preds)
```

**Scored:** a stated, deterministic tie-break. Interviewers ask "what if two classes tie?"
and the answer must be a rule, not a shrug. Also worth volunteering: k-NN has no training
cost and O(n·d) query cost, which inverts the usual tradeoff and makes it a poor fit for
low-latency serving without an ANN index.

### 2.5 Best split by Gini

```python
def gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p ** 2)

def best_split(X, y):
    best = (None, None, -np.inf)
    parent = gini(y)
    for f in range(X.shape[1]):
        order = np.argsort(X[:, f])
        xs, ys = X[order, f], y[order]
        for i in range(1, len(xs)):
            if xs[i] == xs[i - 1]:
                continue                     # cannot split between equal values
            thr = (xs[i] + xs[i - 1]) / 2
            left, right = ys[:i], ys[i:]
            child = (len(left) * gini(left) + len(right) * gini(right)) / len(ys)
            gain = parent - child
            if gain > best[2]:
                best = (f, thr, gain)
    return best
```

**Scored:** skipping duplicate values (splitting between two identical feature values is not
a valid split and produces an empty child), and the *weighted* child impurity — averaging the
two children unweighted is a common and wrong shortcut.

### 2.6 Multinomial Naive Bayes

```python
class MultinomialNB:
    def fit(self, docs, labels, alpha=1.0):
        self.classes_ = sorted(set(labels))
        vocab = sorted({t for d in docs for t in d})
        self.vocab_ = {t: i for i, t in enumerate(vocab)}
        V = len(vocab)
        self.log_prior_, self.log_lik_ = {}, {}
        for c in self.classes_:
            cdocs = [d for d, l in zip(docs, labels) if l == c]
            self.log_prior_[c] = np.log(len(cdocs) / len(docs))
            counts = np.zeros(V)
            for d in cdocs:
                for t in d:
                    if t in self.vocab_:
                        counts[self.vocab_[t]] += 1
            self.log_lik_[c] = np.log((counts + alpha) / (counts.sum() + alpha * V))
        return self

    def predict_one(self, doc):
        scores = {}
        for c in self.classes_:
            s = self.log_prior_[c]
            for t in doc:
                if t in self.vocab_:
                    s += self.log_lik_[c][self.vocab_[t]]
            scores[c] = s
        return max(scores, key=scores.get)
```

**Scored:** log space, and the reason — multiplying hundreds of probabilities underflows
float64 to exactly zero, after which every class ties at zero and the classifier is
worthless. Also scored: the smoothing denominator is `alpha * V`, not `alpha`; getting that
wrong means the likelihoods do not sum to one.

### 2.7 PCA

```python
def pca(X, k):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(0)                        # centering is mandatory
    cov = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)          # eigh: cov is symmetric
    order = np.argsort(vals)[::-1]            # eigh returns ASCENDING
    vals, vecs = vals[order], vecs[:, order]
    return Xc @ vecs[:, :k], vals[:k] / vals.sum()
```

**Scored:** `eigh` rather than `eig` (the covariance matrix is symmetric; `eigh` is faster and
returns real values guaranteed), and reversing the order because `eigh` returns ascending
eigenvalues — taking `[:k]` without reversing gives you the *least* important components,
which is a silent, plausible-looking failure. Centering is the third check.

### 2.8 ROC-AUC with ties

```python
def roc_auc(y_true, scores):
    y_true = np.asarray(y_true)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC undefined with only one class present")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    s = np.asarray(scores)[order]             # average ranks within tie groups
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1

    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
```

**Scored:** tie handling via average ranks. The Mann–Whitney identity — AUC equals the
probability a random positive scores above a random negative — is the insight to state out
loud; it explains both the rank formula and why AUC is threshold-independent. Raising on a
single-class input rather than returning a meaningless number is also checked.

### 2.9 Stratified k-fold

```python
def stratified_kfold(y, k=5, seed=None):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) < k:
            raise ValueError(f"class {c} has {len(idx)} members, fewer than k={k}")
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % k    # deal round-robin
    for f in range(k):
        yield np.where(fold_of != f)[0], np.where(fold_of == f)[0]
```

**Scored:** dealing each class's shuffled indices round-robin across folds is the whole
technique — it keeps proportions right without any bookkeeping. Raising a clear error for an
under-populated class beats producing folds that silently lack a class entirely.

### 2.10 GD variants

```python
def sgd(grad_fn, w, lr=0.01, steps=1000):
    for _ in range(steps):
        w = w - lr * grad_fn(w)
    return w

def momentum(grad_fn, w, lr=0.01, beta=0.9, steps=1000):
    v = np.zeros_like(w)
    for _ in range(steps):
        v = beta * v + grad_fn(w)
        w = w - lr * v
    return w

def adam(grad_fn, w, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, steps=1000):
    m, v = np.zeros_like(w), np.zeros_like(w)
    for t in range(1, steps + 1):
        g = grad_fn(w)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        m_hat = m / (1 - b1 ** t)              # bias correction
        v_hat = v / (1 - b2 ** t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    return w
```

**Scored:** bias correction, and why it matters — `m` and `v` start at zero, so early updates
are biased toward zero; without correction the first steps are far too small and training
starts slowly. Also: `eps` goes outside the square root. Inside, it does not do its job.

---

## Tier 3 — Deep learning

### 3.1 Two-layer backprop

```python
def forward_backward(X, Y, W1, b1, W2, b2):
    # forward
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    P = softmax(Z2, axis=1)
    n = len(X)
    loss = -np.sum(Y * np.log(P + 1e-12)) / n

    # backward
    dZ2 = (P - Y) / n                          # softmax+CE collapse to this
    dW2, db2 = A1.T @ dZ2, dZ2.sum(0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)                       # ReLU derivative
    dW1, db1 = X.T @ dZ1, dZ1.sum(0)
    return loss, dW1, db1, dW2, db2
```

**Scored:** knowing that softmax composed with cross-entropy has gradient `(P − Y)` — the
Jacobian of softmax alone is a messy matrix, and the fact that it collapses when paired with
CE is exactly the insight being tested. Gate on `Z1 > 0` (pre-activation), not `A1 > 0`;
they agree here but the former is the correct derivative. Verify with:

```python
def numerical_grad(f, w, eps=1e-5):
    g = np.zeros_like(w)
    it = np.nditer(w, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = w[i]
        w[i] = old + eps; hi = f()
        w[i] = old - eps; lo = f()
        w[i] = old
        g[i] = (hi - lo) / (2 * eps)           # central difference
        it.iternext()
    return g
```

Central difference, not forward — the error is O(eps²) rather than O(eps).

### 3.2 Scaled dot-product attention

```python
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask                 # additive -inf mask
    W = softmax(scores, axis=-1)
    return W @ V, W

def causal_mask(n):
    return np.triu(np.full((n, n), -np.inf), k=1)
```

**Scored:** the `sqrt(d_k)` and the reason. For `Q`, `K` with unit-variance entries, the dot
product over `d_k` dimensions has variance `d_k`, so the logits grow with dimension; feeding
large-magnitude logits into softmax saturates it, and the gradient vanishes. Dividing by
`sqrt(d_k)` restores unit variance. Additive `-inf` masking rather than multiplicative
zeroing is the other check — multiplying after softmax breaks the normalization.

### 3.3 Multi-head attention

```python
def multi_head_attention(X, Wq, Wk, Wv, Wo, h, mask=None):
    B, T, D = X.shape
    d_head = D // h
    def split(M):
        return M.reshape(B, T, h, d_head).transpose(0, 2, 1, 3)   # B,h,T,d
    Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    out, _ = attention(Q, K, V, mask)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)              # merge heads
    return out @ Wo
```

**Scored:** the transpose to `(B, h, T, d_head)` before attention and the exact inverse
afterward. Reshaping without transposing interleaves head dimensions with sequence positions
and produces wrong results that still have the right shape — the worst kind of bug, and the
one this problem exists to test.

### 3.4 LayerNorm forward and backward

```python
def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma * xhat + beta, (xhat, gamma, np.sqrt(var + eps))

def layernorm_backward(dout, cache):
    xhat, gamma, std = cache
    D = xhat.shape[-1]
    dxhat = dout * gamma
    dx = (dxhat
          - dxhat.mean(-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / std
    dgamma = (dout * xhat).sum(tuple(range(dout.ndim - 1)))
    dbeta = dout.sum(tuple(range(dout.ndim - 1)))
    return dx, dgamma, dbeta
```

**Scored:** the two correction terms in `dx`. They exist because `mu` and `var` are both
functions of every element of `x`, so each input affects the output through three paths.
Candidates who differentiate only the numerator get a plausible-looking gradient that fails a
numerical check — run one to confirm.

### 3.5 PyTorch training loop

```python
import torch

def train(model, train_loader, val_loader, epochs, lr, device="cpu"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()                                  # (1)
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()                            # (2)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()                                   # (3)
        correct = total = 0
        with torch.no_grad():                          # (4)
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += yb.numel()
        print(f"epoch {epoch}: val_acc={correct / total:.4f}")
```

**Scored:** the four numbered lines, which are the four most commonly omitted. `(1)` and
`(3)` toggle dropout and batchnorm behavior — forgetting `eval()` makes validation metrics
noisy and pessimistic. `(2)` is required because PyTorch accumulates gradients by default;
omitting it silently sums gradients across batches. `(4)` prevents building a graph you will
never backprop through, which otherwise wastes memory and can OOM on large validation sets.
Passing `CrossEntropyLoss` raw logits — not softmax output — is the fifth check.

### 3.6 Inverted dropout

```python
def dropout(x, p, training=True, rng=None):
    if not training or p == 0:
        return x
    rng = rng or np.random.default_rng()
    mask = (rng.random(x.shape) >= p) / (1 - p)   # scale at TRAIN time
    return x * mask
```

**Scored:** the `/(1 - p)` living in the training path. This is "inverted" dropout, and it is
what every framework does, because it makes inference a plain forward pass with no dropout
bookkeeping at all. The original formulation scaled at inference instead; naming the
difference is a strong signal.

### 3.7 Positional encoding

```python
def sinusoidal_pe(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
    return pe
```

**Scored:** the tradeoff. Sinusoidal encodings extrapolate to sequence lengths never seen in
training and add no parameters; learned encodings typically fit the training distribution
slightly better but are hard-capped at the maximum trained length. Modern long-context models
mostly use neither — RoPE rotates Q and K rather than adding to the embedding, which is what
makes context extension by interpolation possible. See
[10-llms/07-context-window-extension.md](../../10-llms/07-context-window-extension.md).

### 3.8 Beam search

```python
def beam_search(next_probs, start, eos, beam=3, max_len=50, alpha=0.7):
    beams = [(start, 0.0, False)]              # (tokens, logprob, finished)
    for _ in range(max_len):
        if all(f for _, _, f in beams):
            break
        cands = []
        for toks, lp, fin in beams:
            if fin:
                cands.append((toks, lp, True))  # finished beams carry over unchanged
                continue
            probs = next_probs(toks)
            for tok in np.argsort(probs)[-beam:]:
                cands.append((toks + [int(tok)],
                              lp + float(np.log(probs[tok] + 1e-12)),
                              int(tok) == eos))
        beams = sorted(cands,
                       key=lambda b: b[1] / (len(b[0]) ** alpha),
                       reverse=True)[:beam]
    return max(beams, key=lambda b: b[1] / (len(b[0]) ** alpha))[0]
```

**Scored:** finished beams stop extending but stay in the pool, and length normalization is
applied. Without normalization, beam search systematically prefers short sequences, because
every additional token adds a negative log-probability — a well-known failure that produces
truncated translations and summaries.

---

## Tier 4 — Applied and messy

### 4.1 Leak-free pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
pipe = Pipeline([("pre", pre), ("clf", model)])
scores = cross_val_score(pipe, X, y, cv=5)     # preprocessing refit inside each fold
```

**Scored:** the entire point is that preprocessing is *inside* the pipeline, so
`cross_val_score` refits it per fold. The leak this prevents: calling `fit_transform` on all
of `X` before splitting lets the imputer's median and the scaler's mean absorb information
from every validation fold, which inflates CV scores — sometimes dramatically on small data.
`handle_unknown="ignore"` prevents a fold containing an unseen category from crashing.

### 4.2 Walk-forward split with a gap

```python
def walk_forward(n, n_splits=5, test_size=None, gap=0):
    test_size = test_size or n // (n_splits + 1)
    for i in range(n_splits):
        test_end = n - (n_splits - i - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start - gap
        if train_end <= 0:
            continue
        yield np.arange(0, train_end), np.arange(test_start, test_end)
```

**Scored:** training indices always strictly precede test indices, and the `gap` exists to
handle label lag — if your label is "did this customer churn in the next 30 days," then rows
in the final 30 days before the test period have labels that overlap the test window, and
training on them leaks. Standard k-fold is invalid because it trains on future data to
predict the past, producing scores that cannot be realized in production.

### 4.3 Imbalanced classification

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

model = LGBMClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
model.fit(X_train, y_train)
p = model.predict_proba(X_val)[:, 1]

ap = average_precision_score(y_val, p)                    # PR-AUC: the metric to trust
prec, rec, thr = precision_recall_curve(y_val, p)
# choose the threshold from the business constraint, not from 0.5
viable = np.where(prec[:-1] >= 0.90)[0]
threshold = thr[viable[np.argmax(rec[viable])]] if len(viable) else 0.5
```

**Scored:** three points. PR-AUC over ROC-AUC at 1:200 — ROC-AUC is dominated by the large
negative class and stays deceptively high for a useless model. Accuracy is disqualifying:
predicting all-negative scores 99.5%. And the threshold comes from a stated business
constraint (here: precision ≥ 90%, maximize recall subject to it), never from the 0.5
default. Resampling is worth *mentioning* — SMOTE, undersampling — but class weighting is
usually the better first move, and any resampling must happen inside CV folds only.

### 4.4 Welford's online variance

```python
class RunningStats:
    def __init__(self):
        self.n, self.mean, self.M2 = 0, 0.0, 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)      # note: uses the UPDATED mean

    @property
    def variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else float("nan")
```

**Scored:** the second `delta` uses the updated mean — using the old one is the classic
transcription error and gives a subtly wrong answer. The naive `E[x²] − E[x]²` formulation
subtracts two large, nearly-equal numbers, so catastrophic cancellation can produce a
*negative* variance for data with a large mean and small spread. Welford is numerically
stable and single-pass, which is why it is what streaming systems actually use.

### 4.5 Top-k cosine similarity

```python
def topk_cosine(query, docs, k=10, docs_are_normalized=False):
    q = query / np.linalg.norm(query)
    D = docs if docs_are_normalized else docs / np.linalg.norm(docs, axis=1, keepdims=True)
    sims = D @ q                                   # normalized dot == cosine
    idx = np.argpartition(-sims, k)[:k]             # O(n), not O(n log n)
    return idx[np.argsort(-sims[idx])]              # sort only the k survivors
```

**Scored:** `argpartition` over a full `argsort` — O(n) versus O(n log n), which matters at
scale — and pre-normalizing the document matrix once rather than per query. The scaling
answer: at N = 10 million, exact search stops being viable and you move to an ANN index
(HNSW, IVF-PQ), accepting approximate recall for a large latency win. See
[11-llm-applications/02-rag.md](../../11-llm-applications/02-rag.md).

### 4.6 Leak-free rolling aggregates

```python
def rolling_features(df, window="7D"):
    df = df.sort_values(["user_id", "timestamp"])
    out = (df.set_index("timestamp")
             .groupby("user_id")["value"]
             .rolling(window, closed="left")      # EXCLUDE the current row
             .agg(["mean", "sum", "count"]))
    return out.reset_index()
```

**Scored:** `closed="left"`. The default window includes the current row, which means the
feature for row *t* contains row *t*'s own value — at training time this leaks the target's
own event into its features, and at serving time the feature cannot be computed the same way,
giving you train/serve skew on top of the leak. Sorting before grouping is the other check.
