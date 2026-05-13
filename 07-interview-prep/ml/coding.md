# Coding

This file is for the coding round where you have to do two things at once:

- write working code
- narrate like a calm, competent human

That second part matters a lot.

Good ML coding interviews are not only checking syntax.
They are checking whether you understand:

- the algorithm
- edge cases
- tradeoffs
- what you would do in production

---

# 1. MSE and MAE

## What to say out loud

MSE averages squared error, so it punishes large mistakes more heavily.
MAE averages absolute error, so it is more robust to outliers.

That one sentence already makes the coding answer stronger.

## Code

```python
def mse(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Inputs must have the same length")
    if not y_true:
        raise ValueError("Inputs must be non-empty")
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


def mae(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Inputs must have the same length")
    if not y_true:
        raise ValueError("Inputs must be non-empty")
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)
```

**Nice follow-up line**

In production I would probably use NumPy or a vectorized implementation, but this is the clearest from-scratch version.

---

# 2. Linear Regression with Gradient Descent

## What to say out loud

I am using gradient descent here because it shows the learning logic clearly, even though linear regression also has a closed-form solution.

That is a strong framing line.

## Code

```python
class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.epochs):
            y_pred = [self.w * x + self.b for x in X]
            dw = sum((yp - yt) * x for x, yt, yp in zip(X, y, y_pred)) * 2 / n
            db = sum(yp - yt for yt, yp in zip(y, y_pred)) * 2 / n
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return [self.w * x + self.b for x in X]
```

**What interviewers like hearing**

- "I would add convergence checks in a fuller implementation."
- "For multiple features I would move to vectorized dot products."

---

# 3. Logistic Regression

## What to say out loud

Logistic regression uses a linear score, then pushes it through a sigmoid so the output becomes a probability between 0 and 1.

## Code

```python
import math


class LogisticRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0

    def _sigmoid(self, z):
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1 + ez)

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.epochs):
            probs = [self._sigmoid(self.w * x + self.b) for x in X]
            dw = sum((p - yt) * x for x, yt, p in zip(X, y, probs)) / n
            db = sum(p - yt for yt, p in zip(y, probs)) / n
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return [self._sigmoid(self.w * x + self.b) for x in X]

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]
```

**Useful narration**

I used a numerically safer sigmoid variant to avoid overflow for very large negative or positive inputs.

That is a lovely little detail to say.

---

# 4. K-Means

## What to say out loud

K-Means alternates between assigning points to the nearest centroid and recomputing centroids from assigned points.

That is the whole loop.

## Code

```python
def kmeans(points, k, max_iters=100):
    centroids = points[:k]
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]

        for point in points:
            idx = min(
                range(k),
                key=lambda i: sum((a - b) ** 2 for a, b in zip(point, centroids[i]))
            )
            clusters[idx].append(point)

        new_centroids = []
        for cluster, old_centroid in zip(clusters, centroids):
            if not cluster:
                new_centroids.append(old_centroid)
                continue
            dims = len(cluster[0])
            mean = [
                sum(point[d] for point in cluster) / len(cluster)
                for d in range(dims)
            ]
            new_centroids.append(mean)

        if new_centroids == centroids:
            break
        centroids = new_centroids

    return centroids
```

**Good follow-up**

In production I would use better centroid initialization like k-means++ and think carefully about scaling.

---

# 5. Precision, Recall, and F1

## What to say out loud

Precision tells me how reliable positive predictions are, recall tells me how much of the true positive class I recovered, and F1 balances the two.

## Code

```python
def precision_recall_f1(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
```

**Why this version is nice**

It handles division-by-zero cleanly instead of exploding dramatically.

Always appreciated.

---

# 6. Activation Functions

## What to say out loud

I usually implement the clean scalar versions first, and then I would vectorize if needed.

## Code

```python
import math


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1 + ex)


def tanh(x):
    return math.tanh(x)


def relu(x):
    return max(0.0, x)


def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x


def softmax(xs):
    max_x = max(xs)
    exps = [math.exp(x - max_x) for x in xs]
    total = sum(exps)
    return [v / total for v in exps]
```

**Good narration**

I subtract the max in softmax for numerical stability.

That is exactly the kind of small detail that signals real familiarity.

---

# 7. KNN

## What to say out loud

KNN is a lazy learner: training is just storing data, and the real work happens at inference when we compute nearest neighbors.

## Code

```python
from collections import Counter


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _distance(self, a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def predict_one(self, x_query):
        distances = [
            (self._distance(x_query, x_train), y)
            for x_train, y in zip(self.X_train, self.y_train)
        ]
        neighbors = sorted(distances, key=lambda item: item[0])[:self.k]
        return Counter(label for _, label in neighbors).most_common(1)[0][0]

    def predict(self, X):
        return [self.predict_one(x) for x in X]
```

**Nice line to add**

I would definitely normalize features before using this in practice because KNN is distance-sensitive.

---

# 8. K-Fold Cross Validation

## What to say out loud

The core idea is to rotate which slice acts as validation so that performance is less dependent on one lucky split.

## Code

```python
def k_fold_split(data, k):
    n = len(data)
    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = n if i == k - 1 else (i + 1) * fold_size
        val = data[start:end]
        train = data[:start] + data[end:]
        folds.append((train, val))

    return folds
```

**Good follow-up**

If the label distribution is imbalanced, I would prefer stratified k-fold rather than plain slicing.

---

# 9. Pandas Data Cleaning

## What to say out loud

I would show that I care about duplicates, missingness, basic type handling, and reproducibility, not just loading the CSV and hoping for the best.

## Code

```python
import pandas as pd


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.dropna(subset=["target"])

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].fillna("missing")

    return df
```

**Useful comment**

In production I would put this logic into a tested preprocessing pipeline rather than scattering notebook-only cleanup steps.

---

# 10. How to Narrate Well While Coding

This is the bonus section most people need.

While coding, say things like:

- "I will start with the simplest correct version."
- "I am adding edge-case handling here."
- "I would vectorize this in a production implementation."
- "This is `O(nk)` because I compare each point to each centroid."
- "I am choosing clarity over micro-optimization first."

That narration makes you sound collaborative, structured, and deliberate.

Which is exactly what interviewers want.

---

# Mini Pop Quiz

What is better in a live coding round:

- the fanciest possible implementation
- or the clearest correct baseline plus smart commentary

Correct answer:

The second one.

Always.
