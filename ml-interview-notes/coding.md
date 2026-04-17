# Coding

These notes are meant for whiteboard or live-coding interviews. The right answer is usually a clean baseline implementation plus a short explanation of edge cases, complexity, and what you would improve in production.

---

# Q1: Write a Python function to compute the mean squared error (MSE).

**What to say**

MSE is the average squared difference between predictions and targets. Squaring makes large errors matter more and keeps the metric differentiable, which is why it is common in regression.

```python
def mse(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Inputs must have the same length")
    if not y_true:
        raise ValueError("Inputs must be non-empty")
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
```

**Interview note**

Mention that if outliers are a concern, MAE may be more robust.

---

# Q2: Write a Python function to compute the mean absolute error (MAE).

**What to say**

MAE is the average absolute difference between predictions and targets. It is easier to interpret than MSE and less sensitive to outliers because the penalty grows linearly.

```python
def mae(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Inputs must have the same length")
    if not y_true:
        raise ValueError("Inputs must be non-empty")
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)
```

**Interview note**

If asked to compare with RMSE, say RMSE penalizes large errors more heavily and returns to the target unit scale.

---

# Q3: Implement a simple linear regression model from scratch.

**What to say**

For an interview, I would usually write a gradient-descent version because it demonstrates both the model and the optimization. I would also mention that in practice linear regression can be solved in closed form, but gradient descent scales better conceptually to larger models.

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

**Interview note**

If multiple features are needed, say you would replace the scalar weight with a vector and use a dot product.

---

# Q4: Implement a simple logistic regression model from scratch.

**What to say**

Logistic regression predicts probabilities, so I would compute a linear score, pass it through a sigmoid, and train with gradient descent on log loss.

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

**Interview note**

Mention that for real problems you would add regularization, vectorized computation, and multi-feature support.

---

# Q5: Implement K-Nearest Neighbors (KNN).

**What to say**

KNN is simple: store the training data, compute distances to a query point, take the nearest `k`, and vote. The important tradeoff is that training is trivial but inference can be expensive.

```python
from collections import Counter


def knn_predict(X_train, y_train, x_query, k=3):
    distances = []
    for x, y in zip(X_train, y_train):
        distance = sum((a - b) ** 2 for a, b in zip(x, x_query)) ** 0.5
        distances.append((distance, y))
    neighbors = sorted(distances, key=lambda item: item[0])[:k]
    return Counter(label for _, label in neighbors).most_common(1)[0][0]
```

**Interview note**

Say that scaling matters because KNN relies directly on the distance metric.

---

# Q6: Implement Sigmoid, Tanh, ReLU, LeakyReLU, and Softmax Activation Functions.

**What to say**

When coding activations, the main interview point is to show clean formulas and note numerical stability for softmax.

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

**Interview note**

Softmax should subtract the maximum input before exponentiating to avoid overflow.

---

# Q7: How would you implement k-means clustering?

**What to say**

The core loop is initialize centroids, assign each point to the nearest centroid, recompute centroids, and repeat until convergence or a maximum number of iterations.

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

**Interview note**

Mention k-means++ initialization and sensitivity to scaling and outliers.

---

# Q8: Write code to perform k-fold cross-validation.

**What to say**

I would explain that the key idea is to rotate the validation fold so every example serves as validation once, then average the scores.

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

**Interview note**

If labels are imbalanced, say you would use stratified folds rather than plain slicing.

---

# Q9: How would you use Pandas to load and clean data?

**What to say**

The right answer is not just code. It should show that I know how to inspect types, missingness, duplicates, and obvious data-quality issues before modeling.

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

**Interview note**

Add that real cleaning depends on the domain and should be done in a reproducible pipeline, not ad hoc notebook steps.

---

# Q10: Implement k-nearest neighbors from scratch.

**What to say**

If the interviewer wants a fuller implementation, I would wrap KNN into a class and make the distance function explicit.

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

**Interview note**

Production improvements include KD-trees, ANN search, weighting by distance, and feature scaling.

---

# Q11: Write code to calculate precision and recall.

**What to say**

This is a good place to show careful handling of edge cases such as division by zero.

```python
def precision_recall(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall
```

**Interview note**

If asked what comes next, say F1 is the harmonic mean of precision and recall and is useful when you need one summary number.
