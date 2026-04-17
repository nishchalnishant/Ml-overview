# Coding

This hub contains minimalist, from-scratch implementations of core ML algorithms and metrics. In interviews, these serve as "whiteboard" baselines to demonstrate your understanding of the underlying math and engineering logic.

---

# Q1: Implement Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## 1. 🔹 Direct Answer
**MSE** is the average of squared differences between predictions and targets: $MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$.
**MAE** is the average of absolute differences: $MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|$.

## 2. 🔹 Intuition
MSE penalizes outliers heavily (squaring large errors), whereas MAE treats all errors linearly. MSE is mathematically "nicer" (differentiable everywhere), making it standard for optimization.

## 3. 🔹 Practical Perspective
- **Use MSE:** When you want to minimize large deviations and have no outliers.
- **Use MAE:** When the dataset is noisy or contains extreme outliers you don't want to overfit to.

## 4. 🔹 Code Snippet
```python
def mse(y_true, y_pred):
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

def mae(y_true, y_pred):
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)
```

## 5. 🔹 Difficulty Tag: 🟢 Easy

---

# Q2: Implement Linear Regression using Gradient Descent.

## 1. 🔹 Direct Answer
Linear Regression finds the line $y = wx + b$ that minimizes MSE. While it has a closed-form solution (Normal Equation), Gradient Descent is the scalable approach used in deep learning.

## 2. 🔹 Intuition
Imagine a ball rolling down a hill; the "hill" is the loss surface, and the gradient tells us the direction of steepest descent.

## 3. 🔹 Deep Dive
The gradients for MSE loss $(y - (wx+b))^2$ are:
- $\frac{\partial L}{\partial w} = \frac{2}{n} \sum (wx+b - y) \cdot x$
- $\frac{\partial L}{\partial b} = \frac{2}{n} \sum (wx+b - y)$

## 4. 🔹 Code Snippet
```python
class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr, self.epochs = lr, epochs
        self.w, self.b = 0.0, 0.0

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.epochs):
            y_pred = [self.w * x + self.b for x in X]
            dw = sum((yp - yt) * x for x, yt, yp in zip(X, y, y_pred)) * 2 / n
            db = sum(yp - yt for yt, yp in zip(y, y_pred)) * 2 / n
            self.w -= self.lr * dw
            self.b -= self.lr * db
```

## 5. 🔹 Difficulty Tag: 🟡 Medium

---

# Q3: Implement Logistic Regression (Binary Classification).

## 1. 🔹 Direct Answer
Logistic Regression maps linear scores to probabilities using the **Sigmoid** function: $P(y=1) = \sigma(wx+b)$. It is trained using **Cross-Entropy Loss**.

## 2. 🔹 Intuition
We are "squashing" the infinite line of linear regression into a $[0, 1]$ probability range.

## 3. 🔹 Code Snippet
```python
import math

class LogisticRegressionGD:
    def _sigmoid(self, z):
        return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))

    def fit(self, X, y, lr=0.1, epochs=1000):
        self.w, self.b = 0.0, 0.0
        n = len(X)
        for _ in range(epochs):
            probs = [self._sigmoid(self.w * x + self.b) for x in X]
            # Gradient of Binary Cross Entropy
            dw = sum((p - yt) * x for x, yt, p in zip(X, y, probs)) / n
            db = sum(p - yt for yt, p in zip(y, probs)) / n
            self.w -= lr * dw
            self.b -= lr * db
```

## 4. 🔹 Difficulty Tag: 🟡 Medium

---

# Q4: Implement K-Means Clustering.

## 1. 🔹 Direct Answer
K-Means is an unsupervised algorithm that partitions data into $K$ clusters by iteratively assigning points to the nearest centroid and recomputing centroids.

## 2. 🔹 Intuition
"Birds of a feather flock together." We find the average positions (centroids) of groups and refine them until they stop moving.

## 3. 🔹 Practical Perspective
- **Pros:** Simple, scales well to large datasets.
- **Cons:** Must specify $K$, sensitive to initialization and outliers.

## 4. 🔹 Code Snippet (Simplified)
```python
def kmeans(points, k, iters=100):
    centroids = points[:k]
    for _ in range(iters):
        clusters = [[] for _ in range(k)]
        for p in points:
            dist = [sum((a-b)**2 for a,b in zip(p, c)) for c in centroids]
            clusters[dist.index(min(dist))].append(p)
        
        new_centroids = [[sum(p[d] for p in cl)/len(cl) for d in range(len(p))] if cl else centroids[i] 
                         for i, cl in enumerate(clusters)]
        if new_centroids == centroids: break
        centroids = new_centroids
    return centroids
```

## 5. 🔹 Difficulty Tag: 🟡 Medium

---

# Q5: Implement Precision, Recall, and F1-Score.

## 1. 🔹 Direct Answer
- **Precision:** $\frac{TP}{TP+FP}$ (Of those we called positive, how many were?)
- **Recall:** $\frac{TP}{TP+FN}$ (Of the actual positives, how many did we catch?)
- **F1:** $\frac{2 \cdot P \cdot R}{P+R}$ (Harmonic mean of the two).

## 2. 🔹 Code Snippet
```python
def metrics(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1
```

## 3. 🔹 Difficulty Tag: 🟢 Easy
