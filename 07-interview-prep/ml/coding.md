# Coding

## What This File Is For

ML coding interviews test two things simultaneously: whether you can write correct code and whether you understand what you are writing well enough to reason about it under pressure. The structure for each topic:

1. What the interviewer is actually testing — the underlying competency
2. The reasoning structure — why first-principles thinkers approach it this way
3. The pattern in action — clean, working code with the key design decisions made explicit
4. Common traps — where people go wrong and why

---

# 1. MSE and MAE

## What the interviewer is actually testing

Whether you understand the loss function's geometric interpretation and can connect its mathematical form to its behavior on outliers. This is the first question in many ML coding screens — the expected answer is clean code plus one sentence of insight.

## The reasoning structure

**MSE** squares the error: `(y - ŷ)²`. Squaring does two things: makes all errors positive, and amplifies large errors relative to small ones. The gradient of MSE is `2(ŷ - y)`, which is proportional to the error. A prediction that is 10 units off contributes 100× more to the loss than a prediction that is 1 unit off, and its gradient is 10× larger. MSE is sensitive to outliers.

**MAE** takes the absolute error: `|y - ŷ|`. Every unit of error contributes linearly. The gradient is ±1 regardless of error magnitude. MAE is robust to outliers.

**When to use each:** MSE when large errors are especially costly and you want the model to focus on reducing them (regression with consequential mistakes). MAE when the training data contains outliers that should not disproportionately influence the model.

## The pattern in action

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

**What to say while writing:** "I am adding input validation first — mismatched lengths are a silent bug without this check. In a production implementation I would use NumPy for vectorized computation, but the loop version shows the logic clearly."

## Common traps

**Not handling edge cases.** Empty inputs, mismatched lengths, and NaN values are the first things an interviewer checks. Write the guard clauses before the computation.

**Stating the formula without explaining the gradient.** If asked to compare MSE and MAE, the insight is in the gradient: MSE's gradient scales with error magnitude (sensitive to outliers), MAE's gradient is constant (robust). The square in MSE is the source of the outlier sensitivity.

---

# 2. Linear Regression with Gradient Descent

## What the interviewer is actually testing

Whether you can implement the training loop of a parametric model from scratch, including the gradient computation. The closed-form solution exists for linear regression, but gradient descent shows the learning logic that generalizes to non-convex problems.

## The reasoning structure

**The gradient.** MSE loss L = (1/n) Σ (ŷ_i - y_i)². 

For ŷ = wx + b:
- `∂L/∂w = (2/n) Σ (ŷ_i - y_i) · x_i`
- `∂L/∂b = (2/n) Σ (ŷ_i - y_i)`

The gradient points in the direction of steepest ascent. We subtract the gradient (scaled by learning rate) to descend.

**Key parameters:**
- Learning rate too high → overshoots the minimum, diverges
- Learning rate too low → converges correctly but slowly
- Number of epochs → stopping criterion; in practice, use early stopping on a validation loss

## The pattern in action

```python
class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0

    def fit(self, X, y):
        n = len(X)
        for epoch in range(self.epochs):
            y_pred = [self.w * x + self.b for x in X]
            dw = sum((yp - yt) * x for x, yt, yp in zip(X, y, y_pred)) * 2 / n
            db = sum(yp - yt for yt, yp in zip(y, y_pred)) * 2 / n
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return [self.w * x + self.b for x in X]
```

**What to say while writing:** "This is gradient descent on MSE loss. The gradient for w is the average of each example's error times its feature value — the chain rule through ŷ = wx + b. I would add convergence checking (stop when ‖∇L‖ < ε) in a fuller implementation. For multiple features, I would vectorize: `w -= lr * (2/n) * X.T @ (X @ w - y)`."

## Common traps

**Getting the gradient sign wrong.** We subtract the gradient: `w -= lr * dw`. Subtracting means we move in the direction that decreases the loss. A sign error produces divergence.

**Forgetting the 1/n normalization.** Without normalizing by n, the gradient magnitude scales with dataset size, making the effective learning rate dependent on n. Always normalize by the batch size.

**Not mentioning the closed-form alternative.** `w = (X^T X)^{-1} X^T y`. Mention that this exists and that gradient descent is used when the matrix is too large to invert or when the problem is non-convex.

---

# 3. Logistic Regression

## What the interviewer is actually testing

Whether you understand why the sigmoid function is the natural choice for binary classification (it bounds the output between 0 and 1 and produces calibrated probabilities from an assumed Bernoulli model), and whether you can implement numerically stable sigmoid.

## The reasoning structure

**Why sigmoid.** Logistic regression models log-odds: `log(p/(1-p)) = wx + b`. Solving for p gives `p = σ(wx + b) = 1 / (1 + e^{-z})`. The sigmoid function converts the linear score to a probability.

**The gradient of BCE with sigmoid cancels beautifully:** `∂L/∂w = (1/n) Σ (σ(z_i) - y_i) · x_i`. The sigmoid derivative `σ(z)(1-σ(z))` and the BCE gradient cancel, leaving a simple residual. This is why logistic regression is well-behaved numerically.

**Numerical stability.** For large negative z: `exp(-z)` overflows. For large positive z: `exp(-z)` underflows (but is effectively 0, so the formula works). The safe implementation branches on the sign of z.

## The pattern in action

```python
import math


class LogisticRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0

    def _sigmoid(self, z):
        # Numerically stable: avoid overflow for large negative z
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1 + ez)

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.epochs):
            probs = [self._sigmoid(self.w * x + self.b) for x in X]
            # Gradient of BCE loss with sigmoid: (p - y) (the cancellation is exact)
            dw = sum((p - yt) * x for x, yt, p in zip(X, y, probs)) / n
            db = sum(p - yt for yt, p in zip(y, probs)) / n
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return [self._sigmoid(self.w * x + self.b) for x in X]

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]
```

**What to say while writing:** "I am exposing `predict_proba` separately from `predict` because the probability output is useful for threshold tuning and calibration analysis — you should not hardcode 0.5 in production. The sigmoid implementation branches on z ≥ 0 to avoid overflow."

## Common traps

**Using the naive sigmoid for all inputs.** `1 / (1 + exp(-z))` for z = -1000 computes `exp(1000)`, which overflows to inf. Implement the stable version.

**Hardcoding threshold at 0.5.** In production, the threshold should be set based on the precision-recall tradeoff for the specific business cost structure. Expose `predict_proba` and let the caller decide the threshold.

---

# 4. Softmax and Cross-Entropy Loss

## What the interviewer is actually testing

Whether you can implement numerically stable softmax and connect it to the cross-entropy loss, and whether you know why computing `log(softmax(x))` directly is unstable.

## The reasoning structure

**The stability problem.** `softmax(x)_k = exp(x_k) / Σ_j exp(x_j)`. For large x_k, `exp(x_k)` overflows. For all x_k small and negative, `exp(x_k)` underflows to 0, producing 0/0.

**The fix.** Shift x by its max before exponentiation: `softmax(x - max(x))`. The softmax output is invariant to constant shifts: `exp(x_k - c) / Σ_j exp(x_j - c) = exp(x_k) / Σ_j exp(x_j)`. After shifting by max(x), all inputs are ≤ 0, so no overflow.

**Log-softmax is more stable than log(softmax).** `log_softmax(x)_k = x_k - max(x) - log(Σ_j exp(x_j - max(x)))`. Compute the log directly instead of computing softmax first and then logging. This is what PyTorch's `log_softmax` does.

## The pattern in action

```python
import math


def softmax(logits):
    """Numerically stable softmax."""
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def log_softmax(logits):
    """Numerically stable log-softmax (more stable than log(softmax))."""
    max_logit = max(logits)
    shifted = [x - max_logit for x in logits]
    log_sum_exp = math.log(sum(math.exp(x) for x in shifted))
    return [x - log_sum_exp for x in shifted]


def cross_entropy_loss(logits, true_class):
    """
    Cross-entropy loss for a single example.
    logits: list of raw logit values (unnormalized)
    true_class: integer index of the correct class
    Returns: scalar loss
    """
    log_probs = log_softmax(logits)
    return -log_probs[true_class]


def batch_cross_entropy(logits_batch, labels):
    """Average cross-entropy over a batch."""
    n = len(labels)
    return sum(cross_entropy_loss(logits, y) for logits, y in zip(logits_batch, labels)) / n
```

**What to say while writing:** "I compute log-softmax rather than log(softmax(x)) because the log and the division cancel in a numerically stable way. This is why PyTorch's `CrossEntropyLoss` takes raw logits rather than probabilities — it applies log-softmax internally."

## Common traps

**`log(softmax(x))` for numerical precision.** This computes small probabilities first, then logs them. Numbers like 1e-40 are representable in float64 but log(1e-40) has poor precision. Computing log directly from logits avoids this.

**Forgetting the negative sign in cross-entropy.** Cross-entropy is `-Σ y_k log(p_k)`. The negative sign makes it a loss (something to minimize). Omitting the negative gives a value that needs maximization, not minimization.

---

# 5. Precision, Recall, and F1

## What the interviewer is actually testing

Whether you understand the confusion matrix well enough to compute its derived metrics correctly, and whether you can explain the tradeoff that the F1 score encodes.

## The reasoning structure

**The confusion matrix.** Four cells: TP (correctly predicted positive), FP (incorrectly predicted positive), FN (missed positive), TN (correctly predicted negative).

**Precision:** of all predicted positives, how many were actually positive? TP / (TP + FP). Measures the quality of positive predictions.

**Recall:** of all actual positives, how many did we predict positive? TP / (TP + FN). Measures coverage of the positive class.

**F1:** harmonic mean of precision and recall. `2PR / (P + R)`. The harmonic mean is used because it penalizes extreme imbalance between precision and recall more than the arithmetic mean. A model with precision=1.0 and recall=0.01 has arithmetic mean=0.505 but F1=0.02.

**When to use each:** precision matters when false positives are costly (spam detection — falsely flagging legitimate email). Recall matters when false negatives are costly (cancer screening — missing a real cancer). F1 when both matter equally.

## The pattern in action

```python
def precision_recall_f1(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def confusion_matrix_stats(y_true, y_pred):
    """Returns TP, FP, FN, TN for further analysis."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    return tp, fp, fn, tn
```

**What to say while writing:** "I handle division by zero in all three metrics — precision and recall are undefined when the denominator is zero, and returning 0.0 is the conventional choice. I would normally also compute ROC-AUC or precision-recall AUC for full evaluation."

## Common traps

**Division by zero.** If there are no predicted positives (TP + FP = 0), precision is undefined. If there are no actual positives (TP + FN = 0), recall is undefined. Both cases occur in practice and must be handled.

**F1 at the wrong threshold.** F1 depends on the classification threshold. Reporting a single F1 without specifying the threshold is incomplete. Use precision-recall AUC for threshold-independent evaluation.

---

# 6. Activation Functions

## What the interviewer is actually testing

Whether you can implement the functions, explain their gradient properties, and reason about when to use each.

## The reasoning structure

**Why activation functions exist.** Without non-linearity, a deep network is just a linear transformation (composition of linear functions is linear). Activation functions introduce non-linearity, enabling the network to learn complex mappings.

**Key properties:**

**Sigmoid** `σ(x) = 1/(1+e^{-x})`: output in (0,1), derivative `σ(x)(1-σ(x))` ≤ 0.25. Saturates (gradient ≈ 0) for large |x|. Causes vanishing gradients in deep networks.

**Tanh** `tanh(x)`: output in (-1,1), derivative `1 - tanh²(x)` ≤ 1. Zero-centered (unlike sigmoid). Still saturates for large |x|.

**ReLU** `max(0,x)`: derivative is 1 for x > 0, 0 for x ≤ 0. No saturation for positive inputs. Sparse activation. Dead ReLU problem: neurons that receive negative inputs stop contributing gradient.

**LeakyReLU** `max(αx, x)`: derivative is 1 for x > 0, α for x ≤ 0. Prevents dead ReLUs by allowing a small gradient for negative inputs.

**Softmax** for multi-class output: converts logits to a probability distribution (sums to 1). Not used in hidden layers.

## The pattern in action

```python
import math


def sigmoid(x):
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1 + ex)


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return max(0.0, x)


def relu_derivative(x):
    return 1.0 if x > 0 else 0.0


def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x


def leaky_relu_derivative(x, alpha=0.01):
    return 1.0 if x > 0 else alpha


def softmax(xs):
    """Numerically stable softmax: subtract max before exponentiation."""
    max_x = max(xs)
    exps = [math.exp(x - max_x) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]
```

**What to say while writing:** "I subtract max(x) in softmax for numerical stability — softmax is invariant to constant shifts, so this doesn't change the output but prevents overflow. The sigmoid derivative has a maximum of 0.25 at x=0, which is why stacking sigmoid layers causes vanishing gradients: 0.25^20 ≈ 10^{-12}."

## Common traps

**Forgetting sigmoid's gradient bound.** The maximum gradient of sigmoid is 0.25 (not 1). This fact directly explains vanishing gradients in deep sigmoid networks.

**Using non-stable softmax.** Without subtracting max, softmax overflows for logits > 709 (where exp(709) ≈ float64 max). Always subtract max first.

---

# 7. K-Means

## What the interviewer is actually testing

Whether you can implement the E-M (Expectation-Maximization) structure of K-Means and handle the edge case of empty clusters.

## The reasoning structure

**The algorithm alternates between two steps:**
- E-step: assign each point to the nearest centroid
- M-step: update each centroid to the mean of its assigned points

Convergence: the assignments stop changing. In practice, K-Means always converges but may converge to a local minimum.

**The empty cluster problem.** If a centroid is initialized far from any points, it may never receive an assignment. The update step would divide by zero. The standard fix: keep the old centroid when a cluster is empty.

**Initialization matters.** Random initialization from the data points (as below) can produce poor solutions. K-Means++ initializes centroids to be spread out, reducing the probability of poor local minima.

**Complexity.** O(n × k × d × iterations): n points, k centroids, d dimensions, number of iterations until convergence.

## The pattern in action

```python
def euclidean_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def kmeans(points, k, max_iters=100):
    """
    Simple K-Means. In production: use k-means++ initialization
    and scale features before calling.
    """
    # Initialize with first k points (naive; k-means++ is better)
    centroids = [list(p) for p in points[:k]]

    for _ in range(max_iters):
        # E-step: assign each point to nearest centroid
        clusters = [[] for _ in range(k)]
        for point in points:
            nearest = min(
                range(k),
                key=lambda i: euclidean_distance(point, centroids[i])
            )
            clusters[nearest].append(point)

        # M-step: update centroids to mean of assigned points
        new_centroids = []
        for i, (cluster, old_centroid) in enumerate(zip(clusters, centroids)):
            if not cluster:
                # Empty cluster: keep old centroid
                new_centroids.append(old_centroid)
            else:
                d = len(cluster[0])
                mean = [
                    sum(point[dim] for point in cluster) / len(cluster)
                    for dim in range(d)
                ]
                new_centroids.append(mean)

        # Check convergence
        if new_centroids == centroids:
            break
        centroids = new_centroids

    return centroids, clusters
```

**What to say while writing:** "I return both centroids and the final cluster assignments — just returning centroids loses information. The empty cluster case is important: a naive implementation would divide by zero. In production I would use K-Means++ initialization and feature normalization."

## Common traps

**Not handling empty clusters.** This is the specific edge case interviewers look for. If you write the M-step without checking, your code will throw a ZeroDivisionError.

**Comparing centroids with == for convergence.** Floating-point centroids may never be exactly equal across iterations even when visually converged. In production, check whether any point's assignment changed in this iteration (assignment convergence), or check whether centroid movement is below a tolerance.

---

# 8. K-Fold Cross Validation

## What the interviewer is actually testing

Whether you understand why cross-validation gives a better performance estimate than a single train-test split, and whether you can implement the splitting logic correctly.

## The reasoning structure

**The problem with a single split.** A model's performance on one validation set has high variance — it depends on which specific examples ended up in the validation set. If you are unlucky (the validation set is unrepresentative), you get a misleading estimate.

**K-fold solution.** Rotate through k equal-sized folds. Each fold serves as validation once. Average performance across folds gives a lower-variance estimate that uses all data for evaluation.

**The cost.** K-fold requires training k models. For k=5, 5× training cost. For expensive models, use larger fold sizes (k=3) or a single temporal holdout.

**Stratified k-fold.** When classes are imbalanced, random folding may produce folds with very different class ratios. Stratified folding ensures each fold has approximately the same class distribution.

## The pattern in action

```python
def k_fold_split(data, k, shuffle=False, seed=None):
    """
    Returns k (train, validation) splits.
    Each example appears in exactly one validation fold.
    """
    if shuffle:
        import random
        rng = random.Random(seed)
        data = list(data)
        rng.shuffle(data)

    n = len(data)
    fold_size = n // k
    remainder = n % k

    folds = []
    start = 0
    for i in range(k):
        # Distribute remainder examples one per fold
        end = start + fold_size + (1 if i < remainder else 0)
        val = data[start:end]
        train = data[:start] + data[end:]
        folds.append((train, val))
        start = end

    return folds


def cross_validate(model_class, data, labels, k=5):
    """Simple k-fold CV returning per-fold validation accuracies."""
    paired = list(zip(data, labels))
    folds = k_fold_split(paired, k, shuffle=True, seed=42)

    scores = []
    for train_pairs, val_pairs in folds:
        X_train, y_train = zip(*train_pairs)
        X_val, y_val = zip(*val_pairs)

        model = model_class()
        model.fit(list(X_train), list(y_train))
        y_pred = model.predict(list(X_val))

        accuracy = sum(p == t for p, t in zip(y_pred, y_val)) / len(y_val)
        scores.append(accuracy)

    return scores
```

**What to say while writing:** "I distribute remainder examples one per fold to handle datasets not evenly divisible by k. For time-series data, I would use forward-chaining validation instead of shuffled k-fold — never shuffle temporal data before splitting."

## Common traps

**Shuffling time-series data before k-fold.** This is temporal leakage. For time-series, use walk-forward validation (train on past, evaluate on future) not k-fold.

**Fitting preprocessing on the full dataset before k-fold.** If you standardize features using the full dataset before folding, the validation set's statistics leak into the training preprocessing. Fit preprocessing inside each fold, on training data only.

---

# 9. Backpropagation (Simple Network)

## What the interviewer is actually testing

Whether you understand the chain rule applied to a computation graph, and whether you can track gradient flow through multiple operations manually.

## The reasoning structure

**Backpropagation is the chain rule applied systematically.** For a computation `L(f(g(x)))`:
```
∂L/∂x = ∂L/∂f · ∂f/∂g · ∂g/∂x
```

In a neural network, each layer's gradient is computed from the gradient of the next layer (from the output) multiplied by the local gradient of the operation.

**A single-layer network: L = MSE(σ(Wx + b), y)**

Forward pass:
1. z = Wx + b  (linear)
2. a = σ(z)     (sigmoid)
3. L = (a - y)²  (MSE, single example for simplicity)

Backward pass (chain rule):
1. ∂L/∂a = 2(a - y)
2. ∂L/∂z = ∂L/∂a · σ'(z) = 2(a - y) · a(1-a)
3. ∂L/∂W = ∂L/∂z · x^T
4. ∂L/∂b = ∂L/∂z

## The pattern in action

```python
import math


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1 + ez)


class TwoLayerNet:
    """
    Single hidden layer: input → sigmoid → output → MSE
    Single feature for clarity. Extend to vectors with numpy.
    """
    def __init__(self, lr=0.01):
        self.lr = lr
        self.w1 = 0.1   # input → hidden weight
        self.b1 = 0.0   # hidden bias
        self.w2 = 0.1   # hidden → output weight
        self.b2 = 0.0   # output bias

    def forward(self, x):
        """Forward pass. Store intermediates for backward."""
        self.x = x
        self.z1 = self.w1 * x + self.b1
        self.a1 = sigmoid(self.z1)          # hidden activation
        self.z2 = self.w2 * self.a1 + self.b2
        return self.z2                       # output (no activation for regression)

    def backward(self, y_true):
        """Backward pass via chain rule."""
        y_pred = self.z2

        # Output layer gradients (MSE loss: L = (y_pred - y_true)^2)
        dL_dz2 = 2 * (y_pred - y_true)
        dL_dw2 = dL_dz2 * self.a1
        dL_db2 = dL_dz2

        # Hidden layer gradients (chain through output weights and sigmoid)
        dL_da1 = dL_dz2 * self.w2
        dL_dz1 = dL_da1 * self.a1 * (1 - self.a1)  # sigmoid derivative
        dL_dw1 = dL_dz1 * self.x
        dL_db1 = dL_dz1

        # Update parameters
        self.w2 -= self.lr * dL_dw2
        self.b2 -= self.lr * dL_db2
        self.w1 -= self.lr * dL_dw1
        self.b1 -= self.lr * dL_db1

    def train_step(self, x, y):
        self.forward(x)
        self.backward(y)
```

**What to say while writing:** "I store the intermediate values in the forward pass so the backward pass can use them. This is the standard pattern: in PyTorch, the autograd tape stores the same intermediate values. The gradient of sigmoid is a(1-a), where a is the already-computed sigmoid output — no need to recompute."

## Common traps

**Not storing forward-pass intermediates.** The backward pass needs the values computed during the forward pass. In the example, `self.a1` must be stored to compute the sigmoid derivative.

**Forgetting to chain through all layers.** The gradient of w1 requires the chain: ∂L/∂w1 = ∂L/∂z2 · w2 · σ'(z1) · x. Every intermediate must be chained through.

---

# 10. Pandas Data Cleaning

## What the interviewer is actually testing

Whether you treat data cleaning as a systematic engineering task — reproducible, tested, documented — rather than ad-hoc notebook commands.

## The reasoning structure

**Data cleaning decisions are not neutral.** How you impute nulls, handle outliers, and encode categoricals affects the model. The choices should be:
- Made explicitly (not by default)
- Fit on training data only (not on the full dataset)
- Reproducible (same transformations at training and serving time)

**Common issues in order of priority:**
1. Schema violations: unexpected types, column name mismatches
2. Missing target labels: rows with null labels must be dropped or imputed carefully
3. Duplicate rows: may indicate logging bugs or data corruption
4. Missing features: impute with training-time statistics, not full-dataset statistics

## The pattern in action

```python
import pandas as pd


def load_and_clean(path, target_col="label"):
    """
    Loads and cleans a CSV dataset.
    
    In production: extract these transforms into a fitted sklearn Pipeline
    so that imputation statistics are fit on training data and applied
    consistently to validation/test/serving data.
    """
    df = pd.read_csv(path)

    # 1. Remove exact duplicates
    df = df.drop_duplicates()

    # 2. Drop rows with missing target (cannot train on these)
    df = df.dropna(subset=[target_col])

    # 3. Separate target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 4. Impute missing values in features
    # IMPORTANT: in production, fit these statistics on training data only
    numeric_cols = X.select_dtypes(include="number").columns
    categorical_cols = X.select_dtypes(exclude="number").columns

    for col in numeric_cols:
        median = X[col].median()
        X[col] = X[col].fillna(median)

    for col in categorical_cols:
        X[col] = X[col].fillna("__missing__")

    return X, y


def split_and_clean(df, target_col, test_size=0.2, seed=42):
    """
    Correct ordering: split first, then fit imputation on train only.
    """
    train_df = df.sample(frac=1 - test_size, random_state=seed)
    test_df = df.drop(train_df.index)

    # Fit imputation statistics on training data
    numeric_cols = train_df.select_dtypes(include="number").columns
    imputation_stats = {col: train_df[col].median() for col in numeric_cols}

    # Apply to both splits using training statistics
    for col, stat in imputation_stats.items():
        train_df[col] = train_df[col].fillna(stat)
        test_df[col] = test_df[col].fillna(stat)

    return train_df, test_df
```

**What to say while writing:** "The critical correctness point is that imputation statistics must be fit on training data only, then applied to test and serving data. Fitting on the full dataset is a subtle form of data leakage — the test median leaks into the training pipeline. In production this logic would be in a fitted sklearn `Pipeline` with `SimpleImputer`, not standalone code."

## Common traps

**Fitting imputation on the full dataset.** Computing the median across train and test, then using that median for imputation, allows test data statistics to influence training. Split first, fit imputation on training only.

**Using `fillna` with `inplace=True` carelessly.** `inplace=True` modifies the original DataFrame. This is correct in some contexts but dangerous in others (e.g., if the same DataFrame is referenced elsewhere in the notebook). Default to returning a new DataFrame.

---

# 11. How to Narrate While Coding

The narration is part of the answer. Coding silently is suboptimal — the interviewer cannot assess your reasoning if they only see the output.

**What to say at each stage:**

Before writing: "I am going to start with the simplest correct version. I will handle edge cases as I go and note where I would optimize in production."

While writing the core logic: "The gradient here is [formula]. I am normalizing by n to make the learning rate invariant to dataset size."

When adding a specific line: "I am adding a null check here because [situation] will produce a ZeroDivisionError without it."

After the basic version works: "The time complexity is O(n × k) for each K-Means iteration. I would vectorize this with NumPy in production to get a 50–100× speedup."

**Key phrases:**
- "I am choosing clarity over micro-optimization here."
- "I would extract this into a separate function in a real codebase."
- "This is O(n²) — for production I would use [faster approach] instead."
- "I am adding this edge case because [specific input] would otherwise fail."

**What not to say:**
- Nothing (silence makes the interviewer uncertain about your reasoning)
- "I think this is right" without explaining why
- Overly long explanations before writing a line

The goal is to sound like someone who knows what they are building, not someone who is figuring it out as they go.

---

# Quick Diagnostics

**If you are stuck during coding:**

Narrate what you know and what you are uncertain about: "I know the gradient of MSE is 2(ŷ - y), and the chain through the linear layer gives the gradient for w as the error times the input feature. I am checking the sign convention before writing the update."

This is better than silence and shows the interviewer your reasoning process.

**If you finish early:**

Add one of: (1) input validation for edge cases, (2) a comment about the production version, (3) complexity analysis. These demonstrate the production mindset that distinguishes senior from junior candidates.

**If asked to test your code:**

Write a small test case with a known answer. For MSE: `mse([3, 4], [2, 4])` should return `0.5`. For K-Means: use 4 clearly separated clusters and verify that centroids converge to their centers. Testing with a known correct answer signals systematic quality thinking.
