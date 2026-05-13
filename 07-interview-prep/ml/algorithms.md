# ML Algorithms

This file is your "know the players" section.

Not every algorithm needs a dramatic monologue.
But you should know:

- what problem it solves
- why it works
- when it fails
- what you would choose instead

That is how you stop sounding like a glossary and start sounding like an engineer.

---

# 1. Decision Trees

A decision tree keeps splitting the feature space into smaller and smaller regions until the data in each region becomes more pure.

In classification, purity usually means better separation by:

- Gini
- entropy

In regression, it usually means lower variance or MSE inside a node.

**Simple interview answer**

A decision tree recursively chooses the feature split that most reduces impurity and uses the resulting leaves to make predictions.

**Why people like trees**

- easy to explain
- little preprocessing needed
- naturally handle non-linearity
- capture feature interactions

**Why they can be messy**

- high variance
- easy to overfit
- unstable if data changes slightly

---

# 2. Gini vs Entropy

Both measure node impurity. Know the formulas cold.

**Gini impurity:**
$$\text{Gini}(t) = 1 - \sum_{c=1}^{C} p_c^2$$

where $p_c$ is the proportion of class $c$ at node $t$. Range: 0 (pure) to $1 - 1/C$ (maximum impurity).

**Entropy:**
$$H(t) = -\sum_{c=1}^{C} p_c \log_2 p_c$$

Range: 0 (pure) to $\log_2 C$ (maximum impurity).

**Information gain** (what the tree maximizes at each split):
$$\text{IG}(t, \text{split}) = H(t) - \sum_{k \in \{\text{left, right}\}} \frac{|t_k|}{|t|} H(t_k)$$

**Gini vs Entropy in practice:**
- Gini is faster to compute (no logarithm)
- Entropy penalizes impure splits more strongly
- In practice, they choose the same split > 95% of the time
- sklearn default: Gini for classification, MSE for regression

---

# 3. Random Forest

Random Forest is a collection of decision trees trained with extra randomness.

It uses:

- bootstrap sampling of rows (each tree sees ~63% of data, with replacement)
- random feature subsets at splits: at each node, consider only $\sqrt{p}$ features (classification) or $p/3$ (regression) out of $p$ total

Then averages the trees.

Why that helps:

One tree is unstable. Many decorrelated trees are much more reliable.

**Prediction (classification):** majority vote across $T$ trees
**Prediction (regression):** $\hat{y} = \frac{1}{T}\sum_{t=1}^T h_t(x)$

**Out-of-bag (OOB) error:** each tree is tested on the ~37% of samples not used in its bootstrap. Gives a free validation estimate without a separate holdout.

**Key hyperparameters:**
- `n_estimators`: more trees = lower variance, diminishing returns after ~100–200
- `max_features`: controls decorrelation — smaller = more decorrelation, higher bias
- `max_depth`: limits individual tree complexity

**Short answer:** Random Forest reduces variance by averaging many decorrelated decision trees trained on bootstrap samples with random feature subsets at each split.

---

# 4. Bagging vs Boosting

This distinction matters a lot.

## Bagging

Train many models independently.
Then average them.

Main win:

- reduces variance

## Boosting

Train models sequentially.
Each new model focuses on earlier mistakes.

Main win:

- reduces bias

**Easy memory trick**

- bagging = committee vote
- boosting = strict tutor correcting you after every mistake

---

# 5. Gradient Boosting and XGBoost

Gradient boosting builds models stage by stage.

**Gradient boosting objective at step $m$:**
$$F_m(x) = F_{m-1}(x) + \alpha h_m(x)$$

where $h_m$ is a new tree fit to the **negative gradient** (pseudo-residuals) of the loss:
$$r_i = -\left[\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

For MSE loss, pseudo-residuals = actual residuals. For log loss, they are probability errors.

## XGBoost

XGBoost uses a **second-order Taylor expansion** of the loss. At each step:

$$\mathcal{L}^{(m)} \approx \sum_{i=1}^n \left[ g_i f_m(x_i) + \frac{1}{2} h_i f_m(x_i)^2 \right] + \Omega(f_m)$$

where:
- $g_i = \partial_{\hat{y}} \mathcal{L}(y_i, \hat{y}^{(m-1)})$ — first-order gradient
- $h_i = \partial_{\hat{y}}^2 \mathcal{L}(y_i, \hat{y}^{(m-1)})$ — second-order (Hessian)
- $\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$ — regularization (T = leaves, w = leaf weights)

**Optimal leaf weight** for a given tree structure:
$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

**Split gain** (used to find best split):
$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

**Key hyperparameters:** `n_estimators`, `max_depth` (3–6 typical), `learning_rate` (0.01–0.3), `subsample` (row sampling), `colsample_bytree` (column sampling), `lambda`/`alpha` (L2/L1 regularization).

**Short interview answer:** XGBoost is a regularized, optimized gradient boosting implementation that uses second-order gradients for split finding and adds L1/L2 regularization on leaf weights.

---

# 6. Random Forest vs XGBoost

This is one of those classic comparison questions.

## Random Forest

- easier default
- more robust
- harder to overfit
- strong baseline

## XGBoost

- usually stronger when tuned well
- more sensitive to hyperparameters
- more likely to overfit noisy data if unchecked

**Cricket analogy**

Random Forest is the stable middle-order batter.
Rarely spectacular, rarely disastrous.

XGBoost is the aggressive match-winner.
Can take the game away.
Can also get caught at long-on if the shot selection is bad.

---

# 7. Logistic Regression

Despite the name, logistic regression is a classification model.

**Model:**
$$P(y=1 \mid x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^Tx+b)}}$$

**Log-odds (logit):** $\log \frac{P(y=1|x)}{P(y=0|x)} = w^T x + b$ — linear in features.

**Loss:** Binary cross-entropy:
$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \left[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \right]$$

**Gradient:**
$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} X^T (\hat{p} - y)$$

**Regularization in sklearn:** parameter `C` = $1/\lambda$ (inverse regularization strength). Low C = strong regularization. L1 → sparse weights; L2 → small but non-zero weights.

**Decision boundary:** linear in feature space. For non-linear boundaries, add polynomial features or use a neural net.

**When it shines:**
- tabular data with interpretability requirements
- strong, fast baseline
- datasets < 100k rows where gradient boosting may be overkill
- calibrated probabilities needed (logistic regression is naturally well-calibrated)

---

# 8. Linear Regression vs Logistic Regression

This gets asked a lot because people mix them up.

## Linear Regression

Predicts a continuous value.

## Logistic Regression

Predicts probability of a class.

**The subtle but important line**

Logistic regression is linear in the **log-odds**, not in the output probability itself.

That sentence alone makes your answer better than average.

---

# 9. SVM and the Kernel Trick

SVM finds the decision boundary that maximizes the margin between classes.

**Primal (hard-margin):**
$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t. } y_i(w^T x_i + b) \geq 1 \; \forall i$$

**Soft-margin (with slack $\xi_i$):**
$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_i \xi_i \quad \text{s.t. } y_i(w^T x_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0$$

$C$ controls the bias-variance tradeoff: high $C$ = small margin, few violations (risk of overfitting); low $C$ = wide margin, more violations (risk of underfitting).

**The margin width:** $\frac{2}{\|w\|}$ — maximizing margin = minimizing $\|w\|$.

**Support vectors:** only the training points at the margin boundary determine the decision surface.

## Kernel Trick

Replace dot product $x_i^T x_j$ with a kernel function $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$ without explicitly computing $\phi$.

| Kernel | Formula | Use |
| :--- | :--- | :--- |
| Linear | $x_i^T x_j$ | Linearly separable data |
| Polynomial | $(x_i^T x_j + c)^d$ | Polynomial boundaries |
| RBF/Gaussian | $\exp(-\gamma\|x_i - x_j\|^2)$ | Most common default |

**Short answer:** SVM maximizes the margin between classes. The kernel trick implicitly maps data to high-dimensional space where non-linear boundaries become linear, without computing the mapping explicitly.

---

# 10. K-Nearest Neighbors

KNN predicts based on the closest examples in the training data.

For classification:

- majority vote

For regression:

- average nearby values

Why it is useful:

- simple
- intuitive
- strong teaching baseline

Why it is annoying:

- slow at inference
- very sensitive to scale
- weak in high dimensions

**Fashion analogy**

KNN is like saying:

> "Show me the five outfits most similar to this one, then guess the style based on those."

Works fine if your similarity notion is good.
Breaks if your features are messy.

---

# 11. K-Means

K-Means minimizes within-cluster sum of squares (WCSS):

$$\min_{\{C_k\}} \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|^2$$

**Algorithm (Lloyd's):**
1. Initialize $K$ centroids (random or K-Means++ for better convergence)
2. Assign each point to nearest centroid: $c_i = \arg\min_k \|x_i - \mu_k\|^2$
3. Update centroids: $\mu_k = \frac{1}{|C_k|}\sum_{i \in C_k} x_i$
4. Repeat until convergence

**Time complexity:** $O(nKd \cdot \text{iterations})$ where $n$ = points, $K$ = clusters, $d$ = dimensions.

**K-Means++ initialization:** choose each new centroid proportional to $D(x)^2$ (squared distance to nearest existing centroid). Provides $O(\log K)$ approximation guarantee vs. random init.

**Choosing K:**
- **Elbow method:** plot WCSS vs K; look for the "elbow" where adding K gives diminishing returns
- **Silhouette score:** $s = \frac{b - a}{\max(a, b)}$ where $a$ = mean intra-cluster distance, $b$ = mean nearest-cluster distance. Range [-1, 1]; higher is better.

**Limitations:** assumes spherical, similarly-sized clusters; sensitive to outliers and initialization; must specify K in advance.

**Alternatives:** DBSCAN (density-based, finds arbitrary shapes), GMM (soft assignments), Hierarchical (no K needed).

---

# 12. Naive Bayes

**Bayes' theorem applied to classification:**
$$P(y \mid x) = \frac{P(x \mid y) P(y)}{P(x)} \propto P(y) \prod_{j=1}^d P(x_j \mid y)$$

The "naive" assumption: features $x_1, \ldots, x_d$ are **conditionally independent** given the class $y$.

**Prediction:**
$$\hat{y} = \arg\max_y P(y) \prod_{j=1}^d P(x_j \mid y)$$

In practice use log-probabilities to avoid underflow:
$$\hat{y} = \arg\max_y \left[ \log P(y) + \sum_{j=1}^d \log P(x_j \mid y) \right]$$

**Variants:**
| Variant | $P(x_j \mid y)$ model | Best for |
| :--- | :--- | :--- |
| Gaussian NB | $\mathcal{N}(\mu_{jy}, \sigma_{jy}^2)$ | Continuous features |
| Multinomial NB | Categorical counts | Text classification (word counts) |
| Bernoulli NB | Bernoulli (0/1) | Binary features |

**Laplace smoothing** (avoids zero probabilities for unseen words):
$$P(x_j = v \mid y) = \frac{\text{count}(x_j=v, y) + \alpha}{\text{count}(y) + \alpha \cdot |V|}$$

Despite the unrealistic independence assumption, Naive Bayes works well when features are sparse (text), speed matters, or data is limited.

---

# 13. Decision Boundary

The decision boundary is the surface where a model switches from predicting one class to another.

Why it matters:

It tells you how flexible the model is.

- linear models = simple boundaries
- deep models / boosted trees = more complex boundaries

This is a good concept to mention when comparing model capacity.

---

# 14. Dimensionality Reduction

Dimensionality reduction means compressing data into fewer variables while preserving useful structure.

Why do it?

- faster training
- less noise
- easier visualization
- better behavior in high dimension

It is not always about performance.
Sometimes it is about sanity.

---

# 15. PCA

PCA finds orthogonal directions (principal components) of maximum variance via eigendecomposition of the covariance matrix.

**Steps:**
1. Center the data: $\tilde{X} = X - \bar{X}$
2. Compute covariance matrix: $\Sigma = \frac{1}{n-1} \tilde{X}^T \tilde{X}$
3. Eigendecompose: $\Sigma = V \Lambda V^T$ where $V$ = eigenvectors, $\Lambda$ = diagonal eigenvalues
4. Sort by eigenvalue (variance) descending
5. Project: $Z = \tilde{X} V_k$ where $V_k$ = top $k$ eigenvectors

**Equivalently** via SVD: $\tilde{X} = U \Sigma V^T$; principal components are right singular vectors $V$.

**Explained variance ratio:**
$$\text{EVR}_k = \frac{\lambda_k}{\sum_j \lambda_j}$$

Choose $k$ to retain 95% cumulative explained variance for most applications.

**Important caveat:** PCA is unsupervised — directions of highest variance are not necessarily most predictive for the target. LDA (Linear Discriminant Analysis) is the supervised alternative.

**When to use:** dimensionality reduction before KNN/SVM, denoising, visualization (t-SNE is better for visualization), decorrelating features before linear models.

**Limitations:** linear only; sensitive to scale (must standardize first); not interpretable components.

---

# 16. Gradient Descent and Variants

Even classical algorithms get wrapped around optimization eventually.

Important variants:

- batch gradient descent
- SGD
- mini-batch SGD
- momentum
- Adam

The practical lesson is:

Optimization choice changes:

- convergence speed
- stability
- sometimes generalization

---

# Quick Thought Experiment

You have medium-sized structured tabular data with:

- missing values
- mixed feature types
- not much feature engineering time

What is your first strong baseline?

Answer:

Usually a tree ensemble, especially boosted trees.

Not because it is trendy.
Because it is often the right hammer.

---

# Mini Pop Quiz

Which model is more likely to need feature scaling:

- KNN
- Random Forest

Answer:

KNN.

Because distance-based models care deeply about feature scale.
Tree splits usually do not.
