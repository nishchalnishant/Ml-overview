# Unsupervised Learning Mastery (Deep-Dive)

This track explores the challenge of extracting structure from unlabeled data. It covers clustering, dimensionality reduction, and anomaly detection.

---

# 1. Clustering Paradigms

## Comparison Table

| Paradigm | Key Algorithm | Best For | Weakness |
| :--- | :--- | :--- | :--- |
| **Centroid-based** | K-Means | Spherical, even clusters | Must pick K; sensitive to outliers |
| **Density-based** | DBSCAN | Arbitrary shapes, noise | Struggles with varying densities |
| **Probabilistic** | GMM | Soft clustering, elliptical shapes | Sensitive to initialization |
| **Hierarchical** | Agglomerative | Dendrograms, small data | $O(N^3)$ complexity |

---

# 2. K-Means Clustering

**Objective:** Minimize within-cluster sum of squares (inertia):

$$J = \sum_{j=1}^{K} \sum_{x \in C_j} \|x - \mu_j\|^2$$

**Algorithm:**
1. Initialize $K$ centroids
2. Assign each point to nearest centroid
3. Recompute centroids as cluster means
4. Repeat until convergence

**Convergence is guaranteed** (inertia decreases monotonically), but the solution may be a local minimum.

### Choosing K

1. **Elbow Method:** plot inertia vs $K$; pick the "elbow" where reduction slows.
2. **Silhouette Score:** measures cohesion (within-cluster distance) vs separation (nearest-cluster distance):

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Range $[-1, 1]$. Higher is better. Average over all points.

### K-Means++

Standard K-Means is sensitive to random initialization. K-Means++ fixes this:
1. Choose first centroid uniformly at random
2. Choose next centroid with probability $\propto d(x, \text{nearest centroid})^2$
3. Repeat until $K$ centroids are chosen

This leads to $O(\log K)$ approximation guarantee and much better practical results.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, labels))

best_k = range(2, 11)[silhouette_scores.index(max(silhouette_scores))]
```

---

# 3. DBSCAN

**Parameters:**
- `ε` (eps): neighborhood radius
- `MinPts` (min_samples): minimum points to form a dense region

**Point types:**
- **Core point:** at least `MinPts` points within `ε`
- **Border point:** within `ε` of a core point but fewer than `MinPts` neighbors
- **Noise:** neither core nor border

**Algorithm:** expand clusters by connecting core points that are within `ε` of each other.

**Advantages:** no need to specify $K$; handles arbitrary shapes; robust to outliers (labeled $-1$).

**Weakness:** struggles with varying densities. HDBSCAN fixes this with hierarchical density estimation.

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
```

---

# 4. Gaussian Mixture Models (GMM)

**Model:** data generated from $K$ Gaussian components:

$$P(x) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

where $\pi_k$ are mixing weights ($\sum \pi_k = 1$).

**Fitting via EM algorithm:**
- **E-step:** compute soft assignments $r_{ik} = P(z_k | x_i)$ (responsibility of component $k$ for point $i$)
- **M-step:** update $\mu_k$, $\Sigma_k$, $\pi_k$ using weighted MLE

**Advantages over K-Means:** soft assignments (probabilistic cluster membership), can model elliptical clusters via $\Sigma_k$.

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)   # soft assignments
```

---

# 5. Dimensionality Reduction

## PCA (Principal Component Analysis)

**Goal:** find directions of maximum variance (principal components).

**Steps:**
1. Center data: $X \leftarrow X - \bar{X}$
2. Compute covariance matrix: $\Sigma = \frac{1}{m} X^T X$
3. Eigen-decompose: $\Sigma v = \lambda v$
4. Project onto top-$k$ eigenvectors

**Explained variance ratio:** $\frac{\lambda_i}{\sum_j \lambda_j}$ — choose $k$ where cumulative ratio exceeds threshold (e.g., 95%).

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, svd_solver='full')  # keep 95% variance
X_reduced = pca.fit_transform(X_train)
print(f"Reduced from {X_train.shape[1]} to {X_reduced.shape[1]} features")
print(f"Explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")
```

## t-SNE

**Goal:** preserve local similarity structure for visualization.

**Idea:** model pairwise similarities as probabilities in high-d and low-d spaces; minimize KL divergence between them.

**Limitations:**
- $O(N^2)$ complexity (slow for $N > 10$k without approximations)
- Stochastic — different runs give different layouts
- Distances between clusters are **not** meaningful
- Use only for 2D/3D visualization, not as features for downstream tasks

## UMAP

**Goal:** preserve both local and global structure via topological manifold learning.

**Advantages over t-SNE:**
- Much faster: $O(N \log N)$
- Preserves more global structure
- Deterministic (with fixed `random_state`)
- Can be used as general-purpose dimensionality reduction (not just visualization)

```python
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_2d = reducer.fit_transform(X)
```

### PCA vs t-SNE vs UMAP Summary

| Property | PCA | t-SNE | UMAP |
| :--- | :--- | :--- | :--- |
| **Structure preserved** | Global variance | Local neighborhoods | Local + global |
| **Speed** | Fast | Slow ($O(N^2)$) | Fast ($O(N \log N)$) |
| **Deterministic** | Yes | No | Yes (with seed) |
| **Use as features** | Yes | No | Yes |
| **Interpretable axes** | Yes (PCs) | No | No |

---

# 6. Anomaly Detection

## Isolation Forest

**How it works:** randomly partition features recursively until a point is isolated. Anomalies require fewer splits (shorter path length).

**Anomaly score:** $s(x) = 2^{-E[h(x)] / c(n)}$ where $E[h(x)]$ is average path length.

Score close to 1 → anomaly. Score near 0.5 → normal.

**Why it wins for high-dim data:** density estimation is hard in high dimensions; Isolation Forest avoids it entirely.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso.fit(X_train)
scores = iso.decision_function(X_test)  # higher = more normal
preds = iso.predict(X_test)             # 1 = normal, -1 = anomaly
```

## Other Methods

| Method | Approach | Best For |
| :--- | :--- | :--- |
| **One-Class SVM** | Learn hypersphere around normal data | Low-dim, clear boundary |
| **Autoencoder** | High reconstruction error = anomaly | Complex patterns, images |
| **Local Outlier Factor** | Compare local density to neighbors | Non-uniform density |
| **Z-score / IQR** | Statistical threshold | Simple univariate |

---

# 7. Evaluation Metrics (No Ground Truth)

- **Silhouette Coefficient:** range $[-1, 1]$. High = dense, well-separated clusters.
- **Calinski-Harabasz Index:** ratio of between-cluster to within-cluster dispersion. Higher is better.
- **Davies-Bouldin Index:** average similarity between each cluster and its most similar one. Lower is better.

When ground truth is available (e.g., benchmarking):
- **Adjusted Rand Index (ARI):** adjusted for chance, range $[-1, 1]$
- **Normalized Mutual Information (NMI):** mutual information normalized to $[0, 1]$

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

print("Silhouette:", silhouette_score(X, labels))
print("Calinski-Harabasz:", calinski_harabasz_score(X, labels))
print("Davies-Bouldin:", davies_bouldin_score(X, labels))
```

---

> [!TIP]
> **Production Recommendation:** For visualization of millions of points, use **UMAP**. For customer segmentation on tabular data, **K-Means++** tuned via **Silhouette Score** is the industry baseline. For anomaly detection on high-dimensional data, **Isolation Forest** is the fast, reliable default.
