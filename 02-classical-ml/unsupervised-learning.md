---
module: Classical Ml
topic: Unsupervised Learning
subtopic: ""
status: unread
tags: [classicalml, ml, unsupervised-learning]
---
# Unsupervised Learning

---

## Clustering Paradigms

| Paradigm | Key Algorithm | Best For | Weakness |
| :--- | :--- | :--- | :--- |
| **Centroid-based** | K-Means | Spherical, even clusters | Must pick K; sensitive to outliers |
| **Density-based** | DBSCAN | Arbitrary shapes, noise | Struggles with varying densities |
| **Probabilistic** | GMM | Soft clustering, elliptical shapes | Sensitive to initialization |
| **Hierarchical** | Agglomerative | Dendrograms, small data | $O(N^3)$ complexity |

---

## K-Means Clustering

**The problem**: you have unlabeled points and want to group similar ones together, but "similar" is relative — a point near the boundary between two groups could belong to either. You need a tractable definition of "good grouping" that you can actually optimize.

**The core insight**: define a good grouping as one where points are close to their group's center. Minimize the total distance from every point to its assigned centroid. This turns an exponentially hard combinatorial problem into a simple alternating-optimization: fix assignments and update centroids, then fix centroids and update assignments.

**The mechanics**: minimize within-cluster sum of squares (inertia):

$$J = \sum_{j=1}^{K} \sum_{x \in C_j} \|x - \mu_j\|^2$$

Algorithm:
1. Initialize $K$ centroids
2. Assign each point to nearest centroid
3. Recompute centroids as cluster means
4. Repeat until convergence

Convergence is guaranteed (inertia decreases monotonically), but the solution may be a local minimum.

### Choosing K

**Elbow Method**: plot inertia vs $K$; pick the "elbow" where reduction slows.

**Silhouette Score**: measures cohesion (within-cluster distance) vs separation (nearest-cluster distance):

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Range $[-1, 1]$. Higher is better. Average over all points.

### K-Means++

**The problem**: random initialization often places multiple centroids in the same dense region, leaving other regions uncovered and causing poor local minima.

**The core insight**: spread initial centroids far apart by choosing each new centroid with probability proportional to its squared distance from the nearest already-chosen centroid. This gives K-Means++ an $O(\log K)$ approximation guarantee.

1. Choose first centroid uniformly at random
2. Choose next centroid with probability $\propto d(x, \text{nearest centroid})^2$
3. Repeat until $K$ centroids are chosen

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

**What breaks**: K-Means assumes clusters are spherical and roughly equal in size — it partitions space by nearest centroid, which draws Voronoi boundaries (straight lines). Elongated clusters, clusters of very different sizes, and non-convex shapes all break it. Outliers pull centroids away from the true cluster center because the squared-distance objective penalizes distant points heavily.

---

## DBSCAN

**The problem**: K-Means forces every point into a cluster even if it is noise, and cannot find clusters of arbitrary shapes. Real data often has clusters that curve, branch, or interleave — straight Voronoi boundaries cannot separate them.

**The core insight**: define a cluster not by distance to a center, but by density. Points that belong together form dense regions separated by sparse ones. Any point surrounded by enough neighbors is a core of a cluster; points reachable from cores belong to the cluster; everything else is noise.

**The mechanics**:

Parameters:
- `ε` (eps): neighborhood radius
- `MinPts` (min_samples): minimum points to form a dense region

Point types:
- **Core point**: at least `MinPts` points within `ε`
- **Border point**: within `ε` of a core point but fewer than `MinPts` neighbors
- **Noise**: neither core nor border

Algorithm: expand clusters by connecting core points that are within `ε` of each other.

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
```

**What breaks**: DBSCAN requires a single density threshold `ε` across the entire dataset. When clusters have varying densities (tight cluster next to a loose cluster), no single `ε` works — if you set it low enough to resolve the tight cluster, the loose cluster dissolves into noise. HDBSCAN fixes this with hierarchical density estimation, finding clusters at multiple density levels simultaneously.

---

## Gaussian Mixture Models (GMM)

**The problem**: K-Means gives hard assignments — each point belongs to exactly one cluster. Real data rarely has hard boundaries. A customer who scores 50 on two lifestyle dimensions is genuinely ambiguous between two segments. You want a model that expresses this uncertainty as a probability.

**The core insight**: assume the data was generated by a mixture of Gaussian distributions. Each Gaussian is a cluster. Soft-assign each point to each cluster proportionally to how well that cluster's Gaussian explains it. Then re-estimate the Gaussians given those assignments. Iterate.

**The mechanics**: data is assumed drawn from $K$ Gaussian components:

$$P(x) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

where $\pi_k$ are mixing weights ($\sum \pi_k = 1$).

Fitting via EM algorithm:
- **E-step**: compute soft assignments $r_{ik} = P(z_k | x_i)$ (responsibility of component $k$ for point $i$)
- **M-step**: update $\mu_k$, $\Sigma_k$, $\pi_k$ using weighted MLE

Each covariance matrix $\Sigma_k$ allows clusters to be elliptical in any orientation — strictly more expressive than K-Means's spherical assumption.

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)   # soft assignments
```

**What breaks**: GMMs are sensitive to initialization and can converge to degenerate solutions where one component collapses onto a single point (its variance goes to zero). Like K-Means, GMMs require the number of components $K$ to be specified. The Gaussian assumption fails for data with heavy tails or multi-modal distributions within a single cluster.

---

## Dimensionality Reduction

### PCA (Principal Component Analysis)

**The problem**: high-dimensional data is expensive to store, slow to compute on, and hard to visualize. Many features are correlated — stock prices for companies in the same sector move together, pixel intensities in adjacent image patches are similar. The data lives on a lower-dimensional structure even though it has thousands of columns.

**The core insight**: find the directions in which the data varies most. Project onto those directions. The first direction (principal component) captures the most variance; the second captures the most remaining variance orthogonal to the first; and so on. You can drop the trailing directions without losing much information.

**The mechanics**:
1. Center data: $X \leftarrow X - \bar{X}$
2. Compute covariance matrix: $\Sigma = \frac{1}{m} X^T X$
3. Eigen-decompose: $\Sigma v = \lambda v$
4. Project onto top-$k$ eigenvectors

Explained variance ratio: $\frac{\lambda_i}{\sum_j \lambda_j}$ — choose $k$ where cumulative ratio exceeds threshold (e.g., 95%).

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, svd_solver='full')  # keep 95% variance
X_reduced = pca.fit_transform(X_train)
print(f"Reduced from {X_train.shape[1]} to {X_reduced.shape[1]} features")
print(f"Explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")
```

**What breaks**: PCA finds linear structure only. If the true low-dimensional structure is a curved manifold (e.g., a Swiss roll), PCA will project it onto a flat plane that folds the manifold on top of itself, destroying the structure. It is also sensitive to feature scales — always standardize before applying PCA. And PCA components are linear combinations of all original features, so interpretability is lost.

---

### t-SNE

**The problem**: PCA is linear, so it cannot unroll curved manifolds or reveal clusters that are only apparent in non-linear structure. When you reduce MNIST digit embeddings with PCA, different digits overlap. You need a method that keeps nearby points nearby even if that requires distorting global geometry.

**The core insight**: represent pairwise similarities as probability distributions — high probability if two points are close. Do this in both the high-dimensional and low-dimensional spaces. Minimize the KL divergence between them, forcing the 2D layout to match the high-dimensional neighborhood structure. The t-distribution in low-d (heavier tails than Gaussian) allows distant points to spread out freely while keeping nearby clusters tight.

**What breaks**:
- $O(N^2)$ complexity — slow for $N > 10$k without approximations (Barnes-Hut)
- Stochastic — different runs give different layouts
- Distances between clusters are not meaningful — only local neighborhood structure is preserved
- Cannot be used as features for downstream tasks; only for 2D/3D visualization

---

### UMAP

**The problem**: t-SNE sacrifices global structure for local fidelity, and is too slow for large datasets. You want a method that preserves both local and global structure, is fast enough for production, and produces embeddings that can be used as features.

**The core insight**: model the data as a topological manifold using fuzzy simplicial sets. Construct a graph where edge weights represent the probability that two points are connected under the manifold's metric. Find a low-dimensional representation whose graph structure matches, preserving both local (neighbor) and global (between-cluster) relationships. The manifold-theoretic foundation makes it faster and more structure-preserving than t-SNE.

**Advantages over t-SNE**:
- Much faster: $O(N \log N)$
- Preserves more global structure
- Deterministic (with fixed `random_state`)
- Can be used as general-purpose dimensionality reduction (not just visualization)

```python
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_2d = reducer.fit_transform(X)
```

**What breaks**: UMAP's global structure preservation is relative — it is better than t-SNE but still not as faithful as PCA for global variance. Cluster sizes and distances in UMAP plots can be misleading. The `n_neighbors` hyperparameter controls the local/global tradeoff: too small → fragmented local structure, too large → global structure dominates and local clusters blur.

### PCA vs t-SNE vs UMAP Summary

| Property | PCA | t-SNE | UMAP |
| :--- | :--- | :--- | :--- |
| **Structure preserved** | Global variance | Local neighborhoods | Local + global |
| **Speed** | Fast | Slow ($O(N^2)$) | Fast ($O(N \log N)$) |
| **Deterministic** | Yes | No | Yes (with seed) |
| **Use as features** | Yes | No | Yes |
| **Interpretable axes** | Yes (PCs) | No | No |

---

## Anomaly Detection

### Isolation Forest

**The problem**: density-based anomaly detectors (like One-Class SVM) struggle in high dimensions because estimating density is hard — the curse of dimensionality makes all points look equally distant from each other.

**The core insight**: anomalies are rare and different. If you randomly partition the feature space, anomalies end up isolated quickly — they sit in sparse regions where a few random cuts are enough to separate them from all other points. Normal points sit in dense regions and require many cuts. You don't need to model density at all — just measure how many cuts it takes to isolate a point.

**The mechanics**: randomly partition features recursively until a point is isolated. Anomalies require fewer splits (shorter path length).

Anomaly score: $s(x) = 2^{-E[h(x)] / c(n)}$ where $E[h(x)]$ is average path length.

Score close to 1 → anomaly. Score near 0.5 → normal.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso.fit(X_train)
scores = iso.decision_function(X_test)  # higher = more normal
preds = iso.predict(X_test)             # 1 = normal, -1 = anomaly
```

**What breaks**: Isolation Forest struggles when anomalies are clustered together — a group of anomalies forms a dense region, making them look normal. It also performs poorly when normal data has multi-modal structure and anomalies sit between the modes, or when anomalies are only detectable through specific feature interactions.

### Other Anomaly Detection Methods

| Method | Approach | Best For |
| :--- | :--- | :--- |
| **One-Class SVM** | Learn hypersphere around normal data | Low-dim, clear boundary |
| **Autoencoder** | High reconstruction error = anomaly | Complex patterns, images |
| **Local Outlier Factor** | Compare local density to neighbors | Non-uniform density |
| **Z-score / IQR** | Statistical threshold | Simple univariate |

**Local Outlier Factor** solves Isolation Forest's weakness with locally varying density: instead of a global isolation measure, it compares each point's density to its neighbors' densities. A point is anomalous if it is in a sparse region surrounded by denser neighborhoods.

---

## Evaluation Metrics

### Without Ground Truth

**The problem**: there is no target variable to evaluate against. You can run any algorithm and get clusters, but how do you know if they are meaningful?

**The core insight**: a good clustering is one where points within a cluster are more similar to each other than to points in other clusters. Measure this internal consistency without needing labels.

- **Silhouette Coefficient**: range $[-1, 1]$. Measures how much closer each point is to its own cluster than to the nearest other cluster. High = dense, well-separated clusters.
- **Calinski-Harabasz Index**: ratio of between-cluster to within-cluster dispersion. Higher is better.
- **Davies-Bouldin Index**: average similarity between each cluster and its most similar one. Lower is better.

When ground truth is available (e.g., benchmarking):
- **Adjusted Rand Index (ARI)**: adjusted for chance, range $[-1, 1]$
- **Normalized Mutual Information (NMI)**: mutual information normalized to $[0, 1]$

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

print("Silhouette:", silhouette_score(X, labels))
print("Calinski-Harabasz:", calinski_harabasz_score(X, labels))
print("Davies-Bouldin:", davies_bouldin_score(X, labels))
```

**What breaks**: all internal metrics have blind spots. Silhouette favors convex, compact clusters and will rank K-Means highly even when the true structure is non-convex. Calinski-Harabasz is biased toward more clusters. None of these metrics measures whether the clusters are useful for your downstream task — only domain knowledge can verify that.

---

> [!TIP]
> **Production Recommendation:** For visualization of millions of points, use **UMAP**. For customer segmentation on tabular data, **K-Means++** tuned via **Silhouette Score** is the industry baseline. For anomaly detection on high-dimensional data, **Isolation Forest** is the fast, reliable default.

---

## Canonical Interview Q&As

**Q: Derive the k-means objective and explain why the algorithm converges but not necessarily to the global optimum.**
A: k-means minimizes the within-cluster sum of squared distances: J = Σ_{k=1}^K Σ_{x∈C_k} ||x - μ_k||². The algorithm alternates: (E-step) assign each point to the nearest centroid — this minimizes J for fixed μ_k since each point goes to its closest center; (M-step) update each centroid to the mean of its cluster — this minimizes J for fixed assignments since the mean minimizes sum of squared deviations. Each step monotonically decreases J, and J is bounded below by 0, so convergence is guaranteed. However, the algorithm converges to a local minimum, not global, because: the assignment step is greedy (a point stays with its current cluster even if a small perturbation would reduce global J), and the problem is NP-hard in general. In practice: run k-means 10-20 times with different random seeds (k-means++ initialization dramatically improves seed quality by choosing initial centroids proportionally to their distance from existing centroids), keep the run with lowest final J. k-means++ gives O(log k) approximation to the optimal solution and in practice reduces the variance in final J by 5-10×.

**Q: What are the assumptions behind k-means and when does it fail?**
A: k-means assumes: (1) clusters are spherical (uses Euclidean distance — elongated or non-convex clusters get split); (2) clusters have similar size (assignment is by distance, so large clusters absorb small nearby ones); (3) clusters have similar variance (same issue — a tight cluster next to a spread-out cluster will lose points to the spread cluster); (4) the correct k is known. Failure modes: concentric circles or crescent-shaped clusters → use DBSCAN or spectral clustering; clusters with very different densities → DBSCAN with adaptive ε, or GMM with different covariances; k unknown → use elbow method (plot J vs k, look for the elbow) or silhouette score (measures how similar a point is to its cluster vs other clusters; higher = better separation). GMM fixes the spherical assumption by modeling each cluster as a Gaussian with its own covariance matrix, estimated via EM. The trade-off: GMM is more flexible but requires more data, is more expensive (O(d²) per step for full covariance), and can overfit on small datasets.

**Q: Explain PCA from first principles — what is it actually computing and why do the eigenvectors matter?**
A: PCA finds the directions of maximum variance in the data. Given centered data X (n×d), the covariance matrix is C = XᵀX/(n-1) (d×d). The eigendecomposition C = VΛVᵀ gives eigenvectors V (principal components) and eigenvalues Λ (variance explained along each direction). The first principal component v₁ is the direction that maximizes Var(Xv) = vᵀCv subject to ||v||=1 — by the Rayleigh quotient, this is the eigenvector with the largest eigenvalue. Projecting X onto the top k eigenvectors: Z = XV_k (n×k) gives the k-dimensional representation that preserves maximum variance. Why it works for compression: most real datasets have rapidly decaying eigenvalues — the top 10 components might capture 95% of variance. Connection to SVD: X = UΣVᵀ, where the right singular vectors V are exactly the PCA eigenvectors. SVD is numerically preferred over eigendecomposition because it avoids computing XᵀX explicitly (which can lose precision). Limitation: PCA is linear — it can't capture nonlinear structure. For nonlinear dimensionality reduction: UMAP (preserves local and some global structure, scalable), t-SNE (preserves local structure only, for visualization), Kernel PCA (implicit nonlinear mapping via kernel trick).

## Flashcards

**ε (eps)?** #flashcard
neighborhood radius

**MinPts (min_samples)?** #flashcard
minimum points to form a dense region

**Core point?** #flashcard
at least MinPts points within ε

**Border point?** #flashcard
within ε of a core point but fewer than MinPts neighbors

**Noise?** #flashcard
neither core nor border

**E-step?** #flashcard
compute soft assignments $r_{ik} = P(z_k | x_i)$ (responsibility of component $k$ for point $i$)

**M-step?** #flashcard
update $\mu_k$, $\Sigma_k$, $\pi_k$ using weighted MLE

**$O(N^2)$ complexity?** #flashcard
slow for $N > 10$k without approximations (Barnes-Hut)

**Stochastic?** #flashcard
different runs give different layouts

**Distances between clusters are not meaningful?** #flashcard
only local neighborhood structure is preserved

**Cannot be used as features for downstream tasks; only for 2D/3D visualization?** #flashcard
Cannot be used as features for downstream tasks; only for 2D/3D visualization

**Much faster?** #flashcard
$O(N \log N)$

**Preserves more global structure?** #flashcard
Preserves more global structure

**Deterministic (with fixed random_state)?** #flashcard
Deterministic (with fixed random_state)

**Can be used as general-purpose dimensionality reduction (not just visualization)?** #flashcard
Can be used as general-purpose dimensionality reduction (not just visualization)

**Silhouette Coefficient?** #flashcard
range $[-1, 1]$. Measures how much closer each point is to its own cluster than to the nearest other cluster. High = dense, well-separated clusters.

**Calinski-Harabasz Index?** #flashcard
ratio of between-cluster to within-cluster dispersion. Higher is better.

**Davies-Bouldin Index?** #flashcard
average similarity between each cluster and its most similar one. Lower is better.

**Adjusted Rand Index (ARI)?** #flashcard
adjusted for chance, range $[-1, 1]$

**Normalized Mutual Information (NMI)?** #flashcard
mutual information normalized to $[0, 1]$
