---
module: Classical ML
topic: Unsupervised Learning
subtopic: ""
status: unread
tags: [classicalml, ml, unsupervised-learning]
---
# Unsupervised Learning

---

## Executive Summary & Cheatsheet

### Algorithm Table

| Algorithm | Best for | Key hyperparameters | Watch out for |
| :--- | :--- | :--- | :--- |
| **K-Means** | Round-ish blobs, customer segmentation, fast | K, n_init, max_iter | Must choose K; outlier-sensitive; assumes spherical clusters |
| **Hierarchical (Agglomerative)** | Dendrograms, interpretable hierarchy | linkage, distance metric | O(N²) memory; slow at large N |
| **DBSCAN** | Arbitrary shapes, noisy data, outlier detection | eps, min_samples | Struggles with varying density; sensitive to eps |
| **PCA** | Dimensionality reduction, decorrelation | n_components | Linear only; components are not interpretable features |
| **t-SNE** | 2D/3D visualization, explore local cluster structure | perplexity, learning_rate | Not for global structure; non-deterministic; costly at large N |
| **UMAP** | Visualization at scale, local + some global | n_neighbors, min_dist | Distances between clusters less meaningful than within |

### Choosing K (for K-Means)
- **Elbow method:** plot inertia (WCSS) vs K → pick the "elbow"
- **Silhouette score:** range [-1, 1]; higher = tighter + better separated
- Use **K-Means++** initialization to reduce bad restarts

### Dimensionality Reduction: Three Cameras
- **PCA:** Linear, fast, global structure, deterministic. Use for preprocessing.
- **t-SNE:** Non-linear, slow, no global structure, stochastic. Use for final 2D viz.
- **UMAP:** Non-linear, medium-fast, partial global structure, stochastic. Use for viz at scale.

### Evaluation (no labels)
| Metric | What it measures | Range | Better = |
| :--- | :--- | :--- | :--- |
| **Silhouette** | Cohesion vs separation | [-1, 1] | Higher |
| **Calinski-Harabasz** | Between/within cluster ratio | [0, ∞) | Higher |
| **Davies-Bouldin** | Cluster compactness + separation | [0, ∞) | Lower |

*Honest caveat: Internal metrics measure geometry, not business value. Always sanity-check segments against domain knowledge.*

---

## Deep Dive

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

## Hierarchical Clustering (Agglomerative)

**The problem**: K-Means requires knowing $K$ upfront, and returns a single flat partition. You may not know the right $K$, or you may want to explore the structure at multiple granularities — are there 3 natural segments, or 10 sub-segments within those 3?

**The core insight**: build a hierarchy of merges. Start with every point as its own cluster. At each step, merge the two most similar clusters. Stop when you have one cluster. The result is a tree (dendrogram) that encodes every possible partition simultaneously — you can cut it at any level to get any number of clusters.

**The mechanics** (bottom-up agglomerative):
1. Assign each point to its own cluster ($n$ clusters)
2. Compute pairwise distances between all clusters
3. Merge the two closest clusters
4. Repeat until one cluster remains (or stopping criterion)
5. Cut the dendrogram at the desired height to extract $K$ clusters

**Linkage criteria** — how to measure the distance between two clusters:

| Linkage | Distance definition | Shape bias | When to use |
|---------|-------------------|-----------|------------|
| **Single** | Minimum distance between any pair | Elongated, chain-like | Non-convex shapes; sensitive to outliers (chaining effect) |
| **Complete** | Maximum distance between any pair | Compact, spherical | When you want tightly bounded clusters |
| **Average (UPGMA)** | Mean distance between all pairs | Moderate | General-purpose, less sensitive than single/complete |
| **Ward's** | Merge that minimizes increase in total within-cluster variance | Spherical, compact | Best default; analogous to K-Means objective |

Ward's linkage is the most commonly used. At each merge, it chooses the pair whose merge minimizes:
$$\Delta = \frac{n_A \cdot n_B}{n_A + n_B} \|\mu_A - \mu_B\|^2$$

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Compute linkage matrix
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='lastp', p=20)   # show last 20 merges
plt.title('Ward Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster size (in parentheses)')
plt.ylabel('Distance')
plt.show()

# Cut at a specific number of clusters
labels = fcluster(Z, t=4, criterion='maxclust')   # extract 4 clusters

# Or cut at a specific distance threshold
labels = fcluster(Z, t=5.0, criterion='distance')
```

**Reading the dendrogram**: long vertical lines indicate a large distance jump when two clusters merged — this is where you should cut. A cut just below a long line gives natural clusters. A cut near the top gives few large clusters; near the bottom gives many small clusters.

**Complexity**: $O(N^3)$ time and $O(N^2)$ space for naive implementation. Limits use to $N < 10,000$ unless approximate methods are used (`fastcluster` library implements $O(N^2)$ Ward's).

**What breaks**: hierarchical clustering is sensitive to outliers (especially with single linkage), cannot be incrementally updated (must rerun from scratch if data changes), and is too slow for large datasets. Does not naturally handle high-dimensional data — use dimensionality reduction first.

**When to use over K-Means**: when the number of clusters is unknown; when you need to explore cluster structure at multiple resolutions; when clusters are nested or hierarchical by nature (e.g., biological taxonomy, organizational hierarchies).

---

## Dimensionality Reduction

> Full treatment of PCA, t-SNE, UMAP, Kernel PCA, ICA, LDA, NMF, and Autoencoders is in **[10-dimensionality-reduction.md](05-dimensionality-reduction.md)**. This section provides a quick reference for the clustering context.

**PCA**: finds linear directions of maximum variance. Use before clustering to remove noise dimensions and reduce the curse of dimensionality. Always standardize features first. Choose $k$ by the explained variance ratio — 95% cumulative variance is a common threshold.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, svd_solver='full')
X_reduced = pca.fit_transform(X_scaled)
```

**t-SNE**: minimizes KL divergence between pairwise similarity distributions in high-d and low-d. Preserves local neighborhoods only. $O(N^2)$ — use Barnes-Hut approximation for $N > 10$k. Output is only for visualization; do not use as features for downstream modeling.

**UMAP**: topological manifold learning. $O(N \log N)$. Preserves both local and global structure better than t-SNE. Deterministic with fixed seed. Can be used as a preprocessing step for clustering or classification.

| | PCA | t-SNE | UMAP |
|---|---|---|---|
| Structure | Global variance | Local only | Local + global |
| Speed | Fast | Slow | Fast |
| Use as features | Yes | No | Yes |
| Best for | Preprocessing, noise removal | Visualization | Visualization + features |

---

## Anomaly Detection

> **Full treatment:** [10-anomaly-detection.md](10-anomaly-detection.md) — isolation vs. density vs. boundary framings, the `contamination` trap, threshold-as-alert-budget, and interview angles. The table below is the quick selector for the unsupervised-learning context.

**Quick method selector:**

| Method | Approach | Best for |
|--------|----------|---------|
| **Isolation Forest** | Anomalies isolate in fewer random splits | High-dimensional tabular data; fast default |
| **One-Class SVM** | Hypersphere around normal data | Low-dimensional data with a clear boundary |
| **LOF** | Compare local density to neighbors | Non-uniform density; anomalies in varying-density regions |
| **Elliptic Envelope** | Fit a robust Gaussian; flag low-probability points | Gaussian-distributed data |
| **Autoencoder** | High reconstruction error = anomaly | Complex patterns, images, sequences |

**Default choice:** Isolation Forest for tabular data. Use `contamination` to set the expected fraction of anomalies (e.g., 0.05 for 5%).

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso.fit(X_train)
preds = iso.predict(X_test)   # 1 = normal, -1 = anomaly
```

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
A: PCA finds the directions of maximum variance in the data — the eigenvectors of the covariance matrix, ranked by eigenvalue (variance explained). Full derivation (Lagrange-multiplier argument, SVD connection, explained-variance-ratio/scree-plot practice) lives in [10-dimensionality-reduction.md](05-dimensionality-reduction.md#pca-principal-component-analysis), the canonical source. Limitation: PCA is linear — it can't capture nonlinear structure. For nonlinear dimensionality reduction: UMAP (preserves local and some global structure, scalable), t-SNE (preserves local structure only, for visualization), Kernel PCA (implicit nonlinear mapping via kernel trick).


