# Day 10-11: Unsupervised Learning Algorithms

## 📋 Executive Summary
| Category | Algorithm | Core Objective | Key Hyperparameter |
|----------|-----------|----------------|--------------------|
| **Clustering** | **K-Means** | Minimize Inertia | $K$ (Clusters) |
| **Clustering** | **DBSCAN** | Density-based grouping | $\epsilon$ (Radius) |
| **Dim. Reduction** | **PCA** | Preserve Max Variance | $n\_components$ |
| **Dim. Reduction** | **t-SNE** | Preserve Local Topology | Perplexity |

---

## 🔬 1. Clustering Techniques

### K-Means
An iterative algorithm that assigns points to the nearest centroid.
- **Objective**: Minimize Inertia (Within-cluster sum of squares).
- **The Elbow Method**: Plot Inertia vs. $K$ to find the "elbow".

### Hierarchical Clustering
- **Agglomerative**: Bottom-up approach. Start with individual points and merge them.
- **Dendrogram**: A tree-like chart used to visualize the merging process and decide on the number of clusters.

---

## 📐 2. Dimensionality Reduction

### Principal Component Analysis (PCA)
A linear transformation that finds orthogonal axes (Principal Components) along which variance is maximized.
- **Steps**: Covariance matrix $\rightarrow$ Eigenvalues/Eigenvectors $\rightarrow$ Sort and pick top $k$.

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
A non-linear method primarily used for **visualization**.
- **Intuition**: Maps neighbors in high-dim space to neighbors in low-dim space.
- **Warning**: Does NOT preserve global distances; only local structure is guaranteed.

---

## ❓ Interview Questions

**1. "What are the limitations of K-Means?"**
> It assumes clusters are spherical and equal-sized. It is sensitive to outliers and requires the number of clusters $K$ to be predefined. It can also get stuck in local minima (fixed by `n_init`).

**2. "How would you handle high-dimensional categorical data for clustering?"**
> standard K-Means uses Euclidean distance, which isn't suitable for categories. Use **K-Modes** or embed the categories into a continuous space first using techniques like Entity Embeddings.

**3. "Why is PCA considered a feature extraction technique rather than selection?"**
> Feature selection keeps a subset of original features. PCA creates completely *new* features (linear combinations of the original ones), so it is "Extracting" new representation.

---

## 💻 Cluster Evaluation
```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3).fit(X)
# Silhouette Score: Closer to 1 is better (well-separated)
score = silhouette_score(X, model.labels_)
```
