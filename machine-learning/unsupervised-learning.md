# Unsupervised Learning Mastery (Deep-Dive)

This track explores the challenge of extracting structure from unlabeled data. It covers clustering, dimensionality reduction, and anomaly detection.

---

# 1. 🔹 Clustering Paradigms

## Q1: Explain the tradeoffs between the 3 major clustering families.

### 🔹 Comparison Table

| Paradigm | Key Algorithm | Best For | Weakness |
| :--- | :--- | :--- | :--- |
| **Centroid-based** | K-Means | Spherical, even clusters. | Must pick K; sensitive to outliers. |
| **Density-based** | DBSCAN | Arbitrary shapes, Noise. | Struggles with varying densities. |
| **Probabilistic** | GMM | Soft clustering, Ellipses. | Sensitive to initialization. |
| **Hierarchical** | Agglomerative | Dendrograms, Small data. | $O(N^3)$ complexity (Slow). |

---

# 2. 🔹 K-Means Clustering

## Q2: How do you choose the optimal "K"?

### 🔹 Direct Answer
1. **The Elbow Method:** Plot the Within-Cluster Sum of Squares (Inertia) against K. Look for the "elbow" where the reduction in inertia slows significantly.
2. **Silhouette Score:** Measures cohesion vs. separation. A higher score (closer to 1) indicate well-defined clusters that are far from neighboring clusters.

### 🔹 Pro Tip: K-Means++
Standard K-Means is sensitive to random initialization. **K-Means++** chooses the first centroid randomly and subsequent centroids proportionally to their distance from the nearest existing centroid. This leads to faster convergence and better global optima.

---

# 3. 🔹 Dimensionality Reduction

## Q3: PCA vs. t-SNE vs. UMAP.

### 🔹 Direct Answer
- **PCA (Linear):** Preserves **global** variance. Fast, deterministic, and interpretable.
- **t-SNE (Non-Linear):** Preserves **local** similarities. Highly expensive, stochastic, and focuses on 2D/3D visualization.
- **UMAP (Non-Linear):** Preserves both **local** and **global** structure. Much faster than t-SNE and scalable to large datasets.

---

# 4. 🔹 Anomaly Detection

## Q4: Why is Isolation Forest the standard for high-dimensional data?

### 🔹 Direct Answer
Unlike density-based methods that try to define "normal" data (which is hard in high dimensions), **Isolation Forests** work by randomly splitting features until a point is isolated. Anomalies are "few and different," so they isolate with far fewer splits (shorter path lengths) than normal data.

---

# 5. 🔹 Evaluation Metrics

## Q5: How do you evaluate a model with no ground truth?

### 🔹 Metrics
- **Silhouette Coefficient:** Range [-1, 1]. High value = dense, well-separated clusters.
- **Calinski-Harabasz Index:** Ratio of between-cluster dispersion to within-cluster dispersion.
- **Davies-Bouldin Index:** Average similarity between each cluster and its most similar one. Lower is better.

---

> [!TIP]
> **Production Recommendation:** For visualization of millions of points, use **UMAP**. For customer segmentation on tabular data, **K-Means++** centered via **Silhouette Score** is the industry baseline.
