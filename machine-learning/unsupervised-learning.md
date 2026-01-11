# Unsupervised Learning: Finding Hidden Patterns

> [!IMPORTANT]
> **Executive Summary for Interviewees**
> 1. **Clustering:** Grouping points. **K-Means** (centroid-based, spherical) vs **DBSCAN** (density-based, arbitrary shapes, handles noise) vs **GMM** (probabilistic, elliptical).
> 2. **Dim. Reduction:** **PCA** (linear, preserves variance) vs **t-SNE** (non-linear, preserves local structure, great for visualization) vs **UMAP** (faster, preserves both local and global structure).
> 3. **Anomaly Detection:** **Isolation Forest** (isolates anomalies with few splits) is usually the go-to for high-dim data.
> 4. **Key Metric:** **Silhouette Score** (cohesion vs separation).

---

## Algorithm Classification Framework

Unsupervised learning can be divided into several core tasks:

| **Task** | **Purpose** | **Key Algorithms** | **When to Use** |
|----------|-------------|-------------------|----------------|
| **Clustering** | Group similar data points | K-Means, DBSCAN, Hierarchical, GMM | Customer segmentation, image segmentation |
| **Dim. Reduction** | Reduce feature count | PCA, t-SNE, UMAP, LDA (Unsup) | Visualization, noise reduction, speed |
| **Association** | Find rules (if A then B) | Apriori, FP-Growth | Market basket analysis |
| **Anomaly Detection** | Find rare, unusual points | Isolation Forest, One-Class SVM | Fraud detection, equipment failure |

---

## 1. Clustering Algorithms

### Algorithms Overview

| **Algorithm** | **Type** | **Complexity** | **Best For** | **Key Limitation** |
|--------------|----------|----------------|--------------|-------------------|
| **K-Means** | Centroid-based | O(n·k·i·p) | Spherical, even-sized clusters | Must specify K; sensitive to outliers |
| **DBSCAN** | Density-based | O(n²) to O(n·log n) | Non-spherical, varying shapes | Struggles with varying densities |
| **Hierarchical** | Tree-based | O(n³), O(n²) with spatial index | Small data, dendrogram needs | Extremely slow on large data |
| **GMM** | Distribution-based | O(n·k·i·p) | Soft clustering, elliptical shapes | Sensitive to initialization |

*n = samples, k = clusters, i = iterations, p = features*

### Detailed Algorithms

#### K-Means Clustering
**Purpose:** Partition data into K distinct, non-overlapping subgroups.

**Algorithm:**
1. Initialize K centroids randomly.
2. **Assignment:** Assign each point to its nearest centroid (usually Euclidean distance).
3. **Update:** Recalculate centroids as the mean of all points in that cluster.
4. Repeat until convergence.

**Inertia (Objective Function):**
```
Inertia = Σ Σ ||x_i - c_j||²
```
*Goal: Minimize within-cluster sum of squares (WCSS).*

**How to Choose K?**
- **Elbow Method:** Plot WCSS vs K; look for the "elbow" where reduction slows down.
- **Silhouette Method:** Measures how similar a point is to its own cluster vs others.

**Interview Tip:** Mention **K-means++** initialization, which chooses initial centroids far apart, leading to faster and more stable convergence.

---

#### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
**Purpose:** Group points that are closely packed, marking outliers in low-density regions.

**Hyperparameters:**
- **eps (ε):** Maximum distance between two points to be considered neighbors.
- **minSamples:** Minimum number of points required to form a "dense" region.

**Point Types:**
- **Core Point:** Has ≥ minSamples within distance ε.
- **Border Point:** Within ε of a core point but has < minSamples neighbors.
- **Noise (Outlier):** Neither a core nor a border point.

**Pros:**
- Doesn't require specifying K.
- Handles non-spherical shapes (e.g., moons, circles).
- Robust to outliers (built-in noise detection).

---

#### Hierarchical Clustering
**Purpose:** Build a tree-like hierarchy of clusters.

**Types:**
- **Agglomerative (Bottom-up):** Start with each point as a cluster, merge most similar.
- **Divisive (Top-down):** Start with one cluster, split into smaller groups.

**Linkage Methods (Similarity metrics):**
- **Single:** Min distance between points in two clusters (can cause "chaining").
- **Complete:** Max distance between points (produces compact clusters).
- **Average:** Average distance between all point pairs.
- **Ward:** Minimizes variance within clusters (most common with Agglomerative).

---

#### Gaussian Mixture Models (GMM)
**Purpose:** Probabilistic clustering where each cluster is modeled as a Gaussian distribution.

**Key Difference:**
- **Soft Clustering:** Provides a *probability* (responsibility) $\gamma_{ik}$ that point $i$ belongs to cluster $k$.
- **Flexibility:** Can model elliptical clusters via covariance matrices (K-means assumes spherical/identity covariance).

**Training: Expectation-Maximization (EM) Algorithm**
1. **E-Step:** Calculate the responsibility $\gamma_{ik}$ of each Gaussian for each data point using the current parameters.
2. **M-Step:** Update the mean, covariance, and mixing proportions of the Gaussians to maximize the likelihood of the data.

**AIC/BIC:** Used to select the optimal number of components. BIC penalizes model complexity more heavily than AIC.

---

## 2. Dimensionality Reduction

### Techniques Overview

| **Technique** | **Type** | **Preserves** | **Best For** |
|---------------|----------|---------------|--------------|
| **PCA** | Linear | Global variance | Compression, denoising, speed |
| **t-SNE** | Non-Linear | Local structures | 2D/3D high-quality visualization |
| **UMAP** | Non-Linear | Local + Global | Visualization, faster than t-SNE |
| **LDA** (Unsup) | Linear | Class separation | Feature extraction (supervised version common) |

### Detailed Techniques

#### PCA (Principal Component Analysis)
**How it works:** Finds new orthogonal axes (Principal Components) that maximize variance.

**Mathematical Steps:**
1. Standardize the data.
2. Compute Covariance Matrix.
3. Find Eigenvectors (directions) and Eigenvalues (magnitude of variance).
4. Sort by eigenvalues and keep top K components.

**Interview Question:** *"Is PCA sensitive to scale?"*
> **Yes.** You MUST standardize features before PCA, or features with larger ranges will dominate the principal components.

---

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)
**How it works:** Maps high-dim distances to conditional probabilities using a Gaussian distribution, then minimizes the Kullback-Leibler (KL) divergence between high-dim and low-dim probability distributions.

**Key Concept: The Perplexity Hyperparameter**
- **Perplexity:** Effectively the number of nearest neighbors each point considers. High perplexity balances local and global structure; too low results in "clumpy" artifacts.
- **t-Distribution:** Used in the low-dim space because it has "fatter tails," which prevents the "crowding problem" (all points collapsing to the center).

**Limitation:** It is stochastic (different runs yield different results), non-deterministic, and cannot project new points.

#### UMAP (Uniform Manifold Approximation and Projection)
**Better than t-SNE?**
- **Faster:** Much more scalable for large datasets.
- **Structure:** Preserves both local *and* more global structure than t-SNE.
- **Mapping:** Can learn a mapping that can be applied to new, unseen data.

---

## 3. Evaluation Metrics

Evaluating unsupervised models is difficult because there is no ground truth.

| **Metric** | **Range** | **What it measures** | **Higher/Lower better?** |
|------------|-----------|--------------------|------------------------|
| **Silhouette Score** | [-1, 1] | Separation vs Cohesion | Higher (close to 1) is better |
| **Inertia (WCSS)** | [0, ∞) | Within-cluster distance | Lower is better (but watch the elbow) |
| **Calinski-Harabasz** | [0, ∞) | Ratio of between-to-within variance | Higher is better |
| **Davies-Bouldin** | [0, ∞) | Avg similarity between clusters | Lower is better |

---

## 4. Anomaly Detection

| **Algorithm** | **Mechanism** | **Best For** |
|---------------|---------------|--------------|
| **Isolation Forest** | Random splits; anomalies isolate faster (short paths). | High-dim, non-linear data. |
| **Local Outlier Factor** | Compares density of a point to its neighbors. | Varying density clusters. |
| **One-Class SVM** | Learns a boundary around "normal" data. | Low-dim, structured data. |

**Interview Insight: "Why Isolation Forest?"**
Most algorithms try to define "normal" data and find what's left. Isolation Forest focuses on identifying anomalies *directly* by exploiting their susceptibility to isolation.

---

## Common Interview Questions

**1. "How do you choose the number of clusters (K) in K-means?"**
> I use a combination of the **Elbow Method** (looking for the point where the decrease in inertia slows significantly) and the **Silhouette Score** (aiming for a value closer to 1, indicating clear separation).

**2. "When would you prefer DBSCAN over K-means?"**
> When the clusters have non-spherical shapes, vary in density, or when the data contains significant noise/outliers that shouldn't be forced into a cluster. Also, when I don't know the number of clusters in advance.

**3. "What is the difference between PCA and t-SNE?"**
> PCA is a linear technique that focuses on preserving global variance; it's fast and deterministic. t-SNE is a non-linear, stochastic technique that focuses on preserving local structures, making it much better for visualization but slower and potentially misleading about global distances.

**4. "Why standardizing before PCA is essential?"**
> Because PCA seeks to maximize variance. If one feature is measured in 'kilometers' and another in 'meters', the one in meters will have a much higher numerical variance even if they are equally important, causing PCA to biasedly favor the meters scale.

---

## Python Code Snippets

### K-Means with Silhouette Score
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Calculate Metric
score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {score:.2f}")
```

### PCA for Visualization
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Visualization")
plt.show()

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
```


