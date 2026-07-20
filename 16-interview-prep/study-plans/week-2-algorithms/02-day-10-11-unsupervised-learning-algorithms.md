---
module: Interview Prep
topic: Week 2 Algorithms
subtopic: Day 10 11 Unsupervised Learning Algorithms
status: unread
tags: [studyplans, ml, week-2-algorithms-day-10-11-un]
---
# Day 10-11: Unsupervised Learning Algorithms

## Why This Topic Comes Here

After two days of supervised algorithms, you have a solid picture of how models learn when they have labels. Unsupervised learning is introduced now because (1) many real-world problems have no labels, and (2) unsupervised techniques — especially dimensionality reduction — are often used as preprocessing steps before supervised models. PCA, for example, is a tool you might apply before any of the algorithms from days 8-9. Understanding unsupervised learning also sharpens your understanding of supervised learning: the absence of a loss signal makes you think harder about what "learning" actually means.

---

## Executive Summary

| Category | Algorithm | Core Objective | Key Hyperparameter |
|----------|-----------|----------------|--------------------|
| **Clustering** | **K-Means** | Minimize Inertia | $K$ (Clusters) |
| **Clustering** | **DBSCAN** | Density-based grouping | $\epsilon$ (Radius) |
| **Dim. Reduction** | **PCA** | Preserve Max Variance | $n\_components$ |
| **Dim. Reduction** | **t-SNE** | Preserve Local Topology | Perplexity |

---

## 1. Clustering Techniques

**Why clustering is harder to evaluate than supervised learning:** In supervised learning, ground truth exists. In clustering, there is no externally correct answer — only the question of whether the clusters are useful for your task. This forces you to think about evaluation more carefully than you did for supervised models.

### K-Means

An iterative algorithm that assigns points to the nearest centroid, then recomputes centroids.
- **Objective**: Minimize Inertia (Within-cluster sum of squares).
- **The Elbow Method**: Plot Inertia vs. $K$ to find the "elbow" — the point where adding clusters no longer significantly reduces inertia.

**Key insight:** K-Means optimizes the *within-cluster* distance, not whether clusters are meaningful for your problem. A K-Means solution that perfectly minimizes inertia can still produce clusters that have no semantic coherence. The algorithm finds the mathematically tightest partitioning of the space — whether that partitioning corresponds to something meaningful is a question only you and your domain can answer.

**How to verify understanding:** K-Means with $K=10$ produces lower inertia than $K=3$ on your dataset. Does this mean $K=10$ is the better clustering? Explain what additional evidence you would need before choosing $K$.

**What trips people up:** Treating the elbow as objective. The elbow method is heuristic. On many real datasets, there is no sharp elbow — inertia decreases smoothly. In those cases, you must rely on silhouette scores, domain knowledge, or downstream task performance to choose $K$.

### Hierarchical Clustering

- **Agglomerative**: Bottom-up approach. Start with individual points and merge them.
- **Dendrogram**: A tree-like chart used to visualize the merging process and decide on the number of clusters.

**Key insight:** Hierarchical clustering does not require you to pre-specify $K$. The dendrogram shows the full merging history, and you cut the tree at whatever height makes sense for your task. This makes it more exploratory than K-Means — you can look at 3-cluster and 7-cluster solutions from a single run.

**How to verify understanding:** You run hierarchical clustering and observe that two clusters merge at a very high distance (late in the process). What does this tell you about those two clusters relative to the others?

**What trips people up:** Assuming hierarchical clustering scales to large datasets. Agglomerative clustering is $O(n^3)$ without approximations. For large $n$, K-Means or mini-batch K-Means is required.

### DBSCAN

**Key insight:** Unlike K-Means, DBSCAN finds clusters of arbitrary shape and identifies noise points (outliers) explicitly. It does not require you to specify $K$ — it discovers the number of clusters from the data's density structure. But it requires you to specify $\epsilon$ (neighborhood radius) and `min_samples`, which depend on the data scale and density — and these are just as hard to choose as $K$.

**How to verify understanding:** K-Means assigns every point to a cluster. DBSCAN can label points as noise. When is this noise-labeling property a feature rather than a limitation?

**What trips people up:** Using DBSCAN on data with varying densities. DBSCAN uses a single global $\epsilon$, so regions of different densities will either merge into one cluster (if $\epsilon$ is large) or produce many tiny clusters (if $\epsilon$ is small). HDBSCAN addresses this but is less commonly implemented.

---

## 2. Dimensionality Reduction

**Why dimensionality reduction belongs in the same session as clustering:** Both are unsupervised. More importantly, dimensionality reduction is often applied before clustering — both for computational reasons and because distance metrics in very high dimensions are unreliable. PCA at 50 components followed by K-Means often outperforms K-Means on 10,000 raw features.

### Principal Component Analysis (PCA)

A linear transformation that finds orthogonal axes (Principal Components) along which variance is maximized.
- **Steps**: Covariance matrix $\rightarrow$ Eigenvalues/Eigenvectors $\rightarrow$ Sort and pick top $k$.

**Key insight:** PCA maximizes variance, not predictive power. These are not the same. A feature that varies a lot may carry no information about the target. Conversely, a low-variance feature might be highly predictive. PCA applied before supervised learning can discard the most predictive signal if that signal lies in low-variance directions. This is why PCA as a preprocessing step for supervised models should be validated, not assumed.

**How to verify understanding:** You apply PCA to 100 features and retain the top 10 components (capturing 95% of variance). Your supervised model's performance drops significantly. What is the most likely explanation?

**What trips people up:** Confusing PCA (feature extraction — creates new features as linear combinations) with feature selection (which keeps original features). This matters for interpretability: you cannot trace a PCA component back to an individual original feature in a straightforward way.

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

A non-linear method primarily used for **visualization**.
- **Intuition**: Maps neighbors in high-dim space to neighbors in low-dim space.
- **Warning**: Does NOT preserve global distances; only local structure is guaranteed.

**Key insight:** t-SNE is a visualization tool, not a preprocessing method. The 2D coordinates produced by t-SNE are not reliable features for a downstream model — they are not reproducible across runs (stochastic), they do not preserve global distances, and they are highly sensitive to perplexity. The only valid use of t-SNE output is human visual inspection.

**How to verify understanding:** A colleague uses t-SNE to reduce 500-dimensional embeddings to 2D, then trains a classifier on the 2D coordinates. Explain why this is wrong and what should be done instead.

**What trips people up:** Interpreting the scale of t-SNE clusters. The size and spacing of clusters in a t-SNE plot are not meaningful — t-SNE stretches and compresses the space to create separation. Two clusters that appear far apart in the plot may not be far apart in the original space.

---

## Cluster Evaluation

```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3).fit(X)
# Silhouette Score: Closer to 1 is better (well-separated clusters)
# Closer to -1 means the point may belong to a neighboring cluster
score = silhouette_score(X, model.labels_)
```

---

## Interview Questions

**1. "What are the limitations of K-Means?"**
> It assumes clusters are spherical and equal-sized. It is sensitive to outliers and requires the number of clusters $K$ to be predefined. It can also get stuck in local minima (mitigated by `n_init` multiple restarts).

**2. "How would you handle high-dimensional categorical data for clustering?"**
> Standard K-Means uses Euclidean distance, which isn't suitable for categories. Use **K-Modes** or embed the categories into a continuous space first using techniques like Entity Embeddings.

**3. "Why is PCA considered a feature extraction technique rather than selection?"**
> Feature selection keeps a subset of original features. PCA creates completely new features (linear combinations of the original ones), so it is "extracting" a new representation rather than selecting from the existing one.
