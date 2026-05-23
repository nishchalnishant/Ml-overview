---
module: Classical Ml
topic: Unsupervised Learning Snappy
subtopic: ""
status: unread
tags: [classicalml, ml, unsupervised-learning-snappy]
---
# Unsupervised Learning — 1-Page Cheat Sheet

## Algorithm Table

| Algorithm | Best for | Key hyperparameters | Watch out for |
| :--- | :--- | :--- | :--- |
| **K-Means** | Round-ish blobs, customer segmentation, fast | K, n_init, max_iter | Must choose K; outlier-sensitive; assumes spherical equal-size clusters |
| **Hierarchical (Agglomerative)** | Dendrograms, interpretable hierarchy, small N | linkage (ward/complete/average), distance metric | O(N²) memory; slow at large N |
| **DBSCAN** | Arbitrary shapes, noisy data, outlier detection | eps, min_samples | Struggles with varying density; sensitive to eps choice |
| **PCA** | Dimensionality reduction, decorrelation, preprocessing | n_components (or variance threshold) | Linear only; components are not interpretable features |
| **t-SNE** | 2D/3D visualization, explore local cluster structure | perplexity, learning_rate, n_iter | Not for global structure; non-deterministic; costly at large N |
| **UMAP** | Visualization at scale, local + some global structure | n_neighbors, min_dist, n_components | Non-deterministic; distances between clusters less meaningful than within |

---

## Choosing K (for K-Means)

- **Elbow method:** plot inertia (WCSS) vs K → pick the "elbow" (diminishing returns)
- **Silhouette score:** range [-1, 1]; higher = tighter + better separated; sweep K and pick peak
- Use **K-Means++** initialization (spreads centroids) to reduce bad restarts

---

## Dimensionality Reduction: Three Cameras

| | PCA | t-SNE | UMAP |
| :--- | :--- | :--- | :--- |
| Type | Linear | Non-linear | Non-linear |
| Speed | Fast | Slow | Medium–fast |
| Global structure | Yes | No | Partial |
| Deterministic | Yes | No (seed) | No (seed) |
| Use for | Preprocessing, de-correlation | Final 2D viz | Viz at scale |

**Rule:** PCA first for preprocessing/speed. t-SNE/UMAP only for visualization (never as production features).

---

## Evaluation (no labels)

| Metric | What it measures | Range | Better = |
| :--- | :--- | :--- | :--- |
| **Silhouette** | Cohesion vs separation | [-1, 1] | Higher |
| **Calinski-Harabasz** | Between/within cluster ratio | [0, ∞) | Higher |
| **Davies-Bouldin** | Cluster compactness + separation | [0, ∞) | Lower |

**Honest caveat:** Internal metrics measure geometry, not business value. Always sanity-check segments against domain knowledge.

---

## Key distinctions

**DBSCAN vs K-Means:** DBSCAN finds arbitrary shapes and labels outliers as noise. K-Means forces every point into a cluster and assumes circular shapes.

**PCA vs Autoencoders:** PCA is linear and deterministic. Autoencoders learn non-linear compression but are harder to interpret and need more data.

**Anomaly detection options:**
- **Isolation Forest** — randomly isolates points; anomalies = short path lengths; works in high-D
- **DBSCAN noise points** — points that don't belong to any dense region
- **Reconstruction error** (autoencoder) — high error = anomaly

## Flashcards

**Elbow method?** #flashcard
plot inertia (WCSS) vs K → pick the "elbow" (diminishing returns)

**Silhouette score?** #flashcard
range [-1, 1]; higher = tighter + better separated; sweep K and pick peak

**Use K-Means++ initialization (spreads centroids) to reduce bad restarts?** #flashcard
Use K-Means++ initialization (spreads centroids) to reduce bad restarts

**Isolation Forest?** #flashcard
randomly isolates points; anomalies = short path lengths; works in high-D

**DBSCAN noise points?** #flashcard
points that don't belong to any dense region

**Reconstruction error (autoencoder)?** #flashcard
high error = anomaly
