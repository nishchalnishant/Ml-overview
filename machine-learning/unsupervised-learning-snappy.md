# Unsupervised learning (deep-dive)

**Cold open:** No labels — only **geometry**, **density**, and **structure**. You’re clustering customers, compressing features, or hunting outliers. Same discipline as **anomaly detection** in infra: “normal” is a moving target.

**DevOps parallel:** Unsupervised work is like finding **traffic patterns** without anyone telling you which requests were “bad” — you define normal via **distance**, **density**, or **reconstruction error**.

---

## 1. Clustering families — pick the pitch before the bowler

### Q: Tradeoffs across major clustering styles?

| Style | Hero algorithm | Best for | Annoying weakness |
| :--- | :--- | :--- | :--- |
| **Centroid** | K-Means | Round-ish, similar-sized blobs | Must choose **K**; outliers bully centroids |
| **Density** | DBSCAN | Weird shapes + noise as “noise” | Struggles when density varies a lot |
| **Probabilistic** | GMM | Soft assignment, ellipses | Init-sensitive; needs assumptions |
| **Hierarchical** | Agglomerative | Dendrograms, interpretability | $O(N^3)$ naively — slow on big $N$ |

**MI strategy analogy:** K-Means is a **fixed field** (K zones). DBSCAN is **reading the pitch** — adapts where crowds thicken or thin.

---

## 2. K-Means — choosing K without astrology

### Q: How do you pick K?

**Direct answer:**
1. **Elbow (inertia vs. K):** Find where shrinking WCSS **diminishing returns** — the “elbow” is a vibe check, not a theorem.
2. **Silhouette:** Balance **cohesion** (tight within cluster) vs. **separation** (far from neighbors) — closer to **1** is happier.

**Pro move:** **K-Means++** initialization — spread initial centroids out — faster convergence, less “random bad day” sensitivity.

**Quick thought experiment:** *Silhouette is high but the business hates the segments.* What’s wrong? → **Metric / features** might not encode business value — math ≠ semantics.

---

## 3. PCA vs. t-SNE vs. UMAP — three different cameras

### Q: When do you use which?

- **PCA (linear):** Finds directions of **max variance** — fast, deterministic, **global** structure. Great preprocessing for visualization *or* de-correlating features.
- **t-SNE (non-linear):** Obsessed with **local** neighborhoods — gorgeous 2D plots, **not** a faithful global map; costly; random seed matters.
- **UMAP (non-linear):** Often **faster** than t-SNE at scale; tries to keep **local + some global** structure — default for big scatter plots.

**Remaster analogy:** PCA is **mono → stereo clarity** (linear remix). t-SNE is **vinyl warmth for neighbors** — can distort the whole album. UMAP tries to keep **both** the hook and the room tone.

---

## 4. Isolation Forest — why it scales to weird high-D data

### Q: Why Isolation Forest for high-dimensional anomalies?

**Direct answer:** Density in high-D is **cursed** — hard to define “normal volume.” Isolation Forest **randomly splits** features until points are isolated. **Anomalies** are few and off-manifold → they get isolated in **fewer splits** (short path length). No explicit density model required.

**Azure ops analogy:** Like finding the **one pod** that fails health checks faster than the herd — not by modeling “healthy traffic,” but by **isolating** weirdness with cheap random probes.

---

## 5. Evaluation without ground-truth labels

### Q: Metrics when nobody drew true clusters?

- **Silhouette** — [-1, 1], higher = tighter & separated (with distance assumptions).
- **Calinski-Harabasz** — between vs. within cluster spread ratio — higher often better.
- **Davies-Bouldin** — lower = clusters compact & far apart.

**Honest line for interviews:** “Internal metrics only tell **cohesion** — for business value you still need **downstream** KPIs or human review.”

---

> **Production defaults:** **UMAP** for big viz; **K-Means++** + silhouette sweeps for classic segmentation; always sanity-check segments against **business** meaning, not just curves.
