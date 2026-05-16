# Anomaly Detection

Anomaly detection identifies rare observations that differ significantly from the majority of data. Also called outlier detection, novelty detection, or one-class classification depending on context.

---

## Taxonomy

| Setting | Training Data | Goal |
|---------|--------------|------|
| Outlier detection | Contains anomalies | Flag anomalies in training set |
| Novelty detection | Clean (no anomalies) | Detect anomalies at inference |
| One-class classification | Only normal class | Binary: normal vs anomaly |

---

## Algorithms

### Isolation Forest

Randomly partition the feature space by splitting on random features at random thresholds. Anomalies are isolated in fewer splits (shorter path length from root).

**Anomaly score:** `s(x, n) = 2^{-E[h(x)] / c(n)}`  
where `h(x)` = path length, `c(n)` = expected path length for n samples.

```python
from sklearn.ensemble import IsolationForest

clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
clf.fit(X_train)
scores = clf.decision_function(X_test)   # higher = more normal
labels = clf.predict(X_test)             # 1 = normal, -1 = anomaly
```

**Strengths:** Scales to high-d data, no distance computation, handles irrelevant features well.  
**Weaknesses:** Poor for clustered anomalies (masking effect), not suitable for very small datasets.

---

### One-Class SVM (OCSVM)

Learns a decision boundary that separates normal data from the origin in a kernel-induced feature space. Finds a hyperplane maximizing the margin between normal data and the origin.

**Objective:** minimize `½‖w‖² - ρ + (1/νn)∑ξᵢ`

```python
from sklearn.svm import OneClassSVM

clf = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
clf.fit(X_train)
labels = clf.predict(X_test)   # 1 = normal, -1 = anomaly
```

- `nu`: upper bound on fraction of outliers (also lower bound on support vectors ratio)
- Works best in low-to-medium dimensions; kernel choice matters

---

### Local Outlier Factor (LOF)

Compares the local density of a point to that of its neighbors. Points in low-density regions relative to their neighbors get high LOF scores.

**LOF(k, o):** `(∑_{p ∈ N_k(o)} lrd_k(p)) / (|N_k(o)| × lrd_k(o))`

```python
from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
labels = clf.fit_predict(X)   # -1 = outlier
scores = -clf.negative_outlier_factor_
```

**Strengths:** Handles varying density, works well locally.  
**Weaknesses:** Expensive on large datasets (O(n²)), requires tuning `k`.

---

### Elliptic Envelope

Fits a Gaussian distribution to the data; flags points with low probability density. Robust version uses the Minimum Covariance Determinant (MCD) estimator.

```python
from sklearn.covariance import EllipticEnvelope

clf = EllipticEnvelope(contamination=0.05, support_fraction=0.8)
clf.fit(X_train)
labels = clf.predict(X_test)
```

**Assumption:** Data is approximately Gaussian. Fails for multi-modal distributions.

---

### Autoencoder-Based Anomaly Detection

Train an autoencoder on normal data. Anomalies produce high reconstruction error because the model hasn't seen similar patterns.

```python
import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, input_dim))
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Threshold on reconstruction error
recon_error = ((model(X) - X) ** 2).mean(dim=1)
anomalies = recon_error > threshold  # threshold = mean + k*std on val set
```

**For time series:** LSTM autoencoders capture temporal patterns; anomalies = high sequential reconstruction error.

---

### HDBSCAN (Density-Based)

Hierarchical extension of DBSCAN. Points classified as noise (`label = -1`) are outliers. Better than DBSCAN: automatically selects `eps`, handles variable density.

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
labels = clusterer.fit_predict(X)
outlier_scores = clusterer.outlier_scores_
```

---

## Evaluation

Ground truth is rare in anomaly detection — most datasets are highly imbalanced.

| Metric | When to Use |
|--------|-------------|
| ROC-AUC | Overall discrimination ability |
| PR-AUC / Average Precision | Preferred for imbalanced data |
| F1 at threshold | When a specific threshold is needed |
| Precision@K | Top-K flagged anomalies |

**Threshold selection:** Use `contamination` parameter or set threshold on held-out validation set with known labels. Never use test labels to pick threshold.

---

## Unsupervised vs Semi-Supervised

| | Unsupervised | Semi-Supervised |
|---|---|---|
| Labels needed | No | Only normal-class labels |
| Examples | IForest, LOF, OCSVM | Autoencoder trained on normal data |
| Common setting | EDA / exploration | Production monitoring |

---

## Production Patterns

**Score calibration:** Raw anomaly scores have no absolute meaning. Calibrate to probabilities using a held-out labeled set.

**Ensemble:** Average or rank-aggregate scores from multiple detectors (IForest + LOF + AE) to reduce false positives.

**Streaming anomaly detection:** Use half-space trees, RRCF (Robust Random Cut Forest), or online sliding-window LOF.

**Threshold drift:** Retrain or recalibrate thresholds periodically as the normal distribution shifts.

---

## When to Use Each

| Algorithm | Best for |
|-----------|---------|
| Isolation Forest | High-dimensional tabular data, large datasets |
| OCSVM | Low-dimensional, clean training data available |
| LOF | Local anomalies, varying cluster density |
| Elliptic Envelope | Gaussian data, few features |
| Autoencoder | Images, sequences, unstructured data |
| HDBSCAN | When clustering is also needed |

---

## Key Interview Points

- Isolation Forest is `O(n log n)` — preferred at scale.
- LOF uses local densities — better for datasets with uneven density.
- With no labels, use PR-AUC or ROC-AUC on a small labeled eval set.
- `contamination` is a hyperparameter, not ground truth; validate it.
- For time series anomalies, see `03-deep-learning/methods/time-series.md`.
