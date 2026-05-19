# Anomaly Detection

---

## The Problem Anomaly Detection Solves

**The problem**: In a stream of network packets, 99.9% are normal. In a transaction ledger, 0.1% are fraudulent. In a factory sensor log, equipment failure manifests as a brief, rare pattern buried in months of normal readings. You cannot supervise the detection of these anomalies in the traditional sense — you have few or no labeled examples of what "anomaly" looks like. All you have is a large corpus of what "normal" looks like.

**The core insight**: Normal data occupies a compact region of feature space — high-density, predictable, consistent with past observations. Anomalies fall outside that region. Instead of learning a decision boundary between two labeled classes, anomaly detection learns the shape of the normal region and flags anything that doesn't fit.

**Three distinct settings**:
- **Outlier detection**: The training set may already contain anomalies. Goal: identify which training examples are outliers.
- **Novelty detection**: Training set is clean (only normal examples). Goal: flag anomalies at inference time.
- **One-class classification**: Training set contains only the normal class. Goal: binary decision — normal vs not-normal — for new inputs.

---

## Isolation Forest

**The problem**: You have high-dimensional tabular data and need a scalable anomaly detector that doesn't require a density estimate or a distance computation.

**The core insight**: Anomalies are few and different. In a high-dimensional space, isolating an anomalous point from normal points requires very few random splits — it sits in a sparse region and any split quickly separates it. Normal points, packed into a dense cluster, require many more splits to isolate. Anomaly score = how easy it is to isolate the point.

**The mechanics**: Build an ensemble of isolation trees. Each tree: randomly select a feature, randomly select a split threshold within that feature's range, recursively partition. The path length from the root to a sample's leaf node is the number of splits required to isolate it. Short average path length across trees = anomaly.

Anomaly score:

$$s(x, n) = 2^{-E[h(x)] / c(n)}$$

where $h(x)$ is the average path length across trees and $c(n)$ is the expected path length for a random sample in a dataset of size $n$ (the normalization constant). Score near 1 = very anomalous. Score near 0.5 = ambiguous. Score below 0.5 = normal.

```python
from sklearn.ensemble import IsolationForest

clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
clf.fit(X_train)
scores = clf.decision_function(X_test)   # higher = more normal
labels = clf.predict(X_test)             # 1 = normal, -1 = anomaly
```

The `contamination` parameter sets the threshold — what fraction of training samples the model considers anomalous. It is a hyperparameter, not ground truth.

**What breaks**: Isolation Forest assumes anomalies are isolated. Clustered anomalies — a small but dense group of anomalous points — are harder to isolate than isolated normal points in sparse regions of the feature space. This is the "masking effect." Also performs poorly on very small datasets — the tree structure degenerates when n is small.

---

## One-Class SVM (OCSVM)

**The problem**: You have a clean training set of normal examples and want to learn an explicit decision boundary around them — flag anything outside the boundary at inference.

**The core insight**: In a kernel-induced feature space, find the smallest sphere (or hyperplane with maximum margin from the origin) that encloses most normal training data. Points outside the sphere are anomalies. The kernel trick allows this sphere to be non-linear in the original space.

**The mechanics**: Minimize $\frac{1}{2}\|w\|^2 - \rho + \frac{1}{\nu n}\sum \xi_i$ subject to $\langle w, \phi(x_i) \rangle \geq \rho - \xi_i$. The $\nu$ parameter is both an upper bound on the fraction of outliers and a lower bound on the fraction of support vectors.

```python
from sklearn.svm import OneClassSVM

clf = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
clf.fit(X_train)
labels = clf.predict(X_test)   # 1 = normal, -1 = anomaly
```

**What breaks**: OCSVM is a kernel method — it scales as O(n²) or worse. For large datasets it is computationally infeasible without approximations. The RBF kernel is sensitive to feature scale — always standardize inputs. OCSVM has no natural probability output; the decision function gives a signed distance from the boundary, not a calibrated score.

---

## Local Outlier Factor (LOF)

**The problem**: Anomalies can be global (far from all data) or local (normal in the global sense but surrounded by a much denser cluster). A point that lies in a medium-density region might be an anomaly if all its neighbors are in an extremely dense cluster.

**The core insight**: Anomaly score should be relative to the local neighborhood, not the global distribution. Compare the density of a point to the density of its neighbors. A point is anomalous if it is in a lower-density region than its neighbors.

**The mechanics**: For each point $o$, compute its reachability distance to neighbors, then its local reachability density (LRD) — inverse of average reachability distance to its k nearest neighbors. LOF is the ratio of average LRD of neighbors to LRD of $o$ itself.

$$\text{LOF}_k(o) = \frac{\sum_{p \in N_k(o)} \frac{\text{lrd}_k(p)}{\text{lrd}_k(o)}}{|N_k(o)|}$$

LOF ≈ 1: normal density region. LOF >> 1: the point's neighbors are in a much denser region — anomaly.

```python
from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
labels = clf.fit_predict(X)       # -1 = outlier
scores = -clf.negative_outlier_factor_
```

**What breaks**: LOF is O(n²) in computation — scales poorly to large datasets. The choice of k matters: small k makes LOF sensitive to local micro-structure; large k makes it more global. LOF does not support `.predict()` on new data after fitting — it is transductive. For novelty detection on new data, use `novelty=True`.

---

## Elliptic Envelope

**The problem**: Your data is approximately Gaussian and you want a principled statistical model of "normal" — flag points with low probability density under that model.

**The core insight**: Fit a multivariate Gaussian to the training data. The Mahalanobis distance from the mean, scaled by the covariance, defines how unusual a point is. Points with very high Mahalanobis distance are anomalies.

**The mechanics**: Estimate mean $\mu$ and covariance $\Sigma$ from the training data using the robust Minimum Covariance Determinant (MCD) estimator — which fits the Gaussian to the densest subset of the data, making the estimate robust to existing outliers. Flag points where $(x - \mu)^T \Sigma^{-1} (x - \mu)$ exceeds a threshold.

```python
from sklearn.covariance import EllipticEnvelope

clf = EllipticEnvelope(contamination=0.05, support_fraction=0.8)
clf.fit(X_train)
labels = clf.predict(X_test)
```

**What breaks**: Requires the data to be approximately Gaussian. Fails entirely on multimodal distributions or data with heavy tails. Degrades in high dimensions — the MCD estimator becomes unstable when the number of features approaches or exceeds the number of samples.

---

## Autoencoder-Based Anomaly Detection

**The problem**: Your normal data is high-dimensional — images, sequences, text embeddings. Classical density estimators and distance-based methods fail in high dimensions. You need a learned model of "what normal looks like."

**The core insight**: Train an autoencoder (encoder + decoder) on normal data only. The model learns to compress and reconstruct patterns it has seen. When an anomalous input arrives — one with a pattern the autoencoder never learned — it cannot reconstruct it accurately. High reconstruction error = anomaly.

**The mechanics**: Encoder maps input to a low-dimensional latent vector. Decoder reconstructs the input from the latent vector. Both trained to minimize reconstruction error on normal data only. At inference, compute reconstruction error for each input. Score distribution on normal validation data provides the threshold.

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Threshold: mean + k*std of reconstruction errors on normal validation data
recon_error = ((model(X_test) - X_test) ** 2).mean(dim=1)
anomalies   = recon_error > threshold
```

For time series: LSTM autoencoders capture temporal patterns. Anomalies produce high reconstruction error across a window of timesteps.

**What breaks**: The autoencoder can learn to reconstruct anomalies well if the bottleneck is too large — it memorizes the training set including noise. Too small a bottleneck and it can't reconstruct even normal inputs. The threshold (mean + k×std on normal validation data) requires labeled normal data; setting k is a design choice that trades precision for recall. Autoencoders can also fail if normal data is highly variable — the model's normal reconstruction error is already high, leaving no headroom to detect anomalies.

---

## HDBSCAN

**The problem**: You want to detect anomalies as part of a clustering analysis — points that don't belong to any cluster are noise, and noise is by definition anomalous.

**The core insight**: Dense regions of the feature space form clusters. Points in low-density regions between clusters are noise. HDBSCAN identifies these automatically, without requiring you to specify the number of clusters or a fixed density threshold.

**The mechanics**: Build a hierarchy of clusters from dense to sparse regions. Points that never belong to any stable cluster across the hierarchy are labeled as noise (cluster label = -1). Noise label = anomaly. Additionally, HDBSCAN computes an outlier score for every point, reflecting how weakly it belongs to its assigned cluster.

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
labels    = clusterer.fit_predict(X)
outlier_scores = clusterer.outlier_scores_   # higher = more anomalous
anomalies = labels == -1
```

**What breaks**: HDBSCAN is O(n log n) to O(n²) depending on implementation and dimensionality. Outlier scores are not calibrated probabilities. Tuning `min_cluster_size` and `min_samples` significantly affects what counts as noise. Not suitable for online/streaming anomaly detection — the entire dataset must be present to build the hierarchy.

---

## Evaluation

**The problem**: You can't evaluate an anomaly detector the same way you evaluate a classifier. Ground truth labels are rare. The positive class (anomaly) is vanishingly small. You need metrics that reflect actual usefulness.

**The core insight**: If you have any labeled anomalies (even a small holdout set), use precision-recall metrics — same reasoning as imbalanced classification. If you have no labels, evaluation must be domain-specific (manual review of flagged examples, or comparison to known events).

| Metric | When to use |
|---|---|
| ROC-AUC | Overall discrimination ability when labels are available |
| PR-AUC / Average Precision | Preferred for imbalanced anomaly datasets |
| F1 at threshold | When a specific operating threshold is required |
| Precision@K | Top-K flagged examples — useful when human review capacity is fixed |

**Threshold selection**: Set threshold from the distribution of anomaly scores on a normal validation set (e.g., 99th percentile). Never use test labels to pick the threshold.

---

## Production Patterns

**Calibrated scores**: Raw anomaly scores from IsolationForest or LOF have no absolute meaning across datasets. Calibrate to probabilities using a held-out labeled set before using scores as operational thresholds.

**Ensembling detectors**: Averaging or rank-aggregating scores from multiple detectors (IsolationForest + LOF + Autoencoder) reduces false positives without retraining. Different detectors catch different anomaly types.

**Streaming detection**: For real-time anomaly detection (network intrusion, sensor monitoring), use streaming-compatible methods: Robust Random Cut Forest (RRCF), Half-Space Trees, or online sliding-window LOF.

**Threshold drift**: The normal data distribution shifts over time — what was anomalous last year may be normal today. Periodically recalibrate thresholds against recent normal data or retrain the detector.

---

## Choosing an Algorithm

| Algorithm | Best for | Avoid when |
|---|---|---|
| Isolation Forest | High-dimensional tabular data, large datasets (> 10k rows) | Small datasets, clustered anomalies |
| One-Class SVM | Low-dimensional, clean training data, well-defined boundary | Large n (quadratic scaling) |
| LOF | Local anomalies, datasets with varying cluster density | Large n, streaming data |
| Elliptic Envelope | Near-Gaussian low-dimensional data | Multimodal distributions, high dimensions |
| Autoencoder | Images, sequences, unstructured or high-dimensional inputs | Small datasets (underfits), variable normal patterns |
| HDBSCAN | When clustering structure is also needed | Online detection, streaming data |
