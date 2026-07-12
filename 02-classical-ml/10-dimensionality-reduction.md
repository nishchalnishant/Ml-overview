---
module: Classical Ml
topic: Dimensionality Reduction
subtopic: ""
status: unread
tags: [classicalml, ml, dimensionality-reduction]
---
# Dimensionality Reduction

---

## TL;DR

| Method | Type | Linear | Supervised | Scalable | Preserves |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PCA | Global | Yes | No | Yes | Variance |
| Kernel PCA | Global | No | No | No | Kernel distances |
| ICA | Global | Yes | No | Yes | Statistical independence |
| LDA | Global | Yes | Yes | Yes | Class separability |
| t-SNE | Local | No | No | No | Local neighborhoods |
| UMAP | Local+Global | No | Optional | Yes | Topology |
| NMF | Global | Yes (non-neg) | No | Yes | Parts-based structure |
| Autoencoder | Global | No | Optional | Yes | Reconstruction |

**Core rule**: PCA for interpretability and preprocessing; t-SNE/UMAP for visualization; LDA when you have labels; autoencoders for complex non-linear structure.

---

## Why Dimensionality Reduction

### The Curse of Dimensionality

**The problem**: in high dimensions, geometric intuitions break. Adding more features often hurts rather than helps.

**Volume explosion**: a unit hypercube in $d$ dimensions has volume 1. A sphere inscribed inside it has volume proportional to $\pi^{d/2} / \Gamma(d/2 + 1)$, which goes to zero as $d \to \infty$. Almost all high-dimensional space is near the corners.

**Data sparsity**: to maintain the same sample density as 1D with $n$ points per unit length, in $d$ dimensions you need $n^d$ points. With 10 samples per unit in 1D, you need $10^{10}$ points in 10D.

**Distance concentration**: in high dimensions, distances between random points concentrate around the same value. Formally, for points drawn from a distribution with bounded moments:

$$\frac{\max_i d(q, x_i) - \min_i d(q, x_i)}{\min_i d(q, x_i)} \to 0 \quad \text{as } d \to \infty$$

This means nearest-neighbor search becomes meaningless — all points are approximately equidistant. K-NN classifiers, kernel methods, and any distance-based algorithm degrade silently.

**Practical thresholds**: effects become noticeable around $d \approx 10$; severe for $d > 100$.

### The Manifold Hypothesis

**The insight**: high-dimensional data rarely uses all available dimensions. Images of faces live in a roughly 50-dimensional manifold embedded in $10^6$-dimensional pixel space. The true degrees of freedom are far fewer than the ambient dimension.

**Definition**: a $k$-dimensional manifold is a space that looks locally like $\mathbb{R}^k$, even though it is embedded in higher-dimensional ambient space. The Swiss roll is a 2D manifold in 3D space.

**Why it matters**: if data lies on a $k$-dimensional manifold, any algorithm that does not exploit this structure wastes capacity on the $d - k$ irrelevant directions. Dimensionality reduction recovers the intrinsic coordinates.

**Evidence**: PCA often explains 90%+ of variance with far fewer components than features; random projections onto low-dimensional subspaces preserve pairwise distances (Johnson-Lindenstrauss lemma).

---

## PCA (Principal Component Analysis)

**The problem**: you have $n$ points in $\mathbb{R}^d$. You want to project them onto a $k$-dimensional subspace $(k \ll d)$ while losing as little information as possible.

**The core insight**: "information" can be operationalized as variance. The directions along which data varies the most carry the most signal; directions of near-zero variance are noise. Find the $k$ orthogonal directions that jointly capture maximum variance.

### Derivation: Variance Maximization

Center the data: $\tilde{X} = X - \bar{x}$. The covariance matrix is:

$$\Sigma = \frac{1}{n-1} \tilde{X}^T \tilde{X} \in \mathbb{R}^{d \times d}$$

Find a unit vector $w_1$ that maximizes projected variance:

$$w_1 = \argmax_{\|w\|=1} w^T \Sigma w$$

Using Lagrange multipliers: $\Sigma w = \lambda w$. So $w_1$ is the eigenvector of $\Sigma$ corresponding to the largest eigenvalue $\lambda_1$.

The second principal component $w_2$ maximizes variance subject to $w_2 \perp w_1$, giving the second eigenvector. By induction, the $k$-th PC is the $k$-th eigenvector of $\Sigma$.

### SVD Connection

PCA via eigenvectors of $\Sigma$ is numerically unstable for large $d$. The SVD approach is preferred:

$$\tilde{X} = U S V^T$$

where $U \in \mathbb{R}^{n \times n}$, $S \in \mathbb{R}^{n \times d}$ diagonal, $V \in \mathbb{R}^{d \times d}$. The columns of $V$ are the principal components; the singular values $\sigma_i = s_{ii}$ relate to eigenvalues by $\lambda_i = \sigma_i^2 / (n-1)$.

Truncated SVD retains only the top $k$ singular values/vectors and runs in $O(ndk)$ rather than $O(d^3)$.

### Explained Variance Ratio

$$\text{EVR}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i}$$

Typical practice: choose $k$ to explain 90-95% of variance. The **scree plot** shows eigenvalues in decreasing order; look for an "elbow" where the curve flattens. Below the elbow, components capture noise.

```python
from sklearn.decomposition import PCA
import numpy as np

pca = PCA().fit(X)

# Explained variance ratio
evr = pca.explained_variance_ratio_
cumulative = np.cumsum(evr)
k = np.searchsorted(cumulative, 0.95) + 1  # components for 95% variance

# Scree plot values
eigenvalues = pca.explained_variance_  # λ_i values
```

### PCA Whitening

Transform projected coordinates so each component has unit variance:

$$z_i = \frac{w_i^T x}{\sqrt{\lambda_i}}$$

Whitening decorrelates features and equalizes their scales. Used as preprocessing before neural networks, k-means, and other algorithms sensitive to feature scale. Caveat: amplifies noise in low-variance directions.

### When to Use PCA

- Preprocessing before algorithms sensitive to dimensionality (k-NN, SVM, k-means)
- Collinear features: PCA consolidates correlated inputs
- Visualization: first 2-3 PCs for quick scatter plots
- Compression: reduce storage while retaining most variance
- **Do not use** when features are categorical, when non-linear structure dominates, or when components need to be interpretable as original features

---

## Kernel PCA *(niche — research/theory interviews only)*

**The problem**: PCA is linear — it can only find structure in the span of the original features. Data with non-linear manifold structure (concentric circles, Swiss rolls) cannot be separated by any linear projection.

**The core insight**: map data to a high-dimensional feature space via a kernel trick (as in SVMs) so it becomes linearly separable there, then run PCA in that space — without ever computing the mapping explicitly.

**Common kernels**: RBF (local non-linear structure), polynomial (interactions up to degree $p$), sigmoid.

**Limitations**: $O(n^2)$ memory / $O(n^3)$ to decompose — impractical past $n \approx 10^4$. No direct out-of-sample projection (needs Nyström approximation). Results are sensitive to kernel choice with no universal guidance.

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1, fit_inverse_transform=True)
X_kpca = kpca.fit_transform(X)
```

---

## ICA (Independent Component Analysis) *(niche — research/theory interviews only)*

**The problem**: PCA finds uncorrelated components (zero covariance), which is weaker than statistical independence. Audio mixed from multiple microphones or EEG signals from multiple brain sources are independent, not just uncorrelated.

**The core insight**: by the Central Limit Theorem, a mixture of independent non-Gaussian sources looks more Gaussian than any individual source. So recovering the sources means finding the unmixing matrix that makes the outputs *least* Gaussian (measured via kurtosis or negentropy). **FastICA** is the standard algorithm, converging cubically via a fixed-point iteration on a non-Gaussianity measure.

### PCA vs ICA

| Property | PCA | ICA |
| :--- | :--- | :--- |
| **Components** | Uncorrelated | Independent |
| **Uses second-order stats** | Yes | No (higher-order) |
| **Unique up to** | Rotation | Sign + permutation |
| **Handles Gaussian sources** | Yes | No (Gaussian has no unique ICA solution) |
| **Primary use** | Variance compression | Source separation |

**Blind source separation**: ICA recovers original audio tracks from a mixture of microphones — the "cocktail party problem". Also used for EEG artifact removal (isolate eye-blink components).

**Preprocessing**: whiten data with PCA first (reduces ICA to finding an orthogonal rotation), then apply FastICA.

```python
from sklearn.decomposition import FastICA

ica = FastICA(n_components=3, random_state=42, max_iter=200)
S_estimated = ica.fit_transform(X_mixed)  # recovered sources
```

---

## LDA (Linear Discriminant Analysis)

**The problem**: PCA ignores class labels. The directions of maximum variance in the full dataset may be orthogonal to the directions that best separate classes. You need supervised dimensionality reduction.

**The core insight**: find a projection that simultaneously minimizes within-class scatter (points of the same class should cluster tightly) and maximizes between-class scatter (class centroids should be far apart).

### Scatter Matrices

Let $\mu_c$ be class $c$'s mean, $\mu$ the global mean, $n_c$ the number of samples in class $c$, and $C$ the number of classes.

**Within-class scatter** $S_W$: measures how spread out samples are within each class:

$$S_W = \sum_{c=1}^C \sum_{x \in \mathcal{C}_c} (x - \mu_c)(x - \mu_c)^T$$

**Between-class scatter** $S_B$: measures how separated class centers are:

$$S_B = \sum_{c=1}^C n_c (\mu_c - \mu)(\mu_c - \mu)^T$$

### Fisher's Criterion

Find projection $w$ that maximizes:

$$J(w) = \frac{w^T S_B w}{w^T S_W w}$$

This is a generalized eigenproblem: $S_B w = \lambda S_W w$, equivalently $S_W^{-1} S_B w = \lambda w$.

The top $k$ eigenvectors (discriminants) form the LDA projection. Note: with $C$ classes, $S_B$ has rank at most $C-1$, so LDA can produce at most $C-1$ discriminants regardless of $d$.

### Connection to Bayes Classifier

Under Gaussian class-conditionals with equal covariance (the LDA generative model assumption), LDA's decision boundary is identical to the Bayes-optimal linear classifier. LDA simultaneously does dimensionality reduction and classification.

**QDA** (Quadratic DA): allows class-specific covariances. More flexible but requires $O(C \cdot d^2)$ parameters; collapses to LDA when covariances are equal.

### LDA vs PCA

| Property | PCA | LDA |
| :--- | :--- | :--- |
| **Supervision** | No | Yes |
| **Max components** | $d$ | $C-1$ |
| **Objective** | Max variance | Max class separation |
| **Best for** | Preprocessing | Classification preprocessing |
| **Assumption** | None on labels | Gaussian, equal covariance |

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
```

---

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

**The problem**: PCA is linear and cannot unroll non-linear manifolds for visualization. You want a 2D map where nearby points in the original space remain nearby.

**The core insight**: model pairwise similarities in high dimensions as a probability distribution, then find a low-dimensional arrangement whose pairwise similarities match. The "crowding problem" (medium-distance points collapse onto short-distance points in 2D) is fixed by using a heavy-tailed t-distribution in low dimensions.

### The Algorithm

**Step 1 — High-dimensional affinities**: for each pair $(i, j)$, compute a symmetrized conditional probability proportional to a Gaussian centered at $x_i$:

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}, \quad p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

The bandwidth $\sigma_i$ is set per-point to achieve a target **perplexity** $= 2^{H(P_i)}$ where $H$ is the Shannon entropy of $P_i$. Binary search over $\sigma_i$ for each point.

**Step 2 — Low-dimensional affinities**: in 2D, use a Student t-distribution with 1 degree of freedom (Cauchy) instead of a Gaussian. Its heavy tail lets moderately similar points spread out in 2D, resolving the crowding problem.

**Step 3 — Minimize KL divergence** between the high- and low-dimensional affinity distributions $P$ and $Q$, via momentum-based gradient descent (typically 1000 iterations).

### Perplexity Parameter

Perplexity controls the effective number of neighbors each point considers. Intuition: it sets $\sigma_i$ so that each point's neighborhood distribution has a specific entropy.

- **Low perplexity (5-15)**: very local structure; may fragment clusters
- **High perplexity (30-50)**: broader neighborhoods; more global-like
- **Typical default**: 30-50

Rule of thumb: perplexity should be less than $n/3$. Results are sensitive to perplexity; run multiple values.

### What t-SNE Does NOT Preserve

**Distance is not preserved**: cluster separation in a t-SNE plot carries no meaning. Two clusters that look far apart may actually be close in the original space.

**Cluster size is not preserved**: denser clusters are artificially expanded; sparse clusters are compressed. This is a consequence of the adaptive $\sigma_i$.

**Global structure**: t-SNE optimizes local affinities. The relative positions of distant clusters are not reliable.

**Crowding problem**: even with the t-distribution fix, t-SNE in 2D struggles with datasets that have intrinsically $k > 2$ dimensional manifolds. Points from different parts of the manifold that happen to be equidistant get crammed together.

### Common Misuses

- Interpreting cluster distance as semantic distance — meaningless
- Interpreting cluster size as cluster density — misleading
- Running t-SNE once and treating it as ground truth — always check multiple random seeds and perplexities
- Using t-SNE for downstream ML tasks — it has no out-of-sample extension

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X)
# Note: fit_transform only; no transform() for new points
```

---

## UMAP (Uniform Manifold Approximation and Projection)

**The problem**: t-SNE is slow ($O(n^2)$ naively, $O(n \log n)$ with Barnes-Hut), has no out-of-sample extension, and discards global structure. You want a faster visualization method that is also usable as a general-purpose embedding.

**The core insight**: model the data as a weighted topological graph — a fuzzy simplicial complex that approximates the manifold. Find a low-dimensional representation whose graph matches. The uniform distribution assumption sets the metric on the manifold.

### Topological Foundations (Simplified)

**Step 1 — Build high-dimensional graph**: for each point $x_i$, find its $k$ nearest neighbors. Connect $x_i$ to each neighbor $x_j$ with a fuzzy edge weight:

$$v_{ij} = \exp\left(-\frac{d(x_i, x_j) - \rho_i}{\sigma_i}\right)$$

where $\rho_i$ is the distance to the nearest neighbor (ensures the closest neighbor always has weight 1) and $\sigma_i$ is calibrated so that $\sum_j v_{ij} = \log_2(k)$. Symmetrize: $w_{ij} = v_{ij} + v_{ji} - v_{ij} v_{ji}$.

**Step 2 — Build low-dimensional graph**: initialize with Laplacian eigenmaps or random, then define attraction/repulsion forces. Low-dimensional edge weights use a differentiable approximation to a t-distribution family:

$$q_{ij} = (1 + a \|y_i - y_j\|^{2b})^{-1}$$

$a$ and $b$ are fit to the `min_dist` parameter.

**Step 3 — Minimize cross-entropy** between the two graphs using stochastic gradient descent with negative sampling — much faster than t-SNE's gradient.

### Key Parameters

**`n_neighbors`** (default 15): controls local vs global structure tradeoff.
- Small (2-5): very local; may fragment global structure
- Large (50-200): global structure preserved; local detail lost

**`min_dist`** (default 0.1): minimum distance between points in the embedding.
- Small (0.0-0.1): tight clusters; good for cluster visualization
- Large (0.5-1.0): loose, spread-out embedding; good for exploring global topology

### UMAP vs t-SNE

| Property | t-SNE | UMAP |
| :--- | :--- | :--- |
| **Speed** | $O(n \log n)$ | $O(n^{1.14})$ approx — much faster |
| **Global structure** | Weak | Stronger |
| **Out-of-sample** | No | Yes (`transform()` supported) |
| **Supervised** | No | Yes (`y` parameter for semi-supervised) |
| **Theoretical basis** | Information theory | Riemannian geometry / topology |
| **Cluster separation** | Unreliable | More trustworthy, still imperfect |
| **Initialization** | Random | Spectral (more reproducible) |

```python
import umap

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)

# Out-of-sample: works natively
X_new_embedded = reducer.transform(X_new)
```

---

## NMF (Non-negative Matrix Factorization) *(niche — research/theory interviews only)*

**The problem**: PCA components can have negative values, making them hard to interpret as "parts". For data that is inherently additive (images, documents, audio spectra), you want a parts-based decomposition where each component is a non-negative building block.

**The core insight**: constrain both the basis vectors and their coefficients to be non-negative ($X \approx WH$, all non-negative). Data becomes a non-negative combination of parts — additive, not subtractive — which yields sparse, localized, interpretable representations. Solved via multiplicative update rules (Lee & Seung) that guarantee non-negativity and monotonic convergence to a local minimum (the problem is non-convex jointly in $W, H$). A KL-divergence variant of the loss works better for count data (text, genomics).

### Applications

**Topic modeling**: $X$ is a term-document matrix. $H$ rows are "topics" (word distributions); $W$ columns indicate topic mixtures per document. Unlike LDA (Bayesian topic model), NMF has no probabilistic interpretation but often gives comparable results faster.

**Image decomposition**: faces decompose into parts (eyes, nose, mouth) rather than the holistic eigenfaces from PCA. The non-negativity prevents cancellation.

**Audio source separation**: NMF on spectrograms separates instruments. Each component is a spectral pattern (basis) combined with a temporal activation (coefficient).

**Genomics**: gene expression data; components correspond to cell subtypes or biological processes.

```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=10, init='nndsvda', max_iter=400, random_state=42)
W = nmf.fit_transform(X)   # n x k: sample loadings
H = nmf.components_         # k x d: topic/part definitions
X_approx = W @ H
```

**Choosing $k$**: no equivalent to PCA's explained variance ratio; use reconstruction error vs $k$ plot or domain knowledge. Regularization ($\ell_1$ on $H$ for sparsity) helps interpretability.

---

## Autoencoders for Dimensionality Reduction

**The problem**: all methods above are either linear or have limited capacity for complex manifolds. Deep non-linear transformations can capture structure that no linear or kernel method can.

**The core insight**: train a neural network to compress data into a low-dimensional bottleneck, then reconstruct it. The bottleneck forces the network to learn a compact representation — any information not encodable in $k$ dimensions is discarded.

### Architecture

```
Input (d) → Encoder → Bottleneck (k) → Decoder → Reconstruction (d)
```

Encoder $f_\theta: \mathbb{R}^d \to \mathbb{R}^k$ and decoder $g_\phi: \mathbb{R}^k \to \mathbb{R}^d$. Minimize reconstruction loss:

$$\mathcal{L} = \|x - g_\phi(f_\theta(x))\|^2$$

The bottleneck representation $z = f_\theta(x)$ is the embedding.

### Vs Linear Methods

| Property | PCA | Autoencoder |
| :--- | :--- | :--- |
| **Linearity** | Linear | Arbitrary non-linear |
| **Global optimum** | Guaranteed | Local minima |
| **Speed** | Fast (SVD) | Slow (gradient descent) |
| **Interpretability** | Components are directions | Latent dims are opaque |
| **Generalization** | Strong | Can overfit |
| **Out-of-sample** | Trivial | Trivial (forward pass) |

PCA is the special case of a linear autoencoder with no activation functions (the learned subspace converges to the PCA subspace).

### VAE (Variational Autoencoder) as Probabilistic DR *(niche — research/theory interviews only)*

**The problem with standard autoencoders**: the latent space is not structured — nearby latent points can decode to unrelated outputs, so interpolation and generation fail.

**The core insight**: encode each input to a distribution rather than a point, and add a KL regularizer pulling that distribution toward a standard Gaussian prior. This makes the latent space smooth (nearby points decode similarly) and gives it a proper probabilistic structure. The mean of the encoded distribution serves as the embedding. Trained with the **reparameterization trick** ($z = \mu + \sigma \odot \epsilon$) so gradients can flow through the sampling step.

---

## Choosing the Right Method

| Criterion | Recommendation |
| :--- | :--- |
| **Need interpretable components** | PCA, NMF (non-negative data), ICA (source separation) |
| **Have class labels** | LDA (linear), supervised UMAP |
| **Visualization only** | t-SNE (small datasets), UMAP (any size) |
| **Downstream ML preprocessing** | PCA, LDA (classified), UMAP |
| **Non-linear structure, small data** | Kernel PCA, t-SNE |
| **Non-linear structure, large data** | UMAP, autoencoders |
| **Parts-based, non-negative data** | NMF |
| **Very complex manifold** | Autoencoders, VAE |
| **Need out-of-sample transform** | PCA, LDA, UMAP, autoencoders (not t-SNE) |
| **Speed is critical** | PCA (fastest), UMAP, then t-SNE, kernel PCA last |

### Decision Flow

```
Has labels?
├── Yes → LDA first; if non-linear, supervised UMAP
└── No → Is data non-negative?
          ├── Yes → NMF (if parts interpretation matters)
          └── No → Is goal visualization?
                    ├── Yes → UMAP (or t-SNE for small n)
                    └── No → Is structure linear?
                              ├── Yes → PCA
                              └── No → Autoencoder (large data) / Kernel PCA (small data)
```

### Complexity Summary

| Method | Fit time | Memory | Out-of-sample |
| :--- | :--- | :--- | :--- |
| PCA | $O(nd\min(n,d))$ | $O(d^2)$ | $O(dk)$ |
| Kernel PCA | $O(n^3)$ | $O(n^2)$ | Approx only |
| ICA | $O(ndk \cdot \text{iter})$ | $O(dk)$ | $O(dk)$ |
| LDA | $O(nd^2)$ | $O(d^2)$ | $O(dk)$ |
| t-SNE | $O(n \log n)$ | $O(n)$ | No |
| UMAP | $O(n^{1.14})$ approx | $O(n)$ | Yes |
| NMF | $O(ndk \cdot \text{iter})$ | $O((n+d)k)$ | $O(dk)$ |
| Autoencoder | Varies | Varies | $O(\text{forward pass})$ |

---

## Interview Questions

**Q1: Why does PCA fail on non-linear manifolds, and how would you detect this?**

PCA finds the best-fit linear subspace. If the data lies on a curved manifold (e.g., Swiss roll), the linear subspace it finds will "cut through" the manifold rather than unroll it. Detection: (1) compare reconstruction error of PCA to kernel PCA — if kernel PCA is substantially lower, there is non-linear structure; (2) check if residual dimensions still show systematic patterns (autocorrelation, non-Gaussianity) after PCA; (3) compare local and global pairwise distance preservation — PCA should preserve both if the manifold is truly linear.

---

**Q2: You have a dataset with 1000 features and 500 samples. Which dimensionality reduction methods are problematic and why?**

Kernel PCA requires an $n \times n = 500 \times 500$ matrix — feasible here but the eigenproblem cost scales as $O(n^3)$. LDA can only produce $C-1$ discriminants, independent of $n$ and $d$, but computing $S_W^{-1}$ is problematic when $d > n$ (covariance matrix is rank-deficient). Fix: regularized LDA or run PCA first to bring $d < n$. PCA via full eigendecomposition of $\Sigma \in \mathbb{R}^{d \times d} = \mathbb{R}^{1000 \times 1000}$ is $O(d^3) = O(10^9)$ — use truncated SVD on $\tilde{X}$ instead, which is $O(ndk)$ with $n < d$.

---

**Q3: Explain the crowding problem in t-SNE and how it is addressed.**

In high dimensions, a volume element at distance $r$ scales as $r^{d-1}$. In 2D, it only scales as $r$. This means medium-distance neighbors in high dimensions do not have enough space in 2D — they crowd around the near neighbors. PCA and early SNE models used Gaussian distributions in both spaces; the Gaussian's light tail cannot accommodate the number of medium-distance points that need to be nearby. t-SNE uses a Student t-distribution (one degree of freedom — a Cauchy) in the low-dimensional space. Its heavy tail allows many points to be placed at moderate distances while still matching the high-dimensional affinity structure. The gradient naturally pushes moderately close high-dimensional pairs farther apart in 2D, creating visible cluster separation.

---

**Q4: PCA is a special case of which broader methods? Describe two connections.**

(1) **SVD**: PCA on centered $X$ is equivalent to truncated SVD of the centered data matrix. The right singular vectors are the principal components; singular values are proportional to standard deviations along each component. This is how sklearn implements PCA.

(2) **Linear autoencoder**: a single-layer autoencoder with linear activations and bottleneck of size $k$, trained with MSE loss, converges to the same subspace as PCA (up to rotation within the subspace). The encoder weights span the PCA subspace; there is no unique solution because any rotation within the PCA subspace is equally valid.

(3) **Factor analysis**: PCA is MLE under the factor analysis model when noise variance is equal across dimensions (isotropic noise). Factor analysis generalizes to heteroskedastic noise.

---

**Q5: Why can't you use t-SNE embeddings as features for a downstream classifier?**

Three reasons: (1) No out-of-sample transform — you cannot embed test points the same way without rerunning the entire algorithm on the combined dataset. Rerunning changes the embeddings of training points. (2) Distance is not preserved — two points that are far apart in t-SNE may be close in the original space, so the classifier would learn spurious distance-based features. (3) Stochasticity — t-SNE embeddings vary across runs; features are not reproducible. Use PCA, UMAP (which has a `transform()` method), or autoencoders for DR as a preprocessing step before classification.

---

**Q6: When would you prefer NMF over PCA, and what is the main algorithmic challenge?**

Prefer NMF when: (1) data is non-negative by construction (pixel intensities, word counts, gene expression, audio spectra) and you need the decomposition to be physically interpretable — PCA components can have negative values that cancel each other out, making them hard to interpret; (2) you want a parts-based representation where each feature is a non-negative mixture of basis parts. The algorithmic challenge is non-convexity: the objective $\|X - WH\|_F^2$ is convex in $W$ alone or $H$ alone, but not jointly. Multiplicative updates converge to a local minimum; result depends on initialization. `nndsvda` initialization (SVD-based, non-negative double SVD) substantially improves results over random initialization.

---

**Q7: A t-SNE plot shows three clearly separated clusters. A colleague concludes the three groups are well-separated in the original space. What do you say?**

Cluster separation in t-SNE is not a reliable indicator of separation in the original space. The KL divergence objective heavily penalizes placing nearby high-dimensional points far apart in 2D (attractive forces dominate for close pairs). Repulsive forces push unrelated clusters apart regardless of their original distance. This means: (1) distinct clusters in t-SNE almost certainly correspond to distinct groups, but their inter-cluster distances are meaningless; (2) a single blob in t-SNE does not mean the data is unimodal — it may mean all structure is at a scale smaller than the perplexity. To assess actual separation, compute class-conditional distributions, run a classifier, or compute Mahalanobis distances in the original space.

For active-recall drilling on these terms, see [classical-ml-flashcards.md](classical-ml-flashcards.md).
