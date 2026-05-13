# Maths for ML Interviews

---

# 1. Eigenvalues and Eigenvectors

For a square matrix $A \in \mathbb{R}^{n \times n}$, a non-zero vector $\mathbf{v}$ is an eigenvector if:

$$A\mathbf{v} = \lambda \mathbf{v}$$

where $\lambda$ is the corresponding eigenvalue.

**Finding eigenvalues:** solve $\det(A - \lambda I) = 0$ (characteristic polynomial).

**Properties:**
- Symmetric matrices ($A = A^T$) have real eigenvalues and orthogonal eigenvectors
- Eigenvalues of a covariance matrix are non-negative (PSD property)
- Sum of eigenvalues = trace$(A)$; product = det$(A)$

**In ML:**
- PCA: eigenvectors of the covariance matrix are the principal components; eigenvalues are the variance explained
- Spectral clustering: uses eigenvectors of graph Laplacian
- PageRank: stationary distribution is the leading eigenvector

**Power iteration** (efficient for the largest eigenvalue):
```python
v = np.random.randn(n)
for _ in range(num_iters):
    v = A @ v
    v /= np.linalg.norm(v)   # leading eigenvector
eigenvalue = v @ A @ v        # Rayleigh quotient
```

---

# 2. SVD (Singular Value Decomposition)

Any matrix $A \in \mathbb{R}^{m \times n}$ decomposes as:

$$A = U \Sigma V^T$$

- $U \in \mathbb{R}^{m \times m}$: left singular vectors (orthonormal columns)
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $V \in \mathbb{R}^{n \times n}$: right singular vectors (orthonormal columns)

**Truncated SVD** (rank-$k$ approximation):

$$A_k = U_k \Sigma_k V_k^T, \quad k \ll \min(m, n)$$

This is the best rank-$k$ approximation in Frobenius norm (Eckart-Young theorem):

$$A_k = \arg\min_{\text{rank}(B) \leq k} \|A - B\|_F$$

**Connection to eigenvalues:** singular values of $A$ are square roots of eigenvalues of $A^T A$ (or $A A^T$).

**In ML:** compression, denoising, LSA (Latent Semantic Analysis), collaborative filtering, whitening.

```python
U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
A_k = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]   # rank-k approximation
```

---

# 3. PCA and SVD Connection

PCA on centered data matrix $X \in \mathbb{R}^{n \times d}$ (zero-mean columns):

**Via covariance matrix:**

$$C = \frac{1}{n-1} X^T X, \quad C = V \Lambda V^T \quad \text{(eigendecomposition)}$$

**Via SVD (numerically preferred):**

$$X = U \Sigma V^T \implies C = \frac{1}{n-1} V \Sigma^2 V^T$$

Principal components are the columns of $V$. The $i$-th eigenvalue $\lambda_i = \sigma_i^2 / (n-1)$.

**Variance explained by $k$ components:**

$$\text{explained} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=k)
X_reduced = pca.fit_transform(X)
print(pca.explained_variance_ratio_.cumsum())
```

---

# 4. Jacobian and Hessian

## Jacobian

For $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J \in \mathbb{R}^{m \times n}$:

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

In vector notation: $J = \nabla_\mathbf{x} \mathbf{f}(\mathbf{x})$.

**In deep learning:** Jacobian of a layer maps input perturbations to output perturbations. Chain rule = product of Jacobians across layers.

## Hessian

For $f: \mathbb{R}^n \to \mathbb{R}$, the Hessian $H \in \mathbb{R}^{n \times n}$:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

The Hessian is symmetric when $f$ has continuous second partial derivatives.

**Curvature interpretation:**
- $H \succ 0$ (positive definite): local minimum — loss curves upward in all directions
- $H \prec 0$ (negative definite): local maximum
- Mixed eigenvalues: saddle point

**Newton's method** uses the Hessian directly:

$$\theta \leftarrow \theta - H^{-1} \nabla_\theta L$$

Quadratic convergence near the optimum, but $O(n^3)$ inversion cost makes it impractical for large networks.

**Condition number** $\kappa(H) = \sigma_{\max} / \sigma_{\min}$ — high condition number means some directions are much steeper than others, causing slow convergence for first-order methods.

---

# 5. Positive Semi-Definite (PSD) Matrices

A symmetric matrix $A$ is PSD if:

$$\mathbf{x}^T A \mathbf{x} \geq 0 \quad \forall \mathbf{x} \in \mathbb{R}^n$$

Equivalently: all eigenvalues $\lambda_i \geq 0$.

**Why covariance matrices are PSD:**

$$\mathbf{x}^T C \mathbf{x} = \mathbf{x}^T \left(\frac{1}{n} X^T X\right) \mathbf{x} = \frac{1}{n} \|X\mathbf{x}\|^2 \geq 0$$

**Kernel matrices must be PSD** (Mercer's theorem) — this ensures the kernel corresponds to a valid inner product in some feature space. Gram matrix $K_{ij} = k(x_i, x_j)$ must be PSD.

---

# 6. L1 and L2 Norms

## L1 Norm

$$\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|$$

Encourages sparsity: the L1 ball has corners at axis-aligned points, so solutions tend to land where some coordinates are exactly zero.

## L2 Norm

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$$

Smooth, rotationally invariant. L2 regularization shrinks all weights uniformly without zeroing them.

## Lp Norm (general)

$$\|\mathbf{x}\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$$

- $p \to 1$: L1 (sparsity)
- $p = 2$: Euclidean
- $p \to \infty$: $\max_i |x_i|$ (Chebyshev norm)

**Regularization penalties:**

| Regularizer | Penalty | Effect |
| :--- | :--- | :--- |
| Lasso (L1) | $\lambda \|\mathbf{w}\|_1$ | Sparsity, feature selection |
| Ridge (L2) | $\frac{\lambda}{2}\|\mathbf{w}\|_2^2$ | Smooth weight shrinkage |
| Elastic Net | $\lambda_1 \|\mathbf{w}\|_1 + \frac{\lambda_2}{2}\|\mathbf{w}\|_2^2$ | Sparse + stable |

---

# 7. KL Divergence

Measures how distribution $P$ diverges from reference $Q$:

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_P\left[\log \frac{P(x)}{Q(x)}\right]$$

For continuous distributions: $D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} dx$

**Properties:**
- $D_{KL}(P \| Q) \geq 0$ (Gibbs' inequality), with equality iff $P = Q$
- **Not symmetric**: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$
- Relation to cross-entropy: $D_{KL}(P \| Q) = H(P, Q) - H(P)$

**Forward vs reverse KL:**
- $D_{KL}(P \| Q)$: zero-avoiding — $Q$ spreads to cover all of $P$ (used in VI mean-field → overestimates variance)
- $D_{KL}(Q \| P)$: zero-forcing — $Q$ concentrates on modes of $P$

**In ML:**
- VAE: $D_{KL}(q_\phi(z|x) \| p(z))$ — penalizes deviation of approximate posterior from prior
- Knowledge distillation: $D_{KL}(p_{\text{teacher}} \| p_{\text{student}})$
- RL PPO: $D_{KL}(\pi_{\text{old}} \| \pi_{\text{new}})$ as trust-region constraint

**For Gaussians** (closed form):

$$D_{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

---

# 8. Lagrange Multipliers

Optimize $f(\mathbf{x})$ subject to equality constraint $g(\mathbf{x}) = 0$:

$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda g(\mathbf{x})$$

At the optimum: $\nabla_\mathbf{x} \mathcal{L} = 0$ and $\nabla_\lambda \mathcal{L} = 0$, giving:

$$\nabla f(\mathbf{x}^*) = \lambda \nabla g(\mathbf{x}^*)$$

Geometric meaning: at the constrained optimum, the gradient of the objective is parallel to the gradient of the constraint.

**KKT conditions** (inequality constraints $g(\mathbf{x}) \leq 0$):

$$\nabla f = \sum_i \mu_i \nabla g_i, \quad \mu_i \geq 0, \quad \mu_i g_i(\mathbf{x}^*) = 0$$

**In ML:**
- SVM hard margin: maximize margin $\frac{2}{\|w\|}$ subject to $y_i(w^T x_i + b) \geq 1$
- The dual form reveals that only support vectors ($\mu_i > 0$) define the decision boundary

---

# 9. Information Theory Basics

## Entropy

$$H(X) = -\sum_x P(x) \log P(x) = \mathbb{E}[-\log P(X)]$$

Maximum entropy for uniform distribution: $H = \log K$ for $K$ classes.

## Cross-Entropy

$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

Used as the training loss for classification: minimizing $H(P_{\text{true}}, P_{\text{model}})$ is equivalent to maximizing log-likelihood.

Relation: $H(P, Q) = H(P) + D_{KL}(P \| Q)$. Since $H(P)$ is fixed, minimizing cross-entropy = minimizing KL divergence.

## Mutual Information

$$I(X; Y) = D_{KL}(P(X, Y) \| P(X) P(Y)) = H(X) - H(X|Y)$$

How much knowing $Y$ reduces uncertainty about $X$.

---

# 10. Quick Reference Table

| Concept | Formula | Key use |
| :--- | :--- | :--- |
| Eigenvalue | $A\mathbf{v} = \lambda\mathbf{v}$ | PCA, spectral methods |
| SVD | $A = U\Sigma V^T$ | Compression, LSA |
| Jacobian | $J_{ij} = \partial f_i / \partial x_j$ | Backprop, sensitivity |
| Hessian | $H_{ij} = \partial^2 f / \partial x_i \partial x_j$ | Curvature, Newton's method |
| PSD | $\mathbf{x}^T A \mathbf{x} \geq 0$ | Covariance, kernels |
| L1 norm | $\sum_i \|x_i\|$ | Sparsity |
| L2 norm | $\sqrt{\sum_i x_i^2}$ | Regularization |
| KL divergence | $\sum P \log(P/Q)$ | VAE, distillation |
| Cross-entropy | $-\sum P \log Q$ | Classification loss |
| Entropy | $-\sum P \log P$ | Information content |
