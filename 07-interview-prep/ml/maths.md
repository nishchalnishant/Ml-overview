---
module: Interview Prep
topic: Ml
subtopic: Maths
status: unread
tags: [interviewprep, ml, ml-maths]
---
# Maths for ML Interviews

---

## What This File Is For

Every topic is structured around the four questions that matter in an interview:
1. What the interviewer is actually testing
2. The reasoning structure — why first-principles thinkers approach it this way
3. The pattern in action — a worked example
4. Common traps — where people go wrong and why

---

## 1. Eigenvalues and Eigenvectors

**What the interviewer is testing:** Whether you understand eigenvectors as the natural coordinate system for a linear transformation — not just the equation $Av = \lambda v$. The question is usually a stepping stone to PCA, spectral methods, or understanding why certain matrices have special properties.

**The reasoning structure:** A matrix $A$ describes a linear transformation. Most vectors change both direction and magnitude when multiplied by $A$. Eigenvectors are the special directions that do not change direction — they are only scaled by factor $\lambda$ (the eigenvalue). This makes eigenvectors the "natural axes" of the transformation.

For a square matrix $A \in \mathbb{R}^{n \times n}$:
$$A\mathbf{v} = \lambda \mathbf{v}$$

Finding eigenvalues: solve $\det(A - \lambda I) = 0$ (characteristic polynomial).

Key properties:
- Symmetric matrices ($A = A^T$) have real eigenvalues and orthogonal eigenvectors — this is the spectral theorem, which is why PCA works cleanly
- $\sum \lambda_i = \text{tr}(A)$; $\prod \lambda_i = \det(A)$
- Eigenvalues of a covariance matrix are non-negative because covariance matrices are PSD

Power iteration (efficient for the largest eigenvalue):
```python
v = np.random.randn(n)
for _ in range(num_iters):
    v = A @ v
    v /= np.linalg.norm(v)   # leading eigenvector
eigenvalue = v @ A @ v        # Rayleigh quotient
```

**The pattern in action:** In PCA, you want the directions of maximum variance in your data. The covariance matrix $C = X^TX/(n-1)$ is symmetric. Its eigenvectors are orthogonal directions; the corresponding eigenvalues tell you how much variance lies along each direction. The first eigenvector (largest eigenvalue) points in the direction of greatest variance. This is not a coincidence — it follows directly from the definition of eigenvectors as the natural axes of the linear transformation that $C$ represents.

**Common traps:**
- Mixing up eigenvectors and principal components. PCA's principal components are the eigenvectors of the covariance matrix, but they become useful by projecting data onto them — computing $Xv$ for each eigenvector $v$.
- Assuming eigenvalues are always real. They are real for symmetric matrices. For arbitrary square matrices, eigenvalues can be complex.
- Saying "the eigenvalue is the variance explained." The eigenvalue of the covariance matrix equals the variance in the direction of its eigenvector — but only after the data is properly centered and the covariance is correctly computed.

---

## 2. SVD (Singular Value Decomposition)

**What the interviewer is testing:** Whether you understand SVD as the generalization of eigendecomposition to non-square matrices, and can reason about its consequences — rank-k approximation, dimensionality reduction, and the connection to eigendecomposition.

**The reasoning structure:** Eigendecomposition only works for square matrices. SVD decomposes any matrix $A \in \mathbb{R}^{m \times n}$:

$$A = U \Sigma V^T$$

- $U \in \mathbb{R}^{m \times m}$: left singular vectors (orthonormal columns) — the "output directions"
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $V \in \mathbb{R}^{n \times n}$: right singular vectors (orthonormal columns) — the "input directions"

Singular values are square roots of eigenvalues of $A^T A$ (or $A A^T$). So SVD and eigendecomposition are related but distinct.

**Truncated SVD** (rank-$k$ approximation):
$$A_k = U_k \Sigma_k V_k^T, \quad k \ll \min(m, n)$$

By the Eckart-Young theorem, this is the best rank-$k$ approximation in Frobenius norm:
$$A_k = \arg\min_{\text{rank}(B) \leq k} \|A - B\|_F$$

The residual error is $\|A - A_k\|_F^2 = \sum_{i>k} \sigma_i^2$ — the sum of squared discarded singular values.

```python
U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
A_k = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]   # rank-k approximation
```

**The pattern in action:** A movie rating matrix (users × movies) is sparse and noisy. SVD extracts $k$ latent factors — directions in both user space and movie space that explain most of the variance. The rank-$k$ approximation denoises the matrix by discarding directions that explain little variance. This is the foundation of collaborative filtering.

**Common traps:**
- Confusing SVD with eigendecomposition. SVD works on rectangular matrices and produces left and right singular vectors; eigendecomposition requires square matrices and produces one set of eigenvectors. The singular values of $A$ relate to eigenvalues of $A^TA$, not directly to eigenvalues of $A$.
- Not knowing that PCA is computed via SVD of the data matrix, not eigendecomposition of the covariance matrix — they produce the same result but SVD is numerically superior and does not require explicitly forming $X^TX$.

---

## 3. PCA and SVD Connection

**What the interviewer is testing:** Whether you can derive that PCA on centered data is equivalent to computing SVD of the data matrix — understanding both the mathematical equivalence and why SVD is preferred numerically.

**The reasoning structure:** PCA on centered data matrix $X \in \mathbb{R}^{n \times d}$ (zero-mean columns) has two equivalent formulations:

**Via covariance matrix eigendecomposition:**
$$C = \frac{1}{n-1} X^T X, \quad C = V \Lambda V^T$$
Principal components are eigenvectors $V$; eigenvalues $\Lambda$ are variances.

**Via SVD of the data matrix (numerically preferred):**
$$X = U \Sigma V^T \implies C = \frac{1}{n-1} V \Sigma^2 V^T$$
The principal components are still $V$. The $i$-th variance is $\lambda_i = \sigma_i^2 / (n-1)$.

SVD is preferred because forming $X^TX$ explicitly doubles the condition number — if $X$ has condition number $\kappa$, then $X^TX$ has condition number $\kappa^2$. For ill-conditioned data, eigendecomposition of $X^TX$ loses numerical precision that SVD preserves.

**Variance explained:**
$$\text{explained by top-}k = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{d} \sigma_i^2}$$

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=k)
X_reduced = pca.fit_transform(X)
print(pca.explained_variance_ratio_.cumsum())
```

**The pattern in action:** You have a dataset with 1,000 features that are highly correlated (genomics data). Computing $X^TX$ explicitly produces a nearly singular $1000 \times 1000$ matrix. Eigendecomposition on this matrix gives numerically unstable results for small eigenvalues. SVD of $X$ computes the same principal components with better numerical stability.

**Common traps:**
- Forgetting to center the data before PCA. Uncentered PCA finds directions of maximum variance including the mean offset, not directions of maximum variance around the mean.
- Interpreting PCA as removing noise. PCA removes low-variance directions — which are often noise, but not always. If important signal lies in low-variance directions (which can happen in adversarial settings), PCA discards it.

---

## 4. Jacobian and Hessian

**What the interviewer is testing:** Whether you can reason about derivatives of vector-valued functions (Jacobian) and second-order structure of the loss landscape (Hessian) — and connect these to how deep learning training works.

**The reasoning structure:**

**Jacobian:** For $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J \in \mathbb{R}^{m \times n}$:
$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

The Jacobian tells you how each output dimension changes with each input dimension. In backpropagation, the chain rule is the product of Jacobians across layers — the gradient of the loss with respect to the input of a layer is the Jacobian of that layer transposed times the upstream gradient.

**Hessian:** For $f: \mathbb{R}^n \to \mathbb{R}$, the Hessian $H \in \mathbb{R}^{n \times n}$:
$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

The Hessian encodes the local curvature of the loss surface:
- $H \succ 0$ (positive definite): local minimum — loss curves upward in all directions
- $H \prec 0$ (negative definite): local maximum
- Mixed eigenvalues: saddle point

**Condition number** $\kappa(H) = \sigma_{\max} / \sigma_{\min}$: the ratio of the steepest to shallowest direction. High condition number means gradient descent takes small steps in the shallow direction (to avoid overshooting the steep direction), causing slow convergence. Adaptive optimizers (Adam) address this by maintaining per-parameter learning rate estimates that roughly compensate for different curvatures.

Newton's method uses the Hessian directly:
$$\theta \leftarrow \theta - H^{-1} \nabla_\theta L$$
Quadratic convergence near the optimum, but $O(n^3)$ inversion cost makes it impractical for large networks ($n \sim 10^9$ parameters).

**The pattern in action:** A logistic regression model converges slowly with gradient descent. The features have very different scales — one feature has values in [0, 1] and another in [0, 10000]. The Hessian of the loss has a high condition number because the curvature along the high-scale feature direction is much greater than along the low-scale direction. Normalizing features to the same scale equalizes the condition number and speeds convergence dramatically.

**Common traps:**
- Treating the Hessian as computationally accessible for deep learning. For a model with $10^9$ parameters, the Hessian has $10^{18}$ entries — storing it is impossible. Fisher information matrix approximations and diagonal Hessian approximations are used in practice.
- Confusing saddle points with local minima. In high-dimensional loss landscapes, what appears to be a local minimum under a few directions is often a saddle point in other directions. Deep networks are believed to have very few poor local minima but many saddle points.

---

## 5. Positive Semi-Definite (PSD) Matrices

**What the interviewer is testing:** Whether you understand PSD as a property with structural implications — not just as a definition — and why it appears so often in ML (covariance matrices, kernel matrices, Hessians).

**The reasoning structure:** A symmetric matrix $A$ is positive semi-definite if:
$$\mathbf{x}^T A \mathbf{x} \geq 0 \quad \forall \mathbf{x} \in \mathbb{R}^n$$

Equivalently: all eigenvalues $\lambda_i \geq 0$.

**Why covariance matrices are PSD:**
$$\mathbf{x}^T C \mathbf{x} = \mathbf{x}^T \left(\frac{1}{n} X^T X\right) \mathbf{x} = \frac{1}{n} \|X\mathbf{x}\|^2 \geq 0$$
This is not a coincidence — any matrix of the form $B^TB$ is PSD.

**Why kernel matrices must be PSD (Mercer's theorem):** A kernel function $k(x_i, x_j)$ defines a valid inner product in some feature space if and only if the Gram matrix $K_{ij} = k(x_i, x_j)$ is PSD for any set of inputs. This guarantees the kernel corresponds to a geometrically consistent inner product space, which is what makes SVMs with kernel tricks mathematically valid.

**Why Hessians at local minima are PSD:** At a local minimum, the curvature is non-negative in every direction — moving in any direction increases the loss. This means $\mathbf{x}^T H \mathbf{x} \geq 0$, i.e., $H$ is PSD. A Hessian with negative eigenvalues indicates a saddle point or local maximum, not a minimum.

**The pattern in action:** You compute a kernel matrix for SVM and find the optimization fails to converge. Checking the eigenvalues of the kernel matrix reveals negative values — the kernel function you chose does not satisfy Mercer's condition. A common fix is adding a small multiple of the identity matrix ($K + \epsilon I$) to make it PSD by shifting all eigenvalues up. The resulting kernel is called a regularized kernel.

**Common traps:**
- Confusing PSD and PD. PSD allows zero eigenvalues; PD requires all eigenvalues strictly positive. A covariance matrix can be singular (PSD but not PD) if some features are exact linear combinations of others.
- Thinking any symmetric matrix is PSD. A symmetric matrix with negative eigenvalues is not PSD — for example, the matrix $[[-1, 0], [0, -1]]$ is symmetric but not PSD.

---

## 6. L1 and L2 Norms

**What the interviewer is testing:** Whether you can explain the geometric reason L1 produces sparsity and L2 does not — not just say "L1 causes sparsity." The mechanism is what matters.

**The reasoning structure:**

**L1 Norm:**
$$\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|$$

**L2 Norm:**
$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$$

**Why L1 produces sparsity:** The L1 constraint set (the ball $\|\mathbf{x}\|_1 \leq c$) is a polytope with corners at axis-aligned points. When you minimize an objective subject to an L1 ball constraint, the solution tends to land at a corner — where some coordinates are exactly zero. The L2 ball is round: the solution tends to land on the smooth surface, where no coordinate is exactly zero (just small). Geometrically, the corner structure of the L1 ball promotes sparse solutions; the smooth structure of the L2 ball promotes uniform shrinkage.

**General Lp norm:**
$$\|\mathbf{x}\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$$

As $p$ decreases from 2 to 1 to 0, the ball develops sharper corners at axis-aligned points, promoting increasing sparsity.

| Regularizer | Penalty | Effect |
| :--- | :--- | :--- |
| Lasso (L1) | $\lambda \|\mathbf{w}\|_1$ | Sparsity, feature selection |
| Ridge (L2) | $\frac{\lambda}{2}\|\mathbf{w}\|_2^2$ | Smooth weight shrinkage |
| Elastic Net | $\lambda_1 \|\mathbf{w}\|_1 + \frac{\lambda_2}{2}\|\mathbf{w}\|_2^2$ | Sparse + stable groups |

Bayesian connection: L1 regularization = MAP with Laplace prior; L2 = MAP with Gaussian prior.

**The pattern in action:** You have 10,000 features for a text classification task but expect only 100 to be relevant. L1 regularization (Lasso) produces a model where 9,900 feature weights are exactly zero — automatic feature selection. L2 regularization (Ridge) produces a model where all 10,000 feature weights are small but non-zero — the model uses all features, just with small contributions. For interpretability and feature identification, L1 is appropriate; for prediction accuracy when features are correlated, Ridge often works better.

**Common traps:**
- Saying "L1 works better for feature selection without explaining why." The geometric reason (corner structure of the L1 ball) is what interviewers want.
- Choosing L1 when features are highly correlated. L1 tends to arbitrarily select one feature from a correlated group and zero out the rest. Elastic Net handles this by combining L1's sparsity with L2's stability under correlation.

---

## 7. KL Divergence

**What the interviewer is testing:** Whether you understand KL divergence as an asymmetric measure and can explain why forward vs reverse KL produce different behavior — a directly testable consequence that distinguishes deep understanding from memorized facts.

**The reasoning structure:** KL divergence measures how distribution $P$ diverges from reference $Q$:

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_P\left[\log \frac{P(x)}{Q(x)}\right]$$

For continuous distributions: $D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} dx$

Properties:
- $D_{KL}(P \| Q) \geq 0$ (Gibbs' inequality), with equality iff $P = Q$
- Not symmetric: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$
- Relation to cross-entropy: $D_{KL}(P \| Q) = H(P, Q) - H(P)$

**Forward vs reverse KL — the asymmetry matters:**
- $D_{KL}(P \| Q)$ (forward, "inclusive divergence"): $Q$ must be non-zero wherever $P$ is non-zero. If $P(x) > 0$ and $Q(x) = 0$, the divergence is infinite. This forces $Q$ to spread to cover all modes of $P$ — it is **zero-avoiding** for $Q$. In variational inference, this overestimates the variance of the approximate posterior.
- $D_{KL}(Q \| P)$ (reverse, "exclusive divergence"): if $Q(x) > 0$ but $P(x) = 0$, the divergence is infinite. This forces $Q$ to concentrate on regions where $P$ is large — it is **zero-forcing**, producing mode-seeking behavior. Used in variational inference mean-field approximations.

For Gaussians (closed form):
$$D_{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

ML uses:
- VAE ELBO: $D_{KL}(q_\phi(z|x) \| p(z))$ — penalizes deviation of approximate posterior from prior
- Knowledge distillation: $D_{KL}(p_{\text{teacher}} \| p_{\text{student}})$
- RLHF KL penalty: $D_{KL}(\pi_\phi \| \pi_{\text{SFT}})$ — prevents reward hacking

**The pattern in action:** In variational autoencoders, the ELBO loss contains a KL term $D_{KL}(q_\phi(z|x) \| p(z))$ that encourages the encoder's approximate posterior to be close to the prior $\mathcal{N}(0, I)$. Using reverse KL here would make the encoder concentrate on modes — a posterior that is always near the prior, losing information. Using forward KL forces the encoder to be non-zero wherever the prior is non-zero, but the prior is Gaussian and already covers everything, so forward KL encourages the posterior to spread appropriately.

**Common traps:**
- Treating KL divergence as a distance. It is not symmetric and does not satisfy the triangle inequality. It is a divergence, not a metric.
- Forgetting the asymmetry when discussing loss functions. Cross-entropy minimization (which is equivalent to minimizing $D_{KL}(P_{\text{true}} \| P_{\text{model}})$) is not the same as minimizing $D_{KL}(P_{\text{model}} \| P_{\text{true}})$.

---

## 8. Lagrange Multipliers and KKT Conditions

**What the interviewer is testing:** Whether you can derive why optimization with constraints requires Lagrange multipliers, and apply KKT conditions to the SVM formulation — a common applied question.

**The reasoning structure:** Standard gradient descent finds unconstrained optima. When the feasible region is restricted by constraints, the unconstrained optimum may be infeasible, and we need to find the optimum on the constraint boundary.

**Equality constraints — Lagrange multipliers:**
Optimize $f(\mathbf{x})$ subject to $g(\mathbf{x}) = 0$:
$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda g(\mathbf{x})$$

At the optimum: $\nabla_\mathbf{x} \mathcal{L} = 0$ gives $\nabla f(\mathbf{x}^*) = \lambda \nabla g(\mathbf{x}^*)$.

Geometric meaning: at the constrained optimum, the gradient of the objective is parallel to the gradient of the constraint. If they were not parallel, you could move along the constraint surface to decrease the objective.

**Inequality constraints — KKT conditions:**
For $g_i(\mathbf{x}) \leq 0$:
$$\nabla f = \sum_i \mu_i \nabla g_i, \quad \mu_i \geq 0, \quad \mu_i g_i(\mathbf{x}^*) = 0$$

The last condition $\mu_i g_i = 0$ is complementary slackness: either $\mu_i = 0$ (constraint is inactive — optimum is in the interior) or $g_i(\mathbf{x}^*) = 0$ (constraint is active — optimum is on the boundary).

**SVM application:** The hard-margin SVM maximizes the margin $\frac{2}{\|w\|}$ subject to $y_i(w^Tx_i + b) \geq 1$ for all training points. The KKT conditions show that only the support vectors (training points on the margin boundary) have $\mu_i > 0$ — all other points have $\mu_i = 0$. This means the optimal hyperplane is defined entirely by the support vectors.

**The pattern in action:** You want to find the minimum-norm solution to an underdetermined linear system $Ax = b$ (more variables than equations, so infinitely many solutions). Minimize $\|x\|^2$ subject to $Ax = b$. The Lagrangian is $\mathcal{L} = \|x\|^2 - \lambda^T(Ax - b)$. Setting $\partial\mathcal{L}/\partial x = 0$ gives $2x = A^T\lambda$, so $x = A^T\lambda/2$. Substituting into $Ax = b$ gives $AA^T\lambda = 2b$, yielding the pseudoinverse solution $x = A^T(AA^T)^{-1}b$.

**Common traps:**
- Applying Lagrange multipliers to inequality constraints. Lagrange multipliers are for equality constraints; KKT conditions generalize to inequalities. The key addition is complementary slackness.
- Not checking constraint qualification (KKT regularity conditions). KKT conditions are necessary for optimality only when certain regularity conditions hold — in particular, the gradients of the active constraints must be linearly independent.

---

## 9. Information Theory Basics

**What the interviewer is testing:** Whether you can derive why cross-entropy is the right loss function for classification, and explain mutual information in terms of what it measures — not just recite the formulas.

**The reasoning structure:**

**Entropy:**
$$H(X) = -\sum_x P(x) \log P(x) = \mathbb{E}[-\log P(X)]$$

Entropy is the expected surprise of drawing from $P$. Uniform distribution has maximum entropy $H = \log K$ for $K$ classes — maximum uncertainty. A deterministic distribution (one outcome has probability 1) has $H = 0$ — no uncertainty.

**Cross-Entropy:**
$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

Cross-entropy measures the expected code length when you use $Q$ to encode outcomes distributed according to $P$. When $Q$ perfectly matches $P$, $H(P, Q) = H(P)$. When $Q$ mismatches $P$, $H(P, Q) > H(P)$.

Using cross-entropy as the training loss for classification: minimizing $H(P_{\text{true}}, P_{\text{model}})$ over the model parameters is equivalent to maximizing log-likelihood. The relationship:
$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$
Since $H(P)$ is fixed (it is the entropy of the true label distribution, not a function of model parameters), minimizing cross-entropy = minimizing KL divergence from the true distribution to the model.

**Mutual Information:**
$$I(X; Y) = D_{KL}(P(X, Y) \| P(X) P(Y)) = H(X) - H(X|Y)$$

$I(X; Y)$ measures how much knowing $Y$ reduces uncertainty about $X$. It is symmetric ($I(X; Y) = I(Y; X)$) and zero iff $X$ and $Y$ are independent.

**The pattern in action:** You have 10 classes and your model outputs a near-uniform distribution over all of them. The cross-entropy loss is $-\sum_x P_{\text{true}}(x) \log P_{\text{model}}(x) \approx \log 10 \approx 2.3$ nats. As training improves, the model concentrates probability on the correct class and cross-entropy decreases toward 0 (entropy of the true one-hot distribution). The loss function encodes the information-theoretic cost of using the wrong distribution to encode the truth.

**Common traps:**
- Confusing entropy and cross-entropy. Entropy $H(P)$ measures the uncertainty in $P$ — a property of the data. Cross-entropy $H(P, Q)$ measures the cost of approximating $P$ with $Q$ — a property of both the data and the model. Minimizing training loss minimizes cross-entropy, not entropy.
- Treating mutual information as only applicable to discrete distributions. Mutual information extends to continuous distributions via density integrals, though it is harder to estimate from samples.

---

## Quick Reference Table

| Concept | Formula | Key use |
| :--- | :--- | :--- |
| Eigenvalue | $A\mathbf{v} = \lambda\mathbf{v}$ | PCA, spectral methods |
| SVD | $A = U\Sigma V^T$ | Compression, LSA, PCA |
| Jacobian | $J_{ij} = \partial f_i / \partial x_j$ | Backprop, chain rule |
| Hessian | $H_{ij} = \partial^2 f / \partial x_i \partial x_j$ | Curvature, Newton's method |
| PSD | $\mathbf{x}^T A \mathbf{x} \geq 0$ | Covariance, kernels |
| L1 norm | $\sum_i |x_i|$ | Sparsity, feature selection |
| L2 norm | $\sqrt{\sum_i x_i^2}$ | Regularization, Euclidean distance |
| KL divergence | $\sum P \log(P/Q)$ | VAE, distillation, RLHF |
| Cross-entropy | $-\sum P \log Q$ | Classification loss |
| Entropy | $-\sum P \log P$ | Uncertainty, information content |
| Mutual information | $H(X) - H(X\|Y)$ | Feature selection, independence |

## Rapid Recall

### Symmetric matrices ($A = A^T$) have real eigenvalues and orthogonal eigenvectors
- Direct Answer: this is the spectral theorem, which is why PCA works cleanly
- Why: This matters because it tells you how to reason about symmetric matrices ($a = a^t$) have real eigenvalues and orthogonal eigenvectors.
- Pitfall: Don't answer "Symmetric matrices ($A = A^T$) have real eigenvalues and orthogonal eigenvectors" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: this is the spectral theorem, which is why PCA works cleanly

### $\sum \lambda_i = \text{tr}(A)$; $\prod \lambda_i = \det(A)$
- Direct Answer: $\sum \lambda_i = \text{tr}(A)$; $\prod \lambda_i = \det(A)$
- Why: This matters because it tells you how to reason about $\sum \lambda_i = \text{tr}(a)$; $\prod \lambda_i = \det(a)$.
- Pitfall: Don't answer "$\sum \lambda_i = \text{tr}(A)$; $\prod \lambda_i = \det(A)$" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: $\sum \lambda_i = \text{tr}(A)$; $\prod \lambda_i = \det(A)$

### Eigenvalues of a covariance matrix are non-negative because covariance matrices are PSD
- Direct Answer: Eigenvalues of a covariance matrix are non-negative because covariance matrices are PSD
- Why: This matters because it tells you how to reason about eigenvalues of a covariance matrix are non-negative because covariance matrices are psd.
- Pitfall: Don't answer "Eigenvalues of a covariance matrix are non-negative because covariance matrices are PSD" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Eigenvalues of a covariance matrix are non-negative because covariance matrices are PSD

### Mixing up eigenvectors and principal components. PCA's principal components are the eigenvectors of the covariance matrix, but they become useful by projecting data onto them
- Direct Answer: computing $Xv$ for each eigenvector $v$.
- Why: This matters because it tells you how to reason about mixing up eigenvectors and principal components. pca's principal components are the eigenvectors of the covariance matrix, but they become useful by projecting data onto them.
- Pitfall: Don't answer "Mixing up eigenvectors and principal components. PCA's principal components are the eigenvectors of the covariance matrix, but they become useful by projecting data onto them" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: computing $Xv$ for each eigenvector $v$.

### Assuming eigenvalues are always real. They are real for symmetric matrices. For arbitrary square matrices, eigenvalues can be complex.
- Direct Answer: Assuming eigenvalues are always real. They are real for symmetric matrices. For arbitrary square matrices, eigenvalues can be complex.
- Why: This matters because it tells you how to reason about assuming eigenvalues are always real. they are real for symmetric matrices. for arbitrary square matrices, eigenvalues can be complex..
- Pitfall: Don't answer "Assuming eigenvalues are always real. They are real for symmetric matrices. For arbitrary square matrices, eigenvalues can be complex." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Assuming eigenvalues are always real. They are real for symmetric matrices. For arbitrary square matrices, eigenvalues can be complex.

### Saying "the eigenvalue is the variance explained." The eigenvalue of the covariance matrix equals the variance in the direction of its eigenvector
- Direct Answer: but only after the data is properly centered and the covariance is correctly computed.
- Why: This matters because it tells you how to reason about saying "the eigenvalue is the variance explained." the eigenvalue of the covariance matrix equals the variance in the direction of its eigenvector.
- Pitfall: Don't answer "Saying "the eigenvalue is the variance explained." The eigenvalue of the covariance matrix equals the variance in the direction of its eigenvector" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: but only after the data is properly centered and the covariance is correctly computed.

### $U \in \mathbb{R}^{m \times m}$: left singular vectors (orthonormal columns)
- Direct Answer: the "output directions"
- Why: This matters because it tells you how to reason about $u \in \mathbb{r}^{m \times m}$: left singular vectors (orthonormal columns).
- Pitfall: Don't answer "$U \in \mathbb{R}^{m \times m}$: left singular vectors (orthonormal columns)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: the "output directions"

### $\Sigma \in \mathbb{R}^{m \times n}$
- Direct Answer: diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- Why: This matters because it tells you how to reason about $\sigma \in \mathbb{r}^{m \times n}$.
- Pitfall: Don't answer "$\Sigma \in \mathbb{R}^{m \times n}$" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$

### $V \in \mathbb{R}^{n \times n}$: right singular vectors (orthonormal columns)
- Direct Answer: the "input directions"
- Why: This matters because it tells you how to reason about $v \in \mathbb{r}^{n \times n}$: right singular vectors (orthonormal columns).
- Pitfall: Don't answer "$V \in \mathbb{R}^{n \times n}$: right singular vectors (orthonormal columns)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: the "input directions"

### Confusing SVD with eigendecomposition. SVD works on rectangular matrices and produces left and right singular vectors; eigendecomposition requires square matrices and produces one set of eigenvectors. The singular values of $A$ relate to eigenvalues of $A^TA$, not directly to eigenvalues of $A$.
- Direct Answer: Confusing SVD with eigendecomposition. SVD works on rectangular matrices and produces left and right singular vectors; eigendecomposition requires square matrices and produces one set of eigenvectors. The singular values of $A$ relate to eigenvalues of $A^TA$, not directly to eigenvalues of $A$.
- Why: This matters because it tells you how to reason about confusing svd with eigendecomposition. svd works on rectangular matrices and produces left and right singular vectors; eigendecomposition requires square matrices and produces one set of eigenvectors. the singular values of $a$ relate to eigenvalues of $a^ta$, not directly to eigenvalues of $a$..
- Pitfall: Don't answer "Confusing SVD with eigendecomposition. SVD works on rectangular matrices and produces left and right singular vectors; eigendecomposition requires square matrices and produces one set of eigenvectors. The singular values of $A$ relate to eigenvalues of $A^TA$, not directly to eigenvalues of $A$." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Confusing SVD with eigendecomposition. SVD works on rectangular matrices and produces left and right singular vectors; eigendecomposition requires square matrices and produces one…

### Not knowing that PCA is computed via SVD of the data matrix, not eigendecomposition of the covariance matrix
- Direct Answer: they produce the same result but SVD is numerically superior and does not require explicitly forming $X^TX$.
- Why: This matters because it tells you how to reason about not knowing that pca is computed via svd of the data matrix, not eigendecomposition of the covariance matrix.
- Pitfall: Don't answer "Not knowing that PCA is computed via SVD of the data matrix, not eigendecomposition of the covariance matrix" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: they produce the same result but SVD is numerically superior and does not require explicitly forming $X^TX$.

### Forgetting to center the data before PCA. Uncentered PCA finds directions of maximum variance including the mean offset, not directions of maximum variance around the mean.
- Direct Answer: Forgetting to center the data before PCA. Uncentered PCA finds directions of maximum variance including the mean offset, not directions of maximum variance around the mean.
- Why: This matters because it tells you how to reason about forgetting to center the data before pca. uncentered pca finds directions of maximum variance including the mean offset, not directions of maximum variance around the mean..
- Pitfall: Don't answer "Forgetting to center the data before PCA. Uncentered PCA finds directions of maximum variance including the mean offset, not directions of maximum variance around the mean." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Forgetting to center the data before PCA. Uncentered PCA finds directions of maximum variance including the mean offset, not directions of maximum variance around the mean.

### Interpreting PCA as removing noise. PCA removes low-variance directions
- Direct Answer: which are often noise, but not always. If important signal lies in low-variance directions (which can happen in adversarial settings), PCA discards it.
- Why: This matters because it tells you how to reason about interpreting pca as removing noise. pca removes low-variance directions.
- Pitfall: Don't answer "Interpreting PCA as removing noise. PCA removes low-variance directions" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: which are often noise, but not always. If important signal lies in low-variance directions (which can happen in adversarial settings), PCA discards it.

### $H \succ 0$ (positive definite): local minimum
- Direct Answer: loss curves upward in all directions
- Why: This matters because it tells you how to reason about $h \succ 0$ (positive definite): local minimum.
- Pitfall: Don't answer "$H \succ 0$ (positive definite): local minimum" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: loss curves upward in all directions

### $H \prec 0$ (negative definite)
- Direct Answer: local maximum
- Why: This matters because it tells you how to reason about $h \prec 0$ (negative definite).
- Pitfall: Don't answer "$H \prec 0$ (negative definite)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: local maximum

### Mixed eigenvalues
- Direct Answer: saddle point
- Why: This matters because it tells you how to reason about mixed eigenvalues.
- Pitfall: Don't answer "Mixed eigenvalues" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: saddle point

### Treating the Hessian as computationally accessible for deep learning. For a model with $10^9$ parameters, the Hessian has $10^{18}$ entries
- Direct Answer: storing it is impossible. Fisher information matrix approximations and diagonal Hessian approximations are used in practice.
- Why: This matters because it tells you how to reason about treating the hessian as computationally accessible for deep learning. for a model with $10^9$ parameters, the hessian has $10^{18}$ entries.
- Pitfall: Don't answer "Treating the Hessian as computationally accessible for deep learning. For a model with $10^9$ parameters, the Hessian has $10^{18}$ entries" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: storing it is impossible. Fisher information matrix approximations and diagonal Hessian approximations are used in practice.

### Confusing saddle points with local minima. In high-dimensional loss landscapes, what appears to be a local minimum under a few directions is often a saddle point in other directions. Deep networks are believed to have very few poor local minima but many saddle points.
- Direct Answer: Confusing saddle points with local minima. In high-dimensional loss landscapes, what appears to be a local minimum under a few directions is often a saddle point in other directions. Deep networks are believed to have very few poor local minima but many saddle points.
- Why: This matters because it tells you how to reason about confusing saddle points with local minima. in high-dimensional loss landscapes, what appears to be a local minimum under a few directions is often a saddle point in other directions. deep networks are believed to have very few poor local minima but many saddle points..
- Pitfall: Don't answer "Confusing saddle points with local minima. In high-dimensional loss landscapes, what appears to be a local minimum under a few directions is often a saddle point in other directions. Deep networks are believed to have very few poor local minima but many saddle points." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Confusing saddle points with local minima. In high-dimensional loss landscapes, what appears to be a local minimum under a few directions is often a saddle point in other directio…

### Confusing PSD and PD. PSD allows zero eigenvalues; PD requires all eigenvalues strictly positive. A covariance matrix can be singular (PSD but not PD) if some features are exact linear combinations of others.
- Direct Answer: Confusing PSD and PD. PSD allows zero eigenvalues; PD requires all eigenvalues strictly positive. A covariance matrix can be singular (PSD but not PD) if some features are exact linear combinations of others.
- Why: This matters because it tells you how to reason about confusing psd and pd. psd allows zero eigenvalues; pd requires all eigenvalues strictly positive. a covariance matrix can be singular (psd but not pd) if some features are exact linear combinations of others..
- Pitfall: Don't answer "Confusing PSD and PD. PSD allows zero eigenvalues; PD requires all eigenvalues strictly positive. A covariance matrix can be singular (PSD but not PD) if some features are exact linear combinations of others." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Confusing PSD and PD. PSD allows zero eigenvalues; PD requires all eigenvalues strictly positive. A covariance matrix can be singular (PSD but not PD) if some features are exact l…

### Thinking any symmetric matrix is PSD. A symmetric matrix with negative eigenvalues is not PSD
- Direct Answer: for example, the matrix $[[-1, 0], [0, -1]]$ is symmetric but not PSD.
- Why: This matters because it tells you how to reason about thinking any symmetric matrix is psd. a symmetric matrix with negative eigenvalues is not psd.
- Pitfall: Don't answer "Thinking any symmetric matrix is PSD. A symmetric matrix with negative eigenvalues is not PSD" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: for example, the matrix $[[-1, 0], [0, -1]]$ is symmetric but not PSD.

### Saying "L1 works better for feature selection without explaining why." The geometric reason (corner structure of the L1 ball) is what interviewers want.
- Direct Answer: Saying "L1 works better for feature selection without explaining why." The geometric reason (corner structure of the L1 ball) is what interviewers want.
- Why: This matters because it tells you how to reason about saying "l1 works better for feature selection without explaining why." the geometric reason (corner structure of the l1 ball) is what interviewers want..
- Pitfall: Don't answer "Saying "L1 works better for feature selection without explaining why." The geometric reason (corner structure of the L1 ball) is what interviewers want." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Saying "L1 works better for feature selection without explaining why." The geometric reason (corner structure of the L1 ball) is what interviewers want.

### Choosing L1 when features are highly correlated. L1 tends to arbitrarily select one feature from a correlated group and zero out the rest. Elastic Net handles this by combining L1's sparsity with L2's stability under correlation.
- Direct Answer: Choosing L1 when features are highly correlated. L1 tends to arbitrarily select one feature from a correlated group and zero out the rest. Elastic Net handles this by combining L1's sparsity with L2's stability under correlation.
- Why: This matters because it tells you how to reason about choosing l1 when features are highly correlated. l1 tends to arbitrarily select one feature from a correlated group and zero out the rest. elastic net handles this by combining l1's sparsity with l2's stability under correlation..
- Pitfall: Don't answer "Choosing L1 when features are highly correlated. L1 tends to arbitrarily select one feature from a correlated group and zero out the rest. Elastic Net handles this by combining L1's sparsity with L2's stability under correlation." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Choosing L1 when features are highly correlated. L1 tends to arbitrarily select one feature from a correlated group and zero out the rest. Elastic Net handles this by combining L1…

### $D_{KL}(P \| Q) \geq 0$ (Gibbs' inequality), with equality iff $P = Q$
- Direct Answer: $D_{KL}(P \| Q) \geq 0$ (Gibbs' inequality), with equality iff $P = Q$
- Why: This matters because it tells you how to reason about $d_{kl}(p \| q) \geq 0$ (gibbs' inequality), with equality iff $p = q$.
- Pitfall: Don't answer "$D_{KL}(P \| Q) \geq 0$ (Gibbs' inequality), with equality iff $P = Q$" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: $D_{KL}(P \| Q) \geq 0$ (Gibbs' inequality), with equality iff $P = Q$

### Not symmetric
- Direct Answer: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$
- Why: This matters because it tells you how to reason about not symmetric.
- Pitfall: Don't answer "Not symmetric" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$

### Relation to cross-entropy
- Direct Answer: $D_{KL}(P \| Q) = H(P, Q) - H(P)$
- Why: This matters because it tells you how to reason about relation to cross-entropy.
- Pitfall: Don't answer "Relation to cross-entropy" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: $D_{KL}(P \| Q) = H(P, Q) - H(P)$

### $D_{KL}(P \| Q)$ (forward, "inclusive divergence"): $Q$ must be non-zero wherever $P$ is non-zero. If $P(x) > 0$ and $Q(x) = 0$, the divergence is infinite. This forces $Q$ to spread to cover all modes of $P$
- Direct Answer: it is zero-avoiding for $Q$. In variational inference, this overestimates the variance of the approximate posterior.
- Why: This matters because it tells you how to reason about $d_{kl}(p \| q)$ (forward, "inclusive divergence"): $q$ must be non-zero wherever $p$ is non-zero. if $p(x) > 0$ and $q(x) = 0$, the divergence is infinite. this forces $q$ to spread to cover all modes of $p$.
- Pitfall: Don't answer "$D_{KL}(P \| Q)$ (forward, "inclusive divergence"): $Q$ must be non-zero wherever $P$ is non-zero. If $P(x) > 0$ and $Q(x) = 0$, the divergence is infinite. This forces $Q$ to spread to cover all modes of $P$" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: it is zero-avoiding for $Q$. In variational inference, this overestimates the variance of the approximate posterior.

### $D_{KL}(Q \| P)$ (reverse, "exclusive divergence"): if $Q(x) > 0$ but $P(x) = 0$, the divergence is infinite. This forces $Q$ to concentrate on regions where $P$ is large
- Direct Answer: it is zero-forcing, producing mode-seeking behavior. Used in variational inference mean-field approximations.
- Why: This matters because it tells you how to reason about $d_{kl}(q \| p)$ (reverse, "exclusive divergence"): if $q(x) > 0$ but $p(x) = 0$, the divergence is infinite. this forces $q$ to concentrate on regions where $p$ is large.
- Pitfall: Don't answer "$D_{KL}(Q \| P)$ (reverse, "exclusive divergence"): if $Q(x) > 0$ but $P(x) = 0$, the divergence is infinite. This forces $Q$ to concentrate on regions where $P$ is large" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: it is zero-forcing, producing mode-seeking behavior. Used in variational inference mean-field approximations.

### VAE ELBO: $D_{KL}(q_\phi(z|x) \| p(z))$
- Direct Answer: penalizes deviation of approximate posterior from prior
- Why: This matters because it tells you how to reason about vae elbo: $d_{kl}(q_\phi(z|x) \| p(z))$.
- Pitfall: Don't answer "VAE ELBO: $D_{KL}(q_\phi(z|x) \| p(z))$" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: penalizes deviation of approximate posterior from prior

### Knowledge distillation
- Direct Answer: $D_{KL}(p_{\text{teacher}} \| p_{\text{student}})$
- Why: This matters because it tells you how to reason about knowledge distillation.
- Pitfall: Don't answer "Knowledge distillation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: $D_{KL}(p_{\text{teacher}} \| p_{\text{student}})$

### RLHF KL penalty: $D_{KL}(\pi_\phi \| \pi_{\text{SFT}})$
- Direct Answer: prevents reward hacking
- Why: This matters because it tells you how to reason about rlhf kl penalty: $d_{kl}(\pi_\phi \| \pi_{\text{sft}})$.
- Pitfall: Don't answer "RLHF KL penalty: $D_{KL}(\pi_\phi \| \pi_{\text{SFT}})$" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: prevents reward hacking

### Treating KL divergence as a distance. It is not symmetric and does not satisfy the triangle inequality. It is a divergence, not a metric.
- Direct Answer: Treating KL divergence as a distance. It is not symmetric and does not satisfy the triangle inequality. It is a divergence, not a metric.
- Why: This matters because it tells you how to reason about treating kl divergence as a distance. it is not symmetric and does not satisfy the triangle inequality. it is a divergence, not a metric..
- Pitfall: Don't answer "Treating KL divergence as a distance. It is not symmetric and does not satisfy the triangle inequality. It is a divergence, not a metric." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Treating KL divergence as a distance. It is not symmetric and does not satisfy the triangle inequality. It is a divergence, not a metric.

### Forgetting the asymmetry when discussing loss functions. Cross-entropy minimization (which is equivalent to minimizing $D_{KL}(P_{\text{true}} \| P_{\text{model}})$) is not the same as minimizing $D_{KL}(P_{\text{model}} \| P_{\text{true}})$.
- Direct Answer: Forgetting the asymmetry when discussing loss functions. Cross-entropy minimization (which is equivalent to minimizing $D_{KL}(P_{\text{true}} \| P_{\text{model}})$) is not the same as minimizing $D_{KL}(P_{\text{model}} \| P_{\text{true}})$.
- Why: This matters because it tells you how to reason about forgetting the asymmetry when discussing loss functions. cross-entropy minimization (which is equivalent to minimizing $d_{kl}(p_{\text{true}} \| p_{\text{model}})$) is not the same as minimizing $d_{kl}(p_{\text{model}} \| p_{\text{true}})$..
- Pitfall: Don't answer "Forgetting the asymmetry when discussing loss functions. Cross-entropy minimization (which is equivalent to minimizing $D_{KL}(P_{\text{true}} \| P_{\text{model}})$) is not the same as minimizing $D_{KL}(P_{\text{model}} \| P_{\text{true}})$." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Forgetting the asymmetry when discussing loss functions. Cross-entropy minimization (which is equivalent to minimizing $D_{KL}(P_{\text{true}} \| P_{\text{model}})$) is not the sa…

### Applying Lagrange multipliers to inequality constraints. Lagrange multipliers are for equality constraints; KKT conditions generalize to inequalities. The key addition is complementary slackness.
- Direct Answer: Applying Lagrange multipliers to inequality constraints. Lagrange multipliers are for equality constraints; KKT conditions generalize to inequalities. The key addition is complementary slackness.
- Why: This matters because it tells you how to reason about applying lagrange multipliers to inequality constraints. lagrange multipliers are for equality constraints; kkt conditions generalize to inequalities. the key addition is complementary slackness..
- Pitfall: Don't answer "Applying Lagrange multipliers to inequality constraints. Lagrange multipliers are for equality constraints; KKT conditions generalize to inequalities. The key addition is complementary slackness." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Applying Lagrange multipliers to inequality constraints. Lagrange multipliers are for equality constraints; KKT conditions generalize to inequalities. The key addition is compleme…

### Not checking constraint qualification (KKT regularity conditions). KKT conditions are necessary for optimality only when certain regularity conditions hold
- Direct Answer: in particular, the gradients of the active constraints must be linearly independent.
- Why: This matters because it tells you how to reason about not checking constraint qualification (kkt regularity conditions). kkt conditions are necessary for optimality only when certain regularity conditions hold.
- Pitfall: Don't answer "Not checking constraint qualification (KKT regularity conditions). KKT conditions are necessary for optimality only when certain regularity conditions hold" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: in particular, the gradients of the active constraints must be linearly independent.

### Confusing entropy and cross-entropy. Entropy $H(P)$ measures the uncertainty in $P$
- Direct Answer: a property of the data. Cross-entropy $H(P, Q)$ measures the cost of approximating $P$ with $Q$ — a property of both the data and the model. Minimizing training loss minimizes cross-entropy, not entropy.
- Why: This matters because it tells you how to reason about confusing entropy and cross-entropy. entropy $h(p)$ measures the uncertainty in $p$.
- Pitfall: Don't answer "Confusing entropy and cross-entropy. Entropy $H(P)$ measures the uncertainty in $P$" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: a property of the data. Cross-entropy $H(P, Q)$ measures the cost of approximating $P$ with $Q$ — a property of both the data and the model. Minimizing training loss minimizes cro…

### Treating mutual information as only applicable to discrete distributions. Mutual information extends to continuous distributions via density integrals, though it is harder to estimate from samples.
- Direct Answer: Treating mutual information as only applicable to discrete distributions. Mutual information extends to continuous distributions via density integrals, though it is harder to estimate from samples.
- Why: This matters because it tells you how to reason about treating mutual information as only applicable to discrete distributions. mutual information extends to continuous distributions via density integrals, though it is harder to estimate from samples..
- Pitfall: Don't answer "Treating mutual information as only applicable to discrete distributions. Mutual information extends to continuous distributions via density integrals, though it is harder to estimate from samples." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Treating mutual information as only applicable to discrete distributions. Mutual information extends to continuous distributions via density integrals, though it is harder to esti…
