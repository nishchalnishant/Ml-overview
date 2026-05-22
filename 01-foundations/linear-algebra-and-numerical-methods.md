# Linear Algebra and Numerical Methods

---

**TL;DR:** Most of ML is matrix operations dressed up as learning. Understanding how matrices decompose, how gradients flow through them, and where floating-point arithmetic breaks down is what separates practitioners who debug effectively from those who cargo-cult hyperparameters. SVD is the Swiss army knife of numerical linear algebra. Condition numbers predict when your optimizer will suffer. Log-sum-exp saves your loss from going to -∞.

---

## Vectors and Matrices

### Dot Products and Norms

**The problem:** You need to measure similarity between embeddings, magnitude of weight vectors, or distance between data points — and you need the right norm for the job.

**The core insight:** Different norms penalize different structures. L1 promotes sparsity. L2 penalizes large weights smoothly. The Frobenius norm is the L2 norm flattened over all matrix entries.

**The mechanics:**

Dot product of vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

The cosine angle interpretation is why dot-product similarity works for embedding search — it measures directional alignment regardless of magnitude (once normalized).

**Vector norms:**

| Norm | Formula | ML Use |
|------|---------|--------|
| L1 ($\ell_1$) | $\sum_i \|x_i\|$ | Lasso, sparsity |
| L2 ($\ell_2$) | $\sqrt{\sum_i x_i^2}$ | Ridge, gradient norms, cosine sim |
| L∞ | $\max_i \|x_i\|$ | Adversarial perturbation budgets |
| Frobenius | $\sqrt{\sum_{i,j} A_{ij}^2}$ | Weight regularization on matrices |

**Frobenius norm:**

$$\|A\|_F = \sqrt{\text{tr}(A^\top A)} = \sqrt{\sum_{i,j} A_{ij}^2}$$

It equals the L2 norm of the vector of singular values: $\|A\|_F = \sqrt{\sigma_1^2 + \cdots + \sigma_r^2}$.

### Matrix Multiplication Complexity

**The problem:** Matrix multiplies dominate transformer forward passes. Knowing their cost guides batching decisions and hardware utilization analysis.

**Naive:** $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$ → $C \in \mathbb{R}^{m \times n}$ costs $O(mnk)$ multiplications. For square $n \times n$ matrices this is $O(n^3)$.

**In practice:** The $O(n^3)$ constant matters less than memory bandwidth. A $4096 \times 4096$ matrix multiply is compute-bound on a GPU because data stays in fast SRAM; a sequence of small matmuls may be memory-bound.

**Strassen:** Reduces to $O(n^{2.807})$ via recursive block decomposition. Almost never used in deep learning — numerical instability, poor cache behavior, and tiny real-world gains at practical sizes make cuBLAS's highly tuned $O(n^3)$ faster in practice.

### Broadcasting Rules

**The problem:** NumPy/PyTorch silently reshape tensors during operations, which causes subtle shape bugs that are hard to catch without understanding the rules.

**The mechanics:** Two tensors are broadcast-compatible if, when aligned from the right, each dimension pair is either equal or one of them is 1.

```
A: (32, 1, 64)
B:     (10, 64)   → (1, 10, 64) after left-padding
C: (32, 10, 64)   ← result shape
```

**What breaks:** Broadcasting never copies data (it uses strides), so the operation is memory-efficient but a wrong broadcast silently computes the wrong answer. Always assert shapes before operations in numerical code.

---

## Eigenvalues and Eigenvectors

### Definition and Geometric Intuition

**The problem:** You need to understand what a linear transformation does to space — does it stretch, compress, or rotate? Eigenvectors are the directions that don't rotate.

**Definition:** For matrix $A \in \mathbb{R}^{n \times n}$:

$$A\mathbf{v} = \lambda \mathbf{v}$$

$\mathbf{v}$ is the eigenvector (unchanged in direction), $\lambda$ is the eigenvalue (scale factor).

**Geometric intuition:** Apply $A$ to a sphere of unit vectors. The result is an ellipsoid. The eigenvectors are the axes of that ellipsoid; the eigenvalues are the axis lengths. Directions with large eigenvalues get stretched most — those are the directions that dominate the transformation.

### Eigendecomposition

For a diagonalizable matrix:

$$A = V \Lambda V^{-1}$$

where columns of $V$ are eigenvectors and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

**When it applies:** Any matrix with $n$ linearly independent eigenvectors. For **symmetric matrices** ($A = A^\top$) this always works and is cleaner:

$$A = Q \Lambda Q^\top$$

where $Q$ is orthogonal ($Q^\top Q = I$). Symmetric matrices have:
- Real eigenvalues (always)
- Orthogonal eigenvectors (always)
- Non-negative eigenvalues iff positive semidefinite

**What breaks:** Non-symmetric matrices may have complex eigenvalues and repeated eigenvalues may not yield a full eigenbasis (defective matrices).

### Connection to PCA

PCA computes the eigendecomposition of the sample covariance matrix $\Sigma = \frac{1}{n} X^\top X$. The principal components are the eigenvectors; the eigenvalues are the explained variance per direction. Projecting $X$ onto the top-$k$ eigenvectors gives the best rank-$k$ approximation in terms of explained variance — this is directly equivalent to truncated SVD (see below).

---

## Singular Value Decomposition (SVD)

### Full vs. Thin SVD

**The problem:** Eigendecomposition only applies to square matrices. SVD generalizes decomposition to any $m \times n$ matrix.

$$A = U \Sigma V^\top$$

| | $U$ | $\Sigma$ | $V^\top$ |
|--|-----|---------|---------|
| **Full SVD** | $m \times m$ | $m \times n$ | $n \times n$ |
| **Thin (economy) SVD** | $m \times r$ | $r \times r$ | $r \times n$ |

where $r = \min(m, n)$. Thin SVD is what `np.linalg.svd(A, full_matrices=False)` returns and what you almost always want.

**Singular values** $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0$ are the diagonal entries of $\Sigma$. Left singular vectors (columns of $U$) are orthonormal; right singular vectors (rows of $V^\top$) are orthonormal.

### Relation to Eigendecomposition

$$A^\top A = V \Sigma^\top U^\top U \Sigma V^\top = V (\Sigma^\top \Sigma) V^\top$$

So the right singular vectors of $A$ are the eigenvectors of $A^\top A$, and $\sigma_i^2$ are the corresponding eigenvalues. Similarly, $U$ diagonalizes $AA^\top$.

**For PCA:** The SVD of the centered data matrix $X$ directly gives principal components ($V$) and projected scores ($U\Sigma$) without explicitly forming the covariance matrix — numerically more stable for high-dimensional data.

### Low-Rank Approximation

The Eckart-Young theorem: the best rank-$k$ approximation of $A$ in both the Frobenius and spectral norms is:

$$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = U_k \Sigma_k V_k^\top$$

The approximation error is $\|A - A_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2$.

**ML uses:**
- **PCA:** Drop small singular values to reduce dimensionality
- **Collaborative filtering / matrix factorization:** Approximate user-item matrices with low-rank factors
- **LSA (Latent Semantic Analysis):** Low-rank approximation of TF-IDF term-document matrix
- **Weight compression:** Low-rank factorization of large weight matrices (LoRA is exactly this: $\Delta W = BA$ where $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, $r \ll \min(m,n)$)

### Moore-Penrose Pseudoinverse

$$A^+ = V \Sigma^+ U^\top$$

where $\Sigma^+$ replaces each non-zero diagonal $\sigma_i$ with $1/\sigma_i$ and leaves zeros as zeros.

The pseudoinverse gives the minimum-norm least-squares solution to $Ax = b$: $x^* = A^+ b$. When $A$ has full column rank, $A^+ = (A^\top A)^{-1} A^\top$ — the normal equations solution.

---

## Matrix Calculus

### Gradient of a Scalar with Respect to a Vector/Matrix

**Convention:** Numerator layout — gradient has the same shape as the denominator.

For $f: \mathbb{R}^n \to \mathbb{R}$, $\nabla_\mathbf{x} f \in \mathbb{R}^n$ where $(\nabla_\mathbf{x} f)_i = \partial f / \partial x_i$.

For $f: \mathbb{R}^{m \times n} \to \mathbb{R}$, $\nabla_W f \in \mathbb{R}^{m \times n}$ where $(\nabla_W f)_{ij} = \partial f / \partial W_{ij}$.

### Jacobian and Hessian

**Jacobian:** For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}, \quad J_{ij} = \frac{\partial f_i}{\partial x_j}$$

Backpropagation passes the upstream gradient as a row vector left-multiplied by the Jacobian: $\mathbf{g}^\top = \mathbf{g}_{\text{upstream}}^\top J$. This is why backprop is a sequence of vector-Jacobian products (VJPs), not full Jacobian materializations.

**Hessian:** For $f: \mathbb{R}^n \to \mathbb{R}$:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

The Hessian is symmetric for smooth $f$ (Schwarz's theorem). Its eigenvalues determine the curvature — positive definite Hessian means local minimum; indefinite means saddle point.

### Common Identities Used in ML

| Expression | Gradient |
|------------|---------|
| $\mathbf{a}^\top \mathbf{x}$ wrt $\mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^\top A \mathbf{x}$ wrt $\mathbf{x}$ | $(A + A^\top)\mathbf{x}$; $2A\mathbf{x}$ if $A$ symmetric |
| $\text{tr}(A^\top B)$ wrt $A$ | $B$ |
| $\log\det(A)$ wrt $A$ | $A^{-\top}$ |
| $\|A\|_F^2$ wrt $A$ | $2A$ |

### Linear Layer Gradient (dL/dW)

Forward: $Z = XW + b$, where $X \in \mathbb{R}^{n \times d_{\text{in}}}$, $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$.

Given upstream gradient $\frac{\partial L}{\partial Z} \in \mathbb{R}^{n \times d_{\text{out}}}$:

$$\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Z} \quad \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$$

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} W^\top \quad \in \mathbb{R}^{n \times d_{\text{in}}}$$

$$\frac{\partial L}{\partial b} = \mathbf{1}^\top \frac{\partial L}{\partial Z} \quad \in \mathbb{R}^{1 \times d_{\text{out}}}$$

**Intuition:** The weight gradient is an outer product summed over batch examples; the input gradient backpropagates signal through $W^\top$, the transpose (inverse direction of the forward transform).

---

## Special Matrices

### Positive Definite and Positive Semidefinite

A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is:

- **Positive definite (PD):** $\mathbf{x}^\top A \mathbf{x} > 0$ for all $\mathbf{x} \neq 0$
- **Positive semidefinite (PSD):** $\mathbf{x}^\top A \mathbf{x} \geq 0$ for all $\mathbf{x}$

**Tests for PD:**
1. All eigenvalues $> 0$
2. All leading principal minors $> 0$ (Sylvester's criterion)
3. Cholesky factorization $A = LL^\top$ exists with positive diagonal on $L$

**Geometric intuition:** PD matrices define ellipsoids. The quadratic form $\mathbf{x}^\top A \mathbf{x}$ is a bowl with no flat or downward directions. Loss functions with PD Hessian have a unique minimum.

**ML relevance:** Covariance matrices are always PSD. Gram matrices ($X^\top X$) are PSD. A PD Hessian at a critical point certifies a local minimum.

### Orthogonal Matrices

$Q^\top Q = QQ^\top = I$ (square). Rectangular $Q$ with $Q^\top Q = I$ is called isometric or semi-orthogonal.

**Properties:**
- $\det(Q) = \pm 1$
- $\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$ — length-preserving (isometry)
- Eigenvalues lie on the unit circle in $\mathbb{C}$

**ML relevance:** Householder reflections and Givens rotations in QR decomposition. Orthogonal weight initialization (preserve gradient norms). Rotary position embeddings (RoPE) apply orthogonal rotations to query/key vectors.

### Covariance Matrices

For data matrix $X \in \mathbb{R}^{n \times d}$ (centered):

$$\Sigma = \frac{1}{n} X^\top X \in \mathbb{R}^{d \times d}$$

Always PSD. Diagonal entries are feature variances; off-diagonal entries are covariances. Its eigendecomposition is PCA. Whitening transforms data to have identity covariance: $\tilde{X} = X \Sigma^{-1/2}$, which improves conditioning and can accelerate optimization.

---

## Numerical Stability

### Floating-Point Arithmetic

IEEE 754 double precision: 64-bit representation with 52 bits of mantissa → machine epsilon $\epsilon \approx 2.2 \times 10^{-16}$.

**Catastrophic cancellation:** When two nearly equal large numbers are subtracted, the relative error of the result explodes.

```python
# Bad: catastrophic cancellation
x = 1e15 + 1.0 - 1e15  # → 0.0, should be 1.0

# Better: reorder to avoid subtracting large near-equal quantities
```

**Overflow/underflow:** $e^{1000}$ overflows float32 (max $\approx 3.4 \times 10^{38}$). Probabilities raised to large powers underflow to zero. Always work in log-space when multiplying many probabilities.

### Log-Sum-Exp Trick

Computing $\log \sum_i e^{x_i}$ directly overflows. The stable form:

$$\log \sum_{i} e^{x_i} = c + \log \sum_{i} e^{x_i - c}, \quad c = \max_i x_i$$

Subtracting the max ensures the largest term is $e^0 = 1$ (no overflow) and small terms underflow gracefully to zero (contributing negligibly anyway).

### Stable Softmax

**Naive:** $\text{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$ — overflows for large $x_i$.

**Stable:** Subtract $c = \max_i x_i$ before exponentiating:

$$\text{softmax}(x)_i = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}$$

Mathematically identical (the $c$ cancels), numerically safe.

### Stable Cross-Entropy

Combining softmax + cross-entropy in one pass avoids computing intermediate softmax probabilities that can underflow:

$$L = -x_{y} + \log\sum_j e^{x_j} = -x_{y} + c + \log\sum_j e^{x_j - c}$$

PyTorch's `F.cross_entropy` does this internally. Never compute `log(softmax(x))` separately — use `log_softmax` which applies the same cancellation.

---

## Condition Numbers

### Definition

The condition number of a matrix $A$ (for linear systems, using the 2-norm):

$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}$$

where $\sigma_{\max}, \sigma_{\min}$ are the largest and smallest singular values.

**Interpretation:** If you perturb the right-hand side $b$ in $Ax = b$ by a relative amount $\epsilon$, the solution $x$ can change by up to $\kappa(A) \cdot \epsilon$ relatively. A condition number of $10^6$ means you lose 6 digits of precision.

### Connection to Unstable Gradients

**Ill-conditioned feature matrices:** If the input features $X$ have very different scales, $X^\top X$ has a large condition number. Gradient steps that reduce loss in one direction may wildly overshoot in another. This is why feature normalization (standardization) is not cosmetic — it directly improves conditioning.

**Ill-conditioned weight matrices:** If $W$ has a large $\kappa(W)$, gradients propagating through $W^\top$ in backprop amplify in some directions and vanish in others. This is the mechanism behind the vanishing/exploding gradient problem in deep networks.

**Rule of thumb:** $\kappa \approx 1$ is ideal. $\kappa > 10^6$ in float32 (which has ~7 decimal digits of precision) means the system is effectively singular.

---

## Ill-Posed Problems and Regularization

### Why Linear Systems Are Ill-Posed

A system $Ax = b$ is ill-posed when $A$ is rank-deficient or near-singular: either no solution exists, or infinitely many solutions exist, or the solution is wildly sensitive to perturbations in $b$.

In ML: the OLS estimator $\hat{w} = (X^\top X)^{-1} X^\top y$ breaks when $X^\top X$ is singular (multicollinearity, more features than samples).

### Tikhonov Regularization (Ridge Regression)

**Regularization as stabilization:** Add a multiple of the identity to ensure $X^\top X + \lambda I$ is always invertible:

$$\hat{w}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y$$

**Derivation from optimization:**

$$\hat{w} = \arg\min_w \|Xw - y\|_2^2 + \lambda \|w\|_2^2$$

Setting gradient to zero: $2X^\top(Xw - y) + 2\lambda w = 0$ → $(X^\top X + \lambda I)w = X^\top y$.

**Effect on singular values:** Via SVD, $X = U\Sigma V^\top$:

$$\hat{w}_{\text{ridge}} = V \text{diag}\!\left(\frac{\sigma_i}{\sigma_i^2 + \lambda}\right) U^\top y$$

versus OLS: $V \text{diag}(1/\sigma_i) U^\top y$. Ridge shrinks contributions from small singular values (noisy directions) toward zero rather than amplifying them. This directly controls the condition number: $\kappa_{\text{eff}} = (\sigma_{\max}^2 + \lambda)/(\sigma_{\min}^2 + \lambda)$.

### Moore-Penrose Pseudoinverse as $\lambda \to 0^+$

The pseudoinverse $A^+$ is the limit of Tikhonov regularization as $\lambda \to 0$. It zeros out contributions from directions with $\sigma_i = 0$ (or numerically near zero below a threshold). This gives the minimum-norm least-squares solution.

---

## Computational Complexity and Memory Layout

### Matrix Multiply Complexity

| Algorithm | Complexity | Notes |
|-----------|-----------|-------|
| Naive | $O(n^3)$ | Standard; highly optimized in BLAS |
| Strassen | $O(n^{2.807})$ | Rarely used in practice |
| Best known | $O(n^{2.371...})$ | Theoretical; impractical constant |

For rectangular $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$: cost is $O(mnk)$. In transformers, attention is $O(n^2 d)$ in sequence length $n$, context length dominates for large $n$.

### Memory Layout

**Row-major (C order):** Row elements are contiguous in memory. $A[i,:]$ is a contiguous read; $A[:,j]$ strides across memory.

**Column-major (Fortran/BLAS order):** Column elements are contiguous. BLAS routines are designed for column-major; NumPy defaults to row-major but transposes are zero-copy (just swap strides).

**Cache-friendly operations:**
- Access memory in stride-1 (contiguous) patterns
- A naive $C = AB$ loop over $(i, j, k)$ causes cache misses on $B$'s column accesses in row-major layout → reorder to $(i, k, j)$ or use blocked (tiled) matmul
- GPU tensor cores operate on tiles; padding dimensions to multiples of 8 or 16 maximizes utilization

**Practical impact:**
- `x.contiguous()` in PyTorch forces row-major layout; many ops silently call this, triggering a copy
- Transpose of a large matrix is not free in terms of subsequent operation performance even if the `T` itself is free (zero-copy)
- Batch matrix multiply (`torch.bmm`) is more cache-friendly than a Python loop over batch dimension

---

## Interview Questions

**Q1: Why does PCA via SVD on the data matrix $X$ produce the same result as eigendecomposition of the covariance matrix $X^\top X$?**

Because $X^\top X = V \Sigma^2 V^\top$ from the SVD of $X$. The right singular vectors $V$ are exactly the eigenvectors of $X^\top X$, with eigenvalues $\sigma_i^2$. SVD on $X$ directly is preferred because it is numerically more stable — forming $X^\top X$ explicitly squares the condition number.

---

**Q2: When would you prefer L1 regularization over L2, and why does L1 produce sparse solutions?**

L1 produces sparsity because its gradient is $\pm 1$ regardless of weight magnitude — it applies constant "pressure" toward zero, so small weights get pushed all the way to zero. L2's gradient is $2w$ — it weakens as the weight shrinks, so it shrinks weights toward (but not to) zero. Prefer L1 when you believe the true signal is sparse (few features matter), e.g., in feature selection. Prefer L2 when all features contribute and you just want to prevent large weights. In practice, ElasticNet (L1 + L2) handles correlated features better than L1 alone.

---

**Q3: What does it mean for a matrix to be positive definite, and why does it matter for optimization?**

A symmetric matrix is positive definite if $\mathbf{x}^\top A \mathbf{x} > 0$ for all nonzero $\mathbf{x}$, equivalently if all eigenvalues are positive. For optimization: the Hessian of a loss function is PD at a critical point iff it is a strict local minimum (all curvatures positive). Gradient descent converges in at most $\kappa(H)$ condition-number steps proportional to the Hessian condition number. Newton's method uses the Hessian inverse to normalize curvature, achieving quadratic convergence near minima — but requires $H$ to be PD (no saddle point directions).

---

**Q4: Why is the log-sum-exp trick necessary and how does it work?**

Direct computation of $\log(\sum_i e^{x_i})$ overflows float32 when $x_i \gtrsim 88$. The trick exploits the identity $\log \sum e^{x_i} = c + \log \sum e^{x_i - c}$ for any constant $c$ — mathematically unchanged, numerically safe with $c = \max_i x_i$. This same mechanism underlies stable softmax (subtract max before normalizing) and numerically stable cross-entropy loss. Anytime you're computing log-probabilities or normalizing exponentials, use this.

---

**Q5: What is the condition number, and how does it relate to training instability?**

The condition number $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ measures how much a linear system amplifies input perturbations. In ML, ill-conditioned input features ($\kappa(X^\top X) \gg 1$) cause loss landscapes with elongated contours — gradient descent oscillates across the narrow direction. Batch normalization and layer normalization directly improve conditioning of intermediate activations. Weight matrices with large condition numbers cause vanishing gradients along small-singular-value directions and exploding gradients along large ones — spectral normalization controls $\sigma_{\max}$ to bound this.

---

**Q6: Explain LoRA's low-rank decomposition in terms of SVD.**

In full fine-tuning, $W \in \mathbb{R}^{m \times n}$ changes by $\Delta W$ of rank $\min(m,n)$. LoRA constrains $\Delta W = BA$ where $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, $r \ll \min(m,n)$. This is exactly a rank-$r$ matrix parameterized in factored form — analogous to keeping only the top-$r$ singular components of $\Delta W$. The Eckart-Young theorem guarantees this is the best possible rank-$r$ approximation. The assumption is that the "task-relevant" update lives in a low-dimensional subspace of the full weight space, which is empirically supported by the observation that trained $\Delta W$ matrices have rapidly decaying singular values.

---

**Q7: What is the difference between a pseudoinverse and a matrix inverse, and when do you need the former?**

The inverse $A^{-1}$ exists only for square, full-rank matrices and satisfies $AA^{-1} = I$. The pseudoinverse $A^+$ exists for any matrix: it is defined via SVD as $V\Sigma^+ U^\top$ where $\Sigma^+$ inverts non-zero singular values and zeroes out zero singular values. When $Ax = b$ has no solution ($b \notin \text{col}(A)$), $A^+ b$ gives the minimum-norm least-squares solution. You need it whenever you solve overdetermined systems (more equations than unknowns) — the exact scenario in fitting a linear model with MSE loss. Numerically, you should use `np.linalg.lstsq` rather than explicitly forming $A^+$, as it is more stable.
