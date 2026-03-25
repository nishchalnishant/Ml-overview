# Maths

---

# Q1: Eigenvalues and Eigenvectors

## 1. 🔹 Direct Answer
For a square matrix **A**, a non-zero vector **v** is an **eigenvector** if **Av = λv** for some scalar **λ** (the **eigenvalue**). Eigenvectors are directions that **A** only **scales**; eigenvalues are those scale factors.

## 2. 🔹 Intuition
Think of **A** as a linear transformation: most vectors both rotate and stretch. Eigenvectors are the special directions that **stay on the same line**—only their length changes by **λ**. Like pushing a spring along its axis vs sideways.

## 3. 🔹 Deep Dive
- Solve **det(A − λI) = 0** for eigenvalues **λ**; for each **λ**, solve **(A − λI)v = 0** for **v**.
- **Symmetric** real matrices have real eigenvalues and orthogonal eigenvectors (PCA uses this).
- **Spectral theorem**: diagonalizable **A = QΛQ⁻¹** (when enough independent eigenvectors).

## 4. 🔹 Practical Perspective
- **PCA**: eigenvectors of covariance matrix = principal directions; eigenvalues = variance explained.
- **PageRank / graph spectra**, stability analysis, ODE linear systems **dx/dt = Ax**.
- **When not primary**: non-square matrices use **SVD** instead (generalizes eigen decomposition).

## 5. 🔹 Code Snippet
```python
import numpy as np
A = np.array([[2.0, 1.0], [1.0, 2.0]])
w, V = np.linalg.eig(A)  # w = eigenvalues, columns of V = eigenvectors
# verify: A @ V[:,0] ≈ w[0] * V[:,0]
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Relation to SVD? **A:** SVD works for any rectangular **A**; for PSD matrices, eigenvalues of **AᵀA** relate to singular values squared.
2. **Q:** Defective matrix? **A:** Not full set of eigenvectors—Jordan form instead of full diagonalization.
3. **Q:** PCA—why eigenvectors of covariance? **A:** They maximize variance **vᵀΣv** subject to **||v||=1** (Rayleigh quotient).

## 7. 🔹 Common Mistakes
- Confusing eigenvalues with singular values for general non-symmetric **A**.
- Forgetting eigenvectors are defined up to scale (normalize for numerics).

## 8. 🔹 Comparison / Connections
SVD, PCA, spectral clustering, matrix conditioning, power iteration for top eigenvector.

## 9. 🔹 One-line Revision
Eigenpairs **(λ, v)** satisfy **Av = λv**; power PCA, spectra, and understanding linear maps along invariant directions.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q2: What is the Singular Value Decomposition (SVD), and how does it relate to PCA?

## 1. 🔹 Direct Answer
Any **m×n** matrix **A** admits **SVD**: **A = U Σ Vᵀ** with **U** (m×m) and **V** (n×n) orthogonal, **Σ** diagonal with **singular values σᵢ ≥ 0**. **PCA** of **centered data matrix **X** is closely related: **principal directions** are **right singular vectors** **V**; **variance** along each component is **σᵢ²/(n−1)** for sample covariance **XᵀX/(n−1)**.

## 2. 🔹 Intuition
SVD finds the **best low-rank** approximation of **A** (Eckart–Young): keep top **k** singular values for denoising and compression.

## 3. 🔹 Deep Dive
- **Economy SVD**: only non-zero σ’s—efficient for rank-**r** matrices.
- **AᵀA** eigenvalues = **σᵢ²**; **AAᵀ** shares non-zero σ’s.

## 4. 🔹 Practical Perspective
**Truncated SVD** for **LSI**, **recommendations** (matrix factorization), **numerical** rank determination.

## 5. 🔹 Code Snippet
```python
import numpy as np
U, s, Vt = np.linalg.svd(X, full_matrices=False)
X_k = (U[:, :k] * s[:k]) @ Vt[:k, :]  # rank-k approximation
```

## 6. 🔹 Interview Follow-ups
1. **Q:** PCA without centering? **A:** First PCA component may track mean direction—not “variance” in usual sense.

## 7. 🔹 Common Mistakes
Confusing **singular values** of **X** with **eigenvalues** of **X** (square) without squaring relationship via **XᵀX**.

## 8. 🔹 Comparison / Connections
Eigendecomposition (square symmetric PSD), random projection.

## 9. 🔹 One-line Revision
SVD generalizes eigen-decomposition to rectangular matrices; PCA on centered data follows from **XᵀX** spectrum via **V** and **σ²**.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q3: How does the chain rule apply in backpropagation for neural networks?

## 1. 🔹 Direct Answer
**Loss L** depends on **weights** through a **composition** of layers **L = L(h_L(…h₁(x)))**. **∂L/∂w** = **∂L/∂h** · **∂h/∂…** … **chain** of Jacobians. **Reverse-mode** autodiff (backprop) applies chain rule **once** per weight by reusing upstream gradient **∂L/∂h**—**efficient** for scalar **L** and many parameters.

## 2. 🔹 Intuition
Blame flows **backward**: how much did this weight change the loss **through every path** that uses it?

## 3. 🔹 Deep Dive
For vector **h = f(Wx)**, **∂L/∂W = (∂L/∂h) xᵀ** (outer product form for linear layer). **Vanishing** if many Jacobians have singular values **smaller than 1** so the long chain product shrinks.

## 4. 🔹 Practical Perspective
Frameworks build **computation graphs**; you rarely differentiate by hand—**know** **shapes** for debugging.

## 5. 🔹 Code Snippet
```python
# PyTorch autograd handles this
loss.backward()  # dL/dW populated for all requires_grad=True
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Forward-mode JVPs? **A:** One column of Jacobian at a time—useful for Hessian-vector products.

## 7. 🔹 Common Mistakes
Thinking backprop is **O(1)**—same asymptotic order as forward, but **memory** stores activations.

## 8. 🔹 Comparison / Connections
Adjoint methods, manual backprop through softmax+CE (clean gradient).

## 9. 🔹 One-line Revision
Backprop is reverse-mode chain rule on the computation graph—efficient scalar-to-many-parameter gradients.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q4: What does it mean for a matrix to be positive semi-definite (PSD), and why does the covariance matrix have this property?

## 1. 🔹 Direct Answer
**Symmetric** **A** is **PSD** if **xᵀAx ≥ 0** for all **x** (equivalently all **eigenvalues ≥ 0**). **Covariance** **Σ = E[(X−μ)(X−μ)ᵀ]** is PSD because **xᵀΣx = Var(xᵀX) ≥ 0**—variance of any linear combination is nonnegative.

## 2. 🔹 Intuition
No direction in feature space has **negative** variance under the data distribution.

## 3. 🔹 Deep Dive
**Gram matrices** **G = XᵀX** are PSD; **kernel** matrices in kernel methods are PSD (Mercer).

## 4. 🔹 Practical Perspective
**Cholesky** requires PSD (strictly PD for numerical stability); **Mahalanobis** distance uses **Σ⁻¹**.

## 5. 🔹 Code Snippet
```python
import numpy as np
C = np.cov(X, rowvar=False)
assert np.all(np.linalg.eigvalsh(C) >= -1e-8)  # PSD up to numerics
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Not PSD empirically? **A:** Finite sample / numerical error—regularize **C + εI**.

## 7. 🔹 Common Mistakes
Using covariance without **centering** data first.

## 8. 🔹 Comparison / Connections
Kernel PCA, Gaussian processes.

## 9. 🔹 One-line Revision
PSD matrices generalize nonnegative scalars; covariance is PSD because it is variance of linear projections.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

