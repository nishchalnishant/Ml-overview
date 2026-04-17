# Maths

These notes are written for interview delivery: start with the first paragraph, then use the bullet points if the interviewer asks you to go deeper.

---

# Q1: Eigenvalues and Eigenvectors

**Interview-ready answer**

For a square matrix `A`, an eigenvector `v` is a non-zero vector whose direction is preserved by the linear transform, and the corresponding eigenvalue `lambda` tells you how much that direction is stretched or shrunk: `Av = lambda v`. In interviews, the key intuition is that eigenvectors reveal the "natural directions" of a transformation. That matters because many ML methods, especially PCA and spectral methods, look for the dominant directions in the data.

**Go deeper if asked**

- Most vectors change both direction and magnitude under a matrix; eigenvectors are the special directions that only scale.
- You find eigenvalues by solving `det(A - lambda I) = 0`, then solve `(A - lambda I)v = 0` for the corresponding eigenvectors.
- For symmetric matrices, eigenvalues are real and eigenvectors for different eigenvalues are orthogonal. That is why covariance matrices are so convenient in ML.
- Large positive eigenvalues usually correspond to directions with strong signal or high variance; very small eigenvalues often correspond to redundant directions.

**Why interviewers care**

- In PCA, eigenvectors of the covariance matrix give the principal directions.
- In graph ML and spectral clustering, eigenvectors of graph Laplacians capture structure.
- In optimization, eigenvalues of the Hessian tell you about curvature and conditioning.

**Common pitfall**

Do not confuse eigenvalues with singular values. Singular values come from SVD and work for rectangular matrices; eigenvalues are defined for square matrices.

---

# Q2: What is the Singular Value Decomposition (SVD), and how does it relate to PCA?

**Interview-ready answer**

SVD factorizes any matrix `X` as `U Sigma V^T`, where `U` and `V` are orthonormal matrices and `Sigma` contains the singular values. In ML, SVD is important because it gives the best low-rank approximation of a matrix, which makes it useful for compression, denoising, latent factor models, and dimensionality reduction. PCA is closely related: if your data matrix is centered, the principal directions are the right singular vectors in `V`, and the variance explained by each principal component is proportional to the squared singular values.

**Go deeper if asked**

- SVD works for any `m x n` matrix, unlike eigendecomposition which requires a square matrix.
- Truncating SVD to the top `k` singular values gives the best rank-`k` approximation in least-squares sense.
- If `X` is centered, then `X^T X / (n - 1)` is the covariance matrix. Its eigenvectors are PCA directions, and its eigenvalues equal `sigma_i^2 / (n - 1)`.
- In practice, PCA is often implemented through SVD because it is numerically stable.

**Where it shows up**

- PCA and latent semantic analysis
- Recommender systems and matrix factorization
- Compression of embeddings or dense feature matrices

**Common pitfall**

If the data is not centered before PCA, the first component can partly capture the mean rather than the directions of variation you actually care about.

---

# Q3: How does the chain rule apply in backpropagation for neural networks?

**Interview-ready answer**

Backpropagation is just the chain rule applied efficiently to a composition of functions. A neural network is a stack of layers, so the loss depends on each parameter through many intermediate computations. Instead of differentiating each parameter independently from scratch, backprop starts at the loss and propagates gradients backward through the computation graph, reusing intermediate results. That is why training deep networks is computationally feasible.

**Go deeper if asked**

- If `z = Wx + b`, `a = sigma(z)`, and `L` depends on `a`, then `dL/dW` is obtained by combining `dL/da`, `da/dz`, and `dz/dW`.
- Reverse-mode autodiff is efficient when you have one scalar output, such as loss, and many parameters.
- The backward pass is usually the same order of complexity as the forward pass, but it needs stored activations, so memory becomes a major constraint.
- Vanishing and exploding gradients come from repeatedly multiplying by Jacobians whose norms are much smaller or much larger than 1.

**Good interview framing**

If asked for intuition, say: "The forward pass computes predictions; the backward pass assigns credit or blame to every parameter for the final error."

**Common pitfall**

Backprop is not the optimizer. Backprop computes gradients; SGD, Adam, and related methods use those gradients to update parameters.

---

# Q4: What does it mean for a matrix to be positive semi-definite (PSD), and why does the covariance matrix have this property?

**Interview-ready answer**

A symmetric matrix `A` is positive semi-definite if `x^T A x >= 0` for every vector `x`. Intuitively, that means the matrix never produces negative quadratic energy. Covariance matrices are PSD because for any direction `x`, the quantity `x^T Sigma x` is exactly the variance of the projection of the data onto that direction, and variance can never be negative.

**Go deeper if asked**

- For symmetric matrices, being PSD is equivalent to saying all eigenvalues are non-negative.
- Covariance is `Sigma = E[(X - mu)(X - mu)^T]`. Then `x^T Sigma x = Var(x^T X)`, which proves PSD directly.
- Gram matrices like `X^T X` are also PSD, which is why kernels and similarity matrices often have this property.
- In practice, sample covariance matrices can appear slightly non-PSD because of numerical issues, so people often add a small `epsilon I` term.

**Why it matters in ML**

- PCA relies on the covariance matrix being PSD.
- Kernel methods require PSD kernels.
- Gaussian models use covariance matrices, and optimization or inference can fail if they are not numerically well-behaved.

**Common pitfall**

Positive semi-definite allows zero eigenvalues; positive definite means all eigenvalues are strictly positive, which is a stronger condition.
