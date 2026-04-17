# Maths

This file is for surviving the part of the interview where someone smiles gently and says:

> "Let's go a little deeper into the math."

Which is corporate poetry for:

Please do not fake this.

So here we go.

---

# 1. Why Math Matters in ML

You do not need to become a chalk-dust performance artist.

But you do need to understand the math behind:

- optimization
- dimensionality reduction
- similarity
- curvature
- uncertainty

The goal is not formula dumping.
The goal is geometric intuition plus clean explanation.

---

# 2. Eigenvalues and Eigenvectors

For a square matrix `A`, an eigenvector is a direction that does not change direction when `A` is applied.

It only gets scaled.

That scale factor is the eigenvalue.

Mathematically:

- `Av = lambda v`

**Intuition**

Imagine stretching fabric.
Most directions twist and move.
But a few "natural directions" only get stretched or compressed.

Those are the eigenvectors.

**Why it matters in ML**

- PCA
- covariance analysis
- spectral methods
- curvature intuition

**Short answer**

Eigenvectors show the natural directions of a transformation, and eigenvalues tell you how strongly the transformation acts along those directions.

---

# 3. SVD

Singular Value Decomposition factorizes a matrix into:

- `U`
- `Sigma`
- `V^T`

You do not need to say this like it is a sacred chant.

What matters is the intuition:

SVD breaks a transformation into rotations plus scaling.

Why it matters:

- compression
- denoising
- low-rank approximation
- latent semantic analysis
- recommendation systems

**Short interview answer**

SVD is a general matrix factorization that reveals the most important directions and strengths of a transformation, even when the matrix is not square.

---

# 4. PCA and SVD Connection

PCA finds the directions of maximum variance.

SVD is often the practical way to compute that structure.

So if someone asks how PCA relates to SVD, the key idea is:

PCA on centered data can be derived from the SVD of the data matrix.

That is the connection worth saying cleanly.

---

# 5. Chain Rule and Backpropagation

The chain rule is one of the most important ideas in deep learning.

If a function is built from smaller functions, the derivative of the full system depends on how those smaller derivatives multiply together.

That is exactly what happens in neural networks.

Each layer feeds into the next.
So to compute the effect of an early weight on final loss, you multiply local sensitivities through the chain.

**Short answer**

Backpropagation is just the chain rule applied efficiently across a computation graph.

**DevOps analogy**

If a production incident happens, you trace impact backward through services and dependencies.
Backprop traces error backward through layers and dependencies.

Same detective energy.

---

# 6. Jacobian and Hessian

These sound intimidating until you simplify them.

## Jacobian

Matrix of first derivatives.

Useful when a function has:

- multiple inputs
- multiple outputs

Think:

How do all outputs change with all inputs?

## Hessian

Matrix of second derivatives.

Think:

How is the curvature changing?

Why it matters:

- optimization behavior
- curvature analysis
- sharp vs flat regions

**Easy memory trick**

- Jacobian = slope map
- Hessian = curvature map

---

# 7. Positive Semi-Definite (PSD) Matrices

A matrix is PSD if:

- `x^T A x >= 0` for all `x`

The intuitive meaning:

it never produces negative quadratic energy.

Why ML cares:

- covariance matrices are PSD
- kernel matrices should be PSD
- many optimization guarantees rely on PSD structure

**Short answer**

Covariance matrices are PSD because variance along any direction cannot be negative.

That is a very clean line and worth remembering.

---

# 8. L1 Norm vs L2 Norm

## L1 Norm

Sum of absolute values.

Encourages:

- sparsity

## L2 Norm

Square root of sum of squared values.

Encourages:

- smooth shrinkage
- stability

This shows up everywhere:

- regularization
- geometry
- distance
- optimization

---

# 9. KL Divergence

KL divergence measures how one probability distribution differs from another reference distribution.

Important caveat:

It is not symmetric.

That means:

- `KL(P || Q)` is not the same as `KL(Q || P)`

Why it matters in ML:

- VAEs
- distribution matching
- information-theoretic losses
- cross-entropy relationships

**Short answer**

KL divergence measures how much extra information is lost when one distribution is used to approximate another.

---

# 10. Lagrange Multipliers

Lagrange multipliers let you optimize a function under constraints.

That is the key idea.

Why it matters in interviews:

Because it connects beautifully to SVMs and constrained optimization.

**Short answer**

Lagrange multipliers convert a constrained optimization problem into a form where the objective and constraints can be optimized together.

That is enough for many interviews.

---

# 11. Hessian Intuition for Optimization

Why do interviewers sometimes bring up the Hessian?

Because it tells you about curvature.

High curvature can mean:

- unstable updates
- sensitivity to step size

Low curvature can mean:

- flatter region
- slower movement

This matters when discussing:

- optimizer stability
- exploding gradients
- second-order methods

---

# 12. Quick Revision Table

| Concept | Plain-English Meaning | Why ML Cares |
|---|---|---|
| Eigenvector | natural direction of transformation | PCA, spectra |
| Eigenvalue | stretch strength in that direction | variance, stability |
| SVD | rotation + scaling decomposition | compression, topics, recsys |
| Jacobian | first-derivative map | backprop structure |
| Hessian | second-derivative curvature map | optimization |
| PSD | no negative quadratic form | covariance, kernels |
| KL divergence | mismatch between distributions | VAEs, CE relation |

---

# Quick Thought Experiment

An interviewer asks:

> "Why do we care about eigenvectors in PCA?"

Your clean answer:

Because they identify the directions in feature space along which the data varies the most, and those directions become the principal components.

Short.
Correct.
No smoke machine needed.

---

# Mini Pop Quiz

Which one is symmetric by definition:

- KL divergence
- covariance matrix

Answer:

Covariance matrix.

KL divergence is not symmetric.

And yes, that tiny distinction matters.
