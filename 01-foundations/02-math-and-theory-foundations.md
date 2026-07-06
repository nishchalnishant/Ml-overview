---
module: Foundations
topic: Math and Theory
status: unread
tags: [foundations, math, theory, linear-algebra, revision]
---
# Math and Theory Foundations

**For:** Engineers who need the mathematical rigor to understand how models actually learn, why they fail, and how to read academic papers.
**Use:** A deep reference for linear algebra, probability, information theory, and optimization.

---

## 1. Linear Algebra and Numerical Methods

Linear algebra is the foundational language of Machine Learning. It provides the tools to represent data (tensors), transformations (weights), and optimization landscapes.

### 1.1 Vectors and Matrices
- **Vector Space:** A set of elements closed under addition and scalar multiplication.
- **Span:** The set of all possible linear combinations of a set of vectors.
- **Linear Independence:** No vector in the set can be written as a combination of others.
- **Basis:** A linearly independent spanning set.
- **Rank:** The dimension of the vector space generated (or spanned) by its columns.
- **Null Space (Kernel):** Solutions to $Ax = 0$.

### 1.2 Eigenvalues and Eigenvectors
$Av = \lambda v$
A vector $v$ that, when transformed by $A$, only changes in scale ($\lambda$), not in direction.
- **Trace:** Sum of diagonal elements = Sum of eigenvalues.
- **Determinant:** Product of eigenvalues. Represents the "volume scaling factor" of the transformation.
- **Positive Definite:** All $\lambda_i > 0$. Ensures the function is strictly convex (unique global minimum).

### 1.3 Singular Value Decomposition (SVD)
SVD generalizes eigendecomposition to non-square matrices: $A = U \Sigma V^T$
- **$U$**: Left singular vectors (orthonormal).
- **$\Sigma$**: Singular values (diagonal, non-negative, sorted).
- **$V^T$**: Right singular vectors (orthonormal).

**Why it matters:** SVD is the foundation of PCA. It provides the best low-rank approximation of a matrix (Eckart-Young-Mirsky Theorem). It compresses data by keeping only the largest singular values.

### 1.4 Matrix Calculus Basics
Derivatives of scalars with respect to vectors (gradients) are crucial for backpropagation.
- $\nabla_x (a^T x) = a$
- $\nabla_x (x^T A x) = (A + A^T)x \quad$ (If $A$ is symmetric, this is $2Ax$)

### 1.5 Numerical Stability & Condition Numbers
The **Condition Number** $\kappa(A) = \frac{\sigma_{max}}{\sigma_{min}}$ (ratio of largest to smallest **singular value**, from the SVD above) measures how sensitive the solution of $Ax=b$ is to small changes in $b$. For symmetric positive-definite $A$, singular values equal eigenvalues, so $\kappa(A) = \lambda_{max}/\lambda_{min}$ — but that special case is not the general definition.
- $\kappa(A) \approx 1$: Well-conditioned (stable).
- $\kappa(A) \gg 1$: Ill-conditioned (unstable, small rounding errors blow up).
**Why it matters:** In deep learning, poor condition numbers lead to vanishing/exploding gradients. Regularization ($\lambda I$) artificially improves the condition number by increasing $\lambda_{min}$.

### 1.6 Computational Complexity and Memory Layout
- Matrix-Vector Product ($N \times N$ by $N \times 1$): $O(N^2)$
- Matrix-Matrix Product ($N \times N$ by $N \times N$): $O(N^3)$ (naively)
- SVD: $O(\min(mn^2, m^2n))$

**Memory Layout:**
Rows vs. Columns. Cache locality dictates that you must multiply matrices in a way that respects their memory layout (C uses row-major, Fortran/BLAS often use column-major). This drastically affects performance.

---

## 2. Calculus for Machine Learning

### 2.1 The Chain Rule
If $y = g(u)$ and $u = f(x)$, then $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$.
*Why it matters:* This is the theoretical basis of Backpropagation.

### 2.2 Gradients and the Jacobian
- **Gradient ($\nabla f$):** The vector of partial derivatives of a scalar function. Points in the direction of steepest ascent.
- **Jacobian ($J$):** The matrix of all first-order partial derivatives of a vector-valued function. Used when transforming vector spaces.
- **Hessian ($H$):** The matrix of second-order partial derivatives. Determines the curvature of the loss surface. If $H$ is positive-definite, you are at a minimum.

---

## 3. Probability and Statistics

### 3.1 Core Concepts
- **Random Variable:** Maps outcomes to numbers.
- **PMF / PDF:** Probability Mass/Density Function. $P(X=x)$.
- **Expectation:** The long-run average. $E[X] = \sum x \cdot P(x)$.
- **Variance:** Spread of the distribution. $Var(X) = E[(X - E[X])^2]$.

### 3.2 Bayes' Theorem
$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$
* "What is the probability of my hypothesis $A$ given the data $B$?"

---

## 4. Maximum Likelihood and MAP Estimation

### Maximum Likelihood Estimation (MLE)
"What parameters $\theta$ make the observed data most probable?"
$$\hat{\theta}_{MLE} = \arg\max_\theta \sum \log P(x_i | \theta)$$
- **Connection to Loss:** Minimizing Mean Squared Error (MSE) is mathematically identical to MLE under the assumption of Gaussian noise. Minimizing Cross-Entropy is MLE under a Bernoulli/Multinomial distribution.

### Maximum A Posteriori (MAP) Estimation
"What parameters $\theta$ are most probable given the data AND our prior beliefs?"
$$\hat{\theta}_{MAP} = \arg\max_\theta [ \sum \log P(x_i | \theta) + \log P(\theta) ]$$
- **Connection to Regularization:** MAP with a Gaussian prior on the weights is mathematically equivalent to L2 Regularization (Ridge). A Laplace prior is equivalent to L1 Regularization (Lasso).

---

## 5. Information Theory

### 5.1 Entropy ($H$)
The measure of uncertainty or "surprise" in a distribution.
$H(p) = - \sum p_i \log p_i$
A deterministic event has 0 entropy. A fair coin flip has maximum entropy.

### 5.2 Cross-Entropy
Measures how many bits are needed to encode data from true distribution $p$ using an estimated distribution $q$.
$H(p, q) = - \sum p_i \log q_i$
*Why it matters:* This is the standard loss function for classification. We want our predicted distribution $q$ to match the true label distribution $p$.

### 5.3 KL Divergence
Measures the distance between two distributions.
$D_{KL}(p || q) = \sum p_i \log \frac{p_i}{q_i}$
It is asymmetrical. Cross-Entropy $H(p,q) = H(p) + D_{KL}(p||q)$. Since $H(p)$ is fixed, minimizing cross-entropy minimizes KL divergence.

> **Deep dive:** See [Information Theory](05-information-theory.md) for Shannon entropy, mutual information, ELBO derivation, and the complete cross-entropy ↔ MLE ↔ KL unification.

---

## 5b. Bayesian Statistics

### 5b.1 Bayesian Inference Framework

Bayes' theorem for model parameters $\theta$ given data $D$:

$$P(\theta | D) = \frac{P(D | \theta) \cdot P(\theta)}{P(D)}$$

| Term | Name | Intuition |
|---|---|---|
| $P(\theta)$ | **Prior** | Belief about $\theta$ before seeing data |
| $P(D\|\theta)$ | **Likelihood** | How probable is $D$ if $\theta$ is true? |
| $P(\theta\|D)$ | **Posterior** | Updated belief after seeing data |
| $P(D)$ | **Evidence** / Marginal | Normalizing constant (often intractable) |

### 5b.2 Conjugate Priors

A **conjugate prior** is a prior distribution that, when combined with a given likelihood, produces a posterior of the same family. This makes Bayesian updating analytically tractable.

| Likelihood | Conjugate Prior | Example |
|---|---|---|
| Bernoulli | Beta | Coin flips, click-through rate |
| Categorical/Multinomial | Dirichlet | Topic models, word counts |
| Gaussian (known var) | Gaussian | Continuous measurements |
| Poisson | Gamma | Event counts per unit time |

**Example:** Beta-Bernoulli model for coin flips.
Prior: $P(p) = \text{Beta}(\alpha, \beta)$ — encodes $\alpha-1$ heads and $\beta-1$ tails prior belief.  
After observing $h$ heads and $t$ tails:  
Posterior: $P(p|data) = \text{Beta}(\alpha + h, \beta + t)$ — same family, just updated counts!

### 5b.3 MLE vs MAP vs Full Bayesian

| Approach | Formula | What it returns | Regularization |
|---|---|---|---|
| **MLE** | $\arg\max_\theta P(D\|\theta)$ | Point estimate | None |
| **MAP** | $\arg\max_\theta P(D\|\theta)P(\theta)$ | Point estimate | Implicit via prior |
| **Full Bayes** | Compute full $P(\theta\|D)$ | Distribution | Full uncertainty |

**MAP = MLE + Regularization:**
- Gaussian prior $P(\theta) = \mathcal{N}(0, \lambda^{-1}I)$ → MAP equivalent to L2 (Ridge) regularization
- Laplace prior $P(\theta) \propto \exp(-\lambda|\theta|)$ → MAP equivalent to L1 (Lasso) regularization

### 5b.4 Bayesian Inference in Practice (MCMC)

When the posterior is intractable (most real models), we approximate it:

- **MCMC (Markov Chain Monte Carlo):** Sample from the posterior directly via a Markov chain. Metropolis-Hastings and Hamiltonian Monte Carlo (HMC) are common algorithms. Exact but slow for high dimensions.
- **Variational Inference:** Approximate the posterior with a simpler distribution $q(\theta)$ from a family $Q$. Minimize $D_{KL}(q(\theta) \| p(\theta|D))$. This is what the **ELBO** optimizes. Fast but approximate.
- **Laplace Approximation:** Fit a Gaussian to the posterior at its mode (the MAP estimate). Fast, but assumes the posterior is approximately Gaussian.

### 5b.5 Connections to Deep Learning

| Bayesian concept | Deep Learning analogue |
|---|---|
| Prior $p(\theta)$ | L2/L1 weight regularization |
| Posterior predictive | Ensemble / MC Dropout predictions |
| Variational inference | VAE: $q(z\|x)$ approximates $p(z\|x)$ |
| Bayes optimal classifier | Theoretically minimum error classifier |
| Bayesian Optimization | Gaussian Process surrogate for HPO |

## 6. Generalization Theory

### 6.1 VC Dimension
The maximum number of points a model can perfectly shatter (classify in all possible ways). A measure of model capacity. If VC dimension is infinite (like 1-NN or large neural networks), classical theory says it should overfit.

### 6.2 Double Descent
Classical bias-variance theory states that test error forms a U-shape as capacity increases. Modern deep learning observes "Double Descent": once model capacity exceeds the number of parameters needed to memorize the data (interpolation threshold), test error drops again. Over-parameterization acts as implicit regularization.

### 6.3 PAC Learning
Probably Approximately Correct. Guarantees that, with high probability ($1-\delta$), a model will have an error bounded by $\epsilon$, given sufficient data.

---

## 7. Optimization Theory

### 7.1 Convexity
A function is convex if a line segment between any two points on the graph lies above the graph. Convex functions have a single, global minimum. Deep learning loss landscapes are **highly non-convex**.

### 7.2 Gradient Descent
$\theta = \theta - \alpha \nabla J(\theta)$
- **SGD:** Uses 1 example per step. Very noisy, but fast and provides implicit regularization.
- **Mini-batch SGD:** Uses $B$ examples. Balances noise and hardware vectorization.

### 7.3 Advanced Optimizers
- **Momentum:** Averages past gradients to build velocity, dampening oscillations.
- **Adam:** Computes adaptive learning rates for each parameter by keeping exponentially decaying averages of past gradients and squared gradients.

### 7.4 Loss Landscapes
Deep neural networks have millions of saddle points (where the gradient is zero, but it's only a minimum in some dimensions). Optimizers like SGD + Momentum easily escape saddle points but can get stuck in sharp minima.

---

## 8. Feature Engineering

Feature engineering bridges the gap between raw data and the model's inductive bias.
- **Standardization (Z-score):** Centers around 0, variance 1. Essential for distance-based models (KNN, SVM, PCA) and neural networks (helps gradient flow).
- **Normalization (Min-Max):** Scales between [0,1]. Used when bounds are strictly known (e.g., image pixels).
- **One-Hot Encoding:** For nominal categories.
- **Embeddings:** Dense vector representations for high-cardinality categorical variables (words, user IDs).
- **Polynomial Features:** Helps linear models capture non-linear relationships.

---

## 9. Matrix Calculus Cookbook

This is the set of derivatives you need to derive backpropagation and attention from scratch. All results assume real-valued matrices.

### 9.1 Scalar by Vector (Gradient)

| Expression | Gradient w.r.t. $\mathbf{x}$ | Notes |
|---|---|---|
| $a^T x$ | $a$ | Linear form |
| $x^T A x$ | $(A + A^T)x$ | Quadratic form; $2Ax$ if $A$ symmetric |
| $\|x\|^2 = x^T x$ | $2x$ | Special case of above |
| $\|Ax - b\|^2$ | $2A^T(Ax - b)$ | Setting to 0 gives the normal equation |

**Deriving the Normal Equation:**
$$\nabla_\theta \|X\theta - y\|^2 = 2X^T(X\theta - y) = 0 \implies \theta = (X^TX)^{-1}X^Ty$$

### 9.2 Vector by Vector (Jacobian)

If $\mathbf{y} = f(\mathbf{x})$ and $\mathbf{y} \in \mathbb{R}^m$, $\mathbf{x} \in \mathbb{R}^n$, the Jacobian $J \in \mathbb{R}^{m \times n}$:

$$J_{ij} = \frac{\partial y_i}{\partial x_j}$$

**Softmax Jacobian:** For softmax output $\mathbf{s}$:
$$\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$$
This is the expression you need when deriving the backprop through the softmax layer.

### 9.3 Attention Score Gradient

Attention: $\text{Attn} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

The $\sqrt{d_k}$ scaling exists because: without it, for large $d_k$, dot products $QK^T$ grow in magnitude proportionally to $\sqrt{d_k}$, pushing softmax into saturation (near-zero gradients in the flat regions). To see this: if each element of $q$ and $k$ is independently drawn from $\mathcal{N}(0,1)$, then $q \cdot k \sim \mathcal{N}(0, d_k)$, so variance grows linearly with $d_k$. Dividing by $\sqrt{d_k}$ normalizes to unit variance.

### 9.4 Chain Rule for Matrices (Backpropagation)

For a computational graph $L = f(z)$, $z = Wx + b$:
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial z} \cdot x^T, \quad \frac{\partial L}{\partial x} = W^T \cdot \frac{\partial L}{\partial z}, \quad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}$$

This pattern (upstream gradient × local derivative) is the core of every backprop implementation.

---

## 10. Convex Optimization

### 10.1 Convex Sets and Functions

- A set $C$ is **convex** if for any $x, y \in C$, the line segment $\theta x + (1-\theta)y \in C$ for $\theta \in [0,1]$.
- A function $f$ is **convex** if $f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$.
- **Why it matters:** Convex functions have **no local minima that aren't global**. Any gradient method converges to the global optimum.
- **ML examples of convex problems:** Linear regression (MSE), logistic regression, SVMs, Lasso.
- **Non-convex:** Neural networks, matrix factorization, GMMs. But empirically, SGD finds good solutions anyway.

### 10.2 Conditions for Optimality

**First-order condition:** $f$ is minimized at $x^*$ iff $\nabla f(x^*) = 0$ (for unconstrained convex).

**Second-order condition:** If $\nabla^2 f(x^*)$ (Hessian) is positive definite, $x^*$ is a local minimum. If the Hessian is positive semi-definite everywhere, $f$ is convex.

**KKT Conditions** (constrained optimization $\min f(x)$ s.t. $g_i(x) \leq 0$):
1. **Primal feasibility:** $g_i(x^*) \leq 0$
2. **Dual feasibility:** $\lambda_i \geq 0$
3. **Complementary slackness:** $\lambda_i g_i(x^*) = 0$ (either constraint is active or multiplier is zero)
4. **Stationarity:** $\nabla f(x^*) + \sum_i \lambda_i \nabla g_i(x^*) = 0$

**SVMs use KKT:** The support vectors are exactly the training points where the margin constraint is active ($g_i = 0$).

### 10.3 Why SGD Works on Non-Convex Landscapes

Deep network loss surfaces have many local minima and saddle points. The key empirical finding:
1. **Local minima are nearly as good as global:** In high dimensions, most local minima are in dense clusters with similar (good) loss values.
2. **Saddle points are escapable:** SGD's noise gradient estimates cause random perturbations that push the optimizer off saddle points.
3. **Sharp vs. flat minima:** SGD with larger learning rates tends to find **flat minima** (wide basins), which generalize better — the model is less sensitive to weight perturbations.

---

## Practice Exercises

**Exercise 1 (Linear Algebra):** Show that PCA is equivalent to computing the eigenvectors of the covariance matrix $C = \frac{1}{n}X^TX$. Use SVD: $X = U\Sigma V^T$. What are the principal components?

**Exercise 2 (Matrix Calculus):** Derive the normal equation $\theta^* = (X^TX)^{-1}X^Ty$ from scratch by differentiating $\|X\theta - y\|_2^2$ w.r.t. $\theta$ and setting the gradient to zero.

**Exercise 3 (Probability):** Prove that MAP estimation with a Gaussian prior $p(\theta) \propto \exp(-\lambda\|\theta\|^2)$ is equivalent to L2-regularized MLE (Ridge regression). *Hint: take the log of both sides of Bayes' theorem.*

**Exercise 4 (Information Theory):** Derive the cross-entropy loss from the MLE objective. Show that $\arg\min_\theta H(P_{data}, P_\theta) = \arg\min_\theta D_{KL}(P_{data} \| P_\theta)$.

**Exercise 5 (Optimization):** Why does gradient descent find the global minimum for logistic regression but not for a neural network? What properties of logistic regression guarantee this?

---

## Where to Next

- **Information Theory (entropy, KL, cross-entropy, mutual information)** → [information-theory.md](05-information-theory.md)
- **Python implementation of these concepts** → [03-python-and-data-tooling.md](03-python-and-data-tooling.md)
- **Optimizers that implement these gradients** → [03-deep-learning/components/06-optimisers.md](../03-deep-learning/components/06-optimisers.md)
