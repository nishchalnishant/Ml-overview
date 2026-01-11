# Math Derivations for ML Interviews

Many senior interviews will ask you to "Get on the whiteboard and derive...". Here are the most common derivations.

---

## 1. The Bias-Variance Decomposition
**Objective:** Decompose Mean Squared Error (MSE).

Given $y = f(x) + \epsilon$ and a model $\hat{f}(x)$.
- **Error:** $E[(y - \hat{f})^2]$
- **Derivation Steps:**
  1. Add and subtract $E[\hat{f}]$ inside the square.
  2. Expand $(a + b)^2$.
  3. Simplify terms using the property that $E[\epsilon] = 0$ and $f(x)$ is deterministic.
- **Result:**
  $$\text{MSE} = \text{Bias}[\hat{f}]^2 + \text{Var}[\hat{f}] + \sigma^2$$
  *Where $\sigma^2$ is irreducible noise.*

---

## 2. Gradient of Logistic Regression (Binary Cross-Entropy)
**Objective:** Show how the weights update.

- **Loss:** $J(\theta) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$
- **Sigmoid:** $\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$
- **Derivative of Sigmoid:** $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- **Using Chain Rule:** $\frac{\partial J}{\partial \theta} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial \theta}$
- **Result:**
  $$\nabla_\theta J = (\hat{y} - y)x$$
  *Insight: Same form as Linear Regression, but $\hat{y}$ is a probability.*

---

## 3. Backpropagation (Simple Chain Rule)
**Objective:** Derive the update for shared weights.

For $L = f(g(h(x)))$:
- $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$
- **Interview Key:** Explain why we store the **intermediate activations** (cache). We need them for the backward pass to avoid $O(N^2)$ re-computation.

---

## 4. PCA: Maximizing Variance
**Objective:** Explain why we use Eigenvectors.

- We want a unit vector $u$ that maximizes $\text{Var}(X u)$.
- $\text{Var}(X u) = u^T \Sigma u$ (where $\Sigma$ is the covariance matrix).
- Using **Lagrange Multipliers** to constrain $||u||=1$: $L = u^T \Sigma u - \lambda(u^T u - 1)$.
- $\frac{\partial L}{\partial u} = 2\Sigma u - 2\lambda u = 0 \rightarrow \Sigma u = \lambda u$.
- **Result:** The vector that maximizes variance is the **Principal Eigenvector** of the covariance matrix.

---

## 5. Normal Equation (Closed-form Solution for Linear Reg)
**Objective:** Derive $\theta = (X^T X)^{-1} X^T y$.

- **Loss:** $J = ||X\theta - y||^2 = (X\theta - y)^T (X\theta - y)$
- **Expand:** $\theta^T X^T X \theta - 2y^T X \theta + y^T y$
- **Gradient:** $\nabla_\theta J = 2X^T X \theta - 2X^T y = 0$
- **Solve:** $X^T X \theta = X^T y \rightarrow \theta = (X^T X)^{-1} X^T y$
- **Catch:** This is only solvable if $X^T X$ is invertible (no perfect multicollinearity).
