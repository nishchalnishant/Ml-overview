# Optimization Theory — Deep Dive for ML Interviews

Advanced optimization concepts beyond basic gradient descent: geometry of loss landscapes, variance reduction, second-order methods, natural gradient, and modern techniques like SAM.

---

## Table of Contents

1. [Convex vs Non-Convex Optimization](#1-convex-vs-non-convex-optimization)
2. [Variance Reduction Methods](#2-variance-reduction-methods)
3. [Second-Order Methods](#3-second-order-methods)
4. [Natural Gradient](#4-natural-gradient)
5. [Information Geometry](#5-information-geometry)
6. [Sharpness-Aware Minimization (SAM)](#6-sharpness-aware-minimization-sam)
7. [Learning Rate Schedules](#7-learning-rate-schedules)
8. [Gradient Clipping](#8-gradient-clipping)
9. [Loss Landscape](#9-loss-landscape)
10. [Key Interview Points](#10-key-interview-points)

---

## 1. Convex vs Non-Convex Optimization

### Definitions

A function $f : \mathbb{R}^d \to \mathbb{R}$ is **convex** if for all $x, y$ and $\lambda \in [0,1]$:

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

Equivalently (for twice-differentiable $f$): the Hessian $\nabla^2 f(x) \succeq 0$ everywhere (positive semi-definite).

**Strictly convex** ($\succ 0$): unique global minimum.

### Local vs Global Minima

| Property | Convex | Non-Convex |
| :--- | :--- | :--- |
| Local minima | Every local min is global | Local minima may not be global |
| Saddle points | None | Exponentially many at high dimension |
| Convergence guarantee | Yes (to global) | No (in general) |
| Examples | Logistic regression loss, SVM | Neural network loss |

### Saddle Points

A **saddle point** $\theta^*$ satisfies $\nabla L(\theta^*) = 0$ but is neither a local max nor local min — the Hessian has both positive and negative eigenvalues.

In $d$ dimensions, a random critical point is a saddle point with probability approaching 1 as $d \to \infty$ (exponentially more saddle points than local minima). SGD with noise escapes saddle points faster than gradient descent because the stochastic gradient provides perturbations in all directions.

**Strict saddle property:** if every saddle point has at least one strictly negative Hessian eigenvalue, then gradient descent with small noise converges to a local minimum almost surely. Many practical losses satisfy this.

### Why Deep Learning Still Works

Despite non-convexity, deep networks train well due to:

**1. Overparameterization**

When the number of parameters $d \gg n$ (training points), the loss landscape has many global minima. The network has enough capacity that zero-training-loss solutions are dense in parameter space. Gradient descent finds one of them reliably.

- In the infinite-width limit, networks behave as kernel methods (Neural Tangent Kernel regime) and the loss is approximately convex.
- Overparameterized systems: the loss Hessian at initialization has many zero eigenvalues — the landscape is flat, making optimization easier.

**2. Implicit Regularization**

SGD does not merely minimize the training loss — the specific path it takes through parameter space introduces an implicit bias:

- For linear models: gradient descent converges to the **minimum-norm** solution.
- For deep linear networks: gradient descent converges to solutions with **minimum nuclear norm** (low effective rank).
- For nonlinear networks: SGD prefers flat minima (solutions where the loss changes little under perturbation), which generalize better.

The implicit regularization arises from: discrete step size, stochastic gradients, and the geometry of the level sets.

**3. Benign Non-Convexity**

Empirically, all local minima found in practice have similar loss values (especially when $d \gg n$). The energy barriers between local minima are low, and mode connectivity (see Section 9) shows many local minima are connected by low-loss paths.

---

## 2. Variance Reduction Methods

### Why SGD Has High Variance

Standard SGD uses a random minibatch $\mathcal{B}_t \subset \{1, \ldots, n\}$ to estimate the full gradient:

$$g_t = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \nabla \ell_i(\theta_t)$$

The estimator is **unbiased**: $\mathbb{E}[g_t] = \nabla L(\theta_t)$, but has variance:

$$\text{Var}(g_t) = \frac{\sigma^2}{|\mathcal{B}_t|}, \quad \sigma^2 = \frac{1}{n} \sum_{i=1}^n \|\nabla \ell_i(\theta) - \nabla L(\theta)\|^2$$

This variance does **not** go to zero as training progresses (unless the batch size grows). It forces a suboptimal convergence rate for plain SGD:

- **Strongly convex:** SGD converges at $O(1/T)$ but with a noise floor; cannot reach exact minimum.
- **Non-strongly convex:** $O(1/\sqrt{T})$ convergence to a stationary point.

Variance reduction methods reduce $\sigma^2 \to 0$ as $\theta \to \theta^*$, recovering linear convergence (for strongly convex problems).

---

### SVRG — Stochastic Variance Reduced Gradient

**Reference:** Johnson & Zhang, 2013.

**Key idea:** periodically compute the full gradient as a "snapshot", then use it to correct minibatch gradients.

**Algorithm:**

```
outer loop (epochs s = 1, 2, ...):
    compute full gradient: μ̃ = (1/n) Σᵢ ∇ℓᵢ(θ̃)   # snapshot, cost O(n)
    set θ₀ = θ̃

    inner loop (steps t = 1, ..., m):
        sample i uniformly from {1,...,n}
        vₜ = ∇ℓᵢ(θₜ) - ∇ℓᵢ(θ̃) + μ̃              # variance-reduced gradient
        θₜ₊₁ = θₜ - η · vₜ

    set θ̃ = θₘ  (or a random iterate from inner loop)
```

**Why this reduces variance:**

$$\text{Var}(v_t) = \text{Var}(\nabla\ell_i(\theta_t) - \nabla\ell_i(\tilde{\theta}))$$

As $\theta_t \to \tilde{\theta} \to \theta^*$, the two gradient terms $\nabla\ell_i(\theta_t)$ and $\nabla\ell_i(\tilde{\theta})$ become nearly equal, so their difference $\to 0$. The corrected estimator $v_t$ remains **unbiased** and has **vanishing variance** near the optimum.

**Convergence (strongly convex):** $O(\rho^T)$ geometric convergence, i.e., linear convergence — exponentially faster than SGD's $O(1/T)$.

**Cost:** One full pass per epoch for the snapshot + $m$ stochastic steps. Typically $m = 2n$ so cost per epoch $\approx 3n$ gradient evaluations.

---

### SAGA

**Reference:** Defazio et al., 2014.

**Key idea:** maintain a table of the most recent gradient for every data point. No periodic full gradient pass needed.

**Algorithm:**

```
Initialize: compute ∇ℓᵢ(θ₀) for all i; store in table gᵢ
φ̄ = (1/n) Σᵢ gᵢ   # running mean of stored gradients

for each step t:
    sample i uniformly
    compute ∇ℓᵢ(θₜ)                              # fresh gradient for i
    update:  vₜ = ∇ℓᵢ(θₜ) - gᵢ + φ̄              # variance-reduced, unbiased
    θₜ₊₁ = θₜ - η · vₜ
    update table: gᵢ ← ∇ℓᵢ(θₜ)                  # replace stored gradient
    update mean:  φ̄ ← φ̄ + (1/n)(∇ℓᵢ(θₜ) - old gᵢ)
```

**Properties:**

| Property | SVRG | SAGA |
| :--- | :--- | :--- |
| Memory | $O(d)$ (just $\tilde{\theta}$) | $O(nd)$ (full gradient table) |
| Full gradient pass | Yes, every epoch | No (amortized via table) |
| Unbiased | Yes | Yes |
| Convergence (strongly convex) | $O(\rho^T)$ linear | $O(\rho^T)$ linear |
| Supports proximal ops | With modification | Natively |

**Memory tradeoff:** SAGA's $O(nd)$ table is feasible for moderate $n$ and $d$ but prohibitive for large neural networks — this is the key reason SVRG-style methods are more popular in deep learning.

### Convergence Summary

| Method | Rate | Notes |
| :--- | :--- | :--- |
| GD (convex) | $O(1/T)$ | Full gradient each step |
| SGD (convex) | $O(1/\sqrt{T})$ | Noise floor remains |
| SGD (strongly convex) | $O(1/T)$ | Diminishing LR needed |
| SVRG / SAGA (strongly convex) | $O(\rho^T)$ linear | Variance reduced |
| Adam (non-convex) | $O(1/\sqrt{T})$ | Adaptive, but no better asymptotic |

---

## 3. Second-Order Methods

### Newton's Method

The Newton step minimizes the second-order Taylor expansion of $L$:

$$\theta_{t+1} = \theta_t - H_t^{-1} g_t$$

where $H_t = \nabla^2 L(\theta_t)$ is the Hessian and $g_t = \nabla L(\theta_t)$.

**Why it works:** rescales the gradient by curvature — large curvature directions get small steps; flat directions get large steps. Converges **quadratically** near the optimum for convex problems.

**Why it fails for deep learning:**

1. **Cost of computing $H$:** $d^2$ entries, where $d \sim 10^8$ for modern networks. Infeasible.
2. **Cost of inverting $H$:** $O(d^3)$. Catastrophically expensive.
3. **Non-PSD Hessian:** neural network Hessians are often indefinite (saddle points), making the Newton step point uphill.
4. **Overfitting to curvature:** the Hessian at a single batch is noisy; the Newton step amplifies this noise.

### Quasi-Newton: L-BFGS

**Limited-memory BFGS** approximates $H^{-1}$ using only the last $k$ gradient differences (typically $k = 5$–$20$):

$$s_t = \theta_{t} - \theta_{t-1}, \quad y_t = g_t - g_{t-1}$$

$$H^{-1}_{t+1} \approx \text{rank-}(2k) \text{ update of } H^{-1}_t$$

The update stores only $\{s_t, y_t\}_{t-k}^t$ — memory $O(kd)$ instead of $O(d^2)$.

**Used in:** full-batch or large-batch settings (e.g., second phase of training, fine-tuning with large batches). Not competitive with Adam for stochastic mini-batch training because the curvature estimate degrades with noisy gradients.

### K-FAC — Kronecker-Factored Approximate Curvature

**Reference:** Martens & Grosse, 2015.

Approximates the **Fisher information matrix** (a proxy for the Hessian) for neural networks using a Kronecker product structure.

For a layer with weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$, the Fisher block for that layer is:

$$F_W \approx A \otimes G$$

where:
- $A = \mathbb{E}[a a^\top]$, $a$ = layer input activations ($d_{in} \times d_{in}$)
- $G = \mathbb{E}[\delta \delta^\top]$, $\delta$ = backpropagated gradient signal ($d_{out} \times d_{out}$)

**Why Kronecker?** The gradient of the loss w.r.t. $W$ is $\delta a^\top$; under mild independence assumptions, the Fisher factorizes as this Kronecker product.

**Inverse:** $(A \otimes G)^{-1} = A^{-1} \otimes G^{-1}$ — invert two small matrices instead of one huge one.

**Cost:** $O(d_{in}^3 + d_{out}^3)$ per layer instead of $O((d_{in} d_{out})^3)$.

**In practice:** K-FAC can reduce the number of training steps by $5$–$10\times$ vs. Adam, at the cost of higher per-step compute. Used in Distributed K-FAC for large-scale training.

---

## 4. Natural Gradient

### Motivation: Parameter Space vs. Distribution Space

Standard gradient descent performs steepest descent in **Euclidean parameter space**:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

The step size $\|\Delta\theta\|_2$ is measured in parameter space, but two parameter vectors $\theta$ and $\theta'$ that are close in $\ell_2$ norm may define very different distributions $p_\theta$ and $p_{\theta'}$ (or vice versa). The Euclidean metric ignores the statistical geometry of the model.

**Example:** for a Bernoulli model $p_\theta = \text{Bern}(\sigma(\theta))$, changing $\theta$ from $0 \to 1$ has different effect on the distribution depending on where you start (sigmoid is nonlinear). Euclidean steps do not respect this.

### The Natural Gradient Update

The **Fisher information matrix** defines a Riemannian metric on the statistical manifold:

$$F(\theta) = \mathbb{E}_{x \sim p_\theta}\left[\nabla_\theta \log p_\theta(x) \, \nabla_\theta \log p_\theta(x)^\top\right]$$

The **natural gradient** is the steepest descent direction in the metric induced by $F$:

$$\tilde{\nabla}_\theta L = F(\theta)^{-1} \nabla_\theta L(\theta)$$

**Update rule:**

$$\theta_{t+1} = \theta_t - \eta \, F(\theta_t)^{-1} \nabla_\theta L(\theta_t)$$

### Why It Matters

1. **Reparameterization invariance:** if you reparameterize $\phi = g(\theta)$, the natural gradient update gives the same result (in distribution space) regardless of the parameterization. Standard gradient descent does not have this property.

2. **Faster convergence:** the natural gradient accounts for the curvature of the KL-divergence ball around the current parameters. It takes steps that move a fixed amount in distribution space rather than parameter space — avoiding tiny steps in flat directions and huge steps in curved directions.

3. **Fisher = Hessian of KL:** locally, $\text{KL}(p_\theta \| p_{\theta + \delta}) \approx \frac{1}{2} \delta^\top F(\theta) \delta$. The natural gradient step minimizes the loss subject to a KL-divergence trust region.

### Connection to NGD and TRPO

**Natural Gradient Descent (NGD):** direct application to neural networks. Impractical naively ($F$ is $d \times d$); K-FAC is the standard approximation.

**TRPO (Trust Region Policy Optimization):** in RL, the policy $\pi_\theta$ is a distribution over actions. TRPO solves:

$$\max_\theta \, \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} A(s,a)\right] \quad \text{s.t.} \quad \mathbb{E}\left[\text{KL}(\pi_{\theta_\text{old}} \| \pi_\theta)\right] \leq \delta$$

The first-order approximation to this constrained problem yields a natural gradient step. PPO approximates TRPO using a clipped surrogate objective, avoiding the expensive constraint.

---

## 5. Information Geometry

### The Fisher Information Matrix as a Riemannian Metric

The set of probability distributions $\{p_\theta : \theta \in \Theta\}$ forms a **statistical manifold**. The Fisher information matrix:

$$F_{ij}(\theta) = \mathbb{E}_{p_\theta}\left[\frac{\partial \log p_\theta}{\partial \theta_i} \frac{\partial \log p_\theta}{\partial \theta_j}\right]$$

defines a **Riemannian metric** on this manifold. This is the unique metric (up to scaling) that is:
- Invariant to sufficient statistics
- Invariant to reparameterization
- Consistent with the data processing inequality

The infinitesimal squared distance between $p_\theta$ and $p_{\theta + d\theta}$ is:

$$ds^2 = d\theta^\top F(\theta) \, d\theta$$

### KL Divergence as Local Distance

The KL divergence is **not** a metric (not symmetric), but locally it approximates the Fisher metric:

$$\text{KL}(p_\theta \| p_{\theta+\epsilon}) = \frac{1}{2} \epsilon^\top F(\theta) \epsilon + O(\|\epsilon\|^3)$$

This is why the natural gradient step — which preconditions by $F^{-1}$ — is equivalent to steepest descent under the KL divergence geometry.

### Exponential Family Manifolds

For exponential family distributions $p_\theta(x) = h(x) \exp(\theta^\top T(x) - A(\theta))$:

- The Fisher matrix is $F(\theta) = \nabla^2 A(\theta)$ (Hessian of the log-partition function).
- The manifold has a **dual coordinate system**: natural parameters $\theta$ and expectation parameters $\mu = \nabla A(\theta) = \mathbb{E}[T(x)]$.
- The **Legendre transform** connects the two: $A^*(\mu) = \theta^\top \mu - A(\theta)$.
- In expectation coordinates, the Fisher metric takes a simpler form.

This duality underlies the EM algorithm: E-step projects onto the constraint manifold in expectation coordinates; M-step maximizes in natural coordinates.

### Why Natural Gradient = Fisher-Preconditioned Gradient

Steepest descent on the manifold: minimize $L(\theta + d\theta)$ subject to $\|d\theta\|_F^2 = d\theta^\top F(\theta) d\theta \leq \epsilon^2$.

Using Lagrange multipliers:

$$d\theta^* = -\lambda F(\theta)^{-1} \nabla_\theta L(\theta)$$

Setting $\lambda$ to control step size: $\theta_{t+1} = \theta_t - \eta F(\theta_t)^{-1} \nabla L(\theta_t)$.

This is exactly the natural gradient update. The Fisher metric defines "equal distance in distribution space" and $F^{-1}$ corrects for the distortion introduced by the parameterization.

---

## 6. Sharpness-Aware Minimization (SAM)

### Motivation: Flat Minima Generalize Better

The **sharpness** of a minimum $\theta^*$ is characterized by the maximum eigenvalue of the Hessian $\lambda_{\max}(H)$. Sharp minima have large $\lambda_{\max}$: a small perturbation to $\theta$ causes a large increase in loss. Flat minima are robust to perturbation.

**Generalization intuition (PAC-Bayes):** a parameter vector $\theta$ in a flat region can be perturbed by a Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ with minimal effect on the loss. The PAC-Bayes bound gives:

$$L_{\text{test}}(\theta) \leq L_{\text{train}}(\theta^{\epsilon}) + O\!\left(\sqrt{\frac{\|\theta\|^2/\sigma^2 + \log(1/\delta)}{n}}\right)$$

Minimizing the loss at the perturbed point $\theta + \epsilon$ (worst-case perturbation) encourages flat minima.

### SAM Algorithm

**Objective:** find $\theta$ that minimizes the **worst-case perturbed loss**:

$$\min_\theta \max_{\|\epsilon\|_2 \leq \rho} L(\theta + \epsilon)$$

**Two-step update per iteration:**

**Step 1 — Find the worst perturbation** (gradient ascent, first-order approximation):

$$\hat{\epsilon} = \rho \cdot \frac{\nabla_\theta L(\theta)}{\|\nabla_\theta L(\theta)\|_2}$$

This is the steepest ascent direction scaled to lie on the $\ell_2$ ball of radius $\rho$.

**Step 2 — Update at the perturbed point:**

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t + \hat{\epsilon})$$

Note: the gradient in Step 2 is computed at $\theta + \hat{\epsilon}$, not at $\theta$.

**Total cost:** 2 forward + 2 backward passes per step (vs. 1+1 for SGD). The extra backward pass is the main cost overhead.

### ASAM — Adaptive SAM

Standard SAM uses a uniform ball $\|\epsilon\|_2 \leq \rho$, which is scale-invariant only if all parameters have the same scale. **ASAM** (Kwon et al., 2021) uses a parameter-adaptive norm:

$$\hat{\epsilon} = \rho \cdot \frac{T_\theta \nabla_\theta L(\theta)}{\|T_\theta \nabla_\theta L(\theta)\|_2}, \quad T_\theta = \text{diag}(|\theta_1|, \ldots, |\theta_d|)$$

This is scale-invariant: if you rescale $\theta_i \to c\theta_i$, the perturbation scales accordingly.

### Python Implementation

```python
import torch

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization optimizer.
    Wraps a base optimizer (e.g., SGD or Adam).

    Usage:
        optimizer = SAM(model.parameters(), torch.optim.SGD, lr=0.1, momentum=0.9, rho=0.05)

        # Training loop:
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)   # perturb weights

        criterion(model(inputs), targets).backward()
        optimizer.second_step(zero_grad=True)  # update at perturbed point
    """

    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"rho must be non-negative, got {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Compute and apply the worst-case perturbation epsilon-hat."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # ASAM: scale perturbation by |theta|
                if group["adaptive"]:
                    scale_p = torch.abs(p) * scale
                else:
                    scale_p = scale
                e_w = p.grad * scale_p
                p.add_(e_w)                     # perturb: theta += epsilon
                self.state[p]["e_w"] = e_w      # store perturbation for undo

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Undo perturbation, then apply base optimizer step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])    # undo: theta -= epsilon
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        """Compute global gradient norm across all parameter groups."""
        # Collect all grads into one norm computation for numerical stability
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else torch.ones_like(p)) * p.grad).norm(p=2)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def sam_training_step(model, inputs, targets, criterion, optimizer):
    """One SAM training step — two forward/backward passes."""
    # Pass 1: compute gradient at current theta
    optimizer.zero_grad()
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # Pass 2: compute gradient at theta + epsilon, then update
    criterion(model(inputs), targets).backward()
    optimizer.second_step(zero_grad=True)
    return loss.item()
```

---

## 7. Learning Rate Schedules

### Warm-Up

**Rationale:** adaptive optimizers (Adam, AdamW) maintain exponential moving averages of gradients:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

At $t=1$, $m_1$ and $v_1$ are poor estimates (only one gradient seen). The bias correction $\hat{m}_t = m_t / (1-\beta_1^t)$ partially fixes this, but with typical $\beta_2 = 0.999$, $v_t$ takes $\sim 1000$ steps to stabilize. Early steps with an unstable $v_t$ produce erratic effective learning rates.

**Warm-up:** linearly increase $\eta$ from $\eta_{\min} \approx 0$ to $\eta_{\max}$ over $T_{\text{warm}}$ steps. During warm-up, large but uncertain gradient estimates cause less damage because the LR is small. Also prevents early large updates from sending weights to bad regions.

**Typical warm-up duration:** 5–10% of total training steps for transformers; shorter for CNNs.

### Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{t}{T}\pi\right)\right)$$

Smoothly decays LR from $\eta_{\max}$ to $\eta_{\min}$ over $T$ steps. The cosine shape (slow start, fast middle, slow end) matches the typical loss curvature profile. **Cosine with restarts** (SGDR): after each cycle, reset LR and repeat — with optionally longer cycles each time.

### Cyclical Learning Rate (CLR)

LR oscillates between $\eta_{\min}$ and $\eta_{\max}$ in a triangular or cosine wave. Key insight from Smith (2017): periodically increasing the LR helps escape sharp local minima and saddle points. The high LR phase acts as a perturbation; the low LR phase allows convergence.

### 1-Cycle Policy

Smith & Touvron's 1-cycle policy for fast training:

1. **Phase 1 (warm-up):** LR rises from $\eta_{\max}/\text{div\_factor}$ to $\eta_{\max}$ over $\sim 30\%$ of steps.
2. **Phase 2 (anneal):** LR drops from $\eta_{\max}$ to $\eta_{\max}/\text{div\_factor}$ over $\sim 70\%$ of steps.
3. **Phase 3 (fine anneal):** LR drops to $\eta_{\max}/\text{final\_div\_factor}$ over last few steps.

Momentum is cycled inversely: high when LR is low, low when LR is high.

### Code

```python
import torch
import math

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                     eta_min=0.0, last_epoch=-1):
    """
    Linear warm-up followed by cosine annealing.
    Standard schedule for transformer pre-training.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warm-up
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale to [eta_min_ratio, 1.0] range
        # (eta_min handling: here we return a multiplier; eta_min added below)
        return max(eta_min, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_cyclical_lr_schedule(optimizer, step_size_up, base_lr, max_lr, mode="triangular2"):
    """
    Cyclical LR: triangular or triangular2 (halves max_lr each cycle).
    """
    return torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size_up,
        mode=mode,
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
    )


def get_one_cycle_schedule(optimizer, max_lr, total_steps, pct_start=0.3, div_factor=25.0,
                            final_div_factor=1e4):
    """
    1-cycle policy: warm-up to max_lr, then cosine anneal down.
    pct_start: fraction of total_steps used for warm-up phase.
    """
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        anneal_strategy="cos",
        cycle_momentum=True,
    )


# Example usage
if __name__ == "__main__":
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    TOTAL_STEPS = 10_000
    WARMUP_STEPS = 1_000

    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, TOTAL_STEPS)

    for step in range(TOTAL_STEPS):
        optimizer.step()
        scheduler.step()
```

---

## 8. Gradient Clipping

### By Value

Clip each gradient component individually:

$$g_i \leftarrow \text{clip}(g_i, -c, c) = \max(-c, \min(c, g_i))$$

**Problem:** distorts the gradient direction. If only some components are large, clipping them changes the direction of $g$ in a non-uniform way. The resulting update no longer points in the (approximate) steepest descent direction.

### By Norm

Clip the entire gradient vector to have $\ell_2$ norm at most $c$:

$$g \leftarrow g \cdot \min\!\left(1, \frac{c}{\|g\|_2}\right)$$

**Properties:**

- Preserves the gradient direction (only scales the magnitude).
- If $\|g\|_2 \leq c$: no clipping (gradient unchanged).
- If $\|g\|_2 > c$: uniformly scales all components so the norm equals $c$.
- More principled: the update is a properly scaled descent direction.

**Why norm clipping is preferred:** gradient direction conveys useful optimization information. By-value clipping destroys this direction; by-norm clipping preserves it.

### Connection to Exploding Gradients in RNNs

In a vanilla RNN with weight matrix $W$, the gradient through $T$ time steps involves:

$$\frac{\partial L}{\partial h_0} = \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}} \cdot \frac{\partial L}{\partial h_T} = W^T \cdot \delta_T$$

If $\lambda_{\max}(W) > 1$: $\|W^T\| = O(\lambda_{\max}^T)$ — **exponential growth** (exploding gradients).
If $\lambda_{\max}(W) < 1$: $\|W^T\| \to 0$ — **exponential decay** (vanishing gradients).

**Gradient clipping** (Pascanu et al., 2013) directly addresses exploding gradients: when $\|\nabla L\| > c$, normalize to $c$. This does not fix vanishing gradients (which require architectural solutions: LSTMs, GRUs, skip connections, or attention).

**Implementation:**

```python
import torch

def clip_gradients(model, max_norm=1.0, norm_type=2.0):
    """
    Clip gradients by global norm. Returns the norm before clipping.
    Call after loss.backward() and before optimizer.step().
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm,
                                                 norm_type=norm_type)
    return total_norm.item()

# In training loop:
# loss.backward()
# grad_norm = clip_gradients(model, max_norm=1.0)
# optimizer.step()
```

**When to use:**

| Scenario | Typical `max_norm` |
| :--- | :--- |
| RNN / LSTM training | 1.0–5.0 |
| Transformer (language model) | 1.0 |
| RL policy gradients | 0.5–1.0 |
| CNNs (generally stable) | Optional / 10.0 |

**Monitoring:** track `grad_norm` during training. A consistent norm close to `max_norm` indicates clipping is active and parameters are in a high-curvature region. Sudden spikes indicate instability.

---

## 9. Loss Landscape

### Mode Connectivity

**Observation** (Garipov et al., 2018; Draxler et al., 2018): any two independently trained neural network solutions $\theta_A$ and $\theta_B$ can be connected by a path $\phi(t), t \in [0,1]$ in parameter space such that the loss $L(\phi(t))$ remains approximately constant (and low) along the path.

**Linear mode connectivity:** $L(\lambda \theta_A + (1-\lambda)\theta_B) \approx L(\theta_A)$ for all $\lambda \in [0,1]$.

- Not always true for independently trained networks (the interpolated network may have high loss in between).
- **True after alignment:** permuting the neurons of one network (matching neurons that serve the same function) restores linear mode connectivity. This is the basis of **model merging** and **model soups**.

**Nonlinear paths:** even without permutation, there exists a piecewise-linear or curved path connecting $\theta_A$ and $\theta_B$ through low-loss regions (Garipov et al. use a Bezier curve with a trainable midpoint).

**Implications:**

1. The loss landscape is not a collection of isolated minima — it has a connected **"valley"** or **"flat basin"** at the bottom.
2. Model ensembling by weight averaging (Model Soups, Wortsman et al., 2022) works precisely because models lie in the same mode-connected basin.

### Sharp vs. Flat Minima and Generalization

**Sharp minimum:** high $\lambda_{\max}(H)$, small basin — the loss increases rapidly in all high-curvature directions. Small distribution shift (train $\to$ test) easily moves you out of the basin.

**Flat minimum:** low $\lambda_{\max}(H)$, wide basin — the loss is insensitive to perturbations. Distribution shift has less effect; the solution is more robust.

**Theoretical connection:** the PAC-Bayes bound for a Gaussian perturbation $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ of parameters $\theta$:

$$L_{\text{test}}(\theta) \leq \frac{1}{n} \sum_{i=1}^n \mathbb{E}_\epsilon[L_i(\theta + \epsilon)] + \text{complexity penalty}$$

Minimizing the expected perturbed loss encourages flat minima (which is exactly what SAM does).

**The Keskar et al. (2017) finding:** large-batch SGD converges to sharp minima; small-batch SGD converges to flat minima. Small batches have higher gradient noise, which acts as an implicit regularizer that drives the optimizer toward flat regions.

**Counterpoint (Dinh et al., 2017):** sharpness is not invariant to reparameterization — you can reparameterize any sharp minimum to be flat. ASAM addresses this with scale-invariant sharpness.

### Neural Tangent Kernel (NTK) Perspective

**Setup:** consider a neural network $f(x; \theta)$ with random initialization $\theta_0$. The NTK is:

$$K_{\text{NTK}}(x, x') = \nabla_\theta f(x; \theta)^\top \nabla_\theta f(x'; \theta)$$

**Infinite-width limit** (Jacot et al., 2018): as network width $\to \infty$:

1. The NTK $K_{\text{NTK}}$ converges to a deterministic kernel $K^*$ at initialization and **remains constant** throughout training.
2. Training dynamics reduce to **kernel gradient descent** — a linear ODE:

$$\dot{f}(x; \theta_t) = -K^* (f(\cdot; \theta_t) - y)$$

3. The loss decreases exponentially: $L(t) \leq L(0) e^{-2\lambda_{\min}(K^*) t}$.

**Implications:**

- In the NTK regime, neural networks are equivalent to kernel machines with kernel $K^*$.
- The model does not learn features — representations (the kernel) are fixed at init.
- Generalization = generalization of the NTK kernel (classical kernel learning theory applies).

**Practical relevance:**

- Real networks (finite width, large LR) operate **outside** the NTK regime — they do learn features ("feature learning regime" / "mean-field regime").
- NTK provides: (a) a solvable model to understand optimization dynamics, (b) predictions that match finite networks at small LR / large width, (c) connection between initialization and trainability ($\lambda_{\min}(K^*)$ must be positive).

**Connection to loss landscape:** in the NTK regime, the loss landscape is approximately **convex** — a single global minimum exists and gradient descent finds it. Feature learning regime has non-convex landscape but the overparameterization + implicit regularization arguments (Section 1) apply.

---

## 10. Key Interview Points

### Convex / Non-Convex

- Deep learning is non-convex but works due to overparameterization (dense global minima) and implicit regularization (SGD bias toward flat, low-norm solutions).
- Saddle points dominate in high dimensions; noise in SGD helps escape them.
- "Strict saddle" property: if every saddle point has a direction of negative curvature, GD with noise converges to a local minimum.

### Variance Reduction

- Plain SGD noise floor prevents convergence to exact minimum; SVRG/SAGA achieve linear convergence for strongly convex problems by reducing gradient variance to zero near the optimum.
- SVRG: periodic full gradient snapshot, $O(n)$ cost amortized. SAGA: $O(nd)$ memory table. Neither is practical for modern deep nets (memory/compute), but the principles underlie techniques like large-batch training + learning rate scaling.

### Second-Order Methods

- Newton's method: $O(d^2)$ memory, $O(d^3)$ compute — infeasible for deep nets.
- L-BFGS: $O(kd)$ memory, works well in full-batch regime. Fails with stochastic gradients.
- K-FAC: approximates Fisher with Kronecker structure. Practical for neural nets; used in distributed training.

### Natural Gradient

- Standard gradient measures steepest descent in parameter space (Euclidean metric); natural gradient measures it in distribution space (Fisher metric).
- The natural gradient is reparameterization-invariant.
- Connection: K-FAC approximates the natural gradient; TRPO in RL solves a trust-region problem whose first-order solution is the natural gradient step.

### Information Geometry

- The Fisher matrix is the unique Riemannian metric on the statistical manifold compatible with the data processing inequality.
- KL divergence is locally approximated by the Fisher metric: $\text{KL}(p_\theta \| p_{\theta+\epsilon}) \approx \frac{1}{2}\epsilon^\top F \epsilon$.
- Exponential family: Fisher = Hessian of log-partition function; natural/expectation parameter duality underlies EM.

### SAM

- Seeks flat minima by minimizing worst-case perturbed loss within an $\ell_2$ ball.
- Two backward passes per step: first for the perturbation direction, second for the actual update.
- ASAM makes perturbation scale-invariant.
- Empirically improves generalization, especially in vision (ViT, ResNet benchmarks).

### Learning Rate Schedules

- Warm-up: Adam's moment estimates are unreliable early in training; small LR early avoids destructive updates.
- Cosine annealing: smooth, principled decay that matches typical loss curvature profiles.
- 1-cycle: enables "super-convergence" — training in far fewer steps by aggressively cycling LR.

### Gradient Clipping

- By norm is preferred over by value: preserves gradient direction.
- Essential for RNNs/LSTMs where $\|W^T\|$ grows exponentially with sequence length.
- Does not fix vanishing gradients (need architecture changes: LSTM gates, attention, skip connections).

### Loss Landscape

- Mode connectivity: independently trained models can be connected by low-loss paths (after permutation alignment). Foundation for model merging / model soups.
- Flat minima generalize better; SAM, small batch size, and high LR noise all encourage flat minima.
- NTK: infinite-width networks stay in a fixed kernel regime; finite-width networks with large LR operate in the feature-learning regime. NTK gives a linearized, convex view of training dynamics.

### Connecting Themes

```
Optimization goal:    min_θ L(θ)         [training loss]
Generalization goal:  min_θ L_test(θ)    [test loss]

SGD noise          →  implicit regularization  →  flat minima  →  better generalization
SAM                →  explicit flat-minima objective
Natural gradient   →  geometry-aware updates   →  faster convergence
K-FAC              →  tractable natural gradient approximation
Warm-up + cosine   →  stable + efficient convergence trajectory
```

---

*Cross-reference: `optimization.md` (gradient descent basics, Adam, AdamW), `math-derivations.md` (backprop, matrix calculus).*
