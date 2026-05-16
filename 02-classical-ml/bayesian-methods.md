# Bayesian Methods in ML

Bayesian methods treat model parameters as random variables with probability distributions, not fixed point estimates. This enables uncertainty quantification and principled incorporation of prior knowledge.

---

## Core Concepts

### Bayes' Theorem

`P(θ | D) = P(D | θ) × P(θ) / P(D)`

| Term | Name | Role |
|------|------|------|
| `P(θ)` | Prior | Beliefs before seeing data |
| `P(D | θ)` | Likelihood | How well θ explains data |
| `P(θ | D)` | Posterior | Updated beliefs after data |
| `P(D)` | Evidence / Marginal likelihood | Normalizing constant |

### MLE vs MAP vs Full Bayesian

| Method | Estimate | Formula |
|--------|----------|---------|
| MLE | Point estimate | `θ̂_MLE = argmax P(D | θ)` |
| MAP | Point estimate with prior | `θ̂_MAP = argmax P(D | θ) P(θ)` |
| Full Bayesian | Distribution | `P(θ | D)` — no single value |

**MAP with Gaussian prior = L2 regularization (Ridge).** The prior `P(θ) ∝ exp(-λ‖θ‖²)` penalizes large weights.

---

## Conjugate Priors

When the prior and posterior belong to the same distribution family, the posterior has a closed form.

| Likelihood | Conjugate Prior | Posterior |
|-----------|----------------|-----------|
| Bernoulli | Beta | Beta |
| Categorical | Dirichlet | Dirichlet |
| Gaussian (known var) | Gaussian | Gaussian |
| Poisson | Gamma | Gamma |

```python
# Beta-Binomial: coin flip example
from scipy.stats import beta

prior_a, prior_b = 1, 1   # uniform prior
n_heads, n_tails = 7, 3   # observed data
posterior = beta(prior_a + n_heads, prior_b + n_tails)
print(posterior.mean(), posterior.interval(0.95))
```

---

## Gaussian Processes (GPs)

A GP defines a distribution over functions. Any finite collection of function values follows a multivariate Gaussian.

`f(x) ~ GP(m(x), k(x, x'))`

- `m(x)` = mean function (often 0)
- `k(x, x')` = kernel/covariance function (encodes smoothness, periodicity, etc.)

### GP Regression

Given training data `(X, y)`, the posterior predictive at new point `x*`:

```
f* | X, y, x* ~ N(μ*, Σ*)
μ* = k(x*, X) [k(X, X) + σ²I]⁻¹ y
Σ* = k(x*, x*) - k(x*, X) [k(X, X) + σ²I]⁻¹ k(X, x*)
```

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
gp.fit(X_train, y_train)
y_pred, y_std = gp.predict(X_test, return_std=True)
```

**Key kernels:**
- RBF (squared exponential): smooth functions
- Matérn: controls smoothness via ν parameter
- Periodic: periodic functions
- Linear: Bayesian linear regression

**Limitation:** `O(n³)` training, `O(n²)` memory. Use sparse GPs (inducing points) for large datasets.

**Applications:** Hyperparameter optimization (Bayesian optimization), spatial modeling, small-data regression with uncertainty.

---

## Bayesian Optimization

Use a surrogate model (usually a GP) to optimize expensive black-box functions.

```
For t = 1, 2, ..., T:
    Fit GP to observations {(x_i, f(x_i))}
    Maximize acquisition function to get next x_t
    Evaluate f(x_t) (expensive step)
    Add (x_t, f(x_t)) to observations
```

**Acquisition functions:**

| Function | Intuition |
|----------|-----------|
| Expected Improvement (EI) | Expected gain over current best |
| Upper Confidence Bound (UCB) | Optimism under uncertainty |
| Probability of Improvement (PI) | Probability of beating current best |

```python
from bayes_opt import BayesianOptimization

def objective(lr, n_estimators):
    model = RandomForestClassifier(n_estimators=int(n_estimators), ...)
    return cross_val_score(model, X, y).mean()

optimizer = BayesianOptimization(
    f=objective,
    pbounds={'lr': (1e-4, 1e-1), 'n_estimators': (10, 300)}
)
optimizer.maximize(n_iter=30)
```

**Use case:** Hyperparameter search when each evaluation (training + validation) costs minutes to hours.

---

## Bayesian Neural Networks (BNNs)

Place a prior over network weights `P(W)`. After observing data, infer `P(W | D)`. Predictions use the full predictive distribution:

`P(y* | x*, D) = ∫ P(y* | x*, W) P(W | D) dW`

The integral is intractable → approximations needed.

### Variational Inference (VI)

Approximate `P(W | D)` with a simpler distribution `q(W; φ)` by minimizing KL divergence:

`L(φ) = E_q[log P(D | W)] - KL[q(W; φ) || P(W)]`  
= Expected log-likelihood - KL penalty

This is the **ELBO** (Evidence Lower BOund). Maximizing ELBO = minimizing KL from approximate to true posterior.

**Mean-field VI:** Each weight has independent Gaussian `q(w_i) = N(μ_i, σ_i²)`. Sample weights during forward pass, backprop through μ and σ using the reparameterization trick.

```python
# Reparameterization trick
mu = nn.Parameter(...)
log_sigma = nn.Parameter(...)
W = mu + torch.exp(log_sigma) * torch.randn_like(mu)  # sample W
```

**Libraries:** `pyro`, `torchbnn`, `tensorflow-probability`

### MC-Dropout (Gal & Ghahramani, 2016)

Train with dropout normally. At inference, keep dropout **ON** and run multiple forward passes. The variance across passes approximates epistemic uncertainty.

```python
model.train()  # keeps dropout active
n_samples = 50
preds = torch.stack([model(x) for _ in range(n_samples)])
mean_pred = preds.mean(0)
uncertainty = preds.var(0)
```

**Practical:** No change to training. Strong approximate uncertainty, fast to implement.

### Deep Ensembles

Train N independent models with different random seeds. Ensemble predictions: average softmax outputs or moments.

- N=5 ensembles give strong uncertainty estimates, often beating BNNs
- Expensive: N× training compute and memory

---

## Variational Autoencoders (VAEs) — Bayesian View

A VAE is a latent variable model with:
- Prior: `P(z) = N(0, I)`
- Likelihood: `P(x | z) = N(μ_θ(z), σ_θ(z))`
- Approximate posterior: `q_φ(z | x) = N(μ_φ(x), σ_φ(x))`

Training maximizes the ELBO:
`L = E_q[log P(x | z)] - KL[q(z | x) || P(z)]`

---

## Probabilistic Programming

Libraries that express models as programs and infer posteriors automatically.

| Library | Backend | Strength |
|---------|---------|---------|
| **Pyro** | PyTorch | Flexible, DL integration |
| **NumPyro** | JAX | Fast, vectorized |
| **Stan** | C++ | MCMC gold standard, HMC |
| **PyMC** | Aesara/PyTorch | User-friendly, rich ecosystem |

```python
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO

def model(X, y):
    w = pyro.sample("w", dist.Normal(0, 1).expand([X.shape[1]]).to_event(1))
    b = pyro.sample("b", dist.Normal(0, 1))
    mu = X @ w + b
    pyro.sample("y", dist.Normal(mu, 0.5), obs=y)

def guide(X, y):
    w_loc = pyro.param("w_loc", torch.zeros(X.shape[1]))
    w_scale = pyro.param("w_scale", torch.ones(X.shape[1]), constraint=constraints.positive)
    pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))
```

---

## When to Use Bayesian Methods

| Situation | Use |
|-----------|-----|
| Small data, need uncertainty | Gaussian Processes, BNNs |
| Hyperparameter tuning (expensive eval) | Bayesian Optimization |
| Need calibrated probabilities | MC-Dropout, Deep Ensembles |
| Prior domain knowledge available | Bayesian models with informative priors |
| Anomaly detection with uncertainty | GP-based methods |

**Avoid when:** Large data (GPs don't scale), low-latency inference (sampling is slow), or when a well-calibrated frequentist model suffices.

---

## Key Interview Points

- MAP = MLE + prior; Ridge regression is MAP with Gaussian prior.
- GPs give principled uncertainty but scale as O(n³).
- MC-Dropout is the cheapest way to get uncertainty estimates from neural nets.
- Deep ensembles often outperform variational BNNs for uncertainty, at the cost of compute.
- Bayesian Optimization is the standard approach for expensive hyperparameter tuning (Optuna, Ax, HyperOpt internally use it).
- ELBO = reconstruction term + KL regularization; used in VAEs and VI.
