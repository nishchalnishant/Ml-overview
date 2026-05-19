# Bayesian Methods in ML

---

## Core Concepts

### The Problem with Point Estimates

**The problem**: standard ML models produce a single answer — a weight vector, a class label, a predicted value. But that answer carries no information about how uncertain the model is. A model trained on 10 examples and a model trained on 10 million examples might output the same number, even though one should be trusted and the other should not.

**The core insight**: instead of estimating a single "best" value for parameters, maintain a full probability distribution over them. Before seeing data, encode what you believe (the prior). After seeing data, update those beliefs according to Bayes' theorem. The result is a posterior distribution — a complete picture of what the data implies, including uncertainty.

**The mechanics**:

`P(θ | D) = P(D | θ) × P(θ) / P(D)`

| Term | Name | Role |
|------|------|------|
| `P(θ)` | Prior | Beliefs before seeing data |
| `P(D | θ)` | Likelihood | How well θ explains data |
| `P(θ | D)` | Posterior | Updated beliefs after data |
| `P(D)` | Evidence / Marginal likelihood | Normalizing constant |

**What breaks**: the posterior is usually intractable — computing `P(D)` requires integrating over all possible parameter values. For most models, this integral has no closed form, which is why Bayesian ML requires approximations (conjugate priors, variational inference, MCMC) rather than exact computation.

---

### MLE vs MAP vs Full Bayesian

**The problem**: if the full posterior is intractable, what do you do?

**The core insight**: different levels of approximation trade off computational cost against the richness of uncertainty information you retain.

| Method | Estimate | Formula |
|--------|----------|---------|
| MLE | Point estimate | `θ̂_MLE = argmax P(D | θ)` |
| MAP | Point estimate with prior | `θ̂_MAP = argmax P(D | θ) P(θ)` |
| Full Bayesian | Distribution | `P(θ | D)` — no single value |

**MAP with Gaussian prior = L2 regularization (Ridge)**: the Gaussian prior `P(θ) ∝ exp(-λ‖θ‖²)` adds a penalty for large weights. Maximizing `P(D|θ)P(θ)` is the same as minimizing the negative log-likelihood plus `λ‖θ‖²`. Ridge regularization is MAP estimation in disguise.

**What breaks**: MAP gives you a point estimate, not a distribution — you lose uncertainty quantification. The specific regularization penalty you choose (L1, L2) corresponds to a specific prior (Laplace, Gaussian). You are always making a prior assumption; the question is whether you make it explicitly.

---

## Conjugate Priors

**The problem**: updating a prior with data requires computing the posterior, which involves an integral that usually has no closed form. For many practical cases, you want exact posteriors without MCMC or variational methods.

**The core insight**: certain pairs of likelihoods and priors are mathematically compatible — the posterior comes out in the same family as the prior, just with updated parameters. These are conjugate pairs. The posterior update is then a simple formula rather than an integral.

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

**What breaks**: conjugate priors only exist for a limited set of likelihood-prior pairs. Most real models (neural networks, complex GLMs) have no conjugate prior — the posterior has no closed form, and you are forced to approximate.

---

## Gaussian Processes (GPs)

**The problem**: parametric regression models (linear regression, neural nets) commit to a specific functional form upfront. But you often do not know the shape of the function you are fitting. You want a model that is flexible over *all* possible functions, while still expressing beliefs about smoothness and uncertainty.

**The core insight**: instead of putting a prior over parameters of a fixed function, put a prior directly over functions. A Gaussian Process defines a distribution over functions: any finite set of function evaluations has a joint Gaussian distribution. The kernel function encodes your prior beliefs about function properties (smoothness, periodicity, scale).

**The mechanics**:

`f(x) ~ GP(m(x), k(x, x'))`

- `m(x)` = mean function (often 0)
- `k(x, x')` = kernel/covariance function (encodes smoothness, periodicity, etc.)

Given training data `(X, y)`, the posterior predictive at new point `x*` is Gaussian with:

```
μ* = k(x*, X) [k(X, X) + σ²I]⁻¹ y
Σ* = k(x*, x*) - k(x*, X) [k(X, X) + σ²I]⁻¹ k(X, x*)
```

The posterior mean `μ*` is the prediction; the posterior variance `Σ*` is the uncertainty — wide where data is sparse, narrow where data is dense.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
gp.fit(X_train, y_train)
y_pred, y_std = gp.predict(X_test, return_std=True)
```

**Key kernels**:
- **RBF (squared exponential)**: smooth functions — used when you expect the function to vary smoothly
- **Matérn**: controls smoothness via ν parameter — less smooth than RBF, more realistic for many physical processes
- **Periodic**: for functions with known periodicity
- **Linear**: Bayesian linear regression as a special case

**What breaks**: `O(n³)` training, `O(n²)` memory — a GP with 10,000 training points requires inverting a 10,000 × 10,000 matrix, which is ~3 minutes and ~800MB. Use sparse GPs (inducing points) for large datasets. GPs also struggle in high dimensions: the kernel must measure meaningful similarity, and in high dimensions, all points become equally distant (curse of dimensionality).

---

## Bayesian Optimization

**The problem**: hyperparameter search requires evaluating the model on a validation set at each candidate hyperparameter configuration. Each evaluation means training from scratch — minutes to hours per evaluation. Grid search wastes evaluations on obviously bad regions. Random search is better but still ignores information from past evaluations.

**The core insight**: treat hyperparameter optimization as a sequential decision problem. Fit a cheap surrogate model (usually a GP) to the evaluations you have collected so far. Use the surrogate to predict which hyperparameter configuration is most promising — balancing exploitation (go where the surrogate thinks is good) with exploration (go where the surrogate is uncertain). Evaluate that configuration, update the surrogate, repeat.

**The mechanics**:

```
For t = 1, 2, ..., T:
    Fit GP to observations {(x_i, f(x_i))}
    Maximize acquisition function to get next x_t
    Evaluate f(x_t) (expensive step)
    Add (x_t, f(x_t)) to observations
```

**Acquisition functions**: trade off where the surrogate predicts a good value (exploitation) vs where uncertainty is high (exploration):

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

**What breaks**: the GP surrogate assumes smooth, continuous hyperparameter landscapes. Hyperparameters like architecture choices (number of layers, activation function type) are discrete or categorical — the GP's notion of distance breaks down. Tools like Optuna handle this with tree-structured surrogates. Bayesian optimization is also sequential by design — it cannot parallelize evaluations without modifications (e.g., batch BO with fantasized observations).

---

## Bayesian Neural Networks (BNNs)

**The problem**: standard neural networks produce point estimates of weights. The network outputs a single prediction with no indication of how confident it is. You cannot distinguish "I've seen thousands of examples like this and am confident" from "this is completely unlike my training data and I'm guessing."

**The core insight**: place a prior over network weights. After seeing data, infer the posterior over weights. At test time, integrate over all plausible weight settings — predictions that are consistent across many weight settings are confident; predictions that vary wildly across weights signal high uncertainty.

**The mechanics**: predictions use the full predictive distribution:

`P(y* | x*, D) = ∫ P(y* | x*, W) P(W | D) dW`

The integral is intractable for any realistic network → approximations needed.

---

### Variational Inference (VI)

**The problem**: the true posterior over weights `P(W | D)` has no closed form. You need to approximate it.

**The core insight**: restrict attention to a tractable family of distributions `q(W; φ)`. Find the member of that family closest to the true posterior by minimizing KL divergence. This transforms an integration problem into an optimization problem.

`L(φ) = E_q[log P(D | W)] - KL[q(W; φ) || P(W)]`
= Expected log-likelihood − KL penalty

This is the **ELBO** (Evidence Lower Bound). Maximizing ELBO = minimizing KL from approximate to true posterior.

**Mean-field VI**: each weight has independent Gaussian `q(w_i) = N(μ_i, σ_i²)`. Sample weights during forward pass, backprop through μ and σ using the reparameterization trick (which makes the sampling operation differentiable):

```python
# Reparameterization trick
mu = nn.Parameter(...)
log_sigma = nn.Parameter(...)
W = mu + torch.exp(log_sigma) * torch.randn_like(mu)  # sample W
```

**Libraries**: `pyro`, `torchbnn`, `tensorflow-probability`

**What breaks**: mean-field VI assumes each weight is independent — a severe approximation. The true posterior has complex correlations between weights. VI tends to underestimate posterior variance (it collapses the posterior toward the mean). Training doubles the parameter count (μ and σ for every weight) and is significantly more expensive than standard training.

---

### MC-Dropout (Gal & Ghahramani, 2016)

**The problem**: VI requires architectural changes and doubles training cost. You want uncertainty estimates from an existing trained network with minimal modification.

**The core insight**: dropout randomly zeros weights during training — this is equivalent to approximate variational inference over the network weights with a specific prior. At inference, keeping dropout active and averaging multiple forward passes gives a Monte Carlo estimate of the predictive distribution. The variance across passes approximates epistemic uncertainty.

```python
model.train()  # keeps dropout active
n_samples = 50
preds = torch.stack([model(x) for _ in range(n_samples)])
mean_pred = preds.mean(0)
uncertainty = preds.var(0)
```

**What breaks**: dropout was not designed as an inference procedure, and the approximation quality is limited. The uncertainty estimates are often poorly calibrated — especially out-of-distribution, where they tend to underestimate uncertainty. The interpretation as Bayesian inference only holds if the dropout rate matches the prior; in practice, this is not tuned.

---

### Deep Ensembles

**The problem**: VI and MC-Dropout produce approximate posteriors with known deficiencies. You want more reliable uncertainty estimates, and are willing to pay more compute.

**The core insight**: train $N$ independent models with different random seeds. Each one finds a different local minimum with a different implicit prior. Ensemble their predictions. This is not formally Bayesian but empirically produces better-calibrated uncertainty than VI or MC-Dropout, because independent training runs explore genuinely different function hypotheses.

- $N=5$ ensembles give strong uncertainty estimates, often beating VI-based BNNs
- Expensive: $N\times$ training compute and memory

**What breaks**: even independent training runs are correlated — they share the same architecture, the same data, and the same training procedure. Ensemble uncertainty collapses when all members make the same systematic error (e.g., all trained on the same biased data). Cost scales linearly with $N$, making large ensembles impractical for expensive models.

---

## Variational Autoencoders (VAEs) — Bayesian View

**The problem**: standard autoencoders learn a compressed representation, but the latent space is irregular — nearby points in latent space may decode to completely different outputs, and large regions may decode to garbage. You cannot use the latent space as a generative model.

**The core insight**: impose a prior over the latent space. Force the encoder to produce a distribution over latent codes rather than a single code. Regularize by penalizing how much this distribution deviates from the prior. The result is a smooth, structured latent space where nearby points decode to similar outputs and any point sampled from the prior decodes to a valid output.

A VAE is a latent variable model:
- Prior: `P(z) = N(0, I)`
- Likelihood: `P(x | z) = N(μ_θ(z), σ_θ(z))`
- Approximate posterior: `q_φ(z | x) = N(μ_φ(x), σ_φ(x))`

Training maximizes the ELBO:
`L = E_q[log P(x | z)] - KL[q(z | x) || P(z)]`

The reconstruction term pushes the decoder to faithfully reproduce inputs; the KL term pushes the encoder's distribution toward the prior, regularizing the latent space.

**What breaks**: the Gaussian posterior assumption means the VAE cannot model complex multi-modal posteriors. The reconstruction loss (MSE for images) tends to produce blurry outputs because it averages over modes. The KL penalty can "collapse" — the encoder learns to ignore the input and always output the prior, and the decoder learns to ignore the latent code entirely.

---

## Probabilistic Programming

**The problem**: custom Bayesian models require deriving custom inference procedures — a significant implementation burden that prevents practitioners from experimenting with model structure.

**The core insight**: express the model as a probabilistic program (a generative story) and let the inference engine handle the math. The program specifies the prior and likelihood; the library automatically constructs the posterior computation.

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

**What breaks**: automatic inference is not magic. Complex models with many latent variables converge slowly with VI. MCMC is exact in the limit but requires careful tuning (step sizes, warmup). Hierarchical models with correlated parameters can cause both VI and MCMC to fail silently — the inference converges, but to the wrong posterior.

---

## When to Use Bayesian Methods

| Situation | Use |
|-----------|-----|
| Small data, need uncertainty | Gaussian Processes, BNNs |
| Hyperparameter tuning (expensive eval) | Bayesian Optimization |
| Need calibrated probabilities | MC-Dropout, Deep Ensembles |
| Prior domain knowledge available | Bayesian models with informative priors |
| Anomaly detection with uncertainty | GP-based methods |

**Avoid when**: large data (GPs don't scale), low-latency inference (sampling is slow), or when a well-calibrated frequentist model suffices.

---

## Key Interview Points

- MAP = MLE + prior; Ridge regression is MAP with Gaussian prior.
- GPs give principled uncertainty but scale as O(n³).
- MC-Dropout is the cheapest way to get uncertainty estimates from neural nets.
- Deep ensembles often outperform variational BNNs for uncertainty, at the cost of compute.
- Bayesian Optimization is the standard approach for expensive hyperparameter tuning (Optuna, Ax, HyperOpt internally use it).
- ELBO = reconstruction term + KL regularization; used in VAEs and VI.
