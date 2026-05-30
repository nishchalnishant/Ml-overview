---
module: Foundations
topic: Revision Card
subtopic: ""
status: unread
tags: [foundations, math, revision, cheatsheet]
---
# Foundations — Mathematical and Conceptual Reference

**Read this before any other section.** These are the primitives everything else builds on. This document covers linear algebra, calculus for ML, probability and statistics, information theory, optimization theory, generalization theory, MLE/MAP estimation, and feature engineering fundamentals.

---

## Linear Algebra Essentials

| Concept | Key fact | Why it matters |
|---------|----------|----------------|
| Matrix multiply | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$ | Neural net forward pass is matrix multiplication |
| Transpose | $(AB)^T = B^T A^T$ | Backprop gradient derivations |
| Eigendecomposition | $Av = \lambda v$ | PCA, understanding covariance |
| SVD | $A = U\Sigma V^T$ | Dimensionality reduction, LoRA |
| Dot product | $a \cdot b = \|a\|\|b\|\cos\theta$ | Attention similarity, cosine similarity |

**Gotcha:** Matrix multiplication is not commutative — $AB \neq BA$ in general.

---

## Calculus for Machine Learning

### Chain Rule and Backpropagation

**Chain rule (the engine of backprop):**
$$\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}$$

In multiple dimensions: $\nabla_x L = J_f^T \nabla_y L$, where $J_f$ is the Jacobian of $f: x \mapsto y$.

**Gradient:** vector of partial derivatives. Points in direction of steepest ascent. Gradient descent steps opposite: $\theta \leftarrow \theta - \alpha \nabla_\theta L$

**Partial derivative intuition:** how much does the output change if I wiggle only this one input, holding all others fixed? The gradient is the vector of all partial derivatives simultaneously.

**Why it matters:** every training step computes the gradient of loss with respect to all parameters via the chain rule and steps in the negative direction. In deep networks, this is a product of Jacobians — each layer contributes one factor.

### Jacobian and Hessian

**Jacobian** of $f: \mathbb{R}^n \to \mathbb{R}^m$:
$$J_{ij} = \frac{\partial f_i}{\partial x_j} \in \mathbb{R}^{m \times n}$$

In backprop, the upstream gradient $g^T$ is left-multiplied by the Jacobian: $g_{\text{in}}^T = g_{\text{out}}^T J$. This is a vector-Jacobian product (VJP) — never materialize the full Jacobian in practice.

**Hessian** of scalar $f: \mathbb{R}^n \to \mathbb{R}$:
$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

The Hessian is symmetric for smooth $f$. Its eigenvalues determine curvature. At a critical point ($\nabla f = 0$):
- All eigenvalues positive → local minimum
- All eigenvalues negative → local maximum
- Mixed signs → saddle point (no local extremum)

Neural networks have many saddle points. Gradient descent typically escapes them via gradient noise, but they slow down convergence.

### Taylor Series and Second-Order Methods

The second-order Taylor expansion of a loss around current parameters $\theta_0$:
$$\mathcal{L}(\theta) \approx \mathcal{L}(\theta_0) + \nabla \mathcal{L}^T (\theta - \theta_0) + \frac{1}{2}(\theta - \theta_0)^T H (\theta - \theta_0)$$

**Newton's method** minimizes this quadratic approximation exactly:
$$\theta \leftarrow \theta - H^{-1} \nabla \mathcal{L}$$

This achieves quadratic convergence near a minimum (error squares each step) versus gradient descent's linear convergence. However, computing and inverting the $n \times n$ Hessian is $O(n^3)$ — infeasible for millions of parameters. Newton's method is used for small-scale problems (logistic regression with few features) and as inspiration for quasi-Newton methods.

**L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno):** approximates $H^{-1}$ using only the last $k$ gradient updates, using $O(kn)$ memory instead of $O(n^2)$. The standard optimizer for non-stochastic ML problems (e.g., training SVMs, logistic regression on large-but-finite datasets with exact gradients). Not used for deep learning because stochastic gradients make the Hessian approximation unreliable.

### Convexity

A function $f$ is **convex** if the line segment between any two points on the graph lies above (or on) the graph:
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y), \quad \forall \lambda \in [0,1]$$

Equivalently: $H \succeq 0$ (positive semidefinite Hessian) everywhere.

**Why it matters:**
- Convex loss → gradient descent finds the global minimum (no local minima, no saddle points can trap it)
- Linear regression (MSE), logistic regression, SVM, Lasso, Ridge all have convex losses
- Neural networks are non-convex — GD finds a good local minimum (empirically sufficient, but no guarantee)
- Sum of convex functions is convex. L1 and L2 penalties are convex — adding them to a convex loss preserves convexity.

**Strictly convex:** unique global minimum. L2 (Ridge) loss is strictly convex. L1 (Lasso) is convex but not strictly convex — the minimum may not be unique when features are correlated.

### Gradients of Common ML Operations

| Operation | Expression | Gradient wrt $x$ |
|-----------|-----------|-------------------|
| Linear | $f = w^T x$ | $w$ |
| Quadratic form | $f = x^T A x$ | $(A + A^T)x$ (if $A$ symmetric: $2Ax$) |
| MSE | $f = \|Xw - y\|_2^2$ | $2X^T(Xw - y)$ wrt $w$ |
| L2 norm² | $f = \|w\|_2^2$ | $2w$ |
| Sigmoid | $\sigma(x) = 1/(1+e^{-x})$ | $\sigma(x)(1-\sigma(x))$ |
| Softmax + CE | $f = -\log \hat{y}_{true}$ | $\hat{y} - y$ (predicted - one-hot) |
| Cross-entropy | $f = -y \log \hat{y}$ | $-y/\hat{y}$ |

The softmax + cross-entropy gradient simplifying to $\hat{y} - y$ is why this combination is numerically preferred — the gradient is bounded and well-behaved regardless of how extreme the logits are.

---

## Probability and Statistics

### Probability Fundamentals

| Concept | Definition | Key use |
|---------|-----------|---------|
| Conditional probability | $P(A|B) = \frac{P(A \cap B)}{P(B)}$ | Naive Bayes, Bayes' theorem |
| Bayes' theorem | $P(H|E) = \frac{P(E|H)P(H)}{P(E)}$ | Bayesian inference, priors |
| Expectation | $\mathbb{E}[X] = \sum_x x \cdot P(X=x)$ | Loss functions, risk |
| Variance | $\text{Var}(X) = \mathbb{E}[(X - \mu)^2]$ | Bias-variance tradeoff |
| Covariance | $\text{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$ | Feature correlation, PCA |
| KL Divergence | $D_{KL}(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}$ | Cross-entropy loss, RLHF KL penalty |

**Law of total expectation:** $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X|Y]]$

**Law of total variance:** $\text{Var}(X) = \mathbb{E}[\text{Var}(X|Y)] + \text{Var}(\mathbb{E}[X|Y])$

**Cross-entropy:** $H(P,Q) = -\sum_x P(x)\log Q(x)$ = $H(P) + D_{KL}(P\|Q)$. Minimizing cross-entropy ≡ minimizing KL divergence from true distribution (the entropy term $H(P)$ is constant wrt model parameters).

### Key Probability Distributions

| Distribution | PMF/PDF | Mean | Variance | ML Use |
|-------------|---------|------|----------|--------|
| Bernoulli($p$) | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | Binary classification output |
| Binomial($n,p$) | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | Multiple binary trials |
| Categorical($\mathbf{p}$) | $\prod_k p_k^{[x=k]}$ | — | — | Softmax output, multi-class |
| Poisson($\lambda$) | $e^{-\lambda}\lambda^k/k!$ | $\lambda$ | $\lambda$ | Count data, NLP token counts |
| Normal $\mathcal{N}(\mu,\sigma^2)$ | $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | Weight init, noise modeling, CLT |
| Multivariate Normal | $(2\pi)^{-d/2}|\Sigma|^{-1/2}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$ | $\mu$ | $\Sigma$ | Gaussian processes, GMMs |
| Uniform($a,b$) | $1/(b-a)$ | $(a+b)/2$ | $(b-a)^2/12$ | Random initialization range |
| Exponential($\lambda$) | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | Wait times, NHST |
| Beta($\alpha,\beta$) | $x^{\alpha-1}(1-x)^{\beta-1}/B(\alpha,\beta)$ | $\alpha/(\alpha+\beta)$ | — | Prior for probabilities (conjugate to Bernoulli) |
| Dirichlet($\alpha$) | $\prod_k x_k^{\alpha_k-1}/B(\alpha)$ | $\alpha_k/\sum\alpha_i$ | — | Prior for categorical distributions |

**Why the Normal distribution dominates ML:**
1. Central Limit Theorem: sums of many independent random variables converge to Normal regardless of their distribution
2. Maximum entropy distribution for a given mean and variance — the "least informative" prior
3. Mathematically tractable: product of Gaussians is Gaussian; linear transforms of Gaussians are Gaussian
4. Many optimization objectives (MSE loss) implicitly assume Gaussian noise

### Bayes' Theorem in Depth

$$P(\theta | \mathcal{D}) = \frac{P(\mathcal{D}|\theta) \cdot P(\theta)}{P(\mathcal{D})}$$

- $P(\theta | \mathcal{D})$: **posterior** — updated belief about parameters after seeing data
- $P(\mathcal{D}|\theta)$: **likelihood** — probability of observing this data given parameters
- $P(\theta)$: **prior** — belief about parameters before seeing data
- $P(\mathcal{D})$: **marginal likelihood** (evidence) — normalizing constant; expensive to compute ($\int P(\mathcal{D}|\theta)P(\theta)d\theta$)

**Practical consequence:** Computing $P(\mathcal{D})$ exactly requires integrating over all possible parameters — intractable for complex models. This is why approximate inference (variational inference, MCMC) exists.

### Hypothesis Testing

**Null hypothesis ($H_0$):** the default "nothing is happening" claim (e.g., the two models perform identically). **Alternative hypothesis ($H_1$):** what you're trying to show.

**p-value:** probability of observing a test statistic at least as extreme as the one observed, *assuming $H_0$ is true*. It is NOT: the probability that $H_0$ is true; the probability that your result is due to chance; the probability your finding will replicate.

**Type I error (false positive):** reject $H_0$ when it is true. Probability = $\alpha$ (significance level, typically 0.05). Controlled by design.

**Type II error (false negative):** fail to reject $H_0$ when $H_1$ is true. Probability = $\beta$. Power = $1 - \beta$.

**Statistical power:** probability of detecting an effect of a given size if it truly exists. Increases with: larger sample size, larger effect size, higher $\alpha$.

**Multiple testing problem:** running 20 independent tests at $\alpha=0.05$ expects 1 false positive even if all nulls are true. Bonferroni correction: use $\alpha/m$ for $m$ tests. False Discovery Rate (FDR) control (Benjamini-Hochberg) is less conservative for large-scale testing (e.g., genomics, feature selection).

**Confidence interval:** if you repeated the experiment and recomputed the CI infinitely, $1-\alpha$ fraction of them would contain the true parameter. A 95% CI does NOT mean "95% probability the true value is in this interval" — the true value is either in it or not.

**T-test:** compare means of two samples. Assumes approximately normal data (or large samples by CLT). Paired t-test when samples are matched (same users before/after).

**Chi-squared test:** test independence between categorical variables. Used to validate that data splits are drawn from same distribution.

---

## Maximum Likelihood Estimation (MLE)

### The Core Principle

**Problem:** You have a dataset $\mathcal{D} = \{x_1, \ldots, x_n\}$ assumed to come from a parametric distribution $P(x|\theta)$. What value of $\theta$ best explains this data?

**Core insight:** Choose $\theta$ to maximize the probability of observing the data you actually observed.

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(\mathcal{D}|\theta) = \arg\max_\theta \prod_{i=1}^n P(x_i|\theta)$$

In practice, take the log (log-likelihood) to convert the product to a sum and improve numerical stability:

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log P(x_i|\theta)$$

Minimizing negative log-likelihood is identical to maximizing likelihood — and negative log-likelihood is the standard loss function for probabilistic models.

### MLE Derivations for Common Models

**Linear regression:** Assume $y_i = w^T x_i + \epsilon_i$ where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$. The likelihood is:

$$P(\mathcal{D}|w) = \prod_i \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(y_i - w^T x_i)^2}{2\sigma^2}\right)$$

Taking the negative log-likelihood and dropping constants:
$$-\log P(\mathcal{D}|w) \propto \sum_i (y_i - w^T x_i)^2 = \|y - Xw\|_2^2$$

**Conclusion: MSE loss = MLE under Gaussian noise assumption.** If you believe errors are Gaussian, MSE is the principled loss.

**Logistic regression:** Assume $P(y=1|x,w) = \sigma(w^T x)$. Negative log-likelihood:

$$-\log P(\mathcal{D}|w) = -\sum_i [y_i \log \sigma(w^T x_i) + (1-y_i)\log(1-\sigma(w^T x_i))]$$

This is binary cross-entropy. **Conclusion: Binary cross-entropy loss = MLE under Bernoulli output assumption.**

**Multi-class classification:** Assume $P(y=k|x,w) = \text{softmax}(w_k^T x)$. Negative log-likelihood = categorical cross-entropy. **Conclusion: Cross-entropy loss = MLE for any classification problem modeled as a categorical distribution.**

**MLE properties:**
- **Consistency:** $\hat{\theta}_{\text{MLE}} \to \theta^*$ as $n \to \infty$ (converges to true value)
- **Asymptotic efficiency:** achieves the Cramér-Rao lower bound (lowest possible variance among unbiased estimators) for large $n$
- **Invariance:** if $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$
- **Failure mode:** can overfit with small data (maximum likelihood explains the observed data perfectly, which may require extreme parameter values)

---

## Maximum A Posteriori (MAP) Estimation

**Problem:** MLE overfits with small data. Can we incorporate prior knowledge about reasonable parameter values?

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta|\mathcal{D}) = \arg\max_\theta P(\mathcal{D}|\theta) P(\theta)$$

Taking log:
$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[\log P(\mathcal{D}|\theta) + \log P(\theta)\right]$$

**MAP = MLE + log prior penalty.** The prior acts as a regularizer.

### MAP and Regularization Are Equivalent

**L2 regularization (Ridge) = MAP with Gaussian prior:**

If $P(\theta) = \mathcal{N}(0, \tau^2 I)$, then $\log P(\theta) = -\frac{1}{2\tau^2}\|\theta\|_2^2 + \text{const}$.

$$\hat{\theta}_{\text{MAP}} = \arg\min_\theta \underbrace{\|y - X\theta\|_2^2}_{\text{negative log-likelihood}} + \underbrace{\frac{\sigma^2}{\tau^2}\|\theta\|_2^2}_{\text{L2 penalty}}$$

The regularization strength $\lambda = \sigma^2/\tau^2$ encodes the ratio of noise variance to prior variance.

**L1 regularization (Lasso) = MAP with Laplace prior:**

If $P(\theta) \propto \exp(-|\theta|/b)$ (Laplace distribution), the log prior is $-|\theta|/b$:

$$\hat{\theta}_{\text{MAP}} = \arg\min_\theta \|y - X\theta\|_2^2 + \frac{\sigma^2}{b}\|\theta\|_1$$

The Laplace distribution has heavier tails than Gaussian and a sharper peak at zero — it assigns more prior probability to exact zeros, which is why MAP with a Laplace prior produces sparse solutions.

**Summary table:**

| Method | Formula | Equivalent to | Assumes |
|--------|---------|---------------|---------|
| MLE | $\arg\max \log P(\mathcal{D}|\theta)$ | ERM (no regularization) | No preference for any $\theta$ |
| MAP + Gaussian prior | $\arg\max \log P(\mathcal{D}|\theta) - \lambda\|\theta\|_2^2$ | L2 regularization | Parameters are small (Gaussian) |
| MAP + Laplace prior | $\arg\max \log P(\mathcal{D}|\theta) - \lambda\|\theta\|_1$ | L1 regularization | Parameters are sparse (Laplace) |
| Full Bayes | $P(\theta|\mathcal{D}) \propto P(\mathcal{D}|\theta)P(\theta)$ | Posterior distribution | Posterior fully characterizes uncertainty |

---

## Generalization Theory

### The Fundamental Problem of Learning

You observe $n$ training examples from an unknown distribution $\mathcal{D}$. You pick a hypothesis $h$ that minimizes training loss. The question is: how well will $h$ perform on new examples from $\mathcal{D}$ — its *generalization error*?

The gap between training error and generalization error is the *generalization gap*. The goal of learning theory is to bound this gap.

### Empirical Risk Minimization (ERM)

**Empirical risk:** average loss on the training set:
$$\hat{R}(h) = \frac{1}{n}\sum_{i=1}^n \ell(h(x_i), y_i)$$

**True risk:** expected loss over the full data distribution:
$$R(h) = \mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(h(x), y)]$$

**ERM principle:** choose $h^* = \arg\min_{h \in \mathcal{H}} \hat{R}(h)$. This is what every supervised learning algorithm does.

**The problem:** $\hat{R}(h^*)$ may be much lower than $R(h^*)$ — the model overfits. The gap is larger when $\mathcal{H}$ is more expressive (can achieve lower $\hat{R}$ on noise).

### VC Dimension

The **Vapnik-Chervonenkis (VC) dimension** of a hypothesis class $\mathcal{H}$ is the largest set of points that $\mathcal{H}$ can *shatter* (correctly classify for every possible labeling).

- Linear classifiers in $\mathbb{R}^d$: VC dim = $d+1$
- Polynomial classifiers of degree $k$ in $\mathbb{R}^d$: VC dim = $O(d^k)$
- Neural networks: VC dim grows with parameters (approximately)

**Fundamental theorem (VC theory):** For any $\delta > 0$, with probability $\geq 1-\delta$ over the training sample:

$$R(h) \leq \hat{R}(h) + O\!\left(\sqrt{\frac{\text{VC-dim}(\mathcal{H}) \cdot \log(n/\delta)}{n}}\right)$$

**Interpretation:**
- Generalization improves with more data $n$
- Generalization worsens with more expressive hypothesis class (higher VC dim)
- The bound is uniform over all $h \in \mathcal{H}$ — it holds even for the worst-case choice of $h$
- The bound is often loose in practice (neural networks generalize far better than VC theory predicts)

### Bias-Variance Decomposition (Formal)

For regression with squared loss, the expected prediction error at a point $x$ decomposes as:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{\left(\mathbb{E}[\hat{f}(x)] - f^*(x)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[\left(\hat{f}(x) - \mathbb{E}[\hat{f}(x)]\right)^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible noise}}$$

Where:
- $f^*(x)$ is the true function
- $\hat{f}(x)$ is the model's prediction (a random variable, varying across training sets)
- $\mathbb{E}$ is over the randomness in training data sampling

**Bias** measures how far off the model is on average — how wrong its structural assumptions are. **Variance** measures how sensitive the model is to which particular training set it was trained on.

This decomposition is for a *fixed* point $x$. Integrated over the data distribution:
$$\text{Expected total error} = \text{Bias}^2_{\text{avg}} + \text{Variance}_{\text{avg}} + \sigma^2$$

### PAC Learning

**Probably Approximately Correct (PAC) learning:** a learning algorithm is PAC-learnable if, for any $\epsilon > 0$ (accuracy) and $\delta > 0$ (confidence), there exists a sample size $n(\epsilon, \delta)$ such that with $n \geq n(\epsilon, \delta)$ examples, the algorithm returns a hypothesis with error $\leq \epsilon$ with probability $\geq 1-\delta$.

For a finite hypothesis class $|\mathcal{H}|$:
$$n \geq \frac{1}{\epsilon}\left(\log|\mathcal{H}| + \log\frac{1}{\delta}\right)$$

For infinite hypothesis classes, VC dimension replaces $\log|\mathcal{H}|$.

**Sample complexity** tells you how much data you need to learn a concept. This is why:
- Simple models (low VC dim) need less data
- Complex models need more data to generalize
- In practice, neural networks violate these bounds — they generalize even in the interpolating regime (training error = 0, but still generalize well)

### Double Descent

Classical statistics says generalization error has a U-shape with model complexity (bias-variance tradeoff). Modern deep learning exhibits **double descent**: after a critical regime where the model first fits the training data exactly, further increasing model size causes generalization error to *decrease again*.

This occurs because:
- Overparameterized models have many interpolating solutions
- Gradient descent finds the minimum-norm interpolator, which can generalize well
- The classical bias-variance tradeoff describes the under-parameterized regime; the second descent is in the over-parameterized regime

**Practical consequence:** very large models can be better than intermediate-sized models at the same test error. The optimal model size is not always in the "Goldilocks" zone — sometimes bigger is simply better.

### No Free Lunch Theorem

**Statement:** averaged over all possible data-generating distributions, every learning algorithm performs identically. No algorithm is universally superior.

**Implication:** performance claims for ML algorithms are always conditional on the problem structure (data distribution). "XGBoost always wins on tabular data" is shorthand for "XGBoost makes assumptions that happen to match most tabular data distributions better than alternatives." Those assumptions can fail.

**Practical takeaway:** always validate on your specific data. Leaderboard results from other datasets are priors, not guarantees. Algorithm selection is itself a learning problem.

---

## Information Theory

### Entropy

**Shannon entropy** of a discrete random variable $X$ with distribution $P$:
$$H(X) = -\sum_x P(x)\log_2 P(x) \quad \text{(bits)}$$

Or with natural log (nats):
$$H(X) = -\sum_x P(x)\ln P(x)$$

**Properties:**
- $H(X) \geq 0$ always
- $H(X) = 0$ when one outcome has probability 1 (no uncertainty)
- $H(X)$ is maximized by the uniform distribution: $H_{\max} = \log_2 k$ for $k$ equally likely outcomes
- **Concavity:** $H(\lambda P_1 + (1-\lambda)P_2) \geq \lambda H(P_1) + (1-\lambda)H(P_2)$

**Differential entropy** for continuous $X$ with density $p(x)$:
$$h(X) = -\int p(x)\log p(x)\, dx$$

Can be negative (unlike discrete entropy). The Gaussian maximizes differential entropy among distributions with fixed variance: $h(\mathcal{N}(\mu,\sigma^2)) = \frac{1}{2}\ln(2\pi e \sigma^2)$.

### Cross-Entropy and KL Divergence

**Cross-entropy** between distributions $P$ (true) and $Q$ (predicted):
$$H(P, Q) = -\sum_x P(x)\log Q(x)$$

Decomposition:
$$H(P, Q) = H(P) + D_{KL}(P\|Q)$$

Since $H(P)$ is fixed (it depends on the data, not the model), minimizing cross-entropy is equivalent to minimizing $D_{KL}(P\|Q)$ — pulling the model distribution toward the true distribution.

**Cross-entropy loss:** the standard classification loss is:
$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \sum_c y_{ic}\log \hat{y}_{ic}$$

where $y_{ic}$ is the one-hot true label and $\hat{y}_{ic}$ is the predicted probability. For each example, this equals $-\log \hat{y}_{i,\text{true}}$ — the negative log probability assigned to the correct class.

**KL divergence** properties:
- $D_{KL}(P\|Q) \geq 0$ (Gibbs' inequality)
- $D_{KL}(P\|Q) = 0$ iff $P = Q$ almost everywhere
- **Not symmetric:** $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ in general
- $D_{KL}(P\|Q)$ is undefined when $Q(x) = 0$ and $P(x) > 0$ (the model assigns zero probability to an event that occurs)

**Forward vs. reverse KL:**
- $D_{KL}(P\|Q)$: minimized by $Q$ that covers all modes of $P$ but may add spurious probability elsewhere (mean-seeking). Used in maximum likelihood (variational inference as ELBO).
- $D_{KL}(Q\|P)$: minimized by $Q$ that concentrates on one mode of $P$ (mode-seeking). Used in some variational methods.

### Mutual Information

$$I(X; Y) = \sum_{x,y} P(x,y)\log\frac{P(x,y)}{P(x)P(y)} = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

$I(X;Y)$ measures how much knowing $Y$ reduces uncertainty about $X$ (and vice versa).

**Properties:**
- $I(X;Y) \geq 0$; equals 0 iff $X$ and $Y$ are independent
- $I(X;Y) = D_{KL}(P(X,Y)\|P(X)P(Y))$ — measures how far the joint is from the product of marginals
- **Data processing inequality:** for any function $f$, $I(X; f(Y)) \leq I(X; Y)$. Processing cannot increase information.

**ML uses:**
- Feature selection: select features with high mutual information with the label
- Information bottleneck (Tishby et al.): learn representations that compress $X$ while preserving $I(Z; Y)$ — a framework for understanding what neural networks learn
- Maximum mutual information objectives in self-supervised learning

### Information Gain (Decision Trees)

When splitting a dataset on feature $A$:

$$\text{IG}(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

CART uses Gini impurity instead of entropy (computationally cheaper; similar splits in practice):
$$\text{Gini}(S) = 1 - \sum_c p_c^2$$

---

## Optimization Theory

### Gradient Descent Convergence

**For convex, $L$-smooth functions** (gradient is Lipschitz with constant $L$), gradient descent with step size $\alpha \leq 1/L$ converges at rate:

$$\mathcal{L}(\theta_t) - \mathcal{L}(\theta^*) \leq \frac{\|\theta_0 - \theta^*\|_2^2}{2\alpha t}$$

This is $O(1/t)$ convergence — need $O(1/\epsilon)$ steps to reach $\epsilon$-optimal.

**For $\mu$-strongly convex functions** (additionally, $H \succeq \mu I$), gradient descent converges geometrically:
$$\mathcal{L}(\theta_t) - \mathcal{L}(\theta^*) \leq \left(1 - \frac{\mu}{L}\right)^t (\mathcal{L}(\theta_0) - \mathcal{L}(\theta^*))$$

The convergence rate is $\left(1 - \mu/L\right)^t$ — the condition number $\kappa = L/\mu$ controls convergence. High $\kappa$ → many iterations needed. **This is why poorly conditioned features (different scales) slow training.**

### SGD Convergence and Noise

Stochastic gradient descent uses a noisy gradient estimate $\tilde{g}_t = \nabla \mathcal{L}(\theta_t; \xi_t)$ where $\xi_t$ is a random mini-batch. For convex losses:

- SGD converges at rate $O(1/\sqrt{t})$ — slower than full-batch GD's $O(1/t)$
- But each step costs $1/n$ of full batch GD — SGD does many more steps in the same wall-clock time
- **Noise variance scales inversely with batch size:** $\text{Var}(\tilde{g}) \propto 1/B$
- **Linear scaling rule:** if you multiply batch size by $k$, multiply learning rate by $k$ (approximately), to keep update dynamics similar

SGD noise is not just a nuisance. It provides:
1. **Implicit regularization:** gradient noise prevents settling into sharp minima with high Hessian trace; prefers flat minima that generalize better
2. **Escaping saddle points:** exact GD can get trapped; SGD escapes via noise
3. **Exploration:** noise helps find better loss basins in non-convex landscapes

### Adam and Adaptive Methods

**Why adaptive rates help:** loss landscapes for neural nets have very different curvatures in different directions. A single global learning rate is simultaneously too large in high-curvature directions and too small in low-curvature ones. This produces oscillation in some directions and slow progress in others.

**Adam update (full derivation):**
```
g_t = ∇L(θ_{t-1})              # gradient
m_t = β₁ m_{t-1} + (1-β₁) g_t  # first moment (EMA of gradients)
v_t = β₂ v_{t-1} + (1-β₂) g_t² # second moment (EMA of squared gradients)
m̂_t = m_t / (1 - β₁ᵗ)          # bias correction (moments start near 0)
v̂_t = v_t / (1 - β₂ᵗ)
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

Typical defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\alpha = 3\times10^{-4}$.

$\sqrt{v̂_t}$ approximates the RMS of recent gradients. Dividing by it effectively normalizes the update by a per-parameter adaptive scale. Parameters with consistently large gradients get smaller effective learning rates; rarely-updated sparse parameters get larger effective rates.

**Adam failure modes:**
- Can converge to worse solutions than SGD+momentum in some settings (not guaranteed to find flat minima)
- Weight decay in standard Adam is entangled with adaptive scaling — the actual decay depends on the gradient history. **AdamW fixes this** by decoupling weight decay from gradient scaling.
- "Adam is the right choice for getting started; SGD+momentum can win with more tuning and longer training."

**Learning rate warmup:** at initialization, gradient estimates have high variance (bias in Adam's moment estimates). A low initial LR prevents destructive large updates before the moments are reliable. Warmup duration: typically 1–10% of total training steps. Mandatory for transformers.

### Loss Landscape Geometry

**Flat vs. sharp minima:**

A *sharp minimum* has large Hessian eigenvalues — small perturbations of parameters cause large loss increases. A *flat minimum* has small Hessian eigenvalues — the loss is insensitive to small perturbations.

Flat minima generalize better because the distribution shift between training and test data is equivalent to a parameter perturbation: a flat minimum remains low-loss under the shift, while a sharp minimum may not.

**Sharpness-Aware Minimization (SAM):** explicitly seeks flat minima by minimizing $\max_{\|\epsilon\| \leq \rho} \mathcal{L}(\theta + \epsilon)$ — the loss at the worst-case perturbation within a ball of radius $\rho$.

**Saddle points:** in $n$-dimensional parameter spaces, a critical point with mixed Hessian eigenvalues (some positive, some negative) is a saddle point. In high dimensions, saddle points are far more common than local minima. Gradient descent escapes them via noise (SGD) or by using momentum (Adam).

---

## Statistics Fast Reference

**Central Limit Theorem:** the mean of $n$ i.i.d. samples from any distribution with mean $\mu$ and variance $\sigma^2$ converges in distribution to $\mathcal{N}(\mu, \sigma^2/n)$ as $n \to \infty$. Why it matters: justifies Gaussian assumptions in many ML methods and enables asymptotic inference.

**Law of Large Numbers (LLN):** the sample mean converges to the true expectation as $n \to \infty$. Foundation of ERM — with enough data, training loss approximates test loss.

**Estimator properties:**

| Property | Definition |
|----------|-----------|
| Unbiased | $\mathbb{E}[\hat{\theta}] = \theta^*$ |
| Consistent | $\hat{\theta} \to \theta^*$ as $n \to \infty$ |
| Efficient | Achieves Cramér-Rao lower bound (minimum variance among unbiased estimators) |
| Sufficient statistic | Captures all information about $\theta$ in the data |

**Cramér-Rao lower bound:** for any unbiased estimator $\hat{\theta}$:
$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$
where $I(\theta) = \mathbb{E}\left[\left(\frac{\partial}{\partial\theta}\log P(x|\theta)\right)^2\right]$ is the **Fisher information** — how much a single observation tells you about $\theta$.

MLE achieves this bound asymptotically — it is the most efficient unbiased estimator for large $n$.

**Distributions to know:**

| Distribution | Params | Use in ML |
|-------------|--------|----------|
| Normal $\mathcal{N}(\mu,\sigma^2)$ | mean, variance | Weight init, noise modeling |
| Bernoulli | p | Binary classification output |
| Categorical | $p_1,...,p_k$ | Softmax output |
| Uniform | a, b | Random initialization range |
| Beta | $\alpha, \beta$ | Prior over probabilities |
| Dirichlet | $\alpha_1,\ldots,\alpha_k$ | Prior over categorical distributions |

**p-value:** probability of seeing this result (or more extreme) if null hypothesis is true. Threshold (0.05) is arbitrary. Low p-value = strong evidence against null, not proof of effect size.

**Confidence interval:** if repeated infinitely, 95% of computed CIs would contain true parameter. Does NOT mean 95% probability that this specific interval contains it.

---

## Optimization Fundamentals

**Gradient descent:**
$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(\theta_t)$$

**Variants:**
- **Batch GD:** full dataset per step — accurate gradient, slow per update
- **Stochastic GD (SGD):** one example per step — noisy gradient, fast, can escape local minima
- **Mini-batch GD:** k examples per step — balance of accuracy and speed; standard in practice

**Learning rate too high:** loss oscillates or diverges. Too low: very slow convergence. **Warmup:** start low, ramp up, then decay (cosine schedule).

**Convex vs non-convex:** convex loss → GD finds global minimum. Neural nets are non-convex → GD finds good local minimum (in practice, sufficient).

---

## Information Theory in 2 Minutes

**Entropy:** $H(X) = -\sum_x P(x)\log P(x)$ — expected bits to encode a message. Higher entropy = more uncertainty.

**Cross-entropy loss:** $\mathcal{L} = -\sum_c y_c \log \hat{y}_c$ — how many bits needed to encode true labels with predicted distribution. Minimizing it aligns predictions with truth.

**KL divergence:** non-symmetric distance between distributions. $D_{KL}(P\|Q) \geq 0$, equals 0 only when P=Q. Used in VAEs, RLHF KL penalty, knowledge distillation.

---

## Feature Engineering

### Why Features Matter

The choice of feature representation is often more impactful than model selection. A decision tree on informative hand-crafted features can outperform a deep neural net on raw uninformative features. "Features" encode domain knowledge that the model would otherwise have to learn from scratch with more data.

### Numerical Features

**Scaling:**
- **Standardization** ($z$-score): $x' = (x - \mu)/\sigma$. Produces mean 0, std 1. Required for algorithms sensitive to scale: SVMs, linear regression, PCA, KNN, neural nets.
- **Min-max normalization:** $x' = (x - x_{\min})/(x_{\max} - x_{\min})$. Produces [0,1] range. Sensitive to outliers.
- **Robust scaling:** $x' = (x - \text{median})/\text{IQR}$. Resistant to outliers.
- **Log transform:** $x' = \log(1+x)$. Compresses heavy-tailed distributions; useful for counts, monetary values, population data. Never apply to zero or negative values.
- **Power transforms (Box-Cox, Yeo-Johnson):** generalize log transform; find the best-fit monotone transform to make a distribution normal.

**Always fit scalers on training data only.** Fitting on test/validation data is leakage.

**Binning:** discretize continuous features into bins. Useful when the relationship is non-monotone (e.g., age affects risk differently for young vs. middle-aged vs. elderly). Reduces sensitivity to outliers. Can be combined with one-hot encoding.

### Categorical Features

**One-hot encoding:** create a binary column per category. Produces sparse, high-dimensional vectors. Loses no information; works with all algorithms. Problem: high cardinality (e.g., zip codes) explodes dimensionality.

**Label encoding:** map categories to integers 0, 1, ..., k-1. Compact but implies ordinal ordering. Only appropriate for ordinal categories or tree-based models (which don't interpret numerical magnitude).

**Target encoding (mean encoding):** replace each category value with the mean of the target variable for that category. Powerful but high risk of leakage — must be done using out-of-fold means from cross-validation, not the full training target.

**Embeddings:** for very high cardinality (user IDs, product IDs), learn a dense low-dimensional embedding as part of the model. Captures semantic similarity. Requires enough data per category.

**Hashing:** `hash(category) % k`. Fixed-size output regardless of cardinality. Loses reversibility and can collide, but works well for very high cardinality.

### Interaction and Derived Features

**Polynomial features:** $x_1^2, x_2^2, x_1 x_2$. Captures non-linear relationships in linear models. Explodes dimensionality: $d$ features → $O(d^k)$ for degree $k$.

**Ratio features:** $x_1 / x_2$ when the ratio is meaningful (e.g., price per square foot, debt-to-income ratio).

**Difference features:** $x_{t} - x_{t-1}$ for time series — rate of change often more predictive than absolute value.

**Aggregate features over groups:** mean/std/min/max of a feature per user, per store, per date. Captures group-level context.

**Domain-specific transforms:** distance from a reference point, hour of day, day of week (cyclical encoding: $\sin(2\pi h/24)$, $\cos(2\pi h/24)$ — wraps 23:00 and 0:00 close together in feature space).

### Cyclical Encoding

For periodic features (hour of day, day of week, month of year, angle), integer encoding treats midnight and 11pm as maximally different. Correct encoding:

$$\sin\!\left(\frac{2\pi \cdot x}{\text{period}}\right), \quad \cos\!\left(\frac{2\pi \cdot x}{\text{period}}\right)$$

This maps each value to a unit circle, where nearby points in the cycle are nearby in feature space.

### Missing Data

**Missing completely at random (MCAR):** missingness is unrelated to any variable. Mean/median imputation is unbiased but underestimates variance.

**Missing at random (MAR):** missingness depends on observed variables. Model-based imputation (iterative imputation, MICE) is appropriate.

**Missing not at random (MNAR):** missingness depends on the missing value itself (e.g., people with very high income skip the income question). No imputation is unbiased without modeling the missingness mechanism. Add a binary "was this field missing?" indicator as a separate feature.

**Imputation strategies:**
- Mean/median imputation: simple, fast, underestimates variance, distorts correlations
- Mode imputation for categoricals
- K-NN imputation: use nearest neighbors to estimate missing value; computationally expensive
- Iterative imputation (MICE): iterate over features, treating each missing column as a regression target
- Tree-based models can handle missing values natively (XGBoost, LightGBM)

**Critical:** fit imputer on training data only; apply the same transform to validation/test.

### Feature Selection

**Filter methods:** score each feature independently of the model. Fast; ignores feature interactions.
- Correlation with target (Pearson for continuous, point-biserial for binary)
- Mutual information with target
- Chi-squared test for categorical features
- ANOVA F-statistic for continuous features vs. categorical target

**Wrapper methods:** use model performance to evaluate feature subsets. Expensive; model-specific.
- Forward selection: start empty, add the feature that most improves performance each step
- Backward elimination: start full, remove the least important feature each step
- Recursive Feature Elimination (RFE): fit model, remove lowest-importance features, repeat

**Embedded methods:** feature selection happens during model training.
- Lasso: zero weights = feature removal
- Tree feature importance (impurity-based or permutation-based)
- L1 regularization in neural networks

**Permutation importance:** permute a feature's values and measure performance drop. Model-agnostic; captures actual contribution accounting for interactions. Computationally expensive.

---

## Core ML Concepts Map

```
Problem Definition
    ↓
Data (collection, cleaning, features)
    ↓
Model (hypothesis class)
    ↓
Loss Function (what to minimize)
    ↓
Optimizer (how to minimize it)
    ↓
Evaluation (does it generalize?)
    ↓
Deployment (does it work in the real world?)
```

**Every ML algorithm is an instantiation of this loop.**

---

## Key Terminology Quick Reference

| Term | One-line definition |
|------|-------------------|
| Hypothesis class | Set of all functions the model can represent |
| Generalization | Performance on unseen data from same distribution |
| Distribution shift | Test data comes from different distribution than training |
| Inductive bias | Assumptions baked into model architecture |
| Expressivity | How complex a function the model can approximate |
| Sample complexity | How many examples needed to learn a concept |
| No free lunch | No algorithm works best on all problems |
| ERM | Empirical Risk Minimization — minimize average training loss |
| True risk | Expected loss over the full data distribution |
| VC dimension | Largest set of points a hypothesis class can shatter |
| Consistent estimator | Converges to true value as n → ∞ |
| Sufficient statistic | Captures all information about θ in the data |
| Fisher information | How much a single observation tells you about θ |
| Flat minimum | Minimum with small Hessian eigenvalues; generalizes better |
| Double descent | Performance improves again beyond the interpolation threshold with larger models |

---

## "Explain to a 5-year-old" Templates

**What is a neural network?**
→ Layers of linear transformations alternated with non-linearities. Each layer learns to detect patterns that the next layer combines. The whole thing is trained end-to-end by gradient descent.

**What is gradient descent?**
→ Compute how wrong you are (loss). Compute which direction makes it less wrong (gradient). Take a small step in that direction. Repeat millions of times.

**What is overfitting?**
→ The model memorized the training set including its noise, rather than learning the underlying pattern. It performs well on training data, poorly on new data.

**What is the curse of dimensionality?**
→ In high dimensions, all points are far apart. Distance metrics lose meaning. You need exponentially more data to cover the space. Feature selection and dimensionality reduction fight this.

**What is MLE?**
→ Choose model parameters that make the data you actually observed as probable as possible. Minimizing MSE for regression is MLE under a Gaussian noise assumption. Minimizing cross-entropy for classification is MLE under a Bernoulli/categorical distribution assumption.

**What is MAP estimation?**
→ MLE plus a prior belief about what reasonable parameter values look like. Adding a Gaussian prior is exactly equivalent to L2 regularization. Adding a Laplace prior is exactly equivalent to L1 regularization.

**What is generalization?**
→ A model generalizes if it performs well on data from the same distribution as training, but which it has never seen. VC theory says generalization improves with more data and simpler models. Neural networks violate classical bounds but still generalize — via flat minima, implicit regularization from SGD, and overparameterization.
