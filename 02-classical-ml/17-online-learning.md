---
module: Classical Ml
topic: Online Learning
subtopic: ""
status: unread
tags: [classicalml, ml, online-learning]
---
# Online Learning

---

## TL;DR

Online learning processes data one example at a time, updating the model after each observation. This is not just an engineering choice — it is the right framework whenever data arrives as a stream, memory is bounded, or the underlying distribution shifts over time. The theoretical backbone is the **regret framework**: can an algorithm, over T rounds, do nearly as well as the best fixed strategy in hindsight? Algorithms like OGD and FTRL provide O(√T) regret bounds. Concept drift, Hoeffding trees, bandits, and reservoir sampling round out the core toolkit.

---

## Online vs Batch Learning

**The problem**: batch learning assumes a fixed dataset — load it, train on it, deploy the model. This fails when data volumes exceed memory, data arrives continuously, or the statistical properties of the data evolve over time. Retraining from scratch every time is expensive and slow.

**The core insight**: process examples one at a time. Maintain a model state. Update it incrementally after each observation. Discard the example when done (or store it in a fixed-size buffer). The model is always current.

| Dimension | Batch Learning | Online Learning |
|---|---|---|
| Data assumption | Fixed, finite dataset | Stream, potentially infinite |
| Memory | Entire dataset in memory | O(1) or bounded buffer |
| Update frequency | One pass (or many epochs) over full data | One update per example |
| Non-stationarity | Requires periodic retraining | Naturally adapts |
| Convergence guarantee | Strong (given fixed distribution) | Regret-based (competitive with best fixed model) |
| Typical use case | Offline model training | Ad click prediction, recommendation, fraud |

**When online learning is necessary**:
- **Streaming data**: financial ticks, sensor feeds, log streams — data cannot be stored or reprocessed
- **Memory constraints**: dataset is too large to fit in RAM; each example is processed and discarded
- **Non-stationarity**: the generating distribution drifts — yesterday's model is stale; the model must adapt continuously
- **Low-latency personalization**: the model must incorporate user feedback immediately (next-page recommendation)

---

## Regret Framework

**The problem**: how do we define "learning well" when there is no fixed distribution to evaluate against? In online learning, the data can be adversarial — chosen by an opponent who sees the algorithm.

**The core insight**: forget absolute performance. Measure *relative* performance — how much worse does the online algorithm do compared to the best fixed strategy chosen in hindsight?

### Cumulative Regret

Over T rounds, the algorithm produces predictions ŷ₁, …, ŷ_T and suffers losses ℓ₁, …, ℓ_T. The **cumulative regret** against a comparator class H is:

```
R_T = Σ_{t=1}^{T} ℓ_t(ŷ_t) - min_{h ∈ H} Σ_{t=1}^{T} ℓ_t(h(x_t))
```

The algorithm is **no-regret** (or **Hannan consistent**) if R_T / T → 0 as T → ∞. This means the algorithm converges to performing as well as the best fixed strategy in hindsight, even on adversarial sequences.

### Adversarial vs Stochastic Settings

| Setting | Assumption | Typical Regret Bound | Example |
|---|---|---|---|
| **Stochastic** | Examples drawn i.i.d. from fixed distribution | O(√T) or better | Online SGD on stationary data |
| **Adversarial** | Examples chosen by an adaptive adversary | O(√T) in general | Experts problem, game-theoretic settings |
| **Oblivious adversary** | Data sequence fixed before game starts | O(√T) | Most theoretical bounds |
| **Adaptive adversary** | Data depends on algorithm's past actions | Harder; requires randomization | Bandit feedback |

A regret bound of O(√T) means the *per-round* regret R_T / T = O(1/√T) vanishes — the algorithm is no-regret. A bound of O(log T) is stronger (faster convergence, occurs with strongly convex losses).

---

## Online Gradient Descent (OGD)

**The problem**: standard gradient descent processes the full dataset to compute a gradient. We need an update rule that uses only the current example.

**The core insight**: take a gradient step on the loss of the current example only. The stochasticity of this gradient averages out over time.

### Update Rule

```
w_{t+1} = Π_W(w_t - η_t ∇ℓ_t(w_t))
```

where:
- `w_t` is the weight vector at round t
- `η_t` is the learning rate (step size), often `η_t = η / √t`
- `∇ℓ_t(w_t)` is the gradient of the loss at time t evaluated at `w_t`
- `Π_W` is projection onto the feasible set W (ensures w stays in bounds)

### Regret Bound

For convex losses and bounded gradients (||∇ℓ_t|| ≤ G) and domain diameter D:

```
R_T ≤ D·G·√T       (convex losses, η_t = D/(G√t))
R_T ≤ O(log T)     (strongly convex losses)
```

The O(√T) bound is tight — no algorithm can achieve better in the adversarial setting with only convex loss assumptions.

```python
# Online gradient descent — one update per example
import numpy as np

class OnlineGD:
    def __init__(self, d, eta=0.1):
        self.w = np.zeros(d)
        self.t = 0
        self.eta = eta

    def predict(self, x):
        return self.w @ x

    def update(self, x, y):
        self.t += 1
        eta_t = self.eta / np.sqrt(self.t)
        grad = (self.w @ x - y) * x          # squared loss gradient
        self.w -= eta_t * grad
```

### OGD vs Batch GD

| Property | Batch GD | Online GD |
|---|---|---|
| Gradient | Full dataset average | Single example (noisy) |
| Per-update cost | O(n·d) | O(d) |
| Convergence | Deterministic, smooth | Probabilistic, regret-bounded |
| Non-stationary data | Fails without retraining | Naturally adapts |
| Memory | O(n) | O(d) |

---

## FTRL (Follow The Regularized Leader)

**The problem**: OGD works but does not take full advantage of the history of gradients. On sparse data (most features zero on any given example), OGD wastes updates — learning rates decay for all features even when most were not observed.

**The core insight**: at each round, choose the action that minimizes the cumulative loss on all *past* rounds plus a regularization term. FTRL has memory — it uses the entire gradient history, which gives better per-feature adaptivity.

### Algorithm

At round t, FTRL solves:

```
w_{t+1} = argmin_w [ Σ_{s=1}^{t} ∇ℓ_s · w  +  R(w) ]
```

where R(w) is a regularizer (e.g., L2: λ||w||²). This is equivalent to a closed-form update that adapts per-feature learning rates based on observed gradient history.

**FTRL-Proximal** (used in practice for L1 sparsity):

```
w_{t+1,i} = 0                             if |z_{t,i}| ≤ λ₁
           = -(z_{t,i} - sign(z_{t,i})·λ₁) / (λ₂ + (β + √n_{t,i})/η)   otherwise
```

where `z_t` accumulates past gradients and `n_t` accumulates squared gradients (AdaGrad-style).

### Why FTRL Beats OGD for Sparse Features

- **OGD**: learning rate for feature i decays as 1/√t regardless of how often feature i appeared. Rare features get unfairly penalized.
- **FTRL + AdaGrad**: learning rate for feature i decays as `1/√(number of times feature i was non-zero)`. Rare features stay at high learning rates longer — they learn more when they do appear.
- **L1 regularization + FTRL**: naturally produces sparse models. OGD with L1 does not produce exact zeros. FTRL-proximal does — critical for serving models with millions of features.

**Google's Logistic Regression for Ad Click Prediction** (McMahan et al., 2013): deployed FTRL-Proximal at billion-feature scale. Key findings: per-feature learning rates critical for sparse data; L1 sparsity reduced model size by 50% with no accuracy loss; rolling window of training data handled non-stationarity.

```python
# Conceptual FTRL-Proximal for one feature
def ftrl_proximal_update(z_i, n_i, g_i, eta, beta, lambda1, lambda2):
    n_i += g_i ** 2
    sigma_i = (np.sqrt(n_i) - np.sqrt(n_i - g_i**2)) / eta
    z_i += g_i - sigma_i * w_i  # z accumulates adjusted gradients

    if abs(z_i) <= lambda1:
        w_i = 0.0
    else:
        w_i = -(z_i - np.sign(z_i) * lambda1) / (lambda2 + (beta + np.sqrt(n_i)) / eta)
    return w_i, z_i, n_i
```

| Algorithm | Regret (convex) | Sparse model | Per-feature LR | Practical use |
|---|---|---|---|---|
| OGD | O(√T) | No | No | General baseline |
| AdaGrad | O(√T) | No | Yes | NLP, sparse gradients |
| FTRL | O(√T) | No | Depends | Theory |
| FTRL-Proximal | O(√T) | Yes (L1) | Yes | Ad prediction at scale |

---

## Online-to-Batch Conversion

**The problem**: online learning produces a sequence of models w₁, w₂, …, w_T. Which one do you deploy? The last one may have overfit to recent examples; any single model may be far from optimal.

**The core insight**: average the online models. The averaged model has batch generalization guarantees derived from the online regret bound.

### The Conversion

Given an online algorithm with cumulative regret R_T, the **average model** w̄_T = (1/T) Σ_t w_t satisfies:

```
E[ℓ(w̄_T)] - min_w ℓ(w) ≤ R_T / T
```

If R_T = O(√T), then the excess batch risk is O(1/√T) — matching the standard batch learning rate. This means:

- Online algorithms are not just streaming tools — they are valid optimization algorithms for batch problems
- SGD with averaging (Polyak-Ruppert averaging) is exactly this: online GD followed by model averaging
- In practice, **exponential moving average** (EMA) of weights is used instead of uniform average, giving more weight to recent models

```python
# Polyak-Ruppert averaging
ema_w = np.zeros(d)
alpha = 0.99  # EMA decay

for x, y in stream:
    online_model.update(x, y)
    ema_w = alpha * ema_w + (1 - alpha) * online_model.w

# ema_w is the deployment model
```

---

## Concept Drift

**The problem**: a deployed model was trained on historical data. The real world changes. The joint distribution P(X, Y) shifts over time. The model becomes stale — not because it was wrong initially, but because the world changed.

**The core insight**: monitor for distribution shift, detect it, and adapt. The type of drift determines the adaptation strategy.

### Types of Concept Drift

| Type | Description | Example | Adaptation |
|---|---|---|---|
| **Sudden** | Abrupt shift at a specific time t | System failure, policy change | Reset model, train fresh window |
| **Gradual** | Slow drift, old and new distributions overlap over time | Seasonal preference shift | Decay old examples (sliding window, higher LR) |
| **Incremental** | Steady monotonic change | User aging, language evolution | Continuous adaptation |
| **Recurring** | Old concepts reappear cyclically | Seasonal patterns (Black Friday) | Store model snapshots per season |
| **Blip** | Temporary anomaly that reverts | One-off event (news spike) | Ignore/smooth if short-lived |

### ADWIN (Adaptive Windowing)

ADWIN maintains a variable-size sliding window over a stream of values. It detects drift by testing whether the mean of any sub-window differs significantly from the rest.

**Key property**: ADWIN guarantees that if the distribution is stationary, no false alarm is raised with high probability. If drift occurs, it is detected in O(log n) time after the change point.

```
For window W: drop oldest elements while
∃ split of W into W₀, W₁ such that |μ̂(W₀) - μ̂(W₁)| ≥ ε_cut

where ε_cut = √( (1/(2m₀) + 1/(2m₁)) · ln(4n/δ) )
```

ADWIN is used as a drift detector inside streaming tree algorithms (Hoeffding Adaptive Trees).

### Page-Hinkley Test

Detects a persistent shift in the mean of a sequence. Accumulates the running sum of deviations from an estimated mean and triggers an alarm when this sum exceeds a threshold.

```
Update: m_t = m_{t-1} + (x_t - x̄_t - δ)
Alarm:  M_T - m_t > λ     where M_T = max_{t'≤T} m_{t'}
```

- `δ`: minimum magnitude of change to detect (sensitivity)
- `λ`: alarm threshold (false alarm control)

### Drift Adaptation Strategies

| Strategy | Mechanism | When to use |
|---|---|---|
| **Sliding window** | Train only on the last W examples | Gradual drift, sufficient data density |
| **Exponential weighting** | Weight recent examples more heavily | Gradual drift, smooth adaptation |
| **Drift detection + reset** | Detect drift, discard model, retrain | Sudden drift |
| **Ensemble of windows** | Maintain models for multiple windows, weight by recent accuracy | Recurring drift |
| **Always-online** | Continuously update, no explicit detection | When drift is expected to be slow and continuous |

---

## Hoeffding Trees (Very Fast Decision Trees)

**The problem**: batch decision trees (CART, C4.5) require multiple passes over the full dataset to find splits. This is impossible on infinite streams and prohibitively slow when data is too large for memory.

**The core insight**: you do not need to see all examples to be confident in a split. The **Hoeffding bound** gives a statistical guarantee: after seeing n examples, the true mean of a bounded random variable is within ε of the sample mean with probability 1 − δ:

```
ε = √( R² · ln(1/δ) / (2n) )
```

where R is the range of the variable.

### When to Split

A Hoeffding tree maintains sufficient statistics at each leaf (counts per class per feature). To decide whether to split on feature a vs the next-best feature b:

```
Split on a if:  G(a) - G(b) > ε_H(n, δ)

where ε_H = √( R² · ln(1/δ) / (2n) ) and G is information gain (or Gini impurity reduction)
```

If `G(a) - G(b) < ε_H`: need more data — wait.
If `G(a) - G(b) > ε_H`: the best split is identifiable with confidence 1 − δ — split now.
If `G(a) - G(b) → 0` and `ε_H < τ` (tie-breaking threshold): force a split.

### Memory Efficiency

- Only sufficient statistics are stored per leaf (class counts, feature histograms)
- Once a node is split, it transitions from leaf to internal node — no full dataset is retained
- In practice, memory is bounded by capping the number of leaves (least-recently-used eviction)

### Comparison to Batch CART

| Property | Batch CART | Hoeffding Tree (VFDT) |
|---|---|---|
| Data passes | Multiple (full dataset each) | One pass, each example seen once |
| Memory | O(n) (stores dataset) | O(leaves · features · classes) |
| Split criterion | Exact information gain | Statistically guaranteed same result with prob. 1 − δ |
| Non-stationary | Requires retraining | CVFDT/HAT variants adapt to drift |
| Speed | Slow on large data | Very fast (stream-native) |
| Accuracy | Optimal given enough data | Converges to CART accuracy asymptotically |

**Hoeffding Adaptive Tree (HAT)**: combines VFDT with ADWIN drift detection. When drift is detected at a node, it grows an alternative subtree and replaces the current subtree when the alternative achieves better accuracy.

```python
# scikit-multiflow / River implementation
from river import tree

model = tree.HoeffdingTreeClassifier(
    grace_period=200,        # minimum examples before considering a split
    delta=1e-7,              # confidence level for Hoeffding bound
    tau=0.05,                # tie-breaking threshold
    max_size=200,            # memory cap in MB
)

for x, y in stream:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
```

---

## Reservoir Sampling

**The problem**: you want to maintain a uniform random sample of size k from a stream of unknown length n. You cannot store the full stream. You need each element to have equal probability k/n of being in the sample.

### Algorithm R (Vitter, 1985)

```
Initialize reservoir = first k elements

For each new element x_t (t > k):
    j = random integer in [1, t]
    if j ≤ k:
        reservoir[j] = x_t    # replace a random element
```

**Invariant**: after processing t elements, each of the t elements has exactly probability k/t of being in the reservoir. This is maintained at every step without knowing n in advance.

```python
import random

def reservoir_sample(stream, k):
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir
```

### When to Use for Model Training

- **Streaming training data**: maintain a representative sample when you cannot store everything
- **Class-balanced sampling**: maintain per-class reservoirs to ensure balance for classifiers
- **Replay buffers**: in continual learning, reservoir sampling provides a principled way to retain old examples to avoid catastrophic forgetting
- **Caveat**: reservoir sampling gives a uniform sample over *time*, not over the *current distribution* — old (potentially drifted) examples are retained with equal probability as recent ones. When concept drift is present, a **recency-biased** reservoir (decaying acceptance probability) is preferred

---

## Multi-Armed Bandits

**The problem**: you must choose among K actions repeatedly. Each action has an unknown reward distribution. You want to maximize cumulative reward, but to learn reward distributions you must explore. Spending too much time exploring costs reward; exploiting too early gets stuck on suboptimal actions. This is the **exploration-exploitation tradeoff**.

Bandits are online learning problems with *partial feedback*: you only observe the reward of the chosen action, not the counterfactual rewards of unchosen actions.

### Epsilon-Greedy

```
With probability ε: choose a random arm (explore)
With probability 1 − ε: choose the arm with highest estimated mean reward (exploit)
```

Simple but wasteful — it explores uniformly, including arms already known to be bad. Does not achieve sub-linear regret with fixed ε. Use decaying ε_t = ε₀/√t for O(√T) regret.

### UCB1 (Upper Confidence Bound)

**The core insight**: be optimistic under uncertainty. Choose the arm whose *upper confidence bound* on mean reward is highest. Arms with few pulls have wide bounds → naturally explored.

```
UCB1 score for arm i at time t:  μ̂_i + √(2 ln t / n_i)

where μ̂_i = empirical mean reward, n_i = number of pulls of arm i
```

**Regret bound**: O(K log T) — logarithmic in T, optimal up to constants.

```python
import numpy as np

class UCB1:
    def __init__(self, K):
        self.K = K
        self.counts = np.zeros(K)
        self.means = np.zeros(K)
        self.t = 0

    def select(self):
        self.t += 1
        if self.t <= self.K:
            return self.t - 1          # pull each arm once first
        ucb = self.means + np.sqrt(2 * np.log(self.t) / self.counts)
        return np.argmax(ucb)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.means[arm] += (reward - self.means[arm]) / self.counts[arm]
```

### Thompson Sampling

**The core insight**: maintain a Bayesian posterior over each arm's reward distribution. At each round, sample one value from each arm's posterior and pull the arm with the highest sample.

```
For Bernoulli rewards with Beta prior:
    arm i has prior Beta(α_i, β_i)
    At round t: sample θ_i ~ Beta(α_i, β_i) for each arm
    Pull arm i* = argmax θ_i
    Update: if reward=1: α_{i*} += 1; else: β_{i*} += 1
```

**Regret bound**: O(K log T) — matches UCB1 empirically, often better in practice.

| Algorithm | Regret | Exploration | Requires | Use case |
|---|---|---|---|---|
| Epsilon-greedy (fixed) | O(T) | Uniform | Nothing | Baseline only |
| Epsilon-greedy (decaying) | O(√T) | Uniform | Nothing | Simple deployments |
| UCB1 | O(K log T) | Optimism | Bounded rewards | General purpose |
| Thompson Sampling | O(K log T) | Posterior sampling | Prior specification | Strong empirical performance |
| EXP3 | O(√KT log K) | Adversarial | Nothing | Adversarial setting |

### Contextual Bandits (LinUCB)

**The problem**: the optimal arm depends on context (features of the current user/item). Standard bandits ignore context.

**The core insight**: model the expected reward as a linear function of context: E[r | x, a] = x^T θ_a. Maintain a ridge regression estimate of θ_a per arm, and use its confidence ellipsoid as the UCB.

```
LinUCB selects arm a* = argmax_a ( x^T θ̂_a + α √(x^T A_a⁻¹ x) )

where A_a = X_a^T X_a + I (design matrix with regularization)
      θ̂_a = A_a⁻¹ b_a          (ridge regression estimate)
      α: exploration parameter
```

**Connection to online learning**: contextual bandits are a special case of online learning with partial feedback (only one arm's reward is observed per round). The regret framework applies: LinUCB achieves O(d√T log T) regret where d is the context dimension.

---

## Production Considerations

### Partial Fit APIs

sklearn's `partial_fit` API enables incremental learning without storing the full dataset:

```python
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB

# All support partial_fit(X_batch, y_batch, classes=[0, 1])
model = SGDClassifier(loss='log_loss', learning_rate='optimal')

for batch in data_stream:
    X_batch, y_batch = batch
    model.partial_fit(X_batch, y_batch, classes=[0, 1])
```

**Gotchas**: sklearn's `StandardScaler` also has `partial_fit` — always normalize online. The `classes` argument must be passed on the first call for classifiers.

### River (formerly scikit-multiflow)

Purpose-built for online learning in Python. Fully incremental API: `learn_one(x, y)` / `predict_one(x)` for every model.

```python
from river import linear_model, optim, preprocessing, compose

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression(optimizer=optim.SGD(0.01))
)

for x, y in stream:
    y_pred = model.predict_proba_one(x)
    model.learn_one(x, y)
```

### Vowpal Wabbit (VW)

Industrial-strength online learning library. Implements FTRL, OGD, contextual bandits. Used internally at Microsoft for Bing and Azure. Key features:

- Hashing trick for high-dimensional sparse features (no feature dictionary needed)
- Native support for FTRL-Proximal, AdaGrad, Adam
- Contextual bandits with exploration (epsilon-greedy, bagging, cover)
- Sub-second latency per example at billion-feature scale

```bash
# VW input format: label |namespace feature:value feature:value
echo "1 |user age:25 country:US |item genre:action recency:3" | vw --loss_function logistic
```

### Feature Drift vs Label Drift

| Type | Definition | Detection | Impact |
|---|---|---|---|
| **Feature drift** (covariate shift) | P(X) changes, P(Y\|X) stable | Monitor input feature distributions (PSI, KL divergence) | Model predictions degrade; recalibration may suffice |
| **Label drift** (prior shift) | P(Y) changes, P(X\|Y) stable | Monitor label distribution in labeled batches | Re-weight classes; retrain if severe |
| **Concept drift** | P(Y\|X) changes | Monitor prediction error on labeled stream | Must retrain or adapt model |
| **Data quality drift** | Upstream feature pipeline changes | Monitor nulls, type errors, range violations | Fix pipeline; may not require model change |

### Model Versioning in Streaming Pipelines

- **Shadow deployment**: run new model in parallel, log predictions, compare against live model offline before traffic cutover
- **Champion-Challenger**: route a fraction of traffic (5–10%) to the new model; promote when metric improves
- **Rolling update**: gradually increase challenger traffic; auto-rollback if metrics degrade by threshold
- **Versioned feature stores**: online feature stores (Feast, Tecton) snapshot feature schemas per model version — prevents training-serving skew when features change

---

## Interview Questions

**Q1: What is regret in online learning, and why is O(√T) considered good?**

Regret is the cumulative loss of the algorithm minus the cumulative loss of the best fixed strategy in hindsight. O(√T) regret means per-round regret is O(1/√T) → 0, so the algorithm converges to optimal average performance. This is called "no-regret." O(√T) is tight for convex losses in the adversarial setting — no algorithm can do better without stronger assumptions (e.g., strong convexity gives O(log T)).

**Q2: Why does FTRL with L1 regularization produce sparse models, but OGD with L1 does not?**

OGD applies a subgradient step: the update is `w ← w - η·(∇ℓ + λ·sign(w))`. This pushes weights toward zero but rarely exactly reaches zero due to floating-point arithmetic and step size dynamics. FTRL-Proximal solves a closed-form proximal problem at each step, which analytically sets weights to exactly zero when the cumulative gradient signal is below the L1 threshold. This is critical for serving models with millions of features where sparsity reduces memory and compute.

**Q3: Explain the Hoeffding bound and how it enables streaming decision trees.**

The Hoeffding bound states: given n i.i.d. observations of a bounded random variable in [a, b], the sample mean deviates from the true mean by more than ε with probability at most exp(−2nε²/(b−a)²). In Hoeffding trees, this bounds how many examples are needed to confidently identify the best split feature. When the information gain difference between the top two features exceeds ε_H(n, δ), we can split with confidence 1 − δ — without ever storing the raw data. This makes the tree provably converge to the same splits as a batch CART tree while requiring only one data pass.

**Q4: You're building a real-time fraud detection system. New fraud patterns emerge frequently. How do you handle concept drift?**

Multi-layered approach: (1) **Detection**: run ADWIN or Page-Hinkley on the model's error rate on labeled recent transactions. Alert when drift is detected. (2) **Adaptation**: use a sliding window model trained on the last W days of labeled data — fraud patterns have short half-lives, so staleness is the primary risk. (3) **Ensemble**: maintain a fast-adapting online model (updated per transaction) and a slow batch model (retrained weekly); blend predictions, weighting the online model higher after detected drift. (4) **Labels**: fraud labels arrive with delay — design the pipeline to handle delayed feedback by buffering, then triggering a model update when the label arrives.

**Q5: What is the difference between a contextual bandit and a supervised learning problem?**

In supervised learning, you observe the label (true outcome) for every input. In contextual bandits, you only observe the reward for the action you took — not what the reward would have been for unchosen actions (counterfactual). This is the core challenge. Supervised learning is a special case of contextual bandits where all actions' outcomes are observed simultaneously. The bandit problem requires balancing exploration (trying actions to learn their rewards) with exploitation (using current knowledge to maximize reward). Offline evaluation is also harder — logged bandit data has policy bias (actions are non-uniformly distributed), requiring off-policy estimators like IPS (inverse propensity scoring).

**Q6: When would you choose Thompson Sampling over UCB1?**

Thompson Sampling is generally preferred in practice because: (1) it naturally handles non-stationary rewards by using sliding-window posteriors; (2) it performs better empirically in small-sample regimes despite the same asymptotic regret bound; (3) it extends naturally to complex reward models (Bayesian neural networks, Gaussian processes) where confidence intervals are hard to compute but sampling is easy. UCB1 is preferred when you need deterministic behavior (reproducibility), when computing posteriors is expensive, or in adversarial settings where Bayesian assumptions do not hold.

**Q7: Explain reservoir sampling. Why might it be inappropriate when concept drift is present?**

Reservoir sampling maintains a uniform random sample of size k from a stream, where each element has probability k/t of being retained. It gives a statistically uniform sample over all examples seen so far. When concept drift is present, this is problematic: old examples (from a different distribution) are retained with the same probability as recent ones. A model trained on this reservoir learns a mixture of old and new distributions, degrading performance on the current distribution. Alternatives: (1) **sliding window**: only retain the last W examples — recency implies relevance; (2) **time-decayed reservoir**: replace elements with probability proportional to recency; (3) **ADWIN-based**: detect drift and reset the reservoir.

## Flashcards

**Streaming data: financial ticks, sensor feeds, log streams?** #flashcard
data cannot be stored or reprocessed

**Memory constraints?** #flashcard
dataset is too large to fit in RAM; each example is processed and discarded

**Non-stationarity: the generating distribution drifts?** #flashcard
yesterday's model is stale; the model must adapt continuously

**Low-latency personalization?** #flashcard
the model must incorporate user feedback immediately (next-page recommendation)

**w_t is the weight vector at round t?** #flashcard
w_t is the weight vector at round t

**η_t is the learning rate (step size), often η_t = η / √t?** #flashcard
η_t is the learning rate (step size), often η_t = η / √t

**∇ℓ_t(w_t) is the gradient of the loss at time t evaluated at w_t?** #flashcard
∇ℓ_t(w_t) is the gradient of the loss at time t evaluated at w_t

**Π_W is projection onto the feasible set W (ensures w stays in bounds)?** #flashcard
Π_W is projection onto the feasible set W (ensures w stays in bounds)

**OGD?** #flashcard
learning rate for feature i decays as 1/√t regardless of how often feature i appeared. Rare features get unfairly penalized.

**FTRL + AdaGrad: learning rate for feature i decays as 1/√(number of times feature i was non-zero). Rare features stay at high learning rates longer?** #flashcard
they learn more when they do appear.

**L1 regularization + FTRL: naturally produces sparse models. OGD with L1 does not produce exact zeros. FTRL-proximal does?** #flashcard
critical for serving models with millions of features.

**Online algorithms are not just streaming tools?** #flashcard
they are valid optimization algorithms for batch problems

**SGD with averaging (Polyak-Ruppert averaging) is exactly this?** #flashcard
online GD followed by model averaging

**In practice, exponential moving average (EMA) of weights is used instead of uniform average, giving more weight to recent models?** #flashcard
In practice, exponential moving average (EMA) of weights is used instead of uniform average, giving more weight to recent models

**δ?** #flashcard
minimum magnitude of change to detect (sensitivity)

**λ?** #flashcard
alarm threshold (false alarm control)

**Only sufficient statistics are stored per leaf (class counts, feature histograms)?** #flashcard
Only sufficient statistics are stored per leaf (class counts, feature histograms)

**Once a node is split, it transitions from leaf to internal node?** #flashcard
no full dataset is retained

**In practice, memory is bounded by capping the number of leaves (least-recently-used eviction)?** #flashcard
In practice, memory is bounded by capping the number of leaves (least-recently-used eviction)

**Streaming training data?** #flashcard
maintain a representative sample when you cannot store everything

**Class-balanced sampling?** #flashcard
maintain per-class reservoirs to ensure balance for classifiers

**Replay buffers?** #flashcard
in continual learning, reservoir sampling provides a principled way to retain old examples to avoid catastrophic forgetting

**Caveat: reservoir sampling gives a uniform sample over time, not over the current distribution?** #flashcard
old (potentially drifted) examples are retained with equal probability as recent ones. When concept drift is present, a recency-biased reservoir (decaying acceptance probability) is preferred

**Hashing trick for high-dimensional sparse features (no feature dictionary needed)?** #flashcard
Hashing trick for high-dimensional sparse features (no feature dictionary needed)

**Native support for FTRL-Proximal, AdaGrad, Adam?** #flashcard
Native support for FTRL-Proximal, AdaGrad, Adam

**Contextual bandits with exploration (epsilon-greedy, bagging, cover)?** #flashcard
Contextual bandits with exploration (epsilon-greedy, bagging, cover)

**Sub-second latency per example at billion-feature scale?** #flashcard
Sub-second latency per example at billion-feature scale

**Shadow deployment?** #flashcard
run new model in parallel, log predictions, compare against live model offline before traffic cutover

**Champion-Challenger?** #flashcard
route a fraction of traffic (5–10%) to the new model; promote when metric improves

**Rolling update?** #flashcard
gradually increase challenger traffic; auto-rollback if metrics degrade by threshold

**Versioned feature stores: online feature stores (Feast, Tecton) snapshot feature schemas per model version?** #flashcard
prevents training-serving skew when features change
