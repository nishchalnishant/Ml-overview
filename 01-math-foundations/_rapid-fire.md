---
module: Interview Prep
topic: Ml
subtopic: Statistics Probability
status: unread
tags: [interviewprep, ml, ml-statistics-probability]
---
# Statistics and Probability — Interview Reference

---

## The Testing Pattern

Interviewers use statistics questions to separate two populations: candidates who memorized definitions, and candidates who understand what the tools are for. The tell is whether you can answer "what goes wrong if you don't use this?" for every concept. A p-value question answered as "the probability the null hypothesis is true" is an instant signal you've memorized the wrong thing.

Three layers interviewers probe:
1. **Descriptive** — can you summarize data and understand what statistics capture vs miss?
2. **Inferential** — do you know what a p-value, CI, and hypothesis test actually measure, including their common misinterpretations?
3. **Probabilistic** — can you connect Bayes' theorem and MAP estimation to the regularization you use every day?

---

## 1. Core Distributions

### The problem they solve

You can't reason about random events, error bars, or model uncertainty without naming the structure of randomness. Every distribution encodes assumptions about how the world generates data. Picking the wrong distribution model is a systematic error, not noise.

| Distribution | PMF / PDF | Mean | Variance | Use in ML |
| :--- | :--- | :--- | :--- | :--- |
| Bernoulli$(p)$ | $P(X=1)=p$ | $p$ | $p(1-p)$ | Binary classification output — each prediction is one Bernoulli trial |
| Binomial$(n,p)$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | Count of successes in $n$ trials — e.g., correct predictions in a batch |
| Poisson$(\lambda)$ | $e^{-\lambda}\lambda^k/k!$ | $\lambda$ | $\lambda$ | Rare event counts (clicks per hour, errors per day) — key property: mean = variance |
| Exponential$(\lambda)$ | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | Time between Poisson events — memoryless property |
| Normal$(\mu,\sigma^2)$ | $\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | Residuals, weight initialization — arises from CLT |

**Gaussian properties that matter in practice:**
- Sum of independent Gaussians: $X+Y \sim \mathcal{N}(\mu_X+\mu_Y, \sigma_X^2+\sigma_Y^2)$ — why adding noise sources adds variances
- 68-95-99.7 rule: $P(\mu \pm \sigma) \approx 68\%$, $P(\mu \pm 2\sigma) \approx 95\%$, $P(\mu \pm 3\sigma) \approx 99.7\%$ — the basis for anomaly detection thresholds

**What goes wrong without knowing which distribution applies:** treating count data (Poisson) with a Normal model assumes negative counts are possible. Fitting a Gaussian to heavy-tailed data massively underestimates extreme event probability — the 2008 financial crisis was partly caused by risk models using Gaussian assumptions on fat-tailed distributions.

---

## 2. Descriptive Statistics

### The problem they solve

Raw data is uninterpretable at scale. Descriptive statistics are lossy compression — they discard information to expose structure. The trap is forgetting what they discard.

$$\text{Mean: } \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$$

$$\text{Variance: } s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2$$

Why $n-1$ (Bessel's correction): the sample mean $\bar{x}$ is estimated from the same data, introducing one constraint. Using $n$ underestimates variance. Dividing by $n-1$ gives an unbiased estimator.

$$\text{Covariance: } \text{Cov}(X,Y) = \frac{1}{n-1}\sum(x_i - \bar{x})(y_i - \bar{y})$$

$$\text{Pearson correlation: } r = \frac{\text{Cov}(X,Y)}{s_X s_Y} \in [-1, 1]$$

Pearson $r$ measures **linear** co-movement only. Two variables can have $r = 0$ and a strong nonlinear relationship (e.g., $Y = X^2$ centered at zero).

**Common trap — correlation vs causation:** ML models exploit correlation to make predictions. That is not the same as understanding why the relationship exists. A model predicting ice cream sales from drowning rates would have high accuracy. Both are caused by summer heat. Deploy it for policy decisions and you get nonsense.

---

## 3. Bayes' Theorem

### The problem it solves

You observe evidence. You want to update your belief about a hypothesis. Without a formal mechanism, humans are bad at this — they anchor on priors (base rate neglect) or ignore them entirely (prosecutor's fallacy). Bayes' theorem is the mathematically correct rule for belief updating.

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

In ML language:
$$\underbrace{P(\theta \mid D)}_{\text{posterior}} = \frac{\underbrace{P(D \mid \theta)}_{\text{likelihood}} \cdot \underbrace{P(\theta)}_{\text{prior}}}{\underbrace{P(D)}_{\text{normalizing constant}}}$$

**The failure mode this prevents:** base rate neglect. A disease test is 99% accurate. The disease affects 0.1% of the population. You test positive. What's the probability you have the disease?

Most people say ~99%. The correct answer: $P(\text{disease} \mid +) = \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.01 \times 0.999} \approx 9\%$. The base rate of 0.1% dominates.

**Naive Bayes classifier:** assumes features are conditionally independent given class:
$$P(y \mid x_1, \ldots, x_n) \propto P(y) \prod_{i=1}^n P(x_i \mid y)$$

This assumption is almost never true in practice, yet Naive Bayes performs well on text classification. Why? Because even with wrong probability estimates, the decision boundary (which class is more probable) is often correct. The independence assumption biases the probability scores — but classification only needs the correct argmax.

---

## 4. MLE vs MAP

### The problem MAP solves

MLE finds parameters that maximize the probability of observing the training data. The failure mode: with limited data, MLE overfits to noise. A coin flipped 3 times, landing heads all three times — MLE gives $\hat{p} = 1.0$. That's clearly wrong.

MAP adds a prior — a belief about likely parameter values before seeing data. The prior acts as a regularizer.

**MLE:**
$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(D \mid \theta) = \arg\max_\theta \sum_i \log P(x_i \mid \theta)$$

**MAP:**
$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta [\log P(D \mid \theta) + \log P(\theta)]$$

The second term is the log prior — it penalizes unlikely parameters. This is exactly what regularization does:

| Prior | Log prior | Regularization term |
| :--- | :--- | :--- |
| Gaussian: $P(\theta) \propto e^{-\lambda\|\theta\|^2}$ | $-\lambda\|\theta\|^2$ | L2 / Ridge |
| Laplace: $P(\theta) \propto e^{-\lambda|\theta|}$ | $-\lambda|\theta|$ | L1 / Lasso |

**The insight:** when you add L2 regularization, you are not doing something ad hoc — you are performing MAP estimation with a Gaussian prior on weights. The regularization strength $\lambda$ is your prior variance inverted. L1's Laplace prior has heavier tails at zero, which is why it produces exact zeros. L2's Gaussian prior pulls all weights toward zero but smoothly — never exactly.

---

## 5. Hypothesis Testing

### The problem it solves

You observe a difference between two groups (model A vs model B, treatment vs control). How do you know the difference is real and not a fluke of sampling? Hypothesis testing provides a framework for making that judgment.

**The framework:** assume the null hypothesis (no difference) is true. How surprising is what you observed?

$$p\text{-value} = P(\text{observed statistic or more extreme} \mid H_0 \text{ is true})$$

**What a p-value is NOT** — this is what interviewers specifically test:
- Not $P(H_0 \text{ is true})$ — that requires a prior (Bayesian reasoning)
- Not "the probability your result is a fluke"
- Not a measure of effect size — a tiny, meaningless difference can be highly significant with enough data
- Not stable — two experiments with $p = 0.049$ and $p = 0.051$ are effectively identical; the 0.05 threshold is arbitrary

**Type I / Type II errors — the cost tradeoff:**

| | Predict $H_0$ | Predict $H_1$ |
| :--- | :--- | :--- |
| **True $H_0$** | Correct | **Type I error** $(\alpha)$ — false positive |
| **True $H_1$** | **Type II error** $(\beta)$ — false negative | Correct |

- $\alpha$ = significance level = Type I error rate. You set this; typical value 0.05.
- Power = $1 - \beta$ = probability of detecting a true effect. Set target (commonly 0.8 or 0.9); determines required sample size.

The tradeoff: reducing $\alpha$ (being more conservative about false positives) increases $\beta$ (more false negatives) for fixed sample size. The only way to reduce both is more data.

**Z-test for proportions** (the A/B test workhorse):
$$z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(\frac{1}{n_1}+\frac{1}{n_2})}}, \quad \hat{p} = \frac{x_1+x_2}{n_1+n_2}$$

---

## 6. Confidence Intervals

### The problem they solve

A point estimate (the sample mean) is almost certainly wrong — you know the true mean is near $\bar{x}$ but not exactly. A confidence interval quantifies the uncertainty in that estimate.

$$\text{CI} = \bar{x} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}$$

For 95% CI: $z_{0.025} = 1.96$. The $\sqrt{n}$ in the denominator is the CLT result: standard error shrinks as you collect more data.

**The correct interpretation** (and this is a common interview trap):

- Correct: "If we repeated this experiment many times and computed a CI each time, 95% of those intervals would contain the true parameter."
- Incorrect: "There is a 95% probability the true value is in this specific interval."

The true parameter is fixed — it's not random. The interval is what's random (it varies across experiments). Once you've computed a specific interval, the true value either is or isn't in it — probability 0 or 1. The 95% refers to the long-run frequency of the procedure, not any individual interval.

**Why the distinction matters in ML:** when you report "model accuracy = 87% ± 2%," you're implicitly stating a CI. The correct claim is about the reliability of the measurement process, not about a probability on the true accuracy.

---

## 7. Central Limit Theorem and Law of Large Numbers

### The problem they solve

How can we reason about population parameters from samples? These two theorems are the theoretical foundation that makes statistical inference possible.

**Law of Large Numbers (LLN):** with enough data, sample averages converge to the true mean:
$$\bar{X}_n \xrightarrow{p} \mu \quad \text{as } n \to \infty$$

This is why: more labeled data = more reliable performance metrics; more training data = closer to the true data distribution.

**Central Limit Theorem (CLT):** the sampling distribution of the mean becomes Gaussian regardless of the original distribution:
$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1) \quad \text{as } n \to \infty$$

**Why CLT matters for ML specifically:**
- It justifies treating test set accuracy (an average over test examples) as approximately Gaussian — enabling confidence intervals and hypothesis tests
- It explains why neural network weights at initialization, if they're a sum of many random contributions, tend to be Gaussian
- It's why batch normalization works: the mean and variance of a minibatch approximate population statistics better as batch size grows

**What goes wrong without it:** if CLT didn't hold, you couldn't construct confidence intervals for model performance without knowing the exact distribution of your errors — which you never do.

---

## 8. Probability Rules

### Why you need to know them for ML interviews

These aren't trivia — they're the algebra you need to derive Naive Bayes, understand independence assumptions, and decompose joint probabilities.

**Addition rule:**
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Multiplication rule:**
$$P(A \cap B) = P(A \mid B) \cdot P(B)$$

**Independence:** $A \perp B$ iff $P(A \cap B) = P(A) \cdot P(B)$ — knowing one tells you nothing about the other

**Total probability:**
$$P(B) = \sum_i P(B \mid A_i) P(A_i)$$

This is the denominator in Bayes' theorem — often omitted because it's just a normalizing constant, but it's necessary when computing absolute probabilities.

**Conditional independence vs marginal independence:** two variables can be marginally independent but conditionally dependent (or vice versa). Example: disease A and disease B may be marginally independent, but if you know the patient is hospitalized (a collider), learning about A gives information about B. This is Berkson's bias — conditioning on a collider creates spurious correlations.

---

## 9. Information Theory

### The problem it solves

How do you measure uncertainty? How do you measure how wrong a probability distribution is? Gradient descent needs a loss function — for classification, cross-entropy is that loss. Understanding why requires information theory.

**Entropy** — the average surprise in a distribution:
$$H(X) = -\sum_x P(x) \log P(x)$$

Maximum entropy = uniform distribution (maximum uncertainty). Zero entropy = deterministic (knowing the distribution tells you the outcome exactly). Units: bits (log base 2) or nats (natural log).

**Why entropy is the right measure:** if an event has probability $p$, its information content is $-\log p$ (rare events are more informative). Entropy is the expectation of information content over the distribution.

**Cross-entropy** — how well distribution $q$ approximates true distribution $p$:
$$H(p, q) = -\sum_x p(x) \log q(x)$$

When $p$ is one-hot (true labels) and $q$ is your model's softmax output, this is exactly the classification loss:
$$\mathcal{L} = -\log q(y_\text{true})$$

**KL divergence** — how much information is lost using $q$ to approximate $p$:
$$D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p,q) - H(p)$$

Key properties:
- $D_{\text{KL}} \geq 0$, equals 0 iff $p = q$ exactly
- Not symmetric: $D_{\text{KL}}(p\|q) \neq D_{\text{KL}}(q\|p)$ — the "forward" and "reverse" KL make different approximation choices
- Minimizing cross-entropy loss = minimizing KL divergence to true labels + constant (since $H(p)$ is fixed)

**Where KL appears in ML:**
- VAE loss: KL between the approximate posterior and the prior regularizes the latent space
- DPO: KL penalty between the policy and the reference model controls how far fine-tuning can deviate
- Knowledge distillation: KL between teacher and student output distributions

---

## 10. A/B Testing for ML

### The failure modes this design prevents

Deploying a new model without a proper experiment gives you: correlation between deployment and metrics, not causation. Seasonality, other product changes, and novelty effects all move your metrics. A/B testing isolates the causal effect of your model.

**Sample size calculation** (two-proportion z-test):
$$n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\bar{p}(1-\bar{p})}{\delta^2}$$

Where $\delta$ is the minimum detectable effect (the smallest difference worth detecting), $\bar{p}$ is the baseline rate. This must be computed before the experiment, not after.

```python
from scipy import stats

def ab_test(control_conversions, control_n, treatment_conversions, treatment_n, alpha=0.05):
    p_control = control_conversions / control_n
    p_treatment = treatment_conversions / treatment_n
    p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)

    se = (p_pool * (1 - p_pool) * (1/control_n + 1/treatment_n)) ** 0.5
    z_stat = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-tailed

    return {
        "lift": (p_treatment - p_control) / p_control,
        "z_stat": z_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
    }
```

**Common pitfalls — the ones interviewers specifically ask about:**

**Peeking (the most common mistake):** checking the p-value daily and stopping when $p < 0.05$ inflates the Type I error rate from 5% to ~30% at the extreme. You're doing many hypothesis tests, one per day, and taking the most favorable one. Fix: pre-commit to the sample size, don't look until it's reached. Or use sequential testing methods (SPRT, always-valid p-values) designed for continuous monitoring.

**Multiple comparisons:** testing 20 metrics at $\alpha=0.05$ expects 1 false positive by chance. Use Bonferroni correction ($\alpha / m$) or Benjamini-Hochberg FDR control. Pick your primary metric in advance; treat the rest as secondary.

**Simpson's paradox:** the aggregated treatment effect is positive, but negative in every subgroup. Arises when subgroup sizes are imbalanced between treatment and control. Always segment by major stratification variables before concluding.

**Novelty effect:** users click on new things just because they're new, not because they're better. A/B test that runs only a week may show inflated CTR for the new model. Run for at least 2 weeks and monitor if the effect decays.

---

## Common Traps

**"p < 0.05 means the null hypothesis is false."** No — it means the data would be unusual if the null were true. The null could still be true and you got unlucky (5% of the time you will).

**"A large sample proves significance, therefore the effect matters."** With $n = 10M$, a 0.001% lift is statistically significant but practically irrelevant. Statistical significance is not effect size.

**"Correlation of 0 means independence."** Only for Gaussian variables. In general, correlation measures only linear dependence. Mutual information is a better measure of general dependence.

**"MAP is always better than MLE."** With large data, the likelihood overwhelms the prior and MAP ≈ MLE. MAP's advantage is with small data — when the prior matters. Using a strong prior on a large dataset can introduce bias.
