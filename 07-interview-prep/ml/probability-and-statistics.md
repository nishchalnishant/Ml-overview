# Probability and Statistics

This file is for the part of ML interviews where people suddenly become very interested in uncertainty, distributions, and whether you know the difference between "looks good" and "is actually supported by evidence."

Do not panic.

Think of stats as the quality-control layer for your intuition.

---

# 1. Why Probability and Statistics Matter in ML

Machine learning is full of uncertainty:

- noisy data
- imperfect labels
- sample bias
- random initialization
- fluctuating metrics

Statistics is what helps you reason through that uncertainty without bluffing.

If DevOps gives you observability for systems, stats gives you observability for data and decisions.

---

# 2. Normal Distribution

The normal distribution is the classic bell curve.

It is:

- continuous
- symmetric
- defined by mean and variance

Why it matters so much:

Because many aggregate phenomena behave approximately normally, especially when they are the combined result of many small random effects.

**Short interview answer**

The normal distribution is fundamental because many natural processes and many sampling-based estimates end up approximately normal, which makes it central to inference and modeling.

---

# 3. Common Distributions You Should Actually Remember

Know these cleanly — with their formulas.

## Bernoulli

One binary event.
$$P(X=k) = p^k (1-p)^{1-k}, \quad k \in \{0,1\}$$
$$\mathbb{E}[X] = p, \quad \text{Var}[X] = p(1-p)$$

## Binomial

Number of successes in $n$ Bernoulli trials.
$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$
$$\mathbb{E}[X] = np, \quad \text{Var}[X] = np(1-p)$$

Approximates Normal when $np \geq 5$ and $n(1-p) \geq 5$.

## Poisson

Count of events in a fixed interval with rate $\lambda$.
$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$
$$\mathbb{E}[X] = \lambda, \quad \text{Var}[X] = \lambda$$

## Exponential

Time between Poisson events with rate $\lambda$.
$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$
$$\mathbb{E}[X] = 1/\lambda, \quad \text{Var}[X] = 1/\lambda^2$$

Memoryless: $P(X > s+t \mid X > s) = P(X > t)$.

## Normal

Continuous bell curve parameterized by mean $\mu$ and variance $\sigma^2$.
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
$$\mathbb{E}[X] = \mu, \quad \text{Var}[X] = \sigma^2$$

68-95-99.7 rule: $P(|X - \mu| < k\sigma)$ = 68%, 95%, 99.7% for $k$ = 1, 2, 3.

**Memory trick:**
- Bernoulli = one shot
- Binomial = many shots
- Poisson = event count
- Exponential = waiting time

---

# 4. Poisson vs Binomial

This gets asked more often than people expect.

## Binomial

Use when:

- fixed number of trials
- each trial has success/failure

## Poisson

Use when:

- counting events in time or space
- number of potential trials is not the natural framing

**Short answer**

Binomial is about successes out of a fixed number of trials; Poisson is about event counts over an interval.

---

# 5. Mean, Median, Mode

These are basic, but they matter because they reveal what kind of distribution you are dealing with.

## Mean

Average.

Sensitive to outliers.

## Median

Middle value.

More robust under skew or extreme values.

## Mode

Most frequent value.

Useful for:

- categorical data
- highly repeated values

**Fashion pricing analogy**

If most outfits cost 5k to 15k and one couture piece costs 4 lakhs, the mean can become dramatic very quickly.
The median usually stays calmer.

---

# 6. Variance and Standard Deviation

Variance measures how spread out values are around the mean.

Standard deviation is just the square root of variance.

Why both matter:

- variance is mathematically convenient
- standard deviation is easier to interpret

Think of them as:

- variance = the formula-friendly version
- standard deviation = the human-friendly version

---

# 7. Correlation vs Covariance

## Covariance

Measures whether two variables move together.

Problem:

It depends on units, so the raw number can be hard to interpret.

## Correlation

Normalized version of covariance.

Much easier to compare across feature pairs because it stays between:

- -1
- 0
- 1

**Important caveat**

Correlation measures linear relationship.
Zero correlation does not mean no relationship.

---

# 8. Correlation vs Causation

This is one of the most important sanity checks in data work.

Just because two things move together does not mean one causes the other.

Possible reasons for correlation:

- confounder
- reverse causality
- coincidence
- selection bias

**Short interview line**

Predictive ML often benefits from correlation, but decision-making and intervention require causal thinking.

That is a very strong sentence.

---

# 9. Law of Large Numbers vs Central Limit Theorem

These two get mixed up a lot.

## Law of Large Numbers

As sample size grows, the sample mean gets closer to the true population mean.

## Central Limit Theorem

As sample size grows, the sampling distribution of the mean becomes approximately normal, under broad conditions.

**Easy memory trick**

- LLN = estimate stabilizes
- CLT = sampling distribution becomes normal

---

# 10. Hypothesis Testing — Key Formulas

## Z-test (known population variance, large n)

$$z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$$

Reject $H_0$ if $|z| > z_{\alpha/2}$ (e.g., $z_{0.025} = 1.96$ for $\alpha = 0.05$ two-tailed).

## Two-sample Z-test (for proportions — common in A/B testing)

$$z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}$$

where $\hat{p} = (x_1 + x_2) / (n_1 + n_2)$ is the pooled proportion.

## T-test (unknown variance, small n)

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}, \quad \text{degrees of freedom} = n-1$$

## Statistical power and sample size

$$\text{Power} = P(\text{reject } H_0 \mid H_1 \text{ true}) = 1 - \beta$$

Required sample size (per group, two proportions):
$$n = \frac{2(z_{\alpha/2} + z_\beta)^2 \cdot p(1-p)}{(\Delta p)^2}$$

where $\Delta p$ is the minimum detectable effect, $z_{\alpha/2} = 1.96$ ($\alpha=0.05$), $z_\beta = 0.84$ (80% power).

**Typical A/B test parameters:** $\alpha = 0.05$ (Type I error), power = 0.8 (Type II = 0.2), MDE = minimum business-meaningful effect.

## P-Value

A p-value is the probability of observing a result at least as extreme as the one you got, **assuming the null hypothesis is true**.

Not:
- the probability the null is true
- the probability your idea is correct
- the probability of the effect being real

**Short answer:** a p-value tells you how surprising the observed data would be if there were actually no effect.

---

# 11. Statistical Significance

If the p-value is below a threshold like 0.05, people often say the result is statistically significant.

But be careful.

Statistical significance does **not** guarantee:

- practical importance
- causality
- production value

This is where many interview answers become weak.

Do not stop at:

> "p < 0.05"

Talk about:

- effect size
- confidence intervals
- business importance

That is the stronger answer.

---

# 12. Type I and Type II Errors

## Type I Error

False positive.

You think there is an effect when there is not.

## Type II Error

False negative.

You miss a real effect.

**Courtroom analogy**

- Type I = convicting an innocent person
- Type II = letting the guilty go free

Still the cleanest analogy.

---

# 13. Confidence Intervals

A confidence interval gives a range of plausible values for an estimate.

It is useful because it tells you more than a single point estimate.

It shows:

- uncertainty
- stability
- likely effect range

**Short interview answer**

Confidence intervals are often more informative than p-values alone because they communicate both uncertainty and effect size.

---

# 14. Bayes' Theorem

**Formal statement:**
$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}$$

where:
- $P(H)$ = **prior** — belief before seeing evidence
- $P(E \mid H)$ = **likelihood** — probability of evidence given hypothesis
- $P(H \mid E)$ = **posterior** — updated belief after evidence
- $P(E) = \sum_h P(E \mid H=h) P(H=h)$ = **marginal likelihood** (normalizing constant)

**Medical example** (base rate fallacy — often asked in interviews):

A disease affects 1% of population. Test has 99% sensitivity (TPR) and 99% specificity (TNR). You test positive — what is the probability you have the disease?

$$P(\text{disease} \mid +) = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.01 \times 0.99} = \frac{0.0099}{0.0198} = 50\%$$

Despite 99% test accuracy, only 50% probability because the disease is rare.

**ML connections:**
- Naive Bayes: apply Bayes' theorem with conditional independence assumption
- MAP estimation: $\hat{\theta} = \arg\max_\theta P(\theta \mid D) = \arg\max_\theta P(D \mid \theta) P(\theta)$
- Bayesian deep learning: treat weights as distributions, not point estimates

---

# 15. Naive Bayes

Naive Bayes uses Bayes' theorem and assumes features are conditionally independent given the class.

That assumption is clearly simplistic.

And yet it works surprisingly well in:

- text classification
- sparse feature settings
- small-data baselines

Why?

Because simple probabilistic structure can still be very effective.

---

# 16. MLE vs MAP

## MLE (Maximum Likelihood Estimation)

Pick parameters that maximize the probability of observed data:
$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(D \mid \theta) = \arg\max_\theta \prod_{i=1}^n p(x_i \mid \theta)$$

In practice, maximize log-likelihood:
$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log p(x_i \mid \theta)$$

## MAP (Maximum A Posteriori)

Pick parameters that maximize posterior probability — likelihood weighted by prior:
$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid D) = \arg\max_\theta \left[ \log P(D \mid \theta) + \log P(\theta) \right]$$

**Regularization as MAP:**

| Prior on $\theta$ | MAP objective | Equivalent regularization |
| :--- | :--- | :--- |
| Gaussian: $\theta \sim \mathcal{N}(0, 1/\lambda)$ | $\log P(D\|\theta) - \frac{\lambda}{2}\|\theta\|^2$ | L2 / Ridge |
| Laplace: $\theta \sim \text{Laplace}(0, 1/\lambda)$ | $\log P(D\|\theta) - \lambda\|\theta\|_1$ | L1 / Lasso |

With large data, MLE ≈ MAP (data overwhelms prior). With small data, prior regularizes and prevents overfitting.

**The punchline:** minimizing cross-entropy loss with L2 regularization is Bayesian MAP estimation with a Gaussian prior on weights.

---

# 17. Bootstrap

Bootstrap is a resampling method.

You repeatedly sample with replacement from the observed data and compute the statistic again and again.

Why it is useful:

- estimate uncertainty
- estimate confidence intervals
- avoid relying on closed-form math when it is inconvenient

It is practical, intuitive, and very interview-worthy.

---

# 18. Statistical Tests for Comparing Models

The right test depends on context.

Examples:

- paired t-test for repeated performance comparisons when assumptions are okay
- McNemar's test for paired classification outcomes
- non-parametric alternatives when distribution assumptions are weak

**Good answer style**

Do not just name a test.
Say:

> "I would choose a test based on whether predictions are paired, whether assumptions hold, and whether I care about model outputs or business outcomes."

That sounds mature.

---

# Quick Thought Experiment

You run an A/B test on two ranking models.
Model B improves CTR by 0.2%.

Before celebrating, what do you ask?

- Is it statistically significant?
- Is the effect size meaningful?
- Did guardrail metrics worsen?
- Was the experiment powered enough?

That is how you keep data science from becoming astrology.

---

# Mini Pop Quiz

If I say:

> "The p-value is 0.03, so there is a 97% chance the model is better."

Is that correct?

No.

That is not what p-values mean.

And fixing that misunderstanding already puts you ahead of a surprising number of people.
