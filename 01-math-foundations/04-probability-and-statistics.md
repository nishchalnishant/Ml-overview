---
module: Interview Prep
topic: Ml
subtopic: Probability And Statistics
status: unread
tags: [interviewprep, ml, ml-probability-and-statistics]
---
# Probability and Statistics

**Primary reference:** [Statistics & Probability Rapid-Fire](_rapid-fire.md) | [Canonical Stats Questions](05-canonical-stats-questions.md)

---

## 1. Why Probability and Statistics Matter in ML

Every ML output is a claim made under uncertainty. Labels are noisy, samples are biased, metrics fluctuate across splits. Probability lets you quantify that uncertainty; statistics lets you decide what's signal vs noise.

**Example:** Test accuracy goes from 82.3% to 83.1% after a change. Is that real or noise? Run a significance test on the per-example predictions to get a p-value. At n=10,000 that 0.8pp gap is likely real; at n=500 it might not be.

**Trap:** treating metrics as exact numbers instead of estimates with sampling variance — every accuracy computed on a finite test set has a confidence interval.

---

## 2. Common Distributions

Identify the distribution from the generative process:
- Fixed number of independent binary trials → **Binomial**
- Rare independent events at a constant rate over an interval → **Poisson**
- Time between independent Poisson events → **Exponential**
- Sum of many independent small effects → **Normal** (CLT)

| Distribution | PMF/PDF | Mean | Variance | ML context |
|---|---|---|---|---|
| Bernoulli | $p^k(1-p)^{1-k}$ | $p$ | $p(1-p)$ | click/no-click, label |
| Binomial | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | # correct in n test examples |
| Poisson | $\lambda^k e^{-\lambda}/k!$ | $\lambda$ | $\lambda$ | requests/sec to a server |
| Exponential | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | time to churn/failure |
| Normal | $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | weight init, CLT sampling dists |

Binomial approximates Normal when $np \geq 5$ and $n(1-p) \geq 5$. Poisson variance = mean — if observed variance is much bigger, use negative binomial instead. Exponential is memoryless: $P(X>s+t \mid X>s) = P(X>t)$.

**Traps:** assuming everything is normal without checking skew; confusing Binomial (fixed n) with Poisson (no natural n, counting arrivals).

---

## 3. Law of Large Numbers vs Central Limit Theorem

**LLN:** the sample mean converges to the true mean as $n \to \infty$. Says nothing about the shape of the distribution.

**CLT:** the standardized sample mean converges to $\mathcal{N}(0,1)$ regardless of the original distribution's shape:
$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1)$$

Rule of thumb: $n \geq 30$ is usually enough, more if the underlying distribution is heavily skewed.

Together: LLN says the estimate is consistent; CLT tells you how uncertain it is, which is what lets you build confidence intervals even on non-normal data.

**Traps:** CLT is about the sampling distribution of the mean, not the raw data. It also requires independence — doesn't hold directly for correlated time-series data.

---

## 4. Hypothesis Testing

Set up: $H_0$ (no effect) vs $H_1$ (effect), and $\alpha$ = acceptable false-positive rate.

**Z-test** (known variance / large n): $z = \frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}$, reject if $|z| > 1.96$ for $\alpha=0.05$.

**Two-sample Z-test for proportions (A/B test):**
$$z = \frac{\hat p_1 - \hat p_2}{\sqrt{\hat p(1-\hat p)(1/n_1 + 1/n_2)}}, \quad \hat p = \frac{x_1+x_2}{n_1+n_2}$$

**T-test** (unknown variance, the usual case): $t = \frac{\bar x - \mu_0}{s/\sqrt n}$, df = $n-1$. Approaches z-test as n grows.

**P-value:** probability of a result this extreme or more, *assuming* $H_0$ is true. Not the probability $H_0$ is true.

**Example:** A/B test, control CTR 4.2% (n=100k) vs treatment 4.5% (n=100k) → z ≈ 4.5, p < 0.001. Statistically significant, but is +0.3pp worth the engineering cost? That's a business call, not a stats one.

**Traps:**
- Stopping a test early once it looks significant inflates Type I error — fix sample size up front.
- Reporting p-values without effect size.
- Testing many metrics at once inflates false positives — use Bonferroni or Benjamini-Hochberg.

---

## 5. Type I and Type II Errors

- **Type I ($\alpha$):** false positive — reject $H_0$ when true.
- **Type II ($\beta$):** false negative — fail to reject $H_0$ when false.

You can't shrink both at a fixed sample size — lowering $\alpha$ raises $\beta$. The right tradeoff depends on the relative cost of each error. Power = $1-\beta$.

Sample size per group (two proportions):
$$n = \frac{2(z_{\alpha/2}+z_\beta)^2 p(1-p)}{(\Delta p)^2}$$

**Example:** missed cancer diagnosis (Type II) is worse than a false alarm (Type I) → lower the detection threshold. Fraud freezing legitimate accounts (Type I) is costly in UX → require stronger evidence, accept some missed fraud.

**Traps:** treating $\alpha=0.05$ as sacred rather than a convention; skipping power analysis before running an experiment (underpowered tests miss real effects).

---

## 6. Confidence Intervals

A 95% CI: if you repeated the sampling+construction process many times, 95% of the resulting intervals would contain the true value. It's not "95% probability the true value is in this one interval" (frequentist parameter is fixed, interval is random).

CIs communicate effect magnitude and practical significance, not just "significant or not." A CI of $[-0.001, 0.011]$ contains zero (not significant); $[0.008, 0.012]$ is real and substantial.

Wald interval for a proportion: $\hat p \pm z_{\alpha/2}\sqrt{\hat p(1-\hat p)/n}$. For small n or extreme $\hat p$, use the Wilson interval instead — Wald can go outside [0,1].

**Traps:** overlapping individual CIs don't imply the difference is non-significant — compute a CI for the difference itself. Wald intervals break down for small proportions (e.g. CTR 0.1%, n=1000).

---

## 7. Bayes' Theorem

$$P(H\mid E) = \frac{P(E\mid H)P(H)}{P(E)}$$

$P(H)$ = prior, $P(E\mid H)$ = likelihood, $P(H\mid E)$ = posterior.

**Base rate fallacy:** disease affects 1% of population, test is 99% sensitive/specific, you test positive:
$$P(\text{disease}\mid +) = \frac{0.99\times0.01}{0.99\times0.01+0.01\times0.99} = 50\%$$

Only 50%, despite 99% test accuracy, because the disease is rare — the prior dominates.

**ML connections:** MAP estimation ($\hat\theta_{MAP} = \arg\max P(D\mid\theta)P(\theta)$, prior regularizes); Naive Bayes (Bayes + feature independence); Bayesian NNs (distributions over weights).

**Example:** fraud detector, 95% sensitivity, 99% specificity, 0.1% fraud rate → $P(\text{fraud}\mid+) \approx 8.7\%$. Over 91% of flags are false positives — not a model failure, a base-rate consequence.

**Trap:** forgetting the base rate in any "given a positive test" problem.

---

## 8. MLE vs MAP

**MLE:** $\hat\theta_{MLE} = \arg\max_\theta \sum_i \log p(x_i\mid\theta)$ — no prior, can overfit on small data.

**MAP:** $\hat\theta_{MAP} = \arg\max_\theta \left[\sum_i \log p(x_i\mid\theta) + \log P(\theta)\right]$ — prior acts as regularizer.

| Prior on $\theta$ | Equivalent regularization |
|---|---|
| Gaussian | L2 / Ridge |
| Laplace | L1 / Lasso |

Cross-entropy + L2 = MAP with a Gaussian prior on weights. As $n\to\infty$, MLE $\approx$ MAP (data overwhelms prior); with small $n$, the prior prevents overfitting.

**Trap:** saying "L2 prevents overfitting" without naming the mechanism (Gaussian prior penalizing large weights).

---

## 9. Bootstrap

Resampling approach to get a confidence interval when the sampling distribution has no closed form (medians, AUC, custom metrics).

1. Draw $B$ samples of size $n$ with replacement from your data.
2. Compute the statistic on each.
3. The 2.5th/97.5th percentiles of the $B$ values give a 95% CI.

**Example:** CI for AUC on a 500-example test set — no closed-form SE exists. Bootstrap 1,000 resamples, take percentiles.

**Traps:** bootstrap doesn't add information, only quantifies uncertainty; unreliable for very small samples ($n<30$); use $B \geq 1000$.

---

## Quick Diagnostics

**Asked to interpret a p-value:** state what it is (P(data this extreme | $H_0$)) and isn't (P($H_0$ true)). Add effect size and CI.

**Asked about A/B test design:** anchor on sample size from MDE/$\alpha$/power. Mention early-stopping inflates Type I error, and multiple-comparison correction if testing several metrics.

**Asked about high accuracy on rare-event detection:** ask about class imbalance first. Precision/recall/F1/PR-AUC are the right metrics, not accuracy. Use Bayes' theorem to reason about PPV given the base rate.
