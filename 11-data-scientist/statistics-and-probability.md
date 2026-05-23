---
module: Data Scientist
topic: Statistics And Probability
subtopic: ""
status: unread
tags: [datascientist, ml, statistics-and-probability]
---
# Statistics and Probability

---

## 1. Descriptive Statistics

### Central Tendency

**The problem**: You have 1,000 numbers. You need to communicate "where is this data centered?" in a single number. Without that, you cannot compare two datasets, set a baseline, or report anything useful.

**Core insight**: "Center" is ambiguous, and different definitions of center are robust to different kinds of data pathology. The choice of summary statistic is a design decision, not a lookup.

---

**Mean**

- **Why it exists**: the most natural notion of "typical value" — share resources equally among all observations. If you had to replace every value with a single number so total sum is preserved, that number is the mean.
- **Core insight**: it minimizes total squared deviation from that single number. It is the least-squares center.
- **The formula**: $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$. The shape is just "add them up, divide by how many" — that's the definition of an arithmetic average.
- **What breaks**: one extreme outlier drags the mean toward it. If your data contains a billionaire in a room of middle-income people, the mean income is meaningless as a description of "typical."

---

**Median**

- **Why it exists**: when outliers exist, the mean lies. You need a measure of center that ignores the extremes entirely. If the mean is a balance point, the median is the midpoint — half the data is above, half below.
- **Core insight**: sort the data, find the middle. No arithmetic involved, so no single value can distort it.
- **The formula**: middle value after sorting; average of two middle values when $n$ is even. No formula because it's a positional statistic.
- **What breaks**: the median discards all information about magnitude. If your data is {1, 5, 9}, the median is 5; if it's {4, 5, 6}, the median is also 5. Both cases look identical to the median even though the spread is completely different.

---

**Mode**

- **Why it exists**: mean and median assume your data has a sensible numerical ordering. For categorical data ("favorite color", "country of origin"), you cannot average or sort. You can only count. The mode is the most common category.
- **Core insight**: find the value with the highest frequency.
- **What breaks**: a distribution can have multiple modes (bimodal, multimodal), which signals you may have mixed populations. Reporting a single mode hides that structure.

---

### Spread

**The problem**: two datasets can have the same mean and median but look completely different. {5, 5, 5, 5} and {1, 3, 7, 9} both have mean 5. The mean alone tells you nothing about how concentrated or dispersed the data is. You need a second number to describe spread.

---

**Variance**

- **Why it exists**: to measure "how far from the mean are points, on average." Simply averaging the raw deviations $x_i - \bar{x}$ gives zero — positive and negative deviations cancel by definition. You need to eliminate cancellation.
- **Core insight**: square the deviations before averaging. Squaring makes all terms positive AND penalizes large deviations disproportionately more than small ones, which matches the intuition that a point far from the mean is more "surprising."
- **The formula**: $s^2 = \frac{1}{n-1}\sum_{i=1}^n(x_i - \bar{x})^2$. The $n-1$ denominator (Bessel's correction) exists because $\bar{x}$ is itself estimated from the same sample, so the residuals $(x_i - \bar{x})$ are artificially compressed — they can't span the full range they would around the true $\mu$. Using $n-1$ corrects for this bias.
- **What breaks**: variance is in squared units. If your data is in meters, variance is in meters². That makes it uninterpretable on the original scale.

---

**Standard Deviation**

- **Why it exists**: variance is uninterpretable because of squared units. You need spread expressed in the same units as the data.
- **Core insight**: take the square root of variance to undo the squaring.
- **The formula**: $s = \sqrt{s^2}$. The shape follows directly — it's just the root of variance.
- **What breaks**: std is still sensitive to outliers because it inherits variance's squared deviations. One large outlier inflates $s$ substantially.

---

**IQR (Interquartile Range)**

- **Why it exists**: std is sensitive to outliers. You need a measure of spread that, like the median, ignores the extremes. The IQR measures the width of the "middle half" of your data.
- **Core insight**: find the 25th and 75th percentiles (Q1 and Q3), subtract them. Whatever is happening in the tails is irrelevant.
- **The formula**: IQR = Q3 − Q1. The shape is a difference of positional statistics, so it inherits robustness from percentile-based summaries.
- **What breaks**: IQR ignores the tails entirely, so it cannot distinguish a distribution with light tails from one with heavy tails — both can have the same IQR.

---

**Percentiles**

- **Why they exist**: a single summary number (mean, median) loses all information about the distribution's shape. You might need to know: "what is the worst-case for 95% of users?" or "where does the top decile start?" Percentiles give you the distribution's shape at any granularity you want.
- **Core insight**: the $p$-th percentile is the value below which $p$% of observations fall. It's a positional statistic — sort the data, find the position.
- **What breaks**: with small $n$, percentiles are noisy estimates. The 99th percentile of a 50-point sample is just one or two data points and can swing wildly with new data.

---

### Shape

**The problem**: mean and std describe location and scale, but two distributions can share both and still look completely different. One might be symmetric, another skewed right, another might have frequent extreme values. You need numbers that describe the shape of the distribution, not just its center and spread.

---

**Skewness**

- **Why it exists**: you want to know whether the data is symmetric or "lopsided." Without this, you might apply a Normal-distribution analysis to data that is severely one-sided — all your p-values and confidence intervals will be wrong.
- **Core insight**: measure the average cubed deviation from the mean, normalized by $s^3$. Cubing preserves sign (unlike squaring), so asymmetry shows up as a nonzero result. Positive cubing means long right tail dominates; negative means long left tail.
- **The formula** (Fisher's): $g_1 = \frac{n}{(n-1)(n-2)} \sum\left(\frac{x_i - \bar{x}}{s}\right)^3$. The correction factor adjusts for small-sample bias.
  - Positive (right) skew: long right tail, mean > median
  - Negative (left) skew: long left tail, mean < median
- **What breaks**: skewness is sensitive to outliers because it involves cubed deviations. A single extreme value can dominate the entire statistic.

---

**Kurtosis**

- **Why it exists**: you have two distributions with the same mean, std, and skewness. One has a sharp peak with heavy tails (extreme values happen more often than Normal), the other is flat. Kurtosis measures this tail-heaviness. In practice: financial returns are leptokurtic — the Normal distribution drastically underestimates the probability of crashes.
- **Core insight**: measure the average fourth-power deviation, normalized by $s^4$. The fourth power heavily amplifies tail observations, so the statistic is dominated by them. Subtracting 3 centers it at zero for a Normal distribution (excess kurtosis).
- **The formula** (excess kurtosis): $g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum\left(\frac{x_i-\bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$
  - Excess kurtosis = 0: Normal distribution
  - Leptokurtic (> 0): heavy tails, sharp peak — e.g., financial returns
  - Platykurtic (< 0): thin tails, flat peak
- **What breaks**: kurtosis is extremely sensitive to outliers — the fourth power means a single extreme observation dominates. With small samples it is nearly meaningless.

---

## 2. Probability Distributions

**The problem**: you want to model random phenomena — number of defective items in a batch, waiting time until an event, whether a user clicks a button. But "random" is too vague to compute with. You need a mathematical object that specifies exactly how probable each outcome is. That object is a probability distribution.

**Core insight**: a distribution is just a function that assigns probabilities to outcomes in a consistent way (probabilities are non-negative and sum to 1). Different physical processes naturally produce different distributional shapes.

---

### Normal Distribution $\mathcal{N}(\mu, \sigma^2)$

- **Why it exists**: you are averaging many independent random contributions. The Central Limit Theorem (see Section 9) guarantees the sum converges to a specific bell-shaped distribution regardless of the source distribution's shape. The Normal is the attractor that all well-behaved averages converge to.
- **Core insight**: it is the maximum-entropy distribution given a fixed mean and variance — i.e., if all you know about a quantity is its mean and spread, Normal is the least-informative (most honest) assumption.
- **The formula**: $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$. The shape: the exponent is a squared Mahalanobis distance from $\mu$, so probability decays as a Gaussian bell curve. The normalizing constant $\frac{1}{\sigma\sqrt{2\pi}}$ exists to make the integral equal 1.
- **68-95-99.7 rule**: 68% of mass within ±1σ, 95% within ±2σ, 99.7% within ±3σ. These are consequences of integrating the Gaussian PDF over those intervals — useful for quick mental arithmetic.
- **What breaks**: real data has tails that are heavier than Normal. Modeling financial returns or rare events as Normal drastically underestimates the probability of extreme outcomes. Normality must be verified, not assumed.

---

### Binomial $B(n, p)$

- **Why it exists**: you flip a biased coin $n$ times and want to know how many heads. More concretely: you run an experiment on $n$ users with click-through probability $p$. How probable is each possible number of successes?
- **Core insight**: each trial is independent and has exactly two outcomes. The number of ways to arrange $k$ successes among $n$ trials is $\binom{n}{k}$. Each specific arrangement has probability $p^k(1-p)^{n-k}$.
- **The formula**: $P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$. Mean: $np$ (expected successes). Variance: $np(1-p)$ (uncertainty is highest when $p=0.5$).
- **What breaks**: the independence assumption. If trials are correlated (one user's click influences another's), the Binomial is wrong. Also: when $n$ is large and $p$ is small, the Binomial becomes computationally awkward — use Poisson instead ($\lambda = np$).

---

### Poisson $\text{Pois}(\lambda)$

- **Why it exists**: you want to model the count of events that happen rarely and independently in a fixed time window — server errors per hour, calls to a call center per minute. The Binomial requires knowing $n$ and $p$ separately, but you often only know the rate $\lambda = np$.
- **Core insight**: take Binomial as $n \to \infty$ and $p \to 0$ with $\lambda = np$ fixed. The Binomial formula converges to the Poisson formula in this limit.
- **The formula**: $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$. The shape: $\lambda^k / k!$ counts "how many ways to get $k$ events at rate $\lambda$," and $e^{-\lambda}$ is the normalization. Mean = Variance = $\lambda$ — a signature property that lets you detect overdispersion (when variance exceeds mean) as a model violation.
- **What breaks**: the equal mean-variance constraint. Real count data is often overdispersed (variance > mean), e.g., web traffic with bursty behavior. Use Negative Binomial instead. Also assumes events are independent within the interval.

---

### Beta $\text{Beta}(\alpha, \beta)$

- **Why it exists**: you want to model a probability itself as a random variable — "what is the true click-through rate of this ad?" A probability must lie in $[0,1]$. Normal and Gamma distributions don't respect that constraint. Beta is the natural distribution over $[0,1]$.
- **Core insight**: if you've observed $\alpha - 1$ successes and $\beta - 1$ failures, the Beta distribution is the posterior for the success probability under a uniform prior. $\alpha$ and $\beta$ act like pseudo-counts.
- **The formula**: PDF $\propto x^{\alpha-1}(1-x)^{\beta-1}$ on $[0,1]$. Mean: $\frac{\alpha}{\alpha+\beta}$. Mode: $\frac{\alpha-1}{\alpha+\beta-2}$ (for $\alpha, \beta > 1$). $\text{Beta}(1,1)$ = Uniform — no prior information.
- **Conjugate prior property**: if prior is $\text{Beta}(\alpha, \beta)$ and you observe $k$ successes in $n$ trials, posterior is $\text{Beta}(\alpha+k, \beta+n-k)$. This makes Bayesian updating closed-form.
- **What breaks**: Beta is only for data confined to $[0,1]$. For general proportions outside that range, it doesn't apply. Also, $\alpha < 1$ or $\beta < 1$ creates U-shaped or J-shaped distributions, which may be appropriate but are counterintuitive.

---

### Gamma $\text{Gamma}(\alpha, \beta)$

- **Why it exists**: you want to model waiting times or positive continuous quantities. How long until the $\alpha$-th event in a Poisson process? How do you put a prior on a rate parameter $\lambda > 0$? The Gamma distribution is the answer in both cases.
- **Core insight**: waiting time until the first event in a Poisson process is Exponential. Waiting time until the $\alpha$-th event is the sum of $\alpha$ independent Exponentials — and sums of i.i.d. Exponentials follow a Gamma distribution.
- **The formula**: Mean: $\alpha/\beta$, Variance: $\alpha/\beta^2$. Special cases: Exponential ($\alpha=1$), Chi-squared ($\alpha=k/2$, $\beta=1/2$).
- **What breaks**: Gamma assumes events arrive independently at a constant rate. Real processes often have time-varying rates (non-stationarity), making the Poisson/Gamma framework inappropriate.

---

### t-Distribution $t_\nu$

- **Why it exists**: you want to do inference on a mean, but you don't know the population standard deviation and your sample is small. If you use the Normal distribution and plug in the sample std, you underestimate uncertainty — the resulting confidence intervals are too narrow and tests reject too often. The t-distribution corrects for this by having heavier tails.
- **Core insight**: when you estimate $\sigma$ from data, you introduce additional uncertainty. The ratio $(\bar{X} - \mu) / (s/\sqrt{n})$ doesn't follow a Normal — it follows a t-distribution with $n-1$ degrees of freedom. As $n$ grows and $s$ becomes a reliable estimate of $\sigma$, the t-distribution converges to Normal.
- **The formula**: degrees of freedom $\nu = n - 1$ for one-sample tests. Heavier tails for small $\nu$; as $\nu \to \infty$, converges to $\mathcal{N}(0,1)$.
- **What breaks**: the t-distribution still assumes the underlying data is approximately Normal (or $n$ is large enough for CLT). With very non-Normal data and small $n$, even t-tests can mislead. Use non-parametric alternatives.

---

### Chi-Squared $\chi^2_k$

- **Why it exists**: you want to test how much a set of observed frequencies deviates from expected frequencies, or whether two categorical variables are associated. You need a test statistic that accumulates evidence from all categories simultaneously. Summing squared Normally distributed quantities gives you a chi-squared distribution.
- **Core insight**: if $Z_1, \ldots, Z_k$ are independent standard Normals, then $\sum Z_i^2 \sim \chi^2_k$. Standardized residuals $(O-E)/\sqrt{E}$ are approximately Normal, so their squares sum to a chi-squared statistic.
- **The formula**: Mean: $k$, Variance: $2k$, where $k$ = degrees of freedom.
- **What breaks**: the chi-squared approximation requires expected cell counts $\geq 5$. With sparse cells, the Normal approximation for residuals breaks down and the test becomes unreliable. Use Fisher's exact test for small samples.

---

### F-Distribution $F_{d_1, d_2}$

- **Why it exists**: you want to compare variances across groups or test whether a regression model explains significantly more variance than a null model. You need a distribution for a ratio of variances. The F-distribution is that ratio.
- **Core insight**: the ratio of two independent chi-squared statistics, each divided by their degrees of freedom, follows an F-distribution. $F = \frac{\chi^2_{d_1}/d_1}{\chi^2_{d_2}/d_2}$. If the two variances are equal, this ratio is near 1; large F means the numerator variance is disproportionately large.
- **What breaks**: F-tests assume Normality of residuals and homoscedasticity. For ANOVA, violations of these assumptions inflate Type I error rates.

---

## 3. Hypothesis Testing

**The problem**: you run an A/B test. Treatment group's mean is 2% higher than control. Is this a real effect or just random noise from sampling? You need a principled procedure for deciding when observed data is surprising enough under "no effect" to conclude the effect is real.

**Core insight**: assume there is no effect. Compute how likely your observed data (or more extreme data) would be under that assumption. If that probability is very small, the data is hard to explain by chance alone, and you reject the "no effect" assumption.

---

### The Framework

- **Null hypothesis $H_0$**: the boring baseline — "no effect," "no difference," "status quo is true." Assumed true until evidence against it.
- **Alternative hypothesis $H_1$**: what you want to detect — "there is a difference," "the treatment works."
- **p-value**: the probability of observing a test statistic at least as extreme as yours, *assuming $H_0$ is true*. This is not the probability that $H_0$ is true — a common and costly misinterpretation.
- **Significance level $\alpha$**: your pre-specified threshold for "surprising enough." If p-value < $\alpha$, reject $H_0$. The threshold controls how often you falsely reject when $H_0$ is true.
- **Type I error ($\alpha$)**: reject $H_0$ when it is true — false positive. You are willing to tolerate this at rate $\alpha$.
- **Type II error ($\beta$)**: fail to reject $H_0$ when it is false — false negative. You miss a real effect.
- **Power = $1 - \beta$**: probability of detecting a true effect. Power depends on sample size, effect size, and $\alpha$.
- **One-tailed vs two-tailed**: one-tailed tests a directional hypothesis (e.g., "treatment is better than control"). Two-tailed tests for any difference. Use two-tailed by default — one-tailed is only valid when direction is pre-specified and theoretically motivated, never post-hoc.

---

### z-test

- **Why it exists**: you want to test whether a sample mean differs from a known value, and you either know the population standard deviation or have a large enough sample that your sample std is reliable.
- **Core insight**: standardize the sample mean under $H_0$. The result follows $\mathcal{N}(0,1)$ (by CLT for large $n$), so you can use Normal tables to find p-values.
- **The formula**: $z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}$. The shape: numerator is "how far from the hypothesized mean," denominator is the standard error (how much the sample mean varies by chance).
- **What breaks**: requires known $\sigma$ or $n \geq 30$ for CLT to be adequate. With small $n$ and unknown $\sigma$, use a t-test.

---

### t-test

- **Why it exists**: same problem as z-test, but you don't know $\sigma$ and your sample is small. Plugging in $s$ for $\sigma$ in the z-test underestimates uncertainty. The t-test uses the t-distribution to account for the extra variability introduced by estimating $\sigma$.
- **Core insight**: replacing $\sigma$ with $s$ changes the distribution of the test statistic from Normal to t (heavier tails). The degrees of freedom parameter governs how much heavier.
- **The formulas**:
  - **One-sample**: $t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$, df = $n-1$
  - **Two-sample independent**: $t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_p^2(1/n_1 + 1/n_2)}}$ (pooled), or Welch's t-test when variances are unequal (use Welch's by default)
  - **Paired**: $t = \frac{\bar{d}}{s_d/\sqrt{n}}$ where $d_i = x_{i,\text{after}} - x_{i,\text{before}}$ — use when each unit appears in both conditions, because pairing removes between-subject noise
- **What breaks**: assumes approximate Normality of the data (or $n$ large enough for CLT). With small $n$ and severely non-Normal data, p-values are unreliable. Use Mann-Whitney U instead.

---

### ANOVA

- **Why it exists**: you have more than two groups and want to know if any means differ. Running pairwise t-tests inflates Type I error (see Section 4). ANOVA tests all groups simultaneously in one test.
- **Core insight**: if all groups have the same mean, then variance between group means should be comparable to variance within groups (just noise). If between-group variance is much larger than within-group variance, some means must differ.
- **The formula**: $F = \frac{MS_B}{MS_W} = \frac{SS_B/(k-1)}{SS_W/(N-k)}$, where $k$ = number of groups, $N$ = total observations. Large $F$ is evidence that between-group variance exceeds within-group variance.
- **What breaks**: ANOVA only tells you *some* means differ — not which pairs. Post-hoc tests (Tukey HSD, Bonferroni-corrected pairwise t-tests) are needed to identify which pairs. ANOVA also assumes homoscedasticity (equal variances); use Welch's ANOVA otherwise.

---

### Chi-Squared Test

- **Why it exists**: you have categorical data — not means, but counts. You want to know if two categorical variables are associated (e.g., "does gender affect product preference?") or if observed frequencies match a theoretical distribution.
- **Core insight**: under the null hypothesis of independence, the expected count in each cell is (row total × column total) / grand total. Deviations of observed from expected, squared and normalized, accumulate into a chi-squared statistic.
- **The formula**: $\chi^2 = \sum \frac{(O-E)^2}{E}$
  - Independence test: df = $(r-1)(c-1)$ for an $r \times c$ contingency table
  - Goodness-of-fit test: df = $k-1$ for $k$ categories
- **What breaks**: expected cell count < 5 makes the chi-squared approximation unreliable. Collapse categories or use Fisher's exact test.

---

### Non-Parametric Tests

- **Why they exist**: t-tests and ANOVA assume approximately Normal data (or large $n$). With small samples and clearly non-Normal data — ordinal scales, skewed distributions, heavy tails — those assumptions fail. Non-parametric tests make no distributional assumptions because they operate on ranks rather than raw values.

**Mann-Whitney U**
- **Core insight**: instead of comparing means, ask: if you pick one observation from each group at random, which group's observation is more likely to be larger? Test whether $P(X > Y) = 0.5$.
- **The formula**: convert all observations to ranks, then compute the rank-sum statistic. Equivalent to testing whether distributions are identical (shift alternative).
- **What breaks**: the test is less powerful than a t-test when the t-test's assumptions are actually met. It also does not directly test means — it tests distributional dominance.

**Kruskal-Wallis**
- **Core insight**: non-parametric analogue of one-way ANOVA. Rank all observations ignoring group membership, then test whether rank distributions differ across groups.
- **What breaks**: like ANOVA, it only detects that some groups differ — post-hoc pairwise Mann-Whitney tests are needed with Bonferroni correction.

---

## 4. Multiple Testing

**The problem**: you test 20 hypotheses simultaneously, all true nulls, each at $\alpha = 0.05$. What is the probability of getting at least one false positive?

$P(\text{at least one false positive}) = 1 - (1 - 0.05)^{20} \approx 0.64$

64% chance of a false discovery even when nothing is real. Running 20 A/B tests at once and reporting the "winner" without correction virtually guarantees you will ship a false improvement.

**Core insight**: the $\alpha = 0.05$ threshold controls the false positive rate *per test*. When you run many tests, errors accumulate. You need to either tighten each threshold or control the overall error rate at the family level.

---

**Bonferroni Correction**

- **Why it exists**: the simplest solution — divide your desired family-wise error rate by the number of tests to get the per-test threshold.
- **Core insight**: by the union bound, $P(\text{any false positive}) \leq \sum_i P(\text{false positive}_i) = m \cdot \alpha'$. Setting $\alpha' = \alpha/m$ guarantees $P(\text{any false positive}) \leq \alpha$.
- **The formula**: $\alpha_{\text{adjusted}} = \alpha / m$.
- **What breaks**: Bonferroni is very conservative when tests are positively correlated (as they often are in practice — e.g., testing correlated genes). It throws away power unnecessarily, increasing Type II errors. Use when you absolutely need to control FWER and tests are independent.

---

**Benjamini-Hochberg FDR**

- **Why it exists**: Bonferroni controls the probability of *any* false positive. But when you are testing thousands of genes, getting one or two false discoveries in a list of 50 true discoveries may be acceptable — you just want to bound what fraction of your discoveries are false. FDR (False Discovery Rate) controls the expected proportion of false discoveries among all rejections.
- **Core insight**: sort p-values from smallest to largest. Reject all hypotheses up to the largest $p_{(i)}$ satisfying $p_{(i)} \leq \frac{i}{m} \cdot q^*$. This linearly relaxes the threshold as you move down the sorted list.
- **The formula**: sort $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$. Find the largest $i$ such that $p_{(i)} \leq \frac{i}{m} q^*$. Reject all $H_{(1)}, \ldots, H_{(i)}$. Target FDR is $q^*$.
- **What breaks**: BH controls expected FDR under independence or positive dependence. For arbitrary dependence, use BHY (Benjamini-Hochberg-Yekutieli). FDR also does not control per-comparison error rates — individual discoveries may still be false.

---

## 5. Confidence Intervals

**The problem**: you have a point estimate $\hat{\mu} = 52$ from your sample. But you know that if you repeated the study, you'd get a slightly different number. How should you communicate the uncertainty around your estimate? A single number without any uncertainty bound is misleading.

**Core insight**: construct an interval that will contain the true parameter in a specified fraction of repeated experiments. This is a property of the *procedure*, not of any specific interval.

---

**Frequentist Confidence Interval**

- **Why this wording matters**: a 95% CI does not mean "there is a 95% probability that $\mu$ is in this interval." The true $\mu$ is a fixed unknown — it either is or isn't in your specific interval. The 95% is a property of the procedure: if you repeated the experiment many times, 95% of the constructed intervals would contain $\mu$.
- **The formula** (for a mean, Normal case): $\bar{x} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}$ where $z_{\alpha/2} = 1.96$ for 95%. The half-width is exactly the margin of error.
- **Relation to hypothesis tests**: a 95% CI is the set of all $\mu_0$ values you would *fail* to reject at $\alpha = 0.05$. This duality is exact — rejecting $H_0: \mu = \mu_0$ iff $\mu_0 \notin \text{95\% CI}$.
- **What breaks**: the formula above assumes Normality (or large $n$). For small $n$, use t-distribution critical values instead of z.

---

**Bootstrap Confidence Interval**

- **Why it exists**: for complex statistics (median, correlation, ratio of variances, model coefficients), there is no closed-form sampling distribution. You cannot derive a formula. The bootstrap solves this by simulating the sampling process using the data itself.
- **Core insight**: treat your observed sample as a proxy for the population. Resample from it with replacement $B$ times (each resample is the same size as the original). Compute your statistic on each resample. The distribution of that statistic across resamples approximates the sampling distribution.
- **The formula** (percentile method): $[\hat{\theta}_{(\alpha/2)}, \hat{\theta}_{(1-\alpha/2)}]$ — use the $\alpha/2$ and $1-\alpha/2$ quantiles of the bootstrap distribution. BCa (bias-corrected accelerated) bootstrap corrects for both bias and skewness in the bootstrap distribution and is preferred in practice.
- **What breaks**: the bootstrap assumes your sample is representative of the population. With extreme outliers or very small $n$, the bootstrap distribution may be a poor approximation. Also computationally intensive for large datasets.

---

## 6. Bayesian Statistics

**The problem**: you run a clinical trial with 10 patients and 7 recover. The frequentist estimate for recovery rate is 70%. But you know from prior studies that recovery rates for this disease are typically 40-60%. How do you incorporate that prior knowledge? How do you update beliefs as new evidence arrives?

**Core insight**: treat the unknown parameter $\theta$ as a random variable with a probability distribution. Specify your prior beliefs about $\theta$ before seeing data. Update that distribution using Bayes' theorem when data arrives. The posterior distribution is your updated belief.

---

### The Core Framework

- **Prior $P(\theta)$**: your belief about $\theta$ before seeing data. Can encode genuine prior knowledge or be deliberately vague (uninformative prior).
- **Likelihood $P(D|\theta)$**: probability of observing your data given a specific value of $\theta$. This is the same function frequentists maximize for MLE.
- **Posterior $P(\theta|D) \propto P(D|\theta) \cdot P(\theta)$**: your updated belief after seeing data. Proportional to likelihood × prior. The normalizing constant is $P(D) = \int P(D|\theta)P(\theta)d\theta$.
- **Why this formula has this shape**: Bayes' theorem is just the definition of conditional probability rearranged: $P(\theta|D) = P(D|\theta)P(\theta)/P(D)$. The posterior is prior belief weighted by how well $\theta$ explains the data.

---

### Estimation Under the Bayesian Framework

- **MLE**: $\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(D|\theta)$ — maximize the likelihood, ignoring the prior entirely. Equivalent to the mode of the posterior under a uniform prior.
- **MAP (Maximum A Posteriori)**: $\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta|D)$ — maximize the posterior. Uses prior information. Equivalent to MLE with L2 regularization when prior is Gaussian, or L1 regularization when prior is Laplace. This is why regularization has a Bayesian interpretation.
- **Posterior mean**: the full Bayesian estimate; minimizes expected squared error. Does not reduce the posterior to a point — you can report the entire distribution.

---

### Conjugate Priors

- **Why they exist**: computing the posterior integral $P(D) = \int P(D|\theta)P(\theta)d\theta$ is often intractable. Conjugate priors are chosen so the posterior has the same functional form as the prior, making the update a simple parameter update with no integration.

| Likelihood | Conjugate Prior | Posterior |
| :--- | :--- | :--- |
| Bernoulli/Binomial | Beta($\alpha$, $\beta$) | Beta($\alpha+k$, $\beta+n-k$) |
| Poisson | Gamma($\alpha$, $\beta$) | Gamma($\alpha+\sum x$, $\beta+n$) |
| Normal (known $\sigma$) | Normal($\mu_0$, $\tau^2$) | Normal (weighted average of prior mean and sample mean) |

- **What breaks**: conjugate priors are mathematically convenient but may not represent your actual prior beliefs. Using a conjugate prior when your genuine prior is non-conjugate introduces prior misspecification bias.

---

### Credible Intervals vs Confidence Intervals

- **Why this distinction matters**: "the parameter has a 95% probability of being in this range" sounds like what a confidence interval says — but it isn't.
- **Credible interval** (Bayesian): directly states $P(\theta \in [a,b] | D) = 0.95$. This is a genuine probability statement about $\theta$, valid because $\theta$ is treated as a random variable with a posterior distribution.
- **Confidence interval** (frequentist): $\theta$ is fixed. The interval is random (it depends on data). 95% CI means: the procedure produces intervals that cover the true $\theta$ in 95% of repeated experiments. No probability statement about any specific interval is valid.
- **Practically**: the two are numerically similar when the prior is weak relative to the data. The distinction matters most when $n$ is small and the prior is informative.

---

## 7. Correlation

**The problem**: you have two variables — say, hours studied and exam score. You want to know: are these related? Do they move together? Having to look at a scatterplot every time is impractical. You need a single number that summarizes the strength and direction of the relationship.

**Core insight**: standardize both variables and measure how much they move in the same direction simultaneously. When both are above their means together and below their means together, the product of their standardized values is positive — strong positive correlation. When one is above while the other is below, the product is negative.

---

### Pearson $r$

- **Why it exists**: measures the strength of a *linear* relationship between two continuous variables.
- **Core insight**: compute the average product of standardized deviations. If $x$ and $y$ tend to be on the same side of their means simultaneously, this product is positive on average.
- **The formula**: $r = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \sum(y_i-\bar{y})^2}}$. The denominator normalizes so that $r \in [-1, 1]$.
- **What breaks**: Pearson measures only *linear* association. Two variables can be perfectly related (e.g., $y = x^2$) and have $r \approx 0$. Also sensitive to outliers — one extreme point can make a weak relationship look strong or vice versa.

---

### Spearman $\rho$

- **Why it exists**: Pearson fails for non-linear monotonic relationships and is distorted by outliers. You want a correlation measure that captures "as $x$ increases, does $y$ tend to increase?" without assuming linearity, and without being dominated by extreme values.
- **Core insight**: replace raw values with their ranks, then apply Pearson's formula to the ranks. Ranks are bounded and treat the spacing between observations uniformly, making the statistic robust.
- **The formula**: Pearson $r$ applied to the ranks of $x$ and $y$. Measures monotonic (not necessarily linear) association.
- **What breaks**: Spearman discards magnitude information. It cannot distinguish a steep relationship from a shallow one — only the rank ordering matters.

---

### Kendall $\tau$

- **Why it exists**: Spearman can be unstable with small samples or many ties. Kendall $\tau$ provides a more interpretable and robust alternative.
- **Core insight**: for every pair of observations $(i, j)$, ask whether they are *concordant* (both $x$ and $y$ rank the same way: $x_i > x_j$ and $y_i > y_j$, or vice versa) or *discordant* (they rank oppositely). The correlation is just the excess of concordant over discordant pairs.
- **The formula**: $\tau = \frac{\text{concordant pairs} - \text{discordant pairs}}{\binom{n}{2}}$. Directly interpretable as $P(\text{concordant}) - P(\text{discordant})$ for a randomly selected pair.
- **What breaks**: $O(n^2)$ computation for large $n$ (though $O(n \log n)$ algorithms exist). Like Spearman, it detects association but cannot characterize its functional form.

---

### Partial Correlation

- **Why it exists**: $X$ and $Y$ may appear correlated only because both are driven by a third variable $Z$ (a confounder). You want to measure the direct relationship between $X$ and $Y$ after removing $Z$'s influence.
- **Core insight**: regress $X$ on $Z$ and regress $Y$ on $Z$. Take the residuals from both regressions. The partial correlation of $X$ and $Y$ controlling for $Z$ is the Pearson correlation between those residuals.
- **What breaks**: partial correlation only removes the *linear* effect of $Z$. Non-linear confounding remains.

---

### Correlation ≠ Causation

- **Why this warning is here**: correlation is seductive. A high $r$ value feels like evidence of causation. It isn't.
- **Spurious correlation**: both $X$ and $Y$ are caused by a common variable $Z$. Example: ice cream sales correlate with drowning deaths — both are caused by summer heat.
- **Reverse causation**: $Y$ causes $X$, not $X$ causes $Y$. Example: hospitals are correlated with sickness — hospitals don't cause sickness.
- **Coincidental correlation**: no causal connection whatsoever. Example: per-capita cheese consumption correlates with people dying tangled in bedsheets.
- **Core principle**: to establish causation you need either a randomized experiment (which eliminates confounding by design) or a valid quasi-experimental design with strong assumptions.

---

## 8. Regression Inference

**The problem**: you fit a regression line to data: $\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x$. You get $\hat{\beta}_1 = 2.3$. But is that a reliable estimate, or noise? If you collected different data, would you get a similar coefficient? You need not just point estimates but uncertainty quantification for regression coefficients.

**Core insight**: OLS estimates are random variables — they depend on the sample. Under certain assumptions, their sampling distributions are known, enabling hypothesis tests and confidence intervals for each coefficient.

---

### OLS Assumptions (LINE)

Each assumption is listed alongside what *breaks* when it is violated:

- **Linearity**: $E[Y|X] = X\beta$. If the true relationship is non-linear and you fit a line, your coefficients are biased estimates of a misspecified quantity. Detected with residual-vs-fitted plots.
- **Independence**: residuals $\epsilon_i$ are independent across observations. Violated by time series (autocorrelation) or grouped data (clustering). Produces standard errors that are too small, inflating t-statistics.
- **Normality**: residuals are approximately Normal — needed for exact finite-sample inference. With large $n$, CLT makes this less critical for the distribution of estimates.
- **Equal variance (homoscedasticity)**: $\text{Var}(\epsilon_i) = \sigma^2$ constant across $X$. Violated when residuals fan out at higher fitted values (heteroscedasticity). Makes standard errors wrong — use HC (heteroscedasticity-consistent) robust standard errors.

---

### Model Fit

**R²**
- **Why it exists**: you want to know how much of the outcome's variance your model explains. An absolute measure of fit.
- **Core insight**: compare your model's residual variance to the variance you'd have using just the mean: $R^2 = 1 - SS_{\text{res}}/SS_{\text{tot}}$.
- **What breaks**: $R^2$ never decreases when you add more predictors, even useless random noise ones. A model with 100 random predictors will have high $R^2$ with enough data. Use Adjusted $R^2$ for model comparison.

**Adjusted R²**
- **Why it exists**: $R^2$ rewards complexity. You need a version that penalizes for additional predictors.
- **The formula**: $\bar{R}^2 = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$, where $p$ = number of predictors. Adding a predictor only improves $\bar{R}^2$ if it explains more variance than it costs in degrees of freedom.

**F-test for overall fit**
- **Why it exists**: after fitting a model with $p$ predictors, you want to test whether *any* of them are useful — i.e., whether the model is better than just predicting the mean.
- **The formula**: $F = \frac{R^2/p}{(1-R^2)/(n-p-1)}$. Under $H_0: \beta_1 = \ldots = \beta_p = 0$, this follows $F_{p, n-p-1}$.

**t-test per coefficient**
- **Why it exists**: the F-test is global. You want to know whether each individual predictor contributes, *given all other predictors are already in the model*.
- **The formula**: $t = \hat{\beta}_j / \text{SE}(\hat{\beta}_j)$, df = $n-p-1$. The SE depends on all other predictors — this is why multicollinearity inflates SEs.

---

### Multicollinearity

- **The problem**: two predictors $X_1$ and $X_2$ are highly correlated. When you include both in a regression, the model cannot attribute credit independently — it can trade off any value of $\hat{\beta}_1$ against a compensating value of $\hat{\beta}_2$ and still fit the data equally well. Coefficient estimates become wildly unstable.
- **Core insight**: measure how much of $X_j$'s variance is explained by the other predictors. If $X_j$ is nearly a linear combination of the others, its coefficient is poorly identified.
- **The formula**: VIF (Variance Inflation Factor) $= \frac{1}{1 - R^2_j}$, where $R^2_j$ is the $R^2$ from regressing $X_j$ on all other predictors. VIF = 5 means the SE of $\hat{\beta}_j$ is $\sqrt{5} \approx 2.2\times$ larger than it would be without collinearity. VIF > 10: severe.
- **What breaks**: high VIF makes individual coefficients uninterpretable, but it doesn't necessarily hurt prediction accuracy. If your goal is prediction (not inference on coefficients), multicollinearity is less of a problem.
- **Remedies**: drop one of the correlated features, combine them (PCA), or use ridge regression which explicitly penalizes large coefficients and is robust to collinearity.

---

## 9. Central Limit Theorem and Law of Large Numbers

**The problem**: you want to estimate a population mean. You collect a sample. But why should your sample mean tell you anything reliable about the population mean? And if you could somehow collect an infinite sample, would your estimate converge? These are two different questions with two different answers.

---

### Law of Large Numbers (LLN)

- **The question**: does the sample mean actually converge to the true mean as $n$ grows? Without this guarantee, statistical estimation has no foundation.
- **Core insight**: averaging is a noise-reduction process. Each observation is the true mean plus noise. When you average $n$ observations, the noise from different observations is independent and partially cancels. As $n \to \infty$, the noise cancels completely and you are left with the signal.
- **The formulas**:
  - **Weak LLN**: $\bar{X}_n \xrightarrow{p} \mu$ (convergence in probability) — for any $\epsilon > 0$, $P(|\bar{X}_n - \mu| > \epsilon) \to 0$
  - **Strong LLN**: $\bar{X}_n \to \mu$ almost surely — the sample paths converge, not just the probabilities
- **Why both versions exist**: weak LLN allows occasional bad samples; strong LLN says bad samples become increasingly rare. For most practical purposes, weak LLN suffices.
- **Foundation for Monte Carlo**: the LLN guarantees that averaging many random samples from a distribution converges to the expectation. This is why Monte Carlo simulation works for integration.
- **What breaks**: LLN requires finite mean. For heavy-tailed distributions like Cauchy (which has undefined mean), the sample mean does not converge — it wanders without bound. The LLN also says nothing about *how fast* convergence happens — that's the CLT's job.

---

### Central Limit Theorem (CLT)

- **The question**: the LLN says the sample mean converges to $\mu$. But how fast? And what does the sampling distribution of $\bar{X}$ look like for finite $n$? Without knowing the shape of the sampling distribution, you cannot build confidence intervals or run hypothesis tests.
- **Core insight**: regardless of the population distribution's shape (as long as it has finite mean and variance), the distribution of the standardized sample mean converges to a standard Normal. The Normal distribution is the universal limit for averages of i.i.d. random variables.
- **The formula**: for i.i.d. samples with finite mean $\mu$ and variance $\sigma^2$: $\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$, equivalently $\bar{X}_n \approx \mathcal{N}(\mu, \sigma^2/n)$ for large $n$.
- **Why this formula has this shape**: the $\sqrt{n}$ scaling is not arbitrary. Averaging $n$ observations reduces variance by a factor of $n$ (variances add for independent variables, so variance of $\bar{X}$ is $\sigma^2/n$). The standard error $\sigma/\sqrt{n}$ is the square root of that.
- **Practical consequence**: this is why t-tests, z-tests, and Normal-based confidence intervals are valid for large samples even when the underlying data is not Normal. The CLT licenses the Normality assumption for sample means.
- **What breaks**:
  - **Heavy-tailed distributions**: if variance is infinite (Cauchy distribution, Pareto with tail index $\leq 2$), CLT does not apply. The distribution of the sample mean does not converge to Normal.
  - **Small $n$**: "large enough" depends on the skewness of the distribution. Roughly: $n \geq 30$ for mildly skewed, $n \geq 100$ for heavily skewed, $n \geq 1000$ for distributions with extreme kurtosis.
  - **Dependent observations**: CLT requires independence (or weak dependence). Time series data with strong autocorrelation violates this — specialized CLTs for dependent processes exist but have stricter conditions.
  - **Rate of convergence**: CLT says the limiting distribution is Normal but says nothing about how closely the finite-sample distribution matches it. The Berry-Esseen theorem quantifies this rate: the approximation error is $O(1/\sqrt{n})$.

---

## Canonical Interview Q&As

**Q: Explain the central limit theorem and why it's the foundation of most A/B test analysis.**
A: The CLT states: for independent, identically distributed random variables X_1,...,X_n with mean μ and variance σ², the sample mean X̄ = (1/n)ΣX_i converges in distribution to N(μ, σ²/n) as n→∞, regardless of the underlying distribution of X_i. This is foundational to A/B testing because: (1) we never know the true distribution of conversion rates, revenue, or engagement — users don't follow a textbook distribution; (2) the CLT guarantees that the sample mean (the metric we care about) is approximately normally distributed for large enough samples (~30+ per variant for symmetric metrics, 1000+ for heavy-tailed metrics like revenue); (3) this normality lets us compute p-values and confidence intervals analytically using the Z-test or t-test without assuming anything about the underlying data. The key assumptions that must hold: independence (users shouldn't influence each other — violates in social networks), identical distribution (users don't change behavior mid-experiment — violated by novelty effects). When these fail (e.g., network effects), the CLT-based analysis underestimates variance, producing false positives.

**Q: What is MLE and derive the MLE estimator for the mean of a Gaussian.**
A: Maximum Likelihood Estimation finds parameters θ that maximize the probability of observing the data: θ_MLE = argmax_θ P(data | θ). For data x_1,...,x_n drawn from N(μ, σ²), the likelihood is L(μ,σ²) = Π_i (1/√(2πσ²))·exp(-(x_i-μ)²/(2σ²)). Taking the log-likelihood: ℓ = -n/2·log(2πσ²) - Σ(x_i-μ)²/(2σ²). Setting ∂ℓ/∂μ = 0: Σ(x_i-μ)/σ² = 0 → μ_MLE = (1/n)Σx_i = x̄. The sample mean is the MLE for the Gaussian mean — it's the value of μ that makes the observed data most probable. For σ²: ∂ℓ/∂σ² = 0 gives σ²_MLE = (1/n)Σ(x_i-x̄)² (biased by factor (n-1)/n; the unbiased estimator s² = (1/(n-1))Σ(x_i-x̄)² is preferred). MLE has desirable properties: consistency (converges to true θ as n→∞), asymptotic normality, and asymptotic efficiency (minimum variance among unbiased estimators). Connection to cross-entropy: minimizing cross-entropy loss is equivalent to MLE under the model's distributional assumptions.

**Q: What is a p-value and what is a common misconception about it?**
A: The p-value is the probability of observing a test statistic as extreme as or more extreme than the one computed, assuming the null hypothesis is true: p = P(|T| ≥ |t_obs| | H₀). Common misconception: "p < 0.05 means there's a 95% chance the effect is real" — this is wrong. The p-value is a statement about the data given H₀, not about H₀ given the data. It doesn't tell you: the probability the null is true, the size of the effect, or the probability that results will replicate. What it does tell you: how surprising the observed data would be if the null were true — low p-value means the data is inconsistent with the null. Correct interpretation: "If the null hypothesis were true (no difference between variants), we would observe a test statistic this extreme or more by chance with probability p." Additional common errors: (1) p-hacking — running multiple tests and stopping when p < 0.05 inflates false positive rate; fix with Bonferroni correction or sequential testing (always-valid p-values); (2) underpowered tests — large p-value doesn't mean no effect, it means insufficient evidence; (3) significant ≠ practically meaningful — a 0.1% conversion improvement may be statistically significant with millions of users but not worth shipping.

## Flashcards

**Why it exists: the most natural notion of "typical value"?** #flashcard
share resources equally among all observations. If you had to replace every value with a single number so total sum is preserved, that number is the mean.

**Core insight?** #flashcard
it minimizes total squared deviation from that single number. It is the least-squares center.

**The formula: $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$. The shape is just "add them up, divide by how many"?** #flashcard
that's the definition of an arithmetic average.

**What breaks?** #flashcard
one extreme outlier drags the mean toward it. If your data contains a billionaire in a room of middle-income people, the mean income is meaningless as a description of "typical."

**Why it exists: when outliers exist, the mean lies. You need a measure of center that ignores the extremes entirely. If the mean is a balance point, the median is the midpoint?** #flashcard
half the data is above, half below.

**Core insight?** #flashcard
sort the data, find the middle. No arithmetic involved, so no single value can distort it.

**The formula?** #flashcard
middle value after sorting; average of two middle values when $n$ is even. No formula because it's a positional statistic.

**What breaks?** #flashcard
the median discards all information about magnitude. If your data is {1, 5, 9}, the median is 5; if it's {4, 5, 6}, the median is also 5. Both cases look identical to the median even though the spread is completely different.

**Why it exists?** #flashcard
mean and median assume your data has a sensible numerical ordering. For categorical data ("favorite color", "country of origin"), you cannot average or sort. You can only count. The mode is the most common category.

**Core insight?** #flashcard
find the value with the highest frequency.

**What breaks?** #flashcard
a distribution can have multiple modes (bimodal, multimodal), which signals you may have mixed populations. Reporting a single mode hides that structure.

**Why it exists: to measure "how far from the mean are points, on average." Simply averaging the raw deviations $x_i - \bar{x}$ gives zero?** #flashcard
positive and negative deviations cancel by definition. You need to eliminate cancellation.

**Core insight?** #flashcard
square the deviations before averaging. Squaring makes all terms positive AND penalizes large deviations disproportionately more than small ones, which matches the intuition that a point far from the mean is more "surprising."

**The formula: $s^2 = \frac{1}{n-1}\sum_{i=1}^n(x_i - \bar{x})^2$. The $n-1$ denominator (Bessel's correction) exists because $\bar{x}$ is itself estimated from the same sample, so the residuals $(x_i - \bar{x})$ are artificially compressed?** #flashcard
they can't span the full range they would around the true $\mu$. Using $n-1$ corrects for this bias.

**What breaks?** #flashcard
variance is in squared units. If your data is in meters, variance is in meters². That makes it uninterpretable on the original scale.

**Why it exists?** #flashcard
variance is uninterpretable because of squared units. You need spread expressed in the same units as the data.

**Core insight?** #flashcard
take the square root of variance to undo the squaring.

**The formula: $s = \sqrt{s^2}$. The shape follows directly?** #flashcard
it's just the root of variance.

**What breaks?** #flashcard
std is still sensitive to outliers because it inherits variance's squared deviations. One large outlier inflates $s$ substantially.

**Why it exists?** #flashcard
std is sensitive to outliers. You need a measure of spread that, like the median, ignores the extremes. The IQR measures the width of the "middle half" of your data.

**Core insight?** #flashcard
find the 25th and 75th percentiles (Q1 and Q3), subtract them. Whatever is happening in the tails is irrelevant.

**The formula?** #flashcard
IQR = Q3 − Q1. The shape is a difference of positional statistics, so it inherits robustness from percentile-based summaries.

**What breaks: IQR ignores the tails entirely, so it cannot distinguish a distribution with light tails from one with heavy tails?** #flashcard
both can have the same IQR.

**Why they exist?** #flashcard
a single summary number (mean, median) loses all information about the distribution's shape. You might need to know: "what is the worst-case for 95% of users?" or "where does the top decile start?" Percentiles give you the distribution's shape at any granularity you want.

**Core insight: the $p$-th percentile is the value below which $p$% of observations fall. It's a positional statistic?** #flashcard
sort the data, find the position.

**What breaks?** #flashcard
with small $n$, percentiles are noisy estimates. The 99th percentile of a 50-point sample is just one or two data points and can swing wildly with new data.

**Why it exists: you want to know whether the data is symmetric or "lopsided." Without this, you might apply a Normal-distribution analysis to data that is severely one-sided?** #flashcard
all your p-values and confidence intervals will be wrong.

**Core insight?** #flashcard
measure the average cubed deviation from the mean, normalized by $s^3$. Cubing preserves sign (unlike squaring), so asymmetry shows up as a nonzero result. Positive cubing means long right tail dominates; negative means long left tail.

**The formula (Fisher's)?** #flashcard
$g_1 = \frac{n}{(n-1)(n-2)} \sum\left(\frac{x_i - \bar{x}}{s}\right)^3$. The correction factor adjusts for small-sample bias.

**Positive (right) skew?** #flashcard
long right tail, mean > median

**Negative (left) skew?** #flashcard
long left tail, mean < median

**What breaks?** #flashcard
skewness is sensitive to outliers because it involves cubed deviations. A single extreme value can dominate the entire statistic.

**Why it exists: you have two distributions with the same mean, std, and skewness. One has a sharp peak with heavy tails (extreme values happen more often than Normal), the other is flat. Kurtosis measures this tail-heaviness. In practice: financial returns are leptokurtic?** #flashcard
the Normal distribution drastically underestimates the probability of crashes.

**Core insight?** #flashcard
measure the average fourth-power deviation, normalized by $s^4$. The fourth power heavily amplifies tail observations, so the statistic is dominated by them. Subtracting 3 centers it at zero for a Normal distribution (excess kurtosis).

**The formula (excess kurtosis)?** #flashcard
$g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum\left(\frac{x_i-\bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$

**Excess kurtosis = 0?** #flashcard
Normal distribution

**Leptokurtic (> 0): heavy tails, sharp peak?** #flashcard
e.g., financial returns

**Platykurtic (< 0)?** #flashcard
thin tails, flat peak

**What breaks: kurtosis is extremely sensitive to outliers?** #flashcard
the fourth power means a single extreme observation dominates. With small samples it is nearly meaningless.

**Why it exists?** #flashcard
you are averaging many independent random contributions. The Central Limit Theorem (see Section 9) guarantees the sum converges to a specific bell-shaped distribution regardless of the source distribution's shape. The Normal is the attractor that all well-behaved averages converge to.

**Core insight: it is the maximum-entropy distribution given a fixed mean and variance?** #flashcard
i.e., if all you know about a quantity is its mean and spread, Normal is the least-informative (most honest) assumption.

**The formula?** #flashcard
$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$. The shape: the exponent is a squared Mahalanobis distance from $\mu$, so probability decays as a Gaussian bell curve. The normalizing constant $\frac{1}{\sigma\sqrt{2\pi}}$ exists to make the integral equal 1.

**68-95-99.7 rule: 68% of mass within ±1σ, 95% within ±2σ, 99.7% within ±3σ. These are consequences of integrating the Gaussian PDF over those intervals?** #flashcard
useful for quick mental arithmetic.

**What breaks?** #flashcard
real data has tails that are heavier than Normal. Modeling financial returns or rare events as Normal drastically underestimates the probability of extreme outcomes. Normality must be verified, not assumed.

**Why it exists?** #flashcard
you flip a biased coin $n$ times and want to know how many heads. More concretely: you run an experiment on $n$ users with click-through probability $p$. How probable is each possible number of successes?

**Core insight?** #flashcard
each trial is independent and has exactly two outcomes. The number of ways to arrange $k$ successes among $n$ trials is $\binom{n}{k}$. Each specific arrangement has probability $p^k(1-p)^{n-k}$.

**The formula?** #flashcard
$P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$. Mean: $np$ (expected successes). Variance: $np(1-p)$ (uncertainty is highest when $p=0.5$).

**What breaks: the independence assumption. If trials are correlated (one user's click influences another's), the Binomial is wrong. Also: when $n$ is large and $p$ is small, the Binomial becomes computationally awkward?** #flashcard
use Poisson instead ($\lambda = np$).

**Why it exists: you want to model the count of events that happen rarely and independently in a fixed time window?** #flashcard
server errors per hour, calls to a call center per minute. The Binomial requires knowing $n$ and $p$ separately, but you often only know the rate $\lambda = np$.

**Core insight?** #flashcard
take Binomial as $n \to \infty$ and $p \to 0$ with $\lambda = np$ fixed. The Binomial formula converges to the Poisson formula in this limit.

**The formula: $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$. The shape: $\lambda^k / k!$ counts "how many ways to get $k$ events at rate $\lambda$," and $e^{-\lambda}$ is the normalization. Mean = Variance = $\lambda$?** #flashcard
a signature property that lets you detect overdispersion (when variance exceeds mean) as a model violation.

**What breaks?** #flashcard
the equal mean-variance constraint. Real count data is often overdispersed (variance > mean), e.g., web traffic with bursty behavior. Use Negative Binomial instead. Also assumes events are independent within the interval.

**Why it exists: you want to model a probability itself as a random variable?** #flashcard
"what is the true click-through rate of this ad?" A probability must lie in $[0,1]$. Normal and Gamma distributions don't respect that constraint. Beta is the natural distribution over $[0,1]$.

**Core insight?** #flashcard
if you've observed $\alpha - 1$ successes and $\beta - 1$ failures, the Beta distribution is the posterior for the success probability under a uniform prior. $\alpha$ and $\beta$ act like pseudo-counts.

**The formula: PDF $\propto x^{\alpha-1}(1-x)^{\beta-1}$ on $[0,1]$. Mean: $\frac{\alpha}{\alpha+\beta}$. Mode: $\frac{\alpha-1}{\alpha+\beta-2}$ (for $\alpha, \beta > 1$). $\text{Beta}(1,1)$ = Uniform?** #flashcard
no prior information.

**Conjugate prior property?** #flashcard
if prior is $\text{Beta}(\alpha, \beta)$ and you observe $k$ successes in $n$ trials, posterior is $\text{Beta}(\alpha+k, \beta+n-k)$. This makes Bayesian updating closed-form.

**What breaks?** #flashcard
Beta is only for data confined to $[0,1]$. For general proportions outside that range, it doesn't apply. Also, $\alpha < 1$ or $\beta < 1$ creates U-shaped or J-shaped distributions, which may be appropriate but are counterintuitive.

**Why it exists?** #flashcard
you want to model waiting times or positive continuous quantities. How long until the $\alpha$-th event in a Poisson process? How do you put a prior on a rate parameter $\lambda > 0$? The Gamma distribution is the answer in both cases.

**Core insight: waiting time until the first event in a Poisson process is Exponential. Waiting time until the $\alpha$-th event is the sum of $\alpha$ independent Exponentials?** #flashcard
and sums of i.i.d. Exponentials follow a Gamma distribution.

**The formula?** #flashcard
Mean: $\alpha/\beta$, Variance: $\alpha/\beta^2$. Special cases: Exponential ($\alpha=1$), Chi-squared ($\alpha=k/2$, $\beta=1/2$).

**What breaks?** #flashcard
Gamma assumes events arrive independently at a constant rate. Real processes often have time-varying rates (non-stationarity), making the Poisson/Gamma framework inappropriate.

**Why it exists: you want to do inference on a mean, but you don't know the population standard deviation and your sample is small. If you use the Normal distribution and plug in the sample std, you underestimate uncertainty?** #flashcard
the resulting confidence intervals are too narrow and tests reject too often. The t-distribution corrects for this by having heavier tails.

**Core insight: when you estimate $\sigma$ from data, you introduce additional uncertainty. The ratio $(\bar{X} - \mu) / (s/\sqrt{n})$ doesn't follow a Normal?** #flashcard
it follows a t-distribution with $n-1$ degrees of freedom. As $n$ grows and $s$ becomes a reliable estimate of $\sigma$, the t-distribution converges to Normal.

**The formula?** #flashcard
degrees of freedom $\nu = n - 1$ for one-sample tests. Heavier tails for small $\nu$; as $\nu \to \infty$, converges to $\mathcal{N}(0,1)$.

**What breaks?** #flashcard
the t-distribution still assumes the underlying data is approximately Normal (or $n$ is large enough for CLT). With very non-Normal data and small $n$, even t-tests can mislead. Use non-parametric alternatives.

**Why it exists?** #flashcard
you want to test how much a set of observed frequencies deviates from expected frequencies, or whether two categorical variables are associated. You need a test statistic that accumulates evidence from all categories simultaneously. Summing squared Normally distributed quantities gives you a chi-squared distribution.

**Core insight?** #flashcard
if $Z_1, \ldots, Z_k$ are independent standard Normals, then $\sum Z_i^2 \sim \chi^2_k$. Standardized residuals $(O-E)/\sqrt{E}$ are approximately Normal, so their squares sum to a chi-squared statistic.

**The formula?** #flashcard
Mean: $k$, Variance: $2k$, where $k$ = degrees of freedom.

**What breaks?** #flashcard
the chi-squared approximation requires expected cell counts $\geq 5$. With sparse cells, the Normal approximation for residuals breaks down and the test becomes unreliable. Use Fisher's exact test for small samples.

**Why it exists?** #flashcard
you want to compare variances across groups or test whether a regression model explains significantly more variance than a null model. You need a distribution for a ratio of variances. The F-distribution is that ratio.

**Core insight?** #flashcard
the ratio of two independent chi-squared statistics, each divided by their degrees of freedom, follows an F-distribution. $F = \frac{\chi^2_{d_1}/d_1}{\chi^2_{d_2}/d_2}$. If the two variances are equal, this ratio is near 1; large F means the numerator variance is disproportionately large.

**What breaks?** #flashcard
F-tests assume Normality of residuals and homoscedasticity. For ANOVA, violations of these assumptions inflate Type I error rates.

**Null hypothesis $H_0$: the boring baseline?** #flashcard
"no effect," "no difference," "status quo is true." Assumed true until evidence against it.

**Alternative hypothesis $H_1$: what you want to detect?** #flashcard
"there is a difference," "the treatment works."

**p-value: the probability of observing a test statistic at least as extreme as yours, assuming $H_0$ is true. This is not the probability that $H_0$ is true?** #flashcard
a common and costly misinterpretation.

**Significance level $\alpha$?** #flashcard
your pre-specified threshold for "surprising enough." If p-value < $\alpha$, reject $H_0$. The threshold controls how often you falsely reject when $H_0$ is true.

**Type I error ($\alpha$): reject $H_0$ when it is true?** #flashcard
false positive. You are willing to tolerate this at rate $\alpha$.

**Type II error ($\beta$): fail to reject $H_0$ when it is false?** #flashcard
false negative. You miss a real effect.

**Power = $1 - \beta$?** #flashcard
probability of detecting a true effect. Power depends on sample size, effect size, and $\alpha$.

**One-tailed vs two-tailed: one-tailed tests a directional hypothesis (e.g., "treatment is better than control"). Two-tailed tests for any difference. Use two-tailed by default?** #flashcard
one-tailed is only valid when direction is pre-specified and theoretically motivated, never post-hoc.

**Why it exists?** #flashcard
you want to test whether a sample mean differs from a known value, and you either know the population standard deviation or have a large enough sample that your sample std is reliable.

**Core insight?** #flashcard
standardize the sample mean under $H_0$. The result follows $\mathcal{N}(0,1)$ (by CLT for large $n$), so you can use Normal tables to find p-values.

**The formula?** #flashcard
$z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}$. The shape: numerator is "how far from the hypothesized mean," denominator is the standard error (how much the sample mean varies by chance).

**What breaks?** #flashcard
requires known $\sigma$ or $n \geq 30$ for CLT to be adequate. With small $n$ and unknown $\sigma$, use a t-test.

**Why it exists?** #flashcard
same problem as z-test, but you don't know $\sigma$ and your sample is small. Plugging in $s$ for $\sigma$ in the z-test underestimates uncertainty. The t-test uses the t-distribution to account for the extra variability introduced by estimating $\sigma$.

**Core insight?** #flashcard
replacing $\sigma$ with $s$ changes the distribution of the test statistic from Normal to t (heavier tails). The degrees of freedom parameter governs how much heavier.

**The formulas:?** #flashcard
The formulas:

**One-sample?** #flashcard
$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$, df = $n-1$

**Two-sample independent?** #flashcard
$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_p^2(1/n_1 + 1/n_2)}}$ (pooled), or Welch's t-test when variances are unequal (use Welch's by default)

**Paired: $t = \frac{\bar{d}}{s_d/\sqrt{n}}$ where $d_i = x_{i,\text{after}} - x_{i,\text{before}}$?** #flashcard
use when each unit appears in both conditions, because pairing removes between-subject noise

**What breaks?** #flashcard
assumes approximate Normality of the data (or $n$ large enough for CLT). With small $n$ and severely non-Normal data, p-values are unreliable. Use Mann-Whitney U instead.

**Why it exists?** #flashcard
you have more than two groups and want to know if any means differ. Running pairwise t-tests inflates Type I error (see Section 4). ANOVA tests all groups simultaneously in one test.

**Core insight?** #flashcard
if all groups have the same mean, then variance between group means should be comparable to variance within groups (just noise). If between-group variance is much larger than within-group variance, some means must differ.

**The formula?** #flashcard
$F = \frac{MS_B}{MS_W} = \frac{SS_B/(k-1)}{SS_W/(N-k)}$, where $k$ = number of groups, $N$ = total observations. Large $F$ is evidence that between-group variance exceeds within-group variance.

**What breaks: ANOVA only tells you some means differ?** #flashcard
not which pairs. Post-hoc tests (Tukey HSD, Bonferroni-corrected pairwise t-tests) are needed to identify which pairs. ANOVA also assumes homoscedasticity (equal variances); use Welch's ANOVA otherwise.

**Why it exists: you have categorical data?** #flashcard
not means, but counts. You want to know if two categorical variables are associated (e.g., "does gender affect product preference?") or if observed frequencies match a theoretical distribution.

**Core insight?** #flashcard
under the null hypothesis of independence, the expected count in each cell is (row total × column total) / grand total. Deviations of observed from expected, squared and normalized, accumulate into a chi-squared statistic.

**The formula?** #flashcard
$\chi^2 = \sum \frac{(O-E)^2}{E}$

**Independence test?** #flashcard
df = $(r-1)(c-1)$ for an $r \times c$ contingency table

**Goodness-of-fit test?** #flashcard
df = $k-1$ for $k$ categories

**What breaks?** #flashcard
expected cell count < 5 makes the chi-squared approximation unreliable. Collapse categories or use Fisher's exact test.

**Why they exist: t-tests and ANOVA assume approximately Normal data (or large $n$). With small samples and clearly non-Normal data?** #flashcard
ordinal scales, skewed distributions, heavy tails — those assumptions fail. Non-parametric tests make no distributional assumptions because they operate on ranks rather than raw values.

**Core insight?** #flashcard
instead of comparing means, ask: if you pick one observation from each group at random, which group's observation is more likely to be larger? Test whether $P(X > Y) = 0.5$.

**The formula?** #flashcard
convert all observations to ranks, then compute the rank-sum statistic. Equivalent to testing whether distributions are identical (shift alternative).

**What breaks: the test is less powerful than a t-test when the t-test's assumptions are actually met. It also does not directly test means?** #flashcard
it tests distributional dominance.

**Core insight?** #flashcard
non-parametric analogue of one-way ANOVA. Rank all observations ignoring group membership, then test whether rank distributions differ across groups.

**What breaks: like ANOVA, it only detects that some groups differ?** #flashcard
post-hoc pairwise Mann-Whitney tests are needed with Bonferroni correction.

**Why it exists: the simplest solution?** #flashcard
divide your desired family-wise error rate by the number of tests to get the per-test threshold.

**Core insight?** #flashcard
by the union bound, $P(\text{any false positive}) \leq \sum_i P(\text{false positive}_i) = m \cdot \alpha'$. Setting $\alpha' = \alpha/m$ guarantees $P(\text{any false positive}) \leq \alpha$.

**The formula?** #flashcard
$\alpha_{\text{adjusted}} = \alpha / m$.

**What breaks: Bonferroni is very conservative when tests are positively correlated (as they often are in practice?** #flashcard
e.g., testing correlated genes). It throws away power unnecessarily, increasing Type II errors. Use when you absolutely need to control FWER and tests are independent.

**Why it exists: Bonferroni controls the probability of any false positive. But when you are testing thousands of genes, getting one or two false discoveries in a list of 50 true discoveries may be acceptable?** #flashcard
you just want to bound what fraction of your discoveries are false. FDR (False Discovery Rate) controls the expected proportion of false discoveries among all rejections.

**Core insight?** #flashcard
sort p-values from smallest to largest. Reject all hypotheses up to the largest $p_{(i)}$ satisfying $p_{(i)} \leq \frac{i}{m} \cdot q^*$. This linearly relaxes the threshold as you move down the sorted list.

**The formula?** #flashcard
sort $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$. Find the largest $i$ such that $p_{(i)} \leq \frac{i}{m} q^$. Reject all $H_{(1)}, \ldots, H_{(i)}$. Target FDR is $q^$.

**What breaks: BH controls expected FDR under independence or positive dependence. For arbitrary dependence, use BHY (Benjamini-Hochberg-Yekutieli). FDR also does not control per-comparison error rates?** #flashcard
individual discoveries may still be false.

**Why this wording matters: a 95% CI does not mean "there is a 95% probability that $\mu$ is in this interval." The true $\mu$ is a fixed unknown?** #flashcard
it either is or isn't in your specific interval. The 95% is a property of the procedure: if you repeated the experiment many times, 95% of the constructed intervals would contain $\mu$.

**The formula (for a mean, Normal case)?** #flashcard
$\bar{x} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}$ where $z_{\alpha/2} = 1.96$ for 95%. The half-width is exactly the margin of error.

**Relation to hypothesis tests: a 95% CI is the set of all $\mu_0$ values you would fail to reject at $\alpha = 0.05$. This duality is exact?** #flashcard
rejecting $H_0: \mu = \mu_0$ iff $\mu_0 \notin \text{95\% CI}$.

**What breaks?** #flashcard
the formula above assumes Normality (or large $n$). For small $n$, use t-distribution critical values instead of z.

**Why it exists?** #flashcard
for complex statistics (median, correlation, ratio of variances, model coefficients), there is no closed-form sampling distribution. You cannot derive a formula. The bootstrap solves this by simulating the sampling process using the data itself.

**Core insight?** #flashcard
treat your observed sample as a proxy for the population. Resample from it with replacement $B$ times (each resample is the same size as the original). Compute your statistic on each resample. The distribution of that statistic across resamples approximates the sampling distribution.

**The formula (percentile method): $[\hat{\theta}_{(\alpha/2)}, \hat{\theta}_{(1-\alpha/2)}]$?** #flashcard
use the $\alpha/2$ and $1-\alpha/2$ quantiles of the bootstrap distribution. BCa (bias-corrected accelerated) bootstrap corrects for both bias and skewness in the bootstrap distribution and is preferred in practice.

**What breaks?** #flashcard
the bootstrap assumes your sample is representative of the population. With extreme outliers or very small $n$, the bootstrap distribution may be a poor approximation. Also computationally intensive for large datasets.

**Prior $P(\theta)$?** #flashcard
your belief about $\theta$ before seeing data. Can encode genuine prior knowledge or be deliberately vague (uninformative prior).

**Likelihood $P(D|\theta)$?** #flashcard
probability of observing your data given a specific value of $\theta$. This is the same function frequentists maximize for MLE.

**Posterior $P(\theta|D) \propto P(D|\theta) \cdot P(\theta)$?** #flashcard
your updated belief after seeing data. Proportional to likelihood × prior. The normalizing constant is $P(D) = \int P(D|\theta)P(\theta)d\theta$.

**Why this formula has this shape?** #flashcard
Bayes' theorem is just the definition of conditional probability rearranged: $P(\theta|D) = P(D|\theta)P(\theta)/P(D)$. The posterior is prior belief weighted by how well $\theta$ explains the data.

**MLE: $\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(D|\theta)$?** #flashcard
maximize the likelihood, ignoring the prior entirely. Equivalent to the mode of the posterior under a uniform prior.

**MAP (Maximum A Posteriori): $\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta|D)$?** #flashcard
maximize the posterior. Uses prior information. Equivalent to MLE with L2 regularization when prior is Gaussian, or L1 regularization when prior is Laplace. This is why regularization has a Bayesian interpretation.

**Posterior mean: the full Bayesian estimate; minimizes expected squared error. Does not reduce the posterior to a point?** #flashcard
you can report the entire distribution.

**Why they exist?** #flashcard
computing the posterior integral $P(D) = \int P(D|\theta)P(\theta)d\theta$ is often intractable. Conjugate priors are chosen so the posterior has the same functional form as the prior, making the update a simple parameter update with no integration.

**What breaks?** #flashcard
conjugate priors are mathematically convenient but may not represent your actual prior beliefs. Using a conjugate prior when your genuine prior is non-conjugate introduces prior misspecification bias.

**Why this distinction matters: "the parameter has a 95% probability of being in this range" sounds like what a confidence interval says?** #flashcard
but it isn't.

**Credible interval (Bayesian)?** #flashcard
directly states $P(\theta \in [a,b] | D) = 0.95$. This is a genuine probability statement about $\theta$, valid because $\theta$ is treated as a random variable with a posterior distribution.

**Confidence interval (frequentist)?** #flashcard
$\theta$ is fixed. The interval is random (it depends on data). 95% CI means: the procedure produces intervals that cover the true $\theta$ in 95% of repeated experiments. No probability statement about any specific interval is valid.

**Practically?** #flashcard
the two are numerically similar when the prior is weak relative to the data. The distinction matters most when $n$ is small and the prior is informative.

**Why it exists?** #flashcard
measures the strength of a linear relationship between two continuous variables.

**Core insight?** #flashcard
compute the average product of standardized deviations. If $x$ and $y$ tend to be on the same side of their means simultaneously, this product is positive on average.

**The formula?** #flashcard
$r = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \sum(y_i-\bar{y})^2}}$. The denominator normalizes so that $r \in [-1, 1]$.

**What breaks: Pearson measures only linear association. Two variables can be perfectly related (e.g., $y = x^2$) and have $r \approx 0$. Also sensitive to outliers?** #flashcard
one extreme point can make a weak relationship look strong or vice versa.

**Why it exists?** #flashcard
Pearson fails for non-linear monotonic relationships and is distorted by outliers. You want a correlation measure that captures "as $x$ increases, does $y$ tend to increase?" without assuming linearity, and without being dominated by extreme values.

**Core insight?** #flashcard
replace raw values with their ranks, then apply Pearson's formula to the ranks. Ranks are bounded and treat the spacing between observations uniformly, making the statistic robust.

**The formula?** #flashcard
Pearson $r$ applied to the ranks of $x$ and $y$. Measures monotonic (not necessarily linear) association.

**What breaks: Spearman discards magnitude information. It cannot distinguish a steep relationship from a shallow one?** #flashcard
only the rank ordering matters.

**Why it exists?** #flashcard
Spearman can be unstable with small samples or many ties. Kendall $\tau$ provides a more interpretable and robust alternative.

**Core insight?** #flashcard
for every pair of observations $(i, j)$, ask whether they are concordant (both $x$ and $y$ rank the same way: $x_i > x_j$ and $y_i > y_j$, or vice versa) or discordant (they rank oppositely). The correlation is just the excess of concordant over discordant pairs.

**The formula?** #flashcard
$\tau = \frac{\text{concordant pairs} - \text{discordant pairs}}{\binom{n}{2}}$. Directly interpretable as $P(\text{concordant}) - P(\text{discordant})$ for a randomly selected pair.

**What breaks?** #flashcard
$O(n^2)$ computation for large $n$ (though $O(n \log n)$ algorithms exist). Like Spearman, it detects association but cannot characterize its functional form.

**Why it exists?** #flashcard
$X$ and $Y$ may appear correlated only because both are driven by a third variable $Z$ (a confounder). You want to measure the direct relationship between $X$ and $Y$ after removing $Z$'s influence.

**Core insight?** #flashcard
regress $X$ on $Z$ and regress $Y$ on $Z$. Take the residuals from both regressions. The partial correlation of $X$ and $Y$ controlling for $Z$ is the Pearson correlation between those residuals.

**What breaks?** #flashcard
partial correlation only removes the linear effect of $Z$. Non-linear confounding remains.

**Why this warning is here?** #flashcard
correlation is seductive. A high $r$ value feels like evidence of causation. It isn't.

**Spurious correlation: both $X$ and $Y$ are caused by a common variable $Z$. Example: ice cream sales correlate with drowning deaths?** #flashcard
both are caused by summer heat.

**Reverse causation: $Y$ causes $X$, not $X$ causes $Y$. Example: hospitals are correlated with sickness?** #flashcard
hospitals don't cause sickness.

**Coincidental correlation?** #flashcard
no causal connection whatsoever. Example: per-capita cheese consumption correlates with people dying tangled in bedsheets.

**Core principle?** #flashcard
to establish causation you need either a randomized experiment (which eliminates confounding by design) or a valid quasi-experimental design with strong assumptions.

**Linearity?** #flashcard
$E[Y|X] = X\beta$. If the true relationship is non-linear and you fit a line, your coefficients are biased estimates of a misspecified quantity. Detected with residual-vs-fitted plots.

**Independence?** #flashcard
residuals $\epsilon_i$ are independent across observations. Violated by time series (autocorrelation) or grouped data (clustering). Produces standard errors that are too small, inflating t-statistics.

**Normality: residuals are approximately Normal?** #flashcard
needed for exact finite-sample inference. With large $n$, CLT makes this less critical for the distribution of estimates.

**Equal variance (homoscedasticity): $\text{Var}(\epsilon_i) = \sigma^2$ constant across $X$. Violated when residuals fan out at higher fitted values (heteroscedasticity). Makes standard errors wrong?** #flashcard
use HC (heteroscedasticity-consistent) robust standard errors.

**Why it exists?** #flashcard
you want to know how much of the outcome's variance your model explains. An absolute measure of fit.

**Core insight?** #flashcard
compare your model's residual variance to the variance you'd have using just the mean: $R^2 = 1 - SS_{\text{res}}/SS_{\text{tot}}$.

**What breaks?** #flashcard
$R^2$ never decreases when you add more predictors, even useless random noise ones. A model with 100 random predictors will have high $R^2$ with enough data. Use Adjusted $R^2$ for model comparison.

**Why it exists?** #flashcard
$R^2$ rewards complexity. You need a version that penalizes for additional predictors.

**The formula?** #flashcard
$\bar{R}^2 = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$, where $p$ = number of predictors. Adding a predictor only improves $\bar{R}^2$ if it explains more variance than it costs in degrees of freedom.

**Why it exists: after fitting a model with $p$ predictors, you want to test whether any of them are useful?** #flashcard
i.e., whether the model is better than just predicting the mean.

**The formula?** #flashcard
$F = \frac{R^2/p}{(1-R^2)/(n-p-1)}$. Under $H_0: \beta_1 = \ldots = \beta_p = 0$, this follows $F_{p, n-p-1}$.

**Why it exists?** #flashcard
the F-test is global. You want to know whether each individual predictor contributes, given all other predictors are already in the model.

**The formula: $t = \hat{\beta}_j / \text{SE}(\hat{\beta}_j)$, df = $n-p-1$. The SE depends on all other predictors?** #flashcard
this is why multicollinearity inflates SEs.

**The problem: two predictors $X_1$ and $X_2$ are highly correlated. When you include both in a regression, the model cannot attribute credit independently?** #flashcard
it can trade off any value of $\hat{\beta}_1$ against a compensating value of $\hat{\beta}_2$ and still fit the data equally well. Coefficient estimates become wildly unstable.

**Core insight?** #flashcard
measure how much of $X_j$'s variance is explained by the other predictors. If $X_j$ is nearly a linear combination of the others, its coefficient is poorly identified.

**The formula?** #flashcard
VIF (Variance Inflation Factor) $= \frac{1}{1 - R^2_j}$, where $R^2_j$ is the $R^2$ from regressing $X_j$ on all other predictors. VIF = 5 means the SE of $\hat{\beta}_j$ is $\sqrt{5} \approx 2.2\times$ larger than it would be without collinearity. VIF > 10: severe.

**What breaks?** #flashcard
high VIF makes individual coefficients uninterpretable, but it doesn't necessarily hurt prediction accuracy. If your goal is prediction (not inference on coefficients), multicollinearity is less of a problem.

**Remedies?** #flashcard
drop one of the correlated features, combine them (PCA), or use ridge regression which explicitly penalizes large coefficients and is robust to collinearity.

**The question?** #flashcard
does the sample mean actually converge to the true mean as $n$ grows? Without this guarantee, statistical estimation has no foundation.

**Core insight?** #flashcard
averaging is a noise-reduction process. Each observation is the true mean plus noise. When you average $n$ observations, the noise from different observations is independent and partially cancels. As $n \to \infty$, the noise cancels completely and you are left with the signal.

**The formulas:?** #flashcard
The formulas:

**Weak LLN: $\bar{X}_n \xrightarrow{p} \mu$ (convergence in probability)?** #flashcard
for any $\epsilon > 0$, $P(|\bar{X}_n - \mu| > \epsilon) \to 0$

**Strong LLN: $\bar{X}_n \to \mu$ almost surely?** #flashcard
the sample paths converge, not just the probabilities

**Why both versions exist?** #flashcard
weak LLN allows occasional bad samples; strong LLN says bad samples become increasingly rare. For most practical purposes, weak LLN suffices.

**Foundation for Monte Carlo?** #flashcard
the LLN guarantees that averaging many random samples from a distribution converges to the expectation. This is why Monte Carlo simulation works for integration.

**What breaks: LLN requires finite mean. For heavy-tailed distributions like Cauchy (which has undefined mean), the sample mean does not converge?** #flashcard
it wanders without bound. The LLN also says nothing about how fast convergence happens — that's the CLT's job.

**The question?** #flashcard
the LLN says the sample mean converges to $\mu$. But how fast? And what does the sampling distribution of $\bar{X}$ look like for finite $n$? Without knowing the shape of the sampling distribution, you cannot build confidence intervals or run hypothesis tests.

**Core insight?** #flashcard
regardless of the population distribution's shape (as long as it has finite mean and variance), the distribution of the standardized sample mean converges to a standard Normal. The Normal distribution is the universal limit for averages of i.i.d. random variables.

**The formula?** #flashcard
for i.i.d. samples with finite mean $\mu$ and variance $\sigma^2$: $\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$, equivalently $\bar{X}_n \approx \mathcal{N}(\mu, \sigma^2/n)$ for large $n$.

**Why this formula has this shape?** #flashcard
the $\sqrt{n}$ scaling is not arbitrary. Averaging $n$ observations reduces variance by a factor of $n$ (variances add for independent variables, so variance of $\bar{X}$ is $\sigma^2/n$). The standard error $\sigma/\sqrt{n}$ is the square root of that.

**Practical consequence?** #flashcard
this is why t-tests, z-tests, and Normal-based confidence intervals are valid for large samples even when the underlying data is not Normal. The CLT licenses the Normality assumption for sample means.

**What breaks:?** #flashcard
What breaks:

**Heavy-tailed distributions?** #flashcard
if variance is infinite (Cauchy distribution, Pareto with tail index $\leq 2$), CLT does not apply. The distribution of the sample mean does not converge to Normal.

**Small $n$?** #flashcard
"large enough" depends on the skewness of the distribution. Roughly: $n \geq 30$ for mildly skewed, $n \geq 100$ for heavily skewed, $n \geq 1000$ for distributions with extreme kurtosis.

**Dependent observations: CLT requires independence (or weak dependence). Time series data with strong autocorrelation violates this?** #flashcard
specialized CLTs for dependent processes exist but have stricter conditions.

**Rate of convergence?** #flashcard
CLT says the limiting distribution is Normal but says nothing about how closely the finite-sample distribution matches it. The Berry-Esseen theorem quantifies this rate: the approximation error is $O(1/\sqrt{n})$.
