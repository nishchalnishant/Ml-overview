---
module: Interview Prep
topic: Ml
subtopic: Probability And Statistics
status: unread
tags: [interviewprep, ml, ml-probability-and-statistics]
---
# Probability and Statistics

**Primary reference:** [Probability and Statistics deep dive](../../02-ml-fundamentals/math/probability-statistics.md)

---

## What This File Is For

Every topic is structured around the four questions that matter in an interview:
1. What the interviewer is actually testing
2. The reasoning structure — why first-principles thinkers approach it this way
3. The pattern in action — a worked example
4. Common traps — where people go wrong and why

---

## 1. Why Probability and Statistics Matter in ML

**What the interviewer is testing:** Whether you understand that every ML output is a claim about the world made under uncertainty, and whether you can reason about that uncertainty rigorously rather than treating model outputs as ground truth.

**The reasoning structure:** A model trained on finite data makes predictions that generalize imperfectly. Labels are noisy. Samples are biased. Metrics fluctuate across evaluation splits. Probability is the language for quantifying these uncertainties precisely. Statistics is the framework for making decisions that are warranted by evidence rather than noise.

Without probability, you cannot distinguish a real improvement from random variation. You cannot choose the right loss function, understand why regularization works, or reason about why a model fails. Every fundamental component of ML — loss functions, priors, uncertainty estimation, A/B testing — is an application of probability.

**The pattern in action:** A model's test accuracy improves from 82.3% to 83.1% after a change. Should you ship it? Without statistical reasoning, the answer is "it went up, ship it." With statistical reasoning, you ask whether the difference is larger than what random variation in the test set would produce. Running a paired significance test on the per-example predictions gives you a p-value. If n=10,000, that 0.8pp difference is likely significant. If n=500, it might not be. The decision requires probability, not just comparison.

**Common traps:** Treating metrics as exact quantities rather than estimates with sampling variance. Every accuracy number computed on a finite test set has a confidence interval. Not accounting for this leads to false conclusions about model improvements — shipping a change that appears to improve accuracy when the improvement is within sampling noise.

---

## 2. Common Distributions

**What the interviewer is testing:** Whether you can identify which distribution governs a given phenomenon and explain the structural reason why — not just pattern-match a name to a scenario.

**The reasoning structure:** Distributions arise from specific generative processes. Identifying the right distribution requires identifying the process:
- Is there a fixed number of independent binary trials? Binomial.
- Is there an interval during which independent rare events can occur at a constant rate? Poisson.
- Are you measuring time between independent Poisson events? Exponential.
- Is the quantity the sum of many independent small effects? Normal (by CLT).
- Is the quantity constrained to [0,1] and the result of counting successes with unknown rate? Beta or Binomial.

Knowing which distribution applies lets you derive the likelihood, compute expectations analytically, and choose appropriate tests.

### Bernoulli

Single binary trial with success probability $p$:
$$P(X=k) = p^k (1-p)^{1-k}, \quad k \in \{0,1\}$$
$$\mathbb{E}[X] = p, \quad \text{Var}[X] = p(1-p)$$

ML context: click/no-click, fraud/not-fraud, positive/negative label.

### Binomial

Number of successes in $n$ independent Bernoulli trials:
$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$
$$\mathbb{E}[X] = np, \quad \text{Var}[X] = np(1-p)$$

Approximates Normal when $np \geq 5$ and $n(1-p) \geq 5$.

ML context: number of correct predictions in $n$ test examples; number of users who click in a batch.

### Poisson

Count of events in a fixed interval when events are independent and occur at constant rate $\lambda$:
$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$
$$\mathbb{E}[X] = \lambda, \quad \text{Var}[X] = \lambda$$

Variance equals mean — a useful diagnostic: if you observe a count distribution with variance much larger than its mean, Poisson does not fit and a negative binomial might.

ML context: number of requests per second to a serving endpoint; rare disease occurrences in a population.

### Exponential

Time between consecutive Poisson events:
$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$
$$\mathbb{E}[X] = 1/\lambda, \quad \text{Var}[X] = 1/\lambda^2$$

Memoryless property: $P(X > s+t \mid X > s) = P(X > t)$. The distribution forgets how long it has been waiting — the probability of waiting $t$ more time is independent of how long you have already waited.

ML context: time until a user churns; time between failures in a deployed system; inference latency (when dominated by rare long-tail events, this is not appropriate — use log-normal instead).

### Normal

Continuous bell curve arising from sums of many independent contributions:
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
$$\mathbb{E}[X] = \mu, \quad \text{Var}[X] = \sigma^2$$

68-95-99.7 rule: $P(\mu - k\sigma < X < \mu + k\sigma)$ = 68%, 95%, 99.7% for $k$ = 1, 2, 3.

ML context: weight initialization (He, Xavier), noise models, sampling distributions of estimates via CLT.

**The pattern in action:** A model predicts delivery times. The delivery time for a single package is not normally distributed (right-skewed by traffic incidents). But the mean delivery time across 200 packages on a given day is approximately normal by CLT. These require different treatments: individual prediction uses a log-normal model; the batch mean uses normal-based confidence intervals. Applying the same model to both is wrong.

**Common traps:**
- Assuming everything is normal because it is mathematically convenient. Check skewness, kurtosis, and whether the normal approximation conditions hold.
- Confusing Binomial and Poisson. Binomial requires a fixed number of trials $n$. If there is no natural $n$ (counting arrivals in a time window), Poisson is correct.

---

## 3. Law of Large Numbers vs Central Limit Theorem

**What the interviewer is testing:** Whether you can distinguish two fundamental convergence results that are routinely conflated — and explain what each one says and does not say.

**The reasoning structure:** These are different claims about what happens as sample size $n$ grows:

**Law of Large Numbers (LLN):** The sample mean converges to the population mean.
$$\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{p} \mu \quad \text{as } n \to \infty$$
LLN says the estimate gets closer to the truth as $n$ grows. It says nothing about the shape of the distribution of that estimate.

**Central Limit Theorem (CLT):** The standardized sample mean converges in distribution to a standard normal, regardless of the original distribution's shape.
$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty$$
CLT says the sampling distribution of the mean becomes bell-shaped. It enables you to construct confidence intervals and conduct hypothesis tests even when the underlying data is not normal. The rule of thumb is $n \geq 30$ for the approximation to be adequate, though this depends on the skewness of the underlying distribution.

The two theorems together justify standard statistical inference: LLN ensures the sample mean is a consistent estimator; CLT enables computing how uncertain that estimate is.

**The pattern in action:** You measure the average inference latency of a model across $n$ requests. Each individual latency is exponentially distributed (right-skewed, most requests fast with occasional slow outliers). LLN tells you the sample mean latency will converge to the true mean as you collect more requests. CLT tells you that sample means from many batches of 50 requests will be approximately normally distributed — which lets you construct a confidence interval for mean latency even though the underlying distribution is not normal.

**Common traps:**
- Saying "CLT says the data becomes normal." CLT says the sampling distribution of the mean becomes normal, not the raw data. Applying normal-based tests to raw skewed data is wrong; applying them to sample means of large batches is justified.
- Forgetting CLT requires independence. For time-series data where observations are correlated, CLT does not apply directly.

---

## 4. Hypothesis Testing

**What the interviewer is testing:** Whether you can set up a test correctly, interpret results precisely, and explain what statistical significance does and does not mean.

**The reasoning structure:** Hypothesis testing is a framework for making binary decisions under uncertainty while controlling the rate of errors. You specify:
- $H_0$: the null hypothesis (e.g., "the change has no effect")
- $H_1$: the alternative hypothesis (e.g., "the change improves the metric")
- $\alpha$: the acceptable rate of rejecting $H_0$ when it is actually true (Type I error rate)

You then compute a test statistic and assess whether the data is sufficiently inconsistent with $H_0$.

### Z-test (known population variance or large $n$)

$$z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$$

Reject $H_0$ if $|z| > z_{\alpha/2}$ (e.g., $z_{0.025} = 1.96$ for $\alpha = 0.05$ two-tailed).

### Two-sample Z-test for proportions (A/B testing)

$$z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}$$

where $\hat{p} = (x_1 + x_2)/(n_1 + n_2)$ is the pooled proportion.

### T-test (unknown variance, any $n$)

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}, \quad \text{degrees of freedom} = n - 1$$

Use when population standard deviation $\sigma$ is unknown — which is almost always in practice. For large $n$, the t-distribution approaches normal.

### P-value

The p-value is the probability of observing a test statistic at least as extreme as the one computed, **assuming $H_0$ is true**. It is not the probability that $H_0$ is true. It is not the probability the effect is real.

A p-value of 0.03 means: if there were truly no effect, you would see a result this extreme or more extreme 3% of the time. It does not mean "there is a 97% chance the model is better."

**The pattern in action:** You run an A/B test on a product recommendation model. Control: 100,000 users, CTR = 4.2%. Treatment: 100,000 users, CTR = 4.5%. Pooled proportion $\hat{p} = 0.0435$. Test statistic $z \approx 4.5$, $p < 0.001$. Statistically significant — but is it practically significant? A 0.3pp improvement in CTR at this scale corresponds to roughly 300 additional clicks per 100k users. Whether this justifies the engineering and maintenance cost of a new model is a business question, not a statistics question.

**Common traps:**
- Stopping the test early when significance is reached. Running repeated tests inflates Type I error — you will find spurious significance by chance. Pre-register the sample size and stop only when it is reached.
- Reporting p-values without effect sizes. A statistically significant result with a tiny effect is often practically useless.
- Multiple comparisons. Testing 20 metrics at $\alpha = 0.05$ means approximately 1 will show spurious significance by chance. Apply Bonferroni correction (divide $\alpha$ by number of tests) or control the false discovery rate (Benjamini-Hochberg).

---

## 5. Type I and Type II Errors

**What the interviewer is testing:** Whether you can reason about the cost asymmetry between error types and set decision thresholds accordingly — not just recite definitions.

**The reasoning structure:** Every binary classifier or hypothesis test makes two types of mistakes:
- **Type I error (false positive, $\alpha$):** Reject $H_0$ when it is true. You claimed an effect exists when it does not.
- **Type II error (false negative, $\beta$):** Fail to reject $H_0$ when it is false. A real effect went undetected.

The critical point: you cannot minimize both simultaneously with a fixed sample size. Lowering $\alpha$ (requiring stronger evidence to reject) increases $\beta$ (missing real effects). The right tradeoff depends on the relative cost of each error type in the specific application.

Statistical power is $1 - \beta$: the probability of detecting a real effect.

Required sample size per group (two proportions, equal group size):
$$n = \frac{2(z_{\alpha/2} + z_\beta)^2 \cdot p(1-p)}{(\Delta p)^2}$$

where $\Delta p$ is the minimum detectable effect (MDE), $z_{\alpha/2} = 1.96$ ($\alpha = 0.05$), $z_\beta = 0.84$ (80% power). Smaller MDE and smaller $\alpha$ both require larger samples.

**The pattern in action:** A medical screening test for a serious disease: a missed diagnosis (Type II) is catastrophic because the patient goes untreated. A false alarm (Type I) leads to unnecessary follow-up testing — costly but not dangerous. You set a very low detection threshold, accepting more false positives to minimize false negatives. Conversely, a fraud detection system that freezes accounts on false positives is costly in customer experience — you might require stronger evidence before blocking, accepting some missed fraud to protect innocent users. The same statistical framework, opposite optimization direction.

**Common traps:**
- Treating $\alpha = 0.05$ as sacred. This threshold is a convention, not a law. The appropriate threshold depends on the relative cost of Type I and Type II errors.
- Not computing required sample size before running an experiment. Underpowered experiments fail to detect real effects and waste resources. Power analysis before data collection is mandatory.

---

## 6. Confidence Intervals

**What the interviewer is testing:** Whether you can interpret confidence intervals correctly — which is harder than it sounds — and explain why they are more informative than p-values alone.

**The reasoning structure:** A 95% confidence interval is a procedure: if you repeated the sampling and interval construction process many times, 95% of the resulting intervals would contain the true parameter value. It is not a statement that the true value is in this particular interval with 95% probability — in the frequentist framework, the parameter is fixed and the interval is random.

What confidence intervals communicate that p-values do not:
- The magnitude of the effect, not just whether it crosses a significance threshold
- The uncertainty around the estimate
- Whether the effect is practically significant even if statistically significant

A 95% CI for a difference in CTR of [0.001, 0.005] tells you the improvement is real and small. A CI of [-0.001, 0.011] contains zero — not significant. A CI of [0.008, 0.012] tells you the improvement is real and substantial enough to matter.

For a proportion $\hat{p}$ based on $n$ observations (Wald interval):
$$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

For small $n$ or extreme $\hat{p}$ (close to 0 or 1), use the Wilson interval instead — the Wald interval produces intervals that fall outside [0,1] and has poor coverage properties.

**The pattern in action:** A model achieves 87.4% accuracy on a test set of 500 examples. The 95% CI is approximately $[84.3\%, 90.5\%]$. A competing model achieves 88.1% accuracy with CI $[85.0\%, 91.2\%]$. The overlapping intervals suggest you cannot distinguish the models based on this evaluation alone. Reporting point estimates without CIs would have made Model 2 appear clearly better — an illusion from ignoring sampling variance.

**Common traps:**
- Interpreting overlapping confidence intervals as "the difference is not significant." Overlapping individual CIs do not directly imply the difference is non-significant. You need a CI for the difference specifically, not two individual CIs.
- Computing Wald intervals for small proportions. For CTR of 0.1% with n=1000, the Wald interval will include negative values. Use the Wilson or Clopper-Pearson interval.

---

## 7. Bayes' Theorem

**What the interviewer is testing:** Whether you can apply Bayesian reasoning in context — particularly whether you account for prior probability (base rate) rather than only the likelihood of the evidence.

**The reasoning structure:** Bayes' theorem describes how to update beliefs given evidence:

$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}$$

- $P(H)$: prior — belief about the hypothesis before seeing evidence
- $P(E \mid H)$: likelihood — how probable the evidence is if the hypothesis is true
- $P(H \mid E)$: posterior — updated belief after evidence
- $P(E) = \sum_h P(E \mid H=h) P(H=h)$: marginal likelihood (normalizing constant)

The **base rate fallacy** — the most common Bayes error — is ignoring the prior:

A disease affects 1% of the population. A test has 99% sensitivity (true positive rate) and 99% specificity (true negative rate). You test positive. What is the probability you have the disease?

$$P(\text{disease} \mid +) = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.01 \times 0.99} = \frac{0.0099}{0.0198} = 50\%$$

Despite 99% test accuracy, only 50% posterior probability — because the disease is rare. The prior dominates. Intuition fails here because people focus on the test's accuracy rather than the rarity of the condition.

**ML connections:**
- MAP estimation: $\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid D) = \arg\max_\theta [P(D \mid \theta) P(\theta)]$ — the prior regularizes the estimate
- Naive Bayes classifier: applies Bayes' theorem with conditional independence assumption on features
- Bayesian neural networks: maintain distributions over weights rather than point estimates

**The pattern in action:** A fraud detection system has 95% sensitivity and 99% specificity. Fraud rate is 0.1% of transactions.

$$P(\text{fraud} \mid +) = \frac{0.95 \times 0.001}{0.95 \times 0.001 + 0.01 \times 0.999} \approx \frac{0.00095}{0.01094} \approx 8.7\%$$

91.3% of flagged transactions are false positives. This is not a model failure — it is a mathematical consequence of the base rate. Addressing it requires raising the specificity threshold or changing the investigation workflow, not just improving the model. The base rate imposes a hard ceiling on precision at any given sensitivity.

**Common traps:** Forgetting the base rate in any "given a positive test result" problem. The posterior depends critically on the prior, and the prior is the base rate. An impressively accurate test on a very rare condition will still produce mostly false positives.

---

## 8. MLE vs MAP

**What the interviewer is testing:** Whether you understand that regularization is a Bayesian prior — not an arbitrary penalty — and why this framing changes how you think about model selection and regularization strength.

**The reasoning structure:** Both MLE and MAP find a single parameter estimate, but they optimize different objectives:

**MLE (Maximum Likelihood Estimation):** Find parameters that make the observed data most probable:
$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(D \mid \theta) = \arg\max_\theta \sum_{i=1}^n \log p(x_i \mid \theta)$$

MLE ignores prior knowledge about what reasonable parameter values look like. With small data, MLE overfits because it chases the noise in the training data.

**MAP (Maximum A Posteriori):** Find parameters that maximize the posterior — likelihood weighted by prior:
$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid D) = \arg\max_\theta \left[\sum_i \log p(x_i \mid \theta) + \log P(\theta)\right]$$

The prior $\log P(\theta)$ acts as a regularizer:

| Prior on $\theta$ | MAP objective includes | Equivalent regularization |
| :--- | :--- | :--- |
| Gaussian $\mathcal{N}(0, 1/\lambda)$ | $- \frac{\lambda}{2}\|\theta\|^2$ | L2 / Ridge |
| Laplace with scale $1/\lambda$ | $- \lambda\|\theta\|_1$ | L1 / Lasso |

When you minimize cross-entropy plus L2 regularization, you are doing Bayesian MAP estimation with a Gaussian prior on weights. The regularization strength $\lambda$ encodes how strongly you believe weights should be small. A large $\lambda$ says "I believe the true weights are very close to zero."

As $n \to \infty$, the data overwhelms the prior and MLE $\approx$ MAP. With small $n$, the prior dominates and prevents overfitting.

**The pattern in action:** Train logistic regression on 50 examples with 100 features. Without regularization (MLE): 100% training accuracy by fitting noise — many features get extreme weights that happen to separate the training examples. With L2 regularization (MAP with Gaussian prior): weights are shrunk toward zero and the model generalizes better because the prior encodes the assumption that true effects should be moderate, not extreme. The hyperparameter $\lambda$ controls how much the prior dominates over the data.

**Common traps:**
- Saying "L2 regularization prevents overfitting" without the mechanism. The mechanism is that it corresponds to a Gaussian prior that penalizes extreme parameter values — it is Bayesian MAP, not an arbitrary constraint.
- Not connecting regularization strength to the data-to-feature ratio. When you have many more features than examples, a stronger prior (larger $\lambda$) is appropriate; when you have many more examples than features, the prior matters less.

---

## 9. Bootstrap

**What the interviewer is testing:** Whether you understand resampling as a non-parametric approach to quantifying uncertainty when closed-form sampling distributions are unavailable.

**The reasoning structure:** Standard confidence intervals assume you know the sampling distribution of your statistic — for the mean, CLT tells you it is approximately normal. For complex statistics (medians, AUC, custom ML metrics, correlation coefficients), the sampling distribution has no closed form.

Bootstrap approximates the sampling distribution empirically:
1. Draw $B$ bootstrap samples, each of size $n$, by sampling with replacement from your observed data
2. Compute the statistic on each bootstrap sample
3. The distribution of bootstrap statistics approximates the sampling distribution

The 95% bootstrap percentile CI is the 2.5th and 97.5th percentiles of the $B$ bootstrap statistic values. The BCa (bias-corrected and accelerated) bootstrap gives better coverage for skewed statistics.

The key insight: sampling with replacement from the observed data approximates the act of drawing a new sample from the population. Each bootstrap sample is like a plausible alternative dataset you could have observed.

**The pattern in action:** You want a confidence interval for AUC on a test set of 500 examples. There is no closed-form formula for the standard error of AUC. Generate 1,000 bootstrap samples of 500 examples each (with replacement). Compute AUC on each. The 2.5th and 97.5th percentiles of the 1,000 AUC values give a 95% CI. This CI accounts for sampling variability in the test set without distributional assumptions.

**Common traps:**
- Treating bootstrap as a way to generate more training data. It does not add information — it quantifies uncertainty about a statistic computed from existing data.
- Bootstrap does not work well with very small samples ($n < 30$) because the empirical distribution is a poor proxy for the population. The bootstrap samples a distribution that already has limited diversity.
- Not using enough bootstrap iterations. $B = 100$ is too few for precise CI estimation; $B = 1000$ is a safe minimum; $B = 10000$ is better for tail percentiles.

---

## Quick Diagnostics

**If asked to interpret a p-value:**
State what it is (probability of observing a result this extreme or more extreme under $H_0$) and what it is not (probability $H_0$ is true; probability the effect is real). Then add effect size and confidence interval before concluding.

**If asked about A/B test design:**
Anchor on sample size calculation from minimum detectable effect, $\alpha$, and desired power before running the test. Mention that stopping early inflates Type I error. Mention multiple comparisons if testing more than one metric.

**If asked about a model showing high accuracy on a rare-event detection task:**
Immediately ask about class imbalance. A model predicting "not fraud" 100% of the time achieves 99.9% accuracy when fraud rate is 0.1%. Precision, recall, and F1 are the right metrics; accuracy is not. Then invoke Bayes' theorem to reason about the positive predictive value given the base rate.

## Flashcards

**Is there a fixed number of independent binary trials? Binomial.?** #flashcard
Is there a fixed number of independent binary trials? Binomial.

**Is there an interval during which independent rare events can occur at a constant rate? Poisson.?** #flashcard
Is there an interval during which independent rare events can occur at a constant rate? Poisson.

**Are you measuring time between independent Poisson events? Exponential.?** #flashcard
Are you measuring time between independent Poisson events? Exponential.

**Is the quantity the sum of many independent small effects? Normal (by CLT).?** #flashcard
Is the quantity the sum of many independent small effects? Normal (by CLT).

**Is the quantity constrained to [0,1] and the result of counting successes with unknown rate? Beta or Binomial.?** #flashcard
Is the quantity constrained to [0,1] and the result of counting successes with unknown rate? Beta or Binomial.

**Assuming everything is normal because it is mathematically convenient. Check skewness, kurtosis, and whether the normal approximation conditions hold.?** #flashcard
Assuming everything is normal because it is mathematically convenient. Check skewness, kurtosis, and whether the normal approximation conditions hold.

**Confusing Binomial and Poisson. Binomial requires a fixed number of trials $n$. If there is no natural $n$ (counting arrivals in a time window), Poisson is correct.?** #flashcard
Confusing Binomial and Poisson. Binomial requires a fixed number of trials $n$. If there is no natural $n$ (counting arrivals in a time window), Poisson is correct.

**Saying "CLT says the data becomes normal." CLT says the sampling distribution of the mean becomes normal, not the raw data. Applying normal-based tests to raw skewed data is wrong; applying them to sample means of large batches is justified.?** #flashcard
Saying "CLT says the data becomes normal." CLT says the sampling distribution of the mean becomes normal, not the raw data. Applying normal-based tests to raw skewed data is wrong; applying them to sample means of large batches is justified.

**Forgetting CLT requires independence. For time-series data where observations are correlated, CLT does not apply directly.?** #flashcard
Forgetting CLT requires independence. For time-series data where observations are correlated, CLT does not apply directly.

**$H_0$?** #flashcard
the null hypothesis (e.g., "the change has no effect")

**$H_1$?** #flashcard
the alternative hypothesis (e.g., "the change improves the metric")

**$\alpha$?** #flashcard
the acceptable rate of rejecting $H_0$ when it is actually true (Type I error rate)

**Stopping the test early when significance is reached. Running repeated tests inflates Type I error?** #flashcard
you will find spurious significance by chance. Pre-register the sample size and stop only when it is reached.

**Reporting p-values without effect sizes. A statistically significant result with a tiny effect is often practically useless.?** #flashcard
Reporting p-values without effect sizes. A statistically significant result with a tiny effect is often practically useless.

**Multiple comparisons. Testing 20 metrics at $\alpha = 0.05$ means approximately 1 will show spurious significance by chance. Apply Bonferroni correction (divide $\alpha$ by number of tests) or control the false discovery rate (Benjamini-Hochberg).?** #flashcard
Multiple comparisons. Testing 20 metrics at $\alpha = 0.05$ means approximately 1 will show spurious significance by chance. Apply Bonferroni correction (divide $\alpha$ by number of tests) or control the false discovery rate (Benjamini-Hochberg).

**Type I error (false positive, $\alpha$)?** #flashcard
Reject $H_0$ when it is true. You claimed an effect exists when it does not.

**Type II error (false negative, $\beta$)?** #flashcard
Fail to reject $H_0$ when it is false. A real effect went undetected.

**Treating $\alpha = 0.05$ as sacred. This threshold is a convention, not a law. The appropriate threshold depends on the relative cost of Type I and Type II errors.?** #flashcard
Treating $\alpha = 0.05$ as sacred. This threshold is a convention, not a law. The appropriate threshold depends on the relative cost of Type I and Type II errors.

**Not computing required sample size before running an experiment. Underpowered experiments fail to detect real effects and waste resources. Power analysis before data collection is mandatory.?** #flashcard
Not computing required sample size before running an experiment. Underpowered experiments fail to detect real effects and waste resources. Power analysis before data collection is mandatory.

**The magnitude of the effect, not just whether it crosses a significance threshold?** #flashcard
The magnitude of the effect, not just whether it crosses a significance threshold

**The uncertainty around the estimate?** #flashcard
The uncertainty around the estimate

**Whether the effect is practically significant even if statistically significant?** #flashcard
Whether the effect is practically significant even if statistically significant

**Interpreting overlapping confidence intervals as "the difference is not significant." Overlapping individual CIs do not directly imply the difference is non-significant. You need a CI for the difference specifically, not two individual CIs.?** #flashcard
Interpreting overlapping confidence intervals as "the difference is not significant." Overlapping individual CIs do not directly imply the difference is non-significant. You need a CI for the difference specifically, not two individual CIs.

**Computing Wald intervals for small proportions. For CTR of 0.1% with n=1000, the Wald interval will include negative values. Use the Wilson or Clopper-Pearson interval.?** #flashcard
Computing Wald intervals for small proportions. For CTR of 0.1% with n=1000, the Wald interval will include negative values. Use the Wilson or Clopper-Pearson interval.

**$P(H)$: prior?** #flashcard
belief about the hypothesis before seeing evidence

**$P(E \mid H)$: likelihood?** #flashcard
how probable the evidence is if the hypothesis is true

**$P(H \mid E)$: posterior?** #flashcard
updated belief after evidence

**$P(E) = \sum_h P(E \mid H=h) P(H=h)$?** #flashcard
marginal likelihood (normalizing constant)

**MAP estimation: $\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid D) = \arg\max_\theta [P(D \mid \theta) P(\theta)]$?** #flashcard
the prior regularizes the estimate

**Naive Bayes classifier?** #flashcard
applies Bayes' theorem with conditional independence assumption on features

**Bayesian neural networks?** #flashcard
maintain distributions over weights rather than point estimates

**Saying "L2 regularization prevents overfitting" without the mechanism. The mechanism is that it corresponds to a Gaussian prior that penalizes extreme parameter values?** #flashcard
it is Bayesian MAP, not an arbitrary constraint.

**Not connecting regularization strength to the data-to-feature ratio. When you have many more features than examples, a stronger prior (larger $\lambda$) is appropriate; when you have many more examples than features, the prior matters less.?** #flashcard
Not connecting regularization strength to the data-to-feature ratio. When you have many more features than examples, a stronger prior (larger $\lambda$) is appropriate; when you have many more examples than features, the prior matters less.

**Treating bootstrap as a way to generate more training data. It does not add information?** #flashcard
it quantifies uncertainty about a statistic computed from existing data.

**Bootstrap does not work well with very small samples ($n < 30$) because the empirical distribution is a poor proxy for the population. The bootstrap samples a distribution that already has limited diversity.?** #flashcard
Bootstrap does not work well with very small samples ($n < 30$) because the empirical distribution is a poor proxy for the population. The bootstrap samples a distribution that already has limited diversity.

**Not using enough bootstrap iterations. $B = 100$ is too few for precise CI estimation; $B = 1000$ is a safe minimum; $B = 10000$ is better for tail percentiles.?** #flashcard
Not using enough bootstrap iterations. $B = 100$ is too few for precise CI estimation; $B = 1000$ is a safe minimum; $B = 10000$ is better for tail percentiles.
