# Statistics and Probability

---

## 1. Descriptive Statistics

### Central Tendency
- **Mean**: $\bar{x} = \frac{1}{n}\sum x_i$ — sensitive to outliers
- **Median**: middle value after sorting — robust to outliers; preferred for skewed distributions
- **Mode**: most frequent value — useful for categorical data; a distribution can be multimodal

### Spread
- **Variance**: $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$ — Bessel's correction (n-1) for unbiased sample estimate
- **Standard deviation**: $s = \sqrt{s^2}$ — same units as data
- **IQR**: Q3 − Q1 — resistant to outliers; used in box plot fences
- **Percentiles**: $p$-th percentile = value below which $p$% of observations fall

### Shape
- **Skewness (Fisher's formula)**: $g_1 = \frac{n}{(n-1)(n-2)} \sum\left(\frac{x_i - \bar{x}}{s}\right)^3$
  - Positive (right) skew: long right tail, mean > median
  - Negative (left) skew: long left tail, mean < median
- **Excess kurtosis**: $g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum\left(\frac{x_i-\bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$
  - Normal distribution: excess kurtosis = 0
  - Leptokurtic (> 0): heavy tails, sharp peak — financial returns
  - Platykurtic (< 0): thin tails, flat peak

---

## 2. Probability Distributions

### Normal Distribution $\mathcal{N}(\mu, \sigma^2)$
- PDF: $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
- **68-95-99.7 rule**: 68% within ±1σ, 95% within ±2σ, 99.7% within ±3σ
- **Central Limit Theorem (CLT)**: sample mean $\bar{X} \sim \mathcal{N}(\mu, \sigma^2/n)$ for large $n$, regardless of population distribution

### Binomial $B(n, p)$
- Models number of successes in $n$ independent Bernoulli trials
- $P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$
- Mean: $np$, Variance: $np(1-p)$
- Approximates Normal when $np \geq 5$ and $n(1-p) \geq 5$

### Poisson $\text{Pois}(\lambda)$
- Models count of rare events in fixed interval; limit of Binomial as $n \to \infty$, $p \to 0$, $\lambda = np$
- $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$
- Mean = Variance = $\lambda$
- Use cases: arrivals, defects, page views in a time window

### Beta $\text{Beta}(\alpha, \beta)$
- Defined on $[0,1]$; models probabilities or proportions
- Mean: $\frac{\alpha}{\alpha+\beta}$, Mode: $\frac{\alpha-1}{\alpha+\beta-2}$ for $\alpha,\beta > 1$
- **Conjugate prior** for Bernoulli/Binomial likelihood: posterior remains Beta after observing successes/failures
- $\text{Beta}(1,1)$ = Uniform prior

### Gamma $\text{Gamma}(\alpha, \beta)$
- Models waiting time until $\alpha$ events in a Poisson process with rate $\beta$
- Mean: $\alpha/\beta$, Variance: $\alpha/\beta^2$
- Special cases: Exponential ($\alpha=1$), Chi-squared ($\alpha=k/2$, $\beta=1/2$)

### t-Distribution $t_\nu$
- Heavier tails than Normal; parameter $\nu$ = degrees of freedom
- As $\nu \to \infty$, converges to standard Normal
- Used when $\sigma$ unknown and $n$ small; robust to non-normality for moderate $\nu$

### Chi-Squared $\chi^2_k$
- Sum of $k$ squared standard Normal variables
- Used in goodness-of-fit and independence tests
- Mean: $k$, Variance: $2k$

### F-Distribution $F_{d_1, d_2}$
- Ratio of two chi-squared variables divided by their degrees of freedom
- Used in ANOVA and testing equality of variances
- $F = \frac{\chi^2_{d_1}/d_1}{\chi^2_{d_2}/d_2}$

---

## 3. Hypothesis Testing

### Framework
- **Null hypothesis** $H_0$: status quo; assumed true until evidence against it
- **Alternative hypothesis** $H_1$: what you want to detect
- **Type I error (α)**: reject $H_0$ when true — false positive; controlled by significance level
- **Type II error (β)**: fail to reject $H_0$ when false — false negative
- **Power** = $1 - \beta$: probability of correctly detecting a true effect
- **p-value**: probability of observing a test statistic at least as extreme as the one observed, assuming $H_0$ is true — not the probability that $H_0$ is true
- **One-tailed**: directional hypothesis; **two-tailed**: non-directional — use two-tailed by default unless direction is pre-specified

### z-test
- Conditions: known population $\sigma$, or large $n$ (≥ 30)
- $z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}$

### t-test
- Conditions: unknown $\sigma$, approximately normal data (or $n \geq 30$ by CLT)
- **One-sample**: $t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$, df = $n-1$
- **Two-sample (independent)**: $t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_p^2(1/n_1 + 1/n_2)}}$, Welch's t when variances unequal
- **Paired**: $t = \frac{\bar{d}}{s_d/\sqrt{n}}$ — use when observations are matched (before/after, same subject)

### ANOVA
- Tests whether means differ across $\geq 3$ groups
- $F = \frac{\text{between-group variance}}{\text{within-group variance}} = \frac{MS_B}{MS_W}$
- $MS_B = \frac{SS_B}{k-1}$, $MS_W = \frac{SS_W}{N-k}$ where $k$ = groups, $N$ = total observations
- Significant F → at least one group mean differs; post-hoc tests (Tukey, Bonferroni) identify which pairs

### Chi-Squared Test
- **Independence**: tests whether two categorical variables are associated; $\chi^2 = \sum \frac{(O-E)^2}{E}$, df = $(r-1)(c-1)$
- **Goodness of fit**: tests whether observed distribution matches expected; df = $k-1$
- Expected cell count ≥ 5 for valid approximation

### Non-Parametric Tests
- **Mann-Whitney U**: compares two independent groups on ordinal/non-normal data; tests whether distributions are identical; equivalent to testing whether $P(X > Y) = 0.5$
- **Kruskal-Wallis**: non-parametric analogue of one-way ANOVA; ranks all observations then tests whether rank distributions differ across groups

---

## 4. Multiple Testing

### The Problem
- With $m$ independent tests at $\alpha = 0.05$: probability of at least one false positive = $1 - (1-0.05)^m$
- At $m = 20$: ~64% chance of false positive

### Corrections
- **Bonferroni**: $\alpha_{\text{adjusted}} = \alpha / m$ — controls family-wise error rate (FWER); conservative when tests are correlated
- **Benjamini-Hochberg FDR**: sort p-values $p_{(1)} \leq \ldots \leq p_{(m)}$; reject $H_{(i)}$ if $p_{(i)} \leq \frac{i}{m} \cdot q^*$ where $q^*$ is target FDR; less conservative than Bonferroni, controls expected proportion of false discoveries
- **When naive thresholds fail**: testing 100 metrics simultaneously with $\alpha = 0.05$ produces ~5 false positives by chance even with no real effects

---

## 5. Confidence Intervals

- A 95% CI means: if you repeated the experiment many times, 95% of constructed intervals would contain the true parameter — not that there's a 95% probability the parameter is in this specific interval
- **Relation to hypothesis tests**: reject $H_0: \mu = \mu_0$ at level $\alpha$ iff $\mu_0 \notin (1-\alpha)$ CI
- **Bootstrap CI**: resample data with replacement $B$ times, compute statistic each time; percentile method: $[\hat{\theta}_{(\alpha/2)}, \hat{\theta}_{(1-\alpha/2)}]$; BCa (bias-corrected accelerated) for skewed distributions

---

## 6. Bayesian Statistics

### Core Framework
- **Prior** $P(\theta)$: belief about parameter before data
- **Likelihood** $P(D|\theta)$: probability of data given parameter
- **Posterior** $P(\theta|D) \propto P(D|\theta) \cdot P(\theta)$: updated belief after data

### Estimation
- **MLE**: $\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(D|\theta)$ — maximizes likelihood; equivalent to mode of likelihood
- **MAP**: $\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta|D)$ — mode of posterior; MLE with regularization via prior
- **Posterior mean**: full Bayesian estimate; minimizes expected squared error

### Conjugate Priors
| Likelihood | Conjugate Prior | Posterior |
| :--- | :--- | :--- |
| Bernoulli/Binomial | Beta($\alpha$, $\beta$) | Beta($\alpha+k$, $\beta+n-k$) |
| Poisson | Gamma($\alpha$, $\beta$) | Gamma($\alpha+\sum x$, $\beta+n$) |
| Normal (known $\sigma$) | Normal | Normal |

### Credible Intervals vs Confidence Intervals
- **Credible interval**: direct probability statement — "there is 95% probability the parameter is in this interval" — valid under Bayesian framework
- **Confidence interval**: frequentist — the interval construction procedure captures the true parameter 95% of the time
- Practically similar numerically but semantically different

---

## 7. Correlation

### Pearson $r$
- $r = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \sum(y_i-\bar{y})^2}}$
- Measures linear association; $r \in [-1, 1]$
- Sensitive to outliers; assumes continuous variables, approximately bivariate normal

### Spearman $\rho$
- Pearson $r$ applied to ranks of $x$ and $y$
- Measures monotonic (not necessarily linear) association
- Robust to outliers; valid for ordinal data

### Kendall $\tau$
- $\tau = \frac{\text{concordant pairs} - \text{discordant pairs}}{\binom{n}{2}}$
- More interpretable: probability that two randomly selected pairs are concordant minus probability they are discordant
- More robust than Spearman for small samples with many ties

### Partial Correlation
- Correlation between $X$ and $Y$ after controlling for $Z$: remove the linear effect of $Z$ from both $X$ and $Y$, then correlate residuals
- Distinguishes direct relationship from relationship mediated by a confounder

### Correlation ≠ Causation
- Spurious correlation: both variables driven by common cause (confounder)
- Reverse causation: $Y$ causes $X$, not $X$ causes $Y$
- Coincidental correlation: no causal link — e.g., Nicolas Cage films correlate with pool drownings

---

## 8. Regression Inference

### OLS Assumptions (LINE)
- **L**inearity: $E[Y|X] = X\beta$
- **I**ndependence: residuals $\epsilon_i$ are independent
- **N**ormality: residuals normally distributed — needed for finite-sample inference, not for asymptotic
- **E**qual variance (homoscedasticity): $\text{Var}(\epsilon_i) = \sigma^2$ constant across $X$; violation = heteroscedasticity → use robust standard errors

### Model Fit
- **R²**: proportion of variance in $Y$ explained by model; $R^2 = 1 - SS_{\text{res}}/SS_{\text{tot}}$; never decreases with added features
- **Adjusted R²**: penalizes for number of predictors; $\bar{R}^2 = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$; use for model comparison
- **F-test for overall fit**: tests $H_0: \beta_1 = \ldots = \beta_p = 0$; $F = \frac{R^2/p}{(1-R^2)/(n-p-1)}$
- **t-test per coefficient**: $t = \hat{\beta}_j / \text{SE}(\hat{\beta}_j)$ — tests whether individual predictor contributes given others in model

### Multicollinearity
- **VIF (Variance Inflation Factor)**: $\text{VIF}_j = \frac{1}{1 - R^2_j}$ where $R^2_j$ is from regressing $X_j$ on all other predictors
- VIF > 5: moderate concern; VIF > 10: severe — inflates standard errors, makes coefficients unstable
- Remedies: drop one correlated feature, PCA, ridge regression

---

## 9. Central Limit Theorem and Law of Large Numbers

### CLT
- For i.i.d. samples with finite mean $\mu$ and variance $\sigma^2$: $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$
- Enables using Normal-based inference (z/t-tests) regardless of population shape for large $n$
- **When CLT breaks**: heavy-tailed distributions (infinite variance — Cauchy, Pareto with $\alpha \leq 2$); small $n$; strong dependence between observations

### LLN
- **Weak LLN**: $\bar{X}_n \xrightarrow{p} \mu$ as $n \to \infty$ (convergence in probability)
- **Strong LLN**: $\bar{X}_n \to \mu$ almost surely
- Foundation for Monte Carlo methods: average of random samples converges to expectation
- Why it matters for sampling: larger samples give more reliable estimates; but it says nothing about rate — that's the CLT
