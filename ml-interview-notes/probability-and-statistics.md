# Probability and Statistics

---

# Q1: Explain the Bias-Variance Tradeoff.

## 1. 🔹 Direct Answer
**Bias**: error from **wrong assumptions** (underfitting). **Variance**: sensitivity to **training set** noise (overfitting). **Tradeoff**: simpler models **↑ bias ↓ variance**; complex **↓ bias ↑ variance**. **Goal**: minimize **expected test error** = bias² + variance + irreducible noise.

## 2. 🔹 Intuition
**Rigidity** vs **wiggly** fit—find sweet spot for **generalization**.

## 3. 🔹 Deep Dive
**MSE decomposition** for regression; **double descent** complicates classical U-shape for deep nets.

## 4. 🔹 Practical Perspective
**Learning curves** diagnose; **regularization** reduces variance.

## 5. 🔹 Code Snippet
```python
from sklearn.model_selection import learning_curve
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Ensemble effect? **A:** Bagging ↓ variance; boosting ↓ bias.

## 7. 🔹 Common Mistakes
Thinking deeper nets always increase variance—depends on training regime.

## 8. 🔹 Comparison / Connections
Regularization, model capacity.

## 9. 🔹 One-line Revision
Bias-variance decomposes generalization error—tune model complexity and regularization.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q2: Explain different probability distributions (Normal, Binomial, Poisson, Uniform).

## 1. 🔹 Direct Answer
**Uniform**: equal probability over **[a,b]**—max entropy continuous bounded. **Normal**: bell curve, **CLT** limit—mean μ, variance σ². **Binomial**: **n** independent **Bernoulli** trials—count of successes. **Poisson**: **rare events** in interval—**rate λ**, mean=variance.

## 2. 🔹 Intuition
Pick distribution matching **generative** story and **support**.

## 3. 🔹 Deep Dive
**PMF/PDF** forms; **conjugate priors** (Beta-Binomial, Gamma-Poisson).

## 4. 🔹 Practical Perspective
**Check** assumptions (independence for Binomial).

## 5. 🔹 Code Snippet
```python
import numpy as np
np.random.normal(0, 1, size=1000)
np.random.binomial(n=10, p=0.3, size=1000)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** When normal approx binomial? **A:** Large n, p not extreme.

## 7. 🔹 Common Mistakes
Using Poisson when **overdispersed**—negative binomial instead.

## 8. 🔹 Comparison / Connections
Exponential family, GLMs.

## 9. 🔹 One-line Revision
Match distribution to data-generating process: Uniform flat, Normal symmetric, Binomial counts, Poisson rare events.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q3: What is the normal distribution and its functions?

## 1. 🔹 Direct Answer
**Gaussian** **N(μ,σ²)** PDF **∝ exp(−(x−μ)²/(2σ²))**—symmetric, **mean=median=mode**. **CDF** Φ(z) no closed form—use **erf** tables. **Standard normal** μ=0, σ=1.

## 2. 🔹 Intuition
Natural **aggregate** of many small effects (**CLT**).

## 3. 🔹 Deep Dive
**68-95-99.7** rule; **log-normal** if log is normal.

## 4. 🔹 Practical Perspective
**Outliers** heavy-tailed real data—**robust** methods needed.

## 5. 🔹 Code Snippet
```python
from scipy.stats import norm
norm.ppf(0.975)  # ~1.96
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Multivariate? **A:** Covariance matrix Σ in exponent.

## 7. 🔹 Common Mistakes
Assuming **residuals** normal without checking QQ plot.

## 8. 🔹 Comparison / Connections
t-distribution (unknown variance), KL divergence between Gaussians.

## 9. 🔹 One-line Revision
Normal distribution is the symmetric bell curve parameterized by mean and variance—central to CLT and inference.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q4–Q11: Common distributions (Exponential, Binomial, Bernoulli, Multinomial, Log-normal, Logistic, Gamma, Poisson)

## Q4 Exponential
**Direct**: Time between **Poisson** events—**memoryless**, rate λ, mean **1/λ**. **Use**: survival, waiting times. **PDF**: λe^(−λx).

## Q5 Binomial
**Direct**: Number of successes in **n** **i.i.d.** Bernoulli(p). **Mean** np, **variance** np(1−p).

## Q6 Bernoulli
**Direct**: Single trial **0/1** with prob p—building block of Binomial.

## Q7 Multinomial
**Direct**: Extension of Binomial to **K** categories—counts **n₁…n_K** from n trials with probabilities **p**.

## Q8 Log-normal
**Direct**: If **log X ~ Normal**, then **X** is log-normal—**positive** skew (incomes, latency). **Multiplicative** processes.

## Q9 Logistic
**Direct**: S-shaped CDF for **logistic** function—similar to normal CDF, **heavier tails**. Used in **logistic distribution** less than probit.

## Q10 Gamma
**Direct**: **Sum** of exponentials or **conjugate** to Poisson rate; **shape k**, **scale θ**—models **waiting** until k events.

## Q11 Poisson
**Direct**: Counts in fixed interval with rate λ—**mean=variance**; **PMF** P(k)=λ^k e^{-λ}/k!.

### Interview Follow-ups (shared)
**When Poisson vs Binomial?** Poisson approximates Binomial when **n large, p small** (rare events). **Overdispersion?** Use **negative binomial**.

### One-line Revision
Master **story** for each: Bernoulli trial → Binomial sum → Poisson rare limit; Exponential/Gamma for waiting; Log-normal for positive multiplicative; Multinomial for categorical counts.

### Difficulty Tag
🟡 Medium

---

# Q12: When would you use a Poisson distribution over a Binomial distribution?

## 1. 🔹 Direct Answer
**Poisson** when **n large**, **p small**, only care about **rate** λ≈np—**unbounded** count support simpler. **Binomial** when **fixed n trials** with known **p** per trial and **finite** max count.

## 2. 🔹 Intuition
Rare events per minute—don’t know **n** explicitly—model **λ** directly.

## 3. 🔹 Deep Dive
**Poisson limit** of Binomial as n→∞, p→0, np=λ.

## 4. 🔹 Practical Perspective
**Exposure** offsets in GLM: **log μ = xᵀβ + log T** for varying intervals T.

## 5. 🔹 Code Snippet
```python
from scipy.stats import poisson, binom
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Zero-inflation? **A:** Mixture model—many more zeros than Poisson.

## 7. 🔹 Common Mistakes
Using Poisson for **underdispersed** counts.

## 8. 🔹 Comparison / Connections
Negative binomial for overdispersion.

## 9. 🔹 One-line Revision
Use Poisson for rare-event counts / rates; Binomial for fixed known trial counts—Poisson is Binomial limit.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q13–Q14: Variance and Stddev

## Q13 What is variance?
**σ² = E[(X−μ)²]**—average squared deviation. **Units** squared.

## Q14 What is stddev?
**σ = √variance**—same units as X—**interpretable** spread.

### One-line Revision
Variance is expected squared distance from mean; stddev is its square root for interpretability.

### Difficulty Tag
🟢 Easy

---

# Q15: Explain mean, median, and mode.

## 1. 🔹 Direct Answer
**Mean**: arithmetic average—**sensitive** to outliers. **Median**: 50th **percentile**—**robust** central tendency. **Mode**: most **frequent** value—categorical/discrete; multimodal distributions exist.

## 2. 🔹 Intuition
Mean = center of mass; median = half-half split; mode = peak.

## 3. 🔹 Deep Dive
**Skew**: mean pulled toward tail; **median** often better for **reporting** income.

## 4. 🔹 Practical Perspective
**Choose** based on **outliers** and **business** meaning.

## 5. 🔹 Code Snippet
```python
import numpy as np
np.mean(x); np.median(x)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Trimmed mean? **A:** Robust compromise.

## 7. 🔹 Common Mistakes
Reporting mean **salary** in skewed org without median.

## 8. 🔹 Comparison / Connections
Expectation vs robust estimators.

## 9. 🔹 One-line Revision
Mean is sensitive average; median is robust center; mode is most common value.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q16: Correlation vs covariance.

## 1. 🔹 Direct Answer
**Cov(X,Y)=E[(X−μ_x)(Y−μ_y)]**—**scale-dependent**. **Corr ρ = Cov/(σ_x σ_y)** ∈ **[−1,1]**—**unitless**, **linear** association strength.

## 2. 🔹 Intuition
Covariance has **units** X·Y; correlation **normalized**.

## 3. 🔹 Deep Dive
**Independence** ⇒ Cov=0, but **Cov=0** **≠** independent (nonlinear deps).

## 4. 🔹 Practical Perspective
**Pearson** linear; **Spearman** rank (monotonic).

## 5. 🔹 Code Snippet
```python
np.corrcoef(x, y)[0,1]
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Spurious correlation? **A:** Third variable—confounding.

## 7. 🔹 Common Mistakes
Causation from correlation.

## 8. 🔹 Comparison / Connections
Mutual information for nonlinear dependence.

## 9. 🔹 One-line Revision
Covariance scales with units; correlation is standardized linear association—not causation.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q17: Correlation +1, 0, −1.

## 1. 🔹 Direct Answer
**+1**: perfect **positive linear** relationship. **−1**: perfect **negative linear**. **0**: **no linear** correlation (nonlinear possible).

## 2. 🔹 Intuition
Strength and direction of **linearity**.

## 3. 🔹 Deep Dive
**ρ²** fraction of variance explained in simple linear regression.

## 4. 🔹 Practical Perspective
**Near 0** can hide **strong U-shaped** relationship—plot data.

## 5. 🔹 Code Snippet
```text
ρ = Cov(X,Y)/(σ_x σ_y)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Nonlinear strong dep with ρ=0? **A:** y=x² symmetric around 0 if balanced—Pearson ~0.

## 7. 🔹 Common Mistakes
Treating **Spearman** same as Pearson without stating.

## 8. 🔹 Comparison / Connections
Cosine similarity centered vectors.

## 9. 🔹 One-line Revision
±1 perfect linear alignment; 0 no linear correlation—always visualize.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q18: Correlation vs Causation.

## 1. 🔹 Direct Answer
**Correlation** is **association**; **causation** requires **intervention** / **identification** strategy (**RCT**, **instrument**, **adjustment** with correct DAG). **Confounding** produces **spurious** correlation.

## 2. 🔹 Intuition
Ice cream and drowning correlate—**summer** confound.

## 3. 🔹 Deep Dive
**Pearl** causal graphs; **Rubin** potential outcomes.

## 4. 🔹 Practical Perspective
**A/B tests** for product causality; **observational** ML needs **domain** expertise.

## 5. 🔹 Code Snippet
```text
N/A
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Simpson’s paradox? **A:** Trend reverses when aggregating—confounding structure.

## 7. 🔹 Common Mistakes
Feature importance ⇒ causal effect.

## 8. 🔹 Comparison / Connections
Fairness, uplift modeling.

## 9. 🔹 One-line Revision
Correlation does not imply causation—need design or causal assumptions.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q19–Q20: Type I / II errors

## Q19 Type I and II
**Type I (α)**: **false positive**—reject true null. **Type II (β)**: **false negative**—fail to reject false null. **Power** = 1−β.

## Q20 Connection to ML
**Precision/Recall** analogs: FP vs FN trade-offs; **threshold** moves along ROC.

### One-line Revision
Type I false alarm; Type II miss—balance via α, sample size, and decision thresholds.

### Difficulty Tag
🟡 Medium

---

# Q21–Q24: p-value, significance, limitations, hypothesis testing in ML

## Q21 p-value
**Probability** of observing test statistic **at least as extreme** **if null true**—**not** P(null true|data).

## Q22 Statistical significance
**Traditional** α=0.05 threshold—**arbitrary**; prefer **CIs**, **effect sizes**, **Bayesian** view in many modern analyses.

## Q23 Limitations of p-values
**p-hacking**, **multiple comparisons**, ignores **effect size**, **misinterpretation** as P(H0).

## Q24 Hypothesis testing in ML
**A/B tests** for launches; **permutation** tests for model comparison; **avoid** testing **many** metrics without correction.

### One-line Revision
p-value is not posterior probability; use effect sizes, CIs, and pre-registration—multiple testing needs correction.

### Difficulty Tag
🟣 Hard

---

# Q25: What statistical tests would you use to compare two models?

## 1. 🔹 Direct Answer
**Paired** tests on **same** validation examples: **McNemar** (binary correct/incorrect), **paired t-test** on **losses** (check normality), **Wilcoxon signed-rank** **nonparametric**, **bootstrap** confidence intervals on **metric diff**. **Multiple datasets**: **5x2cv** test (legacy) or **bootstrap**.

## 2. 🔹 Intuition
**Paired** reduces variance by controlling **example** difficulty.

## 3. 🔹 Deep Dive
**Diebold-Mariano** for time series forecast comparison.

## 4. 🔹 Practical Perspective
Report **effect size** (AUC lift) not only p-value.

## 5. 🔹 Code Snippet
```python
from scipy.stats import wilcoxon
wilcoxon(scores_a, scores_b)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Why not just compare val accuracy? **A:** Need uncertainty—could be noise.

## 7. 🔹 Common Mistakes
**Data snooping** thousands of models then testing—invalid.

## 8. 🔹 Comparison / Connections
Bayesian model comparison.

## 9. 🔹 One-line Revision
Use paired nonparametric or bootstrap tests on same folds; report CIs and effect sizes.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q26: How do you assess if a feature is statistically significant?

## 1. 🔹 Direct Answer
In **linear regression**: **t-test** on **coefficient** (H0: β=0) with **p-value** / **CI**. For **many** features: **FDR** **Benjamini-Hochberg** control. **Mutual information** / **permutation** importance **nonparametric**. **Caution**: **statistical** significance **≠** **practical** importance.

## 2. 🔹 Intuition
Is signal **larger** than **noise** given **sample size**?

## 3. 🔹 Deep Dive
**Multicollinearity** inflates SE—**VIF** check.

## 4. 🔹 Practical Perspective
**Effect size** and **business** impact matter more than tiny p-values with huge n.

## 5. 🔹 Code Snippet
```python
import statsmodels.api as sm
sm.OLS(y, X).fit().summary()
```

## 6. 🔹 Interview Follow-ups
1. **Q:** High p but large coef? **A:** High variance—need more data or regularization.

## 7. 🔹 Common Mistakes
**Stepwise** regression p-hacking.

## 8. 🔹 Comparison / Connections
SHAP for model-agnostic importance.

## 9. 🔹 One-line Revision
Significance tests help but require multiplicity control and practical effect sizes—domain validation essential.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q27: Confidence interval and usage.

## 1. 🔹 Direct Answer
**95% CI**: interval **procedure** such that **95%** of repeated samples would contain **true parameter** (**frequentist**). **Interpretation**: **not** “95% prob parameter in this interval” for single interval—common misread.

## 2. 🔹 Intuition
**Plausible range** given data for parameter.

## 3. 🔹 Deep Dive
**Normal approx** **μ̂ ± 1.96 SE**; **bootstrap** CI for **complicated** metrics.

## 4. 🔹 Practical Perspective
Report **CI** with **A/B** results—not only point lift.

## 5. 🔹 Code Snippet
```python
import scipy.stats as st
st.t.interval(0.95, df=n-1, loc=mean, scale=se)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Bayesian credible interval? **A:** Posterior probability mass—different interpretation.

## 7. 🔹 Common Mistakes
Misinterpreting frequentist CI as Bayesian.

## 8. 🔹 Comparison / Connections
Prediction intervals vs confidence intervals.

## 9. 🔹 One-line Revision
Frequentist CIs capture sampling variability—pair with bootstrap for metrics without closed form.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q28–Q29: z-score and t-score

## Q28
**z = (x−μ)/σ**—standardize if **σ known** (often normal approx). **t** uses **sample** **s** with **n−1** df—**wider** tails for **small n**.

## Q29 When use each
**t** for **unknown variance** small samples; **z** large n (**CLT**) or known σ.

### One-line Revision
t-distribution accounts for estimating σ from small samples—→ z as n grows.

### Difficulty Tag
🟡 Medium

---

# Q30: Bayes' Theorem and Naive Bayes / Bayesian methods.

## 1. 🔹 Direct Answer
**P(A|B) = P(B|A)P(A)/P(B)**—invert conditional probabilities. **Naive Bayes** uses Bayes with **feature independence** given class. **Bayesian ML** places **priors** on parameters—**posterior** via Bayes rule (**MAP** vs **full posterior**).

## 2. 🔹 Intuition
**Prior** beliefs updated by **likelihood** from data.

## 3. 🔹 Deep Dive
**Conjugate** priors for closed-form updates; **MCMC** for general.

## 4. 🔹 Practical Perspective
**Calibration** benefits; **computation** heavier than point estimates.

## 5. 🔹 Code Snippet
```python
from sklearn.naive_bayes import GaussianNB
```

## 6. 🔹 Interview Follow-ups
1. **Q:** MAP vs MLE? **A:** MAP adds prior—like regularization.

## 7. 🔹 Common Mistakes
Thinking Bayesian always needs “subjective” priors—can use **uninformative**.

## 8. 🔹 Comparison / Connections
Laplace smoothing as Dirichlet prior.

## 9. 🔹 One-line Revision
Bayes rule inverts conditionals; Naive Bayes assumes conditional feature independence for tractable classification.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q31: MLE vs MAP.

## 1. 🔹 Direct Answer
**MLE**: **θ̂ = argmax_θ P(D|θ)**—**no prior**. **MAP**: **argmax_θ P(θ|D) ∝ P(D|θ)P(θ)**—includes **prior** **P(θ)**. **MAP** = **regularized** estimate; **Gaussian prior** → **L2**-like shrinkage.

## 2. 🔹 Intuition
MAP **pulls** estimates toward **prior** belief.

## 3. 🔹 Deep Dive
**Asymptotic** MLE properties under regularity; **MAP** not invariant under reparametrization same way.

## 4. 🔹 Practical Perspective
**Laplace** prior sparsity links to **L1**.

## 5. 🔹 Code Snippet
```text
MAP ∝ likelihood × prior
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Full Bayes vs MAP? **A:** MAP point estimate; full posterior gives uncertainty.

## 7. 🔹 Common Mistakes
Confusing **prior** with **regularization** purpose—related but philosophically nuanced.

## 8. 🔹 Comparison / Connections
Empirical Bayes.

## 9. 🔹 One-line Revision
MLE maximizes likelihood; MAP adds prior—acts like regularization.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q32: What is Maximum Likelihood Estimation (MLE)?

## 1. 🔹 Direct Answer
**MLE** chooses parameters **maximizing** **likelihood** **P(D|θ)** (or log-likelihood **sum**). **Asymptotically** efficient under assumptions—**connects** to **minimizing** **cross-entropy** for classification.

## 2. 🔹 Intuition
Pick θ making observed data **most probable**.

## 3. 🔹 Deep Dive
**Score** function; **Fisher information** for variance.

## 4. 🔹 Practical Perspective
**Overfitting** if model too rich—**regularization** / **MAP**.

## 5. 🔹 Code Snippet
```python
import numpy as np
# Bernoulli MLE for p is sample mean
p_mle = x.mean()
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Invariance? **A:** MLE of g(θ) is g(θ̂) under smooth g—nice property.

## 7. 🔹 Common Mistakes
MLE **always** “best”—can be biased in finite samples (variance estimation).

## 8. 🔹 Comparison / Connections
Method of moments, EM algorithm.

## 9. 🔹 One-line Revision
MLE maximizes data likelihood—foundation for logistic loss, Gaussian noise assumptions in regression.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q33: Bayesian vs Frequentist; CLT; sampling; bootstrap (combined essentials)

## Bayesian vs Frequentist
**Bayesian**: probability over **parameters** (epistemic). **Frequentist**: parameters **fixed**, randomness in **data**—**repeated sampling** interpretation.

## CLT
Sample mean **≈ Normal** for large n (i.i.d. finite variance)—justifies **z/t** tests and **confidence** approximations.

## Sampling techniques
**Simple random**, **stratified**, **cluster** (reduce cost), **importance sampling** for expectations.

## Bootstrap
**Resample data with replacement** to estimate **sampling distribution** of statistic—**nonparametric**, **no** distribution assumption.

### One-line Revision
Frequentist inference fixes parameters; Bayesian updates beliefs; CLT enables normal approximations; bootstrap estimates uncertainty empirically.

### Difficulty Tag
🟣 Hard

---
