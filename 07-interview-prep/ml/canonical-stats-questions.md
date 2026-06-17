---
module: Interview Prep
topic: Ml
subtopic: Canonical Stats Questions
status: unread
tags: [interviewprep, ml, ml-canonical-stats-questions]
---
# Canonical Statistics Interview Questions

The statistical foundations that appear in every ML interview. Covers probability theory, inference, experimental design, and the traps interviewers set.

---

## 1. Bayes' Theorem and the Base Rate Fallacy

### Bayes' Theorem
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} = \frac{P(B|A) \cdot P(A)}{P(B|A)P(A) + P(B|\bar{A})P(\bar{A})}$$

### Classic Trap: Medical Test

**Problem:** A disease affects 1% of the population. A test has 99% sensitivity (P(+|disease)=0.99) and 99% specificity (P(-|no disease)=0.99). You test positive. What is P(disease|+)?

**Base rate fallacy error:** Most people answer ~99%.

**Correct calculation:**
$$P(disease|+) = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.01 \times 0.99} = \frac{0.0099}{0.0198} = 0.5$$

**Only 50%** — despite a 99% accurate test, because disease is rare (1% prior). The false positives from the 99% of healthy people swamp the true positives.

```python
def bayes_update(prior, sensitivity, specificity):
    """P(condition|positive test)"""
    p_pos_given_cond = sensitivity
    p_pos_given_no_cond = 1 - specificity
    p_pos = p_pos_given_cond * prior + p_pos_given_no_cond * (1 - prior)
    return (p_pos_given_cond * prior) / p_pos

# Disease prev 1%, test 99% sensitivity/specificity
print(bayes_update(0.01, 0.99, 0.99))  # 0.50
# Disease prev 10%
print(bayes_update(0.10, 0.99, 0.99))  # 0.917
# Disease prev 50%
print(bayes_update(0.50, 0.99, 0.99))  # 0.990
```

**Interview takeaway:** Prior probability (base rate) fundamentally changes the interpretation of any test result. High sensitivity/specificity alone does not guarantee high positive predictive value.

**ML application:** Spam detection at 0.01% spam rate. Even a 99.9% accurate model has poor PPV. Must tune threshold or reject with "not enough signal" when prior is extreme.

---

## 2. p-values — Explaining to Non-Technical Stakeholders

### Formal Definition
$$p\text{-value} = P(\text{observing data this extreme or more extreme} | H_0 \text{ is true})$$

### Common Misconceptions

| What p-value is NOT | Correct interpretation |
|---|---|
| P(H₀ is true) | P(data | H₀ is true) |
| P(H₁ is true | data) | No statement about H₁ probability |
| Effect size | p < 0.05 does not mean large effect |
| Replication probability | P(p < 0.05 in next study) ≠ 1 - p |

### Non-Technical Explanation

"Imagine the treatment has absolutely no effect. If we ran this experiment 100 times with no effect, we'd expect to see results as extreme as what we measured about [p×100]% of the time. A small p-value means: 'if nothing was happening, this result would be surprising.' It doesn't prove the effect exists — it just means the result is unlikely under the 'nothing happening' assumption."

### The Right Questions to Ask After "p < 0.05"

1. What is the **effect size**? (Cohen's d, relative risk, absolute difference)
2. What is the **confidence interval**? (direction and practical significance)
3. Was this a **pre-registered hypothesis** or did you look at many outcomes?
4. What is the **statistical power**? (were we adequately powered?)
5. Is there a **multiple testing** correction?

---

## 3. Multiple Testing (Bonferroni and Benjamini-Hochberg)

**The problem:** Test 20 independent hypotheses at α=0.05. Expected false positives = 20 × 0.05 = 1, even if all H₀ are true.

**Family-wise error rate (FWER):** P(at least one false positive) = 1 - (1-α)^m ≈ 1 for large m.

### Bonferroni Correction (Conservative)

$$\alpha_{adjusted} = \frac{\alpha}{m}$$

For m=100 tests and α=0.05: αadjusted = 0.0005. Reject Hᵢ if pᵢ < 0.0005.

**Problem:** Too conservative — many false negatives when tests are correlated.

### Benjamini-Hochberg (FDR Control)

Controls **False Discovery Rate** (expected fraction of rejections that are false positives), not FWER.

**Algorithm:**
1. Sort p-values: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍m₎
2. Find largest k such that $p_{(k)} \leq \frac{k}{m} \alpha$
3. Reject all H₍ᵢ₎ for i ≤ k

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.001, 0.02, 0.03, 0.04, 0.8, 0.9]

# Bonferroni
reject_bf, p_corrected_bf, _, _ = multipletests(p_values, method='bonferroni', alpha=0.05)

# Benjamini-Hochberg
reject_bh, p_corrected_bh, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)

print("Bonferroni rejections:", reject_bf)
print("BH rejections:", reject_bh)
```

**When to use which:**
- Bonferroni: confirmatory study, FWER control critical (clinical trials, p-hacking risk high)
- BH: exploratory study, many hypotheses (genomics, feature selection), willing to tolerate some FDR

**ML application:** A/B testing 50 metrics simultaneously → BH correction to avoid declaring winners from noise. Feature selection with 1000 features → BH on feature importance tests.

---

## 4. Simpson's Paradox

**Definition:** A trend appears in subgroups but reverses (or disappears) in the aggregate.

**Classic example (UC Berkeley admissions 1973):**

| | Men | Women |
|---|---|---|
| Applied | 8442 | 4321 |
| Admitted | 44% | 35% |

Looks like gender discrimination. But per-department:

| Dept | Men rate | Women rate | Women applicants |
|---|---|---|---|
| A | 62% | 82% | 108 |
| B | 63% | 68% | 25 |
| C | 37% | 34% | 593 |
| D | 33% | 35% | 375 |

Women had higher admission rates in most departments, but applied to more competitive (lower acceptance rate) departments. The aggregate flipped due to the **confounding variable** (department choice).

**Why it happens:** A lurking variable (confound) Z correlates with both X (treatment) and Y (outcome). Aggregating ignores Z.

```python
import pandas as pd

# Simulated example
data = pd.DataFrame({
    'treatment': [1,1,0,0,1,1,0,0],
    'group': ['A','A','A','A','B','B','B','B'],
    'success': [6,7,2,3,3,4,8,9]
})

# Aggregate: treatment looks worse
print(data.groupby('treatment')['success'].mean())

# By group: treatment looks better in both groups
print(data.groupby(['group', 'treatment'])['success'].mean())
```

**Interview rule:** Always check aggregated vs stratified results when making causal claims. Simpson's paradox is a red flag that a confounder exists.

---

## 5. Bootstrap Confidence Intervals

**Use case:** When you can't assume normality or don't know the sampling distribution analytically.

**Algorithm:**
1. Resample with replacement N times from your data
2. Compute statistic θ̂* for each resample
3. Take 2.5th and 97.5th percentiles as the 95% CI

$$\text{CI}_{95\%} = [\hat{\theta}^*_{(0.025)}, \hat{\theta}^*_{(0.975)}]$$

```python
import numpy as np

def bootstrap_ci(data, statistic_fn, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval."""
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_fn(resample))
    
    alpha = 1 - ci
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return lower, upper

# Example: CI for median AUC across cross-validation folds
auc_scores = [0.82, 0.85, 0.83, 0.79, 0.87]
lower, upper = bootstrap_ci(auc_scores, np.median)
print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
```

**Bias-corrected accelerated (BCa) bootstrap:** Corrects for bias and skewness in the bootstrap distribution — more accurate than percentile method for small samples or skewed statistics.

**When bootstrap is essential:**
- Non-standard statistics (median, quantiles, AUC, F1 score)
- Small samples where CLT doesn't apply
- Complex model metrics (e.g., "average NDCG across 5 folds")

---

## 6. MLE vs MAP

### Maximum Likelihood Estimation (MLE)

$$\hat{\theta}_{MLE} = \arg\max_\theta P(D|\theta) = \arg\max_\theta \sum_i \log P(x_i|\theta)$$

**No prior.** Just finds parameters that make the observed data most probable.

**Example:** Coin flip (Bernoulli). n heads in N flips.
$$\hat{p}_{MLE} = \frac{n}{N}$$

### Maximum A Posteriori (MAP)

$$\hat{\theta}_{MAP} = \arg\max_\theta P(\theta|D) = \arg\max_\theta [P(D|\theta) \cdot P(\theta)]$$

Incorporates prior P(θ). With log: $\arg\max_\theta [\log P(D|\theta) + \log P(\theta)]$.

**Example:** Coin flip with Beta(α, β) prior.
$$\hat{p}_{MAP} = \frac{n + \alpha - 1}{N + \alpha + \beta - 2}$$

With α=β=2 (prior: fair coin): $\hat{p}_{MAP} = \frac{n+1}{N+2}$ — pushes estimate toward 0.5.

### Equivalences

| Prior | MAP equivalent |
|---|---|
| Gaussian P(θ) = N(0, σ²) | L2 regularization (ridge) |
| Laplace P(θ) = Laplace(0, b) | L1 regularization (lasso) |
| Uniform P(θ) | MLE (no regularization) |

**L2 regularization is MAP with a Gaussian prior** — this is the Bayesian interpretation of weight decay.

```python
# MLE: no regularization
# MAP with Gaussian prior: L2 penalty
# Loss = NLL + (1/2σ²) ||θ||²   ← σ² controls prior strength

from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)  # MAP with Gaussian prior, σ² = 1/alpha
lasso = Lasso(alpha=1.0)  # MAP with Laplace prior
```

---

## 7. Central Limit Theorem and When It Fails

### CLT Statement

For i.i.d. samples X₁, ..., Xₙ from a distribution with mean μ and variance σ²:

$$\sqrt{n} \cdot \frac{\bar{X}_n - \mu}{\sigma} \xrightarrow{d} N(0, 1) \text{ as } n \to \infty$$

**Practical rule:** CLT approximation holds for n ≥ 30 in most cases.

### When CLT Fails

| Condition | Problem | Fix |
|---|---|---|
| Heavy tails (power law) | Variance may not exist | Bootstrap, robust statistics |
| High skewness, small n | n < 30 insufficient | Exact tests, permutation |
| Dependent samples | i.i.d. assumption violated | Time series methods, block bootstrap |
| Extreme imbalance (1:10000) | Rare event | Exact binomial, Poisson approximation |

**Example: Ad CTR analysis**
- CTR = 0.001 (1 in 1000), n = 100 users
- Expected clicks = 0.1 — CLT fails badly
- Use exact binomial or Poisson test instead

```python
from scipy.stats import binom_test, poisson

# Exact binomial test: observed 5 clicks in 1000 impressions, expected p=0.003
result = binom_test(5, 1000, 0.003, alternative='two-sided')
print(f"p-value: {result:.4f}")

# For Poisson: observed 5, expected lambda=3
from scipy.stats import poisson
p = 1 - poisson.cdf(4, mu=3)  # P(X >= 5)
print(f"p-value: {p:.4f}")
```

---

## 8. Causal Study Design (Observational vs Experimental)

### Hierarchy of Evidence

| Design | Causal strength | Example |
|---|---|---|
| RCT (A/B test) | Gold standard | Feature flag experiment |
| Quasi-experiment (IV, DID) | Strong | Rollout by region |
| Regression discontinuity | Strong (local) | Cutoff threshold effects |
| Propensity score matching | Moderate | Observational with controls |
| Regression on observational | Weak | Correlational analysis |

### Instrument Variables (IV)

Used when treatment is endogenous (correlated with unobserved confounders).

**Requirements:**
1. Z is correlated with treatment T (relevance)
2. Z affects outcome Y only through T (exclusion restriction)
3. Z is independent of unobserved confounders (exogeneity)

**Example:** Estimating effect of education on wages. Problem: unobserved ability confounds education and wages. Instrument: distance to nearest college (affects education access but not wages directly).

**IV estimator:**
$$\hat{\beta}_{IV} = \frac{\text{Cov}(Z, Y)}{\text{Cov}(Z, T)}$$

### Difference-in-Differences (DiD)

**Estimate treatment effect from natural experiment:**
$$\text{ATT} = (\bar{Y}_{treated,post} - \bar{Y}_{treated,pre}) - (\bar{Y}_{control,post} - \bar{Y}_{control,pre})$$

**Parallel trends assumption:** In absence of treatment, treated and control groups would have evolved in parallel.

```python
import pandas as pd
from statsmodels.formula.api import ols

def diff_in_diff(df):
    """
    df columns: treated (0/1), post (0/1), outcome
    """
    model = ols('outcome ~ treated + post + treated:post', data=df).fit()
    # Coefficient on treated:post = ATT
    return model.params['treated:post'], model.pvalues['treated:post']
```

---

## 9. Common Statistical Traps in ML Interviews

**Trap 1: Using accuracy for imbalanced classification**
- 99% accuracy on 1% positive class by predicting all negative
- Fix: precision, recall, F1, PR-AUC, cost-sensitive metrics

**Trap 2: Data leakage via feature engineering**
- Scaling with statistics computed on full dataset (including test)
- Fix: fit scaler on train only, transform train+test

**Trap 3: Multiple comparison without correction**
- "We ran 20 A/B tests and this one was significant"
- Fix: Bonferroni or BH correction; pre-register primary metrics

**Trap 4: Interpreting correlation as causation**
- Ice cream sales and drowning rates correlate (confounder: summer)
- Fix: draw the DAG, identify confounders, use proper causal methods

**Trap 5: p-hacking / HARKing**
- Hypothesizing after results known, stopping data collection when p < 0.05
- Fix: pre-registration, sequential testing with alpha spending

**Trap 6: Survivorship bias in model evaluation**
- Evaluating a trading model only on stocks that existed in 2020
- Evaluating a cancer detection model only on confirmed cancer cases
- Fix: ensure evaluation set represents the population the model will see

---

## Canonical Interview Q&As

**Q: Explain p-values to a product manager who wants to know if our experiment worked.**  
A: "Imagine the new feature has absolutely no effect — the two versions are identical. If we ran this experiment 100 times with a useless feature, about [p×100] of those experiments would show results as different as what we're seeing just by random chance. Since p=0.03, only 3 out of 100 experiments would show results this extreme by chance. That means it's quite unlikely the feature is useless — we have reasonable evidence the effect is real. But the effect size is [X]% change in [metric], and our 95% confidence interval is [range] — that tells you whether it's actually worth shipping."

**Q: How would you handle testing 50 metrics in a single A/B experiment?**  
A: Designate one primary metric before the experiment (pre-registration). Apply Benjamini-Hochberg correction to the other 49 secondary metrics, targeting 5% FDR. Report the BH-adjusted p-values. Be suspicious of any surprising secondary metric not in the pre-registration — it could be a false positive that needs a dedicated follow-up experiment. The primary metric decision is the one that matters for shipping; secondary metrics provide directional signal only.

**Q: You're analyzing whether a new treatment increases revenue. Treated users have higher average income. How do you handle this?**  
A: Income is a confounding variable — it may independently affect revenue and also correlates with treatment assignment (if treatment was non-random). Options in order of rigor: (1) RCT — randomize treatment to eliminate confounding; (2) Propensity score matching/weighting — estimate P(treatment|income) and reweight or match; (3) DiD — if treatment was rolled out at different times to income groups, use the timing as a quasi-instrument; (4) Control for income in regression. The key question: was treatment randomly assigned? If yes, income imbalance is by chance and the concern is less severe. If not, you must control for it.

**Q: What's the difference between confidence intervals and credible intervals?**  
A: Confidence interval (frequentist): a procedure that, if repeated many times, would contain the true parameter 95% of the time. Any specific CI either contains the true value or doesn't — we can't say P(θ in CI) = 95%. Credible interval (Bayesian): given the data and prior, P(θ in CI | data) = 95%. It directly expresses uncertainty about the parameter given what we observed. Practically: credible intervals can be reported more intuitively ("there's a 95% probability the effect is between X and Y"). Confidence intervals have a subtle interpretation that's easy to misstate. For large samples with weak priors, the two are numerically similar.
