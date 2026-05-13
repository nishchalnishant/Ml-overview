# Causal Inference and Experimentation

A practical guide for ML interviews — built around intuition first, math second.

---

## Table of Contents

1. [Correlation vs Causation](#1-correlation-vs-causation)
2. [The Fundamental Problem of Causal Inference](#2-the-fundamental-problem-of-causal-inference)
3. [A/B Testing Framework](#3-ab-testing-framework)
4. [Common A/B Testing Pitfalls](#4-common-ab-testing-pitfalls)
5. [Bayesian A/B Testing](#5-bayesian-ab-testing)
6. [Multi-Armed Bandits](#6-multi-armed-bandits)
7. [Observational Studies and Confounders](#7-observational-studies-and-confounders)
8. [Propensity Score Matching](#8-propensity-score-matching)
9. [Difference-in-Differences](#9-difference-in-differences)
10. [Regression Discontinuity Design](#10-regression-discontinuity-design)
11. [Instrumental Variables](#11-instrumental-variables)
12. [Causal Graphs (DAGs)](#12-causal-graphs-dags)
13. [Uplift Modeling / Heterogeneous Treatment Effects](#13-uplift-modeling--heterogeneous-treatment-effects)
14. [Experimentation in ML Systems](#14-experimentation-in-ml-systems)
15. [Common Interview Questions](#15-common-interview-questions)

---

## 1. Correlation vs Causation

### The Ice Cream and Drowning Problem

Every summer, ice cream sales spike. So do drowning deaths. If you plotted them on a graph, the correlation is striking — almost suspiciously tidy. Does eating ice cream make you drown?

Obviously not. Both are driven by a **confounder**: hot weather. People eat more ice cream when it's hot. People swim more when it's hot. The relationship between ice cream and drowning is entirely explained by a third variable that causes both.

This is the canonical example of **spurious correlation** — a statistical relationship that has no causal interpretation.

### Shoe Size and Vocabulary

Children with larger shoe sizes tend to have larger vocabularies. This sounds absurd until you realize the confounder is **age**. Older children have bigger feet and know more words. Shoe size doesn't cause vocabulary growth; age drives both.

### Why This Matters in Practice

In ML, you constantly deal with observational data — user logs, transaction histories, sensor readings. These datasets are riddled with correlations that aren't causal. A model trained to predict "will a user churn?" might learn that users who contact customer support are more likely to churn. That's correlation. But if you intervene by *forcing* all users to contact support, churn won't go down — it might go up.

**Predictive models learn associations. Causal models learn mechanisms.**

The distinction matters enormously when you're making decisions:
- "Which users will churn?" → predictive, correlation is fine
- "What should we do to prevent churn?" → causal, you need intervention effects

### Formal Definition

The correlation between X and Y:

```
Corr(X, Y) = Cov(X, Y) / (σ_X · σ_Y)
```

Correlation tells you: "when X is high, is Y typically high too?" It says nothing about what happens to Y if you *set* X to a particular value.

Causation asks: "if I intervene and change X, what happens to Y?" This is the difference between observing the world and acting on it.

---

## 2. The Fundamental Problem of Causal Inference

### The Counterfactual Framework

Imagine a clinical trial for a new drug. Patient Alice gets the drug and recovers. Did the drug cure her? To answer this definitively, you'd need to know what would have happened to Alice if she *hadn't* taken the drug. That alternative timeline — Alice without the drug — is called the **counterfactual**.

The fundamental problem: **you can never observe both the treated and untreated outcome for the same individual at the same time.**

Formally, let:
- `Y_i(1)` = outcome for individual i if treated
- `Y_i(0)` = outcome for individual i if untreated
- `T_i` = 1 if treated, 0 if not

The **Individual Treatment Effect (ITE)** would be:
```
ITE_i = Y_i(1) - Y_i(0)
```

But we only ever observe one of these. If Alice is treated, we see `Y_Alice(1)`. We never see `Y_Alice(0)`. This missing data problem is inherent to causal inference — it's not a limitation of your dataset or method, it's a fundamental logical constraint.

### What We Can Estimate Instead

Since individual effects are unobservable, we target population-level quantities:

**Average Treatment Effect (ATE)**:
```
ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]
```

**Average Treatment Effect on the Treated (ATT)**:
```
ATT = E[Y(1) - Y(0) | T = 1]
```

ATT answers: "For people who actually received the treatment, how much did it help?" This is often more policy-relevant than ATE, because if a treatment has side effects, you care most about whether it helped the people you gave it to.

### The Selection Problem

If you naively compare treated and untreated groups in observational data:
```
E[Y | T=1] - E[Y | T=0]
```

This is not ATE. It's ATE *plus* selection bias:
```
E[Y | T=1] - E[Y | T=0] = ATE + E[Y(0) | T=1] - E[Y(0) | T=0]
```

The second term is selection bias — the difference in baseline outcomes between the groups. Healthier people are more likely to take vitamins. Wealthier people are more likely to use premium services. The treated and control groups are systematically different before treatment ever begins.

Randomization eliminates selection bias by ensuring that treatment assignment is independent of potential outcomes. That's why randomized controlled trials are the gold standard.

---

## 3. A/B Testing Framework

A/B testing is the workhorse of causal inference in tech. You split users randomly into a control group (A) and a treatment group (B), change one thing, and measure the effect. Because assignment is random, any difference in outcomes is causally attributable to the change.

### Setting Up the Experiment

**Define the unit of randomization.** Usually this is a user, but it could be a session, a device, or a geographic region. The unit should be stable and the right level of granularity for your question.

**Define the metric.** Be specific before you run the experiment. "We want to improve engagement" is not a metric. "We want to increase the 7-day retention rate among new users" is.

**Define success before you start.** If you define success after seeing the data, you'll fool yourself.

### Hypothesis Testing

The framework:

1. State the **null hypothesis H₀**: usually "the treatment has no effect"
   - Example: "The new button color does not change click-through rate"
2. State the **alternative hypothesis H₁**: "the treatment has an effect" (two-sided) or "the treatment increases CTR" (one-sided)
3. Collect data
4. Compute a test statistic
5. Decide whether to reject H₀

The key insight: hypothesis testing is designed to **control false positives**. You're asking: "if the null were true, how unlikely is this data?"

### P-Values: What They Actually Mean

The p-value is the probability of observing data at least as extreme as what you saw, **assuming the null hypothesis is true**.

```
p-value = P(data at least as extreme as observed | H₀ is true)
```

What it is NOT:
- Not the probability that H₀ is true
- Not the probability that your result is due to chance
- Not the probability that you made an error

If p = 0.03, it means: "If there were truly no effect, we'd see a result this extreme or more extreme only 3% of the time." It's a measure of how surprising the data is under the null.

A common misconception: "p < 0.05 means there's a 95% chance the effect is real." This is wrong. P-values say nothing about the probability of hypotheses — that's Bayesian territory.

### Statistical Significance

A result is **statistically significant** at level α if p < α. The conventional choice is α = 0.05, meaning you accept a 5% false positive rate.

This is arbitrary. In medicine, α = 0.01 or smaller is common. In physics, they use α = 0.0000003 (5-sigma). In many tech settings, α = 0.05 is fine because the cost of a false positive is low.

For a two-sample proportion test (e.g., comparing click rates):

```
z = (p̂₁ - p̂₂) / SE

SE = sqrt(p̂(1-p̂)(1/n₁ + 1/n₂))

where p̂ = (x₁ + x₂) / (n₁ + n₂)  # pooled proportion
```

Reject H₀ if |z| > z_{α/2} (for two-sided test). At α = 0.05, z_{0.025} = 1.96.

### Type I and Type II Errors

|  | H₀ is true | H₀ is false |
|---|---|---|
| **Reject H₀** | Type I Error (False Positive) | Correct (True Positive) |
| **Fail to reject H₀** | Correct (True Negative) | Type II Error (False Negative) |

**Type I Error (False Positive)**:
- You conclude the treatment works when it doesn't
- Probability = α (significance level)
- Example: Launching a new button color that actually has no effect, wasting engineering resources

**Type II Error (False Negative)**:
- You conclude the treatment doesn't work when it does
- Probability = β
- Example: Killing a feature that would have increased revenue by 5%

**Power** = 1 - β = probability of detecting a real effect.

The tension: to reduce Type I errors, make α smaller. But with a fixed sample size, reducing α increases β (more false negatives). To reduce both simultaneously, you need more data.

### Power Calculations and Sample Size

Before running an experiment, you need to determine how many users you need. The inputs are:

- **α**: significance level (typically 0.05)
- **Power (1-β)**: typically 0.80 or 0.90 (80-90% chance of detecting a real effect)
- **Baseline rate (p₀)**: the current metric value in control
- **Minimum Detectable Effect (MDE)**: the smallest effect you care about detecting

For a two-proportion z-test, the sample size per group is approximately:

```
n = (z_{α/2} + z_β)² · [p₁(1-p₁) + p₂(1-p₂)] / (p₁ - p₂)²
```

Where z_{α/2} = 1.96 (for α=0.05, two-sided) and z_β = 0.84 (for 80% power).

```python
import numpy as np
from scipy import stats

def sample_size_two_proportions(
    p_control: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True
) -> int:
    """
    Calculate required sample size per group for a two-proportion z-test.
    
    Args:
        p_control: baseline conversion rate in control group
        mde: minimum detectable effect (absolute, e.g. 0.02 for 2pp lift)
        alpha: significance level
        power: desired statistical power
        two_sided: whether to use a two-sided test
    
    Returns:
        Required sample size per group (round up)
    """
    p_treatment = p_control + mde
    
    # z-scores
    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    
    # pooled proportion under H0
    p_pooled = (p_control + p_treatment) / 2
    
    # sample size formula
    numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
                 z_beta * np.sqrt(p_control * (1 - p_control) + p_treatment * (1 - p_treatment))) ** 2
    denominator = (p_treatment - p_control) ** 2
    
    n = numerator / denominator
    return int(np.ceil(n))


# Example: button color A/B test
# Current CTR = 10%, want to detect a 2pp lift (10% -> 12%)
n = sample_size_two_proportions(
    p_control=0.10,
    mde=0.02,
    alpha=0.05,
    power=0.80
)
print(f"Required sample size per group: {n:,}")
# Output: Required sample size per group: 3,524

# Doubling the MDE to 4pp drastically reduces required sample size
n_large_mde = sample_size_two_proportions(p_control=0.10, mde=0.04)
print(f"With 4pp MDE: {n_large_mde:,} per group")
# Output: With 4pp MDE: 944 per group
```

**Key intuition**: sample size scales with 1/MDE². If you want to detect an effect that's half as small, you need 4x as many users. This is why having a clear MDE matters — vague goals lead to underpowered experiments.

### Effect Size: Cohen's d and Relative Lift

**Cohen's d** measures effect size for continuous outcomes in units of standard deviations:

```
d = (μ₁ - μ₂) / σ_pooled

σ_pooled = sqrt((s₁² + s₂²) / 2)
```

Guidelines (Cohen, 1988):
- d = 0.2: small effect
- d = 0.5: medium effect
- d = 0.8: large effect

**Relative lift** is more intuitive for business metrics:
```
Relative lift = (metric_treatment - metric_control) / metric_control
```

Example: Control CTR = 10%, Treatment CTR = 11%
- Absolute lift = 1 percentage point
- Relative lift = 1% / 10% = 10%

Be careful about which one you report. "10% improvement" sounds much better than "1 percentage point improvement" but they describe the same result. Always clarify which you mean.

```python
def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d for two independent groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def run_ab_test(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05):
    """Run a two-sample t-test and report results."""
    t_stat, p_value = stats.ttest_ind(control, treatment)
    d = cohens_d(treatment, control)
    relative_lift = (np.mean(treatment) - np.mean(control)) / np.mean(control)
    
    print(f"Control mean:    {np.mean(control):.4f}")
    print(f"Treatment mean:  {np.mean(treatment):.4f}")
    print(f"Relative lift:   {relative_lift:.2%}")
    print(f"Cohen's d:       {d:.3f}")
    print(f"t-statistic:     {t_stat:.3f}")
    print(f"p-value:         {p_value:.4f}")
    print(f"Significant:     {p_value < alpha}")


# Simulate an A/B test on page load time (lower is better)
np.random.seed(42)
control_times = np.random.normal(loc=3.5, scale=1.2, size=1000)   # 3.5s avg
treatment_times = np.random.normal(loc=3.2, scale=1.2, size=1000)  # 3.2s avg

run_ab_test(control_times, treatment_times)
```

---

## 4. Common A/B Testing Pitfalls

These pitfalls are responsible for most bad A/B test decisions in practice. Interviewers love asking about them.

### Peeking / Early Stopping

Imagine you're running an A/B test and you check the results daily. On day 3, p = 0.04 — significant! You stop the experiment and declare victory.

This is **peeking**, and it massively inflates your false positive rate.

Why? Every time you look at the data and make a decision, you're running a hypothesis test. If you check 20 times over the course of an experiment and stop whenever p < 0.05, you're not running at 5% false positive rate — you're running at something closer to 30-40%.

The p-value is only valid at a pre-specified sample size. Looking at it before that point and making decisions breaks the frequentist guarantees.

**Solutions**:
1. **Pre-commit to a sample size** and only look at the final result
2. **Sequential testing** with alpha-spending functions (e.g., O'Brien-Fleming): allows interim analysis while controlling overall Type I error
3. **Always-valid p-values** (e.g., mixture sequential ratio tests): valid at any stopping time
4. **Bayesian methods**: naturally handle continuous monitoring (covered in section 5)

```python
# Illustration of peeking inflation
import numpy as np

def simulate_peeking(n_simulations=10000, max_n=1000, check_every=10, alpha=0.05):
    """
    Simulate how often we incorrectly reject H0 when peeking.
    Under H0, both groups are identical (no real effect).
    """
    false_positives = 0
    
    for _ in range(n_simulations):
        control = []
        treatment = []
        rejected = False
        
        for i in range(check_every, max_n + 1, check_every):
            # Add new observations (null effect: both from same distribution)
            control.extend(np.random.normal(0, 1, check_every).tolist())
            treatment.extend(np.random.normal(0, 1, check_every).tolist())
            
            # Peek: run t-test
            _, p = stats.ttest_ind(control, treatment)
            if p < alpha:
                rejected = True
                break
        
        if rejected:
            false_positives += 1
    
    return false_positives / n_simulations

# This will return ~0.30-0.40 instead of 0.05
# (too slow to run here, but the concept is clear)
```

### Multiple Testing Correction

You're launching a redesigned checkout page and you measure 10 metrics: CTR, conversion rate, revenue per user, time on page, bounce rate, return visits, cart abandonment, customer satisfaction, support tickets, and refund rate.

Even if the redesign does nothing, at α = 0.05, you'd expect to see 0.5 "significant" results purely by chance. If you run 100 experiments, 5 will appear significant even if all null effects are true.

**Family-Wise Error Rate (FWER)** = probability of at least one false positive across all tests.

**Bonferroni Correction**: divide α by the number of tests:
```
α_adjusted = α / m

where m = number of tests
```

Simple and conservative. For 10 tests at α = 0.05: each test uses α = 0.005.

**False Discovery Rate (FDR)** — Benjamini-Hochberg procedure:

More powerful than Bonferroni. Controls the *expected proportion* of rejected hypotheses that are false positives.

```python
from statsmodels.stats.multitest import multipletests

# Example: 10 p-values from testing different metrics
p_values = [0.001, 0.04, 0.08, 0.12, 0.20, 0.35, 0.40, 0.52, 0.68, 0.90]

# Bonferroni
_, bonferroni_corrected, _, _ = multipletests(p_values, method='bonferroni', alpha=0.05)

# Benjamini-Hochberg (FDR)
_, bh_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)

print("P-value | Bonferroni | BH (FDR)")
print("-" * 40)
for p, b, bh in zip(p_values, bonferroni_corrected, bh_corrected):
    print(f"  {p:.3f}  |   {'Yes' if b else 'No ':3}    |   {'Yes' if bh else 'No'}")
```

Use Bonferroni when one false positive is very costly (e.g., a medical treatment with side effects). Use BH when you're doing exploratory analysis and the cost of missing a real effect is high.

### Simpson's Paradox

The Berkeley admissions case (1973): University of California Berkeley was sued for gender bias. Looking at overall admissions, men were admitted at 44% and women at 35%. Seems like bias.

But when you look at each department separately, women had *higher* or equal admission rates in almost every department. How?

Women applied in higher proportions to competitive departments (English, law) with low overall admission rates. Men applied more to STEM departments with high admission rates. The aggregation reversed the direction of the relationship.

**Simpson's Paradox**: a trend that appears in aggregated data reverses or disappears when you disaggregate by a confounding variable.

In A/B testing, this appears when:
- Your treatment and control groups differ in composition (e.g., different mix of mobile vs desktop)
- The metric differs across subgroups
- The subgroup distribution correlates with treatment assignment

**Detection**: always segment your results by key dimensions (device, geography, user cohort, etc.). If the aggregated result and segmented results tell different stories, you have Simpson's Paradox.

**Fix**: control for the confounding variable in your analysis, or ensure your randomization is stratified.

### Network Effects / SUTVA Violation

The **Stable Unit Treatment Value Assumption (SUTVA)** assumes:
1. The treatment effect for unit i doesn't depend on the treatment of unit j
2. There's only one version of the treatment

SUTVA is violated when users interact with each other — which is basically every social network.

Example: You're testing a new notification system on Twitter. Some users are in treatment (get new notifications) and some are in control (old notifications). But users interact! A user in control might see increased activity from their treatment-group friends and become more active themselves — contaminating the control group.

This leads to **underestimating treatment effects**: the control group gets "infected" by the treatment, making the difference look smaller than it is.

**Solutions**:
- **Cluster randomization**: randomize by cluster (geographic region, social community) rather than individual user. The downside is fewer effective units, reducing power.
- **Ego-network experiments**: randomize at the level of connected components
- **Time-based experiments**: run A/B test at different time periods (dangerous: confounds with time trends)
- **Two-sided marketplace experiments**: geo-holdout tests

### Novelty Effects

When you launch a new feature, users notice the change and interact with it out of curiosity — not because it's genuinely better. CTR might spike initially just because it's new.

Conversely, users habituated to the old experience might initially perform worse on a new one even if it's objectively better (**change aversion**).

Both bias your short-term experiment results.

**Detection**: look at the treatment effect over time within the experiment. Does the lift diminish as users acclimate? Plot daily or weekly treatment effect estimates.

**Fix**: run experiments for longer (2-4 weeks minimum for products with daily active users). Or focus on "holdback" users — users who've been on the new experience long enough to have adapted.

---

## 5. Bayesian A/B Testing

Frequentist A/B testing answers: "How often would I get a result this extreme if H₀ were true?" Bayesian testing answers the question you actually care about: "Given the data, what's the probability that variant B is better than A?"

### The Setup

You model the conversion rates as random variables with prior distributions. After observing data, you update to posterior distributions using Bayes' theorem.

For a binary metric (conversion rate), the natural choice is a Beta-Binomial model:
- Prior: `θ ~ Beta(α, β)`
- Likelihood: `x | θ ~ Binomial(n, θ)`
- Posterior: `θ | x ~ Beta(α + x, β + n - x)`

The Beta distribution is the conjugate prior for the Binomial, which means the posterior has the same form as the prior — convenient and computationally free.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

def bayesian_ab_test(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 100_000
) -> dict:
    """
    Bayesian A/B test using Beta-Binomial conjugate model.
    Returns probability that treatment > control.
    """
    # Posterior parameters
    alpha_control = prior_alpha + control_conversions
    beta_control = prior_beta + (control_total - control_conversions)
    
    alpha_treatment = prior_alpha + treatment_conversions
    beta_treatment = prior_beta + (treatment_total - treatment_conversions)
    
    # Monte Carlo: sample from both posteriors
    samples_control = beta_dist.rvs(alpha_control, beta_control, size=n_samples)
    samples_treatment = beta_dist.rvs(alpha_treatment, beta_treatment, size=n_samples)
    
    prob_treatment_better = np.mean(samples_treatment > samples_control)
    
    # Expected lift
    expected_lift = np.mean(samples_treatment - samples_control)
    
    # Credible interval for lift
    lift_samples = samples_treatment - samples_control
    ci_low, ci_high = np.percentile(lift_samples, [2.5, 97.5])
    
    return {
        "prob_treatment_better": prob_treatment_better,
        "expected_lift": expected_lift,
        "credible_interval_95": (ci_low, ci_high),
        "posterior_control": (alpha_control, beta_control),
        "posterior_treatment": (alpha_treatment, beta_treatment),
    }


# Example: 100 conversions / 1000 visitors in control
#          115 conversions / 1000 visitors in treatment
result = bayesian_ab_test(
    control_conversions=100, control_total=1000,
    treatment_conversions=115, treatment_total=1000
)

print(f"P(treatment > control): {result['prob_treatment_better']:.3f}")
print(f"Expected lift: {result['expected_lift']:.4f}")
print(f"95% credible interval: {result['credible_interval_95']}")
```

### Advantages over Frequentist Testing

1. **No peeking problem**: the posterior is valid at any sample size. You can check it every day.
2. **Intuitive interpretation**: "There's a 94% probability that the treatment is better" is what everyone *wants* to say about a p-value, but can't.
3. **Incorporates prior knowledge**: if you know from past experiments that effects tend to be small, you can encode that as a prior and avoid being fooled by noise.
4. **Quantifies magnitude**: you get a posterior distribution over the lift, not just a binary significant/not-significant.

### Expected Loss Framework

Instead of probability of being better, you can frame decisions in terms of **expected loss**:

```
Loss(choose A | B is better) = E[max(0, θ_B - θ_A)]
Loss(choose B | A is better) = E[max(0, θ_A - θ_B)]
```

Stop the experiment and choose B when expected loss from choosing B falls below a threshold (e.g., 0.001 relative to baseline). This naturally balances exploration and exploitation.

---

## 6. Multi-Armed Bandits

A/B testing is wasteful: you spend half your traffic on the losing variant throughout the experiment. Multi-armed bandits (MAB) address this by dynamically allocating more traffic to better-performing variants.

The name comes from the "one-armed bandit" slot machine problem: you have K slot machines (arms), each with an unknown reward distribution. How do you maximize total reward while figuring out which machine pays best?

### The Exploration-Exploitation Tradeoff

- **Exploitation**: pull the arm with the highest known expected reward
- **Exploration**: try arms you haven't pulled much to reduce uncertainty

Pure exploitation gets stuck on suboptimal arms early. Pure exploration wastes pulls on arms you already know are bad.

### Thompson Sampling

Thompson Sampling is a Bayesian algorithm that naturally balances exploration and exploitation:

1. Maintain a Beta posterior for each arm's conversion rate
2. Each round: sample once from each arm's posterior
3. Pull the arm with the highest sampled value

```python
import numpy as np
from scipy.stats import beta as beta_dist

class ThompsonSampling:
    """
    Thompson Sampling for Bernoulli bandits.
    Each arm has a Beta posterior over its conversion rate.
    """
    
    def __init__(self, n_arms: int, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.n_arms = n_arms
        self.alpha = np.full(n_arms, prior_alpha)
        self.beta = np.full(n_arms, prior_beta)
        self.pulls = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)
    
    def select_arm(self) -> int:
        """Sample from each posterior and select the highest."""
        samples = beta_dist.rvs(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, arm: int, reward: int):
        """Update the posterior for the selected arm."""
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
    
    def get_conversion_rates(self) -> np.ndarray:
        """Return posterior mean for each arm."""
        return self.alpha / (self.alpha + self.beta)


def simulate_bandit(true_rates: list, n_rounds: int = 10000):
    """Simulate Thompson Sampling against a fixed-split A/B test."""
    n_arms = len(true_rates)
    ts = ThompsonSampling(n_arms)
    
    ts_rewards = []
    ab_rewards = []
    
    for t in range(n_rounds):
        # Thompson Sampling
        arm = ts.select_arm()
        reward = int(np.random.random() < true_rates[arm])
        ts.update(arm, reward)
        ts_rewards.append(reward)
        
        # Fixed 50/50 A/B (for 2 arms)
        ab_arm = t % n_arms
        ab_reward = int(np.random.random() < true_rates[ab_arm])
        ab_rewards.append(ab_reward)
    
    print(f"True rates: {true_rates}")
    print(f"Thompson Sampling total reward: {sum(ts_rewards):,}")
    print(f"Equal-split A/B total reward:   {sum(ab_rewards):,}")
    print(f"TS arm allocation: {ts.pulls}")
    
    return ts, ts_rewards


# Two arms: control at 10%, treatment at 13%
ts_model, _ = simulate_bandit([0.10, 0.13], n_rounds=10000)
```

**Why Thompson Sampling works**: when you're uncertain about an arm, its posterior is wide, so samples from it are sometimes very high. This encourages exploration. As you pull it more, the posterior tightens and it only gets selected when its mean is genuinely high.

### Upper Confidence Bound (UCB)

A deterministic alternative to Thompson Sampling:

```
UCB_i(t) = x̄_i + sqrt(2 * ln(t) / n_i)
```

Where:
- `x̄_i` = average reward from arm i so far
- `t` = total number of rounds
- `n_i` = number of times arm i was pulled

Select the arm with the highest UCB. The second term is the uncertainty bonus — it's large when an arm has been pulled few times (high uncertainty) and shrinks as we gather data.

```python
class UCB1:
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.pulls = np.zeros(n_arms)
        self.total_reward = np.zeros(n_arms)
        self.t = 0
    
    def select_arm(self) -> int:
        self.t += 1
        # Pull each arm at least once first
        for arm in range(self.n_arms):
            if self.pulls[arm] == 0:
                return arm
        
        means = self.total_reward / self.pulls
        ucb_scores = means + np.sqrt(2 * np.log(self.t) / self.pulls)
        return np.argmax(ucb_scores)
    
    def update(self, arm: int, reward: float):
        self.pulls[arm] += 1
        self.total_reward[arm] += reward
```

### When to Use Bandits vs A/B Tests

| Criterion | A/B Test | Multi-Armed Bandit |
|---|---|---|
| Goal | Inference (what is the effect?) | Optimization (maximize reward) |
| Traffic to losers | Equal (wasteful) | Minimized |
| Statistical validity | Well-established | Harder to do valid inference |
| Multiple variants | Hard (multiple testing) | Natural |
| Non-stationary rewards | No | Can handle with sliding windows |

Bandits are great when you have many variants (pricing, recommendations) and you care more about cumulative performance than clean inference. A/B tests are better when you need a clean causal estimate (e.g., to convince stakeholders, or for a medical trial).

---

## 7. Observational Studies and Confounders

Most real-world causal questions can't be answered with a randomized experiment:
- You can't randomly assign people to smoke
- You can't randomly assign companies to raise their minimum wage
- You can't randomly assign users to have been active for 5 years vs 1 year

In these cases you work with **observational data** — data collected without intervention — and try to adjust for confounders.

### What is a Confounder?

A confounder Z is a variable that:
1. Causes the treatment T
2. Causes the outcome Y

The causal structure: `Z → T` and `Z → Y`. This creates a backdoor path from T to Y through Z that isn't a causal path.

Example: You're studying whether LinkedIn Premium membership increases job offers. But wealthier, more senior professionals are both more likely to pay for Premium *and* more likely to receive job offers. Seniority is a confounder.

If you naively compare Premium vs non-Premium members, you'll overestimate the effect of Premium because you're also capturing the seniority effect.

### The Ignorability Assumption

For observational causal inference to work, you need the **ignorability** (or unconfoundedness) assumption:

```
(Y(0), Y(1)) ⊥ T | X
```

Conditional on observed covariates X, the treatment is as good as randomly assigned. In other words: you've measured all the confounders.

This is a strong, untestable assumption. You can never fully verify it from data alone. The best you can do is:
1. Think carefully about what confounders exist
2. Measure as many as possible
3. Run sensitivity analyses to assess how much an unmeasured confounder would need to affect things to overturn your conclusions

### Common Confounding Patterns

**Healthy user bias**: people who take health supplements also exercise more, eat better, and go to the doctor regularly. Studies on supplement effectiveness are often confounded by general health-consciousness.

**Survivorship bias**: studying the characteristics of successful companies ignores the companies that failed with the same characteristics.

**Reverse causation**: "hospitals are dangerous" (sick people go to hospitals, not hospitals making people sick).

---

## 8. Propensity Score Matching

### The Idea

If treatment assignment isn't random, make it *look* random by conditioning on confounders. The **propensity score** is the probability of being treated given observed covariates:

```
e(X) = P(T = 1 | X)
```

The **propensity score theorem** (Rosenbaum & Rubin, 1983): if treatment is ignorable given X, it's also ignorable given the propensity score:

```
(Y(0), Y(1)) ⊥ T | e(X)
```

This is powerful: instead of conditioning on high-dimensional X (which might require matching on dozens of variables), you can condition on a single scalar.

### Estimation

1. **Estimate the propensity score**: fit a logistic regression (or more flexible model like gradient boosting) predicting T from X
2. **Match treated to control units** with similar propensity scores
3. **Estimate treatment effect** on the matched sample

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: list,
    caliper: float = 0.05  # max allowed PS difference
) -> dict:
    """
    Estimate ATT via nearest-neighbor propensity score matching.
    """
    X = df[covariate_cols].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    
    # Step 1: Estimate propensity scores
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, T)
    ps = lr.predict_proba(X_scaled)[:, 1]
    
    # Step 2: Match each treated unit to closest control unit
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    
    ps_treated = ps[treated_idx].reshape(-1, 1)
    ps_control = ps[control_idx].reshape(-1, 1)
    
    # Distance matrix
    distances = cdist(ps_treated, ps_control, metric='euclidean')
    
    matched_pairs = []
    used_control = set()
    
    for i, t_idx in enumerate(treated_idx):
        # Find closest unused control within caliper
        dists = distances[i]
        sorted_ctrl = np.argsort(dists)
        
        for ctrl_rank_idx in sorted_ctrl:
            ctrl_original_idx = control_idx[ctrl_rank_idx]
            if ctrl_original_idx not in used_control and dists[ctrl_rank_idx] < caliper:
                matched_pairs.append((t_idx, ctrl_original_idx))
                used_control.add(ctrl_original_idx)
                break
    
    if not matched_pairs:
        raise ValueError("No matched pairs found. Try increasing caliper.")
    
    treated_outcomes = [Y[t] for t, c in matched_pairs]
    control_outcomes = [Y[c] for t, c in matched_pairs]
    
    att_estimate = np.mean(np.array(treated_outcomes) - np.array(control_outcomes))
    se = np.std(np.array(treated_outcomes) - np.array(control_outcomes)) / np.sqrt(len(matched_pairs))
    
    return {
        "att": att_estimate,
        "se": se,
        "n_matched": len(matched_pairs),
        "n_treated": len(treated_idx),
        "propensity_scores": ps,
    }
```

### Checking Balance

After matching, verify that treated and control groups look similar on observed covariates. The standard diagnostic is the **standardized mean difference (SMD)**:

```
SMD = (x̄_treated - x̄_control) / sqrt((s²_treated + s²_control) / 2)
```

An SMD below 0.1 is generally considered good balance. If balance is poor, try different caliper widths or use a more flexible propensity score model.

### Inverse Probability Weighting (IPW)

An alternative to matching: weight each observation by the inverse of its propensity score.

```
Weight for treated unit i: 1 / e(X_i)
Weight for control unit i: 1 / (1 - e(X_i))
```

This creates a pseudo-population where treatment is uncorrelated with covariates. The IPW estimator for ATE:

```
ATE_IPW = (1/n) * Σ [ T_i * Y_i / e(X_i) - (1-T_i) * Y_i / (1-e(X_i)) ]
```

IPW uses all data (unlike matching, which discards unmatched units) but can be unstable when propensity scores are near 0 or 1 (extreme weights).

**Doubly Robust Estimator** combines propensity score model with an outcome model — it's consistent if *either* model is correctly specified. This is best practice when you can afford to fit both.

---

## 9. Difference-in-Differences

### The Core Idea

DiD is one of the most widely used quasi-experimental designs. It exploits a policy change or natural experiment where a treatment is applied to some units but not others, at a specific point in time.

The key insight: instead of comparing treated vs control after treatment, compare the *change* in outcomes over time. This removes time-invariant confounders (as long as the parallel trends assumption holds).

```
DiD = (Ȳ_treated,post - Ȳ_treated,pre) - (Ȳ_control,post - Ȳ_control,pre)
```

Visual intuition: imagine plotting the outcome for treated and control groups over time. If both groups were trending similarly before the treatment (parallel trends), you can attribute any divergence after treatment to the treatment itself.

### Classic Example: Card and Krueger (1994)

New Jersey raised its minimum wage in 1992. Pennsylvania (neighboring state) did not. Did the minimum wage increase hurt employment in the fast food industry?

- Treatment group: NJ fast food restaurants
- Control group: PA fast food restaurants
- Pre-period: before the wage increase
- Post-period: after the wage increase

DiD estimate: employment change in NJ minus employment change in PA. Card and Krueger found a *positive* (or null) employment effect — a landmark result that challenged standard labor economics.

### The Regression Framework

DiD is typically estimated via OLS:

```
Y_it = β₀ + β₁ · Treated_i + β₂ · Post_t + β₃ · (Treated_i × Post_t) + ε_it
```

- `β₁`: baseline difference between treated and control (fixed effect for group)
- `β₂`: common time trend (fixed effect for time)
- `β₃`: **the DiD estimator** — the differential change in the treated group post-treatment

```python
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols

def simulate_did_example():
    """
    Simulate a DiD dataset and estimate the treatment effect.
    
    Scenario: An e-commerce platform runs a loyalty program in one city (treated)
    but not another (control). We observe revenue per user before and after.
    """
    np.random.seed(42)
    n_per_group = 500
    
    # Control group: pre-period revenue ~$50, grows by $5 post-period
    control_pre = np.random.normal(50, 10, n_per_group)
    control_post = np.random.normal(55, 10, n_per_group)  # +$5 natural trend
    
    # Treated group: pre-period revenue ~$50, grows by $5 (trend) + $8 (treatment)
    treated_pre = np.random.normal(50, 10, n_per_group)
    treated_post = np.random.normal(63, 10, n_per_group)  # +$13 = $5 trend + $8 treatment
    
    df = pd.DataFrame({
        'revenue': np.concatenate([control_pre, control_post, treated_pre, treated_post]),
        'treated': np.concatenate([np.zeros(n_per_group*2), np.ones(n_per_group*2)]),
        'post': np.concatenate([np.zeros(n_per_group), np.ones(n_per_group),
                                np.zeros(n_per_group), np.ones(n_per_group)])
    })
    df['treated_x_post'] = df['treated'] * df['post']
    
    # OLS DiD
    model = ols('revenue ~ treated + post + treated_x_post', data=df).fit()
    print(model.summary().tables[1])
    print(f"\nDiD estimate (β₃): {model.params['treated_x_post']:.2f}")
    print(f"True treatment effect: ~8.0")
    
    return df, model

df, model = simulate_did_example()
```

### The Parallel Trends Assumption

The critical assumption: **in the absence of treatment, treated and control groups would have followed the same time trend**.

You can't test this directly (you don't observe the counterfactual). But you can provide supporting evidence by:
1. Plotting pre-treatment trends for both groups — they should be parallel
2. Running a "placebo" DiD using earlier time periods (before treatment) — the "effect" should be zero
3. Testing for differential pre-trends using an event study specification

If pre-trends differ, DiD is biased. In that case, consider synthetic control or augmented DiD.

---

## 10. Regression Discontinuity Design

### The Core Idea

Sometimes treatment is assigned based on crossing a threshold of some running variable. Regression Discontinuity (RD) exploits this threshold as a source of quasi-random variation.

Key insight: units just below and just above the threshold are very similar (they differ only in whether they happened to cross the cutoff). Near the threshold, assignment is *as good as random*.

Classic example: a scholarship is awarded to students who score above 70 on an entrance exam. Students who scored 69 vs 71 are very similar in ability, but one group got the scholarship and one didn't. Comparing their later outcomes gives a causal estimate of the scholarship effect.

### Sharp vs Fuzzy RD

**Sharp RD**: treatment is a deterministic step function of the running variable
```
T_i = 1(X_i ≥ c)
```

**Fuzzy RD**: crossing the threshold changes the *probability* of treatment (but doesn't fully determine it). Use instrumental variables logic: the threshold is an instrument for treatment.

### The Estimand

In sharp RD, you estimate the treatment effect at the cutoff:
```
τ_RD = lim_{x↓c} E[Y | X = x] - lim_{x↑c} E[Y | X = x]
```

This is a **local** average treatment effect — only valid at the threshold. People far from the threshold might respond differently.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

def simulate_rdd():
    """
    Simulate a sharp RD design.
    Running variable: test score (0-100)
    Cutoff: 60 (above = get tutoring program)
    True effect: 5 points on final exam
    """
    np.random.seed(42)
    n = 2000
    
    score = np.random.uniform(30, 90, n)
    treatment = (score >= 60).astype(int)
    
    # Outcome: final exam score
    # Continuous relationship with running variable + discontinuous treatment effect
    noise = np.random.normal(0, 5, n)
    outcome = 40 + 0.5 * score + 5 * treatment + noise  # true effect = 5
    
    df = pd.DataFrame({'score': score, 'treatment': treatment, 'outcome': outcome})
    df['score_centered'] = df['score'] - 60
    
    # Local linear regression on both sides of cutoff
    bandwidth = 10  # use only units within 10 points of cutoff
    df_local = df[np.abs(df['score_centered']) <= bandwidth].copy()
    
    model = ols(
        'outcome ~ score_centered * treatment',
        data=df_local
    ).fit()
    
    rdd_estimate = model.params['treatment']
    print(f"RDD estimate: {rdd_estimate:.2f}")
    print(f"True effect: 5.00")
    
    return df, model


# Key diagnostics for RDD:
# 1. Density test (McCrary test): no bunching at cutoff (would suggest manipulation)
# 2. Covariate balance at cutoff: predetermined covariates shouldn't jump at cutoff
# 3. Placebo cutoffs: no discontinuity at other values of the running variable
# 4. Sensitivity to bandwidth choice
```

### Bandwidth Selection

The bandwidth (how far from the cutoff you include) involves a bias-variance tradeoff:
- Wider bandwidth → more data → lower variance, but units are less comparable → higher bias
- Narrower bandwidth → less bias, but fewer observations → higher variance

Optimal bandwidth selection algorithms (Imbens-Kalyanaraman, Calonico-Cattaneo-Titiunik) choose the bandwidth that minimizes mean squared error.

---

## 11. Instrumental Variables

### The Problem IV Solves

Suppose you want to estimate the effect of education on earnings. You can't randomize education. And there are many confounders (family wealth, ability, ambition) that are hard to measure.

IV finds a **natural experiment** embedded in the data: a variable that affects the treatment (education) but affects the outcome (earnings) *only through* the treatment.

### The Two Conditions

A valid instrument Z must satisfy:

1. **Relevance**: Z is correlated with T
   - Testable: check first-stage F-statistic (rule of thumb: F > 10)
   - `Cov(Z, T) ≠ 0`

2. **Exclusion restriction**: Z affects Y only through T
   - NOT directly testable — requires subject matter knowledge and a compelling argument
   - `Z ⊥ Y | T` (once we condition on T, Z tells us nothing about Y)

3. **Independence**: Z is independent of confounders
   - `Z ⊥ U` (Z is "as good as randomly assigned")

### Classic Examples

**Angrist (1990) — Vietnam draft lottery**: Is military service bad for earnings? The draft lottery was truly random, making it a valid instrument. Those assigned a low draft number were more likely to serve — but lottery number itself doesn't directly affect earnings.

**Card (1995) — College proximity**: Does college education increase earnings? Instrument: whether a person grew up near a 4-year college. This increases the probability of attending college but doesn't directly affect earnings.

### Two-Stage Least Squares (2SLS)

The standard estimator for IV:

**Stage 1**: Regress T on Z (and covariates X):
```
T̂ = α₀ + α₁Z + α₂X + ν
```

**Stage 2**: Regress Y on T̂ (predicted treatment):
```
Y = β₀ + β₁T̂ + β₂X + ε
```

The IV estimate `β₁` is the LATE — Local Average Treatment Effect — the effect for **compliers**: units whose treatment status is changed by the instrument.

```python
from linearmodels.iv import IV2SLS
import pandas as pd
import numpy as np

def simulate_iv_example():
    """
    Simulated IV example: effect of tutoring on test scores.
    Instrument: random scholarship offer (which causes tutoring)
    Confounder: student ability (unobserved)
    """
    np.random.seed(42)
    n = 1000
    
    # Unobserved confounder: ability
    ability = np.random.normal(0, 1, n)
    
    # Instrument: scholarship offer (random)
    scholarship = np.random.binomial(1, 0.5, n)
    
    # Treatment: tutoring (affected by scholarship and ability)
    noise_t = np.random.normal(0, 1, n)
    tutoring_latent = 0.8 * scholarship + 0.5 * ability + noise_t
    tutoring = (tutoring_latent > 0).astype(int)
    
    # Outcome: test score (affected by tutoring and ability)
    noise_y = np.random.normal(0, 2, n)
    test_score = 60 + 5 * tutoring + 10 * ability + noise_y  # true effect = 5
    
    df = pd.DataFrame({
        'test_score': test_score,
        'tutoring': tutoring,
        'scholarship': scholarship,
        'const': 1.0
    })
    
    # IV estimation
    iv_model = IV2SLS(
        dependent=df['test_score'],
        exog=df[['const']],
        endog=df[['tutoring']],
        instruments=df[['scholarship']]
    ).fit()
    
    print(f"IV estimate: {iv_model.params['tutoring']:.2f}")
    print(f"True effect: 5.00")
    
    # Naive OLS (biased by ability confounder)
    from statsmodels.formula.api import ols
    ols_model = ols('test_score ~ tutoring', data=df).fit()
    print(f"OLS estimate (biased): {ols_model.params['tutoring']:.2f}")
    
    return iv_model


# Key caveats:
# 1. IV identifies LATE (effect for compliers), not ATE
# 2. Weak instrument (low first-stage F) leads to large IV variance
# 3. Exclusion restriction is untestable — the most common criticism
```

### Weak Instruments

If the instrument barely predicts treatment (weak first stage), the IV estimate has huge variance and is sensitive to small violations of the exclusion restriction. Always report the first-stage F-statistic. If F < 10, you have a weak instrument problem.

---

## 12. Causal Graphs (DAGs)

### What is a DAG?

A **Directed Acyclic Graph (DAG)** is a formal representation of causal relationships. Nodes represent variables; directed edges (`A → B`) represent "A causally affects B." Acyclic means no circular causation (A → B → A is not allowed).

DAGs are powerful because they tell you exactly which variables you need to control for to estimate a causal effect — and which you should *not* control for.

### Three Types of Paths

Given a DAG, there are three structural patterns that matter:

**Chain (Mediation)**: `A → B → C`
- A causes B which causes C
- B is a **mediator**: it lies on the causal path from A to C
- If you condition on B, you block the causal path and can't estimate A's total effect on C

**Fork (Confounder)**: `A ← C → B`
- C causes both A and B
- C is a **confounder**: it creates a spurious correlation between A and B
- To estimate the causal effect of A on B, you *must* condition on C

**Collider**: `A → C ← B`
- Both A and B cause C
- C is a **collider**: by default, it doesn't create a spurious correlation between A and B
- If you condition on C (or a descendant of C), you *induce* a spurious correlation between A and B — **conditioning on a collider opens a backdoor path**

### d-Separation

d-separation is the graphical criterion for reading conditional independencies from a DAG.

Two nodes X and Y are d-separated by a set Z if all paths between them are **blocked**. A path is blocked if:
- It contains a chain `A → B → C` or fork `A ← B → C` where the middle node B is in Z
- It contains a collider `A → B ← C` where B is NOT in Z (and no descendant of B is in Z)

If X and Y are d-separated by Z, then `X ⊥ Y | Z` (they are conditionally independent).

### The Backdoor Criterion

To estimate the causal effect of T on Y, you need to block all **backdoor paths** — paths from T to Y that start with an arrow *into* T (i.e., that don't go through a descendant of T).

A set Z satisfies the backdoor criterion if:
1. Z blocks all backdoor paths from T to Y
2. Z contains no descendant of T

If the backdoor criterion is satisfied, conditioning on Z gives the causal effect:
```
P(Y | do(T=t)) = Σ_z P(Y | T=t, Z=z) P(Z=z)
```

This is the **backdoor adjustment formula** — it says how to go from observational distributions to interventional distributions.

### do-Calculus (Intuition)

Pearl's **do-calculus** is a formal set of rules for transforming expressions involving `do(·)` (interventions) into expressions involving only ordinary conditional probabilities.

The key notation: `P(Y | do(T = t))` means "the distribution of Y when we **intervene** and set T to t" — as opposed to `P(Y | T = t)` which means "the distribution of Y when we **observe** T = t."

The difference: when we intervene, we cut all arrows into T (we sever the influence of confounders). When we observe T = t, we select the subpopulation where T happened to be t (confounders are still active).

Three rules of do-calculus (intuition):
1. **Insertion/deletion of observations**: under certain graph conditions, we can add or remove conditioning variables
2. **Action/observation exchange**: under certain conditions, `do(T)` can be replaced by observing T
3. **Insertion/deletion of actions**: under certain conditions, actions can be removed entirely

Full do-calculus is sufficient to solve any identifiable causal query. If a query isn't solvable by do-calculus, it's not non-parametrically identified from observational data.

```python
# Illustration: detecting when we should NOT condition on a variable
# Scenario: hiring decision (H), gender (G), qualification (Q)
# Bad practice: conditioning on a collider

# True structure:
# G → H ← Q (both gender and qualification affect hiring)
# G and Q are marginally independent

import numpy as np
from scipy.stats import pointbiserialr

np.random.seed(42)
n = 10000

# Gender (0=male, 1=female) and qualification — truly independent
gender = np.random.binomial(1, 0.5, n)
qualification = np.random.normal(0, 1, n)

# Hiring: function of gender and qualification (with some bias)
hire_prob = 1 / (1 + np.exp(-(qualification - 0.3 * gender)))
hired = np.random.binomial(1, hire_prob, n)

# Marginally: gender and qualification are independent
corr_marginal, p_marginal = pointbiserialr(gender, qualification)
print(f"Marginal correlation G-Q: {corr_marginal:.4f} (p={p_marginal:.3f})")

# Among hired employees (conditioning on collider): spurious correlation!
mask = hired == 1
corr_conditional, p_conditional = pointbiserialr(gender[mask], qualification[mask])
print(f"Conditional correlation G-Q | hired: {corr_conditional:.4f} (p={p_conditional:.3f})")
# Now they appear negatively correlated: among hired, being female predicts lower qualification
# This is an artifact of collider bias — not a real relationship
```

---

## 13. Uplift Modeling / Heterogeneous Treatment Effects

### Beyond Average Effects

ATE is a population average. But real populations are heterogeneous — the treatment might help some people a lot, hurt others, and do nothing for the rest.

**Uplift modeling** (also called **Heterogeneous Treatment Effects** or **CATE estimation**) aims to estimate the treatment effect for specific individuals or subgroups:

```
CATE(x) = E[Y(1) - Y(0) | X = x]
```

This is incredibly actionable: instead of treating everyone, you can target only the people for whom the treatment has a positive effect.

### The Four Segments

In a marketing context, customers can be classified into:

| | Would respond without treatment | Would NOT respond without treatment |
|---|---|---|
| **Responds to treatment** | Sleeping Dog (harm) | Sure Thing (wasted spend) / **Persuadable** |
| **Doesn't respond to treatment** | Lost Cause | Lost Cause |

You want to target **Persuadables**: people who would not have converted without the treatment but will with it. Targeting Sure Things wastes budget. Targeting Lost Causes or Sleeping Dogs is futile or harmful.

### Meta-Learner Approaches

**S-Learner** (Single model):
- Fit one model with treatment indicator as a feature: `μ(X, T)`
- CATE: `τ(x) = μ(x, 1) - μ(x, 0)`
- Problem: regularization may shrink the treatment coefficient, understating heterogeneity

**T-Learner** (Two models):
- Fit separate outcome models for treated and control: `μ₁(X)` and `μ₀(X)`
- CATE: `τ(x) = μ₁(x) - μ₀(x)`
- Problem: high variance when treated/control samples are small

**X-Learner** (for imbalanced treatment groups):
1. Fit `μ₁(X)` on treated, `μ₀(X)` on control
2. Compute "imputed treatment effects":
   - For treated: `D₁ᵢ = Y_i - μ₀(X_i)` (actual outcome minus predicted control outcome)
   - For control: `D₀ᵢ = μ₁(X_i) - Y_i` (predicted treated outcome minus actual outcome)
3. Fit CATE models on each set: `τ₁(X)` from treated, `τ₀(X)` from control
4. Combine: `τ(x) = e(x) · τ₀(x) + (1-e(x)) · τ₁(x)` where `e(x)` is the propensity score

```python
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

class TLearner:
    """T-Learner for CATE estimation."""
    
    def __init__(self, base_model=None):
        self.model_treated = base_model or GradientBoostingRegressor(n_estimators=100)
        self.model_control = GradientBoostingRegressor(n_estimators=100)
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        treated_mask = T == 1
        self.model_treated.fit(X[treated_mask], Y[treated_mask])
        self.model_control.fit(X[~treated_mask], Y[~treated_mask])
        return self
    
    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self.model_treated.predict(X) - self.model_control.predict(X)


class SLearner:
    """S-Learner for CATE estimation."""
    
    def __init__(self, base_model=None):
        self.model = base_model or GradientBoostingRegressor(n_estimators=100)
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        XT = np.column_stack([X, T])
        self.model.fit(XT, Y)
        return self
    
    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        XT_treated = np.column_stack([X, np.ones(n)])
        XT_control = np.column_stack([X, np.zeros(n)])
        return self.model.predict(XT_treated) - self.model.predict(XT_control)


# Evaluate uplift models using Qini curves / AUUC
def qini_curve(y: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray):
    """
    Compute Qini curve: cumulative incremental outcome vs fraction targeted.
    Higher AUUC (Area Under Uplift Curve) = better uplift model.
    """
    df = pd.DataFrame({'y': y, 't': treatment, 'score': uplift_score})
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    n = len(df)
    qini = np.zeros(n + 1)
    
    for i in range(n):
        subset = df.iloc[:i+1]
        n_t = subset['t'].sum()
        n_c = len(subset) - n_t
        
        if n_t > 0 and n_c > 0:
            qini[i+1] = (subset[subset['t']==1]['y'].sum() / n_t - 
                         subset[subset['t']==0]['y'].sum() / n_c) * n_t
        
    return qini
```

### Causal Forests

Causal forests (Wager & Athey, 2018) extend random forests for CATE estimation. At each split, they optimize for **heterogeneity in treatment effects** rather than prediction accuracy. The key features:

- **Honesty**: uses different subsamples for building tree structure vs estimating effects (reduces overfitting)
- **Local centering**: residualizes Y and T against their predictions from forest to reduce confounding
- **Variance estimation**: provides confidence intervals for CATE estimates

The `econml` and `causalml` Python libraries implement causal forests and other CATE estimators.

---

## 14. Experimentation in ML Systems

### Model A/B Tests (Online Evaluation)

Testing a new ML model is different from testing a UI change:
- The "treatment" is a different model, not a visible feature
- Models can interact with each other (e.g., recommendation models affect what items exist to rank)
- Effects can be non-linear and hard to predict offline

The framework is the same: randomly split traffic, serve predictions from model A to one group and model B to another, measure business metrics.

**Key challenge**: latency. If model B is more accurate but 100ms slower, the latency itself is a treatment effect. Always measure latency alongside accuracy metrics.

### Offline vs Online Metrics Gap

This is one of the most important (and frustrating) problems in applied ML.

An offline metric is computed on a fixed held-out dataset: RMSE, AUC-ROC, precision@k. An online metric is computed from live user behavior: CTR, conversion rate, session length.

Models that improve offline metrics *often do not improve online metrics*, and vice versa.

**Reasons for the gap**:
1. **Distribution shift**: the held-out set doesn't represent future production data
2. **Label leakage**: offline labels may be proxies for, not the same as, what users actually want
3. **Feedback loops**: the model affects user behavior which affects future training data (not captured offline)
4. **Position bias in labels**: items shown in position 1 get more clicks regardless of quality; offline evaluation may not account for this
5. **Missing counterfactuals**: you can only evaluate on items that were shown, not all possible items

**Best practice**: always define the online metric before running offline evaluation. Design offline metrics to be predictive of the online metric. Use **interleaving experiments** for ranking models (present both models' recommendations in a single list, randomize position) for higher sensitivity.

### Shadow Mode / Champion-Challenger

**Shadow mode** (also called "dark launch"):
- New model runs in production but its decisions are not acted upon
- Used to validate infrastructure, check prediction distribution, detect errors
- Does not prove the model is better — users don't experience its outputs

**Champion-challenger**:
- Current model (champion) handles most traffic
- New model (challenger) handles a small slice (e.g., 5%)
- Monitor safety and business metrics in real-time
- If challenger is better and stable, gradually increase its traffic share (canary deployment)

The advantage of champion-challenger over full A/B test: you limit downside risk. If the challenger fails catastrophically, only 5% of users are affected.

### Interleaving for Ranking

For recommendation and search systems, standard A/B tests require huge sample sizes because ranking quality is hard to measure at the user level. **Interleaving** is far more sensitive:

1. For each request, generate ranking from model A and model B
2. Merge the rankings (various strategies: team-draft interleaving, balanced interleaving)
3. Track which model's items get clicked/engaged with

Interleaving can be 100x more sensitive than A/B testing for ranking quality, enabling faster model iteration.

### Metrics Hierarchy

Design your metric system in a hierarchy:

1. **North star metric**: the single metric that best captures long-term value (e.g., long-term revenue, DAU, 90-day retention). Hard to move but if you move it, it matters.
2. **Driver metrics**: metrics that drive the north star, faster to measure (CTR, session length, 7-day retention)
3. **Guardrail metrics**: metrics you must not decrease (latency, error rate, customer support volume)
4. **Diagnostic metrics**: help explain why driver metrics moved (click position distribution, query coverage)

An experiment should primarily optimize driver metrics without hurting guardrails. Improving the north star directly is hard and slow; usually you work through driver metrics.

---

## 15. Common Interview Questions

**Q: How would you design an A/B test for a new checkout button?**

Start with the goal (increase conversion rate). Define the unit of randomization (user, not session — to avoid the same user seeing different experiences). Define the metric (purchase conversion rate) and guardrails (latency, error rate). Compute the required sample size based on a realistic MDE (e.g., 5% relative lift on the current 3% conversion rate). Set runtime based on how long it takes to achieve that sample size plus at least one full business cycle (to account for day-of-week effects). Pre-register the analysis plan. Run the experiment, then analyze once.

---

**Q: Your A/B test shows p = 0.04. Is it significant? Should you ship?**

p = 0.04 is statistically significant at α = 0.05. But statistical significance alone doesn't mean you should ship. Ask:
- What is the effect size? A statistically significant but tiny effect might not be worth engineering cost
- What are the confidence intervals? Are they practically meaningful?
- How do guardrail metrics look?
- Was this a pre-registered analysis or were there multiple looks?
- Is the effect consistent across subgroups?
- Is the experiment well-powered? A just-significant result in an underpowered experiment is less convincing than one in a well-powered experiment

---

**Q: What is the difference between ATE, ATT, and LATE?**

- **ATE** (Average Treatment Effect): average effect across the entire population
- **ATT** (Average Treatment Effect on the Treated): average effect among those who were treated — relevant when treatment is self-selected
- **LATE** (Local Average Treatment Effect): average effect for compliers — those whose treatment status is changed by the instrument. LATE is what IV estimates; it's local to the subpopulation affected by the instrument.

If the treatment effect is homogeneous (same for everyone), ATE = ATT = LATE. With heterogeneous effects, they can differ substantially.

---

**Q: You see a 10% lift in your A/B test. How do you know if it's real?**

Checklist:
1. Was the experiment pre-registered with a sample size calculation?
2. Was there only one look at the data, or did you peek multiple times?
3. Is the result consistent across segments (device, geography, user cohort)?
4. Are there any signs of sample ratio mismatch (unequal split)?
5. Are there signs of novelty effect (does the lift decay over time within the experiment)?
6. Are guardrail metrics stable?
7. Did you run the analysis on the full pre-specified runtime?
8. If multiple metrics were tested, was there multiple testing correction?

---

**Q: Explain Simpson's Paradox. When does it occur and how do you detect it?**

Simpson's Paradox occurs when a trend observed in aggregated data reverses when you disaggregate by a confounding variable. It arises when the confounding variable is correlated with both the grouping variable and the outcome, and the groups have unequal sizes.

Detection: always segment your results. If overall and subgroup results disagree in direction, you have Simpson's Paradox.

Classic example: in Berkeley admissions, women appeared to have lower overall admission rates, but had equal or higher rates in each department. The confound was department choice — women applied to more selective departments.

---

**Q: What is the difference between propensity score matching and DiD?**

PSM is a cross-sectional method: it adjusts for observed confounders in a single time period by matching treated and control units with similar propensity scores. It requires the **ignorability assumption** — all confounders are measured.

DiD is a panel method: it uses repeated observations over time to difference out time-invariant unobserved confounders. It requires the **parallel trends assumption** — treated and control groups would have trended similarly without treatment.

DiD handles unobserved confounders (as long as they're time-invariant); PSM handles observed confounders only. You can combine both: PSM on pre-treatment covariates to improve comparability, then DiD to handle residual unobserved differences.

---

**Q: When would you use a bandit instead of an A/B test?**

Use a bandit when:
- You have many variants (pricing optimization, content selection, recommendation ranking)
- The cost of sending traffic to losing variants is high (real revenue loss)
- You care more about cumulative performance than clean statistical inference
- The system is online and decisions are frequent

Use A/B test when:
- You need a clean, defensible causal estimate (e.g., to justify a major product decision)
- You need to understand effect size and confidence intervals
- You're testing for harmful effects (safety, bias) where you need to ensure statistical rigor
- You have few variants (2-3)

---

**Q: What is SUTVA and when is it violated?**

SUTVA (Stable Unit Treatment Value Assumption) has two components:
1. No interference: the treatment of unit i does not affect the outcome of unit j
2. No hidden versions of treatment: there's only one "treatment" (not multiple forms)

It's violated on social networks (user interactions), marketplaces (supply/demand effects), household experiments (spillover within household), and geographic experiments (cross-city effects).

When SUTVA is violated, standard A/B test estimates are biased. Use cluster randomization, network experiment designs, or bipartite graph experiments.

---

**Q: Explain the parallel trends assumption in DiD. How do you test for it?**

Parallel trends assumes that in the absence of treatment, the treated and control groups would have had the same trend in the outcome. It's not directly testable (we can't observe the counterfactual trend for the treated group).

Indirect tests:
1. **Pre-trend visualization**: plot both groups' outcomes before the treatment date — they should move together
2. **Pre-trend regression test**: include time dummies × treatment interactions in the pre-period and test if they're jointly zero
3. **Placebo treatment date**: apply DiD using a fake treatment date in the pre-period and verify the "effect" is zero
4. **Falsification tests**: outcomes unaffected by the treatment should show no discontinuity at the treatment date

If pre-trends differ, consider using a synthetic control or controlling for group-specific trends.

---

**Q: What is a collider, and why is conditioning on one dangerous?**

A collider is a variable that is a common effect of two other variables: `A → C ← B`. By default, C being a collider means A and B are independent — the collider "blocks" the path.

But if you condition on C (include it as a covariate, filter on it, or stratify by it), you open the path between A and B — inducing a spurious correlation even if none exists in the population.

Example: among athletes who make a professional team, technical skill and physical ability are negatively correlated. If you could only study professional athletes (conditioning on "made the team"), you'd incorrectly conclude that skill and physical ability trade off against each other.

In ML: conditioning on a collider can create selection bias and lead to incorrect causal conclusions. Drawing the DAG before choosing controls helps avoid this mistake.

---

**Q: What is the fundamental problem of causal inference?**

For any individual, you can only observe one potential outcome — either the treated or untreated outcome, never both. The unobserved outcome is the counterfactual. Since individual treatment effects require comparing both potential outcomes for the same unit, they're fundamentally unidentifiable.

The workaround: we estimate population-level averages (ATE, ATT) using groups of individuals as proxies for each other's counterfactuals. Randomization ensures these groups are comparable in expectation. Quasi-experimental methods try to achieve comparability observationally.

---

*Last updated for ML interview preparation. Covers frequentist and Bayesian experimentation, quasi-experimental designs, causal graphical models, and practical pitfalls.*
