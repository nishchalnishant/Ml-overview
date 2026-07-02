---
module: Emerging Topics
topic: Experimentation and Causal Inference
subtopic: ""
status: unread
tags: [emergingtopics, ml, experimentation-and-causal-inference]
---
# Experimentation and Causal Inference

---

## 1. The Problem

You change your checkout button from gray to green. Revenue goes up 12% the following week. Did the button cause the increase?

Maybe. Or maybe it was a seasonal shopping surge, a promotional email that went out on the same day, a competitor outage, or the fact that the users who happened to visit that week were more purchase-intent users anyway. Without a controlled comparison, you have a correlation and a story — not a causal estimate.

The foundational problem in experimentation: **correlation is not causation, and selection bias makes correlations misleading by default.** The people who use LinkedIn Premium are different from people who don't — they're more senior, more proactive, more likely to get job offers regardless. Naively comparing Premium vs non-Premium users would show an effect even if Premium did nothing.

The reason this matters is not philosophical. If you misattribute a revenue increase to a product change you made, you will make more of that kind of change. If you misattribute a user behavior to a feature rather than to the type of user who selects into using it, your product decisions will be driven by selection bias rather than causal insight.

---

## 2. The Core Insight: Counterfactuals

The fundamental problem of causal inference: you want to know what would have happened to the same units if the treatment had been different. But you can only observe each unit in one world — either treated or untreated, never both.

For unit i:
- `Y_i(1)` = outcome if treated
- `Y_i(0)` = outcome if untreated
- **Individual Treatment Effect**: `ITE_i = Y_i(1) - Y_i(0)`

You can never observe both. The one you don't observe is the **counterfactual**.

What you can estimate is a population average:
- **ATE** (Average Treatment Effect): `E[Y(1) - Y(0)]` — average over everyone
- **ATT** (Average Treatment Effect on the Treated): `E[Y(1) - Y(0) | T=1]` — average over those who received treatment

The only way to estimate these cleanly is to ensure that the treatment and control groups are comparable — that the only systematic difference between them is the treatment itself. **Randomization is the mechanism that achieves this.**

When you randomly assign users to treatment or control, treatment assignment is independent of all potential outcomes:
```
(Y(0), Y(1)) ⊥ T
```

This means the control group is a valid counterfactual for the treatment group in expectation.

---

## 3. A/B Testing: Randomization Made Operational

### The Setup

An A/B test is a randomized controlled experiment. You randomly split users into two groups, show group A the old experience and group B the new experience, then measure whether outcomes differ.

The key design decisions:

**Unit of randomization**: what gets randomized? Usually the user (not the session or pageview), to avoid the same user seeing both experiences.

**Metric**: what are you measuring? Define this before running the test. The primary metric should be your hypothesis target (conversion rate, session length, revenue per user). Also define guardrail metrics — things you must not break (latency, error rate).

**Sample size**: how long do you need to run? Determined by minimum detectable effect (MDE), baseline metric value, and desired statistical power.

### Sample Size Calculation

For a binary metric (e.g., conversion rate):

```python
import numpy as np
from scipy import stats

def sample_size_two_proportions(
    p_baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True
) -> int:
    """
    Required sample size per group for a two-proportion z-test.
    
    p_baseline: baseline conversion rate
    mde: minimum detectable effect (relative lift, e.g., 0.05 = 5%)
    alpha: type I error rate
    power: 1 - type II error rate (probability of detecting a true effect)
    """
    p_treatment = p_baseline * (1 + mde)
    
    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_beta = stats.norm.ppf(power)
    
    # Pooled proportion
    p_pooled = (p_baseline + p_treatment) / 2
    
    # Standard formula for two proportions
    numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
                 z_beta * np.sqrt(p_baseline * (1 - p_baseline) + 
                                   p_treatment * (1 - p_treatment))) ** 2
    denominator = (p_treatment - p_baseline) ** 2
    
    n_per_group = int(np.ceil(numerator / denominator))
    return n_per_group


def cohens_d(mean1, mean2, pooled_std):
    """Effect size for continuous metrics."""
    return abs(mean1 - mean2) / pooled_std


# Example: current checkout conversion rate 3%, want to detect 5% relative lift
n = sample_size_two_proportions(p_baseline=0.03, mde=0.05, alpha=0.05, power=0.80)
print(f"Required sample per group: {n:,}")  # ~60,000 per group
```

### Running the Test

```python
from scipy import stats
import numpy as np

def run_ab_test(
    control_conversions: int, control_total: int,
    treatment_conversions: int, treatment_total: int,
    alpha: float = 0.05
) -> dict:
    """
    Two-proportion z-test for binary metric A/B test.
    """
    p_control = control_conversions / control_total
    p_treatment = treatment_conversions / treatment_total
    
    # Pooled proportion
    p_pooled = (control_conversions + treatment_conversions) / (control_total + treatment_total)
    
    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
    
    # Z-statistic
    z = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Confidence interval for absolute lift
    se_diff = np.sqrt(
        p_control * (1 - p_control) / control_total +
        p_treatment * (1 - p_treatment) / treatment_total
    )
    diff = p_treatment - p_control
    ci = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)
    
    relative_lift = (p_treatment - p_control) / p_control
    
    return {
        "p_control": p_control,
        "p_treatment": p_treatment,
        "absolute_lift": diff,
        "relative_lift": relative_lift,
        "z_statistic": z,
        "p_value": p_value,
        "ci_95": ci,
        "significant": p_value < alpha
    }

result = run_ab_test(
    control_conversions=300, control_total=10000,
    treatment_conversions=336, treatment_total=10000
)
print(f"Relative lift: {result['relative_lift']:.1%}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']}")
```

### What p-values Actually Mean

A p-value of 0.04 means: if the null hypothesis (no effect) were true, you would see a result this extreme or more extreme 4% of the time. It does **not** mean:
- There's a 96% chance the treatment works
- The effect is practically meaningful
- You should ship

A p-value is evidence against the null, not evidence for the alternative. An effect can be statistically significant and practically trivial (large n, tiny lift), or practically large but not statistically significant (small n).

**Type I error**: reject H₀ when it's true (false positive). Rate = α.
**Type II error**: fail to reject H₀ when it's false (false negative). Rate = β. Power = 1 − β.

---

## 4. What Breaks in A/B Testing

### Peeking (Early Stopping)

You run the experiment, check the dashboard every day, and stop when p < 0.05. This is peeking, and it inflates your false positive rate dramatically.

```python
import numpy as np
from scipy import stats

def simulate_peeking_false_positive_rate(
    n_simulations: int = 1000,
    max_n: int = 10000,
    check_every: int = 100,
    alpha: float = 0.05
) -> float:
    """
    Simulate what happens when you peek at results continuously.
    Under the null (no effect), how often do you incorrectly reject?
    """
    false_positives = 0
    
    for _ in range(n_simulations):
        control = []
        treatment = []
        rejected = False
        
        for i in range(check_every, max_n + 1, check_every):
            control.extend(np.random.normal(0, 1, check_every).tolist())
            treatment.extend(np.random.normal(0, 1, check_every).tolist())
            
            _, p = stats.ttest_ind(control, treatment)
            if p < alpha:
                rejected = True
                break
        
        if rejected:
            false_positives += 1
    
    return false_positives / n_simulations

# Returns ~0.30-0.40 instead of 0.05 — peeking inflates false positives 6-8x
```

**Why it happens**: each look is an additional opportunity to incorrectly reject. The p-value process is not monotone — it can cross the threshold by chance at any look and never come back.

**Fixes**:
- Pre-register the sample size and only look once, at the end
- Use sequential testing with α-spending functions (O'Brien-Fleming, Pocock)
- Use Bayesian testing, where the posterior is valid at any sample size

### Multiple Testing

You launch a redesigned checkout page and measure 10 metrics. Even if the redesign does nothing, at α = 0.05 you expect 0.5 false positives. Across 100 experiments, 5 will appear significant by chance.

**Bonferroni correction**: divide α by number of tests.
```
α_adjusted = α / m
```
Simple, conservative. For 10 tests: each uses α = 0.005.

**Benjamini-Hochberg (False Discovery Rate)**: controls the expected proportion of rejected hypotheses that are false positives. More powerful than Bonferroni when you have many tests.

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.001, 0.04, 0.08, 0.12, 0.20, 0.35, 0.40, 0.52, 0.68, 0.90]

_, bonferroni_corrected, _, _ = multipletests(p_values, method='bonferroni', alpha=0.05)
_, bh_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
```

Use Bonferroni when one false positive is very costly. Use BH for exploratory analysis.

### Simpson's Paradox

UC Berkeley (1973): men admitted at 44%, women at 35%. Apparent gender bias. But in nearly every department, women had equal or higher admission rates. Women applied to more competitive departments in higher proportions.

A trend in aggregated data reverses or disappears when disaggregated by a confounding variable.

In A/B testing: if treatment and control differ in composition (mobile vs desktop mix), and the metric differs across subgroups, and subgroup distribution correlates with treatment — you have Simpson's Paradox.

**Detection**: always segment results by key dimensions. If aggregated and segmented results tell different stories, you have it.

**Fix**: stratified randomization, or control for the confound in analysis.

### SUTVA Violations (Network Effects)

The **Stable Unit Treatment Value Assumption** requires:
1. Treatment of unit i does not affect outcome of unit j
2. There is only one version of treatment

SUTVA is violated on social networks. You test a new notification system: treatment users become more active, which increases content volume, which makes control users more active — contaminating the control group. The treatment effect appears smaller than it is.

**Fixes**: cluster randomization (randomize by geographic region, social community, not individual), ego-network experiments, time-based holdouts.

### Novelty Effects and Change Aversion

New features spike engagement because they're new, not because they're better (novelty). Users also initially underperform on redesigned experiences even when objectively better (change aversion).

**Detection**: plot treatment effect over time within the experiment. Does the lift decay?

**Fix**: run experiments long enough for users to habituate. Minimum 2-4 weeks for products with daily active users.

---

## 5. Bayesian A/B Testing

Frequentist A/B testing answers: "How often would I get a result this extreme if H₀ were true?" This is not what people actually want to know.

Bayesian testing answers: "Given the data, what's the probability that B is better than A?"

### Beta-Binomial Model

For binary metrics (conversion rates):

```python
import numpy as np
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
    Beta-Binomial conjugate model.
    Prior: θ ~ Beta(α, β)
    Likelihood: x | θ ~ Binomial(n, θ)
    Posterior: θ | x ~ Beta(α + x, β + n - x)
    """
    alpha_control = prior_alpha + control_conversions
    beta_control = prior_beta + (control_total - control_conversions)
    
    alpha_treatment = prior_alpha + treatment_conversions
    beta_treatment = prior_beta + (treatment_total - treatment_conversions)
    
    samples_control = beta_dist.rvs(alpha_control, beta_control, size=n_samples)
    samples_treatment = beta_dist.rvs(alpha_treatment, beta_treatment, size=n_samples)
    
    prob_treatment_better = np.mean(samples_treatment > samples_control)
    lift_samples = samples_treatment - samples_control
    
    return {
        "prob_treatment_better": prob_treatment_better,
        "expected_lift": np.mean(lift_samples),
        "credible_interval_95": tuple(np.percentile(lift_samples, [2.5, 97.5])),
    }

result = bayesian_ab_test(
    control_conversions=100, control_total=1000,
    treatment_conversions=115, treatment_total=1000
)
print(f"P(treatment > control): {result['prob_treatment_better']:.3f}")
```

Advantages:
- **No peeking problem**: the posterior is valid at any sample size
- **Interpretable output**: "94% probability treatment is better" — what everyone wants to say about a p-value but can't
- **Prior knowledge**: encode past experiment results as prior
- **Magnitude estimation**: full posterior distribution over lift, not just binary significant/not

### Expected Loss Framework

Instead of probability of being better:
```
Loss(choose A | B is better) = E[max(0, θ_B - θ_A)]
Loss(choose B | A is better) = E[max(0, θ_A - θ_B)]
```

Stop when expected loss from choosing B falls below a threshold. This balances exploration and exploitation without a fixed sample size.

---

## 6. Multi-Armed Bandits

A/B testing is wasteful: half your traffic goes to the losing variant throughout the entire experiment. Multi-armed bandits (MAB) dynamically allocate more traffic to better-performing variants.

The exploration-exploitation tradeoff: pure exploitation gets stuck on early leaders; pure exploration wastes budget on arms you already know are bad.

### Thompson Sampling

```python
import numpy as np
from scipy.stats import beta as beta_dist

class ThompsonSampling:
    """Bernoulli bandit with Beta posteriors over conversion rates."""
    
    def __init__(self, n_arms: int, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.n_arms = n_arms
        self.alpha = np.full(n_arms, prior_alpha)
        self.beta = np.full(n_arms, prior_beta)
    
    def select_arm(self) -> int:
        samples = beta_dist.rvs(self.alpha, self.beta)
        return int(np.argmax(samples))
    
    def update(self, arm: int, reward: int):
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
    
    def get_conversion_rates(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)
```

**Why it works**: when you're uncertain about an arm, its posterior is wide, so samples sometimes hit high values — encouraging exploration. As you pull it more, the posterior tightens. It only gets selected when its mean is genuinely high.

### Upper Confidence Bound (UCB1)

Deterministic alternative:
```
UCB_i(t) = x̄_i + sqrt(2 * ln(t) / n_i)
```

The second term is the uncertainty bonus — large for rarely-pulled arms, shrinks with more data.

```python
class UCB1:
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.pulls = np.zeros(n_arms)
        self.total_reward = np.zeros(n_arms)
        self.t = 0
    
    def select_arm(self) -> int:
        self.t += 1
        for arm in range(self.n_arms):
            if self.pulls[arm] == 0:
                return arm
        means = self.total_reward / self.pulls
        ucb_scores = means + np.sqrt(2 * np.log(self.t) / self.pulls)
        return int(np.argmax(ucb_scores))
    
    def update(self, arm: int, reward: float):
        self.pulls[arm] += 1
        self.total_reward[arm] += reward
```

### When to Use Bandits vs A/B Tests

| Criterion | A/B Test | Bandit |
|---|---|---|
| Goal | Causal inference | Maximize cumulative reward |
| Traffic to losers | Equal (wasteful) | Minimized |
| Statistical validity | Well-established | Harder to do inference |
| Multiple variants | Hard (multiple testing) | Natural |
| Non-stationary rewards | No | Handles with decay/sliding window |

Bandits for pricing, recommendations, content selection — many variants, frequent decisions, care about cumulative performance. A/B tests when you need a clean causal estimate for a major decision.

---

## 7. Observational Studies: When You Can't Randomize

Most real-world causal questions can't be answered with randomization:
- You can't randomly assign people to smoke
- You can't randomly assign companies to raise minimum wage
- You can't retroactively randomize which users adopted a feature

In these cases you work with observational data and try to adjust for confounders.

### What Makes a Confounder

Variable Z is a confounder of the T → Y relationship if:
1. Z causes T (Z → T)
2. Z causes Y (Z → Y)

This creates a backdoor path T ← Z → Y — a non-causal association between T and Y.

Example: LinkedIn Premium and job offers. Senior, proactive professionals are more likely to pay for Premium and more likely to receive job offers. Seniority is a confounder. Naive comparison overestimates Premium's effect.

### The Ignorability Assumption

For observational causal inference to work:
```
(Y(0), Y(1)) ⊥ T | X
```

Conditional on observed covariates X, treatment is as good as randomly assigned. You've measured all the confounders. This is untestable from data alone — it requires domain knowledge.

---

## 8. Propensity Score Matching

### The Insight

The **propensity score** is the probability of being treated given observed covariates:
```
e(X) = P(T = 1 | X)
```

**Rosenbaum & Rubin (1983)**: if treatment is ignorable given X, it's also ignorable given e(X):
```
(Y(0), Y(1)) ⊥ T | e(X)
```

Instead of matching on high-dimensional X (dozens of variables), match on a single scalar.

### Implementation

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
    caliper: float = 0.05
) -> dict:
    """
    Estimate ATT via nearest-neighbor propensity score matching.
    caliper: max allowed propensity score difference for a valid match.
    """
    X = df[covariate_cols].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, T)
    ps = lr.predict_proba(X_scaled)[:, 1]
    
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    
    ps_treated = ps[treated_idx].reshape(-1, 1)
    ps_control = ps[control_idx].reshape(-1, 1)
    
    distances = cdist(ps_treated, ps_control, metric='euclidean')
    
    matched_pairs = []
    used_control = set()
    
    for i, t_idx in enumerate(treated_idx):
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
    
    treated_outcomes = np.array([Y[t] for t, c in matched_pairs])
    control_outcomes = np.array([Y[c] for t, c in matched_pairs])
    
    att_estimate = np.mean(treated_outcomes - control_outcomes)
    se = np.std(treated_outcomes - control_outcomes) / np.sqrt(len(matched_pairs))
    
    return {
        "att": att_estimate,
        "se": se,
        "n_matched": len(matched_pairs),
        "n_treated": len(treated_idx),
        "propensity_scores": ps,
    }
```

### Checking Balance

After matching, verify treated and control groups look similar on observed covariates. Use the **standardized mean difference (SMD)**:
```
SMD = (x̄_treated - x̄_control) / sqrt((s²_treated + s²_control) / 2)
```
SMD < 0.1 is generally good balance.

### Inverse Probability Weighting

Alternative to matching: weight each observation by the inverse of its propensity score.
```
ATE_IPW = (1/n) * Σ [ T_i * Y_i / e(X_i) - (1-T_i) * Y_i / (1-e(X_i)) ]
```

IPW uses all data (unlike matching, which discards unmatched units) but is unstable when propensity scores are near 0 or 1.

**Doubly Robust Estimator**: combines propensity score model with an outcome model. Consistent if either model is correctly specified.

---

## 9. Difference-in-Differences

### The Problem PSM Doesn't Solve

PSM conditions on observed confounders. But many important confounders are unobserved. DiD handles time-invariant unobserved confounders by using the pre-treatment period as a baseline.

### The Core Idea

Compare the **change over time** rather than levels. This removes time-invariant confounders.

```
DiD = (Ȳ_treated,post - Ȳ_treated,pre) - (Ȳ_control,post - Ȳ_control,pre)
```

**Card and Krueger (1994)**: New Jersey raised minimum wage in 1992; Pennsylvania did not. DiD compared employment changes in NJ fast food restaurants vs PA. Result: no employment decrease — a landmark challenge to standard labor economics.

### The Regression Framework

```python
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols

def simulate_did_example():
    """
    DiD: loyalty program rolled out in one city (treated), not another (control).
    True treatment effect: $8 revenue per user.
    Natural time trend: +$5 in both cities.
    """
    np.random.seed(42)
    n = 500
    
    control_pre = np.random.normal(50, 10, n)
    control_post = np.random.normal(55, 10, n)     # +$5 natural trend
    treated_pre = np.random.normal(50, 10, n)
    treated_post = np.random.normal(63, 10, n)     # +$5 trend + $8 treatment
    
    df = pd.DataFrame({
        'revenue': np.concatenate([control_pre, control_post, treated_pre, treated_post]),
        'treated': np.concatenate([np.zeros(n*2), np.ones(n*2)]),
        'post': np.concatenate([np.zeros(n), np.ones(n), np.zeros(n), np.ones(n)])
    })
    df['treated_x_post'] = df['treated'] * df['post']
    
    # β₃ is the DiD estimator
    model = ols('revenue ~ treated + post + treated_x_post', data=df).fit()
    print(f"DiD estimate (β₃): {model.params['treated_x_post']:.2f}")  # ~8.0
    
    return df, model
```

`β₃` is the DiD estimator — the differential change in the treated group after treatment, controlling for group baseline (`β₁`) and common time trend (`β₂`).

### The Parallel Trends Assumption

**The critical assumption**: in the absence of treatment, treated and control groups would have followed the same time trend.

Untestable directly, but provide supporting evidence:
1. **Pre-trend visualization**: plot both groups before treatment — should be parallel
2. **Placebo DiD**: apply DiD to an earlier time period — the "effect" should be zero
3. **Event study**: include time × treatment interactions in the pre-period and test jointly

If pre-trends differ, DiD is biased. Consider synthetic control.

---

## 10. Regression Discontinuity Design

### The Core Idea

When treatment is assigned by crossing a threshold of a running variable, units just below and just above the threshold are very similar — they differ only in whether they crossed the cutoff. Near the threshold, assignment is as good as random.

Classic example: scholarship awarded to students scoring above 70. Students scoring 69 vs 71 are very similar in ability; one group got the scholarship. Comparing their later outcomes gives a causal estimate of the scholarship effect.

### The Estimand

In **sharp RD** (treatment is a deterministic step function of the running variable):
```
τ_RD = lim_{x↓c} E[Y | X = x] - lim_{x↑c} E[Y | X = x]
```

This is a local average treatment effect — valid only at the threshold. People far from the cutoff may respond differently.

```python
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

def simulate_rdd():
    """
    Tutoring program for students scoring >= 60.
    True effect: +5 points on final exam.
    """
    np.random.seed(42)
    n = 2000
    
    score = np.random.uniform(30, 90, n)
    treatment = (score >= 60).astype(int)
    outcome = 40 + 0.5 * score + 5 * treatment + np.random.normal(0, 5, n)
    
    df = pd.DataFrame({'score': score, 'treatment': treatment, 'outcome': outcome})
    df['score_centered'] = df['score'] - 60
    
    # Local linear regression: only units within bandwidth of cutoff
    bandwidth = 10
    df_local = df[np.abs(df['score_centered']) <= bandwidth].copy()
    
    model = ols('outcome ~ score_centered * treatment', data=df_local).fit()
    print(f"RDD estimate: {model.params['treatment']:.2f}")  # ~5.0
    
    return df, model
```

**In fuzzy RD** (crossing threshold changes probability but not certainty of treatment): use the threshold as an instrument for treatment.

### Key Diagnostics

1. **McCrary test**: no bunching at the cutoff (would suggest manipulation of running variable)
2. **Covariate balance**: predetermined covariates shouldn't jump at cutoff
3. **Placebo cutoffs**: no discontinuity at other values of running variable
4. **Bandwidth sensitivity**: results should be stable across reasonable bandwidths

**Bandwidth tradeoff**: wider = more data, lower variance, more bias; narrower = less bias, less data, higher variance. Calonico-Cattaneo-Titiunik provides data-driven optimal bandwidth.

---

## 11. Instrumental Variables

### The Problem

You want to estimate the effect of education on earnings. You can't randomize education. Confounders (family wealth, ability, ambition) are hard to measure.

**IV** finds a variable that causes the treatment but affects the outcome only through the treatment — a natural experiment embedded in the data.

### The Two Conditions

A valid instrument Z must satisfy:
1. **Relevance**: Z is correlated with T (`Cov(Z, T) ≠ 0`, testable by first-stage F-statistic)
2. **Exclusion restriction**: Z affects Y only through T — untestable, requires subject matter argument
3. **Independence**: Z is independent of confounders (as good as randomly assigned)

### Classic Examples

**Angrist (1990)**: Is military service bad for earnings? Vietnam draft lottery was truly random. Draft number is the instrument — affects probability of military service, doesn't directly affect earnings.

**Card (1995)**: Does education increase earnings? Instrument: proximity to a 4-year college growing up. Increases college attendance probability without directly affecting earnings.

### Two-Stage Least Squares (2SLS)

**Stage 1**: Regress T on Z (and covariates X): `T̂ = α₀ + α₁Z + α₂X + ν`

**Stage 2**: Regress Y on T̂: `Y = β₀ + β₁T̂ + β₂X + ε`

The IV estimate `β₁` is the **LATE** — Local Average Treatment Effect for compliers (units whose treatment status changes because of the instrument).

```python
from linearmodels.iv import IV2SLS
import pandas as pd
import numpy as np

def simulate_iv_example():
    """
    Tutoring → test scores. Unobserved confounder: ability.
    Instrument: random scholarship offer.
    True effect: +5 points.
    """
    np.random.seed(42)
    n = 1000
    
    ability = np.random.normal(0, 1, n)
    scholarship = np.random.binomial(1, 0.5, n)
    
    tutoring_latent = 0.8 * scholarship + 0.5 * ability + np.random.normal(0, 1, n)
    tutoring = (tutoring_latent > 0).astype(int)
    
    test_score = 60 + 5 * tutoring + 10 * ability + np.random.normal(0, 2, n)
    
    df = pd.DataFrame({
        'test_score': test_score,
        'tutoring': tutoring,
        'scholarship': scholarship,
        'const': 1.0
    })
    
    iv_model = IV2SLS(
        dependent=df['test_score'],
        exog=df[['const']],
        endog=df[['tutoring']],
        instruments=df[['scholarship']]
    ).fit()
    
    print(f"IV estimate: {iv_model.params['tutoring']:.2f}")   # ~5.0
    
    from statsmodels.formula.api import ols
    ols_model = ols('test_score ~ tutoring', data=df).fit()
    print(f"OLS estimate (biased by ability): {ols_model.params['tutoring']:.2f}")  # >5
    
    return iv_model
```

**Weak instruments**: if the first-stage F-statistic is low (rule of thumb: F < 10), the instrument barely predicts treatment. IV estimates have huge variance and are sensitive to small violations of exclusion restriction. Always report first-stage F.

---

## 12. Causal Graphs (DAGs)

### What DAGs Are

A **Directed Acyclic Graph** is a formal representation of causal structure. Nodes = variables, directed edges = causal effects. DAGs tell you exactly which variables to control for — and which to avoid controlling for.

### Three Structural Patterns

**Chain (Mediation)**: `A → B → C`
B is a mediator. If you condition on B, you block the causal path from A to C — can't estimate total effect.

**Fork (Confounder)**: `A ← C → B`
C causes both A and B, creating spurious correlation. Must condition on C to estimate A's effect on B.

**Collider**: `A → C ← B`
Both A and B cause C. By default, no spurious correlation between A and B. **If you condition on C, you induce spurious correlation** — this is collider bias.

### The Backdoor Criterion

To estimate the causal effect of T on Y, block all **backdoor paths** — paths from T to Y that start with an arrow into T.

Set Z satisfies the backdoor criterion if:
1. Z blocks all backdoor paths from T to Y
2. Z contains no descendant of T

If backdoor criterion is satisfied:
```
P(Y | do(T=t)) = Σ_z P(Y | T=t, Z=z) P(Z=z)
```

### do-Calculus

The key distinction: `P(Y | T = t)` vs `P(Y | do(T = t))`.

**Observing** T = t: select the subpopulation where T happened to be t. Confounders are still active.

**Intervening** T = t: cut all arrows into T, then set T = t. Confounders are severed.

```python
import numpy as np
from scipy.stats import pointbiserialr

np.random.seed(42)
n = 10000

# Collider example: G → H ← Q (gender and qualification both affect hiring)
# G and Q are truly independent
gender = np.random.binomial(1, 0.5, n)
qualification = np.random.normal(0, 1, n)

hire_prob = 1 / (1 + np.exp(-(qualification - 0.3 * gender)))
hired = np.random.binomial(1, hire_prob, n)

# Marginally: gender and qualification are independent
corr_marginal, p_marginal = pointbiserialr(gender, qualification)
print(f"Marginal G-Q correlation: {corr_marginal:.4f} (p={p_marginal:.3f})")  # near 0

# Conditioning on collider (hired): spurious negative correlation
mask = hired == 1
corr_conditional, p_conditional = pointbiserialr(gender[mask], qualification[mask])
print(f"Conditional G-Q | hired: {corr_conditional:.4f}")  # negative
# Among hired people: female predicts lower qualification — an artifact of collider bias
```

---

## 13. Uplift Modeling / Heterogeneous Treatment Effects

### Beyond Average Effects

ATE is a population average. The treatment might help some people a lot, hurt others, and do nothing for the rest.

**CATE** (Conditional Average Treatment Effect):
```
CATE(x) = E[Y(1) - Y(0) | X = x]
```

In marketing, users fall into four groups:
- **Persuadables**: would not have converted without treatment, will with it — the target
- **Sure Things**: would convert with or without treatment — wasteful to target
- **Lost Causes**: won't convert regardless
- **Sleeping Dogs**: worse with treatment — harmful to target

### Meta-Learner Approaches

**T-Learner**: fit separate outcome models for treated and control, take the difference.

**S-Learner**: fit one model with treatment as a feature, predict at T=1 and T=0.

**X-Learner**: handles imbalanced treatment groups by imputing treatment effects.

```python
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

class TLearner:
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
```

### Evaluating Uplift Models: Qini Curves

```python
def qini_curve(y: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray):
    """
    Qini curve: cumulative incremental outcome vs fraction targeted.
    Higher AUUC = better uplift model.
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

Causal forests (Wager & Athey, 2018) extend random forests for CATE estimation. At each split, they optimize for **heterogeneity in treatment effects** rather than prediction accuracy.

Key properties:
- **Honesty**: different subsamples for building tree structure vs estimating effects
- **Local centering**: residualizes Y and T to reduce confounding
- **Confidence intervals**: provides valid uncertainty estimates for CATE

Implementations: `econml`, `causalml`.

---

## 14. Experimentation in ML Systems

### Model A/B Tests

Testing a new ML model differs from testing a UI change:
- The "treatment" is invisible to users — all effects are through model outputs
- Models can interact (recommendation models affect the item pool)
- Effects can be non-linear and hard to predict offline

Same framework: randomly split traffic, serve model A to one group and model B to another, measure business metrics.

**Latency is always a treatment effect.** If model B is more accurate but 100ms slower, the latency itself has a treatment effect. Always measure both.

### The Offline-Online Metrics Gap

Models that improve offline metrics (AUC-ROC, RMSE) often do not improve online metrics (CTR, conversion, session length), and vice versa.

**Reasons for the gap**:
1. **Distribution shift**: held-out set doesn't represent future production traffic
2. **Label proxies**: offline labels are proxies for what users actually want
3. **Feedback loops**: model affects user behavior, which affects future data — not captured offline
4. **Position bias**: items shown in position 1 get more clicks regardless of quality
5. **Missing counterfactuals**: can only evaluate items that were shown, not all possible items

Best practice: define the online metric first, then design offline metrics to be predictive of it. Use interleaving for ranking models.

### Shadow Mode and Champion-Challenger

**Shadow mode**: new model runs in production but its decisions are not acted upon. Validates infrastructure and prediction distribution, but proves nothing about user impact.

**Champion-challenger**: current model handles most traffic; new model handles a small slice (5%). Monitor business metrics in real-time. If challenger is better and stable, gradually increase traffic share (canary deployment).

Limits downside risk: if challenger fails, only 5% of users are affected.

### Interleaving for Ranking

For search and recommendation, standard A/B tests need huge sample sizes because ranking quality is hard to measure at the user level.

1. Generate rankings from model A and model B for each request
2. Interleave the rankings (team-draft or balanced interleaving)
3. Track which model's items get engaged with

Interleaving can be 100x more sensitive than A/B testing for ranking quality — faster model iteration.

### Metrics Hierarchy

Design metric systems in layers:

1. **North star**: single metric capturing long-term value (90-day retention, LTV). Hard to move; if you move it, it matters.
2. **Driver metrics**: move faster, drive north star (7-day retention, session length, CTR)
3. **Guardrail metrics**: must not decrease (latency, error rate, support volume)
4. **Diagnostic metrics**: explain why driver metrics moved

An experiment optimizes driver metrics without degrading guardrails. Moving the north star directly is slow; work through drivers.

---

## What Breaks

**Randomization doesn't eliminate all confounders.** It eliminates selection bias in expectation, but with small samples you can still get unlucky imbalance. Always check balance on key characteristics.

**Ignorability is untestable.** Propensity score matching and other observational methods only control for measured confounders. If a key confounder is unmeasured, the estimate is biased. Sensitivity analysis is the only recourse.

**Parallel trends can fail silently.** Pre-trend tests only detect violations in the pre-period. If the treatment and control groups were converging or diverging before treatment in a way you didn't detect, DiD is biased.

**Exclusion restriction is the IV Achilles' heel.** If the instrument has a direct path to the outcome (outside of the treatment), 2SLS is inconsistent. This is untestable and is the reason most IV applications are contested.

**LATE ≠ ATE.** IV estimates the effect for compliers — often a minority of the population. External validity (generalizing to the full population) requires additional assumptions.

**Collider bias is easy to miss.** Conditioning on a post-treatment variable, a common effect, or a selection variable induces spurious correlations. Drawing the DAG before choosing controls is the only reliable prevention.

**Uplift models require holdout validation.** Standard cross-validation can't validate uplift models because you never observe both potential outcomes. Qini curves on a proper holdout set (with randomized treatment) are required.

---

## Key Interview Points

- **Fundamental problem of causal inference**: you can observe Y(1) or Y(0) for each unit, never both. ITE is unidentifiable. ATE and ATT are identified through group comparisons.
- **Randomization** makes treatment independent of all potential outcomes, enabling unbiased estimation of ATE.
- **p-value** is the probability of seeing data this extreme if the null is true — not the probability the null is true, not the probability the treatment works.
- **Peeking** inflates false positive rates to 30-40% for a nominal 5% — the p-value process is not monotone.
- **Bayesian A/B**: beta-binomial conjugate model, posterior is Beta(α + conversions, β + non-conversions). Valid to check at any sample size.
- **Thompson Sampling**: sample from each arm's posterior, pull the arm with highest sample. Exploration is automatic — uncertain arms have wide posteriors that sometimes sample high.
- **Propensity score theorem**: conditioning on e(X) = P(T=1|X) is sufficient to achieve ignorability. Reduces high-dimensional matching to a scalar.
- **DiD identifies treatment effect** by differencing out time-invariant unobserved confounders. Requires parallel trends assumption.
- **RDD** exploits threshold assignment: units just below/above are comparable. Estimates a LATE at the threshold.
- **IV/2SLS** requires relevance (F > 10 in first stage), exclusion restriction (untestable), and independence. Estimates LATE for compliers.
- **Collider bias**: conditioning on a common effect of two variables induces spurious correlation between them — even if they were marginally independent.
- **CATE estimation**: T-learner fits separate outcome models and takes the difference. Causal forests (Wager & Athey) optimize splits for treatment effect heterogeneity with honest estimation and valid CIs.
- **Interleaving**: 100x more sensitive than A/B testing for ranking quality by tracking engagement with each model's items in a merged ranking.
