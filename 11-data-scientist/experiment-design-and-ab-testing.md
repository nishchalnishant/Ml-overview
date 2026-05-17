# Experiment Design and A/B Testing

---

## 1. A/B Test Lifecycle

1. **Hypothesis**: specific, directional, falsifiable — "Changing CTA button color from grey to green will increase click-through rate"
2. **Metric selection**: primary metric (OEC), secondary metrics, guardrail metrics — all defined before launch
3. **Power analysis**: compute required sample size from effect size, $\alpha$, $\beta$, and baseline variance
4. **Randomization**: assign units to control/treatment; validate split balance on pre-experiment covariates
5. **Run**: instrument logging, monitor for data pipeline issues; avoid peeking at results mid-experiment
6. **Analyze**: compute test statistic, p-value, confidence interval; check pre-experiment covariate balance (SRM check)
7. **Decide**: ship, reject, or iterate — decision based on statistical and practical significance

---

## 2. Sample Size and Power

### Formula for Means
$$n = \frac{2\sigma^2 (z_{\alpha/2} + z_\beta)^2}{\delta^2}$$

- $\sigma^2$: variance of the metric (from historical data)
- $\delta$: minimum detectable effect (MDE)
- $z_{\alpha/2}$: critical value for significance level (1.96 for $\alpha=0.05$, two-tailed)
- $z_\beta$: critical value for power (0.84 for 80% power, 1.28 for 90% power)
- $n$: required per arm

### Formula for Proportions
$$n = \frac{(z_{\alpha/2} + z_\beta)^2 [p_1(1-p_1) + p_2(1-p_2)]}{\delta^2}$$

Often simplified to $n = \frac{2\bar{p}(1-\bar{p})(z_{\alpha/2}+z_\beta)^2}{\delta^2}$ where $\bar{p}$ is pooled proportion.

### Key Relationships
- Halving MDE → 4× sample size (quadratic relationship)
- Increasing power from 80% to 90% → ~30% more samples
- Reducing $\alpha$ from 0.05 to 0.01 → larger critical value → more samples

---

## 3. Minimum Detectable Effect (MDE)

- Smallest difference that is **business-meaningful** — not the smallest statistically detectable difference
- MDE drives required sample size: smaller MDE = longer experiment
- Setting MDE requires product judgment: "We only care about shipping this change if it moves CTR by ≥ 0.5pp"
- Express MDE in relative terms (5% lift) or absolute (0.5pp) — be consistent with how you measured $\sigma$

---

## 4. Randomization

### Unit of Randomization
- **User-level**: stable assignment; prevents carry-over; required for personalization features
- **Session-level**: user may see both variants; valid only for changes with no memory effect
- **Request-level**: high power but high contamination risk (same user gets both variants)
- Choose the unit that matches the unit where SUTVA holds

### SUTVA (Stable Unit Treatment Value Assumption)
- Potential outcome for unit $i$ depends only on $i$'s own treatment — not on other units' assignments
- Violated by: social features (user A's feed changes based on user B's treatment), marketplace supply/demand effects, shared resources

### Network Effects
- Treatment of user A affects control user B through the social graph
- SUTVA violation inflates apparent treatment effect or masks it
- Solutions: cluster randomization (by social cluster), ego-network randomization, switchback experiments

### Balance Check (SRM — Sample Ratio Mismatch)
- Verify treatment:control ratio matches the configured split
- SRM indicates randomization bug, not a real effect — do not analyze results with SRM present
- Test: $\chi^2$ on observed vs expected unit counts per arm

---

## 5. Temporal Confounds

### Novelty Effect
- Users interact more with a change because it's new, not because it's better
- Inflates short-term metrics; decays over time
- Mitigation: run experiment long enough to observe stabilization (typically 2–4 weeks for frequent users)

### Primacy Effect
- Existing users are accustomed to old UX; new UX may underperform initially
- Effect reverses as users adapt — opposite direction from novelty effect
- Holdout the change for a cohort of new users to separate novelty/primacy from true effect

---

## 6. Metrics Framework

### Metric Taxonomy
- **OEC (Overall Evaluation Criterion)**: single primary metric — the one you're trying to move
- **Secondary metrics**: provide context; not decision criteria
- **Guardrail metrics**: must not degrade — latency, error rate, revenue, user trust signals
- **Diagnostic metrics**: help understand mechanism of the primary metric change

### Multiple Metrics Problem
- Testing 20 metrics at $\alpha = 0.05$ → ~1 false positive by chance
- Apply Bonferroni or BH correction when testing multiple primary metrics
- Pre-register which metric is the OEC to avoid HARKing

---

## 7. Sequential Testing

### The Peeking Problem
- Standard t-test p-values are only valid at a pre-specified sample size; peeking and stopping early inflates Type I error
- Peeking 5× at scheduled intervals with $\alpha = 0.05$ → effective $\alpha \approx 0.19$

### Solutions
- **mSPRT (mixture Sequential Probability Ratio Test)**: always-valid p-values; can stop at any time with controlled error rate
- **Alpha spending functions** (O'Brien-Fleming, Pocock): allocate $\alpha$ budget across planned interim looks
- **Fixed-horizon with blinded monitoring**: only monitor for data quality issues, not outcomes, until planned end date

---

## 8. Variance Reduction

### CUPED (Controlled-experiment Using Pre-Experiment Data)
- Adjust the outcome using a pre-experiment covariate (same metric from pre-period)
- $Y_{\text{adjusted}} = Y - \theta \cdot (X - \bar{X})$, where $\theta = \text{Cov}(Y, X) / \text{Var}(X)$
- Variance reduction: $\text{Var}(Y_{\text{adj}}) = \text{Var}(Y)(1 - \rho^2)$ where $\rho$ = correlation between $Y$ and $X$
- Typical variance reduction: 30–70% — allows smaller sample size or shorter experiment duration
- Requires pre-experiment data; does not introduce bias when covariate is pre-treatment

### CUPAC (Control Using Predictions as Covariates)
- Use a ML model prediction of $Y$ as the covariate instead of historical $Y$
- Captures non-linear relationships between covariates and outcome
- Stronger variance reduction than CUPED when ML model is accurate

### Stratified Sampling
- Pre-divide population into strata (e.g., new/returning users, platform); randomize within strata
- Guarantees proportional representation; reduces variance from between-strata differences

---

## 9. Common Pitfalls

| Pitfall | Description | Mitigation |
| :--- | :--- | :--- |
| **p-hacking** | Running test until p < 0.05, then stopping | Sequential testing or fixed horizon |
| **HARKing** | Hypothesizing After Results Known — redefining OEC post-run | Pre-registration |
| **Survivorship bias** | Analyzing only users who returned after day 1 | Include all users exposed, even churned |
| **Simpson's paradox** | Aggregate effect reverses within subgroups due to imbalanced strata | Stratify analysis; check subgroup trends |
| **Imbalanced splits** | SRM from logging bug, bot traffic | SRM check before analysis |
| **Leakage** | Control users exposed to treatment (contamination) | Verify assignment isolation |
| **Multiple comparisons** | 20 variants × 20 metrics = many false positives | Correction, limit variants, pre-register |
| **Stat sig ≠ practical sig** | Very large $n$ finds tiny meaningless effects | Report effect size and CI, not just p-value |

---

## 10. Bayesian A/B Testing

### Beta-Binomial Model
- Conversion rate: prior $\text{Beta}(\alpha_0, \beta_0)$; after $k$ successes in $n$ trials: posterior $\text{Beta}(\alpha_0 + k, \beta_0 + n - k)$
- **Posterior probability of being best**: $P(\theta_B > \theta_A)$ — computed via Monte Carlo sampling from posteriors
- **Expected loss**: $E[\max(\theta_A - \theta_B, 0)]$ — how much you lose by choosing the wrong variant; stop when expected loss < threshold

### Advantages over Frequentist
- Direct probability statements about which variant is better
- No fixed sample size required — can stop with controlled expected loss
- Incorporates prior knowledge about conversion rates

---

## 11. Multi-Armed Bandits

### Tradeoff vs Pure A/B
- Pure A/B: all exploration upfront, then exploit winner; maximizes learning but incurs regret during test
- Bandits: balance exploration and exploitation dynamically; reduce regret during experiment

### Algorithms
- **ε-greedy**: with probability $\epsilon$ explore uniformly; otherwise exploit best arm; simple but wasteful exploration
- **UCB (Upper Confidence Bound)**: choose arm $k$ that maximizes $\hat{\mu}_k + c\sqrt{\frac{\ln t}{n_k}}$; optimistic under uncertainty; no randomness
- **Thompson Sampling**: sample from each arm's posterior; select arm with highest sample; naturally balances exploration/exploitation; near-optimal regret

### When to Use Bandits vs A/B
- Bandits: short experiments, many arms, immediate feedback, regret minimization matters (e.g., real-time ad optimization)
- A/B: when learning the exact effect size matters, when delayed outcomes make bandits unstable, when regulatory requirements need clean control

---

## 12. Switchback Experiments

- For marketplace/two-sided platforms where SUTVA is violated (driver allocation, pricing, logistics)
- **Time-based randomization**: alternate treatment/control across time windows (e.g., 30-min slots)
- Analysis: compare time periods under treatment vs control, controlling for time-of-day and day-of-week effects
- Variance: high due to temporal autocorrelation — larger time slots reduce carry-over but reduce statistical power

---

## 13. Holdout Groups

- **Long-term holdout**: keep $x$% of users in control permanently; measure compounding effects of shipped changes
- Answers: "What is the cumulative effect of all changes shipped in the last 6 months?"
- **Cannibalization detection**: holdout for one product area measures whether it steals engagement from another
- Holdout size: balance long-term measurement value vs opportunity cost of not shipping to all users
