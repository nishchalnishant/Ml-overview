# Experiment Design and A/B Testing

---

## 1. The A/B Test Lifecycle

### The problem

You have a product change and want to know whether it makes things better or worse. Your intuition might be right. The data you already have might seem to confirm it. But without a controlled comparison, you cannot separate the effect of your change from everything else that happened at the same time — seasonality, user growth, concurrent product changes, or simply regression to the mean. You need a procedure that isolates cause.

### The core insight

Run your change on a randomly selected subset of users simultaneously with a control group. Because assignment is random, the two groups are statistically identical on every dimension — including dimensions you cannot measure. Any systematic difference in outcomes can therefore be attributed to the change, not to who the users are.

### The procedure that follows

The lifecycle is not arbitrary ritual. Each step exists because skipping it introduces a specific failure mode.

1. **Hypothesis** — state a specific, directional, falsifiable claim before seeing any data ("changing CTA color from grey to green will increase click-through rate"). Specificity before launch prevents you from choosing your hypothesis after seeing results.

2. **Metric selection** — choose one primary metric (the Overall Evaluation Criterion, OEC), secondary metrics for context, and guardrail metrics that must not degrade. Choosing metrics before launch prevents you from picking the metric that happened to move.

3. **Power analysis** — calculate the sample size needed to detect a business-meaningful effect with acceptable error rates. Without this step, you run experiments that are too short to detect real effects, or too long, wasting time.

4. **Randomization** — assign units to control and treatment. Validate that the split is balanced on pre-experiment covariates to catch bugs before they corrupt the analysis.

5. **Run** — instrument logging to record exposure and outcome. Avoid looking at results during the run; the reasons why are explained in Section 7.

6. **Analyze** — compute the test statistic and confidence interval. Check for sample ratio mismatch (SRM) first; if SRM is present, the randomization was corrupted and results are invalid.

7. **Decide** — ship, reject, or iterate. The decision criterion is statistical significance AND practical significance. A tiny effect that is statistically significant from a huge sample may not be worth shipping.

---

## 2. Sample Size and Statistical Power

### The problem

You run an experiment for a week, see a positive trend, and ship. Three months later the metric has not moved. What went wrong? You likely ran an underpowered experiment — one where the sample was too small to reliably detect the effect you cared about. An underpowered test will miss real effects (Type II error) and, perversely, any effects it does detect are more likely to be noise than signal (the winner's curse).

### The core insight

A test has two ways to be wrong. A Type I error (false positive) concludes there is an effect when there is none. A Type II error (false negative) misses a real effect. Both are controlled by a common mechanism: the signal-to-noise ratio of your test statistic. The signal is the true difference $\delta$ between groups. The noise is $\sigma/\sqrt{n}$, the standard error of your estimator. You need $n$ large enough that a real effect of size $\delta$ produces a test statistic large enough to be detected with probability $1 - \beta$ (power), while keeping the probability of a false alarm below $\alpha$.

### Why the formula has this shape

For a two-sample t-test on means, the critical quantity is the standardized difference the test must distinguish. To reject the null at significance level $\alpha$ (two-tailed), the test statistic must exceed $z_{\alpha/2}$. To detect a true effect $\delta$ with power $1-\beta$, the non-centrality of the test statistic must also exceed $z_\beta$. Setting up the algebra: the test statistic under the alternative hypothesis is distributed around $\delta / (\sigma\sqrt{2/n})$. Requiring this to exceed $z_{\alpha/2} + z_\beta$ and solving for $n$:

$$n = \frac{2\sigma^2 (z_{\alpha/2} + z_\beta)^2}{\delta^2}$$

where $n$ is the required sample per arm. The factor of 2 in the numerator comes from having two arms each contributing $\sigma^2/n$ to the variance of the difference. For proportions, $\sigma^2 = p(1-p)$, so the formula becomes:

$$n = \frac{(z_{\alpha/2} + z_\beta)^2 [p_1(1-p_1) + p_2(1-p_2)]}{\delta^2}$$

Often approximated using the pooled proportion $\bar{p}$: $n \approx \frac{2\bar{p}(1-\bar{p})(z_{\alpha/2}+z_\beta)^2}{\delta^2}$.

Common plug-in values: $z_{\alpha/2} = 1.96$ for $\alpha = 0.05$ two-tailed; $z_\beta = 0.84$ for 80% power; $z_\beta = 1.28$ for 90% power.

### Key relationships that follow from the formula

- Halving MDE ($\delta \to \delta/2$) quadruples required $n$, because $\delta$ appears squared in the denominator. The quadratic relationship means aggressive MDE choices are expensive.
- Increasing power from 80% to 90% increases $(z_{\alpha/2} + z_\beta)^2$ from $(1.96+0.84)^2 = 7.84$ to $(1.96+1.28)^2 = 10.50$, roughly a 34% increase in sample.
- Tightening $\alpha$ from 0.05 to 0.01 raises $z_{\alpha/2}$ from 1.96 to 2.58, increasing sample by $\approx 44\%$.

### What breaks

If you estimate $\sigma^2$ from a small historical sample, the formula gives a confident but wrong answer. Variance estimates from short time windows may miss weekly seasonality cycles. Use at least 4 weeks of historical data to estimate $\sigma^2$, and when in doubt, run longer rather than shorter.

---

## 3. Minimum Detectable Effect (MDE)

### The problem

The power formula requires you to specify $\delta$ — the effect size. But you do not know the effect size before running the experiment; that is what you are trying to find out. What should you plug in?

### The core insight

The question is not "what effect will we see?" but "what effect would we need to see before we'd bother shipping?" This is a business question, not a statistical one. The MDE is the smallest effect that is worth the cost of shipping the change. Anything smaller — even if real — does not justify the engineering and operational overhead.

### Why MDE drives experiment duration

Once you set MDE, you have fixed $\delta$ in the sample size formula. Reducing MDE below the business-meaningful threshold buys you nothing useful: you would be powering the experiment to detect effects too small to act on. Setting MDE too high means your experiment cannot detect effects that would actually matter, leading to false null results.

### What breaks

Two common mistakes: (1) setting MDE to whatever effect size the team "expects" to see — this is circular and will lead to underpowered experiments when the true effect is smaller than expected; (2) setting MDE to the smallest statistically detectable effect for a given runtime, rather than the smallest businessmeaningful effect — this inverts the logic and leads to experiments that run too long. Express MDE in the same units as your metric; verify it is consistent with your variance estimate (relative vs absolute).

---

## 4. Randomization

### The problem

The entire validity of an experiment rests on the claim that treatment and control groups are comparable. But users differ systematically in behavior, demographics, and tenure. If heavy users end up disproportionately in treatment, you will see a positive effect that has nothing to do with your change. Randomization is the mechanism that makes the groups comparable in expectation.

### The core insight

Random assignment does not guarantee exact balance on any single variable — it guarantees that any imbalance is attributable to chance, not systematic bias. With large enough samples, this chance imbalance becomes negligible. More importantly, randomization balances on every variable, including ones you cannot measure.

### Choosing the unit of randomization

The unit of randomization should be the unit where the treatment effect is isolated and stable.

**User-level randomization** assigns each user to one arm for the duration of the experiment. This is the default. It ensures each user gets a consistent experience, satisfies SUTVA (see below) for non-social features, and allows measurement of effects that develop over multiple sessions.

**Session-level randomization** assigns each session independently. The same user may see treatment in one session and control in another. Valid only for changes with no carry-over effect — a user's behavior in session 2 is unaffected by which variant they saw in session 1. Rare in practice; usually a sign of implementation convenience overriding experimental validity.

**Request-level randomization** is almost never appropriate for user-facing features. It maximizes power by treating each request as independent, but a single user experiencing both variants within a session contaminates both arms.

### SUTVA — why it matters

**The problem it solves**: if one user's outcome depends on another user's treatment assignment, your model of the experiment is wrong. You assumed each unit's potential outcome is fixed given its own assignment, but it is actually a function of the entire assignment vector. The estimated treatment effect is then not the causal effect of the treatment on an individual.

**The core insight**: Stable Unit Treatment Value Assumption (SUTVA) requires that (1) the potential outcome for unit $i$ depends only on $i$'s own assignment, and (2) there is a single version of each treatment. When SUTVA holds, the estimated treatment effect on your sample generalizes as written. When it does not, the effect you estimate is a net of direct and indirect effects whose composition is ambiguous.

**Where SUTVA breaks**: social features (user A's ranking changes when user B is in treatment), two-sided marketplaces (more drivers in treatment area reduces supply for control area), shared infrastructure (treatment arm consumes more capacity, degrading control arm latency), viral products (treatment users send more invitations, exposing control users to the product change). Detecting SUTVA violations requires domain reasoning, not statistical tests.

**Network effects specifically**: in a social network, user A in control receives content and notifications influenced by user B in treatment, through the social graph. The control arm is contaminated. The measured treatment effect understates the true effect if treatment propagates positively to control, or overstates it if treatment diverts engagement from control. Solutions include cluster randomization (assign entire social communities to one arm), ego-network randomization, or switchback experiments (Section 12).

### Balance check: Sample Ratio Mismatch (SRM)

**The problem**: you configured a 50/50 split but observe 48/52 after the experiment. Is this chance variation or a systematic bug?

**The core insight**: under correct randomization, the observed split should match the configured split within sampling error. A statistically significant mismatch indicates a bug in the assignment or logging pipeline — not a real effect. Analyzing results from a corrupted randomization produces biased estimates, period.

**The test**: apply a $\chi^2$ goodness-of-fit test comparing observed counts per arm to expected counts. If $p < 0.001$ (a conservative threshold given the large sample sizes typical in A/B tests), declare SRM and halt analysis until the root cause is identified. Common causes include: logging that triggers only after user interaction (interaction rates differ by variant), bots being assigned differently than humans, session-based systems that re-assign returning users.

---

## 5. Temporal Confounds

### The problem

You run a two-week experiment, see a positive effect, and ship. A month later, the metric has reverted to baseline. Or the opposite: you see no effect and kill the experiment, but the change would have performed well had you run it longer. Temporal confounds are effects on your metric that are correlated with time in experiment, not just with treatment assignment.

### Novelty effect

**The problem**: users who encounter an unfamiliar UI interact with it more out of curiosity, not because it is better. This inflates short-term treatment metrics.

**The insight**: the novelty effect is a property of change, not of the new design. If the test were run long enough, novelty wears off and the metric stabilizes at its true equilibrium. A two-week experiment that captures novelty is answering a subtly different question: "what is the effect of introducing this change?" not "what is the effect of this change at steady state?"

**Mitigation**: run long enough for frequent users (who have had many exposures) to stabilize. Segment by exposure count and check whether treatment-control difference is shrinking over time. If so, wait for stabilization or report the trend explicitly.

### Primacy effect

**The problem**: existing users are anchored to the old UX. A new UX requires a learning curve before it performs at its potential. Short-run effects are artificially suppressed.

**The insight**: primacy is the mirror image of novelty — it produces a negative initial effect that decays as users adapt. An experiment capturing the primacy period underestimates the steady-state value of the change.

**Distinguishing them**: novelty and primacy are both temporal effects but in opposite directions. Running the experiment longer helps. Separately analyzing new users (who have no baseline anchoring and no novelty) provides a clean signal on the steady-state effect. If new users show a positive effect and returning users show a negative or neutral effect initially, that pattern is consistent with primacy for returning users.

---

## 6. Metrics Framework

### The problem

An experiment moves your primary metric by +0.3%, crashes page load time by 200ms, and slightly reduces 7-day retention. Do you ship? Without a pre-specified framework for how metrics relate to each other, this question dissolves into politics rather than analysis.

### The structure that follows from the problem

**OEC (Overall Evaluation Criterion)**: one metric, specified before launch, that determines the ship decision. You commit to this metric to prevent HARKing — redefining success after seeing results. The OEC should be a leading indicator of long-term user and business value, not a raw count that gaming can inflate.

**Secondary metrics**: provide mechanistic context. If OEC increases, secondary metrics explain why. They are not decision criteria; you do not veto a ship because a secondary metric moved negatively unless you understand why and judge it material.

**Guardrail metrics**: define the region of acceptable experiments. Latency, error rates, crash rates, and revenue are typical guardrails. If any guardrail degrades beyond a threshold, the experiment is blocked regardless of what the OEC does. Guardrails formalize the constraint "we do not ship changes that break core product health."

**Diagnostic metrics**: more granular signals that help debug unexpected guardrail or OEC movements. Not part of the decision framework; part of the investigation toolkit.

### Multiple metrics and false positives

**The problem**: if you test 20 metrics simultaneously at $\alpha = 0.05$, the expected number of false positives is one — even if the treatment does nothing. More variants and more metrics compound this problem.

**The insight**: the significance level $\alpha$ is the per-comparison error rate. When you make $m$ comparisons, the family-wise error rate (probability of at least one false positive) is $1 - (1-\alpha)^m$, which grows rapidly with $m$.

**Corrections**: Bonferroni divides $\alpha$ by $m$, which is conservative but simple. Benjamini-Hochberg controls the false discovery rate (expected proportion of false positives among positives) at level $q$, which is less conservative for large $m$. Both corrections should be applied only to the metrics in the pre-specified primary analysis, not to diagnostic fishing.

---

## 7. Sequential Testing and the Peeking Problem

### The problem

Experiments take weeks. It is tempting to check results daily and stop early if significance is reached. This is catastrophically wrong, and understanding exactly why requires understanding what a p-value actually is.

### The core insight

A p-value computed at sample size $n$ is the probability of seeing a test statistic at least as extreme as observed, assuming the null hypothesis is true and the sample size was fixed at $n$ before data collection began. If you check the p-value at multiple sample sizes and stop at the first significant result, you are no longer using the p-value in the way that justifies its interpretation. You have made multiple looks at the data, each generating a p-value, and selected the smallest. This is multiple comparisons in the time dimension.

**Quantifying the damage**: under the null, the p-value is uniform on [0,1] at any fixed sample size. But if you check 5 times and stop at the first $p < 0.05$, simulation shows the effective false positive rate is approximately 0.19 — nearly four times the nominal 0.05.

### Solutions

**Fixed-horizon with blinded monitoring**: the cleanest approach. Determine sample size in advance, run until that sample size is reached, analyze once. Monitor only for data quality issues (SRM, logging gaps), never for outcomes. This is the standard for regulatory environments and academic experiments.

**Alpha spending functions**: pre-commit to a schedule of $k$ interim looks and distribute the $\alpha$ budget across looks using a spending function. O'Brien-Fleming spends very little $\alpha$ at early looks (strong evidence required to stop early) and reserves most $\alpha$ for the final look. Pocock spends $\alpha$ equally across all looks, making early stopping easier but requiring a more stringent threshold at the final look.

**mSPRT (mixture Sequential Probability Ratio Test)**: provides always-valid p-values — the false positive rate is controlled at level $\alpha$ regardless of when you stop, even if you look after every new observation. The mechanics: instead of comparing the likelihood ratio at a fixed $n$, mSPRT mixes the likelihood ratio over a prior distribution on the effect size. The resulting test statistic is a martingale under the null, which is what makes the always-valid property hold. Widely used in industry (Optimizely's Stats Engine is based on this).

**What breaks with mSPRT**: always-valid p-values come with reduced power relative to a fixed-horizon test at the same final sample size, because the test must hedge against early stopping. If you know you will never stop early, a fixed-horizon test is more powerful.

---

## 8. Variance Reduction: CUPED and CUPAC

### Why variance reduction matters

The power formula tells you that to detect a real effect, you need the signal (true difference $\delta$) to exceed the noise (standard error $\sigma/\sqrt{n}$). You can achieve better power in two ways: increase $n$ (run longer) or reduce $\sigma^2$ (reduce the metric variance). Reducing variance is often dramatically more efficient than adding users, especially when user acquisition is the bottleneck.

### CUPED — the insight

**The problem**: your outcome metric $Y$ (e.g., revenue per user) has high variance. Some of that variance is because users are genuinely different — heavy users will always have higher revenue than light users, regardless of treatment. This between-user variance is noise with respect to the treatment effect. The treatment effect adds a small signal on top of a large background of user heterogeneity.

**The insight behind CUPED**: before the experiment starts, you already have data on each user. If a user's pre-experiment behavior predicts their in-experiment behavior (which it does strongly — heavy users stay heavy users), you can subtract out the predictable component. What remains has lower variance because you have removed a correlated baseline.

**Why this formula**: $Y_{\text{adjusted}} = Y - \theta(X - \bar{X})$, where $X$ is a pre-experiment covariate (often the same metric from a pre-period). You subtract a scaled version of the deviation of $X$ from its mean. The scaling $\theta = \text{Cov}(Y, X) / \text{Var}(X)$ is chosen to minimally project out the component of $Y$ correlated with $X$ — this is the OLS coefficient from regressing $Y$ on $X$. The $\bar{X}$ centering ensures the mean of $Y_{\text{adjusted}}$ equals the mean of $Y$, so the treatment effect estimate is unbiased.

**The variance reduction**: $\text{Var}(Y_{\text{adjusted}}) = \text{Var}(Y)(1 - \rho^2)$, where $\rho = \text{Corr}(Y, X)$. If past behavior explains 50% of future variance ($\rho = 0.7$), you get a 49% variance reduction — meaning you need roughly half the users for the same power. Typical variance reductions in practice: 30–70% depending on metric stability.

**What breaks**: if the covariate $X$ is post-experiment — i.e., measured during or after the treatment period — the adjustment introduces bias, because $X$ itself may be affected by the treatment. The covariate must be strictly pre-treatment. Also, CUPED implicitly assumes a linear relationship between $X$ and $Y$; if the relationship is non-linear, variance reduction is suboptimal.

### CUPAC — extending the insight

**The problem with CUPED**: the pre-experiment value of $Y$ is a powerful but linear predictor of future $Y$. Users' behavior depends on many interacting features. A linear regression on a single covariate leaves residual variance that a richer model could explain.

**The insight behind CUPAC**: use a machine learning model — trained on pre-experiment data — to predict each user's outcome $Y$. This prediction $\hat{Y}$ is the covariate. Because $\hat{Y}$ captures non-linear relationships between many user features and the outcome, it can explain more variance than a single linear covariate.

**The mechanics**: train a model on historical data (features: user attributes, past behavior; target: the outcome metric in a pre-period). Generate out-of-fold predictions to avoid overfitting. Use these predictions as $X$ in the CUPED formula. Since $\hat{Y}$ is purely a function of pre-experiment data, the adjustment remains unbiased.

**What breaks**: if the model is trained on data that includes any signal from the treatment period, predictions are contaminated and the adjustment is biased. Model complexity also introduces infrastructure overhead — CUPAC requires maintaining a prediction pipeline alongside the experiment pipeline.

### Stratified sampling

**The insight**: if the population has identifiable subgroups with different mean outcomes (new users vs returning, mobile vs desktop), between-strata variance inflates overall variance without contributing information about the treatment effect. Stratified randomization assigns proportional numbers of units from each stratum to each arm, eliminating between-strata variance from the treatment effect estimator.

**What breaks**: strata must be defined before randomization. Post-stratification analysis can recover some of the benefit but is more complex and requires careful handling to avoid bias.

---

## 9. Common Pitfalls

### p-hacking

**The problem it creates**: you run a test, check daily, and stop when $p < 0.05$. You report a statistically significant result. The result does not replicate. Why? Because your stopping rule was "stop when significant," which inflates the false positive rate as explained in Section 7. The p-value at the stopping point is not 0.05 — it is much higher.

**Mitigation**: pre-register the sample size and analysis date. Use sequential testing methods if early stopping is required.

### HARKing (Hypothesizing After Results Known)

**The problem**: you run an experiment and observe that a specific segment (say, iOS users in the US) showed a large positive effect even though the overall OEC was flat. You reframe the analysis as if iOS US users were always the intended audience and report the positive result.

**Why this is wrong**: you tested a hypothesis generated from the data, using the same data to test it. This is exploratory analysis dressed as confirmatory analysis. The $p$-value from this test is not a valid Type I error rate for the hypothesis you stated, because you chose the hypothesis by selecting the subgroup with the best observed outcome.

**Mitigation**: pre-register the OEC and analysis plan. Label any post-hoc subgroup analysis as exploratory with no inferential weight.

### Survivorship bias

**The problem**: you analyze only users who returned to the product after day 1, reasoning that day-0 users "didn't really experience the change." But whether a user returns on day 1 can itself be affected by the treatment — a worse experience might cause them to churn. Excluding churned users biases the treatment effect estimate upward.

**Mitigation**: always analyze all users who were exposed to the experiment, including those who subsequently churned. The intent-to-treat estimand is the correct default.

### Simpson's paradox

**The problem**: the aggregate effect shows treatment is better than control. But when you break down by user segment, control is better within every segment. This is not a contradiction — it can happen when the segment composition of treatment and control arms is imbalanced and segments have different baseline outcome levels.

**Mitigation**: check covariate balance across arms after randomization. If significant imbalance exists, use stratification or regression adjustment in the analysis.

### Leakage / contamination

**The problem**: control users are exposed to the treatment. A user randomized to control visits a friend who is in treatment; the friend shares a feature only available in treatment. Or, a server-side feature that should only activate for treatment users activates for some control users due to a caching bug. Contamination dilutes the measured treatment effect toward zero.

**Mitigation**: verify assignment isolation through logging — confirm that treatment users see treatment and control users see control at exposure time, not just at assignment time.

### Statistical significance without practical significance

**The problem**: at very large sample sizes, even a 0.01% improvement is statistically significant. Reporting $p < 0.001$ for a 0.01% effect implies you should ship the change, but the effect is smaller than measurement noise and likely smaller than the cost of maintaining the change.

**Mitigation**: always report the point estimate and confidence interval alongside the p-value. Make the ship decision based on whether the confidence interval overlaps the MDE, not on whether $p < 0.05$.

---

## 10. Bayesian A/B Testing

### The problem with the frequentist framing

After a frequentist test, you can say: "if the null is true, the probability of seeing data this extreme is 0.04." You cannot say: "the probability that treatment is better than control is 96%." But that second statement is what decision-makers need. The frequentist framework provides calibrated error rates for long-run repeated decisions; it does not provide the posterior probability of the hypothesis given the data.

### The core insight

Bayesian inference directly models the uncertainty over the quantity of interest — here, the true conversion rate $\theta$ or the difference $\theta_B - \theta_A$. By starting with a prior distribution over $\theta$ and updating it with observed data via Bayes' theorem, you obtain a posterior distribution that represents your updated belief. You can then compute the probability that treatment is better, or the expected loss from choosing the wrong arm.

### Beta-binomial model

For conversion rate experiments, the natural prior is $\text{Beta}(\alpha_0, \beta_0)$. The Beta distribution is conjugate to the Bernoulli likelihood, meaning the posterior after observing $k$ conversions in $n$ trials is $\text{Beta}(\alpha_0 + k, \beta_0 + n - k)$. This closed form requires no MCMC.

**Why this shape**: $\alpha_0$ can be thought of as pseudo-successes and $\beta_0$ as pseudo-failures in the prior. Weak priors use small values (e.g., $\alpha_0 = \beta_0 = 1$, uniform); informative priors encode historical conversion rate knowledge.

**Posterior probability of being best**: $P(\theta_B > \theta_A)$ is computed by Monte Carlo — draw many samples from both posteriors and compute the fraction where the treatment sample exceeds the control sample. No closed form is needed, and the interpretation is direct: a 90% posterior probability means, given your prior and the data, you believe there is a 90% chance treatment is better.

**Expected loss**: $E[\max(\theta_A - \theta_B, 0)]$ is the expected cost of choosing treatment when control is actually better. Stop when expected loss falls below a business-set threshold. This framing matches the actual decision problem: you care about the magnitude of the error, not just whether an error occurred.

### What breaks

Bayesian tests are sensitive to the prior for small samples. An informative prior that is wrong in direction can require many observations to overcome. The prior must be set before seeing the data; choosing a prior after seeing results is equivalent to HARKing in the frequentist world. Also, the beta-binomial model assumes conversions are independent Bernoulli trials — variance underestimation occurs if users have multiple sessions and their session-level conversions are correlated.

---

## 11. Multi-Armed Bandits

### The problem pure A/B testing creates

A pure A/B test runs a 50/50 split for the entire experiment duration, then ships the winner. During the experiment, roughly half the users experience the inferior variant. If the experiment runs for two weeks, you have two weeks of users receiving a worse experience in service of learning. This is called regret — the cost of exploration.

### The core insight

If you already have strong evidence that arm B is better than arm A after day 3, why are you still sending 50% of users to arm A on day 10? A bandit algorithm allocates traffic dynamically: arms that are performing well get more traffic; arms performing poorly get less. This reduces regret — fewer users experience the inferior option — while still gathering enough data on underexplored arms to avoid locking in early.

### Why this creates a tradeoff

By reducing exploration of inferior arms, you also reduce the precision with which you can estimate their true performance. If the apparent ranking reverses late in the experiment, you may have insufficient data to detect it. Bandits optimize for low regret, not for precise effect size estimation. The tradeoff is: bandits are better when you care about user outcomes during the experiment; pure A/B is better when you care about learning the true effect size to inform future decisions.

### Algorithms

**ε-greedy**: with probability $\epsilon$, choose an arm uniformly at random (explore); with probability $1-\epsilon$, choose the arm with the highest estimated mean (exploit). Simple to implement but crude — it wastes exploration budget on clearly inferior arms.

**Upper Confidence Bound (UCB)**: choose the arm that maximizes $\hat{\mu}_k + c\sqrt{\ln t / n_k}$, where $\hat{\mu}_k$ is the estimated mean of arm $k$, $t$ is the total number of rounds so far, and $n_k$ is the number of times arm $k$ has been pulled. The second term inflates the estimated value of arms that have been tried fewer times — the "optimism in the face of uncertainty" principle. UCB is deterministic and achieves near-optimal regret bounds.

**Thompson Sampling**: maintain a posterior distribution over the reward parameter of each arm. At each round, sample one value from each arm's posterior and pull the arm with the highest sample. As data accumulates, posteriors concentrate and the algorithm exploits the best arm with increasing probability. Thompson Sampling is Bayesian, naturally integrates prior knowledge, and achieves near-optimal regret empirically without the tuning parameter $c$ required by UCB.

### When to use bandits vs. A/B

Use bandits when: the experiment has many arms (more than 2–3), the per-user cost of exposure is high, feedback is immediate, and you care about minimizing regret during the experiment. Typical use cases: ad creative selection, push notification content, real-time recommendation ranking.

Use pure A/B when: you need to estimate the precise causal effect (bandits' non-random allocation makes causal inference harder), outcomes are delayed (long attribution windows make bandit updates noisy), you have regulatory requirements for controlled trials, or you are testing fundamental product changes where understanding mechanism matters as much as outcome.

---

## 12. Switchback Experiments

### The problem randomizing users does not solve

On a two-sided marketplace — ridesharing, food delivery, logistics — every unit affects every other unit through shared supply and demand. If you put half the drivers in treatment (a new dispatch algorithm), those drivers are competing for riders with control drivers. Riders in the control area are being served partly by treatment drivers. There is no clean randomization boundary at the user level; SUTVA is violated by construction.

### The core insight

If user-level randomization is impossible because units share a pool, randomize time instead. Alternate the entire platform between treatment and control across time windows (e.g., 30-minute slots). During a treatment window, every driver and rider experiences treatment. During a control window, everyone experiences control. The comparison is between time periods, not between people.

### Why this works

In a switchback, the unit of randomization is the time window, not the user. SUTVA applies to time windows under the assumption that spillover between adjacent windows is limited. The validity assumption shifts from "user A's outcome is unaffected by user B's assignment" to "what happens in the 9am window is largely unaffected by what happened in the 8:30am window."

### The analysis

Compare outcomes (e.g., average wait time, utilization rate) in treatment windows vs control windows, controlling for time-of-day and day-of-week fixed effects. The regression model is:

$$Y_t = \alpha + \beta \cdot \mathbb{1}[\text{treatment window}] + \text{time-of-day FE} + \text{day-of-week FE} + \varepsilon_t$$

The coefficient $\beta$ is the estimated average treatment effect.

### What breaks

Carry-over effects: treatment in window $t$ affects window $t+1$ because drivers repositioned themselves, queue state changed, or user behavior was altered. This violates the independence assumption between windows. Longer windows reduce carry-over (the system has more time to equilibrate between periods) but reduce the number of experimental units (time windows), reducing statistical power. Choosing window length is a bias-variance tradeoff with no universal answer — it requires domain knowledge about the system's equilibration time. Adding buffer periods between treatment and control windows (where data is discarded) can reduce carry-over at the cost of fewer usable observations.

---

## 13. Holdout Groups

### The problem shipping experiments creates

Your team ships 5 features per month, each individually positive. After 6 months, you notice user engagement has not grown as expected. Is there interaction between the features? Are some effects time-limited? Did some features create latent damage not visible in short experiments? Standard A/B tests answer "did this feature move the OEC in a two-week window?" but not "what is the compounding effect of 12 months of product changes?"

### The core insight

Keep a small fraction of users permanently in a frozen control state — they never receive any new features. After 6 or 12 months, compare their outcomes to the rest of the user base. The difference represents the cumulative impact of everything shipped during that period. This is a holdout group.

### What it measures that experiments cannot

Individual A/B tests measure marginal effects under the assumption that the baseline is fixed. Holdouts measure the integral of all changes under realistic compound conditions. A feature that looks neutral in a two-week test may show a negative effect in a holdout if it increases short-term engagement at the cost of long-term retention — a signal that only emerges over months.

### Cannibalization detection

A holdout for product area A can detect whether changes in area A cannibalize engagement in area B. If users in the holdout (who do not receive area A changes) show higher engagement in area B than users who received area A changes, the changes cannibalizer. This cross-product effect is invisible in A/B tests that only measure metrics in the area under test.

### What breaks

**Opportunity cost**: holdout users are permanently excluded from potentially beneficial features. As the holdout accumulates more "missed" changes, this cost grows. Typical holdout size is 1–5%; larger holdouts increase measurement precision but increase opportunity cost.

**Divergence**: after many months, holdout users have had a fundamentally different product experience from the rest of the user base. The comparison between holdout and non-holdout becomes less interpretable because the two groups are experiencing different products, not just different versions of the same feature. Holdouts should be refreshed periodically (the old holdout is released and a new one is constituted).

**Novelty contamination**: when the holdout is eventually released and receives all withheld features simultaneously, the resulting behavior spike is contaminated by novelty effects. Holdout release should be gradual.
