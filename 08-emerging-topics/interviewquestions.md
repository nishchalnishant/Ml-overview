---
module: Emerging Topics
topic: Experimentation and Causal Inference
subtopic: Interview Questions
status: unread
tags: [emergingtopics, ml, experimentation-and-causal-inference, interview-questions]
---
# Experimentation & Causal Inference — Interview Questions

> Companion Q&A bank to [experimentation-and-causal-inference/01-experimentation-and-causal-inference.md](experimentation-and-causal-inference/01-experimentation-and-causal-inference.md). That file teaches the concepts top to bottom; this file drills them in interview format — definitions, "X vs Y" tradeoffs, and applied/scenario questions an interviewer would actually ask.

> Questions are organized by difficulty (Easy → Medium → Hard) rather than by topic, so you can calibrate expectations for the seniority level you're interviewing at. Topic tags in brackets after each question indicate which area it draws from.

---

## Easy

### Q: What is an A/B test, and why is randomization the key ingredient? [Fundamentals]

An A/B test is a randomized controlled experiment: users (or other units) are randomly split into a control group (existing experience) and one or more treatment groups (new experience), and outcomes are compared. Randomization matters because it makes treatment assignment independent of all potential outcomes:

```
(Y(0), Y(1)) ⊥ T
```

This guarantees that, in expectation, the only systematic difference between groups is the treatment itself — so any observed difference in outcomes can be attributed to the treatment rather than to pre-existing differences between the kinds of users who ended up in each group (selection bias).

**Gotcha**: randomization removes confounding *in expectation*, not in every single run. With small samples you can still get unlucky imbalance on an important covariate — always check covariate balance (e.g., via A/A tests or pre-period metrics) rather than assuming randomization solved everything.

---

### Q: What is statistical power, and what happens if a test is underpowered? [Statistical Concepts]

Power = 1 − β = P(reject H₀ | H₀ is false) — the probability of detecting a real effect of a given size. An underpowered test (e.g., stopped early or run on too few users) has a high chance of a false negative: a real effect exists but the test fails to reach significance, and the team wrongly concludes "no effect" and kills a good change.

**Gotcha**: a non-significant result is not evidence of no effect if the test was underpowered for the effect size in question — always report the MDE the test *could* have detected at the sample size achieved, not just "not significant."

---

### Q: What are guardrail metrics and why do you need them alongside your primary metric? [Fundamentals]

Guardrail metrics are metrics you must not regress even if the primary metric improves — latency, crash rate, error rate, unsubscribe rate, support ticket volume. A change might lift conversion by making the checkout flow more aggressive but also spike complaints or slow the page down (which itself has a treatment effect on conversion).

**Gotcha (follow-up interviewers love)**: "your test shows a significant lift in the primary metric — what else do you check before shipping?" Answer: guardrails, segment-level effects (Simpson's paradox), novelty decay over time, statistical significance vs practical significance, and whether the sample was actually random (SRM check, see below).

---

### Q: What is Sample Ratio Mismatch (SRM) and why is it a red flag? [Fundamentals]

SRM is when the observed ratio of users across arms deviates significantly from the intended allocation (e.g., you configured 50/50 but observe 52/48 with a huge sample, which is statistically impossible under correct randomization). You test it with a simple chi-squared goodness-of-fit test against the expected ratio.

SRM indicates a bug in the experiment pipeline — e.g., one variant's page load fails and users bounce before being logged, differential bot filtering, or non-atomic assignment across some funnel step — not a covariate imbalance you can fix statistically. **Any experiment with SRM should be considered invalid** regardless of how significant the metric result looks, because the assignment mechanism itself is broken and the two groups are no longer comparable populations.

**Gotcha**: SRM is often caused by something correlated with the outcome (e.g., slower variant times out more on low-end devices), which means the metric bias isn't random — it's directional and can make a neutral or negative feature look like a winner.

---

### Q: What does a p-value of 0.03 actually mean, and what are the two most common misinterpretations? [Statistical Concepts]

It means: if the null hypothesis (no true effect) were true, you would observe a result at least this extreme 3% of the time due to sampling noise alone. It is evidence *against* the null, not a direct probability that the alternative is true.

Common misinterpretations:
1. "There's a 97% chance the treatment works" — wrong; that's confusing P(data | H₀) with P(H₀ | data), a Bayesian quantity the frequentist p-value doesn't give you.
2. "The effect is large/important" — wrong; with enough sample size, even a trivial, practically meaningless effect becomes statistically significant. Statistical significance is about confidence in the sign/existence of an effect, not its magnitude.

**Gotcha**: ask candidates to distinguish Type I error (false positive, rate α) from Type II error (false negative, rate β) and to state that power = 1 − β.

---

### Q: What's the difference between statistical significance and practical significance, and how do you operationalize the latter? [Statistical Concepts]

Statistical significance answers "is this effect distinguishable from zero given the noise level and sample size?" Practical significance answers "is this effect large enough to matter for the business/user?" You operationalize practical significance by defining an MDE or a minimum meaningful effect *before* the test (e.g., "we only care about lifts ≥ 1% because anything smaller isn't worth the engineering cost to maintain this code path"), and by looking at the confidence interval, not just the point estimate — if the CI is [0.01%, 0.02%], it's statistically significant but practically dead.

**Gotcha**: this is exactly why sample size should be based on the smallest *practically* meaningful effect (MDE), not "as much traffic as we have" — otherwise you'll detect tiny effects that don't warrant action, and every large-traffic experiment "wins."

---

### Q: What is Simpson's Paradox and how would you detect it in an A/B test? [Statistical Concepts]

Simpson's Paradox is when a trend present in aggregated data reverses or disappears when the data is split by a subgroup/confounding variable. Classic example: UC Berkeley 1973 admissions — men were admitted at a higher aggregate rate than women, but within nearly every department women had an equal or higher admission rate; women simply applied to more competitive departments in higher proportion.

In A/B testing, this happens when (1) treatment and control differ in subgroup composition (e.g., different mobile/desktop mix due to a randomization or logging bug), (2) the metric differs across those subgroups, and (3) subgroup membership correlates with the arm. The aggregate result can then show a lift (or a loss) that doesn't hold, or even reverses, within every individual subgroup.

**Detection**: always segment results by key dimensions (platform, geography, new vs. returning user) and compare to the aggregate story. **Fix**: stratified/blocked randomization so subgroup mix is balanced by design, or explicitly control for the confound in the analysis (e.g., regression with subgroup fixed effects, or a weighted average of subgroup-level effects).

**Gotcha**: Simpson's Paradox is often the real explanation behind "aggregate metric moved but I can't explain why" — always ask "did the traffic mix shift during the test?" before trusting an aggregate result.

---

### Q: State the fundamental problem of causal inference and define ITE, ATE, and ATT. [Causal Inference]

For each unit `i`, define potential outcomes `Y_i(1)` (if treated) and `Y_i(0)` (if not). The Individual Treatment Effect is `ITE_i = Y_i(1) - Y_i(0)`. The fundamental problem: you can only ever observe one of `Y_i(1)` or `Y_i(0)` for a given unit — the other is a counterfactual, forever unobserved. So ITE is not identifiable from data for any single unit.

What *is* identifiable, under the right assumptions, are population-level averages:
- **ATE** = `E[Y(1) - Y(0)]` — average over the whole population.
- **ATT** = `E[Y(1) - Y(0) | T=1]` — average over just the units that actually received treatment (often what observational methods like matching most directly estimate).

**Gotcha**: ATE and ATT can differ substantially if treatment effects are heterogeneous and treatment assignment correlates with the size of the effect (e.g., people who self-select into a program tend to be those who'd benefit most) — always clarify which estimand a method targets before comparing results across methods.

---

### Q: What is a confounder, formally, and why does it bias a naive comparison? [Causal Inference]

Z is a confounder of the effect of T on Y if Z causes T (`Z → T`) and Z causes Y (`Z → Y`). This creates a "backdoor path" `T ← Z → Y` — a non-causal, purely associative channel between T and Y that a naive comparison of treated vs. untreated groups cannot distinguish from the true causal effect.

Worked example: does LinkedIn Premium cause more job offers? Seniority is a confounder — senior, proactive professionals are both more likely to pay for Premium *and* more likely to receive offers regardless of Premium. A naive `E[offers | Premium] - E[offers | no Premium]` overstates Premium's true causal effect because it's partly just picking up the seniority difference between the groups.

**Gotcha**: a variable that is caused *by* treatment (a mediator, `T → M → Y`) or that is a common effect of T and Y (a collider) is *not* a confounder, and controlling for it introduces bias rather than removing it — see the DAG questions below.

---

### Q: What is CATE, and why would you want it instead of just ATE? [Uplift Modeling]

Conditional Average Treatment Effect: `CATE(x) = E[Y(1) - Y(0) | X=x]` — the average treatment effect within the subpopulation sharing covariate values x. ATE is a single population-wide number; it can mask enormous heterogeneity — a marketing campaign might have a strongly positive effect on some users, zero effect on others, and even a negative effect on a third group, all averaging out to a modest positive ATE that hides these different underlying stories.

Practically, in a marketing/targeting context, you often care less about "does this work on average" and more about "who should I target" — which requires CATE (or an uplift score), not ATE.

**Gotcha**: interviewers may ask you to name the four canonical response types in an uplift context — Persuadables (converts only if treated — the actual target), Sure Things (convert regardless — wasted spend if targeted), Lost Causes (never convert), and Sleeping Dogs (convert *less* if treated — actively harmful to target, e.g., a discount email that reminds an otherwise-loyal customer to comparison-shop).

---

### Q: What is an A/A test, and what does it tell you that an A/B test cannot? [Pitfalls]

An A/A test randomly splits users into two groups but shows *both* groups the identical experience (no real treatment). Since there's no true effect, you'd expect roughly a `α` (e.g., 5%) rate of "significant" results across many repeated A/A tests purely by chance.

It's a diagnostic tool for the experimentation *platform* itself, not for any specific product hypothesis: it validates that randomization is working correctly (no SRM), that variance estimates and confidence intervals are well-calibrated (if A/A tests reject the null far more than 5% of the time, something in your logging/metric pipeline or randomization is broken), and it gives you an empirical estimate of a metric's natural variance/noise floor, useful for more realistic sample size planning than textbook formulas alone.

**Gotcha**: a common mistake is treating an occasional A/A "significant" result as alarming on its own — under the null, you *expect* about 1 in 20 A/A tests to cross p < 0.05 by chance; the concern is only if the rate across many A/A tests is systematically higher than the nominal α, which would indicate a pipeline or calibration bug.

---

### Q: How does the Bayesian A/B testing framework differ conceptually from frequentist testing? [Bayesian & Bandits]

Frequentist: fix a null hypothesis, compute the probability of observing data this extreme *if the null were true* (p-value), and reject or fail to reject at a pre-chosen α. It's a statement about the sampling distribution of the test statistic under repeated experiments.

Bayesian: put a prior distribution on the unknown conversion rates (e.g., `θ ~ Beta(α, β)`), update to a posterior using observed data via Bayes' rule, and report `P(θ_treatment > θ_control | data)` directly — which is the actual question stakeholders want answered ("what's the probability B is better?"). The posterior is valid to inspect at any sample size, so there's no peeking penalty in the same sense as frequentist sequential looks (though you should still account for optional-stopping effects on decision quality/cost, even if the posterior itself remains a valid summary of current evidence).

**Worked example (Beta-Binomial)**: prior `Beta(1,1)` (uniform), observe `x` conversions out of `n`, posterior is `Beta(1+x, 1+n-x)`. Sample from both arms' posteriors many times and compute the fraction of draws where treatment > control — that fraction *is* `P(treatment better)`.

**Gotcha**: Bayesian results still depend on the prior, especially with small samples — an interviewer may ask "what if you pick an overly informative prior?" Answer: it can bias the posterior toward the prior mean when data is sparse; use weakly informative or empirically-calibrated priors (e.g., from historical experiments) and check sensitivity to the prior choice.

---

## Medium

### Q: Explain the difference between Value-based and Policy-based RL, and why PPO is currently the industry default. [Reinforcement Learning]
A: **Value-based RL** (like Q-learning or DQN) learns to estimate the expected cumulative reward of taking a specific action in a specific state, and the policy simply picks the action with the highest estimated value. It struggles with continuous action spaces and stochastic environments. **Policy-based RL** (like REINFORCE) directly parametrizes the policy (the probability distribution over actions given a state) and updates it via gradient ascent to maximize expected reward. This handles continuous actions and stochastic policies well but can suffer from high variance and unstable training. **PPO (Proximal Policy Optimization)** is an actor-critic method (combining both) that clips the policy update to prevent the new policy from changing too wildly from the old policy in a single step. This clipping provides the stability of trust-region methods (like TRPO) but with the simplicity and computational efficiency of first-order gradient descent, making it the industry default for both game AI and LLM alignment (RLHF).

### Q: What is a Graph Neural Network (GNN), and what problem does it solve that a standard CNN or MLP cannot? [Graph Neural Networks]
A: An MLP assumes input features are independent and have no inherent order. A CNN assumes inputs exist on a regular, Euclidean grid (like pixels in an image or frames in audio) where local proximity implies relationship. But much of the real world—social networks, molecules, protein interactions, or player interactions in a game engine—is structured as a graph: nodes connected by edges of arbitrary topology. A **GNN** solves this by using "message passing." Each node updates its own embedding by aggregating the features of its immediate neighbors. After $k$ layers of message passing, a node's embedding contains structural and feature information from its $k$-hop neighborhood. This allows the network to learn representations that are invariant to graph isomorphism (node ordering doesn't matter) and can scale to graphs of varying sizes, which neither an MLP nor a CNN can do.


### Q: Walk through how you'd size an A/B test before launching it. [Fundamentals]

You need four inputs: baseline metric value, minimum detectable effect (MDE), significance level (α), and power (1 − β). For a binary metric (e.g. conversion rate), sample size per group is:

```
n = [ z_(α/2) * sqrt(2 p̄(1-p̄)) + z_β * sqrt(p₀(1-p₀) + p₁(1-p₁)) ]² / (p₁ - p₀)²
```

where `p₀` is baseline, `p₁ = p₀(1 + MDE)` is the target treatment rate, and `p̄` is the pooled proportion.

Worked example: baseline conversion 3%, want to detect a 5% relative lift, α = 0.05 (two-sided), power = 0.80 → roughly 60,000 users per group. Divide by daily traffic to get test duration.

**Gotcha/follow-up**: interviewers will ask what happens if you halve the MDE — sample size roughly quadruples (it scales with `1/effect²`), which is why teams are tempted to accept a bigger MDE (and thus miss smaller but real effects) rather than run longer. Also: sample size formulas assume independent observations — if you have repeated measures per user or cluster-correlated data, effective sample size is smaller than raw n (design effect).

---

### Q: What's the difference between the unit of randomization and the unit of analysis, and why does it matter? [Fundamentals]

Unit of randomization is what gets assigned to treatment/control (usually the user, via a hash of user ID). Unit of analysis is what you compute the metric over (could be user, session, or event). If you randomize by user but a metric is computed per-session, and users generate different numbers of sessions in each arm, the effective sample size and variance calculations get complicated — sessions aren't independent within a user.

Also, randomizing at the wrong grain causes contamination: if you randomize by session but a user can appear in both control and treatment sessions, that user's baseline behavior may leak into both arms, biasing the estimate toward zero.

**Gotcha**: always randomize at the coarsest level where SUTVA is more plausible (e.g., user, not pageview) and pick metrics/analysis that match — often this requires clustering standard errors by the randomization unit.

---

### Q: You ran an A/B test and got p = 0.04 with a 2% relative lift on a metric with millions of users. Should you ship? [Fundamentals]

Not automatically. First check: is 2% practically meaningful given the cost of maintaining/shipping the change? With millions of users, even a 0.1% true effect will be statistically significant, so statistical significance alone is a low bar at that scale. Second: check guardrails and segments (did it help mobile but hurt desktop?). Third: check for novelty effects — is the lift stable across the whole experiment window, or front-loaded and decaying? Fourth: confirm no SRM and that the metric's confidence interval, not just the point estimate, supports a decision (e.g., a 95% CI of [0.1%, 3.9%] is a much weaker case than [1.5%, 2.5%]).

**Gotcha**: interviewers often follow up with "the lift is significant but a related core business metric (e.g., revenue) didn't move — why?" — see the dedicated pitfall question below.

---

### Q: What is the Bonferroni correction, and when would you prefer the Benjamini-Hochberg (FDR) procedure instead? [Statistical Concepts]

Bonferroni: divide α by the number of tests m, so each individual test uses `α/m`. It controls the family-wise error rate (probability of *any* false positive across all tests) but is conservative — power drops fast as m grows.

Benjamini-Hochberg: controls the false discovery rate (expected proportion of rejected hypotheses that are false positives), which is a less strict criterion and yields more power when you're running many tests (e.g., 50 metrics or a large-scale experimentation platform running hundreds of experiments).

**Rule of thumb given in this repo's material**: use Bonferroni when a single false positive is very costly (e.g., a safety-critical guardrail metric); use BH for exploratory analysis where you can tolerate some false discoveries in exchange for more power (e.g., scanning many secondary/diagnostic metrics for interesting signals).

**Gotcha**: multiple testing isn't just about explicit statistical tests — running a single experiment but looking at 10 different metrics, or slicing by 5 segments, is still multiple testing and needs correction (or explicit designation of one primary metric decided in advance).

---

### Q: What is SUTVA, how is it violated in social/networked products, and how do you fix it? [Statistical Concepts]

The Stable Unit Treatment Value Assumption requires: (1) one unit's treatment doesn't affect another unit's outcome (no interference), and (2) there's only one version of the treatment. Violation example: testing a new notification system on a social network — treated users post more, which increases the content control users see in their feed, making control users more active too. This contaminates the control group, which now partially reflects the treatment effect, biasing the *measured* effect toward zero (attenuation bias) even though the true effect exists.

**Fixes**: cluster/graph-cluster randomization (randomize whole geographic regions or densely-connected social communities rather than individual users, so spillover happens mostly within an arm rather than across arms), ego-network randomization, or time-based (switchback) designs where the whole system gets treatment or control for alternating time windows.

**Gotcha**: cluster randomization increases the effective variance of the estimate (fewer independent units = clusters, not individual users) — you need many more clusters, and standard errors must be clustered accordingly, or you'll understate uncertainty.

---

### Q: What are novelty effects and change aversion, and how do you distinguish them from a true treatment effect? [Statistical Concepts]

Novelty effect: users engage more with a new feature simply because it's new/different, not because it's better — the lift decays over time as novelty wears off. Change aversion: users initially perform worse on a redesigned (but objectively better) experience because they're unfamiliar with it, then improve as they adapt — the effect can *understate* the true long-run benefit early on.

**Detection**: plot the treatment effect (or the metric gap between arms) over time within the experiment, segmented by new cohorts entering during the test vs. users who've been exposed since day 1. If the effect decays (novelty) or grows (change aversion wearing off) over the experiment window, that's the signature.

**Fix**: run the experiment long enough for behavior to stabilize (2-4 weeks is a common rule of thumb for daily-active-user products), and consider looking only at users newly exposed within a fixed recent window to get a "steady-state" read, separate from the legacy cohort's trajectory.

**Gotcha**: novelty and change aversion pull the same lever (biased short-run reads) but in *opposite* directions, so a flat, non-significant aggregate effect over a short test could actually be hiding a real underlying effect that novelty and change aversion are canceling out on different user segments.

---

### Q: Compare multi-armed bandits and A/B tests. When would you choose one over the other? [Bayesian & Bandits]

| Criterion | A/B Test | Bandit |
|---|---|---|
| Goal | Unbiased causal estimate | Maximize cumulative reward while learning |
| Traffic to losing arm | Fixed, often 50/50 (wasteful) | Dynamically reduced as evidence accumulates |
| Statistical inference | Well-established, clean CIs | Harder — adaptive sampling breaks classical inference |
| Many variants | Multiple-testing burden grows | Naturally handles many arms |
| Non-stationary environment | Not designed for it | Can adapt (sliding windows, discounting) |

Use A/B tests when you need a clean, defensible causal estimate for a consequential, one-time decision (e.g., "should we ship this pricing change company-wide?") where you can tolerate the exploration cost. Use bandits when you have many candidate variants, frequent/continuous decisions, and care primarily about cumulative outcomes during the learning process itself (e.g., picking the best headline out of 20 for an article, real-time bid/price optimization).

**Gotcha**: bandits make post-hoc statistical inference on "was arm B truly better" harder because the sample sizes per arm are themselves a function of the observed data (adaptive sampling bias) — if you need a rigorous causal claim afterward, that's a strike against pure bandits; some teams run a short A/B phase to validate before switching to a bandit for ongoing optimization.

---

### Q: Explain propensity score matching and the theorem that justifies reducing to a scalar. [Causal Inference]

The propensity score is `e(X) = P(T=1 | X)` — the probability of receiving treatment given covariates. Rosenbaum & Rubin (1983) proved that if treatment is ignorable given the full covariate vector X, it's also ignorable given just the scalar `e(X)`:

```
(Y(0), Y(1)) ⊥ T | e(X)
```

This means instead of trying to find control units that match a treated unit on dozens of covariates simultaneously (curse of dimensionality), you can estimate `e(X)` with a model (commonly logistic regression) and match treated/control units whose propensity scores are close (e.g., nearest-neighbor matching within a caliper, often 0.05-0.2 standard deviations of the logit of the propensity score).

**Balance check**: after matching, verify the standardized mean difference (SMD) on each covariate: `SMD = (x̄_t - x̄_c) / sqrt((s²_t + s²_c)/2)`; SMD < 0.1 is generally considered good balance. If balance isn't achieved, the matching/model specification needs revision (e.g., add interaction terms, try a different caliper).

**Gotcha**: PSM only balances *observed* covariates — it does nothing for unmeasured confounders, so it rests on the same untestable ignorability assumption as any other observational adjustment. It also discards unmatched units, which can hurt precision and shift the estimand toward a subpopulation with common support.

---

### Q: Compare propensity score matching, inverse probability weighting (IPW), and the doubly robust estimator. [Causal Inference]

- **Matching**: pairs each treated unit with similar-propensity control unit(s); only uses matched units, discards the rest; ATT estimate is simply the mean outcome difference within matched pairs.
- **IPW**: instead of discarding data, reweights every observation by the inverse of its propensity score: `ATE_IPW = (1/n) Σ [T_i Y_i / e(X_i) - (1-T_i) Y_i / (1-e(X_i))]`. Uses the full sample, but is unstable (huge variance) when propensity scores are near 0 or 1, since you're dividing by a near-zero number.
- **Doubly robust**: combines a propensity model and an outcome regression model; the estimator remains consistent if *either* model (not necessarily both) is correctly specified — giving you "two chances" to get the causal estimate right.

**Gotcha**: interviewers often ask "what breaks IPW in practice?" — extreme propensity scores near 0/1 (poor overlap/common support), which happens when treatment is nearly deterministic given some covariate combination; trimming or stabilized weights are common mitigations, but the doubly robust approach is generally preferred in practice because of its robustness property.

---

### Q: What problem does difference-in-differences solve that propensity score matching cannot? [Causal Inference]

PSM only adjusts for *observed* confounders captured in X. Many real confounders are unobserved (e.g., unmeasured local economic conditions, baseline motivation). DiD sidesteps this for *time-invariant* unobserved confounders by using each unit's own pre-treatment period as a baseline and differencing out anything constant over time within a group.

```
DiD = (Ȳ_treated,post - Ȳ_treated,pre) - (Ȳ_control,post - Ȳ_control,pre)
```

Classic example (Card & Krueger, 1994): New Jersey raised its minimum wage in 1992; Pennsylvania did not. Comparing raw post-period employment levels between states would be confounded by any structural difference between NJ and PA. DiD instead compares the *change* in employment in NJ to the *change* in PA — famously finding no employment decrease, challenging conventional labor-economics predictions.

**Regression form**: `Y = β₀ + β₁·treated + β₂·post + β₃·(treated×post) + ε`, where `β₃` is the DiD estimator.

**Gotcha**: DiD only removes *time-invariant* confounding — if an unobserved confounder itself changes differently over time between the groups (e.g., NJ and PA had diverging economic trends unrelated to the minimum wage law), DiD is still biased. This is exactly what the parallel trends assumption is meant to rule out.

---

### Q: Compare the S-learner, T-learner, and X-learner approaches to CATE estimation. [Uplift Modeling]

- **S-learner** ("single" model): fit one model `f(X, T) → Y` with treatment as just another input feature, then estimate CATE as `f(X, T=1) - f(X, T=0)`. Simple, but if the model regularizes treatment's coefficient toward zero (common with many features and few treated units), it can badly underestimate treatment effect heterogeneity — the model may effectively "ignore" T if it's not a strong global predictor.
- **T-learner** ("two" models): fit separate outcome models on the treated group and the control group independently, then CATE = `f_treated(X) - f_control(X)`. Avoids the S-learner's shrinkage problem but can be noisy if the treated and control groups are very different sizes, since each model is trained on much less data than the S-learner's pooled model.
- **X-learner**: designed for imbalanced treatment/control group sizes. Roughly: impute treatment effects for each unit using the *other* group's model (e.g., for a treated unit, effect ≈ observed outcome − control model's prediction at that X), fit models to predict these imputed effects separately on each side, then combine the two CATE estimates with a weighting function (often based on the propensity score) that leans more on whichever model was trained on more data.

**Gotcha**: none of these fundamentally solve the identification problem — they all still require the ignorability assumption for the underlying outcome models to give valid CATE estimates (or a valid instrument/experimental design). Meta-learners are an estimation strategy, not a free pass around confounding.

---

### Q: Why is testing a new ML model different from testing a UI change, and what's a subtle treatment effect people forget to measure? [ML Systems]

Key differences: the "treatment" itself is invisible to end users (they don't see a model, they see its outputs), models can interact with each other and with the item/content pool (e.g., a new recommender changes what other models see as available inventory or as training data for tomorrow), and effects can be highly non-linear and hard to anticipate from offline metrics.

**The subtle, commonly forgotten treatment effect: latency.** If Model B is more accurate than Model A but adds 100ms of inference latency, that latency itself has a causal effect on the outcome metric (users bounce, conversions drop) that's entangled with the model-quality effect. If you don't measure and control for latency separately, you can't tell whether an observed lift/drop is due to model quality or the incidental speed change — always report both accuracy-related and latency/guardrail metrics for model A/B tests.

**Gotcha**: this generalizes — any change bundles *all* of its side effects (latency, memory footprint, downstream cache invalidation) into a single "treatment," so a full causal read requires either isolating these dimensions in separate tests or explicitly measuring and reporting them as covariates/guardrails.

---

### Q: Why do offline metric improvements (AUC, RMSE) often fail to translate into online metric improvements (CTR, conversion)? Give the main causes and a mitigation for each. [ML Systems]

1. **Distribution shift**: the offline held-out set reflects historical traffic, which may not represent future production traffic (especially after the new model itself changes user behavior). *Mitigation*: use the most recent data for evaluation, monitor for drift, and validate on a rolling/time-based split rather than a random split.
2. **Label proxies**: offline labels (e.g., "did they click") are imperfect proxies for the true objective (e.g., long-term satisfaction). *Mitigation*: align offline labels as closely as possible to the online north-star metric, and validate that offline metric movement historically correlates with online metric movement.
3. **Feedback loops**: the model's own predictions affect future user behavior and future training data, creating a loop that a static offline evaluation cannot capture. *Mitigation*: shadow mode / online monitoring; periodically retrain and re-validate against fresh interaction data.
4. **Position bias**: items in top positions get more engagement regardless of true relevance, inflating the apparent quality of whatever model/policy happened to generate the historical logs used for offline eval. *Mitigation*: use counterfactual/off-policy evaluation techniques (e.g., inverse propensity scoring on logged bandit feedback) that correct for the logging policy, or use interleaving online.
5. **Missing counterfactuals**: you can only evaluate offline on items that were actually shown historically, not the full space of items the new model might surface. *Mitigation*: online experimentation is ultimately required to see genuinely novel candidate outputs.

**Gotcha**: the takeaway interviewers want to hear is "define the online metric first, then design/validate the offline proxy metric against it" — not the reverse (optimizing an offline metric and hoping it transfers).

---

### Q: Compare shadow mode, champion-challenger, and canary deployment for rolling out a new model. [ML Systems]

- **Shadow mode**: the new model runs in production and produces predictions, but those predictions are never acted upon (only logged/compared to the current model's live outputs). Validates infrastructure correctness, latency behavior, and prediction distribution — but tells you nothing about actual user-facing impact, since users never see the new model's decisions.
- **Champion-challenger**: the current model ("champion") serves most traffic, and the new model ("challenger") serves a small held-out slice (e.g., 5%) whose real outcomes are measured and compared — this is effectively a live, small-scale A/B test.
- **Canary deployment**: similar mechanism to champion-challenger, but framed as a staged *rollout* — start the new model on a small traffic percentage, monitor guardrails, and progressively ramp up traffic share if metrics hold, until it fully replaces the champion (or is rolled back).

**Gotcha**: shadow mode is often mistakenly treated as "we validated the model" — it validates the *system*, not the *causal user-facing effect*. Only champion-challenger/canary designs (with real traffic exposure) can measure actual business impact, because shadow predictions have no behavioral consequence to observe.

---

### Q: Design a metrics hierarchy for an experimentation program, and explain why you don't just optimize the north star metric directly in every experiment. [ML Systems]

A typical hierarchy:
1. **North star metric**: captures long-term value (e.g., 90-day retention, LTV) — the thing you ultimately care about, but slow-moving, noisy at the individual-experiment timescale, and often insensitive to any single feature change.
2. **Driver metrics**: move faster and are believed (ideally validated) to drive the north star — e.g., 7-day retention, session length, CTR.
3. **Guardrail metrics**: must not regress — latency, error rate, support ticket volume, unsubscribe rate.
4. **Diagnostic metrics**: help explain *why* a driver metric moved (e.g., funnel step conversion rates, feature adoption rate).

You don't optimize the north star directly in every experiment because most individual experiments are too small/short to move a slow, high-variance, long-horizon metric with statistical power — you'd need enormous sample sizes and long durations to detect anything. Instead, you validate (periodically, via holdout experiments or long-run analysis) that driver metrics genuinely causally predict the north star, then let individual experiments optimize driver metrics quickly, trusting that in aggregate this steers the north star.

**Gotcha**: a system that only ever optimizes driver metrics without periodically re-validating the driver → north-star causal link risks "metric gaming" — teams can win on a driver metric (e.g., increase notification-driven session count) while quietly harming the true long-term objective (e.g., annoying users into churn) if the driver-to-north-star relationship breaks down or was never causal to begin with (Goodhart's Law in an experimentation context).

---

### Q: You see a statistically significant lift in your primary metric, but a related core business metric didn't move (or moved the wrong way). What do you check? [Pitfalls]

A structured checklist:
1. **Sample Ratio Mismatch**: run the chi-squared SRM check first — if allocation is off, nothing else in the analysis is trustworthy.
2. **Metric definition mismatch**: confirm the primary metric and the business metric are actually measuring related things at the same granularity/timeframe (e.g., primary metric is "add to cart," business metric is "revenue" — an add-to-cart lift doesn't guarantee a checkout completion or revenue lift if the traffic added is lower-intent).
3. **Segment/Simpson's Paradox check**: segment both metrics by platform, geography, user tenure — does the aggregate story hold within each segment, or does it reverse?
4. **Statistical power on the business metric**: business metrics (e.g., revenue) are often far noisier (heavy-tailed) than the primary metric (e.g., a binary click), so the test may simply be underpowered to detect a real revenue effect at the sample size run — check the achievable MDE for the noisier metric.
5. **Time-window/decay effects**: check whether the primary metric's lift is a novelty effect that hasn't yet propagated to (or will never propagate to) the downstream business metric.
6. **Guardrail cost**: check whether some guardrail metric moved negatively (e.g., latency increased, or a different product line's conversion dropped due to cannibalization) and is offsetting the gain elsewhere.
7. **Causal chain validity**: revisit whether the primary metric was ever causally linked to the business metric, or whether that link was assumed rather than validated.

**Gotcha**: the instinctive wrong move is to keep "metric-mining" until you find something significant that explains the story you want — pre-register which of these checks you'll run, and be honest that "the primary metric moved but the business metric didn't" might simply mean the primary metric was a poor proxy, not that there's a hidden explanation to be found by digging until something turns up significant.

---

### Q: An experiment on a new checkout flow shows a large lift within the first 3 days, but by day 14 the lift has shrunk to near zero. What's going on and what would you do? [Pitfalls]

Most likely a **novelty effect**: the new flow is attracting attention/curiosity-driven engagement that fades as users habituate, or the initial cohort exposed happens to be an atypical early-adopter population. Alternatively this could be a **change aversion** effect resolving in the opposite direction if the *baseline* effect was initially negative and grew positive — but shrinking-to-zero from an initial positive lift is the classic novelty signature.

What to do: (1) plot the day-by-day (or cohort-by-cohort, isolating users newly exposed each day) treatment effect rather than relying on the cumulative average, which can hide this decay; (2) extend the test duration to see if the effect stabilizes at a new steady-state (possibly still positive, just smaller than the initial spike, or possibly genuinely zero); (3) check whether the decaying pattern is concentrated in returning users re-encountering the new flow (novelty wearing off) vs. consistent across newly arriving users (which would suggest the "true" steady-state effect is actually near zero, not a novelty artifact) — segmenting by tenure is the key diagnostic that distinguishes true novelty decay from "the initial 3 days just had an unrepresentative population."

**Gotcha**: many teams stop the experiment right at the initial lift and ship, which is functionally the same mistake as peeking — the correct process bakes a pre-committed duration long enough to observe stabilization into the experiment design up front.

---

### Q: What is a causal DAG, and what are the three canonical structural motifs (chain, fork, collider)? [Causal Inference]

A Directed Acyclic Graph represents assumed causal structure: nodes are variables, directed edges are causal effects, and it tells you exactly which variables you must, may, or must not control for.

- **Chain (mediation)**: `A → B → C`. B is a mediator on the causal pathway from A to C. Conditioning on B blocks that path, so you'd only be estimating the *direct* effect of A on C (not through B) — if you wanted the *total* effect, you should not control for a mediator.
- **Fork (confounder)**: `A ← C → B`. C is a common cause of both A and B, creating a spurious (non-causal) association between them. You *must* condition on C to isolate the true causal relationship (if any) between A and B.
- **Collider**: `A → C ← B`. A and B are both causes of C, but there's no causal link between A and B. By default they're marginally independent (no spurious correlation). **Conditioning on the collider C induces a spurious association between A and B** — this is collider bias, and it's often the opposite move of what a naive analyst would do.

**Gotcha**: the most common real-world mistake is controlling for what looks like "just another covariate" without checking whether it's actually a collider or a mediator — the fix is the backdoor criterion (a set Z of variables that blocks all backdoor paths from T to Y and contains no descendant of T), which requires actually drawing the DAG rather than throwing every available covariate into a regression.

---

## Hard

### Q: How does Monte Carlo Tree Search (MCTS) integrate with neural networks in systems like AlphaGo? [Reinforcement Learning]
A: Pure MCTS explores a game tree by randomly simulating playouts to the end of the game to estimate the value of a state. In complex games (like Go or real-time strategy), the state space is too vast for random playouts to be useful, and the tree is too deep to search exhaustively. Systems like AlphaGo solve this by replacing the random simulation with two neural networks: a **Policy Network** and a **Value Network**. During MCTS expansion, the Policy Network (acting as a prior) heavily prunes the search space by only suggesting highly probable, strong moves, dramatically reducing the branching factor. During MCTS evaluation, instead of playing the game to the end, the search stops and uses the Value Network to instantly predict the winner from that intermediate state, cutting off the depth of the search. In turn, the improved move probabilities generated by the MCTS search become the training targets to improve the Policy Network in the next iteration—forming a powerful self-play reinforcement learning loop.


### Q: Derive/explain the sample size formula for a two-proportion test and explain what each term controls. [Statistical Concepts]

```
n = [ z_(α/2)·sqrt(2p̄(1-p̄)) + z_β·sqrt(p₀(1-p₀)+p₁(1-p₁)) ]² / (p₁-p₀)²
```

- `z_(α/2)`: how extreme a z-score you require to call something significant (bigger for smaller α → more conservative, needs more data).
- `z_β`: how much power you want (bigger for higher power → needs more data).
- `p̄`: pooled proportion under H₀, used for the standard error of the test statistic.
- `(p₁ - p₀)²` in the denominator: the effect size squared — halving the effect you want to detect quadruples the required sample size.

**Gotcha**: candidates often forget the effect appears squared in the denominator — this is why "just detect any effect, however small" is not a free lunch; MDE and sample size trade off nonlinearly.

---

### Q: Explain confidence intervals for the difference in two proportions, and how you'd construct one. [Statistical Concepts]

For control conversion `p_c` and treatment `p_t`, the (unpooled) standard error of the difference is:

```
SE_diff = sqrt( p_c(1-p_c)/n_c + p_t(1-p_t)/n_t )
```

A 95% CI for the true lift is `(p_t - p_c) ± 1.96 · SE_diff`. Interpretation: if you repeated this experiment many times, 95% of such intervals would contain the true difference. It does **not** mean "there's a 95% chance the true effect is in this specific interval" (that's a Bayesian credible-interval interpretation).

**Gotcha**: note the pooled SE (used under H₀ for the hypothesis test's z-statistic) differs slightly from the unpooled SE (used for the CI around the observed difference) — interviewers sometimes probe whether you know these are two different quantities used for two different purposes.

---

### Q: Your dashboard shows p < 0.05 on day 3 of a planned 14-day test. Should you stop early and ship? [Fundamentals]

No — this is peeking, and stopping the moment you cross a significance threshold inflates the false-positive rate dramatically (simulations show a nominal 5% test can have a 30-40% false-positive rate under continuous monitoring), because each daily check is another independent chance for noise to cross the threshold, and once it crosses you stop — you never let it random-walk back. The p-value process is not monotonic under the null.

**Fixes to cite**: (1) pre-register a fixed sample size/duration and look once at the end; (2) use a sequential testing framework with an alpha-spending function (O'Brien-Fleming or Pocock) that adjusts the significance threshold at each look to preserve overall α; (3) switch to Bayesian testing, where the posterior is a valid probability statement at any sample size and doesn't require a stopping-rule correction.

---

### Q: What is the expected loss framework for Bayesian testing, and why would you use it instead of "probability B is better"? [Bayesian & Bandits]

Instead of stopping once `P(B > A)` crosses some threshold (e.g., 95%), compute the expected loss of choosing the wrong arm:

```
Loss(choose A) = E[max(0, θ_B - θ_A)]
Loss(choose B) = E[max(0, θ_A - θ_B)]
```

You stop when the expected loss of your preferred choice drops below an acceptable threshold (e.g., "I'm willing to lose at most 0.1pp of conversion rate in expectation if I'm wrong"). This is more decision-theoretic than a pure probability threshold because `P(B > A) = 90%` could still carry a huge expected loss if the plausible downside when A is actually better is large — probability of superiority ignores the magnitude of the potential loss.

**Gotcha**: expected loss requires deciding on a business-relevant loss unit (e.g., revenue, not just "conversion rate points") — a common follow-up is "how do you translate expected loss in conversion rate into a dollar figure for decision-making?"

---

### Q: Explain Thompson Sampling and why it naturally balances exploration and exploitation. [Bayesian & Bandits]

Maintain a posterior distribution over each arm's true reward rate (e.g., `Beta(α_i, β_i)` for a Bernoulli/conversion-rate arm). At each round, draw one sample from *each* arm's current posterior and pull the arm with the highest sampled value; then update that arm's posterior with the observed reward.

This balances exploration/exploitation automatically: an arm you're uncertain about has a wide posterior, so its random samples occasionally land very high, giving it a chance to be selected (exploration). As you pull an arm more, its posterior narrows around the true rate, so it only keeps winning if its mean is genuinely competitive (exploitation). No explicit epsilon or schedule is needed.

**Gotcha**: Thompson Sampling is a *stochastic* policy — a common follow-up asks you to contrast it with UCB1, which is *deterministic*: `UCB_i(t) = x̄_i + sqrt(2 ln t / n_i)`, where the bonus term (not the sampling randomness) drives exploration and shrinks as an arm gets pulled more. Both achieve similar asymptotic regret guarantees but behave differently with small samples (Thompson Sampling tends to perform better empirically and handles delayed/batched feedback more naturally).

---

### Q: What is the ignorability (unconfoundedness) assumption, and why is it untestable? [Causal Inference]

Formally: `(Y(0), Y(1)) ⊥ T | X` — conditional on the observed covariates X, treatment assignment is as good as random. Equivalently, you've measured every variable that confounds the T-Y relationship, and once you control for X, there's no remaining backdoor path.

It's untestable from the observed data alone because you never observe both potential outcomes for any unit — you cannot check whether, conditional on X, treated and control groups truly have the same *potential* outcome distributions; you can only check that they look similar on *observed* X. Any unmeasured confounder (e.g., "ambition," which is hard to quantify) can silently violate ignorability, and no amount of data manipulation reveals this — it requires domain knowledge and often sensitivity analysis (e.g., Rosenbaum bounds) to argue how much unmeasured confounding would be needed to overturn the conclusion.

**Gotcha**: this is the central weakness of every observational method (PSM, regression adjustment, etc.) except those that don't rely on ignorability given X directly (DiD leverages a different assumption — parallel trends; IV leverages exogenous variation from an instrument).

---

### Q: What is the parallel trends assumption, why is it untestable, and how do you build confidence in it anyway? [Causal Inference]

Parallel trends: absent treatment, the treated and control groups would have followed the same trajectory over time. It's fundamentally untestable because it's a statement about a counterfactual (what *would have* happened to the treated group without treatment) — you can never observe that world.

You build supporting (not proving) evidence via:
1. **Pre-trend visualization**: plot both groups' outcome over several pre-treatment periods — they should track each other closely before treatment starts.
2. **Placebo/falsification test**: run the same DiD specification using only pre-treatment periods, pretending treatment happened earlier — the estimated "effect" should be statistically indistinguishable from zero.
3. **Event-study specification**: include a full set of period × treatment interaction terms (not just a single post-indicator) and test that all pre-treatment interaction coefficients are jointly zero.

If pre-trends are clearly diverging, DiD as specified is biased, and you should consider a synthetic control method instead (constructing a weighted combination of control units that best reproduces the treated unit's pre-period trajectory).

**Gotcha**: passing a pre-trend test only rules out *detectable, linear* violations in the pre-period — it says nothing about a confound that only starts diverging exactly when treatment begins (a subtle but real critique interviewers may raise).

---

### Q: Explain regression discontinuity design (RDD) and the difference between sharp and fuzzy RD. [Causal Inference]

RDD exploits situations where treatment is assigned based on crossing a threshold ("cutoff") of a running/forcing variable — e.g., a scholarship awarded to students scoring ≥ 70 on an exam. The core intuition: a student scoring 69 and one scoring 71 are nearly identical in every way except which side of the cutoff they fall on, so comparing their later outcomes near the threshold approximates a local randomized experiment.

**Sharp RD**: treatment is a deterministic function of the running variable (score ≥ 70 always gets it). Estimand:
```
τ_RD = lim_{x↓c} E[Y|X=x] - lim_{x↑c} E[Y|X=x]
```
estimated via local linear regression on both sides of the cutoff, typically within a bandwidth window.

**Fuzzy RD**: crossing the threshold changes the *probability* of treatment but doesn't guarantee it (e.g., eligibility increases uptake but some eligible people don't enroll, some ineligible people find a workaround). Here you use the threshold-crossing indicator as an instrument for actual treatment receipt (essentially IV/2SLS localized near the cutoff).

**Gotcha**: RDD estimates a *local* average treatment effect valid only near the cutoff — it says nothing about how the treatment would affect someone far from the threshold (e.g., a student scoring 30). Key diagnostics to name: McCrary density test (checking for suspicious bunching of the running variable just above/below cutoff, which would suggest manipulation/gaming), covariate balance at the cutoff, placebo cutoffs elsewhere in the running variable, and bandwidth sensitivity analysis.

---

### Q: What are the two (or three) conditions for a valid instrumental variable, and why is the exclusion restriction the hardest to defend? [Causal Inference]

A valid instrument Z for treatment T (affecting outcome Y) must satisfy:
1. **Relevance**: Z is correlated with T (`Cov(Z,T) ≠ 0`) — testable via the first-stage regression's F-statistic.
2. **Exclusion restriction**: Z affects Y *only* through its effect on T — no direct path from Z to Y. This is fundamentally untestable because it requires knowing there's no unmodeled pathway, which you can't verify from data alone; it requires a substantive, defensible argument.
3. **Independence**: Z is (as good as) randomly assigned with respect to the confounders of T and Y — i.e., Z is not itself correlated with omitted confounders.

Classic examples: Angrist (1990) used Vietnam draft lottery numbers (truly randomized) as an instrument for military service to study effects on earnings — the lottery number plausibly affects earnings *only* by changing the chance of serving. Card (1995) used proximity to a 4-year college as an instrument for education, arguing growing up near a college affects attendance likelihood but not earnings directly (a more contestable assumption, since proximity could correlate with local labor markets).

**Gotcha**: exclusion restriction violations are the single most common reason IV studies get challenged in peer review/interviews — always be ready with a concrete story for why your instrument couldn't possibly affect the outcome through any channel besides the treatment.

---

### Q: What is 2SLS, and what does the IV estimate actually represent (LATE vs ATE)? [Causal Inference]

Two-Stage Least Squares:
- **Stage 1**: regress treatment T on instrument Z (and any exogenous covariates X): `T̂ = α₀ + α₁Z + α₂X`.
- **Stage 2**: regress outcome Y on the *predicted* treatment from stage 1: `Y = β₀ + β₁T̂ + β₂X`.

`β₁` is the IV estimate. Crucially, it identifies the **Local Average Treatment Effect (LATE)** — the average effect only among "compliers": units whose treatment status is actually changed by the instrument (e.g., people who enroll in college *because* they live near one, but wouldn't otherwise). It says nothing directly about "always-takers" (would attend regardless) or "never-takers" (wouldn't attend regardless).

**Gotcha (weak instruments)**: if the first-stage F-statistic is low (rule of thumb: F < 10), the instrument barely predicts treatment, which inflates the variance of the 2SLS estimate and makes it very sensitive to even small exclusion-restriction violations (weak-instrument bias can be worse than plain OLS bias in the worst case). Always report the first-stage F-statistic. Also flag that **LATE ≠ ATE**: generalizing a LATE finding to the whole population requires an additional (often implausible) homogeneity assumption.

---

### Q: Compare diff-in-differences, propensity score matching, RDD, and IV. When would you pick each? [Causal Inference]

| Method | Key assumption | Best suited for |
|---|---|---|
| PSM / IPW | Ignorability given observed X (no unmeasured confounders) | Rich observed covariates, cross-sectional data, no natural experiment available |
| DiD | Parallel trends (time-invariant unobserved confounders) | Panel/longitudinal data with a clear pre/post and treated/control group (e.g., a policy rolled out in one region) |
| RDD | Continuity of potential outcomes at a threshold | Treatment assigned via an arbitrary, sharp eligibility cutoff (test score, age, income threshold) |
| IV | Relevance + exclusion restriction + independence | A genuine natural experiment/instrument exists that shifts treatment without directly affecting the outcome |

Decision process in an interview: first ask "can I randomize?" → if yes, A/B test. If not, ask "do I have rich pre-treatment covariates and reason to believe I've measured all confounders?" → PSM/IPW. "Do I have panel data with a clear policy change hitting one group but not another, plus pre-period data to check trends?" → DiD. "Is treatment assigned by a hard threshold on some running variable?" → RDD. "Is there some exogenous, as-good-as-random source of variation in treatment that plausibly doesn't affect the outcome directly?" → IV.

**Gotcha**: these aren't mutually exclusive — well-designed causal analyses often triangulate across multiple methods (e.g., PSM + DiD combined, or RDD as a robustness check on an IV result) and check whether the conclusions agree.

---

### Q: Give a concrete worked example of collider bias. [Causal Inference]

Consider hiring: gender (G) and qualification (Q) both independently affect the probability of being hired (H): `G → H ← Q`. Suppose G and Q are truly statistically independent in the underlying population (no real relationship between gender and qualification).

If you condition on H = hired (e.g., by only looking at data from people who got the job, which is exactly what happens if your dataset is "our current employees"), you can induce a spurious *negative* correlation between G and Q within that hired subpopulation — even though no such relationship exists in the general population. Intuitively: among hired people, if someone has a less "typical" gender-associated advantage in the hiring model, they must have been more qualified to make it through, and vice versa — hiring "explains away" the variation, creating an artificial trade-off between G and Q conditional on H=1.

This is empirically demonstrable: simulate G and Q as independent, generate hiring probability as a function of both, then check correlation between G and Q (a) marginally (near zero) and (b) restricted to the hired subsample (nonzero, often sizable). This is also known as "explaining away" or Berkson's paradox.

**Gotcha**: this exact mechanism is why naive "controlling for everything available" in observational hiring/promotion bias studies can produce misleading or even sign-reversed conclusions if the sample is implicitly conditioned on a downstream collider like "currently employed" or "was promoted."

---

### Q: What is `do`-calculus, and how does `P(Y | do(T=t))` differ from `P(Y | T=t)`? [Causal Inference]

`P(Y | T=t)` is the *observational* conditional distribution: you passively select the subpopulation where T happened to equal t and look at their Y — but that subpopulation may differ systematically from the full population precisely because of the confounders that led them to have T=t. Confounding pathways remain fully active.

`P(Y | do(T=t))` is the *interventional* distribution: imagine reaching into the system and forcely setting every unit's T to t, severing all the causal arrows that normally point *into* T (i.e., removing the influence of whatever normally determines T, including confounders), then observing Y. This is what a randomized experiment actually estimates.

If the backdoor criterion is satisfied by covariate set Z, do-calculus tells you the interventional distribution is recoverable from purely observational data via the **backdoor adjustment formula**:
```
P(Y | do(T=t)) = Σ_z P(Y | T=t, Z=z) P(Z=z)
```
i.e., stratify by Z, compute the conditional outcome distribution within each stratum, then average across the marginal distribution of Z — this is exactly the intuition behind standardization/g-computation and connects directly back to why propensity/matching methods work.

**Gotcha**: this distinction — "observing" vs. "intervening" — is the single most important idea to be able to explain crisply in a causal inference interview; many candidates can recite the notation without being able to explain *why* they differ (confounding pathways active vs. severed).

---

### Q: How do you evaluate an uplift model if you never observe the ground-truth individual treatment effect? [Uplift Modeling]

You can never observe both potential outcomes for the same unit, so standard supervised-learning metrics (MSE, accuracy against ground truth) don't apply directly to CATE predictions. Instead, use a **held-out randomized dataset** (crucial: the evaluation set itself must have random treatment assignment) and a **Qini curve** (or its AUUC — Area Under the Uplift Curve): sort units by predicted uplift score descending, and at each cumulative fraction targeted, compute the *actual* incremental outcome achieved by targeting that group (using the randomization to get an unbiased estimate of incremental lift per targeted fraction, since you compare the treated vs. control rate strictly within that top-k subset). A better uplift model produces a Qini curve that rises faster and has larger area than random targeting.

**Gotcha**: this evaluation is only valid if the treatment assignment in the evaluation set was randomized — if you evaluate an uplift model against propensity-matched or otherwise observational treatment assignment, the "incremental outcome per group" computation is itself confounded, defeating the purpose. Standard k-fold cross-validation with non-random treatment does not substitute for this.

---

### Q: What makes causal forests different from a standard random forest, and why does "honesty" matter? [Uplift Modeling]

Causal forests (Wager & Athey, 2018) adapt the random forest algorithm to CATE estimation. Instead of choosing splits to maximize prediction accuracy of Y, they choose splits to maximize *heterogeneity in the estimated treatment effect* between the resulting child nodes — i.e., the tree is built to separate units with different CATEs from each other, not to best predict the outcome itself.

**Honesty**: causal forests use disjoint subsamples for (a) deciding the tree structure/splits and (b) estimating the treatment effect within each resulting leaf. This separation is what allows the method to produce valid confidence intervals for CATE — if the same data were used for both, the effect estimates within adaptively-chosen leaves would be biased (a form of overfitting/selection bias, since the leaf boundaries were chosen partly based on where effects looked extreme in that very data).

**Gotcha**: candidates should know the practical libraries (`econml`, `causalml`) and be able to state that causal forests still rely on unconfoundedness/ignorability given the covariates used to build the forest — they solve the *estimation/heterogeneity* problem, not the *identification* problem.

---

### Q: What is interleaving, and why is it far more statistically sensitive than a standard A/B test for ranking quality? [ML Systems]

Interleaving merges results from ranking model A and ranking model B into a single combined ranking shown to the *same* user in a single session (using a method like team-draft interleaving, which alternates picks from each ranker while avoiding duplicates), then tracks which model's items receive engagement (clicks).

It's far more sensitive because each user acts as their own control — you're doing a within-user paired comparison rather than a between-user comparison, which eliminates the enormous between-user variance in engagement propensity that dominates the noise in a standard between-subjects A/B test on ranking quality. This can make interleaving up to roughly 100x more sample-efficient than a standard A/B test for detecting ranking-quality differences, enabling much faster iteration cycles.

**Gotcha**: interleaving measures *relative* engagement preference between two rankers, not an absolute business outcome (e.g., total revenue or session length) — you still need standard A/B tests to validate that a ranking improvement measured via interleaving actually moves the metrics the business ultimately cares about.

---

### Q: You're asked to estimate the effect of a product feature that users opt into voluntarily, using only observational (non-randomized) data. Walk through your approach and its limitations. [Pitfalls]

Approach: (1) draw the causal DAG — identify plausible confounders (Z) that affect both adoption and the outcome (e.g., user tenure, engagement level, device type); (2) check whether you have panel/pre-period data — if yes, DiD is attractive since it also absorbs time-invariant unobserved confounders, contingent on parallel trends holding (test with pre-trend plots); (3) if only cross-sectional data is available, estimate propensity scores for adoption given observed covariates, use matching or IPW (ideally a doubly robust estimator) to estimate ATT, and rigorously check covariate balance post-matching; (4) look for a plausible natural experiment or instrument (e.g., a feature rollout that was staggered for unrelated infrastructure reasons, creating exogenous variation in *when* users could adopt) that could support an IV or RDD approach as a robustness check; (5) run a sensitivity analysis (e.g., Rosenbaum bounds or an E-value) to quantify how strong an unmeasured confounder would need to be to overturn your conclusion.

**Core limitation to state explicitly**: since adoption is voluntary, users who opt in are almost certainly systematically different from those who don't (self-selection) on both observed and *unobserved* dimensions (motivation, need for the feature, sophistication) — every observational method here rests on the ignorability assumption, which is fundamentally untestable. The single most convincing thing you could do, if at all possible going forward, is retroactively push for a randomized rollout (e.g., a holdout group denied early access) for the *next* version of this question.

**Gotcha**: interviewers are listening for whether you (a) volunteer that observational causal claims are inherently more fragile than experimental ones, and (b) don't overstate confidence in a point estimate without discussing what would invalidate it.

---

### Q: Why can conditioning on a "control variable" sometimes make your causal estimate worse instead of better? Give two distinct mechanisms. [Pitfalls]

Two distinct failure modes, both from getting the DAG wrong:

1. **Conditioning on a mediator**: if the variable lies on the causal path from treatment to outcome (`T → M → Y`), controlling for it blocks part of the very effect you're trying to measure, biasing the estimated total effect toward zero (or estimating only a direct-effect component, which may not be what you intended).
2. **Conditioning on a collider**: if the variable is a common effect of two other variables (`A → C ← B`), conditioning on C induces a spurious statistical association between A and B where none causally exists — this can create the *appearance* of a treatment effect (or bias an existing one) purely as an artifact of restricting the sample based on the collider (e.g., only analyzing "currently active users" when activity itself is caused by both the treatment and some other factor).

The general lesson: "control for everything available" is not a safe default in causal analysis (unlike pure predictive modeling, where more features are often fine) — you need the DAG to determine which variables belong in the adjustment set, because some "controls" actively introduce bias rather than removing it.

**Gotcha**: this is a good check on whether a candidate actually understands causal inference vs. just knows regression — ask them to identify, in a DAG you sketch live, which variables are safe to control for and which are mediators/colliders that must be excluded.

---

### Q: What is the offline-online gap, and how would you design a validation process before fully trusting an offline proxy metric? [Pitfalls]

The offline-online gap is the general phenomenon (detailed above) where an offline metric improvement does not guarantee — and can even be negatively correlated with — an online/business metric improvement, due to distribution shift, proxy-label imperfection, feedback loops, position bias, and missing counterfactuals.

To validate an offline proxy before trusting it broadly: run a series of past experiments where you have both the offline metric delta and the ground-truth online metric delta, and check the correlation between them across that historical set (not just anecdotally for one experiment). If the correlation is weak or the sign flips even occasionally, the offline metric isn't a reliable stand-in and every model iteration decision it informs needs an online confirmation step before being trusted for a launch decision. Some teams build a formal "offline-online metric correlation" dashboard that's revisited periodically since the relationship itself can drift as the product and user base evolve.

**Gotcha**: teams often calibrate this correlation once and never revisit it — as the underlying product, user base, or content pool shifts, a previously well-validated offline proxy can silently decouple from the true online objective, so the correlation check needs to be an ongoing process, not a one-time validation.

---

## Back to top

[Emerging Topics README](README.md) · [Experimentation & Causal Inference source material](experimentation-and-causal-inference/01-experimentation-and-causal-inference.md)
