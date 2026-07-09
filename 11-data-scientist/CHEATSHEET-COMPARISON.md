# Data Scientist Cheat Sheet — Comparison Edition

Exhaustive rapid-review reference for every technique/method documented in this folder. Grouped for quick lookup + comparison tables wherever multiple competing options exist.

---

## 1. Causal Inference Methods

### Randomized Controlled Trial (RCT / A-B Test)
- **What it is**: Random assignment to treatment/control eliminates confounding by construction; ATE = E[Y(1)]-E[Y(0)] estimated by simple difference in means.
- **Pros**: Gold standard, no confounding assumption needed, simple analysis, credible to stakeholders.
- **Cons**: Not always feasible/ethical, can be slow, network/interference effects violate SUTVA, novelty effects.
- **When to pick over alternatives**: Whenever randomization is feasible — always preferred over quasi-experimental methods.
- **Key assumption**: SUTVA (no interference, single version of treatment).

### Propensity Score Matching (PSM)
- **What it is**: Match treated and control units on estimated P(treatment | covariates) to construct a comparable control group.
- **Pros**: Intuitive, produces interpretable "matched pairs," reduces dimensionality of matching problem to one score.
- **Cons**: Only balances observed covariates (no protection from unobserved confounding), sensitive to propensity model specification, can discard many unmatched units (loss of efficiency), overlap/common support required.
- **When to pick over alternatives**: Pick PSM over regression adjustment when treatment/control covariate distributions barely overlap and you want to explicitly enforce/inspect common support before estimating effects. Prefer doubly robust over plain PSM when you're unsure the propensity model is correctly specified.
- **Key assumption**: Unconfoundedness/ignorability — no unobserved confounders — plus common support (overlap).

### Inverse Propensity Weighting (IPW)
- **What it is**: Reweight observations by 1/P(treatment) (treated) or 1/(1-P(treatment)) (control) to create a pseudo-population where treatment is independent of covariates.
- **Pros**: Uses all data (no discarding like matching), simple to implement, can estimate ATE or ATT flexibly.
- **Cons**: Extreme propensity scores (near 0 or 1) create huge weights → high variance/instability; fully dependent on correct propensity model.
- **When to pick over alternatives**: Pick IPW over PSM when you want to retain full sample size and effective sample efficiency matters more than exact matching. Avoid when propensity scores cluster near 0/1 (poor overlap) — doubly robust or trimming needed.
- **Key formula**: ATE estimator $\frac{1}{n}\sum \frac{T_iY_i}{\hat e(X_i)} - \frac{(1-T_i)Y_i}{1-\hat e(X_i)}$.

### Doubly Robust Estimation (AIPW / TMLE)
- **What it is**: Combines an outcome regression model and a propensity score model; estimator is consistent if *either* model is correctly specified.
- **Pros**: Protection against misspecification of one of the two models, generally more efficient than IPW alone.
- **Cons**: More complex to implement/explain, still fails if *both* models are misspecified, requires more modeling effort upfront.
- **When to pick over alternatives**: Pick doubly robust over PSM/IPW whenever you can fit both an outcome model and a propensity model — it's the safer default in most modern causal pipelines. Avoid when data is too sparse to fit either model reliably.
- **Key assumption**: Unconfoundedness + correct specification of at least one of {propensity model, outcome model}.

### Difference-in-Differences (DiD)
- **What it is**: Compares the change in outcomes over time between a treated group and a control group to net out common trends.
- **Pros**: Controls for time-invariant confounders and common shocks, works with observational panel/repeated cross-section data, easy to communicate (parallel trends visual).
- **Cons**: Requires parallel trends assumption (untestable directly, only pre-trend proxy), staggered adoption creates bias in classic two-way fixed effects (TWFE) estimators, sensitive to anticipation effects.
- **When to pick over alternatives**: Pick DiD over PSM/IPW when you have pre/post panel data and a plausible untreated comparison group with similar pre-trends. Use Callaway-Sant'Anna or Sun-Abraham estimators instead of plain TWFE when treatment timing is staggered across units.
- **Key assumption**: Parallel trends — absent treatment, treated and control groups would have moved in parallel.

### Regression Discontinuity Design (RDD)
- **What it is**: Exploits a threshold/cutoff rule (e.g., eligibility score ≥ X) where units just above/below are treated as quasi-randomized; compares outcomes near the cutoff.
- **Pros**: Very credible identification near the cutoff (close to local randomization), transparent and visually verifiable (McCrary test for manipulation).
- **Cons**: Only identifies a Local Average Treatment Effect (LATE) at the cutoff — not generalizable to the full population, sensitive to bandwidth choice, requires enough data density near cutoff, can be invalidated by manipulation of the running variable.
- **When to pick over alternatives**: Pick RDD over PSM/DiD when treatment assignment is literally rule-based on a continuous running variable with a hard cutoff (e.g., test score thresholds, age eligibility) — this beats matching methods because assignment is quasi-random by design near the threshold. Avoid when there's evidence of manipulation/sorting around the cutoff (use McCrary density test).
- **Key assumption**: No manipulation of the running variable around the cutoff; continuity of potential outcomes at the threshold. Sharp RDD: deterministic treatment at cutoff; Fuzzy RDD: probabilistic jump, requires IV-style correction.

### Instrumental Variables (IV / 2SLS)
- **What it is**: Uses a variable (instrument) that affects treatment but has no direct effect on the outcome except through treatment, to isolate exogenous variation.
- **Pros**: Can recover causal effects even with unobserved confounding, well-established econometric machinery (2SLS).
- **Cons**: Valid instruments are hard to find, weak instruments cause severely biased/inflated-variance estimates, only identifies LATE for compliers (not ATE), exclusion restriction is untestable.
- **When to pick over alternatives**: Pick IV over PSM/DiD/RDD when you suspect unobserved confounding that no covariate-adjustment method can fix, and you have a credible instrument. Avoid when your instrument is weak (check first-stage F-stat < 10) — this is worse than OLS bias in many cases.
- **Key assumption**: Relevance (instrument correlates with treatment), exclusion restriction (instrument affects outcome only via treatment), independence (instrument as good as randomly assigned).

### Synthetic Control
- **What it is**: Constructs a weighted combination of untreated units ("synthetic" comparison) that best replicates the treated unit's pre-treatment trajectory, then compares post-treatment divergence.
- **Pros**: Useful for single treated unit (e.g., one state/country policy change), transparent construction of the counterfactual, doesn't require parallel trends assumption directly (matches on pre-treatment outcomes/covariates).
- **Cons**: Requires a long pre-treatment period to fit weights well, no formal standard errors (inference via placebo/permutation tests), donor pool quality matters heavily.
- **When to pick over alternatives**: Pick synthetic control over DiD when there's exactly one (or very few) treated unit and many potential control units — DiD's parallel-trends assumption is harder to defend with N=1 treated unit.
- **Key assumption**: The synthetic control (weighted donor pool) accurately reproduces the counterfactual trajectory of the treated unit absent treatment.

### Interrupted Time Series (ITS)
- **What it is**: Models the outcome trend before and after an intervention using time as the running variable, testing for a level/slope change at the intervention point.
- **Pros**: Works with a single time series (no control group needed), directly visualizes the "interruption."
- **Cons**: Confounded by any other event coinciding with the intervention, requires enough pre/post data points, sensitive to model of the trend (linear vs. nonlinear).
- **When to pick over alternatives**: Pick ITS over DiD/synthetic control when no comparison group/donor pool exists at all — it's the fallback for single-series policy evaluation.
- **Key assumption**: No other confounding event coincides with the intervention timing.

### DAGs / Backdoor & Front-door Criteria
- **What it is**: Directed Acyclic Graphs formalize causal assumptions; backdoor criterion identifies which variables to condition on to block confounding paths; front-door criterion allows identification via mediators when backdoor variables are unobserved.
- **Pros**: Makes causal assumptions explicit and falsifiable/debatable, guides correct covariate selection (avoids conditioning on colliders or mediators).
- **Cons**: Requires domain knowledge to draw correctly, wrong DAG → wrong adjustment set, front-door criterion requires strong assumptions about the mediator.
- **When to pick over alternatives**: Use DAGs *before* choosing PSM/IPW/regression adjustment — they tell you which covariates belong in the adjustment set. Not an estimator itself but a design tool.
- **Key concept**: d-separation; conditioning on a collider induces spurious association (collider bias); conditioning on a mediator blocks part of the causal effect you want to measure.

### Mediation Analysis (Baron-Kenny / NDE-NIE)
- **What it is**: Decomposes a total effect into a direct effect (X→Y) and an indirect effect through a mediator (X→M→Y).
- **Pros**: Answers "why/how" a treatment works, not just "whether."
- **Cons**: Requires strong sequential ignorability assumptions (no unmeasured confounding of mediator-outcome relationship), Baron-Kenny's classic causal-steps approach is often underpowered and superseded by NDE/NIE (natural direct/indirect effect) formal counterfactual framework.
- **When to pick over alternatives**: Use when the research question is explicitly about mechanism, after a causal effect is already established by RCT/quasi-experiment.
- **Key assumption**: No unmeasured mediator-outcome confounding (in addition to standard treatment-outcome unconfoundedness).

### Uplift Modeling / Heterogeneous Treatment Effects (HTE)
**T-learner**
- **What it is**: Fit two separate outcome models (one on treated, one on control), take the difference in predictions as the individual treatment effect (CATE).
- **Pros**: Simple, flexible (any base learner), handles strong non-linearity per-arm.
- **Cons**: Each model only sees half the data, can suffer from regularization bias when treatment/control response surfaces are similar.
- **When to pick over alternatives**: Pick T-learner when treatment/control groups are large and their response functions are very different in shape.

**S-learner**
- **What it is**: Fit a single outcome model with treatment indicator as a feature; CATE = prediction with T=1 minus prediction with T=0.
- **Pros**: Simple, uses all data in one model, works fine when treatment effect is small/homogeneous.
- **Cons**: Model can "ignore" the treatment feature (regularization shrinks its effect) — tends to underestimate heterogeneous effects.
- **When to pick over alternatives**: Pick S-learner over T-learner when treatment effect is expected to be small or data is limited (needs the pooling for stability). Avoid when treatment effect is highly heterogeneous — S-learner biases toward zero effect.

**X-learner**
- **What it is**: Two-stage estimator — first estimates outcome models per arm (like T-learner), imputes individual treatment effects, then fits a second-stage model on those imputed effects, weighted by propensity.
- **Pros**: Performs well with imbalanced treatment/control group sizes, more sample-efficient than T-learner in that regime.
- **Cons**: More complex pipeline (more models to fit and can go wrong), needs propensity estimates too.
- **When to pick over alternatives**: Pick X-learner over T-learner when treatment group is much smaller than control (or vice versa) — this is its specific design advantage.

**Causal Forest**
- **What it is**: Tree-ensemble (generalized random forest) that directly estimates CATE by recursively partitioning covariate space to maximize treatment effect heterogeneity.
- **Pros**: Nonparametric, provides honest confidence intervals for CATE, handles high-dimensional covariates well.
- **Cons**: Computationally heavier, less interpretable than meta-learners, still needs unconfoundedness to hold.
- **When to pick over alternatives**: Pick causal forest over meta-learners (S/T/X) when you need valid statistical inference (CIs) on heterogeneous effects, not just point estimates.

**Evaluation: Qini curve / AUUC**
- **What it is**: Qini curve plots cumulative incremental gain from targeting users ranked by predicted uplift; AUUC (Area Under Uplift Curve) summarizes it into one number.
- **Pros**: Directly measures whether uplift ranking creates business value (unlike accuracy metrics which don't apply to CATE).
- **Cons**: Requires a held-out RCT/randomized dataset to evaluate honestly; sensitive to sample size in top deciles.
- **When to pick over alternatives**: This is the standard evaluation metric for any uplift model — use in place of AUC/RMSE, which don't measure treatment-effect ranking quality.

### Causal Inference Method Comparison Table

| Method | Best for | Avoid when | Key assumption/tradeoff |
| :--- | :--- | :--- | :--- |
| RCT | Any feasible experiment | Ethical/practical constraints prevent randomization | SUTVA; gold standard |
| PSM | Observational data, want interpretable matched pairs | Poor covariate overlap; unobserved confounders suspected | Unconfoundedness + common support |
| IPW | Want to retain full sample, flexible ATE/ATT | Propensity scores near 0/1 (extreme weights) | Correct propensity model |
| Doubly Robust (AIPW) | Uncertain which model (outcome/propensity) is right | Both models plausibly misspecified | Only one of two models needs to be correct |
| DiD | Panel/repeated cross-section with pre/post + control group | No plausible parallel trends; staggered adoption with TWFE | Parallel trends |
| RDD | Rule-based treatment assignment via cutoff | Manipulation/sorting around cutoff | Continuity at threshold; local effect only |
| IV / 2SLS | Unobserved confounding + credible instrument available | Weak instrument (F<10) or exclusion restriction doubtful | Relevance + exclusion restriction |
| Synthetic Control | Single treated unit, many donor candidates | Short pre-treatment history | Synthetic unit replicates counterfactual |
| Interrupted Time Series | Single series, no control group at all | Other coincident events/confounds | No concurrent confounding events |
| Mediation Analysis | Explaining mechanism after effect established | Unmeasured mediator-outcome confounding | Sequential ignorability |

### Uplift/HTE Model Comparison Table

| Method | Best for | Avoid when | Key tradeoff |
| :--- | :--- | :--- | :--- |
| S-learner | Small data, homogeneous/small effects | Highly heterogeneous treatment effects | Biases effect toward zero |
| T-learner | Large data per arm, very different response shapes | Small/imbalanced arms | Two independently-fit models, no data sharing |
| X-learner | Imbalanced treatment/control sizes | Simplicity is priority | Extra propensity-weighted second stage |
| Causal Forest | Need honest CIs on heterogeneous effects | Need fast/simple/interpretable model | Nonparametric, computationally heavier |
| Qini/AUUC | Evaluating any uplift model | No randomized holdout available | Needs RCT-labeled eval set |

---

## 2. Experimentation Design & A/B Testing

### Standard A/B Test (Fixed-Horizon)
- **What it is**: Randomize users into two (or more) arms, run for a pre-computed sample size, analyze once at the end with a fixed-horizon hypothesis test.
- **Pros**: Simple, well-understood inference (valid p-values/CIs), easy to power analytically.
- **Cons**: Cannot peek early without inflating false positive rate, slow if effect is large and obvious early, wastes exposure to an inferior variant for the full duration.
- **When to pick over alternatives**: Default choice when you need clean statistical inference and can tolerate waiting for the full pre-computed sample size. Prefer over bandits when the primary goal is *learning* (accurate effect estimate) rather than *earning* (maximizing reward during the test).
- **Key formula**: Sample size via power analysis, e.g. $n = \frac{2(z_{\alpha/2}+z_\beta)^2\sigma^2}{\delta^2}$ (continuous), or Cohen's h effect size for proportions.

### Multi-Armed Bandits (ε-greedy, UCB, Thompson Sampling)
- **What it is**: Adaptive allocation algorithms that shift traffic toward better-performing arms in real time, balancing exploration and exploitation.
- **Pros**: Minimizes regret (opportunity cost) during the test itself, good when the cost of exposing users to a bad variant is high, works well with many arms.
- **Cons**: Harder to get clean, unbiased causal estimates post-hoc (adaptive allocation biases naive analysis), less standard for reporting statistically "significant" results to stakeholders, non-stationarity (time-varying effects) complicates convergence.
- **When to pick over alternatives**: Pick bandits over fixed-horizon A/B when the goal is maximizing cumulative reward during the experiment (e.g., ad creative selection, pricing) rather than precise effect-size estimation. Avoid when you need a defensible p-value/CI for a launch decision — use a fixed-horizon test or always-valid sequential method instead.
- **Key tradeoff**: Explore/exploit tradeoff — UCB and Thompson Sampling both adaptively reduce exploration as confidence grows; Thompson Sampling uses posterior sampling (Bayesian), UCB uses optimism-under-uncertainty bounds.

### Switchback Experiments
- **What it is**: Randomize the *treatment* over time at the level of a whole unit (e.g., a city, marketplace) — switching the entire unit between treatment and control in alternating time windows — instead of randomizing individual users.
- **Pros**: Handles interference/network effects that violate SUTVA in marketplace settings (e.g., pricing, dispatch algorithms where one user's treatment affects others in the same market).
- **Cons**: Much lower effective sample size (few independent time-blocks, not thousands of users), carryover effects between switches bias estimates, needs careful choice of switch duration.
- **When to pick over alternatives**: Pick switchback over standard A/B when treatment has spillover effects within a shared marketplace/network (e.g., surge pricing, driver dispatch) — user-level randomization would violate SUTVA. Avoid when carryover effects are large relative to switch window.
- **Key assumption**: Limited carryover between adjacent time blocks; sufficiently long switch windows to reach steady state.

### Holdout Groups
- **What it is**: A long-running group of users permanently excluded from a feature/campaign to measure its true long-term incremental impact.
- **Pros**: Measures cumulative/long-run effects that short experiments miss (e.g., long-term retention impact of a feature), useful for measuring marketing incrementality.
- **Cons**: Costly to maintain (foregone value to the held-out users for a long period), organizational pressure to "let them in."
- **When to pick over alternatives**: Pick holdouts over standard A/B tests when measuring long-term/cumulative effects (e.g., "what if we turned off all marketing for this group for 6 months") rather than a single feature's short-term lift.

### Bayesian A/B Testing (Beta-Binomial, Posterior Probability, Expected Loss)
- **What it is**: Models conversion rate with a Beta prior updated to a Beta posterior given observed successes/failures; decisions made via posterior probability that B > A or expected loss from choosing wrong.
- **Pros**: Intuitive probabilistic statements ("95% probability B is better"), naturally supports continuous monitoring without the peeking penalty of naive frequentist tests, can incorporate priors from historical data.
- **Cons**: Choice of prior can be scrutinized/gamed, "probability B is better" doesn't tell you the magnitude of practical benefit without expected-loss framing, less standardized across orgs than frequentist p-values.
- **When to pick over alternatives**: Pick Bayesian over frequentist fixed-horizon when stakeholders want intuitive probability statements and you want to monitor continuously without a sequential-testing correction. Use expected loss (not just posterior probability) to decide when to stop, since probability alone ignores magnitude.
- **Key formula**: Conjugate update — Beta(α,β) prior + k successes/n trials → Beta(α+k, β+n-k) posterior.

### Sequential Testing (Alpha-Spending: O'Brien-Fleming, Pocock, mSPRT)
- **What it is**: Statistical frameworks that allow valid repeated peeking at results during an experiment by spending the total error budget (α) across multiple looks according to a pre-specified schedule.
- **Pros**: Lets you stop early for clear wins/losses without inflating Type I error, avoids the "wait for full fixed horizon" cost.
- **Cons**: Requires committing to a peeking schedule/spending function in advance, O'Brien-Fleming is conservative early (hard to stop early) while Pocock spends more error early (easier early stopping but less late-stage power), added statistical/engineering complexity.
- **When to pick over alternatives**: Pick sequential testing over fixed-horizon when stakeholders will inevitably peek at dashboards mid-experiment — better to control for it formally than pretend it isn't happening. Choose O'Brien-Fleming when you want strong protection against early false stops with full power preserved for the final look; choose Pocock when early stopping for large effects matters most. mSPRT (always-valid p-values) suits continuous monitoring without a fixed schedule.
- **Key tradeoff**: Spending function shape trades off early stopping power vs. final-look power.

### Variance Reduction: CUPED / CUPAC / Stratified Sampling
- **What it is**: CUPED (Controlled-experiment Using Pre-Experiment Data) adjusts the outcome using a pre-period covariate correlated with the metric to reduce variance; CUPAC generalizes this with an ML-predicted covariate; stratified sampling assigns treatment within strata to balance known covariates.
- **Pros**: Directly reduces required sample size / increases power for the same traffic, no bias introduced if covariate is truly pre-experiment.
- **Cons**: Requires a good pre-period covariate (weak correlation → little variance reduction), CUPAC needs an extra model trained without leakage from treatment period, stratification requires knowing strata in advance and adds assignment complexity.
- **When to pick over alternatives**: Use CUPED/CUPAC whenever a strong pre-experiment predictor of the outcome exists (e.g., past 30-day spend predicting future spend) — nearly free power gain. Use stratified sampling instead when the key covariate is categorical and you want guaranteed balance rather than just a post-hoc adjustment.
- **Key formula**: $Y_{adj} = Y - \theta(X - \bar X)$ where $\theta = Cov(Y,X)/Var(X)$, X = pre-period covariate.

### Common Experimentation Pitfalls
- **SRM (Sample Ratio Mismatch)**: Observed traffic split deviates from intended allocation (e.g., 48/52 instead of 50/50) — signals a broken randomization/logging pipeline; invalidates the whole experiment until fixed. Detect via chi-squared goodness-of-fit test on arm counts.
- **Novelty/Primacy effects**: Users react to the *change* itself (novelty) or resist the *change* initially (primacy), not the treatment's steady-state effect — bias early results; mitigate by running long enough to reach steady state or by analyzing only new users.
- **p-hacking / HARKing**: Repeatedly testing until significance is found (p-hacking) or hypothesizing after results are known (HARKing) — inflates false positive rate; fix with pre-registration and correction for multiple looks/comparisons.
- **Survivorship bias**: Analyzing only users who remained in the experiment (ignoring those who churned/dropped) biases estimates — always account for attrition in the denominator.
- **Simpson's Paradox**: An effect that appears in aggregate reverses when data is split by a subgroup (e.g., confounded by segment mix shifting between arms) — always check segment-level consistency, especially when randomization unit and analysis unit differ.
- **Leakage/interference**: Users in the same network/household/market affecting each other's outcomes across arms — violates SUTVA; requires cluster-level randomization (e.g., switchback, geo-based).

### Experimentation Design Comparison Table

| Method | Best for | Avoid when | Key tradeoff |
| :--- | :--- | :--- | :--- |
| Fixed-horizon A/B | Clean statistical inference for a launch decision | Need adaptive real-time optimization | Simple but rigid; no early stopping |
| Multi-armed bandit | Maximizing reward during the test itself | Need a clean p-value/CI for stakeholders | Minimizes regret, complicates causal inference |
| Switchback | Marketplace/network interference (SUTVA violated at user level) | Long carryover effects relative to switch window | Lower effective N; unit is time-block not user |
| Holdout group | Long-run/cumulative incrementality measurement | Short-term feature decisions | Costly to maintain, org pressure to end it |
| Bayesian A/B | Intuitive stakeholder communication, continuous monitoring | Need standardized frequentist reporting | Prior choice can be scrutinized |
| Sequential testing | Early stopping while controlling Type I error | No commitment to a peeking schedule | Early vs. late power tradeoff (O'Brien-Fleming vs. Pocock) |
| CUPED/CUPAC | Strong pre-period covariate available | No good pre-period predictor exists | Free variance reduction if covariate is valid |

---

## 3. Statistics & Hypothesis Testing

### Hypothesis Test Selection

**Z-test**
- **What it is**: Tests sample mean vs. hypothesized value when population σ is known or n is large.
- **Pros**: Simple, exact under known σ.
- **Cons**: Requires known σ or large n (≥30) for CLT to apply.
- **When to pick over alternatives**: Pick z-test over t-test only when σ is genuinely known or n is large; otherwise t-test is safer.
- **Key formula**: $z = \frac{\bar x - \mu_0}{\sigma/\sqrt n}$.

**t-test (one-sample, two-sample/Welch's, paired)**
- **What it is**: Tests means when population σ is unknown, using sample s and t-distribution to account for extra estimation uncertainty.
- **Pros**: Valid for small samples, Welch's variant doesn't assume equal variances.
- **Cons**: Assumes approximate normality (or CLT via large n); use Welch's by default (unequal variance assumption is safer than pooled).
- **When to pick over alternatives**: Use paired t-test when the same units appear in both conditions (removes between-subject noise); use Welch's two-sample over pooled two-sample by default; switch to Mann-Whitney U when data is small-n and clearly non-normal.
- **Key formula**: One-sample $t=\frac{\bar x-\mu_0}{s/\sqrt n}$, df=n-1; Welch's uses adjusted df.

**ANOVA (one-way)**
- **What it is**: Tests whether means differ across 3+ groups simultaneously via ratio of between-group to within-group variance.
- **Pros**: Avoids Type I error inflation from running many pairwise t-tests.
- **Cons**: Only tells you *some* group differs, not which pair — needs post-hoc tests (Tukey HSD/Bonferroni-corrected pairwise); assumes homoscedasticity (else use Welch's ANOVA).
- **When to pick over alternatives**: Pick ANOVA over multiple pairwise t-tests whenever comparing 3+ groups. Pick Kruskal-Wallis instead when data is non-normal/ordinal.
- **Key formula**: $F = \frac{MS_B}{MS_W}$.

**Chi-Squared Test (independence / goodness-of-fit)**
- **What it is**: Tests association between categorical variables (independence) or fit to a theoretical distribution (goodness-of-fit) using squared deviations of observed vs. expected counts.
- **Pros**: Works directly on counts/frequencies, no distributional assumption on the categories themselves.
- **Cons**: Requires expected cell counts ≥5 or the approximation breaks down; use Fisher's exact test for sparse tables.
- **When to pick over alternatives**: Use for categorical association questions (e.g., "does conversion differ by segment") where t-tests/ANOVA don't apply. Switch to Fisher's exact test when cell counts are small.
- **Key formula**: $\chi^2=\sum(O-E)^2/E$; df=(r-1)(c-1) for independence, k-1 for goodness-of-fit.

**Non-parametric tests (Mann-Whitney U, Kruskal-Wallis)**
- **What it is**: Rank-based tests of distributional difference that make no normality assumption — Mann-Whitney for 2 groups, Kruskal-Wallis for 3+.
- **Pros**: Robust to non-normal/skewed/ordinal data and outliers.
- **Cons**: Less powerful than t-test/ANOVA when normality actually holds; tests distributional dominance/rank shift, not means directly; Kruskal-Wallis also needs post-hoc correction for pairwise comparisons.
- **When to pick over alternatives**: Pick over t-test/ANOVA when sample is small and clearly non-normal (skewed, heavy-tailed, ordinal).

#### Hypothesis Test Comparison Table

| Test | Best for | Avoid when | Key assumption |
| :--- | :--- | :--- | :--- |
| Z-test | Known σ or large n, mean vs. fixed value | σ unknown & small n | Normal/CLT, known σ |
| t-test (Welch's) | Comparing 2 group means, unknown σ | Severely non-normal + small n | Approx. normality |
| Paired t-test | Same units measured twice (before/after) | Independent samples | Differences approx. normal |
| ANOVA | 3+ group means simultaneously | Only 2 groups (use t-test) or non-normal | Homoscedasticity, normal residuals |
| Chi-squared | Categorical association/goodness-of-fit | Expected cell counts < 5 | Independence, adequate cell counts |
| Mann-Whitney U / Kruskal-Wallis | Non-normal/ordinal/outlier-heavy data | Assumptions of parametric test actually hold (loses power) | Rank-based, no normality needed |

### Multiple Testing Correction

**Bonferroni**
- **What it is**: Divide α by number of tests m to control family-wise error rate (FWER).
- **Pros**: Simple, guarantees FWER control via union bound.
- **Cons**: Very conservative, especially under positively correlated tests — throws away power.
- **When to pick over alternatives**: Use when you need strict FWER control (any single false positive is costly) and tests are roughly independent. Prefer BH/FDR when testing many hypotheses (e.g., hundreds of metrics) where some false discoveries are tolerable.
- **Key formula**: $\alpha_{adj} = \alpha/m$.

**Benjamini-Hochberg (BH / FDR)**
- **What it is**: Controls the expected proportion of false discoveries among rejected hypotheses (FDR) rather than any-false-positive (FWER); step-up procedure on sorted p-values.
- **Pros**: Much more power than Bonferroni, appropriate when many discoveries are expected and being made.
- **Cons**: Individual "discoveries" may still be false — doesn't control per-comparison error; needs independence or positive dependence (else use BHY).
- **When to pick over alternatives**: Pick BH over Bonferroni when testing many hypotheses (e.g., many experiment metrics/guardrails, many genes) where you can tolerate some false discoveries as long as the overall rate is controlled.
- **Key formula**: Reject $H_{(1)},...,H_{(i)}$ where $i$ = largest index with $p_{(i)} \le \frac{i}{m}q^*$.

#### Multiple Testing Comparison Table

| Method | Best for | Avoid when | Key tradeoff |
| :--- | :--- | :--- | :--- |
| Bonferroni | Few tests, need strict FWER control | Many tests / correlated tests (too conservative) | Controls FWER, low power |
| Benjamini-Hochberg | Many tests, tolerate some false discoveries | Need to guarantee zero false positives | Controls FDR, higher power |

### Confidence Intervals

**Frequentist CI**
- **What it is**: Interval such that the *procedure* captures the true parameter in 95% of repeated experiments (not a probability statement about this specific interval).
- **Pros**: Analytically simple, directly dual to hypothesis testing (reject H0 iff μ0 ∉ 95% CI).
- **Cons**: Requires (approx) normality; frequently misinterpreted as "95% probability μ is in this interval" — wrong.
- **When to pick over alternatives**: Use when a closed-form sampling distribution exists (means, proportions under CLT). Use bootstrap instead when no closed form exists (medians, ratios, model coefficients, correlations).
- **Key formula**: $\bar x \pm z_{\alpha/2}\cdot s/\sqrt n$ (or t-critical value for small n).

**Bootstrap CI (percentile, BCa)**
- **What it is**: Resample data with replacement B times, recompute the statistic each time, use the resulting distribution to form a CI.
- **Pros**: Works for any statistic without a closed-form sampling distribution (median, correlation, model coefficients, ratios).
- **Cons**: Computationally expensive at scale, poor with extreme outliers or very small n, percentile method can be biased/skewed — BCa corrects for this and is preferred in practice.
- **When to pick over alternatives**: Pick bootstrap over analytic CI whenever the statistic has no known sampling distribution, or you're unsure normality holds.
- **Key formula**: Percentile method uses $[\hat\theta_{(\alpha/2)}, \hat\theta_{(1-\alpha/2)}]$ from the bootstrap distribution.

**Credible Interval (Bayesian)**
- **What it is**: Directly states P(θ ∈ [a,b] | data) = 0.95 — a genuine probability statement about the parameter, given a posterior distribution.
- **Pros**: Intuitive interpretation, matches how people naturally want to talk about uncertainty.
- **Cons**: Requires specifying a prior (defensible or not), numerically similar to frequentist CI when prior is weak relative to data — the distinction mostly matters with small n/strong prior.
- **When to pick over alternatives**: Pick over frequentist CI when stakeholders want direct probability statements about the parameter itself (e.g., "95% probability conversion lift is between 1% and 3%").

#### Confidence Interval Comparison Table

| Method | Best for | Avoid when | Key distinction |
| :--- | :--- | :--- | :--- |
| Frequentist CI | Standard means/proportions, closed-form exists | No closed-form statistic, or small-n heavy skew | Coverage property of procedure, not a probability of θ |
| Bootstrap (BCa) | Complex statistics (median, ratio, coefficients) | Extreme outliers, tiny n, huge datasets (cost) | Simulates sampling distribution empirically |
| Bayesian credible interval | Direct probability statement about θ desired | Prior hard to defend / must be "objective" | Genuine P(θ∈interval\|data) |

### Correlation Methods

**Pearson r**
- **What it is**: Measures strength of *linear* association between two continuous variables.
- **Pros**: Simple, well understood, directly interpretable [-1,1] scale.
- **Cons**: Misses non-linear relationships entirely (can be ~0 even for a perfect non-linear relationship like y=x²), sensitive to outliers.
- **When to pick over alternatives**: Use when relationship is plausibly linear and data is roughly free of extreme outliers. Move to Spearman/Kendall for monotonic-but-nonlinear or outlier-heavy data.
- **Key formula**: $r=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum(x_i-\bar x)^2\sum(y_i-\bar y)^2}}$.

**Spearman rank correlation**
- **What it is**: Pearson's formula applied to ranks instead of raw values — captures monotonic (not necessarily linear) association, robust to outliers.
- **Pros**: Robust to outliers, captures any monotonic relationship.
- **Cons**: Discards magnitude information — can't distinguish a steep relationship from a shallow one; unstable with many ties/small samples (use Kendall instead in that case).
- **When to pick over alternatives**: Pick over Pearson when relationship is monotonic but non-linear, or outliers are present. Pick Kendall over Spearman when sample is small or has many tied ranks.

**Kendall's tau**
- **What it is**: Measures correlation via the excess of concordant over discordant pairs among all pairwise comparisons.
- **Pros**: More robust/interpretable with small samples and ties than Spearman; direct probability interpretation.
- **Cons**: O(n²) computation (though O(n log n) algorithms exist), still can't characterize functional form.
- **When to pick over alternatives**: Pick over Spearman for small n or heavily tied data.

**Partial correlation**
- **What it is**: Pearson correlation between the residuals of X and Y after each is regressed on a third variable Z — isolates the direct X-Y relationship net of Z's linear influence.
- **Pros**: Removes confounding from a known third variable.
- **Cons**: Only removes *linear* effects of Z — non-linear confounding remains.
- **When to pick over alternatives**: Use when you suspect a specific confounder Z is inflating a raw Pearson correlation between X and Y.

#### Correlation Method Comparison Table

| Method | Best for | Avoid when | Key tradeoff |
| :--- | :--- | :--- | :--- |
| Pearson | Linear relationships, no major outliers | Non-linear/monotonic relationships, outlier-heavy data | Only captures linear association |
| Spearman | Monotonic non-linear relationships, outlier-robust | Many ties, small n | Discards magnitude, keeps rank order |
| Kendall's tau | Small n, many ties | Large n (computational cost) | Concordant vs discordant pair count |
| Partial correlation | Removing a known linear confounder | Confounding is non-linear | Only nets out linear effect of Z |

### Regression Inference

**OLS with LINE assumptions**
- **What it is**: Ordinary least squares regression inference relies on Linearity, Independence, Normality, Equal variance (homoscedasticity) of residuals.
- **Pros**: Well-understood closed-form inference (t-tests, F-test, CIs on coefficients).
- **Cons**: Each assumption has a specific failure mode — non-linearity biases coefficients, dependence (autocorrelation/clustering) shrinks SEs artificially, heteroscedasticity invalidates standard SEs.
- **Key formula/fix**: Use HC (heteroscedasticity-consistent, e.g. HC3) robust SEs when homoscedasticity fails; use clustered SEs when observations are grouped (e.g., users within markets).

**R² vs Adjusted R²**
- **What it is**: R² = fraction of outcome variance explained; Adjusted R² penalizes for the number of predictors.
- **Pros/Cons**: R² never decreases when adding predictors (even noise) — misleading for model comparison; Adjusted R² corrects this.
- **When to pick over alternatives**: Always prefer Adjusted R² over raw R² when comparing models with different numbers of predictors.
- **Key formula**: $\bar R^2 = 1-\frac{(1-R^2)(n-1)}{n-p-1}$.

**F-test vs per-coefficient t-test**
- **What it is**: F-test asks "is the model as a whole better than the mean-only model?"; per-coefficient t-test asks "does this specific predictor matter given the others are already included?"
- **When to pick over alternatives**: Use F-test for the global "is this model useful at all" question; use individual t-tests to decide which specific predictors to keep/drop.

**VIF (multicollinearity diagnostic)**
- **What it is**: Measures how much a predictor's variance is inflated due to correlation with other predictors.
- **Pros**: Flags unstable/uninterpretable coefficients before they mislead inference.
- **Cons**: High VIF hurts interpretability of coefficients but not necessarily prediction accuracy.
- **When to pick over alternatives**: Check VIF whenever doing coefficient-based inference (not needed for pure prediction tasks). VIF > 5 moderate concern, > 10 severe — consider dropping/combining features (PCA) or ridge regression.
- **Key formula**: $VIF_j = 1/(1-R_j^2)$.

### Effect Size Measures

- **Cohen's d**: Standardized mean difference using pooled SD — use for two-group continuous comparisons to report practical significance alongside p-values.
- **Glass's delta**: Like Cohen's d but uses only the control group's SD — pick over Cohen's d when treatment is expected to change variance, not just the mean (avoids conflating variance change into the effect size denominator).
- **Hedges' g**: Cohen's d with a small-sample bias correction — pick over Cohen's d when n is small.
- **Cramér's V**: Effect size for categorical association (chi-squared based) — use instead of Cohen's d when variables are categorical, not continuous.
- **Odds Ratio vs. Relative Risk**: OR compares odds, RR compares probabilities directly — they diverge substantially at high baseline rates; pick RR when communicating to non-technical audiences (more intuitive), OR when doing logistic regression (natural parameterization).
- **NNT (Number Needed to Treat)**: 1/absolute risk reduction — communicates real-world impact ("how many users must see this feature for one extra conversion") better than relative measures alone.

### Distribution Choice Quick Reference

| Distribution | Use when | What breaks |
| :--- | :--- | :--- |
| Normal | Averages of many independent effects (CLT limit) | Heavy real-world tails underestimated |
| Binomial | Fixed n trials, independent binary outcomes | Independence violated; large n small p → use Poisson |
| Poisson | Rare independent events in fixed window/rate known | Overdispersion (variance > mean) → use Negative Binomial |
| Beta | Modeling a probability itself (e.g., CTR), Bayesian conjugate prior for Bernoulli/Binomial | Only valid on [0,1] |
| Gamma | Waiting times / positive continuous quantities, prior for rate params | Assumes constant-rate arrivals (stationarity) |
| t-distribution | Mean inference with unknown σ, small n | Still needs approx. normal underlying data |
| Chi-squared | Sum of squared normals; goodness-of-fit/independence tests | Needs expected cell counts ≥5 |
| F-distribution | Ratio of variances (ANOVA, nested model comparison) | Assumes normal residuals + homoscedasticity |

### Bayesian vs Frequentist Estimation

| Concept | Frequentist | Bayesian |
| :--- | :--- | :--- |
| Parameter view | Fixed, unknown | Random variable with a distribution |
| Point estimate | MLE (maximize likelihood only) | MAP (maximize posterior; ≈ MLE + regularization) or posterior mean |
| Interval | Confidence interval (property of procedure) | Credible interval (direct probability statement) |
| Regularization link | N/A | Gaussian prior ↔ L2, Laplace prior ↔ L1 |
| Best for | Standardized, large-sample inference | Small n with genuine prior knowledge, intuitive probability statements |

---

## 4. EDA & Data Quality

### Outlier Detection

**IQR Fences (Tukey's method)**
- **What it is**: Flags values outside [Q1-1.5·IQR, Q3+1.5·IQR] as outliers.
- **Pros**: Robust (based on quartiles, not mean/SD), simple, standard for box plots.
- **Cons**: Fixed 1.5 multiplier is a convention, not universal; can flag too many points on skewed distributions.
- **When to pick over alternatives**: Default choice for quick univariate outlier screening, especially with skewed data (more robust than Z-score there).

**Z-score vs Modified Z-score (MAD-based)**
- **What it is**: Z-score flags |x-mean|/SD > threshold (commonly 3); Modified Z-score uses median and MAD instead of mean/SD: $M_i = 0.6745(x_i-\tilde x)/MAD$, flag |M_i| > 3.5.
- **Pros**: Z-score is simple and standard; Modified Z-score is robust to the very outliers you're trying to detect (mean/SD are themselves outlier-sensitive).
- **Cons**: Standard Z-score is compromised by the outliers it's trying to find (masking effect) — a single huge outlier inflates SD and hides itself and others.
- **When to pick over alternatives**: Pick Modified Z-score (MAD-based) over standard Z-score whenever the outlier fraction might be non-trivial, since standard Z-score's own statistics get contaminated.

**Domain rules**
- **What it is**: Hard-coded valid ranges from business/physical knowledge (e.g., age can't be negative or >150).
- **Pros**: Catches data-entry errors that statistical methods would miss (a "150-year-old" isn't a statistical outlier if many entries are wrong the same way).
- **Cons**: Requires domain expertise, doesn't generalize automatically to new fields.
- **When to pick over alternatives**: Always apply alongside statistical methods — domain rules catch systematic/logical errors that IQR/Z-score can't.

**Winsorizing**
- **What it is**: Caps extreme values at a percentile threshold (e.g., clip below 1st / above 99th percentile) rather than removing them.
- **Pros**: Retains sample size, reduces influence of extreme values without deleting data.
- **Cons**: Distorts the true distribution shape at the tails; arbitrary choice of cap percentile.
- **When to pick over alternatives**: Use when the outlier has "outsized influence" but isn't a confirmed data error — better than deletion when you want to keep the observation's other information.

#### Outlier Handling Action Table

| Diagnosis | Action |
| :--- | :--- |
| Confirmed data error | Remove |
| Legitimate extreme value | Keep, or log-transform |
| Outsized influence on a model | Winsorize |
| Distinct sub-population | Model separately / add indicator feature |

### Missing Data

**MCAR (Missing Completely at Random)**
- **What it is**: Missingness unrelated to any observed or unobserved variable.
- **Best strategy**: Mean/median imputation or complete-case analysis is unbiased.

**MAR (Missing at Random)**
- **What it is**: Missingness depends on observed variables but not the missing value itself.
- **Best strategy**: MICE (Multiple Imputation by Chained Equations) or KNN imputation — leverages observed covariates to model missingness.

**MNAR (Missing Not at Random)**
- **What it is**: Missingness depends on the unobserved value itself (e.g., high earners refuse to report income).
- **Best strategy**: Sensitivity analysis, explicit selection models, domain-informed fill values, or an `is_missing` indicator flag — no imputation method is unbiased without modeling the missingness mechanism itself.
- **When to pick over alternatives**: The mechanism (MCAR/MAR/MNAR) — not preference — dictates the correct method; misdiagnosing MNAR as MAR and applying MICE anyway will silently bias results.

#### Missing Data Mechanism Comparison Table

| Mechanism | Best for | Avoid when | Key indicator |
| :--- | :--- | :--- | :--- |
| MCAR | Mean/median imputation, complete-case | Missingness actually correlates with any variable | No pattern in missingness |
| MAR | MICE, KNN imputation | Missingness depends on the missing value itself | Missingness explained by observed covariates |
| MNAR | Sensitivity analysis, selection model, flag+fill | Standard imputation used without acknowledging bias | Missingness depends on unobserved value |

### Distribution Shift / Data Drift Detection

**Kolmogorov-Smirnov (KS) test**
- **What it is**: Tests whether two samples come from the same distribution via the max distance between their empirical CDFs: $D=\sup_x|\hat F_1(x)-\hat F_2(x)|$.
- **Pros**: Non-parametric, sensitive to any distributional difference (location, scale, shape).
- **Cons**: A hypothesis test — p-value shrinks with sample size even for practically trivial shifts at massive scale (statistical vs. practical significance mismatch).
- **When to pick over alternatives**: Use for a formal statistical test of whether train/serving distributions differ, on moderate sample sizes.

**Population Stability Index (PSI)**
- **What it is**: $\sum(Actual\% - Expected\%)\ln(Actual\%/Expected\%)$ across bucketed values; thresholds: <0.1 stable, 0.1-0.25 moderate shift, ≥0.25 major shift.
- **Pros**: Gives a magnitude score (not just a p-value), industry-standard thresholds, doesn't blow up with huge sample sizes the way KS p-values do.
- **Cons**: Requires choosing bucket boundaries, less standard outside credit-risk/ML-monitoring contexts.
- **When to pick over alternatives**: Pick PSI over KS test for production monitoring dashboards where you want an actionable magnitude/threshold rather than a p-value that's hypersensitive to sample size.

**Adversarial validation**
- **What it is**: Train a classifier to distinguish train vs. serving (or train vs. test) rows; high AUC means the two datasets are distinguishable — i.e., meaningful drift/leakage exists.
- **Pros**: Automatically finds *which features* drive the shift (via feature importance) — more diagnostic than a single summary statistic per feature.
- **Cons**: Only tells you *that* distributions differ and roughly *where*, not the practical impact on model performance directly.
- **When to pick over alternatives**: Use adversarial validation when you suspect multivariate/joint distribution shift that univariate KS/PSI checks (one feature at a time) would miss.

#### Distribution Shift Comparison Table

| Method | Best for | Avoid when | Key output |
| :--- | :--- | :--- | :--- |
| KS test | Formal univariate test, moderate n | Very large n (p-value oversensitive) | p-value on CDF distance |
| PSI | Production monitoring with actionable thresholds | Need statistical significance framing | Magnitude score with standard cutoffs |
| Adversarial validation | Detecting multivariate/joint shift, finding which features drive it | Only need a single feature's shift magnitude | Classifier AUC + feature importances |

### Univariate/Bivariate/Multivariate EDA Toolkit

| Technique | Best for | Key note |
| :--- | :--- | :--- |
| Histogram (Sturges'/Freedman-Diaconis bins) | Shape of a single distribution | Freedman-Diaconis better for skewed/outlier-heavy data (uses IQR, robust bin width) |
| KDE (Silverman's/Scott's bandwidth) | Smooth density estimate | Bandwidth choice controls over/under-smoothing |
| Box plot | Compare spread/outliers across groups | Uses IQR fences by convention |
| Q-Q plot | Test normality visually | Deviations from the line show tail/skew departure from Normal |
| ECDF | Distribution comparisons without binning artifacts | No bin-width choice needed, unlike histograms |
| Scatter + LOWESS | Bivariate relationship, non-linear trend | LOWESS reveals non-linearity a Pearson r would miss |
| Hex bin / 2D KDE / alpha transparency | Overplotting with large N | Prevents a solid blob of points from hiding density structure |
| Violin plot | Distribution shape across categories | Combines box plot + density |
| Pair plot / correlation heatmap | Multivariate relationships, many variables at once | Heatmap scales better with many variables than pair plot |
| PCA biplot | Dimensionality reduction + variable relationships together | Loses interpretability of original units |
| Parallel coordinates | Multivariate patterns across many dimensions per row | Gets cluttered with too many rows/lines |

### Data Quality Dimensions
Completeness, Consistency, Timeliness, Accuracy, Uniqueness, Validity — each is a distinct failure mode requiring different checks (e.g., Uniqueness → dedup checks; Timeliness → freshness SLAs; Validity → schema/range constraints).

### Cardinality Handling

| Method | Best for | Avoid when |
| :--- | :--- | :--- |
| Target encoding | High-cardinality categorical, tree/linear models | Small data (leakage/overfitting risk without CV) |
| Embeddings | Very high cardinality, deep learning models | Simple models / small data (overkill) |
| Frequency encoding | Quick baseline, tree models | Category frequency isn't informative of the target |

### Leakage Detection Checklist
- Post-event features (features only available after the outcome occurred).
- Proxy targets (feature that's a deterministic function of the label).
- Time leakage in CV (shuffling time series data, using future info to predict past).
- Target-in-features (accidental inclusion of a transformed target column).

---

## 5. SQL & Data Manipulation

### Window Function Selection

**ROW_NUMBER vs RANK vs DENSE_RANK**
- **What they are**: All number rows within a partition by an ordering; ROW_NUMBER gives unique sequential numbers even for ties; RANK gives ties the same number then skips (1,1,3); DENSE_RANK gives ties the same number without skipping (1,1,2).
- **When to pick over alternatives**: Use ROW_NUMBER for deduplication (exactly one row per group). Use RANK when "top N" should reflect skipped gaps consistent with tie count. Use DENSE_RANK when "top 3" should always mean exactly 3 distinct value-levels regardless of ties.
- **What breaks**: ROW_NUMBER ties break arbitrarily — can't predict which duplicate survives without a deterministic secondary sort key.

**NTILE**
- **What it is**: Divides ordered rows into n equal-sized buckets (e.g., quartiles) dynamically from the data.
- **When to pick over alternatives**: Use for percentile/quartile bucketing when you don't know thresholds in advance and want the engine to infer them.
- **What breaks**: Non-divisible row counts give uneven bucket sizes (extra rows go to earlier buckets).

**LAG/LEAD vs self-join**
- **What it is**: LAG/LEAD reach backward/forward within a partition's order without a join.
- **When to pick over alternatives**: Always prefer LAG/LEAD over a self-join for period-over-period comparisons — cheaper and simpler.
- **What breaks**: First/last row in each partition returns NULL unless a default is supplied.

**FIRST_VALUE/LAST_VALUE**
- **What breaks**: LAST_VALUE's default frame (`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`) returns the *current* row, not the partition's true last row — must explicitly set `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`.

**ROLLUP vs CUBE vs GROUPING SETS**
- **What they are**: All extend GROUP BY to produce subtotal rows. ROLLUP produces a hierarchy of subtotals rolling up one dimension at a time (right to left). CUBE produces every possible subset of grouping columns (2^n combinations). GROUPING SETS lets you explicitly enumerate exactly the combinations you want.
- **When to pick over alternatives**: Use ROLLUP when subtotals follow a natural hierarchy (region → grand total). Use CUBE when you need every cross-tab combination and dimensions are low-cardinality. Use GROUPING SETS when you need arbitrary, non-hierarchical combinations that ROLLUP/CUBE can't express.
- **What breaks**: CUBE explodes combinatorially on high-cardinality dimensions (100^3 rows for 3 dims × 100 values). Both ROLLUP/CUBE produce ambiguous NULLs — use `GROUPING()` to distinguish real NULLs from subtotal NULLs.

#### Window/Aggregation Function Comparison Table

| Function/Clause | Best for | Avoid when | Key gotcha |
| :--- | :--- | :--- | :--- |
| ROW_NUMBER | Dedup, exactly 1 row per group | Need to keep tied rows | Arbitrary tiebreak without secondary sort |
| RANK / DENSE_RANK | Top-N with ties | — | RANK skips numbers after ties, DENSE_RANK doesn't |
| NTILE | Dynamic quantile buckets | Need exact equal bucket sizes | Uneven buckets on non-divisible row counts |
| LAG/LEAD | Period-over-period deltas | — | NULL at partition boundary |
| ROLLUP | Hierarchical subtotals | Non-hierarchical combos needed | Ambiguous NULLs (use GROUPING()) |
| CUBE | All cross-tab subtotal combos | High-cardinality dimensions | Combinatorial explosion |
| GROUPING SETS | Arbitrary specific combos | Simple hierarchy suffices (ROLLUP simpler) | Same NULL ambiguity |

### Join Type Selection

| Join type | Best for | Avoid when | Key gotcha |
| :--- | :--- | :--- | :--- |
| INNER JOIN | Only rows existing on both sides matter | Referential integrity violations would silently drop rows you need | Orphaned rows silently disappear |
| LEFT JOIN | Keep all left-table rows even without a match | — | Filtering right-table column in WHERE (not ON) silently converts to INNER JOIN |
| FULL OUTER JOIN | Reconciliation between two systems | Large distributed engines (forces full shuffle) | Poor performance at scale; consider UNION ALL + grouping instead |
| CROSS JOIN | Building a complete spine (date × segment grid) | Both sides are large tables | Cartesian explosion (1M × 1M = 1 trillion rows) |
| SELF JOIN | One level of hierarchy (employee → manager) | Arbitrary-depth hierarchy | Use recursive CTE instead for multi-level trees |

**NULL-safe join**: Standard `=` never matches NULL to NULL (`UNKNOWN`, not `TRUE`) — use `IS NOT DISTINCT FROM` when NULL-to-NULL matching is genuinely needed; anti-join pattern (`LEFT JOIN ... WHERE right.key IS NULL`) is the standard way to find non-matches.

### CTEs vs Subqueries vs Recursive CTEs
- **Plain CTE**: Best for readability of multi-step queries; in most modern engines (Postgres 12+, Snowflake, BigQuery) inlined by the optimizer — no automatic perf difference vs. subqueries. Force materialization explicitly if a CTE is referenced many times and recomputation is wasteful.
- **Recursive CTE**: Required for arbitrary-depth hierarchy traversal (org charts, bill-of-materials) that a self-join can't express. Needs a depth guard or cycle detection (`CYCLE ... SET ... USING`) to avoid infinite loops on cyclic data.

### Set Operations

| Operation | Best for | Avoid when | Key gotcha |
| :--- | :--- | :--- | :--- |
| UNION ALL | Combining partitions/queries, keep all rows | Need dedup | — |
| UNION | Explicit deduplication needed | Performance matters and dupes are fine | Implicit DISTINCT sort/hash is expensive |
| INTERSECT | Rows present in both queries | Need counts, not just presence | Implicitly deduplicates |
| EXCEPT | Rows in first query but not second (anti-join style) | Need counts | Implicitly deduplicates |

### Performance Techniques
- **Indexes**: B-tree index turns O(n) scan into O(log n) lookup; tradeoff is slower writes + disk space. Composite index column order matters — `(user_id, event_time)` speeds queries filtering on `user_id` alone or both, not `event_time` alone. Wrapping an indexed column in a function (`DATE(event_time)`) defeats the index — rewrite as a range predicate or add a functional index.
- **EXPLAIN / EXPLAIN ANALYZE**: Compare estimated vs. actual row counts — large gaps signal stale statistics (`ANALYZE table` after bulk loads); `Seq Scan` on large filtered tables signals missing index; `Nested Loop` with a large side should likely be a Hash/Merge Join.
- **Partition pruning**: Always filter on the partition column to skip whole partitions.
- **Predicate pushdown**: Filter before joining/aggregating to reduce row counts early.
- **Avoid SELECT \***: In columnar stores, reading only needed columns can cut I/O by 90%+.

### Analytics Query Patterns
- **Funnel analysis**: Conditional aggregation (`MAX(CASE WHEN event=... THEN 1 ELSE 0 END)`) pivots long-format events into per-user step flags; "ever did X" doesn't enforce ordering — use `MIN(event_time)` per step + ordering filter for strict sequential funnels.
- **Cohort retention**: Join each user's cohort week to their subsequent activity weeks; use LEFT JOIN (not JOIN) plus a cross join spine if you need explicit zero-retention rows rather than missing rows.
- **Sessionization**: LAG to compute inter-event gaps → flag session starts where gap exceeds threshold (e.g., 30 min) → running SUM of flags as session ID.
- **Deduplication**: ROW_NUMBER partitioned by logical key, ordered by a "most complete/recent" tiebreak column, filter to rn=1; add a deterministic secondary sort key or reruns produce different results.
- **Rolling distinct actives (7d/28d)**: `COUNT(DISTINCT ...) OVER (...)` is not supported in most engines for a window frame — must join a date spine to a pre-aggregated (user, day) table instead, or use approximate distinct counts (HyperLogLog) at scale.
- **Percentiles (p50/p95/p99)**: `PERCENTILE_CONT` (interpolated) preferred over `PERCENTILE_DISC` (nearest actual value) for latency SLOs; it's an ordered-set aggregate, not a window function — can't compute a rolling percentile directly; use `APPROX_QUANTILES`/`APPROX_PERCENTILE` at billion-row scale.

---

## 6. Business Metrics & Analytics

### Metric Hierarchy: Input vs Output vs Guardrail
- **Input (leading)**: Team-controlled actions (features shipped, onboarding steps) — fast-moving, but moving them doesn't guarantee outputs move.
- **Output (lagging)**: Business outcomes (revenue, retention, MAU) — validates the causal theory but is slow and hard to move directly.
- **Guardrail**: Harm-detection metrics (latency, refund rate, error rate) — a violation vetoes shipping even if outputs improve.
- **When to pick over alternatives**: Track all three simultaneously — input-only teams "stay busy going nowhere," output-only teams "have no levers," and omitting guardrails is how orgs end up optimizing spam notifications to inflate DAU.

### North Star Metric
- **What it is**: One coordinating metric that measures genuine value delivered to users and is a leading indicator of sustainable revenue.
- **Pros**: Coordinates decision-making across teams around one number.
- **Cons**: Can be gamed (DAU → notification spam); must always be paired with guardrails; single-metric optimization has a "shadow" failure mode per product.
- **When to pick over alternatives**: Choose an NSM over tracking revenue directly when revenue could rise for reasons that don't reflect real value delivery (e.g., aggressive monetization tactics).

### AARRR Funnel (Acquisition, Activation, Retention, Referral, Revenue)
- **What it is**: Models the full customer journey to find the stage with the largest binding constraint (biggest absolute drop-off).
- **When to pick over alternatives**: Use to diagnose *where* a revenue problem originates before investing — pouring more into acquisition when the leak is at activation wastes spend. Note: not always linear (B2B referral can precede revenue; viral products can see referral before retention) — map the actual user journey first.

### Retention Curve Shape (flattening vs. reaching zero)
- **What it is**: Percent of a cohort still active at day N, plotted against N.
- **Key insight**: A curve that flattens = product-market fit for a segment (sustainable base); a curve that reaches zero = no sustained value, growth is 100% acquisition-dependent.
- **What breaks**: Redefining "active" too loosely inflates retention artificially; survivorship bias makes old surviving cohorts look great (bad users already churned) — look at curve shape, not late-stage absolute value.

### Stickiness (DAU/MAU, DAU/WAU)
- **What it is**: Fraction of monthly (or weekly) actives who are also active today — measures engagement density, not just scale.
- **When to pick over alternatives**: Use DAU/WAU instead of DAU/MAU for products with a natural weekly (not daily) cadence (B2B tools, weekly fitness apps) — otherwise stickiness looks artificially low. Meaningless for inherently low-frequency products (tax filing, annual renewals) — don't force a daily-habit target onto them.

### LTV Estimation

**Formula-based LTV**
- **What it is**: LTV = ARPU × Gross Margin × (1/monthly churn rate), assuming constant churn (geometric distribution of customer lifetime).
- **Pros**: Simple, requires only current-period metrics, no long history needed.
- **Cons**: Assumes constant churn — real churn is front-loaded (early cohorts churn fast, survivors are stickier) so this underestimates true LTV; uses average ARPU, hiding segment heterogeneity; using revenue instead of gross margin overstates LTV.
- **When to pick over alternatives**: Use when you lack 24+ months of cohort history (early-stage company) and need a directional estimate now.

**Cohort survival curve LTV**
- **What it is**: LTV = Σ(survival rate at month t × ARPU × gross margin) summed over the observed/projected horizon, using actual retention curves instead of assuming constant churn.
- **Pros**: Captures real front-loaded churn shape, more accurate.
- **Cons**: Requires substantial historical data (24+ months ideally), pricing/product changes make older cohorts non-comparable to current ones, projecting the tail forward assumes the curve shape persists (understates LTV if retention is actively improving).
- **When to pick over alternatives**: Pick over formula-based LTV whenever sufficient cohort history exists — it's strictly more accurate when data supports it.

#### LTV Method Comparison Table

| Method | Best for | Avoid when | Key tradeoff |
| :--- | :--- | :--- | :--- |
| Formula-based (ARPU × margin / churn) | Early-stage, no long cohort history | Churn is strongly front-loaded | Assumes constant churn, underestimates true LTV |
| Cohort survival curve | 24+ months of history available | Pricing/product changed recently (cohorts not comparable) | Uses actual curve shape, more accurate but data-hungry |

### Churn Measurement

| Metric | What it answers | Key formula/note |
| :--- | :--- | :--- |
| Logo churn | Are we losing customers (count)? | Customers lost / customers at start |
| Revenue churn | Are we losing revenue (dollars)? | Can diverge sharply from logo churn (losing many small customers ≠ losing revenue if one big customer expands) |
| Gross Revenue Retention (GRR) | Revenue held excluding expansions | (Start MRR − Churned − Contraction) / Start MRR; capped at 100% |
| Net Revenue Retention (NDR/NRR) | Revenue held including expansions | Can exceed 100%; NDR>100% means the business grows with zero new customers |

**What breaks**: comparing monthly vs. annual churn directly (2% monthly compounds to ~22% annual — not the same as "2% churn"); measuring against end-of-period (not start-of-period) customer count flatters high-churn periods.

### Anomaly Detection in Metrics

| Method | Best for | Avoid when | Key tradeoff |
| :--- | :--- | :--- | :--- |
| Simple k-σ threshold on day-of-week baseline | Quick baseline with seasonal adjustment | Slow-developing regressions (single-day threshold misses gradual decline) | k=2 catches ~95% of real anomalies with manageable false-positive rate |
| STL seasonal decomposition | Separating trend/seasonal/residual before flagging | Simple day-of-week patterns alone (STL is overkill) | Anomalies = unusual residuals after decomposition |
| Prophet | Multiple seasonalities + holidays + trend changepoints | Very simple series (overkill) | Good default for business time series |
| CUSUM | Detecting persistent gradual shifts | Need to catch single-day spikes | Built for slow regressions, not spikes |

### Counter-Metrics / Guardrails and Goodhart's Law
- **Core principle**: Every optimized metric has a "shadow" — a way to move the number without creating real value (DAU → notification spam; MRR → aggressive discounting; conversion rate → removing screening friction that also screens bad customers).
- **Mitigation**: Pair every target metric with at least one counter-metric enforced at the experiment-evaluation stage (not an afterthought); rotate metrics periodically; use multiple metrics simultaneously (harder to game 5 at once); do qualitative audits (interviews, session replay) to confirm metric gains reflect real UX improvement.

### OKR vs KPI

| Dimension | KPI | OKR |
| :--- | :--- | :--- |
| Purpose | Monitor ongoing health | Set and track ambitious goals |
| Time horizon | Continuous | Quarterly/annual |
| Achievement | Healthy = within normal range | ~70% achievement is the target (Google standard) |
| Response to miss | Investigate root cause | Reflect on approach |
| Failure mode | Turning a KPI into a target (Goodhart's Law risk) | Setting OKRs too conservatively (becomes a status report) |

**When to pick over alternatives**: Use KPIs to answer "is the business/process operating normally right now" (continuous monitoring); use OKRs to answer "what ambitious thing are we trying to achieve this quarter." Never give a KPI a stretch target — that converts it into an OKR and invites Goodhart's Law gaming.
