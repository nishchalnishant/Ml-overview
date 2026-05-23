---
module: Data Scientist
topic: Causal Inference
subtopic: ""
status: unread
tags: [datascientist, ml, causal-inference]
---
# Causal Inference

---

## 1. The Fundamental Problem: You Can Never Observe a Counterfactual

You want to answer the question: "Did this thing cause that outcome?" You run a drug trial. Patient A takes the drug and recovers. Did the drug cause the recovery? You don't know, because you can never observe what would have happened to Patient A if they had *not* taken the drug. They can't be in both states simultaneously. Every causal question bottoms out at this impossibility.

This is not a data problem — it is a logical one. No amount of data on one person gives you both the treated and untreated version of that same person. Causal inference is the discipline of making principled progress anyway.

---

## 2. Potential Outcomes Framework (Rubin Causal Model)

### Why we need the notation

To reason carefully about causation, you need a way to talk about outcomes that *didn't happen*. Ordinary statistics only models what you observe. Potential outcomes extend this to both worlds: the world where a unit was treated and the world where it wasn't. Without this notation, statements like "the effect of X on Y" are ambiguous about which comparison is being made and for whom.

**Notation**:
- $Y_i(1)$: what outcome unit $i$ *would have* if treated
- $Y_i(0)$: what outcome unit $i$ *would have* if not treated
- $D_i \in \{0,1\}$: observed treatment indicator
- $Y_i = D_i Y_i(1) + (1 - D_i) Y_i(0)$: you only ever see one of the two

The causal effect for unit $i$ is $Y_i(1) - Y_i(0)$. This is unobservable at the individual level. Period. Every method in causal inference is a strategy for estimating it in aggregate.

---

### Why we need ATE

You want a population-level answer to "does this treatment work?" Not for one person but across everyone. You want to compare, in expectation, the world where everyone gets the treatment versus nobody does. This is the Average Treatment Effect.

**The formula**: $\tau_{ATE} = E[Y(1) - Y(0)]$

Its simplicity hides the hard part: $Y(1)$ and $Y(0)$ cannot both be observed for any single unit. The expectation is over the population, but each person contributes only one observation.

**What breaks**: If you naively compute $E[Y | D=1] - E[Y | D=0]$ from data, you get:

$$E[Y|D=1] - E[Y|D=0] = \underbrace{E[Y(1)|D=1] - E[Y(0)|D=1]}_{\text{ATT}} + \underbrace{E[Y(0)|D=1] - E[Y(0)|D=0]}_{\text{selection bias}}$$

The second term is non-zero whenever treated and untreated units differ in their baseline potential outcomes — i.e., almost always in observational data.

---

### Why we need ATT and ATU separately

Sometimes you don't want the average effect for everyone. A job training program might only be deployed to unemployed workers — you care whether *those people* benefit, not whether Harvard graduates would benefit. ATT answers "did the treatment work for the people who actually got it?"

**ATT**: $\tau_{ATT} = E[Y(1) - Y(0) \mid D=1]$

**ATU**: $\tau_{ATU} = E[Y(1) - Y(0) \mid D=0]$ — useful when you're considering expanding a program to untreated units.

**The relationship**: $ATE = P(D=1) \cdot ATT + P(D=0) \cdot ATU$. ATE equals ATT only when treatment assignment is independent of potential outcomes — when people don't self-select based on expected benefit.

---

## 3. Randomized Controlled Trials: The Benchmark

### Why randomization solves the selection bias problem

The fundamental obstacle is that treated and untreated groups may differ in their baseline potential outcomes $Y(0)$. Randomization eliminates this by construction: when treatment is assigned by coin flip, it cannot be correlated with anything about the unit, including $Y(0)$.

Formally, randomization gives you:

$$D_i \perp (Y_i(0), Y_i(1))$$

Treatment is independent of potential outcomes. This means $E[Y(0)|D=1] = E[Y(0)|D=0]$, so selection bias vanishes. The naive difference in means is now an unbiased estimator of ATE:

$$\hat{\tau} = \bar{Y}_{D=1} - \bar{Y}_{D=0}$$

No adjustment needed. No model needed. This is why RCTs are called the gold standard.

---

### What breaks in RCTs

**Non-compliance**: Units assigned to treatment don't take it, or units assigned to control somehow get it. Your randomized assignment $Z$ no longer equals treatment received $D$. The ITT (Intent-to-Treat) estimate $E[Y|Z=1] - E[Y|Z=0]$ is unbiased for the effect of assignment, not treatment. To recover the effect of actual treatment, you use IV (see Section 7).

**Attrition bias**: Units drop out of the study non-randomly. If sicker patients drop out of the treatment arm, your observed outcomes are not representative. The randomization is broken for the observed sample.

**Spillovers (SUTVA violation)**: The potential outcomes framework assumes $Y_i(D_i)$ depends only on unit $i$'s treatment, not others'. If treating one person affects neighbors — vaccination preventing transmission, price changes affecting untreated competitors — the framework breaks. SUTVA (Stable Unit Treatment Value Assumption) is the assumption that it doesn't.

**External validity**: An RCT in one population doesn't tell you the effect in a different population. The ATE you estimated is for the people in your study. Generalizing requires additional assumptions about similarity.

---

## 4. Observational Studies: Three Distinct Threats

### Confounding

You want to estimate the effect of treatment $D$ on outcome $Y$, but there's a variable $Z$ that influences both who gets treated and what outcome they'd have. $Z$ is a confounder.

Example: sicker patients receive more intensive treatment. If you compare intensive-treatment patients to routine-care patients, intensive treatment looks harmful — because patients receiving it were already sicker. $Z$ = illness severity causes both $D$ = intensive treatment and $Y$ = worse outcomes.

Confounding is not a statistical artifact. It is a structural feature of how the data was generated. No amount of larger samples fixes it. You need to either measure and condition on $Z$, or use a design that breaks the $D \leftarrow Z$ link.

---

### Selection bias

Selection bias is a specific form of confounding where the mechanism is *who ends up in your sample or treatment group*. Units self-select into treatment based on characteristics that predict their potential outcomes.

Students who enroll in tutoring programs tend to be more motivated — they'd do better even without tutoring. Patients who seek a procedure tend to be healthier than average — they recover better for reasons unrelated to the procedure.

The decomposition from Section 2 shows the problem precisely: $E[Y|D=1] - E[Y|D=0]$ conflates the treatment effect with baseline differences in $Y(0)$.

**What breaks**: The naive comparison systematically over- or underestimates the true causal effect depending on the direction of selection.

---

### Omitted Variable Bias (OVB)

You run a regression of $Y$ on $D$ but leave out a confounder $Z$ that belongs in the model. What happens to your estimate of $\beta_D$?

**Why the formula has this shape**: when you omit $Z$, your estimate of $\beta_D$ absorbs whatever part of $Z$ is correlated with $D$. The amount it gets absorbed is exactly $\beta_Z \cdot \delta_{ZD}$, where $\delta_{ZD}$ is the coefficient from a regression of $Z$ on $D$ (how much $D$ predicts $Z$).

$$\hat{\beta}_D = \beta_D + \beta_Z \cdot \delta_{ZD}$$

**Using this for sensitivity analysis**: You can bound bias by asking: how strongly would an unmeasured confounder need to relate to both $D$ and $Y$ to explain away your estimated effect? If the answer is "more strongly than any measured covariate relates to either," your finding is robust. This is the logic behind Oster (2019) and Cinelli-Hazlett sensitivity analysis.

**What breaks**: The sign of the bias depends on the sign of *both* $\beta_Z$ (Z's effect on Y) and $\delta_{ZD}$ (Z's association with D). You can have upward or downward bias depending on whether the confounding works with or against your estimate.

---

## 5. Propensity Score Methods

### Why the propensity score exists

Suppose you want to control for confounders $X$ to estimate causal effects. If $X$ is a 50-dimensional vector, matching or conditioning on it directly is nearly impossible — the curse of dimensionality. You need a way to summarize the confounding in $X$ without losing the information needed to remove bias.

The key insight is that you don't need to balance $X$ directly. You only need to make treatment assignment *as-if random* conditional on something. Rosenbaum and Rubin (1983) showed that conditioning on a single scalar — the probability of treatment given $X$ — is sufficient.

**The propensity score**: $e(X) = P(D=1 \mid X)$

**The balancing property**: $D \perp X \mid e(X)$

Within subgroups of units with the same propensity score, treatment assignment is independent of $X$. You've collapsed a high-dimensional balancing problem into a one-dimensional one.

**What breaks**: The propensity score only removes bias from *observed* confounders. Unmeasured confounders are not addressed. Propensity score methods require the *ignorability assumption*: conditional on $X$, treatment is as-if random. This is also called unconfoundedness or selection on observables.

---

### Propensity Score Matching (PSM)

**The problem it solves**: You want treated and control groups that are comparable on observed covariates. Direct matching on $X$ fails in high dimensions because you can't find close matches. Matching on the propensity score $e(X)$ reduces this to one dimension.

**Procedure**: Estimate $e(X)$ via logistic regression or a flexible classifier. For each treated unit, find control unit(s) with the nearest $e(X)$. Use only matched pairs for analysis.

**Variants**: nearest-neighbor (1:1 or 1:k), caliper matching (only match if $|e(X_i) - e(X_j)| < \epsilon$), kernel matching (weighted average of all controls, weighted by proximity).

**How to assess success**: Check covariate balance in the matched sample using standardized mean differences (SMD). SMD = (mean difference) / (pooled SD). SMD < 0.1 per covariate indicates good balance.

**What breaks**: PSM throws away unmatched units, reducing efficiency. If the propensity score model is misspecified, the resulting matches may not actually balance covariates. Poor overlap — regions where $e(X) \approx 0$ or $e(X) \approx 1$ — means few or no valid matches; estimates in those regions are extrapolations.

---

### Inverse Probability Weighting (IPW)

**The problem it solves**: Instead of discarding units to create balance, you want to reweight the entire sample to create a pseudo-population where treatment is independent of $X$. Treated units who look like control units (low $e(X)$) are upweighted; control units who look like treated units (high $e(X)$) are upweighted. The result is a reweighted sample where covariates are balanced.

**The weights**: $w_i = \frac{D_i}{e(X_i)} + \frac{1-D_i}{1-e(X_i)}$

Treated units are weighted by the inverse of their probability of being treated. Control units are weighted by the inverse of their probability of being control. The IPW estimator:

$$\hat{\tau}_{IPW} = \frac{1}{n}\sum_i w_i Y_i \cdot (2D_i - 1)$$

**Why this shape**: The weight $1/e(X)$ for treated units corrects for the fact that treated units with low propensity (who look like controls) are underrepresented in the treated group. Upweighting them restores representativeness.

**What breaks**: Extreme propensity scores ($e(X) \approx 0$ or $1$) produce enormous weights, causing high variance or numerical instability. Stabilized weights — multiplying by the marginal treatment probability — reduce this. Trimming units with extreme propensity scores reduces variance at the cost of changing the estimand.

---

### Doubly Robust Estimators

**The problem they solve**: Both propensity score models and outcome regression models can be misspecified. If your propensity score model is wrong, IPW is biased. If your outcome model is wrong, regression adjustment is biased. Can you get an estimator that is correct if *either* model is right?

Yes. The doubly robust (DR) estimator combines both:

$$\hat{\tau}_{DR} = \frac{1}{n}\sum_i \left[\frac{D_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} + \hat{\mu}_1(X_i) - \frac{(1-D_i)(Y_i - \hat{\mu}_0(X_i))}{1-\hat{e}(X_i)} - \hat{\mu}_0(X_i)\right]$$

**Why this shape**: The terms $\hat{\mu}_1(X_i)$ and $\hat{\mu}_0(X_i)$ are the predicted outcomes under each treatment. The IPW residual terms — $\frac{D_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)}$ — correct for any misfit of the outcome model using the propensity. If the outcome model is perfect, the residuals are zero and the IPW correction contributes nothing. If the propensity model is perfect, the IPW correction removes all bias regardless of outcome model accuracy.

**What breaks**: Double robustness does not extend to the case where both models are wrong. It also does not guarantee good *variance* properties — if both models are somewhat wrong, estimates can be unstable. Targeted Maximum Likelihood Estimation (TMLE) extends this framework with better theoretical properties.

---

## 6. Difference-in-Differences (DiD)

### Why DiD exists

You want to measure the effect of a policy that was applied to some units but not others. You can't run an RCT. And you can't just compare treated and control units after treatment — the two groups might differ for many reasons unrelated to the policy.

But you observe both groups before and after the policy. The key insight: if you subtract each group's pre-period value from their post-period value, you remove all time-invariant differences between groups. Then if you subtract the control group's change from the treated group's change, you also remove any time trends that affected both groups equally.

**The estimator**:

$$\hat{\tau}_{DiD} = (\bar{Y}_{treat,post} - \bar{Y}_{treat,pre}) - (\bar{Y}_{control,post} - \bar{Y}_{control,pre})$$

The first difference removes time-invariant treated-group characteristics. Subtracting the control group's difference removes common time trends. What remains, if the key assumption holds, is the causal effect.

---

### The parallel trends assumption

**Why it's necessary**: DiD doesn't remove *differential* time trends — trends that would have affected the treatment group differently even without the treatment. If outcomes were already growing faster in the treatment group before the policy, DiD will attribute that faster growth to the policy.

**The assumption**: In the absence of treatment, the average outcomes of treated and control groups would have followed the same trend over time. Formally: $E[Y(0)_{t=1} - Y(0)_{t=0} \mid D=1] = E[Y(0)_{t=1} - Y(0)_{t=0} \mid D=0]$.

**Testing it**: You can't test this assumption for the post-treatment period — you can't observe the counterfactual trend. But you can check whether trends were parallel *before* treatment. Event study plots show estimated treatment effects at each time period. Pre-treatment coefficients that are close to zero and jointly insignificant support (but don't prove) the assumption.

**What breaks**: If the treatment group was on a different trajectory before the treatment — say, a state's economy was already recovering — DiD misattributes pre-existing trends to the policy. Also, if control units are affected by spillovers from the treatment, their $Y(0)$ trend is contaminated.

---

### Regression formulation

```
Y_it = α + β·Treat_i + γ·Post_t + τ·(Treat_i × Post_t) + ε_it
```

- $\beta$: time-invariant difference between groups
- $\gamma$: common time trend
- $\tau$: the DiD estimator — the extra change in the treatment group relative to control

With panel data, add unit fixed effects (absorbs all time-invariant unit heterogeneity) and time fixed effects (absorbs common shocks): Two-Way Fixed Effects (TWFE). Cluster standard errors at the unit of treatment assignment, not at the observation level.

---

### Staggered adoption

**The problem**: The classic 2x2 DiD with one pre and one post period breaks down when different units adopt treatment at different times. When you use TWFE with staggered timing, the coefficient on the treatment indicator is a weighted average of all pairwise DiD comparisons — including comparisons that use *already-treated* units as controls for *later-treated* units. If treatment effects are heterogeneous across timing groups, these negative weights can produce estimates with the wrong sign even when every group has a positive effect.

**Why this happens**: In TWFE, early-treated units have already been "used up" as treated observations. When they serve as comparison units for late adopters, the residualized comparison implicitly differenced their treatment effect out — which subtracts a positive effect from the control comparison, biasing estimates downward.

**Solutions**: Callaway-Sant'Anna and Sun-Abraham estimators explicitly define group-time treatment effects — the ATT for units first treated at time $g$, measured at time $t$. These are aggregated without the problematic negative weights. Always test for effect heterogeneity across timing groups before relying on TWFE.

---

## 7. Regression Discontinuity Design (RDD)

### Why RDD works

Many real-world treatment assignments use a threshold rule: admitted to a program if a score exceeds 60, eligible for a subsidy if income is below a cutoff, subject to a regulation if firm size exceeds a threshold. These rules create a discontinuity — an abrupt jump in treatment probability at a precise cutoff value of some running variable.

The insight: units just below and just above the cutoff are nearly identical in all respects except treatment status. The cutoff creates a near-randomization in a narrow neighborhood. You can compare outcomes on either side to estimate the causal effect, because any smooth differences in confounders will be continuous at the cutoff while the treatment jumps discontinuously.

---

### Sharp RDD

**When treatment assignment is fully determined by the cutoff**: $D_i = \mathbf{1}[X_i \geq c]$

**The estimand**: the treatment effect for units right at the cutoff — a Local Average Treatment Effect (LATE) for cutoff units.

$$\hat{\tau}_{RD} = \lim_{x \downarrow c} E[Y \mid X=x] - \lim_{x \uparrow c} E[Y \mid X=x]$$

**Estimation**: fit local linear regression separately on each side of the cutoff using observations within a bandwidth $h$:

$$\hat{\tau}_{RD} = \hat{\alpha}_R - \hat{\alpha}_L$$

where $\hat{\alpha}_R$ and $\hat{\alpha}_L$ are the intercepts at the cutoff from right- and left-side regressions.

**Why local linear rather than higher-order polynomials**: High-degree global polynomials fit poorly near the boundary and produce extreme estimates. Local linear regression is better-behaved near the cutoff, has lower bias-variance tradeoff at boundary points, and is standard practice.

---

### Fuzzy RDD

**When the cutoff creates a jump in treatment probability but not a deterministic assignment**: $P(D=1 \mid X)$ has a discontinuity at $c$ but $D$ is not a deterministic function of $X$.

Examples: a scholarship eligibility cutoff where not all eligible students apply, a referral threshold where not all referred patients receive treatment.

**The solution**: The discontinuity in $P(D=1 \mid X)$ at $c$ serves as an instrument for actual treatment $D$. Estimate via 2SLS (see Section 8 for IV mechanics):

$$\hat{\tau}_{Fuzzy} = \frac{\text{jump in } E[Y|X] \text{ at } c}{\text{jump in } E[D|X] \text{ at } c}$$

The fuzzy RD estimate is the treatment effect scaled by the increase in treatment probability at the cutoff — it's a LATE for compliers at the cutoff.

---

### Bandwidth selection

**The tradeoff**: narrow bandwidth means you use only observations very close to the cutoff — very plausible local comparability (low bias), but few observations (high variance). Wide bandwidth gives you more data but the treatment and control sides may differ for reasons other than treatment (higher bias from non-linearity or confounding).

**Optimal bandwidth**: the Calonico-Cattaneo-Titiunik (CCT) and Imbens-Kalyanaraman (IK) methods select bandwidth to minimize mean squared error of the RD estimator. These are data-driven and account for both bias and variance.

**What breaks**: The optimal bandwidth minimizes MSE at the asymptotically dominant terms, but in finite samples can still perform poorly. Always check sensitivity of estimates to different bandwidths. If the estimate changes dramatically across reasonable bandwidths, the RD is fragile.

---

### Validity tests

**McCrary density test**: If units can manipulate their running variable to just barely cross the cutoff, the local randomization argument breaks down. People just above the cutoff are no longer comparable to people just below — they're self-selected high achievers who pushed themselves past the threshold. Test for this by checking whether the density of the running variable is continuous at the cutoff. A spike just above the cutoff is evidence of manipulation.

**Covariate balance at cutoff**: Pre-determined characteristics (age, gender, baseline outcomes) should not jump at the cutoff — they weren't affected by treatment. Run RD specifications using pre-treatment covariates as outcomes. Significant discontinuities are evidence the cutoff reflects something other than the intended treatment assignment rule.

**Placebo cutoffs**: Run RD at cutoffs where no treatment actually occurred. These should yield null effects. Significant effects at fake cutoffs suggest the discontinuity in outcomes isn't driven by the treatment.

---

## 8. Instrumental Variables (IV)

### Why IV exists

Sometimes unmeasured confounders prevent you from estimating a causal effect, and no design (RCT, matching, DiD) can eliminate them. But suppose you can find a variable $Z$ that affects treatment $D$ but has no effect on $Y$ except through $D$, and is itself unrelated to the confounders. This variable — an instrument — lets you isolate the part of treatment variation that is free of confounding.

Think of $Z$ as a natural experiment: random variation in $D$ created by something exogenous. You use only that exogenous variation to estimate the effect of $D$ on $Y$.

---

### Three conditions for a valid instrument

**1. Relevance**: $Z$ must actually affect $D$. If $Z$ doesn't move $D$, you have nothing to work with. Testable: run a first-stage regression of $D$ on $Z$. The F-statistic on $Z$ must be large.

**2. Exclusion restriction**: $Z$ affects $Y$ only through $D$, not through any other path. This is the critical assumption and is almost never fully testable — it requires substantive domain knowledge. Any direct effect of $Z$ on $Y$ not mediated by $D$ violates this assumption and biases IV estimates.

**3. Independence (exogeneity)**: $Z$ is independent of unmeasured confounders of $D$ and $Y$. Essentially, $Z$ is as-if randomly assigned. For natural experiments, this requires arguing that $Z$ isn't systematically correlated with other factors affecting $Y$.

---

### Two-Stage Least Squares (2SLS)

**Why 2SLS has this structure**: The exclusion restriction says $Z$ only relates to $Y$ through $D$. So: first, isolate the part of $D$ that is due to $Z$ (first stage). Second, use that isolated variation to estimate the effect on $Y$ (second stage). Anything else in $D$ — the part correlated with confounders — gets removed in the first stage.

**First stage**: Regress $D$ on $Z$ (and controls $X$):
$$\hat{D}_i = \hat{\pi}_0 + \hat{\pi}_1 Z_i + \hat{\pi}_X X_i$$

**Second stage**: Regress $Y$ on $\hat{D}$ (and controls $X$):
$$Y_i = \hat{\alpha} + \hat{\tau}_{IV} \hat{D}_i + \hat{\beta}_X X_i + \epsilon_i$$

$\hat{\tau}_{IV}$ is the IV estimate. With one instrument and one endogenous variable, this equals the Wald estimator: $\hat{\tau}_{IV} = \frac{Cov(Y, Z)}{Cov(D, Z)}$.

---

### What IV actually estimates: LATE

IV does not estimate ATE or ATT in general. It estimates the **Local Average Treatment Effect** — the treatment effect for *compliers* only.

**Compliers**: units whose treatment status changes because of $Z$. If $Z$ is a draft lottery, compliers are people who served in the military because they were drafted and would not have served otherwise.

**Non-compliers**:
- Always-takers: treated regardless of $Z$
- Never-takers: untreated regardless of $Z$
- Defiers: do the opposite of what $Z$ suggests (usually assumed away via monotonicity)

IV cannot say anything about always-takers or never-takers — their treatment didn't change with $Z$, so the instrument provides no information about their treatment effects.

**Why this matters**: LATE may be very different from ATE. If the marginal complier population is unusual (e.g., people who respond to a specific recruitment campaign), the LATE generalizes only to similar recruitment scenarios, not to the broader population.

---

### Weak instrument problem

**The problem**: If $Z$ barely affects $D$ (weak instrument), the first-stage variation is small. Your IV estimate divides by a nearly-zero covariance, amplifying any noise or minor exclusion restriction violations into large bias. Weak IV estimates can be worse than OLS.

**Why the rule of thumb is F > 10**: The first-stage F-statistic measures the strength of the instrument. Stock-Yogo (2005) showed that F < 10 produces IV estimates with relative bias exceeding 10% of OLS bias in the worst case. This threshold has become standard, though F > 20 is preferable.

**What breaks**: Even with F > 10, if you have many instruments (overidentification) or near-weak instruments, standard IV inference is distorted. Use Anderson-Rubin confidence sets (valid even under weak instruments) or LIML (Limited Information Maximum Likelihood, less biased than 2SLS with multiple instruments).

---

### Classic instruments and why they work

**Vietnam draft lottery (Angrist)**: Lottery numbers determined draft eligibility. Lottery number $Z$ satisfies relevance (affects military service), exclusion (lottery number doesn't directly affect earnings except through service), and independence (it was actually random). LATE = effect of military service on earnings for lottery-compliers.

**Distance to college (Card)**: People living near a college are more likely to attend. Distance $Z$ affects college attendance $D$. Exclusion: distance affects earnings only through education, not through other channels (questionable — rural areas may have other features correlated with earnings). LATE = returns to education for people at the margin of attending based on proximity.

**Quarter of birth (Angrist-Krueger)**: School entry age rules mean students born in different quarters face different compulsory schooling requirements. Birth quarter $Z$ affects years of schooling $D$. Controversial exclusion restriction — some evidence birth quarter correlates with other factors.

---

## 9. Synthetic Control

### Why synthetic control exists

DiD requires a control group that parallels the treatment group in trends. IV requires an instrument. What if you have a single treated unit — one state, one country, one firm — and no single control unit tracks it well? California passes a policy, and no individual state is a good comparison. You need a counterfactual for a single unit.

The insight: a weighted combination of multiple control units may track the treated unit better than any single control. Find the weights that make the synthetic control match the treated unit's pre-treatment trajectory, then observe the post-treatment gap.

---

### Construction and identification

**The synthetic control**: $\hat{Y}_1(0)_t = \sum_{j=2}^{J} w_j Y_{jt}$

Weights $w_j \geq 0$, $\sum w_j = 1$ (convex combination). The weights are chosen to minimize the distance between the treated unit's pre-treatment outcomes (and covariates) and the weighted average of donor units' pre-treatment outcomes.

**Why convex combinations matter**: Restricting weights to be non-negative and sum to one prevents extrapolation outside the convex hull of the donor pool. If pre-treatment fit is good, post-treatment extrapolation is plausible. Unconstrained regression would allow negative weights — which could achieve perfect fit by extrapolation, making inference unreliable.

**The estimate**: $\hat{\tau}_t = Y_{1t} - \hat{Y}_1(0)_t$ for each post-treatment period $t$.

**What breaks**: The synthetic control requires good pre-treatment fit. If no convex combination of donors tracks the treated unit, you're extrapolating. Also, the method requires the treated unit to be inside the convex hull of donors on pre-treatment characteristics — if the treated unit is extreme, fit will be poor.

---

### Inference via permutation

**The problem**: You have one treated unit, so you can't do conventional hypothesis testing. There's no distribution of treatment effects to compare against.

**The solution**: Apply the synthetic control procedure to each donor unit as if it were treated. For each, compute the ratio of post-treatment RMSPE to pre-treatment RMSPE — a measure of how large the post-treatment gap is relative to pre-treatment fit.

**The p-value**: proportion of donor-unit placebos with post/pre RMSPE ratio as large as the treated unit's. If the treated unit's gap is unusually large compared to all placebos, that's evidence of a real effect.

**What breaks**: If the donor pool is small, permutation inference has low power — you can't get a p-value below 1/J. Placebos with poor pre-treatment fit should be excluded; including them inflates the distribution and makes it harder to find significance.

---

## 10. Interrupted Time Series (ITS)

### Why ITS exists

You have a single unit observed over time. A policy is introduced at a known date. You want to know whether outcomes changed after the policy in a way that exceeds what would have happened anyway.

ITS uses the pre-policy trend to project the counterfactual post-policy trajectory. The deviation from this projection is the estimated effect.

---

### Segmented regression

**The model**:
```
Y_t = β_0 + β_1·t + β_2·D_t + β_3·(t - t_interrupt)·D_t + ε_t
```

Where $D_t = \mathbf{1}[t \geq t_{interrupt}]$.

- $\beta_0$: pre-policy intercept
- $\beta_1$: pre-policy slope (time trend)
- $\beta_2$: immediate level change at the interruption (step change)
- $\beta_3$: change in slope after the interruption (ramp change)

**Why both terms**: Some interventions cause an immediate jump (a new drug becomes available — prescriptions jump overnight). Others cause a gradual change (a public health campaign — behavior shifts slowly). Both can co-occur.

**What breaks**: ITS assumes the pre-period trend would have continued unchanged absent the intervention. This fails if: (a) other events coincide with the interruption, (b) the pre-period trend was non-linear and you modeled it linearly, or (c) there's regression to the mean near the interruption (often happens when interventions are triggered by bad outcomes).

---

### Control series

**Why it's needed**: A single ITS cannot rule out that any other contemporaneous change caused the shift. Adding a concurrent control series (not subject to the intervention but measured over the same period) lets you difference out common trends, exactly like DiD.

If both treated and control series shift at the same time, the shift is not attributable to the intervention. Only the *differential* change in the treated series identifies the effect.

---

## 11. DAGs (Directed Acyclic Graphs)

### Why we need a graphical language

In observational studies, you need to decide which variables to adjust for. Common advice ("control for everything you can") is wrong — as we'll see, controlling for the wrong variables can introduce bias rather than remove it. DAGs give you a formal language for reading off exactly which variables to condition on, and which to avoid, directly from a causal graph.

A DAG encodes your causal beliefs: $A \to B$ means "$A$ directly causes $B$." Acyclic means no variable can cause itself. Each variable is independent of its non-descendants given its parents.

---

### Paths and d-separation

**The question DAGs answer**: Given a causal graph, does conditioning on set $Z$ make $X$ and $Y$ independent? This tells you whether your regression adjustment is sufficient to remove confounding.

Three path structures determine whether information flows between $X$ and $Y$:

**Chain**: $X \to M \to Y$ — information flows from $X$ to $Y$ through $M$. Conditioning on $M$ blocks this path.

**Fork**: $X \leftarrow C \rightarrow Y$ — $C$ is a common cause. Information flows from $X$ to $Y$ through $C$. Conditioning on $C$ blocks this path.

**Collider**: $X \to B \leftarrow Y$ — $B$ is a common effect. Information does *not* flow from $X$ to $Y$ through $B$ when $B$ is unobserved. But **conditioning on $B$ opens this path** — creating a spurious association.

**d-separation**: A set $Z$ d-separates $X$ from $Y$ if it blocks every path between them — blocking chains and forks by conditioning on the intermediate variable, and not conditioning on colliders (or their descendants).

---

### Backdoor criterion

**The problem it solves**: You want to estimate the causal effect of $X$ on $Y$ but confounders create backdoor paths — paths that enter $X$ from behind (via $X$'s causes). You need to identify a set $Z$ sufficient to close all backdoor paths without opening new ones.

**The criterion**: A set $Z$ satisfies the backdoor criterion for $(X, Y)$ if:
1. $Z$ blocks all backdoor paths from $X$ to $Y$ (paths that have an arrow entering $X$)
2. $Z$ contains no descendants of $X$

If $Z$ satisfies the backdoor criterion, then conditioning on $Z$ in a regression gives the causal effect of $X$ on $Y$:

$$P(Y \mid do(X)) = \sum_z P(Y \mid X, Z=z) P(Z=z)$$

**What breaks**: Violating condition 2 is the most common error. If $Z$ is a descendant of $X$ — something $X$ influences — conditioning on it blocks part of the causal path from $X$ to $Y$, creating bias. Researchers often condition on post-treatment variables (which are descendants of treatment) and unknowingly attenuate their estimates.

---

### Collider bias

**Why conditioning on a collider creates spurious associations**: Suppose $A$ and $B$ are two independent causes of $C$: $A \to C \leftarrow B$. In the population, $A$ and $B$ are uncorrelated. But if you condition on $C$ — restrict your sample to units where $C=1$ — then knowing $A=1$ tells you something about $B$: it's less likely to be 1 (since $A$ alone can explain $C$). Conditioning on $C$ induces a negative correlation between $A$ and $B$ that doesn't exist in the full population.

**Real examples**:
- **Berkson's paradox**: In a hospital, having disease A and disease B are independent in the general population. But in hospitalized patients (conditioning on hospitalization = collider), they appear negatively correlated, because either disease can cause hospitalization.
- **Selection bias as collider bias**: If you study a phenomenon in a selected sample (people who respond to surveys, firms that survived), you've conditioned on a collider — survival or participation is often a common effect of the study variables.
- **Controlling for mediators**: If $X \to M \to Y$ and you condition on $M$, you've conditioned on a descendant of $X$ and blocked the causal path — this is a case where conditioning introduces bias by blocking the mechanism.

---

### Front-door criterion

**When backdoor adjustment fails**: If there is unmeasured confounding $U$ between $X$ and $Y$ ($X \leftarrow U \rightarrow Y$), you cannot block all backdoor paths by conditioning on observables. But if there exists a mediator $M$ on all directed paths from $X$ to $Y$, and $M$ has no backdoor paths of its own that aren't blocked, you can still identify the causal effect.

**The mechanism**: First estimate the effect of $X$ on $M$ (no confounding, since $U$ has no direct path to $M$). Then estimate the effect of $M$ on $Y$ controlling for $X$ (which blocks the backdoor $M \leftarrow X \leftarrow U \rightarrow Y$). Multiply and integrate.

---

## 12. Mediation Analysis

### Why mediation matters

Knowing that $X$ causes $Y$ is valuable, but often you want to know *how*: through what mechanism? A drug might lower blood pressure through two mechanisms — reducing inflammation and changing kidney function. Understanding the mechanism helps you design better interventions and understand what would happen if you intervened on the mechanism directly.

---

### Direct and indirect effects

**The decomposition**:
- **Total effect** (TE): the full causal effect of $X$ on $Y$
- **Natural Direct Effect** (NDE): the effect of $X$ on $Y$ when $M$ is *fixed at its value under control* — i.e., the path not through $M$
- **Natural Indirect Effect** (NIE): the effect of $X$ on $Y$ that operates through $M$ changing

Total effect = NDE + NIE.

**Why "natural"**: "Natural" refers to the mediator taking its naturally-occurring value, not an experimentally-set value. This requires potential outcomes notation: $Y(x, M(x'))$ = outcome when $X=x$ and $M$ is set to what it would be under $X=x'$. The natural indirect effect is $E[Y(1, M(1)) - Y(1, M(0))]$ — change in $Y$ from changing $M$ as if going from control to treatment, while holding $X$ at treated.

---

### Baron-Kenny steps

**The procedure** (appropriate for linear models):
1. Regress $Y$ on $X$: coefficient $c$ (total effect)
2. Regress $M$ on $X$: coefficient $a$ ($X \to M$ path)
3. Regress $Y$ on $X$ and $M$: coefficient $b$ on $M$, coefficient $c'$ on $X$ (direct effect)

**Indirect effect**: $a \cdot b$; proportion mediated: $ab/c = (c - c')/c$

**Why this works (and when it doesn't)**: In a linear model with no interactions and no unmeasured treatment-mediator confounding, $ab = c - c'$ exactly. In non-linear models or with interaction terms, this decomposition fails. The correct approach requires potential outcomes.

**What breaks**: The critical identification assumption is no unmeasured confounding between $M$ and $Y$ conditional on $X$ and other covariates. If something unmeasured affects both the mediator and the outcome (a mediator-outcome confounder), Baron-Kenny gives biased decompositions. This is often violated in practice and cannot be tested from observational data alone.

---

## 13. Uplift Modeling (Heterogeneous Treatment Effects)

### Why HTE matters

Average treatment effects tell you whether a treatment works on average. But most treatments have heterogeneous effects — they work well for some people and poorly or even negatively for others. If you can predict *who* benefits most, you can target treatment efficiently: deploy only to positive-effect units, withhold from negative-effect units (the "sleeping dogs" who would be harmed or deterred by treatment).

The target quantity is the Conditional Average Treatment Effect:

$$\tau(x) = E[Y(1) - Y(0) \mid X = x]$$

This is a function of covariates, not a single number.

---

### T-Learner (Two-Model)

**The logic**: If you had data on treated and control separately, you'd fit one outcome model per group. The CATE is just the difference.

**Procedure**: Train $\hat{\mu}_1(x) = E[Y \mid D=1, X=x]$ on treated observations and $\hat{\mu}_0(x) = E[Y \mid D=0, X=x]$ on control observations. Predict:

$$\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$$

**What breaks**: Each model is trained on its own subgroup, which may have very different covariate distributions. This can cause large extrapolation errors when predicting $\hat{\mu}_1(x)$ for control-region values of $x$. Also, when one group is much smaller, that model has high variance, inflating uncertainty in $\hat{\tau}(x)$.

---

### S-Learner (Single-Model)

**The logic**: Instead of separate models, include treatment as a feature and train one model.

**Procedure**: Train $\hat{\mu}(x, d) = E[Y \mid D=d, X=x]$ on all data with $D$ as an input. Predict:

$$\hat{\tau}(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)$$

**What breaks**: Regularization in the single model may shrink the coefficient on $D$ toward zero, regularizing the treatment effect away. If $D$ is not a high-importance feature relative to other covariates, the model will learn that treatment doesn't matter — regardless of the truth. S-learner is most useful when you believe treatment effect heterogeneity is smooth and related to other feature patterns.

---

### X-Learner

**The problem it solves**: T-learner underperforms when one group is much smaller than the other (say 10x more controls than treated). The treated-group model $\hat{\mu}_1$ is estimated on few observations; the control-group model $\hat{\mu}_0$ is well-estimated. Can you use the well-estimated model to impute missing potential outcomes and create a better training signal?

**Stage 1**: Fit T-learner models $\hat{\mu}_0$ and $\hat{\mu}_1$.

**Stage 2**: For each treated unit $i$, impute individual treatment effect: $\tilde{\tau}_i^1 = Y_i - \hat{\mu}_0(X_i)$. For each control unit $j$, impute: $\tilde{\tau}_j^0 = \hat{\mu}_1(X_j) - Y_j$.

**Stage 3**: Fit a model $\hat{\tau}^1(x)$ to $\{(X_i, \tilde{\tau}_i^1)\}_{i \in \text{treated}}$ and $\hat{\tau}^0(x)$ to $\{(X_j, \tilde{\tau}_j^0)\}_{j \in \text{control}}$. Combine using propensity score:

$$\hat{\tau}(x) = e(x) \hat{\tau}^0(x) + (1 - e(x)) \hat{\tau}^1(x)$$

**Why the propensity-weighted combination**: Units with high propensity $e(x) \approx 1$ are likely treated — $\hat{\tau}^0(x)$ was imputed using a well-trained $\hat{\mu}_1$, so weight toward the control-group imputed model. Units with low propensity $e(x) \approx 0$ are more like controls — weight toward the treated-group imputed model.

**What breaks**: X-learner compounds errors across stages — misfit in stage 1 propagates to stage 2 imputation and then stage 3 modeling. It is not doubly robust. In large balanced samples, it offers little advantage over T-learner.

---

### Causal Forests (Wager & Athey)

**Why standard random forests don't work for CATE**: Random forests minimize prediction error for $Y$. But we want to estimate $\tau(x) = E[Y(1) - Y(0) \mid X=x]$, which requires modeling the treatment-covariate interaction, not just $Y$. A standard forest will happily ignore treatment effect heterogeneity if it's not predictive of $Y$ in absolute terms.

**The key innovations**:

**Honest splitting**: Use separate observations for choosing the split variable/value and for estimating leaf outcomes. Without this, splits are chosen to overfit, and the same data used for splitting produces optimistically biased estimates. Honesty is what allows valid inference.

**Targeting heterogeneity**: Causal forests use a splitting criterion based on heterogeneity in treatment effects, not in $Y$ alone. Each leaf's treatment effect estimate is a local IPW or local DiM estimate.

**Local centering (residual-on-residual)**: Regress out the main effects of $X$ on both $Y$ and $D$ before fitting the forest. This removes the large main effects and allows the forest to focus on the residual variation — the heterogeneity in treatment effects.

**Asymptotic normality**: Under honest splitting and regularity conditions, individual CATE estimates $\hat{\tau}(x)$ are asymptotically normal, enabling pointwise confidence intervals. This is rare for nonparametric methods.

**Implementation**: `grf` package in R, `econml` in Python. Output: $\hat{\tau}(x)$ and confidence intervals for each unit. Also supports variable importance for heterogeneity drivers, omnibus test for whether heterogeneity exists, and best linear projection of CATE onto covariates.

**What breaks**: Like all nonparametric methods, causal forests require large samples to achieve good CATE estimates. In small samples, individual-level estimates are noisy. The method also requires the unconfoundedness assumption — without randomization or sufficient observed covariates, estimates are biased. Performance degrades with many irrelevant covariates relative to sample size.

---

### Evaluating uplift models

**The core problem**: You can never observe the ground truth $\tau(x) = Y_i(1) - Y_i(0)$ for any individual. Standard ML evaluation metrics (RMSE, AUC) require knowing the label. How do you evaluate a CATE model?

**Qini curve and AUUC (Area Under Uplift Curve)**:
Rank units by predicted $\hat{\tau}(x)$ from highest to lowest. At each proportion $k$ of the population targeted, compute the incremental gain: the difference in outcome rates between randomly-selected treated and control units within that top-$k$ group (using holdout RCT data or careful observational adjustment). A model that correctly identifies high-responders will show large incremental gain in the top percentiles.

The Area Under the Uplift Curve (AUUC) summarizes performance across all targeting thresholds. A random model has AUUC of 0.5; a perfect model has AUUC of 1.

**Uplift@k**: The uplift (incremental outcome rate) when targeting the top $k$% of predicted responders. Most practically useful for fixed-budget deployment decisions.

**CATE variance**: In RCT data, you can also test whether the predicted $\hat{\tau}(x)$ actually explains heterogeneity: regress observed individual DiM estimates on $\hat{\tau}(x)$ and test whether the slope is significantly different from zero. A flat relationship means the model isn't capturing real heterogeneity.

## Flashcards

**$Y_i(1)$?** #flashcard
what outcome unit $i$ would have if treated

**$Y_i(0)$?** #flashcard
what outcome unit $i$ would have if not treated

**$D_i \in \{0,1\}$?** #flashcard
observed treatment indicator

**$Y_i = D_i Y_i(1) + (1 - D_i) Y_i(0)$?** #flashcard
you only ever see one of the two

**$\beta$?** #flashcard
time-invariant difference between groups

**$\gamma$?** #flashcard
common time trend

**$\tau$: the DiD estimator?** #flashcard
the extra change in the treatment group relative to control

**Always-takers?** #flashcard
treated regardless of $Z$

**Never-takers?** #flashcard
untreated regardless of $Z$

**Defiers?** #flashcard
do the opposite of what $Z$ suggests (usually assumed away via monotonicity)

**$\beta_0$?** #flashcard
pre-policy intercept

**$\beta_1$?** #flashcard
pre-policy slope (time trend)

**$\beta_2$?** #flashcard
immediate level change at the interruption (step change)

**$\beta_3$?** #flashcard
change in slope after the interruption (ramp change)

**Berkson's paradox?** #flashcard
In a hospital, having disease A and disease B are independent in the general population. But in hospitalized patients (conditioning on hospitalization = collider), they appear negatively correlated, because either disease can cause hospitalization.

**Selection bias as collider bias: If you study a phenomenon in a selected sample (people who respond to surveys, firms that survived), you've conditioned on a collider?** #flashcard
survival or participation is often a common effect of the study variables.

**Controlling for mediators: If $X \to M \to Y$ and you condition on $M$, you've conditioned on a descendant of $X$ and blocked the causal path?** #flashcard
this is a case where conditioning introduces bias by blocking the mechanism.

**Total effect (TE)?** #flashcard
the full causal effect of $X$ on $Y$

**Natural Direct Effect (NDE): the effect of $X$ on $Y$ when $M$ is fixed at its value under control?** #flashcard
i.e., the path not through $M$

**Natural Indirect Effect (NIE)?** #flashcard
the effect of $X$ on $Y$ that operates through $M$ changing
