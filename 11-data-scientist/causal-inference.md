# Causal Inference

---

## 1. Potential Outcomes Framework (Rubin Causal Model)

### Notation
- $Y_i(1)$: potential outcome for unit $i$ under treatment
- $Y_i(0)$: potential outcome for unit $i$ under control
- $D_i \in \{0, 1\}$: treatment indicator
- Observed outcome: $Y_i = D_i Y_i(1) + (1-D_i) Y_i(0)$

### Causal Estimands
- **ATE (Average Treatment Effect)**: $\tau = E[Y(1) - Y(0)]$ — average over entire population
- **ATT (Average Treatment Effect on the Treated)**: $E[Y(1) - Y(0) | D=1]$ — average for those who received treatment
- **ATU (Average Treatment Effect on the Untreated)**: $E[Y(1) - Y(0) | D=0]$
- ATE = ATT only when treatment assignment is independent of potential outcomes (ignorability)

### Fundamental Problem of Causal Inference
- For each unit, only one potential outcome is observed: $Y_i(D_i)$
- The counterfactual $Y_i(1-D_i)$ is never observed
- Causal inference = estimating the unobserved counterfactual using structural assumptions + data

---

## 2. Randomized Controlled Trials (RCTs)

- Random assignment ensures $D_i \perp (Y_i(0), Y_i(1))$ — treatment independent of potential outcomes
- Eliminates selection bias: treated and control groups are identical in expectation on all covariates
- Naive difference in means is an unbiased estimator of ATE: $\hat{\tau} = \bar{Y}_1 - \bar{Y}_0$
- **Internal validity**: the estimated effect is valid for the study population
- **External validity**: the effect generalizes to other populations/contexts — RCTs don't guarantee this
- Failures of RCTs in practice: non-compliance (use IV), spillovers (network effects), attrition bias

---

## 3. Observational Studies: Threats to Validity

### Confounding
- Variable $Z$ causes both treatment $D$ and outcome $Y$ — creates spurious association
- Example: sicker patients receive more intensive treatment → treatment appears harmful if health is not controlled

### Selection Bias
- Units select into treatment based on characteristics correlated with potential outcomes
- $E[Y|D=1] - E[Y|D=0] \neq ATE$ because of selection

### Omitted Variable Bias (OVB)
- $\hat{\beta}_D = \beta_D + \beta_Z \cdot \delta_{ZD}$ where $\delta_{ZD}$ = coefficient from regressing omitted $Z$ on $D$
- Sign and magnitude of OVB depends on: direction of $Z \to Y$ and direction of $Z \to D$
- Key for sensitivity analysis: "How strong would an unmeasured confounder need to be to explain away this finding?"

---

## 4. Propensity Score Methods

### Propensity Score
- $e(X) = P(D=1|X)$ — probability of treatment given observed covariates
- **Balancing property**: $D \perp X | e(X)$ — within strata of equal propensity, treatment is as-if random
- Estimated via logistic regression or ML classifier on pre-treatment covariates

### PSM (Propensity Score Matching)
- Match each treated unit to one or more control units with similar $e(X)$
- Nearest-neighbor, caliper (maximum distance), kernel matching variants
- Assess balance post-matching: standardized mean differences (SMD) < 0.1 per covariate

### IPW (Inverse Probability Weighting)
- Weight each unit by $w_i = \frac{D_i}{e(X_i)} + \frac{1-D_i}{1-e(X_i)}$
- Creates pseudo-population where treatment is independent of covariates
- Sensitive to extreme propensity scores (near 0 or 1) — use stabilized weights or trim

### Doubly Robust Estimators
- Combine outcome model $\hat{Y}(d, X)$ and propensity model $\hat{e}(X)$
- Consistent if either model is correctly specified (but not necessarily both)
- $\hat{\tau}_{DR} = \frac{1}{n}\sum\left[\frac{D_i(Y_i - \hat{Y}(1,X_i))}{\hat{e}(X_i)} + \hat{Y}(1,X_i) - \frac{(1-D_i)(Y_i-\hat{Y}(0,X_i))}{1-\hat{e}(X_i)} - \hat{Y}(0,X_i)\right]$

---

## 5. Difference-in-Differences (DiD)

### Setup
- Two groups (treatment, control), two time periods (pre, post)
- Treatment group receives intervention between periods; control does not
- $\hat{\tau}_{DiD} = (\bar{Y}_{treat,post} - \bar{Y}_{treat,pre}) - (\bar{Y}_{control,post} - \bar{Y}_{control,pre})$

### Parallel Trends Assumption
- In the absence of treatment, treatment and control groups would have followed the same trend
- Untestable for the post-treatment period; check pre-treatment trend parallel movement
- **Event study plots**: plot $\hat{\tau}_t$ for all periods $t$; pre-treatment coefficients should be near zero

### Regression Formulation
```
Y_it = α + β·Treat_i + γ·Post_t + τ·(Treat_i × Post_t) + ε_it
```
- $\tau$ = DiD estimator; add unit and time fixed effects for robustness
- Cluster standard errors at the unit of treatment assignment

### Staggered Adoption
- Different units adopt treatment at different times
- Classic 2×2 DiD is biased when effects are heterogeneous across timing groups (Callaway-Sant'Anna, Sun-Abraham estimators)
- Test for heterogeneous timing effects before using simple TWFE (Two-Way Fixed Effects)

---

## 6. Regression Discontinuity Design (RDD)

### Intuition
- Units assigned to treatment based on whether a running variable $X$ crosses a threshold $c$
- Units just below and above $c$ are locally comparable — as-if randomized near the cutoff

### Sharp RDD
- $D_i = \mathbf{1}[X_i \geq c]$ — deterministic assignment
- $\hat{\tau}_{RD} = \lim_{x \downarrow c} E[Y|X=x] - \lim_{x \uparrow c} E[Y|X=x]$
- Estimate via local linear regression on either side of cutoff

### Fuzzy RDD
- Treatment probability jumps at $c$ but is not deterministic
- Use the discontinuity as an IV: instrument $D$ with $\mathbf{1}[X \geq c]$; estimate via 2SLS

### Bandwidth Selection
- Optimal bandwidth (Imbens-Kalyanaraman, Calonico-Cattaneo-Titiunik): trades bias vs variance
- Narrower bandwidth: less bias, higher variance; wider bandwidth: more data but more bias

### Validity Tests
- **McCrary density test**: test for manipulation of running variable — density of $X$ should be continuous at $c$
- **Placebo cutoffs**: estimate RD at non-treatment cutoffs — should yield null effects
- **Covariate balance**: pre-determined covariates should be continuous at $c$

---

## 7. Instrumental Variables (IV)

### Conditions for a Valid Instrument $Z$
1. **Relevance**: $Z$ is correlated with treatment $D$ — testable: first-stage F-statistic
2. **Exclusion restriction**: $Z$ affects outcome $Y$ only through $D$ — not directly testable, requires domain reasoning
3. **Independence/Exogeneity**: $Z$ is independent of unmeasured confounders — requires domain argument

### Two-Stage Least Squares (2SLS)
- Stage 1: regress $D$ on $Z$ (and controls) → get $\hat{D}$
- Stage 2: regress $Y$ on $\hat{D}$ (and controls) → coefficient on $\hat{D}$ is IV estimate
- Estimates LATE (Local Average Treatment Effect): effect for compliers only (units whose treatment changes because of $Z$)

### Weak Instrument Test
- First-stage F-statistic < 10: weak instrument — IV estimates unreliable, large finite-sample bias toward OLS
- Rule of thumb: F > 10 for single instrument (Stock-Yogo critical values for formal test)
- Weak instruments: use LIML or Anderson-Rubin confidence sets

### Classic Instruments
- Draft lottery → military service → earnings (Angrist)
- Distance to college → college attendance → earnings
- Quarter of birth → schooling → earnings (Angrist-Krueger)

---

## 8. Synthetic Control

### Setup
- Single treated unit observed over time; no single control unit serves as counterfactual
- Construct counterfactual as weighted average of donor units: $\hat{Y}_1(0) = \sum_{j=2}^{J} w_j Y_j$
- Weights chosen to match pre-treatment outcomes (and covariates) of treated unit

### Properties
- Weights are non-negative and sum to 1 (convex combination — prevents extrapolation)
- Transparent: donor pool contribution is explicit
- Post-treatment gap $Y_1 - \hat{Y}_1(0)$ is the estimated treatment effect

### Inference
- Permutation inference: apply synthetic control to each donor unit; estimate effects for placebo units
- p-value = proportion of placebos with post/pre RMSPE ratio ≥ treated unit

---

## 9. Interrupted Time Series (ITS)

### Segmented Regression
```
Y_t = β_0 + β_1·t + β_2·D_t + β_3·t·D_t + ε_t
```
- $D_t = \mathbf{1}[t \geq t_{\text{interrupt}}]$
- $\beta_2$: immediate level change at interruption
- $\beta_3$: change in slope after interruption
- Add seasonal terms for periodic data

### Control Series
- Include a concurrent control series not subject to the intervention
- DiD-style: subtract control series trend from treated series to remove secular trend

---

## 10. DAGs (Directed Acyclic Graphs)

### Concepts
- **Node**: variable; **Edge**: $A \to B$ means $A$ causes $B$ (directly)
- **Path**: sequence of nodes connected by edges (any direction)
- **d-separation**: set $Z$ d-separates $X$ from $Y$ if all paths between them are blocked given $Z$
- **Blocked path**: chain $A \to B \to C$ blocked by conditioning on $B$; fork $A \leftarrow B \rightarrow C$ blocked by conditioning on $B$; collider $A \to B \leftarrow C$ blocked unless conditioning on $B$

### Adjustment Criteria
- **Backdoor criterion**: set $Z$ satisfies backdoor criterion for $(X, Y)$ if: (1) $Z$ blocks all backdoor paths from $X$ to $Y$; (2) $Z$ contains no descendants of $X$. Conditioning on $Z$ gives causal effect.
- **Front-door criterion**: when backdoor adjustment is impossible, use mediator $M$ on all paths from $X$ to $Y$

### Collider Bias
- Conditioning on a collider $B$ (common effect of $X$ and $Y$) opens a path between $X$ and $Y$ — creates spurious association
- Example: conditioning on "hospitalized" creates spurious correlation between two independent causes of hospitalization
- Selection bias is often collider bias: sampling on a collider

---

## 11. Mediation Analysis

### Framework
- Total effect: $X \to Y$ (direct + indirect)
- Direct effect: $X \to Y$ not through mediator $M$
- Indirect effect: $X \to M \to Y$

### Baron-Kenny Steps
1. Regress $Y$ on $X$: coefficient $c$ (total effect)
2. Regress $M$ on $X$: coefficient $a$ (X → M)
3. Regress $Y$ on $X$ and $M$: coefficient $b$ on $M$ (M → Y controlling X), coefficient $c'$ on $X$ (direct effect)
4. Indirect effect = $ab$; proportion mediated = $ab/c$

### Modern Approach
- Baron-Kenny is ad hoc; use potential outcomes or SEM (Structural Equation Modeling) for formal identification
- Identification of natural direct/indirect effects requires no unmeasured treatment-mediator confounding

---

## 12. Uplift Modeling (Heterogeneous Treatment Effects)

### Goal
- Estimate $\tau(x) = E[Y(1) - Y(0) | X = x]$ — CATE (Conditional Average Treatment Effect)
- Useful for targeting: treat only units where treatment has positive expected effect

### Learners

**T-learner (Two-model)**
- Train separate outcome models: $\hat{\mu}_1(x) = E[Y|D=1, X=x]$, $\hat{\mu}_0(x) = E[Y|D=0, X=x]$
- $\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$
- Simple but high variance when treatment groups have different feature distributions

**S-learner (Single-model)**
- Train one model including treatment as a feature: $\hat{\mu}(x, d) = E[Y|D=d, X=x]$
- $\hat{\tau}(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)$
- May regularize treatment effect to zero if $D$ is not an important feature

**X-learner**
- Stage 1: fit T-learner models $\hat{\mu}_0$, $\hat{\mu}_1$
- Stage 2: impute treatment effects: $\tilde{\tau}_i^1 = Y_i - \hat{\mu}_0(X_i)$ for treated; $\tilde{\tau}_i^0 = \hat{\mu}_1(X_i) - Y_i$ for control
- Stage 3: regress $\tilde{\tau}^1$ on $X$ in treated group; $\tilde{\tau}^0$ on $X$ in control group; combine with propensity score weighting
- Better than T-learner when treatment/control sample sizes are imbalanced

**Causal Forests (Wager & Athey)**
- Adaptation of random forests that targets $\tau(x)$ directly
- Uses honest splitting (separate observations for split selection and leaf estimation)
- Provides asymptotically normal CATE estimates; supports confidence intervals
- Implementation: `econml`, `grf` (R), `causalml`

### Evaluation
- No ground truth for $\tau(x)$ — use Qini curve / AUUC (Area Under Uplift Curve)
- Rank units by predicted uplift; measure incremental gain vs random targeting
- Uplift@k: uplift among top-$k$ predicted targets
