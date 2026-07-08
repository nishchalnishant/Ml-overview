---
module: Data Scientist
topic: Data Scientist Interview Questions
subtopic: ""
status: unread
tags: [datascientist, ml, interview-questions]
---
# Data Scientist Interview Questions

A consolidated question bank spanning general DS interview prep, EDA, SQL, statistics, business metrics, causal inference, and experiment design. Distinct from ML Engineer prep: heavier emphasis on SQL fluency, business judgment, and experimentation rigor than on model architecture or deployment.

Organized by difficulty (Easy → Medium → Hard) with topical subheadings retained within each tier for navigability.

---

## Easy

#### General DS Interview Prep

##### Q: What separates a Data Scientist interview from an ML Engineer interview?
A: ML Engineer interviews weight model architecture, training infrastructure, and deployment/serving concerns heavily. DS interviews weight: SQL fluency (most DS work is querying and shaping data before any modeling happens), statistical rigor (correctly interpreting p-values, choosing the right test, reasoning about power), business/product sense (translating an ambiguous question like "is the product healthy" into a measurement plan), and experimentation (designing and analyzing A/B tests end-to-end). Both roles share regression/classification fundamentals, but DS interviews rarely go deep on backprop, distributed training, or model-serving latency.

#### EDA & Data Quality

##### Q: Explain the difference between MCAR, MAR, and MNAR missing data, and why the distinction matters.
A: **MCAR (Missing Completely At Random)**: the probability of missingness is unrelated to any observed or unobserved variable — e.g., a sensor randomly drops readings. Safe to drop rows/complete-case analysis without bias, though you lose power. **MAR (Missing At Random)**: missingness depends on observed variables but not on the missing value itself — e.g., older users are less likely to fill in an "income" field, but conditional on age, missingness is random. Multiple imputation (e.g., MICE) using the observed variables produces unbiased estimates. **MNAR (Missing Not At Random)**: missingness depends on the unobserved value itself — e.g., high earners are less likely to disclose income specifically because it's high. No amount of conditioning on observed variables fixes this; you need a model of the missingness mechanism itself, or domain knowledge to bound the bias. The distinction matters because the "fix" (drop, impute, or model the mechanism) depends entirely on which regime you're in, and using MCAR-appropriate fixes under MNAR silently biases every downstream estimate.

##### Q: What is the SCQA framework, and why would a Data Scientist use it?
A: Situation-Complication-Question-Answer is a storytelling structure for presenting analysis to non-technical stakeholders: state the **Situation** (context everyone agrees on), introduce the **Complication** (the problem or tension — a metric moved, a decision is needed), pose the **Question** that follows naturally from the complication, then give the **Answer** (your recommendation) before the supporting detail. It matters because raw analytical narratives (methodology first, conclusion last) bury the actionable insight; SCQA front-loads the "so what" the way executives actually want to consume a data-driven recommendation.

#### SQL & Data Manipulation

##### Q: Write a SQL query to find the second-highest salary in each department.
A: Use `DENSE_RANK()` (not `ROW_NUMBER()`, which would arbitrarily break ties) partitioned by department:
```sql
WITH ranked AS (
  SELECT
    department_id,
    employee_id,
    salary,
    DENSE_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rnk
  FROM employees
)
SELECT department_id, employee_id, salary
FROM ranked
WHERE rnk = 2;
```
`DENSE_RANK` ensures that if two employees tie for the highest salary, the "second highest" is a genuinely different salary value, not just the next row.

##### Q: Write a SQL query to compute a rolling 7-day average of daily revenue.
A:
```sql
SELECT
  date,
  revenue,
  AVG(revenue) OVER (
    ORDER BY date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS rolling_7d_avg
FROM daily_revenue
ORDER BY date;
```
`ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` gives a 7-row window (today plus the prior 6 days). Use `RANGE` instead of `ROWS` only if you need to handle gaps in the date sequence by calendar distance rather than row count.

##### Q: Write a query to deduplicate a table, keeping only the most recent record per user.
A:
```sql
WITH ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC) AS rn
  FROM user_events
)
SELECT * EXCEPT(rn)
FROM ranked
WHERE rn = 1;
```
`ROW_NUMBER()` is correct here (not `RANK`) because you want exactly one row per user even if timestamps tie — pick a deterministic tiebreaker (e.g., add `, event_id DESC` to the `ORDER BY`) to make the result reproducible.

##### Q: What's the difference between `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()`? When would each cause a bug if misused?
A: `ROW_NUMBER()` assigns a strictly increasing integer with no ties — two equal values get different numbers, arbitrarily ordered unless you add a tiebreaker. `RANK()` gives tied rows the same rank but skips the next rank(s) (1, 1, 3). `DENSE_RANK()` gives tied rows the same rank without skipping (1, 1, 2). Bugs: using `ROW_NUMBER()` for "top N by score" when scores tie silently drops rows that should be included; using `RANK()` for pagination breaks the total row count math because of the skipped ranks; using `DENSE_RANK()` when you actually want unique per-row identity (e.g., deduplication) leaves duplicate rows in the output since ties share a rank.

#### Statistics & Probability

##### Q: Explain the difference between population variance and sample variance, and why Bessel's correction exists.
A: Population variance: σ² = (1/n)Σ(xᵢ - μ)². Sample variance: s² = (1/(n-1))Σ(xᵢ - x̄)². Dividing by (n-1) instead of n corrects a systematic downward bias: because x̄ is computed from the same sample used to compute the deviations, the sample mean is by construction the value that minimizes Σ(xᵢ - x̄)², so using the sample mean understates the true spread around the (unknown) population mean. Dividing by (n-1) instead of n exactly corrects this bias in expectation, making s² an unbiased estimator of σ². The correction matters more at small n (dividing by n-1 vs n is a big relative difference at n=5, negligible at n=10,000).

##### Q: What is the Central Limit Theorem, and why does it justify using t-tests/z-tests on non-normal data?
A: The CLT states that the sampling distribution of the mean of i.i.d. random variables converges to a Normal distribution as n grows, regardless of the shape of the underlying distribution: (X̄ₙ - μ)/(σ/√n) → N(0,1). This is precisely why t-tests and z-tests — which assume normally distributed test statistics — remain valid even when the raw data is skewed or non-normal, as long as n is large enough. Practical thresholds: n≥30 is often sufficient for mild skew, n≥100 for heavier skew, and n≥1,000+ for extreme kurtosis/heavy tails, per Berry-Esseen bounds on convergence rate. This is why a t-test on highly skewed revenue-per-user data is still often defensible at typical A/B test sample sizes, but a Mann-Whitney U test is safer at small n.

##### Q: A p-value is 0.03. What does that actually mean, and what's the most common misinterpretation?
A: Correct interpretation: if the null hypothesis were true, there is a 3% probability of observing a test statistic at least as extreme as the one you observed. It is a statement about the probability of the data given the null, not a statement about the probability the null is true. The most common misinterpretation: "there's a 3% chance the null hypothesis is true" (or equivalently, "97% chance the effect is real") — this reverses the conditional and is not what a p-value measures; that quantity would require Bayesian priors to compute. Since p=0.03 < α=0.05, you reject the null at the 5% significance level — but should always pair this with the confidence interval (e.g., [+0.2%, +2.8%]) to communicate the plausible range and lower bound of the effect, since the point estimate alone overstates precision.

##### Q: When do you use a t-test vs. a z-test vs. Mann-Whitney U vs. chi-squared?
A: **z-test**: comparing means or proportions when population variance is known or n is large (proportions almost always use z since variance is a function of p). **t-test**: comparing means when population variance is unknown and estimated from the sample — one-sample, two-sample (pooled if variances are assumed equal, Welch's if not — Welch's is the safer default), or paired (same units measured twice, e.g., before/after — pairing removes between-subject variance and increases power versus an unpaired test). **ANOVA**: comparing means across more than two groups; F = MS_between/MS_within; a significant F requires a post-hoc test (e.g., Tukey HSD) with multiple-comparison correction to identify which pairs differ. **Chi-squared test**: testing independence between two categorical variables, or goodness-of-fit against an expected distribution (also the standard SRM check). **Mann-Whitney U / Kruskal-Wallis**: non-parametric alternatives to the t-test/ANOVA when data is heavily skewed or ordinal and sample size is too small to rely on the CLT. **Fisher's exact test**: categorical comparison with small expected cell counts, where the chi-squared approximation breaks down.

##### Q: Explain Type I error, Type II error, and statistical power, and how they trade off.
A: Type I error (α, false positive): rejecting a true null — concluding there's an effect when there isn't. Type II error (β, false negative): failing to reject a false null — missing a real effect. Power = 1 - β: the probability of correctly detecting a real effect of a given size. These trade off through sample size and effect size: for fixed n, lowering α (being more conservative about false positives) increases β (more likely to miss real effects) unless you also increase n. The sample size formula n = 2σ²(z_{α/2}+z_β)²/δ² makes this explicit — tightening α from 0.05 to 0.01 raises z_{α/2} from 1.96 to 2.58 (~44% more sample needed for the same power); raising power from 80% to 90% increases the required sample by ~34%.

##### Q: Explain the difference between a confidence interval and a credible interval.
A: A **frequentist confidence interval** (e.g., 95% CI) has the interpretation: if you repeated the sampling and estimation procedure many times, 95% of the constructed intervals would contain the true (fixed, unknown) parameter. It does NOT mean "there's a 95% probability the true parameter is in this particular interval" — the parameter is fixed, not random, under the frequentist framework, so that probability statement is undefined. A **Bayesian credible interval** does support that exact interpretation: given your prior and the observed data, there is a 95% posterior probability the parameter lies in the interval — because in the Bayesian framework the parameter itself is treated as a random variable with a distribution. This distinction is a frequent interview trap: candidates often state the credible-interval interpretation for a frequentist CI, which is technically wrong.

#### Metrics & Business Analytics

##### Q: How do you choose a North Star Metric for a product?
A: Four criteria: (1) it measures genuine value exchanged with the user, not just activity (e.g., Airbnb uses "nights booked," not "searches performed," because booking reflects realized value to both host and guest); (2) it's a leading indicator of revenue — it should move before revenue does, giving you an early read; (3) it's hard to game — optimizing it directly shouldn't create incentives to degrade the actual user experience (a raw "time on site" metric is easy to inflate with addictive-but-low-value design); (4) it's understandable across the org — a metric only the data team can compute or interpret won't drive aligned decision-making. Examples: Facebook = DAU, Spotify = time listening, Slack = messages sent, Duolingo = daily active learners, HubSpot = weekly active teams. The metric should sit atop an Input/Output/Guardrail hierarchy: company goal → North Star → driver/input metrics that teams can actually move → guardrail metrics that must not degrade as a side effect.

##### Q: What's the difference between an OKR and a KPI, and what's a common failure mode when teams conflate them?
A: A **KPI** (Key Performance Indicator) is an ongoing health metric tracked continuously to monitor the state of the business (e.g., churn rate, DAU) — it doesn't have an inherent target or deadline, it's just measured. An **OKR** (Objective and Key Result) is a time-bound goal-setting framework: the Objective is a qualitative, ambitious statement of intent, and Key Results are specific, measurable outcomes that would indicate the objective was achieved within a quarter or similar period. Common failure mode: treating a KPI as if it were an OKR key result — setting "increase DAU" as a Key Result without connecting it to a specific initiative or Objective turns quarterly planning into simply chasing a number that may be influenced by many factors outside the team's control, and conflates a monitoring metric (which should stay stable or improve gradually) with a stretch goal (which is meant to be ambitious and possibly not fully achieved). A well-formed Key Result should be a metric the team can causally move through specific, identified actions within the OKR period.

#### Causal Inference

##### Q: What is the "fundamental problem of causal inference," and how does the potential outcomes framework formalize it?
A: You can never observe both what happened to a unit under treatment and what would have happened to that same unit under control simultaneously — only one potential outcome is ever realized per unit. This isn't a data limitation; it's a logical impossibility. The potential outcomes (Rubin Causal Model) framework formalizes this with notation: Y_i(1) is what unit i's outcome would be if treated, Y_i(0) if untreated, and the observed outcome is Y_i = D_i·Y_i(1) + (1-D_i)·Y_i(0) — you only ever see one term. The individual causal effect Y_i(1) - Y_i(0) is unobservable; every causal inference method is a strategy for estimating this in aggregate (e.g., ATE = E[Y(1) - Y(0)]) using assumptions that substitute for the missing counterfactual.

#### Experiment Design & A/B Testing

##### Q: What is SUTVA?
A: The Stable Unit Treatment Value Assumption requires that (1) a unit's potential outcome depends only on its own treatment assignment, not on other units' assignments, and (2) there's a single, consistent version of the treatment. When SUTVA holds, a standard user-randomized A/B test's effect estimate cleanly generalizes to the population; when it's violated (e.g., through shared infrastructure, social spillovers, or a shared marketplace), the estimated effect is a tangled mix of direct and spillover effects and the simple two-sample comparison is no longer valid.

##### Q: When would you use a multi-armed bandit instead of a standard fixed-split A/B test?
A: Bandits dynamically shift traffic toward better-performing arms during the experiment itself, reducing "regret" — the cost of continuing to expose users to an inferior variant once you have strong evidence against it — whereas a standard A/B test holds a fixed split (e.g., 50/50) for its entire duration even after the winner becomes statistically clear. Use bandits when: there are many arms (more than 2-3, where a fixed A/B split would take too long to power each comparison), the cost of exposing a user to an inferior arm is high, feedback is immediate, and you primarily care about minimizing regret/user harm during the test itself — typical uses are ad creative selection, push notification content, and real-time ranking. Use standard A/B testing instead when you need a precise, unbiased estimate of the effect size, outcomes are delayed, there are regulatory/reproducibility requirements, or the goal is understanding mechanism as much as outcome. Algorithms: ε-greedy (simple, wastes exploration on clearly bad arms), UCB (adds an uncertainty bonus favoring under-explored arms), and Thompson Sampling (Bayesian posterior sampling, typically strong empirical performance).

##### Q: What is a holdout group, and what does it measure that a standard A/B test cannot?
A: A holdout group is a small (typically 1-5%) slice of users permanently excluded from all new features shipped over an extended period (e.g., 6-12 months), later compared to the rest of the user base to measure the cumulative, compounding effect of everything shipped — something individual A/B tests, each measuring a marginal effect over a short window against a moving baseline, cannot capture. It can reveal that a feature which looked neutral or positive in a 2-week test actually harms long-term retention, and can detect cross-feature cannibalization invisible to a test that only measures metrics local to the area under test. Costs: opportunity cost grows the longer the holdout persists, and releasing the holdout eventually causes a novelty-contaminated spike — mitigated by refreshing holdouts periodically and releasing gradually.

---

## Medium

#### General DS Interview Prep

##### Q: Walk through how you'd investigate "DAU dropped 20% yesterday."
A: Five-step structure, in order:
1. **Is it real?** Check the data pipeline health (are events still being logged/processed?), check the dashboard query itself (wrong filter, wrong date range), and check whether the same drop appears in the raw event table, not just the dashboard. Confirm whether the drop is in the raw numerator or whether the denominator/definition of "active" changed.
2. **Scope it.** Slice by platform (iOS/Android/web), geography, user segment (new vs. returning, logged-in vs. anonymous), and feature area. A drop isolated to one slice points to a local cause (a broken release); a uniform drop across all slices points to a global cause (logging outage, holiday).
3. **Timeline it.** Pull the drop down to the hour. A drop concentrated in specific hours suggests a timezone bug or batch-processing failure; a step change at a specific time correlates with a deploy.
4. **Generate and rank hypotheses by prior probability and cost to check** — check the cheap, likely explanations (a bad deploy, a tracking regression) before rare ones (a genuine behavior shift). Cross-reference the timeline against product releases, infra deploys, data pipeline changes, and marketing campaign starts/ends.
5. **Communicate uncertainty honestly** — state what you know, what you don't, and what data would resolve the ambiguity, rather than presenting a guess as fact.

##### Q: Give an example of a product-sense question and how you'd structure an answer.
A: Example: "We're adding a 'Save for Later' button to a shopping app. How would you measure success?" Structure: (1) clarify the goal the feature serves (increase eventual purchase conversion, reduce cart abandonment, or just improve UX?); (2) propose a primary metric tied to that goal (e.g., % of saved items that convert to purchase within 30 days); (3) propose guardrails (does adding a button reduce direct add-to-cart rate by giving users an "escape hatch"? does it clutter the UI and hurt overall conversion?); (4) propose a measurement plan (A/B test, randomized at user level, powered to detect a business-meaningful lift); (5) state what you'd watch for post-launch (segment heterogeneity — power users vs. new users may respond differently).

##### Q: How do you approach a case study / take-home with raw data and an open-ended prompt?
A: Structure the response like a mini-report: (1) restate the business question in your own words and state assumptions explicitly; (2) do EDA first — describe the data, check for quality issues, and report anything that changes the analysis plan; (3) pick a primary metric and analytical method, and justify why; (4) present findings with appropriate uncertainty (confidence intervals, not just point estimates); (5) end with a recommendation and the caveats/limitations of the analysis. Interviewers weight the reasoning and communication as much as the final number.

##### Q: Describe a time you had to make a decision with incomplete data (STAR format).
A: **Situation**: Evaluating whether to sunset a low-engagement feature that had a strong retention correlation among a small user segment. **Task**: 3 weeks, access to behavioral logs, no survey data, no established causality. **Action**: Estimated the upper-bound revenue risk using cohort LTV data assuming the segment churned at the rate of other low-engagement users; ran a fast 10-user usability session for qualitative signal; presented the recommendation explicitly bounded by uncertainty, proposing a 6-month staged deprecation with exit-survey data collection built in rather than an immediate cut. **Result**: Staged deprecation was approved; exit surveys revealed 20% of the segment used the feature as a passive but valued function. It was redesigned instead of removed, yielding +4% retention in that segment. The key interview signal here is being explicit about what you didn't know and building a decision that degrades gracefully if your assumption is wrong.

##### Q: What is the "winner's curse" and why does it matter for reported experiment results?
A: When you run many tests (many metrics, many segments, many experiments) and then select the one(s) that show significant or the largest effect, the reported effect size is systematically inflated relative to the true effect — you selected on the estimate, not the truth, and noise pushed the selected estimates in the favorable direction. This is why pre-registering the primary metric and being skeptical of "we found a huge win in this subgroup" post-hoc claims matters: the subgroup was found by searching, and the effect will likely shrink on replication.

##### Q: What are the "universal reasoning traps" you actively check for in your own analysis?
A: Correlation vs. causation (a relationship doesn't establish which way, or whether a confounder drives both); selection bias (is my sample representative, or did some filtering process change composition — e.g., survivorship bias, non-response bias); statistical vs. practical significance (large n makes tiny effects "significant" — check the confidence interval against a business-relevant threshold, not just the p-value); base rate neglect (a positive test on a rare condition is often still likely a false positive); Simpson's paradox (aggregate trend can reverse within every subgroup if segment composition differs between groups); novelty effects (short-run lift from unfamiliarity, not true value); and regression to the mean (extreme observations tend to be less extreme next time, independent of any intervention).

##### Q: Walk through the base-rate neglect trap with a worked example.
A: A disease test is 99% sensitive and 99% specific; the disease's true prevalence is 0.1%. Someone tests positive — what's P(disease | positive)? By Bayes: P(D|+) = [P(+|D)·P(D)] / [P(+|D)·P(D) + P(+|¬D)·P(¬D)] = (0.99 × 0.001) / (0.99 × 0.001 + 0.01 × 0.999) ≈ 0.00099 / 0.01098 ≈ 9%. Despite 99%/99% accuracy, a positive result only implies ~9% probability of disease, because the pool of healthy people is so much larger that even a 1% false-positive rate among them dwarfs the true positives. This generalizes directly to fraud detection, anomaly flags, and any low-base-rate classification problem — "high accuracy" claims are meaningless without knowing the base rate.

##### Q: What is Simpson's Paradox, and how would you detect it in your own analysis?
A: Simpson's Paradox is when an aggregate trend reverses (or a treatment appears better/worse overall) once you condition on a subgroup — for example, Drug A having a higher aggregate cure rate than Drug B, while Drug B outperforms Drug A within every individual patient severity subgroup, because Drug A was disproportionately given to less-severe patients. Detection: always check whether the groups being compared have similar composition on key covariates before trusting an aggregate comparison; when in doubt, report both the aggregate and the segment-level breakdown, and use a stratified or regression-adjusted estimate if segment sizes are imbalanced across groups.

#### EDA & Data Quality

##### Q: How do you detect outliers, and what's wrong with using a fixed z-score threshold?
A: The IQR fence method: Q1 - 1.5×IQR and Q3 + 1.5×IQR define "mild" outlier bounds (3×IQR for "extreme"). It's robust to skew and doesn't assume normality. The z-score method (|x - mean|/std > 3) assumes normality and uses the mean/std, both of which are themselves distorted by the outliers you're trying to detect — a single extreme value inflates the standard deviation, potentially masking other outliers (masking effect). The fix is the **modified z-score** using the median and MAD (median absolute deviation): M_i = 0.6745(x_i - median) / MAD, thresholding at |M_i| > 3.5. Median and MAD have a much higher breakdown point than mean/std, so they aren't corrupted by the very outliers you're measuring.

##### Q: How would you detect that your production model is seeing a different data distribution than it was trained on?
A: Three complementary techniques: (1) **Kolmogorov-Smirnov test** per feature, comparing the training distribution to a recent production sample — flags features whose marginal distribution shifted significantly; (2) **Population Stability Index (PSI)**, which buckets a feature into deciles and compares bucket proportions between train and production: PSI < 0.1 is stable, 0.1–0.25 is moderate shift worth monitoring, >0.25 is a significant shift requiring investigation or retraining; (3) **Adversarial validation** — train a classifier to distinguish "is this row from train or from production," using the same features as the model. If the classifier achieves high AUC (e.g., >0.7), the two distributions are meaningfully different, and the top features by importance in that classifier tell you which features drifted.

##### Q: What is data leakage, and how do you check for it before trusting a model's performance?
A: Leakage is when information that would not be available at prediction time sneaks into the training features, causing performance to look far better in offline evaluation than it will in production. Common sources: features computed using a window that includes the future (e.g., "total lifetime purchases" as a feature to predict churn, when lifetime includes post-churn data); target leakage via a proxy (a field that's essentially a restatement of the label, like "account_closed_reason" when predicting churn); train/test split done randomly on time-series data rather than by time, letting future information leak backward. Checks: inspect features with suspiciously high individual predictive power (near-perfect single-feature AUC is a red flag); reconstruct the feature computation and verify the "as-of" timestamp used is strictly before the label's outcome window; always split time-ordered data by time, never randomly, when the deployment scenario is forecasting the future from the past.

##### Q: What are the core dimensions of data quality you'd check before trusting a new dataset?
A: Completeness (missing values, missing rows relative to expected volume), accuracy (do values match ground truth / a trusted source), consistency (does the same entity have consistent values across tables — e.g., a user's country field doesn't disagree between two systems), timeliness (is the data fresh enough for the decision), uniqueness (are there unintended duplicate rows from a join fan-out or double-ingestion), and validity (do values conform to expected type/range/format — negative ages, dates in the future, categorical values outside the expected set).

##### Q: How do you handle a categorical feature with very high cardinality (e.g., 50,000 unique zip codes)?
A: Options depend on downstream use. For tree-based models: target encoding (replace category with a smoothed estimate of the target mean for that category, using cross-validation or leave-one-out to avoid leakage) or frequency encoding. For linear/regularized models: group rare categories into an "other" bucket based on a frequency threshold, or use hashing tricks (feature hashing) to control dimensionality. Always be wary of target encoding leakage — compute the encoding only from training folds, never from the row being encoded itself, or you leak the label into the feature.

#### SQL & Data Manipulation

##### Q: Write a SQL query to compute D1, D7, and D30 retention for a cohort defined by signup date.
A:
```sql
WITH signups AS (
  SELECT user_id, DATE(signup_at) AS cohort_date
  FROM users
),
activity AS (
  SELECT user_id, DATE(event_at) AS activity_date
  FROM events
)
SELECT
  s.cohort_date,
  COUNT(DISTINCT s.user_id) AS cohort_size,
  COUNT(DISTINCT CASE WHEN a.activity_date = s.cohort_date + INTERVAL '1 day' THEN s.user_id END) * 1.0
    / COUNT(DISTINCT s.user_id) AS d1_retention,
  COUNT(DISTINCT CASE WHEN a.activity_date = s.cohort_date + INTERVAL '7 day' THEN s.user_id END) * 1.0
    / COUNT(DISTINCT s.user_id) AS d7_retention,
  COUNT(DISTINCT CASE WHEN a.activity_date = s.cohort_date + INTERVAL '30 day' THEN s.user_id END) * 1.0
    / COUNT(DISTINCT s.user_id) AS d30_retention
FROM signups s
LEFT JOIN activity a ON a.user_id = s.user_id
GROUP BY s.cohort_date
ORDER BY s.cohort_date;
```
Key subtlety: retention is defined relative to the cohort date, so joining activity and filtering with `CASE WHEN` inside the aggregate (rather than filtering in the `WHERE`) lets you compute multiple retention windows in a single pass over the joined data.

##### Q: Write a SQL query for a conversion funnel (Viewed → Added to Cart → Purchased) with drop-off rates.
A:
```sql
WITH funnel AS (
  SELECT
    user_id,
    MAX(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS viewed,
    MAX(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS added,
    MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchased
  FROM events
  WHERE event_date BETWEEN '2026-06-01' AND '2026-06-30'
  GROUP BY user_id
)
SELECT
  SUM(viewed) AS viewed_count,
  SUM(CASE WHEN viewed = 1 THEN added ELSE 0 END) AS added_count,
  SUM(CASE WHEN added = 1 THEN purchased ELSE 0 END) AS purchased_count,
  SUM(CASE WHEN viewed = 1 THEN added ELSE 0 END) * 1.0 / NULLIF(SUM(viewed), 0) AS view_to_cart_rate,
  SUM(CASE WHEN added = 1 THEN purchased ELSE 0 END) * 1.0 / NULLIF(SUM(added), 0) AS cart_to_purchase_rate
FROM funnel;
```
The nested `CASE WHEN viewed = 1 THEN added` pattern enforces that each funnel step is conditioned on having completed the prior step, which is what makes it a funnel rather than three independent counts.

##### Q: Explain the difference between `LEFT JOIN` and `INNER JOIN` when the right table has duplicate keys, and why this commonly causes silent bugs.
A: If the right table has multiple rows matching a single key in the left table, both `INNER JOIN` and `LEFT JOIN` will fan out — the left row gets duplicated once per matching right row. This is the most common source of "my aggregate numbers doubled after adding a join" bugs: a seemingly innocent join to a dimension table that unexpectedly has more than one row per key (e.g., a slowly-changing dimension table with historical versions) silently multiplies your fact table's row count, inflating `SUM()`/`COUNT()` results. Defense: before joining, verify the join key is unique in the table you're joining to (`SELECT key, COUNT(*) FROM table GROUP BY key HAVING COUNT(*) > 1`), or aggregate the right table down to one row per key before joining.

##### Q: Write a query using a recursive CTE to generate a date spine, then join it to sparse daily data to fill in zero-activity days.
A:
```sql
WITH RECURSIVE date_spine AS (
  SELECT DATE '2026-01-01' AS date
  UNION ALL
  SELECT date + INTERVAL '1 day'
  FROM date_spine
  WHERE date < DATE '2026-06-30'
)
SELECT
  ds.date,
  COALESCE(SUM(e.revenue), 0) AS revenue
FROM date_spine ds
LEFT JOIN events e ON e.event_date = ds.date
GROUP BY ds.date
ORDER BY ds.date;
```
This is a standard pattern for time-series analysis: without the date spine, days with zero events simply don't appear in a `GROUP BY event_date` query, which silently breaks any rolling average or day-over-day comparison that assumes a contiguous daily series. `COALESCE` converts the `NULL` from the `LEFT JOIN` non-match into an explicit `0`.

##### Q: What is Sample Ratio Mismatch (SRM) and how would you check for it in SQL?
A: SRM is when the observed split between experiment arms deviates from the configured split (e.g., configured 50/50, observed 48/52) by more than chance would explain — it signals a bug in randomization or logging, not a real effect, and any downstream analysis on an SRM-affected experiment is invalid until the root cause is found. Check with a chi-squared goodness-of-fit test comparing observed vs. expected counts per arm:
```sql
SELECT
  variant,
  COUNT(DISTINCT user_id) AS n_users
FROM experiment_assignments
WHERE experiment_id = 'exp_123'
GROUP BY variant;
```
Take the resulting counts into a chi-squared test (e.g., `scipy.stats.chisquare`) against the expected 50/50 split; a p-value below a conservative threshold like 0.001 (conservative given the typically huge sample sizes in A/B tests) flags SRM and analysis should halt until the pipeline issue is found.

##### Q: Write a query to compute Monthly Active Users (MAU) and Daily Active Users (DAU) for every day in a quarter, plus DAU/MAU stickiness.
A:
```sql
WITH daily AS (
  SELECT event_date, COUNT(DISTINCT user_id) AS dau
  FROM events
  GROUP BY event_date
),
monthly AS (
  SELECT
    d.event_date,
    COUNT(DISTINCT e.user_id) AS mau
  FROM daily d
  JOIN events e
    ON e.event_date BETWEEN d.event_date - INTERVAL '29 day' AND d.event_date
  GROUP BY d.event_date
)
SELECT
  d.event_date,
  d.dau,
  m.mau,
  d.dau * 1.0 / NULLIF(m.mau, 0) AS stickiness
FROM daily d
JOIN monthly m ON m.event_date = d.event_date
ORDER BY d.event_date;
```
MAU here is a trailing 30-day distinct-user count as of each date (not calendar-month MAU), which is the standard "rolling 28/30-day actives" definition used for stickiness because it avoids month-boundary artifacts.

##### Q: How would you use `QUALIFY` (or an equivalent CTE pattern) to filter on a window function without a subquery?
A: `QUALIFY` lets you filter directly on a window function result in the same query, in engines that support it (Snowflake, BigQuery), avoiding a wrapping CTE/subquery:
```sql
SELECT
  user_id,
  order_id,
  order_amount,
  ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_amount DESC) AS rn
FROM orders
QUALIFY rn = 1;
```
Without `QUALIFY`, you'd need `SELECT * FROM (... ) WHERE rn = 1` — functionally identical, `QUALIFY` is purely a readability/conciseness win. This matters in interviews because using it correctly signals familiarity with modern warehouse SQL, not just ANSI-92 syntax.

##### Q: What does `EXPLAIN` tell you, and what's a concrete case where you'd use it to fix a slow query?
A: `EXPLAIN` (or `EXPLAIN ANALYZE` for actual runtime stats) shows the query planner's execution plan: which indexes are used, join algorithms chosen (nested loop vs. hash join vs. merge join), and estimated vs. actual row counts at each step. A concrete case: a query joining a 500M-row events table to a 10-row lookup table is slow — `EXPLAIN` reveals a sequential scan on the events table because a filter predicate isn't using an existing index (e.g., the predicate wraps the column in a function like `DATE(created_at) = '2026-01-01'`, which prevents index usage). Fix: rewrite the predicate as a sargable range (`created_at >= '2026-01-01' AND created_at < '2026-01-02'`) so the planner can use the index, or add an index on the filtered/joined column if none exists.

#### Statistics & Probability

##### Q: Walk through the multiple testing problem with a concrete numeric example, and how you'd correct for it.
A: If you run 20 independent hypothesis tests at α=0.05 and the null is true for all of them, the probability of at least one false positive is 1 - (0.95)²⁰ ≈ 0.64 — a 64% chance of a spurious "significant" result purely from running many tests, even though each individual test controls its error rate correctly. Corrections: **Bonferroni** sets α_adjusted = α/m (0.05/20 = 0.0025 here) — controls the family-wise error rate but is conservative, especially as m grows, and reduces power. **Benjamini-Hochberg (FDR)** controls the expected proportion of false positives among rejected hypotheses rather than the probability of any false positive: sort p-values ascending, find the largest i such that p₍ᵢ₎ ≤ (i/m)·q, and reject all hypotheses up to that rank. FDR is standard for dashboards monitoring many metrics simultaneously (tolerating a controlled fraction of false alarms); Bonferroni is standard for a small number of pre-specified primary/confirmatory metrics.

##### Q: Explain MLE vs. MAP vs. posterior mean, and connect MAP to regularization.
A: **MLE** (Maximum Likelihood Estimation) picks the parameter value that maximizes P(data | θ), ignoring any prior belief about θ. **MAP** (Maximum A Posteriori) maximizes P(θ | data) ∝ P(data | θ)·P(θ) — incorporating a prior. **Posterior mean** is the expectation of the full posterior distribution, using more information than just its mode (MAP). The regularization connection: placing a Gaussian prior on regression coefficients and taking the MAP estimate is mathematically equivalent to L2 (ridge) regularization — the negative log of a Gaussian prior is a quadratic penalty term added to the log-likelihood. A Laplace prior similarly gives L1 (lasso) regularization, since the negative log of a Laplace density is an absolute-value penalty. This means regularization can be reinterpreted as an implicit Bayesian prior on model parameters, which is a common "aha" framing interviewers look for.

##### Q: What's the difference between Cohen's d and a p-value, and why do you need both?
A: A p-value tells you whether an observed difference is unlikely to have arisen by chance under the null — it's a function of both the effect size and the sample size, so at large n even a trivially small effect produces a tiny p-value. Cohen's d = (mean₁ - mean₂)/pooled_SD is a standardized effect size that is independent of sample size — it tells you how large the effect actually is in units of variability, with conventional benchmarks of 0.2 (small), 0.5 (medium), 0.8 (large). You need both: p-value answers "is this likely real," effect size answers "is this big enough to matter." A result can be statistically significant (p<0.001) with a negligible effect size (d=0.02) at large n — the practical-vs-statistical-significance trap — so reporting only the p-value is misleading for a ship/no-ship decision.

##### Q: A conversion rate goes from 50% to 70% between two groups. Compute the relative risk and the odds ratio, and explain why they differ.
A: Relative Risk = P(event|treatment)/P(event|control) = 0.70/0.50 = 1.4 — a 40% relative increase. Odds Ratio = [0.70/(1-0.70)] / [0.50/(1-0.50)] = (0.70/0.30)/(0.50/0.50) = 2.33/1.00 = 2.33. They differ because odds and probability diverge as probability moves away from small values — odds grow super-linearly as p approaches 1, so OR overstates the effect relative to RR whenever the baseline probability isn't small. This is a common interview trap: OR is frequently misreported and misread as if it were RR (e.g., "2.33x more likely" when the true relative risk is only 1.4x), which meaningfully changes the practical interpretation of an effect, especially in medical and epidemiological contexts where OR is the standard reported statistic from logistic regression.

##### Q: What are the OLS "LINE" assumptions, and what specifically breaks if each is violated?
A: **Linearity**: the relationship between predictors and outcome is linear in the parameters — if violated, the model is systematically biased (misses curvature), visible in a residuals-vs-fitted plot showing a pattern rather than random scatter. **Independence**: residuals are independent of each other — violated by time-series autocorrelation or clustered data (e.g., repeated measures per user); standard errors become invalid (usually understated), inflating false-positive rates, even though point estimates may remain unbiased. **Normality** (of residuals, not the raw data): needed for exact small-sample inference (t-tests, F-tests on coefficients); with large n, the CLT rescues you even under non-normal residuals. **Equal variance (homoscedasticity)**: if violated (heteroscedasticity — e.g., variance grows with the fitted value), OLS coefficients remain unbiased but standard errors are wrong, again distorting inference; fix with robust/heteroscedasticity-consistent standard errors or a variance-stabilizing transform.

##### Q: What is multicollinearity, how do you detect it, and why does it matter?
A: Multicollinearity is when two or more predictors in a regression are highly correlated with each other. It doesn't bias the overall model's predictions, but it inflates the variance of the individual coefficient estimates, making them unstable (small data changes swing coefficients wildly, sometimes flipping sign) and their individual t-tests unreliable, even though the overall F-test and R² remain valid. Detect via the **Variance Inflation Factor**: VIF_j = 1/(1 - R²_j), where R²_j is from regressing predictor j on all other predictors; VIF > 10 is generally considered severe. Fixes: drop or combine correlated predictors, use regularization (ridge handles multicollinearity gracefully by shrinking correlated coefficients together), or use domain knowledge to pick the more interpretable/reliable variable of a correlated pair.

##### Q: Explain Pearson correlation vs. Spearman correlation vs. Kendall's tau, and when you'd choose each.
A: **Pearson's r** measures linear association between two continuous variables and is sensitive to outliers and non-linear (even monotonic) relationships — it can be near zero for a strong non-linear monotonic relationship. **Spearman's rho** computes Pearson's r on the ranks of the data rather than raw values — captures any monotonic relationship (not just linear) and is robust to outliers since ranks compress extreme values. **Kendall's tau** is based on the number of concordant vs. discordant pairs and is more robust to small sample sizes and ties than Spearman, though more computationally expensive (O(n²)); often preferred for small samples or when the data has many tied ranks. Rule of thumb: use Pearson when you specifically care about and expect a linear relationship; use Spearman/Kendall when you only care about monotonic association or the data is ordinal/has outliers.

##### Q: Why does "correlation does not imply causation" hold, and what are the three main ways a correlation can be non-causal?
A: A correlation between X and Y can arise from: (1) **X causes Y** (the causal case you're hoping for), (2) **reverse causation** — Y actually causes X (e.g., companies with more support tickets might be assumed to have worse products, but it could be that higher usage/more engaged customers both use the product more and file more tickets), (3) **confounding** — a third variable Z causes both X and Y independently, producing a spurious association with no direct causal link between them (the classic example: ice cream sales and drowning deaths are correlated because both are driven by hot summer weather), or (4) **pure coincidence** in small samples or from testing many pairs (spurious correlation). Without an experimental design (randomization) or a causal-inference method that addresses confounding explicitly (DiD, IV, RDD, matching), an observed correlation alone cannot distinguish among these explanations.

#### Metrics & Business Analytics

##### Q: The CEO says "engagement dropped 10% last week." How do you investigate?
A: Same discipline as the DAU-drop question in General DS Prep, applied specifically to "engagement" (which is often an ambiguous composite metric): (1) pin down the exact metric definition being referenced — engagement could mean DAU, session count, time-on-app, or a composite score, and a change in the metric's own definition or a dashboard bug is a common false alarm; (2) verify the data pipeline and confirm the drop appears in raw event counts, not just an aggregated dashboard; (3) segment by platform, geography, user tenure (new vs. returning), and feature surface to localize the drop — a drop isolated to one platform points to a release regression, a uniform global drop points to a pipeline or seasonal cause; (4) check the calendar — is this a holiday week, a known seasonal dip, or does it follow a typical day-of-week pattern that a naive week-over-week comparison ignores; (5) cross-reference the timeline against recent releases, A/B tests that concluded/shipped, marketing campaign changes, or competitor actions; (6) once localized, quantify business impact (is this within normal week-to-week variance, or several standard deviations from the rolling baseline) before escalating further.

##### Q: Walk through the AARRR framework and how you'd use it to find a growth bottleneck.
A: AARRR = Acquisition (how do users find the product), Activation (do they experience its core value quickly), Retention (do they come back), Referral (do they bring others), Revenue (do they pay). Each stage has a conversion rate to the next, and the framework's value is in identifying which stage is the actual constraint on overall growth, since effort spent optimizing a non-bottleneck stage yields little compounding benefit. Worked example: 100k acquisitions → 60% activation → 40% D30 retention → 5% paid conversion. Compare the drop-off severity at each transition against benchmarks: a 60% activation rate might be healthy, but a 40% D30 retention rate is often the weakest link relative to typical benchmarks for a consumer product — meaning growth investment is better spent on retention mechanics (onboarding follow-up, re-engagement) than on further acquisition spend, since new acquisition is being funneled into a leaky retention bucket.

##### Q: How do you calculate LTV, and what are the two main methodological approaches?
A: **Formula-based (simple) method**: LTV = ARPU × Gross Margin × (1 / Monthly Churn Rate). Worked example: ARPU=$50/month, Gross Margin=75%, monthly churn=3% → LTV ≈ $50 × 0.75 × (1/0.03) ≈ $1,250. This assumes constant churn and ARPU over the customer's lifetime, which is often wrong (churn typically decreases over tenure — the users who survive the first few months churn less than average). **Cohort survival-curve method**: LTV = Σ_t (Survival Rate at month t × ARPU × Gross Margin), summed over the projected lifetime — this uses the actual empirical retention curve per cohort rather than assuming constant churn, and is more accurate when churn is front-loaded (very common in subscription products). Use LTV:CAC ratio to judge acquisition efficiency: <1 is destroying value, 1–3 is marginal, ~3 is healthy for SaaS, >5 suggests underinvestment in growth (you could be spending more on acquisition profitably).

##### Q: What's the difference between logo churn, revenue churn (GRR), and NDR, and why do SaaS companies report NDR separately?
A: **Logo churn**: the percentage of customers (accounts) that cancel in a period, regardless of their contract size. **Gross Revenue Retention (GRR)** = (Starting MRR − Churned MRR − Contraction MRR) / Starting MRR — measures revenue retained from existing customers, excluding any growth, capped at 100%. **Net Dollar Retention (NDR)** = GRR + Expansion MRR effect, i.e., it also credits upsell/expansion revenue from existing customers, and can exceed 100%. Benchmarks: NDR <85% is a serious problem, 85–100% is adequate, 100–110% is good, >120% is exceptional (common at high-growth enterprise SaaS). Companies report NDR separately from logo churn because a company can have high logo churn (losing many small accounts) while still having excellent NDR (a few large accounts expanding enough to more than offset the losses) — the two metrics answer different investor/strategic questions (customer satisfaction breadth vs. revenue durability).

##### Q: What is Goodhart's Law, and give two concrete examples of it playing out in a tech company.
A: "When a measure becomes a target, it ceases to be a good measure" — once people know they're evaluated on a metric, they optimize the metric itself rather than the underlying goal it was meant to proxy for, often at the expense of that goal. Examples: (1) engineers evaluated on "lines of code shipped" write bloated, unnecessarily verbose code rather than concise, well-designed solutions; (2) a product team evaluated purely on DAU introduces notification spam or dark patterns (auto-playing content, infinite scroll nudges) that boost the raw DAU number while degrading genuine user satisfaction and long-term retention. Mitigations: pair every optimized metric with a counter-metric/guardrail (DAU paired with notification opt-out rate; session length paired with task completion rate), rotate which metrics are emphasized over time so no single number gets over-optimized, use holdout groups to catch long-run cumulative harm invisible to short-term metric tracking, and periodically audit whether metric improvements correspond to genuine qualitative improvements reported by users.

##### Q: How would you design an anomaly detection system for a key business metric dashboard?
A: Three components: (1) **baseline construction** — model expected seasonality (day-of-week, time-of-day), trend, and autocorrelation, rather than comparing to a flat historical average (naive comparisons flag every Monday as an "anomaly" if weekends are typically slower); use STL decomposition or Prophet to separate trend/seasonality/residual, or CUSUM for detecting sustained small shifts; (2) **thresholding** — flag points where the residual (actual minus expected) exceeds k standard deviations of the historical residual distribution, tuning k to balance false-alarm rate against detection sensitivity; (3) **root-cause investigation protocol triggered automatically on alert** — segment the anomaly by dimension (platform, region, user segment) to localize it, correlate the anomaly timestamp against recent deployments/config changes, and check whether correlated metrics moved together (a marketing-driven traffic spike should show up in acquisition metrics too, not just the one flagged metric) to distinguish a real business event from a data pipeline issue.

##### Q: A guardrail metric (e.g., page load latency) degrades slightly during an experiment that shows a positive primary metric lift. How do you decide whether to ship?
A: This is a guardrail trade-off decision, not an automatic block or automatic ship: (1) quantify the magnitude of guardrail degradation against its own defined threshold — guardrails should have pre-specified acceptable degradation bounds set before the experiment, not judged reactively; (2) understand the causal mechanism — does the primary metric lift plausibly come at the direct expense of the guardrail (e.g., richer content increases latency but also increases engagement), or are they unrelated and the guardrail move might be noise; (3) check whether the guardrail degradation itself crosses a threshold known to affect the primary metric or retention at a longer horizon (e.g., published research on load-time thresholds and bounce rate); (4) if the guardrail move is within pre-specified tolerance, ship; if it exceeds tolerance, consider a follow-up experiment isolating the latency-causing component, or ship with a required remediation plan and monitoring rather than an unconditional ship.

#### Causal Inference

##### Q: Why does randomization solve the selection bias problem, in formal terms?
A: The naive estimator E[Y|D=1] - E[Y|D=0] decomposes into ATT + selection bias, where selection bias = E[Y(0)|D=1] - E[Y(0)|D=0] — the difference in what the treated group would have looked like even without treatment, versus the actual control group. This term is non-zero whenever treatment assignment correlates with baseline potential outcomes, which is the default in observational data (sicker patients get more treatment, motivated students seek tutoring). Randomization assigns D independently of (Y(0), Y(1)) by construction — a coin flip cannot correlate with anything about the unit — so E[Y(0)|D=1] = E[Y(0)|D=0], the selection bias term vanishes identically, and the naive difference in means becomes an unbiased estimator of ATE with no adjustment or modeling required.

##### Q: Give three ways an RCT can still produce a biased or misleading estimate, even with correct randomization.
A: (1) **Non-compliance**: units assigned to treatment don't take it (or vice versa) — assignment Z no longer equals actual treatment received D; comparing by assignment gives the Intent-to-Treat effect (the effect of being offered treatment), not the effect of treatment itself, and recovering the latter requires an IV approach using assignment as an instrument for actual uptake. (2) **Attrition bias**: units drop out non-randomly (e.g., sicker patients leave the treatment arm), so the observed outcomes at the end are no longer representative of the originally randomized groups — randomization is effectively broken for the analyzed sample. (3) **SUTVA violations / spillovers**: if treating one unit affects another (vaccination reducing transmission to unvaccinated neighbors, a marketplace feature changing supply available to the control group), the assumption that a unit's outcome depends only on its own assignment breaks, and the measured effect conflates direct and indirect/spillover effects in an ambiguous way.

##### Q: Explain Difference-in-Differences with an example, including its key identifying assumption.
A: Scenario: a loyalty program launches in one city; you want its causal effect on revenue but can't randomize a whole city. Find a comparable city that didn't get the program (control), and measure revenue in both cities before and after launch. DiD estimator: (revenue change in treatment city) − (revenue change in control city). Subtracting each city's own before/after difference removes time-invariant city-specific factors; subtracting the control city's change further removes common time trends (economy-wide seasonality, inflation) that would have affected both cities regardless of the program. What remains, if the key assumption holds, is the causal effect of the program. **Parallel trends assumption**: absent the program, the two cities' revenue would have followed the same trend over time. This is fundamentally untestable for the post-period (you can't observe the counterfactual), but you can build confidence by showing pre-treatment trends were similar via an event-study plot with statistically insignificant pre-treatment coefficients.

##### Q: What is a confounder, and how do DAGs help decide what to control for?
A: A confounder is a variable that causally influences both the treatment and the outcome, creating a spurious association between them if not accounted for (e.g., illness severity drives both who receives intensive treatment and who has worse outcomes, making intensive treatment look harmful in a naive comparison). Directed Acyclic Graphs formalize which variables to control for using the **backdoor criterion**: a set Z is sufficient to block confounding if it blocks every "backdoor path" (a path entering the treatment variable via an arrow, rather than leaving it) between treatment and outcome, and Z contains no descendants of the treatment. Critically, DAGs also show when controlling for a variable *introduces* bias: conditioning on a **collider** (a common effect of two other variables, X → B ← Y) creates a spurious association between X and Y that didn't exist unconditionally — this is the mechanism behind Berkson's paradox and much observational-study selection bias (e.g., studying only survey respondents or company survivors implicitly conditions on a collider).

#### Experiment Design & A/B Testing

##### Q: How do you choose a Minimum Detectable Effect (MDE), and what are the two most common mistakes teams make when setting it?
A: MDE should answer "what's the smallest effect that would be worth the cost of shipping this change?" — a business question about the threshold of actionability, not a statistical or a "what do we expect to see" question. Once fixed, MDE determines required sample size and thus experiment duration. Two common mistakes: (1) setting MDE to whatever effect the team "expects" — this is circular reasoning and leads to underpowered experiments if the true effect turns out smaller than the optimistic expectation; (2) inverting the logic and setting MDE to "whatever is statistically detectable given how long we're willing to run the test," rather than "the smallest effect that matters to the business" — this can produce experiments that either run needlessly long (chasing effects too small to matter) or ship changes based on underpowered null results (missing effects that would have mattered, because the experiment was too short to detect them, not because the effect doesn't exist).

##### Q: What is CUPED and how does it improve statistical power without more users?
A: CUPED (Controlled-experiment Using Pre-Experiment Data) reduces the variance of the outcome metric using a pre-experiment covariate correlated with it, rather than reducing variance by collecting more data. Formula: Y_adjusted = Y - θ(X - X̄), where X is a pre-experiment covariate (often the same metric measured before the experiment started) and θ = Cov(Y,X)/Var(X) is the OLS coefficient of Y on X. Centering by X̄ ensures E[Y_adjusted] = E[Y], so the treatment effect estimate stays unbiased — only the variance shrinks: Var(Y_adjusted) = Var(Y)(1 - ρ²) where ρ = Corr(Y,X). If pre-experiment behavior is strongly predictive of in-experiment behavior (ρ=0.7, common for engagement/revenue metrics with high autocorrelation), you get a 1 - 0.49 = 51% variance reduction, meaning you need roughly half the sample for the same statistical power — in practice this can halve required experiment duration. Critical requirement: X must be strictly pre-treatment; if it's measured during or after the treatment period, the treatment itself could have influenced X, biasing the adjustment. CUPAC extends the idea by using an ML model's out-of-fold predictions (trained on pre-experiment data) as the covariate instead of a single linear proxy, capturing non-linear relationships for further variance reduction.

##### Q: What is a switchback experiment, and when would you use one instead of standard user-level randomization?
A: A switchback experiment randomizes time windows (e.g., alternating 30-minute blocks) rather than users — during a treatment window, the entire system (all drivers, all riders) experiences treatment; during a control window, everyone experiences control. Use this specifically when user-level randomization is invalidated by shared-resource SUTVA violations, most commonly two-sided marketplaces (ridesharing, food delivery, logistics) where every unit's outcome depends on the pooled supply/demand, making a clean user-level randomization boundary impossible. Analysis regresses the outcome on a treatment-window indicator, controlling for time-of-day and day-of-week fixed effects: Y_t = α + β·1[treatment window] + time-of-day FE + day-of-week FE + ε_t, where β is the estimated average treatment effect. Key failure mode: carry-over effects between adjacent windows (e.g., driver repositioning or queue state from a treatment window persists into the following control window), which violates the independence-between-windows assumption; longer windows reduce carry-over but also reduce the number of independent experimental units (fewer windows = lower power), so window length is a bias-variance tradeoff requiring domain knowledge about the system's equilibration time, sometimes mitigated with discarded buffer periods between windows.

##### Q: What's the difference between frequentist and Bayesian A/B testing, and what practical decision-making advantage does the Bayesian framing offer?
A: The frequentist framework can only state: "if the null were true, data this extreme would occur with probability p" — a statement about the data given a hypothesis, not a probability of the hypothesis itself; strictly, you cannot say "there's a 96% chance treatment is better." Bayesian A/B testing directly models uncertainty over the parameter of interest — e.g., placing a Beta(α₀,β₀) prior on each arm's conversion rate (conjugate to the Bernoulli likelihood, so the posterior after observing k conversions in n trials is simply Beta(α₀+k, β₀+n-k), no MCMC needed) — and from the resulting posteriors can directly compute P(θ_B > θ_A) via Monte Carlo sampling, which is exactly the probability decision-makers intuitively want. It also enables **expected loss** decision-making: E[max(θ_A - θ_B, 0)] quantifies the expected cost of wrongly choosing treatment, letting you stop once that expected cost drops below a business-defined threshold — a framing that directly matches the actual business decision (magnitude of potential error) rather than a binary significant/not-significant threshold. Caveat: Bayesian results are sensitive to the chosen prior at small sample sizes, and the prior must be fixed before seeing data — choosing or adjusting a prior after seeing results is the Bayesian analogue of HARKing.

---

## Hard

#### Statistics & Probability

##### Q: A test shows a 1.5% lift in conversion with p=0.03. Do you ship? Walk through your reasoning.
A: Six-step reasoning: (1) p=0.03 < 0.05 means you'd reject the null — under the null, a lift this large or larger would occur by chance 3% of the time. (2) Don't stop at the point estimate — look at the confidence interval, e.g., [+0.2%, +2.8%]; the lower bound (+0.2%) is the conservative number to use in a business case, since the true effect could be as small as that. (3) Translate the lift into revenue: conversion lift × revenue per conversion × traffic volume — a 1.5% lift on a $100M/year surface is ~$1.5M/year, but the same lift on a $1M/year surface is ~$15K, which likely doesn't justify engineering/maintenance cost. (4) Check guardrails — did latency, error rate, refund rate, or other counter-metrics move in the wrong direction? A positive primary metric with a degraded guardrail requires an explicit trade-off decision, not an automatic ship. (5) Check for heterogeneity — segment by new vs. returning users, platform, region; a +5% lift for new users and -1% for returning users is a materially different decision (ship to new users only, or investigate the mechanism) than a uniform +1.5%. (6) Decide explicitly: ship, ship with monitoring, roll back, or run a follow-up experiment — and state why, rather than defaulting to "p<0.05 means ship."

#### Causal Inference

##### Q: What goes wrong with standard Two-Way Fixed Effects (TWFE) DiD under staggered treatment adoption?
A: With a single 2x2 DiD (one pre-period, one post-period, treatment starts at one time for everyone), TWFE is unbiased. But when different units adopt treatment at different times, the TWFE coefficient becomes a weighted average of all pairwise DiD comparisons — including comparisons where an already-treated unit serves as the "control" for a later-treated unit. If treatment effects are heterogeneous over time (e.g., effects grow after initial adoption), these comparisons implicitly subtract the early adopter's own (positive) treatment effect from what should be a clean control comparison, which can produce negative weights and even the wrong overall sign despite every group having a genuinely positive effect. Fix: use estimators explicitly designed for staggered adoption, like Callaway-Sant'Anna or Sun-Abraham, which compute group-time-specific ATTs and aggregate them without the problematic negative weighting; always test for treatment-effect heterogeneity across adoption cohorts before trusting a naive TWFE result.

##### Q: Explain Instrumental Variables, the three validity conditions, and what IV actually estimates (LATE).
A: IV is used when an unmeasured confounder makes D and Y both driven by something you can't observe or control for, but you can find a variable Z that only affects Y through its effect on D. Three conditions: (1) **Relevance** — Z must actually move D (testable via first-stage F-statistic, should exceed 10, ideally >20); (2) **Exclusion restriction** — Z affects Y only through D, no other path (not directly testable, requires domain argument); (3) **Independence/exogeneity** — Z is as-good-as-randomly assigned, uncorrelated with unmeasured confounders of D and Y. Mechanically, 2SLS: first stage regresses D on Z to isolate the exogenous variation in D; second stage regresses Y on the fitted D̂ from stage one. Critically, IV does not estimate ATE — it estimates the **Local Average Treatment Effect**, the effect only for "compliers" (units whose treatment status is actually moved by the instrument), excluding always-takers and never-takers whose treatment doesn't respond to Z. If compliers are an unusual subpopulation (e.g., people who respond to a specific recruitment channel), the LATE may not generalize to the broader population of interest.

##### Q: What is the weak instrument problem, and why is F > 10 the standard rule of thumb?
A: If Z only weakly affects D, the first-stage variation used to isolate exogenous variation in D is small, and the IV estimator — which effectively divides Cov(Y,Z) by Cov(D,Z) — divides by a near-zero quantity, amplifying any sampling noise or minor exclusion-restriction violation into large bias in the final estimate; weak-instrument IV estimates can end up worse than a naive OLS estimate. Stock and Yogo (2005) showed that a first-stage F-statistic below 10 produces IV estimates whose relative bias can exceed 10% of the OLS bias in the worst case, which is the origin of the F>10 threshold (F>20 is safer). Even above F=10, if you have multiple/weak instruments or near-weak conditions, standard inference is distorted; robust alternatives include Anderson-Rubin confidence sets (valid under weak instruments) or LIML estimation.

##### Q: What is Regression Discontinuity Design, and what's the difference between sharp and fuzzy RDD?
A: RDD exploits situations where treatment is assigned based on a threshold on a running variable (e.g., scored above 60 → admitted to a program). Units just above and just below the cutoff are assumed nearly identical except for treatment status, so comparing outcomes right at the cutoff approximates a local randomized experiment. **Sharp RDD**: treatment is a deterministic function of the running variable, D = 1[X ≥ c]; the estimate is the gap between local linear regressions fit just above and just below the cutoff, and identifies a Local Average Treatment Effect for units at the cutoff. **Fuzzy RDD**: the cutoff creates a jump in the *probability* of treatment but doesn't deterministically assign it (e.g., not everyone above a scholarship cutoff actually enrolls) — estimated by scaling the jump in outcome by the jump in treatment probability at the cutoff (equivalent to using the cutoff as an instrument via 2SLS), identifying a LATE for compliers at the cutoff. Validity is checked via the McCrary density test (looking for manipulation/bunching just above the cutoff) and covariate-balance tests at the cutoff.

##### Q: What's the difference between the T-Learner, S-Learner, and X-Learner approaches to estimating heterogeneous treatment effects (CATE)?
A: All three estimate τ(x) = E[Y(1) - Y(0) | X=x], the treatment effect as a function of covariates, useful for targeting who benefits most from a treatment. **T-Learner**: fit two separate outcome models, one on treated units and one on control units, and take the difference of their predictions — simple, but the two models can extrapolate poorly into each other's covariate regions, especially with imbalanced group sizes. **S-Learner**: fit a single model with treatment as just another input feature, predict under both D=1 and D=0, and difference — avoids the extrapolation issue, but regularization can shrink the treatment indicator's coefficient toward zero if D isn't a strong predictor relative to other features, effectively hiding real heterogeneity. **X-Learner**: designed for imbalanced group sizes — uses the well-estimated model from the larger group to impute individual treatment effects for the smaller group, fits models to those imputed effects, then combines the two using propensity-score weighting; more robust when treated/control groups are very different sizes, at the cost of compounding errors across its multiple stages. Causal Forests extend this idea with honest splitting and a treatment-effect-heterogeneity-targeted splitting criterion, additionally providing valid pointwise confidence intervals for τ(x), which none of the meta-learners provide out of the box.

##### Q: You can't run an RCT for a policy change (e.g., a new minimum wage law in one state). How would you estimate its causal effect, and what design would you choose?
A: Follow the decision tree for when randomization isn't possible: (1) if there's a sharp eligibility cutoff (e.g., a specific income or firm-size threshold determines exposure to the policy) → Regression Discontinuity; (2) if the policy applies to one group over time but not a comparable other group → Difference-in-Differences, provided you can find a plausible control state/region with similar pre-treatment trends; (3) if there's a variable that shifts exposure to the policy but has no direct effect on the outcome except through that exposure → Instrumental Variables; (4) if none of the above apply but you have rich observational covariates, use Propensity Score Matching or IPW, explicitly caveating that this only adjusts for observed confounders, not unmeasured ones. For a single state with no good comparison state, Synthetic Control (constructing a weighted combination of other states that tracks the treated state's pre-policy trajectory) is often the strongest option, with inference done via placebo permutation across the donor pool since there's only one treated unit.

#### Experiment Design & A/B Testing

##### Q: Walk through the full design of an A/B test for a new recommendation algorithm, including what could go wrong.
A: (1) **Define metrics**: one primary OEC (e.g., 7-day retention or revenue per user), guardrails (latency, error rate, abuse signals) — pre-specified to prevent HARKing. (2) **Power analysis**: n = 2σ²(z_{α/2}+z_β)²/δ² where δ is the minimum detectable effect, α=0.05, power=80% (z_β=0.84); for a retention metric with σ=0.4 and MDE=1%, n ≈ 12,800 per arm. (3) **Randomization unit**: user level (not session or request level) via a deterministic hash of user_id + experiment_id, to avoid the same user seeing both variants and to allow measuring effects that build across sessions. (4) **Pre-launch validation**: run an A/A test first — if two identical "variants" show a significant difference, the randomization or logging pipeline is broken. (5) **Run for the full pre-committed duration** — stopping early on a peek inflates false positives. (6) **Analyze**: check SRM first (halt if present); two-sample t-test or z-test on the primary metric; check guardrails; check for SUTVA violations (did control users interact with treatment users through shared infrastructure or social features?). What can go wrong: novelty effects (short-term lift from unfamiliarity that fades), seasonal confounding (the test window includes a holiday), SRM from a logging bug, and metric insensitivity (a real effect exists but the metric is too noisy to detect it at the chosen sample size).

##### Q: Derive, at a high level, why the sample size formula has the shape n = 2σ²(z_α/2 + z_β)²/δ².
A: You need the test statistic to reliably exceed the critical value z_{α/2} under the alternative hypothesis with probability equal to your desired power (1-β). Under the alternative, the test statistic is centered around δ/(σ√(2/n)) — the true effect scaled by the standard error of the difference between two sample means (the factor of 2 arises because each arm independently contributes σ²/n of variance to the difference, so Var(difference) = 2σ²/n). Setting this non-centrality to just clear z_{α/2} + z_β (the sum ensures both the significance threshold and the power requirement are simultaneously satisfied) and solving for n yields n = 2σ²(z_{α/2}+z_β)²/δ². Practical implications that follow directly from the algebra: halving the MDE (δ→δ/2) quadruples the required n (δ is squared in the denominator); increasing power from 80% to 90% raises the multiplier (z_{α/2}+z_β)² from 7.84 to 10.50 (~34% more sample); tightening α from 0.05 to 0.01 raises z_{α/2} from 1.96 to 2.58 (~44% more sample).

##### Q: What is the "peeking problem" in sequential testing, and how bad is it quantitatively?
A: A p-value's validity assumes the sample size was fixed in advance; if you check significance repeatedly as data accumulates and stop the first time p < 0.05, you're effectively running multiple implicit hypothesis tests (once per look) and reporting only the most favorable one — a form of multiple comparisons in the time dimension, not the metric dimension. Quantitatively: checking results 5 times and stopping at the first p < 0.05 inflates the effective false-positive rate to roughly 19%, nearly 4x the nominal 5% — and the inflation grows worse the more frequently you peek. Solutions: (1) **fixed-horizon testing** — commit to a sample size in advance, analyze once, monitor only for data-quality issues (SRM, logging gaps) during the run, never for outcome significance; (2) **alpha-spending functions** (O'Brien-Fleming spends little α early and reserves most for the final look; Pocock spends α evenly) that pre-allocate a fixed number of interim looks with a controlled total error rate; (3) **mSPRT / always-valid p-values**, which mix the likelihood ratio over a prior on effect size to produce a martingale test statistic whose false-positive rate is controlled regardless of when you stop — this is the basis for Optimizely's Stats Engine — at the cost of somewhat reduced power relative to a fixed-horizon test if you never actually needed to stop early.

##### Q: What is SUTVA, and give two real product scenarios where it's violated?
A: The Stable Unit Treatment Value Assumption requires that (1) a unit's potential outcome depends only on its own treatment assignment, not on other units' assignments, and (2) there's a single, consistent version of the treatment. When SUTVA holds, a standard user-randomized A/B test's effect estimate cleanly generalizes; when it's violated, the estimated effect is a tangled mix of direct and spillover effects. Real violations: (1) **two-sided marketplaces** — in ridesharing, putting half of drivers into a treatment arm (a new dispatch algorithm) changes the supply available to riders in the control arm too, since drivers and riders share the same pool; the "control" experience is contaminated by the treatment's effect on the shared market. (2) **social/viral features** — if treatment users send more invitations or their content ranking changes based on being in treatment, control users' feeds or notification volume are indirectly affected through the social graph, diluting or inflating the measured effect in an ambiguous direction depending on whether the spillover is positive or negative. Fixes for known SUTVA violations: cluster-level randomization (randomize entire social communities or geographic markets together) or switchback experiments (randomize time windows instead of users, so the whole system is in one arm at a time).

##### Q: List the most common A/B testing pitfalls and how you'd mitigate each.
A: **p-hacking**: checking results repeatedly and stopping at the first significant result inflates false positives (see peeking problem); mitigate by pre-registering sample size and analysis date, or using a proper sequential method. **HARKing**: reframing a post-hoc discovered subgroup effect as if it were the original hypothesis; mitigate by pre-registering the OEC and explicitly labeling any subgroup analysis as exploratory with no inferential weight. **Survivorship bias**: analyzing only users who remained active past day 1 excludes users who may have churned specifically because of the treatment, biasing the estimate upward; mitigate by always using intent-to-treat analysis on all originally exposed users, including subsequent churners. **Simpson's paradox**: aggregate result contradicts every subgroup's result due to imbalanced segment composition across arms; mitigate by checking covariate balance post-randomization and using stratification or regression adjustment if imbalance is found. **Leakage/contamination**: control users are inadvertently exposed to treatment (social spillover, caching bugs); mitigate by verifying assignment isolation at exposure time via logging, not just at randomization time. **Statistical without practical significance**: a tiny, meaningless effect reaches significance purely due to huge sample size; mitigate by always reporting the point estimate and CI, and basing the ship decision on whether the CI clears the pre-specified MDE, not merely on p<0.05.

##### Q: A stakeholder wants to run 5 experiments simultaneously on overlapping user populations. What issues should you flag?
A: (1) **Interaction effects** — two simultaneous changes might interact (positively or negatively) in ways neither experiment alone would reveal; e.g., a pricing test and a UI redesign test running on the same users could confound each other's measured effect if the redesign changes how users perceive the price change. Mitigate with orthogonal/factorial experiment design (ensuring random assignment to one experiment is independent of assignment to others) or by explicitly excluding users in one experiment from any other experiment touching related surfaces. (2) **Multiple testing inflation** — even without interaction, monitoring 5 experiments' primary and secondary metrics simultaneously raises the family-wise false-positive rate; apply FDR control across the standard dashboard of metrics being monitored. (3) **Guardrail attribution** — if a shared guardrail metric (like latency) degrades, it may be hard to attribute the cause to a specific one of the 5 concurrent experiments; maintaining a shared experiment registry with metadata on affected surfaces helps isolate causes. (4) **Sample size dilution** — if the same population is split across 5 experiments' variants, the effective sample per cell shrinks (e.g., 2 arms × 5 experiments = up to 32 combinations if fully crossed), which can under-power each individual test; verify the power calculation accounts for the actual overlapping-experiment traffic allocation, not a naive single-experiment assumption.

##### Q: How would you determine the right unit of randomization for an experiment testing a change to a two-sided marketplace's matching algorithm?
A: Standard user-level randomization is inappropriate here because it violates SUTVA — a two-sided marketplace's supply and demand are shared, so treatment-arm drivers/listings compete for and interact with control-arm users through the same pool, contaminating both arms' measured outcomes. The right approach depends on the strength of the network/marketplace coupling: (1) if the marketplace can be partitioned into genuinely independent sub-markets (e.g., geographically separate cities with no cross-city matching), **cluster/geo-based randomization** at the market level preserves SUTVA within each cluster while giving you enough independent clusters for inference; (2) if the marketplace can't be geographically partitioned (a single city, all users pooled), use a **switchback design**, randomizing time windows so the entire market is in one arm at a time, with fixed effects for time-of-day/day-of-week and attention to carry-over effects between adjacent windows when choosing window length. In both cases, standard user-level power calculations don't directly apply — you must compute power based on the number of independent clusters or time windows, which is typically far smaller than the number of users, often requiring a longer experiment duration than a naive user-level power calculation would suggest.

##### Q: A "significant" holdout-vs-control comparison shows the holdout group has *higher* satisfaction scores after 9 months of withheld features. How do you interpret this, and what are the confounds?
A: A naive read is "our shipped features made things worse" — but several confounds must be ruled out before that conclusion is safe. (1) **Self-selection into survival**: if satisfaction is measured only among users still active at 9 months, differential attrition between the holdout and mainline groups (e.g., the mainline group's least-satisfied users churned out and stopped being surveyed, while holdout's low-satisfaction users had nothing new to dislike and also churned similarly) can flip the comparison — this needs an intent-to-treat check using the original cohorts, not just currently active users. (2) **Survey response bias**: if survey response rates differ between groups (holdout users might be systematically different responders), the observed satisfaction gap partly reflects who answered, not true population sentiment — check response rates and consider non-response weighting. (3) **Composition drift**: over 9 months, organic user mix (new signups, marketing campaigns) could differentially populate holdout vs. mainline if the randomization/assignment mechanism isn't perfectly sticky over time — verify the two groups still have balanced covariates at month 9, not just at assignment. (4) **Aggregation masking heterogeneity**: it's possible most shipped features genuinely helped, but one or two specific changes were actively harmful and dominate the aggregate holdout effect — decompose by feature-launch cohort within the mainline group to identify which specific launches drove the gap, rather than treating "everything shipped" as a monolith. Only after ruling these out would you trust the result enough to launch feature-level post-hoc analyses or holdout-informed rollbacks.

---

## Appendix: Reference Formulas & Frameworks

- **Sample size**: n = 2σ²(z_{α/2}+z_β)²/δ²
- **CUPED adjustment**: Y_adjusted = Y - θ(X - X̄), θ = Cov(Y,X)/Var(X)
- **VIF**: VIF_j = 1/(1 - R²_j)
- **Modified z-score**: M_i = 0.6745(x_i - median)/MAD
- **LTV (simple)**: ARPU × Gross Margin × (1/Monthly Churn Rate)
- **Bayes' theorem**: P(D|+) = P(+|D)P(D) / [P(+|D)P(D) + P(+|¬D)P(¬D)]
- **AARRR**: Acquisition → Activation → Retention → Referral → Revenue
- **SCQA**: Situation → Complication → Question → Answer
