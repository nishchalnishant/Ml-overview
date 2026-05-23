---
module: Data Scientist
topic: Data Scientist Interview Prep
subtopic: ""
status: unread
tags: [datascientist, ml, data-scientist-interview-prep]
---
# Data Scientist Interview Prep: First-Principles Reasoning Guide

This guide is not a template collection. It is a reasoning guide. For each interview area, it explains *why* the correct reasoning structure looks the way it does — so you can reconstruct the right answer to any variant of a question, not just the question you memorized.

---

## Table of Contents

1. [How to Use This Guide](#how-to-use-this-guide)
2. [Product Sense Questions](#product-sense-questions)
3. [Metrics Investigation](#metrics-investigation)
4. [A/B Test Design](#ab-test-design)
5. [A/B Test Analysis and Interpretation](#ab-test-analysis-and-interpretation)
6. [SQL Patterns](#sql-patterns)
7. [Statistics Questions](#statistics-questions)
8. [Causal Inference](#causal-inference)
9. [Python and Pandas Patterns](#python-and-pandas-patterns)
10. [Behavioral Questions (STAR)](#behavioral-questions-star)
11. [Case Study Structure](#case-study-structure)
12. [Universal Reasoning Traps](#universal-reasoning-traps)

---

## How to Use This Guide

The goal of a DS interview is to demonstrate that your thinking is *structured and self-correcting* — not that you've memorized the right answer. Interviewers at strong companies (Meta, Google, Airbnb, Stripe, Netflix) are actively looking for:

- **Hypothesis generation**: can you enumerate the possible causes of something before committing to one?
- **Prioritization**: given many hypotheses, do you check the most likely / cheapest to disprove first?
- **Self-awareness about uncertainty**: do you know what you don't know, and what additional data would change your answer?
- **Calibration**: do you distinguish between "statistically significant" and "practically meaningful"?

These are the things this guide teaches. Read each section to understand the reasoning structure, not the answer template.

---

## Product Sense Questions

### What the interviewer is actually testing

Not: whether you know what a "North Star metric" is.

Yes: whether you can think from the perspective of the *business* and the *user* simultaneously — and whether you understand that a metric is a proxy for something real, and proxies can be gamed or mislead you.

A product sense question ("how would you measure success of X") is really asking: do you understand what the feature is *for*, who it serves, what "working" looks like mechanistically at the user level, and how that translates to observable data?

### The reasoning structure

Start with first principles: a feature exists to change user behavior in a way that benefits the user and the business. Before picking any metric, you need to answer:

**What is the causal chain?** Feature exists → user discovers it → user tries it → user gets value from it → user behavior changes → business outcome improves. Each step in that chain has a measurable signal. Your metric choices should be grounded in which step you care most about.

The reason most people's metric answers are weak is they skip this step and jump straight to naming a metric ("I'd track DAU"). DAU is a company-level metric. It tells you almost nothing about whether a specific feature worked — it's too far downstream and too aggregated.

The second thing to anchor: a metric is a proxy for value. Ask yourself: *if this metric goes up, does that mean the feature is actually working?* If you can construct a scenario where the metric goes up but users are not better off, the metric is gameable or misleading. A good metric is hard to game without actually delivering value.

Third: metrics should come in pairs — a *progress metric* and a *guardrail metric*. The progress metric tells you if the feature is succeeding. The guardrail tells you if success is being achieved by cannibalizing something else or harming a different user population. Without guardrails, you can make any metric go up by creating a bad user experience elsewhere.

### The pattern in action

**Question**: "How would you measure success of adding a 'Save for Later' button in a shopping app?"

**Step 1 — What is this feature for?** Users want to save items they're interested in but not ready to buy. The business wants to capture purchase intent and convert it later rather than lose it. So the causal chain is: user sees item → saves it → returns and purchases it.

**Step 2 — What is the primary metric?** The thing we actually care about: did saves lead to purchases that wouldn't have happened otherwise? Primary metric: **purchase rate of saved items within 28 days** (not just "saves," because saves without eventual purchase mean nothing).

**Step 3 — Secondary signals**: What tells you the feature is being used as intended, before the purchase outcome is visible? Save rate (adoption), return visits to saved-items page, session frequency for users who saved vs. not.

**Step 4 — Guardrails**: What must not degrade? Overall purchase conversion (saves might replace immediate purchases), cart abandonment rate (if users save instead of buying because checkout is annoying, you've diagnosed a different problem), session time (if saves are causing users to disengage).

**Step 5 — Segments**: Is the effect heterogeneous? New vs. returning users likely have different behavior (returning users may use saves as a wish list; new users may be exploring). Mobile vs. web may differ due to purchase intent signals. High-value vs. low-value items (saves on expensive items have different conversion dynamics).

**Step 6 — Trade-off articulation**: Short-term, saves may reduce immediate purchase conversion. Long-term, they may improve LTV by capturing intent that was previously lost. You should be explicit about this trade-off and know which time horizon the business cares about.

### Common traps

**Trap 1 — Picking an activity metric instead of an outcome metric.** "I'd track the number of saves" — this measures feature adoption, not value. A feature can be heavily used and useless (or harmful). Always trace back to what outcome the feature is meant to drive.

**Trap 2 — Not specifying a denominator.** "Purchase rate" means nothing without saying purchases per what — per user, per session, per saved item? The choice of denominator determines what question you're answering.

**Trap 3 — Ignoring cannibalization.** If saves go up and immediate purchases go down by an equal amount, the feature did nothing except delay revenue recognition. You need guardrails specifically for this.

**Trap 4 — Treating all users as one segment.** A feature might be highly valuable for one user type and neutral or harmful for another. Aggregate metrics hide this. If you only look at averages, you'll ship features that help your most-active users and hurt new users (or vice versa).

**Trap 5 — Confusing "the metric went up" with "the feature caused the metric to go up."** Without an experiment, correlation with time is not causation. A save feature launching during the holiday season will show amazing purchase numbers — but that's the holiday, not the feature.

---

## Metrics Investigation

### What the interviewer is actually testing

Can you systematically eliminate hypotheses, or do you jump to conclusions? A metric drop has many possible causes and most of them are NOT the interesting product story — they're data pipeline issues, bot traffic, or a change in a single user segment. The interviewer wants to see you don't waste engineering time chasing a fake signal or a product problem that's actually an instrumentation bug.

### The reasoning structure

The core principle: **before you explain *why* something happened, confirm *that* it happened.** This sounds trivially obvious, and yet the majority of candidates skip it. They hear "DAU dropped 20%" and immediately start theorizing about feature bugs or competitor launches. But roughly 30–40% of "metric drops" in real companies are instrumentation or pipeline issues, not real drops. Checking this first costs 5 minutes and can save 2 weeks of misdirected investigation.

The second principle: **scope before cause**. Once you know the drop is real, figure out *where* it is before figuring out *why*. A drop that is isolated to iOS users has a very different cause space than a drop that is global across all platforms. A drop that affects only new users has a different cause than one affecting returning users. You segment first because the shape of the drop tells you what hypotheses to form.

The third principle: **correlate with events before forming causal narratives**. What changed when the drop started? Product releases, infrastructure changes, marketing spend, external events (holidays, news, competitor actions). Timeline correlation is cheap and dramatically narrows your hypothesis space.

Only after these three steps do you form causal hypotheses and test them. At that point you have a much smaller, more targeted set of things to investigate.

### The pattern in action

**Question**: "DAU dropped 20% — how do you investigate?"

**Step 1 — Is it real?**
- Check data pipeline health: are events being logged and processed?
- Check the dashboard query: is there a filter applied that shouldn't be, or a date range error?
- Check multiple data sources: does the drop appear in the raw event table, not just the dashboard?
- Check if the drop is in raw counts or if the denominator changed (e.g., definition of "active" changed).

**Step 2 — What is the scope?**
- Platform: iOS / Android / web — is the drop isolated to one?
- Geography: one country / region, or global?
- User segment: new users vs. returning? Specific user cohort? Logged-in vs. anonymous?
- Feature area: is it tied to a specific surface (home feed, search, checkout)?
- Time of day: did it drop uniformly across all hours, or only during certain hours (could suggest timezone or batch processing issue)?

Each segmentation narrows the hypothesis space. If the drop is only on iOS, you look at an iOS app release or App Store change. If it's global, it's more systemic.

**Step 3 — When did it start exactly, and what else happened then?**
- Pull the timeline down to the hour if possible.
- Cross-reference with: product releases, infrastructure deploys, data pipeline changes, marketing campaign start/end, external events.

**Step 4 — Form and rank causal hypotheses**
Now, with the scope and timeline established, form explicit hypotheses ranked by prior probability:
1. Data/logging issue (high prior — very common)
2. A/B test or feature flag rollout affecting a segment
3. Product bug (especially if scoped to a platform or feature area)
4. External factor (holiday, news event, competitor)
5. Organic product quality degradation

For each hypothesis: what data would confirm or rule it out?

**Step 5 — Communicate what you'd do next**
Summarize what you found, what the leading hypothesis is, and what you'd need to verify it. If you haven't nailed the root cause yet, be explicit about why and what the next investigation steps are.

**Why this order matters**: if you skip Step 1 and the data is wrong, all subsequent investigation is wasted. If you skip Step 2 and the drop is scoped to one segment, you'll look for global explanations that don't exist. The structure is ordered by "cost to check" and "prior probability of being a dead end."

### Common traps

**Trap 1 — Starting with product hypotheses before checking data integrity.** Engineers hate being pulled into a debugging session for a metric drop that turns out to be a logging bug. Check the data first.

**Trap 2 — Not segmenting.** "DAU dropped 20%" — you start investigating, find nothing, and conclude "it just happened." Usually the drop is concentrated in one segment and global averages hide it.

**Trap 3 — Post-hoc story construction.** You find a product release that happened around the same time as the drop and declare it the cause. Correlation in time is not causation. Always ask: what would need to be true in the data to confirm this hypothesis?

**Trap 4 — Ignoring the possibility of a good cause.** Some DAU drops are caused by fixing bot traffic, removing spam users, or deduplicating accounts. Before you start fixing "the problem," check whether the drop represents a real loss of real users.

**Trap 5 — Declaring root cause before ruling out alternatives.** Good investigation explicitly states: "My leading hypothesis is X. Alternative explanations Y and Z are less likely because [data point]. To fully confirm X I'd need [additional check]."

---

## A/B Test Design

### What the interviewer is actually testing

Two things: (1) do you understand what question an experiment is actually answering, and (2) can you anticipate and neutralize the ways an experiment can give you the wrong answer?

An experiment has one purpose: to estimate the causal effect of a change by removing the confounding effect of everything else. The interviewer wants to see that you understand this at a mechanical level — not just that you know the vocabulary.

### The reasoning structure

An A/B test works because randomization makes the treatment and control groups identical in expectation on all variables except the one you're changing. This is the only way to isolate causality. If anything other than the treatment differs between groups, your estimate is biased.

From this principle, you can derive every design decision:

- **Unit of randomization** should be the unit that receives the treatment and generates the outcome. Usually this is the user. If you randomize at the page level but measure user-level retention, you've introduced contamination (one user can be in both groups).
- **Randomization must be independent across units.** If users in the same household share an account, or if a user can refer another user (network effects), then one user being in treatment affects another user's behavior. This violates independence.
- **Sample size** follows from the minimum effect size you care about, the baseline metric rate, and the acceptable error rates. There's no "standard" sample size — it depends on your specific context.
- **Duration** must account for novelty effects, day-of-week variation, and seasonal patterns. A 2-day test that runs over a weekend is not representative of a regular week.

### The pattern in action

**Question**: "Design an experiment to test whether adding a chatbot to the checkout page improves purchase conversion."

**Step 1 — Define the estimand.** What effect are we trying to measure? The causal effect of chatbot access (not usage, access — because you can't randomize who uses it, only who sees it) on checkout completion.

**Step 2 — Unit of randomization.** User ID. Why not session ID? A user could start a session in control and another in treatment — their behavior would be contaminated. Randomize at user level, analyze at user level.

**Step 3 — Control group.** No chatbot on checkout page. Treatment group: chatbot visible on checkout page. Make sure the only difference between the pages is the chatbot — don't A/B test two things at once unless you have a 2x2 factorial design.

**Step 4 — Primary metric.** Checkout completion rate (purchases / users who entered checkout). Secondary: order value (does chatbot help with upsell?), chatbot engagement rate (what fraction use it — useful for understanding mechanism even though you don't randomize on it).

**Step 5 — Guardrail metrics.** Overall session length (chatbot shouldn't be a distraction), support ticket volume (chatbot shouldn't replace human support in a bad way), cart abandonment rate at earlier funnel steps.

**Step 6 — Sample size calculation.** You need: current checkout completion rate (baseline), minimum detectable effect (what improvement is worth shipping — say 2%), desired power (0.8), significance level (0.05). From these, calculate n per group. Do not start the experiment without this calculation or you'll either underpower it and miss real effects, or run it too long and waste time.

**Step 7 — Duration.** Run for at least 1–2 full weeks to capture day-of-week variation. Check if there are upcoming seasonal events that could confound results (holidays, promotions). Avoid running through a product launch that changes baseline behavior.

**Step 8 — Threats to validity.** 
- Network effects: does one user's chatbot use affect another's experience? (Probably no, in this case — the chatbot is user-specific.)
- Novelty effect: users might engage with the chatbot just because it's new. Consider running for long enough that novelty wears off, or track whether effect degrades over time.
- Heterogeneous effects: the chatbot might help new users (who have questions) and be irrelevant to returning users (who know what they want). Plan to segment.

### Common traps

**Trap 1 — Randomizing at the wrong level.** The classic error: randomizing at session level or page level when the treatment and its effects span a user's entire session or multiple sessions. Network effects (social products, referrals) require cluster randomization at a higher unit.

**Trap 2 — Multiple testing without correction.** You test 10 metrics. Two show p < 0.05. You declare victory. But at 0.05 significance level with 10 independent tests, you expect 0.5 false positives — in practice, with correlated metrics, you'll have spurious signals. Either pre-register your primary metric and treat others as exploratory, or apply Bonferroni/Benjamini-Hochberg correction.

**Trap 3 — Running until significant.** You check the p-value every day and stop when it hits 0.05. This inflates Type I error dramatically — you're doing multiple comparisons in time. Pre-specify your sample size and run until you reach it, or use sequential testing methods if you need to peek.

**Trap 4 — Underpowering.** You run the experiment for 3 days with insufficient sample size, see p = 0.3, and conclude "no effect." You can't conclude no effect from a low-powered null result. You can only conclude "we can't detect an effect at this sample size." A null result from an underpowered test is uninformative.

**Trap 5 — Ignoring the novelty effect.** A new feature will often show inflated engagement in the first week as curious users explore it. If you stop the test after week 1, you may ship something that performs well only due to novelty. Check whether the effect persists over time.

---

## A/B Test Analysis and Interpretation

### What the interviewer is actually testing

Not: can you calculate a p-value. Yes: do you understand what a p-value means, what it does NOT mean, and how to make a correct ship/no-ship decision under uncertainty?

Most candidates can compute a two-sample t-test. The interviewers are testing whether you understand the inferential logic behind it — and whether you can correctly interpret a result that isn't a clean "significant" or "not significant."

### The reasoning structure

The p-value answers one specific question: *if the null hypothesis (no effect) were true, how likely would we be to observe data at least this extreme?* It does NOT tell you the probability that the null is true. It does NOT tell you the size of the effect. It does NOT tell you whether the effect is practically meaningful.

This distinction matters enormously in product settings. A test with 10 million users can produce p < 0.0001 for an effect of 0.01% — technically "significant," but not worth shipping if your engineering/maintenance cost exceeds the benefit of that 0.01% lift.

The correct analytical sequence: (1) check for data quality issues, (2) check randomization worked (SRM check), (3) compute the estimate and its confidence interval, (4) interpret practical significance, not just statistical significance, (5) look for heterogeneous effects, (6) make a decision.

**Sample Ratio Mismatch (SRM)**: before analyzing any experiment results, check that the actual split ratio matches the intended split ratio. If you expected 50/50 and got 48/52, something went wrong with the randomization — users may be self-selecting into groups, the assignment may be buggy, or there's a logging issue. An SRM invalidates the experiment because randomization is no longer guaranteeing exchangeability between groups.

### The pattern in action

**Question**: "Your A/B test ran for 2 weeks. Treatment shows a 1.5% lift in conversion with p = 0.03. Do you ship it?"

**Step 1 — Data quality check.**
- Did the experiment run cleanly? Were there any infrastructure changes or test overlaps during the run?
- Check the SRM: intended 50/50 split, actual split 49.8/50.2 — within noise, no SRM. If you had 48/52, flag this.

**Step 2 — Statistical interpretation.**
- p = 0.03 means: under the null, there's a 3% chance of observing a 1.5% or greater lift by chance. This is below α = 0.05, so you reject the null.
- But state the confidence interval: say it's [+0.2%, +2.8%]. The true effect is somewhere in here with 95% confidence. The *minimum* plausible effect is +0.2% — that's the lower bound you should use for the business case.

**Step 3 — Practical significance.**
- Is a 1.5% lift in conversion worth shipping? Depends on: what is conversion rate × revenue × volume? If this is on a surface that drives $100M/year, a 1.5% lift is $1.5M. If it's on a surface that drives $1M/year, it's $15K. The threshold for shipping depends on engineering cost and opportunity cost.

**Step 4 — Guardrail metrics.**
- Did session length, support tickets, or other guardrails degrade? If a guardrail moved significantly in the wrong direction, you don't ship even if the primary metric is positive — you investigate the trade-off.

**Step 5 — Heterogeneous effects.**
- Break by new vs. returning users, platform, region. If the lift is +5% for new users and −1% for returning users, you have a more nuanced decision: maybe ship only for new users, or investigate why it hurt returning users.

**Step 6 — Decision.**
- Ship, ship with monitoring, roll back, or run follow-up experiment? Be explicit about which and why.

### Common traps

**Trap 1 — Treating p < 0.05 as the end of the analysis.** Statistical significance is the starting point, not the conclusion. You still need to evaluate practical significance, check guardrails, and look for heterogeneity.

**Trap 2 — Treating p > 0.05 as "no effect."** A null result from a properly powered test means you can rule out effects larger than your MDE. It does NOT mean no effect exists. It means you cannot detect one at this power level. The distinction matters when deciding whether to invest in more data collection.

**Trap 3 — Reporting the point estimate as the "true effect."** The point estimate (1.5% lift) is your best guess. The confidence interval is what you should be reasoning from. If the CI is [+0.1%, +3.0%], the lower bound should inform your business case conservatively.

**Trap 4 — Ignoring SRM.** An SRM means randomization failed. Analyzing an experiment with SRM as if it were valid produces biased estimates. The correct action is to investigate and fix the SRM before trusting results, or declare the experiment invalid.

**Trap 5 — Confusing intent-to-treat with per-protocol analysis.** Your randomized groups are "saw the treatment" and "saw the control." You should analyze based on assignment, not on whether the user actually used the feature (per-protocol). Switching to per-protocol analysis introduces selection bias — users who chose to engage with the feature are not representative of all assigned users.

---

## SQL Patterns

### What the interviewer is actually testing

Not: can you memorize syntax. Yes: do you understand the *logical structure* of data problems well enough to translate them into set operations and aggregations?

SQL questions test whether you understand: what is a join actually doing (set intersection / cross product), what does a window function do that a GROUP BY doesn't, how do you handle duplicates correctly, and how do you think about the shape of the output before writing the query.

### The reasoning structure

Before writing any SQL, do two things:
1. **Describe the shape of the output**: what are the rows, what are the columns, what is the grain?
2. **Work backward from output to input**: given this desired output grain, what joins and aggregations do I need?

This prevents the most common SQL error: joining tables at different grains without understanding the result, producing inadvertent row duplication.

**Join semantics from first principles**: a JOIN is a filtered cross product. INNER JOIN keeps rows where the key exists in both tables. LEFT JOIN keeps all rows from the left table, filling NULL for unmatched right-side columns. The key insight is that if there are multiple rows in the right table for a single row in the left table, LEFT JOIN multiplies the left rows — this is often a bug and almost always unintended.

**Window functions from first principles**: a window function applies a computation across a set of rows *without collapsing them*. GROUP BY collapses — you lose the original rows. OVER() preserves them. This is why window functions are essential for: computing a running total while keeping each row, ranking rows within a group, computing the difference between a row and its group average, or finding the first/last event in a sequence.

### The pattern in action

**Pattern 1 — Funnel / conversion**

Problem: given a table of user events with event_type, calculate the conversion rate from step A to step B to step C.

```sql
-- First, get per-user: did they complete each step?
WITH steps AS (
  SELECT
    user_id,
    MAX(CASE WHEN event_type = 'step_a' THEN 1 ELSE 0 END) AS did_a,
    MAX(CASE WHEN event_type = 'step_b' THEN 1 ELSE 0 END) AS did_b,
    MAX(CASE WHEN event_type = 'step_c' THEN 1 ELSE 0 END) AS did_c
  FROM events
  GROUP BY user_id
)
SELECT
  SUM(did_a) AS users_at_a,
  SUM(did_b) AS users_at_b,
  SUM(did_c) AS users_at_c,
  1.0 * SUM(did_b) / NULLIF(SUM(did_a), 0) AS a_to_b_conversion,
  1.0 * SUM(did_c) / NULLIF(SUM(did_b), 0) AS b_to_c_conversion
FROM steps;
```

Why this structure: collapsing to user-level first (the per-user MAX) gives you one row per user. Then aggregating gives you counts across users. If you tried to compute this directly from event rows, you'd need to be careful about counting distinct users at each step — the MAX approach is clean and explicit.

**Pattern 2 — Retention**

Problem: compute D7 retention for each cohort of users by signup week.

```sql
WITH cohorts AS (
  SELECT user_id, DATE_TRUNC('week', signup_date) AS cohort_week
  FROM users
),
activity AS (
  SELECT DISTINCT user_id, DATE_TRUNC('day', event_date) AS active_day
  FROM events
)
SELECT
  c.cohort_week,
  COUNT(DISTINCT c.user_id) AS cohort_size,
  COUNT(DISTINCT CASE WHEN a.active_day = c.cohort_week + INTERVAL '7 days'
                      THEN a.user_id END) AS retained_d7,
  1.0 * COUNT(DISTINCT CASE WHEN a.active_day = c.cohort_week + INTERVAL '7 days'
                             THEN a.user_id END)
        / COUNT(DISTINCT c.user_id) AS d7_retention
FROM cohorts c
LEFT JOIN activity a ON c.user_id = a.user_id
GROUP BY c.cohort_week
ORDER BY c.cohort_week;
```

Why LEFT JOIN here: you want all cohort users in the denominator, including those who did not return. INNER JOIN would exclude them and inflate retention.

**Pattern 3 — Window functions for rank-within-group**

Problem: for each user, find their most recent purchase.

```sql
WITH ranked AS (
  SELECT
    user_id,
    order_id,
    purchase_date,
    order_value,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY purchase_date DESC) AS rn
  FROM orders
)
SELECT user_id, order_id, purchase_date, order_value
FROM ranked
WHERE rn = 1;
```

Why ROW_NUMBER and not MAX: MAX(purchase_date) gets you the date but not the other columns (order_id, order_value) for that row. The window function pattern — rank within partition, then filter to rank = 1 — is the general solution for "get the full row of the most recent / largest / first event per entity."

**Pattern 4 — Deduplication**

Problem: a table has duplicate user rows from multiple data sources. Deduplicate by user_id, keeping the row with the most recent updated_at.

```sql
WITH deduped AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC) AS rn
  FROM raw_users
)
SELECT * EXCEPT (rn)  -- or SELECT col1, col2, col3
FROM deduped
WHERE rn = 1;
```

The reasoning is the same as Pattern 3. ROW_NUMBER() over a partition gives you a rank within that partition; taking rn = 1 gives you exactly one row per partition key.

### Common traps

**Trap 1 — Fan-out from joins.** If you join users to orders (one user, many orders) and then try to count users, you'll over-count unless you use COUNT(DISTINCT user_id). Better: understand the grain mismatch and aggregate appropriately before or after the join.

**Trap 2 — NULL handling.** NULL in SQL is not zero. COUNT(column) excludes NULLs. COUNT(*) includes them. SUM(NULL) = NULL, not 0 — use COALESCE when needed. Division by zero: use NULLIF(denominator, 0) to return NULL instead of an error when the denominator is zero.

**Trap 3 — Using HAVING vs WHERE incorrectly.** WHERE filters rows before aggregation. HAVING filters aggregated results. If you write WHERE SUM(amount) > 100, you get an error. You need HAVING SUM(amount) > 100.

**Trap 4 — Not thinking about the output grain before writing the query.** Write down "one row per user per week" before you write any SQL. If you don't know the grain you want, you'll write a query and not know if the result is correct.

**Trap 5 — Using subqueries where CTEs are clearer.** Both work. But nested subqueries are harder to debug. Prefer CTEs (WITH clauses) because you can run each stage independently to verify its output.

---

## Statistics Questions

### What the interviewer is actually testing

Not: can you state the definition of a p-value. Yes: do you understand the underlying logic of statistical inference well enough to reason about *which* test to use, *when* statistical results can mislead you, and *how* to communicate uncertainty correctly to non-technical stakeholders?

### The reasoning structure

Statistical inference has one goal: make claims about a population based on a sample, while being honest about uncertainty. Every concept follows from this:

- **p-value**: if the null were true, how often would I see data this extreme? Used to decide whether to reject the null. Does NOT measure the probability the null is true.
- **Confidence interval**: a range that, if you repeated the experiment many times, would contain the true parameter 95% of the time. Used to communicate the uncertainty in your estimate.
- **Power**: probability of correctly detecting an effect of a given size if it exists. Determined by sample size, effect size, and significance threshold. Low power → high false negative rate.
- **Central Limit Theorem**: the sampling distribution of the mean converges to normal as n grows, regardless of the underlying distribution. This is why t-tests and z-tests work on non-normal data when n is large enough.

**Choosing which test to use**: the choice follows from the structure of your data:
- Comparing two means (continuous outcome, two groups): two-sample t-test (or Mann-Whitney U if distribution is highly non-normal at small n).
- Comparing proportions (binary outcome, two groups): two-proportion z-test or chi-squared test.
- Comparing more than two groups: ANOVA (then post-hoc tests if significant, with multiple comparison correction).
- Paired data (same user before/after): paired t-test, not independent-samples t-test — the pairing removes between-subject variance and increases power.
- Count data with small expected counts: Fisher's exact test, not chi-squared.

### The pattern in action

**Question**: "What is a p-value? Explain it to a non-technical stakeholder."

**First-principles answer first**: a p-value is the probability of observing data at least as extreme as what you observed, *assuming the null hypothesis is true*. It is NOT the probability that your hypothesis is correct, and NOT the probability that the effect is real.

**Translation for a stakeholder**: "Imagine the new feature actually had zero effect. If we ran this experiment 100 times under that assumption, we'd see a result this large just by chance about 3 times out of 100. So the result is unlikely to be a fluke — but 'unlikely' is not 'impossible,' and it's not proof of anything."

**Question**: "When should you use a t-test vs. a non-parametric test?"

A t-test assumes the sampling distribution of the mean is approximately normal. By the CLT, this is generally satisfied when n is large (n > 30 is a rough rule, but it depends on how skewed the distribution is). For small samples from highly skewed or heavy-tailed distributions, the CLT approximation is poor and a non-parametric test (Mann-Whitney U) makes fewer assumptions.

In practice, for large-scale A/B tests (thousands of users), the t-test is almost always fine because the CLT kicks in. For clinical trials with n = 15 per arm, distribution shape matters and you should consider non-parametric alternatives.

**Question**: "Your experiment shows p = 0.04 and the lift is 0.1%. Do you ship?"

The p-value tells you the result is statistically unlikely under the null. But: a 0.1% lift may be practically meaningless — less than noise in your measurement, less than the cost of code maintenance, less than the variance in your baseline. Practical significance is separate from statistical significance. At large sample sizes, you can detect effects too small to matter. The right question is: does this 0.1% lift justify the cost of shipping and maintaining this feature?

### Common traps

**Trap 1 — "p < 0.05 means the result is real."** This is the most common and most costly misinterpretation. p < 0.05 means: if the null were true, we'd see this by chance 5% of the time. That's not the same as "the effect is real." With 100 tests, you expect 5 false positives even if all nulls are true.

**Trap 2 — "The p-value is the probability the null is true."** No. This would be a Bayesian posterior, which requires a prior. The p-value is a frequentist concept — it doesn't tell you anything about the probability of the hypothesis.

**Trap 3 — Not reporting confidence intervals.** A p-value alone conveys significance but not magnitude. Always report the effect size and its confidence interval. A CI of [+0.01%, +0.19%] tells a very different story than [+0.8%, +2.2%] even if both are p < 0.05.

**Trap 4 — Using the wrong test for paired data.** If you're comparing the same users before and after (or matched pairs), an independent-samples t-test ignores the correlation structure and loses power. A paired t-test is more powerful because it removes between-subject variance.

**Trap 5 — Conflating statistical significance with scientific or practical importance.** Statistical significance is a property of your sample and sample size. Practical importance is a property of the real world. These are independent. Large samples can make trivially small effects significant; small samples can make practically large effects non-significant.

---

## Causal Inference

### What the interviewer is actually testing

Can you reason correctly about what your data can and cannot tell you about cause and effect? Most observational data is confounded — users who do X are systematically different from users who don't, and that difference may explain the outcome, not the X itself.

The interviewer wants to see: you know when A/B testing is not possible, you know what assumptions are needed for observational methods, and you understand the key concept of confounding.

### The reasoning structure

Causality requires three things: (1) association between cause and effect, (2) temporal precedence (cause comes before effect), (3) elimination of alternative explanations (confounders). Observational data typically establishes (1) and sometimes (2) but struggles with (3).

A **confounder** is a variable that affects both the treatment (the thing you're studying) and the outcome. If you want to know whether exercise causes lower blood pressure, but people who exercise also tend to eat better, then "eating better" is a confounder — it explains part of the observed association between exercise and blood pressure. Without controlling for it, your estimate is biased.

The fundamental problem of causal inference: for each user, you can observe either the outcome with treatment or without treatment — never both. The causal effect is the difference between these two potential outcomes. Since you can't observe both, you need a method that constructs a credible counterfactual.

A/B testing solves this by randomization: on average, the control group *is* the counterfactual for the treatment group because randomization makes them statistically identical at baseline. When you can't randomize, you need other methods.

**Difference-in-Differences (DiD)**: compares the change over time in a treated group to the change over time in a control group. Assumes the two groups would have had parallel trends in the absence of treatment. Good for: policy changes, market launches, natural experiments where treatment happens to one group but not another.

**Instrumental Variables (IV)**: uses an instrument — a variable that affects treatment assignment but affects the outcome *only through* the treatment. This breaks the endogeneity between treatment and outcome. Hard to find valid instruments in practice.

**Regression Discontinuity (RD)**: exploits a sharp cutoff in treatment assignment (e.g., users above a credit score threshold get a feature). Users just above and just below the cutoff are assumed to be similar except for treatment status. Estimates the local average treatment effect at the cutoff.

**Propensity Score Matching**: model the probability of treatment given observed covariates (the propensity score), then match treated and control units with similar propensity scores. This controls for observed confounders but cannot control for *unobserved* confounders.

### The pattern in action

**Question**: "We launched a new loyalty program in one city. How do you estimate its effect on revenue?"

You can't randomize — the program was rolled out to an entire city. How do you estimate the causal effect?

**DiD approach:**
- Find a comparable city that did not get the loyalty program (the "control city").
- Measure revenue in both cities before and after the launch.
- The DiD estimate: (revenue change in treatment city) − (revenue change in control city).
- This removes time-invariant city-specific factors (the "difference" across cities) and common time trends (the "difference" across time periods).
- Key assumption: **parallel trends** — in the absence of the loyalty program, the two cities would have experienced the same revenue trend. This is untestable but you can provide evidence by showing pre-treatment trends were similar.

**What makes this assumption suspicious**: if the treatment city was selected *because* it was performing badly (regression to the mean will make it look like the program worked), or if there are other simultaneous differences between the cities, the parallel trends assumption fails.

**Question**: "Heavy users of a feature have 40% higher retention. Does the feature cause retention?"

This is correlation in observational data. Heavy users are different from light users along many dimensions: they may be more engaged overall, have more invested in the product, or have used the product for longer. Any of these could explain higher retention independent of the feature.

To estimate causal effect, you'd want: (1) a randomized experiment forcing some users to see vs. not see the feature, or (2) an IV — something that shifts feature usage for some users but otherwise shouldn't affect retention, or (3) DiD if there was a time when the feature was available in some markets but not others.

Simply adjusting for observed user characteristics (regression) controls for measured confounders but leaves unobserved confounders untouched.

### Common traps

**Trap 1 — Confusing correlation with causation in observational data.** The most common failure mode. "Users who use feature X have higher LTV" — this is a selection effect, not a causal claim. Users who use more features are better users. You haven't established that X causes LTV, only that X and LTV are correlated (and both are likely caused by user engagement level).

**Trap 2 — Believing that controlling for observables removes all confounding.** It doesn't. Regression on observed covariates removes the confounding from those variables. Unobserved confounders remain. The only way to deal with unobserved confounding is randomization, a valid instrument, or a natural experiment design.

**Trap 3 — Forgetting the parallel trends assumption in DiD.** DiD is not magic. It requires that the treatment and control groups would have trended similarly in the absence of treatment. Always show pre-treatment trend comparison and discuss whether the assumption is plausible.

**Trap 4 — Overfitting the propensity score.** If you include too many variables in your propensity model, you match treated and control units so well on observables that you destroy statistical power. There's a bias-variance trade-off in observational methods.

**Trap 5 — Using pre-post comparison without a control group.** "Revenue went up 15% after we launched the feature." Maybe the whole market went up 15%. Without a control group, you can't separate the effect of your intervention from external trends.

---

## Python and Pandas Patterns

### What the interviewer is actually testing

Not: do you know the pandas API. Yes: do you understand data manipulation at the conceptual level — specifically: groupby-apply semantics, merge behavior, vectorization vs. iteration, and how to handle missing data correctly?

### The reasoning structure

Pandas is built on two core concepts: (1) every operation on a DataFrame or Series produces a new object (immutability by default), and (2) operations are vectorized — they operate on arrays, not row by row. Violating the second principle (using Python loops over rows) is the most common performance mistake.

**Groupby semantics**: `df.groupby('key').agg(...)` is conceptually the same as SQL GROUP BY — split the DataFrame by key, apply a function to each group, combine results. The key insight is what the function receives: each group is a sub-DataFrame. This means you can pass any function that takes a DataFrame and returns a scalar, Series, or DataFrame — enabling complex per-group computations.

**Merge semantics**: same as SQL joins. The `how` parameter specifies inner/left/right/outer. Critical: if the right key is not unique (one-to-many), the merge will produce multiple rows for each left row — this is often unintended. Always check the cardinality of your merge keys and the shape of the result.

**Missing data**: NaN propagates through arithmetic operations. `df['col'].mean()` skips NaN by default. `df['col'].sum()` skips NaN by default. But `df['col'].fillna(0)` before an operation changes the semantics — filling NaN with 0 before computing a mean changes the denominator. Know what you want before you fill.

### The pattern in action

**Pattern 1 — Groupby with custom aggregation**

```python
# Per-user: first purchase date, total spend, number of orders
user_stats = df.groupby('user_id').agg(
    first_purchase=('purchase_date', 'min'),
    total_spend=('order_value', 'sum'),
    order_count=('order_id', 'count')
).reset_index()
```

Why `.reset_index()`: after groupby, user_id becomes the index. reset_index converts it back to a regular column, which is usually what you want for subsequent merges or filtering.

**Pattern 2 — Window function equivalent (rolling, cumulative)**

```python
# Running total spend per user, ordered by date
df = df.sort_values(['user_id', 'purchase_date'])
df['cumulative_spend'] = df.groupby('user_id')['order_value'].cumsum()
```

This is equivalent to `SUM(order_value) OVER (PARTITION BY user_id ORDER BY purchase_date)` in SQL. Key: sort first, then apply the window function.

**Pattern 3 — Applying a function to each group**

```python
# For each user, flag whether their spend in each order is above their own median
def flag_above_median(group):
    median = group['order_value'].median()
    group['above_median'] = group['order_value'] > median
    return group

df = df.groupby('user_id', group_keys=False).apply(flag_above_median)
```

When to use `apply` vs. vectorized operations: `apply` is slower but handles cases where the per-group computation cannot be expressed as a built-in aggregation. For simple aggregations (mean, sum, count), use `agg` or direct method calls — they're faster.

**Pattern 4 — Merging with cardinality awareness**

```python
# Before merging, check uniqueness of keys
assert users['user_id'].nunique() == len(users), "Duplicate user_ids in users table"
orders_with_user = orders.merge(users[['user_id', 'region']], on='user_id', how='left')
# After merge, verify shape
assert len(orders_with_user) == len(orders), "Unexpected row multiplication in merge"
```

Always validate merge shapes when the result matters. A one-to-many merge from a table you thought was one-to-one will silently multiply your rows.

**Pattern 5 — Avoiding loops with vectorized operations**

```python
# WRONG: row-by-row iteration (slow)
for i, row in df.iterrows():
    if row['status'] == 'active':
        df.at[i, 'label'] = 'keep'
    else:
        df.at[i, 'label'] = 'drop'

# RIGHT: vectorized conditional
df['label'] = np.where(df['status'] == 'active', 'keep', 'drop')
```

Why: `iterrows` is O(n) Python loop overhead. Vectorized operations use C-level array operations — typically 100x faster for large DataFrames.

### Common traps

**Trap 1 — Chained indexing / SettingWithCopyWarning.** `df[df['col'] > 0]['new_col'] = 1` does NOT reliably modify the original DataFrame. Use `.loc`: `df.loc[df['col'] > 0, 'new_col'] = 1`.

**Trap 2 — Forgetting that groupby aggregation with 'count' counts non-null values.** If you want to count rows in a group regardless of NaN, use `'size'` (which counts all rows) instead of `'count'` (which counts non-null values for that column).

**Trap 3 — Implicit index alignment in operations.** When you do arithmetic between two Series, pandas aligns on index. If the indices don't match, you get NaNs. Use `.values` to strip index before arithmetic if you don't want alignment behavior.

**Trap 4 — Modifying a DataFrame inside groupby apply without returning it.** The function passed to `apply` must return the modified group. If you modify in place without returning, the result won't propagate.

**Trap 5 — Not resetting index after groupby.** After `groupby().agg()`, the group keys become the index. Subsequent merges on those keys will fail or behave unexpectedly if you forget to `reset_index()`.

---

## Behavioral Questions (STAR)

### What the interviewer is actually testing

Not: can you tell a good story. Yes: do you operate with autonomy, ownership, and sound judgment under ambiguity — and can you demonstrate this through specific past behavior?

Behavioral questions are predictive proxies for future behavior. The interviewer is not trying to evaluate your storytelling — they're looking for evidence that you have done the thing they need you to do in this role.

### The reasoning structure

STAR (Situation, Task, Action, Result) is the correct structure because it makes the story falsifiable and specific: interviewers can ask follow-up questions about the actual situation, and vague or rehearsed answers fall apart under drilling.

The most common failure mode is answering with "we" throughout — it's unclear what *you* specifically did. The second failure mode is spending too long on Situation and Task (context) and not enough on Action (what you personally did and why) and Result (what changed).

From first principles: the interviewer is asking "have you done this before?" Your job is to give them the most specific, evidence-rich answer that answers "yes." That means: a real situation (named, dated, with stakes), a real action (specific decisions *you* made, with your reasoning), and a real result (quantified, attributed correctly to your contribution).

The quality of your answer is determined by the specificity and credibility of the Action and Result. Anyone can describe a situation. What distinguishes candidates is what they actually *did* — the judgment calls, the trade-offs they made, the things they sacrificed, the things that didn't work and what they learned.

### The pattern in action

**Question**: "Tell me about a time you had to make a decision with incomplete data."

**Weak answer (common)**: "I was working on a project where we didn't have all the data we needed. I gathered what data I could and made a recommendation to the team. We went with it and it worked out."

This says nothing. It's unfalsifiable. Every DS has done this. The interviewer can't evaluate you from this.

**Strong answer (first principles)**:
- **Situation**: "In Q3 last year I was analyzing whether to sunset a feature that had low engagement but strong retention correlation among a small user segment."
- **Task**: "I had 3 weeks and access to behavioral logs, but no survey data and no direct causality established — I didn't know if low engagement meant users hated the feature or used it passively."
- **Action**: "I decided to: (1) estimate the upper-bound value of the segment using cohort LTV data — if that segment churned at the rate of other low-engagement users, the revenue risk was ~$X; (2) run a 10-user usability session to get qualitative signal quickly; (3) explicitly present the recommendation as 'confident to within the bounds of this uncertainty' and recommended a 6-month staged deprecation with exit survey data collection built in, rather than an immediate sunset."
- **Result**: "The staged deprecation was approved. The exit surveys 3 months in showed that 20% of the segment used the feature as a passive 'save' function they valued but rarely clicked. We redesigned it instead of sunsetting it and saw +4% retention in that segment."

What makes this strong: the action section shows specific reasoning under uncertainty, explicit acknowledgment of what you didn't know, and a decision that was designed to *reduce* the cost of being wrong. The result is specific and measured.

### Common traps

**Trap 1 — Using "we" for everything.** "We decided to..." "We ran the analysis..." The interviewer needs to know what *you* did. Be specific about your personal contributions: "I built the model, while my teammate handled the data pipeline."

**Trap 2 — Describing ideal process instead of what you actually did.** "In this kind of situation I would normally..." — the question asked for something you actually did, not what you theoretically do. Past tense, specific, real.

**Trap 3 — Results without numbers.** "The project was successful." Successful how? By how much? For which stakeholder? Quantify where possible; if you can't, give a qualitative calibration ("the VP of Product ended up citing this analysis in the quarterly all-hands as the basis for the roadmap decision").

**Trap 4 — Stories where you had no agency.** If the decision was made by your manager and you executed it, that's not a useful story for most behavioral questions. Find examples where you had genuine ownership of a decision, even if the scale was small.

**Trap 5 — No tension or learning.** The best behavioral stories have a moment where something was uncertain, difficult, or where you made a choice that involved a trade-off. Stories with no tension are flat. Stories where you learned something from a partial failure or adjustment are often more compelling than stories where everything went perfectly.

---

## Case Study Structure

### What the interviewer is actually testing

Can you take an ambiguous, open-ended problem and impose structure on it — without losing sight of the actual business question? A case study interview is not about finding the "right answer." It's about demonstrating that your process for exploring an ambiguous problem is sound.

### The reasoning structure

The key mistake candidates make in case studies is starting with analysis before defining the question. You need to know what you're trying to answer before you look at data — otherwise you're doing exploratory data analysis without purpose, which is slow and unfocused.

The correct sequence:
1. **Clarify the business question**: what decision does this analysis inform? Who is the decision-maker? What would they do differently based on your findings?
2. **Define the success criteria**: what would "this analysis is complete and useful" look like?
3. **Hypothesize before looking at data**: what do you expect to find, and why? This forces you to be explicit about your prior beliefs, which you can then update with data.
4. **Data exploration with purpose**: look at the data to test your hypotheses, not to "see what's interesting."
5. **Communicate uncertainty**: what can you conclude confidently, what remains uncertain, and what additional data or analysis would change your conclusions?

This sequence is important because it's how rigorous analysis works in practice. Exploratory analysis without hypothesis discipline leads to p-hacking and confirmation bias — you'll find patterns that aren't real because you're looking for *something* interesting.

### The pattern in action

**Question**: "You have 6 months of ride-sharing transaction data. Explore it and tell us something interesting."

**Wrong approach**: open the data, run correlations, look at time series, run clustering, report whatever has a high correlation coefficient. This produces noise as signal and doesn't answer anything.

**Right approach**:

**Step 1 — Define the question space.** Before touching the data: what questions would be useful to answer for a ride-sharing business? Growth (are trips increasing?), quality (are ratings improving?), economics (what drives fare per trip?), churn (which drivers/riders are leaving?), efficiency (are rides matching supply and demand well?). Pick one or two to anchor on.

**Step 2 — Form hypotheses.** "I expect surge pricing to be correlated with driver supply constraints during certain hours. I expect rider ratings to correlate with trip duration and time of day." Having hypotheses makes your analysis directional.

**Step 3 — Exploratory analysis to test hypotheses.** Look at surge pricing by hour of day vs. driver supply. Look at rating distributions by trip length quartile. Now you're answering questions, not just describing data.

**Step 4 — Find one interesting insight with a recommended action.** "Surge pricing peaks between 10pm–2am Friday and Saturday, but driver supply is actually highest during those hours — yet trips are under-supplied. This suggests there's a geographic mismatch: drivers are concentrated in suburban areas during late-night hours when demand is highest in the city center. Recommendation: test a driver incentive for city-center positioning during late-night weekend hours."

This is a good case study answer because it: is specific, surprising, grounded in data, has a proposed action, and suggests a follow-up experiment.

**Step 5 — State limitations.** "This analysis is based on 6 months of data from one city. It doesn't account for seasonality beyond that window, doesn't include driver-reported reasons for positioning, and the supply/demand mismatch hypothesis would need to be confirmed with GPS density data."

### Common traps

**Trap 1 — Starting with analysis before clarifying the question.** You spend 20 minutes on EDA and produce a scatter plot that is beautiful but irrelevant. Always ask: what decision does this inform?

**Trap 2 — Not stating hypotheses before looking at data.** If you form hypotheses after looking at the data, you're data dredging. The hypotheses should come from first principles or domain knowledge.

**Trap 3 — Overconfident conclusions from observational data.** "Drivers who work more hours earn more — so we should require minimum hours." Drivers who work more hours earn more because they're more motivated — requiring hours would hurt motivated drivers and not change unmotivated ones. Correlation ≠ causation, and policies based on observed correlations often backfire.

**Trap 4 — Reporting everything you found.** A case study is not a data dump. It's a communication task. Present the one or two most important findings with the clearest action implications. Everything else is supporting material or backup.

**Trap 5 — Not communicating uncertainty.** Every finding comes with caveats about sample size, potential confounders, and what additional data would change the conclusion. Presenting findings as certain when they're not is the fastest way to lose credibility with a technical interviewer.

---

## Universal Reasoning Traps

These are the fundamental errors in data reasoning. They appear in every interview domain. Being able to name them when you see them — and avoid them in your own analysis — is a key differentiator.

---

### Trap: Correlation vs. Causation

**What it is**: inferring that because A is correlated with B, A causes B.

**Why it happens**: humans are pattern-recognition machines. We see a relationship and immediately construct a causal narrative. This is fast and often useful, but systematically wrong in the presence of confounders.

**Why it matters**: decisions made based on spurious correlations waste resources at best, cause harm at worst. If you see that "premium tier users have 3x LTV" and interpret this as "the premium tier causes higher LTV," you'll over-invest in forcing users to upgrade rather than in attracting the kind of users who self-select into premium (who were already going to have high LTV).

**First-principles defense**: every time you see a correlation, ask: (1) what common causes could explain both variables? (2) could the direction be reversed (reverse causation)? (3) is there a third variable that explains both? Without an experimental design or a credible identification strategy, you cannot make a causal claim from observational correlation.

---

### Trap: Selection Bias

**What it is**: drawing conclusions from a sample that is systematically different from the population you care about.

**Why it happens**: we analyze the data we have, not the data we need. The data we have is rarely a random sample of the full population.

**Classic examples**: 
- Survivorship bias: analyzing only the companies that survived to ask what made them successful, ignoring all the equally-positioned companies that failed.
- Sampling bias in A/B tests: if early users of a new feature are self-selected enthusiasts, their behavior doesn't predict how the full user base will respond when it's rolled out.
- Non-response bias in surveys: users who respond to a satisfaction survey are not representative of all users — disengaged or dissatisfied users are less likely to respond.

**First-principles defense**: always ask: who is in my sample, and who is not? Would the missing cases systematically differ from the present cases in a way that would change my conclusion?

---

### Trap: Statistical Significance vs. Practical Significance

**What it is**: concluding that a result is meaningful because it is statistically significant, without considering whether the effect size is large enough to matter.

**Why it happens**: p-values are the default output of statistical tests, and the < 0.05 threshold is a proxy for "real." But with large enough sample sizes, trivially small effects become statistically significant.

**Why it matters**: you can "prove" with p < 0.0001 that a feature change increases session time by 2 seconds. Is 2 seconds worth the engineering cost? Is it even detectable by users? Statistical significance says nothing about this.

**First-principles defense**: always pair a p-value with a confidence interval and a business interpretation. Ask: "if the lower bound of the confidence interval is the true effect, does it still justify shipping this?"

---

### Trap: Base Rate Neglect

**What it is**: ignoring the prior probability of an event when interpreting evidence.

**Classic example (Bayes)**: a medical test has 99% sensitivity (true positive rate) and 99% specificity (true negative rate). You test positive. What's the probability you have the disease?

Most people say ~99%. The correct answer depends on the base rate. If the disease affects 0.1% of the population, then out of 100,000 people: 100 have the disease, 99 test positive (true positives). 99,900 don't have it, 999 test positive (false positives). So P(disease | positive test) = 99 / (99 + 999) ≈ 9%. The low base rate dominates.

**Why it matters in DS**: if you build a fraud detection model with 99% precision and 99% recall, but fraud affects 0.01% of transactions, you still get a lot of false positives in absolute numbers. Evaluate your model against the base rate, not just against itself.

**First-principles defense**: always ask what the base rate is before interpreting a conditional probability or classifier output. Precision and recall without base rate context can be misleading.

---

### Trap: Simpson's Paradox

**What it is**: an aggregate trend reverses when the data is segmented — a relationship that holds in the aggregate disappears or flips when you look within subgroups.

**Classic example**: Drug A has a higher overall recovery rate than Drug B. But for mild cases, Drug B is better. For severe cases, Drug B is better. Aggregate: Drug A looks better. Why? Drug A was disproportionately given to mild cases (the easier-to-treat group). The mix of case severity is a confounder.

**Why it matters in DS**: aggregate metrics can mislead you in exactly this way. "Feature X users have higher retention" — but if feature X is only surfaced to your most-engaged users, the higher retention is driven by user type, not the feature.

**First-principles defense**: segment before concluding. If you see an aggregate relationship that matters, always check whether it holds within meaningful subgroups. If it reverses, the subgroup composition is a confounder and the aggregate result is misleading.

---

### Trap: Novelty Effect

**What it is**: users engage with something new because it's new, not because it's better. The effect decays as novelty wears off.

**Why it matters**: an A/B test that runs for 1–2 weeks may capture only the novelty effect of a new feature. If the test ends there, you ship something that shows 10% lift in week 1 and −3% lift in week 8 (after novelty decays).

**First-principles defense**: for features that are inherently experience-based (design changes, new interaction patterns), run the experiment long enough for novelty to decay (typically 3–4 weeks). Alternatively, look at engagement trends over time within the treatment group — if engagement is declining toward control levels, you're seeing a novelty effect.

---

### Trap: Regression to the Mean

**What it is**: extreme observations tend to be followed by less extreme observations — not because anything changed, but because extreme values often contain a large noise component.

**Why it matters**: if you intervene on your worst-performing users/locations/products and see improvement, part of that improvement is regression to the mean, not your intervention. This inflates before-after comparisons without control groups.

**First-principles defense**: always use a control group for before-after comparisons. The control group will also regress to the mean, and the difference-in-differences estimate removes this artifact.

---

### Trap: Ignoring Multiple Comparisons

**What it is**: testing many hypotheses with the same significance threshold increases the probability of at least one false positive.

**Why it matters**: if you test 20 metrics at p < 0.05, you expect 1 false positive on average even if all nulls are true. If you then select and report the significant metric, you've committed the "winner's curse" — your reported effect is likely inflated and may not replicate.

**First-principles defense**: pre-register your primary metric and hypothesis before looking at data. For secondary metrics, apply multiple comparison corrections (Bonferroni for conservative control; Benjamini-Hochberg for FDR control). Be explicit about what is confirmatory and what is exploratory.

---

## Quick Reference: Choosing the Right Test

| Situation | Test |
|---|---|
| Two group means, continuous outcome | Two-sample t-test (large n) or Mann-Whitney U (small n, skewed) |
| Two group proportions | Two-proportion z-test or chi-squared |
| Paired data (before/after same units) | Paired t-test |
| More than two groups | ANOVA + post-hoc with correction |
| Count data, small expected counts | Fisher's exact test |
| Testing independence of two categorical variables | Chi-squared test of independence |
| Goodness of fit (observed vs. expected distribution) | Chi-squared goodness of fit |
| Survival / time-to-event data | Log-rank test, Cox proportional hazards |
| Sample ratio mismatch in an experiment | Chi-squared test on observed vs. expected group sizes |

---

## Quick Reference: Experiment Decision Tree

```
Is randomization possible?
├── YES → Run A/B test
│   ├── Can you randomize at user level?
│   │   ├── YES → Standard A/B test
│   │   └── NO (network effects, geographic) → Cluster randomization or geo-based test
│   └── Does the treatment affect users' exposure to each other?
│       ├── NO → Standard analysis
│       └── YES → SUTVA violation; use cluster-level analysis
└── NO (policy, legal, market-level)
    ├── Is there a sharp assignment cutoff? → Regression Discontinuity
    ├── Was treatment applied to one group but not another over time? → Difference-in-Differences
    ├── Is there a variable that shifts treatment but not outcome directly? → Instrumental Variables
    └── Can you match on observable characteristics? → Propensity Score Matching (note: only removes observed confounding)
```

---

## Interview Format by Company Type (Reference)

| Round Type | What's Tested | Common At |
|---|---|---|
| Product sense / metrics | Define success metrics, investigate drops, design measurement | Meta, Google, Airbnb, Uber |
| A/B testing / experiment design | Design experiment, power analysis, analyze results | All data-heavy companies |
| SQL | Window functions, funnel/retention/dedup queries | All |
| Statistics | p-value, CLT, CI, which test to use | All |
| Causal inference | Observational studies, DiD, when A/B isn't possible | Airbnb, Lyft, Netflix |
| Python/pandas | Data manipulation, groupby, merging, vectorized ops | ML-heavy DS roles |
| Case study / take-home | End-to-end analysis from raw data | Stripe, DoorDash, others |
| Behavioral (STAR) | Ambiguous problems, stakeholder conflicts, impact | All |

## Flashcards

**Hypothesis generation?** #flashcard
can you enumerate the possible causes of something before committing to one?

**Prioritization?** #flashcard
given many hypotheses, do you check the most likely / cheapest to disprove first?

**Self-awareness about uncertainty?** #flashcard
do you know what you don't know, and what additional data would change your answer?

**Calibration?** #flashcard
do you distinguish between "statistically significant" and "practically meaningful"?

**Check data pipeline health?** #flashcard
are events being logged and processed?

**Check the dashboard query?** #flashcard
is there a filter applied that shouldn't be, or a date range error?

**Check multiple data sources?** #flashcard
does the drop appear in the raw event table, not just the dashboard?

**Check if the drop is in raw counts or if the denominator changed (e.g., definition of "active" changed).?** #flashcard
Check if the drop is in raw counts or if the denominator changed (e.g., definition of "active" changed).

**Platform: iOS / Android / web?** #flashcard
is the drop isolated to one?

**Geography?** #flashcard
one country / region, or global?

**User segment?** #flashcard
new users vs. returning? Specific user cohort? Logged-in vs. anonymous?

**Feature area?** #flashcard
is it tied to a specific surface (home feed, search, checkout)?

**Time of day?** #flashcard
did it drop uniformly across all hours, or only during certain hours (could suggest timezone or batch processing issue)?

**Pull the timeline down to the hour if possible.?** #flashcard
Pull the timeline down to the hour if possible.

**Cross-reference with?** #flashcard
product releases, infrastructure deploys, data pipeline changes, marketing campaign start/end, external events.

**Unit of randomization should be the unit that receives the treatment and generates the outcome. Usually this is the user. If you randomize at the page level but measure user-level retention, you've introduced contamination (one user can be in both groups).?** #flashcard
Unit of randomization should be the unit that receives the treatment and generates the outcome. Usually this is the user. If you randomize at the page level but measure user-level retention, you've introduced contamination (one user can be in both groups).

**Randomization must be independent across units. If users in the same household share an account, or if a user can refer another user (network effects), then one user being in treatment affects another user's behavior. This violates independence.?** #flashcard
Randomization must be independent across units. If users in the same household share an account, or if a user can refer another user (network effects), then one user being in treatment affects another user's behavior. This violates independence.

**Sample size follows from the minimum effect size you care about, the baseline metric rate, and the acceptable error rates. There's no "standard" sample size?** #flashcard
it depends on your specific context.

**Duration must account for novelty effects, day-of-week variation, and seasonal patterns. A 2-day test that runs over a weekend is not representative of a regular week.?** #flashcard
Duration must account for novelty effects, day-of-week variation, and seasonal patterns. A 2-day test that runs over a weekend is not representative of a regular week.

**Network effects: does one user's chatbot use affect another's experience? (Probably no, in this case?** #flashcard
the chatbot is user-specific.)

**Novelty effect?** #flashcard
users might engage with the chatbot just because it's new. Consider running for long enough that novelty wears off, or track whether effect degrades over time.

**Heterogeneous effects?** #flashcard
the chatbot might help new users (who have questions) and be irrelevant to returning users (who know what they want). Plan to segment.

**Did the experiment run cleanly? Were there any infrastructure changes or test overlaps during the run?** #flashcard
Did the experiment run cleanly? Were there any infrastructure changes or test overlaps during the run?

**Check the SRM: intended 50/50 split, actual split 49.8/50.2?** #flashcard
within noise, no SRM. If you had 48/52, flag this.

**p = 0.03 means?** #flashcard
under the null, there's a 3% chance of observing a 1.5% or greater lift by chance. This is below α = 0.05, so you reject the null.

**But state the confidence interval: say it's [+0.2%, +2.8%]. The true effect is somewhere in here with 95% confidence. The minimum plausible effect is +0.2%?** #flashcard
that's the lower bound you should use for the business case.

**Is a 1.5% lift in conversion worth shipping? Depends on?** #flashcard
what is conversion rate × revenue × volume? If this is on a surface that drives $100M/year, a 1.5% lift is $1.5M. If it's on a surface that drives $1M/year, it's $15K. The threshold for shipping depends on engineering cost and opportunity cost.

**Did session length, support tickets, or other guardrails degrade? If a guardrail moved significantly in the wrong direction, you don't ship even if the primary metric is positive?** #flashcard
you investigate the trade-off.

**Break by new vs. returning users, platform, region. If the lift is +5% for new users and −1% for returning users, you have a more nuanced decision?** #flashcard
maybe ship only for new users, or investigate why it hurt returning users.

**Ship, ship with monitoring, roll back, or run follow-up experiment? Be explicit about which and why.?** #flashcard
Ship, ship with monitoring, roll back, or run follow-up experiment? Be explicit about which and why.

**p-value?** #flashcard
if the null were true, how often would I see data this extreme? Used to decide whether to reject the null. Does NOT measure the probability the null is true.

**Confidence interval?** #flashcard
a range that, if you repeated the experiment many times, would contain the true parameter 95% of the time. Used to communicate the uncertainty in your estimate.

**Power?** #flashcard
probability of correctly detecting an effect of a given size if it exists. Determined by sample size, effect size, and significance threshold. Low power → high false negative rate.

**Central Limit Theorem?** #flashcard
the sampling distribution of the mean converges to normal as n grows, regardless of the underlying distribution. This is why t-tests and z-tests work on non-normal data when n is large enough.

**Comparing two means (continuous outcome, two groups)?** #flashcard
two-sample t-test (or Mann-Whitney U if distribution is highly non-normal at small n).

**Comparing proportions (binary outcome, two groups)?** #flashcard
two-proportion z-test or chi-squared test.

**Comparing more than two groups?** #flashcard
ANOVA (then post-hoc tests if significant, with multiple comparison correction).

**Paired data (same user before/after): paired t-test, not independent-samples t-test?** #flashcard
the pairing removes between-subject variance and increases power.

**Count data with small expected counts?** #flashcard
Fisher's exact test, not chi-squared.

**Find a comparable city that did not get the loyalty program (the "control city").?** #flashcard
Find a comparable city that did not get the loyalty program (the "control city").

**Measure revenue in both cities before and after the launch.?** #flashcard
Measure revenue in both cities before and after the launch.

**The DiD estimate?** #flashcard
(revenue change in treatment city) − (revenue change in control city).

**This removes time-invariant city-specific factors (the "difference" across cities) and common time trends (the "difference" across time periods).?** #flashcard
This removes time-invariant city-specific factors (the "difference" across cities) and common time trends (the "difference" across time periods).

**Key assumption: parallel trends?** #flashcard
in the absence of the loyalty program, the two cities would have experienced the same revenue trend. This is untestable but you can provide evidence by showing pre-treatment trends were similar.

**Situation?** #flashcard
"In Q3 last year I was analyzing whether to sunset a feature that had low engagement but strong retention correlation among a small user segment."

**Task: "I had 3 weeks and access to behavioral logs, but no survey data and no direct causality established?** #flashcard
I didn't know if low engagement meant users hated the feature or used it passively."

**Action: "I decided to: (1) estimate the upper-bound value of the segment using cohort LTV data?** #flashcard
if that segment churned at the rate of other low-engagement users, the revenue risk was ~$X; (2) run a 10-user usability session to get qualitative signal quickly; (3) explicitly present the recommendation as 'confident to within the bounds of this uncertainty' and recommended a 6-month staged deprecation with exit survey data collection built in, rather than an immediate sunset."

**Result?** #flashcard
"The staged deprecation was approved. The exit surveys 3 months in showed that 20% of the segment used the feature as a passive 'save' function they valued but rarely clicked. We redesigned it instead of sunsetting it and saw +4% retention in that segment."

**Survivorship bias?** #flashcard
analyzing only the companies that survived to ask what made them successful, ignoring all the equally-positioned companies that failed.

**Sampling bias in A/B tests?** #flashcard
if early users of a new feature are self-selected enthusiasts, their behavior doesn't predict how the full user base will respond when it's rolled out.

**Non-response bias in surveys: users who respond to a satisfaction survey are not representative of all users?** #flashcard
disengaged or dissatisfied users are less likely to respond.
