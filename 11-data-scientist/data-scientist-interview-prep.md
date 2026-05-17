# Data Scientist Interview Prep

---

## 1. Interview Format (by Company Type)

| Round Type | What's Tested | Common at |
| :--- | :--- | :--- |
| **Product sense / metrics** | Define success metrics, investigate drops, design measurement | Meta, Google, Airbnb, Uber |
| **A/B testing / experiment design** | Design experiment, power analysis, analyze results | All data-heavy companies |
| **SQL** | Window functions, funnel/retention/dedup queries | All |
| **Statistics** | p-value, CLT, CI, which test to use | All |
| **Causal inference** | Observational studies, DiD, when A/B isn't possible | Airbnb, Lyft, Netflix |
| **Python/pandas** | Data manipulation, groupby, merging, vectorized ops | ML-heavy DS roles |
| **Coding (algo)** | Light leetcode (easy/medium), array/dict/string | Some companies |
| **Case study / take-home** | End-to-end analysis from raw data | Stripe, DoorDash, others |
| **Behavioral (STAR)** | Ambiguous problems, stakeholder conflicts, impact | All |

---

## 2. Product Sense Questions

### Framework: Goal → Metrics → Guardrails → Segments → Trade-offs

**Template for "How would you measure success of feature X?"**

1. **Clarify the goal**: What problem does the feature solve? Who is it for?
2. **Primary metric (OEC)**: What single number captures feature success? (adoption rate, D7 retention for the feature, conversion lift)
3. **Secondary metrics**: What supporting signals indicate the feature is working? (session depth, feature engagement frequency)
4. **Guardrail metrics**: What must not degrade? (overall DAU, latency, core feature usage, NPS)
5. **Segment analysis**: Does the effect differ by user type, region, platform, new vs returning?
6. **Trade-offs**: Is there a tension between short-term engagement and long-term value? Could this cannibalize another feature?

**Example**: "Measure success of adding a 'Save for Later' button in a shopping app"
- Primary: 28-day purchase rate for items saved
- Secondary: save rate (adoption), cart abandonment rate
- Guardrails: overall purchase conversion, session length
- Segments: new vs returning users, mobile vs web
- Trade-off: saves may replace immediate purchases short-term but improve long-term LTV

---

## 3. Metrics Investigation Questions

### Framework: Verify → Segment → Correlate → External → Root Cause

**Template for "DAU dropped 20% — how do you investigate?"**

1. **Verify data integrity**:
   - Is this a logging issue? Check event pipeline health, row counts, timestamp anomalies
   - Is it one platform (iOS/Android/web) or global?
   - Is the metric definition consistent? (Did the query change?)

2. **Characterize the drop**:
   - When did it start? Sudden vs gradual
   - Is it recovering or continuing to decline?

3. **Segment**:
   - Platform: iOS vs Android vs web
   - Region: US vs international
   - User type: new vs returning, power users vs casual
   - Acquisition channel: organic vs paid
   - Which segment accounts for most of the absolute drop?

4. **Correlate with internal changes**:
   - Any product releases, experiments, or infrastructure changes on that date?
   - Marketing campaign starts/stops?
   - Push notification changes?

5. **External factors**:
   - Holiday, seasonal pattern, competitor launch?
   - App store update/removal?

6. **Root cause and action**:
   - Hypothesize mechanism → find corroborating evidence → propose fix → verify with experiment

---

## 4. A/B Test Questions

### Framework: Unit → Hypothesis → Metric → Sample Size → Duration → Risks

**Template for "How would you design an experiment for X?"**

1. **Unit of randomization**: User-level (for personalization, retention effects), session-level (only for stateless changes)
2. **Hypothesis**: directional, specific — "Reducing checkout steps from 3 to 1 will increase purchase conversion rate"
3. **Primary metric**: one OEC (e.g., purchase CVR)
4. **Guardrail metrics**: latency, error rate, cart abandonment
5. **Sample size**: estimate from baseline CVR, target MDE, desired power (80%); duration = n / daily eligible users per arm
6. **Randomization check**: SRM test after launch; covariate balance check
7. **Risks**:
   - Network effects? (use user-level, not session-level)
   - Novelty effect? (run 2+ weeks)
   - Multiple segments? (pre-specify subgroup analyses)
   - Multiple metrics? (apply correction or pre-specify OEC)

**Common follow-up traps**:
- "The p-value was 0.04, should we ship?" → check effect size, CI, guardrails, practical significance
- "Halfway through, results look significant, should we stop?" → no; use sequential testing; peeking inflates Type I error
- "Control group is smaller than expected" → SRM bug; investigate before analyzing

---

## 5. SQL Questions

### Common Patterns Tested

**Retention (Day-N)**
```sql
SELECT a.user_id
FROM events a
JOIN events b ON a.user_id = b.user_id
              AND DATE_DIFF('day', a.first_event, b.event_date) = 7
WHERE a.event_date = a.first_event;
```

**Funnel conversion**
```sql
SELECT
    COUNT(DISTINCT CASE WHEN event = 'view' THEN user_id END) AS views,
    COUNT(DISTINCT CASE WHEN event = 'click' THEN user_id END) AS clicks,
    COUNT(DISTINCT CASE WHEN event = 'purchase' THEN user_id END) AS purchases
FROM events;
```

**Ranking / top-N per group**
```sql
SELECT * FROM (
    SELECT *, RANK() OVER (PARTITION BY category ORDER BY revenue DESC) AS rk
    FROM products
) t WHERE rk <= 3;
```

**Deduplication**
```sql
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC) AS rn
    FROM users
) t WHERE rn = 1;
```

**Running total**
```sql
SELECT date, revenue,
       SUM(revenue) OVER (ORDER BY date ROWS UNBOUNDED PRECEDING) AS cumulative_revenue
FROM daily_revenue;
```

**Interview tips**:
- Clarify data model upfront (what does one row represent?)
- Handle NULLs explicitly (`COALESCE`, `NULLIF`)
- Use CTEs for multi-step logic — shows readability
- Verify edge cases: user with no activity, zero denominators, duplicate keys

---

## 6. Statistics Questions

**"What is a p-value?"**
- Probability of observing a test statistic as extreme as the one computed (or more extreme), assuming H₀ is true
- It is NOT: probability that H₀ is true; probability that the result is due to chance; probability that the alternative is true
- Common trap: p < 0.05 does not mean the effect is large or practically meaningful

**"What is a confidence interval?"**
- If the experiment were repeated many times, 95% of constructed CIs would contain the true parameter
- Not: 95% probability that the true value is in this specific interval

**"When do you use a t-test vs z-test?"**
- t-test: unknown population σ, using sample s; small or large n
- z-test: known population σ (rare); or large n where t ≈ z
- In practice: always use t-test for means; z-test for proportions (large n, Binomial approximation)

**"When do you use non-parametric tests?"**
- Data is ordinal or ranked
- Sample size too small for CLT to apply (<30)
- Distribution is heavily skewed with outliers that can't be removed
- Mann-Whitney U instead of t-test; Kruskal-Wallis instead of ANOVA

**"What are Type I and Type II errors?"**
- Type I (false positive, α): reject H₀ when it's true → false alarm; controlled by significance level
- Type II (false negative, β): fail to reject H₀ when it's false → miss a real effect; controlled by power = 1 − β
- Tradeoff: decreasing α increases β (lower power) for fixed sample size

---

## 7. Causal Inference Questions

**"A feature launched in one market — how do you measure impact?"**

Approach:
1. Is there a comparable control market? → DiD
2. Can you construct a synthetic control from multiple markets?
3. Was there a sharp eligibility cutoff? → RDD
4. Is there an instrument? → IV

**DiD answer**:
- Identify markets with similar pre-treatment trends (parallel trends)
- Validate with event study: plot $\hat{\tau}_t$ for pre-treatment periods — should be ~0
- Estimate: (treated market post - treated market pre) - (control market post - control market pre)
- Cluster standard errors at market level
- Check for: market-level spillovers, staggered rollout issues

**"We can't run an A/B test — how do you estimate causal effect?"**
- Propensity score matching or IPW on observational data (requires no unmeasured confounders)
- DiD if there's a before/after and a comparable control group
- RDD if there's a sharp eligibility threshold
- IV if there's a valid instrument
- Synthetic control if single treated unit with many donor units
- Always state the key identifying assumption and how you'd test it

---

## 8. Python/Pandas Patterns

```python
import pandas as pd
import numpy as np

# GroupBy with multiple aggregations
df.groupby('cohort').agg(
    revenue=('revenue', 'sum'),
    orders=('order_id', 'nunique'),
    users=('user_id', 'nunique'),
    avg_order_value=('revenue', 'mean')
).reset_index()

# Merge (equivalent to SQL JOIN)
pd.merge(left, right, on='user_id', how='left')
pd.merge(left, right, left_on='uid', right_on='user_id', how='inner')

# Pivot table
df.pivot_table(index='cohort_week', columns='weeks_since_signup',
               values='user_id', aggfunc='nunique')

# Apply (use sparingly; prefer vectorized)
df['category'] = df['revenue'].apply(lambda x: 'high' if x > 100 else 'low')
# Faster vectorized equivalent:
df['category'] = np.where(df['revenue'] > 100, 'high', 'low')

# Vectorized string operations
df['email_domain'] = df['email'].str.split('@').str[1]
df[df['name'].str.contains('John', case=False)]

# Date operations
df['date'] = pd.to_datetime(df['date'])
df['week'] = df['date'].dt.to_period('W')
df['day_of_week'] = df['date'].dt.dayofweek

# Chained operations (readable pipeline)
result = (
    df
    .query("status == 'active' and revenue > 0")
    .assign(revenue_per_session=lambda d: d['revenue'] / d['sessions'])
    .groupby('country')['revenue_per_session'].mean()
    .sort_values(ascending=False)
    .head(10)
)
```

---

## 9. Behavioral Questions (STAR Format)

**STAR**: Situation → Task → Action → Result (quantified)

### Common Themes in DS Behavioral

**Data-driven decision under ambiguity**
- S: stakeholder wanted to launch feature without data; timeline was tight
- T: build quick evidence base to inform go/no-go
- A: designed holdout test on 5% of users; analyzed proxy metrics within 2 days
- R: showed 12% drop in engagement; feature redesigned; eventual launch lifted retention 8%

**Stakeholder conflict over metrics**
- S: product team and finance team disagreed on success metric for pricing change
- T: align on common definition before experiment launch
- A: facilitated joint session; defined primary OEC (revenue/user) and guardrails (churn rate); pre-registered
- R: clear agreement prevented post-hoc metric shopping; experiment ran cleanly

**Ambiguous problem**
- Clarify scope and constraints before diving in
- Propose multiple approaches with tradeoffs
- Recommend the most practical one given data/time constraints

---

## 10. Case Study Structure

**Template for open-ended DS case studies**:

1. **Clarify**: What is the business goal? What data is available? What's the time constraint?
2. **Define success**: What does a good outcome look like? Primary metric?
3. **Data needed**: What tables/signals would you need? What would one row represent?
4. **Analysis plan**:
   - EDA: distribution of key variables, missingness, outliers
   - Core analysis: experiment design, statistical test, or modeling approach
   - Segmentation: which user groups to analyze separately
5. **Caveats**: What assumptions are you making? What could go wrong (data quality, confounders)?
6. **Recommendation**: Clear, actionable, with quantified uncertainty

**Time management in whiteboard case studies**:
- Spend 20% on clarifying, 40% on analysis plan, 30% on findings/recommendation, 10% on caveats
- Always ask before going deep: "Does this direction seem right?"

---

## 11. Common Traps

| Trap | Description | Correct Framing |
| :--- | :--- | :--- |
| **Correlation ≠ causation** | Reporting correlated features as drivers | Explicitly name the confounders; use causal language only with causal design |
| **Selection bias** | Analyzing only active users, ignoring churned | Define cohort at entry; include all exposed units |
| **Ignoring data quality** | Building model on dirty data | Always check missingness, distributions, leakage before modeling |
| **Statistical vs practical significance** | Large n → tiny p-value on meaningless effect | Report effect size, CI, and business impact, not just p-value |
| **P-hacking** | Testing until significant | Pre-register hypothesis and metric; sequential testing if needed |
| **Survivorship bias** | Using only successful examples | Include failures; analyze the full population |
| **Simpson's paradox** | Aggregated trend reverses in subgroups | Always stratify; check if subgroup imbalance drives aggregate result |
| **Ignoring seasonality** | Comparing week to prior week without calendar adjustment | Use YoY comparison or seasonal decomposition |
| **Short-term thinking** | 7-day experiment for feature with learning curve | Run long enough for novelty to wear off; consider holdout for long-term effects |
| **Metric gaming** | Optimizing the metric without improving the underlying construct | Attach counter-metrics; apply Goodhart's Law thinking |
