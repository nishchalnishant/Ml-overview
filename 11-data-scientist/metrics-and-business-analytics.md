# Metrics and Business Analytics

---

## 1. Product Metrics Framework

### Metric Taxonomy
- **Input metrics** (leading indicators): actions and investments the team controls — features shipped, content uploaded, emails sent
- **Output metrics** (lagging indicators): business outcomes — revenue, retention, MAU
- **Guardrail metrics**: must not degrade — latency, error rate, support tickets, refund rate; prevent metric gaming

### Hierarchy
```
Company goal (e.g., grow revenue)
  └── North Star Metric (e.g., weekly active users)
        ├── Driver metrics (e.g., new user activation rate, feature adoption)
        │     └── Input metrics (e.g., onboarding steps completed)
        └── Guardrail metrics (e.g., p99 latency, error rate)
```

---

## 2. North Star Metric

- Single metric that best captures the core value delivered to users — if it grows, the business grows sustainably
- Must be: measurable, leading indicator of revenue, not easily gamed, understood company-wide

| Product | North Star Metric |
| :--- | :--- |
| Facebook | Daily Active Users |
| Airbnb | Nights booked |
| Spotify | Time listening |
| Slack | Messages sent |
| Duolingo | Daily active learners |
| HubSpot | Weekly active teams |

- Danger: NSM can be gamed (e.g., spam notifications inflate DAU); pair with guardrails
- NSM is not revenue — revenue is a consequence of delivering user value

---

## 3. AARRR Funnel (Pirate Metrics)

| Stage | Definition | Example Metrics |
| :--- | :--- | :--- |
| **Acquisition** | Users find the product | Sessions, new signups, CAC, organic vs paid split |
| **Activation** | First positive experience | Aha moment completion rate, Day-1 retention, onboarding completion |
| **Retention** | Users return over time | D7/D28 retention, WAU/MAU, L28 |
| **Referral** | Users bring others | K-factor (viral coefficient), NPS, invite send rate |
| **Revenue** | Monetization | MRR, ARPU, LTV, conversion to paid |

- Funnel identifies the weakest stage — the biggest leverage point for growth
- Leaky funnel: high acquisition but low activation → fix onboarding, not ads

---

## 4. Retention Curves

### Point-in-Time Retention
- **Day-N retention**: fraction of users from day-0 cohort still active on day N
- Common milestones: D1 (≥ 40% is strong for consumer), D7, D28, D90

### L-Metrics (Rolling Window)
- **L28**: number of days a user was active in the last 28 days; range 0–28
- **L7/L28 ratio** (stickiness): `L7 / L28`; ranges 0–1; above 0.5 = very engaged users (active more than half of recent days)
- L-metrics smooth out weekly cycles and capture habitual use

### Retention Curve Shape
- **Flattening**: retention curve stabilizes above zero → product has a retained core
- **Declining to zero**: product has no habit formation; needs lifecycle campaigns or feature improvement
- Benchmark varies by category: social (D28 > 25%), utility (D28 > 40%), gaming (D7 > 20%)

---

## 5. Engagement Metrics

| Metric | Formula | Interpretation |
| :--- | :--- | :--- |
| DAU | count distinct active users per day | Daily pulse |
| WAU | count distinct active users per week | Weekly health |
| MAU | count distinct active users per month | Scale |
| DAU/MAU | DAU ÷ MAU | Stickiness (0 = no habit, 1 = all monthly users active daily) |
| Session length | avg(session_end - session_start) | Depth of engagement per visit |
| Session depth | avg(events per session) | Breadth of feature usage |
| Feature adoption | users who used feature X ÷ eligible users | Reach of specific feature |

- DAU/MAU > 0.2 is considered good for most consumer apps; Facebook historically ~0.65
- Distinguish active definition: log-in counts vs meaningful action (sent message, made purchase)

---

## 6. Revenue Metrics

| Metric | Formula | Notes |
| :--- | :--- | :--- |
| **MRR** | Sum of monthly recurring revenue | SaaS backbone; track new, expansion, contraction, churn components |
| **ARR** | MRR × 12 | Annualized view; useful for large enterprise contracts |
| **ARPU** | Revenue ÷ total users | All users including free |
| **ARPPU** | Revenue ÷ paying users | Excludes non-paying; higher than ARPU |
| **LTV** | ARPU × gross_margin ÷ churn_rate | Customer lifetime value; see Section 7 |
| **CAC** | Sales & marketing spend ÷ new customers | Cost to acquire one customer |
| **Payback period** | CAC ÷ (ARPU × gross_margin) | Months to recoup acquisition cost |
| **Gross margin** | (Revenue − COGS) ÷ Revenue | Often 60–80% for SaaS, 30–50% for marketplace |

---

## 7. LTV Calculation

### Simple LTV
$$\text{LTV} = \text{ARPU} \times \frac{\text{Gross Margin}}{\text{Monthly Churn Rate}}$$

- Assumes constant ARPU and geometric retention
- Example: ARPU = \$50, margin = 70%, monthly churn = 5% → LTV = \$50 × 0.7 / 0.05 = \$700

### Cohort-Based LTV (Survival Curve)
- Track each cohort's cumulative revenue over time
- Fit a survival function to retention curve: often exponential or power law
- Extrapolate beyond observation window using fitted curve
- More accurate than formula-based; captures cohort heterogeneity

### LTV:CAC Ratio
- Healthy SaaS benchmark: LTV:CAC ≥ 3
- LTV:CAC < 1: losing money on every customer
- Payback period target: < 12 months (consumer), < 18 months (enterprise)

---

## 8. Churn and Retention

### Definitions
- **Logo churn rate**: % of customers who cancel in a period; $\frac{\text{customers lost}}{\text{customers at start of period}}$
- **Revenue churn rate**: % of MRR lost from existing customers (cancellations + downgrades)
- **Gross revenue retention (GRR)**: MRR retained excluding expansion; GRR ≤ 100%
- **Net Dollar Retention (NDR)**: $\frac{\text{MRR from existing customers (incl. expansion, excl. new)}}{\text{MRR at start of period}}$

### NDR Interpretation
- NDR > 100%: expansion revenue from existing customers exceeds churn + contraction → growth without new customers
- NDR 90–100%: stable, retention needs work
- NDR < 90%: revenue declining from existing base — existential risk
- Best-in-class SaaS (Snowflake, Datadog): NDR 130–150%

---

## 9. Conversion Funnel

- Define stages: e.g., visit → signup → activation → first purchase → repeat purchase
- Compute per-step conversion rate: $\text{CVR}_{i \to i+1} = \frac{\text{users completing step } i+1}{\text{users completing step } i}$
- Drop-off analysis: which step loses the most users relative to entry volume?
- Segment funnel by: acquisition channel, device type, geography, user segment — drop-off patterns differ
- Intervention priority: fix the step with the largest absolute number of lost users, weighted by downstream LTV

---

## 10. Anomaly Detection in Metrics

### Control Charts (Statistical Process Control)
- Track rolling mean ± 2σ (warning) and ± 3σ (action) bands
- Computed on a rolling window from recent stable period; update baseline periodically

### Seasonal Decomposition Before Alerting
- Decompose metric into trend + seasonality + residual (STL decomposition)
- Alert on residual anomalies, not raw values — avoids weekly/holiday false alerts
- Tools: `statsmodels.tsa.seasonal.STL`, `Prophet`

### PSI-Based Alerting
- Use PSI on input feature distributions to detect data pipeline issues upstream of metric degradation

### Practical Anomaly Checklist
1. Is data pipeline working? (check row counts, freshness)
2. Is the anomaly real or a logging change?
3. Is it a single segment or global?
4. Does it correlate with a deployment, marketing event, or external event?
5. Is it within historical volatility or statistically significant?

---

## 11. Counter-Metrics and Guardrails

- Every metric optimization creates an incentive to game it → attach counter-metrics
- Examples:
  - Optimizing CTR → add `irrelevant click rate` or user satisfaction guardrail
  - Optimizing session length → add `content quality rating`
  - Optimizing notifications sent → add `notification disable rate`, `DAU impact`
- Guardrails are non-negotiable: any experiment degrading a guardrail metric is rejected regardless of OEC gain

---

## 12. Goodhart's Law

> "When a measure becomes a target, it ceases to be a good measure." — Charles Goodhart

### Implications for Metric Design
- Optimizing a proxy metric destroys its predictive validity for the underlying goal
- Examples:
  - **Click-through rate**: optimize → clickbait; users click but bounce immediately
  - **Lines of code**: optimize → verbose, unmaintainable code
  - **Tickets closed**: optimize → close tickets without resolving issues
  - **Engagement time**: optimize → addictive dark patterns that don't deliver value

### Mitigations
- Measure the underlying goal directly when possible (rare)
- Use multiple metrics that are difficult to jointly game
- Rotate metrics to prevent optimization to a fixed target
- Add qualitative checks (user research, satisfaction surveys) alongside quantitative metrics
- Distinguish manipulation of the metric from improvement in the underlying construct

---

## 13. OKR vs KPI

| | OKR | KPI |
| :--- | :--- | :--- |
| **Full name** | Objectives and Key Results | Key Performance Indicator |
| **Purpose** | Set ambitious goals and measure progress | Monitor ongoing operational health |
| **Time horizon** | Quarterly or annual | Ongoing (weekly/monthly dashboards) |
| **Target type** | Aspirational (60–70% attainment = success) | Threshold-based (must stay above/below) |
| **Ownership** | Team-level, cross-functional | Team or individual |
| **Example** | O: Improve user onboarding. KR: Increase D7 retention from 25% to 35% | D7 retention monitored weekly as health metric |

### Cascading OKRs
- Company OKRs → Team OKRs → Individual KRs
- Each level's KRs should directly contribute to the parent level's Objectives
- Avoid vanity KRs that are easily met without business impact
- OKRs should be outcome-focused, not output-focused ("increase D7 retention" not "ship 5 features")
