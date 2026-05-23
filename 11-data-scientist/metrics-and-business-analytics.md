---
module: Data Scientist
topic: Metrics And Business Analytics
subtopic: ""
status: unread
tags: [datascientist, ml, metrics-and-business-analytics]
---
# Metrics and Business Analytics

---

## 1. The Problem with Measuring Products

You ship a product. You want to know if it is working. You could track everything — pageviews, clicks, sessions, revenue, errors, signups — but tracking everything gives you noise, not signal. You end up with dashboards nobody reads, arguments about which number to optimize, and teams that improve one metric by degrading another.

The discipline of metrics design exists to answer a harder question than "is this number going up?": **which numbers, if they go up, mean the product is genuinely getting better, and which ones can go up for bad reasons?**

Every framework in this document is an answer to some version of that question.

---

## 2. Product Metrics Framework: Input, Output, Guardrail

### The problem

Suppose your goal is to grow revenue. Revenue is a lagging signal — by the time it moves, the decision that caused it was made weeks or months ago. You cannot steer a product by watching revenue alone.

But if you only watch leading signals — like features shipped or emails sent — you can stay very busy while the business decays.

You need metrics at multiple time horizons with different purposes.

### The core insight

Every metric a team tracks can be classified by what it measures and who controls it:

- Some metrics measure **what the team does** (inputs). These are controllable, fast-moving, and leading.
- Some metrics measure **what the business gets** (outputs). These are the actual outcomes — slower-moving, harder to game, harder to move directly.
- Some metrics exist only to **detect harm** (guardrails). These should not degrade when you optimize the others.

### The mechanics

**Input metrics (leading indicators):** Actions and investments the team controls. Features shipped, onboarding steps completed, content uploaded, A/B tests launched, support articles written. Moving these should, via your causal theory, move outputs.

**Output metrics (lagging indicators):** Business outcomes. Revenue, retention, MAU, conversion rate, NPS. These validate whether your causal theory was correct.

**Guardrail metrics:** Hard constraints — latency, error rate, refund rate, support ticket volume, account deletion rate. They exist because output metrics can be improved in ways that damage the product or the business long-term. A guardrail violation is a veto on shipping even if all other metrics look good.

### The hierarchy

```
Company goal (e.g., grow sustainable revenue)
  └── North Star Metric (e.g., weekly engaged users)
        ├── Driver metrics (e.g., activation rate, feature adoption)
        │     └── Input metrics (e.g., onboarding completion, time-to-value)
        └── Guardrail metrics (e.g., p99 latency, error rate, refund rate)
```

### What breaks

Teams that only track outputs have no levers. Teams that only track inputs can stay busy going nowhere. Guardrails are the most commonly omitted — and their absence is how organizations end up optimizing spam notifications to inflate DAU, or discounting aggressively to inflate MRR while destroying gross margin.

---

## 3. North Star Metric

### The problem

A team needs to make hundreds of decisions per quarter — which features to build, which experiments to run, how to allocate engineering time. If every decision is evaluated against a different metric, you get incoherence. One team optimizes for revenue, another for engagement, another for growth, and they make decisions that conflict with each other without realizing it.

The North Star Metric exists to solve the coordination problem: give every team a single number that, if improved, means the product is genuinely delivering more value to users.

### The core insight

Revenue is a consequence of value delivery, not the same thing as value delivery. A company that extracts money from users without delivering value can show short-term revenue growth while destroying long-term retention. A North Star Metric should measure value delivered to users — because sustainable revenue follows from that.

The test: if your NSM goes up for a bad reason (e.g., users are confused and clicking more), is the business actually better? If yes, you have the wrong metric.

### The mechanics

Criteria for a good North Star Metric:
1. **Measures value exchange** — captures the moment the product actually does something useful for a user
2. **Leading indicator of revenue** — growing it should eventually cause revenue to grow
3. **Hard to game without delivering value** — you cannot easily manufacture the metric without actually helping users
4. **Understandable across the company** — an engineer and a marketer should both be able to explain it

| Product | North Star Metric | Why |
| :--- | :--- | :--- |
| Facebook | Daily Active Users | Ad revenue requires daily attention |
| Airbnb | Nights booked | Direct value delivery to both sides of marketplace |
| Spotify | Time listening | Subscription value = hours of music delivered |
| Slack | Messages sent | Value is communication, not login |
| Duolingo | Daily active learners | Learning requires frequency |
| HubSpot | Weekly active teams | B2B value = teams using the product, not just paying for seats |

### What breaks

**The NSM can be gamed.** Facebook optimizing for DAU triggered a decade of notification spam. If you are not pairing the NSM with guardrails (time-well-spent metrics, unsubscribe rates, support volumes), you will optimize for a hollow version of the number.

**The NSM is not the only metric.** It is the coordinating metric. Individual teams need driver metrics that are causally upstream of the NSM. Fixing onboarding improves activation, activation improves Day-1 retention, Day-1 retention feeds into the DAU count — that causal chain is what makes the NSM actionable.

**Single-metric optimization is dangerous.** Every NSM has a shadow. For Airbnb, obsessing over nights booked can lead to tolerating bad host behavior. For Spotify, time listening could be gamed by low-quality ambient playlists. Always hold the NSM against its guardrails.

---

## 4. AARRR Funnel (Pirate Metrics)

### The problem

A product team needs to know where value is leaking. A user might hear about the product (acquisition), sign up (partial activation), try it once, and never return (retention failure). The revenue problem is often a retention problem in disguise. But if you only track signups and revenue, you cannot see where the leak is.

The AARRR framework forces you to model the entire customer journey as a funnel, so you can identify which stage is the binding constraint.

### The core insight

Users pass through qualitatively different states, and improving the wrong state wastes resources. Pouring more users into a broken activation step is pouring water into a leaky bucket. You must find the leak before you invest in acquisition.

### The mechanics

| Stage | What it measures | Example metrics | Diagnostic question |
| :--- | :--- | :--- | :--- |
| **Acquisition** | Users find and arrive | Sessions, new signups, CAC, channel mix | Where are users coming from and what does each channel cost? |
| **Activation** | First genuine value moment | "Aha moment" completion rate, D1 retention, onboarding completion | Do new users experience the product's core value? |
| **Retention** | Users return over time | D7/D28/D90 retention, L28, stickiness | Are users forming habits? |
| **Referral** | Users bring other users | K-factor, NPS, invite-send rate | Does the product grow itself? |
| **Revenue** | Monetization | MRR, ARPU, LTV, paid conversion rate | Is value delivery translating to willingness to pay? |

### Using the funnel

Step 1: Measure conversion rates between each stage.
Step 2: Find the stage with the largest absolute drop-off.
Step 3: Fix that stage before optimizing any other.

A product with 100,000 acquisitions per month, 60% activation, 40% D30 retention, and 5% paid conversion is losing 40% of potential revenue at retention, not at conversion. Adding more acquisition spend there would be wasteful.

### What breaks

**The funnel is not always linear.** In B2B, a champion inside a company may activate and refer internally before revenue is ever discussed. In viral consumer products, referral happens before retention is established. Map your actual user journey before forcing it into AARRR.

**Activation is the most commonly misdiagnosed stage.** Teams measure "account creation" as activation when the real activation moment is the first value experience — which might be 5 minutes or 5 days later. You need qualitative research to find the true "aha moment" before you can measure it.

---

## 5. Retention Curves

### The problem

You know users sign up. The question that actually determines whether you have a business is: what fraction of them are still using the product months from now?

A product with no retention is a bucket with no bottom. You can fill it with acquisition spend, but the moment you stop spending, the active user count collapses. A product with strong retention builds a compounding base.

### The core insight

Retention curves tell you whether your product has found product-market fit with a segment of users. When you plot "percent of a cohort still active at day N" against N, the shape of the curve is the most important fact about your product's long-term health.

**A curve that flattens** means some fraction of users have found permanent value. That flat residual is the foundation of a sustainable business.

**A curve that reaches zero** means the product has not found sustained value for anyone. Growth is entirely a function of new acquisition. You do not have a sustainable business yet.

### The mechanics

**Point-in-time retention metrics:**

- **D1 retention:** Percent of users who return on day 2 after signup. Measures first impression and activation quality.
- **D7 retention:** Percent of users active in week 1 who return in week 2. Measures early habit formation.
- **D28 retention:** Percent of users still active at one month. Measures whether the product has become part of a routine.
- **D90 retention:** Three-month retention. For consumer apps, this approaches the "retained vs churned" bifurcation point.

Rough industry benchmarks (vary significantly by product category):

| Stage | Consumer app | B2B SaaS |
| :--- | :--- | :--- |
| D1 | 25–40% | 60–80% |
| D7 | 10–20% | 40–60% |
| D28 | 5–15% | 25–45% |

**L28 (Days Active in Last 28):** The number of distinct days a user was active in the last 28. A distribution of L28 across your user base shows you not just whether users are retained, but how deeply they use the product. A bimodal distribution (many users at L28=1–2 and many at L28=20+) suggests you have two different user segments with different needs.

**Cohort retention curves:** Rather than point-in-time metrics, plot the full retention curve for each monthly acquisition cohort on the same chart. This reveals two things:
1. Whether retention is improving over time (newer cohorts retaining better than older ones — a sign of product improvement)
2. Whether older cohorts show a flattening pattern (evidence of a loyal retained base)

### What breaks

**Redefining "active" to inflate retention.** If you set the bar for "active" at any server-side event — including background processes, push notification delivery, or auto-refresh — you can manufacture high retention numbers. The meaningful definition of active is a deliberate user-initiated action that reflects intent to use the product.

**Survivorship bias in long-term cohorts.** Old cohorts that are still active look great because all the bad users churned long ago. The signal is in the shape of the curve, not the absolute value of late-stage retention.

**Ignoring the zero-retention case.** If your D90 curve is still declining with no sign of flattening, you do not have product-market fit. This is the most important retention signal there is, and it is easy to miss if you are only looking at aggregated MAU.

---

## 6. Stickiness: DAU/MAU

### The question DAU/MAU answers

DAU and MAU individually tell you scale — how many people used the product in a given period. But they tell you nothing about engagement density. A product with 1M MAU could mean 1M people using it once a month (minimal habituation) or 1M people using it nearly every day (deep habit). The raw MAU number looks identical in both cases.

DAU/MAU collapses this ambiguity into a single 0–1 number: of all the people who used your product this month, what fraction used it today? Higher values mean users are coming back more frequently — the product is a habit, not an occasional visit.

### Why this matters

Stickiness predicts retention because habituated users are harder to churn. If a user builds a product into their daily routine, switching costs increase, and sensitivity to competitor offerings decreases. Facebook has historically targeted DAU/MAU of 0.65–0.70. A new consumer app at 0.10–0.20 is early in habit formation. Enterprise SaaS products often see DAU/MAU in the 0.20–0.40 range — not because engagement is low but because the natural usage frequency is lower.

**DAU/WAU** gives a weekly version of the same question, more appropriate for products with weekly natural cadence (work tools, weekly newsletters, fitness apps with 3x/week targets).

### What breaks

DAU/MAU is meaningless for inherently low-frequency products. A tax-filing product, an annual insurance renewal app, or a once-per-pregnancy parenting resource cannot and should not target high DAU/MAU. Using stickiness as a universal goal will lead those teams to add artificial engagement mechanics (notifications, gamification, daily streaks) that do not serve users and add noise to the metric.

Always ask: **what is the natural usage frequency for this product?** Then choose the time window (DAU/WAU vs. DAU/MAU vs. WAU/MAU) that matches that cadence. Stickiness is only interpretable relative to the product's natural rhythm.

---

## 7. Engagement Metrics: DAU, WAU, MAU

### The problem

You need a single number to answer "how big is the active user base right now?" But what counts as "active"? A user who logged in once 29 days ago versus a user who uses the app daily both count as MAU. The aggregated count hides this difference.

DAU, WAU, and MAU are population counts at different time horizons. Their usefulness comes from comparing them to each other (stickiness) and tracking their trend lines over time (growth vs. contraction).

### The mechanics

- **DAU (Daily Active Users):** Unique users who performed a meaningful action on a given day. The highest-frequency measure of product health.
- **WAU (Weekly Active Users):** Unique users active in a 7-day window. Smooths day-of-week effects. More appropriate as a primary metric for products used on a work-week cadence.
- **MAU (Monthly Active Users):** Unique users active in a 30-day window. The standard denominator in DAU/MAU and the most commonly reported investor metric.

**Day-of-week effects:** Consumer apps typically see DAU dip on Tuesdays/Wednesdays and spike on Sundays. B2B apps invert this — high DAU Monday–Thursday, low on weekends. Always compare day-over-day (same day of the week, year-over-year or month-over-month) rather than raw sequential days.

**The moving average:** A 7-day rolling average of DAU is the standard smoothed signal. It removes weekend/weekday noise while remaining sensitive to real changes.

### What breaks

**The definition of "active" is load-bearing.** If you include background app refreshes, email opens, or push notification deliveries, you can manufacture any DAU you want. The definition must be anchored to a deliberate user action — a search, a message sent, a purchase, content viewed for more than N seconds. Document this definition and never change it without creating a break in the time series.

**MAU as a vanity metric.** Because MAU uses a 30-day window, a user who opened the app once on day 1 of the month counts the same as a daily user for the full month. MAU is useful for reporting to investors and for the denominator of DAU/MAU. It is not a sufficient engagement signal on its own.

---

## 8. Revenue Metrics

### The problem

Revenue can grow for different reasons, and not all of those reasons indicate a healthy business. New customer acquisition inflates revenue even with 100% annual churn. One-time upsells disguise flat recurring revenue. Average revenue per user can rise even as the user base shrinks. You need metrics that decompose revenue into its structural components so you can understand what is actually happening.

### MRR and ARR

**The question:** How much predictable revenue does the business generate each period?

**Core insight:** Subscription businesses derive their value from recurring, predictable cash flows. MRR (Monthly Recurring Revenue) strips out one-time payments, contracts, and non-recurring items to show only the revenue stream you can count on next month. ARR (Annual Recurring Revenue) = MRR × 12, used as a business-scale metric.

**Mechanics:**

MRR has five components that must be tracked separately:

| Component | What it is | Sign |
| :--- | :--- | :--- |
| **New MRR** | Revenue from new customers | + |
| **Expansion MRR** | Revenue from upgrades/upsells of existing customers | + |
| **Contraction MRR** | Revenue lost to downgrades of existing customers | − |
| **Churned MRR** | Revenue lost from cancellations | − |
| **Reactivation MRR** | Revenue from previously churned customers | + |

Net New MRR = New MRR + Expansion MRR − Contraction MRR − Churned MRR + Reactivation MRR

**What breaks:** Including one-time fees, setup charges, or professional services in MRR inflates the number and misrepresents recurring revenue quality. Investors and analysts will adjust for this, and you should too.

### ARPU and ARPPU

**The question:** On average, how much does each user (or paying user) generate?

**ARPU (Average Revenue Per User)** = Total Revenue / Total Active Users

**ARPPU (Average Revenue Per Paying User)** = Total Revenue / Total Paying Users

**Why both matter:** ARPU tells you monetization efficiency across your entire user base. ARPPU tells you the yield from users who have converted to paying. The gap between them reflects your conversion rate. A product with ARPU = $2 and ARPPU = $20 has a 10% conversion rate.

**Segmenting matters:** ARPU averaged across user segments is almost always misleading. Power users paying $50/month dilute the average if most users are on a free tier. Track ARPU per cohort, per channel, per plan, not just in aggregate.

**What breaks:** ARPU can rise as you lose low-value users (denominator shrinks faster than numerator). Rising ARPU in a shrinking user base is not a positive signal — check both the numerator and denominator trend independently.

### Gross Margin

**The question:** After paying the direct costs of delivering the product, how much do you keep?

Gross Margin = (Revenue − Cost of Goods Sold) / Revenue

For SaaS, COGS includes hosting, third-party APIs, customer support, and payment processing. High gross margin (70–80%+ for pure SaaS) is what makes the unit economics of customer acquisition work — a high-gross-margin business can afford to spend more on acquisition and still be profitable per customer at scale.

**What breaks:** Teams that exclude support costs or infrastructure costs from COGS overstate gross margin. This matters enormously in LTV calculations — LTV is built on gross margin, not revenue.

---

## 9. LTV: Customer Lifetime Value

### The problem

Knowing that a customer pays you $30/month is not enough to decide how much to spend acquiring them. If they stay 3 months, you recover $90 — not much room for acquisition cost. If they stay 3 years, you recover $1,080. LTV answers the question: **how much total value does the average customer deliver over their full relationship with the product?**

Without LTV, you cannot make rational decisions about customer acquisition cost (CAC). With it, LTV:CAC becomes the fundamental unit economics ratio.

### Method 1: Formula-based LTV

**Core insight:** If churn is roughly constant over time, the expected lifetime of a customer follows a geometric distribution. The expected number of months a customer remains active = 1 / monthly churn rate.

**Mechanics:**

```
Monthly Churn Rate = (Customers Lost in Month) / (Customers at Start of Month)

Average Customer Lifetime = 1 / Monthly Churn Rate

LTV = ARPU × Gross Margin × Average Customer Lifetime
    = ARPU × Gross Margin × (1 / Monthly Churn Rate)
```

Example: ARPU = $50/month, Gross Margin = 75%, Monthly Churn = 3%

```
LTV = $50 × 0.75 × (1 / 0.03)
    = $37.50 × 33.3
    = ~$1,250
```

**What breaks:**
- This formula assumes constant churn. Real churn is often front-loaded (many users churn in the first 1–3 months; survivors are much stickier). The formula will underestimate LTV for products with strong early churn and flat later churn.
- It uses average ARPU, which hides segment heterogeneity. Enterprise customers have radically different LTV than SMB customers.
- It uses gross margin — if you use revenue instead, you overstate LTV by the reciprocal of gross margin.

### Method 2: Cohort survival curve LTV

**Core insight:** Rather than assuming constant churn, observe actual retention curves from historical cohorts and integrate the area under the curve.

**Mechanics:**

For each monthly cohort, measure the fraction of customers still active at each month (the survival curve). Multiply the survival rate at each month by the monthly revenue per customer and gross margin. Sum across all months.

```
LTV = Σ (Survival Rate at Month t × ARPU × Gross Margin)
      for t = 1 to T
```

This is more accurate because it incorporates the actual shape of retention, including front-loaded churn and eventual flattening.

**In practice:** Plot the survival curves of multiple monthly cohorts on the same chart. If older cohorts still have surviving customers generating revenue, project forward using the flattened tail as an asymptote. Cap the projection at a reasonable horizon (36–60 months for most products) to avoid speculative extrapolation.

**What breaks:**
- Cohort LTV requires enough historical data (ideally 24+ months) to observe the curve shape. Early-stage companies cannot yet calculate this accurately.
- Product and pricing changes affect cohort survival. A cohort acquired under a different pricing model has a different LTV than current cohorts. Use only comparably acquired cohorts.
- Projecting survival curves forward requires assuming the current curve shape continues. If you are actively improving retention, historical curves understate future LTV.

### LTV:CAC ratio

**The question:** For every dollar spent acquiring a customer, how many dollars of gross margin do you eventually recover?

```
LTV:CAC = LTV / Customer Acquisition Cost
```

Benchmarks:
- LTV:CAC < 1: You are destroying value on every customer acquired
- LTV:CAC 1–3: Marginal; viable only with very fast payback periods
- LTV:CAC 3: The traditional "healthy SaaS" benchmark
- LTV:CAC > 5: Often a signal to invest more aggressively in acquisition (you are leaving money on the table)

### Payback period

**The question:** How many months until the gross margin from a customer recoups the acquisition cost?

```
Payback Period (months) = CAC / (ARPU × Gross Margin)
```

Payback period is a cash flow metric, not a profitability metric. A business with a 24-month payback period needs to finance customer acquisition for two years before breaking even per customer. This is why high-growth SaaS companies consume so much capital — they are pre-financing future LTV. Investors with long time horizons accept this; lenders often do not.

Benchmark: < 12 months payback is generally considered healthy for venture-backed SaaS. Sub-6-month payback enables very aggressive growth with minimal external capital.

---

## 10. Churn

### The problem

You can be growing in new customers while the business is actually declining, if churn is high enough. Understanding churn precisely is the difference between correctly diagnosing a retention crisis and being fooled by gross-up numbers.

### Logo churn vs. revenue churn

**The question logo churn answers:** Are we losing customers?

```
Logo Churn Rate = (Customers Lost in Period) / (Customers at Start of Period)
```

**The question revenue churn answers:** Are we losing revenue?

These two numbers can diverge sharply. Losing 10 small customers while retaining 1 large enterprise customer is 10 logo churns and negative revenue churn (if the enterprise expands). Which one you optimize for depends on your business model.

### Gross vs. Net Revenue Retention

**Gross Revenue Retention (GRR):** Measures what fraction of last period's revenue you still have this period, counting only losses (contraction + churn). Expansions are excluded.

```
GRR = (Starting MRR − Churned MRR − Contraction MRR) / Starting MRR
```

GRR can never exceed 100%. It tells you how well you are holding on to revenue from existing customers, before accounting for upsells.

**Net Dollar Retention / Net Revenue Retention (NDR/NRR):** Measures what fraction of last period's revenue you have this period, including expansions. Can exceed 100%.

```
NDR = (Starting MRR − Churned MRR − Contraction MRR + Expansion MRR) / Starting MRR
```

### Why NDR > 100% is the holy grail

If NDR > 100%, your existing customer base grows in revenue even with zero new customers. This means the business can grow — slower, but without acquisition spend — purely through expansion of the installed base. This dramatically changes the economics of the business and its capital requirements.

Benchmarks:
- NDR < 85%: Serious retention problem; growth requires massive acquisition to offset losses
- NDR 85–100%: Adequate; growth requires new customers but churn is not catastrophic
- NDR 100–110%: Good; existing customers are expanding
- NDR > 120%: Exceptional; common in best-in-class PLG and enterprise SaaS

### What breaks

**Defining the measurement period inconsistently.** Monthly churn and annual churn are not directly comparable. Monthly churn of 2% compounds to ~22% annual churn — dramatically different from a company reporting "2% churn" that turns out to mean annual. Always specify the period.

**Using the wrong denominator.** If you measure churn against end-of-period customers rather than start-of-period customers, you can make high-churn months look better (you lost customers, so the denominator shrank). Use start-of-period consistently.

**Conflating logo and revenue churn.** A business that is losing many small customers but retaining large ones will show good NRR and alarming logo churn simultaneously. Each tells a different story. A business that replaces churned customers with larger ones may be inadvertently concentrating revenue risk.

---

## 11. Conversion Funnel Analysis

### The problem

Users move through a sequence of steps to complete a goal — sign up, activate, purchase, upgrade. At every step, some fraction drop off. You need to know which step is the bottleneck, by how much, and why.

### The core insight

A funnel is only as strong as its weakest step. A 10% improvement at the step with the largest absolute drop-off is almost always more valuable than a 10% improvement anywhere else. Funnel analysis tells you where to focus.

### The mechanics

**Step 1: Define the funnel steps explicitly.** What are the ordered actions a user must take? For a checkout funnel: product page view → add to cart → begin checkout → enter payment → confirm purchase.

**Step 2: Measure conversion rates at each step.**

```
Step conversion rate = Users who complete step N+1 / Users who complete step N
```

**Step 3: Measure absolute drop-off at each step.**

```
Absolute drop-off at step N = Users who complete step N − Users who complete step N+1
```

The step with the largest absolute drop-off is the highest-leverage optimization target.

**Step 4: Segment the funnel.** Do different acquisition channels, geographies, device types, or user segments have different conversion rates at specific steps? A mobile funnel may collapse at payment entry (small keyboard, card entry friction) while a desktop funnel does not. Segmentation reveals where the friction is structural.

**Step 5: Time-in-step analysis.** For multi-session funnels, how long do users spend at each step? A step where users spend a long time before dropping off suggests consideration or confusion, not just disinterest.

### What breaks

**Non-linear funnels.** If users can skip steps, revisit steps, or enter from multiple points, a linear funnel model is wrong. Map the actual user flow first (using session replay, path analysis) before assuming linearity.

**Attribution errors.** A user who starts checkout on mobile and completes on desktop may appear as two separate funnel starts. Without cross-device identity resolution, you misattribute drop-off to the mobile payment step when none actually occurred.

**Optimizing a local funnel metric at the expense of downstream quality.** Simplifying checkout to remove friction can increase conversion rate while attracting more fraudulent transactions or reducing purchase intent quality. Always check downstream metrics (refund rate, order completion, 30-day repurchase) after funnel optimization.

---

## 12. Anomaly Detection in Metrics

### The problem

Metrics move every day. Most of that movement is noise — random variation, day-of-week effects, seasonal patterns. But sometimes a metric drop is a real signal: a bug, an infrastructure outage, a competitor action, or a product regression. Distinguishing signal from noise is the anomaly detection problem.

### The core insight

An anomaly is a data point that is unlikely under the model of normal variation. The challenge is that "normal variation" is itself time-varying, seasonal, and correlated across metrics. A simple threshold ("alert if DAU drops more than 5%") will fire constantly on normal week-to-week variation and miss slow-developing regressions.

### The mechanics

**Baseline construction:**

For any metric, establish a baseline that accounts for:
- **Seasonality:** Day of week, time of year, holidays
- **Trend:** Underlying growth or decay direction
- **Autocorrelation:** Today's value is correlated with yesterday's

A simple baseline: for each day-of-week, compute the trailing 4-week average of the same weekday. The expected value for "this Tuesday" is the average of the past 4 Tuesdays.

**Anomaly thresholds:**

Alert when the observed value deviates from the baseline by more than k standard deviations (using the historical std of day-of-week deviations). k = 2 catches ~95% of real anomalies with manageable false positive rate for most business metrics.

**More sophisticated approaches:**

- **Seasonal decomposition (STL):** Decompose the time series into trend + seasonal + residual. Anomalies are unusual residuals.
- **Prophet (Facebook):** Handles multiple seasonality, holidays, and trend changepoints. Good default for business time series with weekly and annual patterns.
- **CUSUM (Cumulative Sum):** Detects persistent directional shifts (slow regressions) rather than single-day spikes. Appropriate when you care about "the metric has been gradually declining for 10 days" rather than "the metric dropped sharply today."

**Root cause investigation protocol:**

When an anomaly is detected:
1. Segment the metric by all major dimensions (platform, country, user segment, acquisition channel). Localized anomalies point to root causes; broad anomalies suggest infrastructure or product-wide issues.
2. Correlate timing with deployments, infrastructure events, or external events (holidays, competitor launches).
3. Check correlated metrics: if DAU drops, does session length also drop? (engagement problem) or does session count drop while length is unchanged? (acquisition/notification problem)

### What breaks

**Alert fatigue.** If your anomaly detection fires too frequently on noise, on-call engineers stop reading the alerts. Calibrate false positive rates deliberately — it is better to miss some anomalies than to make alerts meaningless.

**Assuming anomalies are always bad.** A sudden spike in signups might be anomalous and positive. Anomaly detection should flag unusual events for investigation, not automatically trigger panic.

**Ignoring seasonality in the baseline.** A metric drop on New Year's Day is not an anomaly. Failing to account for known seasonal patterns produces alerts on every holiday. Always build seasonal decomposition into your baseline model before computing residuals.

---

## 13. Counter-Metrics and Guardrails

### The problem

Every metric you optimize can be gamed. Not necessarily by malicious actors — often the optimization pressure itself causes well-intentioned teams to find ways to move the metric that do not correspond to real improvement. Guardrail metrics are the defense against this.

### The core insight

Every target metric has a shadow metric: a correlated observable that goes in the wrong direction if you improve the target for bad reasons. If you want to maximize DAU, the shadow metric is spam notification volume. If you want to maximize MRR, the shadow metric is refund rate. Guardrails are formalized shadow metrics.

### The mechanics

**For each primary metric you optimize, define at least one counter-metric** that should not degrade:

| Primary metric | Potential gaming | Counter-metric |
| :--- | :--- | :--- |
| DAU | Aggressive notifications | Notification opt-out rate, app uninstall rate |
| Session length | Confusion (users can't find what they want) | Task completion rate, search abandonment |
| MRR | Aggressive discounting | Gross margin, average discount depth |
| Conversion rate | Removing friction that screens bad customers | Refund rate, 30-day repurchase rate |
| NPS | Only surveying happy customers | Survey response rate, sampling methodology |
| Onboarding completion | Making onboarding so short it is meaningless | Day-7 engagement after onboarding |

**The guardrail contract:** A team's experiment or feature ships only if it improves the target metric without statistically significant degradation of the guardrails. This is enforced at the experiment evaluation stage, not as an afterthought.

### What breaks

**Setting guardrail thresholds too loosely.** If you allow a 10% degradation of refund rate as acceptable, you are not actually guarding against anything. Guardrail thresholds should be set based on business risk, not based on what is easy to pass.

**Not having enough guardrails.** A team with one guardrail will find the one unguarded dimension to game. Coverage matters.

**Treating all guardrail violations as blockers equally.** A 0.1% degradation of p99 latency is different from a 20% degradation. Define criticality levels and response protocols.

---

## 14. Goodhart's Law

### The problem

You set a metric as a target. The team optimizes for it. The metric improves. But the underlying thing you cared about does not. This is not a failure of execution — it is a structural property of measurement.

### The core insight

**Goodhart's Law:** "When a measure becomes a target, it ceases to be a good measure."

The reason is that any metric is a proxy for the thing you actually care about. The correlation between the proxy and the underlying value holds in normal conditions. When you apply optimization pressure to the proxy, you find and exploit the gap between the proxy and the real thing.

Examples:

- **Lines of code as a productivity metric:** Engineers write more, less efficient code.
- **NPS as a customer success metric:** Customer success teams cherry-pick survey timing (just after a successful support resolution) to inflate scores.
- **Story points as a velocity metric:** Teams inflate point estimates to hit velocity targets.
- **DAU as a product health metric:** Teams add dark patterns (guilt-trip notifications, artificial reengagement prompts) to inflate daily opens.
- **Time-on-site as an engagement metric:** Confusing interfaces, autoplay content, and infinite scroll inflate time-on-site without delivering value.

### How to resist Goodhart's Law

1. **Rotate metrics periodically.** If teams know the metric will change, they are less likely to build gaming mechanisms into the product.
2. **Use multiple metrics simultaneously.** It is harder to simultaneously game five metrics in the same direction.
3. **Pair every target metric with a counter-metric** (see Section 13).
4. **Distinguish "metric is improving" from "the underlying thing is improving."** Always ask: could this metric move in this direction for a bad reason? What would that look like in the data?
5. **Do qualitative audits.** User interviews, session replays, and support ticket analysis reveal whether metric improvements correspond to actual user experience improvements.

---

## 15. OKR vs. KPI

### The problem

Organizations need two related but distinct things from their measurement systems: a way to set and track ambitious goals (goal-setting), and a way to monitor the ongoing health of the business (monitoring). These are different jobs, and conflating them produces dysfunction.

### KPIs: monitoring business health

**What question KPIs answer:** "Is the business operating normally?"

KPIs (Key Performance Indicators) are the vital signs of the business. They are not goals — they are ongoing measurements of whether key processes are working. They should be tracked continuously, they should have known baselines and acceptable ranges, and they should trigger investigation when they fall outside those ranges.

Examples: Monthly churn rate, DAU, gross margin, support ticket volume, p99 latency.

A KPI is not a stretch target. If your KPI has a target of "+20% by Q4," it has been misclassified — that is an objective, not a health indicator.

### OKRs: setting ambitious goals

**What question OKRs answer:** "What are we trying to accomplish this quarter, and how will we know if we succeeded?"

OKRs (Objectives and Key Results) are a goal-setting framework:

- **Objective:** Qualitative statement of the desired outcome. Ambitious, motivating, memorable. "Make checkout so frictionless that paying feels effortless."
- **Key Results:** Two to four quantitative measures that, if achieved, would demonstrate the objective was reached. "Reduce checkout drop-off by 25%; reduce average time-to-purchase by 40%; increase mobile conversion rate from 2.1% to 3.5%."

OKRs are designed to be somewhat uncomfortable — if you consistently hit 100% of your OKRs, your OKRs are not ambitious enough. Google's framework targets 70% achievement as a sign of appropriate stretch.

### The distinction in practice

| Dimension | KPI | OKR |
| :--- | :--- | :--- |
| Purpose | Monitor ongoing health | Set and track ambitious goals |
| Time horizon | Continuous | Quarterly / annual |
| Achievement | Healthy = within normal range | Ambitious = 70% achievement is good |
| Response to missing target | Investigate cause | Reflect on what to do differently |
| Relationship to strategy | Confirms stability | Drives direction |

### What breaks

**Treating OKRs as KPIs:** Setting an OKR target at the level of "what we expect to happen anyway" makes OKRs into scheduled status reports, not goal-setting tools. OKRs should require new thinking, not just execution of the existing roadmap.

**Treating KPIs as OKRs:** "Increase churn from 3% to 2.5% by Q4" sounds like a KPI turned into a target. The problem is that a KPI with a target is subject to Goodhart's Law — teams will find ways to move the number rather than addressing the underlying retention problem.

**Too many OKRs.** Three to five key results per objective, two to three objectives per team, per quarter. More than this dilutes focus. The purpose of OKRs is forced prioritization, not comprehensive coverage.

**Disconnected OKRs across teams.** An OKR framework only produces company-wide coordination if team-level OKRs cascade from and support company-level OKRs. If every team sets its OKRs independently, you get local optimization with no coherent direction.

---

## Quick Reference: When to Use Which Metric

| Question | Primary metric | Watch out for |
| :--- | :--- | :--- |
| Is product usage growing? | DAU/WAU/MAU trend | Bot traffic, notification spam |
| Are users habituated? | DAU/MAU stickiness | Wrong for low-frequency products |
| Are users retained? | D7/D28/D90 retention, L28 | "Active" definition, survivorship bias |
| Do I have PMF? | Retention curve shape (does it flatten?) | Aggregate vs. segment curves |
| Are early users activating? | D1 retention, activation rate | True "aha moment" vs. account creation |
| How large is recurring revenue? | MRR, ARR | One-time fees included by mistake |
| How much does a customer generate? | ARPU, ARPPU | Averages hiding segment differences |
| Can I afford to acquire customers? | LTV:CAC, payback period | LTV projection using wrong churn model |
| Are we losing revenue from existing customers? | GRR, NDR | Period definition, expansion included/excluded |
| Where is the funnel leaking? | Step conversion rates, absolute drop-off | Non-linear flows, cross-device attribution |
| Is a metric move real or noise? | Anomaly detection with seasonal baseline | Alert fatigue, seasonality not modeled |
| Are we improving the right thing? | Primary metric + counter-metrics | Goodhart's Law |
| What are we trying to achieve? | OKRs | OKRs set too conservatively |
| Is the business operating normally? | KPIs | KPIs turned into targets |

## Flashcards

**Some metrics measure what the team does (inputs). These are controllable, fast-moving, and leading.?** #flashcard
Some metrics measure what the team does (inputs). These are controllable, fast-moving, and leading.

**Some metrics measure what the business gets (outputs). These are the actual outcomes?** #flashcard
slower-moving, harder to game, harder to move directly.

**Some metrics exist only to detect harm (guardrails). These should not degrade when you optimize the others.?** #flashcard
Some metrics exist only to detect harm (guardrails). These should not degrade when you optimize the others.

**D1 retention?** #flashcard
Percent of users who return on day 2 after signup. Measures first impression and activation quality.

**D7 retention?** #flashcard
Percent of users active in week 1 who return in week 2. Measures early habit formation.

**D28 retention?** #flashcard
Percent of users still active at one month. Measures whether the product has become part of a routine.

**D90 retention?** #flashcard
Three-month retention. For consumer apps, this approaches the "retained vs churned" bifurcation point.

**DAU (Daily Active Users)?** #flashcard
Unique users who performed a meaningful action on a given day. The highest-frequency measure of product health.

**WAU (Weekly Active Users)?** #flashcard
Unique users active in a 7-day window. Smooths day-of-week effects. More appropriate as a primary metric for products used on a work-week cadence.

**MAU (Monthly Active Users)?** #flashcard
Unique users active in a 30-day window. The standard denominator in DAU/MAU and the most commonly reported investor metric.

**This formula assumes constant churn. Real churn is often front-loaded (many users churn in the first 1–3 months; survivors are much stickier). The formula will underestimate LTV for products with strong early churn and flat later churn.?** #flashcard
This formula assumes constant churn. Real churn is often front-loaded (many users churn in the first 1–3 months; survivors are much stickier). The formula will underestimate LTV for products with strong early churn and flat later churn.

**It uses average ARPU, which hides segment heterogeneity. Enterprise customers have radically different LTV than SMB customers.?** #flashcard
It uses average ARPU, which hides segment heterogeneity. Enterprise customers have radically different LTV than SMB customers.

**It uses gross margin?** #flashcard
if you use revenue instead, you overstate LTV by the reciprocal of gross margin.

**Cohort LTV requires enough historical data (ideally 24+ months) to observe the curve shape. Early-stage companies cannot yet calculate this accurately.?** #flashcard
Cohort LTV requires enough historical data (ideally 24+ months) to observe the curve shape. Early-stage companies cannot yet calculate this accurately.

**Product and pricing changes affect cohort survival. A cohort acquired under a different pricing model has a different LTV than current cohorts. Use only comparably acquired cohorts.?** #flashcard
Product and pricing changes affect cohort survival. A cohort acquired under a different pricing model has a different LTV than current cohorts. Use only comparably acquired cohorts.

**Projecting survival curves forward requires assuming the current curve shape continues. If you are actively improving retention, historical curves understate future LTV.?** #flashcard
Projecting survival curves forward requires assuming the current curve shape continues. If you are actively improving retention, historical curves understate future LTV.

**LTV:CAC < 1?** #flashcard
You are destroying value on every customer acquired

**LTV:CAC 1–3?** #flashcard
Marginal; viable only with very fast payback periods

**LTV:CAC 3?** #flashcard
The traditional "healthy SaaS" benchmark

**LTV:CAC > 5?** #flashcard
Often a signal to invest more aggressively in acquisition (you are leaving money on the table)

**NDR < 85%?** #flashcard
Serious retention problem; growth requires massive acquisition to offset losses

**NDR 85–100%?** #flashcard
Adequate; growth requires new customers but churn is not catastrophic

**NDR 100–110%?** #flashcard
Good; existing customers are expanding

**NDR > 120%?** #flashcard
Exceptional; common in best-in-class PLG and enterprise SaaS

**Seasonality?** #flashcard
Day of week, time of year, holidays

**Trend?** #flashcard
Underlying growth or decay direction

**Autocorrelation?** #flashcard
Today's value is correlated with yesterday's

**Seasonal decomposition (STL)?** #flashcard
Decompose the time series into trend + seasonal + residual. Anomalies are unusual residuals.

**Prophet (Facebook)?** #flashcard
Handles multiple seasonality, holidays, and trend changepoints. Good default for business time series with weekly and annual patterns.

**CUSUM (Cumulative Sum)?** #flashcard
Detects persistent directional shifts (slow regressions) rather than single-day spikes. Appropriate when you care about "the metric has been gradually declining for 10 days" rather than "the metric dropped sharply today."

**Lines of code as a productivity metric?** #flashcard
Engineers write more, less efficient code.

**NPS as a customer success metric?** #flashcard
Customer success teams cherry-pick survey timing (just after a successful support resolution) to inflate scores.

**Story points as a velocity metric?** #flashcard
Teams inflate point estimates to hit velocity targets.

**DAU as a product health metric?** #flashcard
Teams add dark patterns (guilt-trip notifications, artificial reengagement prompts) to inflate daily opens.

**Time-on-site as an engagement metric?** #flashcard
Confusing interfaces, autoplay content, and infinite scroll inflate time-on-site without delivering value.

**Objective?** #flashcard
Qualitative statement of the desired outcome. Ambitious, motivating, memorable. "Make checkout so frictionless that paying feels effortless."

**Key Results?** #flashcard
Two to four quantitative measures that, if achieved, would demonstrate the objective was reached. "Reduce checkout drop-off by 25%; reduce average time-to-purchase by 40%; increase mobile conversion rate from 2.1% to 3.5%."
