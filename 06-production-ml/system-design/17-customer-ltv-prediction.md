---
module: Production Ml
topic: System Design
subtopic: Customer Ltv Prediction
status: unread
tags: [productionml, ml, system-design-customer-ltv-pre]
---
# Customer Lifetime Value (LTV) Prediction System Design

End-to-end ML system for predicting customer lifetime value at e-commerce, subscription, and fintech scale. Common system design question at companies running performance marketing, CRM, and growth programs (Shopify, Airbnb, DoorDash, Stripe, Netflix).

**Scale:** 100M+ customers, daily batch scoring, real-time scoring for new users at acquisition, LTV predictions feeding paid acquisition bidding.

---

## 1. Problem Framing

### Clarifying Questions

- **Business model?** Subscription (contractual) vs e-commerce/marketplace (non-contractual) — different statistical models apply.
- **Prediction horizon?** 90-day LTV (tactical, acquisition bidding) vs 1-year LTV (strategic, cohort planning) vs lifetime (theoretical).
- **Customer population?** New users at acquisition (cold start) vs existing customers (full history).
- **Primary use case?**
  - Acquisition bidding: bid = predicted LTV × margin − CAC → needs calibrated expected value
  - Retention/churn: who to target with discount/win-back → needs relative ranking
  - Upsell/pricing: what offer to show → needs segment-level calibration
  - Financial planning: cohort revenue forecasting → needs aggregate accuracy, not individual
- **Revenue definition?** Gross, net (after refunds), or contribution margin?
- **Latency requirement?** Acquisition bidding needs <200ms; CRM batch jobs can run nightly.
- **Feedback delay?** How long before realized revenue is observed (weeks to years)?

### Metric Trade-offs

| Metric | What it measures | Use case |
|---|---|---|
| MAE / RMSE on individual predictions | Point accuracy | Not recommended — LTV is inherently uncertain |
| Calibration (expected = actual) at segment level | Bid accuracy | Acquisition bidding — undercalibrated → overbid |
| Spearman rank correlation | Ranking quality | Retention targeting |
| MAPE at cohort level | Aggregate accuracy | Financial planning |
| Gini / AUC of cumulative revenue | Top-decile concentration | "Top 20% of customers generate 80% of revenue" |

**Calibration matters more than RMSE for bidding.** If predicted LTV = $120 but actual = $80, you overbid by 50% and erode margin. Aggregate calibration at the channel × cohort level matters more than individual accuracy.

```
Optimal CPA bid = E[LTV | channel, device, cohort] × gross_margin_rate − target_ROAS_floor
```

LTV distributions are heavily right-skewed (Pareto-like), so RMSE is dominated by the top 1% of customers — a model predicting the mean for everyone scores well on RMSE but has zero discriminative value.

---

## 2. LTV Definitions

**Historical LTV (realized):** `Σ revenue(t_i)` for all transactions up to time t. Simple, used as training labels, but not predictive on its own.

**Predicted LTV (discounted future revenue):**

```
predicted_LTV(customer, t, horizon) = E[ Σ_{τ=t}^{t+horizon} r_τ × (1+d)^{-(τ-t)} ]
```

Expectation is over the joint distribution of purchase occurrence and purchase amount.

### Contractual vs Non-Contractual

| Setting | "Active" defined by | Examples | Model |
|---|---|---|---|
| Contractual | Explicit cancellation | SaaS, subscription boxes, insurance | Survival analysis, DCF on MRR |
| Non-contractual | Latent — customer just stops buying | E-commerce, ride-share, delivery | BG/NBD, probabilistic models |

In non-contractual settings you never observe "the last purchase" — a customer who bought 6 months ago might be churned or just in a long gap. This is the central modeling challenge.

### The Censoring Problem

Training data is right-censored: customers acquired recently have shorter observation windows and thus less observed revenue, regardless of true LTV.

**Naive (wrong):** train directly on historical_LTV → model learns "new = low LTV" → underestimates new acquisitions.

**Correct:** model the generative process (purchase probability × purchase value × repeat rate), then integrate over the horizon using survival/hazard functions.

---

## 3. System Architecture

```
Customer Touch-points (purchases, sessions, support, app opens)
  │
  ▼
Event Stream (Kafka / Kinesis)
  │
  ├──► Real-time Feature Computation (Flink)
  └──► Batch Feature Engineering (Spark/dbt, daily)
          │
          ▼
     Feature Store (Redis / Feast) ◄── cold-start priors (channel, device, geo)
          │
          ├──► Probabilistic Model (BG/NBD + Gamma-Gamma)
          └──► ML Model (GBM / sequence / multi-task)
                  │
                  ▼
          Ensemble / Calibration Layer ◄── segment-level isotonic regression
                  │
          ┌───────┴────────┐
          ▼                ▼
   Real-time API      Batch Scoring
   (<200ms, new       (existing users,
   user acquisition)   nightly)
          │                │
          ▼                ▼
   Downstream consumers: value-based bidding, CRM/retention,
   dynamic pricing, financial planning, cohort analytics
          │
          ▼
   Monitoring / Feedback Loop ◄── delayed actual revenue labels
```

---

## 4. Probabilistic Models: BG/NBD and Gamma-Gamma

### Why Probabilistic Models for Sparse Data

For customers with only 1–3 purchases, gradient boosting overfits to noise. Probabilistic models impose structural priors from population behavior:

- Make principled predictions from a single purchase event
- Return a distribution over future purchases, not just a point estimate
- Parameters (β, r, α, a, b) are business-interpretable
- Need only recency, frequency, tenure — no feature engineering

### BG/NBD Model (Fader, Hardie, Lee, 2005)

The "Buy-Till-You-Die" model for non-contractual settings. Two latent processes:

1. **Purchase process (while active):** Poisson process with individual rate λ; population λ ~ Gamma(r, α).
2. **Dropout process:** after each purchase, customer "dies" with probability p; population p ~ Beta(a, b).

```
P(alive | x, t_x, T) = P(alive) / [P(alive) + P(died after last purchase)]

E[X(t) | x, t_x, T] = (r + x)(α + T) / (α + T + t) × P(alive | ...) × [conditional expected transactions if alive]
```

where `x` = observed transactions in `[0,T]`, `t_x` = recency, `T` = customer age.

```python
from lifetimes import BetaGeoFitter

bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(frequency=df['frequency'], recency=df['recency'], T=df['T'])

df['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    t=52, frequency=df['frequency'], recency=df['recency'], T=df['T']
)
```

### Gamma-Gamma Model for Monetary Value

Assumes average transaction value is independent of purchase frequency (empirically valid). Models the population distribution of mean transaction value.

```python
from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(df['frequency'], df['avg_monetary_value'])

ltv = ggf.customer_lifetime_value(
    bgf, df['frequency'], df['recency'], df['T'], df['avg_monetary_value'],
    time=12, discount_rate=0.01
)
```

**Pareto/NBD** is BG/NBD's predecessor — dropout can occur any time, not just post-purchase. Roughly equivalent results but costlier to compute; BG/NBD is preferred in practice.

### When to Use Probabilistic vs ML Models

| Condition | Prefer Probabilistic | Prefer ML |
|---|---|---|
| Customer history depth | <5 purchases | 10+ purchases |
| Feature richness | Only RFM available | Rich behavioral/contextual features |
| Population size | Small (1K–100K) | Large (1M+) |
| Interpretability requirement | High | Flexible |
| Cold start | BG/NBD with priors | Not applicable |
| Macro covariates (seasonality) | Limited | Strong |

---

## 5. ML Approaches

### 5.1 Gradient Boosting on RFM Features

Most practical approach for e-commerce with >1M customers and rich features.

```python
# Regress on log(1 + LTV) to handle the right-skewed distribution.
# Label = revenue in 90 days after a fixed cutoff, using customers
# with >=12 months history so the label is fully observed.

df['ltv_90d'] = df['revenue_post_cutoff'].clip(upper=percentile_99)
df['log_ltv'] = np.log1p(df['ltv_90d'])

model = LGBMRegressor(
    objective='regression', n_estimators=500, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=50
)
model.fit(X_train, y_train['log_ltv'])
predictions = np.expm1(model.predict(X_test))
```

Alternatives to log-transform for skew: Tweedie regression (models zero-inflation natively), two-stage classify-then-regress, or quantile regression for median LTV.

### 5.2 Sequence Models on Purchase History

For customers with rich transaction sequences, an LSTM/Transformer over ordered purchases captures temporal patterns that RFM aggregates miss. Input: sequence of (amount, category, channel, days_since_prev); output: predicted LTV over next 90 days.

```python
class LTVTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.item_embed = nn.Embedding(num_categories, d_model)
        self.amount_proj = nn.Linear(1, d_model)
        self.time_embed = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, token_ids, amounts, time_gaps, mask):
        x = self.item_embed(token_ids) + self.amount_proj(amounts) + self.time_embed(time_gaps)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.head(x[:, -1, :])
```

Requires >5 events of history and is more expensive to serve — use for high-value segments, fall back to GBM for sparse users.

### 5.3 Survival Analysis

Frames churn as time-to-event, natural for non-contractual settings.

**Cox Proportional Hazards:** `h(t|x) = h_0(t) × exp(β^T x)` — gives hazard ratio, not directly usable for LTV without integrating the survival function.

**Weibull AFT:** `log(T) = β^T x + σε` — more interpretable, features directly scale time-to-churn.

```python
from lifelines import WeibullAFTFitter

aft = WeibullAFTFitter()
aft.fit(df, duration_col='days_to_churn', event_col='churned')

survival_prob_90 = aft.predict_survival_function(df).loc[90]
expected_revenue = df['avg_daily_revenue'] * aft.predict_expectation(df).clip(upper=90)
```

### 5.4 Multi-Task Learning

Predict churn probability, purchase frequency, and average order value jointly. Shared representations improve sample efficiency for sparse customers.

```python
class MultiTaskLTV(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.churn_head = nn.Linear(128, 1)
        self.frequency_head = nn.Linear(128, 1)
        self.aov_head = nn.Linear(128, 1)

    def forward(self, x):
        shared = self.shared(x)
        return {
            'churn_prob': torch.sigmoid(self.churn_head(shared)),
            'frequency': F.softplus(self.frequency_head(shared)),
            'log_aov': self.aov_head(shared)
        }

    def predict_ltv(self, x, horizon_days):
        out = self.forward(x)
        p_active = 1 - out['churn_prob']
        expected_purchases = out['frequency'] * horizon_days / 30 * p_active
        return expected_purchases * torch.exp(out['log_aov'])
```

---

## 6. Feature Engineering

### RFM Features (baseline, always include)

| Feature | Definition | Notes |
|---|---|---|
| `recency_days` | Days since last purchase | Log-transform; non-linear effect |
| `frequency` | Distinct purchase days | Avoids multi-item order inflation |
| `monetary_total` | Cumulative net revenue | Use net (after returns) |
| `monetary_avg` | Average order value | Clip at 99th percentile |
| `frequency_30d/90d/365d` | Multi-window purchase counts | Captures acceleration/deceleration |
| `recency_trend` | frequency_30d / frequency_90d | Increasing engagement = positive signal |
| `days_since_first_purchase` | Customer age | Longer tenure = more predictable |

### Behavioral and Product Features

| Feature | Signal |
|---|---|
| Browse-to-purchase ratio | High browse, no purchase = lower intent |
| Cart abandonment rate | Price sensitivity |
| Category breadth | Multi-category = higher retention |
| Return rate | Negatively predicts net LTV |
| App vs web ratio | App users typically higher LTV |
| Subscription upgrade flag | Strong positive signal |
| Payment method | Credit card users often higher LTV than BNPL |
| Coupon dependency | Discount-only purchasers have lower organic LTV |

### Cohort and Channel Features

Individual-level models miss cohort effects — add cohort-level features (cohort avg LTV, cohort retention rate, channel LTV index) to control for them.

For acquisition bidding, LTV must condition on channel since acquisition source correlates strongly with eventual value (e.g., organic search and referral customers often outperform paid prospecting).

---

## 7. Handling Censoring

Customers acquired recently are right-censored — we only observe LTV up to the present.

**Naive (wrong):** label = total revenue to date → model learns "time since acquisition" as the dominant feature → all new customers predicted low LTV.

**Correct — time-conditioned labels:** revenue over a fixed future window from a fixed cutoff date.

```python
CUTOFF_DATE = "2024-01-01"
HORIZON_DAYS = 90

# Keep only customers acquired >= HORIZON_DAYS before cutoff (label fully observed)
# and with a purchase before cutoff (avoid trivial zeros).
df_train = df[
    (df['acquisition_date'] <= CUTOFF_DATE - timedelta(days=HORIZON_DAYS)) &
    (df['first_purchase_date'] < CUTOFF_DATE)
].copy()
df_train['ltv_label'] = df_train['revenue_cutoff_to_cutoff_plus_90d']
```

Use several historical cutoff dates and pool cohorts to increase training data and reduce temporal overfitting.

**Survival framing** handles censoring natively — "still active at last observation" is treated as censored, not churned, avoiding selection bias toward only fully-observed (churned) customers.

```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(durations=df['tenure_days'], event_observed=df['has_churned'])
expected_remaining = kmf.median_survival_time_ - T
```

At serving time, condition the prediction on being active:

```
LTV(t, t+horizon | active at t) = E[revenue | x, recency, T] × P(active through t+horizon | x, T)
```

---

## 8. Segment-Level Calibration

A model can be well-calibrated in aggregate but badly calibrated within segments. For acquisition bidding you bid per channel/audience — if LTV is overestimated 40% for one channel, you overbid systematically there.

```python
# Step 1: train base model (GBM or BG/NBD)
# Step 2: compute calibration error by segment
segments = ['channel', 'device', 'acquisition_cohort_month', 'geo_tier']
for segment_col in segments:
    for segment_val in df[segment_col].unique():
        mask = df[segment_col] == segment_val
        calibration_error[segment_col][segment_val] = (
            df.loc[mask, 'actual_ltv_90d'].mean() / df.loc[mask, 'predicted_ltv'].mean()
        )

# Step 3: fit isotonic regression per segment
from sklearn.isotonic import IsotonicRegression

calibrators = {}
for channel in channels:
    mask = val_df['channel'] == channel
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(val_df.loc[mask, 'predicted_ltv'], val_df.loc[mask, 'actual_ltv_90d'])
    calibrators[channel] = ir

def calibrated_ltv(raw_pred, channel):
    return calibrators.get(channel, default_calibrator).predict([raw_pred])[0]
```

### Acquisition Bidding Integration

LTV prediction feeds ad platforms (Google/Meta value-based bidding) as the conversion value:

```python
def on_new_acquisition(user_id, channel, device, features):
    ltv_pred = ltv_model.predict(features)
    calibrated = calibrated_ltv(ltv_pred, channel)
    send_conversion_event(user_id=user_id, conversion_value=calibrated * GROSS_MARGIN_RATE, currency='USD')
```

```
Max CPA = E[LTV | channel, device, audience] × margin_rate / target_ROAS

Example: E[LTV|FB prospecting]=$45, margin=0.35, target_ROAS=2.0 → Max CPA = $7.88
```

When a customer touches multiple channels before converting, SHAP values on the LTV model can attribute LTV credit across channels for multi-touch attribution.

---

## 9. Serving and Downstream Integration

### Real-Time Scoring: New Users at Acquisition

New users have no purchase history — predict from channel/campaign, device, geo, landing page, referrer/UTM, time of day.

```python
# Separate "acquisition-time" model trained only on signup-available features
acquisition_model = LGBMRegressor(...)
acquisition_model.fit(X_acquisition_features, y_ltv_90d)
# Served via low-latency API on the acquisition pixel fire, P99 target ~50ms
```

```
New visit → tracking pixel → serverless function
    → feature lookup from Redis (channel/geo priors)
    → model inference (pre-loaded in memory)
    → LTV score → conversion event → Google/Meta Ads API
```

### Batch Scoring: Existing Customers

```
Nightly Spark job → pull features from warehouse
    → apply BG/NBD + Gamma-Gamma (or GBM) → write scores to feature store
    → downstream consumers read from feature store
```

### Downstream Consumers

| Consumer | LTV Usage | Update Frequency |
|---|---|---|
| CRM / Marketing | Segment by LTV tier for targeting | Daily |
| Retention team | Flag high-LTV customers with churn signals | Daily |
| Pricing engine | Discounts inversely proportional to LTV | Real-time |
| Customer success | Prioritize support queues | Real-time |
| Finance | Cohort LTV curves for forecasting | Weekly |
| Product | A/B test measurement | Per-experiment |

---

## 10. Model Monitoring and Feedback Loop

**Prediction quality (lagged, observable after horizon):**

```python
monitoring_df = predictions_made_90d_ago.merge(realized_revenue, on='customer_id')

metrics = {
    'overall_calibration': monitoring_df['actual_ltv'].mean() / monitoring_df['predicted_ltv'].mean(),
    'channel_calibration': monitoring_df.groupby('channel').apply(
        lambda g: g['actual_ltv'].mean() / g['predicted_ltv'].mean()),
    'spearman_rank_corr': monitoring_df[['predicted_ltv', 'actual_ltv']].corr('spearman').iloc[0, 1],
    'top_decile_capture': (
        monitoring_df.nlargest(n // 10, 'predicted_ltv')['actual_ltv'].sum() /
        monitoring_df.nlargest(n // 10, 'actual_ltv')['actual_ltv'].sum())
}
```

**Distributional drift (observable immediately):** KS-test on prediction distribution vs rolling baseline, RFM feature drift, cohort purchase-rate changes.

**Retraining triggers:**

| Trigger | Action |
|---|---|
| Channel calibration error > 15% | Retrain calibration layer on recent cohorts |
| Feature drift p-value < 0.01 | Full model retrain |
| Major macro event | Emergency reweighting of recent cohorts |
| Seasonal product mix shift | Add seasonal features, retrain |

---

## 11. Failure Modes

**Selection bias in training data.** Model trained only on customers who purchased at least once, but scored on all new signups, many of whom never convert. Fix: two-stage model — P(converts) × LTV(if converts).

```python
expected_ltv = conversion_model.predict_proba(X)[:, 1] * ltv_model.predict(X)
```

**Simpson's paradox across cohorts.** Newer cohorts show lower observed LTV purely from censoring, which can look like "LTV is declining." Fix: always include `tenure_at_prediction_time` as a feature, and use multiple historical cutoffs so the model doesn't learn cohort recency as a proxy for maturity.

**Overfitting to historical regimes.** Model trained on a bull-market period mis-predicts after conditions normalize. Fix: include macro signals, time-decay weight recent cohorts, monitor cohort calibration continuously.

**Macro shocks (COVID, recession).** Pre-shock models mis-predict badly post-shock. Fix: maintain models trained on different time windows, fall back to short-window behavior during detected shocks, add macro index features that can shift predictions without full retraining.

**Customer gaming.** High-value customers may deliberately reduce engagement to appear low-value and receive win-back offers. Fix: don't expose LTV tiers to customers; use LTV for external bidding rather than as the sole driver of internal discount logic; watch for anomalous engagement drop patterns.

**Cold-start overconfidence.** Acquisition-time predictions have high uncertainty but are often served as point estimates, causing systematic overbidding on low-data segments. Fix: output prediction intervals and bid on a conservative percentile when segment sample size is small.

```python
def bid_ltv(predicted_ltv, prediction_std, segment_n):
    confidence_discount = max(0.7, 1 - 2 / np.sqrt(segment_n))
    return predicted_ltv - 0.5 * prediction_std * (1 - confidence_discount)
```

---

## 12. Interview Questions and Answers

**Q1: Why not just use historical average LTV as the prediction?**

Historical LTV is censored — churned customers have low historical LTV, while active customers are right-censored and still adding revenue. Using the raw average systematically underestimates true expected LTV for active customers. You need a model that conditions on current engagement and projects forward.

**Q2: BG/NBD assumes purchase rate and churn are independent. Is that realistic?**

Frequently violated — customers who buy more often tend to be more engaged and less likely to churn. Options: (1) accept the approximation, since BG/NBD is empirically fairly robust; (2) use a correlated latent variable model; (3) layer ML features (session depth, support tickets) on top to capture engagement signal BG/NBD misses.

**Q3: LTV predictions used for bidding create a feedback loop — how do you handle it?**

Bidding more on high-predicted-LTV customers changes who gets acquired, which shifts training data distribution and can compound model bias. Mitigations: exploration (occasionally bid regardless of prediction to keep an unbiased sample), channel-level holdouts for causal identification, and doubly-robust estimators to correct for acquisition propensity.

**Q4: Product wants LTV as the primary A/B test metric. Concerns?**

(1) **Label delay** — LTV takes 90+ days to realize, slowing iteration; use validated 30-day surrogate metrics instead. (2) **Variance** — LTV is Pareto-distributed, so tests need far more samples than conversion-rate tests. (3) **Model dependence** — the LTV model was trained on pre-change behavior and may not generalize to a new experience. Prefer realized revenue when timelines allow.

**Q5: How does LTV prediction differ for subscription vs e-commerce?**

Subscriptions (contractual): churn is an explicit event, so LTV ≈ MRR / churn_rate, or survival analysis for more precision — the hard part is predicting upgrades/downgrades. E-commerce (non-contractual): churn is latent, so BG/NBD or survival analysis is needed to estimate "are they still active?" A hybrid business needs separate models or a contractual-indicator feature.

**Q6: How would you detect that your LTV model has gone stale?**

(1) Cohort calibration drift — compare 90-day-old predictions to realized revenue; retrain if actual/predicted drifts outside ~0.9–1.1, tracked per channel. (2) Feature distribution shift — weekly KS-tests on RFM inputs. (3) Rank stability — a drop in top-decile capture rate signals lost discriminative power even with stable mean calibration. Trigger full retraining if at least two signals fire together.

**Q7: What have companies like Shopify and Airbnb found that differs from textbook approaches?**

Shopify found BG/NBD performs well even on sparse merchant history, and that cohort-level calibration matters more than individual accuracy for planning use cases; cohort "vintage" predicts LTV almost as strongly as behavioral features. Airbnb models bidirectional host/guest LTV jointly via multi-task learning, and found naive long-horizon (1-year) models suffer badly from cohort-maturity bias, addressed with propensity-weighted correction. Both emphasize that an accurate model applied to the wrong decision produces no business value.

---

## 13. Reference Papers and Systems

| Reference | Key Contribution |
|---|---|
| Fader, Hardie, Lee (2005) — "Counting Your Customers the Easy Way" | BG/NBD model; foundational for non-contractual LTV |
| Fader, Hardie (2013) — "The Gamma-Gamma Model of Monetary Value" | Gamma-Gamma model for the AOV component |
| Schweidel, Knox (2013) | Extension of BG/NBD with marketing interventions |
| Airbnb Engineering Blog (2019) — "Estimating and Combining LTV" | Multi-task LTV at marketplace scale |
| Shopify Engineering Blog (2020) — "Predicting Customer Lifetime Value" | Practical BG/NBD deployment at e-commerce scale |
| Lifetimes Python library | Open-source BG/NBD and Gamma-Gamma implementations |

## Flashcards

**Contractual vs non-contractual LTV — key difference?** #flashcard
Contractual: churn is an explicit event (cancellation). Non-contractual: churn is latent — you never observe "the last purchase."

**Why is training on raw historical LTV wrong?** #flashcard
It's right-censored — recently acquired customers have less observed revenue regardless of true LTV, so the model learns to underpredict new customers.

**BG/NBD — what are the two latent processes?** #flashcard
Purchase process (Poisson rate λ ~ Gamma(r,α)) and dropout process (probability p ~ Beta(a,b)) after each purchase.

**Gamma-Gamma model — what does it add to BG/NBD?** #flashcard
Models average monetary value per transaction, assumed independent of purchase frequency, to convert purchase counts into revenue.

**When to prefer probabilistic (BG/NBD) over ML models?** #flashcard
Sparse history (<5 purchases), only RFM features available, small population, high interpretability need.

**Why optimize calibration over RMSE for LTV used in bidding?** #flashcard
LTV is Pareto-distributed; RMSE is dominated by top 1% of customers. Segment-level calibration (predicted vs actual mean) directly drives correct bid amounts.

**How do you correctly construct LTV training labels to avoid censoring bias?** #flashcard
Use a fixed cutoff date and fixed-length horizon (e.g. 90 days), keep only customers acquired well before the cutoff so the label is fully observed, and pool multiple historical cutoffs.

**Two-stage model for LTV — why needed?** #flashcard
Training data only has purchasers, but scoring happens on all new signups (including non-purchasers). Fix: multiply P(converts) x LTV(if converts).

**Simpson's paradox risk in LTV modeling?** #flashcard
Newer cohorts appear lower-LTV purely due to censoring, which can look like a declining trend. Fix by including tenure as a feature and using multiple cutoffs.

**Main LTV monitoring signals?** #flashcard
Cohort calibration drift (actual/predicted over time), feature distribution drift (KS-test), and rank stability (top-decile capture rate).
