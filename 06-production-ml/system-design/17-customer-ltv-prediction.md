---
module: Production Ml
topic: System Design
subtopic: Customer Ltv Prediction
status: unread
tags: [productionml, ml, system-design-customer-ltv-pre]
---
# Customer Lifetime Value (LTV) Prediction System Design

End-to-end ML system for predicting customer lifetime value at e-commerce, subscription, and fintech scale. Canonical system design question at companies running performance marketing, CRM, and growth programs (Shopify, Airbnb, DoorDash, Stripe, Netflix).

**Scale:** 100M+ customers, daily batch scoring, real-time scoring for new users at acquisition, LTV predictions feeding $10M+/day in paid acquisition bidding.

---

## 1. Problem Framing

### Clarifying Questions

- **Business model?** Subscription (contractual) vs e-commerce/marketplace (non-contractual) — fundamentally different statistical models apply
- **Prediction horizon?** 90-day LTV (tactical, for acquisition bidding) vs 1-year LTV (strategic, for cohort planning) vs lifetime (theoretical)
- **Customer population?** New users at acquisition (cold start) vs existing customers (full history available)
- **Primary use case?**
  - Acquisition bidding: bid = predicted LTV × margin − CAC → needs calibrated expected value
  - Retention/churn: who to target with discount/win-back → needs relative ranking
  - Upsell/pricing: what offer to show → needs segment-level calibration
  - Financial planning: cohort revenue forecasting → needs aggregate accuracy, not individual
- **Revenue definition?** Gross revenue, net revenue (after refunds), contribution margin, gross profit?
- **Discount rate?** For DCF-based LTV, what WACC or hurdle rate to use?
- **Latency requirement?** Acquisition bidding needs <200ms; CRM batch jobs can run nightly
- **Feedback delay?** How long before we observe realized revenue for a prediction? (weeks to years)

### Metric Trade-offs

**Individual vs aggregate accuracy:**

| Metric | What it measures | Use case |
|---|---|---|
| MAE / RMSE on individual predictions | Point accuracy | Not recommended — LTV is inherently uncertain |
| Calibration (expected = actual) at segment level | Bid accuracy | Acquisition bidding — undercalibrated → overbid |
| Spearman rank correlation | Ranking quality | Retention targeting — who to contact |
| MAPE at cohort level | Aggregate accuracy | Financial planning |
| Gini / AUC of cumulative revenue | Top-decile concentration | "Top 20% of customers generate 80% of revenue" |

**The calibration imperative for bidding:**

If predicted LTV = $120 but actual LTV = $80, you systematically overbid by 50%, eroding all margin. Aggregate calibration at the channel × cohort level matters more than individual RMSE.

```
Optimal CPA bid = E[LTV | channel, device, cohort] × gross_margin_rate
                                                     − target_ROAS_floor
```

**Why not optimize RMSE?** LTV distributions are heavily right-skewed (Pareto-like). RMSE is dominated by the top 1% of customers. A model that predicts the mean for everyone scores well on RMSE but provides zero discriminative signal.

---

## 2. LTV Definitions and Formulations

### Historical LTV (realized)

```
historical_LTV(customer, t) = Σ revenue(t_i) for all transactions t_i ≤ t
```

Simple but not predictive. Used as training labels.

### Predicted LTV (discounted future revenue)

```
predicted_LTV(customer, t, horizon) = E[ Σ_{τ=t}^{t+horizon} r_τ × (1+d)^{-(τ-t)} ]
```

where r_τ is revenue at time τ and d is the periodic discount rate. The expectation is over the joint distribution of purchase occurrence and purchase amount.

### Contractual vs Non-Contractual Settings

| Setting | Definition of "active" | Examples | Model |
|---|---|---|---|
| Contractual | Customer explicitly churns (cancels) | SaaS, subscription boxes, insurance | Survival analysis, DCF on MRR |
| Non-contractual | Churn is latent — customer just stops buying | E-commerce, ride-share, food delivery | BG/NBD, probabilistic models |

In non-contractual settings, you never observe "the last purchase." A customer who bought 6 months ago might be churned or just in a long inter-purchase interval. This is the central modeling challenge.

### The Censoring Problem

Training data for LTV is right-censored: customers acquired recently have short observation windows. A customer acquired 30 days ago has by definition less observed revenue than a customer acquired 2 years ago, regardless of their true LTV.

**Naive approach (wrong):** Train on historical_LTV directly. Model learns that new customers have low LTV. Prediction for new acquisition = underestimate.

**Correct approach:** Model the generative process — purchase probability × purchase value × repeat purchase rate — then integrate over the prediction horizon with survival/hazard functions.

---

## 3. System Architecture

```
Customer Touch-points
  │  Purchases, sessions, support tickets, app opens
  │
  ▼
┌─────────────────────────────────────────────────────┐
│              Event Stream (Kafka / Kinesis)          │
│   purchase_events, session_events, return_events     │
└───────────────────┬─────────────────────────────────┘
                    │
          ┌─────────┴──────────┐
          │                    │
          ▼                    ▼
  ┌──────────────┐    ┌──────────────────┐
  │  Real-time   │    │  Batch Feature   │
  │  Feature     │    │  Engineering     │
  │  Computation │    │  (Spark / dbt)   │
  │  (Flink)     │    │  daily refresh   │
  └──────┬───────┘    └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
          ┌─────────────────┐
          │  Feature Store  │◄──── Cold-start priors
          │  (Redis / Feast)│       (channel, device, geo)
          └────────┬────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
         ▼                    ▼
 ┌───────────────┐   ┌────────────────┐
 │  Probabilistic│   │  ML Model      │
 │  Model        │   │  (GBM/LSTM/    │
 │  (BG/NBD +    │   │   Multi-task)  │
 │   Gamma-Gamma)│   │                │
 └───────┬───────┘   └───────┬────────┘
         │                   │
         └──────────┬─────────┘
                    │
                    ▼
          ┌─────────────────┐
          │  Ensemble /     │
          │  Calibration    │◄──── Segment-level isotonic regression
          │  Layer          │
          └────────┬────────┘
                   │
         ┌─────────┴────────────────────────────────┐
         │                                           │
         ▼                                           ▼
┌─────────────────┐                     ┌────────────────────┐
│  Real-time API  │                     │  Batch Scoring     │
│  (<200ms)       │                     │  (existing users,  │
│  New user at    │                     │   nightly)         │
│  acquisition    │                     └──────────┬─────────┘
└────────┬────────┘                                │
         │                                         │
         ▼                                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Downstream Consumers                       │
│                                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │ Value-Based   │  │  CRM /        │  │  Dynamic        │  │
│  │ Bidding       │  │  Retention    │  │  Pricing /      │  │
│  │ (Google/Meta) │  │  Campaigns    │  │  Offer Engine   │  │
│  └───────────────┘  └───────────────┘  └─────────────────┘  │
│                                                              │
│  ┌───────────────┐  ┌───────────────┐                        │
│  │  Financial    │  │  Cohort       │                        │
│  │  Planning     │  │  Analytics    │                        │
│  └───────────────┘  └───────────────┘                        │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Monitoring /   │◄──── Actual revenue labels (delayed)
│  Feedback Loop  │      Cohort calibration dashboards
└─────────────────┘
```

---

## 4. Probabilistic Models: BG/NBD and Gamma-Gamma

### Why Probabilistic Models for Sparse Data

For customers with only 1–3 purchases, gradient boosting overfits to idiosyncratic noise. Probabilistic models impose strong structural priors derived from observed population behavior:

- **Sparse data advantage:** BG/NBD makes principled predictions from a single purchase event
- **Uncertainty quantification:** Returns a distribution over future purchases, not just a point estimate
- **Interpretability:** Model parameters (β, r, α, a, b) have business-meaningful interpretations
- **No feature engineering required:** Model inputs are just recency, frequency, tenure (T)

### BG/NBD Model (Fader, Hardie, and Lee, 2005)

The Buy-Till-You-Die model for non-contractual settings. Two latent processes:

**1. Purchase process (while active):** Customer makes purchases following a Poisson process with individual rate λ. The population distribution of λ follows a Gamma(r, α).

**2. Dropout process:** After each purchase (or at any time), the customer "dies" (churns) with probability p. The population distribution of p follows a Beta(a, b).

**Key equations:**

```
P(alive | x, t_x, T) = P(alive) / [P(alive) + P(died after last purchase)]

E[X(t) | x, t_x, T] = (r + x) × (α + T) / (α + T + t) × P(alive | ...)
                       × [conditional expected transactions if alive]
```

where:
- `x` = number of observed transactions in `[0, T]`
- `t_x` = recency (time of last transaction)
- `T` = customer age (time since acquisition)

**Fitting BG/NBD:**

```python
from lifetimes import BetaGeoFitter

bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(
    frequency=df['frequency'],    # number of repeat purchases
    recency=df['recency'],        # weeks since last purchase
    T=df['T']                     # weeks since acquisition
)

# Predict expected purchases in next 52 weeks
df['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    t=52,
    frequency=df['frequency'],
    recency=df['recency'],
    T=df['T']
)
```

### Gamma-Gamma Model for Monetary Value

Assumes average transaction value is independent of purchase frequency (empirically valid). Individual mean transaction value follows a Gamma distribution; population of mean values follows another Gamma.

```python
from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(df['frequency'], df['avg_monetary_value'])

# Compute LTV
ltv = ggf.customer_lifetime_value(
    bgf,
    df['frequency'],
    df['recency'],
    df['T'],
    df['avg_monetary_value'],
    time=12,          # 12 months
    discount_rate=0.01  # monthly discount rate
)
```

### Pareto/NBD

Predecessor to BG/NBD. Dropout can occur at any time (not just post-purchase). Mathematically equivalent in many settings but computationally more expensive. BG/NBD is preferred in practice.

### When to Use Probabilistic vs ML Models

| Condition | Prefer Probabilistic | Prefer ML |
|---|---|---|
| Customer history depth | <5 purchases | 10+ purchases |
| Feature richness | Only RFM available | Rich behavioral/contextual features |
| Population size | Small (1K–100K) | Large (1M+) |
| Interpretability requirement | High | Flexible |
| Cold start (new customer) | BG/NBD with priors | Not applicable |
| Macro covariates (seasonality) | Limited | Strong |

---

## 5. ML Approaches

### 5.1 Gradient Boosting on RFM Features

Most practical approach for e-commerce with >1M customers and rich feature sets.

**Target variable construction:**

```python
# Regression on log(1 + LTV) to handle right-skewed distribution
# LTV_90d = sum of purchases in 90 days after a fixed cutoff date
# Use customers with >= 12 months of history so 90-day label is fully observed

df['ltv_90d'] = df['revenue_post_cutoff'].clip(upper=percentile_99)
df['log_ltv'] = np.log1p(df['ltv_90d'])

# Train XGBoost / LightGBM
model = LGBMRegressor(
    objective='regression',
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=50
)
model.fit(X_train, y_train['log_ltv'])
predictions = np.expm1(model.predict(X_test))
```

**Handling the skewed distribution:** Alternatives to log-transform:

- Tweedie regression (compound Poisson-Gamma) — models zero-inflation natively
- Two-stage: classify purchasers vs non-purchasers, then regress on purchaser LTV
- Quantile regression — predict median LTV rather than mean

### 5.2 Sequence Models on Purchase History

For customers with rich transaction sequences, LSTM or Transformer over the ordered purchase history captures temporal dependencies that RFM aggregations miss.

```
Input sequence: [purchase_1, purchase_2, ..., purchase_n]
Each token: (amount, category, channel, days_since_prev, session_context)
Output: predicted LTV over next 90 days
```

**Architecture:**

```python
class LTVTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.item_embed = nn.Embedding(num_categories, d_model)
        self.amount_proj = nn.Linear(1, d_model)
        self.time_embed = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, token_ids, amounts, time_gaps, mask):
        x = self.item_embed(token_ids) + self.amount_proj(amounts) + self.time_embed(time_gaps)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x[:, -1, :]  # last token representation
        return self.head(x)
```

**Practical limitation:** Sequence models require substantial history (>5 events) and are expensive to serve. Use for high-value customer segments; fall back to GBM for sparse users.

### 5.3 Survival Analysis

Frames churn prediction as a time-to-event problem. Natural for the non-contractual setting.

**Cox Proportional Hazards:**

```
h(t | x) = h_0(t) × exp(β^T x)
```

where h_0(t) is the baseline hazard. Predicts hazard ratio, not directly usable for LTV without integrating over the survival function.

**Weibull Accelerated Failure Time (AFT):**

```
log(T) = β^T x + σε,   ε ~ extreme value
```

More interpretable: features directly scale time-to-churn. Better for business communication.

**Integration for LTV:**

```python
# LTV = integral of E[revenue per unit time] × S(t) dt over horizon
# S(t) = survival probability = P(customer still active at time t)

from lifelines import WeibullAFTFitter

aft = WeibullAFTFitter()
aft.fit(df, duration_col='days_to_churn', event_col='churned')

# Probability customer is still active at day 90
survival_prob_90 = aft.predict_survival_function(df).loc[90]

# Expected revenue = avg_daily_revenue × integral_0_90 S(t) dt
expected_revenue = df['avg_daily_revenue'] * aft.predict_expectation(df).clip(upper=90)
```

### 5.4 Multi-Task Learning

Predict multiple correlated outputs jointly: churn probability, purchase frequency, and average order value. Shared representations improve sample efficiency, especially for sparse customers.

```python
class MultiTaskLTV(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.churn_head    = nn.Linear(128, 1)   # binary, sigmoid
        self.frequency_head = nn.Linear(128, 1)  # Poisson rate, softplus
        self.aov_head      = nn.Linear(128, 1)   # log-normal mean

    def forward(self, x):
        shared = self.shared(x)
        return {
            'churn_prob':  torch.sigmoid(self.churn_head(shared)),
            'frequency':   F.softplus(self.frequency_head(shared)),
            'log_aov':     self.aov_head(shared)
        }

    def predict_ltv(self, x, horizon_days):
        out = self.forward(x)
        p_active = 1 - out['churn_prob']
        expected_purchases = out['frequency'] * horizon_days / 30 * p_active
        expected_aov = torch.exp(out['log_aov'])
        return expected_purchases * expected_aov
```

---

## 6. Feature Engineering

### RFM Features (baseline, always include)

| Feature | Definition | Notes |
|---|---|---|
| `recency_days` | Days since last purchase | Log-transform; non-linear effect |
| `frequency` | Number of distinct purchase days | Not total orders (avoids multi-item order inflation) |
| `monetary_total` | Cumulative net revenue | Use net (after returns) |
| `monetary_avg` | Average order value | Clip outliers at 99th percentile |
| `frequency_30d` / `90d` / `365d` | Multi-window purchase counts | Captures acceleration/deceleration |
| `recency_trend` | frequency_30d / frequency_90d | Increasing engagement = positive signal |
| `days_since_first_purchase` | Customer age | Longer-tenured customers are more predictable |

### Behavioral Sequence Features

| Feature | Signal |
|---|---|
| Browse-to-purchase ratio | High browse without purchase = lower intent |
| Cart abandonment rate | Strong indicator of price sensitivity |
| Category breadth | Multi-category customers have higher retention |
| Return rate | High returns negatively predict net LTV |
| Support ticket frequency | Predictor of churn, not LTV itself |
| App vs web ratio | App users typically have higher LTV |
| Session frequency / length | Engagement proxy |

### Product and Payment Features

| Feature | Signal |
|---|---|
| Primary category | Commodity categories (grocery) churn less but lower AOV |
| Brand affinity score | Preference for premium brands → higher AOV |
| Subscription upgrade flag | Strong positive LTV signal |
| Payment method | Credit card users have higher LTV than BNPL in e-commerce |
| Coupon / discount dependency | Discount-only purchasers have lower organic LTV |

### Cohort-Level Features

Individual-level models miss cohort effects. Add cohort features to control for:

```python
# Cohort-level features joined to individual records
cohort_features = {
    'cohort_month_avg_ltv_90d': cohort_ltv_lookup[customer.cohort_month],
    'cohort_retention_rate_90d': cohort_retention[customer.cohort_month],
    'acquisition_channel_ltv_index': channel_ltv_index[customer.channel],
    'macro_consumer_confidence_at_acquisition': macro_data[customer.acquisition_date],
}
```

### Channel Attribution Features

For acquisition bidding, the LTV model must condition on channel since acquired customers from different channels have systematically different LTV:

| Channel | Typical LTV index (relative to direct) |
|---|---|
| Organic search | 1.3× |
| Paid search (branded) | 1.1× |
| Paid social (prospecting) | 0.7× |
| Influencer / affiliate | 0.6× |
| Email re-engagement | 1.5× (existing) |
| Referral | 1.4× |

---

## 7. Handling Censoring

### The Right-Censoring Problem

Customers acquired recently are right-censored: we only observe their LTV up to the present, not their eventual lifetime LTV.

**Naive training (wrong):**

```
Label = total revenue to date
→ Customers acquired 3 months ago always have less revenue than those acquired 3 years ago
→ Model learns "time since acquisition" as the dominant feature
→ All new customers predicted to have low LTV
```

**Correct approach — time-conditioned prediction:**

Define the label as "revenue over a fixed future window from a fixed cutoff date":

```python
CUTOFF_DATE = "2024-01-01"
HORIZON_DAYS = 90

# Only use customers who:
# 1. Were acquired at least (HORIZON_DAYS) before cutoff (so label is fully observed)
# 2. Had at least one purchase before cutoff (otherwise trivially zero)

df_train = df[
    (df['acquisition_date'] <= CUTOFF_DATE - timedelta(days=HORIZON_DAYS)) &
    (df['first_purchase_date'] < CUTOFF_DATE)
].copy()

df_train['ltv_label'] = df_train['revenue_cutoff_to_cutoff_plus_90d']
```

**Multiple cutoff dates:** Use several historical cutoffs to increase training data and reduce temporal overfitting:

```python
cutoffs = ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01"]
# Sample customers from each cutoff
# Compute features as of cutoff date, label as revenue in next 90 days
# Combine all cutoff cohorts for training
```

### Survival Analysis Framing for Censored Data

Survival models handle censoring natively by treating "customer still active at last observation" as a censored event rather than a churn event. This avoids the selection bias of only including churned customers with complete LTV observations.

```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(
    durations=df['tenure_days'],
    event_observed=df['has_churned']  # 0 = censored (still active)
)

# Expected remaining tenure conditional on being active for T days
expected_remaining = kmf.median_survival_time_ - T
```

### Time-Conditioned Predictions at Serving

At serving time, for an existing customer active for T days with x purchases:

```
LTV(t, t + horizon | active at t) = E[revenue | x purchases, recency, T] × P(active through t+horizon | x, T)
```

This conditions out customers already churned and avoids predicting LTV for dead customers.

---

## 8. Segment-Level Calibration

### Why Individual Calibration Is Insufficient

A model can be well-calibrated in aggregate (mean prediction = mean actual) but badly calibrated within segments. For acquisition bidding, you bid on channels and audiences — if the model overestimates LTV for Facebook prospecting audiences by 40%, you overbid systematically on that channel.

### Calibration Framework

**Step 1: Train base model** (GBM or BG/NBD).

**Step 2: Compute calibration errors by segment:**

```python
segments = ['channel', 'device', 'acquisition_cohort_month', 'geo_tier']

for segment_col in segments:
    for segment_val in df[segment_col].unique():
        mask = df[segment_col] == segment_val
        pred_mean = df.loc[mask, 'predicted_ltv'].mean()
        actual_mean = df.loc[mask, 'actual_ltv_90d'].mean()
        calibration_error[segment_col][segment_val] = actual_mean / pred_mean
```

**Step 3: Apply isotonic regression calibrator per segment:**

```python
from sklearn.isotonic import IsotonicRegression

# Fit isotonic regression on validation set per channel
calibrators = {}
for channel in channels:
    mask = val_df['channel'] == channel
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(val_df.loc[mask, 'predicted_ltv'], val_df.loc[mask, 'actual_ltv_90d'])
    calibrators[channel] = ir

# Apply at serving time
def calibrated_ltv(raw_pred, channel):
    return calibrators.get(channel, default_calibrator).predict([raw_pred])[0]
```

### Acquisition Bidding Integration (Value-Based Bidding)

**Google Ads / Meta Conversions API:**

```python
# Send LTV prediction as the "conversion value" at the time of acquisition
# Google's tROAS bidding then optimizes CPA bids based on expected value

def on_new_acquisition(user_id, channel, device, features):
    ltv_pred = ltv_model.predict(features)
    calibrated = calibrated_ltv(ltv_pred, channel)
    
    # Send to Google Ads
    send_conversion_event(
        user_id=user_id,
        conversion_value=calibrated * GROSS_MARGIN_RATE,
        currency='USD'
    )
```

**Bid formula:**

```
Max CPA = E[LTV | channel, device, audience] × margin_rate / target_ROAS

Example:
  E[LTV | FB prospecting] = $45
  margin_rate = 0.35
  target_ROAS = 2.0
  Max CPA = $45 × 0.35 / 2.0 = $7.88
```

### Shapley-Based Channel Attribution

When a customer has been touched by multiple channels before conversion, LTV credit attribution uses Shapley values to fairly attribute LTV to each channel.

```python
import shap

explainer = shap.TreeExplainer(ltv_model)
shap_values = explainer.shap_values(X_customer)

# Channel attribution = SHAP value for channel features
channel_ltv_attribution = {
    'paid_social_shap': shap_values[channel_feature_indices].sum(),
    'email_shap': shap_values[email_feature_indices].sum()
}
```

---

## 9. Serving and Downstream Integration

### Real-Time Scoring: New Users at Acquisition

**Challenge:** New user has no purchase history. Must predict LTV from:
- Acquisition channel and campaign
- Device type, operating system, browser
- Geographic location (city, DMA)
- Landing page / product viewed
- Time of day and day of week
- Referrer / UTM parameters

**Cold-start model:**

```python
# Separate "acquisition-time LTV model" trained only on features available at signup
# Features: channel, device, geo, utm_source, utm_medium, utm_campaign, landing_page
# Label: LTV over next 90 days (from historical cohorts)

acquisition_model = LGBMRegressor(...)
acquisition_model.fit(X_acquisition_features, y_ltv_90d)

# Served as low-latency API called during acquisition pixel fire
# P99 latency target: 50ms
```

**Serving infrastructure:**

```
New visit → tracking pixel → Lambda/Cloud Run function
    → feature lookup from Redis (channel/geo priors)
    → model inference (pre-loaded in memory)
    → LTV score → conversion event payload
    → Google/Meta Ads API
```

### Batch Scoring: Existing Customers

Nightly batch job scoring all active customers:

```
Spark job → pull customer features from data warehouse
          → apply BG/NBD + Gamma-Gamma (or GBM)
          → write LTV scores to feature store
          → downstream consumers read from feature store
```

### Downstream Consumer Integration

| Consumer | LTV Usage | Update Frequency |
|---|---|---|
| CRM / Marketing Cloud | Segment customers by LTV tier for campaign targeting | Daily |
| Retention team | Flag high-LTV customers showing churn signals | Daily |
| Pricing engine | Offer dynamic discounts inversely proportional to LTV | Real-time |
| Customer success | Prioritize support queues by LTV | Real-time |
| Finance / planning | Cohort LTV curves for revenue forecasting | Weekly |
| Product | A/B test measurement using LTV as primary metric | Per-experiment |

---

## 10. Model Monitoring and Feedback Loop

### Metrics to Monitor

**Prediction quality (lagged — observable after horizon):**

```python
# Compare predictions made 90 days ago against realized revenue
monitoring_df = predictions_made_90d_ago.merge(realized_revenue, on='customer_id')

metrics = {
    'overall_calibration': monitoring_df['actual_ltv'].mean() / monitoring_df['predicted_ltv'].mean(),
    'channel_calibration': monitoring_df.groupby('channel').apply(
        lambda g: g['actual_ltv'].mean() / g['predicted_ltv'].mean()
    ),
    'spearman_rank_corr': monitoring_df[['predicted_ltv', 'actual_ltv']].corr('spearman').iloc[0, 1],
    'top_decile_capture': (
        monitoring_df.nlargest(n // 10, 'predicted_ltv')['actual_ltv'].sum() /
        monitoring_df.nlargest(n // 10, 'actual_ltv')['actual_ltv'].sum()
    )
}
```

**Distributional drift (observable immediately):**

- KS-test on prediction distribution vs 30-day rolling baseline
- Feature drift detection on RFM features (population shift in new cohorts)
- Cohort-level purchase rate changes

**Retraining triggers:**

| Trigger | Action |
|---|---|
| Channel calibration error > 15% | Retrain calibration layer on recent cohorts |
| Feature drift p-value < 0.01 | Full model retrain |
| Major macro event (economic shock) | Emergency reweighting of recent cohorts |
| Seasonal product mix shift | Add seasonal features, retrain |

---

## 11. Failure Modes

### Selection Bias in Training Data

**Problem:** Model trained on customers who made at least one purchase. Customers who signed up but never purchased are excluded. At acquisition scoring time, model is applied to all new users including those who will never purchase.

**Mitigation:** Two-stage model — first predict P(makes first purchase), then predict LTV conditional on purchasing. Multiply for expected LTV.

```python
p_convert = conversion_model.predict_proba(X)[:, 1]
ltv_if_convert = ltv_model.predict(X)
expected_ltv = p_convert * ltv_if_convert
```

### Simpson's Paradox Across Cohorts

**Problem:** Newer cohorts have lower observed LTV (censoring). A naive model trained on all cohorts appears to show "LTV is declining over time" — actually just a cohort maturity effect.

**Detection:**

```python
# Plot median predicted_ltv vs acquisition_month for cohorts at same tenure
# Should be flat if model is well-specified
cohort_calibration = df.groupby(['acquisition_cohort', 'tenure_bucket']).apply(
    lambda g: g['actual_ltv'].mean() / g['predicted_ltv'].mean()
)
```

**Mitigation:** Always control for `tenure_at_prediction_time` as a feature. Use multiple historical cutoff dates to prevent model from learning recency of cohort as a proxy for maturity.

### Overfitting to Historical Purchase Patterns

**Problem:** Model learns from bull-market e-commerce years (2020–2021). Post-pandemic normalization looks like "LTV decline" but is actually mean reversion.

**Mitigation:**
- Include macroeconomic signals (consumer confidence, unemployment) as features
- Weight recent cohorts more heavily in training (time-decay weighting)
- Monitor cohort-level calibration continuously

### Macro Shocks (COVID, Recession)

**Problem:** LTV models trained pre-shock dramatically mis-predict post-shock. March 2020: e-commerce LTV spiked; travel LTV went to zero.

**Mitigation:**
- Maintain ensemble of models trained on different time windows
- During shock detection, fall back to short-window (30-day) behavior model
- Add macro index features that can shift predictions without full retraining

```python
# Macro-adjusted prediction
macro_adjustment = macro_model.predict(current_macro_features) / macro_model.predict(baseline_macro)
adjusted_ltv = base_ltv_prediction * macro_adjustment
```

### Customer Gaming

**Problem:** High-value customers learn that high LTV leads to fewer discounts. Customers deliberately reduce purchase frequency to appear low-value and receive win-back offers.

**Mitigation:**
- Do not expose LTV tiers directly to customers
- Use LTV for CPA bidding (external — customers can't observe), not for internal discount logic alone
- Add anomaly detection for "strategically reduced engagement" patterns

### Cold-Start Overconfidence

**Problem:** Acquisition-time model has high uncertainty but outputs point predictions. Bidding on point predictions leads to systematic overbidding for low-data segments.

**Mitigation:** Output prediction intervals. For acquisition bidding, bid on a conservative percentile (e.g., 25th percentile) rather than the mean when segment sample size is small.

```python
def bid_ltv(predicted_ltv, prediction_std, segment_n):
    confidence_discount = max(0.7, 1 - 2 / np.sqrt(segment_n))
    conservative_ltv = predicted_ltv - 0.5 * prediction_std * (1 - confidence_discount)
    return conservative_ltv
```

---

## 12. Interview Questions and Answers

**Q1: Why not just use historical average LTV as the prediction?**

Historical average LTV has severe censoring bias: customers who churned early have low historical LTV, but customers currently active are right-censored. Using the average ignores that your best customers are still actively adding revenue. The historical average systematically underestimates true expected LTV for active customers. You need a model that conditions on current engagement signals and projects forward.

---

**Q2: BG/NBD assumes purchase rate and churn are independent. Is that a realistic assumption?**

Frequently violated. Customers who buy more often tend to be more engaged and less likely to churn — a positive correlation between λ (purchase rate) and p (survival probability). The original BG/NBD model ignores this. In practice, you can: (1) accept the approximation — BG/NBD is still remarkably robust in practice (Fader et al. validate on multiple datasets); (2) use a correlated latent variable model; (3) layer on ML features that capture engagement beyond frequency (session depth, support tickets) to capture what BG/NBD misses.

---

**Q3: How do you handle the fact that LTV predictions used for bidding create a feedback loop?**

If you bid more for high-predicted-LTV customers, you acquire more of them. If the model is right, great. If the model has systematic bias for a segment, you either over-acquire or under-acquire that segment, which shifts your training data distribution. In the next training cycle, the model learns from a biased sample.

Mitigation: (1) Exploration — occasionally bid on random samples regardless of LTV prediction to maintain an unbiased observational study. (2) Causal identification — use holdout groups at the channel level. (3) Counterfactual correction — use doubly-robust estimators that correct for propensity to acquire different segments.

---

**Q4: Product asks to use LTV as the primary metric for A/B tests. What are the concerns?**

Three main issues: (1) **Label delay** — LTV takes 90+ days to realize. Running experiments for 90 days to measure LTV significantly increases cost and slows iteration. Use surrogate metrics (30-day engagement) that are validated as proxies for 90-day LTV. (2) **Variance** — LTV has very high variance (Pareto distribution). Power calculations for LTV-based experiments require 5–10× more samples than conversion-rate experiments. (3) **Model dependence** — if you A/B test changes to the product, the LTV model was trained on pre-change behavior. The model itself may not generalize to the new user experience. Prefer realized revenue over prediction horizons when resources allow.

---

**Q5: How does LTV prediction differ for a subscription business vs e-commerce?**

In subscriptions (contractual): churn is observed explicitly (cancellation event). You can model monthly churn rate directly and compute LTV as MRR / churn_rate for a simple estimate, or use survival analysis for more precision. The hard part is predicting upgrade/downgrade, not churn.

In non-contractual e-commerce: churn is latent — you never see "the last purchase." A customer who bought 4 months ago may be churned or just in a long inter-purchase interval. BG/NBD and survival analysis are designed for this. The hard part is estimating "are they still alive?" without explicit signal.

A hybrid SaaS + marketplace (e.g., Shopify merchants) requires separate models per customer type or a unified model with a contractual indicator feature.

---

**Q6: How would you detect that your LTV model has gone stale?**

Three signals:

1. **Cohort calibration drift:** Compare predictions made 90 days ago against realized revenue. If overall_calibration = actual/predicted drifts outside 0.9–1.1, retrain. Track this separately per channel.

2. **Feature distribution shift:** Run KS-tests on input feature distributions weekly. RFM distribution shifts signal population changes (new acquisition channels, macroeconomic effects).

3. **Rank stability:** If top-decile capture rate (fraction of actual top-10% LTV customers correctly ranked in predicted top 10%) drops sharply, the model has lost discriminative power even if mean calibration is stable.

Trigger full retraining if any two of these three signals fire simultaneously.

---

**Q7: Shopify and Airbnb have published LTV work. What did they find that differs from textbook approaches?**

**Shopify (Harper and Mustanir, 2020):** Found that BG/NBD performs surprisingly well for merchant LTV prediction even with sparse history. The key practical finding: aggregate calibration at the cohort level matters far more than individual-level accuracy for financial planning use cases. They also found strong cohort effects — the "vintage" of when a merchant was acquired predicts LTV almost as much as behavioral features.

**Airbnb (Weakliem et al., 2020):** Hosts and guests have bidirectional LTV (a host's LTV depends on guest quality and vice versa). They use multi-task learning to jointly predict both. Also found that the prediction horizon matters enormously: 1-year LTV models trained naively on historical data had large cohort-maturity bias; they addressed this with a propensity-weighted approach to correct for observational censoring.

Both companies emphasize that LTV models are only as good as their downstream use cases. A highly accurate model used for a wrong decision (e.g., using LTV rank for discount allocation when LTV-sensitive customers are price-inelastic) generates zero business value.

---

## 13. Reference Papers and Systems

| Reference | Key Contribution |
|---|---|
| Fader, Hardie, Lee (2005) — "Counting Your Customers the Easy Way" | BG/NBD model; foundational paper for non-contractual LTV |
| Fader, Hardie (2013) — "The Gamma-Gamma Model of Monetary Value" | Gamma-Gamma model for AOV component of LTV |
| Schweidel, Knox (2013) — "Incorporating Direct Marketing Activity into Latent Attrition Models" | Extension of BG/NBD with marketing interventions |
| Airbnb Engineering Blog (2019) — "Estimating and Combining LTV" | Multi-task LTV at marketplace scale |
| Shopify Engineering Blog (2020) — "Predicting Customer Lifetime Value" | Practical BG/NBD deployment at e-commerce scale |
| Rudolph et al. (2022) — "Survival Analysis Meets NLP" | Sequence model + survival analysis hybrid |
| Lifetimes Python library | Open-source BG/NBD and Gamma-Gamma implementations |

## Flashcards

**Business model? Subscription (contractual) vs e-commerce/marketplace (non-contractual)?** #flashcard
fundamentally different statistical models apply

**Prediction horizon? 90-day LTV (tactical, for acquisition bidding) vs 1-year LTV (strategic, for cohort planning) vs lifetime (theoretical)?** #flashcard
Prediction horizon? 90-day LTV (tactical, for acquisition bidding) vs 1-year LTV (strategic, for cohort planning) vs lifetime (theoretical)

**Customer population? New users at acquisition (cold start) vs existing customers (full history available)?** #flashcard
Customer population? New users at acquisition (cold start) vs existing customers (full history available)

**Primary use case?** #flashcard
Primary use case?

**Acquisition bidding?** #flashcard
bid = predicted LTV × margin − CAC → needs calibrated expected value

**Retention/churn?** #flashcard
who to target with discount/win-back → needs relative ranking

**Upsell/pricing?** #flashcard
what offer to show → needs segment-level calibration

**Financial planning?** #flashcard
cohort revenue forecasting → needs aggregate accuracy, not individual

**Revenue definition? Gross revenue, net revenue (after refunds), contribution margin, gross profit?** #flashcard
Revenue definition? Gross revenue, net revenue (after refunds), contribution margin, gross profit?

**Discount rate? For DCF-based LTV, what WACC or hurdle rate to use?** #flashcard
Discount rate? For DCF-based LTV, what WACC or hurdle rate to use?

**Latency requirement? Acquisition bidding needs <200ms; CRM batch jobs can run nightly?** #flashcard
Latency requirement? Acquisition bidding needs <200ms; CRM batch jobs can run nightly

**Feedback delay? How long before we observe realized revenue for a prediction? (weeks to years)?** #flashcard
Feedback delay? How long before we observe realized revenue for a prediction? (weeks to years)

**Sparse data advantage?** #flashcard
BG/NBD makes principled predictions from a single purchase event

**Uncertainty quantification?** #flashcard
Returns a distribution over future purchases, not just a point estimate

**Interpretability?** #flashcard
Model parameters (β, r, α, a, b) have business-meaningful interpretations

**No feature engineering required?** #flashcard
Model inputs are just recency, frequency, tenure (T)

**x = number of observed transactions in [0, T]?** #flashcard
x = number of observed transactions in [0, T]

**t_x = recency (time of last transaction)?** #flashcard
t_x = recency (time of last transaction)

**T = customer age (time since acquisition)?** #flashcard
T = customer age (time since acquisition)

**Tweedie regression (compound Poisson-Gamma)?** #flashcard
models zero-inflation natively

**Two-stage?** #flashcard
classify purchasers vs non-purchasers, then regress on purchaser LTV

**Quantile regression?** #flashcard
predict median LTV rather than mean

**Acquisition channel and campaign?** #flashcard
Acquisition channel and campaign

**Device type, operating system, browser?** #flashcard
Device type, operating system, browser

**Geographic location (city, DMA)?** #flashcard
Geographic location (city, DMA)

**Landing page / product viewed?** #flashcard
Landing page / product viewed

**Time of day and day of week?** #flashcard
Time of day and day of week

**Referrer / UTM parameters?** #flashcard
Referrer / UTM parameters

**KS-test on prediction distribution vs 30-day rolling baseline?** #flashcard
KS-test on prediction distribution vs 30-day rolling baseline

**Feature drift detection on RFM features (population shift in new cohorts)?** #flashcard
Feature drift detection on RFM features (population shift in new cohorts)

**Cohort-level purchase rate changes?** #flashcard
Cohort-level purchase rate changes

**Include macroeconomic signals (consumer confidence, unemployment) as features?** #flashcard
Include macroeconomic signals (consumer confidence, unemployment) as features

**Weight recent cohorts more heavily in training (time-decay weighting)?** #flashcard
Weight recent cohorts more heavily in training (time-decay weighting)

**Monitor cohort-level calibration continuously?** #flashcard
Monitor cohort-level calibration continuously

**Maintain ensemble of models trained on different time windows?** #flashcard
Maintain ensemble of models trained on different time windows

**During shock detection, fall back to short-window (30-day) behavior model?** #flashcard
During shock detection, fall back to short-window (30-day) behavior model

**Add macro index features that can shift predictions without full retraining?** #flashcard
Add macro index features that can shift predictions without full retraining

**Do not expose LTV tiers directly to customers?** #flashcard
Do not expose LTV tiers directly to customers

**Use LTV for CPA bidding (external?** #flashcard
customers can't observe), not for internal discount logic alone

**Add anomaly detection for "strategically reduced engagement" patterns?** #flashcard
Add anomaly detection for "strategically reduced engagement" patterns
