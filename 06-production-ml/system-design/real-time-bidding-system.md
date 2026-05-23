---
module: Production Ml
topic: System Design
subtopic: Real Time Bidding System
status: unread
tags: [productionml, ml, system-design-real-time-biddin]
---
# Real-Time Bidding (RTB) / Programmatic Advertising ML System

End-to-end ML system for programmatic display and video advertising. Canonical system design question at ad-tech companies (The Trade Desk, Criteo, Google DV360, Amazon DSP) and interview loops for senior/staff ML engineers in adtech.

**Scale:** 500K+ bid requests/second, 10ms hard deadline (OpenRTB spec), 100B+ auctions/day, global multi-datacenter deployment.

---

## 1. Problem Framing

### Clarifying Questions

- **DSP or SSP perspective?** Demand-Side Platform (DSP) bids to buy impressions; Supply-Side Platform (SSP) runs the auction and passes through bids. Most ML complexity lives on the DSP side.
- **Display vs. video?** Video CPMs are 5–10x higher; viewability and completion rate replace CTR as primary signals. Video allows pre-roll, mid-roll, outstream; each has different user intent.
- **Brand vs. performance campaign?** Brand: optimize reach, frequency, viewability, brand lift lift surveys. Performance: optimize CPA, ROAS, down-funnel conversions. Model objectives and bid formulas differ fundamentally.
- **First-price vs. second-price auction?** Most exchanges moved to first-price (2019+). Bid shading is mandatory in first-price; not needed in second-price (Vickrey). The system must know the auction type per exchange.
- **Retargeting vs. prospecting?** Retargeting (known converters): high signal, small audience. Prospecting (lookalike): relies heavily on pCTR/pCVR model generalization.
- **What conversion window?** 7-day, 30-day, 60-day post-click? Longer windows mean delayed labels, which break standard online learning and require survival modeling.
- **Privacy regime?** EU GDPR, US CCPA/CPRA, Canada PIPEDA. Cookieless environments (Safari ITP, Chrome Privacy Sandbox) affect user ID resolution fundamentally.
- **Budget allocation?** Fixed daily budget per campaign, or portfolio bidding across multiple campaigns with shared budget?

### Business Metrics

| Metric | Definition | Why It Matters |
|---|---|---|
| ROAS | Revenue / Ad Spend | Primary advertiser KPI for performance campaigns |
| CPA | Cost per Acquisition | Absolute cost efficiency; tied to campaign pacing |
| Win Rate | Impressions won / Bid requests | Low win rate = under-spending; high = over-paying |
| Spend Efficiency | Actual spend / Budget target | Over-delivery = overspend; under-delivery = client churn |
| CPM | Cost per thousand impressions | Raw buying cost; normalized by audience quality |
| Impression Viewability | Measured viewable / Served | Only viewable impressions deliver brand value |

### ML Metrics

| Model | Primary Metric | Secondary Metric | Failure Mode |
|---|---|---|---|
| pCTR | Log-loss (calibration) | AUC-ROC (ranking) | Overconfident → overbid; underconfident → underbid |
| pCVR | Expected Calibration Error (ECE) | PR-AUC | Delayed labels cause distribution shift |
| Bid Landscape | RMSE of clearing price percentile | Coverage at p50/p90 | Systematic underestimation → lost auctions |
| Win Rate Model | Brier score | Calibration curve | Mis-estimation → wrong throttle factor |

**The calibration constraint is non-negotiable.** An uncalibrated pCTR of 0.05 when truth is 0.01 causes 5x overbid. Model AUC can be 0.75 but if calibration is off by 2x, the campaign loses money.

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

Target ECE < 0.005 for production bidding models.

---

## 2. Scale & Hard Constraints

### The 10ms Problem

OpenRTB 2.x specification mandates the DSP respond within **100ms** of receiving the bid request. In practice, exchanges enforce **10–50ms** before timing out and treating the response as a no-bid. At Criteo scale, the constraint is effectively **10ms end-to-end** including:

- Network transit: ~2–4ms (co-located PoP)
- Bid request parsing: ~0.5ms
- User ID lookup: ~1ms (in-memory hash map or local Redis)
- Feature enrichment: ~1–2ms (pre-computed batch features)
- ML inference: ~2–4ms (target for pCTR + pCVR)
- Bid calculation + response serialization: ~0.5ms

**This eliminates any model requiring sequential DB lookups, Python GIL, or network I/O in the hot path.**

### Scale Estimates

```
500K bid requests/sec
× 10ms timeout
= 5,000 concurrent in-flight requests per PoP

At 100B auctions/day:
  100B / 86,400s ≈ 1.16M requests/sec peak
  (500K is p50; plan for 3x headroom = 1.5M rps burst)

Win rate ~5–10% (competitive categories):
  5M–10M won impressions/day needing attribution

Click-through rate ~0.1%:
  5K–10K clicks/day per mid-size campaign

Conversion rate ~1–5% of clicks:
  50–500 conversions/day → sparse, delayed feedback
```

### Infrastructure Constraints

- **Latency budget:** 10ms p99 — not p50. p99 must hold under traffic spikes.
- **Availability:** 99.99% uptime. A 60-second outage = 30M+ missed bid opportunities.
- **Model size:** pCTR model must fit in L3 cache or at most local DRAM. No network calls to model servers in hot path.
- **Feature freshness:** User recency signals (last-click, session activity) must be ≤5 minutes stale.
- **Budget atomicity:** No overshooting daily budget by >1%. Requires distributed rate limiting with strong consistency for high-spend campaigns.

---

## 3. System Architecture

```
PUBLISHER ECOSYSTEM                    DSP ECOSYSTEM
───────────────────────────────────────────────────────────────────────
                                                    ┌─────────────────┐
Publisher Website/App                               │  Campaign Mgmt  │
        │                                           │  (budgets,      │
        │  Ad Slot Available                        │  targeting,     │
        ▼                                           │  creatives)     │
 ┌─────────────┐                                   └────────┬────────┘
 │    SSP /    │ ──── Bid Request (OpenRTB) ────►  ┌────────▼────────┐
 │  Ad Exchange│ ◄─── Bid Response (<10ms)  ──────  │   Bid Server    │
 │  (DoubleClick│                                   │  (C++/Go/Rust)  │
 │  AppNexus,  │                                   └────────┬────────┘
 │  Pubmatic)  │                                           │
 └─────────────┘                          ┌────────────────┼──────────────────┐
        │                                 │                │                  │
        │  Auction                        ▼                ▼                  ▼
        │  (Vickrey or                ┌───────┐     ┌──────────┐     ┌───────────────┐
        │  First-Price)               │  ID   │     │ Feature  │     │  ML Scoring   │
        ▼                             │Resolve│     │  Lookup  │     │  Service      │
 ┌─────────────┐                     │Service│     │  (Redis  │     │  pCTR + pCVR  │
 │  Win Notice │                     │(cookie│     │  ~1ms)   │     │  (ONNX/TRT    │
 │  (event URL)│                     │ sync) │     └──────────┘     │   ~2-4ms)     │
 └──────┬──────┘                     └───────┘                      └───────────────┘
        │                                                                    │
        │  Impression/Click/Conversion                                       │
        ▼                                                            ┌───────▼──────┐
 ┌─────────────────────────────┐                                    │  Bid Price   │
 │   Attribution Pipeline      │                                    │  Calculator  │
 │   (Kafka → Flink)           │◄──── Conversion Pixels ───────────│  (KKT opt)   │
 └──────────────┬──────────────┘                                    └──────────────┘
                │                                                           │
                ▼                                                           │
 ┌─────────────────────────────┐                                           │
 │   Training Data Store       │              ┌────────────────────────────┘
 │   (S3 + feature store)      │              │
 └──────────────┬──────────────┘              │  Real-time feedback
                │                             ▼
                ▼                    ┌─────────────────┐
 ┌─────────────────────────────┐    │  Budget Pacing  │
 │   Model Training Pipeline   │    │  Service (PID   │
 │   (daily retrain + FTRL     │    │  controller)    │
 │    online learning)         │    └─────────────────┘
 └─────────────────────────────┘
```

### Data Flow per Bid Request (10ms budget)

```
T+0ms    Bid request arrives at edge PoP (co-located with exchange)
T+0.3ms  OpenRTB JSON parsed → internal proto
T+0.5ms  User ID resolved (cookie → internal GUID, or hashed email / MAID)
T+1.0ms  Targeting filter: check campaign eligibility (geo, device, category block-list)
T+1.5ms  Batch feature lookup from local Redis / in-process cache
T+3.5ms  pCTR model inference (ONNX Runtime, quantized int8)
T+5.0ms  pCVR model inference (lighter model, ~1ms)
T+6.0ms  Win rate / clearing price estimation
T+7.0ms  Bid price calculation (optimal bid formula)
T+7.5ms  Budget pacing check (token bucket, atomic decrement)
T+8.0ms  Bid response serialized (protobuf → JSON)
T+8.0ms  Response sent; 2ms network return to exchange
```

---

## 4. Bid Request Processing

### OpenRTB Schema (Simplified)

```json
{
  "id": "abc123",
  "imp": [{
    "id": "1",
    "banner": {"w": 300, "h": 250},
    "bidfloor": 0.50,
    "bidfloorcur": "USD"
  }],
  "site": {
    "domain": "example.com",
    "cat": ["IAB1"],
    "page": "https://example.com/article/123"
  },
  "user": {
    "id": "u789",
    "buyeruid": "buyer-mapped-id"
  },
  "device": {
    "ua": "Mozilla/5.0...",
    "ip": "203.0.113.42",
    "geo": {"country": "US", "metro": "807"}
  },
  "at": 1,
  "tmax": 100
}
```

Key fields for ML:
- `imp.bidfloor`: Minimum acceptable bid (hard constraint on bid price)
- `user.buyeruid`: DSP-mapped user ID (result of cookie sync)
- `site.cat`: IAB content categories (contextual signal)
- `device.*`: Device type, OS, browser (feature engineering)
- `at`: Auction type — 1=first-price, 2=second-price

### User ID Resolution

**Cookie sync (web):** SSP redirects user browser to DSP pixel; DSP stores SSP user ID ↔ DSP user ID mapping in key-value store. At bid time, `buyeruid` in request maps to internal user profile.

**Mobile (MAID):** iOS IDFA (post-ATT ~30% availability), Android GAID. Stored as hashed SHA-256 for privacy-safe join.

**Privacy-preserving IDs (2024+):**
- **Unified ID 2.0 (UID2):** Hashed+encrypted email/phone; DSP must hold decryption keys
- **Google Privacy Sandbox Topics API:** Browser returns 3 interest topics per request; no cross-site user identifier
- **Contextual fallback:** When no user ID is available, rely entirely on page context, device type, time-of-day features

```python
def resolve_user_id(bid_request: BidRequest) -> Optional[str]:
    """Resolve to internal GUID using priority chain."""
    # 1. Buyer UID from cookie sync (highest signal)
    if bid_request.user.buyeruid:
        return cookie_sync_store.get(bid_request.user.buyeruid)
    
    # 2. Hashed email / MAID match
    if bid_request.user.eids:
        for eid in bid_request.user.eids:
            if eid.source == "liveramp.com":
                internal_id = id_graph.lookup_rampid(eid.uids[0].id)
                if internal_id:
                    return internal_id
    
    # 3. Probabilistic fingerprint (device + IP + UA hash)
    # Only where legally permissible
    fp = fingerprint(bid_request.device.ip, bid_request.device.ua)
    return fingerprint_store.get(fp)  # May return None → contextual-only
```

### Real-Time Feature Enrichment

Features are pre-computed offline and served from a local Redis instance (co-located with bid server). No synchronous feature computation in the hot path.

| Feature Group | Source | Staleness | Storage |
|---|---|---|---|
| User interest vector (128-dim) | Daily batch job | 24h | Redis hash |
| User recency signals (last click, last visit) | Stream (Flink) | 5 min | Redis string |
| Campaign performance stats (CTR, CVR trailing 7d) | Hourly batch | 1h | Redis hash |
| Contextual embedding (page URL → topic vector) | Pre-computed, URL hash key | — | In-process LRU |
| Historical win rates per exchange/format | Daily | 24h | In-process map |
| Device type → avg CTR lookup table | Weekly | 7d | In-process map |

---

## 5. ML Models

### 5.1 pCTR Model — Click Probability

**Goal:** Estimate P(click | impression, user, ad, context).

**Architecture choice: FTRL-Proximal (Follow The Regularized Leader)**

For latency-critical bidding, Logistic Regression trained with FTRL is the production workhorse (used by Google, Criteo, Yahoo). Reasons:
- Inference is a dot product: O(n_features) in microseconds
- L1 regularization produces sparse models (millions of features, ~1M nonzero weights)
- Online updates: weight updates can be applied to the serving model every few minutes without redeployment
- Interpretable: SHAP values map directly to feature weights

```python
class FTRLModel:
    """FTRL-Proximal with L1/L2 regularization for online CTR prediction."""
    
    def __init__(self, alpha=0.1, beta=1.0, l1=1.0, l2=1.0):
        self.alpha = alpha  # learning rate
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.z = defaultdict(float)  # per-feature accumulators
        self.n = defaultdict(float)  # per-feature gradient squared sum
        self.w = {}                  # current weights (sparse)
    
    def _get_weight(self, feature_id: int) -> float:
        z_i = self.z[feature_id]
        if abs(z_i) <= self.l1:
            return 0.0  # L1 sparsity
        sign = 1.0 if z_i > 0 else -1.0
        n_i = self.n[feature_id]
        return -(z_i - sign * self.l1) / (
            (self.beta + math.sqrt(n_i)) / self.alpha + self.l2
        )
    
    def predict(self, features: List[int]) -> float:
        score = sum(self._get_weight(f) for f in features)
        return 1.0 / (1.0 + math.exp(-score))  # sigmoid
    
    def update(self, features: List[int], p: float, y: float):
        grad = p - y  # cross-entropy gradient
        for f in features:
            sigma = (math.sqrt(self.n[f] + grad**2) - math.sqrt(self.n[f])) / self.alpha
            self.z[f] += grad - sigma * self._get_weight(f)
            self.n[f] += grad**2
```

**Feature hashing:** Raw features (user_id, campaign_id, domain, device_type, hour_of_week) are hashed to int buckets (2^24 = 16M buckets). Cross-features (user_id × campaign_id) are common and captured via conjunction hashing.

**Neural upgrade (when latency allows ~4ms):**

For non-real-time scoring (e.g., audience segment scoring, daily model refresh), a two-tower model captures cross-feature interactions:

```
User tower:    user_id_emb(64) + recency_features → MLP(64, 32)
Ad tower:      campaign_id_emb(32) + format_features → MLP(32, 16)
Context tower: domain_emb(32) + iab_cat_emb(16) → MLP(32, 16)
                            ↓
              Concatenate → MLP(64, 32, 1) → sigmoid(pCTR)
```

Embeddings are trained offline; at inference, user embedding is pre-fetched from Redis (lookup replaces forward pass through user tower).

### 5.2 pCVR Model — Conversion Probability

**The delayed feedback problem** is the central challenge: a click today may convert in 7–30 days. Naively training on recent data causes:
1. **Attribution bias:** Impressions from yesterday appear non-converting because conversions haven't arrived yet
2. **Distribution shift:** Model trained on complete data sees "future" labels that aren't available at training time

**Delayed feedback correction (Chapelle et al., 2014 / JD.com 2019):**

Model conversion as a survival problem:
- P(convert | click) = pCVR
- P(convert within window W | convert) ~ exponential(λ)
- At inference time, predict pCVR ignoring timing
- Use importance weighting to correct for truncated observation window in training

```python
def compute_dfm_weight(click_time: float, 
                        conversion_time: Optional[float],
                        current_time: float,
                        elapsed_window: float = 7 * 86400) -> float:
    """
    Delayed Feedback Model importance weight.
    Upweights recently-clicked examples to correct for truncated labels.
    """
    elapsed = current_time - click_time
    if conversion_time is not None:
        # Positive example: weight = 1
        return 1.0
    else:
        # Negative so far: weight = 1 / P(no conversion observed yet)
        # Assuming exponential delay with rate lambda
        lam = 1.0 / (3 * 86400)  # avg 3-day conversion delay
        p_no_conversion_yet = math.exp(-lam * elapsed)
        return 1.0 / max(p_no_conversion_yet, 0.01)
```

### 5.3 Bid Landscape Model — Clearing Price Distribution

To bid optimally, the DSP must estimate P(win | bid=b), which requires modeling the distribution of the market clearing price (the price at which the auction clears).

**Approach: Survival model on clearing price**

Treat "did I win at bid b?" as a right-censored observation. If I bid $2.00 and lost, the clearing price was ≥$2.00 (censored). If I won at $2.00, clearing price was ≤$2.00.

```python
import numpy as np
from scipy.stats import lognorm

def fit_clearing_price_distribution(
    won_prices: np.ndarray,
    lost_bids: np.ndarray  # bids that lost; clearing price > these
) -> tuple:
    """
    Fit log-normal distribution to clearing prices using MLE
    with censored observations (Kaplan-Meier or parametric MLE).
    """
    # MLE for log-normal with right-censoring
    # won_prices: fully observed (clearing price = win price in first-price)
    # lost_bids: right-censored at lost bid value
    log_won = np.log(won_prices + 1e-6)
    mu_init = log_won.mean()
    sigma_init = log_won.std()
    
    # In practice: use lifelines.WeibullFitter or statsmodels
    # for proper censored MLE
    return mu_init, sigma_init

def win_probability(bid: float, mu: float, sigma: float) -> float:
    """P(win | bid=b) = P(clearing_price <= b) = CDF of log-normal."""
    return lognorm.cdf(bid, s=sigma, scale=np.exp(mu))
```

---

## 6. Bid Calculation

### Optimal Bid Formula

For a performance campaign maximizing conversions subject to budget constraint:

$$\text{bid}^* = \text{pCTR} \times \text{pCVR} \times \text{CPA\_target} \times \text{win\_adjustment}$$

Where `win_adjustment` accounts for first-price auction dynamics:

- **Second-price (Vickrey):** Bid = true value = pCTR × pCVR × CPA_target. Dominant strategy; no strategic shading needed.
- **First-price:** You pay what you bid. Optimal bid < true value. Bid shading required.

### Bid Shading (First-Price Auctions)

In a first-price auction, bidding true value gives zero surplus. Optimal strategy (from auction theory):

$$b^* = v - \frac{1 - F(v)}{f(v)}$$

where v = true value, F = CDF of competitors' values, f = PDF.

Since F is unknown, estimate it empirically from bid landscape model:

```python
def shade_bid(true_value: float, 
              clearing_price_mu: float,
              clearing_price_sigma: float,
              shading_factor: float = 0.85) -> float:
    """
    First-price bid shading.
    Shade toward expected clearing price to retain positive surplus.
    """
    # Expected clearing price (given we would win at true_value)
    # E[clearing_price | clearing_price < true_value]
    from scipy.stats import lognorm
    truncated_mean = lognorm.expect(
        lambda x: x,
        args=(clearing_price_sigma,),
        scale=np.exp(clearing_price_mu),
        lb=0, ub=true_value
    )
    
    # Blend between true value and expected clearing price
    shaded_bid = shading_factor * true_value + (1 - shading_factor) * truncated_mean
    return max(shaded_bid, floor_price)  # never bid below floor
```

### Budget-Constrained Bidding (KKT Conditions)

For a campaign with daily budget B and N auction opportunities, the Lagrangian optimization (from Perlich et al., KDD 2012):

$$\max_{b_i} \sum_i P(\text{win}_i | b_i) \cdot v_i \quad \text{s.t.} \quad \sum_i P(\text{win}_i | b_i) \cdot b_i \leq B$$

KKT conditions yield:

$$b_i^* = v_i \cdot \frac{1}{1 + \lambda^*}$$

where λ* is the shadow price of the budget constraint (Lagrange multiplier). λ* > 0 means budget is binding; bidder shades all bids by factor 1/(1+λ*).

In practice, λ* is tuned by the pacing controller to maintain target spend rate.

```python
class KKTBidder:
    def __init__(self, daily_budget: float, target_cpa: float):
        self.daily_budget = daily_budget
        self.target_cpa = target_cpa
        self.lambda_multiplier = 1.0  # shadow price, updated by pacer
    
    def compute_bid(self, p_ctr: float, p_cvr: float) -> float:
        true_value = p_ctr * p_cvr * self.target_cpa
        shaded_bid = true_value / (1.0 + self.lambda_multiplier)
        return shaded_bid
    
    def update_lambda(self, spend_rate: float, target_spend_rate: float):
        """Increase lambda if overspending, decrease if underspending."""
        if spend_rate > target_spend_rate * 1.05:
            self.lambda_multiplier *= 1.1   # tighten
        elif spend_rate < target_spend_rate * 0.95:
            self.lambda_multiplier *= 0.9   # loosen
        self.lambda_multiplier = max(0.0, min(self.lambda_multiplier, 10.0))
```

### Throttling Strategies

When budget is constrained, two strategies exist:

1. **Bid shading:** Participate in all auctions but bid lower → lower win rate
2. **Bid throttling (random dropping):** Skip a fraction of auctions, bid full value on remainder → maintains win price, reduces volume

Throttling is preferred when the audience is highly selective (not all impressions are equally valuable) and you want to preserve the bid signal for model training.

---

## 7. Budget Pacing

### Problem

A $10,000 daily budget must be spent as smoothly as possible across 24 hours. Naive bidding (bid on every opportunity) exhausts budget by 10am, missing afternoon/evening peak traffic.

### Pacing Modes

| Mode | Behavior | Use Case |
|---|---|---|
| Even/Uniform | Distribute spend proportionally to traffic volume | Standard performance campaigns |
| ASAP | Spend as fast as possible until exhausted | Flash sales, time-critical promotions |
| Dayparting | Concentrate budget in specified hours | Campaigns targeting business hours |
| Objective-based | Maximize conversions; let pacer allocate hours | Smart bidding (Google's approach) |

### PID Controller for Spend Rate

Model budget pacing as a control theory problem. The "plant" is the bidding system; the "setpoint" is target spend rate.

```python
class PacingPIDController:
    """
    PID controller for budget pacing.
    Controls lambda_multiplier (bid dampener) to hit target spend rate.
    """
    
    def __init__(self, kp=0.5, ki=0.1, kd=0.05, update_interval_s=60):
        self.kp = kp    # proportional gain
        self.ki = ki    # integral gain (corrects persistent offset)
        self.kd = kd    # derivative gain (dampens oscillation)
        self.integral = 0.0
        self.prev_error = 0.0
        self.update_interval = update_interval_s
        self.lambda_multiplier = 1.0
    
    def update(self, 
               actual_spend_rate: float,   # $/min actual
               target_spend_rate: float,   # $/min target
               remaining_budget: float,
               remaining_time_s: float) -> float:
        
        # Dynamic target: adjust for remaining budget and time
        adjusted_target = remaining_budget / max(remaining_time_s / 60, 1)
        
        error = actual_spend_rate - adjusted_target
        self.integral += error * self.update_interval
        derivative = (error - self.prev_error) / self.update_interval
        
        delta = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.lambda_multiplier = max(0.0, self.lambda_multiplier + delta)
        self.prev_error = error
        
        return self.lambda_multiplier
```

### Daily Budget Exhaustion Prediction

To prevent over/under-delivery, predict end-of-day spend given current trajectory:

$$\hat{S}_{day} = S_{current} + \hat{R}_{remaining} \times \bar{CPM} \times \text{win\_rate}$$

where R_remaining is forecasted remaining impression volume (from historical traffic patterns, hour-of-week model).

Budget guard rails:
- Hard stop: atomic counter in Redis, bid server checks before every response
- Soft stop at 98%: reduce lambda aggressively to decelerate gracefully
- Over-delivery cap: never exceed 110% of daily budget (contractual SLA)

```python
class BudgetGuard:
    def __init__(self, redis_client, campaign_id: str, daily_budget_cents: int):
        self.redis = redis_client
        self.key = f"budget:{campaign_id}:spent_today"
        self.limit = int(daily_budget_cents * 1.10)  # 10% over-delivery cap
    
    def try_spend(self, amount_cents: int) -> bool:
        """Atomic check-and-increment. Returns False if budget exhausted."""
        new_val = self.redis.incrby(self.key, amount_cents)
        if new_val > self.limit:
            self.redis.decrby(self.key, amount_cents)  # rollback
            return False
        return True
```

---

## 8. Attribution & Feedback Loop

### Attribution Models

| Model | Description | Bias | Use Case |
|---|---|---|---|
| Last-click | 100% credit to last clicked ad | Overvalues bottom-funnel retargeting | Baseline, easy to implement |
| First-click | 100% credit to first touchpoint | Overvalues awareness campaigns | Brand lift measurement |
| Linear | Equal credit across all touchpoints | Treats all touches as equally valuable | Multi-channel reporting |
| Time-decay | More credit to recent touchpoints | Recency bias | Short purchase cycles |
| Data-driven (DDA) | ML model assigns credit based on incrementality | Requires scale (>10K conversions/month) | Google Smart Bidding, large DSPs |

### Delayed Conversion Pipeline

```
Click event (T+0)                Conversion event (T+7 days)
      │                                    │
      ▼                                    ▼
Kafka topic:                         Kafka topic:
  click_stream                        pixel_fires
      │                                    │
      └──────────────────┬─────────────────┘
                         ▼
               ┌─────────────────┐
               │   Flink Join    │  Join on user_id + attribution window
               │  (7-day state)  │  State TTL = 30 days
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  Attribution    │  Assign credit using chosen model
               │  Engine         │
               └────────┬────────┘
                        │
               ┌────────▼────────┐   ┌──────────────┐
               │  Training       │──►│  Model       │
               │  Label Store    │   │  Retraining  │
               └─────────────────┘   └──────────────┘
```

### Counterfactual Attribution

Standard attribution asks "who got credit for the conversion?" but the causal question is "did the ad *cause* the conversion?" Answering this requires a holdout/control group:

- **Ghost bidding (PSA holdout):** Win the auction but serve a PSA (public service ad) instead of the campaign ad. Measure conversion rate of holdout vs. exposed group.
- **Geo holdout:** Suppress bidding in randomly selected DMAs; compare conversion rates.
- **Conversion lift study:** Run prospecting campaign; randomly assign 10% of eligible users to holdout.

```python
def estimate_incrementality(
    exposed_conversions: int,
    exposed_impressions: int,
    control_conversions: int, 
    control_impressions: int
) -> dict:
    """Estimate incremental conversion rate using holdout test."""
    cvr_exposed = exposed_conversions / exposed_impressions
    cvr_control = control_conversions / control_impressions
    
    incremental_cvr = cvr_exposed - cvr_control
    iROAS = incremental_cvr / (1 / cpm)  # incremental conversions per dollar
    
    # Statistical significance (two-proportion z-test)
    from statsmodels.stats.proportion import proportions_ztest
    z_stat, p_value = proportions_ztest(
        [exposed_conversions, control_conversions],
        [exposed_impressions, control_impressions]
    )
    
    return {
        "incremental_cvr": incremental_cvr,
        "lift_pct": incremental_cvr / cvr_control * 100,
        "p_value": p_value,
        "significant": p_value < 0.05
    }
```

### Multi-Touch Attribution (MTA) Model

For advertisers with multiple campaigns (display + search + social), a Shapley value-based MTA model distributes credit fairly:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [v(S \cup \{i\}) - v(S)]$$

where v(S) = conversion rate when touchpoint set S is present.

---

## 9. Privacy Constraints

### GDPR / CCPA Impact

| Requirement | Impact on RTB |
|---|---|
| Consent for behavioral targeting | TCF 2.0 consent string in bid request; must check before using user ID |
| Right to erasure | User ID deletion must propagate to feature store, training data, model parameters |
| Data minimization | Cannot store raw bid requests indefinitely; aggregation required after 30 days |
| No sensitive category targeting | Religion, health, political — must filter IAB sensitive categories |
| Purpose limitation | Data collected for click attribution cannot be repurposed for lookalike modeling without fresh consent |

**TCF 2.0 check in hot path:**

```python
def check_consent(tcf_string: str, purpose_id: int) -> bool:
    """Parse IAB TCF 2.0 consent string to verify user consent for purpose."""
    from iab_tcf import decode_tc_string
    tc = decode_tc_string(tcf_string)
    return tc.purpose_consents.get(purpose_id, False)

# Purpose IDs:
# 1 = Store/access device info (cookies)
# 3 = Create personalised ad profile
# 4 = Select personalised ads
```

### Cookie Deprecation & Privacy Sandbox

Chrome's Privacy Sandbox replaces third-party cookies with:

**Topics API:** Browser classifies user's recent browsing into ~350 topics (IAB taxonomy subset). Provides up to 3 topics per request, randomly sampled from recent weeks.

```python
# Topics API response in bid request (proposed OpenRTB extension)
# user.ext.topics = [{"segtax": 600, "segment": [{"id": "39"}]}]
# Segment 39 = "Sports/Olympics"

def extract_topics_features(bid_request: BidRequest) -> List[int]:
    """Extract Privacy Sandbox topics as contextual features."""
    topics = []
    if bid_request.user.ext and "topics" in bid_request.user.ext:
        for topic_group in bid_request.user.ext.topics:
            if topic_group.get("segtax") == 600:  # IAB topics taxonomy
                topics.extend([int(s["id"]) for s in topic_group.get("segment", [])])
    return topics  # up to 3 topic IDs; use as categorical features
```

**Protected Audience API (FLEDGE):** Remarketing lists stored in browser; auction runs in browser sandbox. DSP cannot observe individual user IDs. Signals back to bidder are aggregated via Private Aggregation API.

### Contextual Targeting Fallback

When user ID is absent (Safari, Firefox, opted-out Chrome):

```
Feature priority (no cookie):
  1. Page content embedding (URL → topic vector via NLP model)
  2. IAB content categories from bid request
  3. Device type (mobile/desktop/tablet)
  4. OS/browser signal
  5. Geographic (DMA, country)
  6. Time-of-day, day-of-week
  7. Deal ID / PMP targeting (premium inventory)
```

Contextual-only models achieve ~60–70% of the AUC of user-identified models for prospecting. Retargeting (which requires user ID) falls to near-baseline without cookies.

---

## 10. Failure Modes

### 10.1 Bid Flooding / Traffic Quality Attacks

**Symptom:** Win rate spikes abnormally, spend accelerates, conversions don't follow.
**Root cause:** Bot/IVT (Invalid Traffic) — automated scripts generating fake bid requests on low-quality publisher inventory.

**Detection:**
- Anomaly detection on win-rate-to-conversion-rate ratio (should be stable; IVT decouples them)
- Seller.json + ads.txt validation: reject inventory from unauthorized resellers
- Data center IP range blocking: data center IPs should not appear as user IPs
- User agent consistency check: UA string vs. device properties mismatch

```python
def score_traffic_quality(bid_request: BidRequest) -> float:
    """Return IVT probability. Score > 0.7 → no-bid."""
    signals = []
    
    # Data center IP check
    if is_datacenter_ip(bid_request.device.ip):
        signals.append(0.9)
    
    # UA/device consistency
    parsed_ua = parse_user_agent(bid_request.device.ua)
    if parsed_ua.device_type != bid_request.device.devicetype:
        signals.append(0.8)
    
    # Velocity: same user ID in >100 auctions/minute is anomalous
    uid_velocity = get_uid_velocity(bid_request.user.id)
    if uid_velocity > 100:
        signals.append(min(uid_velocity / 1000, 1.0))
    
    return max(signals) if signals else 0.0
```

### 10.2 Click Fraud

**Symptom:** High CTR on publisher, low post-click engagement, high bounce rate.
**Mitigation:** Click validation pipeline — filter clicks with no subsequent page interaction within 30 seconds; use click-through URL with redirect through click verification service; integrate with GIVT/SIVT classifications from IAS or DoubleVerify.

### 10.3 Latency Spikes → Timeout Cascade

**Symptom:** Win rate drops to near-zero; no bid responses sent.
**Root cause:** Redis latency spike (GC pause, network hiccup) → feature lookup takes 8ms → total request time > 10ms → all requests timeout.

**Mitigations:**
- Fallback features: if Redis lookup times out at 2ms, use average features for the campaign (stale but non-blocking)
- Circuit breaker: if Redis error rate > 5%, bypass feature lookup entirely and use default CTR for campaign
- Latency SLO alerting: p99 latency > 6ms triggers PagerDuty before the 10ms wall is hit
- Shadow traffic: route 1% of bid requests to canary serving stack; measure latency distribution before full rollout

```python
def fetch_user_features_with_fallback(
    user_id: str,
    redis_client,
    timeout_ms: int = 2
) -> dict:
    try:
        features = redis_client.hgetall(
            f"user:{user_id}:features",
            timeout=timeout_ms / 1000
        )
        return features if features else get_default_features()
    except (redis.TimeoutError, redis.ConnectionError):
        METRICS.increment("feature_lookup.timeout")
        return get_default_features()  # pre-computed averages
```

### 10.4 Budget Over-Delivery

**Symptom:** Campaign spends 120% of daily budget.
**Root cause:** Race condition in distributed budget check — multiple bid servers decremented concurrently; Redis counter not atomic, or Lua script error.

**Fix:** Use Redis `INCRBY` (atomic) not read-then-write. Implement a two-level budget check: local token bucket (fast, per-process, approximate) + global Redis counter (authoritative, checked every N impressions).

```python
class TwoLevelBudgetGuard:
    def __init__(self, global_redis, campaign_id, daily_budget_cents):
        self.redis = global_redis
        self.campaign_id = campaign_id
        self.local_tokens = 0          # local bucket (allows batching)
        self.local_batch_size = 100    # refill local from global every 100 cents
        self.daily_limit = int(daily_budget_cents * 1.02)
    
    def try_spend(self, amount_cents: int) -> bool:
        if self.local_tokens >= amount_cents:
            self.local_tokens -= amount_cents
            return True
        # Refill local from global (atomic)
        new_global = self.redis.incrby(
            f"budget:{self.campaign_id}:spent", self.local_batch_size
        )
        if new_global > self.daily_limit:
            self.redis.decrby(f"budget:{self.campaign_id}:spent", self.local_batch_size)
            return False
        self.local_tokens = self.local_batch_size - amount_cents
        return True
```

### 10.5 Pacing PID Oscillation

**Symptom:** Spend alternates between zero (over-throttled) and burst (under-throttled) in 15-minute cycles.
**Root cause:** Integral windup in PID controller. When budget is fully exhausted, integral term accumulates a large negative value; when budget resets (next day), controller overcorrects.

**Fix:** Anti-windup: clamp integral term when actuator (lambda) is saturated. Reset integral at budget boundary.

### 10.6 Attribution Window Mismatch

**Symptom:** Campaign shows positive ROAS in DSP dashboard but negative ROI in advertiser's own analytics.
**Root cause:** DSP uses 30-day post-click window; advertiser CRM uses 7-day. Conversions from day 8–30 counted by DSP but not by advertiser.

**Fix:** Align attribution windows in contract/insertion order. Expose configurable window in reporting API. Always report both 7-day and 30-day numbers. For model training, train pCVR models with the contractually defined window.

### 10.7 Model Staleness During Traffic Spikes

**Symptom:** CTR predictions systematically low during major events (Super Bowl, Black Friday); win rate drops as competitors bid higher.
**Root cause:** Model trained on historical average traffic; event traffic has different user intent and higher advertiser competition (CPMs spike 3–5x).

**Mitigation:** Online learning (FTRL update from live data), scheduled model hotswaps with event-aware priors, real-time bid floor signal as feature.

---

## 11. Interview Q&A

**Q1: Why is calibration more important than AUC for a bidding model?**

AUC measures ranking quality — whether the model correctly orders impressions by value. A model with high AUC but poor calibration still ranks impressions correctly, but the absolute bid prices are wrong. If pCTR is 3x overestimated, the DSP bids 3x too much and destroys advertiser ROI. Budget constraints mean overbidding on every impression compounds quickly. The bid formula `bid = pCTR × pCVR × CPA_target` requires both the ranking AND the scale to be correct. ECE < 0.005 is a production gate; AUC improvement from 0.73 to 0.75 is secondary.

**Q2: How do you handle the delayed feedback problem in pCVR training?**

Conversion labels for clicks made today may not arrive for 7–30 days. If we train on the last 24 hours, recent examples have systematically missing positive labels, causing the model to underestimate pCVR for recent user behavior and overestimate for older (complete) data. The Delayed Feedback Model (Chapelle 2014) applies importance weights inversely proportional to P(no conversion observed yet), which is estimated from the empirical conversion delay distribution. Alternatively, model conversion timing with a survival model (Weibull hazard), separate from conversion probability, and combine at inference.

**Q3: Explain bid shading and when it is/is not necessary.**

In a second-price (Vickrey) auction, truthful bidding is a dominant strategy: bid true value = pCTR × pCVR × CPA_target. You only pay the second-highest bid regardless. Shading reduces win rate without improving profit. In a first-price auction, you pay exactly what you bid. Bidding true value gives zero surplus (you win and pay full value). Optimal strategy shades toward the expected market clearing price, maintaining positive surplus. Most exchanges moved to first-price post-2019. Bid shading implementations estimate the clearing price distribution from historical win/loss data and blend between true value and expected clearing price. The exact shade factor depends on campaign competitiveness and inventory scarcity.

**Q4: How does the PID pacing controller handle sudden traffic drops (e.g., midnight in target timezone)?**

A pure PID controller will see spend rate drop to near zero (midnight = low traffic) and drastically lower lambda (open the spend tap wide open). When traffic resumes at 8am, it will overbid aggressively. Fixes: (1) traffic-aware target: set target_spend_rate = budget_remaining / expected_impressions_remaining (using a time-of-day traffic forecast), not a flat linear schedule; (2) feedforward term: add a term proportional to the traffic forecast change rate; (3) clamp lambda to [0.1, 5.0] to prevent extreme oscillation. Google's Smart Bidding uses a learned target-spend allocation across hours rather than a PID.

**Q5: How would you design the system for a world without third-party cookies?**

Three parallel tracks: (1) **Contextual targeting:** NLP model on page content (URL + title + body snippet) → topic embeddings; train pCTR on (contextual_features, ad_features) without any user ID; achieves ~60–70% of AUC of user-identified model. (2) **First-party data activation:** Publisher first-party IDs (logged-in users), advertiser CRM data matched via clean room (InfoSum, LiveRamp Data Clean Room), hashed email match. (3) **Privacy Sandbox:** Implement FLEDGE for retargeting (interest groups stored in browser), Topics API for interest-based prospecting, Private Aggregation API for conversion measurement. Measurement strategy shifts from user-level attribution to aggregate/modeled attribution (Privacy Preserving Attribution, MPC-based).

**Q6: A campaign has a 5% win rate. How do you diagnose whether this is a feature or a budget problem?**

First, check the bid price distribution vs. clearing price distribution: if DSP bids $1.50 and median clearing price is $4.00, the campaign is systematically underbidding (model or CPA target issue, not budget). If bid prices are competitive but win rate is still low (5%), check: (a) targeting too narrow — small audience = scarce supply, high competition; (b) frequency cap — user already seen ad N times, excluded from bidding; (c) creative quality score — some exchanges apply quality multipliers; (d) budget pacing throttle — lambda too high → bids shaded too aggressively. Use bid landscape visualization: plot win rate as a function of bid price (empirical P(win|b)). If the curve is steep (small bid increase → large win rate gain), increasing bids is ROI-positive. If the curve is flat, the inventory is genuinely scarce at any price.

**Q7: How would you prevent the training-serving skew in this system?**

RTB has severe training-serving skew because: (1) bid requests are filtered before ML scoring (only 5% of eligible requests reach the model); (2) only won impressions generate training labels; (3) feature computation at training time (batch) differs from serving (real-time). Mitigations: log the exact feature vector used at bid time alongside the bid decision (prediction logging); train only on impressions that were actually served (won); for exposure-selection bias, use inverse propensity scoring — weight each training example by 1/P(selected) where P(selected) is the probability the request passed all filters. For feature skew: generate training features by replaying logged feature values rather than recomputing from raw signals.

---

## 12. Key Numbers to Memorize

| Metric | Value |
|---|---|
| OpenRTB hard deadline | 100ms spec; 10–50ms exchange enforcement |
| Typical DSP win rate | 3–15% (depends on category competitiveness) |
| Average display CPM | $2–$10 (open exchange); $15–$50 (PMPs) |
| Video CPM premium | 4–8x display CPM |
| pCTR range | 0.05%–0.5% display; 0.5%–2% search |
| Conversion rate (post-click) | 1–5% e-commerce |
| Attribution delay | 7–30 days typical; 90 days some verticals |
| Feature freshness target | <5 min for recency signals; 1h for aggregates |
| Model retraining frequency | Daily (batch); every 15 min (FTRL online) |
| Budget over-delivery tolerance | ±10% contractual SLA |

---

## 13. References & Further Reading

- **OpenRTB 2.6 Spec** — IAB Tech Lab (protocol schema and timing requirements)
- **Chapelle, O. (2014)** — "Modeling Delayed Feedback in Display Advertising" (KDD 2014; DFM for pCVR)
- **Zhang, W. et al. (2014)** — "Optimal Real-Time Bidding for Display Advertising" (KDD 2014; KKT conditions)
- **McMahan, H.B. et al. (2013)** — "Ad Click Prediction: A View from the Trenches" (KDD 2013; FTRL at Google scale)
- **Perlich, C. et al. (2012)** — "Bid Optimizing and Inventory Scoring in Targeted Online Advertising" (KDD 2012)
- **Criteo AI Lab** — "A Practical Exploration of Echo Chamber Effects in Online Advertising" (offline evaluation bias)
- **Google Privacy Sandbox** — Protected Audience API spec (FLEDGE successor)
- **IAB Tech Lab** — Seller.json, ads.txt, SupplyChain Object (supply path validation)

## Flashcards

**DSP or SSP perspective? Demand-Side Platform (DSP) bids to buy impressions; Supply-Side Platform (SSP) runs the auction and passes through bids. Most ML complexity lives on the DSP side.?** #flashcard
DSP or SSP perspective? Demand-Side Platform (DSP) bids to buy impressions; Supply-Side Platform (SSP) runs the auction and passes through bids. Most ML complexity lives on the DSP side.

**Display vs. video? Video CPMs are 5–10x higher; viewability and completion rate replace CTR as primary signals. Video allows pre-roll, mid-roll, outstream; each has different user intent.?** #flashcard
Display vs. video? Video CPMs are 5–10x higher; viewability and completion rate replace CTR as primary signals. Video allows pre-roll, mid-roll, outstream; each has different user intent.

**Brand vs. performance campaign? Brand?** #flashcard
optimize reach, frequency, viewability, brand lift lift surveys. Performance: optimize CPA, ROAS, down-funnel conversions. Model objectives and bid formulas differ fundamentally.

**First-price vs. second-price auction? Most exchanges moved to first-price (2019+). Bid shading is mandatory in first-price; not needed in second-price (Vickrey). The system must know the auction type per exchange.?** #flashcard
First-price vs. second-price auction? Most exchanges moved to first-price (2019+). Bid shading is mandatory in first-price; not needed in second-price (Vickrey). The system must know the auction type per exchange.

**Retargeting vs. prospecting? Retargeting (known converters)?** #flashcard
high signal, small audience. Prospecting (lookalike): relies heavily on pCTR/pCVR model generalization.

**What conversion window? 7-day, 30-day, 60-day post-click? Longer windows mean delayed labels, which break standard online learning and require survival modeling.?** #flashcard
What conversion window? 7-day, 30-day, 60-day post-click? Longer windows mean delayed labels, which break standard online learning and require survival modeling.

**Privacy regime? EU GDPR, US CCPA/CPRA, Canada PIPEDA. Cookieless environments (Safari ITP, Chrome Privacy Sandbox) affect user ID resolution fundamentally.?** #flashcard
Privacy regime? EU GDPR, US CCPA/CPRA, Canada PIPEDA. Cookieless environments (Safari ITP, Chrome Privacy Sandbox) affect user ID resolution fundamentally.

**Budget allocation? Fixed daily budget per campaign, or portfolio bidding across multiple campaigns with shared budget?** #flashcard
Budget allocation? Fixed daily budget per campaign, or portfolio bidding across multiple campaigns with shared budget?

**Network transit?** #flashcard
~2–4ms (co-located PoP)

**Bid request parsing?** #flashcard
~0.5ms

**User ID lookup?** #flashcard
~1ms (in-memory hash map or local Redis)

**Feature enrichment?** #flashcard
~1–2ms (pre-computed batch features)

**ML inference?** #flashcard
~2–4ms (target for pCTR + pCVR)

**Bid calculation + response serialization?** #flashcard
~0.5ms

**Latency budget: 10ms p99?** #flashcard
not p50. p99 must hold under traffic spikes.

**Availability?** #flashcard
99.99% uptime. A 60-second outage = 30M+ missed bid opportunities.

**Model size?** #flashcard
pCTR model must fit in L3 cache or at most local DRAM. No network calls to model servers in hot path.

**Feature freshness?** #flashcard
User recency signals (last-click, session activity) must be ≤5 minutes stale.

**Budget atomicity?** #flashcard
No overshooting daily budget by >1%. Requires distributed rate limiting with strong consistency for high-spend campaigns.

**imp.bidfloor?** #flashcard
Minimum acceptable bid (hard constraint on bid price)

**user.buyeruid?** #flashcard
DSP-mapped user ID (result of cookie sync)

**site.cat?** #flashcard
IAB content categories (contextual signal)

**device.*?** #flashcard
Device type, OS, browser (feature engineering)

**at: Auction type?** #flashcard
1=first-price, 2=second-price

**Unified ID 2.0 (UID2)?** #flashcard
Hashed+encrypted email/phone; DSP must hold decryption keys

**Google Privacy Sandbox Topics API?** #flashcard
Browser returns 3 interest topics per request; no cross-site user identifier

**Contextual fallback?** #flashcard
When no user ID is available, rely entirely on page context, device type, time-of-day features

**Inference is a dot product?** #flashcard
O(n_features) in microseconds

**L1 regularization produces sparse models (millions of features, ~1M nonzero weights)?** #flashcard
L1 regularization produces sparse models (millions of features, ~1M nonzero weights)

**Online updates?** #flashcard
weight updates can be applied to the serving model every few minutes without redeployment

**Interpretable?** #flashcard
SHAP values map directly to feature weights

**P(convert | click) = pCVR?** #flashcard
P(convert | click) = pCVR

**P(convert within window W | convert) ~ exponential(λ)?** #flashcard
P(convert within window W | convert) ~ exponential(λ)

**At inference time, predict pCVR ignoring timing?** #flashcard
At inference time, predict pCVR ignoring timing

**Use importance weighting to correct for truncated observation window in training?** #flashcard
Use importance weighting to correct for truncated observation window in training

**Second-price (Vickrey)?** #flashcard
Bid = true value = pCTR × pCVR × CPA_target. Dominant strategy; no strategic shading needed.

**First-price?** #flashcard
You pay what you bid. Optimal bid < true value. Bid shading required.

**Hard stop?** #flashcard
atomic counter in Redis, bid server checks before every response

**Soft stop at 98%?** #flashcard
reduce lambda aggressively to decelerate gracefully

**Over-delivery cap?** #flashcard
never exceed 110% of daily budget (contractual SLA)

**Ghost bidding (PSA holdout)?** #flashcard
Win the auction but serve a PSA (public service ad) instead of the campaign ad. Measure conversion rate of holdout vs. exposed group.

**Geo holdout?** #flashcard
Suppress bidding in randomly selected DMAs; compare conversion rates.

**Conversion lift study?** #flashcard
Run prospecting campaign; randomly assign 10% of eligible users to holdout.

**Anomaly detection on win-rate-to-conversion-rate ratio (should be stable; IVT decouples them)?** #flashcard
Anomaly detection on win-rate-to-conversion-rate ratio (should be stable; IVT decouples them)

**Seller.json + ads.txt validation?** #flashcard
reject inventory from unauthorized resellers

**Data center IP range blocking?** #flashcard
data center IPs should not appear as user IPs

**User agent consistency check?** #flashcard
UA string vs. device properties mismatch

**Fallback features?** #flashcard
if Redis lookup times out at 2ms, use average features for the campaign (stale but non-blocking)

**Circuit breaker?** #flashcard
if Redis error rate > 5%, bypass feature lookup entirely and use default CTR for campaign

**Latency SLO alerting?** #flashcard
p99 latency > 6ms triggers PagerDuty before the 10ms wall is hit

**Shadow traffic?** #flashcard
route 1% of bid requests to canary serving stack; measure latency distribution before full rollout

**OpenRTB 2.6 Spec?** #flashcard
IAB Tech Lab (protocol schema and timing requirements)

**Chapelle, O. (2014)?** #flashcard
"Modeling Delayed Feedback in Display Advertising" (KDD 2014; DFM for pCVR)

**Zhang, W. et al. (2014)?** #flashcard
"Optimal Real-Time Bidding for Display Advertising" (KDD 2014; KKT conditions)

**McMahan, H.B. et al. (2013)?** #flashcard
"Ad Click Prediction: A View from the Trenches" (KDD 2013; FTRL at Google scale)

**Perlich, C. et al. (2012)?** #flashcard
"Bid Optimizing and Inventory Scoring in Targeted Online Advertising" (KDD 2012)

**Criteo AI Lab?** #flashcard
"A Practical Exploration of Echo Chamber Effects in Online Advertising" (offline evaluation bias)

**Google Privacy Sandbox?** #flashcard
Protected Audience API spec (FLEDGE successor)

**IAB Tech Lab?** #flashcard
Seller.json, ads.txt, SupplyChain Object (supply path validation)
