---
module: System Design
topic: System Design
subtopic: Real Time Bidding System
status: unread
tags: [productionml, ml, system-design-real-time-biddin]
---
# Real-Time Bidding (RTB) / Programmatic Advertising ML System

End-to-end ML system for programmatic display and video advertising. A canonical system design question at ad-tech companies (The Trade Desk, Criteo, Google DV360, Amazon DSP).

**Scale:** 500K+ bid requests/second, 10ms hard deadline (OpenRTB spec), 100B+ auctions/day.

---

## 1. Problem Framing

### Clarifying Questions

- **DSP or SSP perspective?** DSP (Demand-Side Platform) bids to buy impressions; SSP (Supply-Side Platform) runs the auction. Most ML complexity lives on the DSP side.
- **Display vs. video?** Video CPMs are 5–10x higher; viewability and completion rate replace CTR as the primary signal.
- **Brand vs. performance campaign?** Brand optimizes reach/viewability/lift; performance optimizes CPA/ROAS. Model objectives and bid formulas differ.
- **First-price vs. second-price auction?** Most exchanges moved to first-price (2019+). Bid shading is mandatory in first-price, unnecessary in second-price (Vickrey).
- **Retargeting vs. prospecting?** Retargeting (known converters): high signal, small audience. Prospecting: relies on pCTR/pCVR generalization.
- **Conversion window?** 7/30/60-day post-click. Longer windows mean more delayed labels, which break naive online learning.
- **Privacy regime?** GDPR/CCPA plus cookieless environments (Safari ITP, Chrome Privacy Sandbox) affect user ID resolution.
- **Budget allocation?** Fixed daily budget per campaign vs. portfolio bidding with shared budget.

### Business Metrics

| Metric | Definition | Why It Matters |
|---|---|---|
| ROAS | Revenue / Ad Spend | Primary advertiser KPI for performance campaigns |
| CPA | Cost per Acquisition | Cost efficiency tied to pacing |
| Win Rate | Impressions won / Bid requests | Low = under-spending; high = over-paying |
| Spend Efficiency | Actual spend / Budget target | Over-delivery = overspend; under-delivery = churn |
| CPM | Cost per thousand impressions | Raw buying cost, normalized by audience quality |
| Viewability | Viewable / Served | Only viewable impressions deliver brand value |

### ML Metrics

| Model | Primary Metric | Secondary Metric | Failure Mode |
|---|---|---|---|
| pCTR | Log-loss (calibration) | AUC-ROC (ranking) | Overconfident → overbid; underconfident → underbid |
| pCVR | Expected Calibration Error (ECE) | PR-AUC | Delayed labels cause distribution shift |
| Bid Landscape | RMSE of clearing price percentile | Coverage at p50/p90 | Underestimation → lost auctions |
| Win Rate Model | Brier score | Calibration curve | Mis-estimation → wrong throttle factor |

**Calibration is non-negotiable.** An uncalibrated pCTR of 0.05 when the truth is 0.01 causes a 5x overbid. AUC can be 0.75, but if calibration is off by 2x the campaign loses money.

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

Target ECE < 0.005 for production bidding models.

---

## 2. Scale & Hard Constraints

### The 10ms Problem

OpenRTB mandates a DSP response within 100ms, but exchanges enforce 10–50ms before treating a slow response as a no-bid. Budget breakdown for a ~10ms end-to-end path:

- Network transit: ~2–4ms (co-located PoP)
- Bid request parsing: ~0.5ms
- User ID lookup: ~1ms (in-memory hash map or local Redis)
- Feature enrichment: ~1–2ms (pre-computed batch features)
- ML inference: ~2–4ms (pCTR + pCVR)
- Bid calculation + response serialization: ~0.5ms

This rules out any model needing sequential DB lookups, a Python GIL, or network I/O in the hot path.

### Scale Estimates

```
500K bid requests/sec × 10ms timeout
= 5,000 concurrent in-flight requests per PoP

100B auctions/day ≈ 1.16M requests/sec peak
(500K is p50; plan for ~1.5M rps burst)

Win rate ~5–10%: 5M–10M won impressions/day needing attribution
CTR ~0.1%: 5K–10K clicks/day per mid-size campaign
CVR ~1–5% of clicks: 50–500 conversions/day → sparse, delayed feedback
```

### Infrastructure Constraints

- **Latency budget:** 10ms p99, not p50 — must hold under traffic spikes.
- **Availability:** 99.99% uptime. A 60-second outage = 30M+ missed bid opportunities.
- **Model size:** pCTR model must fit in L3 cache or local DRAM — no network calls in the hot path.
- **Feature freshness:** recency signals must be ≤5 minutes stale.
- **Budget atomicity:** no overshooting daily budget by >1%; needs distributed rate limiting with strong consistency for high-spend campaigns.

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
 │             │                                   │  (C++/Go/Rust)  │
 └─────────────┘                                   └────────┬────────┘
        │                                 ┌────────────────┼──────────────────┐
        │  Auction                        ▼                ▼                  ▼
        ▼                             ┌───────┐     ┌──────────┐     ┌───────────────┐
 ┌─────────────┐                     │  ID   │     │ Feature  │     │  ML Scoring   │
 │  Win Notice │                     │Resolve│     │  Lookup  │     │  Service      │
 │  (event URL)│                     │Service│     │  (Redis  │     │  pCTR + pCVR  │
 └──────┬──────┘                     └───────┘     │  ~1ms)   │     │  (ONNX/TRT)   │
        │                                            └──────────┘     └───────────────┘
        │  Impression/Click/Conversion                                       │
        ▼                                                            ┌───────▼──────┐
 ┌─────────────────────────────┐                                    │  Bid Price   │
 │   Attribution Pipeline      │◄──── Conversion Pixels ───────────│  Calculator  │
 │   (Kafka → Flink)           │                                    └──────────────┘
 └──────────────┬──────────────┘                                           │
                ▼                                                           │
 ┌─────────────────────────────┐              ┌────────────────────────────┘
 │   Training Data Store       │              │  Real-time feedback
 │   (S3 + feature store)      │              ▼
 └──────────────┬──────────────┘    ┌─────────────────┐
                ▼                    │  Budget Pacing  │
 ┌─────────────────────────────┐    │  Service (PID)  │
 │   Model Training Pipeline   │    └─────────────────┘
 │   (daily retrain + FTRL)    │
 └─────────────────────────────┘
```

### Data Flow per Bid Request (10ms budget)

```
T+0ms    Bid request arrives at edge PoP
T+0.3ms  OpenRTB JSON parsed → internal proto
T+0.5ms  User ID resolved (cookie → internal GUID, or hashed email/MAID)
T+1.0ms  Targeting filter: campaign eligibility (geo, device, block-list)
T+1.5ms  Batch feature lookup from local Redis / in-process cache
T+3.5ms  pCTR inference (ONNX Runtime, quantized int8)
T+5.0ms  pCVR inference (lighter model, ~1ms)
T+6.0ms  Win rate / clearing price estimation
T+7.0ms  Bid price calculation
T+7.5ms  Budget pacing check (token bucket, atomic decrement)
T+8.0ms  Bid response serialized; ~2ms network return
```

---

## 4. Bid Request Processing

### OpenRTB Schema (Simplified)

```json
{
  "id": "abc123",
  "imp": [{"id": "1", "banner": {"w": 300, "h": 250}, "bidfloor": 0.50, "bidfloorcur": "USD"}],
  "site": {"domain": "example.com", "cat": ["IAB1"], "page": "https://example.com/article/123"},
  "user": {"id": "u789", "buyeruid": "buyer-mapped-id"},
  "device": {"ua": "Mozilla/5.0...", "ip": "203.0.113.42", "geo": {"country": "US", "metro": "807"}},
  "at": 1,
  "tmax": 100
}
```

Key fields: `imp.bidfloor` (min bid), `user.buyeruid` (DSP-mapped ID from cookie sync), `site.cat` (contextual signal), `device.*` (feature engineering), `at` (1=first-price, 2=second-price).

### User ID Resolution

- **Cookie sync (web):** SSP redirects the browser to a DSP pixel; DSP stores SSP-ID ↔ DSP-ID mapping. At bid time, `buyeruid` maps to the internal profile.
- **Mobile:** iOS IDFA (post-ATT ~30% availability), Android GAID — stored hashed (SHA-256).
- **Privacy-preserving IDs:** Unified ID 2.0 (hashed/encrypted email, needs decryption keys), Google Privacy Sandbox Topics API (3 interest topics per request, no cross-site ID), or contextual fallback (page context, device, time-of-day) when no ID is available.

```python
def resolve_user_id(bid_request: BidRequest) -> Optional[str]:
    """Resolve to internal GUID using priority chain."""
    if bid_request.user.buyeruid:
        return cookie_sync_store.get(bid_request.user.buyeruid)

    if bid_request.user.eids:
        for eid in bid_request.user.eids:
            if eid.source == "liveramp.com":
                internal_id = id_graph.lookup_rampid(eid.uids[0].id)
                if internal_id:
                    return internal_id

    # Probabilistic fingerprint, only where legally permissible
    fp = fingerprint(bid_request.device.ip, bid_request.device.ua)
    return fingerprint_store.get(fp)  # May return None → contextual-only
```

### Real-Time Feature Enrichment

Features are pre-computed offline and served from local Redis co-located with the bid server. No synchronous feature computation in the hot path.

| Feature Group | Source | Staleness | Storage |
|---|---|---|---|
| User interest vector (128-dim) | Daily batch job | 24h | Redis hash |
| User recency (last click/visit) | Stream (Flink) | 5 min | Redis string |
| Campaign stats (CTR, CVR 7d) | Hourly batch | 1h | Redis hash |
| Contextual embedding (URL → topic) | Pre-computed | — | In-process LRU |
| Historical win rate per exchange | Daily | 24h | In-process map |
| Device type → avg CTR | Weekly | 7d | In-process map |

---

## 5. ML Models

### 5.1 pCTR Model — Click Probability

**Goal:** Estimate P(click | impression, user, ad, context).

**Architecture: FTRL-Proximal (Follow The Regularized Leader)**

For latency-critical bidding, logistic regression trained with FTRL is the production workhorse (Google, Criteo, Yahoo):
- Inference is a dot product — O(n_features) in microseconds
- L1 regularization gives sparse models (millions of features, ~1M nonzero weights)
- Online updates can be pushed to the serving model every few minutes without redeployment
- Interpretable: weights map directly to SHAP-style attribution

```python
class FTRLModel:
    """FTRL-Proximal with L1/L2 regularization for online CTR prediction."""

    def __init__(self, alpha=0.1, beta=1.0, l1=1.0, l2=1.0):
        self.alpha, self.beta, self.l1, self.l2 = alpha, beta, l1, l2
        self.z = defaultdict(float)  # per-feature accumulators
        self.n = defaultdict(float)  # per-feature gradient squared sum

    def _get_weight(self, feature_id: int) -> float:
        z_i = self.z[feature_id]
        if abs(z_i) <= self.l1:
            return 0.0  # L1 sparsity
        sign = 1.0 if z_i > 0 else -1.0
        n_i = self.n[feature_id]
        return -(z_i - sign * self.l1) / ((self.beta + math.sqrt(n_i)) / self.alpha + self.l2)

    def predict(self, features: List[int]) -> float:
        score = sum(self._get_weight(f) for f in features)
        return 1.0 / (1.0 + math.exp(-score))

    def update(self, features: List[int], p: float, y: float):
        grad = p - y  # cross-entropy gradient
        for f in features:
            sigma = (math.sqrt(self.n[f] + grad**2) - math.sqrt(self.n[f])) / self.alpha
            self.z[f] += grad - sigma * self._get_weight(f)
            self.n[f] += grad**2
```

**Feature hashing:** raw features (user_id, campaign_id, domain, device_type, hour_of_week) are hashed into ~16M buckets. Cross-features (user_id × campaign_id) are captured via conjunction hashing.

**Neural upgrade** (when latency allows, e.g. for offline segment scoring): a two-tower model with user/ad/context towers concatenated into an MLP producing pCTR. Embeddings are trained offline; at inference, the user embedding is pre-fetched from Redis instead of computed.

### 5.2 pCVR Model — Conversion Probability

**Delayed feedback** is the central challenge: a click today may convert 7–30 days later. Training naively on recent data causes attribution bias (recent impressions look non-converting because conversions haven't arrived) and distribution shift.

**Delayed feedback correction (Chapelle et al., 2014):** model conversion as survival — predict pCVR ignoring timing, and use importance weighting to correct for the truncated observation window during training.

```python
def compute_dfm_weight(click_time, conversion_time, current_time, elapsed_window=7*86400):
    """Delayed Feedback Model weight: upweights recent clicks to correct truncated labels."""
    elapsed = current_time - click_time
    if conversion_time is not None:
        return 1.0  # positive example
    lam = 1.0 / (3 * 86400)  # avg 3-day conversion delay
    p_no_conversion_yet = math.exp(-lam * elapsed)
    return 1.0 / max(p_no_conversion_yet, 0.01)
```

### 5.3 Bid Landscape Model — Clearing Price Distribution

To bid optimally, the DSP estimates P(win | bid=b) from the market clearing price distribution. If a bid loses, the true clearing price is ≥ that bid (right-censored); if it wins, clearing price ≤ bid.

```python
from scipy.stats import lognorm

def fit_clearing_price_distribution(won_prices, lost_bids):
    """Fit log-normal clearing-price distribution via censored MLE
    (in practice use lifelines.WeibullFitter or statsmodels)."""
    log_won = np.log(won_prices + 1e-6)
    return log_won.mean(), log_won.std()

def win_probability(bid, mu, sigma):
    """P(win | bid=b) = CDF of log-normal clearing price."""
    return lognorm.cdf(bid, s=sigma, scale=np.exp(mu))
```

---

## 6. Bid Calculation

### Optimal Bid Formula

$$\text{bid}^* = \text{pCTR} \times \text{pCVR} \times \text{CPA\_target} \times \text{win\_adjustment}$$

- **Second-price (Vickrey):** bid = true value. Dominant strategy, no shading needed.
- **First-price:** you pay what you bid, so optimal bid < true value — shading required.

### Bid Shading (First-Price Auctions)

Bidding true value in a first-price auction gives zero surplus. Auction theory gives:

$$b^* = v - \frac{1 - F(v)}{f(v)}$$

Since the competitor value distribution F is unknown, estimate it from the bid landscape model:

```python
def shade_bid(true_value, clearing_price_mu, clearing_price_sigma, shading_factor=0.85):
    """Shade toward expected clearing price to retain positive surplus."""
    truncated_mean = lognorm.expect(
        lambda x: x, args=(clearing_price_sigma,),
        scale=np.exp(clearing_price_mu), lb=0, ub=true_value
    )
    shaded_bid = shading_factor * true_value + (1 - shading_factor) * truncated_mean
    return max(shaded_bid, floor_price)
```

### Budget-Constrained Bidding

For a campaign with daily budget B, the Lagrangian optimization (Perlich et al., KDD 2012) yields:

$$b_i^* = v_i \cdot \frac{1}{1 + \lambda^*}$$

where λ* is the shadow price of the budget constraint — tuned by the pacing controller to hit target spend rate. λ* > 0 means the budget is binding and all bids get shaded by 1/(1+λ*).

```python
class KKTBidder:
    def __init__(self, daily_budget, target_cpa):
        self.daily_budget = daily_budget
        self.target_cpa = target_cpa
        self.lambda_multiplier = 1.0  # shadow price, updated by pacer

    def compute_bid(self, p_ctr, p_cvr):
        true_value = p_ctr * p_cvr * self.target_cpa
        return true_value / (1.0 + self.lambda_multiplier)

    def update_lambda(self, spend_rate, target_spend_rate):
        if spend_rate > target_spend_rate * 1.05:
            self.lambda_multiplier *= 1.1
        elif spend_rate < target_spend_rate * 0.95:
            self.lambda_multiplier *= 0.9
        self.lambda_multiplier = max(0.0, min(self.lambda_multiplier, 10.0))
```

### Throttling Strategies

Two ways to slow spend: **shading** (bid lower on every auction) or **throttling** (skip a random fraction of auctions, bid full value on the rest). Throttling is preferred when inventory is heterogeneous and you want to preserve a clean bid signal for training.

---

## 7. Budget Pacing

### Problem

A $10,000 daily budget must spend smoothly across 24 hours. Naive bidding on every opportunity exhausts budget by mid-morning, missing afternoon/evening peak traffic.

### Pacing Modes

| Mode | Behavior | Use Case |
|---|---|---|
| Even/Uniform | Spend proportional to traffic volume | Standard performance campaigns |
| ASAP | Spend as fast as possible | Flash sales, time-critical promos |
| Dayparting | Concentrate spend in specified hours | Business-hours targeting |
| Objective-based | Maximize conversions; pacer allocates hours | Smart bidding |

### PID Controller for Spend Rate

Budget pacing is modeled as a control problem: the bidding system is the plant, target spend rate is the setpoint.

```python
class PacingPIDController:
    """Controls lambda_multiplier (bid dampener) to hit target spend rate."""

    def __init__(self, kp=0.5, ki=0.1, kd=0.05, update_interval_s=60):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.update_interval = update_interval_s
        self.lambda_multiplier = 1.0

    def update(self, actual_spend_rate, target_spend_rate, remaining_budget, remaining_time_s):
        adjusted_target = remaining_budget / max(remaining_time_s / 60, 1)
        error = actual_spend_rate - adjusted_target
        self.integral += error * self.update_interval
        derivative = (error - self.prev_error) / self.update_interval
        delta = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.lambda_multiplier = max(0.0, self.lambda_multiplier + delta)
        self.prev_error = error
        return self.lambda_multiplier
```

### Budget Guard Rails

- Hard stop: atomic counter in Redis, checked before every response
- Soft stop at 98%: decelerate spend gracefully
- Over-delivery cap: never exceed 110% of daily budget (contractual SLA)

```python
class BudgetGuard:
    def __init__(self, redis_client, campaign_id, daily_budget_cents):
        self.redis = redis_client
        self.key = f"budget:{campaign_id}:spent_today"
        self.limit = int(daily_budget_cents * 1.10)

    def try_spend(self, amount_cents: int) -> bool:
        """Atomic check-and-increment. Returns False if budget exhausted."""
        new_val = self.redis.incrby(self.key, amount_cents)
        if new_val > self.limit:
            self.redis.decrby(self.key, amount_cents)
            return False
        return True
```

---

## 8. Attribution & Feedback Loop

### Attribution Models

| Model | Description | Bias | Use Case |
|---|---|---|---|
| Last-click | 100% credit to last click | Overvalues retargeting | Baseline |
| First-click | 100% credit to first touch | Overvalues awareness | Brand lift |
| Linear | Equal credit across touches | Ignores relative value | Multi-channel reporting |
| Time-decay | More credit to recent touches | Recency bias | Short purchase cycles |
| Data-driven (DDA) | ML assigns credit by incrementality | Needs scale (>10K conv/mo) | Google Smart Bidding, large DSPs |

### Delayed Conversion Pipeline

```
Click event (T+0)                Conversion event (T+7 days)
      │                                    │
      ▼                                    ▼
click_stream (Kafka)                pixel_fires (Kafka)
      └──────────────────┬─────────────────┘
                         ▼
               Flink Join (7-day state, TTL 30 days)
                         │
                         ▼
               Attribution Engine (assigns credit)
                         │
               ┌────────▼────────┐   ┌──────────────┐
               │  Training Label │──►│    Model     │
               │     Store       │   │  Retraining  │
               └─────────────────┘   └──────────────┘
```

### Counterfactual Attribution

Standard attribution answers "who got credit," not "did the ad cause the conversion." Causal measurement needs a holdout:

- **Ghost bidding (PSA holdout):** win the auction but serve a public-service ad; compare conversion rate to the exposed group.
- **Geo holdout:** suppress bidding in randomly selected DMAs and compare.
- **Conversion lift study:** randomly assign a fraction of eligible users to holdout.

```python
def estimate_incrementality(exposed_conversions, exposed_impressions, control_conversions, control_impressions):
    """Incremental conversion rate from a holdout test, with significance check."""
    cvr_exposed = exposed_conversions / exposed_impressions
    cvr_control = control_conversions / control_impressions
    incremental_cvr = cvr_exposed - cvr_control

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

For advertisers running multiple channels, multi-touch attribution (e.g. Shapley-value credit assignment) can distribute credit across touchpoints more fairly than single-touch heuristics.

---

## 9. Privacy Constraints

### GDPR / CCPA Impact

| Requirement | Impact on RTB |
|---|---|
| Consent for behavioral targeting | TCF 2.0 consent string checked before using any user ID |
| Right to erasure | Deletion must propagate to feature store, training data, model params |
| Data minimization | Raw bid requests can't be stored indefinitely; aggregate after 30 days |
| No sensitive-category targeting | Filter IAB sensitive categories (religion, health, politics) |
| Purpose limitation | Data collected for attribution can't be reused for lookalike modeling without fresh consent |

```python
def check_consent(tcf_string: str, purpose_id: int) -> bool:
    """Parse IAB TCF 2.0 consent string for purpose consent."""
    from iab_tcf import decode_tc_string
    tc = decode_tc_string(tcf_string)
    return tc.purpose_consents.get(purpose_id, False)
```

### Cookie Deprecation & Privacy Sandbox

Chrome's Privacy Sandbox replaces third-party cookies with:

- **Topics API:** browser classifies recent browsing into ~350 IAB topics, returning up to 3 per request.
- **Protected Audience API (FLEDGE):** remarketing lists live in the browser; the auction runs in-browser, and the DSP never sees individual user IDs — signals return aggregated via the Private Aggregation API.

```python
def extract_topics_features(bid_request: BidRequest) -> List[int]:
    """Extract Privacy Sandbox topics as contextual features."""
    topics = []
    if bid_request.user.ext and "topics" in bid_request.user.ext:
        for topic_group in bid_request.user.ext.topics:
            if topic_group.get("segtax") == 600:  # IAB topics taxonomy
                topics.extend([int(s["id"]) for s in topic_group.get("segment", [])])
    return topics
```

### Contextual Targeting Fallback

When no user ID is available (Safari, Firefox, opted-out Chrome), fall back in priority order: page content embedding → IAB categories → device type → OS/browser → geography → time-of-day → deal ID/PMP targeting.

Contextual-only models reach ~60–70% of the AUC of user-identified models for prospecting; retargeting drops to near-baseline without cookies.

---

## 10. Failure Modes

### 10.1 Bid Flooding / Traffic Quality Attacks

**Symptom:** win rate spikes, spend accelerates, conversions don't follow.
**Cause:** bot/IVT traffic — automated bid requests from low-quality inventory.
**Detection:** anomaly detection on win-rate-to-conversion-rate ratio, seller.json/ads.txt validation, data-center IP blocking, UA/device consistency checks.

```python
def score_traffic_quality(bid_request: BidRequest) -> float:
    """Return IVT probability. Score > 0.7 → no-bid."""
    signals = []
    if is_datacenter_ip(bid_request.device.ip):
        signals.append(0.9)
    parsed_ua = parse_user_agent(bid_request.device.ua)
    if parsed_ua.device_type != bid_request.device.devicetype:
        signals.append(0.8)
    uid_velocity = get_uid_velocity(bid_request.user.id)
    if uid_velocity > 100:
        signals.append(min(uid_velocity / 1000, 1.0))
    return max(signals) if signals else 0.0
```

### 10.2 Click Fraud

**Symptom:** high CTR, low post-click engagement, high bounce rate.
**Mitigation:** filter clicks with no follow-up interaction within 30s; route click-throughs via a verification redirect; integrate GIVT/SIVT classifications from IAS or DoubleVerify.

### 10.3 Latency Spikes → Timeout Cascade

**Symptom:** win rate drops to near zero.
**Cause:** a Redis latency spike pushes feature lookup past budget, so the total request exceeds 10ms and times out.
**Mitigations:** fallback to default/average features on timeout, circuit breaker on Redis error rate, p99 latency alerting before the 10ms wall, shadow traffic to canary before full rollout.

```python
def fetch_user_features_with_fallback(user_id, redis_client, timeout_ms=2):
    try:
        features = redis_client.hgetall(f"user:{user_id}:features", timeout=timeout_ms / 1000)
        return features if features else get_default_features()
    except (redis.TimeoutError, redis.ConnectionError):
        METRICS.increment("feature_lookup.timeout")
        return get_default_features()
```

### 10.4 Budget Over-Delivery

**Symptom:** campaign spends 120% of daily budget.
**Cause:** race condition — multiple bid servers decrement a non-atomic counter concurrently.
**Fix:** atomic `INCRBY`, plus a two-level check: fast local token bucket (approximate) backed by an authoritative global Redis counter.

```python
class TwoLevelBudgetGuard:
    def __init__(self, global_redis, campaign_id, daily_budget_cents):
        self.redis = global_redis
        self.campaign_id = campaign_id
        self.local_tokens = 0
        self.local_batch_size = 100
        self.daily_limit = int(daily_budget_cents * 1.02)

    def try_spend(self, amount_cents: int) -> bool:
        if self.local_tokens >= amount_cents:
            self.local_tokens -= amount_cents
            return True
        new_global = self.redis.incrby(f"budget:{self.campaign_id}:spent", self.local_batch_size)
        if new_global > self.daily_limit:
            self.redis.decrby(f"budget:{self.campaign_id}:spent", self.local_batch_size)
            return False
        self.local_tokens = self.local_batch_size - amount_cents
        return True
```

### 10.5 Pacing PID Oscillation

**Symptom:** spend alternates between zero and burst in ~15-minute cycles.
**Cause:** integral windup — the integral term accumulates a large value while budget is exhausted, then overcorrects on reset.
**Fix:** anti-windup clamp on the integral term when lambda is saturated; reset integral at budget boundaries.

### 10.6 Attribution Window Mismatch

**Symptom:** DSP dashboard shows positive ROAS but advertiser's own analytics show negative ROI.
**Cause:** DSP uses a 30-day post-click window; advertiser CRM uses 7-day.
**Fix:** align windows contractually, report both 7-day and 30-day numbers, and train pCVR on the contractually defined window.

### 10.7 Model Staleness During Traffic Spikes

**Symptom:** CTR predictions run systematically low during major events; win rate drops as competitors bid higher.
**Cause:** model trained on average traffic doesn't reflect event-driven intent and competition (CPMs spike 3–5x).
**Mitigation:** online learning (FTRL updates from live data), scheduled model hotswaps with event-aware priors, real-time bid floor as a feature.

---

## 11. Interview Q&A

**Q1: Why is calibration more important than AUC for a bidding model?**

AUC measures ranking quality; it doesn't guarantee bid prices are correct in absolute terms. If pCTR is 3x overestimated, the DSP bids 3x too much and destroys ROI even though impressions are still ranked correctly. Since `bid = pCTR × pCVR × CPA_target`, both ranking and scale must be right. ECE < 0.005 is a production gate; a 0.73→0.75 AUC bump is secondary.

**Q2: How do you handle the delayed feedback problem in pCVR training?**

Conversion labels can arrive 7–30 days after a click. Training on the last 24 hours alone systematically undercounts recent positives. The Delayed Feedback Model (Chapelle 2014) applies importance weights inversely proportional to P(no conversion observed yet), estimated from the empirical delay distribution. Alternatively, model conversion timing with a survival model, separate from conversion probability, and combine at inference.

**Q3: Explain bid shading and when it is/isn't necessary.**

In second-price auctions, truthful bidding (bid = true value) is dominant — you only pay the second-highest bid, so shading just lowers win rate without improving profit. In first-price auctions you pay exactly your bid, so bidding true value gives zero surplus; optimal strategy shades toward the expected clearing price, estimated from historical win/loss data. Most exchanges moved to first-price after 2019.

**Q4: How does the PID pacing controller handle sudden traffic drops (e.g. midnight)?**

A naive PID sees spend rate drop near zero and opens the spend tap wide, then overbids when traffic resumes. Fixes: a traffic-aware target (`budget_remaining / expected_impressions_remaining` using a time-of-day forecast) instead of a flat schedule, a feedforward term for forecasted traffic change, and clamping lambda to a reasonable range.

**Q5: How would you design the system for a world without third-party cookies?**

Three tracks: (1) contextual targeting — an NLP model over page content produces topic embeddings, achieving ~60–70% of the AUC of user-identified models; (2) first-party data activation via clean rooms and hashed email matching; (3) Privacy Sandbox — FLEDGE for retargeting, Topics API for prospecting, Private Aggregation API for measurement. Measurement shifts from user-level to aggregate/modeled attribution.

**Q6: A campaign has a 5% win rate. Feature problem or budget problem?**

Compare bid price to clearing price. If bids are well below clearing price, it's a bidding/CPA-target issue. If bids are competitive but win rate is still low, check targeting breadth, frequency caps, creative quality scores, and pacing throttle (lambda too high). Plotting empirical win rate vs. bid price shows whether raising bids is worth it — a steep curve means yes, a flat one means inventory is genuinely scarce.

**Q7: How would you prevent training-serving skew in this system?**

RTB has severe skew: only ~5% of eligible requests reach the model after filtering, only won impressions produce labels, and offline feature computation can differ from real-time serving. Mitigations: log the exact feature vector used at bid time, train only on served (won) impressions, apply inverse propensity weighting (1/P(selected)) to correct exposure-selection bias, and replay logged features for training rather than recomputing them.

---

## 12. Key Numbers to Memorize

| Metric | Value |
|---|---|
| OpenRTB hard deadline | 100ms spec; 10–50ms exchange enforcement |
| Typical DSP win rate | 3–15% |
| Average display CPM | $2–$10 (open exchange); $15–$50 (PMPs) |
| Video CPM premium | 4–8x display CPM |
| pCTR range | 0.05%–0.5% display; 0.5%–2% search |
| Conversion rate (post-click) | 1–5% e-commerce |
| Attribution delay | 7–30 days typical; up to 90 days in some verticals |
| Feature freshness target | <5 min recency signals; 1h aggregates |
| Retraining frequency | Daily (batch); every ~15 min (FTRL online) |
| Budget over-delivery tolerance | ±10% contractual SLA |

---

## Flashcards

**Why is calibration a harder requirement than AUC for a bidding model?** #flashcard
Bid price is computed as `pCTR × pCVR × CPA_target` — an absolute value, not a rank. A 3x-overestimated pCTR makes the DSP bid 3x too much regardless of how well the model ranks impressions relative to each other, directly destroying ROI.

**Why does FTRL-Proximal (not a neural net) remain the production workhorse for pCTR in the 10ms bidding path?** #flashcard
Inference is a single sparse dot product over microseconds with no GPU/network dependency, L1 regularization keeps the model sparse enough to fit in local memory, and per-feature online updates can be pushed to serving every few minutes without a full redeploy — all critical under a hard 10ms budget.

**Why does delayed feedback (Chapelle 2014) break naive pCVR training on recent data?** #flashcard
A click today may convert 7-30 days later, so recent impressions look artificially non-converting purely because their conversion window hasn't closed yet — training on them as negatives biases the model low. The fix upweights recent clicks by the inverse probability that a conversion "hasn't happened yet."

**Why is bid shading necessary in first-price auctions but not second-price?** #flashcard
In second-price (Vickrey) auctions you pay the second-highest bid, so bidding your true value is the dominant strategy. In first-price auctions you pay exactly what you bid, so bidding true value gives zero surplus — optimal bidding shades below true value toward the expected clearing price.

**Why do budget pacing systems use a PID controller instead of a fixed hourly spend target?** #flashcard
Traffic volume and clearing prices vary throughout the day; a fixed schedule either overspends early (missing high-value evening traffic) or underspends late. A PID controller continuously adjusts a bid dampener (lambda) based on the error between actual and target spend rate, correcting in real time.

**Why can PID-based pacing oscillate between zero spend and spend bursts?** #flashcard
Integral windup: when budget is exhausted, the integral error term keeps accumulating, then massively overcorrects once budget resets — causing wide spend swings. Fixed with an anti-windup clamp on the integral term when lambda is saturated.

**Why does a Redis latency spike cause win rate to collapse to near zero rather than just slowing responses?** #flashcard
The bid path has a hard ~10ms deadline enforced by exchanges; if feature lookup latency pushes total request time past that wall, the exchange treats the DSP as a no-bid rather than waiting — a small latency spike converts into a complete loss of bidding volume until a fallback to default features kicks in.

**Why does the shift to first-price auctions and cookie deprecation both push toward the same architectural change?** #flashcard
Both remove a source of certainty the DSP used to rely on (guaranteed pay-second-price, guaranteed persistent user ID) and require estimating a distribution instead of a known value — clearing-price distributions for bid shading, and contextual/aggregate signals (Topics API, FLEDGE) replacing deterministic user IDs.

**Why is inverse propensity weighting needed when training pCTR/pCVR on logged bidding data?** #flashcard
Only ~5% of bid requests survive filtering to reach the model, and only won impressions produce labels — training naively on this subset reproduces the selection bias of whatever policy logged the data. Weighting by 1/P(selected) corrects the exposure bias so the model reflects the true population, not just what was previously bid on and won.

**Why is a two-level (local + global) budget guard needed instead of one atomic Redis counter?** #flashcard
Checking a single global counter on every bid adds network latency that competes with the 10ms budget; a local token bucket handles most checks approximately and fast, batching updates to the authoritative global Redis counter periodically — trading small over-delivery risk (bounded by batch size) for latency headroom.

---

## 13. References & Further Reading

- **OpenRTB 2.6 Spec** — IAB Tech Lab
- **Chapelle, O. (2014)** — "Modeling Delayed Feedback in Display Advertising" (KDD)
- **Zhang, W. et al. (2014)** — "Optimal Real-Time Bidding for Display Advertising" (KDD)
- **McMahan, H.B. et al. (2013)** — "Ad Click Prediction: A View from the Trenches" (KDD)
- **Perlich, C. et al. (2012)** — "Bid Optimizing and Inventory Scoring in Targeted Online Advertising" (KDD)
- **Google Privacy Sandbox** — Protected Audience API spec (FLEDGE successor)
- **IAB Tech Lab** — Seller.json, ads.txt, SupplyChain Object
