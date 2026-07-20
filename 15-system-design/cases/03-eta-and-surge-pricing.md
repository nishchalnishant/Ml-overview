---
module: System Design
topic: System Design
subtopic: Eta And Surge Pricing
status: unread
tags: [productionml, ml, system-design-eta-and-surge-pr]
---
# Ride-Hailing ETA Prediction & Dynamic Surge Pricing

End-to-end ML system for real-time ETA estimation and dynamic pricing in ride-hailing. Canonical system design question at Uber, Lyft, DoorDash, and any marketplace with real-time supply/demand balancing.

**Scale:** 5M+ trips/day, <500ms ETA response, city-level real-time supply/demand.

---

## 1. Problem Framing

Two interlinked ML problems that share features but have different objectives, latency, and failure modes.

### Clarifying Questions

**ETA:**
- What is the ETA used for — rider display, matching, or dispatch?
- Pickup ETA (driver → rider), trip ETA (rider → destination), or both?
- Acceptable error — ±2 min? ±20%? Does it scale with trip length?
- Real-time updates as the driver moves, or a single pre-trip estimate?
- Show a range ("8–12 min") or a point estimate?

**Surge:**
- Geographic granularity — city, neighborhood, or hex cell?
- Regulatory price caps (vary by city/country)?
- Does surge affect only rider price, or also driver earnings?
- Show the multiplier (1.8x) or just the final price?
- How long until a new city has enough data to price well?

### Metric Trade-offs

**ETA Metrics:**

| Metric | Formula | Target |
|---|---|---|
| MAE | mean(\|predicted − actual\|) | <90s for <30 min trips |
| MAPE | mean(\|predicted − actual\| / actual) | <10% |
| p90 error | 90th pct of absolute errors | <3 min |
| Late arrival rate | % predictions < actual | <15% (asymmetric cost) |

**Asymmetric loss matters:** underestimating ETA ("3 min" when it's 7) is worse than overestimating — riders cancel or rate poorly. Train with asymmetric loss or calibrate post-hoc.

$$\mathcal{L}_{asym}(\hat{y}, y) = \begin{cases} \alpha \cdot (\hat{y} - y)^2 & \hat{y} > y \\ \beta \cdot (\hat{y} - y)^2 & \hat{y} < y \end{cases}, \quad \beta > \alpha \ (\text{e.g., } \beta = 2\alpha)$$

**Surge Metrics:**

| Metric | Description | Direction |
|---|---|---|
| Trip completion rate | % of requests completed | Maximize |
| Driver utilization | % of time drivers have a passenger | Maximize |
| Marketplace balance | Supply/demand ratio per cell | Target ≈ 1.0 |
| Rider wait time | Median request-to-pickup time | Minimize |
| Revenue per driver-hour | Platform revenue efficiency | Maximize |
| Surge cancellation lift | % cancellation increase during surge | Minimize |

**Core trade-off:** higher surge increases driver supply but reduces rider demand. The optimal price clears the market — minimizing both unfulfilled requests and idle drivers.

---

## 2. Scale and Requirements

| Dimension | Target |
|---|---|
| Trip volume | 5M+ trips/day (~58 TPS avg, 200 TPS peak) |
| ETA request volume | 50M+/day |
| ETA response latency | p99 <500ms |
| Surge recalculation frequency | Every 30–60s per cell |
| Driver GPS update rate | Every 4s per driver |
| Active drivers (peak) | 500K+ globally |
| Cities | 700+, with city-specific models |
| Map update lag tolerance | <24h for major road changes |

---

## 3. System Architecture

```
Rider App                          Driver App
    │                                  │
    │ Request (origin, dest)            │ GPS pings (every 4s)
    ▼                                  ▼
 ┌──────────────────────────────────────────────────┐
 │                  API Gateway                      │
 └──────┬───────────────────────────────┬───────────┘
        │                               │
        ▼                               ▼
 ┌─────────────┐               ┌─────────────────┐
 │  ETA Service│               │  Supply Tracker  │
 │  (<500ms)   │               │  (driver states) │
 └──────┬──────┘               └────────┬────────┘
        │                               │
        │  ┌────────────────────────────┘
        │  │
        ▼  ▼
 ┌─────────────────────┐      ┌─────────────────────┐
 │   Feature Assembly  │◄─────│   Real-Time Feature  │
 │   (Road + Context)  │      │   Store (Redis)       │
 └──────────┬──────────┘      └─────────────────────┘
            │
            ▼
 ┌──────────────────────┐
 │   ETA Model Server   │──── City-partitioned TensorFlow Serving
 │   (per-city models)  │     Fallback: rule-based grid lookup
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │   Matching Engine    │──── Combines ETA + driver score → dispatch
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │   Surge Calculator   │◄─── H3 hex cells (supply/demand counts)
 │   (every 30–60s)     │◄─── Event detector (concerts, weather)
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │   Price Engine       │──── Multiplier + guardrails → rider quote
 └──────────────────────┘

 ════════════════════ Offline / Training Path ════════════════════

 Trip Logs (Kafka) → GPS Trace Cleaner → Map Matcher
       │                                       │
       │                                       ▼
       │                            ┌──────────────────┐
       │                            │  Label Generator  │
       │                            │ (actual duration) │
       └────────────────────────────┤                   │
                                    └────────┬─────────┘
                                             │
                            ┌────────────────┴──────────────────┐
                            │                                   │
                     ┌──────▼──────┐                  ┌────────▼───────┐
                     │  ETA Model  │                  │ Pricing Model  │
                     │  Training   │                  │ Training       │
                     │  (per city) │                  │ (elasticity)   │
                     └─────────────┘                  └────────────────┘
```

---

## 4. ETA Prediction

### Feature Engineering

**Route-Level (from road graph)**

| Feature | Description |
|---|---|
| free_flow_duration | Sum of edge weights at free-flow speed |
| historical_avg_duration | Historical mean for this OD pair by hour |
| historical_p75_duration | 75th pct duration (uncertainty buffer) |
| route_length_km | Total route distance |
| num_turns | Number of turns on route |
| num_signalized_intersections | Traffic light count |
| highway_fraction | % of route on highway |

**Real-Time Contextual**

| Feature | Description | Update Freq |
|---|---|---|
| current_speed_ratio | Observed / free-flow speed per segment | 1 min |
| segment_congestion_score | 0–1 congestion per segment | 1 min |
| weather_condition | Rain/snow/clear + intensity | 5 min |
| time_of_day_embedding | Cyclical sin/cos of hour + day | Per request |
| is_rush_hour | Boolean flag | Per request |
| active_events_nearby | Event count in 2km radius | 5 min |

**Driver-Level (pickup ETA):** driver_current_speed, driver_distance_to_pickup (road distance, not straight-line), driver_heading_alignment, driver_recent_completion_rate.

**Temporal encoding** (avoids boundary artifacts at hour 23→0):
```python
def cyclical_time_encoding(hour: int, day_of_week: int):
    hour_sin, hour_cos = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    dow_sin, dow_cos = np.sin(2*np.pi*day_of_week/7), np.cos(2*np.pi*day_of_week/7)
    return hour_sin, hour_cos, dow_sin, dow_cos
```

### Model Choices

**Gradient Boosted Trees (LightGBM):** strong baseline, ~5ms inference, handles missing features (weather gaps, GPS gaps), interpretable. Per-city model trained on last 90 days, updated weekly.

**Uber's DeepETA (production):** transformer encoder over road segment sequences, learns segment-level travel time embeddings. Multi-task: ETA + route deviation probability. ~50ms inference, better MAPE in complex cities.

**GNN on road network:** models roads as a graph (node = intersection, edge = segment) so neighborhood context improves segment estimates. Slower to train/serve; useful for cities with complex topology.

**Recommendation:** LightGBM as primary, DeepETA for high-traffic cities, GNN for longer-horizon research.

```python
import lightgbm as lgb

eta_model = lgb.LGBMRegressor(
    n_estimators=800, max_depth=7, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
    objective='huber', alpha=0.9,   # robust to outlier trips (accidents, detours)
    n_jobs=-1
)
eta_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
```

### Uncertainty Estimation

Point estimates aren't enough — riders and the matching engine need confidence intervals.

- **Quantile regression:** train separate models at p10/p50/p90, show rider "8–12 min" range.
- **NGBoost:** outputs a full distribution (mean + variance) from one model.
- **Conformal prediction:** calibrates intervals on held-out data without distributional assumptions — works as a post-hoc wrapper on any base model.

```python
quantile_models = {}
for q in [0.1, 0.5, 0.9]:
    m = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=800)
    m.fit(X_train, y_train)
    quantile_models[q] = m

def predict_eta_with_range(features):
    return {
        'estimate': quantile_models[0.5].predict(features)[0],
        'low': quantile_models[0.1].predict(features)[0],
        'high': quantile_models[0.9].predict(features)[0],
    }
```

### Multi-Task Learning

Train one model on two targets: ETA (regression) and route deviation probability (will the driver take a different route?). Shared backbone captures common route-quality features; deviation probability widens the ETA confidence interval dynamically.

---

## 5. Road Network Modeling

### Graph Representation

Road network as a directed graph $G=(V,E)$: nodes = intersections (lat/lng, signal type), edges = segments (speed limit, road type, lane count, historical travel times by hour/day).

### H3 Hexagonal Grid

Uber's H3 discretizes Earth into hierarchical hex cells.

**Why hexagons over squares:** all neighbors are equidistant (square diagonals are √2 farther, causing anisotropic smoothing). Clean hierarchical resolution (res 7 ≈ 1.4 km², res 9 ≈ 0.1 km²).

```python
import h3

def get_hex_features(lat, lng, resolution=9):
    cell_id = h3.geo_to_h3(lat, lng, resolution)
    return cell_id, redis_client.hgetall(f"h3:{cell_id}:features")

def get_route_hex_sequence(waypoints, resolution=9):
    """Encode route as a deduplicated sequence of H3 cells (DeepETA-style input)."""
    cells = [h3.geo_to_h3(lat, lng, resolution) for lat, lng in waypoints]
    return [c for i, c in enumerate(cells) if i == 0 or c != cells[i-1]]
```

### GNN for Route Encoding

```python
from torch_geometric.nn import GATConv

class RoadNetworkGNN(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim=128):
        super().__init__()
        self.conv1 = GATConv(node_features, hidden_dim, heads=4, edge_dim=edge_features)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, edge_dim=edge_features)
        self.segment_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr).relu()
        h = self.conv2(h, edge_index, edge_attr).relu()
        return self.segment_head(h).squeeze(-1)   # per-segment seconds
```

Segment-level speed features update every minute from GPS traces; GNN weights retrain nightly.

---

## 6. Surge Pricing Model

### Supply/Demand Ratio

At H3 cell $c$, time $t$:

$$\text{SDR}(c, t) = \frac{\text{available\_drivers}(c, t)}{\text{rider\_requests}(c, t) + \epsilon}$$

SDR < 1 → excess demand, surge needed. SDR > 1.5 → excess supply, reduce price. SDR ≈ 1 → balanced.

```python
def compute_sdr_per_cell(redis_client, h3_cells, window_seconds=60):
    results = {}
    for cell in h3_cells:
        drivers = int(redis_client.get(f"h3:{cell}:active_drivers") or 0)
        requests = int(redis_client.get(f"h3:{cell}:open_requests") or 0)
        results[cell] = drivers / (requests + 1e-3)
    return results
```

### Price Elasticity

$$\epsilon_d = \frac{\Delta Q / Q}{\Delta P / P}$$

Ride-hailing demand elasticity is typically -0.5 to -1.5. Estimate from historical A/B tests and natural price variation via log-log regression:

```python
elasticity_model = smf.ols(
    'np.log(requests) ~ np.log(price) + hour + day_of_week + weather + event_nearby',
    data=historical_pricing_data
).fit()
price_elasticity = elasticity_model.params['np.log(price)']
```

Driver supply elasticity (how many drivers come online at higher surge) is roughly +0.3 to +0.8.

### Surge Multiplier Calculation

**Rule-based baseline:** SDR-to-multiplier mapping with smoothing — no surge above SDR 1.5, linear ramp 1.0x→1.5x down to SDR 0.75, steeper ramp below that, capped at max surge.

**ML-based pricing (production):** features include current + neighboring cell SDR (spillover), SDR trend, historical elasticity for this cell/time, event indicator, weather, driver earnings vs. target. Objective: maximize `completion_rate * expected_revenue` subject to `wait_time < threshold` and `surge <= regulatory_cap`.

### Reinforcement Learning for Dynamic Pricing

- **State:** SDR, time, weather, events, recent price history
- **Action:** surge multiplier from a discrete set {1.0, 1.2, 1.5, 1.8, 2.0, 2.5}
- **Reward:** `completed_trips * fare - λ * unfulfilled_requests - μ * cancellations`
- **Environment:** simulator built on historical demand/supply response curves
- **Algorithm:** PPO, pre-trained in simulation, then fine-tuned online with constrained exploration

**Guardrails (non-negotiable):** hard cap on max multiplier, cap on change per step (no 1.0→2.5 jump), minimum hold duration before dropping surge, per-city regulatory caps, emergency crisis-mode override.

### Spatial Smoothing

Raw SDR is noisy. Smooth across H3 k-ring neighbors to avoid sharp price boundaries at cell edges (e.g., blend 60% own cell, 40% neighbor average).

---

## 7. Real-Time Features

### Driver GPS Processing

Drivers ping every 4s. Raw GPS is noisy and must be map-matched (snap to road graph via HMM/Viterbi), used to compute instantaneous speed, and classified into a state (moving, stopped, idle, on-trip). Driver state is cached in Redis with a short TTL; segment speed observations feed a rolling window used for ETA features.

### Geospatial Indexing (H3 + S2)

**H3** (hex, Uber): best for demand/supply aggregation and surge. **S2** (quadtree, Google): best for range queries and nearest-driver lookup. Production systems typically use both — H3 for pricing cells, S2 or Redis GEO for nearest-driver search.

### Event Detection

Pull from ticketing/venue/sports APIs to detect upcoming events near a city, tracking active event count, expected attendance, and time until the nearest event ends — used to anticipate demand spikes before they show up in SDR.

---

## 8. Training Data Pipeline

### GPS Trace Processing

```
Raw GPS pings (Kafka) 
    → Noise filter (Kalman filter / outlier removal)
    → Map matching (HMM Viterbi on OSRM road graph)
    → Trace compression (Douglas-Peucker)
    → Trip segmentation
    → Feature extraction (per-segment speeds, times)
    → Label generation (actual trip duration)
    → Feature store write (Hive/Parquet)
```

**Map matching:** the true path is the most likely sequence of road segments given noisy GPS observations:

$$P(\text{path} \mid \text{GPS}) \propto P(\text{GPS} \mid \text{path}) \cdot P(\text{path})$$

Emission probability is Gaussian around each segment; transition probability enforces route plausibility (consistent direction, no teleporting). Solved with Viterbi. Production systems use Valhalla or OSRM rather than a custom implementation.

### Label Generation

`actual_duration = dropoff_ts - pickup_ts`, with filtering: drop cancelled trips, GPS gaps > 60s, durations outside (0, 2h], and outliers beyond 3 std within a distance bucket.

### Training Cadence

| Model | Frequency | Data window |
|---|---|---|
| ETA (LightGBM, per city) | Weekly | 90 days |
| ETA (DeepETA, shared backbone) | Monthly | 1 year |
| GNN segment speeds | Nightly | 30 days |
| Surge (rule-based calibration) | Daily | 7 days |
| Price elasticity model | Weekly | 90 days |
| RL pricing policy | Continuous (off-policy) | Rolling 30 days |

---

## 9. Model Serving

### City-Partitioned Serving

Request routes to a city-specific model cluster (e.g., TF Serving, 2 replicas). Feature assembly (<50ms) pulls the route from OSRM, real-time speeds from Redis, cached weather, and time encoding. Model inference (<100ms) returns ETA + confidence interval.

### Precomputed ETA Grids

For common OD pairs (airport ↔ downtown, stadium ↔ transit hub), precompute ETA every 5 minutes and cache in Redis. Covers ~30% of requests at <5ms latency.

### Fallback Hierarchy

1. ML model (city-specific LightGBM or DeepETA)
2. Precomputed ETA grid (nearest matching cell pair)
3. Historical average for this OD by hour/day
4. Free-flow estimate (distance / avg city speed)

```python
def get_eta_with_fallback(origin, destination, city, timeout_ms=400):
    try:
        with timeout(timeout_ms):
            return ml_eta_service.predict(origin, destination, city)
    except (TimeoutError, ModelUnavailableError):
        pass

    cached = redis_client.get(build_cache_key(origin, destination, city))
    if cached:
        return float(cached), 'cache'

    historical = historical_eta_db.lookup(origin, destination, current_hour())
    if historical:
        return historical, 'historical'

    return free_flow_estimate(origin, destination), 'fallback'
```

### Latency Breakdown (500ms budget)

| Component | Budget |
|---|---|
| Network + API gateway | 20ms |
| Route computation (OSRM) | 80ms |
| Feature assembly (Redis) | 50ms |
| Model inference | 100ms |
| Surge calculation | 50ms |
| Response serialization | 10ms |
| **Total** | **310ms** (headroom for p99) |

---

## 10. Failure Modes

**Map changes invalidate models.** New/closed roads or speed limit changes make old predictions systematically wrong. Detect via sudden jumps in mean residual error per segment. Mitigate by subscribing to map change feeds, invalidating affected segment caches, triggering retrain within 24h, and inflating uncertainty for stale segments.

**Weather edge cases.** Extreme weather creates conditions rarely seen in training; predictions collapse. Mitigate with a hard override (multiply base ETA by a weather scalar above a precipitation threshold), a separate weather-conditioned model, and conservative fallback ETAs during declared weather events.

**Event-driven demand spikes.** A stadium/concert lets out and thousands request rides at once, outside normal patterns. Detect via request rate vs. 7-day moving average (>3σ triggers event mode). Mitigate by pre-announcing surge before the event ends, pre-positioning drivers, pre-warming the ETA cache for venue destinations, and using the event calendar to pre-scale infrastructure.

**Drivers gaming surge zones.** Drivers go offline to create artificial scarcity, triggering surge, then come back online. Detect via correlated offline/online bursts in a small area. Mitigate by requiring sustained shortage (>5 min) before surge triggers, and excluding recently-offline drivers from the surge bonus in that cell.

**Cold start for new cities.** No trip history for ETA, no elasticity data for pricing. Mitigate with transfer learning from a similar city, a conservative fixed price schedule for the first 4–6 weeks, synthetic trip simulation from map data, and gradual rollout of city-specific models once enough trips accumulate (~10K).

**Model staleness during rapid traffic changes.** An accident causes gridlock the model doesn't expect. Mitigate by letting a live traffic feed (Waze/HERE) override segment speeds, and applying a dynamic correction factor when observed driver speed falls well below the model's assumed speed.

---

## 11. Interview Angles

### Q1: Why does ETA accuracy matter beyond rider satisfaction? [Easy]

ETA feeds the matching engine's dispatch decisions — poor ETA means sub-optimal driver assignment even with enough supply. It also drives upfront fare estimation; systematic under-prediction means drivers earn less than the quoted fare, hurting driver retention.

---

**Cross-questions to expect:**
- *"Would you rather be 2 minutes early or 2 minutes late?"* → Early, and this asymmetry should be in the loss. Late costs cancellations and trust; early costs a little idle time. A symmetric MAE trains a model that treats those as equal, so quantile loss at something above the median is the standard fix.
- *"Does ETA feed its own training data?"* → Yes, and that's the subtle risk. A bad ETA changes dispatch, which changes which trips happen, which becomes tomorrow's training set. Trips that were never dispatched because the ETA looked bad are absent from your data entirely.

**Trap:** Optimizing mean error. Riders experience the tail — the 95th-percentile miss drives cancellations, and a model can improve MAE while making the tail worse.

---

### Q2: How would you handle ETA for airport pickups, which behave very differently from street hailing? [Medium]

Treat it as a separate model with airport-specific features: flight landing time (aviation APIs), baggage claim wait by terminal, staging-lot-to-curb time. Use a two-stage prediction — staging lot → curb transition, then curb ETA once dispatched. Per-airport data density supports dedicated models.

---

**Cross-questions to expect:**
- *"Separate model or one model with a flag?"* → Start with a flag plus airport-specific features; split only when you can show the residuals differ structurally and the segment has enough volume to support its own training. Every extra model is a permanent maintenance and monitoring cost.
- *"What breaks when the flight API fails?"* → You need a degradation path to a scheduled-time prior rather than a null feature. Airports are exactly where a silently-zeroed feature produces a confidently wrong answer, and it's a segment with high visibility.

**Trap:** Presenting airports as one case. Each has its own geometry, staging rules, and terminal layout — a per-airport calibration layer usually beats one clever global airport model.

---

### Q3: Why H3 hex cells instead of square grids for surge? [Easy]

Square grids have unequal neighbor distances (diagonal is √2× farther), causing anisotropic smoothing artifacts at grid corners. Hex cells are equidistant to all six neighbors, giving isotropic smoothing, plus clean hierarchical aggregation (each resolution-7 cell = 7 resolution-8 cells).

---

**Cross-questions to expect:**
- *"Is the hierarchy exact?"* → No, and it's a real gotcha. Hexagons don't tile hierarchically the way squares do, so H3's parent-child relation is approximate — a child cell isn't perfectly contained in its parent. Fine for aggregation, wrong if you need exact partitioning for accounting.
- *"How do you pick the resolution?"* → By data density, not geometry. Cells need enough demand and supply events per interval to make SDR a stable statistic; too fine and you're modelling noise, too coarse and you smear distinct neighbourhoods together. Dense city centres and suburbs often warrant different resolutions.

**Trap:** Over-arguing geometry. The isotropy point is real but minor next to the practical wins — a stable global cell ID scheme, cheap neighbour lookups, and clean aggregation. Interviewers notice when a candidate treats tessellation elegance as the main benefit.

---

### Q4: How do you prevent surge from oscillating (surge up → drivers join → SDR flips → surge drops → drivers leave → repeat)? [Hard]

Temporal hysteresis (require SDR to stay past a threshold for N minutes before changing surge), supply-response forecasting instead of pure reactive thresholds, capped multiplier change per interval, and spatial smoothing so neighboring cells dampen local volatility.

---

**Cross-questions to expect:**
- *"Name the control-theory failure."* → Feedback with delay. Drivers respond minutes after the price changes, so a purely reactive controller is always correcting a state that has already moved. Hysteresis and rate limiting are damping; forecasting supply response is the actual fix because it puts the delay in the model.
- *"What does hysteresis cost?"* → Responsiveness during genuine demand shocks — a concert letting out needs a fast response, and the same damping that prevents oscillation delays it. Event calendars exist partly to let you relax the damping when a spike is expected.
- *"How would you detect oscillation in production?"* → Autocorrelation of the multiplier per cell, or counting sign changes per hour. A cell flipping several times an hour is oscillating regardless of whether riders complain.

**Trap:** Treating this as a smoothing problem. Spatial smoothing hides oscillation from dashboards without removing it — drivers still experience the flipping, and the trust cost is what actually matters.

---

### Q5: How would you detect ETA model degradation in production? [Medium]

Monitor rolling ETA residual (`predicted - actual`) stratified by city/trip type/time, alerting on mean error or MAPE thresholds. Secondary signals: cancellation spikes right after ETA is shown, driver rating dips correlated with underestimates, and city-specific drift vs. peers. Use shadow mode to compare new vs. old model residuals before cutover.

---

**Cross-questions to expect:**
- *"Aggregate residual is flat but something's wrong — what do you check?"* → Stratify. Offsetting errors across segments cancel in the mean: a model over-predicting suburbs and under-predicting downtown looks perfect globally. City, time-of-day, trip length, and weather are the standard cuts.
- *"How do you separate model degradation from the world changing?"* → You often can't from residuals alone, which is why shadow mode matters. If a retrained challenger shows the same residuals, the world moved; if it recovers, your model went stale. Road closures and construction are the common genuine causes.
- *"Any signal available before the trip completes?"* → Yes — prediction distribution shift is immediate, while residuals wait for trip completion. On a long trip that's a 30-minute-plus lag on your only ground truth.

**Trap:** Alerting on MAPE alone. It's unstable for short trips: a 2-minute miss on a 4-minute trip is 50% error and largely unavoidable, so a shift in trip-length mix fires the alert without any model change.

---

### Q6: How do you handle a 10x traffic spike (New Year's Eve)? [Medium]

Pre-provisioned autoscaling for model serving and feature store replicas, cache pre-warming for top OD pairs hours ahead, graceful degradation to cached/historical ETAs under load instead of timing out, pre-configured surge parameters for known events, and load shedding on low-priority (non-booking) ETA calls.

---

**Cross-questions to expect:**
- *"Autoscaling is reactive — is that enough?"* → No. Scale-up takes minutes and the spike is faster, so you pre-provision against a forecast for known events. Autoscaling handles the unforeseen; the calendar handles New Year's Eve.
- *"What does the model do that's wrong on that night?"* → Everything it learned about typical traffic is off-distribution — travel times, demand patterns, and cancellation behaviour all shift. Historical same-event data is worth more than recent data, and holding out last year's NYE is a reasonable validation choice.
- *"What do you shed first?"* → Speculative ETA calls — map browsing, pre-request estimates — before anything on the booking path. Deciding this under load is too late; the priority tiers have to exist beforehand.

**Trap:** Answering purely with infrastructure. The failure on peak nights is usually accuracy, not availability: the system stays up and quotes confidently wrong ETAs, which is worse than a visible degradation banner.

---

### Q7: Surge pricing raises revenue but completion rate drops 2%. Roll back or not? [Hard]

Not necessarily bad — could reflect shedding low-value demand that was congesting the platform. Check driver utilization (did it improve?), wait times for completed trips, and driver earnings (retention signal). Also consider the counterfactual: in a supply-constrained market, the no-surge baseline's "completion rate" is inflated by requests that never get a driver anyway. Use a holdout-cell A/B test to measure the true causal effect, and roll back only if net marketplace value (fulfilled trips × fare) actually decreased.

---

**Cross-questions to expect:**
- *"Who is being priced out, and does that matter beyond revenue?"* → It matters a great deal. If the shed demand concentrates in lower-income areas or specific times, you have an equity and regulatory problem that the aggregate marketplace-value number will never surface. Segment the shed demand before declaring the trade-off acceptable.
- *"How do you A/B test a marketplace at all?"* → Not by randomizing riders — treatment and control compete for the same drivers, so interference biases the result. Switchback tests over time within a region, or region-level randomization, are the standard answers, and both cost statistical power.
- *"Which single number decides it?"* → There isn't one. Fulfilled trips, driver earnings per hour, rider wait time, and the equity cut all matter, and they trade off. Naming the decision criteria before running the test is the discipline being tested.

**Trap:** Defending surge purely as efficient rationing. That's the economics answer and it's incomplete — it ignores distributional effects and the reputational cost, both of which have forced real policy changes at these companies.

---

## Flashcards

**Why is asymmetric loss used for ETA prediction instead of standard MSE?** #flashcard
Underestimating ETA ("3 min" when it's actually 7) causes rider cancellations and poor ratings, while overestimating just makes the rider pleasantly surprised — the cost of the two error directions isn't symmetric, so penalizing under-prediction more heavily (β > α) better matches the real business cost.

**Why does surge pricing need both a supply/demand ratio AND a price elasticity estimate, not just one?** #flashcard
SDR tells you the current imbalance (too many riders vs. drivers), but elasticity tells you how much price change is needed to close that gap — without elasticity you don't know if a 1.2x or 2.5x multiplier is required to bring supply and demand back into balance.

**Why do production surge systems use H3 hex cells instead of square grids?** #flashcard
Square grid neighbors are unequally distant (diagonal neighbors are √2× farther than adjacent ones), causing anisotropic smoothing artifacts at cell corners; hexagons are equidistant to all six neighbors, giving isotropic smoothing and clean hierarchical aggregation.

**Why can't surge multipliers be set by pure reactive thresholds on SDR?** #flashcard
Reactive-only pricing oscillates: surge triggers → drivers rush in → SDR flips to oversupply → surge drops → drivers leave → shortage returns. Fixing this requires temporal hysteresis (sustained imbalance before changing price), capped change per interval, and spatial smoothing across neighboring cells.

**Why does map matching (snapping noisy GPS to road segments) require an HMM/Viterbi approach instead of nearest-edge lookup?** #flashcard
Raw GPS has enough noise that naive nearest-edge snapping picks the wrong parallel road or jumps illogically between pings; Viterbi finds the most likely sequence of road segments by combining emission probability (GPS distance to a segment) with transition probability (route plausibility — consistent direction, no teleporting).

**Why is a fallback hierarchy (ML model → cache → historical average → free-flow estimate) necessary for ETA serving?** #flashcard
The ETA response has a hard <500ms budget; if the ML model times out or a city-specific cluster is unavailable, the system must degrade gracefully to cheaper, always-available estimates rather than fail the request outright.

**Why is a two-stage model useful for airport pickup ETAs specifically?** #flashcard
Airport pickups have a distinct structure — staging-lot wait time depends on flight landing/baggage claim, and curb ETA only starts once a driver is dispatched from the lot — so a single generic ETA model misses the staging-to-curb transition that dominates total wait time.

**Why does a sudden citywide map change (new road, closed road) silently degrade an ETA model instead of causing an obvious error?** #flashcard
The model keeps producing confident-looking numbers based on stale road-graph assumptions; there's no crash, just a systematic bias in mean residual error per affected segment — requiring active monitoring of residuals (not just uptime) to catch it.

**Why do ride-hailing platforms need a conservative pricing schedule for new cities instead of letting the ML pricing model run immediately?** #flashcard
A new city has no trip history for ETA and no historical price/demand data for elasticity estimation, so an ML pricing model would be fitting noise; a fixed schedule plus transfer learning from a similar city avoids erratic early pricing until ~10K trips accumulate.

**Why might a 2% drop in completion rate after introducing surge NOT be a signal to roll back the pricing model?** #flashcard
In a supply-constrained market, some of that "completion rate" was inflated by requests that would never get a driver anyway; surge can shed low-value demand while improving driver utilization and earnings. The real test is a holdout A/B measuring net marketplace value (fulfilled trips × fare), not raw completion rate alone.

---

## References

- Uber Engineering: [DeepETA: How Uber Predicts Arrival Times Using Deep Learning](https://www.uber.com/blog/deepeta-how-uber-predicts-arrival-times/) (2022)
- Uber Engineering: [H3: Uber's Hexagonal Hierarchical Spatial Index](https://www.uber.com/blog/h3/) (2018)
- Uber Research: [Forecasting at Uber: An Introduction](https://www.uber.com/blog/forecasting-introduction/) (2018)
- Lyft Engineering: [Lyft's Marketplace Pricing](https://eng.lyft.com/lyft-marketplace-pricing-b54c2c8e3a1b) (2018)
- Uber Research: [Surge Pricing Solves the Wild Goose Chase](https://faculty.chicagobooth.edu/christopher.knittel/research/papers/uber_surge_goose_chase.pdf) — Castillo et al., AER 2017
- Valhalla map matching: [https://github.com/valhalla/valhalla](https://github.com/valhalla/valhalla)
- OSRM routing engine: [http://project-osrm.org/](http://project-osrm.org/)
