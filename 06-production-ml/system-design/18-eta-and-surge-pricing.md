---
module: Production Ml
topic: System Design
subtopic: Eta And Surge Pricing
status: unread
tags: [productionml, ml, system-design-eta-and-surge-pr]
---
# Ride-Hailing ETA Prediction & Dynamic Surge Pricing

End-to-end ML system for real-time ETA estimation and dynamic pricing in ride-hailing platforms. Canonical system design question at Uber, Lyft, DoorDash, and any marketplace with real-time supply/demand balancing.

**Scale:** 5M+ trips/day, <500ms ETA response, city-level real-time supply/demand, global deployment with city-specific models.

---

## 1. Problem Framing

Two interlinked ML problems that share features but have different objectives, latency requirements, and failure modes.

### Clarifying Questions

**ETA Prediction:**
- **What is the ETA used for?** Pre-booking estimate shown to rider, matching optimization, or driver dispatch?
- **Which leg of the trip?** Pickup ETA (driver → rider), trip ETA (rider → destination), or both?
- **Acceptable error?** ±2 minutes? ±20%? Does it change by trip length?
- **Real-time vs batch?** Should ETA update as the driver moves, or is a single pre-trip estimate sufficient?
- **Uncertainty communication?** Show a range ("8–12 min") or a point estimate?

**Surge Pricing:**
- **Geographic granularity?** City-level, neighborhood-level, or hexagonal cell-level?
- **Price cap policy?** Regulatory limits on surge multipliers vary by city/country
- **Driver-side incentives?** Does surge only affect rider price, or also driver earnings?
- **Surge transparency?** Do we show surge multiplier (1.8x) or just final price?
- **New city cold start?** How long until the pricing model has enough data?

### Metric Trade-offs

**ETA Metrics:**

| Metric | Formula | Target |
|---|---|---|
| MAE | mean(|predicted − actual|) | <90 seconds for <30 min trips |
| MAPE | mean(|predicted − actual| / actual) | <10% |
| p90 error | 90th pct of absolute errors | <3 minutes |
| Late arrival rate | % predictions < actual | <15% (asymmetric cost) |

**Asymmetric loss matters:** Underestimating ETA (telling rider "3 min" when it's 7 min) is worse than overestimating. Riders cancel, rate poorly, or defect. Train with asymmetric loss or add calibration post-hoc.

$$\mathcal{L}_{asym}(\hat{y}, y) = \begin{cases} \alpha \cdot (\hat{y} - y)^2 & \text{if } \hat{y} > y \text{ (overestimate)} \\ \beta \cdot (\hat{y} - y)^2 & \text{if } \hat{y} < y \text{ (underestimate, higher penalty)} \end{cases}$$

where $\beta > \alpha$ (e.g., $\beta = 2\alpha$).

**Surge Pricing Metrics:**

| Metric | Description | Direction |
|---|---|---|
| Trip completion rate | % of requests that result in completed trips | Maximize |
| Driver utilization | % of time drivers have a passenger | Maximize |
| Marketplace balance | Supply/demand ratio per cell | Target ≈ 1.0 |
| Rider wait time | Median time from request to pickup | Minimize |
| Revenue per available driver-hour | Platform revenue efficiency | Maximize |
| Surge cancellation lift | % increase in cancellations during surge | Minimize |

**The core trade-off:** Higher surge multiplier increases driver supply but reduces rider demand. The optimal price clears the market — minimizes both unfulfilled requests (no driver) and idle drivers.

---

## 2. Scale and Requirements

| Dimension | Target |
|---|---|
| Trip volume | 5M+ trips/day (~58 TPS average, 200 TPS peak) |
| ETA request volume | 50M+/day (many requests before booking) |
| ETA response latency | p99 <500ms |
| Surge recalculation frequency | Every 30–60 seconds per cell |
| Driver GPS update rate | Every 4 seconds per driver |
| Active drivers (peak) | 500K+ globally |
| Cities | 700+ cities with city-specific models |
| Map update lag tolerance | <24 hours for major road changes |

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

**Route-Level Features (from road graph)**

| Feature | Description | Source |
|---|---|---|
| free_flow_duration | Sum of edge weights at free-flow speed | Map graph |
| historical_avg_duration | Historical mean for this OD pair by hour | Trip DB |
| historical_p75_duration | 75th pct duration (buffers for uncertainty) | Trip DB |
| route_length_km | Total route distance | Map graph |
| num_turns | Number of turns on route | Routing engine |
| num_signalized_intersections | Traffic light count | Map annotation |
| highway_fraction | % of route on highway | Map annotation |

**Real-Time Contextual Features**

| Feature | Description | Update Frequency |
|---|---|---|
| current_speed_ratio | Observed speed / free-flow speed per segment | 1 min |
| segment_congestion_score | 0–1 congestion level per road segment | 1 min |
| weather_condition | Rain/snow/clear + intensity | 5 min |
| visibility_km | Visibility (affects driving speed) | 5 min |
| time_of_day_embedding | Cyclical sin/cos encoding of hour + day | Per request |
| is_rush_hour | Boolean flag | Per request |
| active_events_nearby | Count of events in 2km radius | 5 min |

**Driver-Level Features (for pickup ETA)**

| Feature | Description |
|---|---|
| driver_current_speed | Current GPS-derived speed |
| driver_distance_to_pickup | Road distance, not straight-line |
| driver_heading_alignment | Is driver moving toward pickup? |
| driver_recent_completion_rate | Proxy for driver behavior |

**Temporal Encoding:**
```python
def cyclical_time_encoding(hour: int, day_of_week: int):
    """Encode time features cyclically to avoid boundary artifacts."""
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin  = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos  = np.cos(2 * np.pi * day_of_week / 7)
    return hour_sin, hour_cos, dow_sin, dow_cos
```

### Model Choices

**Option 1: Gradient Boosted Trees (LightGBM)**
- Strong baseline, fast inference (~5ms)
- Handles missing features (weather unavailable, GPS gaps)
- Per-city model: train on last 90 days of trips, update weekly
- Feature importance interpretable for debugging

**Option 2: Uber's DeepETA (production)**
- Transformer-based encoder of road network segments
- Processes sequence of road segments on the route
- Learns segment-level travel time embeddings
- Multi-task: predicts ETA + probability of route deviation
- ~50ms inference, significantly better MAPE vs GBT in complex cities

**Option 3: GNN on Road Network**
- Model the road network as a directed graph
- Node features: road type, speed limit, historical speed
- Edge features: turn penalties, signal timing
- GNN aggregates neighborhood context to improve segment-level estimates
- Slower to train and serve; beneficial for cities with complex topology

**Production recommendation:** LightGBM as primary with DeepETA for high-traffic cities. GNN for research/long-horizon improvement.

```python
# LightGBM per-city ETA model
import lightgbm as lgb

eta_model = lgb.LGBMRegressor(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=50,
    objective='huber',       # robust to outlier trips (accidents, detours)
    alpha=0.9,               # huber threshold
    n_jobs=-1
)

eta_model.fit(
    X_train, y_train,        # y = actual trip duration in seconds
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
```

### Uncertainty Estimation

Point estimates are insufficient — riders and the matching engine need confidence intervals.

**Approach 1: Quantile Regression**
Train separate models for p10, p50, p90 predictions. Show rider "8–12 min" range.

```python
# Train three models at different quantiles
quantile_models = {}
for q in [0.1, 0.5, 0.9]:
    model = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=800)
    model.fit(X_train, y_train)
    quantile_models[q] = model

def predict_eta_with_range(features):
    p10 = quantile_models[0.1].predict(features)[0]
    p50 = quantile_models[0.5].predict(features)[0]
    p90 = quantile_models[0.9].predict(features)[0]
    return {'estimate': p50, 'low': p10, 'high': p90}
```

**Approach 2: Natural Gradient Boosting (NGBoost)**
Outputs a full distribution (mean + variance) from a single model.

**Approach 3: Conformal Prediction**
Calibrate prediction intervals on held-out data without distributional assumptions. Works well as a post-hoc wrapper on any base model.

### Multi-Task Learning

Train a single model on two targets simultaneously:
1. ETA (regression)
2. Route deviation probability (binary classification — will driver take a different route?)

Shared backbone captures common route quality features; separate heads for each task. Deviation probability used to widen ETA confidence interval dynamically.

---

## 5. Road Network Modeling

### Graph Representation

The road network is a directed graph $G = (V, E)$:
- **Nodes $V$:** Road intersections
- **Edges $E$:** Road segments between intersections
- **Node features:** Latitude, longitude, intersection type (signal/stop/free-flow)
- **Edge features:** Speed limit, road type (highway/arterial/residential), lane count, length, historical travel times by hour/day

### H3 Hexagonal Grid

Uber's H3 library discretizes the Earth into hierarchical hexagonal cells.

**Why hexagons over grids?**
- All neighbors are equidistant (not true for squares — diagonals are farther)
- Clean hierarchical resolution (resolution 7 ≈ 1.4 km², resolution 9 ≈ 0.1 km²)
- Efficient geospatial indexing: `h3.geo_to_h3(lat, lng, resolution=9)`

**Usage in ETA:**
```python
import h3

def get_hex_features(lat: float, lng: float, resolution: int = 9):
    """Get H3 cell for a coordinate and fetch precomputed cell features."""
    cell_id = h3.geo_to_h3(lat, lng, resolution)
    # Fetch from Redis: avg speed, congestion, active driver count
    features = redis_client.hgetall(f"h3:{cell_id}:features")
    return cell_id, features

def get_route_hex_sequence(waypoints: list, resolution: int = 9):
    """Encode route as sequence of H3 cells for DeepETA-style model."""
    cells = [h3.geo_to_h3(lat, lng, resolution) for lat, lng in waypoints]
    # Deduplicate consecutive cells
    return [c for i, c in enumerate(cells) if i == 0 or c != cells[i-1]]
```

### GNN for Route Encoding

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class RoadNetworkGNN(nn.Module):
    def __init__(self, node_features: int, edge_features: int, hidden_dim: int = 128):
        super().__init__()
        # Graph attention layers
        self.conv1 = GATConv(node_features, hidden_dim, heads=4, edge_dim=edge_features)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, edge_dim=edge_features)
        # Segment-level speed prediction
        self.segment_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, node_features]
        # Propagate traffic context across local neighborhood
        h = self.conv1(x, edge_index, edge_attr).relu()
        h = self.conv2(h, edge_index, edge_attr).relu()
        # Per-segment travel time (seconds)
        return self.segment_head(h).squeeze(-1)

def route_eta_from_gnn(segment_ids: list, gnn_model, graph_data):
    """Sum per-segment predictions to get total route ETA."""
    node_embeddings = gnn_model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    segment_times = node_embeddings[segment_ids]
    return segment_times.sum().item()
```

**Update frequency:** GNN weights updated nightly. Segment-level speed features updated every 1 minute from GPS traces.

---

## 6. Surge Pricing Model

### Supply/Demand Ratio

At each H3 cell $c$ and time $t$:

$$\text{SDR}(c, t) = \frac{\text{available\_drivers}(c, t)}{\text{rider\_requests}(c, t) + \epsilon}$$

- SDR < 1: excess demand → surge needed
- SDR > 1.5: excess supply → reduce price
- SDR ≈ 1: balanced market

```python
def compute_sdr_per_cell(redis_client, h3_cells: list, window_seconds: int = 60):
    """Compute supply-demand ratio for a list of H3 cells."""
    results = {}
    for cell in h3_cells:
        drivers = int(redis_client.get(f"h3:{cell}:active_drivers") or 0)
        requests = int(redis_client.get(f"h3:{cell}:open_requests") or 0)
        results[cell] = drivers / (requests + 1e-3)
    return results
```

### Price Elasticity Estimation

**Price elasticity of demand** measures rider sensitivity to price changes:

$$\epsilon_d = \frac{\Delta Q / Q}{\Delta P / P}$$

Typical ride-hailing values: $\epsilon_d \approx -0.5$ to $-1.5$ (inelastic to moderately elastic depending on trip type, city, time of day).

Estimate elasticity from historical A/B tests and natural price variation:

```python
import statsmodels.formula.api as smf

# Log-log regression for elasticity estimation
# ln(demand) = α + ε_d * ln(price) + β * controls
elasticity_model = smf.ols(
    'np.log(requests) ~ np.log(price) + hour + day_of_week + weather + event_nearby',
    data=historical_pricing_data
).fit()

price_elasticity = elasticity_model.params['np.log(price)']
```

**Driver supply elasticity** (how many drivers come online at higher surge):
$$\epsilon_s = \frac{\Delta S / S}{\Delta P / P} \approx +0.3 \text{ to } +0.8$$

### Surge Multiplier Calculation

**Rule-based baseline:**

```python
def calculate_surge_multiplier(sdr: float, config: dict) -> float:
    """
    Simple SDR-to-multiplier mapping with smoothing.
    sdr < 0.25: extreme shortage → max surge
    sdr > 1.5:  excess supply  → no surge
    """
    if sdr >= config['balanced_threshold']:     # 1.5
        return 1.0
    elif sdr >= config['moderate_threshold']:   # 0.75
        # Linear interpolation 1.0 → 1.5x
        ratio = (config['balanced_threshold'] - sdr) / \
                (config['balanced_threshold'] - config['moderate_threshold'])
        return 1.0 + ratio * 0.5
    elif sdr >= config['high_threshold']:       # 0.4
        return 1.5 + (config['moderate_threshold'] - sdr) / \
               (config['moderate_threshold'] - config['high_threshold']) * 1.0
    else:
        return min(config['max_surge'], 2.5 + (0.4 - sdr) * 5.0)
```

**ML-based pricing (production):**

Features for the pricing model:
- Current SDR at cell + neighboring cells (spillover effects)
- SDR trend (improving or worsening)
- Historical price elasticity for this cell/time
- Event indicator and estimated demand spike
- Weather condition
- Current driver earnings vs. daily target

Target: maximize `trip_completion_rate * expected_revenue` subject to:
- `wait_time < threshold`
- `surge_multiplier <= regulatory_cap`

### Reinforcement Learning for Dynamic Pricing

**Formulation:**
- **State:** SDR, time, weather, events, recent price history
- **Action:** Surge multiplier from discrete set {1.0, 1.2, 1.5, 1.8, 2.0, 2.5}
- **Reward:** `completed_trips * fare - λ * unfulfilled_requests - μ * rider_cancellations`
- **Environment:** Simulated marketplace using historical demand/supply response curves

**Algorithm:** Proximal Policy Optimization (PPO) with simulator pre-training, then fine-tuned online with constrained exploration.

**Guardrails (non-negotiable):**
```python
SURGE_GUARDRAILS = {
    'max_multiplier': 3.0,             # hard cap
    'max_increase_per_step': 0.5,      # no jump from 1.0 to 2.5 instantly
    'min_duration_minutes': 5,          # hold surge for at least 5 min
    'regulatory_caps': {               # city-specific
        'NYC': 2.5,
        'London': 3.0,
    },
    'crisis_mode_enabled': False,       # emergency price cap override
}
```

### Spatial Smoothing

Raw SDR is noisy. Apply spatial smoothing across neighboring H3 cells to prevent sharp price boundaries:

```python
def smooth_surge_across_cells(cell_surges: dict, resolution: int = 9) -> dict:
    """Smooth surge multipliers by averaging with k-ring neighbors."""
    smoothed = {}
    for cell, surge in cell_surges.items():
        neighbors = h3.k_ring(cell, k=1)           # 7 cells including self
        neighbor_surges = [cell_surges.get(n, 1.0) for n in neighbors]
        smoothed[cell] = 0.6 * surge + 0.4 * np.mean(neighbor_surges)
    return smoothed
```

---

## 7. Real-Time Features

### Driver GPS Processing

Drivers emit GPS pings every 4 seconds. Raw GPS is noisy — must be:
1. **Map-matched:** Snap GPS coordinates to road network (HMM-based Viterbi algorithm)
2. **Speed-computed:** Derive instantaneous speed from consecutive pings
3. **State-classified:** Moving, stopped at light, idle, on-trip

```python
def process_driver_ping(driver_id: str, lat: float, lng: float, timestamp: float):
    prev = redis_client.hgetall(f"driver:{driver_id}:state")
    
    # Snap to road (simplified — real system uses Valhalla/OSRM)
    snapped_lat, snapped_lng, road_segment_id = map_match(lat, lng)
    
    # Compute speed
    if prev:
        dt = timestamp - float(prev['timestamp'])
        dist = haversine(float(prev['lat']), float(prev['lng']), snapped_lat, snapped_lng)
        speed_mps = dist / dt if dt > 0 else 0
    else:
        speed_mps = 0
    
    # Update driver state in Redis (TTL = 30s)
    redis_client.hset(f"driver:{driver_id}:state", mapping={
        'lat': snapped_lat, 'lng': snapped_lng,
        'segment_id': road_segment_id,
        'speed_mps': speed_mps,
        'timestamp': timestamp,
        'h3_cell': h3.geo_to_h3(snapped_lat, snapped_lng, resolution=9)
    })
    redis_client.expire(f"driver:{driver_id}:state", 30)
    
    # Update segment speed (aggregated for ETA features)
    redis_client.lpush(f"segment:{road_segment_id}:speeds", speed_mps)
    redis_client.ltrim(f"segment:{road_segment_id}:speeds", 0, 99)  # last 100 pings
```

### Geospatial Indexing (H3 + S2)

**H3** (hexagonal, Uber): best for demand/supply aggregation and surge pricing
**S2** (quadtree, Google): best for range queries and nearest-driver lookup

```python
# Count active drivers per H3 cell for surge calculation
def count_drivers_per_h3_cell(active_driver_cells: list, resolution: int = 9) -> dict:
    from collections import Counter
    return Counter(active_driver_cells)

# Find nearest available drivers using S2
def find_nearest_drivers(pickup_lat: float, pickup_lng: float,
                          max_radius_m: int = 5000, k: int = 10):
    import s2sphere
    center = s2sphere.LatLng.from_degrees(pickup_lat, pickup_lng)
    cap = s2sphere.Cap.from_axis_angle(
        center.to_point(),
        s2sphere.Angle.from_degrees(max_radius_m / 111320)
    )
    # Query driver spatial index (backed by Redis GEO or custom S2 index)
    return driver_spatial_index.query(cap, k=k)
```

### Event Detection

```python
def detect_demand_events(city: str, timestamp: float) -> dict:
    """
    Detect events that cause demand spikes.
    Sources: Ticketmaster API, sports league APIs, concert venue feeds.
    """
    upcoming = event_calendar.query(city=city, time_window=(timestamp, timestamp + 7200))
    
    return {
        'active_events': len(upcoming),
        'max_expected_attendance': max((e['attendance'] for e in upcoming), default=0),
        'event_types': list({e['type'] for e in upcoming}),
        'nearest_event_end_minutes': min(
            ((e['end_time'] - timestamp) / 60 for e in upcoming), default=999
        )
    }
```

---

## 8. Training Data Pipeline

### GPS Trace Processing

Raw GPS data is high-volume and noisy. Must compress and clean before training.

**Pipeline stages:**

```
Raw GPS pings (Kafka) 
    → Noise filter (Kalman filter / outlier removal)
    → Map matching (HMM Viterbi on OSRM road graph)
    → Trace compression (Douglas-Peucker algorithm)
    → Trip segmentation (group by trip_id)
    → Feature extraction (per-segment speeds, times)
    → Label generation (actual trip duration)
    → Feature store write (Hive/Parquet)
```

**Map Matching (HMM-based):**
GPS observations are noisy. The true path on the road network is the most likely sequence of road segments given the observed coordinates:

$$P(\text{path} | \text{GPS}) \propto P(\text{GPS} | \text{path}) \cdot P(\text{path})$$

- Emission probability: Gaussian around each road segment
- Transition probability: route plausibility (consistent direction, no teleportation)
- Solved with Viterbi algorithm

```python
def map_match_trace(gps_points: list, road_graph) -> list:
    """
    Returns list of (road_segment_id, entry_time, exit_time) tuples.
    Uses open-source Valhalla or OSRM for production.
    """
    # Simplified: in practice use Valhalla's map matching API
    candidates = [road_graph.get_candidates(lat, lng, radius=50)
                  for lat, lng in gps_points]
    return viterbi_map_match(candidates, transition_costs=road_graph)
```

### Label Generation

```python
def generate_eta_labels(trip_records: pd.DataFrame) -> pd.DataFrame:
    """
    actual_duration = dropoff_timestamp - pickup_timestamp
    Remove: cancelled trips, trips with GPS gaps > 60s,
            outliers (> 3 std from mean for trip distance bucket)
    """
    df = trip_records.copy()
    df['actual_duration_s'] = (df['dropoff_ts'] - df['pickup_ts']).dt.total_seconds()

    # Filter bad labels
    df = df[df['actual_duration_s'] > 0]
    df = df[df['actual_duration_s'] < 7200]   # < 2 hours
    df = df[df['gps_gap_max_s'] < 60]         # no large GPS gaps
    df = df[~df['trip_cancelled']]

    # Outlier removal per distance bucket
    df['distance_bucket'] = pd.cut(df['distance_km'], bins=[0,2,5,10,20,100])
    df['duration_zscore'] = df.groupby('distance_bucket')['actual_duration_s'] \
                              .transform(lambda x: (x - x.mean()) / x.std())
    return df[df['duration_zscore'].abs() < 3]
```

### Training Cadence

| Model | Training frequency | Data window |
|---|---|---|
| ETA (LightGBM, per city) | Weekly | 90 days |
| ETA (DeepETA, shared backbone) | Monthly | 1 year |
| GNN segment speeds | Nightly | 30 days |
| Surge pricing (rule-based calibration) | Daily | 7 days |
| Price elasticity model | Weekly | 90 days |
| RL pricing policy | Continuous (off-policy) | Rolling 30 days |

---

## 9. Model Serving

### City-Partitioned Serving

```
ETA Request (city=NYC, origin=(40.75, -73.99), dest=(40.68, -73.94))
    │
    ▼
City Router → NYC ETA Model Cluster
    │           (2 replicas, TF Serving)
    ▼
Feature Assembly (<50ms)
    │  ├── Road route from OSRM: segment list
    │  ├── Real-time speeds from Redis: per segment
    │  ├── Weather from cache: current conditions
    │  └── Time encoding: current UTC
    ▼
Model Inference (<100ms)
    │
    ▼
ETA + Confidence Interval → Response
```

### Precomputed ETA Grids

For common O-D pairs (airport to downtown, stadium to transit hub), precompute ETA every 5 minutes and cache in Redis. Covers ~30% of requests at <5ms latency.

```python
def precompute_common_etas(city: str, redis_client, interval_minutes: int = 5):
    """Pre-warm ETA cache for high-frequency OD pairs."""
    common_ods = get_top_od_pairs(city, top_k=10000)
    
    for origin, destination in common_ods:
        features = assemble_features(origin, destination)
        eta = eta_model.predict(features)
        
        cache_key = f"eta:{city}:{h3.geo_to_h3(*origin, 8)}:{h3.geo_to_h3(*destination, 8)}"
        redis_client.setex(cache_key, interval_minutes * 60, eta)
```

### Fallback Hierarchy

1. **Primary:** ML model (city-specific LightGBM or DeepETA)
2. **Fallback 1:** Precomputed ETA grid (nearest matching cell pair)
3. **Fallback 2:** Historical average for this OD by hour/day
4. **Fallback 3:** Free-flow estimate (route distance / avg city speed)

```python
def get_eta_with_fallback(origin, destination, city, timeout_ms=400):
    try:
        with timeout(timeout_ms):
            return ml_eta_service.predict(origin, destination, city)
    except (TimeoutError, ModelUnavailableError):
        pass  # fall through

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
| Feature assembly (Redis lookups) | 50ms |
| Model inference | 100ms |
| Surge pricing calculation | 50ms |
| Response serialization | 10ms |
| **Total** | **310ms** (headroom for p99) |

---

## 10. Failure Modes

### Map Changes Invalidating Models

**Problem:** A new road opens, a road closes for construction, or speed limits change. ETA model was trained on old graph data — predictions for affected routes become systematically wrong.

**Detection:** Monitor ETA residual per road segment. A sudden jump in mean error on segment $s$ signals a change.

**Mitigation:**
- Subscribe to map change feeds (OpenStreetMap changesets, HERE Maps updates)
- Invalidate per-segment feature cache when a change is detected
- Trigger city-specific model retrain within 24 hours
- Increase prediction uncertainty for segments with stale features

### Weather Edge Cases

**Problem:** Extreme weather (blizzard, hurricane, flash flood) creates conditions the model has rarely seen. Predictions collapse.

**Mitigation:**
- Hard override: if precipitation > threshold, multiply base ETA by weather scalar
- Separate weather-conditioned model for rain/snow trained on historical storm data
- Fallback to conservative (high) ETA estimates during declared weather events

### Event-Driven Demand Spikes

**Problem:** Concert lets out — 20,000 people request rides simultaneously. Model was trained on normal patterns; surge and ETA both wrong.

**Detection:** Monitor real-time request rate vs. 7-day moving average. >3σ deviation triggers event mode.

**Mitigation:**
- Pre-announce surge before event ends ("prices will increase after the game")
- Pre-position drivers by broadcasting surge ahead of time
- Pre-warm ETA cache for venue → common destinations
- Use event calendar to forecast demand and pre-scale infrastructure

### Driver Gaming Surge Zones

**Problem:** Drivers learn that turning off their app creates artificial scarcity in a cell, triggering surge, then they re-enable. Classic "hold out" behavior.

**Detection:** Anomalous simultaneous offline/online events in a small geographic area. High correlation between driver-off events and surge onset in the same cell.

**Mitigation:**
- Require sustained supply shortage (>5 min) before surge triggers
- Exclude drivers who recently went offline in that cell from surge earnings bonus
- Monitor for correlated offline patterns across driver cohorts

### Cold Start for New Cities

**Problem:** New city launch — no historical trip data for ETA model, no elasticity data for pricing model.

**Mitigation:**
1. **Transfer learning:** Initialize ETA model from nearby city or city with similar topology
2. **Manual price schedule:** Use conservative fixed pricing for first 4–6 weeks
3. **Synthetic data:** Simulate trips using map data + city speed profiles
4. **Active data collection:** Run free/discounted rides to seed training data
5. **Gradual rollout:** City-specific models replace transferred model as data accumulates (threshold: ~10K trips)

### Model Staleness During Rapid Traffic Changes

**Problem:** A major accident causes sudden gridlock. ETA model trained on normal days predicts 12 minutes; actual is 45 minutes.

**Mitigation:**
- Real-time traffic feed overrides segment speeds in feature assembly (Waze/HERE Traffic APIs)
- Compare live GPS speed from driver fleet vs. model's assumed speed per segment
- If current_speed < 0.3 * model_assumed_speed on a segment, apply dynamic correction factor

---

## 11. Interview Questions

**Q1: Why does ETA accuracy matter for business metrics beyond rider satisfaction?**

ETA feeds the matching engine — the system decides which driver to dispatch partly based on whose arrival time minimizes total wait. Poor ETA means sub-optimal driver assignments, increasing actual wait times even when enough supply exists. ETA is also used for trip fare estimation in upfront pricing; systematic under-prediction causes drivers to earn less than shown to riders, increasing driver churn.

---

**Q2: How would you handle the ETA prediction problem at airport pickups, which have very different dynamics from street hailing?**

Airport pickups have a designated staging lot where drivers wait, a structured flow (arrivals → baggage → curbside), and high variance due to flight delays. Treat airport pickup as a separate model with airport-specific features: flight landing time (from aviation APIs), baggage claim wait time by terminal, lot-to-curb travel time. Use a two-stage prediction: (1) driver staging lot → curb transition time, (2) curb ETA once driver is dispatched. Historical data is dense enough per airport to train dedicated models.

---

**Q3: Why use H3 hexagonal cells instead of square grids for surge pricing?**

Square grids have two different neighbor distances (adjacent sides vs. diagonal corners differ by $\sqrt{2}$). This creates anisotropic spatial smoothing — surge bleeds differently in cardinal vs. diagonal directions, causing artifacts at grid corners. Hexagonal cells have equal distance to all six neighbors, making spatial smoothing isotropic. H3 also supports clean hierarchical aggregation: each resolution-7 cell is composed of exactly 7 resolution-8 cells, enabling multi-scale surge analysis without boundary distortion.

---

**Q4: How do you prevent the surge pricing model from oscillating? (Surge goes up, drivers come online, SDR tips to excess, surge drops, drivers go offline, SDR tips to shortage, repeat)**

Several mechanisms: (1) **Temporal hysteresis** — require SDR to stay below threshold for N minutes before raising surge, and above threshold for M minutes before dropping it. (2) **Supply forecasting** — predict how many drivers will come online given current surge, and anticipate the supply response before setting the multiplier. (3) **Smooth multiplier changes** — cap change per interval at 0.2x. (4) **Neighboring cell coordination** — smooth surge spatially so drivers coming from adjacent cells dampen local volatility. In practice, Uber uses a predictive model for supply response rather than purely reactive SDR thresholds.

---

**Q5: How would you detect if your ETA model has degraded in production?**

Primary signal: **ETA residual monitoring** — compute `(predicted_ETA - actual_duration)` for all completed trips in a rolling 1-hour window, stratified by city, trip type, time of day. Alert if mean error > 2 minutes or MAPE > 15% in any stratum. Secondary signals: (1) rider cancellation rate spike after ETA is shown (early cancellations suggest ETA was too high), (2) low driver rating events correlated with large ETA underestimates, (3) systematic drift in a particular city's model vs. others. Use shadow mode for new model versions — compute both old and new model predictions, compare residuals before cutover.

---

**Q6: How would you design the system to handle 10x traffic spikes (New Year's Eve)?**

Pre-planned capacity scaling: (1) **Horizontal scaling** — auto-scale ETA model serving replicas and feature store read replicas based on request rate, with pre-provisioned capacity ahead of known events. (2) **Cache pre-warming** — 2 hours before the event, pre-compute ETAs for the top 50K origin-destination pairs in the city and populate Redis. (3) **Graceful degradation** — under extreme load, fall back to cached/historical ETAs rather than timing out. (4) **Surge infrastructure** — pre-configure city-specific surge parameters for known demand spikes (stadium events, New Year's) based on historical data. (5) **Load shedding** — lower request priority for non-booking ETA calls (exploratory price checks) vs. active booking requests.

---

**Q7: Your surge pricing model increases revenue but rider completion rate drops by 2%. How do you decide whether to roll back?**

This is a classic marketplace optimization problem. A 2% drop in completion rate sounds bad but could be acceptable if it represents the removal of low-value demand that was congesting the platform (riders requesting at surge who would have cancelled anyway). Key questions: (1) What happened to driver utilization? If it increased, the market cleared more efficiently. (2) What happened to rider wait times for completed trips? If wait time improved, drivers are better matched. (3) Did driver earnings per hour increase? If yes, driver retention improves over time. (4) What is the counterfactual? In a supply-constrained market, not increasing surge means some requests get no driver anyway — the "completion rate" of the no-surge baseline is artificially inflated by unfulfilled requests. Use a holdout cell experiment (A/B test across H3 cells) to measure true causal effect before rolling out city-wide. Roll back only if net marketplace value (fulfilled trips × fare) decreased.

---

## References

- Uber Engineering: [DeepETA: How Uber Predicts Arrival Times Using Deep Learning](https://www.uber.com/blog/deepeta-how-uber-predicts-arrival-times/) (2022)
- Uber Engineering: [H3: Uber's Hexagonal Hierarchical Spatial Index](https://www.uber.com/blog/h3/) (2018)
- Uber Research: [Forecasting at Uber: An Introduction](https://www.uber.com/blog/forecasting-introduction/) (2018)
- Lyft Engineering: [Lyft's Marketplace Pricing](https://eng.lyft.com/lyft-marketplace-pricing-b54c2c8e3a1b) (2018)
- Uber Research: [Surge Pricing Solves the Wild Goose Chase](https://faculty.chicagobooth.edu/christopher.knittel/research/papers/uber_surge_goose_chase.pdf) — Castillo et al., AER 2017
- Valhalla map matching: [https://github.com/valhalla/valhalla](https://github.com/valhalla/valhalla)
- OSRM routing engine: [http://project-osrm.org/](http://project-osrm.org/)

## Flashcards

**What is the ETA used for? Pre-booking estimate shown to rider, matching optimization, or driver dispatch?** #flashcard
What is the ETA used for? Pre-booking estimate shown to rider, matching optimization, or driver dispatch?

**Which leg of the trip? Pickup ETA (driver → rider), trip ETA (rider → destination), or both?** #flashcard
Which leg of the trip? Pickup ETA (driver → rider), trip ETA (rider → destination), or both?

**Acceptable error? ±2 minutes? ±20%? Does it change by trip length?** #flashcard
Acceptable error? ±2 minutes? ±20%? Does it change by trip length?

**Real-time vs batch? Should ETA update as the driver moves, or is a single pre-trip estimate sufficient?** #flashcard
Real-time vs batch? Should ETA update as the driver moves, or is a single pre-trip estimate sufficient?

**Uncertainty communication? Show a range ("8–12 min") or a point estimate?** #flashcard
Uncertainty communication? Show a range ("8–12 min") or a point estimate?

**Geographic granularity? City-level, neighborhood-level, or hexagonal cell-level?** #flashcard
Geographic granularity? City-level, neighborhood-level, or hexagonal cell-level?

**Price cap policy? Regulatory limits on surge multipliers vary by city/country?** #flashcard
Price cap policy? Regulatory limits on surge multipliers vary by city/country

**Driver-side incentives? Does surge only affect rider price, or also driver earnings?** #flashcard
Driver-side incentives? Does surge only affect rider price, or also driver earnings?

**Surge transparency? Do we show surge multiplier (1.8x) or just final price?** #flashcard
Surge transparency? Do we show surge multiplier (1.8x) or just final price?

**New city cold start? How long until the pricing model has enough data?** #flashcard
New city cold start? How long until the pricing model has enough data?

**Strong baseline, fast inference (~5ms)?** #flashcard
Strong baseline, fast inference (~5ms)

**Handles missing features (weather unavailable, GPS gaps)?** #flashcard
Handles missing features (weather unavailable, GPS gaps)

**Per-city model?** #flashcard
train on last 90 days of trips, update weekly

**Feature importance interpretable for debugging?** #flashcard
Feature importance interpretable for debugging

**Transformer-based encoder of road network segments?** #flashcard
Transformer-based encoder of road network segments

**Processes sequence of road segments on the route?** #flashcard
Processes sequence of road segments on the route

**Learns segment-level travel time embeddings?** #flashcard
Learns segment-level travel time embeddings

**Multi-task?** #flashcard
predicts ETA + probability of route deviation

**~50ms inference, significantly better MAPE vs GBT in complex cities?** #flashcard
~50ms inference, significantly better MAPE vs GBT in complex cities

**Model the road network as a directed graph?** #flashcard
Model the road network as a directed graph

**Node features?** #flashcard
road type, speed limit, historical speed

**Edge features?** #flashcard
turn penalties, signal timing

**GNN aggregates neighborhood context to improve segment-level estimates?** #flashcard
GNN aggregates neighborhood context to improve segment-level estimates

**Slower to train and serve; beneficial for cities with complex topology?** #flashcard
Slower to train and serve; beneficial for cities with complex topology

**Nodes $V$?** #flashcard
Road intersections

**Edges $E$?** #flashcard
Road segments between intersections

**Node features?** #flashcard
Latitude, longitude, intersection type (signal/stop/free-flow)

**Edge features?** #flashcard
Speed limit, road type (highway/arterial/residential), lane count, length, historical travel times by hour/day

**All neighbors are equidistant (not true for squares?** #flashcard
diagonals are farther)

**Clean hierarchical resolution (resolution 7 ≈ 1.4 km², resolution 9 ≈ 0.1 km²)?** #flashcard
Clean hierarchical resolution (resolution 7 ≈ 1.4 km², resolution 9 ≈ 0.1 km²)

**Efficient geospatial indexing?** #flashcard
h3.geo_to_h3(lat, lng, resolution=9)

**SDR < 1?** #flashcard
excess demand → surge needed

**SDR > 1.5?** #flashcard
excess supply → reduce price

**SDR ≈ 1?** #flashcard
balanced market

**Current SDR at cell + neighboring cells (spillover effects)?** #flashcard
Current SDR at cell + neighboring cells (spillover effects)

**SDR trend (improving or worsening)?** #flashcard
SDR trend (improving or worsening)

**Historical price elasticity for this cell/time?** #flashcard
Historical price elasticity for this cell/time

**Event indicator and estimated demand spike?** #flashcard
Event indicator and estimated demand spike

**Weather condition?** #flashcard
Weather condition

**Current driver earnings vs. daily target?** #flashcard
Current driver earnings vs. daily target

**wait_time < threshold?** #flashcard
wait_time < threshold

**surge_multiplier <= regulatory_cap?** #flashcard
surge_multiplier <= regulatory_cap

**State?** #flashcard
SDR, time, weather, events, recent price history

**Action?** #flashcard
Surge multiplier from discrete set {1.0, 1.2, 1.5, 1.8, 2.0, 2.5}

**Reward?** #flashcard
completed_trips  fare - λ  unfulfilled_requests - μ * rider_cancellations

**Environment?** #flashcard
Simulated marketplace using historical demand/supply response curves

**Emission probability?** #flashcard
Gaussian around each road segment

**Transition probability?** #flashcard
route plausibility (consistent direction, no teleportation)

**Solved with Viterbi algorithm?** #flashcard
Solved with Viterbi algorithm

**Subscribe to map change feeds (OpenStreetMap changesets, HERE Maps updates)?** #flashcard
Subscribe to map change feeds (OpenStreetMap changesets, HERE Maps updates)

**Invalidate per-segment feature cache when a change is detected?** #flashcard
Invalidate per-segment feature cache when a change is detected

**Trigger city-specific model retrain within 24 hours?** #flashcard
Trigger city-specific model retrain within 24 hours

**Increase prediction uncertainty for segments with stale features?** #flashcard
Increase prediction uncertainty for segments with stale features

**Hard override?** #flashcard
if precipitation > threshold, multiply base ETA by weather scalar

**Separate weather-conditioned model for rain/snow trained on historical storm data?** #flashcard
Separate weather-conditioned model for rain/snow trained on historical storm data

**Fallback to conservative (high) ETA estimates during declared weather events?** #flashcard
Fallback to conservative (high) ETA estimates during declared weather events

**Pre-announce surge before event ends ("prices will increase after the game")?** #flashcard
Pre-announce surge before event ends ("prices will increase after the game")

**Pre-position drivers by broadcasting surge ahead of time?** #flashcard
Pre-position drivers by broadcasting surge ahead of time

**Pre-warm ETA cache for venue → common destinations?** #flashcard
Pre-warm ETA cache for venue → common destinations

**Use event calendar to forecast demand and pre-scale infrastructure?** #flashcard
Use event calendar to forecast demand and pre-scale infrastructure

**Require sustained supply shortage (>5 min) before surge triggers?** #flashcard
Require sustained supply shortage (>5 min) before surge triggers

**Exclude drivers who recently went offline in that cell from surge earnings bonus?** #flashcard
Exclude drivers who recently went offline in that cell from surge earnings bonus

**Monitor for correlated offline patterns across driver cohorts?** #flashcard
Monitor for correlated offline patterns across driver cohorts

**Real-time traffic feed overrides segment speeds in feature assembly (Waze/HERE Traffic APIs)?** #flashcard
Real-time traffic feed overrides segment speeds in feature assembly (Waze/HERE Traffic APIs)

**Compare live GPS speed from driver fleet vs. model's assumed speed per segment?** #flashcard
Compare live GPS speed from driver fleet vs. model's assumed speed per segment

**If current_speed < 0.3 * model_assumed_speed on a segment, apply dynamic correction factor?** #flashcard
If current_speed < 0.3 * model_assumed_speed on a segment, apply dynamic correction factor

**Uber Engineering?** #flashcard
[DeepETA: How Uber Predicts Arrival Times Using Deep Learning](https://www.uber.com/blog/deepeta-how-uber-predicts-arrival-times/) (2022)

**Uber Engineering?** #flashcard
[H3: Uber's Hexagonal Hierarchical Spatial Index](https://www.uber.com/blog/h3/) (2018)

**Uber Research?** #flashcard
[Forecasting at Uber: An Introduction](https://www.uber.com/blog/forecasting-introduction/) (2018)

**Lyft Engineering?** #flashcard
[Lyft's Marketplace Pricing](https://eng.lyft.com/lyft-marketplace-pricing-b54c2c8e3a1b) (2018)

**Uber Research: [Surge Pricing Solves the Wild Goose Chase](https://faculty.chicagobooth.edu/christopher.knittel/research/papers/uber_surge_goose_chase.pdf)?** #flashcard
Castillo et al., AER 2017

**Valhalla map matching?** #flashcard
[https://github.com/valhalla/valhalla](https://github.com/valhalla/valhalla)

**OSRM routing engine?** #flashcard
[http://project-osrm.org/](http://project-osrm.org/)
