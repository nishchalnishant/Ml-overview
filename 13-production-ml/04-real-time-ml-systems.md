---
module: Production ML
topic: System Design
subtopic: Real Time Ml Systems
status: unread
tags: [productionml, ml, system-design-real-time-ml-sys]
---
# Real-Time ML Systems

**TL;DR**: Real-time ML is an engineering problem as much as a modeling problem. The hard parts are latency budget management, feature consistency between training and serving, feedback loops that poison future data, and graceful degradation when the stack falls apart. Interview signal comes from knowing where latency lives, how consistency breaks, and what to do when it does.

---

## 1. Latency Tiers

```
Tier            | Latency       | Use cases                          | Cost multiplier
----------------|---------------|------------------------------------|----------------
Synchronous RT  | <10ms         | Ad bidding, fraud, autocomplete    | 10-50x batch
Near-real-time  | <100ms        | Payment scoring, search ranking    | 5-10x batch
Soft RT         | <1s           | Feed ranking, recommendations      | 2-5x batch
Near-real-time  | 1s-1min       | Notification triggers              | 1.5-2x batch
Batch           | Minutes-hours | Offline training, reporting        | 1x baseline
```

**How to choose**:

```
Question                                       | Points to...
-----------------------------------------------|--------------------
Does a human wait for the response?            | <100ms
Does latency affect revenue (e.g. ad auction)? | <10ms
Is the event window short (fraud velocity)?    | Streaming features
Can the model score at request time?           | Synchronous serving
Is the data always available at request time?  | Synchronous serving
Are labels delayed days/weeks?                 | Batch retraining ok
```

**Cost tradeoffs**: the jump from batch to <100ms requires persistent infrastructure (online feature store, always-on model servers, caching layers). The jump from <100ms to <10ms additionally requires co-location, in-process model execution, and precomputed everything. Engineering cost scales super-linearly with latency requirement.

---

## 2. Stream Processing Architecture

### Kafka: Partitions, Consumer Groups, Delivery Semantics

```
Producer                         Kafka Topic
                           ┌─────────────────────────┐
Transaction events ──────► │ Partition 0  [0][1][2]   │
(keyed by user_id)          │ Partition 1  [0][1][2]   │ ◄── Consumer Group A
                           │ Partition 2  [0][1][2]   │     (Flink job, 3 tasks)
                           └─────────────────────────┘
                                                         ◄── Consumer Group B
                                                              (audit log writer)
```

**Partitioning rules**:

```
Key choice       | Ordering guarantee         | Watch out for
-----------------|----------------------------|--------------------------------------
user_id          | Per-user ordering          | Hot partitions for power users
merchant_id      | Per-merchant ordering      | Skewed merchant traffic
None (random)    | No ordering                | Max throughput, no state co-location
Composite key    | Finer-grained ordering     | Increased partition count required
```

**Delivery semantics**:

| Semantic | How | Latency | Risk |
|---|---|---|---|
| At-most-once | No retry, auto-commit offset | Lowest | Silent data loss |
| At-least-once | Retry on failure, commit after processing | Medium | Duplicate processing |
| Exactly-once | Kafka transactions + idempotent producer | Highest | 2-3x latency overhead |

**Exactly-once costs more than people expect**. Use it for financial ledgers. For ML feature aggregations, at-least-once with idempotent writes (upsert by event ID) is the pragmatic default.

### Flink vs Spark Streaming vs Kafka Streams

```
Dimension           | Flink              | Spark Streaming      | Kafka Streams
--------------------|--------------------|-----------------------|--------------------
Processing model    | True streaming     | Micro-batch (seconds)| True streaming
State management    | Built-in, scalable | External (checkpts)  | Local RocksDB
Latency floor       | ~1ms               | ~100ms-1s            | ~1ms
Exactly-once        | Yes (native)       | Yes (with effort)    | Yes (native)
ML integration      | Manual             | MLlib, pandas UDFs   | Manual
Operational burden  | High               | Medium (on Spark)    | Low
Use when            | Complex stateful   | Already on Spark     | Simple, low-ops
                    | aggregations       | ecosystem            | transformations
```

**The practical answer**: Flink for ML feature computation (complex windowed state, low latency), Kafka Streams for lightweight per-event transforms, Spark Streaming only if you're already invested in the Spark ecosystem.

### Watermarks and Late Data Handling

**The problem**: network jitter means events arrive out of order. Event timestamped 10:00:00 arrives at 10:00:45. The 10:00 window closed at 10:00:30. Drop it, or hold the window open?

```
Watermark tolerance | Window closes at    | Tradeoff
--------------------|---------------------|--------------------------------------
0s (event time)     | Window end exactly  | Zero tolerance, fast, lossy
10s                 | Window end + 10s    | Catches minor jitter, minimal delay
60s                 | Window end + 60s    | Catches mobile/flaky clients, 1min lag
5min                | Window end + 5min   | Billing/audit accuracy, high latency
```

**Side output for late events**: instead of dropping events beyond the watermark, route them to a side output for reconciliation or retraining data.

```python
# Flink: handle late arrivals without losing them
late_stream = (
    windowed_stream
    .sideOutputLateData(late_output_tag)
)
# late events go to reconciliation topic, not /dev/null
```

**Late data in ML**: late events corrupt real-time features if silently dropped. Correct response: materialize what you had at decision time, reconcile later, use point-in-time correct features for training.

---

## 3. Real-Time Feature Serving

### Online Feature Store Design

```
                    ┌─────────────────────────────────────────┐
Write path          │  Stream Processor (Flink)               │
                    │  ├── compute user_txn_count_5min        │
                    │  ├── compute user_velocity_zscore       │
Event ─────────────►│  └── compute merchant_risk_score        │
                    └──────────────┬──────────────────────────┘
                                   │ write (async, <5ms SLA)
                    ┌──────────────▼──────────────────────────┐
                    │         Online Feature Store            │
                    │  Redis Cluster  │  Cassandra            │
                    │  (hot features) │  (large embeddings)   │
                    └──────────────┬──────────────────────────┘
                                   │ read (<5ms P99)
Read path           ┌──────────────▼──────────────────────────┐
                    │    Model Serving API                    │
Request ───────────►│    1. fetch features (batch lookup)     │
                    │    2. assemble feature vector           │
                    │    3. model.predict()                   │
                    └─────────────────────────────────────────┘
```

### Redis vs Cassandra for Feature Serving

```
Dimension         | Redis                       | Cassandra
------------------|-----------------------------|---------------------------------
P99 read latency  | <1ms (in-memory)            | 2-5ms (LSM-tree, disk)
Data size limit   | RAM-bound (~GBs per node)   | Disk-bound (TBs per node)
Use for           | Hot user/session features   | Large embeddings, cold entities
Eviction          | TTL + LRU                   | TTL per column
Clustering        | Redis Cluster (hash slots)  | Consistent hashing, RF=3
Feature types     | Scalars, short vectors      | Large vectors, blobs
```

**Hybrid pattern**: hot features (computed in last hour, needed in <1ms) in Redis with TTL. Cold features (embeddings, historical aggregates) in Cassandra or DynamoDB. Read path tries Redis first, falls back to Cassandra, falls back to default values.

### Feature Freshness Guarantees

```
Freshness class | Max staleness | Implementation
----------------|---------------|-----------------------------------------------
Real-time       | <1s           | Direct write from stream processor
Near-real-time  | <1min         | Micro-batch + Redis write
Hourly          | <1h           | Cron job + bulk write
Daily           | <24h          | Batch ETL + offline→online sync
```

**Freshness SLA monitoring**: track `feature_write_timestamp` alongside feature values. At serving time, check if `now - write_time > threshold` and alert or use fallback. Stale features cause silent model degradation — harder to catch than missing features.

### Point-in-Time Correctness

**The training-serving skew problem**: at training time, you accidentally include features computed with data from after the label was observed. The model learns signal it will never have at serving time.

```
Naive (wrong):
  label_time = 2024-01-10 10:05:00 (fraud confirmed)
  feature = user_txn_count_24h computed at 2024-01-10 23:59:00  ← future data leak

Point-in-time correct:
  label_time = 2024-01-10 10:05:00
  feature = user_txn_count_24h as of 2024-01-10 10:05:00  ← correct
```

**Implementation**: feature stores (Feast, Tecton, Hopsworks) support point-in-time joins using `as_of` timestamps. When generating training data, always pass `entity_df` with event timestamps; the feature store returns the feature value that was valid at that exact moment.

---

## 4. Dual-Write Consistency Problem

**The scenario**: on a payment event, you must (1) write updated features to Redis and (2) publish an event to Kafka for downstream consumers. If only one write succeeds, systems diverge.

```
                    ┌─────────────────┐
Event arrives ─────►│  Application    │
                    │  service        │
                    └───┬─────────────┘
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
       Redis (feature        Kafka (event
          store)               bus)
              │                    │
         write ok?           write ok?
         ↓          ↓         ↓         ↓
        YES         NO       YES        NO
                    ↑                   ↑
               Redis stale        Kafka missing
               Kafka has event    Redis updated
               → downstream       → downstream
                 sees wrong        misses update
                 features
```

### Solutions

**Change Data Capture (CDC)**:

```
Application writes to DB only → Debezium reads DB WAL → publishes to Kafka
                                                       → writes to Redis

Pros: single source of truth, atomic at DB level
Cons: added latency (WAL lag ~100-500ms), Debezium operational burden
```

**Event Sourcing**:

```
Application writes event to Kafka only (the source of truth)
Feature compute service consumes Kafka → writes to Redis

Pros: Kafka is single source of truth, natural audit log
Cons: eventual consistency, Redis may lag by seconds
```

**Saga Pattern** (for multi-step distributed transactions):

```
Step 1: write to Redis (compensating action: delete)
Step 2: publish to Kafka
  if Step 2 fails → execute compensation: delete from Redis
  if Step 1 fails → abort, no Kafka publish

Pros: no distributed transaction coordinator needed
Cons: compensation logic complexity, window of inconsistency during rollback
```

**Practical choice**:

| Use case | Recommendation |
|---|---|
| Financial ledger | CDC with Debezium |
| ML feature store | Event sourcing (Kafka-first) |
| Simple dual-write | Outbox pattern (write to DB + outbox table, CDC publishes outbox) |
| Low latency critical | Accept eventual consistency, monitor lag |

---

## 5. Latency Budget Analysis

### Where Time Is Spent (P99 Breakdown)

```
Component                   | Typical P99  | Optimization levers
----------------------------|--------------|--------------------------------------------
Client → load balancer      | 1-5ms        | CDN edge, geo-routing
Load balancer → API         | 1-2ms        | Keep-alive connections, gRPC
Feature fetch (Redis)       | 1-5ms        | Batch fetch, connection pool, local cache
Feature fetch (Cassandra)   | 5-20ms       | Read repair off, speculative reads
Feature assembly            | 0.5-2ms      | Pre-allocate numpy arrays
Model inference (CPU)       | 10-100ms     | ONNX, quantization, smaller model
Model inference (GPU)       | 2-20ms       | Dynamic batching, TensorRT
Post-processing             | 0.5-5ms      | Vectorize, avoid Python loops
Write result to cache       | 1-3ms        | Async (fire-and-forget)
API → client                | 1-5ms        | Compression, binary protocol
```

**Total budget example for <100ms P99**:

```
Feature fetch (batched):   5ms
Model inference:          15ms
Pre/post-processing:       3ms
Network (both ways):      10ms
Overhead (serialization):  5ms
                          ----
Budget used:              38ms
Safety margin:            62ms  ← absorbs tail latency spikes
```

### P99 vs P50 Gap

A large P99/P50 gap (e.g., P50=10ms, P99=200ms) signals:
- GC pauses (JVM heap pressure, Python GC)
- Thundering herd on cache miss
- Stragglers in batched feature fetch (wait for slowest key)
- CPU throttling in container (check `cpu_throttled_seconds`)

**Mitigation**: hedged requests (send to 2 replicas, take first response), speculative execution in Flink, async feature prefetch.

---

## 6. Model Serving at Low Latency

### Batching Strategies

```
Strategy         | Latency impact    | Throughput  | Use case
-----------------|-------------------|-------------|---------------------------
No batching      | Lowest (per req)  | Low         | <1ms SLA, simple models
Static batching  | Fixed wait time   | Medium      | Predictable load
Dynamic batching | Variable wait     | High        | GPU serving, burst traffic
Micro-batching   | ms-scale wait     | Very high   | Triton, TensorRT server
```

**Dynamic batching** (TensorRT Inference Server / Triton): accumulate requests for up to N microseconds or until batch size B, then execute as one kernel call. For GPU inference, batch size 8-32 often gives 5-10x throughput vs batch size 1 with <2ms added latency.

```python
# Triton dynamic batching config
model_config = {
    "dynamic_batching": {
        "preferred_batch_size": [8, 16],
        "max_queue_delay_microseconds": 1000,  # wait up to 1ms
    }
}
```

### Model Optimization Stack

```
Optimization      | Latency reduction | Accuracy impact  | Effort
------------------|-------------------|------------------|--------
ONNX export       | 1.5-3x            | None             | Low
INT8 quantization | 2-4x              | <1% typical      | Medium
TensorRT compile  | 3-10x (GPU)       | <0.5%            | Medium
Layer fusion      | 1.2-2x            | None             | Low (automatic)
Pruning (80%)     | 2-5x              | 1-3%             | High
Knowledge distill | 3-10x (smaller)   | 2-5%             | High
```

**Inference optimization priority**: profile first. 80% of latency is usually in one component. Fix that before generalizing.

### Model Caching and Connection Pooling

```python
# Pattern: warm model cache + connection pool
import redis
from functools import lru_cache

# Model cache: LRU in-process, avoid re-loading weights
@lru_cache(maxsize=4)
def load_model(model_version: str):
    return onnxruntime.InferenceSession(f"models/{model_version}.onnx")

# Feature store connection pool
redis_pool = redis.ConnectionPool(
    host='redis-cluster',
    max_connections=50,      # tune to QPS * feature_fetch_time
    socket_timeout=0.005,    # 5ms timeout, not infinite
    socket_connect_timeout=0.010,
)

def get_features(user_ids: list[str]) -> dict:
    r = redis.Redis(connection_pool=redis_pool)
    pipe = r.pipeline()
    for uid in user_ids:
        pipe.hgetall(f"features:{uid}")
    return dict(zip(user_ids, pipe.execute()))  # batch fetch
```

**Connection pool sizing**: `pool_size = QPS × avg_feature_fetch_latency`. If QPS=1000 and fetch=5ms, minimum pool size=5. Set to 2-3x that for burst headroom.

---

## 7. Feedback Loops in Production

### How Predictions Corrupt Future Data

```
t=0: model scores users, shows ads to predicted high-CTR users
t=1: low-CTR users never shown ads → no click data → model thinks they have 0 CTR
t=2: model retrained on biased data → even more extreme top/bottom split
t=3: exposure bias entrenches; model can't generalize to unexposed population
```

**Three feedback loop types**:

| Loop type | Example | Effect |
|---|---|---|
| Exposure bias | Recommend only popular items | Popular gets more popular, long tail dies |
| Label shift | Fraud model flags more → fraud adapts | Adversarial drift |
| Delayed labels | Churn labels arrive 30 days late | Training data always stale |

### Delayed Labels

**The mechanics**: model predicts churn at day 0. Ground truth (did user churn?) only known at day 30. If you retrain every week, most labels are unconfirmed.

```
Strategies:
1. Proxy labels: use leading indicators (session frequency drop, feature usage)
2. Label delay modeling: weight recent labels lower in loss function
3. Survival analysis: model time-to-event, not binary outcome
4. Embargo window: don't use last N days of data for training (avoids partial labels)
```

### Drift Detection and Correction

```
Drift type      | What shifts                    | Detection method
----------------|--------------------------------|---------------------------
Data drift      | Input feature distribution     | PSI, KS test, MMD
Concept drift   | P(y|X) changes                 | Monitor prediction accuracy
Label drift     | P(y) changes                   | Monitor label distribution
Feature drift   | Upstream data pipeline changes | Schema validation + stats
```

**Response**:

```
Drift severity  | Response
----------------|---------------------------------------------------
Minor (PSI<0.1) | Alert only, monitor
Moderate        | Trigger retraining, increase monitoring cadence
Severe          | Shadow model, fallback to conservative model
Distribution ∅  | Halt serving, escalate to on-call
```

---

## 8. Multi-Objective Optimization

### Pareto Tradeoffs in Production ML

Real systems optimize several competing objectives simultaneously:

```
Objective pairs         | Tradeoff
------------------------|--------------------------------------------------
Precision vs Recall     | Lower threshold: more recalls, more false positives
Latency vs Accuracy     | Smaller model: faster, less accurate
Cost vs Throughput      | Fewer GPUs: cheaper, higher latency
Fairness vs Accuracy    | Constrained optimization often reduces raw accuracy
```

### Scalarization Approaches

**Weighted sum**: `score = w1 * precision + w2 * (-latency_ms/100)`. Simple, but weights are arbitrary; doesn't find all Pareto-optimal points.

**Constrained optimization** (most practical):

```python
# Maximize recall subject to latency < 50ms and precision > 0.9
# Use during threshold search / model selection
candidates = []
for threshold in np.arange(0.1, 0.9, 0.01):
    metrics = evaluate(model, threshold)
    if metrics['p99_latency_ms'] < 50 and metrics['precision'] > 0.90:
        candidates.append((threshold, metrics['recall']))

best_threshold = max(candidates, key=lambda x: x[1])[0]
```

**Pareto front serving policies**:

```
Policy                   | When to use
-------------------------|------------------------------------------
Single model, tuned      | Objectives align most of the time
Ensemble with routing    | Different user segments need diff tradeoffs
Multi-armed bandit       | Explore tradeoff space live in production
Contextual model switch  | Low-stakes → fast model, high-stakes → slow model
```

**Practical example** — fraud detection:
- High-value transactions: maximize recall (miss fewer frauds), tolerate 200ms
- Micro-transactions: maximize precision (false positives are costly), require <20ms

Route by transaction amount to different model versions with different thresholds.

---

## 9. Failure Modes and Graceful Degradation

### Failure Taxonomy

```
Failure                       | Symptom                        | Severity
------------------------------|--------------------------------|----------
Feature store down            | Feature fetch timeout          | High
Model server OOM              | 503 from serving endpoint      | Critical
Stale features (>TTL)         | Silent accuracy degradation    | Medium
Kafka consumer lag >10min     | Feature drift, stale state     | High
GC pause                      | P99 spike                      | Medium
Model version mismatch        | Schema error at inference      | Critical
Training-serving skew         | Production accuracy lower      | Medium
```

### Fallback Hierarchy

```
Level 1: Primary path
         Online features + latest model version
                    ↓ (if feature store latency >20ms)
Level 2: Cached predictions
         Last prediction for this entity, TTL-bounded (5min for fraud)
                    ↓ (if model server unavailable)
Level 3: Simpler fallback model
         Logistic regression, rules-based scorer — in-process, no network
                    ↓ (if all ML unavailable)
Level 4: Business rules
         Hard-coded thresholds: block txn > $10,000 from new account
                    ↓ (last resort)
Level 5: Default action
         Allow all (fraud) or block all (safety-critical) based on risk tolerance
```

### Circuit Breaker Pattern

```python
import time
from enum import Enum

class State(Enum):
    CLOSED = "closed"       # normal operation
    OPEN = "open"           # failing, use fallback
    HALF_OPEN = "half_open" # testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=30):
        self.state = State.CLOSED
        self.failures = 0
        self.threshold = failure_threshold
        self.last_failure_time = None
        self.timeout = timeout

    def call(self, fn, fallback_fn, *args):
        if self.state == State.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = State.HALF_OPEN
            else:
                return fallback_fn(*args)   # circuit open, use fallback

        try:
            result = fn(*args)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            return fallback_fn(*args)

    def _on_success(self):
        self.failures = 0
        self.state = State.CLOSED

    def _on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.threshold:
            self.state = State.OPEN
```

**Timeout budget**: set feature store timeout to ~10% of total latency budget. A 100ms SLA means a 10ms timeout on Redis. A hanging feature call should not consume the entire budget before falling back.

---

## 10. Case Study: Real-Time Fraud Detection

### End-to-End Architecture

```
                     ┌──────────────────────────────────────────────────────┐
                     │                    WRITE PATH                        │
                     │                                                      │
Payment event        │  Kafka         Flink Job               Online Store  │
─────────────────────►  [topic:txns] ─► compute velocity   ─► Redis        │
                     │               ─► compute geo anomaly ─► Redis        │
                     │               ─► compute device risk ─► Redis        │
                     │               ─► raw event           ─► Kafka[audit] │
                     └──────────────────────────────────────────────────────┘

                     ┌──────────────────────────────────────────────────────┐
                     │                     READ PATH (<100ms budget)        │
                     │                                                      │
Payment API          │  Auth Service                                        │
(synchronous call)   │  ├── fetch features from Redis   [5ms]               │
─────────────────────►  ├── fetch user history Cassandra [8ms]              │
                     │  ├── assemble feature vector      [1ms]              │
                     │  ├── call fraud model API         [15ms]             │
                     │  ├── apply business rules         [1ms]              │
                     │  └── return {allow, block, review}[2ms]             │
                     │                              total: ~32ms P50        │
                     └──────────────────────────────────────────────────────┘

                     ┌──────────────────────────────────────────────────────┐
                     │                   FEEDBACK PATH                      │
                     │                                                      │
                     │  Chargeback events ─► Kafka[labels] ─► Label store  │
                     │  Manual review     ─► Label store                   │
                     │  Label store + features ─► Training pipeline        │
                     │  Training pipeline ─► Model registry ─► Canary deploy│
                     └──────────────────────────────────────────────────────┘
```

### Feature Set

```
Feature group         | Features                          | Staleness
----------------------|-----------------------------------|----------
Velocity (streaming)  | txn_count_5min, total_amount_1h   | <5s
Account history       | avg_daily_spend, account_age_days | Daily
Device signals        | device_seen_before, new_country   | <1min
Merchant risk         | merchant_fraud_rate_30d           | Hourly
Graph features        | shared_device_with_fraud_account  | Hourly
```

### Decision Logic

```python
def score_transaction(txn: dict) -> dict:
    features = fetch_features_batch(
        user_id=txn['user_id'],
        merchant_id=txn['merchant_id'],
        device_id=txn['device_id'],
    )

    # Hard rules — no ML needed, fast
    if txn['amount'] > 50_000 and features['account_age_days'] < 7:
        return {'decision': 'block', 'reason': 'new_account_high_value'}

    # ML score
    score = fraud_model.predict(features)

    # Dynamic threshold by transaction type
    threshold = get_threshold(txn['channel'], txn['amount'])
    decision = 'block' if score > threshold else 'allow'

    # Async: log decision + score for feedback loop
    publish_async('fraud-decisions', {
        'txn_id': txn['id'],
        'score': score,
        'features_snapshot': features,  # point-in-time snapshot for training
        'decision': decision,
        'timestamp': time.time(),
    })

    return {'decision': decision, 'score': score}
```

### Monitoring Checklist

```
Signal                  | Alert threshold       | Indicates
------------------------|----------------------|-----------------------------
Feature fetch P99       | >20ms                | Redis pressure
Model inference P99     | >50ms                | Model server overload
Kafka consumer lag      | >30s                 | Stream processor falling behind
Fraud rate (model)      | >3σ from rolling avg | Drift or incident
False positive rate     | >2x baseline         | Model degraded or distribution shift
Feature null rate       | >1%                  | Upstream data pipeline broken
```

---

## 11. Interview Questions

**Q1: You need to serve a gradient boosted model in <50ms P99. Walk me through how you'd architect this.**

Decompose the latency budget: ~10ms for feature fetch (Redis, batched), ~5ms for pre-processing, ~15ms for model inference (ONNX-optimized GBM, CPU), ~10ms for network. Export the model to ONNX, profile the feature fetch (single round trip with pipeline), cache static features in-process with a TTL. Set circuit breakers on the feature store with a 10ms timeout and a cached-prediction fallback. Monitor P99 at each component boundary, not just end-to-end.

---

**Q2: Your real-time feature store shows that 5% of feature values are stale by >10 minutes. What do you investigate?**

First check Kafka consumer lag on the Flink job — if lag is high, the stream processor isn't keeping up (CPU, memory, partition skew). Check if any partitions are hot. Next, check Redis write latency — if the Flink→Redis write path is slow, features may be computed but not yet stored. Check TTL configuration — values may be expiring before refresh. Check watermark settings — if the watermark is too conservative, windows may not emit. Correlate the stale entities: are they all the same user segment, merchant, or device type? Skew often means a hotspot.

---

**Q3: How do you prevent training-serving skew in a system where features are computed in real time?**

Three things: (1) point-in-time correct training data — when generating training examples, use the feature value that existed at the label's event time, not the current value. Feature stores with time-travel support this. (2) Log features at serving time — attach the feature snapshot to the prediction log. Use that logged snapshot for training, not recomputed features. (3) Validate continuously — run a shadow job that recomputes features offline for recent predictions and compares to the logged values; alert on systematic differences.

---

**Q4: A fraud model's precision drops from 92% to 78% in production over six weeks. The feature distributions look the same. What's happening?**

Concept drift: P(fraud | X) changed even though P(X) is stable. Likely causes: fraudsters adapted to the model's decision boundary (adversarial shift), fraud patterns shifted seasonally, or the population of transactions changed (new merchant categories, new payment rails). Investigate: look at the cases where the model is newly wrong — are they concentrated in a segment? Check if the decision boundary itself shifted (score distribution vs outcome). Short-term fix: lower threshold to restore recall. Medium-term: retrain with recent labeled data. Long-term: add adversarial training, shorter retraining cadence, ensemble with rule-based signals that are harder to game.

---

**Q5: Describe the dual-write consistency problem and your preferred solution for an ML feature store.**

When a single event must update both a feature store and a message queue, a partial failure leaves them inconsistent. For ML feature stores, I prefer the event-sourcing approach: Kafka is the single source of truth. The application publishes the event to Kafka only. The feature compute service (Flink) consumes Kafka and writes to Redis. Redis is a derived, eventually-consistent projection of Kafka. This means feature values may lag by seconds, which is acceptable for most ML use cases. For the rare case where strong consistency matters (financial ledger features), CDC with Debezium reads the DB write-ahead log and publishes to Kafka, so the DB is the source of truth and Kafka is the fan-out mechanism.

---

**Q6: How would you handle a model that must make decisions within 10ms?**

At 10ms, every network hop is expensive. Design principles: (1) embed the model in-process (no network call) — load ONNX/TFLite directly in the application server. (2) Precompute as much as possible — store fully assembled feature vectors in Redis, not raw features requiring aggregation at serve time. (3) Use a simpler model — a 100-tree GBM with INT8 quantization can score in <1ms in-process. (4) Warm everything up at startup — model weights loaded, Redis connections established, JVM/interpreter warmed. (5) Pin to a CPU core — avoid context switching during inference. (6) Profile with realistic traffic — P99 under load is different from P99 on a quiet machine.

---

**Q7: You're designing a recommendation system that must serve 10,000 RPS with P99 < 100ms. What are the main risks and how do you mitigate them?**

Main risks: (1) Feature fetch fan-out — if each request fetches features for 50 candidates, that's 500k Redis reads/sec. Mitigate with batch pipelining and candidate pre-filtering. (2) Model inference throughput — at 10k RPS, you need ~10k inferences/sec. Profile batch size vs latency tradeoff; GPU with dynamic batching likely required. (3) Cold start — new users/items have no features; need a fallback (popularity-based, demographic features). (4) Thundering herd — cache expiry causes simultaneous re-fetches; use staggered TTLs and probabilistic early expiration. (5) Feedback loop — showing only top-ranked items creates exposure bias; add exploration (epsilon-greedy or UCB) and log counterfactuals for unbiased offline evaluation.

## Flashcards

**What four things typically cause a large P99/P50 latency gap?** #flashcard
GC pauses (JVM/Python), thundering herd on cache miss, stragglers in a batched feature fetch (waiting for the slowest key), and CPU throttling in a container.

**Why is exactly-once Kafka delivery usually overkill for ML feature pipelines?** #flashcard
It costs 2-3x latency overhead. At-least-once delivery with idempotent writes (upsert by event ID) is the pragmatic default; reserve exactly-once for financial ledgers.

**What is point-in-time correctness and why does violating it cause training-serving skew?** #flashcard
Training features must be computed using only data available as of the label's event time. If a feature is computed with data from after the label was observed, the model learns signal it will never have at serving time (future leakage).

**Event sourcing vs. CDC for the dual-write consistency problem — when to use each?** #flashcard
Event sourcing (Kafka as sole source of truth, feature store derived from it) fits ML feature stores — eventual consistency is acceptable. CDC (Debezium reads DB WAL, publishes to Kafka) fits financial ledgers where the DB must be the single source of truth.

**How does exposure bias become a self-reinforcing feedback loop?** #flashcard
Model shows ads/items only to predicted high-CTR users; low-scored users are never shown anything, so no click data accumulates for them; retraining on this biased data pushes the split further, and the model can't generalize to the unexposed population.

**Fraud detection: why route high-value vs. micro-transactions to different thresholds?** #flashcard
High-value transactions should maximize recall (miss fewer frauds) and can tolerate ~200ms latency; micro-transactions should maximize precision (false positives are costly relative to transaction value) and need <20ms latency.

**What's the fallback hierarchy when a real-time ML system's primary path fails?** #flashcard
Online features + latest model → cached predictions (TTL-bounded) → simpler in-process fallback model (e.g. logistic regression) → hard-coded business rules → default action (allow-all or block-all based on risk tolerance).
