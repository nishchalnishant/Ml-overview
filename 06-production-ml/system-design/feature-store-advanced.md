# Feature Store Architecture

A feature store is the data infrastructure layer between raw data and ML models. Getting it wrong causes training-serving skew, stale features, and repeated feature computation across teams.

**Core responsibilities:** point-in-time correct features, online/offline parity, streaming feature computation, feature sharing across models.

---

## 1. The Core Problem: Training-Serving Skew

Without a feature store, features are recomputed differently in training and serving:

```
Training pipeline:
  SELECT user_id, COUNT(*) as txn_count_7d
  FROM transactions WHERE date BETWEEN t-7d AND t
  → Batch job, runs nightly, uses full historical data

Serving pipeline:
  redis.get(f"user:{user_id}:txn_count_7d")
  → Cached value, updated every 10 minutes
```

**Skew sources:**
- Different aggregation windows (7 days exact vs approximately 7 days)
- Different null handling
- Different data sources (warehouse vs operational DB)
- Staleness (serving cache is hours behind training data)

**Impact:** 5–15% model degradation in production vs offline metrics is often explained by this skew.

---

## 2. Architecture

```
Raw Data Sources
│
├── Batch (data warehouse: BigQuery, Snowflake)
├── Streaming (Kafka, Kinesis)
└── Online (operational DB, APIs)
          │
          ▼
┌──────────────────────────────┐
│     Feature Transformation   │
│  Spark (batch) / Flink (stream)│
└──────────────┬───────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌─────────────┐
│  Offline    │  │   Online    │
│  Store      │  │   Store     │
│ (S3/BigQuery│  │ (Redis/     │
│  Parquet)   │  │  Cassandra) │
└─────────────┘  └─────────────┘
       │               │
       ▼               ▼
  Training          Serving
  (point-in-        (<10ms
   time join)        lookup)
```

---

## 3. Point-in-Time Correctness

**The key correctness constraint:** when training, features must reflect what was known at the time of the label event — not what was computed later.

**Without point-in-time correctness (data leakage):**
```sql
-- WRONG: uses features that weren't available at prediction time
SELECT f.user_7d_spend, l.fraud_label
FROM features f
JOIN transactions t ON f.user_id = t.user_id
JOIN labels l ON t.txn_id = l.txn_id
-- f.user_7d_spend may include transactions AFTER t.txn_time!
```

**With point-in-time correctness:**
```sql
-- CORRECT: only use features computed before the event
SELECT f.user_7d_spend, l.fraud_label
FROM transactions t
JOIN labels l ON t.txn_id = l.txn_id
-- Get the most recent feature snapshot before the transaction
JOIN LATERAL (
    SELECT user_7d_spend
    FROM feature_snapshots
    WHERE user_id = t.user_id
      AND snapshot_time <= t.txn_time  -- ← key constraint
    ORDER BY snapshot_time DESC
    LIMIT 1
) f ON TRUE
```

### Feast Implementation

```python
from feast import FeatureStore, FeatureView, Entity, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

# Define feature view
user_stats_fv = FeatureView(
    name="user_statistics",
    entities=["user_id"],
    ttl=timedelta(days=7),
    schema=[
        Field(name="txn_count_7d", dtype=Int64),
        Field(name="avg_txn_amount_7d", dtype=Float32),
        Field(name="unique_merchants_30d", dtype=Int64),
    ],
    source=FileSource(path="s3://features/user_stats/", timestamp_field="event_timestamp"),
)

store = FeatureStore(repo_path=".")
store.apply([user_stats_fv])

# Training: point-in-time join
entity_df = pd.DataFrame({
    "user_id": labels["user_id"],
    "event_timestamp": labels["txn_time"],  # label event time
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["user_statistics:txn_count_7d", "user_statistics:avg_txn_amount_7d"],
).to_df()
# Feast performs the point-in-time join: finds feature value <= event_timestamp for each row

# Serving: online lookup
online_features = store.get_online_features(
    features=["user_statistics:txn_count_7d"],
    entity_rows=[{"user_id": "u_12345"}],
).to_dict()
```

---

## 4. Online/Offline Parity

**Parity test:** compare feature distributions from historical store vs online store for the same entities and timestamps.

```python
def check_online_offline_parity(store, feature_names, entity_ids, sample_size=1000):
    """Detect online/offline feature discrepancy."""
    # Offline: historical features at current time
    entity_df = pd.DataFrame({
        "user_id": entity_ids[:sample_size],
        "event_timestamp": [datetime.now()] * sample_size
    })
    offline_features = store.get_historical_features(
        entity_df=entity_df,
        features=feature_names
    ).to_df()
    
    # Online: current online store values
    online_features = store.get_online_features(
        features=feature_names,
        entity_rows=[{"user_id": uid} for uid in entity_ids[:sample_size]]
    ).to_df()
    
    # Compare distributions
    discrepancies = {}
    for feat in feature_names:
        name = feat.split(":")[1]
        offline_vals = offline_features[name].dropna()
        online_vals = online_features[name].dropna()
        
        if len(offline_vals) > 0 and len(online_vals) > 0:
            from scipy.stats import ks_2samp
            stat, p_value = ks_2samp(offline_vals, online_vals)
            mean_diff = abs(offline_vals.mean() - online_vals.mean()) / (offline_vals.std() + 1e-10)
        
        discrepancies[name] = {
            "ks_statistic": stat,
            "p_value": p_value,
            "normalized_mean_diff": mean_diff,
            "alert": stat > 0.1 or mean_diff > 0.1
        }
    
    return discrepancies
```

**Parity SLA:**
- Mean feature value: within 5% between online/offline
- KS statistic: < 0.1
- Null rate: within 1 percentage point

---

## 5. Streaming Features

For real-time features (e.g., transactions in last 1 hour), batch computation is too slow.

**Architecture: Lambda pattern**
```
Kafka events → Flink → Redis (online store)
                    → S3 / BigQuery (offline store, for training)
```

**Flink windowed aggregation:**
```python
# Apache Flink windowed feature computation
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import AggregateFunction, WindowFunction
from pyflink.datastream.window import TumblingEventTimeWindows, Time

class TxnVelocityAgg(AggregateFunction):
    def create_accumulator(self):
        return {"count": 0, "total_amount": 0.0}
    
    def add(self, value, accumulator):
        accumulator["count"] += 1
        accumulator["total_amount"] += value["amount"]
        return accumulator
    
    def get_result(self, accumulator):
        return accumulator
    
    def merge(self, a, b):
        return {
            "count": a["count"] + b["count"],
            "total_amount": a["total_amount"] + b["total_amount"]
        }

# Stream: 1-hour tumbling window per user
env = StreamExecutionEnvironment.get_execution_environment()
transactions = env.add_source(kafka_source)

velocity_features = (
    transactions
    .key_by(lambda x: x["user_id"])
    .window(TumblingEventTimeWindows.of(Time.hours(1)))
    .aggregate(TxnVelocityAgg(), ...)
)

# Write to Redis (online) and S3 (offline) simultaneously
velocity_features.add_sink(redis_sink)
velocity_features.add_sink(s3_sink)
```

**Backfill problem:** when deploying a new streaming feature, you need historical values for training. Solution: replay Kafka events through the same Flink job on historical data.

---

## 6. Feature Freshness vs Latency Trade-offs

| Feature type | Computation | Update cadence | Staleness tolerance | Storage |
|---|---|---|---|---|
| Real-time (velocity, last txn) | Flink/Spark Streaming | < 1 min | Seconds | Redis |
| Near-real-time (daily stats) | Spark batch | 1–24 hours | Minutes | Redis + S3 |
| Historical (user lifetime value) | SQL warehouse | Daily/weekly | Hours | BigQuery + Redis cache |
| Model predictions as features | Async inference | Event-driven | Seconds | Redis |

**Decision framework:**
- Feature changes in < 5 min and impacts model output → streaming required
- Feature changes hourly and model runs < 1 day before prediction → batch OK
- Feature changes daily → can use offline store with daily refresh

---

## 7. Feast vs Tecton Architecture Comparison

| Dimension | Feast (open source) | Tecton (managed) |
|---|---|---|
| Deployment | Self-hosted | SaaS / on-prem |
| Streaming features | Via Spark/Flink integration | Native Spark Streaming |
| Point-in-time joins | Yes (Spark SQL) | Yes (optimized) |
| Online store | Redis, DynamoDB, SQLite | Redis, DynamoDB |
| Offline store | BigQuery, Redshift, S3 | Same |
| Feature monitoring | Basic | Advanced (drift detection) |
| Feature sharing | Feature registry | Feature registry + governance |
| Best for | Control, cost, flexibility | Enterprise, managed scale |

---

## 8. Feature Reuse and Registry

**Problem:** Without a registry, team A computes `user_7d_txn_count` and team B independently computes `user_txn_count_past_week` — same feature, twice the cost.

**Feature registry structure:**
```yaml
feature_views:
  user_transaction_stats:
    entity: user_id
    source: transactions_table
    features:
      - name: txn_count_7d
        dtype: int64
        description: "Number of transactions in last 7 days"
        owner: fraud_team
        tags: [fraud, velocity]
      - name: total_spend_30d
        dtype: float32
        description: "Total spend in USD in last 30 days"
        owner: risk_team
    ttl: 7 days
    freshness_sla: 1 hour
```

**Discovery:** teams search registry by entity, tag, or description before writing new features.

---

## 9. Feature Monitoring

**Three types of drift to monitor:**

1. **Feature drift (covariate shift):** distribution of feature values changes
   ```python
   # PSI for feature drift monitoring
   def population_stability_index(reference, current, buckets=10):
       """PSI > 0.25 indicates significant drift."""
       ref_hist, edges = np.histogram(reference, bins=buckets, density=True)
       cur_hist, _ = np.histogram(current, bins=edges, density=True)
       
       # Avoid log(0)
       ref_hist = np.clip(ref_hist, 1e-10, None)
       cur_hist = np.clip(cur_hist, 1e-10, None)
       
       psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
       return psi
   ```

2. **Feature freshness:** is the feature being updated on schedule?
   ```python
   def check_feature_freshness(store, feature_view_name, max_staleness_hours=2):
       latest = store.get_latest_feature_timestamp(feature_view_name)
       staleness = (datetime.now() - latest).total_seconds() / 3600
       if staleness > max_staleness_hours:
           alert(f"{feature_view_name} is {staleness:.1f}h stale, SLA={max_staleness_hours}h")
   ```

3. **Online/offline parity drift:** did parity worsen after a pipeline change?

---

## Canonical Interview Q&As

**Q: What is point-in-time correctness and why does it matter?**  
A: Point-in-time correctness means that when constructing training data, each training example uses only features that were available at the time of the target event — not features computed from data that arrived later. Violating this causes leakage: the model sees future information during training but not at serving time, inflating offline metrics. The most common violation: joining on entity key without a timestamp constraint, so a user's "30-day spend" in the training feature includes transactions that happened after the fraud event you're predicting. Fix: for every feature, store the computation timestamp, and during historical joins, fetch the most recent feature value with timestamp ≤ event timestamp.

**Q: How do you ensure online/offline parity?**  
A: Four controls: (1) Same transformation code — both online and offline pipelines use the same feature computation code, often from a shared library; (2) Same data source — online pipeline writes derived features back to S3 as it computes, so offline training can use the exact same materialized values; (3) Shadow testing — before launch, compare online store values with offline historical values for the same entity/timestamp pairs; alert if distributions diverge (KS > 0.1 or mean diff > 5%); (4) Monitor parity continuously — run the shadow test on a sample every hour; if parity degrades after a pipeline update, roll back. Root causes of parity failure: different null handling, timezone bugs, different data freshness, schema changes in upstream data.

**Q: When would you use streaming features vs batch features?**  
A: The decision turns on two axes: how fast the feature changes, and how much it affects model output. Real-time velocity features (transactions in last 1 hour, login attempts in last 10 minutes) change in minutes and directly impact fraud/risk models — they require streaming computation via Flink/Kafka. User lifetime value or 90-day spending patterns change slowly — daily batch jobs are sufficient. The operational cost of streaming is high (Flink cluster, Kafka, always-on infrastructure), so only invest in streaming when the model measurably degrades with stale features. Test by deliberately delaying feature refresh in offline evaluation and measuring PR-AUC drop.

**Q: How do you backfill a streaming feature for historical training data?**  
A: Streaming features are challenging to backfill because the computation is defined on an event stream with a specific timestamp. The correct approach: replay the raw event stream (from Kafka's topic retention or S3 archive) through the same Flink/Spark streaming job, but in batch mode, writing outputs with their original event timestamps to the offline store (S3/BigQuery). This produces historically correct feature values that match what the streaming pipeline would have produced in real time. Key gotchas: (1) ensure watermarking logic handles late arrivals the same way as production; (2) write results with `feature_timestamp = window_end_time`, not backfill time; (3) backfilled features must pass point-in-time parity checks against any held-out period before using for training.
