# Interview 14 — Real-Time Feature Store Design
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer working on the core ML Platform team. Multiple game teams are building real-time models (e.g., matchmaking, fraud detection, in-game recommendations). They all struggle with the same problem: joining historical batch data (e.g., "lifetime spend") with real-time streaming data (e.g., "items bought in the last 5 minutes") at inference time.

Your task is to **design and implement a Real-Time Feature Store architecture** that serves features to ML models in <10ms, while ensuring consistency between training data and serving data.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- **Training-Serving Skew:** How do we guarantee the features used for training perfectly match the features served in production?
- **Latency & Throughput:** What are the read/write QPS constraints?
- **Point-in-Time Correctness:** When generating training data, how do we prevent data leakage (time travel)?
- **Data Sources:** Where is the raw data coming from? (Kafka, Snowflake, S3?)
- **Tools:** Are we building from scratch or using an open-source framework like Feast/Hopsworks?

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"What is the read latency SLA for model inference?"**
   → *Answer: p99 < 10ms.*

2. **"How do we prevent training-serving skew?"**
   → *Answer: We need a system where a data scientist defines a feature once, and it is automatically computed for both the online and offline stores.*

3. **"How do we handle 'Point-in-Time' joins for historical training data?"**
   → *Answer: We need a mechanism to join features exactly as they existed at `timestamp=T` without leaking data from `T+1`.*

4. **"Are we allowed to use open-source feature stores, or build from scratch?"**
   → *Answer: You should base your architecture on industry standards like Feast, but you must explain how the underlying infrastructure works.*

---

## Part 4 — Expected Assumptions

- **Dual-Store Architecture:** An Offline Store (Snowflake/BigQuery) for training, and an Online Store (Redis/DynamoDB) for low-latency inference.
- **Ingestion:** Two paths. Batch ingestion (Airflow daily jobs) and Streaming ingestion (Kafka -> Flink -> Online Store).
- **Point-in-time correctness (AS OF joins):** Crucial for building accurate training datasets.

---

## Part 5 — High-Level Solution

```
  [Data Sources]
  Kafka (Streaming) ➔ Flink (Streaming Aggregations)
  Snowflake (Batch) ➔ Airflow (Batch Aggregations)
       │                    │
       ▼                    ▼
  [Feature Registry / Orchestrator (e.g., Feast)]
  (Data scientists define features here in Python/YAML)
       │                    │
       ▼                    ▼
  [Online Store]       [Offline Store]
  Redis / DynamoDB     Parquet on S3 / Snowflake
  (Low latency read)   (High volume, historical)
       │                    │
       ▼                    ▼
  [Inference Service]  [Model Training Job]
  (Fetches features)   (Point-in-time Joins)
```

**Core ML Component:** The Feature Store itself doesn't train models; it manages the data lifecycle. The critical algorithmic component is the `AS OF` join for offline training data generation.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Feature Definition
- Data scientists write a Python file defining a `FeatureView`.
- This definition includes the data source, the schema, and the TTL (Time-To-Live).

### Step 2: Online Serving (Redis)
- Fast key-value lookups.
- Key format: `feature_view_name:entity_id` (e.g., `user_stats:player_123`).
- Value: Protobuf or JSON containing the feature values.

### Step 3: Offline Training Data Generation
- When a Data Scientist wants to train a model, they provide an "Entity DataFrame" containing `[player_id, event_timestamp, label]`.
- The Feature Store joins the historical features to this dataframe such that `feature_timestamp <= event_timestamp`.

### Step 4: Materialization
- A background job (e.g., Feast materialization) that copies the latest values from the Offline store to the Online store to keep them in sync.

---

## Part 7 — Complete Python Code

*Note: We will mock a simplified version of Feast's core mechanics to demonstrate understanding of the underlying logic.*

```python
"""
feature_store.py - Core logic for Point-in-Time Joins and Online Serving
"""
import logging
import pandas as pd
import redis
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Offline Store Logic (Point-in-Time Join)
# ---------------------------------------------------------------------------
def get_historical_features(
    entity_df: pd.DataFrame, 
    feature_df: pd.DataFrame, 
    entity_keys: list, 
    timestamp_col: str = "event_timestamp"
) -> pd.DataFrame:
    """
    Performs an AS OF join (Point-in-Time join).
    Ensures no future data leakage.
    
    entity_df: The base events (e.g., labels and timestamps)
    feature_df: The historical feature logs
    """
    logger.info("Performing Point-in-Time join...")
    
    # Sort both dataframes by timestamp (required for merge_asof)
    entity_df = entity_df.sort_values(timestamp_col)
    feature_df = feature_df.sort_values(timestamp_col)
    
    # Perform the exact AS OF join
    # For each row in entity_df, find the last row in feature_df where 
    # feature_df.timestamp <= entity_df.timestamp, matching on entity_keys
    joined_df = pd.merge_asof(
        entity_df,
        feature_df,
        on=timestamp_col,
        by=entity_keys,
        direction="backward", # Crucial: never look forward in time
        tolerance=pd.Timedelta("30d") # Features older than 30 days are considered stale/null
    )
    
    return joined_df

# ---------------------------------------------------------------------------
# Online Store Logic (Redis Serving)
# ---------------------------------------------------------------------------
class OnlineFeatureStore:
    def __init__(self, redis_host='localhost', port=6379):
        self.redis = redis.Redis(host=redis_host, port=port, decode_responses=True)
        
    def push_features(self, feature_view: str, df: pd.DataFrame, entity_key: str):
        """Materializes latest features into Redis."""
        pipeline = self.redis.pipeline()
        
        # In a real system, you'd only push the LATEST timestamp for each entity
        latest_df = df.sort_values('event_timestamp').groupby(entity_key).last().reset_index()
        
        for _, row in latest_df.iterrows():
            entity_id = row[entity_key]
            redis_key = f"{feature_view}:{entity_id}"
            
            # Drop the entity key and timestamp from the stored payload to save space
            payload = row.drop([entity_key, 'event_timestamp']).to_dict()
            pipeline.set(redis_key, json.dumps(payload))
            
        pipeline.execute()
        logger.info(f"Pushed {len(latest_df)} entities to Online Store.")
        
    def get_online_features(self, feature_view: str, entity_ids: list) -> dict:
        """Fetch features in real-time (<10ms)."""
        keys = [f"{feature_view}:{eid}" for eid in entity_ids]
        
        # MGET is O(N) where N is number of keys, very fast
        raw_values = self.redis.mget(keys)
        
        results = {}
        for eid, raw_val in zip(entity_ids, raw_values):
            if raw_val:
                results[eid] = json.loads(raw_val)
            else:
                results[eid] = None # Cache miss
                
        return results

# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Mock Data
    events = pd.DataFrame({
        "player_id": [1, 1, 2],
        "event_timestamp": pd.to_datetime(["2023-01-05", "2023-01-10", "2023-01-08"]),
        "is_fraud": [0, 1, 0]
    })
    
    historical_features = pd.DataFrame({
        "player_id": [1, 1, 2, 2],
        "event_timestamp": pd.to_datetime(["2023-01-01", "2023-01-09", "2023-01-01", "2023-01-09"]),
        "total_spend": [10.5, 50.0, 5.0, 5.0]
    })
    
    # 2. Offline Training Data Generation
    training_data = get_historical_features(events, historical_features, entity_keys=["player_id"])
    print("--- Training Data (Point in Time) ---")
    print(training_data)
    # Notice that for player 1 on 2023-01-05, total_spend is 10.5 (from Jan 1), NOT 50.0 (from Jan 9). No data leakage.
    
    # 3. Online Serving
    # store = OnlineFeatureStore()
    # store.push_features("player_stats", historical_features, "player_id")
    # print(store.get_online_features("player_stats", [1, 2, 3]))
```

---

## Part 8 — Deployment

### Infrastructure
- **Redis Cluster:** For the Online Store. Requires highly available replication, as any downtime takes down all dependent ML inference services.
- **Airflow:** Schedules the Materialization jobs. E.g., `feast materialize` runs every 15 minutes to sync Snowflake -> Redis.
- **Kafka + Flink:** For streaming features (e.g., "kills in last 60 seconds"). Flink calculates the sliding window and writes directly to Redis, bypassing Snowflake for the online path.

---

## Part 9 — Unit Testing

```python
import pandas as pd
from feature_store import get_historical_features

def test_point_in_time_join():
    entity_df = pd.DataFrame({
        "id": [1],
        "event_timestamp": pd.to_datetime(["2023-01-05"])
    })
    
    feature_df = pd.DataFrame({
        "id": [1, 1, 1],
        "event_timestamp": pd.to_datetime(["2023-01-01", "2023-01-04", "2023-01-06"]),
        "feat_val": [10, 40, 60]
    })
    
    joined = get_historical_features(entity_df, feature_df, ["id"])
    
    # Must join with Jan 4 (val=40). Jan 6 is in the future. Jan 1 is too old.
    assert joined.iloc[0]["feat_val"] == 40
```

---

## Part 10 — Integration Testing

- **Data Consistency Test:**
  - Push a known DataFrame to the Offline Store (Parquet/Snowflake).
  - Run the Materialization job.
  - Query the Online Store (Redis).
  - Assert that `Offline_Store.latest() == Online_Store.get()`. This proves there is no training-serving skew.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Write Amplification (Redis)** | Updating Redis every time a player moves in-game is too expensive. We must batch writes or use a streaming engine (Flink) that only emits state changes every 5 seconds. |
| **Offline Join Compute** | `pd.merge_asof` runs in Python memory. For Terabytes of data, this OOMs. We must push the AS OF join down to Snowflake (using `LEFT JOIN ... WHERE f.timestamp <= e.timestamp QUALIFY ROW_NUMBER() = 1`) or use PySpark. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Snowflake vs Spark for Offline Joins | Snowflake is easy for SQL users but point-in-time joins are computationally expensive and can consume warehouse credits quickly. Spark is cheaper at scale but harder to maintain. |
| JSON vs Protobuf in Redis | JSON is readable and easy to debug. Protobuf is 10x smaller in memory and faster to deserialize, which is critical when storing billions of keys in expensive Redis RAM. |

---

## Part 13 — Alternative Approaches

1. **Embedded Databases (RocksDB):** Instead of a central Redis cluster, inference microservices embed RocksDB and read Kafka topics directly. Extremely low latency (microsecond), but state is duplicated across pods.
2. **Hopsworks / Databricks Feature Store:** Buy vs Build. Open-source Feast is great, but managed solutions handle the messy PySpark point-in-time joins for you automatically.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Materialization Job Fails | Online features get stale | If features are stale, the model might make bad predictions. Implement a `feature_timestamp` in the Redis payload. The Inference service checks `now() - feature_timestamp`. If > 24 hours, it triggers a fallback heuristic instead of ML. |
| Data Leakage | Future data sneaks into training | Strict unit testing on the `AS OF` join. Forbid standard `LEFT JOIN` on dates in the data warehouse. |

---

## Part 15 — Debugging

**Symptom:** A model's accuracy drops from 95% in offline training to 60% in live production.

**Debugging steps:**
1. This is the definition of **Training-Serving Skew**.
2. Capture the actual feature payloads served by Redis in production for 1 hour.
3. Compare them to the features generated by the Offline Store for that exact same hour.
4. **Common Culprit:** A categorical feature (e.g., `device_type`) was one-hot encoded as a string (`"IOS"`) in the data warehouse, but the real-time API is sending it as lowercase (`"ios"`). The model drops the feature.
5. **Fix:** Centralize transformation logic. Do not allow the client API to apply ad-hoc transformations.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `redis_mget_latency_ms` | > 10ms → Scale Redis or check network topology |
| `feature_staleness_seconds` | > Materialization Schedule + 1h → Job is failing |
| `online_offline_consistency_score` | Checked daily via sampling. If < 100%, investigate. |

---

## Part 17 — Production Improvements

1. **On-the-fly Transformations:** Support features that require request-time data. E.g., `feature = distance(user_location, server_location)`. `server_location` is only known at inference time. The Feature Store should fetch the historical `user_location` from Redis, and apply the Python `distance` function dynamically before returning.
2. **Feature Monitoring:** Automatically calculate drift (PSI) for every feature in the store daily, alerting data scientists if the underlying data distribution shifts.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"In your `merge_asof` logic, what happens if the feature dataframe has a row from Jan 1st, but the event is on Dec 31st of the same year? Does the model get an 11-month old feature?"**
2. **"We have 10,000 features in the store. An inference service requests 50 features for a player. How do you prevent fetching all 10,000 from Redis?"**
3. **"How does the Flink streaming pipeline update the Offline Store for training? Do we write Kafka data to Snowflake and Redis simultaneously?"**

---

## Part 19 — Ideal Answers

**Q1 (Stale features):**
> "An 11-month old feature is likely useless and introduces noise. We must use the `tolerance` parameter in `merge_asof` (e.g., `tolerance=pd.Timedelta('30d')`). If no feature exists within the last 30 days, the join returns `NULL`, and the downstream model handles it via imputation (e.g., XGBoost handles NaNs natively)."

**Q2 (Feature Selection in Redis):**
> "We organize Redis keys by `FeatureView`, which represents a logical group of features (e.g., `player_daily_stats`). The inference service only queries the specific FeatureViews it needs. If a View contains 100 features but we only need 50, we still fetch the whole JSON/Protobuf payload for that View, as fetching is fast, but we parse out the 50 we need. We shouldn't store 10,000 individual keys per player."

**Q3 (Lambda Architecture / Streaming sync):**
> "Writing to Snowflake and Redis simultaneously from Flink is an anti-pattern (two-phase commit problem). Instead, we use a Lambda or Kappa architecture. Flink writes the streaming aggregations to Redis (Online). Concurrently, the raw Kafka events are dumped to S3 (Data Lake) via Kafka Connect. A nightly Airflow job runs a batch aggregation on S3 and loads it into Snowflake (Offline). We accept that the Offline store is delayed by 24h for training purposes."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Understands Point-in-Time (AS OF) joins natively.
- Clearly articulates the architectural split between Online (Redis) and Offline (Snowflake/Parquet).
- Answers the Data Leakage and Training-Serving Skew questions definitively.
- Explains the Lambda architecture for syncing streaming data to the offline store.

### Hire
- Basic understanding of Feature Stores.
- Implements the Redis lookup correctly.
- Understands `merge_asof` with some prompting.
- Code is clean and handles missing data.

### Lean Hire
- Misses the point-in-time join entirely, performing standard SQL `LEFT JOIN` which causes massive data leakage.
- Fixes it when the interviewer points out the time-travel bug.

### Lean No Hire
- Thinks a Feature Store is just a Postgres database holding static data.
- Doesn't understand the difference between training data generation and real-time inference serving.

### No Hire
- Fails to write any data manipulation code (Pandas/SQL).
- Conceptually lost on the purpose of the problem.
