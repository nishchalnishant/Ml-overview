# Interview 14 — Real-Time Feature Store (Condensed)

Multiple game teams need to join batch features ("lifetime spend") with streaming features ("items bought in last 5 min") at inference time, in <10ms, with training/serving consistency. Design a real-time feature store.

## Core Architecture
```
Kafka → Flink (streaming agg) ─┐
Snowflake → Airflow (batch agg)─┼─→ Feature Registry (Feast-style defs)
                                 │
                    ┌────────────┴────────────┐
              Online Store                Offline Store
              (Redis/DynamoDB)             (Parquet/Snowflake)
                    │                            │
             Inference Service            Training Job
             (<10ms fetch)                (point-in-time AS OF join)
```
- Dual-store split: Online (Redis, low-latency KV) vs Offline (Snowflake/S3, historical).
- Feature Registry: single definition source, avoids skew — the key ML-adjacent technique here.
- Materialization job (e.g. `feast materialize` every 15 min) syncs offline → online.
- Streaming path (Flink) writes directly to Redis, bypassing warehouse for freshness.
- Core algorithm: `merge_asof`/AS-OF join (`direction=backward`, with `tolerance`) — prevents label leakage.
- Redis key = `feature_view:entity_id`, value = JSON/Protobuf blob of grouped features.

## Talking Points That Signal Seniority
- Proactively names "training-serving skew" and "point-in-time correctness" before being asked.
- States AS-OF join must use `tolerance` window — stale features beyond N days should return NULL, not a stale value.
- Distinguishes Lambda vs two-phase-commit: never write Kafka → Snowflake and Redis simultaneously; use Kafka Connect → S3 → nightly batch job to Snowflake instead.
- Flags write amplification risk in Redis (e.g., per-tick game state) and proposes Flink emitting only on windowed state changes.
- Mentions pushing AS-OF joins down into Snowflake/Spark for TB-scale data instead of `pd.merge_asof` in memory (OOM risk).
- Proposes feature freshness check at serving time (`now() - feature_timestamp`) with fallback heuristic if stale.
- Raises on-the-fly / request-time transformations (e.g., distance from user to server) as a production gap in naive designs.
- Suggests daily PSI-based feature drift monitoring, not just infra metrics.

## Top 3 Tradeoffs
- Snowflake vs Spark for offline joins — Snowflake is SQL-friendly but AS-OF joins burn warehouse credits fast; Spark scales cheaper but costs more engineering effort.
- JSON vs Protobuf in Redis — JSON is debuggable; Protobuf is ~10x smaller/faster, matters at billions of keys where Redis RAM is the expensive resource.
- Central Redis cluster vs embedded RocksDB per inference pod — RocksDB gives microsecond reads but duplicates state across pods and complicates consistency.

## Biggest Pitfall
Performing a standard `LEFT JOIN` on timestamps instead of a true point-in-time AS-OF join — this leaks future data into training and silently inflates offline accuracy versus production, the single fastest way to go from Hire to No Hire on this problem.
