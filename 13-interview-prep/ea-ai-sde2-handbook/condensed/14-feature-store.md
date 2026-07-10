# Interview 14 — Real-Time Feature Store (Condensed)

Multiple game teams need to join batch features ("lifetime spend") with streaming features ("items bought in last 5 min") at inference time, in <10ms, with training/serving consistency. Design a real-time feature store.

## Clarifying Questions to Ask
- What's the read latency SLA? → p99 < 10ms.
- How do we prevent training-serving skew? → Feature defined once, computed identically for online + offline stores.
- How do we do point-in-time joins for training data? → Must join features as of `timestamp=T`, no leakage from `T+1`.
- Build from scratch or use OSS? → Base on Feast-like standard, but explain underlying mechanics.
- What are read/write QPS constraints? → Drives Redis cluster sizing / batching decisions.
- Data sources? → Kafka (streaming) + Snowflake/S3 (batch).

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

## Toughest Follow-ups
**Q: Feature dataframe has a Jan 1 row but the event is Dec 31 same year — does the model get an 11-month-old feature?**
A: Yes, unless bounded — that's why `merge_asof` needs a `tolerance` (e.g. 30d). Beyond tolerance the join returns NULL and the model imputes (tree models handle NaN natively); an unbounded AS-OF join silently feeding stale features is a correctness bug.

**Q: 10,000 features exist in the store; inference needs 50 — how do you avoid fetching all 10,000 from Redis?**
A: Organize keys by FeatureView (logical groups, e.g. `player_daily_stats`), and inference only queries the FeatureViews it needs. Within a view it's fine to fetch the whole blob and parse the needed subset — fetching is cheap, it's the number of round trips/keys that matters, so never store one key per feature.

**Q: How does streaming (Flink) get reflected in the offline store used for training — write to Snowflake and Redis at once?**
A: No — that's a two-phase-commit anti-pattern. Use Lambda/Kappa: Flink writes only to Redis (online); raw Kafka events land in S3 via Kafka Connect; a nightly Airflow batch job aggregates S3 into Snowflake. Accept ~24h offline staleness as the cost of avoiding dual-write consistency issues.

## Biggest Pitfall
Performing a standard `LEFT JOIN` on timestamps instead of a true point-in-time AS-OF join — this leaks future data into training and silently inflates offline accuracy versus production, the single fastest way to go from Hire to No Hire on this problem.
