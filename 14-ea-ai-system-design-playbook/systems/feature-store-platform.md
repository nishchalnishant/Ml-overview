# Feature Store Platform

## 1. Problem Framing & Requirement Gathering

EA runs dozens of live-service titles (FC 25, Apex Legends, Battlefield, The Sims, madden franchise, mobile titles) that each need ML features for: matchmaking skill estimation, churn prediction, live-ops offer personalization, toxicity/cheat detection, LTV prediction, and recommendation (bundles, cosmetics). Each of these consumes overlapping raw signals (session counts, purchase history, kill/death ratios, squad composition) but today every team recomputes these features independently in bespoke Spark jobs and duplicated online caches — leading to:

- **Training/serving skew**: offline Spark pipeline computes `avg_kd_7d` with a different window boundary than the online Flink job.
- **Duplicated compute**: 6 teams independently join the same telemetry event stream.
- **No point-in-time correctness**: offline joins leak future information into training sets (label leakage from post-event telemetry).
- **No lineage/versioning**: a feature definition changes silently, and nobody can tell which model was trained on which feature version.

The ask: build a **centralized Feature Store Platform** — a single system of record for feature definitions, offline (batch, training) and online (low-latency, serving) feature values, with strict point-in-time (PIT) correctness, versioning, and skew prevention — usable by all EA studios as a shared platform (multi-tenant).

**Primary interview framing**: this is a *platform* system design question (infra-for-ML), not a single-model-serving question. The interviewer wants to see: dual-path (batch+stream) data modeling, PIT-correct joins, schema/version governance, and consistency guarantees between online/offline paths.

## 2. Functional Requirements

- FR1: Feature producers (data/ML engineers) can **register feature definitions** (name, entity, dtype, transformation, owner, TTL) via a declarative spec (YAML/Python DSL), versioned in git.
- FR2: Platform **computes features in batch** (offline store) from data lake sources (telemetry, purchases, match results) on a schedule or trigger.
- FR3: Platform **computes/streams features in near-real-time** (online store) from Kafka telemetry topics for low-latency serving.
- FR4: Given an **entity key + timestamp**, offline store returns the feature value that was true **as of that timestamp** (point-in-time correctness) — no future leakage.
- FR5: Given an **entity key** at request time, online store returns the **latest** feature value within an SLA of single-digit milliseconds.
- FR6: **Training set generation**: given a list of (entity, label, event_timestamp) rows, join point-in-time-correct feature values across N feature groups ("get_historical_features").
- FR7: **Online retrieval API**: given entity ids + feature list, return current values ("get_online_features") for real-time inference.
- FR8: Support **feature versioning** — multiple versions of a feature definition can coexist; models pin to a version.
- FR9: Detect and prevent **training/serving skew** — same transformation code path for batch and streaming (shared feature logic), skew-monitoring job that diffs offline vs online values for sampled entities.
- FR10: **Feature discovery** — searchable catalog/UI, lineage graph (raw source → transformation → feature → model).
- FR11: **Backfill** — recompute historical feature values when a definition changes or a bug is fixed.
- FR12: **Access control** — per-team namespaces, PII-tagged features require elevated approval.

## 3. Non-Functional Requirements (latency, availability, throughput, consistency, cost)

| Dimension | Target |
|---|---|
| Online read latency | p50 ≤ 3 ms, p99 ≤ 10 ms, p99.9 ≤ 25 ms (single entity, ~30 features) |
| Online read throughput | 2M reads/sec peak platform-wide across titles |
| Online write (streaming ingest) | 500K events/sec sustained, 1.2M/sec peak (live-ops launch spikes) |
| Availability (online store) | 99.99% (≈ 52 min/year downtime) — matchmaking/anti-cheat is latency- and availability-critical |
| Availability (offline store / batch) | 99.9% — batch is retry-tolerant |
| Consistency | Online: eventually consistent (bounded staleness ≤ 60s from event to serving availability). Offline: point-in-time correct = strongly correct w.r.t. event-time, not wall-clock. |
| Durability | Offline store (feature history) durability 99.999999999% (11 nines, object-store backed) |
| Cost ceiling | Feature store infra ≤ 8% of total ML platform spend |
| Freshness | Streaming features: ≤60s lag p99. Batch daily features: ≤ 4h after source data lands. |

## 4. Clarifying Questions an interviewer would expect you to ask

1. Is this single-title or **multi-tenant across studios** (affects namespace/isolation design)?
2. What entity types are we keying on — player_id only, or also match_id, squad_id, item_id (multi-entity joins)?
3. What's the acceptable **staleness bound** for online features — is 30s okay for churn scoring but too slow for anti-cheat?
4. Do we need **streaming aggregations** (rolling windows, e.g. "kills in last 10 min") or only point lookups of precomputed values?
5. Who owns **PIT-correctness enforcement** — is it a platform guarantee, or can teams opt out for speed?
6. What's the **write path for backfills** — do they go through the same pipeline as live ingestion, or a separate batch overwrite path?
7. Do downstream **models require feature vectors as of training label time**, implying we need an event-time index, not just ingestion-time?
8. Is there a need for **cross-title feature sharing** (e.g., a "player is a whale" feature reused across FC and Apex) — implies a shared entity resolution layer?
9. What existing infra can we build on (Kafka footprint, Spark/Databricks, existing key-value stores) vs. greenfield?
10. What's the **budget for online store replicas** — do we need multi-region for latency, or is single-region with edge caching sufficient?

## 5. Assumptions (explicit, numbered)

1. ~55M MAU across EA's top 6 live-service titles combined; average concurrent players at peak ~2.5M.
2. Each active player session triggers ~40 telemetry events/min (kills, purchases, match end, session heartbeat).
3. ~1,200 distinct feature definitions across ~150 feature groups, owned by ~12 ML teams.
4. Average inference request touches ~25–40 features across 3–5 feature groups.
5. Online store must serve matchmaking (called every ~90s per active match, ~2.5M concurrent → ~28K QPS baseline from matchmaking alone) plus real-time personalization (every page/screen transition, ~10x that).
6. Offline store retains 2 years of historical feature values for backtesting/compliance.
7. Feature transformation code is written once (Python/SQL DSL) and compiled to both a Spark batch job and a Flink streaming job (shared logic via a common IR) — this is the core skew-prevention mechanism.
8. Existing EA data lake is on S3/Parquet with a Delta Lake-style transactional layer already in place (reuse, don't replace).
9. Kafka is the existing telemetry backbone (per earlier chapters), ~150 topics, retained 7 days hot.
10. Teams accept **eventual consistency** on the online path in exchange for the stated 60s staleness bound.

## 6. Capacity Estimation (QPS, storage, model size, GPU/CPU counts, back-of-envelope math shown)

**Online read QPS**
- Matchmaking: 2.5M concurrent players / avg match search cycle 90s → ~28K QPS baseline.
- Live-ops personalization (store screens, offer surfacing): ~2.5M concurrent × 1 lookup / 45s avg session interaction → ~55K QPS.
- Anti-cheat / toxicity real-time scoring: sampled at ~15% of match-events, 40 events/min/player × 2.5M × 0.15 / 60 → ~250K QPS (dominant load).
- Total sustained online read: ~330K QPS; provision for 3x peak burst (tournament/launch day) → **1M QPS design target**, matches stated 2M ceiling with headroom.

**Online write (streaming feature computation) QPS**
- 2.5M concurrent × 40 events/min / 60 = **1.67M raw events/sec** ingested into Kafka.
- After filtering to feature-relevant events (~30%) and windowed aggregation fan-in, feature-store write QPS ≈ **500K/sec** sustained, consistent with NFR.

**Storage — online store**
- Entities: ~55M player profiles + ~5M active match/session objects.
- Per-entity feature payload: 40 features × avg 16 bytes (numeric/short string) + overhead ≈ 1.2 KB/entity.
- Online KV footprint: 60M entities × 1.2 KB ≈ **72 GB** raw; with replication factor 3 → **216 GB**. Trivially fits in a memory-tier cluster (fits in RAM across a modest Redis/DynamoDB DAX cluster).

**Storage — offline store (feature history)**
- 1,200 features × 55M distinct players × daily snapshot for churn/LTV features (subset, say 300 features are daily-snapshotted) + event-level streaming features logged at event-granularity.
- Daily snapshot table: 55M rows × 300 features × 8 bytes avg (Parquet columnar, compressed ~4x) ≈ 55M × 2.4KB / 4 ≈ **33 GB/day** compressed.
- 2-year retention: 33GB × 730 ≈ **24 TB** for snapshot tables.
- Event-level streaming feature log (for PIT joins on high-frequency features): ~500K writes/sec × 200 bytes avg × 86,400s × compression(1/5) ≈ **1.7 TB/day** → 2-year retention with tiering (hot 30 days, cold/Glacier beyond) ≈ 51 TB hot + ~1.2 PB cold archive.
- **Total offline footprint ≈ 1.3 PB**, dominated by cold-tier event logs — justifies aggressive lifecycle policies (compact to daily/hourly rollups after 30 days, drop raw event grain).

**Compute**
- Streaming (Flink) cluster: 500K events/sec sustained aggregation, assume 1 vCPU handles ~8K simple windowed-agg events/sec → **~65 vCPUs** minimum, provision 4x for stateful window overhead + backpressure headroom → **~260 vCPUs** (≈ 33 x 8-vCPU task managers).
- Batch (Spark) cluster: nightly full recompute of 300 daily features over 55M entities joined against ~2TB/day of new telemetry — a 200-executor (4 vCPU/16GB each) cluster completes in ~45 min nightly window.
- Online KV serving cluster: 1M QPS at ~3ms p50 → with a node handling ~40K QPS (Redis Cluster shard, single-threaded core saturation), need **~25 shards** minimum; provision 3x for burst/failover → **~75 shards**, 3-way replicated → ~225 nodes total (small instance class, e.g. 4vCPU/32GB memory-optimized).
- No GPU requirement in the feature store itself — GPUs live in the downstream model-serving chapter; feature store is CPU/memory-bound infra.

## 7. High-Level Architecture (with an ASCII diagram)

```
                         ┌───────────────────────────────────────────┐
                         │           Feature Definition Registry       │
                         │  (git-backed YAML/Python DSL, versioned,     │
                         │   compiled to shared IR -> Spark + Flink)    │
                         └───────────────┬───────────────────────────┘
                                         │ compiled feature specs
                 ┌───────────────────────┼─────────────────────────────┐
                 ▼                                                     ▼
     ┌───────────────────────┐                              ┌───────────────────────┐
     │   Batch Compute (Spark) │                              │ Streaming Compute (Flink)│
     │  reads Delta/Parquet     │                              │ reads Kafka telemetry     │
     │  lake, nightly/hourly    │                              │ topics, windowed aggs     │
     └───────────┬─────────────┘                              └───────────┬─────────────┘
                 │ writes                                                  │ writes
                 ▼                                                         ▼
     ┌───────────────────────┐                              ┌───────────────────────┐
     │   OFFLINE STORE         │                              │   ONLINE STORE          │
     │  (Delta/Parquet on S3,   │◄──────── dual-write ───────►│ (Redis Cluster / DynamoDB│
     │  event-time indexed,     │   consistency/skew monitor   │  DAX, key = entity_id)  │
     │  point-in-time API)      │                              │  key-value, ~3-10ms p99 │
     └───────────┬─────────────┘                              └───────────┬─────────────┘
                 │                                                         │
                 ▼                                                         ▼
     ┌───────────────────────┐                              ┌───────────────────────┐
     │ Training Set Generator  │                              │  Online Feature API     │
     │ get_historical_features │                              │  get_online_features    │
     │ (PIT join engine)        │                              │  (gRPC/REST, <10ms p99)│
     └───────────┬─────────────┘                              └───────────┬─────────────┘
                 ▼                                                         ▼
        ┌──────────────────┐                                    ┌──────────────────────┐
        │ Training Pipelines │                                    │ Online Inference /      │
        │ (model training)   │                                    │ Matchmaking / Live-Ops  │
        └──────────────────┘                                    └──────────────────────┘

     Cross-cutting: Feature Catalog & Lineage UI | Skew Monitor | Access Control | Metadata Store (Postgres)
```

Data flow summary:
1. Feature authors define a transformation once → compiled to both Spark (batch) and Flink (streaming) execution plans from a shared intermediate representation (IR) — this is the crux of skew prevention.
2. Batch path recomputes/backfills the offline store from the data lake on schedule.
3. Streaming path continuously updates the online store from Kafka in near-real-time.
4. A **skew monitor** periodically samples entities, computes the same feature via both paths, and diffs — alerting on divergence.
5. Training pipelines pull point-in-time-correct historical feature values; serving pipelines pull latest online values — using the *same feature registry/version* to guarantee identical semantics.

## 8. Low-Level Components (responsibility, interface, scaling unit)

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| **Feature Registry Service** | Store/version feature definitions, compile DSL → IR, expose catalog/lineage | gRPC + REST, git-webhook triggered | Stateless; scales with registry read QPS (low) |
| **Batch Compute Engine (Spark)** | Execute batch feature transforms, backfills, nightly snapshots | Spark job submitted via Airflow DAG | Scale by executor count per job (data volume) |
| **Streaming Compute Engine (Flink)** | Execute windowed/stateful streaming transforms from Kafka | Flink jobs, keyed by entity_id | Scale by parallelism (Kafka partition count) |
| **Offline Store** | Durable, PIT-indexed feature history | Delta Lake tables on S3, queried via Spark/Trino | Scale via partitioning (date + entity hash) |
| **Online Store** | Low-latency KV lookups of latest feature values | Redis Cluster (hash-tagged) / DynamoDB | Scale via shard count (hash slots) |
| **PIT Join Engine (Historical Retrieval)** | Given entity+event_timestamp list, produce leakage-free training set | Spark job, `get_historical_features(entity_df, feature_refs)` | Scale by executor count, driven by training-set size |
| **Online Feature API Gateway** | Serve `get_online_features` with SLA, fan out to shards, feature transform-at-read (on-demand features) | gRPC (primary), REST (fallback) | Stateless, horizontal pod autoscale on QPS |
| **Skew Monitor** | Sample entities, compare batch vs streaming computed values, emit divergence metrics | Scheduled job + metrics pipeline | Scales with sample rate (fixed small budget) |
| **Metadata Store** | Feature versions, ownership, TTL, access policy, lineage edges | Postgres (small, relational) | Vertical scale sufficient; read replicas for catalog UI |
| **Access Control Service** | Namespace isolation, PII tagging enforcement, per-team RBAC | Integrated into gateway as middleware | Stateless, scales with gateway |

## 9. API Design (concrete endpoint signatures, request/response schemas, versioning)

**Versioning strategy**: URI-versioned (`/v1/...`), feature-definition versioning is independent and referenced by `feature_view:version` string (e.g. `player_churn_features:v3`) so API surface stability is decoupled from feature schema evolution.

### 9.1 Online retrieval

```
POST /v1/online-features:get
Request:
{
  "entities": [{"player_id": "p_9182734"}, {"player_id": "p_2231098"}],
  "feature_refs": [
    "player_activity_features:v2:sessions_7d",
    "player_activity_features:v2:avg_kd_7d",
    "purchase_features:v5:whale_score"
  ],
  "consumer": "matchmaking-service"     // for auth + rate limiting
}

Response: 200 OK
{
  "results": [
    {
      "player_id": "p_9182734",
      "features": {
        "player_activity_features:v2:sessions_7d": 14,
        "player_activity_features:v2:avg_kd_7d": 1.82,
        "purchase_features:v5:whale_score": 0.31
      },
      "as_of": "2026-07-09T04:12:31.204Z",
      "staleness_ms": 8213
    }
  ]
}
```

### 9.2 Historical / training set retrieval (point-in-time)

```
POST /v1/historical-features:get
Request:
{
  "entity_df_uri": "s3://ea-ml/labels/churn_train_2026_06.parquet",  // cols: player_id, event_timestamp, label
  "feature_refs": ["player_activity_features:v2:*", "purchase_features:v5:whale_score"],
  "output_uri": "s3://ea-ml/training-sets/churn_v14.parquet"
}

Response: 202 Accepted
{ "job_id": "hist-job-88213", "status_url": "/v1/jobs/hist-job-88213" }
```

### 9.3 Feature registration

```
PUT /v1/feature-views/{name}
Request:
{
  "name": "purchase_features",
  "version": "v5",
  "entity": "player_id",
  "owner": "live-ops-ml-team",
  "ttl_seconds": 172800,
  "features": [
    {"name": "whale_score", "dtype": "float32", "transform": "sql://transforms/whale_score.sql"}
  ],
  "pii": false
}
Response: 201 Created { "feature_view": "purchase_features:v5", "compiled": true }
```

### 9.4 Catalog / lineage

```
GET /v1/feature-views?owner=live-ops-ml-team&search=whale
GET /v1/feature-views/{name}/{version}/lineage
```

| Endpoint | Method | Latency SLA | Auth |
|---|---|---|---|
| `/v1/online-features:get` | POST | p99 ≤ 10ms | mTLS service-to-service |
| `/v1/historical-features:get` | POST | async, job-based | OAuth2 client-credentials |
| `/v1/feature-views/{name}` | PUT/GET | p99 ≤ 200ms | OAuth2 + RBAC (write requires owner role) |
| `/v1/feature-views/{name}/{version}/lineage` | GET | p99 ≤ 300ms | OAuth2 |

## 10. Database Design (schema sketches, SQL/NoSQL/columnar choice, partitioning/sharding key)

**Online store — Redis Cluster (primary) with DynamoDB as durable fallback tier**
- Why KV over relational: sub-10ms point lookups by entity key at 1M+ QPS; no ad-hoc query needs at serving time.
- Key: `{feature_view}:{version}:{entity_id}` → hash of field→value (Redis HASH), TTL set per feature_view.
- Sharding key: `entity_id` (hashed into 16384 Redis hash slots), ensures uniform distribution across live players.

```
HSET fv:purchase_features:v5:p_9182734 whale_score 0.31 last_purchase_days 4
EXPIRE fv:purchase_features:v5:p_9182734 172800
```

**Offline store — Delta Lake (Parquet on S3), columnar**
- Why columnar: analytical scans over billions of historical rows for PIT joins and backfills; compression + column pruning critical at 1.3PB scale.
- Partitioning: `feature_view / dt (event date) / entity_id_bucket (hash mod 256)` — enables date-range pruning for PIT joins and avoids small-file problem.

```sql
CREATE TABLE offline_features.purchase_features_v5 (
  player_id       STRING,
  event_timestamp TIMESTAMP,   -- when the feature became true (event-time)
  ingestion_timestamp TIMESTAMP, -- when it landed (for audit/debug only)
  whale_score     FLOAT,
  feature_version STRING
)
USING DELTA
PARTITIONED BY (dt, entity_bucket);
```

**Metadata store — Postgres (relational)**
- Why relational: strongly-consistent, low-volume, relationship-heavy (owners, lineage edges, access policies) — a graph-ish schema but small enough that Postgres + recursive CTEs suffice; no need for a dedicated graph DB at this scale.

```sql
CREATE TABLE feature_views (
  name VARCHAR, version VARCHAR, entity VARCHAR, owner VARCHAR,
  ttl_seconds INT, pii BOOLEAN, created_at TIMESTAMPTZ,
  PRIMARY KEY (name, version)
);
CREATE TABLE lineage_edges (
  src_feature VARCHAR, dst_model VARCHAR, created_at TIMESTAMPTZ
);
```

## 11. Caching (what's cached, invalidation, cache-aside vs write-through)

- **Online store itself is a cache-like tier**: Redis with TTL matching feature freshness contract (write-through from Flink — streaming compute writes directly to Redis on every window emit, not lazily populated).
- **Gateway-local L1 cache**: 200ms micro-cache (in-process LRU, ~50MB per gateway pod) for extremely hot entities (e.g., top-viewed player profiles during a tournament) to absorb read amplification — cache-aside, invalidated purely by short TTL (no active invalidation needed given 200ms window).
- **Feature registry cache**: feature-view metadata (schema, TTL, owner) cached in gateway pods for 60s, invalidated via a pub/sub "registry changed" event on registration/update (write-through invalidation, not just TTL expiry, since schema changes must propagate fast to avoid serving stale-shape data).
- **Historical-join result cache**: repeated identical `get_historical_features` requests (same entity_df + feature set hash) are memoized in the offline store's result cache (S3-backed, keyed by content hash) for 24h to avoid recomputation during iterative model development.
- Explicitly **not cached**: PIT join intermediate state (too large, changes per request) and raw Kafka streams (already durable in Kafka itself for 7 days).

## 12. Queues & Async Processing (at-least-once vs exactly-once, DLQ)

- **Ingestion queue**: Kafka topics per telemetry domain (match_events, purchase_events, session_events) — **at-least-once** delivery; consumers (Flink jobs) are idempotent via entity_id + event_id dedup keys stored in Flink keyed state (effectively exactly-once feature *values*, even though transport is at-least-once).
- **Backfill job queue**: Airflow-triggered Spark jobs queued via a job-scheduler queue (SQS-backed); **at-least-once** with idempotent overwrite semantics (backfills always fully overwrite a partition, so re-running is safe).
- **Historical retrieval jobs**: async job queue (`/v1/historical-features:get` returns 202 + job_id) — backed by a job queue table + worker pool; retried up to 3x on transient Spark cluster failure.
- **Dead-letter handling**: malformed telemetry events (schema validation failure at Flink ingestion) routed to a `*.dlq` Kafka topic; DLQ consumer alerts owning team and stores samples in S3 for debugging. DLQ rate > 0.1% of topic volume triggers a P2 alert (see Alerting section).
- **Exactly-once semantics** are achieved end-to-end for streaming feature *writes* to the online store via: Flink's exactly-once checkpointing + idempotent upserts keyed by (entity_id, feature_view, window_end_timestamp) — replays converge to the same final value, not duplicated increments.

## 13. Streaming & Event-Driven Architecture (topics, event schemas, consumer groups)

**Kafka topics (subset relevant to feature store)**

| Topic | Partitions | Retention | Schema (Avro) |
|---|---|---|---|
| `telemetry.match_events` | 512 | 7 days | `{match_id, player_id, event_type, ts, payload}` |
| `telemetry.purchase_events` | 128 | 14 days | `{player_id, sku, amount_usd, ts, currency}` |
| `telemetry.session_events` | 256 | 7 days | `{player_id, session_id, event_type, ts}` |
| `feature_store.registry_changes` | 8 | 30 days | `{feature_view, version, change_type, ts}` (pub/sub for cache invalidation) |
| `feature_store.skew_alerts` | 4 | 30 days | `{feature_ref, entity_sample, offline_val, online_val, delta}` |

**Consumer groups**
- `flink-feature-compute-{feature_group}` — one consumer group per streaming feature group, parallelism = partition count, keyed by `player_id` to co-locate windowed state.
- `skew-monitor-sampler` — low-parallelism consumer sampling ~0.01% of events for cross-path comparison.
- `dlq-alert-consumer` — watches all `*.dlq` topics, low volume.

**Event schema evolution**: Avro with Schema Registry, **backward-compatible only** enforced (new optional fields with defaults) — a breaking schema change requires a new topic version (`telemetry.match_events.v2`), preventing silent consumer breakage across 12 teams' Flink jobs.

## 14. Model Serving

N/A as a *primary* concern of this chapter — the feature store **feeds** model serving (covered in the dedicated Model Serving / Inference Platform chapter) rather than hosting models itself. Relevant boundary interactions:

- Online Feature API is called **synchronously, in the critical path** of model inference — feature retrieval latency budget (10ms p99) must be a small fraction of total inference SLA (typically 50-100ms end-to-end for matchmaking/personalization models).
- The feature store supports **on-demand transforms** (request-time feature computation from raw request payload, e.g. combining request-time context with precomputed features) — executed inside the Online Feature API gateway process (lightweight Python/WASM sandboxed UDFs), not a separate model-serving framework.
- No GPU usage in this system; serving framework choice (Triton/TorchServe) is out of scope here.

## 15. Feature Store (online/offline split, point-in-time correctness)

This *is* the system — detailed treatment:

**Online/offline split rationale**
- Offline store optimized for **large-scale, historical, analytical** access (training set generation, backfills) — columnar, batch-oriented, eventually-consistent-is-fine because it's used for training, not live decisions.
- Online store optimized for **single-key, low-latency, latest-value** access — row/KV-oriented, must be fast and "good enough" fresh, not necessarily perfectly consistent with offline at every instant.

**Point-in-time (PIT) correctness — the core hard problem**
- Every feature value in the offline store is stored with an **event_timestamp** (when it became true) distinct from **ingestion_timestamp** (when it was written) — critical because pipelines run late (e.g., yesterday's snapshot job runs at 2am today).
- `get_historical_features(entity_df, feature_refs)`: for each row `(entity_id, label_event_timestamp)` in the entity dataframe, join to the **most recent feature value where `feature.event_timestamp <= label_event_timestamp`**, per feature_view, per entity — implemented as a point-in-time "as-of" join (Spark: sort-merge join with a watermark filter, not a naive equi-join), **never a value that was computed only after the label event**.
- Example leakage bug this prevents: training a churn model on `sessions_7d` computed *including* the day the player actually churned would leak the label into the feature — PIT join clips the feature window to strictly precede the label timestamp.
- **TTL/staleness semantics differ deliberately**: offline PIT join has no "staleness" concept (it's exact as of a timestamp); online store has an explicit staleness bound (60s) because it always serves "latest known" value, not "value as of now" in a mathematically exact sense — this asymmetry must be called out explicitly in an interview, it's the single most commonly glossed-over subtlety.

**Feature versioning**
- Every feature definition is `(feature_view_name, version)` — a model manifest pins exact versions used at training time (`purchase_features:v5`); if the definition logic changes, a new version (`v6`) is created rather than mutating v5 in place, guaranteeing reproducibility of any historical training run.
- Old versions retained per a deprecation policy (default 180 days) unless a model still references them (registry tracks active consumers via lineage edges, blocking deletion of in-use versions).

**Training/serving skew prevention mechanism**
- Single feature transformation authored once in a DSL (Python/SQL subset) → compiled to a shared IR → code-generated into both a Spark batch UDF and a Flink streaming operator. This guarantees **identical arithmetic/business logic** in both paths (not just "we tried to keep them in sync manually").
- Continuous **skew monitor**: samples ~0.01% of entities every 5 min, computes the feature via both the batch path (using latest offline snapshot) and reads the online value, diffs, and emits `feature_store.skew_alerts`. Threshold: mean relative delta > 2% over a rolling 1h window on any feature_view triggers a P2 alert.

## 16. Vector Database

**N/A** — this platform serves scalar/tabular numeric and categorical features for tabular models (churn, LTV, matchmaking ELO, anti-cheat classifiers), not embedding similarity search. Semantic/embedding retrieval (e.g., cosmetic recommendation via item embeddings) is explicitly out of scope and belongs to the separate RAG/Recommendation platform chapter, which owns a vector DB (e.g., pgvector/Milvus). The feature store may *store* a precomputed embedding as an opaque feature value (e.g., `player_embedding_v3: float[64]`) for a downstream model to consume, but does not perform ANN search itself.

## 17. Embedding Pipelines

**N/A for this chapter's primary scope**, with one caveat: the feature store *can* host the batch pipeline that **produces** entity embeddings (e.g., a nightly Spark job running a trained player-behavior embedding model over telemetry to produce a 64-dim vector feature), treating the embedding as just another versioned feature (`player_behavior_embedding:v2`). The training of the embedding model itself and any ANN indexing is out of scope — owned by the Recommendation/Vector-Search chapter.

## 18. Inference Pipelines (request lifecycle end-to-end)

```
Client (matchmaking service) request
        │
        ▼
Online Feature API Gateway (gRPC)
        │  1. AuthN/AuthZ check (mTLS + RBAC)
        │  2. Rate limit check (per-consumer token bucket)
        │  3. Resolve feature_refs -> shard routing (consistent hash on entity_id)
        ▼
   ┌─────────────┬─────────────┬─────────────┐
   │ Redis shard  │ Redis shard  │ Redis shard  │   (parallel fan-out, ~1-3ms each)
   │  (fv A)      │  (fv B)      │  (fv C)      │
   └─────┬───────┴──────┬──────┴──────┬──────┘
         ▼              ▼             ▼
   Gateway merges results, applies on-demand transform UDFs (request-time context)
        │
        ▼
   Response assembled (feature vector + staleness metadata)
        │
        ▼
   Client passes feature vector -> Model Serving (Triton/TorchServe) -> prediction
        │
        ▼
   Matchmaking decision / live-ops offer shown to player
```

**Latency budget** (p99, 10ms feature-fetch SLA nested inside a larger ~80ms total inference SLA):
- AuthN/rate-limit: 0.5ms
- Shard routing + fan-out: 0.5ms
- Redis parallel reads (bottleneck): 3-6ms
- On-demand UDF transform: 1-2ms
- Serialization/network: 1-2ms
- **Total feature-fetch: ~8-10ms**, leaving ~70ms for the actual model forward pass + network round trip to client.

## 19. Training Pipelines (data prep, orchestration, distributed training)

- **Data prep**: ML engineer submits an entity dataframe (player_id, label, event_timestamp) → calls `get_historical_features` → Spark PIT-join job reads relevant offline feature partitions (pruned by date range) → writes a materialized training-set Parquet file to S3.
- **Orchestration**: Airflow DAG: `label_generation → historical_feature_join → data_validation (Great Expectations schema/null checks) → train → eval → register_model`.
- **Distributed training**: for large tabular models (e.g., gradient-boosted trees at 55M-row scale) use distributed XGBoost/LightGBM on Spark; for deep models (e.g., sequence-based churn transformer) use PyTorch DDP across a small GPU cluster (4-8 A100s) — feature store's job ends at producing the materialized training set; training orchestration itself is largely handed off to the Training Platform chapter, but the **contract boundary** is: training pipeline must record the exact `feature_view:version` set used, written into model metadata for reproducibility and skew-audit lineage.
- **Reproducibility guarantee**: because offline joins are PIT-correct and versioned, re-running the same DAG 6 months later against the same label set + same feature versions reproduces an identical training set (modulo any explicit backfill correction).

## 20. Retraining Strategy (cadence, triggers)

- **Scheduled cadence**: most tabular models (churn, LTV) retrain weekly; matchmaking ELO-adjacent models retrain daily given fast-moving meta shifts (patch releases).
- **Triggered retraining**:
  - Drift monitor (Section 21) crosses threshold → auto-trigger retrain pipeline.
  - Feature-view version bump on a feature used by a production model → notify owning team via lineage graph, recommend retrain (not automatic, to avoid silent behavior change without review).
  - Major game patch/season launch (known meta shift) → manual trigger via live-ops calendar hook.
- **Feature-store-specific retraining consideration**: retraining must re-run `get_historical_features` against the **current** feature version pins (or explicitly upgrade pins) — the platform surfaces a diff report of which feature versions changed since last training run to make this an informed, auditable decision rather than an accidental drift-inducing swap.

## 21. Drift Detection (data drift, concept drift, metrics, thresholds)

| Drift Type | What's Monitored | Metric | Threshold |
|---|---|---|---|
| **Feature data drift** | Distribution of each serving-time feature value vs. its training-time distribution | PSI (Population Stability Index) per feature | PSI > 0.2 → warning; PSI > 0.3 → P2 alert, recommend retrain |
| **Online/offline skew** | Same entity's feature value computed via both paths | Relative delta (mean, p99) | mean delta > 2% over 1h rolling window → P2 |
| **Freshness drift** | Staleness of online feature values (event_time vs now) | p99 staleness (seconds) | p99 > 60s sustained 5 min → P1 |
| **Schema drift** | New/missing fields, dtype changes in raw telemetry feeding transforms | Schema Registry compatibility check | Any breaking change → hard fail at ingestion (not just alert) |
| **Concept drift (downstream)** | Model prediction distribution / label correlation shift (feature store surfaces the *inputs*; model-quality drift owned by ML platform, but feature store exposes per-feature importance-weighted drift score) | Feature-importance-weighted PSI aggregate | Aggregate weighted PSI > 0.25 → flag to model owner for investigation |
| **Volume drift** | Sudden drop/spike in feature write volume per feature_view (proxy for upstream pipeline breakage) | z-score on hourly write count vs 7-day rolling baseline | \|z\| > 4 → P1 |

## 22. Monitoring (infra, model quality, business metrics)

- **Infra**: Redis cluster CPU/memory/hit-rate, Kafka consumer lag per consumer group, Flink checkpoint duration/backpressure, Spark job duration/executor failures, S3 request throttling.
- **Feature-store-specific**: per-feature-view write QPS, per-feature-view read QPS, staleness distribution (p50/p99), PIT-join job duration and row counts, skew-alert rate, DLQ rate per topic, registry change frequency.
- **Model quality (proxied)**: feature-importance-weighted drift scores surfaced per model, feature coverage (% of requests where a requested feature was missing/defaulted — "null rate").
- **Business metrics**: correlation dashboards tying feature freshness/skew incidents to downstream business KPIs (e.g., matchmaking fairness complaints, personalization CTR) — owned jointly with the consuming team but instrumented via shared trace IDs.
- **Dashboards**: Grafana (infra + feature metrics), a dedicated "Feature Health Scorecard" per feature_view surfaced in the catalog UI (traffic-light: green/yellow/red based on drift + skew + freshness).

## 23. Alerting (alert conditions, thresholds, on-call routing)

| Condition | Severity | Threshold | Routing |
|---|---|---|---|
| Online store p99 read latency | P1 | > 25ms for 5 min | Feature-platform on-call (PagerDuty) |
| Online store availability | P1 | error rate > 1% for 2 min | Feature-platform on-call, auto-page |
| Kafka consumer lag (Flink) | P2 | lag > 2 min sustained 10 min | Feature-platform on-call |
| Skew alert (mean delta > 2%) | P2 | rolling 1h | Owning feature team (routed via feature_view owner metadata) |
| PSI drift > 0.3 | P2 | per feature | Owning model team, ticket auto-filed |
| DLQ rate > 0.1% of topic volume | P2 | 15 min window | Owning telemetry team |
| Offline batch job failure (backfill/nightly) | P3 | any failure | Feature-platform on-call, non-paging (Slack) |
| Registry schema breaking-change attempt | P3 | on attempt (blocked, not silently failed) | Requesting team, informational |
| Staleness p99 > 60s | P1 | sustained 5 min | Feature-platform on-call |

- On-call routing uses the metadata store's `owner` field to auto-route feature-quality alerts to the correct team, while infra alerts always route to the central feature-platform SRE rotation.

## 24. Logging (structured logging strategy, PII handling, retention)

- **Structured JSON logs** for every API call: `{trace_id, consumer, feature_refs, entity_count, latency_ms, cache_hit, staleness_ms, timestamp}` — no raw feature *values* logged by default (avoid PII leakage into log aggregation systems).
- **PII handling**: feature_views tagged `pii: true` in the registry (e.g., anything derived from real name, payment info, precise geolocation) — these features' *values* are never written to logs, only referenced by name; access requires elevated RBAC role + logged access audit trail (who queried which PII feature, when, for which entity, retained 1 year for compliance).
- **Access audit log**: separate append-only log stream (`feature_store.access_audit`) capturing every read of a PII-tagged feature — required for GDPR/CCPA data-subject-access-request tooling.
- **Retention**: operational logs (latency, errors) 30 days hot (Elasticsearch/OpenSearch) then archived to S3 for 1 year; access-audit logs retained 2 years minimum per legal requirement; DLQ sample logs retained 30 days.

## 25. Security (authn/authz, data encryption, threat model)

- **Encryption at rest**: Redis online store encrypted volumes (or Redis Enterprise TLS+encryption-at-rest); Delta Lake/S3 offline store uses SSE-KMS with per-studio KMS keys (supports tenant isolation).
- **Encryption in transit**: mTLS between all internal services (gateway ↔ Redis, Flink ↔ Kafka, gateway ↔ consumers).
- **Threat model specific to this system**:
  - *Cross-tenant data leakage*: a bug in namespace isolation exposes Studio A's player features to Studio B's model — mitigated via strict per-feature_view namespace ACLs enforced at the gateway (not just documentation convention), and row-level tagging by studio_id.
  - *PII exfiltration via bulk historical export*: a malicious/compromised client requests `get_historical_features` for PII-tagged features at scale — mitigated via per-consumer export volume rate limits + mandatory approval workflow for PII feature_view access + audit logging (Section 24).
  - *Feature poisoning*: a compromised upstream telemetry producer injects malformed/adversarial events to skew a feature (e.g., inflate a "trustworthiness" feature used by anti-cheat) — mitigated via schema validation at ingestion, anomaly detection on volume/value-distribution (ties into drift detection), and a "quarantine" mode where a feature_view can be marked untrusted, freezing it from serving until reviewed.
  - *Registry tampering*: unauthorized modification of a feature transformation to silently alter production behavior — mitigated via git-backed registry requiring PR review + signed commits, no direct write path to production compiled artifacts without CI validation.

## 26. Authentication (service-to-service and end-user auth)

- **Service-to-service**: mTLS certificates issued via internal PKI (SPIFFE/SPIRE-style identity), rotated every 24h; every internal call (gateway→Redis, Flink→Kafka broker) authenticated via mutual cert.
- **Consumer authentication (team/service level)**: OAuth2 client-credentials flow — each consuming service (matchmaking-service, live-ops-service) has a registered client_id/client_secret, token scoped to specific feature_view namespaces it's authorized to read.
- **No direct end-user auth** — the feature store is never called directly by a player-facing client; all access is server-to-server, mediated by the calling service which itself handles end-user auth upstream.
- **Human access (catalog UI, registry writes)**: SSO (SAML/OIDC via EA's internal identity provider), RBAC roles (`viewer`, `feature_owner`, `platform_admin`).

## 27. Rate Limiting (algorithm choice, per-tenant limits)

- **Algorithm**: token bucket per consumer (client_id), implemented at the gateway using a local+Redis-backed distributed counter (sliding window approximation) — chosen over fixed-window to avoid burst-at-boundary issues given bursty match-start traffic patterns.
- **Limits**:
  - Per-consumer default: 50K QPS sustained, burst to 100K QPS for 10s.
  - Historical/bulk export endpoint: 5 concurrent jobs per team, max 500M rows per job (prevents accidental/malicious full-table dumps).
  - PII-tagged feature reads: separate, stricter bucket — 5K QPS default, requiring explicit quota increase request reviewed by data-governance.
- **Tenant isolation**: each studio's traffic is bucketed independently so a traffic spike from FC 25 (World Cup event) cannot starve Apex Legends' matchmaking reads — enforced via per-namespace resource quotas at the Redis Cluster level (dedicated shards or slot-range reservations for the largest tenants).

## 28. Autoscaling (metrics-driven policy, HPA/VPA/KEDA specifics)

- **Online Feature API Gateway**: Kubernetes HPA on custom metric `requests_per_second_per_pod`, target 8K QPS/pod, min 20 replicas, max 300 replicas; scale-up stabilization window 30s (fast reaction for tournament traffic spikes), scale-down window 5 min (avoid flapping).
- **Flink streaming jobs**: reactive scaling via Flink's Adaptive Scheduler tied to Kafka consumer lag — KEDA `kafka` scaler triggers Flink parallelism increase when lag exceeds 30s-equivalent backlog.
- **Redis Cluster**: not auto-scaled in real-time (stateful resharding is expensive/risky); instead capacity-planned with 40% headroom and manually resharded ahead of known events (World Cup launch, season premieres) based on live-ops calendar.
- **Spark batch clusters**: ephemeral, autoscaled per-job via cluster manager (e.g., Databricks autoscaling min 20/max 200 executors) based on stage-level task backlog, torn down after job completion (cost control).

## 29. Cost Optimization (concrete levers)

- **Spot/preemptible instances** for all Spark batch jobs (nightly recompute, backfills) — batch is checkpoint-tolerant, ~65% cost reduction vs on-demand; retry-on-preemption built into Airflow task retries.
- **Storage tiering**: offline event-level logs moved from S3 Standard → S3 Infrequent Access after 30 days → Glacier Deep Archive after 180 days (per the 1.3PB footprint estimate, this is the single largest cost lever — raw event grain beyond 30 days is rarely queried directly, only via pre-aggregated rollups).
- **Aggregation-before-storage**: compact event-level streaming features into hourly/daily rollups after 30 days, dropping raw grain, cutting offline storage growth by an estimated 5-8x for older data.
- **Cache-first serving**: gateway L1 micro-cache (Section 11) reduces Redis read amplification ~15-20% for hot-entity workloads (tournament leaderboard viewers), directly reducing Redis node count needed.
- **Right-sized Redis tier**: given the ~216GB replicated online footprint fits comfortably in memory, avoid over-provisioning — use reserved instances for the steady-state 75-shard baseline, burst capacity only for known event spikes (live-ops calendar-driven pre-scaling rather than always-on peak capacity).
- **Feature deprecation lifecycle**: auto-flag feature_views with zero consumers (via lineage graph) for 90+ days → archive/delete, reducing both storage and streaming-compute waste (a stale Flink job still consuming Kafka for an unused feature is pure cost).

## 30. Disaster Recovery (RTO/RPO targets, backup strategy)

| Component | RTO | RPO | Backup Strategy |
|---|---|---|---|
| Online store (Redis) | 5 min | 60s (acceptable — rehydrate from streaming replay) | Multi-AZ replicas + periodic RDB snapshots every 15 min; on total loss, replay last 10 min of Kafka to rehydrate hot data |
| Offline store (Delta/S3) | 1 hr | 0 (durable object storage, versioned) | S3 versioning + cross-region replication (CRR) to a DR region |
| Metadata store (Postgres) | 15 min | 5 min | Continuous WAL archiving + automated snapshots every 5 min, PITR-capable |
| Kafka | 30 min | near-0 (replication factor 3, ack=all for feature-critical topics) | Multi-broker replication; MirrorMaker2 cross-region for DR |
| Feature Registry (git) | near-0 | 0 | Git itself is the durable source of truth; compiled artifacts rebuildable from source |

- **Full-region failure drill**: quarterly game-day exercise validating that online store can be cold-started in a secondary region from (a) Delta Lake offline snapshots + (b) replayed Kafka backlog, meeting the 5min/60s RTO/RPO for the hottest feature_views first (prioritized rehydration order defined per feature_view criticality tier).

## 31. Multi-Region Deployment (active-active vs active-passive, replication, latency routing)

- **Topology**: **active-active** across 3 regions (US-East, EU-West, APAC-Southeast) matching EA's existing game server regions — feature store must be co-located with matchmaking/inference to hit the 10ms online SLA (cross-region round trip alone would blow the budget).
- **Online store**: per-region independent Redis Cluster, **not** synchronously replicated cross-region (would violate latency SLA) — each region's online store is fed by a **regional** Flink consumer group reading from a **globally-replicated** Kafka topic (via Kafka MirrorMaker2 / Confluent Cluster Linking), so all regions converge to the same feature values within the 60s staleness bound, independently.
- **Offline store**: single logical store, physically replicated cross-region via S3 CRR — training jobs can run in any region against a consistent (eventually, within minutes) copy.
- **Latency routing**: clients (matchmaking/inference services) call their **local region's** Online Feature API via regional service discovery / GeoDNS — no cross-region online reads in the steady state.

```
   US-East                      EU-West                     APAC-SE
 ┌─────────────┐             ┌─────────────┐             ┌─────────────┐
 │ Flink (region)│            │ Flink (region)│            │ Flink (region)│
 │ Redis (region)│            │ Redis (region)│            │ Redis (region)│
 │ Gateway (region)│          │ Gateway (region)│          │ Gateway (region)│
 └──────┬───────┘             └──────┬───────┘             └──────┬───────┘
        │  reads/writes local          │                           │
        ▼                              ▼                           ▼
 ┌───────────────────── Kafka (globally mirrored via MM2) ─────────────────────┐
 └───────────────────────────────────────────────────────────────────────────┘
        │                              │                           │
        ▼                              ▼                           ▼
        └────────────── Offline Store (S3 CRR, eventually consistent global copy) ──────────────┘
```

- **Failure isolation benefit**: a regional Redis outage in APAC does not affect US-East/EU-West serving — degrades only that region (falls back to a "safe default feature" mode, see Failure Modes).

## 32. Blue/Green Deployment (how it applies here)

- **Feature Registry compiled artifacts** (Spark UDFs, Flink operators generated from the DSL) are blue/green deployed: new compiled version deployed alongside old, streaming jobs for a feature_view are cut over via a coordinated stop-the-old/start-the-new with Flink savepoint restore (state migrated from old operator's savepoint into the new job graph) — avoids reprocessing full history.
- **Online Feature API Gateway**: standard blue/green at the Kubernetes Service level — new gateway version deployed to a parallel Deployment, traffic cut over via Service selector swap once smoke tests (synthetic `get_online_features` calls against known entities with expected values) pass.
- **Rollback**: because feature_view versions are immutable and additive (v5 doesn't overwrite v4), rolling back a gateway or compute-engine deployment does not risk data corruption — worst case is briefly serving from a slightly older compiled transform, not data loss.

## 33. Canary Deployment (traffic-split strategy, health-check gates)

- **New feature_view version rollout** (e.g., `purchase_features:v6` replacing v5): canary at the **consumer** level, not infra level — a model/service can opt into v6 for 5% of traffic while 95% still reads v5, comparing downstream model metrics (conversion rate, prediction distribution) before full cutover. This is a feature-store-specific canary pattern distinct from typical infra canaries.
- **Gateway code canary**: 5% of gateway pods run the new build, traffic-split via Kubernetes Service with weighted routing (Istio/Linkerd), health-gated on: p99 latency delta < 10% vs baseline, error rate < 0.1%, skew-alert rate unchanged.
- **Streaming compute canary**: new Flink operator version processes a shadow copy of the Kafka topic (separate consumer group, writes to a shadow Redis namespace) — compared against production online values for 24h before promotion, catching subtle logic bugs before they touch real serving traffic.

## 34. Rollback Strategy (automated triggers, mechanics)

- **Automated triggers**: canary health-check gate failure (Section 33) → auto-abort canary, Argo Rollouts (or similar) reverts Service weights to 100% previous version within 60s.
- **Feature-version rollback**: because versions are immutable, "rollback" = repointing consumers' pinned `feature_view:version` back to the prior version via the registry (a metadata change, not a data migration) — takes effect on next model/config reload, typically < 5 min.
- **Streaming job rollback**: Flink savepoint taken immediately before any operator upgrade; rollback = restore prior job graph from that savepoint, resuming from the exact offset — no data loss, no reprocessing.
- **Offline backfill rollback**: backfill jobs write to a **new** Delta table version (Delta Lake's built-in time travel) — a bad backfill is rolled back via `RESTORE TABLE ... TO VERSION AS OF n`, not a destructive overwrite-undo.

## 35. Observability (tracing, metrics, logs correlation)

- **Tracing**: every `get_online_features` / `get_historical_features` call carries a `trace_id` (OpenTelemetry) propagated from the calling inference service, through the gateway, into Redis calls and on-demand UDF execution — enables end-to-end trace from "matchmaking requested a match" → "feature fetch" → "model inference" → "match formed", critical for diagnosing which layer contributed to a latency spike.
- **Metrics**: Prometheus-scraped, feature-store-specific metrics (Section 22) correlated with standard RED metrics (Rate/Errors/Duration) per component, visualized in Grafana with trace-exemplar links (click a p99 latency spike → jump to a representative trace).
- **Logs correlation**: structured logs (Section 24) include `trace_id`, enabling log-to-trace pivoting in the observability backend (e.g., Grafana Tempo + Loki, or Datadog APM) — a skew alert can be traced back to the exact batch job run + streaming operator version that produced the divergent values.
- **Feature-specific observability artifact**: a per-feature_view "health scorecard" combining trace-derived latency, drift metrics, and skew rate into one view — the primary debugging entry point for on-call.

## 36. Kubernetes Deployment (manifest sketch)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: online-feature-gateway
  namespace: feature-store
spec:
  replicas: 20
  selector:
    matchLabels: { app: online-feature-gateway }
  template:
    metadata:
      labels: { app: online-feature-gateway }
    spec:
      containers:
        - name: gateway
          image: ea-registry/feature-gateway:v42
          resources:
            requests: { cpu: "1", memory: "1Gi" }
            limits:   { cpu: "2", memory: "2Gi" }
          ports: [{ containerPort: 8080 }]
          env:
            - name: REDIS_CLUSTER_ENDPOINT
              valueFrom: { configMapKeyRef: { name: fs-config, key: redis_endpoint } }
---
apiVersion: v1
kind: Service
metadata: { name: online-feature-gateway, namespace: feature-store }
spec:
  selector: { app: online-feature-gateway }
  ports: [{ port: 443, targetPort: 8080 }]
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: online-feature-gateway-hpa, namespace: feature-store }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: online-feature-gateway }
  minReplicas: 20
  maxReplicas: 300
  metrics:
    - type: Pods
      pods:
        metric: { name: requests_per_second_per_pod }
        target: { type: AverageValue, averageValue: "8000" }
  behavior:
    scaleUp:   { stabilizationWindowSeconds: 30 }
    scaleDown: { stabilizationWindowSeconds: 300 }
```

## 37. Terraform Infrastructure (core infra snippet)

```hcl
resource "aws_elasticache_replication_group" "feature_store_online" {
  replication_group_id       = "fs-online-us-east"
  description                = "Online feature store - Redis Cluster mode"
  engine                     = "redis"
  engine_version             = "7.1"
  node_type                  = "cache.r7g.xlarge"
  num_node_groups            = 75           # shards, per capacity estimate
  replicas_per_node_group    = 2            # 3-way replication
  automatic_failover_enabled = true
  multi_az_enabled           = true
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  parameter_group_name       = "default.redis7.cluster.on"
}

resource "aws_s3_bucket" "feature_store_offline" {
  bucket = "ea-feature-store-offline-us-east"
}

resource "aws_s3_bucket_lifecycle_configuration" "offline_tiering" {
  bucket = aws_s3_bucket.feature_store_offline.id
  rule {
    id     = "event-grain-tiering"
    status = "Enabled"
    transition { days = 30  storage_class = "STANDARD_IA" }
    transition { days = 180 storage_class = "DEEP_ARCHIVE" }
  }
}

resource "aws_msk_cluster" "telemetry_backbone" {
  cluster_name           = "ea-telemetry-kafka"
  kafka_version          = "3.6.0"
  number_of_broker_nodes = 24
  broker_node_group_info {
    instance_type   = "kafka.m5.2xlarge"
    ebs_volume_size = 2000
  }
}
```

## 38. Why This Architecture

- Splitting online/offline stores by access pattern (KV low-latency vs. columnar analytical) is the only way to hit both the 10ms serving SLA and the multi-petabyte historical analytical needs simultaneously — a single store optimized for one starves the other.
- Compiling feature logic once to a shared IR, then generating both Spark and Flink execution paths, directly attacks the root cause of training/serving skew (divergent hand-written implementations) rather than only detecting it after the fact.
- Event-time indexing (vs. ingestion-time) is the only correct foundation for point-in-time joins — retrofitting PIT correctness onto an ingestion-time-only system is a common, expensive mistake this design avoids from day one.
- Regional active-active topology matches EA's existing game-server geography, keeping the latency-critical online path local while still achieving eventual global consistency via Kafka mirroring — avoids the false choice between "fast" and "consistent" at global scale by scoping strong requirements to what's actually latency-sensitive (online) vs. tolerant (offline).

## 39. Alternative Architectures

| Alternative | Description | Why Rejected / When Preferred |
|---|---|---|
| **Single unified store (e.g., all features in a distributed SQL DB like CockroachDB/Spanner)** | One store serving both batch analytical and low-latency point lookups | Rejected: can't simultaneously hit <10ms p99 at 1M+ QPS *and* efficiently scan petabytes for PIT joins; would need to over-provision for one workload to satisfy the other. Preferred only at much smaller scale (<10K QPS, <10TB) where operational simplicity outweighs the split-store complexity. |
| **Off-the-shelf managed feature store (e.g., Tecton, Databricks Feature Store, SageMaker Feature Store) as-is, no custom compilation layer** | Adopt vendor product wholesale, write feature logic separately for batch/stream per vendor's model | Rejected as sole solution because most vendor products still require separate batch/stream implementations (skew risk remains) unless the vendor itself offers a shared-IR compiler (some newer offerings do) — viable if EA is willing to accept vendor lock-in and the vendor's specific skew-prevention guarantees meet the bar; reasonable for a smaller studio without platform-eng investment appetite. |
| **Ingestion-time-only feature store (no event-time distinction)** | Simpler schema, only tracks when data landed, not when it became true | Rejected: cannot guarantee PIT correctness, silently leaks future information into training sets — acceptable only for exploratory/non-production prototyping, never for production model training. |
| **Fully synchronous cross-region online store (strong global consistency)** | Single global Redis/Spanner-backed online store, synchronously replicated | Rejected: cross-region round trip alone (~80-150ms) blows the 10ms SLA; only viable if regional latency requirements were far looser (e.g., a batch-scored nightly recommendation system with no real-time constraint). |

## 40. Tradeoffs

| Decision | Benefit | Cost |
|---|---|---|
| Separate online/offline stores | Each optimized for its access pattern; meets both latency and analytical needs | Operational complexity of keeping two systems consistent; skew monitoring overhead |
| Shared-IR compiled transforms (vs. hand-written dual implementations) | Eliminates most skew at the source | Upfront investment in DSL/compiler tooling; constrains feature authors to expressible transform subset (can't do arbitrary Python in the shared path) |
| Event-time indexing for PIT correctness | Prevents label leakage, enables reproducible training sets | More complex schema (dual timestamps), harder mental model for feature authors new to the platform |
| Regional active-active with eventual cross-region consistency | Meets latency SLA globally, isolates regional failures | Momentary cross-region feature value divergence (bounded by mirroring lag, typically seconds) — unacceptable for a hypothetical globally-synchronous fairness requirement (e.g., a strictly-global leaderboard feature), which would need a different pattern |
| Aggressive storage tiering/lifecycle | Major cost reduction (~1.3PB footprint controlled) | Slower/costlier access to old raw event-grain data (Glacier retrieval latency in hours) if a rare deep historical debug is needed |
| Immutable, additive feature versioning | Reproducibility, safe rollback | Storage/metadata growth over time (must actively deprecate); risk of "version sprawl" without governance |

## 41. Failure Modes (concrete scenarios and mitigations)

1. **Redis shard failure (hardware/AZ outage)** — Mitigation: multi-AZ replica promotion (automatic failover, <30s), client-side retry with exponential backoff; worst case that region serves stale/default values for affected entity range for the failover window.
2. **Flink job crash-loop due to bad event schema** — Mitigation: schema validation at ingestion routes malformed events to DLQ instead of crashing the job; circuit breaker halts a feature_view's streaming updates (serving last-known-good online value) rather than serving garbage.
3. **Kafka partition skew causing hot-partition backpressure** (e.g., a viral event on one popular streamer/match) — Mitigation: partitioning key includes a salted sub-key for extremely hot entities to spread load; backpressure triggers KEDA-driven Flink parallelism increase.
4. **Silent feature transformation bug shipped to production** (e.g., off-by-one in a windowed aggregation) — Mitigation: canary comparison against shadow Flink job (Section 33) catches most before full rollout; skew monitor catches divergence from offline recompute as a second line of defense.
5. **Offline PIT join accidentally includes future data** (bug in as-of join logic) — Mitigation: automated unit/property tests asserting no feature's `event_timestamp` in a joined training row exceeds the label's `event_timestamp`; CI gate blocks merge if violated.
6. **Cross-region Kafka mirroring lag spike** (network partition between regions) — Mitigation: each region continues serving from local, slightly-more-stale data (bounded by staleness SLA monitoring, which pages if exceeded); no cross-region synchronous dependency to fail.
7. **Registry/metadata store (Postgres) outage** — Mitigation: gateway caches feature-view metadata locally (Section 11) so serving continues in a degraded (no-new-registrations) mode; read replicas absorb read load, primary failure triggers automatic promotion.
8. **Backfill job corrupts a large historical partition** — Mitigation: Delta Lake time-travel `RESTORE TABLE`, backfills always write new table versions rather than in-place mutation.

## 42. Scaling Bottlenecks (where this breaks first at 10x/100x scale)

- **At 10x (10M+ QPS online reads)**: Redis Cluster shard count (~750 shards) becomes an operational management burden (resharding coordination, connection overhead per client fanning out to hundreds of shards) — first bottleneck is **client-side fan-out cost and connection pool exhaustion** at the gateway, not Redis itself; mitigation path: move to a proxy-based topology (e.g., Redis Cluster Proxy / Envoy) or a managed multi-tenant KV service (DynamoDB DAX) that abstracts shard fan-out.
- **At 10x**: Kafka partition count for the hottest topics (match_events at 512 partitions) saturates broker connection limits — needs topic re-partitioning and possibly broker fleet expansion beyond current MSK sizing.
- **At 100x (165M+ QPS)**: single-region active-active with per-region full replication of all feature_views becomes cost-prohibitive — would need **tiered feature placement** (only replicate hot/latency-critical feature_views per-region, keep cold/rarely-accessed ones in a single "home" region with cross-region read fallback accepting higher latency).
- **At 100x**: offline store PIT join Spark jobs (currently ~45 min nightly) would need fundamental redesign — likely incremental/streaming materialization of point-in-time feature snapshots rather than full recompute, since a 45-min job scaled 100x on data volume alone would blow any reasonable nightly window.
- **Metadata store (Postgres)** for the feature registry itself stays small regardless of traffic scale (feature *definitions* don't grow with QPS) — this component is not expected to be a bottleneck even at 100x, a useful thing to point out in an interview to show you're not blindly scaling everything.

## 43. Latency Bottlenecks (p50/p99 budget breakdown)

**Online feature retrieval (target p99 ≤ 10ms):**

| Stage | p50 | p99 |
|---|---|---|
| AuthN/RBAC check | 0.1ms | 0.5ms |
| Rate limiter check | 0.1ms | 0.3ms |
| Shard routing (consistent hash) | 0.05ms | 0.1ms |
| Redis network round trip (parallel fan-out) | 1.5ms | 6ms (dominant, tail-heavy due to shard-level GC pauses / occasional network jitter) |
| On-demand UDF transform | 0.5ms | 2ms |
| Response serialization | 0.2ms | 0.8ms |
| **Total** | **~2.5ms** | **~9.7ms** |

- **Where time is actually spent**: the Redis network round trip dominates p99, specifically driven by the **slowest shard in the fan-out set** (tail latency amplification — a 30-feature request touching 5 shards is only as fast as its slowest shard response). This is the primary target for optimization (shard-count-vs-fanout-width tuning, client-side timeout+partial-response strategies for non-critical features).
- **Historical/training join latency** is not on the interactive path and is budgeted in minutes/hours, not ms — no tail-latency concern there, throughput/cost is the relevant axis instead.

## 44. Cost Bottlenecks (what actually drives the bill)

- **#1: Offline storage at 1.3PB scale**, specifically the event-level (pre-rollup) data — even with tiering, the sheer volume of raw telemetry-derived feature history dominates storage spend; the biggest lever is aggressive, earlier rollup/compaction (don't wait 30 days if 7 would do for most feature_views).
- **#2: Streaming compute (Flink) always-on cost** — unlike batch (ephemeral, spot-eligible), streaming clusters run 24/7 on reliable (non-spot, or carefully-managed spot with fast state recovery) infrastructure; ~260 vCPUs baseline is modest, but this scales roughly linearly with feature_view count, and **unused/zombie feature_views left running is pure waste** (ties to the deprecation lifecycle lever in Section 29).
- **#3: Online store over-provisioning for peak events** — if capacity is sized for tournament/launch-day peaks and left at that size year-round rather than calendar-driven pre-scaling, this is a recurring unnecessary cost; the fix is operational (live-ops calendar integration), not architectural.
- **#4: Cross-region data transfer** — Kafka mirroring and S3 CRR across 3 regions incurs continuous egress cost proportional to feature write volume; mitigated by only mirroring feature_views that actually need multi-region freshness (not blanket-replicating everything).

## 45. Interview Follow-Up Questions

1. How do you guarantee point-in-time correctness when the label-generation job and the feature-computation job run on completely different schedules and can both be "late"?
2. Walk me through what happens, end to end, if a feature transformation bug is deployed to the streaming path but not yet to the batch path — how does the system detect and contain it?
3. Why not just use one store (e.g., a fast distributed SQL database) for both training and serving to avoid the skew problem entirely?
4. How would you extend this design to support a feature that requires a rolling 90-day window aggregation efficiently in both batch and streaming?
5. What happens to in-flight `get_online_features` requests during a Redis cluster resharding operation?
6. How do you prevent a single noisy tenant (studio) from degrading feature-store latency for everyone else?
7. If the skew monitor itself has a bug and stops firing alerts, how would you find out, and how long could skew go undetected?
8. How would you design feature versioning to support gradual model migration from `v5` to `v6` of a feature without breaking currently-deployed models?
9. What's your approach to handling a feature whose "correct" value genuinely depends on very recent data (sub-second freshness) — does the 60s staleness bound still hold?
10. How do you reason about cost tradeoffs between recomputing features from raw data vs. storing more precomputed feature history?

## 46. Ideal Answers

1. **PIT correctness under asynchronous schedules**: correctness is enforced by anchoring every feature write to an explicit **event_timestamp** (when the underlying fact became true), never wall-clock/ingestion time. The PIT join engine filters strictly on `feature.event_timestamp <= label.event_timestamp` regardless of when either was actually computed or landed — a feature computed today with event_timestamp of three weeks ago is joined as if it were three weeks old. Lateness only affects *availability* of a feature value at join time (if the batch job hasn't run yet, that feature value simply isn't there yet), never *correctness* of values that are present.

2. **Streaming-only bug containment**: the skew monitor's continuous sampling would surface a divergence between the online (buggy) and offline (correct) values for affected entities within one monitoring cycle (~5 min); this triggers a P2 skew alert routed to the owning team via the feature_view's registered owner. Blast radius containment: the canary/shadow-deployment process (Section 33) is designed to catch this *before* full production rollout by comparing shadow Flink output against production for 24h; if it still slips through, the fix is to roll back via Flink savepoint restore to the pre-bug operator version, and mark the affected time window's online values as suspect in the audit log for any model that consumed them during that window (enabling downstream teams to assess retraining need).

3. **Why not one unified store**: a single store optimized for sub-10ms point lookups at 1M+ QPS (row/KV-oriented, in-memory-tier) is structurally unsuited to efficient large-range analytical scans over petabytes of historical data (columnar, compression, partition pruning) needed for PIT joins — and vice versa, a columnar analytical store cannot hit single-digit-millisecond point-lookup SLAs at that throughput. Distributed SQL databases (Spanner-class) narrow this gap but still generally underperform a purpose-built KV cache on raw point-lookup latency at this scale, and would cost significantly more to run at the required QPS. The two-store split is a direct consequence of genuinely different, non-overlapping performance requirements — not just habit.

4. **90-day rolling window feature**: for the streaming path, maintain incremental windowed state in Flink (e.g., a mergeable sketch or exact sliding-window aggregate keyed by entity_id) rather than recomputing from scratch each event — Flink's built-in windowing with RocksDB state backend handles this at the required scale, checkpointed for fault tolerance. For the batch path, the same 90-day window is computed via a Spark job reading the appropriate partition range, using the *same compiled aggregation logic* from the shared IR to guarantee identical semantics (e.g., identical handling of window boundary inclusivity) between the two paths — this is exactly the class of feature the shared-IR compiler is designed to keep consistent.

5. **Resharding and in-flight requests**: Redis Cluster resharding moves hash slots gradually with `ASK`/`MOVED` redirection semantics — in-flight requests to a slot being migrated get a redirect response, and a well-behaved client (or a smart proxy layer) transparently follows the redirect, adding a few hundred microseconds of extra latency during the migration window rather than failing. Resharding is scheduled during low-traffic windows and executed gradually (small batches of slots at a time) specifically to bound this latency impact, monitored against the same p99 SLA dashboards to auto-pause if impact exceeds budget.

6. **Noisy-tenant isolation**: per-consumer token-bucket rate limiting (Section 27) is the first line of defense, but the more robust mitigation is **physical resource isolation** for the largest tenants — dedicating specific Redis shards / hash-slot ranges to the highest-volume studios rather than fully sharing a pooled cluster, so a traffic spike from one studio's live event can only exhaust its own reserved capacity, not a shared pool. Smaller/lower-volume tenants share a common pool with fair-share quotas.

7. **Monitoring the monitor**: the skew monitor itself emits a basic heartbeat/liveness metric (last successful comparison run timestamp, sample count processed) independent of whether it *finds* skew — a missing heartbeat for >15 min pages on-call directly, distinct from a skew-content alert. This dead-man's-switch pattern ensures silent monitor failure is caught within minutes, not discovered retroactively during a data-quality incident review. Additionally, periodic synthetic injection (deliberately writing a known-divergent test value) validates the detector's sensitivity end-to-end.

8. **Gradual version migration**: because feature_view versions are immutable and additive, v5 and v6 coexist indefinitely — a model's manifest simply pins whichever version it was trained against. Migration is consumer-driven: a team updates their model's feature references to `v6`, retrains/validates against it (potentially via the canary pattern in Section 33, comparing v5-served vs v6-served production metrics), and cuts over independently of any other consumer still on v5. The registry's lineage graph tracks exactly which consumers reference which versions, so a version can only be deprecated/deleted once its last consumer has migrated — preventing accidental breakage.

9. **Sub-second freshness exception**: the 60s staleness bound is a platform *default*, not a hard universal ceiling — for a feature_view with a genuinely tighter freshness requirement (e.g., a live in-match anti-cheat signal), the platform supports a lower-latency tier: smaller Flink micro-batch/window-emit intervals (e.g., 1-2s) and a tighter online-store TTL/staleness monitoring threshold, at the cost of higher streaming-compute overhead (more frequent state checkpoints, higher write amplification to the online store) — this is registered per-feature_view as an SLA override, and the skew/staleness alerting thresholds are correspondingly tightened just for that feature_view rather than loosened platform-wide.

10. **Recompute vs. store tradeoff**: this is fundamentally a storage-cost vs. compute-cost curve — storing more precomputed history reduces repeated computation (good when many models/teams reuse the same historical features for different training windows) but grows the 1.3PB-class storage footprint linearly; recomputing on-demand from raw data saves storage but multiplies compute cost with every training run that touches that feature, and risks reproducibility drift if the recompute logic changes between runs. The chosen middle ground: precompute and store daily/hourly rollups (cheap relative to raw event grain) for anything with multiple consumers or reuse across training runs, but keep only short-retention raw event-grain data and rely on recompute-from-source (accepting the compute cost) for rarely-accessed, single-consumer, exploratory feature definitions — governed by an access-frequency-driven lifecycle policy rather than a blanket rule.
