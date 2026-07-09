# Streaming ML Platform

## 1. Problem Framing & Requirement Gathering

Design a streaming ML platform that computes features and serves predictions **in real time** off live event streams (player telemetry, matchmaking events, in-game purchases, chat/toxicity signals) for EA live-service titles (e.g., Apex Legends, FIFA Ultimate Team, The Sims). Unlike a batch feature store + offline-trained model served behind a REST endpoint, this system must:

- Compute windowed aggregates (e.g., "kills in last 5 min", "spend in last 24h", "session churn signal") continuously from Kafka topics via Flink.
- Serve inference using **features that are seconds old**, not hours old.
- Support **online/incremental model updates** from the same streams (e.g., contextual bandit for matchmaking, anti-cheat anomaly scoring, dynamic difficulty adjustment) without a full batch retrain cycle.
- Guarantee point-in-time correctness between what the model saw at train time and what it sees at serve time, despite both being streaming-derived.

Primary use cases at EA:
1. **Real-time anti-cheat scoring** — flag aimbot/wallhack behavior within seconds of anomalous input patterns.
2. **Live matchmaking quality prediction** — predict match quality/churn risk to rebalance lobbies before match start.
3. **Dynamic pricing/offer personalization** in FIFA Ultimate Team packs based on last-N-minutes behavior.
4. **Toxicity/chat moderation scoring** streaming over voice-to-text and chat events.

## 2. Functional Requirements

- FR1: Ingest player telemetry events (kills, deaths, purchases, chat, input traces) from Kafka at the platform's edge.
- FR2: Compute streaming feature aggregates using tumbling/sliding/session windows (e.g., 1-min, 5-min, 1-hour, 24-hour windows).
- FR3: Serve low-latency point predictions (`p99 < 100ms`) using the freshest available streaming features joined with static/offline features.
- FR4: Support online model updates (incremental gradient steps, bandit parameter updates) triggered by labeled events arriving on the stream (e.g., "ban confirmed", "purchase completed", "match churned").
- FR5: Maintain a consistent online feature store queryable both by the streaming job (for building training data) and the inference service (for serving).
- FR6: Provide backfill/replay capability — reprocess a Kafka topic from an offset to rebuild feature state after a bug fix.
- FR7: Version features, models, and window-definitions; support side-by-side shadow scoring of a new model version.
- FR8: Emit prediction + feature snapshot logs for offline evaluation, drift detection, and audit.
- FR9: Support per-title, per-region isolation (a bug in Apex's pipeline must not affect FIFA's).

## 3. Non-Functional Requirements

| Dimension | Target |
|---|---|
| Inference latency | p50 < 30ms, p99 < 100ms, p99.9 < 250ms (in-process feature lookup + model forward pass) |
| Feature freshness | p99 staleness < 2s from event ingestion to feature-store visibility |
| Streaming throughput | Sustain 2.5M events/sec platform-wide at peak (see Section 6) |
| Availability | 99.95% for inference path (≈4.4 hrs/yr downtime budget); 99.9% for streaming feature pipeline |
| Consistency | Eventual consistency for online feature store; read-your-writes NOT required across regions; exactly-once semantics for feature aggregation (no double-counted kills) |
| Durability | Kafka topics retained 7 days (replay window); feature snapshots retained 30 days for audit |
| Cost | Streaming compute (Flink) + online store must stay under $0.0004 per 1K events processed fully loaded |
| Model update latency | Online model parameter refresh visible to serving layer within 60s of a labeled event |
| Scalability | Linear horizontal scaling to 5x traffic during a title launch/live-event spike without redesign |

## 4. Clarifying Questions an Interviewer Would Expect

1. Is "online learning" true per-event SGD, or periodic micro-batch model refresh (e.g., every 60s) from a streaming-materialized training set? (Changes architecture significantly.)
2. What's the acceptable staleness for features used in anti-cheat vs. matchmaking — are all use cases equally latency-sensitive?
3. Do we need exactly-once end-to-end (event → feature → prediction → downstream action), or is at-least-once with idempotent consumers acceptable?
4. Is inference synchronous (blocking a game client request) or asynchronous (scoring published to a topic, consumed later)?
5. How many concurrent titles/games share this platform vs. per-title deployments?
6. What's the label latency — how long after a prediction do we learn the ground truth (e.g., "was this actually a cheater")? This dictates online-learning feedback loop design.
7. Regulatory constraints — GDPR/COPPA on player telemetry, especially for minors (EA has many titles with underage players)?
8. Is model rollback/shadow-mode required before promoting an online-updated model to production traffic?
9. What's the blast radius if the streaming feature pipeline falls behind — do we degrade to stale-feature serving, fallback model, or reject requests?

## 5. Assumptions

1. Platform serves 8 live-service titles concurrently, largest (Apex-scale) contributing 60% of traffic.
2. Peak concurrent players platform-wide: 6M CCU during a live event; average 1.8M CCU.
3. Each active player emits ~15 telemetry events/minute (movement ticks aggregated client-side, kills, purchases, chat).
4. Online learning = **micro-batch incremental updates** every 30-60s from a streaming-materialized labeled dataset, not per-event SGD (per-event SGD deemed too unstable for production given noisy reward/label signals) — confirmed as reasonable default absent interviewer override.
5. Inference is synchronous for anti-cheat (blocks a "kick/allow" decision) and asynchronous for pricing/personalization (published to a recommendation topic).
6. Exactly-once semantics required for financial/purchase-related aggregates; at-least-once + idempotency acceptable for telemetry-derived behavioral features.
7. Feature vectors are small: 80-150 numeric/categorical features per entity (player, match, or player-match pair).
8. Models are small-to-medium (gradient boosted trees for anti-cheat, logistic/linear contextual bandit for matchmaking, <50MB serialized) — no LLM-scale models in this system.
9. GDPR applies; EU player data must stay in EU region infra.

## 6. Capacity Estimation

**Event ingestion volume**
- 6M peak CCU × 15 events/min = 90M events/min = **1.5M events/sec** peak telemetry.
- Add purchase/chat/matchmaking event streams: +40% overhead → **~2.1M events/sec** peak, rounded to **2.5M events/sec** design target (Assumption/NFR alignment, includes replay traffic headroom).

**Kafka sizing**
- Avg event size: 400 bytes (protobuf-encoded).
- Peak ingress bandwidth: 2.5M × 400B = 1GB/sec = **8 Gbps**.
- With 3x replication: 24 Gbps cluster network throughput → needs ~30-40 broker nodes (each broker sustaining ~1-1.5 Gbps safely) across topics.
- 7-day retention: 1GB/sec × 604,800s ≈ 604 TB raw, ×3 replication ≈ **1.8 PB** Kafka storage footprint (tiered storage to S3-compatible cold tier after 24h to cut local disk needs to ~120TB hot).

**Flink cluster**
- Windowed aggregation (1-min, 5-min, 1-hr, 24-hr tumbling/sliding windows) per player/match key.
- Rule of thumb: 1 vCPU handles ~15-25K events/sec for simple aggregation with RocksDB state backend.
- 2.5M events/sec ÷ 20K/sec/core ≈ **125 task-manager vCPUs** minimum, provisioned at 2.5x for window-state skew and checkpoint overhead → **~320 vCPUs** (≈40 nodes at 8 vCPU each).
- State size: 6M concurrent player keys × ~2KB feature state each = 12GB working state, ×24hr window retention factor ≈ **~80-120GB total RocksDB state**, comfortably fits distributed across task managers with local SSD.

**Online feature store (serving side)**
- Read QPS: inference calls ≈ prediction QPS. Anti-cheat scores every active player every 10s: 6M/10s = 600K reads/sec. Matchmaking/pricing adds ~150K reads/sec. Total **~750K feature-store reads/sec**.
- Write QPS from Flink sink: one write per window-close per key per window-size ≈ 6M keys / 60s (finest window) ≈ 100K writes/sec sustained, bursts to 250K/sec.
- Feature store storage: 6M active entities × 150 features × 8 bytes ≈ 7.2GB hot in-memory (Redis-class), trivially sized; snapshot/audit copy in columnar store: 30 days × 100K writes/sec × 1.2KB row ≈ **~310 TB/month** in cold columnar storage (compressed ~4x → ~80TB/month actual).

**Model serving**
- Anti-cheat GBT model: CPU inference, ~0.5ms/inference on 1 vCPU. 600K QPS ÷ (1000ms/0.5ms per core=2000/sec/core) = **300 vCPUs** for anti-cheat scoring alone, provisioned to 450 vCPUs at 1.5x headroom.
- No GPU requirement for this system's models (tree ensembles + linear/bandit models) — GPU only needed if EA later folds in embedding-based player models; out of scope here.

**Online learning compute**
- Micro-batch retrain every 30-60s over a rolling window of ~2M labeled events: GBT incremental fit or warm-start logistic regression, ~15-30s compute on 16 vCPUs per title → **8 titles × 16 vCPUs = 128 vCPUs** dedicated online-trainer pool.

**Total rough compute footprint**: ~320 (Flink) + ~450 (serving) + ~128 (online trainer) + control-plane/misc ≈ **~950-1000 vCPUs** platform-wide at peak, plus Kafka broker fleet (~35 nodes) — no GPUs required for baseline scope.

## 7. High-Level Architecture

```
                                   ┌─────────────────────────┐
                                   │   Game Clients / Servers │
                                   │ (Apex, FIFA, Sims, ...)  │
                                   └────────────┬─────────────┘
                                                │ gRPC/HTTPS (telemetry SDK)
                                                ▼
                          ┌───────────────────────────────────────┐
                          │  Edge Ingestion Gateway (regional)     │
                          │  - authn, schema validation, batching  │
                          └────────────────────┬────────────────────┘
                                                ▼
                          ┌───────────────────────────────────────┐
                          │            Kafka (per-region)          │
                          │  topics: telemetry.raw, purchases,     │
                          │  matchmaking.events, chat.events,      │
                          │  labels.confirmed                      │
                          └───────┬───────────────────┬───────────┘
                                  ▼                   ▼
                  ┌───────────────────────┐   ┌──────────────────────────┐
                  │  Flink Streaming Job   │   │  Flink Streaming Job      │
                  │  (Windowed Feature      │   │  (Labeled Training-Set    │
                  │   Aggregation)          │   │   Materialization)        │
                  │  - tumbling/sliding win │   │  - joins labels+features  │
                  │  - exactly-once sink    │   │  - emits training rows    │
                  └──────────┬─────────────┘   └───────────┬──────────────┘
                             ▼                              ▼
                  ┌───────────────────────┐      ┌──────────────────────────┐
                  │  Online Feature Store  │      │   Streaming Trainer       │
                  │  (Redis/ScyllaDB, hot) │      │   (micro-batch, 30-60s)   │
                  │  + offline columnar     │      │   -> new model params     │
                  │    replica (audit)      │      └───────────┬──────────────┘
                  └──────────┬─────────────┘                  ▼
                             │                        ┌──────────────────────┐
                             │                        │   Model Registry      │
                             │                        │  (versioned params)   │
                             │                        └───────────┬───────────┘
                             ▼                                    ▼
                  ┌─────────────────────────────────────────────────────┐
                  │              Inference Service (stateless)           │
                  │  - fetch feature vec (online store)                  │
                  │  - fetch latest model shard (hot-reload from registry)│
                  │  - score, apply business rule, respond/publish        │
                  └───────┬─────────────────────────────┬─────────────────┘
                          ▼                              ▼
              ┌───────────────────────┐       ┌───────────────────────────┐
              │  Sync response to      │       │  prediction.events topic   │
              │  game server (anti-    │       │  (async consumers: pricing,│
              │  cheat kick/allow)      │       │  matchmaking rebalancer)   │
              └───────────────────────┘       └───────────────────────────┘
                          │
                          ▼
              ┌───────────────────────────────────────────────────┐
              │ Monitoring/Drift/Logging plane (cross-cutting):     │
              │ Prometheus, feature/pred snapshot log -> S3/Iceberg,│
              │ drift detectors, alerting                           │
              └───────────────────────────────────────────────────┘
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| Edge Ingestion Gateway | AuthN, schema validation, batching, backpressure to clients | gRPC/HTTP from game client/server SDK | Horizontal, stateless pods behind regional LB |
| Kafka Cluster | Durable, ordered (per-key) event log; replay source of truth | Producer/consumer API, topic per event class | Partition count (broker + partition scale-out) |
| Flink Feature-Aggregation Job | Compute windowed aggregates, write to online store | Kafka source → keyed windowed operators → sink | Task-manager slots, keyed by player/match ID |
| Flink Label-Join Job | Join late-arriving labels with historical feature snapshots for training set | Kafka source (labels + feature-log topic) → join → sink to training-set topic | Same as above, separate job graph |
| Online Feature Store | Low-latency point lookups of latest feature vector per entity | gRPC `GetFeatures(entity_id, feature_set_version)` | Sharded by entity ID hash |
| Offline Feature Store (columnar) | Historical feature snapshots for audit, backfill, offline eval | Iceberg/Parquet on S3, queried via Trino/Spark | Partitioned by date + title |
| Streaming Trainer | Consume materialized training-set topic, run incremental fit, push new params | Kafka consumer → training loop → Model Registry API | One consumer group per title, parallel by title shard |
| Model Registry | Version, store, and serve model artifacts/params with promotion workflow | REST/gRPC `GetLatestModel(title, model_name, stage)` | Stateless read replicas behind cache |
| Inference Service | Serve predictions; combine online features + model; apply decision rules | gRPC `Predict(entity_id, context)` | Horizontal, stateless, autoscaled on QPS/CPU |
| Drift Detector | Compare live feature/prediction distributions vs. baseline | Batch job over snapshot logs, cron/streaming hybrid | Scales with number of monitored feature sets |
| Prediction/Feature Logger | Durable audit trail for every scored request | Async sink to Kafka → Iceberg | Scales with prediction QPS |

## 9. API Design

**Feature Store API**
```
GetFeatures(request) -> FeatureVectorResponse
  request:
    entity_id: string          # player_id or match_id or composite key
    feature_set: string        # e.g. "anticheat_v3", "matchmaking_v7"
    as_of: optional timestamp  # for point-in-time / backfill queries
  response:
    features: map<string, float>
    freshness_ms: int          # age of staleest constituent feature
    feature_set_version: string
```
- Version via `feature_set` name suffix (`_v1`, `_v2`) — never mutate a live feature set in place; new version = new column family in online store.

**Inference API**
```
POST /v1/titles/{title}/models/{model_name}/predict
Request:
{
  "entity_id": "player_12345",
  "context": { "match_id": "m_998", "region": "eu-west" },
  "model_version": "latest" | "v2026_07_08_shadow"
}
Response:
{
  "prediction": 0.87,
  "decision": "FLAG_REVIEW",
  "model_version": "v2026_07_08_1400",
  "feature_freshness_ms": 340,
  "trace_id": "..."
}
```
- Versioning: URL path is stable (`/v1`); model_version field selects registry entry; default `"latest"` resolves to current production-stage pointer, enabling shadow (`model_version=shadow`) calls without a new endpoint.

**Async Prediction Consumption** (pricing/matchmaking)
- Topic `prediction.events.{title}` — consumers pull via Kafka consumer group; schema registry enforces Avro/Protobuf schema evolution rules (backward-compatible only).

**Model Registry API**
```
POST /v1/models/{title}/{model_name}/versions   # register new online-trained version
GET  /v1/models/{title}/{model_name}/latest?stage=production|shadow|canary
POST /v1/models/{title}/{model_name}/promote     # move version between stages
```

| Endpoint | Method | SLA | Auth |
|---|---|---|---|
| `/v1/titles/{t}/models/{m}/predict` | POST | p99 100ms | mTLS service token |
| `/v1/features/{feature_set}` (GetFeatures) | gRPC | p99 15ms | mTLS |
| `/v1/models/.../latest` | GET | p99 20ms (cached) | mTLS |
| `/v1/models/.../promote` | POST | best-effort, human-gated | OAuth2 + RBAC (MLE role) |

## 10. Database Design

| Store | Type | Why | Partition/Shard Key |
|---|---|---|---|
| Online Feature Store | Redis Cluster (hot, <10ms) or ScyllaDB (if >TB-scale state) | Sub-ms/low-ms point lookups at 750K QPS; ScyllaDB preferred over Redis when working set exceeds cluster RAM budget cost-effectively | Hash(entity_id) |
| Offline Feature/Prediction Log | Apache Iceberg on S3, queried via Trino/Spark | Columnar, cheap, supports time-travel for point-in-time correctness and backfill audits | Partitioned by `title`, `event_date` |
| Model Registry Metadata | PostgreSQL | Strong consistency needed for version pointers, promotion state machine, small dataset | Row-level, indexed by `(title, model_name, version)` |
| Model Artifacts | Blob store (S3-compatible) | Large binary blobs, versioned, immutable | Keyed by `{title}/{model_name}/{version}/model.bin` |
| Kafka (event log) | Log-structured, not a DB but system-of-record for replay | Ordered per-partition, replay-capable | Partitioned by entity_id for per-player ordering guarantee |

Schema sketch — online feature store (Redis hash per entity):
```
KEY: fs:anticheat_v3:{player_id}
FIELDS: kills_5m, deaths_5m, headshot_ratio_1m, input_variance_10s,
        last_updated_ts, session_id
TTL: 6 hours (auto-expire stale players, avoids unbounded growth)
```

Schema sketch — Iceberg prediction log table:
```
prediction_log (
  trace_id STRING,
  title STRING,
  entity_id STRING,
  model_version STRING,
  feature_snapshot MAP<STRING, DOUBLE>,
  prediction DOUBLE,
  decision STRING,
  event_ts TIMESTAMP,
  ingest_ts TIMESTAMP
) PARTITIONED BY (title, days(event_ts))
```

Point-in-time correctness: training-set materialization job joins `labels.confirmed` events against the **feature snapshot captured at prediction time** (logged in `prediction_log`), never against the *current* online store value — prevents label leakage from future feature states.

## 11. Caching

| Cached Item | Cache Type | Invalidation | Pattern |
|---|---|---|---|
| Latest model artifact (in inference pod memory) | Local in-process cache | TTL 60s + registry push notification (Kafka `model.updates` topic) | Cache-aside with async refresh |
| `GetLatestModel` registry lookups | Redis cache in front of Postgres | Write-through on promotion event | Write-through |
| Hot feature vectors (very active players) | L1 in inference-pod LRU (avoid Redis round-trip) | TTL 2s (must respect freshness NFR) | Cache-aside, short TTL |
| Static/offline features (player profile, cosmetic tier) | CDN/Redis, longer TTL (5 min) | Explicit invalidation on profile update event | Cache-aside |

- No write-through for the online feature store itself — Flink sink is the sole writer; inference is read-only, so no cache-consistency-with-write concern there beyond staleness bound.
- Model hot-reload avoids restart-based deploys: inference pods subscribe to `model.updates` topic and swap the in-memory model pointer atomically (no request drop).

## 12. Queues & Async Processing

| Queue/Topic | Delivery Semantics | Why | Dead-Letter Handling |
|---|---|---|---|
| `telemetry.raw.*` | At-least-once | High volume, idempotent aggregation (dedup by event_id) makes exactly-once unnecessary here | Malformed events → `telemetry.dlq` with schema-validation error tag; alert if DLQ rate > 0.1% |
| `purchases.*` | Exactly-once (Kafka transactional producer + idempotent consumer) | Financial correctness required — no double-charging or double-counted spend features | Failed transactional writes retried with exponential backoff; after 5 failures → `purchases.dlq`, paged to on-call |
| `labels.confirmed` | At-least-once + idempotent upsert keyed by `(entity_id, label_type, window)` | Ban confirmations, purchase completions — late and duplicate delivery expected | N/A (idempotent by design) |
| `model.updates` | At-least-once, small volume | Notifies inference pods of new model version | N/A — pods poll registry as fallback if notification missed |
| `prediction.events.*` | At-least-once | Downstream consumers (pricing engine) are naturally idempotent (keyed by trace_id) | `prediction.dlq` for schema-registry rejections |

- Flink checkpointing (RocksDB state backend, checkpoint every 10s to S3) provides exactly-once **within** the Flink job graph for windowed aggregation, even though upstream Kafka delivery is at-least-once — dedup via event_id in state.

## 13. Streaming & Event-Driven Architecture

**Core topics**

| Topic | Partitions | Key | Retention | Schema |
|---|---|---|---|---|
| `telemetry.raw.{title}` | 200 | player_id | 24h hot + 7d tiered | Protobuf, schema-registry enforced |
| `purchases.{title}` | 50 | player_id | 90d (financial audit) | Protobuf |
| `matchmaking.events.{title}` | 100 | match_id | 24h | Protobuf |
| `chat.events.{title}` | 100 | player_id | 24h | Protobuf |
| `labels.confirmed.{title}` | 50 | entity_id | 30d | Protobuf |
| `feature.updates.{title}` (Flink sink) | 200 | entity_id | 7d (replay-driven rebuild) | Avro |
| `training.materialized.{title}` | 50 | entity_id | 14d | Avro |
| `model.updates.{title}` | 4 | model_name | 3d | JSON (small control-plane messages) |
| `prediction.events.{title}` | 100 | entity_id | 7d | Protobuf |

**Consumer groups**
- `flink-feature-agg-{title}`: one per title, parallelism = partition count of `telemetry.raw`.
- `flink-label-join-{title}`: joins `labels.confirmed` + replayed `feature.updates`.
- `streaming-trainer-{title}`: consumes `training.materialized.{title}`.
- `inference-model-watcher`: lightweight, consumes `model.updates.*` across all titles (single small consumer group, broadcast-like).
- `drift-detector-{title}`: consumes `prediction.events` + `feature.updates` for live distribution comparisons.

- Windowing: tumbling windows for fixed-cadence features (1-min kill count), sliding windows for smoothed signals (5-min headshot ratio with 1-min slide), session windows for matchmaking (gap-based, 10-min inactivity closes session).
- Watermarking: bounded out-of-orderness of 5s for telemetry (client clock skew tolerance); late events beyond watermark routed to a `late-events` side output and reconciled in the offline job.

## 14. Model Serving

- **Framework**: Custom lightweight inference service (Rust/Go/Java) hosting GBT (XGBoost/LightGBM) and linear/bandit models — **not** a heavyweight GPU serving stack (Triton/TF-Serving) since models are small and CPU-bound; avoids GPU cost/ops overhead for this workload class.
- **Batching**: Micro-batching disabled for the synchronous anti-cheat path (batching adds latency for a single-entity request); enabled for the async pricing path where the recommendation consumer can batch 50-100 entities per model call for throughput (GBT inference vectorizes well).
- **Multi-model**: One inference service hosts multiple model versions per title simultaneously (production, canary, shadow) — request carries `model_version`; shadow traffic is mirrored (fire-and-forget, results logged not returned) to validate a newly online-trained model before promotion.
- **Hardware**: CPU-only fleet, autoscaled (see Section 28); AVX2-optimized XGBoost inference; no GPU needed unless a future embedding-based player model is added (explicitly out of current scope).
- **Hot reload**: model swap via atomic pointer update in-process, triggered by `model.updates` topic, avoiding pod restarts and cold-start latency spikes.

## 15. Feature Store

- **Online store**: Redis Cluster / ScyllaDB, serves `GetFeatures` at p99 <15ms, holds only the latest value per (entity, feature_set) — no history.
- **Offline store**: Iceberg/Parquet, holds full history of feature snapshots (one row per prediction event, not per window-close) for training-set reconstruction and audits.
- **Point-in-time correctness**: Because both training labels and serving happen off the same streaming pipeline, the platform avoids the classic offline/online skew by **logging the exact feature vector used at prediction time** (`prediction_log.feature_snapshot`) rather than re-deriving features retroactively from the online store (which would reflect a later, updated state). Training-set materialization always joins against this logged snapshot, never live-queries the online store for historical timestamps.
- **Feature versioning**: `feature_set` version bump required whenever a window definition or aggregation function changes — prevents silently mixing old-window and new-window semantics mid-stream. Dual-write both versions during a migration window until all consumers move to the new version.

## 16. Vector Database

**N/A for this system's baseline scope.** The models in play (GBT anti-cheat classifier, linear/bandit matchmaking model, pricing propensity model) operate on structured numeric/categorical windowed aggregates, not embeddings requiring similarity search. If EA later adds a semantic player-behavior embedding model (e.g., for smurf detection via behavioral similarity), a vector DB (e.g., pgvector or a dedicated ANN index) would be introduced as an **additional** component alongside, not replacing, this streaming feature/model-serving core — out of scope here since it changes the problem to embedding-similarity retrieval rather than streaming feature/label online learning.

## 17. Embedding Pipelines

**N/A for this system's baseline scope**, for the same reason as Section 16 — features here are hand-engineered windowed aggregates (counts, ratios, rates) rather than learned dense embeddings. If a future extension adds player/match embeddings (e.g., learned via a two-tower model for matchmaking quality), an embedding pipeline would sit upstream of the online feature store, periodically (or streaming-) recomputing embeddings and writing them as additional feature-store fields — architecturally additive, not a redesign.

## 18. Inference Pipelines (Request Lifecycle End-to-End)

```
[Game Server] --(1) telemetry event--> [Kafka: telemetry.raw]
                                              │
                                    (2) Flink windowed agg
                                              │
                                              ▼
                                  [Online Feature Store write]
                                              │
[Game Server] --(3) needs anti-cheat decision--> [Inference Service]
                                              │
                       (4) GetFeatures(player_id, "anticheat_v3")
                                              │  <-- p99 15ms
                                              ▼
                              [Online Feature Store read]
                                              │
                       (5) fetch cached model pointer (local, ~0ms)
                                              │
                       (6) model.predict(feature_vec)  <-- ~0.5-2ms
                                              │
                       (7) apply business rule (threshold/hysteresis)
                                              │
                       (8) log prediction+features (async, non-blocking)
                                              │
[Game Server] <--(9) decision: ALLOW/FLAG/KICK-- [Inference Service]
                                              │
                       (10) publish to prediction.events (async, for
                            downstream matchmaking rebalancer / audit)
```

**Latency budget (p99 100ms target)**:
- Network client→gateway→inference: 20ms
- Feature store read: 15ms
- Model inference (CPU, GBT): 3ms
- Business-rule + serialization: 2ms
- Response network: 20ms
- **Buffer/queueing headroom**: ~40ms (absorbs GC pauses, connection pool contention, cross-AZ hops)

## 19. Training Pipelines

- **Data prep**: `flink-label-join` job continuously joins `labels.confirmed` with historical `prediction_log.feature_snapshot` (matched by trace_id or entity_id+time-window), emitting labeled rows to `training.materialized.{title}`.
- **Orchestration**: `streaming-trainer` service (per title) consumes `training.materialized`, accumulates a rolling window (last 2M rows or last 60s, whichever first), and runs:
  - GBT: incremental boosting rounds (warm-start from last checkpoint) via XGBoost's `xgb_model=` continuation, not full retrain.
  - Linear/bandit: closed-form or SGD warm-started update (Thompson-sampling posterior update for contextual bandit).
- **Distributed training**: Not required at this scale (models are small, <50MB, training set per micro-batch is a few million rows) — single-node incremental fit per title is sufficient; distributed training (Horovod/Spark MLlib) reserved for the periodic **full offline retrain** (nightly/weekly) that rebuilds the model from scratch on the full historical Iceberg dataset to correct for online-update drift accumulation.
- **Nightly full retrain**: Spark job over Iceberg tables, distributed across a 20-40 node Spark cluster, produces a fresh baseline model that the online-trainer then continues to incrementally update through the next day — bounds long-run divergence from pure online updates.

## 20. Retraining Strategy

| Trigger Type | Cadence/Condition | Action |
|---|---|---|
| Scheduled (baseline) | Every 30-60s micro-batch | Incremental online update pushed to `shadow` stage automatically |
| Scheduled (full rebuild) | Nightly (per title, off-peak region hours) | Full Spark retrain from Iceberg, replaces `production` baseline after offline eval gate |
| Data-drift triggered | PSI > 0.2 on any top-10 feature (see Sec. 21) | Force early full retrain, page ML on-call |
| Concept-drift triggered | Live AUC/precision drop > 5% vs. 7-day rolling baseline | Halt online-update promotion to production; fall back to last-known-good model; alert |
| Label-volume triggered | Labeled event rate drops >50% (e.g., anti-cheat review backlog) | Pause online trainer (avoid overfitting to sparse/biased labels), alert |
| Manual | ML engineer requests via Model Registry API | Ad-hoc retrain job, requires approval to promote |

- Promotion gate: every online-updated model version enters `shadow` automatically; promotion to `production` requires either (a) automated check passing 15-min shadow evaluation window with no regression beyond threshold, or (b) human approval for the nightly full-retrain baseline swap.

## 21. Drift Detection

| Drift Type | Metric | Threshold | Action |
|---|---|---|---|
| Feature (data) drift | Population Stability Index (PSI) per feature, computed hourly over sliding 24h window vs. 7-day-prior baseline | PSI > 0.2 = alert (moderate), PSI > 0.3 = force retrain | Alert ML on-call; auto-trigger full retrain at 0.3 |
| Feature drift (categorical) | Chi-squared / Jensen-Shannon divergence on categorical feature distributions | JSD > 0.1 | Alert |
| Concept drift | Rolling 7-day AUC/PR-AUC (anti-cheat), rolling win-rate-quality-correlation (matchmaking) | Drop > 5% relative vs. trailing baseline | Halt promotion, fallback model, alert |
| Prediction drift | Distribution of prediction scores (mean, p50/p95) hourly vs. baseline | Mean shift > 0.15 absolute (on 0-1 score) | Alert, investigate before drift confirmed as concept vs. upstream bug |
| Label delay/skew | Time between prediction and label arrival, and label positive-rate | Label latency p90 > 2x historical median | Alert (may indicate broken confirmation pipeline, silently degrading online learning) |
| Online-vs-offline divergence | Diff between online-trained model's shadow predictions and nightly full-retrain baseline predictions on same holdout | KL divergence > 0.05 | Investigate online-update drift accumulation; consider capping online-update magnitude |

## 22. Monitoring

| Layer | What's Monitored |
|---|---|
| Infra | Kafka broker lag (per consumer group), partition skew, Flink checkpoint duration/failures, task-manager backpressure, Redis/Scylla p99 latency + memory pressure, inference pod CPU/mem/QPS |
| Model quality | Rolling AUC/precision/recall (anti-cheat), calibration (reliability diagrams), bandit regret estimate (matchmaking), shadow-vs-production agreement rate |
| Data pipeline | Event ingestion rate vs. expected (per title, per region), schema-validation rejection rate, DLQ depth, label-join match rate (% of predictions eventually labeled) |
| Business | False-positive ban rate (player complaints/appeals), match-quality NPS proxy, conversion rate on personalized offers, revenue impact of pricing model |
| Freshness | Feature-store write-to-read staleness (p50/p99), model-update propagation latency to inference pods |

- Dashboards: Grafana (Prometheus + Kafka Exporter + Flink metrics reporter), plus a dedicated "ML Health" dashboard combining drift scores + business KPIs for weekly model-review meetings.

## 23. Alerting

| Alert | Condition | Severity | Routing |
|---|---|---|---|
| Kafka consumer lag | Lag > 30s sustained for 2min on feature-agg consumer group | P1 | Page streaming on-call |
| Flink job restart loop | >3 restarts in 10min | P1 | Page streaming on-call |
| Inference p99 latency | > 150ms for 5min | P1 | Page serving on-call |
| Inference error rate | > 1% for 5min | P1 | Page serving on-call |
| Feature staleness | p99 > 5s for 5min | P2 | Page ML on-call |
| PSI drift threshold breach | PSI > 0.3 | P2 | Alert ML on-call (business hours if <0.3, page if severe) |
| Concept drift (AUC drop) | > 5% relative drop, confirmed over 3 consecutive hourly windows | P1 | Page ML on-call, auto-fallback triggered |
| DLQ depth | > 1000 messages or > 0.1% of topic volume | P2 | Alert data-platform on-call |
| Model promotion failure | Shadow eval gate fails 3x consecutively | P3 | Ticket to ML team, no page |
| Purchase topic transactional write failure | Any failure after retries | P1 | Page payments + streaming on-call jointly |

- Routing via PagerDuty; P1 = page immediately 24/7, P2 = page during business hours / page after 30min unacked, P3 = ticket only.
- Alert fatigue control: PSI/drift alerts deduped and aggregated per title per hour to avoid noisy paging during known live-events (which naturally shift distributions).

## 24. Logging

- **Structured logging**: JSON logs with mandatory fields — `trace_id`, `title`, `entity_id_hash` (not raw ID, see PII below), `service`, `timestamp`, `latency_ms`, `model_version`.
- **PII handling**:
  - Raw player IDs and any PII (chat text, real names, payment tokens) never written to general application logs — only to the dedicated, access-controlled `prediction_log` table with field-level encryption for chat/text content.
  - `entity_id` in logs is a salted hash for correlation without exposing raw player ID to log-viewing engineers; reverse lookup requires a separate, audited service.
  - Chat/toxicity event text is redacted in Flink processing logs, retained only in the encrypted prediction-log store with restricted RBAC (moderation team + auditors only).
  - COPPA-flagged accounts (known minors) get stricter retention: telemetry logs auto-purged at 30 days instead of 90.
- **Retention**: Application/infra logs 14 days (Loki/ELK); prediction/feature audit logs 30 days hot (Iceberg), 1 year cold-archived (Glacier-class) for compliance and long-horizon offline eval; purchase-related logs 7 years (financial compliance).
- **Correlation**: every log line, span, and metric carries `trace_id` propagated from the initial game-client request through Kafka headers into Flink and back into the inference response — enables full request reconstruction across the async boundary.

## 25. Security

- **Threat model specific to this system**:
  - Cheaters reverse-engineering the anti-cheat model via repeated probing (model extraction / adversarial evasion) — mitigate via rate limiting on `/predict` per player, response minimalism (return decision, not raw score, to end clients), and periodic feature/model rotation.
  - Compromised game-server credentials injecting fabricated telemetry to poison online-learning updates (data poisoning attack on the streaming trainer) — mitigate via per-source anomaly detection on event volume/pattern before events enter the aggregation pipeline, and capping the influence of any single entity_id's events on a micro-batch update.
  - Insider/lateral-movement risk to the online feature store (contains behavioral profiles) — mitigate via network segmentation (feature store not internet-reachable), mTLS between services, least-privilege IAM.
  - Label-source spoofing (fake "ban confirmed" events) manipulating the model into learning wrong associations — mitigate via signed label events (only the moderation service's signed producer credentials accepted on `labels.confirmed`).
- **Encryption**: TLS 1.3 in transit everywhere (client↔gateway, service↔service via mTLS, Kafka inter-broker and client encryption enabled); at rest — Kafka topic encryption for `purchases.*`/`chat.events.*`, KMS-managed keys, Iceberg tables encrypted via S3-SSE-KMS, Redis with encryption-at-rest enabled for feature store.
- **Data minimization**: feature vectors avoid storing raw chat text/PII directly as features — text is scored by a separate toxicity classifier upstream, only the resulting numeric score enters the feature store.

## 26. Authentication

- **Service-to-service**: mTLS with short-lived certs issued by internal PKI (e.g., SPIFFE/SPIRE), rotated every 24h; every internal call (inference→feature-store, Flink→Kafka, trainer→registry) authenticated this way.
- **Game client/server → Edge Gateway**: Game servers (trusted, EA-operated) authenticate via service tokens (OAuth2 client-credentials grant, scoped per title); game clients (less trusted, run on player devices) never call the ML platform directly — all telemetry routes through the EA-operated game server first, which is the actual authenticated caller. This avoids exposing the ML platform's ingestion API to untrusted client binaries.
- **Human/operator access** (Model Registry promote, drift dashboards): SSO (Okta/OIDC) + RBAC — roles: `ml-engineer` (promote to shadow), `ml-lead` (promote to production), `auditor` (read-only on prediction logs with PII redaction).

## 27. Rate Limiting

- **Algorithm**: Token bucket per (title, player_id) at the Edge Gateway for telemetry ingestion — burst capacity 50 events, refill 20/sec, generously above the 15/min average but bounds abusive clients (e.g., a cheat tool spamming events to try to manipulate windowed aggregates).
- **Inference `/predict` endpoint**: Sliding-window counter per calling game-server instance (not per player, since callers are game servers) — 50K req/sec per server-fleet-shard, matched to expected legitimate traffic; exceeding it returns `429` with backoff hint, protects against a misbehaving game-server build hammering inference.
- **Model Registry `/promote`**: Simple fixed low limit (10/min per user) — human-driven action, limiting is about accident prevention, not load protection.
- **Per-tenant (per-title) limits**: each title gets a provisioned QPS quota on shared Kafka/Flink/inference infra to prevent one title's traffic spike (e.g., a new FIFA drop event) from starving another title's SLA — enforced via Kafka quotas (broker-level per-client-id byte-rate quotas) and per-title inference pod pool reservations (see Section 28).

## 28. Autoscaling

- **Inference Service**: Kubernetes HPA on custom metric `inference_qps_per_pod` (target 800 QPS/pod) plus CPU utilization (target 60%) as a secondary signal; scale range 20-200 pods per title-shard; scale-up cooldown 30s (fast, since traffic spikes at live-events are sharp), scale-down cooldown 5min (avoid flapping).
- **Flink Task Managers**: Not HPA-based (stateful, rebalancing keyed state is expensive) — instead, reactive scaling via Flink's Adaptive Scheduler, triggered on sustained backpressure/consumer-lag metric crossing threshold for 5min, with a controlled rescale (checkpoint, redistribute key groups, resume) rather than abrupt pod churn.
- **Kafka brokers**: Manually provisioned with headroom (not autoscaled — broker addition requires partition reassignment, done as a planned capacity operation ahead of known live-events, e.g., a title launch).
- **Online Feature Store (Redis/Scylla)**: VPA-style vertical headroom monitoring + manual cluster resize playbook for planned events; Scylla supports online node addition for organic growth.
- **KEDA** used for the async `prediction.events` consumers (pricing/matchmaking rebalancer) — scale based on Kafka consumer lag directly (KEDA Kafka scaler), 0-to-N capable during quiet periods to save cost.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-service-apex
  namespace: ml-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-service-apex
  minReplicas: 20
  maxReplicas: 200
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 300
  metrics:
    - type: Pods
      pods:
        metric:
          name: inference_qps_per_pod
        target:
          type: AverageValue
          averageValue: "800"
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
```

## 29. Cost Optimization

- **Spot/preemptible instances**: Flink task managers on spot (stateful but checkpointed to S3 every 10s — a spot reclaim costs at most a 10-30s recovery, acceptable); nightly full-retrain Spark cluster entirely on spot (batch, restartable). Inference service kept on-demand (latency-critical, spot reclaim would violate p99 SLA).
- **Batching for async path**: pricing/matchmaking async predictions batched (50-100 entities/call) — amortizes per-call overhead, cuts vCPU-seconds ~30% vs. per-entity calls.
- **Model distillation/size**: GBT models pruned/quantized (float32→float16 where framework supports) to cut memory footprint and inference vCPU-ms; matchmaking bandit model kept linear (cheapest possible) rather than a deeper model given marginal AUC gain didn't justify 3x compute.
- **Tiered Kafka storage**: hot local-disk retention cut to 24h, older data tiered to S3-compatible object storage — cuts broker disk/node count roughly in half vs. naive 7-day local retention.
- **Cache-aside on model/feature reads**: reduces Redis/Scylla read QPS load, allowing a smaller cluster than naive 750K QPS provisioning would require.
- **Right-sized online-store TTLs**: 6-hour TTL on inactive-player feature keys prevents unbounded memory growth charging for stale entities.
- **Per-title autoscale floors tuned to actual traffic share** (Apex gets a higher min-replica floor than a lower-traffic title) rather than uniform floors across all 8 titles.

## 30. Disaster Recovery

- **RTO**: 15 minutes for inference path (can fail over to secondary region or degrade to stale-feature/fallback-model mode faster than a full region rebuild). 4 hours for full Flink state rebuild from Kafka replay in a from-scratch DR region.
- **RPO**: Near-zero for Kafka (replicated across 3 AZs synchronously within region; cross-region async replication with <60s lag target for the standby region). Feature store RPO ~2s (acceptable per freshness NFR — losing 2s of feature updates is tolerable, unlike losing a financial transaction).
- **Backup strategy**:
  - Kafka: cross-region MirrorMaker2 replication of all topics to a DR region, async.
  - Flink: checkpoints to S3 (cross-region replicated bucket); a DR-region Flink cluster can resume from the latest checkpoint + replay Kafka from the checkpoint offset.
  - Model Registry (Postgres): continuous WAL shipping to standby + daily snapshot; model artifacts already durably in cross-region-replicated blob storage.
  - Online feature store: NOT backed up in the traditional sense (rebuild via Flink replay is faster and cheaper than snapshot/restore of a 7GB hot cache) — treated as a rebuildable cache, not a source of truth.
- **Runbook**: DR drills quarterly — simulate region loss, verify Flink resumes from checkpoint + Kafka replay within RTO, verify inference fails over to DR region's model registry replica.

## 31. Multi-Region Deployment

- **Topology**: Active-active across 3 regions (US-East, EU-West, APAC), each region serving its local player population — chosen because inference is latency-critical (p99 100ms) and cross-region round-trips alone can exceed that budget.
- **Data residency**: EU player telemetry/features/predictions stay in EU-West (GDPR) — enforced at the Edge Gateway via routing rules keyed on account region, not just network geo-IP (a player traveling shouldn't cause their persistent data to relocate).
- **Replication**:
  - Kafka: each region has its own independent cluster (no cross-region synchronous writes — would blow the latency budget); MirrorMaker2 async replication feeds a global aggregate topic (for cross-region model training on global patterns, e.g., global anti-cheat model trained nightly on unified nightly Iceberg data, with EU rows access-restricted per GDPR).
  - Model Registry: Postgres primary in US-East with read replicas in EU-West/APAC; promotions replicate async (~1-5s lag) — a newly promoted model is visible platform-wide within seconds, acceptable since promotion is not a per-request hot path.
  - Online feature store: independent per-region cluster, no cross-region replication (a player's session/features are region-local by construction since they connect to their nearest game server).
- **Latency routing**: GeoDNS + Anycast routes game-server-to-ML-platform calls to the nearest healthy region; health-check-based failover reroutes a region's traffic to the next-nearest region if local inference SLA breaches for >2min (accepting a residency exception only for true DR failover, logged/flagged for compliance review).

```
        ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
        │  US-East     │        │  EU-West     │        │  APAC        │
        │  (active)    │        │  (active)    │        │  (active)    │
        │ Kafka+Flink  │        │ Kafka+Flink  │        │ Kafka+Flink  │
        │ +Inference   │        │ +Inference   │        │ +Inference   │
        └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
               │  async MirrorMaker2 replication (aggregate/global topics)
               └────────────────────┴────────────────────────┘
                                     │
                         ┌───────────────────────┐
                         │ Global Nightly Retrain  │
                         │ (Spark, region-aware,   │
                         │  GDPR row-level filter) │
                         └───────────────────────┘
      GeoDNS/Anycast routes each game-server to nearest healthy region;
      failover reroutes on sustained SLA breach (>2min), logged for compliance.
```

## 32. Blue/Green Deployment

- Applies primarily to the **Inference Service** and **Flink job graph** deployments (not the online-learning model versions, which use canary/shadow instead — see Section 33).
- Inference Service: new code version (e.g., a serialization format change, a new feature-set schema support) deployed as a fully separate "green" fleet behind the same load balancer target group; traffic cut over via LB weight change from 100/0 to 0/100 once green passes smoke tests (schema compatibility, synthetic request replay); blue fleet kept warm for 30 min post-cutover for instant rollback.
- Flink job graph changes (e.g., new window logic): blue/green via **parallel pipeline run** — green Flink job consumes the same Kafka topics from the current offset (not a fresh replay) in parallel with blue, writes to a separate `feature.updates.green` output/online-store namespace; once validated (feature values match expected sanity checks over a burn-in window), inference service's feature-set pointer is cut over to green's namespace, then blue job is cancelled.

## 33. Canary Deployment

- **Model canary** (distinct from code blue/green): every online-trainer-produced model version auto-enters `shadow` stage (0% live traffic, predictions logged not returned). After shadow burn-in (15 min, or a configurable N-labeled-events threshold), it's eligible for canary.
- **Canary traffic split**: 5% of a title's inference traffic routed to the canary model version for 30 min, selected by consistent hashing on entity_id (same players stay on canary throughout the window, avoiding flip-flopping decisions for a given player mid-session).
- **Health-check gates to progress canary → production**:
  1. Error rate on canary path ≤ baseline + 0.1%.
  2. Latency p99 on canary path within 10% of baseline.
  3. Business-metric proxy (false-positive-flag rate for anti-cheat, or offer-conversion rate for pricing) not regressed beyond a pre-registered threshold (e.g., FP rate +0.5% absolute triggers auto-abort).
  4. Drift/divergence check (Section 21's online-vs-offline divergence metric) within bounds.
- On all gates passing, ramp 5% → 25% → 100% over the next hour; any gate failure at any ramp stage triggers automatic rollback to the prior production version (see Section 34).

## 34. Rollback Strategy

- **Automated triggers**:
  - Canary health-check gate failure (Section 33) → auto-rollback to last production model version, no human in the loop, completes within 1 model-registry-pointer-update propagation cycle (~5-10s to inference pods via `model.updates` topic).
  - Inference Service error-rate/latency SLO breach post-code-deploy → automated rollback via deployment controller (Argo Rollouts) reverting to prior ReplicaSet.
  - Concept-drift alarm (Section 21, P1) → automatic fallback to last-known-good model version pinned in registry as `stable`, bypassing the currently-promoted (possibly drifting) production version.
- **Rollback mechanics**:
  - Model rollback = pointer change in Model Registry (`production` stage tag moves back to prior version ID) + `model.updates` notification — no redeploy needed, inference pods hot-swap in-memory model within seconds.
  - Code rollback = standard Kubernetes/Argo Rollouts revert to previous ReplicaSet revision.
  - Flink job rollback = resume blue job graph from its last valid checkpoint (kept warm during green burn-in per Section 32), cancel green.
- **Always keep N-1 model version warm/cached** in every inference pod (not just latest) specifically to make rollback a zero-fetch-latency operation.

## 35. Observability

- **Metrics** (Prometheus): infra (Kafka lag, Flink checkpoint duration, pod CPU/mem/QPS), model-quality (rolling AUC, calibration), business (FP rate, conversion) — all discussed in Section 22, surfaced with consistent `title`/`model_version`/`region` labels for slicing.
- **Tracing** (OpenTelemetry, spans exported to Jaeger/Tempo): a single `trace_id` spans the full async chain — client request → Edge Gateway → Kafka produce span → Flink processing span (via trace-context propagation in Kafka headers) → feature-store write → inference request → feature-store read → model predict → response. This is the hardest part of observability here because the pipeline crosses sync/async boundaries; trace context is carried in Kafka message headers to stitch spans across the streaming hop.
- **Logs**: correlated to traces via shared `trace_id` field (Section 24); log aggregation in Loki, queryable by trace_id to reconstruct exactly what features/model version produced a given decision — critical for anti-cheat appeal investigations ("why was this player flagged?").
- **Three pillars tied together**: a drift alert (metric) links to a Grafana panel showing the affected `model_version`, which links to a saved trace query showing sample predictions from that window, which links to the structured log entries with full feature snapshots for those predictions — enabling a single on-call engineer to go from "AUC dropped" to "here's the specific feature that's out of distribution" in one investigation flow.

## 36. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service-apex
  namespace: ml-serving
  labels: { app: inference-service, title: apex }
spec:
  replicas: 20
  selector:
    matchLabels: { app: inference-service, title: apex }
  template:
    metadata:
      labels: { app: inference-service, title: apex }
    spec:
      containers:
        - name: inference
          image: registry.ea.internal/ml/inference-service:2026.07.08
          resources:
            requests: { cpu: "2", memory: "2Gi" }
            limits:   { cpu: "4", memory: "4Gi" }
          ports:
            - containerPort: 8080
          env:
            - name: TITLE
              value: "apex"
            - name: FEATURE_STORE_ADDR
              value: "redis-apex.ml-serving.svc.cluster.local:6379"
            - name: MODEL_REGISTRY_ADDR
              value: "model-registry.ml-platform.svc.cluster.local:9090"
          readinessProbe:
            httpGet: { path: /healthz, port: 8080 }
            initialDelaySeconds: 5
            periodSeconds: 5
          livenessProbe:
            httpGet: { path: /healthz, port: 8080 }
            initialDelaySeconds: 15
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: inference-service-apex
  namespace: ml-serving
spec:
  selector: { app: inference-service, title: apex }
  ports:
    - port: 443
      targetPort: 8080
  type: ClusterIP
```

## 37. Terraform Infrastructure

```hcl
# Kafka cluster (MSK-style managed Kafka) for the streaming ingest layer
resource "aws_msk_cluster" "telemetry" {
  cluster_name           = "ml-telemetry-us-east"
  kafka_version          = "3.6.0"
  number_of_broker_nodes = 36

  broker_node_group_info {
    instance_type   = "kafka.m5.2xlarge"
    client_subnets  = var.private_subnet_ids
    security_groups = [aws_security_group.kafka.id]
    storage_info {
      ebs_storage_info { volume_size = 2000 }
    }
  }

  encryption_info {
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
  }

  tags = { Environment = "prod", System = "streaming-ml-platform" }
}

# Redis (online feature store) - ElastiCache
resource "aws_elasticache_replication_group" "feature_store" {
  replication_group_id       = "ml-feature-store-apex"
  description                = "Online feature store for Apex anti-cheat/matchmaking"
  node_type                  = "cache.r6g.2xlarge"
  num_node_groups            = 8
  replicas_per_node_group    = 2
  automatic_failover_enabled = true
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  engine_version             = "7.1"
}

# EKS node group for Flink task managers, spot-provisioned
resource "aws_eks_node_group" "flink_taskmanagers" {
  cluster_name    = aws_eks_cluster.ml_platform.name
  node_group_name = "flink-tm-spot"
  node_role_arn   = aws_iam_role.flink_node_role.arn
  subnet_ids      = var.private_subnet_ids
  capacity_type   = "SPOT"
  instance_types  = ["m5.2xlarge", "m5a.2xlarge", "m5n.2xlarge"]

  scaling_config {
    desired_size = 40
    min_size     = 20
    max_size     = 100
  }
}
```

## 38. Why This Architecture

- Decouples **feature computation** (Flink, stateful, windowed) from **serving** (stateless inference pods reading a materialized online store) — each scales independently on its own bottleneck (CPU-bound windowing vs. QPS-bound point lookups).
- Kafka as the universal backbone gives replay-ability (critical for backfills, bug recovery, and reconstructing training sets) that a pure request/response pipeline wouldn't offer.
- Logging the exact feature snapshot at prediction time (rather than re-deriving historically) sidesteps the single hardest correctness bug in streaming ML systems — train/serve skew from a feature store's mutable "current state."
- Micro-batch online learning (30-60s) balances the product need for fast adaptation (matchmaking/anti-cheat signals shift within minutes during a live event) against the operational stability of per-event SGD, which is fragile to noisy/adversarial single-event updates in a system attackers actively probe.
- Shadow/canary gating on every online-trained model version protects against the unique risk of continuous learning: a bad batch of labels (or a poisoning attempt) silently degrading production without a human in the loop, if promotion were fully automatic and ungated.

## 39. Alternative Architectures

| Alternative | Description | Why Rejected / When Preferred |
|---|---|---|
| Pure batch pipeline (hourly/daily feature + retrain) | Traditional Spark batch ETL + periodically retrained model behind a REST endpoint | Rejected: cannot meet <2s feature-freshness or fast anti-cheat reaction requirement; would be preferred if the use case were e.g. weekly matchmaking-tier recalculation with no real-time need |
| Per-event synchronous SGD online learning | Update model weights on every single labeled event as it arrives, no micro-batching | Rejected as default: too sensitive to label noise/adversarial single-event poisoning, harder to gate/canary at per-event granularity; would be preferred for a very low-traffic, low-stakes personalization signal where instant adaptation outweighs stability risk |
| Lambda architecture (separate batch + speed layers merged at query time) | Maintain both a batch-computed feature layer and a streaming speed layer, merge views at read time | Considered: adds real value for long-window features (24h+) computed more cheaply in batch; rejected as the *sole* architecture because merge-at-query-time adds serving-path latency and complexity; instead adopted selectively — nightly Spark full-retrain acts as the "batch layer" baseline that the streaming trainer incrementally builds on (a lightweight Lambda-inspired hybrid, not full Lambda) |
| Fully synchronous request-scoped feature computation (compute features on-demand per inference call, no precomputed store) | Skip the online feature store; compute windowed aggregates on the fly from raw event history per request | Rejected: recomputing a 24h window per request at 750K QPS is computationally infeasible within the latency budget; would only be viable for very low QPS, low-window-size use cases |

## 40. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Micro-batch (30-60s) online learning vs. per-event SGD | Stability, easier to gate/canary, resilient to single noisy labels | Slower adaptation than true per-event learning; a very fast-evolving cheat pattern has up to ~60s+shadow-burn-in lag before full mitigation |
| At-least-once for telemetry vs. exactly-once everywhere | Simpler, cheaper, higher throughput for non-financial data | Requires idempotent aggregation logic (dedup by event_id) everywhere in Flink state, added engineering complexity |
| Redis/Scylla online store (no history) + Iceberg offline store (full history) | Cheap, fast serving; full audit/backfill capability preserved separately | Two systems to keep schema-consistent; a feature-set version bump requires coordinated migration across both |
| Region-local Kafka/Flink/inference (no cross-region sync) | Meets latency SLA, respects data residency | No single global view without async replication lag; global model training must tolerate slightly stale/incomplete cross-region data |
| Shadow/canary gating on every online model update | Prevents silent production degradation from bad label batches | Adds latency-to-production for genuinely good updates (minimum ~45min from shadow to full ramp) |
| CPU-only serving (GBT/linear models) | No GPU cost/ops burden, simple autoscaling | Caps future model complexity (can't casually add a deep/embedding model without new hardware planning) |

## 41. Failure Modes

| Failure | Symptom | Mitigation |
|---|---|---|
| Flink job falls behind (backpressure from a partition hot-key, e.g., a viral streamer's match) | Feature staleness spikes past 2s SLA for affected entities | Key-salting for hot keys; inference falls back to last-known-good cached feature value with a staleness flag rather than blocking; alert if staleness > 5s sustained |
| Kafka broker/AZ outage | Producer errors, consumer lag spike | Multi-AZ replication (min.insync.replicas=2), automatic leader re-election; Edge Gateway buffers/retries with backoff on producer failure |
| Online feature store node failure | Elevated read latency/errors for entities sharded to that node | Redis Cluster/Scylla replica promotion (automatic); inference falls back to a default/neutral feature vector with a "degraded" flag rather than failing the request outright, feeding a conservative default decision (e.g., don't auto-kick on missing anti-cheat features, only flag for review) |
| Bad label batch (e.g., moderation tool bug marks innocent players as banned) | Online trainer learns corrupted association | Shadow/canary gate should catch the resulting model quality regression before full promotion; additionally, label-source anomaly detection (sudden spike in ban-confirmation rate) triggers an independent alert |
| Model registry unavailable | Inference pods can't fetch new model versions | Pods retain last successfully loaded model in memory indefinitely (registry is only consulted for updates, not per-request) — serving continues uninterrupted on stale-but-functional model |
| Cross-region replication lag spike (MirrorMaker2) | Global nightly retrain sees incomplete cross-region data | Nightly job waits for a replication-lag-below-threshold signal before kicking off, or proceeds with a logged "partial data" flag on the resulting model version |
| Poisoning attack via fabricated telemetry | Online-trained model quality degrades against real player population | Per-entity update-influence capping in the streaming trainer (no single entity_id can dominate a micro-batch's gradient contribution); anomaly detection on event-source patterns |

## 42. Scaling Bottlenecks

- **At 10x traffic (25M events/sec)**: Kafka partition count and broker fleet become the first bottleneck — current 200-partition topics would need re-partitioning (a disruptive operation) well before 10x; plan proactively for higher partition counts from day one even if underutilized initially. Flink task-manager count scales roughly linearly (~3,200 vCPUs), still feasible but requires state-backend (RocksDB) tuning for larger per-key state and more frequent checkpoint tuning to avoid checkpoint-duration creep.
- **At 100x traffic (250M events/sec)**: This breaks the single-region-cluster model entirely — would require sharding Kafka/Flink/feature-store horizontally *within* a region (multiple parallel pipeline instances per title, not just more brokers), essentially a re-architecture into per-title-shard sub-clusters. The online feature store's 750K QPS at baseline becomes 75M QPS at 100x — no single Redis/Scylla cluster design handles that; would need a multi-cluster sharded-by-player-cohort feature store with a routing layer, a significant new component.
- **Model registry (Postgres)** is a likely bottleneck far before compute scale — a single-writer relational store for version metadata is fine at current promotion frequency (a handful of promotions/min across 8 titles) but would need to move to a more horizontally scalable metadata store if promotion frequency or title count grew by 10-100x.
- **Online-trainer per-title single-node incremental fit** — at 10x label volume, a single-node fit within the 30-60s micro-batch window may no longer complete in time; would need to shard the label stream by entity-cohort and run parallel partial-model updates with periodic merge (mirrors distributed online-learning literature, e.g., parameter-server pattern) — real added complexity at that scale.

## 43. Latency Bottlenecks

**p50/p99 budget breakdown for synchronous anti-cheat inference (target p99 100ms)**:

| Stage | p50 | p99 |
|---|---|---|
| Client/game-server → Edge Gateway network | 5ms | 15ms |
| Edge Gateway auth + routing | 1ms | 3ms |
| Gateway → Inference Service network (intra-region) | 2ms | 5ms |
| Feature store `GetFeatures` read | 4ms | 15ms |
| Model inference (GBT forward pass) | 1ms | 3ms |
| Business-rule application + response serialization | 0.5ms | 2ms |
| Inference Service → Gateway → client response network | 5ms | 15ms |
| Queueing/scheduling jitter (GC, connection pool, thread contention) | 2ms | 30ms |
| **Total** | **~20ms** | **~88ms** |

- The dominant p99 tail contributor is **queueing/scheduling jitter** (GC pauses in JVM-based components, connection-pool contention under load) and **feature-store read tail latency** (Redis/Scylla p99 under high fan-in load) — both are where engineering effort on tail-latency reduction (e.g., request hedging on feature-store reads, GC tuning or moving hot-path services to Go/Rust) pays off most.
- Cross-AZ hops (if inference pod and feature-store shard land in different AZs) can silently add 1-2ms each — pod anti-affinity/topology-aware routing keeps these co-located where possible.

## 44. Cost Bottlenecks

- **Kafka broker fleet** (36+ nodes, 3x replication, cross-region MirrorMaker2 replication) is typically the single largest line item in a streaming-first architecture — replication factor and cross-region async replication both multiply storage/network cost; tiered storage (Section 29) is the primary lever.
- **Online feature store (Redis/Scylla) node count**, sized for 750K QPS with headroom — over-provisioning "just in case" for live-event spikes (6M CCU events) sized year-round rather than elastically is a common silent cost sink; right-sizing to actual sustained (not peak) load with a documented burst-capacity plan for known live-events reduces this.
- **Cross-region data transfer** (MirrorMaker2, global nightly retrain data movement) — egress costs from moving raw telemetry/feature data between regions for the "global" model training step; mitigated by aggregating/pre-filtering (only send the columns needed for global training, not raw telemetry) before cross-region transfer.
- **Nightly full-retrain Spark cluster**, even on spot, run across 8 titles nightly — if titles' data volumes are uneven, a shared elastic cluster sized for the largest title but running for all 8 sequentially wastes idle capacity; per-title right-sized ephemeral clusters (spin up/down) are more cost-efficient than one large shared cluster.
- **Prediction/feature audit logging** at prediction-QPS scale (750K/sec) into Iceberg — the write amplification and downstream storage (80TB/month compressed) is a recurring, easy-to-overlook cost; retention-tier discipline (Section 24) and columnar compression are the main levers.

## 45. Interview Follow-Up Questions

1. How would you detect and mitigate a bad-actor game server injecting fabricated telemetry to poison the online learning loop?
2. Walk through exactly how you guarantee point-in-time correctness between the feature vector used for a prediction and the feature vector later used to build the training set from a confirmed label.
3. What happens if the online feature store returns a value that's 10 seconds stale — how does the inference service know, and what does it do?
4. Why micro-batch (30-60s) online learning instead of true per-event streaming SGD? What would change your recommendation?
5. How do you prevent an online-trained model from drifting away from the nightly full-retrain baseline over the course of a day, and how would you detect if it had?
6. Your Flink job's checkpoint duration has crept from 5s to 45s over the past week — walk through your diagnosis process.
7. How would you extend this system to support an embedding-based player-similarity model for smurf detection, and what would you need to add (referencing Sections 16/17)?
8. Design the canary gating logic in more detail — what statistical test would you use to decide "regressed" vs. "noise" on a 5% traffic sample over 30 minutes?
9. How does data residency (GDPR) change your multi-region topology, and what happens when a global model needs to be trained across regions?
10. If Kafka partition count is your first bottleneck at 10x scale, why not just over-provision partitions from day one — what's the tradeoff?

## 46. Ideal Answers

1. **Poisoning mitigation**: Cap any single entity_id's gradient/statistical influence on a micro-batch update (influence capping/clipping), run per-source anomaly detection on event volume and pattern *before* events enter the aggregation pipeline (e.g., z-score on events/sec per source), require signed producer credentials on the `labels.confirmed` topic so only the moderation service can assert ground truth, and rely on the shadow/canary gate to catch any resulting model-quality regression before it reaches production traffic — defense in depth rather than a single control.

2. **Point-in-time correctness**: Log the exact feature vector used at prediction time into `prediction_log.feature_snapshot`, keyed by `trace_id`. When a label arrives later (e.g., `labels.confirmed`), the label-join Flink job joins the label against that *logged snapshot*, not against a fresh query to the (by-then-mutated) online feature store. This guarantees the training example reflects what the model actually saw, avoiding future-information leakage that would inflate offline eval metrics relative to true production performance.

3. **Stale feature handling**: Every `GetFeatures` response includes a `freshness_ms` field. The inference service checks this against a per-feature-set staleness threshold; if breached, it either (a) falls back to a cached last-known-good value with a "degraded" flag propagated into the decision logic (e.g., anti-cheat defaults to "flag for human review" instead of "auto-kick" when features are stale), or (b) for use cases where stale features are unacceptable, returns a fail-open/fail-closed decision per business policy, logged distinctly from a normal-confidence prediction for later audit.

4. **Micro-batch vs. per-event SGD**: Per-event updates are maximally reactive but fragile — a single noisy or adversarial label can swing model weights, and there's no natural unit of work to gate/canary (you'd be canarying every single update, operationally infeasible). Micro-batching gives a stable, gateable unit of change every 30-60s, which is fast enough for the product's actual reaction-time requirements (anti-cheat review, matchmaking rebalancing) while remaining operable. I'd reconsider per-event only for a very low-stakes, low-traffic signal where instant reactivity clearly dominates stability concerns (e.g., a single-player difficulty slider with no adversarial pressure).

5. **Online-vs-baseline divergence**: Track KL divergence (or simpler: prediction-distribution delta) between the currently-serving online-updated model and the last nightly full-retrain baseline, evaluated on a shared holdout set, computed hourly. A threshold breach (e.g., KL > 0.05) triggers an alert and can auto-force an early full retrain rather than waiting for the nightly cadence — this bounds how far "continuous learning drift" can accumulate before being corrected against ground truth.

6. **Checkpoint duration diagnosis**: Check for (a) state size growth — is a key's state unbounded due to a missing TTL/window-close bug; (b) key skew — one hot key (viral event) holding disproportionate state slowing the checkpoint barrier alignment; (c) backpressure from a slow sink (is the online-store write path itself degraded, causing the sink buffer to back up and delay checkpoint completion); (d) infra-level issues — spot node reclaims causing task-manager churn mid-checkpoint. Start with Flink's checkpoint UI (per-subtask duration breakdown) to localize to a specific operator/subtask before assuming a systemic cause.

7. **Adding embeddings**: Introduce an embedding-computation pipeline (e.g., a two-tower model producing player/match embeddings, retrained periodically) that writes embedding vectors as additional fields into the existing online feature store schema — no redesign of the streaming/serving core needed. If similarity search (not just embedding-as-feature) is required (e.g., "find players behaviorally similar to known cheaters"), add a vector DB (e.g., pgvector or a dedicated ANN index) as a new component alongside the feature store, with an ANN index (HNSW for accuracy/latency balance at this QPS scale) — additive, not a replacement of the streaming architecture.

8. **Canary statistical test**: Use a sequential testing approach (e.g., sequential probability ratio test or a Bayesian A/B framework) rather than a single fixed-sample t-test, since we want the option to abort early on clear regression without waiting the full 30 minutes, while controlling false-positive-abort rate given the noisy, non-stationary nature of live player behavior. Pre-register the minimum detectable effect size (e.g., 0.5% absolute FP-rate increase) and the alpha/power tradeoffs before the canary starts, to avoid post-hoc threshold shopping.

9. **GDPR and multi-region training**: EU player data and models trained on it must stay within EU infrastructure boundaries; a "global" model can still be trained by aggregating *derived, non-PII statistics* (e.g., anonymized/aggregated feature distributions or model gradients, not raw event-level EU data) cross-region, or by training region-local models and only sharing model *parameters* (not underlying data) globally via a federated-learning-style pattern if a single global model is required. In practice for this system, the pragmatic choice is region-local models by default, with a global model only trained on non-EU data plus opt-in/anonymized EU aggregates, reviewed by legal/compliance before shipping.

10. **Partition over-provisioning tradeoff**: More partitions increase per-broker file-handle/memory overhead, increase end-to-end latency slightly (more replication fan-out), and slow down leader election/controller failover time (more metadata to shuffle) — so blindly over-provisioning for a hypothetical 10x isn't free. The better approach is provisioning partitions for a realistic 2-3x near-term growth target with a documented, tested repartitioning runbook (using a tool like Kafka's partition reassignment or Cruise Control) for the eventual larger jump, rather than either under-provisioning (causing a painful mid-life re-partition) or wildly over-provisioning (paying an ongoing tax for capacity you may never use).
