# Player Churn Prediction

## 1. Problem Framing & Requirement Gathering

- Predict probability a player stops engaging with a live-service title (e.g., Battlefield, Apex-style shooter, FIFA/EA Sports FC Ultimate Team) within a future window (7/14/30 days), so retention teams can intervene before the player leaves.
- Consumers: CRM/lifecycle marketing (push notifications, in-game offers), live-ops (dynamic difficulty/economy tuning), player-support prioritization, exec dashboards (cohort health).
- Two operating modes required by the business: (a) nightly batch scores for the full active player base feeding CRM campaigns, (b) near-real-time scores triggered by session-end/key events (e.g., rage-quit after loss streak) for in-session intervention (offer surfaced on next login, or matchmaking adjustment).
- Churn is a moving target across titles — a "churned" FIFA Ultimate Team player looks different from a churned live-service shooter player, so the system must support per-title models sharing infrastructure, not one global model.
- Business KPI: lift in D30 retention for treated vs. control cohorts (holdout group withheld from interventions to measure causal lift), incrementality of offers (avoid cannibalizing spend on players who'd stay anyway).

## 2. Functional Requirements

- FR1: Compute a churn-probability score (0-1) per `player_id` per `title_id` on a nightly batch cadence for all players active in trailing 30 days.
- FR2: Compute/refresh a churn score in near-real-time (< 5 min end-to-end) after significant telemetry events (session end, purchase, match loss streak, friend-list changes).
- FR3: Expose scores via a low-latency read API for downstream consumers (CRM engine, live-ops rules engine, in-game offer service).
- FR4: Trigger interventions (push notification, in-game offer, matchmaking nudge) when score crosses a title-specific threshold, respecting frequency caps and holdout groups.
- FR5: Maintain point-in-time correct features for training so offline/online skew is minimized.
- FR6: Support per-title model versions, A/B test different model versions/thresholds concurrently.
- FR7: Provide explainability (top contributing features) for support/live-ops tooling and for compliance review of automated offers.
- FR8: Retrain models on a fixed cadence and on drift-triggered basis; support rollback to prior model version.
- FR9: Log every scoring decision and every triggered intervention for auditability and incrementality measurement.

## 3. Non-Functional Requirements (latency, availability, throughput, consistency, cost)

| Dimension | Requirement |
|---|---|
| Latency (batch path) | Full player-base scoring completes within nightly 4-hour window (e.g., 01:00-05:00 UTC) |
| Latency (near-real-time path) | p99 event-to-score-available < 5 min; p99 score-read API < 100 ms |
| Availability | Score-read API: 99.9% (CRM/live-ops degrade gracefully to last-known score on failure); scoring pipeline: 99.5% (batch can retry next window) |
| Throughput | Batch: ~120M player-title rows/night; streaming: sustained 8k events/sec, burst 40k events/sec at peak concurrent-player hours |
| Consistency | Eventual consistency acceptable for scores (staleness up to 24h for batch, 5 min for streaming); feature store must be point-in-time correct for training (no leakage) |
| Cost | GPU/CPU training budget capped per title per retrain cycle; inference must run predominantly on CPU (tree ensembles) to keep serving cost low relative to LLM-heavy systems elsewhere in the stack |
| Durability | Feature and score history retained per data-retention policy (see Logging section) with no data loss on write path (queue-backed) |

## 4. Clarifying Questions an interviewer would expect you to ask

1. Which titles are in scope — one flagship live-service title, or a shared platform across EA Sports FC, Apex, The Sims, Battlefield? (Drives multi-tenancy design.)
2. What does "churn" mean operationally — no login in N days, no monetization event, or a survival-analysis time-to-event definition?
3. What's the intervention budget/frequency cap — how many pushes/offers per player per week before fatigue?
4. Is there an existing player telemetry event bus (e.g., EA's central telemetry pipeline) we integrate with, or do we build ingestion from scratch?
5. Do we need cross-title churn signals (a player churning from FC but active in Apex) or is this strictly single-title?
6. What's the acceptable false-positive cost — sending an offer to a player who wouldn't have churned (cannibalization) vs. false negative (losing a player silently)?
7. Regulatory constraints — GDPR/CCPA player consent for behavioral profiling and targeted offers, especially for minors (COPPA)?
8. Who owns model interpretability sign-off — legal/compliance review needed before automated pricing/offer decisions?
9. What's existing feature-store/ML-platform tooling at EA (e.g., internal platform, SageMaker, Vertex) we should build on vs. greenfield?
10. Is holdout/control-group infrastructure for incrementality testing already standardized, or does this system need to build it?

## 5. Assumptions (explicit, numbered)

1. Flagship live-service title with 25M monthly active users (MAU), 6M daily active users (DAU).
2. Churn defined as: no login within 14 consecutive days, evaluated with a 30-day prediction horizon (predict at day T whether player is inactive across [T+1, T+30] with no session).
3. Telemetry already flows into a central Kafka-based event bus (EA-wide pattern); this system is a consumer, not the ingestion owner.
4. Average player generates ~150 telemetry events/session, ~2.3 sessions/day for active players.
5. Model family: gradient-boosted trees (LightGBM/XGBoost) for tabular churn features — no deep learning required for baseline; DL (sequence models on session history) considered as a v2 uplift.
6. Feature store is shared infra reused by other player-modeling systems (LTV, matchmaking skill, fraud) — this chapter treats it as integrated, not standalone.
7. Batch scoring window: nightly, 4-hour SLA, using Spark on EMR/Databricks-style cluster.
8. Near-real-time path targets only "high-value moment" triggers (post-session, post-purchase-refund, loss-streak >=5), not every event — reduces streaming compute cost.
9. Intervention channels (push notification service, in-game offer service) are existing systems this system calls via API; not built here.
10. One retrain cycle per title per week baseline, with drift-triggered ad hoc retrains.
11. Cost target: inference infra cost should stay under 3% of the retention-lift dollar value generated (guardrail used in capacity/cost sections).

## 6. Capacity Estimation (QPS, storage, model size, GPU/CPU counts, back-of-envelope math shown)

**Batch scoring:**
- Player-title rows scored nightly: 25M MAU × 1.0 (single title assumption for base math, note multi-title multiplies) ≈ 25M rows/night; assume up to 5 titles sharing platform → 120M rows/night total (per assumption in NFR table).
- Per-row feature vector: ~300 features × 4 bytes (float32) ≈ 1.2 KB/row → 120M × 1.2 KB ≈ 144 GB raw feature read per batch run.
- Spark cluster sizing: target 4-hour SLA, LightGBM batch inference throughput ~50k rows/sec/core (tree inference is cheap) → 120M rows / 50k rows/sec ≈ 2,400 core-seconds ≈ 40 core-minutes theoretical; practically, I/O and feature joins dominate, so provision for shuffle-heavy joins: 64-node cluster × 16 cores = 1,024 cores, target job wall-clock ~45-60 min including feature assembly, leaving headroom in the 4-hour window for retries/backfill.
- CPU-only cluster; no GPU needed for GBT batch inference.

**Near-real-time scoring:**
- Trigger-worthy events: session-end + purchase-refund + loss-streak, estimated ~15% of DAU sessions/day trigger scoring → 6M DAU × 2.3 sessions/day × 0.15 ≈ 2.07M scoring requests/day.
- Average QPS = 2.07M / 86,400 s ≈ 24 QPS sustained; peak concurrent hours (evening peak, 3x average) ≈ 72 QPS; provision for 150 QPS burst headroom.
- Per-inference latency budget (see Section 43) target p99 < 50 ms compute time on a CPU inference pod (LightGBM booster, ~300 features) — GBT inference for a single row is sub-millisecond to a few ms; budget dominated by feature fetch, not compute.
- Model size: LightGBM ensemble, ~2,000 trees × depth 6 ≈ 15-40 MB serialized per title model; 5 titles × 2 model versions (canary + stable) ≈ 400 MB total in-memory across serving pods — trivially fits on CPU pod RAM (no GPU, no sharding needed).

**Storage:**
- Feature store offline (historical, for training): 120M rows/day × 1.2 KB × 365 days ≈ 52.6 TB/year raw (before compression); with columnar compression (Parquet, ~5:1) ≈ 10.5 TB/year effective — stored in data lake (S3-equivalent), partitioned by date.
- Online feature store (low-latency serving): only latest feature vector per player needed, not history → 25M players × 5 titles × 1.2 KB ≈ 150 GB in a key-value store (Redis/DynamoDB-class) — easily fits in-memory tier with replication.
- Score history log: 120M batch rows + ~2M near-real-time rows/day, each ~200 bytes (player_id, title_id, score, model_version, timestamp, top-3 features) ≈ 24.4 GB/day ≈ 8.9 TB/year — append-only, partitioned, cheap columnar storage (compressed further to ~1.8 TB/year effective).

**Compute footprint summary:**
- Batch training (weekly, per title): distributed LightGBM training on ~150M historical player-days sample, 32-core CPU cluster, ~2-3 hours/run × 5 titles ≈ 10-15 CPU-cluster-hours/week — no GPU required (tree models); GPU only considered if v2 sequence-DL model is adopted (would need ~4x V100/A10-class GPUs for a few hours/week per title).
- Serving fleet: ~12-20 CPU pods (4 vCPU/8GB each) behind autoscaler for near-real-time path at baseline load, bursting to ~40 pods at peak.

## 7. High-Level Architecture

```
                         ┌────────────────────────────────────────────────┐
                         │              Game Clients (all titles)          │
                         └───────────────────────┬──────────────────────────┘
                                                  │ telemetry events
                                                  ▼
                         ┌────────────────────────────────────────────────┐
                         │        EA Central Telemetry Bus (Kafka)         │
                         │   topics: session.end, purchase, match.result,  │
                         │           social.friend, loss.streak            │
                         └───────────┬───────────────────────┬──────────────┘
                                     │                        │
                     (near-real-time path)              (batch path)
                                     │                        │
                                     ▼                        ▼
                  ┌──────────────────────────┐   ┌──────────────────────────────┐
                  │ Streaming Feature/Trigger │   │  Nightly Batch Feature ETL     │
                  │ Consumer (Flink/KStreams) │   │  (Spark on EMR/Databricks)     │
                  │ - windowed aggregates     │   │ - full feature recompute       │
                  │ - trigger rules           │   │ - joins w/ offline feature     │
                  └───────────┬───────────────┘   │   store history                │
                              │                     └───────────┬───────────────────┘
                              ▼                                 ▼
                  ┌──────────────────────────┐   ┌──────────────────────────────┐
                  │  Online Feature Store     │◄──┤  Offline Feature Store         │
                  │  (Redis/DynamoDB)         │   │  (Data Lake, Parquet, S3)      │
                  └───────────┬───────────────┘   └───────────┬───────────────────┘
                              │                                 │
                              ▼                                 ▼
                  ┌──────────────────────────┐   ┌──────────────────────────────┐
                  │  Near-Real-Time Model     │   │   Batch Scoring Job            │
                  │  Serving (KServe/Triton   │   │   (Spark + model broadcast)    │
                  │  CPU pods, GBT models)    │   │                                │
                  └───────────┬───────────────┘   └───────────┬───────────────────┘
                              │                                 │
                              └───────────────┬─────────────────┘
                                               ▼
                                 ┌───────────────────────────┐
                                 │   Score Store (write path)  │
                                 │  Cassandra/DynamoDB +        │
                                 │  Score History Log (Kafka →  │
                                 │  lake)                       │
                                 └────────────┬────────────────┘
                                              │
                                              ▼
                          ┌────────────────────────────────────────┐
                          │      Intervention Decision Engine        │
                          │  (threshold rules, holdout assignment,    │
                          │   frequency cap enforcement)              │
                          └───────────┬───────────────┬───────────────┘
                                      │               │
                                      ▼               ▼
                       ┌───────────────────┐  ┌────────────────────┐
                       │ Push Notification  │  │ In-Game Offer/CRM   │
                       │ Service (existing) │  │ Service (existing)  │
                       └───────────────────┘  └────────────────────┘

           ┌───────────────────────────────────────────────────────────────┐
           │  Training Pipeline (orchestrated via Airflow):                  │
           │  Offline Feature Store → Train/Val split (time-based) →         │
           │  LightGBM training → Eval vs. holdout → Model Registry →         │
           │  Canary deploy → Promote/Rollback                               │
           └───────────────────────────────────────────────────────────────┘

           ┌───────────────────────────────────────────────────────────────┐
           │  Monitoring/Drift/Alerting Plane: metrics from every box above  │
           │  → Prometheus/Grafana, drift jobs, PagerDuty routing            │
           └───────────────────────────────────────────────────────────────┘
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| Streaming Trigger Consumer | Consume telemetry topics, compute windowed aggregates (session count, loss streak), decide if a scoring trigger fires | Kafka consumer group, emits `score.request` events | Partition count of input topic; horizontal consumer scaling |
| Batch Feature ETL | Nightly recompute of full feature set per player-title from raw telemetry + offline store | Spark job, reads Parquet/Delta, writes to offline feature store | Executor count on Spark cluster |
| Online Feature Store | Serve latest feature vector per player with < 10ms read latency | gRPC/REST `GET /features/{player_id}/{title_id}` | Redis cluster shards / DynamoDB partitions |
| Offline Feature Store | Historical, point-in-time correct feature snapshots for training | Batch read via Spark/SQL, time-travel query support (Delta Lake) | Data lake storage, partitioned by date/title |
| Near-Real-Time Model Server | Serve GBT model inference on demand, < 50ms compute | gRPC `Predict(player_id, title_id, feature_vector) -> score` | Pod replicas behind HPA on CPU/QPS |
| Batch Scoring Job | Score all active players nightly, write to score store | Spark job broadcasting model, writes bulk to Cassandra/DynamoDB | Executor count, model broadcast size |
| Score Store | Durable low-latency store of current + recent scores | `GET /score/{player_id}/{title_id}`, `PUT` on write path | Partition key = player_id, replicated |
| Intervention Decision Engine | Apply threshold + holdout + frequency-cap logic, call downstream channels | Consumes `score.updated` events, calls Push/Offer APIs | Stateless workers, scale on event backlog |
| Training Orchestrator | Schedule/execute retrain DAGs, manage train/val splits, trigger eval | Airflow DAG, `POST /train` internal trigger | Worker pool, one DAG run per title |
| Model Registry | Version, store, and serve model artifacts with metadata (metrics, lineage) | MLflow-style API, `GET /models/{title}/{version}` | Backed by object storage + metadata DB |
| Drift Monitor | Compute feature/prediction/label drift metrics on schedule | Batch job reading recent scores + features vs. training baseline | Scheduled job, scales with feature count |
| Explainability Service | Compute SHAP/feature-contribution for a given score on demand (support tooling) | `GET /explain/{player_id}/{title_id}` | Stateless, CPU-bound, scales with support tool QPS |

## 9. API Design (concrete endpoint signatures, request/response schemas, versioning)

**Score Read API** — consumed by CRM, live-ops, support tools.

```
GET /v1/scores/{title_id}/{player_id}
Response 200:
{
  "player_id": "p_9f2a...",
  "title_id": "fc25",
  "churn_score": 0.734,
  "score_horizon_days": 30,
  "model_version": "fc25-churn-v14",
  "scored_at": "2026-07-08T02:14:00Z",
  "source": "batch",              // "batch" | "near_real_time"
  "top_features": [
    {"feature": "sessions_last_7d", "contribution": -0.21},
    {"feature": "loss_streak_current", "contribution": 0.18},
    {"feature": "days_since_last_purchase", "contribution": 0.14}
  ]
}
Response 404: player has no score yet (new player, cold-start)
Response 503: score store degraded — client falls back to cached/last-known
```

**Batch Score Fetch (bulk, for CRM segment export)**
```
POST /v1/scores/{title_id}/batch
Body: { "player_ids": ["p_1", "p_2", ...], "max": 10000 }
Response: { "scores": [ {...same shape as above...}, ... ], "missing": ["p_3"] }
```

**Score Write (internal, from scoring jobs only, service-auth required)**
```
POST /internal/v1/scores/{title_id}
Body: { "player_id": "...", "churn_score": 0.55, "model_version": "...", "source": "near_real_time" }
Response 202 Accepted (async write to store + history log)
```

**Trigger Intervention Decision (internal, decision engine → channels)**
```
POST /internal/v1/interventions/evaluate
Body: { "player_id": "...", "title_id": "...", "churn_score": 0.81 }
Response: { "action": "push_notification", "campaign_id": "retention_offer_42", "holdout": false }
          | { "action": "none", "reason": "frequency_cap_exceeded" }
          | { "action": "none", "reason": "holdout_group" }
```

**Explainability API**
```
GET /v1/explain/{title_id}/{player_id}?score_id={id}
Response: { "shap_values": {...}, "base_value": 0.31, "model_version": "..." }
```

**Versioning:** URI-based (`/v1/`, `/v2/`) for breaking schema changes; `model_version` field is independent of API version and tracked separately in the model registry — allows model rollouts without API contract changes. Deprecation policy: 2 major API versions supported concurrently, 90-day sunset notice.

## 10. Database Design (schema sketches, choice of SQL/NoSQL/columnar and why, partitioning/sharding key)

**Score Store — Cassandra (or DynamoDB)**, chosen for high write throughput (nightly bulk batch write of 120M rows within SLA window) + low-latency point reads, and natural partitioning by player.

```sql
-- Cassandra CQL sketch
CREATE TABLE scores (
  title_id       text,
  player_id      text,
  churn_score    float,
  model_version  text,
  scored_at      timestamp,
  source         text,       -- batch | near_real_time
  top_features   map<text, float>,
  PRIMARY KEY ((title_id, player_id))
);
```
Partition key: `(title_id, player_id)` — even distribution, no hot partitions since player_ids are high-cardinality; title_id included so per-title read/write load can be isolated/monitored.

**Offline Feature Store — Delta Lake / Parquet on object storage**, columnar, chosen for cost-efficient large scans + time-travel (point-in-time correctness for training joins).

```
Table: features_daily
Partitioned by: dt (date), title_id
Columns: player_id, dt, title_id, sessions_last_7d, sessions_last_30d,
         loss_streak_current, days_since_last_purchase, avg_session_length_min,
         friend_count_active, purchase_count_90d, ... (~300 cols total)
```

**Online Feature Store — Redis (hash per player) or DynamoDB**, chosen for sub-10ms reads needed by near-real-time serving path; only stores *latest* feature snapshot, not history (history lives offline).

```
Key: feat:{title_id}:{player_id}
Value: hash of ~300 feature fields, TTL = 48h (refreshed on each event/batch run)
```

**Model Registry metadata — relational (Postgres)**, chosen because metadata is low-volume, highly relational (model → metrics → lineage → deployment history), and needs transactional consistency for promotion/rollback bookkeeping.

```sql
CREATE TABLE model_versions (
  id SERIAL PRIMARY KEY,
  title_id TEXT NOT NULL,
  version TEXT NOT NULL,
  training_run_id TEXT,
  auc_offline FLOAT,
  logloss_offline FLOAT,
  artifact_uri TEXT,
  status TEXT, -- staged | canary | production | retired
  created_at TIMESTAMP DEFAULT now()
);
```

**Sharding:** Score store and online feature store shard by `player_id` hash (Cassandra token ring / DynamoDB partition key) — even distribution across 25M+ players; `title_id` prefix used only for per-title monitoring/isolation, not as primary shard driver (would create skew since title populations differ 10x in size).

## 11. Caching (what's cached, cache invalidation strategy, cache-aside vs write-through)

- **Read-through cache in front of Score Store**: CRM/live-ops queries are read-heavy and tolerate staleness (batch scores valid ~24h). Cache-aside pattern: read API checks local/Redis cache first, falls back to Cassandra on miss, populates cache with TTL = 6h.
- **Online feature store is itself a cache** in front of the offline store: write-through on every batch/streaming feature update — features written to Redis immediately after computation, never lazily backfilled from offline store (would violate the < 5 min near-real-time SLA).
- **Model artifact cache**: serving pods cache loaded model binary in-process memory (no cold-load per request); invalidated on new model promotion via registry webhook triggering pod rolling restart or hot-swap.
- **Invalidation strategy**: 
  - Score cache: TTL-based (6h) + explicit invalidation on new near-real-time score write (publish `score.updated` event, cache layer subscribes and evicts key).
  - Feature cache: overwritten on every write (write-through), TTL 48h as safety net against stale/orphaned keys for inactive players.
- **No caching of raw telemetry** — high write volume, low re-read value; goes straight to lake.

## 12. Queues & Async Processing (what's queued, at-least-once vs exactly-once, dead-letter handling)

| Queue/Topic | Producer | Consumer | Delivery Semantics | Dead-Letter Handling |
|---|---|---|---|---|
| `telemetry.raw.*` | Game clients | Streaming consumer, batch ETL | At-least-once (Kafka default) | N/A — raw ingestion has separate schema-validation DLQ upstream (owned by central telemetry team) |
| `score.request` | Streaming trigger consumer | Near-real-time model server | At-least-once; consumer idempotent on `(player_id, title_id, event_id)` | Retry 3x with backoff, then to `score.request.dlq`; alert if DLQ depth > 500 |
| `score.updated` | Both scoring paths | Cache invalidator, decision engine, history log writer | At-least-once, consumers dedupe via `scored_at` monotonic check | DLQ + reprocessing job run hourly |
| `intervention.trigger` | Decision engine | Push/Offer service adapters | At-least-once; downstream services must be idempotent on `campaign_id + player_id + day` | DLQ; failures alert live-ops on-call, manual replay tool |

- Exactly-once not required anywhere in this system — scoring and interventions are idempotent by design (re-scoring same player produces same/similar result; re-sending a push is deduped by campaign+player+day key downstream), so at-least-once + idempotent consumers is the simpler, cheaper choice over exactly-once transactional processing.
- Backpressure: streaming consumer applies rate limiting/sampling if `score.request` backlog exceeds threshold — better to skip low-priority near-real-time triggers under load than fall behind SLA (graceful degradation to batch-only scoring for that player).

## 13. Streaming & Event-Driven Architecture (topics, event schemas, consumer groups)

**Topics:**
- `session.end` — `{player_id, title_id, session_id, duration_sec, end_reason, timestamp}`
- `match.result` — `{player_id, title_id, match_id, result, loss_streak_delta, timestamp}`
- `purchase` — `{player_id, title_id, sku, amount_usd, currency, timestamp}`
- `social.friend_change` — `{player_id, title_id, friend_id, action, timestamp}`
- `score.request` — `{player_id, title_id, trigger_reason, requested_at}`
- `score.updated` — `{player_id, title_id, churn_score, model_version, source, scored_at}`

**Consumer groups:**
- `cg-streaming-features`: consumes raw telemetry topics, maintains windowed aggregates (Flink, keyed by player_id), emits `score.request` when trigger rule fires.
- `cg-nrt-scorer`: consumes `score.request`, calls online feature store + model server, publishes `score.updated`.
- `cg-cache-invalidator`: consumes `score.updated`, evicts/refreshes read cache.
- `cg-history-writer`: consumes `score.updated`, appends to score history log (lake).
- `cg-decision-engine`: consumes `score.updated`, evaluates intervention rules.
- Partitioning key across all topics: `player_id` — guarantees ordering per player (important so loss-streak windowed state and score updates for the same player are processed in order) while allowing horizontal scale-out by partition count (e.g., 128 partitions per topic).

## 14. Model Serving (serving framework choice, batching, multi-model, hardware)

- **Framework**: KServe (or Seldon/Triton) on Kubernetes for the near-real-time path, serving LightGBM models via a lightweight Python/C++ inference server (Treelite-compiled for speed). Triton chosen if standardizing across teams that also serve DL models; KServe/plain gRPC service is sufficient and simpler for pure GBT.
- **Hardware**: CPU-only pods (tree ensembles don't benefit from GPU); 4 vCPU / 8GB RAM per pod is enough to hold multiple title models in memory and serve at target QPS.
- **Batching**: micro-batching not required for near-real-time path given low per-request compute cost (sub-ms to few-ms tree traversal) — request-level serving is simpler and meets the 50ms compute budget. Batch scoring path uses Spark's native row-wise/vectorized inference (Spark UDF wrapping Treelite predictor) instead of a serving framework.
- **Multi-model**: one serving deployment hosts multiple title models (fc25-churn-v14, apex-churn-v9, sims-churn-v6, etc.) as separate loaded model objects behind a router keyed by `title_id`; also hosts stable + canary version of each concurrently for A/B routing.
- **Model format**: models exported to a portable format (ONNX or Treelite) at training time to decouple training framework (LightGBM/XGBoost) from serving runtime.

## 15. Feature Store (online/offline split, point-in-time correctness)

- **Offline store** (Delta Lake): full historical feature snapshots partitioned by date; used exclusively for training/backtesting. Point-in-time correctness enforced via time-travel queries — when building a training row for player X labeled at day T, features are joined "as of" day T-1 only, never using data computed after the label window, preventing label leakage.
- **Online store** (Redis/DynamoDB): only current feature snapshot per player, refreshed by both batch ETL (nightly) and streaming consumer (event-triggered partial updates for high-value features like `loss_streak_current`).
- **Feature parity enforcement**: same feature transformation code (shared library) used in both the offline Spark batch job and the streaming Flink job, to avoid train/serve skew — critical failure mode otherwise (see Section 41).
- **Point-in-time join framework**: training pipeline uses a feature-store client (Feast-like) that accepts an "event timestamp" per training row and resolves the correct historical feature values, rather than doing manual as-of joins per pipeline — reduces bugs and standardizes correctness across all player-modeling systems (churn, LTV, matchmaking) sharing this store.

## 16. Vector Database (if applicable — indexing strategy, ANN algorithm choice, else state N/A and why)

**N/A for the core churn classifier.** The primary model is a tabular GBT over engineered aggregate features (session counts, purchase recency, social graph counts) — no similarity search or embedding retrieval is on the critical path for scoring or intervention decisions.

Noted extension (not in scope for this chapter's baseline): if a v2 sequence/embedding model is adopted to represent session behavior as embeddings for player-similarity-based cohort analysis (e.g., "find players behaviorally similar to recent churners" for live-ops targeting), a vector DB (pgvector/FAISS/Milvus) with an HNSW index would be introduced then — deferred because it adds infra cost without clear baseline lift over tabular features for churn *prediction* specifically (useful for churn *analysis*/segmentation, a separate consumer).

## 17. Embedding Pipelines (if applicable, else N/A and why)

**N/A for the baseline model.** Feature engineering here is hand-crafted aggregates (counts, recency, ratios) over telemetry, not learned embeddings — this keeps the model interpretable (required for support tooling and compliance review of automated offers, per FR7) and avoids the operational overhead of an embedding pipeline (versioning, drift-of-embedding-space, re-embedding on model updates).

Noted extension: a v2 uplift path could feed a sequence model (e.g., Transformer over session-event sequences) producing a learned player-state embedding as an additional GBT feature; this would introduce an embedding pipeline (batch re-embed on retrain cadence, embedding versioned alongside model version) — deferred pending baseline model ROI validation, consistent with Assumption 5.

## 18. Inference Pipelines (request lifecycle end-to-end)

**Near-real-time path, end-to-end:**

```
1. Player finishes match, client emits `match.result` event
2. Kafka → cg-streaming-features consumer
3. Flink job updates windowed state (loss_streak_current += 1)
4. Trigger rule fires (loss_streak >= 5) → emit `score.request`
5. cg-nrt-scorer consumes request
6. Fetch feature vector: GET online-feature-store (Redis) — ~5ms
7. Call model server: gRPC Predict(features) — ~5-15ms compute
8. Publish `score.updated` (churn_score=0.81, source=near_real_time)
9. cg-cache-invalidator evicts stale cache entry — ~1ms
10. cg-history-writer appends to score history log (async, off critical path)
11. cg-decision-engine evaluates: score > title threshold (0.75)?
      → check holdout group assignment (hash(player_id, campaign_id) % 100 < 10 => holdout)
      → check frequency cap (max 2 offers/week) via a rate-limit store
      → if eligible: POST intervention.trigger → Push Notification Service
12. Push Notification Service delivers to player's next app-open / immediate push
```

Total budget target: step 2-11 within 5 minutes p99 (dominated by Kafka consumer lag under load, not compute — see Section 43).

**Batch path (parallel diagram — request lifecycle at nightly scale):**

```
01:00 UTC: Airflow triggers batch DAG
  → Spark job reads offline feature store (last 24h delta + rolling windows)
  → Joins with static player/title metadata
  → Broadcasts model (Treelite binary) to executors
  → Vectorized predict() across 120M rows
  → Writes scores to Cassandra (bulk upsert) + score history log (Parquet append)
  → Emits `score.updated` events for downstream cache warm + decision engine batch pass
05:00 UTC: SLA checkpoint — job must be complete; alert if not
```

## 19. Training Pipelines (data prep, training orchestration, distributed training if relevant)

- **Data prep**: pull labeled training set from offline feature store — label = did player have zero sessions in [T+1, T+30] given features as of T (time-based split, no random shuffling across time to avoid leakage). Typical training set: trailing 6 months of player-days, downsampled for class balance (churners are minority class, ~15-20% base rate depending on title/segment) using stratified sampling + class weighting rather than naive oversampling (avoids duplicating noisy synthetic patterns).
- **Train/val/test split**: strictly time-based — train on months 1-4, validate on month 5, test on month 6 (holdout) — mirrors production reality where model always predicts forward in time, never interpolates.
- **Orchestration**: Airflow DAG per title: `extract_features → build_labels → train_test_split → train_lightgbm → evaluate → register_model → canary_deploy_gate`.
- **Distributed training**: LightGBM's native distributed mode (feature-parallel/data-parallel via Dask or Spark integration) across the training cluster for titles with 100M+ row training sets; not GPU-distributed since tree boosting doesn't need it — CPU cluster with 32-64 cores suffices, training completes in 1-3 hours per title.
- **Hyperparameter search**: Bayesian optimization (Optuna) over tree depth, learning rate, num_leaves, L1/L2 regularization — run as parallel trials on the same CPU cluster, gated to a fixed compute budget (e.g., 200 trials max) to bound cost.
- **Evaluation gate**: model must beat current production model on AUC-PR (churn is imbalanced, so PR curve preferred over ROC) and calibration (Brier score) on the time-based holdout before promotion is even offered as a candidate.

## 20. Retraining Strategy (cadence, triggers)

| Trigger Type | Condition | Action |
|---|---|---|
| Scheduled | Weekly, per title (e.g., every Monday 00:00 UTC) | Full retrain DAG run, candidate model evaluated against gate |
| Data drift | PSI (population stability index) on top-20 features > 0.2 vs. training baseline | Trigger ad hoc retrain within 24h |
| Concept drift | Rolling 7-day AUC-PR on live labeled outcomes drops > 5% relative vs. last known-good | Trigger ad hoc retrain + alert ML on-call |
| Business event | Major game content patch / new season launch / new monetization mechanic | Manual-trigger retrain (player behavior distribution shifts predictably around content drops) |
| Label availability lag | N/A directly, but retrain scheduling accounts for 30-day label maturation — a training run on day D can only use labels for events up to D-30 | Built into DAG's data windowing logic |

- Retrain does not auto-promote — always gated by offline eval + canary (Sections 33-34) before becoming production traffic.

## 21. Drift Detection (data drift, concept drift, what metrics, what thresholds)

| Drift Type | Metric | Threshold | Action |
|---|---|---|---|
| Feature/data drift | Population Stability Index (PSI) per feature, computed daily vs. training-time distribution | PSI > 0.1 = investigate; PSI > 0.2 = trigger retrain | Drift monitor job flags feature, dashboards show trend |
| Prediction drift | KL divergence / PSI on score distribution (are we predicting more/fewer high-risk players than usual?) | Score-distribution PSI > 0.15 | Alert ML on-call, check for upstream telemetry pipeline break before assuming true behavior shift |
| Concept drift | Rolling AUC-PR and calibration (Brier score) on matured labels (30-day lag) | Relative AUC-PR drop > 5% over 7-day rolling window | Trigger ad hoc retrain, notify model owner |
| Schema drift | Feature null-rate, new/missing columns vs. expected schema | Any unexpected schema change | Hard-fail the feature pipeline (fail-closed, don't silently serve garbage features) |
| Label drift | Base churn rate shift (e.g., new season causes spike in churn base rate) | Base rate change > 20% relative week-over-week | Flag for investigation — may be legitimate (content drought) not model failure |

## 22. Monitoring (what's monitored: infra, model quality, business metrics)

**Infra:**
- Kafka consumer lag per consumer group (target: `score.request` lag < 30s p99).
- Model server latency (p50/p95/p99), error rate, pod CPU/memory utilization.
- Spark batch job duration, executor failure rate, shuffle spill.
- Feature store read latency (online) and write throughput (both stores).

**Model quality:**
- Offline: AUC-PR, log-loss, calibration curve, per-cohort fairness slices (e.g., new vs. veteran players, spend tiers) tracked per training run in the model registry.
- Online: rolling AUC-PR on matured labels, prediction distribution drift (Section 21), feature drift (PSI).
- Score staleness: % of players with score older than SLA (24h batch / 5min near-real-time).

**Business:**
- Retention lift: D7/D30 retention of treated vs. holdout cohort per campaign.
- Intervention volume: pushes/offers sent per day, frequency-cap saturation rate.
- Incrementality: conversion rate on offers among high-churn-score vs. control, cost-per-incremental-retained-player.
- False-positive cost proxy: offer redemption rate among players who would have stayed anyway (estimated via holdout comparison).

## 23. Alerting (alert conditions, thresholds, on-call routing)

| Alert | Condition | Severity | Route |
|---|---|---|---|
| Batch SLA miss | Nightly batch job not complete by 05:00 UTC | P1 | ML Platform on-call (PagerDuty) |
| Near-real-time SLA breach | p99 event-to-score latency > 5 min for 15 min sustained | P2 | ML Platform on-call |
| Consumer lag | `score.request` lag > 2 min | P2 | ML Platform on-call |
| Model server error rate | > 1% error rate over 5 min window | P1 | ML Platform on-call, auto-page |
| Concept drift | Rolling AUC-PR drop > 5% | P3 | Model owner (Slack + ticket), not a page |
| Feature schema break | Unexpected schema change detected | P1 | Data engineering on-call + ML Platform |
| Intervention volume anomaly | Daily intervention count > 3x 7-day average | P2 | Live-ops on-call (possible runaway trigger bug) |
| Retention KPI regression | Weekly D30 lift goes negative for 2 consecutive weeks | P3 | Product/ML leads, weekly review, not paged |

- Routing tiers: P1 = page immediately, 15-min response SLA; P2 = page during business hours / page if sustained > 30 min off-hours; P3 = ticket + async review, no paging.

## 24. Logging (structured logging strategy, PII handling, retention)

- **Structured logs**: JSON-formatted, every scoring event and intervention decision logged with `trace_id`, `player_id` (pseudonymized — see below), `title_id`, `model_version`, `decision`, `timestamp`.
- **PII handling**: `player_id` used internally is EA's pseudonymous player/account ID, not raw PII (email/name); any log line touching raw PII (e.g., for support escalation lookups) routes through a separate access-controlled PII-handling service, not general application logs. Logs are scrubbed of free-text fields that could contain PII (e.g., no raw chat/voice content ever enters this pipeline).
- **Retention**: 
  - Operational logs (application/infra): 30 days hot (searchable, e.g., in a log aggregation platform), 1 year cold archive for incident forensics.
  - Score history log: 2 years (supports long-horizon model evaluation and regulatory audit of automated-decision history), then aggregated/anonymized.
  - Intervention decision log: 2 years, tied to consent/compliance requirements around automated profiling decisions (GDPR Article 22 relevance — right to explanation).
- **Access control**: score/intervention logs containing player-level detail restricted to ML platform + compliance roles; aggregated/anonymized views available more broadly for business reporting.

## 25. Security (authn/authz, data encryption, threat model specific to this system)

**Threat model specific to churn prediction:**
- Model inversion / membership inference: could an attacker query the score API to infer whether a specific player was in the training set, or reconstruct sensitive behavioral attributes? Mitigate via rate limiting (Section 27), no raw-feature echo in API responses beyond top-3 abstracted feature names (not raw values), and access restricted to authenticated internal services.
- Manipulation for exploit: could a player deliberately manipulate telemetry (e.g., bot farming fake "loss streaks") to trigger discount offers? Mitigate via anomaly detection on trigger-event rates per player, and by not exposing threshold logic externally.
- Data exfiltration: bulk score/feature export endpoints are high-value targets (behavioral profiles at scale) — require stricter authz scopes than single-player lookups, audit-logged access, rate-limited bulk endpoint (Section 9).
- Insider risk: engineers with feature-store access can see granular behavioral data — access via least-privilege IAM roles, no direct prod data access for local dev (synthetic/sampled data used instead).

**Encryption**: TLS 1.2+ in transit for all service-to-service and client-facing calls; at-rest encryption (KMS-managed keys) for offline feature store, score history log, and model artifacts. Field-level encryption not typically needed since player_id is already pseudonymous, but purchase-amount and payment-adjacent fields get additional access scoping.

## 26. Authentication (service-to-service and end-user auth mechanism)

- **Service-to-service**: mTLS + short-lived JWT (service identity tokens issued by internal identity provider, e.g., SPIFFE/SPIRE-style workload identity) for all internal calls (CRM → Score API, Decision Engine → Push Service). Tokens scoped per-service with least-privilege claims (e.g., `read:scores`, `write:scores:internal`).
- **End-user auth**: N/A directly — no end-user (player) ever calls this system's APIs directly; all access is via internal consuming services (CRM, live-ops tools) that themselves authenticate players through EA's account/identity platform separately. Internal support/ops tooling (e.g., a dashboard letting a live-ops analyst look up a player's score) authenticates analysts via corporate SSO (OIDC) with RBAC roles gating access to player-level detail.

## 27. Rate Limiting (algorithm choice, per-user/per-tenant limits)

- **Algorithm**: token bucket per calling service (not per end-user, since callers are internal services, not players) — allows burst tolerance (e.g., CRM kicking off a large campaign batch pull) while capping sustained rate.
- **Score Read API**: 500 req/sec per calling service, burst to 1,000; bulk batch endpoint capped at 10,000 player_ids per call, 10 calls/min per service.
- **Explainability API**: stricter — 50 req/sec per service (heavier compute, SHAP calculation), since it's a support-tool-facing endpoint with lower expected volume.
- **Internal write path** (scoring jobs writing scores): not rate-limited by token bucket but by backpressure/batch-size controls in the Spark/streaming jobs themselves.
- **Per-player intervention frequency cap** (distinct from API rate limiting — a business rule, not infra protection): enforced in the Decision Engine via a sliding-window counter in a fast KV store (e.g., max 2 interventions/player/7-day window), independent of API-layer rate limiting.

## 28. Autoscaling (metrics-driven autoscaling policy, HPA/VPA/KEDA specifics)

- **Near-real-time model server**: Kubernetes HPA on custom metric = `p99 inference latency` and `request queue depth`, secondary trigger on CPU utilization (target 60%). Min replicas 8 (baseline ~24 QPS needs), max replicas 40 (peak ~150 QPS burst headroom per Section 6).
- **Streaming trigger consumer (Flink)**: scaled via KEDA on Kafka consumer-lag metric for `session.end`/`match.result` topics — scale out task managers when lag > 10k messages, scale in when lag stable near-zero for 10 min.
- **Decision Engine workers**: KEDA scaling on `score.updated` topic consumer lag, since this is a stateless, embarrassingly-parallel consumer.
- **Batch Spark cluster**: not HPA-driven (transient job, not a long-running service) — cluster sized at job submission time via Airflow DAG parameters (e.g., dynamic executor allocation within Spark itself, min 32 / max 128 executors based on partition count of the day's input).
- **VPA**: applied to the model server pods for right-sizing memory requests over time (model sizes grow slowly as feature count/tree count increases across retrains) — set to "recommend-only" mode reviewed monthly rather than auto-apply, to avoid surprise restarts during peak traffic.

## 29. Cost Optimization (concrete levers: spot instances, caching, model distillation, batching)

- **Spot instances**: batch Spark training/scoring clusters run on spot/preemptible CPU instances (training/batch scoring is fault-tolerant, checkpointable, and not latency-critical) — estimated 60-70% cost reduction vs. on-demand for this workload class.
- **CPU-only inference**: avoiding GPU serving entirely for the GBT baseline (Section 14) is itself the single biggest cost lever vs. a DL-first approach — CPU pods are an order of magnitude cheaper per inference than GPU-backed serving.
- **Caching**: read-through score cache (Section 11) drastically cuts read QPS hitting Cassandra, reducing provisioned read capacity/cost on the score store.
- **Batching in scoring**: near-real-time triggers limited to "high-value moments" only (Assumption 8) rather than scoring on every telemetry event — cuts streaming compute and model-server QPS by an estimated 80%+ vs. naive "score on every event" design.
- **Model distillation/compression**: not critical for GBT (already cheap), but tree-count/depth capped via the HPO budget (Section 19) to bound both training and inference cost — a 2,000-tree ensemble is a deliberate ceiling, not a default from unconstrained search.
- **Storage tiering**: offline feature store older than 90 days moved to cold/archive storage tier (data lake lifecycle policy); score history beyond 1 year compressed further and moved off primary lake storage class.
- **Right-sizing via VPA recommendations** (Section 28) to avoid over-provisioned memory on serving pods.

## 30. Disaster Recovery (RTO/RPO targets, backup strategy)

| Component | RTO | RPO | Backup Strategy |
|---|---|---|---|
| Score Store (Cassandra/DynamoDB) | 1 hour | 15 min | Multi-AZ replication (RF=3), point-in-time backup snapshots every 15 min |
| Offline Feature Store (data lake) | 4 hours | 24 hours | Versioned object storage (immutable, append-only Delta Lake), cross-region replication |
| Model Registry (Postgres) | 1 hour | 5 min | Continuous WAL streaming to standby + daily full snapshot |
| Online Feature Store (Redis) | 30 min | Best-effort (rebuildable) | Not backed up directly — rebuildable from offline store + recent streaming replay within ~10-15 min; treated as a rebuildable cache, not source of truth |
| Kafka topics | 30 min | Near-zero (replicated) | Multi-broker replication factor 3, cross-AZ |

- Overall system RTO target: 4 hours to full batch-path recovery (worst case), near-real-time path degrades gracefully to "batch-only scoring" during an outage rather than being a hard dependency for player-facing systems.
- Runbook: on Score Store outage, Score Read API falls back to last-cached value + serves a `stale=true` flag rather than hard-failing, so CRM/live-ops consumers degrade gracefully instead of erroring.

## 31. Multi-Region Deployment (active-active vs active-passive, data replication, latency routing)

- **Topology**: active-active across 2-3 regions (e.g., US-East, EU-West, matching EA's player base geography) for the near-real-time serving path — players routed to nearest region for lowest latency; each region has its own model-server replica set and online feature store shard.
- **Data replication**: score store and online feature store use region-local writes with async cross-region replication (eventual consistency acceptable given staleness tolerance in NFRs) rather than synchronous multi-region writes (would blow the latency budget).
- **Offline feature store / training**: centralized in a single primary region (training doesn't need multi-region — it's not latency-sensitive) with cross-region read replicas/backups for DR only.
- **Latency routing**: GeoDNS or global load balancer (e.g., latency-based routing policy) directs telemetry/scoring requests to nearest healthy region; failover to next-nearest region on regional health-check failure.

```
                     ┌─────────────────────────┐
                     │   Global LB / GeoDNS      │
                     └───────────┬───────────────┘
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
     ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
     │  US-East Region   │ │  EU-West Region   │ │  APAC Region      │
     │  - Model servers   │ │  - Model servers   │ │  - Model servers   │
     │  - Online feat.     │ │  - Online feat.     │ │  - Online feat.     │
     │    store (local)    │ │    store (local)    │ │    store (local)    │
     │  - Score store      │ │  - Score store      │ │  - Score store      │
     │    (region-local)   │ │    (region-local)   │ │    (region-local)   │
     └─────────┬───────────┘ └─────────┬───────────┘ └─────────┬───────────┘
               │  async cross-region replication (scores, features)         │
               └──────────────────────────┴──────────────────────────────────┘
                                           │
                                           ▼
                          ┌───────────────────────────────┐
                          │  Central: Offline Feature Store, │
                          │  Training Pipeline, Model Registry│
                          │  (single primary region)          │
                          └───────────────────────────────┘
```

## 32. Blue/Green Deployment (how it applies to this system specifically)

- Applies primarily to the **model-server deployment** and the **Decision Engine** service code (not the model itself, which uses canary — Section 33).
- Blue/Green for infra/code changes: new version of the serving container (e.g., upgraded Treelite runtime, new feature-fetch client) deployed as a fully separate "green" pod fleet alongside "blue" (current production); traffic switched via load-balancer target group swap once green passes smoke tests (health checks, synthetic scoring requests against known test players with expected score ranges).
- Rollback = instant traffic swap back to blue fleet (no redeploy needed) — critical for a system where a bad deploy mid-way through the nightly batch SLA window could cause an SLA miss; blue/green avoids the extended rollback time of a rolling update.
- Database schema changes (e.g., adding a field to the score store) done as expand-contract migrations compatible with both blue and green code paths during the transition window.

## 33. Canary Deployment (traffic-split strategy, health-check gates specific to this system)

- Applies to **new model versions** specifically (distinct from Blue/Green which is for service code).
- New model version (e.g., `fc25-churn-v15`) deployed alongside current production (`v14`) in the same serving fleet; traffic split starts at 5% of near-real-time scoring requests routed to v15, remainder to v14.
- **Health-check gates** before ramping traffic (5% → 25% → 50% → 100% over ~1 week):
  - Infra health: latency/error rate parity with v14 (no regression).
  - Model quality: rolling AUC-PR on matured labels for the canary slice must be >= v14's, calibration (Brier score) within tolerance.
  - Business guardrail: intervention volume triggered by v15 must not deviate > 20% from v14 baseline (catches a miscalibrated model that suddenly flags everyone as high-risk).
  - No fairness regression across cohort slices (spend tier, tenure) vs. v14.
- Batch scoring path also runs shadow scoring with the canary model (compute both v14 and v15 scores nightly, only v14 drives interventions) before the canary is trusted enough to influence real batch decisions — cheap to do since batch inference cost is low.

## 34. Rollback Strategy (automated triggers, rollback mechanics)

- **Automated rollback triggers**:
  - Model server error rate > 5% for 5 min sustained during canary ramp → auto-revert traffic split to 0% on canary.
  - Canary AUC-PR drop > 5% relative vs. production on rolling labeled window → auto-halt ramp, alert model owner, do not auto-promote further.
  - Intervention volume guardrail breach (Section 33) → auto-halt ramp.
- **Mechanics**: traffic-split config (canary weight) stored in a feature-flag/config service; rollback = config change (weight → 0), no redeploy required, takes effect within seconds at the router level.
- **Model registry rollback**: `status` field (Section 10 schema) flips previous production version back to `production`, current bad version marked `retired`; model server picks up registry change via polling/webhook within ~1 min.
- **Batch path rollback**: if a bad batch scoring run is detected post-hoc (e.g., anomalous score distribution caught by drift monitor after the fact), a "replay" job re-runs the prior night's model version against the same feature snapshot and overwrites the score store — mitigates having sent a night's worth of bad interventions before detection; combined with the intervention volume alert (Section 23) to catch this fast.

## 35. Observability (tracing, metrics, logs correlation — the three pillars applied here)

- **Tracing**: distributed trace (OpenTelemetry) with a `trace_id` propagated from the originating telemetry event through streaming consumer → feature fetch → model inference → decision engine → intervention call — lets an engineer answer "why did/didn't player X get an offer" end-to-end across ~6 services.
- **Metrics**: Prometheus-scraped metrics from every component (Section 22) feeding Grafana dashboards: golden signals (latency, traffic, errors, saturation) per service + ML-specific metrics (drift scores, AUC-PR trend, score distribution histograms).
- **Logs**: structured JSON logs (Section 24) correlated via `trace_id` so a single dashboard/query (e.g., in a log platform like Elastic/Loki) can pull the full request path's log lines alongside its trace.
- **Correlation in practice**: a support engineer investigating "player complained they got a churn offer despite being an active player" pulls up `trace_id` from the intervention log → sees the trace spanning feature-fetch (what features were used) → model server (what score, what model version) → decision engine (why threshold triggered) → cross-references drift dashboard for that day (was there a feature pipeline bug that day?).

## 36. Kubernetes Deployment (a concrete manifest sketch or Deployment/Service/HPA YAML snippet relevant to this system)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-model-server
  labels:
    app: churn-model-server
spec:
  replicas: 8
  selector:
    matchLabels:
      app: churn-model-server
  template:
    metadata:
      labels:
        app: churn-model-server
    spec:
      containers:
        - name: model-server
          image: registry.ea.internal/churn-model-server:v14
          resources:
            requests: { cpu: "2", memory: "4Gi" }
            limits: { cpu: "4", memory: "8Gi" }
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet: { path: /healthz, port: 8080 }
            initialDelaySeconds: 5
          env:
            - name: MODEL_REGISTRY_URL
              value: "http://model-registry.ml-platform.svc.cluster.local"
---
apiVersion: v1
kind: Service
metadata:
  name: churn-model-server-svc
spec:
  selector:
    app: churn-model-server
  ports:
    - port: 80
      targetPort: 8080
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: churn-model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: churn-model-server
  minReplicas: 8
  maxReplicas: 40
  metrics:
    - type: Pods
      pods:
        metric:
          name: p99_inference_latency_ms
        target:
          type: AverageValue
          averageValue: "50"
```

## 37. Terraform Infrastructure (a concrete Terraform snippet sketch for the core infra of this system)

```hcl
resource "aws_msk_cluster" "telemetry_bus" {
  cluster_name           = "churn-telemetry-bus"
  kafka_version          = "3.6.0"
  number_of_broker_nodes = 6

  broker_node_group_info {
    instance_type   = "kafka.m5.xlarge"
    ebs_volume_size = 500
    client_subnets  = var.private_subnet_ids
    security_groups = [aws_security_group.kafka_sg.id]
  }

  encryption_info {
    encryption_in_transit {
      client_broker = "TLS"
    }
  }
}

resource "aws_dynamodb_table" "churn_scores" {
  name         = "churn-scores"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "title_id"
  range_key    = "player_id"

  attribute {
    name = "title_id"
    type = "S"
  }
  attribute {
    name = "player_id"
    type = "S"
  }

  replica {
    region_name = "eu-west-1"
  }
  replica {
    region_name = "ap-southeast-1"
  }

  point_in_time_recovery {
    enabled = true
  }
}

resource "aws_emr_cluster" "batch_scoring" {
  name          = "churn-batch-scoring"
  release_label = "emr-6.15.0"
  applications  = ["Spark"]

  master_instance_group {
    instance_type = "m5.xlarge"
  }
  core_instance_group {
    instance_type  = "m5.2xlarge"
    instance_count = 64
    bid_price      = "0.30" # spot pricing for cost optimization
  }

  auto_termination_policy {
    idle_timeout = 300
  }
}
```

## 38. Why This Architecture (justification)

- Separating batch and near-real-time paths matches the actual business need (bulk CRM campaigns vs. moment-based intervention) instead of forcing everything through one low-latency system, which would be needlessly expensive (Section 29 CPU-only rationale) or one batch-only system, which would miss the in-session intervention window entirely.
- GBT over deep learning as the baseline model keeps interpretability (compliance/support requirement, FR7) and cost (CPU inference, Section 6/29) in check, while leaving a clear, isolated extension path (Sections 16/17) for embedding-based uplift without redesigning the core system.
- Shared feature store across player-modeling systems (churn, LTV, matchmaking) amortizes point-in-time-correctness engineering investment across multiple teams rather than each system reinventing it — a recurring EA-scale pattern.
- Event-driven design with idempotent, at-least-once processing (Section 12) is simpler to operate and reason about than exactly-once distributed transactions, and this domain tolerates the small risk of duplicate/near-duplicate scoring without correctness issues.
- Canary + shadow scoring for models (Section 33) directly guards against the single highest-blast-radius failure mode: a miscalibrated model silently flooding players with unwarranted offers (cost + brand trust risk) before it's caught.

## 39. Alternative Architectures (at least 2 alternatives with why they were rejected or when they'd be preferred)

| Alternative | Description | Why Rejected (or When Preferred) |
|---|---|---|
| Pure streaming (score every player on every event, no nightly batch) | Fully event-driven, always-fresh scores | Rejected as default: massively higher steady-state compute cost for freshness the business doesn't need for CRM-scale campaigns (24h staleness is fine); would be preferred if the product required truly continuous per-second scoring (e.g., real-time matchmaking risk scoring), not the case here |
| Pure batch (no near-real-time path at all) | Simpler ops, single nightly Spark job only | Rejected: misses FR2's in-session intervention window (e.g., can't react to a rage-quit loss streak same-session); would be preferred for titles/markets with lower intervention ROI where a same-day nightly score is sufficient and near-real-time infra cost isn't justified |
| Deep learning sequence model (Transformer over session-event sequences) as the primary model from day one | Learns representations directly from raw event sequences, no hand-engineered features | Rejected as v1 default: higher infra cost (GPU serving or heavier CPU inference), harder to explain to compliance/support (FR7), longer iteration cycle; would be preferred once baseline GBT plateaus and the team has validated that sequence signal genuinely lifts AUC-PR beyond aggregate features — treated as a v2 investment (Sections 16/17), not a v1 bet |
| Fully centralized single-region deployment | One region serves all traffic globally | Rejected: violates near-real-time latency budget for non-US players (cross-region RTT alone could eat the 50ms compute budget); acceptable only for a title with a geographically concentrated player base or in early-access/soft-launch phase before scaling to global multi-region |

## 40. Tradeoffs

| Decision | Tradeoff |
|---|---|
| GBT over DL baseline | + Interpretable, cheap, fast to iterate / − May leave AUC-PR upside on the table vs. sequence models |
| At-least-once + idempotent consumers over exactly-once | + Simpler, cheaper, easier to operate / − Requires discipline that every consumer is truly idempotent; a bug here causes duplicate (not lost) processing, which is the safer failure direction |
| Eventual consistency on scores | + Enables caching, cross-region replication, cost savings / − CRM/live-ops occasionally act on up-to-24h-stale data; acceptable given business tolerance |
| Separate batch + near-real-time paths | + Right-sized cost/latency per use case / − Two code paths to maintain, risk of feature-computation drift between them (mitigated by shared feature-transform library, Section 15) |
| Canary + shadow scoring before full model promotion | + Catches bad models before player-facing blast radius / − Slower time-to-production for genuinely good models (1-week ramp) |
| Shared feature store across player-modeling systems | + Amortized correctness engineering, consistency / − Coupling risk: a schema change for LTV's needs could inadvertently affect churn's feature pipeline; requires strong contract/versioning discipline |
| Active-active multi-region with async replication | + Low latency globally / − Eventual consistency edge cases (e.g., player switches region mid-session, sees slightly stale score) |

## 41. Failure Modes

| Failure Scenario | Impact | Mitigation |
|---|---|---|
| Telemetry bus outage/lag spike | Streaming trigger consumer starves, near-real-time scores stop updating | Graceful degrade to last-known batch score; alert on consumer lag (Section 23); near-real-time path is explicitly non-critical-path for player-facing systems |
| Feature pipeline schema break (upstream telemetry team changes event schema) | Feature computation fails or silently produces nulls/garbage | Schema validation fail-closed (Section 21) — pipeline halts and alerts rather than serving bad features; contract testing with upstream telemetry team |
| Model server serves stale/wrong model version after a bad deploy | Wrong scores drive real interventions | Canary + shadow scoring gate (Section 33), automated rollback triggers (Section 34) |
| Online feature store outage | Near-real-time scoring can't fetch features | Falls back to most recent offline/batch feature snapshot with a `degraded=true` flag; online store treated as rebuildable cache, not source of truth (Section 30) |
| Label leakage bug in training pipeline (future data leaks into features) | Offline metrics look artificially great, production model underperforms silently | Strict time-based point-in-time joins (Section 15/19) enforced via feature-store client, not manual joins; canary/shadow scoring catches real-world underperformance before full rollout |
| Runaway intervention trigger bug (e.g., threshold logic bug fires on 100% of players) | Mass over-messaging, player trust/brand damage, CRM channel exhaustion | Intervention volume anomaly alert (Section 23), frequency caps enforced independently of model logic, automated rollback (Section 34) |
| Cross-region replication lag during regional failover | Player scored in newly-active region sees stale/missing data momentarily | Documented as acceptable eventual-consistency edge case (Section 40); read API returns `stale=true` rather than erroring |

## 42. Scaling Bottlenecks (where this breaks first at 10x/100x scale)

- **At 10x (250M MAU across more titles)**: Batch feature ETL shuffle-heavy joins become the first bottleneck — the 4-hour nightly SLA window gets tight; mitigation: partition batch jobs per title running in parallel rather than one monolithic job, and pre-aggregate more incrementally (streaming upserts into offline store) rather than full nightly recompute.
- **At 10x**: Online feature store (Redis) memory footprint (150GB → 1.5TB) starts requiring careful cluster resharding/capacity planning; mitigation: move colder/less-frequently-read player features to a tiered store, keep only "hot" recently-active players in the fastest tier.
- **At 100x**: Cassandra/DynamoDB score store write throughput during the nightly batch bulk-write becomes the dominant constraint (120M rows → 12B rows nightly is unrealistic literally, but multi-title/multi-region write amplification at large scale is the real risk) — mitigation: move to a true bulk-load pattern (write to a staging table/S3 then bulk-import) rather than row-by-row upserts.
- **At 100x**: Training pipeline's single-title CPU cluster training time grows with data volume — distributed LightGBM scales reasonably but eventually the Bayesian HPO search (Section 19) becomes the long pole; mitigation: cap HPO trial budget more aggressively, or move to a warm-start HPO strategy reusing prior week's best hyperparameters as a prior.
- **Streaming path**: Kafka partition count and consumer group parallelism become the bottleneck well before compute does — mitigation: partition count planning ahead of scale (over-provision partitions early since repartitioning topics later is disruptive).

## 43. Latency Bottlenecks (where time is actually spent, p50/p99 budget breakdown)

**Near-real-time path (target: 5 min p99 end-to-end):**

| Stage | p50 | p99 | Notes |
|---|---|---|---|
| Kafka produce → consume (telemetry to streaming consumer) | 200ms | 3-5s | Normally fast; dominant tail risk under partition rebalance/lag |
| Windowed aggregate update (Flink state) | 50ms | 500ms | State backend I/O under load |
| Trigger evaluation → `score.request` emit | 10ms | 100ms | Cheap, rule-based |
| `score.request` queue wait (consumer lag) | 1s | **60-180s** | Largest, most variable contributor — dominant tail risk under load spikes |
| Feature fetch (online store) | 5ms | 15ms | Redis, in-memory |
| Model inference compute | 3ms | 15ms | Tree traversal, negligible |
| `score.updated` publish + cache invalidation | 10ms | 100ms | Async, off critical path for the actual decision |
| Decision engine evaluation (threshold + holdout + freq cap) | 10ms | 50ms | KV lookups |
| **Total (excluding downstream push delivery)** | ~1.3s | **~60-200s**, budget allows up to 300s | Queue lag, not compute, is where time actually goes |

**Score Read API (target p99 < 100ms):**

| Stage | p50 | p99 |
|---|---|---|
| Cache lookup (hit path) | 2ms | 8ms |
| Cache miss → Cassandra read | 10ms | 40ms |
| Network/serialization overhead | 5ms | 15ms |
| **Total** | ~7ms (cache hit) | ~65ms (cache miss worst case) |

- Key insight for interview discussion: compute (model inference) is never the bottleneck in this system — queueing/consumer-lag and I/O (feature fetch, cache/DB reads) dominate. This is the opposite profile from an LLM-serving system where GPU compute dominates.

## 44. Cost Bottlenecks (what actually drives the bill)

- **#1: Batch feature ETL compute (Spark/EMR cluster hours)** — largest recurring line item given nightly full-table joins across 120M+ rows and ~300 features; driven by shuffle volume, not raw row count. Mitigated via spot instances (Section 29) and incremental (not full) recompute where possible.
- **#2: Data storage (offline feature store + score history)** — ~10.5TB/year effective feature storage + ~1.8TB/year score history compounds year-over-year without lifecycle policies; mitigated via tiering to cold storage after 90 days.
- **#3: Online feature store / cache infra (Redis cluster)** — always-on, provisioned for peak, in-memory (expensive per GB vs. disk); this is why only "latest snapshot" is kept online, not history.
- **#4: Near-real-time serving fleet** — smaller than #1-3 in absolute terms because it's CPU-only and traffic is filtered to high-value triggers only (Assumption 8); would become #1 if the team ever expanded to "score every event" without justification — a classic cost trap in this system's design space.
- **#5: Cross-region replication egress** (multi-region section) — data transfer costs for score/feature replication across regions, non-trivial at 25M+ player scale but small relative to compute/storage above.
- Cost guardrail from Assumption 11 (inference infra < 3% of retention-lift value) should be reviewed quarterly against actual campaign ROI to catch cost creep before it becomes a P&L problem.

## 45. Interview Follow-Up Questions

1. How would you validate that your churn label definition (14-day inactivity, 30-day horizon) actually correlates with real business-meaningful churn, not just short-term absence (e.g., a player on vacation)?
2. Your near-real-time path only triggers on a curated set of "high-value moments." How would you decide if you're missing important trigger events, and how would you test adding a new trigger without blowing up compute cost?
3. Walk me through exactly how you'd detect and prevent train/serve skew between the batch Spark feature pipeline and the streaming Flink feature pipeline.
4. How do you measure true incrementality of an intervention (not just correlation between high score and offer redemption)?
5. If AUC-PR looks great offline but the retention-lift A/B test shows no improvement, what's your debugging process?
6. How would you extend this system to share signal across titles (a player active in Apex but churning from FC) without creating tight coupling between title-specific pipelines?
7. What would change in your design if the near-real-time SLA requirement tightened from 5 minutes to 5 seconds?
8. How do you prevent the model from learning to target players who are price-sensitive/discount-seeking rather than genuinely at risk of churning (adverse selection in the offer loop)?
9. Your system holds out a control group for incrementality measurement — how do you size that holdout, and how do you justify the "cost" of not intervening on genuinely at-risk holdout players?
10. How would you detect and handle a feedback loop where past interventions themselves become a feature that biases future churn predictions?

## 46. Ideal Answers

1. **Label validation**: Run a retrospective cohort study comparing "14-day inactive" players against longer windows (60/90-day) to measure what fraction genuinely never return (true churn) vs. return later (false churn signal, e.g., vacation/seasonal). Use survival analysis (Kaplan-Meier curves) to understand the actual return-time distribution and pick a horizon where the false-churn rate is acceptably low; validate against known seasonal patterns (holiday breaks, title-specific season cadences) and exclude/flag known false-positive segments (e.g., players who churn every summer and return every fall) rather than treating the label as ground truth.

2. **Trigger completeness**: Instrument a periodic "shadow" analysis: score a random sample of all events (not just curated triggers) against the model and measure how often high-score-worthy moments occur outside the curated trigger set (i.e., compare against what nightly batch would have caught). If a meaningful fraction of eventual churners had a detectable near-real-time signal that wasn't in the curated trigger list, add it — but gate the addition behind a cost/benefit estimate (expected QPS increase × serving cost vs. expected incremental retention value from catching it sooner), and roll out the new trigger itself behind a canary/percentage ramp to bound blast radius.

3. **Train/serve skew prevention**: Enforce a single shared feature-transformation library (same code, ideally same language/runtime, e.g., compiled UDFs) invoked by both the Spark batch job and the Flink streaming job rather than reimplementing logic twice. Add automated parity tests that periodically compute the same feature via both pipelines for a sample of players and assert numerical equality within tolerance; alert on divergence. Additionally, validate skew empirically by comparing the online-store feature snapshot against the offline store's corresponding date partition and tracking a "feature parity drift" metric as a first-class monitored signal, not just a one-time test.

4. **Measuring incrementality**: Never trust "high score + redeemed offer" as proof of causal impact — always compare against a randomized holdout/control group that receives no intervention (or a placebo). Compute incremental retention lift = retention(treated) − retention(holdout) within the same predicted-high-risk score band, and incremental revenue/cost-per-incremental-retained-player. Use this, not raw redemption rate, as the north-star metric reported to the business; also watch for cannibalization by segmenting lift by score band (offers to very-low-score players should show near-zero incremental lift — if they don't, something's confounded).

5. **Offline-online metric mismatch debugging**: Check, in order: (a) train/serve feature skew (Section/Q3) — most common root cause; (b) whether the offline eval set's label distribution matches the live population being targeted (e.g., offline eval on all players but live campaign only targets a specific segment); (c) whether the "high-risk" score threshold used to select the campaign audience is actually where AUC-PR gains manifest (a model can improve AUC-PR globally while providing no improvement in the specific score range used for targeting); (d) confirm the A/B test itself has enough statistical power (sample size, effect size assumptions) — a real lift might exist but be underpowered to detect; (e) verify the intervention mechanism (push/offer) itself didn't change concurrently, confounding attribution.

6. **Cross-title signal sharing**: Introduce a lightweight cross-title feature layer (e.g., "days since last session on any EA title," "total portfolio spend across titles") computed centrally and made available as optional additional features to each title's otherwise-independent model, rather than merging into one monolithic cross-title model. This preserves per-title model independence/ownership (different teams, different retrain cadences) while allowing titles to opt into cross-title signal as just another feature column — bounded coupling via a well-versioned shared feature contract, not a shared model or pipeline.

7. **5-second SLA**: Would require moving from a queue-mediated pipeline to a synchronous, in-request-path computation — likely embedding feature computation directly into the game session/event-handling service (or a sidecar) rather than round-tripping through Kafka + separate consumers, since consumer lag is the dominant tail latency cost (Section 43) at 5-minute SLA. Would also need to pre-compute/cache more aggressively (features updated synchronously on write, not batched), accept a much smaller feature set (skip any feature requiring heavy joins), and likely co-locate model server with the game backend/edge to cut network hops — fundamentally a different, tighter-coupled architecture, justified only if the business case for sub-5-second intervention (vs. current 5-minute) is very strong, since it multiplies infra cost and operational complexity.

8. **Adverse selection guard**: Explicitly monitor and report model performance/offer-effectiveness segmented by prior discount-seeking behavior and price sensitivity signals; if the model or the intervention-decision logic implicitly favors discount-responsive players, incremental lift analysis (Q4) segmented by this cohort will reveal it (discount-seekers show high redemption but low true incremental retention, since they'd have used a discount anyway or churn regardless). Mitigate by excluding pure price/discount-usage features from the churn model itself when possible (keep the model focused on engagement/behavioral signals) and by having the Decision Engine's offer-selection logic (separate from the churn model) apply its own incrementality-aware targeting rather than naively targeting by churn score alone.

9. **Holdout sizing**: Size the holdout using standard A/B power analysis — given the expected baseline retention rate, minimum detectable effect size the business cares about (e.g., 1-2pp lift), and desired statistical power (80-90%) and significance level (95%), compute required sample size per arm; typically a holdout of 5-10% of the eligible high-risk population is enough at EA's scale (millions of players) to detect meaningful effects while keeping the "opportunity cost" of untreated at-risk players small in absolute terms. Justify the cost explicitly to stakeholders as the price of measurement — without a holdout, the business cannot distinguish real intervention value from noise, risking much larger misallocated spend long-term.

10. **Feedback loop detection**: Explicitly track whether "received intervention in the past N days" is (accidentally or intentionally) leaking into current features, since a player who got an offer and stayed will look like a "low-risk" player in ways caused by the intervention itself, not by their underlying propensity — creating a self-reinforcing loop that degrades signal over time. Mitigate by either explicitly excluding recent-intervention-exposure from features, or if included, treating it as a controlled variable and monitoring model performance separately on intervention-naive vs. intervention-exposed populations; periodically retrain/validate using only a "clean" (holdout, never-intervened) population's outcomes as the true label source to check whether the production model's apparent performance is inflated by the feedback loop.
</content>
