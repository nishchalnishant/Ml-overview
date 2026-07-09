# Continuous Training Platform

## 1. Problem Framing & Requirement Gathering

EA runs dozens of live-service titles (FC 24/25, Apex Legends, The Sims, Battlefield) where ML models decay fast: matchmaking skill models drift as metas shift, toxicity classifiers drift as slang evolves, churn/LTV models drift after content drops, and fraud/cheat-detection models face adversarial drift daily. Manual retraining (data scientist pulls data, retrains, eyeballs metrics, files a deploy ticket) takes 1-2 weeks per cycle — too slow for a live game where a bad matchmaking model degrades retention within days.

**Goal**: build a platform that automatically detects when a production model's inputs or performance have drifted, retrains it on fresh data, validates the new candidate against rigorous gates, shadow-tests it against real traffic, and promotes or rolls back — with humans in the loop only for exceptions.

**Who uses this platform**: ML engineers (define pipelines, gates), data scientists (own model logic, review promotion decisions), SRE/on-call (respond to pipeline/serving incidents), game teams (consumers of model predictions via the inference/serving layer — out of scope here except as the promotion target).

## 2. Functional Requirements

- FR1: Ingest labeled training data from feature store + game telemetry on a scheduled and event-driven basis.
- FR2: Detect data drift (input distribution shift) and concept drift (label/performance shift) per deployed model.
- FR3: Trigger retraining jobs automatically on drift, on schedule (cadence per model family), or on manual request.
- FR4: Orchestrate distributed training jobs (data prep → train → evaluate) with lineage tracking (data version, code version, hyperparameters).
- FR5: Run a validation gate suite (offline metrics vs. champion, fairness/bias checks, data quality checks, performance regression checks) before any candidate can proceed.
- FR6: Run shadow evaluation — candidate model scores live production traffic in parallel with the champion, no impact on served predictions, results compared over a bake-in window.
- FR7: Support staged promotion (shadow → canary % traffic → full rollout) with automatic gate checks at each stage.
- FR8: Support one-click / automatic rollback to last-known-good model version.
- FR9: Maintain a full model registry with versioning, metadata, approval audit trail.
- FR10: Expose APIs/UI for humans to inspect drift reports, approve/reject promotions, and force rollbacks.
- FR11: Support multiple model types (gradient-boosted trees for churn/LTV, deep nets for matchmaking/toxicity, embedding models for recommendation) under one orchestration framework.

## 3. Non-Functional Requirements

| Dimension | Target | Notes |
|---|---|---|
| Retraining pipeline latency | < 4 hrs end-to-end (data prep → trained candidate) for daily-cadence models; < 45 min for high-frequency fraud models | Drives orchestration/parallelism design |
| Drift detection latency | < 15 min from event ingestion to drift signal | Near-real-time streaming stats |
| Availability of orchestration control plane | 99.9% | Not per-inference SLA; batch system |
| Availability of shadow/canary evaluation service | 99.95% | Runs alongside live serving path |
| Throughput | Support 50+ concurrent model pipelines across titles; ingest 200K events/sec telemetry peak (live-service launch day) | EA-scale peak traffic |
| Consistency | Strong consistency for model registry state transitions (no two promotions racing); eventual consistency acceptable for drift dashboards | |
| Cost | GPU training cost < $150K/month across all titles at steady state | Spot + scheduling levers |
| Durability | Zero data loss on training datasets and model artifacts (11 nines via object storage) | |
| Auditability | Every promotion/rollback traceable to triggering signal, approver, gate results | Compliance (loot box / regional regulation contexts) |

## 4. Clarifying Questions an Interviewer Would Expect

1. How many distinct model families/titles must this platform support simultaneously — one game or all of EA?
2. What's the acceptable blast radius of a bad promotion — can a bad matchmaking model go to 100% of players, or is canary mandatory?
3. Is human approval required for every promotion, or only when automated gates are borderline?
4. What's the retraining cadence expectation per model class — hourly (fraud), daily (churn), weekly (recsys embeddings)?
5. Do we own feature store and labeling infra, or integrate with existing ones?
6. What's the rollback SLA — seconds (serving-level flag flip) or minutes (redeploy)?
7. Are there regulatory constraints (COPPA, GDPR, loot-box regulations) affecting what data can be used for training or requiring explainability?
8. Is training done in-house on EA GPU clusters or via a cloud provider (AWS/GCP) with reserved + spot capacity?
9. What's the tolerance for false-positive drift triggers causing wasted retraining spend vs. false negatives causing stale models in production?
10. Do shadow/canary evaluations require real player traffic replay, or is a held-out offline test set sufficient for some model classes?

## 5. Assumptions

1. Platform serves 6 major live-service titles initially, ~40 production model endpoints total.
2. Peak telemetry ingestion: 200K events/sec (new title launch or LiveOps event), steady-state ~30K events/sec.
3. Median model: gradient-boosted tree or small deep net (10M-200M params), trained on 50-500GB feature snapshots.
4. One large model class exists: toxicity/chat classifier fine-tuned from a 1-3B parameter transformer.
5. GPU fleet: mix of on-prem A100s (reserved capacity) + cloud spot A10G/A100 burst capacity.
6. Human-in-the-loop approval required for full rollout; automated approval permitted for canary stage if gates pass with margin.
7. Rollback SLA target: < 60 seconds via feature-flag/model-version pointer flip (no redeploy needed).
8. Training data retention: 13 months rolling (for seasonality) in cold storage, 30 days hot.
9. All services deployed on Kubernetes (EKS-equivalent), same infra family as EA's existing platform services.
10. Drift detection runs on streaming feature snapshots sampled at 1-5% of full traffic for cost control, full-fidelity for fraud models.

## 6. Capacity Estimation

**Telemetry ingestion**
- Peak: 200K events/sec × ~1KB/event (player action + context) = 200 MB/sec ≈ 17.3 TB/day peak-equivalent, realistically ~30K events/sec steady = 30MB/sec ≈ 2.6TB/day steady across all titles.

**Feature store writes**
- Assume 20% of raw events become materialized features: steady ~6K writes/sec, peak ~40K writes/sec.

**Training data volume per retrain cycle**
- Churn/LTV model: 500GB feature snapshot (30-day player rolling window, ~20M MAU × ~2.5KB/player-day features × 10 days sampled).
- Matchmaking skill model: 50GB per daily retrain (match-level features, ~5M matches/day × 10KB).
- Toxicity transformer fine-tune: 20GB text corpus per weekly retrain.

**GPU/CPU sizing**
- GBT models (CPU-bound): 32-64 vCPU jobs, ~20-40 min per training run. ~10 model families × 1 run/day = 10 jobs/day, fits on a 256-vCPU autoscaled CPU pool.
- Deep net matchmaking model: 8× A100 (data-parallel), ~90 min/run, daily cadence → 1 job/day/title × 6 titles = 6 jobs/day needing 8 GPUs each = need pool of ~48 GPUs if run concurrently, or queue to run serially on 8-16 GPU reserved pool (accept longer wall time).
- Toxicity transformer fine-tune (1-3B params): 16× A100, ~4 hrs/run, weekly cadence = low absolute GPU-hours (16×4×6 titles×4/month ≈ 1,536 GPU-hrs/month) but bursty.
- Total steady-state reserved GPU pool: 32 A100s on-prem; burst to cloud spot for launch-day drift storms (up to +64 A100 spot).

**Back-of-envelope training cost**
- Cloud A100 spot ≈ $1.5/hr. Monthly GPU-hours across all jobs ≈ (48 GPUs × 1.5hr × 6 jobs/day × 30) for matchmaking + 1,536 (toxicity) + misc ≈ ~14,500 GPU-hrs/month × $1.5 ≈ $21.7K/month cloud burst; on-prem reserved amortized cost separately budgeted — total stays under $150K/month NFR including storage/compute overhead.

**Storage**
- Hot feature store (30 days): 2.6TB/day × 30 ≈ 78TB (steady-state; provisioned for 150TB with peak days).
- Cold training archive (13 months): ~2.6TB/day × 30% sampled × 395 days ≈ 308TB, lifecycle-tiered to cheap object storage (Glacier-equivalent).
- Model registry artifacts: 40 endpoints × ~20 versions retained × avg 2GB (deep net) = 1.6TB; GBT models negligible (~MBs).

**Shadow/canary evaluation load**
- Shadow scoring adds 100% duplicate inference load on shadowed traffic subset (sampled 5-10% of production QPS per model) — e.g., matchmaking at 5K predictions/sec production → shadow at 250-500/sec, negligible relative to primary serving fleet.

## 7. High-Level Architecture

```
                         ┌───────────────────────────────────────────────┐
                         │              GAME CLIENTS / SERVERS            │
                         │   (Apex, FC, Sims, Battlefield telemetry)      │
                         └───────────────────┬────────────────────────────┘
                                              │ events (Kafka producers)
                                              ▼
                         ┌───────────────────────────────────────────────┐
                         │        STREAMING BUS (Kafka / MSK)             │
                         │  topics: player.events, match.results,         │
                         │  chat.messages, model.predictions              │
                         └───────┬───────────────────────┬────────────────┘
                                 │                        │
                     ┌───────────▼──────────┐   ┌─────────▼─────────────┐
                     │ FEATURE PIPELINE     │   │ DRIFT DETECTION SVC    │
                     │ (Flink/Spark Struct  │   │ streaming stats (PSI,  │
                     │  Streaming) → Feature│   │ KS-test) + perf-metric │
                     │ Store (online+offline)│  │ comparators            │
                     └───────────┬──────────┘   └─────────┬─────────────┘
                                 │                          │ drift signal
                                 │                          ▼
                                 │              ┌─────────────────────────┐
                                 │              │ RETRAIN ORCHESTRATOR    │
                                 │              │ (Airflow/Argo Workflows)│
                                 │              │ - schedule triggers     │
                                 │              │ - drift triggers        │
                                 │◄─────────────┤ - manual triggers       │
                                 │  reads        └───────────┬─────────────┘
                                 │  training data             │ launches
                                 ▼                            ▼
                     ┌─────────────────────┐      ┌───────────────────────┐
                     │ OFFLINE FEATURE      │      │ TRAINING CLUSTER      │
                     │ STORE (point-in-time)│      │ (K8s + GPU/CPU pools, │
                     │ Parquet/Delta on S3  │      │  Ray/Horovod for DDP) │
                     └─────────────────────┘      └───────────┬────────────┘
                                                                │ candidate model
                                                                ▼
                                                   ┌─────────────────────────┐
                                                   │ VALIDATION GATE SVC     │
                                                   │ offline metrics, bias,  │
                                                   │ data-quality, regression│
                                                   └───────────┬─────────────┘
                                                     pass│         │fail
                                                         ▼         ▼
                                        ┌─────────────────────┐  reject + alert
                                        │ MODEL REGISTRY       │
                                        │ (versions, lineage,  │
                                        │  audit trail)        │
                                        └───────────┬───────────┘
                                                     │ candidate flagged "shadow-ready"
                                                     ▼
                                        ┌─────────────────────────┐
                                        │ SHADOW EVALUATION SVC    │◄──── live prod traffic mirror
                                        │ scores traffic in parallel│
                                        │ compares vs champion      │
                                        └───────────┬───────────────┘
                                             pass │      │ fail
                                                  ▼      ▼
                                    ┌───────────────────┐  reject + alert
                                    │ CANARY CONTROLLER  │
                                    │ 1% → 10% → 50% →   │
                                    │ 100% traffic ramp   │
                                    └─────────┬───────────┘
                                              │ gate pass at each stage
                                              ▼
                                    ┌───────────────────────┐
                                    │ PRODUCTION MODEL       │
                                    │ SERVING (Triton/KServe)│
                                    └─────────┬───────────────┘
                                              │ predictions + outcomes
                                              ▼
                                    ┌───────────────────────┐
                                    │ ROLLBACK CONTROLLER    │
                                    │ (auto on SLO breach)   │
                                    └───────────────────────┘
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| Streaming Bus (Kafka) | Durable, ordered event transport for telemetry, match results, predictions | Topic pub/sub, Avro/Protobuf schemas | Partitions per topic (scale by title + event type) |
| Feature Pipeline | Transform raw events → online (low-latency) + offline (point-in-time) features | Flink jobs consuming Kafka, writing to Redis (online) + Delta Lake (offline) | Flink task parallelism, per-topic consumer groups |
| Drift Detection Service | Compute streaming distributional stats (PSI, KL-divergence, KS-test) and performance deltas (accuracy, AUC via delayed labels) per deployed model | Consumes prediction + outcome topics, exposes `/drift/{model_id}` metrics API | Horizontally scaled stateless consumers, one partition group per model |
| Retrain Orchestrator | DAG scheduling of data-prep → train → validate → register; handles triggers (cron, drift, manual) | Argo Workflows CRDs / Airflow DAGs; REST trigger API | One orchestrator control plane, workers autoscale on queue depth |
| Offline Feature Store | Point-in-time correct historical features for training | Parquet/Delta tables on S3, queried via Spark/Trino | Partitioned by title + date |
| Training Cluster | Execute distributed training jobs (DDP, Horovod, Ray Train) | K8s Jobs with GPU node selectors; job spec includes data version + code version | GPU node pool autoscaler (cluster autoscaler + node groups) |
| Validation Gate Service | Run offline eval suite: holdout metrics vs. champion, fairness slices, data-quality checks (schema, nulls, cardinality), performance regression thresholds | Batch job invoked post-training, writes pass/fail + report to Model Registry | Stateless batch workers |
| Model Registry | Source of truth for model versions, lineage (data/code/hyperparam hashes), stage (staging/shadow/canary/prod), audit trail | REST/gRPC API (MLflow-like); backed by Postgres + S3 for artifacts | Postgres read replicas; artifact store scales via object storage |
| Shadow Evaluation Service | Mirror live traffic to candidate model, log predictions without serving them, compare vs. champion outcomes over bake window | Sidecar/shadow-router intercepting serving requests | Scales with sampled shadow traffic %, independent pool from prod serving |
| Canary Controller | Manage staged traffic ramp with automatic gate checks (latency, error rate, business metric deltas) at each stage | Works with service mesh (Istio/Linkerd) traffic splitting | Control-plane singleton per model; stateless |
| Model Serving | Serve production predictions at required latency/throughput | Triton Inference Server / KServe, gRPC + REST | Horizontal pod autoscaling on QPS/GPU util |
| Rollback Controller | Detect SLO breach or gate failure post-promotion, flip model-version pointer to last-known-good | Watches serving metrics, calls Registry API to demote | Stateless controller, single active instance w/ leader election |

## 9. API Design

Base path: `/api/v1`. All endpoints authenticated via mTLS (service-to-service) or OAuth2 (human/UI).

| Endpoint | Method | Purpose |
|---|---|---|
| `/models/{model_id}/versions` | GET | List all versions + stage + metrics |
| `/models/{model_id}/versions/{version}/promote` | POST | Request promotion to next stage |
| `/models/{model_id}/versions/{version}/rollback` | POST | Force rollback to previous prod version |
| `/models/{model_id}/drift` | GET | Current drift metrics (PSI, KS-stat, perf delta) |
| `/models/{model_id}/retrain` | POST | Manually trigger retrain pipeline |
| `/pipelines/{run_id}` | GET | Retrain pipeline run status + stage logs |
| `/pipelines/{run_id}/gates` | GET | Validation gate results detail |
| `/shadow/{model_id}/report` | GET | Shadow evaluation comparison report |
| `/canary/{model_id}/status` | GET | Current canary ramp %, health metrics |

**Example: trigger retrain**
```json
POST /api/v1/models/matchmaking-skill-fc25/retrain
{
  "trigger_reason": "drift_detected",
  "drift_report_id": "drift-8891",
  "requested_by": "system:drift-detector",
  "training_config_override": null
}
```
Response:
```json
{
  "run_id": "run-20260708-0421",
  "status": "QUEUED",
  "estimated_start": "2026-07-08T04:25:00Z"
}
```

**Example: promote candidate**
```json
POST /api/v1/models/matchmaking-skill-fc25/versions/v47/promote
{ "target_stage": "canary", "approved_by": "auto-gate", "canary_start_pct": 1 }
```
Response:
```json
{ "version": "v47", "stage": "canary", "traffic_pct": 1, "promoted_at": "2026-07-08T09:00:00Z" }
```

**Versioning**: URI-versioned (`/v1`), additive-only changes within major version; breaking changes (schema removal) require `/v2` with 90-day dual-run deprecation window. Model version strings are immutable content-addressed (`v47` maps to a hash of data+code+hyperparams).

## 10. Database Design

| Store | Type | Used For | Partitioning/Sharding Key |
|---|---|---|---|
| Model Registry metadata | PostgreSQL (relational) | Model versions, stages, lineage, approvals — needs strong consistency + transactions for stage transitions | Sharded by `model_id` hash if scale demands; single primary sufficient at 40 models |
| Offline feature store | Delta Lake / Parquet on S3 (columnar) | Point-in-time training snapshots, large scan-heavy reads | Partitioned by `title`, `date` |
| Online feature store | Redis / DynamoDB (KV) | Low-latency feature lookups at inference time | Partitioned by `player_id` |
| Drift metrics time series | Time-series DB (Timescale/Prometheus remote-write) | Streaming PSI/KS-stat history per model | Partitioned by `model_id` + time bucket |
| Event log (raw telemetry) | Kafka (durable log) → cold to S3/Parquet | Replayable source of truth | Partitioned by `title` + `event_type`, Kafka partitions by `player_id` hash |
| Shadow/canary evaluation results | Columnar (ClickHouse) | High-cardinality comparison queries (per-segment metric deltas) | Partitioned by `model_id`, `date` |
| Audit trail | Append-only table in Postgres (or dedicated audit log service) | Compliance: who approved/rejected what, when | Partitioned by `model_id` + year |

**Why relational for registry**: promotion/rollback is a state machine requiring ACID transactions (two operators can't both promote the same model version simultaneously — need row-level locking). **Why columnar for features/drift**: append-heavy, scan-heavy analytical workloads benefit from columnar compression and predicate pushdown.

## 11. Caching

| Cached Item | Strategy | Invalidation |
|---|---|---|
| Online features (per-player) | Cache-aside, Redis | TTL 15 min + explicit invalidation on new feature-pipeline write |
| Model artifact (for serving pods) | Write-through to local NVMe/pod cache on load | Invalidated on new version promotion; old version evicted after canary completes and no rollback window remains (24h grace) |
| Drift threshold config | Cache-aside, in-memory per drift-detector instance | Invalidated via config-change pub/sub event (Kafka topic `config.updates`) |
| Model Registry "current prod version" pointer | Cache-aside at serving gateway (avoids Postgres hit per request) | Invalidated immediately via pub/sub on promote/rollback (must be near-synchronous — this is the one cache where staleness directly causes serving wrong model) |
| Validation gate baseline metrics (champion's last eval) | Cache-aside | Invalidated when champion changes |

Cache-aside dominates because most reads are much more frequent than writes (feature reads >> feature writes); write-through used only for the model-version pointer where staleness risk is highest.

## 12. Queues & Async Processing

| Queue | Purpose | Delivery Semantics | Dead-Letter Handling |
|---|---|---|---|
| `retrain.requests` | Orchestrator trigger queue (drift/schedule/manual) | At-least-once (Kafka) | Failed dequeues after 3 retries → DLQ `retrain.requests.dlq`, paged to ML on-call |
| `training.jobs` | Dispatch training job specs to K8s job controller | At-least-once, idempotent job IDs (dedupe by `run_id`) | Job launch failures after 5 retries → DLQ, alert |
| `validation.results` | Gate results → Registry update | At-least-once, Registry update is idempotent upsert keyed on `version_id` | DLQ + manual replay tool |
| `shadow.predictions` | Async logging of shadow scores for later comparison | At-least-once, dedup on `request_id` | DLQ; dropped shadow predictions degrade sample size but don't block prod |
| `model.promotions` | Fan-out promotion events to serving fleet, canary controller, audit log | At-least-once, consumers idempotent (apply-if-version-newer) | DLQ + automatic retry with backoff; critical path, paged if DLQ depth > 0 |

Exactly-once avoided deliberately (adds complexity/latency); all consumers designed idempotent instead — standard tradeoff for this domain since duplicate drift signals or duplicate gate-checks are harmless if deduped by ID.

## 13. Streaming & Event-Driven Architecture

| Topic | Schema (key fields) | Producers | Consumer Groups |
|---|---|---|---|
| `player.events` | `player_id, event_type, timestamp, title, payload` (Avro) | Game clients/servers | feature-pipeline, drift-detector (sampled) |
| `match.results` | `match_id, players[], outcome, skill_deltas` | Game backend | feature-pipeline, matchmaking-model-drift-detector |
| `model.predictions` | `request_id, model_id, version, features_hash, prediction, latency_ms` | Serving fleet | drift-detector, shadow-eval-comparator, audit-logger |
| `model.outcomes` (delayed labels) | `request_id, actual_label, observed_at` | Label-generation jobs (e.g., churn realized 30 days later) | drift-detector (concept drift), validation-gate (regression checks) |
| `model.promotions` | `model_id, version, from_stage, to_stage, approver, timestamp` | Registry / Canary Controller | canary-controller, audit-logger, notification-service |
| `config.updates` | `config_key, new_value, updated_by` | Admin UI | all config-cache-holding services |

Consumer groups sized 1:1 with logical scaling unit (e.g., drift-detector group has one consumer per model-partition-group so a single model's drift computation doesn't bottleneck on another's).

## 14. Model Serving

- **Framework**: Triton Inference Server for deep nets (matchmaking, toxicity transformer) — supports dynamic batching, multi-framework (PyTorch/TensorRT), concurrent model execution. KServe wraps it for K8s-native autoscaling + canary routing. GBT models (churn/LTV) served via lightweight custom Python/ONNX runtime sidecars — overkill to put XGBoost on Triton GPU infra.
- **Batching**: dynamic batching window 5-10ms for matchmaking (latency-sensitive, real-time queue), up to 100ms batching window acceptable for churn/LTV (batch nightly scoring, not per-request).
- **Multi-model**: single Triton instance hosts multiple model versions concurrently (champion + canary) via model repository versioning — enables instant traffic-split without redeploying pods.
- **Hardware**: matchmaking/toxicity on GPU (T4/A10G, cost-optimized for inference vs A100 training); GBT models on CPU-only pods.
- **Shadow duplication**: shadow requests routed via sidecar proxy that forwards a sampled copy async (fire-and-forget, doesn't block prod response path) to the candidate model instance.

## 15. Feature Store

- **Online store**: Redis/DynamoDB, p99 read < 5ms, serves inference-time feature lookups (e.g., player's rolling 7-day KDA, recent chat toxicity score).
- **Offline store**: Delta Lake on S3, serves training data generation and backtesting.
- **Point-in-time correctness**: every feature write is timestamped; training data joins use as-of joins (feature value "as it was" at label-generation time, not current value) to prevent label leakage — critical because a churn label generated today must join against features computed before the churn event, not after. Implemented via Delta Lake time-travel queries / point-in-time join framework (Feast-style).
- **Feature freshness SLA**: online features refreshed within 5 min of source event for behavioral features; some features (season-long stats) refreshed daily batch.
- **Consistency between online/offline**: same feature transformation logic (shared feature definition code) compiled to both a streaming (Flink) and batch (Spark) execution path to avoid train/serve skew.

## 16. Vector Database

N/A for the core retraining/drift/promotion loop described in this chapter — this platform's primary model classes (matchmaking skill, churn/LTV, toxicity classification, fraud) are not similarity-search-driven. A vector DB *would* apply if this platform also managed a recommendation/embedding-based matchmaking model requiring nearest-neighbor player-similarity lookups — in that case an ANN index (HNSW via a system like pgvector/Milvus) would sit in the Feature Store layer for online similarity queries at serving time, not in the training/promotion pipeline itself. Not detailed further here since it's out of scope for this chapter's named systems (drift, retraining, gating, shadow, rollback).

## 17. Embedding Pipelines

Partially applicable: the toxicity/chat classifier consumes text embeddings (from a fine-tuned transformer's encoder) as an intermediate representation, but there is no standalone embedding-serving pipeline as a first-class platform component here — embeddings are computed inline as part of the model's forward pass during both training and serving, not precomputed/stored/reused across models. If EA later builds a shared "player embedding" service (e.g., a universal player-behavior embedding reused across recsys, matchmaking, and churn models), that would warrant its own embedding pipeline chapter (batch embedding generation, versioning, staleness handling) — marking N/A here to avoid scope creep beyond the named system boundaries.

## 18. Inference Pipelines (Request Lifecycle End-to-End)

```
Player action (e.g., queues for match)
        │
        ▼
Game backend calls Serving Gateway  ──────────────► [gateway checks cached
        │                                             "current prod version"
        │                                             pointer, ~0.1ms]
        ▼
Feature fetch (online store, Redis)  ~3-5ms
        │
        ▼
Primary inference request → Triton (champion model)  ~8-15ms (batched)
        │                         │
        │                         └──► Shadow router (async, fire-and-forget)
        │                                    │
        │                                    ▼
        │                          Shadow/candidate model inference
        │                                    │
        │                                    ▼
        │                          Log to shadow.predictions topic
        │                          (compared later, does not block response)
        ▼
Response returned to game backend  (total p50 ~15ms, p99 ~45ms)
        │
        ▼
Prediction logged to model.predictions topic (async)
        │
        ▼
[30 days later] Outcome observed (e.g., player churned or not)
        │
        ▼
model.outcomes topic → drift-detector computes concept-drift delta
```

## 19. Training Pipelines

- **Data prep**: Orchestrator DAG step pulls point-in-time feature snapshot from offline store, runs data-quality checks (schema validation, null-rate thresholds, label leakage checks), splits train/val/test with time-based split (never random shuffle — prevents future leakage).
- **Training orchestration**: Argo Workflows DAG: `data_prep → train → offline_eval → validation_gate → register`. Each step is a K8s Job with resource requests matching model class (CPU pool for GBT, GPU pool for deep nets).
- **Distributed training**: deep nets (matchmaking, toxicity) use PyTorch DDP across 8-16 GPUs via Ray Train/Torchrun; gradient checkpointing for the 1-3B toxicity transformer to fit batch sizes on A100-40GB. GBT models (XGBoost/LightGBM) use distributed histogram-based training across CPU workers (Dask-XGBoost) for the 500GB churn dataset.
- **Lineage tracking**: every run records data snapshot hash, code commit SHA, hyperparameter config, environment/container digest — all stored in Model Registry for full reproducibility.
- **Hyperparameter search**: lightweight Bayesian search (Optuna) for scheduled full retrains; incremental fine-tuning (warm-start from champion weights) for drift-triggered fast retrains to cut training time.

## 20. Retraining Strategy

| Model Class | Cadence | Triggers |
|---|---|---|
| Fraud/cheat detection | Continuous (near-real-time incremental updates) + full retrain every 6 hrs | Concept drift (precision/recall drop > threshold), adversarial pattern spike |
| Matchmaking skill | Daily scheduled + drift-triggered | Data drift (PSI > 0.2 on skill feature distribution), scheduled cron, major patch/meta-shift event |
| Churn/LTV | Weekly scheduled + drift-triggered | Concept drift (calibration error increase), content-drop events (manual trigger from LiveOps calendar) |
| Toxicity/chat classifier | Weekly scheduled + drift-triggered | Data drift on vocabulary/slang distribution (embedding-space drift), manual trigger after moderation policy change |
| Recommendation/personalization | Daily incremental, weekly full retrain | Engagement metric drop, catalog change (new item drop) |

Drift-triggered retrains take priority queue position over scheduled ones; scheduled retrains still run as a backstop even if drift detectors are silent (catches drift the detectors miss).

## 21. Drift Detection

| Drift Type | Metric | Threshold (example: matchmaking model) | Action |
|---|---|---|---|
| Data drift (input feature distribution) | PSI (Population Stability Index) per feature | PSI > 0.2 (moderate) → warn; PSI > 0.3 (major) → trigger retrain | Retrain trigger fired to orchestrator |
| Data drift (categorical features) | Chi-squared / KL-divergence | KL > 0.15 | Retrain trigger |
| Concept drift (label relationship change) | Rolling AUC/accuracy on delayed labels vs. training-time baseline | AUC drop > 3 points sustained over 24h window | Retrain trigger + page on-call if drop > 8 points (active incident) |
| Concept drift (calibration) | Expected Calibration Error (ECE) | ECE increase > 0.05 | Retrain trigger |
| Prediction drift (proxy when labels delayed) | PSI on prediction score distribution | PSI > 0.25 | Early-warning flag, not auto-trigger (noisy proxy) |
| Embedding-space drift (toxicity model) | Centroid distance / MMD on embedding distribution | MMD > empirical 95th percentile of historical baseline | Retrain trigger |

Statistical tests run on streaming windows (1hr tumbling + 24hr rolling) to separate noise from sustained drift; require 2 consecutive windows breaching threshold before firing a trigger (debounce to avoid retrain thrashing).

## 22. Monitoring

| Category | Metrics |
|---|---|
| Infra | K8s pod health, GPU utilization %, training job queue depth, Kafka consumer lag, feature-store read/write latency |
| Pipeline health | Retrain success/failure rate, pipeline duration (p50/p99), gate pass/fail rate, time-to-promote |
| Model quality | Offline eval metrics per version (AUC, F1, calibration), shadow-vs-champion delta, canary-stage metric deltas |
| Drift | PSI/KL per feature per model, concept-drift AUC delta, embedding-drift MMD |
| Business | Match quality (win-rate balance), player retention delta post-promotion, false-positive ban rate (fraud model), toxicity false-positive/negative rate |
| Cost | GPU-hours consumed per model family per day, spot vs. on-demand spend ratio |

## 23. Alerting

| Alert | Condition | Routing |
|---|---|---|
| Retrain pipeline failure | Job fails after all retries | Page ML on-call (PagerDuty), Slack `#mlops-alerts` |
| Drift trigger storm | > 5 models trigger drift-retrain within 1hr | Page ML on-call — likely upstream data pipeline issue, not real drift |
| Validation gate rejection | Candidate fails gate | Notify owning DS team (Slack), no page (non-urgent) |
| Shadow evaluation regression | Candidate underperforms champion by > threshold during bake window | Notify owning DS team, block auto-promotion |
| Canary SLO breach | Latency p99 > budget or error rate > 1% during canary stage | Auto-rollback + page SRE on-call |
| Concept drift severe (AUC drop > 8pts) | Sustained 2 windows | Page ML on-call immediately (model actively degrading in prod) |
| Registry promotion race/conflict | Two concurrent promotion attempts on same version | Page ML platform team (control-plane bug signal) |
| DLQ depth > 0 on `model.promotions` | Any message | Page SRE on-call (critical path) |

## 24. Logging

- **Structured logging**: JSON logs with correlation `request_id`/`run_id` across all services (orchestrator, training jobs, gate service, shadow service) — enables tracing one retrain cycle across every component.
- **PII handling**: player-identifying fields (`player_id`, IP, chat text) hashed/tokenized before entering training logs; raw chat text for toxicity training stored in a restricted-access enclave with field-level encryption, never in general-purpose logs.
- **Retention**: operational logs 30 days hot (searchable, e.g., OpenSearch), 1 year cold (S3, compliance/audit). Audit trail (promotions/rollbacks/approvals) retained 3 years for compliance review.
- **Redaction**: log pipeline applies a PII-scrubbing filter (regex + ML-based PII detector) before indexing into the general observability stack; only the restricted enclave gets unredacted chat content.

## 25. Security

- **Threat model specifics**:
  - Data poisoning: adversary submits crafted telemetry (e.g., bots feeding fake match results) to bias retraining → mitigated by anomaly detection on training data ingestion + data-quality gate rejecting statistically anomalous batches.
  - Model exfiltration: attacker with registry access downloads proprietary matchmaking/anti-cheat model weights → mitigated by artifact encryption at rest, access-scoped IAM roles, audit logging on every artifact download.
  - Promotion pipeline compromise: attacker forges a "gate passed" signal to force a malicious model into production → mitigated by signed gate results (service identity signs the pass/fail verdict), Registry verifies signature before allowing stage transition.
  - Adversarial drift gaming: cheaters intentionally shift behavior to induce retraining that "learns" cheat patterns as normal → mitigated by held-out adversarial test sets that don't get overwritten by drift-triggered retrains, human review required for fraud-model promotions.
- **Encryption**: TLS 1.3 in transit for all service-to-service calls; AES-256 at rest for feature store, training data, model artifacts; field-level encryption for chat text in toxicity pipeline.
- **Least privilege**: each service has scoped IAM role (training jobs can read offline store + write to registry staging area only; serving fleet can read registry prod pointer only, not write).

## 26. Authentication

- **Service-to-service**: mTLS via service mesh (Istio) — each service has a SPIFFE identity; the orchestrator, gate service, registry, and serving fleet mutually authenticate certs, no shared secrets.
- **Human/UI access**: OAuth2/OIDC via EA's internal SSO; role-based access control — DS engineers can trigger retrains/view drift for their own models, only ML platform admins can modify gate thresholds or force-promote bypassing gates.
- **CI/CD pipeline identity**: training job containers assume a workload identity (IRSA-equivalent) scoped to read specific S3 prefixes and write to a specific registry namespace — no long-lived credentials baked into images.

## 27. Rate Limiting

- **Retrain trigger API**: token-bucket limiter per model_id — max 1 manual retrain trigger per model per 10 minutes (prevents accidental retrain storms from UI misclicks or buggy automation), burst of 3 allowed for legitimate rapid-iteration during incident response.
- **Drift-detector-fired triggers**: not rate-limited by the same bucket but debounced (section 21) — separate circuit-breaker: if a single model fires > 3 drift-triggered retrains in 24h, auto-pause further auto-triggers and page on-call (signals detector misconfiguration rather than real drift).
- **Registry API (read)**: per-client rate limit 100 req/sec (sliding window) — dashboards/UI polling shouldn't overwhelm Postgres.
- **Shadow traffic sampling**: rate-limited at the router level to cap shadow inference load at 10% of primary traffic regardless of configured sample rate, as a safety valve against shadow service resource exhaustion impacting prod (shared node pool risk).

## 28. Autoscaling

- **Training GPU pool**: cluster-autoscaler + KEDA scaling on `training.jobs` queue depth — scale node group 0→N GPU nodes based on pending job count, scale-to-zero when idle (cost lever).
- **Serving fleet (Triton/KServe)**: HPA on custom metric `inference_queue_time_ms` (better signal than CPU/GPU util alone for batched inference) + `gpu_utilization` from DCGM exporter; target GPU util 70%.
- **Feature pipeline (Flink)**: VPA for per-task-manager memory tuning + manual parallelism bump during known launch-day traffic spikes (proactive scaling ahead of predicted LiveOps events, not purely reactive).
- **Drift detector**: HPA on Kafka consumer lag (KEDA Kafka scaler) — lag > 10K messages triggers scale-out.
- **Shadow evaluation service**: HPA on shadow-traffic QPS, capped hard at a max replica count to enforce the rate-limit safety valve above.

## 29. Cost Optimization

- **Spot instances**: all training jobs (not serving) run on spot GPU capacity with checkpoint-resume every N steps — training-job design tolerates spot preemption by resuming from last checkpoint (adds ~5% overhead, saves ~65-70% vs on-demand).
- **Scale-to-zero**: training GPU pool scales to 0 nodes when no jobs queued (most of the day for daily/weekly cadence models).
- **Model distillation**: toxicity transformer distilled to a smaller student model for serving (keep large model only for periodic label-generation/teacher retraining), cuts serving GPU cost ~4x.
- **Batching**: dynamic batching on Triton amortizes GPU cost across concurrent requests; nightly batch scoring for churn/LTV instead of real-time serving where business requirement allows.
- **Sampling for drift detection**: compute full-fidelity drift only for high-risk models (fraud); 1-5% sampling for lower-risk models cuts drift-compute cost substantially with negligible statistical power loss at EA's traffic volume.
- **Storage tiering**: cold training data (>30 days) auto-lifecycles to cheaper storage tier; feature snapshots deduplicated across overlapping training windows.
- **Reserved + spot hybrid**: baseline steady-state GPU need covered by reserved/on-prem capacity (cheaper amortized), burst-only workloads (launch days, drift storms) use spot/on-demand cloud.

## 30. Disaster Recovery

| Target | Value |
|---|---|
| RTO (control plane: orchestrator, registry) | 30 minutes |
| RPO (model registry metadata) | 5 minutes (Postgres continuous WAL archiving + point-in-time restore) |
| RTO (serving fleet, if region fails) | 5 minutes (failover to standby region, section 31) |
| RPO (training data) | 0 (Kafka durable log + S3 versioned buckets, replayable from source) |
| Backup strategy | Postgres: automated snapshots every 6h + continuous WAL shipping to S3. Model artifacts: S3 cross-region replication. Kafka: replication factor 3 across AZs. |
| DR drill cadence | Quarterly game-day: simulate registry DB loss, restore from backup, verify last-known-good model still servable within RTO |

## 31. Multi-Region Deployment

```
        Region: US-EAST (primary/active)          Region: EU-WEST (active)
   ┌───────────────────────────────┐        ┌───────────────────────────────┐
   │ Serving Fleet (Triton/KServe) │        │ Serving Fleet (Triton/KServe) │
   │ Feature Store (online, local) │        │ Feature Store (online, local) │
   │ Model Registry (read replica) │◄──────►│ Model Registry (read replica) │
   └───────────────┬───────────────┘  sync  └───────────────┬───────────────┘
                    │                                        │
                    └───────────────┬────────────────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │  Model Registry PRIMARY (US)   │
                     │  Training Orchestrator (US)    │
                     │  Training GPU Cluster (US)     │
                     └───────────────────────────────┘
```

- **Topology**: active-active for *serving* (players routed to nearest region for latency — matchmaking/toxicity inference is latency-sensitive), active-passive for *training/orchestration control plane* (single source of truth in US-EAST to avoid split-brain on promotion decisions; EU registry is a read replica that receives promotion events async).
- **Data replication**: model artifacts replicated cross-region via S3 CRR (typically < 2 min lag); registry promotion events propagated via a global event bus (Kafka MirrorMaker or equivalent) so EU serving fleet picks up new prod model version within seconds of US promotion.
- **Latency routing**: GeoDNS/Anycast routes player traffic to nearest serving region; feature store online reads stay region-local (no cross-region hop on the hot path).
- **Failure handling**: if US-EAST control plane fails, EU can continue serving the last-known-good model (registry read replica still has last-synced state) but cannot process new promotions/retrains until US recovers or a manual failover promotes EU registry replica to primary.

## 32. Blue/Green Deployment

Applied at the **model-serving version** level, not infra level: Triton hosts both "blue" (current champion) and "green" (newly-validated candidate) model versions simultaneously in the same model repository. Traffic-split is controlled by the Canary Controller's routing rule (not a full infra swap) — this is effectively blue/green *within* the serving layer, with the canary ramp (section 33) as the gradual cutover mechanism rather than an instantaneous full swap. Full instantaneous blue/green (100% cutover) is reserved for emergency rollback only (flip pointer back to blue instantly) — normal promotions always go through gradual canary, not a hard blue/green swap, because model-quality regressions are often subtle (calibration drift) and only surface under gradual real-traffic exposure.

## 33. Canary Deployment

| Stage | Traffic % | Duration (min) | Gate Checks |
|---|---|---|---|
| Shadow | 0% (mirrored, not served) | 24-48 hrs | Prediction distribution match, no serving-path errors, latency overhead acceptable |
| Canary-1 | 1% | 2-4 hrs | Error rate < 0.5%, p99 latency within 10% of champion, no business-metric regression (win-rate balance, ban false-positive rate) |
| Canary-2 | 10% | 4-8 hrs | Same gates + statistical significance check on key business metric delta |
| Canary-3 | 50% | 8-24 hrs | Same gates, sustained over longer window to catch delayed-label effects (e.g., churn impact) |
| Full rollout | 100% | — | Final gate pass + (for high-risk model classes) human sign-off |

Traffic split implemented via service-mesh weighted routing (Istio VirtualService weight fields) keyed on request hash (sticky per-player assignment within a stage, avoids a player flip-flopping between champion/candidate mid-session which would create inconsistent matchmaking experience).

## 34. Rollback Strategy

- **Automated triggers**: canary-stage SLO breach (latency, error rate), business-metric regression beyond threshold, concept-drift alarm firing on the *new* version post-full-rollout.
- **Mechanics**: Rollback Controller calls Registry API to flip the "current prod version" pointer back to last-known-good; because serving fleet reads this pointer via a near-real-time pub/sub-invalidated cache (section 11), rollback propagates to all serving pods within seconds — no redeploy, no pod restart, since both versions are already loaded in the Triton model repository during the canary window.
- **Post-rollback**: candidate automatically demoted to "rejected" stage in registry, owning DS team notified with the triggering metric snapshot attached, retrain-with-fix can be manually requested.
- **Rollback of a fully-promoted-and-old-version-evicted model**: if the previous champion's artifact was already evicted (past 24h grace period, section 11), rollback instead re-loads the artifact from the registry's S3 store — adds ~1-2 min load time, an accepted tradeoff vs. keeping every old version warm indefinitely.

## 35. Observability

- **Metrics**: Prometheus scrapes all services; Grafana dashboards per model_id (drift, gate pass rate, canary health) and per-pipeline-run.
- **Tracing**: OpenTelemetry spans across the full retrain lifecycle — one trace per `run_id` spans data-prep → train → gate → shadow → canary, so a Principal engineer debugging "why did promotion take 6 hours" can see exactly which stage stalled.
- **Logs**: correlated via `run_id`/`request_id` injected into every log line, queryable in OpenSearch; trace_id linked so a Grafana metric anomaly can pivot directly to the relevant trace and logs.
- **Correlation in practice**: a canary SLO-breach alert links to (a) the Grafana panel showing the metric spike, (b) the distributed trace of that specific inference request, (c) the structured logs from the serving pod at that timestamp, (d) the model_id's registry page showing which version was live — all cross-linked via shared IDs, not siloed dashboards.

## 36. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: drift-detector-matchmaking
  labels: { app: drift-detector, model: matchmaking-skill }
spec:
  replicas: 3
  selector: { matchLabels: { app: drift-detector, model: matchmaking-skill } }
  template:
    metadata: { labels: { app: drift-detector, model: matchmaking-skill } }
    spec:
      containers:
        - name: drift-detector
          image: ea-mlplatform/drift-detector:1.14.2
          resources:
            requests: { cpu: "1", memory: "2Gi" }
            limits: { cpu: "2", memory: "4Gi" }
          env:
            - name: MODEL_ID
              value: matchmaking-skill-fc25
            - name: KAFKA_BROKERS
              valueFrom: { configMapKeyRef: { name: kafka-config, key: brokers } }
---
apiVersion: v1
kind: Service
metadata: { name: drift-detector-matchmaking }
spec:
  selector: { app: drift-detector, model: matchmaking-skill }
  ports: [{ port: 8080, targetPort: 8080 }]
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: drift-detector-matchmaking-hpa }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: drift-detector-matchmaking }
  minReplicas: 2
  maxReplicas: 12
  metrics:
    - type: External
      external:
        metric: { name: kafka_consumergroup_lag, selector: { matchLabels: { topic: player.events } } }
        target: { type: AverageValue, averageValue: "10000" }
```

## 37. Terraform Infrastructure

```hcl
resource "aws_eks_node_group" "gpu_training_pool" {
  cluster_name    = aws_eks_cluster.ml_platform.name
  node_group_name = "gpu-training-spot"
  node_role_arn   = aws_iam_role.training_node_role.arn
  subnet_ids      = var.private_subnet_ids

  capacity_type  = "SPOT"
  instance_types = ["p4d.24xlarge", "p3.8xlarge"]

  scaling_config {
    desired_size = 0
    min_size     = 0
    max_size     = 16
  }

  labels = { workload = "training", accelerator = "nvidia-a100" }
  taint {
    key = "training-only", value = "true", effect = "NO_SCHEDULE"
  }
}

resource "aws_msk_cluster" "telemetry_bus" {
  cluster_name           = "ea-ml-telemetry"
  kafka_version          = "3.6.0"
  number_of_broker_nodes = 9

  broker_node_group_info {
    instance_type   = "kafka.m5.4xlarge"
    client_subnets  = var.private_subnet_ids
    storage_info { ebs_storage_info { volume_size = 2000 } }
  }
}

resource "aws_s3_bucket" "model_registry_artifacts" {
  bucket = "ea-ml-model-registry-artifacts"
}

resource "aws_s3_bucket_replication_configuration" "artifact_crr" {
  bucket = aws_s3_bucket.model_registry_artifacts.id
  role   = aws_iam_role.replication_role.arn
  rule {
    id     = "cross-region-replica"
    status = "Enabled"
    destination { bucket = aws_s3_bucket.model_registry_artifacts_eu.arn }
  }
}

resource "aws_db_instance" "model_registry_db" {
  identifier              = "model-registry-postgres"
  engine                  = "postgres"
  instance_class          = "db.r6g.xlarge"
  allocated_storage       = 200
  backup_retention_period = 7
  multi_az                = true
}
```

## 38. Why This Architecture

- Event-driven core (Kafka) decouples telemetry ingestion from feature computation, drift detection, and downstream consumers — necessary at EA scale where 6+ titles produce heterogeneous, bursty event volumes.
- Separating drift detection (streaming) from retraining (batch DAG) lets each scale independently — drift detection must be near-real-time (15 min), retraining tolerates hours.
- Staged promotion (validation gate → shadow → canary) matches the risk profile of live-service games where a bad model directly harms player experience and retention; no single gate is trusted alone.
- Model registry as the single source of truth for "what's live" enables instant, redeploy-free rollback — critical given the < 60s rollback SLA.
- Reusing the same orchestration framework across model classes (GBT and deep nets) reduces platform surface area vs. bespoke pipelines per team, while still allowing per-model resource/cadence customization.

## 39. Alternative Architectures

| Alternative | Description | Why Rejected / When Preferred |
|---|---|---|
| Fully manual retraining with CI/CD-triggered scripts | DS manually kicks off retrain scripts, reviews metrics in a notebook, files a deploy ticket | Rejected: too slow (1-2 week cycle) for live-service drift; would be acceptable for a low-stakes, low-drift offline model (e.g., an internal analytics model) |
| Single monolithic "AutoML" pipeline with no shadow/canary, direct-to-prod on gate pass | Simpler pipeline: train → validate → deploy immediately at 100% | Rejected: no real-traffic validation before full exposure, too risky for player-facing models; acceptable for very low-blast-radius internal models (e.g., a support-ticket triage classifier) |
| Fully synchronous exactly-once event pipeline (transactional outbox + exactly-once Kafka semantics everywhere) | Stronger delivery guarantees end-to-end | Rejected: added latency/complexity not justified since all consumers are idempotent by design; would reconsider if financial-transaction-level correctness were required (e.g., real-money payment fraud scoring with strict dedup requirements) |
| Per-title independent platforms (no shared orchestration) | Each game team builds its own retrain/drift/promote stack | Rejected: duplicated engineering effort across 6+ titles, inconsistent gating rigor; would make sense only if titles had wildly incompatible tech stacks or orgs were fully decentralized with no platform mandate |

## 40. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| At-least-once + idempotent consumers over exactly-once | Lower latency, simpler infra | Requires discipline — every consumer must implement correct dedup logic |
| Staged canary over instant blue/green promotion | Catches subtle regressions (calibration, delayed-label effects) | Slower time-to-full-rollout (can be 24-48+ hrs) |
| Shared orchestration platform across model classes | Less duplicated engineering, consistent gates | One-size-fits-all DAG framework adds abstraction overhead for unusual model classes |
| Active-passive control plane / active-active serving | Avoids split-brain on promotion decisions while keeping serving low-latency globally | EU can't independently promote/retrain during US outage |
| Sampling-based drift detection for low-risk models | Big cost savings | Slightly reduced statistical power to detect subtle drift early |
| Debounced drift triggers (2 consecutive windows) | Avoids retrain thrashing from noise | Adds detection latency (extra window before trigger fires) |

## 41. Failure Modes

| Scenario | Impact | Mitigation |
|---|---|---|
| Kafka broker outage in a region | Feature pipeline and drift detection stall for that region | Multi-AZ replication factor 3; cross-region consumers can fail over to mirrored topic |
| Training data corruption (bad upstream ETL) | Candidate model trained on garbage data, could pass gates if gates don't catch the specific corruption | Data-quality gate checks (schema, null-rate, distribution sanity) run before training even starts, not just after |
| Validation gate service bug lets a bad model pass | Bad model reaches shadow/canary | Shadow stage acts as second independent check against real traffic; canary auto-rollback catches what shadow misses |
| Registry Postgres primary failure | Can't record new promotions, but existing prod pointer still cached and servable | Multi-AZ RDS failover (~1-2 min), serving fleet unaffected due to cache |
| Drift-detector false-positive storm (e.g., a schema change misread as drift) | Wasted retrain compute, alert fatigue | Debounce windows + drift-trigger circuit breaker (section 27), schema-change events distinguished from statistical drift |
| GPU spot capacity unavailable during high-demand window | Training delayed, drift-triggered retrain SLA missed | Fallback to on-demand burst capacity for high-priority (fraud) jobs; lower-priority jobs queue and wait |
| Adversarial data poisoning (coordinated bot telemetry) | Skews retraining, potential silent model degradation | Anomaly detection on ingestion, held-out adversarial-resistant eval sets, human sign-off for fraud-model promotions |
| Canary stuck at low traffic % (gate never confidently passes) | Slow rollout, DS team blocked | Timeout + escalation policy: after 48h at a canary stage, auto-page owning team for manual decision rather than infinite wait |

## 42. Scaling Bottlenecks

- **At 10x event volume** (~2-3M events/sec): Kafka partition count and Flink parallelism become the first bottleneck — feature pipeline lag grows past freshness SLA; mitigation is partition rebalancing and Flink task-manager horizontal scale, already anticipated via KEDA-driven scaling, but Postgres-backed registry read load from more frequent drift/gate checks starts to strain single-primary writes.
- **At 100x** (~20-30M events/sec, EA-wide across all titles + new acquisitions): single-region Kafka cluster becomes untenable — would need topic sharding across multiple Kafka clusters per title-cluster with a federated drift-detection layer; offline feature store scan costs balloon (Parquet full-table scans for point-in-time joins) requiring more aggressive partitioning/Z-ordering or a move to a purpose-built feature-store product with better indexing.
- **Model registry at scale**: at 40 models this Postgres-backed design is fine; at 500+ models (hypothetical EA-wide platform including all internal ML use cases) would need sharding by model_id or a move to a distributed metadata store, since promotion transactions plus high-frequency drift-status reads compound.

## 43. Latency Bottlenecks

**Retrain pipeline (p50/p99 budget, target < 4hr for daily-cadence models)**

| Stage | p50 | p99 |
|---|---|---|
| Data prep (feature snapshot pull + quality checks) | 20 min | 45 min |
| Training (deep net, 8xA100 DDP) | 90 min | 150 min |
| Offline eval + validation gates | 15 min | 30 min |
| Registry registration + notification | 1 min | 3 min |
| **Total (excludes shadow/canary bake time)** | **~2.1 hr** | **~3.7 hr** |

Bottleneck: training stage dominates, especially GPU contention when multiple titles' jobs queue for the same reserved pool — mitigated by spot burst capacity, but spot preemption adds variance to p99.

**Inference request (from section 18), target p50 15ms / p99 45ms**

| Stage | p50 | p99 |
|---|---|---|
| Feature fetch (Redis) | 3ms | 8ms |
| Model inference (batched, Triton) | 8ms | 25ms |
| Network/gateway overhead | 2ms | 6ms |
| Serialization/misc | 2ms | 6ms |
| **Total** | **~15ms** | **~45ms** |

Bottleneck at p99: batching queue wait time when traffic bursts (launch-day matchmaking queue spikes) — mitigated by aggressive HPA on inference-queue-time metric, but there's an inherent floor since larger batches reduce per-request GPU cost but raise queueing latency (direct tradeoff dial).

## 44. Cost Bottlenecks

- **GPU training compute** is the single largest lever (~$20-25K/month cloud burst alone at estimated volumes) — dominated by the toxicity transformer's fine-tuning cost and matchmaking deep-net's daily cadence across 6 titles; distillation and incremental (warm-start) retraining instead of full retrains are the biggest cost-reduction levers.
- **Storage growth**: 308TB cold archive at 13-month retention is a steady linear cost driver — lifecycle policies to cheaper storage tiers and more aggressive sampling of what's retained (not every raw event needs 13-month retention, only aggregated features) directly cuts this.
- **Shadow evaluation duplicate inference**: scales with number of concurrent candidate models in bake-in windows — if many titles retrain simultaneously (e.g., post a company-wide platform update), shadow compute cost spikes; capped by the rate-limit safety valve (section 27) which trades cost control for slightly reduced shadow statistical power under contention.
- **Idle reserved GPU capacity**: over-provisioning the on-prem reserved pool for peak (rather than steady-state) wastes spend during off-peak; right-sizing reserved vs. spot-burst split requires ongoing tuning against actual utilization telemetry.

## 45. Interview Follow-Up Questions

1. How do you prevent a drift-triggered retrain storm from cascading into a cost incident?
2. Walk through what happens if the validation gate service itself has a bug that inverts pass/fail logic.
3. How do you guarantee point-in-time correctness when the online and offline feature stores are updated by different pipelines with different latencies?
4. What's your strategy if two model versions are promoted to canary simultaneously by two different engineers?
5. How would you detect concept drift for a model whose labels are delayed by 30+ days (e.g., churn)?
6. If the shadow evaluation shows the candidate is better on aggregate but worse on a specific player segment, what do you do?
7. How do you handle a rollback when the previous champion's artifact has already been evicted from the serving fleet?
8. What changes in this design if EA acquires a new studio and needs to onboard a 7th title with a completely different tech stack?
9. How do you reason about the tradeoff between fast incremental retraining (warm-start) and full retraining from scratch?
10. How would multi-region active-active serving handle a promotion approved in one region while another region is mid-incident?

## 46. Ideal Answers

1. **Retrain storm prevention**: Per-model rate limiting on retrain triggers (max 1 manual/10min) plus a circuit breaker that auto-pauses a model's drift-auto-triggering if it fires >3 times in 24h, escalating to on-call instead of retraining blindly — distinguishes real sustained drift from a noisy/misconfigured detector, and caps blast radius on GPU spend.

2. **Gate service inverted logic bug**: Defense in depth — even if the gate wrongly passes a bad candidate, the shadow evaluation stage independently re-validates against live traffic before any serving exposure, and canary stage 1 (1% traffic) has hard automatic SLO/business-metric rollback triggers independent of the gate's verdict. Additionally, gate results are signed by the gate service's identity and the registry logs every pass/fail with full metric payload for post-incident audit — an inverted-logic bug would still be caught by shadow/canary before meaningful player impact, and the audit trail lets you pinpoint exactly which promotions need retroactive review.

3. **Point-in-time correctness across differing pipeline latencies**: Every feature write carries an event-time timestamp (not processing-time), and training-data generation uses as-of/point-in-time joins against the offline store keyed on that event-time — so even if online and offline pipelines have different processing lag, the *logical* correctness of "what did this feature look like at label time" is preserved because it's a timestamp-based join, not a "current value" lookup. Shared feature-transformation code compiled to both streaming and batch paths further prevents train/serve skew from divergent logic (not just divergent timing).

4. **Concurrent promotion race**: Registry promotion is a transactional state-machine update in Postgres — the promote operation does a compare-and-swap on the model version's current stage (`WHERE stage = 'shadow'` before setting `stage = 'canary'`), so the second concurrent request fails the conditional update and gets a 409-equivalent conflict response, forcing that engineer to re-fetch current state rather than silently double-promoting.

5. **Concept drift with delayed labels**: Use two complementary signals — (a) a fast proxy: prediction-score distribution drift (PSI on model outputs) as an early-warning signal available immediately, since prediction drift often precedes/correlates with eventual concept drift; (b) the ground-truth signal once labels resolve (30-day churn outcome), computing rolling AUC/calibration against the delayed-label stream and comparing to the training-time baseline. The proxy triggers a "watch" state; the delayed ground-truth confirms and fires the actual retrain trigger, balancing responsiveness against waiting a full month for confirmed signal.

6. **Segment-level regression despite aggregate improvement**: Never gate purely on aggregate metrics — the validation gate and shadow comparison both run fairness/segment-slice checks (by region, by skill tier, by player tenure) as a required gate criterion, not just overall AUC. A candidate that improves aggregate but regresses a meaningful segment fails the gate and requires explicit DS sign-off to override with documented justification (e.g., intentionally deprioritizing a shrinking segment) — this override itself is logged in the audit trail.

7. **Rollback after artifact eviction**: The registry always retains the artifact in cold S3 storage even after eviction from the hot serving-node cache (eviction only removes it from the Triton model repository's warm set, not from the source of truth). Rollback in this case takes an extra 1-2 minutes to re-download and load the artifact rather than an instant pointer flip — an accepted latency tradeoff since keeping every historical version warm indefinitely doesn't scale; the grace period (24h) is tuned so this slow-path rollback is rare, triggering only for old versions being rolled back to well after they've been stable-superseded.

8. **Onboarding a new studio with a different stack**: The platform's core contracts (event schemas on Kafka, the Model Registry API, the validation-gate interface) are stack-agnostic — a new title's telemetry just needs a schema-conformant producer and a feature-pipeline adapter; training jobs are just K8s Jobs, so any training code that can run in a container (regardless of language/framework) plugs into the same orchestrator DAG. The main onboarding cost is building the title-specific feature transformations and gate thresholds, not re-architecting the platform.

9. **Incremental warm-start vs. full retrain**: Warm-start incremental retraining is faster and cheaper (fine-tunes from existing weights, fewer epochs) and is preferred for frequent/fast-cadence models (fraud) where speed matters more than eliminating accumulated bias. Full retrain-from-scratch is preferred periodically (weekly/monthly backstop) regardless of incremental cadence, because incremental updates can accumulate subtle bias or drift in ways a fresh fit on the full current data distribution corrects — treat incremental as the fast-response tool and full retrain as the periodic correctness reset.

10. **Multi-region promotion during regional incident**: Promotions are only ever authored in the active control-plane region (US-EAST primary) — EU never independently promotes, it only consumes replicated promotion events. If EU is mid-incident, a US promotion still fires but EU's replica may lag until the incident resolves, meaning EU keeps serving its last-synced model version (safe, if stale) rather than risking a partial/inconsistent promotion applied only in one region. If US itself is down, promotions simply queue/fail until control-plane recovery or a manual decision to fail over primary registry role to EU — deliberately no automatic multi-master promotion to avoid split-brain on "which model is actually in prod."
