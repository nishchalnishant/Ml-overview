# ML Platform

## 1. Problem Framing & Requirement Gathering

EA runs 40+ live-service titles (FIFA/FC, Apex Legends, Battlefield, The Sims, mobile studios) each with independent data science teams building churn models, matchmaking/skill models, LTV/monetization models, anti-cheat classifiers, recommendation systems, and content-moderation models. Today each studio hand-rolls its own training scripts, GPU allocation, and deployment glue — duplicated effort, inconsistent governance, no shared feature reuse, no central cost visibility.

Build an **internal ML Platform**: a paved-road system providing pipelines, orchestration, compute scheduling, and self-service model deployment so any studio team can go from notebook to production-serving endpoint without owning infrastructure.

This is a **platform-for-platforms** problem: the customers are internal ML engineers/data scientists (~800 across EA), not end players directly. Success = time-to-production for a new model drops from weeks to days, GPU utilization goes up, and governance (access control, lineage, cost attribution) is centralized.

## 2. Functional Requirements

- Self-service pipeline authoring: define DAGs (data prep → feature build → train → eval → register → deploy) as code.
- Managed compute scheduling: submit training jobs (CPU/GPU/multi-GPU/multi-node) without manual cluster ops.
- Model registry: versioned models with lineage (data version, code commit, hyperparameters, metrics).
- Self-service deployment: one-click/one-API promote a registered model to a serving endpoint (batch or online).
- Feature store: shared, reusable features across teams (online low-latency + offline for training).
- Experiment tracking: metrics, artifacts, comparisons across runs.
- Multi-tenant isolation: studio A cannot see/consume studio B's data or exhaust its compute quota.
- Automated retraining triggers (schedule- or drift-based).
- Cost attribution per team/project/model.
- Rollback and canary support for online-serving models.

## 3. Non-Functional Requirements (latency, availability, throughput, consistency, cost)

| Dimension | Target |
|---|---|
| Control-plane API availability | 99.9% (pipeline submission, registry, UI) |
| Online inference serving availability | 99.95% per deployed endpoint (studio-facing SLA) |
| Online inference p99 latency | < 50ms at the model-server hop (excl. network) |
| Batch scoring throughput | 500M rows/day sustained across platform |
| Training job scheduling latency | < 30s queue-to-running for priority jobs |
| Feature store online read p99 | < 10ms |
| Consistency | Eventual consistency acceptable for registry/metadata; strong consistency required for feature-store point-in-time joins used in training |
| Cost | GPU utilization > 65% fleet-wide (vs. ~20-30% typical of siloed team-owned clusters) |
| Multi-tenancy | Hard quota isolation; no noisy-neighbor GPU starvation |

## 4. Clarifying Questions an interviewer would expect you to ask

1. Is this platform for training only, serving only, or both end-to-end?
2. How many internal teams/tenants at launch, and expected growth in 2 years?
3. Do we own the GPU fleet (on-prem/colo) or is this cloud-elastic (EA uses hybrid: on-prem for steady-state + cloud burst)?
4. What's the blast radius requirement — must one tenant's bad job never affect another's SLA?
5. Do models serve real-time player-facing traffic (matchmaking, anti-cheat) or offline batch (churn scoring)? Both?
6. Is there an existing data lake / feature warehouse we integrate with, or build fresh?
7. What governance/compliance constraints exist (COPPA for younger player base titles, regional data residency for EU/China)?
8. Do we need to support arbitrary frameworks (PyTorch, TF, XGBoost, JAX) or standardize on one?
9. What's the expected long tail — mostly small XGBoost churn models, or also large-scale deep learning (recommender embeddings, LLM-based chat moderation)?
10. Who owns on-call for the platform itself vs. for individual models deployed on it?

## 5. Assumptions (explicit, numbered)

1. 800 internal ML practitioners across ~40 studios/teams; ~250 actively training models weekly.
2. ~3,000 registered models platform-wide; ~600 actively serving in production at any time.
3. Hybrid infra: 2,000 on-prem GPUs (A100/H100 mix) + cloud burst capacity (AWS/GCP) capped at 20% of steady-state fleet cost.
4. Average training job: 4-GPU, 6-hour job; large jobs (recommender embeddings) up to 64-GPU multi-node, 24h.
5. Online-serving models: median QPS 200, P99 title (e.g., matchmaking skill model) up to 50K QPS during peak (FIFA World Cup-style live event).
6. Batch scoring: nightly churn/LTV scoring over up to 150M player profiles across all titles combined.
7. Feature store holds ~5,000 shared feature definitions, ~50TB offline (Parquet/Iceberg), ~2TB online (Redis/DynamoDB-class).
8. Frameworks supported: PyTorch, TensorFlow, XGBoost/LightGBM, scikit-learn — containerized, not source-integrated.
9. Multi-region: primary in US-East (on-prem Virginia DC) + US-West and EU (Dublin) for data residency and latency.
10. Retention: raw telemetry 13 months, model artifacts indefinite (versioned), training datasets snapshotted per run (90-day default, extendable).

## 6. Capacity Estimation (QPS, storage, model size, GPU/CPU counts, back-of-envelope math shown)

**Online serving QPS (aggregate across platform):**
- 600 production endpoints, median 200 QPS, some spike to 50K QPS during live events.
- Aggregate steady state: 600 × 200 = 120,000 QPS baseline; peak burst (live events, 5 titles concurrently spiking) ≈ 120,000 + 5×50,000 = 370,000 QPS platform-wide.
- Provision serving fleet for 500,000 QPS peak with headroom.

**GPU fleet sizing (training):**
- 250 active training teams, avg 2 jobs/week/team, avg job 4 GPUs × 6h = 24 GPU-hours.
- Weekly GPU-hours demand: 250 × 2 × 24 = 12,000 GPU-hours/week ≈ 1,714 GPU-hours/day.
- At 24h/day/GPU capacity, that's ~71 GPUs needed for steady load *if perfectly packed*; real-world utilization target 65% → provision ~110 GPUs for routine jobs.
- Add large multi-node jobs (recommender/embedding retraining, ~10 jobs/week × 64 GPU × 24h = 15,360 GPU-hours/week ≈ 2,194/day → ~91 more GPUs at 100% pack, ~140 at 65% target).
- Total steady-state GPU pool: ~250 GPUs dedicated + elastic burst pool of 500 (on-prem 2,000-GPU shared fleet also serves non-ML-platform workloads like rendering farms, so ML platform gets a carved quota, bursting via cloud when queue depth > threshold).

**Model serving compute:**
- Small tabular models (XGBoost, churn/LTV): CPU-only, ~1 vCPU/core handles ~500 QPS at <10ms → 500K QPS / 500 = ~1,000 vCPUs for tabular fleet with 2x redundancy = 2,000 vCPUs.
- Deep learning models (recommenders, embeddings, moderation): GPU inference, T4/L4-class, ~2,000 QPS/GPU at batch-4 → for 100K QPS of DL traffic: 50 GPUs, ×2 for HA/regions = 100 GPUs.

**Storage:**
- Offline feature store: 50TB Parquet/Iceberg on S3-class object storage, growing ~5TB/month.
- Online feature store: 2TB in-memory/SSD-backed KV (Redis Cluster / DynamoDB), sized for sub-10ms reads.
- Model registry artifacts: 3,000 models × avg 500MB (checkpoints, larger for DL) ≈ 1.5PB cumulative (dedup via layer-sharing helps; realistic ~600TB after content-addressed storage).
- Training datasets: snapshotted, 90-day retention, ~200TB rolling.
- Telemetry raw event log feeding feature pipelines: EA-wide ~2M events/sec at peak (all titles) landing in Kafka → downstream this platform consumes a filtered subset (~200K events/sec) for feature computation.

**Back-of-envelope cost sanity check:**
- 250 steady-state GPUs (H100-class, ~$3/hr amortized on-prem) × 24h × 30d ≈ $540K/month compute-equivalent cost internally attributed.
- Cloud burst (500 GPU-hours pool, ~15% utilization actually bursts to cloud at ~$5/hr on-demand) ≈ 500×0.15×24×30×$5 ≈ $270K/month — motivates aggressive spot/preemptible use (Section 29).

## 7. High-Level Architecture

```
                                   ┌────────────────────────────────────────┐
                                   │        Studio ML Engineers (800)        │
                                   │   SDK / CLI / Web UI / Notebook plugin  │
                                   └───────────────────┬──────────────────────┘
                                                        │ REST/gRPC (authN via SSO+mTLS)
                                                        ▼
                          ┌────────────────────────────────────────────────────┐
                          │                Control Plane (API GW)               │
                          │   Pipeline API │ Registry API │ Deploy API │ Quota  │
                          └───────┬───────────────┬───────────────┬────────────┘
                                  │               │               │
                 ┌────────────────┘       ┌───────┘               └──────────────┐
                 ▼                        ▼                                       ▼
     ┌──────────────────────┐  ┌────────────────────────┐        ┌───────────────────────────┐
     │  Orchestration Engine │  │   Model Registry &       │        │  Deployment Controller     │
     │  (Argo Workflows/     │  │   Experiment Tracking     │        │  (K8s operator: canary,    │
     │   Airflow DAGs)       │  │   (metadata store)        │        │   blue/green, rollback)    │
     └──────────┬────────────┘  └────────────┬───────────────┘        └─────────────┬─────────────┘
                │                            │                                       │
                ▼                            ▼                                       ▼
     ┌──────────────────────┐   ┌─────────────────────────┐          ┌───────────────────────────┐
     │  Compute Scheduler     │   │   Artifact Store (S3/     │          │   Model Serving Fleet      │
     │  (Kubernetes + Volcano/│   │   content-addressed,       │          │  (Triton/TorchServe/KServe │
     │   Slurm-on-K8s for     │   │   versioned checkpoints)   │          │   + CPU tabular fleet)     │
     │   gang-scheduled GPU)  │   └─────────────────────────┘          └─────────────┬─────────────┘
     └──────────┬────────────┘                                                       │
                │                                                                     ▼
                ▼                                                        ┌───────────────────────────┐
     ┌──────────────────────┐        ┌────────────────────────┐         │  Studio game services /    │
     │  GPU/CPU Fleet         │◄──────┤  Feature Store            │         │  matchmaking / live game   │
     │  (on-prem + cloud burst│       │  Online (Redis/DynamoDB) │◄────────┤  calling inference API     │
     │   spot/preemptible)    │       │  Offline (Iceberg/Parquet)│         └───────────────────────────┘
     └──────────┬────────────┘        └───────────┬────────────┘
                │                                  ▲
                ▼                                  │
     ┌──────────────────────┐        ┌────────────────────────┐
     │  Training Data Lake    │───────►  Feature Pipelines       │
     │  (raw telemetry via    │        │  (Spark/Flink jobs,      │
     │   Kafka → Iceberg)     │        │  batch + streaming)      │
     └──────────────────────┘        └────────────────────────┘
                ▲
                │
     ┌──────────────────────┐
     │  Player Telemetry      │
     │  Kafka (EA-wide,       │
     │  ~2M events/sec)       │
     └──────────────────────┘

  Cross-cutting: Observability (Prometheus/Grafana/Jaeger), Drift Detection Service,
  Cost/Quota Service, IAM (SSO+RBAC), Alerting (PagerDuty)
```

## 8. Low-Level Components

**API Gateway / Control Plane**
- Responsibility: authn/authz, request routing, quota enforcement, rate limiting.
- Interface: REST + gRPC; OpenAPI spec published per tenant.
- Scaling unit: stateless pods behind L7 LB, scale on request rate.

**Orchestration Engine (Argo Workflows on K8s)**
- Responsibility: DAG execution for pipelines (data prep → train → eval → register).
- Interface: YAML/Python SDK DAG submission; webhook callbacks on stage completion.
- Scaling unit: workflow controller shards by namespace (tenant); horizontally scale controller replicas.

**Compute Scheduler (Volcano/Kueue on Kubernetes)**
- Responsibility: gang-scheduling for multi-GPU distributed jobs, fair-share queueing across tenants, preemption policy.
- Interface: PodGroup CRDs, priority classes, ResourceQuota per namespace.
- Scaling unit: cluster autoscaler adds/removes GPU nodes based on queue depth.

**Model Registry & Experiment Tracking**
- Responsibility: versioned model metadata, lineage graph (data snapshot hash, code commit, hyperparams, metrics), experiment comparison.
- Interface: REST API (`register_model`, `get_lineage`, `promote_stage`); backed by relational metadata store + object storage for artifacts.
- Scaling unit: read-heavy — scale via read replicas; writes are low-QPS (per training run, not per-inference).

**Deployment Controller (custom K8s operator)**
- Responsibility: reconciles desired model-serving state (canary %, replica count, resource class) with actual K8s Deployments.
- Interface: `ModelDeployment` CRD (declarative: model_uri, traffic_split, min/max replicas).
- Scaling unit: operator itself is low-load (control loop); scales models it manages independently.

**Feature Store**
- Responsibility: online low-latency feature serving + offline point-in-time-correct training data generation.
- Interface: `get_online_features(entity_ids, feature_refs)`, `get_historical_features(entity_df, feature_refs, timestamp)`.
- Scaling unit: online store scales as KV cluster (shard by entity_id hash); offline scales as data lake compute (Spark).

**Model Serving Fleet**
- Responsibility: host models behind inference endpoints; batching, multi-model hosting, autoscaling.
- Interface: `POST /v1/models/{model}/predict`.
- Scaling unit: per-model Deployment + HPA; shared GPU pool for low-QPS models (multi-model serving on Triton).

**Drift Detection Service**
- Responsibility: continuously compare production feature/prediction distributions vs. training baseline.
- Interface: subscribes to inference logs (Kafka), emits drift-score metrics + alerts.
- Scaling unit: stream processing job (Flink), scales with inference log volume.

**Cost/Quota Service**
- Responsibility: track GPU-hour/storage consumption per tenant, enforce quotas, generate chargeback reports.
- Interface: internal metering API, integrates with scheduler admission control.
- Scaling unit: low-QPS aggregation service, batch rollups nightly.

## 9. API Design

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/pipelines` | POST | Submit a new pipeline DAG definition |
| `/v1/pipelines/{id}/runs` | POST | Trigger a run of a pipeline |
| `/v1/pipelines/{id}/runs/{run_id}` | GET | Poll run status/metrics |
| `/v1/models` | POST | Register a new model version |
| `/v1/models/{name}/versions/{v}` | GET | Fetch model metadata + lineage |
| `/v1/models/{name}/versions/{v}/promote` | POST | Promote model to staging/prod |
| `/v1/deployments` | POST | Create a serving deployment (canary config) |
| `/v1/deployments/{id}` | PATCH | Update traffic split / rollback |
| `/v1/deployments/{id}` | DELETE | Tear down endpoint |
| `/v1/features/online` | POST | Batch fetch online features by entity ids |
| `/v1/features/historical` | POST | Point-in-time offline feature join |
| `/v1/models/{name}/predict` | POST | Inference call (per-deployment alias) |
| `/v1/quota/{tenant}` | GET | Current GPU/storage quota usage |

Example: register model version
```json
POST /v1/models
{
  "name": "fifa-churn-predictor",
  "version": "2026.07.08-1",
  "framework": "xgboost",
  "artifact_uri": "s3://ml-platform-registry/fifa-churn/2026.07.08-1/model.tar.gz",
  "training_run_id": "run-8f2a1c",
  "data_snapshot_hash": "sha256:9c1e...",
  "metrics": {"auc": 0.881, "logloss": 0.312},
  "tenant": "fifa-liveops"
}
```
Response:
```json
{
  "model_id": "mdl_7f3a9",
  "status": "REGISTERED",
  "registered_at": "2026-07-08T14:02:11Z"
}
```

Versioning: URI-path major version (`/v1/`); model versions are immutable semantic-ish tags (`{date}-{seq}`); breaking API changes ship as `/v2/` with 6-month deprecation window on `/v1/`.

## 10. Database Design

| Store | Type | Used for | Partition/Shard Key |
|---|---|---|---|
| Registry metadata DB | PostgreSQL (RDS-class, multi-AZ) | Model/version/lineage/experiment metadata | Shard by `tenant_id` (logical schema-per-tenant at scale) |
| Artifact store | S3-compatible object store, content-addressed | Model checkpoints, datasets snapshots | Key = content hash prefix (even distribution) |
| Offline feature store | Apache Iceberg on S3, queried via Spark/Trino | Historical point-in-time features, training sets | Partitioned by `event_date` + `entity_type` |
| Online feature store | DynamoDB / Redis Cluster | Low-latency feature reads at inference | Partition key = `entity_id` (hash-based) |
| Pipeline/run metadata | PostgreSQL | DAG definitions, run status, task states | Shard by `tenant_id` |
| Metrics/time-series | Prometheus + long-term store (Thanos/Mimir) | Infra + model quality metrics | Sharded by metric label cardinality (per-tenant remote-write) |
| Audit log | Append-only columnar (ClickHouse) | Access logs, deployment history, compliance | Partitioned by `date` |

Why: Postgres for registry — needs relational integrity (foreign keys: model → run → dataset → deployment) and moderate write volume. Iceberg/columnar for offline features — needs cheap massive scans + time-travel for point-in-time correctness. KV (DynamoDB/Redis) for online — needs single-digit-ms point lookups, not range scans. ClickHouse for audit/logs — high-ingest append-only analytical queries.

## 11. Caching

| What's cached | Layer | Strategy |
|---|---|---|
| Online feature values | Redis in front of DynamoDB | Cache-aside; TTL matches feature freshness SLA (seconds-to-minutes) |
| Model artifacts (hot models) | Local NVMe on serving nodes / CDN-like artifact cache | Write-through on deploy; invalidate on new version promote |
| Registry metadata reads (model lookup by name) | In-process LRU + Redis | Cache-aside, TTL 60s, invalidate on promote/update event |
| Inference results for deterministic batch requests (e.g., repeated churn scoring same day) | Redis, keyed on `(model_version, input_hash)` | Cache-aside, TTL until next scoring cycle |
| Feature schema/config | Local in-memory per pod, refreshed via watch on config CRD | Push-based invalidation (K8s informer pattern) |

Invalidation: registry and deployment changes emit events (Kafka topic `ml-platform.registry.events`) that caching layers subscribe to for active invalidation, rather than relying solely on TTL — critical for model version cache (serving stale model pointer post-rollback is a correctness bug, not just a staleness annoyance).

## 12. Queues & Async Processing

| Queue | Purpose | Delivery guarantee | Dead-letter handling |
|---|---|---|---|
| `pipeline.run.requests` | Async trigger pipeline runs | At-least-once (Kafka) | After 3 retries → DLQ topic, alert pipeline owner |
| `training.job.submissions` | Hand off to compute scheduler | At-least-once | Failed admission (quota exceeded) → DLQ + Slack notify tenant |
| `model.deploy.requests` | Async deployment reconciliation | At-least-once, idempotent by `deployment_id` | 5 retries w/ backoff, then DLQ + page on-call if prod-tier |
| `feature.pipeline.triggers` | Kick off feature recompute (batch/streaming) | At-least-once | DLQ + auto-retry next scheduled window |
| `inference.log.stream` | Async logging of predictions for drift/monitoring | At-least-once (best-effort, can drop under extreme backpressure — not correctness-critical) | Sampled DLQ for debugging only |

Exactly-once is NOT attempted end-to-end (expensive); instead all consumers are designed idempotent (dedup by `run_id`/`deployment_id`/content hash) so at-least-once + idempotent processing yields effectively-exactly-once outcomes. Dead-letter queues route to a `ml-platform-dlq-inspector` service that classifies failure (transient vs. permanent) and either auto-replays or flags for human triage.

## 13. Streaming & Event-Driven Architecture

| Topic | Producer | Consumer(s) | Schema |
|---|---|---|---|
| `player.telemetry.raw` | Game clients/servers (EA-wide) | Feature pipelines, data lake ingestion | Avro, `{player_id, session_id, event_type, event_ts, payload}` |
| `ml-platform.registry.events` | Registry service | Cache invalidators, audit logger | `{event_type: REGISTERED|PROMOTED|ROLLED_BACK, model_id, version, tenant, ts}` |
| `ml-platform.deployment.events` | Deployment controller | Monitoring, cost service, drift service | `{deployment_id, model_id, version, traffic_split, replicas, ts}` |
| `inference.predictions.log` | Model serving fleet | Drift detection, offline eval, feature-log join for future training | `{model_id, version, request_id, features_used, prediction, latency_ms, ts}` |
| `training.job.status` | Compute scheduler | Orchestration engine, notification service | `{job_id, tenant, status: QUEUED|RUNNING|SUCCEEDED|FAILED, gpu_hours_consumed, ts}` |

Consumer groups: each downstream (drift, cost, audit) runs its own consumer group off `inference.predictions.log` so a slow consumer (e.g., drift service backlog) never blocks others. Partitioning key = `model_id` to preserve per-model ordering for drift windowing.

## 14. Model Serving

- Framework: **Triton Inference Server** for GPU/DL models (supports TensorRT, ONNX, PyTorch backends, dynamic batching, multi-model on shared GPU via concurrent model execution). **Seldon-core/KServe on top of Triton + a lightweight CPU-only server (e.g., MLServer)** for tabular XGBoost/LightGBM models.
- Batching: dynamic batching window 5-10ms for DL models to trade small latency for throughput (bundling matchmaking-skill-score requests during peak).
- Multi-model: low-QPS studio models (long tail — hundreds of small models) share GPU pools via Triton's concurrent model instances rather than 1 GPU/model (which would waste ~90% of the fleet given median 200 QPS).
- Hardware: L4/T4-class GPUs for inference (cost-efficient vs. A100/H100 reserved for training); CPU fleet (standard compute-optimized instances) for tabular models.
- Canary + shadow serving supported natively via the deployment controller's traffic-split (Section 33).

## 15. Feature Store

- Online/offline split: offline (Iceberg/Parquet, queried via Spark/Trino) for training-set generation; online (Redis/DynamoDB) for inference-time low-latency lookups. Same feature *definitions* (declared once in feature repo, e.g., Feast-style) compile to both materialization paths — avoids train/serve skew from divergent logic.
- Point-in-time correctness: offline joins use `as_of_timestamp` per training example so features reflect only data available *before* the label event — prevents leakage (e.g., a churn label at day 30 must not join against features computed using day 31+ telemetry). Implemented via point-in-time join in Spark against Iceberg's time-travel snapshots.
- Feature freshness SLAs vary: real-time features (last-5-min play session stats) refreshed via streaming (Flink) into online store within seconds; slow features (30-day rolling spend) refreshed nightly batch.
- Feature versioning: each feature definition versioned; models pin a feature-view version at training time to guarantee reproducibility even if the feature logic later changes.

## 16. Vector Database

Applicable, but narrow scope: used for a subset of platform tenants building **content recommendation** (in-game store item recommendations, Battlefield's cross-title friend/content suggestions) and **similarity-based anti-cheat/fraud detection** (embedding player behavior sequences to find near-duplicate cheat signatures).

- Store: managed vector DB (e.g., pgvector for smaller tenants co-located with existing Postgres; standalone Milvus/OpenSearch-kNN for large-scale, >100M vector, tenants) — platform offers both tiers, tenant chooses based on scale.
- Indexing: HNSW for the large-scale tier (good recall/latency tradeoff, supports incremental inserts — important since player embeddings update continuously) over IVF-PQ (better for static, memory-constrained, batch-built indexes) — chosen because live-service data is continuously appended, not a one-time batch load.
- ANN parameters: `M=16, ef_construction=200` typical starting point for HNSW; tuned per tenant recall target (95%+ recall@10 for recommendations, allowed to relax to ~90% for anti-cheat clustering where speed matters more).
- Not a platform-wide requirement — most tenants (churn/LTV/matchmaking) never touch it; it's an optional platform capability, not core infra.

## 17. Embedding Pipelines

Applicable for the recommendation/anti-cheat tenants above.

- Player/item embeddings generated via two-tower or transformer-based encoders, trained via the standard training pipeline (Section 19), output embeddings written to the vector DB via a dedicated `embedding-sync` pipeline stage.
- Batch re-embedding: nightly full re-embed of active-player population (rolling 30-day active players, ~40M across relevant titles) for large periodic model updates.
- Incremental embedding: streaming job (Flink) computes updated embeddings for players with new session activity within the last hour, upserts into vector DB — keeps recommendations fresh without full nightly-only cadence.
- Embedding drift monitored same as model drift (Section 21) — embedding-space centroid shift signals upstream behavior change or a broken feature pipeline.

## 18. Inference Pipelines

Request lifecycle for an online prediction (e.g., matchmaking skill-score call):

```
Game Server                 API Gateway         Feature Store        Model Server        Logging/Drift
    │                           │                     │                    │                   │
    │ POST /predict             │                     │                    │                   │
    │──────────────────────────►│                     │                    │                   │
    │                           │ authN/authZ, rate-  │                    │                   │
    │                           │ limit check          │                    │                   │
    │                           │                     │                    │                   │
    │                           │ get_online_features  │                    │                   │
    │                           │────────────────────►│                    │                   │
    │                           │◄────────────────────│ (p99 <10ms)        │                   │
    │                           │  features returned   │                    │                   │
    │                           │                     │                    │                   │
    │                           │ predict(features)    │                    │                   │
    │                           │─────────────────────────────────────────►│                   │
    │                           │                     │       dynamic batch, GPU/CPU infer       │
    │                           │◄─────────────────────────────────────────│ (p99 <30ms)        │
    │                           │  prediction + score  │                    │                   │
    │                           │                     │                    │  async log         │
    │                           │─────────────────────────────────────────────────────────────►│
    │◄──────────────────────────│                     │                    │                   │
    │  response (score, latency)│                     │                    │                   │
```

- Total budget target: p99 < 50ms at platform hop (gateway + feature fetch + inference), excluding game-server-to-gateway network which studios budget separately.
- Async logging to `inference.predictions.log` is fire-and-forget (does not block response) — feeds drift detection and future training data collection (joined later with observed outcome labels, e.g., did the matched game end in a blowout).
- Failure handling: feature-store timeout → serve with cached/default feature values (degrade gracefully) rather than fail the request; model-server timeout → circuit breaker falls back to previous stable model version or a rule-based default.

## 19. Training Pipelines

- Data prep: Spark jobs pull from offline feature store with point-in-time join against label events (e.g., churn = no login in 14 days), output versioned training dataset snapshot (Iceberg snapshot ID pinned in lineage).
- Training orchestration: Argo Workflow DAG — `data_prep → train → evaluate → register`. Each stage a container; parameters injected via workflow inputs (hyperparameters, data snapshot ref).
- Distributed training: for large DL models (recommender two-tower, embedding models), use PyTorch DDP / Horovod across multi-node GPU pool, gang-scheduled via Volcano (all-or-nothing scheduling — a 64-GPU job doesn't start with only 40 GPUs available, avoiding wasted partial allocation).
- Hyperparameter search: Ray Tune or Optuna integrated as a pipeline stage, launches parallel trial pods, reports back to experiment tracking.
- Reproducibility: every registered model links immutably to `(code_commit_sha, data_snapshot_id, hyperparameter_config, container_image_digest)` — anyone can re-run byte-for-byte.

## 20. Retraining Strategy (cadence, triggers)

| Trigger type | Example | Action |
|---|---|---|
| Scheduled | Weekly retrain for churn models (data staleness tolerance) | Cron-triggered pipeline run via Argo CronWorkflow |
| Data drift threshold | Feature distribution PSI > 0.2 vs. training baseline | Auto-trigger retrain pipeline, notify model owner |
| Concept drift / performance decay | Live AUC (via delayed-label backtesting) drops > 3% from registered baseline | Auto-trigger retrain + page model owner if drop > 8% (likely broken upstream, not just decay) |
| Manual | New feature added, bug fix in labeling logic | Data scientist triggers via API/UI |
| Live-event driven | Major content patch/season launch changes player behavior distribution (e.g., new FIFA season) | Pre-scheduled retrain job timed to patch release |

Cadence guidance by model class: fast-moving (matchmaking skill, fraud) — daily/near-real-time incremental retrain; medium (churn/LTV) — weekly; slow (long-horizon monetization models) — monthly.

## 21. Drift Detection (data drift, concept drift, what metrics, what thresholds)

- **Data/feature drift**: Population Stability Index (PSI) and KL-divergence computed per feature, comparing rolling production feature distribution (7-day window) vs. training-time baseline. Threshold: PSI > 0.1 = warn, > 0.2 = trigger retrain workflow.
- **Concept drift**: requires delayed ground-truth labels (e.g., churn realized 14 days later); track rolling AUC/logloss on a held-out "recent labeled" sample. Threshold: relative AUC drop > 3% = warn + auto-retrain trigger; > 8% = page on-call (likely pipeline break, not gradual decay).
- **Prediction drift** (proxy when labels are delayed/unavailable, e.g., real-time matchmaking): monitor output score distribution shift (KS-test) vs. training-time output distribution — early warning before labeled concept drift is even measurable.
- **Embedding drift** (Section 17 tenants): centroid/covariance shift in embedding space, monitored via Mahalanobis distance on daily embedding batches.
- Implementation: Flink streaming job consumes `inference.predictions.log`, computes rolling statistics windowed per model, emits metrics to Prometheus with per-model labels; drift score surfaced on model's dashboard + registry page.

## 22. Monitoring

| Category | What's monitored |
|---|---|
| Infra | GPU/CPU utilization, queue depth (compute scheduler), node health, network I/O, storage IOPS |
| Serving | Request rate, latency (p50/p95/p99), error rate, batch size distribution, GPU memory saturation |
| Model quality | Drift scores (Section 21), rolling AUC/logloss/RMSE per model, prediction distribution histograms |
| Pipeline health | DAG success/failure rate, stage duration, data-prep row counts (schema/volume anomaly detection) |
| Business | Downstream impact — e.g., churn-model-driven retention campaign conversion, matchmaking match-quality score, cost-per-prediction |
| Platform ops | Quota utilization per tenant, API gateway error rate, registry availability, deployment rollout success rate |

Dashboards: per-tenant Grafana dashboard auto-generated from deployment metadata; platform-wide "fleet health" dashboard for platform SRE team.

## 23. Alerting

| Alert | Condition | Routing |
|---|---|---|
| Serving endpoint down | Availability < 99.95% over 5 min window | Page model-owning team on-call (PagerDuty) |
| Latency SLA breach | p99 > 2x target for 5 consecutive min | Page model owner; platform on-call if fleet-wide |
| Concept drift severe | AUC drop > 8% | Page model owner |
| GPU queue starvation | Job wait time p95 > 30 min for priority-tier jobs | Page platform infra on-call |
| Quota near-exhaustion | Tenant at 90% GPU-hour quota | Slack notify tenant lead (non-paging) |
| Registry/control-plane down | Control plane 5xx rate > 5% | Page platform on-call, highest severity (blocks all deployments) |
| DLQ backlog growing | DLQ depth > 1000 or growing for 15 min | Page platform on-call |
| Cost anomaly | Daily spend > 150% of 7-day rolling avg for a tenant | Slack notify + weekly cost review flag |

On-call routing: two-tier — platform infra on-call owns control plane/scheduler/registry; individual model owners own their model's quality/business-metric alerts (platform provides the alerting infra, not the response).

## 24. Logging

- Structured JSON logs everywhere (`trace_id`, `tenant_id`, `model_id`, `request_id`, `ts`, `level`, `message` fields standardized platform-wide).
- PII handling: raw player telemetry may contain player_id (pseudonymous but joinable), device info, IP. Logs at the platform layer log `player_id_hash` (salted, tenant-scoped hash) not raw IDs; raw PII stays in the governed data lake with stricter access controls, never in general-purpose application/debug logs.
- Feature values logged for drift/debugging are logged in aggregate/statistical form by default; raw per-request feature vectors retained only in a restricted, access-audited log store with 30-day TTL (vs. 13-month telemetry retention) to minimize PII surface area.
- Retention: application/debug logs 30 days (hot, searchable via ELK/Loki); audit logs (deployment changes, access, promotions) retained 2 years in ClickHouse for compliance; inference prediction logs retained 90 days for retraining/backtesting, then aggregated-and-purged.
- Regional data residency: EU player data logs stay in EU region storage (Dublin), never replicated to US for logs/analytics, per GDPR.

## 25. Security (authn/authz, data encryption, threat model specific to this system)

- AuthN: enterprise SSO (SAML/OIDC via Okta) for human users (data scientists/UI); mTLS + short-lived service tokens (SPIFFE/SPIRE identities) for service-to-service.
- AuthZ: RBAC scoped per tenant/namespace — a studio's ML engineer has `deploy`/`train` rights only within their tenant namespace; cross-tenant read requires explicit data-sharing grant (e.g., a shared "cross-title fraud" feature set).
- Encryption: at-rest (KMS-managed keys per tenant for artifact store, feature store); in-transit (TLS 1.3 everywhere, mTLS internal).
- Threat model specifics:
  - **Model/data exfiltration**: a malicious/compromised tenant credential attempting to pull another studio's model artifacts or training data → mitigated by per-tenant IAM scoping + artifact store bucket policies + audit logging with anomaly alerts on unusual access patterns.
  - **Training data poisoning**: an insider or compromised upstream telemetry pipeline injecting adversarial data to bias a model (e.g., manipulate anti-cheat model to whitelist cheating behavior) → mitigated by data validation/schema checks pre-training, lineage tracking to trace bad data back to source, anomaly detection on training data statistics.
  - **Model extraction/inversion via inference API abuse**: an attacker hammering a public-facing inference endpoint (e.g., matchmaking score API) to reconstruct model logic or extract training data → mitigated by rate limiting (Section 27), query budget per API key, output rounding/noise for sensitive models.
  - **Supply-chain**: malicious container image submitted as a training job → mitigated by image scanning, signed images, restricted base-image allowlist for the compute scheduler.
  - **GPU resource abuse**: a compromised or bug-ridden job (e.g., cryptomining disguised as training) consuming GPU quota → mitigated by resource quotas, anomaly detection on job resource-usage patterns vs. declared job type.

## 26. Authentication (service-to-service and end-user auth mechanism)

- End-user (data scientists, via UI/CLI/SDK): OIDC login through Okta → short-lived JWT (15 min) + refresh token; CLI/SDK use device-code flow, tokens cached locally encrypted.
- Service-to-service (control plane → compute scheduler → serving fleet → feature store): mTLS with SPIFFE/SPIRE-issued workload identities, auto-rotated certs (24h TTL), no long-lived static secrets.
- Game-server-to-inference-API (external-to-platform but internal-to-EA callers): API key + mTLS at the network edge (game servers are trusted EA infra, but still scoped per-title API keys tied to quota/rate-limit policy) — not end-player-facing directly, so no OAuth-for-consumers needed here.
- Cross-region service auth: regional identity federation so a US-issued service identity is recognized (with residency-aware policy checks) when calling EU-region feature store for cross-region model deployments.

## 27. Rate Limiting

- Algorithm: **token bucket** per API key/tenant at the gateway — smooths bursts (natural for live-service traffic spikes during in-game events) while enforcing sustained-rate caps.
- Per-tenant limits: default 5,000 QPS per tenant for inference API (studio can request quota increase via platform team, reviewed against fleet capacity); control-plane APIs (pipeline submission, registry) capped lower (e.g., 100 req/s) since they're not meant for high-frequency calling.
- Per-model granularity: additionally rate-limited per deployed model (protects shared multi-model GPU pools from one noisy model starving others sharing the same Triton instance).
- Burst allowance: token bucket burst capacity = 2x sustained rate for up to 10s, covering live-event traffic spikes (matchmaking during a major esports event) without hard-rejecting legitimate surges.
- Over-limit behavior: HTTP 429 with `Retry-After` header; critical tier-1 models (anti-cheat, matchmaking) get priority lanes so platform-level congestion doesn't throttle them ahead of lower-priority batch/experimental traffic.

## 28. Autoscaling

- Model-serving HPA: scale on custom metric = `inference_queue_depth` and `p99_latency`, not just CPU% (CPU% is a poor proxy for GPU-bound or I/O-bound inference workloads). Target: keep p99 < 50ms; scale out when queue depth > 20 requests per replica sustained 30s.
- VPA for training job containers: right-sizes CPU/memory requests based on historical usage per job type (avoids over-provisioning generic "large" pod sizes for jobs that only use half the requested memory).
- KEDA for event-driven scaling of feature pipeline consumers: scale Flink/Spark streaming consumers based on Kafka consumer-group lag (`inference.predictions.log` lag drives drift-service replica count).
- Cluster autoscaler for GPU node pools: scale node count based on Volcano queue depth (pending PodGroups), with separate node pools for on-demand (baseline) vs. spot/preemptible (burst) — spot pool scales aggressively for interruptible training jobs, on-demand pool scales conservatively for prod-serving nodes.
- Cooldown tuning: serving fleet scale-down cooldown longer (10 min) than scale-up (1 min) — avoids flapping during bursty live-service traffic patterns typical of in-game events.

## 29. Cost Optimization (concrete levers: spot instances, caching, model distillation, batching)

- **Spot/preemptible instances**: all interruptible training jobs (non-time-critical, checkpointed) scheduled on spot pool — targets 60%+ of routine training GPU-hours on spot at ~60-70% discount vs. on-demand; checkpoint every 10 min so preemption loses < 10 min of work.
- **Dynamic batching**: as in Section 14, batching inference requests improves GPU throughput 3-5x for DL models vs. per-request execution, directly cutting GPU-count needed for a given QPS target.
- **Multi-model serving**: sharing GPUs across the long tail of low-QPS models (Section 14) avoids provisioning 1 GPU per model — estimated to cut serving GPU count for long-tail models by ~80%.
- **Model distillation/quantization**: large DL models (recommender towers) distilled to smaller student models or quantized to INT8/FP16 for serving — cuts inference latency and GPU memory footprint, enabling smaller/cheaper instance types (T4/L4 instead of A100 for serving).
- **Feature caching**: reduces redundant online feature-store reads (Section 11), cutting DynamoDB/Redis read costs and load.
- **Tiered storage**: cold model artifacts/old dataset snapshots moved to infrequent-access/glacier-class storage after 90 days of no access.
- **Idle detection**: auto-terminate notebook/dev GPU sessions idle > 30 min (common waste pattern — data scientist leaves a Jupyter GPU session running overnight).
- **Right-sizing via VPA**: (Section 28) prevents over-requesting CPU/memory for training jobs, improving bin-packing density on shared nodes.
- **Reserved capacity for baseline**: on-prem GPU fleet sized to steady-state median load (not peak) with cloud burst absorbing spikes — avoids paying cloud on-demand rates for baseline, EA-owned capacity is cheaper amortized.

## 30. Operational Concerns (Deployment, Reliability, Infra)

At SDE2 scope, treat this as a checklist rather than a design exercise: **backups** (automated snapshots of the model registry, feature store, and any stateful service, with a tested restore path), **rollback** (every deploy must be revertible to the last-known-good version — the model registry and CI/CD pipeline should make this a one-command operation), **canary/blue-green rollout** (shift a small percentage of traffic first, watch error rate and key business/model metrics, then ramp), and **basic observability** (dashboards + alerts on latency, error rate, and the top 2-3 model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level infra-architecture concerns — worth knowing they exist, not worth rehearsing the manifests.

## 38. Why This Architecture

- Kubernetes-native compute scheduling (Volcano/Kueue) chosen over bespoke Slurm-only cluster because EA already runs K8s fleet-wide for other services — reuses existing ops expertise, IAM integration, and observability stack rather than standing up a parallel Slurm ops practice.
- Separating control plane (registry/orchestration) from data plane (serving/training compute) lets each scale and fail independently — a registry outage shouldn't take down already-running inference; a GPU node failure shouldn't corrupt registry state.
- Feast-style unified feature definitions (single source of truth compiling to both online/offline) directly targets the single biggest real-world ML bug class: train/serve skew.
- Multi-tenant-by-design (namespace-per-tenant, quota-enforced) matches EA's actual org structure (40+ semi-autonomous studios) — a shared monolithic model/pipeline would fight studio autonomy; full per-studio platforms would fight cost/governance goals. Namespaced multi-tenancy is the middle path.
- Canary-by-default with drift-aware gates specifically targets the game-industry failure pattern where a model can look infra-healthy (normal latency/error rate) while being quality-broken (e.g., a matchmaking model that's technically "up" but producing bad skill scores after a bad training run) — pure infra health checks would miss this class of failure.

## 39. Alternative Architectures

| Alternative | Description | Why rejected / when preferred |
|---|---|---|
| Fully managed cloud ML platform (SageMaker/Vertex AI) end-to-end | Use vendor platform instead of building internal one | Rejected as EA's primary platform: hybrid on-prem GPU fleet already exists (sunk capex), vendor lock-in risk at 40-studio scale, per-request/per-training-hour vendor pricing gets expensive at EA's volume, and game-specific feature-store/telemetry integration needs custom work anyway. Would be preferred for a smaller company without existing GPU capex or ML platform team headcount to build/operate this. |
| Per-studio independent ML stacks (status quo, no shared platform) | Each studio owns its full stack | Rejected as target state — duplicated engineering effort, no shared feature reuse, poor fleet utilization (20-30% vs. 65%+ target), inconsistent governance. Would be "preferred" only in a conglomerate with near-zero data/infra sharing needs across business units — not EA's case, where cross-title player behavior patterns and shared infra investment have real reuse value. |
| Serverless-only inference (e.g., Lambda/Cloud Functions per model) | No persistent serving fleet, invoke-per-request | Rejected for latency-critical live traffic (cold starts violate p99 < 50ms budget) and cost at EA's sustained QPS (120K+ baseline QPS makes serverless per-invocation pricing worse than provisioned fleet). Would be preferred for the platform's low-QPS, latency-tolerant long tail (e.g., ad-hoc batch scoring jobs) — in fact used as a secondary path for that segment. |
| Single global model-serving region (no multi-region active-active) | One region serves all traffic | Rejected due to GDPR data residency (EU player data) and latency for EU/APAC live traffic; a single-region outage would also be a full platform-wide serving outage. Would be acceptable only for a platform serving strictly one geography with no residency constraints. |

## 40. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Multi-tenant shared fleet vs. per-team dedicated clusters | Higher utilization, lower cost, centralized governance | Requires robust quota/isolation engineering; noisy-neighbor risk if isolation has gaps |
| Feast-style unified feature definitions | Eliminates train/serve skew | Migration cost for teams with existing bespoke feature pipelines; added abstraction layer |
| At-least-once + idempotent processing (not exactly-once) | Simpler, cheaper, more available | Requires discipline — every consumer must correctly implement idempotency; a bug here causes silent duplicate processing |
| Active-active multi-region serving | Low latency globally, resilient to regional failure | 2-3x infra cost vs. single region; cross-region data consistency complexity |
| Canary-by-default for model rollout | Catches bad models before full-blast-radius impact | Slower time-to-full-rollout (hours vs. instant); requires delayed-label models to have longer, less-certain bake periods |
| Spot instances for training | ~60-70% cost savings | Preemption risk requires checkpointing discipline; not viable for prod-serving nodes |
| Shared multi-model GPU serving for long tail | Massive utilization gain for low-QPS models | Noisy-neighbor risk within a shared GPU instance; harder per-model resource accounting |
| Centralized platform team owns infra, studio teams own models | Clear separation of concerns, scalable ops model | Platform team becomes a dependency/bottleneck for infra changes; requires strong internal API contracts |

## 41. Failure Modes

| Failure | Concrete scenario | Mitigation |
|---|---|---|
| Feature store online read timeout | Redis node hot-partitioning on a popular title's entity_id range during a live event spike | Cache-aside fallback to last-known-good cached value; circuit breaker + graceful degradation (serve with stale/default features rather than fail request) |
| Bad model silently promoted | Training pipeline bug produces a model that passes offline eval metrics but is broken on a specific feature edge case not covered by eval set | Canary gates with drift-aware checks (Section 33) catch it before 100% rollout; shadow-serving option for high-risk models compares silently against production before any real traffic |
| GPU fleet exhaustion during major live-event launch | All studios simultaneously want retraining capacity ahead of a big content drop (e.g., World Cup mode launch) | Priority-tier quota reservations bookable in advance; cloud burst absorbs overflow; fair-share scheduler prevents one tenant monopolizing |
| Registry database outage | Postgres primary fails | Multi-AZ automatic failover (RTO ~15 min per Section 30); already-deployed models keep serving unaffected since serving doesn't depend on registry at request time |
| Cross-region replication lag causes stale model in a region | A rollback in US-East hasn't propagated to EU yet, EU still serves rolled-back-bad version | Deployment events (Section 13) drive active invalidation, not just artifact replication lag; critical-severity rollbacks trigger synchronous cross-region push, not async wait |
| Cascading retraining storm | Drift detection false-positive triggers auto-retraining across many models simultaneously (e.g., a shared upstream feature pipeline bug looks like "drift" everywhere at once) | Rate-limit auto-triggered retrains per time window; require correlation check (is this drift isolated or platform-wide, suggesting upstream break) before mass-triggering |
| Poisoned training data | Bug in an upstream telemetry pipeline corrupts a feature for 3 days before detected | Data validation/schema+statistical checks as a mandatory pre-training pipeline stage; lineage allows identifying and excluding the corrupted data window on discovery |

## 42. Scaling Bottlenecks

- **At 10x scale (8,000 practitioners, 6,000 prod endpoints)**: registry Postgres write/read load becomes a bottleneck for high-frequency experiment logging (many parallel HPO trials each logging metrics) — mitigation: move high-frequency metric logging off the relational store to a time-series store, keep Postgres for durable metadata only.
- **At 10x scale**: shared multi-model GPU pools hit scheduling contention — the "pack many small models on one GPU" strategy degrades as the long tail itself grows 10x; needs smarter bin-packing (ILP-based placement) instead of simple round-robin.
- **At 100x scale (EA-wide-plus-acquisitions, ~5M QPS aggregate)**: single-region control-plane-as-active-passive (Section 31) becomes a bottleneck/risk — would need to move to active-active control plane with proper multi-master conflict resolution, a much harder consistency problem (registry writes across regions).
- **At 100x scale**: online feature store partition hot-spotting on the most popular titles' entity ranges becomes severe even with hashing — needs per-tenant dedicated shard pools for the largest titles rather than fully shared multi-tenant KV cluster.
- **At 100x scale**: Kafka topic `player.telemetry.raw` at 2M events/sec today would approach tens of millions/sec — requires re-partitioning strategy and likely tiered ingestion (per-title Kafka clusters federated) rather than one EA-wide cluster.
- **Compute scheduler queue depth**: gang-scheduling for large multi-node jobs (64+ GPU) already causes head-of-line blocking at current scale during contention; at 10x, priority inversion (small job stuck behind a huge job waiting for capacity) needs smarter preemption/backfill scheduling (e.g., Slurm-style backfill algorithm ported to the K8s scheduler).

## 43. Latency Bottlenecks

p99 = 50ms budget breakdown for a typical online inference request:

| Hop | p50 | p99 | Notes |
|---|---|---|---|
| API gateway (authN/authZ, rate-limit check) | 1ms | 3ms | In-memory token bucket check, cached JWT validation |
| Online feature fetch | 3ms | 10ms | Redis cache-aside; p99 tail from cache misses hitting DynamoDB |
| Model inference (CPU tabular) | 2ms | 8ms | XGBoost, small model |
| Model inference (GPU DL, batched) | 8ms | 25ms | Dynamic batching adds queueing delay in exchange for throughput; p99 tail from batch-window wait + larger model compute |
| Async logging enqueue | <1ms | 1ms | Fire-and-forget, doesn't block response |
| **Total (tabular path)** | **~6ms** | **~21ms** | Well within budget |
| **Total (DL path)** | **~12ms** | **~38ms** | Within 50ms budget but tighter — batching window is the main lever if it needs tightening |

Where time is actually spent at p99: the feature-store cache-miss tail and the DL dynamic-batching queue wait dominate — both are the first places to look when a specific model's p99 regresses. Network hop game-server→gateway is excluded (studio-owned budget) but typically adds another 10-30ms depending on region proximity — this is why multi-region serving (Section 31) matters more for overall player-experienced latency than any single platform-hop optimization.

## 44. Cost Bottlenecks

- **GPU compute (training + serving)** is the dominant line item — estimated ~$800K+/month combined (Section 6 estimates) vs. storage/networking which are an order of magnitude smaller.
- Within GPU spend, **training compute for large distributed jobs** (multi-node embedding/recommender retraining) is disproportionate: 10 large jobs/week consuming ~2,200 GPU-hours/day vs. thousands of small jobs consuming less in aggregate — a small number of large-model teams drive a large fraction of the bill (classic long-tail-vs-head cost distribution).
- **Idle/low-utilization GPU time** (jobs requesting more GPUs than they use, or dev/notebook sessions left running) is the largest *avoidable* cost bucket — utilization sitting at industry-typical 20-30% before platform intervention means roughly 2-3x more GPU spend than necessary for the same delivered work; closing this gap toward the 65% target is the single highest-leverage cost lever (bigger than spot-instance discounts alone).
- **Cross-region data egress/replication** (model artifacts, cross-region feature replication) is a smaller but non-trivial and easy-to-overlook cost, especially with 3-region active-active serving replicating large DL model artifacts (multi-GB checkpoints) on every promotion.
- **Object storage growth** (1.5PB+ cumulative model artifacts before dedup) grows unbounded without lifecycle policies — tiering to infrequent-access/glacier after 90 days (Section 29) is necessary to prevent this becoming a silent multi-year cost creep.

## 45. Interview Follow-Up Questions

1. How do you prevent one tenant's runaway training job from starving GPU capacity for a time-critical tier-1 model retrain (e.g., anti-cheat during an active cheating wave)?
2. Walk through exactly how you guarantee point-in-time correctness in the feature store — what specifically prevents label leakage?
3. Your drift detection system fires a false-positive storm across 50 models simultaneously — how do you distinguish "real platform-wide drift" from "drift detector bug," and what's your incident response?
4. How would you redesign the registry's consistency model if you had to move from active-passive to active-active multi-region?
5. What's your strategy if a single studio's model (e.g., matchmaking) needs 10ms p99 but another's (e.g., nightly batch churn scoring) is fine with seconds of latency — how does one platform serve both well without over-engineering the common case?
6. How do you handle a model that was trained on data later found to violate data-residency/compliance rules (e.g., EU player data accidentally included in a US-trained global model)?
7. Explain the tradeoff between canary bake-period length and time-to-value for a data scientist who needs to ship a model fix quickly — how would you let them override the default safely?
8. How do you attribute GPU cost fairly when multiple tenants share a GPU via multi-model serving — what's the metering mechanism?
9. What happens to in-flight inference requests during a blue/green cutover — how do you guarantee zero dropped requests?
10. How would this architecture change if EA acquired a studio that insists on keeping its own separate ML stack for a transition period — what's the integration path?

## 46. Ideal Answers

1. **Runaway job isolation**: Enforce hard per-tenant ResourceQuota at the namespace level so no tenant can exceed its GPU-hour ceiling. Also reserve a priority-tier capacity pool with preemption rights for tier-1 retrain jobs, so critical work always has capacity even under full fleet contention.

2. **Point-in-time correctness**: Every training example carries an `event_timestamp`, and the offline feature join uses time-travel/snapshot queries (an `AS OF` join) to fetch feature values as they existed at that timestamp, not the latest values. This structurally prevents leakage since the join engine has no access to future snapshots.

3. **Distinguishing real drift from detector bug**: If many unrelated models across tenants show drift simultaneously, the prior shifts toward a shared upstream dependency breaking rather than independent real shifts. Auto-pause mass-retraining via a circuit breaker and page platform on-call to check the shared dependency first.

4. **Active-active registry**: Move from single-writer Postgres to partitioning writes by tenant-to-home-region, avoiding cross-region conflicts for the common case, with conflict-free replication only needed for rare cross-region operations like global model promotion. Accept eventual consistency for non-home-region reads, with read-your-writes guaranteed at home.

5. **Heterogeneous latency needs**: Expose deployment profiles (e.g., "real-time" with dedicated capacity and strict p99 gates vs. "batch" with spot-eligible best-effort scheduling) selected declaratively at deployment time. Most models get sensible defaults; only tier-1 low-latency needs opt into the costlier profile.

6. **Compliance violation in trained model**: Immediately quarantine the model and trace lineage (Section 8/19) to find the tainted training run/data snapshot and every downstream model that consumed it. Retrain from a corrected snapshot and treat it as a P1 compliance incident — lineage tracking is what bounds the blast radius here.

7. **Canary bake-period override**: Offer a documented "expedited canary" path with a shorter (not zero) bake period for low-risk, non-tier-1 model classes, with an audit trail for the override. Tier-1 models never skip gates regardless of urgency.

8. **Fair GPU cost attribution under multi-model sharing**: Meter at the request level, tagging each inference request with `tenant_id`/`model_id` and attributing cost proportional to actual compute-time consumed, not wall-clock hosting time. This avoids unfairly splitting cost evenly when one model does far more work than another on a shared GPU.

9. **Zero-drop blue/green cutover**: Use connection draining — the load balancer stops sending new requests to blue at cutover but lets in-flight requests finish (grace period matched to p99 request duration). Green is already warm and health-check-passing before traffic shifts, so there's no cold-start gap.

10. **Acquired studio with separate stack**: Treat it like external-but-trusted tenant onboarding — its own namespace with standard isolation, but relax the platform-pipeline requirement via a thin adapter/shim so their existing tooling can push into the registry format. This gets them under central governance fast while giving a longer runway to fully migrate.
