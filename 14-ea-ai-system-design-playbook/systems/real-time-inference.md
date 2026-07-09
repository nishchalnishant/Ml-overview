# Real-Time Inference Platform

## 1. Problem Framing & Requirement Gathering

Design a **low-latency online model serving platform** for EA-scale live-service games — the shared infra layer that any game studio (FIFA/EA SPORTS FC, Apex Legends, The Sims, Battlefield) calls to get a model prediction inline with a live request: matchmaking skill/latency scoring, real-time toxicity/chat moderation, dynamic difficulty adjustment, churn-risk-triggered offers, anti-cheat scoring, live bidding for ad-funded titles.

Key framing decisions to state up front:
- This is a **platform**, not a single model — many teams onboard many models (multi-tenant).
- Serving is **synchronous, request/response**, in the hot path of gameplay or matchmaking — not batch scoring.
- Must support both **GPU-bound** models (deep ranking, embeddings, vision-based anti-cheat) and **CPU-bound** models (GBDT churn/fraud scores) behind one control plane.
- Optimize for **tail latency and availability** over raw accuracy iteration speed (that's the training platform's job, which this system consumes from).

## 2. Functional Requirements

- FR1: Accept a prediction request (feature vector or raw features + entity IDs) and return a model output synchronously.
- FR2: Support multiple models, multiple versions per model, multiple frameworks (PyTorch, TensorFlow, XGBoost/LightGBM, ONNX) on shared infra.
- FR3: Support dynamic **request batching** to improve GPU utilization without breaching latency SLOs.
- FR4: Fetch **online features** (player state, session context) from a feature store when the caller doesn't pass full features.
- FR5: Support **canary and shadow traffic** for new model versions.
- FR6: Support **A/B routing** between model versions/variants per experiment config.
- FR7: Return calibrated confidence/uncertainty alongside point predictions where the model supports it.
- FR8: Provide an **explainability hook** (feature attribution) for flagged high-stakes decisions (e.g., ban/fraud actions), queryable asynchronously.
- FR9: Support graceful **degradation** — return a cached/default/heuristic prediction if the model service is unavailable rather than failing the caller's request.
- FR10: Provide a self-service **model registration/deployment** API for ML teams (no platform-team-in-the-loop deploys).

## 3. Non-Functional Requirements (latency, availability, throughput, consistency, cost)

| Dimension | Target |
|---|---|
| Latency (p50) | ≤ 15 ms end-to-end (excluding network from client) |
| Latency (p99) | ≤ 60 ms end-to-end |
| Latency (p99.9) | ≤ 150 ms (circuit-break/fallback beyond this) |
| Availability | 99.95% per model endpoint (≈ 4.4 hrs/yr downtime budget) |
| Throughput | Sustain 250K QPS platform-wide at peak (global), burst to 400K during live-event spikes |
| Consistency | Feature reads: read-your-writes not required; bounded staleness ≤ 1s for online features acceptable |
| Cost | ≤ $0.15 per 10K inferences blended CPU+GPU average |
| Durability | Prediction logs durable for 30 days min (drift/audit), model artifacts durable indefinitely (versioned) |
| Multi-tenancy isolation | Noisy-neighbor model must not degrade other tenants' p99 by >10% |

## 4. Clarifying Questions an interviewer would expect you to ask

1. Is this single-game or shared platform across all EA studios? (Assume shared platform.)
2. What's the mix of GPU vs CPU models expected — deep learning heavy or mostly GBDT/linear? (Assume 30% GPU-bound, 70% CPU-bound by model count, but GPU models dominate compute cost.)
3. Do callers pass pre-computed features, or does the platform own feature retrieval? (Assume both supported, feature-store path is opt-in.)
4. Is synchronous the only mode, or do we also need async/batch scoring through the same registry? (Assume this chapter scopes to synchronous; batch is a separate pipeline reusing the model registry.)
5. What's acceptable staleness for a "stale model still serving" during a bad deploy? (Assume auto-rollback within 5 minutes of SLO breach.)
6. Are there regulatory/compliance constraints (COPPA for younger player bases, GDPR data residency in EU)? (Assume yes — must support EU data residency.)
7. Who owns feature computation correctness — this platform or upstream data teams? (Assume upstream teams own pipelines; platform owns online store and point-in-time serving correctness.)
8. What's the failure behavior contract with calling services — block, degrade, or fail-open? (Assume fail-open with default score + flag.)
9. Peak traffic pattern — is it diurnal only, or driven by game-launch/event spikes? (Assume both; live-service events can cause 3-5x spikes in minutes.)

## 5. Assumptions

1. 120M monthly active players across EA's live-service portfolio; ~15M concurrent at global peak (weekend evenings, overlapping US/EU prime time).
2. Average of 1.8 inference calls per player-action-relevant event (matchmaking, chat message, purchase-intent trigger).
3. 400 distinct models registered platform-wide; 60 receive >80% of traffic (power-law distribution).
4. Average feature vector: 150 features, ~2 KB serialized.
5. GPU models: average batch-eligible with 8 ms max batching window.
6. 70% of traffic is CPU-servable (GBDT/linear/small MLP), 30% requires GPU (deep ranking, sequence models, embedding-heavy).
7. Model artifact sizes: CPU models avg 50 MB, GPU deep models avg 2-8 GB.
8. Infra runs on Kubernetes across 3 cloud regions (US-East, US-West, EU-West) plus on-prem GPU capacity for base load.
9. Feature store online layer already exists as shared infra (see Section 15); this platform is a consumer.

## 6. Capacity Estimation

**QPS:**
- 15M concurrent players × 1.8 inferences per relevant action, actions avg every ~20s active session → per-player rate ≈ 0.09 req/s.
- Peak platform QPS ≈ 15M × 0.09 ≈ **1.35M raw feature-touching events/s** — but only a fraction hit the *inference* platform vs. pre-computed batch scores. Assume 18% require real-time inference → **≈ 240K QPS** at peak. Matches stated NFR target (250K).
- Per-model: top model (matchmaking skill score) at ~35K QPS alone; median of the 400 models at ~50 QPS.

**Storage:**
- Prediction/audit logs: 240K QPS × 2 KB avg (request+response+metadata) × 86,400s/day ≈ **41 TB/day** raw → compressed (Parquet, ~5:1) ≈ 8.2 TB/day → 30-day retention ≈ **246 TB** hot/warm columnar storage.
- Model artifact store: 400 models × 5 versions retained avg × mix of sizes ≈ (280 CPU-heavy models × 50MB × 5) + (120 GPU models × 4GB avg × 5) ≈ 70 GB + 2.4 PB... recompute: 120 × 4GB × 5 = 2,400 GB = 2.4 TB. Total artifact store ≈ **2.5 TB** (object storage, cheap).

**GPU/CPU counts:**
- GPU-bound traffic: 240K QPS × 30% ≈ 72K QPS. Assume A10G-class GPU sustains ~800 inferences/s per model instance with dynamic batching (batch=16, ~20ms/batch). → 72,000 / 800 ≈ **90 GPU replicas** at steady peak, provision **~130** for headroom/failover (N+1 per AZ × 3 AZs).
- CPU-bound traffic: 240K QPS × 70% ≈ 168K QPS. Assume a CPU pod (4 vCPU) sustains ~2,000 QPS for a small GBDT model. → 168,000 / 2,000 ≈ **84 CPU pods** at peak, provision **~120** with headroom.
- Feature-store read load: assume 60% of requests need a live fetch (rest pass full features) → 240K × 0.6 = 144K reads/s against the online store (sized in Section 15).

**Bandwidth:**
- Avg request+response payload ~4 KB combined × 240K QPS ≈ **960 MB/s ≈ 7.7 Gbps** aggregate ingress+egress at peak — fits comfortably behind standard LB/mesh sizing per region (÷3 regions ≈ 2.6 Gbps/region).

**Cost back-of-envelope:**
- GPU: 130 × A10G-class (~$1.20/hr on-demand, ~$0.40/hr spot-blended average with reserved) ≈ 130 × $0.55/hr blended ≈ **$71.5/hr ≈ $52K/month** for GPU serving fleet.
- CPU: 120 pods × 4 vCPU × ~$0.05/vCPU-hr ≈ **$24/hr ≈ $17.3K/month**.
- Against 240K QPS sustained ≈ 240K × 86,400 × 30 ≈ 622B inferences/month → blended compute cost ≈ $69.3K / 622B × 10,000 ≈ **$0.0011 per 10K inferences** for compute alone (well under the $0.15 target; remaining budget covers feature store, logging, networking, control plane).

## 7. High-Level Architecture

```
                                   ┌─────────────────────────┐
                                   │   Game Client / Service   │
                                   │ (matchmaking, chat, etc.) │
                                   └────────────┬─────────────┘
                                                │ gRPC/HTTPS
                                                ▼
                              ┌───────────────────────────────────┐
                              │        API Gateway / Edge          │
                              │  (authn, rate limit, region route) │
                              └────────────────┬────────────────────┘
                                                │
                                                ▼
                              ┌───────────────────────────────────┐
                              │      Inference Router Service       │
                              │ (model resolution, A/B, canary,     │
                              │  request validation, fallback logic)│
                              └───┬──────────────┬──────────────┬───┘
                                  │              │              │
                     ┌────────────▼──┐   ┌───────▼───────┐  ┌───▼─────────────┐
                     │ Feature Store  │   │ Batching Layer │  │ Fallback/Cache   │
                     │ Online Reader  │   │ (dynamic micro-│  │ (default scores, │
                     │ (Redis/KV)     │   │  batch queue)  │  │  last-known-good)│
                     └────────────────┘   └───────┬───────┘  └─────────────────┘
                                                    │
                       ┌────────────────────────────┼────────────────────────────┐
                       ▼                            ▼                            ▼
             ┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
             │ GPU Model Servers │        │ CPU Model Servers │        │ Explainability Svc│
             │ (Triton / KServe, │        │ (Triton CPU / ONNX│        │ (async, SHAP/     │
             │  multi-model,     │        │  RT, GBDT runtime) │        │  attribution)     │
             │  autoscaled)      │        │  autoscaled)       │        │                    │
             └────────┬─────────┘        └─────────┬─────────┘        └──────────────────┘
                       │                            │
                       └──────────────┬─────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │  Response Assembler /       │
                         │  Post-processing (calib,    │
                         │  thresholds, logging hook)  │
                         └─────────────┬───────────────┘
                                       │
                     ┌─────────────────┼─────────────────────┐
                     ▼                 ▼                     ▼
           ┌───────────────┐ ┌──────────────────┐  ┌──────────────────────┐
           │ Caller Response│ │ Prediction Log     │  │ Streaming Bus (Kafka) │
           │ (sync return)  │ │ Store (columnar)   │  │ → drift/monitoring    │
           └───────────────┘ └──────────────────┘  └──────────────────────┘

     Control Plane (out of hot path):
     ┌──────────────────────────────────────────────────────────────────────┐
     │ Model Registry │ Deployment Orchestrator │ Autoscaler Controller │    │
     │ Config/Feature-Flag Service │ Drift Detector │ Retraining Trigger Svc │
     └──────────────────────────────────────────────────────────────────────┘
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| API Gateway / Edge | TLS termination, authn, coarse rate limiting, geo-routing | HTTPS/gRPC ingress | Stateless — scales on connection count |
| Inference Router | Resolve model+version for request, apply A/B/canary rules, orchestrate feature fetch + batching + fallback | Internal gRPC | Stateless — CPU-bound, scales on QPS |
| Feature Store Online Reader | Low-latency KV lookups for player/session features | gRPC `GetFeatures(entity_ids, feature_set)` | Read replicas, scales on read QPS |
| Batching Layer | Accumulate requests into micro-batches per model within latency window | In-process queue + timer | Scales with model replica count |
| GPU Model Servers | Execute GPU-bound model inference, multi-model hosting | Triton `Infer` gRPC/HTTP | GPU replica count, autoscaled on queue depth + GPU util |
| CPU Model Servers | Execute CPU-bound model inference (GBDT/linear/small NN) | Same protocol, CPU runtime | Pod count, autoscaled on CPU util + QPS |
| Fallback/Cache Service | Serve last-known-good or heuristic default when primary fails | In-memory + Redis backing | Scales with router replica count |
| Response Assembler | Calibration, threshold application, response shaping, fire-and-forget logging | In-process library in router | N/A (embedded) |
| Prediction Log Store | Durable store of request/response/features for audit + drift | Kafka → columnar sink (Iceberg/Parquet) | Partition count on Kafka, storage scales independently |
| Model Registry | Version metadata, lineage, approval state, artifact pointers | REST/gRPC CRUD | Small, HA pair sufficient |
| Deployment Orchestrator | Rolls out new model versions (canary/blue-green), talks to K8s | Internal API + K8s API | Control-plane singleton (leader-elected) |
| Drift Detector | Streaming job comparing live feature/prediction distributions to training baseline | Consumes Kafka topic | Scales with partition count |
| Explainability Service | Async SHAP/attribution computation for flagged decisions | Queue-consumer | Scales with queue depth |

## 9. API Design

**Base URL:** `https://inference.ea-platform.internal/v2`

### Predict (synchronous, single or micro-batch)

```
POST /v2/models/{model_name}/versions/{version}:predict
Headers: Authorization: Bearer <service-jwt>, X-Request-Id, X-Tenant-Id
```

Request:
```json
{
  "instances": [
    {
      "entity_ids": {"player_id": "p_9182734", "session_id": "s_44821"},
      "features": {"skill_rating": 1423, "recent_win_rate": 0.52},
      "feature_set_ref": "matchmaking_v3"
    }
  ],
  "options": {
    "explain": false,
    "timeout_ms": 40,
    "fallback_allowed": true
  }
}
```

Response:
```json
{
  "predictions": [
    {"score": 0.812, "confidence": 0.93, "model_version": "matchmaking:v3.2.1"}
  ],
  "served_by": "gpu-pool-us-east-1c",
  "latency_ms": 11,
  "degraded": false
}
```

### Model resolution shortcut (alias-based, no version pinning)

```
POST /v2/predict?model_alias=matchmaking-prod
```
Router resolves alias → active canary-aware version per traffic-split config.

### Explainability (async)

```
POST /v2/models/{model_name}:explain
GET  /v2/explanations/{explanation_id}
```

### Model registration (control plane)

```
POST   /v2/registry/models                    — register new model
POST   /v2/registry/models/{name}/versions     — upload new version + metadata
PATCH  /v2/registry/models/{name}/traffic      — update canary/A-B split
GET    /v2/registry/models/{name}/versions     — list versions + status
```

**Versioning:** URL-path major version (`/v2`) for the platform API itself; model version is a first-class path/query param, decoupled from API version. Model version strings follow `semver` (`v3.2.1`); router supports `latest`, `champion`, or pinned exact version.

| Endpoint | Method | Purpose | Auth |
|---|---|---|---|
| `/v2/models/{name}/versions/{v}:predict` | POST | Synchronous scoring | Service JWT |
| `/v2/predict?model_alias=` | POST | Alias-routed scoring | Service JWT |
| `/v2/models/{name}:explain` | POST | Queue explanation job | Service JWT |
| `/v2/explanations/{id}` | GET | Fetch explanation result | Service JWT |
| `/v2/registry/models` | POST | Register model | ML-team OAuth + RBAC |
| `/v2/registry/models/{name}/traffic` | PATCH | Update traffic split | ML-team OAuth + RBAC |

## 10. Database Design

| Store | Type | Used For | Why |
|---|---|---|---|
| Model Registry DB | PostgreSQL (relational) | Model/version metadata, lineage, approval workflow, traffic-split config | Strong consistency needed for deployment state; low volume, relational integrity (FKs between model→version→deployment) matters |
| Online Feature Store | Redis Cluster / DynamoDB-style KV | Point lookups by entity_id for serving-time features | Sub-ms p99 reads, simple key-value access pattern, horizontally shardable |
| Prediction Log Store | Columnar (Apache Iceberg on S3/Parquet) | Durable audit log, drift analysis input | Write-heavy, append-only, analytical queries (drift jobs, audits) over huge volume — columnar cheap at 246 TB retained |
| Explainability Result Store | Document store (MongoDB-style) | Variable-shape attribution payloads keyed by explanation_id | Schema-flexible per model type |
| Config/Feature-Flag Store | etcd-backed or Postgres + cache | Canary %, kill switches, per-tenant rate limits | Needs strong consistency + fast propagation to routers |

**Registry schema sketch:**
```sql
CREATE TABLE models (
  model_id UUID PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  owner_team TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE model_versions (
  version_id UUID PRIMARY KEY,
  model_id UUID REFERENCES models(model_id),
  semver TEXT NOT NULL,
  artifact_uri TEXT NOT NULL,
  framework TEXT NOT NULL,          -- pytorch|tf|onnx|xgboost
  hardware_class TEXT NOT NULL,     -- gpu|cpu
  status TEXT NOT NULL,             -- staged|canary|champion|retired
  registered_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE traffic_splits (
  model_id UUID REFERENCES models(model_id),
  version_id UUID REFERENCES model_versions(version_id),
  weight_pct NUMERIC(5,2),
  updated_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (model_id, version_id)
);
```

**Partitioning/sharding:**
- Prediction log store: partitioned by `date` + bucketed by `model_id` hash (16 buckets) — bounds file counts, enables per-model drift queries to prune partitions.
- Online feature store: sharded by `player_id` hash (consistent hashing, 4096 slots across nodes) for even load distribution and rebalancing on node add/remove.
- Registry DB: not sharded (low volume, single primary + read replicas per region suffices).

## 11. Caching

| Cache | What's Cached | Strategy | Invalidation |
|---|---|---|---|
| Feature cache (router-local, in-memory LRU) | Recently fetched feature vectors per entity_id | Cache-aside, ~500ms TTL | TTL expiry; explicit purge on feature-store write event (via pub/sub) |
| Model artifact cache (on serving node) | Loaded model weights in GPU/CPU memory | Write-through on deploy | Evicted on new version promotion or LRU under memory pressure (multi-model serving) |
| Last-known-good prediction cache | Most recent successful prediction per entity+model, for fallback | Write-through on every successful response | TTL 5 min; used only when primary path fails/times out |
| Config cache (traffic-split, kill-switch flags) | Router-local copy of control-plane config | Cache-aside w/ push-based invalidation (watch on etcd/config service) | Push-invalidated within ~200ms of control-plane change |

Cache-aside dominates for feature/config data (read-heavy, tolerant of brief staleness). Write-through used specifically for last-known-good fallback cache and model artifacts, where the "write" (successful inference / new deploy) must be reflected before it's needed for degraded-path correctness.

## 12. Queues & Async Processing

| Queue | Payload | Delivery Semantics | Dead-Letter Handling |
|---|---|---|---|
| Prediction-log ingestion (Kafka) | request+response+features+metadata | At-least-once (consumer idempotent via request_id dedup) | Failed sink writes retried 5x w/ backoff → DLQ topic → alert if DLQ depth > 10K |
| Explainability job queue | model_name, version, instance, explanation_id | At-least-once | Retry 3x → DLQ; DLQ consumer pages on-call if attribution backlog > 15 min |
| Model deployment events | version promoted/rolled-back | Exactly-once (via idempotent deployment orchestrator keyed on version_id) | Failed rollout steps halt pipeline, alert deploy owner, no silent DLQ (deploys must be seen) |
| Drift-metric computation jobs | windowed batch references (time range, model_id) | At-least-once, idempotent recompute | Failures logged; job retried on next scheduled window if miss persists >2 windows |

Exactly-once is only truly needed for **deployment state transitions** (double-promoting or double-rolling-back a version is dangerous); everything data-plane-adjacent (logs, explainability, drift) tolerates at-least-once with idempotent consumers since duplicate log rows / duplicate explanation computations are cheap and harmless.

## 13. Streaming & Event-Driven Architecture

**Kafka topics:**

| Topic | Producer | Consumers | Schema (Avro/Protobuf) |
|---|---|---|---|
| `inference.predictions.v1` | Response Assembler | Prediction Log sink, Drift Detector, Explainability trigger | `{request_id, model_name, version, entity_ids, features, prediction, confidence, latency_ms, served_by, ts}` |
| `inference.errors.v1` | Router (on failure/fallback) | Alerting pipeline, SRE dashboards | `{request_id, model_name, error_type, fallback_used, ts}` |
| `feature-store.updates.v1` | Feature pipelines (upstream) | Router feature-cache invalidator | `{entity_id, feature_set, updated_features, ts}` |
| `registry.deployment-events.v1` | Deployment Orchestrator | Autoscaler, Drift Detector (baseline reset), Audit log | `{model_name, version, action, actor, ts}` |

**Consumer groups:**
- `drift-detector-group`: partitioned consumption keyed by `model_id` hash so each model's distribution stream is processed in order by one consumer instance (ordering matters for windowed stats).
- `prediction-log-sink-group`: high-parallelism, order doesn't matter, scales to partition count (64 partitions on `inference.predictions.v1`).
- `explainability-trigger-group`: filters for `flagged=true` predictions only (fraud/ban-adjacent), low volume.

## 14. Model Serving

**Framework choice:** NVIDIA Triton Inference Server as the common serving runtime — supports PyTorch, TensorFlow SavedModel, ONNX, and custom Python backends (for GBDT via a thin wrapper) in one process, with native **dynamic batching** and **multi-model** concurrent execution (concurrent model instances sharing a GPU via MPS/CUDA streams).

- **Multi-model serving:** Triton model repository per node hosts N models; models below a QPS threshold are packed onto shared GPU replicas (bin-packed by memory footprint + expected QPS) rather than each getting dedicated GPUs — critical given the power-law traffic (340 of 400 models are long-tail, low-QPS).
- **Dynamic batching:** configured per model — `max_queue_delay_microseconds` tuned per model's latency budget (e.g., 8ms window for a model with 15ms p50 budget, vs 2ms window for a model with 8ms budget). Preferred batch sizes set to GPU-efficient powers of 2 (8/16/32).
- **CPU serving:** GBDT models (XGBoost/LightGBM) served via Triton's FIL (Forest Inference Library) backend or a lightweight custom backend — CPU-only, no batching benefit needed since these are already sub-ms per inference; horizontal pod scaling handles throughput.
- **Hardware:** GPU pool uses A10G-class (cost-efficient inference GPU, not training-grade A100/H100) since these are inference-only, latency-bound, not throughput-of-huge-batches-bound workloads. CPU pool uses standard compute-optimized instances.
- **Model warm-pool:** every registered "champion" model kept loaded in at least 2 replicas per region even at near-zero traffic, to avoid cold-load latency spikes (multi-GB model load from object storage can take seconds — unacceptable mid-request).

## 15. Feature Store

- **Online store:** Redis Cluster (or equivalent low-latency KV), sharded by `player_id`, holding last-computed values for ~150 real-time features per entity, refreshed by streaming feature pipelines (session state, rolling win-rate, recent-chat-toxicity-score, etc.) with target staleness ≤ 1s.
- **Offline store:** columnar warehouse (feature history, used for training) — same feature definitions computed in batch for historical backfill.
- **Point-in-time correctness:** feature definitions registered once (feature registry) with both a streaming (online) and batch (offline) computation path guaranteed logically equivalent; training-time feature retrieval joins on **event timestamp** against the offline store to avoid future leakage (i.e., "what would this feature have been at request time," not "what is it now"). This platform, as an online-serving consumer, always reads *current* online values — the point-in-time guarantee matters for the training pipeline (Section 19), not this serving path, but serving depends on the two paths staying consistent (skew is a drift-detector signal, Section 21).
- Feature-freshness SLA breach (staleness > 5s) triggers a router-side fallback to cached last-known-good feature values rather than blocking.

## 16. Vector Database

**N/A for the core low-latency scoring path** — the majority of registered models (matchmaking scores, churn/fraud GBDTs, DDA) consume structured/tabular features, not embedding similarity search.

Exception: models that *use* embeddings as inputs (e.g., a player-behavior embedding feeding a churn model) fetch a **precomputed embedding vector directly from the online feature store** (treated as just another feature, fixed-size float array) rather than doing an ANN similarity search at inference time. If a *recommendation-style* nearest-neighbor lookup is needed (e.g., "find similar cosmetic items"), that's a distinct use case better covered under a Recommendation/Search system chapter with its own vector DB (e.g., HNSW-based) — out of scope here since it changes the latency/consistency profile substantially (ANN search adds tens of ms).

## 17. Embedding Pipelines

**Partially applicable.** Some served models (deep ranking/DDA/toxicity) consume embeddings as inputs, but this platform does not *compute* embeddings inline in the request path — that would add unacceptable latency and duplicate work across callers.

- Embeddings (player-behavior, item, text/chat) are computed **offline/streaming upstream** by a separate embedding-generation pipeline (batch nightly for slow-moving entities like items; streaming near-real-time for fast-moving ones like session behavior), then written into the online feature store as fixed-size vectors.
- This platform's only embedding-adjacent responsibility: serving models whose **first layer** is an embedding lookup table (e.g., learned item/player-ID embeddings baked into the model artifact itself) — those are just part of the model graph executed by Triton, not a separate pipeline.
- Rationale for keeping generation out-of-band: decouples expensive embedding refresh (can be GPU-heavy, batched, run on a schedule) from the strict-latency inference path, and lets multiple models reuse the same precomputed embedding without recomputation.

## 18. Inference Pipelines

**Request lifecycle, end-to-end:**

```
t=0ms    Client sends request → API Gateway
t=0.5ms  Gateway: TLS terminate, authn check, rate-limit check, region route
t=1ms    Inference Router receives request
t=1.2ms  Router: resolve model alias → version (check canary/AB config, cached locally)
t=1.5ms  Router: check if full features provided; if not → async call to Feature Store Online Reader
t=1.5-4ms  Feature Store Reader: KV lookup by entity_id, return feature vector (parallel fan-out if multi-entity)
t=4ms    Router: assemble model input tensor, enqueue into batching layer for resolved model+version
t=4-12ms Batching Layer: accumulate micro-batch (up to configured window, e.g. 8ms) or flush early if batch full
t=12ms   Batch dispatched to Model Server (Triton) — GPU or CPU pool per hardware_class
t=12-20ms Model Server: forward pass, return raw output tensor(s)
t=20ms   Response Assembler: apply calibration/thresholding, attach metadata (version, latency, served_by)
t=20.2ms Fire-and-forget: publish to inference.predictions.v1 (Kafka), do not block response
t=20.5ms Router returns response to Gateway → Client
-----
Total p50 target: ~15ms as stated (above trace shows a p90-ish trace with batching wait included)
On failure at any stage past t=1ms: Router checks options.fallback_allowed → serves Last-Known-Good cache
  or static default → tags response degraded:true → still publishes to inference.errors.v1
```

Second ASCII diagram — **fallback/degradation path**:

```
              ┌───────────────┐
   request →  │ Inference     │
              │ Router        │
              └───────┬───────┘
                      │ try primary path (feature fetch → batch → model server)
                      ▼
              ┌───────────────┐   success   ┌────────────────────┐
              │ Primary Path   │───────────▶│ Return prediction   │
              │ (timeout=40ms) │             │ (degraded:false)    │
              └───────┬───────┘             └────────────────────┘
                      │ timeout / error
                      ▼
              ┌───────────────┐   hit       ┌────────────────────┐
              │ Last-Known-Good│───────────▶│ Return cached score  │
              │ Cache lookup   │             │ (degraded:true)      │
              └───────┬───────┘             └────────────────────┘
                      │ miss
                      ▼
              ┌───────────────┐             ┌────────────────────┐
              │ Static Default │────────────▶│ Return default score │
              │ / Heuristic    │             │ (degraded:true)      │
              └───────────────┘             └────────────────────┘
```

## 19. Training Pipelines

- **Data prep:** offline feature store joins (point-in-time correct) → labeled training sets written to versioned datasets (data-version-controlled alongside code, e.g. lakeFS/DVC-style pointers) → validated via schema/statistics checks (expected ranges, null rates) before training kicks off.
- **Training orchestration:** Kubeflow Pipelines / Argo Workflows DAGs — feature extraction → train → eval → validation-gate → registration. Triggered by (a) schedule, (b) retraining-strategy signal (Section 20), or (c) manual ML-engineer trigger.
- **Distributed training:** relevant for the GPU-bound deep ranking/DDA models — PyTorch DDP across multi-GPU nodes for large sequence/ranking models; GBDT models train single-node (XGBoost histogram method, sometimes multi-core distributed via Dask for very large tabular sets).
- **Handoff to serving:** successful, validated training run registers a new `model_version` in the Registry with status `staged`; does not auto-promote — promotion goes through canary (Section 33).
- This platform (real-time inference) is a **consumer** of the training pipeline's output artifacts; the training pipeline itself is detailed fully in a separate "Model Training Platform" chapter — referenced here only for the interface boundary (artifact_uri + metadata handed to the Registry).

## 20. Retraining Strategy

**Cadence:**
- High-traffic, fast-drifting models (matchmaking skill, live-event DDA): retrain **weekly**, scheduled.
- Medium-drift models (churn, chat toxicity): retrain **bi-weekly to monthly**.
- Low-drift, stable models (some fraud heurist422 features): **quarterly** or trigger-only.

**Triggers (in addition to schedule):**
- Drift Detector fires **data drift** alert (PSI > threshold, Section 21) on a top-20 model → auto-kick a retraining job (still requires human approval to promote).
- Concept drift signal: online business-metric proxy (e.g., match-quality complaint rate, chat-report rate) degrades beyond threshold for 3 consecutive days.
- Major game content patch/season launch — scheduled proactive retrain regardless of drift signal (known distribution shift event, e.g., new weapon/hero/item release changes player behavior).
- Manual trigger by model owner via Registry API.

## 21. Drift Detection

| Drift Type | Metric | Threshold | Action |
|---|---|---|---|
| Data drift (feature distribution shift) | Population Stability Index (PSI) per feature, computed daily vs training baseline | PSI > 0.2 = alert (moderate), PSI > 0.3 = page + auto-retrain trigger | Alert model owner; if top-20 model, auto-kick retraining job |
| Data drift (categorical features) | Chi-squared / Jensen-Shannon divergence vs baseline | JS divergence > 0.1 | Alert model owner |
| Concept drift (label/outcome relationship shift) | Rolling online accuracy proxy (e.g., matchmaking: post-match skill-rating-delta variance vs prediction) | Proxy error increase > 15% over 7-day rolling baseline | Page on-call ML engineer, flag model for review |
| Prediction drift (output distribution shift, no ground truth needed) | KL divergence of prediction score distribution, daily vs 30-day baseline | KL > 0.15 | Alert; investigate upstream feature pipeline health first (often a feature-pipeline bug, not real drift) |
| Feature-serving skew (online vs offline) | Distribution comparison of same feature computed online vs offline for same entity/timestamp | Any feature with mismatch rate > 1% of sampled requests | Page platform team — indicates feature-store bug, higher urgency than model drift |

Computed by the streaming Drift Detector consuming `inference.predictions.v1`, windowed hourly/daily, with baselines refreshed at each model promotion (new champion resets the baseline).

## 22. Monitoring

**Infra:**
- GPU utilization %, GPU memory occupancy, batch queue depth, batch fill-rate (actual vs preferred batch size), pod restart counts, node autoscaler activity.
- Request rate, error rate, latency percentiles (p50/p90/p99/p99.9) per model+version+region.

**Model quality:**
- Drift metrics (Section 21), calibration curve drift, prediction confidence distribution, fallback/degraded-response rate.
- Per-version comparison during canary (champion vs challenger latency + quality deltas).

**Business metrics:**
- Match quality proxy (skill-delta variance, session length post-match), chat-moderation false-positive appeals rate, churn-model precision@k against actual observed churn, fraud-model $ prevented vs false-positive player friction.
- Cost-per-10K-inferences trend (catches silent cost regressions from batching misconfiguration or GPU underutilization).

## 23. Alerting

| Condition | Threshold | Severity | Routing |
|---|---|---|---|
| p99 latency breach | > 60ms sustained 5 min | High | Page platform on-call |
| p99.9 latency breach | > 150ms sustained 2 min | Critical | Page platform on-call + auto-trigger circuit breaker |
| Error rate | > 1% over 5 min | High | Page platform on-call |
| Fallback/degraded rate | > 5% over 10 min | Medium | Notify platform on-call (Slack), page if > 15% |
| GPU pool utilization | > 90% sustained 10 min (pre-saturation) | Medium | Notify platform on-call, autoscaler should already be reacting |
| Drift PSI > 0.3 on top-20 model | immediate | Medium | Notify model-owner team channel, auto-kick retrain |
| Feature-serving skew > 1% | immediate | High | Page platform on-call (data-correctness issue) |
| DLQ depth (any queue) | > 10K messages | Medium | Notify data-eng on-call |
| Deployment rollout failure | any | High | Page deploying engineer + platform on-call |
| Registry DB replication lag | > 30s | Medium | Notify platform on-call |

On-call routing: platform infra issues → Inference Platform SRE rotation; model-quality issues → respective model-owner team's on-call (routed via model metadata in Registry, e.g. PagerDuty service mapped per `owner_team`).

## 24. Logging

- **Structured logging:** every request emits a structured JSON log (via `inference.predictions.v1` for successes, `inference.errors.v1` for failures) with `request_id`, `model_name/version`, `latency_ms`, `served_by`, `degraded` flag — never free-text log lines for anything queryable.
- **PII handling:** raw player identifiers (`player_id`) are **pseudonymized** (salted hash) before landing in the long-retention Prediction Log Store; a separate short-retention (7-day), access-controlled raw-ID mapping table exists only for active incident debugging, purged automatically.
- **Chat/text features** (for toxicity models) are never logged in raw form in the long-term store — only derived scores/features; raw text retained max 48h in a restricted-access debug store for appeals handling, per data-minimization policy.
- **Retention:** prediction logs 30 days hot (Section 6), archived to cold storage (compressed Parquet) for 1 year for audit/compliance, then deleted unless under legal hold. Debug/raw-ID logs 7 days. Explanation results 90 days (appeals window).
- **Regional data residency:** EU-region traffic's logs stay in EU-region storage (GDPR); no cross-region replication of raw log data, only aggregated/anonymized drift metrics replicate globally.

## 25. Security

**Threat model specific to this system:**
- **Model exfiltration:** attacker attempts to extract model weights via the artifact store or via repeated query-based model-stealing (systematic probing to reconstruct decision boundary). Mitigation: artifact store IAM-locked to serving nodes + registry service only; per-tenant/per-caller rate limiting (Section 27) bounds query-based extraction feasibility; anomaly detection on unusually systematic query patterns (e.g., one caller sweeping feature space).
- **Adversarial input manipulation:** malicious client crafts feature inputs to force a favorable prediction (e.g., manipulate anti-cheat or fraud-score inputs). Mitigation: features that are security-sensitive (fraud, anti-cheat) are computed server-side by the platform's own feature store (not caller-suppliable), never accepted directly from untrusted client-facing services without an internal-service-only trust boundary.
- **Prediction-log data exposure:** long-retention logs contain behavioral data; breach risk if store is misconfigured public. Mitigation: encryption at rest (Section below), strict IAM, PII pseudonymization (Section 24).
- **Model poisoning via feature pipeline compromise:** if upstream feature pipeline is compromised, bad features flow into both training and serving. Mitigation: feature-serving skew monitor (Section 21) and schema/range validation gate at feature-store write time.

**Encryption:**
- At rest: all stores (Registry DB, Feature Store, Prediction Log Store, artifact store) encrypted with provider-managed KMS keys, per-region keys (supports residency + crypto-shredding for deletion requests).
- In transit: mTLS between all internal services (Gateway↔Router↔Model Servers↔Feature Store), enforced via service mesh (Istio/Linkerd) sidecars.

## 26. Authentication

- **Service-to-service:** mTLS certificate identity (SPIFFE/SPIFFE-ID-based) issued per service via the mesh's CA, short-lived (24h rotation); additionally, a signed **service JWT** (issued by an internal STS) carries `tenant_id`/`owner_team` claims for authorization decisions at the Router layer (defense in depth beyond mesh identity).
- **End-user auth:** end players never call this platform directly — always via a game-service backend that has already authenticated the player (EA account/session token validated upstream). This platform trusts the calling service's identity, not end-user tokens directly, but propagates a `player_id` claim for feature lookups/audit.
- **Control-plane (model registration/deploy) auth:** OAuth2 + RBAC — ML engineers authenticate via corporate SSO, scoped permissions per `owner_team` (a churn-model team cannot modify matchmaking-model traffic splits).

## 27. Rate Limiting

- **Algorithm:** token bucket per (tenant/caller service, model_name) pair — allows short bursts (live-event spikes) while bounding sustained abuse, implemented at the Gateway layer (cheap, stateless-ish via distributed counter in Redis with a sliding approximation).
- **Limits:** default per-tenant limit sized at 1.5x their provisioned/expected peak QPS (from onboarding capacity request), with a hard ceiling per model to protect shared multi-tenant GPU pools from one noisy tenant.
- **Per-tenant isolation:** high-priority tenants (top-20 models) get **dedicated capacity pools** (reserved GPU/CPU replica quota) rather than relying purely on rate limiting — rate limiting is the backstop for long-tail/unexpected-spike tenants sharing pooled capacity.
- **Response on limit breach:** HTTP 429 / gRPC `RESOURCE_EXHAUSTED` with `Retry-After` — calling services expected to have their own fallback (most game services already have local static defaults for exactly this case).

## 28. Autoscaling

- **GPU pools:** KEDA-driven HPA scaling on a custom metric — **batch queue depth** and **GPU utilization %**, not just CPU (CPU is nearly irrelevant for GPU-bound Triton pods). Scale-out trigger: queue depth > 50 for 30s OR GPU util > 75% sustained 1 min. Scale-in: conservative, 10-minute cooldown to avoid thrashing given multi-second GPU pod cold-start (image pull + model load).
- **CPU pools:** standard HPA on CPU utilization (target 60%) + request-rate custom metric, faster scale-in cooldown (2 min) since CPU pod cold-start is sub-second.
- **VPA:** used in recommendation-only mode for right-sizing memory requests on GPU pods hosting multiple small models (avoid OOM-kills from memory-request under-provisioning without fighting HPA's scaling decisions).
- **Predictive pre-scaling:** for known live-service events (scheduled in-game events, new season launch) — scheduled scale-up (cron-triggered baseline replica floor increase) ahead of expected traffic, since reactive autoscaling alone can't react fast enough to a 3-5x spike within minutes.
- **Warm pool floor:** minimum replica floor per region per champion model (never scale to zero for top-20 models) to avoid cold-start latency spikes; long-tail models permitted to scale toward a low floor (e.g., 2 replicas) but not zero, given multi-GB load times.

## 29. Cost Optimization

- **Spot/preemptible instances:** long-tail CPU model pools and non-champion GPU replicas run on spot capacity (interruption-tolerant given N+1 redundancy and fast pod rescheduling); champion models' baseline floor stays on-demand/reserved for interruption-sensitivity.
- **Multi-model bin-packing:** long-tail models (340 of 400) share GPU replicas via Triton's multi-model hosting rather than 1 GPU per model — biggest single lever, avoids ~300 idle dedicated GPUs.
- **Dynamic batching tuning:** improves GPU throughput per replica (fewer replicas needed for same QPS) — directly reduces GPU-hour spend; tuned per-model to the latency budget's slack.
- **Model distillation/quantization:** large deep-ranking models distilled to smaller student models or quantized (FP16/INT8) where accuracy loss is acceptable, cutting both GPU memory footprint (more models per GPU) and per-inference latency/cost.
- **Reserved capacity for baseline load:** commit to reserved instances for the predictable diurnal floor (e.g., 60% of average GPU need), spot/on-demand for the variable peak — classic baseline-vs-burst cost split.
- **Cache hit reduction of redundant inference:** last-known-good cache (Section 11) also functions as a cost lever — repeated identical requests within TTL window (e.g., rapid retries) avoid redundant GPU compute.
- **Log/storage tiering:** prediction logs move hot→cold storage aggressively (7-day hot window instead of full 30-day hot) to cut columnar storage cost, since drift jobs mostly need recent data with occasional cold-storage backfill queries.

## 30. Disaster Recovery

| Target | Value |
|---|---|
| RTO (platform-wide outage) | 15 minutes (fail over to secondary region) |
| RTO (single model version bad deploy) | 5 minutes (auto-rollback) |
| RPO (prediction logs) | ≤ 1 minute (Kafka replication lag tolerance) |
| RPO (model registry state) | 0 (synchronous multi-AZ Postgres replication) |
| RPO (online feature store) | ≤ 5 seconds (async cross-region replication, acceptable given feature staleness tolerance) |

- **Backup strategy:** Registry DB — continuous WAL archiving + daily snapshot, cross-region replica. Model artifacts — versioned, immutable in object storage with cross-region replication (artifacts never deleted, only marked retired). Prediction logs — Kafka topic replicated across 3 brokers min, sink checkpoints allow replay from last committed offset on sink failure.
- **Runbook triggers:** region-level health-check failure → traffic-manager reroutes to healthy regions (Section 31) automatically within RTO target; region rejoin requires manual verification before receiving traffic again (avoid flapping into a still-degraded region).

## 31. Multi-Region Deployment

**Topology: Active-active** across US-East, US-West, EU-West.

- Player traffic routed to nearest healthy region via latency-based DNS/Anycast + Gateway-level health checks; EU traffic pinned to EU-West by default for residency, with documented consent-based fallback only in a full EU-region outage (rare, policy-gated).
- Each region runs a **full independent serving stack** (Router, Model Servers, Feature Store replica, local Kafka cluster) — no cross-region synchronous calls in the hot path (would blow the latency budget).
- **Data replication:** Model Registry — single global source of truth (US-East primary) with async read replicas in each region; deployments propagate to all regions' local config within ~5s via the config-push mechanism (Section 11). Online Feature Store — per-region local store populated by region-local streaming pipelines reading from a globally-replicated raw event stream (each region computes its own online features locally to avoid cross-region read latency; only the upstream raw event topic is globally replicated, not the derived KV store).
- **Failure isolation:** a region losing its feature-store or model-server pool degrades to fallback/default scores locally (Section 12/18) rather than cross-region calling (cross-region call would add 80-150ms round trip, violating p99 budget outright).

```
        US-West Region              US-East Region             EU-West Region
     ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
     │ Gateway/Router     │      │ Gateway/Router     │      │ Gateway/Router     │
     │ GPU/CPU pools       │      │ GPU/CPU pools       │      │ GPU/CPU pools       │
     │ Local Feature Store │      │ Local Feature Store │      │ Local Feature Store │
     │ Local Kafka         │      │ Local Kafka         │      │ Local Kafka (EU-res)│
     │ Registry read-replica│     │ Registry PRIMARY     │      │ Registry read-replica│
     └─────────┬───────────┘      └─────────┬───────────┘      └─────────┬───────────┘
               │                            │                            │
               └──────────── async replication / config push ────────────┘
                         (registry writes, raw event stream for
                          per-region feature computation, drift
                          metric rollups only — no hot-path calls)
```

## 32. Blue/Green Deployment

Applies primarily at the **serving infrastructure/runtime** level (Triton version upgrades, base image/dependency changes, K8s node-pool migrations) rather than per-model-version rollout (that's canary, Section 33):

- Stand up a full parallel "green" fleet (new Triton version / new node pool AMI) alongside "blue" (current), both loaded with the same set of active model versions.
- Shift a small % of infra-level traffic (via Gateway routing weight, not model traffic-split) to green, validate infra-level health metrics (pod stability, latency parity, no memory leaks under multi-model load) over a soak period (e.g., 2-4 hours).
- Full cutover once green matches blue on latency/error parity; blue kept warm for immediate rollback for 24h before decommission.
- Used specifically for platform-level changes (serving runtime upgrade) because a bad runtime upgrade could silently corrupt *all* models simultaneously — too risky for a gradual canary-only approach at the infra layer.

## 33. Canary Deployment

Applies at the **per-model-version** level — the routine deployment path for ML teams shipping new model versions:

- New version registered as `staged` → promoted to `canary` with initial traffic weight (e.g., 1-5%), configurable via `PATCH /v2/registry/models/{name}/traffic`.
- **Health-check gates** (automated, must pass before auto-increment to next traffic step):
  - Latency parity: canary p99 within 10% of champion p99.
  - Error rate: canary error rate not exceeding champion by more than 0.5 percentage points.
  - Prediction-distribution sanity: canary output distribution not wildly divergent from champion (KL divergence below a guard threshold) — catches gross bugs (e.g., wrong feature ordering) before business-metric-level canary analysis even begins.
  - Business-metric guard (where available in near-real-time, e.g., a fast-feedback proxy): no significant regression over a minimum soak window (e.g., 2 hours at each traffic step).
- Traffic ramp: 1% → 5% → 25% → 50% → 100%, each step gated by the above, with a minimum soak time per step (auto-advance disabled by default for high-stakes models like fraud/anti-cheat — requires human sign-off at each gate for those).
- Shadow-mode option (FR5): new version receives mirrored real traffic with **no impact on served response** (response from champion still returned to caller) purely for offline comparison before even entering canary — used for the riskiest model classes.

## 34. Rollback Strategy

- **Automated triggers:** any canary health-check gate failure (Section 33) → automatic traffic weight reset to 0% for the canary version, alert deploying engineer, no human action required to *stop* the bleeding (human required to diagnose/re-attempt).
- **SLO-breach auto-rollback:** if a **champion** version itself starts breaching platform SLOs post-full-rollout (detected within the first 30 minutes post-100%-cutover window especially) — Deployment Orchestrator auto-reverts traffic to the immediately prior champion version (kept warm, not decommissioned, for 24h post-promotion specifically to enable this).
- **Rollback mechanics:** traffic-split is purely a config change (weights in `traffic_splits` table, pushed to routers) — no redeploy/rebuild needed, so rollback is a config-propagation-speed operation (~seconds to low single-digit minutes for full propagation), not an infra operation. This is the core reason model artifacts for the prior version are kept loaded/warm rather than evicted immediately on promotion.
- **Manual rollback:** available at any time via the same traffic-split API, used for slow-burn quality regressions caught by drift/business-metric monitoring (not fast enough for automated triggers, but still needing before the next scheduled retrain).

## 35. Observability

- **Tracing:** distributed tracing (OpenTelemetry) spans the full request lifecycle — Gateway → Router → Feature Store call → Batching wait → Model Server → Response Assembler — with `request_id` as the trace correlation key propagated via headers/metadata at every hop. Critical for diagnosing *where* in an 11ms request the time actually went (Section 43) without guessing.
- **Metrics:** RED metrics (Rate, Errors, Duration) per component + USE metrics (Utilization, Saturation, Errors) per GPU/CPU resource pool, all in a time-series store (Prometheus-compatible) with per-model-version labels for canary comparison dashboards.
- **Logs:** structured logs (Section 24) tagged with the same `request_id`/`trace_id`, shippable to a log-aggregation backend, correlatable with traces via that shared ID — enables "show me the full trace + logs for this one failed request" workflows during incident response.
- **Correlation in practice:** an alert fires on p99 breach (metrics) → on-call pulls the trace waterfall for a sample of slow `request_id`s in that window (traces) → cross-references structured error logs for those same IDs (logs) to find e.g. "batching queue depth spiked because one GPU node was OOM-killed" — the three pillars used together, not in isolation.

## 36. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-gpu-pool-matchmaking
  labels: {app: triton-server, hardware: gpu, model-tier: champion}
spec:
  replicas: 6
  selector:
    matchLabels: {app: triton-server, pool: matchmaking-gpu}
  template:
    metadata:
      labels: {app: triton-server, pool: matchmaking-gpu}
    spec:
      nodeSelector: {accelerator: nvidia-a10g}
      containers:
        - name: triton
          image: registry.ea.internal/triton-server:24.05-multimodel
          resources:
            requests: {cpu: "4", memory: "16Gi", nvidia.com/gpu: "1"}
            limits:   {cpu: "8", memory: "24Gi", nvidia.com/gpu: "1"}
          args: ["--model-repository=s3://model-artifacts/matchmaking",
                  "--dynamic-batching", "--max-queue-delay-microseconds=8000"]
          readinessProbe:
            httpGet: {path: /v2/health/ready, port: 8000}
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata: {name: triton-matchmaking-svc}
spec:
  selector: {pool: matchmaking-gpu}
  ports: [{port: 8001, targetPort: 8001, name: grpc}]
---
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata: {name: triton-matchmaking-scaler}
spec:
  scaleTargetRef: {name: triton-gpu-pool-matchmaking}
  minReplicaCount: 6
  maxReplicaCount: 40
  cooldownPeriod: 600
  triggers:
    - type: prometheus
      metadata:
        query: avg(triton_batch_queue_depth{pool="matchmaking-gpu"})
        threshold: "50"
```

## 37. Terraform Infrastructure

```hcl
resource "google_container_node_pool" "gpu_inference_pool" {
  name       = "inference-gpu-a10g"
  cluster    = google_container_cluster.inference_platform.name
  location   = var.region
  node_count = 3

  autoscaling {
    min_node_count = 3
    max_node_count = 20
  }

  node_config {
    machine_type = "g2-standard-8"
    guest_accelerator {
      type  = "nvidia-l4"
      count = 1
    }
    labels = { pool = "gpu-inference", tier = "champion" }
    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }
  }
}

resource "aws_elasticache_replication_group" "feature_store_online" {
  replication_group_id       = "feature-store-${var.region}"
  description                = "Online feature store - low latency KV"
  node_type                  = "cache.r6g.xlarge"
  num_node_groups            = 8
  replicas_per_node_group    = 2
  automatic_failover_enabled = true
  multi_az_enabled           = true
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
}

resource "aws_msk_cluster" "inference_events" {
  cluster_name           = "inference-predictions-${var.region}"
  kafka_version           = "3.6.0"
  number_of_broker_nodes  = 6
  broker_node_group_info {
    instance_type   = "kafka.m5.2xlarge"
    client_subnets  = var.private_subnet_ids
    storage_info { ebs_storage_info { volume_size = 2000 } }
  }
  encryption_info { encryption_in_transit { client_broker = "TLS" } }
}
```

## 38. Why This Architecture

- **Multi-tenant, framework-agnostic serving (Triton)** avoids every ML team building bespoke serving infra — one platform, one on-call rotation, consistent SLOs, and enables cost-efficient multi-model bin-packing across the long tail of 340 low-QPS models, which would be prohibitively expensive with dedicated infra per model.
- **Decoupling feature computation/embedding generation from the hot path** keeps the synchronous request path lean (feature store lookup, not feature *computation*) — essential to hit the 15ms p50 budget.
- **Config-driven traffic-split rollback** (vs. redeploy-based rollback) makes rollback a sub-minute operation, which matters enormously given live-service games have zero tolerance for a bad matchmaking/anti-cheat model staying live during peak weekend traffic.
- **Region-local everything in the hot path** (no cross-region synchronous calls) is the only way to hit global p99 targets while operating active-active across 3 regions with EU residency constraints.
- **Fail-open with graceful degradation** matches the actual risk profile of the consuming systems — a slightly-stale matchmaking score is vastly preferable to a blocked matchmaking request; this shapes almost every latency/availability tradeoff made above.

## 39. Alternative Architectures

| Alternative | Description | Why Rejected / When Preferred |
|---|---|---|
| Per-team dedicated serving stacks (no shared platform) | Each game studio runs its own model server, own autoscaling, own on-call | Rejected at EA scale: massively duplicates GPU idle capacity (can't bin-pack across teams' long-tail models), duplicates on-call/ops burden across ~dozens of teams. Preferred only for a single-studio, single-model bespoke system with unique hardware needs (e.g., a highly specialized vision model needing custom accelerators) that doesn't fit the shared multi-model pattern. |
| Serverless/FaaS-based inference (e.g., Lambda-style per-request cold containers) | Each request spins a function invocation | Rejected: cold-start latency (hundreds of ms to seconds for GPU-backed functions) blows the p99 budget entirely; no viable dynamic batching across concurrent invocations. Preferred for very low-QPS, latency-insensitive internal tools (e.g., an offline dashboard scoring tool), not this hot-path system. |
| Fully embedded model-in-client / edge inference | Ship small model to game client, infer locally (no network round-trip) | Rejected as the primary pattern here because most of these models (matchmaking, fraud, anti-cheat) require server-side authoritative features and must not be client-manipulable (anti-cheat/fraud explicitly must NOT run client-side, trivially bypassed). Preferred/used elsewhere for pure UX-personalization models with no anti-cheat/fraud sensitivity and strict offline-play requirements — a different chapter's problem. |
| Batch-only pre-computed scores (no real-time path at all) | Precompute all player scores on a schedule, serve from a simple KV cache | Rejected as the sole approach: session-context-dependent decisions (in-match DDA, live chat toxicity) need current-session features unavailable at batch time. Preferred as a *complement* (many "real-time" needs are actually served by pre-computed batch scores plus this platform only for the genuinely session-live subset) — in fact the real system is a hybrid, and this chapter's 18% real-time-fraction assumption (Section 6) reflects that split. |

## 40. Tradeoffs

| Decision | Benefit | Cost/Risk |
|---|---|---|
| Multi-model GPU bin-packing | Huge cost savings on long-tail models | Noisy-neighbor risk; requires careful resource isolation/QoS tuning on Triton |
| Fail-open degradation | Availability from caller's perspective stays high even during platform incidents | Silent accuracy degradation risk if `degraded` flag isn't actually monitored/alerted on by callers |
| Region-local feature stores (no cross-region reads) | Meets latency budget, respects data residency | Feature values can diverge slightly across regions for a roaming player mid-session (edge case, accepted) |
| Config-based instant rollback | Sub-minute recovery from bad deploys | Requires keeping N-1 model version warm/loaded at all times — extra memory/GPU cost overhead |
| Dynamic batching | Big GPU throughput/cost win | Adds latency (batching window) directly into the critical path — a direct latency-vs-cost dial that must be tuned per model |
| Shared platform vs per-team stacks | Ops efficiency, consistent SLOs, cost efficiency | Platform team becomes a critical dependency/bottleneck for all ML teams; requires strong self-service tooling to avoid becoming a deployment queue |
| Synchronous feature-store fetch in hot path (vs. requiring callers to always pass full features) | Much better DX for calling teams, consistent features | Adds a network hop + dependency into the latency-critical path; feature-store outage now directly threatens inference SLOs (mitigated by fallback cache) |

## 41. Failure Modes

| Scenario | Symptom | Mitigation |
|---|---|---|
| Feature Store online cluster partial outage | Elevated latency/timeouts on feature fetch step | Router falls back to feature cache (Section 11) or last-known-good prediction; circuit breaker trips after N consecutive timeouts to skip straight to fallback |
| GPU node OOM-kill from multi-model over-packing | Sudden pod restarts, batch queue backup, latency spike for co-located models | VPA-informed bin-packing limits, per-model memory quotas enforced in Triton config, alerting on OOM-kill rate |
| Bad model version silently degrades quality without breaching latency/error SLOs | Canary health checks (latency/error) pass, but business-metric quality regresses | Business-metric guard gate in canary process (Section 33) + drift/prediction-distribution monitoring catches this even post-full-rollout |
| Kafka broker outage (prediction-log topic) | Prediction logging fire-and-forget fails; drift jobs starve for data | Logging path is explicitly non-blocking for the serving response (client unaffected); DLQ/retry on producer side; drift jobs alert on stale-input if no new data arrives within expected window |
| Registry DB primary failure | Deployment/traffic-split changes can't be made; new registrations blocked | Read replicas keep routers functioning off cached last-known config (routers don't need live DB access per-request, only config-push updates); automated failover promotes replica within RTO target |
| Thundering herd on a live-event traffic spike (3-5x in minutes) | Reactive autoscaling can't keep up, queue depth spikes, latency breaches | Predictive pre-scaling for known events (Section 28); warm-pool floor absorbs first wave; rate limiting protects shared pools from spillover into unrelated tenants |
| Cross-region config propagation lag during incident | A region continues routing to a version that was just rolled back elsewhere | Config push targets all regions simultaneously via a global pub/sub, but a region-specific outage in receiving that push is itself alerted on (config-propagation-lag monitor) |

## 42. Scaling Bottlenecks

**At 10x (2.4M QPS platform-wide):**
- Feature Store online read layer becomes the first bottleneck (1.44M reads/s at 10x) — requires more aggressive sharding (beyond 4096 slots) and likely a move toward co-locating feature-store read replicas physically nearer serving nodes (rack-locality) to shave network hops.
- Batching layer's fixed windows start under-serving very-high-QPS models — need per-model adaptive batching windows that shrink automatically as arrival rate increases (natural batch fill happens faster, don't need the full window).
- Registry DB config-push fan-out to routers (currently trivial at hundreds of router replicas) starts needing a proper hierarchical pub/sub (regional relay nodes) rather than flat fan-out.

**At 100x (24M QPS):**
- GPU capacity itself becomes the binding constraint — 900+ GPU replicas at that scale starts hitting real capacity/quota ceilings with cloud providers regionally; would require multi-cloud GPU sourcing or heavier investment in distillation/quantization to shrink per-inference compute.
- Kafka topic partition counts (64 currently) become insufficient for consumer parallelism on the prediction-log topic — needs repartitioning (operationally nontrivial, requires careful migration) or a move to a log system with easier elastic partitioning.
- Single global Model Registry primary becomes a write-bottleneck if deployment frequency also scales with platform growth (more models × more teams × more frequent deploys) — would need to shard the registry itself by model-owner-team or move to a multi-primary/CRDT-based design.

## 43. Latency Bottlenecks

**p50 budget breakdown (~15ms target):**

| Stage | p50 time | Notes |
|---|---|---|
| Gateway (TLS, authn, routing) | 0.5 ms | Amortized via connection reuse, cached authn |
| Router model resolution | 0.3 ms | Local cache hit for alias→version |
| Feature store fetch | 2.5 ms | Network hop + KV lookup; dominant non-compute cost |
| Batching wait | 4-8 ms | Configurable dial — biggest tunable lever |
| Model forward pass (GPU) | 3-5 ms | Depends on model size/batch size |
| Response assembly + fire-and-forget log | 0.3 ms | Negligible, async publish |
| **Total** | **~11-16.5 ms** | Matches stated p50 target |

**p99 budget breakdown (~60ms target) — where the tail comes from:**
- Feature store p99 (vs p50 2.5ms) can spike to 15-20ms under shard hot-spotting (a popular player_id pattern, or a resharding event mid-flight).
- Batching layer worst-case wait is the full configured window (8ms) plus queueing delay if the GPU pool itself is momentarily saturated (scale-up lag) — can add another 10-15ms.
- GPU cold path (model not yet warm on an autoscaled-in replica, or model evicted from a multi-model pool under memory pressure) — model reload can cost tens of ms to seconds; this is the single largest tail-latency risk, mitigated by the warm-pool floor (Section 14/28).
- Cross-AZ network jitter within a region contributes a few ms of tail variance.

**Biggest single lever:** the batching window is the most directly *controllable* latency-vs-throughput dial in this whole system — tightening it recovers latency budget at direct GPU-cost expense, and vice versa.

## 44. Cost Bottlenecks

- **GPU fleet is the dominant cost driver** (Section 6: ~$52K/month vs $17.3K/month CPU) despite GPU-bound traffic being only 30% of QPS — deep models are simply far more compute-expensive per inference than GBDTs.
- **Under-utilized long-tail GPU capacity** if bin-packing/multi-model hosting isn't aggressively tuned — the single biggest risk of *cost regression* over time as more low-QPS models onboard without corresponding pool consolidation.
- **Warm-pool floors** (never-scale-to-zero for champion models × 3 regions × N+1 redundancy) represent a fixed cost floor independent of actual traffic — worth periodically auditing which models truly need "champion" always-warm treatment vs. can tolerate a lower floor.
- **Prediction log storage at 246 TB/30-days hot** — the log/storage tiering aggressiveness (Section 29) directly trades query convenience for drift jobs against storage spend; a common latent cost creep if hot-retention windows aren't actively enforced.
- **Cross-region data egress** for any accidentally-introduced cross-region calls (a bug or a misconfigured fallback path calling a neighboring region) — architected against (Section 31) but a classic silent-bill-creep failure mode if a code change violates the region-locality invariant.

## 45. Interview Follow-Up Questions

1. How would you change this design if the p99 latency budget were 10ms instead of 60ms?
2. Walk me through exactly what happens if the Feature Store's Redis cluster fails over mid-request — what does the caller actually see?
3. How do you decide which models get dedicated GPU capacity vs. share a multi-model pool?
4. Your drift detector fires a PSI alert on a low-traffic model at 3am — should that page anyone? Why or why not?
5. How would you extend this platform to support a genuinely new modality — say, real-time voice/audio moderation — without a full redesign?
6. What's your strategy if two different model versions need mutually incompatible feature schemas during a canary rollout?
7. How do you prevent the Model Registry from becoming a single point of failure for the entire platform's deploy velocity?
8. If GPU costs suddenly need to drop by 40% next quarter with no traffic reduction, what are your first three levers?
9. How would multi-region active-active change if EA acquired a studio with its own existing inference stack that needed to be merged in?
10. Explain the exact mechanism by which a bad model rollout gets detected and rolled back automatically — what could go wrong with that automation itself?

## 46. Ideal Answers

1. **10ms p99 instead of 60ms:** Eliminate the batching window almost entirely (accept lower GPU utilization, more replicas) or restrict to CPU-only/small-model tier where compute itself is sub-ms; move feature fetch to be pre-fetched/pushed proactively (e.g., feature store streams updates directly into a request-scoped cache ahead of time via session-start hydration) rather than fetched synchronously per request; likely requires co-locating feature store and model server on the same node/rack to cut network hops; accept meaningfully higher GPU-hour cost as the direct tradeoff (Section 43's batching-window lever pushed to near-zero).

2. **Redis failover mid-request:** Router's feature-fetch call times out against its configured budget (small, e.g. 5-8ms) → circuit breaker (if recent failure rate is high) or per-request timeout triggers fallback path → Router checks in-process/Redis-backed last-known-good cache for that entity+model → if hit, returns cached prediction tagged `degraded:true`; if miss, returns static default, still `degraded:true`. Caller's request is never blocked waiting for Redis's actual failover to complete (seconds), it fails fast into the fallback path within single-digit ms.

3. **Dedicated vs. shared GPU capacity:** Primarily a function of (a) QPS — models above a throughput threshold (e.g., top-20 by traffic, ~80% of total QPS) get dedicated/reserved replica pools to guarantee isolation and predictable autoscaling behavior; (b) latency sensitivity/business criticality — even a lower-QPS model gets dedicated capacity if it's high-stakes (fraud/anti-cheat) and can't tolerate noisy-neighbor variance; (c) memory footprint — very large models that would dominate a shared pool's memory budget get dedicated placement regardless of QPS. Everything else defaults to shared multi-model pools, bin-packed by expected QPS × memory footprint.

4. **3am drift alert on low-traffic model:** No, shouldn't page — route to a non-urgent notification (Slack/ticket to model-owner team) reviewed next business day, because (a) low-traffic models have inherently noisier drift statistics (small sample size → PSI naturally more volatile) and (b) the blast radius of a stale low-traffic model is small. Reserve paging for top-20-by-traffic models or any model in a "high-stakes" tier (fraud/anti-cheat/safety) regardless of traffic — severity should be a function of both statistical confidence and business blast radius, not just threshold breach.

5. **Adding real-time voice/audio moderation:** The core router/batching/multi-model-serving/autoscaling infra is modality-agnostic and reusable as-is; what's new is (a) a much larger payload size (audio chunks vs. small feature vectors) — likely needs streaming/chunked gRPC rather than single-request/response, and possibly a dedicated ingest path bypassing the generic feature-store-fetch step entirely since audio is the direct input; (b) GPU sizing assumptions change (audio models often larger/more compute-heavy per inference than tabular models) — capacity estimation (Section 6) needs redoing for this modality specifically, not extrapolated from existing numbers; (c) likely needs its own hardware_class tier and possibly dedicated node pools given different memory/compute profile. Core control plane (Registry, canary, rollback, drift) extends unchanged.

6. **Incompatible feature schemas across versions during canary:** The Router's model-input-assembly step must be version-aware — resolve feature_set requirements per specific model_version (not per model_name), meaning the feature store lookup step also needs to know which feature_set_ref to request based on the resolved version, and the registry must track feature-schema dependencies per version explicitly. During the canary window, both schemas' required features must be simultaneously computable/available online (a coordination requirement with the upstream feature pipeline team — the new schema's features must already be backfilled/live in the online store *before* the model version enters canary, not simultaneously).

7. **Registry as a deploy-velocity bottleneck:** Keep the Registry itself simple/low-write-volume (metadata + pointers only, never model weights or high-frequency data) so its own scaling ceiling stays far above realistic deploy frequency; make it read-replica-heavy since reads (routers checking config) vastly outnumber writes (deploys); ensure routers cache config locally and only need the Registry for periodic refresh/push-based updates, not per-request lookups — this decouples request-path availability from Registry availability entirely, so even a full Registry outage only blocks *new deployments*, not existing traffic serving.

8. **Cut GPU cost 40% with no traffic reduction:** First lever — audit and tighten multi-model bin-packing (likely the single biggest low-effort win if any dedicated-but-low-QPS models exist that could move to shared pools). Second — push distillation/quantization on the largest few models (biggest GPU-hour consumers by model, not by count, given the power-law — a handful of high-QPS deep models likely account for a disproportionate share of GPU-hours). Third — increase spot-instance mix for anything not in the "champion always-on-demand" tier, and re-tune batching windows toward higher throughput wherever there's unused latency budget headroom (Section 43 tradeoff pushed the other direction from Q1).

9. **Acquired studio with existing inference stack:** Treat it as a migration, not a merge-in-place — run the acquired stack as-is initially (avoid forced-rewrite risk to a live game), onboard it to the shared Model Registry as a control-plane-only integration first (visibility/governance without touching their serving path), then incrementally migrate individual models onto the shared Triton/K8s serving layer model-by-model via the normal canary process, prioritized by which of their models would benefit most from shared multi-model bin-packing (their highest-cost, lowest-traffic long-tail models first) — avoid a single big-bang cutover given it's a live-service game with real players.

10. **Automated rollback mechanism and its own failure modes:** Mechanism: health-check gate (latency/error/distribution) evaluated continuously during canary ramp and for a post-100%-cutover soak window → breach triggers Deployment Orchestrator to zero-out the new version's traffic weight via the same config-push path used for normal traffic-split changes → routers pick up the reverted config within the standard propagation window (seconds). What could go wrong with the automation itself: (a) the automated gate's own metrics pipeline could be lagging/broken, causing either false-negative (bad version never caught) or false-positive (good version wrongly rolled back) — mitigated by monitoring the *health-check pipeline's own freshness* as a meta-metric; (b) rollback config-push itself could fail to propagate to one region (the exact failure mode in Section 41's cross-region propagation-lag scenario) — mitigated by an explicit propagation-confirmation step before considering rollback "complete," with escalation if any region doesn't ack within an SLA.
