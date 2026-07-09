# Model Registry

## 1. Problem Framing & Requirement Gathering

EA ships ML models across dozens of live-service titles (FC, Apex Legends, Battlefield, The Sims) plus platform-wide systems (matchmaking, anti-cheat, churn prediction, toxicity detection, live-ops LiveOps offer targeting). Each title's data science team trains models independently, but promotion to production today is ad-hoc: model files are dropped in S3 buckets with inconsistent naming, lineage to training data/code is tribal knowledge, and rollback during a live incident (e.g., a matchmaking model regressing skill-based matchmaking quality mid-tournament) can take hours because nobody can definitively answer "what was running yesterday, and what changed."

A **Model Registry** is the system of record for:
- Every trained model version, its lineage (training data snapshot, code commit, hyperparameters, environment).
- Approval/promotion workflow (dev → staging → canary → production) with auditable sign-off.
- Physical artifact storage (weights, tokenizers, preprocessing pipelines, ONNX/TorchScript exports).
- Integration points for CI/CD (model build pipelines) and CD (serving deployment triggers).

This is an **internal platform system**, not a player-facing system — but its availability directly gates whether any ML-driven feature (matchmaking, recommendations, anti-cheat scoring) can be updated safely, and its correctness gates whether a bad model reaches 30M+ concurrent players.

## 2. Functional Requirements

- FR1: Register a new model version with artifact(s), metadata, and lineage (dataset ID, code commit SHA, training job ID, hyperparameters, metrics).
- FR2: Immutable versioning per model name (`matchmaking-skill-model:v47`), semantic tagging (`@staging`, `@production`, `@canary`).
- FR3: Stage transition workflow: `None → Staging → Canary → Production → Archived`, each transition gated by required approvals (automated eval gates + human sign-off for Production).
- FR4: Full lineage graph query: given a production incident, trace model version → training data snapshot → feature definitions → code commit → training job logs.
- FR5: Artifact storage supporting multiple formats (PyTorch `.pt`, ONNX, TensorRT engine, TorchScript, tokenizer configs, feature transformers/pickled sklearn objects).
- FR6: CI/CD integration: registry emits webhooks/events on stage transitions to trigger serving deployment pipelines; CI pipelines push new versions on training job completion.
- FR7: Model comparison: diff metrics (offline eval metrics, fairness metrics, latency benchmarks) between any two versions.
- FR8: Rollback: one-call API to re-promote a prior version to `@production`, with automatic notification to serving layer.
- FR9: Access control: per-title, per-model RBAC (a Battlefield DS engineer cannot promote a FC matchmaking model to production).
- FR10: Search/discovery: find all models by title, task type (classification/ranking/regression), framework, owner team.
- FR11: Deprecation & retention policy enforcement (auto-archive models unused for N days, subject to compliance holds).
- FR12: Audit log: immutable record of every registration, transition, approval, and deletion, with actor identity.

## 3. Non-Functional Requirements

| Dimension | Target | Rationale |
|---|---|---|
| Latency (metadata read, e.g. "get production version") | p99 < 50 ms | Serving layer may query registry at deploy-time/cold-start, not per-inference |
| Latency (artifact download, e.g. 2 GB model) | < 30 s from co-located blob store | Pod cold-start / autoscale event must not stall on artifact fetch |
| Availability (metadata/control plane) | 99.95% (≈4.4 hrs/yr downtime budget) | Gates deploys; brief outage tolerable, but not during live incident response |
| Availability (artifact storage read path) | 99.99% | Directly blocks pod autoscaling/serving during peak traffic (e.g., FC Ultimate Team drop event) |
| Throughput (registrations) | 500-2,000 model versions/day across EA (peak during major title launch weeks) | ~50 active ML teams, frequent retraining, hyperparameter sweeps registering many candidates |
| Throughput (metadata reads) | 5,000 QPS peak (CI/CD polling, dashboards, serving lookups) | Many consumers polling registry state |
| Consistency | Strong consistency on stage-transition state (no two readers see different "current production version") | Split-brain on "what's in prod" is a live-incident risk |
| Consistency (artifact store) | Read-after-write strong consistency required for freshly registered artifacts | CI/CD immediately fetches after registration |
| Durability | 11 nines (S3-standard equivalent), cross-region replicated | Models are often irreproducible (data since deleted, compute cost to retrain large models) |
| Cost | Storage cost amortized; target < $0.02 per model-version-GB-month effective | Thousands of versions × multi-GB checkpoints accumulate fast |

## 4. Clarifying Questions an Interviewer Would Expect

1. Is this registry shared across all EA titles/studios, or per-studio with a federated view? (Assume shared platform, per-title namespaces.)
2. Do we need to support non-DL classical ML artifacts (XGBoost, sklearn) alongside deep learning checkpoints? (Yes — matchmaking uses gradient boosted trees; recommendation uses deep two-tower models.)
3. Is approval a human-in-the-loop gate, or can it be fully automated based on offline eval thresholds?
4. What's the largest single artifact size we need to support? (LLM fine-tunes for narrative/dialogue systems can be 15-70GB; most gameplay models are 10MB-2GB.)
5. Do we need bit-for-bit reproducibility (exact retraining reproducibility) or just lineage traceability?
6. Is the registry also the feature-schema source of truth, or does it just reference an external Feature Store schema version?
7. What compliance/regulatory retention requirements apply (COPPA for younger player titles, GDPR right-to-erasure interactions with training data lineage)?
8. Do canary/shadow deployments get their own registry entries, or are they ephemeral references to a Staging version?
9. Multi-region: are training and serving co-located per region, or centralized training with global serving fan-out?
10. What's acceptable RPO/RTO if the registry's metadata store is lost — can we always reconstruct from CI/CD pipeline logs, or is the registry the sole source of truth?

## 5. Assumptions

1. ~50 active ML teams across ~20 titles/platform-orgs, each maintaining 5-30 distinct model "families."
2. Average model artifact size: 500 MB (ranges 5 MB classical models to 40 GB LLM fine-tunes).
3. ~1,500 new model version registrations/day across EA at steady state, spiking to 4,000/day during major title launch crunch (e.g., FC annual release, new Apex season).
4. Metadata store must serve as system-of-record independent of any single title's infra — platform team owns it centrally.
5. Artifact storage backed by cloud blob storage (S3-compatible), metadata in a relational store for transactional guarantees on stage transitions.
6. CI/CD is Jenkins/GitHub Actions/Argo Workflows-based per title, integrating via REST + event webhooks — registry does not replace CI/CD, it's a dependency/gate within it.
7. Serving layer (see companion "Model Serving" chapter) polls or subscribes to registry for "current production pointer" per model name.
8. Human approval required for Production promotion on player-facing/high-blast-radius models (matchmaking, anti-cheat, monetization); auto-promotion permitted for internal-only models (analytics scoring) that pass eval gates.
9. Global player base ~500M registered accounts, ~30M peak concurrent across EA's live titles, but registry load is driven by ML team activity, not player count directly.
10. On-prem GPU clusters + cloud burst capacity for training; registry itself is CPU-only control-plane infra.

## 6. Capacity Estimation

**Model version volume:**
- Steady state: 1,500 registrations/day → ~0.017 registrations/sec average; peak (launch week, parallel hyperparameter sweeps): 4,000/day ≈ 0.05/sec sustained, bursty to ~5/sec during sweep completion windows (100 trials finishing within a 20 min window).
- Annual accumulation: 1,500/day × 365 ≈ 548,000 new versions/year across EA. Over 3 years (before aggressive archival kicks in): ~1.6M model version records.

**Storage:**
- Average artifact 500 MB × 548,000/year ≈ 274 TB/year raw.
- With retention policy (auto-archive to cold storage after 90 days unused, delete after 2 years except production/compliance-held versions): steady-state hot storage ≈ 274 TB × (90/365) ≈ 67 TB hot at any time; cold/archive tier holds the multi-year backlog (~500-800 TB in Glacier-class storage after 3 years).
- Metadata row size: ~2 KB/version (JSON lineage blob + indexed columns) × 1.6M rows ≈ 3.2 GB — trivially fits in a normal RDBMS, no sharding needed for metadata alone.
- Audit log: ~5 events/version-lifecycle (register, staging, canary, prod, archive) × 1.6M versions × 1 KB/event ≈ 8 GB/3yr — negligible.

**Throughput:**
- Metadata reads: 5,000 QPS peak — dominated by CI/CD polling (each of ~50 teams' pipelines polling every few seconds during active deploys) + serving-layer "get current production pointer" checks at pod startup (autoscale events: e.g., FC launch day scaling matchmaking service from 200 to 2,000 pods in 10 min → 1,800 registry lookups in 600s ≈ 3 QPS from that alone, small relative to polling load).
- Artifact reads: at pod cold-start, each new serving pod fetches the model artifact once. At peak autoscale (2,000 pods spinning up over 10 min, 500MB avg artifact) = 2,000 × 500MB = 1 TB pulled in 600s ≈ 1.7 GB/s aggregate egress from artifact store — requires CDN/regional caching, not a single origin fetch path.
- Writes (registrations): low absolute QPS (<1 QPS average) but each write is a multi-artifact multipart upload (500 MB average, up to 40 GB for LLM checkpoints) — dominates bandwidth in, not request rate.

**Compute (control plane, not training):**
- Registry API service: stateless, CPU-only. 5,000 QPS at ~5ms p50 compute per request ⇒ ~25 vCPU-seconds/sec of work ⇒ roughly 8-12 modest (4 vCPU) instances behind a load balancer with headroom, autoscaled 2x for launch weeks.
- No GPU required for the registry control plane itself — GPUs are consumed upstream by training jobs and downstream by serving, out of scope for this chapter.

## 7. High-Level Architecture

```
                                   ┌─────────────────────────┐
                                   │   Auth/IAM (SSO, RBAC)  │
                                   └────────────┬─────────────┘
                                                │
 ┌──────────────┐     register/promote    ┌────▼─────────────────────┐        ┌───────────────────┐
 │ Training Job  │ ───────────────────────▶│   Model Registry API      │◀──────▶│  Web UI / Dashboard│
 │ (CI Pipeline) │                          │  (stateless, REST/gRPC)  │        └───────────────────┘
 └──────┬────────┘                          └────┬───────────┬────────┘
        │ upload artifact                        │           │  reads/writes
        │                                         │           │
        ▼                                         ▼           ▼
 ┌───────────────┐                     ┌────────────────┐  ┌──────────────────────┐
 │ Object Storage │                    │ Metadata Store  │  │  Audit Log Store      │
 │ (S3-compatible,│◀──── presigned ────│ (Postgres,      │  │ (append-only,         │
 │ versioned,     │      URLs          │  strongly       │  │  WORM, e.g. S3 +      │
 │ multi-region)  │                    │  consistent)    │  │  Object Lock)         │
 └──────┬─────────┘                    └────────┬────────┘  └──────────────────────┘
        │  CDN/regional cache                    │ emits
        ▼                                        ▼
 ┌───────────────┐                     ┌────────────────────┐
 │ Serving Fleet  │◀──── poll/webhook ─│  Event Bus (Kafka)  │──▶ Notification (Slack/PagerDuty)
 │ (per-title)    │   "prod pointer   │  model.promoted,     │
 └───────────────┘    changed" event  │  model.registered,   │──▶ CI/CD Trigger (deploy canary)
                                       │  model.archived      │
                                       └──────────┬───────────┘
                                                  │
                                                  ▼
                                       ┌────────────────────┐
                                       │ Eval/Gate Service   │
                                       │ (offline metrics,   │
                                       │  fairness, latency  │
                                       │  benchmark runner)  │
                                       └────────────────────┘
```

Data flow: training job completes → uploads artifact directly to object storage (presigned URL) → registers metadata via API → API writes transactional row in metadata store + emits `model.registered` event → Eval/Gate Service consumes event, runs automated evaluation, posts result back to API → if gates pass and (for prod) human approves via UI → stage transition write → `model.promoted` event → serving fleet's watchers/CI-CD consume event → canary or full rollout triggered.

## 8. Low-Level Components

**Registry API Service**
- Responsibility: CRUD on model/version/stage-transition entities; enforces RBAC and stage-gate rules; issues presigned upload/download URLs (never proxies large artifact bytes itself).
- Interface: REST + gRPC (gRPC for high-frequency internal CI/CD and serving-layer polling; REST for UI/dashboard/external tooling).
- Scaling unit: stateless pods behind an L7 load balancer, horizontal autoscale on CPU/QPS; no local state.

**Metadata Store (Postgres, primary)**
- Responsibility: transactional source of truth for model, version, stage-pointer, approval records.
- Interface: internal only, accessed via Registry API service (never directly by clients).
- Scaling unit: primary + read replicas per region; vertical scale sufficient given ~3GB total data size — read replicas exist for read-latency/availability, not for data-volume sharding.

**Object Storage (Artifact Store)**
- Responsibility: durable, versioned, immutable storage of model binaries and associated files (tokenizers, transformers, configs) as a content-addressed bundle per version.
- Interface: S3 API; clients get short-lived presigned URLs from Registry API, upload/download directly.
- Scaling unit: effectively infinite (managed object store); regional replication buckets fanned out via async cross-region replication + CDN edge caching for hot (production/canary) versions.

**Event Bus (Kafka)**
- Responsibility: decouple registry writes from downstream consumers (CI/CD triggers, serving-layer watchers, notification systems, audit pipeline).
- Interface: topics `model.registered`, `model.stage_changed`, `model.archived`, `model.eval_completed`.
- Scaling unit: partitioned by `model_name` key to preserve per-model ordering; consumer groups per downstream system.

**Eval/Gate Service**
- Responsibility: runs automated offline evaluation (accuracy/AUC/NDCG thresholds vs. baseline, fairness slice checks, latency/throughput micro-benchmark on the exported artifact) before allowing Staging→Canary and Canary→Production transitions.
- Interface: consumes `model.registered`/`model.stage_transition_requested` events, calls back into Registry API with pass/fail + metrics payload.
- Scaling unit: job-queue-backed worker pool (Kubernetes Jobs), scales with registration volume; GPU-backed workers for latency-benchmarking DL models.

**Audit Log Store**
- Responsibility: immutable, tamper-evident record of every state change and who/what triggered it (human user, service account, automated gate).
- Interface: write-only append via Registry API internal hook; read via dedicated compliance/audit query API (separate from hot path).
- Scaling unit: write-once object storage with Object Lock/WORM; scales trivially given low event volume.

**Lineage Graph Index**
- Responsibility: fast traversal queries ("show me everything that fed into production model X") across model version → dataset snapshot → feature schema version → training job → code commit.
- Interface: graph query API (could be a lightweight graph DB like Neo4j, or a materialized adjacency table in Postgres for moderate scale — given our ~1.6M-node scale, a well-indexed relational adjacency table is sufficient and avoids operating a second datastore).
- Scaling unit: read replica of metadata store; recomputed/materialized incrementally on write.

## 9. API Design

Base path: `/api/v1/registry` (gRPC service `ModelRegistryService` mirrors same operations).

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/models` | Register a new model family (name, owning team, task type) |
| POST | `/models/{model_name}/versions` | Register a new version (metadata + presigned upload handshake) |
| GET | `/models/{model_name}/versions/{version}` | Fetch version metadata + lineage |
| GET | `/models/{model_name}/versions?stage=production` | List versions filtered by stage/tag |
| POST | `/models/{model_name}/versions/{version}/transition` | Request stage transition (triggers gate eval / approval flow) |
| POST | `/models/{model_name}/versions/{version}/approve` | Human approval action (requires RBAC role) |
| POST | `/models/{model_name}/versions/{version}/rollback` | Re-promote a prior version to production, atomically demoting current |
| GET | `/models/{model_name}/lineage/{version}` | Full lineage graph (dataset, code, job, features) |
| GET | `/models/{model_name}/diff?a={v1}&b={v2}` | Metric/config diff between two versions |
| DELETE | `/models/{model_name}/versions/{version}` | Soft-delete/archive (subject to compliance hold check) |
| GET | `/audit/{model_name}` | Audit trail for a model family |

Example — register version request:
```json
POST /api/v1/registry/models/matchmaking-skill-model/versions
{
  "training_job_id": "job-2026-07-08-a91f",
  "code_commit_sha": "3f9a1c2",
  "dataset_snapshot_id": "ds-fc-ranked-2026-06-30",
  "feature_schema_version": "feat-schema-v12",
  "hyperparameters": {"lr": 0.001, "depth": 8, "n_estimators": 400},
  "framework": "xgboost==2.0.3",
  "artifact_manifest": [
    {"name": "model.xgb", "size_bytes": 42000000, "sha256": "..."},
    {"name": "feature_transformer.pkl", "size_bytes": 120000, "sha256": "..."}
  ],
  "offline_metrics": {"auc": 0.812, "ndcg@10": 0.734}
}
```
Response includes `version_id`, `presigned_upload_urls[]` (one per manifest entry), `status: "PENDING_UPLOAD"`.

Versioning: URI-path versioned API (`/v1/`); model version identifiers are immutable monotonic integers per model name (`v1, v2, ...`), never reused. Stage tags (`@staging`, `@production`, `@canary`) are mutable pointers, distinct from immutable version numbers — this separation is critical so "production" can move without renaming/duplicating artifacts.

## 10. Database Design

**Choice: Postgres (relational) for metadata + transactional stage pointers; S3-compatible object store for binaries; append-only object storage (Object Lock) for audit log.**

Rationale: stage-transition correctness requires ACID transactions (the "current production version" pointer must never be ambiguous under concurrent writes) — a relational store with row-level locking is the right fit given only ~3-5 GB of metadata at 3-year scale; no need for a distributed NoSQL system that trades consistency for scale we don't need here. Object storage is the natural fit for large immutable binary blobs — never store blobs in the relational DB.

Core schema sketch:
```sql
CREATE TABLE models (
  model_name        TEXT PRIMARY KEY,
  owning_team       TEXT NOT NULL,
  task_type         TEXT NOT NULL,      -- classification|ranking|regression|generative
  created_at        TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE model_versions (
  model_name         TEXT REFERENCES models(model_name),
  version            INT NOT NULL,
  training_job_id    TEXT,
  code_commit_sha    TEXT,
  dataset_snapshot_id TEXT,
  feature_schema_version TEXT,
  hyperparameters    JSONB,
  offline_metrics    JSONB,
  framework          TEXT,
  artifact_uri_prefix TEXT NOT NULL,    -- s3://bucket/model_name/v47/
  created_by         TEXT NOT NULL,
  created_at         TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (model_name, version)
);

CREATE TABLE stage_pointers (
  model_name   TEXT REFERENCES models(model_name),
  stage        TEXT NOT NULL,           -- staging|canary|production
  version      INT NOT NULL,
  updated_by   TEXT NOT NULL,
  updated_at   TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (model_name, stage)       -- one current pointer per stage per model
);

CREATE TABLE approvals (
  id            BIGSERIAL PRIMARY KEY,
  model_name    TEXT,
  version       INT,
  target_stage  TEXT,
  approver      TEXT,
  gate_results  JSONB,                  -- automated eval outcomes
  approved_at   TIMESTAMPTZ DEFAULT now()
);
```

Partitioning/sharding: not required at metadata scale (~1.6M rows/3yr). If EA-wide scale grows 10x (16M rows) or multi-tenant isolation becomes a hard requirement, partition `model_versions` and `approvals` by `model_name` hash or by studio/title namespace — Postgres native declarative partitioning suffices; no need for a distributed SQL layer (CockroachDB/Spanner) until write QPS to the transactional tables exceeds a single primary's capacity, which our estimates (< 1 write QPS average) don't approach.

Audit table is intentionally append-only and stored separately (object storage with Object Lock, or a separate Postgres table with revoked UPDATE/DELETE grants) to guarantee tamper-evidence independent of the mutable operational tables.

## 11. Caching

- **What's cached:** (a) "current production/canary pointer per model" — hottest read path, queried by every serving pod at startup and by CI/CD polling; (b) model metadata blobs (immutable once written, perfect cache candidates); (c) artifact bytes at CDN edge for production/canary-tagged versions only (not every historical version).
- **Cache-aside** for stage pointers: Registry API reads through Redis first, falls back to Postgres on miss, populates cache with short TTL (5s) — short TTL bounds staleness window for a value that can change (unlike immutable version metadata) while still absorbing the bulk of read QPS (5,000 QPS reduced to a trickle hitting Postgres).
- **Write-through + explicit invalidation** on stage transition: the transition transaction that updates `stage_pointers` in Postgres synchronously also invalidates (deletes) the Redis key for that `(model_name, stage)` pair before returning success, and republishes the fresh value — guarantees no reader can observe a stale pointer past the transaction boundary, combined with the 5s TTL as a safety net if invalidation delivery fails.
- **Immutable metadata caching:** `model_versions` rows never change after creation — cache with no TTL (or very long TTL, e.g. 24h) keyed by `(model_name, version)`, safe because immutability means no invalidation logic needed at all.
- **CDN for artifacts:** production/canary-tagged artifact objects are pushed to a CDN distribution (or regional edge cache) proactively on promotion (pre-warm), rather than cold on first pod request — avoids the "2,000 pods cold-starting simultaneously, all missing CDN cache, all hitting origin" thundering-herd on launch-day autoscale events.

## 12. Queues & Async Processing

- **What's queued:** (a) artifact upload-completion → trigger eval job; (b) stage-transition-requested → trigger gate evaluation workflow; (c) promotion → trigger downstream CI/CD deploy webhook fan-out; (d) archival sweep jobs (nightly batch scan for retention-policy violations).
- **Delivery semantics:** at-least-once for all registry events (Kafka default), consumers must be idempotent — e.g., the Eval/Gate Service keys its evaluation-run record by `(model_name, version, gate_type)` with an upsert, so a duplicate delivery re-running an eval is a no-op if already completed, not a duplicate side effect.
- **Exactly-once is not attempted** at the messaging layer (unnecessary complexity/cost) — idempotency at the consumer is cheaper and sufficient given the "gate evaluation" and "trigger deploy" operations are naturally idempotent when keyed correctly.
- **Dead-letter handling:** after 5 retries with exponential backoff, failed event processing (e.g., Eval/Gate Service worker crash mid-run) routes to a `model-registry-dlq` topic; a dedicated on-call dashboard surfaces DLQ depth > 0 as a paging alert (a stuck promotion gate blocks a live incident rollback, so DLQ backlog on `model.stage_transition_requested` is high severity).
- CI/CD deploy-trigger webhooks specifically: fire-and-forget publish to event bus, but the *consuming* CI/CD system (Jenkins/Argo) is responsible for its own retry/backoff on webhook receipt failures — registry does not track downstream deploy success/failure, only that it emitted the trigger (deploy pipeline status is out of scope, owned by the Serving/CI-CD chapter).

## 13. Streaming & Event-Driven Architecture

Kafka topics (all keyed by `model_name` for per-model ordering guarantees):

| Topic | Schema (Avro/Protobuf) | Producers | Consumer Groups |
|---|---|---|---|
| `model.registered` | `{model_name, version, created_by, artifact_manifest[], timestamp}` | Registry API | Eval/Gate Service, Audit Pipeline, Search Indexer |
| `model.stage_transition_requested` | `{model_name, version, from_stage, to_stage, requested_by, timestamp}` | Registry API | Eval/Gate Service, Notification Service |
| `model.stage_changed` | `{model_name, version, stage, previous_version, updated_by, timestamp}` | Registry API | Serving Fleet Watchers, CI/CD Trigger, Notification Service, Audit Pipeline |
| `model.eval_completed` | `{model_name, version, gate_type, passed, metrics{}, timestamp}` | Eval/Gate Service | Registry API (writes approval record), Dashboard |
| `model.archived` | `{model_name, version, reason, timestamp}` | Registry API (retention sweep) | Storage Lifecycle Manager, Audit Pipeline |

Consumer group design: each downstream system (serving fleet watcher, CI/CD trigger, notification) is its own consumer group so all groups independently receive every event — no competing-consumer semantics across different systems, only within a system's own horizontally-scaled workers (e.g., multiple Eval/Gate Service pods share one consumer group, partitioned work by `model_name` key).

Schema evolution: Avro with a schema registry (Confluent-style), backward-compatible evolution only (new optional fields) — a title's CI/CD consumer written a year ago must not break when the platform team adds a new field to `model.stage_changed`.

## 14. Model Serving

Out of primary scope for this chapter (see companion "Model Serving" chapter) but the registry's serving-facing contract:

- Serving framework choice is per-title (Triton Inference Server for GPU-heavy DL models like recommendation/dialogue; lightweight custom Python/Go services for classical models like XGBoost matchmaking scoring) — the registry is framework-agnostic, storing whatever artifact format the serving layer expects (ONNX, TorchScript, Triton model-repo layout, raw XGBoost booster file).
- Registry's role at serving time: serving fleet's deployment controller subscribes to `model.stage_changed` for `stage=production` (or `canary`), fetches the new artifact via presigned URL / CDN, and performs its own rollout (rolling update or canary traffic-split) — the registry does not itself perform inference or manage the serving fleet's pods, it is purely the artifact/metadata authority the serving CD pipeline reacts to.
- Batching, multi-model colocation, hardware selection (GPU vs CPU) are serving-layer concerns configured per model in a companion `serving_config` metadata field the registry stores but doesn't interpret (e.g., `{"batch_size": 32, "hardware": "T4", "max_replicas": 200}` — passed through to the serving orchestrator).

## 15. Feature Store

The registry does not host feature data — it references a **Feature Store's** schema version (`feature_schema_version` field in `model_versions`) for lineage purposes, treating the Feature Store as an external system of record for online (low-latency KV, e.g. Redis/DynamoDB-backed) and offline (batch, e.g. Parquet on data lake, or a warehouse table) feature data.

- **Point-in-time correctness:** the registry stores the exact `dataset_snapshot_id` used at training time — this is the mechanism that lets us later ask "did this model train on features computed with a point-in-time-correct join, or did it leak future data?" The registry doesn't enforce PIT correctness itself (that's the Feature Store/training-pipeline's job) but it is the durable record that makes an incident post-mortem ("was there label leakage in v47?") answerable months later.
- Online/offline split and ANN/embedding-serving concerns belong to the Feature Store chapter; this registry's only touchpoint is storing an immutable pointer to the schema version, so a served model's expected feature contract is always traceable.

## 16. Vector Database

**N/A for this system.** The Model Registry itself has no similarity-search or nearest-neighbor retrieval workload — it does traditional keyed lookups (by model name + version) and small-scale graph traversal (lineage), both well served by relational indexes. Vector databases are relevant to *embedding-based retrieval systems* (e.g., a recommendation system's candidate generation) which are separate systems that may themselves be *entries in* this registry (an embedding model is just another model version with its own lineage record) — but the registry does not need to operate a vector index to fulfill its own responsibilities.

## 17. Embedding Pipelines

**N/A as a first-class registry responsibility**, with a caveat: embedding models (e.g., a player-behavior embedding model feeding downstream churn/LTV models) are registered exactly like any other model — same version/lineage/promotion workflow. The registry does not run embedding pipelines itself; it is a consumer of their outputs only insofar as an embedding model's artifact and metadata get registered like any other trained model. Embedding *generation/backfill pipelines* are training/inference pipeline concerns (Sections 18-19), not registry-specific.

## 18. Inference Pipelines

Inference (serving-time request lifecycle) is largely out of scope for the registry itself, but the registry sits at the *start* of every inference pipeline's deployment lineage. End-to-end lifecycle showing where the registry participates:

```
Player action (e.g., queues for ranked match)
     │
     ▼
Game client ──▶ Matchmaking Service (serving fleet)
                     │
                     │  [at pod startup / periodic refresh]
                     ▼
              ┌─────────────────────────────┐
              │ Registry: GET /models/       │
              │ matchmaking-skill-model/      │
              │ versions?stage=production     │  ◀── cached (Section 11), p99 < 50ms
              └──────────┬────────────────────┘
                         │ returns version=47, artifact_uri
                         ▼
              Pod fetches artifact (CDN-cached) if not already loaded
                         │
                         ▼
              Loaded model scores live match candidates
                         │
                         ▼
              Response: matched lobby ──▶ Game client
```

The registry's contribution to inference latency is isolated to **cold-start/refresh time**, never per-request — a well-designed serving layer loads the model into memory once and serves thousands of QPS of actual inference without touching the registry again until the next poll/refresh interval (typically 30-60s) or push notification of a new production version.

## 19. Training Pipelines

Also largely a companion-chapter concern, but the registry defines the **contract** training pipelines must fulfill to register a version:
- Data prep: training pipeline reads from a versioned/snapshotted feature dataset (Feature Store offline store or data-lake Parquet snapshot) — the `dataset_snapshot_id` is a required field at registration, not optional, to prevent "we trained on a moving target" lineage gaps.
- Training orchestration: title-specific (Kubeflow Pipelines, Argo Workflows, SageMaker Pipelines, or custom Airflow DAGs) — the registry only requires the orchestrator emit a `training_job_id` that is queryable/linkable (e.g., a URL to the Argo Workflow UI) for lineage drill-down.
- Distributed training (relevant for large embedding/LLM fine-tunes, e.g. dialogue systems): multi-GPU/multi-node jobs (PyTorch DDP/FSDP) run on the training cluster, entirely upstream of the registry — registry only records the *final checkpoint* plus training config (world size, epochs, effective batch size) as metadata for reproducibility, not the intermediate distributed-training mechanics.
- Registry enforces (via required fields at the API level) that no version can be registered without: code commit SHA, dataset snapshot ID, and a training job identifier — this is the single biggest lever the registry has over training-pipeline hygiene EA-wide.

## 20. Retraining Strategy

- **Cadence:** varies by model class —
  - Matchmaking/skill models: weekly retrain (ranked season data drifts as meta shifts).
  - Anti-cheat detection models: continuous/daily retrain given adversarial arms-race dynamics (cheaters adapt within days).
  - Recommendation/LiveOps offer models: daily retrain (fresh engagement signal, high-value from freshness).
  - Churn/LTV models: monthly retrain (slower-moving behavioral patterns, higher cost per retrain given larger feature backfill).
- **Triggers (registry's role):** the registry doesn't itself decide *to* retrain (that's the training-orchestration/monitoring system's job, informed by Drift Detection in Section 21) — but it exposes the API (`GET /models/{name}/versions?stage=production` + timestamp) that a scheduler or drift-monitor queries to decide "how stale is what's in prod" and whether a scheduled or emergency retrain is warranted.
- **Emergency retrain trigger:** drift alert (Section 21) or a live incident (matchmaking quality complaint spike) can trigger an out-of-band retrain job outside the normal cadence — registry treats this identically to a scheduled retrain's resulting version (no special-casing in the data model), the *urgency* is handled by the human/automated approval workflow (expedited approval path with senior sign-off for emergency promotions, still audit-logged identically).

## 21. Drift Detection

Not run by the registry itself (owned by a monitoring/observability system consuming production inference logs), but the registry is the join-key that makes drift detection actionable — every drift alert must resolve to "which exact model version, trained on which data" so a retrain can target the actual problem.

- **Data drift:** population stability index (PSI) or KL-divergence on feature distributions (e.g., player skill-rating distribution, match duration) computed daily, compared serving-time feature distributions vs. the training dataset snapshot the current production version cites — threshold: PSI > 0.2 triggers a warning, PSI > 0.3 triggers a retrain-recommendation alert.
- **Concept drift:** tracked via online proxy metrics (e.g., matchmaking model's predicted win-probability calibration vs. actual match outcomes, sampled and aggregated hourly) — a calibration error (Brier score) increase > 15% relative to the value recorded at promotion time (stored in `offline_metrics` at registration) triggers investigation.
- **Registry's specific contribution:** every drift alert payload includes `model_name` + `version`, and the on-call engineer's first action is a registry lineage lookup (Section 9's `/lineage` endpoint) to see exactly what data/code produced the currently-drifting version, cutting incident diagnosis time from "spelunking S3 buckets" to a single API call.

## 22. Monitoring

- **Infra:** Registry API p50/p99 latency, error rate, Postgres connection pool saturation, Kafka consumer lag per topic/consumer-group, object storage request rate/error rate, CDN cache hit ratio for artifact fetches.
- **Model quality (proxied through registry metadata, computed elsewhere):** offline eval metric deltas between consecutive versions (surfaced via the `/diff` endpoint, dashboarded so a promotion decision has an at-a-glance regression check).
- **Business metrics (via registry's role as the audit trail):** promotion frequency per team (velocity metric — are teams shipping model improvements regularly, or stuck), mean time-to-promotion (registration → production, a DevEx health signal), rollback frequency (a proxy for release-quality — rising rollback rate across a title signals a broken eval-gate process upstream).
- **Registry-specific SLO dashboards:** stage-pointer read latency (the hottest path, Section 3's p99<50ms target), presigned-URL issuance latency, DLQ depth (Section 12).

## 23. Alerting

| Condition | Threshold | Severity | Route |
|---|---|---|---|
| Registry API error rate | > 1% over 5 min | P2 | On-call platform eng (Slack + PagerDuty) |
| Stage-pointer read p99 | > 200ms over 5 min | P2 | On-call platform eng |
| Postgres replication lag | > 10s | P3 | Platform DB on-call |
| Kafka consumer lag (`model.stage_changed`) | > 5,000 messages or > 2 min | P1 | On-call platform eng — a stuck deploy-trigger delays production rollouts/rollbacks |
| DLQ depth > 0 on `model.stage_transition_requested` | any | P1 | On-call platform eng — blocks live incident rollback capability |
| Artifact upload failure rate | > 5% over 15 min | P2 | Platform storage on-call |
| Unapproved production promotion attempt (RBAC bypass attempt) | any | P1 security | Security on-call, immediate page |
| Drift alert (external system, registry-tagged) | PSI > 0.3 / calibration drop > 15% | P2, escalates to P1 if player-facing title in a live event window | Title's ML on-call |
| Rollback executed | any (informational, not paging) | P4 | Posted to title's Slack channel automatically |

## 24. Logging

- **Structured logging:** every Registry API request logged as structured JSON (`request_id, actor, model_name, version, action, latency_ms, outcome`), shipped to a central log pipeline (e.g., Fluent Bit → Kafka → data lake / log search like OpenSearch).
- **PII handling:** the registry's own data (model names, versions, hyperparameters, commit SHAs, team names) is not player PII — the risk surface is indirect: `dataset_snapshot_id` references may point to training data that itself contains player telemetry, but the registry stores only the *pointer*, never the underlying player data. Actor identity in logs (which engineer/service account performed an action) is employee-identifying, not player-identifying, and is retained under normal corporate IT data-handling policy, not GDPR player-data policy.
- **Retention:** operational logs 90 days hot (searchable) + 1 year cold archive for incident forensics; audit log (Section 8/10, distinct from operational logs) retained indefinitely (or per-compliance-mandated minimum, e.g. 7 years for titles under regulatory scrutiny) given its role as the tamper-evident record of who-promoted-what.
- **Correlation:** every log line carries a `request_id` and, where applicable, `model_name:version` — enabling trace-through from "a bad model reached production" back through the exact API call, actor, and eval-gate results that allowed it.

## 25. Security

**Threat model specific to this system:**
- **Model/artifact tampering:** an attacker (or compromised CI credential) swaps a benign registered artifact for a malicious one (e.g., a matchmaking model altered to always favor certain accounts — cheating-as-a-service risk, or an anti-cheat model backdoored to whitelist specific players). Mitigation: artifact SHA-256 checksums recorded at registration and verified on every download; artifacts stored with object-store versioning + immutability (Object Lock) so a version's bytes cannot be silently overwritten post-registration; presigned upload URLs scoped single-use and short-TTL.
- **Unauthorized promotion:** a lower-privileged actor forces a bad/malicious model to `@production` bypassing approval gates. Mitigation: RBAC enforced server-side on every transition endpoint (never trust client-side gating), production promotions require a two-person rule for player-facing/high-blast-radius model classes (configurable per model's declared risk tier).
- **Lineage forgery:** an actor registers a version claiming a `dataset_snapshot_id`/`code_commit_sha` it didn't actually use, to make a bad model appear compliant. Mitigation: where feasible, CI pipeline identity (service account, not human) is the only actor permitted to set lineage fields — humans can request transitions but cannot directly author lineage metadata, closing the "hand-edited lineage" gap.
- **Data at rest/in transit encryption:** metadata store encrypted at rest (KMS-managed keys), artifact bucket server-side encryption (SSE-KMS), all API traffic TLS 1.2+, presigned URLs are HTTPS-only and short-lived (15 min default).
- **Supply-chain risk on framework/serialization formats:** pickled sklearn objects and similar formats are a known arbitrary-code-execution risk at deserialization time — registry enforces artifact-type allowlisting and, where feasible, requires safer serialization formats (e.g., ONNX, safetensors) for new registrations, with legacy pickle support flagged and scanned.

## 26. Authentication

- **Service-to-service (CI/CD, training pipelines, serving fleet watchers):** mTLS + short-lived service-account tokens (OIDC-based, e.g., SPIFFE/SPIRE identities or cloud-native workload identity) — no long-lived static API keys checked into pipeline configs.
- **End-user (human engineers via UI/CLI):** EA corporate SSO (SAML/OIDC through the corporate IdP), session tokens scoped short-lived (1 hour), refreshed via SSO re-auth, MFA enforced for production-promotion-approval actions specifically (step-up auth beyond normal SSO session for the highest-risk action in the system).
- **API tokens for CLI tooling:** personal access tokens scoped to specific model namespaces/permissions, rotatable, auditable per-token usage (so a leaked token's blast radius and revocation are both tractable).

## 27. Rate Limiting

- **Algorithm:** token bucket per (actor, endpoint-class) — smooths bursty CI/CD polling while allowing legitimate short bursts (e.g., a training pipeline registering 100 hyperparameter-sweep candidates in a tight window).
- **Limits:**
  - Per-service-account metadata reads: 100 req/s sustained, burst to 300.
  - Per-service-account registrations (writes): 10 req/s sustained, burst to 30 (covers sweep-completion bursts from Section 6's estimate of ~5/sec peak).
  - Per-human-user UI/API actions: 20 req/s (generous — humans don't script at machine speed except via their own CI, which uses the service-account limits instead).
  - Presigned URL issuance: capped separately (50/s per actor) since each issuance is cheap but a runaway client requesting URLs in a loop without uploading wastes them (short TTL mitigates further).
- **Enforcement point:** at the API gateway/ingress layer (e.g., Envoy with a rate-limit service, or cloud API gateway native rate limiting) — before requests reach the stateless Registry API pods, protecting Postgres from being the failure point under a misbehaving client.
- **Response on limit exceeded:** HTTP 429 with `Retry-After` header; CI/CD clients expected to implement exponential backoff (documented contract, enforced by client-library defaults the platform team provides).

## 28. Autoscaling

- **Registry API pods:** HPA on CPU utilization (target 60%) + custom metric on request queue depth; min replicas 4 (spread across AZs for availability), max 20 (launch-week headroom per Section 6's estimate).
- **Eval/Gate Service workers:** KEDA-based scaling on Kafka consumer-lag for `model.registered`/`model.stage_transition_requested` topics — scale workers up when lag grows (e.g., a launch-week burst of 100 sweep candidates registering near-simultaneously), scale to near-zero during quiet periods (cost lever, Section 29) since eval jobs are bursty/batch in nature, not steady-state traffic.
- **Postgres:** vertical scaling primarily (read replicas added if read QPS grows beyond a single replica's headroom); not a Kubernetes-native autoscaling target given it's a stateful primary/replica topology, typically a managed DB service (RDS/Cloud SQL) with manual/scheduled scaling for known launch-week load, not reactive autoscaling.
- **VPA consideration:** applied to Eval/Gate Service GPU-benchmark workers (right-sizing memory/CPU requests based on historical usage) since GPU worker pods are expensive to over-provision; VPA in "recommendation mode" feeding into periodic manual/CI-driven resource-request tuning rather than live in-place resize (which risks disrupting an in-flight GPU benchmark job).

## 29. Cost Optimization

- **Storage tiering:** artifacts unused for 90 days auto-transition to infrequent-access tier; unused for 1 year (and not production/compliance-held) transition to archive/glacier-class tier — Section 6 estimated ~500-800TB of 3-year backlog, and tiering is the single largest cost lever (glacier-class storage often 5-10x cheaper than standard).
- **Deduplication:** content-addressed storage (SHA-256-keyed) means identical artifacts (e.g., a re-registration of an unchanged feature-transformer pickle across many model versions) are stored once, referenced many times.
- **Spot/preemptible instances:** Eval/Gate Service GPU-benchmark workers (latency/throughput micro-benchmarking of exported artifacts) are stateless, retryable, checkpoint-free jobs — ideal spot-instance candidates, with the job queue (Section 28's KEDA-scaled pool) absorbing preemption via retry.
- **Caching to reduce egress:** CDN pre-warming (Section 11) reduces repeated origin fetches for the same production/canary artifact across thousands of autoscaling serving pods — origin egress cost is a real line item at 1.7GB/s peak fetch rates (Section 6) if not cached.
- **Retention policy enforcement (automated archival sweep, Section 20/23):** without automated enforcement, "keep everything forever out of caution" is the default failure mode of every registry — the nightly sweep job is explicitly a cost-control mechanism, not just hygiene.
- **Right-sizing control-plane compute:** Registry API pods are CPU-only, small (2-4 vCPU) instances — no GPU spend on the control plane itself, a common anti-pattern being colocating eval-benchmark GPU workers with the general API service and over-provisioning GPUs that sit idle most of the day (Section 28's KEDA scale-to-zero avoids this).

## 30. Disaster Recovery

- **RTO (control plane / metadata store):** 15 minutes — automated Postgres failover to a standby replica in another AZ; the platform team's runbook targets sub-15-min recovery given this system gates every title's deploy/rollback capability.
- **RPO (metadata store):** near-zero — synchronous replication to at least one standby within-region, async cross-region replica with < 5 min lag as a secondary safety net for full-region loss.
- **RTO/RPO (artifact store):** RPO near-zero (object storage versioning + cross-region replication, typically sub-15-min replication lag for standard cloud object-store cross-region replication); RTO effectively the time to fail over DNS/routing to the replicated bucket region, target < 30 min.
- **Backup strategy:** nightly full Postgres snapshot + continuous WAL archiving (point-in-time restore capability to any second in the last 30 days); audit log is inherently backup-equivalent given its WORM/immutable nature (no separate backup needed, it *is* the durable record).
- **Runbook priority:** if the registry's metadata store is fully lost and even backups are unavailable (worst case), the system is **partially reconstructable** from CI/CD pipeline logs (training_job_id, commit SHA references exist in Jenkins/Argo history) and from the artifact store's own object metadata/tags — this reconstruction path is explicitly documented and drilled (game-day exercise) as the last-resort recovery, given the assumption (Section 5.10) that the registry should not be a true unrecoverable single point of failure for "what's currently in production," since serving fleets also cache their last-known-good pointer locally.

## 31. Multi-Region Deployment

**Topology: active-active for reads, active-passive-per-write-region for the metadata store's primary.**

```
   US-EAST (primary write region)          EU-WEST (read region)          AP-SOUTHEAST (read region)
 ┌───────────────────────────┐          ┌───────────────────────┐      ┌───────────────────────┐
 │ Registry API (read+write)  │          │ Registry API (read-   │      │ Registry API (read-   │
 │                            │          │ only + async-forward  │      │ only + async-forward  │
 │ Postgres PRIMARY           │──repl───▶│ writes to primary)    │      │ writes to primary)    │
 │                            │          │ Postgres READ REPLICA │      │ Postgres READ REPLICA │
 │ Artifact bucket (primary)  │──repl───▶│ Artifact bucket (repl)│──────▶│ Artifact bucket (repl)│
 │ CDN edge                   │          │ CDN edge               │      │ CDN edge               │
 └───────────────────────────┘          └───────────────────────┘      └───────────────────────┘
        ▲                                        ▲                              ▲
        │  GeoDNS / latency-based routing for read traffic                      │
        └────────────────────────────────────────┴──────────────────────────────┘
```

- **Why active-passive for writes:** stage-transition writes require strong consistency (Section 3) — allowing concurrent writers in multiple regions to the same `stage_pointers` row risks split-brain ("prod is v47 in US, v46 in EU simultaneously"). A single write-primary with synchronous local replicas + async cross-region read replicas avoids this while still serving low-latency reads globally (the dominant traffic pattern per Section 6).
- **Writes from non-primary regions:** forwarded transparently to the primary region by the regional API layer (adds one cross-region round-trip, acceptable given writes are low-QPS, Section 6).
- **Artifact store:** naturally active-active-friendly since objects are immutable once written — replication is one-directional from the region of upload (usually co-located with the training cluster) fanned out to all serving regions, with CDN edges further reducing cross-region read latency for the hot (production-tagged) artifacts.
- **Latency routing:** GeoDNS/Anycast routes each region's serving fleet and CI/CD to their nearest read-capable Registry API instance — matters for a title's serving fleet autoscaling event (Section 6's cold-start fetch path) not incurring cross-continent round-trips at pod-startup time.

## 32. Blue/Green Deployment

Applies at two layers:
- **Registry service itself (the control plane):** standard blue/green for the Registry API pods — new API version deployed alongside old, traffic cut over via load balancer weight shift once health checks pass, old version kept warm for immediate rollback if the new API version regresses (e.g., a schema-migration-adjacent API change).
- **Model version promotion as a blue/green pattern:** conceptually, "current production version" and "candidate version" are a blue/green pair at the model level — the registry's stage-pointer mechanism (Section 10) *is* the blue/green switch primitive for models: promoting `v48` to `@production` while `v47` remains fully intact and instantly re-promotable is exactly blue/green semantics applied to models instead of application code. The registry never deletes the "blue" (previous production) version automatically — it remains available for instant rollback (Section 34) until the retention policy (Section 20/29) archives it after the safety window passes.

## 33. Canary Deployment

- **Traffic-split strategy (model-level canary, the registry's core promotion-workflow value-add):** a version promoted to `@canary` is served to a small percentage of traffic (e.g., 1-5% of matches for a matchmaking model, or a specific low-risk player cohort/region) by the serving layer, which reads the `@canary` pointer distinctly from `@production` and implements its own traffic-split logic (registry only supplies the pointer, not the split percentage — that's a serving-layer config, though the registry stores the intended canary config in `serving_config` metadata for auditability, per Section 14).
- **Health-check gates specific to this system:** before a canary can be auto-promoted (or manually promoted) to full production, the Eval/Gate Service (Section 8) checks: (a) online proxy-metric parity vs. current production (e.g., canary's match-quality complaint rate not statistically worse), (b) latency/error-rate parity from the serving fleet's own telemetry (fed back to the registry via a `model.canary_health` internal signal), (c) no active drift/incident alert tagged to the canary version. Only if all three gates pass within a minimum bake time (e.g., 2 hours minimum, or a minimum sample size of matches scored) is full-production promotion permitted, whether automated or human-approved.
- **Canary failure:** if any gate fails during the bake window, the registry auto-reverts the `@canary` pointer to the last-known-good version and emits a `model.stage_changed` event so the serving fleet immediately routes canary-cohort traffic back — this is functionally identical to the rollback mechanism (Section 34) but scoped to the canary cohort only, never touching the (already-safe) production cohort.

## 34. Rollback Strategy

- **Automated triggers:** (a) canary bake-window gate failure (Section 33) — automatic, no human in the loop, since blast radius is limited to the canary cohort; (b) production drift/incident alert (Section 21/23) crossing a severity threshold — triggers a *recommended* rollback (paged to on-call, one-click confirm) rather than fully automated for production-wide rollback, since an automated full-production revert carries its own risk (e.g., reverting during an unrelated infra incident could compound confusion) — human confirmation required for production-stage rollback, automated for canary-stage.
- **Rollback mechanics:** `POST /models/{name}/versions/{version}/rollback` is an atomic transaction — within a single DB transaction, the target version's `stage_pointers` row for `production` is updated to point to the specified prior version, the previous production version's pointer entry is superseded (retained in history, not deleted), an audit record is written, and the `model.stage_changed` event is emitted synchronously as part of the same request's response path (not just eventually via async event processing) — this ensures the serving fleet's watchers receive the rollback signal with minimal added latency during an active incident, when every second counts.
- **Rollback is never a "redeploy the old artifact"** operation in the sense of re-uploading bytes — it is purely a pointer change, since the prior version's immutable artifact never left storage (Section 32's blue/green framing) — this is what makes registry-mediated rollback fast (seconds, bounded by event propagation + serving fleet's poll/webhook latency) versus a full CI/CD redeploy pipeline (which could take many minutes to rebuild/repackage).

## 35. Observability

- **Tracing:** every Registry API request carries a distributed trace ID (W3C Trace Context / OpenTelemetry), propagated through to the Eval/Gate Service's async processing (trace context embedded in the Kafka event payload) so a single "promote v48 to production" action's full path — API call → DB write → event publish → Eval/Gate consumption → gate result write-back → downstream CI/CD trigger — is reconstructable as one trace across service and async boundaries.
- **Metrics:** the three golden signals (latency, traffic, errors) per endpoint exported via Prometheus/OpenTelemetry metrics; plus registry-specific business metrics (Section 22) as custom Prometheus gauges/counters (`registry_promotions_total{model_name, stage}`, `registry_rollback_total`, `registry_dlq_depth`).
- **Logs:** structured logs (Section 24) tagged with the same trace ID, enabling log-metric-trace correlation in a unified observability backend (e.g., Grafana + Tempo + Loki, or a vendor APM stack) — an on-call engineer paged on "DLQ depth > 0" (Section 23) can pivot directly from the alert to the specific trace of the stuck event, to the logs of the consumer that failed to process it, without manually correlating timestamps across three separate tools.
- **Why this matters more here than in a typical CRUD service:** the registry's failure modes (Section 41) are often *silent* (a stage-transition event lost in Kafka, a stale cache serving an old pointer) rather than loud 500-errors — tracing/correlation is the primary tool for catching "the system reported success but downstream never actually updated," which is the most dangerous class of bug for a system whose entire job is being the single source of truth.

## 36. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-registry-api
  labels: {app: model-registry-api}
spec:
  replicas: 4
  selector:
    matchLabels: {app: model-registry-api}
  template:
    metadata:
      labels: {app: model-registry-api}
    spec:
      containers:
        - name: registry-api
          image: registry.ea.internal/model-registry-api:1.14.2
          ports: [{containerPort: 8080}]
          env:
            - name: DB_HOST
              valueFrom: {secretKeyRef: {name: registry-db-creds, key: host}}
            - name: KAFKA_BROKERS
              value: "kafka-broker-0.internal:9092,kafka-broker-1.internal:9092"
          resources:
            requests: {cpu: "1", memory: "1Gi"}
            limits:   {cpu: "2", memory: "2Gi"}
          readinessProbe:
            httpGet: {path: /healthz, port: 8080}
            initialDelaySeconds: 5
          livenessProbe:
            httpGet: {path: /healthz, port: 8080}
            initialDelaySeconds: 15
---
apiVersion: v1
kind: Service
metadata: {name: model-registry-api}
spec:
  selector: {app: model-registry-api}
  ports: [{port: 443, targetPort: 8080}]
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: {name: model-registry-api-hpa}
spec:
  scaleTargetRef: {apiVersion: apps/v1, kind: Deployment, name: model-registry-api}
  minReplicas: 4
  maxReplicas: 20
  metrics:
    - type: Resource
      resource: {name: cpu, target: {type: Utilization, averageUtilization: 60}}
```

## 37. Terraform Infrastructure

```hcl
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "ea-model-registry-artifacts-${var.region}"

  versioning { enabled = true }

  lifecycle_rule {
    id      = "archive-cold-versions"
    enabled = true
    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }
    transition {
      days          = 365
      storage_class = "GLACIER"
    }
  }
}

resource "aws_s3_bucket_object_lock_configuration" "audit_log" {
  bucket = aws_s3_bucket.audit_log.id
  rule {
    default_retention {
      mode = "COMPLIANCE"
      days = 2555   # 7 years
    }
  }
}

resource "aws_db_instance" "registry_metadata" {
  identifier              = "model-registry-metadata"
  engine                  = "postgres"
  engine_version          = "15.4"
  instance_class          = "db.r6g.xlarge"
  allocated_storage       = 100
  multi_az                = true
  backup_retention_period = 30
  storage_encrypted       = true
  kms_key_id              = aws_kms_key.registry_db.arn
}

resource "aws_msk_cluster" "registry_events" {
  cluster_name           = "model-registry-events"
  kafka_version           = "3.6.0"
  number_of_broker_nodes  = 6
  broker_node_group_info {
    instance_type   = "kafka.m5.large"
    client_subnets  = var.private_subnet_ids
    storage_info { ebs_storage_info { volume_size = 500 } }
  }
}
```

## 38. Why This Architecture

- Separating **metadata (Postgres)** from **artifacts (object storage)** matches each data shape to the store built for it: transactional correctness for pointers/lineage, cheap durable blob storage for large immutable binaries — avoids the anti-pattern of storing multi-GB blobs in a relational DB (bloats backups, wrecks replication).
- **Event-driven decoupling (Kafka)** between the registry and its consumers (serving fleets, CI/CD, notifications) means the registry's availability isn't coupled to every downstream system's availability — a slow/down CI/CD webhook receiver doesn't block a promotion transaction from committing.
- **Immutable versions + mutable stage pointers** as two distinct concepts is the crux of enabling both auditability (nothing about a registered version ever silently changes) and fast, safe rollback (repointing is instant, no data movement).
- **Strong consistency for stage pointers specifically** (not the whole system) is a deliberate, narrow application of ACID guarantees exactly where split-brain is actually dangerous, while everything else (search, dashboards, audit reads) tolerates eventual consistency — avoids over-engineering the entire system to a consistency bar only one table actually needs.

## 39. Alternative Architectures

| Alternative | Description | Why rejected / when preferred |
|---|---|---|
| **Off-the-shelf MLOps platform (MLflow Model Registry, SageMaker Model Registry, Vertex AI Model Registry)** | Use a managed/open-source registry as-is instead of building custom | Preferred when EA's scale/cross-title governance needs are simple and a single cloud vendor is standardized on. Rejected here (or heavily extended) because EA's multi-title, multi-cloud-burst, custom-RBAC-per-studio, and game-specific canary/health-gate requirements exceed what off-the-shelf tools natively support without significant customization — but in a smaller-scope interview answer, "start with MLflow, extend via plugins" is a legitimate pragmatic answer, especially for a single-title team. |
| **Git-based model versioning (e.g., DVC + Git LFS)** | Treat model artifacts as git-tracked large files, use git branches/tags for stage promotion | Appealing for small teams wanting "everything in one familiar tool." Rejected at EA scale because git/DVC's performance and tooling degrade with thousands of large binary versions and dozens of concurrent teams, and it lacks first-class RBAC/approval-workflow primitives — better suited to a single small team's early-stage project. |
| **Fully decentralized per-title registries (no shared platform)** | Each title/studio runs its own lightweight registry, no cross-EA system | Rejected because it recreates today's ad-hoc-S3-bucket problem at a slightly more organized per-title level, loses cross-title visibility (e.g., can't easily audit "which titles are running unapproved models"), and duplicates undifferentiated infra (DB, object storage, RBAC) ~20 times. Would be preferred only if titles have truly incompatible compliance/data-residency requirements that make a shared platform legally infeasible. |
| **NoSQL-only metadata store (e.g., DynamoDB) instead of Postgres** | Use a fully managed key-value/document store for all metadata including stage pointers | Rejected for the stage-pointer table specifically because DynamoDB's conditional-write/transaction support, while workable, is a worse ergonomic fit than native relational transactions for the multi-row invariant checks (e.g., "exactly one production pointer per model, and the previous one must be superseded atomically with the new one"); would be preferred if write QPS were orders of magnitude higher than our estimate (Section 6 shows < 1 QPS average) and a single Postgres primary were a genuine bottleneck. |

## 40. Tradeoffs

| Decision | Benefit | Cost |
|---|---|---|
| Strong consistency on stage pointers | No split-brain on "what's in production" | Single-primary write bottleneck (mitigated: write volume is tiny) |
| Immutable version + mutable pointer model | Instant, safe rollback; strong audit guarantees | Extra conceptual complexity vs. naive "latest wins" versioning |
| Event-driven fan-out (Kafka) vs. direct synchronous webhooks | Downstream-system independence; replay capability | Added infra to operate (Kafka cluster); eventual consistency for non-pointer reads |
| Human-approval gate for production promotion (high-risk models) | Prevents fully-automated bad-model rollout | Slower time-to-production vs. full CI/CD automation |
| CDN pre-warming of production/canary artifacts | Avoids thundering-herd origin fetches at launch-day autoscale | Extra cost/complexity of proactive cache population logic |
| Centralized shared platform vs. per-title registries | Cross-EA governance/audit, no duplicated infra | Platform team becomes a cross-org dependency; single point of policy bottleneck if under-resourced |
| Object-store content addressing (SHA-256 dedup) | Storage savings, tamper-evidence | Extra hashing compute at upload time; slightly more complex artifact-manifest bookkeeping |

## 41. Failure Modes

- **Postgres primary failure mid-transition:** a stage-transition transaction is in-flight when the primary crashes. Mitigation: transaction either commits fully (WAL-durable before ack) or not at all — client (Registry API) surfaces a clear error, retries idempotently (transition requests carry a client-generated idempotency key so a retried "promote v48" doesn't double-process if the first attempt actually did commit before the crash was detected).
- **Kafka partition unavailable, `model.stage_changed` event delayed:** serving fleet's watcher doesn't see the new production pointer promptly. Mitigation: serving fleet also polls the Registry API directly on a fallback interval (e.g., every 60s) independent of event delivery — event-driven push is a latency optimization, not the sole mechanism, so a Kafka hiccup degrades to "60s slower rollout," not "rollout never happens."
- **Presigned URL leaked/reused beyond intended actor:** an artifact upload URL is exposed (e.g., accidentally logged). Mitigation: short TTL (15 min), single-use enforcement where the storage API supports it, and post-upload SHA-256 verification against the manifest recorded at registration time — a swapped/tampered artifact fails checksum verification before it's ever marked "available."
- **Eval/Gate Service silently stuck (e.g., GPU worker pool exhausted, jobs queue indefinitely):** a promotion request appears "pending" forever with no clear signal to the requester. Mitigation: DLQ + lag alerting (Section 12/23) plus a timeout SLA on gate evaluation (e.g., if no result within 30 min, auto-escalate to on-call rather than silently waiting) — prevents a stuck queue from blocking an urgent production promotion during an incident.
- **Cache serving stale stage pointer past invalidation (Redis node failure during invalidation):** a serving pod reads an old production version briefly after a rollback. Mitigation: short TTL (5s) as a hard upper bound on staleness regardless of invalidation delivery success, combined with the serving fleet's independent poll fallback.
- **Cross-region replication lag causes a non-primary region to serve a stale "current production" read during a real incident:** an on-call engineer in EU reads a not-yet-replicated rollback result. Mitigation: read-your-writes guarantee for the specific actor who performed the transition (route their immediate follow-up reads to the primary region for a short window post-write), plus dashboards explicitly surfacing replication lag so on-call knows to route to primary during active incident response.

## 42. Scaling Bottlenecks

- **At 10x scale (15,000 registrations/day, 500 ML teams):** Postgres primary write throughput for the `model_versions`/`approvals` tables becomes noticeable (still likely fine — 10x of <1 QPS average is still low absolute QPS, but launch-week bursts could hit tens of QPS sustained) — first real pressure point is more likely **RBAC/approval-workflow human latency** (500 teams all wanting production sign-off creates an approver bottleneck) rather than raw system throughput; mitigation is workflow/process (delegated approval authority, tiered auto-approval for lower-risk model classes) more than infra scaling.
- **At 100x scale (150,000 registrations/day, EA-wide plus hypothetical additional studios/acquisitions):** metadata store read replicas need to scale out further (more replicas, possibly geo-distributed read replicas beyond the 3-region topology in Section 31); object storage remains effectively unbounded, but **cross-region replication bandwidth** for artifact fan-out becomes a real cost/latency concern (150,000/day × 500MB avg ≈ 75TB/day needing replication to N regions) — likely requires moving from "replicate everything everywhere" to "replicate only production/canary-tagged artifacts to serving regions, replicate cold/experimental versions to a single archive region only."
- **Kafka topic throughput:** at 100x event volume, partition count for `model.stage_changed` etc. needs to scale (more partitions, potentially splitting by model-namespace prefix to preserve per-model ordering while parallelizing across more brokers) — this is a well-understood, mechanical scaling lever, not an architectural rework.
- **First true architectural breaking point:** if a future requirement emerged for *sub-100ms cross-region strongly-consistent writes* (e.g., if some future game required multi-region *active-active* production-promotion authority, not just reads) — the single-write-primary model (Section 31) would need to be replaced with a distributed consensus datastore (e.g., CockroachDB/Spanner), a genuine architecture change, not just a scale-out of the current design.

## 43. Latency Bottlenecks

**p50/p99 budget for the hottest read path — "serving pod fetches current production pointer at cold-start":**

| Stage | p50 | p99 |
|---|---|---|
| Client → Registry API (network) | 2 ms | 8 ms |
| Redis cache lookup (cache-aside, Section 11) | 1 ms | 5 ms (cache miss path adds DB round-trip) |
| [cache miss only] Postgres query | 3 ms | 15 ms |
| Registry API response serialization | 1 ms | 2 ms |
| **Total metadata fetch** | **~4 ms (cache hit)** | **~50 ms (cache miss, matches Section 3 target)** |
| Artifact fetch (CDN cache hit, if not already loaded on pod) | 200 ms (500MB @ ~2.5GB/s CDN edge throughput) | 2-3 s (cold CDN cache, origin fetch + cross-AZ) |

- The metadata lookup is never the bottleneck in absolute terms — it's the **artifact fetch on a true cold cache** that dominates real-world cold-start latency, which is why CDN pre-warming on promotion (Section 11) is not optional polish but a load-bearing latency control for launch-day autoscale events.
- For the **write path** ("promote to production"): p50 ~20ms (single DB transaction + async event publish), p99 ~150ms (includes synchronous event publish acknowledgment per Section 34's design choice to emit the rollback event synchronously) — this path is not QPS-sensitive (Section 6: <1 write QPS average) so the budget is generous by design; optimizing it further has no real payoff versus optimizing the read/artifact-fetch path that's actually under launch-day load.

## 44. Cost Bottlenecks

- **Largest single line item: artifact storage, specifically the 3-year hot+cold backlog (~500-800TB, Section 6).** Directly proportional to (a) how disciplined the archival-sweep policy is (Section 29) and (b) average artifact size — a title moving to larger fine-tuned LLM checkpoints (tens of GB vs. hundreds of MB) for dialogue/narrative systems could 10-50x the per-version storage cost even at constant registration volume, making artifact size the dominant driver more than registration count.
- **Cross-region replication egress** (Section 42) — scales with number of serving regions × artifact size × registration volume; the "replicate only tagged prod/canary versions to serving regions" optimization (Section 42) is the primary lever once this becomes material.
- **GPU spend for Eval/Gate benchmarking workers** — bursty by nature (Section 28's KEDA scale-to-zero), but if benchmark jobs grow in scope (e.g., adding full regression-test suites per promotion rather than lightweight latency checks) this could become a much larger, less bursty GPU cost center — worth monitoring as the eval-gate process matures and gains scope over time ("gate creep" is a real cost-growth pattern in mature MLOps orgs).
- **Control-plane compute (Registry API pods, Postgres instance) is comparatively negligible** — small fixed cost, not a lever worth optimizing further versus the storage/replication/GPU costs above.

## 45. Interview Follow-Up Questions

1. How would you handle a model that needs to be instantly pulled from production across all regions simultaneously due to a legal/compliance takedown request, faster than your normal event-propagation latency allows?
2. Your Eval/Gate Service says a canary passed, but 6 hours after full production promotion, player complaints spike. Walk through your incident response using only this system.
3. How do you prevent two engineers from simultaneously promoting different versions of the same model to production in a race condition?
4. If a title wants bit-for-bit training reproducibility (not just lineage traceability), what would you add to this design?
5. How does this registry interact with a Feature Store's own versioning — what happens if the feature schema changes but the model version doesn't get re-registered?
6. What happens to in-flight inference requests on serving pods when you execute a rollback — do you need any coordination with the serving layer beyond just flipping the pointer?
7. How would you extend this design to support A/B testing multiple production model variants simultaneously, not just a single canary-then-full-production path?
8. Why did you choose Postgres over a distributed SQL system given EA operates at global scale — isn't that under-engineering?
9. How do you handle GDPR right-to-erasure requests when a player's data is baked into an immutable training dataset snapshot that a production model's lineage points to?
10. What's your strategy if the registry needs to support third-party/external model providers (e.g., a licensed third-party anti-cheat ML vendor) alongside internally-trained models?

## 46. Ideal Answers

1. **Instant global takedown:** Don't rely solely on eventual event propagation. Maintain a small, globally-replicated "kill switch" table (or a dedicated fast-path, e.g., a globally-distributed low-latency config store like a CDN-edge key-value store) separate from the main stage-pointer mechanism, specifically for emergency de-activation — serving fleets check this kill-switch on every request (or every few seconds, cached briefly) in addition to their normal poll cycle. This trades a small amount of extra infra/complexity for a genuinely faster (seconds, not the normal ~60s poll interval) global stop capability, justified because legal/compliance takedowns are rare but must be near-instantaneous.

2. **Incident response walkthrough:** Pull the production version's lineage (`GET /lineage/{version}`) to identify training data snapshot, code commit, and eval-gate results recorded at promotion time. Check `model.eval_completed` history to see exactly what the canary gates measured (were they measuring the right proxy metrics, or did they miss the failure mode now manifesting?). Cross-reference the drift-detection system's alerts tagged to this version. Execute rollback (Section 34) to the last-known-good version immediately as a mitigation in parallel with root-cause investigation — rollback and root-cause are not sequential, the registry's fast pointer-flip rollback is specifically designed to decouple "stop the bleeding" from "understand why," since the latter can take hours the former cannot afford.

3. **Race condition prevention:** The `stage_pointers` table's primary key is `(model_name, stage)` — a promotion is a single transactional UPDATE (or INSERT ON CONFLICT) on that row, so the database's own row-level locking serializes concurrent promotion attempts; the second writer's transaction either blocks briefly then applies cleanly (last-writer-wins, by design, since "production" is inherently a single mutable pointer) or the API layer can additionally enforce optimistic concurrency (a `version` counter on the pointer row, requiring the client's promotion request to specify what it expects the *current* pointer to be, rejecting with a conflict error if it's changed) — I'd add the optimistic-concurrency check specifically because "last write silently wins" is dangerous for a high-stakes action like production promotion; better to surface a conflict to the second engineer than silently discard their intent.

4. **Bit-for-bit reproducibility:** Beyond lineage pointers, you'd need to pin and record exact library/framework versions (not just `xgboost==2.0.3` but a full frozen dependency lockfile hash), the exact random seeds used, the exact hardware/driver versions (CUDA version, GPU model — floating-point non-determinism varies across GPU generations), and ideally containerize the training environment itself (record the training container image digest, not just "the code," in the lineage record) so the entire environment — not just the code — is reproducible. This is meaningfully more expensive to maintain (container image storage, stricter environment pinning discipline) and I'd scope it only to models where reproducibility is a hard compliance requirement (e.g., models used in contexts with regulatory audit requirements), not apply it universally given the cost.

5. **Feature schema/model version interaction:** The registry stores `feature_schema_version` at registration time as an immutable lineage fact — it does not automatically re-validate or re-register anything when the Feature Store's schema evolves later. This is intentional: a production model's behavior shouldn't silently change because an unrelated schema migration happened. The gap you're pointing at is real, though — the mitigation is the Feature Store publishing its own schema-change events, and a compatibility-check service (could live in the Eval/Gate Service) subscribing to those events, cross-referencing which production model versions reference the now-changed schema, and raising an alert ("model X's production version references a feature schema that's since evolved incompatibly — verify serving-time features still match what the model expects") rather than silently letting inference-time feature drift go unnoticed.

6. **In-flight requests during rollback:** The registry's rollback is a pointer flip, not a pod restart — in-flight inference requests already using the currently-loaded model in a pod's memory complete normally against whatever was loaded when they started; the pointer change only affects *future* pod behavior (new pods loading fresh, or existing pods on their next poll-triggered refresh reloading the new/reverted artifact). If a title's serving layer supports hot-swapping the in-memory model without a pod restart, there's a brief window where different in-flight requests within the same pod could be served by different model versions during the swap — this needs the serving layer's own request-draining/versioned-routing logic (out of registry's scope, but the registry's event payload should include enough context — old version, new version, timestamp — for the serving layer to implement a clean drain-and-swap if it chooses to).

7. **Multi-variant A/B testing:** Extend the stage-pointer concept from a single mutable pointer per stage to a **weighted set of pointers** for an "experiment" stage — e.g., `production: {v48: 80%, v49: 15%, v50: 5%}` — with the serving layer's traffic-splitter reading this weighted map instead of a single version. The registry's role stays the same (source of truth for the mapping, audit trail of experiment changes) but the schema (Section 10) needs a `stage_pointers` table redesign to support multiple concurrent (version, weight) rows per (model_name, stage) rather than a strict one-row primary key — a real schema evolution, and I'd version this as a v2 API capability rather than retrofitting the existing single-pointer semantics, to avoid breaking the simpler single-version consumers (most models don't need multi-variant experiments).

8. **Postgres vs. distributed SQL:** Not under-engineering — right-sizing. EA's *player-facing* systems operate at global scale, but the registry's *own* write workload (Section 6: <1 QPS average writes, ~3-5GB total metadata at 3-year scale) is nowhere near the regime where a distributed SQL system's operational complexity (more moving parts, harder to reason about failure modes, often higher latency per-transaction due to consensus overhead) pays for itself. The read-heavy, geographically-distributed *read* traffic is handled by read replicas + caching (Sections 11, 31), which is a much simpler and cheaper way to get global read latency without paying the distributed-consensus tax on every write. I'd revisit this decision specifically if write QPS grew by 2-3 orders of magnitude or if true multi-region active-active write authority became a hard requirement (Section 42).

9. **GDPR erasure vs. immutable lineage:** These are in real tension and need an explicit policy, not just an engineering fix. The pragmatic approach: the registry's lineage record stores a *pointer* (dataset snapshot ID) never the underlying player data itself — so an erasure request against the raw training data can be fulfilled at the data-lake/Feature-Store layer independent of the registry's lineage record, which continues to correctly state "this model was trained on snapshot X" (a historical fact) even after snapshot X's underlying data is later redacted/tombstoned for compliance. The registry doesn't need to (and shouldn't) forge history by deleting the lineage pointer — what it should support is a `compliance_hold`/`data_since_redacted` flag on the affected dataset-snapshot references, surfaced in lineage queries, so anyone auditing later sees "this model's training data has since been subject to erasure requests" as an explicit, honest annotation rather than either silently losing the lineage record or falsely pretending the original data is still intact.

10. **Third-party/licensed models:** Register them identically to internally-trained models but with lineage fields reflecting external provenance (`training_job_id` becomes a vendor contract/delivery reference rather than an internal Argo workflow ID, `code_commit_sha` may be N/A or replaced with a vendor model-card/version identifier). The bigger design consideration is trust/verification: a third-party artifact should go through the same checksum-verification and Eval/Gate Service benchmarking (Section 33) before promotion — arguably *more* scrutiny, since you don't control its training lineage the way you do internal models — and RBAC should distinguish "internal model, standard approval" from "external vendor model, requires additional security/legal sign-off gate" as a distinct approval-workflow branch (Section 8/25), not a bolt-on afterthought.
