# Model Registry

## 1. Problem Framing

EA ships ML models across dozens of live-service titles (FC, Apex, Battlefield, The Sims) plus platform-wide systems (matchmaking, anti-cheat, churn prediction, LiveOps targeting). Today, promotion to production is ad-hoc: model files sit in S3 with inconsistent naming, lineage to training data/code is tribal knowledge, and rollback during a live incident can take hours because nobody can answer "what was running yesterday, and what changed."

A **Model Registry** is the system of record for:
- Every trained model version and its lineage (training data snapshot, code commit, hyperparameters, environment).
- Approval/promotion workflow (dev → staging → canary → production) with auditable sign-off.
- Physical artifact storage (weights, tokenizers, preprocessing pipelines, ONNX/TorchScript exports).
- Integration points for CI/CD (build pipelines, deployment triggers).

This is an internal platform system, not player-facing — but its availability gates whether any ML-driven feature can be safely updated, and its correctness gates whether a bad model reaches 30M+ concurrent players.

## 2. Functional Requirements

- FR1: Register a model version with artifact(s), metadata, and lineage (dataset ID, commit SHA, training job ID, hyperparameters, metrics).
- FR2: Immutable versioning per model name (`matchmaking-skill-model:v47`) plus mutable stage tags (`@staging`, `@production`, `@canary`).
- FR3: Stage transitions `None → Staging → Canary → Production → Archived`, gated by automated eval checks + human sign-off for Production.
- FR4: Full lineage query: model version → training data snapshot → feature schema → code commit → training job logs.
- FR5: Multi-format artifact storage (PyTorch, ONNX, TensorRT, TorchScript, tokenizers, pickled sklearn objects).
- FR6: CI/CD integration via webhooks/events on stage transitions.
- FR7: Model comparison: diff metrics between any two versions.
- FR8: Rollback: one-call API to re-promote a prior version to `@production`.
- FR9: Per-title, per-model RBAC.
- FR10: Search by title, task type, framework, owner team.
- FR11: Deprecation/retention policy enforcement (auto-archive unused models).
- FR12: Immutable audit log of every registration, transition, approval, and deletion.

## 3. Non-Functional Requirements

| Dimension | Target | Rationale |
|---|---|---|
| Latency (metadata read) | p99 < 50 ms | Queried at deploy-time/cold-start, not per-inference |
| Latency (artifact download, ~2 GB) | < 30 s | Pod autoscale must not stall on artifact fetch |
| Availability (control plane) | 99.95% | Gates deploys; brief outage tolerable outside incidents |
| Availability (artifact read path) | 99.99% | Blocks pod autoscaling during peak traffic |
| Throughput (registrations) | 500-2,000/day, spiking at launch | ~50 ML teams, frequent retraining/sweeps |
| Throughput (metadata reads) | 5,000 QPS peak | CI/CD polling, dashboards, serving lookups |
| Consistency (stage pointers) | Strong | Split-brain on "what's in prod" is a live-incident risk |
| Consistency (artifact store) | Read-after-write | CI/CD fetches immediately after registration |
| Durability | 11 nines, cross-region replicated | Models are often irreproducible |
| Cost | < $0.02/model-version-GB-month effective | Thousands of multi-GB checkpoints add up fast |

## 5. Assumptions

1. ~50 active ML teams across ~20 titles, each with 5-30 model families.
2. Average artifact size: 500 MB (range: 5 MB classical models to 40 GB LLM fine-tunes).
3. ~1,500 new version registrations/day at steady state, spiking to 4,000/day at launch.
4. Artifact storage on cloud blob storage (S3-compatible); metadata in a relational store for transactional stage-transition guarantees.
5. CI/CD (Jenkins/GitHub Actions/Argo) integrates via REST + webhooks; registry is a gate within CI/CD, not a replacement for it.
6. Human approval required for Production promotion on high-blast-radius models (matchmaking, anti-cheat, monetization); auto-promotion allowed for low-risk internal models passing eval gates.
7. ~500M registered accounts, ~30M peak concurrent — but registry load is driven by ML team activity, not player count.

## 6. Capacity Estimation

**Volume:** 1,500/day steady state (~0.02/sec), bursting to 4,000/day at launch (~5/sec during sweep-completion windows). Annual: ~548K new versions/year; ~1.6M records over 3 years before aggressive archival.

**Storage:** 500 MB avg × 548K/yr ≈ 274 TB/year raw. With 90-day hot retention and 2-year delete policy (except production/compliance-held versions): ~67 TB hot at any time, ~500-800 TB cold/archive after 3 years. Metadata: ~2 KB/version × 1.6M rows ≈ 3.2 GB — trivial, no sharding needed. Audit log: ~8 GB/3yr.

**Throughput:**
- Metadata reads: 5,000 QPS peak, dominated by CI/CD polling + serving-pod "get production pointer" checks at startup.
- Artifact reads: peak autoscale (e.g., 2,000 pods spinning up in 10 min × 500MB) ≈ 1.7 GB/s aggregate egress — needs CDN/regional caching, not single-origin fetch.
- Writes: low QPS (<1 avg) but each is a large multipart upload (up to 40 GB) — bandwidth-bound, not request-rate-bound.

**Compute (control plane only):** 5,000 QPS at ~5ms p50 ⇒ ~25 vCPU-seconds/sec ⇒ roughly 8-12 modest (4 vCPU) instances behind a load balancer, autoscaled 2x for launch weeks. No GPU needed on the control plane — GPUs live upstream (training) and downstream (serving).

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
        ▼                                         ▼           ▼
 ┌───────────────┐                     ┌────────────────┐  ┌──────────────────────┐
 │ Object Storage │                    │ Metadata Store  │  │  Audit Log Store      │
 │ (S3-compatible,│◀──── presigned ────│ (Postgres,      │  │ (append-only,         │
 │ versioned)     │      URLs          │  strongly       │  │  WORM)                │
 └──────┬─────────┘                    │  consistent)    │  └──────────────────────┘
        │  CDN/regional cache          └────────┬────────┘
        ▼                                       ▼ emits
 ┌───────────────┐                     ┌────────────────────┐
 │ Serving Fleet  │◀──── poll/webhook ─│  Event Bus (Kafka)  │──▶ Notification (Slack/PagerDuty)
 │ (per-title)    │   "prod pointer   │  model.promoted,     │──▶ CI/CD Trigger (deploy canary)
 └───────────────┘    changed" event  │  model.registered    │
                                       └──────────┬───────────┘
                                                  ▼
                                       ┌────────────────────┐
                                       │ Eval/Gate Service   │
                                       │ (offline metrics,   │
                                       │  fairness, latency) │
                                       └────────────────────┘
```

Flow: training job completes → uploads artifact to object storage (presigned URL) → registers metadata via API → API writes transactional row + emits `model.registered` → Eval/Gate Service runs automated evaluation, posts result back → if gates pass and (for prod) human approves → stage-transition write → `model.promoted` event → serving fleet/CI-CD consume event → canary or full rollout triggered.

## 8. Key Components

**Registry API Service** — CRUD on model/version/stage-transition entities; enforces RBAC and stage-gate rules; issues presigned upload/download URLs (never proxies large artifact bytes). Stateless pods behind a load balancer, autoscale on CPU/QPS.

**Metadata Store (Postgres)** — transactional source of truth for model, version, stage-pointer, approval records. Primary + read replicas; vertical scale is sufficient given ~3GB total data.

**Object Storage** — durable, versioned, immutable storage of model binaries as a content-addressed bundle per version. Clients get short-lived presigned URLs and upload/download directly. Regional replication + CDN edge caching for hot (production/canary) versions.

**Event Bus (Kafka)** — decouples registry writes from downstream consumers. Topics: `model.registered`, `model.stage_changed`, `model.archived`, `model.eval_completed`. Partitioned by `model_name` for per-model ordering.

**Eval/Gate Service** — runs automated offline evaluation (accuracy/AUC/NDCG vs. baseline, fairness checks, latency benchmark) before Staging→Canary and Canary→Production transitions. Job-queue-backed worker pool, GPU-backed for DL latency benchmarking.

**Audit Log Store** — immutable, tamper-evident record of every state change and actor. Write-only append, WORM object storage.

**Lineage Graph Index** — traversal queries ("what fed into production model X") across version → dataset → feature schema → training job → commit. At our scale (~1.6M nodes), a well-indexed relational adjacency table is sufficient — no need for a separate graph DB.

## 9. API Design

Base path: `/api/v1/registry` (gRPC mirrors same operations).

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/models` | Register a new model family |
| POST | `/models/{model_name}/versions` | Register a new version (metadata + presigned upload handshake) |
| GET | `/models/{model_name}/versions/{version}` | Fetch version metadata + lineage |
| GET | `/models/{model_name}/versions?stage=production` | List versions by stage/tag |
| POST | `/models/{model_name}/versions/{version}/transition` | Request stage transition |
| POST | `/models/{model_name}/versions/{version}/approve` | Human approval (RBAC-gated) |
| POST | `/models/{model_name}/versions/{version}/rollback` | Re-promote a prior version, atomically demoting current |
| GET | `/models/{model_name}/lineage/{version}` | Full lineage graph |
| GET | `/models/{model_name}/diff?a={v1}&b={v2}` | Metric/config diff |
| DELETE | `/models/{model_name}/versions/{version}` | Soft-delete/archive |
| GET | `/audit/{model_name}` | Audit trail |

Example register request:
```json
POST /api/v1/registry/models/matchmaking-skill-model/versions
{
  "training_job_id": "job-2026-07-08-a91f",
  "code_commit_sha": "3f9a1c2",
  "dataset_snapshot_id": "ds-fc-ranked-2026-06-30",
  "feature_schema_version": "feat-schema-v12",
  "hyperparameters": {"lr": 0.001, "depth": 8, "n_estimators": 400},
  "framework": "xgboost==2.0.3",
  "artifact_manifest": [{"name": "model.xgb", "size_bytes": 42000000, "sha256": "..."}],
  "offline_metrics": {"auc": 0.812, "ndcg@10": 0.734}
}
```
Response includes `version_id`, `presigned_upload_urls[]`, `status: "PENDING_UPLOAD"`.

Version identifiers are immutable monotonic integers per model name, never reused. Stage tags are mutable pointers — this separation lets "production" move without renaming/duplicating artifacts.

## 10. Database Design

**Postgres for metadata + transactional stage pointers; S3-compatible object store for binaries; append-only WORM storage for audit log.**

Stage-transition correctness requires ACID (the "current production version" pointer must never be ambiguous under concurrent writes) — a relational store with row-level locking fits given only ~3-5 GB of metadata at 3-year scale. No need for a distributed NoSQL system trading consistency for scale we don't need.

Core schema:
```sql
CREATE TABLE models (
  model_name  TEXT PRIMARY KEY,
  owning_team TEXT NOT NULL,
  task_type   TEXT NOT NULL,  -- classification|ranking|regression|generative
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE model_versions (
  model_name          TEXT REFERENCES models(model_name),
  version              INT NOT NULL,
  training_job_id      TEXT,
  code_commit_sha       TEXT,
  dataset_snapshot_id   TEXT,
  feature_schema_version TEXT,
  hyperparameters       JSONB,
  offline_metrics       JSONB,
  framework             TEXT,
  artifact_uri_prefix   TEXT NOT NULL,  -- s3://bucket/model_name/v47/
  created_by            TEXT NOT NULL,
  created_at            TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (model_name, version)
);

CREATE TABLE stage_pointers (
  model_name TEXT REFERENCES models(model_name),
  stage      TEXT NOT NULL,  -- staging|canary|production
  version    INT NOT NULL,
  updated_by TEXT NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (model_name, stage)  -- one current pointer per stage per model
);

CREATE TABLE approvals (
  id           BIGSERIAL PRIMARY KEY,
  model_name   TEXT,
  version      INT,
  target_stage TEXT,
  approver     TEXT,
  gate_results JSONB,
  approved_at  TIMESTAMPTZ DEFAULT now()
);
```

Partitioning not required at this scale (~1.6M rows/3yr). If volume grows 10x, partition `model_versions`/`approvals` by `model_name` hash or studio namespace — native Postgres partitioning suffices.

## 11. Caching

- **What's cached:** (a) current production/canary pointer per model — hottest read path; (b) model metadata (immutable once written); (c) artifact bytes at CDN edge for production/canary versions only.
- **Cache-aside** for stage pointers: read through Redis, fall back to Postgres on miss, short TTL (5s) bounds staleness while absorbing most of the 5,000 QPS read load.
- **Write-through + invalidation** on stage transition: the transition transaction invalidates the Redis key synchronously before returning success — guarantees no stale read past the transaction boundary, with the 5s TTL as a safety net.
- **Immutable metadata:** `model_versions` rows never change after creation — cache with long/no TTL, no invalidation logic needed.
- **CDN pre-warming:** production/canary artifacts are pushed to CDN proactively on promotion, not cold on first request — avoids thundering-herd origin fetches when thousands of pods cold-start simultaneously on launch day.

## 12. Queues & Async Processing

- **Queued work:** artifact upload-completion → eval job; stage-transition-requested → gate evaluation; promotion → CI/CD deploy webhook fan-out; nightly archival sweep.
- **Delivery semantics:** at-least-once (Kafka default); consumers must be idempotent — e.g., Eval/Gate Service keys evaluation runs by `(model_name, version, gate_type)` with upsert, so duplicate delivery is a no-op.
- **Dead-letter handling:** after 5 retries with backoff, failed processing routes to a DLQ topic; DLQ depth > 0 on `model.stage_transition_requested` pages on-call as high severity, since a stuck gate can block a live-incident rollback.
- CI/CD deploy-trigger webhooks are fire-and-forget from the registry's side; the consuming CI/CD system owns its own retry — the registry only guarantees it emitted the trigger, not downstream deploy success.

## 13. Streaming & Event-Driven Architecture

Kafka topics, all keyed by `model_name` for per-model ordering:

| Topic | Producers | Consumers |
|---|---|---|
| `model.registered` | Registry API | Eval/Gate Service, Audit Pipeline, Search Indexer |
| `model.stage_transition_requested` | Registry API | Eval/Gate Service, Notification Service |
| `model.stage_changed` | Registry API | Serving Fleet Watchers, CI/CD Trigger, Notification, Audit Pipeline |
| `model.eval_completed` | Eval/Gate Service | Registry API (writes approval), Dashboard |
| `model.archived` | Registry API (retention sweep) | Storage Lifecycle Manager, Audit Pipeline |

Each downstream system runs its own consumer group so all receive every event; horizontal scaling happens within a group, partitioned by `model_name`. Schema evolution via Avro + schema registry, backward-compatible only.

## 14. Model Serving (Interface Contract)

Out of scope for this chapter, but the registry's serving-facing contract:
- Serving is framework-agnostic on the registry side (Triton for GPU-heavy DL, lightweight services for classical models); the registry just stores whatever artifact format serving expects.
- Serving's deployment controller subscribes to `model.stage_changed` for `stage=production`/`canary`, fetches the artifact via presigned URL/CDN, and performs its own rollout — the registry never manages serving pods or performs inference.
- Batching/hardware selection are serving-layer concerns, passed through via an opaque `serving_config` field the registry stores but doesn't interpret.

## 15. Feature Store (Interface)

The registry doesn't host feature data — it references a Feature Store's schema version (`feature_schema_version`) for lineage. It stores the exact `dataset_snapshot_id` used at training time, which is the mechanism for later answering "did this model leak future data?" — the registry doesn't enforce point-in-time correctness itself, but is the durable record that makes that question answerable in a post-mortem.

## 16. Vector Database

**N/A.** The registry does keyed lookups (name + version) and small-scale graph traversal, both well served by relational indexes. Embedding-based retrieval systems that use vector DBs are themselves just entries in this registry.

## 17. Inference Pipelines (Where the Registry Sits)

Registry involvement is at pod cold-start/refresh only, never per-request:

```
Player queues for match → Matchmaking Service pod startup
   → GET /models/matchmaking-skill-model/versions?stage=production  (cached, p99 < 50ms)
   → returns version=47, artifact_uri
   → pod fetches artifact (CDN-cached) if not already loaded
   → loaded model scores match candidates → response to client
```

A well-designed serving layer loads the model once and serves thousands of QPS without touching the registry again until the next poll/refresh (typically 30-60s) or push notification.

## 18. Training Pipelines (Contract)

The registry defines what training pipelines must supply to register a version:
- `dataset_snapshot_id` is required (not optional) — prevents "trained on a moving target" lineage gaps.
- Orchestration is title-specific (Kubeflow, Argo, SageMaker, Airflow); registry only requires a queryable `training_job_id` for lineage drill-down.
- For large distributed training jobs (LLM fine-tunes), the registry records only the final checkpoint + training config (world size, epochs, batch size) — not the intermediate distributed-training mechanics.
- No version can register without commit SHA, dataset snapshot ID, and training job ID — the registry's biggest lever over training-pipeline hygiene EA-wide.

## 19. Retraining Strategy

- **Cadence varies by model class:** matchmaking (weekly, meta shifts), anti-cheat (daily, adversarial arms race), recommendation/LiveOps (daily, freshness-driven), churn/LTV (monthly, slower signal).
- The registry doesn't decide *when* to retrain — a drift monitor or scheduler queries `GET /models/{name}/versions?stage=production` + timestamp to judge staleness.
- **Emergency retrains** (triggered by drift alerts or live incidents) are treated identically in the data model to scheduled retrains; only the approval path differs (expedited, senior sign-off, still fully audit-logged).

## 20. Drift Detection (Interface)

Not run by the registry (owned by a monitoring system), but the registry is the join-key that makes drift actionable — every alert must resolve to "which model version, trained on which data."

- **Data drift:** PSI/KL-divergence on feature distributions vs. the training snapshot the current production version cites. PSI > 0.2 = warning, > 0.3 = retrain-recommendation.
- **Concept drift:** online proxy metrics (e.g., calibration vs. actual outcomes); >15% relative Brier score increase vs. the value recorded at promotion time triggers investigation.
- Every drift alert carries `model_name` + `version`, so the on-call engineer's first move is a registry `/lineage` lookup instead of spelunking S3 buckets.

## 21. Monitoring & Alerting

**Monitoring:**
- Infra: API p50/p99 latency, error rate, Postgres pool saturation, Kafka consumer lag, object storage error rate, CDN cache hit ratio.
- Model quality (surfaced via `/diff`): offline eval metric deltas between consecutive versions.
- Business/process: promotion frequency per team, mean time-to-promotion, rollback frequency (proxy for release-quality regression).

**Alerts:**

| Condition | Severity | Route |
|---|---|---|
| API error rate > 1% (5 min) | P2 | On-call platform eng |
| Stage-pointer read p99 > 200ms | P2 | On-call platform eng |
| Kafka consumer lag on `model.stage_changed` > 5,000 msgs / 2 min | P1 | On-call — delays production rollouts/rollbacks |
| DLQ depth > 0 on `model.stage_transition_requested` | P1 | On-call — blocks rollback capability |
| Unapproved production promotion (RBAC bypass) | P1 security | Security on-call, immediate page |
| Drift alert (external, registry-tagged) | P2 (P1 if live event window) | Title's ML on-call |
| Rollback executed | P4 informational | Posted to Slack |

## 22. Logging

- Structured JSON per API request (`request_id, actor, model_name, version, action, latency_ms, outcome`).
- **PII:** registry metadata (names, hyperparameters, commit SHAs) isn't player PII; `dataset_snapshot_id` may point to data that contains player telemetry, but the registry stores only the pointer. Actor identity is employee-identifying, handled under corporate IT policy.
- Retention: operational logs 90 days hot + 1 year cold; audit log retained indefinitely (or per compliance minimum) as the tamper-evident promotion record.
- Every log line carries `request_id` and `model_name:version` for incident trace-through.

## 23. Security

- **Artifact tampering:** SHA-256 checksums recorded at registration, verified on every download; object versioning + immutability (Object Lock) prevents silent overwrite; presigned upload URLs are single-use, short-TTL.
- **Unauthorized promotion:** RBAC enforced server-side on every transition endpoint; two-person rule for player-facing/high-blast-radius model classes.
- **Lineage forgery:** CI pipeline identity (service account) is the only actor permitted to set lineage fields directly — humans can request transitions but not author lineage metadata.
- **Encryption:** metadata store encrypted at rest (KMS), artifact bucket SSE-KMS, TLS 1.2+ everywhere, presigned URLs HTTPS-only with 15 min default TTL.
- **Supply chain:** pickled sklearn objects are an arbitrary-code-execution risk at deserialization — registry enforces artifact-type allowlisting, prefers safer formats (ONNX, safetensors) for new registrations, flags legacy pickle usage.

## 24. Authentication & Rate Limiting

- **Service-to-service:** mTLS + short-lived OIDC/workload-identity tokens — no static API keys in pipeline configs.
- **Human users:** EA corporate SSO (SAML/OIDC), 1-hour session tokens, MFA step-up required specifically for production-promotion approval.
- **CLI tokens:** scoped, rotatable, per-token audit trail.
- **Rate limiting:** token bucket per (actor, endpoint-class), enforced at the API gateway before reaching Registry API pods. Reads: 100 req/s/service-account (burst 300). Writes: 10 req/s (burst 30, covers sweep bursts). Presigned URL issuance capped separately (50/s) since a runaway client can waste short-TTL URLs cheaply. 429 + `Retry-After` on limit exceeded.

## 25. Autoscaling

- Registry API pods: HPA on CPU (target 60%) + queue depth; min 4 (AZ-spread), max 20 (launch-week headroom).
- Eval/Gate Service: KEDA-based scaling on Kafka consumer lag — scales up for launch-week registration bursts, near-zero at quiet times (bursty/batch workload, not steady traffic).
- Postgres: vertical scaling + read replicas, not a Kubernetes autoscaling target given the stateful primary/replica topology.

## 26. Cost Optimization

- **Storage tiering:** artifacts unused 90 days → infrequent-access tier; unused 1 year (and not prod/compliance-held) → archive/glacier tier. This is the single largest lever given the ~500-800TB 3-year backlog.
- **Deduplication:** content-addressed (SHA-256) storage means identical artifacts across versions are stored once.
- **Spot instances** for Eval/Gate GPU-benchmark workers — stateless, retryable, checkpoint-free.
- **CDN pre-warming** reduces repeated origin egress across autoscaling serving pods.
- **Retention enforcement** (nightly sweep) is a cost-control mechanism, not just hygiene — "keep everything forever" is the default failure mode otherwise.
- **Right-sized control plane:** small CPU-only API pods; avoid colocating GPU eval workers with the general API service.

## 27. Operational Concerns

At SDE2 scope, treat this as a checklist: **backups** (automated snapshots + tested restore path), **rollback** (one-command revert to last-known-good), **canary/blue-green rollout** (shift small traffic %, watch error rate and model-quality signals, then ramp), **basic observability** (dashboards + alerts on latency, error rate, top model-quality signals, wired to on-call). Multi-region active-active topology and detailed Kubernetes/Terraform manifests are Staff/Principal-level concerns — worth knowing they exist, not worth rehearsing.

## 28. Why This Architecture

- Separating **metadata (Postgres)** from **artifacts (object storage)** matches each data shape to the right store: transactional correctness for pointers/lineage, cheap durable blobs for large binaries — avoids bloating a relational DB/its backups with multi-GB blobs.
- **Event-driven decoupling (Kafka)** means registry availability isn't coupled to every downstream consumer — a slow CI/CD webhook receiver doesn't block a promotion from committing.
- **Immutable versions + mutable stage pointers** enables both auditability (nothing about a version silently changes) and fast, safe rollback (repointing is instant).
- **Strong consistency scoped narrowly to stage pointers** — everything else (search, dashboards, audit reads) tolerates eventual consistency, avoiding over-engineering the whole system to a bar only one table needs.

## 29. Alternative Architectures

| Alternative | Why rejected / when preferred |
|---|---|
| Off-the-shelf MLOps platform (MLflow, SageMaker/Vertex Model Registry) | Good default for a single-title team on one cloud. Rejected/extended here because EA's multi-title governance, custom RBAC-per-studio, and game-specific canary gates exceed what these support out of the box. |
| Git-based versioning (DVC + Git LFS) | Fine for a small team wanting one familiar tool. Rejected at EA scale — git/DVC tooling degrades with thousands of large binaries and dozens of concurrent teams, and lacks first-class RBAC/approval workflows. |
| Fully decentralized per-title registries | Recreates today's ad-hoc-S3 problem at a slightly more organized level, loses cross-title audit visibility, duplicates infra ~20x. Only preferable if titles have truly incompatible compliance/data-residency needs. |
| NoSQL-only metadata store (DynamoDB) | Rejected for stage pointers specifically — conditional-write support is a worse fit than native relational transactions for "exactly one production pointer per model, atomically superseded." Would reconsider if write QPS were orders of magnitude higher. |

## 30. Tradeoffs

| Decision | Benefit | Cost |
|---|---|---|
| Strong consistency on stage pointers | No split-brain on "what's in production" | Single-primary write bottleneck (fine — write volume is tiny) |
| Immutable version + mutable pointer model | Instant, safe rollback; strong audit trail | Extra conceptual complexity vs. naive "latest wins" |
| Event-driven fan-out vs. sync webhooks | Downstream independence, replay capability | Extra infra to operate; eventual consistency for non-pointer reads |
| Human-approval gate for high-risk promotions | Prevents fully-automated bad rollout | Slower time-to-production |
| CDN pre-warming | Avoids thundering-herd on launch-day autoscale | Extra proactive cache-population complexity |
| Centralized shared platform vs. per-title | Cross-EA governance, no duplicated infra | Platform team is a cross-org dependency/bottleneck |

## 31. Failure Modes

- **Postgres primary fails mid-transition:** transaction either fully commits (WAL-durable) or not at all; client retries idempotently via a client-generated idempotency key.
- **Kafka delay on `model.stage_changed`:** serving fleet also polls the API directly on a fallback interval (~60s) — event push is a latency optimization, not the sole mechanism, so a Kafka hiccup degrades gracefully rather than blocking rollout.
- **Leaked presigned URL:** short TTL (15 min), single-use where supported, plus post-upload SHA-256 verification against the registered manifest catches a swapped artifact before it's marked available.
- **Eval/Gate Service stuck** (e.g., GPU pool exhausted): DLQ + lag alerting plus a timeout SLA (auto-escalate if no result within 30 min) prevents a stuck queue from blocking an urgent promotion.
- **Stale cached stage pointer after a Redis failure during invalidation:** bounded by the 5s TTL regardless of invalidation delivery, plus the serving fleet's independent poll fallback.

## 32. Scaling Bottlenecks

- **At 10x scale** (15K registrations/day, 500 teams): raw DB throughput is still fine; the real pressure point is **human approval-workflow latency** (500 teams wanting sign-off) — solved by process (delegated approval, tiered auto-approval for low-risk classes), not infra.
- **At 100x scale:** metadata read replicas need to scale out further; cross-region artifact replication bandwidth becomes a real cost/latency concern — mitigated by replicating only production/canary-tagged artifacts to serving regions, archiving cold/experimental versions to one region.
- **True architectural breaking point:** a future requirement for sub-100ms cross-region strongly-consistent writes (e.g., true multi-region active-active promotion authority) would require replacing the single-write-primary model with a distributed consensus store (CockroachDB/Spanner) — a genuine rework, not a scale-out.

## 33. Latency Budget

Hottest read path — serving pod fetching the current production pointer at cold-start:

| Stage | p50 | p99 |
|---|---|---|
| Client → Registry API (network) | 2 ms | 8 ms |
| Redis cache lookup | 1 ms | 5 ms (miss adds DB round-trip) |
| [miss only] Postgres query | 3 ms | 15 ms |
| Response serialization | 1 ms | 2 ms |
| **Total metadata fetch** | **~4 ms (hit)** | **~50 ms (miss, meets target)** |
| Artifact fetch (CDN hit) | 200 ms | 2-3 s (cold CDN, origin fetch) |

The artifact fetch on a true cold cache — not the metadata lookup — dominates real-world cold-start latency, which is why CDN pre-warming on promotion is load-bearing, not polish. Write path ("promote to production") is not QPS-sensitive (<1 QPS avg), so its budget (p50 ~20ms, p99 ~150ms) is generous by design.

## 34. Cost Bottlenecks

- **Largest line item: artifact storage** (~500-800TB 3-year backlog). Driven more by average artifact size than registration count — a shift toward larger LLM fine-tunes (tens of GB vs. hundreds of MB) could 10-50x per-version storage cost at constant registration volume.
- **Cross-region replication egress** scales with regions × artifact size × volume; mitigated by replicating only tagged prod/canary versions to serving regions.
- **GPU spend for Eval/Gate benchmarking** is bursty today (scale-to-zero), but could become a much larger, less bursty cost center if benchmark scope grows (e.g., full regression suites per promotion) — worth monitoring as "gate creep."
- Control-plane compute (API pods, Postgres) is comparatively negligible.

## 35. Interview Follow-Up Questions

1. How would you instantly pull a model from production across all regions for a legal/compliance takedown, faster than normal event propagation allows?
2. Eval/Gate says a canary passed, but 6 hours after full promotion, player complaints spike. Walk through incident response using only this system.
3. How do you prevent two engineers from racing to promote different versions of the same model simultaneously?
4. If a title needs bit-for-bit training reproducibility, not just lineage traceability, what would you add?
5. How does this registry interact with Feature Store versioning if the feature schema changes but the model version isn't re-registered?
6. What happens to in-flight inference requests on serving pods during a rollback?
7. How would you extend this to support A/B testing multiple production variants, not just single-canary-then-full?
8. Why Postgres over a distributed SQL system given EA's global scale — isn't that under-engineering?
9. How do you handle GDPR erasure when a player's data is baked into an immutable training snapshot a production model's lineage points to?
10. How would you support third-party/licensed models alongside internally-trained ones?

## 36. Ideal Answers

1. **Instant takedown:** Don't rely solely on eventual event propagation — maintain a small, globally-replicated "kill switch" config store that serving fleets check every few seconds independent of the normal poll cycle. Extra infra, justified because legal takedowns are rare but must be near-instant.

2. **Incident response:** Pull the production version's lineage (training snapshot, commit, eval-gate results), cross-reference drift alerts tagged to that version. Roll back to last-known-good immediately, in parallel with root-cause investigation — the pointer-flip rollback exists to decouple "stop the bleeding" from "understand why."

3. **Race prevention:** `stage_pointers`' primary key is `(model_name, stage)`, so promotion is a single transactional UPDATE serialized by row-level locking. I'd add optimistic concurrency (a version counter the client must match) on top, since silently discarding a conflicting promotion is too risky — better to surface a conflict than silently overwrite intent.

4. **Bit-for-bit reproducibility:** Pin exact dependency lockfile hashes, random seeds, hardware/driver versions, and the training container image digest. Costly to maintain — scope it only to models with a hard regulatory requirement.

5. **Feature schema drift:** The registry stores `feature_schema_version` as an immutable lineage fact and doesn't auto-revalidate on later schema changes (a production model shouldn't silently change behavior from an unrelated migration). Mitigate the gap with the Feature Store publishing schema-change events that a compatibility checker cross-references against production versions.

6. **In-flight requests during rollback:** Rollback is a pointer flip, not a pod restart — in-flight requests finish against whatever was loaded when they started; only future pod refreshes pick up the change. Hot-swap-in-memory logic, if needed, is the serving layer's responsibility.

7. **Multi-variant A/B:** Extend the stage-pointer concept to a weighted set for an "experiment" stage (e.g., `production: {v48: 80%, v49: 15%, v50: 5%}`), with the serving traffic-splitter reading the weighted map. Needs a schema change to support multiple concurrent rows per (model_name, stage) — version it as a v2 capability.

8. **Postgres vs. distributed SQL:** Right-sizing, not under-engineering — write workload (<1 QPS avg) is nowhere near where distributed SQL's operational complexity pays off. Global read latency is handled more simply via read replicas + caching. Revisit only if write QPS grows by orders of magnitude or active-active writes become a hard requirement.

9. **GDPR vs. immutable lineage:** The lineage record stores a pointer (dataset snapshot ID), never the underlying player data — erasure happens at the data-lake layer while the lineage record still correctly states "trained on snapshot X," annotated with a `compliance_hold`/`data_since_redacted` flag rather than deleted or left misleading.

10. **Third-party models:** Register like internal models but with lineage fields reflecting external provenance (vendor contract reference instead of training job ID, vendor model-card version instead of commit SHA). Same (or stricter) checksum verification and Eval/Gate benchmarking before promotion, plus an additional security/legal sign-off gate.
