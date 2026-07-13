# ML Platform

## 1. Problem Framing

EA runs 40+ live-service titles (FIFA/FC, Apex, Battlefield, Sims, mobile) each hand-rolling training scripts, GPU allocation, and deployment glue for churn, matchmaking, LTV, anti-cheat, recommendation, and moderation models. Result: duplicated effort, inconsistent governance, no feature reuse, no cost visibility.

Build an internal **ML Platform**: paved-road pipelines, orchestration, compute scheduling, and self-service deployment so any studio goes from notebook to production endpoint without owning infra.

This is a platform-for-platforms problem — customers are ~800 internal ML engineers, not players. Success = time-to-production drops from weeks to days, GPU utilization rises, governance (access, lineage, cost attribution) centralizes.

## 2. Functional Requirements

- Self-service pipeline authoring (DAG: data prep → feature build → train → eval → register → deploy)
- Managed compute scheduling (CPU/GPU/multi-node) without manual cluster ops
- Model registry with lineage (data version, code commit, hyperparameters, metrics)
- One-click deploy to serving endpoint (batch or online)
- Feature store: shared, reusable, online (low-latency) + offline (training)
- Experiment tracking across runs
- Multi-tenant isolation (no cross-team data/compute leakage)
- Automated retraining triggers (schedule- or drift-based)
- Cost attribution per team/project/model
- Rollback and canary support for online serving

## 3. Non-Functional Requirements

| Dimension | Target |
|---|---|
| Control-plane API availability | 99.9% |
| Online inference availability | 99.95% per endpoint |
| Online inference p99 latency | < 50ms at model-server hop |
| Batch scoring throughput | 500M rows/day platform-wide |
| Training job scheduling latency | < 30s queue-to-running (priority jobs) |
| Feature store online read p99 | < 10ms |
| Consistency | Eventual OK for registry/metadata; strong consistency required for point-in-time feature joins in training |
| Cost | GPU utilization > 65% fleet-wide (vs. ~20-30% typical of siloed clusters) |
| Multi-tenancy | Hard quota isolation, no noisy-neighbor starvation |

## 5. Assumptions

1. 800 ML practitioners across ~40 studios; ~250 actively training weekly.
2. ~3,000 registered models; ~600 actively serving.
3. Hybrid infra: 2,000 on-prem GPUs (A100/H100) + cloud burst capped at 20% of steady-state cost.
4. Average training job: 4-GPU, 6h; large jobs (embeddings) up to 64-GPU multi-node, 24h.
5. Online serving: median 200 QPS, peak title up to 50K QPS during live events.
6. Batch scoring: nightly over up to 150M player profiles.
7. Feature store: ~5,000 feature definitions, ~50TB offline (Iceberg), ~2TB online (Redis/DynamoDB-class).
8. Frameworks: PyTorch, TensorFlow, XGBoost/LightGBM, scikit-learn — containerized, not source-integrated.
9. Multi-region: US-East primary, US-West, EU (Dublin) for residency/latency.
10. Retention: raw telemetry 13 months, model artifacts indefinite, training datasets 90 days default.

## 6. Capacity Estimation

**Online serving QPS**: 600 endpoints × 200 median QPS = 120K baseline; peak burst (5 titles spiking at once) ≈ 120K + 5×50K = 370K QPS. Provision for 500K QPS peak.

**GPU fleet (training)**: 250 teams × 2 jobs/week × 4 GPU × 6h = 12,000 GPU-hours/week (~1,714/day) → ~110 GPUs at 65% utilization target for routine jobs. Large multi-node jobs (~10/week × 64 GPU × 24h) add ~140 GPUs at target utilization. Total steady-state pool: ~250 GPUs dedicated + elastic burst of 500 (carved from the shared 2,000-GPU on-prem fleet, bursts to cloud past a queue-depth threshold).

**Serving compute**: Tabular (CPU): ~500 QPS/vCPu → 500K QPS / 500 ≈ 1,000 vCPUs, ×2 for HA = 2,000 vCPUs. DL (GPU, batched): ~2,000 QPS/GPU → 100K QPS of DL traffic needs 50 GPUs, ×2 for HA/regions = 100 GPUs.

**Storage**: Offline feature store 50TB (growing 5TB/mo); online 2TB KV; model registry ~600TB after content-addressed dedup (3,000 models); training snapshots ~200TB rolling (90-day retention); platform consumes ~200K events/sec off the EA-wide 2M events/sec telemetry Kafka.

**Cost sanity check**: 250 steady-state GPUs (~$3/hr amortized) ≈ $540K/month. Cloud burst (~15% of the 500-GPU pool actually bursts, ~$5/hr on-demand) ≈ $270K/month — motivates aggressive spot use (Section 29).

## 7. High-Level Architecture

```
   Studio ML Engineers (800) — SDK/CLI/UI/Notebook
                    │  REST/gRPC (SSO + mTLS)
                    ▼
        Control Plane (API GW): Pipeline API │ Registry API │ Deploy API │ Quota
        │                    │                        │
        ▼                    ▼                        ▼
 Orchestration        Model Registry &         Deployment Controller
 (Argo/Airflow DAGs)  Experiment Tracking      (K8s operator: canary, rollback)
        │                    │                        │
        ▼                    ▼                        ▼
 Compute Scheduler    Artifact Store            Model Serving Fleet
 (K8s + Volcano,      (S3, content-addressed,   (Triton/TorchServe/KServe
  gang-scheduled)      versioned checkpoints)    + CPU tabular fleet)
        │                                               │
        ▼                                               ▼
 GPU/CPU Fleet   ◄──── Feature Store ─────────►  Studio game services /
 (on-prem +           Online (Redis/DynamoDB)     matchmaking calling
  cloud burst)         Offline (Iceberg/Parquet)   inference API
        ▲                    ▲
        │                    │
 Training Data Lake ───► Feature Pipelines (Spark/Flink, batch+streaming)
        ▲
        │
 Player Telemetry Kafka (EA-wide, ~2M events/sec)

 Cross-cutting: Observability (Prometheus/Grafana/Jaeger), Drift Detection,
 Cost/Quota Service, IAM (SSO+RBAC), Alerting (PagerDuty)
```

## 8. Low-Level Components

- **API Gateway**: authn/authz, routing, quota/rate limiting. Stateless pods behind L7 LB, scale on request rate.
- **Orchestration Engine** (Argo on K8s): DAG execution for pipelines. YAML/Python SDK; sharded controller by tenant namespace.
- **Compute Scheduler** (Volcano/Kueue): gang-scheduling for multi-GPU jobs, fair-share queueing, preemption. PodGroup CRDs + ResourceQuota per namespace; cluster autoscaler on queue depth.
- **Model Registry**: versioned metadata, lineage (data hash, commit, hyperparams, metrics). REST API over relational metadata store + object storage. Read-heavy, scale via read replicas.
- **Deployment Controller** (custom K8s operator): reconciles desired serving state (canary %, replicas) via `ModelDeployment` CRD.
- **Feature Store**: online low-latency serving + offline point-in-time training data. `get_online_features()` / `get_historical_features()`. Online scales as sharded KV; offline as Spark compute.
- **Model Serving Fleet**: hosts inference endpoints, batching, multi-model hosting, autoscaling. Per-model Deployment + HPA; shared GPU pool for low-QPS models.
- **Drift Detection Service**: compares production feature/prediction distributions vs. training baseline via Kafka inference logs; Flink job, scales with log volume.
- **Cost/Quota Service**: tracks GPU-hour/storage per tenant, enforces quotas, nightly chargeback rollups.

## 9. API Design

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/pipelines` | POST | Submit pipeline DAG |
| `/v1/pipelines/{id}/runs` | POST | Trigger a run |
| `/v1/pipelines/{id}/runs/{run_id}` | GET | Poll run status |
| `/v1/models` | POST | Register model version |
| `/v1/models/{name}/versions/{v}` | GET | Fetch metadata + lineage |
| `/v1/models/{name}/versions/{v}/promote` | POST | Promote to staging/prod |
| `/v1/deployments` | POST | Create serving deployment (canary config) |
| `/v1/deployments/{id}` | PATCH | Update traffic split / rollback |
| `/v1/deployments/{id}` | DELETE | Tear down endpoint |
| `/v1/features/online` | POST | Batch fetch online features |
| `/v1/features/historical` | POST | Point-in-time offline join |
| `/v1/models/{name}/predict` | POST | Inference call |
| `/v1/quota/{tenant}` | GET | Current quota usage |

Example — register model:
```json
POST /v1/models
{
  "name": "fifa-churn-predictor",
  "version": "2026.07.08-1",
  "framework": "xgboost",
  "artifact_uri": "s3://ml-platform-registry/fifa-churn/2026.07.08-1/model.tar.gz",
  "training_run_id": "run-8f2a1c",
  "metrics": {"auc": 0.881, "logloss": 0.312},
  "tenant": "fifa-liveops"
}
```

Versioning: `/v1/` URI-path major version; model versions immutable (`{date}-{seq}`); breaking changes ship as `/v2/` with 6-month deprecation on `/v1/`.

## 10. Database Design

| Store | Type | Used for | Partition Key |
|---|---|---|---|
| Registry metadata | PostgreSQL (multi-AZ) | Model/version/lineage/experiment metadata | `tenant_id` |
| Artifact store | S3-compatible, content-addressed | Checkpoints, dataset snapshots | content hash |
| Offline feature store | Iceberg on S3 (Spark/Trino) | Historical point-in-time features | `event_date` + `entity_type` |
| Online feature store | DynamoDB/Redis | Low-latency inference reads | `entity_id` hash |
| Pipeline/run metadata | PostgreSQL | DAG defs, run/task status | `tenant_id` |
| Metrics/time-series | Prometheus + Thanos/Mimir | Infra + model quality metrics | per-tenant remote-write |
| Audit log | ClickHouse | Access logs, deployments, compliance | `date` |

Why: Postgres for registry needs relational integrity (model→run→dataset→deployment FKs). Iceberg for offline features needs cheap scans + time-travel for point-in-time correctness. KV for online needs single-digit-ms point lookups. ClickHouse for high-ingest append-only audit queries.

## 11. Caching

| Cached | Layer | Strategy |
|---|---|---|
| Online feature values | Redis in front of DynamoDB | Cache-aside, TTL matches freshness SLA |
| Hot model artifacts | Local NVMe on serving nodes | Write-through on deploy, invalidate on promote |
| Registry metadata reads | In-process LRU + Redis | Cache-aside, TTL 60s |
| Deterministic batch inference results | Redis, keyed `(model_version, input_hash)` | TTL until next scoring cycle |
| Feature schema/config | In-memory per pod | Push-based via K8s informer watch |

Invalidation: registry/deployment changes emit events (`ml-platform.registry.events`) for active invalidation rather than relying on TTL alone — critical because a stale model-version pointer post-rollback is a correctness bug, not just staleness.

## 12. Queues & Async Processing

| Queue | Purpose | Delivery | DLQ handling |
|---|---|---|---|
| `pipeline.run.requests` | Async trigger pipeline runs | At-least-once | 3 retries → DLQ, alert owner |
| `training.job.submissions` | Hand off to scheduler | At-least-once | Quota-exceeded → DLQ + Slack |
| `model.deploy.requests` | Deployment reconciliation | At-least-once, idempotent by `deployment_id` | 5 retries + backoff → DLQ, page if prod-tier |
| `feature.pipeline.triggers` | Kick off feature recompute | At-least-once | DLQ + auto-retry next window |
| `inference.log.stream` | Async prediction logging | Best-effort (can drop under backpressure) | Sampled DLQ for debugging |

Exactly-once is not attempted end-to-end; consumers are idempotent (dedup by `run_id`/`deployment_id`/content hash), so at-least-once + idempotency yields effectively-exactly-once. A DLQ-inspector service classifies failures (transient vs. permanent) and auto-replays or flags for triage.

## 13. Streaming & Event-Driven Architecture

| Topic | Producer | Consumers | Schema (key fields) |
|---|---|---|---|
| `player.telemetry.raw` | Game clients/servers | Feature pipelines, data lake | `player_id, session_id, event_type, event_ts, payload` |
| `ml-platform.registry.events` | Registry service | Cache invalidators, audit logger | `event_type, model_id, version, tenant, ts` |
| `ml-platform.deployment.events` | Deployment controller | Monitoring, cost, drift services | `deployment_id, model_id, traffic_split, replicas, ts` |
| `inference.predictions.log` | Serving fleet | Drift detection, offline eval, future training data | `model_id, version, request_id, features_used, prediction, latency_ms, ts` |
| `training.job.status` | Compute scheduler | Orchestration, notifications | `job_id, tenant, status, gpu_hours_consumed, ts` |

Each downstream consumer (drift, cost, audit) runs its own consumer group off `inference.predictions.log` so a slow consumer never blocks others. Partition key = `model_id` to preserve per-model ordering for drift windowing.

## 14. Model Serving

- **Triton Inference Server** for GPU/DL models (TensorRT/ONNX/PyTorch backends, dynamic batching, multi-model on shared GPU). **KServe + lightweight CPU server (MLServer)** for tabular XGBoost/LightGBM.
- Dynamic batching window 5-10ms for DL models — trades small latency for throughput.
- Long tail of low-QPS studio models share GPU pools via Triton's concurrent model instances rather than 1 GPU/model (median 200 QPS would waste ~90% of a dedicated fleet).
- Hardware: L4/T4-class GPUs for inference (A100/H100 reserved for training); standard compute instances for tabular CPU fleet.
- Canary + shadow serving via the deployment controller's traffic-split.

## 15. Feature Store

- Online (Redis/DynamoDB) for low-latency inference; offline (Iceberg/Parquet via Spark/Trino) for training-set generation. Same feature definitions (Feast-style) compile to both paths, avoiding train/serve skew.
- Point-in-time correctness: offline joins use `as_of_timestamp` per training example so features reflect only data available before the label event (prevents leakage), via Iceberg time-travel snapshots.
- Freshness varies: real-time features refreshed via streaming (Flink) within seconds; slow features (30-day rolling spend) refreshed nightly.
- Each feature definition is versioned; models pin a feature-view version at training time for reproducibility.

## 16. Vector Database

Narrow scope — used only by content recommendation (in-game store, cross-title suggestions) and similarity-based anti-cheat/fraud tenants (embedding player behavior to find near-duplicate cheat signatures).

- pgvector for smaller tenants (co-located with existing Postgres); standalone Milvus/OpenSearch-kNN for >100M-vector tenants.
- HNSW indexing (not IVF-PQ) — supports incremental inserts, needed since player embeddings update continuously rather than batch-loading once.
- Not a platform-wide requirement; most tenants (churn/LTV/matchmaking) never touch it.

## 17. Embedding Pipelines

For the recommendation/anti-cheat tenants above.

- Player/item embeddings from two-tower or transformer encoders, trained via the standard training pipeline (Section 19); written to vector DB via an `embedding-sync` stage.
- Nightly full re-embed of active players (~40M); streaming incremental updates (Flink) for players with new session activity within the last hour.
- Embedding drift monitored like model drift (Section 21) — centroid shift signals upstream behavior change or a broken pipeline.

## 18. Inference Pipelines

Online prediction lifecycle (e.g., matchmaking skill-score call): Game Server → API Gateway (authN, rate-limit) → Feature Store (`get_online_features`, p99 <10ms) → Model Server (dynamic batch, GPU/CPU infer, p99 <30ms) → response, plus async fire-and-forget log to `inference.predictions.log` for drift/future training.

- Total budget: p99 < 50ms at the platform hop (gateway + feature fetch + inference), excluding game-server-to-gateway network (studio-owned budget).
- Failure handling: feature-store timeout → serve cached/default values (graceful degrade); model-server timeout → circuit breaker falls back to previous stable version or a rule-based default.

## 19. Training Pipelines

- Data prep: Spark jobs join offline feature store against label events with point-in-time correctness, output a versioned Iceberg snapshot pinned in lineage.
- Orchestration: Argo DAG `data_prep → train → evaluate → register`, each stage a container.
- Distributed training: PyTorch DDP/Horovod across multi-node GPU pool, gang-scheduled via Volcano (all-or-nothing — a 64-GPU job doesn't start with only 40 available).
- HPO: Ray Tune/Optuna as a pipeline stage, launches parallel trial pods, reports to experiment tracking.
- Reproducibility: every model links immutably to `(code_commit_sha, data_snapshot_id, hyperparameter_config, image_digest)`.

## 20. Retraining Strategy

| Trigger | Example | Action |
|---|---|---|
| Scheduled | Weekly churn retrain | Argo CronWorkflow |
| Data drift | Feature PSI > 0.2 vs. baseline | Auto-trigger retrain, notify owner |
| Concept drift | Live AUC drops > 3% (delayed-label backtest) | Auto-retrain; page owner if drop > 8% |
| Manual | New feature, labeling bug fix | Triggered via API/UI |
| Live-event driven | Major content patch shifts behavior | Pre-scheduled retrain timed to release |

Cadence: fast-moving (matchmaking, fraud) — daily/near-real-time; medium (churn/LTV) — weekly; slow (monetization) — monthly.

## 21. Drift Detection

- **Data/feature drift**: PSI and KL-divergence per feature, rolling 7-day window vs. training baseline. PSI > 0.1 = warn, > 0.2 = trigger retrain.
- **Concept drift**: needs delayed labels (e.g., churn realized 14 days later); rolling AUC/logloss on recent labeled sample. Relative drop > 3% = warn + auto-retrain; > 8% = page on-call (likely pipeline break).
- **Prediction drift**: proxy when labels are delayed/unavailable — KS-test on output score distribution vs. training-time distribution, an earlier warning signal.
- Implementation: Flink job consumes `inference.predictions.log`, computes rolling per-model stats, emits to Prometheus; drift score shown on model's dashboard.

## 22. Monitoring

| Category | What's monitored |
|---|---|
| Infra | GPU/CPU utilization, queue depth, node health, storage IOPS |
| Serving | Request rate, latency (p50/p95/p99), error rate, batch size, GPU memory |
| Model quality | Drift scores, rolling AUC/logloss/RMSE, prediction distributions |
| Pipeline health | DAG success rate, stage duration, row-count anomalies |
| Business | Downstream impact (retention conversion, match quality, cost-per-prediction) |
| Platform ops | Quota usage, gateway error rate, registry availability, rollout success rate |

Per-tenant Grafana dashboards auto-generated from deployment metadata; platform-wide fleet-health dashboard for SRE.

## 23. Alerting

| Alert | Condition | Routing |
|---|---|---|
| Serving endpoint down | Availability < 99.95% / 5 min | Page model-owning team |
| Latency SLA breach | p99 > 2x target, 5 consecutive min | Page model owner (or platform if fleet-wide) |
| Severe concept drift | AUC drop > 8% | Page model owner |
| GPU queue starvation | Priority-job wait p95 > 30 min | Page platform infra on-call |
| Quota near-exhaustion | Tenant at 90% GPU-hour quota | Slack notify (non-paging) |
| Control-plane down | 5xx rate > 5% | Page platform on-call, highest severity |
| DLQ backlog | Depth > 1000 or growing 15 min | Page platform on-call |
| Cost anomaly | Daily spend > 150% of 7-day avg | Slack notify + weekly review |

On-call is two-tier: platform infra owns control plane/scheduler/registry; model owners own their model's quality/business alerts.

## 24. Logging

- Structured JSON logs (`trace_id`, `tenant_id`, `model_id`, `request_id`, `ts`, `level`, `message`) standardized platform-wide.
- PII: platform-layer logs use a salted, tenant-scoped `player_id_hash`, never raw player_id; raw PII stays in the governed data lake with stricter access.
- Raw per-request feature vectors logged only in a restricted, audited store with 30-day TTL; aggregate/statistical feature logs by default.
- Retention: debug logs 30 days; audit logs 2 years (ClickHouse); prediction logs 90 days then aggregated-and-purged.
- EU player data logs stay in EU region storage, never replicated to US, per GDPR.

## 25. Security

- AuthN: SSO (SAML/OIDC via Okta) for humans; mTLS + short-lived SPIFFE/SPIRE service identities for service-to-service.
- AuthZ: RBAC scoped per tenant namespace; cross-tenant reads require explicit data-sharing grants.
- Encryption: at-rest (per-tenant KMS keys), in-transit (TLS 1.3, mTLS internal).
- Threat model:
  - **Exfiltration** (compromised credential pulling another studio's artifacts) → per-tenant IAM scoping, bucket policies, anomaly-alerted audit logging.
  - **Training data poisoning** (adversarial injection to bias a model, e.g. whitelist cheating) → pre-training schema/statistical validation, lineage tracing, anomaly detection on training stats.
  - **Model extraction via API abuse** → rate limiting, per-key query budgets, output rounding for sensitive models.
  - **Supply-chain** (malicious training container) → image scanning, signed images, restricted base-image allowlist.
  - **GPU abuse** (cryptomining disguised as training) → resource quotas, anomaly detection on usage vs. declared job type.

## 26. Authentication

- End-user: OIDC via Okta → short-lived JWT (15 min) + refresh; CLI/SDK use device-code flow.
- Service-to-service: mTLS with SPIFFE/SPIRE workload identities, auto-rotated certs (24h TTL), no static secrets.
- Game-server-to-inference-API: API key + mTLS at the edge, scoped per-title to quota/rate-limit policy.
- Cross-region: regional identity federation with residency-aware policy checks.

## 27. Rate Limiting

- Token bucket per API key/tenant at the gateway — smooths bursts, enforces sustained caps.
- Default 5,000 QPS/tenant for inference API (quota increase reviewed against capacity); control-plane APIs capped lower (~100 req/s).
- Additional per-model limits protect shared multi-model GPU pools from one noisy model starving others.
- Burst allowance: 2x sustained rate for up to 10s (covers live-event spikes).
- Over-limit: HTTP 429 + `Retry-After`; tier-1 models (anti-cheat, matchmaking) get priority lanes.

## 28. Autoscaling

- Serving HPA scales on `inference_queue_depth` and `p99_latency` (CPU% is a poor proxy for GPU-bound work). Scale out at queue depth > 20/replica sustained 30s.
- VPA right-sizes training job CPU/memory requests from historical usage.
- KEDA scales feature-pipeline consumers on Kafka consumer-group lag.
- Cluster autoscaler scales GPU node pools on Volcano queue depth; separate spot (aggressive, interruptible training) vs. on-demand (conservative, prod-serving) pools.
- Serving scale-down cooldown (10 min) longer than scale-up (1 min) to avoid flapping on bursty traffic.

## 29. Cost Optimization

- **Spot/preemptible**: interruptible, checkpointed training on spot pool — targets 60%+ of routine GPU-hours at 60-70% discount; checkpoint every 10 min to bound preemption loss.
- **Dynamic batching**: 3-5x GPU throughput gain for DL models vs. per-request execution.
- **Multi-model serving**: shares GPUs across the low-QPS long tail — cuts long-tail serving GPU count by ~80% vs. 1 GPU/model.
- **Distillation/quantization**: large DL models distilled/quantized to INT8/FP16 for serving, enabling cheaper instance types.
- **Feature caching**: cuts redundant online feature-store reads/costs.
- **Tiered storage**: cold artifacts/snapshots moved to infrequent-access storage after 90 days.
- **Idle detection**: auto-terminate dev/notebook GPU sessions idle > 30 min.
- **Reserved baseline capacity**: on-prem fleet sized to steady-state median, cloud burst absorbs spikes — cheaper amortized than on-demand baseline.

## 30. Operational Concerns

At SDE2 scope, treat this as a checklist: **backups** (automated snapshots of registry/feature store/stateful services with a tested restore path), **rollback** (every deploy revertible to last-known-good in one command), **canary/blue-green rollout** (shift small traffic %, watch error rate and key metrics, then ramp), **observability** (dashboards + alerts on latency, error rate, top model-quality signals, wired to on-call). Multi-region active-active topology and Terraform/K8s manifest specifics are Staff/Principal-level concerns — know they exist, don't rehearse them.

## 31. Why This Architecture

- Kubernetes-native scheduling (Volcano/Kueue) over bespoke Slurm — EA already runs K8s fleet-wide, reusing existing ops/IAM/observability rather than a parallel Slurm practice.
- Control plane (registry/orchestration) separated from data plane (serving/training compute) so each scales and fails independently — a registry outage shouldn't take down running inference.
- Feast-style unified feature definitions target the single biggest real-world ML bug class: train/serve skew.
- Multi-tenant-by-design (namespace-per-tenant, quota-enforced) matches EA's 40+ semi-autonomous studios — a shared monolith fights studio autonomy, full per-studio platforms fight cost/governance goals.
- Canary-by-default with drift-aware gates targets the game-industry failure pattern where a model looks infra-healthy (normal latency/errors) while being quality-broken (e.g., bad skill scores after a bad training run) — pure infra health checks miss this.

## 32. Alternative Architectures

| Alternative | Why rejected / when preferred |
|---|---|
| Fully managed cloud ML platform (SageMaker/Vertex) | Rejected as primary: sunk on-prem GPU capex, vendor lock-in at 40-studio scale, pricing gets expensive at EA's volume. Preferred for a smaller company without existing GPU capex or platform headcount. |
| Per-studio independent stacks (status quo) | Rejected: duplicated effort, no feature reuse, poor utilization (20-30% vs. 65%+ target). Fine only for a conglomerate with zero cross-unit data sharing needs. |
| Serverless-only inference | Rejected for latency-critical traffic (cold starts violate 50ms p99) and cost at sustained 120K+ QPS. Used as a secondary path for the low-QPS, latency-tolerant long tail. |
| Single global serving region | Rejected: GDPR residency, EU/APAC latency, single point of failure. Acceptable only for a single-geography platform with no residency constraints. |

## 33. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Multi-tenant shared fleet vs. dedicated clusters | Higher utilization, lower cost, central governance | Needs robust quota/isolation; noisy-neighbor risk if gaps exist |
| Feast-style unified feature definitions | Eliminates train/serve skew | Migration cost; added abstraction layer |
| At-least-once + idempotent (not exactly-once) | Simpler, cheaper, more available | Every consumer must implement idempotency correctly |
| Active-active multi-region serving | Low global latency, resilient to regional failure | 2-3x infra cost; cross-region consistency complexity |
| Canary-by-default | Catches bad models before full blast radius | Slower rollout; longer bake period for delayed-label models |
| Spot instances for training | 60-70% cost savings | Preemption risk; not viable for prod-serving nodes |
| Shared multi-model GPU serving | Big utilization gain for long tail | Noisy-neighbor risk; harder per-model cost accounting |
| Centralized platform team, studios own models | Clear separation, scalable ops | Platform team can bottleneck infra changes |

## 34. Failure Modes

| Failure | Scenario | Mitigation |
|---|---|---|
| Feature store read timeout | Redis hot-partitioning during a live event spike | Cache-aside fallback to stale/default value, circuit breaker |
| Bad model silently promoted | Passes offline eval, broken on an uncovered edge case | Canary + drift gates catch pre-rollout; shadow-serving for high-risk models |
| GPU fleet exhaustion | All studios want retrain capacity ahead of a big launch | Priority-tier reservations, cloud burst, fair-share scheduler |
| Registry DB outage | Postgres primary fails | Multi-AZ failover (~15 min RTO); already-deployed models keep serving |
| Cross-region replication lag | Rollback in US-East hasn't hit EU yet | Deployment events drive active invalidation; critical rollbacks push synchronously |
| Cascading retraining storm | Shared upstream feature bug looks like drift everywhere | Rate-limit auto-retrains; correlation check before mass-triggering |
| Poisoned training data | Upstream telemetry bug corrupts a feature for days | Mandatory pre-training validation; lineage isolates and excludes the bad window |

## 35. Scaling Bottlenecks

- **10x (8,000 practitioners, 6,000 endpoints)**: registry Postgres write load from high-frequency HPO metric logging — move metric logging to a time-series store, keep Postgres for durable metadata only. Shared multi-model GPU pools hit scheduling contention as the long tail grows — needs smarter (ILP-based) bin-packing.
- **100x (~5M QPS aggregate)**: single active-passive control plane becomes a bottleneck — needs active-active with multi-master conflict resolution. Online feature store hot-spots on the most popular titles even with hashing — needs dedicated shard pools for the largest titles. `player.telemetry.raw` at tens of millions of events/sec needs re-partitioning and tiered (per-title federated) Kafka clusters.
- **Compute scheduler**: gang-scheduling for 64+ GPU jobs already causes head-of-line blocking under contention; at 10x, needs smarter preemption/backfill scheduling.

## 36. Latency Bottlenecks

p99 = 50ms budget for a typical online inference request:

| Hop | p50 | p99 | Notes |
|---|---|---|---|
| API gateway | 1ms | 3ms | In-memory token bucket, cached JWT |
| Online feature fetch | 3ms | 10ms | Redis cache-aside; tail from cache misses to DynamoDB |
| Inference (CPU tabular) | 2ms | 8ms | XGBoost, small model |
| Inference (GPU DL, batched) | 8ms | 25ms | Batching adds queueing delay for throughput |
| Async logging enqueue | <1ms | 1ms | Fire-and-forget |
| **Total (tabular)** | **~6ms** | **~21ms** | Within budget |
| **Total (DL)** | **~12ms** | **~38ms** | Within budget, tighter — batch window is the main lever |

The feature-store cache-miss tail and DL batching queue wait dominate p99 — first places to check on regression. Game-server→gateway network (10-30ms, studio-owned) is excluded but is why multi-region serving matters for player-experienced latency.

## 37. Cost Bottlenecks

- **GPU compute** (training + serving) dominates, ~$800K+/month — an order of magnitude above storage/networking.
- **Large distributed training jobs** are disproportionate: ~10 large jobs/week (~2,200 GPU-hours/day) vs. many small jobs consuming less in aggregate — classic long-tail-vs-head cost distribution.
- **Idle/low-utilization GPU time** is the largest avoidable cost — utilization at industry-typical 20-30% means ~2-3x more spend than necessary; closing the gap to 65% is the single highest-leverage lever, bigger than spot discounts alone.
- **Cross-region replication** of large DL model artifacts on every promotion is smaller but easy to overlook.
- **Object storage growth** (1.5PB+ before dedup) needs lifecycle tiering to avoid silent multi-year cost creep.

## 38. Interview Follow-Up Questions

1. How do you prevent one tenant's runaway job from starving GPU capacity for a time-critical tier-1 retrain?
2. Walk through how you guarantee point-in-time correctness in the feature store — what prevents label leakage?
3. Drift detection fires a false-positive storm across 50 models — how do you tell real drift from a detector bug?
4. How would you redesign the registry's consistency model moving from active-passive to active-active?
5. One studio's model needs 10ms p99, another is fine with seconds — how does one platform serve both without over-engineering the common case?
6. How do you handle a model trained on data later found to violate data-residency rules?
7. Tradeoff between canary bake-period length and time-to-value — how do you let a data scientist override safely?
8. How do you attribute GPU cost fairly when tenants share a GPU via multi-model serving?
9. What happens to in-flight requests during a blue/green cutover — how do you guarantee zero dropped requests?
10. How would this change if EA acquired a studio insisting on its own ML stack during a transition period?

## 39. Ideal Answers

1. **Runaway job isolation**: Hard per-tenant ResourceQuota at the namespace level, plus a priority-tier capacity pool with preemption rights for tier-1 jobs, so critical work has capacity even under full contention.
2. **Point-in-time correctness**: Every training example carries an `event_timestamp`; offline joins use time-travel/snapshot (`AS OF`) queries so the join engine structurally cannot see future data.
3. **Real drift vs. detector bug**: Many unrelated models drifting simultaneously shifts the prior toward a shared upstream break. Auto-pause mass-retraining via circuit breaker and page platform on-call to check the shared dependency first.
4. **Active-active registry**: Partition writes by tenant-to-home-region to avoid cross-region conflicts for the common case; conflict-free replication only for rare cross-region ops (global promotion); eventual consistency for non-home reads, read-your-writes at home.
5. **Heterogeneous latency needs**: Expose deployment profiles ("real-time" with dedicated capacity/strict gates vs. "batch" with spot-eligible best-effort) selected declaratively at deploy time; sensible defaults, only tier-1 opts into the costlier profile.
6. **Compliance violation**: Quarantine the model, trace lineage to the tainted run/snapshot and every downstream consumer, retrain from a corrected snapshot, treat as P1 — lineage bounds the blast radius.
7. **Canary override**: Documented "expedited canary" with a shorter but non-zero bake period for low-risk model classes, with audit trail. Tier-1 models never skip gates.
8. **Fair cost attribution**: Meter at request level, tag `tenant_id`/`model_id`, attribute cost proportional to actual compute-time consumed, not wall-clock hosting time.
9. **Zero-drop cutover**: Connection draining — LB stops sending new requests to blue at cutover, lets in-flight finish (grace period matched to p99 duration); green is pre-warmed and health-check-passing before the shift.
10. **Acquired studio**: Treat as external-but-trusted tenant onboarding — own namespace with standard isolation, relaxed pipeline requirement via a thin adapter into the registry format, giving fast central governance with a longer migration runway.
