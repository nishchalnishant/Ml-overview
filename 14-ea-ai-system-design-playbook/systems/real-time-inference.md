# Real-Time Inference Platform

## 1. Problem Framing

Design a **low-latency online model serving platform** for EA-scale live-service games — shared infra any studio (FIFA/EA SPORTS FC, Apex, Sims, Battlefield) calls to get a prediction inline with a live request: matchmaking scoring, chat toxicity, dynamic difficulty, churn-triggered offers, anti-cheat scoring.

Key framing:
- **Platform, not a single model** — multi-tenant, many teams onboard many models.
- **Synchronous, request/response**, in the hot path of gameplay/matchmaking — not batch.
- Must support both **GPU-bound** (deep ranking, embeddings, vision anti-cheat) and **CPU-bound** (GBDT churn/fraud) models behind one control plane.
- Optimize for **tail latency and availability** over iteration speed (that's the training platform's job).

## 2. Functional Requirements

- FR1: Accept a prediction request and return a model output synchronously.
- FR2: Support multiple models/versions/frameworks (PyTorch, TF, XGBoost/LightGBM, ONNX) on shared infra.
- FR3: Dynamic **request batching** to improve GPU utilization without breaching latency SLOs.
- FR4: Fetch **online features** from a feature store when the caller doesn't pass full features.
- FR5: Support **canary/shadow traffic** for new model versions.
- FR6: Support **A/B routing** between model variants.
- FR7: Return calibrated confidence alongside point predictions where supported.
- FR8: Async **explainability hook** (feature attribution) for flagged high-stakes decisions (ban/fraud).
- FR9: Graceful **degradation** — return cached/default/heuristic prediction rather than failing the caller.
- FR10: Self-service **model registration/deployment** API (no platform-team-in-the-loop).

## 3. Non-Functional Requirements

| Dimension | Target |
|---|---|
| Latency (p50) | ≤ 15 ms end-to-end |
| Latency (p99) | ≤ 60 ms |
| Latency (p99.9) | ≤ 150 ms (fallback beyond this) |
| Availability | 99.95% per model endpoint |
| Throughput | 250K QPS peak, burst to 400K during live events |
| Consistency | Feature staleness ≤ 1s acceptable; read-your-writes not required |
| Cost | ≤ $0.15 per 10K inferences blended |
| Durability | Prediction logs 30 days min; model artifacts versioned indefinitely |
| Multi-tenancy | One noisy model must not degrade others' p99 by >10% |

## 4. Clarifying Questions

1. Single-game or shared platform? (Assume shared.)
2. GPU vs CPU model mix? (Assume 30% GPU-bound / 70% CPU-bound by count, GPU dominates compute cost.)
3. Do callers pass features, or does platform own retrieval? (Assume both — feature-store path opt-in.)
4. Sync-only, or also batch scoring? (Assume this scopes to sync; batch is a separate pipeline sharing the registry.)
5. Rollback SLA for a bad deploy? (Assume auto-rollback within 5 min of SLO breach.)
6. Compliance constraints (COPPA, GDPR)? (Assume yes — EU data residency required.)
7. Who owns feature correctness? (Upstream teams own pipelines; platform owns online store + serving correctness.)
8. Failure contract with callers? (Fail-open with default score + flag.)
9. Traffic pattern — diurnal or event-driven spikes? (Both; events can cause 3-5x spikes in minutes.)

## 5. Assumptions

1. 120M MAU; ~15M concurrent at global peak.
2. ~1.8 inference calls per player-action event.
3. 400 registered models; 60 receive >80% of traffic (power law).
4. Feature vector: ~150 features, ~2 KB serialized.
5. GPU models batch-eligible, 8ms max batching window.
6. 70% traffic CPU-servable, 30% GPU-required.
7. Artifact sizes: CPU ~50MB avg, GPU ~2-8GB avg.
8. K8s across 3 regions (US-East, US-West, EU-West) plus on-prem GPU base load.
9. Online feature store is existing shared infra (Section 15); this platform is a consumer.

## 6. Capacity Estimation

**QPS:** 15M concurrent × ~0.09 req/s/player ≈ 1.35M raw events/s; ~18% require real-time inference → **≈240K QPS** peak (matches NFR). Top model ~35K QPS; median model ~50 QPS.

**Storage:** 240K QPS × 2KB × 86,400s ≈ 41TB/day raw → ~8.2TB/day compressed (5:1) → **~246TB** for 30-day retention. Artifact store (400 models × 5 versions, mixed sizes) ≈ **~2.5TB**.

**Compute:** GPU traffic 72K QPS ÷ ~800 inf/s per A10G replica (batch=16) ≈ 90 replicas, provision **~130** with headroom. CPU traffic 168K QPS ÷ ~2,000 QPS per pod ≈ 84 pods, provision **~120**. Feature-store reads: ~60% of requests need live fetch → **144K reads/s**.

**Bandwidth:** ~4KB combined payload × 240K QPS ≈ 960MB/s (~7.7 Gbps aggregate, ~2.6 Gbps/region).

**Cost:** GPU fleet ≈ $52K/month, CPU fleet ≈ $17.3K/month. Against ~622B inferences/month, compute cost ≈ **$0.001 per 10K inferences** — well under the $0.15 target, leaving budget for feature store, logging, control plane.

## 7. High-Level Architecture

```
Game Client/Service → API Gateway (authn, rate limit, geo-route)
   → Inference Router (model resolution, A/B, canary, fallback logic)
        ├─ Feature Store Online Reader (Redis/KV)
        ├─ Batching Layer (dynamic micro-batch queue)
        └─ Fallback/Cache (last-known-good, default)
   → GPU Model Servers (Triton/KServe) | CPU Model Servers (Triton CPU/ONNX RT)
        └─ (async) Explainability Service (SHAP/attribution)
   → Response Assembler (calibration, thresholds, logging hook)
        ├─ Caller response (sync return)
        ├─ Prediction Log Store (columnar)
        └─ Kafka → drift/monitoring

Control Plane (out of hot path):
  Model Registry | Deployment Orchestrator | Autoscaler Controller
  Config/Feature-Flag Service | Drift Detector | Retraining Trigger
```

## 8. Low-Level Components

| Component | Responsibility | Scaling Unit |
|---|---|---|
| API Gateway | TLS, authn, rate limit, geo-routing | Stateless, scales on connections |
| Inference Router | Resolve model+version, A/B/canary rules, orchestrate fetch+batch+fallback | Stateless, CPU-bound |
| Feature Store Reader | Low-latency KV lookups | Read replicas, scales on read QPS |
| Batching Layer | Accumulate micro-batches per model within latency window | Scales with replica count |
| GPU/CPU Model Servers | Execute inference, multi-model hosting | Replica count, autoscaled on queue depth/util |
| Fallback/Cache | Serve last-known-good or heuristic default | Scales with router replicas |
| Prediction Log Store | Durable request/response/feature log | Kafka partitions + independent storage scaling |
| Model Registry | Version metadata, lineage, approval state | Small, HA pair |
| Deployment Orchestrator | Canary/blue-green rollout via K8s | Leader-elected singleton |
| Drift Detector | Streaming comparison to training baseline | Scales with partition count |
| Explainability Service | Async attribution for flagged decisions | Scales with queue depth |

## 9. API Design

**Base:** `https://inference.ea-platform.internal/v2`

```
POST /v2/models/{model_name}/versions/{version}:predict
Headers: Authorization: Bearer <service-jwt>, X-Request-Id, X-Tenant-Id
```

Request:
```json
{
  "instances": [{"entity_ids": {"player_id": "p_9182734"}, "features": {"skill_rating": 1423}}],
  "options": {"explain": false, "timeout_ms": 40, "fallback_allowed": true}
}
```

Response:
```json
{"predictions": [{"score": 0.812, "confidence": 0.93, "model_version": "matchmaking:v3.2.1"}],
 "served_by": "gpu-pool-us-east-1c", "latency_ms": 11, "degraded": false}
```

Alias-based resolution (`POST /v2/predict?model_alias=matchmaking-prod`) lets the router pick the active canary-aware version without version pinning.

Async explainability: `POST /v2/models/{model}:explain`, `GET /v2/explanations/{id}`.

Control plane: `POST /v2/registry/models`, `POST .../versions`, `PATCH .../traffic`, `GET .../versions`.

**Versioning:** URL-path major version (`/v2`) for the API; model version is a separate path param following semver, with router support for `latest`/`champion`/pinned.

## 10. Database Design

| Store | Type | Used For | Why |
|---|---|---|---|
| Model Registry | PostgreSQL | Version metadata, lineage, traffic splits | Strong consistency, low volume, relational FKs |
| Online Feature Store | Redis Cluster | Point lookups by entity_id | Sub-ms p99, simple KV, shardable |
| Prediction Log Store | Iceberg/Parquet on S3 | Audit log, drift input | Write-heavy, append-only, cheap at 246TB |
| Explainability Store | Document store | Variable-shape attribution payloads | Schema-flexible per model type |
| Config Store | etcd/Postgres+cache | Canary %, kill switches, rate limits | Strong consistency, fast propagation |

Registry schema: `models(model_id, name, owner_team)` → `model_versions(version_id, model_id FK, semver, artifact_uri, framework, hardware_class, status)` → `traffic_splits(model_id, version_id, weight_pct)`.

**Partitioning:** Prediction logs partitioned by date + bucketed by model_id hash. Feature store sharded by player_id (consistent hashing). Registry not sharded (low volume, primary + regional read replicas).

## 11. Caching

| Cache | What | Strategy | Invalidation |
|---|---|---|---|
| Feature cache (router-local LRU) | Recent feature vectors | Cache-aside, ~500ms TTL | TTL + pub/sub purge on write |
| Model artifact cache | Loaded weights in memory | Write-through on deploy | LRU eviction under memory pressure |
| Last-known-good prediction | Most recent success per entity+model | Write-through on success | TTL 5 min, fallback-only |
| Config cache | Traffic-split/kill-switch flags | Cache-aside + push invalidation | ~200ms propagation |

Cache-aside for read-heavy, staleness-tolerant data (features, config). Write-through where the fallback/degraded path needs correctness the moment it's needed.

## 12. Queues & Async Processing

| Queue | Payload | Semantics | DLQ |
|---|---|---|---|
| Prediction-log ingestion (Kafka) | request+response+features | At-least-once, idempotent via request_id | Retry 5x → DLQ, alert if depth > 10K |
| Explainability jobs | model, instance, explanation_id | At-least-once | Retry 3x → DLQ, page if backlog > 15 min |
| Deployment events | version promoted/rolled back | Exactly-once (idempotent orchestrator) | Halt pipeline, alert deploy owner — no silent DLQ |
| Drift computation | windowed batch refs | At-least-once, idempotent | Retry on next window |

Exactly-once only matters for deployment state transitions (double-promote/rollback is dangerous); data-plane queues tolerate at-least-once since duplicates are cheap.

## 13. Streaming Architecture

| Topic | Producer | Consumers |
|---|---|---|
| `inference.predictions.v1` | Response Assembler | Log sink, Drift Detector, Explainability trigger |
| `inference.errors.v1` | Router (on fallback) | Alerting, SRE dashboards |
| `feature-store.updates.v1` | Upstream feature pipelines | Router cache invalidator |
| `registry.deployment-events.v1` | Deployment Orchestrator | Autoscaler, Drift baseline reset, Audit log |

Drift-detector consumer group partitions by model_id (ordering matters for windowed stats). Log-sink group is high-parallelism, order-agnostic. Explainability-trigger group filters `flagged=true` only.

## 14. Model Serving

**Framework:** NVIDIA Triton as common serving runtime — supports PyTorch/TF/ONNX/custom Python backends (GBDT via thin wrapper), with native dynamic batching and multi-model concurrent execution.

- **Multi-model serving:** models below a QPS threshold are bin-packed onto shared GPU replicas by memory footprint + expected QPS — critical since 340 of 400 models are long-tail/low-QPS.
- **Dynamic batching:** per-model `max_queue_delay` tuned to each model's latency budget (e.g., 8ms window for 15ms-budget models). Preferred batch sizes as GPU-efficient powers of 2.
- **CPU serving:** GBDT models via Triton's FIL backend — no batching benefit needed, already sub-ms; horizontal pod scaling handles throughput.
- **Hardware:** A10G-class GPUs (cost-efficient inference GPU, not training-grade A100/H100) since workload is latency-bound, not huge-batch-throughput-bound.
- **Warm pool:** every "champion" model kept loaded in ≥2 replicas/region even at near-zero traffic — multi-GB cold load can take seconds, unacceptable mid-request.

## 15. Feature Store

- **Online store:** Redis Cluster sharded by player_id, ~150 real-time features/entity, refreshed by streaming pipelines, target staleness ≤1s.
- **Offline store:** columnar warehouse for training, same feature definitions computed in batch.
- **Point-in-time correctness:** feature registry guarantees streaming and batch paths are logically equivalent; training joins on event timestamp to avoid leakage. This platform always reads *current* online values — point-in-time correctness matters for training, but serving depends on the two paths staying consistent (skew is a drift signal, Section 21).
- Staleness > 5s triggers router-side fallback to cached last-known-good features rather than blocking.

## 16. Vector Database

**N/A for the core path** — most models (matchmaking, churn/fraud, DDA) use structured/tabular features, not similarity search.

Exception: models consuming embeddings as inputs fetch a **precomputed vector directly from the feature store** (just another feature) rather than doing ANN search at inference time. A true nearest-neighbor recommendation use case (e.g., "similar cosmetic items") is a distinct system with its own vector DB — out of scope here since ANN search changes the latency profile substantially.

## 17. Embedding Pipelines

**Partially applicable.** Some models consume embeddings as inputs, but this platform doesn't compute them inline — that would duplicate work and add latency.

- Embeddings (player-behavior, item, text) are computed **offline/streaming upstream** (batch nightly for slow-moving entities, near-real-time for fast-moving ones), written into the feature store as fixed-size vectors.
- This platform's only embedding-adjacent job: serving models whose first layer is an embedding lookup baked into the model artifact — just part of the graph Triton executes.
- Rationale: decouples expensive embedding refresh from the strict-latency path and lets multiple models reuse the same precomputed embedding.

## 18. Inference Pipeline (Request Lifecycle)

```
t=0ms    Client → API Gateway
t=0.5ms  Gateway: TLS, authn, rate-limit, region route
t=1ms    Router receives request, resolves model alias→version (cached)
t=1.5ms  If features not provided → async call to Feature Store Reader
t=1.5-4ms  KV lookup by entity_id (parallel fan-out if multi-entity)
t=4ms    Router assembles input tensor, enqueues into batching layer
t=4-12ms Batching layer accumulates micro-batch (up to window) or flushes early
t=12ms   Batch dispatched to Model Server (GPU or CPU pool)
t=12-20ms Model Server forward pass
t=20ms   Response Assembler: calibration/thresholding, attach metadata
t=20.2ms Fire-and-forget publish to inference.predictions.v1 (non-blocking)
t=20.5ms Response returned to client
```
Total ≈ 15-20ms, matching p50 target. On failure past t=1ms: router checks `fallback_allowed` → serves last-known-good or static default, tags `degraded:true`, still publishes to `inference.errors.v1`.

**Fallback path:** Primary path (timeout=40ms) → on timeout/error → Last-Known-Good cache lookup → on miss → Static Default/Heuristic. Each successful branch returns immediately; only the final default is guaranteed-available.

## 19. Training Pipelines

- **Data prep:** offline feature store joins (point-in-time correct) → versioned training datasets → schema/stat validation before training.
- **Orchestration:** Kubeflow/Argo DAGs — extract → train → eval → validation gate → registration. Triggered by schedule, drift signal, or manual trigger.
- **Distributed training:** PyTorch DDP for large GPU deep-ranking/DDA models; GBDT trains single-node (or Dask-distributed for very large tabular sets).
- **Handoff:** successful runs register a new `model_version` as `staged` — no auto-promotion; promotion goes through canary (Section 20/33).
- This platform is a **consumer** of training-pipeline output; training itself is a separate chapter, referenced here only at the artifact_uri/metadata interface boundary.

## 20. Retraining Strategy

**Cadence:** fast-drifting models (matchmaking, live-event DDA) weekly; medium-drift (churn, toxicity) bi-weekly/monthly; stable models (fraud heuristics) quarterly or trigger-only.

**Triggers:** Drift Detector PSI alert on a top-20 model auto-kicks a retrain job (still needs human approval to promote); concept-drift business-metric degradation over 3 consecutive days; major content patch/season launch (proactive scheduled retrain); manual trigger by model owner.

## 21. Drift Detection

| Drift Type | Metric | Threshold | Action |
|---|---|---|---|
| Data drift (numeric) | PSI vs training baseline, daily | >0.2 alert, >0.3 page + auto-retrain | Alert owner; auto-kick retrain if top-20 model |
| Data drift (categorical) | JS divergence vs baseline | >0.1 | Alert model owner |
| Concept drift | Rolling online accuracy proxy | Error increase >15% over 7-day baseline | Page on-call, flag for review |
| Prediction drift | KL divergence of output distribution | >0.15 | Alert; check feature pipeline health first (usually a bug, not real drift) |
| Feature-serving skew (online vs offline) | Distribution comparison, same entity/timestamp | Mismatch >1% of sampled requests | Page platform team — feature-store bug, higher urgency |

Computed by streaming Drift Detector consuming `inference.predictions.v1`, windowed hourly/daily; baselines reset at each model promotion.

## 22. Monitoring

**Infra:** GPU util %, memory occupancy, batch queue depth/fill-rate, pod restarts, request rate, error rate, latency percentiles per model+version+region.

**Model quality:** drift metrics (Section 21), calibration drift, confidence distribution, fallback/degraded rate, champion-vs-challenger deltas during canary.

**Business:** match-quality proxy, moderation false-positive appeals, churn precision@k, fraud $ prevented vs friction, cost-per-10K-inferences trend (catches silent regressions).

## 23. Alerting

| Condition | Threshold | Severity |
|---|---|---|
| p99 latency breach | >60ms for 5 min | High — page on-call |
| p99.9 latency breach | >150ms for 2 min | Critical — page + auto circuit-break |
| Error rate | >1% over 5 min | High |
| Fallback/degraded rate | >5% for 10 min (page if >15%) | Medium |
| GPU util | >90% for 10 min | Medium — check autoscaler |
| Drift PSI >0.3 (top-20 model) | immediate | Medium — notify owner, auto-retrain |
| Feature-serving skew >1% | immediate | High — data-correctness issue |
| DLQ depth | >10K | Medium |
| Deployment rollout failure | any | High — page deploying engineer |

Routing: platform infra → SRE rotation; model-quality → model-owner team on-call (via Registry metadata).

## 24. Logging

- Structured JSON logs (via Kafka topics) with `request_id`, `model_name/version`, `latency_ms`, `served_by`, `degraded` — never free-text for anything queryable.
- **PII:** player_id pseudonymized (salted hash) in long-retention store; short-retention (7-day) raw-ID mapping exists only for incident debugging.
- Chat/text features never logged raw long-term — only derived scores; raw text max 48h in restricted debug store for appeals.
- **Retention:** logs 30 days hot, 1 year cold (compliance), debug logs 7 days, explanations 90 days.
- **Residency:** EU traffic logs stay in EU storage (GDPR); only anonymized drift metrics replicate globally.

## 25. Security

**Threats:**
- **Model exfiltration** (weight theft or query-based stealing): mitigated by IAM-locked artifact store, per-tenant rate limiting, anomaly detection on systematic query patterns.
- **Adversarial input manipulation** (e.g., gaming fraud/anti-cheat scores): security-sensitive features computed server-side by the platform's own feature store, never accepted directly from untrusted clients.
- **Prediction-log exposure:** encryption at rest, strict IAM, PII pseudonymization.
- **Feature-pipeline poisoning:** feature-serving skew monitor + schema/range validation at write time.

**Encryption:** at rest via provider KMS (per-region keys, supports residency + crypto-shredding); in transit via mTLS service mesh.

## 26. Authentication

- **Service-to-service:** mTLS (SPIFFE identity, 24h rotation) plus a signed service JWT carrying `tenant_id`/`owner_team` for router-level authz.
- **End-user:** players never call this platform directly — always via an already-authenticated game-service backend; platform trusts the caller's identity, propagates `player_id` for lookups/audit.
- **Control-plane:** OAuth2 + RBAC via corporate SSO, scoped per `owner_team`.

## 27. Rate Limiting

- Token bucket per (tenant, model_name), enforced at the Gateway.
- Default limit sized at 1.5x provisioned peak QPS, with a hard per-model ceiling protecting shared pools.
- Top-20 models get **dedicated capacity pools** rather than relying on rate limiting alone; rate limiting is the backstop for long-tail tenants sharing pooled capacity.
- On breach: HTTP 429 / `RESOURCE_EXHAUSTED` with `Retry-After` — callers expected to have their own local fallback.

## 28. Autoscaling

- **GPU pools:** HPA on batch queue depth + GPU utilization (not CPU). Scale-out at queue depth >50 for 30s OR GPU util >75% for 1 min. Scale-in conservative (10-min cooldown) given multi-second cold-start.
- **CPU pools:** standard HPA on CPU util (60% target) + request rate, faster scale-in (2 min) since cold-start is sub-second.
- **Predictive pre-scaling:** scheduled replica-floor increases ahead of known live-service events, since reactive autoscaling can't react fast enough to a 3-5x spike within minutes.
- **Warm-pool floor:** champion models never scale to zero; long-tail models scale to a low floor (e.g., 2), not zero.

## 29. Cost Optimization

- **Spot instances** for long-tail CPU pools and non-champion GPU replicas (N+1 redundancy tolerates interruption); champion baseline stays on-demand/reserved.
- **Multi-model bin-packing** is the biggest lever — avoids ~300 idle dedicated GPUs for long-tail models.
- **Dynamic batching tuning** improves throughput per replica, directly cutting GPU-hour spend.
- **Distillation/quantization** (FP16/INT8) on large models where accuracy loss is acceptable — more models per GPU, lower latency.
- **Reserved capacity** for the predictable diurnal floor, spot/on-demand for variable peak.
- **Last-known-good cache** also cuts cost by avoiding redundant compute on repeated identical requests.
- **Storage tiering:** aggressive hot→cold movement of prediction logs, since drift jobs mostly need recent data.

## 30. Operational Concerns

At SDE2 scope, treat as a checklist: **backups** (registry, feature store, tested restore path), **rollback** (one-command revert to last-known-good version), **canary/blue-green rollout** (small traffic shift first, watch error rate + key metrics, then ramp), **basic observability** (dashboards/alerts on latency, error rate, top 2-3 quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level concerns — know they exist, don't rehearse the manifests.

## 31. Why This Architecture

- **Multi-tenant, framework-agnostic serving (Triton)** avoids every team building bespoke serving infra, and enables cost-efficient bin-packing across 340 long-tail models — prohibitively expensive with dedicated infra per model.
- **Decoupling feature/embedding computation from the hot path** keeps the sync request path lean — essential to hit the 15ms p50 budget.
- **Config-driven rollback** (vs. redeploy) makes rollback sub-minute, critical since live-service games can't tolerate a bad matchmaking/anti-cheat model staying live at peak traffic.
- **Region-local hot path** (no cross-region sync calls) is the only way to hit global p99 targets while respecting EU residency.
- **Fail-open degradation** matches the real risk profile — a stale matchmaking score beats a blocked matchmaking request.

## 32. Alternative Architectures

| Alternative | Why Rejected / When Preferred |
|---|---|
| Per-team dedicated serving stacks | Duplicates idle GPU capacity and on-call burden across teams; preferred only for a single-studio system with unique hardware needs that doesn't fit the shared pattern. |
| Serverless/FaaS per-request | Cold-start latency blows the p99 budget, no cross-invocation batching; fine for low-QPS, latency-insensitive internal tools. |
| Embedded/edge inference on client | Rejected as primary since fraud/anti-cheat must be server-authoritative and non-client-manipulable; fine for pure UX-personalization with offline-play needs. |
| Batch-only precomputed scores | Can't serve session-context-dependent decisions (live DDA, chat toxicity); used as a *complement* — the real system is a hybrid, this platform covers only the session-live subset. |

## 33. Tradeoffs

| Decision | Benefit | Cost/Risk |
|---|---|---|
| Multi-model GPU bin-packing | Big cost savings on long tail | Noisy-neighbor risk, needs isolation tuning |
| Fail-open degradation | High caller-perceived availability | Silent accuracy loss if `degraded` isn't monitored |
| Region-local feature stores | Meets latency + residency | Feature drift across regions for roaming players mid-session |
| Config-based instant rollback | Sub-minute recovery | Requires keeping N-1 version warm — extra memory cost |
| Dynamic batching | Big GPU throughput win | Adds latency directly into the critical path |
| Shared platform vs per-team stacks | Ops/cost efficiency, consistent SLOs | Platform team becomes a critical dependency |
| Synchronous feature fetch in hot path | Better DX, consistent features | Extra latency-critical dependency (mitigated by fallback cache) |

## 34. Failure Modes

| Scenario | Symptom | Mitigation |
|---|---|---|
| Feature Store partial outage | Elevated latency/timeouts | Fallback to cache/last-known-good; circuit breaker after N timeouts |
| GPU OOM from over-packing | Pod restarts, latency spike for co-located models | Per-model memory quotas, OOM-kill-rate alerting |
| Bad model silently degrades quality | Latency/error SLOs pass, business metric regresses | Business-metric guard gate in canary + drift monitoring post-rollout |
| Kafka broker outage | Logging fails, drift jobs starve | Logging is non-blocking for response; DLQ/retry; drift jobs alert on stale input |
| Registry DB primary failure | Deploys/registrations blocked | Routers run off cached config; automated failover promotes replica |
| Thundering herd (event spike) | Autoscaling lags, latency breaches | Predictive pre-scaling, warm-pool floor, rate limiting isolates spillover |

## 35. Scaling Bottlenecks

**At 10x (2.4M QPS):** Feature store read layer becomes the first bottleneck — needs deeper sharding and rack-local read replicas. Batching windows need to become per-model adaptive as arrival rate rises. Registry config-push fan-out needs hierarchical pub/sub instead of flat fan-out.

**At 100x (24M QPS):** GPU capacity itself becomes the binding constraint, likely requiring multi-cloud GPU sourcing or heavier distillation/quantization. Kafka partition counts become insufficient for consumer parallelism. Single Registry primary becomes a write bottleneck, needing sharding by owner-team or a multi-primary design.

## 36. Latency Bottlenecks

**p50 budget (~15ms):** Gateway 0.5ms, router resolution 0.3ms, feature fetch 2.5ms (dominant non-compute cost), batching wait 4-8ms (biggest tunable lever), GPU forward pass 3-5ms, response assembly 0.3ms.

**p99 tail sources:** feature-store hot-spotting can spike fetch to 15-20ms; batching layer can add another 10-15ms under GPU saturation; a cold model reload (not warm on an autoscaled/evicted replica) is the single largest tail risk, mitigated by the warm-pool floor; cross-AZ jitter adds a few ms.

**Biggest lever:** the batching window is the most directly controllable latency-vs-throughput dial — tightening it recovers latency at direct GPU-cost expense.

## 37. Cost Bottlenecks

- GPU fleet dominates cost (~$52K vs $17.3K/month CPU) despite being only 30% of QPS — deep models are far more compute-expensive per inference.
- Under-tuned bin-packing is the biggest risk of cost regression as more low-QPS models onboard.
- Warm-pool floors (never-scale-to-zero × 3 regions × N+1) are a fixed cost independent of traffic — worth periodic audit of which models truly need it.
- Prediction log storage at 246TB/30-days is a common latent cost creep if hot-retention isn't enforced.
- Accidental cross-region calls (bug or misconfigured fallback) are a classic silent-bill-creep risk.

## 38. Interview Follow-Up Questions

1. How would this design change if p99 budget were 10ms instead of 60ms?
2. What happens if the Feature Store's Redis cluster fails over mid-request — what does the caller see?
3. How do you decide which models get dedicated GPU capacity vs. share a pool?
4. A drift detector fires a PSI alert on a low-traffic model at 3am — should that page anyone?
5. How would you extend this to a new modality (real-time voice moderation) without a full redesign?
6. What's your strategy if two model versions need incompatible feature schemas during canary?
7. How do you prevent the Model Registry from becoming a deploy-velocity bottleneck?
8. If GPU costs must drop 40% next quarter with no traffic reduction, what are your first three levers?
9. How would multi-region active-active change if EA acquired a studio with its own inference stack?
10. Explain the automated rollback mechanism — what could go wrong with the automation itself?

## 39. Ideal Answers

1. **10ms p99:** Eliminate the batching window almost entirely, push features into a request-scoped cache instead of fetching synchronously, likely co-locating feature store and model server. Trades higher GPU-hour cost for tighter latency.

2. **Redis failover mid-request:** Router's feature-fetch times out against a tight budget; circuit breaker routes to last-known-good cache (`degraded:true`) or static default on a miss — fails fast within single-digit ms rather than blocking on actual failover.

3. **Dedicated vs. shared GPU:** Driven by QPS (top models get dedicated/reserved pools) and business criticality (e.g., fraud gets dedicated capacity even at lower QPS to avoid noisy-neighbor variance). Everything else defaults to shared, bin-packed pools.

4. **3am drift alert, low-traffic model:** No page — route to non-urgent review next business day. Low-traffic models have noisier drift stats and small blast radius. Severity should weigh statistical confidence and business impact, not just threshold breach.

5. **Adding voice/audio moderation:** Router/batching/multi-model-serving/autoscaling infra is modality-agnostic and reusable. New pieces: larger payloads (streaming/chunked gRPC), different GPU sizing (audio models heavier per inference). Control plane (Registry, canary, rollback, drift) extends unchanged.

6. **Incompatible feature schemas during canary:** Router's input-assembly step must be version-aware, resolving feature requirements per specific model_version, with the registry tracking per-version feature-schema dependencies. Both schemas' features must be live in the online store before the new version enters canary.

7. **Registry as bottleneck:** Keep it low-write-volume (metadata + pointers only), read-replica-heavy. Routers cache config locally with periodic refresh, decoupling request-path availability from Registry availability — an outage only blocks new deploys, not existing serving.

8. **Cut GPU cost 40%:** First, tighten bin-packing (move dedicated-but-low-QPS models to shared pools). Second, distill/quantize the highest-QPS models (power-law dominance of GPU-hours). Then increase spot mix and retune batching windows toward throughput where latency has slack.

9. **Acquiring a studio with its own stack:** Treat as migration, not merge-in-place — run the acquired stack as-is, onboard to the shared Registry as control-plane-only first, then incrementally migrate models via the normal canary process. Avoid a big-bang cutover on a live game.

10. **Automated rollback mechanism:** A health-check gate (latency/error/distribution) during canary ramp and post-cutover soak triggers the Deployment Orchestrator to zero out the new version's traffic via config-push. Main risk: the health-check pipeline's own metrics could lag or break, causing false negatives/positives — so its freshness is itself monitored as a meta-metric.
