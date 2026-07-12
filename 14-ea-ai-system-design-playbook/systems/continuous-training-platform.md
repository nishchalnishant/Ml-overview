# Continuous Training Platform

## 1. Problem Framing

EA runs live-service titles (FC, Apex, Sims, Battlefield) where ML models decay fast: matchmaking drifts as metas shift, toxicity classifiers drift as slang evolves, churn/LTV drifts after content drops, fraud models face daily adversarial drift. Manual retraining (pull data, retrain, eyeball metrics, file a ticket) takes 1-2 weeks — too slow when a bad matchmaking model hurts retention within days.

**Goal**: automatically detect drift, retrain on fresh data, validate against gates, shadow-test against real traffic, and promote or roll back — humans in the loop only for exceptions.

**Users**: ML engineers (pipelines/gates), data scientists (model logic, promotion review), SRE/on-call (incidents), game teams (consume predictions via serving, out of scope here).

## 2. Functional Requirements

- FR1: Ingest labeled training data from feature store + telemetry (scheduled + event-driven).
- FR2: Detect data drift (input shift) and concept drift (label/performance shift) per model.
- FR3: Trigger retraining on drift, schedule, or manual request.
- FR4: Orchestrate distributed training (prep → train → eval) with full lineage (data/code/hyperparam versions).
- FR5: Validation gate suite (offline metrics vs. champion, fairness, data quality, regression checks) before promotion.
- FR6: Shadow evaluation — candidate scores live traffic in parallel with champion, no serving impact, compared over a bake window.
- FR7: Staged promotion (shadow → canary % → full rollout) with gate checks at each stage.
- FR8: One-click/automatic rollback to last-known-good version.
- FR9: Model registry with versioning, metadata, audit trail.
- FR10: APIs/UI to inspect drift, approve/reject promotions, force rollback.
- FR11: Support multiple model types (GBTs, deep nets, embedding models) under one orchestration framework.

## 3. Non-Functional Requirements

| Dimension | Target |
|---|---|
| Retrain pipeline latency | < 4 hrs (daily models); < 45 min (fraud) |
| Drift detection latency | < 15 min from event to signal |
| Orchestration control-plane availability | 99.9% |
| Shadow/canary eval availability | 99.95% |
| Throughput | 50+ concurrent pipelines; 200K events/sec peak telemetry |
| Consistency | Strong for registry state transitions; eventual for drift dashboards |
| Cost | GPU training < $150K/month |
| Durability | Zero data loss on datasets/artifacts (object storage) |
| Auditability | Every promotion/rollback traceable to trigger, approver, gate results |

## 4. Clarifying Questions

1. How many model families/titles must this support at once?
2. Acceptable blast radius of a bad promotion — is canary mandatory?
3. Human approval on every promotion, or only borderline gate results?
4. Retrain cadence per model class — hourly (fraud), daily (churn), weekly (recsys)?
5. Do we own the feature store/labeling infra, or integrate with existing ones?
6. Rollback SLA — seconds (flag flip) or minutes (redeploy)?
7. Regulatory constraints (COPPA, GDPR, loot-box rules) on training data or explainability?
8. In-house GPU clusters or cloud reserved+spot?
9. Tolerance for false-positive drift (wasted retrain spend) vs. false negatives (stale prod model)?
10. Does shadow/canary need real traffic replay, or is a held-out offline set enough for some models?

## 5. Assumptions

1. 6 major titles, ~40 production model endpoints.
2. Peak telemetry: 200K events/sec; steady ~30K events/sec.
3. Median model: GBT or small deep net (10M-200M params) on 50-500GB snapshots; one large model class (toxicity, 1-3B param transformer fine-tune).
4. GPU fleet: on-prem A100s (reserved) + cloud spot A10G/A100 burst.
5. Human approval required for full rollout; automated approval allowed for canary if gates pass with margin.
6. Rollback SLA: < 60s via model-version pointer flip, no redeploy.
7. Training data retention: 13 months cold, 30 days hot.
8. Kubernetes-based infra, same family as EA's other platform services.
9. Drift detection samples 1-5% of traffic for cost control; full-fidelity for fraud models.

## 6. Capacity Estimation

- **Telemetry**: steady ~30K events/sec (~30MB/s, ~2.6TB/day); peak 200K/sec (~17TB/day-equivalent).
- **Feature writes**: ~20% of raw events materialize — steady 6K/sec, peak 40K/sec.
- **Training data per cycle**: churn/LTV 500GB (30-day rolling window); matchmaking 50GB/day; toxicity fine-tune 20GB/week.
- **Compute**: GBT jobs are CPU-bound (32-64 vCPU, 20-40 min, ~10 jobs/day on a 256-vCPU pool). Matchmaking deep net: 8×A100, ~90 min/run, daily × 6 titles — queue serially on a 16-32 GPU reserved pool. Toxicity fine-tune: 16×A100, ~4hr, weekly (~1,536 GPU-hrs/month, bursty).
- **Cost**: cloud spot burst ≈ $20-25K/month; total (with on-prem reserved + storage) stays under the $150K/month NFR.
- **Storage**: hot feature store ~78-150TB (30 days); cold archive ~300TB (13 months, tiered to cheap storage); registry artifacts ~1.6TB.
- **Shadow load**: sampled 5-10% of production QPS per model — negligible vs. primary serving fleet.

## 7. High-Level Architecture

```
Game Clients/Servers → Streaming Bus (Kafka: player.events, match.results, chat.messages, model.predictions)
        │                                   │
        ▼                                   ▼
Feature Pipeline (Flink/Spark)      Drift Detection Svc (PSI, KS-test, perf deltas)
        │                                   │ drift signal
        ▼                                   ▼
Offline Feature Store (Delta/S3) ◄── Retrain Orchestrator (Airflow/Argo: schedule + drift + manual triggers)
                                            │ launches
                                            ▼
                                     Training Cluster (K8s + GPU/CPU, Ray/Horovod)
                                            │ candidate model
                                            ▼
                                     Validation Gate Svc (offline metrics, bias, data-quality, regression)
                                     pass │ fail → reject + alert
                                            ▼
                                     Model Registry (versions, lineage, audit trail)
                                            │ "shadow-ready"
                                            ▼
                                     Shadow Evaluation Svc ◄── live prod traffic mirror
                                     pass │ fail → reject + alert
                                            ▼
                                     Canary Controller (1% → 10% → 50% → 100%)
                                            │ gate pass each stage
                                            ▼
                                     Production Model Serving (Triton/KServe)
                                            │ predictions + outcomes
                                            ▼
                                     Rollback Controller (auto on SLO breach)
```

## 8. Key Components

| Component | Responsibility | Scaling Unit |
|---|---|---|
| Streaming Bus (Kafka) | Durable transport for telemetry, results, predictions | Partitions per topic |
| Feature Pipeline | Raw events → online (Redis) + offline (Delta) features | Flink task parallelism |
| Drift Detection | Streaming PSI/KL/KS stats + delayed-label perf deltas per model | Stateless consumers per model-partition |
| Retrain Orchestrator | DAG: prep → train → validate → register; handles all trigger types | Workers autoscale on queue depth |
| Offline Feature Store | Point-in-time correct historical features | Partitioned by title + date |
| Training Cluster | Distributed training (DDP, Horovod, Ray Train) | GPU node pool autoscaler |
| Validation Gate | Holdout metrics vs. champion, fairness slices, data-quality, regression thresholds | Stateless batch workers |
| Model Registry | Source of truth: versions, lineage, stage, audit trail (MLflow-like) | Postgres + object storage |
| Shadow Evaluation | Mirrors live traffic to candidate, logs without serving, compares to champion | Scales with sampled shadow % |
| Canary Controller | Staged traffic ramp with gate checks (latency, error rate, business deltas) | Stateless, service-mesh traffic split |
| Model Serving | Triton/KServe, gRPC+REST | HPA on QPS/GPU util |
| Rollback Controller | Detects SLO/gate breach, flips pointer to last-known-good | Stateless, leader election |

## 9. API Design

Base path `/api/v1`, mTLS (service-to-service) or OAuth2 (human/UI).

| Endpoint | Method | Purpose |
|---|---|---|
| `/models/{id}/versions` | GET | List versions + stage + metrics |
| `/models/{id}/versions/{v}/promote` | POST | Request promotion to next stage |
| `/models/{id}/versions/{v}/rollback` | POST | Force rollback to previous prod version |
| `/models/{id}/drift` | GET | Current drift metrics |
| `/models/{id}/retrain` | POST | Manually trigger retrain |
| `/pipelines/{run_id}` | GET | Pipeline run status |
| `/shadow/{id}/report` | GET | Shadow comparison report |
| `/canary/{id}/status` | GET | Current canary ramp % + health |

Model version strings are immutable and content-addressed (hash of data+code+hyperparams). Breaking API changes require `/v2` with a 90-day dual-run deprecation window.

## 10. Database Design

| Store | Type | Used For |
|---|---|---|
| Model Registry metadata | PostgreSQL | Versions, stages, lineage, approvals — needs ACID for stage transitions |
| Offline feature store | Delta Lake/Parquet on S3 | Point-in-time training snapshots |
| Online feature store | Redis/DynamoDB | Low-latency inference-time lookups |
| Drift metrics | Time-series DB (Timescale/Prometheus) | PSI/KS-stat history per model |
| Event log | Kafka → cold S3/Parquet | Replayable source of truth |
| Shadow/canary results | Columnar (ClickHouse) | High-cardinality per-segment comparisons |
| Audit trail | Append-only Postgres table | Compliance: who approved what, when |

Relational for the registry because promotion/rollback is a state machine needing transactional locking. Columnar for features/drift because those workloads are append-heavy and scan-heavy.

## 11. Caching

| Cached Item | Strategy | Invalidation |
|---|---|---|
| Online features | Cache-aside, Redis | TTL 15 min + explicit on new write |
| Model artifact (serving pods) | Write-through to local NVMe | On new promotion; evicted after 24h grace post-canary |
| Drift threshold config | Cache-aside, in-memory | Pub/sub on config change |
| Registry "current prod version" pointer | Cache-aside at gateway | Near-synchronous pub/sub on promote/rollback — the one cache where staleness directly serves the wrong model |
| Gate baseline metrics | Cache-aside | Invalidated when champion changes |

## 12. Queues & Async Processing

| Queue | Purpose | Delivery |
|---|---|---|
| `retrain.requests` | Trigger queue (drift/schedule/manual) | At-least-once, DLQ + page after 3 retries |
| `training.jobs` | Dispatch specs to K8s job controller | At-least-once, idempotent by `run_id` |
| `validation.results` | Gate results → Registry update | At-least-once, idempotent upsert |
| `shadow.predictions` | Async shadow score logging | At-least-once, dedup by `request_id` |
| `model.promotions` | Fan-out to serving/canary/audit | At-least-once, idempotent consumers, paged if DLQ non-empty |

Exactly-once is deliberately avoided — all consumers are idempotent instead, since duplicate drift signals or gate-checks are harmless once deduped.

## 13. Model Serving

- **Framework**: Triton (deep nets — dynamic batching, multi-framework) wrapped by KServe for K8s autoscaling/canary routing. GBT models served via lightweight ONNX sidecars (Triton is overkill for XGBoost).
- **Batching**: 5-10ms window for latency-sensitive matchmaking; up to 100ms for batch churn/LTV scoring.
- **Multi-model**: one Triton instance hosts champion + canary concurrently via versioned model repo — instant traffic-split, no redeploy.
- **Hardware**: GPU (T4/A10G) for deep nets, CPU for GBTs.
- **Shadow duplication**: sidecar proxy forwards a sampled async copy to the candidate, fire-and-forget, doesn't block the prod response.

## 14. Feature Store

- **Online**: Redis/DynamoDB, p99 < 5ms.
- **Offline**: Delta Lake on S3, for training data generation and backtesting.
- **Point-in-time correctness**: every write is timestamped; training joins are as-of joins against feature state at label time, preventing leakage (e.g., a churn label must join pre-churn feature values).
- **Freshness**: online features refreshed within 5 min for behavioral signals; season-long stats refreshed daily.
- **Train/serve skew prevention**: shared feature-transformation code compiled to both streaming (Flink) and batch (Spark) paths.

## 15. Inference Request Lifecycle

```
Player action → Serving Gateway (cached prod-version pointer, ~0.1ms)
  → Feature fetch (Redis, ~3-5ms)
  → Primary inference (Triton champion, ~8-15ms batched) ──► Shadow router (async, fire-and-forget)
  → Response to game backend (p50 ~15ms, p99 ~45ms)             → candidate inference → logged to shadow.predictions
  → Prediction logged to model.predictions (async)
  → [30 days later] Outcome observed → model.outcomes → drift-detector computes concept-drift delta
```

## 16. Training Pipeline

- **Data prep**: pull point-in-time snapshot, run data-quality checks (schema, null-rate, leakage), time-based train/val/test split (never random — prevents future leakage).
- **Orchestration**: Argo/Airflow DAG `data_prep → train → offline_eval → validation_gate → register`, each step a K8s Job sized to model class.
- **Distributed training**: deep nets use PyTorch DDP across 8-16 GPUs (Ray Train/Torchrun), gradient checkpointing for the large toxicity transformer. GBTs use distributed histogram training (Dask-XGBoost) for large churn datasets.
- **Lineage**: every run records data hash, code SHA, hyperparams, container digest — stored in the registry for reproducibility.
- **Hyperparameter search**: Bayesian search (Optuna) for scheduled full retrains; warm-start incremental fine-tuning for drift-triggered fast retrains.

## 17. Retraining Strategy & Drift Detection

| Model Class | Cadence | Trigger |
|---|---|---|
| Fraud/cheat | Continuous incremental + full retrain every 6h | Precision/recall drop, adversarial pattern spike |
| Matchmaking skill | Daily + drift-triggered | PSI > 0.2 on skill features, meta-shift events |
| Churn/LTV | Weekly + drift-triggered | Calibration error increase, content-drop events |
| Toxicity classifier | Weekly + drift-triggered | Vocabulary/embedding drift, policy changes |
| Recommendation | Daily incremental, weekly full | Engagement drop, catalog change |

Drift-triggered retrains take priority in the queue; scheduled retrains still run as a backstop in case detectors miss something.

| Drift Type | Metric | Action |
|---|---|---|
| Data drift (numeric) | PSI per feature | PSI > 0.2 warn, > 0.3 trigger retrain |
| Data drift (categorical) | KL-divergence | KL > 0.15 triggers retrain |
| Concept drift (performance) | Rolling AUC vs. baseline on delayed labels | Drop > 3pts (24h) triggers; > 8pts pages on-call |
| Concept drift (calibration) | ECE | Increase > 0.05 triggers |
| Prediction drift (proxy) | PSI on score distribution | Early-warning only, noisy proxy |

Require 2 consecutive breaching windows before firing a trigger (debounce, avoids retrain thrashing from noise).

## 18. Monitoring & Alerting

| Category | Metrics |
|---|---|
| Infra | Pod health, GPU util, job queue depth, Kafka consumer lag, feature-store latency |
| Pipeline | Retrain success/failure rate, duration p50/p99, gate pass rate, time-to-promote |
| Model quality | Offline eval per version, shadow-vs-champion delta, canary-stage deltas |
| Drift | PSI/KL per feature, concept-drift AUC delta |
| Business | Match balance, retention delta post-promotion, false-positive ban/toxicity rate |
| Cost | GPU-hours per model family, spot vs. on-demand ratio |

| Alert | Condition | Routing |
|---|---|---|
| Retrain pipeline failure | Job fails after retries | Page ML on-call |
| Drift trigger storm | >5 models trigger within 1hr | Page on-call — likely upstream data issue |
| Gate rejection | Candidate fails | Notify owning DS team, no page |
| Shadow regression | Candidate underperforms champion during bake | Notify DS team, block auto-promotion |
| Canary SLO breach | p99 latency or error-rate budget exceeded | Auto-rollback + page SRE |
| Severe concept drift | AUC drop > 8pts, sustained | Page ML on-call immediately |
| DLQ depth > 0 on `model.promotions` | Any message | Page SRE (critical path) |

**Logging**: structured JSON with correlation `request_id`/`run_id` across services. PII (player_id, IP, chat text) hashed before entering general logs; raw chat text stored in a restricted enclave with field-level encryption. Operational logs: 30 days hot, 1 year cold; audit trail retained 3 years.

## 19. Security & Access

- **Threats**: data poisoning (crafted telemetry biasing retraining) → anomaly detection + data-quality gate on ingestion. Model exfiltration → artifact encryption at rest, scoped IAM, audit logging on downloads. Forged gate results → gate verdicts are signed, registry verifies signature before stage transition. Adversarial drift gaming (cheaters shifting behavior to "teach" cheating as normal) → held-out adversarial eval sets untouched by drift retrains, human sign-off required for fraud-model promotions.
- **Encryption**: TLS 1.3 in transit, AES-256 at rest, field-level encryption for chat text.
- **AuthN**: mTLS/SPIFFE for service-to-service (Istio); OAuth2/OIDC + RBAC for humans (DS engineers trigger/view their own models; only platform admins change gate thresholds or force-promote).
- **Least privilege**: training jobs read offline store + write registry staging only; serving fleet reads registry prod pointer only.

## 20. Rate Limiting & Autoscaling

- Manual retrain trigger: 1 per model per 10 min (token bucket, burst of 3).
- Drift-triggered retrains: debounced (2 windows) plus circuit breaker — >3 triggers/24h auto-pauses and pages on-call (signals detector misconfiguration).
- Shadow traffic: hard-capped at 10% of primary traffic regardless of configured sample rate (protects shared node pool).
- **Autoscaling**: training GPU pool via KEDA on job-queue depth, scale-to-zero when idle. Serving fleet HPA on inference-queue-time + GPU util (target 70%). Drift detector HPA on Kafka consumer lag.

## 21. Cost Optimization

- Spot GPU for training with checkpoint-resume (saves ~65-70% vs. on-demand, ~5% overhead).
- Scale-to-zero GPU pool when idle.
- Distill the toxicity transformer to a smaller serving model (~4x cheaper), keep the large model only for periodic teacher retraining.
- Dynamic batching on Triton; nightly batch scoring instead of real-time where allowed.
- Sampled (not full-fidelity) drift detection for lower-risk models.
- Storage tiering for cold training data; dedup overlapping feature snapshots.

## 22. Operational Concerns

At SDE2 scope, treat this as a checklist: **backups** (automated snapshots of registry/feature store with a tested restore path), **rollback** (one-command revert to last-known-good), **canary rollout** (small traffic % first, watch error rate and model-quality metrics, then ramp), **observability** (dashboards + alerts on latency, error rate, top model-quality signals, wired to on-call). Kubernetes/Terraform manifest details and multi-region active-active topology are Staff/Principal-level concerns — know they exist, don't rehearse the specifics.

## 23. Why This Architecture

- Kafka decouples ingestion from feature computation, drift detection, and downstream consumers — needed given 6+ titles' bursty, heterogeneous volume.
- Separating streaming drift detection from batch retraining lets each scale independently (drift needs 15-min latency, retraining tolerates hours).
- Staged promotion (gate → shadow → canary) matches the risk profile of live games — no single check is trusted alone.
- The registry as single source of truth for "what's live" enables redeploy-free rollback, matching the <60s SLA.
- One orchestration framework across model classes reduces platform surface area vs. bespoke pipelines per team, while still allowing per-model cadence/resource customization.

## 24. Alternative Architectures

| Alternative | Why Rejected / When Preferred |
|---|---|
| Fully manual retraining via scripts + tickets | Too slow (1-2 week cycle) for live-service drift; fine for a low-stakes, low-drift internal model |
| Direct-to-prod on gate pass, no shadow/canary | No real-traffic validation before full exposure — too risky for player-facing models; ok for low-blast-radius internal tools |
| Fully synchronous exactly-once pipeline | Added latency/complexity not justified when all consumers are idempotent; reconsider for strict-dedup financial use cases |
| Per-title independent platforms | Duplicated effort across titles, inconsistent gating rigor; only makes sense with incompatible stacks or fully decentralized orgs |

## 25. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| At-least-once + idempotent consumers | Lower latency, simpler infra | Requires disciplined dedup logic everywhere |
| Staged canary over instant promotion | Catches subtle regressions (calibration, delayed-label effects) | Slower rollout (24-48+ hrs) |
| Shared orchestration across model classes | Less duplicated engineering, consistent gates | Abstraction overhead for unusual model classes |
| Sampling-based drift detection | Big cost savings | Reduced power to catch subtle drift early |
| Debounced drift triggers | Avoids retrain thrashing | Adds detection latency |

## 26. Failure Modes

| Scenario | Impact | Mitigation |
|---|---|---|
| Kafka broker outage in a region | Feature pipeline/drift detection stalls | Multi-AZ replication, cross-region failover |
| Training data corruption | Candidate trained on garbage data | Data-quality gate before training even starts |
| Validation gate bug lets a bad model pass | Bad model reaches shadow/canary | Shadow independently re-validates; canary auto-rollback catches the rest |
| Registry Postgres primary failure | Can't record new promotions | Multi-AZ failover; cached prod pointer keeps serving unaffected |
| Drift false-positive storm | Wasted compute, alert fatigue | Debounce + circuit breaker |
| GPU spot capacity unavailable | Training delayed | Fallback to on-demand for high-priority (fraud) jobs |
| Adversarial data poisoning | Skewed retraining | Ingestion anomaly detection, adversarial-resistant eval sets, human sign-off for fraud models |

## 27. Bottlenecks

- **At 10x event volume**: Kafka partitions/Flink parallelism become the first bottleneck; registry read load from more frequent gate/drift checks strains the single-primary Postgres.
- **At 100x**: single-region Kafka becomes untenable — would need topic sharding + federated drift detection; offline store scans need better partitioning or a purpose-built feature store.
- **Retrain pipeline latency** (target < 4hr): data prep ~20-45min, training ~90-150min (dominant cost, GPU contention), gate eval ~15-30min. Training stage dominates; mitigated by spot burst capacity, though preemption adds p99 variance.
- **Inference latency** (target p50 15ms/p99 45ms): feature fetch ~3-8ms, inference ~8-25ms, overhead ~4-12ms. p99 bottleneck is batching-queue wait during traffic bursts — larger batches cut GPU cost but raise queueing latency (a direct tuning tradeoff).
- **Cost bottleneck**: GPU training compute dominates (~$20-25K/month cloud burst), driven by the toxicity fine-tune and daily matchmaking retrains across 6 titles — distillation and warm-start retraining are the biggest levers. Cold storage (~300TB) is a steady linear cost driver, addressed via lifecycle tiering.

## 28. Interview Follow-Ups

1. How do you prevent a drift-triggered retrain storm from becoming a cost incident?
   - Per-model rate limits (1 manual/10min) plus a circuit breaker that auto-pauses drift-triggering after >3 fires/24h and pages on-call, distinguishing real sustained drift from noisy detectors.

2. What if the validation gate service itself has a bug that inverts pass/fail logic?
   - Defense in depth: shadow evaluation independently re-checks against live traffic, and canary stage 1 has automatic SLO/business-metric rollback independent of the gate's verdict.

3. How do you guarantee point-in-time correctness when online/offline stores update at different latencies?
   - Every feature write carries an event-time timestamp; training joins are as-of joins keyed on that timestamp, so differing pipeline lag doesn't corrupt "what the feature looked like at label time."

4. Two model versions promoted to canary simultaneously by two engineers — what happens?
   - Promotion is a transactional compare-and-swap on stage in Postgres; the second concurrent request fails the conditional update and gets a conflict response.

5. How do you detect concept drift when labels are delayed 30+ days (e.g., churn)?
   - Two signals: a fast proxy (PSI on prediction-score distribution) as early warning, and ground-truth rolling AUC once labels resolve to confirm and fire the actual retrain.

6. Candidate is better in aggregate but worse on one player segment — what do you do?
   - Gate and shadow comparison both run required segment-slice checks (region, skill tier, tenure); a segment regression fails the gate and needs explicit DS sign-off to override.

7. How do you roll back once the previous champion's artifact has been evicted from the serving cache?
   - The registry always keeps the artifact in cold storage even after cache eviction, so rollback takes an extra 1-2 min to re-download instead of an instant pointer flip.

8. What changes to onboard a 7th title on a different tech stack?
   - Core contracts (event schemas, registry API, gate interface) are stack-agnostic and training jobs are just K8s Jobs, so onboarding cost is mostly title-specific feature transforms and gate thresholds, not re-architecture.
