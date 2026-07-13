# Streaming ML Platform

## 1. Problem Framing

Design a streaming ML platform that computes features and serves predictions in real time off live event streams (player telemetry, matchmaking, purchases, chat) for EA live-service titles (Apex, FIFA Ultimate Team, The Sims). Unlike batch feature store + offline model behind a REST endpoint, this system must:

- Compute windowed aggregates ("kills in last 5 min", "spend in 24h") continuously from Kafka via Flink.
- Serve inference on features that are seconds old, not hours old.
- Support online/incremental model updates from the same streams (contextual bandit for matchmaking, anti-cheat anomaly scoring) without a full batch retrain.
- Guarantee point-in-time correctness between train-time and serve-time features, despite both being streaming-derived.

Primary use cases:
1. Real-time anti-cheat scoring — flag aimbot/wallhack behavior within seconds.
2. Live matchmaking quality prediction — rebalance lobbies before match start.
3. Dynamic pricing/offer personalization (FUT packs) based on recent behavior.
4. Toxicity/chat moderation scoring over voice-to-text and chat events.

## 2. Functional Requirements

- FR1: Ingest telemetry (kills, deaths, purchases, chat, input traces) from Kafka at the edge.
- FR2: Compute streaming feature aggregates via tumbling/sliding/session windows (1-min to 24-hour).
- FR3: Serve low-latency predictions (p99 < 100ms) using freshest streaming features joined with static features.
- FR4: Support online model updates (incremental gradient steps, bandit updates) triggered by labeled events ("ban confirmed", "purchase completed").
- FR5: Maintain a consistent online feature store queryable by both the streaming job and inference service.
- FR6: Support backfill/replay — reprocess a Kafka topic from an offset to rebuild feature state.
- FR7: Version features, models, and window definitions; support shadow scoring of new model versions.
- FR8: Log prediction + feature snapshots for offline evaluation, drift detection, audit.
- FR9: Per-title, per-region isolation — a bug in one title's pipeline must not affect another's.

## 3. Non-Functional Requirements

| Dimension | Target |
|---|---|
| Inference latency | p50 < 30ms, p99 < 100ms |
| Feature freshness | p99 staleness < 2s from ingestion to feature-store visibility |
| Streaming throughput | 2.5M events/sec platform-wide at peak |
| Availability | 99.95% inference path, 99.9% streaming pipeline |
| Consistency | Eventual for online feature store; exactly-once for feature aggregation (no double-counted kills) |
| Durability | Kafka retained 7 days; feature snapshots retained 30 days |
| Cost | Streaming compute + online store under $0.0004/1K events fully loaded |
| Model update latency | Online param refresh visible to serving within 60s of a labeled event |
| Scalability | Linear horizontal scaling to 5x traffic during a launch/live-event spike |

## 5. Assumptions

1. Platform serves 8 titles concurrently; largest (Apex-scale) is 60% of traffic.
2. Peak CCU platform-wide: 6M during a live event; average 1.8M.
3. Each active player emits ~15 telemetry events/min.
4. Online learning = micro-batch incremental updates every 30-60s (per-event SGD deemed too unstable given noisy label signals).
5. Inference is synchronous for anti-cheat (kick/allow decision), async for pricing/personalization.
6. Exactly-once required for purchase-related aggregates; at-least-once + idempotency acceptable for behavioral features.
7. Feature vectors: 80-150 numeric/categorical features per entity.
8. Models are small-to-medium (GBTs for anti-cheat, linear/bandit for matchmaking, <50MB) — no LLM-scale models.
9. GDPR applies; EU player data stays in EU region infra.

## 6. Capacity Estimation

**Ingestion**: 6M CCU × 15 events/min ≈ 1.5M events/sec telemetry; +40% for purchase/chat/matchmaking → ~2.5M events/sec design target.

**Kafka**: avg event 400B → 8 Gbps peak ingress, 24 Gbps with 3x replication → ~30-40 broker nodes. 7-day retention ≈ 1.8 PB raw (tiered to cold storage after 24h, cutting hot local disk to ~120TB).

**Flink**: 1 vCPU handles ~15-25K events/sec for simple windowed aggregation. 2.5M/sec ÷ 20K ≈ 125 vCPUs minimum, provisioned at 2.5x for skew/checkpoint overhead → ~320 vCPUs (~40 nodes). State: 6M keys × ~2KB ≈ 80-120GB RocksDB state, fits across task managers with local SSD.

**Online feature store**: reads ≈ prediction QPS — anti-cheat scores every active player every 10s (600K reads/sec) + matchmaking/pricing (~150K) ≈ 750K reads/sec. Writes ≈ 100K/sec sustained (bursts 250K/sec). Hot storage: 6M entities × 150 features × 8B ≈ 7.2GB (trivial). Cold audit copy: ~80TB/month compressed.

**Model serving**: anti-cheat GBT ~0.5ms/inference on 1 vCPU → 600K QPS needs ~300 vCPUs, provisioned to ~450 at 1.5x headroom. No GPU requirement (tree ensembles + linear/bandit models).

**Online learning compute**: micro-batch retrain every 30-60s over ~2M labeled events, ~15-30s on 16 vCPUs/title → 8 × 16 = 128 vCPUs.

**Total**: ~320 (Flink) + ~450 (serving) + ~128 (trainer) ≈ ~950-1000 vCPUs platform-wide at peak, plus ~35 Kafka broker nodes. No GPUs needed for baseline scope.

## 7. High-Level Architecture

```
Game Clients/Servers --gRPC--> Edge Ingestion Gateway (authn, schema validation, batching)
                                        │
                                        ▼
                    Kafka (per-region): telemetry.raw, purchases,
                    matchmaking.events, chat.events, labels.confirmed
                       │                              │
                       ▼                              ▼
        Flink: Windowed Feature Agg        Flink: Label-Join
        (tumbling/sliding windows,          (joins labels + feature
         exactly-once sink)                  snapshots -> training rows)
                       │                              │
                       ▼                              ▼
        Online Feature Store              Streaming Trainer
        (Redis/ScyllaDB, hot)             (micro-batch, 30-60s)
        + offline columnar audit copy              │
                       │                            ▼
                       │                    Model Registry (versioned params)
                       │                            │
                       ▼                            ▼
              Inference Service (stateless)
              - fetch features, fetch latest model shard (hot-reload)
              - score, apply business rule, respond/publish
                       │                            │
                       ▼                            ▼
        Sync response to game server      prediction.events topic
        (anti-cheat kick/allow)           (async: pricing, matchmaking)
                       │
                       ▼
        Monitoring/Drift/Logging plane (cross-cutting):
        Prometheus, snapshot log -> S3/Iceberg, drift detectors, alerting
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| Edge Ingestion Gateway | AuthN, schema validation, batching, backpressure | gRPC/HTTP from client SDK | Stateless pods, regional LB |
| Kafka Cluster | Durable, ordered event log; replay source of truth | Producer/consumer API, topic per event class | Partition/broker scale-out |
| Flink Feature-Aggregation Job | Compute windowed aggregates, write to online store | Kafka source → keyed windowed ops → sink | Task-manager slots, keyed by entity ID |
| Flink Label-Join Job | Join late labels with historical feature snapshots for training set | Kafka source → join → training-set topic | Separate job graph |
| Online Feature Store | Low-latency point lookups | gRPC `GetFeatures(entity_id, feature_set_version)` | Sharded by entity ID hash |
| Offline Feature Store | Historical snapshots for audit, backfill, offline eval | Iceberg/Parquet on S3, Trino/Spark | Partitioned by date + title |
| Streaming Trainer | Consume training-set topic, run incremental fit, push params | Kafka consumer → training loop → Registry API | One consumer group per title |
| Model Registry | Version/store/serve model artifacts with promotion workflow | REST/gRPC `GetLatestModel(title, model_name, stage)` | Stateless read replicas + cache |
| Inference Service | Serve predictions; combine features + model; apply rules | gRPC `Predict(entity_id, context)` | Stateless, autoscaled on QPS/CPU |
| Drift Detector | Compare live feature/prediction distributions vs. baseline | Batch/streaming hybrid over snapshot logs | Scales with monitored feature sets |
| Prediction/Feature Logger | Durable audit trail for every scored request | Async sink to Kafka → Iceberg | Scales with prediction QPS |

## 9. API Design

**Feature Store**: `GetFeatures(entity_id, feature_set, as_of?)` → `{features: map<string,float>, freshness_ms, feature_set_version}`. Version via name suffix (`_v1`, `_v2`) — never mutate a live feature set in place.

**Inference**: `POST /v1/titles/{title}/models/{model_name}/predict` with `{entity_id, context, model_version}` → `{prediction, decision, model_version, feature_freshness_ms, trace_id}`. URL path stable; `model_version="latest"` resolves to production pointer, enabling shadow calls without a new endpoint.

**Async consumption**: `prediction.events.{title}` topic, consumer groups pull via Kafka; schema registry enforces backward-compatible Avro/Protobuf evolution.

**Model Registry**: `POST /versions` (register), `GET /latest?stage=production|shadow|canary`, `POST /promote`.

| Endpoint | SLA | Auth |
|---|---|---|
| `/predict` | p99 100ms | mTLS service token |
| `GetFeatures` (gRPC) | p99 15ms | mTLS |
| `/models/.../latest` | p99 20ms (cached) | mTLS |
| `/models/.../promote` | best-effort, human-gated | OAuth2 + RBAC (MLE role) |

## 10. Database Design

| Store | Type | Why | Partition Key |
|---|---|---|---|
| Online Feature Store | Redis Cluster or ScyllaDB (if >TB-scale) | Sub-ms/low-ms lookups at 750K QPS | Hash(entity_id) |
| Offline Feature/Prediction Log | Iceberg on S3, Trino/Spark | Columnar, cheap, time-travel for point-in-time correctness/backfill | `title`, `event_date` |
| Model Registry Metadata | PostgreSQL | Strong consistency for version pointers, small dataset | `(title, model_name, version)` |
| Model Artifacts | S3-compatible blob store | Large binary blobs, versioned, immutable | `{title}/{model_name}/{version}/model.bin` |
| Kafka | Log-structured system-of-record | Ordered per-partition, replay-capable | Partitioned by entity_id |

Online store record (Redis hash): `fs:anticheat_v3:{player_id}` with fields like `kills_5m, deaths_5m, headshot_ratio_1m, last_updated_ts`; TTL 6h to auto-expire stale players.

Iceberg `prediction_log` table: `trace_id, title, entity_id, model_version, feature_snapshot, prediction, decision, event_ts, ingest_ts`, partitioned by `(title, days(event_ts))`.

**Point-in-time correctness**: the training-materialization job joins `labels.confirmed` against the feature snapshot logged at prediction time (`prediction_log.feature_snapshot`), never the current online-store value — prevents label leakage from future feature states.

## 11. Caching

| Cached Item | Cache Type | Invalidation |
|---|---|---|
| Latest model artifact (inference pod memory) | Local in-process | TTL 60s + registry push (`model.updates` topic) |
| `GetLatestModel` lookups | Redis in front of Postgres | Write-through on promotion |
| Hot feature vectors | L1 in-pod LRU | TTL 2s (respects freshness NFR) |
| Static/offline features (profile, cosmetic tier) | CDN/Redis, 5-min TTL | Explicit invalidation on profile update |

No write-through on the online feature store itself — Flink sink is the sole writer, inference is read-only. Model hot-reload swaps the in-memory pointer atomically via the `model.updates` topic, avoiding restarts.

## 12. Queues & Async Processing

| Topic | Semantics | Why |
|---|---|---|
| `telemetry.raw.*` | At-least-once | High volume; idempotent aggregation (dedup by event_id) makes exactly-once unnecessary |
| `purchases.*` | Exactly-once (transactional producer + idempotent consumer) | Financial correctness — no double-counted spend |
| `labels.confirmed` | At-least-once + idempotent upsert | Late/duplicate delivery expected |
| `model.updates` | At-least-once, small volume | Notifies pods of new model; pods poll registry as fallback |
| `prediction.events.*` | At-least-once | Downstream consumers idempotent by trace_id |

Flink checkpointing (RocksDB, checkpoint every 10s to S3) gives exactly-once within the job graph despite at-least-once upstream delivery, via event_id dedup in state.

## 13. Streaming & Event-Driven Architecture

Core topics are partitioned by `player_id`/`match_id`/`entity_id`, with retention ranging 24h (telemetry) to 90d (purchases, financial audit), schema-registry-enforced Protobuf/Avro.

**Consumer groups**: `flink-feature-agg-{title}` (parallelism = partition count), `flink-label-join-{title}`, `streaming-trainer-{title}`, `inference-model-watcher` (broadcast-like across titles), `drift-detector-{title}`.

**Windowing**: tumbling for fixed-cadence features (1-min kill count), sliding for smoothed signals (5-min headshot ratio, 1-min slide), session windows for matchmaking (gap-based, 10-min inactivity closes session).

**Watermarking**: bounded out-of-orderness of 5s for client clock skew; late events routed to a side output, reconciled offline.

## 14. Model Serving

- Custom lightweight inference service (Rust/Go/Java) hosting GBT (XGBoost/LightGBM) and linear/bandit models — not a GPU serving stack, since models are small and CPU-bound.
- Batching disabled on the synchronous anti-cheat path (adds latency); enabled on the async pricing path (50-100 entities/call, vectorizes well).
- One inference service hosts multiple model versions per title (production, canary, shadow); shadow traffic is mirrored and logged, not returned, to validate before promotion.
- CPU-only fleet, AVX2-optimized inference; no GPU needed unless a future embedding-based model is added.
- Hot reload via atomic in-process pointer swap on the `model.updates` topic — no pod restarts.

## 15. Feature Store

- **Online**: Redis/ScyllaDB, `GetFeatures` at p99 <15ms, holds only the latest value per (entity, feature_set).
- **Offline**: Iceberg/Parquet, full history of feature snapshots (one row per prediction) for training reconstruction and audits.
- **Point-in-time correctness**: logging the exact feature vector used at prediction time avoids the classic offline/online skew — training materialization always joins against the logged snapshot, never a live re-query.
- **Versioning**: bump `feature_set` version whenever a window definition or aggregation changes; dual-write during migration until all consumers move.

## 16. Vector Database

N/A for baseline scope — models operate on structured windowed aggregates, not embeddings. A future semantic player-behavior model (e.g., smurf detection) would add a vector DB (pgvector/ANN index) alongside this core, not replace it.

## 17. Embedding Pipelines

N/A for baseline scope, same reason — features are hand-engineered aggregates, not learned embeddings. A future embedding pipeline (e.g., two-tower model) would sit upstream of the online feature store, writing embeddings as additional fields — additive, not a redesign.

## 18. Inference Pipelines (Request Lifecycle)

```
[Game Server] --telemetry--> [Kafka: telemetry.raw]
                                    │
                          Flink windowed agg
                                    │
                                    ▼
                    [Online Feature Store write]

[Game Server] --needs decision--> [Inference Service]
                                    │
                    GetFeatures(player_id, "anticheat_v3")  <-- p99 15ms
                                    │
                    fetch cached model pointer (~0ms)
                                    │
                    model.predict(feature_vec)  <-- ~0.5-2ms
                                    │
                    apply business rule (threshold/hysteresis)
                                    │
                    log prediction+features (async)
                                    │
[Game Server] <--ALLOW/FLAG/KICK-- [Inference Service]
                                    │
                    publish to prediction.events (async, for
                    downstream matchmaking rebalancer / audit)
```

**Latency budget (p99 100ms target)**: network in 20ms, feature read 15ms, model inference 3ms, business rule + serialization 2ms, response network 20ms, buffer/queueing headroom ~40ms (GC pauses, pool contention, cross-AZ hops).

## 19. Training Pipelines

- **Data prep**: `flink-label-join` continuously joins `labels.confirmed` with historical `prediction_log.feature_snapshot`, emitting labeled rows to `training.materialized.{title}`.
- **Orchestration**: `streaming-trainer` (per title) consumes `training.materialized`, accumulates a rolling window (last 2M rows or 60s), and runs incremental GBT boosting (warm-start via `xgb_model=`) or warm-started linear/bandit updates (Thompson-sampling posterior update).
- **Distributed training**: not required at this scale — single-node incremental fit per title is sufficient. Distributed training (Spark MLlib) is reserved for the nightly full offline retrain, which rebuilds from scratch on the full Iceberg dataset to correct for online-update drift.
- **Nightly full retrain**: Spark job over Iceberg, 20-40 node cluster, produces a fresh baseline that the online trainer continues to update through the next day.

## 20. Retraining Strategy

| Trigger | Cadence/Condition | Action |
|---|---|---|
| Scheduled (baseline) | Every 30-60s micro-batch | Push to `shadow` stage automatically |
| Scheduled (full rebuild) | Nightly per title, off-peak | Full Spark retrain, replaces `production` after eval gate |
| Data-drift triggered | PSI > 0.2 on top-10 feature | Force early retrain, page ML on-call |
| Concept-drift triggered | Live AUC/precision drop > 5% vs. 7-day baseline | Halt promotion, fall back to last-known-good, alert |
| Label-volume triggered | Labeled event rate drops >50% | Pause online trainer (avoid overfitting to sparse labels), alert |
| Manual | ML engineer via Registry API | Ad-hoc retrain, requires approval to promote |

Promotion gate: every online-updated version enters `shadow` automatically; promotion to `production` requires a passing 15-min shadow evaluation or human approval (nightly baseline swap).

## 21. Drift Detection

| Drift Type | Metric | Threshold | Action |
|---|---|---|---|
| Feature (data) drift | PSI per feature, hourly vs. 7-day baseline | >0.2 alert, >0.3 force retrain | Alert ML on-call; auto-retrain at 0.3 |
| Feature drift (categorical) | Jensen-Shannon divergence | JSD > 0.1 | Alert |
| Concept drift | Rolling AUC/PR-AUC or win-rate correlation | Drop > 5% relative | Halt promotion, fallback model, alert |
| Prediction drift | Score distribution mean/p50/p95, hourly | Mean shift > 0.15 absolute | Alert, investigate |
| Label delay/skew | Prediction-to-label latency, positive-rate | p90 > 2x historical median | Alert (may indicate broken pipeline) |
| Online-vs-offline divergence | KL divergence, shadow vs. nightly baseline on holdout | > 0.05 | Investigate drift accumulation; cap update magnitude |

## 22. Monitoring

| Layer | What's Monitored |
|---|---|
| Infra | Kafka consumer lag/partition skew, Flink checkpoint duration, backpressure, Redis/Scylla p99, pod CPU/mem/QPS |
| Model quality | Rolling AUC/precision/recall, calibration, bandit regret, shadow-vs-production agreement |
| Data pipeline | Ingestion rate vs. expected, schema-rejection rate, DLQ depth, label-join match rate |
| Business | False-positive ban rate, match-quality proxy, offer conversion, pricing revenue impact |
| Freshness | Feature write-to-read staleness (p50/p99), model-update propagation latency |

Dashboards: Grafana (Prometheus + Kafka Exporter + Flink metrics), plus a combined "ML Health" dashboard (drift + business KPIs) for weekly model review.

## 23. Alerting

| Alert | Condition | Severity |
|---|---|---|
| Kafka consumer lag | >30s sustained 2min on feature-agg group | P1 |
| Flink restart loop | >3 restarts/10min | P1 |
| Inference p99 latency | >150ms for 5min | P1 |
| Inference error rate | >1% for 5min | P1 |
| Feature staleness | p99 >5s for 5min | P2 |
| PSI drift breach | PSI > 0.3 | P2 |
| Concept drift | >5% relative AUC drop, 3 consecutive hourly windows | P1, auto-fallback |
| DLQ depth | >1000 msgs or >0.1% of topic volume | P2 |
| Model promotion failure | Shadow gate fails 3x consecutively | P3 |
| Purchase write failure | Any failure after retries | P1 |

Routing via PagerDuty: P1 pages 24/7, P2 pages business hours (or after 30min unacked), P3 ticket only. Drift alerts deduped/aggregated per title/hour to avoid noise during known live-events.

## 24. Logging

- Structured JSON logs with `trace_id, title, entity_id_hash, service, timestamp, latency_ms, model_version`.
- **PII**: raw player IDs and PII (chat text, payment tokens) never in general logs — only in the access-controlled, field-encrypted `prediction_log`. `entity_id` in logs is a salted hash. Chat/toxicity text redacted outside the encrypted store. COPPA-flagged accounts get stricter 30-day retention.
- **Retention**: app/infra logs 14 days; prediction/feature audit logs 30 days hot, 1 year cold-archived; purchase logs 7 years (compliance).
- **Correlation**: `trace_id` propagated end-to-end (client → Kafka headers → Flink → inference response) for full request reconstruction.

## 25. Security

- **Threats**: model extraction/evasion via repeated probing (mitigate: rate limiting, return decision not raw score, periodic rotation); data poisoning via fabricated telemetry (mitigate: per-source anomaly detection, cap any single entity's influence on a micro-batch update); insider/lateral-movement risk to the feature store (mitigate: network segmentation, mTLS, least-privilege IAM); label-source spoofing (mitigate: signed producer credentials on `labels.confirmed`).
- **Encryption**: TLS 1.3 in transit everywhere; at rest — Kafka topic encryption for purchases/chat, KMS-managed keys, Iceberg via S3-SSE-KMS, Redis encryption-at-rest.
- **Data minimization**: raw chat/PII never stored as features directly — a separate toxicity classifier scores it upstream, only the numeric score enters the feature store.

## 26. Authentication

- **Service-to-service**: mTLS with short-lived certs (SPIFFE/SPIRE), rotated every 24h.
- **Client/server → Gateway**: game servers (trusted) authenticate via OAuth2 client-credentials scoped per title; game clients never call the platform directly — all telemetry routes through the EA-operated game server.
- **Human/operator access**: SSO (Okta/OIDC) + RBAC — `ml-engineer` (promote to shadow), `ml-lead` (promote to production), `auditor` (read-only, PII-redacted).

## 27. Rate Limiting

- Token bucket per (title, player_id) at the Edge Gateway for telemetry — burst 50, refill 20/sec, bounds abusive clients trying to manipulate windowed aggregates.
- `/predict`: sliding-window counter per game-server instance — 50K req/sec per shard, `429` with backoff on breach.
- `/promote`: fixed low limit (10/min per user) — accident prevention, not load protection.
- Per-title QPS quotas on shared infra (Kafka broker quotas + inference pool reservations) prevent one title's spike from starving another's SLA.

## 28. Autoscaling

- **Inference**: Kubernetes HPA on `inference_qps_per_pod` (target 800 QPS/pod) + CPU (target 60%); scale range 20-200 pods/title-shard; fast scale-up (30s cooldown), slow scale-down (5min, avoid flapping).
- **Flink**: not HPA-based (stateful rebalance is expensive) — reactive scaling via Adaptive Scheduler on sustained backpressure/lag, with controlled checkpoint-redistribute-resume rescale.
- **Kafka brokers**: manually provisioned with headroom ahead of known events (partition reassignment is disruptive, not autoscaled).
- **Online feature store**: vertical headroom monitoring + manual resize playbook; Scylla supports online node addition.
- **KEDA** for async `prediction.events` consumers, scaling on Kafka lag, 0-to-N capable during quiet periods.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-service-apex
spec:
  minReplicas: 20
  maxReplicas: 200
  behavior:
    scaleUp: { stabilizationWindowSeconds: 30 }
    scaleDown: { stabilizationWindowSeconds: 300 }
  metrics:
    - type: Pods
      pods: { metric: { name: inference_qps_per_pod }, target: { type: AverageValue, averageValue: "800" } }
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 60 } }
```

## 29. Cost Optimization

- Flink task managers and the nightly Spark cluster on spot (checkpointed, restartable); inference stays on-demand (latency-critical).
- Async-path batching (50-100 entities/call) cuts vCPU-seconds ~30%.
- GBT models pruned/quantized (float16 where supported); matchmaking bandit kept linear since a deeper model's AUC gain didn't justify 3x compute.
- Tiered Kafka storage (24h hot, rest to object storage) roughly halves broker disk/node count.
- Cache-aside on model/feature reads reduces Redis/Scylla QPS load, allowing a smaller cluster.
- 6-hour TTL on inactive-player feature keys prevents unbounded memory growth.
- Per-title autoscale floors tuned to actual traffic share rather than uniform floors.

## 30. Operational Concerns

At SDE2 scope, treat this as a checklist: backups (automated snapshots of registry/feature store with a tested restore path), rollback (every deploy revertible to last-known-good in one command), canary/blue-green rollout (shift a small traffic % first, watch error rate and key metrics, then ramp), and basic observability (dashboards + alerts on latency, error rate, top model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level concerns — worth knowing they exist, not worth rehearsing manifests.

## 31. Why This Architecture

- Decouples feature computation (Flink, stateful) from serving (stateless pods reading a materialized store) — each scales independently on its own bottleneck.
- Kafka as the universal backbone gives replay-ability, critical for backfills and reconstructing training sets.
- Logging the exact feature snapshot at prediction time sidesteps the hardest correctness bug in streaming ML: train/serve skew from a mutable "current state" feature store.
- Micro-batch online learning (30-60s) balances fast adaptation against the operational stability of per-event SGD, which is fragile to noisy/adversarial updates.
- Shadow/canary gating on every online-trained version protects against continuous learning's unique risk: a bad label batch silently degrading production without a human in the loop.

## 32. Alternative Architectures

| Alternative | Why Rejected / When Preferred |
|---|---|
| Pure batch pipeline (hourly/daily feature + retrain) | Rejected: can't meet <2s freshness or fast anti-cheat reaction; fine for e.g. weekly matchmaking-tier recalculation |
| Per-event synchronous SGD | Rejected as default: too sensitive to label noise/poisoning, hard to gate per-event; fine for a low-traffic, low-stakes signal |
| Lambda architecture (batch + speed layers merged at query time) | Adopted selectively (nightly Spark retrain as batch baseline + streaming trainer on top), not as full Lambda — merge-at-query-time would add serving-path latency |
| Fully synchronous request-scoped feature computation (no precomputed store) | Rejected: recomputing a 24h window per request at 750K QPS is infeasible; only viable at very low QPS/window size |

## 33. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Micro-batch (30-60s) vs. per-event SGD | Stable, gateable, resilient to noisy labels | Up to ~60s+shadow-burn-in lag before full mitigation |
| At-least-once telemetry vs. exactly-once everywhere | Simpler, cheaper, higher throughput | Requires idempotent dedup logic everywhere in Flink state |
| Redis/Scylla (no history) + Iceberg (full history) | Cheap, fast serving; audit/backfill preserved | Two systems to keep schema-consistent |
| Region-local Kafka/Flink/inference | Meets latency SLA, respects data residency | No global view without replication lag; global training tolerates stale cross-region data |
| Shadow/canary gating on every update | Prevents silent production degradation | Adds ~45min minimum latency-to-production for good updates |
| CPU-only serving | No GPU cost/ops burden, simple autoscaling | Caps future model complexity without new hardware planning |

## 34. Failure Modes

| Failure | Symptom | Mitigation |
|---|---|---|
| Flink backpressure from a hot key | Feature staleness spikes for affected entities | Key-salting; fallback to last-known-good cached value with a staleness flag |
| Kafka broker/AZ outage | Producer errors, consumer lag spike | Multi-AZ replication (min.insync.replicas=2), automatic leader re-election, gateway retry/backoff |
| Feature store node failure | Elevated latency/errors for that shard | Automatic replica promotion; fallback to a default/neutral vector with a "degraded" flag (conservative decision, e.g. flag not auto-kick) |
| Bad label batch | Online trainer learns corrupted association | Shadow/canary gate catches the regression; label-source anomaly detection alerts independently |
| Model registry unavailable | Pods can't fetch new versions | Pods retain last successfully loaded model indefinitely; serving continues uninterrupted |
| Cross-region replication lag spike | Nightly retrain sees incomplete data | Wait for lag-below-threshold signal, or proceed with a "partial data" flag |
| Poisoning via fabricated telemetry | Online-trained model quality degrades | Per-entity update-influence capping; anomaly detection on event-source patterns |

## 35. Scaling Bottlenecks

- **10x (25M events/sec)**: Kafka partition count/broker fleet is the first bottleneck — repartitioning is disruptive, so plan higher partition counts proactively. Flink scales roughly linearly (~3,200 vCPUs) but needs RocksDB and checkpoint tuning.
- **100x (250M events/sec)**: breaks the single-region-cluster model — needs horizontal sharding of Kafka/Flink/feature-store within a region (per-title-shard sub-clusters). The feature store's 750K QPS baseline becomes 75M QPS — no single Redis/Scylla cluster handles that; needs a multi-cluster sharded-by-cohort store with a routing layer.
- **Model registry (Postgres)** likely bottlenecks before compute does — fine at current promotion frequency, but needs a more horizontally scalable metadata store at 10-100x promotion volume.
- **Online-trainer single-node fit**: at 10x label volume may not complete within the 30-60s window — would need to shard the label stream by cohort with parallel partial updates and periodic merge (parameter-server pattern).

## 36. Latency Bottlenecks

| Stage | p50 | p99 |
|---|---|---|
| Client → Edge Gateway network | 5ms | 15ms |
| Gateway auth + routing | 1ms | 3ms |
| Gateway → Inference network | 2ms | 5ms |
| Feature store read | 4ms | 15ms |
| Model inference | 1ms | 3ms |
| Business rule + serialization | 0.5ms | 2ms |
| Inference → client response network | 5ms | 15ms |
| Queueing/scheduling jitter | 2ms | 30ms |
| **Total** | **~20ms** | **~88ms** |

The dominant p99 tail contributor is queueing/scheduling jitter (GC pauses, connection-pool contention) and feature-store read tail latency — request hedging on feature reads and GC tuning (or Go/Rust for hot-path services) pay off most. Cross-AZ hops silently add 1-2ms each; pod anti-affinity keeps services co-located.

## 37. Cost Bottlenecks

- **Kafka broker fleet** (3x replication + cross-region MirrorMaker2) is typically the single largest line item — tiered storage is the primary lever.
- **Online feature store node count**: over-provisioning year-round for live-event peaks (rather than sizing to sustained load with a documented burst plan) is a common silent cost sink.
- **Cross-region data transfer** for global model training — mitigated by pre-filtering to only the columns needed before transfer.
- **Nightly full-retrain cluster**: a shared cluster sized for the largest title but run sequentially for all 8 wastes idle capacity; per-title right-sized ephemeral clusters are more efficient.
- **Prediction/feature audit logging** at 750K/sec into Iceberg is an easy-to-overlook recurring cost — retention-tier discipline and compression are the main levers.

## 38. Interview Follow-Up Questions

1. How would you detect and mitigate a bad-actor game server injecting fabricated telemetry to poison the online learning loop?
2. Walk through how you guarantee point-in-time correctness between the feature vector used for a prediction and the one later used for the training set.
3. What happens if the online feature store returns a value that's 10 seconds stale?
4. Why micro-batch (30-60s) online learning instead of true per-event streaming SGD?
5. How do you prevent an online-trained model from drifting away from the nightly baseline, and how would you detect it?
6. Your Flink job's checkpoint duration crept from 5s to 45s — walk through your diagnosis process.
7. How would you extend this system to support an embedding-based player-similarity model for smurf detection?
8. Design the canary gating logic — what statistical test decides "regressed" vs. "noise" on a 5% traffic sample over 30 minutes?
9. How does GDPR change your multi-region topology, and what happens when a global model needs cross-region training data?
10. If Kafka partition count is your first bottleneck at 10x scale, why not just over-provision partitions from day one?

## 39. Ideal Answers

1. **Poisoning mitigation**: cap any single entity's influence on a micro-batch update, run per-source anomaly detection before ingestion, require signed producer credentials on `labels.confirmed`. Shadow/canary gating is the final backstop.

2. **Point-in-time correctness**: log the exact feature vector at prediction time into `prediction_log.feature_snapshot`, keyed by `trace_id`. Join labels against that logged snapshot rather than a fresh feature-store query, avoiding future-information leakage.

3. **Stale feature handling**: `GetFeatures` returns `freshness_ms`, checked against a per-feature-set threshold. If breached, fall back to a cached last-known-good value with a "degraded" flag propagated into the decision logic.

4. **Micro-batch vs. per-event SGD**: per-event updates are maximally reactive but fragile — a single noisy/adversarial label can swing weights, with no natural gating unit. Micro-batching gives a stable, gateable unit of change that's still fast enough for the product's needs.

5. **Online-vs-baseline divergence**: track KL divergence (or a simpler prediction-distribution delta) between the online model and the nightly baseline on a shared holdout, hourly. A threshold breach alerts and can force an early full retrain.

6. **Checkpoint duration diagnosis**: check state size growth (missing TTL/window-close bug), key skew, and sink backpressure. Use Flink's checkpoint UI to localize to a specific operator before assuming a systemic cause.

7. **Adding embeddings**: introduce an embedding pipeline (e.g., two-tower model) writing vectors as additional fields into the existing feature store — no redesign needed. Add a vector DB/ANN index only if similarity search is required.

8. **Canary statistical test**: use a sequential testing approach (SPRT or Bayesian A/B) rather than a fixed-sample t-test, so regressions can be aborted early. Pre-register minimum detectable effect and alpha/power before the canary starts.

9. **GDPR and multi-region training**: EU player data and models trained on it must stay within EU infra. Default to region-local models; any global model trains only on non-EU data plus opt-in/anonymized EU aggregates, reviewed by legal.

10. **Partition over-provisioning tradeoff**: more partitions increase per-broker overhead and slow leader-election/failover, so it isn't free. Better to provision for a realistic 2-3x near-term target with a tested repartitioning runbook for later.
