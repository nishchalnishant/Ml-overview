# Cheat Detection

## 1. Problem Framing & Requirement Gathering

Design an anti-cheat system for a live-service EA title (e.g. Battlefield/Apex-style FPS or FIFA/EA Sports FC online modes) that detects cheating (aimbots, wallhacks, speedhacks, macro/scripting, stat/currency exploits, collusion/boosting) from gameplay telemetry using a combination of deterministic statistical rules, server-authoritative validation, and an ML ensemble — balancing real-time enforcement (kick/flag mid-match) against high-confidence batch ban-waves to minimize false-positive PR damage while staying ahead of an adversarial cheat-development ecosystem.

- **Business goal**: protect competitive integrity and retention (cheaters degrade experience for ~10-20 legitimate players each before being banned); reduce refund/churn cost.
- **Core tension**: real-time detection reduces cheater "playtime damage" but has looser evidence bars (higher FP risk, cheat devs A/B test detection by getting banned fast); batch ban-waves let you accumulate corroborating evidence (higher precision) but let cheaters play for days.
- **Adversarial framing**: this is not a stationary classification problem — cheat vendors patch against detection signatures within days-to-weeks, so the system must assume active adversaries probing it in production.

## 2. Functional Requirements

- FR1: Ingest per-match, per-tick gameplay telemetry (inputs, aim angles, hit registration, movement deltas, RPC calls) from game client and server.
- FR2: Run server-authoritative validation on physics/state-changing actions (position, ammo, health) rejecting impossible states outright.
- FR3: Score every player-session with statistical detectors (e.g. snap-to-head aim ratio, reaction time distribution, headshot % z-score) and ML models (sequence models over input telemetry, GNN over player-interaction graphs for boosting/collusion).
- FR4: Support real-time in-match soft actions: shadow-flag, spectator-alert to human reviewer, or immediate kick for egregious server-validation violations (speedhack, teleport).
- FR5: Support batch ban-wave pipeline: aggregate multi-match evidence, ensemble score, human-in-the-loop review queue, coordinated ban issuance.
- FR6: Maintain per-player cheat-probability score that decays/updates over time (recidivism tracking, smurf/ban-evasion linking via device/hardware fingerprint).
- FR7: Provide appeal workflow with evidence replay (store enough telemetry to reconstruct a match for human review).
- FR8: Feed confirmed-cheat labels back into training data (closed-loop retraining).
- FR9: Detect new/unknown cheat signatures via anomaly detection (unsupervised) independent of labeled-cheat classifiers.
- FR10: Rate-limit/hide detection signals from the client to reduce reverse-engineering surface (server-side-only feature computation).

## 3. Non-Functional Requirements

| Dimension | Target | Notes |
|---|---|---|
| Real-time flag latency | p99 < 500 ms per tick-window scoring | must not add perceptible input lag or matchmaking delay |
| Server-authoritative validation latency | p99 < 50 ms | inline with game server tick (16-33ms/tick budget) |
| Ban-wave batch latency | < 24 h from evidence window close to ban issuance | balances evidence quality vs cheater dwell time |
| Availability (real-time path) | 99.95% | degrade to "log-only" mode rather than block gameplay on outage |
| Availability (batch path) | 99.9% | can tolerate delay, not player-facing |
| Throughput | 2M concurrent sessions peak, ~50k telemetry events/sec/region sustained, 300k events/sec burst (launch day) | EA-scale live-service peak (e.g. Apex/Battlefield launch weekend) |
| Consistency | Eventually consistent ban state across regions within 60s | must propagate fast enough to prevent banned player rejoining in another region |
| False-positive rate (ban-wave) | < 0.02% of banned accounts overturned on appeal | brand-risk-sensitive metric |
| False-negative tolerance | accept higher FN in real-time path if batch path recovers within 24h | layered defense |
| Cost | anti-cheat compute budget ≤ 3-5% of total live-ops infra spend | ML ensemble must not be GPU-cost-prohibitive at this scale |

## 4. Clarifying Questions an Interviewer Would Expect

1. Is this PvP-only (aim/movement cheats) or does it also need economy/exploit detection (duplication, currency farming bots)?
2. Do we control the game client (can we ship kernel-level or user-mode anti-cheat drivers) or is this telemetry-only (e.g. mobile, cross-play with untrusted clients)?
3. What's the acceptable false-positive tolerance — is a wrongful ban a support-ticket cost or a PR/legal cost (esports players, streamers)?
4. Is enforcement immediate-kick, shadow-ban (matched only with other flagged players), or ban-wave only?
5. Do we need cross-title signal sharing (device fingerprint banned in one EA title informs another)?
6. What's the expected adversary sophistication — script-kiddie aimbots vs. paid, actively-maintained cheat-provider services with anti-anti-cheat evasion?
7. Regulatory constraints — GDPR/CCPA on telemetry retention, minors' data (COPPA) affecting what we can log/fingerprint?
8. Is there a competitive/esports mode requiring stricter SLAs and full replay-based manual review?

## 5. Assumptions

1. Title has 25M MAU, 2M peak concurrent players, average session 40 minutes.
2. EA owns the game client and can ship a user-mode anti-cheat agent (not kernel-level, to keep cross-platform/console parity) plus full server authority over game state.
3. Telemetry tick rate: server samples player input/state at 20 Hz (50ms) for scoring; raw client input capture at 60-128Hz retained short-term for replay only.
4. Ban-wave cadence baseline: weekly, with an expedited "high-confidence" daily wave for egregious cases.
5. Historical labeled cheat data exists (~2M confirmed-cheater sessions) from prior title generations to bootstrap supervised models.
6. Device/hardware fingerprinting is legally permitted with ToS consent, used for ban-evasion linkage, not for real-time inference (privacy-conservative).
7. Cross-platform play (PC/console) means feature parity across control schemes must be modeled (e.g., aim-assist confounds aimbot detection on console).
8. Support for 4 live regions: NA, EU, APAC, SA, each with regional game servers; central anti-cheat data plane can be single active region with regional read replicas.

## 6. Capacity Estimation

**Traffic**
- 2M concurrent sessions, telemetry sampled at 20Hz server-side → 2M × 20 = 40M events/sec theoretical max; in practice only ~15-20% of ticks produce a "scoring event" (aggregated per 1-3s window, not per-tick) → effective ingest ≈ **50k-80k events/sec sustained**, burst to **300k/sec** on launch weekends.
- Each event ≈ 1.5 KB (aim deltas, hit-reg, position, timestamp, session id) → sustained ingest bandwidth ≈ 80k × 1.5KB ≈ **120 MB/s**, burst ≈ 450 MB/s.

**Storage**
- Raw telemetry retained 14 days (for appeal replay + short-term feature windows): 80k events/s × 1.5KB × 86,400s × 14d ≈ **145 TB** hot storage (compressed columnar ~3x reduction → ~48 TB actual).
- Aggregated per-session feature vectors retained 180 days for training: 2M sessions/day (avg) × 2KB feature vector × 180d ≈ **720 GB** — trivial, keep in feature store.
- Ban-wave evidence packages (replay + scores) retained 2 years for legal/appeal: ~50k bans/week × 5MB avg replay package × 104 weeks ≈ **26 TB**.

**Model size / compute**
- Statistical detectors: negligible compute, CPU-only, <1ms per session-window.
- Sequence model (e.g. small transformer/LSTM over 128-step input-action sequences) for aimbot detection: ~15M params, fp16 ≈ 30MB — fits comfortably on CPU for batch, GPU for real-time low-latency batched inference.
- GNN for collusion/boosting over player-interaction graph: run batch-only (not real-time), daily on ~2M nodes / ~20M edges subgraph per region.
- Real-time inference load: 80k events/sec, batched into windows → effective inference QPS ≈ **5k-8k inferences/sec** (one score per session-window, not per event) after windowing/aggregation.

**GPU/CPU counts**
- Real-time scoring service: 8k inferences/sec, ~5ms/inference on GPU with batching (batch=32, ~1ms/sample amortized) → 1 GPU (T4/A10-class) handles ~5k inferences/sec comfortably → need **~2-3 GPUs per region** for real-time path with headroom, ×4 regions = **~10-12 GPUs** total, plus 2x for redundancy/failover ≈ **20-24 GPUs** fleet-wide.
- Batch ban-wave pipeline (GNN + ensemble re-score over full weekly cohort ~14M distinct weekly players): run as nightly/weekly Spark+GPU batch job, **16-32 GPUs for a 4-6 hour window**, spot-priced.
- Server-authoritative validation: pure CPU, runs inline on game server fleet (no separate hardware) — amortized into existing game-server compute.

## 7. High-Level Architecture

```
                         ┌───────────────────────────┐
                         │      Game Client            │
                         │ (input capture, user-mode   │
                         │  anti-cheat agent)           │
                         └───────────┬──────────────────┘
                                     │ signed telemetry (TLS)
                                     ▼
                    ┌────────────────────────────────┐
                    │     Game Server (authoritative)  │
                    │  - physics/state validation       │
                    │  - inline rule checks (<50ms)      │
                    │  - emits telemetry events          │
                    └───────┬───────────────┬─────────┘
        immediate kick RPC  │               │ telemetry stream
        (egregious violation)│              ▼
                    ┌────────┘     ┌───────────────────────┐
                    │              │  Ingestion Gateway      │
                    │              │  (Kafka producers,      │
                    │              │   schema validation)     │
                    │              └───────────┬─────────────┘
                    │                          ▼
                    │              ┌───────────────────────┐
                    │              │   Kafka: telemetry.*    │
                    │              │  topics (partitioned     │
                    │              │  by session_id)          │
                    │              └───────┬────────┬────────┘
                    │                      ▼        ▼
                    │        ┌─────────────────┐  ┌──────────────────────┐
                    │        │ Real-Time        │  │ Stream Aggregator      │
                    │        │ Feature Windower  │  │ (Flink) -> Feature      │
                    │        │ (Flink, 1-3s win) │  │ Store (offline write)   │
                    │        └────────┬─────────┘  └──────────┬─────────────┘
                    │                 ▼                       ▼
                    │      ┌────────────────────┐   ┌───────────────────────┐
                    │      │ Real-Time Scoring    │   │  Feature Store          │
                    │      │ Service (statistical  │   │  (online: Redis/       │
                    │      │ rules + light ML,     │◄──┤   offline: Parquet/S3) │
                    │      │ GPU-batched inference)│   └───────────┬───────────┘
                    │      └────────┬──────────────┘               │
                    │               ▼                              │
                    │   ┌─────────────────────┐                    │
                    │   │ Decision Engine       │                    │
                    │   │ (flag/shadow-ban/kick │                    │
                    │   │  threshold policy)     │                    │
                    │   └────┬─────────┬────────┘                    │
                    │        │ kick RPC│ flag event                  │
                    ◄────────┘         ▼                              │
                              ┌────────────────────┐                  │
                              │ Kafka: cheat.flags   │                  │
                              └─────────┬────────────┘                  │
                                        ▼                              ▼
                              ┌───────────────────────────────────────────┐
                              │        Batch Ban-Wave Pipeline               │
                              │  (Spark/Airflow: ensemble re-score,          │
                              │   GNN collusion graph, evidence aggregation) │
                              └───────────┬──────────────────┬──────────────┘
                                          ▼                  ▼
                              ┌─────────────────┐   ┌───────────────────────┐
                              │ Human Review Queue │   │ Ban Issuance Service    │
                              │ (borderline cases)  │   │ (writes ban store,      │
                              └─────────────────┘   │  propagates cross-region)│
                                                      └───────────┬───────────┘
                                                                  ▼
                                                     ┌───────────────────────┐
                                                     │ Ban Store (global,      │
                                                     │ replicated <60s)         │
                                                     └───────────────────────┘

     Feedback loop: confirmed bans + appeal outcomes -> Training Data Lake -> Model Training -> Model Registry -> deployed to Real-Time Scoring Service & Batch Pipeline
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| Game Server Validator | Inline authoritative check of physics/state (position deltas, ammo count, cooldown timers) | In-process function call, sub-tick | Scales with game-server fleet (per-match instance) |
| Ingestion Gateway | Validate schema, auth, rate-limit, produce to Kafka | gRPC/HTTP2 from game servers | Stateless, horizontal pods behind LB |
| Kafka Telemetry Cluster | Durable, ordered (per-session) event bus | Kafka protocol, topics partitioned by `session_id` hash | Partitions/brokers scale with throughput |
| Flink Stream Aggregator | Windowed feature computation (1-3s tumbling/sliding windows) per session | Kafka consumer group → Kafka producer + Feature Store sink | Parallelism = Kafka partitions |
| Real-Time Scoring Service | Runs statistical detectors + lightweight ML ensemble on windowed features | gRPC internal API, batched GPU inference | GPU pods autoscaled on queue depth |
| Decision Engine | Applies policy thresholds (kick/shadow-ban/flag-for-review), rate-limits actions per player | Internal API consuming scores | Stateless, CPU, horizontal |
| Feature Store | Online (low-latency) + offline (training) feature serving | Redis (online) / Parquet on S3 (offline), point-in-time join keys | Redis cluster sharded by player_id |
| Batch Ban-Wave Pipeline | Aggregates multi-session evidence, re-scores with full ensemble incl. GNN, produces ban recommendations | Airflow DAG, Spark jobs | Spark executors autoscale on cluster |
| Human Review Console | UI for reviewers to inspect replay + scores, approve/reject bans | Internal web app + replay service API | Scales with reviewer headcount, not infra-bound |
| Ban Issuance Service | Writes authoritative ban record, propagates to all regions, triggers session termination | REST API, event-driven propagation | Stateless, low QPS, high consistency requirement |
| Replay/Evidence Service | Reconstructs match state from raw telemetry for human review & appeals | Batch job on-demand, reads from cold telemetry store | On-demand, spiky — serverless/batch |
| Device Fingerprint Service | Hardware/HWID fingerprinting for ban-evasion linkage | Internal API, called at login | Stateless, cached lookups |
| Model Training Pipeline | Retrains statistical thresholds + ML models on labeled outcomes | Airflow + distributed training (Horovod/PyTorch DDP) | GPU cluster, batch/spot |
| Model Registry | Versioned model artifacts, approval gates | MLflow/internal registry API | Low QPS, HA required |

## 9. API Design

**Telemetry ingestion (game server → gateway)**
```
POST /v1/telemetry/events
Headers: Authorization: Bearer <server-service-token>, X-Region, X-Schema-Version
Body:
{
  "session_id": "uuid",
  "player_id": "uuid",
  "match_id": "uuid",
  "tick": 8213,
  "ts_server_ms": 1735680000123,
  "events": [
    {"type": "aim_delta", "yaw": 12.4, "pitch": -3.1, "dt_ms": 16},
    {"type": "hit_register", "target_id": "uuid", "hitbox": "head", "weapon": "R99"},
    {"type": "position", "x": 1023.4, "y": -88.2, "z": 12.0, "velocity": 6.7}
  ]
}
Response 202: {"accepted": true, "trace_id": "..."}
```

**Real-time score query (internal, Decision Engine → Scoring Service)**
```
POST /internal/v1/score
Body: {"session_id": "uuid", "window_end_ts": 1735680003000}
Response 200:
{
  "session_id": "uuid",
  "scores": {"aimbot_p": 0.82, "speedhack_p": 0.01, "statistical_anomaly_z": 4.1},
  "ensemble_score": 0.77,
  "model_version": "aimbot-seq-v14",
  "recommended_action": "shadow_flag"
}
```

**Decision action (Decision Engine → Game Server)**
```
POST /v1/enforcement/action
Body: {"session_id":"uuid","action":"kick","reason_code":"SPEEDHACK_SERVER_VALIDATION","evidence_ref":"s3://..."}
Response 200: {"applied": true, "ts": "..."}
```

**Ban-wave query/appeal API (public-facing, versioned)**
```
GET /v2/players/{player_id}/ban-status
Response 200: {"banned": true, "ban_id":"uuid","reason_category":"aim_assistance","issued_at":"...","appeal_eligible": true}

POST /v2/appeals
Body: {"ban_id":"uuid","player_statement":"..."}
Response 202: {"appeal_id":"uuid","status":"under_review","sla_hours":72}
```

**Versioning**: URI-based (`/v1`, `/v2`) for external/public APIs; internal APIs versioned via `model_version` field + protobuf schema evolution (backward-compatible field addition only, deprecation window 90 days).

| Endpoint | Method | Consumer | Latency SLA |
|---|---|---|---|
| /v1/telemetry/events | POST | Game server | p99 < 20ms (async accept) |
| /internal/v1/score | POST | Decision Engine | p99 < 300ms |
| /v1/enforcement/action | POST | Decision Engine → Game server | p99 < 100ms |
| /v2/players/{id}/ban-status | GET | Client/Support tools | p99 < 150ms |
| /v2/appeals | POST | Player-facing web | p99 < 500ms |

## 10. Database Design

| Store | Type | Used For | Why |
|---|---|---|---|
| Kafka | Log/streaming | Telemetry event bus | Ordered per-partition, replay for reprocessing, decouples producers/consumers |
| Raw Telemetry Lake | Columnar (Parquet on S3, partitioned by `date/region/session_id_prefix`) | 14-day hot replay data + long-term cold archive | Cheap, scan-efficient for batch/GNN jobs, compresses well (repetitive numeric telemetry) |
| Feature Store (online) | Redis Cluster, key = `player_id:session_id` | Low-latency real-time feature lookup | Sub-ms reads, TTL-based eviction |
| Feature Store (offline) | Parquet/Delta on S3 | Training data, point-in-time joins | Time-travel queries for correct label alignment |
| Ban Store | Distributed relational (e.g. Spanner/CockroachDB) or DynamoDB global tables | Authoritative ban records, cross-region consistency | Needs strong-enough consistency + global replication; CockroachDB gives serializable consistency with multi-region writes |
| Player Profile / Cheat-Score History | Wide-column store (Cassandra/Bigtable), key = `player_id`, clustering by `timestamp` | Rolling cheat-probability history, recidivism tracking | High write throughput, time-series access pattern |
| Evidence/Replay Metadata | Document store (MongoDB) or relational | Ban-wave case packages, reviewer annotations | Flexible schema (varying evidence types per case) |
| Device Fingerprint Store | Key-value (DynamoDB), key = `hwid_hash` | Ban-evasion linkage | Simple lookup, high read QPS at login |

**Partitioning/sharding**:
- Kafka: partition by `session_id` hash → preserves per-session ordering, spreads load.
- Ban Store: shard by `player_id` hash, geo-partitioned with region-affinity writes, async cross-region replication for reads (RPO < 60s target — see DR section).
- Raw telemetry lake: partitioned by `date` (lifecycle/TTL management) then `region` then `session_id` prefix (parallel scan).
- Cassandra profile store: partition key `player_id`, clustering key `event_ts DESC` for efficient "recent history" reads.

## 11. Caching

| Cache | Contents | Strategy | Invalidation |
|---|---|---|---|
| Online Feature Store (Redis) | Rolling window features (last 30-120s of aim/movement stats per session) | Write-through from Flink aggregator | TTL = session length + 5 min grace |
| Ban-status cache | Player ban lookup for game-server session-start checks | Cache-aside, read-through on miss to Ban Store | TTL 30s (must be short — banned player must not slip in); explicit invalidation push on new ban issuance via pub/sub |
| Model artifact cache | Loaded model weights in scoring-service pods | Local in-memory, refreshed on registry version bump | Pod-level cache-aside with background poller (every 60s) checking registry version |
| Device fingerprint cache | HWID → known-ban-linkage lookups | Cache-aside, Redis | TTL 1h, explicit invalidate on new fingerprint-ban link |
| Reviewer console session data | Evidence package pre-fetch for queue | Write-through on queue enqueue | Invalidate on review completion |

**Cache-aside dominates** for correctness-sensitive lookups (ban status) since staleness risk (banned player still playing) must be bounded and explicitly minimized; **write-through** for feature store because staleness there only affects scoring quality, not security — acceptable to trade a little consistency for lower write latency on the hot path.

## 12. Queues & Async Processing

| Queue | Purpose | Delivery Semantics | DLQ Handling |
|---|---|---|---|
| `telemetry.raw.*` (Kafka) | Ingest buffer from game servers | At-least-once (idempotent consumers dedupe on `(session_id, tick)`) | Poison messages (schema violations) → `telemetry.dlq`, alert + quarantine, do not block partition |
| `cheat.flags` (Kafka) | Real-time flags from scoring service to decision engine & downstream ban-wave input | At-least-once, consumer dedupe via flag_id | Failed enforcement actions retried 3x with backoff, then → `enforcement.dlq`, paged to on-call |
| `ban.issuance.propagation` | Fan-out ban record to all regional read replicas | Exactly-once semantics required (use transactional outbox pattern from Ban Store) | Failures alert immediately — a ban that fails to propagate is a security gap |
| `training.label.updates` | Feed confirmed bans/appeal-overturns back to training pipeline | At-least-once, idempotent upsert by label_id | Reprocessed nightly from DLQ, non-urgent |
| `replay.reconstruction.requests` | On-demand async job to build replay for human review | At-least-once | Retry 5x, then flag case as "evidence unavailable", route to manual raw-log pull |

- **Exactly-once** is reserved only for ban issuance propagation (security-critical, low volume) via transactional outbox + idempotency keys — not attempted on high-volume telemetry (cost-prohibitive, unnecessary).
- Dead-letter queues always paired with a metrics counter + alert threshold (>0.1% DLQ rate triggers page).

## 13. Streaming & Event-Driven Architecture

**Topics**:
- `telemetry.aim` — aim/input deltas, high volume, key=`session_id`, 24 partitions/region.
- `telemetry.movement` — position/velocity, key=`session_id`.
- `telemetry.combat` — hit registration, damage, weapon events, key=`match_id`.
- `cheat.flags` — real-time flag events, key=`player_id`.
- `cheat.ban.issued` — ban decisions, key=`player_id`, consumed by game servers (session kill), profile service, cross-region replicators.
- `training.labels` — confirmed outcomes (ban upheld/overturned), key=`player_id`.

**Event schema (Avro, registry-managed)**:
```json
{
  "type": "record", "name": "AimDeltaEvent",
  "fields": [
    {"name": "session_id", "type": "string"},
    {"name": "player_id", "type": "string"},
    {"name": "tick", "type": "long"},
    {"name": "yaw_delta", "type": "float"},
    {"name": "pitch_delta", "type": "float"},
    {"name": "dt_ms", "type": "int"},
    {"name": "schema_version", "type": "int", "default": 1}
  ]
}
```

**Consumer groups**:
- `flink-feature-windower` (parallelism = partition count) — real-time feature aggregation.
- `batch-lake-sink` — Kafka Connect S3 sink, writes Parquet for offline store.
- `decision-engine-consumer` — consumes `cheat.flags`, applies policy.
- `ban-wave-collector` — accumulates flags per player over rolling 7-day window for batch scoring.
- `cross-region-replicator` — consumes `cheat.ban.issued`, pushes to other regions' Ban Store replicas.

Schema evolution: additive-only fields with defaults, enforced by schema registry compatibility mode `BACKWARD`.

## 14. Model Serving

- **Framework**: NVIDIA Triton Inference Server for the real-time path (supports dynamic batching, multi-model, ONNX/TensorRT-optimized sequence models); statistical detectors run as plain CPU microservices (no serving framework overhead needed — they're closed-form math, not learned weights).
- **Batching**: dynamic batching window of 8-15ms, max batch size 32, tuned so p99 latency budget (300ms internal SLA) is never at risk — batching amortizes GPU cost ~5-8x vs. per-request inference.
- **Multi-model**: Triton hosts multiple model types concurrently per GPU — aimbot sequence model, speedhack classifier (mostly server-validation-based, lightweight), macro/scripting detector — sharing GPU memory via Triton's model concurrency, avoiding one-GPU-per-model waste.
- **Hardware**: T4/A10-class GPUs for real-time (cost-efficient inference-optimized, not training-grade); A100 spot pool for batch GNN/ensemble re-scoring where throughput over a large graph matters more than per-inference latency.
- **Model formats**: exported to ONNX/TensorRT for the sequence models to minimize latency; GNN batch job runs natively in PyTorch Geometric on Spark-orchestrated GPU nodes (no need for low-latency serving format there).
- Canary model versions served on a small percentage of Triton pods (see Canary section) before full rollout.

## 15. Feature Store

- **Online store (Redis Cluster)**: rolling-window aggregates computed by Flink — e.g. `headshot_ratio_60s`, `snap_angle_velocity_p95_30s`, `input_entropy_10s`. Keyed by `session_id`, TTL-bound to session lifetime + grace period. Read by Real-Time Scoring Service, sub-millisecond lookup.
- **Offline store (Parquet/Delta on S3)**: full historical feature vectors per session/player, used for training and batch ban-wave re-scoring. Includes longer-window aggregates (7-day, 30-day rolling cheat-probability, cross-match statistical baselines).
- **Point-in-time correctness**: training pipeline joins labels (ban issued at time T) against feature snapshots as they existed *before* T only — enforced via Delta Lake time-travel queries (`VERSION AS OF` / `TIMESTAMP AS OF`) to prevent label leakage (e.g. must not use post-ban behavioral change as a feature for predicting the ban).
- **Feature parity**: the same feature computation logic (shared Flink/Spark UDF library) is used for both online windowed features and offline batch features to avoid train/serve skew — critical given this system's adversarial nature (skew here doesn't just hurt accuracy, it creates exploitable blind spots).
- Feature registry documents owner, freshness SLA, and drift-monitoring status per feature (important given 40+ behavioral features feeding the ensemble).

## 16. Vector Database

**N/A for core detection path** — cheat detection here is primarily sequence/tabular/graph-based (aim telemetry, statistical aggregates, interaction graphs), not similarity-search-over-embeddings. No nearest-neighbor retrieval is required for the primary detection flow.

**Partial applicability**: a vector store (e.g. pgvector or a lightweight ANN index like HNSW) *could* be used for one adjacent sub-problem — behavioral-signature similarity search to cluster "same cheat provider" fingerprints (embedding a player's input-pattern signature and finding nearest neighbors among known cheat clusters). If implemented, HNSW is the right ANN choice here (moderate dataset size, need for high recall, infrequent writes) over IVF-PQ (which favors billion-scale corpora we don't have — cheat-signature corpus is in the low millions, not billions). This is an optional enrichment for the batch ban-wave pipeline, not core infrastructure.

## 17. Embedding Pipelines

**Partially applicable** — not embeddings-as-primary-artifact like a recommender system, but two supporting uses:
- **Behavioral embeddings**: a small encoder (e.g. 1D-CNN or transformer encoder) compresses a player's rolling input-sequence window into a fixed-size vector for the above similarity-clustering use case (cheat-provider fingerprinting) and for the recidivism/ban-evasion linkage model (comparing a new account's play-style embedding against known-banned players' embeddings).
- **Graph node embeddings**: the collusion/boosting GNN produces node embeddings per player representing their interaction-graph position (who they party with, trade with, boost for) — consumed downstream by the ensemble as engineered features, not exposed as a general-purpose embedding API.
- No large-scale text/image embedding pipeline is needed (this isn't a content/NLP system) — this section is intentionally narrow in scope relative to, e.g., a recommendation or search system chapter.

## 18. Inference Pipelines

**Real-time request lifecycle (end-to-end)**:

```
t=0ms     Player performs action -> Game Client captures input @ 60-128Hz
t=~5ms    Client forwards raw input signal to Game Server (authoritative)
t=~10ms   Game Server validates action against physics rules (position/velocity bounds,
          rate limits) -- INLINE, <50ms budget
              |-- FAIL (impossible state) --> immediate kick RPC, log incident, DONE (~15ms total)
              |-- PASS --> proceed
t=~15ms   Game Server emits telemetry event -> Ingestion Gateway (async, non-blocking)
t=~40ms   Ingestion Gateway validates schema/auth, produces to Kafka telemetry topic
t=~100ms  Flink windower aggregates last 1-3s of events into feature vector,
          writes to Redis (online feature store)
t=~150ms  Feature vector triggers scoring request (event-driven or on 1s tick) to
          Real-Time Scoring Service
t=~180ms  Triton batches request (8-15ms dynamic batch window), runs statistical
          detectors (CPU, <1ms) + ML ensemble (GPU, ~5ms amortized)
t=~250ms  Ensemble score returned to Decision Engine
t=~280ms  Decision Engine applies policy thresholds:
              score > 0.95 -> kick + evidence snapshot
              0.6 < score <= 0.95 -> shadow-flag (matched with other flagged players) + queue for ban-wave
              score <= 0.6 -> log only, feed to offline aggregation
t=~300ms  Action (if any) sent back to Game Server via enforcement API
--------------------------------------------------------------------------
TOTAL: ~300ms p50 real-time path (well within 500ms p99 SLA)
```

**Batch ban-wave lifecycle**: nightly/weekly Airflow DAG pulls 7-day window of `cheat.flags` + raw telemetry aggregates → re-scores with full ensemble (including GNN collusion features, cross-session history, device fingerprint linkage) → produces ranked case list → borderline cases (score 0.5-0.85) to human review queue, high-confidence cases (>0.85) auto-queued for ban issuance pending final automated sanity check (e.g., not a pro player flagged by a false-positive-prone new model version) → Ban Issuance Service writes record → propagation fan-out to all regions.

## 19. Training Pipelines

- **Data prep**: labeled dataset = confirmed bans (positive) + random sample of non-banned high-playtime accounts (negative, weighted to offset extreme class imbalance — cheaters are <1-2% of population). Feature vectors pulled from offline feature store with point-in-time joins (see Section 15).
- **Class imbalance handling**: focal loss or class-weighted cross-entropy; negative sampling ratio tuned ~20:1 rather than full population (100M+ negative sessions would dominate compute for no accuracy benefit).
- **Model types trained**:
  - Statistical thresholds: periodically recalibrated (not "trained" in ML sense) against fresh population distributions per title/patch/season (aim-assist changes, new weapons shift baseline stats).
  - Sequence model (aimbot/macro detection): supervised, PyTorch, trained via DDP across 8 GPUs on ~2M labeled sessions, ~4-hour training run.
  - GNN (collusion/boosting): trained on constructed interaction graphs with known-boosting-ring labels, PyTorch Geometric, single-node multi-GPU (graph doesn't shard trivially across nodes at this scale).
  - Unsupervised anomaly detector (autoencoder or isolation forest over behavioral features): retrained on all-population data (no labels needed) to catch novel/zero-day cheat patterns statistical+supervised models miss.
- **Orchestration**: Airflow DAG triggers training jobs on a Kubernetes-based GPU cluster (Kubeflow Training Operator for distributed PyTorch jobs), artifacts pushed to Model Registry (MLflow) with automatic evaluation-gate (must beat current production model on held-out precision/recall before promotion).
- **Distributed training**: DDP for the sequence model (data-parallel, embarrassingly parallelizable across labeled sessions); GNN training uses graph-sampling (neighbor sampling, e.g. GraphSAGE-style) to keep mini-batches tractable rather than full-graph training.

## 20. Retraining Strategy

- **Cadence**: sequence/statistical models retrained **weekly** (aligned with ban-wave cadence, fresh labels flow in continuously); GNN retrained **bi-weekly** (graph construction is more expensive); unsupervised anomaly detector retrained **daily** (cheap, catches fast-moving adversary shifts).
- **Triggers** (event-driven retraining, not just cadence):
  - Precision on ban-wave review queue drops below 97% (reviewers overturning more auto-flagged cases) → trigger immediate retrain + investigation.
  - New game patch/season ships (weapon rebalance, new movement mechanic) → mandatory statistical threshold recalibration before next ban-wave, since baselines shift.
  - Spike in appeal-overturn rate (>0.05% of bans) → freeze ban-wave auto-issuance, trigger retrain + root-cause.
  - New cheat signature detected by anomaly detector with high volume (>500 sessions/day matching a novel unlabeled cluster) → fast-track labeling (human review sample) + targeted retrain.
- Full model redeploy always gated by canary (Section 33) — no direct hot-swap to 100% traffic.

## 21. Drift Detection

| Drift Type | What's Monitored | Metric | Threshold |
|---|---|---|---|
| Data drift (input distribution) | Feature distributions (aim-angle variance, reaction-time distribution, headshot %) per patch/season | Population Stability Index (PSI) per feature | PSI > 0.2 → alert; > 0.3 → auto-trigger recalibration |
| Concept drift (adversarial) | Ensemble score distribution for known-cheat-provider clusters over time (cheat vendors patch to evade detection) | Rolling precision/recall on a held-out "sentinel" labeled sample refreshed weekly | Recall drop > 5 points week-over-week → alert |
| Label drift | Ratio of ban-wave auto-approved vs. human-overturned | Overturn rate | > 0.5% overturned → pause auto-issuance |
| Feature pipeline drift | Null rate / schema-violation rate in Flink windower output | % events dropped/malformed | > 1% → page on-call (likely a client/schema regression, not a cheat trend) |
| Cross-platform drift | Score distribution divergence between PC vs. console cohorts (aim-assist confound) | KL divergence between cohort score distributions | Divergence exceeding baseline by 2x → flag for platform-specific threshold review |

Drift monitoring runs as a scheduled batch job (daily) comparing current-week feature/score distributions against a rolling 90-day baseline, visualized in the monitoring dashboard (Section 22) with automatic PSI computation via a shared internal drift-monitoring library.

## 22. Monitoring

| Category | Metrics |
|---|---|
| Infra | Kafka consumer lag per topic/partition, Flink checkpoint duration/failures, Triton GPU utilization & queue depth, Redis hit rate & memory pressure, Ban Store replication lag per region |
| Model quality | Precision/recall on rolling sentinel labeled set, PSI per feature, score distribution histograms per model version, calibration curves (predicted prob vs. actual ban-upheld rate) |
| Business | Bans issued/day, appeal rate, appeal-overturn rate, estimated "cheater playtime prevented" (session-time cut short by kicks/bans), player-reported-cheater correlation with system flags (validates against community sentiment) |
| Enforcement pipeline | Time-to-flag (real-time path), time-to-ban (batch path), human review queue depth & reviewer SLA adherence |
| Adversarial signal | Volume of sessions matching unsupervised anomaly clusters (proxy for new/unknown cheat tooling emerging) |

Dashboards: Grafana (infra + drift), internal ML observability tool (model quality, calibration), BI dashboard (business/trust-and-safety metrics for leadership reporting).

## 23. Alerting

| Condition | Threshold | Severity | Routing |
|---|---|---|---|
| Real-time scoring service p99 latency | > 500ms for 5 consecutive min | High | Page on-call SRE (anti-cheat infra rotation) |
| Kafka consumer lag (telemetry topics) | > 60s sustained | High | Page on-call, risk of missed real-time detections |
| Ban propagation failure | any failure in `ban.issuance.propagation` | Critical | Immediate page — security gap, banned player could rejoin |
| Appeal-overturn rate | > 0.5% rolling 7-day | Critical | Page ML on-call + auto-pause ban-wave auto-issuance, escalate to trust & safety lead |
| PSI drift on core feature | > 0.3 | Medium | Ticket to ML team, review within 24h, no auto-page |
| DLQ rate (any topic) | > 0.1% of volume | Medium | Ticket + Slack alert to data platform on-call |
| GPU pool utilization | > 85% sustained 15 min | Medium | Autoscaling should have already fired; alert if HPA/KEDA didn't respond |
| Sentinel recall drop | > 5 pts week-over-week | High | Page ML on-call, likely adversarial evasion event |
| Human review queue depth | > 2x normal backlog | Low-Medium | Slack alert to trust & safety ops lead (staffing issue) |

On-call routing: infra alerts → SRE rotation; model-quality/drift alerts → ML platform on-call; business/policy alerts (overturn rate) → joint ML + Trust & Safety escalation (dual paging, since it has both technical and policy/PR dimensions).

## 24. Logging

- **Structured logging**: all services emit JSON logs with mandatory fields (`trace_id`, `session_id` hashed/tokenized where PII-adjacent, `service`, `model_version`, `latency_ms`, `outcome`) — correlated end-to-end via `trace_id` propagated from client through server, gateway, Kafka headers, scoring service, decision engine.
- **PII handling**: raw telemetry (position, inputs) is not PII itself, but `player_id`, device fingerprint (HWID hash), and IP are — these are encrypted at rest (Section 25) and access-scoped; logs used for debugging/observability reference `player_id` only via a reversible tokenization scheme accessible solely to authorized services (not plaintext in general log aggregation like Splunk/ELK — a separate restricted-access log stream handles anything with real player identifiers).
- **Retention**:
  - Debug/ops logs: 30 days.
  - Telemetry (raw): 14 days hot, then cold-archived to Glacier-class storage for up to 1 year (appeal window + legal hold cases), then deleted unless under active legal hold.
  - Ban evidence packages: 2 years (appeals/legal), access-audited.
  - Training labels: retained indefinitely in de-identified form (player_id replaced with a stable pseudonymous training-key) for longitudinal model evaluation.
- Access to identity-linked logs requires a documented business justification + audit trail (internal access-review tooling), consistent with GDPR data-minimization principle.

## 25. Security

- **Threat model specific to this system**:
  - **Client-side reverse engineering**: cheat developers decompile the client/anti-cheat agent to learn detection signatures → mitigate by keeping all scoring logic server-side (client only captures/forwards signed telemetry, never exposes thresholds/model weights).
  - **Telemetry spoofing**: malicious client sends fabricated "clean" telemetry alongside cheat-assisted inputs → mitigate via server-authoritative validation (client telemetry is corroborating evidence, not the sole truth) + integrity signing of the anti-cheat agent's payload (detect tampering with the agent itself).
  - **Replay/tampering of ban decisions**: attacker attempts to intercept/forge enforcement RPCs → mutual TLS between game server and internal services, signed enforcement action payloads.
  - **Model extraction/probing attacks**: adversary systematically varies inputs to map decision boundaries → rate-limit and monitor for statistically anomalous "probing" session patterns from a single account/device; obscure exact thresholds (never return raw scores to any client-reachable endpoint).
  - **Insider threat**: reviewer or engineer leaking detection signatures or issuing fraudulent unbans → RBAC + audit logging on all ban-store writes and review-console actions, dual-approval for high-profile unban actions.
- **Encryption**: TLS 1.3 in transit everywhere (client↔server, service↔service, service↔Kafka via SASL_SSL); AES-256 at rest for Ban Store, Feature Store (identity-linked fields), and evidence packages; field-level encryption for device fingerprint hashes.
- **Data minimization**: real-time scoring service operates on tokenized `session_id`, not raw `player_id`, wherever feasible to limit blast radius of a compromised scoring pod.

## 26. Authentication

- **Service-to-service**: mTLS with short-lived certs (SPIFFE/SPIRE-issued identities) between game servers, ingestion gateway, Kafka, scoring service, decision engine, ban issuance service — no shared static API keys internally.
- **Game server → anti-cheat platform**: server-issued service tokens (rotated hourly) scoped to specific API permissions (telemetry-write only; no server can directly write ban records).
- **End-user (player-facing appeal API)**: standard EA account OAuth2/OIDC session tokens, scoped to the authenticated player's own `player_id` only (cannot query other players' ban status).
- **Human reviewer console**: SSO (internal IdP) + RBAC roles (reviewer, senior-reviewer/approver, admin) with step-up auth (hardware key/MFA) required for ban-issuance approval actions given the sensitivity.
- **Device fingerprinting**: consent captured via ToS at account creation; fingerprint service authenticated via the same service-token scheme as other internal services, never client-callable directly.

## 27. Rate Limiting

| Surface | Limit | Algorithm | Rationale |
|---|---|---|---|
| Telemetry ingestion per session | 200 events/sec/session | Token bucket | Prevents a compromised/malicious client from flooding ingestion to hide signal in noise or DoS the pipeline |
| Appeal submission (per player) | 3 appeals per ban, 10/day account-wide | Fixed window | Prevent appeal-spam abuse of human review queue |
| Ban-status query API | 100 req/min/IP (public), unlimited for internal service auth | Sliding window log | Protect against scraping/enumeration of ban statuses |
| Internal scoring API | Backpressure via queue depth, not a hard rate limit | Leaky bucket at Triton dynamic-batch queue | Prioritize graceful degradation (see below) over hard rejection |
| Login-time fingerprint lookup | 20 req/sec/service-caller | Token bucket | Protects fingerprint store from runaway retry storms |

- **Degradation policy**: if the real-time scoring path is overloaded beyond rate limits/queue capacity, the Decision Engine fails open to "log-only" mode (no kicks issued) rather than blocking gameplay or introducing latency — prioritizes player experience over real-time enforcement during incidents, relying on the batch ban-wave path as the safety net.

## 28. Autoscaling

- **Real-Time Scoring Service (Triton GPU pods)**: KEDA-based autoscaling on Kafka consumer-lag metric for `cheat.flags`-adjacent windowed-feature-ready queue, and on Triton's internal queue-depth/GPU-utilization metric — scale out when queue wait time p95 > 50ms, target GPU utilization band 60-75% (headroom for batch-window bursts).
- **Ingestion Gateway / Flink Windower**: HPA on CPU utilization (target 60%) plus custom Kafka-lag metric via KEDA ScaledObject — Flink parallelism scales with partition count reassignment during planned scale events (requires care: repartitioning is not instant, so pre-scale ahead of known launch-day traffic via scheduled scaling, not purely reactive).
- **Batch Ban-Wave Pipeline (Spark on GPU)**: cluster autoscaling (e.g. via a managed Spark-on-Kubernetes operator) scales executor count 0→N at DAG start, tears down to 0 after job completion — cost-driven, not latency-driven, since it's an offline nightly/weekly job.
- **Ban Issuance Service**: low, steady QPS — fixed small pod count (3-5 for HA) rather than autoscaled, since throughput is trivial and consistency/availability matter more than elasticity here.
- **VPA**: applied to Flink task managers and the GNN batch job containers to right-size memory requests (graph batch jobs have high, spiky memory needs that are hard to guess statically).

## 29. Cost Optimization

- **Spot/preemptible instances**: batch ban-wave Spark/GNN jobs run entirely on spot GPU pools (tolerant of preemption — checkpointed, restartable) — ~60-70% cost reduction vs. on-demand for the ~16-32 GPU × 4-6hr weekly workload.
- **Model distillation**: real-time sequence model distilled from a larger offline "teacher" ensemble (which includes the GNN and heavier features not available in real-time) into a small student model that only needs real-time-available features — keeps real-time GPU footprint (~20-24 GPUs) far smaller than it would be running the full ensemble online.
- **Dynamic batching**: Triton batching amortizes GPU cost ~5-8x vs. naive per-request inference (Section 14) — single largest real-time-path lever.
- **Statistical-first triage**: cheap CPU-only statistical rules run first and only escalate to GPU ML scoring for sessions crossing a low-confidence threshold — avoids running expensive ML inference on the ~80%+ of sessions with no anomalous signal at all.
- **Tiered storage**: raw telemetry ages from hot (S3 Standard, 14 days) → cold (Glacier-class, up to 1 year) → deleted, cutting storage cost roughly 80% for data past the active-appeal window.
- **Feature store TTLs**: aggressive Redis TTL tuned to session lifetime avoids paying for unbounded online-store growth.
- **Right-sized GPU tier**: T4/A10 (inference-optimized, cheaper) for real-time path; reserve A100 spend only for the batch GNN job where throughput-per-dollar matters more than raw latency.

## 30. Operational Concerns (Deployment, Reliability, Infra)

At SDE2 scope, treat this as a checklist rather than a design exercise: **backups** (automated snapshots of the model registry, feature store, and any stateful service, with a tested restore path), **rollback** (every deploy must be revertible to the last-known-good version — the model registry and CI/CD pipeline should make this a one-command operation), **canary/blue-green rollout** (shift a small percentage of traffic first, watch error rate and key business/model metrics, then ramp), and **basic observability** (dashboards + alerts on latency, error rate, and the top 2-3 model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level infra-architecture concerns — worth knowing they exist, not worth rehearsing the manifests.

## 38. Why This Architecture

- **Layered enforcement (real-time + batch)** directly addresses the core tension in Section 1: real-time stops egregious, server-verifiable cheats (speedhack, teleport) instantly via cheap authoritative checks, while the slower, evidence-richer batch path handles subtler statistical/ML-only signals (aimbot, macro) where false-positive cost is too high to act on with a single session's data.
- **Server-authoritative validation as the first line** minimizes reliance on ML for the cheats that don't need it (physics violations are deterministic, not probabilistic) — reserves the more expensive/fallible ML ensemble for genuinely ambiguous behavioral signals.
- **Event-driven streaming backbone (Kafka/Flink)** decouples telemetry ingestion rate from scoring compute rate, letting the system absorb launch-day bursts (300k events/sec) without back-pressuring the game servers themselves.
- **Statistical-first triage before ML** keeps GPU spend proportional to actually-suspicious traffic (~20% of sessions), not all traffic — necessary at this scale (2M concurrent) to keep the anti-cheat GPU fleet in the tens, not hundreds.
- **Fail-open design** (Section 27) reflects a product-priority: an anti-cheat outage must never become a gameplay-availability outage — cheating is a tolerable-in-the-short-term problem the batch path recovers from; blocked matchmaking is not.

## 39. Alternative Architectures

| Alternative | Description | Why Rejected / When Preferred |
|---|---|---|
| Kernel-level anti-cheat driver (e.g. Vanguard/EAC-style) | Client-side kernel driver with deep OS visibility, real-time process/memory scanning | Rejected as sole strategy here per Assumption 2 (cross-platform/console parity, privacy/support burden); would be preferred for a PC-only, highly competitive/esports title where kernel-level visibility (detecting the cheat process itself, not just its gameplay effects) is worth the support cost and player trust friction |
| Pure rules-engine (no ML) | Only deterministic statistical thresholds, no learned models | Simpler, more explainable, easier to defend in appeals — but rejected as sole approach because adversaries reverse-engineer static thresholds quickly (adjust aim-snap speed just under the threshold); ML ensemble adds resilience to threshold-gaming. Preferred as a bootstrap/MVP for a new/small title without enough labeled data to train ML yet |
| Fully real-time-only enforcement (no ban-wave batch path) | Every flag results in immediate action | Rejected — real-time-only forces a precision/recall tradeoff onto a single decision point with no chance to corroborate across sessions, driving up false-positive rate unacceptably at EA's PR-sensitivity level; could be preferred for a low-stakes casual title where wrongful temporary flags carry low cost |
| Centralized single-region anti-cheat (no regional colocation) | All scoring in one central region regardless of player location | Rejected — adds 100-200ms+ cross-region RTT to the real-time path, blowing the 500ms p99 budget for APAC/SA players; would only be viable for a turn-based/non-latency-sensitive title |

## 40. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Real-time + batch layered enforcement | Balances dwell-time reduction with FP control | Two pipelines to build/maintain, more operational surface area |
| Server-authoritative validation first | Deterministic, cheap, hard to game | Only catches physics-detectable cheats, not aim-assist-style statistical cheats |
| Statistical-first triage before ML | Big GPU cost savings | Slight risk of a sophisticated cheat calibrated to sit just under statistical trigger thresholds (mitigated by ML/anomaly layer still running on a random background sample) |
| Fail-open on real-time outage | Never blocks gameplay | Temporary window of zero real-time enforcement during incidents (batch path is the safety net, but with a 24h lag) |
| Weekly ban-wave cadence | High evidence quality, fewer FPs | Cheaters get up to 7 days of dwell time before high-confidence action |
| Distilled real-time model vs. full ensemble online | Much lower real-time GPU cost | Real-time model is strictly less powerful than the offline ensemble — some cheats only caught in batch |
| Regional colocation of scoring stack | Meets latency SLA | Higher infra footprint (4x the stack vs. one central deployment), more complex ban-consistency-across-regions problem |
| CockroachDB/Spanner-style Ban Store | Strong multi-region consistency for security-critical writes | Higher write latency and operational complexity than a simple regional KV store |

## 41. Failure Modes

| Scenario | Impact | Mitigation |
|---|---|---|
| Kafka broker outage in a region | Telemetry ingestion backs up, real-time flagging stalls | Multi-AZ Kafka replication (3x), producer retry with local buffering on game server, fail-open (no gameplay block) |
| Model registry / Triton model load failure post-deploy | Scoring service serves stale or no model | Health-gated rollout (Section 33), auto-rollback to last-known-good warm model version |
| Ban Store cross-region replication lag spike | Banned player can rejoin briefly in another region before propagation completes | Bounded by 60s target + regional session-start ban-check with short-TTL cache (Section 11); accept small residual risk window, monitored/alerted (Section 23) |
| Reviewer queue backlog explosion (e.g. after a big cheat-provider release drives mass flagging) | Ban-wave delayed, cheaters extended dwell time | Auto-scale reviewer prioritization by confidence score, temporarily raise auto-issuance confidence threshold to reduce queue load without lowering the bar unsafely |
| Adversary discovers and exploits a specific statistical threshold | Mass false-negative for a specific cheat variant | Rotating/obfuscated thresholds, ML ensemble as a second independent layer, unsupervised anomaly detector catches the resulting behavioral outlier even without a signature |
| Bad model version passes canary gates due to a blind spot in health checks (e.g. it under-flags a specific cheat category not represented in canary cohort) | Silent false-negative regression | Sentinel labeled set (Section 21) spans known cheat categories, refreshed regularly, checked as part of canary + weekly regression suite |
| Mass false-positive incident (e.g. new patch shifts legitimate aim-assist stats into "cheat" range) | PR incident, wrongful bans, community backlash | Patch-day mandatory statistical recalibration (Section 20 trigger), canary staged rollout, appeal SLA fast-tracked during incident, temporary ban-wave pause capability |

## 42. Scaling Bottlenecks

- **At 10x scale (20M concurrent)**: Kafka partition count and broker fleet need proportional scale-out (straightforward, well-understood lever); the first real bottleneck is the **Flink windowing layer's state size** (rolling per-session windows in memory) — requires re-tuning parallelism and potentially moving to a more aggressively pruned windowing strategy (shorter windows, more downsampling) to keep state manageable.
- **At 100x scale (200M concurrent — hypothetically EA's entire global live-service base at once)**: the **Ban Store's multi-region consensus writes** become the binding constraint — a globally consistent, serializable-write store doesn't scale writes linearly across regions the way stateless services do; would need to shift toward regional "home-writer" sharding (each player's ban writes are authoritative in their home region only, async-replicated elsewhere) to avoid consensus-protocol latency blowing up under write contention.
- **Human review queue** doesn't scale with infra at all — it's headcount-bound; at 10x traffic, the auto-issuance confidence threshold would need to rise (accept only very high-confidence bans without human review) to keep the review queue from becoming the actual bottleneck on ban-wave throughput.
- **GNN batch job** scales worse than the sequence model — graph size grows superlinearly with player interactions (edges grow faster than nodes), so at 10x MAU the collusion-graph job likely needs to shift from single-node multi-GPU to a distributed graph-processing framework (e.g. DistDGL) well before other components hit their limits.

## 43. Latency Bottlenecks

**p50 real-time path budget (~300ms total, from Section 18 diagram)**:
| Hop | p50 | p99 |
|---|---|---|
| Client capture → server validation | 10ms | 20ms |
| Server → ingestion gateway | 25ms | 60ms |
| Kafka produce/consume → Flink window | 60ms | 150ms |
| Flink window write → feature store read | 15ms | 40ms |
| Scoring service (batching + inference) | 60ms | 120ms |
| Decision engine + enforcement RPC | 30ms | 70ms |
| **Total** | **~200-300ms** | **~460-500ms (at SLA edge)** |

- **Dominant cost at p50**: Kafka→Flink windowing hop (~60ms) — inherent latency of the streaming aggregation step, hard to compress further without moving to smaller windows (tradeoff against feature quality/noise).
- **Dominant cost at p99**: dynamic batching wait time in the scoring service under load — the 8-15ms batching window plus queueing when GPU pods are near saturation; this is the first thing to tune (reduce max batch wait, add more GPU replicas) if p99 SLA is at risk.
- **Batch path latency** is dominated by the GNN job runtime (graph construction + training-time inference over the full weekly interaction graph is the single largest chunk of the ~4-6 hour batch window), not by data movement.

## 44. Cost Bottlenecks

- **Real-time GPU fleet** (~20-24 GPUs across regions, always-on) is the largest steady-state cost driver — unlike the batch path, it can't be spot-priced (needs guaranteed availability for the latency-SLA'd path), so it's the highest $/hour line item.
- **Raw telemetry hot storage** (~48TB compressed, 14-day retention) is the second-largest steady cost — driven directly by the ingest volume (Section 6); the main lever is retention-window tuning (shrinking from 14 to 7 days would materially cut this if appeal-window policy allows).
- **Cross-region Ban Store consensus writes** carry meaningful data-transfer + compute cost at EA's traffic scale, though small relative to the above two.
- **Human review labor cost** isn't infra spend but is the real marginal cost driver of raising ban-wave volume — often the actual constraint on "how aggressive can the ban-wave be" long before infra cost becomes limiting.

## 45. Interview Follow-Up Questions

1. How would you detect a cheat that no existing detector has ever seen (a genuinely novel technique), given supervised models only know what's in the labeled data?
2. How do you prevent the anti-cheat ML system from being used by cheat developers as an oracle to reverse-engineer detection boundaries?
3. Console players get aim-assist from the platform itself — how do you avoid flagging legitimate aim-assist as an aimbot?
4. Walk me through how you'd handle a false-positive ban-wave that wrongfully bans 5,000 legitimate players overnight.
5. How would you extend this system to detect boosting/smurfing/collusion rather than just mechanical cheats like aimbots?
6. What's your strategy if a cheat-provider starts selling a tool specifically designed to mimic legitimate statistical distributions (adversarial evasion)?
7. How do you keep the real-time and batch models from drifting apart in what they consider "cheating" over time?
8. Why not just ship a kernel-level anti-cheat driver like some competitors do — what are you trading off?
9. How would this system change for a mobile title with a much less trustworthy client environment (jailbreak/root risk)?
10. How do you measure whether the system is actually working, beyond ban counts (which can be gamed by metric-chasing)?

## 46. Ideal Answers

1. **Novel cheat detection**: rely on the unsupervised anomaly detector (autoencoder/isolation forest over behavioral features) running independently of labeled classifiers, so it flags statistical outliers regardless of whether a label exists yet. Route high-volume unlabeled anomaly clusters to human review for rapid labeling, then fast-track a targeted retrain (Section 20).
2. **Oracle-proofing**: never expose raw scores or thresholds to any client-reachable surface (Section 25), and keep scoring/feature logic server-side only so the client has no code path to introspect. Monitor for anomalous "boundary probing" from single accounts and periodically rotate thresholds so leaked data can't perfectly predict current boundaries.
3. **Aim-assist confound**: train platform-aware models, segmenting features/calibration by input method (controller vs. mouse/KBM), since aim-assist has a distinct, bounded signature versus an aimbot's impossibly fast/precise snaps. Cross-platform drift monitoring (Section 21) watches for score-distribution divergence between cohorts to catch miscalibration early.
4. **Mass false-positive incident response**: pause further auto-issuance from the implicated model version and fast-track appeals for the affected cohort. Root-cause which canary health-check gate should have caught it, and proactively communicate with affected players rather than waiting for appeals to trickle in.
5. **Collusion/boosting extension**: this is what the GNN component (Sections 17/19) is for — model player-interaction graphs (party composition, trade/match history) to detect anomalous subgraph patterns (e.g. a boosting ring of mutually-queueing accounts) that per-session behavioral features can't see, since collusion is inherently a multi-player, cross-session pattern.
6. **Adversarial evasion / distribution mimicry**: this is why a layered ensemble beats a single model — a cheat tuned to mimic legitimate distributions on one feature dimension is unlikely to mimic all of them simultaneously (reaction time, movement, decision-making). Cross-session recidivism and device/behavioral-embedding similarity clustering (Section 17) also catch "same tool, evolving signature" patterns.
7. **Real-time/batch model drift-apart risk**: enforce shared feature-computation code (Section 15) between both paths to prevent train/serve skew from creeping in as separately-evolving codebases. Jointly evaluate both models against the same weekly sentinel labeled set so divergence shows up immediately.
8. **Kernel-level driver tradeoff**: kernel-level anti-cheat gives deeper visibility (can see the cheat process itself, not just its effects) but costs console/cross-platform parity and adds attack surface and player trust friction. The behavioral/telemetry approach trades some detection power for broader platform support — appropriate for a cross-platform live-service catalog rather than one hardcore-competitive PC title.
9. **Mobile/untrusted-client adaptation**: shift more weight onto server-authoritative validation and behavioral detection, since a jailbroken/rooted device can't be trusted to run any client-side agent honestly. Treat all client-reported telemetry as lower-trust corroborating evidence, not ground truth.
10. **Measuring real effectiveness beyond ban counts**: track player-reported-cheater rate correlated against system-flagged rate, estimated cheater-playtime-prevented, appeal-overturn rate (precision proxy), and sentinel-set recall. Ban count alone can rise just by lowering thresholds, so it must be read alongside precision/recall and trust signals.

