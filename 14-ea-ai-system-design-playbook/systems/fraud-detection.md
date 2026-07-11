# Fraud Detection

## 1. Problem Framing & Requirement Gathering

EA operates live-service games (FC 25, Apex Legends, Battlefield, The Sims) with in-game stores, real-money transactions (V-Bucks-style virtual currency, DLC, battle passes), and account-linked identities across platforms (PSN, Xbox Live, Steam, EA App). Two fraud surfaces dominate:

- **Payment fraud**: stolen credit cards used to buy virtual currency, chargebacks, friendly fraud, gift-card cracking, refund abuse.
- **Account Takeover (ATO)**: credential stuffing, session hijacking, SIM-swap enabled resets, then draining cosmetics/currency or relisting on gray markets.

Both must be scored **inline** (payment authorization, login) with strict latency because a false "hang" on checkout or login directly costs revenue and player trust. The system must also survive **adversarial adaptation** — fraud rings A/B test against the model and mutate behavior weekly.

Core framing questions to align with the interviewer/PM before designing:
- Is this a **prevention** system (block before completion) or **detection** system (flag post-hoc for review/chargeback recovery)? (Answer: both — a real-time blocking layer + an async investigation queue.)
- What's the cost asymmetry? A false negative (missed fraud) costs a chargeback fee (~$15-25) + goods lost (~$50 avg cart) + card network penalties if chargeback ratio > 0.9%. A false positive (blocked legit player) costs revenue + LTV + support tickets. Both matter, but chargeback-ratio breaches risk losing the ability to process cards at all (existential).
- Who consumes scores? Payment gateway (block/step-up/allow), login service (block/MFA/allow), Trust & Safety analysts (investigation queue), finance (chargeback recovery).

## 2. Functional Requirements

- FR1: Score every payment transaction in real time and return `allow | challenge | deny` before authorization completes.
- FR2: Score every login/session-resume event for ATO risk and return `allow | step_up_mfa | deny`.
- FR3: Support human-in-the-loop review queue for `challenge` and borderline scores, with analyst verdict feeding back into training data.
- FR4: Ingest chargeback/dispute events (arrives T+7 to T+90 days after transaction) as delayed ground-truth labels.
- FR5: Support per-title, per-region model variants (FIFA points fraud patterns differ from Apex skin-market fraud).
- FR6: Expose a feature-explanation payload (top contributing features/reason codes) for analyst UI and regulatory adverse-action requirements.
- FR7: Support rapid rule hot-patching (< 5 min) independent of model redeploy, for emergency blocking of an active fraud ring.
- FR8: Maintain an audit trail of every decision (score, features, model version, rule version) for chargeback dispute evidence and compliance.
- FR9: Detect and adapt to adversarial drift (new attack patterns) faster than the manual retrain cycle via online monitoring.
- FR10: Support velocity/graph features (shared device, shared card BIN, shared IP/ASN across accounts) for ring detection.

## 3. Non-Functional Requirements (latency, availability, throughput, consistency, cost)

| Dimension | Target | Rationale |
|---|---|---|
| Latency (payment scoring) | p50 < 40ms, p99 < 150ms, hard timeout 250ms | Sits inline in checkout; PSP (payment service provider) round-trip budget is ~1-2s total |
| Latency (login/ATO scoring) | p50 < 30ms, p99 < 100ms | Login UX tolerance is tighter than checkout |
| Availability | 99.95% (≈4.4 hrs/yr downtime) for scoring path | Revenue-critical; below this, fail-open to rules-only |
| Throughput | Sustain 8,000 payment TPS peak (launch day), 25,000 login-scoring TPS peak | EA-scale live-service launch spikes |
| Consistency | Feature store: eventual consistency (< 5s staleness) acceptable for aggregates; exact-once for decision audit log | Fraud features are inherently near-real-time; audit log must be exact for disputes |
| Cost | < $0.0008 per scored transaction fully loaded (compute+storage+network) | Must stay well under fraud-loss-avoided margin |
| Durability | Decision + label data: 99.999999999% (11 nines, object storage class) | Legal/chargeback evidence retention |
| Fail-open policy | On scoring-service outage, fail to conservative rule-set, never fail to "block all" or "allow all" blindly | Balance availability vs risk |

## 4. Clarifying Questions an Interviewer Would Expect You to Ask

1. Is this greenfield or replacing an existing rules engine? (Affects whether we need a shadow-mode rollout period.)
2. What's the current chargeback ratio and card-network threshold we must stay under (e.g., Visa VDMP triggers at 0.9%)?
3. Do we need to support 3DS/step-up authentication as a native action, or just allow/deny?
4. Is cross-title / cross-studio data sharing permitted (privacy/legal), e.g., can a device fingerprint flagged in FIFA inform Apex scoring?
5. What identity signals are available: hashed device ID, IP/ASN, card BIN, platform account ID (PSN/Xbox/Steam), EA account age?
6. What's the label latency — how long after a transaction do we get a confirmed chargeback or confirmed-fraud analyst verdict?
7. Are there regulatory constraints (GDPR/CCPA) on storing device fingerprints or using them for automated decisions without explanation?
8. What's acceptable false-positive rate on legitimate whales (high-LTV players) — do we need a VIP allowlist/override path?
9. Do we need on-device (console/client) pre-screening signals, or is everything server-side?
10. What's the expected adversary sophistication — scripted bots, human click-farms, or organized rings with insider access?

## 5. Assumptions

1. EA has ~30M monthly active payers across titles (subset of ~500M+ registered accounts), with peak concurrent transaction load during content drops (e.g., FC Ultimate Team pack releases).
2. Average transaction value: $12 (micro-transactions dominate); average cart at risk from a fraud ring burst: $500-$2,000 before detection.
3. Chargeback confirmation label arrives 30-90 days post-transaction (card network dispute window); analyst-confirmed-fraud label arrives 1-48 hours via investigation queue.
4. Class imbalance: confirmed fraud ≈ 0.3-0.6% of payment transactions; ATO incidents ≈ 0.05% of login events.
5. Existing rules engine (velocity checks, blocklists) remains as a fast-path pre-filter and fail-open safety net; ML model augments, doesn't fully replace it initially.
6. PII (card PAN, CVV) never touches the fraud model directly — only tokenized/hashed BIN, last-4, and PSP risk signals are available (PCI-DSS scope reduction).
7. Model retraining infra reuses EA's existing internal ML platform (feature store, training orchestration) shared with other ML systems (matchmaking, recommendations).
8. GPU inference is not required for the core tabular/graph model (gradient-boosted trees + shallow graph embeddings); GPUs are used only for periodic embedding/graph-model training.
9. Multi-region: US-East, EU-West, AP-Southeast, matching EA's existing regional data-residency zones.
10. "Challenge" action for payments maps to step-up 3DS or manual review hold; for login maps to MFA/email verification.

## 6. Capacity Estimation (QPS, storage, model size, GPU/CPU counts)

**Traffic:**
- Payment transactions: 30M MAU × ~2.5 purchases/month avg = 75M tx/month → avg ~29 TPS, peak (launch day, 20x avg) ≈ 8,000 TPS sustained for a few hours.
- Login events: 500M accounts, ~80M DAU × 3 sessions/day = 240M logins/day → avg ~2,800 TPS, peak ≈ 25,000 TPS during a global patch-day login storm.
- Total inference calls/day: ~240M (login) + ~2.5M (payment) ≈ 242.5M/day ≈ 2,800 avg QPS blended, peak ~30,000 QPS combined.

**Compute (CPU-bound tree ensemble serving):**
- GBM (LightGBM/XGBoost) ensemble, ~500 trees depth 6, ~2ms CPU inference per call single-threaded.
- At peak 30,000 QPS with 2ms/call and target p99 150ms (allows queuing headroom), need ~30,000 × 0.002 = 60 core-seconds/sec → with 70% utilization target, ~86 vCPU cores concurrently busy; provision ~120 vCPUs (15 x 8-vCPU pods) per region with autoscale headroom to 2x.
- 3 regions active-active → ~360 vCPUs baseline serving fleet, burstable to 720.

**Storage:**
- Raw event log (payment + login + features snapshot): ~2 KB/event × 242.5M/day ≈ 485 GB/day raw → compressed (Parquet, ~5:1) ≈ 97 GB/day → ~35 TB/year in cold/object storage.
- Feature store online (Redis/key-value): ~50M active entities (accounts+devices) × 2 KB feature vector ≈ 100 GB working set per region.
- Feature store offline (columnar, e.g., Iceberg/Delta on S3): rolling 2-year window ≈ 70 TB.
- Model artifacts: GBM model ≈ 15-40 MB; graph-embedding model ≈ 200-500 MB (node2vec/GraphSAGE embeddings for ~50M entities at 64-dim float32 ≈ 12.8 GB raw embedding table, served from a compact ANN/KV store, not the model file itself).
- Audit/decision log (must retain 7 years for chargeback/legal): ~1 KB/decision × 242.5M/day × 365 × 7 ≈ 618 TB over 7 years in cold storage (Glacier-class), heavily compressed and partitioned by date.

**Training:**
- Daily incremental GBM retrain on ~75M payment rows + ~240M login rows sampled/downsampled (imbalanced, undersample majority to ~10:1 ratio) → effective training set ~5-8M rows/day → trains in ~15-30 min on a 32-vCPU, 128GB RAM instance, no GPU needed.
- Weekly graph-embedding retrain (GraphSAGE over device/IP/card shared-entity graph, ~50M nodes, ~300M edges) → needs GPU: 4x A100 (40GB) for ~3-4 hours weekly job.
- Total GPU footprint: 4x A100 spot instances, used ~4 hrs/week (not a standing pool) — cost-optimized as ephemeral training job, not always-on.

## 7. High-Level Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │                  Client / Game Clients               │
                        │   (Checkout UI, Login UI, Console/PC/Mobile Apps)    │
                        └───────────────┬───────────────────┬─────────────────┘
                                        │                   │
                             Payment tx │                   │ Login/session event
                                        ▼                   ▼
                        ┌───────────────────────┐ ┌───────────────────────┐
                        │  Payment Gateway /PSP │ │   Auth/Identity Svc   │
                        │  (checkout orchestr.) │ │  (login orchestrator) │
                        └───────────┬───────────┘ └───────────┬───────────┘
                                    │  sync call (gRPC)        │ sync call (gRPC)
                                    ▼                          ▼
                        ┌─────────────────────────────────────────────────┐
                        │            Fraud Scoring Gateway (API)          │
                        │   - request validation, auth, rate limit       │
                        │   - fans out to Rules Engine + Model Server    │
                        └───────┬───────────────────────┬─────────────────┘
                                │                        │
                 ┌──────────────▼───────┐     ┌──────────▼───────────────┐
                 │   Rules Engine        │     │   Model Serving Layer     │
                 │ (hot-patchable, fast) │     │ (GBM + graph-embed score) │
                 └──────────┬────────────┘     └──────────┬───────────────┘
                            │                              │  feature lookups
                            │                    ┌──────────▼───────────────┐
                            │                    │   Online Feature Store   │
                            │                    │   (Redis / KV, <5s lag)  │
                            │                    └──────────┬───────────────┘
                            │                              │ backfill/sync
                            │                    ┌──────────▼───────────────┐
                            │                    │  Offline Feature Store   │
                            │                    │ (Iceberg/Delta, batch)   │
                            │                    └──────────┬───────────────┘
                            │                              │
                 ┌──────────▼──────────────────────────────▼───────────────┐
                 │            Decision Aggregator (allow/challenge/deny)     │
                 └──────────┬───────────────────────────────┬───────────────┘
                            │ decision + reason codes        │ async
                            ▼                                ▼
                 ┌─────────────────────┐         ┌────────────────────────────┐
                 │ Response to caller  │         │   Event Bus (Kafka)         │
                 │ (PSP / Auth Svc)    │         │  decisions, tx events,      │
                 └─────────────────────┘         │  chargeback/label events    │
                                                  └───────────┬────────────────┘
                                                              │
                        ┌─────────────────────────────────────┼─────────────────────────┐
                        ▼                                     ▼                         ▼
              ┌───────────────────┐               ┌────────────────────┐   ┌──────────────────────┐
              │ Analyst Review     │               │ Feature Pipeline    │   │ Training Pipeline     │
              │ Queue (T&S UI)     │               │ (stream+batch ETL)  │   │ (daily GBM, weekly     │
              │ verdict feedback   │               │                    │   │  graph embeddings)     │
              └───────────────────┘               └────────────────────┘   └──────────┬────────────┘
                                                                                        │
                                                                              ┌──────────▼────────────┐
                                                                              │  Model Registry +      │
                                                                              │  Drift/Monitoring Svc  │
                                                                              └────────────────────────┘
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| Fraud Scoring Gateway | AuthN/authZ, request validation, timeout budget enforcement, fan-out orchestration | gRPC (internal), REST (external partners) | Stateless pod, scales on QPS/CPU |
| Rules Engine | Deterministic velocity/blocklist/allowlist checks, hot-patchable via config push | In-process library + config API | Stateless, co-located with gateway |
| Model Serving Layer | Loads GBM + graph-embedding lookup, returns score + reason codes | gRPC, batched internally | Stateless pod, scales on QPS; model loaded in-memory |
| Online Feature Store | Low-latency key-value lookups for velocity/aggregate features | Redis Cluster protocol / gRPC facade | Sharded by entity key (account_id/device_id) |
| Offline Feature Store | Historical feature computation, training data source, point-in-time joins | Spark/Trino SQL over Iceberg | Partitioned by date + entity hash |
| Feature Pipeline (stream) | Computes real-time velocity aggregates (txn count/5min, device fan-out) | Flink/Kafka Streams job | Scales by Kafka partition count |
| Decision Aggregator | Combines rule verdict + model score into final action, applies business thresholds | In-process w/ gateway or dedicated service | Stateless |
| Event Bus (Kafka) | Durable transport for decisions, tx events, label events | Kafka producer/consumer API | Partitioned by entity_id, scales via partitions/brokers |
| Analyst Review Queue | Human-in-loop UI + verdict capture, priority-ranked by score | Internal web app + REST API | Backed by queue table/service, scales with analyst headcount (not infra-bound) |
| Training Pipeline | Orchestrates daily GBM retrain, weekly graph embedding retrain | Airflow/Kubeflow DAGs | Batch job, scales via cluster size per run |
| Model Registry | Versioned model artifact storage, promotion gates | REST API (MLflow-like) | Low-QPS metadata service |
| Drift/Monitoring Service | Computes PSI/KS stats, tracks online metrics, triggers alerts | Batch + streaming jobs, Prometheus exporters | Scales with feature count monitored |
| Chargeback Ingestion Svc | Consumes PSP/network dispute feeds, joins to original decision, emits label events | Batch/webhook consumer | Low-QPS, scheduled |

## 9. API Design

**Base path:** `/v2/fraud/score` (versioned; `v1` deprecated, sunset 6 months post-v2 GA)

### POST /v2/fraud/score/payment
```json
Request:
{
  "request_id": "uuid",
  "account_id": "hashed-acct-id",
  "title_id": "fc25",
  "region": "us-east",
  "amount_usd_cents": 1999,
  "currency": "USD",
  "payment_method": {"type": "card", "bin": "411111", "last4": "1111", "token": "psp-token-xyz"},
  "device_fingerprint": "hashed-device-id",
  "ip_address": "hashed-or-truncated-ip",
  "session_context": {"session_age_s": 340, "platform": "psn"},
  "timestamp": "2026-07-08T10:15:00Z"
}

Response:
{
  "request_id": "uuid",
  "decision": "allow | challenge | deny",
  "risk_score": 0.83,
  "reason_codes": ["VELOCITY_HIGH_TXN_COUNT", "NEW_DEVICE_HIGH_VALUE"],
  "model_version": "gbm-2026-07-07-v14",
  "rules_version": "ruleset-2026-07-08-a",
  "decision_id": "uuid-for-audit-trail",
  "latency_ms": 42
}
```

### POST /v2/fraud/score/login
```json
Request: { "request_id", "account_id", "device_fingerprint", "ip_address", "platform", "auth_method", "geo_velocity_flag" }
Response: { "decision": "allow | step_up_mfa | deny", "risk_score": 0.12, "reason_codes": [...], "model_version", "decision_id" }
```

### POST /v2/fraud/feedback/verdict
- Analyst/PSP posts confirmed verdict (`confirmed_fraud | confirmed_legit | chargeback_won | chargeback_lost`) referencing `decision_id`. Feeds label pipeline.

### GET /v2/fraud/decision/{decision_id}
- Retrieves full audit record (features, scores, rule hits) for dispute evidence / analyst review. AuthZ-gated to T&S role.

| Endpoint | Method | Latency SLA | Auth |
|---|---|---|---|
| /v2/fraud/score/payment | POST | p99 150ms | mTLS service-to-service |
| /v2/fraud/score/login | POST | p99 100ms | mTLS service-to-service |
| /v2/fraud/feedback/verdict | POST | best-effort, async | OAuth2 client-credentials (internal tool) |
| /v2/fraud/decision/{id} | GET | p99 500ms | OAuth2 + RBAC (T&S role) |

Versioning: URI-based (`/v1`, `/v2`); backward-compatible field additions allowed within a version; breaking changes (removed/renamed fields, changed decision semantics) require a new version with a 2-quarter dual-run deprecation window.

## 10. Database Design

| Store | Type | Data | Partition/Shard Key | Why |
|---|---|---|---|---|
| Online Feature Store | Redis Cluster (KV) | Real-time velocity aggregates, recent device/IP/account counters | `entity_id` (account or device hash) | Sub-ms reads, TTL-based eviction matches feature freshness needs |
| Offline Feature Store | Iceberg/Delta on S3 (columnar) | Historical features for training, point-in-time joins | Partitioned by `event_date`, clustered by `entity_id_hash` | Columnar scan efficiency for training jobs; time-travel for point-in-time correctness |
| Decision Audit Log | Append-only columnar (e.g., ClickHouse or Iceberg) | Every scoring decision + features + reason codes | Partitioned by `event_date`, indexed by `decision_id`, `account_id` | High write throughput, analytical queries for T&S investigations, long retention |
| Transaction/Label Store | OLTP (Postgres/Aurora) | Transaction records, chargeback status, analyst verdicts | Sharded by `account_id` hash | Strong consistency needed for verdict-to-decision linkage, transactional updates |
| Graph Store (entity relationships) | Graph DB (Neptune/JanusGraph) or precomputed embedding table in KV | Shared-device/shared-card/shared-IP relationships between accounts | Partitioned by node id (account/device) | Ring detection needs multi-hop traversal; embeddings precomputed offline, served from KV at inference time to avoid live graph traversal latency |
| Model Registry | Metadata DB (Postgres) + object storage (S3) for artifacts | Model version, metrics, promotion state | N/A (low volume) | Simple relational metadata suffices |

Rationale for columnar audit log: analysts and chargeback-dispute workflows run wide analytical scans ("show me all decisions from this device in last 30 days") — row-store OLTP would be inefficient at 618TB/7yr scale; columnar compression + partition pruning keeps cost and query latency manageable.

## 11. Caching

| Cache | Content | Strategy | Invalidation |
|---|---|---|---|
| Feature cache (Redis, online store) | Velocity aggregates, recent counters | Write-through from streaming feature pipeline | TTL (5-30 min depending on feature) + explicit overwrite on new event |
| Model artifact cache (in-process, serving pods) | Loaded GBM model + graph embedding shards | Cache-aside, loaded at pod startup and on version-change signal | Pub/sub "new model promoted" event triggers hot-reload without pod restart |
| Rules config cache | Active ruleset | Cache-aside with short TTL (30s) poll or push via config-service watch | Push-based invalidation for emergency hot-patch (< 5 min SLA from FR7) |
| Entity risk-tier cache | Precomputed "known-bad" / "VIP allowlist" flags | Write-through on analyst verdict / allowlist update | Immediate invalidation on write (low volume, high importance) |
| Decision-dedup cache | Recent `request_id` → decision, for idempotent retries from PSP | Cache-aside, short TTL (5 min) | TTL expiry; also protects against double-charging on client retries |

Cache-aside dominates for read-heavy low-staleness-tolerant data (features); write-through used where correctness on next-read matters more than write latency (risk-tier flags, rules).

## 12. Queues & Async Processing

| Queue | Payload | Delivery Guarantee | Dead-Letter Handling |
|---|---|---|---|
| `payment-decisions` (Kafka) | Every scored payment decision + features snapshot | At-least-once | Consumer idempotency via `decision_id`; poison messages after 5 retries → DLQ topic, alert T&S data eng |
| `login-decisions` (Kafka) | Every scored login decision | At-least-once | Same pattern, lower criticality → DLQ retention 7 days |
| `chargeback-labels` (Kafka, from PSP webhook ingestion) | Confirmed chargeback/dispute outcome | At-least-once, exactly-once *effect* via upsert keyed on `decision_id` | Malformed PSP payloads → DLQ + manual reconciliation queue |
| `analyst-verdicts` | Human-reviewed fraud/legit verdicts | At-least-once | Retry with backoff; DLQ after 3 attempts, paged to on-call data eng if DLQ depth > 100 |
| `feature-recompute` | Trigger for backfill when late-arriving events change historical aggregates | At-least-once | Idempotent recompute (deterministic given inputs) — safe to reprocess |
| `model-promotion-events` | New model version promoted, triggers hot-reload on serving fleet | At-least-once, pub/sub fanout | Serving pods poll registry as fallback if event missed |

Exactly-once is **not** pursued at the transport layer (costly, adds latency); instead all consumers are designed idempotent (upsert on natural key: `decision_id`, `account_id+event_date`), achieving exactly-once *effect* on top of at-least-once delivery — standard Kafka pattern.

## 13. Streaming & Event-Driven Architecture

**Topics:**
- `raw.payment.events` (partition key: `account_id`, 64 partitions) — every checkout attempt.
- `raw.login.events` (partition key: `account_id`, 128 partitions) — every login attempt.
- `scored.payment.decisions` / `scored.login.decisions` — post-scoring, includes features+scores for audit and downstream consumers.
- `label.chargeback` / `label.analyst_verdict` — delayed ground truth.
- `feature.velocity.updates` — intermediate stream for Flink aggregation jobs feeding the online feature store.
- `model.lifecycle` — promotion/rollback events.

**Consumer groups:**
- `feature-pipeline-cg`: Flink job computing rolling windows (1m/5m/1h/24h counts of txns per device/card/IP), writes to Redis.
- `audit-log-writer-cg`: writes `scored.*.decisions` to columnar audit store.
- `label-join-cg`: joins `label.*` streams back to original `scored.payment.decisions` by `decision_id`, materializes training labels.
- `graph-builder-cg`: consumes payment+login events to incrementally update the shared-entity graph edges (device↔account, card↔account).
- `drift-monitor-cg`: samples scored decisions to compute real-time feature distribution stats (PSI) vs. training baseline.

**Event schema (Avro, versioned in schema registry):**
```json
{
  "type": "record", "name": "PaymentDecisionEvent", "namespace": "ea.fraud.v2",
  "fields": [
    {"name": "decision_id", "type": "string"},
    {"name": "account_id", "type": "string"},
    {"name": "risk_score", "type": "float"},
    {"name": "decision", "type": {"type": "enum", "name": "Decision", "symbols": ["ALLOW","CHALLENGE","DENY"]}},
    {"name": "model_version", "type": "string"},
    {"name": "features", "type": {"type": "map", "values": "float"}},
    {"name": "event_timestamp", "type": "long", "logicalType": "timestamp-millis"}
  ]
}
```
Schema registry enforces backward compatibility (new optional fields only) so consumers don't break on producer upgrades.

## 14. Model Serving

- **Framework**: GBM (LightGBM) served via a lightweight custom gRPC server (or Triton with FIL backend for tree ensembles) — chosen over heavyweight deep-learning servers because the primary model is tree-based, CPU-friendly, and needs sub-5ms inference.
- **Batching**: Micro-batching disabled for the synchronous payment/login path (single-request low-latency priority); batching *is* used for the offline graph-embedding scoring refresh (batch of 10K entities at a time).
- **Multi-model**: Serving layer hosts (a) per-title GBM variants (FC25, Apex, Sims), (b) a shared cross-title "ring detection" graph-embedding lookup, (c) a fallback global GBM for new/low-data titles — model selection by `title_id` in request, with shared feature-fetch layer.
- **Hardware**: CPU-only fleet (e.g., 8-vCPU pods, AVX2-optimized tree inference) for the online path — no GPU needed given tree-ensemble choice; GPU pool (A100 spot) reserved solely for periodic graph-embedding training, not serving (embeddings are precomputed and served from KV store as plain vector lookups).
- **Canary-aware serving**: pods can hold both `current` and `challenger` model in memory, scoring shadow-traffic against both and logging divergence without affecting the live decision.
- **Cold start**: model artifacts pulled from registry at pod boot (<2s for 40MB GBM); readiness probe blocks traffic until model loaded.

## 15. Feature Store

**Online store** (Redis, <5s staleness): velocity counters (txn count last 5m/1h/24h per device/account/card BIN), recent-decision history, device/IP reputation scores, session-level signals.

**Offline store** (Iceberg, batch): full historical feature snapshots used for training, including slower-moving features (account age, lifetime spend, historical chargeback count, KYC-tier).

**Point-in-time correctness**: Training joins use **as-of joins** against the offline store keyed on `event_timestamp` — for a training row at time T, only features with `feature_computed_at <= T` are joined, preventing label leakage (e.g., a chargeback confirmed 45 days later must never leak into features available to the model *at scoring time*). Implemented via time-travel queries on Iceberg snapshots + a strict feature-freshness contract enforced by the feature pipeline (every feature write is timestamped with its true availability time, not ingestion time).

**Online/offline parity**: same feature-computation logic (shared Flink/Spark UDF library) runs in both the streaming path (writes to Redis) and batch backfill path (writes to Iceberg), avoiding training/serving skew — a common failure mode in fraud systems where online aggregates drift from offline-recomputed ones.

## 16. Vector Database

**Partially applicable.** Graph/entity embeddings (from GraphSAGE over the device-account-card relationship graph) are 64-dim float32 vectors used for similarity-based ring detection ("find accounts whose embedding is close to known fraud rings").

- **Indexing strategy**: HNSW (Hierarchical Navigable Small World) index over the ~50M entity embeddings, rebuilt weekly alongside embedding retraining.
- **ANN choice**: HNSW over IVF-PQ because recall/latency tradeoff favors HNSW at this scale (50M vectors, 64-dim is small enough that HNSW's memory overhead is acceptable, ~15GB index), and query pattern is "top-k nearest known-fraud-cluster centroids," not billions-scale search — doesn't need product quantization's compression.
- Served via a managed vector index (e.g., pgvector on a dedicated instance, or a lightweight FAISS-HNSW service) queried async during the investigation/graph-analysis path, **not** in the synchronous scoring hot path (precomputed cluster-membership flag is cached in the online feature store instead, so the sync path avoids ANN query latency entirely).

## 17. Embedding Pipelines

Applicable — used for entity-relationship (graph) embeddings, not text/content embeddings.

- **Pipeline**: weekly batch job — (1) build graph snapshot (accounts, devices, cards, IPs as nodes; shared-usage as edges) from the last 90 days of events, (2) train GraphSAGE (inductive, so new nodes can get embeddings without full retrain) on 4x A100 GPUs, ~3-4 hrs, (3) compute embeddings for all active entities, (4) write to HNSW index + push a compact "cluster/risk-tier" derived feature to the online feature store for sync-path use, (5) validate embedding stability (cosine similarity drift vs. last week) before promotion.
- **Why GraphSAGE over transductive methods (e.g., plain node2vec)**: inductive capability lets new accounts/devices get a reasonable embedding immediately via neighborhood aggregation, without waiting for the next full retrain — important because fraud rings often use brand-new accounts.

## 18. Inference Pipelines (Request Lifecycle End-to-End)

```
 t=0ms   Client submits checkout → Payment Gateway
 t=2ms   Gateway calls Fraud Scoring Gateway (gRPC, mTLS)
 t=4ms   Scoring Gateway: authZ check, request validated, request_id dedup check (cache)
 t=6ms   Parallel fan-out:
           (a) Rules Engine eval (in-memory, ~1ms)
           (b) Feature fetch from Online Feature Store (Redis, ~2-4ms round trip)
 t=12ms  Feature vector assembled (online features + static account attrs cached in-process)
 t=14ms  Model Serving Layer: GBM inference (~2-3ms) + KV lookup of precomputed
           graph-cluster-risk feature (~1ms)
 t=18ms  Decision Aggregator: combine rule verdict + model score → business threshold
           logic → final decision (allow/challenge/deny) + reason codes
 t=20ms  Response returned to Payment Gateway
 t=22ms  [async, non-blocking] Decision event published to Kafka
           (scored.payment.decisions)
 t=25ms  Payment Gateway proceeds with PSP authorization call (if "allow") or
           triggers 3DS step-up (if "challenge")
 ...     [async, minutes-to-days later] Audit log writer persists decision;
           feature pipeline updates velocity counters; if later a chargeback
           arrives (T+30-90 days), label-join consumer attaches ground truth
           to this decision_id for the next training cycle.
```

p99 budget of 150ms leaves ~130ms headroom above the ~20ms happy-path above for network jitter, Redis tail latency, and GC pauses — deliberately conservative given payment-authorization is revenue-critical.

## 19. Training Pipelines

- **Data prep**: pull labeled examples from offline feature store via point-in-time join (Section 15); labels = confirmed chargeback (positive), confirmed-legit analyst verdict (negative), and "aged-out clean" transactions (no chargeback after 90-day window, weak negative).
- **Imbalance handling**: fraud ≈ 0.3-0.6% of payment rows. Strategy: (a) undersample majority class to ~10:1 negative:positive ratio for GBM training (preserves signal, keeps training set tractable at ~5-8M rows/day), (b) use `scale_pos_weight` in LightGBM to correct for residual imbalance, (c) evaluate on **untouched, naturally-imbalanced** validation set using PR-AUC (not ROC-AUC, which is misleadingly optimistic under heavy imbalance) and precision@fixed-recall (e.g., precision at 80% recall) as the primary offline metric.
- **Orchestration**: Airflow DAG — `extract_features` → `point_in_time_join` → `undersample` → `train_gbm` → `offline_eval` → `bias/fairness_check` → `register_candidate_model` → `shadow_traffic_eval` (7 days) → `promotion_gate` (human or automated approval) → `deploy`.
- **Distributed training**: GBM training itself is single-node (LightGBM handles 5-8M rows on 32 vCPU/128GB in ~15-30 min, no need for distributed tree training at this scale). The weekly GraphSAGE embedding job *is* distributed across 4x A100 GPUs (data-parallel over graph partitions/mini-batches via neighbor sampling).
- **Reproducibility**: every training run pinned to an Iceberg snapshot ID (exact data version) + code commit hash + hyperparameter config, logged to the Model Registry for full lineage.

## 20. Retraining Strategy

| Trigger type | Condition | Action |
|---|---|---|
| Scheduled (cadence) | Daily (GBM), weekly (graph embeddings) | Standard retrain DAG run |
| Drift-triggered | PSI > 0.2 on any top-20 feature, or score-distribution KS-stat > threshold vs. last-week baseline | Kick off out-of-cycle retrain within 4 hours |
| Label-volume triggered | New confirmed-fraud labels exceed 20% of last training set's positive count since last retrain | Out-of-cycle retrain |
| Adversarial-event triggered | T&S reports active large-scale fraud ring (manual signal) | Emergency retrain + simultaneous rules hot-patch (rules act as immediate stopgap while retrain runs) |
| Performance-degradation triggered | Production precision@recall drops > 15% relative vs. 7-day rolling baseline (measured once labels catch up) | Investigate + retrain; may also trigger rollback to last-known-good model |

Retraining is deliberately **fast-cadence (daily)** relative to typical ML systems because fraud is adversarial — attackers adapt within days, so the model must too. This is the core differentiator vs. e.g. a recommendation system's weekly/monthly cadence.

## 21. Drift Detection

| Drift type | Metric | Threshold | Response |
|---|---|---|---|
| Data drift (feature distribution) | PSI (Population Stability Index) per feature, computed hourly on scoring traffic vs. training baseline | PSI > 0.1 = watch, > 0.2 = alert + trigger retrain | Auto-retrain pipeline kicked off; T&S notified |
| Concept drift (label-feature relationship changing) | Rolling precision@80%-recall on labels as they arrive (lagged 30-90 days), compared week-over-week | Relative drop > 15% | Investigate feature importance shifts, consider emergency retrain |
| Score distribution drift | KS-statistic on daily score histogram vs. 7-day rolling baseline | KS > 0.15 | Flag for review; often precedes a new attack pattern |
| Prediction-serving skew | Online-computed feature value vs. offline-recomputed value for same event (sampled 1%) | Mean absolute relative diff > 5% | Page data eng — usually indicates a pipeline bug, not real drift |
| Adversarial pattern drift | Sudden spike in a specific reason-code frequency (e.g., `NEW_DEVICE_HIGH_VALUE` triples in an hour) | > 3x hourly baseline | Auto-alert T&S; may warrant emergency rule hot-patch |

Given the label lag (30-90 days for chargebacks), concept-drift detection **cannot rely solely on lagged labels** — the system also uses proxy signals (score drift, reason-code frequency shifts, analyst-verdict velocity from the faster 1-48hr review-queue path) as early-warning indicators before confirmed chargeback data catches up.

## 22. Monitoring

**Infra:** p50/p95/p99 latency per endpoint, QPS, error rate, pod CPU/memory, Redis hit rate/latency, Kafka consumer lag, GPU utilization during training jobs.

**Model quality:** PR-AUC, precision@recall targets, calibration (predicted vs. observed fraud rate by score bucket), feature importance stability, PSI per feature, prediction-serving skew.

**Business metrics:** chargeback ratio (must stay < card-network threshold, e.g., 0.9%), fraud-loss-avoided ($ value of blocked transactions later confirmed fraudulent), false-positive rate on high-LTV players, analyst queue depth/SLA, step-up/MFA completion rate (proxy for friction cost), $ fraud loss per 10K transactions.

**Dashboards:** real-time ops dashboard (latency/QPS/errors), daily fraud-ops dashboard (chargeback ratio trend, top reason codes, blocked-$ vs confirmed-fraud-$), weekly model-health dashboard (drift metrics, precision/recall trend, shadow-model comparison).

## 23. Alerting

| Alert | Condition | Severity | Routing |
|---|---|---|---|
| Scoring latency p99 > 200ms for 5 min | Latency SLO breach | P1 | Page on-call SRE |
| Scoring service error rate > 1% for 3 min | Availability risk | P1 | Page on-call SRE, auto-trigger fail-open to rules-only |
| Chargeback ratio approaching 0.8% (network threshold 0.9%) | Business-critical breach risk | P1 | Page fraud-ops lead + finance |
| Feature PSI > 0.2 on top-10 feature | Data drift | P2 | Notify ML on-call, auto-kick retrain |
| Kafka consumer lag > 10 min on `payment-decisions` | Audit/label pipeline delay | P2 | Notify data eng on-call |
| DLQ depth > 100 messages | Processing failures accumulating | P2 | Notify data eng on-call |
| Reason-code frequency spike > 3x baseline | Possible active fraud ring | P2 (P1 if $ exposure > $50K/hr estimated) | Notify T&S + fraud-ops |
| Model precision@recall drop > 15% (once labels available) | Concept drift / model decay | P2 | Notify ML team, review retrain candidate |
| Shadow-model divergence from production > threshold | Candidate model instability | P3 | Notify ML on-call, block promotion |

On-call routing: SRE on-call owns infra/latency/availability pages; ML on-call owns drift/model-quality pages; fraud-ops/T&S owns business-metric and active-ring pages — routed via PagerDuty services mapped to each domain, with a shared "fraud-critical" escalation policy for P1s that page all three in parallel.

## 24. Logging

- **Structured logging**: every scoring request emits a structured JSON log line (`decision_id`, `latency_ms`, `decision`, `model_version`, `rules_version`, truncated feature summary) — never free-text — to enable log-based querying and correlation with traces.
- **PII handling**: raw card PAN/CVV never logged (PCI scope — only tokenized references). Device fingerprints, IPs, and account IDs are hashed/salted before logging; the mapping from hash→raw identity lives only in a separate, tightly access-controlled identity-resolution service used solely by authorized T&S investigators with audit-logged access.
- **Retention**: hot logs (searchable, e.g., in Elasticsearch/Datadog) retained 30 days; decision audit trail (columnar store, Section 10) retained 7 years for chargeback/legal evidence, with PII fields access-restricted via row-level security and periodic access review.
- **Right-to-be-forgotten**: account-deletion requests trigger a scrubbing job that removes/pseudonymizes identity-linkable fields from hot logs and analyst-facing views while preserving fraud-signal aggregates (which are already hashed and don't require the raw identity) needed for graph-based ring detection.

## 25. Security

**Threat model specific to this system:**
- **Model extraction/probing**: adversaries send crafted transactions to reverse-engineer decision boundaries → mitigate via rate limiting per account/device on scoring-adjacent endpoints, monitoring for probing patterns (many small transactions varying one feature at a time), and not exposing raw model scores/reason codes to end users (only to internal analysts).
- **Feature poisoning**: fraud rings generate synthetic "legitimate-looking" history (aged accounts, slow-burn behavior) to game future model versions → mitigate via graph-based ring detection (shared device/card/IP) that doesn't rely solely on account-level history, and by weighting recent labels more heavily in retraining.
- **Insider threat**: T&S analyst or engineer with access to allowlist/rules config could whitelist a fraud ring → mitigate via four-eyes approval on allowlist changes, full audit trail of config changes, anomaly detection on analyst verdict patterns.
- **Data exfiltration of the audit log**: 7 years of transaction+identity-linkable data is a high-value target → encryption at rest (KMS-managed keys), field-level encryption for identity-resolution mappings, strict IAM least-privilege, VPC-isolated access.
- **Replay/tampering of decision events**: Kafka topics carrying decisions must be tamper-evident for dispute-evidence integrity → messages signed/HMAC'd at producer, verified at audit-log-writer.

**Encryption**: TLS 1.3 in transit everywhere (mTLS service-to-service); AES-256 at rest for all stores; card tokens never decrypted outside the PCI-scoped PSP boundary.

## 26. Authentication

- **Service-to-service**: mTLS with short-lived certs issued by internal CA (e.g., SPIFFE/SPIRE identities), rotated every 24h; every internal gRPC call authenticated via workload identity, authorized via a service mesh policy (e.g., Istio AuthorizationPolicy) restricting which services may call `/v2/fraud/score/*`.
- **End-user auth** (for the login-scoring path itself): standard EA account auth (OAuth2/OIDC against EA's identity provider), with the fraud-scoring call happening *after* password verification but *before* session issuance, so a "deny"/"step_up_mfa" decision can still block session creation.
- **Analyst/internal-tool auth**: OAuth2 + RBAC (T&S role required for `/v2/fraud/decision/{id}` and verdict submission), SSO via corporate IdP, all access logged.
- **Partner/PSP auth**: mutual API-key + HMAC-signed webhook payloads for chargeback ingestion, with signature verification and replay-window checks (reject webhooks older than 5 min or with reused nonces).

## 27. Rate Limiting

- **Algorithm**: token bucket per (account_id, endpoint) and per (device_fingerprint, endpoint) — token bucket chosen over fixed-window for smoother burst tolerance (a legitimate player retrying a failed payment shouldn't be instantly blocked, but a scripted prober should be).
- **Limits**: 
  - Payment scoring: 10 requests/min per account_id (burst 20), 30 requests/min per device_fingerprint (catches multi-account abuse from one device).
  - Login scoring: 20 requests/min per account_id (burst 40, since password typos happen), 100 requests/min per IP (catches credential-stuffing sprays).
- **Partner-tier limits**: PSP webhook ingestion capped at a negotiated per-partner QPS (e.g., 500 QPS) with a sliding-window counter, separate from per-user limits.
- **Breach response**: exceeding the limit doesn't hard-block silently — it escalates to `challenge`/step-up rather than outright `deny`, avoiding false-positive lockouts of legitimate players on flaky networks, while a *sustained* breach (10x over limit for 5+ min) does trigger a hard block + alert.

## 28. Autoscaling

- **Scoring Gateway + Model Serving pods**: HPA on custom metric `requests_per_second_per_pod` (target: 70% of max sustained throughput per pod) plus a secondary CPU-utilization target (60%) as a fallback signal — scales from a baseline of 15 pods/region to burst 40 pods/region within ~90s (HPA + pre-warmed pod images to avoid cold-start latency spikes).
- **Feature-pipeline Flink jobs**: scaled via Flink's reactive mode tied to Kafka partition lag (KEDA `kafka` scaler) — adds task-manager parallelism when consumer lag exceeds 30s.
- **Redis (online feature store)**: scaled via cluster resharding triggered by memory-utilization > 75% (semi-manual/ops-gated, not fully automatic, given resharding risk).
- **Training jobs**: not autoscaled in the HPA sense — provisioned on-demand (Kubernetes Job / batch cluster) sized per DAG (32 vCPU for GBM, 4x A100 spot for graph embeddings), torn down after completion.
- **VPA**: used in recommendation-only mode for the Scoring Gateway pods to right-size memory/CPU requests based on historical usage, feeding into HPA capacity planning rather than live-resizing (avoids restart-induced latency blips on a latency-critical path).

## 29. Cost Optimization

- **Spot instances**: graph-embedding GPU training (4x A100, ~4hrs/week) run entirely on spot capacity with checkpointing every 30 min — training-only workload tolerates preemption, saves ~60-70% vs. on-demand.
- **Model choice**: GBM over deep neural net for the core scorer avoids GPU serving cost entirely (CPU inference at ~2ms vs. needing a GPU fleet) — the single biggest cost lever given serving QPS scale (30K peak).
- **Caching**: feature cache and rules-config cache reduce redundant computation/Redis-miss fallback-to-offline-store calls, cutting p99 tail-latency-driven overprovisioning.
- **Batching**: offline embedding computation batched (10K entities/batch) to maximize GPU utilization during the weekly training window rather than running many small jobs.
- **Storage tiering**: audit log — hot (queryable) for 30-90 days, then transitioned to cold/archive storage class (e.g., S3 Glacier Instant Retrieval) for the remaining 7-year retention, cutting storage cost ~80% for the long tail.
- **Downsampling for training**: undersampling the majority class (Section 19) cuts daily training compute ~10x vs. training on full 75M-row payment volume.
- **Right-sizing via VPA recommendations**: avoids chronic overprovisioning of the always-on serving fleet, which is the largest standing cost (360-720 vCPUs across regions).
- **Reserved/committed-use for baseline serving fleet**: baseline 360 vCPUs (non-bursty portion) committed for 1-3yr terms at discount, burst capacity (up to 720) on-demand/spot-eligible where latency SLA allows brief spin-up.

## 30. Operational Concerns (Deployment, Reliability, Infra)

At SDE2 scope, treat this as a checklist rather than a design exercise: **backups** (automated snapshots of the model registry, feature store, and any stateful service, with a tested restore path), **rollback** (every deploy must be revertible to the last-known-good version — the model registry and CI/CD pipeline should make this a one-command operation), **canary/blue-green rollout** (shift a small percentage of traffic first, watch error rate and key business/model metrics, then ramp), and **basic observability** (dashboards + alerts on latency, error rate, and the top 2-3 model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level infra-architecture concerns — worth knowing they exist, not worth rehearsing the manifests.

## 38. Why This Architecture

- **Tree ensemble (GBM) over deep net for core scoring**: tabular fraud features (velocity counts, categorical BINs, account attrs) are exactly what GBMs excel at; avoids GPU-serving cost/latency entirely at 30K QPS peak, meeting the sub-150ms p99 budget on commodity CPU.
- **Rules engine kept alongside ML, not replaced**: provides a hot-patchable, sub-5-min-reaction emergency lever and a fail-open safety net — pure ML systems can't react to a live fraud ring as fast as a config push.
- **Separate fast online store + slower offline store**: matches the actual latency/consistency needs of each — velocity features must be near-real-time; deep historical features tolerate batch freshness, and separating them keeps the hot path lean.
- **Async label pipeline decoupled from sync scoring**: chargeback labels arrive weeks later; coupling training-data collection tightly to the sync path would be both unnecessary and risky (no need to block scoring on label availability).
- **Active-active multi-region with regional feature stores**: matches EA's globally distributed player base and keeps the scoring hot path region-local to hit latency SLAs, while global graph/offline pipelines still catch cross-region ring patterns asynchronously.
- **Daily retrain cadence**: explicitly sized to out-pace adversarial adaptation speed, a defining characteristic of fraud systems vs. typical ML systems with weekly/monthly cycles.

## 39. Alternative Architectures

| Alternative | Description | Why Rejected / When Preferred |
|---|---|---|
| Deep neural net (e.g., wide-and-deep or transformer-over-sequence) as primary scorer | Model sequences of user events directly | Rejected as *primary*: GPU serving cost/latency not justified given GBM already hits target precision on tabular+velocity features at this stage; **would be preferred** if EA had rich behavioral clickstream sequences proven to add significant lift and could afford GPU inference fleet, or if scoring latency budget were laxer (e.g., async-only post-hoc scoring) |
| Pure rules-engine (no ML) | Deterministic velocity/blocklist rules only | Rejected: doesn't adapt to novel patterns, ceiling on precision/recall is low, rings quickly learn static rule boundaries; **would be preferred** only for a brand-new title with zero historical fraud-label data (bootstrap phase) |
| Fully synchronous graph traversal at inference time (live graph DB query instead of precomputed embeddings) | Query the graph DB live per-request for ring membership | Rejected: live multi-hop graph traversal at 30K QPS with <150ms budget is infeasible with current graph DB tech at this node/edge scale; **would be preferred** if request volume were much lower (e.g., only for `challenge`-tier deep investigation, which is in fact how it's used today — async, not sync) |
| Fully centralized single-region deployment | One region serves all global traffic | Rejected: violates latency SLA for non-US traffic and creates a single point of failure at EA's global scale; **would be preferred** only for a small-scale regional-only title with no global player base |
| Exactly-once streaming (transactional Kafka + strict dedup at every hop) | Guarantee exactly-once delivery end-to-end | Rejected: adds meaningful latency/complexity for marginal benefit given idempotent-consumer pattern already achieves exactly-once *effect*; **would be preferred** if downstream consumers couldn't be made idempotent (e.g., non-idempotent financial ledger postings, which is why the actual money-movement ledger, outside this system's scope, does use stronger guarantees) |

## 40. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| GBM over deep net | Low latency, no GPU serving cost, interpretable reason codes | Lower ceiling on capturing complex sequential/behavioral patterns |
| Undersampling for training | Tractable training time, better minority-class signal | Requires careful calibration correction; naive undersampling can bias probability estimates if not corrected |
| Regional (not global) online feature store | Meets latency SLA, resilient to single-region outage | Slight blind spot for cross-region ring activity in the sync path (mitigated async via graph pipeline) |
| Rules engine kept alongside ML | Fast emergency response, fail-open safety net | Two systems to maintain, potential for rule/model disagreement requiring an arbitration policy |
| Daily retrain cadence | Keeps pace with adversarial adaptation | Higher training-infra cost and operational overhead vs. weekly/monthly |
| Async label pipeline (30-90 day lag) | Doesn't block scoring, matches real-world chargeback timelines | Slow ground-truth feedback means concept drift can go undetected for weeks without proxy-metric early warning |
| Precomputed graph embeddings vs. live traversal | Meets sync-path latency budget | Embeddings staleness (up to 1 week) for brand-new fraud rings until inductive GraphSAGE / next retrain catches them |
| Active-active multi-region | Best latency, resilience | Higher operational complexity, eventual-consistency edge cases for cross-region entities |

## 41. Failure Modes

| Scenario | Detection | Mitigation |
|---|---|---|
| Online Feature Store (Redis) regional outage | Elevated feature-fetch error rate, latency spike | Serving layer falls back to a reduced feature set (static/cached account attrs only) + more conservative rules-engine weighting; alert fires, autoscaled failover to healthy replica shard |
| Model Serving pods crash-loop after bad deploy | Readiness probe failures, error-rate alert | Auto-rollback via deployment controller (failed rollout triggers automatic revert), blue/green fallback to last-good environment |
| Kafka broker/partition unavailability | Producer errors, consumer lag spike | Producers configured with retries + `acks=all`; if sustained, decisions still return synchronously to caller (Kafka publish is async/non-blocking for the response), but audit-log/feature-pipeline backlog builds — DLQ + backfill once recovered |
| Chargeback-label ingestion pipeline breaks silently | Data-quality monitor detects label-volume anomaly (near-zero new labels for 24h+) | Alert data eng; training pipeline has a guard that refuses to promote a new model if label-volume for the period is anomalously low (protects against training on stale/incomplete labels) |
| Sudden coordinated fraud ring (10,000 accounts, new pattern, no historical signal) | Reason-code frequency spike alert, chargeback-ratio trending alert | Emergency rules hot-patch (block by shared device/IP signature) within minutes while emergency retrain runs in parallel |
| Model serving a stale/incorrect version after partial rollout | Shadow-traffic divergence alert, canary health-gate failure | Canary gating catches before 100% rollout; automated rollback trigger reverts in-memory model swap |
| Cross-region network partition | Elevated cross-region replication lag metrics | Each region continues operating independently (active-active design tolerates partition); reconciliation of offline/graph data once partition heals |
| Analyst review queue backlog spikes (e.g., during a launch-day fraud surge) | Queue-depth SLA alert | Auto-tighten model `challenge` threshold temporarily to reduce queue inflow, page additional analyst on-call, prioritize queue by $ exposure |

## 42. Scaling Bottlenecks

**At 10x traffic (300K peak QPS):**
- Online Feature Store (Redis) becomes the first bottleneck — single-shard hot-key contention on frequently-accessed entities (e.g., a popular shared device/proxy) even with cluster sharding; requires read-replica fan-out and possibly moving hottest counters to a dedicated in-memory sketch (e.g., HyperLogLog/Count-Min Sketch) instead of exact counts.
- Kafka partition count (currently 64-128) becomes throughput-limiting for consumer parallelism; requires partition count increase + consumer group rebalancing, plus broker fleet growth.
- Model Serving CPU fleet scales roughly linearly (860+ vCPUs → ~3,600 vCPUs) — still tractable but cost becomes a bigger conversation, likely pushing toward model distillation or a lighter model variant for low-risk traffic segments.

**At 100x traffic (3M peak QPS)** — hypothetical extreme, useful to reason about the ceiling:
- Synchronous per-request architecture itself strains: the request-response gateway pattern doesn't scale indefinitely without a fundamentally different tiering — likely need a **fast pre-filter tier** (ultra-cheap rules/heuristic model handling 95% of "obviously fine" traffic in <5ms) with only the ambiguous 5% routed to the full GBM+feature-store path, to keep the expensive path's absolute QPS manageable.
- Graph/embedding pipeline (50M entities → 500M entities) breaks the single-machine HNSW index assumption; would need distributed ANN (sharded index across multiple nodes) and distributed graph training (multi-node GraphSAGE, not just multi-GPU single-node).
- Offline feature store query/training-join costs grow with entity count; would need more aggressive pre-aggregation and possibly a move to a streaming-first feature computation model (fully eliminate batch recompute) to keep training-data-prep time bounded.

## 43. Latency Bottlenecks

**p50 budget (~20ms happy path):** network/gateway overhead ~4ms, feature fetch (Redis) ~4ms, rules eval ~1ms, model inference ~3ms, decision aggregation + serialization ~2ms, remaining is transport/queueing overhead.

**p99 budget (150ms target) — where the tail actually comes from:**
- Redis tail latency during cache-shard rebalancing or hot-key contention (can spike to 20-40ms) — largest controllable contributor.
- GC pauses in JVM-based components (if Flink/JVM feature-pipeline shares infra with serving — mitigated by keeping serving-path components in a non-GC-pause-prone runtime, e.g., Go/Rust/C++ for the hot path).
- Cross-AZ network hops if pod scheduling isn't zone-aware (adding 1-5ms per hop) — mitigated via topology-aware routing/pod affinity to same-AZ Redis shards.
- Cold model reload on pod restart mid-traffic (readiness gating prevents serving during load, but a fleet-wide rolling restart can transiently reduce capacity and increase queueing latency) — mitigated by conservative maxUnavailable settings during rollout.
- Graph-cluster-risk feature lookup (KV read of precomputed embedding-derived tag) — normally ~1ms but can spike if that KV store shares infra with the primary velocity-counter Redis under load; kept on a separate cluster to isolate blast radius.

## 44. Cost Bottlenecks

- **Standing CPU serving fleet** (360-720 vCPUs across 3 regions, always-on) is the largest recurring cost — driven by peak-provisioning for launch-day spikes that are infrequent relative to baseline traffic; addressed via aggressive autoscaling floor/ceiling tuning and reserved-instance commitment only for the true baseline.
- **7-year audit-log retention at ~618TB** — even at cold-storage rates this is a meaningful multi-year cost; addressed via storage-class tiering and considering whether full-fidelity retention is needed past the typical 18-24 month active-dispute window vs. a compressed/summarized form for years 3-7 (subject to legal/compliance sign-off).
- **Cross-region data transfer** for active-active replication (offline store sync, model registry replication, MirrorMaker for Kafka) — non-trivial at EA's traffic scale; mitigated by compressing replicated payloads and replicating only aggregated/derived data where raw fidelity isn't needed cross-region.
- **GPU training jobs**, while ephemeral and spot-priced, can spike in cost if a training run fails and retries repeatedly without a circuit breaker — mitigated by a max-retry cap and cost-anomaly alerting on the training-job billing tag.
- **Analyst review queue (human cost, not infra)** — indirectly a "cost bottleneck" in the sense that an overly conservative model (too many `challenge` decisions) scales analyst headcount need faster than infra; model threshold tuning is itself a cost lever, balancing analyst headcount against false-positive player friction.

## 45. Interview Follow-Up Questions

1. How would you handle the fact that fraud labels arrive 30-90 days after scoring — doesn't that make your "daily retrain" claim somewhat hollow?
2. Your chargeback ratio is approaching the card-network threshold (0.9%) — walk me through exactly what changes in the system in the next 24 hours.
3. How do you prevent the model from simply learning to copy the existing rules engine's decisions instead of adding independent signal?
4. A fraud ring figures out your model relies heavily on "device fingerprint reputation" and starts rotating fresh, never-before-seen devices for every transaction. What breaks, and how do you adapt?
5. How do you calibrate the tradeoff between blocking a high-LTV whale player's legitimate $500 purchase vs. missing a $500 fraud transaction?
6. Explain point-in-time correctness in your feature store with a concrete example of how you'd get it wrong.
7. Why GBM and not a neural network here, and under what conditions would you revisit that decision?
8. How would this design change if EA acquired a new studio with a completely different payment/game economy (e.g., subscription-based, no microtransactions)?
9. What's your plan if the online Redis feature store and the offline Iceberg feature store silently drift apart (training/serving skew)?
10. How do you handle a false-positive spike right after a marketing promotion causes a legitimate surge in first-time purchasers who look "new-account-high-value" (a classic false-positive trigger)?

## 46. Ideal Answers

1. **Daily retrain value despite label lag**: Daily retrain isn't mainly about new confirmed-fraud labels (those lag 30-90 days) — its real value is fast incorporation of analyst-verdict labels (1-48hr turnaround) and quick reaction to shifting feature/velocity distributions. A periodic larger "full recalibration" still runs once enough chargeback labels accumulate.

2. **Approaching chargeback threshold**: Tighten `challenge`/`deny` thresholds on the highest-risk-score decile only (to limit false-positive blast radius) and push an emergency rules hot-patch targeting the specific BIN ranges/device clusters driving the chargebacks. Escalate to finance/card-network relations in parallel, since threshold breaches carry contractual implications.

3. **Model vs. rules independence**: Exclude rule-trigger flags as direct input features so the model can't just learn "rule X fired → fraud" (circular). Evaluate model lift specifically on transactions the rules engine passed as "clean" — if it finds no additional fraud there, it isn't adding value.

4. **Device-rotation adaptation**: The graph-embedding/ring-detection layer looks at *shared* signals across accounts (same card BIN, IP/ASN, behavioral timing) that persist even when device identity is rotated, so it isn't fooled by fingerprint changes alone. A spike in "never-seen device" reason codes is itself a drift signal (Section 21) that can trigger a temporary MFA-step-up rule while the graph model's embeddings catch up.

5. **Whale vs. fraud tradeoff**: This is a threshold/policy decision, not purely a modeling one — maintain a VIP/high-LTV allowlist with a higher `challenge` threshold, and route borderline high-LTV cases to step-up verification (3DS) rather than outright `deny`. The threshold is justified by continuously comparing $-weighted cost of false positives (LTV impact) vs. false negatives (chargeback+goods cost).

6. **Point-in-time correctness failure example**: If "lifetime chargeback count" is computed using the *current* count rather than the count *as of the transaction's original timestamp*, older training examples leak future chargeback information the model wouldn't have had at scoring time, inflating offline metrics that then collapse in production. Fix requires the feature store to track `feature_available_at` and join strictly on `feature_available_at <= transaction_timestamp`.

7. **GBM vs. neural net**: GBM was chosen for CPU-only low-latency serving, strong tabular/velocity-feature performance, and interpretability (SHAP-style reason codes for analysts). Would revisit if rich sequential clickstream data showed strong incremental lift, or if latency/infra constraints relaxed enough to justify GPU inference.

8. **New studio with different economy**: The core architecture (scoring gateway, feature store, rules+ML hybrid) is largely payment-model-agnostic, but feature set and label definitions would need rework — subscription fraud (account-sharing, trial abuse) looks different from microtransaction fraud (stolen-card burst-buying). Practically, stand up a per-studio model variant on that studio's own labels while reusing the shared graph/ring-detection layer where permissible.

9. **Online/offline feature drift**: Detected via a prediction-serving-skew monitor (Section 21) comparing online-computed feature values against an offline recomputation for the same event, alerting above ~5% divergence. Typically root-caused to either a bug in one of the two parallel implementations (mitigated by sharing a UDF library across streaming/batch) or a windowing definition mismatch.

10. **Post-promotion false-positive spike**: This is a known, anticipated pattern — the monitoring stack should distinguish "new-account-high-value" reason-code spikes correlated with a known marketing-campaign window from spikes with no such correlation (real attack signal). Practically, temporarily relax that rule during pre-approved promo windows, route affected transactions to `challenge`/step-up rather than `deny`, and retrain afterward so the model learns the pattern isn't inherently risky.
