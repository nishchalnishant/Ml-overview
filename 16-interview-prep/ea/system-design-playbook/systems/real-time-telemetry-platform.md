# Real-Time Telemetry Platform

## 1. Problem Framing

Design a platform that ingests, transports, stores, and serves game telemetry (player actions, match events, economy transactions, crash/perf signals, matchmaking outcomes) at EA scale, near-real-time, feeding operational dashboards and downstream ML (churn, LiveOps balancing, anti-cheat, personalization).

Clarify up front:
- Ingestion-only, or does it own serving (feature store, dashboards, training data)?
- Single flagship title vs. shared platform across 20+ studios with heterogeneous schemas?
- Build vs. extend existing Kafka pipelines?
- Loss-tolerant (analytics) vs. loss-intolerant/latency-critical (anti-cheat, matchmaking)?

Treat this as a **shared, multi-title, multi-tenant platform** — the hardest, most representative version.

## 2. Functional Requirements

- FR1: Ingest events from clients (console/PC/mobile) and backend services via SDK.
- FR2: Validate, dedupe, enrich events (geo, session, device, build) in-flight.
- FR3: Support schema evolution per event type without breaking producers/consumers.
- FR4: Route to hot storage (sub-2s freshness) for dashboards, anti-cheat, LiveOps triggers.
- FR5: Route to cold storage (data lake) for batch analytics and ML training.
- FR6: Query/subscription API for downstream consumers.
- FR7: Backfill/replay of historical streams for reprocessing.
- FR8: Per-title, per-event-type retention and PII redaction policies.
- FR9: Real-time aggregation windows (e.g., kills/min/match) for anti-cheat/matchmaking.
- FR10: Self-service schema registry and event catalog.

## 3. Non-Functional Requirements

| Dimension | Target |
|---|---|
| Ingestion throughput | 3M events/sec peak (season launch), 800K/sec steady |
| Ingest→hot latency (p99) | < 2s |
| Ingest→cold latency (p99) | < 15 min |
| Availability (ingest) | 99.95% |
| Durability (cold tier) | 11 nines, no acknowledged event lost |
| Consistency | At-least-once default; exactly-once for economy/billing events via idempotency keys |
| Schema evolution | Zero-downtime, backward+forward compatible only |
| Cost ceiling | < $0.08 per million events fully loaded |
| Backpressure | Absorb 10x burst for 5 min without data loss |

## 5. Assumptions

1. 70M MAU, 9M CCU peak during a global season launch.
2. ~45 events/min per active player (movement throttled, combat/economy discrete).
3. SDK-side batching/sampling reduces wire traffic to ~800K events/sec steady, 3M/sec peak.
4. Average event size (compressed protobuf): 350 bytes.
5. Hot tier retention: 7 days. Cold tier: 3 years, PII redacted after 90 days.
6. 20 titles, ~150 event schemas, growing 10/month.
7. Anti-cheat/matchmaking need < 2s freshness; BI/ML tolerate 15 min–24 hr.
8. AWS-primary (Kinesis/MSK both viable), on-prem overflow for baseline-load cost savings.

## 6. Capacity Estimation

**Throughput**: steady 800K/s × 350B = 280 MB/s; peak 3M/s × 350B ≈ 1.05 GB/s. Daily volume ≈ **69B events/day** steady (up to ~150B on launch days).

**Kafka sizing**: 3x replication. At ~10MB/s/partition, peak needs ~105 partitions minimum; provision 300/topic for headroom. Cluster network at peak ≈ 3.15 GB/s (with replication) → ~8 brokers minimum, provision 16 for N+1. Hot tier (7-day) ≈ 169 TB/week; tiered storage keeps only ~24h on broker-local disk, rest offloaded to S3.

**Cold storage**: 3-year retention ≈ 26.5 PB raw, ~5.3 PB after Parquet+zstd compression (~5:1). At ~$0.021/GB/month blended → ~$111K/month storage.

**Stream processing (Flink)**: ~1 vCPU per 8K events/sec (decode, validate, enrich, dedupe). Peak ≈ 375 vCPUs, provisioned with autoscaling headroom to ~600.

**ML compute**: streaming feature aggregation is CPU-only; drift jobs run batch/hourly on a small CPU Spark cluster. No GPU needed at ingest — this system doesn't serve DL inference itself.

**Query layer**: ~50K read QPS against hot-tier aggregates, p99 < 50ms, served via caching + materialized views (not raw Kafka reads).

## 7. High-Level Architecture

```
Game Clients/Servers (console, PC, mobile, dedicated servers)
        │ batched, compressed event envelopes (HTTPS/gRPC)
        ▼
Edge Ingest Gateway (authN, regional PoPs, rate limiting)
        ▼
Ingest Service (stateless, schema validation, idempotency keys)
        ▼
Kafka/Kinesis (raw topics, 300 partitions each, tiered hot/warm storage)
   │                                   │
   ▼                                   ▼
Stream Processing (Flink)      Sink Connectors → Cold Lake
 - dedupe (Bloom/Redis)                │
 - enrichment (geo, session)           ▼
 - windowed aggregation        Data Lake (S3, Parquet, Iceberg, partitioned by title/date)
 - DLQ routing                         │
   │                                   ▼
   ▼                          Batch ETL/Spark → Feature Store (offline), training data, BI
Hot Store (Redis/Druid/ClickHouse)
   │
   ▼
Query/Serving API (gRPC/REST) → Dashboards, Anti-Cheat, LiveOps, Feature Store (online)

Cross-cutting: Schema Registry | Monitoring/Tracing | Alerting | IAM/AuthZ
```

## 8. Low-Level Components

| Component | Responsibility | Scaling Unit |
|---|---|---|
| Edge Ingest Gateway | TLS termination, authN, regional routing, coarse rate limiting | Horizontal pods per region, scales with CCU |
| Ingest Service | Schema validation, decompression, idempotency-key assignment, publish to Kafka | Stateless, scales on CPU + producer backpressure |
| Schema Registry | Versioned Avro/Protobuf schemas, compatibility enforcement | Small HA 3-node cluster, cached client-side |
| Kafka/Kinesis Cluster | Durable, per-key-ordered buffer | Brokers/shards scale with partitions & throughput |
| Stream Processing (Flink) | Dedupe, enrichment, windowed aggregation, DLQ routing | Task managers scale with partition count/lag |
| Hot Store (Druid/ClickHouse + Redis) | Serve sub-2s aggregates and recent events | Scales with query QPS + ingest rate, sharded by title |
| Cold Lake (S3 + Iceberg/Glue) | Long-term columnar storage for BI/ML | Cost-scaled, effectively infinite |
| Sink Connectors | Reliable Kafka→S3 delivery, exactly-once | Scales with topic partition count |
| Query/Serving API | Unified read API for all consumers | Stateless, scales with consumer QPS |
| DLQ Processor | Reprocess/quarantine malformed events | Scales with DLQ volume (should be ~zero) |
| Feature Store connector | Materializes streaming aggregates online (Redis/DynamoDB) and offline (S3) | See §15 |

## 9. API Design

**Ingest API**
```
POST /v2/events:batch
Headers: Authorization: Bearer <service-jwt>, X-Idempotency-Key: <uuid>
Body: { "title_id", "schema_version", "events": [{ "event_id", "event_type", "occurred_at", "session_id", "player_id", "payload" }] }
Response 202: { "accepted": 128, "rejected": 2, "rejections": [...] }
```

**Query/Serving API**
```
GET /v2/aggregates/{title_id}/{metric}?window=1m&group_by=match_id&since=<ts>
GET /v2/events/{title_id}/{event_type}:stream   (gRPC server-streaming, for anti-cheat subscription)
POST /v2/replay { title_id, event_type, from, to, sink }
```

**Schema Registry API**
```
POST /schemas/{event_type}/versions   — register (compatibility check: BACKWARD)
GET  /schemas/{event_type}/versions/latest
```

**Versioning**: URI-versioned (`/v2/`) for API surface; payload schema versioned independently via `schema_version` + Schema Registry compatibility gate — additive fields don't need an API bump.

| Endpoint | Method | SLA (p99) | AuthN |
|---|---|---|---|
| `/v2/events:batch` | POST | < 200ms | Service JWT / device-attested token |
| `/v2/aggregates/*` | GET | < 50ms | OAuth2 client-credentials |
| `/v2/events/*:stream` | gRPC stream | < 2s freshness | mTLS (internal only) |
| `/v2/replay` | POST | async | Internal RBAC (data-eng) |

## 10. Database Design

| Store | Type | Used For | Partition Key | Why |
|---|---|---|---|---|
| Kafka/Kinesis | Log | Transport buffer | `title_id + player_id`/`match_id` | Ordering within session, even distribution |
| Redis Cluster | KV, in-memory | Dedupe filters, session state, online features | `player_id` | Sub-ms latency, TTL eviction |
| Druid/ClickHouse | Columnar OLAP | Hot-tier aggregates, dashboards, anti-cheat windows | `title_id` + time segment | Sub-second time-windowed rollups |
| S3 + Iceberg | Columnar (Parquet) | Cold tier, ML training, BI | `title_id/event_type/dt/hour` | Cheap durable storage; schema evolution + time travel |
| DynamoDB | NoSQL KV | Idempotency ledger (exactly-once economy events) | `idempotency_key` | Single-digit ms writes, TTL expiry |
| Postgres (small) | Relational | Schema Registry metadata, event catalog, DLQ audit | N/A | Strong consistency for config-plane |

Iceberg table sketch (`player_action`): `event_id, title_id, event_type, occurred_at, ingested_at, session_id, player_id (pseudonymized after 90d), payload STRUCT, schema_version`, partitioned by `(title_id, days(occurred_at))`.

Columnar cold tier lets analytics/ML queries scan specific columns across billions of rows with predicate pushdown — 10-20x cheaper than row-oriented JSON scans.

## 11. Caching

| Cache | What's Cached | Strategy | Invalidation |
|---|---|---|---|
| Schema Registry client cache | Latest schema per event_type | Cache-aside, in-process | TTL 60s + webhook push |
| Redis dedupe filter | Recent event_id/idempotency keys | Write-through on ingest | TTL = dedupe window (24h) |
| Hot-tier aggregate cache | Pre-computed rollups (1m/5m/1h) | Write-through from Flink | Rolling 7-day eviction |
| Query API response cache | Frequently-hit aggregates | Cache-aside, short TTL (5-10s) | TTL + explicit purge on replay |
| Online feature store cache | Latest feature vector per player | Write-through from streaming job | Overwritten per event; TTL staleness ceiling |

Cache-aside is default (simpler, resilient); write-through used where staleness would directly harm anti-cheat/LiveOps decisions.

## 12. Queues & Async Processing

- **Ingest→Kafka**: at-least-once; client retries with backoff + local disk buffer (up to 5 min) during connectivity loss.
- **Exactly-once path**: economy events carry a client idempotency key; Ingest Service checks DynamoDB ledger (conditional write) before producing — exactly-once *effect* over at-least-once transport.
- **DLQ**: schema-validation failures or Flink deserialization errors route to `*.dlq` with error metadata. Alerts if DLQ rate > 0.01% for 5 min; supports manual replay.
- **Backpressure**: client SDK sheds low-priority telemetry first (never economy/anti-cheat) on `429`; Ingest Service token-bucket limits per title/api_key; Kafka producer/consumer lag triggers Flink autoscaling; circuit breaker fails over to a lower-replication overflow topic rather than rejecting writes outright during extreme bursts.

## 13. Streaming & Event-Driven Architecture

**Topics** (examples, ~150 event types across 20 titles): `raw.{title}.player_action.v3`, `raw.{title}.economy_transaction.v2`, `raw.{title}.match_lifecycle.v1`, `enriched.{title}.player_action`, `agg.{title}.kills_per_minute`, `dlq.{title}.{event_type}`.

**Event schema** (Protobuf, versioned): `event_id, title_id, schema_version, occurred_at, session_id, player_id, ActionPayload (oneof, evolves per action type)`.

**Consumer groups**: `anti-cheat-realtime` (low-latency, small window), `liveops-dashboards` (reads agg.*), `cold-sink-connectors` (writes S3), `feature-store-hydration` (writes online store) — each scales independently, capped by partition count.

**Ordering**: per-key only (`player_id`/`match_id`) — no global ordering, acceptable since downstream aggregations are commutative/associative or session-scoped.

## 14. Model Serving

Primarily a *data plane*, not a model-serving system, but hosts two lightweight inline inference paths:

| Use case | Model | Where | Why here |
|---|---|---|---|
| Anti-cheat heuristic scoring | GBT (< 5MB) | Embedded in Flink job | Sub-2s SLA rules out a network hop to a separate serving cluster |
| Event-quality/anomaly flagging | Isolation forest | Sidecar in Ingest Service | µs-level, avoids adding a hop on the hot path |

Heavier models (churn, matchmaking skill rating, recommendation) are out of scope — separate serving systems that *consume* this platform's feature store/streams. Inline scorer batches per Flink micro-batch (200-500 events/window); CPU-only, no GPU needed.

## 15. Feature Store

- **Online**: Redis/DynamoDB, hydrated from Flink enrichment (e.g., `player:{id}:kills_last_5m`). Read latency < 5ms p99.
- **Offline**: Iceberg tables in S3, same feature definitions computed in batch (shared feature-definition library avoids train/serve skew).
- **Point-in-time correctness**: every row carries `event_time` and `ingested_at`; training joins are `event_time`-based as-of joins so no future leakage. Late data (>24h) triggers versioned backfill jobs so in-flight training runs aren't silently mutated.
- **Freshness SLA**: online features match the 2s hot-tier SLA; offline features on the 15-min cadence with daily full reconciliation.

## 16. Vector Database

**N/A.** This platform transports structured/semi-structured events — no embeddings generated or queried in the ingest/storage path. Titles needing similarity search consume this platform's streams as input to their own embedding pipelines (out of scope here).

## 17. Embedding Pipelines

**N/A**, same reason as §16 — this platform is upstream of embedding generation. Downstream consumers run their own embedding pipelines off `enriched.*` topics or the offline feature store, keeping this system's on-call surface bounded to transport/storage/serving.

## 18. Inference Pipelines

Request lifecycle for the one inline inference — real-time anti-cheat scoring:

```
Client emits player_action event
  → Edge Gateway (authN, ~2ms)
  → Ingest Service (validate, idempotency check, ~5ms)
  → Kafka produce (ack after ISR write, ~10-20ms)
  → Flink micro-batch (200-500 events, ~200ms window trigger)
      → Enrichment (geo, session join via Redis, ~3ms)
      → Anti-cheat scorer (GBT, in-process, batched, ~1ms/event)
          → score > threshold? → emit to flags.anti_cheat → review queue
          → else → emit to enriched.* + windowed aggregate update (Druid, ~10ms)
  → Query API serves aggregate to LiveOps/matchmaking (< 50ms read)

Total ingest→hot-tier-visible p99: ~250-300ms typical, bounded by 2s SLA under worst-case windowing.
```

## 19. Training Pipelines

Trains only the two small inline models; also *produces* training data other systems consume.

- **Data prep**: batch Spark job joins `player_action` + human-reviewed `flags.anti_cheat` outcomes from cold tier, using the shared feature-definition library for parity.
- **Orchestration**: Airflow DAG — extract → point-in-time feature-join → train (single-node XGBoost, no distributed training needed) → validate (holdout + backtest) → register in MLflow → canary deploy into Flink (§20).
- **Cadence**: full retrain weekly, on shared batch compute (not dedicated to this system).

## 20. Retraining & Drift

| Trigger | Condition | Action |
|---|---|---|
| Scheduled | Weekly | Full retrain on trailing 90 days |
| Drift-triggered | Feature PSI > 0.2 on top-10 features | Ad-hoc retrain within 24h |
| Performance-triggered | Anti-cheat precision drops > 5pts vs. baseline | Immediate retrain + rollback pending investigation |
| Schema-change-triggered | New/changed schema affecting model inputs | Re-validate pipeline compatibility, block deploy if breaking |
| New title onboarding | Different cheat patterns | Cold-start with cross-title baseline, dedicated retrain after 30 days |

| Drift type | Metric | Threshold | Action |
|---|---|---|---|
| Data drift (volume/shape) | Per-event-type volume z-score vs. 7-day baseline | \|z\|>3 for 15 min | Page on-call — client bug, bot traffic, or schema break |
| Data drift (feature dist.) | PSI on key anti-cheat features | >0.2 investigate, >0.3 auto-flag | Feeds retraining trigger |
| Concept drift | Rolling precision/recall vs. weekly audit set | Precision drop >5pts | Retrain + shadow-deploy challenger |
| Schema drift | Registry compatibility failures | Any BACKWARD-incompatible change | Hard block at registry |
| Pipeline drift | % events with null enrichment fields | >1% for 10 min | Page data-eng, check enrichment dependency |

## 21. Monitoring & Alerting

| Layer | Metrics |
|---|---|
| Infra | Broker CPU/disk/network, consumer lag, Flink checkpoint duration/failures, Redis hit rate, S3 error rate |
| Pipeline health | Events/sec in vs. out per stage, DLQ rate, schema-failure rate, end-to-end latency percentiles |
| Model quality | Anti-cheat precision/recall (weekly audit), score drift, false-positive ban-appeal rate |
| Business | Events/day per title, active schemas, cost per million events, onboarding lead time |
| SLA | % events meeting hot/cold freshness SLAs, ingest availability (synthetic canaries every 10s/region) |

| Alert | Condition | Severity |
|---|---|---|
| Ingest availability | Canary fails 3 consecutive checks in a region | P1, page primary on-call |
| Consumer lag breach | Lag > 2x freshness SLA for 5 min | P1, page data-eng |
| DLQ spike | > 0.01% traffic for 5 min | P2 (page if > 0.1%) |
| Broker disk pressure | > 85% utilization | P2, page infra |
| Schema compatibility break attempt | Any rejected registration | P3, Slack notify only |
| Model precision degradation | Weekly audit shows > 5pt drop | P2, notify ML on-call, triggers retrain |
| Cost anomaly | Daily spend > 20% above 7-day average | P3, Slack |

On-call is tiered: L1 platform/infra (Kafka, Flink, storage), L2 ML (model quality, drift), L3 data-eng (schema, pipeline correctness).

## 22. Logging & Security

- **Structured logs** (JSON) at every hop with `trace_id, event_id, title_id, stage`; `player_id` pseudonymized in logs, raw PII never logged. Operational logs 30 days hot, 1 year cold archive; event PII purged at 90 days regardless. Immutable audit trail (schema changes, replays, PII access) retained 3 years.

**Threats**: spoofed client telemetry (poisoning anti-cheat data), compromised server credentials (economy event replay/duplication), PII exfiltration from the data lake, DLQ/replay endpoints as a replay-attack vector, client-flood DoS.

**Mitigations**: mTLS + device/build attestation for clients; server-side plausibility checks flag suspicious streams rather than trusting client data; idempotency-key ledger prevents economy replay; row/column-level access policies on the lake with masked PII by default; replay API restricted to RBAC + fully audited; encryption at rest (S3 SSE-KMS, Kafka disk) and in transit (TLS 1.3).

**AuthN**: client→gateway via per-title API key + device attestation + short-lived JWT; service→service via mTLS/SPIFFE; dashboard/BI→API via OAuth2 SSO + RBAC; admin/replay APIs via OAuth2 client-credentials + RBAC + mandatory audit log.

## 23. Rate Limiting & Autoscaling

- **Rate limiting**: token bucket per `(title_id, api_key)` at the gateway. Default 500 events/sec/device. Per-title soft cap at provisioned share, hard cap with `429`+`Retry-After` beyond that. Priority shedding throttles cosmetic/heartbeat telemetry before economy/anti-cheat/crash events.
- **Autoscaling**: Ingest Service/Gateway via HPA (CPU + queue depth). Flink task managers via KEDA on consumer lag (scale out > 30s lag, scale in after 10 min stable). Kafka brokers capacity-planned quarterly, not dynamically autoscaled (stateful, rebalance-expensive) — burst absorbed via tiered storage/backpressure instead. Druid query nodes via HPA on QPS/p99. S3 is inherently elastic; Spark ETL autoscales on job queue depth.

## 24. Cost Optimization

- Spot/preemptible instances for Flink and Spark workers (checkpoint-recoverable) — ~60-70% savings.
- Kafka tiered storage (hot NVMe + warm S3) avoids over-provisioning broker disk.
- S3 lifecycle policies to Glacier tiers after 90 days / 1 year.
- Protobuf + zstd cuts payload size 4-6x vs. JSON.
- Sampling low-value telemetry (e.g., movement heartbeats at 1-in-10) with sampling rate stored as metadata.
- Partition counts right-sized per event type rather than uniform 300.
- Reserved capacity for steady-state baseline; on-demand/spot only for burst headroom.

## 25. Operational Concerns

At SDE2 scope, this is a checklist: **backups** (automated snapshots of model registry/feature store with tested restore), **rollback** (one-command revert to last-known-good), **canary/blue-green rollout** (shift small traffic %, watch error rate + key model metrics, then ramp), **basic observability** (dashboards/alerts on latency, error rate, top model-quality signals, wired to on-call). Kubernetes/Terraform manifests and multi-region active-active topology are Staff/Principal-level — know they exist, don't rehearse the details.

## 26. Why This Architecture

- Kafka/Kinesis as durable buffer decouples variable client-ingest rates from downstream processing — essential given the 10x burst requirement without over-provisioning every consumer for peak.
- Separate hot (Druid/Redis) and cold (S3/Iceberg) tiers match different access patterns: sub-second window queries vs. petabyte scan-heavy analytics.
- Schema Registry with enforced compatibility lets 20+ independent teams evolve ~150 schemas without a central bottleneck.
- Inline lightweight scoring avoids a network hop that would blow the 2s SLA; heavier ML stays out of this system's blast radius by design.
- Region-local hot tiers respect that a player session doesn't need cross-region consistency, while cold-tier replication serves genuinely global consumers (BI, training).

## 27. Alternative Architectures

| Alternative | Why Rejected / When Preferred |
|---|---|
| Single global Kafka cluster (no regional sharding) | Cross-region producer latency hits ingest SLA; only viable for a much smaller, single-region title |
| Lambda architecture (separate batch + speed layers) | Rejected for Kappa-style (single streaming pipeline + replay) to avoid batch/streaming logic drift; reconsider only if streaming reprocessing cost becomes prohibitive |
| Managed serverless (Firehose+Lambda, no Flink) | Preferred for a smaller/simpler title (< 10K events/sec); rejected here since windowed aggregation, dedupe, inline scoring need Flink's stateful model |
| Direct client-to-datalake batch upload (no streaming tier) | Fails the < 2s freshness requirement; only suits pure offline-analytics scope |

## 28. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| At-least-once + idempotency-key exactly-once for critical events | Simple default, strict guarantee where it matters | Non-critical events can rarely double-count (acceptable) |
| Kappa over Lambda | Single source of truth, less logic duplication | Broad backfill/replay at 3-year scale is compute-expensive |
| Inline lightweight models vs. dedicated serving cluster | Meets 2s SLA, no extra hop | Caps model complexity/size runnable inline |
| Region-local hot tier | Meets latency SLA, avoids CAP-theorem pain | Global real-time rollups lag ~15 min via cold lake |
| Tiered Kafka storage (hot broker + S3 warm) | Big disk cost savings | Slightly higher latency for "recent but not hottest" reads |
| Schema Registry hard-blocks incompatible changes | Prevents consumer breakage | Adds friction for teams needing breaking changes (must version instead) |

## 29. Failure Modes

| Scenario | Impact | Mitigation |
|---|---|---|
| Kafka broker AZ outage | Leader elections, brief producer stalls | RF=3 across AZs, min.insync.replicas=2, client retry |
| Flink crash-loop (bad deploy) | Consumer lag grows, freshness SLA breach | Canary/shadow deploy, savepoint restart, lag alerting |
| Schema Registry outage | New registrations blocked | HA 3-node; client-side cache keeps ingest running for known types |
| Regional network partition | Region isolated from replication target | Active-active continues locally; replication catches up post-partition |
| Burst beyond provisioned headroom | Priority shedding, low-priority telemetry dropped | Economy/anti-cheat protected, cosmetic telemetry sampled down first |
| Poison-pill malformed event | Could block naive consumer | Per-message DLQ isolation, Flink continues past it |
| DynamoDB idempotency-ledger throttling | Exactly-once path could fail open | Autoscaling/on-demand capacity; fail-safe logs + alerts rather than silently dropping the check |

## 30. Bottlenecks

- **Scaling (10x, 30M events/sec)**: Kafka partition count (300/topic) becomes the parallelism ceiling — needs re-partitioning or finer per-title sharding. Druid ingestion-per-node becomes a bottleneck requiring segment-sharding redesign. Redis dedupe filter needs re-sizing as cardinality grows.
- **Latency (2s SLA budget)**: dominant p99 contributor is the Flink micro-batch window trigger (up to ~400ms), especially during lag catch-up — first place to check on an SLA breach. Cold-tier 15-min SLA is dominated by deliberate sink-connector batching, not per-event latency.
- **Cost**: cross-AZ/cross-region replication (RF=3 + MirrorMaker2) is the largest "hidden" volume-proportional cost. Cold-tier 3-year retention (~$111K/month) grows as titles onboard — biggest lever is tightening retention per event-type value and sampling low-value events earlier. Provisioning for 10x burst year-round leaves capacity idle ~95% of the time; Kafka brokers can't spot/scale as easily as Flink/Spark.

## 31. Interview Follow-Ups

1. How would the design change to guarantee exactly-once for *all* event types, not just economy?
2. What breaks first if one title suddenly represents 80% of platform traffic?
3. How do you prevent one misbehaving title's schema changes from destabilizing the shared platform?
4. Anti-cheat false-positive rate spikes and starts banning legit players — how fast can you detect and roll back?
5. How do you support a title needing sub-100ms (not sub-2s) freshness?
6. How do you reconcile replay-based correction with 3-year retention and reprocessing cost at scale?
7. What changes if data residency rules require a country's player data never leave its region?
8. How do you test schema evolution before it hits production across 20+ independently-deploying teams?
9. Capacity planning for an unpredictable viral spike vs. a known scheduled event?
10. If most consumers only needed hourly-batch freshness, what would you simplify or remove?

## 32. Ideal Answers

1. **Exactly-once everywhere**: extend idempotency-key dedup (currently economy-only) to all events via a higher-throughput dedup store (sharded Redis + DynamoDB fallback); use Kafka transactional producers + `read_committed` consumers end-to-end.
2. **80% single-title breakout**: partition/segment hot-spotting is the first bottleneck. Mitigate with dedicated per-title capacity (tenant isolation) and a pre-provisioned fast-path tier a title can be promoted into.
3. **Isolating a misbehaving title**: per-title rate limits and topic/partition isolation already contain blast radius at transport; Schema Registry compatibility gates are scoped per event_type, so a bad push only affects that title's consumers.
4. **Anti-cheat false-positive spike**: weekly audit detection is too slow — add a faster proxy signal (ban-appeal rate spike vs. baseline) as a real-time alert. Rollback is already fast (savepoint restart to prior model); the fix is tightening detection latency.
5. **Sub-100ms for one title**: bypass the Flink micro-batch window (the dominant p99 contributor) for that event type — dedicated low-latency consumer group with direct Redis writes, accepting higher per-event compute cost for that narrow slice.
6. **Replay cost vs. retention**: default replay to scoped time-ranges/event-types, not full history; also tighten retention per event-type value rather than uniform 3 years.
7. **Data residency**: extend the region-local hot-tier design to exclude a residency-constrained region's data from cross-region replication and the global cold lake, with region-scoped BI views — a policy flag, not a structural redesign.
8. **Testing schema evolution across teams**: Registry enforces BACKWARD compatibility as a hard gate; pair with shared contract tests against golden historical events and a shadow-consumer canary before marking a schema default.
9. **Viral spike vs. scheduled event**: scheduled events get pre-provisioned, load-tested capacity; unpredictable spikes rely on backpressure/shedding plus fast KEDA autoscaling, since Kafka broker capacity can't react in minutes.
10. **Hourly-batch-only consumers**: drop the hot-tier OLAP layer and inline Flink enrichment, replace with Kafka Connect S3 sink + scheduled Spark batch jobs — safe only if no consumer needs sub-minute freshness.
