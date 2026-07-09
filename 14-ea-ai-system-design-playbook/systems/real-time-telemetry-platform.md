# Real-Time Telemetry Platform

## 1. Problem Framing & Requirement Gathering

Design a platform that ingests, transports, stores, and serves game telemetry (player actions, match events, economy transactions, crash/perf signals, matchmaking outcomes) at EA scale, in near-real-time, feeding both operational dashboards and downstream ML systems (churn models, LiveOps balancing, anti-cheat, personalization, recommendation).

Key framing questions to settle with interviewer up front:
- Is this ingestion-only, or does it own the serving path for downstream consumers (feature store, dashboards, model training)?
- Which titles/franchises? A single flagship live-service shooter (Apex-scale) vs. a shared platform across 20+ studios with heterogeneous schemas.
- Is this a build vs. extend problem (EA already runs Kafka-based pipelines) or greenfield?
- SLA owner: is telemetry loss tolerable (analytics use case) or does it feed real-time anti-cheat / matchmaking (loss-intolerant, latency-critical)?

This chapter treats it as a **shared, multi-title, multi-tenant platform** — the hardest and most representative version of the problem at EA.

## 2. Functional Requirements

- FR1: Ingest structured/semi-structured events from clients (console, PC, mobile) and backend services (matchmaking, economy, chat) via SDK.
- FR2: Validate, deduplicate, and enrich events (geo, session, device, build version) in-flight.
- FR3: Support schema evolution per event type without breaking producers or consumers.
- FR4: Route events to hot storage (sub-second to seconds freshness) for real-time dashboards, anti-cheat, and live ops triggers.
- FR5: Route events to cold storage (data lake) for batch analytics, BI, and ML training data.
- FR6: Expose a query/subscription API for downstream services to consume specific event streams or aggregates.
- FR7: Support backfill/replay of historical event streams for reprocessing (new model versions, bug fixes).
- FR8: Provide per-title, per-event-type retention and PII redaction policies.
- FR9: Support real-time aggregation windows (e.g., "kills per minute per match") for anti-cheat and matchmaking feedback loops.
- FR10: Provide a self-service schema registry and event catalog for game teams.

## 3. Non-Functional Requirements (latency, availability, throughput, consistency, cost)

| Dimension | Target |
|---|---|
| Ingestion throughput | 3M events/sec sustained peak (season launch), 800K events/sec steady-state |
| Ingest-to-hot-storage latency (p99) | < 2 s |
| Ingest-to-cold-storage latency (p99) | < 15 min |
| Availability (ingest path) | 99.95% (≈4.4 hrs downtime/year) |
| Durability | 99.999999999% (11 nines) for cold tier (object storage), no acknowledged event lost |
| Consistency | At-least-once delivery guaranteed; exactly-once for billing/economy-critical event types via idempotency keys |
| Schema evolution | Zero-downtime, backward + forward compatible changes only |
| Cost ceiling | < $0.08 per million events fully loaded (ingest + storage + compute) |
| Backpressure tolerance | Must absorb 10x burst for 5 minutes without data loss (patch-day / esports-event spikes) |

## 4. Clarifying Questions an Interviewer Would Expect

1. Is exactly-once semantics required for *all* events or only monetization/economy events?
2. What's the fan-out — how many distinct downstream consumers (anti-cheat, churn model, BI) read the same stream?
3. Do we need cross-title unified schemas, or is per-title schema autonomy acceptable?
4. What's acceptable data loss during a regional outage — do we fail open (drop) or fail closed (buffer client-side)?
5. Are clients trusted (dedicated servers) or untrusted (player consoles, spoofable telemetry)?
6. Is there a regulatory constraint (GDPR/COPPA) affecting minors' telemetry retention or fields collected?
7. What's the existing infra — Kafka on-prem, MSK, Kinesis, or greenfield choice?
8. Do downstream ML systems need point-in-time-correct joins (feature store), or just aggregate features?
9. What is the burst shape — sudden (patch drop, new season) vs. gradual (player growth)?

## 5. Assumptions

1. 70M monthly active users (MAU) across the platform's title portfolio; 9M concurrent peak (CCU) during a global season launch.
2. Average active player generates 45 events/minute during active play (movement snapshots throttled client-side, combat/economy events sent as discrete).
3. Steady state: 9M CCU × 45 events/min / 60 ≈ 6.75M events/sec is the *client-side generation ceiling*; SDK-side sampling/batching reduces wire traffic to ~800K events/sec steady-state, 3M events/sec peak (batched envelopes, not raw events, hit Kafka).
4. Average event size (post-compression, protobuf): 350 bytes.
5. Hot tier retention: 7 days. Cold tier retention: 3 years (BI/model training), with PII fields redacted after 90 days.
6. 20 game titles onboarded, ~150 distinct event schemas, growing 10/month.
7. Anti-cheat and matchmaking consumers require < 2s freshness; BI/ML training consumers tolerate 15 min–24 hr freshness.
8. Infra runs primarily on AWS (Kinesis/MSK options both viable) with on-prem overflow capacity at 2 EA data centers for cost reasons on baseline load.

## 6. Capacity Estimation

**Throughput**
- Steady state: 800,000 events/sec × 350 B = 280 MB/s ingress.
- Peak (season launch, 10x burst tolerance target): 3,000,000 events/sec × 350 B = 1.05 GB/s ingress.
- Daily event volume: 800K/s × 86,400 s ≈ **69.1 billion events/day** steady-state (matches "billions/day" scope; peaks push this to ~150B on launch days).

**Kafka/Kinesis sizing**
- Assume Kafka (MSK) with 3x replication.
- Partition throughput budget: 10 MB/s/partition (conservative, compressed protobuf).
- Peak partitions needed: 1.05 GB/s ÷ 10 MB/s = **105 partitions minimum**; provision 300 partitions per top-level topic for headroom + consumer parallelism.
- Broker sizing: each broker (d3.2xlarge equivalent, 25 Gbps NIC, NVMe) sustains ~500 MB/s network with replication overhead → peak 1.05 GB/s × 3 (replication) = 3.15 GB/s cluster network → **~8 brokers** minimum, provision 16 for N+1 and rolling upgrades.
- Local retention (7-day hot tier) storage on brokers/tiered storage: 69.1B events/day × 350 B × 7 days ≈ **169 TB/week** hot; with tiered storage (KIP-405 / Kinesis long-term retention offload), broker-local disk only holds ~24h buffer ≈ 24.2 TB, rest offloaded to S3-backed tiered storage.

**Cold storage (data lake)**
- 3-year retention: 69.1B events/day × 350 B × 365 × 3 ≈ **26.5 PB** raw. With columnar compression (Parquet + zstd, ~5:1 typical for repetitive telemetry) → **~5.3 PB** effective.
- S3 cost @ ~$0.021/GB/month (Glacier Instant/Infrequent tiers blended) → 5.3M GB × $0.021 ≈ **$111K/month** storage cost at steady state (before lifecycle tiering discounts kick in further at year 2-3).

**Compute for stream processing (Flink/Kafka Streams enrichment + aggregation layer)**
- Assume enrichment job needs ~1 vCPU per 8,000 events/sec (protobuf decode, validation, geo-IP enrich, dedupe check against Redis/Bloom filter).
- Peak: 3,000,000 / 8,000 ≈ **375 vCPUs**, provisioned as ~48 x 8-vCPU task managers with autoscaling headroom to 600 vCPUs.

**GPU/CPU for downstream ML (feature computation + drift jobs, not raw ingest)**
- Streaming feature aggregation (windowed counts, rates) is CPU-only (no GPU needed at ingest layer).
- Drift-detection jobs (batch, hourly) run on a shared 16-node CPU Spark cluster (32 vCPU/node) — no GPU requirement; this system does not serve DL inference itself (see §14 for how it's consumed by model-serving systems).

**API/query layer QPS**
- Downstream consumers (dashboards, anti-cheat, feature store hydration): estimate 50,000 read QPS against hot tier aggregates, p99 < 50ms — served via a caching + materialized-view layer, not raw Kafka reads.

## 7. High-Level Architecture

```
                         ┌─────────────────────────────────────────────────────┐
                         │                  Game Clients / Servers               │
                         │  (Console, PC, Mobile SDK, Dedicated Game Servers)    │
                         └───────────────┬───────────────────────────────────────┘
                                         │ batched, compressed event envelopes (HTTPS/gRPC)
                                         ▼
                         ┌───────────────────────────────┐
                         │   Edge Ingest Gateway (Envoy   │
                         │   + regional PoPs, authN,      │
                         │   client rate limiting)        │
                         └───────────────┬────────────────┘
                                         ▼
                         ┌───────────────────────────────┐
                         │  Ingest Service (stateless,    │
                         │  schema validation via Schema  │
                         │  Registry, idempotency keys)   │
                         └───────────────┬────────────────┘
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │        Kafka / Kinesis Streams (raw topics)     │
                 │   per-event-type topics, 300 partitions each,   │
                 │   tiered storage (hot: brokers, warm: S3)       │
                 └───────┬───────────────────────┬─────────────────┘
                         ▼                       ▼
        ┌───────────────────────────┐  ┌───────────────────────────────┐
        │  Stream Processing (Flink)│  │  Sink Connectors (Kafka Connect│
        │  - dedupe (Bloom/Redis)   │  │  / Firehose) → Cold Lake       │
        │  - enrichment (geo, sess) │  │                                │
        │  - windowed aggregation   │  └───────────────┬────────────────┘
        │  - DLQ routing on failure │                  ▼
        └───────┬───────────────────┘      ┌────────────────────────────┐
                ▼                          │  Data Lake (S3, Parquet,   │
   ┌─────────────────────────────┐         │  partitioned by title/date)│
   │  Hot Store (Redis / Druid /  │         │  + Glue Catalog / Iceberg  │
   │  ClickHouse real-time OLAP)  │         └───────────────┬────────────┘
   └───────┬───────────────────────┘                        ▼
           ▼                                    ┌────────────────────────────┐
┌────────────────────────────┐                  │  Batch ETL / Spark jobs →   │
│  Query/Serving API (gRPC/   │                  │  Feature Store (offline),  │
│  REST) → Dashboards,        │                  │  Training datasets, BI     │
│  Anti-Cheat, LiveOps,       │                  └────────────────────────────┘
│  Feature Store (online)     │
└────────────────────────────┘

   Cross-cutting: Schema Registry | Monitoring/Tracing | Alerting | IAM/AuthZ
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| Edge Ingest Gateway | TLS termination, client authN, regional routing, coarse rate limiting | HTTPS/gRPC, batched envelope | Horizontal pods behind regional LB, scales with CCU per region |
| Ingest Service | Schema validation (against registry), envelope decompression, idempotency-key assignment, publish to Kafka | Internal gRPC from gateway | Stateless; scales on CPU + Kafka producer backpressure |
| Schema Registry | Stores versioned Avro/Protobuf schemas per event type, enforces compatibility rules | REST API (Confluent Schema Registry compatible) | Small, HA 3-node cluster; read-heavy, cached client-side |
| Kafka/Kinesis Cluster | Durable, ordered (per-key) buffer between producers and consumers | Kafka protocol / Kinesis PutRecords | Brokers/shards scale with partition count & throughput |
| Stream Processing (Flink) | Dedupe, enrichment, windowed aggregation, routing to hot/cold sinks, DLQ on poison messages | Consumes Kafka, emits to Kafka/Redis/S3 | Task managers scale with partition count / backpressure lag |
| Hot Store (real-time OLAP: Druid/ClickHouse + Redis for point lookups) | Serve sub-2s-fresh aggregates and recent raw events | SQL-like query API, Redis GET/MGET | Scales with query QPS + ingestion rate; sharded by title |
| Cold Lake (S3 + Iceberg/Glue) | Long-term columnar storage for BI/ML training | S3 API, Athena/Presto/Spark SQL | Effectively infinite; cost-scaled, not compute-scaled |
| Sink Connectors | Reliable delivery from Kafka to S3 with exactly-once semantics (Kafka Connect S3 sink / Firehose) | Kafka Connect REST API | Scales with topic partition count |
| Query/Serving API | Unified read API for dashboards, anti-cheat, LiveOps, feature store hydration | gRPC/REST, GraphQL for BI | Stateless, scales with consumer QPS |
| DLQ Processor | Reprocess/quarantine malformed or schema-violating events | Consumes DLQ topic, alerts + manual/automated replay | Scales with DLQ volume (should be near-zero) |
| Feature Store connector | Materializes streaming aggregates into online feature store (Redis/DynamoDB) and offline (S3/Parquet) | Write path from Flink, read path via Feature Store API | See §15 |

## 9. API Design

**Ingest API** (client/server → Ingest Service)

```
POST /v2/events:batch
Headers: Authorization: Bearer <service-jwt>, X-Idempotency-Key: <uuid>
Body:
{
  "title_id": "apex-legends",
  "schema_version": "player_action.v3",
  "events": [
    {
      "event_id": "b3f1...",
      "event_type": "player_kill",
      "occurred_at": "2026-07-08T12:34:56.123Z",
      "session_id": "sess_9182",
      "player_id": "p_44821",
      "payload": { "weapon": "r99", "victim_id": "p_11029", "distance_m": 14.2 }
    }
  ]
}

Response 202 Accepted:
{ "accepted": 128, "rejected": 2, "rejections": [ { "event_id": "...", "reason": "SCHEMA_VALIDATION_FAILED" } ] }
```

**Query/Serving API** (downstream consumers)

```
GET /v2/aggregates/{title_id}/{metric}?window=1m&group_by=match_id&since=<ts>
Response 200:
{ "metric": "kills_per_minute", "window": "1m", "series": [ {"match_id": "m_1", "value": 12, "ts": "..."} ] }

GET /v2/events/{title_id}/{event_type}:stream   (gRPC server-streaming, for real-time anti-cheat subscription)

POST /v2/replay
{ "title_id": "apex-legends", "event_type": "player_kill", "from": "2026-06-01T00:00:00Z", "to": "2026-06-02T00:00:00Z", "sink": "kafka-topic:replay-out" }
```

**Schema Registry API**

```
POST /schemas/{event_type}/versions   — register new schema (compatibility check enforced: BACKWARD)
GET  /schemas/{event_type}/versions/latest
```

**Versioning strategy**: URI-versioned (`/v2/`) for the API surface; payload schema versioned independently via `schema_version` field + Schema Registry compatibility gate (no API version bump needed for additive fields).

| Endpoint | Method | SLA (p99) | AuthN |
|---|---|---|---|
| `/v2/events:batch` | POST | < 200ms | Service JWT / device-attested token |
| `/v2/aggregates/*` | GET | < 50ms | OAuth2 client-credentials |
| `/v2/events/*:stream` | gRPC stream | < 2s freshness | mTLS (internal services only) |
| `/v2/replay` | POST | async (job-based) | Internal RBAC (data-eng role) |

## 10. Database Design

| Store | Type | Used For | Partition/Shard Key | Why |
|---|---|---|---|---|
| Kafka/Kinesis | Log (append-only) | Transport buffer | `title_id + player_id` (or `match_id` for match events) | Ordering within a player/match session, even partition distribution |
| Redis Cluster | KV, in-memory | Dedupe bloom filters, session state, point lookups (online feature store) | `player_id` hash slot | Sub-ms latency, TTL-based eviction |
| Druid / ClickHouse | Columnar, real-time OLAP | Hot-tier aggregates, dashboards, anti-cheat windows | `title_id` + time-based segment | Optimized for time-windowed rollups at sub-second query latency |
| S3 + Apache Iceberg | Columnar (Parquet), object store | Cold tier, ML training data, BI | Partitioned by `title_id/event_type/dt=YYYY-MM-DD/hour=HH` | Cheapest durable storage; Iceberg gives schema evolution + time travel for reproducible training sets |
| DynamoDB | NoSQL KV | Idempotency-key ledger (dedupe for exactly-once economy events) | `idempotency_key` | Single-digit ms writes, TTL auto-expiry |
| Postgres (small) | Relational | Schema Registry metadata, event catalog, DLQ audit log | N/A (small dataset) | Strong consistency for config-plane metadata |

Schema sketch (Iceberg table, `player_action` event type):

```
player_action (
  event_id STRING,
  title_id STRING,
  event_type STRING,
  occurred_at TIMESTAMP,
  ingested_at TIMESTAMP,
  session_id STRING,
  player_id STRING,          -- pseudonymized after 90 days
  payload STRUCT<...>,       -- schema-evolved per event_type
  schema_version STRING
)
PARTITIONED BY (title_id, days(occurred_at))
```

Why columnar for cold tier: analytics/ML training queries scan specific columns (e.g., `weapon`, `distance_m`) across billions of rows — columnar + predicate pushdown cuts scan cost 10-20x vs. row-oriented JSON.

## 11. Caching

| Cache | What's Cached | Strategy | Invalidation |
|---|---|---|---|
| Schema Registry client cache | Latest schema per event_type | Cache-aside, in-process | TTL 60s + push invalidation via registry webhook |
| Redis dedupe filter | Recent `event_id`/idempotency keys (Bloom filter + exact-match fallback) | Write-through on ingest | TTL matches dedupe window (24h) |
| Hot-tier aggregate cache | Pre-computed 1m/5m/1h rollups for dashboards | Write-through from Flink job directly into Druid/Redis | Rolling window eviction (7-day hot retention) |
| Query API response cache | Frequently-hit aggregate queries (e.g., "current season KPIs") | Cache-aside, short TTL (5-10s) | TTL-based; explicit purge on backfill/replay completion |
| Online feature store cache | Latest feature vector per player | Write-through from streaming aggregation job | Overwritten on each new event; TTL as staleness ceiling (e.g., 1h) |

Cache-aside is default (simpler, resilient to cache node loss); write-through used specifically where staleness directly harms anti-cheat/LiveOps decisions (hot aggregates, online features) since those consumers cannot tolerate the cache-aside "first read is slow/stale" gap.

## 12. Queues & Async Processing

- **Ingest → Kafka**: at-least-once from client SDK (client retries on non-2xx with exponential backoff + local disk buffer up to 5 min of events during connectivity loss).
- **Exactly-once path**: economy/monetization events carry client-generated `idempotency_key`; Ingest Service checks DynamoDB ledger before producing to Kafka (conditional write) — guarantees exactly-once *effect* even though transport is at-least-once.
- **DLQ**: any event failing schema validation, or Flink job hitting a poison-pill deserialization error, routes to `*.dlq` topic with error metadata (`reason`, `raw_bytes`, `schema_version_attempted`). DLQ Processor alerts if DLQ rate > 0.01% of traffic for 5 min; supports manual replay after schema/producer fix.
- **Backpressure handling**:
  - Client SDK: local ring buffer + sampling degradation (drop low-priority telemetry like cosmetic-movement pings first, never drop economy/anti-cheat events) when server signals `429`/backoff.
  - Ingest Service: sheds load via token-bucket rate limiter per `title_id`/`api_key` before it reaches Kafka producer.
  - Kafka: partition-level producer backpressure via `linger.ms`/`batch.size` tuning; consumer lag monitored, triggers Flink autoscaling before lag exceeds hot-tier freshness SLA.
  - Circuit breaker: if Kafka cluster is saturated, Ingest Service fails over to a regional overflow topic (lower replication factor, best-effort) rather than rejecting client writes outright, favoring availability over strict ordering during extreme bursts.

## 13. Streaming & Event-Driven Architecture

**Topics** (examples, per title, ~150 total event types across 20 titles):
- `raw.{title_id}.player_action.v3`
- `raw.{title_id}.economy_transaction.v2`
- `raw.{title_id}.match_lifecycle.v1`
- `enriched.{title_id}.player_action` (post-Flink enrichment)
- `agg.{title_id}.kills_per_minute` (windowed aggregates)
- `dlq.{title_id}.{event_type}`

**Event schema** (Protobuf, versioned):

```protobuf
message PlayerActionEvent {
  string event_id = 1;
  string title_id = 2;
  string schema_version = 3;
  google.protobuf.Timestamp occurred_at = 4;
  string session_id = 5;
  string player_id = 6;
  ActionPayload payload = 7; // oneof, evolves per action type
}
```

**Consumer groups**:
- `anti-cheat-realtime` (reads `enriched.*`, low-latency, small window)
- `liveops-dashboards` (reads `agg.*`)
- `cold-sink-connectors` (reads `raw.*`, writes S3)
- `feature-store-hydration` (reads `enriched.*`, writes online store)
- Each consumer group scales independently; partition count (300/topic) sets max parallelism ceiling.

**Ordering guarantee**: per-key (`player_id` or `match_id`) ordering only — no global ordering, which is acceptable since downstream aggregations are commutative/associative (counts, sums) or explicitly session-scoped.

## 14. Model Serving

This system is primarily a *data plane*, not a model-serving system — but it hosts two lightweight inference paths inline:

| Use case | Model | Where it runs | Why here vs. separate service |
|---|---|---|---|
| Real-time anti-cheat heuristic scoring | Gradient-boosted tree (small, < 5MB) | Embedded in Flink job (per-event scoring) | Sub-2s SLA requires inline scoring, not a network hop to a separate serving cluster |
| Event-quality/anomaly flagging | Lightweight isolation forest | Sidecar in Ingest Service | Cheap enough (µs-level) to run inline; avoids adding a network hop on the hot path |

Heavier models (churn prediction, matchmaking skill rating updates, recommendation) are **out of scope** for this chapter — they are separate serving systems (see companion chapters) that *consume* this platform's feature store and event streams as input, typically via batch or async request (not embedded here). Serving framework for those: Triton/TorchServe behind the platform's Query API — not detailed further to keep scope on telemetry.

Batching: inline anti-cheat scorer batches per Flink micro-batch (typically 200-500 events/window) rather than per-event RPC, keeping GPU/CPU idle time low without adding latency (windowing already exists for aggregation).

Hardware: CPU-only for both inline models (tree ensembles, isolation forest) — no GPU needed in the ingest/streaming path.

## 15. Feature Store

- **Online store**: Redis/DynamoDB, hydrated directly from the Flink enrichment job — e.g., `player:{id}:kills_last_5m`, `player:{id}:session_kd_ratio`. Read latency target < 5ms p99 for anti-cheat/matchmaking consumers.
- **Offline store**: Iceberg tables in S3, same feature definitions computed in batch (Spark) for training-set generation — guarantees the same feature logic (shared feature-definition library) runs in both streaming and batch paths to avoid train/serve skew.
- **Point-in-time correctness**: every feature row carries `event_time` and `ingested_at`; offline training joins use `event_time`-based as-of joins (not `ingested_at`) so a churn model trained on "features as of day N" never leaks future information from late-arriving events. Late data (arriving > 24h after `event_time`) triggers backfill correction jobs that recompute affected feature partitions, versioned so in-flight training runs aren't silently mutated.
- **Feature freshness SLA**: online features refreshed within the same 2s hot-tier SLA; offline features refreshed on the 15-min cold-tier landing cadence, with a daily full-recompute reconciliation job.

## 16. Vector Database

**N/A for this system.** The telemetry platform transports and stores structured/semi-structured events (typed payloads, numeric/categorical fields) — there are no embeddings generated or queried in the ingestion/storage path itself. Titles that need similarity search (e.g., matchmaking-by-playstyle-embedding, content recommendation) consume this platform's event streams as *input* to their own embedding pipelines and vector stores, which are out of scope here (see Feature Store / Recommendation System chapters).

## 17. Embedding Pipelines

**N/A for this system**, for the same reason as §16 — this platform is upstream of any embedding generation. If a downstream consumer (e.g., a playstyle-clustering model) needs embeddings, it subscribes to `enriched.*` topics or reads the offline feature store and runs its own embedding pipeline; embedding computation is not a responsibility of the telemetry platform to keep this system's blast radius and on-call surface bounded to transport/storage/serving of raw and aggregated events.

## 18. Inference Pipelines

End-to-end request lifecycle for the one true "inference" this system performs inline — real-time anti-cheat scoring:

```
Client emits player_action event
        │
        ▼
Edge Gateway (authN, ~2ms)
        │
        ▼
Ingest Service (schema validate, idempotency check, ~5ms)
        │
        ▼
Kafka produce (raw.title.player_action) — ack after ISR write, ~10-20ms
        │
        ▼
Flink job consumes (micro-batch window, 200-500 events, ~200ms window trigger)
        │
        ├─► Enrichment (geo, session join via Redis lookup, ~3ms)
        │
        ├─► Anti-cheat scorer (GBT model, in-process, ~1ms/event, batched)
        │         │
        │         ▼
        │   score > threshold? ─── yes ──► emit to `flags.anti_cheat` topic ──► real-time review queue
        │         │
        │         no
        │         ▼
        └─► Emit to `enriched.*` topic + windowed aggregate update (Druid write, ~10ms)
                    │
                    ▼
        Query API serves aggregate to LiveOps dashboard / matchmaking service (< 50ms read)

Total ingest→hot-tier-visible p99 budget: ~250-300ms typical, bounded by 2s SLA including window-trigger worst case.
```

## 19. Training Pipelines

The platform itself trains only its two small inline models (anti-cheat GBT, anomaly isolation forest); it also *produces* the training data other systems consume.

- **Data prep**: batch Spark job reads Iceberg cold tier, joins `player_action` + `flags.anti_cheat` labeled outcomes (human-reviewed ground truth) → produces labeled training set, feature-parity enforced via shared feature-definition library (§15).
- **Training orchestration**: Airflow DAG — extract → feature-join (point-in-time correct) → train (single-node XGBoost, small enough to not need distributed training) → validate (holdout + backtested precision/recall on known cheat cases) → register in model registry (MLflow) → canary deploy into Flink job (see §33).
- **Distributed training**: not required for these lightweight models (< 10M rows, < 5MB model). If this platform later hosts heavier inline models, distributed training would move to a separate training-infra chapter — flagged here as an explicit non-goal to keep this system's scope bounded.
- **Cadence**: full retrain weekly; the retrain job itself runs on shared batch compute (not provisioned specifically for this system) to keep this platform's dedicated cost footprint focused on ingest/storage/serving.

## 20. Retraining Strategy

| Trigger type | Condition | Action |
|---|---|---|
| Scheduled | Weekly | Full retrain on trailing 90 days of labeled data |
| Drift-triggered | Feature drift PSI > 0.2 on top-10 features (see §21) | Ad-hoc retrain within 24h |
| Performance-triggered | Anti-cheat precision drops > 5 points vs. baseline (measured via human-review sampling) | Immediate retrain + rollback to previous model version pending investigation |
| Schema-change-triggered | New/changed event schema affecting model input fields | Re-validate feature pipeline compatibility before next scheduled retrain; block deploy if breaking |
| New title onboarding | New game added to platform with different cheat patterns | Cold-start with cross-title baseline model, dedicated retrain after 30 days of title-specific labeled data |

## 21. Drift Detection

| Drift type | Metric | Threshold | Action |
|---|---|---|---|
| Data drift (event volume/shape) | Per-event-type volume vs. 7-day rolling baseline (z-score) | \|z\| > 3 for 15 min | Page on-call; check for client bug, bot traffic, or upstream schema break |
| Data drift (feature distribution) | Population Stability Index (PSI) on key anti-cheat features (kill rate, headshot %, reaction time) | PSI > 0.2 = investigate, > 0.3 = auto-flag for retrain | Feed into retraining trigger (§20) |
| Concept drift (label shift) | Rolling precision/recall of anti-cheat model against human-reviewed sample (weekly audit set) | Precision drop > 5pts absolute | Retrain + shadow-deploy challenger before promoting |
| Schema drift | Schema Registry compatibility check failures | Any BACKWARD-incompatible change proposed | Hard block at registry (fail the schema registration, not silent) |
| Pipeline drift (null/missing rate) | % events with null/default enrichment fields (e.g., geo-IP lookup failure rate) | > 1% for 10 min | Page data-eng on-call, check enrichment dependency health |

## 22. Monitoring

| Layer | Metrics |
|---|---|
| Infra | Kafka broker CPU/disk/network, partition consumer lag (per consumer group), Flink checkpoint duration/failure rate, Redis hit rate & memory, S3 PUT/GET error rate |
| Pipeline health | Events/sec in vs. out per stage, DLQ rate, schema-validation-failure rate, end-to-end latency percentiles (ingest → hot, ingest → cold) |
| Model quality | Anti-cheat precision/recall (weekly human-audit sample), score distribution drift, false-positive ban-appeal rate |
| Business | Events/day per title, active event schemas per title, cost per million events, onboarding lead time for new event types |
| SLA | % of events meeting 2s hot-tier freshness SLA, % meeting 15-min cold-tier SLA, availability of ingest path (synthetic canary events every 10s per region) |

## 23. Alerting

| Alert | Condition | Severity | Routing |
|---|---|---|---|
| Ingest availability | Synthetic canary event fails 3 consecutive checks (30s) in any region | P1 | Page primary on-call (platform team), auto-page secondary if unacked in 5 min |
| Consumer lag breach | Any consumer group lag > 2x the freshness SLA window sustained 5 min | P1 | Page data-eng on-call |
| DLQ spike | DLQ rate > 0.01% of traffic for 5 min | P2 | Slack + ticket, page if > 0.1% |
| Broker disk pressure | Any broker > 85% disk utilization | P2 | Page infra on-call |
| Schema registry compatibility break attempt | Any rejected registration | P3 | Slack notify submitting team, no page |
| Model precision degradation | Weekly audit shows > 5pt precision drop | P2 | Notify ML on-call, triggers retrain workflow |
| Cost anomaly | Daily spend > 20% above 7-day rolling average | P3 | Slack notify platform + finance partner |

On-call routing: tiered — L1 platform/infra on-call (Kafka, Flink, storage), L2 ML on-call (model quality, drift), L3 data-eng on-call (schema, pipeline correctness) — paged based on alert category tag.

## 24. Logging

- **Structured logging**: JSON logs at every hop (Gateway, Ingest Service, Flink) with correlation fields: `trace_id`, `event_id`, `title_id`, `stage`. Emitted to a centralized log pipeline (itself a smaller instance of this same architecture — logs-as-events).
- **PII handling**: `player_id` is pseudonymized at the log layer (never raw account ID in logs); raw PII (IP address, device ID) is redacted from logs entirely, retained only in the primary event payload under stricter access control and shorter retention (90-day auto-purge per §5).
- **Retention**: operational logs 30 days (hot, searchable via OpenSearch); archived to cold storage (S3, compressed) for 1 year for incident forensics; event-payload PII fields purged/pseudonymized at 90 days regardless of log retention tier, per privacy policy.
- **Audit logging**: separate immutable audit trail (who registered/changed a schema, who triggered a replay, who accessed raw PII fields) retained 3 years for compliance.

## 25. Security

**Threat model specific to this system**:
- Spoofed/malicious client telemetry (cheaters sending fabricated events to poison anti-cheat training data or hide cheat signals).
- Compromised game-server credentials used to inject fraudulent economy events (currency duplication via replayed transactions).
- PII exfiltration risk from cold-tier data lake (broad analyst/BI access to raw player data).
- DLQ/replay endpoints as a privilege-escalation vector (replaying old events to trigger stale side effects).
- Denial-of-service via client flood (either malicious or buggy client build) exhausting ingest capacity.

**Mitigations**:
- Mutual TLS + device/build attestation for client SDK connections where platform supports it (console SDKs); server-side plausibility checks (physically impossible action rates/positions) flag suspicious event streams for anti-cheat review rather than trusting client data blindly.
- Idempotency-key ledger (DynamoDB, §12) prevents economy event replay/duplication.
- Data lake access via row/column-level policies (Lake Formation or equivalent) — PII columns masked by default, unmasked access requires elevated, time-boxed, audited grants.
- Replay API (`/v2/replay`) restricted to internal RBAC role, all invocations audit-logged, output routed to isolated topics (never directly overwrites production hot tier).
- Encryption at rest (S3 SSE-KMS, Kafka disk encryption) and in transit (TLS 1.3 everywhere).

## 26. Authentication

- **Client → Edge Gateway**: per-title API key + device/build attestation token (console platforms provide first-party attestation; PC/mobile use a lighter integrity check), short-lived signed JWT issued after initial handshake.
- **Service → Service** (Ingest Service → Kafka, Flink → Redis/Druid, Query API → downstream): mTLS via service mesh (Istio/Linkerd), SPIFFE identities per service.
- **End-user (dashboard/BI) → Query API**: OAuth2 (corporate SSO), scoped to title/role via RBAC (LiveOps analyst vs. data-eng vs. anti-cheat reviewer scopes differ).
- **Internal replay/admin APIs**: OAuth2 client-credentials + RBAC role check + audit log entry mandatory.

## 27. Rate Limiting

- **Algorithm**: token bucket per `(title_id, api_key)` pair at the Edge Gateway — allows short bursts (season launch spikes) while capping sustained abuse.
- **Per-client limits**: default 500 events/sec per device/session (well above legitimate single-player telemetry rate; catches buggy/malicious clients), configurable per title.
- **Per-tenant (title) limits**: soft cap at provisioned capacity share (e.g., a mid-size title gets a 50K events/sec ceiling before triggering priority-based shedding), hard cap with graceful 429 + `Retry-After` beyond that.
- **Priority shedding**: under global backpressure, low-priority event types (cosmetic telemetry, movement heartbeats) are throttled before high-priority types (economy, anti-cheat, crash reports) — implemented as a priority queue at the Ingest Service, not a blind global limiter.

## 28. Autoscaling

- **Ingest Service / Edge Gateway**: HPA on CPU (target 60%) + custom metric on request queue depth; scales 20-400 pods per region.
- **Flink task managers**: KEDA-based autoscaling on Kafka consumer lag metric — scale out when lag > 30s-worth of events, scale in after lag drains and stays low for 10 min (avoid flapping).
- **Kafka brokers**: not autoscaled dynamically (stateful, rebalance-expensive); capacity-planned quarterly with headroom, burst absorbed via tiered storage + backpressure controls rather than broker autoscaling.
- **Hot-tier OLAP (Druid) query nodes**: HPA on query QPS + p99 latency SLO breach.
- **Cold-tier (S3)**: no scaling needed (object storage is inherently elastic); Spark ETL clusters autoscale via EMR/Databricks managed autoscaling on job queue depth.

## 29. Cost Optimization

- **Spot instances**: Flink task managers and Spark ETL workers run on spot/preemptible capacity (stateless-recoverable via checkpointing) — ~60-70% cost reduction vs. on-demand for these tiers.
- **Tiered storage**: Kafka tiered storage (hot on broker NVMe, warm on S3) avoids over-provisioning broker disk for the full 7-day hot retention.
- **Data lake lifecycle policies**: S3 Intelligent-Tiering / explicit lifecycle to Glacier Instant Retrieval after 90 days, Glacier Deep Archive after 1 year for compliance-only retention.
- **Compression**: protobuf + zstd on wire (vs. JSON) cuts payload size ~4-6x, directly reducing both network and storage cost.
- **Sampling for low-value telemetry**: cosmetic/movement-heartbeat events sampled at ingestion (e.g., 1-in-10) rather than fully retained, with sampling rate stored as metadata for unbiased downstream reconstruction.
- **Right-sizing partitions**: over-partitioned topics waste broker overhead; partition count tuned to actual throughput per event type rather than one-size-fits-all 300.
- **Reserved capacity**: baseline (steady-state) Kafka/compute capacity covered by reserved instances/savings plans; only burst headroom relies on on-demand/spot.

## 30. Disaster Recovery

| Target | Value |
|---|---|
| RTO (ingest path) | 15 minutes (regional failover) |
| RPO (hot tier) | Near-zero — Kafka replication factor 3 across AZs; cross-region async replication (MirrorMaker2) with < 60s lag target |
| RPO (cold tier) | Zero — S3 cross-region replication (CRR), versioned buckets |
| Backup strategy | Cold tier is itself the durable backup (11 nines); hot tier backed by cross-AZ replication + periodic snapshot of Druid segments to S3 |
| DR drill cadence | Quarterly regional failover game-day; validates MirrorMaker2 failover + DNS/traffic cutover runbook |
| Client-side resilience | SDK local buffering (up to 5 min) absorbs short outages without data loss, degrades gracefully beyond that (prioritized drop per §27) |

## 31. Multi-Region Deployment

- **Topology**: active-active across 3 regions (US-East, EU-West, AP-Southeast) — each region has a full ingest + hot-tier stack; clients route to nearest region via GeoDNS/Anycast for latency.
- **Data replication**: Kafka MirrorMaker2 replicates raw topics cross-region asynchronously (for global aggregate views and DR); cold tier uses S3 CRR into a single logical global lake (partitioned by region + title) for unified BI/ML training access.
- **Consistency tradeoff**: per-region hot-tier aggregates are region-local and authoritative for that region's real-time consumers (anti-cheat, LiveOps) — no cross-region synchronous consistency requirement, since a player's session is pinned to one region. Global rollups (e.g., worldwide season leaderboard) are computed from the cold-tier unified lake with expected 15-min lag, not from real-time cross-region joins.

```
        US-East Region                EU-West Region              AP-SE Region
   ┌─────────────────────┐       ┌─────────────────────┐     ┌─────────────────────┐
   │ Gateway → Ingest →   │       │ Gateway → Ingest →   │     │ Gateway → Ingest →   │
   │ Kafka → Flink → Hot  │◄─────►│ Kafka → Flink → Hot  │◄───►│ Kafka → Flink → Hot  │
   │ (region-local, auth.)│ MM2   │ (region-local, auth.)│ MM2 │ (region-local, auth.)│
   └──────────┬───────────┘       └──────────┬───────────┘     └──────────┬───────────┘
              │  S3 CRR                       │ S3 CRR                     │ S3 CRR
              └───────────────┬───────────────┴────────────────────────────┘
                              ▼
                 Global Unified Cold Lake (S3 + Iceberg)
                 — single source for global BI/ML training —
```

- **Routing**: GeoDNS + latency-based Route53 policy; failover routing shifts a region's client traffic to nearest healthy region within RTO target if a region degrades.

## 32. Blue/Green Deployment

Applied primarily to the **stateless tiers**: Edge Gateway, Ingest Service, Query API.
- Deploy green stack alongside blue with identical Kafka/Redis/Druid backends (shared state, no data migration needed since these tiers are stateless).
- Shift traffic via load balancer weight from blue → green after smoke tests (schema-validation golden-event replay, synthetic canary traffic) pass on green.
- Kafka/Flink jobs use a **rolling** strategy instead (not blue/green) since Flink jobs carry checkpointed state — a true blue/green would require dual-running and reconciling stateful aggregation, which isn't worth the complexity; instead Flink upgrades use savepoint-based restart (drain → snapshot → redeploy new job version from savepoint).
- Rollback: instant traffic-weight revert to blue stack if green fails post-cutover health checks (< 2 min detection + revert).

## 33. Canary Deployment

Applies to: Flink job logic changes (enrichment rules, anti-cheat model version), Ingest Service schema-validation logic, Query API.

- **Traffic split**: new Flink job version (with new anti-cheat model) deployed as a shadow consumer on a **duplicate consumer group**, reading the same topics in parallel, writing scores to a shadow topic (`flags.anti_cheat.canary`) — zero production impact.
- Compare shadow scores vs. production scores on live traffic for 24-48h (precision/recall proxy via agreement rate + spot-check human review).
- If shadow model's flag-agreement with production is within expected bounds and no regression in false-positive rate, promote: cut over consumer group to canary logic for 5% of titles first, then 25%, 50%, 100%, monitoring DLQ rate/latency/model precision at each step (health-check gates).
- **Health-check gates specifically for this system**: DLQ rate must stay < 0.01%, p99 ingest-to-hot latency must stay < 2s, anti-cheat precision (sampled) must not regress > 2pts, before advancing to next traffic percentage.

## 34. Rollback Strategy

- **Automated triggers**: any canary health-check gate breach (§33) auto-halts traffic-percentage advancement and reverts to prior consumer group/model version within one Flink checkpoint interval.
- **Schema rollback**: Schema Registry enforces compatibility at registration time, so a "bad" schema can't break existing consumers — rollback here means simply not promoting the new schema version as default; old producers/consumers keep working against the last-good version.
- **Flink job rollback**: restart from the last-known-good savepoint (pre-deploy snapshot), replaying only the delta since that checkpoint — bounded reprocessing window (minutes, not hours).
- **Ingest Service/Query API rollback**: standard blue/green traffic-weight revert (§32), typically < 2 min to fully reverted state.
- **Data correction rollback**: if a bad enrichment/aggregation logic wrote incorrect data to hot/cold tier before detection, trigger a backfill/replay job (§9 `/v2/replay`) to recompute affected partitions from raw topics (raw events are immutable source of truth, so derived-data rollback is always a replay, never a raw-data mutation).

## 35. Observability

- **Tracing**: every event carries a `trace_id` from client emission through Gateway → Ingest → Kafka → Flink → hot/cold sinks (OpenTelemetry, trace context propagated via Kafka headers). Enables answering "why did this specific event take 4s to reach the dashboard?" by walking the trace across hops.
- **Metrics**: Prometheus/Cortex for infra + pipeline metrics (§22), federated across 3 regions into a global view; RED metrics (rate, errors, duration) per component.
- **Logs**: correlated to traces via shared `trace_id`/`event_id` fields (§24), searchable in OpenSearch, linked bidirectionally from trace UI (Jaeger/Tempo) to log search.
- **Correlation in practice**: an on-call engineer investigating a latency-SLA breach starts at the metrics dashboard (which stage's p99 spiked) → jumps to traces for representative slow events in that stage → jumps to logs for those specific `trace_id`s to see error details/enrichment failures — three pillars queried as one workflow, not three separate silos.

## 36. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ingest-service
  namespace: telemetry
spec:
  replicas: 40
  selector:
    matchLabels: { app: ingest-service }
  template:
    metadata:
      labels: { app: ingest-service }
    spec:
      containers:
        - name: ingest-service
          image: ea-registry/telemetry/ingest-service:1.42.0
          resources:
            requests: { cpu: "2", memory: "2Gi" }
            limits: { cpu: "4", memory: "4Gi" }
          ports: [{ containerPort: 8080 }]
          env:
            - name: SCHEMA_REGISTRY_URL
              value: "http://schema-registry.telemetry.svc.cluster.local:8081"
            - name: KAFKA_BOOTSTRAP
              valueFrom: { secretKeyRef: { name: kafka-creds, key: bootstrap } }
          livenessProbe:
            httpGet: { path: /healthz, port: 8080 }
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata: { name: ingest-service, namespace: telemetry }
spec:
  selector: { app: ingest-service }
  ports: [{ port: 443, targetPort: 8080 }]
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: ingest-service-hpa, namespace: telemetry }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: ingest-service }
  minReplicas: 20
  maxReplicas: 400
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 60 } }
    - type: Pods
      pods:
        metric: { name: kafka_producer_queue_depth }
        target: { type: AverageValue, averageValue: "1000" }
```

## 37. Terraform Infrastructure

```hcl
resource "aws_msk_cluster" "telemetry" {
  cluster_name           = "telemetry-kafka-${var.region}"
  kafka_version           = "3.6.0"
  number_of_broker_nodes  = 16

  broker_node_group_info {
    instance_type   = "kafka.m5.4xlarge"
    client_subnets  = var.private_subnet_ids
    security_groups = [aws_security_group.msk.id]

    storage_info {
      ebs_storage_info { volume_size = 2000 } # GB per broker, tiered storage offloads rest
    }
  }

  encryption_info {
    encryption_in_transit { client_broker = "TLS", in_cluster = true }
    encryption_at_rest_kms_key_arn = aws_kms_key.telemetry.arn
  }

  configuration_info {
    arn      = aws_msk_configuration.telemetry.arn
    revision = 1
  }
}

resource "aws_s3_bucket" "cold_lake" {
  bucket = "ea-telemetry-cold-lake-${var.region}"
}

resource "aws_s3_bucket_lifecycle_configuration" "cold_lake_tiering" {
  bucket = aws_s3_bucket.cold_lake.id
  rule {
    id     = "tier-down-old-telemetry"
    status = "Enabled"
    transition { days = 90  storage_class = "GLACIER_IR" }
    transition { days = 365 storage_class = "DEEP_ARCHIVE" }
  }
}

resource "aws_s3_bucket_replication_configuration" "cross_region" {
  bucket = aws_s3_bucket.cold_lake.id
  role   = aws_iam_role.replication.arn
  rule {
    id     = "global-replicate"
    status = "Enabled"
    destination { bucket = var.dr_region_bucket_arn storage_class = "STANDARD_IA" }
  }
}
```

## 38. Why This Architecture

- Kafka/Kinesis as the durable buffer decouples highly variable client-ingest rates from downstream processing rates — essential given 10x burst tolerance requirement (season launches) without over-provisioning every downstream consumer for peak.
- Separate hot (Druid/Redis) and cold (S3/Iceberg) tiers match genuinely different access patterns: sub-second point/window queries vs. petabyte-scale scan-heavy analytics — a single store optimized for one would badly serve the other.
- Schema Registry with enforced compatibility rules is the only way to let 20+ independent game teams evolve ~150 event schemas without a central team becoming a bottleneck or breaking consumers.
- Inline lightweight scoring (anti-cheat) avoids an extra network hop that would blow the 2s freshness SLA; heavier ML stays out of this system's blast radius by design (§14/§16/§17).
- Multi-region active-active with region-local hot tiers respects the reality that a player's session/match doesn't need cross-region consistency, while cold-tier global replication serves the genuinely global consumers (BI, training).

## 39. Alternative Architectures

| Alternative | Description | Why Rejected / When Preferred |
|---|---|---|
| Single global Kafka cluster (no regional sharding) | One cluster, all regions produce/consume cross-region | Rejected: cross-region producer latency directly hits ingest p99 SLA; only viable for a much smaller-scale, single-region title |
| Lambda architecture (separate batch + speed layers with full logic duplication) | Fully independent batch recomputation path in addition to streaming | Rejected in favor of Kappa-style (single streaming pipeline, batch reprocessing via replay of the same log) — reduces logic duplication/drift between batch and streaming aggregation code; would reconsider Lambda only if streaming reprocessing cost at 3-year retention became prohibitive |
| Managed serverless ingestion (Kinesis Firehose + Lambda only, no Flink) | Simpler ops, less infra to run | Preferred for a smaller title/lower event-type complexity (e.g., a single mobile title with < 10K events/sec) — rejected here because windowed aggregation, dedupe, and inline scoring at this scale/complexity need Flink's stateful processing model, not Lambda's stateless-per-invocation model |
| Direct client-to-datalake writes (skip streaming tier entirely, batch upload only) | Clients batch-upload files periodically (hourly) directly to S3 | Rejected: fails the < 2s anti-cheat/LiveOps freshness requirement entirely; would only suit a pure offline-analytics-only scope (explicitly out of scope per FR9/FR4) |

## 40. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| At-least-once transport + idempotency-key exactly-once for critical events | Simple, resilient default; strict guarantee only where it matters | Non-critical event types can still double-count in rare edge cases (acceptable per NFR) |
| Kappa architecture (single streaming pipeline + replay) over Lambda | Less logic duplication, single source of truth | Backfill/replay at 3-year cold-tier scale is compute-expensive if triggered broadly |
| Inline lightweight models vs. dedicated serving cluster | Meets 2s SLA, no extra hop | Limits model complexity/size that can run inline; heavier models must live elsewhere |
| Region-local hot tier (no cross-region sync consistency) | Meets latency SLA, avoids CAP-theorem pain | Global real-time rollups (worldwide leaderboards) aren't truly real-time — 15 min lag via cold lake |
| Tiered Kafka storage (hot broker + S3 warm) | Big broker-disk cost savings | Slightly higher read latency for "recent but not hottest" data (rarely hit in practice) |
| Schema Registry hard-blocking incompatible changes | Prevents consumer breakage at the source | Adds friction/lead time for game teams needing a "breaking" change (requires new event_type/version instead) |

## 41. Failure Modes

| Scenario | Impact | Mitigation |
|---|---|---|
| Kafka broker AZ outage | Partition leader elections, brief producer stalls | RF=3 across AZs, min.insync.replicas=2, automatic leader re-election, client retry with backoff |
| Flink job crash-loop (bad enrichment code deploy) | Consumer lag grows, hot-tier freshness SLA breach | Canary/shadow deploy (§33) catches before full rollout; savepoint restart on crash; alert on lag breach |
| Schema Registry outage | New event registrations blocked; existing validated schemas still cached client-side, ingest continues | Registry is HA 3-node; client-side schema cache means brief registry outage doesn't stop ingest of already-known event types |
| Regional network partition | Region isolated from MirrorMaker2 replication target | Region continues operating fully (active-active, region-local authoritative hot tier); replication catches up post-partition, cold-tier global view has temporary gap flagged in metadata |
| Massive burst beyond 10x provisioned headroom (viral event, esports final) | Backpressure triggers priority shedding, low-priority telemetry dropped | Graceful degradation per §12/§27 — economy/anti-cheat events protected, cosmetic telemetry sampled down first |
| Poison-pill malformed event (producer bug) | Could crash naive consumer / block partition processing | DLQ routing isolates bad events per-message, not per-partition; Flink job continues past them |
| DynamoDB idempotency-ledger throttling | Exactly-once economy event path could fail-open to at-least-once temporarily | Provisioned autoscaling + on-demand capacity mode; fail-safe is to log + alert rather than silently drop the idempotency check |

## 42. Scaling Bottlenecks

**At 10x scale (30M events/sec sustained)**:
- Kafka partition count (currently 300/topic) becomes the ceiling on consumer parallelism — would need re-partitioning (operationally disruptive) or moving to per-title topic sharding at a finer grain.
- Hot-tier OLAP (Druid) ingestion rate per historical node becomes a bottleneck; requires horizontal segment sharding redesign, not just adding nodes.
- Redis dedupe filter memory footprint (Bloom filter false-positive rate degrades as key cardinality grows) needs re-sizing or moving to a probabilistic structure with better scaling (e.g., Cuckoo filter) or sharded Redis Cluster expansion.

**At 100x scale (300M events/sec)**:
- Single-region active-active model breaks down — would need finer geo-sharding (per-metro edge ingestion, not just per-continent region) to keep client RTT low and avoid backbone network saturation.
- Schema Registry, while not throughput-bound today, becomes a coordination bottleneck if event-type count grows proportionally (thousands of titles) — would need federated/sharded registries per business unit.
- Cold-tier ETL/Spark batch jobs for retraining/BI at 26.5 PB × 100 ≈ 2.65 exabytes/3yr retention would force much more aggressive sampling/summarization strategies rather than retaining raw events at that scale — likely requires tiered raw-event sampling (retain 100% for N days, then downsample) baked into policy, not just storage-class tiering.

## 43. Latency Bottlenecks

**p50/p99 budget breakdown (ingest → hot-tier-visible, 2s SLA)**:

| Stage | p50 | p99 |
|---|---|---|
| Client → Edge Gateway (network + TLS) | 20ms | 80ms |
| Edge Gateway → Ingest Service (authN, validate) | 5ms | 25ms |
| Ingest Service → Kafka produce (ack after ISR write) | 10ms | 60ms |
| Kafka → Flink consume lag (steady state) | 20ms | 400ms (dominated by micro-batch window trigger) |
| Flink enrichment + scoring | 5ms | 30ms |
| Flink → Hot store write (Druid/Redis) | 10ms | 50ms |
| **Total** | **~70ms** | **~645ms typical, up to 2s under backpressure** |

- **Biggest p99 contributor**: Flink micro-batch window trigger latency — during consumer-lag events (post-deploy catch-up, burst absorption) this is where the SLA budget gets consumed; this is the first place to look when investigating a freshness SLA breach.
- Cold-tier 15-min SLA is dominated by sink-connector batch/flush interval (deliberately batched for S3 write efficiency, not a per-event latency concern).

## 44. Cost Bottlenecks

- **Cross-AZ/cross-region data transfer**: Kafka replication (RF=3, cross-AZ) and MirrorMaker2 cross-region replication are recurring, volume-proportional costs that scale linearly with ingest volume — the single largest "hidden" cost line beyond raw compute/storage.
- **Broker compute for replication overhead**: 3x replication means every byte ingested is effectively 3x'd in cluster network/disk I/o before even reaching sinks.
- **Cold-tier storage at 3-year retention**: even compressed (~5.3 PB effective), this is a steady, large monthly bill (~$111K/month estimated in §6) that only grows as more titles onboard — the biggest lever here is retention-policy tightening (do all event types truly need 3 years?) and more aggressive sampling of low-value event types before they ever reach cold storage.
- **Over-provisioned partition/broker headroom for burst tolerance**: provisioning for 10x burst (season launches) year-round, rather than dynamically, means steady-state days pay for capacity that's idle 95% of the time — mitigated partially by spot-based Flink/Spark tiers, but Kafka brokers themselves (stateful) can't easily follow the same pattern.

## 45. Interview Follow-Up Questions

1. How would you change the design if the platform needed to guarantee exactly-once semantics for *all* event types, not just economy events?
2. Walk me through what breaks first if one game title suddenly represents 80% of total platform traffic (a breakout hit).
3. How do you prevent a single misbehaving title's schema changes from destabilizing the shared platform?
4. What's your strategy if the anti-cheat model's false-positive rate spikes and starts banning legitimate players — how fast can you detect and roll back?
5. How would you support a title that needs sub-100ms (not sub-2s) freshness for a real-time competitive feature?
6. How do you reconcile "replay from raw events" as your correction mechanism with a 3-year cold-tier retention and the cost of reprocessing at that scale?
7. What changes in your multi-region strategy if regulatory requirements (e.g., data residency for a specific country) mandate that certain players' data never leaves their home region?
8. How do you test schema evolution changes before they hit production across 20+ independently-deploying game teams?
9. What's your approach to capacity planning for an unpredictable viral spike (a title suddenly trending) versus a known scheduled event (season launch)?
10. How would the design differ if most consumers only needed hourly-batch freshness — what would you simplify or remove?

## 46. Ideal Answers

1. **Exactly-once everywhere**: Move idempotency-key deduplication (currently economy-only) to the Ingest Service for all event types, backed by a higher-throughput dedup store (sharded Redis with Bloom filter + DynamoDB fallback for exact checks) rather than DynamoDB alone at full volume — accept the added latency/cost (~5-10ms, notable DynamoDB WCU cost at 3M events/sec) as a deliberate tradeoff, and use Kafka transactional producers + `read_committed` consumers end-to-end to eliminate duplicate-on-retry at the transport layer too.

2. **80% single-title breakout**: Partition-level hot-spotting on that title's Kafka partitions and Druid segments becomes the first bottleneck — mitigate by dedicating partition/broker capacity per top-tier title (tenant isolation) rather than uniform sharing, and pre-provision a "breakout tier" fast-path (dedicated topic set, dedicated Flink job, dedicated hot-tier cluster) that a title can be promoted into within hours, not weeks.

3. **Isolating a misbehaving title**: Per-title rate limits (§27) and per-title Kafka topic/partition isolation already contain blast radius at the transport layer; at the schema layer, Schema Registry's hard compatibility gate prevents a bad schema push from breaking other consumers since schemas are scoped per event_type (already namespaced per title), so worst case impacts only that title's own consumers, not the shared platform.

4. **Anti-cheat false-positive spike**: Detection via the weekly human-audit precision metric (§21) is too slow for this — add a faster proxy signal (ban-appeal rate spike, sudden shift in flag-volume per title vs. baseline) as a real-time alert; rollback mechanics are already fast (savepoint restart to prior model version, §34, sub-checkpoint-interval) — the real fix is tightening the detection latency, likely via a lightweight real-time agreement-rate check between current and previous model versions run in shadow continuously, not just during canary windows.

5. **Sub-100ms freshness for one title's feature**: Bypass the Flink micro-batch window (the dominant p99 contributor per §43) for that specific event type/title — use per-event (not micro-batched) processing with a dedicated low-latency consumer group and direct write to Redis (skip Druid's segment-commit latency), accepting higher per-event compute cost for that narrow slice of traffic in exchange for the latency win.

6. **Replay cost vs. 3-year retention**: Replay/reprocessing should default to scoped time-ranges and event-types (not full-history replays), and the cost concern is exactly why retention policy (§29/§44) should be tightened per event-type value rather than uniform 3-year retention — high-value event types (economy, match outcomes) justify the replay cost; low-value ones (movement heartbeats) shouldn't be retained raw for 3 years at all, sidestepping the problem at the policy level rather than only the engineering level.

7. **Data residency requirement**: This is where the region-local hot-tier design already pays off — extend it so a residency-constrained region's data is excluded from cross-region MirrorMaker2 replication and from the global unified cold lake (or replicated only in an anonymized/aggregated form), with region-scoped BI/training views instead of global ones for that player population; this is a policy flag per-title/per-region on the replication and ETL jobs, not a structural redesign.

8. **Testing schema evolution across independent teams**: Schema Registry enforces compatibility mechanically (BACKWARD checks) as a hard gate, but pair it with a shared contract-testing suite each team runs in CI — replaying a golden set of historical events against their new schema/consumer code before merge — plus a shadow-consumer canary (§33 pattern) reading real production traffic with the new schema in a sandboxed consumer group before the schema is marked "default" for new producers.

9. **Viral spike vs. scheduled event capacity planning**: Scheduled events get pre-provisioned capacity (broker/partition headroom reserved in advance, verified via load test) since the timing and rough magnitude are known; viral/unpredictable spikes rely on the backpressure/shedding design (§12/§27) as the actual safety net plus fast-reacting autoscaling (KEDA on consumer lag) for the stateless tiers — the key admission is that Kafka broker capacity itself can't react in minutes, so the 10x burst-tolerance NFR (§3) is fundamentally what protects against the unscheduled case, not reactive scaling.

10. **If most consumers only needed hourly batch freshness**: Drop the hot-tier OLAP layer (Druid/Redis aggregates) and inline Flink enrichment entirely — replace with simple Kafka Connect S3 sink + scheduled batch Spark jobs for enrichment/aggregation, cutting both infra cost and operational complexity substantially (no stateful stream processing to operate); this would only be safe if genuinely *no* consumer (anti-cheat, live dashboards) needed sub-minute freshness, which is why this platform explicitly serves the harder multi-SLA case in this chapter.
