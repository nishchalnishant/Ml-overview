# Live Game Analytics

## 1. Problem Framing & Requirement Gathering

Design a real-time analytics platform for a live-service EA title (think Apex Legends / EA FC Ultimate Team scale) that ingests player telemetry, powers live-ops dashboards (DAU, matchmaking health, purchase funnels, crash rates), computes streaming aggregations, stores results in an OLAP layer for ad-hoc slicing, and fires anomaly alerts (e.g., matchmaking queue times spiking, revenue drop, crash-rate spike after a patch) to live-ops/SRE on-call within minutes of occurrence.

- Primary consumers: live-ops engineers, game designers, SRE, anti-cheat/trust-and-safety, exec dashboards.
- Not in scope here: the anti-cheat ML classifier itself, matchmaking algorithm internals, or the ad-bidding pipeline — this chapter covers the telemetry→dashboard→alert spine they all sit on.

## 2. Functional Requirements

- FR1: Ingest structured game events (match_start, match_end, purchase, crash, login, level_up, queue_joined, queue_matched) from client SDKs and game servers.
- FR2: Provide sub-minute-latency streaming aggregates (e.g., concurrent players, matches/min, revenue/min) sliced by title, platform, region, build version.
- FR3: Provide an OLAP query layer for ad-hoc, multi-dimensional slice-and-dice (funnel analysis, cohort retention) with query latency in seconds.
- FR4: Serve pre-built and custom real-time dashboards (Grafana/Superset-style) refreshing every 5-15s.
- FR5: Detect anomalies (statistical + rule-based) on key live-ops metrics and page on-call within 2-5 minutes of onset.
- FR6: Support backfill/replay of historical events for corrected aggregates and new metric definitions.
- FR7: Support per-title, per-environment (prod/staging) isolation and self-service metric definition by game teams.
- FR8: Retain raw event data long enough for post-incident forensic queries (30-90 days hot, longer cold).

## 3. Non-Functional Requirements (latency, availability, throughput, consistency, cost)

| Dimension | Target |
|---|---|
| Ingestion latency (client emit → durable in stream) | p99 < 2s |
| Streaming aggregate freshness (event → dashboard metric updated) | p99 < 30s |
| OLAP ad-hoc query latency | p50 < 1s, p99 < 5s (billion-row scans) |
| Dashboard load time | p95 < 2s |
| Anomaly alert latency (onset → page) | < 5 min |
| Availability of ingestion path | 99.95% (can't drop telemetry during a launch) |
| Availability of dashboard/query path | 99.9% (degraded OLAP tolerated) |
| Throughput (peak, big title launch day) | 3M events/sec sustained, 6M burst |
| Consistency | Eventual for aggregates (streaming); read-committed for OLAP snapshots |
| Durability | No silent data loss on ingestion; at-least-once delivery guaranteed |
| Cost | Infra cost < 1.5% of title's live-ops revenue-supported budget |

## 5. Assumptions

1. Portfolio-wide shared platform, multi-tenant per title, starting with 3 flagship titles onboarded.
2. Peak combined CCU across titles: 8M concurrent players; average telemetry rate 5 events/player/min → ~670K events/sec average, 3M events/sec burst (patch-day/tournament).
3. Average event size (compressed, Avro/Protobuf): 400 bytes.
4. Raw event retention: 90 days hot (queryable), 2 years cold (S3/Glacier-style, replay-only).
5. Telemetry is semi-trusted: game servers authoritative for match/economy events, client-only events (crash, UI) rate-limited and sanity-checked.
6. Anomaly detection scope: univariate/multivariate statistical (seasonal ESD, EWMA) on ~200 curated live-ops KPIs per title, not full unsupervised anomaly ML on raw event stream.
7. Dashboards: ~5,000 internal users (live-ops, design, SRE, exec) across the company, not player-facing.
8. Existing EA infra: Kafka-compatible streaming backbone and Kubernetes/EKS clusters assumed available (build on top, not from scratch).
9. GDPR/CCPA applies; player IDs pseudonymized at ingestion edge.

## 6. Capacity Estimation

**Ingestion throughput**
- Steady state: 8M CCU × 5 events/min / 60s ≈ 667K events/sec.
- Peak (patch launch, esports event): 3× steady = ~2M events/sec sustained, 6M events/sec burst (30s spikes).
- Bytes/sec steady: 667K × 400B ≈ 267 MB/s → peak ~2.4 GB/s.

**Daily volume**
- Events/day steady: 667K × 86,400 ≈ 57.6B events/day.
- Raw bytes/day: 57.6B × 400B ≈ 23 TB/day (pre-compression on wire; ~7-8 TB/day after columnar compression ~3:1 in OLAP store).

**Storage**
- Hot OLAP (90 days): 7.5 TB/day × 90 ≈ 675 TB compressed columnar.
- Cold archive (2 years): 23 TB/day raw × 730 days × (compressed ~0.3 factor) ≈ ~5 PB compressed on object storage.
- Streaming layer (Kafka retention, 7 days for replay): 23 TB/day × 7 ≈ 161 TB across brokers (×3 replication ≈ 483 TB provisioned).

**Kafka/streaming cluster sizing**
- Target broker throughput ~80 MB/s/broker sustained write incl. replication overhead.
- Peak ingest 2.4 GB/s × replication factor 3 (write amplification for followers counted separately from client-perceived) → client-facing target ~2.4 GB/s / 80 MB/s ≈ 30 brokers minimum; provision 45 brokers (1.5× headroom) across 3 AZs.

**Stream processing (Flink/Kafka Streams) cluster**
- Rule of thumb: 1 vCPU handles ~50K simple events/sec (parse + windowed aggregate).
- Peak 2M events/sec / 50K ≈ 40 vCPUs minimum per stage; with 4 aggregation stages (per-title, per-region, per-platform, global) and 3× overhead for stateful windowing/shuffle: ~480 vCPUs → ~60 nodes (8 vCPU each), autoscale 20-90 nodes.

**OLAP store (ClickHouse/Druid-style) sizing**
- 675 TB hot data, target ~2 TB usable/node (NVMe, replication factor 2 → 4TB raw/node) → ~340 nodes; realistically shard by title+time, ~200-250 nodes at steady 3-title scale, scaling with title onboarding.

**Anomaly detection compute**
- 200 KPIs/title × 3 titles = 600 time series, evaluated every 30s with EWMA/seasonal-ESD: trivially CPU-light, ~4 vCPUs total; not a bottleneck. Model-based (e.g., Prophet-style seasonal models) retrained daily: batch job, ~2 vCPU-hours/day.

**GPU footprint**: none required for core pipeline (statistical anomaly detection is CPU-bound); GPUs only if a learned multivariate anomaly model (e.g., autoencoder over KPI vectors) is added later — estimate 2× A10G for that batch scoring job, non-blocking.

**Dashboard query fleet**: 5,000 users, ~10% concurrently active peak, ~2 queries/user/min → ~17 QPS peak on OLAP query API; trivial, but query cost (bytes scanned) matters more than QPS — cap per-query scan via time-partition pruning.

## 7. High-Level Architecture

```
                         ┌───────────────────────────┐
 Game Clients            │   Edge Ingestion Gateway   │
 (console/PC/mobile) ───▶│  (auth, schema validate,   │
 Game Servers        ───▶│   pseudonymize, rate-limit)│
                         └─────────────┬─────────────┘
                                       │ Protobuf/Avro over gRPC/HTTPS
                                       ▼
                         ┌───────────────────────────┐
                         │   Kafka (event backbone)   │
                         │  topics: match, purchase,  │
                         │  crash, telemetry, login   │
                         └───┬───────────┬───────────┘
                             │           │
              ┌──────────────┘           └───────────────┐
              ▼                                           ▼
   ┌────────────────────┐                     ┌───────────────────────┐
   │ Stream Processing   │                     │  Sink Connectors       │
   │ (Flink) — windowed  │                     │  (Kafka Connect)       │
   │ aggregates, sessions │                     │  → Object Store (raw)  │
   └─────────┬───────────┘                     └───────────┬───────────┘
             │ agg results (per-min/sec)                    │ batch (5 min)
             ▼                                              ▼
   ┌────────────────────┐                       ┌───────────────────────┐
   │  Serving Cache /    │                       │   OLAP Store           │
   │  Time-Series Store  │◀─────ETL/CDC──────────│ (ClickHouse/Druid)     │
   │  (Redis/TSDB)        │                      │  columnar, partitioned │
   └─────────┬───────────┘                       └───────────┬───────────┘
             │                                                │
             ▼                                                ▼
   ┌────────────────────┐                       ┌───────────────────────┐
   │  Anomaly Detection   │                     │  Query API / BFF       │
   │  Service (EWMA,      │                     │  (dashboards, ad-hoc)  │
   │  seasonal-ESD)        │                     └───────────┬───────────┘
   └─────────┬───────────┘                                   │
             │ alerts                                        ▼
             ▼                                     ┌───────────────────┐
   ┌────────────────────┐                          │  Grafana/Superset  │
   │  Alerting/Paging     │                        │  Dashboards          │
   │  (PagerDuty/Slack)   │                        └───────────────────┘
   └────────────────────┘
```

## 8. Low-Level Components

- **Edge Ingestion Gateway**: stateless HTTP/gRPC service; validates schema (Protobuf), authenticates client/server, pseudonymizes player IDs (HMAC), rate-limits abusive clients, writes to Kafka producer. Scaling unit: horizontal pod, scales on request QPS/CPU.
- **Kafka backbone**: partitioned by `title_id + shard(player_id)`; scaling unit: broker count + partitions per topic (start 200 partitions per high-volume topic).
- **Stream Processing (Flink)**: stateful jobs computing tumbling/sliding window aggregates (1s/10s/1min), session windows for match duration, keyed by title/region/platform. Scaling unit: task manager slots, autoscale on consumer lag.
- **Serving Cache (Redis / time-series store)**: holds last N minutes of per-KPI aggregates for dashboard hot-path reads and anomaly detector input. Scaling unit: Redis cluster shards.
- **Sink Connectors**: Kafka Connect jobs batching raw events to object storage (Parquet) every 5 min, partitioned by title/date/hour.
- **OLAP Store**: ClickHouse (or Druid) cluster ingesting from object storage/Kafka directly; columnar, MergeTree-style partitioning. Scaling unit: shard count (data volume) × replica count (query concurrency).
- **Query API / BFF**: translates dashboard/API requests into OLAP SQL, applies row/column-level access control per title, caches frequent queries. Scaling unit: stateless pods on QPS.
- **Anomaly Detection Service**: pulls per-KPI time series from serving cache every 15-30s, runs EWMA/seasonal-ESD/z-score models, evaluates thresholds, emits alert events. Scaling unit: partitioned by KPI-group, horizontal.
- **Alerting/Paging Service**: dedupes, routes by severity/title/team to PagerDuty/Slack/Opsgenie; owns escalation policy state.
- **Dashboard Frontend**: Grafana/Superset backed by Query API; per-title RBAC-scoped dashboards.

## 9. API Design

Versioned REST + gRPC ingestion; REST for query/dashboard BFF.

```
POST /v1/ingest/events
Headers: Authorization: Bearer <service-jwt>, X-Title-Id, X-Schema-Version
Body (Protobuf, gRPC preferred; JSON fallback):
{
  "event_id": "uuid",
  "title_id": "apex-legends",
  "event_type": "match_end",
  "player_id_hash": "sha256...",
  "timestamp": "2026-07-08T10:15:30Z",
  "platform": "ps5",
  "build_version": "3.14.2",
  "payload": { "match_id": "...", "duration_s": 812, "result": "win" }
}
Response: 202 Accepted { "event_id": "uuid", "ingested_at": "..." }
```

```
GET /v2/metrics/{title_id}/timeseries?metric=ccu&granularity=1m&from=...&to=...&region=NA
Response 200:
{
  "metric": "ccu",
  "granularity": "1m",
  "points": [{"ts": "...", "value": 812345}, ...]
}
```

```
POST /v2/query/adhoc
Body: { "title_id": "fc-25", "select": ["region","count(*) as matches"], "from": "match_events",
        "where": "event_ts BETWEEN ... AND ...", "group_by": ["region"] }
Response 200: { "rows": [...], "bytes_scanned": 4200000000, "query_time_ms": 812 }
```

```
GET /v1/alerts?title_id=apex-legends&status=active
Response 200: { "alerts": [{"id":"...", "kpi":"queue_time_p95", "severity":"sev2",
                 "opened_at":"...", "current_value": 45.2, "baseline": 12.1}] }
```

| Endpoint | Method | Purpose | Auth |
|---|---|---|---|
| /v1/ingest/events | POST | Event ingestion (client/server) | Service JWT / mTLS |
| /v2/metrics/{title}/timeseries | GET | Streaming aggregate read | OAuth2 user token, RBAC |
| /v2/query/adhoc | POST | OLAP ad-hoc query | OAuth2 user token, RBAC |
| /v1/alerts | GET/POST | Alert query & manual ack | OAuth2 user token |
| /v1/metric-definitions | POST | Self-service metric registration | OAuth2, studio-scoped |

Versioning: URI-based (`/v1`, `/v2`), additive schema evolution via Protobuf field numbers, deprecation window 6 months with dual-write/dual-read support.

## 10. Database Design

- **Streaming layer**: Kafka — not a database, but schema registry (Avro/Protobuf) enforces compatibility (BACKWARD) per topic.
- **OLAP store**: ClickHouse chosen over Druid/BigQuery-style for this design — self-hostable, low per-query cost, excellent compression, native Kafka table engine for direct ingestion.
  - Table: `match_events (title_id, event_ts DateTime, region, platform, build_version, player_id_hash, match_id, duration_s, result) ENGINE=MergeTree PARTITION BY (title_id, toYYYYMMDD(event_ts)) ORDER BY (title_id, event_ts, region)`.
  - Partition key: `(title_id, date)` — aligns with retention/drop-partition for TTL and with query pattern (almost all queries scope to a title + time range).
  - Sharding key: `title_id` (avoids cross-shard joins for the dominant single-title query pattern); large titles get dedicated shard groups.
- **Serving cache / TSDB**: Redis (sorted sets keyed by `title:kpi:region`) or a lightweight TSDB (e.g., VictoriaMetrics) for last 24h of per-minute aggregates — read-optimized for dashboards.
- **Metadata store**: PostgreSQL for metric definitions, alert rules, dashboard configs, RBAC — relational, low volume, needs strong consistency.
- **Cold archive**: Parquet on S3-compatible object storage, partitioned `title_id/yyyy/mm/dd/hh`, Glacier-tier after 90 days.

## 11. Caching

- **What's cached**: last-N-minutes KPI aggregates (serving cache), frequent ad-hoc query results (Query API cache keyed by query hash + time-bucket), dashboard panel results (5-15s TTL).
- **Strategy**: cache-aside for ad-hoc query results (compute on miss, TTL 30-60s since data is near-real-time anyway); write-through for streaming aggregates (Flink writes directly to Redis/TSDB as it computes windows — always fresh, no invalidation needed since it's append/overwrite by window key).
- **Invalidation**: time-bucketed keys naturally expire (TTL = 2× window size); on metric redefinition, bump a `metric_version` in the cache key to invalidate old entries without explicit deletes.
- **Anti-pattern avoided**: don't cache raw event queries with long TTL — live-ops needs freshness over hit-rate.

## 12. Queues & Async Processing

- **Ingestion → Kafka**: the queue itself; at-least-once delivery (producer acks=all, idempotent producer enabled to avoid dupes on retry).
- **Sink to object storage**: Kafka Connect S3 sink, exactly-once via connector's transactional offset commits + idempotent file writes (temp-file + atomic rename).
- **Anomaly alert dispatch**: async queue (SQS/Kafka topic `alerts`) between detection service and paging service — decouples detection cadence from paging system availability; DLQ after 3 failed delivery attempts, alert routed to a fallback Slack channel + logged for manual triage.
- **Backfill/replay jobs**: queued as batch jobs (Airflow/Argo Workflows), idempotent by design (upsert by `event_id` in OLAP MergeTree via ReplacingMergeTree or explicit dedup step).
- **Dead-letter handling**: malformed events (schema validation failure) routed to `events.dlq` topic with error metadata; sampled for alerting if DLQ rate > 0.1% of traffic (signals a client SDK bug, often after a game patch).

## 13. Streaming & Event-Driven Architecture

- **Topics**: `telemetry.match`, `telemetry.purchase`, `telemetry.crash`, `telemetry.login`, `telemetry.queue`, each partitioned by `hash(title_id, player_id_hash)` for locality and even load; 200 partitions per high-volume topic.
- **Event schema** (Avro, schema-registry enforced):
```json
{
  "type": "record", "name": "MatchEndEvent",
  "fields": [
    {"name": "event_id", "type": "string"},
    {"name": "title_id", "type": "string"},
    {"name": "player_id_hash", "type": "string"},
    {"name": "event_ts", "type": "long", "logicalType": "timestamp-millis"},
    {"name": "region", "type": "string"},
    {"name": "platform", "type": "string"},
    {"name": "build_version", "type": "string"},
    {"name": "duration_s", "type": "int"},
    {"name": "result", "type": {"type": "enum", "name": "Result", "symbols": ["WIN","LOSS","DRAW"]}}
  ]
}
```
- **Consumer groups**: `flink-aggregator-group` (streaming aggregates), `s3-sink-group` (cold storage), `anomaly-feature-group` (feature extraction for detection), `anti-cheat-group` (out of scope but shares topic) — independent offsets, isolated failure domains.
- **Ordering**: per-partition ordering only (per player/title); cross-partition global ordering not required for aggregate correctness given windowed, commutative aggregation functions (sum/count/approx-distinct).

## 14. Model Serving

- No heavy DL model in the critical real-time path — anomaly detection uses lightweight statistical models (EWMA, seasonal-ESD, z-score on residuals), served as a stateless microservice, not via a model-serving framework like Triton.
- If/when a learned multivariate anomaly model (e.g., isolation forest or small autoencoder over the ~200-dim KPI vector) is added: serve via a lightweight FastAPI/TorchServe container, CPU-only (input is tiny — 200 floats per tick, not raw telemetry), batch scoring every 30s across all titles in one call — batching here is about batching KPI-vectors across titles/regions in a single forward pass, not GPU request batching.
- Hardware: CPU-only fleet (4-8 vCPU pods) suffices; no GPU serving needed for this system's core scope.

## 15. Feature Store

- **Online features** (for anomaly detection): rolling statistics per KPI (mean/stddev over trailing 1h/24h/7d windows, day-of-week/hour-of-day seasonal baselines) computed by Flink and stored in the serving cache/TSDB — read at detection time with <100ms latency.
- **Offline features**: same KPI history in OLAP store, used to retrain seasonal baseline models nightly.
- **Point-in-time correctness**: baseline computation for "what was normal at time T" must only use data available before T (no leakage from future data into a seasonal baseline) — enforced by computing baselines via strictly backward-looking window functions in the nightly batch job, and storing baseline snapshots keyed by `(kpi, as_of_date)` rather than recomputing retroactively.
- This is a much thinner feature store than a personalization/ranking system's — features are aggregate KPI statistics, not per-player ML features; a full Feast-style online/offline store is arguably overkill here, a purpose-built TSDB + nightly batch job suffices.

## 16. Vector Database

N/A for this system's core scope — no embedding similarity search, semantic retrieval, or ANN lookups are required for streaming aggregation, OLAP dashboards, or statistical anomaly alerting. (A vector DB would appear in an adjacent system, e.g., semantic log search or player-support ticket similarity, not here.)

## 17. Embedding Pipelines

N/A — no text/image/user embeddings are produced or consumed in this system. Telemetry events are structured, typed fields, not unstructured content requiring embedding. (Contrast with a churn-prediction or LiveOps-personalization system, where player behavior embeddings would be relevant.)

## 18. Inference Pipelines

"Inference" here = anomaly detection scoring end-to-end (there is no player-facing ML inference in this system).

```
Kafka (telemetry.*) 
   │
   ▼
Flink windowed aggregation (1s/10s/1min tumbling windows)
   │  writes per-KPI point every window close
   ▼
Serving Cache (Redis/TSDB) — latest KPI values + rolling stats
   │
   ▼
Anomaly Detection Service (polls every 15-30s)
   │  1. fetch current value + historical baseline (mean, stddev, seasonal offset)
   │  2. compute z-score / EWMA residual / seasonal-ESD test statistic
   │  3. compare against per-KPI threshold (static or dynamically tuned)
   ▼
Decision: within bounds → no-op | breach → emit AlertEvent
   │
   ▼
Alerts topic → Dedup/Correlation Service (suppress storms, group by title+incident)
   │
   ▼
Paging Service (PagerDuty/Slack) → on-call engineer
```
- End-to-end latency budget: window close (≤60s) + cache write (≤1s) + detection poll (≤30s) + alert dispatch (≤5s) + paging (≤30s) ≈ under 2 min typical, bounded at 5 min p99 (matches FR5/NFR).

## 19. Training Pipelines

- No large-scale model training in the critical path. "Training" here = nightly recomputation of seasonal baselines and threshold tuning.
- Data prep: batch job (Spark/Airflow) reads 90 days of per-KPI time series from OLAP store, computes day-of-week/hour-of-day seasonal decomposition (e.g., STL decomposition) per KPI per title.
- Orchestration: Airflow DAG, nightly, per-title fan-out; output written to metadata store (Postgres) as new baseline version.
- If a learned multivariate anomaly model is introduced: training job on Spark/Ray, CPU cluster (small — input dimensionality ~200-1000 features), retrained weekly, no distributed GPU training required at this system's scale.

## 20. Retraining Strategy

- **Cadence**: seasonal baselines recomputed nightly (captures weekday/weekend, patch-day shifts); threshold sensitivity reviewed weekly by live-ops with a human-in-the-loop tuning UI.
- **Triggers for out-of-cadence retrain**: major game patch/content release (baselines invalidated — traffic pattern shifts), new title onboarding (no history yet — start with static thresholds for first 14 days), sustained false-positive rate > 5% over 3 days (auto-flag for baseline refresh).
- **Validation before promotion**: shadow-run new baseline against last 7 days of historical data, compare alert counts/false-positive rate vs current production baseline before swapping.

## 21. Drift Detection

- **Data drift**: schema drift (new/missing event fields from a client SDK update) monitored via schema-registry compatibility checks + DLQ rate monitoring (alert if DLQ > 0.1% of topic volume in 5 min window).
- **Concept drift** (in the "what's normal" sense for anomaly detection): tracked via rolling comparison of current-week seasonal pattern vs prior-week (KL divergence or simple % delta on hour-of-day profile); if divergence > 25% for 3 consecutive days, flag baseline as stale.
- **Metrics tracked**: false-positive alert rate (target < 5%), false-negative rate (estimated via post-incident retro — did we miss a real live-ops issue?), baseline staleness age (days since last successful recompute).
- **Thresholds**: z-score > 3 or EWMA residual > 4× trailing stddev triggers sev3; sustained breach > 5 min escalates to sev2; > 20% deviation on revenue/CCU escalates to sev1 (pages director-level on-call for launch events).

## 22. Monitoring

- **Infra**: Kafka consumer lag per group, broker disk/network utilization, Flink checkpoint duration & backpressure, ClickHouse query queue depth & merge backlog, Redis memory/eviction rate.
- **Model/detection quality**: alert false-positive rate, alert precision (validated via on-call feedback loop — "was this a real issue?" button), baseline staleness, detection latency (event→alert).
- **Business metrics** (the actual dashboards): CCU, matches/min, revenue/min, crash rate, queue time p50/p95, funnel conversion (store view → purchase), retention D1/D7.
- **Pipeline health**: end-to-end freshness lag (event timestamp vs. latest OLAP-queryable timestamp), ingestion error rate, schema validation failure rate.

## 23. Alerting

| Condition | Threshold | Severity | Route |
|---|---|---|---|
| Crash rate spike | > 3× seasonal baseline, 5 min sustained | Sev1 | Page game team + SRE on-call, immediate |
| Revenue/min drop | > 20% below baseline, 10 min sustained | Sev1 | Page live-ops + finance on-call |
| Matchmaking queue time p95 | > 2× baseline, 5 min | Sev2 | Slack + page backend on-call |
| Kafka consumer lag | > 5 min lag on any critical consumer group | Sev2 | Page data-platform on-call |
| DLQ rate | > 0.1% of topic volume, 5 min | Sev3 | Slack #telemetry-health |
| OLAP query error rate | > 2% over 5 min | Sev3 | Slack #data-platform |
| Baseline staleness | > 10 days without successful recompute | Sev4 | Ticket, weekly review |

- Dedup/correlation: group alerts by `title_id + incident_window` to avoid paging storms when one root cause (e.g., a bad patch) trips 5 KPIs simultaneously — collapse into one incident.

## 24. Logging

- **Structured logging**: JSON logs at every hop (gateway, Flink, OLAP ingest, anomaly service) with correlation `trace_id` propagated from ingestion through to alert.
- **PII handling**: player IDs pseudonymized (HMAC-SHA256 with rotating server-side secret) at the edge gateway before anything touches Kafka; raw player IDs never leave the auth boundary; IP addresses truncated/hashed; right-to-erasure handled via a separate tombstone process that purges by pseudonymized ID across hot store, cold archive, and cache within 30 days.
- **Retention**: application/infra logs 30 days hot (Loki/ELK), 1 year cold (compliance); event data retention per Assumption 4 (90 days hot / 2 years cold), governed separately from app logs.
- **Access control on logs**: query-time PII redaction; only trust-and-safety/legal roles can request re-identification via a separate audited lookup service.

## 25. Security

- **Threat model specifics**: (a) spoofed/replayed client telemetry inflating metrics or masking cheating, (b) DDoS on ingestion gateway during a launch, (c) unauthorized cross-title data access (Title A's live-ops shouldn't see Title B's revenue), (d) PII exposure/re-identification of players.
- **Mitigations**: server-authoritative event signing for economy-sensitive events (purchases, match results) — client can't forge; mTLS between game servers and gateway; per-title API keys + row-level security in OLAP/Query API; rate limiting + anomaly-based bot detection on ingestion; encryption at rest (OLAP store, object storage, Kafka topics with encryption enabled) and in transit (TLS 1.3 everywhere).
- **Data encryption**: KMS-managed keys, per-title key separation for tenant isolation, envelope encryption for archived Parquet files.

## 26. Authentication

- **Service-to-service**: mTLS + short-lived service JWTs (SPIFFE/SPIRE-issued identities) between game servers/clients and ingestion gateway, and between internal services (Flink → cache, Query API → OLAP).
- **End-user (dashboard users)**: OAuth2/OIDC via EA's internal SSO, JWT access tokens scoped with `title_id` claims and role (viewer/editor/admin) enforced at Query API layer via RBAC.
- **Client SDK auth**: per-title, per-build API key embedded in SDK, validated + rate-limited at gateway; rotated per major release.

## 27. Rate Limiting

- **Algorithm**: token bucket per `(title_id, player_id_hash)` at the edge gateway — allows short bursts (e.g., rapid match events) while capping sustained abuse.
- **Limits**: 20 events/sec/player sustained, burst 50; per-title global cap sized to 1.5× provisioned capacity (circuit-breaks to shed load gracefully rather than cascading failure into Kafka).
- **Dashboard/Query API**: per-user token bucket, 10 ad-hoc queries/min, with a separate stricter cap on "expensive" queries (full-scan without time-partition pruning) via query-cost estimation before execution.
- **Enforcement layer**: gateway (Envoy/Istio) rate-limit filter backed by Redis for distributed counting.

## 28. Autoscaling

- **Ingestion gateway**: HPA on CPU (target 60%) + custom metric (requests/sec per pod), min 10 / max 200 pods.
- **Flink stream processing**: KEDA scaler on Kafka consumer lag — scale task managers when lag > 10K messages per partition group sustained 2 min; min 20 / max 90 nodes (matches capacity estimate in §6).
- **OLAP query replicas**: HPA on query queue depth + p99 query latency; scale read replicas independent of ingest/storage nodes.
- **Anomaly detection service**: scales on number of KPI-title pairs assigned (partitioned), low churn — mostly fixed-size with headroom.
- **VPA**: applied to Flink task managers for right-sizing memory (stateful window operators are memory-hungry and vary by title volume).

## 29. Cost Optimization

- Spot/preemptible instances for Flink task managers (stateless-restart-tolerant with checkpointing to durable state backend) and OLAP read-replica fleet — 60-70% cost reduction on compute.
- Tiered storage: hot ClickHouse (NVMe, 90 days) → cold Parquet on S3 Infrequent Access → Glacier after 1 year; lifecycle policies automated.
- Query cost governance: enforce time-partition pruning, block full-table scans over 1B rows without explicit override + cost warning shown to user.
- Approximate algorithms: HyperLogLog for distinct-player counts, t-digest for percentile estimates — avoids expensive exact computation at billion-row scale.
- Downsample old data: raw event granularity for 7 days, 1-min rollups for 90 days, hourly rollups beyond that in cold storage.
- Kafka topic retention tuned per topic criticality (7 days for replay-critical topics, 24h for high-volume low-value telemetry like UI-interaction pings).
- Right-size anomaly detection compute — it's CPU-trivial (§6), avoid over-provisioning GPU infra "just in case."

## 30. Operational Concerns (Deployment, Reliability, Infra)

At SDE2 scope, treat this as a checklist rather than a design exercise: **backups** (automated snapshots of the model registry, feature store, and any stateful service, with a tested restore path), **rollback** (every deploy must be revertible to the last-known-good version — the model registry and CI/CD pipeline should make this a one-command operation), **canary/blue-green rollout** (shift a small percentage of traffic first, watch error rate and key business/model metrics, then ramp), and **basic observability** (dashboards + alerts on latency, error rate, and the top 2-3 model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level infra-architecture concerns — worth knowing they exist, not worth rehearsing the manifests.

## 38. Why This Architecture

- Kafka-centric backbone decouples ingestion rate from downstream processing rate — a slow OLAP sink or a Flink restart never blocks client telemetry acceptance.
- Streaming aggregation (Flink) + separate OLAP store cleanly splits two very different access patterns: sub-minute dashboards (cache-backed, narrow KPI set) vs. ad-hoc multi-dimensional exploration (OLAP, wide schema, higher latency tolerance) — trying to serve both from one system forces bad tradeoffs on either.
- Statistical anomaly detection (not a heavyweight ML model) matches the actual problem: live-ops KPIs are well-behaved, seasonal time series where EWMA/seasonal-ESD is interpretable, cheap, and fast to retrain — an opaque deep model would add latency and reduce trust from on-call engineers who need to understand *why* an alert fired.
- Multi-region active-active ingestion matches EA's globally distributed player base and keeps ingestion latency (NFR: p99 < 2s) achievable without cross-continent hops on the hot path.

## 39. Alternative Architectures

**Alternative A — Lambda architecture (separate batch + speed layers reconciled at query time)**
- Rejected as primary design: adds significant operational complexity (maintaining two codepaths for the same aggregation logic); modern streaming engines (Flink) with exactly-once state + OLAP backfill from cold storage achieve the same correctness (Kappa-style) with one codepath.
- Would be preferred if: the org already has deep batch (Spark) investment and streaming correctness requirements are loose.

**Alternative B — Fully managed cloud-native stack (Kinesis + Kinesis Analytics + Redshift/BigQuery + native alerting)**
- Rejected as primary recommendation here given assumption of existing EA on-prem/hybrid Kubernetes investment and cost sensitivity at multi-PB scale (managed OLAP warehouses bill per-query-bytes-scanned, expensive at this dashboard QPS × scan-volume).
- Would be preferred if: team is small, greenfield, no existing K8s/Kafka investment, and willing to trade cost-at-scale for near-zero operational burden early on.

**Alternative C — Single unified store (e.g., all data in one distributed OLAP system, no separate serving cache/TSDB)**
- Rejected: sub-30s dashboard freshness with heavy OLAP concurrent query load competing with real-time writes creates resource contention; a dedicated fast-path cache/TSDB isolates the latency-critical read path from the flexible-but-heavier ad-hoc query path.
- Would be preferred if: dashboard freshness requirement were relaxed to minutes (not seconds), removing the need for a separate hot path.

| Alternative | When to prefer | Why rejected here |
|---|---|---|
| Lambda architecture | Heavy existing batch investment, loose correctness needs | Doubles maintenance burden vs. Kappa/streaming-only |
| Fully managed cloud stack | Small team, greenfield, no K8s/Kafka investment | Cost at PB-scale query volume, less control over tenancy/latency |
| Single unified OLAP (no cache) | Minutes-level freshness acceptable | Can't hit sub-30s dashboard SLA under concurrent ad-hoc load |

## 40. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Kafka + Flink (Kappa) over Lambda | One codepath, simpler correctness | Requires mature streaming expertise on-call |
| ClickHouse (self-hosted) over managed warehouse | Lower cost at scale, full control | Ops burden (upgrades, shard rebalancing) on data-platform team |
| Statistical anomaly detection over learned ML model | Interpretable, cheap, fast to retrain | Misses complex multivariate anomalies a learned model might catch |
| Active-active ingestion / active-passive OLAP | Low ingest latency globally, simple query consistency | Cross-region aggregation adds MirrorMaker lag to "global" KPIs |
| Approximate algorithms (HLL, t-digest) | Massive cost/latency savings at scale | Small, bounded error in reported cardinalities/percentiles |
| Pseudonymization at edge | Strong privacy posture, simpler compliance | Re-identification for trust-and-safety investigations requires extra lookup hop |

## 41. Failure Modes

- **Kafka broker AZ outage**: mitigated by replication factor 3 across AZs; producers configured `acks=all`; brief partition leader-election latency spike (~seconds), no data loss.
- **Flink job crash mid-window**: checkpointed state (RocksDB backend + durable checkpoint store) allows exact restart from last checkpoint; at-least-once reprocessing of a small window causes brief metric double-count risk, mitigated by idempotent/commutative aggregation design (sums recomputed from checkpoint, not double-added).
- **OLAP ingestion falls behind (merge backlog)**: query latency degrades gracefully (serves slightly stale data via serving cache fallback for dashboards); alerting on merge-backlog depth before it becomes user-visible.
- **Client SDK bug post-patch floods malformed events**: schema validation at gateway routes to DLQ instead of poisoning downstream aggregates; DLQ-rate alert catches it within 5 min; affected title's dashboard shows a data-quality banner.
- **Anomaly detection false-positive storm** (e.g., legitimate traffic spike from a content drop mistaken for anomaly): dedup/correlation service groups alerts; on-call can bulk-acknowledge/suppress with a reason code, feeding back into false-positive-rate monitoring (§21/22).
- **Cross-region MirrorMaker lag spike**: global/portfolio-wide KPIs become stale; region-local dashboards unaffected (they read local Kafka/OLAP directly) — failure is contained to the "global rollup" view only.

## 42. Scaling Bottlenecks

- **At 10x (≈70M CCU across expanded portfolio)**: Kafka partition count and broker fleet scale roughly linearly (manageable, ~450 brokers) but ClickHouse shard rebalancing operations become a real operational burden — need to move to a sharding scheme that pre-allocates headroom per title tier rather than reactive resharding.
- **At 10x**: Flink stateful window operators for high-cardinality dimensions (e.g., per-player session state for very popular titles) risk state-size blowup — mitigate with state TTL and RocksDB incremental checkpoints; without this, checkpoint duration grows and can breach recovery-time SLAs.
- **At 100x (≈700M CCU, portfolio-wide across all EA titles + regions)**: single "central aggregation region" for global rollups (§31) becomes a bottleneck — global KPI computation would need to become hierarchical (regional rollups → continental rollups → global), not a single-region fan-in.
- **At 100x**: query API RBAC/row-level security checks against a single Postgres metadata store become a latency bottleneck at high dashboard QPS — would need to cache RBAC decisions or move to a distributed policy engine (e.g., OPA sidecar with local policy cache).

## 43. Latency Bottlenecks

End-to-end p50/p99 budget, event emission → dashboard visible:

| Stage | p50 | p99 |
|---|---|---|
| Client → gateway (network + validation) | 80ms | 400ms |
| Gateway → Kafka produce ack | 20ms | 150ms |
| Kafka → Flink consume + window close (up to window size) | 5s (1s windows, worst-case wait) | 60s (1min windows, worst-case wait) |
| Flink → serving cache write | 10ms | 80ms |
| Dashboard poll/refresh cycle | 2.5s (half of 5s refresh) | 15s (half of 30s refresh) |
| **Total (streaming path)** | **~8s** | **~30s** (meets NFR: p99 < 30s) |

- Biggest lever: window size choice — 1-min tumbling windows trade off freshness vs. aggregation overhead/small-file problem in downstream sinks; most dashboards use 10s or 1min windows, sub-second metrics reserved for the most critical KPIs (CCU, crash rate) only.
- OLAP ad-hoc query p99 (5s target) is dominated by bytes-scanned when queries don't hit partition pruning well — the single biggest lever is enforcing time-range + title_id predicates in the Query API before allowing full execution.

## 44. Cost Bottlenecks

- **#1 driver**: OLAP hot storage (675 TB across replicas) + associated compute nodes — the single largest line item; addressed via aggressive downsampling/tiering (§29).
- **#2 driver**: Kafka broker fleet sized for peak burst (6M events/sec) that's only hit during launches/tournaments — most of the year runs at 10-20% of provisioned capacity; addressed via right-sizing base capacity + burst autoscaling rather than static peak provisioning where the streaming platform supports it.
- **#3 driver**: cross-region data transfer (MirrorMaker replication of full telemetry volume to a central region) — mitigate by replicating only pre-aggregated rollups cross-region for global KPIs, not raw events, keeping raw event replication region-local only.
- **#4 driver**: query-time full scans on OLAP from ungoverned ad-hoc queries — mitigated by query-cost estimation and caps (§29).

## 45. Interview Follow-Up Questions

1. How would you handle a single game title having 10x the traffic of all others combined (the "hot title" / noisy-neighbor problem) in a multi-tenant system?
2. Walk me through what happens if the anomaly detection service itself goes down for an hour during a major incident.
3. How do you prevent a bad client SDK from silently corrupting aggregates before anyone notices?
4. Why did you choose statistical anomaly detection over a learned model, and what would change your mind?
5. How would you support a live-ops engineer who needs a brand-new custom metric within the hour, not after a data-platform team ticket cycle?
6. What happens to in-flight streaming aggregates during a Flink job version upgrade — walk through exactly-once vs at-least-once implications.
7. How would you extend this system to support cross-title/portfolio-wide exec dashboards without violating per-title data isolation?
8. Where would you introduce a learned (ML) component into this system, and what would its input/output contract look like?
9. How do you validate that a "corrected" backfill/replay didn't silently change historical numbers that live-ops already reported to leadership?
10. What's your approach to testing this system — how do you know the streaming aggregation logic is correct before it ships?

## 46. Ideal Answers

1. **Hot title / noisy neighbor**: Partition Kafka topics and OLAP shards by `title_id` so one title's volume can't starve others, and apply per-title quotas at the gateway. Give very large titles dedicated Flink clusters and ClickHouse shard groups as their own tenant tier, while smaller titles share a multi-tenant pool.

2. **Anomaly detection service down for an hour**: Detection is decoupled from ingestion/aggregation, so no data loss occurs — the gap is purely in alerting. A watchdog fires a meta-alert if detection's heartbeat stops, and on recovery it re-runs detection against the retained history to backfill any missed alerts, labeled as late.

3. **Bad client SDK corrupting aggregates**: Schema validation at the edge gateway rejects/quarantines events that don't match the registered schema, routing them to a DLQ instead of the aggregation topics, with sanity-range checks as a second tier. Monitor DLQ rate and per-field value-distribution drift to catch issues (e.g., a units bug) automatically before a human notices a weird dashboard.

4. **Statistical vs learned anomaly detection**: Statistical models are chosen because live-ops KPIs are well-understood, seasonal, low-dimensional signals where EWMA/seasonal-ESD gives interpretable, fast, low-latency detection with no training infra needed. Move to a learned multivariate model if univariate thresholds create too many correlated false positives, or if anomalies only show up in combinations of KPIs that no single threshold would catch.

5. **Self-service custom metrics**: Expose a metric-definition API/UI where a live-ops engineer defines a metric as a SQL-like expression, validated and dry-run for sanity/cost, then automatically compiled into a Flink aggregation rule or OLAP view via the same CI/CD pipeline. This turns a ticket-cycle wait into a self-service, same-hour flow while keeping automated correctness guardrails.

6. **Flink job upgrade — exactly-once vs at-least-once**: Deploy the new job version under a shadow consumer group, validate its output against the old job's live output for a window, then cut the serving-cache writer over and stop the old job with a final savepoint so no in-flight window is lost. Flink's checkpointing plus an idempotent sink gives exactly-once at the aggregate level even though Kafka delivery itself is only at-least-once.

7. **Portfolio-wide exec dashboards without violating isolation**: Keep per-title raw data and detailed OLAP tables isolated with title-scoped RBAC, but build a separate, deliberately coarse-grained aggregation layer (e.g., "daily revenue by title") that only exposes pre-approved, already-aggregated cross-title metrics. This layer has its own exec-only RBAC tier and is populated via a controlled ETL step titles opt into, rather than exec dashboards querying raw tables directly.

8. **Where ML fits**: The natural insertion point is anomaly detection — replacing the univariate statistical model with a lightweight multivariate model (e.g., isolation forest) whose input is the per-tick vector of normalized KPI values and whose output is an anomaly score plus a per-KPI attribution score for explainability. This keeps the heavy telemetry pipeline unchanged — ML sits downstream of aggregation, not upstream.

9. **Validating backfill/replay correctness**: Never overwrite historical reported numbers silently — write corrected aggregates to a new versioned table/partition, run the corrected pipeline in parallel for a validation window, and produce a diff report reviewed before promoting. Any dashboard showing a "corrected" number carries a visible annotation linking to the diff/reason.

10. **Testing streaming aggregation correctness**: Unit tests for individual window/aggregation functions, integration tests replaying a hand-verified production sample through the actual Flink job and diffing against expected output, and shadow-mode validation before promotion. A continuous "canary metric" — a synthetic event stream with a known-correct aggregate — gives an always-on correctness check independent of real traffic.
