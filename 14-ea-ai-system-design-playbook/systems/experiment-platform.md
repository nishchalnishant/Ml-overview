# Experiment Platform (A/B Testing)

## 1. Problem Framing & Requirement Gathering

EA ships live-service games (FIFA/EA FC Ultimate Team, Apex Legends, The Sims, Battlefield) where nearly every product and ML decision — matchmaking tuning, recommendation ranking, monetization offers, difficulty curves, churn-prevention interventions — is validated through controlled experiments before global rollout. The Experiment Platform is the shared infrastructure that:

- Assigns players/sessions/devices to experiment variants consistently and without bias.
- Collects and pipelines the metrics needed to judge experiment outcomes (engagement, revenue, retention, latency, crash rate).
- Runs statistically rigorous analysis (frequentist + sequential) so teams can call experiments early without inflating false-positive rates.
- Protects against regressions via guardrail metrics (crash rate, p99 latency, revenue-per-user floor) that can auto-halt a rollout.
- Supports **ML-specific experimentation**: holdouts for models in production (e.g., "keep 5% of matchmaking on legacy ELO, no ML"), shadow deployments, and champion/challenger tests for ranking or recommendation models.

This is infrastructure, not a single feature — dozens of game studios and central ML teams (Ads, LiveOps, Anti-Cheat, Personalization) are tenants. The interview framing is: **design a multi-tenant experimentation platform that is correct (no assignment bugs, no p-hacking), fast (assignment on the hot path of every game session), and trustworthy (stats team can audit every number).**

## 2. Functional Requirements

- FR1: Register an experiment with variants, allocation percentages, targeting rules (segment, platform, region, game title, ML-model cohort).
- FR2: Assign a stable variant to a unit (player_id, device_id, session_id, or match_id) with deterministic, reproducible bucketing.
- FR3: Support mutually-exclusive experiment layers (a player can be in one experiment per layer, many layers concurrently) to prevent interaction effects.
- FR4: Ingest exposure events ("unit X saw variant Y at time T") and outcome/metric events (purchases, session length, crashes, matchmaking latency).
- FR5: Compute experiment scorecards: mean/proportion metrics, confidence intervals, p-values, lift %, sample ratio mismatch (SRM) checks.
- FR6: Support **sequential testing** (mSPRT / always-valid p-values) so teams can peek daily without inflating Type-I error.
- FR7: Support **guardrail metrics** with automatic experiment pause/rollback when breached.
- FR8: Support **ML model holdouts**: persistent long-lived control groups (e.g., 2% of traffic never gets any ML personalization) to measure long-run model value net of novelty effects.
- FR9: Support mutual exclusion / interaction detection between simultaneously running experiments.
- FR10: Expose a self-serve UI/API for experiment lifecycle: draft → running → paused → concluded → archived.
- FR11: Retroactive/backfill analysis: recompute a scorecard for an arbitrary historical date range.
- FR12: Audit trail: who changed allocation %, when, and why (compliance for revenue-impacting experiments).

## 3. Non-Functional Requirements (latency, availability, throughput, consistency, cost)

| Dimension | Target | Notes |
|---|---|---|
| Assignment latency (p99) | < 10 ms | On critical path of session start / matchmaking / storefront render |
| Assignment availability | 99.99% | Must degrade to deterministic local hashing if service is down — never block gameplay |
| Assignment throughput | 2M QPS peak | Aggregated across all EA titles at global peak (evening NA + EU overlap) |
| Metrics ingestion throughput | 3–5M events/sec peak | Telemetry from matches, purchases, client health |
| Consistency of assignment | Strong, per-unit, permanent for experiment lifetime | Same unit + same experiment => same variant, always, even across region failover |
| Analysis freshness | Guardrails: < 15 min; full scorecards: < 24 hr | Guardrails need near-real-time; deep stats can batch |
| Storage durability | 99.999999999% (11 nines, S3-class) | Raw exposure/metric events are the source of truth for audits |
| Cost ceiling | Experimentation infra ≤ 3% of total telemetry spend | Must ride on existing telemetry pipeline, not duplicate it |

## 5. Assumptions

1. ~90M MAU across EA's top live-service titles combined (FC, Apex, Sims, Battlefield, Madden).
2. Average concurrent players at peak: ~4M globally; average session emits ~150 telemetry events including 2–5 relevant to experiments.
3. ~400 concurrently running experiments platform-wide at any time, across ~15 layers.
4. Unit of randomization is predominantly `player_id` (hashed pseudonymous ID), with a secondary `match_id` mode for matchmaking-level experiments.
5. Client SDKs embedded in game engines (Frostbite, proprietary) can cache assignment + do deterministic local hashing as fallback.
6. Guardrail metrics: crash rate, p99 client-server RTT, session-start failure rate, revenue-per-paying-user floor.
7. Analysis backend is batch (Spark/Trino on a lakehouse) for scorecards, streaming (Flink) for guardrails.
8. ML holdouts are long-lived (multi-quarter), separate lifecycle from short-lived feature experiments (1–4 weeks).
9. Data retention: raw exposure/metric events 13 months (matches finance/compliance needs); aggregated scorecards indefinitely.
10. Multi-region: NA, EU, APAC each have regional game servers; experiment config is replicated globally, event ingestion is regional-first.

## 6. Capacity Estimation

**Assignment QPS**
- Peak concurrent sessions: 4M. Each session requests assignment ~3 times/session (session start, mid-session re-check for long-lived layers, storefront open) → ~12M assignment calls over a session lifetime, but rate-smoothed:
- Assume avg session = 40 min, and each active session issues 1 assignment check per 2 min average (cached client-side otherwise) → 4M sessions / 2 min × 60 = ~33K QPS steady, bursting to ~200K QPS at peak login events (patch-day spikes, esports events). Design for **300K QPS sustained, 1M QPS burst** at the assignment tier with client-side caching absorbing the rest (client caches assignment for session lifetime → real server-side QPS is much lower than requests-per-session suggests).
- Per assignment: unit_id lookup + hash + rule eval ≈ 200 bytes request, 150 bytes response.

**Metrics ingestion**
- 4M concurrent × ~1 relevant exposure/metric event per 10 sec average = 400K events/sec steady, 3M events/sec peak (match end bursts, patch-day).
- Event size ~500 bytes avg (JSON/Avro) → peak ingest bandwidth ≈ 3M × 500B = 1.5 GB/s.

**Storage**
- Raw events: 3M events/sec × 500B × 86,400 sec/day ≈ 130 TB/day raw → compressed (Parquet, ~5:1) ≈ 26 TB/day. 13-month retention ≈ 26TB × 400 days ≈ **10.4 PB** on the lakehouse (columnar, partitioned).
- Experiment metadata (configs, allocations, audit log): trivial, < 50 GB total in Postgres.
- Assignment cache (Redis): 90M MAU × 400 experiments × ~20 bytes (unit+exp+variant packed) worst case = 720 GB if fully materialized — in practice only active experiments per user matter; realistic working set ≈ 90M × 15 concurrently-relevant experiments × 20B ≈ **27 GB**, fits comfortably in a sharded Redis cluster (say 6 shards × 8GB).

**Compute for stats engine**
- Nightly batch scorecard job: 400 experiments × ~10 metrics × Trino query over ~1 day of relevant partition (~26TB/day slice, but filtered to experiment cohorts, effectively scanning tens of GB per experiment after partitioning/pruning) → estimate 400 experiments × 45 sec Trino query ≈ 5 hours of query time; parallelized across a 100-node Trino cluster → wall clock < 20 min.
- Sequential/guardrail streaming: Flink job maintaining ~400 experiments × 5 guardrail metrics × running sufficient statistics (mean, var, count) — state size trivial (~KBs per metric), CPU-bound by event throughput (3M events/sec) → needs ~150 vCPUs at 20K events/sec/core for stateful aggregation, round up to a 200-vCPU Flink cluster with headroom.
- No GPU requirement for the core platform — this is a stats/data-infra system, not a model-serving system, **except** for the ML-holdout evaluation which may call an existing model-serving endpoint (out of scope, budgeted separately) and any propensity-score/CUPED variance-reduction model training (small, CPU-only, retrained daily, < 10 min on 8 vCPUs).

## 7. High-Level Architecture

```
                         ┌─────────────────────────┐
                         │   Experiment Console/API │  (create/manage experiments,
                         │   (Config Service)       │   targeting rules, allocations)
                         └────────────┬─────────────┘
                                      │ writes
                                      ▼
                         ┌─────────────────────────┐
                         │  Config Store (Postgres) │──replicate──► Regional read replicas
                         │  + Audit Log             │
                         └────────────┬─────────────┘
                                      │ push (CDC) to config cache
                                      ▼
        ┌────────────────────────────────────────────────────────┐
        │                Assignment Service (per region)          │
        │  - loads experiment configs from local cache (CDN/Redis)│
        │  - deterministic hash(unit_id, layer_salt) -> variant   │
        │  - SDK fallback: client computes same hash offline      │
        └───────┬───────────────────────────────┬──────────────────┘
                │ assignment result              │ exposure event
                ▼                                 ▼
       ┌────────────────┐              ┌─────────────────────────┐
       │ Game Client/SVC │              │  Event Ingestion Gateway │
       │ (Frostbite SDK) │              │  (Kafka producers)       │
       └────────────────┘              └───────────┬─────────────┘
                                                     │
                             ┌───────────────────────┼───────────────────────┐
                             ▼                       ▼                       ▼
                     ┌───────────────┐      ┌────────────────┐      ┌────────────────┐
                     │ Kafka: exposure│      │ Kafka: metrics  │      │ Kafka: guardrail│
                     │   topic        │      │   topic         │      │   candidate topic│
                     └───────┬───────┘      └────────┬────────┘      └────────┬────────┘
                             │                        │                        │
                    ┌────────▼────────┐      ┌────────▼─────────┐    ┌─────────▼─────────┐
                    │ Lakehouse Sink   │      │ Lakehouse Sink    │    │ Flink Guardrail    │
                    │ (Parquet/Iceberg)│      │ (Parquet/Iceberg) │    │ Streaming Job      │
                    └────────┬─────────┘      └────────┬─────────┘    └─────────┬─────────┘
                             │                          │                        │
                             ▼                          ▼                        ▼
                    ┌──────────────────────────────────────────┐       ┌──────────────────┐
                    │   Batch Stats Engine (Spark/Trino)        │       │ Guardrail Monitor │
                    │   - frequentist scorecards                │       │ - breach detection│
                    │   - sequential/mSPRT always-valid p-values│       │ - auto-pause hook │
                    │   - SRM checks, CUPED variance reduction   │       └────────┬──────────┘
                    └────────────────┬───────────────────────────┘                │
                                     ▼                                            ▼
                          ┌─────────────────────┐                     ┌─────────────────────┐
                          │ Scorecard Store      │                     │ Config Service        │
                          │ (columnar, e.g.       │                     │ (auto-pause / rollback)│
                          │  ClickHouse)          │                     └─────────────────────┘
                          └──────────┬────────────┘
                                     ▼
                          ┌─────────────────────┐
                          │ Experiment Console UI │
                          │ (dashboards, alerts)  │
                          └─────────────────────┘
```

## 8. Low-Level Components

**Config Service**
- Responsibility: CRUD for experiments, variants, allocation %, targeting rules, layers, ML-holdout definitions; enforces mutual exclusion across layers; writes immutable audit log entries.
- Interface: gRPC/REST internal API, consumed by Console UI and by Assignment Service (via CDC-fed cache, not direct query).
- Scaling unit: stateless service in front of Postgres; scales on request rate for admin operations (low volume, tens of QPS) — not a bottleneck.

**Assignment Service**
- Responsibility: given `(unit_id, experiment_or_layer_id, context)`, return a deterministic variant; emit exposure event asynchronously (fire-and-forget, non-blocking).
- Interface: gRPC, colocated in each region; also a shared library (SDK) embedded in game clients replicating the exact same hash function for offline/degraded-mode assignment.
- Scaling unit: horizontally scaled stateless pods behind regional load balancer; scale on QPS (target 5K QPS/pod).

**Event Ingestion Gateway**
- Responsibility: validate, schema-check, and route exposure/metric events into Kafka topics; deduplicate using client-generated idempotency keys.
- Interface: HTTP/gRPC batch endpoint (clients batch events client-side to reduce call volume).
- Scaling unit: scales on ingestion bandwidth (GB/s); partition Kafka topics by `hash(unit_id) % N` for even distribution and to keep a given unit's events ordered per-partition.

**Guardrail Streaming Monitor (Flink)**
- Responsibility: maintain running sufficient statistics per experiment/variant/guardrail metric; evaluate breach conditions every evaluation window (e.g., every 5 min); call Config Service to auto-pause on breach.
- Interface: consumes guardrail-candidate Kafka topic; internal state in RocksDB-backed Flink state store, checkpointed to S3.
- Scaling unit: Flink task slots scale on partition count of the guardrail topic; stateful, so rebalancing is checkpoint-restore, not instant.

**Batch Stats Engine**
- Responsibility: nightly (and on-demand) full scorecards — mean/proportion tests, CIs, p-values, SRM chi-square test, CUPED-adjusted variance reduction, mSPRT sequential boundary evaluation for "peekable" experiments.
- Interface: Spark/Trino jobs orchestrated by Airflow; outputs written to ClickHouse scorecard store.
- Scaling unit: cluster autoscales on job backlog (number of experiments needing (re)computation).

**Experiment Console / Dashboard**
- Responsibility: self-serve UI for creating experiments, viewing scorecards, guardrail status, audit history.
- Interface: REST/GraphQL over Scorecard Store + Config Service.
- Scaling unit: standard stateless web tier.

**ML Holdout Manager**
- Responsibility: manages long-lived holdout cohorts tied to specific model deployments; ensures holdout membership survives model version upgrades (a player held out of "recommendation model v3" stays held out through v4, v5 unless explicitly graduated); coordinates with Model Serving to force control-path (heuristic/legacy) behavior for holdout members.
- Interface: extension of Config Service with a `holdout_cohort` entity type; read by Model Serving at inference time as a feature/flag.
- Scaling unit: piggybacks on Config Service's cache-replication path — holdout membership must be as fast to check as any other assignment (<10ms) since it gates a live inference request.

## 9. API Design

| Endpoint | Method | Purpose |
|---|---|---|
| `/v2/experiments` | POST | Create experiment (draft state) |
| `/v2/experiments/{id}` | GET/PATCH | Read/update experiment config |
| `/v2/experiments/{id}/start` | POST | Transition draft → running |
| `/v2/experiments/{id}/pause` | POST | Manual or auto (guardrail) pause |
| `/v2/assign` | POST | Hot-path assignment call |
| `/v2/expose` | POST | Batch exposure event submission (usually via SDK, not direct) |
| `/v2/experiments/{id}/scorecard` | GET | Latest computed scorecard |
| `/v2/experiments/{id}/scorecard?as_of=DATE` | GET | Backfilled/historical scorecard |
| `/v2/holdouts` | POST/GET | Manage ML holdout cohorts |
| `/v2/holdouts/{id}/check` | POST | Fast holdout-membership check (model serving hot path) |

**`POST /v2/assign` — request**
```json
{
  "unit_id": "hashed_player_id_abc123",
  "unit_type": "player",
  "layer": "matchmaking_layer",
  "context": {"platform": "ps5", "region": "eu-west", "title": "fc25", "app_version": "1.14.2"}
}
```

**`POST /v2/assign` — response**
```json
{
  "assignments": [
    {"experiment_id": "exp_9931", "variant": "treatment_b", "layer": "matchmaking_layer"}
  ],
  "assignment_hash_version": 3,
  "served_from": "regional_cache"
}
```

**`GET /v2/experiments/{id}/scorecard` — response (abridged)**
```json
{
  "experiment_id": "exp_9931",
  "status": "running",
  "as_of": "2026-07-08T00:00:00Z",
  "srm_check": {"passed": true, "chi_sq_p": 0.42},
  "metrics": [
    {
      "name": "session_length_min",
      "control_mean": 38.2, "treatment_mean": 39.6,
      "lift_pct": 3.66, "p_value": 0.021, "ci_95": [0.008, 0.065],
      "sequential_boundary_crossed": true
    }
  ],
  "guardrails": [
    {"name": "crash_rate", "control": 0.0021, "treatment": 0.0023, "breached": false}
  ]
}
```

Versioning: URI-versioned (`/v2/`), with `assignment_hash_version` field so historical assignments remain reproducible even if the hash function/bucketing algorithm changes in v3 — old experiments keep resolving against their original hash version.

## 10. Database Design

| Store | Type | Used for | Partition/Shard key |
|---|---|---|---|
| Config DB | Postgres (relational) | Experiment configs, targeting rules, audit log | Sharded by `title_id` (game) once single-instance write throughput becomes limiting; today single primary + read replicas suffice (low write volume) |
| Assignment Cache | Redis Cluster | Hot-path variant lookups, holdout membership | Hash slot on `unit_id` |
| Raw Event Lake | Iceberg/Parquet on S3 | Exposure + metric events, source of truth for audit/backfill | Partitioned by `event_date` then bucketed by `hash(unit_id) % 256` for scan pruning per-cohort |
| Scorecard Store | ClickHouse (columnar) | Computed scorecards, time series of metric lift over experiment lifetime | Partitioned by `experiment_id` + `date`, ordered by `(experiment_id, variant, metric_name)` for fast rollup queries |
| Guardrail State | Flink RocksDB state + S3 checkpoints | Running sufficient statistics per experiment/variant/metric | Keyed by `(experiment_id, variant, metric)` |

Why this split: Postgres for config because it's low-volume, needs ACID (allocation % changes must be atomic and audited), and relational integrity (foreign keys between experiments/layers/holdouts) matters. Iceberg/Parquet for raw events because 10+ PB scale and columnar scan efficiency for ad-hoc backfill queries dominate. ClickHouse for scorecards because dashboards need sub-second aggregation over metric time series across hundreds of experiments — a workload columnar OLAP stores excel at and Postgres would choke on at this cardinality.

## 11. Caching

- **What's cached**: experiment configs (targeting rules, allocation %) replicated to regional Redis/CDN-like cache via CDC from Postgres; per-unit assignment results (once computed, a variant assignment is permanent for the experiment's life — cache it forever, invalidate only on experiment conclusion/archival).
- **Strategy**: cache-aside for config (Assignment Service reads from Redis, falls back to Postgres read-replica on cache miss, repopulates). Write-through for assignment results — once the Assignment Service computes a variant (deterministic hash, so idempotent), it writes to Redis synchronously before responding, so repeat calls for the same unit are O(1) cache hits and consistent even across service replicas.
- **Invalidation**: config cache invalidated via CDC stream (Postgres WAL → Debezium → cache update) within seconds of an admin change (e.g., emergency allocation rollback to 0%). Assignment-result cache is essentially write-once/never-invalidated during an experiment's active life (bucketing must stay stable — re-randomizing mid-experiment invalidates the whole analysis). On experiment archival, a TTL sweep evicts old assignment entries after retention window.
- Because the hash function is deterministic (`hash(unit_id + experiment_salt) → bucket`), cache misses are cheap to recompute — Redis is a performance optimization, not a correctness dependency. This is critical: **a full Redis cluster loss must not corrupt assignment**, it just adds recompute latency.

## 12. Queues & Async Processing

- **What's queued**: exposure events, metric events, guardrail-candidate events — all via Kafka. Config-change notifications (CDC) also flow through a Kafka topic for cache invalidation fan-out.
- **Delivery semantics**: **at-least-once** end-to-end. Clients attach an idempotency key (`unit_id + event_type + client_timestamp + nonce`); ingestion gateway and lakehouse sink dedupe on this key within a rolling window (24h dedupe cache in Redis/RocksDB). Exactly-once isn't chased at the transport layer — it's cheaper to dedupe at read/aggregation time (idempotent aggregation keys) than to pay for distributed transactions across Kafka+lakehouse.
- **Dead-letter handling**: malformed events (schema validation failure, missing required `unit_id`) route to a `dlq.experiment-events` topic; a separate low-priority consumer logs, alerts if DLQ volume > 0.1% of total, and retains for 7 days for manual replay/inspection. Guardrail-topic DLQ has tighter alerting (any DLQ growth here risks masking a real regression).
- **Backpressure**: ingestion gateway sheds load (returns 429, client SDK buffers and retries with jitter) rather than blocking gameplay — telemetry loss is preferable to session stalls.

## 13. Streaming & Event-Driven Architecture

**Topics**

| Topic | Producers | Consumers | Partitions | Retention |
|---|---|---|---|---|
| `exposure-events` | Assignment Service, game clients | Lakehouse sink, Guardrail Monitor | 512 | 7 days (then lakehouse is source of truth) |
| `metric-events` | Game clients, backend services (purchase, matchmaking) | Lakehouse sink | 1024 | 7 days |
| `guardrail-candidate` | Metric routers (filtered subset tagged as guardrail-relevant) | Flink Guardrail Monitor | 256 | 3 days |
| `config-changes` (CDC) | Debezium on Postgres WAL | Regional cache updaters | 16 | 3 days |
| `dlq.experiment-events` | All ingestion paths on validation failure | DLQ inspector/replay tool | 32 | 7 days |

**Event schema (Avro, `exposure-events`)**
```json
{
  "type": "record", "name": "ExposureEvent",
  "fields": [
    {"name": "unit_id", "type": "string"},
    {"name": "experiment_id", "type": "string"},
    {"name": "variant", "type": "string"},
    {"name": "layer", "type": "string"},
    {"name": "event_ts", "type": "long"},
    {"name": "context", "type": {"type": "map", "values": "string"}},
    {"name": "idempotency_key", "type": "string"}
  ]
}
```

**Consumer groups**: `lakehouse-sink-group` (scales with partition count, exactly-once-ish via Iceberg transactional commits + idempotency key dedup), `guardrail-monitor-group` (Flink consumer group, stateful), `cache-invalidation-group` (small, low-latency-focused, few consumers since config-changes volume is tiny).

## 14. Model Serving

The Experiment Platform itself serves no ML model for its core function (assignment is a hash, not inference) — but it **gates** and **coordinates with** model serving for:
- ML holdouts: Model Serving checks `/v2/holdouts/{id}/check` (or a locally-cached holdout-membership flag) before deciding to run the model at all vs. fall back to heuristic.
- Champion/challenger tests: two model versions both deployed behind Model Serving, with the Experiment Platform's variant assignment used as the routing key.
- CUPED/propensity models used internally by the Batch Stats Engine for variance reduction: small (logistic regression / gradient-boosted trees on pre-experiment covariates), served offline in the Spark job itself (no separate serving tier needed — batch scoring during scorecard computation).
- Hardware: no GPU requirement; holdout-check is a cache lookup (<1ms), CUPED model scoring is CPU batch work inside Spark executors.
- Batching: N/A for the platform itself; Model Serving (the tenant system) handles its own dynamic batching independently — Experiment Platform only supplies the routing decision as a fast synchronous or cached call.

## 15. Feature Store

- **Online store**: not owned by the Experiment Platform, but the platform is a *feature producer* — `current_variant_assignments` and `holdout_membership` are published as low-latency features (via the shared Feature Store's online layer, e.g., Redis-backed) so that downstream ML models (recommendation, matchmaking) can condition behavior on experiment state without a second round-trip.
- **Offline store**: exposure/metric events land in the same lakehouse (Iceberg tables) used by the broader Feature Store for offline training-data joins — e.g., a churn model's training pipeline can join "was this player in the new-onboarding treatment" as a feature.
- **Point-in-time correctness**: exposure events are timestamped at assignment time; offline joins use `event_ts <= label_window_start` filtering (standard PIT join) so that a model can't leak "was assigned treatment" information from *after* the label was generated. This matters specifically for holdout-based training: models must never be trained on data contaminated by knowing a player's holdout status after the fact if that status is meant to reflect a pre-registration design.

## 16. Vector Database

N/A. The Experiment Platform has no embedding/similarity-search workload — assignment is deterministic hashing, not nearest-neighbor retrieval. (Tenant ML systems that this platform experiments *on*, e.g., a recommendation system, may use a vector DB, but that is out of this system's scope.)

## 17. Embedding Pipelines

N/A for the same reason — no unstructured content, no similarity search, no embedding generation is part of assignment, metrics, or stats analysis. The only "model" activity within this system's boundary is lightweight tabular CUPED/propensity scoring, which operates on numeric covariates, not embeddings.

## 18. Inference Pipelines

There's no traditional "inference pipeline" (no forward pass through a neural net on the request path), but the **assignment request lifecycle** is the analogous hot path:

```
Game Client                Assignment Svc            Redis Cache        Kafka
    │  POST /v2/assign          │                        │                │
    ├──────────────────────────►│                         │                │
    │                           │  1. lookup unit_id in   │                │
    │                           │     regional cache       │                │
    │                           ├────────────────────────►│                │
    │                           │◄────────────────────────┤                │
    │                           │  cache hit? return       │                │
    │                           │  cache miss:              │                │
    │                           │  2. load active configs  │                │
    │                           │     for unit's layers     │                │
    │                           │     (from local config    │                │
    │                           │      cache, CDC-fed)       │                │
    │                           │  3. eval targeting rules  │                │
    │                           │     (region/platform/     │                │
    │                           │      segment match)        │                │
    │                           │  4. deterministic hash:    │                │
    │                           │     bucket = hash(unit_id  │                │
    │                           │       + layer_salt) % 10000│                │
    │                           │  5. map bucket -> variant  │                │
    │                           │     per allocation ranges  │                │
    │                           │  6. write-through to cache │                │
    │                           ├────────────────────────►│                │
    │                           │  7. emit exposure event    │                │
    │                           │     (async, fire-and-forget)│               │
    │                           ├─────────────────────────────────────────►│
    │◄──────────────────────────┤  return variant(s)         │                │
    │  total p99 target: <10ms  │                            │                │
```

- Step 2 (config load) is itself cached locally in-process (refreshed via CDC push, not per-request Redis call) to shave network hops on the common path.
- Step 7 is strictly async — a Kafka producer failure must never fail the assignment response.

## 19. Training Pipelines

The platform's only "trained" artifacts are small variance-reduction/propensity models used by the Batch Stats Engine (not player-facing ML):

- **Data prep**: pull pre-experiment covariates (historical session length, spend, tenure) for units entering an experiment, joined from the lakehouse, point-in-time filtered to before experiment start.
- **Training orchestration**: Airflow DAG triggers a daily Spark job that (re)fits a CUPED regression (metric ~ pre-period covariate) per active experiment/metric pair, or a pooled propensity model for non-randomized quasi-experiments (rare, used for holdout graduation analysis).
- **Distributed training**: not needed at this scale (regression on covariates for a few hundred experiments, each with at most a few million rows) — single Spark executor per experiment task is sufficient; job parallelizes *across* experiments, not within a single model fit.
- Model artifacts are ephemeral, versioned per experiment run in the Scorecard Store's metadata, not served — used once per scorecard computation and discarded.

## 20. Retraining Strategy

- CUPED/propensity models: retrained **daily**, scoped per-experiment — there is no long-lived model to go stale, since each experiment gets a freshly fit covariate-adjustment model against its own current data window. "Retraining" here really means "refit," triggered automatically as part of the nightly scorecard DAG.
- Trigger: DAG runs on fixed nightly cadence (00:00 UTC) plus on-demand trigger when a scorecard is manually requested for backfill.
- No drift-based retraining trigger needed for these small covariate models since they're refit from scratch each run — the concept of "staleness" doesn't apply the way it does for a persistent production model.

## 21. Drift Detection

Drift detection in this system is about **experiment/data integrity**, not a persistent ML model's accuracy:

- **Sample Ratio Mismatch (SRM)**: chi-square goodness-of-fit test comparing observed vs. expected allocation ratio per variant. Threshold: p < 0.001 flags SRM (stricter than typical 0.05 because false SRM alarms are costly to chase, but real SRM invalidates the whole experiment). Checked automatically on every scorecard computation.
- **Assignment drift**: monitor whether the *realized* variant distribution matches configured allocation % over a rolling 24h window, per platform/region/segment — catches targeting-rule bugs (e.g., an SDK version silently failing to assign a variant, defaulting everyone to control).
- **Metric distribution drift** (covariate/data drift on inputs to CUPED model): compare pre-period covariate distributions between control/treatment — should be statistically indistinguishable pre-randomization; a shift here indicates a bucketing bug, not a real treatment effect. Use PSI (population stability index) with threshold PSI > 0.2 flags investigation.
- **Concept drift** in the loose sense: guardrail baselines (e.g., "normal" crash rate) are recomputed on a trailing 14-day window so that guardrail thresholds adapt to seasonal shifts (patch releases, esports events) rather than alerting on stale absolute thresholds.

## 22. Monitoring

| Category | Metrics |
|---|---|
| Infra | Assignment Service p50/p99 latency, QPS, error rate; Kafka consumer lag per topic; Redis hit rate, memory pressure; Flink checkpoint duration/failures |
| Data pipeline | Event ingestion rate vs. expected baseline; DLQ volume %; lakehouse sink lag (event_ts to landed-in-Iceberg latency) |
| Experiment integrity | SRM flags count, assignment-distribution drift, config-cache staleness (time since last CDC apply) |
| Model quality (scorecard engine) | CUPED variance-reduction % achieved, propensity model calibration (for graduation analyses) |
| Business/product | Per-experiment lift on core metrics (revenue, session length, retention D1/D7), guardrail breach count/week, number of active experiments per layer (interaction-risk proxy) |

## 23. Alerting

| Condition | Threshold | Route |
|---|---|---|
| Assignment Service p99 > 20ms for 5 min | Page | On-call SRE (experiment-platform-oncall) |
| Assignment error rate > 0.5% | Page | On-call SRE |
| Kafka consumer lag > 5 min on guardrail-candidate topic | Page | On-call SRE (blocks guardrail detection — high severity) |
| Guardrail breach detected (e.g., crash rate +20% relative in treatment) | Immediate auto-pause + Slack/Page to experiment owner + platform on-call | Experiment owner (primary), SRE (secondary) |
| SRM flagged on a running experiment | Slack notification (not paging — needs human judgment) | Experiment owner + data science partner |
| DLQ volume > 0.1% of topic volume for 15 min | Slack | Data platform on-call |
| Config cache staleness > 60s (CDC lag) | Page | On-call SRE |
| Nightly scorecard DAG failure | Slack + retry | Data platform on-call |

## 24. Logging

- **Structured logging**: JSON logs for every assignment decision (unit_id hashed/pseudonymous, experiment_id, variant, latency, cache hit/miss) — enables replay/audit of "why was this player in variant B."
- **PII handling**: `unit_id` is always a pseudonymous hashed identifier at the platform boundary (raw player account IDs are hashed with a rotating-but-consistent-per-experiment salt before entering this system) — the Experiment Platform never stores raw PII (email, real name, payment info). Context fields (region, platform) are coarse-grained, not device fingerprints.
- **Retention**: application/infra logs 30 days (Elastic/Datadog); exposure/metric events (the "logs" that matter for analysis) retained 13 months in the lakehouse per Assumption 9, access-controlled and subject to the same data-governance deletion pipeline as other player telemetry (GDPR/CCPA right-to-erasure propagates a deletion request that purges matching `unit_id` rows via a scheduled compaction job, since Iceberg supports row-level delete).
- Audit log (who changed an allocation %) retained indefinitely, immutable, separate from telemetry logs — compliance requirement, not operational log.

## 25. Security

- **Threat model specifics**:
  - *Assignment manipulation*: a malicious/compromised client claiming a different `unit_id` to force itself into a favorable variant (e.g., a promo-pricing experiment) — mitigated by server-side unit_id derivation from authenticated session token, not client-supplied raw value, wherever the experiment has monetary stakes.
  - *Config tampering*: unauthorized allocation % change (e.g., insider pushing 100% to a variant that benefits them) — mitigated by RBAC + mandatory audit log + optional two-person approval for experiments tagged `monetization-sensitive`.
  - *Data exfiltration*: bulk export of the raw event lake (contains behavioral data across tens of millions of players) — mitigated by column-level access control, row-level tokenization of unit_id, and query-audit logging on the lakehouse.
  - *Guardrail bypass*: a rushed launch disabling guardrail checks to ship faster — mitigated by making guardrail evaluation non-optional infrastructure (can't be disabled per-experiment, only the specific metric list is configurable, and removing crash-rate/latency guardrails entirely requires a config-service admin override that's audited and alerted).
- **Encryption**: TLS in transit everywhere (client→gateway, service→service via mTLS in-mesh); at-rest encryption on S3/lakehouse (KMS-managed keys) and on Postgres (transparent data encryption); Redis cluster encryption-in-transit enabled (AUTH + TLS).

## 26. Authentication

- **Service-to-service**: mTLS via service mesh (Istio/Linkerd-style sidecar) inside the cluster; internal calls (Assignment Service → Config Service, Guardrail Monitor → Config Service for auto-pause) authenticate via short-lived SPIFFE/SVID identity certs.
- **End-user (game client)**: the game client is already authenticated via EA's account/session token (existing auth infra, e.g., EA Account SSO); the Assignment Service trusts the upstream API gateway to have validated this token and forwards a verified, signed `unit_id` claim — the Experiment Platform itself doesn't re-implement player auth.
- **Admin/Console UI**: SSO (Okta/EA internal IdP) + RBAC roles (`experiment-viewer`, `experiment-editor`, `experiment-admin`) scoped per title/team; monetization-sensitive experiments require an elevated role.

## 27. Rate Limiting

- **Algorithm**: token bucket per client/service-identity at the API gateway in front of the Assignment Service — allows short bursts (patch-day login spikes) while capping sustained abuse.
- **Limits**: per-title service accounts get a provisioned QPS budget (e.g., 400K QPS for the largest title, scaled down for smaller titles) reflecting Section 6's capacity plan; per-unit (a single player) limit is much lower (e.g., 10 assignment calls/sec) to catch a buggy client stuck in a retry loop.
- **Admin API** (Config Service): stricter, low limits (tens of QPS) since it's human-driven traffic, not hot-path — protects against accidental scripbehavior (e.g., a bulk-import script hammering the create-experiment endpoint).
- Rejected requests return 429 with `Retry-After`; client SDK is required to implement exponential backoff + jitter (enforced via SDK code review, not just server-side policy).

## 28. Autoscaling

- **Assignment Service**: HPA on custom metric = requests-per-second per pod (target 5K QPS/pod, matching Section 8's scaling-unit assumption), plus a secondary HPA trigger on p99 latency (scale out if p99 > 8ms sustained for 2 min, i.e., react before breaching the 10ms SLA). Min replicas sized for regional baseline traffic (never scale to zero — this is a hot-path service).
- **Guardrail Flink cluster**: scaled via KEDA on Kafka consumer-lag metric for the `guardrail-candidate` topic — lag > threshold (e.g., 30s worth of events) triggers additional task-manager pods; scale-in is conservative (longer cooldown) since Flink rebalancing has checkpoint overhead.
- **Batch Stats Engine (Spark/Trino)**: cluster autoscaling on job queue depth (number of pending experiment-scorecard tasks), typical cloud-native Spark-on-Kubernetes autoscaler; scales to near-zero overnight outside the nightly batch window aside from a small baseline for on-demand backfill requests.
- **Redis cache**: VPA-style vertical headroom monitoring (memory utilization) with manual/semi-automated shard-count increase when sustained > 70% memory utilization, since Redis Cluster resharding is a heavier operation than pod autoscaling.

## 29. Cost Optimization

- **Spot/preemptible instances**: Batch Stats Engine (Spark/Trino nightly jobs) runs almost entirely on spot capacity — batch jobs tolerate preemption/retry, and this is the single largest compute line item after storage; expect 60-70% cost reduction on this tier vs. on-demand.
- **Client-side caching**: the single biggest QPS-reduction lever — SDK caches a unit's assignment for the session lifetime, cutting server-side assignment QPS by roughly an order of magnitude vs. naive per-request calls (Section 6's 200K peak vs. a hypothetical multi-million-QPS naive design).
- **Columnar compression + partition pruning**: Iceberg/Parquet with well-chosen partition/bucket keys (Section 10) keeps ad-hoc scorecard queries scanning GBs instead of TBs — directly cuts Trino compute cost per query.
- **Tiered storage**: raw events older than 30 days move to cold/infrequent-access storage class (still queryable via Trino/Athena-style engines, just slower/cheaper) since most scorecard queries only touch the trailing 30 days; 13-month retention tail is compliance-driven, not query-driven.
- **No GPU spend**: as noted in Section 14, this system carries zero GPU cost — the biggest single lever available is simply not needing that hardware class at all, unlike sibling model-serving systems in this playbook.
- **Right-sized Redis footprint**: only cache "currently relevant" assignments (Section 6 estimate ~27GB working set) rather than materializing all historical assignments — expired/concluded-experiment entries are evicted via TTL sweep.

## 30. Operational Concerns (Deployment, Reliability, Infra)

At SDE2 scope, treat this as a checklist rather than a design exercise: **backups** (automated snapshots of the model registry, feature store, and any stateful service, with a tested restore path), **rollback** (every deploy must be revertible to the last-known-good version — the model registry and CI/CD pipeline should make this a one-command operation), **canary/blue-green rollout** (shift a small percentage of traffic first, watch error rate and key business/model metrics, then ramp), and **basic observability** (dashboards + alerts on latency, error rate, and the top 2-3 model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level infra-architecture concerns — worth knowing they exist, not worth rehearsing the manifests.

## 38. Why This Architecture

- Deterministic hashing (not a database lookup) for the core assignment decision means correctness doesn't depend on cache/network availability — the platform can degrade gracefully to client-side computation, which is non-negotiable given the 10ms/99.99%-availability constraint on a system that sits in front of gameplay.
- Separating the streaming guardrail path (Flink, near-real-time) from the batch scorecard path (Spark/Trino, thorough statistical rigor) matches the actual latency requirements of each: guardrails need to catch a regression within minutes to stop revenue/quality damage; scorecards need CUPED variance reduction, sequential-boundary math, and SRM checks that are computationally heavier and can tolerate a nightly cadence.
- Columnar storage choices (Iceberg/Parquet for raw events, ClickHouse for scorecards) are driven directly by the query patterns: massive scan-and-filter for backfill/audit (Iceberg), fast time-series rollup for dashboards (ClickHouse) — a single one-size-fits-all store (e.g., all-Postgres) would not scale to 10+ PB nor serve sub-second dashboard queries.
- Config as the single source of truth in Postgres, replicated everywhere via CDC, keeps the "who can see/change an experiment" story simple and auditable while still letting the hot path (Assignment Service) never make a synchronous cross-region call to Postgres.

## 39. Alternative Architectures

| Alternative | Description | Why rejected / when preferred |
|---|---|---|
| Fully server-authoritative assignment (no client-side deterministic fallback) | Every assignment requires a live server round-trip, no offline hashing | Rejected: violates the 99.99% availability / <10ms requirement when a region degrades; would stall session start. Preferred only if the game has no offline/console-network-partition concerns (e.g., a pure web app) |
| Single global synchronous database for assignment (strongly consistent global reads/writes) | e.g., Spanner-style globally consistent store for assignment state | Rejected at this QPS/latency budget: cross-region consistency adds tens of ms, unacceptable for hot path; deterministic hash + regional cache achieves the same "same unit always gets same variant" guarantee without the network cost. Preferred if assignment needed to be dynamically mutable per-request (it doesn't — it's a pure function of unit_id + config) |
| Vendor-only experimentation SaaS (e.g., LaunchDarkly/Optimizely-style, external) | Buy instead of build | Rejected at EA's scale/cost/compliance needs: 10PB-scale telemetry ownership, deep integration with proprietary Frostbite SDKs and existing telemetry pipeline, and monetization-sensitive audit requirements make a fully external vendor costly and control-limited. Preferred for smaller studios/early-stage products without in-house data infra |
| Bayesian-only analysis (no frequentist/sequential hybrid) | Pure Bayesian A/B testing (posterior probability of superiority) | Not rejected outright — offered as an option in the stats engine — but not the sole method, because EA's finance/compliance stakeholders are more familiar with p-values/CIs for revenue-impacting decisions, and mSPRT sequential testing gives "peekability" guarantees without needing full Bayesian buy-in org-wide |

## 40. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Deterministic hash assignment (vs. DB-backed assignment) | No network dependency, infinitely scalable, reproducible | Changing the hash function requires careful versioning (Section 9/32); can't do arbitrary dynamic re-assignment |
| Client-side assignment caching | Cuts server QPS ~10x, resilient to backend outages | Client can serve a stale assignment if config changes mid-session (acceptable: experiments shouldn't flip variants mid-session anyway) |
| At-least-once event delivery + downstream dedup | Simpler, cheaper than exactly-once transactional pipelines | Requires disciplined idempotency-key handling everywhere; a dedup bug silently double-counts a metric |
| Regional-first ingestion + async global replication | Low regional ingest latency, resilient to cross-region network issues | Global scorecards have a replication-lag blind spot (typically seconds-minutes, but a spike could delay a same-day guardrail catch for cross-region-relevant metrics) |
| Nightly batch scorecards + streaming guardrails split | Right latency profile for each use case | Two separate codepaths/statistics implementations to keep consistent (risk of guardrail math and scorecard math silently diverging) |
| Single-writer Config Store (not multi-master) | No conflict resolution complexity, unambiguous allocation truth | Regional write availability during a primary-region outage requires failover promotion (not instant active-active writes) |

## 41. Failure Modes

| Scenario | Impact | Mitigation |
|---|---|---|
| Redis cache cluster total outage | Assignment latency spikes (recompute from config + hash every request) but correctness unaffected | Deterministic hash means cache is pure optimization; alert + auto-scale Assignment Service pods to absorb extra CPU from recompute |
| Kafka broker/cluster outage in a region | Exposure/metric events buffered client-side or dropped after buffer limit; guardrail detection blind in that region | Client SDK local buffering with bounded retry; cross-region failover for ingestion if regional Kafka is down long enough; alert loudly since guardrail blindness is high-risk |
| Config CDC pipeline lag/failure | Stale experiment configs served (e.g., a "pause" doesn't propagate) | Alert on CDC lag > 60s (Section 23); Assignment Service exposes "config age" in response headers for debuggability; hard cutoff — if config age > 5 min, Assignment Service refuses to serve *new* experiment assignments (fails safe to "no experiment, default behavior") while still serving already-cached ones |
| Hash-function bug ships (breaks reproducibility) | Players re-randomized mid-experiment, silently invalidating all running experiments | Version-gated hash function (Section 9/32), canary synthetic-assignment check (Section 33) specifically tests for this before rollout |
| Guardrail Monitor Flink job crash-loops | No auto-pause safety net for running experiments | Guardrail Monitor health itself becomes a guardrail — if the monitor is down > 5 min, page immediately (Section 23) and consider auto-pausing all "new" experiment starts platform-wide as a precaution |
| SRM undetected due to stats bug | Team draws false conclusions from a broken experiment, ships a bad model/feature | SRM check is mandatory, non-skippable in scorecard generation; scorecards visually flag SRM failure in red in the Console UI, and API responses include `srm_check.passed=false` explicitly so downstream automation can gate on it |
| ML holdout accidentally "graduated" (merged back into treatment) by a config mistake | Loses the ability to measure long-run causal model impact — expensive, can't be undone retroactively | Holdout cohort changes require elevated RBAC role + audit log + confirmation step distinct from normal experiment edits; holdout membership changes trigger a mandatory Slack notification to the ML model's owning team |

## 42. Scaling Bottlenecks

- **At 10x (≈40M concurrent, ~30M QPS naive / ~2-3M QPS post-caching)**: Redis Cluster shard count becomes the first bottleneck (working set grows roughly linearly with MAU × active experiments) — mitigated by adding shards, but cross-shard hot-key risk emerges if one wildly popular title dominates traffic; needs per-title cache namespace isolation to avoid noisy-neighbor shard hotspotting.
- **At 100x**: Kafka partition count and broker fleet size for `metric-events` becomes the dominant cost and operational-complexity driver (hundreds of millions of events/sec) — likely requires moving to a tiered/regional-topic-per-title model instead of a handful of global topics, plus more aggressive sampling of low-value telemetry (not every session event needs to be a full-fidelity experiment metric).
- **Config Service single-writer Postgres**: fine at current admin-operation volume; would become a bottleneck only if experiment creation/editing itself became extremely high-frequency (e.g., automated ML systems creating thousands of micro-experiments per hour) — at that point, a queue-based async config-apply model would replace direct synchronous writes.
- **Batch Stats Engine**: scales roughly linearly with (number of experiments × metrics), so at 100x experiment count (tens of thousands concurrently) nightly batch windows would no longer fit in an overnight window — would require moving toward incremental/streaming scorecard computation rather than full nightly recompute.

## 43. Latency Bottlenecks

**Assignment request, p50/p99 budget breakdown (target total p99 < 10ms):**

| Step | p50 | p99 |
|---|---|---|
| Network (client → regional gateway) | 1.5 ms | 3 ms |
| Auth/token validation (gateway) | 0.3 ms | 0.8 ms |
| Redis cache lookup (assignment/config) | 0.4 ms | 2 ms (cross-shard or cold key) |
| Targeting-rule evaluation + hash compute | 0.1 ms | 0.3 ms |
| Write-through cache update (async-ish, fire-and-forget) | ~0 ms (non-blocking) | ~0 ms |
| Response serialization + network return | 1.5 ms | 3 ms |
| **Total** | **~3.8 ms** | **~9.1 ms** |

The dominant p99 tail contributors are network RTT (unavoidable physical distance component, mitigated only by regional presence) and cache lookup tail latency (cold key / cross-shard hop) — this is why config and assignment caching (Section 11) and regional colocation (Section 31) are the two highest-leverage latency investments, not the hash computation itself (negligible CPU cost).

## 44. Cost Bottlenecks

- **Raw event storage** (10+ PB over 13-month retention) is the single largest steady-state cost line — driven by telemetry volume, not by the platform's compute. Tiered storage (Section 29) is the primary lever.
- **Kafka cluster fleet size** (broker count × storage × cross-AZ replication traffic) scales with ingest throughput (3-5M events/sec peak) — cross-AZ data transfer specifically is an underappreciated cost driver in cloud-hosted Kafka.
- **Batch Stats Engine compute** scales with (experiments × metrics × query complexity) — CUPED/sequential-boundary computations are more expensive than simple mean-difference tests; spot-instance usage (Section 29) is the main mitigant.
- **Redis cluster** cost scales with working-set size (Section 6 estimate ~27GB) — comparatively small relative to storage/Kafka, not a top-3 cost driver, though under-provisioning it would create the latency bottleneck described in Section 43.
- Notably **not** a cost driver: GPU/model-serving compute (Section 14) — this system's cost profile looks like a data/telemetry-infrastructure system, not an ML-inference system, which is a useful contrast point vs. other systems in this playbook (e.g., a real-time recommendation service).

## 45. Interview Follow-Up Questions

1. How do you prevent two overlapping experiments (in different layers) from confounding each other's results?
2. How would you detect and handle a "novelty effect" where a treatment looks good in week 1 but regresses to baseline by week 4?
3. Walk through how sequential testing (mSPRT) avoids inflating false-positive rate when teams peek at results daily — why doesn't a naive repeated-fixed-horizon-test approach work?
4. How do you handle experiments where the unit of randomization is a match (multiple players) rather than an individual player — what statistical complication does this introduce?
5. A guardrail metric auto-pauses an experiment at 2 AM. Walk through what happens next, end-to-end, including who gets paged and how the rollback propagates.
6. How would you extend this platform to support multi-armed bandit allocation (dynamic traffic reallocation toward the winning variant) instead of fixed-split A/B?
7. How do you validate that your deterministic hash function actually produces a statistically uniform, unbiased distribution across variants?
8. If a model-serving team wants a permanent 5% ML holdout that must survive across a dozen future model version upgrades, what specifically in your design guarantees that holdout membership doesn't leak or get accidentally reset?
9. How would you extend guardrail metrics to catch a *slow-burn* regression (e.g., 0.5%/day degradation over 3 weeks) that no single 15-minute window would flag?
10. What's your approach to experiments that need to respect regional regulatory constraints (e.g., certain monetization mechanics restricted in specific countries)?

## 46. Ideal Answers

1. **Overlapping experiment confounding**: Use mutually-exclusive **layers** — a player is randomized into exactly one experiment per layer via independent hash salts, so experiments in the same layer can't overlap by construction. Cross-layer interactions are caught via periodic interaction-detection analysis (compare metric lift for units in experiment A alone vs. units in both A and B).

2. **Novelty effects**: Track metric lift as a time series (not just a single aggregate) and look for a lift trend that decays over the experiment's duration. A persistent ML holdout (Section 8/15) lets you compare "week 1" vs. "week 8" treatment effect to distinguish novelty from durable value.

3. **Sequential testing (mSPRT)**: A fixed-horizon test's p-value is only valid at one pre-specified sample size; repeatedly checking and stopping at p<0.05 is "peeking," which inflates the false-positive rate well above 5%. mSPRT controls Type-I error *uniformly over time*, so it's valid to check after every new data point without inflating false positives.

4. **Match-level (cluster) randomization**: When randomizing at match_id, individual player outcomes within the same match are correlated (shared map, opponents), violating the i.i.d. assumption behind standard variance estimates. This requires cluster-robust variance estimation — treating each player as an independent sample when only matches were randomized understates variance and inflates significance.

5. **2 AM guardrail auto-pause walkthrough**: Guardrail Monitor detects a breach (e.g., crash rate +20%) and immediately calls Config Service to set treatment allocation to 0%, propagating via CDC within ~2s so new sessions stop entering treatment. On-call and the experiment owner are paged/notified, but resumption requires human investigation — no auto-resume (Section 34).

6. **Multi-armed bandit extension**: Add a bandit-allocation mode where variant allocation is recomputed periodically (e.g., hourly) by a bandit policy (Thompson sampling/UCB) reading recent reward metrics, pushed through the same CDC path as manual changes. Key tension: bandits break the "fixed allocation" assumption behind frequentist scorecards, so bandit experiments need Bayesian/adaptive analysis instead.

7. **Validating hash uniformity**: Run an offline harness that hashes millions of synthetic unit_ids through the production hash function and chi-square-tests the bucket distribution against uniform, plus continuously monitor SRM (Section 21) in production as a live check of the same property.

8. **Guaranteeing holdout persistence across model versions**: Holdout membership is stored as its own first-class entity (`holdout_cohort`), keyed by `unit_id` and a stable `holdout_id` independent of model version, so upgrades reference "respect holdout X" rather than re-deriving membership. Membership changes require elevated RBAC + mandatory notification to prevent silent "graduation."

9. **Slow-burn regression detection**: Complement the fast 15-minute guardrail window with a longer-window check (e.g., 7-day trailing average vs. 30-day-prior baseline) using CUSUM change-point detection to catch small sustained shifts that an instant-value threshold would miss.

10. **Regional regulatory constraints**: Targeting rules (Section 8) support region-exclusion lists evaluated before bucketing, so a player in a restricted region is never entered into the experiment at all rather than being filtered out post-hoc. This exclusion logic carries the same audit-log and elevated-approval requirements as other monetization-sensitive changes (Section 25).
