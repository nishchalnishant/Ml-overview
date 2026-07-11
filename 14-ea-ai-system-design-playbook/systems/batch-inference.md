# Batch Inference Platform

## 1. Problem Framing & Requirement Gathering

- Build a **large-scale scheduled scoring platform** at EA that runs ML models over entire populations of entities (players, accounts, matches, items, sessions) on a recurring or ad-hoc basis, producing predictions consumed by downstream systems — churn-prevention campaigns, matchmaking priors, LiveOps offer targeting, anti-cheat batch re-scoring, toxicity re-classification of historical chat logs, recommendation candidate generation for EA App/Origin storefront.
- Distinguish from **online/real-time inference**: no per-request SLA to a live game client — the unit of work is a *job* over millions-to-billions of rows, and the interesting system-design surface is **distributed compute orchestration (Spark/Ray), data partitioning, cost-optimized cluster scheduling, backfills, and idempotent large-scale writes** — not p99 request latency.
- Primary consumers: LiveOps/CRM teams (nightly churn scores feed email/push campaigns), anti-cheat team (weekly full-population re-scoring when a new cheat-detection model ships), personalization team (daily recommendation refresh), analytics/BI (feature backfills for new model versions), trust & safety (re-scoring historical UGC/chat against updated moderation models).
- Scale anchor: EA has ~700M registered accounts, ~30-45M MAU across live-service titles (FC 24/25, Apex Legends, Madden, The Sims, Battlefield); a "full population" batch job commonly scores 200-500M account-title pairs per run.
- Framing distinguishes this from a training pipeline: batch inference reads a **trained, frozen model artifact** and a large **feature/data snapshot**, and writes predictions at scale — it does not update model weights.
- Core tension to design around: **cost** (idle GPU/CPU capacity is the #1 line item) vs **freshness** (LiveOps wants yesterday's churn score by 6am) vs **correctness under partial failure** (a 6-hour job over 400M rows *will* have node preemptions, skewed partitions, and transient storage errors — the design must make failures cheap to recover from, not catastrophic).

## 2. Functional Requirements

- FR1: Accept a **job definition** (model version, input dataset(s)/time window, output destination, schedule or trigger) and execute it as a distributed batch job.
- FR2: Support **scheduled recurring jobs** (cron-like: hourly/daily/weekly) and **ad-hoc/on-demand jobs** (triggered via API or UI, e.g. "rescore all EU players against the new anti-cheat model").
- FR3: **Partition** input data (by region, title, account-id hash, time window) to parallelize across a distributed compute cluster (Spark on EMR/Databricks-style, or Ray for Python-native model code).
- FR4: Support **backfills** — re-running a job over historical date ranges (e.g., re-score the last 90 days after a moderation-model fix), with idempotent writes so partial/replayed backfills don't double-count or duplicate rows.
- FR5: Support **multiple model types** in the same platform: tabular (XGBoost/LightGBM churn models), deep learning (embeddings-based recsys, PyTorch), and GPU-heavy models (toxicity/vision classifiers on chat/screenshots).
- FR6: **Cost-optimize scheduling** — pack jobs onto spot/preemptible capacity where tolerable, defer non-urgent jobs to off-peak windows, autoscale cluster size to job size.
- FR7: Provide **job-level and row-level observability**: did the job complete, what fraction of rows failed to score, data-quality checks on inputs/outputs.
- FR8: Support **incremental scoring** — only re-score entities whose features changed since last run (delta detection) to cut cost for large-but-mostly-static populations.
- FR9: Write outputs to systems downstream systems can consume at their required latency/format: data warehouse (Snowflake/BigQuery-style columnar store), feature store (online sink for later real-time use), or direct export (CRM/email platform, S3/Parquet for BI).
- FR10: Provide **retry, dead-letter, and partial-success semantics** — a job with 0.4% row-level failures should not fail the whole job, but those rows must be tracked and retriable.
- FR11: Support **multi-tenancy** — anti-cheat, LiveOps, and personalization teams share the platform but get isolated compute quotas, priority tiers, and cost attribution.
- FR12: Expose a **job orchestration API/UI** (submit, cancel, view status, view lineage: which model version + which data version produced which output version).

## 3. Non-Functional Requirements (latency, availability, throughput, consistency, cost)

| Dimension | Target |
|---|---|
| Job completion SLA (nightly churn job, ~350M rows) | p50 ≤ 90 min, p99 ≤ 180 min (must finish before 6am campaign send) |
| Job completion SLA (weekly full re-score, ~600M rows) | ≤ 6h wall clock |
| Ad-hoc job submission → cluster acquisition | p99 ≤ 5 min (from spot capacity pool) |
| Orchestration control-plane availability | 99.95% (scheduler must reliably fire jobs) |
| Data-plane (compute cluster) availability | best-effort per job; platform tolerates node loss without job failure |
| Row-level scoring success rate | ≥ 99.5% of rows scored per run; failed rows retried within same run or flagged for next run |
| Throughput (rows scored / hour, tabular model, CPU cluster) | 4-6M rows/hr per 100-node Spark cluster (XGBoost inference) |
| Throughput (rows scored / hour, DL embedding model, GPU cluster) | 8-12M rows/hr per 32×A10G nodes (batched inference) |
| Consistency model | Output is a **versioned snapshot** (append-only, partitioned by run_id/date) — not mutated in place; downstream reads pick a specific run_id or "latest complete" pointer |
| Backfill correctness | Idempotent: re-running the same (model_version, data_version, partition) produces byte-identical or checksummed-equal output; safe to replay |
| Cost | ≤ $0.00004 per row scored (tabular, blended spot/on-demand); ≤ $0.0006 per row (GPU DL models) — see Section 6/29 |
| Data freshness (feature snapshot → job start) | ≤ 4h staleness for daily jobs, ≤ 15 min for "fast lane" LiveOps flash-sale targeting jobs |

## 4. Clarifying Questions an Interviewer Would Expect You to Ask

1. What's the largest single job in scope — full 700M-account population, or per-title (30-45M MAU) subsets?
2. Is this multi-tenant across teams (anti-cheat, LiveOps, personalization) on shared infra, or a dedicated platform per team?
3. What model types must be supported — pure tabular (CPU-friendly) or also GPU-bound DL/embedding models?
4. What's the hard deadline pressure — is there a "campaign must go out by 6am" constraint, or is this pure batch analytics with loose SLAs?
5. Do we own Spark/Ray infra (self-managed on EKS/EMR) or use a managed service (Databricks, SageMaker Batch Transform, Vertex AI Batch)?
6. What's the backfill frequency and typical window size — do we need to replay 90 days routinely, or is that a rare emergency operation?
7. Is incremental/delta scoring required, or is full re-score every run acceptable given cost budget?
8. What's the downstream consumption pattern — warehouse-only (BI), or does batch output feed a low-latency online store (feature store) for real-time personalization?
9. What's the cost ceiling / who owns the cloud bill — per-team chargeback or centralized platform budget?
10. Are there compliance constraints on batch-processing player data (GDPR erasure requests mid-backfill, data residency per region for EU accounts)?
11. What's acceptable staleness of the model artifact used — must every job pin an exact model version, and how is that governed?

## 5. Assumptions

1. Total addressable population: 700M registered accounts; active scoring population (played in last 90 days) ≈ 400M accounts across titles.
2. Nightly churn-prediction job scores ~350M active-account rows (tabular, XGBoost, ~180 features/row).
3. Weekly anti-cheat re-score job scores ~600M account-match rows (DL classifier, GPU).
4. Daily recsys candidate-generation job scores ~40M MAU × ~500 candidate items = 20B (account, item) pairs, batched via embedding dot-product (ANN-assisted, not brute-force where avoidable).
5. Average row size for tabular feature vector: ~2KB (raw) / ~400 bytes (compressed columnar).
6. Compute substrate: Spark on Kubernetes (EKS) for tabular/ETL-heavy jobs; Ray on Kubernetes for Python-native DL batch inference (better GPU utilization, less JVM overhead for tensor ops).
7. Storage: input feature snapshots and outputs live in a data lakehouse (S3 + Delta Lake/Iceberg format) partitioned by date and region.
8. Model artifacts are versioned and stored in an internal model registry (MLflow-style); batch jobs pin an explicit model_version at submission time.
9. Spot/preemptible instances tolerated for ≥ 80% of compute (checkpointable, restartable jobs); on-demand reserved for the tail 20% to hit SLA when spot capacity is scarce.
10. Jobs run in 3 priority tiers: P0 (LiveOps campaign-blocking, SLA-critical), P1 (standard nightly/weekly), P2 (best-effort backfills/experiments).
11. Regional data residency: EU player data processed in eu-west, not shipped to us-east clusters (GDPR).

## 6. Capacity Estimation (QPS, storage, model size, GPU/CPU counts, back-of-envelope math shown)

**Nightly churn job (tabular, CPU):**
- Rows: 350M accounts. Features/row: 180 (avg 400B compressed) → input size ≈ 350M × 400B ≈ 140GB compressed (Parquet/Delta), ~700GB uncompressed in-memory during shuffle.
- XGBoost inference throughput: ~15,000 rows/sec/core (simple tree ensemble, ~500 trees, depth 6). Target 90-min completion → need 350M / (90×60) ≈ 65,000 rows/sec aggregate.
- Cores needed: 65,000 / 15,000 ≈ 5 cores... but real Spark overhead (shuffle, serialization, GC) cuts effective throughput ~3-4x → budget for ~20-24 effective "core-equivalents" fully saturated, then scale out for parallel partition scanning: practically, **100-node cluster, 8 vCPU/node (800 vCPUs total)**, each node handling ~3.5M rows in ~85 min including I/O.
- Output size: 350M rows × (account_id + 3 scores + metadata ≈ 60B) ≈ 21GB written, partitioned by region/date.

**Weekly anti-cheat re-score (GPU, DL):**
- Rows: 600M account-match pairs. Model: mid-size sequence classifier (~40M params), batched inference.
- Throughput per A10G GPU (batch size 512, seq len 128): ~2,000 rows/sec.
- Target 6h window → need 600M / (6×3600) ≈ 27,800 rows/sec aggregate → 27,800 / 2,000 ≈ 14 GPUs minimum; provision **32×A10G** (Ray cluster) for headroom, preemption tolerance, and to also absorb feature-join/shuffle overhead (data movement, not just compute, dominates at this scale).
- Memory: model ~160MB (fp16) — trivially fits per-GPU; the bottleneck is I/O (reading 600M rows of match telemetry, ~1.5KB/row ≈ 900GB from lakehouse) not GPU compute.

**Daily recsys candidate scoring:**
- 40M MAU × 500 candidates = 20B pairs if brute-force — instead use two-tower embeddings: compute 40M user embeddings (batch DL inference, GPU) + maintain ~50K item embeddings in an ANN index, retrieve top-500 via ANN (not full dot-product over 20B pairs).
- User-embedding batch: 40M rows, ~5,000 rows/sec/GPU (embedding tower forward pass) → 40M/5,000 ≈ 8,000 GPU-seconds ≈ 2.2 GPU-hours → **8×A10G for ~20 min** wall clock, easily fits daily cadence.
- This reframes an apparent 20B-row job into a 40M-row embedding job + ANN lookup — the single biggest capacity lever in this system is **avoiding brute-force cross joins**.

**Storage estimation (90-day retention of run outputs, all jobs):**
- Nightly churn: 21GB/run × 90 ≈ 1.9TB.
- Weekly anti-cheat: ~600M × 60B ≈ 36GB/run × 13 runs/quarter ≈ 470GB.
- Daily recsys: 40M users × top-500 item-ids × 8B ≈ 160GB/run × 90 ≈ 14.4TB (dominant storage cost — mitigated by storing top-50 not top-500, and TTL-ing older runs to 14 days, cutting to ~2.2TB).
- Total steady-state lakehouse footprint for batch-inference outputs: **~5-8TB** after realistic retention tuning, plus input feature snapshots (~140GB/day × 30-day retention ≈ 4.2TB).

**Cluster count / cost back-of-envelope (see Section 29 for full breakdown):**
- ~800 vCPUs × ~14h/day average utilization (nightly + ad-hoc) on 70% spot ≈ blended $0.025/vCPU-hr effective → ~$280/day CPU compute.
- 32×A10G × ~8h/week (anti-cheat) + 8×A10G × 1h/day (recsys) ≈ (256 + 240) GPU-hours/week × ~$1.10/GPU-hr (spot A10G) ≈ ~$545/week GPU compute.

## 7. High-Level Architecture (with an ASCII diagram showing all major components and data flow)

```
                         ┌─────────────────────────────────────────────┐
                         │           Job Orchestration Layer            │
                         │  (Airflow / Dagster-style DAG scheduler)      │
                         │  - cron schedules, ad-hoc API triggers        │
                         │  - job DAG: extract → partition → score →     │
                         │    validate → write → notify                 │
                         └───────────────┬───────────────────────────────┘
                                         │ submits job spec (model_version,
                                         │ data_version, partition plan)
                                         ▼
                         ┌─────────────────────────────────────────────┐
                         │        Cluster / Resource Manager             │
                         │  (Kubernetes + Karpenter/Cluster Autoscaler)  │
                         │  - spins Spark-on-K8s or Ray clusters         │
                         │  - spot-first bin-packing, priority queues    │
                         └───────────────┬───────────────────────────────┘
                                         │
                 ┌───────────────────────┼───────────────────────────┐
                 ▼                       ▼                           ▼
        ┌─────────────────┐   ┌───────────────────┐        ┌──────────────────┐
        │  Spark Cluster   │   │   Ray Cluster      │        │ Model Registry    │
        │  (tabular jobs:  │   │  (DL/GPU batch     │◄───────┤ (MLflow-style)    │
        │  churn, ETL)     │   │  inference: recsys,│  fetch │ pinned model_ver  │
        │                  │   │  anti-cheat, DL)   │  model │ artifacts + meta  │
        └────────┬─────────┘   └─────────┬──────────┘        └──────────────────┘
                 │ read partitions                 │ read partitions
                 ▼                                 ▼
        ┌───────────────────────────────────────────────────────┐
        │              Data Lakehouse (S3 + Iceberg/Delta)        │
        │  - feature snapshots (partitioned: region/date/title)   │
        │  - raw telemetry (match logs, chat, purchase events)     │
        │  - versioned, time-travel enabled for point-in-time reads│
        └───────────────┬───────────────────────────┬─────────────┘
                         │ writes scored output       │ reads (offline
                         ▼ (append-only, run_id part.) │ features)
        ┌───────────────────────────────────────────┐ │
        │     Output Store (partitioned by run_id)    │ │
        │  - churn_scores/, anticheat_flags/,          │◄┘
        │    recsys_candidates/                        │
        └───────────┬───────────────────┬───────────────┘
                     │                   │
                     ▼                   ▼
          ┌────────────────────┐  ┌──────────────────────┐
          │  Feature Store      │  │  Data Warehouse /BI    │
          │  (online sink, e.g. │  │  (Snowflake-style,     │
          │  Redis/DynamoDB)    │  │  analytics queries)    │
          │  - "latest churn    │  │  - CRM export           │
          │    score per user"  │  │  - anti-cheat dashboards│
          └─────────┬──────────┘  └──────────────────────┘
                     │
                     ▼
          ┌────────────────────────────┐
          │ Downstream Consumers         │
          │ - LiveOps CRM/email trigger  │
          │ - Real-time personalization  │
          │   service (reads latest score)│
          │ - Anti-cheat action pipeline  │
          └────────────────────────────┘

   Cross-cutting: Monitoring/Alerting, Data-Quality Validator, Lineage/Metadata
   Store (job → model_version → data_version → output_version) attach to every
   stage above.
```

## 8. Low-Level Components (each major service/component explained: responsibility, interface, scaling unit)

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| **Job Orchestrator** (Airflow/Dagster) | DAG scheduling, dependency resolution, retries, backfill triggering | REST/gRPC API + DSL (Python DAG defs); cron + event triggers | Scheduler replicas (HA pair); worker pool for task execution |
| **Cluster Manager** | Provisions Spark/Ray clusters on K8s, spot/on-demand mix, priority queues | Kubernetes CRDs (SparkApplication, RayCluster) | Node pool autoscaling (Karpenter), per node-group |
| **Partitioner/Planner** | Splits input dataset into balanced partitions (by account-hash, region, time) avoiding skew | Internal library called at job-plan time; reads partition stats from lakehouse metadata | Runs once per job (planning step), cheap — not a scaling bottleneck itself |
| **Spark Executors** | Distributed tabular scoring (UDF-wrapped model inference), joins, aggregations | Spark DataFrame API; model loaded as broadcast variable or Pandas UDF | Executor count = f(partition count, target parallelism) |
| **Ray Workers** | GPU-batched DL inference, embedding computation | Ray Dataset API / Ray Data batch_predict actors | GPU-node count; actor pool size tuned to batch size × GPU memory |
| **Model Registry** | Version, store, and serve model artifacts + metadata (framework, input schema, training data version) | REST API (`get_model(name, version)`), artifact store (S3) | Read-heavy, cached at cluster-node level; registry itself is low-QPS |
| **Data Lakehouse** | Source of truth for feature snapshots and raw telemetry; time-travel for point-in-time correctness | Iceberg/Delta table API (SQL + Spark/Ray connectors) | Storage scales independently (S3); compute is the Spark/Ray layer |
| **Data-Quality Validator** | Pre-job input checks (schema, null-rate, row-count sanity) and post-job output checks (score distribution drift, row-count reconciliation) | Great-Expectations-style rule engine, runs as DAG task | Scales with job count, lightweight per job |
| **Output Writer** | Idempotent, versioned writes to output store (append-only by run_id, atomic commit/rollback via Iceberg transactions) | Iceberg/Delta write API with `MERGE`/overwrite-partition semantics | Scales with Spark/Ray executor count |
| **Feature-Store Sink** | Pushes "latest score" into online KV store for real-time consumption | Batch-write client (bulk PUT to Redis/DynamoDB) | Sharded by account-id hash |
| **Lineage/Metadata Store** | Tracks (job_id → model_version, data_version, output_version, row-counts, timing) | Internal metadata API, backed by Postgres | Low write volume (1 record/job), simple to scale |
| **Notification/Alerting Hook** | Job success/failure/SLA-breach notifications | Webhook to Slack/PagerDuty | N/A — event-driven, stateless |

## 9. API Design (concrete endpoint signatures, request/response schemas, versioning)

```
POST /v1/jobs
Request:
{
  "job_name": "nightly_churn_score",
  "model_ref": {"name": "churn_xgb", "version": "17"},
  "input_spec": {
    "table": "lakehouse.features.player_activity_snapshot",
    "partition_filter": {"date": "2026-07-08", "region": ["NA","EU","APAC"]}
  },
  "output_spec": {
    "table": "lakehouse.outputs.churn_scores",
    "mode": "append_versioned",
    "sink_online": true
  },
  "compute_profile": "cpu-standard-100node",
  "priority": "P0",
  "sla_deadline": "2026-07-09T06:00:00Z",
  "idempotency_key": "churn-2026-07-08-nightly"
}

Response: 202 Accepted
{
  "job_id": "job_9f3a...",
  "status": "QUEUED",
  "estimated_start": "2026-07-09T02:10:00Z"
}
```

```
GET /v1/jobs/{job_id}
Response:
{
  "job_id": "job_9f3a...",
  "status": "RUNNING",           // QUEUED|RUNNING|SUCCEEDED|FAILED|PARTIAL_SUCCESS
  "progress": {"partitions_total": 400, "partitions_done": 267},
  "rows_scored": 233000000,
  "rows_failed": 812000,
  "started_at": "...", "eta": "...",
  "output_version": "run_2026070902"
}
```

```
POST /v1/jobs/{job_id}/backfill
Request:
{
  "date_range": {"start": "2026-04-01", "end": "2026-06-30"},
  "model_ref": {"name": "toxicity_clf", "version": "9"},
  "reason": "moderation model bugfix reprocessing",
  "idempotency_key": "toxicity-backfill-2026Q2-v9"
}
Response: 202 Accepted { "backfill_job_group_id": "bfg_1122...", "sub_jobs": 91 }
```

```
POST /v1/jobs/{job_id}/cancel        → 200 { "status": "CANCELLING" }
GET  /v1/jobs?status=FAILED&team=anticheat&since=2026-07-01  → paginated list
GET  /v1/lineage/{output_version}    → { model_version, data_version, job_id, row_count, checksum }
```

- **Versioning**: URI-versioned (`/v1/...`); job-spec schema is itself versioned (`schema_version` field) so the orchestrator can evolve request shape without breaking existing DAG definitions; model/data refs are explicit and immutable (no "latest" in production job specs — "latest" only allowed in dev/staging submissions).
- **Idempotency**: `idempotency_key` required for all mutating POSTs; duplicate submission within a 24h window returns the original job's `job_id` (200, not 202) instead of double-scheduling.

## 10. Database Design (schema sketches, choice of SQL/NoSQL/columnar and why, partitioning/sharding key)

**Lakehouse tables (Iceberg/Delta on S3 — columnar Parquet, chosen for large-scan batch reads, schema evolution, and time-travel):**

```sql
-- Feature snapshot (input) — columnar, partitioned for scan pruning
CREATE TABLE lakehouse.features.player_activity_snapshot (
  account_id        BIGINT,
  region            STRING,
  title             STRING,
  snapshot_date     DATE,
  feature_vector    STRUCT<sessions_7d: INT, spend_30d: DOUBLE, ...>,  -- ~180 fields
  feature_version   STRING
)
PARTITIONED BY (snapshot_date, region);   -- partition pruning: jobs read 1 day, 1-3 regions

-- Output (scored predictions) — append-only, versioned by run_id
CREATE TABLE lakehouse.outputs.churn_scores (
  account_id      BIGINT,
  churn_score     DOUBLE,
  model_version   STRING,
  run_id          STRING,
  scored_at       TIMESTAMP
)
PARTITIONED BY (run_id);   -- each run isolated; "latest" resolved via lineage pointer, never overwritten in place
```

- **Why columnar (Parquet/Iceberg) over row-store**: batch jobs scan/aggregate over hundreds of millions of rows but touch few columns per pass (e.g., 20 of 180 features for a given model) — columnar storage + partition/column pruning is a 5-10x I/O win over a row-oriented store for this access pattern.
- **Partitioning key**: `(snapshot_date, region)` on inputs — matches the dominant query pattern (score "today's" data for a region-scoped job) and satisfies GDPR data-residency (EU partitions never leave eu-west compute).
- **Output partitioned by `run_id`**: makes every run immutable and independently reproducible/deletable; "current" output is a pointer in the lineage metadata store, not a mutation — this is what makes backfills and rollbacks safe (Section 34).
- **Lineage/metadata store**: Postgres (OLTP, low volume, needs referential integrity: job → model_version → data_version → output_version, foreign-keyed). Not a good fit for the lakehouse (needs point lookups + joins across few thousand rows/day, not scans).
- **Online sink (feature store)**: DynamoDB/Redis, sharded by `account_id` hash — optimized for point-lookup ("give me user X's latest churn score in <5ms") which columnar lakehouse cannot serve.
- **Sharding for online sink**: consistent hashing on `account_id % N shards`, N sized so each shard stays under ~50GB (operational sweet spot for Redis/Dynamo partition performance).

## 11. Caching (what's cached, cache invalidation strategy, cache-aside vs write-through)

| Cached Item | Layer | Strategy | Invalidation |
|---|---|---|---|
| Model artifact (weights/binary) | Local disk on Spark/Ray worker nodes + broadcast variable in-memory | Cache-aside: on job start, check local cache by (model_name, version, checksum); fetch from registry on miss | Never invalidated by mutation (models are immutable per version) — evicted by LRU when node disk pressure high |
| "Latest score" per account | Online feature-store (Redis) | Write-through: batch job writes new score directly to Redis at job completion (bulk pipeline write) | Overwritten by next successful run; TTL of 48h as a safety net (if job stalls, downstream treats missing/stale key as "no signal" not stale-truth) |
| Partition statistics (row counts, skew histograms) | Metadata cache in orchestrator (in-memory + Postgres) | Cache-aside, refreshed at job-plan time | TTL 1h, or explicit invalidation on upstream table schema change event |
| Job-plan (partition assignment) | In-memory within orchestrator for the duration of a running job | Write-once per job execution | Discarded at job completion; recomputed for retries/backfills (skew may have changed) |
| Feature-store read cache (if inference pipeline reads point-in-time features) | Not used here — batch jobs read directly from lakehouse in bulk scan, caching a bulk scan gains nothing | N/A | N/A |

- No general-purpose "prediction cache" (unlike online serving) — batch inference *is* the write path, not a cache-fronted read path; caching value here is almost entirely about **avoiding redundant model-artifact downloads across hundreds of worker nodes**, which is the actual hot path cost (a 2GB DL model pulled by 32 GPU nodes simultaneously can saturate registry egress without node-local caching + broadcast).

## 12. Queues & Async Processing (what's queued, at-least-once vs exactly-once, dead-letter handling)

- **Job-submission queue**: ad-hoc API submissions land on a durable queue (SQS-style) in front of the orchestrator so submission bursts (e.g., 50 backfill sub-jobs fired at once) don't overwhelm the scheduler; **at-least-once** delivery, orchestrator dedupes via `idempotency_key`.
- **Row-level retry queue**: rows that fail scoring (bad feature vector, model exception) within a job are shunted to a **dead-letter partition** rather than failing the whole job (FR10); a separate low-priority "retry sweep" job periodically reprocesses DLQ rows with relaxed validation or flags them for manual review.
- **Semantics**: batch scoring itself is designed for **exactly-once effective output** via idempotent writes (Iceberg atomic partition overwrite keyed by `run_id`) even though the underlying execution is at-least-once (Spark task retries, speculative execution) — the write commit is what's made exactly-once, not the compute.
- **Dead-letter handling**: DLQ rows tagged with `(job_id, partition, error_class, raw_row_ref)`; alert fires if DLQ rate > 1% of partition (signals systemic issue, e.g., upstream schema drift) vs isolated bad rows (< 0.5%, expected background noise from malformed telemetry).
- **Notification queue**: job completion/failure events published async to a notification topic (Section 13) so Slack/PagerDuty/CRM-trigger consumers don't block the job's critical path.

## 13. Streaming & Event-Driven Architecture (topics, event schemas, consumer groups)

- Batch inference is **not** primarily stream-driven, but it both **consumes** upstream streaming events (to build feature snapshots) and **emits** completion events downstream.

**Topics:**

| Topic | Producer | Schema (key fields) | Consumers |
|---|---|---|---|
| `player.telemetry.raw` | Game servers/clients | `{account_id, title, event_type, ts, payload}` | Feature-engineering Spark streaming job → materializes `player_activity_snapshot` daily |
| `batch_job.lifecycle` | Job Orchestrator | `{job_id, status, output_version, rows_scored, rows_failed, ts}` | Notification service, LiveOps CRM trigger, lineage store, monitoring pipeline |
| `model_registry.version_published` | Model Registry | `{model_name, version, published_at, training_data_version}` | Orchestrator (auto-trigger validation job on new model version), on-call dashboards |
| `data_quality.violation` | DQ Validator | `{job_id, check_name, severity, details}` | Alerting service, job orchestrator (can gate job from writing output on CRITICAL) |

- **Consumer groups**: feature-engineering streaming job runs as a Spark Structured Streaming consumer group with 3x parallelism of the topic's partition count for `player.telemetry.raw` (highest-volume topic, ~1M events/sec peak during live-service events); `batch_job.lifecycle` is low-volume (~hundreds/day), single consumer group per downstream system.
- Batch jobs read the **materialized output** of the streaming feature pipeline (a lakehouse table), never the raw stream directly — decouples batch-job scheduling from streaming-topic replay/retention concerns.

## 14. Model Serving (serving framework choice, batching, multi-model, hardware)

- Not "serving" in the online sense (no request/response server) — this is **batch scoring execution**, but the framework choices parallel serving concerns:
  - **Tabular models (XGBoost/LightGBM)**: loaded once per Spark executor as a broadcast variable; scored via vectorized Pandas UDF (`mapInPandas`) processing thousands of rows per micro-batch — avoids per-row Python overhead.
  - **DL models (PyTorch, embeddings, toxicity classifiers)**: run on Ray Data with `map_batches`, using **dynamic batching** (batch size tuned to GPU memory, typically 256-1024) to maximize GPU utilization; TorchScript/ONNX export used to cut Python overhead and enable mixed precision (fp16) for 1.5-2x throughput gain.
  - **Multi-model jobs**: some pipelines chain models (e.g., embedding tower → ANN candidate retrieval → lightweight re-ranker) within a single Ray DAG, avoiding intermediate writes to the lakehouse between stages.
  - **Hardware**: CPU-only (Graviton/x86 spot) for tabular; A10G/L4-class GPUs for DL batch inference (better cost/throughput than A100 for these model sizes — A100 reserved only for training, not batch inference, at this scale).
- **Autoscaling within a job**: Ray/Spark cluster sized at job-plan time based on estimated row count and per-row cost profile (Section 6 math), not reactively mid-job — batch workloads have known size up front, unlike online traffic.

## 15. Feature Store (online/offline split, point-in-time correctness)

- **Offline store**: the lakehouse feature tables (Section 10) — full historical feature snapshots, used for both batch-inference input and training-data generation.
- **Online store**: Redis/DynamoDB sink populated *by* batch jobs — holds only "latest known" feature values for low-latency real-time lookups by other systems (e.g., a live matchmaking service reading a player's latest churn-risk tier).
- **Point-in-time correctness**: every feature snapshot is stamped with `snapshot_date` and `feature_version`; batch jobs and any training pipeline reading historical data must join on an "as-of" timestamp to avoid **label/feature leakage** (e.g., don't score April churn using May's spend feature) — enforced via Iceberg time-travel queries (`AS OF snapshot_date`) rather than reading the mutable "current" table.
- Batch inference is deliberately **read-only against a frozen daily snapshot**, not against live-mutating tables, precisely so a 90-minute job sees a consistent point-in-time view even if upstream ETL is still appending new data mid-run.

## 16. Vector Database (if applicable — indexing strategy, ANN algorithm choice, else state N/A and why)

- **Applicable, narrowly**: the recsys candidate-generation batch job (Section 6) uses an ANN index over ~50K item embeddings to avoid brute-force scoring of 20B (user, item) pairs.
- **Index choice**: HNSW (via FAISS or an internal ANN service) — item catalog is small (tens of thousands) and updates infrequently (new items added ~daily), so HNSW's build cost is cheap relative to its fast, high-recall query performance; IVF-PQ not justified at this catalog size (that tradeoff pays off at 10M+ vectors, not 50K).
- **Usage pattern**: the ANN index is *read-only* during the batch job (40M user-embedding batch queries against the fixed item index); index rebuild happens as a small separate daily job when the item catalog changes, decoupled from the main user-scoring job.
- For the *other* batch jobs in this system (churn scoring, anti-cheat re-scoring) — **N/A**, no vector search involved; these are direct tabular/sequence-model inference with no retrieval step.

## 17. Embedding Pipelines (if applicable, else N/A and why)

- **Applicable** for the recsys and anti-cheat pipelines:
  - Recsys: two-tower model produces user embeddings (batch job, Section 6) and item embeddings (smaller, less frequent batch job) — embeddings persisted to the lakehouse and the ANN index, not recomputed per query.
  - Anti-cheat: player-behavior sequence encoder produces a behavioral embedding consumed by the classifier head; embeddings are an intermediate artifact within the same job, not separately persisted (no reuse case identified yet, avoids unnecessary storage/versioning overhead).
- **Embedding versioning**: user/item embeddings are tagged with the `model_version` that produced them (same lineage discipline as prediction outputs) — critical because mixing embeddings from two model versions in one ANN index silently corrupts similarity results with no visible error.
- **Not applicable** to churn scoring — pure tabular gradient-boosted model, no learned embedding representation involved.

## 18. Inference Pipelines (request lifecycle end-to-end)

```
 [Scheduler fires: 02:00 UTC, "nightly_churn_score"]
        │
        ▼
 (1) Job Orchestrator resolves job spec → validates model_version exists in
     registry, input partition exists in lakehouse, output path clear
        │
        ▼
 (2) Partitioner/Planner: scans lakehouse partition metadata (row counts,
     region skew) → builds balanced partition plan (e.g., 400 partitions
     of ~875K rows each, region-stratified)
        │
        ▼
 (3) Cluster Manager provisions Spark-on-K8s cluster (100 nodes, 70% spot)
     via Karpenter; waits for capacity or falls back to on-demand if spot
     unavailable within 5 min (SLA-protection fallback)
        │
        ▼
 (4) Spark executors: pull model_version=17 artifact (cache-aside from
     node-local disk, else fetch from registry, broadcast to all executors)
        │
        ▼
 (5) Each executor reads its assigned partitions from
     lakehouse.features.player_activity_snapshot (columnar scan, pushdown
     filters on date/region), runs Pandas UDF batch inference
        │
        ▼
 (6) Data-Quality Validator (post-score): checks output row count vs input
     row count (reconciliation), checks score-distribution drift vs last
     7 runs (catches silent model/feature bugs)
        │
        ▼
 (7) Output Writer: atomic Iceberg commit of new partition
     (run_id=run_2026070902) — all-or-nothing per job, dead-letter rows
     written to separate DLQ partition, not blocking main commit
        │
        ▼
 (8) Lineage Store: records (job_id, model_version=17,
     data_version=2026-07-08, output_version=run_2026070902, row_count,
     checksum)
        │
        ▼
 (9) Feature-Store Sink: bulk-writes "latest churn score" to Redis
     (write-through, ~350M keys, batched pipeline writes, ~15 min)
        │
        ▼
 (10) Notification: batch_job.lifecycle event published → LiveOps CRM
      trigger consumes it, kicks off targeted email campaign for
      high-churn-risk cohort
        │
        ▼
 (11) Cluster Manager tears down cluster (scale-to-zero) — cost control
```

## 19. Training Pipelines (data prep, training orchestration, distributed training if relevant)

- Out of primary scope for *this* chapter (training pipeline is a separate system), but the batch-inference platform has hard dependencies on it worth stating:
  - Training data prep reads the **same lakehouse feature tables** via the same point-in-time-correct snapshot mechanism (Section 15) — this shared contract is what prevents train/serve skew between the training pipeline and batch-inference input.
  - Distributed training (e.g., XGBoost via Spark MLlib for churn, PyTorch DDP for embeddings/anti-cheat) runs on a **separate, GPU-heavier cluster profile**, typically on-demand (not spot) because training jobs are less checkpoint-tolerant mid-epoch than embarrassingly-parallel batch scoring.
  - On training completion, the new model artifact is pushed to the Model Registry (Section 8) with an explicit version and validation-eval metrics attached; batch-inference jobs only pick up a new version when explicitly re-pointed (no auto-promotion without a gating validation job, Section 20).

## 20. Retraining Strategy (cadence, triggers)

| Model | Cadence | Trigger Conditions |
|---|---|---|
| Churn model (tabular) | Retrain monthly | Also triggered early if drift monitor (Section 21) flags PSI > 0.25 on top-10 features, or if validation AUC on a rolling holdout drops > 3pts |
| Anti-cheat classifier | Retrain on-demand (new cheat signature discovered) + quarterly scheduled refresh | Trigger: T&S team flags a new cheat pattern with < 70% detection recall on recent labeled samples |
| Recsys embeddings | Retrain weekly (item catalog + engagement patterns shift fast in live-service) | Trigger: CTR on served recommendations drops below control-group baseline by > 5% for 3 consecutive days |
| Toxicity/moderation classifier | Retrain monthly, plus emergency retrain on policy change | Trigger: false-negative rate on sampled human-reviewed escalations exceeds 8% |

- Retraining is decoupled from batch-inference scheduling — batch-inference jobs simply **consume whatever model_version is explicitly pinned**, so a retrain does not implicitly change production behavior until a validation job promotes the new version (Section 33 canary applies to batch jobs too — score a shadow partition with the new model, compare distributions, before full-population cutover).

## 21. Drift Detection (data drift, concept drift, what metrics, what thresholds)

| Drift Type | Metric | Threshold | Action |
|---|---|---|---|
| **Data drift** (input feature distribution shift) | Population Stability Index (PSI) per feature, computed each run vs 30-day rolling baseline | PSI > 0.1 = warn, PSI > 0.25 = page + auto-hold output from downstream consumption | Feature owner investigates upstream telemetry pipeline change |
| **Prediction drift** (output score distribution shift) | KL divergence / score-histogram comparison run-over-run | Shift > 2 std dev from 7-run rolling mean | DQ Validator (Section 8) flags in post-score check; job output quarantined pending review |
| **Concept drift** (model's learned relationship to ground truth decays) | Rolling precision/recall against delayed ground-truth labels (e.g., churn model checked against actual 30-day-later churn outcome) | AUC drop > 3pts over 2 consecutive evaluation windows | Trigger early retrain (Section 20) |
| **Label delay handling** | Ground truth for churn/anti-cheat arrives with lag (30 days for churn realization, days-to-weeks for confirmed cheat bans) | N/A (lag is structural) | Concept-drift checks run on a delayed evaluation job, not real-time — explicitly budgeted as a separate weekly "model-health" batch job, not blocking production scoring |
| **Partition-level skew** | Row-count and null-rate deltas per partition vs historical | > 15% deviation | Job-level alert; job can be configured to hard-fail rather than silently score on corrupted input |

- Drift detection here is itself implemented as **batch jobs** (a meta-application of this same platform) — reinforces that this platform is the natural home for both production scoring and its own model-health monitoring workloads.

## 22. Monitoring (what's monitored: infra, model quality, business metrics)

| Category | Metrics |
|---|---|
| **Infra** | Cluster utilization (CPU/GPU %), spot-preemption rate, job queue depth, time-to-cluster-acquisition, partition skew (max/min partition size ratio), executor OOM/failure rate |
| **Job execution** | Job duration vs SLA, rows scored/sec, DLQ row rate, retry count, job success/failure/partial-success rate per team/tenant |
| **Model quality** | PSI drift scores, prediction-distribution shift, delayed-ground-truth precision/recall (Section 21), calibration (predicted churn-probability vs realized rate) |
| **Cost** | $/row scored per job, spot vs on-demand mix ratio, idle-cluster time, cost-per-team (chargeback), cluster right-sizing efficiency (requested vs actually used resources) |
| **Business** | Downstream campaign conversion lift attributable to churn-score-triggered CRM sends, anti-cheat ban accuracy (appeal overturn rate), recsys CTR/engagement lift |
| **Data lineage/quality** | Schema-change events on input tables, row-count reconciliation (input vs output), freshness lag (snapshot_date vs job execution time) |

## 23. Alerting (alert conditions, thresholds, on-call routing)

| Alert | Condition | Severity | Routing |
|---|---|---|---|
| SLA breach imminent | Job progress extrapolation predicts completion > 30 min past `sla_deadline` | P1 | Page batch-platform on-call; auto-scale cluster or escalate to on-demand fallback |
| Job failed | Job status = FAILED after retries exhausted | P1 (P0-tier jobs) / P2 (others) | Page on-call for P0 tenants (LiveOps campaign-blocking); Slack-only for P2 |
| DLQ rate > 1% | Post-job DQ check | P2 | Slack to owning team, ticket auto-filed |
| Data/prediction drift threshold breach | PSI > 0.25 or prediction-distribution shift > 2σ | P2 (P1 if on a P0-tier model) | Slack to model owner; auto-hold output from CRM/downstream consumption pending sign-off |
| Cost anomaly | Daily spend > 130% of 7-day rolling average | P3 | Slack to platform-cost channel, weekly digest |
| Spot-capacity exhaustion | Cluster fails to acquire spot capacity within 10 min, repeatedly | P2 | Page infra on-call — may indicate regional capacity crunch requiring on-demand fallback policy change |
| Concept drift (delayed eval) | AUC drop > 3pts over 2 windows | P2 | Slack + ticket to ML team, feeds retraining trigger (Section 20) |

- **On-call routing**: batch-platform on-call owns infra/scheduling alerts; model-quality alerts route to the owning ML team (churn team, anti-cheat team, etc.) — platform on-call is not expected to have domain judgment on "is this drift real."

## 24. Logging (structured logging strategy, PII handling, retention)

- **Structured logs** (JSON) at three levels: orchestrator (job lifecycle events), executor (per-partition task logs, Spark/Ray driver+executor logs), and application (DQ check results, row-level error classifications).
- **Correlation**: every log line carries `job_id`, `run_id`, `partition_id`, `model_version` — enables tracing a single row's scoring path across the distributed job for debugging.
- **PII handling**: `account_id` is logged (needed for row-level debugging/lineage), but no raw PII (email, real name, payment info) ever appears in batch-inference logs — feature vectors are logged only as aggregate statistics (row counts, distribution summaries), never raw per-row payloads, in default log verbosity; a "debug mode" that logs sample raw rows requires elevated access grant and auto-redacts direct identifiers beyond `account_id`.
- **Retention**: orchestrator/lifecycle logs retained 1 year (compliance/audit); executor task logs retained 30 days (debugging window, high volume); DQ/lineage records retained indefinitely (small volume, high audit value) in the Postgres lineage store.
- **Region-scoped logging**: EU-partition job logs stored in eu-west logging infra, consistent with data-residency assumption (Section 5.11).

## 25. Security (authn/authz, data encryption, threat model specific to this system)

- **Threat model specifics for batch inference**:
  - A malicious or buggy job submission could trigger an unbounded-cost job (e.g., accidentally requesting a full 700M-row score with an expensive GPU model) — mitigated by per-tenant compute quotas and a cost-estimate gate at submission time that requires elevated approval above a threshold.
  - A compromised job-submission credential could exfiltrate bulk player data via a crafted "output_spec" pointing to an external/unauthorized sink — mitigated by an allowlist of approved output destinations per tenant, enforced at the API layer, not just IAM.
  - Cross-tenant data leakage risk: anti-cheat team's job accidentally reading LiveOps' restricted feature columns — mitigated by column-level ACLs on lakehouse tables (row/column masking policies), not just table-level grants.
  - Model-artifact tampering: a poisoned model version silently deployed to production scoring — mitigated by checksum verification on artifact fetch (Section 8) and signed model artifacts from the registry.
- **Encryption**: at-rest (S3/lakehouse: SSE-KMS; Postgres lineage store: encrypted volumes), in-transit (TLS 1.2+ between all internal services — orchestrator↔cluster manager↔lakehouse↔registry).
- **Data residency enforcement**: IAM policies + network segmentation ensure EU-region compute cannot cross-region-read US-only tagged data and vice versa.

## 26. Authentication (service-to-service and end-user auth mechanism)

- **Service-to-service**: mTLS + short-lived workload identity tokens (SPIFFE/SPIRE-style or cloud-native IRSA on EKS) between orchestrator, cluster manager, Spark/Ray executors, model registry, and lakehouse — no long-lived static credentials embedded in job specs.
- **End-user/operator auth** (job submission via API/UI): OAuth2/OIDC via corporate SSO; job-submission API requires a scoped service-account token tied to a specific team/tenant (used for quota attribution and column-ACL enforcement), not a personal user token, to keep job execution reproducible/auditable independent of who happens to be logged in.
- **Model registry access**: read access (fetch model for scoring) broadly granted to compute-cluster identities; write access (publish new model version) restricted to CI/training-pipeline service identities plus a human-approval gate for P0-tier models (anti-cheat, churn).

## 27. Rate Limiting (algorithm choice, per-user/per-tenant limits)

- Rate limiting here is about **job submission and cluster-resource consumption**, not per-request QPS (no live request traffic).
- **Per-tenant job-submission rate limit**: token-bucket, e.g., 20 ad-hoc job submissions/hour/team, refill continuous — prevents a misbehaving script or CI job from flooding the orchestrator with duplicate/runaway submissions.
- **Per-tenant compute quota** (the more important limit): max concurrent vCPU-hours and GPU-hours per team, enforced at Cluster Manager admission time (reject/queue new cluster requests once a tenant's quota is exhausted) — prevents one team's backfill from starving P0 nightly jobs of spot capacity.
- **Priority-weighted admission**: P0 jobs preempt P2 job cluster requests when capacity is scarce (not a classic rate-limit algorithm, but a priority-queue admission-control policy layered on top of the quota system).
- **Backfill-specific throttling**: large backfills (Section 2, FR4) are automatically chunked and rate-limited to at most N concurrent sub-jobs (e.g., 10) to avoid a single backfill request monopolizing the entire spot-capacity pool and starving regular scheduled jobs.

## 28. Autoscaling (metrics-driven autoscaling policy, HPA/VPA/KEDA specifics)

- **Cluster-level (not pod-level HPA in the traditional sense)**: Karpenter-based node autoscaling on EKS, driven by pending-pod count from Spark/Ray scheduler — when a job's driver requests N executors and no capacity exists, Karpenter provisions nodes from a spot-first, on-demand-fallback node pool within seconds-to-minutes.
- **KEDA** used for the orchestrator's own worker pool (task-execution workers that poll the job-submission queue, Section 12) — scales worker replica count based on queue depth (`ScaledObject` on SQS `ApproximateNumberOfMessages`).
- **Job-internal scaling**: Spark dynamic allocation enabled (executors scale up/down within a running job based on pending-task backlog) — useful for jobs with uneven partition-processing time (skewed regions) so idle executors release back to the pool mid-job rather than sitting reserved for the whole run.
- **Ray autoscaler**: min/max worker bounds set per job profile (e.g., min 8, max 40 GPU workers) — scales based on pending task count in the Ray scheduler queue; scale-down deliberately has a longer cooldown (10 min) than scale-up (1 min) to avoid thrashing on transient batch-boundary lulls.
- **No pod-level HPA on CPU%** for this system — CPU utilization of a single executor pod is a poor scaling signal for batch jobs (bursty by nature); queue depth / pending-partition count is the correct signal.

## 29. Cost Optimization (concrete levers: spot instances, caching, model distillation, batching)

| Lever | Mechanism | Estimated Impact |
|---|---|---|
| **Spot instances** | 70-80% of compute on spot/preemptible (checkpointable Spark/Ray tasks tolerate node loss via task retry) | ~60-65% cost reduction vs all-on-demand |
| **Off-peak scheduling** | P2/best-effort jobs deferred to lowest-demand windows (regional off-peak hours) where spot pricing is cheapest | ~10-15% additional reduction on non-SLA-critical jobs |
| **Incremental/delta scoring** | Only re-score entities with changed features since last run (FR8) instead of full re-score every time | Cuts effective row volume 40-70% for slowly-changing populations (e.g., inactive-but-registered accounts) |
| **Avoiding brute-force cross-joins** | Two-tower embeddings + ANN retrieval instead of full (user × item) scoring (Section 6) | Turns a 20B-row-equivalent job into a 40M-row embedding job — >100x reduction for recsys |
| **Model distillation** | Smaller distilled student model for high-frequency batch jobs (e.g., distilled toxicity classifier for routine re-scoring, full model reserved for escalation review) | 2-4x throughput per GPU, proportional cost cut on the largest recurring GPU job |
| **Mixed precision (fp16/int8)** | DL batch inference in fp16 rather than fp32 | ~1.5-2x GPU throughput gain, no meaningful accuracy loss for classification tasks |
| **Right-sized clusters via cost-estimate gate** | Job-plan step estimates required cluster size from historical row-count/throughput data rather than static over-provisioned defaults | Avoids the common failure mode of "always request the max cluster size just in case" |
| **Scale-to-zero** | Clusters torn down immediately at job completion, not kept warm | Eliminates idle-cluster spend, the single largest waste category in naive batch platforms |
| **Storage lifecycle/TTL** | Aggressive retention tuning on high-volume outputs (Section 6: top-50 not top-500 recsys candidates, 14-day TTL) | Cuts recsys output storage ~85% (14.4TB → ~2.2TB) |

## 30. Operational Concerns (Deployment, Reliability, Infra)

At SDE2 scope, treat this as a checklist rather than a design exercise: **backups** (automated snapshots of the model registry, feature store, and any stateful service, with a tested restore path), **rollback** (every deploy must be revertible to the last-known-good version — the model registry and CI/CD pipeline should make this a one-command operation), **canary/blue-green rollout** (shift a small percentage of traffic first, watch error rate and key business/model metrics, then ramp), and **basic observability** (dashboards + alerts on latency, error rate, and the top 2-3 model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level infra-architecture concerns — worth knowing they exist, not worth rehearsing the manifests.

## 38. Why This Architecture (justification)

- **Separation of orchestration from execution** (Airflow/Dagster vs Spark/Ray) lets scheduling logic evolve (new triggers, priority policies) independently of compute-engine choice, and lets different model types use the compute engine best suited to them (Spark for tabular/ETL-heavy, Ray for Python-native GPU DL) rather than forcing everything through one engine.
- **Versioned, append-only outputs + explicit lineage** make backfills, rollbacks, and multi-model experimentation safe by construction — the single biggest source of production incidents in naive batch platforms (in-place overwrites corrupting "current" state) is designed out.
- **Spot-first, scale-to-zero compute** matches the workload's actual shape (bursty, schedulable, checkpoint-tolerant) — batch inference is close to the ideal spot-instance workload, unlike latency-sensitive online serving.
- **Point-in-time-correct feature snapshots** shared between training and batch-inference input eliminates an entire class of train/serve skew bugs common when these paths diverge.

## 39. Alternative Architectures (at least 2 alternatives with why they were rejected or when they'd be preferred)

| Alternative | Description | Why Rejected / When Preferred |
|---|---|---|
| **Fully managed batch-transform service** (e.g., SageMaker Batch Transform / Vertex AI Batch Prediction end-to-end, no self-managed Spark/Ray) | Cloud-native managed batch inference, no cluster ops | Rejected at EA scale: less control over spot bin-packing/cost optimization at this volume, weaker support for multi-engine (Spark for ETL-heavy tabular joins) workflows, and cross-cloud portability concerns given EA's multi-cloud footprint across studios. **Preferred when**: a smaller team/studio without dedicated platform engineering needs to stand up batch scoring quickly and job volume is modest (tens of millions of rows, not hundreds of millions). |
| **Streaming-only (no batch), score everything on read via a real-time feature+model pipeline** | Replace scheduled batch jobs with an always-on streaming job that continuously scores as new events arrive | Rejected as primary pattern: massively over-provisions compute for workloads that are naturally "check once a day/week" (churn, weekly anti-cheat sweeps); streaming infra also complicates point-in-time correctness for backfills/reproducibility. **Preferred when**: the use case genuinely needs sub-minute freshness (e.g., real-time in-match cheat flagging) — which argues for a *separate* online-inference system, not replacing this one. |
| **Single monolithic Spark-only platform (no Ray)** | Use Spark UDFs for all model types including GPU DL, via Spark-on-GPU/RAPIDS | Rejected: Spark's JVM-centric execution model and RDD/DataFrame abstractions add overhead for tensor-heavy, Python-native DL workloads; GPU utilization and batching control are meaningfully worse than a Ray-native pipeline for embedding/DL inference. **Preferred when**: workload is >95% tabular/CPU with only occasional light DL, where standing up a second compute engine (Ray) isn't worth the operational overhead. |

## 40. Tradeoffs (explicit tradeoff table)

| Decision | Pro | Con |
|---|---|---|
| Spot-first compute (70-80%) | Major cost reduction | Preemption adds variance to job completion time; SLA-critical jobs need on-demand fallback budget |
| Append-only versioned outputs | Safe backfills/rollback, reproducibility | Higher storage footprint (must actively TTL/prune old runs); "what's current" requires a lineage-pointer indirection layer, adding a small amount of consumer-side complexity |
| Two compute engines (Spark + Ray) | Best-fit engine per workload type | Two operational surfaces to maintain, monitor, upgrade, and staff expertise for |
| Point-in-time snapshot reads (no live-table reads) | Consistent, reproducible job inputs | Adds up to ~4h data-freshness lag by design; not suitable for genuinely real-time use cases |
| Incremental/delta scoring | Large cost savings on slow-changing populations | Added complexity in change-detection logic; risk of silently missing an entity whose change wasn't captured by the delta detector (mitigated by periodic full-rescore safety net) |
| Priority-tiered quota system (P0/P1/P2) | Protects SLA-critical jobs from noisy-neighbor starvation | Lower-priority teams (backfills/experiments) experience unpredictable scheduling delays, can frustrate research velocity |

## 41. Failure Modes (concrete failure scenarios and mitigations)

| Failure Scenario | Impact | Mitigation |
|---|---|---|
| Spot-capacity mass reclaim mid-job (e.g., 40% of a 100-node cluster reclaimed simultaneously during a regional spot-price spike) | Job stalls, may miss SLA | Dynamic allocation + task-level retry on remaining/new nodes; on-demand fallback burst capacity reserved for P0 jobs when spot reclaim rate exceeds threshold |
| Severe partition skew (one region's partition is 5x larger due to a live-service event spike) | Long-pole partition delays whole job completion | Partitioner re-balances using recent row-count stats; adaptive re-partitioning mid-job (Spark AQE) splits oversized partitions dynamically |
| Upstream feature-table schema change (new nullable column silently added) | Job may crash or silently produce degraded predictions on unmapped fields | Schema-validation gate at job start (Section 8, DQ Validator); job fails fast rather than scoring on drifted schema |
| Model registry unavailable at job start | Job cannot fetch model artifact, fails to start | Node-local artifact cache (Section 11) allows jobs to proceed using last-cached version if registry is down and version matches; otherwise job queues/retries with backoff |
| Duplicate job submission (retry storm from an upstream caller) | Risk of double-scoring, wasted compute | Idempotency-key dedup at API layer (Section 9) returns existing job_id instead of scheduling duplicate |
| Silent model-quality regression (new model version passes infra checks but is statistically worse) | Bad predictions flow to CRM/anti-cheat actions before detected | Canary partition gate (Section 33) + prediction-distribution DQ check before lineage pointer advances |
| Cross-region data leakage bug (job misconfigured to read EU data from US cluster) | GDPR compliance violation | Network-level segmentation + IAM deny rules prevent cross-region reads regardless of application-layer bugs (defense in depth) |
| Output-write partial failure (Iceberg commit interrupted mid-write) | Risk of partial/corrupt output visible to readers | Atomic transactional commits (all-or-nothing partition write) — readers never see a partially-written run_id |

## 42. Scaling Bottlenecks (where this breaks first at 10x/100x scale)

- **At 10x scale** (~4-6B rows for the largest jobs): partition-planning and shuffle overhead in Spark becomes the first bottleneck — naive hash partitioning starts producing worse skew at this volume; requires moving to more sophisticated range-partitioning/bucketing strategies and possibly splitting single monolithic jobs into region/title-sharded sub-DAGs run in parallel rather than one giant job.
- **At 10x scale**, the lakehouse metadata layer (Iceberg manifest files, partition listing) can become a planning-time bottleneck if partition count grows unmanaged (too many small `run_id` partitions accumulating) — requires proactive compaction/manifest-rewrite jobs.
- **At 100x scale** (~40-60B rows): single-region compute capacity ceilings become real — even with generous spot pools, a single region's available spot inventory for the needed instance families may not satisfy demand at peak, forcing either massive on-demand spend or spreading a single logical job across multiple regions/clouds (adds significant orchestration complexity, cross-region data-movement cost, and residency-compliance surface area).
- **At 100x scale**, the online feature-store sink (bulk write-through of "latest score" for every entity) becomes a write-throughput bottleneck on Redis/DynamoDB — mitigated by moving to a batch-friendly bulk-load mechanism (e.g., DynamoDB S3 import, or Redis `RESTORE`/bulk pipeline with backpressure) rather than naive per-key writes at that volume.
- **Model registry egress** at 100x node count (if node-local caching strategy weren't in place) would saturate registry bandwidth — validates the broadcast/cache-aside design choice (Section 11) as increasingly load-bearing at scale, not just a nice-to-have.

## 43. Latency Bottlenecks (where time is actually spent, p50/p99 budget breakdown)

**Nightly churn job, 90-min p50 budget breakdown:**

| Phase | Time | % of Budget |
|---|---|---|
| Cluster provisioning (spot acquisition) | 4 min | 4.4% |
| Model artifact fetch/broadcast | 2 min | 2.2% |
| Input scan + partition read (I/O-bound) | 28 min | 31% |
| Actual model inference (compute) | 22 min | 24.4% |
| Shuffle/aggregation overhead | 15 min | 16.7% |
| Output write (Iceberg commit) | 6 min | 6.7% |
| DQ validation checks | 5 min | 5.6% |
| Feature-store sink bulk write | 8 min | 8.9% |

- **Dominant cost is I/O (input scan) at ~31%, not model compute (~24%)** — this is the characteristic signature of batch-inference latency (unlike online serving, where model compute usually dominates); optimization effort is better spent on columnar-scan efficiency (better partition pruning, column projection) than on squeezing inference-loop microseconds.
- **p99 blowup drivers**: skewed partitions (one region 3-5x larger than planned) and spot-preemption-triggered task retries are the two biggest contributors to the 90min→180min p50-to-p99 gap — both addressed by adaptive re-partitioning and dynamic allocation (Section 41).

## 44. Cost Bottlenecks (what actually drives the bill)

- **#1 driver: GPU compute for DL batch-inference jobs** (anti-cheat, recsys) — GPU-hour cost per unit compute is 10-20x CPU, so even though these jobs process fewer total rows than the tabular churn job, they can dominate the batch-inference cloud bill if not tightly right-sized (validates model-distillation and mixed-precision levers, Section 29, as high-leverage).
- **#2 driver: storage of high-cardinality outputs** (pre-TTL-tuning, recsys candidate storage was the single largest storage line item at ~14.4TB/90-day — Section 6) — output-retention policy is a cost lever as significant as compute choice, often overlooked.
- **#3 driver: on-demand fallback spend** during spot-capacity shortages for SLA-critical (P0) jobs — if spot availability degrades in a region (e.g., broad demand spike from other tenants on the same cloud), the automatic on-demand fallback (Section 6/33) can spike cost sharply for those windows; worth budgeting an explicit "on-demand fallback cost cap" alert (Section 23) distinct from the general cost-anomaly alert.
- **#4 driver: redundant/duplicate scoring from lack of incremental/delta scoring** on slowly-changing subpopulations — teams that skip FR8 (delta scoring) and always full-rescore pay for re-computing predictions on entities whose features haven't changed since the last run, a pure waste multiplier proportional to population "staleness" rate.

## 45. Interview Follow-Up Questions (at least 8, the kind a Principal-level interviewer would ask to probe depth)

1. How do you guarantee a backfill replaying the same date range twice doesn't double-write or corrupt downstream consumers reading mid-replay?
2. Walk me through what happens if a 400-node Spark job loses 150 spot nodes simultaneously at minute 60 of a 90-minute SLA-critical job.
3. Why Spark for tabular and Ray for DL rather than standardizing on one engine — what's the actual cost of running two engines?
4. How do you prevent a bad model version from silently degrading production predictions before anyone notices, given batch jobs have no live traffic to A/B against in real time?
5. Your recsys job avoids brute-force (user × item) scoring via ANN — what happens to correctness/recall when the item catalog grows from 50K to 5M items?
6. How would you design incremental/delta scoring to avoid silently missing an entity whose relevant feature changed but wasn't detected by your change-detection mechanism?
7. Where exactly does point-in-time correctness get violated most easily in a system like this, and how would you catch it in code review or in CI rather than in production?
8. If EU data-residency rules tightened further (e.g., no cross-region metadata either), what would break in your current lineage/monitoring design, and how would you redesign it?
9. How do you decide when a job's row-level failure rate is "acceptable partial success" versus something that should hard-fail the whole job?
10. At 10x scale, would you keep the append-only/versioned-output pattern, or would storage/compaction costs force a different retention model?

## 46. Ideal Answers (a strong, concise model answer for each follow-up question above)

1. **Idempotent, versioned writes keyed by `(model_version, data_version, partition)` under an explicit `run_id`, plus an `idempotency_key` at the API layer.** A backfill sub-job always writes to the same deterministic `run_id`, so a replay either re-executes into the same atomically-overwritten partition or is short-circuited if the key matches an already-completed job. Consumers never see a half-written state because the "latest complete" pointer only advances after an all-or-nothing commit.
2. **Spark's dynamic allocation degrades gracefully**: lost executors' in-flight tasks are rescheduled onto surviving/new nodes without restarting the whole job. Because the job is P0/SLA-tagged, on-demand fallback capacity is provisioned for the lost nodes and an SLA-breach-imminent alert (Section 23) fires early, giving on-call visibility before the deadline is missed.
3. **The real cost is operational — two schedulers, two failure modes, two on-call runbooks.** It's justified because forcing GPU/DL workloads through Spark or ETL-heavy joins through Ray costs more in engineering time than maintaining both: Spark's optimizer/shuffle machinery wins for large joins, Ray's actor model wins for batched GPU inference. Below roughly 10% of job volume being GPU/DL work, Spark-on-GPU/RAPIDS may suffice instead.
4. **Canary partition (Section 33) plus a hard gate before the lineage "production" pointer advances.** Run the new model version against a held-out slice of the same snapshot the incumbent scored, diff prediction distributions and DQ metrics, and gate promotion on that comparison. Batch outputs being append-only/versioned means a bad run is never visible to consumers unless explicitly promoted.
5. **HNSW's build cost and memory scale roughly linearly with catalog size and stay tractable to a few million vectors, but recall/latency degrade past that.** At 5M items we'd move to IVF-PQ for memory efficiency and add a coarse pre-filter to shrink the search space, re-benchmarking recall@K against brute-force ground truth to confirm the approximation stays acceptable.
6. **Layer a deterministic change-detection signal (hash/timestamp on the feature row) with a periodic full-rescore safety net** (e.g., every 7th run rescores everything regardless of detected deltas), bounding staleness blast radius to one cycle. Also monitor the % of population flagged "changed" per run — an unexplained drop signals the detector itself may be broken.
7. **Most easily violated at the join between a "current" mutable table and a historical feature table** — e.g., joining today's churn job against a live customer-profile table instead of a point-in-time snapshot, leaking future information into a backfill. Caught via a CI lint rule flagging non-time-traveled reads without `AS OF`, and a shared library that only exposes point-in-time-safe read APIs by default.
8. **The global lineage/status view (Section 31) avoids moving raw data cross-region, but job-status metadata could still count as sensitive in a stricter regime.** Redesign: keep the lineage/status store region-local too, replacing the global view with a federated query layer that fans out read-only queries at view-time — trading some query latency for zero cross-region data movement.
9. **Treat it as a statistical/business-risk threshold set jointly with the model owner, not a fixed platform-wide constant.** A 0.5% baseline hard-fails above 1% because that signals a systemic issue rather than isolated bad rows; safety-critical outputs get a tighter threshold, and failed rows are always re-queued rather than silently dropped.
10. **At 10x, keep the append-only pattern for correctness but tighten retention/compaction and treat storage lifecycle as a first-class scaling concern.** Shorten TTLs, add tiered storage (hot runs on standard storage, older runs on Glacier-class), and invest in proactive manifest compaction — what changes is how aggressively old data is demoted, not whether the pattern is abandoned.
</content>
