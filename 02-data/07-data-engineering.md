---
module: Data
topic: Data Engineering
subtopic: ""
status: unread
tags: [data-engineering, batch, streaming, kafka, spark, airflow, dbt, feature-store, data-quality, mlops]
prerequisites: [python-and-data-tooling, data-processing-and-eda]
---
# Data Engineering for ML

> **Related:** The rest of this folder is about *transforming* data you already have (preprocessing, feature engineering, imbalance). This file is about *getting data to the model reliably at scale* — the pipelines, stores, and contracts an ML system runs on. It's the "how does data actually reach production" half that model-centric interviews skip and system-design interviews demand.

The single organizing question: **where does the data live, how does it move, and how do you keep it correct?** Everything below is a variation on that.

---

## Batch vs. Streaming

The first axis. Do you process data in scheduled chunks or as it arrives?

| | Batch | Streaming |
|---|---|---|
| **Unit** | Bounded dataset (a day, an hour) | Unbounded, per-event |
| **Latency** | Minutes to hours | Milliseconds to seconds |
| **Tools** | Spark, dbt, warehouse SQL | Kafka, Flink, Spark Structured Streaming |
| **When** | Training data, daily aggregates, reports | Fraud scoring, real-time recs, live features |
| **Reasoning** | Simpler, cheaper, easy to reprocess | Complex, but required when staleness costs money |

**The default is batch.** Streaming is a cost you take on only when latency genuinely matters — fraud that must be caught in the transaction, a feature that's worthless if an hour old. Interviewers reward candidates who *don't* reach for streaming reflexively; most "real-time" requirements are satisfied by a micro-batch every few minutes.

**Spark** is the batch workhorse: distributed dataframes, lazy execution, spills to disk. **Kafka** is the streaming backbone: a durable, partitioned, replayable log — producers append, consumers read at their own offset. Kafka's replayability is the key property: because it's a log, not a queue, you can reprocess history by resetting the offset, which is what makes stream reprocessing and backfills tractable.

**Lambda vs. Kappa architecture.** Lambda runs a batch layer (accurate, slow) *and* a speed layer (fast, approximate) and merges them — two codebases for the same logic, a maintenance tax. Kappa treats everything as a stream and reprocesses from the log when it needs the "batch" answer — one codebase. The modern lean is Kappa when the streaming engine is expressive enough.

---

## Pipelines and Orchestration

A pipeline is a DAG of tasks with dependencies. Orchestration is scheduling that DAG, handling failures, retries, and backfills.

- **Airflow** — the incumbent orchestrator. You author DAGs in Python; it schedules, retries, and shows lineage. Good at "run these tasks in this order on this schedule," weaker at data-awareness (it orchestrates *tasks*, not *datasets* — though asset-based scheduling narrows this). Alternatives with the same role: Dagster, Prefect, Mage.
- **dbt** — transforms *inside the warehouse*. You write `SELECT` statements; dbt handles dependency ordering, materialization (view/table/incremental), testing, and documentation. It owns the **T in ELT**. dbt doesn't move data; it orchestrates SQL transformations on data already loaded.

**ETL vs. ELT.** ETL transforms *before* loading (transform in Spark, then load clean data). ELT loads raw *first*, then transforms in the warehouse with SQL/dbt. ELT won for analytics because cloud warehouse compute is cheap and elastic, raw data is preserved for reprocessing, and SQL is accessible to more people. ETL still wins when transformation needs non-SQL logic or must happen before data touches an expensive store.

**Idempotency is the property that makes pipelines survivable.** A task that reruns must produce the same result — so a retry after a mid-run crash is safe. The standard pattern is *delete-then-insert by partition* (or `MERGE`), never blind `APPEND`, so a re-run of "process 2026-07-22" overwrites that partition instead of duplicating it. An interviewer asking "your Airflow task failed halfway and retried — what happens?" is testing exactly this.

---

## Data Storage: Warehouse vs. Lake vs. Lakehouse

| | Warehouse | Lake | Lakehouse |
|---|---|---|---|
| **Data** | Structured, schema-on-write | Any format, schema-on-read | Open table format over lake storage |
| **Examples** | Snowflake, BigQuery, Redshift | S3/GCS + Parquet | Delta Lake, Iceberg, Hudi |
| **Strength** | Fast SQL, governance | Cheap, flexible, any data | Warehouse features on lake economics |
| **Weakness** | Costly, structured only | No ACID, no schema enforcement ("swamp") | Newer, added complexity |

**Schema-on-write vs. schema-on-read** is the crux. The warehouse enforces schema at load time — clean but rigid. The lake accepts anything and defers schema to read time — flexible but degenerates into a "data swamp" with no guarantees. The **lakehouse** (Delta/Iceberg) is the reconciliation: it adds ACID transactions, schema enforcement, and time-travel *on top of* cheap object-store Parquet, so you get warehouse correctness at lake cost. This is where the industry is consolidating.

---

## Feature Stores

The piece unique to *ML* data engineering. A feature store manages features for both training and serving, and it exists to solve one specific, expensive bug.

**Train-serve skew** — the defining problem. Training computes a feature one way (a batch SQL over historical data); serving computes it another (a live function on a request). Any discrepancy — different code, different windows, different null handling — silently degrades the model in production while offline metrics look fine. It's insidious because nothing errors; the model just quietly underperforms.

A feature store fixes it with **two synchronized stores fed by shared definitions:**
- **Offline store** — historical feature values (warehouse/lake), used to build training sets. Optimized for large scans.
- **Online store** — latest feature values (Redis/DynamoDB), used at serving. Optimized for low-latency key lookups.

Both are populated from the *same feature definition*, so training and serving compute features identically by construction. Examples: Feast, Tecton, Databricks Feature Store.

**Point-in-time correctness** is the other thing a feature store buys you, and the subtler one. When you build a training set, each label must be joined to feature values **as they were at that label's timestamp** — not their current values. Using a feature's *present* value for a *past* label leaks future information (a form of the label leakage covered in [../04-evaluation](../04-evaluation/README.md)) and inflates offline metrics that then collapse in production. Feature stores do this "as-of" join for you.

**When you don't need one.** If features are simple, batch-only, and shared across few models, a feature store is over-engineering — a warehouse table plus disciplined shared SQL suffices. The store earns its complexity when many models share features, or when the same feature must be served online with sub-100ms latency *and* consistently reproduced offline.

---

## Data Quality and Contracts

A model is a function of its data; bad data fails silently, which is worse than crashing. Quality is engineered, not hoped for.

- **Validation / expectations** — assert properties on every batch: schema, ranges, null rates, uniqueness, referential integrity, distributional checks. Tools: Great Expectations, dbt tests, Deequ. The point is to *fail the pipeline* on bad data rather than let it reach the model.
- **Data contracts** — an explicit, enforced agreement between a data *producer* (often an upstream service team) and *consumers*: schema, semantics, SLAs, and what constitutes a breaking change. Contracts push quality *upstream* to where data is created, instead of consumers endlessly patching breakage downstream. This is the organizational fix for the most common real cause of ML incidents: an upstream team renamed a column or changed units and told no one.
- **Freshness and lineage** — is the data current (freshness SLA), and if a value is wrong, where did it come from (lineage)? Lineage turns "the model is broken" into "this upstream table is broken" quickly.
- **Drift monitoring** — production data distribution moving away from training (covariate/label drift). A data-engineering concern because it's caught at the pipeline boundary, before it silently becomes a model-quality problem. (See model monitoring in [../16-interview-prep](../16-interview-prep/README.md) for the modeling-side response.)

**The mental model:** treat data like code. Version it, test it, review schema changes, monitor it in production, and make breaking changes explicit and negotiated — a contract, not a surprise.

---

## Interview Angles

### Q: A feature works great offline but the model underperforms in production. Where do you look first? [Medium]

**Train-serve skew.** The feature is almost certainly computed differently at training time (batch SQL over history) than at serving time (a live function per request) — different windows, null handling, or code paths. I'd diff the two computations on identical inputs. The systemic fix is a feature store with a shared definition feeding synchronized offline and online stores, so the two *cannot* diverge.

**Cross-questions to expect:**
- *"Could it be point-in-time leakage instead?"* → Yes, and it's the sibling bug: if the training set used features' *current* values rather than their values *as of each label's timestamp*, offline metrics are inflated by future information and production is the honest number. Check the as-of join.
- *"How would a feature store prevent both?"* → One definition computes the feature; offline and online stores are populated from it (kills skew); the offline store supports as-of joins (kills leakage).

---

### Q: When do you actually need streaming instead of batch? [Medium]

When staleness has a real cost measured in seconds — fraud caught during the transaction, a feature that's useless an hour later, live inventory. Otherwise batch (or micro-batch every few minutes) is cheaper, simpler, reprocessable, and easier to reason about. I default to batch and justify streaming by the latency requirement, not by "real-time" as an aspiration.

**Cross-questions to expect:**
- *"What makes streaming harder?"* → Out-of-order and late-arriving events, exactly-once semantics, windowing, and state management — none of which batch has to worry about, since a batch is a bounded, replayable dataset.
- *"Lambda vs. Kappa?"* → Lambda maintains separate batch and speed layers (two implementations of the same logic — a maintenance tax); Kappa uses one streaming path and reprocesses from the log for the "batch" answer. Kappa when the stream engine is expressive enough.

---

### Q: Your Airflow task failed halfway through writing a day's data and retried. How do you make sure you don't get duplicates? [Hard]

**Idempotency.** The task must produce the same result whether it runs once or five times. The standard pattern is partition-scoped overwrite: *delete-then-insert* (or `MERGE`) for the target partition — e.g. "delete 2026-07-22, then insert 2026-07-22" — never a blind `APPEND`. A retry re-runs the whole partition and overwrites the partial write instead of stacking a second copy on top.

**Cross-questions to expect:**
- *"What if a downstream consumer already read the partial data?"* → That's why you also want atomic publish — write to a staging location and swap/commit atomically (what lakehouse table formats like Delta/Iceberg give you via ACID commits), so consumers never see a half-written partition.
- *"How does this interact with exactly-once in streaming?"* → Same goal, harder setting: streaming achieves it with idempotent writes plus offset/transaction coordination (e.g. Kafka transactions), because there's no clean "partition" to overwrite.

**Trap:** Answering "add a retry limit" or "alert on failure." Those handle the *symptom*; idempotency handles the *cause* so retries are simply safe.

---

## Connections

- **Upstream of everything here:** [01-python-and-data-tooling.md](01-python-and-data-tooling.md), [02-data-processing-and-eda.md](02-data-processing-and-eda.md)
- **What the pipelines feed:** [04-feature-engineering.md](04-feature-engineering.md) — feature stores serve engineered features
- **Leakage, the failure this prevents:** point-in-time correctness connects to evaluation integrity in [../04-evaluation/](../04-evaluation/README.md)
- **Production monitoring / drift response:** [../16-interview-prep/](../16-interview-prep/README.md)
