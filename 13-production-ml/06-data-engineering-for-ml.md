---
module: Production ML
topic: System Design
subtopic: Data Engineering For Ml
status: unread
tags: [productionml, ml, system-design-data-engineering]
---
# Data Engineering for ML

> **TL;DR**: ML models are only as good as the data pipelines that feed them. Most ML failures in production are data failures — stale features, silent schema changes, train-serve skew from format mismatches, and runaway compute costs from unpartitioned scans. This doc covers the infrastructure layer between raw data and model inputs.

---

## Data Pipeline Architecture

### Batch vs Streaming

**The problem**: data arrives continuously (user events, transactions, sensor readings), but ML training and some features only need periodic snapshots. The wrong pipeline architecture wastes compute or introduces unacceptable staleness.

**The core insight**: choose the pipeline type based on the acceptable latency of the feature, not the complexity of the implementation. Streaming is operationally harder; only use it when batch is too slow.

```
Pipeline Type | Latency       | Throughput | Complexity | Use Cases
--------------|---------------|------------|------------|--------------------------------------------
Batch         | Minutes–hours | Very high  | Low        | Training data, daily aggregates, reporting
Micro-batch   | 10s–minutes   | High       | Medium     | Near-real-time features, dashboards
Streaming     | <1 second     | Medium     | High       | Fraud velocity, live personalization
```

**Batch pipeline design**:

```python
# Typical batch ML pipeline: source → transform → sink
# Scheduled daily via Airflow / Prefect / Dagster

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, count, avg

spark = SparkSession.builder.appName("batch_feature_pipeline").getOrCreate()

# Read partitioned raw data — partition pruning avoids full scans
raw = spark.read.parquet("s3://data-lake/events/") \
    .filter(col("event_date") >= "2024-01-01")

# Transform
features = raw.groupBy("user_id", "event_date").agg(
    count("*").alias("event_count"),
    avg("session_duration").alias("avg_session_s"),
)

# Write partitioned by date for efficient downstream reads
features.write \
    .partitionBy("event_date") \
    .mode("overwrite") \
    .parquet("s3://feature-store/user_daily_features/")
```

**Streaming pipeline design**:

```python
# Flink SQL: continuous feature computation from Kafka
# Suitable for features with freshness SLA < 1 minute

CREATE TABLE user_events (
    user_id STRING,
    event_type STRING,
    event_time TIMESTAMP(3),
    WATERMARK FOR event_time AS event_time - INTERVAL '10' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'user-events',
    'properties.bootstrap.servers' = 'kafka:9092',
    'format' = 'json'
);

-- 5-minute tumbling window aggregation
CREATE VIEW user_5min_counts AS
SELECT
    user_id,
    COUNT(*) AS event_count_5min,
    TUMBLE_END(event_time, INTERVAL '5' MINUTE) AS window_end
FROM user_events
GROUP BY user_id, TUMBLE(event_time, INTERVAL '5' MINUTE);
```

---

### Lambda vs Kappa Architecture

**Lambda architecture**: maintains two separate processing paths — a batch layer for high-accuracy historical data and a speed layer for real-time approximations. Results are merged at query time.

**Kappa architecture**: eliminates the batch layer entirely. All computation happens through the streaming layer. Historical reprocessing is done by replaying the event log from the beginning.

```
Architecture | Batch Layer | Speed Layer | Complexity       | When to Use
-------------|-------------|-------------|------------------|----------------------------------
Lambda       | Yes         | Yes         | High (two stacks)| When batch and stream disagree;
             |             |             |                  | exact historical backfills needed
Kappa        | No          | Yes         | Lower            | When stream processor handles
             |             |             |                  | replay; unified codebase preferred
```

**The tradeoffs**:

- Lambda is operationally expensive: two codebases that must produce identical results for the same feature. Schema changes require coordinated updates in both paths.
- Kappa requires the stream processor to handle replay at batch throughput — Kafka retention must cover the full reprocessing window (weeks or months of logs).
- Modern default: prefer Kappa with Flink or Spark Structured Streaming. Fall back to Lambda only when you need exact historical correctness that streaming can't guarantee (e.g., late-arriving data beyond the watermark).

---

## Distributed Data Processing

### Apache Spark Internals

**RDDs vs DataFrames vs Datasets**:

```
Abstraction  | Type Safety | Optimization | API Level | Use When
-------------|-------------|--------------|-----------|----------------------------
RDD          | Compile-time| None         | Low       | Custom transformations,
             | (JVM only)  |              |           | non-tabular data
DataFrame    | Runtime     | Catalyst + Tungsten | High | Standard ETL, SQL-like ops
Dataset      | Compile-time| Catalyst + Tungsten | High | JVM-typed pipelines
             | (JVM only)  |              |           | (Scala/Java only)
```

**Rule of thumb**: use DataFrames for all production ML pipelines in Python. The Catalyst optimizer rewrites your query plan; RDD-based code bypasses it and is typically 2-5× slower.

**DAG execution and shuffles**:

Spark translates transformations into a DAG of stages. Stage boundaries occur at **shuffle operations** — wide transformations that require data redistribution across the cluster.

```python
# Narrow transformation: no shuffle, same partition
df.filter(col("amount") > 100)   # stays in same partition
df.select("user_id", "amount")   # stays in same partition

# Wide transformation: shuffle, new partition assignment
df.groupBy("user_id").agg(count("*"))    # SHUFFLE — all rows for same user_id must land on same executor
df.join(other_df, on="user_id")          # SHUFFLE — sort-merge or broadcast join

# Expensive: avoid groupByKey (shuffles all values, not just aggregates)
# Prefer reduceByKey or aggregateByKey (partial aggregation before shuffle)
```

**Partitioning strategies**:

```python
# Default: 200 shuffle partitions (often wrong for your data size)
spark.conf.set("spark.sql.shuffle.partitions", "800")  # tune to ~128MB per partition

# Repartition before expensive join: co-locate matching keys
df1 = df1.repartition(800, "user_id")
df2 = df2.repartition(800, "user_id")
joined = df1.join(df2, on="user_id")  # reduces shuffle — keys already co-located

# Broadcast join: replicate small table to all executors, avoid shuffle entirely
from pyspark.sql.functions import broadcast
result = large_df.join(broadcast(small_lookup_df), on="product_id")
# Use when small_lookup_df < spark.sql.autoBroadcastJoinThreshold (default 10MB)

# Salting: fix skewed partitions (one user has 10M events, others have 100)
from pyspark.sql.functions import concat, lit, (col("user_id") % 10).alias("salt")
df_salted = df.withColumn("user_id_salted", concat(col("user_id"), lit("_"), (col("user_id") % 10).cast("string")))
```

**What breaks**: the most common Spark performance issue is data skew — one partition has orders of magnitude more data than others, causing one executor to run for hours while the rest finish in minutes. Symptoms: Spark UI shows one "stragglers" task. Fix: salt the skewed key or use Spark's built-in skew join hint (`SKEW_JOIN`).

---

### Apache Flink for Streaming

**When to choose Flink over Spark Streaming**: Flink is a true streaming engine with per-event latency; Spark Structured Streaming is micro-batch (minimum ~100ms latency). For features requiring <1 second freshness, use Flink.

```python
# Flink: stateful stream processing with event-time windows
# Critical for correct windowed aggregations over out-of-order data

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import AggregateFunction, WindowFunction
from pyflink.datastream.window import TumblingEventTimeWindows
from pyflink.common.time import Time

class CountAndSum(AggregateFunction):
    """Accumulate count and sum across events in window."""
    def create_accumulator(self):
        return (0, 0.0)  # (count, sum)

    def add(self, value, accumulator):
        return (accumulator[0] + 1, accumulator[1] + value[1])  # value = (user_id, amount)

    def get_result(self, accumulator):
        return accumulator

    def merge(self, a, b):
        return (a[0] + b[0], a[1] + b[1])

env = StreamExecutionEnvironment.get_execution_environment()

stream = env \
    .add_source(kafka_source) \
    .key_by(lambda x: x[0])  # key by user_id \
    .window(TumblingEventTimeWindows.of(Time.minutes(5))) \
    .aggregate(CountAndSum())
```

**Watermarks and late data**: Flink uses watermarks to determine when a window is complete. A watermark of `event_time - 10s` means Flink assumes all events with `event_time < watermark` have arrived; it closes and emits the window result. Late events beyond the allowed lateness are either dropped or routed to a side output for batch correction.

---

### Distributed SQL: Trino/Presto

Trino (open-source fork of Presto) is a federated query engine — it queries Hive, Iceberg, Delta Lake, S3, PostgreSQL, Kafka, and other connectors without moving data.

```sql
-- Trino: federated query across data lake + warehouse
SELECT
    u.user_id,
    u.signup_date,
    s3.purchase_count_30d,
    pg.subscription_tier
FROM hive.ml_features.user_cohorts u              -- Parquet on S3
JOIN iceberg.feature_store.user_aggregates s3      -- Iceberg table
    ON u.user_id = s3.user_id
JOIN postgresql.prod_db.subscriptions pg           -- Live Postgres
    ON u.user_id = pg.user_id
WHERE u.signup_date >= DATE '2024-01-01'
  AND s3.feature_date = CURRENT_DATE - INTERVAL '1' DAY;
```

**Use Trino for**: ad-hoc analysis across heterogeneous stores, backfilling features from historical data, joining warehouse tables with data lake tables. Avoid for: sub-second latency queries (overhead per query ~200ms), streaming reads.

---

## Storage Formats

### Columnar vs Row-Oriented

**Row-oriented** (CSV, JSON, Avro): each row stored contiguously. Fast for writing single records, reading entire rows. Poor for analytical queries that read a few columns from many rows.

**Columnar** (Parquet, ORC, Arrow): each column stored contiguously. Fast for analytical reads (scan only the columns you need), efficient compression (similar values together). Poor for writing single records.

```
Format   | Orientation | Compression | Schema Evolution | Splittable | Best For
---------|-------------|-------------|------------------|------------|-----------------------------
Parquet  | Columnar    | Excellent   | Forward/backward | Yes        | Data lake, Spark, Iceberg
ORC      | Columnar    | Excellent   | Limited          | Yes        | Hive, ORC-optimized engines
Avro     | Row         | Good        | Excellent        | Yes*       | Kafka serialization, Hadoop
Arrow    | Columnar    | Good        | Limited          | No         | In-memory, IPC, pandas↔Spark
CSV/JSON | Row         | Poor        | None             | Yes        | Interchange, debugging only
```

**Compression algorithms**:

```
Algorithm | Ratio  | Speed       | Use When
----------|--------|-------------|--------------------------------------------
Snappy    | Medium | Very fast   | Default for Parquet; fast read/write critical
Zstd      | High   | Fast        | Best ratio/speed tradeoff; Parquet in cold storage
LZ4       | Low    | Fastest     | Hot paths, Arrow IPC, in-memory transfer
Gzip/Zlib | High   | Slow        | Interoperability (HTTP, legacy systems)
Brotli    | Highest| Slowest     | Parquet on cold storage, rarely read
```

**Rule of thumb**: use `Parquet + Zstd` for training data and feature stores. Use `Avro` for Kafka message serialization (schema registry integration). Use `Arrow` for in-process data transfer between Spark and Python workers.

---

### Predicate Pushdown

**The problem**: reading a 10TB Parquet table to find rows where `event_date = '2024-01-15'` is wasteful. The engine should skip row groups that cannot contain matching rows.

**How it works**: Parquet stores min/max statistics per row group and per column chunk. When a filter predicate is applied, the engine checks these statistics and skips row groups where the predicate cannot match.

```python
# Predicate pushdown: Spark pushes the filter into the Parquet reader
# The Parquet reader skips row groups where min(event_date) > '2024-01-15'
# or max(event_date) < '2024-01-15' — no data read for skipped groups

df = spark.read.parquet("s3://data-lake/events/") \
    .filter(col("event_date") == "2024-01-15") \   # pushed to Parquet reader
    .filter(col("amount") > 1000)                   # pushed to Parquet reader

# Verify: check Spark UI → SQL tab → physical plan for "PushedFilters"
# Look for: PushedFilters: [IsNotNull(event_date), EqualTo(event_date,2024-01-15)]
```

**Partition pruning** is a higher-level version: when data is stored in `s3://…/event_date=2024-01-15/` directory partitions, Spark skips listing and reading entire S3 prefixes. Predicate pushdown operates within a file; partition pruning avoids reading the file at all.

**What breaks**: predicate pushdown fails silently when the filter column is computed (e.g., `YEAR(event_time) = 2024` — the engine can't pushdown a derived expression). Always filter on the raw partition column, not a derived expression.

---

## Data Quality Frameworks

### Schema Validation

**The problem**: upstream data sources change schemas without notice — a column is renamed, a type changes from `INT` to `STRING`, a field is dropped. ML pipelines silently ingest corrupt data, producing wrong features and degraded models.

**Great Expectations**:

```python
import great_expectations as gx

context = gx.get_context()

# Define expectations on a training dataset
validator = context.get_validator(
    datasource_name="s3_data_lake",
    data_connector_name="default_inferred_data_connector",
    data_asset_name="user_features",
)

# Schema expectations
validator.expect_column_to_exist("user_id")
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_be_of_type("purchase_count_30d", "LongType")

# Value range expectations
validator.expect_column_values_to_be_between("purchase_count_30d", min_value=0, max_value=10000)
validator.expect_column_values_to_be_between("avg_transaction_amount", min_value=0.0)

# Distribution expectations (drift detection)
validator.expect_column_mean_to_be_between("avg_transaction_amount", min_value=45.0, max_value=75.0)
validator.expect_column_quantile_values_to_be_between(
    "purchase_count_30d",
    quantile_ranges={"quantiles": [0.25, 0.5, 0.75], "value_ranges": [[0, 2], [1, 5], [3, 15]]}
)

results = validator.validate()
if not results["success"]:
    raise ValueError(f"Data quality check failed: {results['statistics']}")
```

**Deequ** (Spark-native, better at scale):

```python
# Deequ: data quality at Spark scale (billions of rows)
from pydeequ.checks import Check, CheckLevel
from pydeequ.verification import VerificationSuite

check = (
    Check(spark, CheckLevel.Error, "user_features_check")
    .isComplete("user_id")
    .isUnique("user_id")
    .isNonNegative("purchase_count_30d")
    .hasCompleteness("avg_transaction_amount", lambda c: c >= 0.95)  # allow 5% nulls
    .satisfies("purchase_count_30d <= 100000", "purchase count reasonable")
)

result = VerificationSuite(spark).onData(df).addCheck(check).run()
```

---

### Data Contracts

**The problem**: schema validation catches what data looks like; data contracts define what data must guarantee — semantics, SLAs, ownership. Contracts are agreements between data producers and consumers.

```yaml
# data_contract.yaml — enforced in CI/CD, breaks pipeline on violation
contract:
  name: user_transaction_features
  version: "2.1.0"
  owner: data-platform-team
  consumers: [fraud-model, recommendation-model]

  schema:
    - name: user_id
      type: string
      nullable: false
      unique: true
    - name: purchase_count_30d
      type: long
      nullable: true
      min: 0
      max: 100000
    - name: feature_timestamp
      type: timestamp
      nullable: false

  sla:
    freshness_max_minutes: 60    # features must be <1h old
    availability_percent: 99.9
    p99_latency_ms: 200

  quality:
    null_rate_max: 0.05           # at most 5% nulls in any column
    row_count_min: 1000000        # at least 1M rows expected daily
```

**Anomaly detection in pipelines**: beyond static expectations, detect statistical anomalies relative to historical baselines.

```python
# Simple z-score anomaly detection on row count
import numpy as np

def check_row_count_anomaly(current_count: int, historical_counts: list, z_threshold: float = 3.0):
    mean = np.mean(historical_counts)
    std = np.std(historical_counts)
    if std == 0:
        return False
    z = (current_count - mean) / std
    if abs(z) > z_threshold:
        raise ValueError(
            f"Row count anomaly: current={current_count}, "
            f"expected={mean:.0f}±{std:.0f}, z={z:.2f}"
        )
```

---

## Data Versioning and Reproducibility

### DVC Mechanics

**The problem**: Git tracks code; it cannot track 100GB training datasets or model artifacts. Without data versioning, you can't reproduce a training run from 6 months ago — the data may have been overwritten.

**How DVC works**: DVC stores a small `.dvc` pointer file in Git (contains MD5 hash + file path). The actual data lives in a remote store (S3, GCS, Azure Blob). Running `dvc pull` restores data matching the pointer in the current Git commit.

```bash
# Initialize DVC in a Git repo
dvc init
git add .dvc/config && git commit -m "init dvc"

# Track a large dataset
dvc add data/training_set.parquet
# Creates: data/training_set.parquet.dvc (tracked by git)
#          .gitignore (excludes the actual file from git)
git add data/training_set.parquet.dvc .gitignore
git commit -m "add training dataset v1"

# Push data to remote
dvc remote add -d myremote s3://ml-artifacts/dvc-cache
dvc push

# Reproduce: checkout a past experiment and restore exact data
git checkout abc1234
dvc pull  # restores data/training_set.parquet as it was at that commit
```

**DVC pipelines** (reproducible experiment DAGs):

```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python preprocess.py --input data/raw.parquet --output data/processed.parquet
    deps:
      - preprocess.py
      - data/raw.parquet
    outs:
      - data/processed.parquet
  train:
    cmd: python train.py --data data/processed.parquet --output models/model.pkl
    deps:
      - train.py
      - data/processed.parquet
    outs:
      - models/model.pkl
    metrics:
      - metrics.json
```

```bash
dvc repro    # only re-runs stages whose inputs changed (hash-based caching)
dvc metrics show  # compare metrics across git branches/commits
```

---

### Delta Lake: ACID Transactions and Time Travel

**The problem**: Parquet on S3 has no ACID guarantees. If a write job fails mid-way, you get a partially written table. Concurrent writers produce corrupt data. There's no rollback.

**How Delta Lake works**: Delta Lake adds a `_delta_log/` directory alongside Parquet files. Every transaction is written as a JSON log entry listing which files were added or removed. The table state at any point in time is reconstructed by replaying the log.

```python
from delta.tables import DeltaTable
from pyspark.sql.functions import col

# Write with ACID guarantees
df.write.format("delta").mode("overwrite").save("s3://data-lake/user_features/")

# MERGE: upsert — update existing rows, insert new rows atomically
delta_table = DeltaTable.forPath(spark, "s3://data-lake/user_features/")

delta_table.alias("target").merge(
    new_data.alias("source"),
    condition="target.user_id = source.user_id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()

# Time travel: read the table as of a past version
df_yesterday = spark.read.format("delta") \
    .option("versionAsOf", 42) \
    .load("s3://data-lake/user_features/")

df_last_week = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-10 00:00:00") \
    .load("s3://data-lake/user_features/")

# Schema evolution: add a column without breaking existing readers
spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")
df_with_new_col.write.format("delta").mode("append").save("s3://data-lake/user_features/")

# Audit: show all transactions
delta_table.history().show(10)
```

---

### Apache Iceberg vs Delta Lake

```
Feature               | Delta Lake               | Apache Iceberg
----------------------|--------------------------|---------------------------
Originated at         | Databricks               | Netflix
Open standard         | Partial (Delta spec open)| Fully open (Apache)
Engine support        | Spark, Databricks, Flink | Spark, Trino, Flink, Hive,
                      | (Trino limited)          | Impala, Dremio
ACID transactions     | Yes                      | Yes
Time travel           | Yes (version/timestamp)  | Yes (snapshot/timestamp)
Schema evolution      | Additive + rename        | Full (add, rename, drop,
                      |                          | type widen)
Partition evolution   | No (re-write required)   | Yes (hidden partitioning,
                      |                          | no re-write)
Row-level deletes     | Yes (merge)              | Yes (copy-on-write or
                      |                          | merge-on-read)
Multi-table ACID      | No                       | No (table-level only)
Metadata scalability  | Transaction log grows    | Manifest files; O(1) for
                      | unbounded; need VACUUM   | table scan planning
Cloud-agnostic        | Yes                      | Yes
```

**When to choose Iceberg over Delta Lake**:
- Multi-engine shops (Trino for ad-hoc + Spark for ETL + Flink for streaming)
- Tables with frequent partition scheme changes (Iceberg's hidden partitioning avoids rewrites)
- Very large tables where Delta transaction log size becomes a bottleneck (>100K transactions)

**When to choose Delta Lake**:
- Databricks-centric stack (native integration, Z-ordering, Auto Optimize)
- Streaming + batch unification on Spark (DeltaStream API)
- Simpler operational model for small-to-medium teams

---

## ETL vs ELT

### Transform Upstream vs In the Warehouse

```
Pattern | Where transformation happens | When to use
--------|------------------------------|------------------------------------------
ETL     | Before loading into target   | Sensitive data (PII redaction before load),
        |                              | target has limited compute, streaming pipelines
ELT     | After loading into target    | Cloud warehouses (BigQuery, Snowflake, Redshift);
        |                              | iterative transformation; raw data preservation
```

**ETL problems**: transformation logic is coupled to ingestion. Schema changes in source require updating the transformation layer before data lands. The raw data is not preserved — you can't reprocess with a new transformation.

**ELT advantages**: load raw data as-is, transform with SQL inside the warehouse. Transformations are versioned SQL, easy to iterate. Reprocessing means re-running SQL, not replaying ingestion.

---

### dbt: ELT Transformation Layer

**dbt** (data build tool) treats SQL transformations as a DAG of version-controlled models with dependency management, testing, and documentation.

```sql
-- models/silver/user_30d_features.sql
-- dbt model: depends on bronze raw events table

{{ config(
    materialized='incremental',
    unique_key='user_id',
    partition_by={'field': 'feature_date', 'data_type': 'date'},
    cluster_by=['user_id']
) }}

WITH daily_events AS (
    SELECT
        user_id,
        DATE(event_time) AS event_date,
        COUNT(*) AS daily_event_count,
        SUM(amount) AS daily_spend
    FROM {{ ref('bronze_raw_events') }}
    {% if is_incremental() %}
    WHERE event_time > (SELECT MAX(feature_date) FROM {{ this }})
    {% endif %}
    GROUP BY 1, 2
)

SELECT
    user_id,
    CURRENT_DATE AS feature_date,
    SUM(daily_event_count) AS event_count_30d,
    SUM(daily_spend) AS total_spend_30d,
    AVG(daily_spend) AS avg_daily_spend_30d
FROM daily_events
WHERE event_date >= CURRENT_DATE - INTERVAL 30 DAY
GROUP BY 1, 2
```

```yaml
# models/silver/schema.yml — dbt tests as data contracts
models:
  - name: user_30d_features
    columns:
      - name: user_id
        tests:
          - not_null
          - unique
      - name: event_count_30d
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
```

---

### Medallion Architecture

**Bronze → Silver → Gold** layers represent increasing levels of data quality and transformation:

```
Layer  | Alias      | Content                        | Format          | Consumers
-------|------------|--------------------------------|-----------------|--------------------
Bronze | Raw        | Exact copy of source data,     | JSON, Avro,     | Silver pipelines,
       |            | append-only, immutable         | raw Parquet     | data debugging
Silver | Cleaned    | Deduplicated, validated,       | Delta/Iceberg   | Gold pipelines,
       |            | type-cast, PII redacted        | Parquet         | data scientists
Gold   | Aggregated | Business-level aggregates,     | Delta/Iceberg   | ML models,
       |            | feature tables, report-ready   | Parquet         | dashboards, APIs
```

**The key rule**: never write back from a higher layer to a lower layer. Bronze is immutable source of truth — if Silver logic is wrong, fix Silver and reprocess, never patch Bronze.

---

## Data Warehousing for ML

### OLAP vs OLTP

```
Property           | OLTP (Postgres, MySQL)     | OLAP (BigQuery, Snowflake, Redshift)
-------------------|---------------------------|--------------------------------------
Query pattern      | Point lookups, small rows | Full column scans, aggregations
Row count per query| 1–100                     | Millions–billions
Write pattern      | Frequent, single-row      | Bulk append, infrequent updates
Schema             | Normalized (3NF)          | Denormalized (star/snowflake)
Index strategy     | B-tree on primary keys    | Partitioning + clustering/Z-order
Storage format     | Row-oriented              | Columnar (internal)
Use in ML          | Feature serving (online)  | Training data, feature backfilling
```

---

### Partitioning, Z-Ordering, and Materialized Views

**Partitioning**: divides a table into independent files by a column value. Enables partition pruning — queries with a filter on the partition column skip entire file sets.

```sql
-- BigQuery: partition by date, cluster by user_id
CREATE TABLE ml_features.user_daily_features
PARTITION BY feature_date
CLUSTER BY user_id AS
SELECT user_id, feature_date, purchase_count_30d, avg_spend
FROM ml_features.silver_events
GROUP BY 1, 2;

-- Query hits only one partition (2024-01-15), not full table
SELECT * FROM ml_features.user_daily_features
WHERE feature_date = '2024-01-15' AND user_id = 'u123';
```

**Z-ordering** (Delta Lake): multi-dimensional locality optimization. Rewrites files so rows with similar values in specified columns are co-located, improving predicate pushdown for those columns.

```python
# Delta Lake: Z-order by columns commonly used in filters
delta_table.optimize().executeZOrderBy("user_id", "event_date")
# After Z-ordering, queries filtering on user_id + event_date skip ~90% of files
```

**Materialized views**: pre-computed query results stored as a table, refreshed on a schedule. Use for expensive aggregations that many downstream pipelines need.

```sql
-- Snowflake: materialized view for 30-day user features
CREATE MATERIALIZED VIEW user_30d_features_mv AS
SELECT
    user_id,
    COUNT(*) AS purchase_count_30d,
    AVG(amount) AS avg_spend_30d,
    COUNT(DISTINCT merchant_id) AS distinct_merchants_30d
FROM transactions
WHERE transaction_date >= CURRENT_DATE - 30
GROUP BY user_id;
-- Snowflake auto-refreshes when base table changes
```

---

### Feature Backfilling

**The problem**: a new feature is defined today, but you need it for all historical training data going back 2 years. The feature doesn't exist in the offline store for past dates.

```python
# Backfill: compute feature values for all historical dates
# Run once; can take hours on large datasets

from datetime import date, timedelta

start_date = date(2022, 1, 1)
end_date = date(2024, 1, 1)
current_date = start_date

while current_date <= end_date:
    date_str = current_date.strftime("%Y-%m-%d")

    # Read only the data available as of current_date (no future leakage)
    historical_data = spark.read.parquet("s3://data-lake/transactions/") \
        .filter(col("transaction_date") <= date_str) \
        .filter(col("transaction_date") >= (current_date - timedelta(days=30)).strftime("%Y-%m-%d"))

    features = historical_data.groupBy("user_id").agg(
        count("*").alias("purchase_count_30d"),
    ).withColumn("feature_date", lit(date_str))

    features.write.format("delta") \
        .mode("append") \
        .partitionBy("feature_date") \
        .save("s3://feature-store/user_features/")

    current_date += timedelta(days=1)
```

**Backfill gotcha**: when backfilling, you must respect the data horizon — only use data available before each target date. Computing a feature for Jan 15 using data from Jan 20 is leakage that will inflate training metrics but not manifest at serving time.

---

## Cost Optimization

### Compression Ratios

Columnar formats with good compression routinely achieve 5-10× compression on ML feature data because feature columns contain repeated or correlated values.

```
Format + Compression  | Typical Ratio | Read Speed  | Notes
----------------------|---------------|-------------|-------------------------------------
Parquet + Snappy      | 3-5×          | Fastest     | Best for frequently-read hot data
Parquet + Zstd (L3)   | 5-8×          | Fast        | Best balance; use for feature stores
Parquet + Zstd (L9)   | 7-12×         | Slower write| Cold/archive storage
ORC + Zlib            | 6-10×         | Medium      | Hive-native; good dictionary encoding
CSV (uncompressed)    | 1×            | Slow        | Never use in production at scale
```

---

### Partition Pruning and Caching

**Partition pruning** — the single highest-impact cost optimization. A table with 3 years of daily partitions scanned without a date filter reads 1095× more data than a single-day query.

```python
# EXPENSIVE: full table scan (3 years × 10GB/day = 10TB read)
spark.read.parquet("s3://data-lake/events/").filter(col("user_id") == "u123")

# CHEAP: partition pruning (reads 1 day = 10GB)
spark.read.parquet("s3://data-lake/events/") \
    .filter(col("event_date") == "2024-01-15") \  # partition column → pruning
    .filter(col("user_id") == "u123")              # predicate pushdown within partition
```

**Caching strategies**:

```
Layer           | Technology         | Latency  | Cost     | Use For
----------------|--------------------|----------|----------|----------------------------
In-process cache| Spark .cache()     | <1ms     | Memory   | Iterative ML training loops
Distributed cache| Alluxio, S3 cache | 1-5ms    | Medium   | Hot feature data near compute
Object store    | S3/GCS             | 50-200ms | Low      | Cold training data
Result cache    | Trino/BigQuery     | <10ms    | Low      | Repeated identical queries
```

```python
# Spark: cache a DataFrame that will be used multiple times in a training loop
features_df = spark.read.parquet("s3://feature-store/user_features/")
features_df.cache()  # or .persist(StorageLevel.MEMORY_AND_DISK)

# Multiple model training iterations reuse cached data without re-reading S3
for hyperparams in hyperparameter_grid:
    model = train(features_df, hyperparams)  # no S3 read after first iteration
```

---

### Compute vs Storage Tradeoffs

```
Decision                          | Cost-Efficient Choice
----------------------------------|------------------------------------------------------------
Store raw + recompute features    | Store pre-computed features if recompute > 15 min daily
Materialize all aggregations      | Materialize only features used by >1 model/pipeline
Large instance vs many small      | Many small instances (better parallelism, spot pricing)
Spark on-demand vs always-on      | On-demand for batch ETL; reserved for streaming
Parquet vs Delta Lake overhead    | Delta adds ~5% storage for logs; worth it for ACID + TT
Query result caching (BigQuery)   | Enable for dashboards; disable for ML pipelines (stale data)
```

**Spot/preemptible instances for ML pipelines**: batch ETL jobs are ideal for spot instances — they can be checkpointed and resumed. Use checkpointing with Delta Lake to avoid reprocessing from scratch on preemption.

```python
# Spark with checkpoint on spot instances
spark.sparkContext.setCheckpointDir("s3://checkpoints/spark/")

rdd = spark.sparkContext.parallelize(large_dataset)
rdd.checkpoint()  # checkpoint after expensive transformation
rdd.count()       # trigger checkpoint write to S3
# If spot instance is preempted, restart from checkpoint, not beginning
```

---

## Interview Questions

**Q1: Your training pipeline takes 6 hours and costs $500 per run. How would you reduce both?**

Start with profiling — most jobs spend 80% of time on 20% of stages. Check for full table scans (missing partition filters), data skew (one task takes 5h while others take 5min), and excessive shuffles (unnecessary wide transformations).

Tactical fixes: add partition filters on date columns to enable pruning (often 10-100× speedup); broadcast small lookup tables instead of shuffle-joining them; use `repartition(n, "join_key")` before expensive joins to co-locate keys; cache intermediate DataFrames used in multiple branches. Switch from `groupByKey` to `reduceByKey` on RDDs to do partial aggregation before shuffle.

For cost: move batch ETL to spot/preemptible instances with checkpointing. Switch from Parquet+Snappy to Parquet+Zstd to reduce data scanned per query.

---

**Q2: How do you detect and handle schema drift in an upstream data feed?**

Schema drift = a column is added, removed, renamed, or type-changed upstream without warning. Silent drift causes null feature values or type errors in training pipelines.

Detection: validate schema on every pipeline run against a registered schema (Great Expectations `expect_column_to_exist`, `expect_column_values_to_be_of_type`). Store the expected schema in a data contract YAML versioned in Git. Alert (PagerDuty/Slack) on any schema mismatch before data is written to the feature store.

Handling: distinguish additive vs breaking changes. New nullable columns → safe to allow with `mergeSchema=true` in Delta Lake. Renamed columns or type changes → block pipeline, alert owner, require explicit contract version bump. Dropped required columns → block immediately.

---

**Q3: Explain the difference between Lambda and Kappa architectures. When would you choose each?**

Lambda maintains a batch layer (reprocesses all historical data for accuracy) and a speed layer (processes recent data with low latency). Queries merge results from both. The fundamental problem is that the same feature must be computed in two different codebases (e.g., Spark batch + Flink streaming), and they must produce identical results — which they often don't due to subtle differences in handling late data, time zones, or null values.

Kappa eliminates the batch layer. The stream processor handles everything, including historical replay by consuming the event log from the beginning. This works when (1) the stream processor can sustain batch throughput during replay, (2) the event log retains sufficient history (Kafka with long retention), and (3) you can afford reprocessing time during a logic change.

Choose Lambda when: strict historical accuracy is required and the stream processor can't guarantee it (e.g., late-arriving events beyond the watermark window), or when you need a trusted batch baseline to validate streaming output.

Choose Kappa when: operational simplicity is paramount, you have a Kafka-based architecture with sufficient retention, and your stream processor (Flink) handles reprocessing efficiently.

---

**Q4: What is the difference between Delta Lake and Apache Iceberg? When would you choose one over the other?**

Both provide ACID transactions, time travel, and schema evolution on top of Parquet files in object storage. The key differences:

Delta Lake has a transaction log (`_delta_log/`) that grows as a sequence of JSON files; it requires periodic `VACUUM` to prevent unbounded log growth. Iceberg uses a snapshot + manifest file hierarchy that is more metadata-scalable for very large tables.

Iceberg has superior partition evolution: you can change the partition scheme without rewriting data. Delta Lake requires a full table rewrite to change partitioning.

Iceberg has broader engine support (Trino, Hive, Impala, Flink, Spark). Delta Lake has tighter Databricks integration with features like Z-ordering and Auto Optimize that aren't in the open Delta spec.

Choose Delta Lake for Databricks-centric stacks or when you need Z-ordering. Choose Iceberg for multi-engine shops (Trino + Spark + Flink) or tables that undergo frequent partition scheme changes.

---

**Q5: How do you ensure reproducibility of a training run from 6 months ago?**

Reproducibility requires pinning four things: code, data, environment, and random seeds.

Code: Git commit hash in the training job metadata. Data: DVC pointer files that map Git commits to exact data versions (MD5 hash of the training dataset). Alternatively, use Delta Lake or Iceberg time travel — store the snapshot ID or timestamp used for training in MLflow experiment metadata. Environment: Docker image SHA or conda lockfile committed to the repo. Random seeds: set and log `numpy`, `torch`, and `sklearn` seeds explicitly.

The MLflow run should record: git_commit, data_version (DVC hash or Delta snapshot ID), docker_image_sha, seed, and all hyperparameters. Given these, any engineer can reproduce the exact training run.

---

**Q6: A downstream model starts degrading. You suspect a data quality issue. How do you debug it?**

Start at the output and work backwards. Check the model's input feature distribution in the last 24-48 hours vs baseline (feature distribution monitoring). Look for sudden changes: a feature that was 95% non-null dropping to 60% non-null; a feature whose mean shifted by >3σ; a feature that's entirely zero or entirely null (upstream feed stopped).

Check the pipeline execution logs: did the ETL job complete? Did it skip any data due to a filter? Did the partition count drop significantly? Look at row counts per partition over time — a sudden drop indicates a missing upstream feed.

Use dbt test results or Great Expectations checkpoints to see which expectations failed and when. Check the data contract freshness SLA — if the pipeline was late, the model consumed stale features.

If you have feature logging at serving time (you should), compare the served feature distribution to what the feature store holds. A divergence indicates training-serving skew introduced by a transformation change.

---

**Q7: You need to backfill 2 years of a new feature. What are the risks and how do you handle them?**

The primary risk is temporal leakage: when computing the feature for a historical date, accidentally using data from after that date. For example, computing "user's 30-day purchase count as of Jan 15, 2022" but including purchases from Jan 16–Feb 15 because you read the full table and filtered incorrectly.

Mitigation: for each target date in the backfill range, explicitly filter the source data to `transaction_date <= target_date` AND `transaction_date >= target_date - 30 days`. Never read from the "current" snapshot of the table when computing historical features.

Secondary risks: cost (2 years × daily compute can be expensive — estimate and use spot instances); partial writes (if the job fails mid-backfill, use Delta Lake MERGE or idempotent writes partitioned by date to resume safely); pipeline conflicts (backfill job competing with production job for cluster resources — schedule during off-peak or use a separate cluster).

After backfilling, validate a sample: take 10 random (user_id, date) pairs, manually compute the feature from raw data, and compare to the backfilled values. This catches off-by-one errors in the time window logic before the feature is used in training.

## Flashcards

**Why is Lambda architecture operationally expensive compared to Kappa?** #flashcard
Lambda requires two codebases (batch + speed layer) that must produce identical results for the same feature; schema changes require coordinated updates in both paths. Kappa uses one streaming codebase and replays the event log for historical reprocessing.

**What does Kappa architecture require from the stream processor and event log to work?** #flashcard
The stream processor must sustain batch-level throughput during replay, and Kafka retention must cover the full reprocessing window (weeks or months of logs) so historical data isn't lost.

**What's the modern default choice between Lambda and Kappa, and when do you fall back?** #flashcard
Prefer Kappa (Flink or Spark Structured Streaming). Fall back to Lambda only when exact historical correctness is required that streaming can't guarantee — e.g., late-arriving data beyond the watermark.

**Iceberg vs. Delta Lake — name one scenario favoring each.** #flashcard
Iceberg: multi-engine shops (Trino + Spark + Flink) or tables with frequent partition scheme changes (hidden partitioning avoids rewrites). Delta Lake: Databricks-centric stacks needing Z-ordering/Auto Optimize, or Spark-unified streaming+batch.

**Why must a feature backfill filter strictly on `transaction_date <= target_date` rather than reading the current table snapshot?** #flashcard
Reading the full/current table and filtering incorrectly can leak future data into a historical feature value (temporal leakage) — inflating offline metrics without a corresponding signal at serving time.

**dbt: what do `not_null`, `unique`, and `accepted_range` tests enforce, and where do they run?** #flashcard
They enforce data-contract-like guarantees (non-null values, uniqueness, valid numeric ranges) on model output columns, run as part of the dbt DAG so a violation fails the pipeline before bad data reaches downstream consumers.
