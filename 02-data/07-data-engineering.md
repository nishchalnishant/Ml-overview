---
module: Data
topic: Data Engineering
subtopic: ""
status: unread
tags: [data-engineering, batch, streaming, kafka, spark, airflow, dbt, feature-store, data-quality, mlops]
prerequisites: [python-and-data-tooling, data-processing-and-eda]
---
# Data Engineering for ML

> **Moved.** This topic now lives in [13-production-ml/06-data-engineering-for-ml.md](../13-production-ml/06-data-engineering-for-ml.md), consolidated with the deeper production-scale treatment (distributed processing, storage formats, data quality frameworks, versioning, cost optimization) so system-design interview prep has one authoritative source instead of two overlapping ones.

See that file for: batch vs. streaming and Lambda/Kappa architecture, pipelines and orchestration (Airflow, dbt, idempotency), warehouse/lake/lakehouse, feature stores (offline/online split, train-serve skew, point-in-time correctness), and data quality/contracts.
