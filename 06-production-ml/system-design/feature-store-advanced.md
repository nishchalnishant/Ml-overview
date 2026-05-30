---
module: Production Ml
topic: System Design
subtopic: Feature Store Advanced
status: unread
tags: [productionml, ml, system-design-feature-store-ad]
---
# Feature Store — Advanced Topics

> All advanced feature store content (online/offline parity testing, streaming feature computation with Flink, feature freshness vs latency trade-offs, feature monitoring with PSI, Feast vs Tecton comparison, and canonical interview Q&As) has been consolidated into [feature-store-architecture.md](./feature-store-architecture.md) to eliminate duplication.

See `feature-store-architecture.md` for:
- Online/offline parity testing with `check_online_offline_parity()`
- `TxnVelocityAgg` Flink `AggregateFunction` for streaming windows
- Streaming backfill strategy
- Feature freshness vs latency decision table
- PSI-based feature drift monitoring
- Feature freshness SLA monitoring
- Open-source tool comparison (Feast, Tecton, Hopsworks, Vertex AI FS, SageMaker FS)
- Canonical interview Q&As (point-in-time correctness, parity, streaming vs batch, backfill)
