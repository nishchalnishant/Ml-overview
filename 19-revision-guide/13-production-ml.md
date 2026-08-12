---
module: Production ML
topic: Revision Guide
subtopic: "13 Production ML"
status: unread
tags: [productionml, ml, revision-guide, mlops, deployment, feature-store]
---

# 13 Production ML — Revision Guide

Condensed revision pass over the whole `13-production-ml/` folder. Use the outline to rebuild the topic map from memory, then work the graded Q&A to stress-test recall and applied reasoning before pressure-testing gaps against the source files.

## Structure Overview

```markdown
- **13 Production ML**
  - **MLOps**
    - MLOps vs DevOps
    - Maturity Levels 0-2
    - Data Versioning DVC
      - Offline Store
    - Great Expectations
    - Experiment Tracking
    - CI CD for ML
    - Retraining Triggers
    - ML Design Patterns
      - Quantization PTQ QAT
  - **Deployment Patterns**
    - Blue-Green
    - Canary
      - Canary Health Checks
    - Shadow
      - Incident Severity P1-P4
    - A/B Test
    - Big-Bang
    - REST vs gRPC
    - Dynamic Batching
    - ONNX TensorRT
    - Edge Inference
    - LLM Serving
  - **Model Governance**
    - Registry Metadata
    - Approval Gates
    - Audit Trails
    - GDPR Right to Erasure
    - Model Cards
    - Semantic Versioning
  - **Real-Time ML Systems**
    - Latency Tiers
    - Kafka Delivery Semantics
    - Flink vs Spark Streaming
    - Online Feature Store
      - Online Store Redis DynamoDB
    - Point-in-Time Correctness
      - Online Offline Parity
    - Dual-Write Problem
      - Idempotency
    - Latency Budget P99
    - Fallback Hierarchy 5-level
    - Circuit Breaker
    - Feedback Loops
    - Multi-Objective Tradeoffs
  - **Streaming Pipelines**
    - 5-Layer Architecture
    - Partitioning Salting
    - Watermarks
      - Exactly-Once Semantics
    - Windowed Aggregation
    - Little's Law Capacity
    - Consumer Lag Monitoring
  - **Data Engineering for ML**
    - Lambda vs Kappa
    - Spark Internals
    - Storage Formats Parquet Avro
    - Delta Lake ACID
    - Iceberg vs Delta
    - ETL vs ELT
    - Medallion Architecture
    - Data Contracts
    - Anomaly Detection z-score
    - Feature Backfilling
  - **Feature Store Architecture**
    - Feature Registry
    - Transformation Layer
    - Feature Freshness SLA
    - PSI Feature Monitoring
    - Build vs Buy Threshold
  - **Embedding Pipelines**
    - Batch Embedding Generation
    - ANN Index FAISS
      - HNSW vs IVF vs Flat
    - Embedding Compression PQ
    - Incremental Index Updates
    - Embedding Cache TTL
    - Cold-Start OOV Handling
  - **Model Registry Versioning**
    - MLflow Registration
    - Stage Transitions
    - Champion Challenger
    - Automatic Rollback
    - Model Lineage
  - **Production Diagnosis**
    - Sudden vs Gradual Accuracy Drop
    - TTFT vs TPOT
    - Training Serving Skew Detection
    - Data Drift vs Concept Drift
    - Retraining Trigger Types
    - Root Cause Ranking
```

## Questions

### Easy

1. In one sentence, what distinguishes MLOps from traditional DevOps?
2. What is the difference between MLOps maturity level 0 and level 1?
3. Name three things DVC versions besides code.
4. What does Great Expectations do in a data pipeline?
5. What's the core difference between shadow deployment and canary deployment?
6. Why is blue-green deployment considered the "safest but most expensive" rollout strategy?
7. Define online inference vs batch inference in one line each.
8. What four fields must a minimal model registry entry track?
9. What is a model card, and who is the intended audience?
10. Name the four canonical incident severity levels (P1-P4) and what distinguishes P1 from P4.
11. What does semantic versioning's MAJOR.MINOR.PATCH mean in an ML model context?
12. What is the "training-serving skew" problem in one sentence?
13. What is point-in-time correctness, informally?
14. Name the four components of a feature store.
15. What's the difference between an offline store and an online store in a feature store?
16. What does PSI (Population Stability Index) measure?
17. Name three storage formats commonly used in ML data pipelines and one property that distinguishes them.
18. What is the Medallion architecture (Bronze/Silver/Gold)?
19. What problem does a circuit breaker pattern solve?
20. What is the difference between data drift and concept drift, at a high level?
21. What does TTFT stand for in LLM serving, and what stage of inference does it correspond to?
22. What is dynamic batching and why does it help throughput?
23. What's the purpose of an ANN (approximate nearest neighbor) index versus exact KNN?
24. Name two ANN index types and one property of each.
25. What is a champion/challenger pattern in model deployment?
26. What triggers a "scheduled" retraining policy versus a "drift-based" one?
27. What is exactly-once semantics in a streaming pipeline, informally?
28. Why would you salt a Kafka partition key?
29. What does a watermark do in stream processing?
30. What's the simplest fallback when an online feature store call times out?

### Medium

1. Why does canary deployment require automated guardrail metrics (error rate, latency, business KPI) rather than just manual observation?
2. Compare shadow deployment and A/B testing: which one exposes real users to risk, and why would you still eventually need the other?
3. Walk through why a feature computed with a 30-day rolling window in a nightly batch job can silently diverge from the same feature computed in real time at serving.
4. Explain the "dual-write consistency problem" between a database and a feature store, and why event sourcing / Kafka-first is often the pragmatic fix.
5. Why is P99 latency a more useful SLO than average latency for a real-time ML system, and what causes a growing P99/P50 gap?
6. Derive why TTFT is compute-bound (prefill) while TPOT is memory-bandwidth-bound (decode) in LLM serving.
7. Explain how a KS test-based skew detector (`detect_skew`) can catch training-serving skew that a simple mean/variance check would miss.
8. Rank the top three root causes of training-serving skew and explain why aggregation window mismatch typically outranks stale normalization stats.
9. Why does Little's Law matter for sizing a Kafka consumer group's parallelism?
10. Explain how a hot partition forms in Kafka and why key salting fixes it but complicates downstream aggregation.
11. Contrast Lambda and Kappa architecture: what operational cost does Kappa remove, and what does it assume about your stream processor's replay capability?
12. Why does Delta Lake's time-travel feature help debug a "silent gradual accuracy drop" incident?
13. Explain predicate pushdown vs partition pruning and why both matter for cost, not just speed.
14. Why is online/offline feature parity monitored with a KS test at the aggregate level insufficient — what does it miss?
15. Design the minimal architecture of a point-in-time-correct feature join, and explain what breaks if you join on wall-clock time instead of event time.
16. Why does HNSW trade memory for query speed compared to IVF, and when would you prefer IVF-PQ instead?
17. Explain why embeddings are stored in fp16 with L2-renormalization after PCA compression rather than raw fp32.
18. Walk through the five-level fallback hierarchy in a real-time ML system (from primary model down to hard-coded default) and give a concrete trigger for skipping straight to level 4.
19. Why does a canary rollout for a fraud model need a guardrail on false-positive rate specifically, not just error rate and latency?
20. Explain how automatic rollback (`trigger_automatic_rollback`) finds the correct prior version, and why "most recent Archived" is the right selection rule.
21. What does ADWIN detect that a static PSI threshold check does not?
22. Why is "AUC improved but CTR stayed flat" a classic sign of a ranking model change hitting a capped candidate set or a ceiling effect rather than a broken model?
23. Explain why ingesting late-arriving data can violate point-in-time correctness even when the feature store enforces event-time joins.
24. Why would a team choose a custom feature store build over Feast/Tecton only above roughly 100TB of data or 100M lookups/day?
25. Contrast Iceberg and Delta Lake on schema evolution and multi-engine support — when does the choice actually matter operationally?
26. Explain why continuous batching + PagedAttention improve LLM serving throughput without hurting per-request latency much.
27. Why does a data contract (schema + SLA as a YAML artifact) reduce incidents more than ad hoc Slack alerts about broken pipelines?
28. Explain why an incremental ANN index (delta index + periodic merge) is preferred over rebuilding the full index on every update.
29. Why does feature backfilling risk label leakage, and how do you structure the backfill job to avoid it?
30. What operational signal distinguishes "P50 latency unchanged, only P99 moved" from a systemic latency regression, and what's the likely cause class?

### Hard

1. A fraud model's shadow-mode metrics look identical to production for two weeks, then diverges sharply after cutover — what classes of bugs does shadow mode structurally fail to catch, and how would you redesign the validation to catch them pre-cutover?
2. You observe PSI flagging drift on 40 of 200 features every day at your traffic volume, but the model's business metric is stable. Is this a data quality signal or a statistical artifact of p-value/threshold sensitivity at scale, and how do you redesign the alerting to separate the two?
3. Design a fallback for a real-time fraud-scoring path when the online feature store degrades to 3x its normal p99 latency but has not fully failed. Walk through each of the 5 fallback levels and state the concrete latency/error threshold at which you'd trip to the next level down.
4. A feature store's online store and offline store are both fed by the same Kafka topic, but the online store also writes computed aggregates back into the offline warehouse for convenience. Explain the failure mode this creates and why "online writes back to offline" is considered an anti-pattern for parity monitoring.
5. Your canary rollout passes all guardrails (error rate, p99 latency, PR-AUC) at 5% traffic but the false-positive rate regresses only after reaching 50% traffic. What does this imply about the traffic-splitting strategy or the guardrail's statistical power, and how would you redesign the rollout steps?
6. A model's aggregation window for a feature straddles the label cutoff time inconsistently between training (computed at midnight UTC) and serving (computed at request time in the user's local timezone). Trace exactly how this produces silent skew that a naive point-in-time join test would not catch.
7. Compare CDC, Event Sourcing, and Saga pattern for solving the dual-write problem between an OLTP database and a feature store — under what traffic/consistency requirements does each become the wrong choice, not just a suboptimal one?
8. You are asked to design retraining triggers combining scheduled, performance-based, and drift-based policies for a model with delayed labels (30-day lag). Explain what proxy signal you'd monitor in the interim and why relying purely on drift-based triggers is dangerous here.
9. In an embedding-based retrieval system, a shard holding 20% of the item catalog goes down. Explain how graceful degradation should behave differently for a head-heavy popularity distribution versus a long-tail catalog, and what the ANN recall impact looks like in each case.
10. Explain why quantizing a ranking model to INT8 can pass all offline accuracy checks but degrade the online business metric — connect this to calibration drift and threshold-based decision boundaries.
11. A circuit breaker trips to OPEN on a feature-lookup dependency, correctly protecting the ML service — but three minutes later HALF_OPEN probes keep failing and flapping back to OPEN. Diagnose two distinct root causes that produce this flapping pattern and how you'd distinguish them from logs alone.
12. Design a rollback-safe migration path for a model registry moving from MLflow's stage-based model (None/Staging/Production/Archived) to an alias-based model — what breaks in your champion/challenger router and canary automation during the cutover window, and how do you sequence the migration to avoid a rollback deadlock?
13. Explain how exactly-once semantics in a Flink pipeline can still produce duplicate feature writes downstream in a Redis online store, and what idempotency key design at the sink prevents it.
14. A model exhibits stable PSI across all features but a slow, multi-week decline in business metric with no infrastructure changes and no upstream schema changes logged. Enumerate at least four non-drift explanations you would rule out before concluding this is concept drift.
15. Design the point-in-time-correct backfill for a new feature that requires a 90-day trailing aggregate, where the raw event log only has reliable retention for 45 days. What are your options, and what bias does each introduce into the training set versus the eventual serving distribution?
16. Two teams' models share a feature store; team A silently changes the null-handling logic in a shared feature's transformation layer to fix a bug for their own model. Trace the blast radius on team B's model, why the parity/skew monitors might not catch it for weeks, and what registry/ownership design would have prevented it.
17. In a multi-objective ranking system balancing engagement, revenue, and fairness constraints, explain why naive linear scalarization of the objectives can produce a rollout that passes every individual guardrail metric yet is rejected by the Pareto-efficiency check — give a concrete numeric scenario.
18. Your production diagnosis flowchart says "check what changed first" — but three independent changes (a feature pipeline deploy, a label-source migration, and a traffic-mix shift from a new marketing channel) landed within the same 6-hour window as an accuracy drop. Design a diagnostic procedure to isolate which change (or combination) is causal, using only production logs and no ability to roll back changes independently.
19. Explain precisely why HNSW's approximate recall degrades disproportionately on tombstoned (soft-deleted) items compared to IVF, and what index maintenance policy avoids the compounding degradation.
20. A/B test on a new ranking model shows a statistically significant lift, but SRM (sample ratio mismatch) checks reveal a 51/49 split against an intended 50/50. Under what conditions is the lift still trustworthy, and under what conditions must the experiment be discarded outright?
21. Design an incident-response runbook for "P99 latency spike, all individual pipeline stage p99s look normal, but end-to-end p99 has doubled" — what architectural cause class produces this exact symptom pattern, and what instrumentation would you add going forward to catch it earlier?
22. In a streaming feature pipeline using tumbling windows, a burst of late-arriving events (beyond the watermark's allowed lateness) is silently dropped. Explain the downstream effect on a model whose training pipeline uses complete offline recomputation (no watermark drop) — what specific skew signature would `detect_skew` show, and would it show up as a mean shift or a distributional shape difference?
23. You must choose between Feast (self-hosted), Tecton (managed), and a custom-built feature store for a platform expecting to scale from 5M to 150M daily lookups within 12 months. Walk through the decision at each scale checkpoint, including what triggers a re-evaluation mid-year.
24. Explain how label leakage can enter a model through feature backfilling even when the backfill job correctly respects point-in-time cutoffs for the feature values themselves — identify the subtler leakage vector.
25. A champion/challenger router uses deterministic hash-based routing on user ID for consistency. Explain why this breaks the assumption of a standard two-sample statistical test comparing champion vs challenger outcomes, and what test design correctly accounts for it.
