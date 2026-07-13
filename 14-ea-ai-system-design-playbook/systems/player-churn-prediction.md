# Player Churn Prediction

## 1. Problem Framing

- Predict probability a player stops engaging with a live-service title (Battlefield, Apex-style shooter, FC Ultimate Team) within a future window (7/14/30 days), so retention teams can intervene before the player leaves.
- Consumers: CRM/lifecycle marketing (push, in-game offers), live-ops (difficulty/economy tuning), player support, exec dashboards.
- Two operating modes: (a) nightly batch scores for the full active base feeding CRM campaigns, (b) near-real-time scores on session-end/key events (e.g., rage-quit after loss streak) for in-session intervention.
- Churn looks different per title (FC Ultimate Team vs. a live-service shooter), so per-title models sharing infrastructure, not one global model.
- Business KPI: lift in D30 retention for treated vs. control (holdout) cohorts; incrementality of offers (avoid paying to retain players who'd stay anyway).

## 2. Functional Requirements

- FR1: Nightly batch churn-probability score (0-1) per `player_id`/`title_id` for all players active in trailing 30 days.
- FR2: Near-real-time score refresh (< 5 min end-to-end) after key telemetry events (session end, purchase, loss streak, friend-list change).
- FR3: Low-latency read API for downstream consumers (CRM, live-ops, offer service).
- FR4: Trigger interventions (push, offer, matchmaking nudge) when score crosses a title-specific threshold, respecting frequency caps and holdout groups.
- FR5: Point-in-time correct training features to minimize offline/online skew.
- FR6: Per-title model versions; concurrent A/B testing of model versions/thresholds.
- FR7: Explainability (top contributing features) for support tooling and compliance review of automated offers.
- FR8: Fixed-cadence and drift-triggered retraining; rollback to prior model version.
- FR9: Log every scoring decision and intervention for auditability and incrementality measurement.

## 3. Non-Functional Requirements

| Dimension | Requirement |
|---|---|
| Latency (batch) | Full player-base scoring within nightly 4-hour window (01:00-05:00 UTC) |
| Latency (near-real-time) | p99 event-to-score < 5 min; p99 score-read API < 100 ms |
| Availability | Score-read API 99.9% (fallback to last-known score on failure); scoring pipeline 99.5% (batch retries next window) |
| Throughput | Batch: ~120M player-title rows/night; streaming: 8k events/sec sustained, 40k burst |
| Consistency | Eventual consistency OK for scores (staleness up to 24h batch / 5 min streaming); feature store must be point-in-time correct for training (no leakage) |
| Cost | Inference predominantly CPU (tree ensembles) to keep serving cost low |
| Durability | Feature/score history retained per policy; no data loss on write path (queue-backed) |

## 5. Assumptions

1. Flagship live-service title: 25M MAU, 6M DAU.
2. Churn = no login within 14 consecutive days, 30-day prediction horizon.
3. Telemetry flows into a central Kafka event bus already; this system is a consumer, not the ingestion owner.
4. ~150 telemetry events/session, ~2.3 sessions/day for active players.
5. Model family: gradient-boosted trees (LightGBM/XGBoost) for tabular features; DL (sequence models) considered as v2 uplift.
6. Feature store is shared infra also used by LTV, matchmaking, fraud systems — treated as integrated here.
7. Batch scoring: nightly, 4-hour SLA, Spark on EMR/Databricks-style cluster.
8. Near-real-time path targets only "high-value moment" triggers (post-session, refund, loss-streak ≥5), not every event, to control streaming cost.
9. Push/offer services are existing systems called via API, not built here.
10. One retrain/title/week baseline, plus drift-triggered ad hoc retrains.
11. Cost guardrail: inference infra cost < 3% of retention-lift dollar value generated.

## 6. Capacity Estimation

**Batch scoring:**
- 25M MAU, up to 5 titles sharing platform → 120M player-title rows/night.
- ~300 features × 4 bytes ≈ 1.2 KB/row → 144 GB raw feature read per run.
- LightGBM inference is cheap (~50k rows/sec/core); I/O and feature joins dominate. Provision a 64-node × 16-core cluster; target 45-60 min wall-clock, leaving headroom in the 4-hour window for retries.
- CPU-only cluster, no GPU needed for GBT inference.

**Near-real-time scoring:**
- Trigger-worthy events ≈ 15% of DAU sessions/day → 6M × 2.3 × 0.15 ≈ 2.07M requests/day.
- Avg QPS ≈ 24; peak (3x) ≈ 72 QPS; provision for 150 QPS burst.
- Compute budget p99 < 50ms; GBT inference itself is sub-ms to a few ms — latency is dominated by feature fetch, not compute.
- Model size: ~2,000 trees, depth 6 ≈ 15-40 MB/title. 5 titles × 2 versions (canary + stable) ≈ 400 MB total — fits easily on CPU pod RAM.

**Storage:**
- Offline feature store: 120M rows/day × 1.2 KB × 365 ≈ 52.6 TB/year raw, ~10.5 TB/year with Parquet compression (~5:1), partitioned by date in a data lake.
- Online feature store (latest snapshot only): 25M players × 5 titles × 1.2 KB ≈ 150 GB in a KV store (Redis/DynamoDB-class).
- Score history log: ~24.4 GB/day ≈ 8.9 TB/year raw, ~1.8 TB/year compressed, append-only.

**Compute footprint:**
- Weekly training: distributed LightGBM on CPU cluster (32 cores), ~2-3 hrs/run × 5 titles ≈ 10-15 CPU-cluster-hours/week. GPU only relevant if a v2 sequence-DL model is adopted.
- Serving fleet: ~12-20 CPU pods (4 vCPU/8GB) baseline, bursting to ~40 at peak.

## 7. High-Level Architecture

```
                         ┌────────────────────────────────────────────────┐
                         │              Game Clients (all titles)          │
                         └───────────────────────┬──────────────────────────┘
                                                  │ telemetry events
                                                  ▼
                         ┌────────────────────────────────────────────────┐
                         │        EA Central Telemetry Bus (Kafka)         │
                         │   topics: session.end, purchase, match.result,  │
                         │           social.friend, loss.streak            │
                         └───────────┬───────────────────────┬──────────────┘
                                     │                        │
                     (near-real-time path)              (batch path)
                                     │                        │
                                     ▼                        ▼
                  ┌──────────────────────────┐   ┌──────────────────────────────┐
                  │ Streaming Feature/Trigger │   │  Nightly Batch Feature ETL     │
                  │ Consumer (Flink/KStreams) │   │  (Spark on EMR/Databricks)     │
                  │ - windowed aggregates     │   │ - full feature recompute       │
                  │ - trigger rules           │   │ - joins w/ offline feature     │
                  └───────────┬───────────────┘   │   store history                │
                              │                     └───────────┬───────────────────┘
                              ▼                                 ▼
                  ┌──────────────────────────┐   ┌──────────────────────────────┐
                  │  Online Feature Store     │◄──┤  Offline Feature Store         │
                  │  (Redis/DynamoDB)         │   │  (Data Lake, Parquet, S3)      │
                  └───────────┬───────────────┘   └───────────┬───────────────────┘
                              │                                 │
                              ▼                                 ▼
                  ┌──────────────────────────┐   ┌──────────────────────────────┐
                  │  Near-Real-Time Model     │   │   Batch Scoring Job            │
                  │  Serving (KServe/Triton   │   │   (Spark + model broadcast)    │
                  │  CPU pods, GBT models)    │   │                                │
                  └───────────┬───────────────┘   └───────────┬───────────────────┘
                              │                                 │
                              └───────────────┬─────────────────┘
                                               ▼
                                 ┌───────────────────────────┐
                                 │   Score Store (write path)  │
                                 │  Cassandra/DynamoDB +        │
                                 │  Score History Log (Kafka →  │
                                 │  lake)                       │
                                 └────────────┬────────────────┘
                                              │
                                              ▼
                          ┌────────────────────────────────────────┐
                          │      Intervention Decision Engine        │
                          │  (threshold rules, holdout assignment,    │
                          │   frequency cap enforcement)              │
                          └───────────┬───────────────┬───────────────┘
                                      │               │
                                      ▼               ▼
                       ┌───────────────────┐  ┌────────────────────┐
                       │ Push Notification  │  │ In-Game Offer/CRM   │
                       │ Service (existing) │  │ Service (existing)  │
                       └───────────────────┘  └────────────────────┘

           ┌───────────────────────────────────────────────────────────────┐
           │  Training Pipeline (orchestrated via Airflow):                  │
           │  Offline Feature Store → Train/Val split (time-based) →         │
           │  LightGBM training → Eval vs. holdout → Model Registry →         │
           │  Canary deploy → Promote/Rollback                               │
           └───────────────────────────────────────────────────────────────┘

           ┌───────────────────────────────────────────────────────────────┐
           │  Monitoring/Drift/Alerting Plane: metrics from every box above  │
           │  → Prometheus/Grafana, drift jobs, PagerDuty routing            │
           └───────────────────────────────────────────────────────────────┘
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| Streaming Trigger Consumer | Windowed aggregates (session count, loss streak), decide if a scoring trigger fires | Kafka consumer group → `score.request` | Input topic partition count |
| Batch Feature ETL | Nightly recompute of full feature set from raw telemetry + offline store | Spark job, Parquet/Delta in/out | Executor count |
| Online Feature Store | Latest feature vector per player, < 10ms read | `GET /features/{player_id}/{title_id}` | Redis shards / DynamoDB partitions |
| Offline Feature Store | Historical, point-in-time correct snapshots for training | Spark/SQL batch read, Delta time-travel | Data lake, partitioned by date/title |
| Near-Real-Time Model Server | GBT inference on demand, < 50ms compute | gRPC `Predict(...) -> score` | Pod replicas behind HPA |
| Batch Scoring Job | Score all active players nightly | Spark job, broadcast model, bulk write | Executor count |
| Score Store | Durable low-latency current + recent scores | `GET/PUT /score/{player_id}/{title_id}` | Partition key = player_id |
| Intervention Decision Engine | Threshold + holdout + frequency-cap logic, calls channels | Consumes `score.updated`, calls Push/Offer APIs | Stateless, scales on backlog |
| Training Orchestrator | Schedule/execute retrain DAGs | Airflow DAG | One DAG run per title |
| Model Registry | Version/store model artifacts + metadata | MLflow-style API | Object storage + metadata DB |
| Drift Monitor | Feature/prediction/label drift metrics | Scheduled batch job | Scales with feature count |
| Explainability Service | SHAP/feature-contribution on demand | `GET /explain/{player_id}/{title_id}` | Stateless, CPU-bound |

## 9. API Design

**Score Read API** (CRM, live-ops, support tools):
```
GET /v1/scores/{title_id}/{player_id}
200: { player_id, title_id, churn_score: 0.734, score_horizon_days: 30,
       model_version, scored_at, source: "batch"|"near_real_time",
       top_features: [{feature, contribution}, ...] }
404: no score yet (cold-start)
503: score store degraded — client falls back to cached/last-known
```

**Batch fetch (CRM segment export):**
```
POST /v1/scores/{title_id}/batch  { player_ids: [...], max: 10000 }
→ { scores: [...], missing: [...] }
```

**Score write (internal, scoring jobs only):**
```
POST /internal/v1/scores/{title_id}  { player_id, churn_score, model_version, source }
→ 202 Accepted (async write to store + history log)
```

**Intervention decision (internal):**
```
POST /internal/v1/interventions/evaluate  { player_id, title_id, churn_score }
→ { action: "push_notification", campaign_id, holdout: false }
  | { action: "none", reason: "frequency_cap_exceeded" | "holdout_group" }
```

**Explainability:**
```
GET /v1/explain/{title_id}/{player_id}?score_id={id}
→ { shap_values, base_value, model_version }
```

**Versioning:** URI-based (`/v1/`, `/v2/`) for breaking changes; `model_version` tracked independently in the registry so model rollouts don't need API changes. 2 major versions supported concurrently, 90-day sunset.

## 10. Database Design

**Score Store — Cassandra/DynamoDB**: high write throughput for nightly bulk writes (120M rows) plus low-latency point reads.

```sql
CREATE TABLE scores (
  title_id text, player_id text, churn_score float, model_version text,
  scored_at timestamp, source text, top_features map<text, float>,
  PRIMARY KEY ((title_id, player_id))
);
```
Partition key `(title_id, player_id)` — high-cardinality player_id avoids hot partitions; title_id enables per-title isolation/monitoring.

**Offline Feature Store — Delta Lake/Parquet**: columnar, cost-efficient scans, time-travel for point-in-time correctness.
```
Table features_daily, partitioned by dt, title_id
Columns: player_id, dt, title_id, sessions_last_7d, sessions_last_30d,
         loss_streak_current, days_since_last_purchase, avg_session_length_min,
         friend_count_active, purchase_count_90d, ... (~300 cols)
```

**Online Feature Store — Redis/DynamoDB**: sub-10ms reads; only latest snapshot, not history.
```
Key: feat:{title_id}:{player_id}  Value: hash of ~300 fields, TTL 48h
```

**Model Registry — Postgres**: low-volume, relational (model → metrics → lineage), needs transactional consistency for promote/rollback.
```sql
CREATE TABLE model_versions (
  id SERIAL PRIMARY KEY, title_id TEXT, version TEXT, training_run_id TEXT,
  auc_offline FLOAT, logloss_offline FLOAT, artifact_uri TEXT,
  status TEXT, created_at TIMESTAMP DEFAULT now()
);
```

**Sharding:** Score store and online feature store shard by `player_id` hash for even distribution (titles vary 10x in population, so title_id isn't the primary shard key).

## 11. Caching

- **Score Read API**: cache-aside in front of Score Store, TTL 6h (batch scores tolerate ~24h staleness anyway). Miss falls back to Cassandra.
- **Online feature store is itself a write-through cache** over the offline store — features written immediately on compute, never lazily backfilled (would break the 5-min SLA).
- **Model artifacts**: cached in-process on serving pods; invalidated via registry webhook on promotion (rolling restart/hot-swap).
- **Invalidation**: score cache evicted on new near-real-time write (`score.updated` event) plus TTL; feature cache overwritten on every write, TTL as safety net for inactive players.
- No caching of raw telemetry — high write volume, low re-read value.

## 12. Queues & Async Processing

| Queue/Topic | Producer → Consumer | Delivery | Dead-Letter Handling |
|---|---|---|---|
| `telemetry.raw.*` | Clients → streaming consumer, batch ETL | At-least-once | Owned upstream by central telemetry team |
| `score.request` | Trigger consumer → NRT model server | At-least-once, idempotent on `(player_id, title_id, event_id)` | Retry 3x, then DLQ; alert if depth > 500 |
| `score.updated` | Both scoring paths → cache/decision/history consumers | At-least-once, dedupe via `scored_at` | DLQ + hourly reprocessing |
| `intervention.trigger` | Decision engine → Push/Offer adapters | At-least-once, idempotent on `campaign_id+player_id+day` | DLQ, live-ops alert, manual replay |

- Exactly-once isn't needed anywhere: scoring/interventions are idempotent by design, so at-least-once is simpler and cheaper.
- Backpressure: streaming consumer rate-limits/samples `score.request` under backlog — degrades gracefully to batch-only scoring rather than breaching SLA.

## 13. Streaming & Event-Driven Architecture

**Topics:** `session.end`, `match.result`, `purchase`, `social.friend_change`, `score.request`, `score.updated` (each keyed with player_id/title_id plus relevant payload fields and timestamp).

**Consumer groups:**
- `cg-streaming-features`: raw telemetry → windowed aggregates (Flink, keyed by player_id) → emits `score.request`.
- `cg-nrt-scorer`: `score.request` → feature store + model server → publishes `score.updated`.
- `cg-cache-invalidator`: `score.updated` → evict/refresh read cache.
- `cg-history-writer`: `score.updated` → append to score history log.
- `cg-decision-engine`: `score.updated` → evaluate intervention rules.
- Partition key across all topics: `player_id` — preserves per-player ordering (important for windowed loss-streak state) while allowing horizontal scale-out (e.g., 128 partitions/topic).

## 14. Model Serving

- **Framework**: KServe (or Seldon/Triton) on Kubernetes for near-real-time path, serving LightGBM via Treelite-compiled inference for speed.
- **Hardware**: CPU-only pods (4 vCPU/8GB) — tree ensembles don't benefit from GPU.
- **Batching**: not needed — per-request compute is sub-ms to a few ms, well under the 50ms budget. Batch path uses Spark's vectorized inference (UDF wrapping Treelite) instead.
- **Multi-model**: one deployment hosts all title models (fc25-churn-v14, apex-churn-v9, etc.) plus stable + canary versions, routed by `title_id`.
- **Model format**: exported to ONNX/Treelite to decouple training framework from serving runtime.

## 15. Feature Store

- **Offline** (Delta Lake): full historical snapshots, partitioned by date, used for training/backtesting. Point-in-time correctness via time-travel — features for a row labeled at day T joined "as of" T-1 only, preventing leakage.
- **Online** (Redis/DynamoDB): current snapshot only, refreshed by batch ETL (nightly) and streaming consumer (event-triggered partial updates for features like `loss_streak_current`).
- **Feature parity**: same transformation code shared between the Spark batch job and Flink streaming job to avoid train/serve skew.
- **Point-in-time join**: training pipeline uses a Feast-like client that resolves historical feature values per an "event timestamp," standardizing correctness across churn/LTV/matchmaking systems on the shared store.

## 16. Vector Database

**N/A for the core classifier.** It's a tabular GBT over engineered aggregates — no similarity search or embedding retrieval on the critical path.

Noted extension: a v2 embedding-based cohort-similarity model (e.g., "find players similar to recent churners") would introduce pgvector/FAISS/Milvus with HNSW — deferred, no clear baseline lift for prediction (useful for churn *analysis*, a separate consumer).

## 17. Embedding Pipelines

**N/A for the baseline.** Features are hand-crafted aggregates (counts, recency, ratios), keeping the model interpretable (FR7) and avoiding embedding-pipeline overhead (versioning, embedding-space drift).

Noted extension: a v2 sequence model (Transformer over session events) could produce a learned player-state embedding as an added GBT feature — deferred pending baseline ROI validation.

## 18. Inference Pipelines

**Near-real-time path:**
```
1. Player finishes match → client emits `match.result`
2. Kafka → cg-streaming-features
3. Flink updates windowed state (loss_streak_current += 1)
4. Trigger fires (loss_streak >= 5) → emit `score.request`
5. cg-nrt-scorer consumes request
6. Fetch feature vector from online store (Redis) — ~5ms
7. Call model server (gRPC Predict) — ~5-15ms compute
8. Publish `score.updated` (score=0.81, source=near_real_time)
9. cg-cache-invalidator evicts stale entry — ~1ms
10. cg-history-writer appends to history log (async, off critical path)
11. cg-decision-engine: score > threshold (0.75)?
      → holdout check (hash(player_id, campaign_id) % 100 < 10)
      → frequency cap check (max 2 offers/week)
      → if eligible: POST intervention.trigger → Push Service
12. Push Service delivers on next app-open / immediate push
```
Budget: steps 2-11 within 5 min p99 (dominated by Kafka consumer lag, not compute).

**Batch path:**
```
01:00 UTC: Airflow triggers batch DAG
  → Spark reads offline feature store (last 24h delta + rolling windows)
  → joins with static player/title metadata
  → broadcasts model (Treelite binary) to executors
  → vectorized predict() across 120M rows
  → writes scores to Cassandra (bulk upsert) + history log (Parquet append)
  → emits `score.updated` for cache warm + decision engine batch pass
05:00 UTC: SLA checkpoint — alert if incomplete
```

## 19. Training Pipelines

- **Data prep**: label = zero sessions in [T+1, T+30] given features as of T (time-based, no shuffling). Trailing 6 months of player-days; churners are minority class (~15-20% base rate) — handled via stratified sampling + class weighting, not naive oversampling.
- **Split**: strictly time-based — train months 1-4, validate month 5, test month 6, mirroring production's forward-only prediction.
- **Orchestration**: Airflow DAG per title: `extract_features → build_labels → train_test_split → train_lightgbm → evaluate → register_model → canary_deploy_gate`.
- **Distributed training**: LightGBM's native distributed mode (Dask/Spark) for 100M+ row titles; CPU-only (32-64 cores), 1-3 hrs/title.
- **HPO**: Bayesian optimization (Optuna) over depth, learning rate, num_leaves, regularization; capped trial budget (e.g., 200) to bound cost.
- **Evaluation gate**: candidate must beat production on AUC-PR (preferred over ROC given class imbalance) and calibration (Brier score) on the time-based holdout before being offered for promotion.

## 20. Retraining Strategy

| Trigger | Condition | Action |
|---|---|---|
| Scheduled | Weekly per title | Full retrain, gated eval |
| Data drift | PSI on top-20 features > 0.2 vs. training baseline | Ad hoc retrain within 24h |
| Concept drift | Rolling 7-day AUC-PR drops > 5% relative | Ad hoc retrain + alert on-call |
| Business event | Content patch / new season / new monetization mechanic | Manual-trigger retrain |
| Label maturity | 30-day label lag | A run on day D can only use labels through D-30, built into DAG windowing |

Retrain never auto-promotes — always gated by offline eval + canary before serving production traffic.

## 21. Drift Detection

| Drift Type | Metric | Threshold | Action |
|---|---|---|---|
| Feature/data | PSI per feature, daily vs. training distribution | >0.1 investigate, >0.2 retrain | Flag in dashboard |
| Prediction | KL/PSI on score distribution | PSI > 0.15 | Alert on-call, rule out upstream pipeline break first |
| Concept | Rolling AUC-PR/Brier on matured (30-day) labels | AUC-PR drop > 5% over 7-day window | Ad hoc retrain, notify owner |
| Schema | Null-rate, new/missing columns | Any unexpected change | Fail-closed — don't serve garbage features |
| Label | Base churn rate shift | > 20% relative WoW | Investigate — may be legitimate (content drought) |

## 22. Monitoring

**Infra**: Kafka consumer lag (`score.request` p99 < 30s), model server latency/error rate/pod utilization, Spark job duration/executor failures, feature store read/write latency.

**Model quality**: offline AUC-PR/log-loss/calibration/fairness slices (new vs. veteran, spend tiers) per training run; online rolling AUC-PR on matured labels, prediction/feature drift; score staleness vs. SLA.

**Business**: D7/D30 retention lift (treated vs. holdout), intervention volume + frequency-cap saturation, incrementality (conversion rate by score band vs. control, cost-per-incremental-retained-player), false-positive cost proxy via holdout comparison.

## 23. Alerting

| Alert | Condition | Severity | Route |
|---|---|---|---|
| Batch SLA miss | Not complete by 05:00 UTC | P1 | ML Platform on-call |
| NRT SLA breach | p99 latency > 5 min, sustained 15 min | P2 | ML Platform on-call |
| Consumer lag | `score.request` lag > 2 min | P2 | ML Platform on-call |
| Model server errors | > 1% over 5 min | P1 | ML Platform on-call, auto-page |
| Concept drift | AUC-PR drop > 5% | P3 | Model owner (Slack/ticket) |
| Schema break | Unexpected change | P1 | Data eng + ML Platform |
| Intervention anomaly | Daily count > 3x 7-day avg | P2 | Live-ops on-call (possible bug) |
| Retention regression | Negative D30 lift, 2 consecutive weeks | P3 | Product/ML leads, weekly review |

P1 = page immediately, 15-min SLA. P2 = page during business hours / if sustained off-hours. P3 = ticket, no paging.

## 24. Logging

- Structured JSON logs for every scoring/intervention decision: `trace_id`, pseudonymous `player_id`, `title_id`, `model_version`, `decision`, `timestamp`.
- PII: `player_id` is EA's pseudonymous account ID, not raw PII; anything touching raw PII routes through a separate access-controlled service. No free-text/chat content in this pipeline.
- Retention: operational logs 30 days hot / 1 year cold; score history 2 years (model eval + audit); intervention decisions 2 years (GDPR Article 22 relevance).
- Access: player-level logs restricted to ML platform + compliance roles; aggregated views broader.

## 25. Security

**Threat model:**
- Model inversion/membership inference via score API — mitigated by rate limiting, abstracted (not raw-value) top-features in responses, internal-only auth.
- Telemetry manipulation to farm offers (fake loss streaks) — mitigated via anomaly detection on trigger-event rates, threshold logic not exposed externally.
- Bulk export endpoints are high-value exfiltration targets — stricter authz scopes, audit logging, rate limits vs. single-player lookups.
- Insider risk on feature-store access — least-privilege IAM, synthetic/sampled data for local dev.

**Encryption**: TLS 1.2+ in transit; KMS-managed at-rest encryption for offline store, history log, model artifacts. Extra access scoping on purchase/payment-adjacent fields.

## 26. Authentication

- Service-to-service: mTLS + short-lived JWT (workload identity, e.g., SPIFFE/SPIRE-style), scoped per-service (`read:scores`, `write:scores:internal`).
- End-user: N/A — no player calls this system directly; consuming services authenticate players separately. Internal ops dashboards use corporate SSO (OIDC) with RBAC.

## 27. Rate Limiting

- Token bucket per calling service (callers are internal services, not end users) — tolerates bursts, caps sustained rate.
- Score Read API: 500 req/s/service, burst 1,000; bulk endpoint 10,000 ids/call, 10 calls/min/service.
- Explainability API: 50 req/s/service (heavier SHAP compute).
- Internal write path: controlled by backpressure/batch-size, not token bucket.
- Per-player intervention frequency cap (business rule, separate from API limits): sliding-window counter in KV store, e.g. max 2/player/7-day window.

## 28. Autoscaling

- NRT model server: HPA on p99 latency + queue depth, secondary CPU trigger (60% target). Min 8 / max 40 replicas.
- Streaming consumer (Flink): KEDA on Kafka consumer lag — scale out above 10k lag, scale in after 10 min stable.
- Decision engine workers: KEDA on `score.updated` lag (stateless, embarrassingly parallel).
- Batch Spark cluster: sized at job submission (Airflow params, dynamic executor allocation, 32-128 executors), not HPA-driven.
- VPA on model server pods for memory right-sizing — recommend-only, reviewed monthly to avoid surprise restarts.

## 29. Cost Optimization

- Spot instances for batch Spark clusters (fault-tolerant, checkpointable) — ~60-70% cost reduction.
- CPU-only inference vs. GPU serving is the single biggest lever over a DL-first approach.
- Read-through score cache cuts read QPS on Cassandra, reducing provisioned capacity.
- Limiting NRT triggers to high-value moments (vs. scoring every event) cuts streaming/model-server load ~80%+.
- Tree-count/depth capped via HPO budget to bound training and inference cost.
- Storage tiering: offline data >90 days to cold storage; score history >1 year compressed further.
- VPA recommendations to avoid over-provisioned serving pod memory.

## 30. Operational Concerns

At SDE2 scope, treat this as a checklist: **backups** (automated snapshots of registry/feature store with tested restore), **rollback** (one-command revert to last-known-good), **canary/blue-green rollout** (shift small traffic %, watch error rate + key metrics, then ramp), **observability** (dashboards/alerts on latency, error rate, top model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level concerns — know they exist, don't rehearse the manifests.

## 31. Why This Architecture

- Separate batch/near-real-time paths match actual business needs (bulk CRM vs. moment-based intervention) instead of one expensive low-latency system or one batch-only system that misses the in-session window.
- GBT over DL keeps interpretability (compliance/support, FR7) and cost (CPU inference) in check, while leaving a clear extension path (embeddings) without redesigning the core system.
- Shared feature store across player-modeling systems amortizes point-in-time-correctness engineering across teams.
- At-least-once + idempotent processing is simpler to operate than exactly-once transactions, and this domain tolerates rare duplicate scoring.
- Canary + shadow scoring guards against the highest-blast-radius failure: a miscalibrated model flooding players with bad offers before it's caught.

## 32. Alternative Architectures

| Alternative | Why Rejected (or When Preferred) |
|---|---|
| Pure streaming (score every player on every event) | Rejected by default: compute cost for freshness the business doesn't need (24h staleness is fine for CRM); preferred for truly continuous per-second scoring needs (e.g., real-time matchmaking risk) |
| Pure batch (no near-real-time path) | Rejected: misses FR2's in-session intervention; preferred for titles/markets where same-day scoring suffices and NRT infra cost isn't justified |
| DL sequence model (Transformer) as primary from day one | Rejected as v1: higher infra cost, harder to explain to compliance, longer iteration; preferred once baseline GBT plateaus and sequence signal is validated to lift AUC-PR — a v2 investment |
| Single-region deployment | Rejected: breaches NRT latency budget for non-US players; acceptable only for a geographically concentrated player base or soft-launch phase |

## 33. Tradeoffs

| Decision | Tradeoff |
|---|---|
| GBT over DL baseline | + Interpretable, cheap, fast to iterate / − may leave AUC-PR upside on the table |
| At-least-once + idempotent consumers | + Simpler, cheaper / − requires strict idempotency discipline; failure mode is duplicate, not lost, processing |
| Eventual consistency on scores | + Caching, replication, cost savings / − up to 24h-stale data occasionally acted on |
| Separate batch + NRT paths | + Right-sized cost/latency per use case / − two code paths, feature-drift risk (mitigated by shared transform library) |
| Canary + shadow scoring | + Catches bad models before blast radius / − slower time-to-production (1-week ramp) |
| Shared feature store | + Amortized correctness engineering / − coupling risk across teams, needs contract/versioning discipline |

## 34. Failure Modes

| Scenario | Impact | Mitigation |
|---|---|---|
| Telemetry bus outage/lag | NRT scores stop updating | Degrade to last-known batch score; alert on consumer lag; NRT is non-critical-path |
| Feature schema break upstream | Feature computation fails or produces garbage | Fail-closed schema validation; contract testing with telemetry team |
| Bad model deploy | Wrong scores drive real interventions | Canary + shadow scoring, automated rollback |
| Online feature store outage | NRT can't fetch features | Fall back to offline snapshot with `degraded=true` flag; online store is a rebuildable cache |
| Label leakage in training | Offline metrics look great, prod underperforms silently | Strict time-based point-in-time joins; canary/shadow catches real-world underperformance |
| Runaway intervention trigger bug | Mass over-messaging, brand damage | Intervention volume anomaly alert, frequency caps independent of model logic, automated rollback |

## 35. Scaling Bottlenecks

- **10x (250M MAU)**: Batch ETL shuffle-heavy joins tighten the 4-hour SLA — mitigate by partitioning batch jobs per title and incrementally upserting rather than full nightly recompute.
- **10x**: Online feature store memory (150GB → 1.5TB) needs resharding — mitigate by tiering cold/inactive players out of the hottest tier.
- **100x**: Score store write throughput during nightly bulk-write becomes the constraint — mitigate with staged bulk-load (write to S3 then bulk-import) instead of row-by-row upserts.
- **100x**: Training cluster time grows with data volume; Bayesian HPO search becomes the long pole — cap trial budget or warm-start HPO from prior week's best hyperparameters.
- **Streaming**: Kafka partition count/consumer parallelism bottlenecks before compute does — over-provision partitions early since repartitioning later is disruptive.

## 36. Latency Bottlenecks

**Near-real-time path (target 5 min p99):**

| Stage | p50 | p99 | Notes |
|---|---|---|---|
| Kafka produce→consume | 200ms | 3-5s | Tail risk under rebalance/lag |
| Windowed aggregate update | 50ms | 500ms | Flink state I/O |
| Trigger eval → emit | 10ms | 100ms | Cheap, rule-based |
| `score.request` queue wait | 1s | 60-180s | Largest, most variable contributor |
| Feature fetch | 5ms | 15ms | Redis |
| Model inference | 3ms | 15ms | Negligible |
| Publish + cache invalidation | 10ms | 100ms | Async, off critical path |
| Decision engine eval | 10ms | 50ms | KV lookups |
| **Total** | ~1.3s | ~60-200s (budget allows 300s) | Queue lag, not compute, dominates |

**Score Read API (target p99 < 100ms):** cache hit ~2/8ms (p50/p99), Cassandra miss ~10/40ms, network overhead ~5/15ms → total ~7ms (hit) to ~65ms (miss).

Key interview point: compute is never the bottleneck here — queueing and I/O dominate, the opposite profile from an LLM-serving system.

## 37. Cost Bottlenecks

1. Batch feature ETL compute (Spark/EMR hours) — largest line item, driven by shuffle volume across 120M rows/300 features; mitigated by spot instances and incremental recompute.
2. Data storage (offline store + score history) — ~10.5TB + ~1.8TB/year, compounds without lifecycle policies; mitigated by cold-tiering after 90 days.
3. Online feature store/cache (Redis) — always-on, provisioned for peak, expensive per GB; hence only latest snapshot kept online.
4. NRT serving fleet — smaller since CPU-only and trigger-filtered; would become #1 if the team ever scored every event without justification.
5. Cross-region replication egress — non-trivial at 25M+ scale but small relative to compute/storage.

Cost guardrail (< 3% of retention-lift value, Assumption 11) should be reviewed quarterly against campaign ROI.

## 38. Interview Follow-Up Questions

1. How would you validate that the churn label (14-day inactivity, 30-day horizon) correlates with real business-meaningful churn, not just short-term absence (e.g., vacation)?
2. The NRT path only triggers on curated "high-value moments." How would you detect missing triggers, and test adding one without blowing up compute cost?
3. How would you detect and prevent train/serve skew between the batch Spark and streaming Flink feature pipelines?
4. How do you measure true incrementality of an intervention, not just correlation between high score and redemption?
5. AUC-PR looks great offline but the A/B test shows no retention lift — what's your debugging process?
6. How would you share signal across titles (Apex-active but FC-churning) without tight coupling between title pipelines?
7. What changes if the NRT SLA tightens from 5 minutes to 5 seconds?
8. How do you prevent the model from targeting price-sensitive/discount-seeking players rather than genuinely at-risk ones?
9. How do you size the incrementality holdout, and justify its cost to stakeholders?
10. How would you detect a feedback loop where past interventions bias future churn predictions?

## 39. Ideal Answers

1. **Label validation**: Retrospective cohort study comparing 14-day inactivity against longer windows (60/90-day) to measure genuine non-return vs. false churn (seasonal). Use survival analysis to pick a horizon with acceptable false-churn rate.

2. **Trigger completeness**: Periodically score a random sample of all events (not just curated triggers) to see how often high-score moments fall outside the trigger set. Add triggers only if serving cost is justified by incremental retention value, rolled out via canary.

3. **Train/serve skew prevention**: Single shared feature-transformation library for both Spark and Flink jobs, plus automated parity tests comparing online vs. offline feature values with alerting on divergence.

4. **Measuring incrementality**: Never trust score+redemption as causal proof — always compare against a randomized holdout. Incremental lift = retention(treated) − retention(holdout) within the same score band.

5. **Debugging offline/online mismatch**: Check in order — (a) train/serve feature skew (most common), (b) offline eval population vs. live targeted segment, (c) whether AUC-PR gains manifest in the targeting score range, (d) A/B test statistical power.

6. **Cross-title signal sharing**: A lightweight, centrally computed cross-title feature layer (e.g., "days since last session on any EA title") offered as optional input to each title's otherwise-independent model, preserving per-title ownership.

7. **5-second SLA**: Requires moving from queue-mediated to synchronous in-request feature computation, since consumer lag dominates tail latency today. Only justified if the business case for sub-5-second intervention is strong — it materially raises cost and coupling.

8. **Adverse selection guard**: Monitor performance segmented by prior discount-seeking behavior (high redemption, low true incremental retention). Exclude pure discount-usage features from the model; have the Decision Engine apply incrementality-aware targeting.

9. **Holdout sizing**: Standard A/B power analysis (baseline retention, minimum detectable effect, power/significance); typically 5-10% of the eligible high-risk population at this scale. Frame the cost as the price of measurement.

10. **Feedback loop detection**: Watch for "recently intervened" leaking into features, making retained treated players look artificially low-risk. Mitigate by excluding recent-intervention-exposure from features, or periodically validating against a clean, never-intervened holdout population.
