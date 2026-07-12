# Recommendation Engine

## 1. Problem Framing

Design a recommendation engine for EA's game store / in-game content surfaces (EA App storefront, item shop, "recommended for you" rails, next-game carousels across Battlefield, Sims, FIFA/EA FC).

- **Goal**: increase conversion, session length, and surface long-tail content (DLC, cosmetics) that organic browsing misses.
- **Core problem**: given a player, context (surface, time, device), and catalog (games, DLC, bundles, cosmetics), return a ranked list of items.
- **Two-stage system**: candidate generation (retrieval) narrows millions of items to hundreds; ranking scores/orders those to a top-K (10-50 shown).
- **Cross-cutting concerns**: cold-start (new players/items), real-time personalization (react to last 5 min of behavior), freshness (new content surfaces within minutes), diversity (avoid recommending the same AAA titles to everyone).

## 2. Functional Requirements

- FR1: Personalized ranked item list for `(user_id, surface, context)` in real time.
- FR2: Multiple surfaces — store homepage, item shop, post-game carousel, email/push batch.
- FR3: Multi-source candidate generation — two-tower ANN, popularity/trending, collaborative filtering, editorial overrides.
- FR4: Unified ranking model with pointwise CTR/CVR prediction.
- FR5: Cold-start handling for new users/items via content-based fallback.
- FR6: Real-time signals (last N minutes of clicks/purchases) folded into personalization within-session.
- FR7: Exclusion rules — owned items, region-locked content, age ratings, parental controls.
- FR8: Every response tagged with an experiment/variant ID for A/B analysis.
- FR9: Explainability metadata for merchandising (why an item was recommended).
- FR10: Batch generation for offline channels (push, email) at scale.

## 3. Non-Functional Requirements

| Dimension | Target |
|---|---|
| Latency (p50 / p99) | 40 ms / 150 ms end-to-end |
| Availability | 99.95% (revenue-impacting) |
| Throughput | 60K QPS peak, global |
| Consistency | Eventual for features (secs-min staleness); strong for "already owned" exclusion |
| Cost ceiling | < $0.0004 per recommendation request |
| Freshness | New catalog items rankable within 15 min of publish |
| Data durability | Feature store & training data: 11 nines (object storage) |

## 4. Clarifying Questions

1. Which surfaces are in scope — store front page only, or also in-game overlays and push/email?
2. Catalog size — thousands (full games) or millions (cosmetics, bundles, UGC)?
3. Single-title or cross-title (recommend across EA's whole portfolio)?
4. Business objective — conversion, revenue, engagement/retention, or blended?
5. Real-time in-session adaptation required, or is daily-batch sufficient?
6. Compliance constraints — loot box regulations, age filtering, regional restrictions?
7. Existing feature store/data platform to integrate with, or greenfield?
8. Acceptable staleness for "already purchased" exclusion?
9. Cold-start support needed for brand-new game launches with zero history?
10. Known traffic peaks (launch day, holiday sale) to provision for?

## 5. Assumptions

1. 40M MAU across EA App + top 5 live-service titles' in-game shops.
2. Catalog: 500K sellable items + 50K recommendable content objects.
3. Avg user generates 25 interaction events/day.
4. Peak QPS is 5x average (regional overlap + promo events).
5. Two-tower retrieval: user tower + item tower, 128-dim embeddings.
6. Ranking: GBDT baseline, migrating to deep ranking (DLRM-style) for mature surfaces.
7. Real-time personalization window: last 30 min of session behavior.
8. ~2,000 new SKUs/week; ~150K new accounts/day.
9. Multi-region (US-East, US-West, EU-West, AP-Southeast) for global latency SLAs.
10. Existing EA data platform (Kafka + S3/Parquet lake) is reused, not rebuilt.

## 6. Capacity Estimation

**Traffic**: 40M MAU x 6 views/day → 240M requests/day → avg ≈ 2,780 QPS. Peak (5x) ≈ **14,000 QPS**, provision to 20,000 QPS burst. Batch channel: 40M users/day over a 2hr window ≈ 5,500 QPS-equivalent (separate, non-p99-sensitive pipeline).

**Retrieval**: catalog embeddings 550K x 128-dim x 4B ≈ 282 MB, fits in memory per shard. User embedding computed online (~200K param MLP, <2ms CPU). HNSW index ≈ 550-650 MB/replica; replicate 6x per region.

**Ranking**: ~300 blended candidates/request (200 ANN + 50 popularity + 50 CF). GBDT (~500 trees) ≈ 8-12ms/request CPU. DLRM (~15M params, fp16 ≈ 30MB) fits one GPU; batched inference (32-64/batch) ≈ 6-10ms/batch p99.

**Fleet (per region, 20K QPS burst)**: ranking GPU ≈ 14 nodes/region x 4 + 30% headroom ≈ **~75 GPU nodes** globally. Retrieval CPU/ANN ≈ 7 nodes/region x 4 ≈ **28 CPU nodes**. Feature store: Redis cluster easily absorbs ~60K ops/sec.

**Storage**: training data ≈ 120GB/day raw → ~11TB/year compressed. Offline feature store ≈ 2TB (90-day window). Online feature store (Redis/DynamoDB) ≈ 85GB total. Model artifacts negligible (tens of MB, versioned).

## 7. High-Level Architecture

```
Client (Store UI / in-game shop)
        │ HTTPS/gRPC
        ▼
API Gateway (authn, rate limit, LB)
        ▼
Recommendation Orchestrator ──┬─────────────┬───────────────┐
        │                     ▼             ▼               ▼
        │          Candidate Gen Svc   Feature Store   Business Rules /
        │          (two-tower ANN,     (Redis online,  Exclusion Filter
        │           popularity, CF)     Parquet offline) (owned/geo/age)
        │                     │             │               │
        └────────────► Ranking Service (GBDT / DLRM) ◄──────┘
                        scores + diversity (MMR)
                              │
                     Response Assembly (top-K,
                     metadata, experiment tag)
                              │
                        back to Client

── Offline / async plane ──
Kafka (player events) → Flink stream processing → Real-time Feature Aggregator → Online Feature Store
Kafka → Data Lake (S3/Parquet) → Training Pipelines (two-tower, ranker) → Model Registry → canary rollout
                                                                        → Vector Index Builder → ANN Index (per region)
```

## 8. Low-Level Components

| Component | Responsibility | Scaling Unit |
|---|---|---|
| API Gateway | AuthN, rate limiting, TLS, routing | Stateless, horizontal via L7 LB |
| Recommendation Orchestrator | Fan-out to candidate gen/feature store/rules, calls ranker, assembles response | Stateless, HPA on QPS |
| Candidate Generation Service | ANN lookup, popularity merge, CF lookup → ~300 deduped candidates | Scales with ANN replicas (CPU-bound) |
| Feature Store (Online) | Low-latency KV lookup of user/item/context features | Sharded via consistent hashing |
| Feature Store (Offline) | Point-in-time correct historical features for training | Storage partitioning |
| Business Rules Engine | Owned-item filter, region/age restrictions, merchandising overrides | Stateless, scales with orchestrator |
| Ranking Service | Scores candidates (GBDT/DLRM), diversity re-ranking (MMR) | GPU-backed for DLRM, autoscale on batch queue depth |
| Vector Index / ANN Service | Nearest-neighbor lookups against item embeddings | Replicated per-region |
| Stream Processor (Flink) | Real-time feature aggregation (last 30 min) | Scales via parallelism/task slots |
| Training Orchestrator | Schedules distributed training jobs | Job-level GPU cluster provisioning |
| Model Registry | Versions models, tracks lineage, gates promotion | Low-throughput, single cluster |
| Experimentation Service | Assigns variant, logs exposure | Stateless |

## 9. API Design

```
POST /v2/recommendations
{
  "user_id": "u_9f3a2c",
  "surface": "STORE_HOMEPAGE",   // STORE_HOMEPAGE | IN_GAME_SHOP | POST_MATCH_CAROUSEL | EMAIL_DIGEST
  "context": { "platform": "PS5", "region": "US-East", "session_id": "s_88213", "current_game_id": "fifa25" },
  "num_results": 20
}

Response: 200 OK
{
  "request_id": "req_7788",
  "results": [
    { "item_id": "dlc_ea_fc25_ultimate_pack", "rank": 1, "score": 0.9123,
      "reason_code": "SIMILAR_TO_RECENT_PURCHASE", "model_version": "ranker-dlrm-v14" }
  ],
  "experiment_id": "exp_reco_diversity_v3", "variant": "treatment_b"
}

Errors: 400, 401, 429, 503 DEGRADED_FALLBACK (popularity-only served)
```

```
POST /v2/recommendations/batch   // async offline channel (push/email)
GET  /v2/recommendations/batch/{job_id}/status
```

**Versioning**: URI-based (`/v1`, `/v2`); old versions supported 6 months post-deprecation; response includes `model_version` for traceability; additive fields are backward compatible, breaking changes bump major version.

## 10. Database Design

| Store | Type | Purpose | Partition Key | Why |
|---|---|---|---|---|
| Online Feature Store | Redis Cluster | Real-time user/item/context features | `user_id` hash slot | Sub-ms latency, native TTL |
| Item Metadata Store | DynamoDB | Catalog metadata (price, region, age rating) | `item_id` | Serverless scale, strong consistency option |
| Offline Feature/Event Lake | S3 + Parquet/Iceberg | Historical events for training | `event_date`, bucketed by user_id | Columnar, cheap at TB scale |
| Ownership/Entitlement DB | PostgreSQL (sharded) | Source of truth: what user owns | `user_id` range | Strong consistency for purchase flow |
| Vector Index Storage | FAISS/HNSW on NVMe + S3 backup | Item embeddings for ANN | Region-specific catalogs | Local disk speed + durable backup |
| Experiment Exposure Log | Kafka → ClickHouse | Log every exposure + variant | `experiment_id`, `date` | Fast aggregate queries for A/B analysis |

```sql
CREATE TABLE entitlements (
  user_id BIGINT NOT NULL, item_id VARCHAR(64) NOT NULL,
  acquired_at TIMESTAMPTZ NOT NULL, platform VARCHAR(16),
  PRIMARY KEY (user_id, item_id)
) PARTITION BY HASH (user_id);
```

## 11. Caching

| Cached Data | TTL | Strategy | Invalidation |
|---|---|---|---|
| User feature vector | 5 min | Cache-aside | TTL + event-driven on purchase/session-end |
| Item metadata | 60s | Cache-aside | Pub/sub on catalog update |
| ANN candidate results | 30s | Cache-aside | TTL only (personalization drifts fast) |
| Ownership/entitlement flags | N/A | **Write-through** from Postgres CDC | CDC stream (Debezium) — must never serve stale "not owned" as "owned" |
| Full response (logged-out fallback) | 2 min | Cache-aside | Not used for logged-in personalized traffic |

Ownership uses write-through deliberately: a false negative (re-showing an owned item) is a trust issue with near-zero staleness tolerance. Everything else uses cache-aside with short TTLs — seconds of personalization staleness is acceptable, and this avoids write amplification on high-churn feature data.

## 12. Queues & Async Processing

| Queue/Topic | Flow | Semantics | DLQ |
|---|---|---|---|
| `player-events` (Kafka) | client → Flink | At-least-once | Malformed → DLQ, alert if rate > 0.1% |
| `catalog-updates` (Kafka) | merchandising → feature store/cache | At-least-once, idempotent upsert | DLQ + manual replay |
| `batch-reco-jobs` (SQS) | scheduler → batch workers | At-least-once, idempotency key = job_id | Retry 3x backoff → DLQ + page if backlog > 30min |
| `entitlement-cdc` (Debezium) | Postgres CDC → Redis updater | At-least-once, offset-checkpointed | Critical path — immediate alert |
| `experiment-exposure-log` | reco service → analytics | At-least-once, dedup downstream | Low priority, 7-day retention |

No queue needs strict exactly-once — all consumers are idempotent (upsert-by-key or offset-checkpointed), cheaper than transactional exactly-once.

## 13. Streaming & Event-Driven Architecture

**Topic `player-events`**: `event_type` (VIEW/CLICK/PURCHASE/SESSION_START/END), `user_id`, `item_id`, `surface`, `timestamp`, `platform`, `session_id`, `value_usd`.

- **Consumer groups**: `realtime-feature-aggregator` (Flink → Redis), `training-data-writer` (Spark → Iceberg), `experiment-analytics` (ClickHouse).
- **Partitioning**: by `user_id` to preserve per-user ordering for session-window aggregation.
- **Retention**: 7 days hot (Kafka), then compacted into the lake.

**Topic `catalog-updates`**: `item_id`, `version`, `price`, `region_availability[]`, `age_rating`, `embedding_refresh_required`. Consumed by ANN incremental updater and feature store.

## 14. Model Serving

- **Two-tower retrieval**: item embeddings precomputed offline, served via ANN index (no per-request GPU). User tower runs online (small MLP), CPU, <2ms.
- **Ranking**: GBDT baseline via Triton (FIL backend) on CPU. DLRM deep ranker via Triton on GPU (A10G) with dynamic batching (max batch 64, max queue delay 5ms).
- **Multi-model serving**: Triton hosts multiple versions concurrently (canary + stable), routed by orchestrator via model-version header.
- **Hardware**: CPU for GBDT+ANN; GPU (A10G) for DLRM; A100 for batch training.
- **Fallback**: if ranking degraded (timeout > 50ms), orchestrator serves popularity-ranked candidates — graceful degradation, not failure.

## 15. Feature Store

- **Offline**: Iceberg/Parquet on S3, event-time versioned. Training uses **point-in-time joins** so only features known as-of the label timestamp are used, preventing label leakage.
- **Online**: Redis, materialized from the same feature definitions (Feast-style) to guarantee train/serve consistency (same transform code path for batch and streaming).
- **Real-time features** (last 30 min): Flink writes directly to Redis for freshness, backfilled to offline store nightly.
- **Versioning**: every feature has a schema version; model records which version it trained against, enforced at serving to prevent skew.

## 16. Vector Database

- **Algorithm**: HNSW, chosen over IVF-PQ — catalog (550K items) is small enough that HNSW's recall/latency tradeoff wins; ~600MB/replica is affordable.
- **Index build**: nightly full rebuild + incremental insert for new items (rebalanced weekly to control graph degradation).
- **Sharding**: one full index per region, replicated 6x — simpler ops, catalog fits in memory so no need to shard by item.
- **Query**: 128-dim user embedding → top-200 ANN, `ef_search=64` tuned for ~2ms p99.
- **Alternative**: ScaNN — comparable recall/latency; HNSW chosen for FAISS ecosystem familiarity.

## 17. Embedding Pipelines

- **User tower**: recent interaction history, demographic bucket, platform, region, spend tier → 128-dim via 3-layer MLP.
- **Item tower**: genre, price tier, franchise, content type, text description embedding (frozen sentence-transformer) → 128-dim.
- **Training**: in-batch negative sampling with softmax loss, logged impressions + random negatives, logQ correction for popularity bias.
- **Refresh**: item embeddings nightly + on-demand for new drops (`catalog-updates` event) to hit the 15-min freshness SLA.
- **Drift check**: cosine similarity of new vs. previous day's item embeddings; large shifts trigger manual review before index swap.

## 18. Inference Pipeline (request lifecycle)

```
t=0ms   Client sends GetRecommendations
t=1ms   Gateway: authn, rate-limit
t=2ms   Orchestrator fans out in parallel:
          Feature Store lookup            [~3ms]
          Candidate Gen (ANN top-200)      [~6ms]
          Popularity list (cached)         [~1ms]
          Entitlement check (Redis)        [~2ms]
t=8ms   Merge/de-dup (~300) → filter owned/region-locked
t=10ms  Ranking: batch-score 300 candidates    [~10ms p50, ~25ms p99]
t=20ms  Diversity re-ranking (MMR, cap same-franchise items in top 10)
t=22ms  Response assembly
t=24ms  Return to client
        (p50 ~25-30ms; p99 budget 150ms covers cache misses, GPU batching delay, cross-AZ hops)
```

## 19. Training Pipelines

- **Data prep**: Spark reads event lake, joins with entitlement + catalog snapshots (point-in-time correct), labels impressions with click (24h) / purchase (7d) attribution windows.
- **Negative sampling**: in-batch negatives + hard negatives from near-miss ANN results (ranked 200-400, not clicked).
- **Orchestration**: Airflow DAG — extract → build training set → train two-tower → eval → train ranker → eval → register → canary rollout.
- **Distributed training**: two-tower/DLRM via PyTorch DDP on 8x A100 (data-parallel). GBDT via distributed XGBoost (~20 CPU nodes).
- **Volume**: ~200M labeled rows/run (30-day window), ~11TB compressed; full two-tower retrain ≈ 3hrs on 8x A100.
- **Evaluation**: offline Recall@200 (retrieval), NDCG@10/AUC (ranking) gated against holdout; IPS-weighted counterfactual eval estimates online lift pre-rollout.

## 20. Retraining Strategy

| Model | Cadence | Trigger-based retrain |
|---|---|---|
| Two-tower retrieval | Daily embedding refresh, weekly full retrain | Embedding drift fail or Recall@200 drop >3% |
| Ranking (GBDT) | Daily incremental | Feature drift (PSI) exceeds threshold |
| Ranking (DLRM) | Weekly full, daily fine-tune | Calibration error > 10% |
| Popularity/trending | Hourly | Event-driven (launch, promo start) |
| Item embeddings | On-demand (within 15 min) | `catalog-updates` event |

Major live-service events (launch day, holiday sale) trigger out-of-band manual retrain + canary ahead of time, since daily cadence may miss pre-event behavior shifts.

## 21. Drift Detection

| Drift Type | Metric | Threshold | Action |
|---|---|---|---|
| Feature data drift | PSI per feature | >0.2 alert, >0.3 auto-block promotion | Investigate source, retrain |
| Embedding drift | Cosine shift of item centroids day-over-day | Mean shift >0.15 | Manual review before index swap |
| Concept drift | CTR/CVR calibration error | >10% | Retrain, widen exploration |
| Candidate concentration | % impressions from top-10 items | >40% | Alert, check diversity re-ranker |
| Cold-start ratio | % requests served pure fallback | >15% sustained | Investigate onboarding/pipeline gaps |

Runs as a daily batch job comparing production distributions against a rolling 7-day baseline.

## 22. Monitoring

| Layer | Metrics |
|---|---|
| Infra | p50/p95/p99 latency, error rate, GPU utilization, Redis hit rate, Kafka lag, queue depth |
| Model quality | Recall@K, NDCG@10, calibration error, score distribution, PSI |
| Business | CTR, CVR, revenue-per-recommendation, session-length uplift, catalog diversity/coverage, cold-start conversion |
| Experiment | Variant lift, sample ratio mismatch, exposure log completeness |
| Data pipeline | Freshness lag, training job success/duration, volume anomalies |

Dashboards: Grafana (infra/model via Prometheus), business dashboard (Looker/Tableau from ClickHouse exposure log).

## 23. Alerting

| Alert | Condition | Severity |
|---|---|---|
| API p99 breach | >150ms for 5 min | High — page SRE |
| Error rate spike | 5xx >1% over 5 min | High — page SRE |
| Ranking degraded-fallback | >10% requests on popularity fallback, 10 min | Medium — Slack ML platform |
| Entitlement CDC lag | >30s | Critical — immediate page |
| Feature cache hit rate | <90% for 15 min | Medium — Slack |
| Model drift breach | Per Sec 21 thresholds | Medium — Slack, auto-block promotion |
| Training pipeline failure | DAG failure | Medium — email/Slack data-eng |
| DLQ backlog | >30 min unprocessed | High — page |

On-call: ML platform (model/serving) separate from data-eng (pipeline/ETL); SRE owns latency/availability, escalates to ML if model-side.

## 24. Logging

- **Structured JSON** per request: `request_id`, hashed `user_id`, `surface`, `latency_ms`, `model_versions_used`, `candidate_count`, `experiment_id/variant`, `error_code`.
- **PII**: raw `user_id` only in short-lived hot path; long-term logs use a salted hash; no free-text PII enters this path (payment info lives in separate PCI-scoped systems).
- **Retention**: 30 days hot (searchable), 1 year cold (S3), exposure logs 2 years anonymized.
- **Access control**: IAM-restricted to ML/data-eng roles; audit trail for compliance; GDPR/CCPA deletion requests propagate into retention pipeline.

## 25. Security

- **Encryption**: TLS 1.3 in transit (mTLS in-mesh); at-rest encryption for all feature/event stores (S3 SSE-KMS, Redis, Postgres TDE).
- **Threat model**:
  - **Manipulation** (bot clicks/purchases to boost ranking) — bot detection on ingestion, anomaly detection on interaction velocity.
  - **Data poisoning** (fake events skewing training) — event validation, rate limiting, outlier filtering.
  - **Model extraction** (querying API to reverse-engineer model/embeddings) — rate limiting, don't return raw scores (ranks only), scraping-pattern monitoring.
  - **PII leakage via embeddings** — embeddings never returned in API responses; embedding store access restricted internally.
  - **Ownership bypass** — entitlement check treated as security-critical, not just relevance-critical.

## 26. Authentication

- **End-user**: OAuth2/OIDC via EA's IdP; short-lived JWT (15 min); gateway validates signature/expiry/audience.
- **Service-to-service**: mTLS in-mesh, SPIFFE workload identity — no shared secrets.
- **Batch/offline**: IAM role-based access (e.g., IRSA), no long-lived static credentials.
- **Token scope**: read-only recommendation access; purchase flow uses separate, more tightly scoped auth.

## 27. Rate Limiting

- **Algorithm**: token bucket at the gateway (per-user) — allows bursts while capping sustained rate.
- **Limits**: 20 req/s/user burst, 5 req/s sustained; batch endpoints capped per-service-account (e.g., 2,000 req/s aggregate).
- **Per-surface tenant limits**: each surface has its own quota pool (Redis-backed token buckets) so one surface's spike can't starve others.
- **Abuse handling**: sustained violations escalate to a longer-lived block and feed the bot-detection signal.

## 28. Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: ranking-service-hpa }
spec:
  minReplicas: 8
  maxReplicas: 60
  metrics:
    - type: Pods
      pods: { metric: { name: triton_inference_queue_depth }, target: { type: AverageValue, averageValue: "10" } }
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 65 } }
  behavior:
    scaleUp: { stabilizationWindowSeconds: 30, policies: [{ type: Percent, value: 100, periodSeconds: 60 }] }
    scaleDown: { stabilizationWindowSeconds: 300, policies: [{ type: Percent, value: 20, periodSeconds: 120 }] }
```

- HPA on orchestrator/candidate-gen keyed on QPS + p99 latency.
- KEDA scales batch inference workers on queue depth (scale-to-zero when idle).
- VPA (recommendation-only, not auto-apply) right-sizes GBDT pods — avoids restart churn during peak.
- GPU node pools use cluster autoscaler with a small warm buffer (2 idle nodes) during known peak windows.

## 29. Cost Optimization

- **Spot instances**: training (A100) on spot with checkpointing — ~60-70% savings; serving GPUs stay on-demand/reserved; batch inference workers use spot.
- **Model distillation**: DLRM distilled from a larger teacher — ~40% GPU inference cost cut, <1% NDCG regression.
- **Caching**: short-TTL caching cuts redundant feature/ANN lookups, ~25% ANN QPS reduction on high-repeat surfaces.
- **Batching**: dynamic batching on Triton improves GPU throughput ~3x vs per-request.
- **CPU vs GPU tiering**: GBDT (CPU) for lower-value surfaces (email digest); GPU DLRM reserved for high-value real-time surfaces.
- **Reserved capacity** for steady-state baseline load; autoscaling/spot handles burst.
- **Storage lifecycle**: raw logs to Glacier after 90 days; Iceberg compaction reduces query cost.

## 30. Operational Concerns

At SDE2 scope, treat this as a checklist: **backups** (automated snapshots of model registry/feature store with a tested restore path), **rollback** (one-command revert to last-known-good via CI/CD), **canary/blue-green rollout** (small traffic slice first, watch error rate + key metrics, then ramp), and **basic observability** (dashboards + alerts on latency, error rate, top model-quality signals, wired to on-call). Kubernetes/Terraform manifests and multi-region active-active topology are Staff/Principal-level concerns — worth knowing they exist, not worth rehearsing.

## 38. Why This Architecture

- **Two-stage retrieval + ranking** is the standard pattern for large-catalog recommendation (Netflix, YouTube, Meta) — scoring 550K items per request with a heavy ranker is infeasible at 14K QPS; ANN narrows the field cheaply first.
- **Two-tower retrieval** decouples user/item computation — item embeddings precomputed offline, only the cheap user tower runs online, enabling sub-10ms retrieval.
- **Online/offline feature store split** guarantees train/serve consistency — skew between training-time and serving-time features is one of the most common silent failure modes in production recommenders.
- **GBDT + DLRM dual-path ranking** ties serving cost to business value rather than using the expensive model everywhere.
- **Multi-region active-active** is justified by EA's global player base and tight 150ms p99 — cross-continent round trips alone would blow the budget single-region.
- **Correctness-critical paths (entitlement) separated** from personalization paths — different consistency requirements need different stores/cache strategies (write-through vs cache-aside).

## 39. Alternative Architectures

| Alternative | Why Rejected (or when preferred) |
|---|---|
| Single-stage ranking (score full catalog per request) | Infeasible at target QPS/latency; only viable for catalogs <~5K items or unconstrained offline use |
| Pure collaborative filtering (matrix factorization only) | Fails on cold-start (new users/items) which is a hard requirement here; used as one blended candidate source, not the sole method |
| Fully batch/offline precomputed recommendations | Preferred when real-time isn't needed (email digest uses this) — rejected as sole approach since it can't react to in-session behavior, hurting conversion on high-intent surfaces |
| Single global model (no per-title tuning) | Rejected for high-traffic titles (FIFA/EA FC, Battlefield) where title-specific patterns justify fine-tuned heads; retained as fallback for smaller titles |

## 40. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Two-stage retrieval+ranking | Scales to large catalogs, cheap at high QPS | Extra complexity, possible recall loss vs. full-catalog scoring |
| HNSW ANN index | Fast, high recall at this scale | Memory-resident; incremental inserts degrade graph (needs rebuild) |
| Write-through entitlement cache | Strong consistency, avoids owned-item bugs | Higher write latency/complexity vs cache-aside |
| Active-active multi-region | Best latency, resilient to regional outage | Replication lag, no multi-master writes for entitlements |
| GBDT + DLRM dual path | Cost-efficient | Two models to maintain and keep feature-consistent |
| Aggressive short-TTL caching | Reduces backend load significantly | Slight personalization staleness (acceptable, kept separate from correctness-critical data) |
| Canary + staged rollout | Catches regressions before full exposure | Slower time-to-full-rollout, added overhead |
| Distilled student ranking model | Cheaper GPU inference | ~1% NDCG regression vs full teacher |

## 41. Failure Modes

| Failure | Impact | Mitigation |
|---|---|---|
| ANN index down (region) | Candidate gen degraded to popularity-only | Circuit breaker falls back within 10ms timeout budget |
| Redis feature store outage | Missing features → ranker uses defaults | Model trained with missing-feature handling; alert fires |
| Entitlement CDC lag/outage | Risk of recommending owned item | Fail-closed: exclude uncertain items rather than risk showing owned content |
| GPU pool exhaustion | Ranking latency spikes, queueing | Autoscaler + fallback to CPU GBDT path |
| Bad model deploy | Poor recommendations, revenue impact | Canary gates catch before full rollout; auto rollback |
| Kafka broker outage | Real-time features stall, training gap | Multi-AZ replication; resume from last committed offset |
| Cross-region replication lag spike | Stale features in secondary region | Monitored lag metric; region removed from rotation if exceeded |
| Retry storm | Amplified load during degradation | Client backoff+jitter; server-side load shedding (503 + Retry-After) |

## 42. Scaling Bottlenecks (10x/100x)

- **At 10x (140K QPS)**: ANN replica count and Redis shard count bottleneck first — need to shard the ANN index (not just add full-copy replicas) and move to a sharded multi-cluster feature store. Ranking GPU fleet grows linearly (~140-150 GPU nodes/region), pushing distillation/INT8 quantization over raw node scaling.
- **At 100x (1.4M QPS, illustrative)**: HNSW full-index-per-replica stops fitting in memory as catalog also grows — shift to sharded ANN (partition by title/genre) with scatter-gather, adding latency. Single-write-primary entitlement DB becomes a write bottleneck during flash sales — revisit to distributed SQL (CockroachDB/Spanner-style). Kafka `user_id` partitioning may create hot partitions for highly active users/bots. Flink windowed-aggregation state size grows, needing more RocksDB state-store capacity.

## 43. Latency Bottlenecks

| Stage | p50 | p99 |
|---|---|---|
| Gateway | 1 ms | 3 ms |
| Feature store lookup | 3 ms | 15 ms (cache miss) |
| Candidate gen | 6 ms | 20 ms (cold shard) |
| Entitlement check | 2 ms | 8 ms |
| Merge/filter/de-dup | 2 ms | 4 ms |
| Ranking (batched) | 10 ms | 40 ms (batching delay under load) |
| Diversity re-ranking | 2 ms | 5 ms |
| Response assembly | 1 ms | 2 ms |
| Network overhead | 3 ms | 10 ms |
| **Total** | **~30 ms** | **~107 ms** (within 150ms budget) |

Biggest p99 contributor: ranking batching queue delay (5ms max-wait tuned to cap the downside of 3x GPU throughput gain). Second: feature-store cache misses, mitigated by pre-warming on session start.

## 44. Cost Bottlenecks

- **GPU ranking fleet** (~75 nodes globally at peak) is the single largest driver — why distillation, batching, and CPU/GPU tiering are first-priority levers.
- **Cross-region replication** (Kafka MirrorMaker2, S3 CRR) — mitigated by compressing/filtering events before replication.
- **Redis memory** — cheap individually, but a recurring line item at 40M users x regions x replication; short TTLs bound it.
- **Training compute** (A100 clusters) — periodic but significant; spot is the main lever.
- **Storage growth** — linear with player base; lifecycle policies (Glacier, compaction) are the control.

## 45. Interview Follow-Up Questions

1. How would you handle a brand-new game title launch with zero historical data — walk through cold-start end-to-end.
2. Ranking model's CTR predictions are well-calibrated offline but conversion drops 5% after rollout — how do you debug this?
3. How do you prevent a filter-bubble / popularity-collapse feedback loop over time?
4. Walk through what happens if the ANN index in one region goes fully down during a peak sales event.
5. How would you extend this to cross-title recommendations?
6. How do you evaluate a new ranking model before full rollout, given online A/B tests take time?
7. How do you balance exploration (long-tail) versus exploitation (known converters)?
8. How would you detect and mitigate bot-driven fake engagement?
9. If the entitlement write-primary in US-East goes down, what's the blast radius?
10. How would this architecture change if the catalog grew from 550K to 50M items (full UGC marketplace)?

## 46. Ideal Answers

1. **New title cold-start**: no collaborative signal exists, so rely on content-based candidates (item-tower from metadata) blended with editorial picks; ranking falls back to content/demographic features. Temporarily raise exploration weight to gather data fast.

2. **Offline-online mismatch**: check train/serve skew first (feature store version mismatch is the most common cause). Also check confounds (pricing change, latency-driven fallback) by segmenting the exposure log by surface/region before blaming the model.

3. **Filter-bubble mitigation**: monitor candidate concentration explicitly, use diversity-aware re-ranking (MMR) instead of pure score-ranking. Correct popularity bias in training (logQ correction) and keep an exploration budget for long-tail items.

4. **Regional ANN outage during peak sale**: circuit breaker detects elevated latency within its ~10ms budget and falls back to precomputed popularity candidates — ranking still runs, users get degraded (not broken) results. Alerting fires; traffic can shift away from the affected region.

5. **Cross-title recommendations**: needs a shared embedding space — a global two-tower model on cross-title data, or per-title embeddings projected into a common space. Start with a genre/franchise affinity heuristic before a full joint embedding model.

6. **Pre-rollout evaluation**: combine offline metrics (Recall@K, NDCG, calibration) with off-policy evaluation (IPS using logged exposure + propensity) to estimate online CTR/CVR lift, de-risking the canary's starting allocation.

7. **Exploration vs exploitation**: treat as a contextual bandit layered on ranking output — reserve a small slot share for exploration (Thompson sampling or epsilon-greedy), log propensities for off-policy eval. Tune exploration rate per surface.

8. **Bot manipulation**: ingestion-time anomaly detection (event velocity per user/IP/device) plus training-data filtering that down-weights flagged accounts. Edge rate limiting and monitoring for suspicious item concentration catch the rest.

9. **Entitlement write-primary outage**: blast radius is scoped to new purchase writes and reads dependent on that primary; other regions' async replicas keep serving (slightly stale). System fails closed — excludes ambiguous items rather than risk showing owned content as unowned.

10. **Catalog growth to 50M (UGC marketplace)**: in-memory full-copy HNSW per replica stops scaling — shift to sharded ANN (by category/creator-tier/recency) or a disk-backed system like DiskANN. Candidate generation needs a stronger pre-filter stage (coarse category/tag) to keep search tractable.
