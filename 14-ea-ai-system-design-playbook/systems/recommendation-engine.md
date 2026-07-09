# Recommendation Engine

## 1. Problem Framing & Requirement Gathering

Design a recommendation engine for EA's game store / in-game content surfaces (e.g., EA App storefront, in-game item shop, "recommended for you" rails, next-game-to-play carousels across Battlefield, The Sims, FIFA/EA FC franchises).

- **Business goal**: increase conversion (purchase/install/engagement), increase session length, surface long-tail content (DLC, cosmetics, indie titles) that organic browsing misses.
- **Core ML problem**: given a player (user), context (surface, time, device), and a catalog of items (games, DLC, bundles, cosmetic items, battle-pass content), return a ranked list of items to display.
- **Two-stage system**: candidate generation (retrieval) narrows millions of items to hundreds, ranking scores/orders those to a final top-K (typically 10-50 shown).
- **Cross-cutting concerns**: cold-start (new players, new items/DLC drops), real-time personalization (react to last 5 minutes of play/purchase behavior), freshness (new content must surface within minutes of catalog publish), fairness/diversity (don't just recommend the same AAA titles to everyone).

## 2. Functional Requirements

- FR1: Return personalized ranked item list for a given `(user_id, surface, context)` in real time.
- FR2: Support multiple surfaces: store homepage, item shop, post-game-end carousel, email/push batch recommendations.
- FR3: Support candidate generation via multiple retrieval sources (two-tower embedding ANN, popularity/trending, collaborative filtering, editorial/merchandising overrides).
- FR4: Rank blended candidates with a unified ranking model incorporating pointwise CTR/CVR prediction.
- FR5: Handle cold-start for new users (no history) and new items (no interactions) via content-based fallback.
- FR6: Incorporate real-time signals (last N minutes of clicks/purchases/game sessions) into personalization within the same session.
- FR7: Support exclusion rules (already-owned items, region-locked content, age-rating filters, parental controls).
- FR8: Support experimentation — every recommendation response is tied to an experiment/variant ID for A/B analysis.
- FR9: Provide explainability metadata for merchandising/business teams (why was this item recommended).
- FR10: Batch recommendation generation for offline channels (push notifications, email digests) at scale.

## 3. Non-Functional Requirements (latency, availability, throughput, consistency, cost)

| Dimension | Target |
|---|---|
| Latency (p50) | 40 ms end-to-end (retrieval + ranking + filtering) |
| Latency (p99) | 150 ms end-to-end |
| Availability | 99.95% (store-facing; revenue-impacting) |
| Throughput | 60K QPS peak (global, across surfaces) |
| Consistency | Eventual consistency acceptable for features (seconds-minutes staleness); strong consistency required for "already owned" exclusion (must never recommend owned item) |
| Cost ceiling | Serving cost target < $0.0004 per recommendation request (blended CPU+GPU+cache) |
| Freshness | New catalog items discoverable in ranking within 15 minutes of publish |
| Data durability | Feature store & training data: 99.999999999% (11 nines, object storage) |

## 4. Clarifying Questions an interviewer would expect you to ask

1. Which surfaces are in scope — store front page only, or also in-game overlays and push/email?
2. What's the catalog size — thousands (full games) or millions (cosmetic items, bundles, UGC)?
3. Is this single-title (e.g., only FIFA Ultimate Team) or cross-title (recommend across EA's whole portfolio)?
4. What's the business objective to optimize — purchase conversion, revenue, engagement/retention, or a blended objective?
5. Do we need real-time in-session adaptation (react to last click) or is daily-batch personalization sufficient?
6. Are there legal/compliance constraints — loot box regulations, age-based content filtering, regional purchase restrictions?
7. Is there an existing feature store / data platform to integrate with, or greenfield?
8. What's acceptable staleness for "already purchased" exclusion — can we tolerate a stale cache for a few seconds?
9. Do we need to support cold-start for brand-new game launches with zero interaction history?
10. What's the traffic shape — is there a known peak (game launch day, holiday sale) we must provision for?

## 5. Assumptions

1. 40M MAU across EA App storefront + top 5 live-service titles' in-game shops.
2. Catalog: 500K sellable items (base games, DLC, bundles, cosmetics) + 50K "content" recommendable objects (game modes, event pages).
3. Average user generates 25 interaction events/day (view, click, purchase, session-start).
4. Peak QPS is 5x average due to regional daytime overlap + promotional events (holiday sale, new title launch).
5. Two-tower retrieval model: user tower + item tower, 128-dim embeddings.
6. Ranking model: gradient-boosted trees (GBDT) baseline, migrating to deep ranking (DLRM-style) for mature surfaces.
7. Real-time personalization window: last 30 minutes of session behavior, sourced from streaming feature aggregation.
8. Cold-start new items: ~2,000 new SKUs/week (cosmetics, weekly drops); new user cold-start: ~150K new accounts/day.
9. Infra: multi-region (US-East, US-West, EU-West, AP-Southeast) to meet latency SLAs for global player base.
10. Existing EA data platform provides Kafka-based event bus and an offline data lake (S3/Parquet) — reused, not rebuilt.

## 6. Capacity Estimation (QPS, storage, model size, GPU/CPU counts, back-of-envelope math shown)

**Traffic**
- 40M MAU, avg 6 recommendation-surface views/day/user → 240M requests/day.
- Avg QPS = 240M / 86,400s ≈ 2,780 QPS.
- Peak QPS (5x avg, holiday sale + launch day) ≈ **14,000 QPS**; provision headroom to **20,000 QPS** burst.
- Batch (push/email) recommendation generation: 40M users x 1/day = 40M inferences, run as offline batch over ~2 hrs window → 40M/7200s ≈ 5,500 QPS-equivalent batch throughput (separate pipeline, not p99-sensitive).

**Candidate generation (retrieval) cost**
- Catalog embeddings: 550K items x 128 dims x 4 bytes (fp32) = 550,000 x 512 bytes ≈ 282 MB — fits comfortably in-memory per ANN shard.
- User embedding computed online per request (two-tower user side): small MLP, ~200K params, inference < 2 ms on CPU.
- ANN index (HNSW): with 550K vectors, memory ≈ 282 MB (vectors) + graph overhead (~1.5-2x) ≈ 550-650 MB per replica. Fits on a single node; replicate 6x per region for redundancy + throughput.

**Ranking model**
- Blended candidate set per request: ~300 candidates (200 ANN + 50 popularity/trending + 50 collaborative filtering) after de-dup.
- GBDT ranking: ~500 trees, depth 6 → scoring 300 candidates x 500 trees ≈ 150K tree traversals/request; CPU inference ≈ 8-12 ms/request (batched).
- Deep ranking (DLRM) variant: ~15M parameters (dense + embedding tables for categorical features), fp16 ≈ 30 MB — fits on single GPU (T4/A10) with room for batching; GPU inference batched (32-64 requests/batch) ≈ 6-10 ms/batch at p99.

**GPU/CPU footprint (per region, at peak 20K QPS burst)**
- Ranking (GPU path): assume GPU node handles ~1,500 QPS at batch size 32, ~10ms/batch → 20,000/1,500 ≈ **14 GPU nodes** (A10G class) per region, x4 regions ≈ 56 GPU nodes globally + 30% headroom ≈ **~75 GPU nodes**.
- Retrieval (CPU/ANN path): each ANN shard handles ~3,000 QPS → 20,000/3,000 ≈ 7 nodes/region x 4 regions ≈ **28 CPU nodes** (16 vCPU each).
- Feature store online lookups: Redis cluster, ~20K QPS x ~3 feature lookups/request = 60K ops/sec → single Redis cluster (6 shards, replicated) handles this comfortably (Redis does 100K+ ops/sec/shard).

**Storage**
- Offline training data: 240M events/day x 500 bytes/event (avg, Parquet compressed) ≈ 120 GB/day raw → ~43 TB/year (pre-compaction); with columnar compression (~4x) ≈ 11 TB/year effective.
- Feature store offline (point-in-time tables): ~2 TB (rolling 90-day window of user/item features).
- Online feature store (Redis/DynamoDB): 40M users x ~2 KB/user feature vector = 80 GB; 550K items x ~1 KB = 0.55 GB. Total online footprint ≈ **~85 GB**, trivially fits in-memory cluster.
- Model artifacts: two-tower (~50 MB), GBDT (~20 MB), DLRM (~30 MB) — negligible storage, but versioned (100s of versions/year) ≈ low GB scale.

## 7. High-Level Architecture (with an ASCII diagram showing all major components and data flow)

```
                                   ┌─────────────────────────┐
                                   │   Client / Game Client   │
                                   │ (Store UI, in-game shop) │
                                   └────────────┬─────────────┘
                                                │ HTTPS/gRPC
                                                ▼
                                   ┌─────────────────────────┐
                                   │   API Gateway / Edge     │
                                   │ (authn, rate limit, LB)  │
                                   └────────────┬─────────────┘
                                                ▼
                                   ┌─────────────────────────┐
                                   │  Recommendation Service  │
                                   │  (orchestrator, gRPC)    │
                                   └──┬───────┬────────┬──────┘
                    ┌─────────────────┘       │        └───────────────────┐
                    ▼                         ▼                            ▼
         ┌────────────────────┐   ┌─────────────────────┐     ┌────────────────────┐
         │ Candidate Gen Svc  │   │  Feature Store       │     │ Business Rules /   │
         │ - Two-tower ANN    │◄──┤  (Online: Redis)     │     │ Exclusion Filter   │
         │ - Popularity/Trend │   │  (Offline: Parquet)  │     │ (owned items, geo, │
         │ - Collaborative FN │   └──────────┬───────────┘     │  age rating)       │
         └─────────┬──────────┘              │                └─────────┬──────────┘
                   │ candidates (~300)        │ features                 │
                   ▼                          ▼                          ▼
         ┌─────────────────────────────────────────────────────────────────┐
         │                     Ranking Service (GBDT / DLRM)                │
         │              scores candidates, applies diversity/MMR           │
         └───────────────────────────────┬───────────────────────────────┘
                                          ▼
                                ┌───────────────────┐
                                │  Response Assembly │
                                │  (top-K, metadata, │
                                │   experiment tag)  │
                                └─────────┬──────────┘
                                          ▼
                                   back to Client

   ── Offline / Async plane ──
   Kafka (player events: view/click/purchase/session) ──► Stream Processing (Flink)
        │                                                        │
        ▼                                                        ▼
   Data Lake (S3/Parquet, event log)                    Real-time Feature Aggregator
        │                                                        │
        ▼                                                        ▼
   Training Pipelines (two-tower, ranking model)          Online Feature Store (Redis)
        │
        ▼
   Model Registry ──► Model Serving (rolled out via canary)
        │
        ▼
   Vector Index Builder (nightly + incremental) ──► ANN Index (per region)
```

## 8. Low-Level Components (each major service/component explained: responsibility, interface, scaling unit)

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| API Gateway | AuthN, rate limiting, TLS termination, request routing | REST/gRPC over HTTPS | Horizontal, stateless pods behind L7 LB |
| Recommendation Orchestrator | Fan-out to candidate gen + feature store + rules, calls ranker, assembles response | gRPC (`GetRecommendations`) | Stateless, scales on QPS via HPA |
| Candidate Generation Service | Runs ANN lookup (two-tower), popularity list merge, CF lookup; returns ~300 deduped candidates | Internal gRPC | Scales with ANN index replicas (CPU-bound) |
| Feature Store (Online) | Low-latency key-value lookup of user/item/context features | Redis/DynamoDB client, gRPC facade | Scales via cluster sharding (consistent hashing) |
| Feature Store (Offline) | Point-in-time correct historical features for training | Parquet on S3, queried via Spark/Trino | Scales via storage partitioning |
| Business Rules Engine | Filters owned items, applies region/age restrictions, merchandising overrides | Internal library/service, rule DSL | Stateless, scales with orchestrator |
| Ranking Service | Scores candidates (GBDT/DLRM), applies diversity re-ranking (MMR) | gRPC, batched requests | GPU-backed for DLRM, autoscales on batch queue depth |
| Vector Index / ANN Service | Serves nearest-neighbor lookups against item embeddings | gRPC wrapper over HNSW/FAISS index | Replicated per-region, rebuilt incrementally |
| Stream Processor (Flink) | Aggregates real-time features (last 30 min clicks/purchases) | Kafka consumer → Redis writer | Scales via parallelism/task slots |
| Training Orchestrator | Schedules/executes distributed training jobs | Airflow/Kubeflow DAGs | Scales via job-level GPU cluster provisioning |
| Model Registry | Versions models, tracks lineage, gates promotion | REST API (MLflow-like) | Low-throughput, single cluster |
| Experimentation Service | Assigns variant, logs exposure | gRPC + event log | Stateless, scales with orchestrator |

## 9. API Design (concrete endpoint signatures, request/response schemas, versioning)

```
POST /v2/recommendations
Headers: Authorization: Bearer <JWT>, X-Correlation-ID
Request:
{
  "user_id": "u_9f3a2c",
  "surface": "STORE_HOMEPAGE",       // enum: STORE_HOMEPAGE | IN_GAME_SHOP | POST_MATCH_CAROUSEL | EMAIL_DIGEST
  "context": {
    "platform": "PS5",
    "region": "US-East",
    "session_id": "s_88213",
    "current_game_id": "fifa25"
  },
  "num_results": 20,
  "experiment_overrides": { "variant": "control" }   // optional, for QA/testing
}

Response: 200 OK
{
  "request_id": "req_7788",
  "results": [
    {
      "item_id": "dlc_ea_fc25_ultimate_pack",
      "rank": 1,
      "score": 0.9123,
      "reason_code": "SIMILAR_TO_RECENT_PURCHASE",
      "model_version": "ranker-dlrm-v14"
    }
  ],
  "experiment_id": "exp_reco_diversity_v3",
  "variant": "treatment_b",
  "served_at_ms": 1720440000123
}

Errors:
400 INVALID_REQUEST, 401 UNAUTHENTICATED, 429 RATE_LIMITED, 503 DEGRADED_FALLBACK (popularity-only served)
```

```
POST /v2/recommendations/batch      // offline/batch channel (push/email)
Request: { "user_ids": [...], "surface": "EMAIL_DIGEST", "num_results": 10 }
Response: { "job_id": "batch_2026_07_08_001", "status": "ACCEPTED" }   // async, results land in data lake

GET /v2/recommendations/batch/{job_id}/status
```

- **Versioning**: URI-based (`/v1`, `/v2`); old versions supported for 6 months post-deprecation notice; response includes `model_version` for traceability. Breaking schema changes require major version bump; additive fields are backward compatible.

## 10. Database Design (schema sketches, choice of SQL/NoSQL/columnar and why, partitioning/sharding key)

| Store | Type | Purpose | Partition/Shard Key | Why |
|---|---|---|---|---|
| Online Feature Store | Redis Cluster (key-value) | User/item/context features for real-time serving | `user_id` hash slot | Sub-ms latency, native TTL for freshness |
| Item Metadata Store | DynamoDB | Catalog metadata (price, region, age rating, ownership flags) | `item_id` (partition), `region` (sort in GSI) | Serverless scale, strong consistency option for ownership checks |
| Offline Feature/Event Lake | S3 + Parquet, Iceberg tables | Historical events for training, point-in-time joins | Partitioned by `event_date`, bucketed by `user_id` hash | Columnar compression, cheap at 10s of TB, works with Spark/Trino |
| Ownership/Entitlement DB | PostgreSQL (sharded) | Source of truth: what user owns | Sharded by `user_id` range | Requires strong consistency + transactional guarantees for purchase flow |
| Vector Index Storage | FAISS/HNSW index snapshots on local NVMe + S3 backup | Item embeddings for ANN | Sharded by item cluster (region-specific catalogs) | In-memory ANN needs local disk speed; S3 for durability/rebuild |
| Experiment Exposure Log | Kafka → columnar (ClickHouse) | Log every recommendation exposure + variant for analysis | Partitioned by `experiment_id`, `date` | Fast aggregate queries for A/B analysis at scale |

**Ownership table sketch (Postgres)**
```sql
CREATE TABLE entitlements (
  user_id       BIGINT NOT NULL,
  item_id       VARCHAR(64) NOT NULL,
  acquired_at   TIMESTAMPTZ NOT NULL,
  platform      VARCHAR(16),
  PRIMARY KEY (user_id, item_id)
) PARTITION BY HASH (user_id);
```

## 11. Caching (what's cached, cache invalidation strategy, cache-aside vs write-through)

| Cached Data | Cache | TTL | Strategy | Invalidation |
|---|---|---|---|---|
| User feature vector | Redis | 5 min | Cache-aside | TTL expiry + event-driven invalidation on purchase/session-end |
| Item metadata (price, availability) | Redis + local in-process LRU | 60 s | Cache-aside | Pub/sub invalidation message on catalog update |
| ANN candidate results (per user, per surface) | Redis, short TTL | 30 s | Cache-aside | TTL only (personalization drifts fast, short TTL sufficient) |
| Ownership/entitlement flags | Redis (write-through from Postgres CDC) | N/A (kept fresh) | Write-through | CDC stream (Debezium) pushes updates immediately — must never serve stale "not owned" as "owned" |
| Full recommendation response (identical user+surface+context within window) | CDN/edge cache for anonymous/logged-out fallback only | 2 min | Cache-aside | Not used for logged-in personalized traffic (correctness risk) |

- Ownership cache uses **write-through** deliberately: false negative (user re-shown owned item) is a trust/UX issue; staleness tolerance is near-zero.
- Everything else uses **cache-aside** with short TTLs — personalization staleness of seconds is acceptable, and cache-aside avoids write amplification for high-churn feature data.

## 12. Queues & Async Processing (what's queued, at-least-once vs exactly-once, dead-letter handling)

| Queue/Topic | Producer → Consumer | Delivery Semantics | DLQ Handling |
|---|---|---|---|
| `player-events` (Kafka) | Game client/telemetry → Flink stream processor | At-least-once (Kafka default) | Poison messages (malformed schema) routed to `player-events-dlq`, alert if DLQ rate > 0.1% |
| `catalog-updates` (Kafka) | Content/merchandising service → Feature store updater, cache invalidator | At-least-once, idempotent consumer (upsert by item_id+version) | DLQ + manual replay tool |
| `batch-reco-jobs` (SQS/RabbitMQ) | Batch scheduler → Batch inference workers | At-least-once, idempotency key = job_id | Retried 3x with backoff, then DLQ + PagerDuty page if backlog > 30 min |
| `entitlement-cdc` (Debezium → Kafka) | Postgres CDC → Redis write-through updater | At-least-once, consumer dedupes via LSN/offset checkpoint | Critical path: DLQ triggers immediate alert (correctness-sensitive) |
| `experiment-exposure-log` | Reco service → analytics pipeline | At-least-once (fine for analytics, dedup downstream by request_id) | DLQ retained 7 days, low priority |

- No queue in this system requires strict exactly-once — all consumers are designed idempotent (upsert-by-key or offset-checkpointed), which is cheaper and simpler than transactional exactly-once delivery.

## 13. Streaming & Event-Driven Architecture (topics, event schemas, consumer groups)

**Topic: `player-events`**
```json
{
  "event_type": "ITEM_VIEW | ITEM_CLICK | PURCHASE | SESSION_START | SESSION_END",
  "user_id": "u_9f3a2c",
  "item_id": "dlc_ea_fc25_ultimate_pack",
  "surface": "STORE_HOMEPAGE",
  "timestamp": 1720440000123,
  "platform": "PS5",
  "session_id": "s_88213",
  "value_usd": 4.99
}
```
- **Consumer groups**: `realtime-feature-aggregator` (Flink, updates Redis short-window features), `training-data-writer` (Spark structured streaming → Iceberg lake), `experiment-analytics` (ClickHouse sink).
- **Partitioning**: by `user_id` to preserve per-user event ordering for session-window aggregation.
- **Retention**: 7 days hot (Kafka), then compacted into lake (indefinite, subject to data retention policy).

**Topic: `catalog-updates`** — schema includes `item_id`, `version`, `price`, `region_availability[]`, `age_rating`, `embedding_refresh_required: bool`. Consumed by ANN index incremental updater and feature store.

## 14. Model Serving (serving framework choice, batching, multi-model, hardware)

- **Two-tower retrieval model**: item tower embeddings precomputed offline, served via ANN index (FAISS/HNSW) — no per-request GPU inference needed for items. User tower runs online (small MLP) on CPU, <2ms.
- **Ranking model**: 
  - GBDT baseline served via **Triton Inference Server** (FIL backend) on CPU — cheap, adequate for tabular features.
  - DLRM-style deep ranker served via **Triton** on GPU (A10G), with **dynamic batching** (max batch 64, max queue delay 5ms) to improve GPU utilization.
- **Multi-model serving**: Triton hosts multiple model versions concurrently (canary v14 + stable v13) with per-model resource pools; routing controlled by orchestrator via model-version header.
- **Hardware**: CPU nodes (c6i.4xlarge-class) for GBDT + ANN; GPU nodes (g5.xlarge/A10G) for DLRM; batch training uses A100 nodes.
- **Fallback**: if ranking service degraded (timeout > 50ms), orchestrator serves popularity-ranked candidates directly (graceful degradation, not full failure).

## 15. Feature Store (online/offline split, point-in-time correctness)

- **Offline store**: Iceberg/Parquet tables on S3, versioned features with event-time timestamps. Training pipeline performs **point-in-time joins** — for each training example (user, item, label, timestamp T), only features known as-of T are joined, preventing label leakage (e.g., can't use "purchased in last hour" feature computed after the label event).
- **Online store**: Redis, materialized from the same feature definitions (via Feast-style feature store abstraction) to guarantee **train/serve consistency** — same transformation code path (feature logic defined once, executed in both batch and streaming context).
- **Real-time features** (session clicks in last 30 min): computed by Flink directly into Redis, bypassing offline store for freshness; backfilled into offline store nightly for training consistency.
- **Feature versioning**: every feature has a schema version; ranking model records which feature version set it was trained against, enforced at serving time to prevent skew.

## 16. Vector Database (if applicable — indexing strategy, ANN algorithm choice)

- **Applicable** — core to candidate generation.
- **Algorithm**: HNSW (Hierarchical Navigable Small World) — chosen over IVF-PQ because catalog size (550K items) is small enough that HNSW's higher recall/lower latency tradeoff wins; memory footprint (~600MB/replica) is affordable at this scale.
- **Index build**: nightly full rebuild (batch, off-peak) + incremental insert path for new items (append-only HNSW insert, rebalanced weekly to control graph degradation).
- **Sharding**: one full index per region (all 550K items) rather than sharding by item — simpler operationally, replicated 6x per region for throughput/redundancy; catalog fits comfortably in memory so no need to shard.
- **Query**: user embedding (128-dim) → top-200 ANN lookup, `ef_search=64` tuned for ~2ms p99 per lookup at this scale.
- **Alternative considered**: ScaNN (Google) — comparable recall/latency; HNSW chosen for broader OSS tooling/ops familiarity (FAISS ecosystem).

## 17. Embedding Pipelines (if applicable, else N/A and why)

- **Applicable.**
- **User tower**: consumes user features (recent interaction history, demographic bucket, platform, region, spend tier) → 128-dim embedding via 3-layer MLP.
- **Item tower**: consumes item metadata (genre, price tier, franchise, content type, text description embedding from a frozen sentence-transformer) → 128-dim embedding.
- **Training**: in-batch negative sampling with softmax loss (standard two-tower retrieval training), sampled from logged impressions + random negatives to correct for popularity bias (logQ correction).
- **Embedding refresh cadence**: item embeddings recomputed nightly (batch) + on-demand for new catalog drops (triggered via `catalog-updates` topic, `embedding_refresh_required` flag) so new DLC is discoverable within the 15-minute freshness SLA.
- **Embedding drift check**: compare cosine similarity distribution of new vs. previous day's item embeddings; large shifts trigger manual review before index swap.

## 18. Inference Pipelines (request lifecycle end-to-end)

```
t=0ms    Client sends GetRecommendations(user_id, surface, context)
t=1ms    API Gateway: authn (JWT verify), rate-limit check → forward
t=2ms    Orchestrator: fan-out (parallel):
           ├─ Feature Store lookup (user features, context features)      [~3ms]
           ├─ Candidate Gen: user-tower inference + ANN top-200 lookup    [~6ms]
           ├─ Candidate Gen: popularity list fetch (precomputed, cached)  [~1ms]
           └─ Entitlement check (owned items) via Redis write-through     [~2ms]
t=8ms    Merge + de-dup candidates (~300 total) → filter owned/region-locked
t=10ms   Ranking Service: batch-score 300 candidates (GBDT or DLRM)       [~10ms p50, ~25ms p99]
t=20ms   Diversity re-ranking (MMR, ensure no >3 items from same franchise in top 10)
t=22ms   Response assembly (attach reason_code, experiment tag, model_version)
t=24ms   Return to client
         (p50 total ~25-30ms; p99 budget 150ms allows for feature-store cache
          misses, GPU batching queue delay, cross-AZ hops)
```

## 19. Training Pipelines (data prep, training orchestration, distributed training if relevant)

- **Data prep**: Spark jobs read `player-events` lake, join with entitlement + catalog snapshots (point-in-time correct), generate labeled training rows (impression → click/purchase within attribution window, e.g. 24h for click, 7d for purchase attribution).
- **Negative sampling**: for retrieval (two-tower), in-batch negatives + hard negative mining from near-miss ANN results (items ranked 200-400 that weren't clicked) to sharpen the embedding space.
- **Orchestration**: Airflow DAG — `extract_events → build_training_set → train_two_tower → eval → train_ranker → eval → register_models → trigger_canary_rollout`.
- **Distributed training**: two-tower and DLRM ranker trained via PyTorch DDP across 8x A100 GPUs (data-parallel; model is small enough that model-parallelism isn't needed). GBDT baseline trained via distributed XGBoost (CPU cluster, ~20 nodes) for large tabular datasets.
- **Training data volume**: ~200M labeled rows/training run (30-day rolling window), ~11 TB compressed — full retrain takes ~3 hrs on the 8x A100 cluster for the two-tower model.
- **Evaluation**: offline metrics (Recall@200 for retrieval, NDCG@10 and AUC for ranking) gated against holdout set before promotion; replay-based counterfactual evaluation (IPS-weighted) to estimate online lift before full rollout.

## 20. Retraining Strategy (cadence, triggers)

| Model | Cadence | Trigger-based retrain |
|---|---|---|
| Two-tower retrieval | Daily (embedding refresh), full retrain weekly | Triggered early if embedding drift check fails or Recall@200 drops >3% on canary |
| Ranking model (GBDT) | Daily incremental retrain | Triggered if feature distribution drift (PSI) exceeds threshold |
| Ranking model (DLRM) | Weekly full retrain, daily fine-tune on last 24h | Triggered by concept drift alert (CTR prediction calibration error > 10%) |
| Popularity/trending lists | Hourly recompute | Event-driven (new title launch, promo event start) |
| Item embeddings (new catalog) | On-demand (within 15 min of publish) | Triggered by `catalog-updates` Kafka event |

- Major live-service events (game launch day, holiday sale start) trigger **out-of-band manual retrain + canary** ahead of the event, since organic daily cadence may not capture pre-event behavior shifts.

## 21. Drift Detection (data drift, concept drift, what metrics, what thresholds)

| Drift Type | Metric | Threshold | Action |
|---|---|---|---|
| Feature data drift | Population Stability Index (PSI) per feature | PSI > 0.2 → alert; > 0.3 → auto-block model promotion | Investigate upstream data source, retrain with recent window |
| Embedding drift | Cosine similarity shift of item embedding centroids day-over-day | Mean shift > 0.15 | Manual review before index swap |
| Concept drift (label distribution) | CTR/CVR calibration error (predicted vs actual, binned) | Calibration error > 10% | Trigger retrain, widen exploration (increase popularity-fallback weight temporarily) |
| Candidate coverage drift | % of impressions from top-10 items (concentration) | > 40% (indicates collapse to popularity, losing personalization) | Alert merchandising + ML team, check diversity re-ranker |
| Cold-start ratio | % of requests served pure fallback (no personalization signal) | > 15% sustained | Investigate onboarding funnel / feature pipeline gaps |

- Drift monitoring runs as a scheduled batch job (daily) comparing production feature/prediction distributions against a rolling 7-day baseline, using Evidently-style drift reports feeding into the monitoring dashboard.

## 22. Monitoring (what's monitored: infra, model quality, business metrics)

| Layer | Metrics |
|---|---|
| Infra | p50/p95/p99 latency per service, error rate, GPU utilization, Redis hit rate, Kafka consumer lag, queue depth |
| Model quality | Recall@K (retrieval), NDCG@10 (ranking), calibration error, prediction score distribution, feature drift (PSI) |
| Business | CTR, conversion rate (CVR), revenue-per-recommendation, session length uplift, diversity/coverage of recommended catalog, cold-start user conversion rate |
| Experiment | Variant-level lift (CTR/CVR delta), sample ratio mismatch checks, exposure log completeness |
| Data pipeline | Freshness lag (event timestamp → feature-available timestamp), training job success/duration, data volume anomalies |

- Dashboards: Grafana (infra + model metrics from Prometheus), business metrics dashboard (Looker/Tableau fed from ClickHouse exposure log).

## 23. Alerting (alert conditions, thresholds, on-call routing)

| Alert | Condition | Severity | Routing |
|---|---|---|---|
| API p99 latency breach | p99 > 150ms for 5 consecutive min | High | Page on-call SRE (PagerDuty) |
| Error rate spike | 5xx rate > 1% over 5 min | High | Page on-call SRE |
| Ranking service degraded-fallback rate | > 10% of requests served popularity-fallback for 10 min | Medium | Slack alert to ML platform team |
| Entitlement CDC lag | Consumer lag > 30s | Critical (correctness risk — could show owned item) | Immediate page |
| Feature store cache hit rate drop | < 90% for 15 min | Medium | Slack alert |
| Model drift (PSI/calibration) breach | Per thresholds in Sec 21 | Medium | Slack alert to ML team, auto-block promotion pipeline |
| Training pipeline failure | DAG failure | Medium | Email + Slack to data eng on-call |
| DLQ backlog | > 30 min unprocessed | High | Page on-call |

- On-call rotation: ML platform on-call (model/serving issues) separate from data-eng on-call (pipeline/ETL issues); infra/SRE on-call owns latency/availability pages, escalates to ML on-call if root cause is model-side.

## 24. Logging (structured logging strategy, PII handling, retention)

- **Structured JSON logs** for every request: `request_id`, `user_id` (hashed/pseudonymized in long-term storage), `surface`, `latency_ms`, `model_versions_used`, `candidate_count`, `experiment_id/variant`, `error_code`.
- **PII handling**: raw `user_id` used only in short-lived hot path (Redis, in-memory); logs destined for long-term storage/analytics have `user_id` replaced with a salted hash; no free-text PII (names, payment info) ever enters the recommendation logging path — those live in separate PCI-scoped systems.
- **Retention**: request logs 30 days hot (searchable, Elasticsearch/OpenSearch), 1 year cold (S3, compliance/audit), exposure logs (for experimentation) retained 2 years (aggregated, anonymized) for longitudinal analysis.
- **Access control**: logs containing hashed user identifiers restricted to ML/data-eng roles via IAM; full audit trail on log access for compliance (GDPR/CCPA "right to be forgotten" must propagate deletion requests into log retention pipeline).

## 25. Security (authn/authz, data encryption, threat model specific to this system)

- **Data encryption**: TLS 1.3 in transit (client↔gateway, service↔service via mTLS in-mesh); encryption at rest (S3 SSE-KMS, Redis encryption-at-rest, Postgres TDE) for all feature/event stores.
- **Threat model specifics**:
  - **Recommendation manipulation**: malicious actor gaming the system (fake clicks/purchases via bots) to boost specific item ranking — mitigate via bot detection on event ingestion, anomaly detection on interaction velocity per user/IP.
  - **Data poisoning**: adversarial injection of fake interaction events to skew training data — mitigate via event validation, rate limiting on event ingestion per user, outlier filtering in training data prep.
  - **Model extraction/inversion**: repeated querying of ranking API to reverse-engineer model or extract embeddings — mitigate via rate limiting, response obfuscation (don't return raw scores, only ranks), monitoring for query patterns consistent with scraping.
  - **PII leakage via embeddings**: user embeddings could theoretically leak behavioral patterns if exposed — embeddings never returned in API responses, access to raw embedding store restricted to internal services only.
  - **Ownership bypass**: incorrect entitlement check could recommend (or worse, unlock via a related flow) content the user hasn't purchased — entitlement check treated as security-critical, not just relevance-critical.

## 26. Authentication (service-to-service and end-user auth mechanism)

- **End-user**: OAuth2/OIDC via EA's central identity provider; client obtains short-lived JWT (15 min expiry), passed as Bearer token to Recommendation API; gateway validates signature + expiry + audience claim.
- **Service-to-service**: mTLS within the service mesh (Istio/Linkerd-style sidecar), SPIFFE/SPIFFE-ID-based workload identity; internal gRPC calls (orchestrator → ranking service, etc.) authenticated via mTLS certs, not shared secrets.
- **Batch/offline jobs**: IAM role-based access (cloud-native, e.g., IRSA on EKS) for S3/data lake access — no long-lived static credentials.
- **Token scope**: user JWT scoped to read-only recommendation access; no write/purchase capability granted through this token (purchase flow uses separate, more tightly scoped commerce auth).

## 27. Rate Limiting (algorithm choice, per-user/per-tenant limits)

- **Algorithm**: token bucket at the API Gateway (per-user) — allows short bursts (e.g., rapid carousel scrolling) while capping sustained rate.
- **Limits**: 20 requests/sec/user burst, 5 requests/sec/user sustained (refill rate); batch endpoints limited per-service-account (e.g., email digest generator capped at 2,000 req/sec aggregate).
- **Per-surface tenant limits**: each surface (store homepage, in-game shop, etc.) registered as a "tenant" with its own quota pool to prevent one surface's traffic spike from starving others — implemented via Redis-backed distributed token bucket counters.
- **Abuse handling**: sustained rate-limit violations (bot-like patterns) escalate to a stricter, longer-lived block (sliding window ban) and feed into the bot-detection signal for the security threat model above.

## 28. Autoscaling (metrics-driven autoscaling policy, HPA/VPA/KEDA specifics)

```yaml
# HPA for ranking-service (GPU-backed DLRM path)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ranking-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ranking-service
  minReplicas: 8
  maxReplicas: 60
  metrics:
    - type: Pods
      pods:
        metric:
          name: triton_inference_queue_depth
        target:
          type: AverageValue
          averageValue: "10"
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 65
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies: [{ type: Percent, value: 100, periodSeconds: 60 }]
    scaleDown:
      stabilizationWindowSeconds: 300
      policies: [{ type: Percent, value: 20, periodSeconds: 120 }]
```

- **HPA** on orchestrator/candidate-gen services keyed on QPS + p99 latency (custom Prometheus adapter metric).
- **KEDA** used for batch inference workers, scaling consumer pods based on `batch-reco-jobs` queue depth (scale-to-zero when idle, since batch jobs are bursty and off-peak).
- **VPA** (recommendation mode only, not auto-apply) for GBDT ranking pods to right-size CPU/memory requests based on observed usage — avoided auto-apply to prevent restart churn during peak traffic.
- GPU node pools use **cluster autoscaler** with a dedicated node group (A10G), scale-up latency mitigated by keeping a small warm buffer (2 idle GPU nodes) during known peak windows (evenings, weekends, launch days).

## 29. Cost Optimization (concrete levers: spot instances, caching, model distillation, batching)

- **Spot instances**: training cluster (A100 nodes) uses spot/preemptible instances with checkpointing every 500 steps — ~60-70% cost reduction on training compute; serving GPU nodes use on-demand/reserved (availability-critical) but batch inference workers (offline channel) use spot.
- **Model distillation**: DLRM ranking model distilled from a larger teacher ensemble (trained offline, expensive) into a smaller student model for serving — cuts GPU inference cost ~40% with <1% NDCG regression.
- **Caching**: aggressive short-TTL caching (Section 11) cuts redundant feature-store and ANN lookups for repeat requests within a session — reduces ANN QPS load ~25% during high-repeat-view surfaces (store homepage refresh).
- **Batching**: dynamic batching on Triton (GPU ranking) improves throughput/GPU-$ by ~3x versus per-request inference.
- **CPU vs GPU tiering**: GBDT ranker (CPU) used for lower-value/lower-traffic surfaces (e.g., email digest), reserving GPU DLRM capacity for high-value real-time surfaces (store homepage, in-game shop) where the accuracy lift matters most for revenue.
- **Reserved capacity + Savings Plans** for baseline steady-state serving fleet (the ~2,780 QPS average load), with autoscaling/spot handling burst above baseline.
- **Storage lifecycle**: raw event logs moved to S3 Glacier after 90 days; Iceberg table compaction reduces small-file overhead and query cost.

## 30. Disaster Recovery (RTO/RPO targets, backup strategy)

| Component | RTO | RPO | Backup Strategy |
|---|---|---|---|
| Online feature store (Redis) | 5 min | 5 min | Multi-AZ replication, automated failover; snapshot every 5 min to S3 |
| Entitlement DB (Postgres) | 15 min | < 1 min | Synchronous replica in secondary AZ, PITR via WAL archiving |
| Vector index (ANN) | 30 min | 24 hr (rebuildable) | Nightly snapshot to S3; can rebuild from item embedding table if snapshot lost |
| Model artifacts | 10 min | 0 (versioned, immutable) | Model registry backed by S3 with versioning + cross-region replication |
| Event lake (S3/Iceberg) | N/A (durable) | Near-zero | S3 cross-region replication, versioning enabled |
| Full regional outage | 15 min (failover to secondary region) | < 5 min (async replication lag) | Multi-region active-active (Section 31) — traffic reroutes via global LB |

- **DR drills**: quarterly game-day exercises simulating regional failover and Redis cluster loss, validated against RTO/RPO targets.

## 31. Multi-Region Deployment (active-active vs active-passive, data replication, latency routing)

```
                         ┌─────────────────────────┐
                         │   Global DNS / Anycast   │
                         │   Latency-based routing  │
                         └───────┬─────────┬────────┘
                 ┌───────────────┘         └───────────────┐
                 ▼                                          ▼
      ┌─────────────────────┐                    ┌─────────────────────┐
      │  Region: US-East     │                    │  Region: EU-West     │
      │  (active)            │◄──async replicate──►│  (active)            │
      │  - Full reco stack   │                    │  - Full reco stack    │
      │  - Redis cluster     │                    │  - Redis cluster      │
      │  - ANN index replica │                    │  - ANN index replica  │
      │  - Ranking GPUs      │                    │  - Ranking GPUs       │
      └──────────┬───────────┘                    └──────────┬────────────┘
                 │                                            │
                 └───────────► Global Feature/Event Lake ◄─────┘
                              (S3 cross-region replicated,
                               Kafka MirrorMaker2 for event bus)
```

- **Topology**: active-active across 4 regions (US-East, US-West, EU-West, AP-Southeast) — recommendation serving is stateless-ish per-region (features replicated), so active-active is feasible and maximizes latency benefit for a global player base.
- **Data replication**: entitlement DB (Postgres) uses regional read-replicas with a single write-primary (in user's home region) + async replication to other regions for read-serving of ownership checks — write path (purchases) always routes to home region to avoid conflict resolution complexity.
- **Event bus**: Kafka MirrorMaker2 replicates `player-events` cross-region so training pipelines see global data.
- **Latency routing**: GeoDNS/Anycast + latency-based routing policy (e.g., Route53 latency records) directs client to nearest healthy region; health-check-based failover removes unhealthy region from rotation within ~30s.
- **Conflict handling**: since ownership writes are single-region-primary, no multi-master write conflicts for the correctness-critical path; feature store writes are region-local and don't require cross-region consistency (personalization data, not authoritative).

## 32. Blue/Green Deployment (how it applies to this system specifically)

- Applied primarily to the **Recommendation Orchestrator** and **Ranking Service** deployments.
- Green environment stood up with new model version + new service code, receiving zero production traffic initially; smoke-tested via shadow traffic (mirrored real requests, responses discarded) to validate latency/error-rate parity.
- Cutover: load balancer switches 100% traffic from blue to green atomically once shadow validation passes (latency p99 within 10% of blue, error rate < 0.1% delta).
- Blue environment kept warm for 1 hour post-cutover as instant-rollback target (just flip LB back) before decommissioning.
- Feature store and vector index are **not** blue/green'd (shared stateful backends) — only the versioned model artifacts and serving code follow blue/green; this avoids the complexity of dual-writing to two feature store copies.

## 33. Canary Deployment (traffic-split strategy, health-check gates specific to this system)

- New ranking model version deployed alongside stable version in Triton (multi-model serving); orchestrator routes a **small traffic slice** (1% → 5% → 25% → 100%) over a staged rollout window (e.g., 1% for 2 hrs, 5% for 6 hrs, 25% for 24 hrs, then full).
- **Health-check gates** at each stage (must pass to proceed):
  - p99 latency within 10% of control.
  - Error rate not exceeding control + 0.2 absolute.
  - Online NDCG@10 (computed from live click-through, near-real-time) not regressed > 2% vs control.
  - No spike in "degraded-fallback" rate.
  - CTR/CVR from experiment exposure log not significantly negative (statistical guardrail metric check).
- Automatic rollback triggers if any gate fails during a stage; a human approval gate required before the 25%→100% final promotion (Principal-level sign-off for revenue-impacting changes).
- Canary specifically isolates **model version** changes; infra/code changes canaried separately via standard rolling deployment with readiness probes.

## 34. Rollback Strategy (automated triggers, rollback mechanics)

- **Automated triggers**: any canary health-check gate failure (Section 33) triggers automatic traffic reversion to stable model version within 1 minute (orchestrator flips routing weight to 0% for canary).
- **Rollback mechanics**: since both model versions are hot in Triton simultaneously during canary, rollback is a **routing change**, not a redeploy — near-instant (<10s propagation via config push/service mesh).
- **Post-rollback**: incident automatically opened, canary model marked "failed" in model registry (blocks re-promotion without manual review), on-call ML engineer paged for RCA.
- **Data-pipeline rollback**: if a bad feature pipeline deploy corrupts online features, rollback = revert feature-computation code + replay last-known-good feature snapshot from offline store into Redis (bounded by point-in-time correctness of the offline store).
- **Full service rollback** (non-model code): standard Kubernetes rollout undo (`kubectl rollout undo`) for orchestrator/API layer, gated by readiness probes before traffic shifts back.

## 35. Observability (tracing, metrics, logs correlation — the three pillars applied here)

- **Tracing**: OpenTelemetry distributed tracing across orchestrator → feature store → candidate gen → ranking service, propagated via `X-Correlation-ID`/trace-context headers; traces exported to Jaeger/Tempo, enabling per-request latency breakdown (identify whether a slow request was ANN lookup, feature fetch, or GPU batching queue delay).
- **Metrics**: Prometheus scrapes per-service latency histograms, error counters, GPU utilization (DCGM exporter), cache hit ratios, queue depths; aggregated in Grafana with per-surface and per-region breakdowns.
- **Logs**: structured JSON logs (Section 24) correlated via `request_id`/trace ID, shipped to OpenSearch; enables jumping from a slow trace span directly to the corresponding log lines and, from there, to the model version and feature snapshot used for that request.
- **Correlation in practice**: a single `request_id` ties together the API access log, the trace spans, the ranking service's model-version log field, and the experiment exposure log entry — enabling full request reconstruction for debugging a specific bad recommendation.
- **SLO dashboards**: burn-rate alerting on the 99.95% availability SLO (fast-burn and slow-burn multi-window alerts) layered on top of raw metrics.

## 36. Kubernetes Deployment (a concrete manifest sketch or Deployment/Service/HPA YAML snippet relevant to this system)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reco-orchestrator
  labels: { app: reco-orchestrator }
spec:
  replicas: 12
  selector:
    matchLabels: { app: reco-orchestrator }
  template:
    metadata:
      labels: { app: reco-orchestrator }
    spec:
      containers:
        - name: orchestrator
          image: ea-registry/reco-orchestrator:2026.07.01
          resources:
            requests: { cpu: "1", memory: "1Gi" }
            limits: { cpu: "2", memory: "2Gi" }
          ports: [{ containerPort: 8080 }]
          readinessProbe:
            httpGet: { path: /healthz, port: 8080 }
            initialDelaySeconds: 5
            periodSeconds: 10
          env:
            - name: FEATURE_STORE_ADDR
              value: "redis-cluster.reco.svc.cluster.local:6379"
---
apiVersion: v1
kind: Service
metadata:
  name: reco-orchestrator-svc
spec:
  selector: { app: reco-orchestrator }
  ports: [{ port: 80, targetPort: 8080 }]
  type: ClusterIP
```
(HPA for this deployment shown in Section 28.)

## 37. Terraform Infrastructure (a concrete Terraform snippet sketch for the core infra of this system)

```hcl
resource "aws_eks_node_group" "ranking_gpu_pool" {
  cluster_name    = aws_eks_cluster.reco_cluster.name
  node_group_name = "ranking-gpu-a10g"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = var.private_subnet_ids

  instance_types = ["g5.xlarge"]
  capacity_type  = "ON_DEMAND"

  scaling_config {
    desired_size = 14
    min_size     = 8
    max_size     = 60
  }

  labels = { workload = "gpu-ranking" }
  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }
}

resource "aws_elasticache_replication_group" "feature_store" {
  replication_group_id       = "reco-feature-store"
  description                 = "Online feature store for recommendation engine"
  node_type                   = "cache.r6g.xlarge"
  num_node_groups             = 6
  replicas_per_node_group     = 2
  automatic_failover_enabled  = true
  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
  engine                      = "redis"
  engine_version               = "7.1"
}
```

## 38. Why This Architecture (justification)

- **Two-stage retrieval + ranking** is the standard, proven pattern for large-catalog recommendation (Netflix, YouTube, Meta) — it's necessary because scoring 550K items per request with a heavy ranking model is computationally infeasible at 14K QPS; ANN retrieval narrows the field cheaply first.
- **Two-tower model for retrieval** decouples user and item computation, allowing item embeddings to be precomputed offline (amortized cost) while only the cheap user-tower runs online — this is what makes sub-10ms retrieval possible at this scale.
- **Feature store with online/offline split** is required to guarantee train/serve consistency — without it, subtle skew between training-time and serving-time feature computation is one of the most common silent failure modes in production recommenders.
- **GBDT + DLRM dual-path ranking** balances cost and quality — not every surface needs the most expensive model; tiering serving cost to business value (Section 29) is a deliberate architectural choice, not an afterthought.
- **Multi-region active-active** is justified by EA's genuinely global player base and the tight p99 latency budget (150ms) — cross-continent round trips alone would blow the budget if single-region.
- **Correctness-critical paths (entitlement/ownership) treated separately** from personalization paths (features) — different consistency requirements demand different data stores and cache strategies (write-through vs cache-aside), which this architecture explicitly separates rather than treating uniformly.

## 39. Alternative Architectures (at least 2 alternatives with why they were rejected or when they'd be preferred)

| Alternative | Description | Why Rejected (or when preferred) |
|---|---|---|
| Single-stage ranking (score full catalog per request) | Skip candidate generation, rank all 550K items directly | Rejected: computationally infeasible at target QPS/latency (550K x DLRM inference per request would blow both latency and GPU cost budgets by orders of magnitude); would only be viable for catalogs < ~5K items or offline batch use cases with no latency constraint |
| Pure collaborative filtering (matrix factorization, no content features) | Classic user-item matrix factorization, no two-tower/content embeddings | Rejected as sole method: fails badly on cold-start (new users/items have no interaction history) which is a hard requirement here given frequent new DLC drops and 150K new users/day; still used as one candidate-gen source blended in, not as the sole retrieval method |
| Fully batch/offline precomputed recommendations (no real-time serving) | Precompute top-N recommendations nightly per user, serve from a lookup table | Preferred *when* real-time personalization isn't required (e.g., email digest surface uses exactly this pattern) — rejected as the *sole* approach because it can't react to in-session behavior (e.g., just-viewed item), which materially hurts conversion on high-intent surfaces like the in-game shop |
| Single global model (no franchise/title-specific tuning) | One model across all EA titles, no per-title fine-tuning | Rejected for mature high-traffic titles (FIFA/EA FC, Battlefield) where title-specific behavioral patterns are strong enough to justify per-title fine-tuned ranking heads; retained as the default/fallback for smaller/newer titles with insufficient data to fine-tune |

## 40. Tradeoffs (explicit tradeoff table)

| Decision | Pro | Con |
|---|---|---|
| Two-stage retrieval+ranking | Scales to large catalogs, cheap at high QPS | Extra architectural complexity, potential recall loss if ANN misses relevant items ranker would've scored highly |
| HNSW ANN index | Fast, high recall at this catalog scale | Memory-resident, incremental inserts degrade graph quality (needs periodic rebuild) |
| Write-through cache for entitlements | Strong consistency, avoids "recommend owned item" bugs | Higher write latency/complexity vs. simple cache-aside |
| Active-active multi-region | Best latency for global users, resilient to regional outage | Operational complexity (replication lag, no multi-master writes for entitlements) |
| GBDT + DLRM dual path | Cost-efficient (cheap model where it's good enough) | Two models to maintain, monitor, and keep feature-consistent |
| Short-TTL caching (aggressive) | Reduces backend load significantly | Slight staleness risk in personalization (acceptable) but requires careful separation from correctness-critical data (ownership) |
| Canary + staged rollout | Catches regressions before full exposure | Slower time-to-full-rollout for genuinely good models; added engineering overhead (shadow traffic, staged gates) |
| Distilled student ranking model | Cheaper GPU inference | Slight accuracy regression vs. full teacher ensemble (~1% NDCG) |

## 41. Failure Modes (concrete failure scenarios and mitigations)

| Failure | Impact | Mitigation |
|---|---|---|
| ANN index service down (region) | Candidate generation degraded to popularity-only | Orchestrator circuit-breaker falls back to popularity/trending candidates within timeout budget (10ms) |
| Redis feature store outage | Missing user features → ranking uses defaults | Ranking model trained with missing-feature handling (default embeddings); degraded but non-zero quality; alert fires |
| Entitlement CDC lag/outage | Risk of recommending owned item | Fail-closed: if entitlement freshness can't be guaranteed, filter conservatively (exclude uncertain items) rather than risk showing owned content |
| GPU node pool exhaustion (ranking) | Ranking latency spikes, requests queue | Autoscaler + fallback to CPU GBDT path if GPU queue depth exceeds threshold |
| Bad model deploy (accuracy regression) | Poor recommendations, revenue impact | Canary gates (Section 33) catch before full rollout; automatic rollback |
| Kafka broker outage | Real-time feature updates stall, training data gap | Multi-AZ Kafka replication; consumers resume from last committed offset on recovery; short outage tolerable given features degrade gracefully to slightly stale |
| Cross-region replication lag spike | Stale features served in secondary region | Monitored lag metric; if lag exceeds threshold, region temporarily removed from active-active rotation |
| Cascading retry storm (client retries on timeout) | Amplifies load during partial degradation | Client-side exponential backoff + jitter; server-side load shedding (reject with 503 + Retry-After beyond capacity threshold) |

## 42. Scaling Bottlenecks (where this breaks first at 10x/100x scale)

- **At 10x (140K QPS)**: ANN index replica count and Redis feature-store shard count become the first bottleneck — need to shard the ANN index (currently full-copy per replica) rather than just adding more full replicas, and move from a single Redis cluster to a sharded multi-cluster feature store topology.
- **At 10x**: ranking GPU fleet grows linearly (~140-150 GPU nodes/region) — cost becomes a forcing function to push harder on distillation/quantization (INT8) rather than just scaling node count.
- **At 100x (1.4M QPS, unrealistic for EA today but illustrative)**: catalog size likely also grows (more titles, more UGC) — HNSW full-index-per-replica stops being viable in memory; would need to shift to sharded ANN (e.g., partition by title/genre) with a scatter-gather query pattern, adding latency and complexity.
- **At 100x**: single-write-primary-per-region entitlement DB becomes a write bottleneck for purchase-heavy events (e.g., massive flash sale) — would need to revisit to a distributed SQL (e.g., CockroachDB/Spanner-style) for the entitlement store.
- **Kafka event throughput**: at 100x event volume, current topic partitioning (by user_id) may create hot partitions for highly active users/bots — would need partition key revisit or secondary sharding dimension.
- **Feature store write amplification**: real-time feature aggregation (Flink) at 100x event volume needs proportionally more stream-processing parallelism; state size for windowed aggregations grows, requiring more RocksDB-backed state store capacity per task manager.

## 43. Latency Bottlenecks (where time is actually spent, p50/p99 budget breakdown)

| Stage | p50 | p99 |
|---|---|---|
| Gateway (authn, rate limit) | 1 ms | 3 ms |
| Feature store lookup | 3 ms | 15 ms (cache miss path) |
| Candidate gen (ANN + popularity) | 6 ms | 20 ms (index contention/cold shard) |
| Entitlement check | 2 ms | 8 ms |
| Merge/filter/de-dup | 2 ms | 4 ms |
| Ranking (GBDT/DLRM, batched) | 10 ms | 40 ms (batching queue delay under load) |
| Diversity re-ranking (MMR) | 2 ms | 5 ms |
| Response assembly | 1 ms | 2 ms |
| Network/cross-service overhead | 3 ms | 10 ms |
| **Total** | **~30 ms** | **~107 ms** (within 150ms budget, ~43ms headroom) |

- **Biggest p99 contributor**: ranking service batching queue delay under load — dynamic batching trades a few ms of queueing for 3x GPU throughput; tuned batch-wait window (5ms max) caps the downside.
- **Second biggest**: feature store cache misses — mitigated by pre-warming hot user features on session start.

## 44. Cost Bottlenecks (what actually drives the bill)

- **GPU ranking fleet** is the single largest cost driver (~75 GPU nodes globally at peak) — dominates over CPU/ANN/storage costs combined; this is why distillation, batching, and CPU/GPU tiering (Section 29) are first-priority levers, not afterthoughts.
- **Cross-region data replication** (Kafka MirrorMaker2, S3 cross-region replication) — data egress costs scale with event volume; mitigated by compressing events and filtering non-essential fields before replication.
- **Redis cluster memory** — while individually cheap, at 40M users x multiple regions x replication factor, aggregate memory footprint is a recurring line item; short TTLs help bound this.
- **Training compute** (A100 clusters) — periodic but significant; spot instances are the main lever here.
- **Storage growth** (event lake) — linear with player base and event volume; lifecycle policies (Glacier tiering, compaction) are the main control.

## 45. Interview Follow-Up Questions

1. How would you handle a brand-new game title launch with zero historical interaction data — walk through cold-start end-to-end.
2. The ranking model's CTR predictions are well-calibrated offline but conversion drops 5% after a rollout — how do you debug this?
3. How do you prevent the recommendation system from creating a filter-bubble / popularity-collapse feedback loop over time?
4. Walk through what happens, in detail, if the ANN index service in one region goes fully down during a peak sales event.
5. How would you extend this system to support cross-title recommendations (recommend a different EA game based on behavior in another)?
6. How do you evaluate whether a new ranking model is actually better, before it's fully rolled out, given that online A/B tests take time?
7. What's your strategy for balancing exploration (surfacing new/long-tail items) versus exploitation (serving known high-converting items)?
8. How would you detect and mitigate recommendation manipulation via bot-driven fake engagement?
9. If the entitlement database write-primary in US-East goes down, what's the blast radius, and how does the system behave?
10. How would you change this architecture if the catalog grew from 550K items to 50M items (e.g., full UGC marketplace)?

## 46. Ideal Answers

1. **Cold-start for new title launch**: No collaborative signal exists yet. Rely on content-based candidate generation (item-tower embedding from metadata/description alone, no interaction history needed) blended with editorial/merchandising-curated candidates and cross-title similarity (players who liked similar genre/franchise titles). Ranking falls back to a simpler model weighted toward content features and demographic priors rather than user-history features (which are null). Increase exploration weight temporarily (bandit-style) to gather interaction data fast, and shrink the real-time feature window's reliance since there's no session history yet. Monitor cold-start-specific conversion metrics separately from steady-state metrics.

2. **Offline-online metric mismatch after rollout**: First check for train/serve skew — feature store version mismatch, or a feature computed differently online vs. in the training pipeline (most common root cause). Check calibration on live traffic specifically (not just offline holdout) — offline eval may not reflect current live distribution due to lag between training data cutoff and rollout. Check for confounds: did the canary rollout coincide with a promo/pricing change, seasonal shift, or an unrelated infra issue (elevated latency causing timeout-fallback to popularity-only for a subset of traffic)? Use the exposure log to segment the regression by surface/region/segment to localize it before assuming the model itself is at fault.

3. **Filter-bubble/popularity-collapse mitigation**: Monitor candidate coverage/concentration metrics (Section 21) explicitly. Use diversity-aware re-ranking (MMR) rather than pure score-ranking. Correct for popularity bias in training (logQ correction in the two-tower softmax loss) so the retrieval model doesn't just learn "popular items get more interactions." Maintain an exploration budget (small % of slots reserved for under-exposed but relevant long-tail items) and track long-tail item exposure as a first-class metric, not just CTR/CVR.

4. **Regional ANN outage during peak sale**: Orchestrator's circuit breaker detects elevated ANN latency/errors within its timeout budget (~10ms) and falls back to popularity/trending candidate list (precomputed, cached, doesn't depend on ANN). Ranking still runs normally on the popularity-sourced candidates, so users still get *a* ranked list, just without the personalized-embedding-similarity candidates — degraded personalization, not a hard outage. Simultaneously, alerting fires (Section 23), on-call investigates; if regional, global LB can reduce traffic weight to the affected region temporarily while ANN service recovers (redeploy from S3 snapshot).

5. **Cross-title recommendations**: Requires a shared embedding space across titles — either a single global two-tower model trained on cross-title interaction data (if a user's Battlefield behavior should inform a FIFA recommendation, need shared user representation), or title-specific embeddings mapped into a common space via a learned projection. Also requires unified entitlement/catalog data across titles (already assumed via central data platform) and business-rule considerations (does merchandising want cross-title promotion, franchise-exclusive placement rules, etc.). Start with a simpler heuristic bridge (genre/franchise affinity rules) before investing in a full joint embedding model, and A/B test incrementally.

6. **Offline model evaluation before full rollout**: Combine offline metrics (Recall@K, NDCG, calibration) with counterfactual/off-policy evaluation — e.g., Inverse Propensity Scoring (IPS) or doubly-robust estimators using logged exposure + propensity (which the experimentation service already logs) to estimate what the new model's online CTR/CVR *would have been* had it served the logged traffic, without needing a live test first. This de-risks the canary's starting traffic allocation and gives an early go/no-go signal before spending real production traffic on the staged rollout.

7. **Exploration vs. exploitation**: Treat it as a contextual bandit problem layered on top of the ranking output — reserve a small percentage of impression slots (e.g., 5-10%) for exploration candidates selected via Thompson sampling or epsilon-greedy over under-explored items, log propensities for later off-policy evaluation, and tune the exploration rate per-surface (more exploration tolerance on lower-stakes surfaces like a "discover" rail, less on high-conversion-critical surfaces like the primary store front where revenue predictability matters more).

8. **Bot-driven engagement manipulation**: Layer defenses: (a) ingestion-time anomaly detection (event velocity per user/IP/device fingerprint far exceeding human-plausible rates), (b) downstream training-data filtering that down-weights or excludes flagged accounts before they influence model training, (c) monitoring for suspicious concentration of engagement on specific items (a sudden anomalous CTR spike on one obscure item is a red flag), (d) rate limiting (Section 27) catching high-frequency abuse at the edge before it even reaches the event pipeline, (e) periodic retroactive audits comparing engagement patterns against known bot-behavior signatures from other EA anti-cheat/fraud systems (shared threat intel across EA's trust & safety platform).

9. **Entitlement write-primary outage (US-East)**: Blast radius is scoped to new purchase writes and any read-path relying on the freshest ownership data being routed to the down primary — reads from other regions' async replicas continue serving (slightly stale, e.g., a purchase made seconds before the outage might not reflect immediately elsewhere), but the system fails closed on uncertainty (Section 41) — if freshness can't be verified, conservatively excludes ambiguous items rather than risk showing owned content as unowned/recommendable. Purchase flow itself (not this system, but upstream commerce) would need its own regional failover runbook; the recommendation engine's mitigation is entirely about defensive filtering during the uncertainty window, not attempting to fix the write path itself.

10. **Catalog growth to 50M items (full UGC marketplace)**: In-memory full-copy HNSW per replica stops working (50M x 512 bytes ≈ 25GB just for vectors, plus graph overhead — still theoretically fittable on a large-memory node but replication cost balloons). Shift to a sharded ANN architecture (partition by category/creator-tier/recency) with scatter-gather querying, or move to a disk-backed ANN system (e.g., DiskANN) to control memory cost. Candidate generation would also need a stronger pre-filtering stage (e.g., coarse category/tag-based filtering before ANN) to keep the search space tractable. Feature store and training data volume also scale roughly 100x, pushing toward more aggressive downsampling/negative-sampling strategies in training and likely a re-architecture of the offline feature pipeline for cost reasons (Section 44 storage/compute bottlenecks compound significantly).
