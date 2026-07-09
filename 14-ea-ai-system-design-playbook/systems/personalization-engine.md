# Personalization Engine

## 1. Problem Framing & Requirement Gathering

Build a per-player personalization system for an EA live-service game portfolio (e.g., a title like Apex-style shooter or a sports live-service game with a season pass economy). The system decides, in real time, **what content, UI layout, and store offers** to show each player — front-end menu tiles, battle-pass nudges, item-shop bundles, matchmaking-adjacent UI (e.g., "recommended mode"), and push/in-game messaging triggers.

Core question the interviewer wants probed: how do you turn player telemetry (session length, purchase history, skill rating, social graph, churn risk) into a low-latency decision service that (a) serves personalized ranked content within the game's UI render budget, (b) integrates with an experimentation platform so every personalization is A/B-testable and attributable, and (c) stays consistent with a feature store so training/serving skew doesn't wreck offline-trained model quality.

- **Business goal**: lift D7/D30 retention, ARPDAU (average revenue per daily active user), and offer conversion rate without degrading perceived game performance.
- **Not in scope**: matchmaking skill-based balancing (separate system), anti-cheat, chat moderation.
- **In scope**: content ranking, offer ranking, UI slot personalization, feature store, online inference, experimentation hooks, real-time context (current session state, current match result).

## 2. Functional Requirements

- FR1: Given `(player_id, surface, context)`, return a ranked list of content/offer candidates within budget, each with a score and an experiment variant tag.
- FR2: Support multiple **surfaces** (home screen, item shop, post-match screen, battle-pass nudge, push notification payload selection) from one platform, each with its own candidate pool and model.
- FR3: Ingest real-time context features (current session length, last match result, current in-game currency balance, party/social state) with <5s freshness.
- FR4: Ingest batch/offline features (7/30/90-day aggregates, churn score, LTV segment, purchase propensity) refreshed daily.
- FR5: Every personalization decision must be tied to an experiment assignment (control/treatment) from the experimentation platform, logged for offline analysis.
- FR6: Support cold-start players (new accounts, <24h old) via fallback rules/popularity models.
- FR7: Support manual override/business rules (e.g., forced promo during a live event) that supersede model output.
- FR8: Provide explainability metadata (top contributing features) for compliance and live-ops debugging.
- FR9: Support multi-title reuse — same platform serves FIFA-style sports title and a shooter title with different feature schemas and models.
- FR10: Support opt-out / regional consent (GDPR/CCPA) — personalization must degrade gracefully to non-personalized defaults per user consent state.

## 3. Non-Functional Requirements

| Dimension | Target |
|---|---|
| Latency | p50 < 30ms, p99 < 100ms for inference call (server-side, excludes client render) |
| Availability | 99.95% for inference path (≈4.4h downtime/year); degraded-mode fallback must be 99.99% |
| Throughput | Peak 250K QPS globally across surfaces (see capacity estimation) |
| Consistency | Feature store: eventual consistency for batch features (<24h staleness acceptable); real-time features <5s staleness; strong consistency NOT required (personalization is not a financial transaction) |
| Cost | Inference cost per request budget: <$0.000015 (1.5¢ per 1K requests) to keep personalization opex under 3% of incremental revenue lift |
| Durability | Feature/event logs durable for 90 days hot, 2 years cold (experimentation analysis, audits) |
| Freshness | Model retrain cadence daily for offer ranker, weekly for content ranker |

## 4. Clarifying Questions an Interviewer Would Expect

1. Is personalization scoped to one title or shared platform across EA's portfolio (multi-tenant)?
2. What's the render surface — native client UI (Unreal/Frostbite) calling a backend, or a web-based store frontend?
3. Is this synchronous request/response (blocking UI render) or can we pre-compute/pre-fetch personalization asynchronously on session start?
4. What's the acceptable staleness for "real-time" context — is a match-just-ended event usable within the same session, or only next session?
5. Do we need on-device fallback when the player is offline or backend is unreachable (console network blips)?
5b. Is there a hard latency budget imposed by the game engine's frame/UI loop, or is this decoupled (loading screen call)?
6. What's the experimentation platform — build vs. buy (EA internal vs. Optimizely-style)? Does it support server-side assignment at our required QPS?
7. Regulatory constraints — which regions require consent-gated personalization (EU/UK, California, and increasingly loot-box/gambling-adjacent regulations for offer personalization)?
8. Do offers have inventory/pricing constraints (can't show two conflicting bundles, must respect a daily "offer budget" per player to avoid fatigue)?
9. Is under-13 (COPPA) traffic present, requiring personalization to be disabled entirely for that cohort?
10. What's acceptable model interpretability requirement for live-ops/finance sign-off on offer pricing personalization?

## 5. Assumptions

1. Portfolio-wide MAU across supported titles: 60M; DAU: 12M (20% DAU/MAU, typical live-service).
2. Peak concurrent players: 2.5M (typical for EA-scale live service during peak evening hours across timezones).
3. Each active session triggers ~8 personalization surface calls (home load, shop open, post-match, 2x push-eligibility checks, battle-pass screen, mode-select, loading-screen tile).
4. Average session length: 35 minutes; average 2.3 sessions/day/DAU.
5. Candidate pool per surface: 50–300 items (offers, content tiles) pre-filtered by eligibility/inventory rules before ranking.
6. Real-time context features arrive via game server telemetry events (match end, purchase, level-up).
7. Feature store must serve both this system and other teams (matchmaking quality, churn-prevention CRM) — multi-consumer.
8. Experimentation platform is a shared EA internal service (assume built, not built here) exposing an assignment API at <10ms p99.
9. Models are gradient-boosted trees (LightGBM) for ranking, not deep neural nets, for the majority of surfaces — one exception: a two-tower embedding model for content/offer similarity (see Vector DB section).
10. GPU usage is limited to embedding model training/batch inference; online serving is CPU-bound (tree models).

## 6. Capacity Estimation

**QPS**
- Peak concurrent players: 2.5M
- Personalization calls per player per active minute ≈ 8 calls / 35 min session ≈ 0.23 calls/min/player
- Peak QPS = 2.5M × 0.23 / 60 ≈ **9,600 QPS sustained**, burst to **~25K QPS** during live-event spikes (2.6x burst factor typical for EA live-events / double-XP weekends)
- Design target: **250K QPS global capacity ceiling** (headroom for portfolio growth + burst + multi-surface fan-out, since one client "screen load" can fan out to 3-4 parallel surface calls)

**Storage**
- Online feature store (real-time + precomputed batch features): 60M players × ~2KB/player (150 features, mixed types) ≈ **120 GB** hot in-memory/KV store.
- Offline feature store (historical, training): 60M players × 90 days × 5KB/day-aggregate ≈ **27 TB**, columnar compressed (~6-8x) → **~3.5 TB** on cold columnar storage.
- Event log (experimentation exposure + telemetry raw): 12M DAU × 8 events/day × 1KB × 365 days ≈ **35 TB/year** raw, compressed ~10x → **3.5 TB/year**.

**Model size**
- LightGBM ranking models: ~500 trees × depth 8, ~2-5 MB serialized per surface model; ~10 surfaces × 3 titles = 30 models × 5MB = **150 MB** total in-memory footprint — trivial, fits in every serving pod's RAM.
- Two-tower embedding model (content/offer embeddings): 128-dim embeddings × 2M distinct content/offer items × 4 bytes = **1 GB** embedding table.

**Compute (serving)**
- CPU-bound tree inference: ~0.5ms per scoring call for 300 candidates (LightGBM batch-predict), single core.
- At 25K QPS burst, assume 4 vCPU per pod handles ~2,000 QPS (with batching + concurrency) → **13 pods** minimum, provision **40 pods** (3x headroom + AZ redundancy) at 4 vCPU/8GB each = 160 vCPU total.
- GPU: only for offline embedding training — 8x A100 nodes, ~6 hours/week retrain job, not part of online serving fleet.

**Feature store QPS**
- Each personalization call reads ~150 features across ~5 feature groups → feature store must sustain 250K × 5 = **1.25M read ops/sec** at peak — drives choice of in-memory KV (Redis/DynamoDB DAX-style) over a general RDBMS.

## 7. High-Level Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              Game Clients (console/PC)       │
                        │        Frostbite/Unreal UI, Store, HUD        │
                        └───────────────────┬───────────────────────────┘
                                            │ gRPC/HTTPS (surface request)
                                            ▼
                        ┌─────────────────────────────────────┐
                        │        API Gateway / Edge (CDN+LB)    │
                        │  authn, rate limit, region routing    │
                        └───────────────────┬───────────────────┘
                                            ▼
                ┌───────────────────────────────────────────────────┐
                │           Personalization Orchestrator Service      │
                │  - candidate retrieval (eligibility/inventory)      │
                │  - experiment assignment lookup                     │
                │  - feature fetch fan-out                            │
                │  - model invocation + business-rule override        │
                │  - response assembly + explainability tags           │
                └───┬───────────┬───────────┬───────────┬────────────┘
                    │           │           │           │
                    ▼           ▼           ▼           ▼
        ┌───────────────┐ ┌───────────┐ ┌───────────┐ ┌────────────────┐
        │ Online Feature │ │ Experiment │ │  Model     │ │ Candidate/     │
        │ Store (Redis/  │ │ Assignment │ │  Serving   │ │ Inventory      │
        │ KV, <5ms p99)  │ │ Service    │ │  (LightGBM │ │ Service (offer  │
        └───────┬────────┘ └─────┬──────┘ │  + 2-tower)│ │ catalog, rules) │
                │                │        └─────┬──────┘ └────────┬────────┘
                │                │              │                 │
                ▼                │              ▼                 │
        ┌───────────────┐        │      ┌───────────────┐         │
        │ Feature        │       │      │ Vector DB       │◄──────┘
        │ Ingestion       │       │      │ (ANN, content/ │
        │ (streaming +    │       │      │ offer embeds)   │
        │ batch)          │       │      └───────────────┘
        └───────┬─────────┘       │
                │                 ▼
                │         ┌───────────────┐
                │         │ Experimentation│
                │         │ Platform (EA   │
                │         │ shared svc)    │
                │         └───────────────┘
                ▼
        ┌─────────────────────────────────────────┐
        │  Streaming Bus (Kafka): telemetry, match- │
        │  end, purchase, session events             │
        └───────┬─────────────────────┬─────────────┘
                │                     │
                ▼                     ▼
        ┌───────────────┐    ┌─────────────────────┐
        │ Offline Feature│    │ Training Pipeline    │
        │ Store (columnar│    │ (batch, Spark +      │
        │ warehouse)     │───▶│ distributed GBDT/     │
        └───────────────┘    │ embedding training)    │
                              └──────────┬─────────────┘
                                         ▼
                              ┌─────────────────────┐
                              │ Model Registry +     │
                              │ CI/CD (canary/blue-  │
                              │ green rollout)        │
                              └─────────────────────┘
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| API Gateway/Edge | AuthN, TLS termination, regional routing, rate limiting | HTTPS/gRPC ingress | Horizontal, per-region, autoscaled on RPS |
| Personalization Orchestrator | Fan-out to feature store/models/experiment service, assemble response | Internal gRPC | Stateless pods, HPA on CPU+QPS |
| Online Feature Store | Serve low-latency feature vectors per player | Key-value `get(player_id, feature_group)` | Sharded by `player_id` hash, read replicas |
| Feature Ingestion (streaming) | Consume Kafka events, compute/update real-time features | Kafka consumer group | Partition-parallel consumers |
| Feature Ingestion (batch) | Nightly Spark jobs compute aggregate features | Batch job (Airflow DAG) | Spark cluster autoscale |
| Model Serving | Host LightGBM + two-tower models, batched inference | gRPC `predict(features[]) -> scores[]` | Stateless pods, HPA on QPS/latency |
| Vector DB | ANN lookup for content/offer embedding similarity | `query(embedding) -> top_k ids` | Sharded index, replica-per-AZ |
| Candidate/Inventory Service | Eligibility filtering, offer inventory/pricing rules | REST/gRPC `getCandidates(surface, player_ctx)` | Stateless, cache-heavy |
| Experimentation Platform | Deterministic variant assignment, exposure logging | `assign(player_id, experiment_id) -> variant` | Shared EA service, out of this system's scope |
| Model Registry / CI-CD | Versioned model artifacts, rollout orchestration | Internal API + CLI | N/A (control plane) |
| Training Pipeline | Offline training (GBDT, two-tower embeddings) | Airflow DAG + Spark/PyTorch DDP | GPU cluster for embeddings, CPU cluster for GBDT |
| Offline Feature Store | Historical point-in-time feature snapshots for training | SQL/columnar query engine | Partitioned by date + player shard |

## 9. API Design

**Primary inference endpoint**

```
POST /v2/personalize/{surface}
Host: personalization.ea-internal.net
Headers: Authorization: Bearer <service-jwt>, X-Player-Session-Id, X-Region

Request:
{
  "player_id": "p_9f21ac...",
  "surface": "item_shop",              // enum: home, item_shop, post_match, battle_pass, push, loading_screen
  "title_id": "fifa25",
  "context": {
    "session_id": "s_88af...",
    "last_match_result": "win",
    "current_currency_balance": 1200,
    "party_size": 2,
    "client_platform": "ps5",
    "consent_state": "personalization_allowed"
  },
  "candidate_pool_hint": ["bundle_412", "bundle_517", "..."],  // optional pre-filter
  "max_results": 10
}

Response: 200 OK
{
  "request_id": "req_abc123",
  "surface": "item_shop",
  "results": [
    {
      "item_id": "bundle_412",
      "score": 0.874,
      "rank": 1,
      "experiment": { "experiment_id": "exp_offer_rank_v3", "variant": "treatment_b" },
      "explain": { "top_features": ["purchase_propensity_7d", "session_win_streak"] }
    }
  ],
  "fallback_used": false,
  "model_version": "offer_ranker_v14.2",
  "ttl_ms": 300000
}
```

**Versioning**: URL path versioned (`/v2/`), backward-compatible field additions only within a major version; breaking changes bump to `/v3/` with 90-day dual-run deprecation window.

| Endpoint | Method | Purpose |
|---|---|---|
| `/v2/personalize/{surface}` | POST | Real-time ranked personalization result |
| `/v2/personalize/batch` | POST | Pre-compute personalization for a batch of players (push campaigns) |
| `/v2/feature/{player_id}` | GET | Debug/introspection: raw feature vector (internal only, RBAC-gated) |
| `/v2/health` | GET | Liveness/readiness probe |
| `/v2/models/{surface}/version` | GET | Currently serving model version (ops/debug) |

## 10. Database Design

| Store | Type | Why | Partition/Shard Key |
|---|---|---|---|
| Online Feature Store | Redis Cluster / DynamoDB (KV) | Sub-5ms reads at 1.25M ops/sec; simple key-value access pattern, no joins needed at serving time | `player_id` hash |
| Offline Feature Store | Columnar warehouse (Parquet on S3 + Trino/Spark, or Snowflake) | Point-in-time correctness queries, cheap compressed storage, analytical scans for training set generation | Partitioned by `event_date`, clustered by `player_id` |
| Candidate/Offer Catalog | Relational (Postgres) | Strong consistency for pricing/inventory rules, transactional updates from live-ops tooling | Sharded by `title_id`, replicated read-heavy |
| Experiment Exposure Log | Append-only columnar (Kafka → S3/Iceberg) | High write volume, analytical read pattern for offline experiment analysis | Partitioned by `experiment_id` + date |
| Vector DB (embeddings) | Specialized ANN store (see Section 16) | Similarity search not expressible efficiently in KV/relational | Sharded by `title_id` + embedding cluster |

Schema sketch — online feature record:
```
Key: player_id
Value (hash map):
  rt:last_match_result: "win"
  rt:session_len_min: 22
  rt:currency_balance: 1200
  batch:churn_score_7d: 0.12
  batch:purchase_propensity_30d: 0.68
  batch:ltv_segment: "whale_tier2"
  updated_at: 1735689600
```

Offline feature store schema (columnar, point-in-time table):
```
player_id | feature_date | churn_score | purchase_propensity | ltv_segment | session_count_7d | ...
```

## 11. Caching

- **What's cached**: (a) candidate pool per surface per title (TTL 5 min, invalidated on inventory change), (b) model prediction for identical `(player_id, surface, context_hash)` within short TTL (30-60s) to absorb rapid repeat calls (e.g., UI re-render), (c) experiment assignment result (cached client-session-scoped, since assignment is deterministic per player+experiment for the session).
- **Strategy**: cache-aside for feature reads (orchestrator checks Redis, falls back to feature store's own persistence layer on miss — though feature store IS the cache in this design, so this mainly applies to a secondary L1 in-process cache in the orchestrator for hot players).
- **Invalidation**: event-driven — a `purchase_completed` or `inventory_updated` Kafka event invalidates the relevant candidate-pool cache key immediately; feature cache entries use short TTL rather than explicit invalidation (simpler, acceptable staleness).
- **Write-through** used for the offer inventory cache specifically — live-ops price/availability changes must propagate immediately (no stale pricing shown), so writes to Postgres catalog also write through to the Redis-backed candidate cache synchronously.
- **Negative caching**: cold-start players with no features cache a "use fallback" flag for 60s to avoid hammering feature store with lookups that will miss anyway.

## 12. Queues & Async Processing

- **What's queued**: (1) raw telemetry events (match end, purchase, level-up) → Kafka, (2) experiment exposure logs, (3) batch personalization requests (push campaign pre-computation), (4) model training trigger jobs, (5) feature-store write-back from streaming aggregators.
- **Delivery semantics**: telemetry ingestion is **at-least-once** (Kafka default) — feature aggregation consumers are idempotent (upsert by `player_id + feature_key`, last-write-wins by event timestamp) to tolerate duplicate delivery.
- **Exactly-once** needed for revenue-impacting events only: purchase-confirmation events feeding LTV/propensity features use Kafka transactional producer + idempotent consumer with dedup table keyed on `transaction_id` (24h dedup window).
- **Dead-letter handling**: malformed/unparseable events route to a DLQ topic (`telemetry.dlq`) after 3 retry attempts with exponential backoff; DLQ monitored, alerts if DLQ rate >0.1% of topic volume; manual replay tooling for recoverable schema-mismatch batches.
- **Batch push personalization**: queued as jobs in a work queue (e.g., SQS-style), processed by a worker pool that calls the same orchestrator API, results written to a campaign delivery table, rate-limited to avoid self-inflicted load spikes on the online path.

## 13. Streaming & Event-Driven Architecture

**Kafka topics**

| Topic | Producer | Consumer(s) | Schema (key fields) |
|---|---|---|---|
| `telemetry.match_end` | Game servers | Feature ingestion, offline warehouse sink | `player_id, match_id, result, duration, timestamp` |
| `telemetry.purchase` | Store/billing service | Feature ingestion (propensity/LTV), finance sink | `player_id, sku, amount, currency, transaction_id` |
| `telemetry.session` | Client/session service | Real-time feature updater | `player_id, session_id, event_type(start/end), timestamp` |
| `personalization.exposure` | Orchestrator | Experimentation analysis pipeline, offline warehouse | `player_id, surface, experiment_id, variant, model_version, request_id` |
| `catalog.inventory_updated` | Live-ops tooling | Candidate cache invalidator | `title_id, item_id, change_type, timestamp` |

- **Consumer groups**: feature-ingestion service runs as a consumer group with partition count = Kafka topic partitions (128 partitions on `telemetry.*` topics, keyed by `player_id` hash for ordering-per-player); experimentation-analysis consumer group is independent, can lag without affecting serving path (analytical, not serving-critical).
- **Schema management**: Avro/Protobuf with a schema registry, backward-compatible evolution enforced (no field removal without deprecation window), consumers validate schema version on read.
- **Ordering guarantee**: per-player ordering preserved via `player_id`-based partition key; cross-player ordering not required.

## 14. Model Serving

- **Framework**: LightGBM models served via a lightweight custom gRPC service (or Triton Inference Server with FIL backend for tree models) — CPU-only, no GPU needed for tree inference at this scale.
- **Two-tower embedding model**: served via Triton or TorchServe for the item-tower forward pass (query embedding computed at request time for the player-tower side); item embeddings precomputed offline and stored in the vector DB (batch inference, not online).
- **Batching**: dynamic request batching at the orchestrator level — up to 300 candidates per player scored in a single LightGBM `predict` call (single-request-multi-candidate batching, not cross-request batching, since each request has a distinct candidate set); Triton dynamic batching used for the two-tower query-embedding path (cross-request batching viable there since input shape is uniform).
- **Multi-model**: one model per `(title_id, surface)` pair — ~30 active models; models loaded into a shared serving fleet, routed by request metadata; model registry provides versioned artifacts, hot-swappable without pod restart (in-memory model reload on registry poll every 60s or push-based reload via control-plane RPC).
- **Hardware**: CPU serving fleet (per Section 6: ~40 pods × 4 vCPU); GPU fleet reserved for offline embedding training/batch item-embedding refresh (runs weekly, not part of online path).
- **Latency**: LightGBM inference for 300 candidates ≈ 0.5-1ms; total serving-path overhead budget must include feature fetch + model call + response assembly (see Section 43 latency budget breakdown).

## 15. Feature Store

- **Online store**: Redis Cluster (or DynamoDB with DAX), holds latest value per feature per player, sub-5ms p99 reads, updated by streaming ingestion (real-time features) and nightly batch job (aggregate features written down from warehouse).
- **Offline store**: columnar warehouse (Parquet/Iceberg on S3, queried via Spark/Trino), holds full feature history for training set generation.
- **Point-in-time correctness**: training set generation uses **time-travel joins** — for a training example labeled at `event_time T`, features are joined as they existed at `T - epsilon`, never using future data. Implemented via Iceberg/Delta time-travel snapshots or an explicit `feature_valid_from/valid_to` interval table, avoiding label leakage (critical bug class: using post-event aggregates like "did player churn this week" computed after the label window).
- **Feature registry**: central definition of every feature (name, owner, freshness SLA, computation logic reference) shared across consuming teams (personalization, churn-prevention, matchmaking-adjacent CRM) — prevents feature drift/duplication across teams.
- **Write path**: streaming ingestion writes real-time features directly to online store; batch pipeline writes aggregate features to offline store first, then syncs the latest snapshot to online store nightly (batch features are at most 24h stale by design, acceptable per NFR).
- **Backfill**: historical feature backfill jobs (for new feature launches) run against offline store only, never touch online store directly, to avoid inconsistent partial-backfill states being served live.

## 16. Vector Database

**Applicable** — used for content/offer similarity and candidate generation (the two-tower model's retrieval stage: "find offers similar to what this player tends to engage with").

- **Index type**: HNSW (Hierarchical Navigable Small World) — chosen over IVF-PQ because candidate pool per title is modest (2M items), HNSW gives better recall at low latency without needing the compression tradeoffs IVF-PQ makes at billion-scale; recall@10 target 0.95+ at <10ms query time.
- **Embedding dimension**: 128-dim (balances retrieval quality vs. index memory footprint — 2M items × 128 × 4 bytes = 1GB, fits in-memory per shard).
- **Sharding**: one index per `title_id` (item catalogs don't overlap across titles), replicated across AZs for availability, rebuilt on a rolling schedule (weekly, matching embedding retrain cadence) with blue/green index swap to avoid query disruption during rebuild.
- **Update strategy**: item embeddings are batch-refreshed weekly (not real-time) since catalog/content churn is slow relative to player behavior churn; new items get a cold-start embedding (category-average vector) until the next batch refresh incorporates real interaction data.

## 17. Embedding Pipelines

**Applicable** — two-tower model produces player embeddings (query tower, computed online per request from real-time+batch features) and item/offer embeddings (item tower, computed offline in batch).

- **Player tower**: small MLP (2-3 layers, ~64K params) taking feature-store vector as input, run online at request time (cheap enough for CPU, <1ms).
- **Item tower**: MLP over item metadata (category, price tier, historical engagement stats), run offline in batch, output written to vector DB.
- **Training**: contrastive loss (in-batch negative sampling) on (player, engaged-item) pairs from historical exposure+engagement logs; retrained weekly alongside the item embedding refresh.
- **Serving-time flow**: request → compute player embedding (online) → ANN query against vector DB (retrieve top-K candidate items) → union with rule-based candidate pool → score all with LightGBM ranker → return top-N.
- **Why not skip embeddings entirely**: pure rule-based/collaborative-filtering candidate generation misses long-tail personalization (new bundle similar to past purchases but never co-purchased by enough players for classic CF); embeddings generalize better for cold-start items.

## 18. Inference Pipelines (Request Lifecycle End-to-End)

```
Client (game UI)
   │  1. POST /v2/personalize/item_shop  (player_id, context)
   ▼
API Gateway                         [~2ms: TLS, authn, rate-limit check]
   │
   ▼
Personalization Orchestrator
   │  2. Parallel fan-out:
   │     ┌─────────────────────────────────────────────┐
   │     │ a) Feature Store GET(player_id)   [~3ms]      │
   │     │ b) Experiment Assignment(player,exp) [~5ms]   │
   │     │ c) Candidate Service getCandidates() [~4ms]    │
   │     └─────────────────────────────────────────────┘
   │  3. If two-tower retrieval enabled for surface:
   │       compute player embedding [~1ms] → ANN query vector DB [~8ms]
   │       → merge with rule-based candidates
   │  4. Model Serving: batch-score all candidates (LightGBM) [~1ms]
   │  5. Apply business-rule overrides (forced promos, inventory caps) [~1ms]
   │  6. Assemble response + explainability tags + experiment tag [~1ms]
   │  7. Async: emit personalization.exposure event to Kafka (fire-and-forget)
   ▼
Response to client                  [total server-side: ~25-30ms p50]
```

- **Failure handling inline**: if feature store times out (>10ms), orchestrator proceeds with cached/default feature vector (degraded personalization, not a hard failure); if model serving fails, fall back to popularity-ranked default list; if experimentation service times out, default to control variant (never block the response on experimentation assignment).
- **Idempotency**: `request_id` allows client retry without double-counting exposure events (dedup on exposure log write).

## 19. Training Pipelines

- **Data prep**: offline feature store (point-in-time joined) + label source (did player engage/purchase within N hours of exposure, from `personalization.exposure` joined with `telemetry.purchase`/engagement events) → training set materialized as Parquet, versioned per training run.
- **Label definition**: offer ranker label = binary purchase-within-24h after exposure; content ranker label = engagement (click/dwell-time weighted) within session.
- **Orchestration**: Airflow DAG — (1) validate feature freshness, (2) materialize training set with point-in-time join, (3) train LightGBM via distributed histogram-based training (LightGBM's native distributed mode across a CPU cluster, ~16 workers) for large titles, (4) train two-tower embeddings via PyTorch DDP across 8x A100 GPUs (weekly cadence, larger compute job), (5) offline eval (AUC, NDCG@10, calibration), (6) push to model registry if eval gates pass, (7) trigger canary rollout.
- **Distributed training**: GBDT training uses LightGBM's `data-parallel` mode (histogram aggregation across workers) for titles with >50M training rows; two-tower model uses PyTorch DDP with gradient all-reduce across GPU nodes, mixed precision (fp16) to cut training time ~40%.
- **Reproducibility**: every training run tagged with feature-store snapshot version + code commit hash + data date range, stored in model registry metadata.

## 20. Retraining Strategy

- **Cadence**: offer ranker retrained **daily** (offer catalog and pricing change frequently, propensity shifts fast around live events); content ranker retrained **weekly**; two-tower embeddings retrained **weekly**.
- **Trigger-based retraining** (in addition to cadence): triggered early if drift detection (Section 21) crosses threshold, or if a live-event/major content patch changes the candidate catalog substantially (>20% new items), or if a business stakeholder requests refresh ahead of a promotional campaign.
- **Champion/challenger**: every retrain produces a challenger model evaluated offline against current champion on a held-out replay set before being eligible for canary rollout — no automatic promotion without eval-gate pass (min NDCG@10 improvement threshold, no regression >1% on any surface segment).

## 21. Drift Detection

| Drift Type | Metric | Threshold / Action |
|---|---|---|
| Data drift (feature distribution) | PSI (Population Stability Index) per feature, computed daily vs. training baseline | PSI > 0.2 on any top-20 important feature → alert + flag for early retrain |
| Concept drift (label relationship change) | Rolling 7-day model AUC/NDCG on live traffic (via exposure+outcome join) vs. training-time eval metric | Drop >5% relative → alert, trigger challenger evaluation |
| Prediction drift | KL divergence of score distribution (daily) vs. prior week | KL > 0.1 → investigate, check for catalog/candidate-pool changes |
| Candidate catalog drift | % new/removed items vs. prior week | >20% change → force embedding refresh out of normal cadence |
| Calibration drift | Predicted vs. actual conversion rate (reliability curve) per score bucket | Brier score degradation >10% → alert |

- Drift computation runs as a nightly batch job over the offline feature store + exposure logs, results pushed to the monitoring dashboard and alerting pipeline (Section 22/23).

## 22. Monitoring

- **Infra**: pod CPU/mem utilization, request latency histograms (p50/p90/p99), error rate, feature-store hit/miss rate, Kafka consumer lag, vector DB query latency, GPU utilization during training jobs.
- **Model quality**: offline eval metrics per training run (AUC, NDCG@10, calibration/Brier), online proxy metrics (click-through on personalized slot vs. control), drift metrics (Section 21).
- **Business metrics**: D1/D7/D30 retention lift (experiment-attributed), ARPDAU lift, offer conversion rate, personalization coverage (% of requests served personalized vs. fallback), experiment sample-ratio-mismatch checks (SRM) to catch broken randomization.
- **Dashboards**: per-surface, per-title breakdowns; live-ops dashboard for real-time offer performance during promotional events.

## 23. Alerting

| Condition | Threshold | Route |
|---|---|---|
| p99 latency | >150ms sustained 5min | Page on-call SRE (PagerDuty) |
| Error rate | >1% over 5min | Page on-call SRE |
| Fallback rate (degraded serving) | >10% of traffic for 10min | Page on-call + notify ML team |
| Feature store staleness | Real-time features >30s stale | Page data-eng on-call |
| Kafka consumer lag | >60s lag on `telemetry.*` topics | Page data-eng on-call |
| Model AUC drop | >5% relative vs. baseline | Notify ML team (ticket, not page) |
| Drift PSI threshold breach | PSI > 0.2 | Notify ML team (ticket) |
| SRM (sample ratio mismatch) in experiment | p < 0.001 chi-sq test | Notify experimentation platform + ML team |
| DLQ volume | >0.1% of topic volume | Notify data-eng |
| Cost anomaly | Daily inference spend >20% above 7-day average | Notify eng lead + finance partner |

- **On-call routing**: infra/latency alerts → SRE on-call (immediate page); model-quality/drift alerts → ML platform on-call (business-hours ticket unless combined with a business-metric regression, which pages).

## 24. Logging

- **Structured logging**: JSON logs with `request_id`, `player_id` (hashed/tokenized, not raw), `surface`, `model_version`, `experiment_id/variant`, `latency_ms`, `fallback_used` — every orchestrator request emits one structured log line.
- **PII handling**: `player_id` stored as a hashed/pseudonymous ID in logs and exposure events (raw account ID never logged); logs containing behavioral context (e.g., purchase amounts) classified as sensitive, access-restricted (RBAC), and excluded from general-purpose log search tools available to all engineers — routed to a restricted log store.
- **Retention**: hot logs (searchable, e.g., ELK/Datadog) retained 30 days; cold archival logs (S3, compliance/audit) retained 2 years for experiment reproducibility and regulatory audit; exposure logs feeding experimentation analysis retained per experimentation platform's own policy (typically 1 year post-experiment-close).
- **Right-to-erasure**: player deletion requests propagate a tombstone event that purges hashed-ID-linked records from hot stores within 30 days and cold archives within 90 days, per GDPR/CCPA obligations.

## 25. Security

- **Threat model specific to this system**:
  - Adversarial manipulation of context features to unlock better offers (e.g., spoofed client sending fake `currency_balance` or `session` context) → mitigate by only trusting server-authoritative telemetry (game server-emitted events), never client-supplied context fields for anything pricing-relevant; client-supplied context is limited to non-sensitive UI-state hints.
  - Feature store as a high-value data-exfiltration target (aggregated behavioral/purchase profiles across 60M players) → encrypt at rest, strict RBAC, network-isolated (private subnet, no public ingress), audit-logged access.
  - Model extraction/inversion (repeated queries to infer training data or reverse-engineer pricing model) → rate-limit debug/introspection endpoints (`/v2/feature/{player_id}`), require elevated internal auth scope, monitor for anomalous query patterns.
  - Offer/pricing manipulation via replayed experiment-assignment requests → deterministic hashing assignment (not client-controlled), signed assignment tokens.
- **Encryption**: TLS 1.3 in transit for all service-to-service and client-to-edge traffic; AES-256 at rest for feature store, offline warehouse, and event logs.
- **Data minimization**: only features with a documented business justification are ingested into the online store (feature registry enforces owner sign-off).

## 26. Authentication

- **Service-to-service**: mTLS within the internal mesh (e.g., Istio/Linkerd) between orchestrator, feature store, model serving, candidate service; short-lived service JWTs issued by an internal identity provider for cross-cluster/cross-region calls, validated at each hop.
- **End-user (client-to-edge)**: player's existing EA account session token (OAuth2-style bearer token issued at game login) validated at the API gateway; gateway attaches a verified, signed `player_id` claim to downstream requests so internal services never trust client-asserted identity directly.
- **Internal debug endpoints**: require an elevated internal-employee SSO-backed token with RBAC scope `personalization:debug`, separate from the player-session auth path.

## 27. Rate Limiting

- **Algorithm**: token bucket per `player_id` at the API gateway (e.g., 20 requests/10s burst, refill 2/s) — generous enough for legitimate UI fan-out (8 surfaces/session) but blocks scripted abuse/scraping of personalized offers.
- **Per-tenant (per-title) limits**: separate token buckets per `title_id` to prevent one title's traffic spike from starving another title's capacity on shared infrastructure — sized proportional to each title's provisioned QPS share.
- **Internal debug endpoint limits**: strict per-employee-token limit (e.g., 60 req/min) since these bypass normal caching and hit feature store directly.
- **Backpressure**: when downstream (feature store/model serving) approaches saturation, gateway applies adaptive rate limiting (reduce token refill rate) rather than hard-failing, prioritizing session-critical surfaces (home, post-match) over lower-priority ones (push pre-computation batch calls) via priority queues.

## 28. Autoscaling

- **Orchestrator/Model Serving pods**: HPA on custom metric `requests_per_second` + CPU utilization (target 60% CPU), scale range 13-60 pods per region (per Section 6 capacity math), scale-up stabilization window 30s (fast reaction to live-event spikes), scale-down stabilization window 5min (avoid flapping).
- **Feature store**: Redis Cluster scaled via read-replica addition (VPA-style vertical headroom on primary shards) triggered by sustained >70% memory utilization or read-latency p99 >8ms.
- **Vector DB**: scaled by index shard replication, KEDA-driven scaling on query-queue depth for the ANN service if using a queue-fronted architecture.
- **Kafka consumers (feature ingestion)**: KEDA scaler on consumer-group lag — scale out consumer pods when lag exceeds 10K messages per partition.
- **Batch training jobs**: no autoscaling — provisioned as scheduled Spark/GPU clusters (fixed size per job), spun up/down by the orchestration DAG (Airflow triggers cluster creation, not a standing autoscaled fleet), since training runs on a schedule not a live-traffic signal.

## 29. Cost Optimization

- **Spot/preemptible instances**: training pipeline GPU nodes (weekly embedding retrain, not latency-sensitive) run on spot instances with checkpointing every N steps to tolerate preemption — ~60-70% cost reduction vs. on-demand.
- **Caching**: aggressive candidate-pool and prediction caching (Section 11) cuts redundant model-serving calls for rapid repeat UI queries — estimated 15-20% reduction in effective serving QPS.
- **Model distillation/compression**: LightGBM models already lightweight; two-tower embedding model uses a compact 128-dim embedding (vs. 512+) trading a small recall loss for 4x smaller vector DB memory footprint and faster ANN queries.
- **Batching**: dynamic batching at Triton for the two-tower query path improves GPU/CPU utilization per request, reducing serving fleet size needed for the same throughput.
- **Right-sizing serving fleet regionally**: don't provision peak-global capacity in every region — route by follow-the-sun traffic patterns, scale down overnight-region fleets aggressively (autoscaling floor of 3 pods vs. 13 baseline during regional off-peak).
- **Tiered storage**: hot online features in Redis (expensive/GB) limited to features with genuine <5s freshness need; everything else lives in cheaper offline columnar storage, synced down only what's needed.
- **Feature reuse across teams**: shared feature store avoids duplicate feature engineering/compute pipelines across churn-prevention, matchmaking, and personalization teams — compute cost amortized.

## 30. Disaster Recovery

- **RTO**: 15 minutes for the online serving path (personalization) — acceptable because fallback mode (non-personalized default content) can serve during recovery; **RPO**: 5 minutes for feature store (acceptable loss of most-recent real-time feature updates, batch features unaffected since offline store is source of truth).
- **Backup strategy**: offline feature store (columnar warehouse) backed up via standard warehouse snapshotting (daily, cross-region replicated); online feature store (Redis) uses cross-AZ replica promotion for AZ failure, and periodic RDB/AOF snapshots to object storage for full-cluster-loss recovery; model registry artifacts stored in versioned, cross-region-replicated object storage (immutable, never deleted, only deprecated).
- **Runbook**: on regional serving-path failure, DNS/traffic-manager fails over to nearest healthy region within RTO target; degraded mode (rule-based fallback content) auto-activates if feature store or model serving is unreachable, ensuring the game UI never blocks on personalization failure (fail-open, not fail-closed, since this is not a security-critical path).

## 31. Multi-Region Deployment

**Active-active** across 3 regions (US-East, EU-West, APAC) — chosen because personalization is latency-sensitive and player base is globally distributed; no single-region failover latency penalty acceptable for a real-time UI-blocking call.

```
        ┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
        │   US-East Region    │     │   EU-West Region    │     │   APAC Region       │
        │  Full serving stack │     │  Full serving stack  │     │  Full serving stack │
        │  (orchestrator,     │     │  (orchestrator,      │     │  (orchestrator,     │
        │   feature store,    │     │   feature store,     │     │   feature store,    │
        │   model serving,    │     │   model serving,     │     │   model serving,    │
        │   vector DB replica)│     │   vector DB replica)  │     │   vector DB replica) │
        └─────────┬──────────┘     └─────────┬──────────┘     └─────────┬──────────┘
                  │                          │                          │
                  └──────────────┬───────────┴───────────┬──────────────┘
                                 ▼                        ▼
                     ┌───────────────────────────────────────────┐
                     │  Global Feature Data Plane (async replication│
                     │  of offline feature store + model registry) │
                     └───────────────────────────────────────────┘
```

- **Latency routing**: GeoDNS/Anycast + client-side region pinning at session start (player routed to nearest region based on game-server region assignment, kept sticky for session duration to avoid feature-store cross-region reads mid-session).
- **Data replication**: offline feature store and model registry replicate cross-region asynchronously (batch sync, acceptable staleness since these aren't real-time paths); online feature store (Redis) is **regional, not globally replicated** — real-time features are regionally scoped since a player's session is pinned to one region anyway, avoiding cross-region write latency/conflict entirely.
- **Experiment consistency**: experiment assignment must be globally consistent (a player shouldn't see different variants if they roam regions) — assignment determinism achieved via consistent hashing on `player_id + experiment_id` (stateless, computed identically regardless of region, no cross-region state needed).

## 32. Blue/Green Deployment

- Applies to **model rollout** and **service code deployment** separately.
- **Model blue/green**: two model-version slots (blue=current champion, green=challenger) loaded simultaneously in the serving fleet; traffic router (orchestrator-level feature flag) shifts a defined % from blue to green; full cutover only after canary gate (Section 33) passes; instant rollback = flip router back to blue (no redeploy needed since both versions are already warm in memory).
- **Service code blue/green**: new orchestrator version deployed to a parallel pod fleet (green), smoke-tested against synthetic traffic, then load balancer cutover; old fleet (blue) kept running for a rollback window (30 min) before decommission.

## 33. Canary Deployment

- **Traffic-split strategy**: new model version starts at 1% of traffic (randomly assigned via consistent hashing, stratified to ensure representative player segments — not just one region/cohort), ramps 1% → 5% → 25% → 100% over 24-48 hours, gated at each step.
- **Health-check gates specific to this system**: (1) latency p99 within 10% of baseline, (2) error rate <0.5%, (3) fallback-rate not elevated, (4) short-window online proxy metric (click-through/engagement on personalized slot) not regressed >2% vs. control, (5) no SRM alert from experimentation platform, (6) offer-conversion rate not regressed (critical business gate — a regression here blocks progression regardless of technical health).
- **Automated gate evaluation**: canary analysis job runs every 2 hours during ramp, auto-halts progression (does not auto-rollback) on gate failure, pages ML on-call for manual decision if a business-metric gate fails (avoids autopilot rollback on noisy day-1 data).

## 34. Rollback Strategy

- **Automated triggers**: p99 latency regression >20%, error rate >2%, fallback rate spike >15%, or a hard business-metric gate breach sustained over 3 consecutive evaluation windows → automatic rollback to last-known-good model version (instant, since blue/green keeps prior version warm).
- **Rollback mechanics**: router-level flip back to prior model version slot (no redeploy); for service-code rollback, traffic shifted back to blue fleet via load balancer weight change; both complete in <60 seconds.
- **Post-rollback**: incident auto-filed, exposure logs from the rolled-back window flagged/excluded from experiment analysis to avoid contaminating results with a known-bad model period.

## 35. Observability

- **Tracing**: distributed tracing (OpenTelemetry) across the full request lifecycle (gateway → orchestrator → feature store/experiment/candidate/model-serving fan-out) with `request_id` propagated as trace ID — enables pinpointing which fan-out call dominates p99 latency for a given slow request.
- **Metrics**: Prometheus-style metrics per component (latency histograms, QPS, error rate, cache hit rate, drift metrics) scraped and visualized in Grafana; business metrics (conversion, retention lift) in a separate analytics dashboard fed by the offline warehouse.
- **Logs**: structured logs (Section 24) correlated via `request_id` and `trace_id`, queryable in a centralized log platform (e.g., Datadog/ELK), linked directly from trace spans to corresponding log lines for one-click debugging.
- **Correlation in practice**: an on-call engineer investigating a p99 latency alert opens the trace for a slow sample request, sees the vector-DB ANN query span took 40ms (vs. 8ms budget), pivots to vector-DB-specific metrics dashboard, finds a shard imbalance, cross-references logs for that shard's recent index-rebuild event — full pillar correlation in under 5 minutes.

## 36. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: personalization-orchestrator
  labels: { app: personalization-orchestrator }
spec:
  replicas: 13
  selector:
    matchLabels: { app: personalization-orchestrator }
  template:
    metadata:
      labels: { app: personalization-orchestrator }
    spec:
      containers:
        - name: orchestrator
          image: registry.ea.internal/personalization/orchestrator:v14.2
          resources:
            requests: { cpu: "2", memory: "4Gi" }
            limits: { cpu: "4", memory: "8Gi" }
          ports: [{ containerPort: 8080 }]
          readinessProbe:
            httpGet: { path: /v2/health, port: 8080 }
            periodSeconds: 5
          env:
            - name: FEATURE_STORE_ENDPOINT
              value: redis-feature-store.svc.cluster.local:6379
---
apiVersion: v1
kind: Service
metadata: { name: personalization-orchestrator-svc }
spec:
  selector: { app: personalization-orchestrator }
  ports: [{ port: 443, targetPort: 8080 }]
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: orchestrator-hpa }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: personalization-orchestrator }
  minReplicas: 13
  maxReplicas: 60
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 60 } }
    - type: Pods
      pods:
        metric: { name: requests_per_second }
        target: { type: AverageValue, averageValue: "2000" }
```

## 37. Terraform Infrastructure

```hcl
resource "aws_elasticache_replication_group" "feature_store" {
  replication_group_id       = "personalization-online-features"
  description                 = "Online feature store for personalization engine"
  node_type                   = "cache.r6g.2xlarge"
  num_node_groups              = 8
  replicas_per_node_group       = 2
  engine                       = "redis"
  engine_version                = "7.1"
  automatic_failover_enabled     = true
  multi_az_enabled               = true
  at_rest_encryption_enabled       = true
  transit_encryption_enabled        = true
}

resource "aws_msk_cluster" "telemetry_bus" {
  cluster_name           = "personalization-telemetry"
  kafka_version           = "3.6.0"
  number_of_broker_nodes    = 9
  broker_node_group_info {
    instance_type   = "kafka.m5.2xlarge"
    client_subnets   = var.private_subnet_ids
    storage_info { ebs_storage_info { volume_size = 1000 } }
  }
  encryption_info {
    encryption_in_transit { client_broker = "TLS", in_cluster = true }
  }
}

resource "aws_eks_node_group" "model_serving_cpu" {
  cluster_name    = var.eks_cluster_name
  node_group_name  = "model-serving-cpu"
  instance_types    = ["c6i.xlarge"]
  scaling_config {
    min_size     = 13
    max_size     = 60
    desired_size  = 20
  }
  labels = { workload = "personalization-model-serving" }
}

resource "aws_instance" "embedding_training_gpu" {
  count         = 8
  instance_type  = "p4d.24xlarge"
  ami            = var.gpu_training_ami
  spot_price      = "12.00"
  tags = { Name = "embedding-training-node", workload = "batch-training" }
}
```

## 38. Why This Architecture

- **Separation of orchestrator/feature-store/model-serving** allows each to scale independently against its own bottleneck (feature-store is read-QPS-bound, model-serving is CPU-bound, orchestrator is fan-out/IO-bound) rather than one monolith scaling on the max of all dimensions.
- **CPU-based tree models as the default**, with GPU reserved for embeddings only, matches the actual latency/cost profile: tabular ranking with <200 features doesn't need a neural net to hit target accuracy, and GBDTs give interpretability (explainability requirement, FR8) that deep models don't provide as cheaply.
- **Fail-open design** (fallback to popularity/rules on any dependency failure) is essential for a UI-blocking call in a live-service game — a broken personalization system must never break the storefront/game UI.
- **Regionally-scoped online feature store + globally consistent experiment assignment** balances latency (no cross-region reads for the hot path) against the correctness requirement that experiment variants stay consistent per player.

## 39. Alternative Architectures

| Alternative | Description | Why Rejected / When Preferred |
|---|---|---|
| Single monolithic deep neural ranker (no feature store, features baked into one training pipeline) | One large DNN model per surface, trained end-to-end, features embedded in the model artifact | Rejected: couples feature engineering to model release cadence, breaks feature reuse across teams, harder to explain to live-ops/finance for offer pricing decisions; would be preferred if the org is small (single-team, single-title) and doesn't need cross-team feature sharing |
| Fully client-side/on-device personalization (rules run on console/PC) | Ranking logic and a small model shipped with the game client, no server round-trip | Rejected as primary: can't incorporate server-authoritative real-time context (current inventory/pricing) or cross-title behavioral features, harder to A/B test server-side, can't update model without a client patch; viable as a **fallback tier** only (already used here for offline/degraded mode) |
| Pre-compute all personalization nightly (no online inference) | Batch job computes personalized content list per player once/day, served from a static lookup at request time | Rejected as sole approach: fails FR3 (real-time context <5s freshness) — can't react to in-session events like match-just-ended; would be preferred for a lower-intensity use case (e.g., weekly digest email content) where staleness is fine and cost matters more than freshness |

## 40. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| GBDT over deep neural ranker | Fast CPU inference, explainable, cheaper to serve | Lower ceiling on modeling complex feature interactions vs. deep cross-networks |
| Regional (non-global) online feature store | Low latency, no cross-region write conflicts | Player roaming regions mid-session gets a cold feature cache in new region |
| Fail-open fallback on dependency failure | UI never blocks/breaks | Silent personalization degradation risk if fallback rate creep goes unnoticed without strict alerting |
| Weekly embedding refresh (not real-time) | Cheaper, simpler infra | New items get generic cold-start embedding for up to a week |
| At-least-once telemetry delivery + idempotent consumers | Simpler, higher-throughput ingestion | Requires careful idempotency design everywhere; a bug here causes silent double-counting |
| Canary-gated auto-halt (not auto-rollback) on business-metric breach | Avoids noisy-data-triggered rollback | Slower response to a genuinely bad model during the gap before human review |

## 41. Failure Modes

| Scenario | Impact | Mitigation |
|---|---|---|
| Feature store cluster AZ outage | Elevated latency/timeouts for affected shard's players | Multi-AZ replicas, automatic failover promotion, orchestrator falls back to cached/default features on timeout |
| Kafka broker overload during a live-event spike (e.g., new season launch) | Feature ingestion lag, real-time features become stale | Autoscaled consumer groups (KEDA), backpressure-aware producers, staleness-tolerant design (features still usable, just less fresh) |
| Bad model pushed to production (e.g., training data bug in latest retrain) | Degraded conversion/engagement for affected surface/title | Canary gate on business metrics catches before 100% rollout; automated rollback on gate breach |
| Experimentation platform outage | Can't assign variants | Default-to-control fallback, never block response on experiment service |
| Vector DB shard corruption/rebuild failure | Content-similarity candidates missing for affected title | Falls back to rule-based candidate pool only (embedding-based retrieval is additive, not sole source) |
| Cascading retry storm from a slow downstream dependency | Orchestrator thread/connection pool exhaustion, latency spike system-wide | Circuit breakers per downstream dependency, bounded retry budgets, timeouts strictly enforced (<10ms feature store, <10ms experiment service) |
| Poisoned/corrupted feature due to upstream telemetry bug (e.g., game patch changes event schema silently) | Model scores degrade silently (not a hard error) | Schema registry enforcement + drift detection (PSI) catches distribution shift within 24h; alerts trigger investigation |

## 42. Scaling Bottlenecks

- **At 10x scale (~2.5B QPS-equivalent portfolio growth... realistically ~25M peak concurrent, 2.5M→25M)**: online feature store read throughput becomes the first bottleneck (1.25M ops/sec → 12.5M ops/sec) — requires moving from a single Redis Cluster topology to a multi-cluster sharded-by-title architecture with a routing layer, and reconsidering whether every feature truly needs <5ms online access vs. tiering more features to a slightly-higher-latency-but-cheaper store.
- **At 100x scale**: candidate/inventory service (Postgres-backed) becomes a bottleneck for write-heavy live-ops pricing updates at global scale — would need to shift to an event-sourced catalog model with CQRS (separate write-optimized and read-optimized stores) rather than a single relational primary.
- **Vector DB** ANN index rebuild time (weekly batch) becomes untenable at 100x item catalog growth (200M items) — would need incremental/streaming index updates (e.g., DiskANN-style) rather than full rebuild.
- **Training pipeline**: GBDT distributed training on >500M rows starts hitting diminishing returns on histogram-based distributed training scalability — would need to move to sampled/incremental training or a switch to a more horizontally-scalable learning algorithm (e.g., online/streaming learning updates rather than full daily retrain from scratch).

## 43. Latency Bottlenecks

**p50 budget (~28ms total)**: gateway 2ms, feature store fetch 3ms, experiment assignment 5ms, candidate service 4ms, vector DB ANN query 8ms (only when embedding retrieval active), model scoring 1ms, response assembly 1ms, fan-out overhead/parallelism inefficiency ~4ms.

**p99 budget (~95ms total)**: dominated by tail latency in feature store (cache miss path, up to 15ms), experiment service under load (up to 20ms), vector DB under index-rebuild contention (up to 25ms), plus GC pauses / cold-pod scheduling variance (~10ms) — the vector DB ANN query and experiment assignment calls are the largest p99 contributors since they're the least-cacheable, most stateful dependencies in the fan-out.

- **Where time is actually spent**: not in model inference (sub-1ms) but in the **fan-out I/O** — three to four parallel network calls (feature store, experiment service, candidate service, optionally vector DB) each with their own tail latency; the theoretical parallel fan-out time is `max()` of the four, but real-world thread-pool/connection-pool contention under load pushes it closer to a blended average, explaining the p50→p99 gap.

## 44. Cost Bottlenecks

- **Primary cost driver**: online feature store (Redis Cluster) memory footprint — 120GB hot data across replicated shards at r6g.2xlarge pricing dominates steady-state infra spend, more than the CPU serving fleet (which is comparatively cheap per Section 6's ~160 vCPU total).
- **Secondary driver**: Kafka/MSK broker fleet sized for peak burst telemetry ingestion (9 brokers at m5.2xlarge) — provisioned for burst headroom that's idle most of the time; a cost lever here is more aggressive autoscaling of broker count or moving to a serverless streaming offering if usage is bursty enough to justify it.
- **Tertiary**: GPU training fleet (8x A100 spot, weekly) is bursty and already spot-optimized, but the two-tower embedding retrain is the single most expensive individual job in the system per run — a lever is reducing retrain frequency further (biweekly) if drift metrics show embeddings remain stable, trading freshness for cost.
- **Hidden cost**: cross-team feature store multi-consumer read load (matchmaking, CRM teams also reading the same store) inflates the read-QPS provisioning beyond what personalization alone needs — cost allocation/chargeback per consuming team recommended to keep incentives aligned on read-efficiency.

## 45. Interview Follow-Up Questions

1. How do you prevent training/serving skew given that features are computed by two different pipelines (streaming vs. batch)?
2. Walk me through how you'd detect and handle a sample-ratio mismatch in an experiment tied to this system.
3. Why LightGBM over a deep learning ranker here, and under what conditions would you reconsider?
4. How does point-in-time correctness actually get enforced in your training-set generation — what's the concrete join logic?
5. If the vector DB ANN index becomes a p99 latency outlier, what are your options without a full re-architecture?
6. How would you extend this system to support cross-title personalization (e.g., recommend a different EA title based on this player's behavior)?
7. What happens if two experiments (e.g., an offer-ranking experiment and a UI-layout experiment) interact and confound each other's results?
8. How do you handle a player who has just made a large purchase — does the system react within the same session, and how?
9. What's your approach to fairness/manipulation concerns — e.g., avoiding a model that learns to target vulnerable/high-spend players ("whales") in a way that raises ethical or regulatory concerns?
10. How would the architecture change if the latency budget were 5ms instead of 30ms?

## 46. Ideal Answers

1. **Training/serving skew**: Enforce a single feature registry as the source of truth for feature *definitions* (not just names) — both streaming and batch pipelines implement the same logical computation from a shared spec/config, ideally code-shared (e.g., same aggregation logic compiled to both a streaming operator and a batch SQL/Spark job). Additionally, log the actual feature values used at serving time (not recomputed) into the exposure log, so training can optionally use served-value replay instead of recomputing offline — this is the most robust skew-elimination technique, at the cost of extra log volume.

2. **SRM detection**: Run a chi-squared goodness-of-fit test daily comparing observed variant allocation counts to expected allocation ratios per experiment; alert if p < 0.001. Root-causes to check: caching/sticky-session bugs causing uneven bucketing, client-side retry logic double-counting one variant, or a fallback path that silently defaults everyone to control under partial outages (masquerading as an SRM). Fix by auditing the assignment determinism (hash function on player_id+experiment_id) and cross-checking exposure-log counts against gateway request counts.

3. **GBDT vs deep ranker**: LightGBM wins here because (a) feature set is primarily tabular/aggregate stats where tree splits capture interactions well, (b) inference cost/latency is far lower on CPU, (c) it's inherently more interpretable for feature-importance explainability required by live-ops/finance sign-off on pricing personalization, (d) training data volume per surface (tens of millions of rows) doesn't yet justify a deep model's higher data appetite. Would reconsider if: candidate pools grow to require learned dense retrieval at scale beyond what the two-tower model handles, or if sequential/session-level modeling (transformer-style next-item prediction) becomes the dominant driver of lift, justifying the added serving complexity.

4. **Point-in-time join**: For each labeled training example (player, exposure_time, outcome), query the offline feature store as of `exposure_time - buffer` (buffer accounts for ingestion lag, e.g., a few minutes) using either Iceberg/Delta time-travel snapshot queries or an explicit feature-versioning table with `valid_from/valid_to` ranges; join on `player_id` and the nearest valid feature snapshot strictly before `exposure_time`. Automated tests assert no feature's `valid_from` exceeds the label's exposure_time in the generated training set (a leakage linter run as part of the training DAG).

5. **Vector DB p99 outlier mitigation without re-architecture**: (a) add more replica shards to spread query load, (b) tune HNSW `ef_search` parameter down slightly to trade a small recall loss for latency, (c) move the ANN query off the synchronous critical path for surfaces where it's not essential — pre-compute top-K candidates asynchronously per session-start rather than per-request, caching the result for the session duration, (d) add a request-level timeout with graceful fallback to rule-based candidates only if the ANN query exceeds budget.

6. **Cross-title personalization**: Requires a shared player-identity graph across titles (linked EA accounts) and a shared feature namespace with title-agnostic features (e.g., "genre affinity," "spend tier across portfolio") alongside title-specific ones; the two-tower embedding approach extends naturally — item tower can include a cross-title catalog, player tower already ingests portfolio-wide behavioral aggregates. Main challenge is governance: cross-title data sharing requires explicit consent modeling and careful handling of titles with different regulatory audiences (e.g., youth-rated titles).

7. **Experiment interaction/confounding**: Use orthogonal randomization (independent hash seeds per experiment layer, "universe" splitting a la classic multi-layer experimentation frameworks) so experiments in *different* layers are statistically independent; for experiments in the *same* layer (e.g., two offer-ranking experiments), mutual exclusion is enforced at the assignment layer. Always run interaction checks post-hoc when two live experiments touch overlapping metrics — flag if effect sizes change significantly when segmenting by the other experiment's variant.

8. **Same-session reactivity to large purchase**: Yes — the `telemetry.purchase` event flows through the streaming feature-ingestion consumer within seconds, updating `rt:currency_balance` and a `recent_purchase_flag` in the online feature store; the next personalization call in the same session (e.g., opening the shop again) reads the updated feature immediately. Business rules also react directly (not just via the model) — e.g., suppress showing the same bundle just purchased, enforce a cooldown before showing another high-value offer to avoid fatigue/manipulation appearance.

9. **Fairness/manipulation concerns with high-spend targeting**: Explicitly monitor and cap "offer-intensity" or "spend-pressure" exposure per player per day (business rule ceiling, not left purely to model optimization) regardless of predicted propensity; exclude or down-weight training signals that would let the model over-index on compulsive-spending patterns (e.g., cap the LTV-segment feature's influence, monitor for feedback loops where the model learns to target players showing addictive engagement patterns); maintain a live-ops/compliance review process for any surface tied to monetary offers, and support fast manual override/kill-switch per segment. This is as much a policy/process control as a modeling one — the architecture must expose the override and monitoring hooks (Section 4 FR7, Section 22) precisely to support this.

10. **5ms budget instead of 30ms**: Would require moving from a synchronous fan-out-per-request model to a **pre-computation** model — compute personalization results asynchronously at session-start (or even predictively pre-fetch for likely-next-surfaces) and serve from a session-scoped cache with a <5ms lookup, since no combination of network fan-out (feature store + experiment + candidate + vector DB, each with irreducible network RTT) can reliably fit in 5ms p99 across regions. This trades some freshness (can't react to same-second events) for latency, and pushes real-time reactivity (Section 18) to a "next surface load" cadence instead of true per-request freshness.
