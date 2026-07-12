# Personalization Engine

## 1. Problem Framing

Build a per-player personalization system for an EA live-service game portfolio. The system decides, in real time, what content, UI layout, and store offers to show each player — menu tiles, battle-pass nudges, item-shop bundles, and push/in-game messaging.

The interviewer wants to see: how do you turn player telemetry (session length, purchase history, skill rating, churn risk) into a low-latency decision service that (a) serves ranked content within the UI render budget, (b) integrates with an experimentation platform so every personalization is A/B-testable, and (c) stays consistent with a feature store so training/serving skew doesn't wreck model quality.

- **Business goal**: lift D7/D30 retention, ARPDAU, and offer conversion without degrading game performance.
- **Not in scope**: matchmaking skill-balancing, anti-cheat, chat moderation.
- **In scope**: content/offer ranking, UI slot personalization, feature store, online inference, experimentation hooks, real-time session context.

## 2. Functional Requirements

- FR1: Given `(player_id, surface, context)`, return ranked content/offer candidates with score + experiment variant tag.
- FR2: Support multiple surfaces (home, item shop, post-match, battle-pass nudge, push) from one platform, each with its own candidate pool and model.
- FR3: Ingest real-time context (session length, last match result, currency balance) with <5s freshness.
- FR4: Ingest batch features (7/30/90-day aggregates, churn score, LTV segment) refreshed daily.
- FR5: Every decision tied to an experiment assignment (control/treatment), logged for offline analysis.
- FR6: Cold-start players (<24h old) handled via fallback rules/popularity models.
- FR7: Manual override/business rules (e.g., forced promo during a live event) can supersede model output.
- FR8: Explainability metadata (top contributing features) for compliance and live-ops debugging.
- FR9: Multi-title reuse — same platform serves different titles with different feature schemas/models.
- FR10: Opt-out / regional consent (GDPR/CCPA) — personalization degrades gracefully to non-personalized defaults.

## 3. Non-Functional Requirements

| Dimension | Target |
|---|---|
| Latency | p50 < 30ms, p99 < 100ms (server-side inference call) |
| Availability | 99.95% inference path; degraded-mode fallback 99.99% |
| Throughput | Peak 250K QPS globally across surfaces |
| Consistency | Batch features: eventual, <24h staleness OK; real-time features <5s staleness; strong consistency not required |
| Cost | <$0.000015/request to keep opex under 3% of incremental revenue lift |
| Durability | Feature/event logs: 90 days hot, 2 years cold |
| Freshness | Retrain daily (offer ranker), weekly (content ranker) |

## 4. Clarifying Questions

1. One title or a shared multi-tenant platform across EA's portfolio?
2. Native client UI (Unreal/Frostbite) or web-based store frontend?
3. Synchronous (blocking UI render) or can we pre-fetch personalization at session start?
4. Acceptable staleness for "real-time" context — usable same session, or only next session?
5. On-device fallback needed for offline/unreachable-backend cases?
6. Hard latency budget from the game engine's UI loop, or decoupled (loading screen)?
7. Build-vs-buy experimentation platform — does it support server-side assignment at our QPS?
8. Which regions require consent-gated personalization (EU/UK, California, loot-box-adjacent regs)?
9. Do offers have inventory/pricing constraints (no conflicting bundles, daily offer-fatigue budget)?
10. Is under-13 (COPPA) traffic present, requiring personalization disabled for that cohort?

## 5. Assumptions

1. Portfolio MAU: 60M; DAU: 12M (20% DAU/MAU).
2. Peak concurrent players: 2.5M.
3. Each session triggers ~8 personalization calls (home, shop, post-match, push checks, battle-pass, mode-select, loading tile).
4. Average session: 35 min; 2.3 sessions/day/DAU.
5. Candidate pool per surface: 50-300 items, pre-filtered by eligibility/inventory rules.
6. Real-time features arrive via game server telemetry (match end, purchase, level-up).
7. Feature store is multi-consumer (also serves matchmaking, churn-prevention CRM).
8. Experimentation platform is a shared EA service, exposing assignment API at <10ms p99.
9. Models are LightGBM (GBDT) for ranking; one exception — a two-tower embedding model for content/offer similarity.
10. GPU used only for embedding training/batch inference; online serving is CPU-bound (tree models).

## 6. Capacity Estimation

**QPS**
- 2.5M peak concurrent × ~0.23 calls/min/player ≈ **9,600 QPS sustained**, bursting to **~25K QPS** during live events.
- Design target: **250K QPS** global ceiling (headroom for portfolio growth, burst, multi-surface fan-out).

**Storage**
- Online feature store: 60M players × ~2KB ≈ **120 GB** hot in-memory/KV.
- Offline feature store: ~27 TB raw → **~3.5 TB** compressed columnar.
- Event log: ~35 TB/year raw → **~3.5 TB/year** compressed.

**Model size**
- LightGBM models: ~30 models (surfaces × titles) × 5MB = **150 MB** total — trivial.
- Two-tower embeddings: 128-dim × 2M items × 4 bytes = **1 GB** embedding table.

**Compute (serving)**
- ~0.5ms per scoring call for 300 candidates, single core.
- At 25K QPS burst: **13 pods** minimum, provision **40 pods** (headroom + AZ redundancy) at 4 vCPU/8GB.
- GPU only for offline embedding training (8x A100, ~6h/week).

**Feature store QPS**
- ~150 features/call × 250K QPS ≈ **1.25M read ops/sec** — drives choice of in-memory KV (Redis/DAX) over RDBMS.

## 7. High-Level Architecture

```
Game Clients (console/PC UI, Store, HUD)
        │ gRPC/HTTPS
        ▼
API Gateway / Edge (authn, rate limit, region routing)
        ▼
Personalization Orchestrator
  - candidate retrieval (eligibility/inventory)
  - experiment assignment lookup
  - feature fetch fan-out
  - model invocation + business-rule override
  - response assembly + explainability tags
        │
   ┌────┼────────────┬───────────────┐
   ▼    ▼             ▼               ▼
Online   Experiment   Model Serving   Candidate/
Feature  Assignment   (LightGBM +     Inventory
Store    Service      2-tower)        Service
   │                     │                │
   ▼                     │                ▼
Feature                  │           Vector DB (ANN,
Ingestion                │           content/offer embeds)
(stream+batch)           ▼
   │              Experimentation Platform (EA shared svc)
   ▼
Streaming Bus (Kafka): telemetry, match-end, purchase, session events
   │
   ├──▶ Offline Feature Store (columnar warehouse)
   └──▶ Training Pipeline (Spark + distributed GBDT/embedding training)
              │
              ▼
        Model Registry + CI/CD (canary/blue-green rollout)
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| API Gateway/Edge | AuthN, TLS, regional routing, rate limiting | HTTPS/gRPC | Horizontal, autoscaled on RPS |
| Personalization Orchestrator | Fan-out to feature store/models/experiment service, assemble response | Internal gRPC | Stateless pods, HPA on CPU+QPS |
| Online Feature Store | Low-latency feature vectors per player | KV `get(player_id, feature_group)` | Sharded by `player_id` hash |
| Feature Ingestion (streaming) | Consume Kafka events, update real-time features | Kafka consumer group | Partition-parallel |
| Feature Ingestion (batch) | Nightly Spark jobs compute aggregates | Airflow DAG | Spark autoscale |
| Model Serving | Host LightGBM + two-tower models | gRPC `predict(features[]) -> scores[]` | Stateless pods, HPA on QPS/latency |
| Vector DB | ANN lookup for embedding similarity | `query(embedding) -> top_k ids` | Sharded index, replica-per-AZ |
| Candidate/Inventory Service | Eligibility filtering, pricing rules | REST/gRPC | Stateless, cache-heavy |
| Experimentation Platform | Deterministic variant assignment, exposure logging | `assign(player_id, experiment_id) -> variant` | Shared EA service |
| Model Registry / CI-CD | Versioned artifacts, rollout orchestration | Internal API | Control plane |
| Training Pipeline | Offline training (GBDT, embeddings) | Airflow + Spark/PyTorch DDP | GPU for embeddings, CPU for GBDT |
| Offline Feature Store | Point-in-time snapshots for training | SQL/columnar engine | Partitioned by date + shard |

## 9. API Design

```
POST /v2/personalize/{surface}
Headers: Authorization: Bearer <service-jwt>, X-Player-Session-Id, X-Region

Request:
{
  "player_id": "p_9f21ac...",
  "surface": "item_shop",
  "title_id": "fifa25",
  "context": {
    "session_id": "s_88af...",
    "last_match_result": "win",
    "current_currency_balance": 1200,
    "consent_state": "personalization_allowed"
  },
  "max_results": 10
}

Response: 200 OK
{
  "request_id": "req_abc123",
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
  "model_version": "offer_ranker_v14.2"
}
```

**Versioning**: URL path versioned (`/v2/`); backward-compatible additions only within a major version; breaking changes bump version with a 90-day dual-run deprecation window.

| Endpoint | Method | Purpose |
|---|---|---|
| `/v2/personalize/{surface}` | POST | Real-time ranked result |
| `/v2/personalize/batch` | POST | Pre-compute for batch of players (push campaigns) |
| `/v2/feature/{player_id}` | GET | Debug: raw feature vector (RBAC-gated) |
| `/v2/health` | GET | Liveness/readiness probe |
| `/v2/models/{surface}/version` | GET | Currently serving model version |

## 10. Database Design

| Store | Type | Why | Partition Key |
|---|---|---|---|
| Online Feature Store | Redis Cluster / DynamoDB | Sub-5ms reads at 1.25M ops/sec, simple KV access | `player_id` hash |
| Offline Feature Store | Columnar warehouse (Parquet/Trino) | Point-in-time queries, cheap compressed storage | `event_date`, clustered by `player_id` |
| Candidate/Offer Catalog | Postgres | Strong consistency for pricing/inventory, transactional live-ops updates | Sharded by `title_id` |
| Experiment Exposure Log | Kafka → S3/Iceberg | High write volume, analytical reads | `experiment_id` + date |
| Vector DB | ANN store (Section 16) | Similarity search not efficient in KV/relational | `title_id` + embedding cluster |

Online feature record (hash map keyed by `player_id`):
```
rt:last_match_result: "win"
rt:currency_balance: 1200
batch:churn_score_7d: 0.12
batch:purchase_propensity_30d: 0.68
batch:ltv_segment: "whale_tier2"
updated_at: 1735689600
```

## 11. Caching

- **Cached**: candidate pool per surface/title (TTL 5 min), model prediction for identical `(player_id, surface, context_hash)` (TTL 30-60s to absorb repeat UI re-renders), experiment assignment (session-scoped, deterministic).
- **Strategy**: cache-aside for feature reads via an in-process orchestrator L1 cache for hot players (the feature store itself is already the primary cache).
- **Invalidation**: event-driven for candidate pools (Kafka `inventory_updated` invalidates immediately); short TTL for feature cache (simpler than explicit invalidation).
- **Write-through** for offer inventory — live-ops price/availability changes must propagate immediately, so Postgres writes also write through to the candidate cache synchronously.
- **Negative caching**: cold-start players cache a "use fallback" flag for 60s to avoid hammering the feature store.

## 12. Queues & Async Processing

- **Queued**: raw telemetry (match end, purchase, level-up), experiment exposure logs, batch personalization requests (push campaigns), model training triggers, feature-store write-backs.
- **Delivery semantics**: telemetry is at-least-once (Kafka default); aggregation consumers are idempotent (upsert by `player_id + feature_key`, last-write-wins by timestamp).
- **Exactly-once** only for revenue-impacting events (purchase confirmations) via transactional producer + dedup table on `transaction_id` (24h window).
- **Dead-letter handling**: malformed events route to a DLQ after 3 retries with backoff; alert if DLQ rate >0.1% of topic volume.
- **Batch push personalization**: queued as jobs, processed by a worker pool calling the same orchestrator API, rate-limited to avoid spiking the online path.

## 13. Streaming & Event-Driven Architecture

| Topic | Producer | Consumer(s) | Key fields |
|---|---|---|---|
| `telemetry.match_end` | Game servers | Feature ingestion, warehouse sink | `player_id, match_id, result, timestamp` |
| `telemetry.purchase` | Store/billing | Feature ingestion, finance sink | `player_id, sku, amount, transaction_id` |
| `telemetry.session` | Client/session service | Real-time feature updater | `player_id, session_id, event_type, timestamp` |
| `personalization.exposure` | Orchestrator | Experimentation analysis, warehouse | `player_id, surface, experiment_id, variant, model_version` |
| `catalog.inventory_updated` | Live-ops tooling | Candidate cache invalidator | `title_id, item_id, change_type` |

- **Consumer groups**: feature-ingestion keyed by `player_id` hash for per-player ordering (128 partitions); experimentation-analysis group runs independently and can lag without affecting serving.
- **Schema management**: Avro/Protobuf with a schema registry, backward-compatible evolution enforced.
- **Ordering**: per-player ordering preserved via partition key; cross-player ordering not required.

## 14. Model Serving

- **Framework**: LightGBM served via a lightweight gRPC service (or Triton with FIL backend) — CPU-only.
- **Two-tower embedding model**: player-tower forward pass computed online per request; item embeddings precomputed offline into the vector DB.
- **Batching**: up to 300 candidates scored per request in one LightGBM `predict` call; Triton dynamic batching used for the two-tower query path (uniform input shape makes cross-request batching viable there).
- **Multi-model**: one model per `(title_id, surface)` — ~30 active models, routed by request metadata, hot-swappable via registry poll/push (no pod restart).
- **Hardware**: CPU serving fleet (~40 pods × 4 vCPU); GPU reserved for offline training/weekly embedding refresh.
- **Latency**: LightGBM scoring for 300 candidates ≈ 0.5-1ms — not the bottleneck (see Section 43).

## 15. Feature Store

- **Online**: Redis Cluster (or DynamoDB DAX), latest value per feature per player, sub-5ms p99 reads.
- **Offline**: columnar warehouse (Parquet/Iceberg, Spark/Trino), full feature history for training.
- **Point-in-time correctness**: training uses time-travel joins — features for a label at time `T` are joined as they existed at `T - epsilon`, never using future data. Implemented via Iceberg time-travel or explicit `valid_from/valid_to` intervals. This avoids label leakage (e.g., using a post-event churn aggregate computed after the label window).
- **Feature registry**: central definition of every feature (owner, freshness SLA, computation logic), shared across teams to prevent feature drift/duplication.
- **Write path**: streaming writes real-time features directly online; batch pipeline writes offline first, then syncs to online nightly (batch features up to 24h stale, per NFR).
- **Backfill**: runs against the offline store only, never touches online directly, to avoid serving inconsistent partial-backfill states.

## 16. Vector Database

Used for content/offer similarity — the two-tower model's retrieval stage ("find offers similar to what this player engages with").

- **Index**: HNSW, chosen over IVF-PQ since the catalog is modest (2M items) and HNSW gives better recall at low latency without IVF-PQ's compression tradeoffs; target recall@10 ≥0.95 at <10ms.
- **Embedding dim**: 128 (2M items × 128 × 4 bytes = 1GB, fits in-memory per shard).
- **Sharding**: one index per `title_id`, replicated across AZs, rebuilt weekly with blue/green swap to avoid query disruption.
- **Updates**: item embeddings batch-refreshed weekly (catalog churn is slow vs. player behavior churn); new items get a category-average cold-start vector until the next refresh.

## 17. Embedding Pipelines

Two-tower model: player embeddings computed online per request; item/offer embeddings computed offline in batch.

- **Player tower**: small MLP (~64K params) over the feature-store vector, run online, <1ms on CPU.
- **Item tower**: MLP over item metadata, run offline, output written to vector DB.
- **Training**: contrastive loss (in-batch negatives) on (player, engaged-item) pairs from exposure+engagement logs; retrained weekly.
- **Serving flow**: request → player embedding (online) → ANN query → union with rule-based candidates → score all with LightGBM ranker → return top-N.
- **Why embeddings at all**: pure rule/CF-based candidate generation misses long-tail personalization (e.g., a new bundle similar to past purchases but not yet co-purchased by enough players); embeddings generalize better for cold-start items.

## 18. Inference Pipeline (Request Lifecycle)

```
Client → POST /v2/personalize/item_shop
  → API Gateway [~2ms: TLS, authn, rate-limit]
  → Orchestrator:
      1. Parallel fan-out: Feature Store GET [~3ms], Experiment Assignment [~5ms], Candidate Service [~4ms]
      2. If two-tower enabled: player embedding [~1ms] → ANN query [~8ms] → merge with rule-based candidates
      3. Model Serving: batch-score all candidates [~1ms]
      4. Apply business-rule overrides [~1ms]
      5. Assemble response + explainability + experiment tag [~1ms]
      6. Async: emit personalization.exposure to Kafka (fire-and-forget)
  → Response [total server-side: ~25-30ms p50]
```

- **Failure handling**: feature-store timeout (>10ms) → proceed with cached/default features (degraded, not a hard failure); model-serving failure → fall back to popularity-ranked default list; experimentation timeout → default to control (never block response on it).
- **Idempotency**: `request_id` lets clients retry without double-counting exposure events.

## 19. Training Pipelines

- **Data prep**: point-in-time joined offline features + label source (purchase/engagement within N hours of exposure) → versioned Parquet training set.
- **Labels**: offer ranker = binary purchase-within-24h; content ranker = engagement (click/dwell-time weighted) within session.
- **Orchestration** (Airflow DAG): validate feature freshness → materialize training set (point-in-time join) → train LightGBM distributed (histogram-based, ~16 CPU workers) → train two-tower via PyTorch DDP on 8x A100 (weekly) → offline eval (AUC, NDCG@10, calibration) → push to registry if gates pass → trigger canary.
- **Reproducibility**: every run tagged with feature-store snapshot version, code commit hash, and data date range.

## 20. Retraining Strategy

- **Cadence**: offer ranker daily (catalog/pricing shift fast), content ranker weekly, two-tower embeddings weekly.
- **Trigger-based**: early retrain if drift crosses threshold (Section 21), candidate catalog changes >20%, or a live-event/campaign requires a refresh.
- **Champion/challenger**: every retrain evaluated offline against current champion on a held-out replay set before canary eligibility — no auto-promotion without an eval-gate pass (min NDCG@10 improvement, no regression >1% on any segment).

## 21. Drift Detection

| Drift Type | Metric | Action |
|---|---|---|
| Data drift | PSI per feature vs. training baseline | PSI > 0.2 on top-20 feature → alert, flag early retrain |
| Concept drift | Rolling 7-day live AUC/NDCG vs. training-time metric | Drop >5% relative → alert, trigger challenger eval |
| Prediction drift | KL divergence of score distribution vs. prior week | KL > 0.1 → investigate |
| Catalog drift | % new/removed items vs. prior week | >20% → force embedding refresh |
| Calibration drift | Predicted vs. actual conversion (reliability curve) | Brier degradation >10% → alert |

Drift computation runs nightly over the offline store + exposure logs, feeding the monitoring/alerting pipeline.

## 22. Monitoring

- **Infra**: pod CPU/mem, latency histograms (p50/p90/p99), error rate, feature-store hit/miss rate, Kafka consumer lag, vector DB query latency.
- **Model quality**: offline eval per training run, online proxy metrics (CTR on personalized slot vs. control), drift metrics.
- **Business**: D1/D7/D30 retention lift (experiment-attributed), ARPDAU lift, offer conversion, personalization coverage (% served vs. fallback), sample-ratio-mismatch (SRM) checks.
- **Dashboards**: per-surface, per-title breakdowns; live-ops dashboard for real-time offer performance during events.

## 23. Alerting

| Condition | Threshold | Route |
|---|---|---|
| p99 latency | >150ms sustained 5min | Page SRE |
| Error rate | >1% over 5min | Page SRE |
| Fallback rate | >10% traffic for 10min | Page SRE + notify ML team |
| Feature staleness | Real-time features >30s stale | Page data-eng |
| Kafka consumer lag | >60s on `telemetry.*` | Page data-eng |
| Model AUC drop | >5% relative vs. baseline | Ticket ML team |
| Drift PSI breach | PSI > 0.2 | Ticket ML team |
| SRM in experiment | p < 0.001 chi-sq | Notify experimentation + ML |
| DLQ volume | >0.1% of topic volume | Notify data-eng |
| Cost anomaly | Daily spend >20% above 7-day avg | Notify eng lead + finance |

- **Routing**: infra/latency → SRE on-call (immediate page); model-quality/drift → ML platform on-call (business-hours ticket unless paired with a business-metric regression).

## 24. Logging

- **Structured logs**: JSON with `request_id`, hashed `player_id`, `surface`, `model_version`, `experiment_id/variant`, `latency_ms`, `fallback_used`.
- **PII**: `player_id` hashed/pseudonymous everywhere (raw ID never logged); behavioral-context logs (e.g., purchase amounts) are RBAC-restricted, excluded from general-purpose search tools.
- **Retention**: hot logs 30 days; cold archival (S3) 2 years for audit/experiment reproducibility.
- **Right-to-erasure**: deletion requests propagate a tombstone purging hashed-ID records from hot stores within 30 days, cold archives within 90 days (GDPR/CCPA).

## 25. Security

- **Threats specific to this system**:
  - Spoofed client context (fake currency/session data) to unlock better offers → only trust server-authoritative telemetry for anything pricing-relevant; client-supplied context limited to non-sensitive UI hints.
  - Feature store as an exfiltration target (behavioral/purchase profiles across 60M players) → encrypt at rest, strict RBAC, network-isolated, audit-logged.
  - Model extraction/inversion via repeated queries → rate-limit debug endpoints, require elevated auth, monitor for anomalous query patterns.
  - Replayed experiment-assignment requests → deterministic hashing assignment, signed tokens.
- **Encryption**: TLS 1.3 in transit, AES-256 at rest.
- **Data minimization**: only features with documented business justification are ingested (feature registry enforces sign-off).

## 26. Authentication

- **Service-to-service**: mTLS within the internal mesh; short-lived service JWTs for cross-cluster calls.
- **Client-to-edge**: player's existing EA account session token validated at the gateway, which attaches a verified `player_id` claim downstream — internal services never trust client-asserted identity.
- **Debug endpoints**: require elevated SSO-backed employee token with RBAC scope `personalization:debug`.

## 27. Rate Limiting

- **Algorithm**: token bucket per `player_id` at the gateway (e.g., 20 req/10s burst) — generous enough for legitimate UI fan-out, blocks scripted abuse.
- **Per-title limits**: separate buckets per `title_id` so one title's spike can't starve another's capacity.
- **Debug endpoints**: strict per-employee-token limit since these bypass caching and hit the feature store directly.
- **Backpressure**: adaptive rate limiting (reduced refill) as downstream saturates, prioritizing session-critical surfaces over batch push pre-computation.

## 28. Autoscaling

- **Orchestrator/Model Serving**: HPA on RPS + CPU (target 60%), 13-60 pods/region, fast scale-up (30s) and slow scale-down (5min) to handle live-event spikes without flapping.
- **Feature store**: read-replica addition triggered by >70% memory or p99 read latency >8ms.
- **Vector DB**: scaled by shard replication; KEDA on query-queue depth if queue-fronted.
- **Kafka consumers**: KEDA on consumer-group lag, scale out past 10K messages/partition.
- **Batch training**: no autoscaling — scheduled fixed-size clusters spun up/down by the DAG.

## 29. Cost Optimization

- **Spot instances**: training GPU nodes (weekly, not latency-sensitive) on spot with checkpointing — ~60-70% savings.
- **Caching**: candidate-pool and prediction caching cuts ~15-20% of redundant model-serving calls.
- **Compact embeddings**: 128-dim (vs. 512+) trades small recall loss for 4x smaller vector DB footprint and faster ANN queries.
- **Dynamic batching**: improves GPU/CPU utilization on the two-tower query path, shrinking fleet size needed.
- **Regional right-sizing**: don't provision peak-global capacity everywhere — scale down overnight-region fleets aggressively.
- **Tiered storage**: only genuinely <5s-freshness features live in expensive Redis; everything else in cheaper offline columnar storage.
- **Shared feature store**: avoids duplicate feature-engineering pipelines across teams, amortizing compute cost.

## 30. Operational Concerns

At SDE2 scope, treat this as a checklist: **backups** (automated snapshots of model registry/feature store with a tested restore path), **rollback** (every deploy revertible to last-known-good in one command), **canary/blue-green rollout** (shift a small traffic % first, watch error rate and key metrics, then ramp), and **basic observability** (dashboards + alerts on latency, error rate, top model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level concerns — worth knowing they exist, not worth rehearsing.

## 31. Why This Architecture

- **Separating orchestrator/feature-store/model-serving** lets each scale against its own bottleneck (feature-store is read-QPS-bound, model-serving is CPU-bound, orchestrator is fan-out/IO-bound) instead of one monolith scaling on the max of all dimensions.
- **CPU tree models by default, GPU only for embeddings** matches the real latency/cost profile — tabular ranking with <200 features doesn't need a neural net, and GBDTs give explainability (FR8) cheaply.
- **Fail-open design** (fallback to popularity/rules on any dependency failure) is essential for a UI-blocking call — personalization must never break the storefront.
- **Regional online feature store + globally consistent experiment assignment** balances low-latency reads against the need for experiment variants to stay consistent per player.

## 32. Alternative Architectures

| Alternative | Why Rejected / When Preferred |
|---|---|
| Single monolithic DNN ranker (no feature store, features baked into training) | Couples feature engineering to model release cadence, breaks cross-team feature reuse, harder to explain for pricing sign-off. Preferred if org is small (single-team, single-title). |
| Fully client-side/on-device personalization | Can't use server-authoritative real-time context, harder to A/B test, needs a client patch to update. Viable only as the fallback tier already used here. |
| Pre-compute all personalization nightly (no online inference) | Fails FR3 (real-time freshness) — can't react to in-session events. Fine for lower-intensity use cases (e.g., weekly digest email) where staleness is acceptable. |

## 33. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| GBDT over deep neural ranker | Fast CPU inference, explainable, cheap | Lower ceiling on complex feature interactions |
| Regional (non-global) feature store | Low latency, no cross-region write conflicts | Roaming players get a cold cache in new region |
| Fail-open fallback | UI never blocks/breaks | Silent degradation risk if fallback creep goes unmonitored |
| Weekly embedding refresh | Cheaper, simpler | New items get generic cold-start embedding for up to a week |
| At-least-once telemetry + idempotent consumers | Simpler, higher throughput | Requires careful idempotency everywhere |
| Canary-gated auto-halt (not auto-rollback) | Avoids noisy-data-triggered rollback | Slower response to a genuinely bad model during human review |

## 34. Failure Modes

| Scenario | Impact | Mitigation |
|---|---|---|
| Feature store AZ outage | Elevated latency/timeouts for affected shard | Multi-AZ replicas, automatic failover, fallback to cached/default features |
| Kafka overload during a live-event spike | Feature ingestion lag, staler real-time features | Autoscaled consumers (KEDA), backpressure, staleness-tolerant design |
| Bad model pushed to prod | Degraded conversion/engagement | Canary gate on business metrics, automated rollback on breach |
| Experimentation platform outage | Can't assign variants | Default-to-control, never block response |
| Vector DB shard corruption | Missing content-similarity candidates | Falls back to rule-based candidate pool (embeddings are additive) |
| Cascading retry storm from a slow dependency | Thread/connection pool exhaustion, system-wide latency spike | Circuit breakers, bounded retry budgets, strict timeouts |
| Poisoned feature from upstream schema change | Silent score degradation | Schema registry enforcement + PSI drift detection catches shift within 24h |

## 35. Scaling Bottlenecks

- **At 10x** (2.5M → 25M peak concurrent): online feature store read throughput (1.25M → 12.5M ops/sec) becomes the first bottleneck — requires multi-cluster sharded-by-title Redis with a routing layer, and reconsidering which features truly need <5ms access.
- **At 100x**: candidate/inventory service (Postgres) becomes a write bottleneck for global live-ops pricing updates — would need an event-sourced catalog with CQRS.
- **Vector DB**: weekly full-index rebuild becomes untenable at 100x catalog growth — would need incremental/streaming index updates (e.g., DiskANN-style).
- **Training**: GBDT distributed training on >500M rows hits diminishing returns — would need sampled/incremental training or streaming learning updates instead of full daily retrains.

## 36. Latency Bottlenecks

**p50 (~28ms)**: gateway 2ms, feature store 3ms, experiment assignment 5ms, candidate service 4ms, vector DB ANN 8ms (when active), model scoring 1ms, response assembly 1ms, fan-out overhead ~4ms.

**p99 (~95ms)**: dominated by feature-store cache-miss tail (up to 15ms), experiment service under load (up to 20ms), vector DB under index-rebuild contention (up to 25ms), plus GC/cold-pod variance (~10ms).

Model inference itself is never the bottleneck (sub-1ms) — time is spent in the **fan-out I/O**: three to four parallel network calls, each with its own tail. Theoretical parallel time is `max()` of the four, but connection-pool contention under load pushes real p99 closer to a blended average.

## 37. Cost Bottlenecks

- **Primary**: online feature store (Redis) memory footprint — 120GB hot data dominates steady-state spend, more than the CPU serving fleet.
- **Secondary**: Kafka/MSK broker fleet sized for peak burst ingestion, mostly idle — lever is more aggressive autoscaling or a serverless streaming offering.
- **Tertiary**: GPU training fleet (weekly two-tower retrain) is the single most expensive individual job — lever is reducing retrain frequency if drift stays stable.
- **Hidden**: cross-team feature-store read load (matchmaking, CRM) inflates provisioning beyond personalization's own needs — recommend chargeback per consuming team.

## 38. Interview Follow-Up Questions

1. How do you prevent training/serving skew given features are computed by two different pipelines?
2. Walk through detecting and handling a sample-ratio mismatch in an experiment.
3. Why LightGBM over a deep learning ranker, and when would you reconsider?
4. How does point-in-time correctness get enforced in training-set generation, concretely?
5. If the vector DB ANN index becomes a p99 outlier, what are your options without a re-architecture?
6. How would you extend this to cross-title personalization?
7. What happens if two experiments interact and confound each other's results?
8. How does the system react within-session to a player's large purchase?
9. How do you address fairness/manipulation concerns — e.g., a model targeting vulnerable high-spend players?
10. How would the architecture change if the latency budget were 5ms instead of 30ms?

## 39. Ideal Answers

1. **Skew**: A single feature registry as source of truth for feature definitions, shared code between streaming/batch pipelines. Logging actual served feature values into the exposure log (for replay in training) is the most robust fix, at the cost of extra log volume.

2. **SRM**: Daily chi-squared goodness-of-fit test on observed vs. expected variant allocation, alert if p < 0.001. Common causes: sticky-session bucketing bugs, client-side retry double-counting, fallback silently defaulting to control.

3. **GBDT vs. deep ranker**: Wins because the feature set is mostly tabular aggregates, CPU inference is cheap, and tree feature importance satisfies interpretability needs for pricing sign-off. Reconsider if candidate pools or sequential/session modeling needs grow enough to justify a deep model.

4. **Point-in-time join**: Query the offline store as of `exposure_time - buffer` using time-travel snapshots or `valid_from/valid_to` versioning, joining the nearest valid snapshot strictly before exposure_time. An automated leakage linter asserts no feature's `valid_from` exceeds the label's exposure_time.

5. **Vector DB p99 mitigation**: Add replica shards, tune HNSW `ef_search` down to trade recall for latency, move the ANN query off the synchronous path (pre-compute top-K per session-start, cached) with a timeout fallback to rule-based candidates.

6. **Cross-title personalization**: Needs a shared player-identity graph and shared feature namespace mixing title-agnostic and title-specific features; the two-tower player tower extends naturally. Main challenge is governance — consent and differing regulatory audiences (e.g., youth-rated titles).

7. **Experiment interaction**: Orthogonal randomization (independent hash seeds per layer) so experiments in different layers are statistically independent, with mutual exclusion within the same layer. Run post-hoc checks flagging effect-size shifts when segmented by another live experiment's variant.

8. **Same-session reactivity**: Yes — the purchase event flows through streaming ingestion within seconds, updating features like `recent_purchase_flag` before the next call. Business rules also react directly, e.g., suppressing the just-purchased bundle and enforcing a cooldown.

9. **Fairness**: Cap offer-intensity exposure per player per day as a hard rule regardless of predicted propensity, and down-weight training signals that would let the model over-index on compulsive-spending patterns. This is as much a policy control as a modeling one — the architecture needs override/monitoring hooks to support it.

10. **5ms budget**: Move from synchronous per-request fan-out to pre-computation at session-start, served from a session-scoped cache with <5ms lookup — network RTT across feature store/experiment/vector DB calls can't reliably fit in 5ms p99. Trades same-second freshness for latency.
