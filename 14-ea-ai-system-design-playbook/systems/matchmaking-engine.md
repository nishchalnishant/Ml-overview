# Matchmaking Engine

## 1. Problem Framing

Design a skill-based matchmaking system for an EA live-service title (e.g., squad shooter or sports title) that pairs players into balanced matches within acceptable wait times, across regions, at global scale.

- Core tension: **fairness (skill parity)** vs **latency (queue wait)** vs **network quality (ping)**.
- Real-time system — players actively wait, so every second of queue time is a UX cost.
- Not a recommendation system — it's continuous constrained optimization + real-time assignment over a live pool.
- Interfaces with: game client, session/lobby service, dedicated game server (DGS) fleet, anti-cheat, telemetry.

## 2. Functional Requirements

- FR1: Accept a matchmaking request (solo or party) with skill rating, region, constraints, game-mode.
- FR2: Group players into match-sized teams (e.g., 10 for 5v5) satisfying skill-balance and latency constraints.
- FR3: Support parties — premade groups queue and match together.
- FR4: Support role-based matchmaking (tank/support/DPS) where applicable.
- FR5: Progressively widen search criteria (skill band, latency) over time to bound wait.
- FR6: Return match assignment: session ID, DGS endpoint, teammates, opponents.
- FR7: Update skill ratings post-match (TrueSkill/Elo).
- FR8: Allow cancel-search / requeue.
- FR9: Support backfill — replace a disconnected mid-match player.
- FR10: Provide ETA estimate to client.

## 3. Non-Functional Requirements

| Attribute | Target |
|---|---|
| Match-found latency (p50) | ≤ 15s |
| Match-found latency (p99) | ≤ 60s (else relax constraints or bot-fill) |
| Ticket ingestion latency | ≤ 50ms p99 |
| Availability | 99.95% (critical path — outage = can't play) |
| Throughput | 500K concurrent searching players at peak (global) |
| Consistency | Strong: "player in exactly one match" (no double-booking). Eventual OK for rating propagation |
| Cost | Compute scales sub-linearly with CCU via batched ticks, not per-player threads |
| Durability | Ticket state survives matchmaker pod restart |

## 4. Clarifying Questions

1. Team size/mode — 1v1, 5v5, battle royale (60-100)? Changes complexity from pairing to bin-packing.
2. Is cross-region matchmaking acceptable to reduce wait time?
3. Are mixed-skill parties averaged, capped by highest, or restricted to narrow spread?
4. Ranked vs unranked with different fairness tolerances?
5. Elo (simple, pairwise) vs TrueSkill (Bayesian, team-native)?
6. Role queues / composition constraints needed?
7. Smurf/new-account detection feeding initial rating?
8. DGS fleet model — pre-warmed pool or on-demand (affects match-to-play latency)?
9. Backfill time budget for a mid-match leaver?
10. Do low-population regions (e.g., SEA at 3am) need special handling?

## 5. Assumptions

1. 40M MAU, 6M DAU, 2M peak CCU globally, ~35% peak-region skew.
2. 25% of DAU actively matchmaking at peak → ~500K concurrent tickets.
3. Mode: 5v5, ranked competitive, TrueSkill-based.
4. Avg match duration: 18 minutes.
5. 7 matchmaking regions (NA-East/West, EU-West/East, APAC-East/SEA, SA).
6. Parties average 1.6 players.
7. Client already authenticated (auth upstream, out of scope).
8. DGS fleet is a separate system — matchmaking treats allocation as a downstream call.
9. Skill rating stored as (mu, sigma) TrueSkill pair per player per mode.
10. Bot-fill or cross-region-relax acceptable after 45s with no balanced match.

## 6. Capacity Estimation

**Ticket ingestion:** 500K concurrent tickets ÷ ~15s avg queue residency ≈ **33,000 tickets/sec** globally; ~11,500/sec in the peak-skew region.

**Matchmaker tick throughput:** Tick loop (~1s) per region-playlist shard. Busiest shard pool ≈ 11,500/sec × 15s ≈ **~170K resident tickets**. Bucketed skill-band + region indexing keeps a matching pass at ~3M ops/tick — trivial per core; shard by skill-decile (10 shards × 17K tickets) to parallelize.

**Storage:**
- Skill profiles: ~120B/row × 40M MAU × 5 modes ≈ **24 GB** (fits in-memory KV, e.g., Redis/DynamoDB DAX).
- Match history: ~4.4M matches/day × ~2KB ≈ **9 GB/day**, 90-day rolling ≈ 800 GB in columnar store.
- Ticket queue state: 500K × ~500B ≈ **250 MB** in-memory, replicated for durability.

**Compute:** TrueSkill update is closed-form Bayesian (factor-graph), sub-ms, **no GPU**. Matching optimization is CPU-bound bucket search. Sizing: 7 regions × ~10 shards × 2 vCPU ≈ **140 vCPUs** at peak, 2x for HA ≈ **280 vCPUs**. Rating-update service: ~514 updates/sec, handled by ~8 vCPUs.

## 7. High-Level Architecture

```
                                   ┌─────────────────────┐
                                   │   Game Client        │
                                   └──────────┬───────────┘
                                              │ gRPC/HTTPS (JWT)
                                   ┌──────────▼───────────┐
                                   │  API Gateway / Edge   │
                                   └──────────┬───────────┘
                       ┌──────────────────────┼───────────────────────┐
             ┌─────────▼─────────┐  ┌─────────▼─────────┐   ┌─────────▼─────────┐
             │ Ticket Service      │  │ Match-Status API   │   │ Skill Rating API   │
             │ (submit/cancel)     │  │ (poll/stream ETA)  │   │ (read profile)     │
             └─────────┬─────────┘  └─────────┬─────────┘   └─────────┬─────────┘
                       ▼                        ▼                       ▼
             ┌─────────────────────────────────────────────────────────────┐
             │  Ticket Store (Redis Cluster, region-sharded)                 │
             │  + Durable WAL (Kafka topic: matchmaking.tickets)             │
             └─────────┬───────────────────────────────────────────────────┘
                       ▼
             ┌─────────────────────────────────────────────────────────────┐
             │   Matchmaker Workers (per region × playlist × skill-shard)   │
             │   - bucket by skill decile + region latency group            │
             │   - tick loop: greedy pairing + local-search balancing       │
             │   - progressive constraint relaxation                        │
             └─────────┬───────────────────────────────────┬──────────────┘
                       │ match formed                       │ no match yet
                       ▼                                     ▼ (re-enqueue, widen)
             ┌─────────────────────┐               ┌─────────────────────┐
             │ Match Assembly &      │               │  Ticket Store (loop) │
             │ DGS Allocation Svc    │               └─────────────────────┘
             └─────────┬───────────┘
                       ▼
             ┌─────────────────────┐        ┌─────────────────────────┐
             │ DGS Fleet (external) │◄──────►│ Session/Lobby Service    │
             └─────────┬───────────┘        └─────────────────────────┘
                       │ match result event
                       ▼
             ┌─────────────────────────────────────────────────────────────┐
             │  Kafka: matchmaking.match_results                             │
             └─────────┬───────────────────────────────────┬───────────────┘
                       ▼                                     ▼
             ┌─────────────────────┐               ┌─────────────────────────┐
             │ Rating Update Svc     │               │ Telemetry / Analytics     │
             │ (TrueSkill recompute) │               │ Pipeline (drift, fairness)│
             └─────────┬───────────┘               └─────────────────────────┘
                       ▼
             ┌─────────────────────┐
             │ Skill Profile Store   │
             │ (KV, region-replicated)│
             └─────────────────────┘
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| Ticket Service | Validate/admit requests; one-active-ticket-per-player | gRPC `SubmitTicket`, `CancelTicket` | Stateless, scales on QPS |
| Ticket Store | Durable, low-latency active-ticket store | Redis hash + Kafka WAL | Sharded by region + playlist |
| Matchmaker Worker | Core pairing algorithm, owns a shard | Tick loop, consumes ticket stream | One worker group per shard |
| Match Assembly Svc | Converts proposed group into confirmed match, requests DGS | gRPC `AllocateMatch` | Stateless, scales with matches/sec |
| Skill Rating Service | Owns TrueSkill mu/sigma, applies post-match updates | gRPC `GetRating`, consumer `UpdateRating` | Stateless + sharded KV backing store |
| Match-Status API | Client polls/streams ETA and final assignment | gRPC-streaming `WatchTicket` | Stateless, scales on watchers |
| Backfill Service | Detects mid-match player loss, finds replacement | Kafka consumer `session.player_left` | Stateless, low volume |
| Telemetry/Fairness Pipeline | Aggregates balance, wait-time, skill-gap metrics | Kafka consumer → stream jobs → warehouse | Scales with match volume |

## 9. API Design

```
POST /v1/matchmaking/tickets
Request:
{ "player_id": "p_9f2a...", "party_id": "party_113" | null,
  "playlist": "ranked_5v5_control", "region_pref": ["NA-EAST","NA-WEST"],
  "max_ping_ms": 80, "role_pref": "DPS" | null, "client_version": "3.4.1" }
Response: 202 Accepted { "ticket_id": "t_88ac...", "estimated_wait_s": 22 }

GET /v1/matchmaking/tickets/{ticket_id}   (or gRPC streaming WatchTicket)
Response:
{ "state": "SEARCHING"|"MATCHED"|"CANCELLED"|"EXPIRED", "estimated_wait_s": 14,
  "match": { "session_id":"s_1029...", "dgs_endpoint":"10.4.2.9:7777",
             "team":["p_1","p_2",...], "opponents":["p_11",...] } | null }

DELETE /v1/matchmaking/tickets/{ticket_id}   → 200 OK { "state": "CANCELLED" }

GET /v1/skill/{player_id}?playlist=ranked_5v5_control
Response: { "mu": 28.4, "sigma": 3.1, "conservative_rating": 19.1, "games_played": 412 }

POST /internal/v1/matches/{session_id}/result   (DGS → platform)
Request: { "winner_team": "A", "roster": [...], "leave_events": [...] }
```

- URI-versioned, additive-only evolution; breaking changes get `/v2/` with dual-run.
- Streaming (`WatchTicket`) preferred over polling at 33K tickets/sec scale.

## 10. Database Design

| Data | Store | Why | Partition Key |
|---|---|---|---|
| Active tickets | Redis Cluster | Sub-ms read/write, TTL, needed for tick loop | `region:playlist` |
| Ticket durability log | Kafka `matchmaking.tickets` (compacted) | Replay on crash; source of truth | `region:playlist` |
| Skill profiles | DynamoDB/Cassandra | High read QPS, simple key access, multi-region | `player_id` / `playlist` |
| Match history | Columnar store | Analytical queries, retraining features | `match_date`, clustered by `playlist` |
| Session/DGS allocation | Postgres | ACID — server assigned exactly once | Sharded by region |

- Skill profiles: eventual consistency with read-your-writes for the updating player.
- Ticket store: strong consistency within a shard via Redis atomic Lua CAS (`SEARCHING → MATCHED`).

## 11. Caching

| Cache | Content | Strategy | Invalidation |
|---|---|---|---|
| L1: Skill profile (in-process) | Hot ratings for queued tickets | Cache-aside, read-through | TTL 60s + event-driven on `rating.updated` |
| L2: Region latency map | Player → DGS ping estimates | Write-through from client probes | TTL 24h |
| Match-status cache | Last-known ticket state | Cache-aside in Redis | Event-driven on state transition |
| Playlist config | Relax-curve tunables | Cache-aside, config-service pull | On config push |

No cache for the ticket pool itself — Redis is the live store, not a cache of one.

## 12. Queues & Async Processing

| Queue | Purpose | Delivery | DLQ |
|---|---|---|---|
| `matchmaking.tickets` | Durable log of submit/cancel, replay source | At-least-once, idempotent via ticket_id+version | Malformed → DLQ + alert |
| `matchmaking.match_results` | Outcomes for rating + telemetry | At-least-once, dedupe on session_id | Retry w/ backoff, then DLQ + page if backlog > 5min |
| `session.player_left` | Triggers backfill | At-least-once | No replacement in 10s → bot-fill fallback |
| `rating.updated` | Cache invalidation fan-out | At-most-once (TTL is backstop) | N/A |

Exactly-once *effective* semantics via idempotency keys, not broker guarantees (which don't hold across the DGS boundary anyway). DLQs alert on depth > 100 or age > 5 min.

## 13. Streaming & Event-Driven Architecture

| Topic | Key Fields | Producers | Consumers |
|---|---|---|---|
| `matchmaking.tickets` | ticket_id, player_id, party_id, playlist, constraints, op | Ticket Service | Matchmaker Workers, Analytics |
| `matchmaking.match_results` | session_id, playlist, roster[], pre_mu/sigma[], outcome | Match Assembly Svc | Rating Update Svc, Fairness Analytics, Warehouse |
| `session.player_left` | session_id, player_id, reason, ts | Session/Lobby Service | Backfill Service |
| `rating.updated` | player_id, playlist, mu, sigma, delta | Rating Update Svc | Cache invalidators, Profile Service |
| `queue.health` | region, playlist, pool_size, avg/p99 wait | Matchmaker Workers | Autoscaler, Dashboard, Alerting |

Consumer groups partitioned identically to producers (`region:playlist`) to preserve ordering. Schema evolution via Avro/Protobuf registry, additive-only within a major version.

## 14. Model Serving

- Core "model" is the **TrueSkill Bayesian update** — closed-form, no neural net, no GPU serving framework needed.
- Optional ML: a **match-quality predictor** (small gradient-boosted tree) ranks candidate groupings before final selection.
  - Loaded in-process into matchmaker worker memory (<5MB, <1ms inference) — avoids a network hop within the tight tick budget.
  - No batching needed; tens of candidates per tick, CPU-bound.
- Separate model per playlist (ranked vs casual quality differs); hot-reloaded from registry every 6h, swapped at tick boundary.

## 15. Feature Store

- **Online**: skill (mu, sigma), recent streaks, party-composition history — served from the same KV skill-profile store.
- **Offline**: match history + engineered features (skill-gap distributions, wait-vs-skill correlation) in the warehouse, used for retraining and fairness audits.
- **Point-in-time correctness**: training joins each match's *pre-match* mu/sigma snapshot, not current rating — using current rating would leak future information via rating drift.
- Online freshness must reflect the last completed match, hence event-driven cache invalidation rather than batch ETL.

## 16. Vector Database

**N/A.** Match quality here is defined over low-dimensional structured features (mu, sigma, ping, party, role) — skill-band bucketing and numeric range queries beat ANN search on both latency and explainability (fairness audits need "why was I matched with X," which embedding distance doesn't give). A vector DB would only apply if adding playstyle/social-similarity matchmaking — out of scope.

## 17. Embedding Pipelines

**N/A**, same reasoning as Section 16 — no unstructured/high-cardinality data in the core matching decision. A future toxicity/playstyle-aware extension would need one; not in scope here.

## 18. Inference Pipeline (Request Lifecycle)

```
t=0ms    Client → Ticket Service: SubmitTicket(player, playlist, region_pref)
t=5ms    Fetch skill rating (cache-aside, ~90% L1 hit)
t=8ms    Write ticket to Redis (state=SEARCHING) + emit Kafka event
t=10ms   202 Accepted with estimated_wait_s
         --- client opens WatchTicket stream ---
t=1000ms Matchmaker tick (shard: NA-EAST, ranked_5v5, decile-6):
           - loads pool (bucketed by skill-band + region)
           - greedy-pairs candidate 5v5 groupings within skill-window
           - scores top-K via match-quality model (<1ms/candidate)
           - selects best grouping meeting skill-gap + ping thresholds
t=1020ms Match Assembly Svc: reserves DGS session (idempotent)
t=1080ms Writes match assignment, ticket state=MATCHED
t=1090ms Kafka emit match_results (pending); client notified via WatchTicket push
t=1100ms Client connects to DGS
         --- match plays ~18 min ---
t=+18min DGS posts result → match_results (final) → Rating Update Svc
         → TrueSkill recompute (<1ms/player) → new mu/sigma → rating.updated
```

p50 target of 15s is dominated by **pool-fill wait**, not compute (single-digit ms).

## 19. Training Pipeline

- **TrueSkill parameters** (default sigma, decay, draw-probability) are config-tuned, not trained — validated via offline replay against historical logs (log-loss on win-probability).
- **Match-quality model**:
  - Data: `match_results` + point-in-time skill snapshots; label = score-differential closeness or churn-after-stomp.
  - Features: skill-gap, sigma-uncertainty sum, party-size mismatch, ping-variance.
  - Orchestration: nightly Airflow/Kubeflow DAG, single-node LightGBM (minutes, low-dimensional — no distributed training needed).
  - Validation: temporal holdout (most recent 3 days), not random split, to avoid rating-drift leakage.
  - Registry: versioned S3 artifact, canary into 5% of shards before full rollout.

## 20. Retraining Strategy

| Trigger | Action |
|---|---|
| Scheduled | Nightly retrain on rolling 90-day window |
| Data drift | Ad-hoc retrain if skill-gap feature distribution shifts past threshold |
| Concept drift | Retrain if calibration degrades beyond threshold |
| New playlist launch | Heuristic-only scoring until ≥2 weeks of data, then first model |
| Major balance patch | Force retrain within 48h |
| TrueSkill hyperparameters | Reviewed quarterly via offline replay, not continuously retrained |

## 21. Drift Detection

| Drift Type | Metric | Threshold |
|---|---|---|
| Data drift — skill distribution | PSI on mu, daily vs 30-day baseline | PSI > 0.2 |
| Data drift — region pool density | Wasserstein distance, hourly vs prior week | >25% shift, 3+ days |
| Concept drift — match-quality model | Rolling 7-day calibration error | Brier degradation > 15% |
| Fairness drift | Win-rate variance by region/party cohort | Outside 45-55% sustained over 10K+ matches |
| Rating inflation | Mean mu drift unexplained by skill change | >2 sigma-units/quarter |

All drift jobs feed a shared `drift_metrics` table wired to Monitoring/Alerting.

## 22. Monitoring

| Category | Metrics |
|---|---|
| Infra | Worker CPU/mem per shard, Redis hit ratio/latency, Kafka consumer lag, DGS allocation success rate |
| Model quality | Match-quality calibration, skill-gap distribution, TrueSkill sigma convergence |
| Business/product | p50/p99 wait per region/playlist, abandon rate, post-match churn, win-rate balance, backfill success |
| Queue health | Live pool size per shard, tickets/sec in vs matches formed/sec |
| Fairness | Skill-gap p95 per match, cross-region-match rate, party-vs-solo win-rate delta |

Dashboards segmented by region × playlist — a global aggregate hides regional pool-starvation (e.g., SEA off-peak).

## 23. Alerting

| Alert | Condition | Routing |
|---|---|---|
| Queue backlog growing | Pool growth, 5 consecutive ticks, no matches formed | Page on-call (P1) |
| p99 wait breach | p99 > 90s for 3 min in any region-playlist | Page on-call (P2) |
| Kafka consumer lag | Lag > 30s on tickets/results topics | Page on-call (P1) |
| DGS allocation failure spike | Failure rate > 5% over 2 min | Page on-call + DGS team (P1) |
| Fairness cohort breach | Cohort win-rate outside 45-55% over 10K matches | Ticket to fairness team (P3) |
| Model calibration degradation | Brier regression > 15% | Ticket to ML team, ad-hoc retrain (P3) |
| Double-booking detected | Ticket matched twice | Immediate page (P0) — correctness bug |

## 24. Logging

- Structured JSON at every state transition (`ticket_submitted`, `ticket_matched`, `rating_updated`, etc.) with `trace_id`, hashed `player_id`, region, playlist, timestamp.
- PII: raw player_id/IP-derived region classified as PII — general logs use salted hash; only privileged tooling can reverse, with audited access.
- Retention: operational logs 30 days hot, 1 year cold archive. Match/rating history retained per account-lifecycle policy (GDPR/CCPA deletion/anonymization on request).
- Sensitive fields (exact IP, device fingerprint) never logged at INFO.

## 25. Security

- **Rating manipulation/boosting**: colluding accounts gaming outcomes — mitigated by anomaly detection on win-rate-vs-expected patterns (feeds anti-cheat).
- **Queue manipulation**: client spoofing region/ping — mitigated by server-side RTT verification and server-side skill lookup (client can't pass its own mu/sigma).
- **Denial of queue**: bot mass-ticket submission — mitigated via per-account rate limiting and reputation scoring at the gateway.
- **Double-booking**: race allowing a player into two matches — mitigated by atomic CAS on ticket state (Redis Lua), single-writer-per-ticket invariant.
- TLS 1.3 in transit, AES-256 at rest, mTLS between Kafka brokers/clients.
- Skill/match data is access-controlled account-linked data, not exposed raw to third parties without anonymization.

## 26. Authentication

- End-user: JWT from EA identity service validated at the gateway on every request; `player_id` extracted from claims, never trusted from payload.
- Service-to-service: mTLS with SPIFFE/SPIRE-style workload identity.
- DGS callback: scoped short-TTL service token, plus session_id must match an outstanding allocation record — prevents forged results.

## 27. Rate Limiting

- Token bucket per player_id at the gateway (e.g., 5 submits/min burst) — submit is inherently low-frequency.
- Separate per-title token-bucket pools if the platform is shared across EA titles.
- Reputation-flagged accounts get a stricter bucket pending review.
- `WatchTicket` limited by max concurrent streams (1 per player), not request rate.

## 28. Autoscaling

- **Matchmaker workers**: KEDA on Kafka consumer lag for `matchmaking.tickets` — add shards when lag > 5s sustained, scale down when pool < 20% capacity for 10 min.
- **Ticket Service/Match-Status API**: standard HPA on CPU (60% target) and latency SLO burn.
- **Skill Rating Service**: HPA on request QPS, tied to `match_results` consumer lag.
- **VPA** (recommendation-only) for matchmaker worker memory right-sizing.
- Predictive pre-scaling ahead of known daily regional peak windows to avoid reactive-scaling lag.

## 29. Cost Optimization

- Spot instances for matchmaker workers — stateless-recoverable (ticket state lives in Redis+Kafka), so interruption just triggers shard rebalance.
- Batched tick processing amortizes compute, avoids O(n²) pairwise costs.
- Skill-profile cache cuts ~90% of reads from the KV store.
- Deliberately small tree model (not deep net) — avoids GPU spend for near-equal lift.
- Off-peak regions shed dedicated shard capacity into a shared low-traffic pool.
- DGS pre-warm size tuned against observed CCU curve (idle cost vs cold-start latency).

## 30. Operational Concerns

At SDE2 scope, treat this as a checklist: **backups** (model registry, feature store, tested restore path), **rollback** (one-command revert to last-known-good), **canary/blue-green rollout** (small traffic shift, watch error rate + key metrics, then ramp), **basic observability** (latency/error-rate/model-quality dashboards wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level concerns — worth knowing they exist, not worth rehearsing.

## Why This Architecture

- **Region-sharded, tick-based matchmaker** matches EA's reality of geographically distributed players with different peak times, avoiding a single global bottleneck.
- **Event-driven backbone (Kafka)** decouples ingestion, matching, and rating updates so each scales/fails independently, with durability free via the log.
- **Redis + Kafka WAL** gives sub-100ms tick-loop reads while surviving process failure — pure-Kafka is too slow for tick reads, pure-in-memory loses tickets on crash.
- **No-GPU/small-model philosophy**: the core problem (Bayesian update, bin-packing) doesn't need deep learning; a small tree model captures non-linear quality signal at low cost.
- **Active-active multi-region** is required by the non-negotiable real-time latency requirement — no cross-ocean matching is acceptable for a shooter.

## Alternative Architectures

| Alternative | Why Rejected / When Preferred |
|---|---|
| Centralized global matchmaker (no regional sharding) | Cross-region ping makes "optimal" matches unplayable; sharding is strictly better. Preferred only for async/turn-based games. |
| Synchronous request-response matchmaking | Doesn't survive network blips, hard to scale held-connection state, can't support progressive relaxation cleanly. Fine for small-scale prototypes. |
| Deep-learning match-quality model | GPU cost/latency for marginal lift, hurts explainability for fairness audits. Justified only if social/toxicity-aware matching becomes a hard requirement. |
| Fully decentralized lobby-browser | No fairness guarantee, poor UX at high CCU, no central skill-balance/anti-cheat enforcement. Used in older/smaller titles only. |

## Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Progressive constraint relaxation | Bounds worst-case wait, keeps queue moving in low-pop windows | Later matches less balanced/higher-ping |
| Region-local pools (no cross-region merge by default) | Best ping, no cross-region coordination cost | Thin pools in low-pop regions/hours |
| TrueSkill vs Elo | Faster new-player convergence, native team support | Harder to explain to players, costlier to audit |
| Party support | Better social UX/retention | Reduces fairness (party synergy not captured in rating) |
| Small tree-based quality model vs heuristic-only | Captures non-linear signal, incremental lift | Added deploy/retrain/drift-monitoring surface |
| At-least-once + idempotency vs exactly-once broker | Simpler, broker-agnostic | Requires careful idempotency-key discipline everywhere |

## Failure Modes

| Scenario | Impact | Mitigation |
|---|---|---|
| Redis ticket-store shard outage | In-flight tickets in that pool lose live state | Kafka WAL replay rebuilds pool on failover within seconds |
| Kafka broker/partition unavailable | Ingestion/result processing stalls | Multi-AZ replication factor 3; producer retry; lag alert at >30s |
| Matchmaker shard stuck/deadlocked | Pool stops forming matches, tickets accumulate | `queue.health` alert within 1-2 ticks; liveness-probe restart; sibling shard absorbs overflow |
| DGS fleet exhausted | Match formed but no server to allocate | Circuit-breaker re-queues with priority boost; alert DGS team |
| Skill Rating store stale/unavailable | Fairness degrades temporarily | Fail-open (assume average skill) over fail-closed |
| Double-booking race | Player pulled into two sessions | Atomic Redis CAS eliminates race; P0 alert + kill-switch as backstop |
| Cascading region failover | Neighbor region's pool overwhelmed | Cap fraction of foreign-region traffic accepted (graceful shed) |

## Scaling Bottlenecks

- **At 10x CCU (5M concurrent, 330K tickets/sec)**: Redis/Kafka shard and partition counts grow proportionally (fine, but rebalancing overhead grows); Skill Rating Service read QPS scales linearly, making cache hit-ratio more critical.
- **At 100x CCU (50M concurrent, likely only relevant if platform-shared across titles)**: the tick-loop pool size per shard is the real break point — needs either far more shards (more cross-shard edge cases) or a streaming/incremental matching algorithm instead of full-pool rescans. DGS fleet provisioning speed becomes the bottleneck before matching logic does. Multi-tenancy across titles would need per-title isolation, not one shared cluster.

## Latency Bottlenecks

**p50 ≈ 15s** (dominated by pool-wait, not compute):
| Stage | Time |
|---|---|
| Ticket submit → accepted | ~10-15ms |
| Wait for tick cycle | 0-1000ms |
| Pool-fill wait | **~13-14s (dominant)** |
| Candidate scoring | <5ms |
| Match Assembly + DGS reservation | ~60-100ms |
| Client connects to DGS | ~100-300ms |

**p99 ≈ 60s**: same fixed costs, but pool-fill wait dominates further in low-population moments as progressive relaxation steps through several multi-second waits.

>95% of both p50 and p99 latency is inherent to supply (are there enough compatible players), not system compute — the architecture's job is to not add meaningfully on top of that.

## Cost Bottlenecks

- Matchmaker worker floor capacity (minReplicas per shard for availability) held 24/7 even off-peak — biggest lever is off-peak shard consolidation.
- Skill Rating KV read/write capacity at 33K/sec ticket rate is the largest recurring cost without the cache layer — hit-ratio is the dominant lever.
- Kafka sizing (brokers, retention) driven by peak ingestion × retention window — longer replay/audit retention trades directly against storage cost.
- Cross-region skill-profile replication cost scales with write rate × replica count (4.4M matches/day × multi-region fanout).
- DGS warm-pool cost is a separate system's line item, but matchmaking's latency requirements directly dictate its size.

## Interview Follow-Up Questions

1. How would you handle a party of 5 queuing against solo players in 5v5 — fairness implication and mitigation?
2. What happens if the match-quality model starts favoring one region's playstyle after a retrain — how do you catch it?
3. Why TrueSkill over Elo, and what does it give you that Elo fundamentally can't?
4. How do you prevent a player from being matched into two sessions simultaneously in a sharded system?
5. p99 wait SLA is breached only in one low-population region at 3am local — what do you actually do?
6. How would this design change for battle royale (100 players) instead of 5v5?
7. How do you detect and handle skill-rating manipulation (queue-dodging, boosting)?
8. If you had to support cross-play (PC + console), how does that change matching constraints?
9. How do you validate the progressive relaxation curve is well-tuned rather than guessed?
10. What would change if the game were turn-based (async) instead of real-time?

## Ideal Answers

1. **Party vs solo fairness**: Apply a party-strength adjustment — inflate effective team rating using a calibrated multiplier from historical party-vs-solo win-rate deltas, and prefer matching parties against similarly-sized parties. Track party-vs-solo win-rate as an explicit fairness metric and recalibrate periodically.

2. **Regional playstyle bias post-retrain**: Segment calibration/Brier-score eval by region before rollout, and gate canary promotion on per-cohort parity, not just the global average.

3. **TrueSkill vs Elo**: TrueSkill models rating as a distribution (mu, sigma), expressing confidence so new players (high sigma) converge faster, and its factor-graph decomposes team outcomes into per-player contributions. Elo is pairwise-native and needs ad hoc hacks for team games.

4. **Preventing double-booking**: Atomic, single-writer ticket-state transitions via a Redis Lua check-and-set (`SEARCHING → MATCHED`) so racing shards can't both claim a ticket — a data-layer correctness invariant, not app-level coordination.

5. **Off-peak regional SLA breach**: Confirm it's a genuine supply problem via `queue.health`, not a bug. Then widen the relaxation curve for that region-hour, allow cross-region matching, or bot-fill as last resort.

6. **100-player battle royale**: Shifts from balanced-team pairing to lobby-filling/bin-packing — fill-to-capacity with skill-tier stratification instead of greedy-pair-and-balance. Pool-fill wait dominates even more, making partial-lobby bot-fill timeouts standard.

7. **Rating manipulation detection**: Primarily an anti-cheat/trust-and-safety problem matchmaking feeds signal into — look for statistical anomalies like repeated same-party win/loss patterns inconsistent with skill deltas. Matchmaking surfaces signals to an integrity pipeline rather than unilaterally banning.

8. **Cross-play PC + console**: Add input-method as an explicit matching dimension (separate mu/sigma sub-pools or lobby composition constraints), since aim-assist and precision differ. Fits as another configurable dimension in the same progressive-constraint framework.

9. **Validating the relaxation curve**: Offline replay against historical ticket-arrival logs with candidate curves, measuring resulting wait-time and fairness distributions. A/B test candidates live via canary, gated on both metrics simultaneously.

10. **Turn-based/async redesign**: Real-time constraints (tick loops, ping-aware routing, live DGS allocation) largely disappear — matching becomes a slower batch process over much larger pools, and region/ping stops being a constraint. Collapses the real-time streaming backbone into a simpler scheduled batch design.
