# Matchmaking Engine

## 1. Problem Framing & Requirement Gathering

Design a skill-based matchmaking system for an EA live-service title (e.g., a squad-based shooter or sports title) that pairs players into balanced matches within acceptable wait times, across regions, at global scale.

- Core tension: **fairness (skill parity)** vs **latency (queue wait time)** vs **network quality (region/ping)**.
- Real-time system: players are actively waiting; every second of queue time is player-experience cost.
- Not a recommendation system — it's a constrained optimization + real-time assignment problem running continuously against a live pool of waiting players.
- Interfaces with: game client, session/lobby service, dedicated game server (DGS) fleet, anti-cheat, telemetry pipeline.

## 2. Functional Requirements

- FR1: Accept a matchmaking request (solo or party) with player skill rating, region, party constraints, and game-mode.
- FR2: Group players into match-sized parties (e.g., 10 players for 5v5) satisfying skill-balance and latency constraints.
- FR3: Support party/premade groups — parties queue and match together, matched to opposing parties.
- FR4: Support role-based matchmaking (e.g., tank/support/DPS) where applicable.
- FR5: Expand search criteria progressively over time (skill band widens, latency threshold relaxes) to bound wait time.
- FR6: Return a match assignment: session ID, DGS endpoint, teammates, opponents.
- FR7: Update skill ratings post-match (TrueSkill/Elo update) from match results.
- FR8: Allow cancel-search / requeue.
- FR9: Support backfill — replace a player who disconnects mid-match.
- FR10: Provide matchmaking health/ETA estimate to client ("~25s wait").

## 3. Non-Functional Requirements (latency, availability, throughput, consistency, cost)

| Attribute | Target |
|---|---|
| Match-found latency (p50) | ≤ 15s |
| Match-found latency (p99) | ≤ 60s (else force-relax constraints or bot-fill) |
| Ticket ingestion latency | ≤ 50ms p99 (enqueue ack to client) |
| Availability | 99.95% (queue service is critical path — outage = can't play) |
| Throughput | Sustain 500K concurrent searching players at peak (global) |
| Consistency | Strong consistency on "player assigned to exactly one match" (no double-booking); eventual consistency OK for skill-rating propagation |
| Cost | Matchmaker compute must scale sub-linearly with CCU via batched ticks, not per-player threads |
| Durability | Ticket state survives matchmaker pod restart (no lost-in-queue players) |

## 4. Clarifying Questions an Interviewer Would Expect You to Ask

1. Team size and mode — 1v1, 5v5, battle royale (60-100 players)? Changes matching complexity from O(n log n) pairing to bin-packing.
2. Is cross-region matchmaking acceptable if it reduces wait time, trading off latency?
3. Do parties of mixed skill get averaged, capped by highest, or restricted to narrow internal skill spread?
4. Is there a ranked vs unranked split with different fairness tolerances?
5. What's the acceptable skill-rating system — Elo (simple, chess-style) vs TrueSkill (Bayesian, handles teams/uncertainty)?
6. Should the system support role queues / composition constraints?
7. Is there a smurf/new-account detection requirement feeding into initial rating?
8. What's the DGS fleet provisioning model — pre-warmed pool or on-demand spin-up (affects match-to-play latency)?
9. Backfill requirements — how is a mid-match leaver replaced, and within what time budget?
10. Regional player density — do low-population regions (e.g., SEA at 3am) need special handling (bot-fill, relaxed skill bands)?

## 5. Assumptions (Explicit, Numbered)

1. Title has 40M MAU, 6M DAU, peak concurrent players (CCU) of 2M globally, peak-region skew ~35% in one region during its evening.
2. 25% of DAU actively matchmaking at any peak moment → ~500K concurrent searching tickets at peak (matches NFR).
3. Mode is 5v5 (10 players/match), ranked competitive shooter, TrueSkill-based rating.
4. Average match duration: 18 minutes → match completion rate ≈ CCU-in-match / 18min.
5. Regions: NA-East, NA-West, EU-West, EU-East, APAC-East, APAC-SEA, SA — 7 matchmaking regions.
6. Parties average 1.6 players (mix of solo and premade groups up to 5).
7. Client already authenticated via EA Account/Origin identity before entering queue (auth handled upstream).
8. DGS fleet is a separate system (assume pool of warm servers is available; this chapter treats DGS allocation as a downstream call).
9. Skill rating stored as (mu, sigma) TrueSkill pair per player per mode/playlist.
10. Acceptable to bot-fill or cross-region-relax after 45s if no balanced human match found.

## 6. Capacity Estimation (QPS, Storage, Model Size, GPU/CPU Counts)

**Ticket ingestion QPS:**
- 500K concurrent tickets, avg ticket lifetime in queue ~15s → new-ticket arrival rate ≈ 500,000 / 15s ≈ **33,000 tickets/sec** globally at peak.
- Per-region peak (35% skew region): ~11,500 tickets/sec.

**Matchmaker tick throughput:**
- Matchmaker runs on a tick loop (e.g., every 1s) per region-playlist shard, evaluating the current pool.
- Pool size per shard at peak: 11,500 tickets/sec × 15s avg residency ≈ **~170K tickets** resident in the busiest region-playlist pool.
- Matching pass over 170K tickets using bucketed skill-band + region indexing: O(n log n) sort/bucket ≈ 170K × 17 ≈ 3M ops/tick — trivial for a single core in <100ms; shard further by skill-decile to parallelize (10 shards × 17K tickets each).

**Storage:**
- Player skill profile: player_id (8B), mu (4B float), sigma (4B float), mode_id (4B), games_played (4B), updated_at (8B) ≈ 40B logical, ~120B with overhead/indexing.
- 40M MAU × 5 modes tracked × 120B ≈ **24 GB** for skill-rating table (fits comfortably in memory-tier KV store, e.g., Redis/DynamoDB DAX).
- Match history (for retraining/audit): 2M CCU / 10 players-per-match × (18 min matches) → ~185K matches/hour → ~4.4M matches/day × ~2KB record (roster, ratings pre/post, outcome) ≈ **~9 GB/day**, retained 90 days rolling in columnar store ≈ 800 GB.
- Ticket queue state: 500K tickets × ~500B (constraints, party info, timestamps) ≈ **250 MB** in-memory — trivial, but must be replicated for durability.

**Model/compute:**
- TrueSkill update is a closed-form Bayesian computation (matrix-free, factor graph message passing) — **no GPU required**; pure CPU, sub-millisecond per match update.
- Matchmaking optimization (bucket search + local search for team balancing) — CPU-bound, no GPU.
- CPU sizing: 7 regions × ~10 shards/region × 2 vCPU per shard-worker (headroom for tick spikes) ≈ **140 vCPUs** globally for the core matching workers at peak, plus 2x for HA/failover ≈ **280 vCPUs** provisioned.
- Rating-update service: event-driven, ~185K matches/hour × 10 players ≈ 1.85M rating updates/hour ≈ 514/sec — a handful of small stateless workers (8 vCPUs total) suffice.

## 7. High-Level Architecture

```
                                   ┌─────────────────────┐
                                   │   Game Client        │
                                   └──────────┬───────────┘
                                              │ gRPC/HTTPS (JWT)
                                   ┌──────────▼───────────┐
                                   │  API Gateway / Edge   │
                                   │  (rate limit, authZ)  │
                                   └──────────┬───────────┘
                                              │
                       ┌──────────────────────┼───────────────────────┐
                       │                       │                       │
             ┌─────────▼─────────┐  ┌─────────▼─────────┐   ┌─────────▼─────────┐
             │ Ticket Service      │  │ Match-Status API   │   │ Skill Rating API   │
             │ (submit/cancel)     │  │ (poll/stream ETA)  │   │ (read profile)     │
             └─────────┬─────────┘  └─────────┬─────────┘   └─────────┬─────────┘
                       │ writes ticket          │ reads state           │ reads
                       ▼                        ▼                       ▼
             ┌─────────────────────────────────────────────────────────────┐
             │        Ticket Store (Redis Cluster, region-sharded)          │
             │        + Durable WAL (Kafka topic: matchmaking.tickets)      │
             └─────────┬───────────────────────────────────────────────────┘
                       │ consumed by
                       ▼
             ┌─────────────────────────────────────────────────────────────┐
             │   Matchmaker Workers (per region × playlist × skill-shard)   │
             │   - bucket by skill decile + region latency group            │
             │   - tick loop: greedy pairing + local-search balancing       │
             │   - progressive constraint relaxation (time-in-queue based)  │
             └─────────┬───────────────────────────────────┬──────────────┘
                       │ match formed                       │ no match yet
                       ▼                                     ▼ (re-enqueue, widen constraints)
             ┌─────────────────────┐               ┌─────────────────────┐
             │ Match Assembly &      │               │  Ticket Store (loop) │
             │ DGS Allocation Svc    │               └─────────────────────┘
             └─────────┬───────────┘
                       │ allocate session
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
| Ticket Service | Validate + admit search requests; enforce one-active-ticket-per-player | gRPC `SubmitTicket`, `CancelTicket` | Stateless; scales on request QPS |
| Ticket Store | Durable, low-latency store of active tickets | Redis hash + Kafka WAL | Sharded by region + playlist |
| Matchmaker Worker | Core pairing algorithm; owns a shard (region × playlist × skill-band) | Internal tick loop, consumes ticket stream | One worker group per shard; horizontal by shard count |
| Match Assembly Svc | Converts a proposed group of 10 tickets into a confirmed match; requests DGS | gRPC `AllocateMatch` | Stateless; scales with matches/sec |
| Skill Rating Service | Owns TrueSkill mu/sigma; applies post-match updates | gRPC `GetRating`, event-consumer `UpdateRating` | Stateless compute + sharded KV backing store |
| Match-Status API | Client polls/streams for ETA and final assignment | gRPC-streaming `WatchTicket` | Stateless; scales on concurrent watchers |
| Backfill Service | Detects mid-match player loss, finds replacement from queue | Kafka consumer on `session.player_left` | Stateless; low volume |
| Telemetry/Fairness Pipeline | Aggregates match balance, wait-time, skill-gap metrics | Kafka consumer → stream jobs → warehouse | Scales with match volume |

## 9. API Design

```
POST /v1/matchmaking/tickets
Request:
{
  "player_id": "p_9f2a...",
  "party_id": "party_113" | null,
  "playlist": "ranked_5v5_control",
  "region_pref": ["NA-EAST", "NA-WEST"],
  "max_ping_ms": 80,
  "role_pref": "DPS" | null,
  "client_version": "3.4.1"
}
Response: 202 Accepted
{ "ticket_id": "t_88ac...", "estimated_wait_s": 22 }

GET /v1/matchmaking/tickets/{ticket_id}   (or gRPC server-streaming WatchTicket)
Response:
{
  "state": "SEARCHING" | "MATCHED" | "CANCELLED" | "EXPIRED",
  "estimated_wait_s": 14,
  "match": {
      "session_id": "s_1029...",
      "dgs_endpoint": "10.4.2.9:7777",
      "team": ["p_1","p_2",...],
      "opponents": ["p_11",...]
  } | null
}

DELETE /v1/matchmaking/tickets/{ticket_id}
Response: 200 OK { "state": "CANCELLED" }

GET /v1/skill/{player_id}?playlist=ranked_5v5_control
Response: { "mu": 28.4, "sigma": 3.1, "conservative_rating": 19.1, "games_played": 412 }

POST /internal/v1/matches/{session_id}/result   (server-to-server, DGS → platform)
Request: { "winner_team": "A", "roster": [...], "leave_events": [...] }
Response: 200 OK
```

- **Versioning**: URI-versioned (`/v1/`), additive-only field evolution; breaking changes ship as `/v2/` with dual-run period ≥ 2 release cycles, gated by client_version.
- Client-facing API favors streaming (`WatchTicket`) over polling to cut load at 33K tickets/sec scale.

## 10. Database Design

| Data | Store | Why | Partition/Shard Key |
|---|---|---|---|
| Active tickets | Redis Cluster (in-memory KV) | Sub-ms read/write, TTL support, needed for real-time tick loop | `region:playlist` hash slot |
| Ticket durability log | Kafka topic `matchmaking.tickets` (compacted) | Replay on matchmaker crash/restart; source of truth for "did we lose this ticket" | Partitioned by `region:playlist` |
| Skill profiles | DynamoDB / Cassandra (wide-column KV) | High read QPS (every ticket submit reads rating), simple key access pattern, multi-region replication | Partition key `player_id`, sort key `playlist` |
| Match history | Columnar store (e.g., Redshift/BigQuery equivalent) | Analytical queries for fairness audits, retraining feature extraction | Partitioned by `match_date`, clustered by `playlist` |
| Session/DGS allocation | Relational (Postgres) — small table, needs strong consistency for "server assigned once" | ACID guarantee against double-allocation | Sharded by region |

- Skill profile store uses **eventual consistency with read-your-writes** for the updating player (session affinity) — acceptable since ratings change slowly relative to matchmaking cadence.
- Ticket store requires **strong consistency within a shard** (no double-matching a ticket) — enforced via Redis atomic Lua script (check-and-set ticket state `SEARCHING → MATCHED`).

## 11. Caching

| Cache | Content | Strategy | Invalidation |
|---|---|---|---|
| L1: Skill profile cache (in-process, matchmaker worker) | Hot player ratings for currently-queued tickets | Cache-aside, read-through on ticket submit | TTL 60s; explicit invalidate on rating-update event (pub/sub) |
| L2: Region latency map | Player → nearest DGS region ping estimates | Write-through from periodic client ping-probe results | TTL 24h, refreshed on client reconnect |
| Match-status cache | Last-known ticket state for polling clients | Cache-aside in Redis, same store as ticket state | Invalidated on every state transition (event-driven) |
| Playlist config (skill-band widening curve, relax schedule) | Static-ish tunables | Cache-aside, CDN/config-service pull | Invalidated on config push (versioned, few-minute propagation) |

- No cache for the ticket pool itself — it *is* the live store (Redis), not a cache of something else.
- Skill-profile cache invalidation is event-driven (via `rating.updated` Kafka topic) rather than TTL-only, to avoid matching players on stale post-match ratings.

## 12. Queues & Async Processing

| Queue | Purpose | Delivery Semantics | DLQ Handling |
|---|---|---|---|
| `matchmaking.tickets` (Kafka) | Durable log of submitted/cancelled tickets, replay source | At-least-once (consumer idempotent via ticket_id + state version) | Poison ticket (malformed) → DLQ topic, alert, manual/auto reprocess |
| `matchmaking.match_results` | Match outcomes for rating update + telemetry | At-least-once, consumers dedupe on `session_id` | Failed rating update → retry with backoff (5x), then DLQ + page on-call if backlog > 5min |
| `session.player_left` | Triggers backfill | At-least-once | If no replacement found in 10s, DLQ event triggers bot-fill fallback |
| `rating.updated` | Cache invalidation fan-out | At-most-once acceptable (cache TTL is backstop) | N/A — self-healing via TTL |

- Exactly-once *effective* semantics achieved via idempotency keys (ticket_id + monotonic version) rather than relying on broker-level exactly-once, which doesn't hold end-to-end across the DGS boundary anyway.
- Dead-letter queues are monitored with alert-on-depth (>100 messages or >5 min age triggers page).

## 13. Streaming & Event-Driven Architecture

**Topics:**

| Topic | Schema (key fields) | Producers | Consumer Groups |
|---|---|---|---|
| `matchmaking.tickets` | ticket_id, player_id, party_id, playlist, region_pref, constraints, ts, op(SUBMIT/CANCEL) | Ticket Service | Matchmaker Workers (per shard), Analytics |
| `matchmaking.match_results` | session_id, playlist, roster[], pre_mu/sigma[], outcome, duration_s | Match Assembly Svc / DGS callback | Rating Update Svc, Fairness Analytics, Data Warehouse Loader |
| `session.player_left` | session_id, player_id, reason(DISCONNECT/AFK/KICK), ts | Session/Lobby Service | Backfill Service |
| `rating.updated` | player_id, playlist, mu, sigma, delta, ts | Rating Update Svc | Cache invalidators, Player Profile Service |
| `queue.health` | region, playlist, pool_size, avg_wait_s, p99_wait_s, ts (emitted every tick) | Matchmaker Workers | Autoscaler, Fairness Dashboard, Alerting |

- Consumer groups are partitioned identically to producer partitioning (`region:playlist`) to preserve ordering guarantees needed for ticket state transitions.
- Schema evolution via Avro/Protobuf with a schema registry; backward-compatible additive fields only within a major version.

## 14. Model Serving

- The "model" here is primarily the **TrueSkill Bayesian update** — a closed-form/factor-graph computation, not a neural net. No traditional model-serving framework (Triton/TF-Serving) needed for the core skill update.
- Where ML *is* served: an optional **match-quality predictor** (gradient-boosted tree, e.g., predicts P(close match) or predicted churn-risk-if-stomped) used to *rank candidate groupings* before final selection.
  - Served via a lightweight in-process feature (LightGBM model loaded into matchmaker worker memory, <5MB, inference <1ms) — avoids network hop given the tight tick-loop latency budget.
  - No batching needed (single-row inference per candidate grouping, tens of candidates per tick).
  - No GPU — tree ensemble inference is CPU-bound and trivial at this scale.
- Multi-model: separate match-quality models per playlist (ranked vs casual have different quality definitions); hot-reloaded from model registry (S3 + version pointer) every 6h, no traffic interruption (swap on tick boundary).

## 15. Feature Store

- **Online store**: player skill (mu, sigma), recent-match streaks, party-composition history — served from the same KV (DynamoDB/Redis) skill profile store; read at ticket-submit and pre-match-quality-scoring time.
- **Offline store**: full match history + engineered features (skill-gap distributions, wait-time-vs-skill-band correlation, per-region pool density over time) in the columnar warehouse, used for retraining the match-quality model and for fairness audits.
- **Point-in-time correctness**: match-quality model training joins each historical match's *pre-match* mu/sigma snapshot (captured at match-result-event time, not current live rating) — critical, since using a player's *current* rating for a match from 3 months ago would leak future information (rating drift) into training labels.
- Feature freshness requirement for online path: rating must reflect the *last completed match*, not stale-by-hours data — hence event-driven cache invalidation (Section 11) rather than batch ETL for the online slice.

## 16. Vector Database

**N/A.** Matchmaking match quality is defined over low-dimensional structured features (mu, sigma, ping, party composition, role) — a handful of numeric fields, not high-dimensional embeddings requiring similarity search. Skill-band bucketing + numeric range queries on a KV/relational index outperform ANN search here on both latency and interpretability grounds (interviewers expect fairness explainability — "why was I matched with X" — which ANN embedding distance doesn't provide). If cosmetic/behavioral similarity matching (e.g., "play-style" embeddings for social matchmaking) were in scope, a vector DB (e.g., pgvector/Milvus) would apply — out of scope for skill-based competitive matchmaking.

## 17. Embedding Pipelines

**N/A**, for the same reason as Section 16 — no unstructured or high-cardinality categorical data requiring dense-vector representation in the core matching decision. (A future extension — play-style/toxicity-aware matchmaking using behavioral embeddings — would introduce an embedding pipeline over telemetry event sequences; not part of this chapter's scope.)

## 18. Inference Pipelines (Request Lifecycle End-to-End)

```
t=0ms    Client → Ticket Service: SubmitTicket(player, playlist, region_pref)
t=5ms    Ticket Service: fetch skill rating (cache-aside, L1 hit ~90% of time)
t=8ms    Ticket Service: write ticket to Redis (state=SEARCHING) + emit Kafka event
t=10ms   202 Accepted returned to client with estimated_wait_s (from queue.health cache)
         --- client opens WatchTicket stream, polls state ---
t=1000ms Matchmaker tick (shard: NA-EAST, ranked_5v5, skill-decile-6):
           - loads current pool (bucketed by skill-band + region)
           - greedy-pairs tickets into candidate 5v5 groupings within skill-window
           - scores top-K candidate groupings via match-quality model (<1ms/candidate)
           - selects best grouping meeting skill-gap + ping thresholds
t=1020ms Match Assembly Svc: reserves DGS session (idempotent allocate call)
t=1080ms Match Assembly Svc: writes match assignment to ticket records (state=MATCHED)
t=1090ms Kafka emit: match_results (pending outcome), players notified via WatchTicket push
t=1100ms Client receives match payload (session_id, dgs_endpoint, roster) → connects to DGS
         --- match plays for ~18 min ---
t=+18min DGS → Match Assembly Svc: POST result (winner, roster, leave events)
         → emits matchmaking.match_results (final) → Rating Update Svc consumes
         → TrueSkill recompute (<1ms/player) → writes new mu/sigma → emits rating.updated
```

- p50 end-to-end "found a match" target of 15s is dominated by **pool-fill wait** (waiting for enough compatible players), not compute — compute is single-digit ms.

## 19. Training Pipelines (Data Prep, Training Orchestration, Distributed Training if Relevant)

- **TrueSkill parameters** (default sigma, skill-decay-per-game, draw-probability) are largely fixed/config-tuned, not "trained" in the ML sense — validated via offline replay simulation against historical match logs (does the parameter set predict actual match outcomes well? — measured via log-loss on win-probability).
- **Match-quality model** (Section 14) training pipeline:
  - Data prep: pull `match_results` + point-in-time skill snapshots from warehouse (Section 15), label = actual match closeness (score differential normalized) or observed churn-after-stomp.
  - Feature engineering: skill-gap (|team_avg_mu_A - team_avg_mu_B|), sigma-uncertainty sum, party-size mismatch, ping-variance within lobby.
  - Training orchestration: batch job (Airflow/Kubeflow Pipelines DAG) — nightly extract → train LightGBM (single-node, seconds-to-minutes; dataset is millions of rows but low-dimensional, no distributed training cluster needed).
  - Validation: hold out most-recent 3 days of matches (temporal split, not random — avoid leakage from rating drift).
  - Model registry: push versioned artifact to S3-backed registry; canary-load into 5% of matchmaker shards before full rollout (Section 33).
- No distributed/multi-GPU training required — this is a small tabular model, not a deep net. If a future toxicity/play-style embedding model were added, that would warrant distributed training infra (out of scope here per Section 17).

## 20. Retraining Strategy (Cadence, Triggers)

| Trigger | Action |
|---|---|
| Scheduled | Retrain match-quality model nightly on rolling 90-day window |
| Data drift alert (Section 21) | Ad-hoc retrain if skill-gap feature distribution shifts > threshold (e.g., new mode launch changes population) |
| Concept drift alert | Retrain if model's predicted-vs-actual match-closeness calibration degrades beyond threshold |
| New playlist/mode launch | Cold-start with heuristic-only quality scoring (no ML) until ≥ 2 weeks of match data accumulated, then train first model version |
| Major balance patch | Force retrain within 48h — patch changes gameplay balance, invalidating prior quality-model assumptions |
| TrueSkill hyperparameters | Reviewed quarterly via offline replay simulation, not continuously retrained |

## 21. Drift Detection (Data Drift, Concept Drift, What Metrics, What Thresholds)

| Drift Type | Metric | Threshold | Detection Method |
|---|---|---|---|
| Data drift — skill distribution | Population Stability Index (PSI) on mu distribution per playlist, daily vs 30-day baseline | PSI > 0.2 → alert | Batch job on warehouse data |
| Data drift — region pool density | Wasserstein distance on hourly pool-size time series vs prior week | > 25% shift sustained 3+ days | Streaming job on `queue.health` |
| Concept drift — match-quality model | Calibration error (predicted-close vs actual-close rate), rolling 7-day | Brier score degradation > 15% vs training-time baseline | Batch eval nightly |
| Fairness drift | Win-rate variance by region/party-size cohort (should be ~50% ± small band) | Any cohort win-rate outside 45-55% sustained over 10K+ matches | Batch fairness audit, weekly + on-demand |
| Rating inflation drift | Mean mu trending up/down globally without corresponding skill change (inflation from draw-rate assumptions) | Mean mu drift > 2 sigma-units/quarter unexplained | Quarterly offline review |

- All drift jobs write to a shared `drift_metrics` warehouse table consumed by the Monitoring dashboard (Section 22) and wired to Alerting (Section 23).

## 22. Monitoring

| Category | Metrics |
|---|---|
| Infra | Matchmaker worker CPU/mem per shard, Redis cluster hit ratio + latency, Kafka consumer lag per topic/partition, DGS allocation success rate |
| Model quality | Match-quality model calibration, skill-gap distribution per match, TrueSkill sigma convergence rate (are uncertainties shrinking as expected with games played) |
| Business/product | p50/p99 wait time per region/playlist, match-abandon rate (cancel-before-match), post-match churn rate, win-rate balance (near-50%), backfill success rate |
| Queue health | Live pool size per shard, tickets/sec in vs matches formed/sec (should track — growing gap = matcher falling behind) |
| Fairness | Skill-gap p95 per formed match, cross-region-match rate (how often relaxation kicked in), party-vs-solo win-rate delta |

- Dashboards segmented by region × playlist — a global aggregate hides regional pool-starvation problems (e.g., SEA off-peak).

## 23. Alerting

| Alert | Condition | Routing |
|---|---|---|
| Queue backlog growing | tickets-in-pool growth rate positive for 5 consecutive ticks with no matches formed | Page on-call matchmaking SRE (P1) |
| p99 wait time breach | p99 wait > 90s sustained 3 min in any region-playlist | Page on-call (P2) |
| Kafka consumer lag | Lag > 30s on `matchmaking.tickets` or `match_results` | Page on-call (P1) |
| DGS allocation failure spike | Allocation failure rate > 5% over 2 min | Page on-call + auto-page DGS fleet team (P1, cross-team) |
| Fairness cohort breach | Any cohort win-rate outside 45-55% over 10K matches | Ticket to fairness/data-science team (P3, non-paging) |
| Model calibration degradation | Brier score regression > 15% | Ticket to ML team, triggers ad-hoc retrain (P3) |
| Double-booking detected | Ticket state machine violation (matched twice) | Immediate page (P0) — correctness bug, player-facing |

- Paging routed via standard on-call rotation tool (PagerDuty-equivalent), with P0/P1 to primary matchmaking on-call, P2/P3 to team Slack channel + ticket queue.

## 24. Logging

- **Structured logging** (JSON) at every state transition: `ticket_submitted`, `ticket_matched`, `ticket_cancelled`, `ticket_expired`, `match_result_received`, `rating_updated` — each with `trace_id`, `player_id` (hashed for non-privileged logs), `region`, `playlist`, `timestamp`.
- **PII handling**: raw `player_id` and IP-derived region data classified as PII under EA privacy policy — logs destined for general observability tooling use a salted hash of player_id; only privileged fraud/support tooling can reverse via a separate lookup service with audit-logged access.
- **Retention**: operational logs (Ticket/Matchmaker services) — 30 days hot (searchable), 1 year cold archive for incident forensics. Match-result/rating-history data — retained per EA data-retention policy tied to account lifecycle (deleted/anonymized on account deletion request, GDPR/CCPA compliance).
- Sensitive fields (exact IP, device fingerprint) never logged at INFO level; only at DEBUG in non-prod with synthetic data.

## 25. Security

- **Threat model specific to this system**:
  - **Rating manipulation / boosting**: colluding accounts intentionally losing/winning to manipulate a teammate's rating — mitigated via anomaly detection on win-rate-vs-expected-outcome patterns (feeds into anti-cheat, not matchmaking itself, but matchmaking emits the signal).
  - **Queue manipulation**: client spoofing region_pref or ping to get easier matches — mitigated by server-side ping verification (server measures actual RTT to candidate DGS, doesn't trust client-reported ping) and server-side skill lookup (client cannot pass its own mu/sigma).
  - **Denial of queue**: bot/scripted mass ticket submission to degrade the pool for real players — mitigated via per-account rate limiting (Section 27) and device/account reputation scoring at the API gateway.
  - **Double-booking exploit**: race condition allowing a player into two matches simultaneously (resource theft, competitive integrity issue) — mitigated by atomic compare-and-swap on ticket state in Redis (Lua script), single-writer-per-ticket invariant.
- **Encryption**: TLS 1.3 for all client-service and service-service traffic; at-rest encryption (AES-256) for skill-profile and match-history stores; Kafka topics encrypted in transit (mTLS between brokers/clients).
- **Data classification**: skill ratings and match history treated as player-account-linked data under EA's data governance — access-controlled, not exposed in raw form to third parties/anti-cheat vendors without anonymization.

## 26. Authentication

- **End-user auth**: client authenticates via EA Account/Origin identity upstream; matchmaking API gateway validates short-lived JWT (issued by EA identity service) on every request, extracts `player_id` from claims (never trusts client-supplied player_id in the payload).
- **Service-to-service auth**: mTLS between internal services (Ticket Service ↔ Matchmaker ↔ Match Assembly ↔ DGS fleet), SPIFFE/SPIRE-style workload identity for cert issuance/rotation.
- **DGS callback auth**: DGS-to-platform result-posting endpoint uses a scoped service-account token (short TTL, rotated), plus session_id must match an outstanding allocation record — prevents forged match results.

## 27. Rate Limiting

- **Algorithm**: token bucket per player_id at the API gateway (e.g., 5 ticket-submits/minute burst, refill 1/12s) — matchmaking submit is inherently low-frequency per legitimate user (you don't resubmit every second).
- **Per-tenant (studio/title) limits**: separate token-bucket pools per title to prevent one game's traffic spike from starving shared gateway infra (if matchmaking platform is shared across EA titles).
- **Abuse-tier limiting**: accounts flagged by reputation scoring get a stricter bucket (e.g., 1/min) pending review.
- **WatchTicket streaming** rate-limited by max concurrent open streams per player (1 — a player has exactly one active ticket) rather than request-rate, since it's a long-lived stream not discrete requests.

## 28. Autoscaling

- **Matchmaker workers**: scale by shard count, driven by `queue.health` metric (pool_size and tick-processing-time) — KEDA scaler on Kafka consumer lag for `matchmaking.tickets` partition backlog; add shard-workers when lag > 5s sustained, scale down when pool_size per shard < 20% of capacity for 10 min (avoid thrashing).
- **Ticket Service / Match-Status API**: standard HPA on CPU (target 60%) and request-latency SLO burn rate; stateless, scales fast (<60s to add pods).
- **Skill Rating Service**: HPA on request QPS (correlates with match-completion rate, itself correlates with CCU) — scale target tied to `match_results` consumer lag.
- **VPA** applied to Matchmaker workers for right-sizing memory (pool size in-memory footprint varies significantly by region/time-of-day) — recommendation-only mode to avoid disruptive restarts during peak.
- Predictive pre-scaling: known daily peak windows (regional evening peak) trigger scheduled scale-up 15 min ahead via CronJob-driven HPA min-replica bump, avoiding reactive-scaling lag during the ramp.

## 29. Cost Optimization

- **Spot/preemptible instances** for Matchmaker workers where feasible — workers are stateless-recoverable (ticket state lives in durable Redis+Kafka, not worker memory alone), so spot interruption just means a brief shard rebalance, not data loss.
- **Batched tick processing** (1s ticks over full pool) instead of per-ticket real-time evaluation — amortizes compute; avoids O(n²) naive-pairwise-check costs that would explode with pool size.
- **Skill-profile cache** (Section 11) cuts ~90% of reads away from the KV store, reducing DynamoDB/Cassandra read-capacity-unit spend substantially at 33K tickets/sec ingestion.
- **Model distillation/simplicity**: match-quality model deliberately kept as a small tree ensemble (not a deep net) — avoids GPU spend entirely for a task where a simple model achieves near-equal lift (interviewer signal: right-sizing model complexity to the problem, not defaulting to deep learning).
- **Off-peak region consolidation**: low-population regions (e.g., SEA at 3am) can shed dedicated shard capacity and merge into a shared low-traffic worker pool rather than holding reserved capacity 24/7.
- **DGS pre-warming cost tradeoff**: balance warm-pool size (idle server cost) against match-assembly latency (cold-start cost) — tuned via observed CCU curve, not fixed.

## 30. Operational Concerns (Deployment, Reliability, Infra)

At SDE2 scope, treat this as a checklist rather than a design exercise: **backups** (automated snapshots of the model registry, feature store, and any stateful service, with a tested restore path), **rollback** (every deploy must be revertible to the last-known-good version — the model registry and CI/CD pipeline should make this a one-command operation), **canary/blue-green rollout** (shift a small percentage of traffic first, watch error rate and key business/model metrics, then ramp), and **basic observability** (dashboards + alerts on latency, error rate, and the top 2-3 model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level infra-architecture concerns — worth knowing they exist, not worth rehearsing the manifests.

## 38. Why This Architecture

- **Region-sharded, tick-based matchmaker** aligns compute with the actual constraint (matching is inherently local/regional) and avoids a single global bottleneck — matches EA's live-service reality of geographically distributed player bases with wildly different peak times.
- **Event-driven backbone (Kafka)** decouples ticket ingestion from matching computation and from rating updates — each can scale/fail independently, and durability comes free via the log (survives matchmaker crash without losing player position in queue).
- **In-memory ticket store (Redis) + durable WAL (Kafka)** gives the sub-100ms read/write latency the tick loop needs while still surviving process/pod failure — a pure-Kafka-only design would be too slow for tick-loop reads; a pure-in-memory-only design would lose tickets on crash.
- **No GPU / small-model philosophy**: the core problem (Bayesian rating update, constrained bin-packing) doesn't need deep learning; keeping the match-quality signal as a small tree model keeps latency and cost low while still capturing non-linear quality signal beyond a hand-tuned heuristic.
- **Active-active multi-region** matches the non-negotiable real-time-competitive latency requirement — no config where cross-ocean matching is acceptable for a shooter.

## 39. Alternative Architectures

| Alternative | Description | Why Rejected / When Preferred |
|---|---|---|
| Centralized global matchmaker (single logical queue, no regional sharding) | One global service evaluates all tickets | Rejected: cross-region ping makes globally "optimal" matches often unplayable; would still need regional constraint filtering, so sharding is strictly better. Preferred only for turn-based/async games with no real-time latency constraint (e.g., async PvP where ping is irrelevant). |
| Pure request-response (synchronous) matchmaking — client blocks on a single call until matched | No ticket/polling model; one long-held connection/request per search | Rejected: doesn't survive client network blips gracefully, harder to scale connection-held state at 500K concurrent, and can't cleanly support progressive constraint relaxation as a background process. Preferred for very small-scale/prototype systems where operational simplicity outweighs scale needs. |
| Deep learning-based match-quality model (e.g., neural net over rich player embeddings) | Learn match quality/churn-risk via embeddings of playstyle, history, social graph | Rejected for this scope: adds GPU serving cost and latency for marginal lift over a well-featured tree model, and hurts explainability (fairness audits, "why was I matched this way" support/regulatory questions). Preferred if/when social-matchmaking or toxicity-aware matching becomes a hard requirement, justifying the embedding investment (see Sections 16/17). |
| Fully decentralized/serverless (each match formed by peer client negotiation, e.g., lobby-browser style) | Clients see a list of open lobbies and self-select | Rejected: no fairness guarantee, poor UX at high CCU (choice paralysis), and no central point to enforce skill-balance or anti-cheat signal — historically used in older/smaller titles but doesn't fit EA-scale competitive matchmaking SLAs. |

## 40. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Progressive constraint relaxation (widen skill band / region over time) | Bounds worst-case wait time; keeps queue moving in low-pop windows | Later matches in the relax curve are less balanced/higher-ping — fairness degrades for the patient-but-unlucky player |
| Region-local ticket pools (no cross-region pool merge by default) | Best possible ping, avoids compute/coordination across regions | Low-population regions/off-hours suffer thin pools → longer waits or forced bot-fill |
| TrueSkill (Bayesian, tracks uncertainty) vs simple Elo | Faster convergence for new players, natively supports team/multi-player matches | More complex to explain to players/support ("why is my rating X"), costlier to reason about at audit time |
| Party (premade group) support | Better social UX, retention | Fundamentally reduces matching fairness (party synergy/comms advantage not captured in skill rating) — a known, accepted tradeoff in the industry |
| Small tree-based match-quality model vs heuristic-only scoring | Captures non-linear signal, incremental win-rate/retention lift | Added deploy/retrain/drift-monitoring surface area vs a simpler, fully-deterministic heuristic |
| At-least-once event delivery + idempotency vs exactly-once broker guarantee | Simpler infra, broker-agnostic, well-understood | Requires careful idempotency-key discipline in every consumer; a missed edge case risks duplicate processing |

## 41. Failure Modes

| Scenario | Impact | Mitigation |
|---|---|---|
| Redis ticket-store shard outage (region) | All in-flight tickets in that shard's pool lose live state | Kafka WAL replay rebuilds pool on failover replica within seconds; clients see a brief wait-time blip, not a hard error |
| Kafka broker/partition unavailable | Ticket ingestion or match-result processing stalls | Multi-AZ MSK cluster with replication factor 3; producer retries with backoff; consumer lag alert fires if stall > 30s |
| Matchmaker shard stuck in tick-loop deadlock/bug | Pool for that shard stops forming matches, tickets silently accumulate | `queue.health` growth-with-no-matches alert (Section 22/23) fires within 1-2 ticks; k8s liveness probe restarts pod; sibling shard can absorb overflow if shard-key rebalancing configured |
| DGS fleet exhausted (no servers to allocate) | Matches form logically but can't get a server — players stuck at "match found, connecting" | Circuit-breaker: Match Assembly Svc fails fast, re-queues affected tickets with priority boost rather than leaving them in limbo; alert to DGS fleet team |
| Skill Rating store returns stale/unavailable data | Matchmaker falls back to cached/default rating (new-player prior) — degrades fairness temporarily, doesn't block matching | Fail-open (assume average skill) rather than fail-closed (block queue) — availability prioritized over perfect fairness for this specific failure |
| Double-booking race (two shards match same ticket) | Player pulled into two sessions — severe correctness/trust issue | Atomic CAS on ticket state in Redis via Lua script eliminates the race at the data layer; P0 alert + kill-switch if detected anyway |
| Cascading region failover (one region's traffic floods neighbor after outage) | Neighbor region's pool overwhelmed, its own wait times spike | Failover routing caps the fraction of foreign-region traffic accepted (graceful shed) rather than unconditionally absorbing all of it |

## 42. Scaling Bottlenecks

- **At 10x CCU (5M concurrent, 330K tickets/sec)**:
  - Redis ticket-store shard count must grow proportionally — fine, horizontally shardable, but shard-rebalancing operational overhead grows.
  - Kafka partition count for `matchmaking.tickets` needs proportional increase; consumer group rebalancing during scale-events becomes a bigger operational risk (rebalance storms briefly pausing consumption).
  - Skill Rating Service read QPS scales linearly — cache hit-ratio becomes even more critical; a cache-layer regression at this scale directly threatens the KV store's provisioned capacity.
- **At 100x CCU (50M concurrent — likely unrealistic for one title but relevant if platform is shared across many EA titles)**:
  - Single biggest break point: **tick-loop pool size per shard** — even with skill-decile bucketing, a 100x pool means either far more shards (coordination overhead, more cross-shard edge cases at bucket boundaries) or algorithmic change needed (e.g., move from greedy-tick to a streaming/incremental matching algorithm that doesn't re-scan the full pool each tick).
  - Match Assembly / DGS allocation becomes bottlenecked by DGS fleet provisioning speed, not matchmaking logic — matchmaking software scales further than the physical server fleet does.
  - Cross-team/shared-platform multi-tenancy (many titles) would require moving from per-title dedicated infra to genuine multi-tenant isolation (noisy-neighbor risk on shared Kafka/Redis clusters) — likely necessitating per-title namespace/cluster isolation rather than one shared mega-cluster.

## 43. Latency Bottlenecks

**p50 = ~15s breakdown** (dominated by pool-wait, not compute):
| Stage | Time |
|---|---|
| Ticket submit → accepted (API, cache read, Redis write, Kafka emit) | ~10-15ms |
| Waiting for tick cycle to pick up ticket | 0-1000ms (tick interval) |
| Pool-fill wait (waiting for enough compatible players) | **~13-14s** (dominant term) |
| Candidate scoring (match-quality model, <1ms × K candidates) | <5ms |
| Match Assembly + DGS reservation | ~60-100ms |
| Client receives assignment, connects to DGS | ~100-300ms (network handshake) |

**p99 = ~60s breakdown**: same fixed-cost stages, but pool-fill wait dominates further — p99 cases are low-population moments (off-peak region, or narrow-skill-band outlier player) where progressive relaxation is actively widening constraints across several relaxation steps (each step has its own multi-second wait before re-evaluation).

- **Where time is actually spent**: >95% of both p50 and p99 latency is *inherent to supply* (are there enough compatible players right now), not system compute — the architecture's job is to minimize the *system-imposed* latency (tick interval, allocation overhead) so it doesn't add meaningfully on top of the unavoidable pool-wait, and to bound the pool-wait via relaxation rather than let it grow unbounded.

## 44. Cost Bottlenecks

- **Compute for matchmaker workers held resident 24/7 across low-traffic hours** — even with autoscaling, a floor of minReplicas per region-playlist shard for availability means some idle-cost floor in low-population regions/hours; biggest lever is aggressive off-peak shard consolidation (Section 29).
- **Skill Rating Service KV store read/write capacity** at 33K/sec ticket-submit-driven read rate — without the cache layer, this is the single largest recurring line item (provisioned read capacity units); cache hit-ratio directly is the dominant cost lever here.
- **Kafka cluster sizing** (broker count, storage retention for replay) driven by peak ingestion rate × retention window — a longer replay/retention window (for DR/audit) directly trades off against broker storage cost.
- **Cross-region data replication** (skill profile global tables) — replication data-transfer cost scales with write rate (post-match updates) × number of replica regions; not matching-volume-dominant but non-trivial at 4.4M matches/day × multi-region fanout.
- **DGS fleet warm-pool** is technically a separate system's cost, but matchmaking's allocation-latency requirements directly dictate how large a warm pool that system must maintain — a tight coupling worth calling out even though DGS fleet sizing itself is out of this chapter's scope.

## 45. Interview Follow-Up Questions

1. How would you handle a party of 5 queuing against solo players in a 5v5 mode — what's the fairness implication and how do you mitigate it?
2. Walk me through what happens if the match-quality model starts systematically favoring one region's playstyle after a retrain — how would you catch it?
3. Why TrueSkill over Elo for this system, and what does TrueSkill give you that Elo fundamentally can't?
4. How do you prevent a player from being matched into two sessions simultaneously in a distributed, sharded system?
5. Your p99 wait time SLA is breached only in one low-population region during its 3am local window — what do you actually do about it, concretely?
6. How would this design change for a battle-royale mode with 100 players per match instead of 5v5?
7. What's your approach to detecting and handling skill-rating manipulation (queue-dodging, intentional loss/win boosting)?
8. If you had to support cross-play matchmaking (PC + console) with different input methods, how does that change the matching constraints?
9. How do you validate that your progressive constraint-relaxation curve is actually well-tuned, rather than guessed?
10. What would you change about this architecture if the game were turn-based (async) instead of real-time?

## 46. Ideal Answers

1. **Party vs solo fairness**: Apply a party-strength adjustment — inflate the effective team rating estimate to account for coordination advantage, using an empirically calibrated multiplier from historical party-vs-solo win-rate deltas, and prefer matching parties against similarly-sized parties first. Track party-vs-solo win-rate as an explicit fairness metric (Section 22) and recalibrate periodically rather than guessing once.

2. **Regional playstyle bias post-retrain**: Segment the calibration/Brier-score eval by region/cohort before rollout, and gate canary promotion (Section 33) on per-cohort parity, not just the global average. Make "per-segment fairness regression" a first-class canary gate rather than an afterthought.

3. **TrueSkill vs Elo**: TrueSkill models rating as a distribution (mu, sigma) rather than a point estimate, so it natively expresses confidence — letting new players (high sigma) converge faster — and its factor-graph decomposes team outcomes into per-player contributions. Vanilla Elo is pairwise/1v1-native and can't handle team games without ad hoc hacks like team-average Elo.

4. **Preventing double-booking**: Ticket state transitions must be atomic and single-writer, implemented via a Redis Lua script that does check-and-set (`SEARCHING → MATCHED`) as one atomic operation, so racing shards can't both claim the same ticket. This is a correctness invariant enforced at the data layer, not via slower, race-prone application-level coordination.

5. **Off-peak regional SLA breach**: First confirm it's a genuine supply problem via `queue.health`, not a system bug. Then widen the skill-band relaxation curve for that region-hour, allow cross-region matching to the next-nearest region, or bot-fill as a last resort.

6. **100-player battle royale**: Matching shifts from balanced-team pairing to lobby-filling/bin-packing — the goal is a lobby of 100 with reasonable skill spread, not tight pairwise balance, so the algorithm changes from greedy-pair-and-balance to fill-to-capacity with skill-tier stratification. Pool-fill wait dominates even more at this scale, making partial-lobby bot-fill timeouts a standard mechanic.

7. **Rating manipulation detection**: This is primarily an anti-cheat/trust-and-safety problem that matchmaking feeds signal into rather than solves alone — look for statistical anomalies like repeated same-party win/loss patterns inconsistent with skill deltas or rating trajectories inconsistent with performance telemetry. Matchmaking's role is to surface these signals to a dedicated integrity pipeline and support soft mitigations (rating-change dampening) rather than unilaterally banning.

8. **Cross-play PC + console**: Add input-method as an explicit matching dimension, either via separate mu/sigma sub-pools per input method or per-lobby input-composition constraints, since aim-assist and mouse-precision produce different effective skill curves. This is a fairness-vs-pool-size tradeoff like region relaxation, so it fits as a configurable dimension in the same progressive-constraint framework.

9. **Validating the relaxation curve**: Run offline replay simulation against historical ticket-arrival logs with candidate relaxation curves, measuring the resulting wait-time and fairness distributions each would produce. Then A/B test candidates live via canary (Section 33), gated on both metrics simultaneously, since optimizing one alone trivially degrades the other.

10. **Turn-based/async redesign**: Real-time constraints (sub-second tick loops, ping-aware routing, live DGS allocation) largely disappear — matching can run as a slower batch process with much larger pools, and region/ping stops being a matching constraint. This would collapse the real-time streaming backbone into a simpler scheduled-batch-job design, showing this architecture's complexity is a direct consequence of the real-time requirement.
