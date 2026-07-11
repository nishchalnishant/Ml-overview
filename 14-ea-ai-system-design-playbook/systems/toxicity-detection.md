# Toxicity Detection

## 1. Problem Framing & Requirement Gathering

Design a real-time toxicity detection system for EA live-service games (FIFA Ultimate Team chat, Apex-style squad voice, in-game text chat, clan/guild forums) that:

- Scores text and voice utterances for toxicity (harassment, hate speech, threats, sexual content, grooming signals) in-line with gameplay.
- Supports 25+ languages/locales across EA's global player base.
- Feeds a human-in-the-loop (HITL) review queue for borderline/appealed cases.
- Minimizes false positives (legitimate banter, competitive trash-talk is core to game culture — over-moderation drives churn) while catching true harms (minors present, brand/legal risk, regulatory exposure under DSA/OSA).

This is a **trust & safety** system, not a generic content classifier — the cost function is asymmetric and politically sensitive: false negatives → PR/legal/child-safety risk; false positives → player churn, community backlash ("EA silenced me for banter").

## 2. Functional Requirements

- FR1: Classify inbound text chat messages into toxicity categories (harassment, hate speech, sexual content, threats/violence, self-harm, spam) with per-category scores.
- FR2: Classify inbound voice chat (streamed audio) via ASR + text classification, near-real-time.
- FR3: Support configurable per-title, per-region policy thresholds (e.g., FIFA EU vs. Apex NA may have different tolerance for profanity).
- FR4: Take enforcement action inline: allow / mute / mask (redact) / block-send / flag-for-review, within the chat pipeline before message delivery for high-confidence cases, or post-hoc for async cases.
- FR5: Route ambiguous/appealed/high-severity cases to a human review queue with context (conversation window, player history, prior reports).
- FR6: Support player reporting (explicit report → priority queue) and proactive detection (no report needed).
- FR7: Multilingual detection including code-mixed text (e.g., Spanglish, romanized Hindi), leetspeak, emoji-obfuscation, evolving slang.
- FR8: Feed enforcement outcomes (mutes, bans, appeals overturned) back into training data.
- FR9: Provide auditability — every enforcement decision must be explainable/traceable for appeals and legal requests.
- FR10: Support shadow-mode / dry-run deployment for new models before enforcement.

## 3. Non-Functional Requirements (latency, availability, throughput, consistency, cost)

| Dimension | Target |
|---|---|
| Inline text moderation latency | p50 < 50ms, p99 < 150ms (must not add perceptible chat lag) |
| Voice moderation latency | p50 < 800ms end-to-end (ASR + classify), acceptable since voice has natural buffering |
| Availability | 99.95% for the inline scoring path (chat must degrade gracefully, never block on ToxDetect being down) |
| Throughput | Sustain 400K messages/sec peak (World Cup / FUT promo events), voice: 150K concurrent streams |
| Consistency | Eventual consistency acceptable for enforcement history; strong consistency required for ban/mute state per player (avoid double-punish or missed-punish races) |
| Cost | Inference cost must stay < $0.000015/message amortized (at 10B msgs/month this is ~$150K/mo ceiling) |
| False-positive rate | < 0.5% of allowed messages should be reviewer-overturned blocks |
| False-negative rate (severe categories) | < 2% miss rate on threats/grooming, audited weekly by human sample review |
| Data freshness for new slang/evasion patterns | Retrain/patch cycle < 7 days |

## 4. Clarifying Questions an interviewer would expect you to ask

1. Is enforcement inline (blocking) or post-hoc (async, message already delivered)? Changes latency budget entirely.
2. Which titles/surfaces are in scope — text chat, voice, usernames, clan tags, forum posts? Each has different SLAs.
3. Is voice moderation required at launch, or text-first with voice as phase 2?
4. What languages/regions are P0 vs. best-effort?
5. Do we need on-device pre-filtering (console/PC client) vs. all server-side?
6. What's the appeals SLA — do players expect resolution in hours or days?
7. Are there legal/regulatory constraints (COPPA for under-13 titles, EU DSA transparency reporting, UK OSA)?
8. Does enforcement differ by player tier (paying customers, pro players, minors)?
9. Is there an existing profanity-filter/word-list system to keep as a fast-path, or fully ML-driven?
10. What human reviewer capacity exists today (headcount, cost per review) — bounds how much can be routed to HITL.
11. Do we need cross-title shared trust signals (a player banned in Apex — flag in FIFA)?

## 5. Assumptions

1. EA has ~120M monthly active accounts across titles; chat-enabled titles cover ~60M MAU.
2. Average chat-active player sends ~40 messages/session, ~2 sessions/day → ~2.4B text messages/day platform-wide, peaking 4x during live events → design for 10B msgs/day capacity headroom.
3. Voice chat adoption is ~15% of chat-active MAU concurrently during peak (~9M concurrent-eligible, ~2M actually transmitting at any peak instant).
4. Inline enforcement is required for text (block-before-send for high-confidence severe categories); voice is near-real-time with 1-2s tolerable buffer.
5. 30 languages cover 95% of volume; long-tail languages route to a smaller multilingual fallback model + human review.
6. Existing profanity word-lists exist and are kept as a cheap first-pass filter, not replaced.
7. Enforcement decisions must be retained 180 days minimum for appeals/legal; anonymized aggregates retained indefinitely for model training.
8. GPU fleet is shared cross-title infra (not per-game silo) to amortize cost.
9. Human review team: ~500 reviewers globally, 24/7 follow-the-sun, average 90 sec/review.
10. False-positive cost is weighted ~3x false-negative cost in the loss function for borderline (non-severe) categories, but inverted (10x) for severe categories (CSAM/grooming/threats).

## 6. Capacity Estimation

**Text volume**
- Baseline: 2.4B msgs/day ≈ 28K msgs/sec average.
- Peak (live event, e.g., World Cup mode drop): 4x → 112K msgs/sec; provision to 400K msgs/sec for burst/safety margin (co-located esports + multiple simultaneous title promos).

**Voice volume**
- 2M concurrent streams at peak, each ~1 utterance/3 sec of active speech → ~660K utterances/sec at the ASR boundary in worst case; realistically voice-activity-detection (VAD) cuts this by ~70% (only speech segments forwarded) → ~200K utterance-segments/sec to classify.

**Model size & compute**
- Text classifier: distilled transformer, ~66M params (DistilBERT-class), multilingual, INT8 quantized → ~70MB, fits comfortably on CPU (ONNX Runtime) or small GPU slice.
- Per-inference latency budget: ~8ms CPU (batched) or ~2ms on T4-class GPU with batching of 32.
- Throughput per GPU (T4, batch=32, seq_len=64): ~6,000 inferences/sec.
- GPUs needed for text at peak (400K msgs/sec, allow 3x margin over avg 112K): 400,000 / 6,000 ≈ **67 GPUs** (T4-equivalent) for the primary scorer; provision 90 with headroom + multi-AZ.
- ASR (voice→text): Whisper-small-class model, ~240M params, ~30ms/sec-of-audio on A10G. At 200K utterance-segments/sec × ~1.5 sec avg segment = 300K sec-of-audio/sec to transcribe. Real-time factor ~0.05 (20x faster than real-time) on A10G batched → throughput ~20 sec-audio/sec/GPU-equivalent effective, so need 300,000/20 ≈ **15,000 GPU-seconds/sec**... reframe: RTF 0.05 means 1 GPU handles 1/0.05 = 20x real-time → processes 20 sec of audio per 1 sec wall time. Need 300,000 sec-audio/sec → 300,000/20 = **15,000 A10G-equivalent GPUs**. This is why voice ASR is the dominant cost driver — mitigate via: (a) only run ASR on VAD-flagged segments already applied above, (b) further sample/triage (only run full ASR on reported players + random audit sample + escalating trust score), reducing real load to ~10% of raw → **~1,500 GPUs** for ASR, still the largest fleet.
- Toxicity-on-transcript reuses the same 66M param text model — negligible incremental compute vs. ASR.

**Storage**
- Message metadata (not full content by default, only flagged content retained): flagged/reviewed messages ~2% of volume → 2.4B × 2% = 48M records/day, ~1KB each (text + scores + context refs) → 48GB/day → ~8.6TB over 180-day retention.
- Voice: only flagged utterance audio snippets retained (10-15 sec clips), ~2% flag rate of the triaged 10% sample → manageable: ~2M clips/day × 200KB avg (compressed opus) → 400GB/day → ~72TB over 180 days (push to cold/object storage after 30 days).
- Feature/embedding store: player trust-score & history features, ~60M active players × ~2KB profile → 120GB, fits in a fast KV store.

**Human review capacity**
- 500 reviewers × 24h/day / 90 sec per review × 3600 ≈ 500 × 960/hr = 480K reviews/hour capacity ≈ 11.5M/day.
- Flag rate must be tuned so escalations to HITL stay ≤ 11M/day (routing logic in section 12).

## 7. High-Level Architecture

```
                         ┌─────────────────────────────────────────────────────────────────┐
                         │                        Game Clients (Console/PC/Mobile)          │
                         │   Text Chat SDK        Voice Chat SDK        Report Button       │
                         └───────┬───────────────────────┬─────────────────────┬────────────┘
                                 │ gRPC/WebSocket         │ RTP/WebRTC audio    │ REST report
                                 ▼                        ▼                     ▼
                    ┌────────────────────┐   ┌──────────────────────┐   ┌───────────────────┐
                    │  Chat Gateway Svc   │   │  Voice Ingest Gateway │   │  Report API Svc   │
                    │ (rate-limit, auth)  │   │ (jitter buffer, VAD)  │   │                    │
                    └─────────┬───────────┘   └───────────┬───────────┘   └─────────┬─────────┘
                              │ publish                    │ publish (VAD segments)  │
                              ▼                            ▼                         │
                    ┌────────────────────────────────────────────────────┐           │
                    │        Kafka: raw-chat-events / raw-voice-events    │◄──────────┘
                    └───────────────────┬──────────────────────────────--┘
                                        │ consume
                                        ▼
                    ┌───────────────────────────────────────────┐
                    │      Fast-Path Filter Service (CPU)        │
                    │  word-list / regex / known-bad hash match  │
                    └───────────────┬─────────────────────────---┘
                        pass-through │  hit → immediate block
                                     ▼
                    ┌────────────────────────────────────────────┐        ┌──────────────────────┐
                    │   ASR Service (voice only, GPU fleet)       │──────► │  Text Toxicity Model  │
                    │   Whisper-small distilled, batched           │        │  (multilingual, GPU/  │
                    └───────────────────────────────────────────┘        │  CPU, INT8, batched)   │
                                                                          └──────────┬────────────┘
                                                                                     │ scores
                                                                                     ▼
                                                                   ┌─────────────────────────────────┐
                                                                   │   Decision Engine / Policy Svc    │
                                                                   │  (thresholds per title/locale,    │
                                                                   │   player trust score lookup)      │
                                                                   └───────┬──────────────┬───────────┘
                                                     allow/mask/block      │              │ ambiguous/severe
                                                     ◄────────────────────┘              ▼
                                                     back to Chat/Voice Gateway   ┌──────────────────────┐
                                                                                  │  Review Queue Kafka    │
                                                                                  │  + Priority Scheduler  │
                                                                                  └──────────┬────────────┘
                                                                                             ▼
                                                                                  ┌──────────────────────┐
                                                                                  │  HITL Review Console   │
                                                                                  │  (reviewer UI, context) │
                                                                                  └──────────┬────────────┘
                                                                                             ▼
                                                                                  ┌──────────────────────┐
                                                                                  │ Enforcement Service    │
                                                                                  │ (mute/ban/appeal state)│
                                                                                  └──────────┬────────────┘
                                                                                             ▼
                                                                    ┌──────────────────────────────────────┐
                                                                    │ Player Trust/Enforcement Store (KV/SQL)│
                                                                    └──────────────┬───────────────────────┘
                                                                                   ▼
                                                                    ┌──────────────────────────────────────┐
                                                                    │ Feature Store + Training Data Lake    │
                                                                    │ (offline, feeds retraining pipeline)  │
                                                                    └──────────────────────────────────────┘
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| Chat Gateway Svc | Auth, per-connection rate limiting, protocol termination (WS/gRPC), publish to Kafka | gRPC bidi stream from client; Kafka producer | Stateless pod, scale on connection count |
| Voice Ingest Gateway | WebRTC/RTP termination, jitter buffering, VAD segmentation | RTP in, VAD-segmented PCM out to Kafka | Scale on concurrent stream count, sticky session per call |
| Fast-Path Filter | Hash/regex match against known-bad terms & hash-list (evasion variants) | Kafka consumer → Kafka producer (annotated) | CPU pods, scale on msg/sec |
| ASR Service | Speech-to-text on VAD segments | gRPC, batched inference server | GPU pods, scale on queue depth |
| Text Toxicity Model Service | Multi-label toxicity scoring | gRPC/REST, batched | GPU/CPU pods, scale on queue depth + latency SLA |
| Decision Engine / Policy Svc | Apply per-title/locale thresholds, blend model score + player trust score, decide action | Internal RPC, low-latency KV lookups | Stateless, scale on QPS, must be <5ms p99 |
| Review Queue Scheduler | Priority ranking (severity, report count, player tier) of items into human queue | Kafka consumer, priority queue impl (e.g. Redis sorted set) | Partition by severity tier |
| HITL Review Console | UI + APIs for reviewers to see context, decide, annotate | REST/GraphQL | Stateless web tier |
| Enforcement Service | Apply mute/ban, track appeal state, ensure idempotent enforcement | REST + event stream | Strongly consistent store required |
| Player Trust/Enforcement Store | System of record for player standing, history | SQL (transactional) | Sharded by player_id |
| Feature Store | Serve player trust/behavioral features online + offline for training | Online: gRPC KV; Offline: Parquet/lake | Partition by player_id |
| Training Pipeline | Periodic/triggered retraining | Batch orchestration (Airflow/Kubeflow) | Job-based, GPU cluster |
| Drift Monitor | Track input/output distribution shift | Batch job + streaming metrics | Scheduled |

## 9. API Design

**Inline text scoring (internal, gateway → decision engine, but shown as service API)**

```
POST /v1/moderate/text
Request:
{
  "message_id": "uuid",
  "player_id": "string",
  "title_id": "fifa25",
  "locale": "en-US",
  "channel": "match_chat",
  "text": "string",
  "context": { "recent_messages": ["..."], "reported_before": false }
}
Response:
{
  "message_id": "uuid",
  "action": "allow | mask | block | review",
  "scores": { "harassment": 0.02, "hate": 0.01, "sexual": 0.0, "threat": 0.85 },
  "model_version": "tox-text-v14.2",
  "decision_latency_ms": 22
}
```

**Voice scoring**
```
POST /v1/moderate/voice-segment
Request: { "call_id", "player_id", "title_id", "audio_ref" (object store pointer), "duration_ms" }
Response: { "transcript": "string", "scores": {...}, "action": "...", "asr_confidence": 0.91 }
```

**Player report**
```
POST /v1/report
Request: { "reporter_id", "reported_player_id", "title_id", "evidence_ref", "category": "harassment" }
Response: { "report_id", "status": "queued", "priority": "high" }
```

**Review queue (reviewer console)**
```
GET  /v1/review/queue?reviewer_id=&locale=&limit=20
POST /v1/review/{item_id}/decision   { "decision": "uphold|overturn", "notes": "..." }
```

**Enforcement / appeals**
```
GET  /v1/players/{player_id}/enforcement-status
POST /v1/appeals   { "player_id", "enforcement_id", "reason": "..." }
```

**Versioning**: URI-based (`/v1/`, `/v2/`) for breaking schema changes; model_version returned in body for shadow/canary tracking independent of API version. Deprecation via `Sunset` header, 90-day notice.

## 10. Database Design

| Store | Data | Type | Why | Partition/Shard Key |
|---|---|---|---|---|
| Player Trust/Enforcement Store | Ban/mute state, strike history | PostgreSQL (transactional, strongly consistent) | Needs ACID for enforcement correctness (no double-ban races, appeal state transitions) | `player_id` (hash-sharded) |
| Flagged Message Store | Retained flagged text + scores + reviewer decision | Columnar (e.g. ClickHouse) | Analytical queries (trend by locale/category), high write volume, append-only | `title_id` + date, secondary index on `player_id` |
| Voice Clip Store | Flagged audio snippets | Object storage (S3-class) + metadata in columnar store | Large binary blobs, lifecycle policies (cold after 30d) | Key = `call_id/segment_id` |
| Review Queue | Pending review items | Redis (sorted sets by priority score) | Needs low-latency pop/peek, ephemeral | Partition by severity tier |
| Feature Store (online) | Player behavioral/trust features | Key-value (DynamoDB/Redis) | Sub-5ms reads at inference time | `player_id` |
| Feature Store (offline) | Historical features for training | Parquet on data lake (S3 + Iceberg/Delta) | Batch scale, point-in-time joins | Partition by `event_date` |
| Word-list / hash-list | Known bad terms, evasion variants | Small KV, replicated to edge/CPU filter nodes | Needs to be in-memory for fast-path filter | Full replica per node (small dataset) |
| Model Registry | Model versions, eval metrics, rollout status | Relational (Postgres) | Small, transactional, needs joins with deployment state | N/A |

## 11. Caching

| Cache | What | Strategy | Invalidation |
|---|---|---|---|
| Player trust score cache | Trust/reputation score used in decision blending | Cache-aside, Redis, TTL 5 min | Event-driven invalidation on new enforcement action (publish to invalidation topic) + TTL backstop |
| Word-list cache | Fast-path bad-term list | Write-through on update, held fully in-memory per filter pod | Push-based refresh via pub/sub on list update (rare, versioned) |
| Model artifact cache | Loaded model weights on inference pods | Loaded at pod startup from model registry/artifact store | New version → new pod rollout (blue/green), no runtime invalidation needed |
| Decision Engine locale/policy config | Per-title/locale thresholds | Cache-aside, local in-memory + Redis backing, TTL 60s | Config service publishes change event; pods poll or subscribe |
| Recent-conversation-context cache | Last N messages per channel for context-aware scoring | Write-through ring buffer in Redis, capped size | Natural eviction (ring buffer overwrite), TTL on channel close |

## 12. Queues & Async Processing

| Queue | Purpose | Delivery Semantics | Dead-letter Handling |
|---|---|---|---|
| `raw-chat-events` (Kafka) | Ingested text messages | At-least-once | Consumer idempotency via `message_id`; poison messages after 3 retries → DLQ topic, alert |
| `raw-voice-segments` (Kafka) | VAD-segmented audio refs | At-least-once | Same pattern; DLQ reviewed by on-call for ASR failures |
| `review-queue` (Kafka → Redis sorted set) | Items needing human review | At-least-once (dedup by item_id on consume) | Items failing enrichment (context fetch fails) → retry topic with backoff, then DLQ + default-to-conservative-action |
| `enforcement-actions` (Kafka) | Mute/ban/appeal state changes | Exactly-once semantics required (idempotent producer + transactional writes to Postgres via outbox pattern) | Failed writes retried via outbox relay; DLQ triggers page (enforcement correctness is high-stakes) |
| `training-data-export` (batch/Kafka Connect → lake) | Async export of labeled data for retraining | At-least-once, dedup at lake via merge-on-write | Failures logged, backfill job re-runs partition |

- Why not exactly-once everywhere: cost/complexity not justified for scoring path (idempotent-by-design via `message_id` dedup is cheaper and sufficient); reserved exactly-once guarantee (via outbox + idempotent consumer) specifically for enforcement state because double-application (double ban days, missed unban) is a correctness/legal issue.

## 13. Streaming & Event-Driven Architecture

| Topic | Producer | Consumer Groups | Schema (key fields) |
|---|---|---|---|
| `raw-chat-events` | Chat Gateway | fast-path-filter-cg, text-model-cg (fan-out), analytics-cg | `{message_id, player_id, title_id, locale, text, ts, channel}` |
| `raw-voice-segments` | Voice Ingest Gateway | asr-cg, analytics-cg | `{segment_id, call_id, player_id, audio_ref, duration_ms, ts}` |
| `moderation-decisions` | Decision Engine | enforcement-cg, review-queue-cg, analytics-cg, training-export-cg | `{message_id, action, scores, model_version, decided_ts}` |
| `player-reports` | Report API | review-queue-cg, trust-score-cg | `{report_id, reporter_id, reported_player_id, category, ts}` |
| `enforcement-events` | Enforcement Svc | trust-score-cg, notification-cg, analytics-cg | `{enforcement_id, player_id, type, duration, reason, ts}` |
| `model-eval-events` | Retraining pipeline | model-registry-cg, alerting-cg | `{model_version, metric, value, eval_set, ts}` |

- Partitioning: `raw-chat-events` partitioned by `player_id` hash to preserve per-player ordering (context window correctness); `raw-voice-segments` partitioned by `call_id`.
- Consumer group scaling: text-model-cg scales with partition count (target 90 GPU-backed consumers matching partition count from capacity estimate).
- Schema registry (Avro/Protobuf) enforced — breaking schema changes require new topic version (`raw-chat-events-v2`), old topic drained before deprecation.

## 14. Model Serving

- **Framework**: Triton Inference Server for GPU-hosted text/ASR models (supports dynamic batching, multi-model, ONNX/TensorRT backends). CPU fast-path filter runs as a lightweight custom service (no ML framework needed, just hashing/regex).
- **Batching**: Dynamic batching window of 5-10ms, max batch 32 for text model; ASR uses streaming batching keyed on VAD segment arrival, max batch 16 (audio is heavier per-unit).
- **Multi-model**: Single Triton fleet hosts (a) primary multilingual text model, (b) 3-4 specialized severe-category models (grooming/CSAM-adjacent signals, self-harm) run in parallel ensemble for high-recall on severe categories, (c) ASR model. Model routing by request type.
- **Hardware**: T4/A10G GPUs for text (cost-efficient, INT8 quantized fits easily); A10G/L4 for ASR (needs more compute per unit). Spot/preemptible instances for offline batch scoring (audit sampling, backfill) — not for inline path.
- **Precision**: INT8 quantization for text model (validated <0.3% accuracy drop, 3x latency win); FP16 for ASR (quantizing ASR degrades WER too much for slang/accented speech).
- **Canary model slots**: Triton supports multiple versions loaded simultaneously — shadow traffic mirrored to candidate version for comparison without affecting production decisions.

## 15. Feature Store

- **Online store**: Redis/DynamoDB-backed, serves player trust score, rolling report-count-last-30-days, prior-strikes-count, account-age, playtime-hours — all needed at <5ms for decision blending.
- **Offline store**: Parquet/Iceberg on data lake, full historical feature snapshots for training joins.
- **Point-in-time correctness**: Every feature write is timestamped; training pipeline joins labels (enforcement outcome) to features **as they existed at message time**, not current state — prevents leakage (e.g., must not use "was later banned" as a feature to predict toxicity of an earlier message; use only pre-message trust score).
- Feature pipeline computes rolling aggregates (e.g., report-count-7d) via streaming (Flink/Kafka Streams) writing to both online store (low latency) and offline store (training) from the same computation to avoid train/serve skew.

## 16. Vector Database

**Applicable — used for near-duplicate/evasion detection, not primary classification.**

- Use case: detect obfuscated toxic phrases (leetspeak, homoglyphs, spacing tricks: "k1ll y0urself", "k i l l") that evade exact-match word-lists, and detect coordinated raids (many players sending near-identical harassment).
- Approach: embed normalized text (character n-gram or small sentence-embedding model) into vectors, ANN search against a store of known-bad embeddings.
- **ANN algorithm**: HNSW (via a vector DB like Milvus/pgvector) — chosen over IVF-PQ because recall matters more than memory footprint here (bad-phrase corpus is modest size, low tens of millions of vectors, fits HNSW memory profile) and query latency must stay in single-digit ms to fit the inline budget.
- Index refresh: near-bad-phrase corpus updated daily from newly confirmed toxic messages (human-reviewed uphold decisions) re-embedded and inserted.
- Not used for: primary toxicity classification (that's the transformer classifier) — vector similarity is a supplementary signal fed into the Decision Engine, not a replacement.

## 17. Embedding Pipelines

**Applicable, narrow scope** (supporting the vector-DB evasion detection above, and cross-lingual generalization).

- Text normalization: lowercase, homoglyph-folding, leetspeak-normalization, emoji-to-text mapping before embedding.
- Embedding model: small multilingual sentence-embedding model (distilled, ~30M params) shared across the fast-path evasion detector and as an auxiliary input feature to the main classifier (concatenated with token features) for better cross-lingual slang generalization.
- Batch embedding pipeline: nightly job re-embeds the confirmed-toxic corpus (from HITL uphold decisions) and refreshes the ANN index.
- Real-time embedding: computed inline as part of the fast-path/decision request (same INT8-quantized model, <3ms), not a separate round-trip.

## 18. Inference Pipelines

**Request lifecycle, text message (inline path):**

```
Client sends message
   │
   ▼
Chat Gateway: authn, rate-limit check, publish to raw-chat-events (Kafka) [~2ms]
   │
   ▼
Fast-Path Filter consumes: exact/hash match check [~1ms]
   │  no hit                              hit → immediate BLOCK, skip ML, notify gateway
   ▼
Text Toxicity Model Service: tokenize, embed, batch-infer [~8-15ms incl. batching wait]
   │
   ▼
Evasion check: normalize + embed + ANN lookup against bad-phrase index [~3ms, parallel with above]
   │
   ▼
Decision Engine: fetch player trust score (cache, ~1ms), blend model score + evasion signal + trust
                 apply per-title/locale threshold policy [~3ms]
   │
   ├─ allow ──────────────────► Chat Gateway delivers message to recipients
   ├─ mask ───────────────────► Gateway delivers redacted text
   ├─ block ──────────────────► Gateway drops, optionally notifies sender
   └─ review (ambiguous/severe)► publish to review-queue; provisional action applied (default: mask+hold)
                                 while awaiting human decision; reviewer decision retroactively
                                 confirms/overturns and updates enforcement store
   │
   ▼
Async (non-blocking): publish moderation-decisions event → analytics, training-export, trust-score update
```

Total inline budget consumed: ~20-25ms typical, well inside p99 150ms target (headroom absorbs Kafka publish jitter, cross-AZ hops).

**Voice path**: audio → VAD → ASR (300-600ms for a 1.5s segment at RTF~0.05, batched) → same text pipeline above → decision applied with ~1-2s total lag (acceptable per NFR).

## 19. Training Pipelines

- **Data sources**: (a) human-reviewed labels (uphold/overturn from HITL — highest quality), (b) player reports with outcome, (c) weak-supervision from word-list hits (noisy positive labels), (d) public toxicity datasets (Jigsaw, multilingual hate-speech corpora) for cold-start/long-tail languages.
- **Label quality tiers**: human-reviewed labels weighted highest in loss; weak-supervision labels down-weighted; active learning selects low-confidence/high-disagreement model predictions for prioritized human labeling.
- **Data prep**: dedup, PII scrubbing (strip usernames/emails/phone numbers from training text — replace with placeholders) before any external annotation vendor sees data; stratified sampling to balance rare severe categories (threats/grooming are <0.1% of raw traffic — oversample in training).
- **Point-in-time feature join**: as in section 15, join trust-score-at-message-time, not current.
- **Training orchestration**: Kubeflow Pipelines / Airflow DAG — steps: extract → validate schema/label distribution → dedup/PII-scrub → tokenize → distributed train → eval on holdout + adversarial eval set (known evasion patterns) → register in model registry if metrics pass gates.
- **Distributed training**: multilingual transformer fine-tune on 8-16 GPU nodes via PyTorch DDP/FSDP; base model pretraining (rare, ~quarterly) uses larger cluster (64+ GPUs), fine-tunes/patches are far cheaper (single 8-GPU node, hours not days).
- **Eval gates before promotion**: per-category F1 on holdout ≥ baseline, false-positive rate on a "benign banter" adversarial test set (curated competitive trash-talk corpus) must not regress — this set exists specifically to catch over-moderation regressions.

## 20. Retraining Strategy

- **Cadence**: scheduled fine-tune refresh every 2 weeks (incorporates newly reviewed labels); base model architecture refresh quarterly.
- **Triggers (event-driven, outside cadence)**:
  - Drift alert crosses threshold (section 21) → expedited retrain.
  - New evasion pattern cluster detected (spike in ANN near-miss queries not matching known-bad) → patch word-list/embeddings within 48h, full retrain within 7 days.
  - Regulatory/policy change (new category required, e.g., new law mandating detection of a specific harm class) → out-of-band retrain.
  - Major false-positive incident (viral complaint, escalation) → hotfix via threshold adjustment first (fast), retrain if root cause is model not policy.
- Canary + shadow evaluation required before any retrained model reaches >5% inline traffic (section 33).

## 21. Drift Detection

| Drift Type | Signal | Metric | Threshold / Action |
|---|---|---|---|
| Data drift (input) | Distribution of message length, language mix, token vocabulary (new slang) vs. training baseline | PSI (Population Stability Index) on token/n-gram frequency, KL divergence on language distribution | PSI > 0.2 → alert; > 0.3 → trigger expedited retrain review |
| Concept drift (label meaning changes) | Rising human-overturn rate on model's "block" decisions, or rising report-rate on "allow" decisions | Weekly overturn-rate trend, rolling 7-day | Overturn rate on blocks > 8% (vs. 3% baseline) → alert; sustained 2 weeks → mandatory retrain |
| Evasion drift | New obfuscation patterns bypassing fast-path + model | Rate of ANN near-miss (close to known-bad but below threshold) growth | Week-over-week growth > 25% in near-miss volume → alert, feeds embedding index refresh |
| Severe-category recall drift | Weekly audit sample (random + targeted) human re-review of model "allow" decisions | Sampled false-negative rate on severe categories | > 2% (NFR threshold) on 500-sample weekly audit → page on-call safety team |
| Locale/language coverage drift | Volume share of a language growing faster than model's training representation | Volume-share vs. training-corpus-share ratio | Ratio > 3x with volume > 1% of traffic → prioritize data collection for that locale |

## 22. Monitoring

- **Infra**: GPU utilization/queue depth per Triton pod, Kafka consumer lag per topic/partition, p50/p95/p99 latency per service hop, error rates, pod restart counts.
- **Model quality**: per-category precision/recall on rolling human-audited sample, calibration (score vs. actual overturn rate), overturn rate trend, model-vs-model agreement (canary vs. prod) during shadow periods.
- **Business metrics**: messages blocked/masked/allowed rate by title/locale, appeal volume and overturn rate, player churn correlation with moderation-action rate (cohort analysis), reviewer throughput and queue depth/wait time, CSAT on appeals.
- **Fairness/equity monitoring**: false-positive rate broken out by locale/language — must not disproportionately over-block any language community relative to its measured toxicity base rate.

## 23. Alerting

| Condition | Threshold | Routing |
|---|---|---|
| Inline decision latency p99 | > 150ms for 5 min | Page on-call SRE (chat-latency channel) |
| Kafka consumer lag (raw-chat-events) | > 30s sustained 5 min | Page on-call SRE |
| GPU fleet queue depth | > 2x steady-state for 3 min | Auto-scale trigger + info alert; page if scaling fails to reduce within 10 min |
| Review queue depth | > 12h backlog projected at current throughput | Page review-ops lead, may trigger threshold auto-tightening (fewer escalations) as stopgap |
| Severe-category false-negative audit | > 2% weekly sample | Page Trust & Safety on-call (not just SRE — policy/legal implication) |
| Overturn rate spike | > 8% blocks overturned, 24h rolling | Page ML on-call, likely threshold/model regression |
| Enforcement DLQ (exactly-once path) | Any message in DLQ | Page immediately — correctness-critical |
| Model canary metric regression | FP rate on benign-banter eval set +2pp vs. prod during shadow | Auto-block promotion, notify ML team (not paged, blocks rollout) |

- On-call routing: tiered — SRE for infra/latency, ML on-call for model-quality regressions, Trust & Safety on-call for severe-category misses (has legal/PR escalation path), Review-Ops for queue/backlog.

## 24. Logging

- **Structured logging**: JSON logs per decision with `message_id`, `model_version`, `scores`, `action`, `latency_ms`, `decision_engine_version` — no free-text message content in default logs (see PII below).
- **PII handling**: raw message text is NOT logged in general application logs; only referenced by `message_id` pointing to the Flagged Message Store (which itself has restricted access, encrypted at rest, and only retains content for messages that were flagged/reviewed — allowed messages' raw text is not retained beyond the transient processing window, typically 24-48h in Kafka before topic retention expiry).
- Access to raw flagged content gated by RBAC (reviewers, T&S investigators, legal-hold requests only) with full audit trail of who viewed what and when.
- **Retention**: transient Kafka raw events: 48h; flagged/reviewed message content: 180 days (appeals window) then archived to cold storage with legal-hold override; aggregated/anonymized metrics: indefinite; voice audio clips: 30 days hot, then cold storage or deletion per per-title data policy, max 180 days.
- Minors' data (COPPA-scoped titles) has separate, shorter default retention and stricter access controls per legal requirements.

## 25. Security

- **Threat model specifics**:
  - Adversarial evasion (players deliberately crafting text/audio to bypass classifiers) — mitigated by evasion-embedding pipeline, adversarial eval sets, rapid patch cycle.
  - Model extraction/probing (bad actors probing the API to reverse-engineer thresholds) — rate-limit and anomaly-detect repeated near-threshold probing patterns from a single account.
  - Data poisoning (coordinated brigading to "train" the model via mass false reports to get a target player wrongly flagged) — report-source diversity checks, anomaly detection on report patterns (many reports from accounts with no shared-session history with the reported player).
  - Insider threat on reviewer console (reviewer misusing access to view/leak flagged content) — full audit logging, least-privilege scoping (reviewers see only assigned queue items, not arbitrary player search without elevated role).
  - PII exposure via flagged content store — encryption at rest (AES-256), encryption in transit (mTLS internal, TLS 1.3 external), field-level encryption for any directly-identifying fields.
- **Data encryption**: at rest for all persistent stores; in transit via mTLS for service-to-service, TLS for client-facing.

## 26. Authentication

- **End-user (player) auth**: existing EA account/session token (OAuth2-based EA Account SSO), validated at Chat/Voice Gateway; short-lived session tokens, refreshed via existing platform auth infra — this system does not reinvent player auth, it consumes the platform's identity assertion.
- **Service-to-service auth**: mTLS with SPIFFE/SPIFFE-style workload identities within the cluster; internal service calls (Gateway → Kafka, Decision Engine → Feature Store) authenticated via short-lived certs issued by an internal CA, rotated automatically (e.g., via cert-manager + Vault).
- **Reviewer console auth**: SSO (corporate IdP, e.g. Okta/SAML) + RBAC roles (reviewer, senior-reviewer, T&S-investigator, admin) + mandatory MFA given access to sensitive content.
- **API-key/service-account auth** for batch/offline jobs (training pipeline pulling from data lake) scoped narrowly via IAM roles, least privilege.

## 27. Rate Limiting

- **Algorithm**: token bucket per player per channel for chat message rate limiting (independent of toxicity system, but toxicity system relies on gateway pre-filtering egregious spam before it reaches the ML path).
- **Per-user limits**: default chat 10 msgs/10s burst, 60/min sustained (tunable per title); voice has no message-count limit but VAD-based segment rate is naturally bounded by speech.
- **Report API rate limiting**: token bucket per reporter, 20 reports/hour default, to prevent brigading/false-report floods; sliding-window counter backing the anomaly detection in section 25.
- **Per-tenant (per-title) limits**: aggregate QPS caps to the shared inference fleet per title, to prevent one title's traffic spike (viral event) from starving another title's inline latency SLA — implemented via weighted fair queuing at the Decision Engine's request admission layer, not simple hard caps (avoid dropping legitimate traffic outright; degrade gracefully to fast-path-only mode under extreme load per title).
- Rate-limit rejections at gateway return fast (no ML call at all) — protects the GPU fleet from being a spam-amplification target.

## 28. Autoscaling

- **Text model fleet (Triton/GPU)**: KEDA-driven HPA scaling on Kafka consumer lag for `raw-chat-events` partitions assigned to the text-model consumer group (custom metric: lag per partition), target lag < 200ms-equivalent messages; scale-out in 30s increments, scale-in with 5 min cooldown to avoid thrash during bursty live-event traffic.
- **ASR fleet**: KEDA scaling on `raw-voice-segments` queue depth; ASR pods are the most expensive so scale-in cooldown is longer (10 min) to avoid GPU cold-start churn (model load time ~20-30s for Whisper-class model).
- **Fast-path filter (CPU)**: standard HPA on CPU utilization (target 60%) — cheap, scales fast, absorbs the bulk of burst traffic before it reaches GPU tier.
- **Decision Engine**: HPA on request QPS + p99 latency composite metric; VPA disabled here (latency-sensitive, want predictable pod sizing not right-sizing churn).
- **Review Console/backend**: scale on active-reviewer-session count, not raw QPS (human-paced workload).
- Pre-scaling: known live-service events (World Cup mode launch, FUT promo drops) trigger scheduled pre-scale (predictive scaling via CronJob nudging min-replica floor up 2h before event) rather than relying purely on reactive HPA, since GPU pod cold-start (image pull + model load) can take 60-90s — too slow for pure reactive scaling during a traffic cliff-edge.

## 29. Cost Optimization

- **Spot/preemptible instances**: used for offline/batch workloads only — training jobs, nightly re-embedding, audit-sampling backfill scoring. Never for the inline serving path (preemption would violate latency/availability SLA).
- **Model distillation**: primary text model is a distilled 66M-param model (from a larger teacher, ~340M) — captures ~97% of teacher's F1 at 5x lower latency/cost; ASR similarly uses Whisper-small distilled rather than large.
- **Quantization**: INT8 for text model cuts GPU/CPU compute cost ~3x with <0.3% accuracy loss (validated via eval gate).
- **Fast-path pre-filtering**: word-list/hash match short-circuits ~15-20% of traffic (obvious spam/profanity) before it ever reaches GPU inference — directly cuts GPU fleet size needed.
- **VAD-based ASR triage**: only forwarding actual speech segments (not silence/background) cuts ASR load ~70%; further triage (full ASR only on reported/flagged-history players + random audit sample) cuts another ~90% of remaining volume — this is the single largest cost lever given ASR dominates the compute bill (section 6).
- **Batching**: dynamic batching (5-10ms window) on GPU inference improves throughput/GPU 3-4x vs. unbatched, directly reducing GPU count needed.
- **Caching**: player trust-score cache avoids redundant feature-store reads at high QPS (cache-aside, 5 min TTL) — cuts feature-store read cost/load significantly.
- **Multi-tenancy**: shared GPU fleet across all EA titles rather than per-title dedicated fleets — smooths peak/trough (different titles peak at different times/regions), improving average utilization from ~35% (siloed) to ~65% (shared) estimate.
- **Tiered storage**: flagged content cold-archived after 30 days (object storage Glacier-class), voice clips similarly — cuts storage cost ~80% vs. keeping everything hot for the full 180-day retention.
- **Reviewer cost**: prioritization/triage (section 12) keeps HITL volume within the fixed 500-reviewer capacity rather than linearly scaling headcount with traffic growth — model confidence thresholds are the lever, not headcount, for cost control.

## 30. Operational Concerns (Deployment, Reliability, Infra)

At SDE2 scope, treat this as a checklist rather than a design exercise: **backups** (automated snapshots of the model registry, feature store, and any stateful service, with a tested restore path), **rollback** (every deploy must be revertible to the last-known-good version — the model registry and CI/CD pipeline should make this a one-command operation), **canary/blue-green rollout** (shift a small percentage of traffic first, watch error rate and key business/model metrics, then ramp), and **basic observability** (dashboards + alerts on latency, error rate, and the top 2-3 model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level infra-architecture concerns — worth knowing they exist, not worth rehearsing the manifests.

## 38. Why This Architecture

- **Streaming-first (Kafka) backbone**: decouples ingestion from scoring, enabling independent scaling of fast-path/GPU tiers and natural replay/audit capability without re-architecting for it later.
- **Layered scoring (fast-path → embedding/ANN → transformer → decision engine)**: cheapest, fastest checks run first, expensive GPU inference only invoked when needed — directly reduces cost (section 29) while preserving latency budget.
- **Separation of scoring from enforcement**: the Decision Engine's output is advisory-plus-policy, and the Enforcement Service is the sole authority on player state changes — this isolation lets model iteration happen fast (shadow/canary) without touching the strongly-consistent, legally-sensitive enforcement path.
- **HITL as a first-class component, not an afterthought**: given the asymmetric, culturally-sensitive cost of false positives in gaming (banter is core to the product), architecture explicitly budgets capacity and routing logic for human review rather than treating it as a fallback.
- **Regional co-location**: matches EA's existing game-server regional topology, avoiding a novel latency problem — moderation infra rides along existing regional chat/voice deployment rather than centralizing.

## 39. Alternative Architectures

| Alternative | Description | Why Rejected / When Preferred |
|---|---|---|
| Fully synchronous request/response (no Kafka backbone), gateway calls model service directly | Simpler, fewer moving parts | Rejected: no natural buffering for burst traffic (World Cup spikes), no replay for retraining, harder to add new consumers (analytics, drift) without touching the hot path. Would be preferred for a much smaller-scale system (single title, low traffic) where operational simplicity outweighs the scaling/decoupling benefit. |
| Client-side (on-device) pre-filtering only, no server-side ML | Run a small model on console/PC to filter before sending | Rejected as sole mechanism: trivially bypassed (modified clients, rooted devices), can't be updated as fast as evasion patterns evolve, no central audit trail for legal/appeals. Could be a *complementary* layer (reduce bandwidth/latency for obvious cases) but not a replacement for server-side authority. |
| Single monolithic "one big model does everything" (no fast-path, no separate ASR/text split) | One large multimodal model scores raw audio/text end to end | Rejected: worse latency (can't cheaply short-circuit obvious cases), harder to iterate/retrain independently per modality, harder to explain/audit per-category decisions for appeals. Might be preferred once multimodal models are cheap enough that the latency/cost gap closes and unified context (text+tone+audio) meaningfully improves accuracy — not yet at EA's latency/cost constraints. |
| Third-party moderation API (fully outsourced, e.g., a vendor SaaS) | Send all chat to an external moderation vendor | Rejected as primary path: unacceptable latency (network hop out of EA infra), data residency/privacy concerns at EA's volume, ongoing per-message cost at 10B/day scale is prohibitive, no control over game-specific slang/culture tuning. Reasonable as a *supplementary* signal or for a new/small title without dedicated ML investment yet. |

## 40. Tradeoffs

| Decision | Pros | Cons |
|---|---|---|
| Inline blocking for high-confidence severe cases vs. always-async | Stops harm before delivery, reduces exposure | Adds latency risk to chat critical path, requires very high availability of the scoring path (mitigated by fail-open) |
| Fail-open on ML outage (fallback to word-list only) | Preserves core chat availability | Temporarily degrades moderation quality during outage — accepted business risk with bounded RTO |
| Regional data sharding for enforcement with async cross-region replication | Meets local latency SLA, avoids single global bottleneck | Eventual consistency window (~30s) where a ban isn't yet globally visible |
| INT8 quantization for text model | 3x latency/cost improvement | Small (<0.3%) accuracy regression, requires careful eval-gate discipline to avoid death-by-a-thousand-cuts drift |
| Aggressive ASR triage (only reported/history/sample) | Massive cost reduction (dominant cost driver) | Reduced proactive detection coverage on voice for players with no prior signal — relies more on reports for first-time offenders |
| HITL review for ambiguous cases | Reduces false-positive player impact, builds trust | Adds latency for those cases (provisional mask+hold), bounded by reviewer capacity — creates a hard ceiling that must be managed via triage thresholds |
| Shared cross-title GPU fleet | Better utilization/cost | Requires careful per-tenant fair-queuing to avoid one title's traffic spike degrading another's SLA |
| Exactly-once semantics only for enforcement path | Correctness where it matters most | Added complexity (outbox pattern) not applied uniformly — team must remember which paths need it |

## 41. Failure Modes

| Scenario | Impact | Mitigation |
|---|---|---|
| GPU fleet fully down (region outage) | No ML scoring available | Fail-open to fast-path-only mode; alert Trust & Safety that proactive detection coverage is degraded; chat continues functioning |
| Kafka broker/partition unavailable | Message backlog builds, potential producer-side backpressure to gateway | Multi-AZ replication factor 3; gateway buffers briefly then sheds load gracefully (rate-limit tightens) rather than blocking player chat entirely |
| ASR service degraded (high latency) | Voice moderation lag exceeds 2s budget | Circuit breaker: beyond a latency threshold, voice segments route to a lower-cost fallback (smaller/faster ASR model) accepting reduced accuracy temporarily, alert fires |
| Model regression post-deploy (missed in canary) | Spike in false positives/negatives in production | Automated rollback triggers (section 34) on gate breach; worst case, manual kill-switch reverts to last-known-good version fleet-wide within minutes |
| Coordinated brigade / mass false reporting | Innocent player wrongly prioritized/actioned | Report-source anomaly detection (section 25), require diverse/independent report sources before escalation, human review required before any ban triggered purely by report volume without model corroboration |
| Enforcement store split-brain during failover | Double-enforcement or missed-enforcement | Postgres synchronous replica + automated failover with fencing; enforcement writes are idempotent keyed on `enforcement_id` |
| Review queue backlog runaway (viral event floods reports) | SLA breach on appeals/review, reviewer burnout | Auto-tighten inline thresholds temporarily (favor auto-action over escalation) as a release valve; surge/overflow reviewer pool activation runbook |
| Silent data drift (new slang undetected) | Gradual false-negative creep, unnoticed until audit | Weekly audit sampling (section 21) is the backstop specifically because drift can be silent in aggregate metrics |

## 42. Scaling Bottlenecks

**At 10x scale (e.g., EA acquires major new live-service title, +240M msgs/day → ~24B/day, voice concurrency to ~20M):**
- ASR fleet becomes untenable at current triage ratios — ~15,000 GPUs even with current triage; must push further (more aggressive sampling, cheaper streaming ASR, or on-device ASR for pre-filtering) or cost becomes the binding constraint.
- Review queue capacity (fixed at ~11.5M/day with 500 reviewers) becomes the hard ceiling — either headcount must scale linearly (expensive, slow to hire/train) or auto-action thresholds must tighten further (risking more false positives) — this is the first true bottleneck, hit before infra bottlenecks.
- Kafka partition count and consumer group sizing need a full re-partitioning exercise; naive partition-per-topic counts chosen for current scale won't linearly absorb 10x without rebalancing work.

**At 100x scale:**
- Enforcement store (single regionally-sharded Postgres per region) likely needs further horizontal sharding beyond simple player_id hash — approaching limits of vertical scaling on a single primary per shard.
- Cross-region replication lag (30s target) may not hold under 100x write volume — would need to revisit consistency model, possibly moving to a globally-distributed database (e.g., Spanner-class) for the enforcement store despite added cost/complexity.
- The entire "fast-path word-list first" cost-saving assumption weakens if adversarial evasion techniques scale/automate (bot-driven bypass attempts) — evasion-detection ANN index and embedding pipeline become primary load-bearing components, not supplementary.

## 43. Latency Bottlenecks

**p50 budget breakdown (text, inline path, ~20-25ms total):**
| Stage | p50 | p99 |
|---|---|---|
| Gateway auth + rate-limit + Kafka publish | 2ms | 8ms |
| Fast-path filter | 1ms | 3ms |
| Text model inference (incl. batching wait) | 8ms | 35ms |
| Evasion/ANN check (parallelized with model inference) | 3ms | 12ms |
| Decision Engine (trust-score fetch + policy apply) | 3ms | 10ms |
| Network/serialization overhead across hops | 3ms | 15ms |
| **Total** | **~20ms** | **~85-120ms** (within 150ms target, but batching wait tail is the main p99 risk) |

- Dominant p99 contributor: dynamic batching wait time on the text model under bursty load — mitigated by tighter batch-window tuning during known peak events and by fleet pre-scaling (section 28).
- Voice path dominant contributor: ASR inference time itself (300-600ms), not the downstream text-classification reuse — this is why the NFR tolerance for voice (p50 <800ms) is set separately and more loosely than text.

## 44. Cost Bottlenecks

- **#1 driver: ASR compute** — even after triage (VAD + sampling), the ASR fleet (~1,500 GPU-equivalents) dwarfs the text model fleet (~90 GPU-equivalents) in raw compute cost; any growth in voice adoption disproportionately drives cost.
- **#2 driver: human review labor** — 500 reviewers globally, 24/7, is a substantial recurring opex line; this scales roughly linearly with escalation volume, making threshold tuning a direct cost lever, not just a quality lever.
- **#3 driver: storage/retention** — flagged content + voice clips at 180-day retention across all regions; mitigated by tiered/cold storage but still a meaningful line item at EA's aggregate volume.
- **#4 driver: cross-region data transfer** — replication traffic (enforcement events, model artifacts) between regions; kept in check by keeping only the enforcement store's severe-path broadcast synchronous and everything else async/batched.
- Levers ranked by cost impact: (1) tighten ASR triage further, (2) improve auto-decision confidence to reduce human-review escalation rate without raising false-positive rate (requires better-calibrated models, not just threshold shifts), (3) storage lifecycle aggressiveness, (4) GPU utilization/batching efficiency.

## 45. Interview Follow-Up Questions

1. How would you handle a language EA has never supported before, launching in 6 weeks, with zero labeled data?
2. A famous streamer claims they were unfairly muted for competitive banter and it's going viral — walk me through how you'd investigate and what you'd change.
3. How do you prevent the model from learning to over-flag a particular dialect or accent as toxic due to biased training data?
4. Your review queue backlog just tripled overnight — what are your first three actions?
5. How would you detect a coordinated brigading attack in real time, as opposed to after the fact?
6. Walk me through exactly what changes in this architecture if the requirement shifts from "inline blocking" to "always post-hoc, message already delivered."
7. How do you handle a minor's account differently from an adult's account in this pipeline, technically?
8. What's your strategy if a court order / legal hold requires you to preserve specific flagged content indefinitely, conflicting with your retention policy?
9. How would you A/B test a stricter moderation policy's impact on player retention without just measuring block-rate?
10. If GPU costs for ASR became the #1 line item on the infra budget and leadership demands a 50% cut, what do you cut first and what do you tell them about the tradeoff?

## 46. Ideal Answers

1. **New language, 6 weeks, no data**: Bootstrap with the multilingual base model's zero-shot cross-lingual transfer plus a curated word-list fast-path, and route all traffic in that language through mandatory human review for the first weeks rather than trusting untuned thresholds. Use those human labels to fine-tune within 2-3 weeks, narrowing review percentage as confidence improves.

2. **Viral wrongful-mute incident**: Pull the `message_id` trace immediately for the full latency/score/decision breakdown, and check model version/canary cohort to see if it's a category calibration issue vs. an isolated policy misconfiguration. If systemic, expedite an overturn plus threshold hotfix and add the phrase pattern to the adversarial eval set so this regression class is gated in future releases.

3. **Dialect/accent bias**: Monitor false-positive rate broken out by locale/dialect explicitly rather than only in aggregate, since aggregate metrics hide disparate impact. Curate adversarial eval sets per dialect group for over-flagging of in-group slang, and use human-review overturn rate by dialect as a bias detector.

4. **Review backlog triples overnight**: Temporarily tighten inline auto-decision thresholds to shrink the fraction routed to human review (a time-boxed stopgap), and check overturn rate on the queue to tell a genuine volume spike apart from a pipeline bug over-escalating benign content. Activate surge reviewer capacity and prioritize by severity tier so worst cases still get fast attention.

5. **Real-time brigading detection**: Look at graph-level signals rather than per-message signals — many distinct reporters targeting one player in a short window with no shared match history, especially with near-identical report timing/content. Feed this as a streaming windowed job that raises a "coordinated-report cluster" signal requiring model-score corroboration before any auto-enforcement.

6. **Inline blocking → always async**: The latency SLA loosens dramatically and the Decision Engine's role shifts from gatekeeper to a post-hoc classifier feeding retroactive enforcement. The key tradeoff is that harmful content reaches recipients before action, so the likely answer is a hybrid: severe categories (grooming, threats) stay inline-blocking, everything else moves async.

7. **Minor accounts**: Stricter default thresholds, mandatory human review for a wider band of ambiguous cases, shorter retention windows per regulatory requirements, and tighter RBAC on flagged content involving minors. Use account age-band as a feature in the Decision Engine's policy layer, not just in training, to apply grooming-detection models preferentially.

8. **Legal hold vs. retention policy conflict**: Retention policy needs a legal-hold override built into the schema from day one — deletion jobs must check a hold-flag before running, and a hold on a `player_id`/`message_id` suspends normal expiry until released by legal. Treating this as an exception process instead is risky since already-deleted data can't be retroactively recovered for a legal request.

9. **A/B testing stricter policy impact on retention**: Block-rate alone is a vanity metric; measure session frequency and D7/D30 retention deltas between cohorts on stricter vs. current thresholds. Segment by whether a player received an action, since stricter moderation can simultaneously reduce toxicity-driven churn and increase false-positive-driven churn, and a net-neutral number could hide two offsetting effects.

10. **50% ASR cost cut mandate**: First lever is tightening triage (full ASR only on reported/escalating-trust-score sessions plus a random audit sample); second lever is a smaller/faster ASR model for the broad tier, reserving the accurate model for escalated cases. Present leadership a tradeoff table of cost saved vs. estimated increase in missed incidents, recommending risk-informed category cuts rather than a uniform reduction.
