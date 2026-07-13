# Toxicity Detection

## 1. Problem Framing

Design real-time toxicity detection for EA live-service games (FIFA chat, squad voice, forums):

- Score text/voice for toxicity (harassment, hate speech, threats, sexual content, grooming) in-line with gameplay.
- Support 25+ languages.
- Feed a human-in-the-loop (HITL) review queue for borderline/appealed cases.
- Minimize false positives (banter/trash-talk is core to game culture) while catching true harms (child safety, legal/regulatory exposure under DSA/OSA).

This is a trust & safety system with an asymmetric, politically sensitive cost function: false negatives → legal/PR/child-safety risk; false positives → churn and "EA silenced me" backlash.

## 2. Functional Requirements

- FR1: Classify text into toxicity categories (harassment, hate, sexual, threats, self-harm, spam) with per-category scores.
- FR2: Classify voice chat via ASR + text classification, near-real-time.
- FR3: Configurable per-title/per-region policy thresholds.
- FR4: Inline enforcement (allow/mask/block/flag-for-review) before delivery for high-confidence cases; post-hoc for async cases.
- FR5: Route ambiguous/severe/appealed cases to human review with context (conversation window, player history).
- FR6: Support player reports (priority queue) and proactive detection.
- FR7: Handle multilingual, code-mixed text, leetspeak, emoji-obfuscation, evolving slang.
- FR8: Feed enforcement outcomes back into training data.
- FR9: Every decision auditable/explainable for appeals and legal.
- FR10: Support shadow-mode deployment for new models.

## 3. Non-Functional Requirements

| Dimension | Target |
|---|---|
| Inline text latency | p50 < 50ms, p99 < 150ms |
| Voice latency | p50 < 800ms end-to-end (buffering tolerated) |
| Availability | 99.95% inline path; never block chat if ToxDetect is down |
| Throughput | 400K msgs/sec peak; 150K concurrent voice streams |
| Consistency | Eventual for enforcement history; strong for ban/mute state (avoid double-punish races) |
| Cost | < $0.000015/message amortized (~$150K/mo at 10B msgs/month) |
| False-positive rate | < 0.5% of allowed messages overturned |
| False-negative rate (severe) | < 2% miss rate on threats/grooming, audited weekly |
| Freshness | Retrain/patch cycle < 7 days for new slang/evasion |

## 5. Assumptions

1. ~120M MAU across titles; chat-enabled titles ~60M MAU.
2. ~40 msgs/session, 2 sessions/day → ~2.4B msgs/day, peaking 4x during events → design for 10B/day headroom.
3. Voice adoption ~15% of chat-active MAU concurrently at peak (~2M transmitting at any instant).
4. Inline block-before-send for high-confidence severe text categories; voice tolerates 1-2s buffer.
5. 30 languages cover 95% of volume; long tail routes to a fallback multilingual model + human review.
6. Existing word-list filter kept as cheap first-pass, not replaced.
7. Enforcement decisions retained 180 days for appeals/legal; anonymized aggregates indefinitely.
8. GPU fleet is shared cross-title infra.
9. ~500 reviewers globally, 24/7, ~90 sec/review.
10. False-positive cost weighted ~3x false-negative for borderline categories, inverted (10x) for severe (CSAM/grooming/threats).

## 6. Capacity Estimation

**Text**: 2.4B/day ≈ 28K/sec avg; peak 112K/sec; provision to 400K/sec for burst headroom.

**Voice**: ~2M concurrent streams at peak. VAD (voice-activity detection) cuts raw load ~70% (only speech segments forwarded) → ~200K utterance-segments/sec to classify.

**Model & compute**:
- Text classifier: distilled multilingual transformer (~66M params, DistilBERT-class), INT8 quantized (~70MB) — runs on CPU or small GPU slice, ~2ms/inference batched on T4.
- ~6,000 inferences/sec per T4 GPU (batch=32) → need ~67 GPUs at 400K msgs/sec peak; provision ~90 with headroom.
- ASR: Whisper-small-class model (~240M params) on A10G. Raw load would need ~15,000 GPU-equivalents — ASR is the dominant cost driver. Mitigate via VAD (already applied) plus triage: full ASR only for reported players + trust-score escalations + random audit sample, cutting load to ~10% of raw → **~1,500 GPUs**, still the largest fleet.
- Toxicity-on-transcript reuses the same 66M param text model — negligible incremental cost vs. ASR.

**Storage**: Only flagged content retained (~2% of volume). Text: ~48M records/day, ~1KB each → ~8.6TB over 180 days. Voice clips: ~2M/day, ~200KB compressed → ~72TB over 180 days (cold storage after 30 days). Feature store (player trust/history): ~60M players × 2KB → 120GB in fast KV.

**Human review**: 500 reviewers × 24h / 90 sec ≈ 11.5M reviews/day capacity — this bounds how much can be escalated to HITL (see routing in §12).

## 7. High-Level Architecture

```
Game Clients (Text/Voice SDK, Report Button)
   │ gRPC/WebSocket        │ RTP/WebRTC         │ REST
   ▼                       ▼                    ▼
Chat Gateway          Voice Ingest Gateway   Report API
(auth, rate-limit)    (jitter buffer, VAD)
   │                       │                    │
   └──────► Kafka: raw-chat-events / raw-voice-events ◄──┘
                        │
                        ▼
          Fast-Path Filter (CPU: word-list/hash match)
             │ pass-through          │ hit → immediate block
             ▼
   ASR Service (voice, GPU) ───► Text Toxicity Model (multilingual, INT8, batched)
                                          │ scores
                                          ▼
                       Decision Engine / Policy Svc
                (per-title/locale thresholds + player trust score)
             │ allow/mask/block            │ ambiguous/severe
             ▼                             ▼
   back to Gateway                 Review Queue (Kafka + priority scheduler)
                                          ▼
                                   HITL Review Console
                                          ▼
                                   Enforcement Service (mute/ban/appeal state)
                                          ▼
                          Player Trust/Enforcement Store (KV/SQL)
                                          ▼
                          Feature Store + Training Data Lake
```

## 8. Low-Level Components

| Component | Responsibility | Scaling Unit |
|---|---|---|
| Chat Gateway | Auth, rate limit, protocol termination, publish to Kafka | Stateless, scale on connections |
| Voice Ingest Gateway | RTP termination, jitter buffer, VAD segmentation | Scale on concurrent streams |
| Fast-Path Filter | Hash/regex match on known-bad terms | CPU, scale on msg/sec |
| ASR Service | Speech-to-text on VAD segments | GPU, scale on queue depth |
| Text Toxicity Model | Multi-label scoring | GPU/CPU, scale on queue depth + latency SLA |
| Decision Engine | Apply thresholds, blend model score + trust score, decide action | Stateless, must be <5ms p99 |
| Review Queue Scheduler | Priority ranking (severity, report count, tier) | Partition by severity tier |
| HITL Review Console | Reviewer UI + context | Stateless web tier |
| Enforcement Service | Apply mute/ban, idempotent state changes | Strongly consistent store |
| Player Trust Store | System of record for standing/history | Sharded by player_id |
| Feature Store | Online + offline player features | Partition by player_id |
| Training Pipeline | Periodic/triggered retraining | Job-based GPU cluster |
| Drift Monitor | Track input/output distribution shift | Scheduled batch |

## 9. API Design

```
POST /v1/moderate/text
Request:  { message_id, player_id, title_id, locale, channel, text,
            context: { recent_messages, reported_before } }
Response: { message_id, action: allow|mask|block|review,
            scores: { harassment, hate, sexual, threat },
            model_version, decision_latency_ms }

POST /v1/moderate/voice-segment
Request:  { call_id, player_id, title_id, audio_ref, duration_ms }
Response: { transcript, scores, action, asr_confidence }

POST /v1/report
Request:  { reporter_id, reported_player_id, title_id, evidence_ref, category }
Response: { report_id, status, priority }

GET  /v1/review/queue?reviewer_id=&locale=&limit=20
POST /v1/review/{item_id}/decision  { decision: uphold|overturn, notes }

GET  /v1/players/{player_id}/enforcement-status
POST /v1/appeals  { player_id, enforcement_id, reason }
```

Versioning: URI-based (`/v1/`); `model_version` returned separately for shadow/canary tracking. Deprecation via `Sunset` header, 90-day notice.

## 10. Database Design

| Store | Data | Type | Why |
|---|---|---|---|
| Player Trust/Enforcement Store | Ban/mute state, strikes | PostgreSQL | ACID needed — no double-ban races |
| Flagged Message Store | Flagged text + scores + decisions | Columnar (ClickHouse) | Analytical queries, append-only |
| Voice Clip Store | Flagged audio snippets | Object storage + metadata | Large blobs, lifecycle policies |
| Review Queue | Pending items | Redis sorted sets | Low-latency pop/peek, ephemeral |
| Feature Store (online) | Trust/behavioral features | KV (DynamoDB/Redis) | Sub-5ms reads |
| Feature Store (offline) | Historical features | Parquet/Iceberg on lake | Batch scale, point-in-time joins |
| Word-list | Known bad terms | Small KV, replicated to edge | In-memory needed for fast-path |
| Model Registry | Versions, eval metrics, rollout state | Relational (Postgres) | Small, transactional |

## 11. Caching

| Cache | Strategy | Invalidation |
|---|---|---|
| Player trust score | Cache-aside, Redis, TTL 5 min | Event-driven on new enforcement + TTL backstop |
| Word-list | In-memory per filter pod | Push-based refresh via pub/sub |
| Model artifacts | Loaded at pod startup | New version → blue/green rollout |
| Policy/threshold config | In-memory + Redis, TTL 60s | Config service publishes change event |
| Recent-conversation context | Redis ring buffer | Natural eviction / TTL on channel close |

## 12. Queues & Async Processing

| Queue | Purpose | Semantics |
|---|---|---|
| `raw-chat-events` | Ingested text | At-least-once, dedup by message_id, DLQ after 3 retries |
| `raw-voice-segments` | VAD-segmented audio refs | At-least-once, DLQ reviewed on-call |
| `review-queue` | Human review items | At-least-once, dedup on consume; enrichment failures → default-to-conservative action |
| `enforcement-actions` | Mute/ban/appeal changes | Exactly-once (idempotent producer + outbox pattern to Postgres) |
| `training-data-export` | Labeled data export | At-least-once, dedup via merge-on-write |

Exactly-once is reserved for enforcement state only (double-ban/missed-unban is a correctness/legal issue); everywhere else, cheaper idempotent-by-`message_id` dedup suffices.

## 13. Streaming Architecture

| Topic | Producer | Key Consumers |
|---|---|---|
| `raw-chat-events` | Chat Gateway | fast-path-filter, text-model, analytics |
| `raw-voice-segments` | Voice Gateway | ASR, analytics |
| `moderation-decisions` | Decision Engine | enforcement, review-queue, analytics, training-export |
| `player-reports` | Report API | review-queue, trust-score |
| `enforcement-events` | Enforcement Svc | trust-score, notifications, analytics |
| `model-eval-events` | Retraining pipeline | model-registry, alerting |

`raw-chat-events` partitioned by `player_id` (preserves per-player ordering for context window); `raw-voice-segments` by `call_id`. Text-model consumer group scales to ~90 GPU-backed consumers matching partition count. Schema registry (Avro/Protobuf) enforced; breaking changes require a new topic version.

## 14. Model Serving

- Triton Inference Server for GPU models (dynamic batching, ONNX/TensorRT). Fast-path filter is a lightweight non-ML service.
- Dynamic batching: 5-10ms window, max batch 32 for text; ASR batches on VAD segment arrival, max 16.
- Multi-model fleet: primary multilingual text model + a few specialized severe-category models (grooming, self-harm) run in ensemble for high recall, plus ASR.
- Hardware: T4/A10G for text (INT8 quantized); A10G/L4 for ASR. Spot instances only for offline/batch scoring, never inline.
- Precision: INT8 for text (<0.3% accuracy drop, 3x latency win); FP16 for ASR (quantization hurts WER on slang/accents too much).
- Canary: shadow traffic mirrored to candidate model version without affecting production decisions.

## 15. Feature Store

- Online (Redis/DynamoDB): trust score, rolling report count, prior strikes, account age — <5ms reads for decision blending.
- Offline: Parquet/Iceberg for training joins.
- Point-in-time correctness: features joined to labels as they existed at message time, not current state (prevents leakage — e.g., can't use "later banned" to predict earlier-message toxicity).
- Streaming pipeline (Flink/Kafka Streams) writes both online and offline from the same computation to avoid train/serve skew.

## 16. Vector Database

Used for near-duplicate/evasion detection, not primary classification.

- Detects obfuscated toxic phrases (leetspeak, homoglyphs, spacing tricks) and coordinated raids (near-identical harassment from many players).
- Normalized text embedded (small sentence-embedding model), ANN search against known-bad embeddings.
- HNSW chosen over IVF-PQ: recall matters more than memory here, corpus is modest (low tens of millions of vectors), and query latency must stay single-digit ms.
- Index refreshed daily from newly confirmed toxic (human-upheld) messages.
- Supplementary signal into the Decision Engine, not a replacement for the transformer classifier.

## 17. Embedding Pipelines

Narrow scope: supports the vector-DB evasion detector and cross-lingual generalization.

- Normalization: lowercase, homoglyph-folding, leetspeak-normalization, emoji-to-text before embedding.
- Small multilingual embedding model (~30M params), shared across the evasion detector and as an auxiliary input to the main classifier.
- Nightly batch job re-embeds the confirmed-toxic corpus and refreshes the ANN index.
- Real-time embedding computed inline (same INT8 model, <3ms) — no separate round-trip.

## 18. Inference Pipelines

**Text (inline path):**

```
Client sends message
  → Gateway: authn, rate-limit, publish to Kafka [~2ms]
  → Fast-Path Filter: hash match [~1ms]  (hit → immediate block, skip ML)
  → Text Toxicity Model: tokenize + batch-infer [~8-15ms incl. batching wait]
  → Evasion/ANN check (parallel with model inference) [~3ms]
  → Decision Engine: fetch trust score, blend signals, apply policy [~3ms]
      ├─ allow  → deliver
      ├─ mask   → deliver redacted
      ├─ block  → drop, notify sender
      └─ review → provisional mask+hold, publish to review-queue; reviewer
                   decision retroactively confirms/overturns
  → async: publish moderation-decisions event (analytics, training, trust-score)
```

Total: ~20-25ms typical, well within p99 150ms target.

**Voice**: audio → VAD → ASR (300-600ms for a 1.5s segment, batched) → same text pipeline → decision applied with ~1-2s total lag (within NFR).

## 19. Training Pipelines

- Data sources: human-reviewed labels (highest quality), player reports with outcome, weak supervision from word-list hits, public toxicity datasets for cold-start/long-tail languages.
- Label weighting: human-reviewed highest, weak-supervision down-weighted; active learning prioritizes low-confidence/high-disagreement predictions for labeling.
- Data prep: dedup, PII scrubbing before any vendor annotation; oversample rare severe categories (<0.1% of raw traffic).
- Point-in-time feature join (as in §15).
- Orchestration (Kubeflow/Airflow): extract → validate → scrub → tokenize → distributed train → eval on holdout + adversarial evasion set → register if gates pass.
- Distributed training: fine-tunes on 8-16 GPU nodes (DDP/FSDP); rare full pretraining uses a larger cluster.
- Eval gates: per-category F1 ≥ baseline; false-positive rate on a curated "benign banter" adversarial set must not regress — this specifically catches over-moderation.

## 20. Retraining Strategy

- Cadence: fine-tune refresh every 2 weeks; base model refresh quarterly.
- Event-driven triggers: drift alert breach → expedited retrain; new evasion cluster detected → patch within 48h, full retrain within 7 days; regulatory change → out-of-band retrain; major FP incident → threshold hotfix first, retrain if root cause is the model.
- Canary + shadow evaluation required before any retrained model exceeds 5% inline traffic.

## 21. Drift Detection

| Drift Type | Signal | Action |
|---|---|---|
| Data drift | Token/language distribution vs. baseline (PSI, KL divergence) | PSI > 0.2 alert, > 0.3 expedited retrain review |
| Concept drift | Rising overturn rate on blocks, or report rate on allows | Overturn rate > 8% (vs. 3% baseline) sustained 2 weeks → mandatory retrain |
| Evasion drift | ANN near-miss volume growth | WoW growth > 25% → alert, feeds index refresh |
| Severe-category recall | Weekly human audit of "allow" decisions | FN rate > 2% on 500-sample audit → page on-call safety team |
| Locale coverage drift | Language volume share vs. training-corpus share | Ratio > 3x with volume > 1% traffic → prioritize data collection |

## 22. Monitoring

- Infra: GPU utilization/queue depth, Kafka consumer lag, p50/p95/p99 latency per hop, error rates.
- Model quality: per-category precision/recall on human-audited sample, calibration, overturn rate trend, canary-vs-prod agreement.
- Business: block/mask/allow rate by title/locale, appeal volume and overturn rate, churn correlation with moderation-action rate, reviewer throughput/queue wait, appeal CSAT.
- Fairness: false-positive rate by locale/language — must not disproportionately over-block any language community relative to its measured toxicity base rate.

## 23. Alerting

| Condition | Threshold | Routing |
|---|---|---|
| Inline latency p99 | > 150ms for 5 min | Page SRE |
| Kafka consumer lag | > 30s sustained 5 min | Page SRE |
| GPU queue depth | > 2x steady-state 3 min | Auto-scale + page if unresolved in 10 min |
| Review queue depth | > 12h projected backlog | Page review-ops, may auto-tighten thresholds |
| Severe-category FN audit | > 2% weekly sample | Page Trust & Safety on-call |
| Overturn rate spike | > 8% 24h rolling | Page ML on-call |
| Enforcement DLQ | Any message | Page immediately (correctness-critical) |
| Canary FP regression | +2pp vs. prod on benign-banter set | Auto-block promotion, notify ML team |

Tiered routing: SRE (infra), ML on-call (model quality), Trust & Safety (severe-category misses — legal/PR path), Review-Ops (queue backlog).

## 24. Logging

- Structured JSON logs: `message_id`, `model_version`, `scores`, `action`, `latency_ms` — no raw message text in default logs.
- Raw text referenced by `message_id` pointing to the Flagged Message Store; allowed messages' text isn't retained beyond the transient Kafka window (~24-48h).
- RBAC-gated access to flagged content (reviewers, T&S, legal-hold) with full audit trail.
- Retention: Kafka raw events 48h; flagged content 180 days (appeals window) then cold archive; aggregated metrics indefinite; voice clips 30 days hot, max 180 days.
- Minors' data (COPPA-scoped titles): shorter retention, stricter access.

## 25. Security

- Adversarial evasion → mitigated by evasion-embedding pipeline, adversarial eval sets, rapid patch cycle.
- Model probing (reverse-engineering thresholds) → rate-limit and anomaly-detect near-threshold probing.
- Data poisoning (brigading via mass false reports) → report-source diversity checks, anomaly detection on report patterns.
- Insider threat (reviewer misuse) → full audit logging, least-privilege queue scoping.
- Encryption: at rest (AES-256) for all stores, in transit (mTLS internal, TLS 1.3 external).

## 26. Authentication

- Player auth: existing EA Account SSO (OAuth2), validated at gateway — this system consumes platform identity, doesn't reinvent it.
- Service-to-service: mTLS with workload identities, short-lived certs via internal CA (cert-manager + Vault).
- Reviewer console: corporate SSO (Okta/SAML) + RBAC roles + mandatory MFA.
- Batch/offline jobs: scoped service accounts, least privilege.

## 27. Rate Limiting

- Token bucket per player per channel for chat (gateway-level, independent of toxicity scoring): default 10 msgs/10s burst, 60/min sustained.
- Report API: 20 reports/hour per reporter, to prevent brigading/false-report floods.
- Per-title aggregate QPS caps on the shared inference fleet, via weighted fair queuing (not hard caps) so one title's spike doesn't starve another's SLA — degrades to fast-path-only under extreme load.
- Rate-limit rejections happen at the gateway, before any ML call — protects GPU fleet from spam amplification.

## 28. Autoscaling

- Text model fleet: KEDA HPA on Kafka consumer lag, target <200ms-equivalent lag; scale-out in 30s, 5 min scale-in cooldown.
- ASR fleet: KEDA on queue depth; longer 10 min scale-in cooldown (model load ~20-30s, avoid cold-start churn).
- Fast-path filter: standard CPU-utilization HPA (target 60%) — absorbs most burst traffic before GPU tier.
- Decision Engine: HPA on QPS + p99 latency composite; predictable sizing preferred over VPA churn.
- Review console: scales on active-reviewer-session count.
- Pre-scaling: known live-service events (World Cup drop, FUT promo) trigger scheduled pre-scale ahead of time, since GPU cold-start (60-90s) is too slow for pure reactive scaling.

## 29. Cost Optimization

- Spot instances: offline/batch only (training, re-embedding, audit backfill) — never inline.
- Model distillation: 66M-param text model captures ~97% of a 340M teacher's F1 at 5x lower cost; ASR similarly uses Whisper-small.
- INT8 quantization: ~3x compute cost cut, <0.3% accuracy loss.
- Fast-path pre-filtering short-circuits ~15-20% of traffic before GPU inference.
- VAD + ASR triage (report-history + audit sample only) is the single largest cost lever — cuts ASR load ~90% beyond VAD alone, since ASR dominates the compute bill.
- Dynamic batching improves GPU throughput 3-4x.
- Trust-score caching cuts feature-store read load.
- Shared cross-title GPU fleet smooths peak/trough across titles, raising utilization from ~35% (siloed) to ~65% (shared).
- Tiered storage: cold-archive after 30 days cuts storage cost ~80%.
- Reviewer cost controlled via triage/confidence thresholds, not linear headcount scaling.

## 30. Operational Concerns

At SDE2 scope, treat this as a checklist: backups (model registry, feature store, tested restore path), rollback (one-command revert to last-known-good), canary/blue-green rollout (shift a small % of traffic, watch key metrics, ramp), and basic observability (dashboards + alerts on latency, error rate, top model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level concerns — worth knowing they exist, not worth rehearsing.

## 31. Why This Architecture

- Streaming-first (Kafka) backbone decouples ingestion from scoring, enabling independent scaling and replay/audit without re-architecting later.
- Layered scoring (fast-path → embedding/ANN → transformer → decision engine): cheapest checks run first, GPU inference only when needed — cuts cost while preserving latency budget.
- Scoring separated from enforcement: Decision Engine is advisory, Enforcement Service is sole authority on player state — lets model iteration move fast (shadow/canary) without touching the strongly-consistent, legally-sensitive enforcement path.
- HITL is first-class, not a fallback: given the asymmetric, culturally-sensitive cost of false positives in gaming, architecture explicitly budgets capacity and routing for human review.
- Regional co-location rides along EA's existing regional chat/voice deployment rather than centralizing.

## 32. Alternative Architectures

| Alternative | Why Rejected / When Preferred |
|---|---|
| Synchronous request/response, no Kafka backbone | Rejected: no buffering for burst traffic, no replay for retraining. Fine for a smaller single-title system where simplicity outweighs decoupling benefits. |
| Client-side-only pre-filtering | Rejected as sole mechanism: trivially bypassed, can't update as fast as evasion evolves, no audit trail. Fine as a complementary layer, not a replacement. |
| Single monolithic multimodal model (no fast-path/ASR split) | Rejected: worse latency, harder to iterate/retrain per modality, harder to audit per-category. Might work once multimodal inference is cheap enough — not yet at EA's constraints. |
| Fully outsourced third-party moderation API | Rejected as primary: latency hop, data residency/privacy concerns, prohibitive per-message cost at 10B/day, no game-specific tuning. Reasonable as a supplementary signal or for a small title without ML investment yet. |

## 33. Tradeoffs

| Decision | Pros | Cons |
|---|---|---|
| Inline blocking for severe cases vs. always-async | Stops harm before delivery | Adds latency risk to chat critical path; needs very high availability (mitigated by fail-open) |
| Fail-open on ML outage (word-list only) | Preserves chat availability | Degrades moderation quality during outage |
| Regional sharding, async cross-region replication | Meets local latency SLA | Eventual consistency window (~30s) before a ban is globally visible |
| INT8 quantization | 3x latency/cost win | Small accuracy regression, needs eval-gate discipline |
| Aggressive ASR triage | Massive cost reduction | Reduced proactive coverage for first-time voice offenders with no prior signal |
| HITL for ambiguous cases | Reduces FP impact, builds trust | Added latency; hard ceiling from reviewer capacity |
| Shared cross-title GPU fleet | Better utilization/cost | Needs fair-queuing to stop one title's spike degrading another's SLA |
| Exactly-once only for enforcement | Correctness where it matters most | Complexity not applied uniformly — team must track which paths need it |

## 34. Failure Modes

| Scenario | Mitigation |
|---|---|
| GPU fleet fully down (region outage) | Fail-open to fast-path-only; alert T&S that proactive coverage is degraded; chat keeps working |
| Kafka broker/partition unavailable | Multi-AZ replication factor 3; gateway sheds load gracefully rather than blocking chat |
| ASR degraded (high latency) | Circuit breaker routes to a smaller/faster fallback ASR model, alert fires |
| Model regression post-deploy | Automated rollback on gate breach; manual kill-switch reverts fleet-wide within minutes |
| Coordinated brigade / mass false reporting | Report-source anomaly detection; require model corroboration before any report-volume-triggered ban |
| Enforcement store split-brain during failover | Synchronous replica + fenced automated failover; writes idempotent on `enforcement_id` |
| Review queue backlog runaway | Auto-tighten inline thresholds as a release valve; surge reviewer pool runbook |
| Silent data drift (new slang) | Weekly audit sampling is the backstop since aggregate metrics can hide this |

## 35. Scaling Bottlenecks

**At 10x scale** (~24B msgs/day, ~20M voice concurrency): ASR fleet becomes untenable even at current triage ratios — needs more aggressive sampling, cheaper streaming ASR, or on-device pre-filtering. Review queue capacity (fixed ~11.5M/day at 500 reviewers) becomes the first true bottleneck — hit before infra limits — forcing either headcount scaling or tighter auto-action thresholds. Kafka partitioning needs a full rebalancing exercise.

**At 100x scale**: Enforcement store needs sharding beyond simple player_id hash. Cross-region replication lag may not hold, possibly requiring a globally-distributed database despite added complexity. The "word-list-first" cost assumption weakens if evasion techniques become automated/bot-driven — the ANN evasion pipeline becomes load-bearing, not supplementary.

## 36. Latency Bottlenecks

**Text inline path, p50 ~20-25ms total:**

| Stage | p50 | p99 |
|---|---|---|
| Gateway (auth, rate-limit, publish) | 2ms | 8ms |
| Fast-path filter | 1ms | 3ms |
| Text model inference (incl. batching wait) | 8ms | 35ms |
| Evasion/ANN check (parallel) | 3ms | 12ms |
| Decision Engine | 3ms | 10ms |
| Network/serialization overhead | 3ms | 15ms |
| **Total** | **~20ms** | **~85-120ms** (within 150ms target) |

Dominant p99 contributor: dynamic batching wait under bursty load — mitigated by tighter batch-window tuning and pre-scaling. Voice's dominant contributor is ASR inference itself (300-600ms), which is why voice has a separate, looser latency NFR.

## 37. Cost Bottlenecks

1. ASR compute — even after triage, ~1,500 GPU-equivalents dwarfs the text fleet (~90); voice adoption growth disproportionately drives cost.
2. Human review labor — 500 reviewers, 24/7, scales roughly linearly with escalation volume; threshold tuning is a direct cost lever.
3. Storage/retention — flagged content + voice clips at 180-day retention; mitigated by tiered/cold storage.
4. Cross-region data transfer — kept in check by making only the enforcement path's severe-case broadcast synchronous.

Levers ranked by impact: (1) tighten ASR triage, (2) improve auto-decision confidence to cut escalation rate without raising FP rate, (3) storage lifecycle aggressiveness, (4) GPU batching efficiency.

## 38. Interview Follow-Up Questions

1. How would you handle a language EA never supported, launching in 6 weeks, with zero labeled data?
2. A famous streamer claims unfair mute for banter and it's going viral — walk through your investigation and fix.
3. How do you prevent the model from learning to over-flag a dialect or accent as toxic?
4. Review queue backlog tripled overnight — first three actions?
5. How would you detect coordinated brigading in real time, not after the fact?
6. What changes in this architecture if the requirement shifts from inline blocking to always post-hoc?
7. How do you handle a minor's account differently in this pipeline, technically?
8. A legal hold requires preserving specific flagged content indefinitely, conflicting with retention policy — what's your strategy?
9. How would you A/B test a stricter moderation policy's impact on retention, beyond block-rate?
10. Leadership demands a 50% cut to ASR GPU cost — what do you cut first, and what's the tradeoff conversation?

## 39. Ideal Answers

1. **New language, 6 weeks, no data**: Bootstrap with the multilingual base model's zero-shot transfer plus a curated word-list fast-path, and route all traffic in that language through mandatory human review initially rather than trusting untuned thresholds. Fine-tune on those labels within 2-3 weeks, narrowing review percentage as confidence improves.

2. **Viral wrongful-mute**: Pull the `message_id` trace for the full score/decision breakdown, check model/canary cohort for a calibration issue vs. isolated policy misconfiguration. If systemic, expedite an overturn plus threshold hotfix and add the pattern to the adversarial eval set so this regression class is gated going forward.

3. **Dialect/accent bias**: Track false-positive rate broken out by locale/dialect, not just in aggregate (aggregate hides disparate impact). Curate per-dialect adversarial eval sets for in-group slang over-flagging, and use overturn rate by dialect as a bias detector.

4. **Backlog triples overnight**: Temporarily tighten inline auto-decision thresholds (time-boxed stopgap) to shrink review volume. Check overturn rate to distinguish a genuine spike from a pipeline bug over-escalating. Activate surge reviewer capacity, prioritize by severity.

5. **Real-time brigading detection**: Use graph-level signals, not per-message — many distinct reporters targeting one player in a short window with no shared match history. Stream this as a windowed job raising a "coordinated-report cluster" signal requiring model corroboration before auto-enforcement.

6. **Inline → always async**: Latency SLA loosens dramatically; Decision Engine shifts from gatekeeper to post-hoc classifier feeding retroactive enforcement. Likely answer: hybrid — severe categories (grooming, threats) stay inline-blocking, everything else moves async.

7. **Minor accounts**: Stricter default thresholds, mandatory human review for a wider ambiguous band, shorter retention, tighter RBAC on flagged content. Use account age-band as a Decision Engine policy feature to preferentially apply grooming-detection models.

8. **Legal hold vs. retention**: Build a legal-hold override into the schema from day one — deletion jobs check a hold-flag before running; a hold suspends normal expiry until legal releases it. Treating this as an ad hoc exception is risky since already-deleted data can't be recovered.

9. **A/B test stricter policy**: Block-rate alone is a vanity metric; measure session frequency and D7/D30 retention deltas between cohorts, segmented by whether a player received an action — stricter moderation can simultaneously cut toxicity-driven churn and raise false-positive-driven churn, and a net-neutral number can hide two offsetting effects.

10. **50% ASR cost cut**: First lever is tighter triage (full ASR only for reported/escalating-trust sessions plus a random audit sample); second is a smaller/faster ASR model for the broad tier, reserving the accurate model for escalated cases. Present a cost-vs-missed-incidents tradeoff table and recommend risk-informed category cuts over a uniform reduction.
