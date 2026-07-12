# RAG Platform

## 1. Problem Framing

- Build an internal + partner-facing **RAG platform** at EA answering natural-language questions grounded in enterprise/game docs: engine docs (Frostbite), design wikis, live-ops runbooks, support KB, patch notes, compliance docs, per-title design docs (FIFA/FC, Apex, Battlefield, Sims).
- Consumers: internal engineers/support agents, designers, live-ops analysts, and later a player-facing bot in EA Help.
- Core ask: retrieve relevant chunks, rerank, feed to an LLM, return a grounded answer **with citations**, minimizing hallucination.
- Differs from generic search: correctness and traceability matter more than pure relevance — a wrong refund-policy or anti-cheat answer has real cost.
- This is a retrieval + generation pipeline, not a training problem. The interesting surface: ingestion freshness, chunking/embedding quality, ANN retrieval at scale, reranking cost/latency, hallucination containment.

## 2. Functional Requirements

- FR1: Continuously ingest heterogeneous docs (Confluence, SharePoint, PDFs, Markdown, Zendesk, Perforce).
- FR2: Chunk docs preserving semantic coherence (headers, tables, code blocks).
- FR3: Embed each chunk, upsert to vector DB with metadata (source, ACL tags, franchise, last-modified).
- FR4: Accept a query (optionally with conversation history), return a synthesized answer.
- FR5: ANN-retrieve top-K, apply metadata filters (ACL, franchise, freshness), rerank with a cross-encoder.
- FR6: Generate an answer via LLM with inline citations back to source docs.
- FR7: Detect/suppress hallucinated claims (citation-coverage check, groundedness scoring).
- FR8: Support multi-turn conversation with query rewriting/decontextualization.
- FR9: Feedback loop (thumbs up/down) feeding eval and reranker retraining.
- FR10: Enforce access control (e.g., CS agent can't see legal-privileged docs; player bot can't see internal docs).
- FR11: Admin/ops surface for re-indexing and per-source ingestion health.

## 3. Non-Functional Requirements

| Dimension | Target |
|---|---|
| p50 / p99 latency (query→answer) | 1.2s / 4.5s |
| Retrieval-only p99 | 150ms |
| Reranking p99 (top-50→top-8) | 250ms |
| Generation p99 | 800ms TTFT, ≤3.5s full |
| Availability (query / ingestion) | 99.9% / 99.5% |
| Peak QPS (global) | 450 sustained, 900 burst (patch-day) |
| Freshness (doc→searchable) | p50 ≤10min, p99 ≤2h |
| Consistency | Eventually consistent index; strongly consistent ACL metadata |
| Hallucination rate | ≤3% target, ≤5% alert |
| Cost/query (fully loaded) | ≤$0.006 blended |

## 4. Clarifying Questions

1. Internal-only or eventually player-facing (changes ACL/moderation/abuse surface massively)?
2. Corpus size and growth rate?
3. Multi-turn in scope for v1, or single-shot first?
4. Multilingual retrieval needed (EA has 20+ locales)?
5. Freshness requirement — can patch-day docs tolerate a 2h lag?
6. Self-host the LLM or call a third-party API (cost, residency, latency)?
7. Acceptable hallucination rate, and who audits it?
8. Regulatory constraints (GDPR erasure, COPPA if player-facing)?
9. Budget ceiling per query/month constraining model choice?
10. On-prem/air-gapped deployment needed for any studio?

## 5. Assumptions

1. 8M source docs → ~60M chunks (avg 512 tokens/chunk).
2. 90% internal query volume, 10% player-facing pilot.
3. 25,000 internal users, ~4,000 concurrent peak.
4. Player-facing pilot capped at 2M MAU.
5. Self-host an open-weight LLM (Llama-3.1-70B-class) for cost/residency, with hosted-frontier fallback for hard queries.
6. Embedding: 768-dim open-weight bi-encoder (BGE-large-class), self-hosted for cost at 60M-chunk scale.
7. Dedicated ANN vector DB, not bolted onto Postgres.
8. ACL tags come from source systems at ingestion; platform enforces, doesn't derive, ACLs.
9. Reranking is a cross-encoder over top-50 candidates.
10. Multi-region: US/EU/APAC, active-active reads, regional-primary writes.
11. Reranker/embedding retraining cadence is monthly, driven by feedback volume.

## 6. Capacity Estimation

**Query volume**: internal 72K/day + player-facing 208K/day ≈ 280K/day → avg 3.24 QPS. Peak factor 8x → **peak ~26 QPS steady, burst 450–900 QPS** (patch-day/incident spikes).

**Corpus & storage**: 8M docs × 8 chunks = 64M chunks. fp32 vectors (768-dim) ≈ 197GB raw; with HNSW overhead ~300–400GB. fp16/int8 quantization cuts this to ~100GB (small recall loss, chosen for cost). Metadata ~32GB separately.

**Embedding pipeline**: ~50K changed docs/day → ~400K chunks/day to re-embed. At ~800 chunks/sec/GPU, that's ~8 min/day of GPU time — trivial; batch hourly on 2 GPUs.

**Reranker**: 50 candidates/query × 26 QPS steady ≈ 1,300 pairs/sec; burst → 45,000 pairs/sec momentarily. At ~1,200 pairs/sec/GPU: 2 GPUs steady, 8–12 autoscaled for burst.

**LLM generation**: ~400 output tok, ~1,800 input tok/query. Self-hosted 70B on 4x A100 (vLLM, tensor-parallel) does ~1,800 tok/s/replica. Worst-case burst demand would need ~200 replica-equivalents; provision **12 replicas (48 A100s)** for steady/moderate burst, overflow beyond that to hosted API with a cost cap.

**Logs**: 280K queries/day × ~4KB ≈ 1.12GB/day → ~120GB/year compressed.

## 7. High-Level Architecture

```
SOURCE SYSTEMS (Confluence, SharePoint, Zendesk, Perforce, PDFs)
        │ webhook / scheduled crawl
        ▼
INGESTION CONNECTORS (per-source adapters, ACL extraction)
        ▼
INGESTION QUEUE (Kafka: docs.raw)
        ▼
CHUNKING & NORMALIZATION (parse, dedupe, header-aware split, table extraction)
        ▼
EMBEDDING SERVICE (GPU pool, versioned bi-encoder, batch inference)
        ▼
   ┌────────────────┴────────────────┐
   ▼                                  ▼
VECTOR DB (HNSW/IVF-PQ, sharded) ◄─joins─ METADATA/DOC STORE (SQL: ACL, source, version)
   ▲
   │ ANN top-K query
QUERY ORCHESTRATION SERVICE
  1. Query rewrite/decontextualize
  2. Embed query
  3. ANN retrieve top-50 (+ACL/metadata filter)
  4. Rerank (cross-encoder) → top-8
  5. Build grounded prompt w/ citations
  6. Call LLM (self-hosted or fallback API)
  7. Groundedness/citation-coverage check
  8. Return answer + citations
   │                          │
   ▼                          ▼
RERANKER SERVICE       LLM SERVING LAYER (vLLM self-host + hosted fallback)
   ▲
API GATEWAY (authn/authz, rate limit)
   ▲
CLIENTS: EA Help bot, internal CS console, Slack/Teams plugin

Cross-cutting: Feedback → Kafka(feedback.raw) → Eval/retraining pipeline
Cross-cutting: OTel monitoring/tracing across all services
```

## 8. Low-Level Components

| Component | Responsibility | Scaling Unit |
|---|---|---|
| Ingestion Connectors | Pull/webhook from sources, normalize, extract ACL | Per-source worker pool, horizontal |
| Chunking Service | Header-aware + sliding-window chunk, dedup via minhash | Stateless, CPU-bound, horizontal |
| Embedding Service | Batch-embed chunks, versioned bi-encoder | GPU pool, autoscale on queue depth |
| Vector DB | Store embeddings + ANN index, serve top-K w/ metadata filter | Sharded by chunk-hash, replicated |
| Metadata Store (SQL) | Source of truth for doc metadata, ACL, versioning | Read replicas, partitioned by franchise/source |
| Query Orchestration | Owns end-to-end query lifecycle | Stateless, horizontal |
| Reranker Service | Cross-encoder scoring of query+chunk pairs | GPU pool, autoscale on latency/queue |
| LLM Serving Layer | Token generation, streaming, continuous batching | GPU pool (tensor-parallel), autoscale |
| Groundedness Checker | Post-hoc claim-to-citation verification | Lightweight model pool, horizontal |
| Feedback Service | Capture thumbs-up/down, flags | Stateless, horizontal |
| Eval/Retraining Pipeline | Batch eval harness, fine-tune trigger | Batch, scheduled (Airflow) |
| API Gateway | AuthN/authZ, rate limiting, routing | Horizontal, stateless |

## 9. API Design

```
POST /v1/query
Request: { session_id, query, conversation_history, franchise_scope, max_citations, stream }
Response: { answer, citations: [{doc_id, title, url, chunk_span, score}],
            groundedness_score, model_version, retrieval_version, latency_ms }
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/query` | POST | Main Q&A, streaming supported |
| `/v1/feedback` | POST | Submit thumbs-up/down/flag |
| `/v1/sources/{id}/reindex` | POST | Admin: force re-ingestion |
| `/v1/sources/{id}/status` | GET | Ingestion health/freshness |
| `/v1/documents/{id}/chunks` | GET | Debug: inspect chunks |
| `/v1/eval/report` | GET | Latest groundedness/hallucination metrics |
| `/v1/admin/acl/refresh` | POST | Force ACL re-sync |

- Versioning: URI-based (`/v1/`), plus embedded `model_version`/`retrieval_version` for A/B routing.
- Auth: OAuth2 JWT bearer on all endpoints; admin requires `role=rag-admin`.

## 10. Database Design

| Store | Type | Why |
|---|---|---|
| Metadata/Doc Store | PostgreSQL | Strong consistency for ACL/versioning, relational joins |
| Vector DB | Purpose-built ANN (Milvus/Weaviate-class) | HNSW/IVF-PQ at 60M+ scale, sub-150ms p99 — pgvector doesn't scale here |
| Query Log Store | Columnar (ClickHouse) | High-volume append-only analytics |
| Feedback Store | Postgres | Needs joins to query log for eval |
| ACL Cache | Redis | Sub-ms ACL lookup on hot path |

```sql
CREATE TABLE documents (
  doc_id UUID PRIMARY KEY, source_system TEXT, franchise TEXT, title TEXT,
  url TEXT, acl_tags TEXT[], content_hash TEXT,
  last_modified TIMESTAMPTZ, ingested_at TIMESTAMPTZ, version INT
) PARTITION BY LIST (source_system);

CREATE TABLE chunks (
  chunk_id UUID PRIMARY KEY, doc_id UUID REFERENCES documents(doc_id),
  chunk_index INT, token_count INT, embedding_version TEXT, vector_shard INT
);
```

Doc-to-chunk fan-out and ACL live in Postgres (source of truth); vector DB holds only `chunk_id → vector + minimal filterable metadata` to support filtered ANN without a hot-path join.

## 11. Caching

| Cache | What | Invalidation |
|---|---|---|
| Query result cache | Full answer for identical (query, scope, ACL-class) | TTL 30min + purge on re-index of cited docs |
| Embedding cache | Query embedding | TTL 1h, keyed by `model_version+text_hash` |
| ACL cache | User → allowed ACL tags | Event-driven on ACL change + 15min TTL safety net |
| Reranked-context cache | Top-8 chunks for normalized query | TTL 5min + purge on doc re-index |
| LLM prefix cache | Shared system-prompt KV cache | Managed by vLLM automatically |

Cache key always includes `embedding_version`/`model_version` to avoid stale-format hits after upgrades. Full-answer hit rate ~12-18% (FAQ-like CS queries) — worth it since generation is the dominant cost.

## 12. Queues & Async Processing

| Topic | Producer → Consumer | Semantics |
|---|---|---|
| `docs.raw` | Connectors → Chunking | At-least-once, 3 retries → DLQ, alert if depth >500 |
| `chunks.embed` | Chunking → Embedding | At-least-once, DLQ + manual replay |
| `vector.upsert` | Embedding → Vector DB writer | At-least-once, idempotent upsert (`chunk_id+embedding_version`) |
| `feedback.raw` | Feedback API → Eval pipeline | At-least-once, DLQ + weekly review |
| `acl.change` | ACL webhook → cache invalidator | At-least-once + TTL safety net, nightly full re-sync fallback |

Exactly-once isn't pursued anywhere — idempotent upserts + TTL safety nets make at-least-once sufficient and far simpler than transactional guarantees across Kafka+DB. Ingestion is async/batch; query path is synchronous.

## 13. Streaming & Event-Driven Architecture

Key topics: `docs.raw`, `chunks.embed`, `vector.upsert` (also shadow-consumed for index-drift audits), `feedback.raw`, `acl.change`, `query.audit` (feeds ClickHouse sink + abuse detection).

Multiple consumer groups per topic where independent teams need the same stream. Schema registry (Avro/Protobuf) enforced, backward-compatible evolution only.

## 14. Model Serving

- **Embedding (bi-encoder)**: batched gRPC/Triton service, batch 64, 20ms max queue delay.
- **Reranker (cross-encoder)**: Triton, dynamic batching, batch 32, 15ms max delay (latency-sensitive).
- **LLM**: vLLM, continuous batching + PagedAttention, 70B-class open-weight, tensor-parallel across 4x A100-80GB. Prefix caching enabled.
- **Multi-model routing**: Tier 1 (default) self-hosted 70B; Tier 2 (escalation) hosted frontier API for complex/ambiguous queries or capacity overflow.
- **Hardware**: A10G for embedding/reranker, A100-80GB for LLM (needs HBM for weights + KV cache).
- Model artifacts versioned in a registry; canary new versions on 5% traffic before full rollout.

## 15. Feature Store

Not a classical tabular use case, but a lightweight "signals" layer feeds the query-complexity classifier and reranker auxiliary features (doc CTR, recency, franchise-affinity).

- **Offline**: nightly batch-computed doc-level signals (30-day CTR, feedback-derived quality score).
- **Online**: materialized to Redis for sub-ms lookup during reranking.
- **Point-in-time correctness**: feature snapshots timestamped (`valid_from`/`valid_to`); retraining joins feedback to features *as they existed at query time* to avoid leakage.

## 16. Vector Database

- **Indexing**: HNSW for hot/high-traffic franchises (recall >0.95, memory-hungry); IVF-PQ for cold/archival content (4-8x memory reduction, ~0.85-0.90 recall). Brute-force rejected — unusable at 60M scale.
- **Filtering**: pre-filter by routing to shards matching franchise/ACL scope, plus post-filter re-check for ACL correctness (defense in depth — never trust index-level filter alone for security).
- **Sharding**: hash on `chunk_id`, 3x replication; re-shard when a shard exceeds 5M vectors or breaches latency SLO.
- **Reindexing**: embedding-version bump triggers background re-embed + shadow-write, blue/green cutover (no downtime).

## 17. Embedding Pipelines

- **Model**: self-hosted bi-encoder (768-dim), fine-tuned on EA-domain query-doc pairs (support tickets↔KB, design-doc search logs).
- **Flow**: header-aware chunker → 512-token chunks, 15% overlap → batch embed (64) → L2-normalize → upsert.
- **Versioning**: every embedding carries `embedding_version`; dual-write old+new index during model upgrades, cut over after backfill + eval gate passes.
- **Backfill cadence**: full re-embed only on model version bump (~quarterly); incremental embed on every doc change (minutes-scale).

## 18. Inference Pipeline (Request Lifecycle)

```
Client → API Gateway (authn, rate limit)
  → Query Orchestration:
     1. Rewrite/decontextualize (~50ms)
     2. Embed query (~15ms, cache check)
     3. ANN retrieve top-50 w/ ACL+franchise filter (~80ms p99)
     4. Rerank top-50→top-8 (~200ms p99)
     5. Post-filter ACL re-check (~5ms)
     6. Build grounded prompt
     7. LLM generation, streamed (TTFT ~800ms, full 2-3.5s)
     8. Groundedness check (~150ms, overlaps with stream tail)
     9. Return answer + citations; emit query.audit async
```

Latency budget (p99≈4.5s): rewrite+embed+retrieve+rerank+prompt-build ≈475ms + generation 3,500ms (dominant) ≈ holds with ~500ms headroom.

Graceful degradation: if reranker is down, fall back to raw ANN top-8. If groundedness score <0.6: player-facing surface blocks with "no confident answer"; internal surface shows answer with a warning banner.

## 19. Training Pipelines

What's actually trained: (a) embedding bi-encoder fine-tune, (b) cross-encoder reranker fine-tune, (c) optional LoRA fine-tune of the generation LLM for tone, (d) query-complexity classifier (routes Tier 1 vs 2).

- **Data**: positive pairs from resolved tickets↔cited KB, search-log clicks, SME-curated pairs; hard negatives from BM25-retrieved-but-irrelevant and low-feedback-score chunks. PII scrubbed before landing in the training lake.
- **Orchestration**: Airflow DAG — extract → time-based train/val/test split (avoid leakage) → train on Kubernetes/Ray.
- **Distributed training**: reranker/embedding fine-tunes are single-node multi-GPU (4-8); LLM LoRA uses multi-node + DeepSpeed ZeRO-2 (8-16 A100s).
- **Eval gate**: new model must beat production on held-out groundedness/recall@k/NDCG by a pre-registered margin before canary eligibility.

## 20. Retraining Strategy

| Model | Cadence | Trigger |
|---|---|---|
| Embedding bi-encoder | Quarterly / ad hoc | Domain drift or recall@10 drop >5% |
| Cross-encoder reranker | Monthly | >50K new labeled pairs or NDCG@8 regression |
| Generation LLM (LoRA) | Quarterly | Tone drift, new base model, hallucination creep |
| Query-complexity classifier | Bi-weekly | Cheap, rolling 2-week feedback window |
| ACL/metadata sync | Continuous + nightly reconciliation | Not ML, but part of freshness story |

Retraining triggers early on feedback-volume/eval-regression signals; calendar cadence is the floor, not the ceiling.

## 21. Drift Detection

| Drift Type | Metric | Threshold |
|---|---|---|
| Query distribution drift | PSI on query-embedding clusters | >0.2 investigate, >0.3 alert |
| Corpus drift (new franchise/vocab) | % chunks from new source in 7 days | >15% → trigger eval refresh |
| Relevance drift | NDCG@8 vs baseline, weekly on golden set | Drop >5% absolute → alert |
| Answer-quality drift | Mean groundedness, 7-day rolling | <0.85 → alert |
| Hallucination drift | % unsupported claims | >5% → page, consider rollback |
| Feedback drift | 7-day rolling thumbs-down % | >10% (vs ~4% baseline) → alert |

Runs as a daily batch job against a frozen golden eval set (500-1,000 curated triples/franchise) plus continuous sampling of live traffic.

## 22. Monitoring

| Category | Metrics |
|---|---|
| Infra | GPU util/memory per pool, Kafka lag, vector DB shard latency/QPS, cache hit rates |
| Model quality | Recall@k, NDCG@8, groundedness distribution, citation-coverage %, hallucination rate |
| Business | Queries/day by surface, CS-deflection rate, thumbs-up rate, session length |
| Freshness | Doc→searchable lag, DLQ depth, ACL-sync lag |
| Cost | $/query, GPU-hour spend by pool, hosted-API fallback spend |
| Availability | Per-service uptime, 5xx rate, LLM timeout rate |

Dashboards segmented by franchise since query patterns and freshness needs differ.

## 23. Alerting

| Alert | Condition | Routing |
|---|---|---|
| Query p99 breach | >6s for 5 min | Sev2, page SRE |
| Hallucination spike | Sampled rate >5%/1h | Sev2, page ML on-call |
| Vector DB shard down | Replica count <2 | Sev1, page infra |
| Kafka consumer lag | >100K messages | Sev3, page data-platform |
| LLM GPU OOM/crash-loop | >3 restarts/10min | Sev2, page ML-infra |
| Groundedness drop | Rolling mean <0.80/30min | Sev2, auto-flag last deploy |
| ACL sync failure | DLQ >500 or reconciliation mismatch >0.1% | Sev1, page security |
| Cost anomaly | Hourly spend >150% of 7-day avg | Sev4, Slack only |

Sev1 = page immediately (security/outage); Sev2 = page (quality/latency); Sev3 = business-hours page; Sev4 = Slack notify.

## 24. Logging

- Structured JSON per request: `trace_id`, `session_id`, `query_id`, `model_version`, `retrieval_version`, `latency_breakdown_ms`, `groundedness_score`, `cited_doc_ids`.
- PII: player-facing queries scrubbed (regex+NER) before long-term storage; unredacted logs kept 7 days for incident debugging, then purged. Internal-employee logs retained 90 days, access-controlled.
- Retention: query audit logs 13 months (redacted after 7 days); raw ingestion logs 30 days; feedback logs retained indefinitely (PII-scrubbed).
- `trace_id` propagated via OTel across all services.

## 25. Security

Threat model specific to RAG:
- **Retrieval-based leakage**: ACL bypass via clever phrasing — mitigated by enforcing ACL at both index pre-filter AND post-filter, never relying on the LLM to withhold info.
- **Prompt injection via ingested docs**: a compromised doc contains "ignore previous instructions" — mitigated by treating retrieved content as untrusted data (never instructions) via strict templating, plus ingestion-time flagging of suspicious imperative content.
- **Citation spoofing**: model fabricates a citation — mitigated by groundedness checker verifying every citation maps to an actual retrieved chunk.
- **Player-facing exfiltration**: probing queries to extract internal content — mitigated by strict corpus segmentation (player-facing index only contains public-cleared docs).

Encryption: TLS 1.3 in transit; at-rest encryption on vector DB, Postgres, log stores (KMS-managed). Every doc tagged with a sensitivity class at ingestion; player-facing index excludes anything above "public."

## 26. Authentication

- End-user: OAuth2/OIDC via EA's IdP (SSO internal, EA Account for player-facing), short-lived JWTs (15min) + refresh tokens.
- Service-to-service: mTLS via service mesh, SPIFFE/SPIRE identities.
- Admin endpoints: MFA-backed session + `role=rag-admin` claim, separately audit-logged.

## 27. Rate Limiting

- Token bucket per identity at the API Gateway, backed by Redis.
- Internal user: 60/min burst, 600/hr sustained. Player-facing: 10/min burst, 100/hr. Per-tenant API key: default 300 QPM.
- Overflow: 429 + `Retry-After`; player-facing adds soft CAPTCHA after repeated 429s.
- Global circuit breaker caps total Tier-2 (expensive fallback) calls/hour to bound cost blast-radius.

## 28. Autoscaling

| Component | Metric | Policy |
|---|---|---|
| Query Orchestration | CPU + in-flight requests | HPA, min 6/max 60 |
| Reranker GPU pool | Queue depth + p99 latency | KEDA, min 2/max 12 GPUs |
| LLM Serving | GPU util + queue length | KEDA, min 16/max 48 GPUs, 2min up/10min down cooldown |
| Embedding Service | Kafka consumer lag | KEDA, min 1/max 4 GPUs |
| Vector DB | Manual/scheduled | Capacity-planned quarterly; read replicas autoscale on QPS |
| API Gateway | Request rate | HPA, min 4/max 40 |

LLM pool scale-up is slowest (model load ~3-5min) — mitigated by a warm minimum pool sized for typical peak, routing burst overflow to Tier-2 API instead of waiting on cold scale-up.

## 29. Cost Optimization

- Spot instances for batch embedding/retraining jobs; LLM serving stays on-demand (SLA-sensitive).
- Vector index quantization (int8/PQ): ~3-4x storage reduction, acceptable recall loss for long-tail content.
- Reranker distilled from a larger teacher model: ~40% GPU-hour reduction at comparable NDCG.
- Full-answer caching avoids repeat LLM generation cost, the highest-cost stage.
- Dynamic/continuous batching across embedding, reranker, LLM amortizes per-request overhead.
- Tiered model routing bounds % of traffic hitting the expensive Tier-2 path.
- Reserved/committed pricing for baseline GPU footprint; only burst is on-demand/spot.
- Archival franchises moved to IVF-PQ cold index instead of always-warm HNSW.

## 30. Operational Concerns

At SDE2 scope this is a checklist, not a design exercise: **backups** (automated snapshots of model registry/feature store/stateful services with a tested restore path), **rollback** (every deploy revertible in one command), **canary/blue-green rollout** (shift small traffic %, watch error rate and key model metrics, ramp), **basic observability** (dashboards + alerts on latency, error rate, top 2-3 model-quality signals). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level — worth knowing they exist, not worth rehearsing manifests.

## 31. Why This Architecture

- Separating retrieval (ANN) / reranking (cross-encoder) / generation (LLM) lets each stage scale on hardware suited to it, instead of one monolithic model doing everything.
- Self-hosting the primary LLM caps cost at EA's volume and satisfies data-residency/IP concerns a third-party API can't fully guarantee; hosted fallback preserves quality headroom for hard queries without paying frontier prices on all traffic.
- ACL enforcement at both index-filter and post-filter (not the LLM) treats access control as a hard security boundary, not a prompt-engineering problem.
- Async, event-driven ingestion decoupled from the synchronous query path means a broken source connector never affects query availability.
- Blue/green + canary for every model swap acknowledges RAG quality regressions are hard to catch in offline eval alone.

## 32. Alternative Architectures

| Alternative | Why Rejected / When Preferred |
|---|---|
| Naive RAG (frontier API, no rerank) | Rejected for cost at EA's volume and lower precision; preferred for a low-QPS prototype/MVP |
| Fine-tuned LLM, no retrieval | Rejected: knowledge goes stale, no citations, retraining a 70B model per doc-change is far costlier than incremental embedding updates |
| GraphRAG (knowledge-graph retrieval) | Rejected for v1 (ingestion complexity, immature tooling); revisit for multi-hop reasoning queries |
| Bi-encoder only, skip reranking | Rejected: precision at top-8 meaningfully worse without reranking; only acceptable if latency budget is extremely tight (<500ms) |

## 33. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Self-hosted 70B vs API-only | Lower marginal cost, data residency | Higher fixed infra cost, ops burden, slightly lower quality ceiling |
| HNSW (hot) + IVF-PQ (cold) hybrid | Cost-efficient per-franchise | Two index types, migration logic |
| At-least-once everywhere | Simpler, resilient | Requires idempotent design discipline, occasional duplicate work |
| ACL pre+post filter | Strong defense in depth | Extra ~5-10ms latency, engineering overhead |
| Blue/green cutovers | Instant, safe rollback | 2x resource cost during cutover |
| Tiered model routing | Bounds cost, keeps quality ceiling | Complexity classifier becomes a reliability dependency |
| Monthly/quarterly retraining | Predictable, cheap | Slower to adapt to sudden vocab shifts, mitigated by event triggers |

## 34. Failure Modes

| Failure | Mitigation |
|---|---|
| Vector DB shard loss (AZ outage) | Remaining replicas serve reads; auto-reprovision; alert if replica count <2 |
| Reranker outage | Fall back to raw ANN ranking, degraded but functional; auto-rollback bad deploy |
| LLM saturation (patch-day surge) | Overflow to Tier-2 up to cost cap; beyond that, queue with "high load" response |
| Stale ACL cache | Post-filter re-check against source-of-truth Postgres catches within request; nightly reconciliation catches residual drift |
| Prompt injection via malicious doc | Retrieved content always templated as untrusted context, never instructions |
| Embedding-version skew | Dual-write/shadow-index ensures atomic cutover via config flag |
| Hallucinated citation | Groundedness checker validates every citation against retrieved-chunk set, strips unverified ones |
| Cascading queue backlog | Backpressure via lag-based alerting, connector rate limiting, DLQ absorbs poison messages |

## 35. Scaling Bottlenecks

- **10x query volume (~2,800 QPS)**: LLM generation capacity is the first wall — GPU fleet grows from ~48 to several hundred A100s; Tier-2 cost becomes the forcing function for further self-hosted investment or distillation.
- **10x corpus (~600M chunks)**: shard count grows ~24→240; cross-shard fan-out latency pressures the 150ms budget, forcing better shard-routing or a more distributed-native vector DB.
- **100x query volume**: reranker throughput becomes a hard bottleneck even with batching — likely need smaller candidate sets or a cheaper late-interaction model (ColBERT-style).
- **100x corpus**: metadata Postgres strains on ACL-join-heavy ingestion queries — would need sharding or a purpose-built policy store.
- **Ingestion connector fan-out**: onboarding new studios doesn't bottleneck compute, it bottlenecks engineering throughput (bespoke connectors) — biggest practical constraint is people-time.

## 36. Latency Bottlenecks

- **p50 (~1.2s)**: generation ~950ms dominates; rewrite/embed/retrieve/rerank together ~215ms.
- **p99 (~4.5s)**: generation ~3,500ms dominates (tail-heavy from longer answers/queue contention); retrieval+rerank+overhead leaves ~500ms headroom.
- Generation is >75% of total latency at both percentiles — highest-leverage optimization target (speculative decoding, better batching, shorter answers).
- Reranking is the second-largest controllable chunk — candidate-set size is a direct lever (50→30 cuts latency roughly proportionally, modest recall cost).
- Retrieval (ANN) is cheap and stable, rarely the bottleneck below very large scale.

## 37. Cost Bottlenecks

- LLM generation GPU-hours dominate the bill (48 continuously-running A100s vs 2-4 GPUs each for embedding/reranker).
- Tier-2 hosted-API fallback is the second-largest and most volatile cost driver — a bad classifier or traffic spike can spike it fast; the hourly cost-cap circuit breaker exists to bound this.
- Vector DB memory footprint (HNSW hot-tier RAM) is a meaningful fixed cost — quantization and cold-tiering are the main levers.
- Cross-region replication egress is smaller but non-trivial, scales with corpus size/update frequency.
- Retraining compute is comparatively cheap (batch jobs, hours not continuous serving).

## 38. Interview Follow-Up Questions

1. How would you detect and quantify hallucination in production without expensive human review of every answer?
2. Walk through what happens end-to-end if the reranker service is completely down.
3. How do you prevent prompt injection in an ingested wiki page from leaking another team's confidential data?
4. Groundedness score is dropping steadily over two weeks but no threshold has fired — how do you catch this?
5. How would you redesign chunking if code-heavy engineering docs were being retrieved poorly vs prose-heavy wikis?
6. EA acquires a studio with 5M more docs — what breaks first, and how do you onboard without a latency regression?
7. How do you decide top-K for ANN retrieval and candidate-set size for the reranker?
8. How would you A/B test a new embedding model without risking a production quality regression?
9. Self-hosted LLM answers are consistently worse than hosted fallback for one query category — do you just route more to Tier 2?
10. How do you handle a GDPR right-to-erasure request for a doc already chunked, embedded, and cited in cached answers/logs?

## 39. Ideal Answers

1. **Hallucination detection**: automated citation-coverage check (NLI-style entailment or LLM-judge) verifying every claim maps to a retrieved, cited chunk, producing a per-answer groundedness score. Human spot-checks and a golden eval set calibrate/audit the automated judge, not act as the primary detector.

2. **Reranker outage**: soft dependency behind a circuit breaker — on failure, serve raw ANN top-8 unranked rather than fail the request, trading precision for availability. The fallback path stays alerted and time-bounded.

3. **Prompt-injection containment**: the prompt template hard-separates fixed system instructions from retrieved context, which is explicitly untrusted and used only for grounding. ACL filtering happens independently of the LLM, so a malicious doc can't leak content it was never given.

4. **Slow groundedness decay**: drift detection tracks a rolling trend and flags a statistically significant negative slope, not just absolute-threshold breaches. A weekly golden-eval re-run isolates true regression from query-mix shifts.

5. **Chunking redesign**: content-type-aware chunking treats code blocks/tables as atomic units while prose keeps header-aware splitting; evaluate whether the embedding model needs a code-aware variant.

6. **Onboarding a new studio**: first bottleneck is connector build time — solve with a templated, config-driven connector framework. Second is vector DB shard capacity — pre-provision shards, run the backfill as a rate-limited batch job decoupled from the live query path.

7. **Top-K/candidate-set tuning**: choose top-K to maximize recall@K while bounding reranker cost (scales linearly with candidates); tune against an offline eval set, picking the knee of the recall@K vs K curve.

8. **Safe embedding-model A/B**: build the new index fully in parallel (blue/green), validate with the golden eval set and shadow-traffic replay comparing top-8 overlap/groundedness, then canary-ramp with automated gates and the old index kept warm for rollback.

9. **Persistent quality gap**: root-cause first (retrieval failure, insufficient context, or genuine capability gap) rather than quietly routing more to Tier 2. Only a real, bounded capability gap should be permanently routed, as a deliberate, cost-monitored decision.

10. **GDPR erasure end-to-end**: cascade through every copy — tombstone the source doc, delete its chunks/embeddings from the vector DB (hard delete, not mark-inactive), purge cached answers citing it, redact from historical logs. Must be a defined, auditable runbook, since deleting only the source doc leaves it discoverable via caches/logs/stale index.
