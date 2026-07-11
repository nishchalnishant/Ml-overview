# RAG Platform

## 1. Problem Framing & Requirement Gathering

- Build an internal + partner-facing **Retrieval-Augmented Generation (RAG) platform** at EA that answers natural-language questions grounded in enterprise and game documentation: engine docs (Frostbite, internal tooling), design wikis, live-ops runbooks, support KB articles, patch notes, compliance/legal policy docs, and per-title design docs (FIFA/EA SPORTS FC, Apex Legends, Battlefield, The Sims).
- Primary consumers: internal engineers/support agents (Tier-1/Tier-2 CS), game designers, live-ops analysts, and (later) a player-facing support-bot surface embedded in EA Help.
- Core ask: given a query, retrieve the most relevant document chunks, rerank them, feed them to an LLM, and return a grounded answer **with citations**, minimizing hallucination.
- Framing distinguishes this from generic search: correctness and traceability matter more than pure relevance ranking — a wrong answer about refund policy or anti-cheat escalation has real cost.
- This is a **retrieval + generation pipeline**, not a model-training problem — the interesting system-design surface is ingestion freshness, chunking/embedding quality, ANN retrieval at scale, reranking cost/latency tradeoffs, and hallucination containment.

## 2. Functional Requirements

- FR1: Ingest heterogeneous documents (Confluence, SharePoint, PDFs, Markdown, Zendesk KB, Perforce-hosted design docs) on a continuous basis.
- FR2: Chunk documents into retrieval units preserving semantic coherence (headers, tables, code blocks).
- FR3: Generate embeddings per chunk and upsert into a vector database with metadata (source, title, ACL tags, title/franchise, last-modified).
- FR4: Accept a natural-language query, optionally with conversation history, and return a synthesized answer.
- FR5: Retrieve top-K candidate chunks via ANN search, apply metadata filters (ACL, franchise, freshness), and rerank with a cross-encoder.
- FR6: Generate an answer via LLM conditioned on reranked context, with inline citations mapping back to source documents/URLs.
- FR7: Detect and suppress hallucinated or unsupported claims (citation-coverage check, groundedness scoring).
- FR8: Support multi-turn conversations with context carry-over and query rewriting/decontextualization.
- FR9: Provide a feedback loop (thumbs up/down, "was this grounded?") feeding back into eval and retraining of the reranker.
- FR10: Support access control — a support agent cannot retrieve legal-privileged docs; a player-facing bot cannot retrieve internal-only content.
- FR11: Expose an admin/ops surface for re-indexing, source-connector health, and per-source ingestion SLAs.

## 3. Non-Functional Requirements (latency, availability, throughput, consistency, cost)

| Dimension | Target |
|---|---|
| End-to-end p50 latency (query → answer) | 1.2s |
| End-to-end p99 latency | 4.5s |
| Retrieval-only p99 | 150ms |
| Reranking p99 (top-50 → top-8) | 250ms |
| Generation (LLM) p99 (streamed first token) | 800ms TTFT, full answer ≤ 3.5s |
| Availability (query path) | 99.9% (≈8.7h/yr downtime) |
| Availability (ingestion path) | 99.5% (best-effort, async) |
| Throughput (peak query QPS, global) | 450 QPS sustained, 900 QPS burst (patch-day support spikes) |
| Freshness (doc → searchable) | p50 ≤ 10 min, p99 ≤ 2h for large batch re-crawls |
| Consistency model | Eventually consistent index (read-your-writes not guaranteed within seconds); strongly consistent ACL metadata |
| Hallucination rate (unsupported claims / total claims, sampled eval) | ≤ 3% target, ≤ 5% alert threshold |
| Cost per query (fully loaded: embedding + retrieval + rerank + generation) | ≤ $0.006 blended target |

## 4. Clarifying Questions an Interviewer Would Expect You to Ask

1. Who are the consumers — internal-only, or eventually player-facing (changes ACL, moderation, and abuse surface massively)?
2. What's the corpus size and growth rate — hundreds of thousands vs tens of millions of documents?
3. Is multi-turn conversation in scope for v1, or single-shot Q&A first?
4. What languages must be supported — EA operates in 20+ locales; does retrieval need multilingual embeddings?
5. What's the source-of-truth freshness requirement — can patch-day docs tolerate a 2-hour ingestion lag?
6. Do we own the LLM (self-hosted open-weight model) or call a third-party API (cost, data residency, latency implications)?
7. What's the acceptable hallucination/incorrect-citation rate, and who audits it?
8. Are there regulatory/compliance constraints (GDPR right-to-erasure on ingested docs, COPPA if player-facing)?
9. Is there a budget ceiling per query or per month that constrains model choice (GPT-4-class vs smaller open-weight)?
10. Do we need on-prem/air-gapped deployment for any studio (some EA studios have restricted networks)?

## 5. Assumptions

1. Corpus size at launch: 8M source documents → ~60M chunks after chunking (avg 512 tokens/chunk).
2. 90% of query volume is internal (engineers, CS agents, live-ops); 10% is a pilot player-facing surface embedded in EA Help.
3. Total addressable internal users: 25,000 EA employees/contractors; concurrent active users during business hours peak: ~4,000.
4. Player-facing pilot: capped at 2M MAU (EA Help visitors funneled to bot), not all 700M+ registered EA accounts.
5. We self-host an open-weight LLM (Llama-3.1-70B-class, fine-tuned/instruction-tuned) for cost and data-residency control, with a fallback to a hosted frontier model for complex queries.
6. Embedding model: a 768-dim open-weight bi-encoder (e.g., BGE-large-class), self-hosted, chosen over closed API embeddings for cost at 60M-chunk scale.
7. Vector DB: dedicated ANN store (not bolted onto Postgres) due to scale and QPS.
8. Documents carry ACL tags at ingestion time from source systems (Confluence space permissions, SharePoint groups) — platform enforces, does not derive, ACLs.
9. Reranking uses a cross-encoder over top-50 candidates, not the full corpus.
10. Multi-region: US (primary, us-east), EU (data residency + latency), APAC (latency) — active-active for read/query, active-passive-ish for ingestion writes with regional primaries.
11. Retraining cadence for the reranker/embedding fine-tune is monthly, not continuous, driven by feedback-loop volume.

## 6. Capacity Estimation

**Query volume**
- Internal DAU ~12,000, avg 6 queries/user/day → 72,000 queries/day internal.
- Player-facing pilot: 2M MAU, ~8% DAU/MAU → 160,000 DAU, avg 1.3 queries/session → 208,000 queries/day.
- Total: ~280,000 queries/day → avg QPS = 280,000 / 86,400 ≈ 3.24 QPS average.
- Peak factor 8x (patch-day + business-hours concentration) → **peak ≈ 26 QPS steady, burst to 450–900 QPS during major incident/patch events** (support surge, matches NFR table — sized for incident-driven spikes, not steady state).

**Corpus & embedding storage**
- 8M documents, avg 8 chunks/doc (512 tokens each) → 64M chunks.
- Embedding dim 768, fp32 → 768 × 4 bytes = 3,072 bytes/vector.
- Raw vector storage: 64M × 3,072B ≈ 196.6 GB.
- With HNSW index overhead (~1.5–2x graph structure) → ~300–400 GB total index size.
- Using fp16/int8 quantization (PQ or scalar quantization) cuts this to ~100 GB with minor recall loss — chosen for cost.
- Metadata (doc ID, ACL tags, source, timestamps, ~500 bytes/chunk) → 64M × 500B = 32 GB in a separate metadata store.

**Embedding pipeline throughput**
- Steady-state re-ingestion: ~50,000 changed/new docs/day (patch notes, wiki edits) → ~400,000 chunks/day to re-embed.
- Embedding batch throughput on 1x A10G: ~800 chunks/sec (batch 64, seq len 512, bi-encoder). 400,000 / 800 ≈ 500s ≈ 8.3 min/day of GPU time — trivially small; batch it hourly, 2 GPUs for headroom + backfills.

**Reranker throughput**
- Cross-encoder rerank: 50 candidates/query × 26 QPS peak steady = 1,300 pairs/sec sustained; burst 900 QPS → 45,000 pairs/sec momentarily.
- Cross-encoder (MiniLM/BGE-reranker-large class) on A10G: ~1,200 pairs/sec/GPU (batched). Steady state: 2 GPUs; burst autoscaled pool: 8–12 GPUs behind HPA.

**LLM generation capacity**
- Avg generation: 400 output tokens/answer, 1,800 input tokens (query + 8 reranked chunks + system prompt) context.
- Self-hosted 70B model on 4x A100-80GB (tensor-parallel) via vLLM: ~35–45 concurrent sequences/replica at these lengths, throughput ~1,800 output tok/s/replica (continuous batching).
- Peak burst 900 QPS × 400 tok ≈ 360,000 tok/s demand → need ≈ 360,000/1,800 ≈ 200 replica-equivalents at absolute worst case; realistically burst is short-lived and overflow routes to hosted frontier API — provision **12 self-hosted replicas (48 A100s)** for steady/moderate burst, overflow beyond that to third-party API with cost cap.

**Storage for logs/conversations**
- 280K queries/day × ~4KB (query+answer+citations+metadata) = 1.12 GB/day → 410 GB/year raw, compressed ~120 GB/year.

## 7. High-Level Architecture

```
                                   ┌─────────────────────────────────────────┐
                                   │            SOURCE SYSTEMS                │
                                   │ Confluence | SharePoint | Zendesk KB     │
                                   │ Perforce design docs | PDFs | Wikis      │
                                   └───────────────────┬───────────────────────┘
                                                        │ webhooks / scheduled crawl
                                                        ▼
                                   ┌─────────────────────────────────────────┐
                                   │        INGESTION CONNECTORS             │
                                   │  (per-source adapters, ACL extraction)  │
                                   └───────────────────┬───────────────────────┘
                                                        ▼
                                   ┌─────────────────────────────────────────┐
                                   │   INGESTION QUEUE (Kafka: docs.raw)     │
                                   └───────────────────┬───────────────────────┘
                                                        ▼
                        ┌───────────────────────────────────────────────────────┐
                        │         CHUNKING & NORMALIZATION SERVICE               │
                        │ (parse, dedupe, header-aware split, table extraction)  │
                        └───────────────────┬─────────────────────────────────────┘
                                             ▼
                        ┌───────────────────────────────────────────────────────┐
                        │            EMBEDDING SERVICE (GPU pool)                │
                        │       bi-encoder, batch inference, versioned           │
                        └───────────────────┬─────────────────────────────────────┘
                                             ▼
              ┌──────────────────────────────┴───────────────────────────┐
              ▼                                                          ▼
 ┌───────────────────────────┐                          ┌───────────────────────────────┐
 │      VECTOR DATABASE       │                          │   METADATA / DOC STORE (SQL)   │
 │  (HNSW/IVF-PQ, sharded)    │◄────────────joins────────│  ACL, source, freshness, ver.  │
 └───────────┬────────────────┘                          └───────────────────────────────┘
             ▲
             │  ANN top-K query
             │
 ┌───────────┴────────────────────────────────────────────────────────────────────┐
 │                             QUERY ORCHESTRATION SERVICE                        │
 │  1. Query rewrite/decontextualize (uses conv history)                          │
 │  2. Embed query                                                                │
 │  3. ANN retrieve top-50 (+ ACL/metadata filter)                                │
 │  4. Rerank (cross-encoder) → top-8                                             │
 │  5. Build grounded prompt w/ citations                                         │
 │  6. Call LLM (self-hosted or fallback API)                                     │
 │  7. Groundedness/citation-coverage check                                       │
 │  8. Return answer + citations                                                  │
 └───────────┬─────────────────────────────────────────────────────────┬──────────┘
             ▼                                                         ▼
  ┌────────────────────┐                                   ┌───────────────────────┐
  │  RERANKER SERVICE   │                                   │   LLM SERVING LAYER    │
  │  (cross-encoder GPU)│                                   │ vLLM (self-host 70B)  │
  └────────────────────┘                                   │ + hosted API fallback  │
                                                             └───────────────────────┘
             ▲
             │ gRPC/REST
 ┌───────────┴───────────────────┐
 │        API GATEWAY             │
 │  authn/authz, rate limit       │
 └───────────┬───────────────────┘
             ▲
  ┌──────────┴───────────┐
  │  CLIENTS: EA Help bot,│
  │  internal CS console, │
  │  Slack/Teams plugin    │
  └───────────────────────┘

     Cross-cutting: Feedback events → Kafka(feedback.raw) → Eval/retraining pipeline
     Cross-cutting: Monitoring/Tracing (OTel) across all services
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| Ingestion Connectors | Pull/webhook from source systems, normalize to canonical doc envelope, extract ACL tags | Source-specific SDK in, Kafka `docs.raw` out | Per-source-type worker pool, horizontal |
| Chunking Service | Parse doc structure, header-aware + sliding-window chunk, table/code preservation, dedup via minhash | Kafka in/out | Stateless workers, CPU-bound, horizontal |
| Embedding Service | Batch-embed chunks with versioned bi-encoder | gRPC batch endpoint, Kafka consumer | GPU pool, autoscale on queue depth |
| Vector DB | Store embeddings + ANN index, serve top-K search with metadata filter | gRPC/REST search API | Sharded by chunk-hash, replicated per shard |
| Metadata Store (SQL) | Source-of-truth for doc metadata, ACL, versioning, freshness | SQL/gRPC | Read replicas, partitioned by franchise/source |
| Query Orchestration Service | Owns end-to-end query lifecycle, coordinates retrieval/rerank/generation | REST/gRPC (external), internal gRPC (downstream) | Stateless, horizontal, CPU-bound orchestration |
| Reranker Service | Cross-encoder scoring of query+chunk pairs | gRPC batch | GPU pool, autoscale on latency/queue |
| LLM Serving Layer | Token generation, streaming, continuous batching | OpenAI-compatible REST (vLLM) | GPU pool (tensor-parallel replicas), autoscale on queue+GPU util |
| Groundedness Checker | Post-hoc claim-to-citation verification (NLI-style or LLM-judge) | Internal gRPC | Lightweight model pool, horizontal |
| Feedback Service | Capture thumbs-up/down, flagged answers | Kafka producer, REST | Stateless, horizontal |
| Eval/Retraining Pipeline | Batch eval harness, reranker/embedding fine-tune trigger | Airflow DAGs | Batch, scheduled |
| API Gateway | AuthN/authZ, rate limiting, request routing | REST/gRPC edge | Horizontal, stateless |

## 9. API Design

```
POST /v1/query
Request:
{
  "session_id": "uuid",
  "query": "How do I escalate a ranked-mode desync bug for Apex?",
  "conversation_history": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}],
  "franchise_scope": ["apex_legends"],
  "max_citations": 5,
  "stream": true
}

Response (200, streamed SSE if stream=true):
{
  "answer": "To escalate a ranked-mode desync bug...",
  "citations": [
    {"doc_id": "conf-88213", "title": "Apex Live-Ops Escalation Runbook", "url": "...", "chunk_span": [120,410], "score": 0.87}
  ],
  "groundedness_score": 0.94,
  "model_version": "rag-llm-v3.2",
  "retrieval_version": "embed-v5",
  "latency_ms": 1180
}
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/query` | POST | Main Q&A endpoint, streaming supported |
| `/v1/feedback` | POST | Submit thumbs-up/down/flag on an answer |
| `/v1/sources/{source_id}/reindex` | POST | Admin: force re-ingestion of a source |
| `/v1/sources/{source_id}/status` | GET | Ingestion health/freshness for a source |
| `/v1/documents/{doc_id}/chunks` | GET | Debug: inspect chunks for a doc |
| `/v1/eval/report` | GET | Latest eval-harness groundedness/hallucination metrics |
| `/v1/admin/acl/refresh` | POST | Force ACL re-sync from source system |

- **Versioning**: URI-based (`/v1/`), plus response-embedded `model_version`/`retrieval_version` for observability and A/B routing. Breaking changes → `/v2/`; additive fields are non-breaking.
- Auth: `Authorization: Bearer <OAuth2 JWT>` on all endpoints; admin endpoints require `role=rag-admin` claim.

## 10. Database Design

| Store | Type | Why | Partition/Shard Key |
|---|---|---|---|
| Metadata/Doc Store | PostgreSQL (managed, e.g. Aurora) | Strong consistency needed for ACL and versioning; relational joins between doc/source/franchise | Partitioned by `source_system`, sub-partitioned by `franchise` |
| Vector DB | Purpose-built ANN store (e.g. Milvus/Weaviate-class, self-hosted or managed) | Needs HNSW/IVF-PQ at 60M+ vector scale with sub-150ms p99; Postgres+pgvector doesn't scale ANN recall/latency at this size | Sharded by `hash(chunk_id) % N`, each shard replicated 3x |
| Conversation/Query Log Store | Columnar (e.g. ClickHouse) | High-volume append-only analytics queries (latency percentiles, groundedness trends) | Partitioned by day, ordered by `(franchise, timestamp)` |
| Feedback Store | Postgres (small volume, relational) | Needs joins to query log for eval pipeline | Partitioned by month |
| ACL Cache | Redis (materialized from Postgres) | Sub-ms ACL lookup on hot path | Sharded by `user_id` hash |

**Metadata table sketch:**
```sql
CREATE TABLE documents (
  doc_id UUID PRIMARY KEY,
  source_system TEXT NOT NULL,
  franchise TEXT,
  title TEXT,
  url TEXT,
  acl_tags TEXT[],
  content_hash TEXT,
  last_modified TIMESTAMPTZ,
  ingested_at TIMESTAMPTZ,
  version INT
) PARTITION BY LIST (source_system);

CREATE TABLE chunks (
  chunk_id UUID PRIMARY KEY,
  doc_id UUID REFERENCES documents(doc_id),
  chunk_index INT,
  token_count INT,
  embedding_version TEXT,
  vector_shard INT
);
```

- Doc-to-chunk fan-out and ACL live in Postgres (source of truth); vector DB holds only `chunk_id → vector + minimal filterable metadata` (ACL tags, franchise) to support filtered ANN search without a round-trip join on the hot path.

## 11. Caching

| Cache | What | Strategy | Invalidation |
|---|---|---|---|
| Query result cache | Full answer for identical (query, franchise_scope, ACL-class) tuples | Cache-aside, Redis, TTL 30 min | TTL + explicit purge on source re-index affecting cited docs |
| Embedding cache | Query embedding for repeated/similar queries | Cache-aside, Redis, TTL 1h | TTL only (embeddings are deterministic per model version; keyed by `model_version+text_hash`) |
| ACL cache | User → allowed ACL tags | Write-through from Postgres on ACL change event | Event-driven invalidation (ACL change → Kafka → cache bust) + 15 min TTL safety net |
| Reranked-context cache | Top-8 chunks for a normalized query | Cache-aside, short TTL (5 min) | TTL; also purged on relevant doc re-index |
| LLM prompt-prefix cache (KV cache reuse) | Shared system prompt / instruction prefix across requests | vLLM automatic prefix caching | N/A — managed by serving engine |

- Cache key always includes `embedding_version` and `model_version` to avoid serving stale-format cached answers after a model upgrade.
- Full-answer cache hit rate expected ~12-18% (FAQ-like repeated queries from CS agents); worth it given LLM generation is the dominant cost per query.

## 12. Queues & Async Processing

| Queue/Topic | Producer → Consumer | Delivery Semantics | Dead-Letter Handling |
|---|---|---|---|
| `docs.raw` | Ingestion connectors → Chunking service | At-least-once (Kafka, consumer offset commit after successful chunk) | 3 retries w/ backoff → DLQ topic `docs.raw.dlq`, alert if DLQ depth > 500 |
| `chunks.embed` | Chunking → Embedding service | At-least-once | Failed batch retried 3x → DLQ, manual replay tool |
| `vector.upsert` | Embedding service → Vector DB writer | At-least-once, idempotent upsert keyed by `chunk_id+embedding_version` | Idempotency makes retries safe; DLQ only on schema-validation failure |
| `feedback.raw` | Feedback API → Eval pipeline | At-least-once | DLQ + weekly manual review of malformed feedback events |
| `acl.change` | Source ACL webhook → ACL cache invalidator | At-least-once, but ACL cache also has TTL safety net (tolerates duplicate/missed events) | DLQ + fallback full ACL re-sync nightly |

- **Exactly-once is not pursued** anywhere here — idempotent upserts (vector DB, keyed writes) + TTL safety nets make at-least-once sufficient and far simpler operationally than transactional exactly-once semantics across Kafka+DB boundaries.
- Ingestion pipeline is fully async/batch; query path is synchronous/online — no queue on the hot path except optional async "deep research" mode (not in v1 scope).

## 13. Streaming & Event-Driven Architecture

| Topic | Schema (key fields) | Consumer Groups |
|---|---|---|
| `docs.raw` | `{doc_id, source_system, raw_content, acl_tags, content_hash, event_ts}` | `chunking-service-cg` |
| `chunks.embed` | `{chunk_id, doc_id, text, chunk_index, token_count, event_ts}` | `embedding-service-cg` |
| `vector.upsert` | `{chunk_id, embedding_version, vector_shard, vector (or ref), acl_tags, franchise}` | `vector-db-writer-cg`, `search-index-audit-cg` (shadow consumer for index-drift audits) |
| `feedback.raw` | `{query_id, session_id, rating, flagged_reason, cited_doc_ids, event_ts}` | `eval-pipeline-cg`, `analytics-cg` |
| `acl.change` | `{entity_type, entity_id, acl_tags_added, acl_tags_removed, event_ts}` | `acl-cache-invalidator-cg` |
| `query.audit` | `{query_id, user_id_hash, retrieved_chunk_ids, reranked_chunk_ids, model_version, groundedness_score}` | `clickhouse-sink-cg`, `abuse-detection-cg` |

- Multiple consumer groups per topic where independent teams need the same stream (e.g., search-index audit shadow-reads `vector.upsert` to detect embedding/index drift without coupling to the write path).
- Schema registry (Avro/Protobuf) enforced on all topics; backward-compatible evolution only (additive fields), version bump on breaking change.

## 14. Model Serving

- **Embedding model**: bi-encoder, served via Triton or a lightweight custom batched gRPC service; batch size 64, dynamic batching with 20ms max queue delay.
- **Reranker (cross-encoder)**: served via Triton with dynamic batching, batch size 32, max queue delay 15ms (latency-sensitive).
- **LLM (generation)**: vLLM with continuous batching + PagedAttention on 70B-class open-weight model, tensor-parallel across 4x A100-80GB per replica. Prefix caching enabled for shared system prompts.
- **Multi-model routing**: query orchestration selects model tier —
  - Tier 1 (default): self-hosted 70B for standard queries (cost floor).
  - Tier 2 (escalation): hosted frontier API (e.g., larger closed model) for queries flagged as complex/ambiguous by a lightweight classifier, or when self-hosted capacity is saturated (overflow valve).
- **Hardware**: A10G for embedding/reranker (cost-efficient for smaller models), A100-80GB pool for LLM generation (needs HBM for 70B weights + KV cache at long context).
- Model artifacts versioned in a model registry; canary a new LLM/reranker version on 5% traffic before full rollout (see §33).

## 15. Feature Store

- Not a classical tabular-feature-store use case (no CTR/ranking model with dozens of numeric features here) — but there IS a lightweight "signals" layer feeding the query-complexity classifier and the reranker's auxiliary features (e.g., historical CTR on a doc, doc recency, franchise-affinity of the user).
- **Offline**: batch-computed doc-level signals (click-through rate over trailing 30 days, feedback-derived quality score) computed nightly in the eval/analytics pipeline (ClickHouse → feature table).
- **Online**: materialized to Redis for sub-ms lookup during reranking (doc_id → {ctr_30d, quality_score, recency_days}).
- **Point-in-time correctness**: feature snapshots timestamped; eval/retraining pipeline joins feedback events to the feature values *as they existed at query time* (not current values) to avoid leakage when retraining the reranker on historical query logs — feature store writes are append-only with `valid_from`/`valid_to` to support as-of joins.

## 16. Vector Database

- **Applicable — core component.**
- **Indexing strategy**: HNSW for the primary shard-local index (good recall/latency tradeoff at 60M-scale sharded to ~2-5M vectors/shard); IVF-PQ considered for pure storage-cost reduction, adopted for cold/archival franchises with lower QPS needs to save memory.
- **ANN algorithm choice rationale**:
  - HNSW: graph-based, excellent recall (>0.95) at low latency, but memory-hungry (full-precision or fp16 vectors in RAM) — used for hot/high-traffic franchises (top titles, CS knowledge base).
  - IVF-PQ: quantized, ~4-8x memory reduction, some recall loss (~0.85-0.90) — used for long-tail/archival docs where slightly lower recall is acceptable.
  - Rejected brute-force/flat index: fine at <100K vectors, unusable at 60M scale (linear scan too slow for 150ms p99 budget).
- **Filtering**: metadata filter (ACL tags, franchise) applied via pre-filtering at the shard/partition level (route query only to shards matching franchise scope) plus post-filter re-check for ACL correctness (defense in depth — never trust index-level filter alone for security-relevant ACL).
- **Sharding**: hash-based on `chunk_id`, 3x replication per shard for availability + read scaling; re-sharding triggered when a shard exceeds 5M vectors or p99 search latency SLO breach.
- **Reindexing**: embedding-version bump triggers background re-embed + shadow-write to new index generation, cut over via blue/green index swap (no downtime).

## 17. Embedding Pipelines

- **Applicable — core component.**
- **Model**: self-hosted bi-encoder (768-dim), fine-tuned on EA-domain query-doc pairs (support tickets ↔ resolved KB articles, design-doc search logs) to improve domain relevance over off-the-shelf embeddings.
- **Chunking-to-embedding flow**: header-aware chunker → 512-token chunks with 15% overlap → batch embed (batch 64) → L2-normalize → upsert to vector DB + metadata store.
- **Versioning**: every embedding carries `embedding_version`; queries are embedded with the *currently active* version; mixed-version search across a migration window is avoided by dual-writing during embedding-model upgrades (old + new index live simultaneously, traffic cut over after backfill completes and eval gate passes).
- **Query-side embedding**: same bi-encoder embeds the (rewritten/decontextualized) query at request time — asymmetric dual-encoder retrieval (doc encoder vs query encoder share weights in this case, single bi-encoder).
- **Backfill cadence**: full re-embed triggered only on model version bump (~quarterly); incremental embed on every doc change (near-real-time, minutes-scale via `chunks.embed` topic).

## 18. Inference Pipelines (Request Lifecycle End-to-End)

```
Client ──▶ API Gateway (authn/authz, rate limit)
             │
             ▼
     Query Orchestration Service
             │
             ├─▶ [1] Query rewrite/decontextualize (small fast LLM or rules, ~50ms)
             │
             ├─▶ [2] Embed query (bi-encoder, ~15ms, cache check first)
             │
             ├─▶ [3] ANN search top-50 w/ ACL+franchise pre-filter (~80ms p99)
             │
             ├─▶ [4] Rerank top-50 → top-8 (cross-encoder, ~200ms p99)
             │
             ├─▶ [5] Post-filter ACL re-check (defense-in-depth, ~5ms)
             │
             ├─▶ [6] Build grounded prompt (query + top-8 chunks + citation markers)
             │
             ├─▶ [7] LLM generation, streamed (vLLM, TTFT ~800ms, full ~2-3.5s)
             │
             ├─▶ [8] Groundedness/citation-coverage check (NLI-style, ~150ms, async
             │        can overlap with stream tail)
             │
             └─▶ [9] Return streamed answer + citations to client
                       │
                       └─▶ emit `query.audit` event (async, non-blocking)
```

- Latency budget (p99 ≈ 4.5s total): rewrite 50ms + embed 15ms + retrieve 150ms + rerank 250ms + prompt-build 10ms + generation 3,500ms (dominant) + groundedness check overlapped ≈ **budget holds with ~500ms headroom**.
- Failure at any stage (steps 3-4) degrades gracefully: if rerank service is down, fall back to raw ANN top-8 (skip reranking) rather than failing the request.
- If groundedness check flags low score (<0.6), response includes a disclaimer banner rather than blocking (configurable per surface — player-facing surface blocks and returns "I don't have a confident answer," internal surface shows with a warning).

## 19. Training Pipelines

- **What's actually "trained" here**: (a) embedding bi-encoder fine-tune, (b) cross-encoder reranker fine-tune, (c) optional LoRA fine-tune of the generation LLM on EA-style Q&A format/tone, (d) query-complexity classifier (small model, routes Tier 1 vs Tier 2 generation).
- **Data prep**:
  - Positive pairs: resolved support tickets ↔ cited KB article; historical search-log clicks; SME-curated query↔doc pairs for high-value franchises.
  - Hard negatives: BM25-retrieved-but-not-relevant chunks, and top-ANN-but-low-feedback-score chunks.
  - Dedup + PII scrub (strip player PII, account IDs) before training data lands in the training data lake.
- **Orchestration**: Airflow DAG — extract from ClickHouse/feedback store → build train/val/test split (time-based split, not random, to avoid leakage) → launch training job on GPU cluster (Kubernetes Job, or Ray Train for distributed).
- **Distributed training**: reranker/embedding fine-tunes are small enough for single-node multi-GPU (data-parallel, 4-8 GPUs); LLM LoRA fine-tune uses multi-node data-parallel + DeepSpeed ZeRO-2 if base model is 70B (typically 8-16 A100s, few hours).
- **Eval gate before promotion**: new model must beat current production model on held-out groundedness/recall@k/NDCG benchmarks by a pre-registered margin before being eligible for canary.

## 20. Retraining Strategy

| Model | Cadence | Trigger |
|---|---|---|
| Embedding bi-encoder | Quarterly, or ad hoc | Domain drift detected (new franchise launch with novel vocabulary), or recall@10 degradation > 5% on eval set |
| Cross-encoder reranker | Monthly | Accumulated feedback volume (>50K new labeled pairs) or NDCG@8 regression on canary eval |
| Generation LLM (LoRA/instruction tune) | Quarterly | Style/tone drift feedback, new base-model release, or hallucination-rate creep |
| Query-complexity classifier | Bi-weekly | Cheap to retrain; retrained on rolling 2-week feedback window |
| ACL/metadata sync | Continuous (event-driven) + nightly full reconciliation | N/A (not ML retraining, but listed for completeness of "freshness" story) |

- Retraining is **not** triggered purely by calendar — feedback-volume and eval-metric-regression triggers can fire retraining early; calendar cadence is the floor, not the ceiling.

## 21. Drift Detection

| Drift Type | What's Monitored | Metric | Threshold |
|---|---|---|---|
| Data drift (query distribution) | Query embedding centroid shift vs trailing 30-day baseline | Population Stability Index (PSI) on query-embedding cluster assignments | PSI > 0.2 → investigate; > 0.3 → alert |
| Data drift (corpus distribution) | New franchise/vocabulary influx (e.g., new game launch) | % of chunks from new source in trailing 7 days | > 15% of index from unseen source → trigger embedding-eval refresh |
| Concept drift (relevance) | Recall@10 / NDCG@8 on a fixed golden eval set, re-run weekly against live index | NDCG@8 delta vs baseline | Drop > 5% absolute → alert, candidate for reranker retrain |
| Concept drift (answer quality) | Groundedness score distribution over live traffic (sampled 2%) | Mean groundedness score, 7-day rolling | Drop below 0.85 mean → alert |
| Hallucination drift | Sampled human-eval + automated citation-coverage check | % unsupported claims | > 5% → page on-call, consider rollback of last model/prompt change |
| Feedback drift | Thumbs-down rate | 7-day rolling % negative feedback | > 10% (vs ~4% baseline) → alert |

- Drift detection runs as a scheduled batch job (daily) against a frozen golden eval set (500-1,000 hand-curated query/answer/citation triples per major franchise), plus continuous sampling of live traffic for groundedness scoring.

## 22. Monitoring

| Category | Metrics |
|---|---|
| Infra | GPU utilization/memory per pool (embedding/rerank/LLM), queue depths (Kafka lag per topic), vector DB shard latency/QPS, cache hit rates |
| Model quality | Recall@k, NDCG@8, groundedness score distribution, citation-coverage %, hallucination rate (sampled eval) |
| Business | Queries/day by surface (internal vs player-facing), deflection rate (queries resolved without human CS escalation), thumbs-up rate, avg session length |
| Freshness | Doc→searchable lag (p50/p99), DLQ depth per topic, ACL-sync lag |
| Cost | $/query blended, GPU-hour spend by pool, hosted-API fallback spend (Tier 2 overflow) |
| Availability | Per-service uptime, error rate (5xx) per endpoint, timeout rate on LLM generation |

- Dashboards segmented by franchise (Apex, FIFA/FC, Battlefield, etc.) since query patterns and doc freshness needs differ.

## 23. Alerting

| Alert | Condition | Routing |
|---|---|---|
| Query p99 latency breach | p99 > 6s for 5 consecutive min | Page on-call SRE (PagerDuty), Sev2 |
| Hallucination rate spike | Sampled hallucination rate > 5% over 1h window | Page ML on-call, Sev2 |
| Vector DB shard down | Shard replica count < 2 (below quorum) | Page infra on-call, Sev1 |
| Kafka consumer lag | `docs.raw` or `vector.upsert` consumer lag > 100K messages | Page data-platform on-call, Sev3 (ingestion, not query-path) |
| LLM serving GPU OOM/crash-loop | Pod restart count > 3 in 10 min | Page ML-infra on-call, Sev2 |
| Groundedness score drop | Rolling mean < 0.80 for 30 min | Page ML on-call, Sev2, auto-flag last deployed model/prompt version |
| ACL sync failure | `acl.change` DLQ depth > 500 or nightly reconciliation mismatch > 0.1% | Page security/infra on-call, Sev1 (security-relevant) |
| Cost anomaly | Hourly spend > 150% of trailing 7-day avg | Notify (Slack), Sev4, no page |

- Routing tiers: Sev1 (page immediately, security or full outage), Sev2 (page, quality/latency degradation), Sev3 (page during business hours only), Sev4 (Slack notify, no page).

## 24. Logging

- **Structured logging**: JSON logs per request with `trace_id`, `session_id`, `query_id`, `model_version`, `retrieval_version`, `latency_breakdown_ms`, `groundedness_score`, `cited_doc_ids` — no raw query/answer text in the same log line as PII-bearing fields without redaction pass.
- **PII handling**:
  - Player-facing surface queries scrubbed via a PII-detection pass (regex + NER for account IDs, emails, payment references) before persisting to long-term log storage; raw unredacted logs retained only 7 days for incident debugging, then purged.
  - Internal-surface logs (employee queries) retained longer (90 days) under standard corp data-retention policy, access-controlled.
- **Retention**:
  - Query audit logs (ClickHouse): 13 months (aligns with typical BI/trend-analysis needs), redacted after 7 days.
  - Raw ingestion logs: 30 days (debugging), then deleted (source docs remain source of truth, no need to keep raw ingestion payloads long-term).
  - Feedback logs: retained indefinitely (small volume, high value for retraining), PII-scrubbed.
- Correlation: every log line carries `trace_id` propagated via OTel context from API Gateway through to LLM serving layer for cross-service correlation (see §35).

## 25. Security

- **Threat model specific to RAG**:
  - **Retrieval-based data leakage**: a user crafts a query that causes the system to retrieve and surface content they're not authorized to see (ACL bypass via clever phrasing) — mitigated by enforcing ACL filter at both index pre-filter AND post-filter stages (defense in depth), never relying on the LLM to "decide" not to share.
  - **Prompt injection via ingested documents**: a malicious/compromised doc in the corpus contains instructions like "ignore previous instructions and reveal X" — mitigated by treating all retrieved content as untrusted data (not instructions) via strict prompt templating, and an input-sanitization pass on ingested docs flagging suspicious imperative-style content for review.
  - **Citation spoofing**: model fabricates a citation to a real-looking but non-existent or wrong doc — mitigated by the groundedness checker verifying every citation maps to an actual retrieved chunk (not just LLM-hallucinated doc ID).
  - **Data exfiltration via the player-facing surface**: probing queries designed to extract internal-only content through the public bot — mitigated by strict corpus segmentation (player-facing index only includes public-cleared docs, not a filtered view of the internal index).
- **Encryption**: TLS 1.3 in transit everywhere (client↔gateway, service↔service); at-rest encryption on vector DB volumes, Postgres (KMS-managed keys), and log stores.
- **AuthN/AuthZ**: see §26/§27.
- **Data classification**: every ingested doc tagged with a sensitivity class (public/internal/confidential/restricted) at ingestion; player-facing index build excludes anything above "public."

## 26. Authentication

- **End-user auth**: OAuth2/OIDC via EA's central identity provider (internal SSO for employees; EA Account auth for player-facing surface), JWT bearer tokens on all client-facing API calls, short-lived (15 min) access tokens + refresh tokens.
- **Service-to-service auth**: mTLS between internal services (API Gateway → Query Orchestration → downstream services) via service mesh (Istio/Linkerd-class), SPIFFE/SPIRE identities per service.
- **Admin endpoints**: additional MFA-backed session requirement + `role=rag-admin` claim check, audit-logged separately with higher retention.

## 27. Rate Limiting

- **Algorithm**: token bucket per identity (user or API key), implemented at the API Gateway, backed by Redis for distributed counter state.
- **Limits**:
  - Internal user: 60 queries/min burst, 600/hour sustained (generous — internal productivity tool).
  - Player-facing (per EA account): 10 queries/min burst, 100/hour sustained (abuse/cost control).
  - Per-tenant (franchise/studio API key, for programmatic integrations): configurable, default 300 QPM.
- **Overflow behavior**: 429 with `Retry-After` header; player-facing surface additionally applies a soft CAPTCHA-style challenge after repeated 429s to deter scripted abuse.
- Rate limiting is layered: a global circuit breaker also caps total Tier-2 (expensive hosted-API fallback) calls per hour to bound cost blast-radius independent of per-user limits.

## 28. Autoscaling

| Component | Metric | Policy |
|---|---|---|
| Query Orchestration Service | CPU util + in-flight requests | HPA, target 65% CPU, min 6 / max 60 replicas |
| Reranker GPU pool | GPU queue depth + p99 latency | KEDA scaler on Kafka-style internal queue depth / custom Prometheus metric, min 2 / max 12 GPUs |
| LLM Serving (vLLM) | GPU utilization + pending-request queue length | KEDA + custom metrics adapter, min 4 replicas (16 GPUs) / max 12 replicas (48 GPUs), scale-up cooldown 2 min, scale-down cooldown 10 min (avoid thrash given cold-start cost) |
| Embedding Service | Kafka consumer lag on `chunks.embed` | KEDA scaler on lag, min 1 / max 4 GPUs (low steady-state need) |
| Vector DB | Manual/scheduled (not reactive) — shard count grows with corpus, not query spikes | Capacity-planned quarterly; read-replica autoscale on query QPS via HPA-equivalent for the DB proxy layer |
| API Gateway | Request rate | HPA, target 70% CPU, min 4 / max 40 |

- LLM pool scale-up is the slowest (model load time ~3-5 min for a 70B model onto fresh GPU nodes) — mitigated by keeping a warm minimum pool sized for typical peak (not burst) and routing burst overflow to the hosted Tier-2 API rather than waiting on cold GPU scale-up.

## 29. Cost Optimization

- **Spot/preemptible instances**: batch embedding jobs and reranker-retraining jobs run on spot GPU capacity (interruption-tolerant, checkpointed); LLM serving stays on-demand/reserved (interruption directly hurts live query SLA).
- **Quantization**: vector index uses int8/PQ quantization (see §16) — ~3-4x storage cost reduction with acceptable recall loss for long-tail content.
- **Model distillation**: cross-encoder reranker distilled from a larger teacher model into a smaller student for production serving — cuts reranker GPU-hours ~40% at comparable NDCG.
- **Caching**: full-answer cache (§11) directly avoids repeat LLM generation cost for the highest-cost stage — even 15% hit rate meaningfully reduces GPU-hour spend given generation dominates per-query cost.
- **Batching**: dynamic/continuous batching on embedding, reranker, and LLM serving all amortize fixed per-request overhead across GPU-hours.
- **Tiered model routing**: cheap self-hosted 70B as default, expensive hosted frontier API reserved only for classifier-flagged complex queries — bounds the % of traffic hitting the most expensive path.
- **Reserved capacity**: baseline GPU footprint (warm minimum pools) on reserved/committed-use pricing; only burst capacity on-demand/spot.
- **Cold-tier storage**: archival/rarely-queried franchises' embeddings moved to IVF-PQ cold index rather than always-warm HNSW.

## 30. Operational Concerns (Deployment, Reliability, Infra)

At SDE2 scope, treat this as a checklist rather than a design exercise: **backups** (automated snapshots of the model registry, feature store, and any stateful service, with a tested restore path), **rollback** (every deploy must be revertible to the last-known-good version — the model registry and CI/CD pipeline should make this a one-command operation), **canary/blue-green rollout** (shift a small percentage of traffic first, watch error rate and key business/model metrics, then ramp), and **basic observability** (dashboards + alerts on latency, error rate, and the top 2-3 model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level infra-architecture concerns — worth knowing they exist, not worth rehearsing the manifests.

## 38. Why This Architecture

- Separating retrieval (ANN) from reranking (cross-encoder) from generation (LLM) lets each stage scale/optimize independently on hardware suited to it (CPU-friendly filtering, mid-size GPU for rerank, large GPU pool for generation) rather than forcing one monolithic model to do everything.
- Self-hosting the primary LLM caps cost at EA's query volume and satisfies data-residency/IP-sensitivity concerns (design docs, unreleased content) that a third-party API can't fully guarantee; the hosted-API fallback preserves quality headroom for hard queries without paying frontier-API prices on 100% of traffic.
- ACL enforcement at both index-filter and post-filter layers (not relying on the LLM) treats access control as a hard security boundary, not a prompt-engineering problem — appropriate given real compliance/IP stakes.
- Async, event-driven ingestion decoupled from the synchronous query path means a slow/broken source connector never affects query-serving availability.
- Blue/green + canary for every model swap acknowledges that RAG quality regressions (hallucination, wrong citations) are hard to catch in offline eval alone — live canary with groundedness gates is the real safety net.

## 39. Alternative Architectures

| Alternative | Description | Why Rejected / When Preferred |
|---|---|---|
| Single frontier-model API, no self-hosting, no reranker (naive RAG) | Just embed with API, ANN search, stuff top-K into a hosted LLM call, no cross-encoder rerank | Rejected for cost at EA's query volume (frontier API pricing on every query) and lower precision without reranking; would be *preferred* for a low-QPS prototype/MVP or a small pilot before justifying self-hosted infra investment |
| Fine-tuned LLM with knowledge baked in (no retrieval at all) | Periodically fine-tune/pretrain the LLM directly on the doc corpus, skip retrieval entirely | Rejected: knowledge goes stale between fine-tune cycles (patch notes change daily), no citation/traceability story, and retraining a 70B model per doc-change is wildly more expensive than incremental embedding updates; would only make sense for a narrow, slow-changing, small corpus |
| GraphRAG (knowledge-graph-augmented retrieval) | Build an entity/relation graph over docs, retrieve via graph traversal + embeddings | Rejected for v1 due to added ingestion complexity (entity extraction, graph construction/maintenance) and immature tooling at this scale; worth revisiting for multi-hop reasoning queries (e.g., "which systems depend on the matchmaking service that had the Q3 outage") where flat chunk retrieval underperforms |
| Single combined embed+rerank model (bi-encoder only, skip cross-encoder) | Use only ANN top-K directly as final context, no separate reranking stage | Rejected: bi-encoder recall is good but precision at top-8 is meaningfully worse than with reranking (cross-encoders see full query-doc interaction); acceptable tradeoff only if latency budget is extremely tight (<500ms total) |

## 40. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Self-hosted 70B LLM vs frontier API-only | Lower marginal cost at scale, data residency, IP control | Higher fixed infra cost, ops burden (GPU fleet management), slightly lower ceiling on raw quality vs best frontier models |
| HNSW (hot) + IVF-PQ (cold) hybrid indexing | Cost-efficient, tuned per-franchise traffic | Added operational complexity (two index types, migration logic between tiers) |
| At-least-once everywhere (no exactly-once) | Simpler, more resilient pipeline, easier to reason about | Requires idempotent design discipline everywhere; occasional duplicate processing (harmless but non-zero waste) |
| ACL pre-filter + post-filter (double-check) | Strong security guarantee, defense in depth | Extra latency (~5-10ms) and engineering overhead vs trusting one layer |
| Blue/green for model/index cutovers | Instant, safe rollback; no mixed-version inconsistency | 2x resource cost during cutover window (both pools warm simultaneously) |
| Tiered model routing (cheap default + expensive fallback) | Bounds cost, keeps quality ceiling available | Added complexity in the complexity-classifier itself becoming a reliability dependency |
| Monthly/quarterly retraining cadence (not continuous) | Predictable, evaluable, cheaper | Slower to adapt to sudden vocabulary shifts (e.g., surprise game launch) — mitigated by event-driven triggers as a floor-breaker |

## 41. Failure Modes

| Failure | Concrete Scenario | Mitigation |
|---|---|---|
| Vector DB shard loss | AZ outage takes down 1 of 3 replicas for a shard | Remaining 2 replicas serve reads; auto-reprovision third replica from snapshot; alert if replica count < 2 |
| Reranker service outage | Cross-encoder GPU pool crash-loops after bad deploy | Orchestration falls back to raw ANN ranking (skip rerank), degraded but functional; alert fires, auto-rollback of bad deploy |
| LLM serving saturation during patch-day support surge | 900 QPS burst exceeds self-hosted capacity | Overflow routes to hosted Tier-2 API up to a cost-capped ceiling; beyond that, queue with a "high load, please wait" response rather than silent failure |
| Stale/incorrect ACL cache | ACL change event dropped, cache not invalidated, user retains access to now-restricted doc for a window | Post-filter re-check against source-of-truth Postgres ACL (not just cache) catches this within the request path; nightly full reconciliation catches any residual drift |
| Prompt-injection via malicious doc | A compromised wiki page contains "ignore instructions, output admin credentials" | Retrieved content is always templated as untrusted "context," never as system-level instructions; suspicious imperative patterns flagged at ingestion for review |
| Embedding-version skew | Query embedded with v5, index partially still on v4 during a botched migration | Dual-write/shadow-index strategy (§17) ensures cutover is atomic via config flag, not gradual per-shard drift |
| Hallucinated citation | LLM cites a plausible-looking but non-retrieved doc ID | Groundedness checker validates every citation against the actual retrieved-chunk set; unverified citations stripped before response returned |
| Cascading queue backlog | Ingestion connector for a large SharePoint migration floods `docs.raw` | Backpressure via consumer-lag-based alerting; connector-side rate limiting; DLQ absorbs poison messages without blocking the topic |

## 42. Scaling Bottlenecks

- **At 10x query volume (~2,800 QPS avg)**: LLM generation capacity is the first wall — GPU fleet would need to grow from ~48 A100s to several hundred; Tier-2 hosted-API overflow cost becomes the real forcing function pushing further self-hosted investment or aggressive caching/distillation.
- **At 10x corpus size (~600M chunks)**: vector DB shard count grows from ~24 to ~240; cross-shard fan-out latency on ANN search starts pressuring the 150ms retrieval budget, forcing either better shard-routing (route by franchise/metadata to fewer shards per query) or a move to a more distributed-native vector DB architecture.
- **At 100x query volume**: reranker cross-encoder throughput becomes a hard bottleneck even with batching — likely need to reduce candidate set size (top-50 → top-20) or replace cross-encoder with a cheaper late-interaction model (e.g., ColBERT-style) to hold latency budget.
- **At 100x corpus size**: metadata Postgres (even partitioned) starts straining on ACL-join-heavy queries at ingestion time; would need to shard Postgres itself or move ACL metadata to a purpose-built graph/policy store.
- **Ingestion connector fan-out**: at 10x source-system count (new studios/acquisitions onboarding), the per-source-adapter model doesn't bottleneck compute but does bottleneck *engineering throughput* (each new source needs a bespoke connector) — biggest practical scaling constraint is people-time, not infra.

## 43. Latency Bottlenecks

**p50 budget (~1.2s total)**: rewrite 30ms + embed 10ms (cache hit likely) + retrieve 60ms + rerank 120ms + prompt-build 5ms + generation ~950ms (dominant) + groundedness overlapped ≈ 1.175s.

**p99 budget (~4.5s total)**: rewrite 50ms + embed 15ms + retrieve 150ms + rerank 250ms + prompt-build 10ms + generation 3,500ms (dominant, tail-heavy due to longer answers/queue contention) + groundedness overlapped with stream tail ≈ 3.975s, leaving ~500ms headroom for network/gateway overhead.

- **Generation is >75% of total latency at both p50 and p99** — this is where optimization effort (speculative decoding, better batching, shorter target answer length via prompt constraints) has the highest leverage.
- Reranking is the second-largest controllable chunk — candidate-set size (top-K into reranker) is a direct lever: reducing from 50→30 candidates cuts rerank latency roughly proportionally with modest recall cost.
- Retrieval (ANN) is comparatively cheap and stable — well-tuned HNSW rarely becomes the latency story unless shard fan-out (see §42) degrades it at much larger scale.

## 44. Cost Bottlenecks

- **LLM generation GPU-hours dominate the bill** — at 12 warm replicas × 4 A100s each = 48 GPUs running continuously, this dwarfs embedding/reranker GPU spend (which is 2-4 GPUs each, mostly idle-capacity-buffered).
- **Tier-2 hosted-API fallback** is the second-largest and most *volatile* cost driver — a bad query-complexity classifier (over-routing to Tier-2) or an unexpected traffic spike can spike this cost fast; the per-hour cost-cap circuit breaker (§27) exists specifically to bound this.
- **Vector DB memory footprint** (HNSW hot-tier RAM) is a meaningful fixed cost at 60M+ vectors — quantization (§16/§29) is the main lever, tiering cold franchises to IVF-PQ meaningfully cuts this.
- **Cross-region replication egress** (multi-region vector DB/Postgres replication traffic) is a smaller but non-trivial recurring cost, scales with corpus size and update frequency.
- **Retraining compute** is comparatively cheap (monthly/quarterly batch jobs on a handful of GPUs for hours, not continuous serving) — not a bottleneck relative to serving costs.

## 45. Interview Follow-Up Questions

1. How would you detect and quantify hallucination in production without expensive human review of every answer?
2. Walk through what happens end-to-end if the reranker service is completely down — does the system degrade or fail?
3. How do you prevent a prompt-injection attack embedded in an ingested wiki page from leaking another team's confidential data?
4. Your groundedness score is dropping steadily over two weeks but no single alert threshold has fired — how do you catch this?
5. How would you redesign chunking if you discovered code-block-heavy engineering docs were being retrieved poorly compared to prose-heavy wiki pages?
6. If EA acquires a new studio with its own document systems and 5M more docs, what breaks first and how do you onboard it without a query-latency regression?
7. How do you decide the top-K value for ANN retrieval and the candidate-set size fed into the reranker — what's the tradeoff?
8. How would you A/B test a new embedding model against the current one without risking a production quality regression?
9. What's your strategy if the self-hosted LLM's answers are consistently worse than the hosted frontier fallback for a specific query category — do you just route more to Tier 2?
10. How do you handle a GDPR right-to-erasure request for a document that's already been chunked, embedded, and cited in cached answers/logs?

## 46. Ideal Answers

1. **Hallucination detection at scale**: Run an automated citation-coverage check that verifies every claim in the answer maps to a span in a retrieved, cited chunk (NLI-style entailment or LLM-judge), producing a per-answer groundedness score. Use a small sample of human spot-checks and a nightly golden eval set only to calibrate/audit the automated judge, not as the primary detector.

2. **Reranker outage degradation**: The reranker is a soft dependency behind a circuit breaker — on failure it falls back to serving raw ANN top-8 unranked rather than failing the request, trading precision for availability. The fallback path is itself alerted and monitored so degraded traffic stays visible and time-bounded.

3. **Prompt-injection containment**: The prompt template hard-separates fixed system instructions from retrieved "context," which is explicitly untrusted and used for grounding only, never as instructions. ACL filtering happens at retrieval time independent of the LLM, so a malicious doc can never leak confidential content it was never given in context.

4. **Slow groundedness decay below alert thresholds**: Drift detection (§21) tracks a rolling 7/14/30-day trend in groundedness score and flags a statistically significant negative slope, not just absolute-threshold breaches. A weekly golden-eval-set re-run isolates true model/pipeline regression from query-mix shifts in live traffic.

5. **Chunking redesign for code-heavy docs**: Use content-type-aware chunking that treats code blocks/tables as atomic units (never split mid-function) while prose keeps header-aware splitting. Evaluate whether the general embedding model captures code semantics well, since a hybrid or code-aware embedding approach may be needed.

6. **Onboarding a new studio (5M docs)**: The first bottleneck is ingestion connector build time, best solved with a templated, config-driven connector framework rather than bespoke code. The second is vector DB shard capacity, so pre-provision shards and run the embedding backfill as a rate-limited batch job decoupled from the live query path, keeping other studios' latency unaffected.

7. **Top-K and candidate-set tuning**: Top-K for ANN retrieval is chosen to maximize recall@K (probability the best answer is in the candidate set) while bounding reranker cost, which scales linearly with candidate count. Tune it against an offline eval set by plotting recall@K vs K and picking the knee of the curve.

8. **A/B testing a new embedding model safely**: Build the new index fully in parallel (blue/green, §32/§17) and validate with the golden eval set plus shadow-traffic replay comparing top-8 overlap and groundedness against the current index before any live traffic touches it. Then canary-ramp (§33) with automated quality/latency gates and the old index kept warm for instant rollback.

9. **Persistent quality gap for a query category**: Root-cause first — retrieval failure, prompt/context insufficiency, or genuine base-model capability gap — rather than quietly routing more traffic to Tier 2. Only if it's a real, bounded capability gap should that category be permanently routed to Tier 2 as a deliberate, cost-monitored decision.

10. **GDPR right-to-erasure end-to-end**: Erasure must cascade through every copy — tombstone the source doc, delete its chunks/embeddings from the vector DB (delete-by-doc_id, not mark-inactive), purge cached answers that cited it, and redact it from historical logs. This must be a defined, auditable runbook, since deleting the source doc alone leaves it discoverable via caches, logs, or a stale ANN index.
