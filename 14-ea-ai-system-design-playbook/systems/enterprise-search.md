# Enterprise Search

## 1. Problem Framing & Requirement Gathering

Design an internal **Enterprise Search** platform for EA: a single search surface over docs (Confluence/Google Docs/Sharepoint-style wikis), source code (monorepos across Frostbite, EA SPORTS, Battlefield, Apex-adjacent tooling), tickets (Jira/ServiceNow), Slack-adjacent chat exports, and design docs. It must combine lexical (BM25-style) and semantic (embedding/ANN) retrieval, apply per-document ACLs at query time, and stay fresh as thousands of engineers/designers/producers edit content daily across dozens of studios and time zones.

Core tension to surface immediately in an interview: **relevance vs. freshness vs. access-control correctness vs. cost**, at a scale of ~40K EA employees, ~5M internal documents/code files/tickets, growing ~15%/year, with sub-second latency expectations because it competes with "just ask a teammate."

## 2. Functional Requirements

- FR1: Full-text (lexical) search across docs, code, tickets, wikis with filters (author, studio, date range, doc type, project).
- FR2: Semantic search — natural-language queries return conceptually relevant results even without keyword overlap ("why does matchmaking retry loop" → matches a doc titled "Session Backoff Design").
- FR3: Hybrid ranking that fuses lexical + semantic + behavioral signals (click-through, recency, authority).
- FR4: ACL-aware retrieval — a result is never shown to a user who lacks read access to the source doc/repo/ticket, enforced at retrieval time, not just at UI render time.
- FR5: Near-real-time indexing — edits/new docs/code commits/ticket updates searchable within a bounded SLA (see NFRs).
- FR6: Code-aware search: symbol search, function/class-level chunking, language-aware tokenization.
- FR7: Snippet generation / highlighting of matched terms and semantically relevant passages.
- FR8: Personalization: boost results from the user's own team/project/studio.
- FR9: Query autocomplete and "did you mean" spell correction.
- FR10: Admin tooling: re-index triggers, ACL sync audit, relevance tuning dashboard, content-source onboarding (connectors).
- FR11: Answer synthesis (RAG-lite): optional LLM-generated summary citing top-k retrieved passages, gated per-source ACL.

## 3. Non-Functional Requirements

| Dimension | Target |
|---|---|
| Latency | p50 ≤ 150 ms, p99 ≤ 600 ms for hybrid search (excluding optional RAG summary path) |
| RAG summary latency | p50 ≤ 2.5 s, p99 ≤ 6 s (separate SLA, async-capable) |
| Availability | 99.9% for search API (43 min/month downtime budget); indexing pipeline 99.5% |
| Throughput | Sustain 1,200 QPS peak (studio-hours overlap), burst to 2,500 QPS during org-wide events (e.g., all-hands doc drops) |
| Freshness / indexing SLA | P95 doc → searchable within 5 min; code commits within 10 min (post CI); ACL revocation propagated within 2 min (security-critical, tighter than content freshness) |
| Consistency | Eventual consistency for content; **read-after-revoke must be near-strong** — an ACL revocation must never lag content visibility by more than 2 min |
| Cost | Target ≤ $0.0006 per query fully loaded (compute + storage + model serving amortized) at steady state |
| Durability | Zero data loss on source-of-truth systems (search index is a derived, rebuildable artifact — not source of truth) |

## 4. Clarifying Questions an Interviewer Would Expect

1. Is the ACL model role-based (RBAC), attribute-based (ABAC), or per-object ACL lists inherited from source systems (Confluence spaces, GitHub repo perms, Jira project roles)?
2. Do we need cross-studio search, or can studios opt out (e.g., unreleased-title code must never leak across studio boundaries even to other EA employees)?
3. Is there a hard compliance requirement (SOX, contractual NDAs with partners) restricting where indexed embeddings/text can be stored (region residency)?
4. What's the tolerance for stale ACLs — is "eventually consistent within 2 min" acceptable, or is this a hard security boundary requiring synchronous ACL check at read time regardless of index staleness?
5. Should code search index private/unreleased-title branches, or only mainline/trunk?
6. Is LLM-based answer synthesis in scope for v1, or is this pure ranked retrieval first?
7. What's acceptable model-serving cost ceiling, and is there an existing internal GPU allocation (shared with other ML platform teams) or dedicated budget?
8. Do we need multi-lingual support (EA has studios in Japan, Sweden, France)?
9. Is click/behavioral data available for learning-to-rank, or do we cold-start with pure relevance heuristics?
10. What existing identity provider (Okta/Azure AD) issues the ACL/group claims we must trust?

## 5. Assumptions

1. 40,000 EA employees/contractors are potential users; ~18,000 are daily active searchers (DAU) during a work week.
2. Corpus: 5M items total — 2M doc/wiki pages, 2.5M code files (chunked to ~8M searchable code chunks), 500K tickets.
3. Average query: 6 tokens; average result set examined: top 10.
4. Identity/ACL source of truth is Okta + GitHub Enterprise + Jira/Confluence native permission APIs, polled + webhook-notified.
5. Embeddings use a 768-dim bi-encoder (domain-adapted from an open base model, fine-tuned on EA internal query-doc click pairs).
6. RAG answer synthesis uses an internally hosted LLM (7B–13B class) for cost control; no external API calls with proprietary code/docs.
7. Studios can mark content "studio-private" — such content is indexed but filtered by a hard ABAC boundary, not just soft ranking demotion.
8. Existing CI/CD emits webhook events on commit/merge; existing wiki systems emit edit webhooks or we poll on 2-min intervals if webhooks unavailable.
9. Infra runs on a mix of EA's internal Kubernetes platform (on-prem + cloud burst, primarily AWS-based per existing EA cloud footprint).

## 6. Capacity Estimation

**Query volume**
- 18,000 DAU × 8 searches/day avg = 144,000 searches/day.
- Peak concentration: 20% of daily volume in the peak hour → 28,800/hour ≈ 8 QPS avg peak... but that's too low given NFR of 1,200 QPS — the NFR anticipates growth + org-wide + IDE-embedded auto-search-as-you-type (each keystroke pause can fire a query). Revised: with search-as-you-type (autocomplete firing every ~300ms pause), effective query volume is ~12x raw "searches" → 144,000 × 12 ≈ 1.73M query-events/day. Peak hour 20% → 346,000/hour ≈ 96 QPS sustained, burst 5x during demo/event traffic → ~480 QPS. Round up with headroom for growth + retries: design for **1,200 QPS steady, 2,500 QPS burst** (matches NFR).

**Storage**
- Lexical inverted index (code + docs + tickets): ~5M docs × avg 4KB extracted text × ~0.3 index overhead factor ≈ 6 GB raw text, inverted index typically 1.5–2x raw text with positions/scoring data ≈ 10–12 GB. Trivial at this scale — but code chunking inflates chunk count: 8M code chunks × 1.5KB avg × index overhead ≈ 18 GB. **Total lexical index: ~30 GB** — fits easily in a sharded Elasticsearch/OpenSearch cluster, replicated 2x = 60 GB.
- Embeddings: (2M doc chunks avg 3 chunks/doc = 6M doc chunks) + 8M code chunks + 500K ticket chunks (avg 1.5 chunks) = ~15.25M vectors. At 768-dim float32 = 3072 bytes/vector → 15.25M × 3072 bytes ≈ **46.8 GB raw vectors**. With HNSW graph overhead (~1.3–1.8x) → **~70–85 GB** in the vector index. Replicate 2x for HA → ~150–170 GB.
- ACL metadata (doc_id → allowed principals): 5M docs × avg 50 bytes principal-list reference (group IDs, not full lists) ≈ 250 MB — trivial, kept in fast KV store, not the bottleneck.

**Model size / GPU-CPU counts**
- Bi-encoder embedding model: 110M–330M param class (e.g., a fine-tuned BERT-base/large-equivalent), fp16 → ~220–660 MB weights. Inference is CPU-viable for query-time single-vector encoding (small, e.g., 6-token query) but batch document embedding at ingest time benefits from GPU.
- Query-time embedding: 1,200 QPS × ~15 ms CPU inference (small batch, short query) per replica → 1 CPU core saturates at ~65 QPS → need ~20 CPU-inference replicas (2 vCPU each) for steady state, ~40 for burst. Alternatively, 2× T4-class GPUs handle this trivially (GPU batch inference ~2ms/query at batch=32) — **choose GPU for query encoding to cut replica count and tail latency**: 2 GPUs handle 2,500 QPS burst comfortably (2ms × batch-32 amortized ≈ 16,000 QPS/GPU theoretical, real-world with batching overhead/network ≈ 3,000–4,000 QPS/GPU) → **2 GPU replicas (A10G/L4 class) with 1 hot standby = 3 total** for query embedding.
- Ingest-time embedding (batch, async): re-embedding backlog of ~5M items initially; steady-state daily deltas ~50K changed items/day. Batch of 50K docs × ~8ms/doc GPU batch-encode ≈ 400 sec compute → trivial, 1 GPU worker handles daily delta in <10 min.
- RAG LLM serving (7B–13B): fp16 13B model ≈ 26 GB VRAM → 1×A100-40GB or 1×H100 per replica. Assume RAG used by ~15% of queries (cost-gated, opt-in "Summarize" button) → 1,200 × 0.15 = 180 QPS-equivalent... actually RAG is a low-QPS, high-latency-tolerant path — model batch/continuous-batching serves ~8–15 concurrent generations per A100 at ~2.5s p50 → need **4–6 A100 replicas** to hold p99 ≤ 6s under peak RAG demand (~30 concurrent requests estimated at peak).
- **Total GPU footprint**: 3 (query embedding) + 1 (ingest embedding, can be spot/preemptible) + 4–6 (RAG LLM) = **8–10 GPUs** at steady state, plus autoscale headroom.

**Bandwidth**: avg response payload (10 results × ~1KB snippet+metadata) = 10 KB/query × 1,200 QPS ≈ 12 MB/s sustained egress — negligible.

## 7. High-Level Architecture

```
                                   ┌─────────────────────────┐
                                   │   Identity Provider     │
                                   │  (Okta / Azure AD)      │
                                   └────────────┬────────────┘
                                                │ group/claims
                                                ▼
 ┌──────────┐   query    ┌──────────────┐  ┌──────────────────┐
 │  Clients  │──────────▶│  API Gateway  │─▶│  AuthN/AuthZ      │
 │ (Web/IDE/ │           │ (rate limit,  │  │  Middleware        │
 │  Slack)   │◀──────────│  routing)     │◀─┤ (JWT + ACL ctx)   │
 └──────────┘  results   └──────┬───────┘  └──────────────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │  Search Orchestrator │  (query understanding,
                     │  Service             │   spell-correct, fan-out)
                     └───┬───────────┬────┘
                         │           │
             ┌───────────┘           └───────────┐
             ▼                                    ▼
   ┌───────────────────┐               ┌────────────────────┐
   │ Lexical Retrieval  │               │ Semantic Retrieval  │
   │ (OpenSearch/BM25)  │               │ (Vector DB / ANN)   │
   └─────────┬──────────┘               └─────────┬──────────┘
             │        candidate sets (top-K each)  │
             └───────────────┬─────────────────────┘
                              ▼
                   ┌────────────────────┐
                   │  ACL Filter Stage   │◀── ACL Index (KV, doc_id→principals)
                   │ (post-retrieval,    │
                   │  pre-ranking gate)  │
                   └─────────┬──────────┘
                              ▼
                   ┌────────────────────┐
                   │  Ranking / Fusion   │◀── Feature Store (CTR, recency,
                   │  Service (LTR model)│    authority, personalization)
                   └─────────┬──────────┘
                              ▼
                   ┌────────────────────┐        ┌───────────────────┐
                   │ Result Assembly &   │───────▶│ Optional RAG LLM   │
                   │ Snippet Generator   │        │ Answer Synthesis   │
                   └─────────┬──────────┘        └───────────────────┘
                              ▼
                        response to client

   ── Indexing / Freshness Pipeline (async, event-driven) ──
 ┌────────────┐  webhooks/CDC ┌───────────────┐  ┌────────────────────┐
 │ Source Sys  │──────────────▶│ Ingestion Bus  │─▶│ Content Normalizer  │
 │ (Wiki/Repo/ │               │ (Kafka topics) │  │ + Chunker + PII scan│
 │  Jira/Ticket│               └───────────────┘  └─────────┬──────────┘
 └────────────┘                                              │
                        ┌─────────────────────────────────────┤
                        ▼                                     ▼
              ┌──────────────────┐                 ┌────────────────────┐
              │ Embedding Service │                 │ ACL Sync Service    │
              │ (GPU batch infer) │                 │ (polls/webhooks IdP │
              └─────────┬────────┘                 │  + source ACL APIs) │
                        │                            └─────────┬──────────┘
                        ▼                                      ▼
              ┌──────────────────┐                 ┌────────────────────┐
              │ Vector DB Writer  │                 │  ACL KV Store       │
              └──────────────────┘                 └────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Lexical Index     │
              │ Writer (OpenSearch)│
              └──────────────────┘
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| API Gateway | AuthN token validation, rate limiting, routing | HTTPS/REST | Stateless, horizontal by request rate |
| Search Orchestrator | Query understanding, spell-correct, fan-out to lexical+semantic, timeout budget enforcement | gRPC internal | Stateless, horizontal by QPS |
| Lexical Retrieval (OpenSearch) | BM25/inverted-index search, filters, faceting | REST (OpenSearch DSL) | Sharded by doc_id hash; scale shards + replicas |
| Semantic Retrieval (Vector DB) | ANN nearest-neighbor search over embeddings | gRPC/REST | Sharded by collection/partition, replica per AZ |
| ACL Filter Stage | Post-retrieval filter dropping unauthorized docs; also pre-filter hints pushed to retrieval when possible | In-process library call + KV lookup | Scales with retrieval QPS; KV read-heavy |
| ACL Sync Service | Polls/consumes webhooks from IdP + source systems, materializes doc_id→principal-set | Kafka consumer + REST poller | Partitioned by source system |
| Ranking/Fusion Service | Learning-to-rank model combining BM25 score, cosine sim, CTR, recency, authority | gRPC internal | CPU-bound, horizontal by QPS |
| Feature Store | Serves online features (CTR, recency, org-graph proximity) for ranking | Online: Redis/KV; Offline: columnar warehouse | Read replicas |
| Embedding Service | Encodes queries (online) and documents (batch/async) into vectors | gRPC (query path), batch job (ingest path) | GPU replica count |
| RAG LLM Service | Optional answer synthesis over top-k passages with citations | gRPC/REST, streaming | GPU replica count, continuous batching |
| Ingestion Bus (Kafka) | Durable event log for content change events from all source connectors | Kafka topics | Partition count per source |
| Content Normalizer/Chunker | Extracts text, strips binary/markup, PII scan/redaction, chunks by semantic boundary (code: function-level; docs: heading-level) | Kafka consumer | Horizontal by partition |
| Vector DB Writer / Lexical Index Writer | Idempotent upsert into respective indexes | Kafka consumer | Horizontal by partition |
| Admin/Relevance Console | Re-index triggers, connector health, relevance tuning, ACL audit trail | Internal web app | N/A (low traffic) |

## 9. API Design

```
GET /v1/search
  ?q=<string>                     required
  &filters.doc_type=code|wiki|ticket
  &filters.studio=<studio_id>
  &filters.date_from=<ISO8601>
  &page_token=<opaque_cursor>
  &top_k=<int, default 10, max 50>
  &mode=hybrid|lexical|semantic   default hybrid
Headers: Authorization: Bearer <JWT>
Response 200:
{
  "query_id": "q_9f8...",
  "results": [
    {
      "doc_id": "conf-238471",
      "title": "Session Backoff Design",
      "snippet": "...matchmaking <em>retry</em> loop uses exponential...",
      "source": "confluence",
      "url": "https://wiki.ea.com/...",
      "score": 0.873,
      "score_breakdown": {"bm25": 12.4, "cosine": 0.81, "ctr_prior": 0.03, "recency": 0.9},
      "last_modified": "2026-06-30T10:02:00Z",
      "acl_scope": "studio:battlefield"
    }
  ],
  "next_page_token": "opaque...",
  "took_ms": 142
}
Errors: 400 (bad query), 401 (auth), 403 (n/a — ACL filtering is silent, never 403 for individual docs), 429, 503

POST /v1/search/summarize        (RAG path, async-capable)
Body: { "query_id": "q_9f8...", "doc_ids": ["conf-238471", "code-11923"] }
Response 202 (if async): { "job_id": "sum_123", "status": "processing" }
GET  /v1/search/summarize/{job_id}
Response 200: { "status": "done", "answer": "...", "citations": [{"doc_id": "...", "span": [12,45]}] }

GET /v1/autocomplete?prefix=<string>&limit=8
Response 200: { "suggestions": ["session backoff", "session timeout config", ...] }

POST /v1/feedback
Body: { "query_id": "q_9f8...", "doc_id": "conf-238471", "action": "click|dismiss|thumbs_down", "rank": 2 }
Response 204

# Admin/connector APIs (internal, separately authz'd)
POST /v1/admin/connectors/{connector_id}/reindex
GET  /v1/admin/connectors/{connector_id}/health
GET  /v1/admin/acl-audit?doc_id=<id>
```

Versioning: URI-path major version (`/v1/`); backward-incompatible changes → `/v2/`; additive fields are non-breaking and ship without version bump. Deprecation: 6-month sunset window with `Sunset` HTTP header once `/v2/` ships.

## 10. Database Design

| Store | Data | Why |
|---|---|---|
| OpenSearch (Elasticsearch-family) | Lexical inverted index, text fields, filters/facets | Purpose-built for BM25 + faceted filtering; mature aggregation support for admin dashboards |
| Vector DB (e.g., Milvus/Weaviate-class, self-hosted) | Dense embeddings + HNSW index, doc_id, metadata for pre-filter | ANN search; supports metadata pre-filtering to combine with ACL/studio scoping before distance computation |
| ACL KV Store (Redis Cluster or DynamoDB-style) | `doc_id → {principal_set_id, studio, visibility_tier}` | Sub-ms point lookups at high fan-out (checked per candidate doc per query); needs strong-enough consistency for revocation SLA |
| Principal Group Cache | `principal_set_id → [user_ids/group_ids]`, `user_id → [group_ids]` | Avoids re-resolving IdP group membership per query; TTL-based refresh |
| Content Metadata Store (Postgres) | Canonical doc metadata: source, owner, last_modified, connector_state, chunk manifest | Relational integrity needed for connector state machine, joins for admin console |
| Feature Store Offline (columnar, e.g., Parquet on blob + warehouse) | Historical CTR, dwell time, query-doc pairs for LTR training | Columnar scan efficiency for training-set generation |
| Kafka | Change-event log (not a DB, but system-of-record for propagation ordering) | Durable, replayable, decouples connectors from indexers |

**Partitioning/sharding**:
- OpenSearch: shard by `hash(doc_id) % N_shards` (N chosen so each shard stays <50GB); route by `studio_id` as a routing hint to reduce cross-shard fan-out for studio-scoped queries.
- Vector DB: partition by content type (`code` / `doc` / `ticket`) — different embedding models/dims possible per type; ANN search only fans into relevant partitions.
- ACL KV: partition by `hash(doc_id)`; principal-group cache partitioned by `hash(user_id)`.
- Postgres metadata: partition table by `source_system` + monthly range partition on `ingested_at` for easy retention pruning.

## 11. Caching

| Cache | Content | Strategy |
|---|---|---|
| Query result cache | Full response for identical (query, user-ACL-fingerprint) pairs | Cache-aside, TTL 60s (short — freshness/ACL sensitive); keyed by hash(query + acl_fingerprint + filters) |
| ACL principal-group cache | user→groups, doc→principal_set | Cache-aside with active invalidation: ACL Sync Service publishes invalidation event on any group/permission change → cache evicts key within the 2-min revocation SLA; TTL backstop of 5 min as safety net |
| Embedding cache | query text → vector (for repeated/autocomplete-driven queries) | Cache-aside, TTL 24h — query text embeddings are stable per model version; invalidated wholesale on model version bump |
| Autocomplete trie/cache | Popular prefixes → suggestions | Write-through, rebuilt from click logs every 15 min |
| Feature store online cache | CTR priors, recency scores | Write-through from streaming aggregation job (updates every 5 min) |

**Invalidation on ACL revocation is the critical path** — cannot rely on TTL alone given the 2-min hard SLA; ACL Sync Service actively pushes invalidation events (via pub/sub) to all cache-holding nodes rather than waiting for expiry. Query result cache TTL kept intentionally short (60s) specifically because a stale cached result could leak revoked-access content if not otherwise invalidated — defense in depth.

## 12. Queues & Async Processing

| Queue/Topic | Payload | Delivery Semantics | Dead-Letter Handling |
|---|---|---|---|
| `content.changed` (Kafka) | `{source, doc_id, change_type, raw_content_ref}` | At-least-once; consumers (Normalizer) are idempotent via `doc_id + content_hash` dedup key | After 5 retries with exponential backoff → DLQ topic `content.changed.dlq`, alert fired, manual replay via admin console |
| `content.normalized` | Chunked, extracted text + metadata | At-least-once; downstream (Embedding Service, Lexical Writer) idempotent upsert by `chunk_id` | Same DLQ pattern; DLQ consumer retries after connector health recheck |
| `acl.changed` | `{principal_id or doc_id, change_type, effective_ts}` | At-least-once, but **ordering matters** — partitioned by `doc_id`/`principal_id` to preserve per-key order within Kafka partition | DLQ + **paging alert** (security-relevant queue; DLQ backlog here is a security incident, not just an ops ticket) |
| `search.feedback` | Click/dismiss events for LTR training | At-least-once, dedup not critical (aggregated statistically) | DLQ with lower urgency; feeds offline training only |
| `embedding.request` (internal work queue) | Batch of chunk_ids to embed | At-least-once; GPU worker idempotent (upsert vector by chunk_id + content_hash) | Retry 3x, then DLQ + Slack alert to ML platform on-call |

No step in this pipeline requires exactly-once delivery because every write is an idempotent upsert keyed by content hash — cheaper and simpler than transactional exactly-once machinery, and correctness (a doc reprocessed twice just overwrites identically) doesn't require it.

## 13. Streaming & Event-Driven Architecture

**Topics**:
- `wiki.raw-events`, `code.commit-events`, `jira.ticket-events`, `slack.export-events` — per-source raw ingestion topics (isolate blast radius of a connector bug).
- `content.changed` — normalized fan-in topic all connectors write to after light validation.
- `content.normalized` — post-chunking/PII-scan output.
- `acl.changed` — ACL delta stream, **partitioned by doc_id/principal_id for ordering**.
- `search.feedback` — click-stream for ranking model training.

**Event schema** (Avro/Protobuf, schema-registry enforced):
```json
{
  "event_id": "uuid",
  "event_type": "content.updated | content.deleted | acl.granted | acl.revoked",
  "doc_id": "string",
  "source_system": "confluence|github|jira",
  "effective_ts": "ISO8601",
  "payload": { "...": "type-specific" },
  "schema_version": 3
}
```

**Consumer groups**:
- `normalizer-cg` consumes `content.changed` (parallelism = partition count, ~24 partitions sized for peak commit-burst days).
- `embedding-cg` and `lexical-writer-cg` both consume `content.normalized` independently (fan-out, different consumer groups so both get every message).
- `acl-sync-cg` consumes `acl.changed` with strict per-key ordering — single consumer per partition, no reordering tolerance.
- `feedback-agg-cg` consumes `search.feedback`, windowed aggregation (5-min tumbling windows) feeding the online feature store.

Schema evolution: additive-only fields by default; breaking changes bump `schema_version` and run dual-write/dual-read during migration window.

## 14. Model Serving

| Model | Framework | Batching | Hardware |
|---|---|---|---|
| Query embedding bi-encoder | Triton Inference Server (ONNX/TensorRT export) | Dynamic batching, max batch 32, 5ms batch window | 2–4× L4/A10G GPU (query path, latency-sensitive) |
| Document embedding (batch/ingest) | Same model, batch offline job | Large static batches (512+) | 1× A10G, spot/preemptible OK (throughput not latency bound) |
| Learning-to-Rank model (gradient-boosted trees, e.g., LightGBM) | Custom lightweight server or embedded in Ranking Service | N/A — per-request scoring of ~50 candidates, sub-ms | CPU only |
| RAG LLM (7B–13B instruct-tuned, internally hosted) | vLLM or TensorRT-LLM with continuous batching | Continuous/in-flight batching, PagedAttention | 4–6× A100-40GB (or H100 if available for better $/token) |
| Spell-correction/autocomplete model | Small n-gram or distilled transformer | N/A, cached aggressively | CPU |

Multi-model serving: Triton hosts the embedding model and LTR feature-scorer as separate model repositories on shared GPU pool with model-level resource quotas; RAG LLM isolated on its own node pool given VRAM footprint and different autoscaling profile (bursty, higher latency tolerance). Canary new model versions via Triton's model versioning (side-by-side `v3`/`v4` directories, traffic-split at the orchestrator layer).

## 15. Feature Store

**Online store** (Redis/low-latency KV): per-(query-type, doc) features needed at serving time —
- `ctr_7d`, `ctr_30d` (rolling click-through rate for doc given similar queries)
- `recency_score` (decay function of `last_modified`)
- `authority_score` (backlink/reference count within corpus, PageRank-style, recomputed nightly)
- `personalization_boost` (user-team/doc-team proximity in org graph)

**Offline store** (Parquet on blob storage + warehouse table): full historical query-doc-click logs used to train the LTR model and the bi-encoder fine-tuning set.

**Point-in-time correctness**: training examples for the LTR model must use feature values **as they existed at query time**, not current values — e.g., a doc's `ctr_7d` at the time a historical query was served, not today's CTR. Achieved via:
- Timestamped feature snapshots written alongside each `search.feedback` event (the online feature values used are logged into the event itself at serving time — "log what you serve" pattern) rather than reconstructed later by joining against a mutable feature table.
- This avoids the classic point-in-time leakage bug (training on future information) that would make offline eval metrics look better than real online performance.

## 16. Vector Database

**Applicable — used for semantic retrieval.**

- **ANN algorithm**: HNSW (Hierarchical Navigable Small World) — chosen over IVF-PQ because corpus size (15M vectors) is modest enough that HNSW's higher memory footprint is affordable, and HNSW gives better recall@k at low latency without a training/clustering step (IVF requires periodic re-clustering as corpus shifts, adding pipeline complexity we don't want given daily content churn).
- **Indexing strategy**: separate HNSW indexes per content-type partition (code / doc / ticket) since embedding distributions differ; `M=16`, `efConstruction=200` (favor recall given corpus size is small enough to afford it), `efSearch` tuned per query-time latency budget (default 64, reduced under load-shedding).
- **Metadata pre-filtering**: ACL/studio scoping applied as a pre-filter bitmap before HNSW traversal where the vector DB supports filtered search natively (avoids the "filter after top-k" problem where all top-k get filtered out and results are empty) — critical because ACL filtering can remove a large fraction of raw candidates for studio-private content.
- **Re-indexing**: incremental upsert supported natively; full index rebuild (needed after model version bump, since embeddings from different model versions aren't comparable) done via blue/green index swap.

## 17. Embedding Pipelines

**Applicable.**

- **Ingest-time**: Content Normalizer chunks docs (heading-boundary for wikis, function/class-boundary for code via AST parsing, ticket-description+comments as one chunk) → Embedding Service batch-encodes → Vector DB Writer upserts with `chunk_id`, `doc_id`, `source`, `acl_ref` metadata.
- **Query-time**: Search Orchestrator sends raw query text to Embedding Service (low-latency path, GPU, dynamic batching) → returns single query vector → passed to Vector DB for ANN search.
- **Model versioning**: embeddings are tagged with `model_version`; mixing vectors from different model versions in one ANN search is invalid (different vector spaces) — re-embedding the full corpus is required on model upgrade, done via shadow index + blue/green swap (see §32).
- **Chunking strategy specifics**: overlap of ~15% between adjacent chunks (sliding window) to avoid losing context at chunk boundaries; max chunk size tuned to model's context window (e.g., 512 tokens) with truncation fallback logged for oversized single-unit content (e.g., a huge generated code file).

## 18. Inference Pipelines (Request Lifecycle End-to-End)

```
Client                 Gateway        Orchestrator      Lexical    Semantic    ACL Filter   Ranking    Assembly
  │  GET /v1/search?q=…   │                │              │           │            │            │           │
  │──────────────────────▶│                │              │           │            │            │           │
  │                       │ validate JWT,  │              │           │            │            │           │
  │                       │ rate-limit     │              │           │            │            │           │
  │                       │───────────────▶│              │           │            │            │           │
  │                       │                │ spell-correct│           │            │            │           │
  │                       │                │ + embed query│           │            │            │           │
  │                       │                │──────────────┼──────────▶│ (embed svc)│            │           │
  │                       │                │  fan-out (parallel, 300ms timeout budget each)      │           │
  │                       │                │─────────────▶│           │            │            │           │
  │                       │                │──────────────┼──────────▶│            │            │           │
  │                       │                │◀──top-200────│           │            │            │           │
  │                       │                │◀──top-200────┼───────────│            │            │           │
  │                       │                │  merge candidate sets (dedup by doc_id)             │           │
  │                       │                │─────────────────────────────────────▶│              │           │
  │                       │                │            ACL check (KV batch lookup, ~5-10ms)     │           │
  │                       │                │◀─────────────────────────────────────│              │           │
  │                       │                │   filtered candidates (~50-150 survive)              │           │
  │                       │                │───────────────────────────────────────────────────▶│           │
  │                       │                │        LTR scoring (fetch online features, ~10ms)   │           │
  │                       │                │◀───────────────────────────────────────────────────│           │
  │                       │                │  top-10 ranked ──────────────────────────────────────────────▶│
  │                       │                │                                                    snippet gen│
  │                       │                │◀───────────────────────────────────────────────────────────────│
  │                       │◀───────────────│                                                                │
  │◀──────────────────────│  200 OK, results, took_ms                                                        │
```

Timeout budget: total p99 600ms → Gateway/auth ~10ms, embed ~15ms, lexical+semantic parallel fan-out ~150ms (bounded by slower of two, both timeout-capped at 200ms with partial-result fallback), ACL filter ~15ms, ranking ~20ms, assembly/snippet ~30ms, network/serialization overhead ~50ms — remaining budget is buffer for tail/GC pauses. If either retrieval leg times out, degrade gracefully (serve lexical-only or semantic-only results) rather than failing the whole request.

## 19. Training Pipelines

- **Data prep**: click logs (`search.feedback`) + served-feature snapshots → offline warehouse table → labeled training pairs (query, doc, relevance_label) where label derived from click + dwell-time heuristics (click + >20s dwell = positive; click + <3s dwell/bounce = weak negative; impressions without click at rank ≤3 = negative).
- **Bi-encoder fine-tuning**: contrastive learning (in-batch negatives + hard negatives mined from BM25 top-50-but-not-clicked) on top of an open base checkpoint; distributed training via data-parallel across 4–8 GPUs (DDP), training run ~4-6 hours per fine-tune cycle on the accumulated click corpus.
- **LTR model training**: LightGBM/XGBoost, LambdaMART objective (optimizes NDCG directly), trained on CPU cluster, retrained more frequently than the bi-encoder since it's cheap (~20 min job).
- **Orchestration**: Airflow/Kubeflow-style DAG — extract features → build training set with point-in-time joins → train → offline eval (NDCG@10, MRR) against held-out query set → gate promotion on eval thresholds → register model in model registry → canary deploy.
- **Distributed training specifics** (bi-encoder): gradient checkpointing + mixed precision (fp16/bf16) to fit larger batch sizes for effective in-batch negative sampling; gradient accumulation if GPU memory constrains batch size below the ~256 needed for stable contrastive loss.

## 20. Retraining Strategy

| Model | Cadence | Trigger |
|---|---|---|
| LTR ranking model | Weekly scheduled | Also triggered ad-hoc if NDCG@10 on shadow traffic drops >5% relative |
| Bi-encoder embeddings | Quarterly, or on-demand | Triggered by: sustained relevance complaints, a major new content type onboarded (e.g., new connector), or base-model upgrade available |
| Autocomplete model | Daily (lightweight, log-driven) | Rolling window retrain, low cost |
| RAG LLM | Not retrained in-house (use vendor/OSS checkpoint updates) | Triggered by base-model version upgrades; prompt/few-shot templates iterated more often than weights |

Bi-encoder retraining is the most expensive/rare because it forces a full corpus re-embedding (~15M vectors) and blue/green index swap — batched deliberately rather than continuously to amortize that cost.

## 21. Drift Detection

| Drift Type | Signal | Metric | Threshold |
|---|---|---|---|
| Data drift (query distribution) | Query embedding centroid shift vs. training distribution | KL-divergence or centroid cosine distance week-over-week | Alert if cosine distance > 0.15 shift |
| Data drift (corpus distribution) | New content types/sources changing corpus embedding distribution | Same centroid-shift technique on document embeddings | Alert if >20% of new content falls outside existing embedding density clusters (novelty detection) |
| Concept drift (relevance) | Click-through rate on top-1/top-3 results trending down | Rolling 7-day CTR@3 | Alert if CTR@3 drops >10% relative week-over-week |
| Concept drift (ranking quality) | NDCG@10 on a fixed golden query set (human-labeled) evaluated nightly against live index | NDCG@10 | Alert if drops below 0.75 (assuming 0.85 baseline) |
| ACL staleness drift | Lag between ACL source-of-truth change and propagation to ACL KV store | p99 propagation latency | Page if p99 > 2 min (matches hard SLA) |
| Embedding staleness | % of corpus embedded with outdated model_version after a version bump | % stale | Alert if re-embedding backlog >5% of corpus after 48h from cutover start |

Zero-result-rate and "reformulation rate" (user immediately re-searches with different terms) tracked as leading indicators of relevance degradation before CTR fully reflects it.

## 22. Monitoring

- **Infra**: QPS, p50/p95/p99 latency per stage (gateway, orchestrator, lexical, semantic, ACL filter, ranking), error rates, GPU utilization/memory, Kafka consumer lag per topic/partition, OpenSearch/vector-DB cluster health (shard status, JVM heap/GC for OpenSearch).
- **Model quality**: NDCG@10, MRR, CTR@1/@3/@10, zero-result-rate, reformulation rate, RAG answer faithfulness/citation-accuracy (sampled human eval + automated citation-overlap check).
- **Pipeline health**: connector last-successful-sync timestamp per source, indexing lag (doc committed → searchable), ACL propagation lag, DLQ depth per topic.
- **Business metrics**: searches-per-DAU, "time-to-first-relevant-click," support-ticket volume tagged "couldn't find docs" (proxy for search failure), adoption rate of RAG summarize feature.
- **Security**: ACL audit log completeness, count of "would-have-leaked" incidents caught by filter-stage tests (canary ACL test docs).

## 23. Alerting

| Condition | Threshold | Severity / Routing |
|---|---|---|
| ACL propagation p99 > 2 min | Sustained 5 min | Page security on-call immediately — potential access-control breach window |
| Search API error rate > 1% | Sustained 3 min | Page search platform on-call |
| p99 latency > 600ms | Sustained 5 min | Page on-call, auto-trigger load-shed (reduce efSearch, disable RAG path) |
| Kafka consumer lag (any critical topic) > 100K messages | Sustained 10 min | Ticket + Slack alert to data platform on-call; page if `acl.changed` lag specifically |
| DLQ depth on `acl.changed` > 0 | Immediate | Page — treated as security incident, not ops backlog |
| Connector sync failure (any source) > 30 min stale | Immediate | Slack alert to platform team, escalate to page at 2h |
| GPU node pool utilization > 90% sustained | 10 min | Autoscale trigger + Slack notify (capacity planning signal) |
| NDCG@10 golden-set drop below 0.75 | Nightly eval job | Ticket to ML team, not paged (not user-facing emergency) |
| RAG LLM error rate > 5% or GPU OOM events | Sustained 5 min | Page ML platform on-call |

## 24. Logging

- **Structured logging**: JSON logs with `trace_id`, `query_id`, `user_id_hash` (never raw user_id in logs — hashed/pseudonymized), `stage_latency_ms`, `result_count`, `doc_ids_returned` (for audit/debugging).
- **PII handling**: query text may contain PII (rare, but e.g., a user pastes an email or employee ID) — query text logged only in a restricted-access log stream with 30-day retention and automated PII scrubbing (regex + lightweight NER) before it reaches general-access analytics logs; full unredacted logs restricted to security/incident-response role.
- **Content PII**: source documents may contain PII (HR docs, PII in tickets) — Content Normalizer runs a PII-scan/classification pass at ingest; PII-flagged content is either excluded from indexing or indexed with an additional ACL tier restricting to HR/legal roles regardless of source-system ACL (defense in depth).
- **Retention**: query logs 90 days (business metrics/training), security/ACL audit logs 1 year (compliance), general infra logs 30 days, DLQ/error logs 14 days.
- **Correlation**: every log line carries `trace_id` propagated via OpenTelemetry context from Gateway through all downstream services, enabling full-request reconstruction across the fan-out architecture.

## 25. Security

- **AuthN**: end-user JWT issued by EA's IdP (Okta), validated at Gateway; service-to-service via mTLS + short-lived SPIFFE/SPIFFE-like identities within the cluster.
- **AuthZ**: ACL enforced at retrieval time (post-candidate-generation filter), **not** merely at UI-render time — critical because relevance/ranking features (CTR, etc.) must never be computed or logged in a way that leaks the existence of documents a user can't see; the ACL filter runs before ranking, not after.
- **Encryption**: at-rest encryption for all indexes (OpenSearch, vector DB, Postgres) via KMS-managed keys; in-transit TLS everywhere including internal service mesh.
- **Threat model specific to this system**:
  - *Cross-studio leakage*: a Battlefield engineer's query surfaces an unreleased Apex-adjacent title's design doc — mitigated by hard ABAC boundary (studio_id tag) checked independently of soft ranking, with periodic canary-doc tests verifying the filter actually blocks.
  - *Embedding inversion attack*: adversary with access to raw vectors attempts to reconstruct sensitive source text from embeddings — mitigate by treating vector DB access itself as ACL-gated infrastructure (not just the search API), encrypting vectors at rest, and not exposing raw vectors via any public API.
  - *Stale ACL cache exploited during a group-removal race*: a just-terminated contractor's session token still valid + ACL cache not yet invalidated — mitigated by short query-cache TTL (60s) plus active push-invalidation, and JWT short expiry (15 min) forcing re-auth that re-checks group membership at the IdP.
  - *RAG prompt injection via malicious doc content*: a document engineered to contain "ignore previous instructions" text could hijack the RAG summarizer — mitigate with instruction-hierarchy prompting, output sanitization, and citation-verification (flag answers whose claims don't overlap cited passage text).
  - *Query log side-channel*: analyzing query logs cross-referenced with click patterns to infer existence of confidential docs — mitigate via access-controlled analytics layer, no raw query-log access without audit trail.

## 26. Authentication

- **End-user**: OAuth2/OIDC via Okta, JWT access tokens (15-min expiry, refresh token rotation), JWT carries group/role claims cached at Gateway for the request's lifetime only (not persisted beyond request scope).
- **Service-to-service**: mTLS via service mesh (e.g., Istio/Linkerd-style sidecar) issuing short-lived certs from an internal CA; each service has a scoped identity (e.g., `embedding-service` can call `vector-db` but not `postgres-metadata` directly).
- **Connector auth to source systems**: OAuth app credentials or service-account tokens per connector (Confluence API token, GitHub App installation token, Jira service account), stored in a secrets manager (Vault-style), rotated on a scheduled cadence (90 days) and on-demand upon suspected compromise.
- **Admin console**: separate stricter authz — requires elevated role claim, all admin actions audit-logged with actor identity.

## 27. Rate Limiting

- **Algorithm**: token bucket per user (allows bursty search-as-you-type behavior while capping sustained abuse), implemented at Gateway using a distributed counter (Redis) with local approximate limiting as fallback under Redis unavailability.
- **Limits**:
  - Per-user: 60 requests/min sustained, burst capacity 20 (accommodates fast typing autocomplete), refill rate ~1/sec.
  - Per-service-account (bots/integrations, e.g., a Slack-search-bot integration): configurable, default 300 req/min, negotiated per integration.
  - Per-tenant/studio aggregate ceiling: soft cap to prevent one studio's automated tooling from starving others — 20% of total cluster capacity max per studio by default, adjustable.
  - RAG summarize endpoint: separate, tighter limit (5 req/min/user) given GPU cost — enforced with a distinct bucket.
- **Response**: `429` with `Retry-After` header; autocomplete degrades gracefully (client-side debounce reduces load before hitting server limits).

## 28. Autoscaling

- **Search Orchestrator / Ranking / ACL Filter (stateless CPU services)**: HPA on CPU utilization (target 60%) + custom metric (in-flight request count) to react faster than CPU-based scaling alone during sudden query bursts.
- **OpenSearch / Vector DB**: scale via replica count (KEDA/HPA on query latency + queue depth metric), shard count is a more manual/planned scaling lever (resharding is heavier-weight, done via capacity-planning cycles, not live autoscale).
- **Embedding Service (GPU, query path)**: KEDA scaler on custom metric — GPU queue depth / inference latency p95; scale-out trigger at p95 > 20ms, scale-in cooldown 5 min to avoid thrashing given GPU cold-start cost.
- **RAG LLM Service**: KEDA scaler on concurrent-request count (continuous batching saturation metric from vLLM's own queue-depth exporter); minimum 2 replicas always warm (cold-start of a 13B model load is 60–90s, unacceptable for on-demand scale-from-zero on the hot path) — scale-to-zero only for a secondary "batch RAG" low-priority queue.
- **VPA**: applied to Content Normalizer/Chunker workers (variable memory footprint depending on doc size) to right-size pod memory requests over time without manual tuning.

## 29. Cost Optimization

- **Spot/preemptible instances** for: ingest-time batch embedding workers, offline LTR training jobs, DLQ reprocessing workers — all tolerant of interruption/restart.
- **Caching** (see §11) cuts redundant embedding computation and repeated ANN searches for popular queries — estimated 25-30% cache-hit rate on query-result cache given repeat-query patterns (common searches like "how to request GPU quota").
- **Model distillation**: distill the 13B RAG model to a smaller 3-4B model for the majority "simple factual" query class (classified by a lightweight router), reserving the larger model only for complex multi-doc synthesis — cuts average GPU-seconds/summary by an estimated 50-60%.
- **Batching**: dynamic batching on embedding service and continuous batching on RAG LLM directly reduce $/query by maximizing GPU utilization per request.
- **efSearch tuning under load**: reduce ANN search's `efSearch` parameter (recall/latency/cost tradeoff knob) during peak load to cut compute per query at a small, acceptable recall cost — a "cost-aware degradation" lever, not just a latency one.
- **Tiered storage**: cold/rarely-accessed doc chunks (e.g., archived tickets >2 years old) can drop from the hot vector index to a cheaper "search on demand" cold path (re-embed on first query, cache result) rather than keeping all 15M vectors hot forever.
- **Right-sizing GPU class**: L4/A10G for embedding (cheaper, sufficient for small models) vs. reserving A100/H100 exclusively for the RAG LLM where VRAM actually requires it.

## 30. Disaster Recovery

- **RTO**: 1 hour for search API (index is a derived artifact — can be rebuilt, but restoring from snapshot is far faster than full re-crawl); 15 min for ACL KV store (security-critical path, kept multi-AZ with fast failover).
- **RPO**: 5 min for content index (acceptable to lose up to 5 min of the newest edits, they'll re-flow through Kafka replay); near-zero (< 1 min) for ACL data — Kafka `acl.changed` topic retained with long retention (7 days) specifically to allow full ACL-state replay if the ACL KV store is corrupted.
- **Backup strategy**:
  - OpenSearch: nightly snapshot to blob storage + continuous translog shipping.
  - Vector DB: nightly full snapshot; since vectors are derived from source content + model version, worst-case recovery = full re-embed from Postgres metadata + source systems (slower, ~hours, used only if snapshot restore also fails).
  - Postgres metadata: continuous WAL archiving + point-in-time restore capability.
  - Kafka: replicated (3x) across AZs; topics central to recovery (`content.changed`, `acl.changed`) retained long enough to fully replay index state from scratch if needed.
- **Runbook priority order on full regional outage**: ACL KV store and gateway auth path restored first (fail closed — no search results served if ACL correctness can't be guaranteed), then lexical index (cheapest/fastest to restore, provides degraded-but-functional search), then vector DB/semantic (slower to restore), then RAG (lowest priority, non-critical path).

## 31. Multi-Region Deployment

- **Topology**: active-active across 2 primary regions (e.g., us-east, eu-west) matching EA's studio geographic distribution, with a 3rd region as cold-standby/DR only.
- **Data replication**: Kafka topics mirrored cross-region (e.g., MirrorMaker2-style) so each region's indexing pipeline can independently build a full local index — avoids cross-region synchronous writes on the hot indexing path. OpenSearch/vector DB indexes are **independently built per region from the same replicated event stream**, not synchronously replicated at the storage layer — cheaper and avoids cross-region write latency, at the cost of small (~seconds) inter-region freshness skew, which is acceptable given the 5-min freshness SLA is already generous relative to that skew.
- **ACL data is the exception**: `acl.changed` topic replication is prioritized/fast-tracked (dedicated higher-throughput mirroring path) because ACL propagation SLA (2 min) is tighter than content freshness SLA and any regional skew here is a security concern, not just a UX one.
- **Latency routing**: GeoDNS/Anycast routes users to nearest healthy region by studio location; failover to secondary region on regional health-check failure, with session-continuity handled by stateless JWT (no region-pinned session state to migrate).
- **Multi-region topology diagram**:

```
        us-east-1 (primary)                         eu-west-1 (primary)
   ┌───────────────────────────┐               ┌───────────────────────────┐
   │  Gateway + Orchestrator    │               │  Gateway + Orchestrator    │
   │  OpenSearch (local index)  │               │  OpenSearch (local index)  │
   │  Vector DB (local index)   │◀──Kafka───────▶│  Vector DB (local index)   │
   │  ACL KV (fast-mirrored)    │  mirroring     │  ACL KV (fast-mirrored)    │
   └─────────────┬─────────────┘               └─────────────┬─────────────┘
                 │                                             │
                 └───────────────┬─────────────────────────────┘
                                  ▼
                       ┌────────────────────┐
                       │  GeoDNS / Global LB  │
                       └──────────┬──────────┘
                                  │
                                  ▼
                            end users (routed to
                            nearest healthy region)

                    (ap-southeast-1: cold-standby DR,
                     receives async Kafka mirror, index
                     rebuild-on-activation)
```

## 32. Blue/Green Deployment

Applies most concretely to **embedding model version upgrades and vector index cutovers**, since old and new embeddings are incompatible in the same ANN space:
1. Deploy new embedding model (v4) alongside existing (v3) — both live in Embedding Service as separate Triton model versions.
2. Build a full "green" vector index using v4 embeddings, re-embedding the entire corpus (batch job, GPU-hours as estimated in §6).
3. Green index validated offline (recall/NDCG comparison against golden query set) before any live traffic.
4. Cut over: Search Orchestrator's vector-DB client config flips from "blue" (v3 index) to "green" (v4 index) — atomic config flag flip, not a gradual drain, because mixing v3/v4 candidates in one ranked list would produce incomparable similarity scores.
5. Blue index kept warm for a rollback window (e.g., 48h) before decommission.

Also applies to the LTR ranking model and RAG LLM service deployments (standard blue/green pod-set swap behind the Ranking Service / RAG Service internal load balancer), though those don't have the same "incompatible vector space" constraint forcing atomic all-or-nothing cutover.

## 33. Canary Deployment

- **Ranking model canary**: new LTR model version receives 5% of traffic (randomly bucketed by `hash(user_id) % 100 < 5`), shadow-scored against control for 48h. Health-check gates before ramp-up: NDCG@10 on live-labeled subset within 2% of control, p99 scoring latency within 10% of control, no error-rate increase.
- **Embedding model canary**: harder to do a true live traffic-split canary given the atomic-cutover constraint (§32) — instead, canary is done **offline against a frozen golden query set and a shadow green index**, not live production traffic split; only after offline gates pass does the blue/green atomic cutover happen (so "canary" here means "shadow evaluation," not "partial live traffic").
- **RAG LLM canary**: new model version gets 10% of RAG-eligible requests, gated on: citation-accuracy rate (automated overlap check) within 3% of control, generation latency p99 within budget, no increase in flagged/unsafe-output rate (safety classifier on outputs).
- **Ramp schedule**: 5% → 25% → 50% → 100% over ~4 days, with automated halt-and-rollback if any gate metric regresses beyond threshold at any stage.

## 34. Rollback Strategy

- **Automated triggers**: error-rate spike >2% above baseline, p99 latency regression >20%, NDCG@10 drop >5% on shadow eval, security-canary-doc leak test failure (any leak = instant automatic rollback, no human-in-the-loop delay given severity).
- **Mechanics**:
  - Ranking/RAG model rollback: traffic-split config reverts to 100% control version — near-instant (config change, not redeploy) since old version's pods/weights stay warm during canary window.
  - Embedding/vector-index rollback: flip the atomic blue/green flag back to blue index (kept warm per §32) — also near-instant, avoids re-running the expensive re-embed job under time pressure.
  - Connector/pipeline rollback: replay from Kafka offset checkpoint prior to the bad deploy, reprocessing content with the previous-version normalizer/chunker logic (requires previous version's container image retained in registry).
- **Post-rollback**: automatic incident ticket creation with the triggering metric snapshot attached; canary gate thresholds reviewed if rollback was a false-positive (metric noise) to avoid alert fatigue.

## 35. Observability

- **Tracing**: OpenTelemetry spans across the full fan-out (Gateway → Orchestrator → Lexical/Semantic parallel legs → ACL Filter → Ranking → Assembly), `trace_id` propagated end-to-end including into the async RAG path (`query_id` links the sync search trace to the async summarize job's trace).
- **Metrics**: Prometheus-style time series for every stage latency, GPU utilization, Kafka lag, cache hit rates — all taggable by region, model_version, studio (for drill-down without needing full trace replay for aggregate questions).
- **Logs**: structured JSON (§24), correlated via `trace_id`/`query_id` so a single Grafana/Kibana-style dashboard can pivot from "this query was slow" (metric) → "here's the trace showing ACL filter stage took 400ms" (trace) → "here's the log line showing ACL KV store had a cache miss storm" (log) in one investigation flow.
- **Golden signals dashboard** (per component): latency, traffic, errors, saturation — standard four-signal view for each of Gateway/Orchestrator/Lexical/Semantic/Ranking/RAG.
- **Synthetic canary queries**: scheduled synthetic searches (including ACL-boundary test queries using dedicated test accounts with known restricted access) run every 1 min against production to catch silent regressions (including security regressions) faster than organic traffic-based alerting would.

## 36. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: search-orchestrator
  labels: { app: search-orchestrator }
spec:
  replicas: 12
  selector: { matchLabels: { app: search-orchestrator } }
  template:
    metadata: { labels: { app: search-orchestrator } }
    spec:
      containers:
        - name: orchestrator
          image: registry.ea.internal/enterprise-search/orchestrator:1.42.0
          resources:
            requests: { cpu: "1", memory: "1Gi" }
            limits: { cpu: "2", memory: "2Gi" }
          ports: [{ containerPort: 8080 }]
          env:
            - { name: LEXICAL_ENDPOINT, value: "opensearch.search-ns.svc:9200" }
            - { name: VECTOR_ENDPOINT, value: "vector-db.search-ns.svc:19530" }
          readinessProbe:
            httpGet: { path: /healthz, port: 8080 }
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata: { name: search-orchestrator }
spec:
  selector: { app: search-orchestrator }
  ports: [{ port: 80, targetPort: 8080 }]
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: search-orchestrator-hpa }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: search-orchestrator }
  minReplicas: 12
  maxReplicas: 60
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 60 } }
    - type: Pods
      pods:
        metric: { name: inflight_requests_per_pod }
        target: { type: AverageValue, averageValue: "40" }
---
# GPU embedding service, KEDA-scaled
apiVersion: apps/v1
kind: Deployment
metadata: { name: embedding-service }
spec:
  replicas: 2
  template:
    spec:
      nodeSelector: { "gpu-type": "l4" }
      containers:
        - name: triton
          image: registry.ea.internal/enterprise-search/embed-triton:2.1.0
          resources:
            limits: { nvidia.com/gpu: 1 }
```

## 37. Terraform Infrastructure

```hcl
# Vector DB cluster (self-hosted on EKS-style nodes with NVMe-backed instances)
resource "aws_eks_node_group" "vector_db_nodes" {
  cluster_name    = var.eks_cluster_name
  node_group_name = "vector-db-nvme"
  node_role_arn   = aws_iam_role.vector_db_node_role.arn
  subnet_ids      = var.private_subnet_ids
  instance_types  = ["r6gd.2xlarge"]        # NVMe local storage for HNSW index
  scaling_config {
    desired_size = 6
    min_size     = 4
    max_size     = 12
  }
  labels = { workload = "vector-db" }
}

# GPU node pool for embedding + RAG serving
resource "aws_eks_node_group" "gpu_inference_nodes" {
  cluster_name    = var.eks_cluster_name
  node_group_name = "gpu-inference"
  node_role_arn   = aws_iam_role.gpu_node_role.arn
  subnet_ids      = var.private_subnet_ids
  instance_types  = ["g5.2xlarge"]          # L4/A10G-class
  capacity_type   = "SPOT"                   # ingest-time batch workers only; RAG pool overridden below
  scaling_config {
    desired_size = 3
    min_size     = 1
    max_size     = 10
  }
  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }
}

# ACL KV store (ElastiCache Redis, multi-AZ, fast failover for security SLA)
resource "aws_elasticache_replication_group" "acl_kv" {
  replication_group_id       = "acl-kv-store"
  description                 = "ACL principal/doc mapping - security critical, low TTL"
  node_type                   = "cache.r6g.large"
  num_cache_clusters           = 3
  automatic_failover_enabled   = true
  multi_az_enabled             = true
  at_rest_encryption_enabled   = true
  transit_encryption_enabled   = true
  snapshot_retention_limit     = 5
}

# Kafka (MSK) for ingestion/event backbone
resource "aws_msk_cluster" "ingestion_bus" {
  cluster_name           = "enterprise-search-events"
  kafka_version          = "3.6.0"
  number_of_broker_nodes = 6
  broker_node_group_info {
    instance_type   = "kafka.m5.xlarge"
    client_subnets  = var.private_subnet_ids
    storage_info {
      ebs_storage_info { volume_size = 500 }
    }
  }
  encryption_info {
    encryption_in_transit { client_broker = "TLS" }
  }
}
```

## 38. Why This Architecture

- **Hybrid lexical+semantic retrieval** covers both precision (exact symbol/error-code matches only BM25 nails) and recall on conceptual queries (only embeddings nail) — pure-semantic search alone regresses badly on code search where exact-token matching matters (function names, error strings).
- **ACL enforced as a dedicated post-retrieval filter stage**, not baked into the ranking model or left to the UI — makes the security boundary auditable/testable in isolation (canary-doc leak tests target exactly this stage) and keeps ranking model logic decoupled from access-control logic (a ranking bug can't become a security bug).
- **Event-driven ingestion (Kafka-centered)** decouples slow/flaky source-system connectors from the indexing pipeline's reliability, and gives natural replay capability for both disaster recovery and index-rebuild-on-model-upgrade scenarios.
- **Separate GPU pools for query-embedding vs. RAG** reflects genuinely different latency/throughput/cost profiles — conflating them would force either over-provisioning cheap embedding capacity to match RAG's VRAM needs, or starving RAG of dedicated capacity.
- **Blue/green (not rolling) for embedding model upgrades** is forced by the mathematical incompatibility of mixing embedding-space versions in one ANN search — this isn't a stylistic choice, it's structurally required.

## 39. Alternative Architectures

| Alternative | Description | Why Rejected / When Preferred |
|---|---|---|
| Pure semantic search (drop lexical/BM25 entirely) | Single vector-DB-only retrieval path | Rejected: code search and exact-identifier/error-string queries regress badly without lexical matching; would be preferred only if corpus were pure natural-language prose (no code) and query patterns were purely conversational |
| ACL enforcement via per-user filtered index (separate index per ACL group) | Pre-materialize index shards per permission group | Rejected: combinatorial explosion of ACL group overlaps at EA scale (thousands of overlapping group memberships) makes index duplication cost prohibitive; would be preferred only with a small, flat number of coarse-grained tenants (e.g., 3-4 broad tiers) |
| Fully managed SaaS enterprise search (e.g., a hosted search-as-a-service vendor) | Outsource retrieval infra entirely | Rejected for EA specifically: unreleased-title code/design docs are highly sensitive IP — sending raw content/embeddings to a third-party vendor's infra is a non-starter without extremely strong contractual/technical guarantees; would be preferred for a company with no comparable IP-sensitivity constraints wanting faster time-to-market |
| Single global index, no per-region local build (synchronous cross-region replication) | One canonical index, all regions read remotely or via sync-replicated storage | Rejected: adds cross-region write latency to the freshness-critical indexing path and creates a single global failure domain; would be preferred only if data residency/regionality weren't a concern and a single-region deployment was acceptable (e.g., a much smaller company without EA's global studio footprint) |

## 40. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Hybrid lexical+semantic fan-out (query both, always) | Best recall+precision balance | 2x retrieval compute cost per query vs. single-method |
| ACL filter post-retrieval (not pre-filter baked into every index) | Clean separation of concerns, auditable | Requires over-fetching (top-200 not top-10) from retrieval to survive filtering, adds compute |
| HNSW over IVF-PQ | Better recall at this corpus scale, no reclustering step needed | Higher memory footprint; wouldn't scale as cheaply to 10x corpus size |
| Independent per-region index builds (not sync-replicated storage) | No cross-region write latency, simpler failure isolation | Small (~seconds) inter-region freshness skew; duplicated indexing compute cost per region |
| Blue/green atomic cutover for embedding upgrades | Avoids incomparable-vector-space bugs | Expensive (full re-embed), can't do gradual live-traffic canary for the embedding model itself |
| Short (60s) query-result cache TTL | Minimizes ACL-staleness security exposure | Lower cache hit rate / higher compute cost than a longer TTL would allow |
| Separate GPU pools per model type | Right-sized cost/latency per workload | More operational surface area (more node pools, more autoscalers to tune) |

## 41. Failure Modes

- **OpenSearch shard unavailable**: partial result set — orchestrator degrades to semantic-only results for affected shard's doc range, logs a partial-failure flag in response metadata, alerts on-call; user sees results, possibly missing a few, rather than a hard error.
- **Vector DB partition down**: symmetric degrade to lexical-only.
- **ACL KV store unavailable**: **fail closed** — if ACL lookups can't be performed, do not return any results from the affected candidate set rather than risk showing unauthorized content; this is the one component where "degrade gracefully" is the wrong instinct.
- **Embedding Service GPU pool saturated**: query embedding requests queue and hit timeout — orchestrator falls back to lexical-only search rather than blocking the whole request past its latency budget.
- **Kafka partition under-replicated / broker loss**: ingestion continues (replication factor 3 tolerates single broker loss), but if majority of replicas for a partition are lost, that partition's writes pause — connectors buffer/retry, freshness SLA temporarily breached, alert fires.
- **ACL Sync Service falls behind (IdP API rate-limited)**: propagation lag grows — this directly threatens the 2-min revocation SLA; mitigated by a dedicated high-priority polling lane for revocation events specifically (processed ahead of grant events in the same consumer, since a missed revocation is a security issue while a delayed grant is just an inconvenience).
- **RAG LLM hallucinates/fabricates a citation**: mitigated by citation-verification post-check (does cited passage actually contain claimed content); if verification fails, response includes a disclaimer or suppresses the specific unverified claim rather than blocking the whole summary.
- **Bad ranking model deployed (relevance regression)**: caught by canary gates (§33) before full rollout; if it slips through, automated rollback trigger (§34) on CTR/NDCG regression.
- **Malicious/buggy connector floods `content.changed` with garbage events**: rate-limit per-source-system on the ingestion topic; circuit-breaker on Content Normalizer per source to prevent one bad connector from starving processing capacity for all others.

## 42. Scaling Bottlenecks

**At 10x scale** (50M documents, 12,000 QPS steady):
- OpenSearch shard count and per-shard size become a real tuning problem — need to increase shard count and likely move from a few large nodes to many smaller nodes; query fan-out coordination overhead (scatter-gather across more shards) starts adding latency.
- Vector DB: 150M vectors pushes HNSW memory footprint into the multi-TB range across the cluster — likely forced to reconsider IVF-PQ (quantization) for the largest partition (code chunks) to control memory cost, accepting a recall hit.
- ACL KV store read QPS scales with retrieval fan-out (each query checks ACLs for ~200-400 candidates) — at 12,000 QPS that's 2.4-4.8M ACL lookups/sec, likely requiring ACL KV store to shard well beyond the current node count and possibly move some lookups to a bloom-filter pre-check to cut load.

**At 100x scale** (500M documents, 120,000 QPS):
- Single-cluster OpenSearch/vector-DB topology breaks down entirely — needs a federated/multi-cluster retrieval layer with a scatter-gather-of-clusters approach, adding a new aggregation tier and materially more tail-latency risk.
- GPU query-embedding fleet scales linearly with QPS — at this scale it's a genuinely large, cost-dominant GPU fleet, likely forcing a move to smaller/distilled embedding models specifically to control this cost curve, or moving query embedding to CPU with quantized models if latency budget allows.
- Kafka topic partition counts and consumer group parallelism need a redesign (likely topic-per-studio or similar further partitioning) to avoid single-partition hot-spotting on high-churn content sources.

## 43. Latency Bottlenecks (p50/p99 Budget Breakdown)

| Stage | p50 | p99 | Notes |
|---|---|---|---|
| Gateway (authn, rate-limit) | 3 ms | 10 ms | JWT validation is local/cached |
| Query embedding | 8 ms | 25 ms | GPU batch inference; p99 tail from batch-window wait |
| Lexical retrieval | 25 ms | 120 ms | p99 tail from shard scatter-gather straggler |
| Semantic retrieval (parallel with lexical) | 20 ms | 150 ms | p99 tail dominated by HNSW efSearch traversal under load |
| ACL filter | 8 ms | 40 ms | p99 tail from KV cache miss requiring source-of-truth fallback |
| Ranking/LTR scoring | 10 ms | 35 ms | CPU-bound, scales with candidate count post-filter |
| Assembly/snippet generation | 15 ms | 60 ms | p99 tail from long-document snippet extraction |
| Network/serialization overhead | 15 ms | 40 ms | Fixed-ish overhead across hops |
| **Total (hybrid search)** | **~104 ms** | **~480 ms** | Within the 150ms/600ms targets with margin |

**Where p99 actually breaks**: the two retrieval legs (lexical + semantic) are the dominant tail contributors — both are subject to "straggler shard" effects (scatter-gather waits for the slowest shard) and both get worse under load (GC pauses in OpenSearch JVM, HNSW traversal cost increase when efSearch isn't adaptively reduced under load). RAG path (not shown above, separate SLA) is dominated almost entirely by LLM token-generation time, not retrieval — that's a fundamentally different bottleneck (compute-bound autoregressive decoding) addressed by continuous batching and shorter default summary lengths.

## 44. Cost Bottlenecks

- **RAG LLM GPU fleet** is the single largest cost line item despite being used on only ~15% of queries — A100-class GPU-hours for a 7-13B model with continuous batching dominate the bill more than the entire retrieval infrastructure combined, because generation is inherently more compute-intensive per request than retrieval/ranking.
- **Vector DB memory footprint** (HNSW, held largely in RAM for latency) is the second major cost driver — memory-optimized instance types for 70-170GB of index data (replicated) cost meaningfully more than the equivalent lexical index's disk-backed footprint.
- **Query embedding GPU fleet**, while smaller than RAG, is a fixed always-on cost (can't scale to zero given latency-sensitive hot path) — the 24/7 baseline of 2-3 GPU replicas is a steady recurring cost regardless of actual query volume during off-peak hours.
- **Cross-region duplication** (§31, independent per-region index builds) roughly doubles indexing compute and storage cost relative to a single-region deployment — an explicit cost paid for latency/resilience.
- **Lowest-cost components, for contrast**: lexical index storage/compute and ACL KV store are comparatively cheap — CPU-only, small memory footprint relative to embeddings — confirming that model-serving GPU cost, not data infrastructure, dominates this system's bill.

## 45. Interview Follow-Up Questions

1. How would you prevent the ACL filter stage from becoming a timing side-channel that leaks document existence (e.g., differing response times reveal whether a hidden doc exists)?
2. Walk through exactly what happens if a user's group membership is revoked mid-session — trace it through every cache layer.
3. Why not just bake ACL filtering into the vector DB's metadata pre-filter and skip the separate ACL Filter Stage entirely?
4. How do you evaluate whether the hybrid fusion (lexical + semantic score combination) is actually better than either alone, and how do you tune the fusion weights?
5. The RAG summarizer cites a passage that doesn't actually support its claim — how do you catch this before it reaches the user, and how do you catch it in aggregate across the system?
6. How would you re-embed 15M vectors with zero downtime and no query-time regression during the multi-hour migration window?
7. What's your strategy if a single studio's code repo is 10x larger than all others combined and dominates index build time / storage?
8. How do you handle a query that spans content the user has partial access to (e.g., can see the ticket but not the linked design doc) — where's the line between helpful and access-control-violating?
9. What's your plan for evaluating relevance quality before you have any click data (cold start)?
10. How would this design change if freshness SLA tightened from 5 minutes to 5 seconds?

## 46. Ideal Answers

1. **Timing side-channel**: normalize response latency for the ACL filter stage regardless of outcome (constant-time-ish filtering — always perform the same KV batch-lookup pattern whether 0 or 200 candidates get dropped), and never surface differential error messages (e.g., never return "403 doc exists but you can't see it" — a filtered-out doc is simply indistinguishable from a nonexistent one in every observable dimension including latency and error codes).

2. **Session ACL revocation trace**: (a) IdP group change fires webhook → ACL Sync Service consumes `acl.changed` (ordered per principal via partition key) → (b) publishes invalidation event → all Redis-backed principal-group caches evict the affected key within ~seconds → (c) query-result cache's 60s TTL naturally expires any pre-revocation cached responses within 60s worst case → (d) user's existing JWT (up to 15-min lifetime) still claims old group membership if group claims are embedded in the token itself — this is the actual gap; mitigate by NOT embedding group claims in the long-lived JWT at all, instead having the Gateway do a live (cached, short-TTL) group-membership check per request against the IdP-backed cache rather than trusting token claims for authorization-relevant groups.

3. **Why not fold ACL into vector DB pre-filter entirely**: partially do this already (metadata pre-filter, §16) for efficiency, but keep a second independent check as defense-in-depth — pre-filter bitmaps can go stale between ACL change and index metadata update propagation; a second, more-current check against the ACL KV store (updated on a tighter SLA than full vector index metadata) catches that gap. Also keeps the security-critical logic in one auditable, independently-testable service rather than smeared across every retrieval backend's own filtering mechanism, which would need to be re-verified per backend.

4. **Evaluating/tuning fusion**: offline, run interleaving experiments (Team-Draft Interleaving) between hybrid vs. lexical-only vs. semantic-only on live traffic samples to get an unbiased preference signal cheaply; online, A/B test fusion weight variants gated on NDCG@10 and CTR@3; tune fusion weights via the LTR model itself treating `bm25_score` and `cosine_sim` as two of several input features rather than hand-tuning a static linear weight — lets the model learn interaction effects (e.g., trust semantic score more for long natural-language queries, lexical more for short code-symbol queries) rather than a single global weight.

5. **Catching bad RAG citations**: per-response, run an automated overlap/entailment check (does the passage text entail the generated claim — lightweight NLI model or embedding-similarity-of-claim-to-cited-span) before returning the answer; below a confidence threshold, either drop that specific claim or append a "low confidence" flag. In aggregate, sample a percentage of RAG responses for human eval weekly, track a "citation faithfulness rate" metric in the model-quality monitoring dashboard (§22), and treat sustained degradation as a retraining/prompt-engineering trigger.

6. **Zero-downtime re-embedding**: build the full new-model-version ("green") vector index as a side-by-side artifact while the "blue" index continues serving 100% of live traffic; validate green offline against golden query set; atomically flip the Search Orchestrator's vector-DB client pointer from blue to green (a config change, not a data migration on the hot path) so there's no window where queries mix v3/v4 vectors; keep blue warm for a rollback window before decommissioning. The multi-hour re-embed compute happens entirely off the critical path.

7. **Dominant large repo**: shard that repo's code index separately from the rest (dedicated shard/partition sized to it) so its churn/size doesn't distort sharding balance for everything else; consider a tighter freshness SLA exception or relaxed one specifically for that repo if its commit velocity is unusually high; ensure ranking/authority features don't let sheer volume from one repo drown out relevance for smaller repos (normalize authority scores within-source rather than globally).

8. **Partial-access query spanning linked content**: treat each object (ticket, doc) as independently ACL-checked — a user sees the ticket they're authorized for, and the response can note "1 linked reference not shown" without revealing the linked doc's title/content/existence details beyond a generic placeholder; never let RAG synthesis silently pull content from a doc the user can't see into a summary they can see (the RAG service must apply the same ACL filter to its retrieved context set before generation, not just to the final displayed results) — this is a common and dangerous class of bug in naive RAG-over-restricted-corpora implementations.

9. **Cold-start relevance evaluation**: build a golden query set via human labeling (SMEs from a few representative teams label relevance for ~200-500 representative queries against candidate docs) to compute offline NDCG before any click data exists; bootstrap the LTR model with heuristic features only (BM25 score, cosine similarity, recency, title-match) rather than CTR-dependent features; run initial ranking as a simpler weighted-sum rule rather than a learned model until enough click data accumulates (typically a few weeks at EA's query volume) to train a real LTR model without overfitting to sparse/noisy early signal.

10. **5-second freshness SLA**: this changes the architecture materially — batch-ish Kafka-consumer-driven indexing (even at "near-real-time" 5-min granularity) can't hit 5s reliably at tail; would need synchronous or near-synchronous write-path indexing (index update completes as part of the source-system save operation's acknowledged path, or a much tighter streaming pipeline with dedicated low-latency partitions), likely sacrificing some throughput/batching efficiency in the Normalizer/Embedding stages (can no longer batch-embed efficiently, must do small/single-item low-latency embedding calls, raising GPU cost per item significantly) — a concrete illustration of the freshness-vs-cost-efficiency tradeoff discussed in §29.
