---
module: AI Engineering
topic: AI System Architecture
subtopic: Data Architecture for AI
status: unread
source: original (lesson paywalled; intro from the course, body written from the outline)
tags: [data-architecture, knowledge-pipeline, ingestion, cdc, freshness, versioning, acl, pii, data-quality, lineage]
---
# Data Architecture for AI

```mermaid
flowchart LR
    SRC[Sources<br/>Drive · Confluence · DB · tickets · S3] --> CON[Connectors<br/>full sync + CDC/webhooks]
    CON --> RAW[(Raw zone<br/>immutable, versioned)]
    RAW --> PARSE[Parse · clean · dedupe]
    PARSE --> PII[PII detect / redact]
    PII --> CHUNK[Chunk + metadata<br/>ACL · source · version · timestamps]
    CHUNK --> EMB[Embed<br/>model@version]
    EMB --> IDX[(Vector + keyword index)]
    IDX --> RET[Retrieval<br/>ACL filter at query time]
    PARSE -.-> QM[Quality monitors]
    IDX -.-> QM
```

The data layer usually caps how good a production AI system's answers can be. A model can improve phrasing, but it **can't reliably fix stale documents, missing permissions, broken parsing, duplicate chunks, or content that never should have been indexed**. This chapter covers the data architecture behind AI apps: **ingestion, transformation, indexing, freshness, versioning, privacy, and quality monitoring**.

## 1. The Knowledge Pipeline
| Stage | Job | Failure if skipped |
|---|---|---|
| **Connect** | Pull from sources via APIs, exports, CDC | Missing content → "I don't know" or hallucination |
| **Land raw** | Store originals immutably with fetch time and source version | Can't reprocess or audit |
| **Parse** | PDF/HTML/Office → text + structure (headings, tables, page numbers) | Garbled tables, lost headings, OCR noise |
| **Clean / dedupe** | Strip boilerplate, nav, signatures; near-dup detection (MinHash/SimHash) | Duplicate chunks crowd out top-k |
| **Enrich** | Language, doc type, owner, product, summary, entities | Weak filtering and routing |
| **Govern** | PII detection, classification labels, ACL capture | Data leaks across users/tenants |
| **Chunk** | Structure-aware chunks with parent references | Context cut mid-thought |
| **Embed** | Versioned embedding model | Mixed vector spaces |
| **Index** | Vector + BM25 + metadata store | Poor recall on exact terms |
| **Serve** | Retrieval with permission filters | Wrong-tenant answers |

## 2. Ingestion Modes
| Mode | How | Freshness | Cost |
|---|---|---|---|
| Full re-sync | Re-crawl everything on a schedule | Hours–days | High; simple |
| Incremental by `updated_at` | Poll for changes since the watermark | Minutes–hours | Low; misses deletes unless tombstones exist |
| **CDC** (Debezium, DB logs) | Stream row changes | Seconds | Low; best for DB-backed content |
| Webhooks / events | Source pushes change events | Seconds | Low; need retry + periodic reconcile |
| Query-time (federated) | Call the source API at question time | Live | Latency + rate limits; good for volatile data (inventory, ticket status) |

Rule: **deletes and permission changes must propagate as fast as content changes**. A deleted HR doc still answering questions is an incident.

## 3. Chunk Record Schema
```python
class Chunk(BaseModel):
    chunk_id: str                 # f"{doc_id}:{chunk_idx}:{content_hash[:8]}"
    doc_id: str
    tenant_id: str
    source: Literal["confluence", "gdrive", "zendesk", "postgres"]
    source_url: str
    title: str
    section_path: list[str]       # ["Billing", "Refunds", "EU policy"]
    text: str
    content_hash: str             # skip re-embedding unchanged chunks
    acl: list[str]                # groups/users allowed; captured at ingest
    classification: Literal["public", "internal", "confidential", "restricted"]
    pii_redacted: bool
    doc_updated_at: datetime
    ingested_at: datetime
    embedding_model: str          # "text-embedding-3-large@2024-01"
    pipeline_version: str         # parser + chunker version
```
`content_hash` makes reprocessing cheap: re-embed only chunks whose hash changed.

## 4. Freshness
| Technique | Detail |
|---|---|
| Freshness SLO per source | e.g., support macros < 15 min, wiki < 4 h, contracts < 24 h |
| Watermarks + lag metric | `now − max(doc_updated_at indexed)` per source; alert on breach |
| Tombstones | Hard-delete vectors on source deletion; periodic full reconcile catches misses |
| Recency in ranking | Boost newer versions; filter superseded docs (`is_latest = true`) |
| Show dates | Cite "Updated 3 Mar 2026" so users can judge staleness |
| Live lookups for volatile facts | Prices, stock, order status via tools, not the index |

## 5. Versioning and Reproducibility
- **Raw zone is immutable**: every fetch stored with source version; reprocess from raw anytime.
- **Pipeline version** (parser, chunker, enrichment prompts) and **embedding model version** on every chunk.
- **Index versions**: build `kb_v42` alongside `kb_v41`, run retrieval evals, switch an alias (blue/green). Never re-embed in place with a new model, because query and doc vectors must come from the same model.
- **Snapshot for evals**: eval runs pin an index version so results are comparable.
- Tools: lakeFS / Delta / Iceberg for data versioning; DVC for files; alias swaps in OpenSearch / Qdrant / pgvector (table swap).

```python
def reindex(new_model):
    name = f"kb_{new_model.slug}_{date.today():%Y%m%d}"
    create_index(name, dim=new_model.dim)
    for batch in iter_chunks(latest_only=True):
        upsert(name, embed(batch, new_model))
    if retrieval_eval(name).recall_at_10 >= retrieval_eval(alias_target("kb")).recall_at_10 - 0.01:
        swap_alias("kb", name)                 # atomic cutover; keep old for rollback
```

## 6. Access Control
| Approach | How | Trade-off |
|---|---|---|
| **Pre-filter by ACL metadata** | Store allowed groups on chunks; filter `acl ∩ user.groups ≠ ∅` in the vector query | Fast; ACL changes must re-sync |
| Post-filter | Retrieve top-k, then check permissions | Simple; may return too few results; must over-fetch |
| Per-tenant index / namespace | Physical isolation | Strongest isolation; many indexes |
| Query-time permission check against the source | Ask the source API "can user X read doc Y?" | Always correct; slower |

Never rely on the LLM to enforce permissions. Filter **before** content enters the prompt, using the **authenticated user's identity**, not anything the model says.

## 7. Privacy and Governance
- **Data minimization**: index only what the feature needs; exclude restricted folders by default (allowlist sources).
- **PII handling**: detect (Presidio, cloud DLP), then redact, tokenize, or drop; keep reversible tokenization only where needed.
- **Right to erasure**: map user → docs → chunks → vectors → caches → logs, so GDPR/DPDP deletion reaches every copy, including semantic caches and eval datasets.
- **Data residency**: region-pinned stores and model endpoints.
- **Provider terms**: zero-data-retention options; no training on your data; log what goes to which provider.
- **Lineage**: for any answer, trace citation → chunk → doc version → source.

## 8. Data Quality Monitoring
| Check | Signal |
|---|---|
| Parse quality | Empty-text rate, avg chars/page, OCR confidence, table extraction failures |
| Duplication | Near-dup ratio per source |
| Coverage | Docs in source vs. docs indexed (per source, per tenant) |
| Freshness | Lag vs. SLO |
| Embedding health | Null/zero vectors, dim mismatch, norm distribution |
| Retrieval quality | Recall@k on a labeled query set per index version; no-result rate in production |
| Answer-side | "I don't know" rate, thumbs-down tied to cited doc ids → fix the doc |

Treat bad answers as possible **data bugs first**: check what was retrieved before touching the prompt.

## 9. Beyond Documents (additions)
| Data | Store | Access pattern |
|---|---|---|
| Unstructured docs | Vector + keyword index | RAG |
| Structured records | Postgres / warehouse | Text-to-SQL or typed tools |
| Relationships | Graph DB / KG | GraphRAG, multi-hop |
| User memory | KV / vector per user | Agent memory ([memory systems](../10-agents/02-agent-memory-systems.md)) |
| Conversation & traces | Log store / warehouse | Evals, analytics, fine-tuning data |
| Feedback & corrections | Labeled dataset store | Evals, few-shot, DPO |

## Takeaways
- Answer quality is capped by the data pipeline: parsing, dedupe, freshness, permissions.
- Store raw immutably, version the pipeline and embedding model, and cut over indexes by alias.
- Enforce ACLs in retrieval before the prompt; propagate deletes and permission changes fast.
- Monitor parse quality, coverage, freshness, and retrieval recall like any production data system.

## References
- Microsoft, *RAG and generative AI: data pipeline guidance* · Debezium docs (CDC)
- Kleppmann, *DDIA* ch. 11 (stream processing) · Microsoft Presidio · OWASP LLM Top 10 (sensitive information disclosure)

## Related
- [Building a production RAG pipeline](../07-rag/02-building-a-production-rag-pipeline.md) · [Vector databases](../06-embeddings-vector-search/03-vector-databases.md) · [Scaling vector search](../06-embeddings-vector-search/05-scaling-vector-search.md)
- [Architecture patterns](01-architecture-patterns-for-ai-applications.md) · [Caching strategies](07-caching-strategies-for-llm-applications.md)
