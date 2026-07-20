# AI Coding Assistant

## 1. Problem Framing & Requirement Gathering

Design an IDE-integrated AI coding assistant (VS Code / JetBrains / internal EA tooling plugin) serving EA's engineering org: gameplay, engine (Frostbite-adjacent), backend live-services, tools teams. Two fundamentally different latency classes share one product:

- **Inline completion (ghost text)**: fires on every keystroke pause, must feel instantaneous, low-value-per-call but astronomically high call volume.
- **Chat / agentic assistance**: multi-turn, repo-context-heavy, can tolerate seconds of latency, executes code in a sandbox, higher value per call but far lower volume.

Both share repo-context retrieval, model serving infra, and security boundaries, but have opposite scaling and cost profiles — this tension drives most of the architecture.

Key framing questions to state upfront in an interview:
- Is this a build-vs-buy (Copilot/Cursor-style vendor) vs in-house model decision? (We assume **hybrid**: vendor-grade base model via API + in-house retrieval/context/fine-tune layer, since EA has proprietary engine code that cannot leave the network unfiltered.)
- Is code execution a hard requirement (agentic "run tests, iterate") or just suggestion generation? (Assume yes — sandboxed execution is in scope per prompt.)
- Single-repo or monorepo-scale (EA has multi-GB monorepos, e.g. Frostbite-integrated titles)?

## 2. Functional Requirements

- **FR1**: Inline single/multi-line code completion as the developer types, ranked and streamed.
- **FR2**: Chat interface: ask questions about the repo, generate functions, explain code, refactor, generate tests.
- **FR3**: Repo-context retrieval — pull relevant files/symbols/snippets (not just open-file context) into prompt context.
- **FR4**: Multi-file edit / agentic mode: propose a diff spanning multiple files, apply on approval.
- **FR5**: Code execution sandbox — run generated code/tests/linters in an isolated environment and feed results back to the model (agentic loop).
- **FR6**: Support multiple languages/engines relevant to EA: C++, C#, Python, Frostbite script (proprietary DSLs), shader code, Lua.
- **FR7**: Respect per-repo access control — a completion for `Repo A` must never leak context from `Repo B` the user cannot access.
- **FR8**: Telemetry capture (accept/reject/edit-distance) for quality measurement and RLHF-style fine-tuning data.
- **FR9**: Offline/degraded mode — local fallback model when backend unreachable (studio LAN outages, VPN issues at remote studios).
- **FR10**: Admin controls — studio-level model/feature toggles, license/seat management, audit log export for compliance/legal (game IP sensitivity).

## 3. Non-Functional Requirements

| Dimension | Inline Completion | Chat / Agentic |
|---|---|---|
| p50 latency (first token) | ≤ 60 ms | ≤ 800 ms |
| p99 latency (first token) | ≤ 200 ms | ≤ 3 s |
| Full completion latency | ≤ 250 ms p99 | streamed, total ≤ 20 s p99 |
| Availability | 99.9% (degrade to local model, not hard fail) | 99.5% |
| Throughput | ~50K req/s peak org-wide | ~2K req/s peak |
| Consistency | Eventually consistent context index (seconds-old staleness acceptable) | Same |
| Cost ceiling | < $0.0003 amortized per accepted suggestion | < $0.05 per chat turn |
| Durability | Telemetry events durable (no loss > 0.01%) | Chat transcripts durable, encrypted at rest |
| Data residency | Studio-region pinning for proprietary repos | Same |

Non-negotiables specific to EA: unreleased-title source code and design docs are among the company's most sensitive assets — a leak (via prompt injection into a third-party model, or cross-repo context bleed) is a legal/business risk, not just an engineering bug.

## 5. Assumptions

1. ~30,000 engineers across EA studios use the assistant; ~18,000 are daily active (DAU) during core dev cycles.
2. Average engineer generates 120 inline completion requests/hour during active coding (roughly 1 every 30s while typing), across an 8-hour day with ~40% active-coding time.
3. Chat is used by ~35% of DAU, averaging 8 chat turns/day per active chat user.
4. Base LLM: a mix of a fine-tuned 7B-class distilled model for inline completion (self-hosted for latency + cost) and a frontier hosted model (30B+ equivalent) for chat/agentic, accessed via private VPC endpoint (not public internet egress) for large studios' compliance needs.
5. Average repo relevant-context window needed: 8K tokens retrieved + 2K prompt scaffolding for inline; up to 32K tokens for chat/agentic multi-file tasks.
6. Code execution sandbox needed for ~15% of chat sessions (test generation/validation loops).
7. Data residency: three regions — NA (Redwood/Vancouver studios), EU (EA DICE, Romania), APAC (minimal, routes to NA with higher latency tolerance).
8. Monorepos up to 40M LOC / 500K files exist (large engine + game monorepo); most repos are 500K–5M LOC.
9. Telemetry retention: 90 days raw, aggregated metrics retained 2 years for model-quality trend analysis.
10. Security baseline: SOC2-equivalent internal controls already exist; this system must integrate with existing SSO/IAM, not build new identity infra.

## 6. Capacity Estimation

**Inline completion QPS**
- 18,000 DAU × 120 requests/hour × 40% active fraction ≈ 18,000 × 48 req/hr = 864,000 req/hr org-wide.
- Peak concentration (core hours, overlapping US/EU workday tail): assume 3x average over a 4-hour peak window → 864,000 × 3 / (4×3600s) ≈ **180 req/s peak sustained**, bursting to ~**500 req/s** across momentary spikes (large studio all hitting save/format-on-type simultaneously). We'll provision for 600 req/s peak with headroom.
- (Note: figure in NFR table above of "50K req/s" was an org-wide multi-year growth ceiling target, not current load — current design target is ~600 req/s peak, scaling path to 5-10x.)

**Chat QPS**
- 18,000 DAU × 35% chat users = 6,300 chat DAU × 8 turns/day = 50,400 turns/day.
- Peak hour concentration (25% of daily volume in busiest hour) ≈ 12,600 turns/hour ≈ **3.5 req/s** average, bursting to **~15 req/s** peak with agentic multi-call sessions (each agentic session issuing 3-6 model calls).

**Context retrieval / indexing storage**
- Total indexed source across all active repos: assume 200 repos avg 3M LOC, ~50 bytes/LOC raw → 200 × 3M × 50B ≈ 30 TB raw source under index.
- Embedding index: chunk at ~150 tokens/chunk (~600 chars) → 30TB / 600B/chunk ≈ 50M chunks. At 768-dim float16 embeddings (1.5KB/vector incl. overhead) → 50M × 1.5KB ≈ **75 GB** vector data (fits on a handful of ANN index nodes), plus metadata/graph index (symbol references, call graphs) roughly another 150-300 GB in a graph/columnar store.
- Reindex delta: assume 5% of code changes daily → 1.5M chunks/day re-embedded ≈ 1.5M × ~5ms embedding compute ≈ 7,500 GPU-seconds/day ≈ **2.1 GPU-hours/day** for incremental embedding (trivial, one shared embedding-serving pool).

**Model serving footprint**
- Inline model (7B, int8, ~7GB weights): throughput per A10G/L4-class GPU with continuous batching ≈ ~120 concurrent low-token-count (avg 60 output tokens) requests/s per GPU at these latency targets (empirical range 80-150 depending on batching efficiency). Target 600 req/s peak → **~6 GPUs** serving pool (round to 8 for headroom + rolling deploys), plus 2x for multi-region replication ≈ **~16-24 GPUs** total steady state.
- Chat model: if self-hosted 30B-equiv at fp8 on A100/H100, throughput ~8-15 concurrent long-context (8K-32K) streaming sessions per H100 with continuous batching. Peak 15 req/s, avg session occupies GPU for ~4s of active generation → concurrency ≈ 15 × 4 ≈ 60 concurrent streams → **~6-8 H100s** per region, x2 regions ≈ **~14 H100s**. If instead routed to a hosted frontier API (VPC-private endpoint), this becomes 0 self-hosted GPUs but a per-token API bill instead (see Cost section).
- Embedding model serving: small (100M-400M param) encoder, 1-2 GPUs (shared L4 pool) suffices for both query-time and background re-embedding.

**Storage for telemetry/logs**
- Inline events: 600 req/s × 86,400s ≈ 52M events/day, ~300 bytes/event (metadata, not code) ≈ 15.6 GB/day raw → 90-day retention ≈ **1.4 TB** in a columnar store (compresses further ~4x with Parquet+ZSTD → ~350 GB effective).
- Chat transcripts: 50,400 turns/day × ~4KB avg (prompt+response+context refs, not full repo) ≈ 200 MB/day → trivial, **~18 GB over 90 days**, encrypted blob store.

**Sandbox execution capacity**
- 15% of 50,400 chat turns/day involve execution ≈ 7,560 executions/day ≈ peak ~2-3 concurrent executions/s during busy hour. Each sandboxed run: ephemeral container, ~2-10s lifetime, ~0.5 vCPU/512MB-2GB RAM. Peak concurrency ≈ 3 req/s × 6s avg lifetime ≈ **~18 concurrent sandbox containers**, provision pool for 100 for burst/test-suite-style multi-file executions.

## 7. High-Level Architecture

```
                                   ┌─────────────────────────────┐
                                   │        IDE Plugin           │
                                   │ (VS Code / JetBrains client) │
                                   │  - keystroke debounce        │
                                   │  - local cache / local model │
                                   │  - chat UI / diff apply      │
                                   └───────┬──────────────┬──────┘
                                           │              │
                          gRPC/HTTP2 (fast)│              │HTTPS (chat, streamed SSE)
                                           ▼              ▼
                          ┌────────────────────┐  ┌──────────────────────┐
                          │ Inline Completion  │  │   Chat / Agentic     │
                          │   Edge Gateway     │  │      Gateway         │
                          │ (low-latency path) │  │ (session-aware)      │
                          └───────┬────────────┘  └──────────┬───────────┘
                                  │                            │
                       ┌──────────▼──────────┐        ┌────────▼─────────┐
                       │  Auth/AuthZ + Rate   │        │  Session Manager  │
                       │  Limiter (shared)    │        │  (chat history,   │
                       └──────────┬───────────┘        │  agent state)     │
                                  │                     └────────┬─────────┘
                                  ▼                              ▼
                     ┌─────────────────────────┐     ┌───────────────────────┐
                     │ Context Retrieval Svc    │◄────┤  Orchestrator/Agent   │
                     │ - symbol index lookup    │     │  Loop (plan→act→obs)  │
                     │ - ANN vector search      │     └──────────┬────────────┘
                     │ - recent-edit cache      │                │
                     └──────────┬──────────────┘                │
                                │                                ▼
                     ┌──────────▼──────────┐        ┌─────────────────────────┐
                     │  Prompt Assembler /   │        │  Code Execution Sandbox │
                     │  Context Ranker       │        │  (gVisor/Firecracker    │
                     └──────────┬────────────┘        │   microVMs, ephemeral)  │
                                │                       └──────────┬──────────────┘
                                ▼                                  │ (results fed back)
                     ┌─────────────────────────┐                   │
                     │   Model Serving Layer     │◄─────────────────┘
                     │ - inline: distilled 7B    │
                     │   (self-hosted, batched)  │
                     │ - chat: frontier model     │
                     │   (VPC-private endpoint    │
                     │    or self-hosted 30B+)    │
                     └──────────┬──────────────┘
                                │
                     ┌──────────▼──────────┐
                     │  Response Streamer /  │
                     │  Post-processor       │
                     │ (secret-scan, license  │
                     │  filter, diff builder) │
                     └──────────┬────────────┘
                                │
                                ▼
                        back to IDE Plugin

   ── Offline / async plane ──────────────────────────────────────────
   Repo Change Events (git push/webhook)
        │
        ▼
   ┌─────────────┐   ┌───────────────┐   ┌────────────────────┐
   │ Repo Indexer │──▶│ Embedding Svc │──▶│ Vector DB + Symbol  │
   │ (parse AST,  │   │ (batch encode)│   │ Graph Store         │
   │  chunk code) │   └───────────────┘   └────────────────────┘
   └─────────────┘

   Telemetry events (accept/reject/edit-dist) ──▶ Kafka ──▶ Feature Store /
                                                              Data Lake ──▶ Training Pipeline ──▶ Model Registry ──▶ Model Serving Layer (redeploy)
```

## 8. Low-Level Components

| Component | Responsibility | Interface | Scaling Unit |
|---|---|---|---|
| IDE Plugin | Debounce keystrokes, render ghost text, manage local cache & fallback model, apply diffs | Local RPC to edge gateway (gRPC-web/HTTP2), local IPC to on-device model | N/A (client-side) |
| Inline Completion Edge Gateway | Terminate low-latency requests, auth check, route to context+model | gRPC, protobuf schema | Stateless — horizontal pod scale on req/s + p99 latency |
| Chat Gateway | Session-aware routing, SSE streaming, longer-lived connections | HTTPS/SSE | Stateless behind sticky-session LB (session affinity for streaming) |
| Auth/AuthZ + Rate Limiter | Validate SSO token, check per-repo entitlement, enforce quotas | Internal gRPC (shared lib/sidecar) | Scales with total gateway req/s; sidecar pattern (Envoy + OPA) |
| Session Manager | Store chat/agent conversation state, tool-call history | Redis-backed, keyed by session_id | Scales with concurrent chat sessions |
| Context Retrieval Service | Given cursor position/query, fetch top-K relevant chunks + symbols | gRPC: `RetrieveContext(repo_id, query, cursor_ctx) -> chunks[]` | Scale with QPS; read replicas of vector/symbol index |
| Prompt Assembler / Context Ranker | Merge retrieved chunks + open-file context + recent edits, truncate to token budget, rank by relevance | In-process library within gateway or dedicated microservice | CPU-bound, scales with req/s |
| Orchestrator / Agent Loop | Multi-step plan→act→observe loop for agentic chat; decides when to call sandbox, when to re-retrieve context | Internal state machine, calls Model Serving + Sandbox + Context Retrieval | Scales with concurrent agent sessions |
| Model Serving Layer | Host inline + chat models with continuous batching | gRPC (Triton/vLLM API) | GPU pool, autoscale on queue depth/GPU utilization |
| Code Execution Sandbox | Ephemeral isolated execution of generated code/tests | gRPC: `Execute(code, lang, timeout) -> stdout/stderr/exit` | MicroVM pool, autoscale on queue depth |
| Response Streamer / Post-processor | Secret-scanning, license/IP-similarity filter, diff construction, redaction | In-line filter chain | Scales with token throughput |
| Repo Indexer | Parse ASTs, chunk files, detect changed regions from git events | Consumes Kafka topic `repo.changes` | Scales with number of repos / change volume |
| Embedding Service | Batch-encode code chunks into vectors | gRPC batch API | GPU pool, autoscale on backlog |
| Vector DB + Symbol Graph Store | ANN search + exact symbol/call-graph lookups | gRPC/REST query API | Sharded by repo_id |
| Feature Store | Serve features for ranking/personalization (user acceptance history, repo popularity signals) | Online: low-latency KV; Offline: batch table | Partition by user_id/repo_id |
| Telemetry Pipeline (Kafka + Lake) | Durable capture of accept/reject/edit-distance events | Kafka producer API | Partition by studio/repo |
| Training Pipeline | Fine-tune/distill models from telemetry + curated repo data | Batch job (Ray/Kubeflow) | GPU cluster, scheduled |
| Model Registry | Version, stage, and roll out models | REST/gRPC + artifact store | N/A (control plane) |

## 9. API Design

Base path: `https://ai-assist.ea-internal.net/api/v1`. All endpoints require `Authorization: Bearer <SSO JWT>`.

**POST /v1/completions** (inline, latency-critical)
```json
// Request
{
  "repo_id": "fifa-live-service",
  "file_path": "src/net/matchmaking.cpp",
  "cursor": {"line": 214, "col": 18},
  "prefix": "...last 2KB before cursor...",
  "suffix": "...next 512B after cursor...",
  "language": "cpp",
  "session_token": "opaque-client-session-id",
  "request_id": "uuid"
}
// Response (streamed as Server-Sent chunks, final frame shown)
{
  "request_id": "uuid",
  "suggestions": [
    {"text": "if (player.IsConnected()) {\n    ", "confidence": 0.82, "truncated_context_ids": ["chunk_991","chunk_772"]}
  ],
  "latency_ms": 47,
  "model_version": "inline-distill-v4.2.1"
}
```

**POST /v1/chat/sessions** — create a chat session
```json
// Request
{"repo_id": "fifa-live-service", "title_hint": "Fix matchmaking timeout bug"}
// Response
{"session_id": "sess_8a1c...", "created_at": "2026-07-09T10:00:00Z"}
```

**POST /v1/chat/sessions/{session_id}/messages** — send a chat turn (SSE stream response)
```json
// Request
{
  "message": "Why does matchmaking timeout under load? Fix it.",
  "attachments": [{"file_path": "src/net/matchmaking.cpp"}],
  "agentic": true,
  "allow_execution": true
}
// Streamed response events
event: token        data: {"delta": "Looking at "}
event: tool_call    data: {"tool": "retrieve_context", "args": {"query": "matchmaking timeout retry logic"}}
event: tool_result  data: {"chunks_returned": 6}
event: tool_call    data: {"tool": "execute_code", "args": {"lang":"cpp","code":"..."}}
event: tool_result  data: {"exit_code": 0, "stdout": "3 tests passed"}
event: diff         data: {"file": "src/net/matchmaking.cpp", "unified_diff": "@@ -210,6 +210,9 @@ ..."}
event: done         data: {"finish_reason": "completed", "total_latency_ms": 6120}
```

**POST /v1/chat/sessions/{session_id}/apply_diff** — apply proposed diff to workspace (client-confirmed)
```json
{"diff_id": "diff_44a", "accept": true}
```

**POST /v1/telemetry/events** — fire-and-forget batch telemetry
```json
{"events": [{"type":"accept","request_id":"uuid","edit_distance":0,"ts":"..."}]}
```

**GET /v1/repos/{repo_id}/index/status** — indexing freshness check
```json
{"repo_id": "fifa-live-service", "last_indexed_commit": "a1b2c3d", "lag_seconds": 42}
```

Versioning: URI-versioned (`/v1/`), additive-only changes within a version; breaking changes ship as `/v2/` with dual-run period ≥ 90 days and client-side capability negotiation header `X-Assist-Client-Version`.

| Endpoint | Method | Latency Class | Auth |
|---|---|---|---|
| /v1/completions | POST | Critical (p99 <200ms) | Service token (IDE plugin identity + user SSO) |
| /v1/chat/sessions | POST | Standard | User SSO |
| /v1/chat/sessions/{id}/messages | POST (SSE) | Relaxed, streamed | User SSO |
| /v1/chat/sessions/{id}/apply_diff | POST | Standard | User SSO + repo write entitlement |
| /v1/telemetry/events | POST | Best-effort async | Service token |
| /v1/repos/{id}/index/status | GET | Standard | User SSO + repo read entitlement |

## 10. Database Design

| Store | Data | Type | Why |
|---|---|---|---|
| Session/Chat Store | Chat transcripts, agent state | Document store (e.g., DynamoDB/Cosmos-style) | Variable-shape conversation JSON, high write rate, TTL-friendly |
| Symbol Graph Store | AST symbols, call graphs, definition/reference edges | Graph-oriented (or property graph on top of columnar) | Traversal queries ("find all callers of X") are graph-native |
| Vector Index | Code embeddings for ANN retrieval | Specialized vector DB (sharded by repo_id) | Purpose-built ANN, ties to Section 16 |
| Telemetry Events | Accept/reject/edit-distance events | Columnar (Parquet on object store, queried via Presto/Trino) | High-volume append-only, analytical queries, cheap storage |
| Entitlements/ACL | User↔repo access mappings | Relational (Postgres) | Strong consistency required for access control, joins with org chart |
| Model Registry Metadata | Model versions, eval scores, rollout state | Relational (Postgres) | Small, transactional, needs referential integrity |
| Feature Store (offline) | Historical features for training | Columnar (Delta/Iceberg on object store) | Point-in-time joins, large scale batch reads |
| Feature Store (online) | Low-latency serving features | KV store (Redis/DynamoDB) | Sub-ms reads at serving time |

**Sharding/partitioning keys:**
- Vector index & symbol graph: partitioned by `repo_id` — a completion request only ever queries its own repo's shard, giving natural isolation (also enforces the FR7 no-cross-repo-leak requirement structurally, not just via filtering).
- Telemetry: partitioned by `(studio_id, date)` for efficient time-range + org-unit analytical queries.
- Entitlements: partitioned by `user_id` hash, replicated read-heavy (reads >> writes, cache aggressively).

**Schema sketch — entitlements (Postgres)**
```sql
CREATE TABLE repo_entitlements (
  user_id       UUID NOT NULL,
  repo_id       TEXT NOT NULL,
  access_level  TEXT NOT NULL CHECK (access_level IN ('read','write','admin')),
  granted_via   TEXT NOT NULL, -- 'sso_group', 'manual', 'title_assignment'
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, repo_id)
);
CREATE INDEX idx_repo_entitlements_repo ON repo_entitlements(repo_id);
```

**Schema sketch — telemetry event (columnar/Parquet logical schema)**
```
request_id STRING, session_id STRING, studio_id STRING, repo_id STRING,
event_type STRING,      -- accept | reject | edit | error
model_version STRING, latency_ms INT, edit_distance INT,
language STRING, ts TIMESTAMP
-- partitioned by (studio_id, date)
```

## 11. Caching

| Cache | What | Strategy | Invalidation |
|---|---|---|---|
| Local IDE cache | Last N completions per file/cursor-window, local small model outputs | Cache-aside, in-memory LRU | TTL 30s, invalidated on file edit outside cached range |
| Edge gateway response cache | Identical (prefix, suffix, repo commit-hash) completion requests (common for repeated triggers, formatter re-runs) | Cache-aside, keyed by hash(prefix+suffix+model_version+commit) | Invalidate on new commit for that repo path, or 5-min TTL |
| Context retrieval cache | Recently retrieved chunk sets per (repo, cursor region) | Cache-aside | Invalidated by repo indexer push (pub/sub on reindex event), else 60s TTL |
| Symbol/vector index warm cache | Hot repo shards kept warm in memory | Write-through on index update from Repo Indexer | Background refresh on each incremental index commit |
| Entitlements cache | user→repo access | Cache-aside, 5-min TTL, explicit bust on SSO group change webhook | Event-driven invalidation + TTL backstop |
| Model weights cache (GPU) | Hot model shards resident in GPU memory | Write-through at deploy time (pre-warm before traffic cutover) | Redeploy on new model version |

Cache-aside dominates because completion/context data is read-heavy and tolerates a few seconds of staleness (NFR: eventually-consistent context index); write-through is reserved for cases where staleness would cause incorrect security-relevant behavior (entitlements, model version pinning).

## 12. Queues & Async Processing

| Queue | Producer → Consumer | Delivery Guarantee | DLQ Handling |
|---|---|---|---|
| `repo.changes` | Git webhook receiver → Repo Indexer | At-least-once | Poison messages (malformed diff) routed to DLQ, alert + manual replay tool |
| `embedding.jobs` | Repo Indexer → Embedding Service | At-least-once (idempotent by chunk_hash) | Retry 3x with backoff, then DLQ; embedding is idempotent so duplicate delivery is safe |
| `telemetry.events` | IDE Plugin/Gateway → Telemetry Pipeline | At-least-once, dedup by request_id downstream | DLQ for schema-invalid events, sampled for debugging, auto-purged after 7 days |
| `agent.tool_calls` | Orchestrator → Sandbox Executor | At-least-once with idempotency key per tool-call | Failed executions surface as tool error to agent loop, not silently dropped; 3 retries then abort agent step |
| `model.eval_jobs` | Training Pipeline → Eval workers | Exactly-once semantics needed (avoid double-counting eval metrics) — achieved via transactional outbox + consumer-side idempotent upsert | DLQ + paging to ML on-call if eval job fails after retries |

Exactly-once is only pursued where double-processing corrupts a metric or a billing count (eval jobs, cost-attribution jobs); everywhere else at-least-once + idempotent consumers is cheaper and sufficient.

## 13. Streaming & Event-Driven Architecture

- **Broker**: Kafka (or Kinesis-equivalent), topics partitioned by `repo_id` or `studio_id` for locality.

| Topic | Schema (key fields) | Consumer Groups |
|---|---|---|
| `repo.changes` | `repo_id, commit_sha, changed_paths[], author, ts` | `repo-indexer-group`, `security-scan-group` (secret-scanning on new commits) |
| `completion.events` | `request_id, repo_id, user_id_hashed, event_type, latency_ms, model_version` | `telemetry-ingest-group`, `realtime-quality-dashboard-group` |
| `chat.events` | `session_id, turn_id, tool_calls[], tokens_in, tokens_out, latency_ms` | `telemetry-ingest-group`, `cost-attribution-group` |
| `model.deploy_events` | `model_version, stage, rollout_pct, ts` | `serving-fleet-config-group` (drives canary traffic splitting) |
| `security.alerts` | `finding_type, repo_id, severity, snippet_hash` | `security-oncall-group` |

- Consumer groups are scaled independently per concern — e.g., the realtime-quality-dashboard consumer can lag without affecting the durable telemetry-ingest path, since they're separate groups reading the same partitioned topic independently.
- Schema evolution via a schema registry (Avro/Protobuf), backward-compatible only; producers cannot ship a breaking schema change without a registry-enforced compatibility check.

## 14. Model Serving

| Model | Framework | Batching | Hardware | Notes |
|---|---|---|---|---|
| Inline completion (distilled ~7B) | vLLM or Triton + TensorRT-LLM | Continuous/in-flight batching, max batch 128, speculative decoding for short completions | L4/A10G, int8 quantized | Latency-critical: small model, short output (~60 tokens avg), aggressive batching window (≤5ms) |
| Chat/agentic (30B+ or hosted frontier) | vLLM (self-hosted) or private API endpoint | Continuous batching, larger batching window (~20-50ms) tolerable given relaxed latency budget | H100 (self-hosted) or managed endpoint | Long-context (up to 32K), streaming output |
| Embedding encoder | Triton | Static/dynamic batching, batch 256 | L4 (shared pool) | Bulk background encode + low-latency query-time encode share pool with priority lanes |
| Ranking/relevance model (context ranker) | Lightweight (gradient-boosted or small transformer) | CPU-served, no batching needed | CPU pool | Cheap enough to not warrant GPU |

- **Multi-model serving**: single GPU fleet hosts multiple model versions concurrently (current stable + canary) via Triton model repository with per-model resource limits; inline and chat pools are physically separate GPU pools because their latency SLOs and batching windows conflict (inline cannot share a batching queue with 8s chat generations).
- **Speculative decoding** for inline: small draft model proposes tokens, main distilled model verifies — cuts p50 latency further for the common case of highly-predictable completions (closing brackets, boilerplate).
- **Quantization**: int8 for inline (accuracy tradeoff acceptable given short completions, high volume); fp8/bf16 for chat (accuracy matters more for multi-step reasoning/agentic correctness).

## 15. Feature Store

- **Online store** (Redis/DynamoDB): per-user acceptance-rate history, per-repo language distribution, recent-edit recency features, session-level context (open files) — all needed at inference time with sub-5ms lookup budget to stay inside the 60ms p50 inline budget.
- **Offline store** (Iceberg/Delta on object store): full historical telemetry joined with repo metadata, used for training the ranking model and for distillation/fine-tuning data curation.
- **Point-in-time correctness**: training the completion-ranking model must join "features as they were at request time" (e.g., user's acceptance rate *before* this request, not including it) — implemented via event-time-based point-in-time joins (Iceberg time-travel / Tecton-style point-in-time queries) to avoid label leakage where a later accept/reject event contaminates an earlier feature snapshot.
- Feature freshness SLA: online features refreshed within 5 minutes of underlying event (acceptable staleness for ranking signal; not safety-critical).

## 16. Vector Database

Applicable — used for semantic code-chunk retrieval feeding both inline context and chat/agentic context.

- **Indexing strategy**: HNSW (Hierarchical Navigable Small World) per-repo-shard index. HNSW chosen over IVF-PQ because:
  - Repo shard sizes (thousands to low-millions of chunks per repo) are small enough that HNSW's higher memory footprint (~1.2-2x raw vector size) is affordable, and HNSW gives better recall at low latency without needing a training/clustering step (IVF requires periodic re-clustering as code changes, adding operational overhead).
  - Recall target ≥ 0.95 at k=20 with query latency ≤ 10ms — HNSW hits this comfortably at these shard sizes.
- **Sharding**: one HNSW index per `repo_id` (ties to Section 10's isolation-by-shard design) — bonus: enables per-repo index deletion (e.g., when a title is decommissioned or access revoked) without touching other repos.
- **Hybrid retrieval**: combine ANN vector search with exact symbol-graph lookup (BM25-style keyword match on identifiers + call-graph traversal) — pure embedding similarity misses exact-name lookups ("find `MatchmakingTimeout` usages") that engineers expect to work reliably; final ranking blends both signal types.
- **Freshness**: incremental upserts on file change (not full rebuild); background compaction job merges HNSW graph periodically (nightly) to bound graph fragmentation from many small incremental inserts.

## 17. Embedding Pipelines

Applicable.

- **Chunking**: AST-aware chunking (not naive fixed-token windows) — split at function/class boundaries where possible, fallback to sliding window with overlap for very large functions (common in game engine code with long functions). Target ~150-300 tokens/chunk.
- **Encoder**: small (100M-400M param) code-specialized encoder (e.g., a CodeBERT-family or in-house fine-tuned encoder), chosen over reusing the large generative model for embeddings — far cheaper per-chunk and sufficient for retrieval quality at this scale.
- **Batch pipeline**: Repo Indexer detects changed files via git diff → chunker → batched embedding requests (256/batch) → upsert into vector shard.
- **Incremental vs full**: incremental re-embed only changed chunks (hash-based change detection at chunk granularity, not file granularity, to avoid re-embedding an entire 2000-line file for a 1-line change).
- **Query-time embedding**: same encoder, low-latency single-item inference path (shared GPU pool, priority-lane scheduling so background bulk jobs don't starve query-time latency).

## 18. Inference Pipelines

**Inline completion — request lifecycle:**

```
1. Keystroke pause (debounce ~150ms client-side)
2. IDE plugin → Edge Gateway: prefix/suffix + cursor + repo_id           [~5ms network, same-region]
3. Gateway: auth check (cached entitlement)                              [~1ms]
4. Context Retrieval: ANN search (repo shard) + recent-edit cache hit    [~8-15ms]
5. Prompt Assembler: merge + truncate to token budget                    [~2ms]
6. Model Serving: continuous-batched inference, speculative decoding     [~25-40ms to first token, streamed]
7. Post-processor: secret-scan regex pass on output                      [~2ms]
8. Stream tokens back to IDE plugin, render ghost text incrementally     [~5ms network]
9. Async (non-blocking): telemetry event emitted to Kafka                [fire-and-forget]

Total p50 ≈ 50-60ms first token, full suggestion ≈ 150-250ms p99
```

**Chat/agentic — request lifecycle (ASCII sequence):**

```
IDE Plugin        Chat Gateway     Orchestrator      Context Svc      Model Serving     Sandbox
    │                  │                │                │                │              │
    │──POST message───▶│                │                │                │              │
    │                  │──create step──▶│                │                │              │
    │                  │                │──retrieve─────▶│                │              │
    │                  │                │◀──chunks───────│                │              │
    │                  │                │──prompt+ctx───────────────────▶│              │
    │                  │◀── stream tokens (plan/explanation) ─────────────│              │
    │◀──SSE tokens──────│                │                │                │              │
    │                  │                │──tool_call: execute_code───────────────────────▶│
    │                  │                │◀──stdout/exit───────────────────────────────────│
    │                  │                │──re-prompt with exec result───▶│              │
    │                  │◀── stream tokens (fix/diff) ──────────────────────│              │
    │◀──SSE diff event──│                │                │                │              │
    │──apply_diff──────▶│                │                │                │              │
```

- Agentic loop bounded by max-steps (e.g., 8 tool calls) and wall-clock timeout (20s) to prevent runaway cost/latency.
- Every tool call (retrieve, execute, re-retrieve) is logged with `session_id` + `turn_id` for replayability/debugging.

## 19. Training Pipelines

- **Base model**: start from an open-weights or vendor-licensed base code model; not trained from scratch (cost-prohibitive, and EA's proprietary corpus alone is not large enough to be competitive as a from-scratch pretraining corpus).
- **Fine-tuning data prep**:
  - Curate from EA's own repos (with consent/opt-out per studio) — dedupe, PII/secret-scrub, license-filter (exclude any third-party-licensed code snippets pulled into repos under incompatible licenses).
  - Weight recent/actively-maintained repos higher; downweight generated/vendored code and dead branches.
  - Construct instruction-tuning pairs from accepted-completion telemetry (context → accepted completion) and from chat sessions where the final diff was accepted/merged (implicit positive signal).
- **Distillation**: distill a large teacher (chat-tier model) down to the 7B inline model — teacher generates completions across sampled real contexts, student trained to match teacher output distribution; this is how the inline model stays cheap/fast while inheriting quality from the larger model.
- **Orchestration**: Ray Train or Kubeflow on a dedicated GPU training cluster (separate from serving fleet — training and serving must not compete for the same GPUs during business hours).
- **Distributed training**: FSDP/ZeRO-stage-3 sharding across an 8-64 H100 node pool for fine-tuning runs (full pretraining not in scope per above); gradient checkpointing to fit longer code-context sequences.
- **Eval before promotion**: held-out repo set (never trained on) + human-in-the-loop preference eval (EA engineers rate sample completions blind A/B) before any candidate model is promoted to canary.

## 20. Retraining Strategy

- **Cadence**: 
  - Inline model distillation refresh: monthly (balances staying current with new coding patterns/frameworks vs. eval/rollout overhead).
  - Chat model fine-tune refresh (if self-hosted) or prompt/retrieval-strategy refresh (if using a hosted frontier model): every 6-8 weeks, aligned with major base-model version updates from vendor.
  - Ranking/relevance model (feature-store-driven): weekly retrain (cheap, fast-moving signal — acceptance patterns shift as new engineers onboard or new repos are added).
- **Trigger-based retraining** (in addition to cadence):
  - Acceptance rate drops > 5 percentage points vs 30-day rolling baseline for a given language/repo cohort → triggers investigation + possible off-cycle retrain.
  - New major language/framework adoption in a large repo (e.g., a studio migrates to a new engine subsystem) with high reject rate on that code → triggers targeted fine-tune data collection.
  - Security-relevant trigger: a new class of unsafe pattern found being suggested (e.g., a CVE-adjacent API misuse) → immediate targeted fine-tune/patch + interim prompt-level guardrail as a stopgap.

## 21. Drift Detection

| Drift Type | Signal | Metric | Threshold |
|---|---|---|---|
| Data drift (input) | Distribution of languages/repo mix hitting the model shifts (e.g., sudden surge in a DSL the model wasn't tuned on) | PSI (Population Stability Index) on language/framework feature distribution, weekly | PSI > 0.2 → investigate; > 0.3 → page ML on-call |
| Data drift (context) | Retrieved-context relevance score distribution shifts (context ranker's own confidence) | Rolling mean of top-k retrieval relevance score | Drop > 10% over 7-day window → flag |
| Concept drift | Acceptance rate / edit-distance-on-accept trending down for stable language mix (model quality degrading relative to evolving codebase conventions) | 7-day rolling acceptance rate, edit-distance-after-accept | Acceptance rate drop > 5pp, or edit-distance-after-accept up > 20% → trigger retrain evaluation |
| Prompt/context drift | Average retrieved-context token utilization vs. budget (symptom of repo growth outpacing chunking strategy) | Context truncation rate | Truncation rate > 15% of requests → re-tune chunking/ranking |
| Label drift (chat) | Diff-apply rate (agentic proposals actually applied vs. discarded) | Rolling apply rate | Drop > 8pp week-over-week → flag for review |

Detection cadence: automated weekly batch job computing all metrics above, dashboarded, with automatic paging only on the "page ML on-call" threshold; softer thresholds surface as dashboard flags reviewed in weekly model-quality sync.

## 22. Monitoring

| Layer | Metrics |
|---|---|
| Infra | GPU utilization, GPU memory, batch queue depth, gateway req/s, error rate, p50/p95/p99 latency per endpoint, sandbox container pool utilization, Kafka consumer lag |
| Model quality | Acceptance rate (inline), edit-distance-after-accept, diff-apply rate (chat), retrieval relevance score, hallucination-flag rate (chat, via heuristic/secondary-model check), context truncation rate |
| Business | DAU/MAU of assistant, per-studio adoption rate, engineer-time-saved estimate (proxy: accepted-completion char count × typing-speed baseline), cost per active user, seat license utilization |
| Security | Secret-scan hit rate (suggestions blocked for containing potential secrets), cross-repo-access-denial count, sandbox escape attempts (should be zero — any nonzero is a P0) |
| Data pipeline | Repo indexing lag (commit → searchable), embedding backlog depth, DLQ depth across all queues |

## 23. Alerting

| Condition | Threshold | Severity | Routing |
|---|---|---|---|
| Inline p99 latency | > 300ms for 5 consecutive minutes | P1 | Page serving on-call |
| Model serving GPU pool error rate | > 2% for 3 minutes | P1 | Page serving on-call |
| Repo indexing lag | > 10 minutes for any actively-edited repo | P2 | Slack alert to platform team |
| Sandbox escape / anomalous syscall detected | any occurrence | P0 | Immediate page to security on-call + auto-quarantine sandbox node |
| Kafka consumer lag (telemetry) | > 15 minutes | P3 | Ticket, next business day |
| DLQ depth (any queue) | > 1000 messages | P2 | Slack alert + auto-created investigation ticket |
| Acceptance rate drop | > 5pp vs 7-day baseline, sustained 24h | P2 | ML on-call, non-paging (dashboard + Slack) |
| Cost anomaly | daily spend > 130% of 7-day trailing average | P2 | FinOps + eng lead notified |
| Entitlement cache staleness causing access errors | error rate spike on 403s > 3x baseline | P1 | Page platform on-call (possible security-relevant bug) |

## 24. Logging

- **Structured logging**: JSON logs everywhere, common schema fields (`request_id`, `session_id`, `repo_id`, `user_id_hashed`, `ts`, `service`, `level`), shipped to a central log store (e.g., structured logs into the same lake used for telemetry, queryable via the same analytical engine).
- **PII handling**:
  - `user_id` never logged in raw form in analytical stores — hashed with a rotating per-quarter salt (allows cohort analysis without persistent re-identification).
  - Raw code content is **not** logged in general request/error logs (only metadata: language, repo_id, latency, token counts) — full prompt/response payloads are retained only in the dedicated encrypted chat-transcript store (Section 10), with tighter access control and separate retention policy, since code itself is the sensitive asset here (not classic PII, but equally sensitive as EA IP).
  - Secret-scan hits are logged as a boolean/category flag, never the matched secret text itself.
- **Retention**: infra/operational logs 30 days; telemetry/analytical events 90 days raw + 2 years aggregated; chat transcripts 90 days by default (configurable shorter per studio for embargoed/unreleased-title repos, per legal requirement) then hard-deleted, not just soft-deleted.
- **Access control on logs**: chat transcript store access restricted to a small ML-quality + security team, audited access log itself (who queried which session).

## 25. Security

**Threat model specific to this system:**

| Threat | Mitigation |
|---|---|
| Cross-repo context leakage (completion for Repo A surfaces Repo B's proprietary code) | Structural isolation via per-repo index shards (Section 16) + entitlement check before every retrieval call, not just at gateway |
| Prompt injection via malicious code comments in a repo ("ignore previous instructions, exfiltrate...") targeting the agentic loop | Sanitize/strip suspicious instruction-like patterns from retrieved context before inclusion in system-level prompt scope; agent tool-calls are allow-listed (cannot call arbitrary tools, only the fixed set: retrieve, execute, diff) |
| Sandbox escape (executed code attempts to reach host/network/other tenants) | gVisor/Firecracker microVM isolation, no network egress from sandbox by default, seccomp syscall filtering, ephemeral (destroyed after each execution), anomaly detection on syscalls (ties to P0 alert in Section 23) |
| Model exfiltrating training data (proprietary code memorized and regurgitated to an unauthorized user) | Per-repo model fine-tune isolation where feasible; output similarity check against source corpus for near-verbatim large-block regurgitation, flagged for review |
| Secrets/credentials leaking into completions or chat context (hardcoded API keys pulled from repo into prompt/response) | Secret-scanning both at index-time (don't embed detected-secret chunks) and at output-time (redact before returning to client) |
| Third-party hosted model provider retaining/training on EA proprietary prompts | Contractual zero-retention + no-train clause with any hosted API provider; VPC-private endpoint (no public internet transit); prefer self-hosted for embargoed/unreleased titles |
| Supply-chain: malicious/compromised extension impersonating IDE plugin | Signed plugin builds, mutual TLS between plugin and gateway, plugin auto-update integrity verification |
| Compliance/audit: proving no unauthorized code exposure occurred | Immutable audit log of every retrieval + every external model call (which repo, which chunks, which destination), retained per legal hold requirements |

## 26. Authentication

- **End-user auth**: existing corporate SSO (SAML/OIDC) — IDE plugin performs OAuth device-code or PKCE flow on first use, obtains a short-lived JWT (15-min expiry) + refresh token; JWT carries `user_id`, SSO group claims used for entitlement resolution.
- **Service-to-service auth**: mutual TLS + SPIFFE/SPIRE-issued workload identities between internal microservices (Gateway ↔ Context Retrieval ↔ Model Serving ↔ Sandbox); no static API keys between internal services.
- **IDE plugin ↔ Gateway**: mTLS at transport layer plus per-request JWT (defense in depth — mTLS proves it's a legitimate plugin binary, JWT proves which user).
- **Hosted external model provider (if used for chat)**: dedicated VPC-private endpoint with its own service credential, scoped to a specific request-signing mechanism, rotated automatically (short-lived tokens via secrets manager, not long-lived static keys).
- **Sandbox executor**: no ambient credentials at all inside the microVM — by design, executed code cannot reach any authenticated internal service (no network egress).

## 27. Rate Limiting

- **Algorithm**: token bucket per user for inline completions (allows short bursts from fast typing while capping sustained abuse), sliding-window counter per user for chat (smoother enforcement given higher per-call cost).
- **Per-user limits**: 
  - Inline: burst 20 req/2s, sustained 300 req/min (well above real human typing-triggered rate, generous to avoid false throttling, but bounds a runaway/misbehaving plugin instance).
  - Chat: 30 turns/hour soft limit (warns, doesn't hard-block), 100 turns/hour hard cap (protects shared GPU pool from a single runaway agentic loop or scripted misuse).
- **Per-tenant (studio) limits**: aggregate GPU-second budget per studio per day, enforced as a soft quota with alerting (Section 23 cost anomaly) rather than hard cutoff — avoids blocking a studio mid-crunch, but flags abnormal consumption for review.
- **Enforcement point**: at the Edge Gateway/Chat Gateway (Envoy + rate-limit service, e.g. Envoy's global rate limit service backed by Redis counters) — enforced before context retrieval/model serving to avoid wasting GPU cycles on requests that will be rejected.
- **Sandbox execution**: separately rate-limited per-session (max 8 executions per agentic session, ties to Section 18's bounded agent loop) to bound cost and prevent it becoming a compute-abuse vector.

## 28. Autoscaling

- **Inline completion gateway/context-retrieval pods**: HPA on `req/s` and p99 latency custom metric (via Prometheus adapter) — scale out aggressively (fast scale-up, 30s stabilization window) since traffic is bursty around start-of-workday and post-lunch.
- **Model serving GPU pools**: KEDA-based autoscaling on **queue depth** (pending requests in the batching queue) rather than raw GPU utilization alone, since GPU util can look "fine" while queue depth silently grows right before a latency SLO breach. Scale-up trigger: queue depth > 20 sustained 15s. Scale-down: conservative (5-min cooldown) given multi-minute GPU cold-start (model load) time.
- **VPA**: applied to context-retrieval and prompt-assembler pods (memory footprint scales with repo shard size/hot cache) — right-sizes requests/limits automatically rather than over-provisioning statically.
- **Sandbox executor pool**: KEDA scaling on `agent.tool_calls` queue depth, fast scale-up (microVM boot ~125ms with Firecracker, cheap to over-provision a small buffer pool).
- **Chat gateway**: HPA on concurrent SSE connections (custom metric) rather than CPU, since a chat gateway pod's bottleneck is open-connection count, not compute.

## 29. Cost Optimization

- **Spot/preemptible instances**: training cluster (Section 19) runs entirely on spot/preemptible GPUs with checkpointing every N steps — training is restartable, tolerates preemption, unlike serving.
- **Model distillation**: inline model is a distilled 7B rather than routing every keystroke-triggered request to the frontier chat-tier model — single largest cost lever given inline's 100x+ volume vs chat.
- **Aggressive caching**: edge response cache + context retrieval cache (Section 11) cuts redundant model calls for repeated/near-identical requests (common with formatter-triggered re-completions).
- **Batching**: continuous batching on GPU serving maximizes throughput/GPU, directly cutting $/request.
- **Quantization**: int8 for inline model halves memory footprint and roughly doubles throughput/GPU vs fp16, with negligible quality loss for short completions.
- **Speculative decoding**: reduces average generated-token latency and GPU-seconds-per-completion for the common "predictable" case.
- **Tiered chat routing**: simple/short chat queries (e.g., "explain this line") routed to a cheaper mid-size model; only complex agentic/multi-file tasks routed to the frontier-tier model — a router classifies query complexity first.
- **Sandbox right-sizing**: ephemeral microVMs sized minimally (0.5 vCPU/512MB default, escalate only if execution requests more), and pooled/pre-warmed rather than cold-started per call.
- **Idle GPU reclaim**: off-peak (overnight per-region) autoscale-down of chat GPU pool to a minimal floor, since chat volume closely tracks working hours per region.
- **Cost attribution & showback**: per-studio cost dashboards (Section 22 business metrics) create organic pressure against wasteful usage patterns (e.g., overly long chat context windows attached unnecessarily).

## 30. Operational Concerns (Deployment, Reliability, Infra)

At SDE2 scope, treat this as a checklist rather than a design exercise: **backups** (automated snapshots of the model registry, feature store, and any stateful service, with a tested restore path), **rollback** (every deploy must be revertible to the last-known-good version — the model registry and CI/CD pipeline should make this a one-command operation), **canary/blue-green rollout** (shift a small percentage of traffic first, watch error rate and key business/model metrics, then ramp), and **basic observability** (dashboards + alerts on latency, error rate, and the top 2-3 model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level infra-architecture concerns — worth knowing they exist, not worth rehearsing the manifests.

## 38. Why This Architecture

- **Split gateways (inline vs chat)** directly reflects the prompt's central tension: one path optimizes for microseconds-matter, high-QPS, small-payload traffic; the other for correctness/depth over many seconds at low QPS. Merging them into one path would force either over-provisioning GPU for the low-value inline path or under-serving chat's larger context needs.
- **Per-repo shard isolation** (vector index, symbol graph) turns EA's hardest non-functional requirement — no cross-repo/cross-title IP leakage — into a structural guarantee rather than a hope pinned entirely on application-layer filtering, which is safer for an asset class (unreleased game source) where a single leak is a business-level incident.
- **Distillation for inline / larger model for chat** is the single biggest cost and latency lever given the 100x+ volume asymmetry between the two paths (Section 6) — serving every keystroke with a frontier model is both unaffordable and unnecessarily slow.
- **Sandboxed, network-isolated code execution** is non-negotiable given the threat model (Section 25): an agentic loop that executes arbitrary model-generated code against real infrastructure is a direct path to a security incident; microVM isolation with no egress is the standard mitigation at acceptable cost/latency overhead (~100ms boot).
- **Event-driven reindexing** (git webhook → Kafka → indexer → embed → upsert) decouples code-change velocity from query-time latency — engineers pushing at 3am shouldn't cause query-time index-rebuild stalls for daytime users.

## 39. Alternative Architectures

| Alternative | Description | Why Rejected / When Preferred |
|---|---|---|
| Single unified model + single gateway for both inline and chat | One large model serves both paths, simpler ops | Rejected: forces either GPU over-provisioning to hit inline's 200ms p99 with a large model, or under-serves chat's context depth; only viable at far lower QPS scale than EA's, or in a very early-stage/low-volume pilot where operational simplicity outweighs cost efficiency |
| Fully third-party SaaS assistant (no in-house context/retrieval layer) | Adopt an off-the-shelf Copilot/Cursor-style product wholesale | Rejected as sole solution: cannot guarantee proprietary engine code isolation/data-residency per studio, and generic retrieval underperforms on EA's proprietary DSLs/engine idioms; **preferred when**: a studio has no unreleased/embargoed IP concerns and just needs generic-language (e.g., pure Python tooling scripts) assistance — a lightweight opt-in SaaS tier could coexist for those cases |
| Fully local, no backend (all inference on developer workstation) | Ship a small local model, no server round-trip at all | Rejected as primary: local-only models lag frontier quality significantly and can't do repo-scale retrieval or heavy agentic execution; **preferred as**: the offline/degraded-mode fallback (already in scope, FR9) — right tradeoff for VPN-outage resilience, wrong tradeoff as the primary architecture |
| Synchronous, non-streaming chat responses | Return full chat completion in one blocking response instead of SSE streaming | Rejected: user-perceived latency for a 20s agentic task would feel broken without incremental token/tool-call visibility; only acceptable for very short, bounded chat use cases (e.g., a fixed one-shot "explain this line" endpoint could reasonably be synchronous) |

## 40. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Distilled small model for inline | Cheap, fast, meets 60ms p50 | Lower ceiling on completion quality/creativity vs frontier model |
| Per-repo vector/symbol shards | Strong isolation, simpler per-repo lifecycle (delete on decommission) | No cross-repo semantic search (can't ask "how do other EA repos solve this" — acceptable given IP isolation goals, but a real capability loss) |
| Self-hosted chat model vs hosted frontier API | Full data control, no per-token vendor bill, no network egress risk | Higher fixed GPU cost, own responsibility for keeping pace with frontier model quality improvements |
| Continuous batching for inline | Maximizes GPU throughput/cost efficiency | Adds a few ms of queueing jitter, complicates p99 tail-latency reasoning vs simple 1-request-per-GPU-slot model |
| Sandbox with no network egress | Strong security posture | Blocks legitimate use cases needing package installs/network calls during generated-test execution; requires pre-baked dependency images as a workaround |
| At-least-once delivery almost everywhere | Simpler, cheaper than exactly-once | Requires idempotent consumers everywhere; a bug in idempotency key handling silently causes duplicate processing rather than loud failure |
| Active-active multi-region with per-region data residency | Meets legal/latency requirements, no giant cross-region index replication cost | Cannot serve a user's home-region content if they're traveling/on a different region's network without explicit re-routing logic; added operational complexity of N independent index fleets |

## 41. Failure Modes

| Scenario | Detection | Mitigation |
|---|---|---|
| Model serving GPU pool OOM under traffic spike (e.g., all-hands hack day surge) | GPU error rate alert (Section 23) | Autoscale on queue depth (Section 28) with pre-provisioned burst headroom; graceful degradation to smaller/faster fallback model if pool saturated |
| Vector index shard for a hot repo becomes corrupted/unavailable | Context retrieval error rate spike on that repo_id | Serve completions with reduced context (open-file-only, no retrieval) rather than hard-failing; background job triggers reindex-from-source for that shard |
| Sandbox executor pool exhausted during a burst of agentic sessions | Sandbox queue depth alert | Reject new execution requests gracefully (agent surfaces "execution unavailable, here's the code without verification" rather than hanging); autoscale microVM pool |
| Kafka broker outage in one region | Consumer lag/connection alert | Producers buffer locally with bounded retry (IDE plugin/gateway already tolerant of async telemetry loss up to a point); cross-region Kafka mirroring for critical topics |
| Entitlements DB primary failover | Elevated 403/500 error rate | Automated Postgres failover to synchronous replica (RTO 30 min per Section 30); gateway retries with backoff during failover window |
| Hosted frontier API provider outage (if used for chat) | Chat gateway error rate spike specific to external calls | Circuit breaker trips, falls back to self-hosted mid-tier model with degraded-quality banner in chat UI, rather than full chat outage |
| Prompt-injection payload discovered live in a popular repo | Security-alerts topic consumer flags anomalous instruction-like content in retrieved chunks | Immediate sanitization filter deploy (hotfix, doesn't require full model redeploy since it's a context-filtering layer), affected repo re-scanned |
| Runaway agentic loop (model keeps calling tools without converging) | Max-steps/wall-clock timeout (Section 18) | Hard-abort at step/time limit, surface partial progress to user, log for training-data review (bad agentic trace excluded from future fine-tune) |

## 42. Scaling Bottlenecks

**At 10x current scale (~180K DAU, ~6,000 req/s inline peak):**
- Inline model serving GPU pool becomes the first bottleneck — 6-8 GPUs today scales roughly linearly to ~60-80 GPUs; still tractable but now a meaningful fleet requiring careful multi-AZ/multi-region GPU capacity planning and likely hitting cloud-provider GPU quota/availability limits, especially for L4/A10G classes during shared-capacity contention.
- Entitlements DB (single Postgres primary per region) starts showing write contention if SSO group changes + repo-access grants spike in volume — likely needs read-replica fan-out sooner, or a move to a distributed SQL option.
- Kafka partition count for `completion.events` topic needs re-partitioning to keep consumer parallelism ahead of 10x event volume (52M/day → ~520M/day).

**At 100x scale (~1.8M DAU — beyond realistic EA headcount, but illustrative of a platform-wide/industry-wide rollout scenario):**
- Per-repo shard model breaks down for the handful of "mega-repos" (the 40M LOC monorepo) — a single shard's HNSW index may need to become itself distributed/sub-sharded by directory/module, reintroducing cross-shard query complexity within what was meant to be an isolation boundary.
- GPU capacity at this scale likely requires dedicated reserved capacity contracts rather than on-demand cloud GPU, fundamentally changing the cost model and capacity-planning cadence (multi-quarter hardware lead times).
- Feature store online layer (Redis/DynamoDB) read volume at 100x could hit single-cluster ceilings, requiring geo-sharded feature serving rather than single-region-per-cluster.
- Telemetry pipeline's columnar lake query layer (Trino/Presto) needs materialized-aggregate pre-computation rather than ad-hoc queries over raw 100x-scale event volume for dashboards to stay responsive.

## 43. Latency Bottlenecks

**Inline completion p50 (~55ms) / p99 (~200ms) budget breakdown:**

| Stage | p50 | p99 |
|---|---|---|
| Client debounce + network to gateway | 5ms | 15ms |
| Auth/entitlement check (cached) | 1ms | 5ms |
| Context retrieval (ANN + cache) | 10ms | 60ms (cache miss + cold shard) |
| Prompt assembly | 2ms | 5ms |
| Model inference (first token, batched, speculative decode) | 30ms | 90ms (batching queue contention at peak) |
| Post-processing (secret scan) | 2ms | 5ms |
| Network back to client | 5ms | 15ms |
| **Total** | **~55ms** | **~195ms** |

- The p99 tail is dominated by two things: **context retrieval cache misses hitting a cold/under-warmed shard**, and **batching queue contention during peak traffic** (a request arrives just after a batch window closes and waits for the next). Both are the primary targets for further p99 optimization (larger warm-cache footprint; tighter/adaptive batching windows).

**Chat/agentic p50 (~800ms first token) / total session latency:**
- First-token latency dominated by prompt assembly with larger context (up to 32K tokens) plus a larger, less-batched model — prefill time on long context is the single biggest lever (mitigated by prompt caching of stable repo-context prefixes across turns in the same session — reusing KV-cache for unchanged context significantly cuts repeated prefill cost turn-over-turn).
- Full agentic session latency (up to ~20s) dominated by the number of tool-call round-trips (each retrieval/execution adds ~50-500ms); minimizing unnecessary re-retrieval (only re-retrieve context when the working file set changes) is the main lever there.

## 44. Cost Bottlenecks

- **GPU fleet (inline serving)** is the largest steady-state cost driver by request volume alone, even with a small distilled model — sheer QPS (Section 6) means even fractions-of-a-cent per request accumulate: ~600 req/s × 86,400s/day ≈ 52M completions/day; at an illustrative $0.00005-0.0001 amortized GPU cost per completion (small model, batched), that's **~$2,600-$5,200/day (~$80K-$155K/month)** just for inline serving compute — the single line item most worth continual optimization (batching efficiency, quantization, speculative decoding all directly attack this).
- **Chat GPU fleet or hosted-API spend**: far lower volume but higher per-call cost; if using a hosted frontier API at (illustrative) $3-15 per million tokens blended, with ~50,400 turns/day × ~3K tokens avg (prompt+context+response) ≈ 150M tokens/day ≈ **$450-$2,250/day** — smaller than inline in aggregate today, but grows faster per-user as agentic usage/context length increases, and is the line item most sensitive to "context bloat" (over-retrieving unnecessary context inflates token cost directly).
- **Training/fine-tuning GPU spend**: bursty (monthly/bi-monthly jobs), but a multi-day H100 training run across dozens of nodes is a large single-event cost — mitigated by spot pricing (Section 29), but still the largest *per-event* line item on the bill even if not the largest steady-state one.
- **Storage** (vector index, telemetry lake, chat transcripts) is comparatively minor (tens of TB, not hundreds) — not a primary cost lever at current scale, revisit at 100x (Section 42).
- **Cross-region data transfer**: entitlements/registry replication is small; the main transfer cost risk is if vector-index or transcript data were ever mistakenly cross-replicated at full volume — explicitly avoided by design (Section 31) partly *for this cost reason* in addition to the residency reason.

## 45. Interview Follow-Up Questions

1. How would you prevent the inline completion model from ever suggesting a hardcoded secret or credential it saw in a *different* repo's training data?
2. Walk me through what happens end-to-end if the vector index for a repo is stale by 10 minutes — what's the actual user-visible impact, and is that acceptable?
3. Your agentic loop calls the sandbox, which times out. What does the model see, and how does it recover gracefully instead of hallucinating a success?
4. How do you decide, technically, whether a chat query should be routed to the cheap mid-tier model vs the frontier model?
5. Two engineers on the same shared branch get conflicting agentic diff suggestions in the same file within seconds of each other — how does your system avoid data races or silently clobbering context?
6. How would you A/B test a new inline completion model's *actual* quality impact, not just latency/error rate — what's your ground truth given you don't have "correct" labels?
7. What's your plan if a hosted frontier model provider changes their retention/training policy without much notice — how fast can you cut over, and to what?
8. How do you bound the blast radius if the repo indexer itself is compromised or bugged and starts embedding malicious/incorrect code into the vector index?
9. At what point does per-repo index sharding stop scaling, and what's your concrete plan for the mega-monorepo case?
10. How would you detect that the assistant is making engineers *slower* in some subtle way (e.g., over-trusting bad suggestions, review overhead) rather than faster?

## 46. Ideal Answers

1. **Secrets/cross-repo leakage**: Structural isolation (per-repo shard, Section 16) prevents a *different* repo's chunks from ever being retrieved into another repo's context, closing the retrieval-time vector. For the training-time vector, apply secret-scanning at data-curation time (never train on flagged chunks) plus output-time secret-scanning as a second independent layer (Section 25) — defense in depth.

2. **Stale index impact**: A 10-minute-stale index means completions may miss very recently added/renamed symbols — worst case, a suggestion references a function that was just renamed. This is acceptable because the open-file content (always fresh, sent directly in the request) covers the most immediately relevant code, so the index gap only affects lower-stakes cross-file freshness.

3. **Sandbox timeout recovery**: The tool-call result surfaces as an explicit typed error event (`tool_result: {"error": "timeout"}`), not silence or a fabricated success, so the model's next step reasons over a real failure signal instead of inventing output. This is why tool-call results are structured/typed rather than free text.

4. **Routing cheap vs frontier model**: A lightweight classifier (or heuristic: query length, multi-file/agentic keywords, attached-file count) scores query complexity before model selection — simple lookups route to mid-tier, multi-step/refactor tasks route to frontier. Bias toward frontier when classifier confidence is low, since cost optimization is secondary to correctness.

5. **Concurrent conflicting diffs**: Diffs are proposed against a specific base file-hash, and `apply_diff` (Section 9) checks the current file hash matches before applying. If another change landed first, the apply is rejected with a conflict signal (optimistic concurrency control) rather than silently overwriting.

6. **A/B testing without ground truth**: Use implicit behavioral signals as proxy labels — acceptance rate, edit-distance-after-accept, and whether accepted code survived to the next commit without reversion. Combine with periodic blind human preference evaluation as a calibration check, since behavioral signals alone can be gamed by reflexively-accepted but wrong code.

7. **Hosted provider policy change**: The architecture keeps a self-hosted mid-tier model as a standing fallback and keeps the router (answer 4) provider-agnostic, so cutover is a routing-config change, not a rearchitecture. Embargoed/unreleased-title studios already default to self-hosted (Section 25), minimizing exposure.

8. **Compromised/buggy indexer blast radius**: The indexer only has write access to its own pipeline's staging area, not the live-serving vector index — embeddings pass schema/anomaly validation before promotion. Index updates are versioned/snapshotted so a bad batch can be rolled back to the last known-good snapshot (Section 30) instead of requiring a full reindex.

9. **Mega-monorepo shard scaling limit**: Per-repo sharding stops scaling once a repo's chunk count exceeds what one HNSW index can serve within latency/recall targets. The fix is sub-sharding by logical module/directory boundary plus a routing layer that narrows to likely-relevant modules before fanning out ANN queries, rather than querying the whole monorepo index every request.

10. **Detecting the assistant making engineers slower**: Track secondary signals beyond acceptance rate — review/PR cycle time trend and post-merge revert/bug-report rate on AI-touched commits. A rising acceptance rate paired with a rising post-merge defect rate is the key red-flag pattern, indicating reflexive acceptance rather than genuine productivity gain.
