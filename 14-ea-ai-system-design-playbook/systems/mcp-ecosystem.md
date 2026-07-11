# MCP Ecosystem

## 1. Problem Framing & Requirement Gathering

EA is rolling out AI agents across the studio portfolio: a live-ops copilot that tunes matchmaking/economy parameters, a QA agent that triages crash reports, a player-support agent that issues refunds/bans, and a content-pipeline agent that touches DCC tools (Maya, Frostbite asset DB). Each of these agents needs to call *tools* — internal services, data warehouses, game-specific APIs — in a standardized, discoverable, auditable way instead of every team hand-rolling bespoke function-calling glue.

We are asked to design **the MCP (Model Context Protocol) ecosystem**: the server/client architecture that lets any LLM-backed agent at EA discover, authenticate to, and invoke tools exposed by dozens of independently-owned MCP servers (telemetry warehouse, EAC anti-cheat, Origin/EA Play entitlements, Frostbite build system, support ticketing), across multiple studios, tenants, and trust boundaries — safely, at EA scale.

This is fundamentally a **distributed API gateway + service-discovery + capability-authorization system with an LLM client on one end**, not a model-serving problem per se (though model serving of the *agent's own LLM* is a dependency we touch on).

## 2. Functional Requirements

- FR1: MCP clients (agent runtimes) can discover available MCP servers and their tool/resource/prompt manifests at runtime (`tools/list`, `resources/list`, `prompts/list`).
- FR2: MCP servers expose typed tool schemas (JSON Schema) that the agent's LLM uses for function-calling; invocation via `tools/call` with structured args and structured/streamed results.
- FR3: Multi-tenant **tool registry**: studios (DICE, Respawn, BioWare, EA SPORTS) register/version/deprecate their own MCP servers without a central release train.
- FR4: Per-tenant, per-agent, per-tool **authorization** — an agent identity can be scoped to a subset of tools/resources on a server (e.g., QA agent can call `crash_report.query` but not `entitlements.grant`).
- FR5: Auth handshake between agent (MCP client) and MCP server supports both human-delegated (OAuth-style, on behalf of a studio employee) and pure service-to-service (machine identity) flows.
- FR6: Session and context propagation — multi-turn tool use within one agent session must reuse a negotiated capability set without re-auth per call.
- FR7: Tool invocation supports sync (request/response), long-running (job submission + polling/streaming progress via `notifications/progress`), and streaming (SSE) transports.
- FR8: Audit trail — every tool call is logged with agent identity, tenant, tool, args (redacted), result hash, latency, outcome.
- FR9: Registry supports search/ranking of tools by capability (semantic search over tool descriptions) so agents with 500+ available tools can shortlist relevant ones.
- FR10: Kill-switch / policy engine can revoke a tool, a server, or an agent's grant in real time (sub-minute propagation) — e.g., a tool found to leak PII gets pulled instantly.
- FR11: Sandboxed execution option for untrusted/community MCP servers (third-party plugin marketplace inside EA's internal tool store).

## 3. Non-Functional Requirements

| Dimension | Target |
|---|---|
| Discovery latency (`tools/list` cold) | p99 < 300 ms |
| Discovery latency (cached) | p99 < 20 ms |
| Tool invocation overhead (gateway added latency, excluding tool's own work) | p50 < 15 ms, p99 < 60 ms |
| Auth handshake (token mint + capability check) | p99 < 50 ms |
| Availability of registry (control plane) | 99.9% (43m/month downtime budget) |
| Availability of gateway (data plane, tool invocation path) | 99.95% |
| Throughput | 8,000 tool calls/sec sustained fleet-wide, burst 25,000/sec during live-ops incidents |
| Consistency of registry reads | Eventually consistent, ≤ 5s propagation acceptable except kill-switch (≤ 60s hard SLA, target 10s) |
| Consistency of authorization decisions | Strongly consistent — never serve a stale "allow" after a revoke |
| Cost | Gateway + registry infra ≤ $0.35 per 1,000 tool calls fully loaded |
| Multi-tenancy isolation | Hard tenant boundary — no cross-studio data leakage even under gateway bug (defense in depth) |

## 4. Clarifying Questions an Interviewer Would Expect You to Ask

1. Are MCP servers hosted by EA centrally, or are they operated independently by each studio (federated ownership)?
2. Do agents act autonomously (cron/live-ops loop) or always on-behalf-of a logged-in human (support agent acting for a CS rep)?
3. Is there a third-party/community MCP server marketplace, or is everything first-party/internal?
4. What's the blast radius requirement — can one compromised agent identity reach every studio's tools, or must it be scoped per-studio by default?
5. Do we need to support MCP servers running on developer laptops (local stdio transport) for internal tooling, in addition to hosted HTTP/SSE servers?
6. What's the tool call volume mix — read-only queries (telemetry lookups) vs. mutating actions (issue refund, ban player, kick off a build)?
7. Is there a compliance requirement (COPPA for players under 13, GDPR) that constrains what tool results can be logged/cached?
8. Do we need human-in-the-loop approval gates for high-risk tools (e.g., ban/refund) baked into the protocol layer, or is that the agent framework's job?
9. What's the expected number of distinct MCP servers and tools at steady state (tens vs. thousands)?
10. Should the registry support semantic/embedding-based tool search, or is a curated namespace hierarchy sufficient at current scale?

## 5. Assumptions (Explicit)

1. ~1,200 distinct MCP servers registered across EA within 18 months (studio services, shared platform services, a handful of vetted third-party dev-tool integrations).
2. ~18,000 distinct tools total (avg 15 tools/server), growing ~25%/quarter.
3. ~6,000 active agent identities (mix of scheduled live-ops agents, on-demand support-desk agents, developer copilots) at steady state, each with its own scoped credential.
4. Peak fleet-wide tool-call rate driven by live-service incident response (e.g., a title-wide outage triggers dozens of diagnostic agents firing in parallel) plus routine QA/support load.
5. MCP transport is predominantly Streamable HTTP (post-2025 MCP spec) with SSE for streaming; local stdio used only for internal developer-workstation tools, out of scope for the hosted gateway's SLAs.
6. Underlying LLM inference (the agent's own model calls) is a separate system (see "LLM Gateway" chapter) — this chapter's latency budgets are for the tool-call path only, not model token generation.
7. Identity provider is EA's internal OIDC (built on Okta) for human-delegated auth; SPIFFE/SPIFFE-compatible mTLS identities for service-to-service.
8. Registry metadata (tool schemas, ownership, versions) fits comfortably in a document store; it is not itself a big-data problem — the scale challenge is authz decision throughput and audit log volume, not registry storage.
9. "Multi-tenant" = studio-level tenancy as the primary isolation boundary; sub-tenant (team-level) scoping is a stretch goal, not v1.

## 6. Capacity Estimation

**Tool call volume**
- Steady state: 6,000 active agents × avg 2 tool calls/min active-session ≈ 200 calls/sec baseline.
- Incident/live-ops burst: 300 concurrent diagnostic-agent sessions × 5 calls/sec each ≈ 1,500 calls/sec added burst.
- Design target from NFR: 8,000/sec sustained, 25,000/sec burst (10x headroom over current projected peak — matches EA's live-service spikiness, e.g. patch-day traffic).

**Discovery (`tools/list`) load**
- Each agent session performs discovery once per session start + on cache-miss/tool-registry-version-bump.
- ~6,000 agents × ~20 session-starts/day / 86,400s ≈ 1.4 req/sec baseline — trivially small; dominated by burst-on-deploy (registry pushes new manifest → cache invalidation storm): model 6,000 agents re-fetching within a 60s window ⇒ 100 req/sec transient spike. Cache absorbs this (see §11).

**Registry storage**
- Tool manifest: ~2 KB JSON Schema avg (name, description, input/output schema, auth scope tags).
- 18,000 tools × 2 KB = 36 MB raw. With versioning (avg 8 historical versions retained) ≈ 288 MB. Trivial — single document-store cluster, not sharded for volume (sharded for tenancy isolation instead, see §10).
- Server metadata: 1,200 servers × 5 KB (ownership, endpoint, TLS cert ref, rate-limit policy) = 6 MB.

**Audit log volume (the real storage driver)**
- 8,000 calls/sec avg × 86,400 s/day = 691M events/day.
- Per-event record ~1.5 KB (identity, tool, redacted args digest, result hash, timing, outcome) ⇒ ~1.04 TB/day raw.
- Retention: 90 days hot (queryable) + 2 years cold (compliance) ⇒ hot tier ≈ 94 TB, cold tier (compressed columnar, ~5:1) ≈ 152 TB over 2 years.
- At 8,000 events/sec this is squarely a **Kafka → columnar store** problem, not a relational-DB-audit-table problem.

**Authz decision throughput**
- Every tool call requires a capability check: 8,000/sec sustained, 25,000/sec burst.
- Target p99 50ms ⇒ policy engine must be in-memory / edge-cached (OPA-style), not a round-trip to a central RDBMS per call.

**Compute footprint (gateway data plane)**
- Stateless gateway pods, each handling ~1,000 req/sec (JSON schema validation + policy check + proxy) ⇒ 25 pods at burst (25,000/sec), 8 pods at steady state (8,000/sec), with HPA between.
- Each pod: 2 vCPU / 2 GiB — no GPU on this path (GPU lives in the LLM gateway, out of scope here).
- Registry control-plane: 3-node document-store cluster (HA), 3-node policy/OPA cluster, both small (< 16 vCPU total) since read-heavy and cached.

**No model-weight sizing here** — MCP gateway/registry is infra, not a model-serving system; the only "model" involved is the optional embedding model for semantic tool search (§9/§16), a small (~100M param) sentence-embedding model, single small GPU or CPU-served, negligible compared to gateway compute.

## 7. High-Level Architecture

```
                         ┌───────────────────────────────────────────┐
                         │              Agent Runtime                 │
                         │  (Live-ops copilot / QA agent / Support     │
                         │   agent / Dev copilot)  — MCP CLIENT        │
                         └───────────────┬─────────────────────────────┘
                                         │ 1. mint session (OIDC/mTLS)
                                         ▼
                 ┌───────────────────────────────────────────────────────┐
                 │                 MCP GATEWAY (edge, stateless)          │
                 │  - TLS termination, mTLS to internal services          │
                 │  - Rate limiting (token bucket, per-agent/per-tenant)  │
                 │  - AuthN verification, AuthZ policy eval (OPA sidecar) │
                 │  - Request routing to correct MCP server               │
                 │  - Schema validation (JSON Schema on tool args)        │
                 │  - Audit event emission (async, fire-and-forget)       │
                 └───────┬───────────────────────┬─────────────┬──────────┘
                         │                       │             │
             2. discover │            3. authz   │   4. audit  │
                         ▼                       ▼             ▼
         ┌───────────────────────┐   ┌────────────────────┐  ┌───────────────────┐
         │   TOOL REGISTRY        │   │  POLICY ENGINE (OPA)│  │  KAFKA (audit topic)│
         │  (multi-tenant, doc DB)│   │  cached capability   │  │  → stream processor │
         │  - server manifests    │   │  grants, sub-second  │  │  → columnar store    │
         │  - tool schemas/versions│   │  revoke propagation  │  │  (audit/compliance)  │
         │  - semantic search idx │   └──────────┬───────────┘  └───────────────────┘
         └───────────┬────────────┘              │
                      │                            │ pub/sub on revoke
                      │              ┌─────────────▼─────────────┐
                      │              │  IDENTITY & GRANT STORE     │
                      │              │  (agent identities, OIDC    │
                      │              │   token scopes, tenant ACLs)│
                      │              └────────────────────────────┘
                      │
        5. route call ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                     MCP SERVERS (per-domain, per-studio)               │
   │  ┌─────────────┐ ┌───────────────┐ ┌────────────────┐ ┌─────────────┐ │
   │  │ Telemetry    │ │ EAC Anti-Cheat│ │ Entitlements/   │ │ Frostbite   │ │
   │  │ Warehouse MCP│ │ MCP Server     │ │ Support MCP     │ │ Build MCP   │ │
   │  └─────────────┘ └───────────────┘ └────────────────┘ └─────────────┘ │
   │        (each independently deployed/owned/scaled by its team)          │
   └──────────────────────────────────────────────────────────────────────┘
                      │
                      ▼ 6. result (sync/streamed/job-poll)
              back through gateway to agent
```

## 8. Low-Level Components

**MCP Gateway (edge/data plane)**
- Responsibility: single ingress for all agent→tool traffic; authn/z enforcement, schema validation, rate limiting, routing, audit emission.
- Interface: MCP-native (JSON-RPC 2.0 over Streamable HTTP/SSE) `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `notifications/progress`.
- Scaling unit: stateless pod, horizontal, scales on req/sec + p99 latency (HPA).

**Tool Registry (control plane)**
- Responsibility: source of truth for which MCP servers exist, their tool manifests, versions, deprecation status, ownership metadata, semantic search index.
- Interface: internal gRPC/REST admin API for studios to register/update servers (CI/CD-driven, not manual); read API consumed by gateway (cached).
- Scaling unit: read-replica fan-out; write path is low-volume (registrations), doesn't need aggressive scaling — 3-node HA cluster sufficient.

**Policy Engine (OPA-based authz)**
- Responsibility: evaluate "can agent X, acting for tenant Y, call tool Z with these args' sensitivity tags" — Rego policies compiled from tenant/tool ACLs.
- Interface: sidecar to gateway, in-process/local-socket evaluation (no network hop) using a locally cached policy bundle, refreshed via pub/sub on change.
- Scaling unit: scales 1:1 with gateway pods (sidecar pattern) — no independent scaling axis.

**Identity & Grant Store**
- Responsibility: agent identity records, OIDC client credentials, per-agent capability grants (tool-level scopes), human-delegation chains (which employee an agent acts on behalf of).
- Interface: OIDC token introspection endpoint + gRPC grant-lookup API.
- Scaling unit: small HA cluster, read-heavy, cached aggressively at gateway (§11).

**MCP Servers (leaf nodes, federated ownership)**
- Responsibility: expose domain tools (telemetry query, ban issuance, build trigger) with their own internal auth/business logic; register manifest with Tool Registry via CI pipeline.
- Interface: standard MCP server SDK (Python/TS/Go) implementing `tools/list`/`tools/call`; each server owns its own scaling (independent deployment, independent SLOs published to registry).
- Scaling unit: per-team, independent — gateway treats them as opaque upstreams with circuit breakers.

**Audit Pipeline**
- Responsibility: durable, queryable record of every tool invocation for compliance, incident forensics, and anomaly detection feed.
- Interface: Kafka topic `mcp.audit.v1` (producer: gateway, async); consumers: stream processor (redaction, enrichment) → columnar sink (compliance) + real-time anomaly detector (security).
- Scaling unit: Kafka partitions keyed by tenant; consumer group scales with partition count.

**Semantic Tool Search Service**
- Responsibility: given an agent's natural-language task, rank the top-K relevant tools out of 18,000 (avoids stuffing every schema into the LLM's context window).
- Interface: internal REST `POST /tool-search {query, tenant, k}` → ranked tool IDs; backed by a small embedding model + ANN index (see §16).
- Scaling unit: stateless retrieval service + vector index shard, scales on query volume (low — one call per agent-session-start typically, not per tool-call).

## 9. API Design

MCP itself is JSON-RPC 2.0; the gateway also exposes REST admin/control APIs. Versioning: MCP protocol version negotiated in `initialize` (`protocolVersion` field, e.g. `"2025-06-18"`); registry/admin REST APIs use URI versioning (`/v1/...`).

**MCP data-plane (JSON-RPC over Streamable HTTP, `POST /mcp`)**

```jsonc
// initialize
→ { "jsonrpc":"2.0", "id":1, "method":"initialize",
    "params": { "protocolVersion":"2025-06-18",
                "clientInfo": {"name":"liveops-copilot","version":"3.2.0"},
                "capabilities": {"sampling":{}, "roots":{"listChanged":true}} } }
← { "jsonrpc":"2.0", "id":1, "result":
    { "protocolVersion":"2025-06-18",
      "serverInfo": {"name":"telemetry-mcp","version":"1.4.0"},
      "capabilities": {"tools":{"listChanged":true}} } }

// tools/list
→ { "jsonrpc":"2.0", "id":2, "method":"tools/list", "params":{"cursor":null} }
← { "jsonrpc":"2.0", "id":2, "result":
    { "tools":[
        { "name":"telemetry.query_matches",
          "description":"Query ranked-match telemetry for a title/date range",
          "inputSchema": {"type":"object",
             "properties": {"title_id":{"type":"string"},
                             "start":{"type":"string","format":"date-time"},
                             "end":{"type":"string","format":"date-time"}},
             "required":["title_id","start","end"]} }
      ], "nextCursor": null } }

// tools/call
→ { "jsonrpc":"2.0", "id":3, "method":"tools/call",
    "params": { "name":"telemetry.query_matches",
                "arguments": {"title_id":"apex-legends","start":"2026-07-01T00:00:00Z","end":"2026-07-08T00:00:00Z"} } }
← { "jsonrpc":"2.0", "id":3, "result":
    { "content":[{"type":"text","text":"{...json result...}"}], "isError": false } }
```

**Admin/control REST API (studio-facing, `/v1`)**

| Endpoint | Method | Purpose | Auth |
|---|---|---|---|
| `/v1/servers` | POST | Register a new MCP server (manifest + endpoint + owner team) | CI service token, scoped to team namespace |
| `/v1/servers/{server_id}` | PUT | Update manifest/version | CI service token |
| `/v1/servers/{server_id}` | DELETE | Deprecate/remove server | CI service token + approval gate |
| `/v1/servers/{server_id}/tools` | GET | List tools for a server (admin view, unfiltered) | Studio admin OIDC |
| `/v1/grants` | POST | Grant an agent identity access to `{server_id, tool_name[]}` | Tenant admin OIDC |
| `/v1/grants/{grant_id}` | DELETE | Revoke grant (propagates ≤10s via pub/sub) | Tenant admin OIDC |
| `/v1/tool-search` | POST | Semantic search: `{query, tenant, k}` → ranked tool IDs | Agent service token |
| `/v1/audit/query` | POST | Query audit log (filtered by tenant, agent, time range) | Security/compliance role |

Response envelope standardized: `{data, error, request_id}`. Breaking schema changes bump `/v2`; MCP protocol version negotiated independently of REST version.

## 10. Database Design

| Store | Type | Data | Why |
|---|---|---|---|
| Tool Registry | Document DB (MongoDB-style) | Server manifests, tool schemas, version history | Schema-per-tool varies wildly (arbitrary JSON Schema) — poor fit for rigid relational columns; document model matches MCP's own JSON-Schema-native manifests |
| Identity & Grant Store | Relational (Postgres) | Agent identities, grants, tenant ACLs, delegation chains | Grants are relational by nature (agent × tool × tenant, foreign keys, referential integrity matters — a dangling grant to a deleted tool is a security bug) |
| Policy bundles | Object store (S3-compatible) + in-memory cache | Compiled Rego bundles | Read-mostly, versioned artifacts, distributed via CDN-like pull, not a live DB |
| Audit Log (hot) | Columnar OLAP (ClickHouse-style) | 90-day queryable tool-call records | High-cardinality analytical queries (by tenant/agent/tool/time) at 691M events/day — row store can't sustain this |
| Audit Log (cold) | Object storage, Parquet, partitioned by date/tenant | 2-year compliance retention | Cheapest durable storage; rarely queried, batch access only |
| Semantic search index | Vector index (see §16) | Tool description embeddings | ANN lookup, not a relational concern |

**Partitioning/sharding keys:**
- Registry: sharded by `tenant_id` (studio) — enforces the hard tenant-isolation NFR at the storage layer, not just app logic.
- Grant store: partitioned by `tenant_id`, secondary index on `agent_id`.
- Audit columnar store: partitioned by `(tenant_id, date)` — matches both compliance query patterns ("show me tenant X's calls last week") and retention/deletion (drop old date partitions cheaply).
- Kafka audit topic: keyed by `tenant_id` to preserve per-tenant ordering and enable per-tenant consumer scaling/backpressure isolation.

## 11. Caching

| Cached item | Cache | TTL/invalidation | Strategy |
|---|---|---|---|
| Tool manifests (`tools/list` results) | Gateway-local in-memory + Redis L2 | TTL 5 min, active invalidation via registry pub/sub on version bump | Cache-aside |
| Compiled policy bundles | Gateway sidecar local memory | Invalidated via pub/sub push on grant change (not polling) — meets ≤10s revoke SLA | Write-through push (registry pushes to subscribers, not pull-based) |
| Agent OIDC token introspection | Redis, TTL = token lifetime (max 15 min) | Natural expiry; explicit bust on identity disable | Cache-aside |
| Semantic tool-search results | Redis, keyed by `(tenant, normalized_query_hash)` | TTL 1 hr | Cache-aside — search is not safety-critical, staleness tolerable |
| Tool-call *results* | **Not cached at gateway by default** — most tools are non-idempotent or return live data (telemetry queries, ban status) | N/A | Individual MCP servers may cache internally (their concern); gateway treats results as opaque |

Why cache-aside dominates: registry/grant reads vastly outnumber writes (thousands of reads/sec vs. registrations/grants in the tens/hour), and staleness tolerance differs sharply by item — hence per-item TTL/invalidation tuning rather than one blanket policy. The one deliberate exception is policy bundles, where push-invalidation is required to hit the sub-10s kill-switch SLA that a pull-TTL model can't guarantee.

## 12. Queues & Async Processing

| Queue | Payload | Delivery semantics | DLQ handling |
|---|---|---|---|
| `mcp.audit.v1` (Kafka) | Tool-call audit events | At-least-once (producer retries on gateway); consumers idempotent via `(agent_id, call_id)` dedup key | Poison messages (malformed audit payload) → `mcp.audit.dlq`, alert if DLQ depth > 100/5min |
| `mcp.grant-revoke.v1` (Kafka, low-latency topic) | Grant revocation events | At-least-once, but consumers apply revokes idempotently (revoke is a set-removal, naturally idempotent) | N/A — revoke messages that fail to apply trigger immediate PagerDuty (security-critical) |
| `mcp.longrunning-jobs` (SQS-style) | Long-running tool invocations (e.g., "kick off Frostbite build") that exceed sync timeout | At-least-once; job has server-side idempotency key supplied by client to avoid duplicate builds | After 3 delivery attempts → DLQ, surfaced to owning MCP server team's on-call |
| `mcp.registry-sync.v1` | Registry manifest change events, fan out to gateway caches | At-least-once, consumers treat as "invalidate then re-fetch" (safe to over-invalidate) | Not critical-path; failed sync just means TTL-based cache eventually catches up |

Exactly-once is explicitly **not** attempted for audit/telemetry-style events (cost/complexity not justified — idempotent consumers achieve effectively-once outcomes cheaply). The one place we'd consider exactly-once semantics is the long-running job queue for mutating actions (ban issuance) — mitigated via client-supplied idempotency keys rather than transactional queue machinery.

## 13. Streaming & Event-Driven Architecture

**Topics**

| Topic | Producer | Consumers | Schema (Avro/Protobuf) |
|---|---|---|---|
| `mcp.audit.v1` | Gateway | Compliance sink, anomaly detector, usage-analytics | `{event_id, ts, tenant_id, agent_id, server_id, tool_name, args_digest, latency_ms, outcome, error_code?}` |
| `mcp.grant-revoke.v1` | Grant Store (on admin action) | All gateway sidecars, Policy Engine | `{grant_id, tenant_id, agent_id, tool_scope[], action: GRANT\|REVOKE, ts}` |
| `mcp.registry-sync.v1` | Tool Registry | Gateway cache layer, semantic-search indexer | `{server_id, version, change_type: CREATE\|UPDATE\|DEPRECATE, manifest_ref, ts}` |
| `mcp.tool-health.v1` | Gateway (circuit breaker state) | Ops dashboards, autoscaler signal | `{server_id, tool_name, error_rate_1m, p99_latency_ms, breaker_state}` |

**Consumer groups**: each downstream concern (compliance, security anomaly detection, analytics) gets its own consumer group off `mcp.audit.v1` so a slow consumer (e.g., a batch analytics job) never backpressures the security-critical revoke path, which lives on a separate low-latency topic entirely.

## 14. Model Serving

The MCP ecosystem itself serves no large model on the critical tool-call path — it's a protocol/gateway layer. The two models involved:

1. **Semantic tool-search embedding model** (§8, §16): small (~100M param) sentence-embedding model (e.g., a distilled BGE/E5-class model), served via a lightweight inference server (Triton or TorchServe), CPU-servable at this size, batched with a 10ms micro-batch window since query volume is low (session-start only, not per-call).
2. **The agent's own LLM** (the thing deciding which tool to call) is out of scope for this chapter — it's served by EA's central LLM Gateway (separate system doc) and merely *acts as* the MCP client. We note the interface: the agent runtime receives `tools/list` output, injects tool schemas into the LLM's context/function-calling API, gets back a `tools/call` request to forward.

No GPU fleet is provisioned specifically for MCP infra; the embedding model above runs on shared CPU inference capacity (a handful of pods), explicitly *not* co-located with the gateway's latency-critical path.

## 15. Feature Store

**N/A for the MCP gateway/registry itself** — there is no ML feature engineering happening in the tool-call routing/authz path; decisions are policy-rule-based (OPA), not model-scored. The adjacent anomaly-detection consumer (§13, "detect an agent calling tools outside its normal pattern") *does* need features (rolling call-rate per agent, tool-diversity entropy, off-hours flag), and for that narrow slice:
- **Online**: last-15-min/1hr rolling aggregates per `(agent_id, tool_name)`, served from a low-latency KV store (Redis) updated by the stream processor consuming `mcp.audit.v1`.
- **Offline**: same features backfilled from the columnar audit store for model training/backtesting.
- **Point-in-time correctness**: anomaly-detection training set must join audit events with the grant-state *as of that event's timestamp* (not current grants) — achieved by storing grant-version snapshots alongside each audit event's `args_digest` context, avoiding label leakage from later revokes.

## 16. Vector Database

**Applicable, narrowly** — used only for the semantic tool-search service (§8), not for any core protocol function.
- **Index**: HNSW (via a managed vector DB or pgvector-with-HNSW) over ~18,000 tool-description embeddings — small enough that IVF-PQ's memory savings aren't needed; HNSW's better recall at this scale (tens of thousands of vectors) is worth the modest extra memory (~18k × 768-dim float32 ≈ 55 MB, trivial).
- **ANN choice justification**: at 18,000 vectors, brute-force cosine search is even arguably viable (< 5ms), but HNSW is chosen for headroom as tool count grows 25%/quarter — re-evaluate IVF-PQ only if this crosses ~1M vectors (unlikely within the planning horizon).
- **Sharding**: single shard, replicated for HA — no partitioning needed at this scale; re-shard-by-tenant only if per-tenant tool catalogs need hard isolation for search (currently search operates over the tenant's *visible* tool subset via a metadata filter, not a physically separate index).

## 17. Embedding Pipelines

**Applicable, narrowly** — same scope as §16.
- **Pipeline**: on tool registration/update (`mcp.registry-sync.v1` event) → embedding worker consumes event → generates embedding from `{tool_name, description, param_names}` concatenated text → upserts into vector index with metadata `{tool_id, server_id, tenant_id, version}`.
- **Model**: same small sentence-embedding model as §14; batched embedding generation (registration events are low-volume, no real-time pressure — batch window of a few seconds is fine).
- **Re-embedding trigger**: on any manifest description change (semantic drift in what the tool does) — versioned so search can fall back to prior embedding if a re-embed job fails, rather than dropping the tool from search results.

## 18. Inference Pipelines (End-to-End Request Lifecycle)

Here "inference" = the full agent tool-call round trip (the closest analog to a model-serving "inference pipeline" in this system).

```
Agent decides to call a tool
        │
        ▼
[1] Agent runtime → MCP Gateway: tools/call (JSON-RPC, mTLS/OIDC bearer)      ~1-2ms network
        │
        ▼
[2] Gateway: verify OIDC token / mTLS cert, extract agent identity             ~3-5ms
        │
        ▼
[3] Gateway → Policy sidecar (local): authz check                              ~1-3ms
        │   (cached grant + policy bundle, no network hop)
        ▼
[4] Gateway: JSON Schema validation of tool arguments                          ~1-2ms
        │
        ▼
[5] Gateway: rate-limit check (token bucket, local + Redis-backed counter)      ~2-5ms
        │
        ▼
[6] Gateway: circuit-breaker check for target MCP server health                ~<1ms
        │
        ▼
[7] Gateway → MCP Server: proxied tools/call over mTLS                         ~network + server's own work
        │                                                                       (5ms-2s depending on tool;
        │                                                                        e.g. telemetry query ~150ms,
        │                                                                        build trigger ~async job)
        ▼
[8] MCP Server: executes tool logic (its own DB/service calls), returns result
        │
        ▼
[9] Gateway: emit async audit event to Kafka (fire-and-forget, doesn't block response)
        │
        ▼
[10] Gateway → Agent: tools/call result (sync) or notifications/progress (streamed)
        │
        ▼
Agent's LLM consumes result, decides next action (may loop back to [1])
```

**Gateway-added overhead budget**: steps 2–6 + 9 (excluding upstream tool execution and step 7's network) must total p50 < 15ms / p99 < 60ms per NFR — this is the number this system is actually accountable for; upstream tool latency is each MCP server team's own SLO.

## 19. Training Pipelines

No model is trained as part of the core MCP protocol path. Two adjacent, small training pipelines exist:

1. **Tool-search embedding model fine-tuning** (infrequent): base sentence-embedding model optionally fine-tuned on EA-internal (query → correct tool) pairs mined from agent session logs where a human corrected the agent's tool choice. Offline batch job, single-GPU fine-tune, run quarterly, not a distributed-training problem at this data size (thousands of labeled pairs).
2. **Anomaly-detection model** (§15): gradient-boosted classifier (or simpler, an isolation-forest) trained on rolling-window call-pattern features to flag anomalous agent behavior (credential misuse, scope creep). Trained offline on the columnar audit store, CPU-only, retrained on the cadence in §20.

Neither requires distributed/multi-GPU training — both are small-data, small-model problems; the emphasis in this system is data *plumbing* (getting clean labeled examples out of audit logs), not training infra.

## 20. Retraining Strategy

| Model | Cadence | Trigger |
|---|---|---|
| Tool-search embedding fine-tune | Quarterly, or ad hoc | Tool catalog grows >30% since last fine-tune, or search click-through/override rate degrades below threshold (see §21) |
| Anomaly-detection model | Weekly scheduled retrain | Also triggered on: (a) false-positive rate reported by security team exceeds 5% of flagged events in a week, (b) a new tool category is onboarded (distribution shift in normal call patterns) |
| Policy bundles (not ML, but analogous "retrain") | On every grant/ACL change | Event-driven, not scheduled — compiled and pushed within seconds |

## 21. Drift Detection

| Drift type | What's monitored | Metric | Threshold |
|---|---|---|---|
| Data drift (tool-search) | Distribution of incoming natural-language queries vs. training query distribution | Embedding-space KL/population stability index on query embeddings, weekly | PSI > 0.2 → flag for review |
| Concept drift (tool-search relevance) | Agent's actual tool selection after search vs. top-K suggested | Override rate (agent picks a tool outside top-5 suggested) | > 15% override rate over 7-day window → trigger retrain |
| Concept drift (anomaly detector) | False positive/negative rate reported by security analysts | Weekly analyst-labeled sample precision/recall | Precision < 0.85 or recall < 0.90 → retrain + threshold review |
| "Protocol drift" (MCP-specific, not ML) | New MCP server versions deviating from expected schema conventions (breaking manifest changes slipping through) | Schema-validation failure rate on `tools/list` ingestion | > 1% of registered servers failing validation in a week → registry team review |

This system's "drift" is as much about *tool ecosystem* drift (schemas, ownership, deprecations) as classic ML data drift — worth calling out explicitly in an interview since it's a distinguishing feature of an MCP-shaped system vs. a typical model-serving one.

## 22. Monitoring

| Category | Metrics |
|---|---|
| Infra | Gateway pod CPU/mem, p50/p99 latency per stage (§18 breakdown), request rate, error rate by MCP server, circuit-breaker state transitions, Kafka consumer lag |
| Registry/control-plane | Registration rate, manifest validation failure rate, cache hit ratio (registry/policy), pub/sub propagation latency (grant revoke → gateway applied) |
| Model quality (search) | Recall@5 on labeled eval set, override rate, embedding staleness (days since last re-embed vs. manifest update) |
| Security | Authz denial rate per agent/tenant (spike = probing behavior), anomaly-detector flag rate, credential age (stale/unrotated service tokens) |
| Business | Tool-call volume by studio/tenant (cost allocation), top-N most-used tools (registry prioritization signal), agent adoption (active agents/week), incident MTTR when agents assist live-ops |

## 23. Alerting

| Alert | Condition | Routing |
|---|---|---|
| Grant-revoke propagation SLA breach | Revoke not applied fleet-wide within 10s | PagerDuty, Security on-call, P1 |
| Gateway p99 latency breach | p99 > 60ms for 5 consecutive min | PagerDuty, Platform on-call, P2 |
| Tool-call error rate spike | Any single tool's error rate > 10% over 5 min | Slack to owning MCP server team, P3; escalate to P2 if sustained 30 min |
| Audit pipeline lag | Kafka consumer lag on `mcp.audit.v1` > 5 min of backlog | PagerDuty, Platform on-call, P2 (compliance risk) |
| Anomaly-detector high-confidence flag | Agent flagged with score > 0.95 (likely credential misuse) | PagerDuty, Security on-call, P1, auto-suspend agent pending review |
| Registry validation failure spike | > 1% of registrations failing schema validation | Slack, Registry team, P3 |
| Semantic search relevance degradation | Override rate > 15% (§21) | Slack, ML platform team, P4 (non-urgent, quality issue) |

On-call routing follows tenant/team ownership where the fault is domain-specific (a single MCP server's errors go to that server's owning team) and central platform on-call for anything cross-cutting (gateway, registry, policy engine, audit pipeline).

## 24. Logging

- **Structured logging** (JSON) at every gateway stage: `request_id`, `agent_id`, `tenant_id`, `tool_name`, `server_id`, `stage_latencies{}`, `outcome`, `trace_id` (for correlation, §35).
- **PII handling**: tool call *arguments* frequently contain player IDs, emails (support/refund tools), payment references. Gateway applies a **redaction policy per tool**, tagged in the tool's manifest (`sensitivity: pii|financial|none`) — PII-tagged fields are hashed/tokenized before logging (`args_digest` in audit schema, §13), never logged in plaintext. Full plaintext args are retained only transiently (in-flight, not persisted) for the tool's own execution.
- **Retention**: application/debug logs 30 days hot (ELK/OpenSearch-style); audit logs per §10 (90 days hot columnar, 2 years cold Parquet) — audit retention is a compliance requirement (GDPR/COPPA-adjacent — need to show what an agent did with a minor player's data, without retaining the raw PII itself).
- **Right-to-erasure**: since audit records use tokenized `args_digest` rather than raw PII, erasure requests are handled by purging the token-mapping table (identity vault) rather than rewriting 2 years of columnar audit history — a deliberate design choice to keep compliance tractable at this log volume.

## 25. Security

**Threat model specific to this system:**
1. **Over-privileged agent** — an agent granted broad tool access gets prompt-injected (via untrusted content in a tool result, e.g., a player-submitted bug report) into calling a destructive tool (issue mass refunds). Mitigation: least-privilege per-agent grants (FR4), human-in-the-loop approval gate flag on high-risk tools enforced at the gateway (not just app-level), output/argument schema validation limiting what an injected instruction can actually construct.
2. **Malicious/compromised MCP server** (especially third-party marketplace ones, FR11) — a server could return crafted responses designed to manipulate the agent's LLM (indirect prompt injection) or exfiltrate data via tool-call arguments. Mitigation: sandboxed execution tier for untrusted servers (network-isolated, capability-scoped even further), response size/content-type allowlisting, no untrusted server ever gets a direct grant to write-capable tools.
3. **Registry poisoning** — a compromised CI credential registers a malicious tool manifest impersonating a trusted namespace. Mitigation: namespace ownership verification (signed commits from the owning team's repo, tied to their SPIFFE identity), manifest diffing/approval gate for any change to a tool's `sensitivity`/auth-scope tags.
4. **Credential/token leakage** — long-lived service tokens for agent identities being a juicy target. Mitigation: short-lived OIDC tokens (15 min max), mTLS with SPIFFE workload identity rotated automatically, no static API keys for service-to-service.
5. **Cross-tenant data leakage via gateway bug** — a routing bug sends studio A's telemetry-tool result to studio B's agent. Mitigation: tenant_id threaded through every layer as a hard partition key (not just a filter), row-level checks at the MCP server too (defense in depth — gateway isn't the only line of defense), chaos/fuzz testing specifically for cross-tenant leakage.

**Encryption**: TLS 1.3 everywhere in transit (mTLS internal, TLS+OIDC bearer external-facing where applicable); audit log columnar store encrypted at rest (KMS-managed keys, per-tenant key where feasible for stronger isolation).

## 26. Authentication

- **Service-to-service** (agent runtime ↔ gateway ↔ MCP server): mTLS with SPIFFE/SPIRE-issued workload identities (X.509-SVID), short-lived (hours), automatically rotated. Gateway verifies the SPIFFE ID matches an expected agent identity registered in the Identity Store.
- **Human-delegated** (support agent acting on behalf of a CS rep, dev copilot acting on behalf of an engineer): OIDC authorization-code flow against EA's internal Okta-backed IdP; the resulting token includes a `delegation_chain` claim (`{human_subject, agent_id, scope}`) so the gateway/audit trail always knows *who* ultimately authorized an action, not just which agent executed it — critical for tools like refund/ban issuance where accountability must trace to a human.
- **Third-party MCP servers** (marketplace, FR11): separate, more restrictive OAuth client-credentials flow with mandatory scopes, sandboxed network tier (§25), no delegation-chain trust (never inherits a human's broader permissions).
- **Session/token propagation**: within one MCP session, the negotiated capability set (from `initialize` handshake) is cached against the session ID so subsequent `tools/call` within the same session skip the full authz re-derivation (though the *token itself* still gets verified per-call) — this is what keeps multi-turn tool use within the 15ms/60ms overhead budget (§18).

## 27. Rate Limiting

- **Algorithm**: token bucket, per-`(agent_id, tool_name)` and a coarser per-`(tenant_id)` bucket — token bucket chosen over sliding-window-log for O(1) memory per key at 18,000-tool × 6,000-agent cardinality, and because it naturally supports controlled bursts (a live-ops incident legitimately needing a burst of diagnostic calls) rather than hard-clipping at a fixed window boundary.
- **Limits** (defaults, overridable per-tool via manifest):
  - Read-only/query tools: 60 req/min per agent, burst 120.
  - Mutating tools (refund, ban, build-trigger): 10 req/min per agent, burst 20 — tighter because blast radius is higher.
  - Per-tenant aggregate ceiling: 2,000 req/min, to protect one studio's traffic spike from starving others (multi-tenant fairness — also acts as a coarse cost-control lever, §29).
- **Enforcement point**: gateway, local token-bucket counters backed by Redis for cross-pod consistency (approximate — slight overcounting tolerated in exchange for avoiding a synchronous Redis round-trip on every single call; a local-first-check-then-async-sync pattern).
- **Response on limit**: standard `429`-equivalent MCP error object with `retry_after_ms`, surfaced to the agent so its LLM can reason about backoff rather than the client silently failing.

## 28. Autoscaling

- **Gateway pods**: Kubernetes HPA on custom metric `requests_per_second_per_pod` (target 800/pod, headroom under the 1,000/pod design capacity from §6) plus a secondary trigger on p99 latency (scale out if p99 > 40ms for 2 min, ahead of the 60ms SLA breach). Min 8 replicas (steady state), max 30 (covers 25,000/sec burst design target).
- **KEDA** used for the audit-pipeline stream processors — scaled on Kafka consumer-group lag (`mcp.audit.v1` lag > 10,000 messages triggers scale-out), since this is an event-driven workload better matched to lag-based scaling than CPU-based HPA.
- **Policy sidecars**: scale 1:1 with gateway pods (no independent policy), so no separate autoscaler needed there.
- **VPA**: applied in recommendation-only mode to the Tool Registry and Identity Store clusters (low-volatility, mostly-read workloads) — used to right-size requests/limits periodically rather than live-resize, to avoid pod restarts on a control-plane component.
- **MCP servers themselves**: each owning team's own autoscaling policy — the gateway's circuit breaker (§8, §18) is the mechanism that protects the rest of the system when an individual MCP server under-scales or falls over, not a shared autoscaler.

## 29. Cost Optimization

- **Spot/preemptible instances** for gateway pods (stateless, fast-drain-and-reschedule tolerant) — 60-70% cost reduction vs. on-demand for the largest compute line item (25 pods at burst).
- **Caching** (§11) collapses what would otherwise be a registry/policy read per tool-call into a handful of reads per cache-TTL window — directly cuts control-plane compute and cross-AZ network charges.
- **Audit log tiering**: 90-day hot columnar (expensive, fast) → 2-year cold Parquet-on-object-storage (cheap, slow) is itself a cost lever — keeping 2 years fully in the hot tier would roughly 8x the compliance-storage bill for no query-pattern benefit (compliance queries are rare, batch, latency-insensitive).
- **Batching the embedding model** (§14): batching tool-search embedding requests into a 10ms window means a couple of small CPU pods cover the workload instead of provisioning for worst-case single-request latency.
- **Per-tenant rate ceilings** (§27) double as a cost-control lever — prevents one studio's misconfigured agent loop from silently inflating the shared infra bill (chargeback model attributes gateway compute cost proportionally to tenant call volume, visible in the business-metrics dashboard, §22).
- **Tool-call result NOT cached by default** (§11) is itself a conscious cost/correctness tradeoff — caching would save compute but risks serving stale ban-status/entitlement data, judged not worth the small savings given the low absolute cost of the gateway's own compute relative to what a stale-data incident would cost.

## 30. Operational Concerns (Deployment, Reliability, Infra)

At SDE2 scope, treat this as a checklist rather than a design exercise: **backups** (automated snapshots of the model registry, feature store, and any stateful service, with a tested restore path), **rollback** (every deploy must be revertible to the last-known-good version — the model registry and CI/CD pipeline should make this a one-command operation), **canary/blue-green rollout** (shift a small percentage of traffic first, watch error rate and key business/model metrics, then ramp), and **basic observability** (dashboards + alerts on latency, error rate, and the top 2-3 model-quality signals, wired to on-call). Kubernetes/Terraform specifics and multi-region active-active topology are Staff/Principal-level infra-architecture concerns — worth knowing they exist, not worth rehearsing the manifests.

## 38. Why This Architecture

- A **stateless gateway + externalized policy engine** cleanly separates the two things that actually need independent scaling curves: raw request throughput (gateway) vs. authorization-decision correctness/freshness (policy), letting each be tuned/scaled without coupling.
- **Federated MCP server ownership** matches EA's actual organizational reality (dozens of semi-autonomous studios) — a centrally-owned monolithic tool-serving layer would recreate the exact bottleneck (one team gatekeeping every studio's tool additions) that MCP as a protocol exists to avoid.
- **Push-based revoke propagation** (rather than short-TTL polling) is the one place we deliberately spend extra engineering effort (pub/sub fan-out) because it's the one NFR (sub-10s kill-switch) that a purely cache-TTL design structurally cannot guarantee.
- **Document store for registry, relational for grants** reflects that tool schemas are naturally polymorphic JSON while grants are naturally relational (agent × tool × tenant) — picking one database paradigm for both would force an awkward compromise in one direction or the other.

## 39. Alternative Architectures

| Alternative | Description | Why rejected (or when preferred) |
|---|---|---|
| Centralized monolithic tool-gateway (no MCP protocol, bespoke per-agent function-calling glue) | Every agent framework directly integrates each tool's API | Rejected: O(agents × tools) integration cost, no standardized discovery/auth, doesn't scale past a handful of agents/tools — this is exactly the pre-MCP status quo EA is moving away from |
| Fully centralized single MCP server exposing all 18,000 tools | One team owns and operates every tool's MCP-facing wrapper | Rejected: recreates a central bottleneck/single point of ownership; would be preferred only in a much smaller org (single studio, < 50 tools) where coordination overhead of federation exceeds its benefit |
| Synchronous-only tool invocation (no job-queue for long-running tools) | Simpler, no async job infra | Rejected: build-trigger/batch-analysis tools routinely exceed reasonable HTTP timeout windows; would be preferred if the tool catalog were 100% read-only/fast-path, which it isn't (mutating/long tools are a real minority but non-trivial) |
| No policy engine — authz baked into each MCP server individually | Each server enforces its own access control | Rejected as the *sole* mechanism (still kept as defense-in-depth per §25) because it can't give a single point of sub-10s fleet-wide revocation, and produces inconsistent authz semantics across 1,200 independently-coded servers; acceptable as a *secondary* layer, not primary |

## 40. Tradeoffs

| Decision | Benefit | Cost |
|---|---|---|
| Federated MCP server ownership | Scales with org, no central bottleneck | Inconsistent quality/reliability across servers; gateway must be defensive (circuit breakers) against every upstream |
| Push-based revoke propagation | Meets sub-10s security SLA | Extra infra (dedicated pub/sub path) vs. simpler TTL-polling |
| Cache-aside for most registry/grant reads | Low latency, low control-plane load | Small window of staleness tolerated everywhere except revoke |
| Document DB for registry | Flexible schema matches arbitrary tool manifests | Weaker referential integrity than relational — must guard against dangling references to deleted servers/tools in app logic |
| Token-bucket rate limiting, Redis-backed approximate counts | O(1) memory, tolerates bursts, avoids sync Redis round-trip per call | Slight overcounting possible across pods — a marginally generous limit in edge cases, judged acceptable |
| Not caching tool-call results | Avoids stale ban/entitlement data | Forgoes compute savings that caching would offer for read-heavy tools |
| Sandboxed tier for third-party servers | Contains blast radius of untrusted code | Extra operational complexity (a second execution environment to maintain) |

## 41. Failure Modes

1. **A popular MCP server (e.g., telemetry) goes down** → circuit breaker trips after error-rate threshold, gateway fails fast with a structured MCP error rather than hanging agent sessions; agents' LLMs can reason about the `isError` response and retry/fallback. Mitigation validated via chaos testing (kill telemetry-MCP pods, verify gateway doesn't cascade-fail).
2. **Policy bundle push fails silently to a subset of gateway pods** (network partition during pub/sub fan-out) → stale policy served from some pods, potential over- or under-permissive authz. Mitigation: policy bundle version is included in every authz decision's audit log; a background reconciliation job periodically diffs each pod's loaded bundle version against the source of truth and force-refreshes laggards, alerting if any pod is >2 versions behind.
3. **Kafka partition unavailability drops audit events** → tool calls still succeed (audit emission is async/non-blocking, correctly decoupled from the request path) but compliance/security visibility has a gap. Mitigation: gateway buffers audit events locally (bounded, with overflow-to-disk) during producer-side Kafka unavailability, replays on recovery; DLQ/lag alerting (§12, §23) catches sustained issues.
4. **Semantic tool-search returns wrong/irrelevant tools** → agent's LLM either fails to find the right tool or (worse) selects a superficially-similar but wrong tool (e.g., a "ban" tool instead of "warn" tool with similar description). Mitigation: manifest descriptions require a human-reviewed clarity bar for high-risk tools at registration time; high-risk tools get mandatory disambiguation confirmation step in the agent framework, not solely relying on search ranking.
5. **Registry primary region outage** → writes (new registrations/grants) blocked, but reads continue from replicas (stale by replication lag) — existing agents keep functioning with their already-cached grants; new agent onboarding blocked until failover completes (~15 min RTO, §30).
6. **A compromised third-party MCP server exfiltrates data via tool-call arguments echoed back in results** → sandboxing (§25) limits network egress from that server's execution tier; response schema validation at the gateway catches obviously malformed/oversized responses, though a sophisticated exfiltration disguised as legitimate tool output is the hardest case — mitigated primarily by *not* granting third-party servers access to sensitive-tagged tools in the first place (policy-level prevention over detection).

## 42. Scaling Bottlenecks

- **At 10x scale (80,000 calls/sec, 12,000 servers, 60,000 agents)**: 
  - Redis-backed rate-limit counters become a hotspot — per-key contention on high-traffic tenants; mitigation: shard Redis by tenant, move to a local-approximate-then-periodic-reconcile model more aggressively.
  - Audit Kafka topic partition count needs re-tuning (currently sized for 8,000/sec baseline) — at 80,000/sec, partition count and consumer parallelism must scale roughly proportionally; columnar sink ingest rate becomes the tighter constraint than Kafka itself.
  - Semantic tool-search index (18,000 → potentially 180,000 tool descriptions) — this is where HNSW's memory footprint (§16) starts to matter and a re-evaluation of IVF-PQ becomes warranted.
- **At 100x scale (800,000 calls/sec)**: this exceeds any single-region gateway fleet's practical pod count under one Kubernetes cluster's control-plane limits — would require gateway sharding by tenant-hash across multiple independent clusters, not just more HPA replicas in one cluster; the Identity & Grant Store (single-writer Postgres, §31) becomes the hard architectural bottleneck, requiring a move to a multi-writer/partitioned grant-store design (e.g., per-tenant grant-store shards) well before 100x is reached.

## 43. Latency Bottlenecks

| Stage | p50 | p99 | Notes |
|---|---|---|---|
| Network (agent → gateway) | 1ms | 3ms | Same-region assumption |
| Auth verification (mTLS/OIDC) | 2ms | 5ms | Cached token introspection |
| Authz check (local policy sidecar) | 1ms | 3ms | No network hop by design |
| Schema validation | 1ms | 2ms | JSON Schema, small payloads |
| Rate-limit check | 1ms | 4ms | Local + async Redis reconcile |
| **Gateway overhead subtotal** | **~6ms** | **~17ms** | Within the 15/60ms budget (§6, §18) with margin |
| Network (gateway → MCP server) | 1-3ms | 5-10ms | Depends on cross-AZ/region routing |
| MCP server's own execution | 5ms (fast query) – 2000ms (complex build-trigger ack) | Highly variable | **The dominant cost for most tool calls** — outside this system's direct control, only its circuit-breaker/timeout policy |
| Async audit emission | 0 (non-blocking) | 0 (non-blocking) | Doesn't sit in critical path |

**Bottom line**: the gateway itself is nowhere near the latency bottleneck at design targets — the MCP server's own execution time dominates end-to-end latency in nearly every real tool call. The system's job is to keep its *own* overhead negligible and use circuit breakers/timeouts so a slow upstream degrades gracefully rather than the gateway becoming the bottleneck by proxy (e.g., thread/connection exhaustion waiting on a slow server).

## 44. Cost Bottlenecks

- **Audit log storage/compute** (hot columnar tier, §6/§10) is the single largest recurring infra line item at steady state — 691M events/day, 94 TB hot retention — dwarfing gateway compute cost.
- **Gateway compute** is the second driver, mostly proportional to call volume, largely mitigated by spot instances (§29) and caching reducing control-plane round-trips per call.
- **Cross-region data transfer** for tool calls that must route to a tool hosted in a different region than the requesting agent (§31) — an underappreciated cost line, worth explicitly monitoring per-tenant to catch mis-routed traffic patterns (e.g., an EU agent routinely calling a US-only tool that should have a regional equivalent).
- **NOT a major cost driver**: the registry/identity control-plane databases (tiny relative to audit volume), and the semantic-search embedding model (small model, low query volume, batched).

## 45. Interview Follow-Up Questions

1. How would you handle a tool that needs to call back into the agent mid-execution (human-in-the-loop approval, or the tool itself needing an LLM completion) — does MCP's sampling capability change your architecture?
2. Your revoke-propagation SLA is 10 seconds — walk through exactly what happens in the worst case where a gateway pod is netsplit from the pub/sub fan-out for 30 seconds.
3. How do you prevent a single studio's misbehaving agent (bug, not malice) from degrading service for every other tenant sharing the gateway fleet?
4. If you had to support 100,000 tools instead of 18,000, what's the first component that breaks and what would you change?
5. How would semantic tool search interact with authorization — could search leak the *existence* of a tool a tenant isn't authorized to call, even if it can't invoke it?
6. Walk me through what changes if an MCP server needs to stream partial results over a multi-minute job (e.g., a long analysis) rather than a simple job-poll model.
7. How do you test/validate a third-party MCP server before admitting it to the marketplace, beyond the sandbox runtime isolation?
8. What's your strategy if two different studios register tools with confusingly similar names/descriptions, and the agent picks the wrong one?
9. How would you extend this design to support sub-tenant (team-level, not just studio-level) authorization scoping without a full re-architecture?
10. What telemetry would tell you the tool registry's semantic search is actively causing harm (agents doing wrong things) rather than just being mediocre?

## 46. Ideal Answers

1. **Sampling/callback handling**: MCP's `sampling/createMessage` lets a server request the *client's* LLM do a completion mid-tool-execution, so the gateway can't treat `tools/call` as strictly request/response-terminal — it must keep the session open and route the callback back through it. Apply the same per-tool authz scoping and rate limit to sampling requests rather than giving the server an unbounded channel.

2. **30s netsplit worst case**: that gateway pod can serve stale grants/policy for up to 30s beyond the 10s SLA, a real breach. The reconciliation job (§41.2) bounds this by proactively pulling on version-lag, and high-risk tools can carry a "must revalidate live" flag forcing a synchronous grant-store check.

3. **Tenant isolation under misbehavior**: the per-tenant rate-limit ceiling (§27) is the first line of defense regardless of per-agent limits. Fair-queuing/priority classes on gateway pods stop one tenant's queue from starving others, and per-tenant usage dashboards (§22) catch slow-building misbehavior before it hits the hard limit.

4. **100,000 tools**: the semantic-search vector index (HNSW) likely stays fine memory-wise at that scale, but the more likely break point is the gateway's registry read cache size. Fix is a smarter partial-cache/LRU strategy keyed by which tools a tenant/agent-class actually uses, not full-catalog caching.

5. **Search leaking tool existence**: yes — a tenant seeing a sensitive tool name in search results (even if denied on `tools/call`) is a real information-disclosure risk. Fix is applying the same authz filter to search as to `tools/list`, pre/post-filtering the vector query rather than relying on call-time denial alone.

6. **Long streaming job**: shift from job-submit+poll to `notifications/progress` over the same session's SSE stream, keeping a long-lived connection instead of a fire-and-forget queue. This needs its own timeout/keepalive tuning distinct from the fast-path sync budget (§18) so a healthy multi-minute job doesn't trip the same circuit breaker as a hung server.

7. **Third-party vetting**: beyond sandbox runtime isolation, add static analysis of the manifest for suspicious scope requests and a manual security review gate before marketplace listing. New servers get a probationary period at reduced rate limits with human-approval on their first N calls, plus continuous anomaly monitoring (§15) at a lower alert threshold than first-party.

8. **Name collision confusion**: namespace tools by `{studio}.{domain}.{action}` so collisions are structurally prevented at the identifier level, and surface the owning studio/team prominently to the agent's LLM as a disambiguating signal. Track override/correction rate (§21) as the metric that catches this in production.

9. **Sub-tenant scoping without re-architecture**: the grant model (§10, `agent_id × tool × tenant`) can add an optional `team_id` column with hierarchical policy evaluation without changing the storage paradigm. The harder part is that the registry currently shards at studio granularity, so sub-tenant scoping lives as a finer authz filter within a studio's shard rather than needing its own partition.

10. **Search actively causing harm, signal**: override rate (§21) tells you search is *mediocre*; what tells you it's *harmful* is correlating search-suggested tool calls with downstream error/rollback rates. If search-suggested selections are followed by corrective/undo actions at a statistically higher rate than explicitly-named ones, that's a direct harm signal.
