# MCP Ecosystem

## 1. Problem Framing

EA is rolling out AI agents across studios: a live-ops copilot, a QA agent that triages crashes, a support agent that issues refunds/bans, and a content-pipeline agent touching DCC tools. Each needs to call *tools* — internal services, data warehouses, game APIs — in a standardized, discoverable, auditable way instead of bespoke function-calling glue per team.

Design **the MCP (Model Context Protocol) ecosystem**: the server/client architecture letting any LLM agent discover, authenticate to, and invoke tools exposed by dozens of independently-owned MCP servers (telemetry warehouse, anti-cheat, entitlements, build system, support ticketing) across studios and trust boundaries, safely and at scale.

This is a **distributed API gateway + service-discovery + capability-authorization system with an LLM client on one end** — not a model-serving problem (though the agent's own LLM serving is a dependency).

## 2. Functional Requirements

- FR1: Clients discover MCP servers and tool/resource/prompt manifests at runtime (`tools/list`, `resources/list`, `prompts/list`).
- FR2: Servers expose typed JSON Schema tool definitions for LLM function-calling; invocation via `tools/call`.
- FR3: Multi-tenant **tool registry** — studios register/version/deprecate their own MCP servers without a central release train.
- FR4: Per-tenant, per-agent, per-tool **authorization** (e.g., QA agent can query crash reports but not grant entitlements).
- FR5: Auth supports both human-delegated (OAuth-style, on behalf of an employee) and service-to-service (machine identity) flows.
- FR6: Session/context propagation — multi-turn tool use reuses a negotiated capability set without re-auth per call.
- FR7: Sync, long-running (job + poll/progress), and streaming (SSE) invocation.
- FR8: Audit trail on every call — identity, tenant, tool, redacted args, result hash, latency, outcome.
- FR9: Semantic search over tool descriptions so agents with 500+ tools can shortlist relevant ones.
- FR10: Kill-switch — revoke a tool/server/agent grant in real time (sub-minute propagation).
- FR11: Sandboxed execution for untrusted/community MCP servers (internal plugin marketplace).

## 3. Non-Functional Requirements

| Dimension | Target |
|---|---|
| Discovery latency (cold / cached) | p99 < 300ms / < 20ms |
| Tool invocation gateway overhead | p50 < 15ms, p99 < 60ms |
| Auth handshake | p99 < 50ms |
| Registry (control plane) availability | 99.9% |
| Gateway (data plane) availability | 99.95% |
| Throughput | 8,000 calls/sec sustained, burst 25,000/sec |
| Registry read consistency | Eventual, ≤5s (kill-switch: ≤60s hard SLA, target 10s) |
| Authz decision consistency | Strong — never serve a stale "allow" after revoke |
| Cost | ≤ $0.35 per 1,000 tool calls, fully loaded |
| Multi-tenancy | Hard tenant boundary, defense in depth |

## 5. Assumptions

1. ~1,200 MCP servers registered within 18 months; ~18,000 tools total (~15/server), growing ~25%/quarter.
2. ~6,000 active agent identities (scheduled, on-demand, developer copilots), each with a scoped credential.
3. Peak load driven by live-service incidents (many diagnostic agents firing at once) plus routine QA/support load.
4. Transport: Streamable HTTP + SSE (MCP 2025 spec); local stdio is developer-only, out of scope for hosted SLAs.
5. Agent LLM inference is a separate system (LLM Gateway) — this doc's latency budgets cover the tool-call path only.
6. Identity: EA's internal OIDC (Okta) for humans, SPIFFE/mTLS for service-to-service.
7. Registry metadata fits in a document store — the scale challenge is authz throughput and audit volume, not registry storage.
8. Tenancy = studio-level; team-level sub-tenant scoping is a stretch goal.

## 6. Capacity Estimation

**Tool calls**: 6,000 agents × ~2 calls/min ≈ 200/sec baseline; incident bursts add ~1,500/sec. Design target 8,000/sec sustained, 25,000/sec burst (10x headroom for live-service spikiness).

**Discovery**: ~1.4 req/sec baseline; a registry push can cause a cache-invalidation storm (~100 req/sec transient) — absorbed by caching (§11).

**Registry storage**: 18,000 tools × ~2KB manifest × ~8 versions ≈ 288 MB; server metadata ~6MB. Trivial — small HA document-store cluster.

**Audit volume (real storage driver)**: 8,000/sec × 86,400s ≈ 691M events/day × ~1.5KB ≈ ~1TB/day raw. Retention: 90 days hot (~94TB) + 2 years cold compressed (~152TB). This is a **Kafka → columnar store** problem.

**Authz throughput**: 8,000-25,000 checks/sec at p99 <50ms requires an in-memory/edge-cached policy engine (OPA-style), not a per-call RDBMS round trip.

**Compute**: stateless gateway pods (~1,000 req/sec each) → 8 pods steady state, 25 at burst, 2 vCPU/2GiB each, no GPU. Control plane (registry + OPA) is small (<16 vCPU), read-heavy and cached.

No model-weight sizing here — the only model involved is a small (~100M param) embedding model for semantic tool search, negligible next to gateway compute.

## 7. High-Level Architecture

```
                    Agent Runtime (MCP CLIENT)
                    Live-ops / QA / Support / Dev copilot
                              │ 1. mint session (OIDC/mTLS)
                              ▼
                 MCP GATEWAY (edge, stateless)
    TLS/mTLS · rate limiting · authn/authz (OPA sidecar)
    routing · JSON Schema validation · async audit emission
       │                    │                  │
2. discover        3. authz │        4. audit  │
       ▼                    ▼                  ▼
  TOOL REGISTRY      POLICY ENGINE (OPA)    KAFKA (audit topic)
  (multi-tenant,      cached grants,        → stream processor
   doc DB, semantic   sub-second revoke     → columnar store
   search idx)         propagation
       │                    │
       │        IDENTITY & GRANT STORE (agent identities,
       │         OIDC scopes, tenant ACLs) — pub/sub on revoke
       │
5. route call ▼
  MCP SERVERS (per-domain, per-studio, independently owned)
  Telemetry · Anti-Cheat · Entitlements/Support · Build MCP
       │
       ▼ 6. result (sync/streamed/job-poll) → back through gateway
```

## 8. Low-Level Components

- **MCP Gateway** (data plane): single ingress for agent→tool traffic; authn/z, schema validation, rate limiting, routing, audit emission. JSON-RPC 2.0 over Streamable HTTP/SSE. Stateless, scales on req/sec + p99 latency.
- **Tool Registry** (control plane): source of truth for servers, manifests, versions, ownership, semantic index. CI/CD-driven writes (low volume); reads cached at gateway. 3-node HA cluster.
- **Policy Engine (OPA)**: evaluates "can agent X, for tenant Y, call tool Z" from Rego policies. Runs as a gateway sidecar (local, no network hop), refreshed via pub/sub. Scales 1:1 with gateway.
- **Identity & Grant Store**: agent identities, OIDC credentials, per-agent tool scopes, human-delegation chains. Small HA cluster, read-heavy, aggressively cached.
- **MCP Servers** (leaf nodes): domain tools with their own auth/business logic, registered via CI. Independently deployed and scaled; gateway treats them as opaque upstreams with circuit breakers.
- **Audit Pipeline**: durable record of every call for compliance/forensics/anomaly detection. Kafka topic (async producer) → stream processor → columnar sink + real-time anomaly detector. Partitioned by tenant.
- **Semantic Tool Search**: ranks top-K relevant tools out of 18,000 given a natural-language task, so the LLM's context isn't stuffed with every schema. Small embedding model + ANN index; low query volume (once per session start).

## 9. API Design

MCP is JSON-RPC 2.0; gateway also exposes REST admin APIs. Protocol version negotiated in `initialize`; admin API uses URI versioning (`/v1`).

**MCP data plane** (`POST /mcp`): `initialize` → capability negotiation; `tools/list` → paginated tool manifests with JSON Schema; `tools/call` → structured args in, structured/streamed content out (`isError` flag).

**Admin/control REST (`/v1`)**

| Endpoint | Method | Purpose | Auth |
|---|---|---|---|
| `/v1/servers` | POST | Register server (manifest + endpoint + owner) | CI token, team-scoped |
| `/v1/servers/{id}` | PUT / DELETE | Update / deprecate | CI token (+approval gate on delete) |
| `/v1/servers/{id}/tools` | GET | Admin tool listing | Studio admin OIDC |
| `/v1/grants` | POST / DELETE | Grant or revoke agent→tool access (revoke propagates ≤10s) | Tenant admin OIDC |
| `/v1/tool-search` | POST | Semantic search `{query, tenant, k}` | Agent service token |
| `/v1/audit/query` | POST | Query audit log | Security/compliance role |

Response envelope: `{data, error, request_id}`. Breaking changes bump `/v2`.

## 10. Database Design

| Store | Type | Data | Why |
|---|---|---|---|
| Tool Registry | Document DB | Server manifests, tool schemas, version history | Arbitrary per-tool JSON Schema — poor fit for rigid columns |
| Identity & Grant Store | Relational (Postgres) | Identities, grants, ACLs, delegation chains | Grants are relational (agent × tool × tenant); referential integrity matters — a dangling grant is a security bug |
| Policy bundles | Object store + in-memory cache | Compiled Rego bundles | Read-mostly, versioned, pull-distributed |
| Audit Log (hot) | Columnar OLAP | 90-day queryable records | High-cardinality analytical queries at 691M events/day |
| Audit Log (cold) | Object storage, Parquet | 2-year compliance retention | Cheap, rarely queried |
| Semantic search index | Vector index (§16) | Tool description embeddings | ANN lookup |

**Partitioning**: registry and grants sharded/partitioned by `tenant_id` (hard isolation at storage layer). Audit columnar store partitioned by `(tenant_id, date)`. Kafka audit topic keyed by `tenant_id`.

## 11. Caching

| Cached item | Cache | Invalidation | Strategy |
|---|---|---|---|
| Tool manifests | Gateway memory + Redis L2 | TTL 5min + pub/sub on version bump | Cache-aside |
| Compiled policy bundles | Gateway sidecar memory | Pub/sub push on grant change (meets ≤10s revoke SLA) | Write-through push |
| OIDC token introspection | Redis, TTL = token life (≤15min) | Natural expiry + explicit bust | Cache-aside |
| Semantic search results | Redis, TTL 1hr | — | Cache-aside (staleness tolerable) |
| Tool-call results | **Not cached** — mostly non-idempotent/live data | N/A | Individual servers may cache internally |

Registry/grant reads vastly outnumber writes, so cache-aside dominates with per-item TTL tuning. Policy bundles are the one exception: push-invalidation is required to hit the sub-10s kill-switch SLA.

## 12. Queues & Async Processing

| Queue | Payload | Delivery | DLQ |
|---|---|---|---|
| `mcp.audit.v1` (Kafka) | Call audit events | At-least-once, idempotent consumers via `(agent_id, call_id)` | Malformed → DLQ, alert if depth > 100/5min |
| `mcp.grant-revoke.v1` (Kafka) | Revocation events | At-least-once, idempotent (set-removal) | Failure → immediate PagerDuty |
| `mcp.longrunning-jobs` | Long-running invocations (e.g., build trigger) | At-least-once, client-supplied idempotency key | 3 attempts → DLQ to owning team |
| `mcp.registry-sync.v1` | Manifest change events | At-least-once, "invalidate then re-fetch" | Non-critical, TTL catches up |

Exactly-once isn't attempted for audit/telemetry (idempotent consumers give effectively-once cheaply). Mutating long-running actions (e.g., bans) use client idempotency keys instead of transactional queue machinery.

## 13. Streaming & Event-Driven Architecture

| Topic | Producer | Consumers | Key fields |
|---|---|---|---|
| `mcp.audit.v1` | Gateway | Compliance, anomaly detector, analytics | tenant, agent, tool, args_digest, latency, outcome |
| `mcp.grant-revoke.v1` | Grant Store | Gateway sidecars, Policy Engine | grant_id, agent_id, scope, action |
| `mcp.registry-sync.v1` | Tool Registry | Gateway cache, search indexer | server_id, version, change_type |
| `mcp.tool-health.v1` | Gateway (circuit breaker) | Ops dashboards, autoscaler | error_rate, p99, breaker_state |

Each downstream concern gets its own consumer group off `mcp.audit.v1` so a slow consumer (e.g. batch analytics) never backpressures the revoke path, which lives on its own low-latency topic.

## 14. Model Serving

No large model sits on the critical tool-call path — this is a protocol/gateway layer. Two models are involved:

1. **Tool-search embedding model**: small (~100M param) sentence-embedding model, CPU-served, batched (10ms window) since query volume is low (session-start only).
2. **The agent's own LLM** is out of scope (served by EA's central LLM Gateway) — it just acts as the MCP client, consuming `tools/list` output and producing `tools/call` requests.

No GPU fleet for MCP infra; the embedding model runs on shared CPU capacity, isolated from the gateway's latency-critical path.

## 15. Feature Store

N/A for the gateway/registry itself — routing/authz decisions are policy-rule-based (OPA), not model-scored. The anomaly-detection consumer (§13) needs features: rolling call-rate and tool-diversity per `(agent_id, tool_name)`, served online from Redis (updated from the audit stream) and backfilled offline from the columnar store. Point-in-time correctness matters — training joins audit events with grant-state *as of that event's time*, not current grants, to avoid label leakage from later revokes.

## 16. Vector Database

Used narrowly for semantic tool search (§8), not any core protocol function.
- **Index**: HNSW over ~18,000 tool-description embeddings (~55MB total) — small enough that brute-force would even work, but HNSW gives headroom as the catalog grows ~25%/quarter. Re-evaluate IVF-PQ only near ~1M vectors.
- **Sharding**: single shard, replicated for HA. Search filters by tenant-visible tool subset at query time rather than physically separate indexes.

## 17. Embedding Pipelines

On tool registration/update (`mcp.registry-sync.v1`), a worker embeds `{name, description, params}` and upserts into the vector index with `{tool_id, server_id, tenant_id, version}` metadata. Same small embedding model as §14, batched (registration is low-volume). Re-embed triggered on manifest description changes; versioned so a failed re-embed falls back to the prior embedding rather than dropping the tool from search.

## 18. Inference Pipelines (End-to-End Request Lifecycle)

"Inference" here = the full agent tool-call round trip:

```
Agent → Gateway: tools/call (mTLS/OIDC)                 ~1-2ms
Gateway: verify token/cert, extract identity              ~3-5ms
Gateway → Policy sidecar (local, cached): authz check      ~1-3ms
Gateway: JSON Schema validation of args                    ~1-2ms
Gateway: rate-limit check (local + async Redis)             ~2-5ms
Gateway: circuit-breaker check on target server              <1ms
Gateway → MCP Server: proxied call (mTLS)         network + server's own work
                                                    (5ms fast query – 2s+ build trigger)
Server executes, returns result
Gateway: async audit emit to Kafka (non-blocking)
Gateway → Agent: result (sync) or notifications/progress (streamed)
Agent's LLM consumes result, may loop back to step 1
```

**Gateway-added overhead** (everything except upstream execution/network) must total p50 <15ms / p99 <60ms per NFR — that's what this system owns; upstream tool latency is each server team's SLO.

## 19. Training Pipelines

No model trains on the core protocol path. Two small adjacent pipelines:
1. **Tool-search embedding fine-tune**: quarterly, offline, single-GPU, on EA-internal (query → corrected tool) pairs mined from session logs — small-data, not distributed training.
2. **Anomaly-detection model**: gradient-boosted classifier or isolation forest on rolling call-pattern features, trained offline on the audit store, CPU-only.

The emphasis is data plumbing (clean labeled examples from audit logs), not training infra.

## 20. Retraining Strategy

| Model | Cadence | Trigger |
|---|---|---|
| Tool-search embedding | Quarterly / ad hoc | Catalog grows >30%, or search override rate degrades (§21) |
| Anomaly detector | Weekly | False-positive rate >5%, or new tool category shifts normal patterns |
| Policy bundles (non-ML) | On every grant/ACL change | Event-driven, compiled and pushed within seconds |

## 21. Drift Detection

| Drift | Metric | Threshold |
|---|---|---|
| Data drift (search queries) | PSI on query embeddings, weekly | PSI > 0.2 → review |
| Concept drift (search relevance) | Override rate (agent picks outside top-5) | >15% over 7 days → retrain |
| Concept drift (anomaly detector) | Analyst-labeled precision/recall | Precision <0.85 or recall <0.90 → retrain |
| "Protocol drift" (MCP-specific) | Schema-validation failure rate on `tools/list` ingestion | >1%/week → registry team review |

Worth noting in interview: this system's "drift" is as much about the tool ecosystem (schema/ownership churn) as classic ML data drift.

## 22. Monitoring

| Category | Metrics |
|---|---|
| Infra | Gateway CPU/mem, p50/p99 per stage, error rate by server, circuit-breaker transitions, Kafka lag |
| Control plane | Registration rate, manifest validation failures, cache hit ratio, revoke propagation latency |
| Search quality | Recall@5, override rate, embedding staleness |
| Security | Authz denial rate by agent/tenant, anomaly flag rate, credential age |
| Business | Call volume by studio, top-N tools, active agents/week, incident MTTR with agent assist |

## 23. Alerting

| Alert | Condition | Routing |
|---|---|---|
| Revoke propagation SLA breach | Not applied fleet-wide within 10s | PagerDuty, Security, P1 |
| Gateway p99 breach | >60ms for 5 min | PagerDuty, Platform, P2 |
| Tool error rate spike | Any tool >10% error over 5 min | Slack owning team, P3→P2 if sustained |
| Audit pipeline lag | Kafka lag >5 min backlog | PagerDuty, Platform, P2 (compliance risk) |
| Anomaly high-confidence flag | Score >0.95 | PagerDuty, Security, P1, auto-suspend agent |
| Registry validation failures | >1% of registrations | Slack, Registry team, P3 |
| Search relevance degradation | Override rate >15% | Slack, ML platform, P4 |

Domain-specific faults route to the owning team; cross-cutting issues (gateway, registry, policy, audit) go to central platform on-call.

## 24. Logging

- Structured JSON at every gateway stage: `request_id`, `agent_id`, `tenant_id`, `tool_name`, `server_id`, stage latencies, outcome, `trace_id`.
- **PII**: tool args often contain player IDs, emails, payment refs. Each tool's manifest tags sensitivity (`pii|financial|none`); tagged fields are hashed/tokenized before logging (`args_digest`) — never logged in plaintext. Full args are transient, not persisted.
- **Retention**: app logs 30 days; audit logs 90 days hot + 2 years cold (compliance requirement).
- **Right-to-erasure**: audit records use tokenized digests, so erasure = purging the token-mapping vault, not rewriting years of columnar history.

## 25. Security

**Threat model:**
1. **Over-privileged/injected agent** — prompt injection via untrusted tool output drives a destructive call. Mitigation: least-privilege grants, human-in-the-loop gate on high-risk tools enforced at the gateway, strict schema validation on args.
2. **Malicious/compromised MCP server** (esp. third-party) — crafted responses manipulate the agent or exfiltrate data. Mitigation: sandboxed execution tier, response allowlisting, no write-capable grants for untrusted servers.
3. **Registry poisoning** — compromised CI credential registers a malicious manifest. Mitigation: signed-commit namespace ownership tied to SPIFFE identity, approval gate on sensitivity/scope changes.
4. **Credential leakage** — long-lived tokens are a target. Mitigation: short-lived OIDC tokens (15min), auto-rotated mTLS/SPIFFE identities, no static API keys.
5. **Cross-tenant leakage via gateway bug** — routing bug leaks studio A's data to studio B. Mitigation: `tenant_id` as a hard partition key at every layer, server-side row checks too (defense in depth), chaos testing for cross-tenant leaks.

**Encryption**: TLS 1.3 everywhere (mTLS internal); audit store encrypted at rest with KMS keys.

## 26. Authentication

- **Service-to-service**: mTLS with SPIFFE/SPIRE workload identities, short-lived, auto-rotated. Gateway matches SPIFFE ID to a registered agent identity.
- **Human-delegated**: OIDC auth-code flow (Okta); token carries a `delegation_chain` claim (`{human_subject, agent_id, scope}`) so audit trail always traces to a human — critical for refund/ban tools.
- **Third-party servers**: separate OAuth client-credentials flow, mandatory scopes, sandboxed network tier, no delegation-chain trust.
- **Session propagation**: negotiated capability set from `initialize` is cached against the session ID so subsequent calls skip full authz re-derivation (token is still verified per call) — this is what keeps overhead inside the 15/60ms budget.

## 27. Rate Limiting

- **Algorithm**: token bucket per `(agent_id, tool_name)` and per `tenant_id` — O(1) memory at this cardinality, and naturally supports legitimate bursts (incident response) rather than hard-clipping.
- **Limits**: read tools 60/min (burst 120); mutating tools (refund, ban, build) 10/min (burst 20) — tighter due to blast radius. Per-tenant ceiling 2,000/min for fairness.
- **Enforcement**: gateway-local counters backed by Redis for cross-pod consistency (approximate, local-first-then-async-sync to avoid a sync Redis round trip per call).
- **Response**: MCP error with `retry_after_ms` so the agent's LLM can reason about backoff.

## 28. Autoscaling

- **Gateway pods**: HPA on requests/sec/pod (target 800) plus p99-latency trigger. Min 8, max 30 replicas.
- **Audit stream processors**: KEDA on Kafka consumer lag.
- **Policy sidecars**: scale 1:1 with gateway (no independent axis).
- **MCP servers**: each team's own autoscaling — gateway circuit breakers protect the system if one under-scales.

## 29. Cost Optimization

- Spot instances for stateless gateway pods (~60-70% savings on the largest compute line item).
- Caching (§11) collapses per-call registry/policy reads into a handful per TTL window.
- Audit tiering (90-day hot columnar → 2-year cold Parquet) avoids an ~8x hot-storage bill.
- Batched embedding requests mean a couple small CPU pods cover tool-search load.
- Per-tenant rate ceilings double as cost control (chargeback by call volume).
- Not caching tool-call results is a deliberate correctness-over-cost tradeoff — staleness risk (stale ban/entitlement data) outweighs the modest compute savings.

## 30. Operational Concerns

At SDE2 scope, treat as a checklist: automated backups with a tested restore path; one-command rollback to last-known-good; canary/blue-green rollout (small traffic slice first, watch error rate, then ramp); dashboards/alerts on latency, error rate, and top model-quality signals wired to on-call. Multi-region active-active topology and detailed Kubernetes/Terraform specifics are Staff/Principal-level concerns — know they exist, don't rehearse the manifests.

## 31. Why This Architecture

- **Stateless gateway + externalized policy engine** separates two things needing independent scaling curves: raw throughput (gateway) vs. authz freshness/correctness (policy).
- **Federated MCP server ownership** matches EA's actual org structure — a centrally-owned tool layer would recreate the gatekeeping bottleneck MCP exists to avoid.
- **Push-based revoke propagation** is the one place worth extra engineering effort (pub/sub fan-out), because it's the one NFR (sub-10s kill-switch) a pure TTL-cache design can't guarantee.
- **Document store for registry, relational for grants** matches each data shape — polymorphic JSON schemas vs. naturally relational grants.

## 32. Alternative Architectures

| Alternative | Why rejected / when preferred |
|---|---|
| Bespoke per-agent function-calling glue (no MCP) | O(agents × tools) integration cost, no standard discovery/auth — the pre-MCP status quo being replaced |
| Single monolithic MCP server for all tools | Recreates a central ownership bottleneck; fine only for a single studio with <50 tools |
| Sync-only invocation (no job queue) | Build/batch tools exceed HTTP timeouts; fine only if the catalog were 100% fast-path read-only |
| No policy engine, authz per-server only | Can't give fleet-wide sub-10s revocation, inconsistent semantics across 1,200 servers; kept as secondary defense-in-depth layer only |

## 33. Tradeoffs

| Decision | Benefit | Cost |
|---|---|---|
| Federated server ownership | Scales with org, no bottleneck | Inconsistent upstream quality; gateway must defend with circuit breakers |
| Push-based revoke | Meets sub-10s SLA | Extra pub/sub infra vs. simple TTL polling |
| Cache-aside for reads | Low latency/load | Staleness window everywhere except revoke |
| Document DB for registry | Flexible schema | Weaker referential integrity — must guard dangling references in app logic |
| Approximate Redis rate limiting | O(1) memory, no sync round trip | Slight overcounting possible across pods |
| No tool-result caching | Avoids stale ban/entitlement data | Forgoes compute savings on read-heavy tools |
| Sandboxed third-party tier | Contains blast radius | Extra environment to maintain |

## 34. Failure Modes

1. **Popular server (e.g. telemetry) down** → circuit breaker trips, gateway fails fast with a structured error instead of hanging sessions. Validated via chaos testing.
2. **Policy push fails to a pod subset** (network partition) → stale policy on some pods. Mitigation: bundle version logged per decision, background reconciliation force-refreshes laggards, alerts if >2 versions behind.
3. **Kafka partition unavailable** → calls still succeed (audit is async/non-blocking) but compliance visibility gaps. Mitigation: bounded local buffering with disk overflow, replay on recovery.
4. **Semantic search returns the wrong tool** (e.g. "ban" vs. "warn") → high-risk tools require human-reviewed manifest clarity and a mandatory disambiguation step, not reliance on ranking alone.
5. **Registry primary region outage** → writes blocked, reads continue from replicas; existing agents keep working off cached grants, new onboarding blocked until failover (~15min RTO).
6. **Compromised third-party server exfiltrates via echoed args** → sandboxing limits egress, schema validation catches malformed responses; primary defense is never granting sensitive-tool access to third-party servers in the first place.

## 35. Scaling Bottlenecks

- **At 10x (80,000 calls/sec)**: Redis rate-limit counters hot-spot on high-traffic tenants → shard by tenant, lean more on approximate local-then-reconcile. Audit Kafka partitions and columnar ingest need re-tuning. Search index (18K→180K tools) is where HNSW memory starts to matter.
- **At 100x**: exceeds a single-region gateway fleet's practical pod count → requires gateway sharding by tenant-hash across clusters. The single-writer Postgres grant store becomes the hard bottleneck, needing per-tenant sharding well before this point.

## 36. Latency Bottlenecks

| Stage | p50 | p99 |
|---|---|---|
| Network (agent→gateway) | 1ms | 3ms |
| Auth verification | 2ms | 5ms |
| Authz check (local sidecar) | 1ms | 3ms |
| Schema validation | 1ms | 2ms |
| Rate-limit check | 1ms | 4ms |
| **Gateway overhead subtotal** | **~6ms** | **~17ms** |
| MCP server's own execution | 5ms–2000ms+ | highly variable — **dominant cost** |
| Async audit emit | 0 (non-blocking) | 0 |

The gateway is nowhere near the bottleneck at design targets — server execution time dominates. The gateway's job is negligible self-overhead plus circuit breakers/timeouts so a slow upstream degrades gracefully.

## 37. Cost Bottlenecks

- **Audit storage/compute** (hot columnar, 691M events/day, 94TB hot) is the largest recurring line item, dwarfing gateway compute.
- **Gateway compute** is second, roughly proportional to call volume, mitigated by spot + caching.
- **Cross-region data transfer** for calls routed to a tool in a different region — worth monitoring per-tenant for mis-routed traffic.
- Registry/identity control-plane DBs and the search embedding model are not major cost drivers.

## 38. Interview Follow-Up Questions

1. How does a tool calling back into the agent mid-execution (human approval, or MCP sampling) change the architecture?
2. Walk through the worst case where a gateway pod is netsplit from revoke pub/sub for 30 seconds against a 10s SLA.
3. How do you stop one studio's misbehaving agent from degrading service for every other tenant?
4. At 100,000 tools instead of 18,000, what breaks first?
5. Could semantic search leak the *existence* of a tool a tenant isn't authorized to call?
6. What changes if a server needs to stream partial results over a multi-minute job instead of job-poll?
7. How do you vet a third-party MCP server before marketplace admission, beyond sandboxing?
8. Two studios register confusingly similar tool names — what's your strategy?
9. How would you add sub-tenant (team-level) authorization without a full re-architecture?
10. What telemetry tells you semantic search is actively causing harm, not just mediocre?

## 39. Ideal Answers

1. **Sampling/callback**: `sampling/createMessage` lets a server request a completion from the client's LLM mid-call, so the gateway can't treat `tools/call` as request/response-terminal — it keeps the session open and routes the callback through it, applying the same authz/rate limits to sampling requests.
2. **30s netsplit**: that pod serves stale grants up to 30s — a real SLA breach. Reconciliation job bounds this via version-lag pulls; high-risk tools can carry a "must revalidate live" flag forcing a synchronous grant-store check.
3. **Tenant isolation**: per-tenant rate ceiling is the first line of defense; fair-queuing/priority classes stop one tenant starving others; usage dashboards catch slow-building misbehavior early.
4. **100,000 tools**: HNSW likely still fine memory-wise; the real break point is gateway registry cache size — fix with a partial/LRU cache keyed by actual per-tenant usage instead of full-catalog caching.
5. **Search leaking existence**: yes, a real information-disclosure risk. Fix: apply the same authz filter to search as to `tools/list`, pre/post-filtering the vector query rather than relying on call-time denial alone.
6. **Long streaming job**: shift from job-submit+poll to `notifications/progress` over the session's SSE stream, with its own timeout/keepalive tuning distinct from the fast-path budget so a healthy long job doesn't trip the same breaker as a hung server.
7. **Third-party vetting**: static analysis of manifest scope requests, manual security review before listing, probationary reduced rate limits with human approval on first N calls, tighter anomaly-alert thresholds.
8. **Name collisions**: namespace tools as `{studio}.{domain}.{action}` to prevent collisions structurally; surface owning studio to the LLM as a disambiguating signal; track override rate as the production signal.
9. **Sub-tenant scoping**: add an optional `team_id` to the grant model with hierarchical policy evaluation — no storage paradigm change needed; it becomes a finer authz filter within a studio's existing shard.
10. **Harm vs. mediocre**: override rate shows mediocrity; correlating search-suggested tool calls with downstream error/rollback rates shows harm — if suggested selections get corrected/undone at a higher rate than explicitly-named ones, that's a direct harm signal.
