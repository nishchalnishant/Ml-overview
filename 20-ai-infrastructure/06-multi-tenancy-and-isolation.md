---
module: AI Infrastructure
topic: Serving
subtopic: Multi Tenancy And Isolation
status: unread
tags: [ai-infrastructure, multi-tenancy, isolation, quota, fairness, noisy-neighbour]
prerequisites: [smart-routing-and-scheduling]
---
# Multi-Tenancy and Isolation

Serving one model well is an engineering problem. Serving many tenants on shared GPUs, each with their own SLO, is the problem an inference *cloud* exists to solve — and the layer no framework ships for you.

---

## 1. The Tension

Isolation and utilization are directly opposed. Dedicate hardware per tenant and isolation is perfect while utilization is terrible — most tenants are idle most of the time. Share everything and utilization is excellent while one tenant's burst destroys everyone's tail latency.

Every design here is a position on that spectrum:

| Model | Isolation | Utilization | Fits |
|---|---|---|---|
| Dedicated GPUs | Total | Poor | Enterprise, compliance, predictable heavy load |
| MIG partitions | Hardware | Moderate | Small models at density |
| Shared replicas + quota | Logical only | High | Bursty many-tenant traffic |
| Shared base + per-tenant LoRA | Logical | Very high | Many fine-tunes of one base |

Real platforms run several tiers at once and let price express the difference.

---

## 2. Noisy Neighbours — the Specific Mechanisms

"Noisy neighbour" is too vague to design against. In LLM serving it has distinct causes with distinct fixes:

**KV cache exhaustion.** One tenant's long-context requests consume the shared KV pool, so others cannot admit sequences or get preempted. *Fix:* per-tenant KV budgets, enforced at admission rather than after allocation.

**Batch-slot monopolization.** A tenant with high concurrency fills the continuous-batching slots; others queue. *Fix:* fair-share scheduling on slots, not just on request count.

**Prefill interference.** One tenant's 100K-token prompts stall everyone's decode steps. *Fix:* chunked prefill, plus per-tenant prompt-length caps.

**Prefix-cache eviction.** A high-volume tenant's traffic evicts another's warm prefixes, silently converting their cache hits into full prefills. *Fix:* cache partitioning or per-tenant reservations. This one is insidious because it degrades a *victim's* TTFT with no signal on the victim's own metrics.

**Adapter thrash.** More active LoRA adapters than fit in memory means near-constant loading. *Fix:* adapter-affinity routing plus residency limits per node.

Naming these individually is what separates a real answer from a hand-wave.

---

## 3. Fairness

**Weighted fair queueing over tokens, not requests.** The unit of fairness must be work, and in LLM serving the honest unit is tokens — prompt tokens weighted for prefill cost plus generated tokens for decode. Fairness measured in requests is trivially gamed by sending bigger requests.

**The unknown-output-length problem.** You cannot charge for a request up front because you do not know how long it will generate. Practical approach: debit tokens as they are produced, and use a per-tenant *in-flight* budget rather than a per-request one, so a tenant's concurrent consumption is bounded even when individual costs are unknown.

**Rate limiting.** Token-bucket on tokens-per-minute is the standard, with a separate concurrency cap. Two limits are needed because they bound different resources: TPM bounds throughput consumption, concurrency bounds KV residency. A tenant within their TPM can still exhaust KV with many simultaneous long-context sessions.

**Anti-starvation.** Strict priority starves the bottom tier under sustained load. Use weighted shares with a floor, or aging that raises priority with wait time, so low-tier requests eventually run rather than never.

---

## 4. Isolation Beyond Performance

Performance isolation is not the only kind, and interviews for cloud roles probe the others:

**Data isolation.** Requests carry tenant data through shared processes. The KV cache holds attention state derived from prompts — prefix-cache *sharing across tenants* is a real information-leak surface, since cache-hit timing can reveal whether another tenant submitted a given prefix. The safe default is to key the prefix cache by tenant, accepting lower hit rates, unless the prefix is a platform-owned system prompt.

**Fault isolation.** A crashed engine takes down every tenant on that replica. Blast radius is limited by spreading a tenant across replicas and by keeping replicas small enough that losing one is not a large fraction of capacity.

**Blast radius on rollout.** A bad model version must not reach all tenants at once — canary by tenant tier, with automatic rollback on SLO regression.

---

## 5. Multi-LoRA as the Density Play

Serving many fine-tunes of one base model is the case where multi-tenancy gets genuinely cheap: base weights are resident once, adapters are small, and requests for different adapters can batch together in the same forward pass.

What the platform layer must add:

- **Adapter registry** with versioning, and load-on-demand from object storage
- **Residency policy** — LRU over adapters, with pinning for premium tenants
- **Affinity routing** so requests reach a replica already holding the adapter
- **Thrash protection** — cap adapters per node; if the active working set exceeds capacity, that is a placement failure, and the fix is to shard tenants across replicas rather than to keep evicting

The failure mode to name: without affinity routing, adapter loads dominate and the density advantage disappears entirely.

---

## 6. Interview Angles

### Q: One tenant's traffic is destroying p99 for everyone. Walk through the diagnosis and fixes. [Hard]
First I would identify *which* mechanism, because the fixes differ. I would check whether the victims' requests are being admitted but queued (batch-slot monopolization), failing to admit at all (KV exhaustion), showing elevated TTFT with normal queue depth (prefix-cache eviction by the noisy tenant), or showing TPOT jitter correlated with the noisy tenant's prompt lengths (prefill interference). Per-tenant metrics are prerequisite — without KV occupancy and token consumption broken out by tenant, this is guesswork. The immediate mitigation is a rate limit and concurrency cap on the offending tenant, plus shedding their traffic first under pressure. The durable fixes are per-tenant KV budgets enforced at admission, fair-share scheduling on batch slots weighted by tokens rather than requests, chunked prefill so long prompts cannot block decode, and tenant-keyed prefix caching. If the tenant is legitimately large, the real answer may be moving them to dedicated capacity — at which point it is a pricing decision, not an engineering one.

### Q: Is it safe to share a prefix cache across tenants? [Hard]
Not by default. The KV cache holds state derived from prompt content, so a cross-tenant shared cache creates a timing side channel: a tenant can probe whether a given prefix is already cached by observing TTFT, which leaks whether some other tenant submitted that content. For prompts containing proprietary instructions or personal data that is a genuine disclosure risk, not a theoretical one. The safe default is to key the cache by tenant, accepting a lower hit rate as the cost of isolation. The defensible exception is platform-owned prefixes — a system prompt the platform itself supplies to everyone — where there is no tenant secret to leak, and sharing those recovers most of the practical benefit since they are usually the longest common prefix anyway.

### Q: How do you rate-limit a service whose per-request cost is unknown in advance? [Medium]
You do not price the request up front — you bound concurrent consumption and debit as work happens. Concretely: a token-bucket on tokens per minute, debited as tokens are actually generated rather than estimated at admission, plus a separate cap on in-flight requests. Two limits are necessary because they bound different resources — TPM bounds throughput, concurrency bounds KV residency, and a tenant within their TPM can still exhaust KV memory with many simultaneous long-context sessions. Prompt tokens are known at admission so they can be charged immediately with a weight reflecting prefill cost; output tokens are metered as they stream. For hard protection, add a max-output-tokens cap per tier, which converts the unbounded tail into a known worst case.

---

## Where to Next

- [Smart Routing and Scheduling](05-smart-routing-and-scheduling.md) — where quota is enforced
- [SLOs and Observability](07-slos-and-observability.md) — per-tenant SLIs
- [Distributed Inference Frameworks](02-distributed-inference-frameworks.md) — why tenancy is usually built, not bought
