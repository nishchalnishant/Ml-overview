---
module: AI Infrastructure
topic: Serving
subtopic: Smart Routing And Scheduling
status: unread
tags: [ai-infrastructure, routing, load-balancing, scheduling, admission-control]
prerequisites: [distributed-inference-frameworks]
---
# Smart Routing and Scheduling

The router is the highest-leverage component in an inference data plane. It decides which replica pays for each request, and a cache-aware router can cut TTFT by a large multiple without adding a single GPU.

---

## 1. Why Generic Load Balancing Fails

An L7 balancer assumes two things that are false for LLM inference:

**"Replicas are interchangeable."** They are not. Each replica holds a *different warm prefix cache* and a *different KV occupancy*. Sending a request to a cold replica when a warm one exists throws away a prefill you already paid for — often hundreds of milliseconds.

**"Requests are roughly uniform."** They are not, by orders of magnitude. A 50-token completion and a 100K-token prompt-heavy request are not comparable units of work, and the cost is not knowable exactly in advance — output length is unknown until generation ends.

Least-connections is better than round-robin but still wrong, because a "connection" is not a unit of load: one long-generating request occupies KV memory for a long time at low compute, while a big prefill is the reverse.

---

## 2. The Signals Worth Routing On

| Signal | Predicts | Cost to obtain |
|---|---|---|
| Prefix-cache overlap | TTFT savings | Router-side prefix index |
| KV cache occupancy | Admission headroom | Engine metric, scraped or pushed |
| Queue depth / waiting tokens | Time-to-schedule | Engine metric |
| In-flight sequence count | Decode step contention | Engine metric |
| Adapter residency | LoRA load cost | Router-side adapter map |
| Prompt length | Prefill cost | Free — it is in the request |

Output length is the one thing you cannot know, which is why scheduling is fundamentally an online decision under uncertainty. Length *prediction* exists but should be treated as a hint that degrades gracefully, never as a correctness input.

---

## 3. Prefix-Aware Routing

The single biggest win available. If a request shares a long prefix with something a replica already has cached, routing it there converts a full prefill into a cache hit.

**Mechanism.** The router keeps a lightweight index — typically hashes of token-block prefixes → replicas holding them. On arrival it computes the request's prefix hashes, finds the replica with the longest match, and weighs that against that replica's current load.

**The tension that defines this problem: affinity versus balance.** Pure affinity sends everything sharing a system prompt to one replica and melts it. Pure balance throws away every cache hit. Real routers score a blend:

```
score(replica) = w₁ · prefix_match_length − w₂ · load_factor
```

with the load term dominating past a saturation threshold, so a hot replica sheds work even at the cost of cache hits.

**Where this pays most:** shared system prompts across tenants, multi-turn conversations (route a session's turns to the same replica), agent loops with constant scaffolds, and few-shot templates. This is also why **session affinity** is a cheap approximation — routing by conversation ID gets much of the benefit without a prefix index.

---

## 4. Admission Control and Queueing

Under overload, the choice is not *whether* to degrade but *how*.

**Admission control** rejects or sheds before accepting work you cannot serve. Accepting everything means every request misses its SLO instead of some requests succeeding — a strictly worse outcome, and the classic failure of systems without it. Reject fast with a clear signal (429 plus `Retry-After`) so clients can back off rather than retry-storm.

**Queue discipline** determines who suffers:

| Discipline | Effect |
|---|---|
| FIFO | Fair, but one huge prefill head-of-line blocks everyone |
| Shortest-job-first | Best mean latency; starves long requests |
| Priority tiers | Business-aligned; needs anti-starvation for low tiers |
| Deadline-aware | Drop work that cannot meet its SLO anyway |

**Head-of-line blocking is the specific hazard.** A 100K-token prefill occupies the engine while short requests wait. **Chunked prefill** is the fix inside the engine — split the prefill across iterations so decode steps interleave — and it is why chunked prefill is really a *fairness* feature as much as a latency one.

**Load shedding order.** When shedding, drop the cheapest-to-lose first: speculative or best-effort work, then retries, then low-priority tenants, and never a request already mid-generation — killing an in-flight sequence wastes everything spent on it and frees KV only after the fact.

---

## 5. Disaggregated Routing

With separate prefill and decode pools, routing becomes two decisions plus a handoff:

1. Route prefill by prefix affinity and prefill-pool load.
2. Transfer KV to a decode worker over the fast interconnect.
3. Route decode by KV headroom and in-flight count.

The handoff is the risk: the transfer must be cheaper than recomputation or the whole design loses. That makes disaggregated routing a bet on interconnect bandwidth — see [interconnects](03-gpu-hardware-and-interconnects.md).

---

## 6. Model-Tier Routing

Distinct from replica routing: choosing *which model* serves a request. Route easy queries to a small cheap model, hard ones to a large one, using a classifier, a confidence threshold, or cascade-with-escalation.

The economics are strong — most production traffic is easy — but two caveats belong in any answer: the router is itself a model that can be wrong and needs its own evaluation, and cascades add latency for escalated requests, so the p99 can worsen even as the mean and cost improve.

---

## 7. Interview Angles

### Q: Design the routing layer for a fleet serving one model across 100 replicas with a shared system prompt. [Hard]
The shared system prompt is the dominant fact: every request has a long common prefix, so prefill is largely redundant work and prefix caching is where the wins are. I would have every replica warm on the shared prefix — that part is universal, so affinity is not needed for it — and then route on the *divergent* portion: a router-side index mapping prefix-block hashes to replicas, picking the longest match, blended with a load term that dominates past a saturation threshold so no replica melts from affinity. Session affinity by conversation ID handles multi-turn cheaply. Above that, admission control with 429 + `Retry-After` under overload, and chunked prefill in the engine so a long prompt cannot head-of-line block short ones. The metrics I would drive it from are per-replica KV occupancy and waiting-token queue depth, pushed or scraped frequently, with the router degrading to least-loaded if those go stale — the routing table must never require a live control-plane call.

### Q: Your p50 TTFT is fine and p99 is 10× worse. Diagnose. [Hard]
A ratio like that says queueing or interference, not model speed — the model itself has no 10× variance. The candidates, in order I would check: head-of-line blocking from long prefills, which I would confirm by correlating p99 events against concurrent prompt length and check whether chunked prefill is enabled; prefix-cache misses, where p50 requests hit a warm cache and p99 requests are the cold ones, visible as a bimodal TTFT distribution; KV-driven preemption, where a replica near capacity evicts and must recompute; and load imbalance, where a few replicas are hot because routing is affinity-heavy without a load term. I would also rule out the boring causes — autoscaling cold starts and a slow model load pulling a fresh replica into rotation before it is warm. The distinguishing evidence is whether p99 events cluster on specific replicas (imbalance or cold start) or spread evenly across the fleet (queueing discipline).

### Q: Why is admission control not optional at scale? [Medium]
Because without it, overload converts into universal SLO failure instead of partial success. If you accept every request, queue depth grows without bound, every request's latency grows with it, and eventually all of them miss their deadline — you have spent full capacity to serve nobody acceptably. Worse, clients time out and retry, which adds load exactly when you are saturated, so the system cannot recover on its own. Admission control makes the degradation deliberate: reject a fraction fast and explicitly with a 429 and `Retry-After` so clients back off, and keep the accepted fraction inside SLO. It is also where tenant policy attaches — you shed best-effort and low-tier traffic before premium — which turns overload from an outage into a business decision you made in advance.

---

## Where to Next

- [Multi-Tenancy and Isolation](06-multi-tenancy-and-isolation.md) — whose requests get shed first
- [SLOs and Observability](07-slos-and-observability.md) — the metrics this routes on
- [Inference Engines Landscape](01-inference-engines-landscape.md) — the caches being routed against
