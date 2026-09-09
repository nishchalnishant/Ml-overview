---
module: AI Infrastructure
topic: Interview Prep
subtopic: Interview Drills
status: unread
tags: [ai-infrastructure, interview, system-design, drills]
prerequisites: [service-architecture-grpc-go]
---
# Interview Drills — Inference Data Plane

Attempt each before reading the notes. These are scoped to the register these interviews use: you are the platform engineer, the model is a given, and the question is how the fleet behaves.

---

## How These Interviews Differ

You are not being asked to improve a model. The recurring shape is: *here is a serving requirement, design the system, then defend it under failure and under cost pressure.*

What consistently separates strong answers:

1. **Ask about traffic shape first.** Prompt-length distribution, output-length distribution, prefix sharing, burstiness, tenant mix. Nearly every design decision downstream depends on these, and answering without asking signals you have not run one of these systems.
2. **Quantify.** Weights in GB, KV per sequence, bandwidth per collective. These are arithmetic, and doing it out loud is the strongest available credibility signal.
3. **Name the bottleneck tier.** Compute, HBM bandwidth, HBM capacity, interconnect, or network — most answers reduce to identifying which one binds.
4. **Separate control plane from data plane** unprompted.
5. **State what you would measure** to confirm, rather than asserting a cause.

---

## Drill 1: Design the Serving Platform

> Design a multi-tenant inference service for a 70B model. 10K tenants, mixed interactive chat and batch summarization, p95 TTFT under 1 s for interactive.

**Clarify first:** prompt-length distribution (summarization implies long prompts, chat short), prefix sharing across tenants, tenant concentration, whether batch work has a deadline.

**Sketch of a strong answer:**

*Sizing.* 70B FP16 = 140 GB, exceeds one 80 GB device → TP required. TP=4 within a node leaves ~45 GB/GPU for KV; two replicas per 8-GPU node. Compute KV per sequence from the GQA head count and target context, divide, and state the resulting concurrency per replica — that number drives everything else.

*The central design decision is separating the two workloads.* Interactive and batch have opposed requirements: interactive needs low TTFT and tolerates low batch efficiency; batch needs throughput and tolerates seconds of queueing. Co-locating them means long summarization prefills head-of-line block chat decode. Separate pools, sized independently, is the clean answer; a single pool with priority scheduling plus chunked prefill is the cheaper one. Name both, pick one, justify.

*Routing.* Prefix-affinity routing blended with load, session affinity for multi-turn, admission control with per-tenant token buckets and concurrency caps.

*Tenancy.* Per-tenant KV budgets at admission, tenant-keyed prefix cache (state the leak reason), fair-share on tokens not requests, batch tier shed first.

*Control plane.* Separate service, out of the request path, pushing placement and policy to routers that keep serving on last-known state.

*SLOs.* TTFT bucketed by prompt length, TPOT per tier, goodput as the capacity metric, burn-rate alerts.

**Failure modes to volunteer:** adapter or cache thrash, one tenant's long prompts poisoning p99, a wedged GPU passing liveness checks, TP group placed across a socket boundary.

---

## Drill 2: The p99 Regression

> TTFT p50 is unchanged, p99 is 8× worse since last week. No deploys. Debug.

**The reasoning to show:** a p99-only regression with a flat p50 is queueing, interference, or a subset of unhealthy replicas — not model speed, which has no such variance.

**Check in order, cheapest first:**

1. Are p99 events concentrated on specific replicas? → cold starts, a throttling or ECC-degraded card, or a straggler in a TP group. Fleet-wide instead? → queueing or traffic-shape change.
2. Did traffic shape change? A new tenant sending long prompts shifts the tail without touching the median. Correlate p99 against prompt length.
3. Is TTFT bimodal? → prefix-cache hit rate dropped. A high-volume tenant may be evicting others' warm prefixes.
4. Preemption rate up? → KV pressure forcing evict-and-recompute.
5. Queue time vs prefill time in traces — this decomposition tells you which half to fix.

**The point being tested is discipline:** decompose the metric, form a hypothesis, name the evidence that would confirm it. Jumping to "add replicas" without diagnosis is the failing answer — if the cause is cache eviction or a bad card, capacity does not fix it.

---

## Drill 3: Cut Cost 40%

> Serving cost is $2M/year. Cut 40% without breaking the latency SLO.

**Order by return per unit of risk:**

1. **Utilization first.** Measure goodput per GPU. Most fleets are provisioned for peak and idle much of the time — batching tuning, autoscaling on queue depth, and consolidating underloaded replicas often recover a large fraction with no quality risk at all.
2. **Quantization.** FP8 or INT8 weights roughly halves weight memory and increases effective bandwidth; more KV room means more concurrency, so throughput gains compound. Requires eval to confirm quality; state that you would gate on task-level evals, not perplexity alone.
3. **Prefix caching.** If prompts share long system prefixes, this removes redundant prefill outright. Pure win where it applies.
4. **Model-tier routing.** Route easy queries to a smaller model. Large potential savings, but the router is a model that can be wrong and needs its own evaluation, and escalation adds tail latency.
5. **Spot/preemptible capacity** for batch tier, with checkpointing and drain handling. Not for interactive.
6. **Speculative decoding** where the acceptance rate justifies the draft-model overhead — it helps latency more reliably than cost, and can hurt throughput at high batch sizes.

**The senior move:** start with measurement, not a technique list. "I would first find whether we are capacity-bound or utilization-bound, because the answer changes the whole plan" is a better opening than naming quantization.

---

## Drill 4: Should You Disaggregate?

> Should prefill and decode run on separate pools?

**Answer with conditions, not a verdict.** In favour: divergent resource profiles (prefill compute-bound and bursty, decode bandwidth-bound and long-lived), the ability to scale and to choose hardware per pool independently, and the removal of prefill-into-decode interference. Against: KV must move from prefill to decode, so it only pays with NVLink or RDMA; it is strictly more moving parts and more failure modes; and chunked prefill solves most of the interference problem inside a single engine at a fraction of the complexity.

**The verdict to give:** worth it at multi-node scale with a fast fabric and a workload where prompt and generation lengths both vary widely. Not worth it for a single-node deployment at moderate traffic — use chunked prefill. Name the measurement that decides it: whether KV transfer time is below prefill recompute time on your fabric.

---

## Drill 5: Rapid Fire

Answer each in two or three sentences.

1. Why is decode memory-bandwidth-bound? *Reads all weights per step for one token per sequence; ~1 FLOP/byte, far under the ratio needed to saturate compute.*
2. Why must TP not cross nodes? *Two all-reduces per layer on the critical path; leaving NVLink costs ~10× bandwidth, paid 2×L times per pass.*
3. What does PagedAttention actually fix? *KV fragmentation — blocks need not be contiguous, so you stop over-reserving for worst-case length, and prefixes can be shared copy-on-write.*
4. Continuous vs static batching? *Per-iteration scheduling refills a finished sequence's slot immediately instead of waiting for the slowest in the batch.*
5. Why is GPU utilization a poor saturation metric? *It reports kernels resident, not useful work; a bandwidth-bound decode loop reads ~95% at a fraction of peak FLOPs. Use KV occupancy and queue depth.*
6. When is MIG the wrong tool? *Large models — instances do not share NVLink for TP and each sees only its HBM slice. MIG is for many small models at density.*
7. Why key the prefix cache per tenant? *Cross-tenant sharing is a timing side channel: TTFT reveals whether another tenant submitted a prefix.*
8. What is goodput and why prefer it? *Tokens/sec from SLO-conformant requests only. Raw throughput can be inflated by batching until everything violates latency.*
9. Why not retry a request mid-generation? *Re-runs prefill, doubling the expensive part, and retries cluster exactly when the system is saturated.*
10. What breaks first as expert parallelism scales? *Load balance — all-to-all is routing-dependent, hot experts become stragglers, and collectives are barriers.*

---

## Where to Next

- [Inference Engines Landscape](01-inference-engines-landscape.md)
- [Smart Routing and Scheduling](05-smart-routing-and-scheduling.md)
- [ML System Design](../15-system-design/02-ml-system-design.md)
