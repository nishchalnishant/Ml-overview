---
module: AI Infrastructure
topic: Operations
subtopic: Slos And Observability
status: unread
tags: [ai-infrastructure, slo, observability, sre, monitoring, incident-response]
prerequisites: [multi-tenancy-and-isolation]
---
# SLOs and Observability for Inference

"Defining SLOs" appears in these JDs as an explicit responsibility because most inference platforms are monitored like web services, and the metrics that matter are different.

---

## 1. Why Generic Web SLOs Fail Here

A standard service SLO is availability plus request latency. Both are wrong-shaped for streaming token generation.

**Latency is not one number.** A request has a time-to-first-token and a per-token cadence, and they have different causes and different user impact. A 2-second TTFT with fast tokens feels sluggish but usable; a 200 ms TTFT that then stalls mid-sentence feels broken. Collapsing them into "request latency" hides both.

**Total duration is not a defect signal.** A request that takes 60 seconds because it generated 2000 tokens is working correctly. Alerting on end-to-end latency generates constant false positives and trains people to ignore the page.

**Correctness is not binary.** A 200 response can contain a degraded answer — truncated by a token cap, served by a fallback model, or produced after a KV preemption. Availability alone will not see it.

---

## 2. The SLIs That Matter

| SLI | Definition | Why |
|---|---|---|
| **TTFT** | Request accepted → first token emitted | Perceived responsiveness; captures queue + prefill |
| **TPOT / ITL** | Mean or p99 inter-token latency | Streaming smoothness; captures decode contention |
| **Throughput** | Output tokens/sec per replica and fleet-wide | Capacity and efficiency |
| **Goodput** | Tokens/sec **from requests that met their SLO** | The honest capacity number |
| **Queue time** | Accept → scheduled | Separates queueing from model cost |
| **Admission rate** | Accepted / offered | Makes shedding visible |
| **KV utilization** | Occupied / total KV blocks | The real saturation signal |
| **Preemption rate** | Sequences evicted and recomputed | Silent latency killer |
| **Cache hit rate** | Prefix blocks reused / requested | Directly drives TTFT |

**Goodput deserves emphasis.** Raw throughput can be increased by batching harder until every request violates its latency SLO — the number goes up while the service gets worse. Goodput counts only SLO-conformant work, so it cannot be gamed that way, and it is the right target for capacity planning and for tuning batch size.

**KV utilization is the saturation metric, not GPU utilization.** GPU utilization as reported by `nvidia-smi` measures whether kernels are running, not whether they are doing useful work. A memory-bandwidth-bound decode loop shows high utilization while achieving a fraction of peak FLOPs. KV occupancy and queue depth predict when you will start rejecting or preempting; GPU utilization does not.

---

## 3. Writing the SLO

An SLO needs a threshold, a percentile, a window, and a **scope** — and scope is where inference SLOs go wrong.

The scoping problem: TTFT depends on prompt length, which the *client* controls. A single "p99 TTFT < 500 ms" target across all traffic is unmeetable when a tenant sends 100K-token prompts, and it makes your error budget hostage to client behaviour.

Fixes, in order of preference:

1. **Bucket by input length** — separate targets for short/medium/long prompt bands.
2. **Normalize** — express as TTFT per 1K prompt tokens.
3. **Cap** — enforce a max prompt length per tier so the tail is bounded by policy.

Similarly, TPOT should be scoped per model and per hardware tier, since a 7B and a 70B on different GPUs have no business sharing a target.

A workable starting set for an interactive tier:

```
TTFT   p95 < 800 ms   for prompts ≤ 4K tokens, 28-day window
TPOT   p95 < 50 ms    per model/hardware tier
Availability  99.9%   non-5xx, excluding deliberate 429 shedding
```

**429s are the subtle one.** Deliberate load shedding is the system working as designed, so counting it against availability punishes correct behaviour. But excluding it entirely means you can hit 100% availability by rejecting everything. The resolution is to track shed rate as its own SLI with its own budget, so it is visible and bounded rather than hidden.

---

## 4. Error Budgets

The budget is the allowed failure over the window: 99.9% over 28 days is roughly 40 minutes. Its function is to make the reliability/velocity tradeoff explicit — budget remaining means ship, budget exhausted means stop shipping and fix.

**Burn-rate alerting** beats threshold alerting. Page on *how fast* the budget is being consumed, with multiple windows: a fast burn (e.g. 14.4× over 1 hour, consuming 2% of budget) pages immediately; a slow burn (e.g. 3× over 6 hours) opens a ticket. This catches both sudden outages and slow degradations while generating far fewer false pages than a static latency threshold, which fires on every transient spike.

---

## 5. GPU and Engine Telemetry

Beyond the standard signals, this domain needs:

**Hardware** (DCGM or equivalent): SM occupancy, HBM bandwidth utilization, memory used, temperature and **power/thermal throttling**, ECC errors, NVLink/PCIe throughput. Throttling and ECC deserve alerts — a card that is thermally throttling or accumulating ECC errors degrades silently and should be drained rather than left in rotation.

**Engine**: running vs waiting sequences, KV blocks used/free, preemptions, cache hit rate, batch size distribution, tokens/sec in and out.

**The distinctive failure mode: silent degradation.** A wedged GPU may still respond to health checks while producing nothing. A liveness probe that only checks process health will not catch it. Health checks should be **functional** — run a tiny generation and verify tokens come back within a bound — and the fleet needs straggler detection, since one slow replica in a TP group stalls the group and one slow replica behind a router quietly poisons the p99.

---

## 6. Tracing and Attribution

Distributed tracing has to span router → scheduler → engine → (transfer) → decode, with spans for queue time, prefill, and decode separately. Without that decomposition, "the request was slow" is unactionable.

Every trace and metric needs **tenant, model, and adapter labels**, because the first question in any incident is whether it is global or one tenant. Watch cardinality: per-request labels will destroy a Prometheus instance, so keep tenant labels bounded (bucket small tenants into an "other" class) and push per-request detail into traces and logs rather than metrics.

---

## 7. Interview Angles

### Q: Define SLOs for a public LLM inference API. [Hard]
I would set separate SLIs for TTFT and TPOT rather than one request-latency target, because they have different causes and different user impact, and because total duration scales with output length and so is not a defect signal at all. TTFT must be scoped by prompt-length band or normalized per 1K prompt tokens — otherwise a tenant sending 100K-token prompts consumes an error budget over something they control, not something I do. TPOT is scoped per model and hardware tier. Availability counts non-5xx responses but excludes deliberate 429 shedding, with shed rate tracked as its own SLI and its own budget so correct load-shedding is not punished while total rejection is still visible. I would alert on multi-window burn rate rather than static thresholds, and target goodput rather than raw throughput for capacity planning, since raw throughput can be inflated by batching past the point where requests violate latency.

### Q: GPU utilization reads 95% but throughput is poor. What is happening? [Hard]
GPU utilization only reports that kernels were resident, not that they were doing useful work, so 95% is consistent with a badly bottlenecked system. The leading explanation for decode is that it is memory-bandwidth-bound: it reads all model weights from HBM per step with arithmetic intensity around 1 FLOP per byte, so the card is saturated on bandwidth while achieving a small fraction of peak FLOPs, and utilization looks high throughout. The things I would check are effective batch size — small batches mean re-reading weights for very few tokens, which is the classic low-throughput-high-utilization case — then collective wait time, since a TP group on the wrong topology spends much of its time in all-reduce that still counts as busy, then preemption and recompute rate, and finally thermal or power throttling, where the clock is reduced but occupancy is unchanged. The right metrics are achieved HBM bandwidth against peak, tokens/sec, and KV occupancy, not utilization.

### Q: How do you detect a replica producing wrong or degraded output rather than failing outright? [Medium]
Liveness and readiness probes will not catch it, because the process is healthy — so the health check has to be functional: issue a small canary generation on a fixed prompt and verify both that tokens return within a latency bound and that the output matches expectations, which for a greedy decode at temperature zero can be an exact match against a known-good reference. Beyond per-replica checks, I would watch for divergence across the fleet on identical inputs, which catches a single bad card or a mismatched weight load, and monitor output-token-length distribution and finish-reason mix per replica, since a card producing garbage or truncating early shifts those distributions well before anyone files a ticket. ECC error counts and throttling state feed the same decision, which is to drain the replica rather than wait for it to fail.

---

## Where to Next

- [Service Architecture, gRPC and Go](08-service-architecture-grpc-go.md) — instrumenting the data plane
- [Smart Routing and Scheduling](05-smart-routing-and-scheduling.md) — what these metrics drive
- [Distributed Training and Serving Reliability](../12-systems-and-scale/06-distributed-training-and-serving-reliability.md) — general ML observability
