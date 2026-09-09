---
module: AI Infrastructure
topic: Serving
subtopic: Distributed Inference Frameworks
status: unread
tags: [ai-infrastructure, llm-d, dynamo, ray-serve, control-plane, data-plane]
prerequisites: [inference-engines-landscape]
---
# Distributed Inference Frameworks

One [engine](01-inference-engines-landscape.md) serves one node. A framework turns many engine replicas into a service with routing, autoscaling, tenancy, and an SLO.

> **Freshness note.** llm-d, Dynamo, and Ray Serve are all moving quickly. Written here at the level of *what problem each solves and how it splits responsibilities*, which is durable; specific APIs and feature claims need checking against current releases.

---

## 1. Control Plane vs Data Plane

This is the distinction the role is named after, and the one to lead with.

**Data plane** — the path a request's bytes actually travel. Ingress, router, engine replica, token stream back. Its properties are latency, throughput, and per-request correctness. It must stay up and fast; it does no thinking about the fleet.

**Control plane** — decides what the fleet looks like. Model placement, replica counts, adapter distribution, config rollout, health and membership. Its properties are correctness and convergence, not latency.

**The invariant that matters: the data plane must survive control-plane failure.** If the controller dies, existing replicas keep serving with their last-known configuration; you lose the ability to scale, deploy, and rebalance, but you do not drop traffic. Designs that put a controller call in the request path violate this and turn a control-plane blip into a full outage. Expect to be asked to spot exactly this in a design review.

The corollary is that they scale differently: the data plane scales with QPS, the control plane with fleet size and change rate. They also fail differently, so they are separate services with separate SLOs.

---

## 2. What the Layer Above the Engine Must Do

Any framework in this space is solving the same list:

| Concern | The question |
|---|---|
| Placement | Which model on which GPUs, with what parallelism, on what topology |
| Routing | Which replica gets this request — see [smart routing](05-smart-routing-and-scheduling.md) |
| Autoscaling | Add/remove replicas against a metric that leads demand |
| Disaggregation | Run prefill and decode on separately-scaled pools |
| KV coordination | Transfer or reuse cache across replicas |
| Health | Detect a wedged GPU and remove it from rotation |
| Rollout | Ship a new model or config without dropping requests |
| Tenancy | Fairness, quotas, isolation — see [multi-tenancy](06-multi-tenancy-and-isolation.md) |

Compare frameworks by which of these they own versus delegate.

---

## 3. llm-d

Kubernetes-native distributed inference, built around the position that **the LLM-aware pieces belong in the Kubernetes routing layer** rather than in a bespoke sidecar stack. Vendor-backed with Red Hat and Google involvement, tracking the Gateway API Inference Extension.

The distinctive claims:

- **Inference-aware gateway routing.** Standard Kubernetes load balancing is round-robin or least-connections over interchangeable pods. That model is wrong for LLMs: replicas are *not* interchangeable, because each holds a different warm prefix cache and a different KV occupancy. llm-d pushes cache-awareness and load-awareness into the gateway.
- **Disaggregated prefill/decode** as a first-class deployment topology, with the two pools scaled independently.
- **vLLM-native**, rather than engine-agnostic by design.

Choose it when you are already Kubernetes-committed and want inference routing expressed in Kubernetes primitives that your existing platform tooling understands.

---

## 4. NVIDIA Dynamo

NVIDIA's datacentre-scale inference framework, successor in spirit to Triton Inference Server and aimed squarely at multi-node disaggregated serving.

The pieces worth naming:

- **Disaggregated prefill/decode** as the central design commitment, not a mode. Prefill is compute-bound and bursty; decode is bandwidth-bound and long-lived. Co-locating them means one workload's spikes damage the other's tail latency, and you cannot scale them independently. Separating them lets each pool have its own replica count, hardware profile, and batching policy.
- **KV-aware routing** — route to the worker whose cache already holds the relevant prefix.
- **NIXL**, a transfer library for moving KV blocks between workers over the fast interconnect, which is what makes disaggregation viable. Disaggregation is only a win if shipping KV from prefill to decode costs less than recomputing it — that is a *bandwidth* bet, and it is why this design assumes NVLink or RDMA rather than commodity networking.
- **Engine-agnostic** backends (vLLM, SGLang, TensorRT-LLM), unlike llm-d's vLLM focus.

Choose it for large multi-node deployments on NVIDIA fabric where disaggregation is the point.

---

## 5. Ray Serve

The general-purpose one. Ray Serve is a serving layer on Ray's distributed runtime, not an LLM-specific system — though `ray.serve.llm` adds LLM-aware pieces on top.

Its real strength is **heterogeneous pipelines**. Production inference is rarely one model call: it is a guardrail classifier, then a retriever, then an embedding model, then the LLM, then a reranker or verifier — with different hardware needs and different scaling curves per stage. Ray Serve models each as an independently-scaled deployment composed into a graph, which purpose-built LLM servers do not do well.

Its weakness is the mirror image: for a single LLM behind an endpoint, it adds a runtime and its failure modes without beating a dedicated engine on throughput.

Choose it when the unit of deployment is a **pipeline**, or when inference shares infrastructure with Ray-based training and batch work.

---

## 6. Comparison

| | llm-d | Dynamo | Ray Serve |
|---|---|---|---|
| Design centre | K8s-native LLM routing | Multi-node disaggregation | General distributed serving |
| Engine coupling | vLLM-focused | Agnostic | Agnostic |
| Disaggregation | Supported | Central commitment | Not the focus |
| Multi-model pipelines | Not the focus | Not the focus | Core strength |
| Scheduling substrate | Kubernetes | Own + K8s | Ray |
| Best when | Already K8s-committed | Large NVIDIA fleet | Pipelines, mixed workloads |

**The build-vs-adopt reality.** Large inference clouds usually end up building their own control plane, because tenancy, quota, billing, and capacity policy are business logic no framework ships. What they adopt is the engine (vLLM/SGLang) and the transfer primitives. Saying this out loud is a credible senior answer — it shows you know where the buy/build line actually falls.

---

## 7. Interview Angles

### Q: Design the control plane for a multi-tenant inference cloud. What is in the request path? [Hard]
As little as possible. The request path should be ingress → auth/quota check → router → engine replica → streamed response, with every one of those steps served from state cached locally in the data plane. Model placement, replica counts, adapter distribution, and tenant policy are computed by the control plane and *pushed* to data-plane components, so a router that loses its controller keeps routing on its last-known table. The control plane runs a reconciliation loop — observe fleet state, diff against desired state, act — with its own SLO on convergence time rather than on latency. The specific anti-pattern to call out is a synchronous controller lookup per request, which makes control-plane availability a multiplier on data-plane availability rather than independent of it.

### Q: When is prefill/decode disaggregation worth its complexity? [Hard]
When the two phases' resource profiles diverge enough that co-locating them forces a bad compromise, and when you have the interconnect to pay for KV transfer. Prefill is compute-bound and bursty; decode is memory-bandwidth-bound and long-lived. Co-located, a burst of long prompts stalls decode steps for every in-flight sequence, which shows up as TPOT jitter that is very hard to fix by tuning. Disaggregation lets each pool scale on its own signal and use its own batching policy. The cost is that KV must move from prefill to decode, so the bet only pays if that transfer is cheaper than recomputation — which requires NVLink or RDMA, and argues against it on commodity networking. It is also strictly more moving parts, so for a single-node deployment serving modest traffic it is not worth it; chunked prefill solves most of the same interference problem within one engine.

### Q: Why is round-robin the wrong load-balancing policy for LLM inference? [Medium]
Because it assumes replicas are interchangeable and requests are uniform, and neither holds. Replicas differ in what their prefix cache has warm and in how much KV memory is currently occupied by in-flight sequences. Requests differ enormously in cost — a 100-token completion and a 100K-token prompt are not the same unit of work. Round-robin will send a request to a cold replica when a warm one exists, throwing away a prefill it already paid for, and will send work to a replica near KV exhaustion while another sits idle. The right signals are prefix-cache affinity and current KV occupancy or queue depth, which is why the routing layer has to be LLM-aware rather than a generic L7 balancer.

---

## Where to Next

- [Smart Routing and Scheduling](05-smart-routing-and-scheduling.md) — the routing policies these frameworks implement
- [Multi-Tenancy and Isolation](06-multi-tenancy-and-isolation.md) — the tenancy layer none of them fully ship
- [SLOs and Observability](07-slos-and-observability.md) — how you know any of it is working
