---
module: Interview Prep
topic: Interview Prep
subtopic: Inference Data Plane
status: unread
tags: [interview, ai-infrastructure, inference, jd-mapped]
prerequisites: [ai-infrastructure]
---
# Inference Data Plane — Role Prep

Targeted prep for AI-infrastructure roles: building the distributed systems that host generative models at scale. Written against the shape of JDs like *Senior Software Engineer — AI Inference Data Plane*, which recur across inference clouds (DigitalOcean, Together, Fireworks, Baseten, Modal, AWS/GCP/Azure inference teams).

The content lives in [`../../20-ai-infrastructure/`](../../20-ai-infrastructure/README.md). This file maps requirements to it and sets expectations about what prep can and cannot cover.

---

## 1. The Role's Actual Shape

**This is a distributed-systems role that happens to serve LLMs.** That framing matters for how you prepare. You are not being hired to train or evaluate models. You are being hired to build proxies, routers, schedulers, and controllers, and to keep GPUs inside an SLO under multi-tenant load.

Consequences:

- Deep ML theory is *not* the bar. You need enough to reason about why decode is bandwidth-bound and what the KV cache costs — not backprop derivations or loss-function design.
- Distributed systems fundamentals **are** the bar: consistency, failure domains, backpressure, queueing, rollout safety.
- Coding rounds are systems coding — concurrency, streaming, data structures — not ML.

The most common failure mode for ML-background candidates is answering platform questions in ML terms. If asked to fix p99, the answer involves queueing and placement, not model architecture.

---

## 2. Requirement → Content Map

| JD requirement | Where |
|---|---|
| Hosting LLMs with **vLLM / SGLang / Modular** | [Inference Engines Landscape](../../20-ai-infrastructure/01-inference-engines-landscape.md) |
| **llm-d, NVIDIA Dynamo, Ray Serve** | [Distributed Inference Frameworks](../../20-ai-infrastructure/02-distributed-inference-frameworks.md) |
| GPU optimization, **NVLink / XGMI / RoCE** | [GPU Hardware and Interconnects](../../20-ai-infrastructure/03-gpu-hardware-and-interconnects.md) |
| **Tensor and data parallelism** | [Parallelism and Topology](../../20-ai-infrastructure/04-parallelism-topology.md) |
| **KV cache optimizations** | [KV Cache, MQA, GQA](../../10-llms/08-kv-cache-and-mqa-gqa.md), [LLM Inference Ops §3–4](../../12-systems-and-scale/04-llm-inference-ops.md) |
| **Smart routing** | [Smart Routing and Scheduling](../../20-ai-infrastructure/05-smart-routing-and-scheduling.md) |
| **Multi-tenant** inference cloud | [Multi-Tenancy and Isolation](../../20-ai-infrastructure/06-multi-tenancy-and-isolation.md) |
| **Continuous batching** | [LLM Serving §4](../../12-systems-and-scale/03-llm-serving.md), [Inference Ops §5](../../12-systems-and-scale/04-llm-inference-ops.md) |
| **Quantization** | [Quantization and Pruning](../../12-systems-and-scale/02-quantization-pruning.md) |
| **LLM architectures** | [Architecture Deep Dive](../../10-llms/03-architecture-deep-dive.md) |
| **Observability, defining SLOs** | [SLOs and Observability](../../20-ai-infrastructure/07-slos-and-observability.md) |
| **gRPC**, microservices | [Service Architecture, gRPC and Go](../../20-ai-infrastructure/08-service-architecture-grpc-go.md) |
| Availability / resiliency | [Distributed Training and Serving Reliability](../../12-systems-and-scale/06-distributed-training-and-serving-reliability.md) |
| Cost / efficiency at scale | [Cost Optimization](../../12-systems-and-scale/05-cost-optimization.md) |
| System design practice | [Interview Drills](../../20-ai-infrastructure/09-interview-drills.md) |

---

## 3. Gaps This Repo Does Not Close

Stated plainly so you can plan around them rather than discover them in the interview.

**Golang.** [`08-service-architecture-grpc-go.md`](../../20-ai-infrastructure/08-service-architecture-grpc-go.md) covers the concurrency and RPC patterns and enough idiom to discuss design credibly. It will not make you fluent. If the JD says "expert in Golang or Python" and your Go is thin, either **choose Python as your stated language** — the JDs almost always allow it — or write real Go before the loop. A useful minimum: a gRPC streaming proxy with context cancellation, a bounded worker pool, and a graceful-shutdown path.

**Hands-on operation.** Reading about NCCL topology debugging is not the same as having done it. If you can, run vLLM locally, serve a small model, load-test it, and watch the metrics. One afternoon of this produces concrete detail that reading cannot, and these interviews reward specificity heavily.

**Kubernetes depth.** The repo touches K8s but does not teach operator patterns, CRDs, or scheduler extension, which matter for a llm-d-style stack.

**Vendor currency.** llm-d, Dynamo, and MAX move fast. The content is written on durable architecture and freshness-stamped; verify specific feature claims against current docs before an interview that names one.

---

## 4. Preparation Sequence

**If you have a week:**

1. `20-ai-infrastructure/03` and `04` — hardware and placement. These constrain everything and are the most common source of concrete, credible detail.
2. `20-ai-infrastructure/01` and `02` — be able to say what each named tool optimizes for and when you would pick it.
3. `20-ai-infrastructure/05`, `06`, `07` — routing, tenancy, SLOs. This is the section most candidates are weakest on and where senior signal is easiest to show.
4. `20-ai-infrastructure/09` — drills, out loud, timed.
5. `20-ai-infrastructure/08` plus real coding practice in your chosen language.

**If you have a day:** `09-interview-drills.md` first to find your gaps, then read only the sections it exposes.

---

## 5. Questions to Ask Them

These signal that you understand the domain, and they surface whether the role is what the JD claims:

- Where is the boundary between your control plane and data plane, and what happens to serving when the control plane is down?
- Do you run disaggregated prefill/decode, and what did the KV transfer cost you?
- What are your actual SLIs — is TTFT scoped by prompt length?
- How is tenant isolation enforced: quota, MIG, dedicated capacity, or a mix?
- Do you route on prefix-cache affinity, and how do you balance affinity against load?
- Which engine, and what would make you switch?
- What is on call like, and what is the most common incident class?

---

## Where to Next

- [AI Infrastructure](../../20-ai-infrastructure/README.md) — the content
- [Interview Drills](../../20-ai-infrastructure/09-interview-drills.md) — start here to find gaps
- [Communication Framework](../01-communication-framework.md) — how to structure the answers
