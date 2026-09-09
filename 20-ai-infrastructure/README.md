---
module: AI Infrastructure
topic: Overview
subtopic: ""
status: unread
tags: [ai-infrastructure, serving, inference, index]
prerequisites: [llm-inference-ops]
---
# AI Infrastructure

The platform engineering under LLM serving. This section exists for a role that did not exist before LLMs: building the **inference data plane** — the distributed system that hosts generative models for many tenants at scale.

**What lives here:** inference engines, distributed serving frameworks, GPU interconnects, parallelism placement, routing, tenancy, SLOs, and the service architecture around them.

Split from `../12-systems-and-scale/` deliberately. That section explains *how inference works* — PagedAttention, batching, TTFT/TPOT math — from an ML engineer's angle. This section explains *how to run it as a fleet* from a platform engineer's angle. The interviews are different: one asks how the KV cache works, the other asks how you keep 10K tenants inside an SLO on 500 GPUs.

## What This Section Is For

Use it when the interview turns into:

- design a multi-tenant inference cloud
- why is p99 bad and what would you measure
- where should tensor parallelism live on this hardware
- how do you route to exploit prefix caches
- what are the SLOs and how do you defend an error budget
- what goes in the request path and what does not

## Files In This Section

- [Inference Engines Landscape](01-inference-engines-landscape.md) — vLLM, SGLang, TensorRT-LLM, MAX; what each optimizes for
- [Distributed Inference Frameworks](02-distributed-inference-frameworks.md) — llm-d, NVIDIA Dynamo, Ray Serve; control plane vs data plane
- [GPU Hardware and Interconnects](03-gpu-hardware-and-interconnects.md) — NVLink, XGMI, RoCE, NCCL, HBM, MIG
- [Parallelism and Topology](04-parallelism-topology.md) — TP/PP/EP/DP and where to place them
- [Smart Routing and Scheduling](05-smart-routing-and-scheduling.md) — prefix affinity, admission control, queueing
- [Multi-Tenancy and Isolation](06-multi-tenancy-and-isolation.md) — fairness, quotas, noisy neighbours, multi-LoRA
- [SLOs and Observability](07-slos-and-observability.md) — TTFT/TPOT SLIs, goodput, error budgets, GPU telemetry
- [Service Architecture, gRPC and Go](08-service-architecture-grpc-go.md) — streaming, cancellation, backpressure, decomposition
- [Interview Drills](09-interview-drills.md) — end-to-end design exercises in this register

## Reading Order

If you only have one pass:

1. `03-gpu-hardware-and-interconnects.md` — the physical constraints everything else obeys
2. `01-inference-engines-landscape.md` — what runs on it
3. `05-smart-routing-and-scheduling.md` — the highest-leverage component
4. `07-slos-and-observability.md` — how you know it works
5. `09-interview-drills.md` — rehearse

Read `04` before an interview naming specific hardware, `06` before one naming multi-tenancy, and `08` before any coding round.

## Prerequisites

This section assumes the mechanics from `../12-systems-and-scale/04-llm-inference-ops.md` (TTFT/TPOT, KV cache math, prefill vs decode) and `../12-systems-and-scale/03-llm-serving.md` (PagedAttention, continuous batching). Read those first if the terms are unfamiliar — this section builds on them rather than repeating them.
