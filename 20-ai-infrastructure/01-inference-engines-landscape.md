---
module: AI Infrastructure
topic: Serving
subtopic: Inference Engines Landscape
status: unread
tags: [ai-infrastructure, vllm, sglang, tensorrt-llm, serving]
prerequisites: [llm-inference-ops]
---
# Inference Engines Landscape

An inference engine owns one node's worth of GPUs and turns requests into tokens. Everything above it — routing, autoscaling, tenancy — is the [distributed framework's](02-distributed-inference-frameworks.md) job.

> **Freshness note.** Feature parity between these engines changes on a months timescale, and the gaps named below close and reopen. What is durable is *what each project optimizes for*, because that follows from its architecture and its sponsor. Reason from the design centre, and say plainly that specific feature claims need re-checking against current releases.

---

## 1. What an Engine Actually Owns

Independent of vendor, an engine is four subsystems:

1. **Memory manager** — allocates the KV cache, decides who gets evicted
2. **Scheduler** — picks which requests run in the next forward pass
3. **Kernels** — the attention and GEMM implementations
4. **Model executor** — shards weights across devices, drives the collectives

Engines differ chiefly in the first two. Kernels converge (everyone adopts FlashAttention-class work), and executors are similar. **Scheduling and memory are the interesting differentiators**, and they are what interview questions should be answered in terms of.

---

## 2. vLLM

The de-facto default, and the reference implementation of **PagedAttention**: KV cache in fixed-size blocks with a block table per sequence, so physical memory need not be contiguous. This removes the fragmentation that forced earlier servers to over-reserve for worst-case sequence length, and it is what makes high-concurrency serving practical.

The consequences worth naming:

- **Copy-on-write sharing.** Sequences sharing a prefix share physical blocks. Parallel sampling and beam search cost far less than *n* independent generations.
- **Automatic prefix caching.** A repeated system prompt is prefilled once and reused across requests — the single biggest TTFT win available in most chat and agent workloads.
- **Continuous batching.** Scheduling happens per iteration, so a finished sequence's slot is refilled immediately instead of at batch boundaries.

Design centre: **throughput and generality**. Broadest model coverage, quickest support for new architectures, largest community. If you have no specific reason to pick otherwise, this is the default.

---

## 3. SGLang

Shares the paged-memory idea but its distinguishing contribution is **RadixAttention**: the prefix cache is organized as a radix tree over token sequences rather than a flat hash of full prefixes.

Why the data structure change matters. A flat prefix cache gets a hit only on an exact prefix match. A radix tree matches the **longest common prefix** across *all* cached sequences, and keeps it resident across requests under LRU eviction. The workloads this transforms:

- **Multi-turn chat** — each turn extends the previous, so every turn hits a long shared path
- **Agent loops** — the scaffold and tool definitions are constant, only the trajectory tail differs
- **Tree/branching search** — branches share everything up to the divergence point
- **Few-shot prompting** — a fixed exemplar block precedes every varying query

The second piece is a **structured-generation frontend**: constrained decoding via a compressed FSM that can skip ahead through tokens forced by a grammar (e.g. the fixed key names in a JSON schema) instead of running a full forward pass per determined token.

Design centre: **workloads with heavy prefix sharing and structured output**. On agent and multi-turn traffic the gap over a flat cache is large; on single-turn traffic with no shared prefix the two are much closer.

---

## 4. TensorRT-LLM

NVIDIA's engine, built on a compiler rather than a runtime interpreter. Models are compiled ahead of time into an optimized engine plan for a specific GPU architecture, batch profile, and precision, with kernel fusion and autotuning applied at build time.

The tradeoff is direct: **peak per-GPU performance on NVIDIA hardware, paid for in flexibility**. Compilation takes minutes to hours, engines are not portable across GPU generations, and shape ranges are fixed at build time. It is also where NVIDIA's newest hardware capabilities land first — FP8 and FP4 paths on new architectures typically appear here before elsewhere.

Choose it when the model set is stable, the hardware is homogeneous NVIDIA, and the last 20–30% of throughput is worth an offline build step in your deploy pipeline. Avoid it when you rotate models frequently or run mixed hardware.

---

## 5. Modular / MAX

MAX is Modular's inference platform, notable less for a scheduling innovation than for an argument about the stack: kernels written in **Mojo** rather than CUDA C++, targeting portability across vendors without giving up hand-tuned performance.

The thesis is that the CUDA moat is really a *kernel-ecosystem* moat, and a language that compiles to multiple accelerator backends dissolves it. What to say if a JD names it: it targets vendor-neutral serving with an alternative kernel-authoring story, and its relevance to a given shop scales with how much that shop wants to avoid single-vendor lock-in. Treat performance and coverage claims as needing current verification.

---

## 6. Choosing

| Requirement | Pick |
|---|---|
| General serving, broad model coverage | vLLM |
| Agent loops, multi-turn, heavy prefix reuse | SGLang |
| Constrained/structured output at high rate | SGLang |
| Max throughput, stable models, all-NVIDIA | TensorRT-LLM |
| Multi-vendor / avoiding CUDA lock-in | MAX, or vLLM on ROCm |
| CPU or edge, single stream | llama.cpp |

**The answer that generalizes past this table:** these converge. PagedAttention is universal now; radix-style prefix caching and chunked prefill are being adopted across the board. Selection should turn on *your workload's prefix-sharing structure, your model-rotation cadence, and your hardware homogeneity* — not on a benchmark number, which will have moved by the time you deploy.

---

## 7. Interview Angles

### Q: When does SGLang's RadixAttention beat vLLM's prefix caching, and when is it a wash? [Hard]
It depends entirely on the prefix structure of the traffic. A flat prefix cache hits on exact prefix matches; a radix tree finds the longest common prefix across all cached sequences and retains it under LRU. So RadixAttention wins where sharing is deep, dynamic, and branching: multi-turn chat where turn *n* extends turn *n−1*, agent loops with a constant scaffold and divergent trajectories, and tree search where branches share a trunk. It is roughly a wash on single-turn requests with unique prompts, since there is no shared structure for either scheme to exploit — there you are comparing kernels and scheduler details, not caching. The right interview move is to ask what the traffic looks like before naming an engine.

### Q: Why is TensorRT-LLM often faster, and why would you still not use it? [Medium]
Faster because it compiles ahead of time: kernel fusion, autotuning against the specific GPU architecture, and fixed shape profiles let it specialize in ways a runtime interpreter cannot, and NVIDIA's newest precision paths land there first. You would still avoid it when model rotation is frequent — every model and every precision change needs a build step measured in minutes to hours, engines do not port across GPU generations, and shape ranges are frozen at build time. A platform serving many customer-supplied or rapidly-changing models pays that cost continuously, which usually outweighs the throughput gain. Homogeneous hardware plus a stable model set is the condition that makes it worthwhile.

### Q: You are asked to support 500 fine-tuned variants of one base model. What does that imply about engine choice? [Hard]
It rules out anything requiring per-model compilation, which effectively removes TensorRT-LLM. The requirement is multi-LoRA serving: one set of base weights resident in HBM, adapters loaded per request and swapped without reloading the base. vLLM and SGLang both support this, so the decision moves to the layer above — adapter registry, LRU residency policy for hot adapters, and routing requests to a replica that already has the adapter warm. The failure mode to name is adapter thrash: if the working set of adapters exceeds what fits in memory, you pay a load on nearly every request. That makes adapter-affinity routing a requirement, not an optimization.

---

## Where to Next

- [Distributed Inference Frameworks](02-distributed-inference-frameworks.md) — orchestrating many engine replicas
- [Smart Routing and Scheduling](05-smart-routing-and-scheduling.md) — exploiting prefix caches from the router
- [LLM Serving](../12-systems-and-scale/03-llm-serving.md) — PagedAttention and batching mechanics in depth
