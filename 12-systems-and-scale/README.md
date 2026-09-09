---
module: Systems and Scale
topic: Overview
subtopic: ""
status: unread
tags: [distributed-training, serving, quantization, index]
prerequisites: [neural-networks]
---
# Systems and Scale

The engineering that separates a model that works from a model that ships at cost.

**What lives here:** distributed training, model compression, quantization and pruning, serving and inference, hardware, cost.

Split from `../13-production-ml/` deliberately: MLOps is process and lifecycle, this is systems engineering. They're asked by different interviewers.

## What This Section Is For

Use this section when the interview turns into:

- how to train on many GPUs without wasting money
- how to keep serving latency low under load
- how to cut inference cost without breaking quality
- how to reason about bottlenecks, memory, and throughput
- how to explain tradeoffs the way a platform engineer would

## Files In This Section

- [Model Compression](01-model-compression.md) — distillation, pruning, compression strategies
- [Quantization and Pruning](02-quantization-pruning.md) — accuracy vs speed vs memory tradeoffs
- [LLM Serving — Production Inference Systems](03-llm-serving.md) — batching, routing, scaling, deployment
- [LLM Inference Operations](04-llm-inference-ops.md) — operational patterns for running LLMs
- [Cost Optimization in ML Systems](05-cost-optimization.md) — where ML spend goes and how to reduce it
- [Distributed Training and Serving Reliability](06-distributed-training-and-serving-reliability.md) — staff-level systems design for training and inference

## Reading Order

If you only have one pass:

1. Start with `06-distributed-training-and-serving-reliability.md`
2. Then read `03-llm-serving.md`
3. Then read `05-cost-optimization.md`
4. Finish with compression and quantization if you need extra detail

## See Also

- [AI Infrastructure](../20-ai-infrastructure/README.md) — the platform-engineering layer above this section: inference engines, routing, multi-tenancy, and SLOs for a serving fleet
