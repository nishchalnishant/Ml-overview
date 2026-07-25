---
module: Systems and Scale
topic: Distributed Training and Serving Reliability
subtopic: ""
status: unread
tags: [distributed-training, serving, reliability, scale, staff-mle]
prerequisites: [deep-learning, transformers]
---
# Distributed Training and Serving Reliability

This is the staff-level bridge between "the model trains" and "the system survives contact with traffic."

Think of it as the DevOps layer of ML at scale:
- training is the build farm
- checkpoints are release artifacts
- communication collectives are network plumbing
- serving is the runtime fleet
- autoscaling and rollback are the safety rails

---

## What Interviewers Are Really Testing

They want to know whether you can reason about:

- where the bottleneck is: compute, memory, network, or storage
- how to scale training without exploding cost
- how to serve models under latency and throughput constraints
- what fails first when traffic or model size grows
- how to keep the system observable and recoverable

If you only talk about model accuracy, you are answering the wrong question.

---

## Distributed Training

### Data Parallelism

Each worker gets a shard of the batch, computes gradients locally, then synchronizes.

**How it works:**
1. Replicate the full model on N workers
2. Split each mini-batch into N micro-batches
3. Each worker computes forward + backward pass independently
4. All-reduce the gradients across workers
5. Each worker applies the same update → models stay in sync

**When to use:** Model fits on one device; you want faster wall-clock time.

**Bottleneck:** Gradient synchronization. For a model with P parameters in fp32, each all-reduce moves 2P × 4 bytes (reduce + broadcast). With 100M params on 8 GPUs: ~800 MB per step.

**Scaling behavior:**
- Ideal: N GPUs → N× throughput
- Reality: communication overhead grows with N; at some point adding GPUs makes things slower
- Rule of thumb: compute-to-communication ratio > 10:1 for good scaling

**Common fixes:**
- Gradient compression (top-k sparsification, 1-bit Adam)
- Overlap communication with backward pass (PyTorch DDP does this)
- Larger batches to amortize sync cost (but watch learning rate warmup)
- Mixed precision to halve gradient size

### Model Parallelism

The model is split across devices because it does not fit on one GPU.

**Tensor parallelism:** split individual layers (e.g., split a large matrix multiply column-wise across GPUs). Each GPU computes part of the output, then all-gather to reassemble. Used inside transformer layers — Megatron-LM splits attention heads and MLP columns.

**Pipeline parallelism:** place different layers on different GPUs. Process micro-batches in a staged pipeline.

| Variant | Splits | Communication pattern | Best for |
|---|---|---|---|
| Tensor parallelism | Within a layer | All-reduce per layer | Very wide layers (LLM attention, MLP) |
| Pipeline parallelism | Between layers | Point-to-point, sequential | Very deep models |

**Pipeline bubbles:** GPUs sit idle waiting for activations from the previous stage. Mitigation:
- GPipe: accumulate micro-batches, then flush
- PipeDream: 1F1B schedule — each GPU alternates forward and backward micro-batches to keep all stages busy
- Interleaved stages: assign non-contiguous layers to each GPU to reduce bubble fraction

**Placement matters:** putting communicating layers on the same node (NVLink) vs. across nodes (InfiniBand) can 10× the bandwidth. Profile before you commit.

### Modern Memory-Saving Variants

**ZeRO (Zero Redundancy Optimizer):**

Standard data parallelism replicates everything: parameters, gradients, optimizer states. ZeRO shards them progressively:

| Stage | What's sharded | Memory reduction | Communication cost |
|---|---|---|---|
| ZeRO-1 | Optimizer states only | ~4× | Same as DDP |
| ZeRO-2 | + Gradients | ~8× | Slightly more |
| ZeRO-3 (FSDP) | + Parameters | ~N× (linear in #GPUs) | All-gather params each forward |

ZeRO-3 / PyTorch FSDP: each GPU holds only 1/N of the parameters. Before each layer's forward pass, all-gather the full parameters; after backward, reduce-scatter the gradients; then discard the non-local shards.

**Trade-off:** ZeRO-3 trades communication for memory. Good when the model barely doesn't fit; wasteful when it fits easily.

**Gradient checkpointing:** Don't store activations during forward pass; recompute them during backward. Trades ~30% extra compute for ~60% memory reduction. Stack with mixed precision for maximum effect.

**Mixed precision training:**
- Forward pass in fp16/bf16 (half the memory, 2× tensor-core throughput)
- Master weights in fp32 (for numerical stability)
- Loss scaling to prevent gradient underflow in fp16
- bf16 preferred over fp16: no loss scaling needed (larger exponent range)

**Activation offloading:** move activations to CPU during forward; prefetch back during backward. Only useful when GPU memory is the binding constraint and CPU-GPU bandwidth is decent (PCIe 4.0+).

### Communication Collectives — What Actually Moves

| Collective | What it does | When used |
|---|---|---|
| All-reduce | Sum gradients across all workers, result on all | DDP gradient sync |
| All-gather | Each worker sends its shard, all get the full tensor | FSDP parameter reconstruction |
| Reduce-scatter | Reduce + scatter in one op | FSDP gradient sharding |
| Broadcast | One worker sends to all | Weight initialization, checkpoint loading |

**Topology matters:**
- Ring all-reduce: 2(N-1)/N × data volume, bandwidth-optimal but latency grows with N
- Tree all-reduce: log(N) latency, less bandwidth-optimal
- NVLink within node (600 GB/s on A100), InfiniBand across nodes (200–400 Gb/s), Ethernet is usually too slow for synchronous training

### Checkpointing Strategy

For long training runs (days/weeks), failure is not if but when.

**Synchronous checkpointing:** all workers pause, write state to shared storage. Simple, but blocks training for minutes on large models.

**Asynchronous checkpointing:** snapshot state in background while training continues. Risk: state may be slightly inconsistent. Mitigation: write to a staging path, validate, then promote.

**What to checkpoint:**
- Model parameters (sharded or full)
- Optimizer states (can be 2-4× model size for Adam)
- Learning rate scheduler state
- Data loader position (for reproducibility)
- RNG states (for exact reproducibility)

**Frequency trade-off:** checkpoint every K steps. Too frequent → storage cost and I/O blocking. Too infrequent → more wasted compute on failure. Rule of thumb: checkpoint interval < MTBF (mean time between failures) for your cluster.

**Elastic training:** workers can join or leave mid-run (TorchElastic, Ray Train). Essential for preemptible/spot instances. Requires checkpointing + resharding.

---

## The Questions That Matter

Ask these before choosing a training strategy:

- Does the model fit on a single GPU? → If yes, data parallelism. If no, add model/pipeline parallelism or ZeRO-3.
- Is the bottleneck memory, not compute? → Gradient checkpointing, mixed precision, activation offloading.
- Is communication cheaper than recomputation? → Determines ZeRO stage choice.
- Are failures frequent enough that checkpointing strategy matters? → Spot instances: yes. Dedicated cluster: maybe not.
- Is the training job long enough that preemption becomes a real risk? → If >24h, elastic training or frequent checkpointing is non-negotiable.

---

## Serving Reliability

Serving is where staff-level judgment shows up fast. The interviewer isn't testing whether you know what a load balancer is — they're testing whether you can reason about the specific failure modes of ML inference under production constraints.

### Serving Patterns

| Pattern | Use when | Latency | Cost efficiency |
|---|---|---|---|
| Real-time inference | User-facing, latency-sensitive | p50 < 100ms | Low (GPUs always warm) |
| Batch inference | High-volume, low-urgency | Minutes to hours | High (burst compute) |
| Async / queue-based | Bursty workloads, tolerance for delay | Seconds to minutes | Medium |
| Cascade / multi-stage | Cheap model filters, expensive model ranks | Varies | High (most requests handled cheaply) |

### Batching for Throughput

**Static batching:** wait for N requests, process together. Simple but adds latency for the first request in the batch.

**Dynamic batching (continuous batching):** as soon as any request in the batch finishes, fill its slot with a new request. Critical for LLM serving where request lengths vary wildly.

**Key trade-off:** batch size ↑ → throughput ↑, but tail latency ↑. Set max batch size based on your p99 latency SLO, not average throughput.

### Autoscaling

**What to scale on:**
- Queue depth (reactive, simple)
- GPU utilization (lagging indicator — high util might mean you're already too slow)
- Request latency percentiles (p95, p99 — directly measures user experience)
- Predictive scaling based on historical traffic patterns (best for known patterns like daily peaks)

**Cold start problem:** GPU model loading takes 10–60 seconds. Mitigations:
- Keep warm replicas (costs money)
- Pre-load models on standby instances
- Use smaller models for cold-start traffic, route to full model when warm

### Reliability Controls

**Canary rollout:** Route 1–5% of traffic to the new model version. Monitor quality metrics (not just latency — model quality). If quality degrades, route 0% and investigate.

**Shadow traffic:** Send a copy of production traffic to the new model. Compare outputs. No user impact. Expensive (double compute) but safest for high-stakes systems.

**Circuit breakers:** When a downstream dependency (feature store, embedding service) fails:
- Serve from cache if available
- Fall back to a simpler model
- Return a safe default rather than an error
- Never let a model dependency take down the whole product

**Rollback:** Keep the previous model version loaded and ready. Rollback should be a config change, not a deployment. Target: rollback in < 60 seconds.

**Blue-green deployment:** Two identical environments. Switch traffic atomically. Zero-downtime deployments. More expensive (2× infrastructure during transition).

---

## Observability for ML Systems

### The Four Signals

| Signal | What to watch | Alert threshold |
|---|---|---|
| Latency | p50, p95, p99 per endpoint | p99 > 2× SLO |
| Throughput | Requests/sec, tokens/sec | Drop > 20% from baseline |
| Error rate | 5xx, timeout, model errors | > 0.1% for critical paths |
| Quality | Model-specific metrics (accuracy proxy, drift score) | Degradation > statistical noise |

### What Makes ML Observability Different

- **Input drift:** The distribution of features changes over time. Monitor feature distributions, not just predictions.
- **Prediction drift:** Model outputs shift without input changes (dependency update, feature pipeline bug).
- **Concept drift:** The true relationship between features and labels changes. Only detectable with delayed ground truth.
- **Silent failures:** The model returns plausible-looking garbage. No error code, no timeout, just wrong answers. This is why quality metrics matter more than availability metrics.

### SLOs for ML Systems

| Metric | Typical SLO | Notes |
|---|---|---|
| Availability | 99.9% (3 nines) | Match the product's SLO, not higher |
| Latency p99 | < 200ms (real-time), < 5min (batch) | Include feature fetch + model inference |
| Freshness | Model < 24h old, features < 5min old | Depends on how fast the world changes |
| Quality | AUC > X, NDCG > Y | Measured on a holdout or with delayed labels |

### Incident Response for ML

The hard part: ML incidents often don't look like traditional outages.

**Common ML incidents:**
1. Feature pipeline breaks → model gets stale or default features → predictions degrade silently
2. New model version has a bug → canary metrics look fine on average but a specific segment is badly affected
3. Upstream data schema changes → feature values are wrong → model behavior changes unpredictably
4. Training data was accidentally filtered → model is biased toward a subpopulation

**Runbook structure:**
1. Detect: quality metrics alert or user reports
2. Triage: is it model, data, or infrastructure?
3. Mitigate: rollback model or disable feature
4. Diagnose: compare feature distributions, prediction distributions, and model versions
5. Fix: retrain, patch pipeline, or update schema
6. Prevent: add monitoring, test, or data contract

---

## Failure Modes You Should Name Out Loud

| Failure | Symptom | First response |
|---|---|---|
| OOM during training | CUDA OOM error | Reduce batch size, enable gradient checkpointing, switch to mixed precision |
| Training stalls | Loss plateaus, GPU util drops | Check for network saturation, dead workers, NaN gradients |
| Slow convergence at scale | Loss decreases slower per wall-clock hour | Adjust learning rate (linear scaling rule), check gradient accumulation |
| Serving latency spikes | p99 blows up | Check for cold starts, GC pauses, batch size, input length distribution |
| Quality drops post-deploy | Metrics degrade on monitoring dashboard | Rollback, diff feature distributions, check data pipeline freshness |
| Cost explosion | Cloud bill spikes | Check for overprovisioned replicas, zombie training jobs, inefficient batching |
| Checkpoint corruption | Training can't resume | Verify checkpoints on write, keep N-2 as fallback |

---

## Production Tradeoffs

| Decision | Benefit | Cost | When to choose it |
|---|---|---|---|
| Bigger batch | Better hardware utilization | Higher memory, delayed updates, need LR warmup | Compute-bound training |
| More model sharding (ZeRO-3) | Fits larger models | More communication overhead | Model doesn't fit in aggregate GPU memory with ZeRO-2 |
| Gradient checkpointing | ~60% less activation memory | ~30% slower training | Memory-bound, can't add GPUs |
| Quantization (INT8/INT4) | 2–4× lower serving cost | Possible quality loss, calibration needed | Serving cost is the binding constraint |
| Dynamic batching | Better throughput | Potential tail-latency impact | Variable-length inputs (LLM, NLP) |
| Spot/preemptible instances | 60–90% cost reduction | Interruptions, need elastic training | Training runs > 4h with good checkpointing |
| Cascade serving | Reduces expensive model calls | System complexity, two models to maintain | High-volume with skewed difficulty distribution |

---

## Worked Example: Sizing a Training Job

**Scenario:** Train a 7B parameter transformer. You have a cluster of 8× A100-80GB GPUs on one node.

**Memory estimation:**
- Parameters: 7B × 4 bytes (fp32) = 28 GB; in bf16 = 14 GB
- Optimizer (Adam): 2 copies of parameters (momentum + variance) in fp32 = 56 GB
- Gradients: 7B × 2 bytes (bf16) = 14 GB
- Activations: depends on batch size and sequence length; for batch=1, seq=2048: ~8–15 GB

**Total per-GPU with full replication:** ~100+ GB. Doesn't fit on a single 80 GB A100.

**Solution path:**
1. Mixed precision (bf16 for forward/backward, fp32 master weights) → halves activation memory
2. ZeRO-2 across 8 GPUs → shard optimizer states and gradients: (56+14)/8 = 8.75 GB per GPU
3. Parameters stay replicated: 14 GB per GPU (bf16)
4. Per-GPU memory: ~14 + 8.75 + ~10 (activations) ≈ 33 GB → fits on A100-80GB with room for batch size > 1
5. If still tight: add gradient checkpointing → activations drop to ~4 GB

**This is the kind of back-of-envelope calculation interviewers want to see.** They're testing whether you can reason about memory composition, not whether you memorize exact numbers.

---

## Interview Angles

### Q: Explain data parallelism vs model parallelism vs pipeline parallelism. [Easy]

Data parallelism: replicate the model, shard the data. Each worker computes gradients on its shard, then all-reduce to synchronize. Best when model fits on one GPU.

Model parallelism: shard the model itself. Tensor parallelism splits layers (attention heads across GPUs); pipeline parallelism places different layers on different GPUs. Required when the model doesn't fit.

Pipeline parallelism introduces "bubbles" — idle time between micro-batches. 1F1B scheduling minimizes this.

**Cross-questions to expect:**
- *"Why does distributed training sometimes get slower as you add GPUs?"* → Communication overhead grows. At some point gradient sync takes longer than the compute it saves. The compute-to-communication ratio drops below the efficiency threshold.
- *"What's the difference between ZeRO-3 and tensor parallelism?"* → ZeRO-3 shards parameters across devices and all-gathers before each layer. Tensor parallelism splits the actual computation within a layer. ZeRO-3 is a memory optimization; tensor parallelism is a compute distribution.

### Q: Your model training loss spikes at step 50k. What do you do? [Medium]

Systematic diagnosis:
1. Check for NaN/Inf in gradients → if yes, reduce learning rate or improve loss scaling
2. Check data pipeline → corrupted batch, extreme outlier, encoding error
3. Check for hardware failure → compare loss across workers
4. Check learning rate schedule → did warmup end or decay kick in at step 50k?

**Cross-questions to expect:**
- *"How do you prevent this?"* → Gradient clipping (global norm), loss spike detection with automatic rollback to last checkpoint, data quality checks before training.
- *"Should you always roll back?"* → Not always. Some spikes are transient and training recovers. Roll back if the loss doesn't recover within 2–5× the spike duration.

### Q: Design a serving system for a 70B parameter LLM with p99 < 2s. [Hard]

Architecture:
- Tensor parallelism across 4–8 GPUs per replica (model doesn't fit on one)
- Continuous batching (vLLM/TGI) — don't wait for a batch to fill
- KV cache with PagedAttention — memory-efficient attention for long sequences
- Quantization: INT8 weights, fp16 KV cache (4× memory reduction for weights)

Reliability:
- Canary deploys with quality metrics (not just latency)
- Autoscale on queue depth + p95 latency
- Circuit breaker to a smaller model if the 70B is overloaded
- Health checks that verify model correctness, not just that the process is alive

Cost:
- At $2/GPU-hour with 8 GPUs per replica: $16/hour per replica
- Target: 10 requests/second per replica → $0.0004 per request
- Compare against API pricing to justify self-hosting

**Trap:** Saying "just use vLLM" without discussing the memory layout, quantization trade-offs, or what happens when the KV cache fills up. The interviewer wants to see you reason about resources, not name-drop tools.

### Q: What metrics would you put on a model-serving dashboard? [Medium]

Four categories:
1. **Infrastructure:** GPU util, memory, request queue depth, error rates
2. **Latency:** p50/p95/p99 broken down by model version, input length, and endpoint
3. **Quality:** prediction distribution shift, feature drift scores, delayed-label accuracy
4. **Business:** conversion rate, user engagement (if measurable), cost per prediction

**Cross-questions to expect:**
- *"Which one alerts you to a problem first?"* → Usually latency (fast feedback). Quality drift is slower but more dangerous because it's silent.
- *"How do you handle the delay between prediction and ground truth?"* → Proxy metrics (click-through, engagement) for fast feedback; delayed-label reconciliation (days/weeks) for true accuracy.

---

## See Also

- [LLM Serving — Production Inference Systems](03-llm-serving.md)
- [LLM Inference Operations](04-llm-inference-ops.md)
- [Cost Optimization in ML Systems](05-cost-optimization.md)
- [ML System Design](../15-system-design/README.md)
- [Production ML](../13-production-ml/README.md)
