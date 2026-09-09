---
module: AI Infrastructure
topic: Serving
subtopic: Parallelism Topology
status: unread
tags: [ai-infrastructure, tensor-parallelism, pipeline-parallelism, expert-parallelism, topology]
prerequisites: [gpu-hardware-and-interconnects]
---
# Parallelism and Topology

Which parallelism to use is decided by what does not fit. Where to place it is decided by the [interconnect](03-gpu-hardware-and-interconnects.md). Most real-world serving underperformance is a placement error, not an algorithm error.

---

## 1. The Four Axes

| Axis | Splits | Communication | Serving role |
|---|---|---|---|
| **Data (DP)** | Requests | None in the forward path | Throughput. Always preferred when the model fits |
| **Tensor (TP)** | Weight matrices within a layer | `all-reduce` ×2 per layer | Fit a model that exceeds one GPU; cut latency |
| **Pipeline (PP)** | Layers into stages | Point-to-point activations at stage boundaries | Fit a model that exceeds one node |
| **Expert (EP)** | MoE experts | `all-to-all`, load-dependent | Fit sparse models; only sane inside a fast domain |

**The decision rule, in order:**

1. Does the model fit on one GPU with room for a useful KV cache? → **DP only.** Replicate. No collectives, best throughput.
2. Fits within one node? → **TP within the NVLink domain**, DP across nodes.
3. Exceeds one node? → **TP inside each node, PP across nodes.** Never TP across nodes.
4. MoE? → **EP**, kept inside a fast domain, composed with TP for the dense parts.

---

## 2. Tensor Parallelism

Megatron-style: column-split the first matrix of a pair, row-split the second, so exactly one `all-reduce` reconciles the result. Applied to both the attention output projection and the MLP down-projection — hence **two collectives per transformer layer**.

**What TP buys:** memory headroom (weights and KV divide by TP degree) and lower latency (each device does 1/TP of the work per layer).

**What it costs:** 2×L collectives per forward pass, on the critical path. For an 80-layer model that is 160 synchronization points.

**Why the TP degree should not exceed one node.** Inside a node the collectives ride NVLink/XGMI and are largely hideable. Crossing a node boundary drops them onto the network — roughly an order of magnitude less bandwidth and far worse latency — and you pay that 160 times per token-batch. The rule "TP degree ≤ GPUs per NVLink domain" is close to absolute for serving.

**Attention heads constrain TP degree.** Heads are split across TP ranks, so TP degree must divide the head count. With GQA the binding constraint is the *KV* head count: if a model has 8 KV heads, TP=16 forces KV head replication, which erodes the memory saving that motivated TP in the first place.

---

## 3. Pipeline Parallelism

Layers split into sequential stages; only activations cross stage boundaries, so communication is point-to-point and small — which is exactly why PP is the right axis for the *slow* links.

**The bubble.** With $S$ stages and $M$ microbatches, idle fraction is roughly $(S-1)/(M+S-1)$. Training hides this with many microbatches. **Serving usually cannot**, because interactive decode generates one token per sequence per step and there are not many microbatches to fill the pipe.

Consequence: PP is a *capacity* mechanism for serving — it makes an otherwise-unservable model servable — and it generally hurts latency. Use it to cross nodes, not to reduce TTFT.

---

## 4. Expert Parallelism

MoE routes each token to a small number of experts, so most weights are idle per token. EP places different experts on different devices and moves tokens to them: `all-to-all` dispatch, compute, `all-to-all` combine.

Two properties make EP operationally distinct:

- **Traffic is load-dependent.** Unlike TP's fixed collectives, `all-to-all` volume depends on routing decisions, so it is bursty and variable.
- **Load imbalance is a live failure mode.** If routing concentrates on a few popular experts, their devices become stragglers, and collectives are barriers — everyone waits. Mitigations are capacity factors (drop or reroute overflow tokens) and replicating hot experts.

MoE's serving appeal is real: activated parameters per token are a fraction of total, so compute per token is low. The catch is that **all** experts must be resident in memory, so MoE trades FLOPs for HBM capacity — which pushes you toward more devices, and therefore toward exactly the collectives that need a fast fabric.

---

## 5. Composition and Placement

Real deployments compose axes. A 3D layout for a large model on a multi-node cluster:

```
Node 0: [GPU0 GPU1 GPU2 GPU3]  ← TP group, NVLink domain, PP stage 0
Node 1: [GPU0 GPU1 GPU2 GPU3]  ← TP group, NVLink domain, PP stage 1
        ↑ PP activations cross the network — small, point-to-point
Whole thing replicated N times for DP
```

**The placement rules, ordered by how often they are violated:**

1. TP groups never span a node boundary.
2. TP groups never span a CPU socket (`SYS` in `nvidia-smi topo -m`) or a PCIe switch when NVLink is available.
3. PP boundaries are where you *want* the slow link, since traffic there is smallest.
4. EP stays within a fast domain; `all-to-all` on commodity networking is the worst case.

A scheduler that allocates by GPU count alone will violate rule 1 routinely. Topology-aware placement is not an optimization here — it is the difference between a working deployment and one that is several times slower for no visible reason.

---

## 6. Sizing Worked Example

**70B model, FP16, 8×80GB node.**

- Weights: 70B × 2 bytes = **140 GB**. Exceeds one 80 GB GPU → TP required.
- TP=2: 70 GB/GPU of weights, leaving ~10 GB for KV and workspace. Workable but tight.
- TP=4: 35 GB/GPU, leaving ~45 GB for KV. Much larger concurrency.
- TP=8: 17.5 GB/GPU, ~62 GB for KV — but 8-way collectives per layer, and diminishing returns.

**The tradeoff is not "which fits" but "which maximizes tokens/sec/GPU."** Higher TP gives more KV room and therefore more concurrent sequences, but adds collective overhead and reduces per-GPU efficiency. TP=4 with two replicas per node often beats TP=8 with one, because two independent replicas have zero cross-replica communication. Benchmark both against *your* sequence-length distribution — the answer moves with context length, since long contexts shift the balance toward needing KV room.

---

## 7. Interview Angles

### Q: Why not just use the highest tensor-parallel degree that fits? [Medium]
Because TP is not free throughput — it is memory headroom bought with communication. Each increment adds collectives on the critical path twice per layer while cutting per-GPU work, so efficiency per GPU falls even as latency improves. Past the point where the model plus a useful KV cache fits, additional TP degree usually costs more in collective overhead than it returns. The counter-consideration is that more TP means more free HBM for KV, which means more concurrent sequences and better throughput — so the optimum depends on your context-length distribution. Long-context workloads push the optimum higher; short-prompt high-QPS workloads push it lower, toward more independent replicas.

### Q: You must serve a model too large for one node. Walk through the layout. [Hard]
TP within each node up to the NVLink domain size, PP across nodes, DP for replication. The reasoning is communication volume matched to link bandwidth: TP's two all-reduces per layer are heavy and frequent, so they belong on the fastest link available and must never cross the network; PP moves only activations at stage boundaries, point-to-point and comparatively tiny, so it is the right axis to put on the slow link. Then I would name the serving-specific caveat: the pipeline bubble, $(S-1)/(M+S-1)$, is hard to hide during interactive decode because there are few microbatches to fill it — so I would keep the stage count as low as capacity permits and treat PP as a way to make the model servable at all, not as a latency optimization. Placement must be topology-verified, not assumed from GPU count.

### Q: What breaks first when you scale expert parallelism? [Hard]
Load balance. `all-to-all` traffic depends on routing decisions, so if the router concentrates tokens on a few popular experts, the devices holding them become stragglers — and because collectives are barriers, every other device waits on the slowest. This shows up as latency variance that worsens with EP degree and does not respond to adding capacity, because the problem is skew rather than aggregate throughput. Mitigations are capacity factors that cap tokens per expert and drop or reroute the overflow, replicating hot experts across devices, and load-balancing auxiliary losses at training time. The related constraint is memory: EP does not reduce total resident parameters, since every expert must be somewhere in HBM.

---

## Where to Next

- [GPU Hardware and Interconnects](03-gpu-hardware-and-interconnects.md) — the fabric these map onto
- [Distributed Inference Frameworks](02-distributed-inference-frameworks.md) — who decides placement
- [Distributed Training and Serving Reliability](../12-systems-and-scale/06-distributed-training-and-serving-reliability.md) — the training-side treatment
