---
module: AI Infrastructure
topic: Hardware
subtopic: Gpu Hardware And Interconnects
status: unread
tags: [ai-infrastructure, gpu, interconnect, nvlink, roce, nccl]
prerequisites: [llm-inference-ops]
---
# GPU Hardware and Interconnects

The layer under the inference engine. When a serving job is slow and the profiler says the GPUs are idle, the answer is almost always here.

> **Freshness note.** Specific part numbers and bandwidth figures below are current to ~mid-2025 and will drift. The *ratios* — HBM vs interconnect vs network, intra-node vs inter-node — are the durable content and are what interviews actually probe.

---

## 1. The Bandwidth Hierarchy

Every performance question in inference reduces to "which tier of this hierarchy did we fall down to?"

| Tier | Mechanism | Order of magnitude | Crossed when |
|---|---|---|---|
| On-chip | SRAM / registers | ~10–20 TB/s | Inside a fused kernel |
| Device memory | HBM3/3e | ~3–8 TB/s | Every weight read during decode |
| Intra-node GPU↔GPU | NVLink / XGMI | ~400–900 GB/s | Tensor parallelism |
| Intra-node fallback | PCIe Gen5 | ~64 GB/s bidirectional | TP without NVLink — a disaster |
| Inter-node | RoCE / InfiniBand | ~25–50 GB/s per NIC | Pipeline stages, KV transfer |
| Storage / object | NVMe, S3 | ~1–10 GB/s | Model load, KV offload |

**The key ratio:** each step down is roughly an order of magnitude. A tensor-parallel collective that falls from NVLink to PCIe costs ~10×. A KV transfer that crosses nodes instead of staying local costs ~20× versus HBM.

**Why decode is memory-bandwidth-bound, in one line:** decode reads all model weights from HBM to produce a single token per sequence. Arithmetic intensity is ~1 FLOP per byte read, far below the ~200–400 FLOP/byte the GPU needs to saturate its compute. You are renting bandwidth, not FLOPs.

---

## 2. NVLink and NVSwitch (NVIDIA)

**NVLink** is a point-to-point GPU-to-GPU link that bypasses PCIe and the CPU entirely. **NVSwitch** is a crossbar that makes every GPU in a node reach every other GPU at full NVLink bandwidth, rather than only its direct neighbours.

Why this matters for serving: tensor parallelism issues an `all-reduce` **twice per transformer layer** (after attention output projection, after the MLP down-projection). For a 80-layer model that is 160 collectives per forward pass. On NVLink+NVSwitch these are cheap enough to hide; on PCIe they dominate the step time.

**Topology awareness is the practical skill.** `nvidia-smi topo -m` prints the connection matrix. The letters that matter:

- `NV#` — NVLink with # links. What you want.
- `PIX`/`PXB` — same PCIe switch. Tolerable for data parallel, bad for TP.
- `SYS` — traffic crosses the CPU sockets over QPI/UPI. Worst case; never place a TP group across this boundary.

A TP group whose members are not all `NV#` to each other is a misconfiguration, and it is a common real-world one — schedulers hand out GPUs by count, not by topology, unless you tell them otherwise.

---

## 3. XGMI / Infinity Fabric (AMD)

AMD's equivalent on MI300-class parts. **xGMI** (inter-chip Global Memory Interconnect) links GPUs within a node; **Infinity Fabric** is the broader coherent fabric spanning CPU and GPU dies.

What to know for an interview that names it:

- Functionally it occupies the same slot as NVLink — high-bandwidth intra-node GPU↔GPU, bypassing PCIe.
- MI300X ships notably large HBM per device (~192 GB), which changes the calculus: models that need TP=4 on smaller cards may fit TP=1 or TP=2, eliminating collectives entirely.
- The software story is the risk, not the hardware. RCCL substitutes for NCCL, and vLLM/SGLang support ROCm, but kernel coverage and quantization-format support lag CUDA. "Does the kernel exist on this backend" is the real portability question.

**The honest framing if asked why a shop runs AMD:** supply availability and $/GB of HBM, accepting a less mature kernel ecosystem. It is a procurement decision as much as a technical one.

---

## 4. Inter-Node: RoCE vs InfiniBand

Once a model or a KV transfer crosses node boundaries, you are on the network.

**RDMA** is the shared idea: the NIC writes directly into remote GPU memory with no CPU copy and no kernel transition. With **GPUDirect RDMA** the path skips host memory entirely — GPU → NIC → wire → NIC → GPU.

| | InfiniBand | RoCE v2 |
|---|---|---|
| Fabric | Dedicated IB switches | Standard Ethernet |
| Flow control | Credit-based, lossless by design | Needs PFC + ECN configured correctly |
| Ops model | Separate network + subnet manager | Reuses existing Ethernet ops |
| Failure mode | Fewer, better understood | Misconfigured PFC → congestion spreading, head-of-line blocking |
| Typical driver | Peak training performance | Cloud economics and fabric convergence |

**RoCE's real gotcha:** it is only lossless if PFC and ECN are configured correctly end to end. RDMA's transport historically assumed a lossless fabric, so packet loss triggers pathological retransmission rather than graceful degradation. When a cloud's inference fabric "randomly" degrades under load, misconfigured or unpropagated PFC is a leading suspect. Newer congestion-control schemes reduce this dependence, but the operational point stands: RoCE moves complexity from the hardware into the network configuration.

Why a public cloud picks RoCE anyway: one Ethernet fabric for storage, tenant traffic, and RDMA, rather than a second specialized network to buy, staff, and debug.

---

## 5. NCCL — the Collective Library

NCCL (RCCL on AMD) implements the collectives every parallelism scheme depends on. It auto-discovers topology and picks ring or tree algorithms per message size.

**The four that matter:**

| Collective | Used by |
|---|---|
| `all-reduce` | Tensor parallelism (twice per layer), data-parallel gradient sync |
| `all-gather` | FSDP/ZeRO parameter gathering, TP output collection |
| `reduce-scatter` | ZeRO gradient partitioning |
| `all-to-all` | MoE expert dispatch and combine |

**Cost model.** For ring all-reduce over $N$ GPUs and $S$ bytes, each GPU sends roughly $2S(N-1)/N$ bytes — asymptotically $2S$, *independent of $N$*. This is why TP scales acceptably within a node: the constant is bandwidth, not GPU count. What kills you is the latency term, which grows with $N$ and with every hop that leaves NVLink.

**MoE is the special case.** `all-to-all` moves tokens to whichever device holds the chosen expert. Traffic depends on routing decisions, so it is *load-dependent and bursty* — unlike TP's fixed, predictable collectives. This is why expert parallelism is far more sensitive to interconnect quality than tensor parallelism, and why MoE serving is usually kept inside a single NVLink domain when possible.

**Debugging.** `NCCL_DEBUG=INFO` prints the chosen topology and algorithm at init. A hang in a collective is usually a mismatch: one rank entered a different collective, or with a different shape, than its peers. Collectives are barriers — one slow or crashed rank stalls all of them, which is why a single failing GPU takes down an entire TP group rather than degrading it.

---

## 6. HBM, and Why Capacity Sets Your Batch Size

For a decoder-only model, device memory holds three things: weights, KV cache, and activations/workspace.

$$\text{KV bytes} = 2 \times L \times H_{kv} \times d_{head} \times \text{seq} \times \text{batch} \times \text{bytes/elem}$$

The leading 2 is K and V. Note $H_{kv}$, not $H$ — this is exactly what GQA and MLA shrink, and it is why those architectural choices are *serving* decisions rather than quality decisions.

**The operational consequence:** weights are fixed, so KV cache gets whatever is left, and that remainder sets your maximum concurrent sequences. Concurrency determines throughput. So HBM capacity, not FLOPs, is usually what caps a serving deployment's tokens/sec/dollar.

---

## 7. MIG and GPU Partitioning

**MIG** (Multi-Instance GPU) partitions one physical GPU into hardware-isolated instances with dedicated SMs, L2 slices, and memory. Isolation is genuine — a noisy neighbour in one instance cannot steal another's bandwidth.

The alternatives, in increasing order of isolation and decreasing order of utilization:

| Mechanism | Isolation | Good for |
|---|---|---|
| Time-slicing | None — contexts interleave | Dev, bursty low-priority work |
| MPS | Weak — shared SMs, no memory protection | Trusted co-located small jobs |
| MIG | Hardware — dedicated SMs and memory | Multi-tenant, per-tenant SLOs |
| Whole GPU | Total | Anything needing full HBM or NVLink |

**The catch that matters for LLM serving:** MIG instances do not share NVLink for tensor parallelism, and each instance sees only its slice of HBM. That makes MIG a poor fit for large models — a 70B model needs full devices and the interconnect between them. MIG's place in an inference cloud is serving *many small models* (embeddings, rerankers, classifiers, small draft models) at high density, while large models get whole GPUs.

---

## 8. Interview Angles

### Q: Your 70B tensor-parallel deployment gets far lower throughput on one node type than another with identical GPUs. Where do you look? [Hard]
Topology first. Run `nvidia-smi topo -m` on both. The likely finding is that the slow node's TP group spans a PCIe switch or a CPU socket (`PXB`/`SYS`) rather than sitting entirely on NVLink (`NV#`). With two all-reduces per layer, an 80-layer model pays that penalty 160 times per forward pass. Confirm with `NCCL_DEBUG=INFO` to see the algorithm and path NCCL selected, and profile to check whether step time is dominated by collective wait. The fix is scheduler-side: request topology-aware placement so TP groups are always allocated within an NVLink domain, rather than by GPU count alone.

### Q: When would you deliberately not use tensor parallelism? [Medium]
When the model fits on one device, or when the only available GPUs are not NVLink-connected. TP buys you memory headroom and lower latency at the cost of a collective twice per layer. If the model fits in one GPU's HBM, running N independent replicas (pure data parallelism) gives strictly better throughput — zero collectives, and each replica gets its own KV cache pool. TP is a memory-pressure remedy first and a latency remedy second; it is never free throughput.

### Q: A cloud provider chooses RoCE over InfiniBand for its inference fabric. Argue both sides. [Medium]
For RoCE: one Ethernet fabric carries tenant traffic, storage, and RDMA, so there is no second specialized network to buy and staff; commodity switching is cheaper per port; the ops team already knows Ethernet. Against: RoCE's losslessness depends on PFC and ECN being correct end to end, and a misconfiguration produces congestion spreading and head-of-line blocking that degrade the whole fabric rather than one link — failure modes that are much rarer on IB's credit-based flow control. For inference specifically the tradeoff leans toward RoCE more than it would for large-scale training, because inference inter-node traffic (KV transfer for disaggregation, pipeline stage handoff) is lighter and burstier than a training job's sustained all-reduce.

---

## Where to Next

- [Parallelism and Topology](04-parallelism-topology.md) — mapping TP/PP/EP onto this fabric
- [Inference Engines Landscape](01-inference-engines-landscape.md) — what consumes this hardware
- [LLM Inference Operations](../12-systems-and-scale/04-llm-inference-ops.md) — the metrics layer above
