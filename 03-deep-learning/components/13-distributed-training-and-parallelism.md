---
module: Deep Learning
topic: Components
subtopic: Distributed Training And Parallelism
status: unread
tags: [deeplearning, ml, components-distributed-trainin]
---
# Distributed Training and Parallelism

How to train models that don't fit on one GPU — and how to think about the decision at every scale.

---

## 1. Why Distributed Training

**Llama-3 70B training:**
- Model weights (bf16): 140 GB
- Adam optimizer states (fp32 master weights + m + v): 840 GB
- Gradients (bf16): 140 GB
- Activations (for a 4K seq batch): ~80 GB
- **Total: >1.2 TB** — requires ~15× A100 80GB GPUs minimum

**Four types of parallelism:**
1. **Data Parallelism (DP):** same model, different data batches across GPUs
2. **Tensor Parallelism (TP):** split individual weight matrices across GPUs
3. **Pipeline Parallelism (PP):** split model layers across GPUs
4. **Sequence Parallelism (SP):** split long sequences across GPUs

---

## 2. Data Parallelism (DP)

**Mechanism:** Replicate full model on N GPUs. Each GPU processes B/N samples. Synchronize gradients after backward pass.

**All-Reduce for gradient sync:**
$$g_{avg} = \frac{1}{N} \sum_{i=1}^N g_i$$

Implemented via ring all-reduce: O(2(N-1)/N × data) communication, effectively O(data) regardless of N.

```python
# PyTorch DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = DDP(model.cuda(), device_ids=[local_rank])

# Gradients automatically averaged across ranks during backward()
loss.backward()  # triggers all-reduce
optimizer.step()
```

**Limitation:** Each GPU must hold the full model. Fails when model > GPU memory.

---

## 3. ZeRO (Zero Redundancy Optimizer)

**Key insight:** In standard DP, each GPU holds full redundant copies of optimizer states, gradients, and parameters. ZeRO eliminates this redundancy.

### ZeRO Stage 1: Optimizer State Sharding

Each GPU holds only 1/N of optimizer states (Adam: fp32 params, momentum, variance).

**Memory reduction:** 4× for optimizer states (momentum + variance + master weights in fp32 = 12 bytes/param, vs 2 bytes/param model weights).

### ZeRO Stage 2: Gradient Sharding

Each GPU holds only 1/N of gradients in addition to stage 1.

**Memory reduction:** ~8× vs standard DP.

### ZeRO Stage 3: Parameter Sharding

All states + parameters sharded. Each GPU holds only 1/N of model parameters.

**Memory reduction:** ~64× vs standard DP (for large models with Adam).

```
Standard DP (100B model, N=64 GPUs):
  Each GPU: weights (200GB) + gradients (200GB) + optim (600GB) = 1000GB ❌

ZeRO Stage 3 (N=64):
  Each GPU: 1000GB / 64 = 15.6GB ✓
```

**Communication overhead:**
- Stage 1: all-reduce on gradients (same as DP)
- Stage 2: reduce-scatter on gradients (slightly better than all-reduce)
- Stage 3: all-gather before each layer's forward, reduce-scatter after each layer's backward

```python
# DeepSpeed ZeRO Stage 3
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},  # CPU offload if needed
        "offload_param": {"device": "cpu"},
        "overlap_comm": True,
    },
    "bf16": {"enabled": True}
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model, optimizer=optimizer, config=ds_config
)
```

### ZeRO Memory Summary

| Stage | Optimizer States | Gradients | Parameters | Memory/GPU (100B, N=64) |
|---|---|---|---|---|
| None (DP) | Full | Full | Full | ~15.6 TB/64 = 1TB ❌ |
| ZeRO-1 | 1/N | Full | Full | ~900 GB/64 = 14 GB ✓ |
| ZeRO-2 | 1/N | 1/N | Full | ~400 GB/64 = 6 GB ✓ |
| ZeRO-3 | 1/N | 1/N | 1/N | ~60 GB/64 = ~1 GB ✓ |

ZeRO-Infinity adds NVMe offload for even larger models.

---

## 4. Tensor Parallelism (TP)

**Mechanism:** Split individual matrix operations across GPUs.

**Column-parallel linear:**
$$Y = XW \text{ where } W = [W_1 | W_2]$$
$$Y = [XW_1 | XW_2] \text{ — each GPU computes half}$$

**Row-parallel linear (inverse):**
$$Y = XW = \sum_i X_i W_i \text{ — all-reduce to sum contributions}$$

**MLP tensor parallel pattern:**
```
FC1 (column-parallel): [d_model → 4d/2] per GPU
→ All-gather intermediate activations is avoided (use column+row pattern)
FC2 (row-parallel): [4d/2 → d_model] per GPU + All-Reduce
```

**Attention tensor parallel:**
```
Q,K,V projections: column-parallel (each GPU handles n_heads/N heads)
Output projection: row-parallel + All-Reduce
```

```python
# Megatron-LM style tensor parallel
class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_size):
        super().__init__()
        # Each GPU holds output_size/tp_size columns
        self.weight = nn.Parameter(torch.randn(output_size // tp_size, input_size))
    
    def forward(self, x):
        return F.linear(x, self.weight)  # partial output, no communication needed yet

class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size // tp_size))
    
    def forward(self, x_partial):
        y_partial = F.linear(x_partial, self.weight)
        # All-reduce to sum contributions from all TP ranks
        dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)
        return y_partial
```

**TP constraint:** Requires high-bandwidth interconnect (NVLink). Each all-reduce at every layer boundary. Works within a node (NVLink: 600 GB/s) but too slow across nodes (InfiniBand: 200 Gb/s).

---

## 5. Pipeline Parallelism (PP)

**Mechanism:** Split model layers across GPUs. GPU 0 holds layers 0–7, GPU 1 holds layers 8–15, etc.

**Naive pipeline (GPipe):**
```
Micro-batch 1:  GPU0 → GPU1 → GPU2 → GPU3
Micro-batch 2:           ← backward ←
```

**Pipeline bubble:** GPUs are idle while waiting for data from previous stage.

**Bubble ratio (GPipe):**
$$\text{bubble} = \frac{p-1}{m + p - 1}$$

where p = pipeline stages, m = micro-batches per batch.

For p=4, m=8: bubble = 3/11 ≈ 27%. Reduce by increasing m (more micro-batches).

**1F1B (One-Forward-One-Backward, PipeDream/Megatron):** Interleave forward and backward passes to reduce bubble.

```
GPU0: F0  F1  F2  F3  B3  B2  B1  B0
GPU1:     F0  F1  F2  F3  B3  B2  B1  B0
GPU2:         F0  F1  F2  F3  B3  B2  B1  B0
GPU3:             F0  F1  F2  F3  B3  B2  B1  B0
```

With 1F1B, bubble ratio = (p-1)/(m+p-1), same formula but GPU memory is reduced vs GPipe.

**Interleaved scheduling (Megatron v2):** Each GPU holds multiple non-contiguous layer chunks.

$$\text{bubble}_{interleaved} = \frac{p-1}{m \cdot v + p - 1}$$

where v = number of chunks per GPU. v=2 reduces bubble by ~50%.

---

## 6. 3D Parallelism

**Combine all three:** DP × TP × PP

```
Topology for 64 GPUs: DP=4, TP=8, PP=2

GPU grid:
  TP group (within node, NVLink):     [GPU0, GPU1, ..., GPU7]
  PP group (across 2 nodes):           [node0/GPU0, node1/GPU0]
  DP group (independent replicas):     [node0/GPU0, node2/GPU0, node4/GPU0, node6/GPU0]
```

**Decision tree for parallelism strategy:**

```
Model fits on 1 GPU?
  YES → Single GPU, done
  NO  →
    Model weights fit on 1 GPU (with CPU offload)?
      YES → ZeRO Stage 2-3 with DP
      NO  →
        Very large model (100B+)?
          YES → TP within nodes + PP across nodes + DP for replicas
          NO  → TP within nodes + DP across nodes
```

**Rule of thumb:**
- TP degree = GPUs per node (typically 8) — keep within NVLink domain
- PP degree = number of nodes if model is very large
- DP = fill remaining GPU budget

```python
# Megatron-LM example config
training_args = {
    "tensor_model_parallel_size": 8,   # TP within node
    "pipeline_model_parallel_size": 4,  # PP across nodes
    "num_gpus": 256,                    # DP = 256/(8*4) = 8 replicas
    "micro_batch_size": 4,
    "global_batch_size": 2048,          # 2048 / (4 microbatch * 8 DP) = 64 grad accum steps
}
```

---

## 7. Gradient Checkpointing

**Problem:** Forward pass activations for backpropagation require O(L × B × S × d) memory.

**Solution:** Recompute activations during backward pass instead of storing them.

**Memory:** O(√L) instead of O(L) — store checkpoints at every √L layers, recompute in between.

**Cost:** ~33% extra FLOPs for the recomputed forward passes.

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        # Don't save activations; recompute during backward
        return checkpoint(self.layer, x, use_reentrant=False)

# Apply to every 2nd layer (balance memory vs compute)
for i, layer in enumerate(model.layers):
    if i % 2 == 0:
        model.layers[i] = CheckpointedTransformerLayer(layer)
```

---

## 8. NVLink vs InfiniBand

| Property | NVLink (within node) | InfiniBand (across nodes) |
|---|---|---|
| Bandwidth | 600 GB/s (A100 NVLink 3.0) | 200–400 Gb/s (~25–50 GB/s) |
| Latency | ~1 μs | ~1–5 μs |
| Topology | All-to-all within node | Fat-tree or Dragonfly |
| Use case | TP all-reduce | DP gradient sync, PP activation |
| Cost | Included with server | ~$100K per switch |

**Practical implication:** TP must stay within a node (NVLink). DP/PP communicate via InfiniBand. NVLink is ~12× faster per byte than InfiniBand — TP communication is affordable at NVLink bandwidth but would bottleneck at InfiniBand rates.

---

## 9. Activation Memory Optimization

**Flash Attention:** Never materializes the full n×n attention matrix.

Standard attention: O(n²) memory for attention scores.
Flash Attention: O(n) memory via tiling — process attention in blocks, accumulate softmax in running fashion.

```
Block size = 64 tokens
Process 64 × 64 blocks of Q×Kᵀ at a time
Keep running max and sum for numerically stable softmax
Never store full [n, n] matrix in HBM → store only [n, d] output
```

Memory savings: 10–15× at n=8K, O(n) vs O(n²).

---

## Trade-offs Summary

| Strategy | Memory savings | Communication overhead | Use case |
|---|---|---|---|
| DP | None | All-reduce gradients | Fits on 1 GPU |
| ZeRO-1 | 4× optimizer | Same as DP | Multi-GPU, optimizer too large |
| ZeRO-2 | 8× optimizer+grad | Same | Default multi-GPU choice |
| ZeRO-3 | 64× all states | 2× DP comm (all-gather) | 100B+ or many GPUs |
| TP | 1/N per dimension | All-reduce each layer | NVLink, within node |
| PP | 1/p layers | Send activations between stages | Cross-node, large models |
| GC | 10–40× activations | +33% FLOPs | Long sequences |

---

## Canonical Interview Q&As

**Q: Walk me through ZeRO Stage 3 — what is sharded and what is the communication overhead?**  
A: ZeRO-3 shards parameters, gradients, and optimizer states equally across N GPUs. Before each layer's forward pass, an all-gather reconstructs the full parameter tensor from N shards (cost: 2×params bandwidth per step). After each layer's backward, a reduce-scatter distributes gradient shards back (cost: 2×params bandwidth). Net result: 2× communication vs standard DP, but N× memory reduction. For N=64, the memory reduction is 64× — enabling training of 100B+ models on commodity hardware. The key trade-off: ZeRO-3 has higher communication overhead than ZeRO-2, so use ZeRO-2 unless model literally doesn't fit with ZeRO-2.

**Q: How do you choose between TP and PP for a 70B model across 16 nodes (8 GPUs each, 128 total)?**  
A: Start with TP=8 within each node (NVLink, fast all-reduce). Then PP=4 across 4 nodes to fit the full model depth. DP=128/(8×4)=4 data replicas. PP instead of more TP across nodes: InfiniBand is 12× slower than NVLink — the all-reduce at every TP layer boundary would be a bottleneck at 200Gb/s InfiniBand. PP only passes activations between pipeline stages (one transmission per micro-batch, not per layer all-reduce), so PP communication is more tolerant of lower-bandwidth interconnects. With 1F1B scheduling and m=16 micro-batches, pipeline bubble ≈ (4-1)/(16+4-1) = 15.8%, acceptable.

**Q: What's the pipeline bubble and how do you reduce it?**  
A: The pipeline bubble is the fraction of time GPUs are idle in a pipelined setup. GPipe bubble = (p-1)/(m+p-1). With p=8 stages and m=8 micro-batches: bubble = 7/15 = 46% — terrible. Fix: (1) increase micro-batches m — more m amortizes the startup/teardown idle time; m=32 gives 7/39 = 18%; (2) interleaved scheduling (Megatron v2) where each GPU holds v non-contiguous chunks: bubble = (p-1)/(mv+p-1); v=2 halves the bubble at the cost of twice the pp communication; (3) zero-bubble PP (2024) eliminates bubble by careful scheduling of optimizer steps.

**Q: When would you use gradient checkpointing and what's the cost?**  
A: Use gradient checkpointing when activation memory is the bottleneck — typically at long sequence lengths (4K+) or large batch sizes. Standard activation memory = O(L × B × S × d). With checkpointing, it's O(√L × B × S × d) by storing activations only at checkpoint boundaries and recomputing the rest during backward. Cost: approximately 33% extra FLOPs for the recomputed segments. Practical rule: apply checkpointing to every other layer (50% recompute) when activation memory exceeds 30% of GPU VRAM. Modern implementations (FlashAttention) already recompute attention scores during backward, so layer checkpointing mainly saves FFN activations.
