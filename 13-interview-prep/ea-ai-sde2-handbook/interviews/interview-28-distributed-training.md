# Interview 28 — Distributed Model Training Infrastructure
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the Central AI team. EA is building a proprietary Foundation Model trained on petabytes of gameplay telemetry and player behavior data. This model has 10 Billion parameters.

Your task is to **design the distributed training infrastructure** to train this massive model across hundreds of GPUs efficiently, handling data parallelism, model parallelism, and fault tolerance.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Cluster Hardware (Are we using AWS p4d instances with NVLink, or a heterogeneous mix?)
- Parallelism Strategy (A 10B model doesn't fit on a single GPU. How do we shard it?)
- Storage/Data Loading (How do we feed petabytes of data to GPUs without starving them?)
- Fault Tolerance (What happens when 1 GPU out of 500 dies midway through a 3-week training run?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"What is the hardware topology? Do we have high-speed interconnects?"**
   → *Answer: Yes, we are using nodes with 8x A100 80GB GPUs, connected via NVLink, and Infiniband between nodes.*

2. **"Does a 10B parameter model fit in the memory of a single A100 80GB GPU?"**
   → *Answer: A 10B model in FP32 takes 40GB just for weights. Optimizer states (Adam) and gradients take another 120GB. No, it does not fit.*

3. **"Where is the petabyte dataset stored, and how is it formatted?"**
   → *Answer: It is stored in AWS S3 in Parquet format.*

---

## Part 4 — Expected Assumptions

- **Architecture:** PyTorch Lightning or native PyTorch Distributed.
- **Parallelism:** Because the model doesn't fit in VRAM, we must use Fully Sharded Data Parallel (FSDP) or DeepSpeed Zero (Stage 3).
- **Data Loading:** Must use optimized streaming (e.g., WebDataset or Ray Data) because petabytes cannot be downloaded to local disk.

---

## Part 5 — High-Level Solution

```
  [AWS S3 (Petabyte Dataset)]
       │
  [Streaming Data Loader (WebDataset)]
       │
  [Training Cluster (e.g., 16 Nodes, 128x A100 GPUs)]
  ┌────────────────────────────────────────────────────────┐
  │ 1. PyTorch FSDP (Fully Sharded Data Parallel) shards   │
  │    the model weights, gradients, and optimizer states  │
  │    across the 128 GPUs.                                │
  │ 2. Forward Pass: GPUs gather necessary weights via     │
  │    All-Gather across Infiniband/NVLink.                │
  │ 3. Backward Pass: Gradients are synced via Reduce-Scatter│
  └────────────────────────────────────────────────────────┘
       │
  [Checkpointing Service]
  Saves distributed checkpoints to S3 every hour for Fault Tolerance.
```

**Core ML Component:** Sharding strategy. Standard Distributed Data Parallel (DDP) duplicates the model on every GPU, which OOMs (Out of Memory) instantly for 10B parameters. FSDP/DeepSpeed is mandatory.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Memory Math & FSDP
- Model: 10B params.
- FP16/BF16 precision: 2 bytes per param = 20GB for weights.
- Adam Optimizer: Needs FP32 master weights (40GB), momentum (40GB), and variance (40GB).
- Total = 140GB. 
- GPU limit = 80GB.
- **Solution:** Use PyTorch FSDP. It slices the 140GB state across multiple GPUs. If we shard across 8 GPUs, each holds ~18GB, leaving plenty of room for batch activations.

### Step 2: Data Loading Bottlenecks
- CPUs cannot parse Parquet files fast enough to keep 8 A100s busy.
- Convert Parquet to WebDataset (tar files). It allows the DataLoader to stream bytes directly from S3 into memory with zero overhead, utilizing multithreading and prefetching.

### Step 3: Fault Tolerance
- If 1 GPU fails, the entire `NCCL` communicator group crashes, killing the 3-week job.
- Use `torchrun` (Torch Distributed Elastic). It detects node failures, pauses training, provisions a new node, restores from the last S3 checkpoint, and resumes automatically.

---

## Part 7 — Complete Python Code

*Note: We will write the core PyTorch FSDP setup script.*

```python
"""
fsdp_trainer.py - Fully Sharded Data Parallel Training
"""
import logging
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dummy 10B Parameter Model
# ---------------------------------------------------------------------------
class MassiveModel(nn.Module):
    def __init__(self):
        super(MassiveModel, self).__init__()
        # Mocking a massive transformer block
        self.embedding = nn.Embedding(50000, 4096)
        self.layers = nn.ModuleList([nn.Linear(4096, 4096) for _ in range(40)])
        self.out = nn.Linear(4096, 50000)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

# ---------------------------------------------------------------------------
# Distributed Setup
# ---------------------------------------------------------------------------
def setup():
    """Initializes the distributed process group via torchrun."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

# ---------------------------------------------------------------------------
# Training Loop with FSDP
# ---------------------------------------------------------------------------
def train():
    local_rank = setup()
    
    # 1. Initialize Model (on CPU first to avoid OOM)
    if local_rank == 0:
        logger.info("Initializing Model...")
    model = MassiveModel()
    
    # 2. Configure FSDP Wrapping Policy
    # We wrap individual layers that exceed 100M parameters.
    # This prevents FSDP from keeping the entire model un-sharded in memory.
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100_000_000
    )
    
    # 3. Wrap Model in FSDP
    # FSDP automatically shards weights, gradients, and optimizer states
    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        device_id=torch.cuda.current_device()
    )
    
    # 4. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training Step (Mocked)
    model.train()
    for step in range(10): # 10 steps for demo
        # Mock streaming data batch
        inputs = torch.randint(0, 50000, (16, 128)).to(local_rank) # Batch 16, Seq 128
        labels = torch.randint(0, 50000, (16, 128)).to(local_rank)
        
        optimizer.zero_grad()
        
        # Forward pass (FSDP automatically All-Gathers the needed shards via NVLink)
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs.view(-1, 50000), labels.view(-1))
        
        # Backward pass (FSDP automatically Reduce-Scatters the gradients)
        loss.backward()
        
        optimizer.step()
        
        if local_rank == 0:
            logger.info(f"Step {step} | Loss: {loss.item():.4f}")
            
    cleanup()

if __name__ == "__main__":
    # To run this script: 
    # torchrun --nproc_per_node=8 fsdp_trainer.py
    # train() # Commented out so it doesn't crash in standard Python execution
    pass
```

---

## Part 8 — Deployment

### Cluster Manager
- Use **Kubernetes (EKS)** with the **Volcano** or **KubeRay** scheduler, which handles gang-scheduling (ensuring all 16 nodes spin up at the exact same time; if 15 spin up and 1 is out of stock, the job fails).

### Mixed Precision
- Training in standard FP32 takes 2x the memory and is 3x slower.
- Enable `bfloat16` (BF16) via PyTorch AMP (Automatic Mixed Precision). BF16 is preferred over FP16 for large models because it has the same dynamic range as FP32, eliminating the need for gradient scaling and preventing overflow errors.

---

## Part 9 — Unit Testing

```python
import torch
import torch.distributed as dist

def test_distributed_environment_vars():
    # Unit tests for distributed code usually mock the environment
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    
    # Attempt to initialize NCCL (Will fail if no GPU, fallback to Gloo for test)
    try:
        dist.init_process_group("gloo")
        assert dist.is_initialized() == True
        dist.destroy_process_group()
    except Exception as e:
        print(f"Skipping test due to environment: {e}")
```

---

## Part 10 — Integration Testing

- **Overfit a single batch:**
  - Before kicking off a $500,000 AWS job, run the FSDP script on 2 GPUs using a single batch of data repeated 100 times.
  - Assert that the Loss reaches exactly `0.000` (or near zero).
  - If the loss doesn't drop to zero, there is a bug in the model architecture or the FSDP sharding configuration (e.g., gradients aren't syncing).

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Communication Bottleneck** | FSDP requires massive network bandwidth to share weights. If you shard across 16 nodes, the cross-node Infiniband becomes the bottleneck. **Solution:** Use Hybrid Sharding (FSDP within the 8 GPUs on a single node via NVLink, and DDP across the nodes via Infiniband). |
| **Model Size grows to 100B** | FSDP might still struggle. Introduce Pipeline Parallelism (Megatron-LM). Node 1 computes Layers 1-10, passes the activations to Node 2 which computes Layers 11-20. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| PyTorch FSDP vs Microsoft DeepSpeed | DeepSpeed (Zero-3) was the pioneer of state sharding and integrates easily via HuggingFace Accelerate. However, FSDP is now natively built into PyTorch, meaning fewer dependency conflicts, better long-term support, and native integration with PyTorch's newer features like `torch.compile`. |
| Synchronous vs Asynchronous Checkpointing | Saving 140GB to S3 every hour synchronously stalls the GPUs (wasting thousands of dollars of idle time). Asynchronous checkpointing writes the state to host RAM (CPU memory), allows the GPUs to resume training immediately, and slowly uploads from RAM to S3 in the background. |

---

## Part 13 — Alternative Approaches

1. **Tensor Parallelism (Megatron-LM):** Instead of sharding the *layers* like FSDP, Tensor Parallelism literally splits a single matrix multiplication across multiple GPUs. Extremely fast, but requires ultra-low latency connections (NVLink). It cannot be done across standard Ethernet.
2. **LoRA Fine-tuning:** If we are just fine-tuning a pre-trained open-source 10B model, do not train all weights. Freeze the model, add Low-Rank Adapters (LoRA), and train the 1% of new weights. This fits on a single GPU (DDP) and avoids FSDP entirely.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Straggler Node | Node 7 has a slightly degraded network card. It processes batches 50% slower. Because DDP/FSDP is synchronous, all other 127 GPUs sit idle waiting for Node 7 to finish. | Monitor GPU utilization (`nvidia-smi` or Datadog). If 127 GPUs show 50% utilization and 1 shows 100%, kill the straggler node immediately. Let `torchrun` replace it. |
| Loss Spike (NaN) | After 2 weeks of perfect training, the loss suddenly jumps to NaN (Not a Number) and the model is ruined. | Caused by an anomalous gradient explosion. Implement **Gradient Clipping** (`torch.nn.utils.clip_grad_norm_`). Keep the last 5 checkpoints. Roll back 3 checkpoints, skip the corrupted data batch, and resume. |

---

## Part 15 — Debugging

**Symptom:** You spin up 16 nodes (128 GPUs). The training starts, but the training speed (tokens per second) is exactly the same as when you used 8 nodes (64 GPUs). You doubled the hardware but got zero speedup.

**Debugging steps:**
1. This is a classic IO / Data Starvation problem. The GPUs are fast, but they are waiting for data.
2. Check GPU Volatile Utility. It will likely be oscillating between 0% and 100%.
3. Check the PyTorch `DataLoader`.
4. **Fix:** Increase `num_workers` in the DataLoader. Pre-fetch more batches into RAM. Ensure the S3 bucket is in the same AWS Region/Availability Zone as the EC2 instances, and upgrade the EC2 network bandwidth (ENA).

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `gpu_utilization_percent` | < 90% → GPUs are starved for data or waiting on network sync. |
| `gpu_memory_usage_gb` | > 78GB (on an 80GB card) → Dangerously close to OOM. Decrease batch size. |
| `loss_variance` | Spikes → Learning rate is too high or data is dirty. |

---

## Part 17 — Production Improvements

1. **Activation Checkpointing (Gradient Checkpointing):** To save even more VRAM, don't store the forward-pass activations in memory. Discard them, and re-compute them dynamically during the backward pass. This trades ~20% more CPU/GPU time for massive memory savings, allowing you to double the batch size.
2. **torch.compile:** Wrap the model in PyTorch 2.0's compiler. It fuses CUDA kernels together (e.g., combining Matrix Multiply and ReLU into one step), yielding a free 15-30% speedup.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"With 128 GPUs, your Global Batch Size is massive (e.g., 4096). Training with massive batch sizes often degrades the final model accuracy or causes the optimizer to fail. How do you adjust hyperparameters to fix this?"**
2. **"You realize that 20% of your S3 data is corrupted (empty files or broken parquet formatting). If the DataLoader hits a bad file, it throws an exception and crashes the entire 128-GPU cluster. How do you handle this?"**
3. **"We want to implement 'Elastic' training. During the night, we can use 32 Spot Instances (cheap). During the day, they are taken away, and we fall back to 8 On-Demand instances. How does FSDP handle the changing number of GPUs dynamically?"**

---

## Part 19 — Ideal Answers

**Q1 (Massive Batch Sizes):**
> "When scaling the global batch size by $K$, we must use the **Linear Scaling Rule**. We scale the Learning Rate by $K$ as well (e.g., if batch size doubles, learning rate doubles) to maintain the signal-to-noise ratio in the gradient updates. We also must implement a **Learning Rate Warmup** phase (gradually increasing LR over the first 5% of training) to prevent the massive LR from destabilizing the early, fragile weights."

**Q2 (Corrupted Data / DataLoader Resilience):**
> "We cannot use standard PyTorch Iterators which fail catastrophically. We must write a custom collate function or use a robust streaming library (like WebDataset). Wrap the data loading step in a `try/except` block. If a file is corrupted, log the error, skip it, and dynamically pull the next file, returning a valid tensor to the GPU so the cluster never crashes."

**Q3 (Elastic Training):**
> "Standard PyTorch DDP/FSDP cannot change world size dynamically without restarting. We use `torchrun` (Elastic). When Spot instances are preempted, `torchrun` catches the SIGTERM signal, cleanly shuts down the process group, and restarts the script with the new world size (e.g., 8). The script simply loads the latest saved checkpoint, re-shards the model across the 8 remaining GPUs, and resumes training from that exact step."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Understands the memory math (why a 10B model OOMs on 80GB).
- Clearly articulates FSDP / DeepSpeed Stage 3 (Sharding states).
- Recognizes the IO bottleneck of feeding petabytes of data from S3.
- Solves the Straggler Node and DataLoader crash problems effectively.

### Hire
- Writes a valid PyTorch distributed training loop.
- Knows what DDP and FSDP are.
- Mentions Mixed Precision (FP16/BF16).

### Lean Hire
- Suggests using `DataParallel` (DP) instead of `DistributedDataParallel` (DDP) — a classic mistake (DP is single-node, multi-thread, extremely slow and deprecated).
- Cannot estimate how much memory a parameter takes.

### Lean No Hire
- Proposes saving the Petabyte dataset to the local SSD of the GPU node.
- Solves the OOM by just "buying bigger GPUs" (ignoring that 80GB is the current ceiling).

### No Hire
- Does not know what PyTorch or Distributed Training is.
- Cannot write a basic training loop.
