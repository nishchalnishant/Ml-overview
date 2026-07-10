# Interview 28 — Distributed Model Training Infrastructure (Condensed)

Design distributed training infra for a proprietary 10B-parameter foundation model trained on petabytes of EA gameplay/player telemetry across hundreds of GPUs — must handle sharding, data throughput, and fault tolerance for a multi-week run.

## Clarifying Questions to Ask
- Hardware topology / interconnect? → 8x A100 80GB per node, NVLink intra-node, Infiniband inter-node.
- Does the model fit on one GPU? → No — FP32 weights alone are 40GB; Adam states/gradients push it to ~140GB vs 80GB card.
- Where/how is data stored? → Petabytes in S3 as Parquet (not GPU-friendly for streaming).
- What happens when a GPU dies mid-run? → Must survive without restarting the whole 3-week job.
- Is this pretraining or fine-tuning an existing open model? → Changes whether FSDP or LoRA is the right call.

## Core Architecture
```
S3 (Parquet → WebDataset tar shards)
   → Streaming DataLoader (prefetch, multi-worker)
   → Training Cluster (16 nodes x 8 A100, FSDP shards weights/grads/optimizer state)
   → Async checkpoint to S3 (hourly, via host RAM buffer)
   → torchrun elastic supervisor (detects node failure, reprovisions, resumes)
```
- **FSDP (or DeepSpeed Zero-3)** — mandatory because standard DDP replicates full model per GPU and OOMs instantly at 10B params.
- **BF16 mixed precision** — same dynamic range as FP32, no gradient scaling needed, ~2x memory/speed win over FP32.
- **WebDataset streaming** — Parquet parsing can't keep 8 A100s fed; tar-shard streaming from S3 removes the CPU bottleneck.
- **torchrun elastic** — handles NCCL group death on node failure without killing the whole job.
- **Hybrid sharding** — FSDP within a node (NVLink), DDP-style sync across nodes (Infiniband) to avoid cross-node bandwidth collapse.

## Talking Points That Signal Seniority
- State the memory math unprompted: 10B params × 2 bytes (BF16) + Adam master/momentum/variance (FP32) ≈ 140GB > 80GB card.
- Proactively flag the IO/data-starvation risk of streaming petabytes from S3 before being asked.
- Mention gang-scheduling (Volcano/KubeRay) — job must not partially start if 1 of 16 nodes isn't ready.
- Call out the straggler-node failure mode: synchronous SGD means 127 idle GPUs waiting on 1 slow node.
- Propose async checkpointing to host RAM instead of blocking on synchronous S3 writes (saves GPU idle $).
- Bring up activation/gradient checkpointing as a lever to increase batch size without more GPUs.
- Note `torch.compile` as a near-free 15–30% speedup via kernel fusion.
- Recognize LoRA as the right alternative if this is fine-tuning, not pretraining from scratch.

## Top 3 Tradeoffs
- **FSDP vs DeepSpeed Zero-3** — DeepSpeed pioneered state sharding, but FSDP is native to PyTorch, fewer dependency conflicts, better `torch.compile` integration.
- **Tensor parallelism vs FSDP** — tensor parallel (Megatron) splits individual matmuls for speed but needs NVLink-grade latency, can't cross standard Ethernet; FSDP is more network-tolerant.
- **Sync vs async checkpointing** — sync checkpointing of 140GB to S3 stalls GPUs (expensive idle time); async writes to host RAM first and uploads in background.

## Toughest Follow-ups
**Q: Global batch size hits ~4096 across 128 GPUs — how do you keep training stable?**
A: Apply the Linear Scaling Rule — scale learning rate proportionally with batch size — plus an LR warmup over the first ~5% of steps so the larger LR doesn't destabilize early fragile weights.

**Q: 20% of S3 Parquet files are corrupted and crash the DataLoader on all 128 GPUs — how do you handle it?**
A: Never let a bad file propagate an uncaught exception; wrap the read/collate step in try/except, log and skip corrupted shards, and pull the next valid file so a valid tensor always reaches the GPU — the cluster must never crash on dirty data.

**Q: You want elastic training that shrinks from 32 spot GPUs to 8 on-demand GPUs when spot capacity is reclaimed — how does FSDP handle a changing world size?**
A: FSDP/DDP can't resize live; use torchrun Elastic — it catches the SIGTERM on preemption, tears down the process group cleanly, and relaunches with the new world size, reloading the latest checkpoint and re-sharding across whatever GPU count remains.

## Biggest Pitfall
Proposing to solve the OOM by just "buying bigger GPUs" or saving the petabyte dataset to local SSD instead of naming FSDP/sharding and streaming — signals no grasp of why an 80GB ceiling is real and immovable.
