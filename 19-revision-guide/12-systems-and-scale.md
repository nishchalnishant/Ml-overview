---
module: Systems and Scale
topic: Revision Guide
subtopic: ""
status: unread
tags: [revision, systems-and-scale, compression, quantization, llm-serving, inference-ops, cost-optimization, distributed-training, reliability]
prerequisites: [12-systems-and-scale]
---
# 12 Systems and Scale — Revision Guide

Single-page revision for the entire `12-systems-and-scale/` folder: model compression, quantization/pruning, LLM serving, inference operations, cost optimization, and distributed training/serving reliability. Use the outline to reconstruct the folder's structure from memory, then self-test with the graded Q&A below.

## Structure Overview

```markdown
- **12-systems-and-scale**
  - **Model Compression**
    - Deployment gap
    - Quantization overview
      - PTQ
      - QAT + STE
      - GPTQ Hessian
      - AWQ salient channels
    - Knowledge distillation
      - Teacher-student soft labels
      - Temperature softmax
      - Intermediate/FitNets
      - Attention transfer
      - Self-distillation
      - Born-again networks
    - Pruning
      - Unstructured pruning
      - Structured pruning
      - Magnitude pruning
      - Lottery ticket hypothesis
    - Low-rank factorization
      - SVD compression
      - LoRA
    - Speculative decoding
    - Prune-distill-quantize pipeline
    - QLoRA
  - **Quantization & Pruning Detail**
    - Uniform quantization math
      - Scale/zero-point
      - Symmetric vs asymmetric
    - Granularity
      - Per-tensor
      - Per-channel
      - Per-group g=128
    - RTN baseline
    - GPTQ algorithm
      - Column-by-column
      - Error compensation
    - AWQ algorithm
      - Scale search
    - QAT STE mechanics
    - QLoRA + NF4
      - Double quantization
    - Format comparison
      - fp32/bf16/fp8/int8/int4/NF4
    - Movement pruning
    - Distillation loss KL
  - **LLM Serving**
    - Prefill vs decode
      - Compute-bound prefill
      - Bandwidth-bound decode
      - Roofline analysis
    - KV cache problem
      - KV cache size formula
      - GQA reduction
      - Memory fragmentation
    - vLLM PagedAttention
      - Block tables
      - Copy-on-write
      - Prefix caching
    - Continuous batching
      - Iteration-level scheduling
    - Serving quantization
      - GPTQ/AWQ/GGUF/FP8/SmoothQuant
    - TensorRT-LLM
      - Kernel fusion
      - In-flight batching
    - Triton Inference Server
    - Tensor parallelism serving
      - Column/row parallel
    - Throughput vs latency knobs
    - Deployment patterns
      - Single-node
      - Multi-node PP+TP
      - Ray Serve
  - **LLM Inference Ops**
    - TTFT / TPOT / E2E
    - Prefill compute analysis
    - Decode bandwidth analysis
    - KV cache capacity planning
    - PagedAttention block manager
    - Continuous batch scheduler
    - Chunked prefill
    - Speculative decoding math
      - Expected accepted tokens
    - Model routing tiers
    - Capacity planning worked example
  - **Cost Optimization**
    - Training cost
      - Spot instances
      - Mixed precision
      - Gradient checkpointing
    - Inference cost
      - Quantization
      - Dynamic batching
      - Cascade serving
      - Result caching Redis
    - Storage cost
      - Tiered storage
      - Parquet compression
    - Cost monitoring
      - Cost per quality unit
  - **Distributed Training & Serving Reliability**
    - Data parallelism
      - All-reduce gradients
    - Model parallelism
      - Tensor parallelism
      - Pipeline parallelism
      - Pipeline bubbles 1F1B
    - ZeRO stages / FSDP
    - Gradient checkpointing
    - Mixed precision
    - Communication collectives
      - All-reduce/gather/scatter/broadcast
      - Ring vs tree topology
    - Checkpointing strategy
      - Sync vs async
      - Elastic training
    - Serving patterns
      - Real-time/batch/async/cascade
    - Autoscaling
      - Cold start problem
    - Reliability controls
      - Canary rollout
      - Shadow traffic
      - Circuit breakers
      - Blue-green deploy
    - Observability
      - Latency/throughput/error/quality
      - Drift types
    - Failure modes
    - Worked sizing example
```

## Questions

### Easy

1. Why is a trained model "over-specified" for inference, and how does that motivate compression?
2. What is the difference between FP16 and INT8 post-training quantization in terms of memory reduction?
3. Define calibration data in the context of PTQ. Why do you need it?
4. What does QAT do differently from PTQ, and what training technique makes it possible?
5. What is knowledge distillation, and what is a "soft label"?
6. What role does temperature play in the distillation softmax?
7. What is the difference between structured and unstructured pruning?
8. State the Lottery Ticket Hypothesis in one or two sentences.
9. What is low-rank factorization, and how does LoRA use it for fine-tuning?
10. What problem does speculative decoding solve, and what are the two models involved called?
11. Define TTFT and TPOT.
12. Why is LLM prefill described as compute-bound and decode as memory-bandwidth-bound?
13. What does the KV cache store, and why is it needed?
14. What is PagedAttention, and what OS concept does it borrow from?
15. What is continuous batching, and how does it differ from static batching?
16. Name three techniques to reduce training cost in the cloud.
17. What is cascade serving, and why does it save cost?
18. What is the difference between data parallelism and model parallelism?
19. What is gradient checkpointing, and what trade-off does it make?
20. What is a canary rollout, and how does it differ from shadow traffic?
21. What are the four "golden signals" to monitor for an ML serving system?
22. What is a pipeline bubble in pipeline parallelism?
23. What does GQA stand for, and how does it shrink the KV cache?

### Medium

1. Why does per-group quantization (g=128) outperform per-tensor quantization for LLM weights, and what does it cost?
2. Explain why GPTQ produces less quantization error than naive round-to-nearest (RTN) at 4-bit.
3. How does AWQ differ conceptually from GPTQ, and why can it be faster to apply?
4. Why does the order of the "prune → distill → quantize" pipeline matter? What breaks if you reorder it?
5. Explain how QLoRA enables fine-tuning a 70B model on a single 48GB GPU. What exactly gets updated during backprop?
6. Why is unstructured pruning often useless for latency improvements on standard GPUs?
7. Compare the Iterative Magnitude Pruning procedure to Movement Pruning — what signal does each use to decide what to prune?
8. Why does the arithmetic intensity of decode change as batch size increases, and at what point does decode become compute-bound?
9. Explain how PagedAttention eliminates internal and external fragmentation compared to naive contiguous KV allocation.
10. Why does GQA reduce KV cache size, and what's the corresponding trade-off versus full multi-head attention?
11. Compare tensor parallelism and pipeline parallelism in terms of communication pattern and typical network placement (NVLink vs InfiniBand).
12. Why does chunked prefill reduce TTFT, and what does it cost in terms of TPOT?
13. Derive intuitively why the expected number of accepted tokens per speculative-decoding step is (1 − α^(K+1)) / (1 − α).
14. Explain why decode throughput scales near-linearly with batch size only up to a point.
15. Walk through why dynamic batching cuts cost with no quality trade-off, while quantization and cascading do involve one.
16. Explain the difference between ZeRO-1, ZeRO-2, and ZeRO-3 in terms of what's sharded and the communication cost incurred.
17. Why is bf16 generally preferred over fp16 for mixed-precision training?
18. What's the reasoning behind checkpoint interval being tied to MTBF of the training cluster?
19. Why must the "quality" signal be monitored separately from latency/error-rate in an ML serving dashboard? Give an example failure it catches that the others miss.
20. Explain why fraud-scoring endpoints should generally avoid result caching, even though caching is cheap and effective elsewhere.
21. You need to reduce TTFT for a chatbot with a very long shared system prompt. What levers do you have, and what does each cost?
22. Why does tensor parallelism efficiency depend on being confined within a single node?
23. Explain why "cost per quality unit" is a better optimization target than absolute cost when picking a model size to deploy.
24. What causes pipeline bubbles, and how does the 1F1B (PipeDream) schedule reduce them?

### Hard

1. GPTQ's error compensation is fit to the calibration distribution X. What happens to quantization quality when the serving distribution diverges from the calibration set, and why doesn't RTN suffer the same failure mode?
2. GPTQ quantizes columns left-to-right, compensating remaining columns via the inverse Hessian. Why can error still pile up in the later columns, and how does AWQ sidestep this specific failure?
3. You're told "INT4 quantization only loses 0.5–1% accuracy on average benchmarks." What does that aggregate number hide, and what specific capabilities degrade disproportionately?
4. Under what serving conditions does INT4 quantization actually become *slower* than INT8 or FP8, despite using fewer bytes per weight?
5. Derive why decode is bandwidth-bound at batch=1 but becomes compute-bound near batch≈500 on an A100, using the arithmetic intensity formula, and explain why real servers rarely reach that regime.
6. If you quantize weights to INT4 but leave the KV cache in fp16, why doesn't decode get a clean 4× speedup, especially at long context lengths?
7. Explain the tension between admitting more long-context requests and more short-context requests into a fixed 80GB GPU's KV cache — why is this a bin-packing problem rather than a smooth trade-off, and how does PagedAttention relieve (but not eliminate) it?
8. Chunked prefill improves TTFT for a new request. Precisely whose latency is being "stolen" to pay for that improvement, and why is this a reallocation rather than a free win?
9. Prefix caching for shared system prompts can silently stop helping or even reduce serving capacity. Explain both failure modes.
10. Why does TP's reliance on frequent all-reduce make it intolerant of InfiniBand-class latency, while PP tolerates it — connect this to each method's communication frequency, not convention.
11. Design a serving architecture for a 70B-parameter LLM that must hit p99 < 2s at minimum cost. Justify your choices for parallelism strategy, quantization, batching, and reliability controls, and state the cost per request.
12. You have a fixed compute-to-communication ratio target for data-parallel training. Derive why adding GPUs eventually makes wall-clock training slower, and identify which lever (batch size, gradient compression, precision) buys back the most headroom.
13. Walk through the full memory budget for fine-tuning a 7B model with Adam on 8×A100-80GB GPUs, and show the sequence of optimizations (mixed precision, ZeRO-2, gradient checkpointing) required to fit it, with approximate per-GPU numbers at each step.
14. A distributed training job's loss spikes at step 50k. Walk through your full diagnostic and mitigation procedure, and explain the criterion for deciding whether to roll back to a checkpoint versus letting training continue.
15. Explain why ZeRO-3/FSDP is described as "trading communication for memory," and construct a scenario where enabling it actually hurts overall throughput.
16. A capacity-planning exercise sizes GPU count using only decode tokens/sec, ignoring prefill entirely. Under what traffic pattern does this undercount the required fleet, and how would you correct the estimate?
17. Speculative decoding gives near-zero speedup on creative generation but 3-4x on reasoning/code. Explain the acceptance-rate mechanism behind this gap, and describe what happens to overall latency if you naively apply a large K to creative-generation traffic.
18. You silently reduce a model's precision to INT4 to cut serving cost by 4x, but a downstream fraud-detection pipeline consuming its outputs starts missing 5% more fraud. Explain why "cost per quality unit" was the correct framework to have used up front, and how you'd instrument the system to catch this before ship instead of after.
19. Compare a canary rollout to shadow traffic for validating a new model version at a high-stakes company. Under what conditions does canary's cost-efficiency backfire — i.e., when does 1-5% traffic exposure still cause unacceptable harm before you can detect and roll back?
20. Explain precisely why "GPU utilization" is a lagging indicator for autoscaling decisions, and construct a scenario where scaling on GPU utilization alone leads to an SLA violation that queue-depth-based scaling would have caught earlier.
