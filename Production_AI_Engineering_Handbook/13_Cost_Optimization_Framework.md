# PART 13: COST OPTIMIZATION FRAMEWORK

## Goal
To teach candidates how to think about AI system economics and make principled decisions that reduce cost without sacrificing quality.

## Mental Model
**"Cost is a feature. A system that bankrupts the company is not a successful AI system."**
At scale, every inefficiency compounds. A 10ms latency improvement and a 20% token reduction have real dollar values.

---

## 13.1 Cost Decomposition

```text
Total AI System Cost
├── Training Cost
│   ├── Compute (GPU hours × $/hr)
│   └── Data storage + egress
├── Inference Cost (usually dominates)
│   ├── GPU/CPU compute per request
│   ├── Memory (model weights in VRAM)
│   └── Network/bandwidth
├── LLM API Cost
│   ├── Input tokens × $/M tokens
│   └── Output tokens × $/M tokens
└── Storage & Pipeline Cost
    ├── Feature Store
    ├── Vector DB
    └── Object storage (model artifacts, datasets)
```

**Key insight:** For most production systems, inference cost >> training cost. Optimize the inference path first.

---

## 13.2 Model Quantization

### Decision Tree
```text
What is the acceptable accuracy loss?
├── < 0.5% degradation acceptable → INT8 quantization (8-bit).
│   └── 4x memory reduction, 2-4x speedup.
├── < 2% degradation acceptable → INT4 quantization (4-bit, GPTQ/GGUF).
│   └── 8x memory reduction. Enables LLMs on consumer GPUs.
└── Zero accuracy loss acceptable → Quantization-Aware Training (QAT).
    └── Trains with simulated quantization. Best accuracy, highest effort.
```

| Precision | Memory vs FP32 | Speedup | Accuracy Loss | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **FP32** | 1x (baseline) | 1x | None | Training |
| **FP16/BF16** | 0.5x | 1.5–2x | Negligible | Production (GPU) |
| **INT8** | 0.25x | 2–4x | < 0.5% | CPU inference |
| **INT4 (GPTQ)** | 0.125x | 3–5x | < 2% | LLMs on single GPU |

---

## 13.3 GPU Optimization

### Utilization Framework
```text
Is GPU utilization < 60%?
├── YES → Underutilized GPU. Investigate:
│   ├── CPU bottleneck (data preprocessing, slow DataLoader) → Prefetch data async.
│   ├── Small batch size → Increase batch size.
│   └── Synchronous code blocking GPU → Profile with torch.profiler.
└── NO → GPU is well-utilized. Look elsewhere for optimization.
```

### Dynamic Batching (Inference)
```text
Instead of:
Request 1 → GPU forward pass → Response 1 (GPU idle 90% of time between requests)
Request 2 → GPU forward pass → Response 2

Use:
Requests 1, 2, 3 → Batch together → Single GPU forward pass → Responses 1, 2, 3
Throughput: 3x improvement at the cost of slightly higher individual latency.
```

---

## 13.4 LLM API Cost Optimization

### Token Reduction Strategies
| Strategy | Description | Savings |
| :--- | :--- | :--- |
| **Prompt compression** | Remove redundant words from system prompt. | 10–30% |
| **Context compression** | Summarize long conversation history. | 20–50% |
| **RAG context pruning** | Only pass the most relevant chunks (top 3, not top 10). | 30–60% |
| **Structured output** | Use JSON mode to avoid verbose explanations. | 10–20% |
| **Semantic caching** | Return cached answer for similar past queries. | 20–40% QPS reduction |
| **Model routing** | Simple queries → GPT-4o-mini. Complex → GPT-4o. | 50–80% |

### Model Routing Framework
```text
[Incoming query]
       │
       ▼
[Query complexity classifier]
├── SIMPLE (single fact, one step) → Small model (GPT-4o-mini, Claude Haiku)
├── MEDIUM (multi-step, structured) → Medium model (GPT-4o, Claude Sonnet)
└── COMPLEX (reasoning, code gen) → Large model (GPT-4o, Claude Opus)
```

Cost example: GPT-4o costs ~10x more than GPT-4o-mini. Routing 80% of traffic to the mini model cuts LLM spend by ~7x.

---

## 13.5 Caching Strategy for Cost Reduction

### What to Cache and Where

```text
MOST EFFECTIVE → Semantic cache (full LLM responses)
    └── Use: GPTCache / Redis + embedding similarity lookup.
    └── Hit rate: 20-40% for typical support chat.

MODERATE → Embedding cache
    └── Cache embeddings for static documents. Never re-embed unchanged chunks.
    └── Tools: In-memory dict, Redis.

MODERATE → Feature cache
    └── Cache computed user/item features for N minutes.
    └── Tools: Redis with TTL.

LEAST EFFECTIVE → Raw API response cache (exact string match)
    └── Only for truly identical queries.
```

---

## 13.6 Compute Cost Optimization

### Spot/Preemptible Instances
```text
Training jobs (fault-tolerant with checkpointing)?
└── YES → Use Spot (AWS) / Preemptible (GCP) instances. Save 60–80%.
    └── Checkpoint every N steps. Job restarts automatically from last checkpoint.

Inference (requires high availability)?
└── NO → Use On-demand instances for serving.
    └── Exception: Low-priority async batch inference can use Spot.
```

### Right-sizing
```text
Step 1: Profile memory usage of the serving pod.
Step 2: Set request to slightly above peak usage (not 2x).
Step 3: Set limit to 1.5x the request (burst allowance).
→ Overclaiming resources is expensive at scale.
   An overprovisioned pod on 1000 nodes wastes significant compute.
```

---

## 13.7 Vector DB Cost Optimization

### Storage Reduction
```text
Full FP32 embedding (1536 dims) = 6KB per vector
10M vectors = 60GB storage

Reduce with:
├── Binary quantization → 96x reduction (1-bit per dimension).
├── INT8 quantization → 4x reduction.
├── Dimensionality reduction (PCA to 512 dims) → 3x reduction.
└── Matryoshka embeddings → Use shorter prefixes of the embedding (2-8x reduction).
```

### Query Cost Reduction
- Use metadata filtering to reduce the search space before ANN (Approximate Nearest Neighbor) search.
- Cache embeddings for frequently queried items.
- Use batch upserts (not single upserts) to reduce write amplification.

---

## 13.8 Cold Start Cost

```text
Cold start: Spinning up a new instance for the first time.
Cost: 10-60 seconds of latency + compute.

Mitigations:
├── Keep minimum instances alive (min_replicas > 0 in K8s HPA).
├── Use container image caching (pull model weights from volume, not internet).
├── Warm pool: Pre-warm instances before peak traffic (scheduled scaling).
└── Serverless: Avoid for latency-sensitive models; use for rare/bursty workloads.
```

---

## 13.9 Cost Monitoring Dashboard

### Key Metrics to Track
| Metric | Description | Alert |
| :--- | :--- | :--- |
| **Cost per prediction** | Total infra cost / # predictions served | Rising trend > 5% week-over-week |
| **GPU utilization** | % time GPU is actively computing | < 40% (underutilized) |
| **Token cost per user** | LLM API spend / active users | Rising trend |
| **Cache hit rate** | % of requests served from cache | < 20% for cacheable workloads |
| **Spot interruption rate** | % of training jobs interrupted | > 10% (need more checkpointing) |

---

## Engineering Checklist

- [ ] Have I profiled inference latency to find the bottleneck before optimizing?
- [ ] Is dynamic batching enabled on the inference server?
- [ ] Are static embeddings (documents, items) cached and never re-computed?
- [ ] Is model quantization applied for CPU inference?
- [ ] Is there a model routing strategy (small model for simple queries)?
- [ ] Are spot instances used for training jobs with checkpointing?
- [ ] Is there a per-user token budget for LLM API calls?

## Interview Follow-up Questions & Best Answers

**Q: "Your LLM-powered feature is costing $500k/month in API calls. How do you reduce it by 70%?"**
*Best Answer:* "I would attack this from multiple angles simultaneously:
1. **Model routing (biggest impact):** Classify query complexity and route 70% of simple queries to GPT-4o-mini (~10x cheaper). Estimated saving: 50–60%.
2. **Semantic caching:** Implement a vector similarity cache (GPTCache). Support queries often repeat the same questions — cache hit rates of 30-40% are realistic. Estimated saving: 30–40% of remaining traffic.
3. **Prompt compression:** Audit the system prompt. Remove boilerplate. Compress long contexts with LLMLingua or summary buffers. Estimated saving: 15–25% per request.
4. **RAG pruning:** Reduce top-K from 10 chunks to 3-5. Estimated saving: 20-40% of context tokens.
Combined, I'd expect to hit the 70% reduction target. I'd measure each change independently using A/B testing to confirm quality is maintained."
