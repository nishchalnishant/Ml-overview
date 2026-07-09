# PART 7: DEPLOYMENT DECISION FRAMEWORK

## Goal
To teach candidates how to select the right serving infrastructure, API protocol, and deployment strategy based on latency, scale, and reliability requirements.

## Mental Model
**Deployment is not an afterthought — it IS the product.**
A model that can't be served reliably at scale has no business value. Think of deployment as shipping the model's predictions, not the model's weights.

---

## 7.1 Serving Protocol Decision Tree

```text
What is the consumer of predictions?
├── INTERNAL service-to-service → gRPC (binary protocol, faster, schema-enforced).
├── EXTERNAL API (mobile, 3rd party) → REST/HTTP (universal compatibility, human-readable).
├── Browser/Web App → REST (standard), or GraphQL for flexible querying.
└── Both internal + external → gRPC internally, REST at the edge via API Gateway.
```

| Protocol | Latency | Payload | Best For |
| :--- | :--- | :--- | :--- |
| **gRPC** | Very low | Binary (Protobuf) | Internal microservices, low-latency |
| **REST (JSON)** | Low-Medium | Text (JSON) | Public APIs, universal compatibility |
| **WebSocket** | Very low | Binary or Text | Streaming, real-time chat, live feeds |
| **Message Queue** | High (async) | Any | Decoupled, async inference (batch jobs) |

---

## 7.2 Inference Pattern Decision Tree

```text
What is the latency requirement?
├── < 100ms → Synchronous Real-time Inference.
│   └── Serve via gRPC/REST, autoscale on CPU/GPU instances.
├── 100ms – 10s → Async Inference with polling or callback.
│   └── Client sends request, gets a job ID, polls for completion.
├── Minutes to hours → Batch Inference.
│   └── Pre-compute all predictions offline, store in DB/cache.
└── Continuous stream → Streaming Inference.
    └── Kafka → ML microservice → downstream consumers.
```

| Pattern | Pros | Cons | Use Case |
| :--- | :--- | :--- | :--- |
| **Sync (Real-time)** | Instant response, fresh predictions. | Requires always-on infra, strict SLA. | Search, fraud check, content moderation. |
| **Async** | Decoupled, handles bursts, cheaper. | Added latency (polling delay). | Document processing, audio transcription. |
| **Batch** | Cheapest, highly reliable. | Stale predictions. | Nightly recommendations, bulk scoring. |
| **Streaming** | Low latency, processes events as they arrive. | Complex stateful logic, ordering issues. | Real-time analytics, live game events. |

---

## 7.3 Model Serialization & Optimization

### Decision Tree
```text
What is the inference runtime?
├── Same framework as training (PyTorch/TF) → Use native format.
│   └── Simpler but slower; lacks runtime optimizations.
├── Need cross-framework portability? → ONNX.
│   └── Convert once, deploy anywhere (different hardware, languages).
├── Need maximum GPU performance? → TensorRT.
│   └── NVIDIA-specific, fuses layers, quantizes, maximizes throughput.
└── CPU inference, serverless, mobile → ONNX Runtime / TorchScript.
```

| Format | Speed | Flexibility | Best For |
| :--- | :--- | :--- | :--- |
| **PyTorch native** | Baseline | Highest | Development, research |
| **TorchScript** | +20% | High | Production, cross-language |
| **ONNX** | +30% | Medium | Cross-framework portability |
| **TensorRT** | +200-400% | Low (NVIDIA only) | GPU inference, latency-critical |
| **llama.cpp (GGUF)** | CPU-optimized | Medium | LLM inference on CPU/edge |

---

## 7.4 CPU vs GPU Inference

### Decision Tree
```text
Model size / complexity?
├── SMALL (Logistic Regression, shallow NN, distilled model) → CPU sufficient.
│   └── Cheap, easy to scale horizontally, no driver complexity.
├── MEDIUM (BERT-base, ResNet-50) → GPU for latency, CPU for cost.
│   └── Evaluate cost per request before committing to GPU.
└── LARGE (LLMs, Stable Diffusion) → GPU required.
    └── Use batching and caching to amortize GPU cost.
```

**Key insight:** GPUs are cost-effective only when utilization is high (>60%). A single request to an idle GPU is far more expensive than the same request on CPU.

---

## 7.5 Deployment Strategies

### Decision Tree
```text
How confident are you in the new model?
├── VERY CONFIDENT → Blue-Green Deployment.
│   └── Run old (blue) and new (green) simultaneously. Switch 100% traffic instantly.
│   └── Advantage: Instant rollback by switching back.
├── MODERATELY CONFIDENT → Canary Deployment.
│   └── Send 1% → 5% → 25% → 100% traffic gradually over hours/days.
│   └── Monitor metrics at each step before promoting.
├── WANT TO VALIDATE → Shadow Deployment.
│   └── New model processes all requests, but responses are discarded.
│   └── Safe validation of latency and output quality on real traffic.
└── EXPERIMENTAL → A/B Testing.
    └── Measure business impact on a controlled user group.
```

| Strategy | Risk | Rollback Speed | Cost | Use When |
| :--- | :--- | :--- | :--- |:--- |
| **Blue-Green** | Low | Instant | High (2x infra) | High-confidence releases |
| **Canary** | Very Low | Fast | Medium | Most production releases |
| **Shadow** | Zero | N/A (no impact) | Medium | Validating new models safely |
| **Rolling Update** | Medium | Slow | Low | Stateless services |
| **A/B Test** | Low | Medium | Medium | Measuring business impact |

---

## 7.6 Scaling & Autoscaling

### Horizontal vs Vertical Scaling
```text
Can one instance handle the load if it's more powerful?
├── YES → Vertical Scaling (bigger machine). Simple, but has limits.
└── NO (traffic is distributed) → Horizontal Scaling (more instances).
    └── Requires stateless serving (store session state externally).
```

### Kubernetes (K8s) for ML Serving
- **Horizontal Pod Autoscaler (HPA):** Scale based on CPU, memory, or custom metrics (QPS, GPU utilization).
- **Resource Limits:** Always set memory and CPU limits on pods to prevent one runaway pod from killing the node.
- **GPU Resource Sharing:** Use NVIDIA MPS for time-sharing a GPU across multiple model pods.

---

## 7.7 Caching

### Decision Tree
```text
Are predictions deterministic for the same input?
├── YES → Cache predictions in Redis with TTL.
│   └── "Semantic caching" for LLMs: cache similar queries using embedding similarity.
├── PARTIAL → Cache candidate lists (batch), re-rank in real-time.
└── NO (personalized, time-sensitive) → Do not cache predictions.
    └── Cache intermediate results (embeddings, features) instead.
```

| Cache Layer | What to Cache | TTL |
| :--- | :--- | :--- |
| **Model-level** | KV cache for LLMs (attention keys/values) | Request duration |
| **Application-level** | User embeddings, item embeddings | Hours to days |
| **Prediction-level** | Full predictions for identical queries | Minutes to hours |
| **Semantic cache** | LLM responses for semantically similar prompts | Hours |

---

## 7.8 Serverless Inference

### When to Use
- **Pros:** Zero management, pay-per-request, automatic scaling to zero.
- **Cons:** Cold start latency (seconds to minutes), memory limits, no GPU support on many platforms.
- **Best For:** Low-traffic or bursty workloads where idle time is high (e.g., a developer tool that runs once a day).

---

## Engineering Checklist

- [ ] Have I defined the SLA (p99 latency target)?
- [ ] Have I set up a health check endpoint and readiness probe?
- [ ] Have I configured autoscaling with realistic load targets?
- [ ] Have I defined a rollback procedure before deploying?
- [ ] Is the serving instance stateless? (State stored in Redis/DB, not in memory)
- [ ] Have I set resource limits on the serving pods?
- [ ] Have I configured a fallback if the model is overloaded (circuit breaker)?

## Production Considerations

- **Model Registry:** Track model versions, performance, and lineage (MLflow, SageMaker Model Registry).
- **Circuit Breaker:** If the model latency exceeds SLA, automatically fall back to a simpler model or cached response rather than letting latency degrade indefinitely.
- **Graceful Shutdown:** On pod termination, finish in-flight requests before shutting down (use `terminationGracePeriodSeconds` in K8s).

## Real-world Examples

- **Twitter/X:** At peak traffic, recommendation scoring runs in batch overnight, and real-time ranking applies lightweight context adjustments (recent tweets, trending topics) on top.
- **Stripe Fraud:** Uses synchronous sub-50ms inference on a gradient boosted tree. Complex deep learning runs async and flags transactions for human review.

## Interview Follow-up Questions & Best Answers

**Q: "How would you deploy a model that serves 100k requests/second?"**
*Best Answer:* "At 100k QPS, I would: 
1. Pre-compute and cache as many predictions as possible in Redis (batch pipeline). 
2. For uncacheable requests, use a horizontally scaled fleet of lightweight CPU or GPU instances behind a load balancer. 
3. Use gRPC (not REST) for internal service calls to reduce serialization overhead.
4. Enable dynamic batching in the serving engine (TorchServe/TensorRT) to batch multiple incoming requests into a single GPU forward pass.
5. Add a circuit breaker with a fallback to the cached predictions if P99 latency exceeds SLA."
