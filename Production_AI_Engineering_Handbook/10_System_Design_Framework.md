# PART 10: SYSTEM DESIGN DECISION FRAMEWORK

## Goal
To teach candidates how to design scalable, reliable AI system architectures by applying distributed systems thinking to ML-specific problems.

## Mental Model
**"Design for failure, not success."**
Every component will fail. Your architecture must degrade gracefully, recover automatically, and alert you before the user notices.

---

## 10.1 The CAP Triangle for AI Systems

```text
CONSISTENCY ──────────────────── AVAILABILITY
      \                               /
       \                             /
        \       (Partition           /
         \       Tolerance)         /
          \________________________/

For AI Systems:
→ Favor AVAILABILITY over CONSISTENCY for recommendation/search (stale results are OK).
→ Favor CONSISTENCY for fraud detection/payments (wrong prediction is expensive).
```

---

## 10.2 Scalability Decision Framework

### Horizontal vs. Vertical Scaling
```text
Is the bottleneck MEMORY (model weights too large)?
├── YES → Model Parallelism (split across GPUs).
└── NO (request throughput bottleneck) → Horizontal scaling.
    ├── Stateless serving pods → Add more pods behind a load balancer.
    └── Stateful (requires session memory) → Use external state store (Redis).
```

### Scaling Patterns for AI
| Layer | Scaling Strategy |
| :--- | :--- |
| **Feature Computation** | Horizontal scale feature servers, cache in Feature Store |
| **Model Inference** | Horizontal scale CPU/GPU pods, dynamic batching |
| **Vector Search** | Shard vector index across nodes (Milvus, Qdrant distributed) |
| **LLM API** | Rate limit management, multiple provider fallback |
| **Training** | Distributed training (PyTorch DDP, DeepSpeed) |

---

## 10.3 Latency Budget Framework

### Define the End-to-End Budget First
```text
Total SLA: 200ms

Budget allocation:
├── API Gateway / Auth: 5ms
├── Feature retrieval (Feature Store): 20ms
├── Model inference: 100ms
├── Post-processing / ranking: 20ms
├── Network / serialization: 15ms
└── Buffer: 40ms

Rule: If any component exceeds its budget, it must be optimized or offloaded async.
```

### Latency Optimization Priority
1. **Cache first:** Can the result be pre-computed and cached?
2. **Compress/quantize:** Can the model be made faster?
3. **Async offload:** Can expensive operations be moved out of the critical path?
4. **Parallelize:** Can multiple operations run in parallel?

---

## 10.4 Caching Architecture

### Multi-Layer Cache Strategy
```text
L1: In-process cache (Python dict, LRU) — Sub-millisecond, limited size
     ↓ (cache miss)
L2: Redis / Memcached — 1-5ms, large capacity
     ↓ (cache miss)
L3: Object Store (S3) — 50ms+, unlimited capacity (batch results)
     ↓ (cache miss)
L4: Live model inference — 100ms+, always fresh
```

### Cache Invalidation Strategy
- **TTL-based:** Expire cache after N seconds. Simple but may serve stale data.
- **Event-driven:** Invalidate cache when the underlying data changes (Pub/Sub).
- **Versioned keys:** Include model version in cache key; new model = automatic cache miss.

---

## 10.5 Message Queues & Streaming

### Decision Tree
```text
Is processing synchronous or asynchronous?
├── SYNC → Direct API call (REST/gRPC).
└── ASYNC → Use a message queue.
    ├── Work needs to be ORDERED? → Kafka (ordered within partition).
    ├── Work needs to be DISTRIBUTED? → SQS / RabbitMQ.
    └── REAL-TIME stream processing? → Kafka Streams / Flink.
```

| System | Throughput | Ordering | Retention | Best For |
| :--- | :--- | :--- | :--- |:--- |
| **Kafka** | Very High | Per-partition | Configurable | Event streaming, high-throughput pipelines |
| **SQS** | High | No (FIFO queue available) | 14 days | Decoupled async job queues |
| **RabbitMQ** | Medium | Yes (routing) | Until consumed | Task queues, work distribution |
| **Redis Streams** | High | Yes | Configurable | Lightweight streaming, simple pub/sub |

### Kafka in ML Pipelines
```text
[Game Events] → Kafka Topic: "player_events"
                    │
                    ▼
       [Feature Engineering Service]
                    │
                    ▼
       Kafka Topic: "player_features"
                    │
           ┌────────┴─────────┐
           ▼                  ▼
  [Model Inference]    [Anomaly Detection]
           │
           ▼
  Kafka Topic: "predictions" → Downstream Consumer
```

---

## 10.6 Feature Store Architecture

### What It Solves
- **Training-serving skew:** Same feature computation logic used in training and serving.
- **Feature reuse:** Multiple models reuse the same precomputed features.
- **Low-latency serving:** Features pre-materialized, served from Redis in < 5ms.

### Feature Store Layers
```text
[Batch Layer] ─── Historical features ──── Offline Store (S3/Snowflake)
[Stream Layer] ── Real-time features ───── Online Store (Redis/DynamoDB)
                                                  │
                                                  ▼
                                        Model Inference Server
```

### Popular Feature Stores
| System | Type | Latency | Best For |
| :--- | :--- | :--- | :--- |
| **Feast** | Open-source | 1–5ms (Redis) | Startups, flexibility |
| **Tecton** | Managed | < 5ms | Enterprise, managed infra |
| **Vertex AI Feature Store** | GCP managed | < 10ms | GCP-native stacks |
| **SageMaker Feature Store** | AWS managed | < 10ms | AWS-native stacks |

---

## 10.7 Vector Database Architecture

### Sharding Strategy for Large Vector Indexes
```text
Vector DB with 1B vectors: Cannot fit on single node.
→ Shard by:
   ├── Namespace (user_id hash) → Each shard handles a subset of users.
   ├── Category (game_type) → Each shard handles a domain.
   └── Random hash → Load balanced, no semantic locality.
```

### Index Types
| Index | Build Time | Query Speed | Memory | Best For |
| :--- | :--- | :--- | :--- |:--- |
| **Flat (Brute Force)** | O(1) | O(n) | Low | Small datasets (<10k) |
| **HNSW** | Slow | Very Fast | High | Production default |
| **IVF-PQ** | Medium | Fast | Low | Large scale, memory-constrained |
| **ScaNN** | Medium | Very Fast | Medium | Google-scale retrieval |

---

## 10.8 Microservices for ML Systems

### Recommended Service Decomposition
```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Ingestion  │    │  Feature     │    │  Training    │
│   Service    │───▶│  Service     │───▶│  Service     │
└──────────────┘    └──────────────┘    └──────────────┘
                           │                    │
                           ▼                    ▼
                    ┌──────────────┐    ┌──────────────┐
                    │  Inference   │    │  Model       │
                    │  Service     │    │  Registry    │
                    └──────────────┘    └──────────────┘
                           │
                    ┌──────────────┐
                    │  Monitoring  │
                    │  Service     │
                    └──────────────┘
```

**Key principle:** Each service owns its own data store. The Inference Service never directly queries the Training Service's database.

---

## 10.9 Event-Driven Architecture for AI

### Pattern: Event Sourcing for ML
```text
Every user action produces an event → Kafka.
Events are consumed by:
├── Feature engineering pipeline (computes features).
├── Label generation pipeline (derives labels from events).
└── Real-time scoring pipeline (triggers immediate predictions).
```

**Benefit:** The system can replay historical events to retrain models or debug failures without relying on mutable database state.

---

## 10.10 Fault Tolerance

### Circuit Breaker Pattern
```text
[Inference Service]
       │
       ▼
[Circuit Breaker]
       │
  CLOSED (normal) → Requests pass through.
       │
  (Errors > threshold) → OPEN (failing fast, return fallback).
       │
  (After timeout) → HALF-OPEN (test one request).
       │
  (Success) → CLOSED again.
```

### Fallback Hierarchy
```text
[Primary: GPU inference] → fails →
[Secondary: CPU inference, smaller model] → fails →
[Tertiary: Cached recent prediction] → fails →
[Final: Static rule-based response]
```

---

## Engineering Checklist

- [ ] Have I defined SLAs for every service in the pipeline?
- [ ] Does every service have a fallback if it fails?
- [ ] Have I designed for stateless serving (state in external store)?
- [ ] Is there a circuit breaker between every dependent service?
- [ ] Have I planned for data consistency during rolling deploys?
- [ ] Is the feature computation identical between training and serving?

## Interview Follow-up Questions & Best Answers

**Q: "How would you design a system that processes 1 million game events per second and makes real-time recommendations?"**
*Best Answer:* "At 1M events/sec, this is a streaming architecture problem. 
I'd structure it as: Game clients → Kafka (partitioned by player_id) → Stream processor (Flink/Kafka Streams) that computes sliding-window features (player's actions in the last 60 seconds) → Feature Store update (Redis) → Real-time recommendation service that queries Redis for features and a fast ML model (LightGBM or small NN via ONNX Runtime). 
Responses are sent back via WebSocket or server-sent events. 
The recommendation model itself was trained offline on historical features stored in Parquet in S3, ensuring no training-serving skew by using the same feature definitions in the Feature Store."
