# Day 22: Designing ML Systems

## Executive Summary: The 5-Step Framework
Designing an ML system is not just about choosing a model; it's about the entire lifecycle.

| Step | Goal | Key Considerations |
|------|------|-------------------|
| **1. Problem Scoping** | Define Objective | Latency, throughput, business metrics |
| **2. Data Engineering** | Build the pipeline | Features, labels, logging, join logic |
| **3. Modeling** | MVP $\rightarrow$ SOTA | Baseline first, then complex architectures |
| **4. Evaluation** | Validate | Precision/Recall, A/B testing, Shadow mode |
| **5. Deployment** | Productionize | Scaling, Monitoring, Retraining cycles |

---

## 1. Scalability Considerations

### Offline vs. Online (Real-time)
- **Offline (Batch)**: Predict on all data once a day (e.g., daily recommendations). High throughput, low cost.
- **Online (Request)**: Predict on-the-fly (e.g., fraud check). Harder to scale, requires ultra-low latency (<100ms).

### Data Storage & Retrieval
- **Feature Store**: A centralized repo to store and serve features for both training and serving, ensuring **feature consistency**.

---

## 2. Common System Design Patterns
- **Retrieval & Ranking**: Common in Search/RecSys. Stage 1 (Retrieval) narrows down billions of items to hundreds. Stage 2 (Ranking) uses a complex model to order the top results.
- **Cascading Classifiers**: Use a cheap model (e.g., Logistic Regression) to filter out 90% of easy cases, then a heavy model (e.g., Transformer) for the difficult ones.

---

## 3. Full Walkthroughs

### Walkthrough 1: Design a Content Recommendation System (Netflix/YouTube scale)

**Problem scoping:**
- Goal: maximize watch time (not clicks — avoids clickbait)
- Scale: 300M users, 500M items, 100M daily active users
- Latency requirement: <200ms for homepage load

**Architecture (two-tower + ranking):**
```
Stage 1 — Retrieval (ANN search):
  User + Item embeddings → dot product → top-500 via FAISS (~20ms)

Stage 2 — Ranking:
  User-item features + context → GBM or wide-and-deep → calibrated scores (~80ms)

Stage 3 — Re-ranking (business rules):
  Diversity, freshness, sponsored slots (~10ms)
```

**Monitoring:** daily drift check on watch probability; alert if p50 engagement drops >5% week-over-week.

---

### Walkthrough 2: Design a Fraud Detection System

**Architecture:**
```
Transaction → Rule engine (0ms) → XGBoost score (~10ms) → DNN sequence model (~30ms)
→ Combined score → Block / Review / Allow
```

**Key decisions:**
- XGBoost for interpretability (regulatory audit, dispute resolution)
- DNN only for hard cases XGBoost is uncertain about
- Calibration: ensure P(fraud | score=0.8) ≈ 0.8
- Daily retrain — fraud patterns shift fast (adversarial)

---

### Walkthrough 3: Design a RAG-based Enterprise Search

**Architecture:**
```
Query → Query Rewriting (LLM) → BM25 + Dense Retrieval → RRF fusion
→ Cross-encoder Reranker (top-50 → top-10)
→ LLM Generation with Citations
```

**Key trade-offs:**
- Hybrid retrieval (sparse + dense) covers keyword-exact and semantic needs
- Cross-encoder reranker adds 100-200ms but significantly improves precision
- Cache frequent query embeddings to reduce latency

---

## 4. Interview Questions

**1. "How would you handle a system requiring real-time predictions with a slow model?"**
> Quantization/distillation → Result caching → Two-stage cascade → Async pre-computation.

**2. "What is Train-Serve Skew and how do you prevent it?"**
> Gap between training data distribution and serving distribution. Prevention: Feature Store (single pipeline), log training feature values, monitor serving distribution vs training distribution.

**3. "How do you decide between Linear Model and DNN?"**
> Start with simplest baseline. Move to DNN only if: >100K samples, complex feature interactions needed, and linear model has plateaued.

**4. "How would you design an LLM-powered feature?"**
> Latency-cost-quality triangle: (1) Define acceptable latency, (2) choose model, (3) cache repeated queries, (4) implement fallback, (5) build eval pipeline, (6) monitor quality drift post-launch.

---

## System Checklist
- [ ] Success metric aligned with business goal?
- [ ] Monitoring strategy for data drift?
- [ ] Cold-start handling for new users/items?
- [ ] Data pipeline scalable (Spark/Flink)?
- [ ] Fallback if model fails or is slow?
- [ ] Class imbalance and calibration handled?
- [ ] Retraining frequency and trigger defined?
