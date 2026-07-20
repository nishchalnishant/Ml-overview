---
module: Study Plans
topic: Week 3 System Design
subtopic: Day 22 Ml System Design
status: unread
tags: [studyplans, ml, week-3-system-design-day-22-ml]
---
# Day 22: Designing ML Systems

## Why This Topic Comes Here

Weeks 1-2 covered algorithms in isolation: given a clean dataset, train a model, evaluate it. Real ML engineering does not work this way. A model that achieves excellent offline metrics but is never reliably deployed, monitored, or retrained is not a production ML system — it is a science project. Day 22 is the bridge from "building models" to "building systems." It comes after the full algorithm survey because you need to know what each algorithm costs (latency, memory, retraining frequency) before you can make sensible design tradeoffs. System design is where those costs become concrete constraints.

---

## Executive Summary: The 5-Step Framework

| Step | Goal | Key Considerations |
|------|------|-------------------|
| **1. Problem Scoping** | Define Objective | Latency, throughput, business metrics |
| **2. Data Engineering** | Build the pipeline | Features, labels, logging, join logic |
| **3. Modeling** | MVP → SOTA | Baseline first, then complex architectures |
| **4. Evaluation** | Validate | Precision/Recall, A/B testing, Shadow mode |
| **5. Deployment** | Productionize | Scaling, Monitoring, Retraining cycles |

---

## 1. Problem Scoping

**Why scoping is the most important step:** An ML system that solves the wrong problem perfectly is worse than no ML system — it consumes resources and produces confident wrong answers. Most real failures in ML production are not technical; they are misaligned objectives (optimizing CTR when the goal is user retention) or missing constraints (a fraud model that is accurate but returns results in 5 seconds on a system requiring <100ms).

**Key insight:** The success metric you optimize in your model must be causally connected to the business metric you care about. CTR is easy to optimize and poorly correlated with long-term engagement. Watch time is harder but closer to "value delivered." There is almost always a gap between what is easy to measure and what you actually want to maximize. Naming this gap explicitly is the first act of good system design.

**How to verify understanding:** A product team asks you to "improve recommendations." Before writing any code, what are the first three questions you ask? Be specific about what information each question gives you and why it determines the system design.

**What trips people up:** Jumping to model selection before defining the problem. The choice between batch and real-time inference, the acceptable latency, the labeling strategy, the cold-start handling — all of these are determined by the problem scope, not by the algorithm. Getting scoping wrong makes every subsequent decision wrong by default.

---

## 2. Scalability Considerations

### Offline vs. Online (Real-time)

- **Offline (Batch)**: Predict on all data once per day (e.g., daily recommendations). High throughput, low cost. Predictions can be precomputed and cached.
- **Online (Real-time)**: Predict on-the-fly at request time (e.g., fraud check). Requires ultra-low latency (<100ms). Model must be small or response can be asynchronous.

**Key insight:** The latency constraint determines your entire model architecture. A 200ms budget for a fraud check leaves room for one expensive model or a cascade of cheaper ones. A 10ms budget forces you toward lookup tables, pre-computed embeddings, or heavily quantized models. Decide latency first; then design the model that fits within it.

**How to verify understanding:** You need to serve personalized recommendations to 300M users with <200ms latency. You want to use a transformer-based ranker. Walk through why this is infeasible as described and what architectural change makes it feasible.

**What trips people up:** Designing models offline that cannot be served online. A model that takes 2 seconds to return a prediction can be accurate and useless simultaneously. Always prototype inference latency early, before committing to an architecture.

### Data Storage & Retrieval

- **Feature Store**: A centralized repository that stores precomputed features, serving them consistently to both training pipelines and inference services. This prevents **train-serve skew** — the bug where training and serving compute the same feature differently.

**Key insight:** Train-serve skew is the most common and hardest-to-debug production failure. The model was trained on features computed one way; the serving pipeline computes them slightly differently (different time window, different join logic, null handling) and the model silently degrades. A feature store with a shared computation layer is the architectural solution.

**How to verify understanding:** Describe a concrete scenario where a feature is computed differently in training vs. serving. What symptoms would you observe in production, and how would you diagnose the root cause?

**What trips people up:** Thinking feature stores are an MLOps nicety. They are a correctness requirement. Without a shared feature computation layer, you are relying on human discipline to keep training and serving code identical — which does not hold at scale or over time.

---

## 3. Common System Design Patterns

- **Retrieval & Ranking**: Standard in search and recommendation. Stage 1 (Retrieval) narrows billions of items to hundreds using efficient approximate nearest neighbor search. Stage 2 (Ranking) applies a complex model to order the top results. The cheap model does not need to be accurate — it needs to keep the relevant items in the candidate set.

- **Cascading Classifiers**: Use a cheap model (e.g., Logistic Regression) to filter 90% of easy cases, then a heavy model (e.g., Transformer) only for the difficult ones. This reduces average compute cost while preserving quality on hard cases.

**Key insight for cascades:** The cheap model's recall is more important than its precision. If the fast model incorrectly filters out a true positive, the expensive model never sees it. You must tune the cheap model's threshold to ensure the valuable signal is not lost at the first stage.

**How to verify understanding:** In a two-stage fraud detection cascade, the fast model (LR) has precision=0.3 and recall=0.95 at its operating threshold. The slow model (DNN) then processes the 5% that pass stage 1. Explain why recall of the first stage is the number that matters most, not precision.

**What trips people up:** Optimizing each stage of a cascade independently. The stages are not independent — the first stage determines what data the second stage sees. You must evaluate the cascade end-to-end, not stage-by-stage.

---

## 4. Full Walkthroughs

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

## 5. Interview Questions

**1. "How would you handle a system requiring real-time predictions with a slow model?"**
> Quantization/distillation → Result caching → Two-stage cascade → Async pre-computation.

**2. "What is Train-Serve Skew and how do you prevent it?"**
> Gap between training data distribution and serving distribution. Prevention: Feature Store (single pipeline), log training feature values, monitor serving distribution vs training distribution.

**3. "How do you decide between a Linear Model and a DNN?"**
> Start with the simplest baseline. Move to DNN only if: >100K samples, complex feature interactions needed, and the linear model has plateaued.

**4. "How would you design an LLM-powered feature?"**
> Latency-cost-quality triangle: (1) Define acceptable latency, (2) choose model size, (3) cache repeated queries, (4) implement fallback, (5) build eval pipeline, (6) monitor quality drift post-launch.

---

## System Checklist

- [ ] Success metric aligned with business goal?
- [ ] Monitoring strategy for data drift?
- [ ] Cold-start handling for new users/items?
- [ ] Data pipeline scalable (Spark/Flink)?
- [ ] Fallback if model fails or is slow?
- [ ] Class imbalance and calibration handled?
- [ ] Retraining frequency and trigger defined?
