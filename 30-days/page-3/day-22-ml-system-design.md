# Day 22: Designing ML Systems

## Executive Summary: The 5-Step Framework
Designing an ML system is not just about choosing a model; it's about the entire lifecycle.

| Step | Goal | Key Considerations |
|------|------|--------------------|
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

## Interview Questions

**1. "How would you handle a system that requires real-time predictions but uses a very slow model?"**
> 1. Use **Model Quantization** or Distillation to speed it up. 2. Implement **Result Caching** for frequent queries. 3. Use an **Asynchronous Architecture** or a two-stage pattern (cheap filter first).

**2. "What is 'Train-Serve Skew' and how do you prevent it?"**
> It's when the model behaves differently in production vs. training. To prevent it: 1. Use a single pipeline for data (Feature Store). 2. Monitor prediction distributions. 3. Ensure no data leakage (future data in training).

**3. "How do you decide between a simple Linear Model and a Deep Neural Network for a new system?"**
> Always start with the **simplest baseline**. If linear regression hits the business metrics, don't use deep learning. Only move to complex models if you have enough data and traditional methods have plateaued.

---

## System Checklist
- [] Is the success metric aligned with the business goal?
- [] Do we have a monitoring strategy for data drift?
- [] How will we handle cold-start problems (new users/items)?
- [] Is the data pipeline scalable (Spark/Flink)?
