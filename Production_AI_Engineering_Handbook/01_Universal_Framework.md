# PART 1: UNIVERSAL AI INTERVIEW FRAMEWORK

## Goal
To provide a structured, repeatable, and scalable mental model that can be applied to any production AI engineering system design or ML system design interview, regardless of the domain (NLP, Vision, RecSys, GenAI).

## Mental Model
Think of the AI system not as a "cool model," but as an end-to-end engineering pipeline designed to solve a business problem within strict constraints (latency, cost, compliance). The model is just a single node in a complex distributed system. 

An SDE-2/Senior candidate is expected to look at the *boundaries* of the system (how data comes in, how predictions go out, how failures are handled) just as much as the internal algorithms.

## Decision Framework

The Universal Framework follows a sequential process, but in a real interview, it is iterative. 

1. **Understand Problem & Business Goal:** Why are we building this? (Revenue, engagement, cost reduction?)
2. **Clarify Requirements & Constraints:** Latency, throughput, scale, budget, privacy.
3. **Success Metrics:** Offline (AUC, F1, NDCG) vs. Online (CTR, Conversion, Session Length).
4. **Data Understanding & Pipeline:** Sources, labels, feature engineering, privacy, streaming vs. batch.
5. **Model Selection:** Baselines (Heuristics) -> Classical ML -> Deep Learning -> State-of-the-Art (LLMs/RAG).
6. **Training & Evaluation:** Data splitting, class imbalance, distributed training, offline evaluation.
7. **Deployment:** Edge vs. Cloud, API design (REST/gRPC), serving infrastructure, quantization.
8. **Monitoring & Observability:** Drift (data, concept), latency spikes, fallback mechanisms.
9. **Scaling & Future Improvements:** Handling 10x traffic, adding personalization, continuous learning.

## Flowchart (ASCII)

```text
[Business Problem]
       │
       ▼
[Requirements & Constraints] ─── (Latency, Scale, Cost)
       │
       ▼
[Metrics Definition] ─────────── (Offline vs Online KPIs)
       │
       ▼
[Data Engineering] ───────────── (Ingestion, Features, Storage)
       │
       ▼
[Model Selection] ────────────── (Heuristic -> Simple ML -> Complex AI)
       │
       ▼
[Training & Evaluation] ──────── (Loss, Optimizer, Offline Evals)
       │
       ▼
[Deployment Architecture] ────── (Batch vs Real-time, CPU vs GPU)
       │
       ▼
[Observability & Monitoring] ─── (Drift, Latency, Alerts)
       │
       ▼
[Failure Recovery] ───────────── (Fallbacks, Retries, Human-in-loop)
```

## Engineering Checklist

- [ ] Did I define the business impact before jumping into architecture?
- [ ] Did I establish hard constraints (e.g., < 100ms latency, PII data handling)?
- [ ] Did I start with the simplest possible baseline?
- [ ] Did I define the training vs. serving skew mitigation?
- [ ] Did I discuss how the model handles unseen data or missing features?
- [ ] Did I define what happens when the model goes down (fallback to heuristic)?
- [ ] Did I discuss closing the feedback loop (how do we get labels for online predictions)?

## Common Mistakes

- **Jumping to the Model:** Immediately proposing an LLM or complex Deep Learning architecture without defining the business goal or data availability.
- **Ignoring Constraints:** Proposing a massive Transformer model for an edge deployment with strict memory limits.
- **Forgetting the Feedback Loop:** Not explaining how the system will capture user feedback to improve future training cycles.
- **Confusing Offline and Online Metrics:** Optimizing for Accuracy/F1 offline, but failing to tie it to Business KPIs (e.g., Click-Through Rate).

## Interview Examples

**Scenario:** Design a toxic chat moderation system for a multiplayer game.
- *Junior Answer:* "I would use a fine-tuned BERT model to classify text as toxic or not."
- *Senior Answer:* "First, let's clarify the constraints. What is our acceptable latency? (e.g., < 50ms so chat isn't delayed). What is the scale? (e.g., 100k messages/sec). Given these constraints, a large LLM is too slow and expensive. I'd propose a cascade architecture: a fast heuristic/regex filter first, followed by a lightweight classical ML model (e.g., Naive Bayes or fastText) for >90% of traffic, and routing ambiguous cases to an optimized DistilBERT model. If the model fails or times out, we fail open (allow the message) but flag for async review."

## Tradeoffs

| Component | Option A | Option B | Tradeoff (Why choose one over the other?) |
| :--- | :--- | :--- | :--- |
| **Model** | Simple (XGBoost) | Complex (Deep Learning) | Simple models are faster, cheaper to serve, and easier to debug, but may plateau in accuracy. Complex models handle unstructured data better but require GPUs and strict latency budgets. |
| **Serving** | Real-time (API) | Batch (Cron job) | Real-time uses the freshest context but requires strict SLAs and high availability. Batch is cheap and robust but predictions might be stale. |
| **Evaluation**| Offline | Online (A/B Test) | Offline is fast and safe but doesn't guarantee real-world impact. Online proves business value but exposes users to potentially bad models. |

## Production Considerations

- **Training-Serving Skew:** Ensure the feature engineering pipeline used for training is the exact same one used in real-time inference (often solved via Feature Stores).
- **Cold Starts:** How does the system behave for new users or items with no history? (Fallback to popular items, or use content-based filtering).
- **Graceful Degradation:** If the feature store is down, can the model serve a prediction using only the request payload?

## Real-world Examples

- **Netflix Recommendation:** Uses batch processing for heavy matrix factorization (overnight), and real-time lightweight ranking (contextual bandits) when the user opens the app based on time of day.
- **Uber ETA:** Uses a combination of graph routing (heuristic/deterministic) combined with Gradient Boosted Trees (XGBoost) to adjust the ETA based on real-time traffic and historical residuals.

## Interview Follow-up Questions & Best Answers

**Q: "Your deep learning model is taking 500ms to infer, but our SLA is 100ms. How do you fix this?"**
*Best Answer:* "I would tackle this at three levels: 
1. **Model level:** Apply quantization (FP32 to INT8), distillation (train a smaller student model), or pruning. 
2. **Infrastructure level:** Use a faster serving engine like TensorRT or ONNX Runtime, and batch incoming requests dynamically. 
3. **Architecture level:** Cache frequent queries (e.g., Redis) or move some computation to an async batch process if possible."

**Q: "How do you know when your model has degraded in production?"**
*Best Answer:* "I monitor three layers:
1. **System metrics:** Latency, memory, CPU/GPU utilization.
2. **Data metrics (Feature Drift):** Distribution shifts in input features using tests like KS-statistic or comparing current distributions to training distributions.
3. **Business metrics (Concept Drift):** Tracking the actual target metric (e.g., CTR). If CTR drops but system metrics are fine, the mapping between features and the target has likely changed, triggering a retraining pipeline."
