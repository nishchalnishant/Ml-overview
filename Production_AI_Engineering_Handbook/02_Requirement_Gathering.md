# PART 2: REQUIREMENT GATHERING FRAMEWORK

## Goal
To teach candidates how to extract ambiguous requirements, establish engineering constraints, and demonstrate product sense before designing an AI system.

## Mental Model
Treat the interviewer as a stakeholder who has a business problem but doesn't fully understand the technical limitations. Your job is to extract the "SLAs" (Service Level Agreements) that will dictate your architecture. 
*If you don't ask about latency, you can't justify why you chose XGBoost over a massive LLM.*

## Decision Framework

Break requirement gathering into three pillars:
1. **Business & User Impact** (Why are we doing this? Who is it for?)
2. **Data & Privacy Constraints** (What do we have? Are we allowed to use it?)
3. **System & Scale Constraints** (How fast? How big? How much money?)

## Decision Tree

```text
Are the requirements clear?
├── YES -> Validate assumptions with the interviewer.
└── NO -> Ask clarifying questions.
    ├── What is the primary business objective? (Revenue, Engagement, Trust?)
    ├── What is the scale? (DAU, QPS, Total Data Size)
    ├── What is the latency requirement? (Real-time vs Async vs Batch)
    └── What is the budget/cost constraint? (GPU inference allowed?)
```

## Flowchart (ASCII)

```text
[Interviewer prompt: "Build a recommendation system"]
       │
       ▼
[Clarify Business Goal] --> "Is this to increase click-through rate, or total watch time?"
       │
       ▼
[Clarify Scale & Traffic] --> "How many users? What is the peak QPS?"
       │
       ▼
[Clarify Latency SLA] --> "Does this need to render in < 200ms?"
       │
       ▼
[Clarify Data Availability] --> "Do we have historical interaction data? Are there cold-start users?"
       │
       ▼
[Summarize Constraints] --> "I will design a system for 10M DAU, 5k QPS peak, <200ms latency, using historical logs."
```

## Engineering Checklist

- [ ] **Business Goal:** Have I defined the exact optimization objective?
- [ ] **Expected Scale:** DAU (Daily Active Users), QPS (Queries Per Second), Data volume (TB/PB)?
- [ ] **Latency SLA:** Batch (hours), Async (seconds), Real-time (milliseconds)?
- [ ] **Accuracy vs. Interpretability:** Do we need to explain the prediction to the user (e.g., Healthcare, Finance)?
- [ ] **Privacy & Compliance:** GDPR, CCPA, HIPAA? Can we store PII?
- [ ] **Data Availability:** Do we have labels? Are they delayed?
- [ ] **Failure Tolerance:** What happens if the AI fails? (Fail open, fail closed, fallback?)
- [ ] **Budget/Cost:** Are GPUs feasible for serving, or must it run on cheap CPUs?

## Common Mistakes

- **Assuming the Data Exists:** Designing a supervised learning system without asking how labels are acquired.
- **Ignoring the Cold Start:** Forgetting to ask what happens when a new user joins or a new item is added.
- **Missing the SLA:** Designing an architecture that takes 2 seconds to infer when the feature is an autocomplete bar that requires <50ms.

## Interview Examples (Excellent Clarifying Questions)

**Prompt:** "Design an AI system to detect fraudulent transactions."
- *Average Question:* "What kind of machine learning model should I use?"
- *Senior Question:* "What is our tolerance for False Positives vs False Negatives? If we block a legitimate transaction, we ruin user trust (costly). If we allow a fraudulent one, we lose money. What is the business priority?"
- *Senior Question:* "Do we have real-time labels for fraud, or are chargebacks delayed by 30-90 days? This will dictate how we handle concept drift."
- *Senior Question:* "Can we block the transaction synchronously, meaning we have a strict <100ms budget to run the model, or do we allow the transaction and flag it asynchronously for review?"

## Tradeoffs

| Constraint | Tradeoff Analysis |
| :--- | :--- |
| **Real-time vs Batch** | Real-time is responsive to immediate context but requires expensive, highly-available infra. Batch is cheap and stable but operates on stale data. |
| **Precision vs Recall** | High precision minimizes user annoyance (fewer false alarms) but catches fewer issues. High recall catches everything but increases operational overhead (more manual reviews). |
| **Cloud vs Edge** | Cloud allows massive models and centralized updates. Edge (on-device) guarantees zero network latency, works offline, and preserves privacy, but severely limits model size and complexity. |

## Production Considerations

- **Delayed Labels:** In many systems (e.g., ad clicks, fraud), you don't know the "true label" immediately. You must design data pipelines that join features at prediction time with labels that arrive days later.
- **Cost of Inference:** A model that costs $0.05 per inference is useless if the transaction value is $0.01.

## Real-world Examples

- **Credit Card Fraud:** Transactions are evaluated in < 50ms using highly optimized tree-based models on CPUs, while complex graph neural networks run asynchronously to update risk profiles.
- **Social Media Feed:** Cannot wait 5 seconds to load. Uses a two-stage pipeline: a fast, cheap candidate generator (batch/cached), and a slightly heavier real-time ranker that runs in parallel across shards.

## Interview Follow-up Questions & Best Answers

**Q: "You mentioned we need to handle 10,000 requests per second. How does that change your choice of model?"**
*Best Answer:* "At 10k QPS, running a deep neural network synchronously on every request will require a massive fleet of GPUs, which might destroy our profit margins. I would shift to a lighter model (like XGBoost or LightGBM) that can be served efficiently on CPUs, or I would redesign the architecture to pre-compute predictions in batch and serve them from a low-latency key-value store like Redis, falling back to a lightweight real-time model only when necessary."
