---
module: Production ML
topic: ML System Design
subtopic: ""
status: unread
tags: [productionml, ml, ml-system-design]
---
# ML System Design

The gap between "model works in a notebook" and "model works in production" is where most ML interviews actually test you.

This section covers building ML systems that are reliable, scalable, maintainable, and measurable — treating ML like a product, not a science project.

## How To Use This Section

- Use this folder for the **production mechanics**: MLOps, deployment, monitoring, governance, and diagnosis.
- Use [15-system-design](../15-system-design/README.md) for the **whiteboard interview flow** and case-study rhythm.
- Use [16-interview-prep](../16-interview-prep/README.md) for the **drill layer** once the concepts are clear.
- If two files seem to overlap, prefer the one with the more specific job:
  - `01-mlops.md`, `02-deployment-patterns.md`, `03-model-governance.md` = platform and process
  - `10-production-diagnosis-flowchart.md` = incident triage
  - `11-system-design-and-mlops.md` = interview-ready production walkthrough

---

## What ML System Design Interviews Test

At senior levels, the interviewer wants to see:

1. **Scoping** — Can you define the problem clearly? What metric matters? What's the latency budget?
2. **Data thinking** — How do you get, clean, label, and version data?
3. **Modeling strategy** — Simple baseline → iterate. Can you justify complexity?
4. **Offline → Online gap** — Why does the model perform worse in production?
5. **Monitoring & feedback loops** — How do you know when it breaks?
6. **Trade-offs** — Accuracy vs latency, freshness vs cost, precision vs recall

The answer "it depends" is always correct — but only if you say what it depends on.

### What good senior answers include

- the first failure mode you expect in production
- the observability signal you would monitor first
- how you would roll back safely
- what you would do if the label arrives late or never arrives
- how the design changes when traffic, latency, or cost changes

---

## Files in this Section

| File | What it covers |
|---|---|
| [System Design Index](README.md) | Entry point for production ML depth and interview routing |
| [MLOps Full Notes](01-mlops.md) | CI/CD for ML, pipelines, monitoring, retraining triggers |
| [Deployment Patterns](02-deployment-patterns.md) | Rollout strategies, latency optimization, serving patterns |
| [Model Governance](03-model-governance.md) | Registry, versioning, approval gates, audit trails, regulatory compliance |
| [Distributed Training and Serving Reliability](../12-systems-and-scale/06-distributed-training-and-serving-reliability.md) | Training at scale, serving reliability, bottlenecks, failure recovery |
| [ML Engineering & Production](../15-system-design/03-machine-learning-engineering.md) | Full lifecycle: data, training, evaluation, deployment, monitoring. Best single-file summary. |
| [Building ML-Powered Applications](../15-system-design/05-building-ml-powered-applications.md) | Product-focused: from prototype to production, what breaks and why |
| [Design Interview Case Studies](../15-system-design/01-design-interview-framework.md) | Structured walkthroughs of common system design prompts |
| [ML Design Patterns](../15-system-design/04-design-patterns.md) | Reusable architectural patterns: transform, serving, reproducibility |
| [Interview Questions](../16-interview-prep/study-plans/interviewquestions.md) | Tiered Easy/Medium/Hard Q&A bank across MLOps, serving, and system design |
| [Recommended Books](../18-resources/01-books-production-ml.md) | Essential reading list for MLOps and production ML engineering |

---

## Common Design Interview Prompts

- Design YouTube's recommendation system
- Design a fraud detection system
- Design a search ranking system
- Design a spam classifier for email
- Design a real-time bidding system
- Design a content moderation system
- Design an NLP-powered customer support bot
- Design a ride-surge pricing model

---

## The Framework That Works

For any ML system design question, walk through:

```
1. Problem Framing
   - What are we optimizing? (business metric → proxy metric → ML objective)
   - Constraints: latency, throughput, freshness, cost

2. Data
   - Sources, volume, labeling strategy
   - Train/val/test splits, potential leakage
   - Feature engineering plan

3. Modeling
   - Start simple (logistic regression → gradient boosting → neural network)
   - Justify complexity with data volume and latency

4. Evaluation
   - Offline: precision/recall/AUC/NDCG (whichever fits)
   - Online: A/B test, shadow mode

5. Deployment
   - Batch vs real-time inference
   - Model versioning, rollback plan

6. Monitoring
   - Data drift, concept drift, prediction distribution shifts
   - Feedback loops and how to close them
```

---

## The Train-Serve Skew Problem

The most common production failure in ML.

**What it is:** Model trained on historical data; serving pipeline computes features differently or with a time lag.

**Example:** Training uses `user_30day_spend` computed in a batch job. Serving computes it in real time with slightly different logic. The model sees a different distribution than it was trained on.

**Prevention:**
- Feature store with a single computation path for both training and serving
- Integration tests that compare training-time and serving-time feature distributions
- Log model inputs in production, then use those logs for retraining

## Staff-Level Lens

When a question sounds simple, push it one layer deeper:

- If the model is too slow, ask whether the problem is batching, feature lookup, model size, or network hops.
- If quality drops, ask whether the issue is labeling, drift, skew, or rollout mismatch.
- If cost spikes, ask whether the issue is overprovisioning, inefficient retraining, or unbounded inference traffic.
- If the system is noisy, ask whether the fix is better monitoring, better data contracts, or a stricter rollout strategy.

## Canonical References

If you want the most complete treatment of a topic, use these as the source of truth:

- [MLOps](01-mlops.md)
- [Deployment patterns](02-deployment-patterns.md)
- [Model governance](03-model-governance.md)
- [Real-time ML systems](04-real-time-ml-systems.md)
- [Streaming ML pipeline](05-streaming-ml-pipeline.md)
- [Data engineering for ML](06-data-engineering-for-ml.md)
- [Feature store architecture](07-feature-store-architecture.md)
- [Embedding pipeline design](08-embedding-pipeline-design.md)
- [Model registry and versioning](09-model-registry-versioning.md)
- [Production diagnosis flowchart](10-production-diagnosis-flowchart.md)
- [System design and MLOps](11-system-design-and-mlops.md)

---

## See Also

- [ML Interview Notes: System Design](11-system-design-and-mlops.md)
- [MLOps Full Notes](01-mlops.md)
- [LLM System Design](../10-llms/interview-notes/07-ai-system-design.md)
- [Distributed Training and Serving Reliability](../12-systems-and-scale/06-distributed-training-and-serving-reliability.md)
