# ML System Design

The gap between "model works in a notebook" and "model works in production" is where most ML interviews actually test you.

This section covers the engineering discipline of building ML systems that are reliable, scalable, maintainable, and measurable — treating ML like a product, not a science project.

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

---

## Files in this Section

| File | What it covers |
|---|---|
| [ML Engineering & Production](machine-learning-engineering.md) | Full lifecycle: data, training, evaluation, deployment, monitoring. The most comprehensive file. |
| [Building ML-Powered Applications](building-machine-learning-powered-applications.md) | Product-focused: from prototype to production, what breaks and why |
| [Design Interview Case Studies](machine-learning-design-interview.md) | Structured walkthroughs of common system design prompts |
| [ML Design Patterns](system-design/machine-learning-design-patterns.md) | Reusable architectural patterns: transform, serving, reproducibility |
| [Recommended Books](books.md) | Essential reading list for MLOps and production ML engineering |

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

**Classic example:** Training uses `user_30day_spend` computed in a batch job. Serving uses a real-time feature computed with slightly different logic. The model sees a different distribution than it was trained on.

**Prevention:**
- Feature store with a single computation path for both training and serving
- Integration tests that compare training-time and serving-time feature distributions
- Log model inputs in production, then use those logs for retraining

---

## See Also

- [ML Interview Notes: System Design](../ml-interview-notes/system-design-and-mlops.md)
- [MLOps Full Notes](../mlops.md)
- [LLM System Design](../llm-interview-notes/ai-system-design.md)
