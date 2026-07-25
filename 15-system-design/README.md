---
module: System Design
topic: System Design
subtopic: ""
status: unread
tags: [productionml, ml, system-design-index]
---
# ML System Design — Case Study Index

Structured prompts and full-system walkthroughs. Start with [ML Engineering & Production](03-machine-learning-engineering.md) for lifecycle framing, then [Design Interview Case Studies](01-design-interview-framework.md) for whiteboard rhythm.

## How This Section Differs

- This folder is the **interview-facing case study layer**.
- [13-production-ml](../13-production-ml/README.md) is the **production mechanics layer**.
- [16-interview-prep](../16-interview-prep/README.md) is the **drill and recall layer**.
- If a question is about “how would you structure the answer,” use the framework files here. If it is about “how does the system actually work in prod,” go back to `13-production-ml`.

## What Makes This Section Staff-Level

The goal is not to memorize one architecture. The goal is to explain:

- why one design is better than another under a latency or cost constraint
- how data flows through the system end to end
- what breaks first at scale
- where to place observability and rollback points
- how to defend a tradeoff under pressure

## Core guides

| Guide | Focus |
| --- | --- |
| [ML Engineering & Production](03-machine-learning-engineering.md) | End-to-end lifecycle |
| [Building ML-Powered Applications](05-building-ml-powered-applications.md) | Product → production gaps |
| [Design Interview Case Studies](01-design-interview-framework.md) | Common prompts + framework |
| [ML Design Patterns](04-design-patterns.md) | Reusable architecture patterns |
| [Production Diagnosis Flowchart](../13-production-ml/10-production-diagnosis-flowchart.md) | When things break in prod |

## Case studies by domain

| Case study | Domain |
| --- | --- |
| [Ad CTR Prediction](cases/01-ad-ctr-prediction.md) | Ads / ranking |
| [News Feed Ranking](cases/08-news-feed-ranking.md) | Social feeds |
| [Search Ranking System](cases/09-search-ranking.md) | Search |
| [End-to-End Recommendation System](cases/11-recommendation-system.md) | Recsys |
| [Video Recommendation](cases/12-video-recommendation.md) | Video recsys |
| [Personalization System](cases/10-personalization.md) | Personalization |
| [Real-Time Bidding System](cases/04-real-time-bidding.md) | Ads auction |
| [Fraud Detection (Full System)](cases/05-fraud-detection.md) | Fraud |
| [Customer LTV Prediction](cases/02-customer-ltv-prediction.md) | Growth / finance |
| [Clinical ML System](cases/06-clinical-ml-system.md) | Healthcare |
| [Content Moderation System](cases/07-content-moderation.md) | Trust & safety |
| [ETA & Surge Pricing](cases/03-eta-and-surge-pricing.md) | Marketplace / logistics |

## Platform & infra patterns

| Topic | File |
| --- | --- |
| Feature store architecture | [feature-store-architecture.md](../13-production-ml/07-feature-store-architecture.md) |
| Embedding pipeline design | [embedding-pipeline-design.md](../13-production-ml/08-embedding-pipeline-design.md) |
| Model registry & versioning | [model-registry-versioning.md](../13-production-ml/09-model-registry-versioning.md) |
| Data engineering for ML | [data-engineering-for-ml.md](../13-production-ml/06-data-engineering-for-ml.md) |
| Streaming ML pipeline | [streaming-ml-pipeline.md](../13-production-ml/05-streaming-ml-pipeline.md) |
| Real-time ML systems | [real-time-ml-systems.md](../13-production-ml/04-real-time-ml-systems.md) |
| Distributed training & serving reliability | [distributed-training-and-serving-reliability.md](../12-systems-and-scale/06-distributed-training-and-serving-reliability.md) |
| Cost optimization | [cost-optimization.md](../12-systems-and-scale/05-cost-optimization.md) |
| A/B testing & experimentation | [ab-testing-experimentation.md](../04-evaluation/06-ab-testing-experimentation.md) |
| LLM inference ops | [llm-inference-ops.md](../12-systems-and-scale/04-llm-inference-ops.md) |

## See also

- [Production ML overview](../README.md)
- [MLOps](../13-production-ml/01-mlops.md)
- [ML interview: system design](../13-production-ml/11-system-design-and-mlops.md)
- [LLM interview: system design](../10-llms/interview-notes/07-ai-system-design.md)

## Canonical Reading Order

1. [ML Engineering & Production](03-machine-learning-engineering.md)
2. [Design Interview Case Studies](01-design-interview-framework.md)
3. The relevant case study under [cases/](cases/)
4. [Production ML](../13-production-ml/README.md) for operational depth
5. [Interview Prep](../16-interview-prep/README.md) for fast recall and drill practice
