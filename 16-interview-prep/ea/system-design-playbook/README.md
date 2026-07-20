# EA AI System Design Playbook

An interview-prep playbook for **SDE-2 / Principal-level AI system design interviews at EA** (Electronic Arts). It covers 23 realistic AI/ML systems spanning LLM-based products, gameplay/player ML, telemetry pipelines, and ML platform infrastructure — the kind of systems an EA interviewer would ask you to design end-to-end on a whiteboard. All 23 chapters below are written and populated (audited at 658–879 lines each).

## The recurring 46-section template

Every chapter follows the same skeleton so structure only needs to be learned once:

1. **Requirements (1–6):** problem framing, functional/non-functional requirements, clarifying questions to ask the interviewer, assumptions, capacity estimation.
2. **Architecture (7–8):** high-level architecture, low-level component breakdown.
3. **APIs & data layer (9–17):** API design, database design, caching, queues/async, streaming, model serving, feature store, vector database, embedding pipelines.
4. **ML-specific pipelines (18–21):** end-to-end inference request lifecycle, training pipelines, retraining strategy, drift detection.
5. **Ops & reliability (22–27):** monitoring, alerting, logging, security, authentication, rate limiting.
6. **Deployment strategy (28–37):** autoscaling, cost optimization, disaster recovery, multi-region deployment, blue/green, canary, rollback, observability, Kubernetes, Terraform.
7. **Why / tradeoffs / closing (38–46):** why this architecture, alternative architectures, tradeoffs, failure modes, scaling/latency/cost bottlenecks, interview follow-up questions, and ideal answers to those questions.

Sections 38–46 are the highest-signal part for interview prep — they're where you argue tradeoffs instead of just describing a design.

## Chapters

### LLM & Agentic Systems
- [RAG Platform](systems/rag-platform.md) — retrieval-augmented generation over enterprise/game content
- [Enterprise Search](systems/enterprise-search.md) — hybrid lexical+semantic search over internal docs/assets
- [Chatbot](systems/chatbot.md) — conversational assistant with memory, safety, and tool use
- [AI Coding Assistant](systems/ai-coding-assistant.md) — code completion/agentic coding tool architecture
- [Agent Platform](systems/agent-platform.md) — multi-step autonomous agent orchestration
- [MCP Ecosystem](systems/mcp-ecosystem.md) — Model Context Protocol tool/server integration architecture

### Gameplay & Player ML Systems
- [Recommendation Engine](systems/recommendation-engine.md) — in-game/store content recommendations
- [Matchmaking Engine](systems/matchmaking-engine.md) — skill-based real-time player matching
- [Player Churn Prediction](systems/player-churn-prediction.md) — early-warning churn scoring pipeline
- [Toxicity Detection](systems/toxicity-detection.md) — real-time chat/voice moderation
- [Personalization Engine](systems/personalization-engine.md) — per-player content/UX personalization
- [Fraud Detection](systems/fraud-detection.md) — payment/account fraud scoring
- [Cheat Detection](systems/cheat-detection.md) — anti-cheat signal pipeline

### Data & Telemetry Systems
- [Live Game Analytics](systems/live-game-analytics.md) — real-time dashboards over gameplay events
- [Real-Time Telemetry Platform](systems/real-time-telemetry-platform.md) — high-throughput event ingestion and stream processing
- [Streaming ML](systems/streaming-ml.md) — online feature/model updates on event streams

### ML Platform & Infra
- [Feature Store Platform](systems/feature-store-platform.md) — offline/online feature parity and serving
- [ML Platform](systems/ml-platform.md) — end-to-end training/serving infra
- [Model Registry](systems/model-registry.md) — versioning, lineage, and promotion of models
- [Experiment Platform](systems/experiment-platform.md) — A/B testing and experimentation infra for ML
- [Batch Inference](systems/batch-inference.md) — large-scale offline scoring pipelines
- [Real-Time Inference](systems/real-time-inference.md) — low-latency online model serving
- [Continuous Training Platform](systems/continuous-training-platform.md) — automated retraining triggered by drift/data freshness

## How to use this playbook

1. Read one chapter end-to-end, including APIs and data layer — don't skip to tradeoffs.
2. Close the file and redo the **interview follow-up questions** section from memory, out loud.
3. Compare your answers against the chapter, note gaps.
4. Move to the next chapter. Revisit weak chapters after finishing a category.

## Recurring cross-system patterns

Themes that show up across most chapters — recognizing them lets you transfer answers between systems in an interview:

- **Canary + shadow evaluation before promotion** — new models/agents run alongside production traffic before taking live decisions.
- **Feature store for train/serve parity** — shared offline/online feature computation to prevent skew.
- **Drift-triggered retraining** — continuous training kicks off on data/concept drift signals, not just a fixed schedule.
- **Multi-region active-active serving** — latency-critical inference paths replicate across regions with local failover.
- **Human-in-the-loop escalation** — moderation/fraud/cheat systems route low-confidence cases to human review rather than auto-acting.
- **Registry-gated deployment** — model registry acts as the single source of truth gating what can be promoted to serving.
