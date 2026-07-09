# PART 17: INTERVIEW PLAYBOOK — THE 90-MINUTE BLUEPRINT

## Goal
To give candidates a precise, repeatable time allocation strategy for a 90-minute AI system design interview that maximizes signal across all evaluation dimensions.

## Mental Model
**The interviewer is evaluating five things simultaneously:**
1. **Requirement gathering** — Do you ask smart questions or dive blindly?
2. **Architecture** — Can you design systems that actually work at scale?
3. **Engineering depth** — Do you understand *why* things work, not just *what* they are?
4. **Tradeoffs** — Can you make principled decisions under constraints?
5. **Communication** — Can you explain complex systems clearly?

---

## Phase 1: Requirement Gathering (Minutes 0–10)

### Goal
Extract the key constraints that will dictate every architectural decision. Never skip this phase.

### Minute-by-Minute Script

**[0:00 – 0:30]** Repeat the problem back in your own words.
> "So we're designing a system to predict player churn for EA's live games. Let me make sure I understand the scope before I dive in..."

**[0:30 – 5:00]** Ask focused clarifying questions (max 6–8).

**Prioritized question list (pick the most relevant 6):**
1. "What is the primary business goal? Are we optimizing for retention rate, or revenue from returning players?"
2. "What is the scale? How many daily active users, and what QPS do we expect?"
3. "What is the latency requirement — does churn prediction need to be real-time, or can we batch overnight?"
4. "Do we have labeled historical data? What are the available data sources?"
5. "What is our failure tolerance? What happens if the model goes down?"
6. "Are there privacy or compliance constraints on player data (GDPR)?"
7. "What is the approximate budget? Are GPU-based serving costs acceptable?"
8. "How do we close the feedback loop — how do we measure if an intervention worked?"

**[5:00 – 8:00]** Summarize constraints.
> "Let me summarize what I've heard. We're building a churn prediction system for 50M daily active players across multiple titles. We need predictions 24 hours in advance (batch is acceptable), and the primary metric is 30-day player reactivation rate after an intervention. We have 3 years of historical event data in S3. Privacy: EU player data must stay in EU. Does that sound right?"

**[8:00 – 10:00]** Define success metrics.
> "For offline evaluation, I'll use PR-AUC since churn is imbalanced. For online, I'll track intervention CTR and 7-day reactivation rate via A/B test."

### Checkpoint (End of Phase 1)
- [ ] Have I confirmed business goal, scale, latency, data availability, and constraints?
- [ ] Have I defined offline and online success metrics?
- [ ] Have I summarized back to the interviewer and gotten confirmation?

---

## Phase 2: Architecture (Minutes 10–20)

### Goal
Present a high-level system design that addresses the stated constraints. Start simple, then layer complexity.

### Minute-by-Minute Script

**[10:00 – 12:00]** Draw the high-level architecture (talk while drawing).
> "Let me sketch the three main components. First, the data pipeline: raw game events flow from Kafka into our data lake in S3. Second, the feature engineering layer computes player-level features using Spark. Third, the model training and serving pipeline..."

**[12:00 – 16:00]** Walk through the architecture end-to-end.
> "Here's the data flow: Game events → Kafka → Spark (feature engineering, daily batch) → Feature store → XGBoost training job → MLflow registry → Batch scoring job → Predictions stored in Redis → Downstream intervention service reads predictions and sends targeted notifications."

**[16:00 – 18:00]** Justify each major component choice in one sentence.
> "Kafka because it's the industry standard for event streaming and decouples producers from consumers. XGBoost because the features are tabular and it's state-of-the-art with SHAP interpretability. MLflow for experiment tracking and model versioning. Redis for low-latency prediction lookup by the intervention service."

**[18:00 – 20:00]** Acknowledge complexity you haven't addressed yet.
> "I've sketched the happy path. I'll address training-serving skew, drift monitoring, and the cold start problem for new players as we go deeper."

### Checkpoint (End of Phase 2)
- [ ] Is the architecture drawn on the whiteboard/diagram?
- [ ] Have I covered: data ingestion, feature engineering, training, serving, and monitoring at high level?
- [ ] Have I justified each component choice in one sentence?
- [ ] Have I flagged open questions to address in the next phase?

---

## Phase 3: Deep Implementation (Minutes 20–60)

### Goal
Demonstrate engineering depth. This is where strong SDE-2 candidates separate from average ones.

### Recommended Deep-Dive Order (by impact)

**[20:00 – 30:00] Feature Engineering (highest impact)**
- What features matter most? (Recency, frequency, monetary — RFM features).
- How are time-based features computed without leakage?
- How is the feature store updated? What's the TTL?
- Training-serving skew prevention?

**[30:00 – 40:00] Model Selection & Training**
- Why XGBoost over alternatives? (Tree-based = better on tabular data, handles missing values, SHAP interpretable).
- How do you handle class imbalance? (scale_pos_weight, PR-AUC as metric).
- Validation strategy? (Temporal split: train on months 1–24, validate on months 25–30, test on month 31–36).
- Hyperparameter tuning? (Optuna, 50 trials, early stopping on validation PR-AUC).

**[40:00 – 50:00] Serving & Deployment**
- Batch inference pipeline (Spark + ONNX Runtime).
- Model versioning strategy (promote from staging → production in MLflow registry).
- Deployment strategy: Canary (test new model on 5% of players before full rollout).
- Rollback trigger: If reactivation rate drops > 5%, auto-rollback.

**[50:00 – 60:00] Edge Cases & Depth Probes**
- **Cold start (new player):** No history → popularity-based fallback by game genre.
- **Cross-title players:** Aggregate features across all EA titles.
- **Concept drift:** Monthly retrain triggered if PSI > 0.2 on key features.
- **GDPR compliance:** EU player features computed and stored in EU region. Delete pipeline for right-to-erasure requests.

### Checkpoint (End of Phase 3)
- [ ] Did I explain feature engineering end-to-end without skipping steps?
- [ ] Did I explain my model choice, hyperparameter strategy, and evaluation approach?
- [ ] Did I explain how the model gets from training to serving?
- [ ] Did I address at least 2–3 edge cases?

---

## Phase 4: Deployment & Monitoring (Minutes 60–75)

### Goal
Show that you think beyond the model. Deployment and monitoring demonstrate production maturity.

### Minute-by-Minute Script

**[60:00 – 65:00] Deployment strategy**
> "For the initial rollout, I'd use a Canary strategy: route 5% of player interventions to the new model's predictions, 95% to the existing system. I'd monitor: reactivation rate (business KPI), prediction score distribution (data quality), and pipeline latency (SLA). After 7 days with stable metrics, I'd ramp to 100%."

**[65:00 – 70:00] Monitoring infrastructure**
> "Three monitoring layers:
> 1. **Infrastructure (Prometheus/Grafana):** CPU/GPU, batch job success/failure, pipeline latency.
> 2. **Data quality (Evidently AI):** Feature drift (PSI), null rate monitoring, schema validation.
> 3. **Business (custom dashboard):** 7-day reactivation rate, intervention CTR, weekly revenue recovery."

**[70:00 – 75:00] Failure recovery**
> "If the batch job fails overnight: the intervention service falls back to the previous day's predictions (stale by 24 hours but acceptable for this use case). If drift is detected: automated alert triggers a retrain. If business metrics drop significantly: auto-rollback to the previous model version in the registry."

---

## Phase 5: Tradeoffs, Improvements & Follow-ups (Minutes 75–90)

### Goal
Show strategic thinking. The best candidates transition from "here's what I built" to "here's what I'd do with more time and resources."

### Minute-by-Minute Script

**[75:00 – 80:00] Key tradeoffs you made**
> "The main tradeoffs in this design:
> 1. **Batch vs. real-time:** I chose batch (24h predictions) for simplicity and cost. Real-time would catch sudden churn events but requires 10x the infrastructure cost.
> 2. **XGBoost vs. Sequence model:** XGBoost is simpler and works well. A Transformer over session sequences would capture temporal patterns better, but requires significantly more engineering effort and training compute.
> 3. **Single model vs. per-title:** A single model across all EA titles is simpler to maintain but may underperform for titles with very different playstyles. Given bandwidth, I'd start single-model and segment later."

**[80:00 – 85:00] What I'd improve with more time**
> "Three improvements, prioritized:
> 1. **Short-term:** Add a session-level sequence model (LSTM/Transformer) on top of the XGBoost to capture temporal decay in engagement patterns. I expect 3–5% AUC improvement.
> 2. **Medium-term:** Move from batch to near-real-time (hourly predictions) using Spark Structured Streaming instead of batch Spark, enabling faster intervention response.
> 3. **Long-term:** Build a multi-armed bandit system for intervention optimization — instead of a single notification strategy, dynamically learn which intervention (discount, new content teaser, social nudge) works best for each player segment."

**[85:00 – 90:00] Anticipated follow-up questions (prep answers)**
- "How would you handle A/B testing for interventions?"
- "What if the intervention itself causes churn?"
- "How would you extend this to cross-game churn prediction?"
- "How would you reduce training time if the dataset grows to 100B rows?"

---

## The Interviewer Evaluation Rubric (What They're Scoring)

| Dimension | Strong Signal | Weak Signal |
| :--- | :--- | :--- |
| **Requirements** | Asks 6-8 targeted questions, summarizes constraints. | Dives into architecture without asking. |
| **Architecture** | Starts simple, layers complexity, justifies each component. | Names buzzwords without justification. |
| **Engineering depth** | Explains *why* (tradeoffs, principles), not just *what*. | Describes what technology does, not why it's chosen. |
| **Failure handling** | Proactively addresses failures, drift, and edge cases. | Only describes the happy path. |
| **Communication** | Concise, structured, checks in with interviewer. | Monologue, doesn't read the room. |
| **Tradeoffs** | Names tension, picks a side, justifies, offers alternative. | "It depends" without resolving the tradeoff. |

---

## Quick Recovery Techniques

**If you go blank:** "Let me take 10 seconds to think through this systematically." [Pause, then use a framework.]

**If you took a wrong architectural path:** "Actually, I want to revise that. I mentioned X, but given our 200ms SLA, Y is the better choice because [reason]."

**If the interviewer seems bored:** "I notice I've been spending a lot of time on feature engineering — should I move to the serving layer, or dive deeper here?"

**If you run out of time:** "In the interest of time, let me summarize the most important points: [3 decisions, 3 tradeoffs, 3 monitoring signals]."

---

## Pre-Interview Ritual (30 minutes before)

```text
5 min  → Review the SJTF answer template (Part 15).
5 min  → Review the universal decision tree (Part 3).
5 min  → Review your top 3 EA-specific domain scenarios (Part 16).
5 min  → Review the deployment strategies (Part 7).
5 min  → Practice one 60-second system summary out loud.
5 min  → Write the constraint checklist on paper: scale, latency, cost, privacy, data.
```

---

## The One-Page Cheat Sheet (Before Any Interview)

```text
CONSTRAINTS TO EXTRACT:         ARCHITECTURE LAYERS:
□ Business goal                 □ Data ingestion
□ Scale (DAU, QPS)              □ Feature engineering
□ Latency SLA                   □ Training pipeline
□ Data availability             □ Model serving
□ Privacy/compliance            □ Monitoring/drift detection
□ Budget/cost                   □ Failure recovery

MODEL SELECTION DEFAULTS:       METRICS BY TASK:
□ Tabular → XGBoost/LGBM        □ Classify → AUC-PR (imbalanced)
□ Text → DistilBERT/FastText    □ Rank → NDCG@K
□ Generation → LLM + RAG        □ Regress → RMSE or MAE
□ Seq data → LSTM/Transformer   □ GenAI → RAGAS, Human eval

DEPLOYMENT DEFAULTS:            MONITORING DEFAULTS:
□ Batch → Spark + ONNX          □ Infra → Prometheus/Grafana
□ Real-time → gRPC + K8s        □ Drift → Evidently AI / PSI
□ Release → Canary              □ Business → Custom dashboard
□ Rollback → MLflow registry    □ Alerts → P99, PSI>0.2, KPI-5%
```
