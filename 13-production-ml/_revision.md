---
module: Production ML
topic: Revision Card
subtopic: ""
status: unread
tags: [production-ml, mlops, revision, cheatsheet]
---
# Production ML — 10-Minute Revision Card

---

## Mental Model: ML in Production = Software + Decay

Software doesn't degrade without code changes. ML models do — because the world changes but training data doesn't. Everything in MLOps addresses this asymmetry.

**The loop:** Data → Train → Evaluate → Deploy → Monitor → (drift detected) → Retrain → repeat

---

## MLOps Core Checklist

| Phase | Must-have | Common failure |
|-------|-----------|----------------|
| Data | Version datasets, validate schema | Training-serving skew (different preprocessing) |
| Training | Reproducible runs (seed, env), experiment tracking | Can't reproduce "the model from Tuesday" |
| Evaluation | Offline + online metrics, sliced eval | Aggregate metric hides subgroup failure |
| Deployment | Canary → full rollout, rollback plan | Big-bang deploys with no rollback |
| Monitoring | Data drift + model drift + latency + errors | Silent degradation caught weeks later |
| Retraining | Trigger on drift or schedule | Stale model, no retrain pipeline |

---

## Deployment Strategies

| Strategy | What | Rollback | Cost | Use when |
|----------|------|----------|------|---------|
| Big-Bang | Replace old model entirely | Hard — restart required | Low | Internal tools, low-stakes |
| Blue-Green | Two full environments, flip traffic | Instant (flip switch back) | 2× infra | High-stakes, need instant rollback |
| Canary | Route 5% → monitor → ramp up | Easy (reduce canary %) | Low extra | Standard production deploy |
| Shadow | Run new model in parallel, don't serve | N/A | 2× compute | Validate before any exposure |
| A/B Test | Split traffic for metric comparison | Route to A | Low | When you need causal impact proof |

**Canary ramp:** 5% → 25% → 50% → 100%, 24h hold at each step. Monitor: latency p50/p99, error rate, business metric.

**Gotcha:** Canary too small → low statistical power. Canary on skewed segment (mobile only) → results don't generalize.

---

## Training-Serving Skew — The Silent Killer

**Definition:** model trained with preprocessing pipeline A, served with pipeline B → different feature distributions → performance collapse.

**Common causes:**
- Feature computed differently in batch (training) vs real-time (serving)
- Missing value imputation with training-time mean instead of online mean
- Different normalization statistics
- Feature available at training time, unavailable at inference (data leakage or latency)

**Fix:** single feature pipeline shared by training and serving (feature store). Log and compare training feature distributions vs serving feature distributions.

---

## Data Drift vs Concept Drift

| Type | What changes | Example | Fix |
|------|-------------|---------|-----|
| **Data drift** (covariate) | Input distribution $P(X)$ | Users shift to mobile, text length changes | Retrain or adapt |
| **Concept drift** | Label relationship $P(Y|X)$ | "good credit" definition changes over time | Retrain on recent data |
| **Label drift** | Output distribution $P(Y)$ | Fraud rate spikes seasonally | Weight recent data, monitor |
| **Upstream drift** | Feature pipeline changes | Data source schema change | Schema validation + alerting |

**Detection:**
- Statistical tests: KS test (continuous), chi-squared (categorical), PSI (Population Stability Index)
- PSI > 0.25 → significant drift → retrain

---

## Feature Stores

**Problem:** same features computed multiple times across teams + training-serving skew + no reuse.

**Feature store solves:**
- **Offline store:** historical features for training (batch, e.g. Hive/BigQuery)
- **Online store:** low-latency feature serving at inference (e.g. Redis, DynamoDB)
- **Point-in-time correctness:** when joining features to labels, only use features available *before* the label event (prevents leakage)

**Key gotcha:** point-in-time join is mandatory. Without it, you use future information at training time → impossible eval metrics → production failure.

---

## Model Registry & Versioning

**Model artifact = weights + config + preprocessing + metadata**

**Stages:** Staging → Validation → Production → Archived

**What to version:** model weights, training code, dataset version, evaluation metrics, hyperparameters, environment (Docker image).

**Without this:** "which model is in production?" becomes unanswerable.

---

## Monitoring Stack

| Signal | Tool | Alert on |
|--------|------|---------|
| Latency | APM (Datadog, Prometheus) | p99 > SLA threshold |
| Error rate | APM | 5xx rate > baseline |
| Prediction distribution | Custom logging | Distribution shift vs training baseline |
| Feature drift | Evidently AI, Arize | PSI > 0.25 |
| Business metric | Analytics | Drop in conversion, click-through |
| Ground truth (delayed) | Label pipeline | Precision/recall below threshold |

**Gotcha:** ground truth is often delayed (churn label available 30 days later). Use leading indicators (prediction confidence distribution, upstream feature drift) as early warnings.

---

## Serving Patterns

| Pattern | Latency | Throughput | Use |
|---------|---------|-----------|-----|
| Synchronous REST | Low (ms) | Medium | Real-time, user-facing |
| Async batch | High (minutes) | Very high | Non-real-time, bulk scoring |
| Streaming (Kafka + model) | Medium | High | Event-driven pipelines |
| Edge inference | Device-local | N/A | Privacy, offline, low-latency |

**LLM serving specifics:**
- Continuous batching: new requests join mid-batch → GPU always busy
- Paged attention (vLLM): KV cache in non-contiguous pages → no memory fragmentation
- Speculative decoding: 2-3× latency win for long outputs

---

## Experiment Tracking Checklist

Every experiment must log:
- [ ] Dataset version + split strategy
- [ ] Hyperparameters (all, not just tuned ones)
- [ ] Random seeds
- [ ] Training loss curve + validation metrics
- [ ] Evaluation on held-out test set (never tune on this)
- [ ] Model artifact location
- [ ] Hardware / compute used

**Tools:** MLflow, Weights & Biases, Neptune, Comet.

---

## System Design Patterns (Quick Reference)

**Two-tower retrieval:** separate encoders for query and item → offline encode items, online encode query → ANN search (FAISS/ScaNN). Used in Google, YouTube recommendations.

**Real-time ML system:** feature computation → feature store → model service → prediction → logging → label join → retrain trigger.

**Fraud detection:** streaming features (last 1h transactions) + batch features (historical profile) → ensemble model → decision + explanation → case management.

---

## Interview Quick-Draws

**"How do you detect model degradation in production?"**
→ Three signals in priority order: (1) business metric drop (source of truth, delayed), (2) prediction distribution shift (early, no labels needed), (3) feature drift (earliest, upstream). Set alerts on all three at different thresholds.

**"What's training-serving skew?"**
→ Model trained with one preprocessing, served with another → different feature distributions → silent accuracy drop. Fix: shared feature pipeline (feature store) and distribution comparison between training features and serving features.

**"Blue-green vs canary?"**
→ Blue-green: two full environments, atomic traffic switch, instant rollback, 2× infra cost. Canary: gradual ramp starting at 5%, cheap, catch issues early, rollback = route more to stable version.

**"How do you do A/B testing for ML models?"**
→ Split traffic randomly (user-level for consistency), run until statistical significance (power analysis first), monitor both primary metric and guardrails. Validate randomization before launch (AA test).

**"What's point-in-time correctness?"**
→ When joining features to training labels, only use feature values available *before* the label event. Without it, you leak future information → impossibly good offline metrics → production collapse.

**"When do you retrain?"**
→ Trigger on: (1) drift detected (PSI > threshold), (2) performance degradation (below SLA), (3) scheduled (weekly/monthly baseline), (4) new training data available. Automate the trigger, not just the training.
