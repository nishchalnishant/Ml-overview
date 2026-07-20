---
module: Interview Prep
topic: Ml
subtopic: System Design And Mlops
status: unread
tags: [interviewprep, ml, ml-system-design-and-mlops]
---
**Primary reference:** [Production ML](README.md) | [System Design](../15-system-design/03-machine-learning-engineering.md)

# ML System Design and MLOps

## What This File Is For

ML system design questions test a different competency than algorithm questions. The interviewer is not asking whether you know what a transformer is. They are asking whether you can decompose a production ML system into components, identify where failures occur, and reason about the tradeoffs that real engineers face when deploying models at scale.

The structure for each topic:
1. What the interviewer is actually testing — the underlying competency
2. The reasoning structure — why first-principles thinkers approach it this way
3. The pattern in action — a worked example
4. Common traps — where people go wrong and why

---

# 1. The ML System Design Frame

## What the interviewer is actually testing

Whether you have a systematic method for scoping and decomposing an ML system before diving into model architecture. The most common failure mode is jumping to model selection before establishing what problem the system is solving.

## The reasoning structure

A production ML system has five components that must be designed explicitly:

1. **Goal and metric** — what does success look like in business terms, and what measurable ML metric proxies it?
2. **Data and labels** — where does training data come from, at what freshness, with what labeling process?
3. **Training pipeline** — how does the model get built, validated, and versioned?
4. **Serving path** — how do features get computed at prediction time, and how does the model run?
5. **Monitoring and retraining loop** — how does the system detect degradation and recover?

The interview trap is treating these as sequential steps when they are actually coupled. The serving path constrains what features you can use (latency), which constrains what you can train on. Define the constraints early or your architecture will not work in production.

**Design order for interviews:**

1. Define goal and success metric
2. Define constraints (latency budget, throughput, cost)
3. Define data sources and label pipeline
4. Define training pipeline (frequency, validation gates)
5. Define serving path (real-time vs batch, feature computation)
6. Define monitoring and drift handling
7. Define rollout strategy and rollback plan

## The pattern in action

**Design a content recommendation system for a video platform.**

Goal: increase watch time. Metric: expected watch time per session (proxy for user satisfaction).

Constraints: sub-100ms recommendation latency, 10M items in catalog, 50M DAU.

**Why two-stage retrieval-ranking is necessary:** 100ms is too short to score 10M items with a heavy model. Two stages solve this.

Stage 1 — Retrieval: approximate nearest neighbor search on learned user and item embeddings. FAISS index returns top 500 candidates in ~10ms. Retrieval optimizes recall; it does not need to be perfect.

Stage 2 — Ranking: a DNN scores 500 candidates using richer features (user history, item metadata, context). Returns top 20. 50–80ms budget.

Feature computation: user embedding and item embeddings are precomputed in batch (hourly), stored in a feature store. Real-time features (current session context, last item watched) are computed at request time and joined in the serving layer.

Monitoring: track CTR and watch time per recommendation position. Feature drift on user embeddings signals distribution shift. PSI > 0.25 on a key feature triggers investigation.

## Common traps

**Designing the model before the serving path.** A complex cross-attention ranker that takes 500ms to run cannot meet a 100ms SLA. The serving constraints come first.

**Not asking about label availability.** Watch time is observable with a delay. A user watches a video for 3 minutes — you know this 3 minutes later. Your training loop must account for this delayed feedback.

**Ignoring feedback loops.** A recommendation system's outputs become the next round of training data. Items the model does not recommend get less engagement data, which makes the model less likely to recommend them, which reduces their data further. This popularity bias amplifies over time.

---

# 2. Retrieval and Ranking: Two-Stage Systems

## What the interviewer is actually testing

Whether you understand why a single heavy model cannot serve a full catalog at query time, and whether you can reason about the recall-precision tradeoff between stages.

## The reasoning structure

**The fundamental tension.** Heavy models are more accurate but slower. Scoring every candidate with a heavy model is O(n × model cost). At n = 10M items and 100ms latency budget, this is computationally infeasible.

**Two-stage solution:**
- **Retrieval (recall-oriented)**: fast, approximate method that finds a small candidate set. Misses some good candidates, but fast. Optimized for recall.
- **Ranking (precision-oriented)**: accurate, heavy model scores only retrieved candidates. Can use expensive features. Optimized for precision on the candidate set.

**Retrieval methods:**
- Vector search / approximate nearest neighbors (FAISS, ScaNN): precomputed embeddings, fast similarity search
- Collaborative filtering (item-item or user-item): based on historical co-engagement
- Inverted index: keyword or tag matching for content retrieval
- Rules-based: freshness filters, content type constraints

**Ranking models:**
- Gradient-boosted trees (fast, interpretable, good for tabular features)
- Deep neural networks (can incorporate embedding features, cross-features)
- Cross encoders (most accurate, most expensive — joint encoding of query and candidate)
- DCN (Deep & Cross Network): explicitly models feature interactions

## The pattern in action

**Search system at an e-commerce company.**

Retrieval: BM25 (text matching) + vector search on product embeddings. Returns top 200 candidates in ~5ms.

Ranking: a gradient-boosted model trained to predict purchase probability. Features: query-product text relevance score, product popularity, user purchase history similarity, price, stock status. Scores 200 candidates in ~20ms.

Why not use a cross encoder for ranking? A cross encoder jointly encodes the query and each product — 200 forward passes of a BERT-sized model at 20ms each = 4 seconds. Not feasible. The two-stage tradeoff sacrifices some ranking quality for latency.

**Retrieval recall matters.** If the best product for the user is not retrieved, the ranker never sees it. Retrieval recall at k (fraction of relevant items in the top k retrieved) must be measured. A retrieval model that retrieves the top 200 and has 95% recall at 200 means 5% of optimal results are discarded before ranking.

## Common traps

**Optimizing the ranker without measuring retrieval recall.** If retrieval recall is 80%, improving the ranker cannot recover the 20% of relevant items that were not retrieved. Fix retrieval first.

**Using the same training data for both stages.** The ranker trains on items that the retrieval model returned. Items the retrieval model never retrieves are never in the ranker's training data. This creates a feedback loop — the retrieval model's biases are amplified by the ranker.

**Ignoring position bias in training data.** Items shown in position 1 get more clicks than items in position 10, regardless of quality. Training a ranker on raw click data learns position bias, not item quality. Use inverse propensity scoring (IPS) to correct for position effects.

---

# 3. Feature Stores and Train-Serve Skew

## What the interviewer is actually testing

Whether you understand that model failures in production often come from features, not model architecture, and whether you can design a system that eliminates the divergence between training-time and serving-time feature computation.

## The reasoning structure

**Train-serve skew** is when the features computed during training differ from the features available at prediction time. This is the most common root cause of production model degradation that is not explained by data drift.

**Sources of skew:**
- Different code paths: training uses batch preprocessing script; serving uses streaming computation. They disagree on how to handle nulls, outliers, or encoding.
- Different data freshness: training uses a daily batch feature; serving uses a version computed an hour ago (or yesterday).
- Leakage during training: training includes a feature that uses future information. At serving time, that information is not yet available.
- Different normalization: training computes mean/std on training set; serving uses a stale statistics snapshot.

**How feature stores prevent skew:**
A feature store is a centralized service that defines features once and serves them consistently to both training and inference. The same code path computes the feature regardless of whether it is requested for a training job or for a live prediction.

**Feature store components:**
- Offline store: historical feature values for training (data warehouse, columnar storage)
- Online store: low-latency key-value lookup for inference (Redis, DynamoDB)
- Feature registry: versioned definitions of how each feature is computed
- Backfill pipeline: populates the offline store from the online computation history

## The pattern in action

**Fraud detection model.** Feature: "number of transactions in the last 24 hours from this IP."

Without a feature store:
- Training: computed via SQL window function over historical transaction table. Gets the exact 24-hour count.
- Serving: implemented in the fraud service. Queries a Redis counter that gets incremented on each transaction. Counter resets at midnight. Produces wrong values for IPs that span the midnight boundary.

Result: the model's training feature and serving feature disagree for all transactions after midnight. Model performance degrades predictably at night.

With a feature store:
- Feature defined once: COUNT transactions WHERE ip = X AND timestamp > NOW() - 24h
- Training: feature store backfills this value from the historical event log using the same logic
- Serving: feature store online layer maintains the rolling counter with the same semantics
- Guaranteed consistency

**Point-in-time correctness.** A credit risk model trained on features computed at time t should not include features that reflect information available at t+1 (e.g., a "default in next 90 days" label computed from data that was not yet observable at prediction time). Feature stores with point-in-time joins prevent this.

## Common traps

**"We tested offline, it worked."** Offline AUC measures whether the model can rank examples in a held-out test set. It does not measure whether the features computed during training match the features available in production.

**Building feature computation logic in two places.** Any time a feature has a training-time definition and a serving-time definition, they will eventually diverge. A feature store enforces one definition.

**Not versioning feature definitions.** A model trained on feature version 1.0 must be served with feature version 1.0. Serving a model trained on v1.0 with v2.0 features (even if the change seems innocuous) is an uncontrolled experiment.

---

# 4. Data Drift and Concept Drift

## What the interviewer is actually testing

Whether you can distinguish between distributional shift in inputs (which may or may not degrade the model) and shift in the underlying mapping from inputs to outputs (which always degrades the model), and whether you can design monitoring that detects both.

## The reasoning structure

**Three types of shift:**

**Data drift** — P(X) changes. The input distribution shifts. The model may still perform well if the mapping P(Y|X) is stable. A fashion model trained on winter inventory suddenly receives summer catalog queries. The inputs are different, but if the model learned a generalizable ranking function, it may still perform.

**Concept drift** — P(Y|X) changes. The underlying relationship changes. A fraud model trained on pre-pandemic behavior faces entirely different fraud patterns post-pandemic. Even if inputs look similar, the labels have changed. This always degrades the model.

**Label drift** — P(Y) changes. The marginal distribution of labels shifts. Seasonal click-through rate changes mean the model's threshold assumptions are wrong even if P(Y|X) is stable.

**Detection methods:**

For data drift (input distribution):
- PSI (Population Stability Index): compare feature bucket distributions between training and current data
- KS test: compare empirical CDFs of continuous features
- Chi-squared test: compare categorical feature distributions

For concept drift (output behavior):
- Prediction distribution shift: monitor the distribution of model scores over time
- Label-prediction agreement: when ground truth labels arrive (with delay), measure accuracy and flag changes
- Business KPI monitoring: downstream metrics (CTR, conversion) that reflect whether the model is making useful decisions

**PSI formula:**
```
PSI = Σ_i (A_i - E_i) * ln(A_i / E_i)
```

where A_i = actual (current) proportion in bucket i, E_i = expected (baseline) proportion.

| PSI | Interpretation |
|-----|----------------|
| < 0.1 | Stable |
| 0.1–0.25 | Some shift — investigate |
| > 0.25 | Significant shift — retrain or alert |

```python
from scipy.stats import ks_2samp
import numpy as np

def psi(baseline_pcts, current_pcts, eps=1e-4):
    return sum(
        (a - e) * np.log((a + eps) / (e + eps))
        for a, e in zip(current_pcts, baseline_pcts)
    )

stat, p_value = ks_2samp(baseline_feature, current_feature)
drifted = p_value < 0.05
```

## The pattern in action

A credit scoring model trained on 2019 data is deployed in 2020. COVID-19 fundamentally changes the relationship between income, employment status, and default risk. P(Y|X) has changed dramatically.

**Detecting concept drift when labels are delayed:**

Ground truth (whether a loan defaults) arrives 12–18 months after the decision. By the time you observe label drift, the model has been making bad decisions for over a year.

**Proxy approach:** monitor the prediction distribution. If model score distributions shift substantially while input features look similar, the model is responding differently to the same inputs — likely concept drift. Trigger retraining immediately rather than waiting for labels.

**Retraining triggers:**
- Scheduled: weekly or monthly regardless of detected drift (baseline hygiene)
- Triggered: PSI > 0.25 on key features, or prediction distribution shift > 2σ from baseline
- Continuous: online learning for high-velocity signals (ads, fraud) where the world changes daily

## Common traps

**"Retrain immediately."** Data drift does not always require retraining. If P(Y|X) is stable and the model generalizes well, retraining on shifted P(X) may hurt if the shift is temporary (seasonal). Diagnose the type of drift before prescribing retraining.

**Monitoring only business KPIs.** By the time a KPI drops 10%, the model has been failing for potentially weeks. Upstream metrics (feature distributions, prediction distributions) provide earlier warning.

**Not accounting for label delay in your monitoring.** If labels arrive 90 days later, your "current model accuracy" is always 90 days stale. Design monitoring to handle the delayed evaluation window explicitly.

---

# 5. Model Deployment Strategies

## What the interviewer is actually testing

Whether you can match deployment strategy to risk level and whether you understand the mechanics of each — not just the names.

## The reasoning structure

**The deployment problem.** Deploying a new model version is a change to a production system. Like any production change, it can break things. The question is: how do you limit blast radius while gaining confidence in the new version?

**Shadow deployment.** New model runs alongside the current model, receives all traffic, but its predictions are not used. Old model's predictions are returned to users.

Use when: you need to validate that the new model is operationally sound (latency, infrastructure, no crashes) and want to compare prediction distributions before going live.

Does not measure: business impact. Users do not see the new predictions.

**Canary deployment.** New model is rolled out to a small percentage of traffic (e.g., 5%). Old model handles the rest.

Use when: blast radius matters. If the new model is wrong for some user segment, only 5% of users are affected. Monitor KPIs on canary vs. control traffic before expanding.

Requirement: traffic must be split consistently per user (same user sees the same model). Otherwise users get inconsistent experiences.

**A/B testing.** Two model versions each handle a defined segment of users. Measure the difference in business metrics between segments.

Use when: you want to measure the actual business impact of a model change, not just technical correctness. Requires a hypothesis, a sample size calculation, and a minimum runtime to reach statistical significance.

**Multi-armed bandit.** Instead of fixed A/B split, dynamically allocate more traffic to better-performing variants.

Use when: rapid iteration matters and the cost of exposing users to the inferior variant is measurable. Common in ads and recommendations.

## The pattern in action

New recommendation model deployment at a streaming platform.

**Phase 1 — Shadow (3 days):** Run the new model in shadow alongside the current model. Validate: latency within budget (P99 < 80ms), no error spikes, prediction distribution looks reasonable (no model collapse — all users getting the same top items).

**Phase 2 — Canary (1 week):** Roll out to 5% of users. Monitor: per-user watch time, CTR, unsubscribe rate. Alert threshold: > 2% relative degradation in watch time triggers automatic rollback.

**Phase 3 — A/B test (2 weeks):** Expand to 50%/50% split. Calculate the minimum detectable effect size (MDE) and sample size required to reach statistical significance at α=0.05, power=0.80. Hold the split for the full required duration even if the canary looked good.

**Phase 4 — Full rollout:** If A/B results are statistically significant and practically meaningful, roll out to 100%.

Rollback plan: model registry stores previous version. Rollback is a flag flip in the serving layer, not a redeployment. P99 rollback time < 5 minutes.

## Common traps

**Treating shadow deployment as validation of business value.** Shadow deployment validates that the model runs, not that it performs better. You learn nothing about user response from shadow results.

**Stopping an A/B test early because it looks good.** Early stopping inflates the false positive rate dramatically (the "peeking problem"). Run the test for the pre-specified duration. If you stopped at day 3 of a planned 14-day test, your p-value is not 0.05 — it is much higher.

**Not defining rollback criteria before deployment.** "We'll rollback if things look bad" is not a plan. Define the specific metric thresholds that trigger an automatic or manual rollback before the deployment starts.

---

# 6. Monitoring in Production

## What the interviewer is actually testing

Whether you understand what to monitor beyond model accuracy, and whether you can reason about the difference between signals that tell you something is wrong versus signals that help you diagnose what is wrong.

## The reasoning structure

**A production ML system can fail in four ways:**
1. Infrastructure failure: the serving system goes down or becomes slow
2. Feature failure: features are computed incorrectly (nulls, stale values, wrong schema)
3. Model failure: the model's predictions are degraded due to drift or distribution shift
4. Business failure: the model's predictions are technically correct but no longer useful for the business objective

Each failure type requires different monitoring signals.

**Monitoring layers:**

| Signal | Failure type | Alert threshold |
|--------|-------------|-----------------|
| Latency P50/P90/P99 | Infrastructure | P99 > SLA threshold |
| Error rate (5xx, timeouts) | Infrastructure | > 0.1% |
| Null rates in critical features | Feature | > 5% nulls on required features |
| Feature distribution PSI | Model/Feature | PSI > 0.25 on key features |
| Prediction score distribution | Model | > 2σ shift from baseline |
| Ground truth accuracy (when labels arrive) | Model | > X% relative degradation |
| Business KPI (CTR, conversion, revenue) | Business | > 5% relative degradation |

**What to alert on vs. what to investigate:** P99 latency spike → alert immediately. Feature PSI > 0.25 → investigate, do not immediately alert unless also seeing KPI degradation. Business KPI drop without feature drift → investigate serving path and model behavior.

## The pattern in action

**Fraud detection model.** A week after deployment, the finance team notices fraud losses increasing. Where to start?

Diagnosis protocol:
1. Check prediction distribution: are scores shifting toward lower fraud probability? If yes, the model is becoming less aggressive.
2. Check feature distributions: PSI on key features (transaction amount, IP age, device fingerprint). If a key feature has PSI > 0.25, check whether the feature computation changed.
3. Check for schema drift: are new fields arriving that were not in training? Are expected fields becoming null?
4. Check ground truth: what do the labeled examples look like from the past week? If confirmed fraud cases are getting low risk scores, concept drift has occurred.

In this example, PSI on "device fingerprint age" jumps from 0.05 to 0.31. Investigation reveals that a device fingerprinting library was updated, changing how fingerprints are computed. Training used the old library. Serving now uses the new one. The feature definitions diverged (train-serve skew).

Fix: retrain the model with features computed using the new library. Short-term: roll back the device fingerprinting library if possible.

## Common traps

**Only monitoring business KPIs.** By the time CTR drops 10%, the model has been failing for weeks. Upstream feature and prediction monitoring catches problems earlier.

**Not versioning what served each prediction.** When investigating a production incident, you need to know which model version, which feature version, and which code served a specific request. Without that, debugging is guesswork.

**Setting alert thresholds too tight.** Natural day-to-day variation in prediction distributions will trigger constant false alarms. Set thresholds at 2–3σ from historical baseline to distinguish signal from noise.

---

# 7. Inference Optimization

## What the interviewer is actually testing

Whether you can reason about model performance as an engineering tradeoff, not just a modeling problem. A model that is too slow to deploy is useless. Inference optimization is the skill of making models production-viable.

## The reasoning structure

**The latency budget is fixed.** User-facing systems have SLAs — 100ms, 50ms, 20ms. The question is how to fit meaningful computation in that budget.

**Five optimization approaches:**

**Quantization.** Replace float32 weights with int8 or int4. Memory bandwidth is usually the bottleneck for inference, not FLOPS. Lower precision reduces memory footprint, increases throughput. 
- Post-training quantization (PTQ): quantize a pretrained model without retraining. Fast to apply.
- Quantization-aware training (QAT): simulate quantization during training. Better accuracy for aggressive quantization.
- Accuracy cost: typically < 1% for int8; higher for int4 without calibration.

**Pruning.** Remove low-magnitude weights or entire attention heads. Sparse models require sparse computation support (hardware matters — NVIDIA sparse tensor cores).
- Unstructured pruning: zeroes out individual weights. Hard to accelerate on current hardware.
- Structured pruning: removes entire neurons, heads, or layers. Directly reduces compute.

**Distillation.** Train a smaller student model to reproduce a larger teacher's outputs.
- Match soft probabilities (temperature scaling), intermediate representations, or both.
- DistilBERT achieved 97% of BERT's performance with 40% fewer parameters and 60% speedup.
- Requires training: higher upfront cost but no inference overhead.

**Caching / precomputation.** Compute results before the request arrives.
- Embedding precomputation: compute user/item embeddings in batch; serve from lookup table.
- KV-cache for LLMs: reuse past key-value computations across tokens in autoregressive generation.

**Batching.** Process multiple requests together to improve hardware utilization.
- Dynamic batching: accumulate requests until batch is full or timeout fires.
- Continuous batching (for LLMs): interleave different-length sequences to maximize GPU utilization. Avoids the fixed-batch-size bottleneck that wastes compute on padding.

## The pattern in action

**Reducing a BERT-based classification model from 200ms to 30ms:**

1. Quantization to int8 (PTQ): 200ms → 120ms. Memory bandwidth bottleneck reduced.
2. Structured pruning (remove 30% of attention heads with lowest mean gradient norm): 120ms → 90ms. Small accuracy drop (0.3 F1 points).
3. ONNX export + TensorRT compilation: 90ms → 50ms. Graph optimization and kernel fusion.
4. Distillation to 4-layer student: 50ms → 30ms. Pre-trained from the pruned model. 98% of the original accuracy on the task-specific benchmark.

At each step: measure accuracy. If it falls below acceptable threshold, do not proceed.

## Common traps

**Quantizing without calibrating.** Post-training quantization requires a calibration dataset to set the quantization scale per layer. Without calibration, accuracy can drop significantly.

**Distillation as the first step.** Distillation takes training time. Quantization and graph compilation are faster wins. Exhaust those before spending training compute on distillation.

**Not profiling before optimizing.** "The model is slow" is not a diagnosis. Profile to find which operation is the bottleneck. It may be preprocessing, feature lookup latency, or network I/O — not the model forward pass at all.

---

# 8. Model Registry and Reproducibility

## What the interviewer is actually testing

Whether you treat ML models with the same operational rigor as software artifacts — versioning, auditability, rollback capability.

## The reasoning structure

**A deployed ML system must be able to answer five questions:**
1. Which model version is serving predictions right now?
2. What training data was that model trained on?
3. What feature definitions did that model use?
4. What code produced that model?
5. What environment (library versions, hardware) was used?

Without this, debugging a production incident is archaeology.

**The model registry** is the central store of model artifacts with metadata.

Minimum viable model registry:
- Model artifact (weights, architecture)
- Training data reference (data hash or snapshot ID)
- Feature definition version
- Training code commit SHA
- Evaluation metrics (at training time, by dataset and group)
- Serving configuration (batch size, hardware, preprocessing steps)
- Deployment history (which versions were live when)

**Why reproducibility matters for debugging.** A model that was working well last Tuesday suddenly degrades on Thursday. The registry tells you: the model version did not change. The feature pipeline code was updated Wednesday (a different team's change). That is the root cause. Without versioning, you would retrain the model trying to fix a feature bug.

## The pattern in action

A recommendation model goes through three versions in a month. The registry records:

| Version | Training data | Feature SHA | Code commit | P@10 | Deployed |
|---------|--------------|-------------|-------------|------|----------|
| v1.2 | 2025-01-01 to 03-31 | f4a2b1 | abc123 | 0.31 | 2025-04-01 |
| v1.3 | 2025-01-01 to 04-30 | f4a2b1 | def456 | 0.33 | 2025-05-01 |
| v1.4 | 2025-01-01 to 05-31 | a9c3d2 | def456 | 0.32 | 2025-06-01 |

v1.4 performs worse than v1.3 despite more training data. Feature SHA changed (a9c3d2 vs f4a2b1). A feature definition changed between versions. The registry makes this comparison possible in 5 minutes. Without it, this would take days.

Rollback: the serving layer reads the "current production model" from the registry. Rolling back v1.4 to v1.3 is a registry record update, not a redeployment.

## Common traps

**Storing models in a file system without metadata.** You can version a file but not query it. A registry enables "find all models trained on data from this date range" and "which model was serving on March 15?"

**Not versioning feature definitions alongside the model.** A model trained on feature v1.0 served with feature v2.0 is an uncontrolled experiment. These must be pinned together.

**Treating reproducibility as academic overhead.** Every hour of a production incident that requires archaeology into past model behavior is a productivity loss that a model registry would have prevented.

---

# 9. Online vs. Batch Inference

## What the interviewer is actually testing

Whether you understand the architectural tradeoffs between serving predictions in real-time versus computing them in advance, and what each approach implies for feature design and system complexity.

## The reasoning structure

**Batch inference:** compute predictions for a large set of inputs in advance, store results, serve the precomputed result.
- Latency at query time: effectively zero (lookup)
- Feature space: can use rich features that take seconds to compute
- Freshness: predictions can be stale (minutes to hours, depending on batch cadence)
- Cost: amortized over batch

**Online (real-time) inference:** compute predictions on each request.
- Latency: must meet SLA (typically < 100ms)
- Feature space: only features computable within the latency budget
- Freshness: always fresh
- Cost: per-request model forward pass

**The decision matrix:**

| Scenario | Use batch | Use online |
|----------|-----------|-----------|
| Nightly email recommendations | Yes (freshness not critical) | No |
| Real-time ad ranking | No | Yes (recency matters) |
| Fraud detection at transaction time | No | Yes (cannot wait for batch) |
| Precomputed "users like you" widgets | Yes (cheap at serve time) | No |
| Personalized search results | Partial (precompute embeddings, rank online) | Mixed |

**The hybrid pattern** is common: precompute embeddings and static features (batch), combine with real-time features (current session, request context), and run a fast ranker online.

## The pattern in action

**Email recommendation system.**

Batch (nightly at 2am):
- For each user: compute top 20 recommended items using collaborative filtering + content model
- Store: user_id → [item_1, item_2, ..., item_20]
- Cost: 5M users × 20 items × model forward pass = handled in batch job

At email generation time (5am):
- Look up precomputed recommendations for each user
- Apply business rules (filter out out-of-stock items, exclude recently purchased)
- Serve

Why not online? No SLA requirement — email generation happens hours before sending. Batch enables richer features (full user history recomputed daily, complex collaborative filtering).

**Contrast: ad auction.** A user performs a search. In 30ms, the ad server must rank hundreds of candidate ads. Cannot precompute: the query is not known in advance. Must be online. Features must be computable in <10ms (precomputed advertiser embeddings + query embedding, no heavy model layers).

## Common traps

**Using online inference when batch would work.** Online inference is more expensive (per-request cost), more complex (serving infrastructure, latency constraints), and more brittle (inference service is a hard dependency). If freshness does not matter, batch is almost always simpler.

**Not designing for feature freshness requirements.** An online recommendation model that says "what did the user buy in the last hour?" needs that feature computed with sub-second latency. If it is not available online, the model cannot use it at serving time even if it performed well on batch data.

**Assuming batch and online features will match.** The train-serve skew problem again. Features computed in the batch training pipeline must be exactly reproducible by the online feature computation at serving time.

---

# 10. MLOps Pipeline Design

## What the interviewer is actually testing

Whether you can design a system that makes model development, deployment, and monitoring a repeatable, automated process — not a manual artisanal activity.

## The reasoning structure

**An MLOps pipeline is CI/CD applied to ML artifacts.** The analogy:

| Software CI/CD | ML Pipeline |
|---------------|-------------|
| Source code | Training code + config |
| Build | Training job |
| Unit tests | Data validation |
| Integration tests | Model evaluation (AUC, fairness, latency) |
| Artifact | Model weights + metadata |
| Deploy | Model serving |
| Monitoring | Drift + KPI monitoring |
| Rollback | Serve previous version |

The key difference: in software, code changes break systems. In ML, code AND data AND labels AND features AND environmental drift can break systems.

**MLOps maturity levels:**

Level 0: Manual training, manual deployment. No automation. Triggered by a data scientist running a notebook.

Level 1: Automated training pipeline (triggered by new data or schedule). Manual deployment gate.

Level 2: Fully automated. New data triggers training, validation gates run, deployment proceeds automatically if gates pass. Monitoring triggers retraining automatically.

## The pattern in action

**Automated retraining pipeline for a fraud detection model:**

```
New data arrives
    → Data validation (schema check, null rates, label distribution)
    → Feature pipeline runs (transform raw transactions → feature matrix)
    → Model training job (configurable hyperparameters, fixed architecture)
    → Evaluation gates:
        - AUC > 0.92 on holdout set
        - AUC on each demographic group > 0.88 (fairness gate)
        - Inference latency P99 < 50ms on standard hardware
    → If all gates pass: promote to staging
    → Smoke tests in staging (shadow traffic comparison vs. current model)
    → If shadow results acceptable: canary to 5% traffic
    → Monitor business KPIs for 24 hours
    → If no regression: promote to 100%
```

Every step is logged. The registry stores the model artifact, the evaluation metrics, the feature definitions, and the deployment decision.

If the fraud model degrades a week later, you can replay the entire pipeline with the same data to understand whether the issue was in the training, the features, or the deployment environment.

## Common traps

**Skipping validation gates under time pressure.** Gates exist to catch problems before they reach production. "We need to ship by Friday" is how a model with 40% error on a demographic group goes live.

**Manual deployment as the "safe" option.** Manual deployments are slower, less consistent, and harder to audit. The discipline of automated pipelines is the safe option.

**Not testing the rollback path.** Define rollback criteria and test the rollback mechanism in staging before the first production deployment. Finding out that rollback takes 45 minutes during a production incident is too late.

---

# Quick Diagnostics

**If your model performs well offline but degrades in production two weeks after deployment:**

Check feature freshness and train-serve skew first (before retraining). Verify that feature computation in production uses the same logic as training. Check for null rate spikes or schema changes in upstream data. Only after ruling out feature failure should you investigate model-level drift. "Retrain immediately" before diagnosing the root cause is like restarting pods before checking why the service broke.

**If asked how to design an ML system in an interview:**

Follow the design order: goal → success metric → constraints → data and labels → training pipeline → serving path → monitoring and drift → rollout and rollback. Do not jump to model selection before establishing the serving constraints — latency budget determines what features and model architectures are feasible.

**If asked how a two-stage retrieval-ranking system handles the relevance-latency tradeoff:**

Retrieval trades precision for speed — it retrieves candidates fast using approximate methods (ANN, BM25) and optimizes recall at the cost of some irrelevant candidates. Ranking trades speed for precision — it scores only retrieved candidates with a heavy model optimized for precision. The system as a whole gets near-ranker precision at near-retrieval latency, because the heavy computation runs over hundreds of candidates, not millions.
