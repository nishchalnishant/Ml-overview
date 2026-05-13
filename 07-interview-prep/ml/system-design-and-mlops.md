**Primary reference:** [Production ML](../../06-production-ml/README.md) | [System Design](../../06-production-ml/system-design/machine-learning-engineering.md)

# ML System Design and MLOps

This is the file where your **Azure + DevOps background becomes a superpower**.

Because ML systems are not just models.
They are delivery systems.

And if you already understand:

- CI/CD
- infra provisioning
- observability
- rollout safety
- production reliability

you already understand half of MLOps.

---

# 1. The ML Lifecycle, Rewritten in DevOps Language

Here is the mapping:

- **raw data** = source input
- **feature pipeline** = build preprocessing stage
- **training job** = model build
- **validation** = test gate
- **model artifact** = release artifact
- **online inference** = deployed microservice
- **monitoring** = SRE + business telemetry
- **retraining** = scheduled release with new artifact + new data

The twist is this:

In software, code changes break systems.
In ML, **code, data, labels, features, or environment drift** can break systems.

So MLOps is CI/CD with more ways to be humbled.

---

# 2. Retrieval + Ranking: Why Big Systems Use Two Stages

In recommendation, search, and feed systems, you often cannot score millions of items with a heavy model in real time.

So you split the system into two parts:

## Retrieval

Fast stage.
Finds a small candidate set.

Examples:

- vector search
- approximate nearest neighbors
- collaborative filtering
- inverted index

## Ranking

Smarter, heavier stage.
Scores only the retrieved candidates.

Examples:

- boosted trees
- DNNs
- cross encoders
- DCN-style ranking models

**DevOps analogy**

Retrieval is like filtering which services even enter the deployment checklist.
Ranking is the deeper review before final promotion.

Do not run the expensive gate on absolutely everything.

---

# 3. Training vs Inference

This one gets asked a lot.

## Training

When the model learns from data.

Usually:

- slower
- heavier
- offline
- more compute intensive

## Inference

When the trained model makes predictions.

Usually:

- faster
- runtime-sensitive
- production-facing
- latency-sensitive

**Azure framing**

Training is your build pipeline.
Inference is your live API behind the load balancer.

One creates the artifact.
The other serves it.

---

# 4. Feature Stores and Train-Serve Skew

One of the most real production problems in ML:

The features used during training do not exactly match the features available in production.

That is called:

- train-serving skew
- online/offline skew

How it happens:

- different code paths
- different data freshness
- different definitions
- leakage during training

## Why feature stores help

A feature store gives you a central place to define, version, and serve features consistently for:

- training
- batch inference
- online inference

**DevOps parallel**

Think of a feature store like a shared, versioned infrastructure module.
Without it, every team writes its own Terraform slightly differently and then acts surprised when production disagrees.

---

# 5. Data Drift vs Concept Drift

This is classic interview gold.

## Data Drift

Input distribution $P(X)$ changes.

Example: a fashion recommendation model trained on winter inventory suddenly sees summer catalog traffic.

## Concept Drift

The relationship $P(Y \mid X)$ changes.

Example: a fraud model was good last month, but fraudsters changed behavior and the old patterns no longer predict fraud well.

## Label Drift

The label marginal $P(Y)$ changes.

Example: seasonal change in click-through rates.

**Easy memory trick:**
- data drift = the world looks different
- concept drift = the rules of the world changed

---

# 5a. Drift Detection — Formulas

## Population Stability Index (PSI)

$$\text{PSI} = \sum_{i=1}^{B} (A_i - E_i) \ln\frac{A_i}{E_i}$$

where $A_i$ = actual (current) proportion in bucket $i$, $E_i$ = expected (baseline) proportion.

| PSI range | Interpretation | Action |
| :--- | :--- | :--- |
| < 0.1 | Stable | Monitor normally |
| 0.1–0.25 | Some shift | Investigate |
| > 0.25 | Significant drift | Retrain / alert |

## Kolmogorov-Smirnov (KS) Test

$$\text{KS} = \sup_x |F_1(x) - F_2(x)|$$

where $F_1, F_2$ are empirical CDFs of baseline and current distributions. p-value threshold: 0.05.

```python
from scipy.stats import ks_2samp

stat, p_value = ks_2samp(baseline_feature, current_feature)
drifted = p_value < 0.05

def psi(baseline_pcts, current_pcts, eps=1e-4):
    """PSI > 0.1 = some drift; > 0.25 = significant drift."""
    return sum(
        (a - e) * np.log((a + eps) / (e + eps))
        for a, e in zip(current_pcts, baseline_pcts)
    )
```

---

# 6. Monitoring Models in Production

You should monitor more than just latency.

| Signal | What to monitor | Alert threshold |
| :--- | :--- | :--- |
| **Input schema** | Null rates, data types | > 5% null on critical features |
| **Feature drift** | PSI per feature | PSI > 0.25 on key features |
| **Prediction drift** | Score distribution shift | > 2σ from baseline |
| **Label quality** | Ground truth arrival delay | If delayed, recalibrate |
| **Latency** | P50, P90, P99 | P99 > SLA |
| **Error rate** | 5xx, timeouts | > 0.1% |
| **Business KPI** | CTR, revenue, conversion | > 5% drop triggers incident |

**Retraining triggers:**
- Scheduled: weekly/monthly regardless of drift
- Triggered: PSI > 0.25 or business KPI drops
- Continuous: online learning for high-velocity signals (ads, fraud)

---

# 7. Deployment Strategies: Shadow, Canary, A/B

## Shadow Deployment

New model runs in parallel but does not affect user output.

Use it to validate:

- latency
- infrastructure behavior
- prediction patterns

## Canary Deployment

Roll out to a small subset first.

Use it when:

- blast radius matters
- rollback needs to be fast

## A/B Testing

Different user groups get different model variants.

Use it when:

- you want to measure actual business impact

**This should feel very familiar if you know release engineering.**

It is the same playbook.
The artifact is just a model instead of a service binary.

---

# 8. Inference Optimization

If the model is too slow, you have options.

## Quantization

Lower precision, smaller model, faster inference.

## Pruning

Remove less important weights or units.

## Distillation

Train a smaller student model to mimic a larger teacher model.

## Caching / Precomputation

Do work before the request arrives when possible.

## Retrieval Staging

Do not use the heaviest model on the full universe.

**Short rule**

If latency is hurting, do not only ask:

> "Can we buy more hardware?"

Also ask:

> "Can we avoid unnecessary work?"

That is elite systems thinking.

---

# 9. Model Registry and Versioning

A production ML system should always know:

- which model version served the prediction
- which feature version fed it
- which data slice trained it
- which code version built it

Without that, debugging becomes folklore.

**DevOps analogy**

This is your model equivalent of:

- image tag
- commit SHA
- release number
- environment metadata

If you would never deploy an unversioned container, do not deploy an unversioned model.

---

# 10. Reproducibility

Every production prediction should be reproducible enough to answer:

- what model made this prediction?
- with what features?
- under what config?

That usually means versioning:

- code
- model weights
- datasets
- feature definitions
- environment

Reproducibility is not academic neatness.
It is operational survival.

---

# 11. MLOps in One Sentence

MLOps is the discipline of making model development, deployment, monitoring, and retraining **repeatable, trustworthy, and production-safe**.

If DevOps made software delivery mature, MLOps is trying to do the same for systems whose behavior changes with data.

---

# 12. Kubernetes and Model Serving

If someone asks how you would deploy a model, do not jump straight to architecture buzzwords.

Talk through:

- packaging the model service
- containerizing it
- exposing inference endpoint
- autoscaling
- monitoring
- rollout strategy
- rollback plan

Yes, Kubernetes matters.
But it is not the story by itself.

The story is:

> "How do I serve predictions reliably at scale?"

---

# 13. Quick Design Frame for Interviews

When asked to design an ML system, use this order:

1. define goal
2. define success metric
3. define constraints
4. define data and labels
5. define training pipeline
6. define serving path
7. define monitoring and drift handling
8. define rollout and rollback

That structure is gold.

It keeps you calm and sounds senior.

---

# How Would You Deploy This Using Azure Pipelines?

Here is a clean answer shape:

1. Train model in a pipeline stage
2. Run validation and performance gates
3. Register model artifact with version metadata
4. Build inference container
5. Deploy to dev/staging
6. Run smoke tests and shadow traffic checks
7. Promote with canary rollout
8. Monitor drift, latency, and business KPI
9. Roll back automatically if guardrails fail

That answer will land beautifully for any MLOps discussion.

---

# Quick Thought Experiment

Your model is accurate offline.
In production, it starts misbehaving two weeks later.

What do you inspect first?

- drift
- feature freshness
- train-serving skew
- threshold assumptions
- delayed-label evaluation

If your instinct was "retrain immediately," slow down.

That is like restarting pods before checking why the service broke.

Possible.
Sometimes useful.
Not the first smart move.
