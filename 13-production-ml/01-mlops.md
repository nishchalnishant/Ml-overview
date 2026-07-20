---
module: Production ML
topic: Mlops
subtopic: ""
status: unread
tags: [productionml, ml, mlops]
---
# MLOps — Machine Learning Operations

---

## Table of Contents

1. [MLOps Overview](#mlops-overview)
2. [ML Lifecycle](#ml-lifecycle)
3. [Data Management](#data-management) — DVC, data versioning, schema validation
4. [Model Development](#model-development) — MLflow, W&B, experiment tracking, hyperparameter sweeps, model registry
5. [Deployment Strategies](#deployment-strategies) — REST serving, gRPC serving, canary, blue-green, shadow, A/B testing
6. [Monitoring and Observability](#monitoring-and-observability) — drift detection, PSI, KS test, concept drift
7. [CI/CD for ML](#cicd-for-ml) — GitHub Actions, Airflow ML pipelines
8. [Infrastructure and Tools](#infrastructure-and-tools) — SageMaker, Vertex AI, Kubernetes, Triton, TorchServe
9. [Best Practices](#best-practices)

---

## MLOps Overview

### The problem

A data scientist ships a model as a notebook. Three months later: it's running on someone's laptop, nobody knows which version is in production, there are two copies with different preprocessing logic, last month's results can't be reproduced, and performance has silently degraded because nobody noticed the data shifted.

This is an engineering problem, not a model problem.

### The core insight

MLOps is DevOps applied to ML, with three extra dimensions: *training data* is a versioned artifact, *model behavior* can degrade with no code change, and *release criteria* include quality metrics over time, not just pass/fail tests.

### MLOps vs DevOps

| Aspect | DevOps | MLOps |
|--------|--------|-------|
| Artifacts | Code, binaries | Code + data + models |
| Testing | Unit, integration tests | Data validation, model validation |
| Deployment | Continuous deployment | Gradual rollout, A/B testing |
| Monitoring | System metrics | Model performance, data drift |
| Versioning | Code versions | Code + data + model versions |

---

## ML Lifecycle

### The problem

Without a defined lifecycle: experiments overwrite each other, "the model from last Tuesday" can't be reproduced, training and serving disagree on preprocessing, and nobody knows when to retrain.

### The core insight

The ML lifecycle is a loop, not a pipeline. Data quality issues found during evaluation send you back to data collection. Drift in production triggers retraining. It only works if every stage is automated and instrumented.

### The mechanics

**Stage 1: Problem definition**

Define the business decision, not just the model target: not "predict churn" but "predict who will churn in 30 days so we can target retention offers." Set both a business KPI (churn reduction) and an ML proxy metric (precision/recall at the decision threshold).

Establish a baseline first (majority class, heuristic, human performance). If the model can't beat it, revisit the framing.

**Stage 2: Data collection and preparation**

```
Raw Data -> Validation -> Cleaning -> Feature Engineering -> Training Data
```

- Schema validation before any transformation (fail fast)
- Data versioning — snapshot the training set, not just the model
- Point-in-time correctness — labels only use features available before the event

**Stage 3: Model development**

```
EDA -> Feature Selection -> Model Training -> Hyperparameter Tuning -> Evaluation
```

Track every experiment: hyperparameters, metrics, data version, code commit, environment. Without this you can't explain why the current model beats the previous one.

**Stage 4: Model deployment**

```
Model Registry -> Staging -> A/B Testing -> Production
```

Never deploy directly from an experiment. Register the artifact, validate in staging, then route a controlled slice of traffic to it.

**Stage 5: Monitoring and maintenance**

```
Performance Monitoring -> Drift Detection -> Retraining -> Feedback Loop
```

Models degrade with no code change because the world changes. Monitoring closes the loop.

---

## Data Management

### The problem

A bad config crashes immediately. Bad or shifted data causes a silent accuracy drop that may take weeks to surface — by then the root cause is buried under months of traffic.

### The core insight

Data needs the same versioning discipline as code. Every training run must be reproducible from a data snapshot plus a code commit. Quality failures must be caught before training, not after deployment.

### The mechanics

**Data versioning with DVC**

```bash
dvc init
dvc add data/train.csv          # DVC stores the file hash, git tracks the .dvc pointer
git add data/train.csv.dvc .gitignore
git commit -m "Add training data v1.0"
dvc push                        # push to remote storage (S3, GCS, Azure Blob)

git checkout <commit-hash>      # reproduce a historical dataset
dvc pull
```

*What breaks:* DVC tracks files by hash — if the same-named file is regenerated with different content, the `.dvc` pointer doesn't auto-update. You must re-add and commit it.

---

**Data quality checks with Great Expectations**

```python
from great_expectations import DataContext

context = DataContext()
batch = context.get_batch("training_data")

batch.expect_column_values_to_not_be_null("user_id")
batch.expect_column_values_to_be_between("age", min_value=0, max_value=120)
batch.expect_column_values_to_be_in_set("country", ["US", "UK", "CA", "AU"])
batch.expect_column_mean_to_be_between("purchase_amount", min_value=10, max_value=500)

results = context.run_validation_operator("action_list", batch)
assert results["success"], f"Data validation failed: {results}"
```

*What breaks:* expectations set on training data may not track legitimate distribution changes (new product lines, new geographies) — they need periodic updates.

---

**Feature stores**

Problem they solve: the same feature computed slightly differently in training (Python, daily batch) vs serving (Java, real-time) — the model trains on one distribution and is served another.

- **Offline store** (Parquet/Delta Lake): historical features for training, with point-in-time joins
- **Online store** (Redis/DynamoDB): pre-materialized features for inference, < 10ms lookup

Benefits: canonical feature definitions computed once and reused, point-in-time correctness prevents leakage, and consistent training/serving features eliminate train-serve skew.

---

## Model Development

### The problem

Without experiment tracking, "the model we deployed last month" is whatever's on someone's laptop. You can't compare experiments fairly, reproduce a result, or explain why Model B beats Model A.

### The core insight

Every experiment is a versioned artifact: code + data + hyperparameters + metrics + environment. If you can't reproduce it from metadata, it didn't happen for production purposes.

### The mechanics

**MLflow experiment tracking**

```python
import mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("fraud_detection_v2")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced")
    model.fit(X_train, y_train)

    mlflow.log_metric("val_auc", roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    mlflow.sklearn.log_model(model, "model")
```

**Weights & Biases (W&B)**

The standard choice for deep learning teams — richer visualizations, built-in hyperparameter sweeps, native PyTorch/HuggingFace integration.

```python
import wandb

run = wandb.init(project="fraud-detection", config={"learning_rate": 3e-4, "batch_size": 256})
for epoch in range(config.n_epochs):
    train_loss, val_auc = train_epoch(model, optimizer)
    wandb.log({"epoch": epoch, "train/loss": train_loss, "val/auc": val_auc})

artifact = wandb.Artifact("fraud-transformer-v3", type="model", metadata={"val_auc": 0.94})
artifact.add_file("model.pt")
run.log_artifact(artifact)
```

W&B Sweeps run automated hyperparameter search (grid, random, or Bayesian):

```python
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/auc", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "batch_size": {"values": [64, 128, 256, 512]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="fraud-detection")
wandb.agent(sweep_id, function=train_sweep, count=50)
```

**MLflow vs W&B**

| Criterion | MLflow | W&B |
|---|---|---|
| Language/framework | Python, any framework | Python, deep-learning focus |
| Self-hosted | Yes, free | Yes, paid (W&B Server) |
| Hyperparameter sweeps | Manual (e.g. + Optuna) | Built-in Bayesian sweeps |
| Visualization | Basic charts | Rich: gradients, media, 3D |
| Best for | Any ML, open-source preference | Deep learning, collaborative teams |

---

**Model registry lifecycle**

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()

mlflow.register_model(f"runs:/{run_id}/model", "fraud_detector")
client.transition_model_version_stage(name="fraud_detector", version=1, stage="Staging")
# after validation:
client.transition_model_version_stage(name="fraud_detector", version=1, stage="Production")
```

Stages: **Development** (experimental) → **Staging** (validated offline) → **Production** (serving live traffic) → **Archived** (deprecated, kept for audit/rollback).

*What breaks:* the registry is only meaningful if promotion is gated by validation criteria. Without automated gates, "Production" becomes a label humans set by hand without evidence.

---

## Deployment Strategies

### The problem

A new model beats production on your eval set. How do you replace the live model without breaking the service, and verify it actually performs better on real users?

### The core insight

Every deployment strategy trades safety against speed. The riskier the change (new architecture, new feature schema, large distribution shift), the more conservative the rollout should be.

### The mechanics

**Batch inference** — process large datasets offline (daily recommendations, nightly reports).
```
Batch Data -> Prediction Job (Spark/Airflow) -> Results Storage -> Serving Layer
```
Pros: high throughput, can use complex/slow models, cheap. Cons: predictions go stale between runs.

**Real-time inference — REST** — used when latency < 100ms, user-facing, simple JSON interface.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np

app = FastAPI()
pipeline = joblib.load("model_with_preprocessing.pkl")

class PredictRequest(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(req: PredictRequest):
    X = np.array(req.features).reshape(1, -1)
    prob = pipeline.predict_proba(X)[0, 1]
    return {"prediction": int(prob > 0.5), "probability": float(prob)}
```

**Real-time inference — gRPC** — used for internal microservices, strict SLAs (< 20ms), binary payloads (embeddings, images). Roughly 3-7x faster than REST at scale due to Protobuf binary encoding and HTTP/2 multiplexing. Contract is defined in a `.proto` file; Python stubs are generated from it, and the server implements the RPC methods.

**REST vs gRPC**

| Criterion | REST + JSON | gRPC + Protobuf |
|---|---|---|
| Payload size | Larger | Compact, binary |
| Latency | Higher (serialization) | ~3-7x faster at scale |
| Streaming | No (without WS) | Bidirectional streaming |
| Browser support | Native | Needs grpc-web proxy |
| Best for | External APIs | Internal services, high TPS |

**Deployment patterns**

| Pattern | Mechanism | Risk | Recovery speed |
|---------|-----------|------|----------------|
| Canary | Gradual traffic shift (5% -> 100%) | Low | Fast (reduce traffic) |
| Blue-green | Atomic environment swap | Medium | Instant (flip LB) |
| Shadow | Parallel, predictions discarded | Zero | N/A |
| A/B test | Split by user cohort | Low | Fast (stop experiment) |

---

## Monitoring and Observability

### The problem

A model that was 94% accurate at deployment is 79% accurate six months later. No code changed. The degradation was gradual and nobody was watching the right signals.

### The core insight

ML monitoring has two layers classic DevOps doesn't: *data drift* (inputs changed) and *model quality* (the input-output relationship changed). System metrics (latency, error rate) tell you the service is running, not that the model is still useful.

### The mechanics

**Model performance monitoring** — alert when accuracy or latency crosses a threshold (e.g. via Prometheus rules on a model-accuracy gauge or p99 prediction latency).

**Data drift detection**

```python
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, CatTargetDriftTab

dashboard = Dashboard(tabs=[DataDriftTab(), CatTargetDriftTab()])
dashboard.calculate(reference_data=training_df, current_data=production_window_df)
dashboard.save("drift_report.html")
```

Detection methods:
- **PSI (Population Stability Index):** < 0.1 stable, 0.1-0.2 monitor, > 0.2 retrain
- **KL Divergence:** statistical distance between distributions, sensitive to tails
- **Kolmogorov-Smirnov test:** non-parametric test for continuous feature shift

**Concept drift** — the relationship between X and y changes, not just the input distribution. Detected by monitoring performance metrics over time and comparing predictions to actuals as they arrive.

Solutions: scheduled or drift-triggered retraining, online learning with a sliding window, or ensembling the current model with a freshly-trained one.

*What breaks:* concept drift needs ground-truth labels, which often arrive late (a loan default may not be known for 6-12 months). Use proxy signals or plan for delayed evaluation.

---

## CI/CD for ML

### The problem

A 4-hour training pipeline, manually evaluated, manually pushed to production. Not repeatable, not auditable, doesn't scale.

### The core insight

ML CI/CD is a DevOps pipeline with three extra gates: data validation, model training, and model evaluation. "Never deploy without passing all gates" still applies.

### The mechanics

**Pipeline stages**

1. Code commit triggers CI
2. Unit tests (data processing, feature engineering)
3. Data validation (schema, range, distribution checks)
4. Model training (on a versioned data snapshot)
5. Model evaluation (vs threshold AND vs current production model)
6. Register model if evaluation passes
7. Optional human approval gate for sensitive models
8. Deploy to staging
9. Shadow or canary on production traffic
10. Promote to full production

**GitHub Actions example**

```yaml
name: ML Pipeline
on:
  push: { branches: [main] }
  schedule: [{ cron: "0 2 * * *" }]   # nightly retraining

jobs:
  train-and-evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: pytest tests/unit/ -v
      - run: python scripts/validate_data.py
      - run: python scripts/train.py
      - run: python scripts/evaluate.py --min-auc 0.85 --min-precision 0.80
      - if: success()
        run: python scripts/register_model.py --stage Staging
```

*What breaks:* nightly retraining without a data-freshness check can train on stale or corrupted data — always gate training on validation. The final step must check actual metric thresholds, not just that training completed.

---

**Apache Airflow for ML pipelines**

A battle-tested DAG scheduler — useful when you need complex dependency management, retries with backoff, SLA alerting, and integration with many data systems. A typical retraining DAG: validate data freshness → train → branch on whether the new model beats a threshold and doesn't regress vs production → register or notify failure.

Best practices: pass large artifacts via S3 paths (not XCom), set `max_active_runs=1` for retraining DAGs, use SLA-miss callbacks, use `ExternalTaskSensor` to wait on upstream data pipelines.

---

**Safety checks before promotion to production**

- Minimum accuracy threshold vs baseline
- No performance regression vs current production model
- Shadow mode testing (at least 24 hours)
- Gradual rollout with monitoring at each step

---

## Infrastructure and Tools

### The problem

Every ML team rediscovers the same infra questions: where to store models, how to scale training, how to serve under load, how to track what's running in production.

### The core insight

The ML infra stack mirrors the software infra stack, with specialized components for data and model management.

### The mechanics

**Cloud ML platforms**

| Platform | ML Services | Best for |
|----------|-------------|----------|
| AWS | SageMaker, EC2, Lambda | Mature ecosystem, broad integrations |
| GCP | Vertex AI, AI Platform | TensorFlow integration, AutoML |
| Azure | Azure ML, Databricks | Enterprise Azure integration, native MLflow |

**AWS SageMaker** provides managed training, hosting, and pipeline orchestration (`sagemaker.workflow.pipeline.Pipeline` chains processing, training, conditional registration, and deployment steps, each in its own managed container). SageMaker Endpoints deploy the registered model with configurable autoscaling and data capture for monitoring.

**Google Vertex AI** uses the Kubeflow Pipelines SDK, fully managed on GCP, with native BigQuery/GCS integration. Vertex AI Model Monitoring detects skew and drift on deployed endpoints automatically.

**SageMaker vs Vertex AI**

| Criterion | SageMaker (AWS) | Vertex AI (GCP) |
|---|---|---|
| Pipeline SDK | SageMaker Pipelines | Kubeflow Pipelines SDK |
| Training data source | S3 | BigQuery + GCS |
| Hyperparameter tuning | SageMaker HPO | Vertex Vizier (Bayesian) |
| Built-in monitoring | Model Monitor | Vertex Model Monitoring |
| Best for | AWS-native teams | GCP/BigQuery shops |

**Containerization** — package the model artifact and preprocessing pipeline in the *same* container image. If they're separate, version drift between model and preprocessing becomes possible.

**Kubernetes for model serving** — treat model servers as stateless microservices: autoscale on CPU/GPU utilization, use readiness (model loaded?) and liveness (server responding?) health checks, and set explicit GPU memory limits so pods aren't evicted at runtime.

**Model serving platforms**

| Tool | Best for | Key features |
|------|----------|--------------|
| TensorFlow Serving | TensorFlow models | High performance, versioning |
| TorchServe | PyTorch models | Multi-model serving, model archiver |
| ONNX Runtime | Cross-framework | Optimized inference, hardware acceleration |
| Seldon Core | Kubernetes-native | Canary, A/B, shadow via CRD |
| Triton Inference Server | Multi-model, GPU | Dynamic batching, concurrent models |

---

## Best Practices

### Reproducibility

"What model is in production and why?" must always be answerable:
- Version code (git), data (DVC/snapshot hash), models (registry), environment (Docker)
- Set random seeds in all training scripts
- Log every experiment with full metadata
- Use Docker for consistent training environments, not per-machine conda envs

### Testing

- **Unit tests:** individual data processing/feature transform functions
- **Integration tests:** end-to-end pipeline from raw data to prediction
- **Model validation tests:** accuracy thresholds, fairness checks, regression vs baseline
- **Data quality checks:** schema, ranges, distributions, missing value rates

### Monitoring

Set up before deployment, not after the first incident:
- Track performance metrics with lag-aware strategies for delayed labels
- Monitor data drift on all features (PSI/KS)
- Alert on latency regression and error rate spikes
- Log predictions with timestamps for offline analysis

### Security

- Encrypt data at rest and in transit
- Use secrets management (AWS Secrets Manager, HashiCorp Vault) — never hardcode credentials
- Access control (IAM, RBAC) on model endpoints and data stores
- Audit logging: who deployed what, when, with what approval

### Cost optimization

- Spot instances for training (with checkpointing)
- Autoscaling for inference (scale to zero when idle)
- Model compression (quantization, pruning, distillation)
- Batch similar requests for GPU efficiency

---

## MLOps Maturity Levels

Teams don't jump from "notebook in production" to full automation overnight. Where you are tells you which problem to solve next.

- **Level 0 — Manual process:** everything by hand, no monitoring. "The model" is a file on someone's laptop.
- **Level 1 — ML pipeline automation:** automated training, experiment tracking, model registry, basic monitoring. Experiments are reproducible; deployments are still manual.
- **Level 2 — CI/CD pipeline automation:** automated testing/deployment with approval gates, continuous monitoring, automated retraining triggers. The team only handles threshold calibration and incident response.
- **Level 3 — Production-grade MLOps:** drift/fairness monitoring, online learning, multi-model management, feature stores, governance docs. A degradation event triggers automated diagnosis, retraining, validation, and deployment with no human in the loop for the happy path.

---

## Interview Topics

**"Explain the difference between model drift and data drift"**

Data drift: input distribution P(X) changes — different users, behavior, or seasonality. Concept drift: the relationship P(Y|X) changes — the same features now predict a different outcome. Both need retraining but have different detection signals: data drift is detectable without labels (compare input distributions); concept drift needs labels (compare accuracy on recent vs historical data).

**"How would you deploy a model to production?"**

Shadow mode first (zero risk, builds confidence) → canary (5% traffic, monitor business metrics for significance) → full rollout. Package model and preprocessing together. Define rollback triggers on accuracy drop, latency regression, error spikes. Monitor data drift from day one.

**"How do you ensure reproducibility?"**

Version code (commit hash), data (DVC pointer/snapshot hash), model (registry version), environment (Docker image with pinned deps). Set random seeds. Log all hyperparameters. Store the training snapshot separately from the live dataset.

**"How would you detect if a model is degrading?"**

Three signal layers: input drift (PSI/KS on features, detectable in hours), prediction drift (output distribution shift, detectable in hours), and performance degradation (accuracy vs ground truth, delayed by label lag). Monitor all three — don't wait for customer complaints.

**"Batch vs real-time inference trade-offs?"**

Batch: high throughput, larger models, cheaper, but predictions go stale between runs. Real-time: low latency (< 100ms target), always fresh, but constrained model complexity and higher serving cost. The deciding factor is how fast ground truth changes and whether a stale prediction is still useful.

## Flashcards

**Schema validation before any transformation?** #flashcard
Fail fast on bad data — validate before any transformation runs.

**Data versioning?** #flashcard
Snapshot the training set, not just the model.

**Point-in-time correctness?** #flashcard
Labels must only use features available before the event.

**Offline store (Parquet/Delta Lake)?** #flashcard
Historical features for training, with point-in-time joins.

**Online store (Redis/DynamoDB)?** #flashcard
Pre-materialized features for inference, < 10ms lookup.

**Why do feature stores prevent training-serving skew?** #flashcard
Feature definitions are canonical — computed once, used identically in both training and serving.

**Model registry stages?** #flashcard
Development (unvalidated) -> Staging (validated offline) -> Production (serving live traffic) -> Archived (deprecated, kept for audit/rollback).

**PSI (Population Stability Index) thresholds?** #flashcard
< 0.1 stable, 0.1-0.2 monitor, > 0.2 retrain.

**KL Divergence?** #flashcard
Statistical distance between distributions; sensitive to tail behavior.

**Kolmogorov-Smirnov Test?** #flashcard
Non-parametric test for continuous feature distribution shift.

**Ways to respond to concept drift?** #flashcard
Retrain on recent data (scheduled or drift-triggered), online learning with a sliding window, or ensemble current model with a freshly-trained one.

**Safety checks before promoting a model to production?** #flashcard
Minimum accuracy threshold vs baseline, no regression vs current production model, 24h+ shadow mode, gradual rollout with monitoring at each step.

**ML system health checks on Kubernetes?** #flashcard
Readiness (is the model loaded?) and liveness (is the server responding?).

**Why must GPU memory limits be set explicitly?** #flashcard
Otherwise pods get evicted at runtime when memory is exceeded.

**Four pillars of reproducibility?** #flashcard
Version code (git), data (DVC/snapshot hash), models (registry), environment (Docker). Plus fixed random seeds and full experiment metadata logging.

**Four categories of ML tests?** #flashcard
Unit tests (data/feature functions), integration tests (end-to-end pipeline), model validation tests (accuracy/fairness/regression), data quality checks (schema/range/distribution).

**Core production monitoring checklist?** #flashcard
Lag-aware performance tracking, data drift on all features (PSI/KS), latency/error-rate alerts, timestamped prediction logging.

**ML security essentials?** #flashcard
Encrypt at rest/in transit, use a secrets manager (never hardcode credentials), IAM/RBAC on endpoints and data stores, audit logging of deployments.

**Cost optimization levers for ML infra?** #flashcard
Spot instances for training (with checkpointing), autoscaling to zero for inference, model compression (quantization/pruning/distillation), request batching for GPU efficiency.
