---
module: Production Ml
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
3. [Data Management](#data-management)
4. [Model Development](#model-development)
5. [Deployment Strategies](#deployment-strategies)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [CI/CD for ML](#cicd-for-ml)
8. [Infrastructure and Tools](#infrastructure-and-tools)
9. [Best Practices](#best-practices)

---

## MLOps Overview

### The problem

You have a trained model. A data scientist ships it as a notebook. Three months later: the model is running on someone's laptop, nobody knows which version is in production, there are two copies with different preprocessing logic, the team cannot reproduce last month's results, and performance has silently degraded because nobody noticed the data distribution shifted.

This is not a model problem. It is an engineering problem.

### The core insight

MLOps is DevOps applied to ML systems, with three additional dimensions that classic software does not have: the *training data* is a versioned artifact, the *model behavior* can degrade without any code change, and the *release criteria* include quality metrics over time, not just test pass/fail.

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

Without a defined lifecycle, ML projects run in circles: experiments overwrite each other, "the model from last Tuesday" cannot be reproduced, training and serving disagree on preprocessing, and the team does not know when to retrain.

### The core insight

The ML lifecycle is a loop, not a pipeline. Every stage feeds back into the previous ones. Data quality problems discovered during evaluation send you back to data collection. Drift detected in production triggers retraining. The lifecycle only works if every stage is automated and instrumented.

### The mechanics

**Stage 1: Problem definition**

Define what the model must actually achieve in terms of a business decision. Not "predict churn" but "predict which users will churn within 30 days so we can target them with a retention offer." Set success metrics at two levels: business KPI (churn reduction) and ML proxy (precision/recall at the decision threshold).

Establish a baseline before building anything: majority class, rule-based heuristic, or human performance. If your model cannot beat the baseline, revisit the problem framing.

**Stage 2: Data collection and preparation**

```
Raw Data -> Validation -> Cleaning -> Feature Engineering -> Training Data
```

Key requirements:
- Schema validation before any transformation (fail fast on bad data)
- Data versioning — snapshot the training set, not just the model
- Point-in-time correctness — labels must only use features available before the event

**Stage 3: Model development**

```
EDA -> Feature Selection -> Model Training -> Hyperparameter Tuning -> Evaluation
```

Track every experiment: hyperparameters, metrics, data version, code commit, environment. Without experiment tracking you cannot explain why the current model is better than the previous one.

**Stage 4: Model deployment**

```
Model Registry -> Staging -> A/B Testing -> Production
```

Never deploy directly from an experiment. Register the artifact, validate it in staging, then route a controlled slice of production traffic to it.

**Stage 5: Monitoring and maintenance**

```
Performance Monitoring -> Drift Detection -> Retraining -> Feedback Loop
```

Monitoring is not optional. Models degrade without any code change because the world changes. The monitoring loop closes the lifecycle.

---

## Data Management

### The problem

In software, a bad config file causes an immediate crash. In ML, bad or shifted data causes a silent accuracy degradation that may take weeks to surface in business metrics. By then, the root cause is buried under months of production traffic.

### The core insight

Data must be versioned with the same discipline as code. Every training run must be reproducible from a specific data snapshot plus a specific code commit. Data quality failures must be caught before training, not discovered after deployment.

### The mechanics

**Data versioning with DVC**

```bash
# Initialize DVC
dvc init

# Track training data (DVC stores the file hash, git tracks the .dvc pointer)
dvc add data/train.csv
git add data/train.csv.dvc .gitignore
git commit -m "Add training data v1.0"

# Push data to remote storage (S3, GCS, Azure Blob)
dvc push

# Reproduce exact dataset for a historical experiment
git checkout <commit-hash>
dvc pull
```

*What breaks:* DVC tracks files by hash — if upstream data is regenerated with the same name but different content, the hash changes but the pointer file in git does not update automatically. You must explicitly re-add and commit the new `.dvc` file.

---

**Data quality checks with Great Expectations**

```python
from great_expectations import DataContext

context = DataContext()
expectation_suite = context.create_expectation_suite("training_data_suite")
batch = context.get_batch("training_data")

# Schema expectations
batch.expect_column_to_exist("user_id")
batch.expect_column_values_to_not_be_null("user_id")

# Range expectations
batch.expect_column_values_to_be_between("age", min_value=0, max_value=120)

# Set membership expectations
batch.expect_column_values_to_be_in_set("country", ["US", "UK", "CA", "AU"])

# Distribution expectations
batch.expect_column_mean_to_be_between("purchase_amount", min_value=10, max_value=500)

# Validate — fail the pipeline if expectations are violated
results = context.run_validation_operator("action_list", batch)
assert results["success"], f"Data validation failed: {results}"
```

*What breaks:* Expectations set on training data may not reflect production distribution changes. Expectations need to be updated when the legitimate data distribution evolves (e.g., new product lines, new geographies).

---

**Feature stores**

The problem feature stores solve: the same feature is computed slightly differently in the training pipeline (Python, daily batch) and the serving pipeline (Java, real-time). The model trains on one distribution and is served another.

Feature stores enforce a single definition for each feature, shared between training and serving:

- **Offline store** (Parquet/Delta Lake): historical features for training, with point-in-time joins
- **Online store** (Redis/DynamoDB): pre-materialized features for inference, < 10ms lookup

Benefits:
- Feature definitions are canonical — computed once, used everywhere
- Point-in-time correctness prevents data leakage in training
- Feature reuse across teams reduces duplicate computation
- Consistent features between training and serving eliminates training-serving skew

---

## Model Development

### The problem

Without experiment tracking, "the model we deployed last month" is whatever is on whoever's laptop. You cannot compare experiments fairly, you cannot reproduce a result, and you cannot explain to stakeholders why Model B outperforms Model A.

### The core insight

Every experiment is a versioned artifact: code + data + hyperparameters + metrics + environment. If you cannot reproduce an experiment from its metadata, the experiment did not happen for production purposes.

### The mechanics

**MLflow experiment tracking**

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("fraud_detection_v2")

with mlflow.start_run():
    # Log all hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("class_weight", "balanced")

    # Train
    model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                   class_weight="balanced")
    model.fit(X_train, y_train)

    # Log metrics
    mlflow.log_metric("train_auc", roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    mlflow.log_metric("val_auc", roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    mlflow.log_metric("val_precision", precision_score(y_val, model.predict(X_val)))
    mlflow.log_metric("val_recall", recall_score(y_val, model.predict(X_val)))

    # Log model artifact and feature importance
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("feature_importance.png")
```

---

**Model registry lifecycle**

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register the best run's model
run_id = best_run.info.run_id
mlflow.register_model(f"runs:/{run_id}/model", "fraud_detector")

# Lifecycle: None -> Staging -> Production -> Archived
client.transition_model_version_stage(
    name="fraud_detector", version=1, stage="Staging"
)

# After staging validation
client.transition_model_version_stage(
    name="fraud_detector", version=1, stage="Production"
)
```

Registry stages:
- **Development:** Experimental models, not validated
- **Staging:** Validated offline, pending production testing
- **Production:** Serving live traffic
- **Archived:** Deprecated, kept for audit and rollback

*What breaks:* Model registry lifecycle is only meaningful if promotion is gated by validation criteria. Without automated gates, "Production" becomes a meaningless label that humans set by hand without evidence.

---

## Deployment Strategies

### The problem

You have a new model that beats the current production model on your evaluation set. How do you replace the production model without breaking the service, and how do you verify it actually performs better on real users?

### The core insight

Every deployment strategy trades safety against speed. The higher the risk of the change (new architecture, new feature schema, large distribution difference), the more conservative the rollout strategy should be.

### The mechanics

**Batch inference**

When: process large datasets offline (daily recommendations, weekly credit scores, nightly reports)

```
Batch Data -> Prediction Job (Spark/Airflow) -> Results Storage -> Serving Layer
```

Pros: high throughput, can use complex/slow models, cost-effective
Cons: predictions become stale between batch runs

**Real-time inference**

When: low latency required (< 100ms), user-facing request-response

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
pipeline = joblib.load("model_with_preprocessing.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    prediction = pipeline.predict([data["features"]])
    return jsonify({"prediction": int(prediction[0])})
```

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

A model that was 94% accurate at deployment is now 79% accurate six months later. No code changed. The degradation happened gradually. Nobody noticed because nobody was watching the right signals.

### The core insight

ML monitoring has two layers that classic DevOps monitoring does not have: *data drift* (the inputs changed) and *model quality* (the input-to-output relationship changed). System metrics (latency, error rate) tell you the service is running. They do not tell you the model is still useful.

### The mechanics

**Model performance monitoring**

```yaml
# Prometheus alert rule
groups:
  - name: ml_model_alerts
    rules:
      - alert: ModelAccuracyDrop
        expr: model_accuracy_gauge < 0.85
        for: 5m
        annotations:
          summary: "Model accuracy dropped below 0.85 threshold"
      - alert: PredictionLatencyHigh
        expr: histogram_quantile(0.99, model_prediction_duration_seconds) > 0.2
        for: 2m
```

**Data drift detection**

```python
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, CatTargetDriftTab

# Compare training reference data vs current production window
dashboard = Dashboard(tabs=[DataDriftTab(), CatTargetDriftTab()])
dashboard.calculate(reference_data=training_df, current_data=production_window_df)
dashboard.save("drift_report.html")
```

Detection methods:
- **PSI (Population Stability Index):** PSI < 0.1 stable, 0.1-0.2 monitor, > 0.2 retrain
- **KL Divergence:** Statistical distance between distributions; sensitive to tail behavior
- **Kolmogorov-Smirnov Test:** Non-parametric test for continuous feature distribution shift

**Concept drift detection**

Concept drift means the relationship between X and y has changed — not just the input distribution, but what the inputs mean for the prediction.

Detection: monitor model performance metrics over time, compare predictions vs actuals, alert when performance degrades below threshold.

Solutions:
- Retrain model with recent data (scheduled or triggered by drift signal)
- Online learning with a sliding window (incremental partial_fit)
- Ensemble the current model with a recently-trained model

*What breaks:* You need ground truth labels to detect concept drift, and labels often arrive with a delay. For a loan default prediction model, you may not know whether a borrower defaulted for 6-12 months. Use proxy signals (early payment behavior) or plan for delayed evaluation.

---

## CI/CD for ML

### The problem

Your training pipeline takes 4 hours to run. Your team manually evaluates the results, makes a judgment call about whether to deploy, and pushes to production by hand. This is not repeatable, not auditable, and does not scale.

### The core insight

The ML CI/CD pipeline is a DevOps pipeline with three additional gates: data validation, model training, and model evaluation. The same discipline of "never deploy without passing all gates" applies — but the gates are different.

### The mechanics

**ML pipeline stages**

1. Code commit triggers CI
2. Unit tests (data processing logic, feature engineering)
3. Data validation (schema checks, range checks, distribution checks)
4. Model training (on versioned data snapshot)
5. Model evaluation (vs threshold AND vs current production model)
6. Register model if evaluation passes
7. Optional: human approval gate for sensitive models
8. Deploy to staging
9. Shadow or canary deployment on production traffic
10. Promote to full production

**GitHub Actions example**

```yaml
name: ML Pipeline

on:
  push:
    branches: [main]
  schedule:
    - cron: "0 2 * * *"  # Nightly retraining

jobs:
  train-and-evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run unit tests
        run: pytest tests/unit/ -v

      - name: Validate data
        run: python scripts/validate_data.py
        # Fails if Great Expectations suite fails

      - name: Train model
        run: python scripts/train.py
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

      - name: Evaluate model
        run: python scripts/evaluate.py --min-auc 0.85 --min-precision 0.80

      - name: Register model
        if: success()
        run: python scripts/register_model.py --stage Staging
```

*What breaks:* Nightly retraining without data freshness checks can train on stale or corrupted data. Always gate training on data validation. "If success" in the register step must check actual metric thresholds, not just that training completed.

---

**Safety checks before promotion to production**

- Minimum accuracy threshold vs baseline
- No performance regression vs current production model
- Shadow mode testing (at least 24 hours of shadow comparison)
- Gradual rollout with monitoring at each step

---

## Infrastructure and Tools

### The problem

Every ML team rediscovers the same infrastructure problems: where do I store trained models, how do I scale training, how do I serve predictions under load, how do I track what is running in production?

### The core insight

The ML infrastructure stack mirrors the software infrastructure stack, with specialized components at each layer for the unique requirements of data and model management.

### The mechanics

**Cloud ML platforms**

| Platform | ML Services | Best for |
|----------|-------------|----------|
| AWS | SageMaker, EC2, Lambda | Mature ecosystem, broad integrations |
| GCP | Vertex AI, AI Platform | TensorFlow integration, AutoML |
| Azure | Azure ML, Databricks | Enterprise Azure integration, MLflow native support |

**Containerization for ML**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifact AND preprocessing pipeline together
COPY model_with_preprocessing.pkl .
COPY app.py .

EXPOSE 8080

CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8080", "app:app"]
```

Key point: the model artifact and preprocessing pipeline must be in the same container image. If they are separate, version drift between model and preprocessing becomes possible.

**Kubernetes for model serving**

Treat model servers like any other stateless microservice:
- Replicas based on traffic (HPA on CPU/GPU utilization)
- Health checks: readiness (is the model loaded?) and liveness (is the server responding?)
- Resource limits: GPU memory must be specified explicitly or pods will be evicted at runtime

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

The question "what model is in production and why?" must always be answerable. This requires:
- Version code (git), data (DVC or snapshot hash), models (registry), environment (Docker)
- Set random seeds in all training scripts
- Log every experiment with full metadata (hyperparameters, data version, code commit)
- Use Docker for consistent training environments — not conda environments on individual machines

### Testing

ML systems need four categories of tests, not just one:
- **Unit tests:** individual data processing functions, feature transformations
- **Integration tests:** end-to-end pipeline from raw data to prediction
- **Model validation tests:** accuracy thresholds, fairness checks, regression vs baseline
- **Data quality checks:** schema, ranges, distributions, missing value rates

### Monitoring

Set up monitoring before deployment, not after the first incident:
- Track model performance metrics (with lag-aware strategies for delayed labels)
- Monitor data drift on all features with PSI/KS
- Set up alerts for latency regression and error rate spikes
- Log predictions with timestamps for offline analysis

### Security

- Encrypt data at rest and in transit
- Use secrets management (AWS Secrets Manager, HashiCorp Vault) — never hardcode credentials
- Implement access control (IAM, RBAC) on model endpoints and data stores
- Audit logging: who deployed what model, when, with what approval

### Cost optimization

- Use spot instances for training jobs (tolerate interruption with checkpointing)
- Auto-scaling for inference (scale to zero when idle, scale out on traffic)
- Model compression (quantization, pruning, distillation) to reduce serving cost
- Batch similar requests together for GPU efficiency

---

## MLOps Maturity Levels

### The problem

Teams do not jump from "notebook in production" to "full automation" overnight. Understanding where you are on the maturity ladder tells you which problems to solve next.

### Level 0: Manual process

Everything is manual: data preparation in notebooks, training by hand, deployment by copying files, no monitoring. The first production failure takes weeks to debug because there is no audit trail.

*Signal that you are here:* "The model" refers to a file on someone's laptop.

### Level 1: ML pipeline automation

Automated training pipeline, experiment tracking, model registry, basic monitoring. You can reproduce any experiment. You cannot continuously retrain or automatically deploy.

*Signal that you are here:* Experiments are logged, but deployments are still manual.

### Level 2: CI/CD pipeline automation

Automated testing, automated deployment with approval gates, continuous monitoring, automated retraining triggers. New data triggers a training run; if the new model passes evaluation, it is staged for deployment automatically.

*Signal that you are here:* The team is not involved in routine retraining — only in threshold calibration and incident response.

### Level 3: Production-grade MLOps

Advanced monitoring (drift detection, fairness monitoring), online learning for fast-moving distributions, multi-model management, feature stores, governance and compliance documentation. The ML system is self-healing for common failure modes.

*Signal that you are here:* A model degradation event triggers automated diagnosis, retraining, validation, and deployment with no human in the loop for the happy path.

---

## Interview Topics

**"Explain the difference between model drift and data drift"**

Data drift: the input feature distribution P(X) changes — different users, different behavior, different time of year. Concept drift: the relationship P(Y|X) changes — the same features now predict a different outcome (e.g., features that predicted fraud one way now predict a different fraud pattern). Both require retraining, but they have different root causes and different detection signals. Data drift is detectable without labels (compare input distributions). Concept drift requires labels (compare model accuracy on recent data vs historical data).

**"How would you deploy a model to production?"**

Start with shadow mode (zero risk, build confidence), move to canary (5% traffic, monitor business metrics for statistical significance), then full rollout. Package the model and preprocessing pipeline together. Define rollback triggers on accuracy drop, latency regression, and error rate spike. Monitor data drift from day one.

**"How do you ensure reproducibility?"**

Version code (git tag or commit hash), data (DVC pointer or snapshot hash in S3), model (registry version), environment (Docker image hash with pinned dependencies). Set random seeds. Log all hyperparameters with MLflow. Store training data snapshot separately from the live dataset — do not retrain on data that may have been modified.

**"How would you detect if a model is degrading?"**

Three layers of signals: input drift (PSI/KS on feature distributions, detectable within hours), prediction drift (output score distribution shifts, detectable within hours), and performance degradation (accuracy vs ground truth, detectable with label delay). Monitor all three. Do not wait for customer complaints.

**"Batch vs real-time inference trade-offs?"**

Batch: higher throughput, can use larger models, cost-effective, but predictions are stale by the time they are served. Real-time: low latency (target < 100ms), always fresh, but constrained model complexity and higher serving cost. The deciding factor is how quickly the ground truth changes and whether a stale prediction is still useful.

## Flashcards

**Schema validation before any transformation (fail fast on bad data)?** #flashcard
Schema validation before any transformation (fail fast on bad data)

**Data versioning?** #flashcard
snapshot the training set, not just the model

**Point-in-time correctness?** #flashcard
labels must only use features available before the event

**Offline store (Parquet/Delta Lake)?** #flashcard
historical features for training, with point-in-time joins

**Online store (Redis/DynamoDB)?** #flashcard
pre-materialized features for inference, < 10ms lookup

**Feature definitions are canonical?** #flashcard
computed once, used everywhere

**Point-in-time correctness prevents data leakage in training?** #flashcard
Point-in-time correctness prevents data leakage in training

**Feature reuse across teams reduces duplicate computation?** #flashcard
Feature reuse across teams reduces duplicate computation

**Consistent features between training and serving eliminates training-serving skew?** #flashcard
Consistent features between training and serving eliminates training-serving skew

**Development?** #flashcard
Experimental models, not validated

**Staging?** #flashcard
Validated offline, pending production testing

**Production?** #flashcard
Serving live traffic

**Archived?** #flashcard
Deprecated, kept for audit and rollback

**name?** #flashcard
ml_model_alerts

**alert?** #flashcard
ModelAccuracyDrop

**alert?** #flashcard
PredictionLatencyHigh

**PSI (Population Stability Index)?** #flashcard
PSI < 0.1 stable, 0.1-0.2 monitor, > 0.2 retrain

**KL Divergence?** #flashcard
Statistical distance between distributions; sensitive to tail behavior

**Kolmogorov-Smirnov Test?** #flashcard
Non-parametric test for continuous feature distribution shift

**Retrain model with recent data (scheduled or triggered by drift signal)?** #flashcard
Retrain model with recent data (scheduled or triggered by drift signal)

**Online learning with a sliding window (incremental partial_fit)?** #flashcard
Online learning with a sliding window (incremental partial_fit)

**Ensemble the current model with a recently-trained model?** #flashcard
Ensemble the current model with a recently-trained model

**cron?** #flashcard
"0 2   *"  # Nightly retraining

**uses?** #flashcard
actions/checkout@v3

**name?** #flashcard
Set up Python

**name?** #flashcard
Install dependencies

**name?** #flashcard
Run unit tests

**name?** #flashcard
Validate data

**name?** #flashcard
Train model

**name?** #flashcard
Evaluate model

**name?** #flashcard
Register model

**Minimum accuracy threshold vs baseline?** #flashcard
Minimum accuracy threshold vs baseline

**No performance regression vs current production model?** #flashcard
No performance regression vs current production model

**Shadow mode testing (at least 24 hours of shadow comparison)?** #flashcard
Shadow mode testing (at least 24 hours of shadow comparison)

**Gradual rollout with monitoring at each step?** #flashcard
Gradual rollout with monitoring at each step

**Replicas based on traffic (HPA on CPU/GPU utilization)?** #flashcard
Replicas based on traffic (HPA on CPU/GPU utilization)

**Health checks?** #flashcard
readiness (is the model loaded?) and liveness (is the server responding?)

**Resource limits?** #flashcard
GPU memory must be specified explicitly or pods will be evicted at runtime

**Version code (git), data (DVC or snapshot hash), models (registry), environment (Docker)?** #flashcard
Version code (git), data (DVC or snapshot hash), models (registry), environment (Docker)

**Set random seeds in all training scripts?** #flashcard
Set random seeds in all training scripts

**Log every experiment with full metadata (hyperparameters, data version, code commit)?** #flashcard
Log every experiment with full metadata (hyperparameters, data version, code commit)

**Use Docker for consistent training environments?** #flashcard
not conda environments on individual machines

**Unit tests?** #flashcard
individual data processing functions, feature transformations

**Integration tests?** #flashcard
end-to-end pipeline from raw data to prediction

**Model validation tests?** #flashcard
accuracy thresholds, fairness checks, regression vs baseline

**Data quality checks?** #flashcard
schema, ranges, distributions, missing value rates

**Track model performance metrics (with lag-aware strategies for delayed labels)?** #flashcard
Track model performance metrics (with lag-aware strategies for delayed labels)

**Monitor data drift on all features with PSI/KS?** #flashcard
Monitor data drift on all features with PSI/KS

**Set up alerts for latency regression and error rate spikes?** #flashcard
Set up alerts for latency regression and error rate spikes

**Log predictions with timestamps for offline analysis?** #flashcard
Log predictions with timestamps for offline analysis

**Encrypt data at rest and in transit?** #flashcard
Encrypt data at rest and in transit

**Use secrets management (AWS Secrets Manager, HashiCorp Vault)?** #flashcard
never hardcode credentials

**Implement access control (IAM, RBAC) on model endpoints and data stores?** #flashcard
Implement access control (IAM, RBAC) on model endpoints and data stores

**Audit logging?** #flashcard
who deployed what model, when, with what approval

**Use spot instances for training jobs (tolerate interruption with checkpointing)?** #flashcard
Use spot instances for training jobs (tolerate interruption with checkpointing)

**Auto-scaling for inference (scale to zero when idle, scale out on traffic)?** #flashcard
Auto-scaling for inference (scale to zero when idle, scale out on traffic)

**Model compression (quantization, pruning, distillation) to reduce serving cost?** #flashcard
Model compression (quantization, pruning, distillation) to reduce serving cost

**Batch similar requests together for GPU efficiency?** #flashcard
Batch similar requests together for GPU efficiency
