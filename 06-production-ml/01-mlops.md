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
3. [Data Management](#data-management) — DVC, data versioning, schema validation
4. [Model Development](#model-development) — MLflow, W&B, experiment tracking, hyperparameter sweeps, model registry
5. [Deployment Strategies](#deployment-strategies) — REST serving, gRPC serving, canary, blue-green, shadow, A/B testing
6. [Monitoring and Observability](#monitoring-and-observability) — drift detection, PSI, KS test, concept drift
7. [CI/CD for ML](#cicd-for-ml) — GitHub Actions, Airflow ML pipelines, Kubeflow Pipelines
8. [Infrastructure and Tools](#infrastructure-and-tools) — SageMaker, Vertex AI, Kubernetes, Triton, TorchServe
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

**Weights & Biases (W&B) experiment tracking**

W&B is the standard choice for deep learning teams. Advantages over MLflow: richer visualizations (gradient histograms, confusion matrices inline), built-in sweep (hyperparameter search), better collaboration (shareable dashboards), and native integration with PyTorch/HuggingFace.

```python
import wandb
import torch
from torch import nn

# Initialize run: all config, code, and artifacts tracked automatically
run = wandb.init(
    project="fraud-detection",
    name="transformer-v3",
    config={
        "learning_rate": 3e-4,
        "batch_size": 256,
        "n_epochs": 50,
        "model_arch": "transformer",
        "data_version": "v4.1.2",
    },
    tags=["fraud", "transformer", "production-candidate"]
)
config = wandb.config  # access config as config.learning_rate

for epoch in range(config.n_epochs):
    train_loss, val_auc = train_epoch(model, optimizer)

    # Log metrics — appear live in W&B dashboard
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/auc": val_auc,
        "val/pr_auc": compute_pr_auc(model, val_loader),
    })

    # Log a confusion matrix as a W&B table
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None, y_true=y_true, preds=y_pred, class_names=["legit", "fraud"]
    )})

# Save model artifact with metadata and lineage
artifact = wandb.Artifact(
    name="fraud-transformer-v3",
    type="model",
    description="Transformer fraud detector, Q1 2024 data",
    metadata={"val_auc": 0.94, "data_version": "v4.1.2"}
)
artifact.add_file("model.pt")
run.log_artifact(artifact)

# Link to W&B Model Registry for promotion tracking
run.link_artifact(artifact, target_path="fraud-detection/fraud-model")
run.finish()
```

**W&B Sweeps** (hyperparameter search):

```python
# Define search space
sweep_config = {
    "method": "bayes",  # Bayesian optimization (vs grid, random)
    "metric": {"name": "val/auc", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "batch_size": {"values": [64, 128, 256, 512]},
        "n_layers": {"values": [2, 4, 6, 8]},
        "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.5},
    }
}

sweep_id = wandb.sweep(sweep_config, project="fraud-detection")

def train_sweep():
    with wandb.init() as run:
        config = run.config
        model = build_model(config)
        val_auc = train_and_evaluate(model, config)
        wandb.log({"val/auc": val_auc})

wandb.agent(sweep_id, function=train_sweep, count=50)  # run 50 trials
```

**MLflow vs W&B — when to choose**:

```
Criterion                | MLflow                  | W&B
-------------------------|-------------------------|---------------------------
Primary language         | Python (any framework)  | Python (deep learning focus)
Self-hosted option       | Yes (free)              | Yes (W&B Server, paid)
Hyperparameter sweeps    | MLflow + Optuna manual  | Built-in Sweeps with Bayes
Visualization depth      | Basic charts            | Rich: gradients, media, 3D
Model registry           | Yes (MLflow Registry)   | Yes (W&B Artifacts + Registry)
HuggingFace integration  | Manual                  | Native (transformers/diffusers)
Best for                 | Any ML, enterprise,     | Deep learning, research,
                         | open-source preference  | collaborative teams
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

**Real-time inference — REST**

When: low latency required (< 100ms), user-facing request-response, simple JSON interface

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np

app = FastAPI()
pipeline = joblib.load("model_with_preprocessing.pkl")

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    X = np.array(req.features).reshape(1, -1)
    prob = pipeline.predict_proba(X)[0, 1]
    return PredictResponse(prediction=int(prob > 0.5), probability=float(prob))
```

**Real-time inference — gRPC**

When: internal microservices, strict latency SLAs (< 20ms), high throughput, or binary payloads (embeddings, images). gRPC is ~7× faster than REST for equivalent payloads due to Protobuf binary encoding and HTTP/2 multiplexing.

Step 1 — define the service contract in Protobuf:

```protobuf
// fraud_service.proto
syntax = "proto3";
package fraud;

service FraudService {
    rpc Predict (PredictRequest) returns (PredictResponse);
    rpc PredictBatch (PredictBatchRequest) returns (PredictBatchResponse);
}

message PredictRequest {
    string entity_id    = 1;
    repeated float features = 2;   // flat feature vector
}

message PredictResponse {
    float fraud_probability = 1;
    int32 decision          = 2;   // 0=legit, 1=fraud
    string model_version    = 3;
}

message PredictBatchRequest {
    repeated PredictRequest requests = 1;
}

message PredictBatchResponse {
    repeated PredictResponse responses = 1;
}
```

Step 2 — generate Python stubs and implement the server:

```python
# compile: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. fraud_service.proto

import grpc
from concurrent import futures
import fraud_service_pb2 as pb2
import fraud_service_pb2_grpc as pb2_grpc
import joblib, numpy as np

class FraudServicer(pb2_grpc.FraudServiceServicer):
    def __init__(self):
        self.model = joblib.load("model.pkl")
        self.model_version = "v2.4.1"

    def Predict(self, request, context):
        X = np.array(request.features).reshape(1, -1)
        prob = float(self.model.predict_proba(X)[0, 1])
        return pb2.PredictResponse(
            fraud_probability=prob,
            decision=int(prob > 0.5),
            model_version=self.model_version,
        )

    def PredictBatch(self, request, context):
        responses = []
        X = np.array([r.features for r in request.requests])
        probs = self.model.predict_proba(X)[:, 1]
        for i, req in enumerate(request.requests):
            responses.append(pb2.PredictResponse(
                fraud_probability=float(probs[i]),
                decision=int(probs[i] > 0.5),
                model_version=self.model_version,
            ))
        return pb2.PredictBatchResponse(responses=responses)

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ]
    )
    pb2_grpc.add_FraudServiceServicer_to_server(FraudServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()
```

Step 3 — client call:

```python
channel = grpc.insecure_channel("fraud-service:50051")
stub = pb2_grpc.FraudServiceStub(channel)
response = stub.Predict(pb2.PredictRequest(entity_id="u123", features=[0.5, 1.2, -0.3]))
print(response.fraud_probability)   # 0.87
```

**REST vs gRPC comparison**:

```
Criterion           | REST + JSON           | gRPC + Protobuf
--------------------|----------------------|---------------------------
Payload size        | ~3-10× larger        | Binary, compact
Latency             | Higher (serialization)| ~3-7× faster at scale
Client generation   | Manual / OpenAPI     | Auto-generated from .proto
Streaming support   | No (without WS)      | Bidirectional streaming
Browser support     | Native               | Needs grpc-web proxy
Human readable      | Yes                  | No (use grpcurl for debug)
Best for            | External APIs, simple| Internal services, high TPS
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

**Apache Airflow for ML pipelines**

Airflow is a battle-tested DAG scheduler for ML pipelines. Use it when you need complex dependency management, retries with backoff, SLA alerting, and integration with many data systems.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta

default_args = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["ml-alerts@company.com"],
}

with DAG(
    dag_id="fraud_model_retraining",
    default_args=default_args,
    schedule_interval="0 2 * * *",   # nightly at 2am UTC
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "fraud", "production"],
) as dag:

    # 1. Validate data freshness before training
    def validate_data(**context):
        run_date = context["ds"]  # execution date as YYYY-MM-DD
        count = check_data_freshness(run_date)
        if count < 100_000:
            raise ValueError(f"Insufficient training data: {count} rows for {run_date}")

    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        provide_context=True,
    )

    # 2. Train model
    train_task = BashOperator(
        task_id="train_model",
        bash_command="""
            python /opt/ml/train.py \
                --run-date {{ ds }} \
                --mlflow-uri {{ var.value.mlflow_uri }}
        """,
        env={"MLFLOW_TRACKING_URI": "{{ var.value.mlflow_uri }}"},
    )

    # 3. Evaluate: branch on whether new model beats threshold
    def evaluate_and_branch(**context):
        new_auc = get_latest_run_metric("val_auc")
        prod_auc = get_production_model_metric("val_auc")
        if new_auc >= 0.85 and new_auc >= prod_auc * 0.99:  # no regression
            return "register_model"
        return "notify_evaluation_failure"

    evaluate_task = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_and_branch,
        provide_context=True,
    )

    register_task = PythonOperator(
        task_id="register_model",
        python_callable=register_to_staging,
    )

    notify_failure_task = PythonOperator(
        task_id="notify_evaluation_failure",
        python_callable=send_slack_alert,
    )

    # DAG dependency graph
    validate_task >> train_task >> evaluate_task >> [register_task, notify_failure_task]
```

**Airflow best practices for ML**:
- Use `XComs` sparingly — pass large artifacts via S3 paths, not through XCom
- Set `max_active_runs=1` for retraining DAGs to avoid parallel training on the same dataset
- Use `SLAMiss` callbacks to alert if training doesn't complete within a time budget
- External task sensors (`ExternalTaskSensor`) to wait for upstream data pipelines before training

---

**Kubeflow Pipelines for ML orchestration**

Kubeflow Pipelines is designed for ML workloads running on Kubernetes. Advantages: native GPU scheduling, component caching (skip re-running unchanged steps), artifact lineage, and portability across GCP/AWS/on-prem.

```python
import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from typing import NamedTuple

# Define pipeline components as Python functions
# Each function runs in its own container

@func_to_container_op
def validate_data_op(data_path: str) -> str:
    """Validate data quality; return path to validated data."""
    import great_expectations as gx
    context = gx.get_context()
    results = context.run_checkpoint(checkpoint_name="training_data_check")
    if not results.success:
        raise ValueError(f"Data validation failed: {results.statistics}")
    return data_path

@func_to_container_op
def train_model_op(data_path: str, learning_rate: float, n_estimators: int) -> NamedTuple(
    "TrainOutput", [("model_uri", str), ("val_auc", float)]
):
    """Train model; return model URI and validation AUC."""
    import mlflow
    with mlflow.start_run() as run:
        model = train(data_path, learning_rate=learning_rate, n_estimators=n_estimators)
        val_auc = evaluate(model)
        mlflow.log_metric("val_auc", val_auc)
        model_uri = mlflow.sklearn.log_model(model, "model").model_uri
    return model_uri, val_auc

@func_to_container_op
def register_model_op(model_uri: str, val_auc: float, threshold: float = 0.85) -> str:
    """Register model if it passes threshold."""
    if val_auc < threshold:
        raise ValueError(f"Model val_auc {val_auc} below threshold {threshold}")
    import mlflow
    return mlflow.register_model(model_uri, "fraud-detector").version

# Define the pipeline
@dsl.pipeline(
    name="Fraud Detection Retraining",
    description="Nightly fraud model retraining pipeline"
)
def fraud_retraining_pipeline(
    data_path: str = "s3://ml-data/training/latest/",
    learning_rate: float = 0.001,
    n_estimators: int = 500,
):
    # Component 1: Validate data
    validate_step = validate_data_op(data_path)

    # Component 2: Train (depends on validate)
    train_step = train_model_op(
        data_path=validate_step.output,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
    )
    train_step.set_gpu_limit(1)           # request 1 GPU
    train_step.set_memory_limit("32G")    # request 32GB RAM
    train_step.set_cpu_limit("8")         # request 8 CPUs

    # Component 3: Register (depends on train)
    register_step = register_model_op(
        model_uri=train_step.outputs["model_uri"],
        val_auc=train_step.outputs["val_auc"],
    )

# Compile and submit
kfp.compiler.Compiler().compile(fraud_retraining_pipeline, "pipeline.yaml")

client = kfp.Client(host="http://kubeflow-pipelines:8888")
run = client.create_run_from_pipeline_func(
    fraud_retraining_pipeline,
    arguments={"learning_rate": 0.001, "n_estimators": 500},
)
```

**Airflow vs Kubeflow Pipelines**:

```
Criterion              | Airflow                    | Kubeflow Pipelines
-----------------------|----------------------------|---------------------------------
Native GPU support     | Via KubernetesPodOperator  | Native (set_gpu_limit)
Component caching      | No                         | Yes (content-addressed cache)
UI                     | DAG + task status          | Visual pipeline + artifact viewer
Scaling                | Celery/K8s executors       | Kubernetes-native
Best for               | Complex DAGs, data eng,    | ML workflows, GPU training,
                       | many integrations          | artifact lineage, Kubernetes shops
Trigger model          | Time-based, sensor-based   | API, recurring run, event trigger
```

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

---

**AWS SageMaker — end-to-end ML pipeline**

SageMaker provides managed training, hosting, and pipeline orchestration. Use `sagemaker.workflow.pipeline.Pipeline` to chain steps; each runs in its own managed container.

```python
import boto3
import sagemaker
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.sklearn import SKLearn
from sagemaker.processing import SKLearnProcessor

session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Pipeline parameters (overridable at runtime)
auc_threshold = ParameterFloat(name="AucThreshold", default_value=0.85)
data_uri      = ParameterString(name="DataUri", default_value="s3://my-bucket/data/")

# 1. Data processing step
sklearn_proc = SKLearnProcessor(
    framework_version="1.2-1",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    role=role,
)
processing_step = ProcessingStep(
    name="PreprocessData",
    processor=sklearn_proc,
    inputs=[sagemaker.processing.ProcessingInput(source=data_uri, destination="/opt/ml/processing/input")],
    outputs=[sagemaker.processing.ProcessingOutput(source="/opt/ml/processing/output", destination="s3://my-bucket/processed/")],
    code="preprocess.py",
)

# 2. Training step
estimator = SKLearn(
    entry_point="train.py",
    framework_version="1.2-1",
    instance_type="ml.m5.2xlarge",
    instance_count=1,
    role=role,
    hyperparameters={"n-estimators": 500, "max-depth": 8},
    metric_definitions=[{"Name": "val:auc", "Regex": "val_auc: ([0-9\\.]+)"}],
)
training_step = TrainingStep(
    name="TrainFraudModel",
    estimator=estimator,
    inputs={"train": sagemaker.inputs.TrainingInput(s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri)},
    depends_on=[processing_step],
)

# 3. Conditional registration (only if AUC passes threshold)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.steps import RegisterModel

register_step = RegisterModel(
    name="RegisterFraudModel",
    estimator=estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/json"],
    response_types=["application/json"],
    model_package_group_name="FraudDetectionModels",
    approval_status="PendingManualApproval",
)

condition_step = ConditionStep(
    name="CheckAUC",
    conditions=[ConditionGreaterThanOrEqualTo(
        left=training_step.properties.FinalMetricDataList["val:auc"].Value,
        right=auc_threshold,
    )],
    if_steps=[register_step],
    else_steps=[],
)

# 4. Assemble and run pipeline
pipeline = Pipeline(
    name="FraudDetectionPipeline",
    parameters=[auc_threshold, data_uri],
    steps=[processing_step, training_step, condition_step],
)
pipeline.upsert(role_arn=role)
execution = pipeline.start(parameters={"AucThreshold": 0.87})
```

**SageMaker Endpoints** — deploy the registered model:

```python
from sagemaker.sklearn.model import SKLearnModel

sm_client = boto3.client("sagemaker")

# Approve the model package first, then deploy
sm_client.update_model_package(
    ModelPackageArn="arn:aws:sagemaker:us-east-1:123456789:model-package/...",
    ModelApprovalStatus="Approved",
)

model = SKLearnModel(
    model_data="s3://my-bucket/model.tar.gz",
    role=role,
    framework_version="1.2-1",
    entry_point="inference.py",
)
predictor = model.deploy(
    initial_instance_count=2,
    instance_type="ml.c5.xlarge",
    endpoint_name="fraud-detector-prod",
    data_capture_config=sagemaker.model_monitor.DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=20,          # capture 20% of requests for monitoring
        destination_s3_uri="s3://my-bucket/data-capture/",
    ),
)
```

---

**Google Vertex AI — end-to-end ML pipeline**

Vertex AI Pipelines uses the same Kubeflow Pipelines SDK but runs fully managed on GCP. Native integration with BigQuery for training data and Cloud Storage for artifacts.

```python
from kfp import dsl
from kfp.v2 import compiler
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

PROJECT_ID = "my-gcp-project"
REGION     = "us-central1"
PIPELINE_ROOT = f"gs://my-bucket/pipeline-root"

# Define components using pre-built Google Cloud Pipeline Components
from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp

@dsl.pipeline(name="fraud-vertex-pipeline", pipeline_root=PIPELINE_ROOT)
def fraud_vertex_pipeline(
    bq_source: str = "bq://project.dataset.fraud_training",
    threshold: float = 0.85,
):
    # 1. Export training data from BigQuery to GCS
    export_data = BigqueryQueryJobOp(
        project=PROJECT_ID,
        location="US",
        query=f"SELECT * FROM `{bq_source}` WHERE DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)",
    )

    # 2. Custom training job (runs in a container you provide)
    train_job = CustomTrainingJobOp(
        project=PROJECT_ID,
        display_name="fraud-training",
        worker_pool_specs=[{
            "machine_spec": {"machine_type": "n1-standard-8", "accelerator_type": "NVIDIA_TESLA_T4", "accelerator_count": 1},
            "replica_count": 1,
            "container_spec": {
                "image_uri": f"gcr.io/{PROJECT_ID}/fraud-trainer:latest",
                "args": ["--gcs-input", export_data.outputs["destination_table"],
                         "--gcs-output", f"{PIPELINE_ROOT}/model/"],
            },
        }],
    ).after(export_data)

    # 3. Upload model to Vertex AI Model Registry
    model_upload = ModelUploadOp(
        project=PROJECT_ID,
        display_name="fraud-detector",
        artifact_uri=train_job.outputs["gcs_output_directory"],
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
    ).after(train_job)

    # 4. Deploy to endpoint
    endpoint = EndpointCreateOp(project=PROJECT_ID, display_name="fraud-endpoint")
    ModelDeployOp(
        model=model_upload.outputs["model"],
        endpoint=endpoint.outputs["endpoint"],
        dedicated_resources_machine_type="n1-standard-4",
        dedicated_resources_min_replica_count=2,
        dedicated_resources_max_replica_count=10,  # autoscale up to 10
        traffic_percentage=100,
    )

# Compile and submit
compiler.Compiler().compile(fraud_vertex_pipeline, package_path="pipeline.json")

aiplatform.init(project=PROJECT_ID, location=REGION)
job = aiplatform.PipelineJob(display_name="fraud-run", template_path="pipeline.json")
job.run(sync=True)
```

**Vertex AI Model Monitoring** — detect drift on deployed endpoints automatically:

```python
from google.cloud.aiplatform_v1beta1 import ModelDeploymentMonitoringJobServiceClient
from google.cloud.aiplatform_v1beta1.types import (
    ModelDeploymentMonitoringJob,
    ModelDeploymentMonitoringObjectiveConfig,
    ThresholdConfig,
)

monitoring_job = ModelDeploymentMonitoringJob(
    display_name="fraud-drift-monitor",
    endpoint=endpoint.resource_name,
    model_deployment_monitoring_objective_configs=[
        ModelDeploymentMonitoringObjectiveConfig(
            deployed_model_id=deployed_model_id,
            objective_config=ModelDeploymentMonitoringObjectiveConfig.ObjectiveConfig(
                training_dataset=ModelDeploymentMonitoringObjectiveConfig.TrainingDataset(
                    bigquery_source={"input_uri": f"bq://{bq_source}"},
                    target_field="label",
                ),
                training_prediction_skew_detection_config={
                    "skew_thresholds": {"feature_1": ThresholdConfig(value=0.3)},
                },
                prediction_drift_detection_config={
                    "drift_thresholds": {"feature_1": ThresholdConfig(value=0.3)},
                },
            ),
        )
    ],
    logging_sampling_strategy={"random_sample_config": {"sample_rate": 0.2}},
    model_deployment_monitoring_schedule_config={"monitor_interval": {"seconds": 3600}},
    alert_config={"email_alert_config": {"user_emails": ["ml-alerts@company.com"]}},
)
```

**SageMaker vs Vertex AI comparison**:

```
Criterion                | SageMaker (AWS)              | Vertex AI (GCP)
-------------------------|------------------------------|----------------------------------
Pipeline SDK             | SageMaker Pipelines (custom) | Kubeflow Pipelines SDK (open)
Training data source     | S3                           | BigQuery + GCS
Model registry           | Model Package Groups         | Vertex Model Registry
Auto-scaling endpoints   | Yes (target-tracking)        | Yes (min/max replicas)
Managed spot training    | Managed Spot Instances       | Preemptible VMs
Hyperparameter tuning    | SageMaker HPO (HyperOpt)    | Vertex Vizier (Bayesian)
Built-in monitoring      | Model Monitor (drift + bias) | Vertex Model Monitoring
Best for                 | AWS-native teams, broad DS   | GCP/BigQuery shops, GKE teams
```

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
