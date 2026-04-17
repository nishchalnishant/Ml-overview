# MLOps — Machine Learning Operations

**The pitch:** You already run pipelines, environments, and rollouts. MLOps is what happens when your **release artifact** isn’t only a container — it’s also **data lineage**, **model weights**, and **behavior under drift**. Think **Azure DevOps** meets **science**: same gates, messier inputs.

**Quick Azure bridge:** *Azure ML jobs* ≈ build agents for training; *model registry* ≈ artifact feed; *managed endpoints / AKS* ≈ your serving tier; *Application Insights + custom metrics* ≈ SLIs for latency **and** quality.

---

## Table of Contents

1. [MLOps Overview](#mlops-overview)
2. [ML Lifecycle](#ml-lifecycle)
3. [Data Management](#data-management)
4. [Model Development](#model-development)
5. [Deployment Strategies](#deployment-strategies)
6. [Monitoring & Observability](#monitoring--observability)
7. [CI/CD for ML](#cicd-for-ml)
8. [Infrastructure & Tools](#infrastructure--tools)
9. [Best Practices](#best-practices)

---

## MLOps Overview

**Definition:** MLOps is how DS + Ops ship models **reliably**: automate the path from data to production, keep experiments reproducible, and watch for the silent bugs (drift) that unit tests will never catch.

**Fashion analogy (one line):** A model in prod is a **look** on a runway — lighting changes (data), trends move (distribution), and the same outfit can “fail” in a new season. MLOps is the **styling + fittings + backup plan** so the show doesn’t flop.

### Key Goals
- **Automation:** Pipeline from data refresh → train → evaluate → register → deploy
- **Reproducibility:** Same commit + data snapshot + seed → same-ish model
- **Scalability:** Train big when needed; serve lean when traffic spikes
- **Reliability:** SLAs for latency **and** model quality over time
- **Collaboration:** Shared contracts on features, data, and release stages

### MLOps vs DevOps

| **Aspect** | **DevOps** | **MLOps** |
|-----------|-----------|----------|
| **Artifacts** | Code, binaries | Code + data + models |
| **Testing** | Unit, integration tests | Data validation, model validation |
| **Deployment** | Continuous deployment | Gradual rollout, A/B testing |
| **Monitoring** | System metrics | Model performance, data drift |
| **Versioning** | Code versions | Code + data + model versions |

**Mini pop quiz:** *Name one thing you version in MLOps that classic DevOps rarely versions.* → **Training data** (or features, or eval sets).

---

## ML Lifecycle

**MI-style one-liner:** This isn’t one super over — it’s a **season**: define the game plan, train hard, pick the playing XI (deploy), then **read the pitch** every match (monitor) and swap players (retrain) before you lose the trophy.

### 1. Problem Definition
- Define business problem
- Set success metrics (business + ML metrics)
- Establish baselines
- Determine data availability

### 2. Data Collection & Preparation
```
Raw Data → Cleaning → Feature Engineering → Training Data
```
- Data validation
- Missing value handling
- Outlier detection
- Feature transformations
- Data versioning (DVC, Delta Lake)

### 3. Model Development
```
EDA → Feature Selection → Model Training → Hyperparameter Tuning → Evaluation
```
- Experiment tracking (MLflow, Weights & Biases)
- Model versioning
- Hyperparameter optimization
- Cross-validation

### 4. Model Deployment
```
Model Registry → Staging → A/B Testing → Production
```
- Model packaging
- Containerization (Docker)
- Deployment strategy (canary, blue-green)
- API development (REST, gRPC)

### 5. Monitoring & Maintenance
```
Performance Monitoring → Drift Detection → Retraining → Feedback Loop
```
- Model performance metrics
- Data drift / concept drift detection
- Alerting and incident response
- Continuous retraining

---

## Data Management

**Why this section hits different:** In DevOps, bad config is visible. In ML, **bad or shifted data** can look fine until revenue walks out the door. Version data like you version **infra-as-code** — because it *is* code for your model.

### Data Versioning
**Tools:** DVC, Delta Lake, lakeFS

**Why?**
- Reproduce experiments
- Track data lineage
- Rollback to previous versions

**Example with DVC:**
```bash
# Initialize DVC
dvc init

# Track data
dvc add data/train.csv
git add data/train.csv.dvc .gitignore
git commit -m "Add training data"

# Push to remote storage
dvc push

# Pull data version
dvc pull
```

### Data Quality Checks
**Tools:** Great Expectations, Evidently

**Checks:**
- Schema validation (correct types, columns)
- Range checks (min/max values)
- Distribution checks (mean, std)
- Missing value thresholds
- Duplicate detection

**Example:**
```python
from great_expectations import DataContext

context = DataContext()

# Define expectations
expectation_suite = context.create_expectation_suite("my_suite")
batch = context.get_batch("my_data")

# Add expectations
batch.expect_column_values_to_not_be_null("user_id")
batch.expect_column_values_to_be_between("age", 0, 120)
batch.expect_column_values_to_be_in_set("country", ["US", "UK", "CA"])

# Validate
results = context.run_validation_operator("action_list", batch)
```

### Feature Stores
**Tools:** Feast, Tecton, AWS SageMaker Feature Store

**Benefits:**
- Share features across teams
- Consistent features (training vs serving)
- Low-latency serving
- Point-in-time correctness (no data leakage)

---

## Model Development

**Remaster analogy:** Training runs are like **remastering a classic track** — same song (objective), new mix (hyperparameters), and you A/B whether the audience actually likes the louder drums (metrics). Ship the mix that wins, not the one that *felt* clever in the studio.

### Experiment Tracking
**Tools:** MLflow, Weights & Biases, Neptune.ai

**Track:**
- Hyperparameters
- Metrics (accuracy, loss)
- Artifacts (models, visualizations)
- Code version (git commit)
- Environment (dependencies)

**MLflow Example:**
```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("n_estimators", 100)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("feature_importance.png")
```

### Model Registry
**Purpose:** Centralized model storage with versioning and lifecycle management

**Stages:**
- **Development:** Experimental models
- **Staging:** Models ready for testing
- **Production:** Models serving live traffic
- **Archived:** Deprecated models

**MLflow Model Registry:**
```python
# Register model
mlflow.register_model("runs:/<run_id>/model", "my_model")

# Transition to staging
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="my_model",
    version=1,
    stage="Staging"
)

# Transition to production
client.transition_model_version_stage(
    name="my_model",
    version=1,
    stage="Production"
)
```

---

## Deployment Strategies

**Deploy = release strategy.** Batch is your **nightly batch job**; real-time is your **always-on API**. Canary / blue-green / shadow are the same ideas you already use — just add **model-specific** success metrics beside CPU and p99.

### Batch Inference
**When:** Process large datasets offline (e.g., daily recommendations, weekly reports)

**Architecture:**
```
Batch Data → Prediction Job (Spark/Airflow) → Results Storage → Serving Layer
```

**Pros:** High throughput, can use complex models, cost-effective
**Cons:** Not real-time, stale predictions

**Tools:** Apache Airflow, AWS Batch, Google Cloud Dataflow

---

### Real-time Inference
**When:** Low latency required (< 100ms), request-response pattern

**Architecture:**
```
User Request → API Gateway → Model Server → Response
```

**Serving Options:**

| **Option** | **Use Case** | **Latency** |
|-----------|-------------|------------|
| **REST API** | General purpose | 50-200ms |
| **gRPC** | Low latency | 10-50ms |
| **Serverless** | Variable traffic | 100-500ms |
| **Edge** | Ultra-low latency | < 10ms |

**Flask Example:**
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = data["features"]
    prediction = model.predict([features])
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

---

### Deployment Patterns

**1. Canary Deployment**
- Gradually route traffic to new model (5% → 25% → 50% → 100%)
- Monitor performance at each stage
- Rollback if problems detected

**2. Blue-Green Deployment**
- Run old (blue) and new (green) models in parallel
- Switch traffic instantly
- Easy rollback

**3. Shadow Mode**
- New model receives traffic but doesn't serve responses
- Compare predictions offline
- Zero risk deployment

**4. A/B Testing**
- Split traffic between models (e.g., 50/50)
- Measure business metrics
- Choose winner based on data

---

## Monitoring & Observability

**Observability here means:** dashboards for **latency + errors** *and* **quality** (accuracy, calibration, business KPIs). Drift is your **“something changed upstream”** alert — like a silent dependency upgrade, but for the world.

### Model Performance Monitoring
**Metrics to Track:**
- Accuracy, precision, recall, F1
- Prediction latency (p50, p95, p99)
- Throughput (requests/second)
- Error rate

**Tools:** Prometheus, Grafana, DataDog

**Example Alert:**
```yaml
alert: ModelAccuracyDrop
expr: model_accuracy < 0.85
for: 5m
annotations:
  summary: "Model accuracy dropped below threshold"
```

### Data Drift Detection
**Data Drift:** Input feature distributions change over time

**Detection Methods:**
- **PSI (Population Stability Index):** Measures distribution change
- **KL Divergence:** Statistical distance between distributions
- **Kolmogorov-Smirnov Test:** Statistical test for distribution difference

**Tools:** Evidently AI, WhyLabs, Fiddler

**Example with Evidently:**
```python
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab

dashboard = Dashboard(tabs=[DataDriftTab()])
dashboard.calculate(reference_data, current_data)
dashboard.save("drift_report.html")
```

### Concept Drift Detection
**Concept Drift:** Relationship between features and target changes

**Detection:**
- Monitor model performance metrics over time
- Compare predictions vs actuals
- Alert when performance degrades

**Solutions:**
- Retrain model with recent data
- Online learning (incremental updates)
- Ensemble with recent models

---

## CI/CD for ML

**How would you wire this in Azure Pipelines?** Trigger on PR or schedule → **validate data** (Great Expectations-style) → **train** in a container or Azure ML job → **evaluate** vs thresholds → **register** model → optional **approval gate** → deploy to staging → **shadow or canary** → production. Same spine as app CI/CD — extra validation steps where the risk lives.

### Continuous Integration
**Steps:**
1. Code commit (git push)
2. Run tests (unit, integration)
3. Data validation
4. Model training
5. Model validation
6. Register model if metrics pass threshold

**Example with GitHub Actions:**
```yaml
name: ML Pipeline

on: [push]

jobs:
  train-and-evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run tests
        run: pytest tests/
      
      - name: Validate data
        run: python scripts/validate_data.py
      
      - name: Train model
        run: python scripts/train.py
      
      - name: Evaluate model
        run: python scripts/evaluate.py
      
      - name: Register model
        if: success()
        run: python scripts/register_model.py
```

### Continuous Deployment
**Automated Pipeline:**
```
Code Change → Tests → Train → Evaluate → Stage → A/B Test → Production
```

**Safety Checks:**
- Minimum accuracy threshold
- Performance regression tests
- Shadow mode testing
- Gradual rollout with monitoring

---

## Infrastructure & Tools

**Kubernetes angle:** Treat model servers like any other workload — replicas, probes, autoscaling — but add **GPU** node pools and **batching** where needed. Your **Helm chart** is just wrapping another stateless service — with bigger Docker images.

### Cloud Platforms

| **Platform** | **ML Services** | **Strengths** |
|-------------|----------------|--------------|
| **AWS** | SageMaker, EC2, Lambda | Mature ecosystem, broad services |
| **GCP** | Vertex AI, AI Platform | TensorFlow integration, AutoML |
| **Azure** | Azure ML, Databricks | Enterprise integration |

### Containerization
**Docker for ML:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl .
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
```

**Kubernetes for Orchestration:**
- Scaling (replicas based on load)
- Load balancing
- Health checks and auto-restart
- Resource management

### Model Serving Platforms

| **Tool** | **Best For** | **Features** |
|---------|-------------|------------|
| **TensorFlow Serving** | TensorFlow models | High performance, versioning |
| **TorchServe** | PyTorch models | Multi-model serving |
| **ONNX Runtime** | Cross-framework | Optimized inference |
| **Seldon Core** | Kubernetes native | ML deployment on K8s |
| **KFServing** | Cloud-native | Serverless, autoscaling |

---

## Best Practices

**Quick thought experiment:** *If tomorrow’s data silently shifts 5%, which practice saves you first — monitoring, versioning, or automated retrain?* (Trick question: you need **all three**, but **versioning + monitoring** buys you time to retrain calmly.)

### 1. Reproducibility
 Version everything: code, data, models, environment
 Use Docker for consistent environments
 Set random seeds
 Document dependencies (requirements.txt, conda env)

### 2. Testing
 Unit tests for data processing logic
 Integration tests for pipelines
 Model validation tests (accuracy thresholds)
 Data quality checks

### 3. Monitoring
 Track model performance metrics
 Monitor data drift
 Set up alerts for degradation
 Log predictions for debugging

### 4. Security
 Encrypt data at rest and in transit
 Use secrets management (AWS Secrets Manager, Vault)
 Implement access control (IAM, RBAC)
 Audit logging

### 5. Cost Optimization
 Use spot instances for training
 Auto-scaling for inference
 Model compression (quantization, pruning)
 Batch similar requests

---

## MLOps Maturity Levels

**Levels = crawl → walk → sprint.** Level 0 is “hero in a notebook.” Level 2+ is where your **Azure DevOps brain** finally feels at home: pipelines, gates, observability, **repeatable** change.

### Level 0: Manual Process
- Manual data preparation
- Notebooks for training
- Manual deployment
- No monitoring

### Level 1: ML Pipeline Automation
- Automated training pipeline
- Experiment tracking
- Model registry
- Basic monitoring

### Level 2: CI/CD Pipeline Automation
- Automated testing
- Automated deployment
- Continuous monitoring
- Automated retraining triggers

### Level 3: Production-Grade MLOps
- Advanced monitoring (drift detection)
- Online learning
- Multi-model management
- Feature stores
- Governance and compliance

---

## Interview Topics

**You’ve got this** — answer in three beats: **definition** → **mechanism** → **production tradeoff** (latency, cost, safety).

**Common Questions:**

1. **"Explain the difference between model drift and data drift"**
   > Data drift: Input distributions change. Concept drift: X→y relationship changes. Both require retraining.

2. **"How would you deploy a model to production?"**
   > Containerize (Docker) → Model registry → Staging → A/B test → Gradual rollout → Monitor → Full deployment

3. **"How do you ensure reproducibility?"**
   > Version code (git), data (DVC), models (registry), environment (Docker), set random seeds

4. **"How would you detect if a model is degrading?"**
   > Monitor performance metrics, track data drift, compare predictions vs actuals, set up alerts

5. **"Batch vs real-time inference trade-offs?"**
   > Batch: Higher throughput, offline, complex models. Real-time: Low latency, online, simpler models.

---

## Resources

**Books:**
- Building Machine Learning Powered Applications (Emmanuel Ameisen)
- Machine Learning Engineering (Andriy Burkov)
- Designing Machine Learning Systems (Chip Huyen)

**Courses:**
- Made With ML - MLOps
- DeepLearning.AI - MLOps Specialization

**Tools Repository:**
- [Awesome MLOps](https://github.com/visenger/awesome-mlops)

**Blogs:**
- Google Cloud MLOps
- AWS Machine Learning Blog
- Neptune.ai Blog
