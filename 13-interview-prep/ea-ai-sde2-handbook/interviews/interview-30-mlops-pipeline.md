# Interview 30 — End-to-End MLOps Pipeline Design
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are the Lead AI Engineer on a new live-service game. Your team has developed a highly accurate ML model for "In-Game Ad Targeting" (predicting which cosmetic item a player is most likely to buy). 

The Data Scientists have handed you a Jupyter Notebook that trains the model perfectly on their laptop. Your task is to **design and implement the end-to-end MLOps pipeline** to take this from a Jupyter Notebook to a robust, automated production system handling millions of daily active users.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- CI/CD & Orchestration (How is code/data actually deployed?)
- Model Registry (How do we track version 1 vs version 2?)
- Deployment Strategy (Canary? Shadow? Blue/Green?)
- Feature Store (How are the features in the notebook replicated in production?)
- Monitoring (How do we know if the model is failing in production?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"How often does the model need to be retrained?"**
   → *Answer: Weekly. New cosmetics drop every Tuesday, so the model needs to learn the new items fast.*

2. **"What is the deployment strategy? Can we A/B test it against the old heuristic?"**
   → *Answer: Yes, you should definitely propose a safe rollout strategy. A bad model will tank our revenue.*

3. **"Where is the inference happening? Batch or Real-time?"**
   → *Answer: Real-time. When the player opens the store menu, we have 200ms to return the targeted item.*

---

## Part 4 — Expected Assumptions

- **Architecture:** 
  1. Data/Feature Pipeline (Airflow/Feast).
  2. Training Pipeline (Kubeflow/Vertex AI).
  3. Model Registry (MLflow).
  4. CI/CD (GitHub Actions).
  5. Serving (FastAPI / Seldon Core).
- **Safe Rollout:** Shadow deployment followed by Canary.

---

## Part 5 — High-Level Solution

```
  [1. Code & CI/CD]
  Data Scientist pushes `train.py` to GitHub ➔ GitHub Actions runs unit tests
       │
       ▼
  [2. Orchestration (Airflow / Kubeflow)]
  Triggered weekly. 
  Extracts Features ➔ Trains XGBoost ➔ Evaluates vs Baseline
       │
       ▼
  [3. Model Registry (MLflow)]
  Logs hyperparams, metrics, and saves the `.pkl` artifact. 
  Tags as "Staging".
       │
       ▼
  [4. Deployment (Shadow Mode ➔ Canary)]
  Model deployed to Kubernetes.
  Inference Service pulls "Production" model from MLflow.
       │
       ▼
  [5. Monitoring (Evidently / Grafana)]
  Logs Predictions and Features. Alerts on Drift or Latency.
```

**Core ML Component:** Connecting the fragmented stages of ML development (Data, Code, Model, Serving) into a traceable, automated loop. If revenue drops, we must be able to trace the live model back to the exact Git commit and data snapshot that generated it.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Disentangling the Notebook
- Rip out the logic from the Jupyter Notebook into modular Python scripts: `data_loader.py`, `train.py`, `evaluate.py`.
- Containerize the training job using Docker so it runs identically on a laptop and on AWS.

### Step 2: The Model Registry (MLflow)
- Never store models on random S3 buckets or hard drives.
- Use MLflow to log the exact configuration. If Model V42 tanks revenue, you can click on V42 in the MLflow UI, see it used `max_depth=15`, and see the Git hash of the code.

### Step 3: Safe Rollout Strategy
- **Shadow Mode:** Deploy the new ML model alongside the legacy system. The game client calls the legacy system for the UI, but asynchronously calls the new ML model in the background. Log both predictions. Compare them offline to ensure the ML model doesn't crash or output garbage.
- **Canary:** Once Shadow mode passes, route 5% of real players to the ML model. Monitor revenue. If it holds, scale to 100%.

---

## Part 7 — Complete Python Code

*Note: We will write the MLflow training and registration script, which is the heart of MLOps tracking.*

```python
"""
train_pipeline.py - Automated Training and MLflow Registration
"""
import logging
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
import mlflow
import mlflow.xgboost
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def run_training_pipeline(data_path: str, max_depth: int, learning_rate: float):
    # MLflow Setup
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment("in_game_store_targeting")
    
    with mlflow.start_run() as run:
        logger.info(f"Started Run ID: {run.info.run_id}")
        
        # 1. Log Parameters (Traceability)
        mlflow.log_params({
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "data_source": data_path
        })
        
        # 2. Load Data (Mocked)
        logger.info("Loading data...")
        # df = pd.read_parquet(data_path)
        # X_train, y_train, X_test, y_test = train_test_split(...)
        
        # Mocking data
        import numpy as np
        X_train, y_train = np.random.rand(100, 5), np.random.randint(0, 2, 100)
        X_test, y_test = np.random.rand(20, 5), np.random.randint(0, 2, 20)
        
        # 3. Train Model
        logger.info("Training XGBoost...")
        model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, eval_metric="logloss")
        model.fit(X_train, y_train)
        
        # 4. Evaluate
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        loss = log_loss(y_test, preds)
        
        # 5. Log Metrics
        mlflow.log_metrics({
            "val_auc": auc,
            "val_log_loss": loss
        })
        logger.info(f"Validation AUC: {auc:.4f}")
        
        # 6. Save and Register Model
        # This uploads the binary to the MLflow artifact store (S3)
        mlflow.xgboost.log_model(
            xgb_model=model, 
            artifact_path="model",
            registered_model_name="Store_Targeting_Model"
        )
        
        # 7. Automated Promotion Logic
        if auc > 0.75: # Baseline threshold
            logger.info("Model meets baseline. Promoting to Staging.")
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name="Store_Targeting_Model",
                version=run.info.run_id, # Simplified versioning for demo
                stage="Staging"
            )
        else:
            logger.warning("Model failed baseline. Discarding.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="s3://data/latest.parquet")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()
    
    # run_training_pipeline(args.data, args.depth, args.lr)
```

---

## Part 8 — Deployment

### Serving Infrastructure
- The Inference Service (FastAPI) does not hardcode the model file.
- On startup, it calls the MLflow API: `GET /models/Store_Targeting_Model/Production`.
- It downloads the `.pkl` file into memory.
- If we want to deploy a new model, we just click "Promote to Production" in the MLflow UI. The Inference Service detects the Webhook, downloads the new weights, and hot-swaps them with zero downtime.

---

## Part 9 — Unit Testing

```python
import xgboost as xgb
import numpy as np

def test_model_inference_latency():
    # Model serving must meet the 200ms SLA
    import time
    
    # Mock model and data
    model = xgb.XGBClassifier()
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    single_payload = np.random.rand(1, 5)
    
    # Measure latency
    start = time.perf_counter()
    pred = model.predict_proba(single_payload)
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    
    assert latency_ms < 50.0 # Strict bound to leave room for network IO
    assert pred.shape == (1, 2)
```

---

## Part 10 — Integration Testing

- **Pipeline E2E Test:**
  - Create a staging DAG in Airflow.
  - Feed it 10 rows of mock parquet data.
  - Assert that the DAG successfully completes the feature extraction, MLflow logs the run, the Model is registered, and the Staging Inference Service can successfully download and return a `200 OK` prediction using the new model ID.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Data Drift** | Retraining weekly assumes the data doesn't break mid-week. Add an asynchronous Drift Detection job (using Evidently AI) that monitors the live API payloads. If PSI (Population Stability Index) spikes above 0.2, fire a PagerDuty alert and auto-trigger the Airflow training DAG early. |
| **Feature Store Sync** | The Jupyter Notebook probably does `df.fillna(0)` or `df.groupby()`. If you copy-paste that logic into the FastAPI service, you will create Training-Serving Skew. You must use a **Feature Store (Feast)** to guarantee the exact same feature engineering logic is used for batch training and real-time inference. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Jupyter Notebooks vs Python Scripts | Notebooks are incredible for fast EDA and visualization. But they are catastrophic for version control (Git diffs of JSON are unreadable), testing, and automated deployment. Always enforce a hard rule: Code must be refactored into modular `.py` files before hitting the main branch. |
| Automatic vs Manual Deployment | Automatically promoting a model to "Production" if `AUC > 0.8` is dangerous (the data might be leaked, yielding a fake high AUC). Best practice: Fully automate CI/CD up to "Staging". Require a human Data Scientist to review the MLflow metrics and click the final "Promote to Prod" button. |

---

## Part 13 — Alternative Approaches

1. **Managed Platforms (Vertex AI / SageMaker):** Instead of manually stringing together Airflow, MLflow, and Kubernetes, use GCP Vertex AI Pipelines. It handles the metadata tracking, orchestration, and managed endpoint scaling natively. Costs more, but saves months of DevOps engineering time.
2. **Serverless Inference:** If store traffic is highly bursty (huge spikes when a patch drops, dead at 3 AM), deploy the inference API on AWS Lambda. It scales to 10,000 QPS instantly and scales to zero to save costs, avoiding the need to manage Kubernetes HPA rules.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| The 0-Day Feature Bug | The game updates, and a feature `player_level` accidentally gets logged as a String instead of an Integer. The FastAPI service crashes trying to feed a string into XGBoost. Revenue halts. | **Schema Validation.** Implement Pydantic or Great Expectations on the API input. If the schema fails, do not pass to the ML model; immediately return a default fallback item (e.g., the most popular cosmetic) so the UI doesn't break. |
| OOM on Deployment | The new model is highly complex and uses 4GB of RAM. The Kubernetes pods only have 2GB allocated. The pods crash-loop on deployment. | Run a memory-profiling step during CI/CD. Assert that the loaded `.pkl` footprint is `< Max_Allowed_RAM` before pushing the Docker image. |

---

## Part 15 — Debugging

**Symptom:** The model is deployed. Revenue drops by 30%. The MLflow metrics show the model had a fantastic AUC of 0.95 during training.

**Debugging steps:**
1. This is the definition of **Data Leakage (Time Travel)**. 
2. The Data Scientist joined the target variable (did they buy the item?) with a feature that happens *after* the purchase (e.g., `item_equipped_status = True`).
3. During training, the model learned: `if item_equipped == True, then predict Buy = 100%`.
4. In production, at the exact moment the store is opened, `item_equipped` is always `False`. The model fails.
5. **Fix:** Rollback immediately. Audit the feature engineering SQL queries and enforce strict Point-in-Time (AS OF) joins to prevent leaking future information into the training set.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `api_500_error_rate` | > 0.1% → Model is crashing on malformed input data. |
| `prediction_distribution_skew` | E.g., The model suddenly predicts 100% of users should buy Item A, and 0% for Item B. This indicates feature drift or a catastrophic model collapse. |
| `p99_inference_latency` | > 150ms → Auto-scale the Kubernetes pods. |

---

## Part 17 — Production Improvements

1. **A/B Testing Platform Integration:** Deploying via Canary isn't enough. We must pipe the ML model's predictions into the core A/B Testing platform (Interview 19). Route 50% of users to ML Model, 50% to Heuristics. Wait 7 days. Prove statistically that the ML model increases Total Revenue (the true business metric) before fully committing.
2. **Feedback Loop Automation:** When a player actually buys an item, push that event to a Kafka topic. Stream that Kafka topic into the Feature Store to instantly update the player's `last_purchase_timestamp` feature, ensuring the ML model's next prediction is hyper-accurate.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"The data scientists want to use a massive Deep Learning model (PyTorch) instead of XGBoost. How does your deployment architecture (FastAPI) change to support high-throughput PyTorch inference?"**
2. **"We have millions of players, but only 0.1% of them actually buy cosmetics. Your automated training pipeline throws an error because the XGBoost model predicts '0' for everyone. How do you automate handling class imbalance in the pipeline?"**
3. **"Explain the difference between a Model Registry (like MLflow) and a standard Git Repository (like GitHub). Why can't we just commit the `.pkl` files to Git?"**

---

## Part 19 — Ideal Answers

**Q1 (Deep Learning Serving):**
> "FastAPI with standard Python workers (Uvicorn) is terrible for PyTorch because of the Global Interpreter Lock (GIL) and lack of dynamic batching. I would rip out FastAPI and replace it with **NVIDIA Triton Inference Server** or **TorchServe**. Triton automatically handles dynamic batching (grouping 16 incoming HTTP requests into a single GPU matrix multiplication) and runs the C++ backend directly, increasing throughput by 10x."

**Q2 (Automated Imbalance Handling):**
> "We must automate the calculation of the `scale_pos_weight` parameter for XGBoost. In the `train.py` script, we dynamically calculate `len(negative_samples) / len(positive_samples)`. If 99.9% are negative, this sets the weight to 1000, forcing the algorithm to penalize false negatives heavily. We should also switch the MLflow automated promotion metric from AUC (which is misleading on highly imbalanced data) to **PR-AUC** (Precision-Recall Area Under Curve)."

**Q3 (Git vs Model Registry):**
> "Git is optimized for text (code diffs). Large binary files (`.pkl` or `.pth`) bloat the repository, making cloning incredibly slow, and Git provides no meaningful diff for a binary file. A Model Registry stores the binary payload in cheap object storage (S3), while storing the metadata (hyperparameters, metrics, Git commit hash, lineage) in a relational database. It provides an API to tag models as 'Staging/Prod', which Git does not support natively for artifacts."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Understands the complete lifecycle: Data -> Train -> Log -> Serve -> Monitor.
- Explicitly warns against deploying Jupyter Notebooks to production.
- Solves the Data Leakage debugging scenario accurately.
- Explains why Git is bad for binaries and why MLflow is required.

### Hire
- Sets up a logical MLflow tracking script.
- Understands the concepts of Shadow deployments and Canary rollouts.
- Knows how to catch inference latency issues.

### Lean Hire
- Suggests manually copy-pasting the `.pkl` file to the production server via FTP or SSH. (Requires interviewer correction to push them towards CI/CD).
- Thinks MLOps just means putting the model in a Docker container, missing the tracking/monitoring aspects.

### Lean No Hire
- Argues that running the Jupyter Notebook in a `screen` session on the server is an acceptable production deployment strategy.

### No Hire
- Does not know what MLOps, CI/CD, or a Model Registry is.
- Cannot explain how to serve an ML model via an API.
