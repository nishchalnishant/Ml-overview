# Interview 02 — Player Churn Prediction & Intervention (Apex Legends)
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer working on the Apex Legends LiveOps team. Player retention is critical. Recently, there has been a noticeable drop in month-over-month active users.
Your task is to **design and implement a machine learning system to predict player churn and trigger automated interventions** (e.g., granting an XP boost, free cosmetics, or sending a targeted "we miss you" email) to retain them before they leave for good.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Definition of "Churn" (e.g., 7 days inactive? 14 days? 30 days?)
- Prediction horizon (are we predicting churn tomorrow or next week?)
- Type of game data available (telemetry, purchase history, social graph)
- Business constraints on interventions (is there a budget for giving away cosmetics?)
- Real-time vs. Batch prediction (do we predict while they play, or run overnight?)
- Evaluation metrics (Accuracy vs. Precision/Recall vs. Lift/Uplift)
- Cost of False Positives (giving away free stuff to players who wouldn't have churned) vs. False Negatives (losing a player).

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"How exactly are we defining 'churn' for a live service game like Apex?"**
   → *Answer: No login for 14 consecutive days.*

2. **"What is our prediction window?"**
   → *Predicting on day T if they will churn between day T+1 and T+14.*

3. **"Does this need to run in real-time, or is a daily batch job sufficient?"**
   → *Daily batch job. We run predictions at 2 AM UTC for all active players.*

4. **"What is the cost associated with interventions?"**
   → *Emails are free. XP boosts have a small virtual economy cost. Legendary cosmetics have a high perceived cost (cannibalizes future sales). We want to minimize giving cosmetics to people who don't need them (False Positives).*

5. **"What data is available in the data warehouse?"**
   → *Daily aggregates of playtime, matches played, kill/death ratio, battle pass progression, friends online, and store purchases.*

6. **"Are we just predicting the probability of churn, or predicting the *effect* of the intervention?"**
   → *For v1, just predict the probability of churn. If P(churn) is high, trigger an intervention based on a static rule.*

---

## Part 4 — Expected Assumptions

- System runs as a daily batch pipeline (Airflow + Snowflake/BigQuery).
- Total active player base to score daily: ~5-10 million.
- Features are aggregated up to day T.
- High recall is less important than high precision for expensive interventions (cosmetics). High recall is fine for cheap interventions (emails).
- Model output: P(churn) ∈ [0, 1].

---

## Part 5 — High-Level Solution

```
  Data Warehouse (Snowflake)
       │
       ▼ (Nightly ETL)
  Feature Store / S3 Parquet
  ┌─────────────────────────────────────┐
  │  Rolling 7d/14d/30d aggregates      │
  │  Session length trends, K/D trends  │
  │  Social activity (friends played)   │
  └─────────────────────────────────────┘
       │
       ▼ (Airflow DAG)
  Batch Inference Job (Spark / Python script)
  ┌─────────────────────────────────────┐
  │  Load pre-trained XGBoost Model     │
  │  Score 10M players                  │
  │  Output: player_id, P(churn)        │
  └─────────────────────────────────────┘
       │
       ▼
  Intervention Rules Engine
  ┌─────────────────────────────────────┐
  │ P > 0.90 ➔ High Risk ➔ Cosmetic   │
  │ P > 0.75 ➔ Med Risk  ➔ XP Boost   │
  │ P > 0.60 ➔ Low Risk  ➔ Email      │
  └─────────────────────────────────────┘
       │
       ▼
  LiveOps API / Marketing Service
```

**Core ML component:** An XGBoost or LightGBM binary classifier trained on historical player snapshots. Given a snapshot on day T, did they churn in [T+1, T+14]?

---

## Part 6 — Step-by-Step Implementation

### Step 1: Data Preparation (Labeling)
- Define a cohort: Players active on Day T.
- Feature window: Day T-30 to Day T.
- Target label: 1 if `max(login_date)` between T+1 and T+14 is NULL, else 0.

### Step 2: Feature Engineering
- **Engagement:** Total playtime (7d, 30d), matches played, battle pass level.
- **Skill/Frustration:** K/D ratio trend (7d avg / 30d avg), number of consecutive losses, report/quit rate.
- **Social:** Days since playing with a friend, % of matches in a pre-made squad.
- **Monetization:** Days since last purchase, total lifetime value (LTV).

### Step 3: Model Training
- Algorithm: XGBoost. Handles missing values well, robust to tabular data, fast inference.
- Imbalance handling: Churn is typically imbalanced (e.g., 10% churn rate). Use `scale_pos_weight`.
- Metrics: PR-AUC (Precision-Recall Area Under Curve), F1-score, and decile lift charts.

### Step 4: Batch Inference Pipeline
- Fetch latest features for all currently active players.
- Apply model `predict_proba()`.
- Save results to a DB table or S3 bucket.

### Step 5: Rules Engine & API
- Read the daily scores.
- Apply thresholds to segment players.
- Call the internal LiveOps API to grant items or trigger CRM emails.

---

## Part 7 — Complete Python Code

```python
"""
churn_pipeline.py - Batch inference for player churn prediction
"""
import logging
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
MODEL_PATH = "/models/xgboost_churn_v1.json"
DB_CONN = "postgresql://user:pass@db-host:5432/analytics"
LIVEOPS_API = "http://liveops-service.internal/api/v1/interventions"

def load_features(engine) -> pd.DataFrame:
    """Load daily features for active players."""
    query = """
    SELECT 
        player_id,
        playtime_7d,
        playtime_30d,
        kd_ratio_7d,
        kd_ratio_30d,
        battle_pass_level,
        friends_played_with_7d,
        days_since_last_purchase,
        consecutive_losses
    FROM player_daily_features
    WHERE is_active_today = TRUE
    """
    logger.info("Fetching features from DB...")
    return pd.read_sql(query, engine)

def predict_churn(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Run XGBoost inference."""
    logger.info("Loading model and predicting...")
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Exclude player_id for inference
    features = df.drop(columns=["player_id"])
    dmatrix = xgb.DMatrix(features)
    
    df["churn_prob"] = model.predict(dmatrix)
    return df[["player_id", "churn_prob"]]

def apply_interventions(scored_df: pd.DataFrame):
    """Rule-based intervention assignment."""
    logger.info("Applying intervention rules...")
    
    payloads = []
    for _, row in scored_df.iterrows():
        prob = row["churn_prob"]
        player_id = row["player_id"]
        
        intervention = None
        if prob >= 0.90:
            intervention = "free_cosmetic_lootbox"
        elif prob >= 0.75:
            intervention = "xp_boost_2x_48h"
        elif prob >= 0.60:
            intervention = "retention_email"
            
        if intervention:
            payloads.append({
                "player_id": player_id,
                "intervention_type": intervention,
                "reason": f"churn_risk_{prob:.2f}"
            })
            
    return payloads

def send_to_liveops(payloads: list):
    """Send payloads to the LiveOps microservice in batches."""
    logger.info(f"Sending {len(payloads)} interventions to LiveOps...")
    batch_size = 500
    for i in range(0, len(payloads), batch_size):
        batch = payloads[i:i+batch_size]
        try:
            resp = requests.post(f"{LIVEOPS_API}/batch", json={"interventions": batch})
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send batch {i}: {e}")
            # In production: implement retry logic / DLQ

def main():
    engine = create_engine(DB_CONN)
    df = load_features(engine)
    if df.empty:
        logger.warning("No players active today. Exiting.")
        return
        
    scored_df = predict_churn(df, MODEL_PATH)
    payloads = apply_interventions(scored_df)
    
    if payloads:
        send_to_liveops(payloads)
    else:
        logger.info("No interventions needed today.")

if __name__ == "__main__":
    main()
```

---

## Part 8 — Deployment

### Kubernetes (CronJob)
Since this is a daily batch job, it should run as a Kubernetes CronJob or via Apache Airflow.

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: churn-prediction-job
  namespace: ml-ops
spec:
  schedule: "0 2 * * *" # 2 AM UTC daily
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: churn-predictor
            image: ea-registry/churn-pipeline:1.0.0
            env:
              - name: DB_PASSWORD
                valueFrom:
                  secretKeyRef:
                    name: db-credentials
                    key: password
            resources:
              requests:
                memory: "4Gi"
                cpu: "1"
              limits:
                memory: "8Gi"
                cpu: "2"
          restartPolicy: OnFailure
```

### Monitoring & Logging
- **Logging:** Standard stdout JSON logs via fluentd to Splunk.
- **Monitoring:** Pushgateway for Prometheus to record batch metrics (e.g., `rows_processed`, `interventions_triggered`, `batch_duration_seconds`).

### CI/CD
- GitHub Actions triggered on PR merge.
- Runs `pytest` on data logic.
- Builds Docker image and pushes to ECR/Artifact Registry.

---

## Part 9 — Unit Testing

```python
import pandas as pd
import pytest
from churn_pipeline import apply_interventions

def test_apply_interventions():
    # Mock data
    data = {
        "player_id": ["p1", "p2", "p3", "p4"],
        "churn_prob": [0.95, 0.80, 0.65, 0.10]
    }
    df = pd.DataFrame(data)
    
    payloads = apply_interventions(df)
    
    # Expect 3 interventions, p4 is ignored
    assert len(payloads) == 3
    
    # Check correct mapping
    interventions = {p["player_id"]: p["intervention_type"] for p in payloads}
    assert interventions["p1"] == "free_cosmetic_lootbox"
    assert interventions["p2"] == "xp_boost_2x_48h"
    assert interventions["p3"] == "retention_email"
    assert "p4" not in interventions
```

---

## Part 10 — Integration Testing

- Use a localized SQLite database or Testcontainers (Postgres).
- Insert 10 synthetic player rows.
- Run `load_features()`.
- Load a dummy XGBoost model (or a very small mocked one) to run `predict_churn()`.
- Use `responses` or `responses_mock` library to intercept the `requests.post` to the LiveOps API and assert that the correct JSON payload was constructed.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Data Volume (10M+ rows)** | Pandas will OOM. Migrate the `predict_churn` step to PySpark using `xgboost.spark` or process in chunks/generators in Python. |
| **Feature Processing** | Push feature computation down to the Data Warehouse (Snowflake) using SQL, rather than doing it in Python memory. |
| **LiveOps API Rate Limits** | A large batch of interventions might DDoS the internal LiveOps API. Use asynchronous batching, rate limiters, or write directly to an SQS/Kafka queue instead of direct HTTP calls. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Batch vs. Real-time inference | Batch is vastly cheaper and simpler to build, but you might miss the exact moment a player rage-quits. |
| Rule-based thresholds | Simple to implement and interpret, but rigid. Doesn't account for the *incrementality* (Uplift) of the intervention. |
| XGBoost vs. Deep Learning | XGBoost is faster to train/tune on tabular data and handles missing values. DL might capture complex sequential patterns (LSTMs on session history) but costs much more to deploy. |

---

## Part 13 — Alternative Approaches

1. **Uplift Modeling (Advanced):** Instead of predicting $P(Churn)$, predict the Treatment Effect: $P(Churn|Intervention) - P(Churn|Control)$. This prevents giving free items to "Sure Things" (will stay anyway) or "Lost Causes" (will leave regardless), focusing only on "Persuadables".
2. **Survival Analysis (Cox Proportional Hazards):** Predicts *when* the user will churn, not just a binary flag for a 14-day window.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Data Warehouse delay | Features are from Day T-2 instead of T-1 | Alert if data freshness SLA is violated. Pause pipeline. |
| LiveOps API down | Interventions lost | Implement DLQ (Dead Letter Queue) or write to Kafka instead of direct HTTP calls. |
| Model concept drift | Massive increase in false positives | Anomaly detection on output distribution. If 50% of players are flagged > 0.90 suddenly, halt and page on-call. |

---

## Part 15 — Debugging

**Symptom:** The LiveOps team reports that we gave away 1,000,000 legendary cosmetics last night, destroying the game's economy.

**Debugging steps:**
1. Halt the CronJob immediately.
2. Check the raw model output distribution for yesterday. Did the average $P(churn)$ spike?
3. If scores spiked, check the input features. Did a new patch break the `playtime_7d` telemetry event, causing everyone's playtime to report as 0? (If playtime = 0, model thinks they are churning).
4. If scores are normal, check the rule engine. Was a threshold accidentally changed in config?

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `churn_batch_duration_seconds` | > 2 hours → Warning |
| `churn_predictions_generated` | < 1M or > 20M → Alert |
| `churn_interventions_triggered_total` | > 50,000/day → Critical Alarm |
| `feature_null_rate_percent` | > 5% → Warning |

**Business Metric Tracking (Lagging):**
- Monitor the actual 14-day retention rate of players assigned to the treatment buckets vs. a 5% global holdout (control group) that receives no interventions.

---

## Part 17 — Production Improvements

1. **Migrate to PySpark:** To handle larger player bases without OOM errors.
2. **Uplift Modeling:** As mentioned, transition from pure churn prediction to CATE (Conditional Average Treatment Effect) estimation.
3. **Continuous Feature Monitoring:** Implement Evidently AI or whylogs to detect data drift on input features before inference.
4. **Experimentation Platform Integration:** Automatically split the interventions into A/B/C tests to evaluate which intervention (XP vs Cosmetic) works best for which segment.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"You mentioned training on 'players active on Day T'. If a player hasn't logged in for 13 days, they are active on Day T-13, but not Day T. How does your model handle players who are already mid-churn?"**
2. **"How do you evaluate if your model is actually saving players, rather than just identifying people who were going to churn anyway?"**
3. **"A new season drops, completely changing player behavior and engagement metrics. How do you handle this sudden concept drift?"**
4. **"Your pandas dataframe takes up 30GB of RAM. How do you rewrite the inference step to run on a standard 4GB pod without using Spark?"**

---

## Part 19 — Ideal Answers

**Q1 (Mid-churn players):**
> "Ah, that's a great point. If we only score players active *today*, we miss players who stopped playing 3 days ago but haven't officially 'churned' (hit 14 days) yet. The cohort definition needs to be: 'Players who have logged in at least once in the last 14 days'. We score all of them. The `days_since_last_login` feature becomes the most critical predictor."

**Q2 (Evaluation of effectiveness):**
> "We must use a global holdout group. Randomly select 5% of players to never receive interventions, regardless of their churn score. We then compare the actual 14-day retention rate of the treated high-risk players against the control high-risk players. If the retention rate is the same, our intervention is useless, even if our model's predictions are perfectly accurate."

**Q3 (Season drop / Concept Drift):**
> "A new season causes massive covariate shift. The model will likely over-predict retention because everyone's playtime spikes. We need two things: (1) Include seasonal features like `days_since_season_start`. (2) Use a shorter training window (e.g., train on the last 14 days instead of 6 months) so the model adapts quickly, or use instance weighting to prioritize recent data."

**Q4 (OOM / Memory constraints):**
> "We can use Python generators and process the SQL query in chunks. Instead of `pd.read_sql`, we use `pd.read_sql(..., chunksize=100_000)`. We iterate over the chunks, run `predict_churn` on each chunk, yield the results, and send them to the API/DB. This keeps the memory footprint bounded to the size of a single chunk."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Immediately asks about the definition of churn and intervention costs.
- Recognizes the risk of giving away expensive items (False Positives) and tailors the threshold.
- Proposes Uplift modeling or at least mentions a control group for evaluation.
- Easily answers the memory chunking question (Q4).
- Understands the data leakage risks in defining the training time window.

### Hire
- Solid batch architecture (Airflow + SQL + Python/XGBoost).
- Uses sensible thresholds.
- Understands that a model predicting churn doesn't inherently solve churn without a valid intervention.
- Can write the chunking logic with a bit of prompting.

### Lean Hire
- Focuses too heavily on model hyperparameter tuning rather than the business logic.
- Struggles with the evaluation strategy (forgets the control group).
- Has a basic working script but misses edge cases (like error handling for the API call).

### Lean No Hire
- Proposes a real-time streaming architecture (Kafka/Flink) for a problem that is clearly a daily batch problem.
- Doesn't handle class imbalance.
- Fails the memory limitation question (Q4).

### No Hire
- Cannot write basic Pandas/Python code.
- Does not understand how to formulate a tabular dataset for a classification problem.
- Does not test or monitor the system.
