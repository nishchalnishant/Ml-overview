# Interview 21 — Automated Data Drift Detection Pipeline
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the ML Ops team. We have dozens of production ML models running across the company (e.g., Matchmaking, Churn Prediction, Fraud Detection). While the models are highly accurate on day one, their performance degrades over time because the underlying data distributions change as game patches are released.

Your task is to **design an Automated Data Drift Detection Pipeline** that monitors the input features of these models, detects when the data has shifted (Concept Drift / Covariate Shift), and alerts the data science team *before* model accuracy drops.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Metric types (How do we measure drift for continuous vs categorical features?)
- Baseline data (What are we comparing the live data against?)
- Scalability (Are we monitoring 10 features or 10,000 features?)
- False Alarms (How do we prevent alerting on expected seasonal shifts, like weekends?)
- Output action (Do we just send a Slack message, or do we auto-trigger retraining?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"What statistical tests are we using for continuous vs categorical features?"**
   → *Answer: I want you to propose the best methods. (Expected: KS-Test for continuous, PSI/Chi-Square for categorical).*

2. **"What is the baseline dataset for comparison?"**
   → *Answer: The exact dataset used to train the current production model.*

3. **"Do we have ground-truth labels in real-time?"**
   → *Answer: No. For churn prediction, we don't know if they churned until 30 days later. You must detect drift based on input features alone.*

4. **"How many features are we monitoring?"**
   → *Answer: Around 5,000 features across 20 models.*

---

## Part 4 — Expected Assumptions

- **Architecture:** Nightly batch job using Apache Airflow and PySpark/Pandas.
- **Metrics:** 
  - **Continuous:** Kolmogorov-Smirnov (KS) test or Population Stability Index (PSI).
  - **Categorical:** Population Stability Index (PSI) or Chi-Square test.
- **Storage:** Drift metrics are stored in a time-series database or warehouse (Snowflake) for dashboarding.

---

## Part 5 — High-Level Solution

```
  [Data Warehouse (Snowflake)]
  Training Data (Baseline)    Live Production Logs (Last 24h)
       │                              │
       ▼                              ▼
  [Airflow Nightly DAG: Drift Detection Job]
  ┌────────────────────────────────────────────────────────┐
  │ 1. For each feature, compute distribution histograms.  │
  │ 2. Compare Baseline vs Live distributions.             │
  │ 3. Calculate KS-Statistic and PSI scores.              │
  └────────────────────────────────────────────────────────┘
       │
       ▼ (Metrics DB)
  [Evaluation & Alerting]
  If PSI > 0.20 or KS p-value < 0.05 ➔ Feature has drifted.
       │
       ▼
  [Action]
  Slack Alert ➔ Data Scientist ➔ Manually trigger Retraining
```

**Core ML Component:** Implementing robust statistical tests (like Population Stability Index) that can handle large-scale data comparisons efficiently, and understanding the difference between covariate shift (features changing) and concept drift (target relationships changing).

---

## Part 6 — Step-by-Step Implementation

### Step 1: Data Extraction
- Pull the training data distribution (only needs to be computed once and cached).
- Pull a sliding window (e.g., last 7 days) of production inference logs. We use 7 days instead of 1 day to smooth out daily seasonality.

### Step 2: Statistical Calculation (PSI)
- **Population Stability Index (PSI)** is the industry standard in finance and ML for drift. 
- It works by binning the continuous data into deciles (10 buckets) based on the training data.
- Then, calculate the percentage of live data that falls into those same buckets.
- Formula: $PSI = \sum (\%Live - \%Train) \times \ln(\frac{\%Live}{\%Train})$

### Step 3: Alerting Thresholds
- PSI < 0.1: No significant drift.
- PSI 0.1 to 0.2: Moderate drift (Monitor).
- PSI > 0.2: Severe drift (Alert & Retrain).

---

## Part 7 — Complete Python Code

```python
"""
drift_monitor.py - Calculates Population Stability Index (PSI)
"""
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Calculates Population Stability Index for a continuous feature.
    expected: Training data array
    actual: Live production data array
    """
    # 1. Define bucket boundaries based on the expected (training) data deciles
    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    bins = np.percentile(expected, breakpoints)
    
    # Ensure min and max encapsulate all actual data to prevent out-of-bounds
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    # 2. Count frequencies in each bucket
    expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)
    
    # 3. Handle zero divisions (add small epsilon)
    epsilon = 0.0001
    expected_percents = np.where(expected_percents == 0, epsilon, expected_percents)
    actual_percents = np.where(actual_percents == 0, epsilon, actual_percents)
    
    # 4. Calculate PSI
    # PSI = sum( (Actual% - Expected%) * ln(Actual% / Expected%) )
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    total_psi = np.sum(psi_values)
    
    return float(total_psi)

def evaluate_drift(training_df: pd.DataFrame, live_df: pd.DataFrame, features: list):
    """Evaluates multiple features and triggers alerts."""
    logger.info(f"Evaluating {len(features)} features for drift...")
    
    drift_report = {}
    
    for feature in features:
        train_data = training_df[feature].values
        live_data = live_df[feature].values
        
        psi_score = calculate_psi(train_data, live_data)
        drift_report[feature] = psi_score
        
        if psi_score > 0.20:
            logger.error(f"🚨 SEVERE DRIFT DETECTED in '{feature}'. PSI: {psi_score:.3f}")
        elif psi_score > 0.10:
            logger.warning(f"⚠️ Moderate drift in '{feature}'. PSI: {psi_score:.3f}")
        else:
            logger.info(f"✅ '{feature}' is stable. PSI: {psi_score:.3f}")
            
    return drift_report

if __name__ == "__main__":
    # Mock Data
    np.random.seed(42)
    
    # Training data: Kills per match (Normal distribution centered at 5)
    training = pd.DataFrame({
        "kills": np.random.normal(5.0, 2.0, 10000),
        "playtime": np.random.normal(60.0, 15.0, 10000)
    })
    
    # Live data: Game patch made weapons stronger, kills shifted up to 8. Playtime stayed same.
    live = pd.DataFrame({
        "kills": np.random.normal(8.0, 2.0, 10000),
        "playtime": np.random.normal(60.5, 15.0, 10000)
    })
    
    evaluate_drift(training, live, ["kills", "playtime"])
    # Expected output: Kills will show severe drift. Playtime will be stable.
```

---

## Part 8 — Deployment

### Apache Airflow
- DAG runs daily at 2:00 AM.
- **Task 1:** Query Snowflake to extract the last 7 days of feature logs.
- **Task 2:** Run the Python PSI/KS calculation script (or push computation to PySpark if data is terabytes).
- **Task 3:** Write results to a Postgres metric database.
- **Task 4:** If PSI > 0.20, trigger Slack webhook with a link to a Grafana dashboard visualizing the distribution shift.

---

## Part 9 — Unit Testing

```python
import numpy as np
from drift_monitor import calculate_psi

def test_psi_identical_distributions():
    # If distributions are identical, PSI should be near 0
    data = np.random.normal(0, 1, 1000)
    psi = calculate_psi(data, data)
    assert psi < 0.01

def test_psi_shifted_distributions():
    # If distributions are completely separate, PSI should be high (>0.2)
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(5, 1, 1000) # Shifted by 5 std devs
    
    psi = calculate_psi(expected, actual)
    assert psi > 1.0 # Massive drift
```

---

## Part 10 — Integration Testing

- **Shadow Mode Retraining:**
  - Inject a known data shift (e.g., multiply a feature by 10) into the staging environment.
  - Assert that the Airflow DAG correctly calculates the PSI > 0.20.
  - Assert that the system automatically initiates an offline model retraining pipeline in Vertex AI / SageMaker to see if the new model performs better on the shifted data.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **10,000 Features** | Pulling raw data for 10,000 features into Python memory will cause OOM. We must compute the histograms (the 10 decile buckets) directly inside Snowflake using SQL `WIDTH_BUCKET` functions. Python only downloads the 10 bin counts (a tiny payload) to calculate the final PSI math. |
| **Multivariate Drift** | PSI is univariate (looks at 1 feature at a time). What if the relationship *between* two features changes, but their individual distributions stay the same? We must train a "Domain Classifier" (a Random Forest trying to predict if a row is from Training or Live data). If the RF achieves high accuracy, multivariate drift has occurred. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| PSI vs Kolmogorov-Smirnov (KS-Test) | KS-Test is highly sensitive and gives a strict p-value. However, at massive scale (millions of rows), KS-Test *always* returns p < 0.05 even for microscopic, irrelevant shifts due to statistical power. PSI is much more stable for large datasets and easier to interpret in business terms. |
| 1-Day vs 7-Day Window | A 1-day live window catches acute pipeline bugs immediately (e.g., a feature suddenly outputs all nulls). A 7-day window catches slow, organic gameplay drift and ignores weekend/weekday seasonality. Both should be implemented in parallel. |

---

## Part 13 — Alternative Approaches

1. **Evidently AI / Great Expectations:** Instead of writing custom PSI functions, use an open-source library like Evidently. It automatically generates HTML reports visualizing the drift for all features.
2. **Autoencoder Reconstruction:** Train an Autoencoder on the training features. In production, pass the live features through the Autoencoder. If the reconstruction error (MSE) trends upwards over the month, the overall feature space is drifting.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Seasonality Alerts | Model predicts in-game store purchases. Purchasing always drops on Tuesdays. PSI triggers an alert every Tuesday. | Use the 7-day rolling window. Alternatively, compare "This Tuesday" vs "Last Tuesday" instead of comparing vs the static Training dataset. |
| Feature Importance Ignorance | PSI triggers a SEVERE alert for `feature_xyz`. But `feature_xyz` is useless and has a 0.001 weight in the XGBoost model. Data scientists waste time investigating. | Weight the alerts. Define `Alert Severity = PSI_Score * Feature_Importance`. Only page humans if the drifted feature is actually driving the model's predictions. |

---

## Part 15 — Debugging

**Symptom:** A model predicting Player Churn has severe feature drift (PSI > 0.5) on almost every feature. The Data Science team retrains the model on the new data, but the new model's accuracy is terrible.

**Debugging steps:**
1. Why is the new data bad? Check the Data Warehouse ingestion pipeline.
2. The drift was not "organic gameplay shift" (Concept Drift). It was a **Data Quality Bug**.
3. A backend engineer changed the telemetry JSON schema, renaming `matches_played` to `match_count`. The ML pipeline filled `matches_played` with NULLs.
4. **Fix:** Do not auto-retrain on drifted data. Always run a Data Quality check (Null rates, type checks) *before* the Drift check. Auto-retraining on broken data corrupts the model.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `data_quality_null_rate` | > 1% change → Data engineering pipeline is broken. |
| `weighted_portfolio_drift` | A macro score combining PSI for all models. If it spikes, a global game patch likely broke ML assumptions. |

---

## Part 17 — Production Improvements

1. **Concept Drift Tracking:** If ground-truth labels *are* eventually available (e.g., 30 days later we know who churned), automatically calculate the accuracy (ROC-AUC) of the 30-day old predictions. If ROC-AUC drops, but feature PSI is stable, the relationship between features and target has changed (Concept Drift), which necessitates retraining with new feature engineering.
2. **Target Drift:** Track the distribution of the model's *predictions*. If the model historically predicted 5% fraud, and today it predicts 20% fraud, trigger an immediate alert. (This requires no ground truth and catches issues instantly).

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"PSI is great for numerical data. But how do you calculate drift on a high-cardinality categorical feature, like `item_name` which has 10,000 distinct strings?"**
2. **"If we have 5000 features, and we use a 95% confidence interval for a statistical test, we will get 250 False Positives purely by chance every single day. How do you stop this alert spam?"**
3. **"We use SHAP values for model explainability. How can we use SHAP to build a better drift detector than standard PSI?"**

---

## Part 19 — Ideal Answers

**Q1 (High Cardinality Categories):**
> "PSI fails on high cardinality because buckets become empty. We must reduce dimensionality first. We can either group the long-tail items into an 'Other' bucket before calculating PSI, or we can use embeddings (convert `item_name` to a dense vector) and track the drift of the vector's centroid distance over time."

**Q2 (False Positives / Multiple Comparisons):**
> "This is the Multiple Comparisons Problem. First, we use the Bonferroni correction (divide the p-value threshold by 5000). Second, as mentioned earlier, we multiply the drift score by the Feature Importance score. We only trigger PagerDuty if it's a Top 20 feature. The other 4,980 features just get logged silently to a dashboard."

**Q3 (SHAP Drift):**
> "Instead of tracking the drift of the raw input features (which might not actually impact the model), we track the drift of the *SHAP values*. If the distribution of a feature changes, but its SHAP distribution remains identical, the model's logic is unaffected, and we shouldn't alert. This isolates 'Harmful Drift' from 'Benign Drift'."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Recommends PSI over KS-Test for large-scale data stability.
- Solves the False Positive / Alert Spam problem by tying alerts to Feature Importance.
- Understands how to calculate metrics in the DB (Snowflake histograms) rather than OOMing Python.
- Accurately answers the SHAP vs Feature Drift question.

### Hire
- Sets up a logical Airflow batch pipeline.
- Writes a mathematical implementation of PSI or KS-Test.
- Recognizes the difference between Data Quality bugs and genuine Data Drift.

### Lean Hire
- Suggests tracking model accuracy (ROC-AUC) in real-time, failing to realize that ground-truth labels for things like Churn are delayed by 30 days.
- Over-relies on standard statistical tests (p-values) without understanding the impact of massive sample sizes forcing p < 0.05 constantly.

### Lean No Hire
- Proposes retraining the model every single night blindly to "solve" drift, wasting massive compute resources and risking model collapse.

### No Hire
- Does not know what Covariate Shift or Data Drift is.
- Cannot explain how to compare two distributions.
