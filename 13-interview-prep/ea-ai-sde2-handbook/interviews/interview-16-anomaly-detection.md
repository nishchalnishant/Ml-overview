# Interview 16 — Real-Time Anomaly Detection on Streaming Data
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the Core Server Infrastructure team. EA operates thousands of game servers globally. We collect real-time server telemetry (CPU, Memory, Network Bandwidth, Active Players, Latency) at a 1-second resolution.

Your task is to **design an ML system that monitors this streaming telemetry and detects anomalous game servers in real-time.** An alert must be fired if a server begins acting erratically (e.g., memory leak, DDoS attack, hardware failure) so DevOps can reboot it before players notice.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Dimensionality (Are we monitoring 5 metrics or 500?)
- Definition of Anomaly (Univariate vs Multivariate? e.g., High CPU is normal if Players are high, but anomalous if Players are 0).
- Scale (How many servers are we tracking concurrently?)
- False Positive tolerance (If we alert DevOps 1000 times a day, they will ignore the system).
- Concept Drift (Game patches change baseline performance permanently).

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"Are we looking for univariate anomalies (one metric spiking) or multivariate (combinations of metrics)?"**
   → *Answer: Multivariate. High CPU is fine if the server is full. High CPU when the server is empty is an anomaly.*

2. **"How many metrics per server, and how many servers?"**
   → *Answer: 20 key metrics per server. 100,000 servers globally.*

3. **"What is the tolerance for False Positives?"**
   → *Answer: Extremely low. We prefer to miss minor anomalies (False Negatives) rather than spam PagerDuty with False Positives.*

4. **"Do we have labeled training data (known failures)?"**
   → *Answer: Very few. Most failures are unique zero-day events. We need an unsupervised approach.*

---

## Part 4 — Expected Assumptions

- **Architecture:** Stream Processing engine (Apache Flink / Spark Streaming) consuming from Kafka.
- **Algorithm:** Unsupervised Multivariate Anomaly Detection. Isolation Forest (fast, CPU-bound) or Autoencoders (deep learning).
- **Windowing:** Evaluating a single 1-second frame is too noisy. We must evaluate a rolling window (e.g., 1-minute averages).

---

## Part 5 — High-Level Solution

```
  Game Servers (100k nodes)
       │ (Telegraf / Prometheus Agent - 1Hz)
       ▼
  Kafka Topic (server_telemetry)
       │
       ▼
  Apache Flink (Streaming Aggregation)
  ┌────────────────────────────────────────────────────────┐
  │ Group by server_id. Calculate 1-minute rolling average │
  │ for all 20 metrics to smooth out micro-spikes.         │
  └────────────────────────────────────────────────────────┘
       │
       ▼
  Inference Service (Isolation Forest / Autoencoder)
  ┌────────────────────────────────────────────────────────┐
  │ Evaluate the 20-dimensional vector.                    │
  │ If Anomaly Score > 99.9th percentile ➔ Trigger Alert   │
  └────────────────────────────────────────────────────────┘
       │
       ▼
  PagerDuty / Auto-Remediation Service
```

**Core ML Component:** An unsupervised Isolation Forest model trained offline on "healthy" server data. It isolates anomalies based on path length in random trees. Handles multivariate relationships well.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Handling Multivariate Data
- A univariate threshold (e.g., `CPU > 90%`) fails.
- Isolation Forest takes the entire 20-d vector `[cpu, ram, bandwidth, players...]`. If `cpu` is 90 and `players` is 0, the model isolates this point quickly because it violates the normal correlation space.

### Step 2: Training Pipeline (Offline)
- Pull 7 days of historical telemetry from a period known to be healthy.
- Train the `IsolationForest` model.
- Crucial: Determine the threshold for alerting. Score a validation set, find the 99.9th percentile of anomaly scores, and hardcode that as the threshold to minimize False Positives.

### Step 3: Streaming Inference API
- Load the trained model into memory.
- Accept incoming 1-minute aggregated vectors from Flink.
- Output `True/False` for anomaly.

---

## Part 7 — Complete Python Code

```python
"""
server_anomaly_detector.py - Multivariate unsupervised anomaly detection
"""
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/iso_forest.pkl"
THRESHOLD_PATH = "models/threshold.npy"

# ---------------------------------------------------------------------------
# Offline Training
# ---------------------------------------------------------------------------
def train_model(historical_data: pd.DataFrame):
    """Trains the Isolation Forest on known healthy data."""
    logger.info("Training Isolation Forest...")
    
    # Drop IDs/Timestamps, keep only the 20 metric columns
    X = historical_data.drop(columns=['server_id', 'timestamp'])
    
    # contamination='auto' lets the algorithm decide, but we will manually
    # set an explicit scoring threshold later for strict control.
    model = IsolationForest(n_estimators=100, contamination=0.001, random_state=42, n_jobs=-1)
    model.fit(X)
    
    # Calculate scores on training data to establish a baseline threshold
    # Note: score_samples returns NEGATIVE anomaly scores. Lower = more anomalous.
    scores = model.score_samples(X)
    
    # Set threshold at the 99.9th percentile of anomalousness (0.1% lowest scores)
    alert_threshold = np.percentile(scores, 0.1)
    
    logger.info(f"Training complete. Strict Alert Threshold set to: {alert_threshold:.4f}")
    
    joblib.dump(model, MODEL_PATH)
    np.save(THRESHOLD_PATH, alert_threshold)

# ---------------------------------------------------------------------------
# Real-Time Inference Service (FastAPI / gRPC)
# ---------------------------------------------------------------------------
class StreamingAnomalyDetector:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.threshold = float(np.load(THRESHOLD_PATH))
        logger.info(f"Loaded model. Alert threshold: {self.threshold}")
        
    def detect(self, features: dict) -> dict:
        """
        Evaluates a single server's 1-minute aggregate vector.
        Features must be in the exact order as training.
        """
        # Convert to 2D array for sklearn
        feature_vector = np.array([list(features.values())])
        
        # Get raw anomaly score (negative)
        score = float(self.model.score_samples(feature_vector)[0])
        
        # If score is LOWER than our 0.1% threshold, it is highly anomalous
        is_anomaly = score < self.threshold
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": score,
            "threshold": self.threshold
        }

# Example execution
if __name__ == "__main__":
    # Mock data
    healthy_data = pd.DataFrame({
        'server_id': ['s1']*1000,
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='M'),
        'cpu_usage': np.random.normal(50, 10, 1000),
        'active_players': np.random.normal(64, 5, 1000),
        'memory_mb': np.random.normal(4000, 200, 1000)
    })
    
    train_model(healthy_data)
    
    detector = StreamingAnomalyDetector()
    
    # Test Normal Server (High CPU, High Players)
    normal_server = {'cpu_usage': 80.0, 'active_players': 64.0, 'memory_mb': 4200.0}
    print("Normal Server:", detector.detect(normal_server))
    
    # Test Anomalous Server (High CPU, 0 Players - e.g. crypto miner or infinite loop)
    anomalous_server = {'cpu_usage': 99.0, 'active_players': 0.0, 'memory_mb': 4000.0}
    print("Anomalous Server:", detector.detect(anomalous_server))
```

---

## Part 8 — Deployment

### Apache Flink
- Reads 100,000 JSON messages per second from Kafka.
- Applies a `TumblingEventTimeWindow` of 60 seconds grouped by `server_id`.
- Emits the 20-d averaged vector to the Inference Service.

### Inference Scaling
- The sklearn Isolation Forest `score_samples` method is CPU-bound and very fast (~1ms per vector).
- Wrap the `StreamingAnomalyDetector` in FastAPI. Deploy 20 replicas in Kubernetes behind a Load Balancer to handle the 1,666 QPS (100k servers / 60 seconds).

---

## Part 9 — Unit Testing

```python
import numpy as np
from server_anomaly_detector import StreamingAnomalyDetector

def test_anomaly_thresholding():
    # Mock the loaded model and threshold
    detector = StreamingAnomalyDetector()
    detector.threshold = -0.75
    
    class MockModel:
        def score_samples(self, X):
            # Return -0.80 for anomalous, -0.50 for normal
            if X[0][0] > 90 and X[0][1] == 0:
                return np.array([-0.80])
            return np.array([-0.50])
            
    detector.model = MockModel()
    
    # Anomalous: Score (-0.80) is less than threshold (-0.75)
    res_bad = detector.detect({'cpu': 99, 'players': 0})
    assert res_bad["is_anomaly"] == True
    
    # Normal: Score (-0.50) is greater than threshold (-0.75)
    res_good = detector.detect({'cpu': 50, 'players': 64})
    assert res_good["is_anomaly"] == False
```

---

## Part 10 — Integration Testing

- Spin up Kafka, Flink, and the FastAPI Inference service in `docker-compose`.
- Write a Python script that pumps 10 minutes of healthy telemetry to Kafka, followed by 1 minute where `server_42` experiences a simulated memory leak (RAM increases linearly by 100MB/sec).
- Assert that the PagerDuty mock endpoint receives exactly one alert for `server_42` at the 11-minute mark.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **100k to 1M Servers** | Sending HTTP requests from Flink to FastAPI for 1M servers creates network bottleneck overhead. Instead, embed the ONNX/Joblib model directly inside the Flink TaskManager as a UDF (User Defined Function). Data never leaves the Flink cluster. |
| **Concept Drift** | Game patches permanently alter performance. A patch that optimizes CPU usage shifts the distribution. The model will trigger false positives continuously. We must implement a rolling retraining architecture (e.g., retrain every 24 hours on the last 7 days of data). |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Isolation Forest vs Autoencoder | Autoencoders capture complex, non-linear deep correlations perfectly, but require GPUs to train and serve efficiently. Isolation Forests are fast, CPU-bound, and easy to interpret, making them vastly cheaper at EA's scale. |
| Alerting Threshold (Strictness) | A 99.9th percentile threshold guarantees very few False Positives, keeping DevOps happy. Tradeoff: It will completely miss "slow-burn" anomalies (False Negatives), like a slow memory leak, until it hits critical mass. |
| 1-Minute Window vs Real-Time | Averaging over 1 minute smooths out noise (e.g., garbage collection spikes). Tradeoff: If a server completely locks up, DevOps won't be alerted for up to 60 seconds. |

---

## Part 13 — Alternative Approaches

1. **Autoencoders (Deep Learning):** Train a Neural Network to reconstruct the 20-d vector. If a server acts weirdly, the reconstruction error (MSE) spikes. Excellent for detecting completely novel failure modes.
2. **State-Space Models (Kalman Filters):** For purely temporal tracking (e.g., detecting if memory usage is trending upwards abnormally over time), Kalman filters are extremely fast and update iteratively without needing large batch windows.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Global Outage | AWS goes down. All servers drop players to 0. CPU drops to 0. | The model flags 100,000 servers as anomalous simultaneously, DDoSing PagerDuty. Implement an **Alarm Suppressor**: If > 5% of the fleet is anomalous simultaneously, block individual server alerts and fire a single "Global Fleet Anomaly" alert. |
| Telemetry Delay | Kafka lags, sending 10 minutes of data at once | The Flink tumbling window gets confused if using processing-time. Must use Event-Time windowing with Watermarks to handle out-of-order/delayed data. |

---

## Part 15 — Debugging

**Symptom:** DevOps complains that the system is alerting for "Memory Anomalies", but when they check the server, RAM is at a totally safe 40%.

**Debugging steps:**
1. Check the 20-d vector that triggered the alert. 
2. RAM is 40%. What are the other 19 metrics? 
3. Discover that `Network_Out` is 0 kbps. The server is actually disconnected from the network, but the process is still running in memory. The multivariate model flagged the *combination* of (Normal RAM + Zero Network) as impossible.
4. **Fix:** The model is technically correct, but the alert message is confusing. Use SHAP values in real-time to extract the Top 3 features that contributed to the anomaly score, and include them in the Slack alert (e.g., `Anomaly detected. Key drivers: Network_Out, Active_Players`).

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `model_alerts_fired_per_hour` | > 50 → The model is miscalibrated or drifted. Halt auto-remediation. |
| `flink_window_watermark_lag` | > 2m → Pipeline is struggling to process real-time events. |
| `inference_latency_ms` | > 50ms → Model server is bottlenecked. |

---

## Part 17 — Production Improvements

1. **SHAP Explanations:** As mentioned in debugging, an alert without context is useless to an SRE. Run TreeSHAP on the anomalous vector and attach the feature contributions to the PagerDuty ticket.
2. **Automated Remediation:** If confidence is extremely high (score < 99.99th percentile) and the failure pattern matches a known historical signature, bypass PagerDuty and trigger a Kubernetes/AWS API call to gracefully reboot the game server node.
3. **Cluster-Aware Detection:** Sometimes a server is fine, but the *entire cluster* (e.g., all servers in `eu-west-1`) is behaving slightly differently than `us-east-1`. Add `region` as a categorical feature, or train regional models.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"A new game mode releases tomorrow that allows 128 players instead of 64. This will permanently alter CPU and RAM distributions. Your model will flag every single server as anomalous. How do you prevent this?"**
2. **"If a server experiences a slow memory leak (RAM goes up by 1% every hour), the 1-minute rolling average won't look anomalous compared to the previous minute. How does your system catch slow degradation?"**
3. **"Isolation Forest cannot be updated online; it must be retrained from scratch. If we have to retrain daily, how do we ensure we don't accidentally include anomalous data in the new training set (which would teach the model that anomalies are normal)?"**

---

## Part 19 — Ideal Answers

**Q1 (Scheduled structural breaks):**
> "For scheduled major updates, we must preemptively disable the model. We place it in 'Shadow Mode' where it calculates scores but doesn't alert. We wait 24 hours to collect the new baseline telemetry for the 128-player mode, retrain the model from scratch on the new data, verify the distributions, and then re-enable alerting."

**Q2 (Slow-burn anomalies):**
> "A 1-minute window only catches point anomalies (sudden spikes). To catch trend anomalies (memory leaks), we need a dual-window architecture. We run a secondary Flink job that calculates a 24-hour rolling slope (derivative) for memory. We feed that slope feature into the model. A steep continuous slope will be flagged as anomalous, even if absolute RAM is currently low."

**Q3 (Contaminated retraining data):**
> "We must sanitize the training data. We use the *previous day's* model to score the new 24 hours of data. Any data points flagged as anomalous by the old model (or correlated with known PagerDuty alerts/reboots) are stripped out of the dataset. We only train the new model on the remaining 'clean' data. This prevents the shifting baseline problem."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Understands multivariate vs univariate anomaly detection clearly.
- Proposes SHAP for alert explainability.
- Answers the slow memory leak question by introducing rate-of-change (derivative/slope) features.
- Grasps the danger of training on contaminated data and proposes a sanitization pipeline.

### Hire
- Selects a valid unsupervised algorithm (Isolation Forest / Autoencoder).
- Uses a stream processing framework (Flink/Spark) to window the data before inference.
- Handles the API and thresholding logic correctly.

### Lean Hire
- Suggests setting 20 static thresholds (e.g., `if CPU > 90 and RAM > 80`). Interviewer must push them toward ML.
- Doesn't think about the false positive rate (alert fatigue).

### Lean No Hire
- Suggests Supervised Learning (e.g., XGBoost predicting 1 or 0) for a problem that explicitly has no labeled failure data.
- Cannot explain how to process 100k servers of streaming data (tries to do it in a simple Python for-loop).

### No Hire
- Does not understand time-series data.
- Cannot explain what an anomaly is mathematically.
