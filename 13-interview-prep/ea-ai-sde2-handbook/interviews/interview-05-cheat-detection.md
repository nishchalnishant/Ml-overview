# Interview 05 — Real-Time Cheat Detection System
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the EA Anti-Cheat team working on a first-person shooter (FPS). The current anti-cheat system relies on client-side memory scanning (like Easy Anti-Cheat), but cheat developers keep bypassing it by running cheats on external hardware. 

Your task is to **design a server-side ML system that detects "Aimbots" in real-time purely from player telemetry data (mouse movements, crosshair placement, hit rates).**

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Data frequency (How often is telemetry sent? e.g., 10Hz, 60Hz?)
- Action taken upon detection (Auto-ban? Flag for manual review? Shadow-ban?)
- Latency (Does it need to ban mid-match, or after the match ends?)
- Class imbalance (What % of players are cheating?)
- Feature set available (Do we have XYZ coordinates? Crosshair angles? Click timestamps?)
- Model transparency (Do we need to explain *why* the model flagged them to customer support?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"What is the frequency of the telemetry data coming from the game server?"**
   → *Answer: The server ticks at 60Hz. We capture player pitch, yaw, and fire events every tick (16ms).*

2. **"Does this system need to ban mid-match, or can we process data post-match?"**
   → *Answer: We want to flag them mid-match, ideally within 2 minutes of the cheating behavior starting, so we can kick them before they ruin the whole game.*

3. **"Do we ban them immediately via ML, or flag for human review?"**
   → *Answer: If confidence is > 99.9%, auto-kick. Otherwise, flag for manual review.*

4. **"What does the training data look like? How do we have ground truth for cheating?"**
   → *Answer: We have a dataset of 10,000 matches manually verified by admins (5,000 normal players, 5,000 confirmed aimbotters).*

5. **"Do we need interpretability for customer support appeals?"**
   → *Answer: Yes. If a user appeals a ban, we need to know roughly what triggered it (e.g., "inhuman snap speed").*

---

## Part 4 — Expected Assumptions

- **Data volume is massive:** 60Hz data for 64 players in a match = 3,840 events per second per server. 
- **Time-Series approach:** Aimbots exhibit unnatural high-frequency patterns (perfect snapping, zero over-correction). This is a time-series classification problem.
- **Architecture:** We cannot process 60Hz data directly through a heavy deep learning model on every tick. We must window the data (e.g., aggregate a 5-second window of combat).

---

## Part 5 — High-Level Solution

```
  Game Server (60Hz Ticks)
       │ (UDP/Protobuf)
       ▼
  Ingestion Layer (Kafka)
       │
       ▼
  Stream Processing (Apache Flink / Spark Structured Streaming)
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Filter: Only keep data when player is firing/aiming  │
  │ 2. Windowing: Extract 5-second combat engagement clips  │
  │ 3. Feature Extraction: Delta angles, snap speed, jitter │
  └─────────────────────────────────────────────────────────┘
       │
       ▼
  Inference Service (FastAPI / gRPC)
  ┌─────────────────────────────────────────────────────────┐
  │ 4. XGBoost / Random Forest / 1D-CNN Model               │
  │ 5. Threshold logic (>0.999 = Ban, >0.90 = Review)       │
  └─────────────────────────────────────────────────────────┘
       │
       ▼
  Action Service (Kicks player via Game Server API)
```

**Core ML Component:** 
A machine learning model (XGBoost on extracted statistical features, or a 1D-CNN/RNN on raw sequences) that evaluates a 5-second window of mouse movement/aim data.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Data Windowing
- We don't care about the player running across the map. We only care about the moment they aim and shoot.
- Define an "Engagement Window": 1 second before first shot, 4 seconds after.

### Step 2: Feature Engineering (Time Series to Tabular)
If using XGBoost (preferred for interpretability and speed):
- **Jitter:** Variance of delta-pitch and delta-yaw. Aimbots often have zero jitter when locked on, or robotic micro-jitters.
- **Snap Speed:** Max degrees turned per millisecond.
- **Overshoot:** Does the crosshair pass the target and correct back? (Humans always overshoot; aimbots rarely do).
- **Time to Target:** How fast from target appearing on screen to crosshair locking on.

### Step 3: Model Architecture
- Train an XGBoost Classifier on these extracted features.
- Why not Deep Learning (LSTM/Transformer)? DL is excellent here but harder to interpret for customer support, and computationally heavier for real-time inference across millions of players. We will use XGBoost for v1.

### Step 4: Stream Processing API
- A service that accepts a chunk of gameplay telemetry, extracts features, and runs inference.

---

## Part 7 — Complete Python Code

```python
"""
aimbot_detector.py - Real-time cheat inference service
"""
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Anti-Cheat ML Service")

# Load model
MODEL_PATH = "models/aimbot_xgb.json"
model = xgb.Booster()
model.load_model(MODEL_PATH)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
class Tick(BaseModel):
    timestamp_ms: int
    pitch: float
    yaw: float
    is_firing: bool

class EngagementPayload(BaseModel):
    match_id: str
    player_id: str
    ticks: List[Tick] # Expected ~300 ticks (5 seconds @ 60Hz)

# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------
def extract_features(ticks: List[Tick]) -> dict:
    """Converts raw 60Hz ticks into statistical features for XGBoost."""
    if len(ticks) < 10:
        return {} # Not enough data
        
    df = pd.DataFrame([t.dict() for t in ticks])
    
    # Calculate angular deltas (frame-to-frame movement)
    df['delta_pitch'] = df['pitch'].diff().fillna(0)
    df['delta_yaw'] = df['yaw'].diff().fillna(0)
    df['angular_speed'] = np.sqrt(df['delta_pitch']**2 + df['delta_yaw']**2)
    
    # Filter only when firing
    firing_df = df[df['is_firing'] == True]
    
    features = {
        # Max snap speed (humans cap out around a certain physical limit)
        "max_snap_speed": df['angular_speed'].max(),
        
        # Variance of movement while firing (aimbots have robotic 0 variance)
        "var_pitch_firing": firing_df['delta_pitch'].var() if not firing_df.empty else 0,
        "var_yaw_firing": firing_df['delta_yaw'].var() if not firing_df.empty else 0,
        
        # Micro-correction count (changes in direction)
        "yaw_direction_changes": (df['delta_yaw'] * df['delta_yaw'].shift(1) < 0).sum(),
        
        # Ratio of time spent firing vs moving
        "firing_ratio": len(firing_df) / len(df)
    }
    return features

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
def execute_ban(player_id: str, match_id: str, reason: str):
    """Call internal Server API to kick player."""
    logger.warning(f"KICK EXECUTED: Player {player_id} in Match {match_id} | {reason}")
    # requests.post(f"http://game-orchestrator/kick/{player_id}")

def flag_for_review(player_id: str, prob: float):
    """Write to DB for human moderator review."""
    logger.info(f"FLAGGED: Player {player_id} | Prob: {prob:.4f}")

@app.post("/v1/analyze_engagement")
async def analyze_engagement(payload: EngagementPayload, bg_tasks: BackgroundTasks):
    # 1. Extract Features
    features = extract_features(payload.ticks)
    if not features:
        return {"status": "skipped", "reason": "insufficient_data"}
        
    # 2. Format for XGBoost
    feature_names = ["max_snap_speed", "var_pitch_firing", "var_yaw_firing", "yaw_direction_changes", "firing_ratio"]
    feature_vector = np.array([[features[k] for k in feature_names]])
    dmatrix = xgb.DMatrix(feature_vector, feature_names=feature_names)
    
    # 3. Inference
    prob = float(model.predict(dmatrix)[0])
    
    # 4. Action Logic
    if prob >= 0.999:
        # Auto-ban logic
        bg_tasks.add_task(
            execute_ban, 
            payload.player_id, 
            payload.match_id, 
            f"Aimbot detected (conf: {prob:.4f})"
        )
        action = "banned"
    elif prob >= 0.90:
        # Send to manual review queue
        bg_tasks.add_task(flag_for_review, payload.player_id, prob)
        action = "flagged"
    else:
        action = "clean"
        
    return {
        "player_id": payload.player_id,
        "probability": prob,
        "action": action
    }
```

---

## Part 8 — Deployment

### Stream Processing Architecture
- The FastAPI code above assumes the client (or Flink job) groups the ticks and sends HTTP requests.
- **In reality:** Game servers dump UDP streams to Kafka.
- **Apache Flink** consumes Kafka, maintains a sliding window of state, detects a "combat engagement", triggers the feature extraction, and calls a deployed gRPC/FastAPI model server.

### Kubernetes
- Standard CPU deployments for XGBoost. Memory requirements are very low.
- HPA (Horizontal Pod Autoscaler) based on CPU usage.

---

## Part 9 — Unit Testing

```python
import pytest
from aimbot_detector import extract_features, Tick

def test_extract_features_robotic_aim():
    # Simulate aimbot: Perfect snap, 0 variance while firing
    ticks = []
    # 5 frames of snap
    for i in range(5):
        ticks.append(Tick(timestamp_ms=i*16, pitch=i*10, yaw=i*10, is_firing=False))
    # 10 frames locked on perfectly (no movement while firing)
    for i in range(5, 15):
        ticks.append(Tick(timestamp_ms=i*16, pitch=50, yaw=50, is_firing=True))
        
    features = extract_features(ticks)
    
    # Assert robotic behavior
    assert features["var_pitch_firing"] == 0.0
    assert features["var_yaw_firing"] == 0.0
    assert features["max_snap_speed"] > 10.0
```

---

## Part 10 — Integration Testing

- Replay a historical match log (JSON stream) containing a known aimbotter.
- Feed it through the Kafka -> Flink -> FastAPI pipeline in a local docker-compose environment.
- Assert that the `execute_ban` function is called for the correct `player_id`.
- Assert that legitimate players in the same match log are NOT banned.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Ingestion Volume** | Millions of concurrent players at 60Hz is terabytes of data per hour. We CANNOT process everything. Game servers must pre-filter: only send telemetry to Kafka *when a player fires a weapon or deals damage*. This reduces volume by 90%. |
| **Stateful Windows** | Flink needs to hold player state in memory (RocksDB) to build the 5-second windows. Partition the Flink cluster by `match_id` to ensure all data for one match hits the same node. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Server-side vs Client-side ML | Client-side is cheap (uses player's CPU) but easily manipulated/bypassed. Server-side is 100% secure from tampering but costs millions in cloud compute. |
| XGBoost vs 1D-CNN/RNN | Hand-crafted features (XGB) might miss complex patterns, but are fast and explainable. DL (RNN) learns raw sequences but is a black box, making ban appeals impossible to verify manually. |
| Strict Ban Threshold (0.999) | High false-negative rate (smart cheats avoid detection), but zero false positives. Banning an innocent pro-player is a massive PR disaster. |

---

## Part 13 — Alternative Approaches

1. **Unsupervised Learning (Isolation Forest/Autoencoders):** Instead of training on known cheats (which go out of date fast), train an autoencoder on *normal* player data. If a player's movement causes a massive reconstruction error, they are playing abnormally. Excellent for zero-day cheat detection.
2. **Graph Neural Networks:** Map the relationships between players (who looks at who through walls) to detect ESP/Wallhacks, rather than just aimbots.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Hardware Lag | Server ticks drop to 20Hz | The model relies on 60Hz data. If tickrate fluctuates, `angular_speed` features will spike unnaturally. The system must ignore data from matches experiencing server degradation. |
| False Positives from Pros | A professional player is banned | Implement a whitelist for known verified pro accounts. Maintain a secondary "Review" queue for high-ELO players rather than auto-banning them. |
| Model Drift | Cheat devs add "humanization" (fake jitter) | Continuously retrain the model. Adversarial setup: use banned players' new data to retrain daily. |

---

## Part 15 — Debugging

**Symptom:** A prominent Twitch streamer is auto-banned live on stream, causing a massive community uproar.

**Debugging steps:**
1. Check the logs: What was the confidence score? (e.g., 0.9995).
2. Look at the SHAP values (feature importance) for that specific inference. Why did the model flag them?
3. Discover that the streamer uses a rare ultra-high DPI mouse setting, resulting in `max_snap_speed` values that look identical to a basic aimbot.
4. Immediate remediation: Unban, add streamer to whitelist.
5. Long-term fix: Collect data from ultra-high DPI legitimate players and inject it into the negative training set.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `ml_auto_bans_per_hour` | > 500% increase → Critical (Model broken) |
| `flink_window_dropped_events` | > 1% → Warning (Processing lagging) |
| `api_inference_latency_ms` | > 50ms → Scale out inference pods |

---

## Part 17 — Production Improvements

1. **SHAP Explanations:** Run SHAP on the XGBoost output and save the top 3 contributing features to the database. When CS agents review the ban, the UI says: "Flagged due to: 1. Unnatural snap speed. 2. Zero recoil variance."
2. **Ensemble Models:** Run the fast XGBoost model. If score > 0.80, route the raw 60Hz sequence to a heavier PyTorch LSTM model for a second opinion before banning.
3. **Hardware Fingerprinting:** Link ML detections to hardware IDs. If an ML ban triggers, ban the hardware to prevent them from making a new free account.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"Cheat developers are smart. They start adding artificial 'jitter' to their aimbots to mimic human variance, breaking your `var_pitch_firing` feature. How do you adapt?"**
2. **"If a player is playing on a controller with Aim Assist, their crosshair behavior looks very different from a mouse. Won't your model flag controller players as aimbotters?"**
3. **"We have 10 million daily players. Running Flink windowing on all their data is costing $50,000 a month in AWS bills. How can we cut the data volume by 99% without losing anti-cheat coverage?"**
4. **"How do you handle 'Wallhacks' (players who can see through walls) with this system? They don't necessarily snap to targets."**

---

## Part 19 — Ideal Answers

**Q1 (Humanized Cheats):**
> "Hand-crafted statistical features fail against humanization. We need to move from XGBoost to Deep Learning on the raw sequences. An LSTM or a 1D-CNN can detect the synthetic nature of the noise (e.g., the frequency spectrum of RNG jitter is different from human muscle micro-tremors). We can also use Frequency Domain features (FFT) in XGBoost to detect unnatural high-frequency noise injection."

**Q2 (Controller vs Mouse):**
> "Yes, this is a massive issue. Aim assist behaves like a weak, smooth aimbot. We must train *separate* models for Mouse+Keyboard and Controller. The telemetry payload must include the `input_device_type`. If we mix them, the model will just learn to ban controller players."

**Q3 (Cost reduction):**
> "We implement a funnel. Layer 1: Heuristics on the game server (e.g., K/D ratio > 10, or headshot % > 80%). Only players who trigger Layer 1 are forwarded to the Kafka/Flink ML pipeline. Furthermore, we only sample 5% of players randomly, plus 100% of reported players. This slashes data volume."

**Q4 (Wallhacks):**
> "Wallhacks require a different feature set. We need spatial features: 'Time spent looking at enemy through opaque geometry', or 'Pre-fire timing' (firing before the enemy rounds the corner). The current windowing logic (engagement only) misses this. We'd need to evaluate movement and crosshair placement *before* engagements."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Understands the extreme data volume constraints and designs a streaming architecture (Kafka/Flink).
- Articulates why XGBoost is preferred for explainability, but understands how DL would solve humanized cheats.
- Recognizes the Controller vs. Mouse aim-assist problem without prompting.
- Provides excellent, game-specific feature engineering (jitter, snap speed, overshoot).

### Hire
- Good system design, uses standard ML serving techniques.
- Identifies the need for windowing the data.
- Handles the ban threshold conservatively to avoid false positives.
- Feature engineering is solid.

### Lean Hire
- Uses standard tabular data approaches but struggles with the time-series/streaming aspect.
- Might suggest a heavy Neural Network that would be too expensive to run at scale.
- Needs prompting to figure out how to reduce data ingestion costs.

### Lean No Hire
- Suggests client-side ML implementation without understanding security risks.
- Cannot articulate how to extract features from a sequence of pitch/yaw coordinates.
- Suggests banning players at 0.70 probability.

### No Hire
- Fails to structure the data pipeline.
- Doesn't understand the difference between batch and streaming.
- Code does not resemble a working inference service.
