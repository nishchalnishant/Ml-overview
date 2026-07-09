# Interview 25 — Anti-Cheat: Client-side Memory Anomaly Detection
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the EA Anti-Cheat (EAAC) team. Cheat developers are bypassing server-side detection by using subtle "Aimbots" and "Wallhacks" that look like human behavior to the server.

To combat this, EAAC runs a kernel-level driver on the player's PC. Your task is to **design a machine learning system that runs locally on the game client to detect anomalies in memory access patterns, input hardware behavior, and OS-level events to flag cheating software.**

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Compute Budget (Running heavy ML models locally will steal CPU cycles from the game).
- Feature Engineering (What data is the anti-cheat driver actually collecting?)
- False Positives (Banning an innocent player is catastrophic).
- Evasion (If the model is on the client, cheat developers will try to reverse engineer it).
- Remediation (Do we ban instantly, or shadow-ban, or flag for review?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"What is the compute budget for this model?"**
   → *Answer: Very tight. We cannot use more than 1% of CPU or 50MB of RAM. The game (e.g., Battlefield) needs those resources.*

2. **"What specific features are we collecting from the OS?"**
   → *Answer: Mouse movement trajectories (X,Y, Time), frequency of `ReadProcessMemory` calls from other apps, and active driver signatures.*

3. **"Do we ban players immediately based on the client model?"**
   → *Answer: No. The client model should act as a 'trigger'. If it flags a user, we upload a high-resolution telemetry package to the server for a heavier cloud-side ML model to review.*

---

## Part 4 — Expected Assumptions

- **Architecture:** A lightweight Edge ML model (e.g., XGBoost / Random Forest or a tiny 1D-CNN) running via ONNX Runtime or native C++ in the background.
- **Features:** Mouse kinematics (jerk, acceleration) and OS memory access anomalies.
- **Security:** Model weights must be encrypted and obfuscated.

---

## Part 5 — High-Level Solution

```
  [Player PC / Game Client]
       │
  [EAAC Kernel Driver]
  Collects: Mouse (X,Y,dt) + Unknown Process Memory Reads
       │
       ▼
  [Local Feature Extraction (C++)]
  Calculates: Mouse Jerk, Snap-to-target speed, Process scan frequency.
       │
       ▼
  [Local ML Inference (LightGBM / XGBoost C++ API)]
  Evaluates features every 10 seconds. CPU cost < 0.1%.
       │
       ▼ (If Anomaly Score > 0.90)
  [Telemetry Trigger] ➔ Encrypts recent data buffer ➔ Sends to Cloud
  
       =========================================================

  [EA Cloud Anti-Cheat Servers]
  Heavy Neural Network evaluates the uploaded buffer.
  If confirmed ➔ Apply Ban (Delayed).
```

**Core ML Component:** Designing a robust feature extraction pipeline for mouse kinematics (Aimbots) and choosing a highly efficient, CPU-bound classification model.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Feature Engineering (Mouse Kinematics)
- Humans move mice in curves with acceleration and deceleration (Fitts's Law).
- Aimbots move mice in perfectly straight lines (shortest path) with unnatural instantaneous acceleration.
- **Features:** Angle of curvature, maximum acceleration, jerk (derivative of acceleration), and time-to-target.

### Step 2: Model Selection
- We cannot use a deep CNN or Transformer on the client (too heavy, requires GPU).
- We use **LightGBM** or a shallow **Random Forest**. Trees evaluate in microseconds and consume almost zero RAM.

### Step 3: Trigger Mechanism
- The model outputs a probability of cheating `P(cheat)`.
- We use a sliding window (e.g., last 10 seconds of gameplay).
- If `P(cheat) > 0.95`, we don't ban. We trigger a "Dump". We package the last 30 seconds of raw input logs and OS telemetry, encrypt it, and send it to the server backend for batch processing.

---

## Part 7 — Complete Python Code

*Note: The production system is C++, but we will write the Python equivalent for feature engineering and training the LightGBM model.*

```python
"""
anticheat_client_model.py - Train LightGBM on Mouse Kinematics
"""
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Feature Engineering
# ---------------------------------------------------------------------------
def extract_kinematics(mouse_data: pd.DataFrame) -> pd.DataFrame:
    """
    mouse_data contains: [timestamp, x, y]
    Extracts physics features to detect robotic aimbots.
    """
    df = mouse_data.copy()
    
    # Calculate delta time and delta distance
    df['dt'] = df['timestamp'].diff().fillna(1.0)
    df['dx'] = df['x'].diff().fillna(0.0)
    df['dy'] = df['y'].diff().fillna(0.0)
    
    # Velocity
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['velocity'] = df['distance'] / df['dt']
    
    # Acceleration
    df['acceleration'] = df['velocity'].diff().fillna(0.0) / df['dt']
    
    # Jerk (Change in acceleration - Robots have massive jerk, humans are smooth)
    df['jerk'] = df['acceleration'].diff().fillna(0.0) / df['dt']
    
    # Angle (Straight lines vs curves)
    df['angle'] = np.arctan2(df['dy'], df['dx'])
    df['angle_change'] = df['angle'].diff().fillna(0.0)
    
    # Aggregate into a single feature vector for this 10-second window
    features = pd.DataFrame([{
        'max_jerk': df['jerk'].max(),
        'mean_jerk': df['jerk'].abs().mean(),
        'zero_angle_variance_pct': (df['angle_change'] == 0).mean(), # Robots move in straight lines
        'max_acceleration': df['acceleration'].max(),
    }])
    
    return features

# ---------------------------------------------------------------------------
# 2. Training Pipeline (Offline)
# ---------------------------------------------------------------------------
def train_model(train_df: pd.DataFrame, labels: pd.Series):
    logger.info("Training LightGBM Anti-Cheat Model...")
    
    # LightGBM is chosen because it compiles to C++ extremely well
    # and has a tiny memory footprint.
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        objective='binary',
        class_weight='balanced' # Aimbots are rare
    )
    
    model.fit(train_df, labels)
    
    # Export for C++ integration (LightGBM can save as a raw text string of trees)
    model.booster_.save_model("aimbot_detector.txt")
    logger.info("Model saved to aimbot_detector.txt")
    return model

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Mock data: Human (Smooth)
    human_data = pd.DataFrame({
        'timestamp': np.arange(0, 100, 16), # 60hz (16ms)
        'x': np.sin(np.linspace(0, 3, 7)) * 100, # Curve
        'y': np.cos(np.linspace(0, 3, 7)) * 100
    })
    
    # Mock data: Aimbot (Snaps instantly in a straight line)
    bot_data = pd.DataFrame({
        'timestamp': np.arange(0, 100, 16),
        'x': np.linspace(0, 500, 7), # Perfect straight line
        'y': np.linspace(0, 500, 7)
    })
    
    human_feat = extract_kinematics(human_data)
    bot_feat = extract_kinematics(bot_data)
    
    print("Human Features:\n", human_feat.iloc[0])
    print("Bot Features:\n", bot_feat.iloc[0])
    
    # Notice: Bot 'zero_angle_variance_pct' will be 1.0 (perfectly straight)
```

---

## Part 8 — Deployment

### Native C++ Integration
- Python is not used in the client. The LightGBM model is exported as `aimbot_detector.txt`.
- The game client uses the `LightGBM C API` to load the text file and run inference.
- Overhead is ~1MB of RAM and < 0.1ms of CPU time per evaluation.

### Model Encryption
- Cheat developers will look for `aimbot_detector.txt` in the game files to reverse engineer the threshold values.
- We must encrypt the model file via AES-256. The decryption key is generated dynamically by the server and sent to the client via a secure handshake at runtime.

---

## Part 9 — Unit Testing

```python
import numpy as np
import pandas as pd
from anticheat_client_model import extract_kinematics

def test_aimbot_angle_variance():
    # A robot moving perfectly diagonally
    bot_data = pd.DataFrame({
        'timestamp': [0, 10, 20, 30],
        'x': [0, 10, 20, 30],
        'y': [0, 10, 20, 30]
    })
    
    features = extract_kinematics(bot_data)
    
    # Angle change should be 0 for all steps (straight line)
    # The percentage of zero changes should be very high
    assert features.iloc[0]['zero_angle_variance_pct'] > 0.60
```

---

## Part 10 — Integration Testing

- **Hardware Replay:**
  - Build a test suite containing 100 hours of legitimate pro-player gameplay (pros have fast reflexes that look like aimbots) and 10 hours of known aimbot gameplay.
  - Run the C++ extraction and LightGBM model over the dataset.
  - Assert that the **False Positive Rate (FPR) is 0%** on the pro-player data. If a pro gets flagged, the feature thresholds are too aggressive.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Data Collection Volume** | We cannot stream 60hz mouse coordinates for 10 million players to the cloud (Petabytes of data, massive AWS bills). This is why Edge ML is strictly required. The client only uploads the ~5MB telemetry package *if* the local model triggers an anomaly, reducing network traffic by 99.9%. |
| **Model Updates** | Cheat devs adapt daily. If we push a new LightGBM model, we don't want to release a 20GB game patch. We use the OTA (Over The Air) CDN to push a new 1MB `.txt` model file silently in the background when the user connects to the main menu. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Decision Trees vs Neural Networks | NNs (e.g., LSTMs for time-series) are more accurate at detecting nuanced behavior. However, they require ONNX Runtime, take ~30MB of RAM, and eat CPU cycles. LightGBM takes 1MB and is practically invisible to the OS. In games, frame-rate is king; we trade ML accuracy for zero performance impact. |
| Ban Delays (Ban Waves) vs Instant Bans | If the model detects a cheat and bans the user instantly, the cheat developer knows *exactly* which feature triggered the ban, making reverse engineering trivial. Ban waves (delaying the ban by 2 weeks) obfuscates the trigger, breaking the cheat dev's feedback loop. |

---

## Part 13 — Alternative Approaches

1. **Hardware ID (HWID) Fingerprinting:** Don't just look at mouse data. Use ML to fingerprint the user's specific hardware components (GPU serial, Motherboard UUID, MAC address variance). Cheat developers spoof these values. Train an anomaly detector to spot "impossible" hardware combinations (e.g., an RTX 4090 paired with an Intel Pentium 3 processor).
2. **Server-Side Only:** Never trust the client. Send sparse telemetry (1Hz) to the server. Use a massive Transformer model on the server to look for impossibly high accuracy ratings or wall-tracking behavior.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| High DPI / Sensitivity | A legitimate player buys a 16,000 DPI mouse. Their cursor twitches violently. The ML model flags it as "High Jerk Aimbot". | The feature extraction must normalize velocity/jerk by the player's in-game sensitivity settings and screen resolution to create a hardware-agnostic baseline. |
| Model Poisoning | Cheat developers realize the ML model is training on player data. They release a bot that plays terribly on purpose to poison the baseline training set. | Always train on a curated, closed dataset of known internal QA testers and vetted professional players. Never auto-train on public telemetry. |

---

## Part 15 — Debugging

**Symptom:** A new cheat called "Humanizer" is released. It adds Bezier-curve smoothing to the aimbot. Your LightGBM model detects 0% of them.

**Debugging steps:**
1. The cheat developer analyzed your features (Jerk, Angle Variance) and explicitly programmed the bot to mimic those distributions.
2. We must shift to a feature they cannot easily fake.
3. **Fix:** Look at OS-level heuristics rather than just mouse math. Add a feature: `time_since_last_unbacked_memory_allocation`. Cheats must inject code into the game's memory space. This OS-level signature cannot be "smoothed" via Bezier curves.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `client_trigger_rate` | > 5% → The Edge model is false-flagging. Stop uploading telemetry before it overloads the backend servers. |
| `ban_appeal_win_rate` | > 1% → Customer service is unbanning players because the ML model was wrong. Retrain immediately. |
| `client_cpu_overhead_ms` | > 1ms → The model is stuttering the game. |

---

## Part 17 — Production Improvements

1. **Ensemble Server Model:** The client model is just a tripwire. When the telemetry hits the server, pass it through an ensemble of 5 heavy models (LSTMs, CNNs on aim-paths, Graph networks on player social connections). Only apply the ban if 4/5 models agree with high confidence.
2. **Honeypots:** Render an invisible player (a bot) behind a wall in the game. Legitimate players cannot see it. Wallhacks will extract the coordinates from memory and snap their crosshairs to it through the wall. The ML model flags "Aiming at Honeypot" as an instant 100% confidence ban.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"A cheat developer finds your encrypted `aimbot_detector` file. They write a script that deletes the file every time the game launches, so the ML model never runs. How do you prevent this?"**
2. **"To avoid detection, a cheat uses a hardware 'DMA' (Direct Memory Access) PCIe card. It reads memory physically, bypassing the OS entirely. Your kernel driver cannot see any unauthorized processes. How does your ML pipeline catch this?"**
3. **"We decide we want to run a sequence model (LSTM) on the client instead of LightGBM. But LSTMs suffer from vanishing gradients and are slow. What alternative modern architecture would you use for Edge time-series, and why?"**

---

## Part 19 — Ideal Answers

**Q1 (Tamper Resistance):**
> "We implement a Heartbeat and Hash-Check. The server expects an encrypted heartbeat payload from the client-side ML model every 60 seconds, containing a hash of the model file. If the file is deleted, the heartbeat fails, and the server kicks the player from the match for 'Anti-Cheat Authentication Failure'."

**Q2 (Hardware DMA Cheats):**
> "If we cannot trust the OS (because DMA bypasses it), we must rely purely on Server-Side Behavioral Analysis. A DMA cheat still has to execute actions in the game world. We use a cloud-side ML model that analyzes the player's crosshair placement relative to enemy positions (e.g., tracking enemies perfectly through walls). No matter what hardware they use to read memory, their *behavior* is mathematically anomalous."

**Q3 (Edge Time-Series Architectures):**
> "Instead of LSTMs, I would use **Temporal Convolutional Networks (TCNs)**. TCNs use 1D dilated convolutions. They are vastly superior for Edge ML because convolutions can be processed in parallel (unlike LSTMs which are strictly sequential), they don't suffer from vanishing gradients, and they compile down to extremely efficient operations in ONNX/TFLite."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Anticipates the extreme CPU/RAM constraints of client-side game programming and avoids proposing heavy Deep Learning.
- Proposes valid physics features (Jerk, Acceleration, Fitts's Law) for Aimbot detection.
- Understands the security implications (obfuscation, encryption, heartbeat).
- Successfully answers the DMA hardware cheat question by pivoting to server-side behavior.

### Hire
- Sets up a logical Edge-to-Cloud architecture (Client triggers, Server verifies).
- Selects LightGBM/XGBoost for the edge model.
- Recognizes the need for ban waves to prevent reverse-engineering.

### Lean Hire
- Focuses entirely on server-side ML and ignores the prompt's requirement to design a client-side anomaly detector.
- Fails to understand why false positives (banning innocent players) are more dangerous than false negatives.

### Lean No Hire
- Proposes running a massive Transformer model on the user's PC to evaluate mouse movements.
- Cannot explain how to extract features from a time-series of X,Y coordinates.

### No Hire
- Does not understand basic kinematics (velocity/acceleration).
- Thinks ML can prevent people from modifying their own computer's memory.
