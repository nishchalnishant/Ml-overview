# Interview 07 — Game Client Crash Prediction
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the Frostbite Engine telemetry team. Client crashes (game freezing or closing unexpectedly) are a massive source of player frustration. We collect high-frequency client telemetry (memory usage, CPU temp, frame drops, network jitter).

Your task is to **design an ML system that predicts if a game client is going to crash in the next 60 seconds.** If predicted, the game will silently auto-save the player's state or gracefully degrade graphics settings to prevent the crash.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Where does the model run? (Server-side or Client-side?)
- Client performance constraints (Can we run an ML model on a user's console without dropping their frame rate?)
- Imbalance (Crashes are extremely rare events, e.g., 0.01% of sessions).
- Features available (Are we getting stack traces, or just numerical hardware stats?)
- Action latency (How long does a graceful auto-save take?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"Should this model run on the server or on the player's client (PC/Console)?"**
   → *Answer: It must run on the client. Sending 1-second telemetry to the server and waiting for a response is too slow and costs too much bandwidth for millions of concurrent players.*

2. **"What is the performance budget for the client-side model?"**
   → *Answer: Extremely tight. Inference must take less than 1 millisecond and use less than 10MB of RAM. We cannot drop the game's framerate.*

3. **"What does the training data look like?"**
   → *Answer: We have historical telemetry (1Hz) uploaded after matches, tagged with `crashed = True/False` at the end of the time series.*

4. **"How rare are crashes?"**
   → *Answer: About 1 in 10,000 gaming sessions result in a crash.*

5. **"If we predict a crash, what action do we take?"**
   → *Answer: We trigger an emergency auto-save and flush the networking queue. This takes ~2 seconds.*

---

## Part 4 — Expected Assumptions

- **Client-Side Edge ML:** The model must be tiny. Deep learning (LSTMs) might be too heavy depending on the framework. A compressed LightGBM/XGBoost model, a linear model, or a highly pruned 1D-CNN using ONNX/TensorFlow Lite.
- **Data:** Time-series of hardware stats (RAM, VRAM, CPU utilization, GPU temperature, frame rendering time).
- **Target:** Binary classification. Given a 10-second window, predict if a crash occurs in the next 60 seconds.

---

## Part 5 — High-Level Solution

```
  [Offline Cloud Environment]
  1. Parse Petabytes of historical telemetry.
  2. Handle severe class imbalance (SMOTE, undersampling).
  3. Train a lightweight XGBoost model.
  4. Export model to C++ compatible format (e.g., Treelite or ONNX).
       │
       ▼ (Model shipped in Game Patch update)
       
  [Game Client (PC/Console) - Runs locally]
  ┌────────────────────────────────────────────────────────┐
  │ Telemetry Thread (1Hz)                                 │
  │  ↳ Circular Buffer of last 10 seconds                  │
  │  ↳ Feature Extraction (rolling mean, max, delta)       │
  │  ↳ Fast Inference (C++ Model Evaluator < 0.1ms)        │
  │  ↳ If P(Crash) > 0.90 ➔ Trigger Auto-Save Thread       │
  └────────────────────────────────────────────────────────┘
```

**Core ML Component:** A tiny, highly optimized tree-based model running locally on the game client's background thread, evaluating a rolling window of hardware metrics.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Label Engineering
- For crashed sessions: The last 60 seconds of telemetry are labeled `1`.
- For normal sessions: Randomly sample 60-second windows and label `0`.
- Because the action (auto-save) takes 2 seconds, predicting a crash 1 second before it happens is useless. The label must reflect a *leading indicator* (e.g., memory leak building up).

### Step 2: Feature Engineering (Time-Series to Tabular)
Instead of feeding raw time-series to an RNN, we calculate rolling statistics to keep inference cheap:
- `vram_usage_delta_10s`: Is VRAM spiking?
- `frame_time_variance_5s`: Is the framerate stuttering erratically?
- `gpu_temp_max_10s`: Is it overheating?

### Step 3: Imbalance Handling
- With a 1:10,000 ratio, a model predicting "No Crash" is 99.99% accurate.
- Use extreme undersampling on the negative class (down to 1:10 ratio) during training.
- Use Precision-Recall AUC (PR-AUC) as the primary evaluation metric, NOT accuracy or ROC-AUC.

### Step 4: Edge Deployment (C++ Compilation)
- Python is not available in the Frostbite engine.
- We must compile the trained XGBoost model into native C/C++ code using `Treelite` or ONNX, which executes in microseconds without heavy runtime dependencies.

---

## Part 7 — Complete Python Code (Training Pipeline)

```python
"""
crash_prediction_train.py - Offline training pipeline for edge model
"""
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc, classification_report
import treelite

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given 1Hz raw telemetry, extract rolling features.
    In production, this exact logic must be replicated in C++ on the client.
    """
    df = df.sort_values(['session_id', 'timestamp'])
    
    # 10-second rolling features
    rolling = df.groupby('session_id').rolling(10, min_periods=10)
    
    features = pd.DataFrame(index=df.index)
    features['vram_delta'] = df['vram_used'] - df.groupby('session_id')['vram_used'].shift(10)
    features['frame_time_var'] = rolling['frame_time_ms'].var().reset_index(0, drop=True)
    features['cpu_temp_max'] = rolling['cpu_temp'].max().reset_index(0, drop=True)
    
    # Target: 1 if this timestamp is within 60s of a crash
    features['target'] = df['will_crash_in_60s']
    
    # Drop rows without full 10s history
    return features.dropna()

def train_model(features: pd.DataFrame):
    logger.info("Handling class imbalance via undersampling...")
    positives = features[features['target'] == 1]
    negatives = features[features['target'] == 0]
    
    # 1:10 ratio for training
    negatives_sampled = negatives.sample(n=len(positives)*10, random_state=42)
    train_df = pd.concat([positives, negatives_sampled]).sample(frac=1.0)
    
    y = train_df['target']
    X = train_df.drop(columns=['target'])
    
    logger.info("Training XGBoost model...")
    # Strict limits on depth and trees to keep the model tiny (under 1MB)
    model = xgb.XGBClassifier(
        max_depth=4,
        n_estimators=50,
        learning_rate=0.1,
        scale_pos_weight=10,
        tree_method='hist'
    )
    model.fit(X, y)
    
    return model, X

def evaluate_model(model, test_X, test_y):
    logger.info("Evaluating on highly imbalanced test set...")
    preds = model.predict_proba(test_X)[:, 1]
    
    precision, recall, _ = precision_recall_curve(test_y, preds)
    pr_auc = auc(recall, precision)
    logger.info(f"PR-AUC: {pr_auc:.4f}")
    
    # Threshold tuning
    y_pred = (preds > 0.95).astype(int)
    print(classification_report(test_y, y_pred))

def compile_to_cpp(model: xgb.XGBClassifier, output_dir: str):
    """
    Compiles the XGBoost model to native C code for the game engine.
    """
    logger.info("Compiling model via Treelite...")
    # Convert XGBoost to Treelite format
    tl_model = treelite.Model.from_xgboost(model.get_booster())
    
    # Generate C source code
    treelite.generate_c_code(tl_model, dirpath=output_dir, params={'parallel_comp': 4})
    logger.info(f"C code generated at {output_dir}/main.c")
    # The game engine team will compile this into a .dll or .so library

if __name__ == "__main__":
    # Mock data execution
    logger.info("Running dummy pipeline...")
    # ... In reality, load from parquet ...
```

---

## Part 8 — Deployment

### Edge Deployment (Client-Side)
- We do not use Docker or Kubernetes for the inference side.
- The `treelite` generated C code is compiled directly into the Frostbite engine executable (e.g., as a static library).
- A background C++ thread runs every 1 second:
  1. Reads memory/CPU registers.
  2. Updates a circular array (ring buffer) of size 10.
  3. Calculates variance/max.
  4. Passes a float array to the `predict()` function generated by Treelite.
  5. Inference takes ~0.01 milliseconds.

### Server Deployment (Training)
- Apache Airflow orchestrates the weekly offline retraining.
- PySpark processes petabytes of telemetry to extract the training datasets.

---

## Part 9 — Unit Testing

```python
import pandas as pd
import numpy as np
from crash_prediction_train import extract_window_features

def test_extract_window_features():
    # Synthetic telemetry for one session
    df = pd.DataFrame({
        'session_id': [1]*15,
        'timestamp': list(range(15)),
        'vram_used': [1000 + i*100 for i in range(15)], # Steadily increasing
        'frame_time_ms': [16]*15,
        'cpu_temp': [70]*15,
        'will_crash_in_60s': [0]*14 + [1]
    })
    
    features = extract_window_features(df)
    
    # First 9 rows dropped due to rolling window (min_periods=10)
    assert len(features) == 6
    
    # VRAM delta for timestamp 10 should be val(10) - val(0)
    # 2000 - 1000 = 1000
    assert features.iloc[0]['vram_delta'] == 1000
    
    # Frame time variance should be 0 (constant 16ms)
    assert features.iloc[0]['frame_time_var'] == 0.0
```

---

## Part 10 — Integration Testing

- **C++ Verification:** We must guarantee that the C++ model produces the *exact* same probabilities as the Python XGBoost model.
- Write a Python script that exports 10,000 edge-case feature vectors to a CSV.
- Write a C++ test binary that loads the CSV, runs the compiled Treelite model, and outputs predictions.
- CI pipeline asserts `max(abs(python_preds - cpp_preds)) < 1e-5`.
- **Performance Profiling:** Run the C++ binary through Valgrind to ensure absolutely zero memory leaks (ironically, the anti-crash model cannot cause a crash).

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Data Ingestion** | We cannot train on all telemetry. We sample 100% of crashed sessions, and 0.1% of healthy sessions at the ingest layer (AWS Kinesis/Kafka) before writing to cold storage. |
| **Model Size Updates** | Over-the-air updates. Instead of waiting for a massive 20GB game patch, the compiled `.dll` or `.onnx` model weights are placed in a 1MB dynamic file that the game downloads silently on the main menu. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Client-side vs Server-side | Client-side uses zero cloud compute and has 0 network latency, but requires extreme model compression and limits algorithm choices. Server-side allows massive deep learning models but is slow and extremely expensive. |
| XGBoost vs Deep Learning (LSTM) | LSTM is theoretically better for time-series, but much heavier to run on a CPU. Treelite-compiled XGBoost evaluates purely via simple `if/else` branching, taking almost zero CPU cycles. |
| False Positives (Triggering save when no crash) | If we auto-save unnecessarily, the game stutters for 2 seconds. A high FP rate ruins the game experience worse than the crash itself. Precision must be tuned very strictly. |

---

## Part 13 — Alternative Approaches

1. **Survival Analysis:** Instead of binary classification ("Will it crash in 60s?"), predict the *time until crash*. (e.g., Weibull distribution).
2. **Anomaly Detection (Autoencoders):** Train an Autoencoder on healthy sessions. If reconstruction error spikes, the system is entering an unstable state. This requires an ONNX runtime on the client.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| The ML model has a memory leak | Model crashes the game | Strict static analysis, Valgrind testing, and C++ sandbox isolation in the engine. |
| Covariate Shift | A new game patch completely changes baseline VRAM usage | The model will constantly trigger false positives. Implement a "Kill Switch" via a cloud config file to instantly disable the client-side model globally without a game patch. |

---

## Part 15 — Debugging

**Symptom:** After a game update, the model stops predicting crashes entirely (False Negative rate = 100%).

**Debugging steps:**
1. Check telemetry distributions in the data warehouse. Did the new patch fix the memory leak that the model was heavily relying on (`vram_delta`)?
2. Did the engineering team rename a telemetry metric (e.g., `cpu_temp` -> `cpu_temperature_c`), causing the C++ feature extractor to pass `0.0` or `NaN` into the model?
3. Check if the model is still executing on the client. Did a game engine optimization disable background telemetry threads?

---

## Part 16 — Monitoring

Since the model runs on the client, we must rely on aggregated telemetry sent back to the server.

| Metric | Alert Threshold |
|--------|----------------|
| `client_ml_trigger_rate` | > 0.5% of sessions → Kill Switch (Likely False Positives) |
| `successful_crash_prevention_rate` | Evaluate offline: Did sessions that triggered an auto-save actually crash shortly after? |
| `model_execution_time_microseconds`| > 1000 µs (1ms) → Alert Engineering |

---

## Part 17 — Production Improvements

1. **Federated Learning:** Instead of uploading raw telemetry to our servers (huge bandwidth), train tiny gradients locally on the user's PC and only upload the model weight updates.
2. **Context-Aware Actions:** If a crash is predicted, but the player is mid-firefight, wait 5 seconds before auto-saving to avoid stuttering during critical gameplay.
3. **GPU Offloading:** Run the inference via DirectML or Vulkan compute shaders so it takes exactly zero cycles away from the main CPU game thread.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"The C++ engineers tell you they cannot implement a rolling 10-second variance function efficiently in the engine. How can you change your model architecture so they only have to pass the *current instantaneous* frame of data, but the model still understands time?"**
2. **"If the model predicts a crash, we trigger an auto-save. If the auto-save prevents the crash, how do you know if your model was correct, or if it was a false positive?"**
3. **"We have 10,000 crashes in our dataset, but there are 50 different *types* of crashes (GPU hang, out of memory, null pointer). Should we train one model for all of them, or 50 models?"**

---

## Part 19 — Ideal Answers

**Q1 (No rolling features in C++):**
> "We can use an Exponential Moving Average (EMA) instead of a strict 10-second rolling window. EMA only requires storing a single scalar state variable from the previous frame: `EMA_today = (value * alpha) + (EMA_yesterday * (1-alpha))`. This requires $O(1)$ memory and $O(1)$ compute. Alternatively, we move to a Recurrent model (like a minimal GRU/RNN via ONNX), where the hidden state holds the temporal memory, allowing the engine to pass only instantaneous features."

**Q2 (Counterfactual evaluation):**
> "This is the classic counterfactual problem in predictive maintenance. To prove the model works, we must use a Holdout Control Group. For 5% of players, the model runs in 'Shadow Mode'. It predicts the crash, logs the prediction to the server, but *does not* trigger the auto-save. We then verify offline if those specific 5% of players actually crashed."

**Q3 (Multi-class vs Binary):**
> "A null pointer exception is a logic bug; it happens instantly without warning and has no build-up in hardware telemetry. An Out-of-Memory (OOM) error builds up over minutes. We should only train our model on crashes that exhibit leading indicators (Resource Exhaustion, Thermal Throttling). Group those into one binary target. Throw out the logic bugs, as ML cannot predict them."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Immediately realizes the latency and bandwidth constraints dictate a client-side (Edge ML) architecture.
- Understands how to compile Python ML models to C++ (Treelite, ONNX, TF Lite).
- Provides excellent, resource-efficient feature engineering (EMA over raw windows).
- Flawlessly handles the counterfactual evaluation question (Shadow Mode).
- Eliminates "logic bug" crashes from the training set.

### Hire
- Designs a good time-series model.
- Mentions undersampling/SMOTE for the extreme class imbalance.
- Needs a slight nudge to realize the model must run locally on the client.
- Explains the C++ integration well conceptually.

### Lean Hire
- Focuses heavily on deep learning (LSTMs). Struggles to adapt when told the model must be under 1MB and run in 0.1ms.
- Does not know how to bridge the gap between Python training and C++ deployment, but understands the ML concepts.

### Lean No Hire
- Designs a server-side API where the game client sends a REST request every second. (This demonstrates a lack of understanding of game architecture).
- Evaluates the highly imbalanced model using Accuracy.

### No Hire
- Fails to formulate the time-series problem.
- Cannot write code to extract rolling features.
- Has no concept of edge deployment.
