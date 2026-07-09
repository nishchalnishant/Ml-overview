# Interview 27 — Multiplayer Latency Compensation (Time-Series Prediction)
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the Core Gameplay team for a fast-paced multiplayer shooter (like Apex Legends). Players across the world experience varying network latency (ping from 20ms to 200ms). When a player shoots at an enemy, the enemy might have already moved on the server by the time the shot registers.

Traditionally, games use linear interpolation (Lerp) or dead reckoning to guess where the enemy is. Your task is to **design an ML-based time-series forecasting model for Client-Side Prediction** that predicts the enemy's future position 100ms in advance, providing a smoother experience than linear math.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Compute Budget (This must run per-enemy, per-frame, on the client).
- Feature Space (Are we just using X,Y,Z, or player inputs?)
- Non-linearity (Why is ML better than linear physics here?)
- Error Correction (What happens when the prediction is wrong and the server sends the true position?)
- Dataset (How do we train this?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"What is the strict compute budget per inference?"**
   → *Answer: < 0.1ms per enemy. If there are 10 enemies on screen, we cannot spend 5ms running models, or the frame rate drops.*

2. **"Why use ML instead of simple physics `position = velocity * time`?"**
   → *Answer: Players don't move linearly. They strafe, jump, and change direction abruptly. We hope ML can learn player movement patterns (e.g., A-D strafing) better than physics.*

3. **"Do we have access to the enemy's controller inputs?"**
   → *Answer: No. The client only receives the enemy's X,Y,Z coordinates delayed by the network ping.*

---

## Part 4 — Expected Assumptions

- **Architecture:** A very small, ultra-fast sequence model. A shallow LSTM, GRU, or a Temporal Convolutional Network (TCN).
- **Features:** A sliding window of the last $N$ known positions and velocities.
- **Output:** Predicted X, Y, Z coordinates for $t + \Delta t$.
- **Correction:** Must implement smooth blending (Exponential Moving Average) to correct the prediction when the server's authoritative state arrives.

---

## Part 5 — High-Level Solution

```
  [Game Client (Enemy Data Receiver)]
  Receives Enemy Pos at t-100ms, t-80ms, t-60ms...
       │
       ▼
  [Feature Buffer]
  Calculates Velocity, Acceleration, and extracts last 10 frames.
       │
       ▼
  [Inference Engine (ONNX Runtime C++ API)]
  ┌────────────────────────────────────────────────────────┐
  │ Model: Shallow GRU (Gated Recurrent Unit).             │
  │ Input: Shape (1, 10, 6) -> 10 frames of (X,Y,Z, Vx,Vy,Vz)
  │ Output: Predicted (X, Y, Z) at t+100ms.                │
  └────────────────────────────────────────────────────────┘
       │
       ▼
  [Animation System] ➔ Smoothly interpolates to the predicted position.
```

**Core ML Component:** Formulating the problem as a multivariate time-series forecasting task and optimizing the model size so it can run 60 times a second without melting the CPU.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Data Collection & Preparation
- **Training Data:** Log raw player movement trajectories on the server at 60Hz.
- **Features:** Do not just use absolute `(X, Y, Z)`. If the player is at `X=10000`, the model might struggle with large floats. Use **delta values**: `(dX, dY, dZ)` from the current position.
- **Windowing:** Use a lookback window of 10 frames (~160ms) to predict the next 6 frames (~100ms ahead).

### Step 2: Model Architecture
- **Why GRU?** GRUs have fewer gates than LSTMs, making them faster to compute, while still capturing short-term temporal dependencies (like zig-zag strafing).
- **Size:** 1 hidden layer, 32 units. (Must be tiny).

### Step 3: Server Correction (Rubber-banding)
- When the server sends the *actual* position at `t=0`, the client realizes its prediction was off by 5 units.
- Do not instantly teleport the enemy to the server position (creates visual jitter/rubber-banding).
- Blend the predicted path and the server path over the next 10 frames.

---

## Part 7 — Complete Python Code

```python
"""
movement_predictor.py - GRU Time-Series Forecasting for Player Movement
"""
import logging
import torch
import torch.nn as nn
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Model Architecture
# ---------------------------------------------------------------------------
class MovementPredictorGRU(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=3, num_layers=1):
        super(MovementPredictorGRU, self).__init__()
        
        # Input: [Batch, SeqLen, Features (dX, dY, dZ, Vx, Vy, Vz)]
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim) # Output: Predicted dX, dY, dZ

    def forward(self, x):
        # We only care about the output from the final time step
        out, _ = self.gru(x)
        final_timestep_out = out[:, -1, :] 
        return self.fc(final_timestep_out)

# ---------------------------------------------------------------------------
# 2. Inference Logic (Simulating Client-Side Execution)
# ---------------------------------------------------------------------------
class ClientPredictionEngine:
    def __init__(self, model_path: str = None):
        self.model = MovementPredictorGRU()
        self.model.eval()
        # In prod, this would be an ONNX Runtime session in C++
        self.sequence_length = 10
        self.history_buffer = []

    def update_and_predict(self, current_pos, current_vel):
        """
        Called every time a network packet arrives.
        current_pos: (X, Y, Z)
        current_vel: (Vx, Vy, Vz)
        """
        # 1. Add to buffer
        feature_vector = np.concatenate([current_pos, current_vel])
        self.history_buffer.append(feature_vector)
        
        # Keep only the last 10 frames
        if len(self.history_buffer) > self.sequence_length:
            self.history_buffer.pop(0)
            
        # 2. Predict if buffer is full
        if len(self.history_buffer) == self.sequence_length:
            # Convert to tensor shape: [1, 10, 6]
            x_tensor = torch.FloatTensor([self.history_buffer])
            
            with torch.no_grad():
                pred_delta = self.model(x_tensor).numpy()[0]
                
            # Absolute prediction = Current Position + Predicted Delta
            predicted_future_pos = current_pos + pred_delta
            return predicted_future_pos
        else:
            # Not enough data, fallback to linear physics
            return current_pos + (current_vel * 0.100) # Assuming 100ms ping

# ---------------------------------------------------------------------------
# Example Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = ClientPredictionEngine()
    
    # Simulating receiving 10 packets from the server
    logger.info("Filling history buffer...")
    for i in range(10):
        pos = np.array([i*1.0, 0.0, 0.0]) # Moving perfectly along X axis
        vel = np.array([1.0, 0.0, 0.0])
        pred = engine.update_and_predict(pos, vel)
        
    logger.info(f"Final True Position: {pos}")
    logger.info(f"ML Predicted Position (+100ms): {pred}")
```

---

## Part 8 — Deployment

### Export and Runtime
- Export the PyTorch GRU to **ONNX**.
- Game engines (Unreal/Frostbite) execute the ONNX file using `ONNX Runtime C++ API`.
- **Threading:** Prediction must happen synchronously on the main thread right before the physics/animation step. Therefore, execution time must be absolutely minimal.

### Batching
- If there are 10 enemies on screen, do not call the model 10 times. Stack the matrices into a batch of `[10, 10, 6]` and run a single inference call to maximize SIMD CPU instructions.

---

## Part 9 — Unit Testing

```python
import torch
from movement_predictor import MovementPredictorGRU

def test_model_latency():
    import time
    model = MovementPredictorGRU()
    model.eval()
    
    dummy_input = torch.randn(10, 10, 6) # Batch of 10 enemies
    
    # Warmup
    for _ in range(10):
        model(dummy_input)
        
    # Measure
    start = time.perf_counter()
    with torch.no_grad():
        out = model(dummy_input)
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    
    # Inference must be blisteringly fast (< 1ms) in Python, faster in C++
    assert latency_ms < 2.0 
    assert out.shape == (10, 3)
```

---

## Part 10 — Integration Testing

- **Replay Validation:**
  - Record 1 hour of multiplayer movement data.
  - Run the ML Predictor vs Standard Linear Extrapolation.
  - Calculate the **Mean Squared Error (MSE)** between the predicted position at $t+100$ and the true server position at $t+100$.
  - Assert that $MSE_{ML} < MSE_{Linear}$. If the ML model is worse than simple math, discard it.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Varying Network Latency** | Ping isn't fixed at 100ms. It fluctuates. The model needs to know *how far ahead* to predict. We must add a 7th feature to the input vector: `target_delta_t`. During training, we randomly ask the model to predict 20ms, 50ms, or 150ms ahead, conditioning it on the network state. |
| **Model Size Optimization** | 32 hidden units in FP32 might still be too slow for mobile/Switch ports. Apply **Weight Pruning** (set weights near zero to exactly zero to create sparse matrices) and **INT8 Quantization** to accelerate inference and reduce L1 cache misses on the CPU. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| LSTMs/GRUs vs TCN (Temporal Convolutional Network) | GRUs process data sequentially, which is inherently slow. TCNs use 1D Convolutions, which can be processed in parallel. TCNs are generally faster on modern CPUs, but GRUs are smaller in memory footprint. |
| Relative (Delta) vs Absolute Coordinates | Using absolute coordinates `(14502, 500, -9932)` ruins neural networks because the activations explode. Delta coordinates `(dX=2.1)` keep inputs small and normalized. Tradeoff: Deltas accumulate rounding errors over time if not constantly synced with the absolute server truth. |

---

## Part 13 — Alternative Approaches

1. **Kalman Filters:** The traditional aerospace approach. A Kalman filter estimates the true state of a noisy system using linear dynamics. It is insanely fast (microseconds) and requires no training data. It is mathematically optimal for *linear* systems, but fails on sudden, unpredictable human inputs (like a sudden 180-degree turn).
2. **Transformer Attention:** Instead of a GRU, use a tiny Attention mechanism. Extremely powerful for finding patterns in the 10-frame window, but attention ($O(N^2)$) is massive overkill for $N=10$ frames and wastes compute.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| The "Ghosting" Problem (Overshoot) | Enemy is running towards a wall. ML model predicts they will run *through* the wall because it doesn't know the wall exists. The player sees the enemy clip into a wall, then rubber-band backward. | The ML output must be validated by the Client Physics Engine. If the ML predicts `X=100`, but there is a wall at `X=95`, the physics engine overrides the ML and clamps the position to `95`. |
| Jitter / High Variance | The model predicts slightly different paths frame-to-frame, causing the character model to vibrate violently. | Apply an Exponential Moving Average (EMA) or a low-pass filter to the ML outputs before passing them to the animation system. |

---

## Part 15 — Debugging

**Symptom:** During testing, the ML model consistently predicts that enemies will suddenly stop moving entirely, causing them to freeze on the client screen until the next server packet arrives.

**Debugging steps:**
1. Why is the model predicting zero velocity? Check the training data.
2. In the training data, 80% of the time, players are standing still (camping, looting).
3. The dataset is massively imbalanced. The ML model learned that predicting "0" is the safest bet to minimize Mean Squared Error (MSE).
4. **Fix:** Balance the dataset. Downsample the "standing still" frames. Introduce a custom loss function that heavily penalizes predicting 0 when the recent velocity history was non-zero.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `prediction_error_distance_m` | > 2 meters → The model is hallucinating wildly. Fallback to Linear Lerp. |
| `inference_time_cpu_cycles` | > 1ms → Turn off the model for distant enemies to save CPU. |

---

## Part 17 — Production Improvements

1. **Level of Detail (LOD) Scaling:** Do not run ML inference on enemies that are 500 meters away. They are too small on screen for the player to notice jitter. Only run the ML predictor on enemies within 50 meters, and use cheap linear math for distant enemies.
2. **Contextual Features:** Add categorical features like `stance` (standing, crouching, sliding). A sliding player has very different physics constraints than a running player, which heavily influences the prediction.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"The player shoots at the predicted ML position and hits it perfectly on their screen. But on the Server, the enemy had actually turned left. Does the server award the hit or deny it?"**
2. **"To fix the wall-clipping issue, you decide to feed the 'Distance to nearest wall' into the GRU. How does this impact the data engineering pipeline on the server?"**
3. **"We have millions of players. Instead of training one global model, could we train personalized models for every single player's movement style?"**

---

## Part 19 — Ideal Answers

**Q1 (Hit Registration / Lag Compensation):**
> "The Server is always authoritative. However, denying the hit frustrates the shooter. We use **Server-side Rewind (Lag Compensation)**. When the server receives the 'Shoot' packet, it looks at the timestamp, rewinds the enemy's hitbox to where it was at that exact timestamp on the server, and checks if it hits. The ML model is *purely visual* to make the enemy move smoothly on the client; it does not dictate hit registration logic."

**Q2 (Adding Spatial Context):**
> "Adding 'Distance to nearest wall' is extremely expensive. During training data collection, the server would have to run physics raycasts for every player on every frame just to build the dataset. This would tank server performance. A better approach is to keep the ML model purely kinematic (X,Y,Z only) and let the Client's physics engine handle collisions post-prediction."

**Q3 (Personalized Models):**
> "Training and serving millions of unique models is impossible. However, we can use a **Global Model with User Embeddings**. We train one massive model. For each player, we generate a small 16-dimensional embedding vector representing their 'playstyle' (e.g., jumpy vs grounded). We pass this embedding as an input to the global GRU model, allowing it to personalize predictions without needing millions of separate model weights."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Recognizes the dataset imbalance problem (players standing still).
- Explicitly separates the Visual Prediction (Client) from Hit Registration (Server Rewind).
- Optimizes the batching execution (SIMD) and proposes LOD scaling to save CPU.
- Uses Deltas instead of Absolute coordinates for training.

### Hire
- Sets up a logical GRU/LSTM architecture.
- Understands the need to keep the model tiny for client-side execution.
- Recognizes that ML output must be clamped/validated by the physics engine to prevent wall-clipping.

### Lean Hire
- Suggests an enormous Transformer model, requiring interviewer correction regarding CPU budgets.
- Doesn't mention how to handle the inevitable error correction when the server packet arrives (rubber-banding).

### Lean No Hire
- Thinks the ML model can dictate hit registration (authoritative client), exposing the game to massive cheating exploits.

### No Hire
- Fails to grasp the concept of network latency.
- Does not know how to handle time-series data.
