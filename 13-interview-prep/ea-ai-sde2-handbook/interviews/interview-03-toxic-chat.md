# Interview 03 — Toxic Chat Moderation at Scale
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the EA Trust & Safety team. You need to build a system to moderate in-game text chat across multiple game titles (Battlefield, Apex Legends, EA Sports FC). The system must detect and block toxic messages (hate speech, severe profanity, harassment) before they reach other players in the lobby.

Your task is to **design and implement a real-time toxic chat moderation microservice.**

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Latency constraints (how fast must we process a message to block it in real-time?)
- Languages supported (English only or global?)
- Throughput/QPS (messages per second at peak?)
- Accuracy trade-offs (False Positives vs. False Negatives)
- Action taken (Block message? Replace with ****? Ban player?)
- Context awareness (does the model need to see previous messages, or evaluate independently?)
- Game slang and lingo (e.g., "kill", "shoot" are normal in a shooter, but bad in a sports game).

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"What is the required latency SLA for this service?"**
   → *Answer: p99 < 50ms. If we take longer, the game client will timeout or feel noticeably laggy.*

2. **"What is the peak QPS (Queries Per Second)?"**
   → *Answer: ~20,000 QPS across all games globally at peak.*

3. **"Do we prefer blocking bad words (high false positive) or allowing some through to avoid blocking normal chat (high false negative)?"**
   → *Answer: We want to heavily minimize False Positives. Players get extremely angry if normal communication is blocked.*

4. **"What languages do we need to support?"**
   → *Answer: Let's focus on English for v1, but the architecture should be extensible to other languages.*

5. **"Do we need context (previous messages) or just the single string?"**
   → *Answer: Single string evaluation for v1 to keep latency low.*

6. **"How do we handle game-specific context (e.g., 'I will kill you' in Battlefield vs. FIFA)?"**
   → *Answer: The request will include the `game_id`. You should use it.*

---

## Part 4 — Expected Assumptions

- Stateless HTTP/gRPC microservice.
- Input: `{"player_id": "...", "game_id": "...", "message": "..."}`
- Output: `{"is_toxic": boolean, "confidence": float}`
- Because latency requirement is <50ms, large LLMs (GPT-4, Llama 70B) are impossible. We need a fast, smaller model (e.g., DistilBERT, fastText, or a regex/dictionary heuristic hybrid).
- Replacing text with asterisks (filtering) is acceptable.

---

## Part 5 — High-Level Solution

```
  Game Server (Client)
       │ (HTTP POST or gRPC)
       ▼
  API Gateway / Load Balancer
       │
       ▼
  Toxicity Moderation Service (FastAPI)
  ┌───────────────────────────────────────────────┐
  │ 1. Fast Cache / Regex Blocklist (O(1))        │
  │ 2. fastText / DistilBERT Inference (<10ms)    │
  │ 3. Game-specific threshold evaluation         │
  └───────────────────────────────────────────────┘
       │
       ▼ (Async logging)
  Kafka (for offline review and model retraining)
```

**Core ML component:** 
A hybrid approach. 
1. A highly optimized Regex/Dictionary filter for obvious slurs (instant).
2. A lightweight transformer model (e.g., DistilRoBERTa fine-tuned on toxic comment data) or `fastText` (sub-millisecond inference) to catch nuanced harassment or obfuscated spelling.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Text Preprocessing
- Lowercase, remove special characters (but be careful of obfuscation like "f@ck").
- Levenshtein distance matching against known slurs (optional, can be slow if not optimized).

### Step 2: The Model
- We will use ONNX Runtime to serve a fine-tuned DistilBERT model. ONNX provides significant CPU/GPU inference speedups compared to raw PyTorch.
- Alternatively, `fastText` is CPU-bound and extremely fast, perfect for 20k QPS if transformers are too heavy. Let's write the code assuming ONNX + DistilBERT.

### Step 3: Game-Specific Logic
- Pass the `game_id`. Use different threshold cutoffs for different games. E.g., Battlefield has a higher threshold for violence words than FIFA.

### Step 4: The Service
- FastAPI with async endpoints.
- Log all decisions asynchronously to Kafka/stdout to avoid blocking the request thread.

---

## Part 7 — Complete Python Code

```python
"""
toxicity_service.py - Real-time chat moderation service
"""
import time
import logging
from typing import Dict, Any
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EA Chat Moderation")

# ---------------------------------------------------------------------------
# Global State & Config
# ---------------------------------------------------------------------------
MODEL_PATH = "models/distilbert-toxic.onnx"
TOKENIZER_NAME = "distilbert-base-uncased"

# Thresholds per game (minimize false positives)
GAME_THRESHOLDS = {
    "battlefield_2042": 0.90,  # More lenient on violence
    "fifa_24": 0.75,           # Stricter
    "sims_4": 0.60,            # Very strict
    "default": 0.85
}

# ---------------------------------------------------------------------------
# ML Components
# ---------------------------------------------------------------------------
class ToxicClassifier:
    def __init__(self):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        logger.info("Loading ONNX model...")
        # Optimize for CPU serving (assuming Kubernetes standard nodes)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
        
    def predict(self, text: str) -> float:
        # 1. Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="np", 
            truncation=True, 
            max_length=128, 
            padding="max_length"
        )
        
        # 2. ONNX Inference
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        
        ort_outs = self.session.run(None, ort_inputs)
        # Assuming output is raw logits of shape (1, 2) [non-toxic, toxic]
        logits = ort_outs[0][0]
        
        # 3. Softmax to get probability
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        return float(probs[1]) # Return probability of 'toxic' class

# Lazy loading on startup
classifier: ToxicClassifier = None

@app.on_event("startup")
async def startup_event():
    global classifier
    classifier = ToxicClassifier()

# ---------------------------------------------------------------------------
# API Models & Endpoints
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    player_id: str
    game_id: str
    message: str

class ChatResponse(BaseModel):
    is_toxic: bool
    confidence: float
    processing_time_ms: float

def log_to_kafka(req: ChatRequest, resp: ChatResponse):
    """Simulated async Kafka producer for auditing and retraining."""
    # In production, use confluent-kafka-python
    logger.debug(f"KAFKA EMIT: {req.game_id} | {req.player_id} | {resp.is_toxic}")

@app.post("/v1/moderate", response_model=ChatResponse)
async def moderate_chat(req: ChatRequest, bg_tasks: BackgroundTasks):
    start_time = time.perf_counter()
    
    if not req.message.strip():
        return ChatResponse(is_toxic=False, confidence=0.0, processing_time_ms=0.0)

    try:
        # Fast path check (Regex blocklist could go here to skip ML)
        
        # ML Inference
        toxic_prob = classifier.predict(req.message)
        
        # Threshold logic
        threshold = GAME_THRESHOLDS.get(req.game_id, GAME_THRESHOLDS["default"])
        is_toxic = toxic_prob >= threshold
        
        proc_time = (time.perf_counter() - start_time) * 1000
        
        resp = ChatResponse(
            is_toxic=is_toxic,
            confidence=toxic_prob,
            processing_time_ms=proc_time
        )
        
        # Log asynchronously so we don't block the request
        bg_tasks.add_task(log_to_kafka, req, resp)
        
        return resp
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        # Fail open: If model fails, allow message to pass so game isn't disrupted
        raise HTTPException(status_code=500, detail="Internal inference error")
```

---

## Part 8 — Deployment

### Kubernetes
- CPU inference is often cheaper and scales better horizontally for small transformer models than GPU serving. We deploy on standard CPU instances.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: toxicity-service
spec:
  replicas: 10
  selector:
    matchLabels:
      app: toxicity-service
  template:
    metadata:
      labels:
        app: toxicity-service
    spec:
      containers:
      - name: api
        image: ea-registry/toxicity-service:latest
        resources:
          requests:
            cpu: "2"
            memory: "2Gi"
          limits:
            cpu: "4"
            memory: "4Gi"
        env:
          - name: OMP_NUM_THREADS
            value: "2" # Match ONNX intra_op config
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: toxicity-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: toxicity-service
  minReplicas: 10
  maxReplicas: 200 # Need massive scale for 20k QPS
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Part 9 — Unit Testing

```python
import pytest
from fastapi.testclient import TestClient
from toxicity_service import app, ToxicClassifier

client = TestClient(app)

# Mock the classifier to avoid loading weights during testing
class MockClassifier:
    def predict(self, text: str) -> float:
        if "badword" in text:
            return 0.95
        return 0.10

@pytest.fixture(autouse=True)
def override_classifier(monkeypatch):
    monkeypatch.setattr("toxicity_service.classifier", MockClassifier())

def test_clean_message():
    response = client.post("/v1/moderate", json={
        "player_id": "123", "game_id": "fifa_24", "message": "gg well played"
    })
    assert response.status_code == 200
    assert response.json()["is_toxic"] == False

def test_toxic_message():
    response = client.post("/v1/moderate", json={
        "player_id": "123", "game_id": "fifa_24", "message": "you are a badword"
    })
    assert response.status_code == 200
    assert response.json()["is_toxic"] == True

def test_empty_message():
    response = client.post("/v1/moderate", json={
        "player_id": "123", "game_id": "fifa_24", "message": "   "
    })
    assert response.status_code == 200
    assert response.json()["is_toxic"] == False

def test_game_thresholds():
    # If prob is 0.80, it should be toxic in FIFA (0.75 threshold) 
    # but NOT toxic in Battlefield (0.90 threshold)
    pass # Implementation requires dynamic mocking of the probability
```

---

## Part 10 — Integration Testing

- Start the FastAPI server using Docker Compose along with a mock Kafka broker (e.g., Redpanda).
- Send 1,000 requests using `Locust` or `k6` to verify that p99 latency stays below 50ms on the target hardware.
- Verify that messages are successfully published to the mock Kafka topic asynchronously.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **20,000 QPS** | A single FastAPI pod with ONNX DistilBERT on 2 CPUs might achieve ~50 QPS. We would need **400 pods**. This is expensive. |
| **Cost Optimization** | Replace DistilBERT with `fastText` for v1. fastText achieves >1,000 QPS per CPU. We would only need ~20 pods. Alternatively, use a Bloom Filter / Regex layer first. If the regex clears it or blocks it, skip ML. Only route complex messages to the ML model. |
| **Caching** | Implement Redis caching for frequent messages (e.g., "gg", "hello", "lol"). These shouldn't hit the ML model. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| DistilBERT vs fastText | DistilBERT has higher accuracy (understands context and obfuscation) but is 20x slower and more expensive than fastText. |
| CPU vs GPU serving | GPU has higher throughput (batching), but CPU has lower latency for single requests (batch size = 1) and is easier to autoscale in K8s. |
| Fail open vs fail closed | Failing open (allow message if service crashes) protects game experience but allows toxicity. Failing closed blocks chat entirely if the AI service drops. |

---

## Part 13 — Alternative Approaches

1. **Client-side filtering:** Send a bloom filter or dictionary to the game client. Evaluate text on the user's PC/console. Pros: Zero server cost, zero latency. Cons: Easily bypassed by hackers, hard to update models without patching the game.
2. **gRPC over HTTP/2:** REST/JSON adds parsing overhead. For sub-10ms latency at high scale, gRPC is strictly superior.
3. **Batching Inference (NVIDIA Triton):** Queue requests for 10ms, batch them (e.g., size 16), run through GPU, scatter results. Increases p50 latency but massively increases throughput.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Traffic Spike (Server Crash) | 200k QPS overwhelms service | Rate limiting at API Gateway; strict auto-scaling limits; Fail Open behavior (return `False`). |
| New bypass trend (L33t sp34k) | False negatives spike | Retrain model weekly. Use character-level embeddings (like Byte-Pair Encoding or Character-CNNs) rather than word-level. |
| Kafka broker down | Data loss for retraining | Async task should catch network errors and fallback to local file logging (promtail/fluentbit). |

---

## Part 15 — Debugging

**Symptom:** p99 Latency suddenly spikes to 200ms, causing players to complain about chat lag.

**Debugging steps:**
1. Check Prometheus metrics for `cpu_utilization`. If pods are at 100% CPU, ONNX is queuing requests internally. HPA (autoscaler) might be too slow.
2. Check if a new model weights file was deployed that is larger/unoptimized.
3. Check memory leaks. If Python memory balloons, garbage collection pauses can cause latency spikes.
4. Investigate input length. Are users copy-pasting 5,000-character copypastas? Ensure the API drops or truncates inputs `> 256` chars early in the request lifecycle.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `api_latency_p99_ms` | > 50ms → Alert |
| `api_5xx_errors` | > 1% → Critical |
| `toxicity_flag_rate` | Sudden drop to 0% or spike to 50% → Anomaly Alert |
| `cache_hit_rate` | Monitor efficiency of the "gg" / "lol" cache |

---

## Part 17 — Production Improvements

1. **Multi-language support:** Switch to `xlm-roberta` (multilingual) or run a fast language detection model (fastText langdetect) to route to specific lightweight models.
2. **Triton Inference Server:** Move the ONNX model out of the FastAPI Python process and into an optimized NVIDIA Triton or TorchServe container.
3. **Shadow Mode Deployment:** When deploying a new model, route traffic to both models, return v1's response to the client, but log v2's response to compare offline.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"Our game allows players to write chat messages that are up to 10,000 characters long (like a guild description). How does your system handle this if the transformer has a `max_length=128`?"**
2. **"Players figure out that if they add an invisible zero-width space character inside a bad word, your model completely misses it. How do you fix this?"**
3. **"At 20,000 QPS, logging every single request to Kafka is saturating our network bandwidth. How can we reduce this without losing the ability to retrain our model on toxic messages?"**
4. **"Why use ONNX Runtime in Python instead of just serving with standard PyTorch `.pt` models?"**

---

## Part 19 — Ideal Answers

**Q1 (Long text):**
> "For text exceeding 128 tokens, we can't just truncate, because the toxicity might be at the end. We need to implement a sliding window chunking mechanism. Split the 10,000 chars into chunks of ~100 tokens, run inference on a batch of chunks, and if *any* chunk exceeds the threshold, flag the whole message. To keep latency low, we should rate-limit or penalize massive text blobs, as they are likely spam."

**Q2 (Zero-width spaces/Obfuscation):**
> "The text preprocessing pipeline needs a robust normalization step. Before tokenization, we apply a regex to strip all non-printable unicode characters (like zero-width spaces). We should also map visually similar characters (e.g., Cyrillic 'а' to Latin 'a', '@' to 'a') using homoglyph detection."

**Q3 (Kafka bandwidth):**
> "We don't need to log everything. We should sample. We can log 100% of messages flagged as toxic (since they are rare and valuable for review), 100% of 'borderline' messages (probabilities between 0.4 and 0.6) for active learning, and only sample 1% of the completely benign messages (probabilities < 0.1)."

**Q4 (ONNX vs PyTorch):**
> "PyTorch is optimized for training and batch processing. ONNX Runtime uses graph optimizations (like operator fusion, constant folding) and is highly optimized for inference, especially on CPU via libraries like OpenMP and Intel MKL. It also removes the Python GIL overhead for the actual computation step, usually providing a 2x-5x speedup for single-batch inference."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Immediately flags latency as the primary constraint.
- Proposes a hybrid architecture (Regex/Cache + ML) to save compute.
- Chooses CPU serving over GPU for single-request low-latency.
- Implements ONNX for speed optimization.
- Answers the zero-width space question with text normalization / homoglyph mapping.
- Proposes sampling for Kafka logging to save bandwidth.

### Hire
- Uses FastAPI correctly with background tasks.
- Recognizes Transformers might be too slow and suggests smaller models or fastText.
- Understands threshold tuning for False Positives.
- Code is clean and handles errors gracefully (Fail Open).

### Lean Hire
- Suggests using an LLM (like GPT-4) via API, but pivots to small models when told latency must be <50ms.
- Code works but puts model loading inside the request handler.
- Does not understand ONNX or inference optimization.

### Lean No Hire
- Fails to optimize for latency.
- Cannot explain how tokenization handles out-of-vocabulary or obfuscated text.
- Suggests batch processing for a real-time requirement.

### No Hire
- Cannot write a basic FastAPI service.
- Doesn't understand the difference between text classification and generation.
- No concept of deployment or monitoring.
