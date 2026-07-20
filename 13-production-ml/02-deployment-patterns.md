---
module: Production Ml
topic: Deployment Patterns
subtopic: ""
status: unread
tags: [productionml, ml, deployment-patterns]
---
# Deployment Patterns

---

## Table of Contents

1. [Deployment Strategies](#deployment-strategies)
2. [Latency Optimization](#latency-optimization)
3. [Feature Store Design](#feature-store-design)
4. [Model Serving Infrastructure](#model-serving-infrastructure)
5. [Online vs Offline Learning](#online-vs-offline-learning)
6. [Training-Serving Skew](#training-serving-skew)
7. [Rollback and Versioning](#rollback-and-versioning)
8. [Multi-Model and Ensemble Serving](#multi-model-and-ensemble-serving)
9. [Key Interview Points](#key-interview-points)

---

## Deployment Strategies

**The problem**: you trained a model that beats the benchmark offline. Now you need to put it in production without taking the service down, without gambling on a broken model serving all your users at once, and without losing the ability to recover if something goes wrong. These three constraints — availability, risk, recovery — are always in tension.

**The core insight**: every deployment strategy is a different answer to the same question: *how much of the real decision do I trust to the new model before I know it works?* You trade speed of rollout against exposure to failure. The right strategy depends on how confident you are in the new model and how bad a mistake would be.

---

### Blue-Green Deployment

**The problem**: you need to deploy a new model version. If it has a bug, rollback requires either reverting in place (slow, risky) or pre-built automation. During a rolling update you might serve a mix of old and new model logic — a state that is hard to reason about.

**The core insight**: keep two complete environments. The "off" environment gets the new version deployed and fully tested with real traffic before you flip the switch.

**The mechanics**: Blue serves 100% of traffic. Deploy to Green. Run smoke tests against Green with zero user impact. When confident, flip the load balancer to Green in a single atomic operation. Blue stays idle for immediate rollback.

```python
def route_request(environment: str) -> str:
    """Load balancer reads this; change 'active' to flip environments."""
    config = load_routing_config()
    return config["active_environment"]  # "blue" or "green"
```

**What breaks**: cost doubles — two full environments run simultaneously. State synchronization is hard: feature stores, model registries, and caches must be consistent across both environments. Works poorly for stateful streaming pipelines where session state is tied to a specific instance.

---

### Canary Deployment

**The problem**: you want confidence from real traffic before full rollout, but you cannot afford to expose every user to a potentially broken model. Smoke tests only cover happy-path inputs; real production traffic reveals edge cases.

**The core insight**: expose a small slice of real traffic to the new model first. Measure on real users, not synthetic tests. Scale up only when metrics confirm the new model is working.

**The mechanics**: route a fixed fraction of traffic (5%) to the new model via deterministic hashing on a request identifier. Monitor business metrics and error rates for a defined window (typically 24 hours minimum), then step up: 5% → 25% → 50% → 100%. Rollback at any step if metrics degrade.

```python
import hashlib

def route_request(request_id: str, canary_fraction: float = 0.05) -> str:
    """
    Deterministic routing: same request_id always routes to the same model.
    This prevents a user from getting inconsistent experiences across requests.
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    bucket = hash_int % 100
    return "canary" if bucket < (canary_fraction * 100) else "production"
```

**What breaks**: if the canary fraction is too small, statistical power is low — you will not detect regressions of less than ~2% before full rollout. If your canary carries a different user segment (only mobile users, only power users), results do not transfer to the full population. Do not graduate from canary to full production based on system metrics alone; use business metrics.

---

### Shadow Deployment

**The problem**: you want to validate a new model on real traffic with zero risk of surfacing bad predictions to users. You need to know what the new model *would* predict on real inputs before you trust it with actual decisions.

**The core insight**: run the new model in parallel with production, but discard its outputs — fire it asynchronously on every request, never block the production path, log both outputs and analyze divergence offline. Full implementation, agreement-rate/score-correlation metrics table, and shadow-to-live rollout steps: see [03-model-governance.md](03-model-governance.md#8-shadow-deployment-for-high-stakes-models).

**What breaks**: shadow inference doubles serving cost. If the new model is slower than production, async shadow calls can stack up and exhaust resources under sustained load — run shadow on isolated compute. Shadow results reflect the production traffic distribution; rare inputs that break the new model are still invisible unless you specifically test for them.

---

### A/B Testing

**The problem**: offline metrics improved, but you do not know whether the new model produces better *business outcomes*. A 2% AUC improvement does not tell you whether users buy more products or churn less.

**The core insight**: split traffic into statistically independent groups. Measure the business metric that actually matters — not the proxy metric you optimized — and run the experiment until you have enough data to detect a meaningful difference.

**The mechanics**: group A receives the production model, group B the challenger. Run for the minimum sample size required to detect your minimum detectable effect at target statistical power.

```python
from scipy import stats

def minimum_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8
) -> int:
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = (z_alpha + z_beta)**2 * (p1 * (1-p1) + p2 * (1-p2)) / (p1 - p2)**2
    return int(n) + 1

def ab_test_significance(
    control_conversions: int, control_n: int,
    treatment_conversions: int, treatment_n: int
) -> dict:
    _, p_value = stats.proportions_ztest(
        [control_conversions, treatment_conversions],
        [control_n, treatment_n]
    )
    control_rate = control_conversions / control_n
    treatment_rate = treatment_conversions / treatment_n
    return {
        "p_value": p_value,
        "significant": p_value < 0.05,
        "lift": (treatment_rate - control_rate) / control_rate
    }
```

**What breaks**: novelty effects inflate treatment metrics in the first few days — users click on new things simply because they are new. Run experiments long enough to let novelty decay (typically 1-2 weeks minimum). Network effects (when user A and user B interact in the product) violate the independence assumption required for valid A/B tests; use cluster-randomized experiments instead. Running too many simultaneous experiments creates interaction effects you cannot untangle.

---

## Latency Optimization

**The problem**: your model achieves the accuracy you need in development, but in production the p99 latency is 800ms and users are abandoning requests. Serving cost is also unsustainable at scale.

**The core insight**: latency has a stack — data retrieval, preprocessing, model forward pass, postprocessing. Optimization at the wrong layer wastes time. The serving stack almost always has one dominant bottleneck. Profile first; optimize the bottleneck.

---

### Profiling the Serving Stack

Before optimizing anything, instrument every layer. Typical bottleneck distribution: feature retrieval (40-60%), model forward pass (20-40%), preprocessing (10-20%).

```python
import time
from contextlib import contextmanager

@contextmanager
def latency_tracker(stage: str, metrics_client):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics_client.histogram(f"inference.latency.{stage}", elapsed_ms)

# Use it:
with latency_tracker("feature_retrieval", metrics):
    features = feature_store.get_online_features(entity_id)

with latency_tracker("model_forward_pass", metrics):
    scores = model.predict(features)
```

---

### Quantization

**The problem**: full-precision (FP32) inference uses 4 bytes per weight. For large models this saturates memory bandwidth and limits throughput on both CPU and GPU.

**The core insight**: the model does not need FP32 at inference time. INT8 arithmetic is 2-4x faster on modern hardware. The accuracy loss from quantization is typically less than 1% on well-calibrated models.

**The mechanics**: INT8 maps FP32 weights to 8-bit integers via `x_int8 = round(x_fp32 / scale) + zero_point`. Calibrate the scale on a representative dataset.

```python
import torch

# Post-training dynamic quantization
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Export to ONNX for cross-framework serving with optimized runtime
torch.onnx.export(model_quantized, dummy_input, "model_int8.onnx", opset_version=13)
```

**What breaks**: INT8 quantization degrades more sharply on models that use activations with wide dynamic range (large embedding tables, high-variance attention scores). Always re-evaluate on a calibration set after quantizing. For GPU serving, FP16 is often a better trade: half-precision with minimal accuracy loss and 2x throughput on modern GPUs.

---

### Dynamic Batching

**The problem**: a single GPU forward pass on batch size 1 is nearly as expensive as batch size 8. You are leaving GPU utilization on the floor.

**The core insight**: accumulate requests that arrive close together in time and process them as a single batch. The GPU's parallelism is wasted on single-item batches.

**The mechanics**: hold incoming requests for up to `max_wait_ms`. When either the batch fills to `max_batch_size` or the timeout expires, flush the batch.

```python
import asyncio
from typing import List

class DynamicBatcher:
    def __init__(self, model, max_batch_size: int = 32, max_wait_ms: float = 10.0):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: List[asyncio.Future] = []
        self._inputs: list = []

    async def predict(self, features: dict) -> dict:
        future = asyncio.get_event_loop().create_future()
        self._queue.append(future)
        self._inputs.append(features)

        if len(self._queue) >= self.max_batch_size:
            await self._flush()
        else:
            asyncio.get_event_loop().call_later(
                self.max_wait_ms / 1000,
                asyncio.ensure_future,
                self._flush()
            )
        return await future

    async def _flush(self):
        if not self._queue:
            return
        batch = self._inputs[:self.max_batch_size]
        futures = self._queue[:self.max_batch_size]
        self._inputs = self._inputs[self.max_batch_size:]
        self._queue = self._queue[self.max_batch_size:]

        results = self.model.predict_batch(batch)
        for future, result in zip(futures, results):
            future.set_result(result)
```

**What breaks**: batching adds latency for the first request in a batch — it must wait up to `max_wait_ms` before being processed. At low traffic, `max_wait_ms` becomes the minimum latency floor regardless of how fast the model is. Tune this window based on your traffic pattern; at high traffic it fills before the timeout and adds near-zero overhead.

---

### TorchScript and ONNX

**The problem**: Python overhead in model forward passes is non-trivial. The Python interpreter adds overhead on every operation, and Python cannot be efficiently parallelized across threads.

**The core insight**: compile the model graph to a static form that bypasses the Python interpreter. ONNX enables cross-framework optimized runtimes (ONNX Runtime, TensorRT).

```python
# TorchScript: static compilation of PyTorch model
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Export to ONNX, then optimize with TensorRT for maximum GPU throughput
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=13)
# trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

**What breaks**: TorchScript requires all code paths to be statically typed. Dynamic control flow (Python conditionals that depend on tensor values) requires explicit annotation. TensorRT optimization is hardware-specific — an engine compiled for an A100 will not run on a T4.

---

## Feature Store Design

**The problem**: you have 15 features for your recommendation model. Three teams compute overlapping versions of the same features using slightly different logic. Feature computation at inference time takes 150ms because it joins three tables live. In your training pipeline, you accidentally used features computed with future data — the model looked great offline but failed in production because those features are not available at prediction time.

**The core insight**: features need two separate systems with a shared contract. An *offline store* for training: high throughput, batch-correct, point-in-time safe. An *online store* for inference: low latency, pre-materialized, single key lookup. The feature store enforces a single definition used in both contexts.

---

### Point-in-Time Correct Joins

**The problem**: when training on a label that occurred at time `t`, the feature values used must be those that were available at time `t - epsilon`, never after. Using future feature values is data leakage — the model learns a mapping that cannot exist at inference time.

**The mechanics**: for each training example, join the most recent feature row whose timestamp is strictly before the event's timestamp. Full implementation, complexity discussion, and sorted-merge/`AS OF` optimizations for production scale: see [system-design/08-feature-store-architecture.md](07-feature-store-architecture.md#point-in-time-correctness).

**What breaks**: if the feature store's timestamp index is coarser than your event stream (daily snapshots for hourly events), point-in-time joins return features that are stale by up to 24 hours — creating a systematic bias in training that does not match how inference works.

---

### Feast Architecture

```python
from feast import FeatureStore, FeatureView, Entity, Field, FileSource
from feast.types import Float64, String
from datetime import timedelta
import pandas as pd

user = Entity(name="user_id", join_keys=["user_id"])

user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=7),
    schema=[
        Field(name="age", dtype=Float64),
        Field(name="purchase_count_7d", dtype=Float64),
        Field(name="last_category", dtype=String),
    ],
    source=FileSource(
        path="data/user_features.parquet",
        timestamp_field="event_timestamp"
    ),
)

store = FeatureStore(repo_path=".")

# Materialize offline features to online store (Redis / DynamoDB)
store.materialize_incremental(end_date=pd.Timestamp.now())

# Inference: single key lookup, < 10ms from online store
features = store.get_online_features(
    features=["user_features:age", "user_features:purchase_count_7d"],
    entity_rows=[{"user_id": "user_123"}]
).to_dict()
```

**What breaks**: materialization lag — the online store reflects the last materialization run, not real-time feature values. For features that must be fresh (current session behavior, event-driven signals), stream materialization via Kafka → Flink → Redis is required. Batch and streaming feature pipelines often diverge in subtle ways (different aggregation windows, different handling of late-arriving events), which creates training-serving skew through the feature store itself.

---

## Model Serving Infrastructure

**The problem**: you have a model artifact. You need it to handle 10,000 requests per second with < 50ms p99 latency, roll out new versions without downtime, serve multiple models efficiently on shared GPU hardware, and give you visibility when something degrades.

**The core insight**: the model artifact is not a service. You need infrastructure that handles request routing, batching, hardware scheduling, health checking, versioning, and observability — separately from your model code.

---

### Serving Framework Selection

| Framework | Best for | Key capability |
|-----------|----------|----------------|
| TorchServe | PyTorch models, multi-model | Handler API, model archiver |
| Triton Inference Server | Multi-framework, GPU, high throughput | Dynamic batching, concurrent model execution |
| BentoML | Rapid deployment, custom preprocessing | Python-native, Docker/K8s integration |
| Ray Serve | Distributed Python, complex pipelines | Actor-based, composable serving pipelines |

### Triton Dynamic Batching Configuration

```protobuf
name: "recommendation_model"
backend: "pytorch_libtorch"
max_batch_size: 64
input [
  { name: "user_features" data_type: TYPE_FP32 dims: [-1, 128] }
]
output [
  { name: "scores" data_type: TYPE_FP32 dims: [-1, 1000] }
]
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 5000
}
instance_group [
  { kind: KIND_GPU count: 2 }
]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:23.10-py3
        resources:
          limits:
            nvidia.com/gpu: "1"
        readinessProbe:
          httpGet:
            path: /v2/health/ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /v2/health/live
            port: 8000
```

**What breaks**: GPU memory fragmentation when serving many models of varying sizes on shared hardware. Without explicit memory limits, a large model can OOM the node and evict all other models. Specify GPU memory fractions per model in Triton's configuration.

---

## Online vs Offline Learning

**The problem**: your fraud detection model was trained on last quarter's transaction data. Fraudsters adapt — within three months, new fraud patterns that were not in the training data account for 40% of losses. Retraining from scratch monthly is expensive and slow.

**The core insight**: offline learning treats the training set as fixed. Online learning updates the model incrementally as each new example arrives. The right choice depends on how fast your data distribution shifts and the cost of stale predictions.

---

### Offline (Batch) Retraining

- Train on a fixed snapshot of historical data
- Deploy periodically (daily, weekly, monthly)
- Simple, auditable, reproducible

**Use when**: distribution shift is slow (weeks-months), labels arrive quickly, audit requirements are strict.

### Online (Incremental) Learning

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=0.01)

for batch in stream_transactions():
    X_batch, y_batch = prepare_features(batch)
    model.partial_fit(X_batch, y_batch, classes=[0, 1])

    if should_evaluate():
        score = model.score(X_eval, y_eval)
        log_metric("online_accuracy", score)
```

**What breaks**: online learning with a high learning rate causes catastrophic forgetting — the model chases the latest distribution and loses knowledge of older patterns. Use a smaller learning rate and maintain a replay buffer of historical examples to interleave with new data.

---

### Streaming Features for Online Serving

When features must be fresh (< 1 second lag):

```
Transaction events → Kafka topic
  → Flink/Spark Streaming (compute: rolling counts, velocity, aggregates)
  → Redis (feature store, TTL = 1 hour)
  → Model server reads from Redis at inference time
```

**What breaks**: streaming feature computation and offline batch feature computation often diverge — different aggregation windows, different handling of late-arriving events. This is a training-serving skew problem introduced through the feature pipeline.

---

### Concept Drift Detection

```python
from river import drift

detector = drift.ADWIN(delta=0.002)

for prediction, actual_label in production_stream():
    error = int(prediction != actual_label)
    detector.update(error)
    if detector.drift_detected:
        trigger_retraining_pipeline()
        detector = drift.ADWIN(delta=0.002)  # reset after triggering
```

Detection methods:
- **ADWIN**: maintains an adaptive window; triggers when the mean within sub-windows differs significantly
- **DDM**: tracks error rate and standard deviation; alerts when error exceeds baseline + threshold
- **Page-Hinkley test**: sequential test that triggers when cumulative deviation exceeds a threshold

**What breaks**: ADWIN has a detection lag — it takes several hundred examples to confirm a drift. In high-stakes applications, you may need to trigger retraining before statistical significance is reached. Use business metric thresholds as an early warning, not just statistical tests.

---

## Training-Serving Skew

**The problem**: your model achieves 94% accuracy in offline evaluation. In production it is 82%. Nobody changed the model. The degradation is real, silent, and grows over time.

**The core insight**: training-serving skew means the features the model saw during training are not the same features it receives at inference time. The distribution is different, the feature values are computed differently, or the timing is wrong. The model is solving a different problem than it was trained on.

---

### Root Causes

1. **Different preprocessing code**: training uses Python/pandas, serving uses Java/SQL — floating-point behavior differs
2. **Feature computation timing**: training uses a daily batch snapshot; serving computes in real-time with incomplete aggregation windows
3. **Missing features at serving time**: a feature available in historical data is not available in the live request
4. **Data leakage in training**: training used post-event features that do not exist at inference time

### Detection with PSI and KS Test

```python
import numpy as np
from scipy import stats

def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10
) -> float:
    """
    PSI < 0.1:   no significant change
    PSI 0.1-0.2: moderate change — monitor
    PSI > 0.2:   significant shift — investigate and likely retrain
    """
    def _psi_bucket(expected_pct, actual_pct):
        if expected_pct == 0 or actual_pct == 0:
            return 0
        return (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)

    exp_pcts = exp_counts / len(expected)
    act_pcts = act_counts / len(actual)

    return sum(_psi_bucket(e, a) for e, a in zip(exp_pcts, act_pcts))

def ks_feature_drift(training_feature: np.ndarray, serving_feature: np.ndarray) -> dict:
    statistic, p_value = stats.ks_2samp(training_feature, serving_feature)
    return {
        "ks_statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < 0.05
    }
```

### The Fix: Shared Feature Pipeline

The only reliable fix is to use identical code for both training and serving. Serialize the full preprocessing pipeline with the model artifact.

```python
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Training: build and save full pipeline together
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", your_model)
])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "model_with_preprocessing.pkl")

# Serving: load and apply the SAME transformations
loaded_pipeline = joblib.load("model_with_preprocessing.pkl")
prediction = loaded_pipeline.predict(raw_features)
```

**What breaks**: if the scaler was fit on training data but production data has a different scale (new product category with much higher prices), predictions will be wrong even with the shared pipeline. The fix addresses code divergence; it does not protect against legitimate distribution shift. Monitor feature statistics at serving time separately.

---

## Rollback and Versioning

**The problem**: you deployed a new model. Three hours later the business conversion rate dropped 8%. You need to roll back to the previous version immediately, without losing the ability to investigate what went wrong.

**The core insight**: rollback is only possible if you have versioned model artifacts, versioned serving configurations, and an automated mechanism to point traffic back to a known-good version. Without version management, you cannot roll back — you can only redeploy from scratch.

---

### Semantic Versioning for Models

```
v{major}.{minor}.{patch}

major: architecture change, incompatible feature schema
minor: retrained with new data, same architecture
patch: threshold adjustment, serving config change only
```

### MLflow Model Registry Lifecycle

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model after training run
run_id = mlflow.active_run().info.run_id
registered = mlflow.register_model(f"runs:/{run_id}/model", "fraud_detector")

# Lifecycle: None → Staging → Production → Archived
client.transition_model_version_stage(
    name="fraud_detector",
    version=registered.version,
    stage="Staging"
)

# After validation, promote to production
client.transition_model_version_stage(
    name="fraud_detector",
    version=registered.version,
    stage="Production"
)

# Archive the previous version (keep artifact for rollback and audit)
client.transition_model_version_stage(
    name="fraud_detector",
    version=str(int(registered.version) - 1),
    stage="Archived"
)
```

### Automated Rollback Trigger

```python
def should_rollback(
    current_metrics: dict,
    baseline_metrics: dict,
    thresholds: dict
) -> bool:
    checks = {
        "accuracy_drop": (
            baseline_metrics["accuracy"] - current_metrics["accuracy"]
            > thresholds.get("max_accuracy_drop", 0.03)
        ),
        "error_rate_spike": (
            current_metrics["error_rate"]
            > thresholds.get("max_error_rate", 0.01)
        ),
        "latency_regression": (
            current_metrics["p99_latency_ms"]
            > thresholds.get("max_p99_latency_ms", 200)
        ),
    }
    failed = {k: v for k, v in checks.items() if v}
    if failed:
        alert(f"Rollback triggered: {list(failed.keys())}")
        return True
    return False
```

**What breaks**: automated rollback without root cause analysis creates rollback loops — the model is rolled back, re-deployed, detected as bad, rolled back again. Add a circuit breaker: after two automated rollbacks, require human approval before re-deploying. Also, rollback does not fix the root cause; it only buys time.

---

## Multi-Model and Ensemble Serving

**The problem**: a single model cannot serve all use cases optimally. High-value users need a sophisticated (slow) model; the majority of requests can be handled by a fast cheap model. Or you want to combine multiple models for better accuracy, but you need this to be manageable in production.

**The core insight**: multi-model serving patterns (cascade, mixture of experts, champion-challenger) are fundamentally about routing decisions — which model handles which request, and how do you combine outputs when multiple models answer?

---

### Cascade Predictor

**The problem**: you want the accuracy of an expensive model but cannot afford to run it on every request.

**The core insight**: route requests through a cheap fast model first. Escalate to the expensive model only when the cheap model is uncertain.

```python
class CascadePredictor:
    def __init__(self, fast_model, accurate_model, confidence_threshold: float = 0.8):
        self.fast = fast_model
        self.accurate = accurate_model
        self.threshold = confidence_threshold

    def predict(self, features: dict) -> dict:
        fast_result = self.fast.predict_proba(features)
        max_confidence = max(fast_result["probabilities"])

        if max_confidence >= self.threshold:
            return {
                "prediction": fast_result["class"],
                "confidence": max_confidence,
                "model_used": "fast"
            }

        accurate_result = self.accurate.predict_proba(features)
        return {
            "prediction": accurate_result["class"],
            "confidence": max(accurate_result["probabilities"]),
            "model_used": "accurate"
        }
```

**What breaks**: the confidence threshold must be calibrated on held-out data. A poorly calibrated model can have high confidence on wrong predictions — the cascade never escalates when it should. Measure the escalation rate in production; if it is near zero, the fast model's confidences are overconfident.

---

### Mixture of Experts

```python
class MixtureOfExperts:
    def __init__(self, experts: list, gating_model):
        self.experts = experts
        self.gating = gating_model

    def predict(self, features: dict) -> dict:
        weights = self.gating.predict_proba(features)  # shape: [n_experts]
        expert_predictions = [model.predict(features) for model in self.experts]
        combined = sum(w * p for w, p in zip(weights, expert_predictions))
        return {"prediction": combined, "expert_weights": weights.tolist()}
```

**What breaks**: training the gating model requires the expert models to already exist. The gating model can collapse — assigning all weight to one expert — if experts do not naturally specialize. Regularize the gating model to encourage load balancing across experts.

---

### Feature Flags for Model Rollout

Control which users get which model version without redeploying:

```python
def get_model_variant(user_id: str, ld_client) -> str:
    context = ld.Context.builder(user_id).kind("user").build()
    return ld_client.variation("model-version", context, "v1")

def serve_prediction(user_id: str, features: dict) -> dict:
    variant = get_model_variant(user_id, ld_client)
    model = model_registry[variant]
    return model.predict(features)
```

**What breaks**: feature flag systems add round-trip latency overhead (typically 1-5ms for local evaluation). If the flag service is down and you have no fallback, all users get the default model regardless of the intended rollout. Always define a safe default that works without the flag service.

---

## Key Interview Points

**"How would you deploy a model to production?"**

Choose the deployment strategy based on risk tolerance: shadow (zero risk, maximum visibility) → canary (partial exposure, statistical validation) → blue-green (instant switchover, full replacement). Package the model with its preprocessing pipeline. Instrument for latency and quality monitoring. Define rollback criteria before deployment and automate rollback. Graduate from canary to full traffic based on business metrics, not just ML metrics.

**"What is training-serving skew?"**

Feature values computed at training time differ from feature values at inference time. Root causes: different code paths computing the same feature, different timing (batch vs real-time aggregation), data leakage in training. Detection: PSI and KS test on feature distributions comparing training data against production traffic. Fix: shared preprocessing library, serialize full pipeline, feature store with point-in-time guarantees.

**"Batch vs real-time inference trade-offs?"**

Batch: high throughput, offline, can use large models, predictions become stale between runs. Real-time: low latency (< 100ms), online, constrained model size, always fresh. Choose based on the freshness requirement: if a 24-hour-old recommendation is acceptable, batch is cheaper. If you need to respond to user behavior in the current session, you need real-time.

**"How do you detect if a model is degrading?"**

Three signals in order of detection speed: (1) data drift — input feature distributions shift, detected within hours using PSI/KS tests; (2) prediction drift — output score distribution changes, detected within hours; (3) performance degradation — actual accuracy drops once labels become available, detected with days-to-weeks lag. Set up automated alerts on all three. Do not wait for labels — by then the damage is done.

**"How do you handle a bad model in production?"**

Rollback immediately to the previous versioned artifact. Do not investigate while users are affected. After rollback: run a post-mortem, compare training data distribution vs production at time of failure, check shadow logs for prediction divergence, review feature pipeline for any upstream data changes. Automate rollback triggers so humans only need to approve, not execute.

## Flashcards

**Which serving framework gives dynamic batching + concurrent multi-model execution on GPU?** #flashcard
Triton Inference Server

**What characterizes offline (batch) retraining?** #flashcard
Trains on a fixed historical snapshot, deployed periodically (daily/weekly/monthly) — simple, auditable, reproducible, but stale between runs.

**ADWIN?** #flashcard
Maintains an adaptive window; triggers when the mean within sub-windows differs significantly.

**DDM?** #flashcard
Tracks error rate and standard deviation; alerts when error exceeds baseline + threshold.

**Page-Hinkley test?** #flashcard
Sequential test that triggers when cumulative deviation exceeds a threshold.
