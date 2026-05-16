# Deployment Patterns for Production ML

Comprehensive reference for deploying, serving, and maintaining ML models in production — covering strategies, infrastructure, optimization, and common failure modes.

---

## Table of Contents

1. [Deployment Strategies](#1-deployment-strategies)
2. [Latency Optimization](#2-latency-optimization)
3. [Feature Store Design](#3-feature-store-design)
4. [Model Serving Infrastructure](#4-model-serving-infrastructure)
5. [Online vs Offline Learning](#5-online-vs-offline-learning)
6. [Training-Serving Skew](#6-training-serving-skew)
7. [Rollback and Versioning](#7-rollback-and-versioning)
8. [Multi-Model and Ensemble Serving](#8-multi-model-and-ensemble-serving)
9. [Key Interview Points](#9-key-interview-points)

---

## 1. Deployment Strategies

Controlled rollout patterns that limit blast radius and enable safe model updates.

---

### Blue/Green Deployment

Two identical environments run in parallel. Traffic is cut over atomically from the old (blue) to the new (green) environment. Rollback = flip traffic back.

```
         Load Balancer
        /              \
  [Blue v1]        [Green v2]   ← 0% traffic initially
       100%               0%

  → switch:
  [Blue v1]        [Green v2]
       0%               100%
```

**Properties:**
- Zero-downtime switch
- Full rollback in seconds (re-point LB)
- Requires 2x infrastructure during transition
- No gradual validation — all traffic moves at once

**When to use:** High-confidence model updates, schema changes that are hard to split, regulated environments requiring clean cutover.

---

### Canary Releases

Route a small fraction of live traffic to the new model, monitor, then gradually increase.

```
Step 1:  old=99%  new=1%   → monitor error rate, latency, business metrics
Step 2:  old=90%  new=10%  → still OK?
Step 3:  old=50%  new=50%  → compare KPIs
Step 4:  old=0%   new=100% → full rollout
```

**Traffic splitting in Kubernetes (Istio VirtualService):**

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: model-service
spec:
  http:
    - match:
        - uri:
            prefix: /predict
      route:
        - destination:
            host: model-v1
            port:
              number: 8080
          weight: 90
        - destination:
            host: model-v2
            port:
              number: 8080
          weight: 10
```

**Gradual rollout triggers:**
- Manual (human approval at each step)
- Automated (Argo Rollouts, Flagger) — advance if p99 latency < threshold AND error rate < threshold for N minutes

**Minimum observation window:** At minimum one full business cycle (day/week) at each traffic fraction before advancing, to catch temporal drift.

---

### Shadow Mode (Mirror Traffic)

Production traffic is duplicated and sent to the new model. The new model's responses are discarded — only logged for comparison. The old model still serves users.

```
Request ──┬──→ [Model v1] ──→ Response (served to user)
          │
          └──→ [Model v2] ──→ Response (logged, NOT served)

Offline:  compare v1 vs v2 outputs distribution, latency, disagreement rate
```

**Use cases:**
- Validate a new model on real traffic before it touches users
- Regression testing: does the new model disagree on any subset?
- Latency profiling under production load shape

**Implementation note:** Shadow requests must be fire-and-forget (async, non-blocking) so they do not affect production latency. Use a sidecar proxy or a message queue to fan out.

```python
import asyncio
import httpx

async def shadow_predict(payload: dict) -> dict:
    """Send to prod and shadow simultaneously; return only prod response."""
    async with httpx.AsyncClient() as client:
        prod_task   = client.post("http://model-v1/predict", json=payload)
        shadow_task = client.post("http://model-v2/predict", json=payload)

        prod_response, shadow_response = await asyncio.gather(
            prod_task,
            shadow_task,
            return_exceptions=True,
        )

    # Log shadow result asynchronously — do not block caller
    asyncio.create_task(log_shadow(payload, shadow_response))

    return prod_response.json()
```

---

### A/B Testing

Randomly assign users to model A or B; measure a business metric (conversion, click-through, revenue). Requires statistical significance before declaring a winner.

**Key concepts:**

| Term | Definition |
|------|-----------|
| MDE (Minimum Detectable Effect) | Smallest effect size worth detecting |
| Statistical power (1-β) | Probability of detecting a real effect; typically 0.80 |
| Significance level (α) | False positive rate; typically 0.05 |
| Sample size | Derived from MDE, α, power, and baseline variance |

**Sample size and significance calculation:**

```python
import numpy as np
from scipy import stats

def minimum_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Calculate minimum users per variant for a two-proportion z-test.

    Args:
        baseline_rate: Current conversion rate (e.g., 0.10 for 10%)
        mde:           Minimum detectable effect as absolute difference (e.g., 0.02)
        alpha:         Type I error rate
        power:         1 - Type II error rate

    Returns:
        Required sample size per variant
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_pool = (p1 + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)   # two-tailed
    z_beta  = stats.norm.ppf(power)

    se_null = np.sqrt(2 * p_pool * (1 - p_pool))
    se_alt  = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))

    n = ((z_alpha * se_null + z_beta * se_alt) / mde) ** 2
    return int(np.ceil(n))


def ab_test_significance(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> dict:
    """Two-proportion z-test for A/B experiment results."""
    p_c = control_conversions   / control_n
    p_t = treatment_conversions / treatment_n
    p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
    z  = (p_t - p_c) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "control_rate":     round(p_c, 4),
        "treatment_rate":   round(p_t, 4),
        "relative_lift":    round((p_t - p_c) / p_c, 4),
        "z_statistic":      round(z, 4),
        "p_value":          round(p_value, 4),
        "significant":      p_value < alpha,
    }


# Example
n = minimum_sample_size(baseline_rate=0.10, mde=0.02)
print(f"Need {n} users per variant")   # ~3,842

result = ab_test_significance(
    control_conversions=980,  control_n=10_000,
    treatment_conversions=1_050, treatment_n=10_000,
)
print(result)
# {'control_rate': 0.098, 'treatment_rate': 0.105, 'relative_lift': 0.0714,
#  'z_statistic': 2.28, 'p_value': 0.0226, 'significant': True}
```

**Common pitfalls:**
- Peeking (stopping early when p < 0.05 inflates false positives — use sequential testing or Bayesian approaches instead)
- Network effects (users influence each other — violates independence assumption)
- Simpson's paradox (aggregate result reverses when stratified)

---

## 2. Latency Optimization

Techniques to reduce model inference latency from seconds to milliseconds.

---

### Profiling with PyTorch Profiler

Find bottlenecks before optimizing.

```python
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

model = MyModel().eval().cuda()
inputs = torch.randn(1, 3, 224, 224).cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
) as prof:
    with torch.no_grad():
        for _ in range(20):          # warm up + profile
            model(inputs)
            prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

Key columns: `cuda_time_total`, `self_cpu_time_total`, `cpu_memory_usage`. Look for ops that dominate; common culprits: attention matmuls, large linear layers, data transfers between CPU/GPU.

---

### Model Quantization

Reduce weight precision to shrink model size and increase throughput.

**Dynamic quantization (CPU, post-training, no calibration data needed):**

```python
import torch
import torch.quantization

model = MyModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model,
    qconfig_spec={torch.nn.Linear},   # quantize Linear layers
    dtype=torch.qint8,
)

torch.save(quantized_model.state_dict(), "model_int8.pt")
```

**Static quantization (requires representative calibration data, better accuracy):**

```python
from torch.quantization import prepare, convert, get_default_qconfig

model.qconfig = get_default_qconfig("fbgemm")   # CPU; use "qnnpack" for mobile
prepare(model, inplace=True)

# Calibration pass
with torch.no_grad():
    for batch in calibration_loader:
        model(batch)

convert(model, inplace=True)
```

**FP16 (GPU):**

```python
model = model.half().cuda()
inputs = inputs.half().cuda()

# Or via autocast (mixed precision):
with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(inputs)
```

**Tradeoffs:**

| Method | Size reduction | Speedup | Accuracy loss |
|--------|---------------|---------|--------------|
| FP16   | 2x            | 1.5–3x  | Negligible   |
| INT8   | 4x            | 2–4x    | Small (~1%)  |
| INT4   | 8x            | 3–6x    | Moderate     |

---

### TorchScript / ONNX Export

Export model out of Python to eliminate interpreter overhead and enable cross-platform deployment.

**TorchScript:**

```python
import torch

model.eval()

# Tracing (works for fixed control flow)
example_input = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, example_input)
traced.save("model_traced.pt")

# Scripting (works for dynamic control flow)
scripted = torch.jit.script(model)
scripted.save("model_scripted.pt")

# Load and run without Python model definition
loaded = torch.jit.load("model_traced.pt")
output = loaded(example_input)
```

**ONNX export:**

```python
import torch
import onnx
import onnxruntime as ort
import numpy as np

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size"},   # dynamic batch dimension
        "output": {0: "batch_size"},
    },
)

# Verify
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# Run with ONNX Runtime
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession("model.onnx", sess_options, providers=["CUDAExecutionProvider"])

output = sess.run(None, {"input": dummy_input.numpy()})
```

---

### TensorRT

NVIDIA's inference optimizer. Fuses layers, selects optimal kernels, and exploits tensor cores.

```python
import tensorrt as trt
import torch
from torch2trt import torch2trt   # pip install torch2trt

model = MyModel().eval().cuda()
x = torch.randn(1, 3, 224, 224).cuda()

# Convert
model_trt = torch2trt(
    model, [x],
    fp16_mode=True,
    max_batch_size=32,
    max_workspace_size=1 << 30,   # 1 GB
)
torch.save(model_trt.state_dict(), "model_trt.pth")

# Benchmark
import time
with torch.no_grad():
    for _ in range(100):   # warm up
        model_trt(x)
    start = time.perf_counter()
    for _ in range(1000):
        model_trt(x)
    print(f"TRT latency: {(time.perf_counter() - start) / 1000 * 1000:.2f} ms")
```

Typical speedup over vanilla PyTorch: 3–10x on NVIDIA GPUs.

---

### Batching Strategies

**Dynamic batching** — accumulate requests over a time window, process together:

```python
import asyncio
from typing import List

class DynamicBatcher:
    def __init__(self, model, max_batch: int = 32, max_wait_ms: float = 10.0):
        self.model    = model
        self.max_batch = max_batch
        self.max_wait  = max_wait_ms / 1000
        self._queue: asyncio.Queue = asyncio.Queue()

    async def predict(self, input_tensor):
        """Called per request; returns result when batch is processed."""
        future = asyncio.get_event_loop().create_future()
        await self._queue.put((input_tensor, future))
        return await future

    async def _batch_loop(self):
        while True:
            items = []
            deadline = asyncio.get_event_loop().time() + self.max_wait

            # collect up to max_batch items or until deadline
            while len(items) < self.max_batch:
                timeout = deadline - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    items.append(item)
                except asyncio.TimeoutError:
                    break

            if not items:
                continue

            tensors, futures = zip(*items)
            batch = torch.stack(tensors)
            with torch.no_grad():
                results = self.model(batch)

            for future, result in zip(futures, results):
                future.set_result(result)
```

**Micro-batching** — split a large batch into smaller chunks that fit in GPU memory, overlap data transfer and computation.

**Async inference** — decouple request receipt from model execution using a queue (Kafka, Redis, SQS). Caller polls or receives a webhook. Good for non-latency-sensitive workloads.

---

## 3. Feature Store Design

A system that stores, computes, and serves features consistently between training and serving.

---

### Online vs Offline Store

| Dimension | Offline Store | Online Store |
|-----------|--------------|--------------|
| Backend | Data warehouse (BigQuery, S3 + Parquet) | Key-value store (Redis, DynamoDB, Cassandra) |
| Latency | Minutes–hours | < 10 ms |
| Use case | Training, batch scoring | Real-time inference |
| Scale | Petabytes | Millions of rows |
| Update frequency | Scheduled jobs (hourly/daily) | Streaming (Kafka) or micro-batch |

**The consistency problem:** The offline store may use a different computation than what runs at serving time, producing different feature values. This is the root cause of training-serving skew (see section 6).

---

### Point-in-Time Correct Joins

When creating a training dataset, feature values must be joined as of the event timestamp — not the current value. Prevents label leakage.

```
Event log:  user=A, timestamp=2024-01-10 14:00, label=1
Feature log: user=A, feature_updated=2024-01-10 12:00, value=5.0   ← use this
             user=A, feature_updated=2024-01-10 16:00, value=7.0   ← do NOT leak
```

```python
import pandas as pd

def point_in_time_join(
    events: pd.DataFrame,          # columns: entity_id, event_timestamp, label
    features: pd.DataFrame,        # columns: entity_id, feature_timestamp, feature_value
) -> pd.DataFrame:
    """Merge feature value that was most recently available before each event."""
    merged = events.merge(features, on="entity_id", how="left")
    # keep only feature rows that precede the event
    valid = merged[merged["feature_timestamp"] <= merged["event_timestamp"]]
    # take the latest available feature per event
    latest = (
        valid.sort_values("feature_timestamp")
             .groupby(["entity_id", "event_timestamp"])
             .last()
             .reset_index()
    )
    return events.merge(latest, on=["entity_id", "event_timestamp"], how="left")
```

---

### Feast Architecture

```
                  ┌──────────────────────────────┐
                  │        Feature Repository     │
                  │  (Python definitions + YAML)  │
                  └──────────────┬───────────────┘
                                 │  feast apply
                  ┌──────────────▼───────────────┐
                  │         Feast Registry        │
                  │   (metadata, schemas, TTLs)   │
                  └──────┬───────────────┬────────┘
                         │               │
              feast materialize     feast get_online_features
                         │               │
              ┌──────────▼──┐      ┌─────▼──────────┐
              │Offline Store│      │  Online Store   │
              │ (BigQuery)  │      │   (Redis)       │
              └─────────────┘      └─────────────────┘
```

**Feast feature store configuration:**

```python
# feature_store.yaml
project: my_ml_project
registry: gs://my-bucket/feast/registry.db
provider: gcp
online_store:
  type: redis
  connection_string: "localhost:6379"
offline_store:
  type: bigquery
  dataset: feast_features
```

```python
# features.py
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType

user = Entity(name="user_id", value_type=ValueType.INT64, description="User identifier")

user_stats_source = FileSource(
    path="s3://my-bucket/user_stats/",
    event_timestamp_column="event_timestamp",
)

user_stats_view = FeatureView(
    name="user_stats",
    entities=["user_id"],
    ttl=timedelta(days=7),
    features=[
        Feature(name="session_count_7d",   dtype=ValueType.INT64),
        Feature(name="avg_order_value_30d", dtype=ValueType.FLOAT),
        Feature(name="days_since_last_login", dtype=ValueType.INT32),
    ],
    source=user_stats_source,
)
```

```python
# Materialize to online store
from feast import FeatureStore
from datetime import datetime, timezone

store = FeatureStore(repo_path=".")
store.materialize_incremental(end_date=datetime.now(tz=timezone.utc))

# Retrieve at serving time
feature_vector = store.get_online_features(
    features=["user_stats:session_count_7d", "user_stats:avg_order_value_30d"],
    entity_rows=[{"user_id": 1234}],
).to_dict()
```

---

### Training-Serving Skew Detection via Feature Store

Log feature vectors at prediction time. Periodically compare their distribution against training data distributions using PSI (Population Stability Index) or KS test.

---

## 4. Model Serving Infrastructure

---

### TorchServe

PyTorch's official model server. Handles REST/gRPC, multi-model serving, batching, metrics.

```bash
# Package model
torch-model-archiver \
  --model-name resnet50 \
  --version 1.0 \
  --model-file model.py \
  --serialized-file resnet50.pt \
  --handler image_classifier \
  --export-path model_store/

# Start server
torchserve \
  --start \
  --model-store model_store/ \
  --models resnet50=resnet50.mar \
  --ts-config config.properties
```

```ini
# config.properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=4
job_queue_size=100
batch_size=8
max_batch_delay=50     # ms
```

---

### Triton Inference Server (NVIDIA)

Multi-framework server (TensorFlow, PyTorch, ONNX, TensorRT, custom). Key features: concurrent model execution, model ensemble, dynamic batching per model.

**Model repository layout:**

```
model_repository/
  resnet50/
    config.pbtxt
    1/
      model.onnx
  bert_encoder/
    config.pbtxt
    1/
      model.plan          # TensorRT engine
```

**config.pbtxt:**

```protobuf
name: "resnet50"
platform: "onnxruntime_onnx"
max_batch_size: 64

input [
  { name: "input"  data_type: TYPE_FP32  dims: [3, 224, 224] }
]
output [
  { name: "output" data_type: TYPE_FP32  dims: [1000] }
]

dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 10000
}

instance_group [
  { kind: KIND_GPU  count: 2 }   # 2 concurrent instances per GPU
]
```

**Model ensemble** (chain preprocessing → model → postprocessing without leaving server):

```protobuf
name: "ensemble_pipeline"
platform: "ensemble"
ensemble_scheduling {
  step [
    { model_name: "preprocess"  model_version: 1
      input_map  { key: "raw_input"   value: "raw_input" }
      output_map { key: "processed"   value: "processed" } },
    { model_name: "resnet50"    model_version: 1
      input_map  { key: "input"       value: "processed" }
      output_map { key: "output"      value: "logits" } },
    { model_name: "postprocess" model_version: 1
      input_map  { key: "logits"      value: "logits" }
      output_map { key: "label"       value: "label" } }
  ]
}
```

---

### BentoML

Framework-agnostic serving with first-class Python packaging and deployment to Kubernetes / BentoCloud.

```python
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

runner = bentoml.sklearn.get("iris_classifier:latest").to_runner()

svc = bentoml.Service("iris_classifier", runners=[runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def predict(input_data: np.ndarray) -> np.ndarray:
    return await runner.predict.async_run(input_data)
```

```bash
bentoml build
bentoml serve iris_classifier:latest --port 3000
bentoml containerize iris_classifier:latest   # produces Docker image
```

---

### Ray Serve

Production-grade serving library built on Ray. Handles autoscaling, request routing, model composition.

```python
import ray
from ray import serve
import torch

ray.init()
serve.start()

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5,
    },
)
class ModelDeployment:
    def __init__(self):
        self.model = torch.jit.load("model_traced.pt").cuda().eval()

    async def __call__(self, request):
        data = await request.json()
        tensor = torch.tensor(data["input"]).cuda()
        with torch.no_grad():
            output = self.model(tensor)
        return {"prediction": output.cpu().tolist()}

deployment = ModelDeployment.bind()
serve.run(deployment, route_prefix="/predict")
```

---

### SLA, Autoscaling, and Request Queuing

| Metric | Typical SLA target | Alert threshold |
|--------|--------------------|----------------|
| p50 latency | < 50 ms | > 100 ms |
| p99 latency | < 200 ms | > 500 ms |
| Error rate | < 0.1% | > 1% |
| Throughput | model-dependent | drop > 20% |

**Autoscaling signals:** CPU/GPU utilization, request queue depth, p99 latency. Prefer queue-depth-based scaling for bursty traffic (faster than utilization-based because it acts before resources are saturated).

---

## 5. Online vs Offline Learning

---

### Batch (Offline) Inference

Model is static. Predictions are computed periodically on a dataset and stored.

```
Raw data → [batch ETL] → Features → [trained model] → Predictions → DB → Application
```

- Latency: not relevant (results pre-computed)
- Staleness: depends on batch frequency (hourly, daily)
- Examples: recommendation pre-scoring, churn propensity for CRM campaigns

---

### Online Learning

Model updates continuously as new data arrives. Required when distribution shifts faster than retraining cycles.

**sklearn `partial_fit`:**

```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

clf   = SGDClassifier(loss="log_loss", learning_rate="adaptive", eta0=0.01)
scaler = StandardScaler()

# Streaming data loop
for batch in streaming_data_source():
    X_batch = np.array([sample["features"] for sample in batch])
    y_batch = np.array([sample["label"]    for sample in batch])

    X_scaled = scaler.partial_fit(X_batch).transform(X_batch)
    clf.partial_fit(X_scaled, y_batch, classes=[0, 1])
```

**river library (purpose-built for streaming ML):**

```python
from river import linear_model, preprocessing, metrics, stream

model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
metric = metrics.ROCAUC()

for x, y in stream.iter_csv("events.csv", target="label"):
    y_pred = model.predict_proba_one(x)
    metric.update(y, y_pred)
    model.learn_one(x, y)

print(f"Running AUC: {metric}")
```

**Concept drift handling:**
- ADWIN (Adaptive Windowing) — detect change in mean, shrink window
- Page-Hinkley test — detect mean shift in stream
- DDM (Drift Detection Method) — monitor error rate
- Response: retrain from scratch, increase learning rate, weight recent samples more

**Streaming features with Kafka + Flink:**

```
Kafka topic (events) → Flink job (aggregate, compute features) → Redis (online store)
                                                                 ↑
                                                     Model reads from here at inference
```

---

## 6. Training-Serving Skew

The model was trained on feature values computed one way; at serving time, the same "feature" is computed differently, producing systematically different values.

---

### Root Causes

| Cause | Example |
|-------|---------|
| Different code paths | Training uses pandas; serving uses SQL |
| Time zone handling | Training assumes UTC; serving sends local time |
| Null/missing treatment | Training fills null=0; serving passes null=-1 |
| Normalization mismatch | Scaler fit on training data, not re-applied at serving |
| Aggregation window drift | "last 7 days" computed at different timestamps |

---

### Detection

```python
import numpy as np
from scipy.stats import ks_2samp

def detect_skew(
    training_features: np.ndarray,
    serving_features: np.ndarray,
    feature_names: list,
    alpha: float = 0.05,
) -> dict:
    """
    KS test for distributional difference between training and serving features.
    Returns dict of {feature_name: (statistic, p_value, skewed_bool)}.
    """
    results = {}
    for i, name in enumerate(feature_names):
        stat, p = ks_2samp(training_features[:, i], serving_features[:, i])
        results[name] = {"ks_statistic": round(stat, 4), "p_value": round(p, 4), "skewed": p < alpha}
    return results

# PSI (Population Stability Index) — industry standard for feature drift
def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """PSI < 0.1 stable, 0.1–0.2 moderate change, > 0.2 significant drift."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_pct = np.histogram(expected, bins=np.percentile(expected, breakpoints))[0] / len(expected)
    actual_pct   = np.histogram(actual,   bins=np.percentile(expected, breakpoints))[0] / len(actual)

    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct   = np.clip(actual_pct,   1e-6, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
```

**Detection pipeline:**
1. Log every (request_id, features, prediction) at serving time
2. Sample training feature distributions at train time, store as reference
3. Hourly/daily: compare serving feature distributions to reference via PSI / KS
4. Alert if PSI > 0.2 for any feature

---

### Fix: Shared Feature Pipeline

```
               ┌─────────────────────────────────────┐
               │        Feature Pipeline (shared)     │
               │   same Python function, same logic   │
               └───────────┬─────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    Offline (training)           Online (serving)
    batch compute → store   streaming compute → cache
```

The gold standard: push the same feature computation function to both training (run as a batch Spark/Flink job) and serving (run as a real-time function). Feature stores (Feast, Tecton, Hopsworks) enforce this by definition.

---

## 7. Rollback and Versioning

---

### Model Registry

A centralized store for model artifacts, metadata, and lifecycle state.

**MLflow:**

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("fraud_detection")

with mlflow.start_run() as run:
    # Train ...
    mlflow.log_params({"n_estimators": 100, "max_depth": 6})
    mlflow.log_metrics({"auc": 0.923, "precision": 0.88})
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="fraud_classifier",
    )

# Promote to staging, then production
from mlflow import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="fraud_classifier", version=3, stage="Production"
)
```

**Semantic versioning for models:**

```
MAJOR.MINOR.PATCH

MAJOR: architecture change, incompatible input/output schema
MINOR: retrain with new data, same architecture
PATCH: hyperparameter tweak, threshold adjustment

e.g., fraud_classifier:2.4.1
```

---

### Automated Rollback Triggers

Monitor these signals post-deployment; roll back if thresholds are breached:

```python
# Pseudo-code for a rollback monitor
ROLLBACK_RULES = [
    {"metric": "error_rate_5m",      "operator": ">", "threshold": 0.05},   # >5% errors
    {"metric": "latency_p99_5m",     "operator": ">", "threshold": 500},    # >500ms p99
    {"metric": "prediction_auc_1h",  "operator": "<", "threshold": 0.85},   # AUC degraded
    {"metric": "null_prediction_rate","operator": ">", "threshold": 0.01},   # >1% nulls
]

def should_rollback(current_metrics: dict) -> tuple[bool, str]:
    for rule in ROLLBACK_RULES:
        value = current_metrics.get(rule["metric"])
        if value is None:
            continue
        if rule["operator"] == ">" and value > rule["threshold"]:
            return True, f"{rule['metric']}={value} > {rule['threshold']}"
        if rule["operator"] == "<" and value < rule["threshold"]:
            return True, f"{rule['metric']}={value} < {rule['threshold']}"
    return False, ""
```

**Rollback mechanism:**
- Blue/green: flip load balancer back to previous environment (< 30 s)
- Kubernetes: `kubectl rollout undo deployment/model-serving`
- Canary: reduce new model weight to 0%
- Registry: `client.transition_model_version_stage(name=..., version=prev_version, stage="Production")`

---

## 8. Multi-Model and Ensemble Serving

---

### Cascade Models

Use a cheap model to filter requests; only send hard cases to the expensive model.

```
Request → [Fast Model (e.g., logistic regression)]
             ├── confidence > 0.95 → return prediction directly (cheap path)
             └── confidence ≤ 0.95 → [Slow Model (e.g., large transformer)] → return
```

```python
class CascadePredictor:
    def __init__(self, fast_model, slow_model, confidence_threshold: float = 0.95):
        self.fast  = fast_model
        self.slow  = slow_model
        self.threshold = confidence_threshold

    def predict(self, features):
        proba = self.fast.predict_proba(features)
        confidence = proba.max(axis=1)

        results = np.empty(len(features), dtype=int)
        easy_mask = confidence >= self.threshold
        hard_mask = ~easy_mask

        if easy_mask.any():
            results[easy_mask] = proba[easy_mask].argmax(axis=1)
        if hard_mask.any():
            results[hard_mask] = self.slow.predict(features[hard_mask])

        return results
```

Typical cost reduction: 70–90% of requests handled by the fast model.

---

### Mixture of Experts Routing

Route each request to the most relevant specialist model based on input characteristics.

```python
class MixtureOfExperts:
    def __init__(self, router, experts: dict):
        """
        router:  model that maps input → expert_name
        experts: {expert_name: model}
        """
        self.router  = router
        self.experts = experts

    def predict(self, features):
        expert_name = self.router.predict(features)
        return self.experts[expert_name].predict(features)
```

---

### Feature Flags for Model Switching

Decouple deployment from activation. Ship new model code to all servers; activate via a flag in a config service (LaunchDarkly, internal config) without redeployment.

```python
import ldclient

ld = ldclient.get()

def predict(user_id: str, features: dict) -> float:
    use_new_model = ld.variation("use-model-v2", {"key": user_id}, default=False)

    if use_new_model:
        return model_v2.predict(features)
    else:
        return model_v1.predict(features)
```

This allows:
- Kill switch (disable v2 instantly for all users)
- Percentage rollout (10% of users get v2)
- Targeted rollout (specific user segments)

---

## 9. Key Interview Points

**Deployment strategies:**
- Blue/green = atomic switch, 2x infra cost, fast rollback. Canary = gradual with live validation. Shadow = safest validation but burns double compute.
- A/B test requires pre-calculated sample size based on MDE — never stop early based on p-value (peeking problem).

**Latency stack:**
- Profile first, optimize second. Common wins in order: batching → quantization (INT8/FP16) → TorchScript/ONNX (eliminate Python overhead) → TensorRT (kernel fusion) → async/parallel serving.
- Dynamic batching trades latency for throughput; tune max_batch_delay to your SLA budget.

**Feature stores:**
- Point-in-time joins are mandatory for correctness — missing them causes label leakage.
- Online store (Redis) for < 10 ms retrieval; offline store (BigQuery/S3) for training.
- Materialization = the process of computing features from raw data and writing to online store.

**Training-serving skew:**
- The single most common silent killer of model performance in production.
- Detection: log serving features, compare distributions to training reference (PSI > 0.2 = alert).
- Fix: shared feature computation code used at both training and serving time.

**Rollback:**
- Automated rollback triggers: p99 latency spike, error rate increase, AUC/metric degradation.
- Model registry stages: Staging → Production. Always keep N-1 version in "Archived" state for fast rollback.
- Semantic versioning: MAJOR for schema/architecture breaks.

**Online learning:**
- `partial_fit` (sklearn) and `river` for stateful streaming updates.
- Monitor concept drift explicitly (ADWIN, DDM) — do not assume the stream is stationary.
- Online learning adds operational complexity (model state management, catastrophic forgetting); prefer offline retraining when latency allows.

**Cascade / ensemble serving:**
- Cascade models are the primary cost-reduction lever when you have a cheap proxy and an expensive oracle.
- Ensemble serving: latency = max(model latencies) for parallel ensembles, sum for serial cascades.
- Feature flags decouple deployment from activation — ship first, activate safely.

**Common interview mistakes to avoid:**
- Saying "just retrain the model" without describing how you detect drift first.
- Treating A/B testing as equivalent to canary deployment (they serve different purposes: A/B = business metric comparison, canary = operational safety).
- Ignoring point-in-time correctness in feature joins.
- Not accounting for cold-start latency when autoscaling (pre-warm replicas).
