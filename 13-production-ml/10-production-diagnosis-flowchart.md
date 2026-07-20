---
module: Production ML
topic: System Design
subtopic: Production Diagnosis Flowchart
status: unread
tags: [productionml, ml, system-design-production-diagn]
---
# Production ML Diagnosis Flowcharts

Decision trees for the four most common production ML incidents. Each tree is a structured interview framework — walk the interviewer through it to demonstrate systematic thinking.

---

## 1. Model Accuracy Drop

```
Accuracy dropped in production
         │
         ▼
Is it sudden (< 1h) or gradual (days/weeks)?
         │
    ┌────┴────┐
  Sudden    Gradual
    │           │
    ▼           ▼
Deployment   Data drift
issue?       or concept drift?
```

### Sudden Drop — Deployment Checklist

```
1. Was a new model deployed recently?
   YES → compare new vs old model predictions on same batch
         → if diverged: rollback
   NO  → continue

2. Did upstream data pipeline change?
   Check: feature schema version, null rates, value distributions
   Tool: data_quality_report.py --compare yesterday today
   YES → fix pipeline, retrain if needed
   NO  → continue

3. Is the label pipeline broken?
   Check: label join success rate, label delay, label distribution
   YES → fix label pipeline (evaluation is wrong, model may be fine)
   NO  → continue

4. Infrastructure issue?
   Check: serving version matches trained version
   Check: feature serving latency (stale features?)
   Check: model binary integrity (checksum)
```

### Gradual Drop — Drift Diagnosis

```
Compute PSI (Population Stability Index) for top features:
  PSI = Σ (P_actual - P_expected) × ln(P_actual / P_expected)
  
  PSI < 0.1  → no significant drift
  0.1–0.25  → moderate drift, monitor
  PSI > 0.25 → significant drift, action required

       PSI high?
          │
     ┌────┴────┐
    YES        NO
     │          │
     ▼          ▼
Feature      Concept drift
drift        (label relationship changed)
     │          │
     ▼          ▼
Fix upstream  KS test on
pipeline or  residuals over
retrain      time windows
```

**PSI computation:**
```python
def psi(expected, actual, buckets=10):
    def scale_range(x, mn, mx, buckets):
        return np.floor(buckets * (x - mn) / (mx - mn + 1e-8)).astype(int)
    
    mn = min(expected.min(), actual.min())
    mx = max(expected.max(), actual.max())
    
    exp_counts = np.bincount(scale_range(expected, mn, mx, buckets), minlength=buckets)
    act_counts = np.bincount(scale_range(actual, mn, mx, buckets), minlength=buckets)
    
    exp_pct = (exp_counts + 1) / (len(expected) + buckets)
    act_pct = (act_counts + 1) / (len(actual) + buckets)
    
    return np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))

# Feature drift: run on all top-20 features
# Label drift: run on predicted score distribution
```

**Concept drift detection (ADWIN algorithm):**
```python
# ADWIN: adaptive windowing — splits window when error rate changes significantly
# Available via: from river.drift import ADWIN
# Use when: true labels arrive with delay (e.g., fraud confirmed 7 days later)

# Simpler: KS test on rolling prediction residuals
from scipy.stats import ks_2samp

def detect_concept_drift(residuals_old, residuals_new, alpha=0.01):
    stat, p_value = ks_2samp(residuals_old, residuals_new)
    return p_value < alpha  # True = drift detected
```

### Data Drift vs Concept Drift — Key Distinction

| Type | Definition | Example | Fix |
|---|---|---|---|
| Data drift (covariate) | P(X) changes, P(Y\|X) stable | User demographics shift | Retrain on recent data |
| Concept drift | P(Y\|X) changes, P(X) stable | Fraud pattern changes | Retrain, possibly new features |
| Label drift | P(Y) changes | Click rates drop seasonally | Recalibrate threshold |
| Virtual drift | P(X) changes into unexplored region | New product category | Cold start treatment |

---

## 2. Latency Spike

### TTFT vs TPOT (LLM-specific)

```
User reports slow response
         │
         ▼
Is it LLM inference or general ML serving?
         │
    ┌────┴────────┐
   LLM          Other ML
    │               │
    ▼               ▼
TTFT or TPOT?    See Section 2b
    │
    ├── TTFT slow (time to first token)
    │   = prefill stage bottleneck
    │   Causes: long context, prompt too large
    │   Fix: reduce prompt length, increase prefill batch size,
    │        enable chunked prefill, use KV cache for shared prefix
    │
    └── TPOT slow (time per output token)
        = decode stage bottleneck
        Causes: small batch (memory-bound), KV cache full
        Fix: increase batch size, enable PagedAttention,
             use speculative decoding, reduce max output length
```

**TTFT formula:**
$$\text{TTFT} \propto \frac{S_{prompt}^2 \cdot d_{model}}{N_{GPU} \cdot \text{FLOPS/s}}$$

Prefill is compute-bound (matrix multiply over full prompt).

**TPOT formula:**
$$\text{TPOT} \propto \frac{2 \cdot P_{model}}{N_{GPU} \cdot \text{Bandwidth}}$$

Decode is memory-bandwidth-bound (load all weights per token).

```
Arithmetic intensity = FLOPs / bytes_accessed

For decode (batch=1):
  FLOPs = 2 × P   (one forward pass, ~2 ops per param)
  Bytes = P × dtype_bytes (load all weights)
  Intensity = 1 op/byte → memory bound

For prefill (batch=1, seq=S):
  FLOPs = 2 × P × S
  Bytes = P × dtype_bytes
  Intensity = S ops/byte → compute bound at S > ~100
```

### 2b. General ML Serving Latency

```
P99 latency spike
      │
      ▼
Single request slow or all requests?
      │
 ┌────┴────┐
All       Single
 │          │
 ▼          ▼
System    Outlier input
issue     (long sequence,
 │         unusual features)
 ▼
Check cascade:

1. Feature retrieval latency (Redis SLOWLOG)
   Threshold: >5ms → Redis memory pressure or hot key
   Fix: increase replicas, shard hot keys

2. Model inference latency (model server metrics)
   Threshold: >20ms for tabular, >50ms for embedding
   Check: batch size, GPU utilization (should be >70%)
   Fix: dynamic batching, GPU upgrade, quantize model

3. Post-processing (business logic, ranking)
   Profiler: add span timing around each stage
   Fix: cache deterministic transformations

4. Network (inter-service calls)
   Check: p99 vs p50 divergence → tail latency issue
   Fix: timeout + retry with jitter, circuit breaker
```

**Latency budget (100ms total example):**
| Stage | Budget | Alert threshold |
|---|---|---|
| Feature store read | 5ms | >10ms |
| Model inference | 30ms | >50ms |
| ANN retrieval | 5ms | >15ms |
| Re-ranking | 10ms | >20ms |
| Network + serialization | 10ms | >20ms |
| Buffer | 40ms | — |

---

## 3. Training-Serving Skew

Training-serving skew is the **#1 silent killer** in production ML. Model validates offline but fails online.

```
Offline AUC good but online CTR poor
              │
              ▼
Compare feature distributions:
  - Training set feature stats (μ, σ, p50, p99)
  - Serving feature stats (same features, live traffic)
              │
         ┌────┴────┐
       Match       Mismatch
         │             │
         ▼             ▼
     Different      Feature
     population     pipeline bug
     or time            │
         │             ▼
         ▼         1. Data type cast difference
     Check            (float32 train vs float64 serve)
     label         2. Aggregation window difference
     quality          (7d in train vs 24h in serve)
                   3. Missing features filled differently
                      (mean vs 0 vs None)
                   4. Feature normalization mismatch
                      (scaler fit on old data)
```

**Skew detection toolkit:**

```python
import pandas as pd
from scipy.stats import ks_2samp

def detect_skew(train_features: pd.DataFrame, 
                serve_features: pd.DataFrame,
                threshold: float = 0.1) -> dict:
    """Compare feature distributions between train and serve."""
    results = {}
    for col in train_features.columns:
        stat, p_val = ks_2samp(
            train_features[col].dropna(),
            serve_features[col].dropna()
        )
        results[col] = {
            'ks_stat': stat,
            'p_value': p_val,
            'skewed': stat > threshold
        }
    return results

# Key: run this on SHADOW traffic before full rollout
# Log feature vectors at serving time → compare to training distribution
```

**Root causes ranked by frequency:**

| Rank | Cause | Detection | Fix |
|---|---|---|---|
| 1 | Feature aggregation window mismatch | Compare window definitions | Align to same window |
| 2 | Null handling difference | Check null rate at serve | Match train preprocessing |
| 3 | Stale normalization stats | Timestamp on scaler artifact | Versioned preprocessing pipeline |
| 4 | Different feature source | Data lineage audit | Single source of truth |
| 5 | Label leakage in training | Point-in-time check | Temporal feature join |

**Prevention:**
```python
# Use same feature computation code for train and serve
# GOOD: Feature defined once in feature store, reused by both
feature_def = FeatureDefinition(
    name="user_7d_click_rate",
    aggregation="mean",
    window_days=7,
    source="clicks_table"
)
# BAD: separate SQL in training notebook, Python in serving code
```

---

## 4. Data Drift vs Concept Drift

```
Model metrics degrading
         │
         ▼
Do you have ground truth labels for recent data?
         │
    ┌────┴────┐
   YES        NO
    │          │
    ▼          ▼
Direct      Proxy
accuracy    signals:
compare     - Score distribution
            - Calibration
            - Upstream feature PSI
```

### Decision Tree

```
                   Has labels?
                 ┌─────┴─────┐
                YES           NO
                 │             │
                 ▼             ▼
           Compute PSI    Score distribution
           on features    shifted?
                 │             │
           ┌────┴───┐    ┌─────┴─────┐
         High     Low   YES           NO
           │       │     │             │
           ▼       ▼     ▼             ▼
        Data     Check  Concept      Wait for
        drift    label  drift        labels,
        likely   dist   likely       watch PSI
```

### Retraining Triggers

| Trigger | Condition | Retraining strategy |
|---|---|---|
| Scheduled | Every 7/14/30 days | Full retrain on rolling window |
| Performance-based | AUC drops >1% | Full retrain or fine-tune |
| Drift-based | PSI >0.25 on top-5 features | Full retrain |
| Volume-based | 1M new labeled samples accumulated | Incremental training |
| Event-based | Major product/market change | Immediate retrain + new features |

**Continuous training vs periodic:**
- **Continuous:** stream training data in real time — high engineering cost, useful for fast-moving patterns (fraud, stock)
- **Periodic (daily/weekly):** simpler, stable — sufficient for most recommendation/ranking

---

## 5. Canonical Interview Q&As

**Q: How do you distinguish data drift from concept drift in production?**  
A: Data drift = P(X) changes; concept drift = P(Y|X) changes. Practically: compute PSI on input features (data drift). If features are stable but model accuracy degrades (requires delayed labels), it's concept drift. Proxy for concept drift without labels: residual distribution shift (KS test on predictions vs actuals from 30d ago). In fraud detection, concept drift is dominant and requires shorter retraining cycles.

**Q: Your model AUC is 0.85 offline but CTR improvement is flat in A/B test. What happened?**  
A: Classic training-serving skew or metric misalignment. Check: (1) feature distribution match between training set and live traffic using KS test, (2) was offline eval done with temporal split? Random splits leak future information, inflating offline AUC, (3) offline AUC measures ranking on historical distribution — if the candidate set differs in production, AUC is not predictive of CTR. (4) Calibration issue: model ranks correctly but predicted probabilities are wrong, affecting downstream threshold logic.

**Q: Walk me through diagnosing a P99 latency spike from 80ms to 350ms in a recommendation API.**  
A: Add distributed tracing if not present (record span per stage). Check each stage: (1) is it all requests or a fraction? Fraction → tail latency, likely outlier inputs or GC pause; (2) is feature store slow? Check Redis hit rate — cache eviction or memory pressure; (3) is model inference slow? Check GPU utilization — low GPU util means request queuing issue; (4) is ANN slow? Probe count too high or index rebuild in progress; (5) Check network — P99 vs P50 divergence signals TCP retransmit or DNS lookup. Fix based on bottleneck found.

**Q: When would you use ADWIN vs PSI for drift detection?**  
A: PSI is batch (offline) — compare feature distributions between a reference window and current window. Good for scheduled monitoring, interpretable thresholds. ADWIN is streaming/online — maintains adaptive window over error rates, automatically shrinks window on detected change. Use ADWIN when you need real-time detection and have labels arriving continuously (click feedback, fraud flags). Use PSI when you only have input features (no labels yet) and run scheduled monitoring jobs.

**Q: How do you prevent silent failures in ML pipelines?**  
A: Defense in depth: (1) schema validation at every pipeline boundary (Great Expectations or custom), (2) feature distribution alerts via PSI on daily batch, (3) prediction distribution monitoring (score histogram) — catches bugs even before labels arrive, (4) shadow mode for new models before full traffic, (5) canary deployment with automated rollback trigger on P99 latency or score distribution shift, (6) data lineage tracking so you can trace any feature value to its source.

## Flashcards

**Sudden accuracy drop vs. gradual drop — what's the first branch in diagnosis?** #flashcard
Sudden (<1h) points to deployment/pipeline issues (rollback, schema change, label pipeline break); gradual (days/weeks) points to data or concept drift.

**What does PSI (Population Stability Index) measure, and what do the thresholds mean?** #flashcard
Shift in a feature's distribution between expected (training) and actual (serving) data. <0.1 no significant drift, 0.1-0.25 moderate (monitor), >0.25 significant (action required).

**Data drift vs. concept drift vs. label drift — what changes in each?** #flashcard
Data drift: P(X) changes, P(Y|X) stable (e.g. demographic shift). Concept drift: P(Y|X) changes, P(X) stable (e.g. fraud pattern changes). Label drift: P(Y) changes (e.g. seasonal CTR drop).

**Why is TTFT compute-bound but TPOT memory-bandwidth-bound in LLM inference?** #flashcard
TTFT (prefill) does one large matmul over the full prompt — arithmetic intensity scales with sequence length. TPOT (decode) loads all model weights per single output token — intensity is ~1 op/byte regardless of batch size, so it's bound by memory bandwidth, not compute.

**Why is training-serving skew called the "#1 silent killer"?** #flashcard
Offline evaluation looks fine (good AUC) but online performance is poor, because it stems from feature computation differences (dtype casts, aggregation window mismatches, different null-fill strategies) between training and serving pipelines — not a model quality problem.

**What's the single biggest prevention for training-serving skew?** #flashcard
Define each feature once in a shared feature store/definition, used identically by both training and serving — never separate SQL-for-training vs. Python-for-serving implementations.

**When would you use ADWIN over PSI for drift detection?** #flashcard
ADWIN is streaming/online and needs continuously arriving labels (e.g. click feedback) — it adaptively shrinks its window on detected change. PSI is batch/offline and only needs input features, suited to scheduled monitoring without labels.

**Continuous vs. periodic retraining — what's the tradeoff?** #flashcard
Continuous (stream training in real time) has high engineering cost but handles fast-moving patterns (fraud, stock). Periodic (daily/weekly) is simpler and sufficient for most recommendation/ranking tasks.
