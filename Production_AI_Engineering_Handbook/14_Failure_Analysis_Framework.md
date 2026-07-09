# PART 14: FAILURE ANALYSIS FRAMEWORK

## Goal
To teach candidates how to systematically debug AI system failures in production, using structured root cause analysis rather than random guessing.

## Mental Model
**"Debug from the outside in: business → service → model → data → infra."**
Always triage from the highest observable layer downward. Don't assume it's the model — it's usually the data pipeline.

---

## 14.1 The Triage Framework

```text
STEP 1: DETECT ──── Is there an active alert? What metric triggered it?
STEP 2: SCOPE ────── Is this affecting all users or a subset? All regions or one?
STEP 3: ISOLATE ─── Which component changed recently? (Deployment? Data? Config?)
STEP 4: DIAGNOSE ── Use logs/traces to pinpoint the exact failure.
STEP 5: FIX ──────── Rollback (fast) or Hotfix (if rollback isn't possible).
STEP 6: VERIFY ───── Confirm business metric recovers.
STEP 7: POSTMORTEM ─ Document. Prevent recurrence.
```

---

## 14.2 Bad Predictions

### Diagnosis Tree
```text
Model predictions are wrong?
├── WHICH types of predictions are wrong?
│   └── Specific user segment / specific input type?
│   └── Check for data distribution shift (PSI, KS test).
│
├── Did a recent deployment change the model?
│   └── YES → Rollback the model to the previous version.
│
├── Did upstream data change?
│   └── YES → Fix the data pipeline. Retrain or fine-tune on recent data.
│
├── Is there a bug in feature engineering?
│   └── Compare features at training time vs. serving time.
│   └── Training-serving skew is #1 cause of bad predictions.
│
└── Has the world changed (concept drift)?
    └── Model is stale. Trigger retraining with recent labeled data.
```

### Training-Serving Skew Checklist
- [ ] Is the same scaler/imputer used in training and serving?
- [ ] Are feature names and dtypes identical?
- [ ] Are categorical encodings identical?
- [ ] Are time-based features computed with the same timezone?
- [ ] Are null/missing values handled identically?

---

## 14.3 High Latency

### Diagnosis Tree
```text
P99 latency is too high?
│
├── Check which service is slow (traces).
│
├── Is it the MODEL INFERENCE?
│   ├── GPU memory pressure? → Reduce batch size or model size.
│   ├── CPU-bound data loading? → Prefetch DataLoader, move preprocessing to GPU.
│   ├── Model too large for single GPU? → Quantize or use smaller model.
│   └── Network overhead? → Switch to gRPC, enable HTTP/2.
│
├── Is it the FEATURE RETRIEVAL?
│   ├── Feature Store timeout? → Check Redis latency, connection pool exhaustion.
│   └── Too many feature lookups? → Batch feature queries.
│
├── Is it the VECTOR SEARCH?
│   ├── Index not loaded in memory? → Ensure index is warmed up.
│   └── Search space too large? → Add metadata pre-filtering.
│
└── Is it the LLM API?
    ├── Rate limit hit? → Add retry with backoff, or use secondary provider.
    └── Long context? → Compress context, reduce top-K retrieved chunks.
```

---

## 14.4 Memory Leaks

### Common Sources in ML Systems
```text
1. DATALOADER not releasing batches → Use `del batch` + `torch.cuda.empty_cache()`.
2. GRADIENTS accumulated without optimizer.zero_grad() → Add before each backward pass.
3. TENSOR kept on GPU unnecessarily → Call .detach() before storing in lists.
4. REFERENCES to model held in closures → Scope management.
5. REDIS connection pool not closed → Use context managers.
```

### Detection
```text
Monitoring:
├── Memory usage grows monotonically over time → Leak.
├── OOM errors after N hours of uptime → Leak.
└── Profile with: torch.profiler, memory_profiler (Python), nvidia-smi.
```

---

## 14.5 GPU Out-of-Memory (OOM)

### Diagnosis Checklist
```text
GPU OOM error?
├── Is the BATCH SIZE too large? → Reduce by half.
├── Are GRADIENTS accumulating? → Add optimizer.zero_grad().
├── Is the MODEL too large for one GPU? → Model parallelism or gradient checkpointing.
├── Are TENSORS not being freed? → Use torch.no_grad() for inference.
└── Are CUDA graphs causing large memory reservations? → Profile allocations.
```

### Memory Reduction Strategies (Priority Order)
1. `torch.no_grad()` for inference — eliminates gradient storage.
2. Gradient checkpointing (recompute activations during backward) — trades compute for memory.
3. Mixed precision (FP16/BF16) — 2x memory reduction.
4. Gradient accumulation — simulate large batch on small GPU memory.
5. Quantization (INT8/INT4) — 4–8x model size reduction.

---

## 14.6 Hallucinations (LLM / RAG)

### Diagnosis Tree
```text
LLM is hallucinating?
│
├── Is the RETRIEVAL wrong (RAG)?
│   ├── Wrong documents retrieved? → Improve chunking, switch to hybrid search.
│   └── Answer not in documents? → Add "I don't know" guardrail.
│
├── Is the PROMPT ambiguous?
│   └── Improve system prompt. Add examples (few-shot).
│
├── Is the MODEL inherently hallucinating on this topic?
│   ├── Knowledge cutoff issue? → Ground with RAG.
│   └── Model size too small? → Upgrade to larger model.
│
└── Is CALIBRATION the issue (overconfident)?
    └── Add self-consistency check. Generate 3 answers; flag divergence.
```

---

## 14.7 Agent Failures

### Common Agent Failure Modes
| Failure | Symptom | Fix |
| :--- | :--- | :--- |
| **Infinite loop** | Agent calls same tool repeatedly | Max step limit + loop detection |
| **Wrong tool selection** | Agent picks wrong tool for task | Improve tool descriptions |
| **Hallucinated tool args** | Agent generates invalid arguments | Add argument schema validation |
| **Context overflow** | Agent loses early context in long runs | Summarize context mid-run |
| **Tool timeout** | External API never responds | Timeouts + retry limits |
| **Self-contradiction** | Agent changes its own instructions | Structured state management |

---

## 14.8 RAG Pipeline Failures

### Failure Mode Diagnosis
```text
RAG returning bad answers?
├── Low retrieval score? → Chunks not semantically matching query.
│   └── Fix: Rewrite query (HyDE), use hybrid search, re-check chunk size.
├── Correct chunks retrieved but wrong answer? → Generation failure.
│   └── Fix: Improve system prompt, upgrade LLM, add few-shot examples.
├── Correct answer in a chunk that wasn't retrieved? → Retrieval failure.
│   └── Fix: Increase top-K temporarily, check chunking strategy.
└── Outdated information? → Knowledge base staleness.
    └── Fix: Add ingestion pipeline trigger on document updates.
```

---

## 14.9 Data Corruption

### Detection
```text
Data corruption symptoms:
├── Sudden spike in null/NaN feature values.
├── Categorical values outside the expected vocabulary.
├── Numeric values outside expected ranges (negative ages, future dates).
├── Schema mismatch errors in the feature pipeline.
└── Training/serving skew spike in monitoring.
```

### Response
```text
Step 1: STOP serving predictions from corrupted data. Use cached predictions or fallback.
Step 2: IDENTIFY the source (which upstream system changed?).
Step 3: QUARANTINE corrupted data (don't let it reach training).
Step 4: BACKFILL correct data from source or backups.
Step 5: RETRAIN if corrupted data already entered training.
```

---

## 14.10 Deployment Failures

### Rollback Decision Tree
```text
New model deployed. Problem detected?
├── Error rate spiked? → Rollback immediately (< 5 minutes).
├── Latency SLA breached? → Rollback.
├── Business metric degraded > 5%? → Rollback.
├── Prediction distribution shifted significantly? → Rollback.
└── Everything looks fine but you're unsure? → Pause canary at 5%. Monitor longer.

Rollback procedure:
Step 1: Route all traffic back to old model (Blue-Green: instant switch).
Step 2: Alert on-call team.
Step 3: Investigate new model offline.
Step 4: Fix issue, validate in staging, re-deploy with wider monitoring.
```

---

## Engineering Checklist

- [ ] Is there a defined rollback procedure documented before every deployment?
- [ ] Is there a root cause analysis template for all P0/P1 incidents?
- [ ] Are training-serving skew metrics monitored in production?
- [ ] Is there a maximum step/cost limit on all agent runs?
- [ ] Is there a circuit breaker on every external dependency?
- [ ] Is there a fallback prediction strategy for all models?

## Interview Follow-up Questions & Best Answers

**Q: "You get a page at 2am: the recommendation model's CTR dropped 30% in the last hour. Walk me through your response."**
*Best Answer:* "I follow a structured triage. 
**First 5 minutes (detect + scope):** Check Grafana — is this all users or a specific cohort? All regions or one? Did anything deploy in the last few hours?
**Minutes 5-15 (isolate):** If a model deployment happened → rollback immediately (don't wait). If no deployment, check upstream: did the feature store break? Are feature distributions shifted?
**Minutes 15-30 (diagnose):** Examine traces for the failed requests. Check if null rate for key features spiked (data pipeline failure). Check if the model's confidence score distribution shifted (data drift vs. code bug).
**Minute 30 (mitigate):** If root cause identified, fix it. If not, switch to fallback (popular items, rule-based recommendations) to restore CTR while investigation continues.
**Post-incident:** Document everything. Add a monitoring alert that would have caught this 20 minutes earlier."
