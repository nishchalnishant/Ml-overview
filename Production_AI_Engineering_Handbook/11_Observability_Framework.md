# PART 11: OBSERVABILITY FRAMEWORK

## Goal
To teach candidates how to design comprehensive monitoring systems that detect failures before users do.

## Mental Model
**"Observability is not about collecting all metrics — it's about knowing what questions to ask."**
The three pillars: **Metrics** (what happened), **Logs** (why it happened), **Traces** (where it happened).

---

## 11.1 The Three Pillars of Observability

```text
         METRICS                    LOGS                    TRACES
    (What is happening?)     (Why did it happen?)    (Where is it slow?)
           │                        │                        │
    Prometheus/Grafana         ELK/CloudWatch           Jaeger/Zipkin
    Business KPIs              Error messages           Request spans
    Model metrics              Debug context            Latency breakdown
```

---

## 11.2 Metrics Hierarchy for ML Systems

### Level 1: Infrastructure Metrics
| Metric | Alert Threshold | Tool |
| :--- | :--- | :--- |
| CPU utilization | > 80% for 5 min | Prometheus |
| GPU utilization | < 30% (waste) or > 95% (saturation) | DCGM Exporter |
| Memory usage | > 85% | Prometheus |
| Disk I/O | > 90% capacity | Prometheus |
| Network bandwidth | > 80% | Prometheus |

### Level 2: Service Metrics
| Metric | Alert Threshold |
| :--- | :--- |
| Request latency P50/P95/P99 | P99 > SLA (e.g., 200ms) |
| Throughput (QPS) | Drop > 20% from baseline |
| Error rate | > 1% |
| Queue depth | Growing without bound |

### Level 3: Model Metrics
| Metric | Alert Threshold |
| :--- | :--- |
| Prediction distribution shift | PSI > 0.2 |
| Feature value drift | KS stat p-value < 0.05 |
| Confidence score distribution | Shift > 10% from baseline |
| Null/missing feature rate | > 5% |

### Level 4: Business Metrics
| Metric | Alert Threshold |
| :--- | :--- |
| CTR / Conversion Rate | Drop > 5% vs. 7-day average |
| User complaint rate | Spike > 2x baseline |
| Revenue impact | Drop > 10% |

---

## 11.3 Drift Detection Framework

### Data Drift (Feature Drift)
```text
Is the input distribution P(X) changing?
├── Numeric features → KS test, Wasserstein distance, PSI.
├── Categorical features → Chi-square test, PSI.
└── Embeddings → Cosine distance from centroid, UMAP visualization.
```

| Metric | Formula | Threshold |
| :--- | :--- | :--- |
| **PSI (Population Stability Index)** | Σ (Actual% - Expected%) × ln(Actual%/Expected%) | < 0.1: stable, 0.1-0.2: monitor, > 0.2: alert |
| **KS Statistic** | Max difference between CDFs | p-value < 0.05: significant drift |
| **Wasserstein** | Earth Mover's Distance | Domain-specific baseline |

### Concept Drift (Model Performance Drift)
```text
Is the relationship P(Y|X) changing?
→ Monitor business KPIs (CTR, revenue, complaints).
→ If KPIs drop but feature distributions are stable → Concept drift.
   Remedy: Retrain model with recent data.

→ If KPIs drop AND features drifted → Data drift causing performance drop.
   Remedy: Fix data pipeline first, then retrain.
```

---

## 11.4 MLflow Integration

### What to Log in Every Training Run
```python
# Key MLflow logging pattern
with mlflow.start_run():
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("data_version", "v2024-01-15")
    mlflow.log_param("git_commit", git_hash)
    
    mlflow.log_metric("val_auc", 0.87)
    mlflow.log_metric("val_f1", 0.72)
    mlflow.log_metric("train_time_sec", 340)
    
    mlflow.log_artifact("feature_importance.png")
    mlflow.sklearn.log_model(model, "model")
```

### Model Registry Stages
```text
None → Staging → Production → Archived
```
Never serve a model directly from an experiment. Always promote through the registry.

---

## 11.5 Alerting Strategy

### Alert Priority Levels
| Level | Condition | Response |
| :--- | :--- | :--- |
| **P0 (Critical)** | Service down, > 5% error rate, P99 > 5x SLA | Page on-call immediately |
| **P1 (High)** | P99 > 2x SLA, PSI > 0.2, KPI drop > 10% | Alert within 15 minutes |
| **P2 (Medium)** | P99 > SLA, PSI 0.1-0.2, KPI drop 5-10% | Alert within 1 hour |
| **P3 (Low)** | Increasing trend, non-urgent degradation | Daily digest |

### Alert Fatigue Prevention
- **Use anomaly-based alerts** (% change from baseline) over absolute thresholds.
- **Aggregate alerts:** Don't alert for each pod individually; alert on the fleet.
- **Require 5-minute sustained breach** before alerting (prevents transient spikes).

---

## 11.6 Distributed Tracing for ML Pipelines

### Trace Span Breakdown
```text
[User Request] ──────────────────────────────── 185ms total
   ├── [API Gateway auth]                        5ms
   ├── [Feature Store lookup]                   15ms
   ├── [Candidate generation]                   20ms
   ├── [Re-ranking model inference]             90ms  ← BOTTLENECK
   ├── [Business rules post-processing]         10ms
   └── [Response serialization]                  5ms
```

Using traces, you immediately identify the re-ranking model as the latency bottleneck.

### Tools
| Tool | Use Case |
| :--- | :--- |
| **Jaeger** | Distributed tracing for microservices |
| **LangSmith** | Agent/LLM call tracing |
| **Phoenix (Arize)** | LLM observability, RAG evaluation |
| **W&B Traces** | ML model tracing integrated with training |

---

## 11.7 Dashboard Design

### Recommended Dashboard Layers
1. **Executive Dashboard:** Business KPIs only. No engineering metrics. CTR, revenue impact.
2. **Service Health Dashboard:** QPS, P99 latency, error rate per service.
3. **Model Health Dashboard:** Prediction distribution, feature drift, confidence scores.
4. **Infra Dashboard:** CPU, GPU, memory, network per pod/node.

---

## 11.8 Incident Response Playbook

```text
[Alert fires]
     │
     ▼
Step 1: ACKNOWLEDGE — Assign an on-call owner. ETA to first response: 5 minutes.
     │
     ▼
Step 2: TRIAGE — Is this a model issue, data issue, or infra issue?
        Check: Error rate, latency metrics, drift metrics. (10 minutes)
     │
     ▼
Step 3: MITIGATE — If known fix available: rollback. If unknown: enable fallback. (15 minutes)
     │
     ▼
Step 4: INVESTIGATE — Root cause analysis using traces and logs.
     │
     ▼
Step 5: RESOLVE — Deploy fix, verify business metrics recover.
     │
     ▼
Step 6: POSTMORTEM — Document root cause, timeline, and preventive measures.
```

---

## Engineering Checklist

- [ ] Are all three pillars (metrics, logs, traces) instrumented?
- [ ] Is there a Grafana dashboard for every deployed model?
- [ ] Is feature drift monitored in production?
- [ ] Are business KPI dashboards separated from engineering dashboards?
- [ ] Is there an automated retrain trigger when drift is detected?
- [ ] Is there a documented incident response playbook?

## Interview Follow-up Questions & Best Answers

**Q: "How would you know within 5 minutes that a new model deployment is causing problems?"**
*Best Answer:* "I set up progressive monitoring with pre-defined rollback triggers. Immediately after deployment, I monitor: 
1. Error rate — any spike above 0.5% triggers automatic rollback (circuit breaker). 
2. P99 latency — if it exceeds 1.5x the SLA, rollback. 
3. Prediction distribution — if the output score distribution shifts significantly from the old model's baseline distribution (measured in real-time), page on-call. 
I use Grafana alerts with 1-minute evaluation windows during active deployment, then relax to 5-minute windows once stable. This is why Canary deployments are critical — I can catch these failures on 1% of traffic before it affects everyone."
