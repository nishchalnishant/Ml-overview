# Interview 30 — End-to-End MLOps Pipeline (Condensed)

**Problem:** Take a Data Scientist's Jupyter Notebook model ("In-Game Ad Targeting" — predict which cosmetic a player buys) and design the end-to-end MLOps pipeline to serve it in real-time to millions of DAU with 200ms SLA.

---

## Clarifying Questions to Ask

- "How often does the model need to be retrained?" → Weekly (new cosmetics drop every Tuesday).
- "What's the deployment strategy — can we A/B test vs the old heuristic?" → Yes, propose a safe rollout; a bad model tanks revenue.
- "Batch or real-time inference?" → Real-time, 200ms budget when the store menu opens.
- "How are features reproduced between notebook and production?" → Not defined — candidate should flag Feature Store need.
- "How do we know if the model is failing in prod?" → Not defined — candidate should propose monitoring/drift detection.

---

## Core Architecture

```
Git push (train.py) → CI unit tests
      ↓
Orchestration (Airflow, weekly) → extract features → train XGBoost → eval vs baseline
      ↓
Model Registry (MLflow) → logs params/metrics/artifact, tags "Staging"
      ↓
Shadow deploy → Canary (5%) → full rollout, model served via FastAPI pulling "Production" tag
      ↓
Monitoring (Evidently/Grafana) → logs predictions/features, alerts on drift/latency
```

- **Model choice:** XGBoost — tabular features, fast to train weekly, cheap to serve under 200ms.
- **Key idea to voice:** the point isn't the model, it's making Data→Code→Model→Serving traceable end-to-end (any revenue drop must trace back to a Git commit + data snapshot).

---

## Talking Points That Signal Seniority

- Proactively says notebooks must never touch prod — refactor to modular `.py`, containerize, before merge to main.
- Proposes Shadow Mode first (compare predictions offline, no user impact) before Canary.
- Insists on human-in-the-loop for final "Promote to Prod" — auto-promotion on AUC alone risks leaked/fake metrics.
- Names Feature Store (Feast) explicitly to prevent training-serving skew from copy-pasted notebook feature logic.
- Flags class imbalance (0.1% buyers) unprompted and proposes `scale_pos_weight` + switching to PR-AUC over AUC.
- Mentions schema validation (Pydantic/Great Expectations) on the inference API with a fallback (most popular item) instead of crashing.
- Proposes piping model output into the real A/B testing platform to prove revenue lift, not just AUC, before full rollout.
- Suggests streaming purchase events via Kafka back into the Feature Store to close the feedback loop in near real time.

---

## Top 3 Tradeoffs

- **Notebooks vs scripts:** great for EDA, catastrophic for versioning/testing/deploy — hard rule to refactor before main branch.
- **Auto vs human-gated promotion:** auto-promoting on `AUC > threshold` risks shipping a leaked/broken model; automate to Staging, require human click to Production.
- **Managed platform (Vertex AI/SageMaker) vs self-hosted (Airflow+MLflow+K8s):** managed costs more $ but saves months of DevOps glue-code; self-hosted gives control/portability.

---

## Toughest Follow-ups

**Q: Data scientists want to swap XGBoost for a large PyTorch model — how does serving change?**
FastAPI/Uvicorn is GIL-bound and lacks dynamic batching, bad for DL throughput. Replace with NVIDIA Triton or TorchServe, which batch concurrent requests into single GPU matmuls and run a C++ backend — roughly 10x throughput gain.

**Q: Pipeline throws errors because 99.9% of labels are negative (0.1% buy) and the model just predicts all-zero — how do you automate imbalance handling?**
Dynamically compute `scale_pos_weight = neg/pos` inside `train.py` so false negatives are penalized correctly. Also swap the automated promotion metric from AUC (misleading under heavy imbalance) to PR-AUC.

**Q: Why not just commit the `.pkl` to Git instead of using a Model Registry?**
Git diffs binaries poorly and bloats repo/clone size. A registry stores the binary in cheap object storage (S3) and metadata (hyperparams, metrics, Git hash, lineage) in a queryable DB, plus native Staging/Prod tagging — none of which Git provides.

---

## Biggest Pitfall

Treating "MLOps" as just wrapping the model in a Docker container — no mention of tracking, versioning, drift monitoring, or a safe (shadow/canary) rollout — is the fastest way to fall to No Hire.
