# Model Governance

> **See also:** [MLOps](mlops.md) | [LLM Training Stability](../05-llms/training-stability.md) | [When Classical ML Wins](../02-classical-ml/when-classical-ml-wins.md)

## Executive Summary

Model governance is the set of processes, tooling, and organizational controls that ensure ML models are built accountably, deployed safely, and retired responsibly. Without governance, teams ship models they can't explain, can't audit, and can't recall when they cause harm. This guide covers the full lifecycle: registry, versioning, approval gates, audit trails, regulatory compliance, incident response, and deployment patterns.

| Governance Layer | Primary risk it mitigates | Tooling |
|-----------------|--------------------------|---------|
| Model registry | Loss of lineage, reproducibility gaps | MLflow, Azure ML Model Registry |
| Versioning strategy | Conflicting deployments, no rollback path | Semantic versioning + hash |
| Approval gates | Unauthorized models reaching production | Azure DevOps gate, manual sign-off |
| Audit trails | Regulatory non-compliance, incident forensics | Azure Monitor, Event Hub |
| GDPR compliance | Right-to-explanation violations | SHAP, model cards |
| Data retention | Privacy violations, stale training data | Azure Policy, lifecycle rules |
| Model cards | Stakeholder misuse, scope creep | Google model card format |
| Incident response | Harm propagation, slow recovery | On-call runbook, canary kills |
| Shadow deployment | Silent production failures | Champion-challenger pattern |

---

## 1. Model Registry: What Metadata to Store

### The Registry as the Source of Truth

A model registry is a centralized catalog where every model that ever touches production — or candidates for production — is registered with sufficient metadata to reproduce, audit, and explain it. The registry is not optional; it is the foundation of all other governance.

**Required metadata for every registered model:**

```yaml
# Example model registry entry
model_name: fraud_detection_v2
model_version: 2.4.1
registration_timestamp: 2026-03-14T09:22:00Z
registered_by: jane.smith@company.com

# Lineage
training_run_id: azureml://runs/abc123def456
training_script: src/train_fraud_model.py
training_script_git_hash: 8f3a2c1
git_branch: main
git_repo: https://github.com/company/ml-platform

# Data
training_dataset:
  name: transactions_2024q4
  version: v3.2
  hash_sha256: 7e4f8a2c...
  row_count: 4821033
  date_range: "2024-10-01 to 2024-12-31"
  location: abfss://data@storageaccount.dfs.core.windows.net/datasets/transactions_2024q4/

validation_dataset:
  name: transactions_2025q1_holdout
  version: v1.0
  hash_sha256: 3b1e9d7a...

# Evaluation scores
metrics:
  auc_roc: 0.9741
  auc_pr: 0.8823
  f1_at_threshold_0.5: 0.7912
  ece: 0.023
  false_positive_rate_at_1pct_fnr: 0.0041
  bias_metrics:
    disparate_impact_ratio: 0.94
    equal_opportunity_difference: 0.012

# Model artifact
artifact_uri: azureml://registries/prod-registry/models/fraud_detection/versions/2.4.1
model_format: mlflow
framework: xgboost==2.0.3
python_version: 3.11.4

# Governance
approval_status: approved
approved_by: risk-model-review-committee
approved_at: 2026-03-15T14:00:00Z
model_card_uri: https://wiki.company.com/models/fraud_detection_v2.4.1
intended_use: Real-time fraud scoring for card-present transactions
out_of_scope: International transactions, business accounts
```

**Dataset hash:** Critical for reproducibility. Without it, "we used the 2024Q4 dataset" is ambiguous — pre- or post-deduplication? which schema version? A verified hash lets an auditor confirm the exact artifact used.

**Azure ML Model Registry:** Provides managed versioning, artifact storage, and metadata attachment. Integrates with Azure DevOps pipelines for automated registration post-training. Model lineage (training run → registered model → deployment) is tracked natively.

---

## 2. Versioning Strategy: Semantic Versioning vs Rolling Hash

### Two Approaches

**Semantic versioning (MAJOR.MINOR.PATCH):**

| Version bump | Trigger |
|-------------|---------|
| MAJOR (1.0.0 → 2.0.0) | Architecture change, feature set change, API-breaking schema change |
| MINOR (1.1.0 → 1.2.0) | Retrain on new data, hyperparameter change, calibration update |
| PATCH (1.1.1 → 1.1.2) | Bugfix in preprocessing, threshold change, label correction |

**Rolling hash versioning:**

Each training run produces a version identifier from: `hash(data_hash + code_commit + hyperparams)`.

```python
import hashlib, json

def compute_model_hash(data_hash, code_commit, hyperparams):
    payload = {"data_hash": data_hash, "code_commit": code_commit, "hyperparams": hyperparams}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

version = f"model-{compute_model_hash('7e4f8a2c', '8f3a2c1', {'n_estimators': 500, 'max_depth': 6})}"
```

**Recommended hybrid:** Semantic versioning for the external-facing API (what consumers call). Rolling hash for internal artifact tracking. Registry maps: `fraud_detection:2.4.1 → model-4a9c3f21b8e7`.

### Version Lifecycle States

```
Draft → In Review → Approved → Deployed (Champion)
                            → Deprecated → Retired
                            → Archived
                 → Rejected
```

- **Draft:** Registered from a training run; not yet reviewed
- **In Review:** Submitted for approval gate; under evaluation
- **Approved:** Cleared for production deployment
- **Deployed:** Currently serving traffic
- **Deprecated:** Still deployed but a newer version exists; teams should migrate
- **Retired:** No longer deployed; artifacts retained per data retention policy
- **Archived:** Past retention period; artifacts deleted or moved to cold storage

**Canary vs blue-green rollout:**
- Canary: gradual percentage rollout (1% → 5% → 20% → 100%). Preferred for models — lets you catch distribution shift before it hits all traffic.
- Blue-green: full swap with instant rollback. Needs 2x capacity but eliminates partial states.

---

## 3. Approval Gates: Who Signs Off Before Production

### The Gate Structure

A governance-compliant model deployment requires multiple sign-offs at different stages:

**Gate 1: Technical review (automated)**
Triggered by CI/CD pipeline on model registration.

- [ ] Model artifact passes integrity check (hash matches registry)
- [ ] All required metadata fields populated
- [ ] Evaluation metrics meet minimum thresholds (AUC > 0.90, ECE < 0.05)
- [ ] Bias metrics within acceptable bounds (disparate impact ratio > 0.80)
- [ ] No training-test leakage detected
- [ ] Model card draft attached

```yaml
# azure-pipelines.yml
- task: AzureMLModelValidation@1
  inputs:
    modelName: "fraud_detection"
    modelVersion: "$(MODEL_VERSION)"
    minAucRoc: 0.90
    maxEce: 0.05
    minDisparateImpact: 0.80
```

**Gate 2: Model owner sign-off**
The data scientist who built the model signs off on model card completeness, known limitations, and training data recency.

**Gate 3: Risk/compliance review (regulated domains)**
Required for credit, insurance, healthcare, employment:
- Fairness analysis across protected groups
- Adverse action notice capability verified (SHAP explanations generated and tested)
- Fits within approved model risk management (MRM) framework

**Gate 4: Business owner sign-off**
Senior stakeholder approves business metrics projections, rollback plan, and operational readiness.

**Azure DevOps implementation:**

```yaml
stages:
- stage: ModelValidation
  jobs:
  - job: AutomatedChecks
    steps:
    - script: python validate_model.py --model-version $(MODEL_VERSION)

- stage: Deployment
  dependsOn: ModelValidation
  condition: succeeded()
  jobs:
  - deployment: ProductionDeploy
    environment: production
    strategy:
      runOnce:
        deploy:
          steps:
          - script: python deploy_model.py --model-version $(MODEL_VERSION)
    approvals:
      - type: requiredTemplate
        approvals:
          - requiredApprovers:
            - "risk-model-review-committee"
            minApprovers: 2
            timeout: 72
```

---

## 4. Audit Trails: What to Log for Regulatory Compliance

### The Audit Trail Requirement

Regulators (OCC, CFPB for banking; EMA, FDA for healthcare; ICO for GDPR) require that you can answer:
- What model made this decision?
- What input did it receive?
- What output did it produce?
- Who approved this model for production?
- What training data was it trained on?
- Has the model changed since the decision was made?

### What to Log at Inference Time

```python
import uuid, datetime, json, hashlib

def log_prediction(model_version, input_features, raw_score, decision, entity_id):
    log_entry = {
        "prediction_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "model_name": "fraud_detection",
        "model_version": model_version,
        "model_artifact_hash": MODEL_ARTIFACT_HASH,  # Loaded at startup
        "entity_id": entity_id,         # Pseudonymous ID, not PII
        "input_feature_hash": hashlib.sha256(
            json.dumps(input_features, sort_keys=True).encode()
        ).hexdigest(),
        "raw_score": float(raw_score),
        "decision": decision,           # "APPROVE" / "REVIEW" / "DECLINE"
        "decision_threshold": THRESHOLD,
        "deployment_environment": "production"
    }
    # Ship to Azure Event Hub → Azure Data Lake for long-term retention
    event_hub_client.send(json.dumps(log_entry))
```

**Do not log raw PII** in the prediction log unless you have explicit consent and a legal basis. Log a pseudonymous entity ID; store the ID-to-PII mapping separately with access controls and independent deletion capability.

### Registry Audit Events

Log these events to an immutable audit log (Azure Monitor + Log Analytics workspace with write-once retention):

- Model registration: who, when, what run ID
- Metadata updates: who changed what, before/after values
- Approval gate events: who approved, what gate, timestamp
- Deployment events: what version, what environment, who triggered
- Rollback events: what triggered, what version rolled back to
- Model retirement: when, who, why

**Retention:** Follow statute-of-limitations-plus-buffer. US credit decisions (FCRA): 25 months required, retain 36 months. GDPR: document retention period in the Record of Processing Activities (ROPA); delete automatically via lifecycle policy.

---

## 5. GDPR and the Right to Explanation

### What Article 22 Requires

GDPR Article 22 restricts automated individual decision-making with legal or similarly significant effects. Articles 13/14 require providing "meaningful information about the logic involved" when such processing is necessary.

The practical minimum: you must explain, in plain language, the primary factors that led to the decision for a specific individual, upon request, within 30 days.

**What GDPR does NOT require:**
- Full model disclosure or algorithm publication
- Mathematical derivation of the model
- A guarantee that the explanation is complete

### How SHAP Satisfies Right-to-Explanation

SHAP (SHapley Additive exPlanations) provides instance-level feature attributions based on Shapley values from cooperative game theory. For tree-based models, TreeSHAP computes exact values efficiently.

```python
import shap, xgboost as xgb, pandas as pd

model = xgb.XGBClassifier()
model.load_model("fraud_detection_v2.4.1.json")
explainer = shap.TreeExplainer(model)

customer_features = pd.DataFrame([{
    "transaction_amount": 850.0,
    "hour_of_day": 3,
    "days_since_last_transaction": 0.2,
    "debt_to_limit_ratio": 0.87,
    "foreign_transaction": 0
}])

shap_values = explainer.shap_values(customer_features)

def generate_adverse_action_notice(shap_values, feature_names, decision):
    contributions = sorted(
        zip(feature_names, shap_values[0]),
        key=lambda x: abs(x[1]), reverse=True
    )
    top_factors = contributions[:3]
    notice = f"Decision: {decision}\n\nPrimary factors:\n"
    for feature, contribution in top_factors:
        direction = "increased" if contribution > 0 else "decreased"
        notice += f"- {feature}: {direction} the risk score\n"
    return notice
```

**Example adverse action notice:**
> Your transaction was flagged for review. Primary factors:
> 1. Transaction time (3 AM): increased fraud risk score
> 2. Transaction amount ($850): increased fraud risk score
> 3. Debt-to-limit ratio (0.87): increased fraud risk score

**SHAP for neural networks:** KernelSHAP (model-agnostic, slow) or DeepSHAP (gradient-based, faster but approximate). For regulated use cases where explanation accuracy is legally material, prefer tree-based models with exact TreeSHAP.

**Right to erasure (Article 17):** If a user requests data deletion, delete their records from the training dataset, retrain or document why the contribution is negligible, and delete their prediction log entries by removing the ID-to-PII mapping (rendering remaining log entries non-identifiable).

---

## 6. Data Retention Policies

### Training Data Retention

**Retention periods by domain:**

| Use case | Jurisdiction | Minimum retention |
|---------|-------------|------------------|
| Credit scoring training data | US (FCRA) | 5 years |
| Insurance underwriting data | EU (Solvency II) | 7 years |
| Medical device training data | US (FDA SaMD) | Useful life + 2 years |
| General ML training data | EU (GDPR) | Purpose-specific; document in ROPA |

**Azure implementation:**

```json
{
  "rules": [{
    "name": "training-data-retention",
    "type": "Lifecycle",
    "definition": {
      "filters": {"blobTypes": ["blockBlob"], "prefixMatch": ["datasets/training/"]},
      "actions": {
        "baseBlob": {
          "tierToCool": {"daysAfterModificationGreaterThan": 90},
          "tierToArchive": {"daysAfterModificationGreaterThan": 365},
          "delete": {"daysAfterModificationGreaterThan": 2190}
        }
      }
    }
  }]
}
```

### Prediction Log Retention

**Practical tiering:**
- **Active (Tier 1, 0–24 months):** Azure Data Lake hot tier. Queryable for audits, monitoring, model retraining.
- **Archive (Tier 2, 24–60 months):** Azure Data Lake cool/archive tier. For regulatory examinations only.
- **Deletion:** Automated via lifecycle policy after retention period.

### Feedback Loop Data

Ground truth labels (confirmed fraud, loan default, diagnosis) that arrive post-prediction must be retained for model validation and retraining. Same or longer retention as prediction logs.

**Data minimization (GDPR Art. 5(1)(e)):** Don't retain raw predictions longer than required. Aggregate or anonymize after the operational need expires.

---

## 7. Model Cards

### What a Model Card Contains

Model cards (Mitchell et al., 2019, Google) are structured documentation communicating model capabilities, limitations, and appropriate use to stakeholders.

```markdown
# Model Card: Fraud Detection v2.4.1

## Model Details
- **Type:** XGBoost binary classifier
- **Version:** 2.4.1
- **Training date:** 2026-03-01
- **Owner:** Risk ML Team (risk-ml@company.com)

## Intended Use
- **Primary:** Real-time fraud scoring for card-present consumer transactions
- **Out of scope:** International transactions; business/commercial cards; CNP transactions

## Training Data
- Transactions from 2024-Q4 (Oct–Dec 2024)
- 4.8M transactions; 0.7% fraud rate
- Geographic scope: United States

## Evaluation Results
| Metric | Value |
|--------|-------|
| AUC-ROC | 0.974 |
| AUC-PR | 0.882 |
| ECE | 0.023 |
| Disparate impact ratio | 0.94 |

## Fairness and Bias
- Evaluated across age bands, zip code income deciles
- No statistically significant disparate impact at 0.05 significance level
- Not evaluated on geographic regions with < 1,000 training transactions

## Limitations and Risks
- Performance may degrade on transaction patterns outside 2024 Q4
- Does not account for seasonal patterns beyond Q4 training window
- Fraud rings adapting behavior may reduce effectiveness

## Ethical Considerations
- Adverse action notices generated via SHAP; provided on request
- Fraud flags trigger human review, not automatic block
```

**Why model cards matter:** Without one, the same model can be deployed in contexts for which it was never evaluated — a US consumer fraud model deployed on international commercial accounts. Model cards make scope explicit and create the reference for post-incident analysis.

**Azure ML:** Model cards can be attached as metadata artifacts to registered models. The Responsible AI dashboard auto-generates portions from evaluation results.

---

## 8. Incident Response: When a Model Causes Harm

### Severity Classification

| Severity | Description | Response time |
|---------|-------------|---------------|
| P1 | Model causing active harm (biased decisions at scale, safety risk) | Rollback within 1 hour |
| P2 | Significant performance degradation (AUC drop > 5%, FPR spike > 2x) | Decision within 4 hours |
| P3 | Drift detected, not yet causing business impact | Remediation within 48 hours |
| P4 | Minor degradation, within acceptable bounds | Next sprint |

### P1 Response Playbook

**Step 1: Rollback (< 15 minutes)**

```bash
az ml online-deployment update \
  --name production-deployment \
  --endpoint-name fraud-scoring-endpoint \
  --traffic "previous-deployment=100 current-deployment=0"
```

**Step 2: Notify stakeholders (< 30 minutes)**
Incident channel, on-call engineer, model owner, compliance officer, business owner.

**Step 3: Scope assessment (< 1 hour)**
How many decisions affected? What time window? Which customer segments? Export prediction logs for the affected period.

**Step 4: Root cause analysis (< 4 hours)**
Was it a data pipeline failure, distribution shift, preprocessing bug, or model degradation? Compare current feature distribution to training baseline.

**Step 5: Remediation**
Fix root cause (patch data pipeline, retrain, recalibrate). Deploy through the full approval gate — incident response does not bypass approval gates for non-emergency fixes.

**Step 6: Post-incident review (< 72 hours)**
Document timeline, root cause, customer impact, remediation steps, and process improvements. File with compliance for regulated domains.

**Regulatory notification:** GDPR Article 33 requires notifying the supervisory authority within 72 hours of a personal data breach. Model-caused harm may qualify if it results from processing of personal data at scale.

**Affected individuals:** In regulated domains, individuals who received adverse decisions from a failing model may be entitled to manual re-review. Maintain a list of affected entity IDs from the prediction logs for this purpose.

---

## 9. Shadow Deployment and Champion-Challenger

### Shadow Deployment

A shadow model receives the same production inputs as the live model but its outputs are not used for decisions — they are logged for comparison.

**Purpose:**
- Validate a new model on live traffic before promoting it
- Detect unexpected behavior on the real input distribution
- Build confidence without risking live decisions

```python
@app.post("/score")
async def score(features: TransactionFeatures):
    # Champion prediction (live)
    champion_score = champion_model.predict_proba(features)

    # Challenger shadow prediction (async, non-blocking)
    asyncio.create_task(
        log_shadow_prediction(challenger_model, features, request_id)
    )

    return {"score": champion_score, "model_version": CHAMPION_VERSION}
```

**Shadow deployment exit criteria:**
- 2 weeks of production traffic logged
- Challenger AUC on live-labeled traffic > Champion by pre-specified margin
- No unexpected score distribution anomalies
- Calibration ECE < 0.05 on recent labeled data

### Champion-Challenger Pattern

Champion-challenger explicitly routes a percentage of live traffic to the challenger and compares business outcomes:

```yaml
# Azure ML traffic split
deployments:
  champion:
    model_version: 2.3.0
    traffic_percentage: 90
  challenger:
    model_version: 2.4.1
    traffic_percentage: 10
```

**Statistical test for promotion:**

```python
from scipy import stats

champion_tp, champion_total = 1820, 20000
challenger_tp, challenger_total = 1950, 20000

stat, p_value = stats.proportions_ztest(
    [challenger_tp, champion_tp],
    [challenger_total, champion_total],
    alternative="larger"
)

if p_value < 0.05:
    print(f"Challenger significantly better (p={p_value:.4f}). Promote.")
else:
    print(f"No significant improvement (p={p_value:.4f}). Keep champion.")
```

**Guardrails during challenger traffic:**
- Challenger cannot exceed 20% traffic until shadow phase passes
- Any P2+ incident on challenger triggers immediate rollback to 0%
- Challenger runs minimum 2 weeks before promotion decision

**When to use shadow vs A/B (champion-challenger):**
- Shadow: when you need to validate on real traffic with zero risk — no user sees challenger outputs; cannot measure business impact.
- Champion-challenger: when you need to measure business impact (conversion, revenue, fraud detection rate). Requires partial real user exposure to the challenger.

---

## 10. Interview Q&A

**Q1: What should a model registry store beyond the model artifact itself?**

> At minimum: training run ID (links to code, hyperparameters, and environment), dataset artifact hash (ensures exact reproducibility), evaluation scores on validation data (AUC, ECE, fairness metrics), approval status and approver identity, model card link, and intended use scope. The dataset hash is the most commonly missed field — "trained on Q4 2024 data" is ambiguous without a verifiable hash of the exact artifact.

**Q2: What is the difference between a model registry and an experiment tracker?**

> An experiment tracker (MLflow, Weights & Biases) logs training runs: metrics, hyperparameters, and artifacts produced during development. A model registry is the promotion system: staging, production, rollback. Registry entries come from experiment tracker artifacts once they've been vetted. The registry answers "what is running in production and why"; the tracker answers "what did we try and how did it perform."

**Q3: What is the right-to-explanation under GDPR and how does SHAP satisfy it?**

> GDPR Article 22 requires meaningful information about the logic of automated decisions that significantly affect individuals — specific to that individual's case, not a general model description. SHAP provides exact Shapley value-based feature attributions for tree-based models. For a credit denial: "Your debt-to-income ratio contributed -0.31 and recent missed payments contributed -0.22 to the decision." This is specific, actionable, legally defensible, and can be delivered within the 30-day response window.

**Q4: Walk me through a shadow deployment and when you'd promote a model.**

> Shadow: route 100% of live traffic to the champion; simultaneously send the same inputs to the challenger and log its predictions without serving them to users. After 2+ weeks, compare on: AUC on labeled outcomes, score distribution shape, calibration ECE. If the challenger passes all checks and is statistically significantly better on the primary metric, promote to champion-challenger at 10% traffic split. After 2 more weeks at 10% with continued improvement and no incidents, promote fully to 100%.

**Q5: What would you log for every production model prediction?**

> Prediction ID (UUID for record linkage), timestamp, model name and version, model artifact hash (immutable proof of which exact model was used), pseudonymous entity ID (not raw PII), input feature hash, raw score, decision, and decision threshold. The artifact hash is the field most teams miss — it allows you to prove in a regulatory audit that the model at decision time was the approved version, not an unauthorized change.

**Q6: What is your incident response playbook when a deployed model starts producing biased outputs?**

> P1: immediate rollback via traffic routing to the previous stable version (< 15 minutes). Notify compliance and business stakeholders. Scope assessment: how many decisions affected, which populations, what time window — export prediction logs. Root cause: data pipeline change, feature drift, or model degradation? All affected individuals may need their adverse decisions manually re-reviewed. File regulatory notification if required (GDPR Art. 33: 72-hour window). Post-incident review within 72 hours, documented for audit trail.

**Q7: What is model risk management and how does it relate to model governance?**

> Model Risk Management (MRM) is the regulatory framework from banking (OCC SR 11-7) for identifying, assessing, and controlling risk from models. Model governance is the tooling and process layer that implements MRM. MRM requires: independent model validation, pre-production approval gates, ongoing monitoring, and periodic revalidation. The model governance infrastructure — registry, approval gates, audit trails — must satisfy MRM requirements for a regulated institution to deploy a model.

**Q8: What does a model card contain and why does it matter?**

> A model card documents: model type and version, intended use and explicitly out-of-scope use cases, training data description and date range, evaluation results including fairness metrics, and known limitations. It matters because the same model artifact can be misapplied: a US consumer fraud model deployed on international commercial transactions will degrade in uncontrolled, unmonitored ways. The model card makes scope explicit at registration time and is the reference document for post-incident analysis and regulatory examination.

**Q9: How do you implement champion-challenger on Azure ML?**

> Define two online deployments under one endpoint: champion at 90% traffic, challenger at 10%. Both log predictions with model version tagged. After 2+ weeks, compare on labeled outcomes using a proportions z-test with a pre-specified significance level and power. Set guardrails: challenger traffic drops to 0% on any P2 incident; challenger cannot exceed 20% until shadow phase is complete. If the challenger is statistically better, update the traffic weights to promote it fully over the following week.

**Q10: How do you handle GDPR data retention for prediction logs?**

> Log decisions using a pseudonymous entity ID, not raw PII. Store the ID-to-PII mapping in a separate system with strict access controls. Document a retention period in the ROPA — e.g., 36 months for credit decisions. Implement Azure Blob Storage lifecycle policies to tier to cool storage at 12 months, archive at 24 months, and delete at 36 months automatically. For right-to-erasure requests: delete the ID-to-PII mapping entry; the remaining log entries are no longer individually identifiable and satisfy Art. 17 compliance.
