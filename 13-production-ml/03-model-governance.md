---
module: Production Ml
topic: Model Governance
subtopic: ""
status: unread
tags: [productionml, ml, model-governance]
---
# Model Governance

> **See also:** [MLOps](01-mlops.md) | [Deployment Patterns](02-deployment-patterns.md) | [LLM Training Stability](../10-llms/05-training-stability.md) | [Model Registry & Versioning (tooling deep-dive)](09-model-registry-versioning.md) — this file covers governance/compliance framing (audit trails, approval gates, model cards); that one covers hands-on registry tooling and canary rollout mechanics.

## 1. Model Registry and Metadata

### The Problem

A regulator asks: "What model made that credit decision on March 14th at 2:47 PM, what data trained it, and who approved it?" Without a registry, the answer requires archaeology through Slack threads, shared drives, and engineer memory. The company fails the audit.

### The Core Insight

Every model artifact needs a single authoritative record that answers: what it is, what trained it, who approved it, and what it has done.

### The Mechanics

**Registry metadata schema:**

```yaml
model:
  name: credit_risk_v2
  version: 2.4.1
  stage: deployed          # Draft | InReview | Approved | Deployed | Deprecated | Retired | Archived

training_lineage:
  dataset_version: s3://ml-data/credit/v47  # DVC hash: sha256:a3f9...
  training_run_id: mlflow://run/8f3a2c1
  training_date: "2024-03-01"
  features_used:
    - payment_history_12m
    - debt_to_income_ratio
    - credit_utilization
  features_excluded:           # document what you deliberately excluded and why
    - zip_code: "proxy for race under ECOA"
    - age: "ADEA protected characteristic"

performance:
  offline:
    auc_roc: 0.847
    pr_auc: 0.621
    ks_statistic: 0.412
    test_set: credit_holdout_2023q4
  fairness:
    demographic_parity_difference: 0.023   # threshold: <0.05
    equalized_odds_difference: 0.031
    protected_attributes_tested: [race_proxy, gender_proxy, age_group]
  calibration:
    brier_score: 0.089
    expected_calibration_error: 0.012

governance:
  model_owner: alice@company.com
  business_owner: lending_product_team
  approval_chain:
    - gate: automated_validation
      passed: true
      timestamp: "2024-03-05T14:22:00Z"
    - gate: model_owner_signoff
      approver: alice@company.com
      timestamp: "2024-03-06T10:15:00Z"
    - gate: risk_compliance
      approver: compliance_team
      notes: "ECOA review passed. GDPR Article 22 documentation attached."
      timestamp: "2024-03-07T16:30:00Z"
    - gate: business_signoff
      approver: vp_lending
      timestamp: "2024-03-08T09:00:00Z"

compliance_flags:
  gdpr_article22_applies: true           # automated individual decision
  adverse_action_notice_required: true
  right_to_erasure_supported: true
  data_retention_years: 7                # FCRA requirement
  explainability_method: TreeSHAP
  model_card_url: s3://governance/cards/credit_risk_v2.4.1.md
```

**Registry operations:**

```python
import mlflow

client = mlflow.MlflowClient()

# Register artifact from training run
model_uri = f"runs:/{run_id}/model"
mv = client.create_model_version(
    name="credit_risk",
    source=model_uri,
    run_id=run_id,
    tags={"dataset_hash": dataset_sha256, "training_date": "2024-03-01"}
)

# Lifecycle transitions require approval evidence
client.transition_model_version_stage(
    name="credit_risk",
    version=mv.version,
    stage="Staging",
    archive_existing_versions=False
)

# Add approval tags
client.set_model_version_tag(
    name="credit_risk",
    version=mv.version,
    key="compliance_approved_by",
    value="compliance_team:2024-03-07"
)
```

### What Breaks

**No dataset versioning:** model performance degrades, team cannot reproduce the trained artifact, audit fails.

**Missing feature exclusion documentation:** regulator infers the excluded features were used, which is worse than acknowledging the exclusion decision.

**Registry without enforcement:** teams bypass it under time pressure; the registry has partial data and provides false confidence.

---

## 2. Versioning Strategy

### The Problem

You deploy a small bug fix to a model. A downstream team's integration breaks because they expected the output format to be unchanged. A "compatibility-breaking vs non-breaking change" distinction would have warned them.

### The Core Insight

Semantic versioning communicates the impact of a change, not just the sequence.

### The Mechanics

**Version: MAJOR.MINOR.PATCH**

| Increment | When | Example |
|-----------|------|---------|
| **MAJOR** | Output schema changes, prediction semantics change, incompatible API change | 1.x.x → 2.0.0 |
| **MINOR** | New features added, backward compatible. Model retrained on more data with same interface | 2.3.x → 2.4.0 |
| **PATCH** | Bug fixes, threshold adjustments, documentation | 2.4.0 → 2.4.1 |

**Lifecycle states and transition rules:**

```
Draft
  │  automated tests pass
  ▼
In Review
  │  model owner approves
  ▼
Approved
  │  compliance signs off (if required)
  ▼
Deployed
  │  canary → full rollout
  ▼
Deprecated        ← new version deployed to replace this one
  │  30-day sunset period
  ▼
Retired           ← no longer serving traffic
  │  after retention period
  ▼
Archived          ← artifacts preserved, model not executable
```

**What triggers a MAJOR version bump (checklist):**

- Output type changes (probability → binary label)
- Score range changes (0–1 → 0–100) even with rescaling
- New required input features (callers must update)
- Prediction semantics change (lower score now means higher risk)
- Model framework change that alters numerical precision

### What Breaks

**Skipping MAJOR bump for breaking changes:** downstream teams get silent failures because their integration was built against the old semantics.

**Not archiving retired models:** audit requires the ability to recreate the decision environment at a point in time. "We deleted it" fails a financial audit.

**Allowing Draft → Deployed transitions without gates:** removes the governance guarantee.

---

## 3. Approval Gates

### The Problem

A model goes to production with a demographic bias that was visible in the metrics but no one was required to look. The problem was not a technical failure — it was a process failure.

### The Core Insight

Each gate tests a different failure mode. Automated gates test objective criteria. Human gates test judgment that cannot yet be automated.

### Gate 1: Automated Validation

Runs in CI/CD. Hard failures block the pipeline.

```python
class ModelValidationGate:
    def __init__(self, challenger, champion, test_data, thresholds):
        self.challenger = challenger
        self.champion = champion
        self.test_data = test_data
        self.thresholds = thresholds

    def run(self) -> dict:
        results = {}

        # 1. Performance: challenger must match champion
        champ_auc = self._auc(self.champion)
        chal_auc = self._auc(self.challenger)
        results["performance_regression"] = (champ_auc - chal_auc) > self.thresholds["max_auc_drop"]

        # 2. Calibration
        ece = self._expected_calibration_error(self.challenger)
        results["calibration_failed"] = ece > self.thresholds["max_ece"]

        # 3. Population Stability Index on input features
        psi = self._psi(self.test_data)
        results["input_drift"] = psi > self.thresholds["max_psi"]  # typically 0.2

        # 4. Fairness
        dpd = self._demographic_parity_difference(self.challenger)
        results["fairness_violation"] = dpd > self.thresholds["max_dpd"]  # typically 0.05

        results["passed"] = not any(results.values())
        return results

    def _auc(self, model):
        from sklearn.metrics import roc_auc_score
        scores = model.predict_proba(self.test_data.X)[:, 1]
        return roc_auc_score(self.test_data.y, scores)

    def _expected_calibration_error(self, model, n_bins=10):
        import numpy as np
        scores = model.predict_proba(self.test_data.X)[:, 1]
        y = self.test_data.y
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (scores >= bin_edges[i]) & (scores < bin_edges[i+1])
            if mask.sum() == 0:
                continue
            bin_acc = y[mask].mean()
            bin_conf = scores[mask].mean()
            ece += mask.mean() * abs(bin_acc - bin_conf)
        return ece

    def _psi(self, data, n_bins=10):
        import numpy as np
        # Compare feature distributions against training baseline
        # Returns max PSI across all features
        psi_values = []
        for col in data.feature_columns:
            expected = self._load_training_distribution(col)
            actual, _ = np.histogram(data.X[col], bins=n_bins, density=True)
            actual = actual / actual.sum()
            expected = expected / expected.sum()
            psi = np.sum((actual - expected) * np.log((actual + 1e-8) / (expected + 1e-8)))
            psi_values.append(psi)
        return max(psi_values)
```

### Gate 2: Model Owner Sign-off

The model owner (the engineer who built the model) attests:
- Training data is appropriate for the use case
- Known failure modes are documented
- Feature importance analysis was reviewed
- No target leakage in feature set

Not delegatable. Requires personal attestation.

### Gate 3: Risk and Compliance Review

Required for: credit, insurance, medical, hiring, any GDPR Article 22 use case.

Checklist:
- [ ] Protected attributes reviewed (direct and proxy)
- [ ] Adverse action notice mechanism tested
- [ ] GDPR Article 22 documentation prepared if automated individual decision
- [ ] ECOA/FHA/FCRA compliance review (for credit/housing models)
- [ ] Data retention policy attached to model version
- [ ] Right to erasure impact assessed

### Gate 4: Business Sign-off

Business owner confirms:
- Model is solving the intended business problem
- Performance thresholds meet business requirements (not just statistical significance)
- Rollback decision criteria are agreed in advance
- On-call runbook exists

### What Breaks

**Automated gate without a human gate:** catches metric failures, misses judgment failures (wrong proxy features, wrong business framing).

**Compliance gate as a rubber stamp:** if reviewers are not empowered to block, the gate provides legal exposure without protection.

**No pre-agreed rollback criteria at Gate 4:** when production problems emerge, the rollback decision becomes a political negotiation under pressure.

---

## 4. Audit Trails

### The Problem

A customer sues over a rejected loan. Discovery requires producing the exact model score, input features, and explanation for that decision — 18 months later. The data has been mutated by subsequent updates and the exact model version is unclear.

### The Core Insight

Every inference event is a legal record. Log it with the immutability and retrievability of a financial transaction.

### The Mechanics

**Inference event schema:**

```python
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class InferenceEvent:
    # Identification
    event_id: str              # UUID, globally unique
    model_name: str
    model_version: str
    timestamp_utc: float

    # Input (pseudonymized)
    entity_id_hash: str        # SHA-256 of entity_id — not the raw ID
    feature_hash: str          # SHA-256 of serialized feature vector
    feature_snapshot: Dict     # actual features — stored separately, encrypted

    # Output
    prediction: Any
    confidence: float
    explanation_top_factors: list  # SHAP top-k factors, not raw values

    # Metadata
    request_latency_ms: float
    serving_node: str

def log_inference_event(
    model_name: str,
    model_version: str,
    entity_id: str,
    features: Dict,
    prediction: Any,
    confidence: float,
    shap_values: Dict,
    latency_ms: float
) -> str:
    import uuid

    # Pseudonymize: log hash of entity ID, not the ID itself
    entity_id_hash = hashlib.sha256(entity_id.encode()).hexdigest()

    # Hash the feature vector for tamper detection
    feature_json = json.dumps(features, sort_keys=True)
    feature_hash = hashlib.sha256(feature_json.encode()).hexdigest()

    # Top-k SHAP factors (direction only, not raw customer data)
    top_factors = sorted(
        [{"feature": k, "direction": "positive" if v > 0 else "negative"}
         for k, v in shap_values.items()],
        key=lambda x: abs(shap_values[x["feature"]]),
        reverse=True
    )[:5]

    event = InferenceEvent(
        event_id=str(uuid.uuid4()),
        model_name=model_name,
        model_version=model_version,
        timestamp_utc=time.time(),
        entity_id_hash=entity_id_hash,
        feature_hash=feature_hash,
        feature_snapshot=features,  # encrypted at rest in audit store
        prediction=prediction,
        confidence=confidence,
        explanation_top_factors=top_factors,
        request_latency_ms=latency_ms,
        serving_node=os.environ.get("HOSTNAME", "unknown")
    )

    # Write to immutable audit store (append-only S3 bucket with object lock,
    # or audit-specific table with no DELETE privilege)
    audit_store.append(asdict(event))

    return event.event_id
```

**Audit store requirements:**

- Append-only (no DELETE, no UPDATE on audit rows)
- Encrypted at rest (AES-256 minimum)
- Accessible only to audit/compliance role (not the serving application)
- Retention enforced by the data retention schedule, not manual deletion

### What Breaks

**Storing raw PII in audit logs:** creates a parallel PII store, multiplying GDPR deletion obligations and breach surface.

**Audit logs in the same database as operational data:** a database migration or "cleanup" script can accidentally destroy audit records.

**Not storing the feature snapshot:** 18 months later, you cannot reconstruct the explanation without knowing what features were used at that exact moment.

---

## 5. Regulatory Compliance

> This is the canonical treatment of GDPR compliance mechanics (Article 22 explanation generation, Article 17 erasure pipeline, Article 33 breach response) as production capabilities. For the machine-unlearning algorithms behind Article 17 erasure (exact retraining vs. influence-function approximation vs. SISA sharded training), see [privacy-and-fairness.md §3](../14-responsible-ai/01-privacy-and-fairness.md).

### The Problem

A model automatically rejects loan applications. EU law says individuals have the right not to be subject to solely automated decisions that significantly affect them (GDPR Article 22). The model produces no explanation and offers no human review. Each automated decision is a violation.

### The Core Insight

Compliance is not documentation — it is a capability. The system must be able to produce explanations, process erasure requests, and detect breaches. "We have a policy" is not a substitute for "the system does this."

### GDPR Article 22: Right to Explanation

Applies when: automated decision has legal or similarly significant effect on an individual.

Required capability: provide a meaningful explanation of the decision logic.

```python
import shap
import numpy as np

def generate_adverse_action_notice(
    model,
    features: dict,
    prediction: int,      # 0 = denied, 1 = approved
    entity_id: str
) -> dict:
    """
    Generate a GDPR Article 22 / ECOA adverse action notice.
    Returns human-readable top reasons for a denial.
    """
    if prediction == 1:
        return {"action": "approved", "reasons": []}

    # Compute SHAP values (TreeSHAP for tree models — exact, not approximate)
    explainer = shap.TreeExplainer(model)
    feature_array = np.array([list(features.values())])
    shap_values = explainer.shap_values(feature_array)

    # For binary classification, shap_values[1] = positive class (approved)
    # Negative SHAP for approved class = factors that reduced approval probability
    if isinstance(shap_values, list):
        denial_shap = {
            name: -val  # negate: high = strong denial factor
            for name, val in zip(features.keys(), shap_values[1][0])
        }
    else:
        denial_shap = {
            name: -val
            for name, val in zip(features.keys(), shap_values[0])
        }

    # Top 4 denial factors (ECOA Regulation B requires specific reasons)
    top_reasons = sorted(
        denial_shap.items(),
        key=lambda x: x[1],
        reverse=True
    )[:4]

    # Map feature names to human-readable reasons
    reason_map = {
        "payment_history_12m": "Insufficient recent payment history",
        "debt_to_income_ratio": "Debt-to-income ratio too high",
        "credit_utilization": "Credit utilization rate too high",
        "length_of_credit_history": "Credit history too short",
        "recent_inquiries": "Too many recent credit inquiries",
    }

    return {
        "action": "denied",
        "model_version": model.__version__,
        "decision_timestamp": time.time(),
        "reasons": [
            reason_map.get(feature, f"Factor: {feature}")
            for feature, _ in top_reasons
            if _ > 0  # only include factors that actually increased denial probability
        ],
        "appeal_instructions": "You may request human review within 30 days by contacting credit@company.com",
        "audit_event_id": log_inference_event(...)  # link to immutable audit record
    }
```

### GDPR Article 17: Right to Erasure

When a subject requests deletion, training data erasure is not sufficient — the model has encoded the training data in its weights.

**Erasure process:**

1. Delete raw training data records for the subject
2. Mark model versions trained on data including the subject as "erasure-affected"
3. If model was used for active decisions on the subject: schedule retraining excluding the subject's data
4. Retraining is not always required immediately — assess whether the subject's data materially influences the model

**Practical implementation:**

```python
class ErasureManager:
    def process_erasure_request(self, subject_id: str) -> dict:
        report = {}

        # 1. Delete raw data
        rows_deleted = self.data_store.delete_by_subject(subject_id)
        report["training_data_deleted"] = rows_deleted

        # 2. Find affected model versions
        affected_versions = self.registry.find_models_trained_on_subject(subject_id)
        report["affected_model_versions"] = affected_versions

        # 3. For each affected version, estimate influence
        for version in affected_versions:
            influence = self._estimate_influence(version, subject_id)
            if influence > self.MATERIAL_INFLUENCE_THRESHOLD:
                self.registry.flag_for_retraining(
                    version,
                    reason=f"erasure:{subject_id}",
                    deadline_days=30
                )

        # 4. Anonymize inference audit logs (cannot delete — retention required)
        # The entity_id_hash already pseudonymizes. Cannot further erase without
        # violating financial audit requirements. Document this tension.
        report["audit_logs"] = "pseudonymized at inference time; cannot erase without audit violation"

        return report
```

### GDPR Article 33: Breach Notification

72-hour clock to notify supervisory authority when a personal data breach occurs.

**ML-specific breach scenarios:**

- Model inversion attack recovers training data from model weights
- Inference audit logs exposed (contains feature snapshots with PII)
- Feature store breach (contains individual-level features)
- Membership inference attack demonstrates training data presence

**Breach response:**

```python
class BreachResponseProtocol:
    def initiate(self, breach_type: str, scope_estimate: int, discovery_time: float):
        """
        breach_type: "model_inversion" | "audit_log_exposure" | "feature_store" | ...
        scope_estimate: estimated number of affected subjects
        discovery_time: unix timestamp of breach discovery
        """
        elapsed_hours = (time.time() - discovery_time) / 3600
        hours_remaining = 72 - elapsed_hours

        steps = {
            "immediate": [
                "Isolate affected model/store (revoke serving credentials)",
                "Preserve forensic evidence (do not delete logs)",
                "Engage DPO and legal",
            ],
            "within_24h": [
                "Determine scope: how many subjects, what data categories",
                "Identify breach vector",
                "Draft supervisory authority notification",
            ],
            "within_72h": [
                f"Submit Article 33 notification ({hours_remaining:.1f}h remaining)",
                "If high risk to subjects: prepare Article 34 subject notification",
            ]
        }

        # Log to immutable incident record
        self.incident_store.create(
            breach_type=breach_type,
            discovery_time=discovery_time,
            scope_estimate=scope_estimate,
            notification_deadline=discovery_time + 72 * 3600
        )

        return steps
```

### Data Retention Schedule

| Domain | Regulation | Retention Period | Applies To |
|--------|------------|-----------------|------------|
| Consumer credit | FCRA | 7 years | Adverse action records, credit decisions |
| Insurance (EU) | Solvency II | 10 years | Underwriting decisions, claims |
| Healthcare | HIPAA | 6 years from creation or last use | PHI, medical decisions |
| Medical device software | FDA SaMD 21 CFR | Device lifetime + 2 years | Decision records for regulated AI |
| EU personal data | GDPR | Minimum necessary | All personal data — no blanket period |

**Implementation:** retention is enforced programmatically, not manually. Audit tables have TTL policies that delete after the retention period. Deletion is logged (you need a record that deletion occurred, even if the data itself is gone).

### What Breaks

**SHAP on a neural network using TreeSHAP:** TreeSHAP only works for tree ensembles. For neural networks, use SHAP DeepExplainer or KernelSHAP (slower, approximate).

**Retraining as the only erasure response:** retraining is expensive. For low-influence subjects, documented pseudonymization may satisfy the regulator. Get legal sign-off on the threshold.

**72-hour clock starting at breach occurrence, not discovery:** the clock starts at discovery. Document the exact discovery timestamp immediately.

---

## 6. Model Cards

> This is the canonical worked example (full template + failure modes), covering the Mitchell et al. 2019 academic framing (Factors/Evaluation Data schema split) and adjacent documentation standards (IBM FactSheets, Datasheets for Datasets).

### The Problem

A model is deployed. Six months later, a new engineer wants to use it for a different use case. There is no documentation of what population it was trained on, what its failure modes are, or what bias analysis was done. The engineer guesses and applies it to a population the model has never seen.

### The Core Insight

A model card is a transferable record of context that cannot be inferred from the model weights alone.

### The Mechanics

**Model card template (Google format):**

```markdown
# Model Card: Credit Risk Scorer v2.4.1

## Model Details
- **Model type:** Gradient Boosted Trees (XGBoost 1.7)
- **Version:** 2.4.1
- **Owners:** Alice Smith (model), Lending Product (business)
- **Date:** 2024-03-08
- **License:** Internal use only

## Intended Use
- **Primary use case:** Automated screening for personal loan applications ($1,000–$50,000)
- **Intended users:** Loan origination system (automated), underwriting team (human review)
- **Out-of-scope uses:**
  - Small business loans (model not validated on business applicants)
  - Mortgage lending (different regulatory framework, not validated)
  - Applications outside the United States

## Training Data
- **Dataset:** Internal credit bureau data, 2017–2023
- **Size:** 2.3M applications (1.8M train, 230K validation, 230K test)
- **Geographic coverage:** 48 contiguous US states
- **Temporal coverage:** 2017–2023; outcome labels are 24-month default
- **Label definition:** Default = 90+ days past due within 24 months of origination

## Performance
| Metric | Value | Benchmark |
|--------|-------|-----------|
| AUC-ROC | 0.847 | Champion: 0.831 |
| PR-AUC | 0.621 | Champion: 0.598 |
| KS Statistic | 0.412 | Champion: 0.389 |
| Brier Score | 0.089 | Champion: 0.101 |

## Fairness Analysis

Evaluated on CFPB protected class proxies (Bayesian Improved Surname Geocoding for race).

| Group | Approval Rate | AUC | False Positive Rate |
|-------|--------------|-----|---------------------|
| Non-Hispanic White | 68.2% | 0.851 | 0.142 |
| Black or African American | 61.4% | 0.839 | 0.158 |
| Hispanic or Latino | 63.1% | 0.843 | 0.151 |
| Asian | 71.3% | 0.856 | 0.134 |

**Demographic Parity Difference (worst pair):** 0.098 (White vs Black)
**Equalized Odds Difference:** 0.024

*Note: Approval rate differences reflect credit risk differences in the training data, not model bias per se. The equalized odds metric is within threshold. Demographic parity difference exceeds 0.05; documented and accepted by compliance per business rationale review.*

## Known Limitations
1. **Thin-file applicants:** AUC drops to 0.71 for applicants with fewer than 3 credit accounts. Score reliability lower for this segment.
2. **Recent graduates:** Model underweights income trajectory; tends to underpredict creditworthiness of applicants <2 years post-graduation.
3. **Self-employed applicants:** Income verification harder; model was not specifically validated on self-employed segment.
4. **Post-2020 economic environment:** Model trained predominantly on pre-pandemic data. Economic regime change may affect calibration.

## Ethical Considerations
- Zip code excluded as proxy for race per ECOA
- Age excluded as direct protected characteristic per ADEA
- SHAP-based adverse action reasons generated for all denials
- Human review available on request per FCRA

## How to Use
- **Input:** Feature vector per schema in `schemas/credit_risk_v2_input.json`
- **Output:** Float in [0, 1] where higher = higher default probability
- **Decision threshold:** 0.15 (tuned for 12% approval rate target; business-configurable)
- **Adverse action:** Call `generate_adverse_action_notice()` for all predictions ≥ threshold
```

### What Breaks

**Model card written once and never updated:** version 2.4.1 may have different fairness numbers than 2.0.0. Model card must be versioned with the model.

**Out-of-scope uses documented but not enforced:** technical controls (feature schema validation, model name checks) are more reliable than documentation.

---

## 7. Incident Response

### The Problem

A model begins producing incorrect predictions in production. Engineers are paged, but there is no agreed protocol for: who makes the rollback call, what evidence is required, and how to communicate to affected users. Each incident is improvised under pressure.

### The Core Insight

Incident response is a pre-committed decision tree. The decisions must be made before the incident, not during it.

### Severity Classification

| Severity | Definition | Response Time | Example |
|----------|-----------|---------------|---------|
| P1 — Critical | >20% error rate or >10% revenue impact. Safety-relevant model failure | 15 min | Fraud model accepting all transactions |
| P2 — Major | Significant accuracy degradation, regulatory model failing | 1 hour | Credit model AUC drops >5% |
| P3 — Moderate | Latency degradation, minor accuracy drift | 4 hours | P99 latency up 200ms |
| P4 — Low | Monitoring alert, no immediate production impact | 24 hours | PSI warning on input feature |

### Rollback Playbook

```bash
#!/bin/bash
# model-rollback.sh — execute this, do not improvise

MODEL_NAME=$1
ROLLBACK_TO_VERSION=$2
ENVIRONMENT=${3:-production}

echo "=== MODEL ROLLBACK INITIATED ==="
echo "Model: $MODEL_NAME | To: $ROLLBACK_TO_VERSION | Env: $ENVIRONMENT"
echo "Initiator: $(git config user.email) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. Validate rollback target exists and is in a deployable state
STATE=$(mlflow models get --name "$MODEL_NAME" --version "$ROLLBACK_TO_VERSION" | jq -r '.current_stage')
if [ "$STATE" != "Production" ] && [ "$STATE" != "Staging" ] && [ "$STATE" != "Approved" ]; then
    echo "ERROR: Version $ROLLBACK_TO_VERSION is in state '$STATE' — not deployable"
    exit 1
fi

# 2. Flip traffic: update Kubernetes deployment to previous image
kubectl set image deployment/$MODEL_NAME \
    model-server=registry.company.com/$MODEL_NAME:$ROLLBACK_TO_VERSION \
    -n $ENVIRONMENT

# 3. Wait for rollout
kubectl rollout status deployment/$MODEL_NAME -n $ENVIRONMENT --timeout=120s

# 4. Verify: run smoke test against the new (old) version
python smoke_test.py --model $MODEL_NAME --version $ROLLBACK_TO_VERSION --env $ENVIRONMENT
if [ $? -ne 0 ]; then
    echo "SMOKE TEST FAILED — rollback may have introduced new issues"
    # Do not auto-rollforward — page the team
    exit 2
fi

# 5. Log to incident record
mlflow runs log-param \
    --run-id "$INCIDENT_RUN_ID" \
    --key "rollback_executed" \
    --value "$(date -u +%Y-%m-%dT%H:%M:%SZ):$MODEL_NAME:$ROLLBACK_TO_VERSION"

echo "=== ROLLBACK COMPLETE ==="
echo "Verify in dashboard: https://grafana.company.com/d/model-health"
```

**Rollback decision criteria (pre-agreed at Gate 4):**

```python
def should_rollback(current_metrics: dict, baseline_metrics: dict, thresholds: dict) -> bool:
    """
    Returns True if rollback should be executed.
    All three conditions are checked; any single violation triggers rollback.
    """
    checks = {
        "accuracy_drop": (
            baseline_metrics["auc"] - current_metrics["auc"]
            > thresholds["max_auc_drop"]       # e.g., 0.03
        ),
        "latency_spike": (
            current_metrics["p99_latency_ms"]
            > thresholds["max_p99_latency_ms"]  # e.g., 200
        ),
        "error_rate": (
            current_metrics["error_rate"]
            > thresholds["max_error_rate"]      # e.g., 0.01
        ),
    }
    return any(checks.values()), checks
```

### Post-Mortem Checklist

Within 48 hours of incident resolution:

- [ ] Timeline of events (detection → diagnosis → rollback → resolution)
- [ ] Root cause identified (data drift? code bug? infrastructure failure? feature pipeline issue?)
- [ ] Impact quantified (affected requests, affected subjects, estimated business impact)
- [ ] Why monitoring did not catch it earlier (or why it did, and the response was slow)
- [ ] Action items with owners and deadlines
- [ ] Model card updated with the failure mode

### What Breaks

**P1 rollback requiring VP approval:** during a P1, decision authority must pre-delegate to the on-call engineer. Approval chains fail under 15-minute response windows.

**Rollback target not tested before incident:** the rollback target itself may have a bug. Smoke tests after rollback are not optional.

**Post-mortem without root cause:** "we rolled back and it fixed it" is not a root cause. If the cause is unknown, the next incident will be identical.

---

## 8. Shadow Deployment for High-Stakes Models

### The Problem

A new credit model has passed all offline evaluations. The team needs production validation before going live, but cannot afford to deny credit to thousands of customers based on an untested model.

### The Core Insight

Shadow deployment runs the new model in production without using its outputs for decisions. You get real traffic, real features, real outcomes — without risk to customers or business.

### The Mechanics

```python
import asyncio
import time
from typing import Any

async def serve_with_shadow(
    request: dict,
    champion_model,
    challenger_model,
    shadow_store
) -> Any:
    """
    Serve the champion's prediction. Fire-and-forget the challenger.
    The challenger's predictions are stored for offline analysis.
    """
    # 1. Get champion prediction — this is what the customer sees
    champion_start = time.perf_counter()
    champion_prediction = champion_model.predict(request["features"])
    champion_latency = (time.perf_counter() - champion_start) * 1000

    # 2. Fire challenger inference asynchronously (do not await — do not block)
    asyncio.create_task(
        _shadow_predict(challenger_model, request, champion_prediction, shadow_store)
    )

    # 3. Return champion prediction to caller immediately
    return champion_prediction

async def _shadow_predict(challenger_model, request, champion_prediction, shadow_store):
    """Runs asynchronously; failure must not propagate to the main serving path."""
    try:
        challenger_start = time.perf_counter()
        challenger_prediction = challenger_model.predict(request["features"])
        challenger_latency = (time.perf_counter() - challenger_start) * 1000

        shadow_store.record({
            "request_id": request["id"],
            "timestamp": time.time(),
            "features_hash": hash(str(sorted(request["features"].items()))),
            "champion_prediction": champion_prediction,
            "challenger_prediction": challenger_prediction,
            "challenger_latency_ms": challenger_latency,
        })
    except Exception as e:
        # Log error but never propagate — champion path must be unaffected
        logger.warning(f"Shadow prediction failed: {e}")
```

**Shadow analysis metrics to compute after accumulating enough traffic:**

| Metric | Purpose | Threshold to Escalate |
|--------|---------|----------------------|
| Agreement rate | Overall alignment | <95% agreement on discrete decisions |
| Score correlation | Rank ordering preserved | Spearman r < 0.90 |
| Disagreement on high-stakes | Cases where predictions diverge on borderline decisions | Manual review required |
| Challenger latency | Verify P99 before routing live traffic | >2x champion P99 |
| Challenger error rate | Serving failures | >0.1% |

**Transitioning from shadow to live:**

1. Shadow phase: run for minimum 2 weeks / 10,000 requests (whichever is longer)
2. Agreement analysis: review disagreement cases with subject matter experts
3. Canary phase: route 5% of traffic to challenger with full monitoring
4. Gradual ramp: 5% → 20% → 50% → 100% with hold periods at each step

### What Breaks

**Shadow path sharing resources with the champion path:** a shadow model that consumes excess CPU degrades champion latency. Shadow must run on isolated compute.

**Not accumulating enough disagreement cases for review:** 10,000 requests with 98% agreement = 200 disagreement cases. Review those 200 — that is where the model behaves differently.

**Skipping canary after shadow:** shadow and canary test different things. Shadow tests model quality with no consequences. Canary tests that the serving infrastructure handles real traffic correctly under load.

---

## Interview Questions

**Q: How do you handle model versioning when you need to support rollback but also need to comply with data retention regulations that require you to be able to reproduce a decision years later?**

A: The requirements are in tension. Rollback requires keeping old model weights accessible and deployable. Data retention requires keeping inference records and the ability to reconstruct the decision environment. The solution is to separate these concerns: model artifacts are versioned in the registry indefinitely (compressed, not running), inference events are logged to an immutable audit store with the model version tag, and the deployment rollback mechanism operates on the registry. When an auditor asks to reproduce a decision from two years ago, you query the audit store for the inference event, identify the model version from the event metadata, and use the archived model artifact to reproduce the prediction. You never need to run the old model live — you just need to preserve the artifact.

**Q: A customer exercises their GDPR right to erasure. Walk through your response.**

A: First, delete their raw training data. Second, query the model registry for all model versions trained on datasets that included the customer. Third, use influence functions or a proxy (leave-one-out on a small validation set) to estimate whether the customer's data materially affected the model weights. For most customers in a large dataset, influence is negligible — document this assessment and consider it resolved. For high-influence cases (a small dataset where the customer was a significant fraction), schedule retraining. Fourth, the inference audit logs cannot be fully erased because financial regulations require retaining decision records. Since you logged the entity ID hash (not the raw ID), you are already pseudonymized. Document the conflict between GDPR Article 17 and your financial audit obligations — most DPAs accept this explanation.

**Q: How do you make TreeSHAP explanations legally defensible for adverse action notices?**

A: Three requirements. First, the explanation must be at the level of the original features (not latent representations), because the explanation is given to a human who does not understand internal model representations. TreeSHAP on tree ensembles gives exact Shapley values — not approximations — which makes it mathematically auditable. Second, the top reasons must be mapped to human-readable language before delivery, and the mapping must be documented and consistent. Third, the explanation must be generated from the exact model and feature values that produced the decision — not a surrogate. The inference event log must store the feature snapshot so the explanation can be reproduced from the audit record, not reconstructed from a different feature state.

**Q: What is the right way to set rollback thresholds?**

A: Set them before deployment, not after the model is in production. The sequence is: (1) during the business sign-off gate, agree on what constitutes "unacceptable model behavior" in terms the business understands — this is usually expressed as revenue impact or error rate, not AUC; (2) translate that business threshold into the metric the monitoring system observes — typically: champion AUC minus current AUC delta, P99 latency, and error rate; (3) document the thresholds in the model registry; (4) configure the monitoring system to fire the rollback alert automatically if thresholds are exceeded. The goal is to make the rollback decision non-discretionary: if metric X exceeds threshold Y, roll back. This removes the political negotiation under pressure.

**Q: How do you prevent training-serving skew from becoming a governance problem?**

A: Training-serving skew is a governance problem when: (a) it causes model behavior to differ from what was evaluated and approved, and (b) no one detects it until an external audit or customer complaint. Prevention requires two things. First, use a shared feature pipeline — the same code that computes features for serving must also be used to generate features for training. This is implemented via a feature store with a point-in-time correct join capability. Second, monitor Population Stability Index on input features in production continuously. If PSI exceeds 0.2 on any feature, the model is operating outside its evaluated distribution, and the approval effectively does not apply. This triggers a re-evaluation gate, not a rollback — the model may still be performing, but the governance coverage must be renewed.

**Q: How does the shadow deployment pattern satisfy regulators for high-stakes model transitions?**

A: Regulators care about two things in model transitions: that the new model has been tested under realistic conditions before it affects decisions, and that there is evidence it performs comparably or better on the dimensions they care about (not just overall AUC, but specifically on protected class proxies). Shadow deployment satisfies the first requirement by accumulating predictions on real traffic without affecting decisions. Satisfying the second requires logging the shadow predictions and running the fairness analysis on actual production traffic — not just on the held-out test set. The shadow period produces a parallel validation dataset that is submitted as part of the compliance gate documentation for the new model version.

## Flashcards

**Why exclude zip code and age from a credit model even if predictive?** #flashcard
Zip code is a proxy for race (ECOA); age is a directly protected characteristic (ADEA). Document exclusions explicitly — an undocumented exclusion looks like the feature was used and hidden.

**What are the four approval gates before a high-stakes model deploys?** #flashcard
1) Automated validation (CI/CD: performance, calibration, PSI, fairness thresholds) 2) Model owner sign-off (non-delegatable attestation) 3) Risk/compliance review (protected attributes, GDPR/ECOA/FCRA) 4) Business sign-off (rollback criteria, runbook).

**What triggers a MAJOR semantic version bump for a model?** #flashcard
Any change that breaks caller assumptions: output type change, score range change (even with rescaling), new required input features, prediction semantics flip, or a framework change that alters numerical precision.

**What must a model owner personally attest to at Gate 2?** #flashcard
Training data fits the use case, known failure modes are documented, feature importance was reviewed, and there is no target leakage — not delegatable to automated checks.

**What does the Gate 3 (risk/compliance) checklist cover?** #flashcard
Protected attribute review (direct + proxy), adverse-action notice mechanism, GDPR Article 22 documentation, ECOA/FHA/FCRA review, data retention policy, right-to-erasure impact.

**What does Gate 4 (business sign-off) require beyond performance metrics?** #flashcard
Confirmation the model solves the actual business problem, that thresholds meet business needs (not just statistical significance), pre-agreed rollback criteria, and an existing on-call runbook.

**What are the non-negotiable properties of an audit log store?** #flashcard
Append-only (no DELETE/UPDATE), encrypted at rest, accessible only to the audit/compliance role (not the serving app), and retention enforced by schedule rather than manual deletion.

**Name four ML-specific GDPR/data breach scenarios.** #flashcard
Model inversion (recovering training data from weights), inference audit log exposure (PII in feature snapshots), feature store breach, and membership inference attacks proving training data presence.

**What are the core fields of a model card (Google format)?** #flashcard
Model details (type/version/owner), intended use and out-of-scope uses, training data provenance, performance vs. benchmark, fairness analysis by group, known limitations, ethical considerations, and how-to-use (input/output/threshold).

**Give an example of a documented "known limitation" in a model card.** #flashcard
E.g., AUC drops to 0.71 for thin-file applicants (<3 credit accounts) — score reliability is lower for that segment, so the model card flags it rather than silently under-serving them.

**What should a post-mortem checklist include within 48 hours of an incident?** #flashcard
Timeline (detection → diagnosis → rollback → resolution), root cause, quantified impact, why monitoring did/didn't catch it, action items with owners/deadlines, and an updated model card.
