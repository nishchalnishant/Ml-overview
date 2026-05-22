# Clinical ML System Design

End-to-end ML system for clinical decision support: sepsis early warning, radiology AI triage, readmission prediction, and drug interaction detection. High-stakes regulatory environment with asymmetric error costs and human-in-the-loop requirements.

**Scale:** Millions of patient encounters/year, <60s alert latency for sepsis, 99.9% uptime (lives depend on it), FDA SaMD regulated.

---

## 1. Problem Framing

### Use Case Spectrum

| Use Case | Input | Output | Latency | FDA Class |
|---|---|---|---|---|
| Sepsis early warning | Continuous vitals + labs | Risk score + alert | <60s | II (De Novo) |
| Chest X-ray triage | DICOM image | Finding + severity | <5 min | II (510(k)) |
| 30-day readmission prediction | Discharge summary + EHR | Risk score | Batch (at discharge) | II |
| Drug interaction detection | Medication orders | Interaction flag + severity | <1s | II (510(k)) |
| ICU mortality prediction | Full ICU record | Mortality probability | Hourly | II |

### Clarifying Questions

- **FDA classification?** Class I (low risk, general wellness) vs Class II (moderate risk, 510(k) or De Novo) vs Class III (high risk, PMA). Does the model replace clinical judgment or augment it?
- **Real-time vs batch?** Continuous monitoring (sepsis) requires streaming inference. Discharge planning (readmission) is batch. This determines the entire architecture.
- **Clinician-facing vs autonomous?** CDS tool that surfaces a recommendation (clinician decides) vs autonomous system that triggers an action (drug dispensing pause). Autonomous = higher FDA class, stricter validation.
- **Which hospitals / EHR systems?** Epic? Cerner? Both? Single-site vs multi-site deployment. Epic has 33% US market; SMART on FHIR availability varies by version.
- **What counts as a label?** Death? ICD-coded diagnosis? Clinician-confirmed? Time to event? Each has different bias and availability characteristics.
- **Retraining cadence allowed?** Post-market surveillance requirements. Locked model (same as validated) vs adaptive model (requires re-validation with each update, per FDA AI/ML action plan 2021).
- **Who bears liability?** Hospital? Vendor? Changes the required audit trail depth and indemnification clauses.

### Asymmetric Error Costs

**False negatives are categorically more costly than false positives in most clinical settings.**

| Error Type | Clinical Consequence | Downstream Cost |
|---|---|---|
| FN: miss sepsis | Patient deteriorates, potentially dies | Wrongful death liability, ICU days |
| FP: flag non-sepsis | Unnecessary blood culture, antibiotics | ~$500 workup cost, antibiotic resistance risk |
| FN: miss lung nodule | Delayed cancer diagnosis | Stage migration, reduced survival |
| FP: flag benign finding | Follow-up CT, patient anxiety | ~$1,000 additional imaging |
| FN: miss drug interaction | Adverse drug event | Hospitalization, potential death |
| FP: block valid medication | Treatment delay | Harm from undertreated condition |

**Cost-sensitive threshold selection:**

$$\text{optimal threshold} = \arg\max_t \left[\text{Sensitivity}(t) \cdot \frac{C_{FN}}{C_{FP}} - (1 - \text{Specificity}(t))\right]$$

where C_FN/C_FP ratio for sepsis is roughly 50–100x. This pushes operating points to high sensitivity (>85%) even at the cost of specificity (50–60%).

**Clinical utility curves** are preferred over ROC/AUC: they incorporate the prevalence and cost ratio, showing net benefit of the model vs treat-all vs treat-none strategies. Vickers & Elkin (2006).

---

## 2. Regulatory and Safety Constraints

### FDA SaMD Classification

Software as a Medical Device (SaMD) is classified by the **significance of information** provided and the **state of the healthcare situation**.

```
                    State of Healthcare Situation
                    ─────────────────────────────────────────
                    Non-serious  │  Serious  │  Critical
                    ─────────────┼───────────┼──────────────
Treat/Diagnose      Class I      │  Class II │  Class III
  (inform)          (exempt)     │  510(k)   │  PMA
                    ─────────────┼───────────┼──────────────
Drive clinical      Class II     │  Class II │  Class III
  management        510(k)       │  De Novo  │  PMA
```

**Pathways:**
- **510(k):** Predicate device exists. 90-day review. Most CDS tools qualify.
- **De Novo:** Novel device, no predicate. 12-month review. Used for first-in-class AI tools (Epic Sepsis Model equivalent).
- **PMA:** Class III. Full clinical trial required. Reserved for autonomous high-stakes decisions (AI-driven drug dispensing).

**Predetermined Change Control Plan (PCCP):** FDA's 2023 guidance allows adaptive models if the vendor pre-specifies what kinds of updates are permissible without re-submission. Critical for retraining pipelines.

### Software Standards

- **IEC 62304:** Medical device software lifecycle. Requires software safety classification (A/B/C), risk management documentation, traceability matrix from requirements to tests.
- **IEC 62366:** Usability engineering — alert design, workflow integration, error prevention.
- **ISO 14971:** Risk management throughout the product lifecycle.
- **HIPAA:** PHI de-identification (Safe Harbor or Expert Determination), BAA with cloud vendors, audit logs for all PHI access, minimum necessary standard.

### Operational Constraints

- **Model lock-in:** The validated model artifact must be frozen. Any weight change = new submission unless covered by PCCP. Use versioned model registry with cryptographic hash.
- **Audit trail:** Every prediction must log: timestamp, patient ID (encrypted), model version, input feature snapshot (or hash), output score, clinician action, outcome (when available). Retained for 7 years minimum.
- **No standard A/B testing:** Randomizing patients to model vs no-model typically requires IRB approval. Shadow mode (model runs in background, no alert shown) is the standard pre-launch approach.
- **Explainability as requirement:** Some state regulations (NY Local Law 144 equivalent in healthcare context) and hospital credentialing committees require the model to provide feature-level explanations.

---

## 3. System Architecture

```
  EHR Systems (Epic, Cerner, Meditech)
           │
           │  HL7 v2 ADT/ORU messages  OR  FHIR R4 API
           ▼
  ┌─────────────────────────────────────────────────────────┐
  │              Data Ingestion Layer                       │
  │  Kafka (streaming) + SFTP batch                        │
  │  HL7 parser → FHIR normalization → PHI tokenization    │
  └──────────────────────┬──────────────────────────────────┘
                         │
           ┌─────────────┼──────────────────┐
           ▼             ▼                  ▼
  ┌──────────────┐ ┌───────────┐   ┌──────────────────┐
  │  OMOP CDM    │ │  Real-    │   │  DICOM / Image   │
  │  Warehouse   │ │  time     │   │  Pipeline        │
  │  (batch/day) │ │  Feature  │   │  (radiology AI)  │
  └──────┬───────┘ │  Store    │   └────────┬─────────┘
         │         │ (Redis)   │            │
         │         └─────┬─────┘            │
         │               │                  │
         └───────┬────────┘                 │
                 ▼                          │
  ┌──────────────────────────┐             │
  │     Feature Extraction   │             │
  │  Vital signs aggregation │             │
  │  Lab value imputation    │             │
  │  Medication encoding     │             │
  │  ICD code embedding      │             │
  └──────────┬───────────────┘             │
             │                             │
             ▼                             ▼
  ┌──────────────────┐          ┌────────────────────┐
  │  Risk Scoring    │          │  Vision Model      │
  │  Models          │          │  (DenseNet/ViT)    │
  │  ─────────────── │          │  Chest X-ray,      │
  │  Sepsis: LSTM +  │          │  Pathology slide   │
  │  Gradient boost  │          └──────────┬─────────┘
  │  Readmission: XGB│                     │
  │  Drug interaction│◄────────────────────┘
  │  rule + embed    │
  └──────────┬───────┘
             │
             ▼
  ┌──────────────────────────────────────────────────────┐
  │         Clinical Decision Support Interface           │
  │  ─────────────────────────────────────────────────── │
  │  Risk score + tier (LOW / ELEVATED / HIGH / CRITICAL)│
  │  SHAP explanation: "Lactate ↑, HR ↑, low UO"        │
  │  Recommended actions (suggested, not mandatory)      │
  │  Confidence interval + calibration info              │
  └──────────────────────────┬───────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────┐
  │              Clinician Action                         │
  │  Accept recommendation │ Dismiss + reason │ Escalate │
  └──────────────────────────┬───────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────┐
  │         Outcome Logging + Feedback Loop               │
  │  Patient outcome (discharge, death, ICU transfer)    │
  │  Clinician override rate + reason codes              │
  │  Alert acceptance rate per unit/provider             │
  └──────────────────────────┬───────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────┐
  │         Drift Monitoring + Model Surveillance        │
  │  Feature distribution shift (KS test, PSI)          │
  │  Score distribution shift                            │
  │  Subgroup calibration tracking                       │
  │  Post-market performance report (FDA requirement)    │
  └──────────────────────────────────────────────────────┘
```

**EHR Integration Points:**
- **Epic SMART on FHIR:** CDS Hooks standard for embedded CDS. `patient-view` hook fires when provider opens chart. `order-sign` hook fires before medication order.
- **Cerner PowerChart:** CCOW-based context sharing; CDS integration via Ignite APIs.
- **HL7 v2 ADT feeds:** Real-time admission/discharge/transfer events drive patient list management.

---

## 4. Data Challenges

### EHR Data Quality

EHR data was designed for billing, not ML. Every assumption made in general ML pipelines breaks.

**Missing data patterns:**
- Labs are missing not at random (MNAR): a normal-appearing creatinine may be missing *because* the patient looked well. Imputation with population mean introduces bias.
- Vital sign gaps during transport, shift change, or patient refusal are systematically different from random gaps.
- Free-text notes contain critical clinical information (allergies, social history) that structured fields miss.

**Strategies:**
```python
# Missingness as a feature — critical in clinical settings
def engineer_missingness_features(df: pd.DataFrame, lab_cols: list) -> pd.DataFrame:
    for col in lab_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)
        df[f"{col}_hours_since_last"] = df.groupby("patient_id")[col].apply(
            lambda s: s.isna().cumsum() * hours_per_step
        )
    return df
```

**Irregular time series:** Vital signs every 4 hours in ward, every 15 minutes in ICU. Labs drawn on clinical suspicion, not at fixed intervals. Models must handle variable-length sequences with irregular timestamps.

**ICD code heterogeneity:**
- ICD-9 vs ICD-10 transition (2015) creates a cliff in longitudinal data.
- Coding practices differ across hospitals (upcoding for reimbursement, undercoding for simplicity).
- Principal vs secondary diagnosis ordering changes meaning.
- ICD-10-CM has ~70,000 codes; grouping to CCS (Clinical Classifications Software) categories (286 groups) is standard preprocessing.

### Data Harmonization: OMOP CDM

The OHDSI OMOP Common Data Model maps institution-specific codes to standard vocabularies:
- SNOMED CT for diagnoses
- RxNorm for medications
- LOINC for lab tests
- CPT4 / HCPCS for procedures

This enables multi-site training and federated validation without transferring raw PHI.

### Selection Bias

**Hospitalized patient bias:** A model trained on hospital EHR data only sees patients sick enough to be admitted. It cannot generalize to ambulatory settings. Readmission models have no information about the 80% of patients who never come back (left-censored outcomes).

**Immortal time bias:** If you define the prediction window as "did the patient get sepsis after hour 6?" but your features include data from hours 0–6, early deterioration signals bleed into the feature window, inflating apparent model performance.

**Label quality:**
- Sepsis-3 definition requires clinical judgment (suspected infection + organ dysfunction). Retrospective coding is inconsistent.
- Death is a clean label but conflates disease severity with care quality.
- Discharge diagnosis is coded weeks after admission; cannot be used for real-time labels.
- **Solution:** Use composite endpoints (ICU transfer OR vasopressor use OR death within 24h) that are objective and timely.

---

## 5. Model Architecture

### Time Series of Clinical Events

Patient data is a multivariate, irregular time series with heterogeneous data types:

```
t=0h    t=2h    t=6h    t=12h   t=24h
 │       │       │        │       │
 ├─ HR   ├─ HR   ├─ HR    ├─ HR   ├─ HR    ← continuous, q4h
 ├─ BP   ├─ BP   ├─ BP    ├─ BP   ├─ BP
 ├─ Temp ├─ Temp ├─ Temp  ├─ Temp ├─ Temp
 │       ├─ WBC  │        ├─ Lac  │        ← labs: irregular
 │       ├─ Cr   │        ├─ Cr   │
 ├─ [IV  │       ├─ [ABX  │       │        ← medications: events
 │  Fluid]       │  start]│
 ├─ [ICD ├─ [ICD │        │       │        ← diagnoses: sparse
 │  code]│  code]│
```

### Model Options

**LSTM for continuous monitoring:**
- Handles variable-length sequences natively.
- Forget gate learns to discount stale measurements.
- Per-timestep output enables early warning at each observation.
- Weakness: struggles with very long sequences (>200 time steps) due to gradient decay.

**Transformer on irregular time series (preferred for new systems):**
- Positional encoding replaced with time-delta encoding: `PE(t) = sin(t/T_max)`.
- Multi-head attention captures cross-variable correlations (rising HR + falling BP = hemodynamic instability).
- `STraTS` (2021) and `ETHAN` (2022) are clinical transformer architectures designed for this.
- Self-supervised pretraining on large EHR corpora (via masked lab prediction) improves sample efficiency.

```python
class ClinicalTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 128, n_heads: int = 8):
        super().__init__()
        # Time-aware positional encoding
        self.time_encoder = nn.Linear(1, d_model)
        self.value_encoder = nn.Linear(n_features, d_model)
        self.missingness_encoder = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, values, times, mask):
        # values: (B, T, F), times: (B, T, 1), mask: (B, T) — True where missing
        time_emb = self.time_encoder(times)
        value_emb = self.value_encoder(values)
        x = time_emb + value_emb
        x = self.transformer(x, src_key_padding_mask=mask)
        return torch.sigmoid(self.output_head(x[:, -1, :]))  # last timestep
```

**Gradient boosting for tabular features (discharge prediction, drug interaction):**
- Aggregated features over fixed windows: min/max/mean/last of each vital and lab.
- Handles mixed types natively (continuous labs, binary flags, ICD codes as categorical).
- Interpretable via SHAP. Faster to train and iterate.
- XGBoost / LightGBM are standard. CatBoost handles high-cardinality ICD codes well.

**Ensemble for robustness:**
- Average transformer score (temporal patterns) + gradient boosting score (cross-sectional features).
- Disagreement between ensemble members is a useful uncertainty signal.
- If transformer score = 0.85 but GBM score = 0.30 → flag for clinician review rather than auto-alert.

### Radiology AI (Image Pathway)

- **DenseNet-121** (Rajpurkar et al., CheXNet 2017) for chest X-ray pathology classification.
- **Vision Transformer (ViT-B/16)** fine-tuned on CheXpert/MIMIC-CXR for improved calibration.
- **Multi-label output:** pneumonia, atelectasis, cardiomegaly, pleural effusion, pneumothorax — not mutually exclusive.
- **Critical finding triage:** separate binary head for "critical finding requiring urgent attention" to drive workflow prioritization.
- Preprocessing: DICOM → PNG, CLAHE normalization, 224×224 or 512×512 depending on finding size.

---

## 6. Handling Class Imbalance in Clinical Settings

### Prevalence Realities

| Condition | Approximate prevalence in target population |
|---|---|
| Sepsis (hospital inpatients) | 2–4% |
| 30-day readmission (general medicine) | 12–15% |
| Pneumonia on chest X-ray (ED) | 5–10% |
| Rare genetic disease | 0.01–0.1% |
| Critical drug interaction | 1–3% of polypharmacy patients |

### Strategies

**Cost-sensitive learning:**
```python
# XGBoost: scale_pos_weight = (negative samples) / (positive samples)
model = XGBClassifier(
    scale_pos_weight=97,  # for 3% prevalence: 97/3
    eval_metric="aucpr",  # PR-AUC more informative than ROC-AUC under imbalance
)
```

**Focal loss for neural models (Lin et al., 2017):**
```python
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()
```

**Operating point selection — do not use default 0.5 threshold:**

The correct operating point is determined by:
1. Hospital policy: "alert when probability > X%"
2. Clinical utility analysis: net benefit curves
3. Stakeholder elicitation: what sensitivity/specificity trade-off do clinicians accept?

A sepsis model at 0.5 threshold with 3% prevalence and 80% sensitivity will have ~40% positive predictive value — 60% of alerts are false positives. Clinical teams routinely ignore such systems (Epic Sepsis Model controversy, UCSF/NEJM 2021).

**Clinical utility curves (net benefit):**
$$NB(\text{threshold}) = \frac{TP}{n} - \frac{FP}{n} \cdot \frac{p_t}{1 - p_t}$$

where p_t is the decision threshold probability. Plot NB vs threshold and compare model vs treat-all vs treat-none. FDA guidance recommends this over ROC alone for clinical validation.

---

## 7. Fairness and Bias

### Known Sources of Bias in Clinical AI

**Pulse oximetry bias:** SpO2 sensors overestimate oxygen saturation in patients with darker skin tones by 3–4 percentage points (Sjoding et al., NEJM 2020). Any model using SpO2 as a feature will have worse performance for Black patients. Solution: include race as a feature (controversial) or train separate calibration layers.

**Dermatology AI:** Esteva et al. (Nature 2017) and subsequent studies showed dermatology classifiers underperformed on darker skin Fitzpatrick scores due to training data dominated by lighter skin images. Datasets like Fitzpatrick17k and DDI are now required for validation.

**Sepsis model disparities:** Wong et al. (npj Digital Medicine 2021) found that commercial sepsis models had worse sensitivity for Black patients vs White patients at the same alert threshold.

**Socioeconomic status proxy features:** ZIP code, insurance type, prior admissions — legitimate predictors of readmission but proxies for SES and race. Including them improves accuracy but entrenches inequity in resource allocation.

### Required Fairness Analyses

| Analysis | Method | Threshold |
|---|---|---|
| Subgroup calibration | Reliability diagrams by race/sex/age | Calibration slope ∈ [0.8, 1.2] per group |
| Sensitivity/specificity parity | Report separately per demographic | Flag if gap >5 percentage points |
| Disparate impact ratio | min(TPR_group) / max(TPR_group) | Must be >0.8 (4/5 rule) |
| Calibration across sites | Hosmer-Lemeshow test per hospital | p > 0.05 before deployment at new site |

**Bias as regulatory requirement:** FDA's 2022 action plan explicitly lists demographic performance analysis as a required component of AI/ML SaMD submissions. Joint Commission (hospital accreditation body) is developing health equity standards that include AI audit requirements.

**Mitigation approaches:**
- **Data-side:** Oversample underrepresented groups; curate demographically balanced validation sets.
- **Model-side:** Adversarial debiasing (fairness constraints during training); multi-task learning with demographic parity constraint.
- **Post-processing:** Separate thresholds per demographic group to equalize sensitivity (controversial, legally complex under anti-discrimination law).
- **Process-side:** Clinical advisory board with diverse representation reviews model outputs before deployment.

---

## 8. Human-in-the-Loop Design

### Alert Fatigue: The Core Product Problem

The Epic Sepsis Model (ESM) controversy (Wong et al., NEJM Evidence 2021) is the canonical case study. ESM had AUROC 0.76 — a respectable metric — but in prospective evaluation:
- 18% of alerts were accepted by nurses.
- 67% of sepsis cases were missed (sensitivity 33%).
- Nurses reported ignoring alerts due to habituation.

**Alert fatigue physics:**
- ICU nurses receive 150–200 alarms per patient per day across all monitoring systems.
- False positive rate >70% causes desensitization within weeks of deployment.
- A model with 90% sensitivity but 70% FPR generates 2.3 false alerts for every true one at 3% prevalence — cognitively overwhelming.

### Tiered Alerting Architecture

```
Score < 0.3:  No alert. Log for retrospective analysis.
Score 0.3–0.5: Passive indicator. Dashboard color change (yellow).
               No interruption. Clinician sees on next chart open.
Score 0.5–0.7: Non-interruptive alert. Sidebar notification.
               One-click dismiss. Reason code required.
Score 0.7–0.85: Interruptive alert. Pop-up with SHAP explanation.
               "High sepsis risk. Lactate elevated, HR rising."
               Accept / Dismiss+reason / Escalate options.
Score >0.85:   Critical alert. Pages RN and attending.
               EHR order set pre-populated (blood cultures, lactate).
               Cannot dismiss without senior physician override.
```

**Alert budget:** Target <2 actionable alerts per nurse per 12-hour shift. Monitor alert burden per unit; auto-suppress if rate exceeds threshold.

### Explainability for Clinical Reasoning

SHAP values are the standard for structured/tabular models. Clinicians need feature-level attribution in clinical language, not raw feature names.

```python
# Map SHAP feature names to clinical language
CLINICAL_EXPLANATIONS = {
    "hr_max_4h":        "Heart rate elevated (max {val:.0f} bpm)",
    "lactate_last":     "Lactate elevated ({val:.1f} mmol/L)",
    "sbp_min_4h":       "Blood pressure low (min {val:.0f} mmHg)",
    "urine_output_8h":  "Low urine output ({val:.0f} mL/8h)",
    "wbc_last":         "Abnormal WBC ({val:.1f} K/uL)",
    "temp_max_4h":      "Fever ({val:.1f}°C)",
}

def generate_clinical_explanation(shap_values: dict, feature_values: dict, top_k: int = 3):
    sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    explanations = []
    for feat, shap_val in sorted_features[:top_k]:
        if feat in CLINICAL_EXPLANATIONS:
            template = CLINICAL_EXPLANATIONS[feat]
            text = template.format(val=feature_values[feat])
            direction = "↑ risk" if shap_val > 0 else "↓ risk"
            explanations.append(f"{text} ({direction})")
    return explanations
```

**Uncertainty communication:** Display calibrated confidence intervals, not just point estimates. "Sepsis probability: 78% (CI: 65–88%)" is more actionable than "ALERT: Sepsis risk HIGH." Clinicians trained in Bayesian reasoning use this information differently.

**Attention visualization for image models:** Grad-CAM heatmaps overlaid on X-ray show which regions drove the prediction. Required for radiologist trust and for identifying spurious correlations (model learning from pacemaker presence rather than lung texture).

---

## 9. Validation and Monitoring

### Validation Hierarchy

| Validation type | Description | When required |
|---|---|---|
| Internal validation | Train/test split, CV | Always |
| Temporal validation | Train 2018–2020, test 2021 | Required for FDA |
| External validation | Different hospital (unseen during training) | Required for multi-site claim |
| Prospective validation | Shadow mode, compare to ground truth after | Required before clinical deployment |
| Prospective RCT | Randomized exposure to model vs control | Required for Class III / PMA; IRB needed |

**Temporal validation is the minimum bar.** Models that skip temporal validation and use random splits on longitudinal data are severely overfit — future leakage from labels coded retrospectively is common.

**Site-specific calibration:** A model trained at academic medical center performs poorly at community hospital due to:
- Different patient population (acuity, demographics)
- Different clinical protocols (order culture affects features)
- Different EHR configuration (which labs are routinely ordered)

**Solution:** Isotonic regression or Platt scaling recalibrates score-to-probability mapping at each new site using 500–1,000 local patient-outcome pairs.

### Continuous Monitoring

```python
class ClinicalModelMonitor:
    def __init__(self, reference_df: pd.DataFrame, feature_cols: list):
        self.reference = reference_df
        self.features = feature_cols
        self.psi_threshold = 0.2      # Population Stability Index
        self.calibration_threshold = 0.1  # Max ECE (Expected Calibration Error)

    def compute_psi(self, current_df: pd.DataFrame, feature: str) -> float:
        # Population Stability Index: <0.1 stable, 0.1–0.2 minor shift, >0.2 major
        ref_pcts, bins = np.histogram(self.reference[feature].dropna(), bins=10, density=True)
        cur_pcts, _ = np.histogram(current_df[feature].dropna(), bins=bins, density=True)
        # Avoid log(0)
        ref_pcts = np.where(ref_pcts == 0, 1e-4, ref_pcts)
        cur_pcts = np.where(cur_pcts == 0, 1e-4, cur_pcts)
        return np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts))

    def check_score_drift(self, current_scores: np.ndarray) -> dict:
        ks_stat, ks_p = ks_2samp(self.reference["score"], current_scores)
        return {"ks_statistic": ks_stat, "p_value": ks_p, "alert": ks_p < 0.001}

    def check_subgroup_calibration(self, df: pd.DataFrame) -> dict:
        results = {}
        for group in df["race_ethnicity"].unique():
            sub = df[df["race_ethnicity"] == group]
            if len(sub) < 50:
                continue
            # Hosmer-Lemeshow or ECE
            results[group] = expected_calibration_error(sub["score"], sub["label"])
        return results
```

**Monitoring triggers for FDA post-market surveillance:**
- PSI > 0.2 on any top-10 feature → flag for review.
- Score distribution shift (KS test p < 0.001) → automatic model pause pending investigation.
- Subgroup ECE > 0.1 → diversity/equity review within 30 days.
- Alert acceptance rate drops >20% → alert fatigue review.
- New ICD-10 codes appearing in >5% of encounters (e.g., COVID U07.1 in March 2020) → model behavior audit.

---

## 10. Deployment Constraints

### Model Versioning with Regulatory Audit Trail

```
Model Registry entry (immutable):
{
  "model_id": "sepsis_v3.2.1",
  "sha256": "a8f3c2...",
  "training_data_hash": "b7e1d4...",
  "validation_auc": 0.821,
  "validation_date": "2024-01-15",
  "fda_submission": "K240123",
  "pccp_change_type": "minor_recalibration",
  "approved_by": ["CMO", "CISO", "FDA_reference"],
  "deployment_sites": ["MGH", "BWH"],
  "locked": true
}
```

Every inference call logs: `{patient_token, model_id, timestamp, feature_hash, score, alert_tier_shown, clinician_id, action_taken, action_timestamp}`.

### Shadow Mode Deployment

Before going live, run the model in shadow mode for 30–90 days:
1. Model generates scores and would-have-been alerts.
2. No alerts shown to clinicians.
3. Compare model predictions to actual outcomes.
4. Compare to existing clinical tools (NEWS2, qSOFA, APACHE II).
5. Measure: sensitivity, PPV at proposed threshold, time-to-alert vs current standard.

Shadow mode does not require IRB because patients are not being treated differently. It does require data access approval and BAA.

### EHR Integration

**Epic SMART on FHIR / CDS Hooks:**
```json
{
  "hookInstance": "uuid",
  "hook": "patient-view",
  "context": {
    "userId": "Practitioner/123",
    "patientId": "Patient/456",
    "encounterId": "Encounter/789"
  }
}
```

Model service receives CDS Hook, fetches FHIR resources (`Observation`, `MedicationRequest`, `Condition`), computes score, returns `cards` array with alert and SMART app link for detailed view.

**Graceful degradation:**
- If feature pipeline fails (EHR downtime), return "model unavailable" rather than defaulting to 0 risk.
- Fallback to rule-based alert (SIRS criteria, qSOFA) if ML service is down.
- Circuit breaker pattern: if model latency >5s, bypass ML and use rules-only path.
- Never silently fail with stale scores — display "last updated 4 hours ago" with timestamp.

---

## 11. Failure Modes

### Alert Fatigue

**Mechanism:** High FPR → clinicians learn to dismiss without reading → true positives missed → clinical harm. Worse in ICU where staff already manages 150+ alarms/day.

**Mitigation:** Tiered alerting, alert budget enforcement, regular alert burden reporting to unit leadership, suppression of cascading alerts (one alert per 4-hour window per patient per condition).

**Detection:** Alert acceptance rate tracked per unit per model. If acceptance rate drops below 15% for a given alert tier, escalate to model review committee.

### Distribution Shift from Protocol Changes

**Mechanism:** Sepsis bundle protocol changes (e.g., lactate removed from standard order set) → feature distribution changes → model miscalibrated → sensitivity drops silently.

**Real example:** COVID-19 pandemic (March 2020) changed admission criteria, medication protocols, patient mix, and coding practices simultaneously. Models trained on pre-COVID data had severely degraded performance within weeks.

**Mitigation:** Monitor PSI on top features weekly. Tie model revalidation schedule to clinical protocol change management process (every protocol change triggers model impact assessment).

### Feedback Loops

**Mechanism:** Model alerts → clinicians intervene earlier → outcomes improve → fewer positive labels in future data → retraining on this data makes model appear less accurate on the intervention-affected population → threshold raised → fewer alerts → outcomes worsen.

This is the **causal feedback loop problem**: the model changes the very outcomes it was trained to predict.

**Mitigation:** Withhold a random 5–10% of patients from receiving alerts (IRB-approved holdout group) for ongoing outcome measurement. Counterfactual logging: record would-have-been scores for patients where clinicians acted before the model triggered.

### Demographic Bias

**Mechanism:** Training data from historically biased healthcare system → model inherits disparities → higher false negative rate for marginalized groups → widens existing health equity gap.

**Mitigation:** Mandatory subgroup calibration analysis before and after deployment. Annual bias audit as part of post-market surveillance. Engage patient advocacy groups in threshold-setting decisions.

### EHR Vendor Lock-in

**Mechanism:** Epic SMART on FHIR app built for Epic 2021 breaks on Epic 2023 upgrade. FHIR implementation varies by vendor (Cerner FHIR ≠ Epic FHIR despite same standard). Multi-site deployment becomes a full-time integration engineering effort.

**Mitigation:** Abstract EHR interface behind FHIR facade. Use HL7 v2 as fallback for sites without FHIR. Maintain a synthetic patient test suite that runs against each EHR integration on every upgrade. Budget 2–3 integration engineers per 10 hospital sites.

### Model Ossification

**Mechanism:** Regulatory lock-in means the deployed model cannot be updated without re-submission. A 2020 model running in 2025 is trained on pre-COVID, pre-GLP1, pre-RSV-vaccine clinical patterns. Model becomes progressively less accurate as medicine evolves.

**Mitigation:** File PCCP at time of initial submission specifying acceptable update types (recalibration, feature set expansion) that do not require new 510(k). Plan revalidation cycles (every 18–24 months) as part of product roadmap.

---

## 12. Interview Questions

**Q1: You're asked to deploy a sepsis prediction model. The validation AUROC is 0.85. Is the model ready for deployment?**

AUROC alone is insufficient. First, AUROC is threshold-agnostic and insensitive to calibration. A perfectly ranked but miscalibrated model (all scores in [0.4, 0.6]) is not deployable. Second, the validation must be temporal and external — random splits overestimate performance due to label leakage in retrospectively coded EHR data. Third, subgroup performance must be evaluated; aggregate AUROC hides 10-percentage-point sensitivity gaps across demographic groups. Fourth, shadow mode prospective validation is required to measure alert burden and acceptance rate under real clinical workflows. AUROC 0.85 is a necessary but not sufficient condition.

---

**Q2: How do you handle missing lab values in a clinical model? Population mean imputation?**

Never population mean imputation in clinical settings — missingness is MNAR (missing not at random). A normal-appearing WBC may be missing *because* the patient looked well. Mean imputation biases the model toward assuming normal values, which suppresses risk scores for patients who are too sick to have labs drawn. Correct approaches: (1) add binary missingness indicator features, (2) add time-since-last-observation features, (3) for time-series models, use masking in the attention mechanism, (4) for gradient boosting, leave as NaN and use native missing value handling (XGBoost/LightGBM support this natively). Forward-fill only within a clinically meaningful window (e.g., 4 hours for vitals, 24 hours for stable labs like creatinine).

---

**Q3: A hospital asks to A/B test the sepsis model — one ICU gets alerts, the other doesn't. How do you set this up?**

This requires IRB (Institutional Review Board) approval because patients in the control arm may receive inferior care due to withheld clinical decision support. The IRB application must justify the clinical equipoise (genuine uncertainty about model benefit), provide a stopping rule (stop if harm detected), and obtain waiver of individual consent given operational nature of the intervention. Timeline: 3–6 months for IRB. Alternative for faster evidence: pre-post study (model off for 6 months, model on for 6 months, compare outcomes), though this has confounders from temporal trends. Another alternative: cluster-randomized design by unit rather than individual patient, reducing contamination.

---

**Q4: The clinical team reports the model "never alerts anymore" six months after launch. What happened?**

Classic feedback loop with possible distribution shift. Investigate: (1) check alert rate trend — if it dropped sharply, check score distribution for shift using PSI; (2) check if any EHR upgrade or clinical protocol change occurred that changed feature distributions (e.g., lactate no longer routinely ordered); (3) check if clinicians customized alert thresholds in the EHR configuration (some EHR implementations allow per-unit threshold overrides); (4) check feature pipeline — silent failures in the ETL pipeline can result in stale or zeroed features, which push scores to a calibrated baseline. Run the model on historical patients with known outcomes and compare scores to initial validation to determine if model behavior has changed vs alert delivery pipeline.

---

**Q5: How do you validate a radiology AI before deployment? The radiologist says "just compare to my reads."**

Radiologist-vs-model comparison is necessary but not sufficient. First, which radiologist? Inter-reader variability for chest X-ray is significant (κ ≈ 0.6–0.7 for pneumonia). Need a multi-reader panel with adjudication for the reference standard, not a single reader. Second, case mix matters — a curated set of interesting cases overestimates real-world performance (spectrum bias). Validation must use consecutive cases from the target clinical workflow. Third, the clinical question is not "is the model as good as a radiologist?" but "does the model improve workflow?" — measure time-to-read for critical findings, overnight error rate, missed finding rate. Fourth, temporal and site-specific validation required. Fifth, subgroup analysis by image acquisition parameters (different X-ray machines produce different image characteristics).

---

**Q6: How do you address the feedback loop problem where the model's alerts change future training data?**

The model intervention confounds the very outcome it predicts. If the sepsis model alerts at t=4h and clinicians treat early, the patient may not progress to sepsis — so the label is 0 (no sepsis) for a patient who was genuinely high risk. Retraining on this data teaches the model that high-risk feature patterns are actually low risk. Solutions: (1) prospective holdout — randomly withhold 5–10% of alerts (IRB required) so you have unintervened control patients; (2) counterfactual labels — label "would have developed sepsis absent intervention" using domain knowledge or causal models; (3) use process labels (antibiotics given, blood cultures drawn) rather than outcome labels (sepsis diagnosis); (4) in the retraining pipeline, upweight patients where no intervention was taken and outcome occurred. This is an open research problem; no perfect solution exists, but ignoring it leads to model degradation over 12–18 months.

---

**Q7: A patient advocacy group raises concerns that your readmission model has worse sensitivity for Black patients. What do you do?**

Immediate actions: (1) reproduce the disparity with rigorous statistical testing — confidence intervals, sample size checks — to confirm it is real and not noise; (2) audit the feature set for proxy variables (ZIP code, insurance type, prior no-shows) that are correlated with race and may encode historical access barriers rather than medical risk; (3) check training data composition — if Black patients are underrepresented, the model has less data to learn from for that group. Medium-term: (4) convene a clinical and ethics review board including patient advocates; (5) evaluate post-hoc recalibration with group-specific thresholds (improves equity but creates legal complexity); (6) consider whether the disparity reflects real outcome differences vs measurement artifact (e.g., if Black patients are systematically undertreated at discharge, the model is accurately predicting a biased system, not perpetuating one). Long-term: (7) work with hospital leadership on upstream data quality improvement, (8) report findings in post-market surveillance per FDA requirements, (9) publish results — suppressing known bias creates liability and perpetuates harm.

---

## References and Further Reading

- **Google Diabetic Retinopathy:** Gulshan et al., *JAMA* 2016. First high-profile FDA-cleared deep learning diagnostic. Notes on deployment challenges: Beede et al., CHI 2020 (why field deployment underperformed lab results).
- **Epic Sepsis Model controversy:** Wong et al., *NEJM Evidence* 2021. Prospective evaluation showed poor sensitivity (33%) despite AUROC 0.76. Canonical case study in the gap between metric performance and clinical utility.
- **Pulse oximetry bias:** Sjoding et al., *NEJM* 2020. Measured SpO2 overestimates arterial oxygen saturation in Black patients; clinical AI using SpO2 inherits this bias.
- **Clinical utility curves:** Vickers & Elkin, *Medical Decision Making* 2006. Decision curve analysis is the recommended framework for evaluating clinical prediction models.
- **FDA AI/ML action plan:** FDA, January 2021. Outlines PCCP framework, post-market surveillance requirements, and transparency guidance for adaptive AI/ML SaMD.
- **OHDSI OMOP CDM:** ohdsi.org. Standard for observational health data harmonization across sites.
- **STraTS:** Tipirneni & Reddy, *AAAI* 2022. Self-supervised transformer for clinical time series with irregular observations.
- **Fairness in clinical AI:** Obermeyer et al., *Science* 2019 (commercial risk score discriminated against Black patients using health costs as proxy for health needs).
