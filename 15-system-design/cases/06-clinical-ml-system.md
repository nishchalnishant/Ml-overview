---
module: Production Ml
topic: System Design
subtopic: Clinical Ml System
status: unread
tags: [productionml, ml, system-design-clinical-ml-syst]
---
# Clinical ML System Design

End-to-end ML system for clinical decision support: sepsis early warning, radiology triage, readmission prediction, drug interaction detection. High-stakes regulatory environment with asymmetric error costs and human-in-the-loop requirements.

**Scale:** Millions of patient encounters/year, <60s alert latency for sepsis, 99.9% uptime, FDA SaMD regulated.

---

## 1. Problem Framing

### Use Case Spectrum

| Use Case | Input | Output | Latency | FDA Class |
|---|---|---|---|---|
| Sepsis early warning | Continuous vitals + labs | Risk score + alert | <60s | II (De Novo) |
| Chest X-ray triage | DICOM image | Finding + severity | <5 min | II (510(k)) |
| 30-day readmission | Discharge summary + EHR | Risk score | Batch | II |
| Drug interaction detection | Medication orders | Interaction flag | <1s | II (510(k)) |
| ICU mortality prediction | Full ICU record | Mortality probability | Hourly | II |

### Clarifying Questions

- **FDA classification?** Class I (low risk) vs Class II (510(k)/De Novo) vs Class III (PMA). Does the model replace or augment clinical judgment?
- **Real-time vs batch?** Continuous monitoring (sepsis) needs streaming inference; discharge planning (readmission) is batch. This shapes the whole architecture.
- **Clinician-facing vs autonomous?** A CDS tool that surfaces a recommendation vs one that triggers an action directly (e.g., pausing drug dispensing). Autonomous = higher FDA class, stricter validation.
- **Which EHR systems?** Epic and/or Cerner, single-site vs multi-site. Epic has ~33% US market share; SMART on FHIR support varies by version.
- **What counts as a label?** Death, ICD-coded diagnosis, clinician confirmation, time-to-event — each has different bias and availability.
- **Retraining cadence?** Locked model (matches what was validated) vs adaptive model (needs re-validation on each update, per FDA's 2021 AI/ML action plan).
- **Who bears liability?** Hospital or vendor — affects required audit trail depth.

### Asymmetric Error Costs

**False negatives are categorically more costly than false positives in most clinical settings.**

| Error Type | Consequence | Downstream Cost |
|---|---|---|
| FN: miss sepsis | Patient deteriorates, may die | Liability, ICU days |
| FP: flag non-sepsis | Unnecessary workup | ~$500, antibiotic resistance risk |
| FN: miss lung nodule | Delayed cancer diagnosis | Stage migration, lower survival |
| FP: flag benign finding | Follow-up CT, anxiety | ~$1,000 |
| FN: miss drug interaction | Adverse drug event | Hospitalization, possible death |
| FP: block valid medication | Treatment delay | Harm from undertreatment |

**Cost-sensitive threshold:**

$$\text{optimal threshold} = \arg\max_t \left[\text{Sensitivity}(t) \cdot \frac{C_{FN}}{C_{FP}} - (1 - \text{Specificity}(t))\right]$$

For sepsis, C_FN/C_FP is roughly 50–100x, pushing operating points to high sensitivity (>85%) even at the cost of specificity (50–60%).

**Clinical utility curves** (net benefit vs threshold) are preferred over ROC/AUC since they incorporate prevalence and cost ratio (Vickers & Elkin, 2006).

---

## 2. Regulatory and Safety Constraints

### FDA SaMD Classification

Classified by significance of information provided and the state of the healthcare situation (non-serious/serious/critical) crossed with informing vs driving clinical management. Higher stakes → higher class.

**Pathways:**
- **510(k):** Predicate device exists. ~90-day review. Most CDS tools qualify.
- **De Novo:** No predicate, novel device. ~12-month review. Used for first-in-class AI tools.
- **PMA:** Class III, full clinical trial required. Reserved for autonomous high-stakes decisions.

**Predetermined Change Control Plan (PCCP):** FDA's 2023 guidance lets vendors pre-specify permissible model update types (e.g., recalibration) that skip re-submission. Important for retraining pipelines.

### Standards and Compliance

- **IEC 62304:** Medical device software lifecycle — safety classification, risk docs, requirements-to-tests traceability.
- **IEC 62366:** Usability engineering (alert design, workflow integration).
- **ISO 14971:** Risk management across the product lifecycle.
- **HIPAA:** PHI de-identification, BAAs with cloud vendors, audit logs, minimum-necessary access.

### Operational Constraints

- **Model lock-in:** The validated artifact is frozen; any weight change needs a new submission unless covered by a PCCP. Use a versioned registry with cryptographic hashes.
- **Audit trail:** Log timestamp, patient ID (encrypted), model version, input snapshot/hash, output score, clinician action, and outcome. Retain 7+ years.
- **No standard A/B testing:** Randomizing patients to model vs no-model usually needs IRB approval. Shadow mode (model runs silently, no alert shown) is the standard pre-launch approach.
- **Explainability:** Hospital credentialing committees and some regulations require feature-level explanations.

---

## 3. System Architecture

```
  EHR Systems (Epic, Cerner, Meditech)
           │  HL7 v2 ADT/ORU  OR  FHIR R4 API
           ▼
  Data Ingestion (Kafka streaming + SFTP batch)
  HL7 parser → FHIR normalization → PHI tokenization
           │
   ┌───────┼──────────────┐
   ▼       ▼               ▼
 OMOP CDM  Real-time      DICOM/Image
 Warehouse Feature Store  Pipeline
 (batch)   (Redis)        (radiology)
   │         │               │
   └────┬────┘               │
        ▼                    │
  Feature Extraction         │
  (vitals, labs, meds,       │
   ICD embeddings)           │
        │                    ▼
        ▼             Vision Model (DenseNet/ViT)
  Risk Scoring Models  Chest X-ray, pathology
  (LSTM/GBM/rules)  ◄───────┘
        │
        ▼
  Clinical Decision Support Interface
  Risk tier (LOW/ELEVATED/HIGH/CRITICAL) + SHAP explanation
  + recommended (non-mandatory) actions + confidence interval
        │
        ▼
  Clinician Action: Accept / Dismiss+reason / Escalate
        │
        ▼
  Outcome Logging + Feedback Loop
  (outcome, override rate, alert acceptance rate)
        │
        ▼
  Drift Monitoring + Model Surveillance
  (PSI/KS drift, subgroup calibration, post-market report)
```

**EHR integration points:**
- **Epic SMART on FHIR:** CDS Hooks. `patient-view` fires on chart open, `order-sign` fires before a medication order.
- **Cerner PowerChart:** CCOW context sharing, CDS via Ignite APIs.
- **HL7 v2 ADT feeds:** Real-time admit/discharge/transfer events drive patient list management.

---

## 4. Data Challenges

### EHR Data Quality

EHR data is built for billing, not ML — most general ML pipeline assumptions break here.

**Missing data:**
- Labs are missing not at random (MNAR): a normal-looking creatinine may be missing *because* the patient looked well. Mean imputation biases scores toward "normal."
- Vital sign gaps during transport/shift change are systematically different from random gaps.
- Free-text notes hold clinical info (allergies, social history) structured fields miss.

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

**Irregular time series:** Vitals every 4h on the ward, every 15min in ICU; labs drawn on clinical suspicion. Models need to handle variable-length, irregularly-timestamped sequences.

**ICD code heterogeneity:** ICD-9→10 transition (2015) breaks longitudinal continuity; coding practices vary by hospital (up/undercoding); ICD-10-CM's ~70,000 codes are usually grouped into ~286 CCS categories for modeling.

### Data Harmonization: OMOP CDM

The OHDSI OMOP model maps institution-specific codes to standard vocabularies — SNOMED CT (diagnoses), RxNorm (medications), LOINC (labs), CPT4/HCPCS (procedures) — enabling multi-site training without moving raw PHI.

### Selection Bias

- **Hospitalized-patient bias:** A model trained only on inpatient EHR data can't generalize to ambulatory settings.
- **Immortal time bias:** If the prediction window is "sepsis after hour 6" but features include hours 0–6, early deterioration signal leaks in and inflates apparent performance.
- **Label quality:** Sepsis-3 needs clinical judgment and is inconsistently coded retrospectively; death conflates severity with care quality; discharge diagnosis is coded too late for real-time labels. **Fix:** use composite endpoints (ICU transfer OR vasopressor use OR death within 24h) — objective and timely.

---

## 5. Model Architecture

Patient data is a multivariate, irregular time series mixing continuous vitals (q4h), irregular labs, medication events, and sparse diagnosis codes.

### Model Options

**LSTM for continuous monitoring:** handles variable-length sequences natively; forget gate discounts stale measurements; per-timestep output enables early warning. Weakness: struggles past ~200 time steps due to gradient decay.

**Transformer on irregular time series (preferred for new systems):** time-delta positional encoding (`PE(t) = sin(t/T_max)`) instead of standard position encoding; multi-head attention captures cross-variable correlations (rising HR + falling BP = instability). STraTS (2021) and ETHAN (2022) are purpose-built clinical transformer architectures; self-supervised pretraining on EHR corpora improves sample efficiency.

```python
class ClinicalTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 128, n_heads: int = 8):
        super().__init__()
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
        x = self.time_encoder(times) + self.value_encoder(values)
        x = self.transformer(x, src_key_padding_mask=mask)
        return torch.sigmoid(self.output_head(x[:, -1, :]))
```

**Gradient boosting for tabular features (discharge, drug interaction):** window-aggregated features (min/max/mean/last), handles mixed types natively, interpretable via SHAP, fast to iterate. XGBoost/LightGBM standard; CatBoost handles high-cardinality ICD codes well.

**Ensemble for robustness:** average transformer (temporal) + GBM (cross-sectional) scores. Large disagreement between them (e.g., 0.85 vs 0.30) is a useful signal to route to clinician review instead of auto-alerting.

### Radiology AI

DenseNet-121 (CheXNet, 2017) or a ViT-B/16 fine-tuned on CheXpert/MIMIC-CXR for chest X-ray classification. Multi-label output (pneumonia, atelectasis, cardiomegaly, effusion, pneumothorax — not mutually exclusive), plus a separate binary "critical finding" head to drive urgent triage. Preprocessing: DICOM→PNG, CLAHE normalization, 224×224 or 512×512.

---

## 6. Handling Class Imbalance

| Condition | Approx. prevalence |
|---|---|
| Sepsis (inpatients) | 2–4% |
| 30-day readmission | 12–15% |
| Pneumonia on chest X-ray (ED) | 5–10% |
| Rare genetic disease | 0.01–0.1% |
| Critical drug interaction | 1–3% of polypharmacy patients |

**Cost-sensitive learning:**
```python
model = XGBClassifier(
    scale_pos_weight=97,  # for 3% prevalence: 97/3
    eval_metric="aucpr",  # PR-AUC more informative than ROC-AUC under imbalance
)
```

**Focal loss (Lin et al., 2017):**
```python
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()
```

**Operating point selection:** never use the default 0.5 threshold. Set it from hospital policy, clinical utility (net benefit) analysis, and clinician-elicited sensitivity/specificity tradeoffs. A sepsis model at 0.5 threshold with 3% prevalence and 80% sensitivity yields ~40% PPV — 60% of alerts are false positives, and clinical teams stop trusting the system (the Epic Sepsis Model controversy, NEJM 2021).

**Net benefit:**
$$NB(\text{threshold}) = \frac{TP}{n} - \frac{FP}{n} \cdot \frac{p_t}{1 - p_t}$$

Plot NB vs threshold to compare model vs treat-all vs treat-none. FDA guidance favors this over ROC alone for clinical validation.

---

## 7. Fairness and Bias

**Known bias sources:**
- **Pulse oximetry:** SpO2 sensors overestimate saturation by 3–4 points in darker-skinned patients (Sjoding et al., NEJM 2020) — any model using SpO2 inherits this.
- **Dermatology AI:** underperforms on darker Fitzpatrick skin tones due to training data skew (Esteva et al., Nature 2017).
- **Sepsis models:** shown to have worse sensitivity for Black patients at the same threshold (Wong et al., 2021).
- **SES proxy features:** ZIP code, insurance type, prior admissions predict readmission well but encode race/SES bias.

**Required fairness analyses:**

| Analysis | Method | Threshold |
|---|---|---|
| Subgroup calibration | Reliability diagrams by race/sex/age | Calibration slope ∈ [0.8, 1.2] |
| Sensitivity/specificity parity | Report per demographic | Flag gap >5pp |
| Disparate impact ratio | min(TPR)/max(TPR) across groups | Must be >0.8 (4/5 rule) |
| Cross-site calibration | Hosmer-Lemeshow per hospital | p > 0.05 before new-site deployment |

FDA's 2022 action plan requires demographic performance analysis in AI/ML SaMD submissions.

**Mitigations:** oversample underrepresented groups and curate balanced validation sets (data-side); adversarial debiasing or demographic-parity constraints (model-side); group-specific thresholds (post-processing, legally complex); diverse clinical advisory board review (process-side).

---

## 8. Human-in-the-Loop Design

### Alert Fatigue

The Epic Sepsis Model (Wong et al., NEJM Evidence 2021) is the canonical cautionary case: AUROC 0.76 looked solid, but prospectively only 18% of alerts were accepted, 67% of sepsis cases were missed (33% sensitivity), and nurses habituated to ignoring alerts.

ICU nurses already see 150–200 alarms/patient/day; FPR >70% causes desensitization within weeks. At 3% prevalence, 90% sensitivity with 70% FPR means 2.3 false alerts per true one.

### Tiered Alerting

```
Score < 0.3:    No alert, logged only.
Score 0.3–0.5:  Passive dashboard indicator, no interruption.
Score 0.5–0.7:  Non-interruptive sidebar alert, one-click dismiss + reason code.
Score 0.7–0.85: Interruptive pop-up with SHAP explanation.
                Accept / Dismiss+reason / Escalate.
Score >0.85:    Critical alert — pages RN and attending,
                pre-populated order set, requires senior override to dismiss.
```

Target: <2 actionable alerts per nurse per 12h shift; auto-suppress if a unit's rate exceeds budget.

### Explainability

SHAP values, translated into clinical language, are standard for tabular models:

```python
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
            text = CLINICAL_EXPLANATIONS[feat].format(val=feature_values[feat])
            direction = "↑ risk" if shap_val > 0 else "↓ risk"
            explanations.append(f"{text} ({direction})")
    return explanations
```

Show calibrated confidence intervals, not just point estimates ("78% (CI 65–88%)" beats "ALERT: HIGH"). For imaging models, Grad-CAM heatmaps show which regions drove the prediction — needed for radiologist trust and catching spurious correlations (e.g., learning from a pacemaker rather than lung texture).

---

## 9. Validation and Monitoring

| Validation type | Description | When required |
|---|---|---|
| Internal | Train/test split, CV | Always |
| Temporal | Train on 2018–2020, test on 2021 | Required for FDA |
| External | Different hospital, unseen in training | Required for multi-site claims |
| Prospective (shadow) | Compare to ground truth post-hoc | Required before clinical deployment |
| Prospective RCT | Randomized exposure | Required for Class III/PMA |

Temporal validation is the minimum bar — random splits on longitudinal EHR data leak future information via retrospective coding and badly overstate performance.

**Site-specific calibration:** performance drops when moving to a new hospital (different acuity mix, protocols, EHR configuration). Recalibrate with isotonic regression or Platt scaling using 500–1,000 local patient-outcome pairs.

### Continuous Monitoring

```python
class ClinicalModelMonitor:
    def __init__(self, reference_df: pd.DataFrame, feature_cols: list):
        self.reference = reference_df
        self.features = feature_cols
        self.psi_threshold = 0.2
        self.calibration_threshold = 0.1  # max ECE

    def compute_psi(self, current_df: pd.DataFrame, feature: str) -> float:
        # <0.1 stable, 0.1-0.2 minor shift, >0.2 major shift
        ref_pcts, bins = np.histogram(self.reference[feature].dropna(), bins=10, density=True)
        cur_pcts, _ = np.histogram(current_df[feature].dropna(), bins=bins, density=True)
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
            results[group] = expected_calibration_error(sub["score"], sub["label"])
        return results
```

**Monitoring triggers:** PSI >0.2 on a top-10 feature → review; score-distribution KS p<0.001 → automatic pause; subgroup ECE >0.1 → equity review within 30 days; alert acceptance drop >20% → fatigue review; a new ICD-10 code appearing in >5% of encounters (e.g., COVID's U07.1) → behavior audit.

---

## 10. Deployment Constraints

### Model Registry / Audit Trail

```
{
  "model_id": "sepsis_v3.2.1",
  "sha256": "a8f3c2...",
  "training_data_hash": "b7e1d4...",
  "validation_auc": 0.821,
  "fda_submission": "K240123",
  "pccp_change_type": "minor_recalibration",
  "approved_by": ["CMO", "CISO", "FDA_reference"],
  "deployment_sites": ["MGH", "BWH"],
  "locked": true
}
```

Every inference logs: `{patient_token, model_id, timestamp, feature_hash, score, alert_tier_shown, clinician_id, action_taken, action_timestamp}`.

### Shadow Mode

Run 30–90 days before go-live: generate scores/would-be alerts without showing them, compare to actual outcomes and to existing tools (NEWS2, qSOFA, APACHE II), and measure sensitivity/PPV at the proposed threshold. No IRB needed (care isn't changed), but requires data access approval and a BAA.

### EHR Integration

CDS Hooks fire (e.g., `patient-view`), the model service fetches FHIR resources (`Observation`, `MedicationRequest`, `Condition`), scores, and returns a `cards` array with the alert.

**Graceful degradation:** if the feature pipeline fails, return "model unavailable" — never default to 0 risk. Fall back to rule-based alerts (SIRS, qSOFA) if the ML service is down. Circuit breaker: bypass ML if latency >5s. Never silently serve stale scores — show a "last updated" timestamp.

---

## 11. Failure Modes

**Alert fatigue:** high FPR → clinicians dismiss without reading → true positives missed. Mitigate with tiered alerting, alert budgets, suppression of duplicate alerts within a time window, and unit-level burden reporting. Detect via acceptance rate <15% for a tier.

**Distribution shift from protocol changes:** e.g., a lactate order-set change silently shifts feature distributions and degrades sensitivity. COVID-19 (March 2020) is the textbook example — admission criteria, protocols, and coding all shifted at once. Mitigate by tying model revalidation to clinical protocol change management and monitoring PSI weekly.

**Feedback loops:** the model changes the outcomes it was trained to predict — early intervention prevents sepsis, so the label becomes 0 for a genuinely high-risk patient, and retraining teaches the model those patterns are low-risk. Mitigate with a randomized holdout (5–10%, IRB-approved) withheld from alerts, and counterfactual logging of would-have-been scores.

**Demographic bias:** inherited from historically biased training data, widening health equity gaps via higher FN rates for marginalized groups. Mitigate with mandatory subgroup calibration checks and annual bias audits.

**EHR vendor lock-in:** FHIR implementations vary by vendor and version; multi-site deployment becomes a heavy integration effort. Mitigate by abstracting behind a FHIR facade, keeping HL7 v2 as fallback, and maintaining a synthetic-patient test suite per integration.

**Model ossification:** regulatory lock-in means a 2020 model running in 2025 reflects pre-COVID clinical patterns. Mitigate by filing a PCCP upfront for allowed update types and planning revalidation cycles (18–24 months).

---

## 12. Interview Questions

**Q1: Validation AUROC is 0.85 for a sepsis model. Is it ready for deployment?**

No. AUROC is threshold-agnostic and insensitive to calibration — a well-ranked but miscalibrated model isn't deployable. Validation must be temporal and external, since random splits on retrospectively-coded EHR data leak information. Subgroup performance must be checked — aggregate AUROC can hide large sensitivity gaps across demographics. And shadow-mode prospective validation is needed to measure real-world alert burden and acceptance. AUROC 0.85 is necessary, not sufficient.

**Q2: How do you handle missing labs? Population mean imputation?**

No — missingness is MNAR. A normal-looking WBC may be missing *because* the patient looked well; mean imputation biases scores toward "normal" for patients too sick to have labs drawn. Better: add missingness indicators and time-since-last-observation features; use attention masking for sequence models; let gradient boosting handle NaN natively; forward-fill only within a clinically meaningful window.

**Q3: A hospital wants to A/B test the sepsis model — one ICU gets alerts, the other doesn't. How?**

Needs IRB approval, since the control arm may get inferior care. The application must establish clinical equipoise, define a stopping rule, and justify a consent waiver. Expect 3–6 months for IRB. Faster alternatives: a pre/post study (confounded by temporal trends) or a cluster-randomized design by unit to reduce contamination.

**Q4: Six months post-launch, the model "never alerts anymore." What happened?**

Likely a feedback loop or distribution shift. Check: whether the alert rate dropped sharply and score distribution shifted (PSI); whether an EHR upgrade or protocol change altered feature availability; whether clinicians overrode thresholds locally; whether the ETL pipeline is silently failing and zeroing out features. Re-run the model on historical patients with known outcomes to isolate model drift from delivery-pipeline bugs.

**Q5: How do you validate radiology AI when the ask is "just compare to my reads"?**

Necessary but not sufficient. Inter-reader variability for chest X-ray is real (κ≈0.6–0.7), so use a multi-reader adjudicated panel, not one radiologist. Use consecutive real-world cases, not a curated interesting-case set (spectrum bias). The real question isn't "as good as a radiologist" but "does it improve the workflow" — measure time-to-read, missed-finding rate. Also require temporal/site validation and subgroup analysis by imaging equipment.

**Q6: How do you address the feedback loop where alerts change future training data?**

The intervention confounds its own label — early treatment prevents the outcome, so a genuinely high-risk case gets labeled negative. Mitigations: an IRB-approved randomized holdout that skips alerts for some patients; counterfactual/causal labeling; using process labels (antibiotics given) instead of outcome labels; upweighting untreated-but-positive cases during retraining. No perfect fix exists — ignoring it causes gradual degradation over 12–18 months.

**Q7: An advocacy group flags worse sensitivity for Black patients in your readmission model. What do you do?**

First confirm the disparity is statistically real. Audit features for race/SES proxies (ZIP code, insurance, prior no-shows) and check training data representation. Convene a clinical/ethics review with patient advocates. Consider group-specific recalibration (equity gain, legal complexity) and whether the gap reflects a real biased-care pattern the model is faithfully learning rather than a modeling artifact. Report per FDA post-market surveillance requirements, and publish — suppressing known bias creates liability and continued harm.

---

## References and Further Reading

- **Google Diabetic Retinopathy:** Gulshan et al., *JAMA* 2016; field deployment gap: Beede et al., CHI 2020.
- **Epic Sepsis Model controversy:** Wong et al., *NEJM Evidence* 2021 — AUROC 0.76 but 33% prospective sensitivity.
- **Pulse oximetry bias:** Sjoding et al., *NEJM* 2020.
- **Clinical utility curves:** Vickers & Elkin, *Medical Decision Making* 2006.
- **FDA AI/ML action plan:** FDA, January 2021 — PCCP framework, post-market surveillance.
- **OHDSI OMOP CDM:** ohdsi.org.
- **STraTS:** Tipirneni & Reddy, *AAAI* 2022.
- **Fairness in clinical AI:** Obermeyer et al., *Science* 2019.

## Flashcards

**FDA classification categories?** #flashcard
Class I (low risk, exempt) vs Class II (moderate, 510(k) or De Novo) vs Class III (high risk, PMA). Key question: does the model replace or augment clinical judgment?

**510(k) vs De Novo vs PMA?** #flashcard
510(k): predicate exists, ~90-day review. De Novo: novel device, no predicate, ~12-month review. PMA: Class III, full clinical trial required.

**PCCP (Predetermined Change Control Plan)?** #flashcard
FDA 2023 guidance letting vendors pre-specify permissible model updates (e.g., recalibration) without new submission.

**IEC 62304 / IEC 62366 / ISO 14971?** #flashcard
62304: medical device software lifecycle (safety classification, traceability). 62366: usability engineering. 14971: risk management across the lifecycle.

**HIPAA requirements for clinical ML?** #flashcard
PHI de-identification (Safe Harbor or Expert Determination), BAAs with cloud vendors, audit logs for all PHI access, minimum-necessary standard.

**Model lock-in and audit trail?** #flashcard
Validated model artifact is frozen; any weight change needs new submission unless under a PCCP. Every prediction logs timestamp, encrypted patient ID, model version, input hash, score, clinician action, outcome — retained 7+ years.

**Why no standard A/B testing in clinical ML?** #flashcard
Randomizing patients to model vs no-model usually requires IRB approval. Shadow mode (silent scoring, no alert shown) is the standard pre-launch approach instead.

**MNAR labs — why not mean imputation?** #flashcard
A normal-looking lab may be missing because the patient looked well (missing not at random). Mean imputation biases the model toward assuming normal values, suppressing risk scores for undertested sick patients.

**OMOP CDM vocabularies?** #flashcard
SNOMED CT (diagnoses), RxNorm (medications), LOINC (labs), CPT4/HCPCS (procedures) — standardizes codes across sites for multi-site training.

**Selection biases in clinical ML?** #flashcard
Hospitalized-patient bias (no ambulatory data) and immortal time bias (feature window overlaps outcome window, inflating apparent performance).

**LSTM vs Transformer for clinical time series?** #flashcard
LSTM: handles variable length natively, forget gate discounts stale data, weak past ~200 steps. Transformer: time-delta positional encoding, multi-head attention over cross-variable correlations, better for irregular long sequences (STraTS, ETHAN).

**Why ensemble transformer + GBM?** #flashcard
Transformer captures temporal patterns, GBM captures cross-sectional features; large disagreement between them (e.g. 0.85 vs 0.30) is a useful signal to route to clinician review instead of auto-alerting.

**Radiology AI standard architectures?** #flashcard
DenseNet-121 (CheXNet) or ViT-B/16 fine-tuned on CheXpert/MIMIC-CXR; multi-label output plus a separate critical-finding head for triage.

**Why not use 0.5 threshold for clinical alerts?** #flashcard
At 3% sepsis prevalence and 80% sensitivity, a 0.5 threshold gives ~40% PPV — 60% of alerts are false positives, causing clinicians to ignore the system (Epic Sepsis Model controversy).

**Net benefit / clinical utility curve?** #flashcard
NB(threshold) = TP/n − FP/n · pt/(1−pt). Compares model vs treat-all vs treat-none; FDA-preferred over ROC alone for clinical validation.

**Known bias sources in clinical AI?** #flashcard
Pulse oximetry overestimates SpO2 in darker-skinned patients (Sjoding 2020); dermatology AI underperforms on darker Fitzpatrick tones; sepsis models show worse sensitivity for Black patients; ZIP code/insurance are SES/race proxies.

**Required fairness analyses?** #flashcard
Subgroup calibration (slope 0.8-1.2), sensitivity/specificity parity (<5pp gap), disparate impact ratio (>0.8, 4/5 rule), cross-site calibration (Hosmer-Lemeshow p>0.05).

**Epic Sepsis Model controversy — key numbers?** #flashcard
AUROC 0.76 in validation, but prospectively only 18% of alerts accepted and 67% of sepsis cases missed (33% sensitivity) — alert fatigue from habituation.

**Tiered alerting architecture?** #flashcard
<0.3 no alert (logged); 0.3-0.5 passive dashboard; 0.5-0.7 non-interruptive sidebar; 0.7-0.85 interruptive pop-up with explanation; >0.85 critical page, requires senior override to dismiss.

**Explainability requirements for clinical models?** #flashcard
SHAP values translated to clinical language (not raw feature names); calibrated confidence intervals over point estimates; Grad-CAM heatmaps for imaging models to show region driving prediction and catch spurious correlations.

**Validation hierarchy for clinical models?** #flashcard
Internal (always) → temporal (required for FDA) → external/cross-site (multi-site claims) → prospective shadow mode (before deployment) → prospective RCT (Class III/PMA).

**Why is temporal validation the minimum bar?** #flashcard
Random splits on longitudinal EHR data leak future information via retrospective coding, severely overstating performance.

**Site-specific calibration fix?** #flashcard
Isotonic regression or Platt scaling recalibrates score-to-probability mapping at a new site using 500-1,000 local patient-outcome pairs.

**Key monitoring triggers?** #flashcard
PSI >0.2 on top feature → review; score KS p<0.001 → auto-pause; subgroup ECE >0.1 → equity review in 30 days; alert acceptance drop >20% → fatigue review; new ICD code >5% of encounters → behavior audit.

**Shadow mode deployment?** #flashcard
Run model 30-90 days generating scores without showing alerts; compare to outcomes and existing tools (NEWS2, qSOFA, APACHE II). No IRB needed since care isn't changed, but needs data access approval and a BAA.

**Graceful degradation for clinical ML services?** #flashcard
On pipeline failure return "model unavailable," never default to 0 risk. Fall back to rule-based alerts (SIRS/qSOFA) if ML service is down. Circuit breaker bypasses ML if latency >5s. Never silently serve stale scores.

**Feedback loop failure mode?** #flashcard
Model alerts → clinicians intervene early → outcome doesn't occur → label becomes negative for a genuinely high-risk case → retraining teaches the model those patterns are low-risk. Mitigate with IRB-approved randomized holdout and counterfactual logging.

**Model ossification failure mode?** #flashcard
Regulatory lock-in prevents updates without re-submission, so an old model reflects outdated clinical patterns (e.g., pre-COVID). Mitigate with an upfront PCCP and planned revalidation cycles (18-24 months).

**EHR vendor lock-in mitigation?** #flashcard
Abstract behind a FHIR facade, keep HL7 v2 as fallback, maintain a synthetic-patient test suite per integration, budget dedicated integration engineering for multi-site rollouts.
