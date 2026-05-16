# Fairness and Bias in Machine Learning

A systematic treatment of bias sources, fairness metrics, mitigation strategies, and practical tooling for building equitable ML systems.

---

## 1. Sources of Bias

### Historical Bias
Data reflects past discriminatory decisions or societal inequities, even when collected perfectly. A hiring model trained on historical résumé outcomes encodes past discrimination in its predictions.

### Representation Bias
Certain subgroups appear far less frequently in training data than in the deployment population, so the model optimizes poorly for those groups. Facial recognition datasets historically over-represent lighter-skinned males, degrading accuracy for darker-skinned females.

### Measurement Bias
Proxy variables are used in place of the true quantity of interest, and the proxy correlates unevenly with the sensitive attribute. Using zip code as a proxy for creditworthiness encodes race due to residential segregation.

### Aggregation Bias
A single model is fit to a heterogeneous population when subgroup relationships differ. A diabetes risk model trained on aggregated data may fit majority populations well while systematically underperforming for minority groups whose physiology or access-to-care patterns differ.

### Deployment Bias
The environment in which the model is used differs from the training context. A risk-assessment tool trained on defendants in one jurisdiction is deployed in another with different demographic composition or judicial norms.

---

## 2. Fairness Metrics

### Individual vs. Group Fairness

**Individual fairness:** similar individuals should receive similar predictions.  
Formally, a model `f` is individually fair if for a task-appropriate similarity metric `d`, `D(f(x), f(x')) ≤ L · d(x, x')` for all pairs `(x, x')`.

**Group fairness:** statistical parity conditions hold across protected groups (defined by sensitive attribute `A`).

---

### Demographic Parity (Statistical Parity)

The positive prediction rate is equal across groups:

```
P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)
```

- Pros: easy to audit; directly targets representation.
- Cons: ignores true label distribution; can enforce equal errors on groups with different base rates.

---

### Equal Opportunity

True positive rates are equal across groups:

```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)
```

Favored when false negatives are the primary harm (e.g., loan denial when creditworthy).

---

### Equalized Odds

Both TPR and FPR are equal across groups:

```
P(Ŷ = 1 | Y = y, A = 0) = P(Ŷ = 1 | Y = y, A = 1)   for y ∈ {0, 1}
```

Stricter than equal opportunity; requires error parity in both directions.

---

### Calibration

The predicted probability reflects the true outcome probability equally across groups:

```
P(Y = 1 | score = s, A = 0) = P(Y = 1 | score = s, A = 1)   ∀ s
```

Critical in risk scoring (recidivism, medical risk) where score interpretation must be consistent.

---

### Counterfactual Fairness

A model is counterfactually fair if, in the counterfactual world where `A` had been different (all else equal), the prediction does not change:

```
P(Ŷ_{A←a}(U) = y | X = x, A = a) = P(Ŷ_{A←a'}(U) = y | X = x, A = a)
```

Requires a causal model; moves beyond correlation-based auditing.

---

### Impossibility Theorem

Chouldechova (2017) and Kleinberg et al. (2016) proved that **demographic parity, equal opportunity, and calibration cannot all hold simultaneously** when base rates differ across groups, except in degenerate cases (perfect classifier or equal base rates). Any fairness criterion choice is a value judgment about which errors matter most.

---

## 3. Pre-processing Methods

### Reweighting
Assign instance weights inversely proportional to group-label frequency so that the weighted dataset is balanced:

```python
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

rw = Reweighing(
    unprivileged_groups=[{"race": 0}],
    privileged_groups=[{"race": 1}]
)
dataset_transf = rw.fit_transform(dataset)
```

### Resampling
Oversample underrepresented (group, label) combinations or undersample overrepresented ones before training. Analogous to class-imbalance techniques but applied at the group × label level.

### Learning Fair Representations (LFR)
Zemel et al. (2013) learn a latent representation `Z` that is maximally predictive of `Y` while being statistically independent of `A`. The encoder is trained with a combined objective:

```
L = L_prediction + λ₁ · L_fairness + λ₂ · L_reconstruction
```

`L_fairness` penalizes statistical dependence between `Z` and `A` (e.g., via MMD or adversarial loss).

### Disparate Impact Remover
Feldman et al. (2015) transform feature distributions so they become identical across groups while preserving rank ordering within groups:

```python
from aif360.algorithms.preprocessing import DisparateImpactRemover

di = DisparateImpactRemover(repair_level=1.0, sensitive_attribute="race")
dataset_transf = di.fit_transform(dataset)
```

`repair_level` in [0, 1] controls the strength of transformation.

---

## 4. In-processing Methods

### Adversarial Debiasing
Zhang et al. (2018): a predictor network `P` predicts label `Ŷ` from features; an adversary network `A` tries to predict the sensitive attribute from `P`'s intermediate representation. `P` is trained to fool the adversary while remaining accurate:

```
L_P = L_prediction - λ · L_adversary
L_A = L_adversary
```

```python
from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow.compat.v1 as tf

sess = tf.Session()
debiased_model = AdversarialDebiasing(
    privileged_groups=[{"sex": 1}],
    unprivileged_groups=[{"sex": 0}],
    scope_name="debiased_classifier",
    debias=True,
    sess=sess
)
debiased_model.fit(dataset_train)
```

### Constraint-based Optimization
Add fairness constraints directly to the training objective via Lagrangian relaxation:

```
min_{θ} L(θ) + λ · |fairness_violation(θ)|
```

Constraints can enforce demographic parity, equalized odds, or bounded group loss gaps. The Lagrange multiplier `λ` is tuned to trade accuracy for fairness.

### Fairlearn Reductions Approach
Agarwal et al. (2018): reduce constrained fairness optimization to a sequence of weighted classification problems, iterating over an exponentiated gradient or grid search over Lagrange multipliers:

```python
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression()
constraint = EqualizedOdds()

mitigator = ExponentiatedGradient(estimator, constraint)
mitigator.fit(X_train, y_train, sensitive_features=A_train)
y_pred = mitigator.predict(X_test)
```

---

## 5. Post-processing Methods

### Threshold Calibration per Group (Equalized Odds Post-processing)
Hardt et al. (2016): given a score from any classifier, find group-specific thresholds `τ_a` that minimize a fairness objective. For equalized odds, solve a linear program over the ROC curves of each group:

```python
from fairlearn.postprocessing import ThresholdOptimizer

postprocess_est = ThresholdOptimizer(
    estimator=base_model,
    constraints="equalized_odds",
    objective="balanced_accuracy_score",
    predict_method="predict_proba"
)
postprocess_est.fit(X_train, y_train, sensitive_features=A_train)
y_pred = postprocess_est.predict(X_test, sensitive_features=A_test)
```

### Reject Option Classification
In the "reject option" zone near the decision boundary (low-confidence predictions), flip decisions in favor of the unprivileged group. Kamiran et al. (2012):

```python
from aif360.algorithms.postprocessing import RejectOptionClassification

roc = RejectOptionClassification(
    unprivileged_groups=[{"race": 0}],
    privileged_groups=[{"race": 1}],
    metric_name="Equal opportunity difference",
    metric_ub=0.05,
    metric_lb=-0.05
)
roc = roc.fit(dataset_val, dataset_val_pred)
dataset_transf_pred = roc.predict(dataset_test_pred)
```

### Platt Scaling per Group
Fit a separate logistic regression calibrator on each group's validation scores to ensure calibrated probabilities are consistent across groups before applying a shared threshold:

```python
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

# calibrate separately per group
calibrated = {}
for group_val in [0, 1]:
    mask = (A_val == group_val)
    cal = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    cal.fit(X_val[mask], y_val[mask])
    calibrated[group_val] = cal

# predict using group-specific calibrator
y_prob = np.where(
    A_test == 0,
    calibrated[0].predict_proba(X_test)[:, 1],
    calibrated[1].predict_proba(X_test)[:, 1]
)
```

---

## 6. Bias Detection

### Disparate Impact Ratio
The ratio of positive prediction rates across groups. The EEOC "80% rule" (four-fifths rule) flags adverse impact when:

```
DI = P(Ŷ = 1 | A = 0) / P(Ŷ = 1 | A = 1) < 0.8
```

```python
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

# On predictions
metric = ClassificationMetric(
    dataset_true,
    dataset_pred,
    unprivileged_groups=[{"race": 0}],
    privileged_groups=[{"race": 1}]
)

print("Disparate Impact:          ", metric.disparate_impact())
print("Equal Opportunity Diff:    ", metric.equal_opportunity_difference())
print("Average Odds Difference:   ", metric.average_odds_difference())
print("Statistical Parity Diff:   ", metric.statistical_parity_difference())
print("Theil Index:               ", metric.theil_index())
```

`statistical_parity_difference` = 0 is ideal; values < 0 indicate disadvantage for the unprivileged group.

### Slice-based Evaluation
Evaluate metrics (accuracy, F1, calibration error) separately for every subgroup and subgroup intersection. Tools like `SliceFinder` (Chung et al., 2019) or `What-If Tool` automate discovery of underperforming slices:

```python
import pandas as pd

results = []
for group in df["race"].unique():
    mask = df["race"] == group
    acc = (y_pred[mask] == y_true[mask]).mean()
    results.append({"group": group, "n": mask.sum(), "accuracy": acc})

pd.DataFrame(results).sort_values("accuracy")
```

### AIF360 End-to-End Audit

```python
from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric

dataset = AdultDataset(
    protected_attribute_names=["sex"],
    privileged_classes=[["Male"]],
    categorical_features=[],
    features_to_keep=["age", "education-num", "hours-per-week", "sex", "income-per-year"]
)

metric_orig = BinaryLabelDatasetMetric(
    dataset,
    unprivileged_groups=[{"sex": 0}],
    privileged_groups=[{"sex": 1}]
)
print("Base rate difference:", metric_orig.mean_difference())
print("Disparate impact:    ", metric_orig.disparate_impact())
```

---

## 7. Intersectionality

Single-attribute fairness audits miss compounding effects. A model may satisfy demographic parity for gender and for race individually while systematically disadvantaging Black women (race × gender intersection).

Key considerations:
- **Data sparsity** at intersections makes metrics unreliable; confidence intervals widen.
- **Gerrymandering** in fairness constraints (Kearns et al., 2018) formalizes the requirement that constraints hold for all subgroups defined by conjunctions of protected attributes.
- **Auditing practice:** enumerate all meaningful intersections, flag cells with `n < 50` as statistically unreliable, and report both point estimates and confidence intervals.

```python
# Slice evaluation across intersections
for race in df["race"].unique():
    for sex in df["sex"].unique():
        mask = (df["race"] == race) & (df["sex"] == sex)
        if mask.sum() < 50:
            continue
        tpr = ((y_pred[mask] == 1) & (y_true[mask] == 1)).sum() / (y_true[mask] == 1).sum()
        print(f"race={race}, sex={sex}, n={mask.sum()}, TPR={tpr:.3f}")
```

---

## 8. LLM-specific Bias

### Stereotyping in Generation
LLMs generate text that associates professions, traits, and behaviors with demographic groups in ways that reflect and amplify training corpus biases. "A doctor walked in. He..." vs. "A nurse walked in. She..." exhibit occupational gender stereotypes.

### Benchmarks

**WinoBias** (Zhao et al., 2018): coreference resolution sentences where the correct antecedent requires ignoring occupational gender stereotypes. Models are scored on whether they resolve references consistently regardless of gender.

**BBQ** (Parrish et al., 2022): Bias Benchmark for QA. Questions with ambiguous contexts paired with questions that have definite answers. Measures whether models default to stereotyped responses under ambiguity across nine social dimensions (age, disability, gender, nationality, physical appearance, race/ethnicity, religion, SES, sexual orientation).

### RLHF Bias Amplification
Reinforcement learning from human feedback can amplify rater biases:
- Raters may prefer fluent, confident-sounding text even when content is stereotyped.
- Rater pools are demographically non-representative, encoding majority-group preferences.
- Reward models trained on preference data can encode systematic biases that are then optimized into the policy model.

Mitigation approaches: diverse rater recruitment, rater calibration training, red-teaming specifically for demographic bias, constitutional AI approaches that include explicit anti-bias principles.

---

## 9. Fairness in Practice

### Accuracy–Fairness Tradeoff
Enforcing fairness constraints almost always reduces accuracy on the majority group or overall. The Pareto frontier of (accuracy, fairness) can be traced by sweeping the constraint bound or Lagrange multiplier. Decision makers must choose a point on this frontier — a values decision, not a technical one.

```python
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score
import numpy as np

epsilons = [0.01, 0.05, 0.10, 0.20]
results = []
for eps in epsilons:
    mitigator = ExponentiatedGradient(
        LogisticRegression(),
        DemographicParity(difference_bound=eps)
    )
    mitigator.fit(X_train, y_train, sensitive_features=A_train)
    y_pred = mitigator.predict(X_test)
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=A_test
    )
    results.append({
        "epsilon": eps,
        "overall_accuracy": mf.overall["accuracy"],
        "group_accuracy_gap": mf.difference()["accuracy"]
    })
```

### Regulatory Context

**EU AI Act (2024):** High-risk AI systems (hiring, credit, education, law enforcement) must undergo conformity assessment including bias audits, maintain technical documentation, and support human oversight. Prohibits subliminal manipulation and social scoring.

**EEOC / Uniform Guidelines on Employee Selection Procedures (US):** Adverse impact analysis required; the four-fifths (80%) rule is the operational threshold. Employers bear burden to demonstrate job-relatedness for any selection procedure with adverse impact.

**Fair Housing Act / ECOA (US):** Prohibit discriminatory outcomes in housing and credit regardless of intent; disparate impact is actionable.

### Documentation Standards

**Model Cards** (Mitchell et al., 2019): Structured document accompanying a model release. Required sections: model details, intended use, factors (relevant demographic groups), metrics (with disaggregated evaluation), evaluation data, training data, ethical considerations, caveats and recommendations.

**Datasheets for Datasets** (Gebru et al., 2021): Analogous documentation for datasets. Covers motivation, composition, collection process, preprocessing, uses, distribution, maintenance. Enables downstream practitioners to assess fitness for use and potential for harm.

---

## 10. Key Interview Points

**Bias sources are upstream of the model.** Historical and representation bias exist in data before any model is trained; preprocessing and documentation are first-order mitigations, not model architecture choices.

**No fairness criterion is universally correct.** Demographic parity, equal opportunity, equalized odds, and calibration formalize different normative positions. The impossibility theorem means you must choose — make the tradeoff explicit.

**Disparate impact ratio below 0.8 is a legal threshold, not a fairness definition.** It is a rough heuristic from employment law; real audits require the full set of metrics and context.

**Pre/in/post-processing form a layered strategy.** Pre-processing is most flexible; in-processing provides tighter constraint enforcement; post-processing can be applied to any black-box model but is limited by the original score quality.

**Intersectionality requires explicit enumeration.** Auditing sex and race separately does not reveal compounding harms. Flag low-sample intersections and report confidence intervals.

**LLMs require benchmark-specific bias evaluation.** Standard classification fairness metrics do not transfer directly; WinoBias, BBQ, and red-teaming prompts are the operational tools.

**Fairness is a sociotechnical problem.** Metrics and code address necessary but not sufficient conditions. Stakeholder engagement, impact assessments, ongoing monitoring, and organizational accountability structures are equally essential.

**Regulatory compliance is a floor, not a ceiling.** The EU AI Act and EEOC rules set minimum requirements; responsible deployment requires substantive commitment beyond legal compliance.
