---
module: Emerging Topics
topic: Fairness And Bias
subtopic: ""
status: unread
tags: [emergingtopics, ml, fairness-and-bias]
---
# Fairness and Bias in Machine Learning

---

## 1. The Problem

In 2016, ProPublica analyzed COMPAS, a recidivism prediction tool used by US courts to inform bail and sentencing decisions. They found that Black defendants who did not reoffend were flagged as high-risk at nearly twice the rate of white defendants who did not reoffend. At the same time, white defendants who did reoffend were more often classified as low-risk.

When Northpointe (COMPAS's maker) responded, they showed that COMPAS was calibrated: among all defendants scored as high-risk, the fraction who actually reoffended was similar across races. Both claims were true simultaneously.

This is not a paradox. It is a mathematical theorem.

When base rates differ across groups — when one group has a higher historical recidivism rate due to structural factors like differential policing and prosecution — no model can simultaneously equalize false positive rates, equalize false negative rates, and remain calibrated. You must choose which error to minimize. That choice is a values decision, not a technical one. COMPAS made that choice implicitly, embedded in a black-box score used to determine human liberty.

This is the bias and fairness problem in ML: **models encode normative choices about which errors matter, but those choices are invisible when the model is treated as an objective algorithm.** And at scale, models encoding bias amplify it. A hiring algorithm trained on historical résumés learns that "women's college" is a negative signal. A facial recognition system trained on non-representative data achieves 99% accuracy on light-skinned males and 65% on dark-skinned females. A credit model using zip code encodes residential segregation.

The damage is documented, at scale, affecting real people's access to credit, employment, healthcare, and freedom.

---

## 2. Where Bias Enters

Bias is not one thing that happens at one place. It accumulates at every stage of the ML pipeline.

**Historical bias:** Data reflects past discriminatory decisions, even when collected perfectly. A hiring model trained on historical résumé outcomes encodes past discrimination, not future merit.

**Representation bias:** Certain subgroups appear far less frequently in training data than in the deployment population. The model optimizes poorly for those groups. ImageNet-scale facial recognition datasets have historically over-represented lighter-skinned males.

**Measurement bias:** Proxy variables correlate unevenly with the sensitive attribute. Using zip code as a proxy for creditworthiness encodes race due to residential segregation. The proxy is measuring the wrong thing for some groups.

**Aggregation bias:** A single model fit to a heterogeneous population when subgroup relationships differ. A diabetes risk model trained on aggregated data may systematically underperform for minority groups whose physiology or healthcare access patterns differ.

**Deployment bias:** The deployment context differs from the training context. A risk-assessment tool trained in one jurisdiction is applied in another with different demographic composition or judicial norms.

---

## 3. Fairness Metrics

The COMPAS case reveals that different stakeholders were using different implicit fairness criteria and each was correct under their own. The field has formalized these criteria to make the choice explicit.

### The Core Question Each Metric Asks

Each metric starts from a different answer to the question: "what kind of error is most harmful?"

---

### Demographic Parity (Statistical Parity)

**Question:** are favorable outcomes distributed equally across groups?

```
P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)
```

The positive prediction rate is equal across groups, regardless of true labels.

Appropriate when: access to a benefit (loans, job interviews) should be proportional to group size in the population. Problematic when: base rates genuinely differ and equalizing prediction rates means making more errors for one group than the other.

---

### Equal Opportunity

**Question:** if you're qualified, do you have an equal chance of being identified as such?

```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)
```

True positive rates are equal across groups. Focuses on false negatives: the harm of denying a benefit to a qualified person.

Appropriate when: false negatives are the primary harm — a creditworthy person denied a loan, a qualified candidate screened out.

---

### Equalized Odds

**Question:** do both types of errors — false positives and false negatives — occur at equal rates across groups?

```
P(Ŷ = 1 | Y = y, A = 0) = P(Ŷ = 1 | Y = y, A = 1)   for y ∈ {0, 1}
```

Both TPR and FPR are equal across groups. This is what equalizing the COMPAS false positive rates would have required.

Stricter than equal opportunity. Requires symmetry of error across groups.

---

### Calibration

**Question:** does a predicted probability of 70% mean the same thing for every group?

```
P(Y = 1 | score = s, A = 0) = P(Y = 1 | score = s, A = 1)   ∀ s
```

The predicted probability reflects the true outcome probability equally across groups. This is what Northpointe showed COMPAS had. Critical in risk scoring where the score's interpretation must be consistent — a 70% recidivism risk score must mean 70% regardless of race.

---

### Counterfactual Fairness

**Question:** if a person's demographic had been different, would the prediction change?

```
P(Ŷ_{A←a}(U) = y | X = x, A = a) = P(Ŷ_{A←a'}(U) = y | X = x, A = a)
```

A model is counterfactually fair if, in a world where `A` had been different (all else equal), the prediction does not change. Requires a causal model. Moves beyond correlation-based auditing to ask whether the sensitive attribute has a causal effect on the prediction.

---

### Individual Fairness

**Question:** do similar individuals receive similar predictions?

```
D(f(x), f(x')) ≤ L · d(x, x')   for all pairs (x, x')
```

Where `d` is a task-appropriate similarity metric. Hard to operationalize — "similar" requires a definition that is itself normative.

---

### The Impossibility Theorem

Chouldechova (2017) and Kleinberg et al. (2016) proved formally what the COMPAS debate showed empirically: **demographic parity, equalized odds (equal FPR and FNR), and calibration cannot all hold simultaneously when base rates differ across groups**, except when the classifier is perfect or base rates are identical.

The arithmetic is direct. With different base rates, if you equalize the FPR across groups, the FNR must differ. If you equalize both, calibration breaks. Any fairness criterion is a normative choice about which errors matter most.

The harm of COMPAS was not that someone chose the wrong criterion. It was that no one acknowledged that a choice was being made. When a model is deployed as an objective algorithm, the embedded values become invisible — and unaccountable.

---

## 4. Pre-processing Methods

Intervene on the training data before the model sees it.

### Reweighting

Assign instance weights inversely proportional to group-label frequency so the weighted dataset is balanced:

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

Oversample underrepresented (group, label) combinations or undersample overrepresented ones. Analogous to class-imbalance techniques applied at the group × label level.

### Learning Fair Representations (LFR)

Zemel et al. (2013): learn a latent representation `Z` that is predictive of `Y` while being statistically independent of `A`. The encoder is trained with:

```
L = L_prediction + λ₁ · L_fairness + λ₂ · L_reconstruction
```

`L_fairness` penalizes statistical dependence between `Z` and `A` via MMD or adversarial loss.

### Disparate Impact Remover

Transform feature distributions to be identical across groups while preserving rank ordering within groups:

```python
from aif360.algorithms.preprocessing import DisparateImpactRemover

di = DisparateImpactRemover(repair_level=1.0, sensitive_attribute="race")
dataset_transf = di.fit_transform(dataset)
```

`repair_level` in [0, 1] controls transformation strength.

---

## 5. In-processing Methods

Intervene during training by encoding fairness constraints into the learning objective.

### Adversarial Debiasing

Zhang et al. (2018): a predictor network predicts label `Ŷ` from features; an adversary network tries to predict the sensitive attribute from the predictor's intermediate representation. The predictor is trained to fool the adversary while remaining accurate:

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

Add fairness constraints to the training objective via Lagrangian relaxation:

```
min_{θ} L(θ) + λ · |fairness_violation(θ)|
```

Constraints can enforce demographic parity, equalized odds, or bounded group loss gaps. λ trades accuracy for fairness.

### Fairlearn Reductions Approach

Agarwal et al. (2018): reduce constrained fairness optimization to a sequence of weighted classification problems:

```python
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from sklearn.linear_model import LogisticRegression

mitigator = ExponentiatedGradient(LogisticRegression(), EqualizedOdds())
mitigator.fit(X_train, y_train, sensitive_features=A_train)
y_pred = mitigator.predict(X_test)
```

---

## 6. Post-processing Methods

Intervene on model outputs after training — useful when the model is a black box.

### Threshold Calibration per Group (Hardt et al., 2016)

Given a score from any classifier, find group-specific thresholds `τ_a` that satisfy a fairness objective. For equalized odds, solve a linear program over the ROC curves of each group:

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

In the low-confidence zone near the decision boundary, flip decisions in favor of the unprivileged group:

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

Fit separate logistic regression calibrators on each group's validation scores:

```python
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

calibrated = {}
for group_val in [0, 1]:
    mask = (A_val == group_val)
    cal = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    cal.fit(X_val[mask], y_val[mask])
    calibrated[group_val] = cal

y_prob = np.where(
    A_test == 0,
    calibrated[0].predict_proba(X_test)[:, 1],
    calibrated[1].predict_proba(X_test)[:, 1]
)
```

---

## 7. Bias Detection

### Disparate Impact Ratio

The ratio of positive prediction rates across groups. The EEOC "80% rule" flags adverse impact when:

```
DI = P(Ŷ = 1 | A = 0) / P(Ŷ = 1 | A = 1) < 0.8
```

```python
from aif360.metrics import ClassificationMetric

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

`statistical_parity_difference` = 0 is ideal; negative values indicate disadvantage for the unprivileged group.

### Slice-based Evaluation

Evaluate metrics separately for every subgroup:

```python
import pandas as pd

results = []
for group in df["race"].unique():
    mask = df["race"] == group
    acc = (y_pred[mask] == y_true[mask]).mean()
    results.append({"group": group, "n": mask.sum(), "accuracy": acc})

pd.DataFrame(results).sort_values("accuracy")
```

---

## 8. Intersectionality

Single-attribute fairness audits miss compounding effects. A model may satisfy demographic parity for gender and for race individually while systematically disadvantaging Black women — the race × gender intersection is not captured by either single-attribute audit.

Kearns et al. (2018): fairness constraints must hold for all subgroups defined by conjunctions of protected attributes, not just each attribute marginally.

Key issues:
- **Data sparsity:** confidence intervals widen at intersections; cells with `n < 50` are statistically unreliable
- **Gerrymandering:** an adversary can find a subgroup that violates fairness even when all marginal audits pass

```python
for race in df["race"].unique():
    for sex in df["sex"].unique():
        mask = (df["race"] == race) & (df["sex"] == sex)
        if mask.sum() < 50:
            continue
        tpr = ((y_pred[mask] == 1) & (y_true[mask] == 1)).sum() / (y_true[mask] == 1).sum()
        print(f"race={race}, sex={sex}, n={mask.sum()}, TPR={tpr:.3f}")
```

---

## 9. LLM-Specific Bias

### Stereotyping in Generation

LLMs generate text that associates professions, traits, and behaviors with demographic groups in ways that reflect and amplify training corpus biases. "A doctor walked in. He..." vs. "A nurse walked in. She..." exhibit occupational gender stereotypes.

### Benchmarks

**WinoBias** (Zhao et al., 2018): coreference resolution sentences where the correct antecedent requires ignoring occupational gender stereotypes. Models are scored on whether they resolve references consistently regardless of gender.

**BBQ** (Parrish et al., 2022): Bias Benchmark for QA. Questions with ambiguous contexts paired with questions that have definite answers. Measures whether models default to stereotyped responses under ambiguity across nine social dimensions (age, disability, gender, nationality, physical appearance, race/ethnicity, religion, SES, sexual orientation).

### RLHF Bias Amplification

RLHF can amplify rater biases: raters may prefer fluent, confident-sounding text even when content is stereotyped; rater pools are demographically non-representative; reward models trained on preference data encode systematic biases that are then optimized into the policy.

Mitigations: diverse rater recruitment, rater calibration training, red-teaming for demographic bias, constitutional AI approaches with explicit anti-bias principles.

---

## 10. Fairness in Practice

### The Accuracy-Fairness Tradeoff

Enforcing fairness constraints almost always reduces accuracy on the majority group or overall. The Pareto frontier of (accuracy, fairness) is traced by sweeping the constraint bound:

```python
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score

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

The operating point on this frontier is a values decision, not a technical one.

### Regulatory Context

**EU AI Act (2024):** High-risk AI systems (hiring, credit, education, law enforcement) must undergo conformity assessment including bias audits, maintain technical documentation, and support human oversight. Prohibits subliminal manipulation and social scoring.

**EEOC / Uniform Guidelines on Employee Selection Procedures (US):** Adverse impact analysis required; the four-fifths (80%) rule is the operational threshold. Employers bear the burden to demonstrate job-relatedness for any selection procedure with adverse impact.

**Fair Housing Act / ECOA (US):** Prohibit discriminatory outcomes in housing and credit regardless of intent. Disparate impact is actionable.

### Documentation Standards

**Model Cards** (Mitchell et al., 2019): model details, intended use, factors (relevant demographic groups), metrics with disaggregated evaluation, evaluation data, training data, ethical considerations, caveats.

**Datasheets for Datasets** (Gebru et al., 2021): motivation, composition, collection process, preprocessing, uses, distribution, maintenance.

---

## 11. What Breaks

**The impossibility theorem is a hard constraint, not a research gap.** With unequal base rates, you cannot simultaneously satisfy demographic parity, equalized odds, and calibration. Any system claiming all three in the general case is wrong. The choice must be explicit and documented.

**Pre-processing can balance distributions without fixing measurement bias.** Reweighting balances training data but cannot fix a feature that encodes race via zip code. The bias is in the measurement, not the distribution.

**Post-processing requires sensitive attributes at inference.** Threshold calibration per group and reject-option classification need `A` at prediction time. Many deployments prohibit collecting or using this.

**Marginal fairness audits miss intersection harms.** Passing separate audits for gender and race provides no guarantee about Black women. Low sample sizes at intersections also make confidence intervals very wide.

**LLM fairness metrics from classification do not transfer.** Disparate impact ratio and equalized odds measure discrete prediction properties. Generation bias requires benchmark-specific evaluation and red-teaming.

**Regulatory compliance is a floor, not a ceiling.** The 80% rule is a 1978 employment law heuristic. Passing a disparate impact test does not mean the system is fair.

---

## Key Interview Points

- The COMPAS case is the canonical example: both ProPublica and Northpointe were correct under different criteria. The impossibility theorem (Chouldechova 2017, Kleinberg 2016) proves that with unequal base rates, you cannot equalize FPR and maintain calibration simultaneously.
- No fairness criterion is universally correct. Demographic parity, equal opportunity, equalized odds, and calibration formalize different normative positions. The tradeoff must be made explicitly — the harm of COMPAS was not picking the wrong criterion, it was making the choice invisible.
- Bias sources are upstream of the model. Historical bias, representation bias, measurement bias, and aggregation bias all exist before training begins.
- Disparate impact ratio below 0.8 is a legal threshold from employment law, not a definition of fairness. Real audits require the full metric suite plus context.
- Pre/in/post-processing form a layered strategy: pre-processing is most flexible; in-processing provides tighter constraint enforcement; post-processing applies to any black-box model but requires sensitive attributes at inference.
- Intersectionality requires explicit enumeration. Auditing sex and race separately does not reveal compounding harms. Flag low-sample intersections (n < 50) and report confidence intervals.
- LLMs require benchmark-specific bias evaluation. WinoBias, BBQ, and red-teaming are the operational tools.

## Flashcards

**Data sparsity?** #flashcard
confidence intervals widen at intersections; cells with n < 50 are statistically unreliable

**Gerrymandering?** #flashcard
an adversary can find a subgroup that violates fairness even when all marginal audits pass

**The COMPAS case is the canonical example?** #flashcard
both ProPublica and Northpointe were correct under different criteria. The impossibility theorem (Chouldechova 2017, Kleinberg 2016) proves that with unequal base rates, you cannot equalize FPR and maintain calibration simultaneously.

**No fairness criterion is universally correct. Demographic parity, equal opportunity, equalized odds, and calibration formalize different normative positions. The tradeoff must be made explicitly?** #flashcard
the harm of COMPAS was not picking the wrong criterion, it was making the choice invisible.

**Bias sources are upstream of the model. Historical bias, representation bias, measurement bias, and aggregation bias all exist before training begins.?** #flashcard
Bias sources are upstream of the model. Historical bias, representation bias, measurement bias, and aggregation bias all exist before training begins.

**Disparate impact ratio below 0.8 is a legal threshold from employment law, not a definition of fairness. Real audits require the full metric suite plus context.?** #flashcard
Disparate impact ratio below 0.8 is a legal threshold from employment law, not a definition of fairness. Real audits require the full metric suite plus context.

**Pre/in/post-processing form a layered strategy?** #flashcard
pre-processing is most flexible; in-processing provides tighter constraint enforcement; post-processing applies to any black-box model but requires sensitive attributes at inference.

**Intersectionality requires explicit enumeration. Auditing sex and race separately does not reveal compounding harms. Flag low-sample intersections (n < 50) and report confidence intervals.?** #flashcard
Intersectionality requires explicit enumeration. Auditing sex and race separately does not reveal compounding harms. Flag low-sample intersections (n < 50) and report confidence intervals.

**LLMs require benchmark-specific bias evaluation. WinoBias, BBQ, and red-teaming are the operational tools.?** #flashcard
LLMs require benchmark-specific bias evaluation. WinoBias, BBQ, and red-teaming are the operational tools.
