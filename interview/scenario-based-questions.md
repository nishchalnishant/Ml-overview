# Scenario-Based Questions

This file is the practical one.

It is less:

- "define entropy"

and more:

- "your model fell over on Tuesday, now what?"

That is why it matters.

---

## 1. Accuracy Dropped Overnight

Check:

- data drift
- feature pipeline changes
- label definition changes
- upstream outages

Do not start by changing the architecture.

---

## 2. Training Loss Down, Validation Loss Up

Usually overfitting.

Common fixes:

- regularization
- more data
- early stopping
- simpler model

---

## 3. Both Training and Validation Are Poor

Usually underfitting or bad features.

Common fixes:

- stronger model
- better features
- more training
- less regularization

---

## 4. Works in Test, Fails in Prod

Look for:

- leakage
- train-serve skew
- drift
- threshold mismatch

That is the high-value checklist.

---

## 5. 99% Accuracy but Useless

Probably class imbalance.

Switch attention to:

- precision
- recall
- PR-AUC
- threshold behavior

---

## 6. Fast 90% Model vs Slow 92% Model

Decide using:

- latency budget
- user impact
- infra cost
- business value of the extra 2%

Not by instinct alone.

---

## 7. Stakeholders Want Launch Tomorrow, You Disagree

Do not grandstand.

Frame it around:

- risk
- missing validation
- rollout alternative
- canary or shadow option

That is a strong answer.

---

## 8. Limited Labeling Budget

Prioritize with:

- active learning
- diverse sampling
- error-focused sampling
- class coverage

---

## 9. Need Explainability for Black-Box Model

Options:

- SHAP
- LIME
- permutation importance
- surrogate models

But always say:

none of these are causal truth.

That makes the answer stronger.

---

## 10. Real-Time Fraud System

Think:

- latency
- imbalance
- thresholding
- review queue
- feedback loop

That structure lands well fast.
