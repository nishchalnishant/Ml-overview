---
module: Interview Prep
topic: EA ML/DL Practical Interview
subtopic: ""
status: unread
tags: [interviewprep, ea, feature-engineering, deep-learning, hpo, cross-validation, production, scaling]
---
# Electronic Arts — ML Practical Interview Prep

Target format (from job description): **solve an ML problem** covering feature engineering →
train a deep learning model → hyperparameter selection → cross-validation → performance
optimization → scaling in production. Likely a live-coding / take-home-style session, probably
on a game-relevant dataset (player behavior, matchmaking, churn, LTV, cheat/toxicity detection,
recommendation, or telemetry-based prediction).

This doc is a route map through existing repo material, ordered to match the six phases in the
job description, plus an EA-flavored problem framing and a live-coding runbook.

---

## 0. Likely problem framing (EA context)

EA is a games company — expect the exercise to be a **supervised tabular/sequence problem**, not
open-ended research. Common shapes at game companies:

- **Player churn prediction** — will a player stop playing in the next N days?
- **LTV / spend prediction** — predict future purchases from early behavior.
- **Matchmaking / skill rating** — predict match outcome or balance quality.
- **Toxicity / cheat detection** — classify chat or gameplay telemetry.
- **Recommendation** — next item/mode/content to surface.

All of these reduce to the same pipeline: tabular or sequential telemetry features → DL model
(MLP / RNN-LSTM / Transformer over event sequences) → binary or regression target. Prepare the
**one end-to-end story** you know cold — see [PRE-INTERVIEW-CHECKLIST.md](PRE-INTERVIEW-CHECKLIST.md#system-design---know-one-end-to-end-example-cold).
Churn or LTV is the safest choice since it maps directly to existing content:
[17-customer-ltv-prediction.md](../06-production-ml/system-design/17-customer-ltv-prediction.md).

---

## 1. Feature Engineering

- [01-foundations/04-data-processing-and-eda.md](../01-foundations/04-data-processing-and-eda.md) — leakage, missing data, encoding
- [02-classical-ml/04-data-preprocessing.md](../02-classical-ml/04-data-preprocessing.md)
- [02-classical-ml/09-feature-selection.md](../02-classical-ml/09-feature-selection.md)
- [02-classical-ml/10-dimensionality-reduction.md](../02-classical-ml/10-dimensionality-reduction.md)
- [02-classical-ml/11-imbalanced-data.md](../02-classical-ml/11-imbalanced-data.md) — churn/fraud-style targets are almost always imbalanced
- [02-classical-ml/20-time-series-analysis.md](../02-classical-ml/20-time-series-analysis.md) — if telemetry is event/session sequences
- [06-production-ml/system-design/08-feature-store-architecture.md](../06-production-ml/system-design/08-feature-store-architecture.md) — point-in-time correctness, train/serve skew

**Say out loud in the interview:**
- Numeric: scaling (standardize for DL — neural nets are scale-sensitive, unlike trees)
- Categorical: embeddings for high-cardinality (player ID, item ID) vs. one-hot for low-cardinality
- Temporal: recency/frequency/monetary-style aggregates, rolling windows, session-level features
- Leakage: any feature computed using data from after the label window is invalid — call this out unprompted
- Missing data: game telemetry is sparse (new players, disconnects) — impute with an explicit "is_missing" flag rather than silently filling

---

## 2. Training a Deep Learning Model

- [03-deep-learning/01-pytorch-fundamentals.md](../03-deep-learning/01-pytorch-fundamentals.md)
- [03-deep-learning/deep-learning-cheatsheet.md](../03-deep-learning/deep-learning-cheatsheet.md) — backprop, init, batch norm, optimizers
- [03-deep-learning/02-transfer-learning.md](../03-deep-learning/02-transfer-learning.md)
- [03-deep-learning/components/](../03-deep-learning/components) and [03-deep-learning/methods/](../03-deep-learning/methods) — architecture-specific notes

**For tabular/sequence data specifically:**
- Start with a simple MLP baseline (embeddings for categoricals + concatenated numeric features) before reaching for anything fancier — say this explicitly, it signals judgment.
- If sequences (session/event history): 1D-CNN or LSTM or a small Transformer encoder over the event stream.
- Loss: binary cross-entropy (churn), or a regression/ranking loss (LTV, matchmaking).
- Always compare the DL model against a gradient-boosted tree baseline (XGBoost/LightGBM) — on tabular data GBTs frequently win or tie; being able to say this without being told shows maturity. Reference: [02-classical-ml/05-when-classical-ml-wins.md](../02-classical-ml/05-when-classical-ml-wins.md).

**Runnable reference implementation in this repo:** [12-projects/01-tabular-ml-pipeline/](../12-projects/01-tabular-ml-pipeline/) — leakage-safe `Pipeline`, baseline vs. gradient-boosted comparison, cross-validation, calibration-aware eval. Walk through `train.py` before the interview; it's the shape of what you'll likely be asked to write live.

---

## 3. Hyperparameter Selection

- [02-classical-ml/07-hyperparameter-optimization.md](../02-classical-ml/07-hyperparameter-optimization.md) — grid vs. random vs. Bayesian (Optuna/Hyperband)

**Key points to hit:**
- Random search > grid search for >2-3 hyperparameters (Bergstra & Bengio result)
- Bayesian optimization (Optuna, TPE) when each trial is expensive (large DL model)
- For DL specifically: learning rate is the single highest-leverage hyperparameter — use an LR range test / warmup + cosine decay rather than hand-tuning
- Early stopping on a validation metric doubles as both a regularizer and a compute-saving HPO technique
- State the search space and budget upfront: "given a fixed compute budget, I'd do random search over N trials with early stopping rather than exhaustive grid search"

---

## 4. Cross-Validation

- [02-classical-ml/06-cross-validation.md](../02-classical-ml/06-cross-validation.md) — K-fold, stratified, group, time-series splits

**Say explicitly:**
- Stratified K-fold if the target is imbalanced (churn, fraud)
- **GroupKFold** if multiple rows per player — splitting by row instead of by player leaks player identity across train/val and inflates validation score. This is the single most common mistake in player-level ML and a very likely trap in the exercise.
- **Time-based split** (not random) if predicting future behavior from past — never let future data leak into training folds. Reference: [02-classical-ml/20-time-series-analysis.md](../02-classical-ml/20-time-series-analysis.md).
- For DL: K-fold CV is expensive (K full training runs) — mention using a single held-out validation split during architecture/HPO search, then a final K-fold or bootstrap pass to report a confidence interval on the chosen config.

---

## 5. Performance Optimization

- [03-deep-learning/deep-learning-cheatsheet.md](../03-deep-learning/deep-learning-cheatsheet.md) — mixed precision, gradient accumulation, batch norm placement
- [06-production-ml/13-cost-optimization.md](../06-production-ml/system-design/13-cost-optimization.md)
- [06-production-ml/11-distributed-training.md](../06-production-ml/system-design/11-distributed-training.md)

**Two senses of "performance" — cover both:**
1. **Model performance**: regularization (dropout, weight decay, label smoothing), class-weighting or focal loss for imbalance, calibration (see [02-classical-ml/14-calibration-and-uncertainty.md](../02-classical-ml/14-calibration-and-uncertainty.md)) if predicted probabilities feed a downstream decision (e.g. an intervention threshold for at-risk players).
2. **Compute/training performance**: mixed-precision (AMP/fp16), larger batch + LR scaling, gradient accumulation for memory limits, profiling to find the actual bottleneck (data loading vs. GPU compute) before optimizing blindly.

---

## 6. Scaling in Production

- [06-production-ml/01-mlops.md](../06-production-ml/01-mlops.md)
- [06-production-ml/02-deployment-patterns.md](../06-production-ml/02-deployment-patterns.md) — batch vs. online serving, shadow deployment, canary
- [06-production-ml/03-model-governance.md](../06-production-ml/03-model-governance.md)
- [06-production-ml/system-design/01-machine-learning-engineering.md](../06-production-ml/system-design/01-machine-learning-engineering.md)
- [06-production-ml/system-design/07-data-engineering-for-ml.md](../06-production-ml/system-design/07-data-engineering-for-ml.md)
- [06-production-ml/system-design/10-model-registry-versioning.md](../06-production-ml/system-design/10-model-registry-versioning.md)
- [06-production-ml/system-design/17-customer-ltv-prediction.md](../06-production-ml/system-design/17-customer-ltv-prediction.md) — closest full system-design writeup to a likely EA scenario
- [06-production-ml/system-design/20-fraud-detection-full-system.md](../06-production-ml/system-design/20-fraud-detection-full-system.md) — if the exercise leans anti-cheat/toxicity

**Points to hit:**
- Batch scoring (nightly churn scores) vs. real-time inference (in-session matchmaking/recommendation) — state the latency requirement first, it determines the whole serving architecture.
- Training-serving skew — feature computation must be identical online and offline (point-in-time correctness); this is the #1 real-world failure mode, mention it unprompted.
- Model versioning/rollback, canary or shadow deployment before full rollout.
- Monitoring: prediction distribution drift, feature drift, delayed ground truth (churn labels arrive weeks later — how do you monitor a model when you can't compute live accuracy?).
- Horizontal scaling of the serving layer + caching hot features; retraining cadence tied to drift detection rather than a fixed calendar schedule.

---

## Live-Coding Runbook (if it's hands-on)

1. Clarify the target, label window, and evaluation metric before writing any code.
2. Load data, check class balance, check for obvious leakage columns (timestamps past the label window, IDs).
3. Build a leakage-safe `sklearn` `Pipeline`/`ColumnTransformer` for preprocessing — mirror [12-projects/01-tabular-ml-pipeline/train.py](../12-projects/01-tabular-ml-pipeline/train.py).
4. Train a fast baseline (logistic regression or GBT) first — gives you a sanity-check number before the DL model.
5. Build the DL model (PyTorch), start with the simplest architecture that could work.
6. Cross-validate with the correct split strategy (grouped/time-based if applicable) — state why before coding it.
7. Tune 2-3 hyperparameters via random search with early stopping, not grid search — say why.
8. Report metrics appropriate to the task (ROC-AUC/PR-AUC for imbalanced classification, not accuracy) — reference [02-classical-ml/12-ml-evaluation-metrics.md](../02-classical-ml/12-ml-evaluation-metrics.md).
9. Close with 60 seconds on how you'd deploy and monitor it in production, unprompted.

---

## Final Pass

- [ ] [PRE-INTERVIEW-CHECKLIST.md](PRE-INTERVIEW-CHECKLIST.md) — 48h/24h/morning-of checklist
- [ ] Run [12-projects/01-tabular-ml-pipeline/](../12-projects/01-tabular-ml-pipeline/) end-to-end once, out loud, narrating each step
- [ ] Have 2 questions ready for the interviewer about EA's ML stack (e.g., "what's the feature store / serving latency budget for player-facing models?")

---

## Later round: production problem + coding + justified tradeoffs

If a later round is framed as "solve a real production problem end to end (with coding), make and
justify tradeoffs" rather than a clean-dataset pipeline drill, switch prep material — see
[ROUND3-tradeoff-drills.md](ROUND3-tradeoff-drills.md) for the tradeoff bank (model choice, serving
architecture, feature pipeline, rollout, metrics) and the clarify-first / narrate-tradeoffs runbook
for ambiguous scenarios.
