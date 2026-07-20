---
module: Projects
topic: Live-Coding Interview Template
subtopic: ""
status: unread
tags: [projects, template, interview, feature-engineering, deep-learning, hpo, cross-validation, production]
---
# Project: Live-Coding ML/DL Interview Template

**What this is:** a fill-in-the-blanks skeleton for the "solve an ML problem end-to-end" interview
format — feature engineering → train a deep learning model → hyperparameter selection →
cross-validation → performance optimization → scaling in production. Generic on purpose: swap in
whatever dataset/target the interviewer gives you, keep the structure.

Unlike [01-tabular-ml-pipeline/](../01-tabular-ml-pipeline/), which is a **finished, runnable
reference implementation** on a fixed synthetic churn dataset, this folder is a **template you
retype/adapt live** — each file has `# TODO(interview)` markers and a narration comment explaining
what to say out loud at that step. Practice by deleting the TODOs and rewriting them from memory
against a new dataset (e.g. an sklearn toy dataset or a Kaggle CSV).

## How to use this in an interview

1. Skim `template/00_problem_framing.md` first — 2 minutes, before writing any code. Fill in the
   target, label window, and metric with the interviewer.
2. Work through `template/` in numeric order. Each script runs standalone once you fill in the
   TODOs, and each also runs as-is on the bundled toy dataset (`sklearn`'s breast-cancer /
   California-housing sets) so you can sanity-check the scaffold before the real data shows up.
3. Narrate before you type — every file's header comment has a "say this out loud" line. Interviewers
   score reasoning, not just working code.
4. If time-boxed, the minimum defensible path is: `01` → `02` → `03` (skip `04`/`05` narration-only,
   mention them verbally) → `06` closing statement.

## 60-minute time-boxed script

Running all 6 files verbatim takes longer than an hour. This is the actual pacing to follow live,
narrating the skipped parts instead of typing them:

| Time | Do |
|---|---|
| 0:00–0:05 | `00_problem_framing.md` out loud — target, label window, metric, constraint, baseline, leakage risks |
| 0:05–0:20 | `01_feature_engineering.py` — build it live against the given data (leakage check, missing flags, `ColumnTransformer`) |
| 0:20–0:25 | `02_baseline_model.py` — fit fast, state the number, move on. Don't tune it |
| 0:25–0:45 | `03_deep_learning_model.py` — the centerpiece. Build the MLP, justify the loss, wire up early stopping |
| 0:45–0:50 | `04_hyperparameter_search.py` + `05_cross_validation.py` — **narrate only**: state the search space and CV strategy you'd use and why, don't run a full search live |
| 0:50–0:60 | `06_performance_and_scaling.py` — closing statement: one model-perf lever, one compute-perf lever, the serving/monitoring checklist |

If the interviewer gives you image, text, or sequence data instead of tabular, say so immediately —
this template's `03` is tabular-specific (embeddings + MLP). Fall back to the architecture patterns
in [03-deep-learning/methods/](../../03-deep-learning/methods/) and adapt the same 6-phase shape
(framing → features → baseline → model → HPO/CV narration → perf/scaling) rather than trying to
force this code to fit.

## Second drill: regression

`template/` is binary classification. Once that feels automatic, drill
[`template-regression/run_regression_drill.py`](template-regression/run_regression_drill.py) —
same 6-phase shape, California-housing dataset, one consolidated script instead of 6 files. It
exists so you're not only fluent in one task type: the deltas (loss = MSE/Huber not
BCEWithLogitsLoss, metric = MAE/RMSE not ROC-AUC, CV = plain KFold not Stratified, no
sigmoid/calibration step) are called out inline in its docstring. Run it standalone:

```bash
cd template-regression
pip install -r requirements.txt
python run_regression_drill.py
```

## Third drill: ambiguous production scenario (not a clean dataset)

`template/` and `template-regression/` both hand you a clean feature/target split — good for
drilling the mechanics, but not representative of a round framed as "solve a real production
problem end to end, make and justify tradeoffs." For that format, drill
[`production-scenario/`](production-scenario/) instead: a fraud-scoring scenario stated only as a
business ask, with messy generated data (label lag, schema drift, merchant concentration, severe
class imbalance) you have to find yourself before modeling anything. See
[`production-scenario/00_scenario_brief.md`](production-scenario/00_scenario_brief.md) to start,
and [07-interview-prep/ROUND3-tradeoff-drills.md](../../16-interview-prep/05-round3-tradeoff-drills.md)
for the tradeoff-justification patterns this drill exercises.

## Scope

Built for two interview archetypes: **tabular, single dataset, binary classification OR
regression, live-coding end-to-end pipeline.** It intentionally does not cover:

- Image/CNN, text/transformer, or sequence models — see [03-deep-learning/methods/](../../03-deep-learning/methods/) for those architectures; reuse this template's phase structure and narration style, not its code.
- Multiclass targets — swap the loss (`nn.CrossEntropyLoss`), the metric (macro-F1 or multiclass ROC-AUC), and the final layer width to `n_classes`; everything else (split ordering, preprocessing contract) carries over unchanged.
- Pure system-design or theory-only interview formats — see [07-interview-prep/](../../07-interview-prep/) instead.
- Messy/raw input handling (schema drift, mixed types needing on-the-spot EDA) — the bundled toy datasets are already clean; budget extra time for this if the real data isn't.

## Structure

| File | Interview phase | What it forces you to say/do |
|---|---|---|
| `template/00_problem_framing.md` | (pre-work) | Clarify target, label window, metric, baseline, constraints before coding |
| `template/01_feature_engineering.py` | Feature engineering | Leakage check, missing-data strategy, numeric/categorical handling, train/test split ordering |
| `template/02_baseline_model.py` | (sanity baseline) | Fast linear/GBT baseline before the DL model — gives a number to beat |
| `template/03_deep_learning_model.py` | Train a deep learning model | PyTorch MLP w/ embeddings, loss choice, training loop, early stopping |
| `template/04_hyperparameter_search.py` | Hyperparameter selection | Random search (GBT) over a stated space/budget, why not grid search; Optuna/TPE + pruning for the DL model, why Bayesian beats random once trials are expensive |
| `template/05_cross_validation.py` | Cross-validation | Correct split strategy (stratified/group/time-based), why it matters here |
| `template/06_performance_and_scaling.py` | Performance optimization + Scaling in production | Model-perf tuning, compute-perf profiling (AMP), serving stub, monitoring checklist |

## Setup

```bash
cd template
pip install -r requirements.txt
```

## Run order (against the bundled toy dataset, before swapping in real data)

```bash
python 01_feature_engineering.py
python 02_baseline_model.py
python 03_deep_learning_model.py
python 04_hyperparameter_search.py
python 05_cross_validation.py
python 06_performance_and_scaling.py
```

## Design principle: same contract every time

Every script shares one preprocessing contract so the pipeline stays leakage-safe no matter which
dataset you drop in:

- Split **before** fitting any transform.
- All imputation/scaling/encoding lives inside one `sklearn.Pipeline` / `ColumnTransformer`, fit
  only on the training fold.
- Any feature using information from after the label's observation window is called out and
  dropped — say this unprompted even if the interviewer didn't ask.

## From trained model to production: the steps in between

Once `03_deep_learning_model.py` (or the baseline) produces a model you're happy with, it isn't
deployable yet. Say these steps out loud — this is a common "walk me through it" follow-up:

1. **Evaluate on the held-out test set exactly once.** Not the validation set used for early
   stopping/HPO — that number is already optimistic. Test is touched once, at the very end, and
   that's the number you report.
2. **Freeze and serialize the whole pipeline, not just the model.** Preprocessing
   (`ColumnTransformer`) + model must ship as one artifact (e.g. `joblib.dump` for sklearn, or
   `torch.save(state_dict)` plus the same preprocessor object) — never re-derive preprocessing
   logic in a different language/service, that's how training-serving skew starts.
3. **Version the artifact.** Tag it with data version, code commit hash, and metrics at minimum —
   you need to answer "which model is live and what was it trained on" at any point later.
4. **Write a reproducible training pipeline, not a notebook.** The interviewer wants to hear that
   the path from raw data to this artifact is a script/pipeline (e.g. `01`→`02`/`03` in this
   template chained together), re-runnable on new data without manual notebook edits.
5. **Wrap inference behind a stable contract.** Define the exact input schema and output schema
   (e.g. a `predict(features: dict) -> {"score": float}` function or a REST endpoint) — this is
   the seam between the ML pipeline and the rest of the system, and it should be tested like any
   other API.
6. **Load-test and latency-check against the constraint from `00_problem_framing.md`.** If the
   answer there was "<100ms online," measure p50/p99 inference latency now, before deployment, not
   after a user complains.
7. **Get a second set of eyes / a review gate.** Model cards, a peer review of the training
   pipeline, or a sign-off step — treat a model artifact with the same rigor as a code change
   before it reaches production.
8. **Only then deploy** — shadow/canary first, which is where the next section picks up.

## After deployment: what happens once the model is live

`06_performance_and_scaling.py` prints a checklist for this, but say it explicitly as its own
closing section — interviewers often ask "the model's live, now what?" as a follow-up:

1. **Shadow / canary first, not a full rollout.** Route a small slice of live traffic (or mirror
   traffic with no user-facing effect) to the new model, compare its predictions against the
   current production model before it serves real decisions.
2. **Roll out gradually, keep the rollback path hot.** Ramp traffic percentage up in stages; keep
   the previous model version deployed and instantly switchable if a metric regresses.
3. **Monitor input drift.** Compare live feature distributions against the training distribution
   (e.g. PSI, KL divergence per feature) — catches upstream data pipeline changes before they tank
   accuracy.
4. **Monitor prediction drift.** Track the distribution of output scores over time, independent of
   whether ground truth is available yet — a sudden shift signals something changed even before
   you can measure error.
5. **Close the label loop.** Log every prediction with an ID; when the real outcome eventually
   arrives (could be minutes or months later, depending on the label window from
   `00_problem_framing.md`), join it back to compute *live* AUC/calibration/error — proxy signals
   (drift, latency, request volume) are not a substitute for this.
6. **Retrain on a schedule or a trigger.** Fixed cadence (e.g. weekly) or trigger-based (drift
   exceeds a threshold, performance drops below a floor) — say which one fits the problem's rate
   of change and why.
7. **Guard training-serving skew on every retrain.** The serving code must keep using the exact
   preprocessing pipeline object the new model was trained with (same joblib/artifact), not a
   reimplementation — this is the most common cause of "worked in offline eval, broke in prod."
8. **Have an incident response path.** Know how to kill/roll back the model in production
   (feature flag, traffic router, or redeploy previous artifact) faster than you can retrain — this
   is the answer when asked "the model just started returning garbage, what do you do right now?"

## Full theory references

Point-by-point mapping into the rest of the repo, same phases as
[07-interview-prep/EA-ml-deep-learning-interview.md](../../16-interview-prep/ea/00-ml-deep-learning-interview.md)
(that doc is company-specific framing; this template is the reusable skeleton behind it):

| Phase | Theory |
|---|---|
| Feature engineering | [01-foundations/04-data-processing-and-eda.md](../../02-data/02-data-processing-and-eda.md), [02-classical-ml/04-data-preprocessing.md](../../02-data/03-data-preprocessing.md), [02-classical-ml/09-feature-selection.md](../../02-data/05-feature-selection.md), [02-classical-ml/11-imbalanced-data.md](../../02-data/06-imbalanced-data.md) |
| Deep learning model | [03-deep-learning/01-pytorch-fundamentals.md](../../05-deep-learning-core/09-pytorch-fundamentals.md), [03-deep-learning/deep-learning-cheatsheet.md](../../05-deep-learning-core/_cheatsheet.md) |
| Hyperparameter selection | [02-classical-ml/07-hyperparameter-optimization.md](../../03-classical-ml/08-hyperparameter-optimization.md) |
| Cross-validation | [02-classical-ml/06-cross-validation.md](../../04-evaluation/03-cross-validation.md) |
| Performance optimization | [03-deep-learning/deep-learning-cheatsheet.md](../../05-deep-learning-core/_cheatsheet.md), [02-classical-ml/14-calibration-and-uncertainty.md](../../04-evaluation/04-calibration-and-uncertainty.md) |
| Scaling in production | [06-production-ml/01-mlops.md](../../13-production-ml/01-mlops.md), [06-production-ml/02-deployment-patterns.md](../../13-production-ml/02-deployment-patterns.md) |

## Where to Next

- **Finished reference implementation** → [../01-tabular-ml-pipeline/](../01-tabular-ml-pipeline/)
- **Company-specific interview routing** → [07-interview-prep/EA-ml-deep-learning-interview.md](../../16-interview-prep/ea/00-ml-deep-learning-interview.md)
- **Full pre-interview checklist** → [07-interview-prep/PRE-INTERVIEW-CHECKLIST.md](../../16-interview-prep/03-pre-interview-checklist.md)
- **Later round: ambiguous production problem + justified tradeoffs (not a clean dataset)** → [07-interview-prep/ROUND3-tradeoff-drills.md](../../16-interview-prep/05-round3-tradeoff-drills.md)
