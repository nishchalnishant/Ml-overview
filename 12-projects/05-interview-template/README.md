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

## Scope

Built for one interview archetype: **tabular, single dataset, binary classification, live-coding
end-to-end pipeline.** It intentionally does not cover:

- Image/CNN, text/transformer, or sequence models — see [03-deep-learning/methods/](../../03-deep-learning/methods/) for those architectures; reuse this template's phase structure and narration style, not its code.
- Regression or multiclass targets — swap the loss (`MSELoss`/`nn.CrossEntropyLoss`), the metric (MAE/RMSE or macro-F1), and the final layer width; everything else (split ordering, preprocessing contract, CV logic minus stratification) carries over unchanged.
- Pure system-design or theory-only interview formats — see [07-interview-prep/](../../07-interview-prep/) instead.
- Messy/raw input handling (schema drift, mixed types needing on-the-spot EDA) — the bundled toy datasets are already clean; budget extra time for this if the real data isn't.

## Structure

| File | Interview phase | What it forces you to say/do |
|---|---|---|
| `template/00_problem_framing.md` | (pre-work) | Clarify target, label window, metric, baseline, constraints before coding |
| `template/01_feature_engineering.py` | Feature engineering | Leakage check, missing-data strategy, numeric/categorical handling, train/test split ordering |
| `template/02_baseline_model.py` | (sanity baseline) | Fast linear/GBT baseline before the DL model — gives a number to beat |
| `template/03_deep_learning_model.py` | Train a deep learning model | PyTorch MLP w/ embeddings, loss choice, training loop, early stopping |
| `template/04_hyperparameter_search.py` | Hyperparameter selection | Random search over a stated space, budget, why not grid search |
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

## Full theory references

Point-by-point mapping into the rest of the repo, same phases as
[07-interview-prep/EA-ml-deep-learning-interview.md](../../07-interview-prep/EA-ml-deep-learning-interview.md)
(that doc is company-specific framing; this template is the reusable skeleton behind it):

| Phase | Theory |
|---|---|
| Feature engineering | [01-foundations/04-data-processing-and-eda.md](../../01-foundations/04-data-processing-and-eda.md), [02-classical-ml/04-data-preprocessing.md](../../02-classical-ml/04-data-preprocessing.md), [02-classical-ml/09-feature-selection.md](../../02-classical-ml/09-feature-selection.md), [02-classical-ml/11-imbalanced-data.md](../../02-classical-ml/11-imbalanced-data.md) |
| Deep learning model | [03-deep-learning/01-pytorch-fundamentals.md](../../03-deep-learning/01-pytorch-fundamentals.md), [03-deep-learning/deep-learning-cheatsheet.md](../../03-deep-learning/deep-learning-cheatsheet.md) |
| Hyperparameter selection | [02-classical-ml/07-hyperparameter-optimization.md](../../02-classical-ml/07-hyperparameter-optimization.md) |
| Cross-validation | [02-classical-ml/06-cross-validation.md](../../02-classical-ml/06-cross-validation.md) |
| Performance optimization | [03-deep-learning/deep-learning-cheatsheet.md](../../03-deep-learning/deep-learning-cheatsheet.md), [02-classical-ml/14-calibration-and-uncertainty.md](../../02-classical-ml/14-calibration-and-uncertainty.md) |
| Scaling in production | [06-production-ml/01-mlops.md](../../06-production-ml/01-mlops.md), [06-production-ml/02-deployment-patterns.md](../../06-production-ml/02-deployment-patterns.md) |

## Where to Next

- **Finished reference implementation** → [../01-tabular-ml-pipeline/](../01-tabular-ml-pipeline/)
- **Company-specific interview routing** → [07-interview-prep/EA-ml-deep-learning-interview.md](../../07-interview-prep/EA-ml-deep-learning-interview.md)
- **Full pre-interview checklist** → [07-interview-prep/PRE-INTERVIEW-CHECKLIST.md](../../07-interview-prep/PRE-INTERVIEW-CHECKLIST.md)
