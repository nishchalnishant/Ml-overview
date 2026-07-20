---
module: Projects
topic: Tabular ML Pipeline
subtopic: ""
status: unread
tags: [projects, tabular, sklearn, xgboost, pipelines, hands-on]
---
# Project: End-to-End Tabular ML Pipeline

**What this is:** a complete, runnable ML pipeline covering the full interview-relevant loop — synthetic churn dataset → leakage-safe feature engineering → classical baseline vs. gradient-boosted model **and** a PyTorch deep learning model → hyperparameter search → cross-validation → performance profiling → serving/scaling stub. It's the applied counterpart to [01-foundations/04-data-processing-and-eda.md](../../02-data/02-data-processing-and-eda.md), [02-classical-ml/](../../02-classical-ml/), and [03-deep-learning/](../../03-deep-learning/).

## Why this project

Every study plan in this repo mentions "build a tabular ML pipeline" as a milestone. This is that milestone, built — small enough to run in under a few minutes on a laptop, complete enough to demonstrate every phase of a "feature engineering → DL model → HPO → CV → perf optimization → production scaling" interview exercise (see [07-interview-prep/EA-ml-deep-learning-interview.md](../../16-interview-prep/ea/00-ml-deep-learning-interview.md)).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python generate_data.py     # writes data/churn.csv (synthetic, deterministic seed)
python train.py              # classical baseline: LogisticRegression + XGBoost, CV comparison, saves model.joblib
python evaluate.py           # loads model.joblib, prints held-out metrics + confusion matrix
python train_dl.py           # PyTorch MLP w/ categorical embeddings, random-search HPO, early stopping, K-fold CV, saves model_dl.pt
python optimize.py           # profiles data-loading vs. compute time, compares fp32 vs. mixed-precision training
uvicorn serve:app --workers 4       # online serving: POST /score
python serve.py in.csv out.csv      # batch scoring path (chunked, same model artifact)
```

## Structure

| File | Purpose |
|---|---|
| `generate_data.py` | Synthetic customer-churn dataset generator (no external download needed, reproducible via seed). |
| `train.py` | Classical pipeline: split → `ColumnTransformer` preprocessing → `LogisticRegression` baseline → `XGBoost` model → cross-validated comparison → save best. |
| `evaluate.py` | Loads the saved sklearn pipeline, evaluates on a held-out test set, prints precision/recall/F1/ROC-AUC and a confusion matrix. |
| `train_dl.py` | PyTorch `TabularMLP` (embeddings for categoricals + numeric features) — random-search hyperparameter tuning with early stopping, then K-fold CV with the winning config to report a confidence interval. |
| `optimize.py` | Performance-optimization pass: profiles data-loading vs. compute time per epoch, compares fp32 vs. mixed-precision (AMP) training on the same config. |
| `serve.py` | Production-serving stub: a stateless FastAPI `/score` endpoint (horizontally scalable, model loaded once per worker) plus a chunked batch-scoring entrypoint for nightly jobs. |
| `requirements.txt` | Pinned-loose dependency list. |

## Which script maps to which interview phase

| Phase (from job description) | File |
|---|---|
| Feature engineering | `generate_data.py` (realistic missingness/skew) + preprocessing in `train.py` / `train_dl.py` |
| Train a deep learning model | `train_dl.py` — `TabularMLP` |
| Hyperparameter selection | `train_dl.py::random_search` |
| Cross-validation | `train.py` (`StratifiedKFold`) and `train_dl.py` (held-out split for HPO, K-fold for the final reported score — see inline comment on why those differ) |
| Performance optimization | `optimize.py` |
| Scaling in production | `serve.py` |

## Design notes (what makes this "leakage-safe")

- All preprocessing (imputation, scaling, encoding) is inside a single `sklearn.Pipeline`, fit only on the training fold — see [04-data-processing-and-eda.md §6](../../02-data/02-data-processing-and-eda.md#6-data-leakage).
- Train/test split happens **before** any transform is fit.
- Cross-validation uses `StratifiedKFold` because churn is imbalanced (~15% positive rate by construction).
- Metric choice is ROC-AUC + PR-AUC, not accuracy, because of the class imbalance.

## Where to Next

- **The concepts behind each step** → [01-foundations/04-data-processing-and-eda.md](../../02-data/02-data-processing-and-eda.md)
- **Gradient boosting theory** → [02-classical-ml/](../../02-classical-ml/)
- **Next project: RAG pipeline** → [../02-rag-pipeline/](../02-rag-pipeline/)
