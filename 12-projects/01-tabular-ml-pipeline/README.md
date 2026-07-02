---
module: Projects
topic: Tabular ML Pipeline
subtopic: ""
status: unread
tags: [projects, tabular, sklearn, xgboost, pipelines, hands-on]
---
# Project: End-to-End Tabular ML Pipeline

**What this is:** a complete, runnable classification pipeline — synthetic churn dataset → EDA → leakage-safe preprocessing → baseline vs. gradient-boosted model → evaluation → serialized artifact. It's the applied counterpart to [01-foundations/04-data-processing-and-eda.md](../../01-foundations/04-data-processing-and-eda.md) and [02-classical-ml/](../../02-classical-ml/).

## Why this project

Every study plan in this repo mentions "build a tabular ML pipeline" as a milestone. This is that milestone, built — small enough to run in under a minute on a laptop, complete enough to demonstrate every step a real pipeline needs (leakage-safe splits, `ColumnTransformer`, cross-validation, calibration-aware evaluation, model comparison, serialization).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python generate_data.py     # writes data/churn.csv (synthetic, deterministic seed)
python train.py              # trains baseline + XGBoost, prints comparison, saves model.joblib
python evaluate.py           # loads model.joblib, prints held-out metrics + confusion matrix
```

## Structure

| File | Purpose |
|---|---|
| `generate_data.py` | Synthetic customer-churn dataset generator (no external download needed, reproducible via seed). |
| `train.py` | Full pipeline: split → `ColumnTransformer` preprocessing → `LogisticRegression` baseline → `XGBoost` model → cross-validated comparison → save best. |
| `evaluate.py` | Loads the saved pipeline, evaluates on a held-out test set, prints precision/recall/F1/ROC-AUC and a confusion matrix. |
| `requirements.txt` | Pinned-loose dependency list. |

## Design notes (what makes this "leakage-safe")

- All preprocessing (imputation, scaling, encoding) is inside a single `sklearn.Pipeline`, fit only on the training fold — see [04-data-processing-and-eda.md §6](../../01-foundations/04-data-processing-and-eda.md#6-data-leakage).
- Train/test split happens **before** any transform is fit.
- Cross-validation uses `StratifiedKFold` because churn is imbalanced (~15% positive rate by construction).
- Metric choice is ROC-AUC + PR-AUC, not accuracy, because of the class imbalance.

## Where to Next

- **The concepts behind each step** → [01-foundations/04-data-processing-and-eda.md](../../01-foundations/04-data-processing-and-eda.md)
- **Gradient boosting theory** → [02-classical-ml/](../../02-classical-ml/)
- **Next project: RAG pipeline** → [../02-rag-pipeline/](../02-rag-pipeline/)
