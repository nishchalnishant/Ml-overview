---
module: Data
topic: Overview
subtopic: ""
status: unread
tags: [data, eda, feature-engineering, index]
prerequisites: [statistics]
---
# Data

Everything between raw data and a model-ready matrix. In interviews this section is underrated: most candidates rush to model choice, while practitioners know the data pipeline is where accuracy is actually won or lost.

**What lives here:** EDA, preprocessing, feature engineering, leakage, validation splits, imbalance.

**The highest-frequency trap in this section** is data leakage — fitting a scaler or an encoder on the full dataset before splitting. It inflates validation scores and is the single most common reason a model that looked strong offline collapses in production.

## Core Modules

| File | What it covers |
| :--- | :--- |
| [01-python-and-data-tooling.md](01-python-and-data-tooling.md) | NumPy, Pandas, Polars, SciPy, Matplotlib/Seaborn — vectorization, broadcasting, memory/performance |
| [02-data-processing-and-eda.md](02-data-processing-and-eda.md) | EDA workflow, missing values, outliers, feature engineering, data leakage, validation strategy, sklearn pipelines |
| [03-sql-for-interviews.md](03-sql-for-interviews.md) | SQL for interviews — execution order, joins, window functions, CTEs, gaps/islands, top-N-per-group, optimization |
| [04-feature-engineering.md](04-feature-engineering.md) | Data preprocessing and feature engineering — encoding, scaling, transforms |
| [05-feature-selection.md](05-feature-selection.md) | Filter/wrapper/embedded methods, mutual information, regularization |
| [06-imbalanced-data.md](06-imbalanced-data.md) | SMOTE, class weighting, resampling, cost-sensitive learning |
| [07-data-engineering.md](07-data-engineering.md) | Data engineering for ML — pipelines, storage, feature stores |

