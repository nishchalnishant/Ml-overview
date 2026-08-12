---
module: Data
topic: Revision Guide
subtopic: ""
status: unread
tags: [revision, data, eda, sql, feature-engineering, feature-selection, imbalanced-data, data-engineering]
prerequisites: [02-data]
---
# 02 Data — Revision Guide

Single-page revision for the entire `02-data/` folder: Python/data tooling, EDA, SQL, feature engineering, feature selection, imbalanced data, and data engineering. Use the outline to reconstruct the folder's structure from memory, then self-test with the graded Q&A below.

## Structure Overview

```markdown
- **02-data**
  - **Python & Data Tooling**
    - NumPy substrate
      - Vectorization
      - Broadcasting rules
      - Views vs copies
      - Memory layout C-order
      - dtype discipline float32/64
    - Pandas
      - Vectorized vs apply vs iterrows
      - GroupBy split-apply-combine
      - Merges/joins fan-out
      - Memory optimization category dtype
      - Chained indexing SettingWithCopy
    - Polars lazy eager
      - Predicate/projection pushdown
      - Streaming collect
    - SciPy stats/optimize/sparse
    - Matplotlib/Seaborn viz
  - **Data Processing & EDA**
    - Workflow loop
    - EDA first pass
      - Univariate analysis
      - Bivariate/multivariate
      - Target analysis
    - Missing values
      - MCAR/MAR/MNAR
      - Imputation strategies
      - Missingness indicator
    - Outliers
      - IQR method
      - Z-score method
      - Error vs signal
    - Feature eng basics
      - Scaling methods
      - Encoding methods
      - Log/poly/datetime transforms
    - Data leakage
      - Preprocessing before split
      - Target leakage
      - Temporal leakage
      - Group leakage
      - Target encoding leakage
    - Validation strategy
      - Stratified K-Fold
      - TimeSeriesSplit
      - GroupKFold
      - Nested CV
    - sklearn Pipelines
    - Data drift
      - Covariate/label/concept shift
      - PSI
      - KS test
  - **SQL for Interviews**
    - Execution order
    - Joins
      - INNER/LEFT/RIGHT/FULL
      - Self-join
      - Anti-join pattern
      - Fan-out bug
    - GROUP BY / HAVING
      - COUNT star vs col
    - Window functions
      - PARTITION BY / OVER
      - ROW_NUMBER/RANK/DENSE_RANK
      - Running totals
      - LAG/LEAD
      - Top-N-per-group
    - Subqueries & CTEs
      - Scalar vs correlated
      - Recursive CTE hierarchy
    - NULL semantics
      - COALESCE/NULLIF
    - Gaps and islands
    - DDL/DML/constraints
      - DELETE vs TRUNCATE vs DROP
      - Upsert
      - PK/FK/UNIQUE/CHECK
    - Transactions & ACID
      - Isolation levels
    - CASE WHEN pivot
    - Dialects
    - Normalization/OLTP vs OLAP
  - **Feature Engineering**
    - Preprocessing-first mindset
    - Scaling Standard/MinMax/Robust
    - Categorical encoding
      - One-hot
      - Ordinal
      - Target encoding + smoothing
      - Frequency encoding
      - Embeddings
    - Missing values MCAR/MAR/MNAR
    - Skewed data
      - Log1p transform
      - Yeo-Johnson
    - Outliers signal vs error
    - Feature crosses/interactions
      - Temporal features
      - Ratio features
    - Leakage deep dive
    - Feature selection intro
    - Feature extraction vs selection
    - Train-serve skew
    - Class imbalance intro
  - **Feature Selection**
    - Filter methods
      - Variance threshold
      - Correlation redundancy
      - Mutual information
      - Chi-squared
      - ANOVA F-stat
    - Wrapper methods
      - RFE / RFECV
      - Sequential forward/backward
    - Embedded methods
      - L1 Lasso / Elastic Net
      - Tree importance MDI
      - Permutation importance
      - SelectFromModel
    - SHAP-based selection
    - Selection vs dim reduction
    - Pipeline integration
  - **Imbalanced Data**
    - Why accuracy fails
    - Metrics
      - Precision/Recall/F1
      - F-beta
      - PR-AUC vs ROC-AUC
      - MCC
    - Resampling
      - Random over/undersample
      - SMOTE
      - ADASYN
      - Tomek Links
      - SMOTEENN/SMOTETomek
    - Cost-sensitive learning
      - class_weight balanced
      - Per-sample weights
      - scale_pos_weight
    - Threshold moving
    - Ensemble methods
      - BalancedRandomForest
      - EasyEnsemble
    - Focal loss
    - Label smoothing
    - Pipeline safety rules
  - **Data Engineering for ML**
    - Batch vs streaming
    - Lambda vs Kappa
    - Spark/Flink/Trino internals
    - Storage formats columnar
    - Predicate pushdown
    - Data quality/schema validation
    - Data contracts
    - Versioning DVC/Delta/Iceberg
    - ETL vs ELT / dbt / medallion
    - Warehousing OLAP/OLTP
    - Feature stores online/offline
      - Point-in-time correctness
    - Cost optimization
```

## Questions

### Easy

1. What is vectorization and why is it faster than a Python `for` loop?
2. State NumPy's broadcasting rule for comparing shapes right-to-left.
3. What's the difference between a view and a copy in NumPy?
4. Why does `df.iterrows()` perform poorly compared to vectorized operations?
5. What does `.agg()` return versus `.transform()` in a pandas groupby?
6. When would you reach for Polars instead of pandas?
7. Why convert a low-cardinality string column to `category` dtype?
8. What is the goal of EDA before modeling?
9. Define MCAR, MAR, and MNAR.
10. Why use median imputation instead of mean imputation for skewed numeric data?
11. What does the IQR outlier rule compute?
12. Name three scaling methods and one situation where each is preferred.
13. Why don't tree-based models need feature scaling?
14. What's the difference between one-hot encoding and target encoding?
15. Define data leakage in one sentence.
16. Why must a `StandardScaler` be fit only on the training set?
17. What SQL clause filters rows before grouping, and which filters groups after?
18. Explain the difference between `COUNT(*)` and `COUNT(column)`.
19. What does `LEFT JOIN` do that `INNER JOIN` doesn't?
20. What is the SQL execution order starting from `FROM`?
21. What's the difference between `RANK()` and `DENSE_RANK()` on a tie?
22. What is a CTE and why prefer it over nested subqueries?
23. Why is `WHERE col = NULL` always false?
24. Why is accuracy a misleading metric on a 99/1 imbalanced dataset?
25. What does SMOTE do differently from random oversampling?
26. What is `class_weight='balanced'` doing mechanically in sklearn?
27. What is the difference between feature selection and feature extraction?
28. What is Population Stability Index (PSI) used for?

### Medium

1. Why does chained indexing (`df[cond]["col"] = x`) sometimes silently fail to modify the DataFrame?
2. Explain why summing along axis=1 is faster than axis=0 on a large row-major NumPy array.
3. Walk through why `df.groupby().apply(custom_python_fn)` is a performance anti-pattern on large data.
4. Why does Polars's lazy `scan_csv → filter → select → collect` outperform pandas on multi-GB aggregations?
5. Why is a missingness indicator column often more useful than the imputed value itself?
6. Give an example of a feature that is a genuine outlier signal rather than a data error, and explain why removing it would hurt the model.
7. Why must target encoding be computed out-of-fold, and what happens if it isn't?
8. Explain why a random K-Fold split is invalid for time-series data and what to use instead.
9. Why does `LEFT JOIN ... WHERE right.col > x` silently degrade into an `INNER JOIN`? How do you fix it while preserving unmatched rows?
10. Explain the join fan-out bug: why does summing a column from the "one" side after a one-to-many join produce inflated totals?
11. Why can't you filter directly on a window function alias in the same `SELECT`'s `WHERE` clause? What's the standard workaround?
12. Derive the logic behind the `date - ROW_NUMBER()` gaps-and-islands trick.
13. When would you prefer a correlated subquery versus a window function to answer "who earns above their department's average," and why do window functions usually win?
14. Why does Pearson correlation-based redundancy removal miss non-linear redundancy, and what filter method catches it instead?
15. Explain why L1 (Lasso) tends to zero out coefficients while L2 does not.
16. Why can MDI-based tree feature importance be biased toward high-cardinality features, and what's the bias-free alternative?
17. Why can permutation importance understate the importance of two correlated, jointly-critical features?
18. Why is PR-AUC preferred over ROC-AUC on a highly imbalanced dataset?
19. Explain why SMOTE can generate synthetic points inside majority-class territory, and what technique mitigates this.
20. Why does threshold-moving cost nothing at training time but still require a held-out validation set?
21. What is train-serve skew, and why doesn't it show up in offline evaluation?
22. Why does a `Pipeline` (sklearn) structurally prevent most leakage bugs during cross-validation?
23. Explain the difference between Lambda and Kappa architecture, and why most modern systems default toward Kappa.
24. Why does columnar storage (Parquet) benefit predicate/projection pushdown more than row-oriented storage?
25. What is point-in-time correctness in a feature store, and why does violating it cause leakage?
26. Why would you choose `RobustScaler` over `StandardScaler` for a feature with heavy outliers?
27. Explain why `HAVING` exists as a separate clause from `WHERE` rather than just extending `WHERE`.

### Hard

1. You fit a `StandardScaler` on the full dataset before splitting, then run 5-fold CV downstream. Explain precisely which numbers are contaminated and estimate the direction of bias on the reported CV score.
2. A model shows 97% AUC offline and 82% in production. Walk through a systematic checklist to isolate whether this is leakage, train-serve skew, or genuine distribution shift.
3. You one-hot encode a categorical feature with 100K unique values across 1M rows. Compute the rough dense memory footprint in float64 and explain why a sparse CSR matrix avoids it — what breaks if you feed that sparse matrix into an estimator that requires dense input?
4. A binary feature is 1 in 99% of rows and the informative 1% signal lives in the minority value. Explain why `VarianceThreshold` with a naive cutoff would silently discard it, and how you'd set the threshold instead.
5. Design an out-of-fold K-fold target encoding scheme from scratch for a categorical feature with categories ranging from 2 to 500,000 occurrences. Address both leakage and unseen-category handling at inference.
6. Two features are individually uncorrelated with the target (near-zero mutual information) but jointly perfectly predictive (e.g., XOR-like interaction). Explain why every filter method in this folder misses this, and which method category would catch it.
7. You SMOTE-oversample before doing feature selection with mutual information, all before your train/test split. Enumerate every leakage point in this pipeline and rewrite the correct order of operations.
8. In a 99.5% negative fraud dataset, class-weighted logistic regression is trained, then a threshold is tuned on the validation PR curve, then the model is redeployed and the true fraud rate in production drifts from 0.5% to 2% over six months. Explain what breaks with the previously tuned threshold and how you'd detect/react to it.
9. Explain why `ROC_AUC` can look nearly identical between two models even when one catches dramatically more real fraud cases in the top-K alerts a fraud team can actually review — and what metric would expose the difference.
10. Write (conceptually) a query that finds departments where the average salary is above the company-wide average, using a CTE, and explain why doing this with a correlated subquery per row would be far less efficient at scale.
11. You need "the 3rd most recent order per customer, including ties on `order_date`." Explain precisely which ranking function you'd use and why `ROW_NUMBER()` would silently drop a legitimate tied row.
12. A recursive CTE walking a management hierarchy never terminates in production despite working in dev. What are the most likely causes (data-side, not syntax) and how do you defend against them?
13. Explain why `SELECT DISTINCT salary ... LIMIT 1 OFFSET 1` breaks the moment there's a duplicate top salary, using the actual row semantics of `OFFSET`.
14. A model trained with `class_weight='balanced'` produces well-separated classes but badly miscalibrated probabilities post-training. Explain the root cause and the fix, and why the fix must be applied on the original (not resampled) distribution.
15. Focal loss with γ=2 is applied to a tabular fraud dataset that already uses `scale_pos_weight`. Explain why stacking these two techniques can over-correct, and what symptom you'd see in the precision-recall curve if it does.
16. A feature store computes `user_last_7d_purchases` from a nightly batch job in training but serves it from a 6-hour-lag Redis cache in production. Explain exactly how this creates train-serve skew even though the feature "exists" identically in both places, and propose two different fixes with different tradeoffs.
17. You detect PSI > 0.25 on a feature that has strong, known seasonality. Explain why this might be a false alarm, and describe a monitoring design that distinguishes seasonal drift from genuine concept drift.
18. Explain why Kappa architecture requires Kafka retention to cover the entire reprocessing window, and what happens to a backfill job if retention is shorter than the required replay range.
19. A junior engineer adds SMOTE inside an `imblearn.Pipeline` but leaves `StandardScaler` outside the pipeline, fit once on the full dataset beforehand. Identify every leakage path that remains despite using the "safe" resampling pattern.
20. You must choose between Lasso and tree-based embedded selection (MDI) on a dataset with 40% pairwise-correlated feature groups and a mix of linear and non-linear signal. Justify a hybrid selection strategy and explain the specific failure mode each method alone would hit.
