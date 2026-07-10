# Interview 21 — Automated Data Drift Detection Pipeline (Condensed)

Dozens of production models (Matchmaking, Churn, Fraud) degrade silently as game patches shift input distributions. Design a pipeline that detects covariate/concept drift on input features and alerts *before* accuracy drops — no real-time ground truth available.

## Clarifying Questions to Ask
- What stat tests for continuous vs categorical features? → propose KS-test / PSI (continuous), PSI or Chi-Square (categorical).
- What's the baseline? → the exact training dataset of the current prod model.
- Do we have real-time ground-truth labels? → No (e.g. churn label lags 30 days) — must detect on input features alone.
- How many features/models? → ~5,000 features across 20 models — scale matters.
- Do we auto-retrain or just alert? → alert only; retraining is a human decision (guard against blind auto-retrain).

## Core Architecture
- Nightly Airflow DAG (batch, not streaming) — drift is a slow signal, doesn't need real-time.
- Baseline = cached training distribution; live = 7-day rolling window (smooths weekday/weekend seasonality) + a 1-day window in parallel (catches acute pipeline breaks like sudden nulls).
- **Core technique: Population Stability Index (PSI)** over KS-test — PSI stays interpretable and stable at large row counts; KS-test's p-value collapses to <0.05 on any trivial shift at scale.
- Compute histograms/deciles **inside the warehouse** (Snowflake `WIDTH_BUCKET`), only pull the ~10 bin counts into Python — avoids OOM at 5,000+ features.
- Metrics written to a time-series/metrics store → Grafana dashboard; Slack/PagerDuty alert on threshold breach.
- Data-quality check (null-rate, type/schema check) runs *before* the drift check, gating it.
- Multivariate drift caught via a Domain Classifier (RF predicting train-vs-live) since PSI is univariate.

## Talking Points That Signal Seniority
- Proactively picks PSI over KS-test specifically because of the multiple-comparisons/large-N p-value collapse problem.
- Weights alerts by `PSI × Feature_Importance` (SHAP/model weight) so a drifted-but-unused feature doesn't page anyone.
- Distinguishes Data Quality bug (e.g. renamed JSON field → nulls) from genuine Concept/Covariate Drift, and refuses to auto-retrain on the former.
- Mentions target/prediction-distribution drift (model output shifting from 5%→20% fraud) as a zero-ground-truth-needed early warning.
- Proposes tracking SHAP-value drift instead of raw feature drift to separate "harmful" from "benign" drift.
- Flags the multiple-comparisons problem at 5,000 features (Bonferroni correction) instead of naively alerting per-feature.
- Computes histograms in-database (SQL) rather than pulling raw rows into Python — shows scale awareness.
- Separately proposes a Domain Classifier for multivariate drift PSI can't see.

## Top Tradeoffs
- **PSI vs KS-test:** KS is statistically stricter but always significant at massive scale; PSI is coarser but business-interpretable and stable — pick PSI as the primary alerting metric.
- **1-day vs 7-day window:** 1-day catches acute pipeline breaks fast but is noisy with seasonality; 7-day smooths seasonality but is slow to catch bugs — run both in parallel.
- **Alert-only vs auto-retrain:** Auto-retraining nightly on drifted data is tempting but risks retraining on corrupted/broken data (data-quality bugs masquerading as drift) — always gate retraining behind a human/DQ check.

## Toughest Follow-ups
**Q: How do you measure drift on a high-cardinality categorical (10,000 distinct `item_name` values)?**
PSI buckets go empty/sparse with high cardinality. Bucket the long tail into "Other" before computing PSI, or embed the category into a dense vector and track drift of the embedding centroid over time instead of raw category frequencies.

**Q: With 5,000 features at 95% confidence, you get ~250 false positives/day by chance. How do you stop alert spam?**
This is the multiple-comparisons problem — apply a Bonferroni correction to tighten the per-feature threshold, and multiply drift score by feature importance so only top-N business-relevant features can page a human; the rest log silently to a dashboard.

**Q: How would SHAP make a better drift detector than raw PSI?**
Track drift of each feature's SHAP value distribution, not just the raw feature distribution. If a feature's input distribution shifts but its SHAP contribution stays stable, the model's decision logic is unaffected — this isolates "benign" drift from drift that actually changes predictions.

## Biggest Pitfall
Proposing blind nightly auto-retraining as the "fix" for drift (without a data-quality gate or human review) — this is the single fastest way to go from Hire to No-Hire, since it risks retraining on corrupted data and wastes compute on every trivial fluctuation.
