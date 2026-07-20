# Interview 21 — Automated Data Drift Detection Pipeline (Condensed)

Dozens of production models (Matchmaking, Churn, Fraud) degrade silently as game patches shift input distributions. Design a pipeline that detects covariate/concept drift on input features and alerts *before* accuracy drops — no real-time ground truth available.

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

## Biggest Pitfall
Proposing blind nightly auto-retraining as the "fix" for drift (without a data-quality gate or human review) — this is the single fastest way to go from Hire to No-Hire, since it risks retraining on corrupted data and wastes compute on every trivial fluctuation.
