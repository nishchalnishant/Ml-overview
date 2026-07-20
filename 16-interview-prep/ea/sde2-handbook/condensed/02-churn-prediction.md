# Churn Prediction & Intervention (Apex Legends) — Cheat Sheet

MoM active users are dropping. Design + build a system to predict player churn and trigger automated interventions (email, XP boost, cosmetics) before players leave for good.

## Core Architecture
- Snowflake/warehouse → nightly ETL → rolling 7d/14d/30d feature aggregates (playtime, K/D trend, social activity).
- Airflow DAG triggers batch inference (Spark/Python) — not real-time, this is a daily-batch problem.
- Model: XGBoost/LightGBM binary classifier — fast on tabular data, handles missing values, no need for DL complexity here.
- Output: player_id, P(churn) → rules engine buckets into risk tiers (>0.90 cosmetic, >0.75 XP boost, >0.60 email).
- Rules engine calls LiveOps API / CRM to execute interventions.
- Class imbalance handled via `scale_pos_weight`; evaluate with PR-AUC + decile lift, not accuracy.

## Talking Points That Signal Seniority
- Ask about churn definition and intervention cost asymmetry before writing any code.
- Flag that cohort should be "active in last 14 days," not "active today" — otherwise mid-churn players (day 3–13 of inactivity) are invisible to scoring.
- Proactively propose a global holdout/control group (~5%) to measure actual causal lift, not just prediction accuracy.
- Mention uplift modeling (CATE) as the real fix — avoid wasting cosmetics on "sure things" and "lost causes," target only "persuadables."
- Call out data leakage risk in defining the training feature window vs. label window.
- Note that a batch CronJob/Airflow DAG is the right call here — pushback on anyone proposing Kafka/Flink real-time for a clearly batch problem.
- Mention chunked/generator-based inference (`pd.read_sql(chunksize=...)`) as the fix for OOM before jumping to Spark.
- Propose drift monitoring on output distribution (e.g., alert if P(churn)>0.90 for 50% of players) to catch feature pipeline breaks early.

## Top 3 Tradeoffs
- **Batch vs. real-time inference** — batch is far cheaper/simpler but misses the exact moment of a rage-quit; matches business need here (daily interventions, not instant).
- **Rule-based thresholds vs. uplift modeling** — thresholds are simple/interpretable but ignore incrementality; uplift targets only persuadable players, saving cosmetics budget.
- **XGBoost vs. deep learning (LSTM on sessions)** — XGBoost trains/deploys faster and handles missing data natively; DL could capture sequential patterns but adds deployment cost with limited tabular data payoff.

## Biggest Pitfall
Treating this as a pure modeling exercise — tuning XGBoost hyperparameters while missing that a model without a control-group evaluation and cost-aware thresholds can't prove it's actually reducing churn, and proposing real-time streaming infra for a problem the business only needs solved daily.
