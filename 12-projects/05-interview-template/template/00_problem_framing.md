# 00 — Problem Framing (do this before writing any code)

Say this out loud, in this order, before touching the keyboard. ~2 minutes.

## 1. Clarify the target
- What exactly are we predicting? Binary / multiclass / regression / ranking?
- What's the label window? ("churn in next 30 days" is a different problem than "ever churns")
- How is the label generated — is it observed directly, or a proxy (e.g. "no login in 30d" as a
  proxy for churn)? Proxies can be wrong; say so.

## 2. Clarify the data
- Row grain: one row per user? per session? per event? (Determines CV strategy later — see `05`.)
- Time range available, and is there a natural train/test time boundary?
- Class balance (for classification) — ask or check immediately; it decides the metric.

## 3. Clarify the constraint that decides architecture
- **Latency**: real-time (<100ms, single request) vs. batch (nightly, can take hours)?
  This single answer determines serving architecture — state it before designing anything.
- **Compute budget**: is there a GPU? How many HPO trials can we actually afford?
- **Interpretability requirement**: does a human need to explain individual predictions
  (e.g. regulated decision) or is pure predictive performance fine?

## 4. State the metric before building anything
- Imbalanced classification → ROC-AUC and PR-AUC, not accuracy.
- Regression → MAE/RMSE, but check if the business cares about relative error (MAPE) instead.
- If predictions feed a downstream threshold/decision, note that calibration matters, not just
  ranking quality.

## 5. State the baseline you'll beat
- "Before the deep learning model, I'll fit a linear/logistic baseline and a gradient-boosted
  tree — if the DL model doesn't clearly beat GBT on tabular data, that's a valid and expected
  outcome to report, not a failure."

## 6. Say the leakage risks you're already watching for
- Any feature computed using information available only after the label's observation window.
- Row-level train/test split when multiple rows share an entity (user, player, session) — leaks
  identity across the split and inflates validation score.
- Preprocessing (scaling, imputation, target encoding) fit on the full dataset before splitting.

---

Once these six are answered (even provisionally — "I'll assume X, correct me if wrong"), move to
`01_feature_engineering.py`.
