# Interview 11 — In-Game Economy Inflation Prediction (Cheat Sheet)

MMORPG auction house prices spiked 400%; designers suspect a gold-injection bug. Design an ML system to detect macro-inflation and exploits in near real-time.

## Core Architecture
```
Game Servers → Kafka → Flink (1hr tumbling window agg: gold_generated, gold_sunk, ah_volume)
   → InfluxDB (time-series store)
   → Anomaly Detection cron (Prophet: train on 30d, predict next hour, get yhat_upper/lower)
   → if actual > yhat_upper*1.1 → Slack/PagerDuty to economy designers
```
- **Model choice: Prophet** — handles daily/weekly seasonality natively, CPU-only, interpretable, robust to missing data (vs LSTM/Autoencoder which need GPUs + tuning).
- Track M0 (money supply) = prior M0 + Faucets − Sinks, plus AH price index (volume-weighted top commodities).
- Dynamic thresholds (forecast bounds) instead of static thresholds — avoids weekend false-alarm fatigue.
- Multivariate correlation check: faucet spike + AH price spike together = high-confidence exploit signal.
- Secondary drill-down: macro alert triggers a player-level Isolation Forest / graph job to find the abuser.

## Talking Points That Signal Seniority
- Frames the problem explicitly as Faucets vs Sinks time-series, not generic "anomaly detection."
- Proactively distinguishes macro (server) vs micro (player) detection and sequences macro-first.
- Justifies Prophet over LSTM/Autoencoder on interpretability + ops cost, not just accuracy.
- Flags that static thresholds break on weekends/patch days and cause alert fatigue.
- Mentions "boiling the frog" — a slow exploit staying just under threshold needs multi-resolution (hourly/daily/weekly) forecasting, not just one window.
- Proposes root-cause automation (e.g., KL-divergence contribution analysis by quest/item/zone) so designers aren't hand-writing SQL for hours.
- Handles structural breaks: patch days need baseline reset / truncated training window, not permanent anomaly flags.
- Suggests predictive "what-if" economy simulation as a designer-facing tool (e.g., impact of a new 1M-gold mount).

## Top 3 Tradeoffs
1. **Prophet vs LSTM** — LSTM handles complex multivariate signal but needs GPUs, tuning, and is opaque; Prophet is fast, interpretable, good enough for seasonal macro trends.
2. **Dynamic vs static thresholds** — static breaks every weekend/patch; dynamic (forecast-bound) thresholds prevent alert fatigue but require retraining discipline.
3. **Macro time-series vs per-player classification** — per-transaction ML is computationally unfeasible at scale in real time; macro-first then drill down is the pragmatic sequencing.

## Biggest Pitfall
Proposing a static threshold (`if gold > 1M: alert()`) instead of a seasonality-aware model — this is the single fastest way to go from Hire to No Hire, since it breaks on every weekend/patch and shows no grasp of the core time-series nature of the problem.
