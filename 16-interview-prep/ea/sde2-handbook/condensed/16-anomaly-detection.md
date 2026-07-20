# Interview 16 — Real-Time Anomaly Detection on Streaming Data (Condensed)

Design an ML system that monitors 1-second server telemetry (CPU, memory, bandwidth, players, latency) across EA's game server fleet and fires an alert when a server behaves erratically (memory leak, DDoS, hardware failure) before players notice.

---

## Core Architecture

```
Servers (1Hz) → Kafka → Flink (1-min tumbling window, group by server_id)
              → Inference Service (Isolation Forest, score_samples)
              → if score < 99.9th-pct threshold → PagerDuty / auto-remediation
```

- **Kafka** — ingest 100k servers streaming telemetry.
- **Flink windowing** — smooth 1s noise into 1-min rolling averages per server.
- **Isolation Forest (unsupervised)** — chosen over Autoencoder: CPU-only, cheap at 100k-server scale, no labeled data needed, handles multivariate correlation via isolation path length.
- **Threshold = 99.9th percentile of training scores** — hardcoded, not `contamination='auto'`, to control FP rate explicitly.
- **Inference service** — FastAPI wrapping the model, ~1ms/vector, horizontally scaled behind LB.
- **Alarm suppressor** — collapse fleet-wide simultaneous alerts (e.g. AWS outage) into one "global anomaly" event.

---

## Talking Points That Signal Seniority

- Proactively distinguishes point anomalies (spikes) from trend anomalies (slow leaks) and proposes a rolling-slope/derivative feature for the latter.
- Flags that a 99.9th-percentile threshold trades false negatives for false positives — states this tradeoff unprompted.
- Raises contaminated-retraining-data risk: use yesterday's model to filter anomalies out of tomorrow's training set before retraining.
- Proposes SHAP/TreeSHAP to attach top contributing features to each alert so it's actionable, not just "anomaly detected."
- Mentions "Shadow Mode" for rolling out a retrained model after a known structural break (e.g., new game mode) before re-enabling alerting.
- Suggests embedding the model as a Flink UDF instead of an HTTP call per-event when scaling past ~1M servers, to avoid network bottleneck.
- Calls out Event-Time windowing with watermarks (not processing-time) so delayed/out-of-order Kafka data doesn't corrupt windows.
- Considers region/cluster as a feature or separate regional models, since regional infra differences aren't truly anomalous.

---

## Top 3 Tradeoffs

- **Isolation Forest vs Autoencoder** — Autoencoders model deep non-linear correlations better but need GPU train/serve; Isolation Forest is CPU-cheap and interpretable, better fit at EA's 100k-server scale.
- **Strict (99.9th pct) vs loose alerting threshold** — strict keeps DevOps trust high but misses slow-burn anomalies until they hit critical mass.
- **1-minute window vs true real-time** — smooths GC-spike noise but delays detection of a full lockup by up to 60s.

---

## Biggest Pitfall

Proposing supervised learning (e.g., XGBoost classifier) for a problem explicitly stated to have no labeled failure data — signals the candidate didn't internalize the constraint and drops straight to No Hire territory.
