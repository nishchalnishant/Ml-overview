# Crash Prediction Cheat Sheet

Predict if a game client will crash in the next 60 seconds from live telemetry (memory, CPU temp, frame drops), so the client can auto-save or degrade gracefully before it happens. The catch: inference must run on the player's device, under brutal latency/size constraints.

## Clarifying Questions
- Server or client-side inference? → **Client**. Round-tripping 1Hz telemetry to a server is too slow/expensive at scale.
- Performance budget? → **<1ms inference, <10MB RAM**, cannot drop framerate.
- What's the training data? → Historical 1Hz telemetry per session, labeled `crashed=True/False` at session end.
- How rare are crashes? → **~1 in 10,000 sessions** (extreme class imbalance).
- What action fires on a positive prediction, and how long does it take? → Emergency auto-save + network flush, **~2 seconds** — so the label must be a leading indicator, not a last-second one.

## Core Architecture
- **Offline (cloud):** ingest petabytes of telemetry, heavily undersample healthy sessions, train a small **XGBoost** model on rolling-window features.
- **Compile for edge:** convert model to native C via **Treelite/ONNX** — no Python runtime on the client.
- **Ship via OTA:** push the compiled model as a small (~1MB) file, not a full game patch.
- **Client thread (1Hz):** ring buffer of last 10s → feature extraction (rolling mean/max/variance/delta) → C++ model eval (~0.01ms) → if `P(crash) > 0.9` trigger auto-save thread.
- **Why XGBoost over LSTM:** tree model compiles to pure `if/else` branching — near-zero CPU cost; LSTM is theoretically better for sequences but too heavy for a game's main thread.

## Talking Points That Signal Seniority
- Proactively says this must be **edge ML**, not a server API — REST-per-second from a game client is a red flag design.
- Names **PR-AUC** as the eval metric, not accuracy/ROC-AUC, given 1:10,000 imbalance.
- Proposes **undersampling the negative class** (e.g., to 1:10) plus `scale_pos_weight` rather than naively training on raw distribution.
- Raises **counterfactual evaluation**: run in "shadow mode" for a holdout % of players — predict but don't act — to measure true prevention rate.
- Suggests replacing rolling windows with an **EMA (O(1) memory/compute)** when engineers say rolling variance is hard to implement in C++.
- Flags that **false positives are worse than the crash** — an unnecessary 2s auto-save stutter hurts UX, so precision must be tuned strictly.
- Mentions a **kill switch** (cloud config flag) to disable the model fleet-wide instantly if covariate shift causes FP storms after a patch.
- Excludes **non-predictable crash types** (null pointer/logic bugs) from training — only keep crashes with leading indicators (thermal, OOM, resource exhaustion).

## Top 3 Tradeoffs
- **Client-side vs server-side:** client has zero network latency/cloud cost but forces extreme model compression; server allows big models but is slow and costly at scale — client wins here.
- **XGBoost vs LSTM:** LSTM captures temporal patterns more naturally, but Treelite-compiled trees run in microseconds on CPU with no runtime — required given the 1ms/10MB budget.
- **Precision vs recall:** over-triggering auto-saves (FPs) directly degrades gameplay every time, so precision is prioritized over catching every crash.

## Toughest Follow-ups
**Q: Engineers can't implement rolling 10s variance efficiently in C++ — how do you adapt?**
Use an EMA (`EMA_t = value*alpha + EMA_{t-1}*(1-alpha)`) — O(1) state, no window buffer needed. Alternatively move to a minimal GRU/RNN via ONNX where the hidden state itself carries temporal memory, so the engine only passes instantaneous features.

**Q: If auto-save prevents the crash, how do you know the model was actually correct?**
Classic counterfactual problem — run a shadow-mode holdout (e.g. 5% of players) where the model predicts and logs but never triggers the action, then check offline whether those sessions crashed anyway.

**Q: 50 different crash types exist (GPU hang, OOM, null pointer) — one model or 50?**
Neither extreme: group only crash types with real leading indicators (thermal throttling, memory exhaustion) into one binary target; drop logic-bug crashes (null pointer) entirely since they're instantaneous with no telemetry buildup for ML to learn from.

## Biggest Pitfall
Designing a server-side/REST-based architecture (client pings server every second) — this single choice signals no understanding of game engine constraints and is close to an automatic No Hire regardless of ML depth elsewhere.
