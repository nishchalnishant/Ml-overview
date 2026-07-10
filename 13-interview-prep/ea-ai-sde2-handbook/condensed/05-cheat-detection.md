# Cheat Sheet — Real-Time Cheat Detection (Aimbot)

Server-side ML system to detect FPS aimbots purely from player telemetry (pitch/yaw/fire events), replacing bypassable client-side memory scanning. Must run at scale (millions of players, 60Hz) with near-real-time action (mid-match kicks).

## Clarifying Questions to Ask
- Telemetry frequency? → 60Hz tick, pitch/yaw/fire captured every 16ms.
- Mid-match or post-match action? → Must flag within ~2 min, mid-match kick.
- Auto-ban or human review? → Auto-kick at >99.9% confidence, else flag for manual review.
- Ground truth source? → 10k admin-verified matches (5k clean / 5k confirmed cheaters).
- Interpretability need? → Yes — CS needs a reason string for ban appeals.
- Class imbalance / cheat prevalence? → Not given upfront; assume rare-event, high-precision requirement.

## Core Architecture
```
Game Server (60Hz) → Kafka → Flink (windowing, only during combat engagements)
   → Feature extraction (snap speed, jitter, overshoot, direction changes)
   → XGBoost inference (FastAPI/gRPC)
   → Threshold logic (>0.999 ban, >0.90 review) → Action service (kick API)
```
- Pre-filter at the source: only stream telemetry while player is firing/aiming — cuts volume ~90%.
- Windowing: 1s before first shot + 4s after = "engagement window," not the whole match.
- Model: XGBoost on hand-crafted features — chosen over LSTM/1D-CNN for speed + explainability (ban appeals need a reason).
- Partition Flink state by `match_id` so a match's ticks land on one node.
- Action layer separate from inference — auto-ban only above a very conservative threshold.

## Talking Points That Signal Seniority
- Proactively flags the extreme data-volume problem (terabytes/hour) and proposes pre-filtering at the game server, not just at ingestion.
- Names the Controller/Aim-Assist vs Mouse+Keyboard confound before being asked — proposes separate models per input device.
- Volunteers SHAP (or equivalent) explanations tied to the ban record so CS can answer appeals with concrete reasons.
- Proposes an ensemble/second-opinion step: cheap XGBoost triage → heavier sequence model (LSTM) only for borderline scores.
- Raises hardware fingerprinting to stop ban evasion via new accounts.
- Recognizes model drift as adversarial (cheat devs adapt) and proposes continuous retraining on newly banned players' data.
- Distinguishes aimbot detection from wallhack/ESP detection — notes engagement-only windowing misses pre-fire/wall-peeking behavior, would need spatial features.
- Treats the 0.999 auto-ban threshold as a deliberate false-negative-for-false-positive tradeoff, not an arbitrary number — explicitly ties it to PR-disaster risk of banning a legitimate pro player.

## Top 3 Tradeoffs
- **Server-side vs client-side ML** — client-side is cheap but trivially bypassed/tampered; server-side is tamper-proof but far more expensive at scale.
- **XGBoost vs sequence DL (LSTM/1D-CNN)** — XGBoost is fast, explainable, and appeal-friendly; DL captures subtler sequence patterns (humanized cheats) but is a black box and heavier to serve.
- **Aggressive vs conservative ban threshold** — a low threshold (e.g., 0.70) catches more cheaters but risks banning innocents (streamer/pro PR disaster); a high threshold (0.999) is safe but lets smarter cheats slip through, pushed to manual review instead.

## Toughest Follow-ups
**Q: Cheaters add artificial jitter to mimic human variance — your `var_pitch_firing` feature breaks. What now?**
Move from hand-crafted stats to raw-sequence deep learning (LSTM/1D-CNN) that can detect the synthetic *nature* of injected noise — e.g., FFT/frequency-domain features distinguish RNG jitter from human muscle micro-tremor. Hand-crafted features alone can't keep up with an adversarial arms race.

**Q: Controller Aim Assist looks like a weak aimbot — won't the model flag controller players?**
Yes, unless input device type is a required field in telemetry and separate models are trained per device (mouse+keyboard vs controller). Mixing the two just teaches the model to ban controller players.

**Q: 10M daily players, Flink windowing costs $50k/mo — cut volume 99% without losing coverage?**
Add a cheap heuristic pre-filter on the game server itself (e.g., K/D > 10, headshot% > 80%) before anything hits Kafka/Flink; only forward players that trip a heuristic, plus a small random sample (~5%) and 100% of player-reported cases, for coverage of novel cheats.

## Biggest Pitfall
Proposing (or defaulting to) client-side detection, or an auto-ban threshold around 0.70 without acknowledging the false-positive/PR blast radius — either signals no understanding of the adversarial/security context this system exists in.
