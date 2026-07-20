# Matchmaking (FIFA Online) — 60-Min Cheat Sheet

Fixed ±200 Elo search radius causes long queues off-peak, lopsided matches when radius widens, and smurfs ruining beginner lobbies. Design a real-time, production-grade skill-based matchmaking (SBMM) system that balances fairness vs. queue time.

## Core Architecture
```
Queue join → Feature extraction (Elo, 7d win rate, playstyle cluster,
             ping zone, time-in-queue)
           → Matchmaking Queue (per-region, priority by wait time)
               - Elo search radius expands with wait time (capped ±200)
               - Hard ping filter (≤80ms)
               - Win-probability model gates match: accept iff P∈[0.45,0.55]
           → Match created → Lobby → Game server allocation
           → Post-match: Elo update + label fed back into daily retrain
```
- **Model choice:** LightGBM (GBDT) on feature-diff vectors, not a neural net — sub-1ms inference is a hard requirement for the tick loop.
- **Calibration:** Platt scaling — without it, the [0.45,0.55] window is meaningless since raw scores aren't true probabilities.
- **Smurf handling (v1):** soft signal — bottom-20%-Elo + >80% recent win rate → hidden MMR overrides displayed Elo, not shown to player.

## Talking Points That Signal Seniority (say these proactively)
- Call out the Elo-lag problem unprompted: Elo only updates post-match, so a player on a win streak is under-rated mid-streak — propose dynamic K-factor or blending in 7-day rolling win rate.
- State the calibration requirement explicitly — a fairness window on uncalibrated scores is meaningless.
- Propose Redis Sorted Set (ZADD/ZRANGEBYSCORE) as the shared queue for HA/statelessness across pods, not just in-memory.
- Flag sparse-region/off-peak fallback design: progressively loosen the fairness window and ping cap with wait time, then PvAI fill with disclosure, before anyone asks.
- Propose smurf detection as an explicit v2 roadmap item (account age + match-history variance + classifier), even though out of scope for v1.
- Mention PSI (Population Stability Index) for detecting feature/meta drift on the win-prob model, and calibration-curve monitoring in production.
- Flag survivorship bias risk when discussing any A/B test design (tighter fairness window → higher dropout → biased remaining population).
- Note TrueSkill (Bayesian, tracks uncertainty) as a principled upgrade path over Elo, especially for new players and win-streak convergence.

## Top 3 Tradeoffs
- **LightGBM vs. neural net for win-prob:** GBDT gives <1ms in-process inference; a NN might be marginally more accurate but needs a GPU server and 10-50ms round trip — latency budget wins.
- **In-memory queue vs. Redis-backed queue:** in-memory is fastest but loses all state on pod restart; Redis adds ~1ms but is durable and horizontally scalable — matters once HA is a requirement.
- **Search radius expansion rate:** the single knob trading queue time against match quality — wider/faster expansion clears queues faster but degrades fairness, and it's the first thing to tune under complaints.

## Biggest Pitfall
Proposing a neural network for the win-probability model without addressing the sub-millisecond latency constraint of the matchmaking tick loop — this alone reads as No Hire regardless of everything else being right.
