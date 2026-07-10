# Matchmaking (FIFA Online) — 60-Min Cheat Sheet

Fixed ±200 Elo search radius causes long queues off-peak, lopsided matches when radius widens, and smurfs ruining beginner lobbies. Design a real-time, production-grade skill-based matchmaking (SBMM) system that balances fairness vs. queue time.

## Clarifying Questions to Ask
- Peak CCU? → ~800k global, ~267k/region (NA/EU/APAC).
- Queue time SLA? → p50<30s, p95<90s, p99<3min.
- Players per match? → 1v1 (Ultimate Team) for v1, 11v11 exists too.
- Definition of "fair"? → win probability ∈ [0.45, 0.55].
- Parties/groups supported? → Not in v1, solo queue only.
- Ping constraint? → ≤80ms RTT hard filter.
- Existing smurf detection? → None — propose one.
- Cross-play? → Yes, with opt-out.

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

## Toughest Follow-ups
**Q: 50 players in APAC queue at 3am, fairness window means nobody matches. Walk me through what happens.**
A: Progressive degradation with wait time — loosen to [0.40,0.60] at 2min, [0.35,0.65] + ping cap to 120ms at 3min, then offer disclosed PvAI fill at 4min. Log every bot-fill event for regional capacity planning.

**Q: Model was trained 6 months ago, meta has shifted since a patch. How do you detect and respond?**
A: Track PSI weekly on key input features (>0.2 = significant shift), and monitor calibration by bucketing predicted probability vs. actual win rate — if the [0.45,0.55] bucket's real win rate drifts to 0.61, model is stale. Retrain on trailing 90-day data and shadow-test before promotion.

**Q: A/B testing a tighter window [0.48,0.52] vs current [0.45,0.55] — how do you avoid bias?**
A: Assign by hashed player_id (not match-level) to avoid contamination, and explicitly measure dropout/abandon rate per arm — the tight-window arm has longer queues, so quitters before match leave a skewed "survivor" population; report retention at 7/30 days alongside queue-time percentiles, not queue time alone.

## Biggest Pitfall
Proposing a neural network for the win-probability model without addressing the sub-millisecond latency constraint of the matchmaking tick loop — this alone reads as No Hire regardless of everything else being right.
