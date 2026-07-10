# Interview 29 — AI-Driven Dynamic Game Difficulty (Condensed)

**Problem:** Single-player Action RPG — players quit on difficulty spikes or get bored when it's too easy. Design a DDA (Dynamic Difficulty Adjustment) AI system that tunes enemy stats/spawns in real time to keep players in "Flow State" without them noticing.

---

## Clarifying Questions to Ask

- **How do we define "Flow State"?** → Target ~80% win rate on first try, but <20% HP remaining in boss fights (tension).
- **Does difficulty change mid-fight?** → No — changes only happen out of sight (e.g., next room spawn), to preserve immersion.
- **What if players sandbag (intentionally fail) to farm an easier mode/achievement?** → Yes, real risk — need to detect intentional failure vs genuine struggle.
- **What's the action space — what can the AI actually tune?** → Subtle levers only (spawn count, aggression), not visibly halving boss HP.
- **Algorithm class expected?** → Contextual Bandit / Bayesian tracking, not supervised learning (we're exploring policies, not fitting static labels).
- **Latency/compute environment?** → Runs client-side/offline-capable (single-player, no guaranteed connectivity).

---

## Core Architecture

```
Game Client → telemetry (accuracy, parry rate, time-to-kill, deaths)
   → DDA Inference Engine: skill estimate → Contextual Bandit picks difficulty "arm"
   → Game Engine loads next encounter with chosen config
   → outcome/reward flows back to update bandit
```

- **Core ML choice:** Contextual Bandit (Thompson Sampling, Bayesian linear regression) — balances exploration (try harder difficulty) vs exploitation (keep current level); no need for full RL/PPO since the action doesn't need long-horizon credit assignment.
- Context = trailing player metrics (last ~5 encounters): damage avoidance, time-to-kill, consumables used.
- Reward shaped explicitly for flow: death = frustration penalty, flawless clear = boredom penalty, close call w/ survival = max reward.
- Arms = discrete difficulty tiers (e.g., -2..+2), not a continuous action space.
- Deployment: runs **on-device/edge** (cheap Bayesian model, works offline); optionally syncs learned priors to server to bootstrap new players ("zero-shot" global prior).

---

## Talking Points That Signal Seniority

- Explicitly separates Boredom vs Frustration as two distinct failure modes of the reward function, not just "win/loss."
- Proactively raises **sandbagging** — detects anomalous skill-variance (e.g., 90% parry rate then sudden intentional deaths) and freezes difficulty rather than reacting.
- Proactively raises **RPG progression override** — caps DDA adjustment magnitude (e.g., ±15%) so grinding for gear still feels meaningful.
- Mentions **hysteresis/momentum** — asymmetric thresholds (3 wins to go up, 1 loss to go down) to prevent oscillating "pity loop" feedback.
- Distinguishes combat deaths from environmental/platforming deaths in feature engineering, not lumping all deaths together.
- Suggests constraining exploration in production (never jump >1 tier, decay epsilon over session time) rather than naive Thompson Sampling.
- Raises multiplayer ambiguity — whose skill context drives difficulty (max-skill for mechanics, per-player scaling for damage).
- Frames the system as an "Adaptive Director" (pacing/behavior shifts, e.g. Left 4 Dead) rather than just stat-padding — bullet sponges are boring.

---

## Top 3 Tradeoffs

- **Rule-based heuristics vs Contextual Bandit:** rules (`if deaths>3: lower`) are simple, predictable, industry-proven; bandits optimize flow mathematically and find hidden feature interactions but need careful reward tuning to avoid weird emergent behavior.
- **Visible vs invisible difficulty scaling:** visible (prompt to switch to Easy) respects agency but bruises pride; invisible DDA preserves ego but risks backlash if players discover it and feel manipulated — always ship an opt-out toggle.
- **Elo/Glicko vs Bandit:** Elo is simple, proven, no ML infra needed, but needs a well-defined win/loss margin; bandits handle richer context but are harder to explain/debug to designers.

---

## Toughest Follow-ups

**Q: Thompson Sampling might randomly assign "Very Hard" to a beginner just to explore. How do you constrain this in production?**
A: Hard-cap exploration to ±1 difficulty tier from current — never let the bandit jump straight to an extreme. Decay epsilon over session time (e.g., lock the model after ~2 hours of play) so long-term players get a stable, exploitation-only experience.

**Q: Using Elo instead — if a player clears a room in 5 min with zero damage, did they "win" or "lose" the exchange?**
A: Binary Elo can't represent this — need a continuous margin-of-victory score, e.g. `(expected_time/actual_time) * health_remaining`, fed into a Glicko-style continuous rating update instead of a 1/0/0.5 outcome.

**Q: The DDA goes viral on Reddit; players feel the game is "lying" to them. How do you handle it?**
A: Reframe publicly as an "Adaptive AI Director" focused on pacing, not stat-nerfing. Strictly honor the progression-override rule so gear upgrades never feel invalidated. Ship a menu toggle for Standard (fixed) vs Director (dynamic) mode — giving players the choice defuses the backlash.

---

## Biggest Pitfall

Proposing full deep RL (e.g., PPO controlling the game engine) for what is fundamentally a lightweight contextual-bandit/statistical-tracking problem — massive over-engineering that reads as not understanding the actual constraints (real-time, on-device, explainable, low-data-per-player).
