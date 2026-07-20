# Interview 29 — AI-Driven Dynamic Game Difficulty (Condensed)

**Problem:** Single-player Action RPG — players quit on difficulty spikes or get bored when it's too easy. Design a DDA (Dynamic Difficulty Adjustment) AI system that tunes enemy stats/spawns in real time to keep players in "Flow State" without them noticing.

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

## Biggest Pitfall

Proposing full deep RL (e.g., PPO controlling the game engine) for what is fundamentally a lightweight contextual-bandit/statistical-tracking problem — massive over-engineering that reads as not understanding the actual constraints (real-time, on-device, explainable, low-data-per-player).
