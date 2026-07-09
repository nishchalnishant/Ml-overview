# Interview 29 — AI-Driven Dynamic Game Difficulty Balancing
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer working on a single-player Action RPG (e.g., Star Wars Jedi: Survivor). Players frequently complain about difficulty spikes—bosses that are too hard cause them to quit the game permanently, while levels that are too easy cause them to get bored.

Your task is to **design a Dynamic Difficulty Adjustment (DDA) AI system** that automatically adjusts the game's difficulty in real-time (e.g., enemy health, damage, attack speed) to keep the player in the optimal "Flow State" without them realizing the game is helping them.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Metric of success (How do we mathematically define the "Flow State"?)
- Player Agency (Will players feel cheated if they grind for better gear, but the AI just makes the enemies stronger to compensate?)
- Latency (Does this adjustment happen mid-fight, or between levels?)
- Action Space (What variables can the AI actually tune?)
- Algorithm (Rule-based vs Reinforcement Learning vs Multi-Armed Bandit?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"How do we define the 'Flow State'?"**
   → *Answer: We aim for a specific win/loss ratio, e.g., the player should win 80% of encounters on the first try, but reach <20% health during boss fights to create tension.*

2. **"Does the difficulty change dynamically mid-fight?"**
   → *Answer: No, changing an enemy's health bar mid-fight breaks immersion. Adjustments should happen out of sight, like when spawning the next room of enemies.*

3. **"What if the player intentionally plays badly to lower the difficulty for an achievement?"**
   → *Answer: Good point (Sandbagging). We need a system to detect intentional failures versus genuine struggles.*

---

## Part 4 — Expected Assumptions

- **Architecture:** Multi-Armed Bandit (Contextual Bandit) or a Bayesian Knowledge Tracing system. Standard supervised learning doesn't work well here because we are actively exploring policies.
- **Features:** Player telemetry (accuracy, parry timing, time-to-kill, death count).
- **Constraints:** Changes must be subtle (e.g., changing AI aggression or spawn counts, rather than openly reducing boss health by 50%).

---

## Part 5 — High-Level Solution

```
  [Game Client]
       │ 1. Player completes Encounter 1 (Sends metrics: damage taken, time)
       ▼
  [DDA Inference Engine (Local or Cloud)]
  ┌────────────────────────────────────────────────────────┐
  │ 1. Calculate Player Skill Estimate (e.g., Elo rating)  │
  │ 2. Contextual Bandit evaluates: Which difficulty "arm" │
  │    (Easy, Normal, Hard spawns) maximizes Flow?         │
  │ 3. Output selected difficulty configuration.           │
  └────────────────────────────────────────────────────────┘
       │
       ▼ 2. Load Encounter 2
  [Game Engine] ➔ Spawns enemies based on new configuration.
```

**Core ML Component:** A Contextual Bandit algorithm (like Thompson Sampling) that balances Exploration (trying a harder difficulty to see if the player improved) and Exploitation (keeping them at the current difficulty because they are struggling).

---

## Part 6 — Step-by-Step Implementation

### Step 1: Feature Extraction (Skill Estimation)
- Track trailing metrics (last 5 encounters):
  - `damage_avoidance`: % of enemy attacks successfully dodged/parried.
  - `time_to_kill`: Speed of clearing the room.
  - `consumables_used`: Did they burn all their healing potions?

### Step 2: Formulating the Reward (Flow State)
- Reward $R$ is high if the player succeeds but takes damage.
- $R = -10$ if player dies (Frustration).
- $R = -5$ if player takes 0 damage and kills instantly (Boredom).
- $R = +10$ if player wins but uses 80% of health/potions (Optimal Tension).

### Step 3: Contextual Bandit Model
- **Context:** The player's Skill Estimate (Features).
- **Action (Arms):** 5 discrete difficulty profiles (`[-2, -1, 0, 1, 2]`).
- **Algorithm:** Vowpal Wabbit (Contextual Bandits) or a custom Thompson Sampling script. The model predicts the expected Reward for each arm given the Context, and selects the best one.

---

## Part 7 — Complete Python Code

```python
"""
dynamic_difficulty.py - Contextual Bandit for DDA (Thompson Sampling)
"""
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contextual Bandit (Thompson Sampling with Bayesian Linear Regression)
# ---------------------------------------------------------------------------
class DDABandit:
    def __init__(self, num_arms=5, context_dim=3):
        self.num_arms = num_arms
        self.context_dim = context_dim
        
        # Bayesian priors for each arm (Difficulty Profile)
        # B_inv is the covariance matrix inverse, mu is the mean vector
        self.B_inv = [np.eye(context_dim) for _ in range(num_arms)]
        self.mu = [np.zeros(context_dim) for _ in range(num_arms)]
        
        # Exploration hyperparameter
        self.v = 0.5 

    def select_difficulty(self, context: np.ndarray) -> int:
        """Selects the best difficulty arm based on the player's context."""
        sampled_rewards = []
        
        for arm in range(self.num_arms):
            # Sample weights from the multivariate normal distribution
            cov = self.v**2 * self.B_inv[arm]
            sampled_theta = np.random.multivariate_normal(self.mu[arm], cov)
            
            # Predict expected reward (Flow State)
            expected_reward = np.dot(sampled_theta, context)
            sampled_rewards.append(expected_reward)
            
        # Choose the arm that maximizes expected flow
        best_arm = int(np.argmax(sampled_rewards))
        logger.info(f"Bandit chose Difficulty Level {best_arm} for context {context}")
        return best_arm

    def update_model(self, arm: int, context: np.ndarray, reward: float):
        """Updates the Bayesian priors after the encounter finishes."""
        # Outer product of context
        context_matrix = np.outer(context, context)
        
        # Update Covariance
        self.B_inv[arm] = np.linalg.inv(np.linalg.inv(self.B_inv[arm]) + context_matrix)
        
        # Update Mean
        self.mu[arm] = self.B_inv[arm] @ (np.linalg.inv(self.B_inv[arm]) @ self.mu[arm] + context * reward)
        logger.info(f"Updated Arm {arm} with Reward {reward}")

# ---------------------------------------------------------------------------
# Reward Calculation (Flow State Logic)
# ---------------------------------------------------------------------------
def calculate_flow_reward(player_died: bool, health_remaining_pct: float) -> float:
    if player_died:
        return -10.0 # Frustration
    elif health_remaining_pct > 0.90:
        return -5.0  # Boredom (Too Easy)
    elif 0.10 <= health_remaining_pct <= 0.30:
        return 10.0  # Optimal Tension (Survived, but it was close)
    else:
        return 5.0   # Good, but could be tighter

# ---------------------------------------------------------------------------
# Execution Simulation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bandit = DDABandit(num_arms=3, context_dim=2) # Arms: Easy, Med, Hard. Context: [Accuracy, ParryRate]
    
    # 1. Player enters room (High skill player context)
    player_context = np.array([0.85, 0.70]) # 85% accuracy, 70% parry
    
    # 2. Bandit selects difficulty
    # Since it's untrained, it will explore randomly (e.g., selects Arm 0: Easy)
    chosen_diff = bandit.select_difficulty(player_context)
    
    # 3. Game plays out... (Player easily wins on Easy)
    reward = calculate_flow_reward(player_died=False, health_remaining_pct=0.95) # Boredom
    
    # 4. Update Model
    # The model learns that [High Skill] + [Easy] = Negative Reward
    bandit.update_model(chosen_diff, player_context, reward)
```

---

## Part 8 — Deployment

### Edge vs Cloud
- **Edge (Local Client):** Highly preferred. Single-player games should work offline. The Bandit algorithm (Bayesian Linear Regression) requires minimal RAM and CPU, making it perfect to run locally in C++ or via ONNX.
- **Global Prio Sync:** When the player goes online, the local model can upload its learned `mu` and `cov` matrices to EA servers. We can average these matrices across millions of players to provide better "Zero-Shot" priors for new players.

---

## Part 9 — Unit Testing

```python
import numpy as np
from dynamic_difficulty import DDABandit, calculate_flow_reward

def test_flow_reward_logic():
    assert calculate_flow_reward(True, 0.0) == -10.0
    assert calculate_flow_reward(False, 0.95) == -5.0
    assert calculate_flow_reward(False, 0.20) == 10.0

def test_bandit_convergence():
    bandit = DDABandit(num_arms=2, context_dim=1)
    context = np.array([1.0])
    
    # Force train the bandit to realize Arm 1 is strictly better than Arm 0
    for _ in range(50):
        bandit.update_model(0, context, -10) # Arm 0 bad
        bandit.update_model(1, context, 10)  # Arm 1 good
        
    # After training, it should reliably pick Arm 1
    # Turn off exploration for the test
    bandit.v = 0.0 
    assert bandit.select_difficulty(context) == 1
```

---

## Part 10 — Integration Testing

- **Simulated Player Bots:**
  - Create 3 hardcoded Player Bots in the game engine: "Noob" (misses 50% of shots), "Average", and "Pro" (perfect parries).
  - Let them play 100 encounters with the DDA system active.
  - Assert that the "Noob" bot stabilizes on Difficulty Arm 0 (Easy), and the "Pro" stabilizes on Arm 4 (Very Hard). If they cross over, the context mapping is flawed.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Sandbagging (Intentional Throwing)** | Players will realize dying 3 times makes the boss easier. They will intentionally die to farm achievements. **Solution:** Track the *variance* of skill. If a player parries 90% of attacks in the level, then suddenly drops their controller at the boss and dies 3 times doing nothing, the system detects this anomaly (intentional failure) and freezes the difficulty. |
| **RPG Progression Override** | If the player spends 10 hours grinding to get the "Ultimate Sword of Doom", but the DDA instantly scales enemy health by 500% to compensate, the player's progression feels meaningless. **Solution:** Cap the DDA bounds. The DDA can only tweak stats by $\pm 15\%$. If they have the Ultimate Sword, they *should* feel overpowered for a while. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Rule-Based (Heuristics) vs Bandits (ML) | Rule-based (`if deaths > 3: lower_difficulty()`) is industry standard (e.g., Resident Evil 4). It is simple and predictable. ML Bandits optimize the exact flow state mathematically and discover hidden relationships (e.g., High accuracy + low parry = specific enemy type), but require careful reward tuning to prevent weird behavior. |
| Visible vs Invisible Scaling | Visible (giving the player a prompt: "You died a lot, want to switch to Easy?") respects player agency but hurts pride. Invisible scaling (DDA) preserves ego but risks angering hardcore players who feel the game is manipulating them. Always add an option in the Settings to disable DDA. |

---

## Part 13 — Alternative Approaches

1. **Elo Rating System (Glicko-2):** Treat the Player and the Game AI as chess opponents. If the Player beats a room quickly, their Elo goes up. The Spawner selects enemies whose combined Elo matches the Player's Elo. Highly proven, mathematically sound, and requires no neural networks.
2. **Behavioral Cloning (Inverse RL):** Train a model to mimic how the game's Lead Designer manually adjusts difficulty during QA playtests, and deploy that model to automate the designer's intuition.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| The "Pity" Feedback Loop | Player is struggling, DDA makes it easy. Player wins. DDA thinks they improved and makes it hard. Player dies. Oscillates wildly. | Add **Hysteresis** (momentum). Make the algorithm highly reluctant to increase difficulty after a recent decrease. It requires 3 consecutive perfect wins to increase, but only 1 loss to decrease. |
| Cooperative Multiplayer | Player 1 is a pro, Player 2 is a beginner. Whose context does the DDA use? | Use the "Max Skill" heuristic for boss mechanics (so the pro isn't bored), but use "Average Skill" for raw damage numbers. Or, dynamically scale damage *per player* (Boss takes more damage if shot by the beginner). |

---

## Part 15 — Debugging

**Symptom:** The Bandit model assigns the "Very Hard" profile to a player who has died 5 times in a row, violating the logic of Flow State.

**Debugging steps:**
1. Check the Context feature vector. 
2. Discover the player has 95% Accuracy, but keeps falling off a cliff (environmental death).
3. The model only looks at combat stats (Accuracy, Parry). Because accuracy is 95%, it assumes the player is a Pro. It doesn't know about environmental deaths.
4. **Fix:** Update the feature engineering. Distinguish between `Combat_Deaths` and `Environment_Deaths`. Do not scale up combat difficulty if the player is struggling with platforming.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `average_deaths_per_boss` | > 10 → The DDA maximum bound is too high. Players are getting stuck. |
| `dda_variance_index` | If 90% of all players end up in "Arm 0 (Easy)", the game's baseline tuning is vastly too difficult. |

---

## Part 17 — Production Improvements

1. **AI Archetype Shifting:** Don't just change health and damage (bullet sponges are boring). Change the *behavior* tree. If the player relies entirely on shields, the DDA should spawn enemies with shield-piercing attacks. If the player snipes, the DDA spawns flanking melee units. This is "Tactical DDA".
2. **Director AI (Left 4 Dead):** Instead of tuning stats, tune the pacing. If the player's "Stress Level" is high (low health, low ammo), the Director stops spawning enemies for 60 seconds to let them breathe, overriding the standard spawn logic.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"The Thompson Sampling algorithm requires exploration. It might randomly assign 'Very Hard' to a beginner just to see what happens. This ruins their experience. How do you constrain exploration in production?"**
2. **"We want to implement the Elo system you mentioned earlier instead of Bandits. If a player takes 5 minutes to clear a room, but they took 0 damage, did they 'Win' or 'Lose' the Elo exchange against the room?"**
3. **"Players discover the DDA system exists, and it goes viral on Reddit. Players are furious that the game 'lies' to them. As an AI Engineer, how do you handle the PR and transparency of this system?"**

---

## Part 19 — Ideal Answers

**Q1 (Safe Exploration):**
> "We implement Epsilon-Safe Exploration. We never allow the algorithm to jump more than 1 difficulty tier at a time. If the player is on Tier 1 (Easy), the bandit is only allowed to explore Tier 2. It is hard-blocked from exploring Tier 5. Furthermore, we decay exploration ($\epsilon$) heavily based on time—after the first 2 hours of gameplay, we lock in the model and stop exploring entirely."

**Q2 (Elo Win/Loss Definition):**
> "Elo requires a binary outcome (1 = Win, 0 = Loss, 0.5 = Draw). Taking 5 minutes with 0 damage is ambiguous. We must define a multi-dimensional margin of victory. We can use a combined heuristic: `Score = (Expected_Time / Actual_Time) * Health_Remaining`. We then feed this continuous score into a modified Glicko rating system rather than standard binary Elo."

**Q3 (Player Agency & PR):**
> "Transparency is key. First, we rename the system in marketing from 'Dynamic Difficulty' to 'Adaptive AI Director' (focusing on pacing rather than nerfing). Second, we strictly adhere to the RPG Progression Override rule: we never invalidate player gear upgrades. Finally, we add a toggle in the menu: 'Standard Mode (Fixed Difficulty)' and 'Director Mode (Dynamic Experience)'. Giving players the choice eliminates the backlash."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Understands the psychological difference between Flow State and raw winning (Boredom vs Frustration).
- Chooses a Contextual Bandit or Elo system over standard Supervised Learning.
- Addresses the "RPG Progression" and "Sandbagging" constraints without prompting.
- Provides a mathematically sound definition of the Reward function.

### Hire
- Writes a working Bandit / RL loop.
- Suggests tracking reasonable player metrics (accuracy, time, health).
- Understands that difficulty shouldn't change mid-fight (immersion breaking).

### Lean Hire
- Suggests hardcoded `if/else` statements instead of ML. (The interviewer specifically asked for an ML AI system).
- Fails to realize that exploring extreme difficulties will cause players to quit.

### Lean No Hire
- Proposes deep reinforcement learning (PPO) to control the game engine, completely over-engineering a problem that requires a simple statistical tracker.

### No Hire
- Does not understand basic game design concepts.
- Cannot write code to track simple metrics.
