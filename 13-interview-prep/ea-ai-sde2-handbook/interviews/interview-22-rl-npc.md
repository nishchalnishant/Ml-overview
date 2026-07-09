# Interview 22 — Deep Reinforcement Learning for NPC AI
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the SEED (Search for Extraordinary Experiences Division) team at EA. Traditional game NPCs are driven by Behavior Trees (if player is close -> shoot, if health is low -> run). This creates predictable and easily exploitable enemies.

Your task is to **design and train a Deep Reinforcement Learning (RL) agent to control an enemy NPC in a 1v1 shooter game.** The agent must learn to navigate, aim, and shoot dynamically, adapting to the player's behavior.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Observation Space (What does the agent "see"? Raycasts vs Pixels?)
- Action Space (Continuous vs Discrete? e.g., move_x, move_y, shoot)
- Reward Function (How do we define "good" behavior without encouraging exploits?)
- Training Infrastructure (How do we run millions of simulation steps fast enough?)
- Game Engine Integration (Unreal Engine? Frostbite? Unity?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"What is the Observation Space? Is it pixels from a camera, or raw game state data?"**
   → *Answer: Pixels are too slow to train. We will use Vector Observations (Agent position, Player position, Health, Raycasts for walls).*

2. **"What is the Action Space?"**
   → *Answer: Continuous for movement (joystick X/Y), discrete for shooting (binary 0/1).*

3. **"How does the game engine communicate with the Python RL training loop?"**
   → *Answer: Assume we have a gRPC bridge between the game engine (Frostbite) and Python, similar to Unity ML-Agents.*

4. **"What is the reward function?"**
   → *Answer: You need to design it. The goal is to kill the player and survive.*

---

## Part 4 — Expected Assumptions

- **Algorithm:** Proximal Policy Optimization (PPO). It is the industry standard for continuous control in gaming because it balances sample efficiency with training stability.
- **Framework:** Ray RLlib for distributed training.
- **Reward Shaping:** Sparse rewards (only rewarding at the end of the match) will fail. Must use dense reward shaping.

---

## Part 5 — High-Level Solution

```
  [Game Engine (Frostbite / Unreal)] -> Runs 100 headless instances
       │ (Observations: Raycasts, Player relative pos)
       │ (Rewards: +1 hit, -1 hit taken)
       ▼ (gRPC / Shared Memory)
  [Ray RLlib (Python)]
  ┌────────────────────────────────────────────────────────┐
  │ Rollout Workers: Collect experiences (State, Action, Reward)
  │ Trainer: Updates PPO Neural Network (Actor-Critic)     │
  └────────────────────────────────────────────────────────┘
       │ (Actions: Move X, Move Y, Shoot)
       ▼
  [Game Engine] -> Steps simulation forward 1 frame.
```

**Core ML Component:** Designing a robust Reward Function and Observation Space. RL algorithms will exploit poorly designed rewards (e.g., if you reward "survival time", the agent will glitch into a wall and hide forever).

---

## Part 6 — Step-by-Step Implementation

### Step 1: Observation Space Design (State)
- **Vector State (Size ~50):**
  - Relative vector to player `(dx, dy, dz)` normalized.
  - Line of Sight (Boolean).
  - Agent Health `(0.0 to 1.0)`.
  - Player Health `(0.0 to 1.0)`.
  - 360-degree Raycasts (e.g., 20 rays) returning distance to nearest wall/obstacle to avoid getting stuck.

### Step 2: Action Space Design
- Continuous actions for smooth movement: `[move_forward, move_right, aim_yaw, aim_pitch]` mapped to `[-1.0, 1.0]`.
- Discrete action for shooting: `[0 or 1]`.

### Step 3: Reward Shaping
- **Dense Rewards:** 
  - +0.1 for facing the player.
  - -0.01 penalty every step (encourages finishing the game quickly, avoids hiding).
- **Sparse (Terminal) Rewards:**
  - +10 for killing the player.
  - -10 for dying.

### Step 4: Algorithm (PPO)
- Use an Actor-Critic architecture. The Actor outputs the actions, the Critic estimates the Value (expected total reward from this state).

---

## Part 7 — Complete Python Code

*Note: We will implement the custom Gym Environment and configure Ray RLlib to train it.*

```python
"""
npc_rl_training.py - Distributed PPO Training for NPC Agent
"""
import logging
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Custom Gym Environment (Bridge to Game Engine)
# ---------------------------------------------------------------------------
class ShooterNPCEnv(gym.Env):
    def __init__(self, env_config):
        super().__init__()
        
        # Observation: Raycasts(20), RelPos(3), Health(2), LoS(1) = 26 dims
        self.observation_space = Box(low=-1.0, high=1.0, shape=(26,), dtype=np.float32)
        
        # Action: MoveX, MoveY, AimX, AimY (Continuous) + Shoot (Discrete)
        # RLlib handles hybrid spaces using Dict or Tuple spaces
        self.action_space = Dict({
            "movement": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "shoot": Discrete(2)
        })
        
        # In reality, this connects via gRPC to headless Unreal Engine
        self.game_client = None 
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        # self.game_client.reset_match()
        mock_obs = np.zeros(26, dtype=np.float32)
        return mock_obs, {}

    def step(self, action):
        self.steps += 1
        
        # Send action to game engine (MoveX, Shoot, etc)
        # obs, raw_reward, done = self.game_client.step(action)
        
        # MOCKING RESPONSE
        obs = np.random.uniform(-1, 1, 26).astype(np.float32)
        reward = -0.01 # Time penalty
        done = False
        
        # Complex Reward Shaping (Mocked logic)
        # if obs["line_of_sight"]: reward += 0.1
        # if self.game_client.agent_hit_player(): reward += 1.0
        # if self.game_client.agent_died(): reward -= 10.0; done = True
        
        if self.steps > 1000: # Max timeout
            done = True
            
        return obs, reward, done, False, {}

# ---------------------------------------------------------------------------
# 2. Distributed Training Configuration
# ---------------------------------------------------------------------------
def train_agent():
    logger.info("Configuring PPO Algorithm...")
    
    config = (
        PPOConfig()
        .environment(ShooterNPCEnv)
        .env_runners(num_env_runners=4) # Number of parallel game engines
        .resources(num_gpus=1) # GPU for Neural Net updates
        .training(
            gamma=0.99,          # Discount factor
            lr=3e-4,             # Learning rate
            clip_param=0.2,      # PPO clipping
            train_batch_size=4000,
            model={
                "fcnet_hiddens": [256, 256], # Actor-Critic network size
                "fcnet_activation": "relu",
            }
        )
    )
    
    algo = config.build()
    
    logger.info("Starting Training Loop...")
    for i in range(10): # 10 iterations for demo (normally 10,000+)
        result = algo.train()
        print(f"Iteration {i}: Mean Reward = {result['env_runners']['episode_reward_mean']:.2f}")
        
        # Save checkpoint periodically
        if i % 5 == 0:
            checkpoint = algo.save()
            print(f"Checkpoint saved at {checkpoint}")

if __name__ == "__main__":
    import ray
    ray.init(ignore_reinit_error=True)
    train_agent()
```

---

## Part 8 — Deployment

### Training Infrastructure
- Run on a Kubernetes cluster. 1 GPU Node for the PPO Trainer. 50 CPU Nodes running headless game engine instances (Rollout Workers) to gather experience as fast as possible.

### Production Inference
- Export the trained PyTorch/TensorFlow network to ONNX.
- Embed ONNX Runtime in the game engine (C++).
- During gameplay, the engine feeds the 26-dim vector to the ONNX model, receives the actions, and applies them to the NPC. Inference takes `< 1ms`.

---

## Part 9 — Unit Testing

```python
import numpy as np
from npc_rl_training import ShooterNPCEnv

def test_env_observation_bounds():
    env = ShooterNPCEnv(config={})
    obs, _ = env.reset()
    
    assert obs.shape == (26,)
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0), "Obs out of bounds"
    
def test_env_action_handling():
    env = ShooterNPCEnv(config={})
    
    # Send a valid action
    action = {
        "movement": np.array([1.0, 0.0, 0.5, -0.5], dtype=np.float32),
        "shoot": 1
    }
    
    obs, reward, done, _, _ = env.step(action)
    
    # Assert time penalty was applied
    assert reward == -0.01
```

---

## Part 10 — Integration Testing

- **Self-Play Baseline:**
  - Before deploying against humans, test the agent against a hardcoded Behavior Tree bot.
  - If the RL agent cannot consistently beat the Behavior Tree bot after 5 million steps of training, the reward function is flawed.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Sample Inefficiency** | RL requires millions of frames. A real-time game runs at 60 FPS (too slow). We must run the game engine completely headless, disabling rendering and audio, and stepping the physics simulation forward at 1000x real-time speed. |
| **Exploration vs Exploitation** | To prevent the agent from finding a local optima (e.g., just spinning in circles to avoid dying), use Entropy Regularization in PPO to force the agent to try random actions early in training. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| RL vs Behavior Trees | Behavior trees are 100% predictable, easy to debug, and designers can manually tweak them ("Make him shoot 10% less"). RL agents are a black box. If an RL agent acts stupid, you cannot manually fix it—you have to tweak the reward function and retrain for hours. |
| Vector vs Pixel Observations | Pixels (Vision) allow the agent to play the exact same game the player plays. However, training CNNs on pixels requires 100x more compute and takes weeks. Vectors (extracting state directly from RAM) train in hours but rely on the engine providing perfect raycasts. |
| PPO vs SAC (Soft Actor-Critic) | SAC is more sample-efficient than PPO (it learns faster), but PPO is vastly more stable and easier to tune. For complex 3D games, PPO is preferred to prevent catastrophic forgetting. |

---

## Part 13 — Alternative Approaches

1. **Imitation Learning (Behavioral Cloning):** Don't use RL at all. Record 1,000 hours of professional players playing the game (State -> Action pairs). Train a Supervised Learning model to mimic them. Much faster than RL, but the agent will never surpass human skill level.
2. **Curriculum Learning:** Don't start the agent in a full 1v1 map. Start it in a tiny empty room where the enemy is stationary. Once it learns to shoot, add walls. Once it navigates walls, make the enemy move. Increases training speed drastically.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Reward Hacking | You reward the agent for "bullets hitting the player". The agent learns to equip a low-damage SMG and shoot the player in the foot indefinitely to rack up infinite points without killing them. | Do not reward intermediate states. Or, cap the maximum reward per episode. Reward the *ultimate goal* (Winning the match). |
| Catastrophic Forgetting | The agent learns to aim. Then it learns to navigate. In the process of learning navigation, it completely forgets how to aim. | Use a massive Replay Buffer (if using off-policy algorithms) or lower the PPO learning rate. Ensure the environment randomizes spawn points so the agent faces diverse states constantly. |

---

## Part 15 — Debugging

**Symptom:** The agent trains for 10 million steps. In production, it immediately runs into a corner and stares at the wall, refusing to shoot.

**Debugging steps:**
1. Check the Reward Function. Is it getting penalized -1 for missing a shot? It learned that shooting = negative reward, so it prefers to do nothing.
2. Check the Observation Space. Is the production vector matching the training vector? (e.g., During training, raycasts ignored transparent glass. In production, the raycast hits the glass, so the agent thinks it's blocked).
3. **Fix:** Normalize the environment. Ensure the training physics engine is a 1:1 replica of the production engine. Remove penalties for missing shots—allow it to explore shooting freely.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `episode_reward_mean` | Should steadily increase. If it plateaus early, adjust Entropy/Learning Rate. |
| `value_function_loss` | High variance indicates the Critic network is failing to predict the future accurately. |
| `kl_divergence` | PPO specific: if KL divergence spikes, the policy updated too aggressively. Lower `clip_param`. |

---

## Part 17 — Production Improvements

1. **Difficulty Scaling:** A perfect RL bot is unfun to play against (it will instantly headshot the player). We need a "Difficulty Slider". We can train multiple agents (Easy, Medium, Hard) by artificially restricting the Easy agent's reaction time or adding Gaussian noise to its aiming actions.
2. **Action Masking:** Prevent the RL model from outputting invalid actions (e.g., trying to 'Reload' when the gun is full). Apply a binary mask to the output logits before the Softmax layer, forcing the probability of invalid actions to zero. Speeds up training immensely.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"Our game has 10 different maps. If we train the agent on Map A, it acts like a genius. If we drop it into Map B, it acts like a toddler. How do we make it generalize?"**
2. **"To make the agent more 'human-like', you decide to use Imitation Learning combined with RL. Describe the architecture combining these two."**
3. **"The agent keeps getting stuck oscillating between two states (e.g., walking left for 1 frame, then right for 1 frame) because both paths seem equally good to the Value network. How do you break this oscillation?"**

---

## Part 19 — Ideal Answers

**Q1 (Generalization / Overfitting):**
> "The agent overfit to the exact geometry of Map A. We must use **Domain Randomization**. During training, we randomly shuffle the map geometry, wall colors, player spawn points, and lighting every episode. The agent learns the *concept* of navigation rather than memorizing the pathing of Map A. Alternatively, we use a recurrent network (LSTM/GRU) so the agent can build a persistent internal map of a new area over time."

**Q2 (Imitation + RL):**
> "We use **Generative Adversarial Imitation Learning (GAIL)** or **RL from Human Feedback (RLHF)**. We first pre-train the actor network using standard Behavioral Cloning on human telemetry (Supervised Learning). Then, we use PPO to fine-tune it. To keep it 'human-like', we add a penalty to the PPO reward function (KL Divergence) if the agent's actions deviate too far from the human baseline policy."

**Q3 (Oscillation / Frame Flickering):**
> "This is a lack of temporal persistence. 
> 1. We can implement **Action Smoothing** (Action $A_t = 0.8 * A_{t-1} + 0.2 * ModelOutput$).
> 2. We can use **Frame Skipping** (Agent only decides a new action every 4 frames, forcing it to commit to a direction). 
> 3. Add an LSTM memory cell so the agent remembers 'I just moved left' and biases its next decision to continue the intended trajectory."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Understands PPO and Actor-Critic architecture.
- Anticipates the dangers of Reward Hacking (the agent cheating the reward function).
- Solves the frame oscillation problem natively (Frame Skipping, LSTMs).
- Proposes Domain Randomization for generalization.

### Hire
- Sets up a custom Gym environment properly.
- Understands continuous vs discrete action spaces.
- Identifies Ray RLlib as the scaling tool.
- Provides a solid, dense reward function.

### Lean Hire
- Tries to train the agent on Pixels (CNNs) without recognizing the massive compute/time penalty for a simple shooter NPC.
- Struggles with defining a robust reward function (Needs interviewer help to avoid exploits).

### Lean No Hire
- Suggests standard Supervised Learning (predicting the next move based on a hardcoded script).
- Does not understand what an Observation Space is.

### No Hire
- Does not know what Reinforcement Learning is.
- Cannot explain the difference between a State, Action, and Reward.
