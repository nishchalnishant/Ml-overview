# Interview 22 — Deep RL for NPC AI (Condensed)

Design + train a Deep RL agent to control a 1v1 shooter NPC (navigate/aim/shoot), replacing predictable Behavior-Tree enemies. Core challenge is reward/observation design + making millions of training steps fast, not the algorithm itself.

## Core Architecture
- Headless game engine instances (Frostbite/Unreal) run 100s in parallel, no rendering, physics stepped at 1000x speed.
- gRPC/shared-memory bridge sends observations out, actions in — same pattern as Unity ML-Agents.
- Ray RLlib rollout workers collect (state, action, reward) trajectories across workers.
- PPO Actor-Critic trainer updates policy — industry standard for continuous control; balances stability + sample efficiency.
- Vector observation (~26-50 dims): relative player position, line-of-sight, health, 360° raycasts for obstacle avoidance.
- Hybrid action space: continuous move/aim + discrete shoot.
- Trained policy exported to ONNX, embedded in-engine (C++) for <1ms inference at runtime.

## Talking Points That Signal Seniority
- Proactively flags reward hacking risk before being asked — e.g., rewarding "hits" lets the agent farm low-damage shots forever instead of killing.
- Names Domain Randomization (randomize map geometry/spawns/lighting each episode) as the fix for map-to-map generalization, unprompted.
- Raises Entropy Regularization as needed to avoid local optima (agent spinning/hiding to farm survival reward).
- Notes a "too-good" RL bot is bad UX — proposes a difficulty slider via noise injection or reaction-time throttling on cheaper agents.
- Suggests Action Masking to zero out invalid-action logits, speeding up training and avoiding wasted exploration.
- Recognizes frame-to-frame oscillation as a temporal-persistence problem and reaches for frame skipping / action smoothing / LSTM memory instead of just "more training."
- Insists training environment must be a bit-for-bit physics replica of production (calls out train/prod skew, e.g. raycasts hitting glass in prod but not training).
- Distinguishes sparse terminal reward (win/lose) from dense shaping reward and explains why pure sparse reward fails to train in reasonable time.

## Top 3 Tradeoffs
- RL vs Behavior Trees: BTs are debuggable and designer-tunable; RL is a black box you can only "fix" by reshaping reward and retraining for hours.
- Vector vs Pixel observations: pixels let the agent see what a human sees but need ~100x compute and weeks to train; vectors train in hours but depend on the engine exposing accurate state.
- PPO vs SAC: SAC is more sample-efficient, but PPO is far more stable/easier to tune — matters because instability in a 3D game causes catastrophic forgetting.

## Biggest Pitfall
Treating reward design as an afterthought — a candidate who can't anticipate reward hacking (or needs heavy interviewer prompting to avoid it) reads as someone who will ship an agent that "wins" by exploiting the reward function rather than playing the game.
