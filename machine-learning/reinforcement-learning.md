# Reinforcement Learning

**Reinforcement learning (RL)** is learning by **trial and error**: an agent takes **actions** in an **environment**, receives **rewards** (or penalties), and improves its **policy** to maximize cumulative reward over time.

---

## Key concepts

- **Agent**: The learner/decision maker. **Environment**: The world the agent interacts with (e.g. game, robot, simulator).
- **State (s)**: Observation of the environment at a time step. **Action (a)**: Choice the agent makes. **Reward (r)**: Scalar feedback (e.g. +1 for goal, -1 for failure).
- **Policy π(a|s)**: Strategy that maps states to actions (or action distributions). **Goal**: Find a policy that maximizes expected **return** (sum of discounted rewards).

---

## MDP (Markov Decision Process)

- **MDP**: (S, A, P, R, γ) — states, actions, transition dynamics P(s'|s,a), reward R(s,a), discount γ.
- **Markov**: Next state and reward depend only on current state and action, not full history.
- **Value function**: V(s) = expected return from state s under policy π. **Q-function**: Q(s,a) = expected return from taking a in s then following π.

---

## Main approaches

| Approach | Idea | Example |
|----------|------|--------|
| **Value-based** | Learn V(s) or Q(s,a); act greedily w.r.t. Q | DQN (deep Q-network) |
| **Policy-based** | Learn π(a|s) directly; optimize expected return | Policy gradient, REINFORCE |
| **Actor-critic** | Learn both policy (actor) and value (critic) | A2C, A3C, PPO |
| **Model-based** | Learn P, R; plan (e.g. tree search) in the model | Dyna-Q, MuZero |

---

## Connection to LLMs: RLHF

- **RLHF** uses RL to align an LLM with human preferences: the **policy** is the LLM; the **reward** comes from a model trained on human preferences; the “environment” is the space of prompts and completions.
- **PPO** is often used to update the LLM so it maximizes the reward while staying close to the initial (SFT) policy (KL penalty).

---

## Quick revision

- **RL** = agent, environment, states, actions, rewards; learn a policy to maximize return.
- **MDP** = formal framework; value and Q-functions. **Value-based** (DQN), **policy-based** (policy gradient), **actor-critic** (PPO).
- **RLHF** applies RL to align LLMs using a human-preference reward model.
