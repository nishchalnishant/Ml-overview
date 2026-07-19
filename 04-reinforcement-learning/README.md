---
module: Reinforcement Learning
topic: Overview
subtopic: ""
status: unread
tags: [rl, overview, index]
prerequisites: [probability, gradient-descent, neural-networks]
---
# Reinforcement Learning

Learning to act from evaluative feedback, rather than from labeled examples. This section exists because RL shows up in interviews through three distinct doors, and most candidates only prepare for one:

1. **RLHF** — the alignment stage of every major LLM is PPO or a descendant. Asked in nearly every LLM-focused loop.
2. **Bandits** — the practical workhorse. Ranking, recommendations, ad selection, and A/B testing all reduce to exploration-vs-exploitation, and interviewers use it to test whether you reach for the *simplest* sufficient tool.
3. **Full RL** — games, robotics, control. Less common, but where the deepest questions live.

The most frequent mistake in RL interviews is reaching for full RL when a bandit would do. A large share of "RL problems" in industry have no meaningful state transitions — a contextual bandit solves them with a fraction of the complexity and sample cost.

---

## Files in This Folder

| File | What it covers |
| :--- | :--- |
| [01-rl-foundations.md](01-rl-foundations.md) | MDPs, the Markov property, return and discounting, V/Q/advantage, Bellman equations |
| [02-bandits-and-exploration.md](02-bandits-and-exploration.md) | Regret, ε-greedy, UCB, Thompson sampling, contextual bandits, non-stationarity |
| [03-value-based-methods.md](03-value-based-methods.md) | TD learning, Q-learning vs SARSA, DQN and its fixes, the deadly triad |
| [04-policy-gradient-methods.md](04-policy-gradient-methods.md) | REINFORCE, baselines and advantage, actor-critic, TRPO/PPO, SAC, RLHF |

**Reading order is sequential.** File 01 defines the vocabulary the rest assume. Files 03 and 04 are the two competing answers to the same question — how do you actually optimize a policy — and are best read as a pair.

If you are short on time and preparing for an LLM role: read 01, skim 02, and read 04 closely. The RLHF material is in 04.

---

## The Core Split

| | Value-based (file 03) | Policy gradient (file 04) |
| :--- | :--- | :--- |
| Learns | $Q(s,a)$, policy is implicit | $\pi_\theta(a \mid s)$ directly |
| Action space | Discrete only | Discrete or continuous |
| Policy type | Deterministic (greedy) | Stochastic by construction |
| Sample efficiency | Higher (off-policy replay) | Lower (on-policy discards data) |
| Stability risk | Value divergence (deadly triad) | Policy collapse from a large update |
| Representative | DQN, Rainbow | PPO, SAC |

---

## What Interviewers Actually Probe

- **Can you tell a bandit problem from an RL problem?** The test is whether actions change future state. If they don't, you don't need RL.
- **Do you understand exploration as a cost, not a setting?** Every exploratory action is a deliberately suboptimal one. Justifying that spend is the real question.
- **Can you name a failure mode without prompting?** Reward hacking, the deadly triad, and reward over-optimization in RLHF are the three that come up most.
- **Do you know why RL is a last resort in production?** Sample inefficiency, unstable training, and hard-to-specify rewards. Supervised learning on logged data is almost always tried first, and often wins.

---

## Connections

- **Bandits and experimentation:** [../06-production-ml/system-design/14-ab-testing-experimentation.md](../06-production-ml/system-design/14-ab-testing-experimentation.md)
- **RLHF and alignment:** [../05-llms/interview-notes/17-advanced-alignment-and-reasoning.md](../05-llms/interview-notes/17-advanced-alignment-and-reasoning.md)
- **Training pipeline context:** [../05-llms/01-training-process.md](../05-llms/01-training-process.md)
