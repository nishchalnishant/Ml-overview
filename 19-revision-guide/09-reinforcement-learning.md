---
module: Reinforcement Learning
topic: Revision Guide
subtopic: ""
status: unread
tags: [rl, revision, outline, qa]
prerequisites: [09-reinforcement-learning]
---
# 09 Reinforcement Learning — Revision Guide

Fast revision for the whole `09-reinforcement-learning/` folder. The outline maps every major topic and subtopic as a hierarchy; the Q&A section drills recall and reasoning across three difficulty tiers. Use the outline to locate a gap, then hit the matching questions.

---

## Structure Overview

```markdown
- **09 Reinforcement Learning**
  - **RL Foundations**
    - MDP tuple (S,A,P,R,γ)
    - Markov property
      - POMDP / frame stacking
    - Return G_t
      - Discount γ: convergence/preference/horizon
    - Policy π(a|s)
    - Value functions
      - V(s)
      - Q(s,a)
      - Advantage A=Q-V
    - Bellman equations
      - Bellman expectation
      - Bellman optimality (max)
    - Credit assignment problem
    - Reward hacking
    - Sample efficiency / offline RL intro
  - **Bandits & Exploration**
    - Exploration-exploitation tradeoff
    - Cumulative regret
      - Linear regret
      - Logarithmic regret (Lai-Robbins)
    - ε-greedy
    - UCB (optimism)
      - Confidence bonus
    - Thompson Sampling
      - Posterior sampling
    - Contextual bandits
      - LinUCB
    - Non-stationarity
    - Bandit vs A/B test
  - **Value-Based Methods**
    - TD learning
      - TD error δ
      - TD vs Monte Carlo
    - Q-learning (off-policy)
    - SARSA (on-policy)
      - Cliff-walking example
    - DQN
      - Experience replay
      - Target network
      - Reward clipping
    - Double DQN
    - Dueling DQN
    - Prioritized replay
    - Rainbow
    - Overestimation bias
    - Deadly triad
      - Function approximation
      - Bootstrapping
      - Off-policy
  - **Policy Gradient Methods**
    - Policy gradient theorem
      - Log-derivative trick
    - REINFORCE
      - High variance problem
    - Baselines
      - Unbiasedness proof
    - Actor-critic
      - Actor network
      - Critic network
    - GAE (λ)
    - TRPO (KL constraint)
    - PPO
      - Probability ratio r_t
      - Clipped surrogate
    - GRPO
      - Group relative advantage
    - DDPG
    - TD3
    - SAC (max entropy)
    - RLHF
      - Reward model
      - KL penalty vs reference
      - Reward over-optimization
    - DPO
  - **Advanced & Deep RL**
    - Offline RL
      - Distributional shift
      - CQL
      - IQL
      - Decision Transformer
    - Model-based RL
      - Dyna
      - World models / Dreamer
      - MuZero
      - Model exploitation
    - Planning / MCTS
      - Selection (UCT/PUCT)
      - Expansion
      - Evaluation
      - Backup
      - AlphaGo
      - AlphaZero
    - Imitation learning
      - Behavior cloning
        - Covariate shift
      - DAgger
      - Inverse RL / GAIL
  - **01 Foundations**
  - **02 Bandits**
  - **03 Value-Based**
  - **04 Policy Gradient**
  - **05 Advanced/Deep RL**
```

---

## Questions

### Easy

1. What is an MDP, and what does each element of the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ represent?
2. State the Markov property in one sentence.
3. What is the return $G_t$, and why do we discount future rewards?
4. What happens at $\gamma = 0$ versus $\gamma = 1$?
5. Define $V^\pi(s)$ and $Q^\pi(s,a)$. How do they differ?
6. What is the advantage function, in words?
7. What is the credit assignment problem?
8. What is reward hacking? Give a simple example.
9. What is the exploration-exploitation trade-off?
10. Describe the ε-greedy algorithm in one line. What's its main weakness?
11. What is cumulative regret, and why is linear regret considered a failure?
12. What is a contextual bandit, and how does it differ from a plain multi-armed bandit?
13. What is temporal-difference (TD) learning, in one sentence?
14. What is the TD error?
15. State the Q-learning update rule.
16. Is Q-learning on-policy or off-policy? Why?
17. What three problems does DQN's experience replay solve?
18. Why does DQN need a target network?
19. What is REINFORCE, at a high level?
20. Why is REINFORCE considered high-variance?
21. What does a baseline do in a policy gradient update?
22. What are the "actor" and "critic" in actor-critic methods?
23. What does PPO stand for, and what problem is it solving?
24. What is RLHF, in one sentence, and which algorithm typically powers it?
25. What is behavior cloning?

### Medium

1. Why is the Markov property a property of the *state representation* rather than the environment itself? Give an example of fixing a non-Markovian state.
2. Walk through why $Q$ lets you act without a model of the environment, but $V$ alone does not.
3. Why does $\gamma$ change which policy is optimal, not just how fast training converges? Reference the three-state chain example.
4. Why is UCB's exploration called "directed" while ε-greedy's is "undirected"?
5. Explain why the UCB bonus term shrinks with $n_i$ and grows (slowly) with $t$. What would break if it didn't do both?
6. Compare Thompson Sampling and UCB on delayed/batched feedback. Which degrades and why?
7. When would you choose a bandit over a classical A/B test, and what statistical guarantee do you give up by doing so?
8. Explain the bias-variance trade-off between Monte Carlo returns and TD targets.
9. Contrast Q-learning and SARSA using the cliff-walking example. Why do they learn different paths?
10. Why does DQN's target network need to be *frozen*, and what specifically goes wrong if it isn't?
11. Explain overestimation bias in Q-learning: why does $\max_a \hat Q(s,a)$ tend to overestimate the true max?
12. How does Double DQN fix overestimation bias mechanically (selection vs. evaluation)?
13. Derive intuitively why subtracting a baseline $b(s)$ from the policy gradient does not introduce bias.
14. What does GAE's $\lambda$ parameter trade off, and what do $\lambda=0$ and $\lambda=1$ each reduce to?
15. Explain PPO's clipped surrogate objective and why the `min` (not just the clip) is essential.
16. How does GRPO remove the need for a critic, and what does it cost you in exchange?
17. Why is SAC's entropy bonus described as making exploration "intrinsic" rather than "bolt-on"?
18. Explain why naive off-policy Q-learning fails on a fixed offline dataset (distributional shift / extrapolation error).
19. How do CQL and IQL each address offline RL's out-of-distribution action problem, and how do their approaches differ?
20. What does MCTS add on top of a trained policy network, and why does repeated self-play actually improve the network in AlphaZero?
21. Why does behavior cloning suffer from compounding error / covariate shift, and how does DAgger address it?
22. Explain why "RLHF is PPO on a language model" — map vocabulary, reward model, and KL penalty onto the standard PPO pieces.
23. A bandit converges to one arm quickly and CTR declines a month later. What's the likely mechanism?
24. Why might a lower discount factor sometimes produce *better* practical performance despite being a worse model of the true objective?
25. When does a problem cross the line from "contextual bandit" to "full RL"?

### Hard

1. Derive the policy gradient theorem's key step (the log-derivative trick) and explain why no derivative of the environment's transition dynamics appears in the final expression.
2. Prove that adding a state-dependent baseline $b(s)$ to the REINFORCE gradient leaves the expected gradient unchanged.
3. Explain the "deadly triad" (function approximation, bootstrapping, off-policy learning) mechanistically — why do all three together risk divergence when any two alone are safe?
4. Your DQN's Q-values grow to $10^6$ while episode reward stays flat. Walk through your full diagnostic and fix sequence.
5. Why can supervised learning on logged data from a deployed policy never exceed that policy, and how does off-policy evaluation attempt to get around this? Where does it break down?
6. Explain why PPO's clip alone (without the `min` against the unclipped objective) can strand the policy with zero gradient in the wrong regime.
7. During RLHF, reward-model score climbs steadily but human evaluators rate the model worse over time. Diagnose the mechanism and lay out your fix priority.
8. Derive why behavior cloning's error compounds *quadratically* in horizon length while DAgger's compounds linearly. What is the underlying independence assumption that fails?
9. Explain MuZero's core departure from AlphaZero: what does it mean to plan in a "value-equivalent" latent space, and why does it avoid needing the environment's rules?
10. Walk through why CQL's regularizer and IQL's expectile regression both avoid querying $Q$ at out-of-distribution actions, but via structurally different mechanisms.
11. Given a three-state MDP, show how changing $\gamma$ from 0.9 to 0.5 flips the optimal policy. What does this imply about treating $\gamma$ as "just a hyperparameter"?
12. You have logged clinical data from a rule-based treatment policy and cannot experiment on patients. Design the offline RL + evaluation pipeline, including how you'd guard against the failure modes you'd expect.
13. Explain why Thompson Sampling would *not* automatically fix a non-stationary bandit problem, and what modification would.
14. A game studio wants to use RL for difficulty balancing. Construct the full interrogation of whether RL is even the right tool, including reward design failure modes.
15. Explain the mechanism by which Dueling DQN separates $V(s)$ and $A(s,a)$, and why this separation is more capacity-efficient than a single $Q$ stream for most states.
16. Contrast the variance-reduction mechanism of the advantage function in actor-critic with the group-relative advantage in GRPO — what statistical assumption does each lean on?
17. What breaks in Decision-Transformer-style return-conditioned sequence modeling if you condition on a return-to-go far outside the training distribution?
18. Explain why model-based RL's compounding rollout error makes long-horizon planning "fiction" beyond some point, and how ensembles / short rollouts mitigate it without solving it.
19. Derive why $O(\log T)$ regret is the theoretical floor for stochastic bandits (Lai-Robbins), and explain intuitively why $\varepsilon$-greedy with fixed $\varepsilon$ cannot achieve it.
20. In PUCT (used in MCTS/AlphaZero), explain precisely what the exploration term is balancing against the exploitation term, and why "most-visited action" is preferred over "highest-value action" at the root.
21. Explain the exact chain of causation by which off-policy data makes the deadly triad's divergence worse than pure function-approximation-plus-bootstrapping alone.
22. Why does GAIL's discriminator-based approach to imitation generalize better to unseen states than plain behavior cloning, despite never explicitly recovering a reward function?
23. Construct a scenario where SARSA's on-policy conservatism produces a worse *deployed* policy than Q-learning's off-policy optimism, and one where the reverse is true.
