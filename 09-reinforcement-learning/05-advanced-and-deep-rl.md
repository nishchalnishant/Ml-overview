---
module: Reinforcement Learning
topic: Advanced and Deep RL
subtopic: ""
status: unread
tags: [rl, offline-rl, model-based, muzero, alphazero, mcts, imitation-learning, cql, iql, decision-transformer]
prerequisites: [rl-foundations, value-based-methods, policy-gradient-methods]
---
# Advanced and Deep RL — Offline, Model-Based, Planning, and Imitation

> **Related:** [Policy Gradient Methods](04-policy-gradient-methods.md) and [Value-Based Methods](03-value-based-methods.md) cover the *online, model-free* core (PPO, SAC, DQN). This file covers the four directions interviews probe *beyond* that core: learning from a fixed dataset (offline RL), learning a model of the world (model-based RL), planning with search (MCTS / AlphaZero / MuZero), and skipping reward entirely (imitation learning).

---

## The Problem It Solves

The model-free online loop — collect fresh trajectories, update, repeat — has three expensive assumptions, and each advanced family drops one.

**You can interact freely.** Online RL needs millions of fresh environment steps. In healthcare, autonomous driving, or recommendation, exploration is dangerous or forbidden — you have *logged data* and cannot collect more on-policy. **Offline RL** learns a policy from a fixed dataset with no new interaction.

**Interaction is cheap.** Even when allowed, environment steps can dominate cost (a physics sim, a real robot). If you instead *learn a model* $\hat P(s'\mid s,a)$, you can generate unlimited synthetic experience and plan inside it. **Model-based RL** trades sample efficiency for the burden of learning an accurate model.

**A reactive policy is enough.** A feedforward policy picks the locally-good action. But in games with deep tactics (Go, chess), you want to *look ahead* — simulate consequences before committing. **Planning / MCTS** spends compute at decision time to act far better than the raw policy alone.

**You even have a reward function.** Sometimes you don't — but you have expert demonstrations. **Imitation learning** recovers a policy (or a reward) from demonstrations without a hand-specified reward at all.

---

## Intuition

- **Offline RL** is *learning to cook from a stack of other people's recipe outcomes without ever entering the kitchen.* The danger is you'll invent a dish no one tried and be wildly overconfident it's delicious — the whole field is about staying humble on actions absent from the data.
- **Model-based RL** is *building a mental simulator so you can rehearse in your head.* Cheap if your simulator is accurate; catastrophic if you trust a flawed one and rehearse fantasies.
- **MCTS** is *thinking ahead a few moves, spending more thought on the promising lines.* The policy is intuition; search is deliberation. AlphaZero is intuition and deliberation improving each other.
- **Imitation learning** is *learning to drive by watching an expert* — trivial until you drift somewhere the expert never went and have no idea how to recover.

---

## The Mechanics

### Offline RL — and the one problem that defines it

Train on a fixed dataset $\mathcal D = \{(s,a,r,s')\}$ collected by some **behavior policy** $\pi_\beta$. Naively running Q-learning fails, and the reason is the single most important idea in offline RL:

**Distributional shift / extrapolation error.** The Bellman target $r + \gamma \max_{a'} Q(s',a')$ takes a max over actions. For actions *absent from the data*, $Q$ is unconstrained garbage — and the max *systematically selects the most overestimated* of those garbage values. The policy then loves out-of-distribution actions precisely because the critic is delusional about them, and there's no environment feedback to correct it. Error compounds. This is why online tricks don't transfer.

The fix is always the same shape: **stay close to the data / be pessimistic about unseen actions.**

- **CQL (Conservative Q-Learning)** adds a regularizer that *pushes down* Q-values on out-of-distribution actions and pushes *up* on in-data actions — learning a conservative lower bound so the max can't chase phantoms.
- **IQL (Implicit Q-Learning)** never queries $Q$ at unseen actions *at all*. It uses expectile regression to approximate the max over *in-support* actions only, then extracts the policy by advantage-weighted regression. No OOD action ever enters a target.
- **Decision Transformer** sidesteps RL entirely: treat the trajectory as a sequence `(return-to-go, state, action, …)` and train a Transformer to predict the next action *conditioned on a desired return*. At test time you condition on a high return-to-go. It's sequence modeling, not value iteration — no bootstrapping, hence no extrapolation blowup.

### Model-based RL

Learn dynamics $\hat P(s'\mid s,a)$ and reward $\hat r$, then use them.

- **Dyna** interleaves real and imagined experience: update from real steps *and* from steps sampled out of the learned model. Cheap way to squeeze more updates from few real interactions.
- **World models / Dreamer** learn dynamics in a compact *latent* space and train the policy purely on latent rollouts ("dreams"), only touching the real environment to collect data. State-of-the-art sample efficiency.
- **MuZero** learns a model in a *value-equivalent* latent space — it never reconstructs the actual next state, only quantities needed for planning (policy, value, reward). It learns to predict *what matters for decisions*, not the pixels.

The universal risk: **model exploitation / compounding error.** The policy finds and exploits the model's inaccuracies (a bug that "gives" reward), and one-step errors compound over a rollout. Mitigations: short rollouts, model ensembles for uncertainty, penalizing reward in high-uncertainty regions.

### Planning: MCTS → AlphaGo → AlphaZero → MuZero

**MCTS** builds a search tree over four repeated phases:
1. **Selection** — descend the tree by **UCT/PUCT**, balancing exploitation ($Q$) against exploration ($\propto P(a)\sqrt{N_{\text{parent}}}/(1+N)$).
2. **Expansion** — add a leaf node.
3. **Evaluation** — estimate the leaf's value (rollout, or a value network).
4. **Backup** — propagate the value up the visited path, updating $Q$ and visit counts $N$.

Act by the *most-visited* child, then reuse the subtree.

The lineage:
- **AlphaGo** — supervised policy from human games + RL fine-tuning + a value net, all guiding MCTS. Beat Lee Sedol.
- **AlphaZero** — drops human data. Self-play only, one network outputs $(P, V)$. MCTS is a *policy-improvement operator*: search produces a better policy than the raw net, the net is trained toward the search result, and the loop bootstraps. General enough for Go, chess, shogi.
- **MuZero** — AlphaZero *without being given the rules*. It learns the model (see above), so it plans in games/environments where the transition dynamics are unknown. Same self-play + MCTS improvement loop, applied to a learned model.

### Imitation learning

- **Behavior cloning (BC)** — supervised learning: map expert states → expert actions. Simple, no environment needed. Its fatal flaw is **compounding error / covariate shift**: a small mistake takes you to a state the expert never visited, where your errors are larger, so you drift further — errors grow *quadratically* in the horizon, not linearly.
- **DAgger (Dataset Aggregation)** — fixes BC by iterating: run your policy, have the *expert label the states you actually visited*, aggregate, retrain. Now training data covers your own error states. Requires an interactive expert.
- **Inverse RL (IRL)** — don't clone actions; **infer the reward function** the expert seems to optimize, then do RL on it. Generalizes better (a reward transfers across dynamics) but is expensive and underdetermined. **GAIL** is the practical modern form: a GAN-style setup where a discriminator learns to tell agent from expert trajectories and the policy is trained to fool it — imitation without ever explicitly recovering the reward.

---

## Worked Example

**Setting:** you have 6 months of logged data from a *rule-based* treatment-recommendation system in a hospital. You may not experiment on patients. Learn a better policy.

This is offline RL, and the trap is immediate. Run vanilla Q-learning and the critic will discover that some rarely-tried treatment has a spuriously high $Q$ (few or zero examples → unconstrained estimate → the max selects it). The learned policy confidently recommends it. There is no way to catch the error from data alone — the whole point is you can't try it.

Use **CQL or IQL**: the policy is regularized to stay within the support of what clinicians actually did, improving *within* the logged action distribution rather than inventing untested treatments. Evaluate with **off-policy evaluation** (weighted importance sampling / doubly-robust estimators) — never with the online return you can't measure. Report the estimate *with* its confidence interval, because OPE variance explodes when the learned policy diverges from $\pi_\beta$. The honest deliverable is "a policy that is provably close to behavior but reweights toward better-outcome actions, with this much estimated lift and this much uncertainty" — not "the optimal policy."

---

## When It Breaks

- **Offline RL, policy far from behavior.** The further the learned policy strays from $\pi_\beta$, the less trustworthy both the policy *and its evaluation*. If your data is narrow (one mediocre behavior policy), offline RL cannot conjure a great policy — no algorithm escapes the support of the data.
- **Model-based, long horizons.** Compounding model error makes long rollouts fiction. Beyond a short horizon the imagined trajectory is uncorrelated with reality.
- **MCTS, huge branching / continuous actions.** Tree search assumes a manageable, discrete action set and a cheap simulator. Continuous control needs progressive widening or a different approach; an expensive simulator makes the thousands of rollouts per move infeasible.
- **Behavior cloning, distribution shift.** No recovery behavior in the data → the classic drift-off-the-road failure. If you can't get an interactive expert (no DAgger), BC is fragile exactly where it matters.
- **Reward-free methods inherit demonstration bias.** Imitation is capped by the expert; it cannot exceed demonstrated behavior, and it faithfully copies the expert's mistakes and biases.

---

## Production Notes

- **When offline RL is worth it:** you have a large, *diverse* logged dataset and online exploration is costly or unsafe. If your logs come from one narrow policy, temper expectations. IQL is the common default (stable, simple, no OOD queries); CQL when you want an explicit conservative value.
- **Off-policy evaluation is not optional.** You cannot A/B a policy you're afraid to deploy. Budget as much rigor for the *evaluator* as the *learner*; a confident policy with a useless OPE estimate is undeployable.
- **Model-based earns its keep when environment steps dominate cost** (real robots, expensive sims). Dreamer-style latent world models are the current sample-efficiency frontier.
- **AlphaZero-style search is for domains with a known (or learnable) simulator and discrete actions** — board games, some scheduling/combinatorial problems, theorem proving. Not a drop-in for continuous robotics.
- **Most real "imitation" is behavior cloning**, because it's just supervised learning and needs no environment. Reach for DAgger/GAIL only when covariate shift actually bites. LLM **SFT is behavior cloning** on demonstrations; RLHF is what's layered on top to go beyond imitation.

---

## Interview Angles

### Q: Why can't you just run DQN or SAC on a fixed offline dataset? [Medium]

Because the Bellman target's $\max_{a'} Q(s',a')$ queries actions absent from the data, where $Q$ is an unconstrained overestimate, and the max *selects* the worst overestimate. The policy then prefers out-of-distribution actions and there's no environment feedback to correct the delusion — error compounds through bootstrapping. Online, exploration would eventually visit those actions and fix the estimate; offline, it never can.

**Cross-questions to expect:**
- *"How do CQL and IQL each address it?"* → CQL adds a regularizer that lower-bounds Q on OOD actions (pushes them down, in-data up). IQL never evaluates $Q$ at OOD actions at all — expectile regression approximates the max over in-support actions only.
- *"Where does Decision Transformer sit?"* → It avoids the problem structurally: no bootstrapping, no max — it's return-conditioned sequence modeling, so extrapolation error can't compound.
- *"What if I just add a behavior-cloning constraint to SAC?"* → That's essentially TD3+BC, a legitimate and strong baseline — the same "stay near the data" idea, implemented as a policy constraint rather than a value penalty.

---

### Q: What does MCTS add on top of a trained policy network, and why does AlphaZero improve at all? [Medium]

The policy net is fast intuition — one forward pass, one action. MCTS is deliberation: it simulates ahead, spending more visits on promising lines, and returns a visit-count distribution that is a *better* policy than the raw net. AlphaZero's loop is a policy-improvement operator: search improves on the net, the net trains toward the search result, and the improved net makes the next search better still. Self-play supplies the data; there's no human ceiling.

**Cross-questions to expect:**
- *"Why the most-visited move, not the highest-value?"* → Visit count is a robust, low-variance signal — PUCT concentrates visits on genuinely good moves, whereas a single high $Q$ can be a noisy leaf estimate.
- *"What's PUCT balancing?"* → Exploitation (mean action value $Q$) against exploration (the prior $P(a)$ times $\sqrt{N_{\text{parent}}}/(1+N)$), so unexplored high-prior moves still get tried.
- *"How is MuZero different?"* → It isn't given the rules — it learns a value-equivalent latent model and plans inside it, so it works where the transition dynamics are unknown.

**Trap:** Saying "MCTS just makes the policy more accurate." It doesn't refine the *network*; it produces a *better policy at decision time* by spending compute, which then becomes a training target. The distinction is the whole self-improvement mechanism.

---

### Q: Behavior cloning gets 99% action-matching accuracy in training but the agent crashes in deployment. Why, and what fixes it? [Hard]

**Covariate shift / compounding error.** BC is supervised learning under the expert's state distribution, but at deployment the agent induces its *own* distribution. A single small error moves it to a state the expert never visited, where predictions are worse, causing a larger error — errors grow quadratically in the horizon, not linearly. High per-step accuracy on the expert's states says nothing about the states the agent actually reaches. There's no recovery behavior in the data because the expert never made those mistakes.

**How I'd fix it:**
1. **DAgger** — run the current policy, have the expert label the states *it actually visited*, aggregate into the dataset, retrain. This puts recovery behavior into training. Needs an interactive/queryable expert.
2. If no interactive expert, **GAIL / inverse RL** — match the expert's *state-visitation distribution* via a discriminator (or infer a reward and do RL), which generalizes to unvisited states far better than pointwise action matching.
3. **Data augmentation on-distribution** — the classic ALVINN/NVIDIA self-driving trick: synthesize off-center camera views with corrective steering labels so the policy has seen recovery.

**Cross-questions to expect:**
- *"Why quadratic error growth specifically?"* → Per-step error probability $\epsilon$ over horizon $T$; once off-distribution the error persists and cascades, giving regret $O(\epsilon T^2)$ for BC vs. $O(\epsilon T)$ for DAgger. It's the standard Ross & Bagnell result.
- *"Isn't 99% accuracy already very high?"* → Independence is the fallacy — errors aren't independent across steps; one drift changes the entire downstream distribution. Accuracy on the wrong distribution is the wrong metric.
- *"How does this connect to LLMs?"* → SFT is behavior cloning on demonstrations and has the same ceiling; it can't recover from its own errors mid-generation, which is part of why RLHF (learning from the model's *own* on-policy outputs) is layered on top.

**Trap:** Reaching for "collect more expert data" or "bigger model." More i.i.d. expert data doesn't cover the agent's *own* error states — that's a distribution problem, not a quantity problem. The fix must change *which* states are labeled.

---

## Connections

- **Foundations:** [01-rl-foundations.md](01-rl-foundations.md) — MDPs, Bellman, the model $P(s'\mid s,a)$ that model-based RL learns
- **Value methods:** [03-value-based-methods.md](03-value-based-methods.md) — the Q-learning / bootstrapping machinery offline RL has to tame
- **Policy gradients:** [04-policy-gradient-methods.md](04-policy-gradient-methods.md) — online model-free core (PPO/SAC) and RLHF; SFT-as-BC vs. RLHF-beyond-imitation
- **Exploration:** [02-bandits-and-exploration.md](02-bandits-and-exploration.md) — UCB is the bandit ancestor of MCTS's UCT
