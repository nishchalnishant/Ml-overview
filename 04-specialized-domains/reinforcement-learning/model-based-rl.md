---
module: Specialized Domains
topic: Reinforcement Learning
subtopic: Model Based Rl
status: unread
tags: [specializeddomains, ml, reinforcement-learning-model-b]
---
# Model-Based RL & Advanced Planning

**TL;DR:** Model-based RL learns a world model (dynamics + reward) to plan or generate synthetic experience, achieving 10–100× better sample efficiency than model-free methods at the cost of compounding model errors. Modern approaches (DreamerV3, MuZero) learn compact latent models that sidestep the need for pixel-level reconstruction. MCTS + learned value/policy functions underpin AlphaZero and o1-style LLM reasoning.

---

## Quick Access

| Section | What you get |
| :--- | :--- |
| [Model-Free vs Model-Based](#1-model-free-vs-model-based) | Core tradeoff, when to use each |
| [Learned World Models](#2-learned-world-models) | Model types, rollout for planning |
| [Dyna Architecture](#3-dyna-architecture) | Dyna-Q, real vs simulated mixing |
| [MCTS](#4-monte-carlo-tree-search-mcts) | UCB1, AlphaZero integration |
| [Latent Dynamics Models](#5-latent-dynamics-models) | DreamerV3, MuZero |
| [Offline RL](#6-offline-rl) | CQL, Decision Transformer, IQL |
| [Inverse RL & Reward Learning](#7-inverse-rl--reward-learning) | Max-entropy IRL, GAIL, RLHF |
| [Hierarchical RL](#8-hierarchical-rl) | Options, skill discovery |
| [Multi-Agent RL](#9-multi-agent-rl) | CTDE, MADDPG, Nash equilibrium |
| [Connection to LLMs](#10-connection-to-llms) | MCTS + world models in o1/reasoning |
| [Interview Q&A](#11-interview-questions) | 6 high-signal questions |

---

# 1. Model-Free vs Model-Based

**The problem:** Pure trial-and-error (model-free) RL requires millions of environment interactions to learn a good policy. For robotics or drug discovery, each interaction is expensive or slow. Can we do better by learning how the world works, not just what to do in it?

**The core insight:** Model-free methods learn a mapping from experience to behavior (policy or value). Model-based methods additionally learn a *model of the environment* — transition dynamics P(s'|s,a) and reward R(s,a) — and use that model to plan or generate synthetic training data.

| Property | Model-Free | Model-Based |
| :--- | :--- | :--- |
| Sample efficiency | Low (needs real env steps) | High (plans or simulates) |
| Asymptotic performance | Often higher (no model bias) | Can be limited by model error |
| Compute per step | Low | Higher (planning / rollout) |
| Stability | Generally more stable | Compounding model errors |
| Examples | DQN, PPO, SAC | Dyna-Q, DreamerV3, MuZero |
| When to prefer | Cheap simulators; games | Real robots; expensive environments |

**What breaks:** A learned model is never perfect. Small per-step prediction errors compound geometrically over a long rollout. Planning for 10 steps with 5% error per step yields ~60% degraded predictions. This is why model-based methods typically use short rollout horizons or operate in compact latent spaces.

---

# 2. Learned World Models

**The core insight:** The world model must predict what happens next. The choice of representation determines what "next" means — full observation reconstruction vs. a latent code.

### Model Types

| Type | What it predicts | Representative use |
| :--- | :--- | :--- |
| Deterministic | Single next state | Simple MPC, linear quadratic regulator |
| Stochastic | Distribution over next states | RSSM in Dreamer; captures aleatoric uncertainty |
| Latent | Compressed representation, no pixel reconstruction | MuZero, DreamerV3 |
| Ensemble | Multiple models, disagreement = epistemic uncertainty | MBPO, PETS |

### Model Rollout for Planning

Given a learned model f_θ(s,a) → s', planning proceeds by:

1. Start from current real state s_0.
2. Simulate k steps using policy π: s_1 = f(s_0,a_0), s_2 = f(s_1,a_1), ...
3. Use the simulated trajectory to update the policy or value function — no real environment interaction needed.

**Key hyperparameter:** rollout horizon k. Longer horizons allow multi-step credit assignment but amplify model error. Typical values: 1–5 steps for MBPO, full episodes in latent space for Dreamer.

### Uncertainty-Aware Models (Ensembles)

Train N independent models. Use *disagreement* (variance across ensemble predictions) as epistemic uncertainty. Only trust rollouts in regions where all models agree — stop rollout or reduce weight when uncertainty spikes.

```python
# Ensemble disagreement as uncertainty signal
preds = [model_i(s, a) for model_i in ensemble]          # N predictions
mean  = torch.stack(preds).mean(0)
uncertainty = torch.stack(preds).var(0).mean()            # scalar uncertainty
if uncertainty > threshold:
    terminate_rollout()
```

---

# 3. Dyna Architecture

**The problem:** Model-free RL is sample-inefficient because every policy update requires new real experience. Can we augment real experience with synthetic data from a learned model?

**The core insight:** Dyna separates two loops — a real-experience loop that improves the model, and a simulated-experience loop that improves the policy. Each real step generates k synthetic steps at negligible additional environment cost.

### Dyna-Q Algorithm

```
Initialize Q(s,a), model M(s,a)
For each step:
    1. Take real action a in env → observe (s, a, r, s')
    2. Update Q directly from (s, a, r, s')          # direct RL
    3. Update model: M(s, a) ← (r, s')               # model learning
    4. Repeat k times:                               # planning
         Sample (s̃, ã) randomly from visited pairs
         r̃, s̃' = M(s̃, ã)
         Q(s̃, ã) ← Q(s̃, ã) + α[r̃ + γ max_a Q(s̃', a) - Q(s̃, ã)]
```

**The tradeoff:** With a perfect model, larger k → faster learning. With an imperfect model, large k propagates model errors into Q. Typical values: k = 5–50 for tabular; shorter for neural models.

### Real vs. Simulated Experience Mixing

| Ratio | Effect |
| :--- | :--- |
| 100% real | Standard Q-learning; low sample efficiency |
| Mixed (k=5) | ~5× fewer environment steps for same performance |
| 100% simulated | Q converges to model's optimal policy, not the real one |

**Compounding model errors:** If the model learns a biased transition (e.g., overestimates velocity), every simulated step drifts further from reality. After 20 steps the simulated state may be physically impossible. Practical fixes: (1) short rollouts, (2) ensemble models with uncertainty termination, (3) conservative use — only use simulated data for value updates, not full trajectories.

---

# 4. Monte Carlo Tree Search (MCTS)

**The problem:** In large combinatorial decision spaces (e.g., Go has ~10^170 states), exhaustive tree search is impossible. How do you efficiently plan by focusing computation on the most promising parts of the tree?

**The core insight:** Balance exploration and exploitation within the search tree. Expand nodes that are both promising (high value) and under-explored (high uncertainty), guided by a learned policy to prune the tree, and a learned value function to avoid simulating to terminal states.

### Four Phases

**1. Selection** — Traverse the tree from root to a leaf, choosing children by UCB1 (Upper Confidence Bound):

$$\text{UCT}(s,a) = Q(s,a) + c \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

Where Q(s,a) = estimated value, P(s,a) = prior from policy network, N(s) = parent visit count, N(s,a) = child visit count. The c coefficient controls exploration strength.

**2. Expansion** — When a leaf node is reached (node not yet in tree), add it and initialize its statistics using the policy prior P(s,·) from the neural network.

**3. Simulation / Evaluation** — Estimate leaf value using either: (a) random rollout to terminal state, or (b) a learned value network V(s). AlphaZero uses (b) exclusively — no random rollouts.

**4. Backpropagation** — Update Q and N for every node on the path from leaf to root:

$$N(s,a) \mathrel{+}= 1 \quad Q(s,a) \leftarrow Q(s,a) + \frac{v - Q(s,a)}{N(s,a)}$$

### AlphaZero Integration

AlphaZero replaces random simulation with a dual-head network f_θ(s) → (p, v):
- p = policy prior over actions (used in UCT)
- v = value estimate (used instead of rollout)

At each MCTS run, the network is queried once per new node expansion. The final action is chosen proportionally to visit counts N(s,a) — visit counts are more robust than Q values because they're averaged over many simulations.

```
Training loop:
    1. Run MCTS from current state → get visit counts π_mcts
    2. Sample action from π_mcts; step environment
    3. When game ends: propagate outcome z (win/lose/draw)
    4. Train f_θ to minimize:
           loss = (z - v)²  +  cross_entropy(p, π_mcts)
```

**Key insight:** The policy network guides search; the search improves the policy. This self-improving loop (policy iteration via search) is why AlphaZero surpasses human play without human knowledge.

---

# 5. Latent Dynamics Models

**The problem:** Learning to predict future observations in pixel space is sample-inefficient and computationally expensive. Most pixels are irrelevant to the task. Can we learn to predict in a compact learned representation?

### DreamerV3

DreamerV3 trains entirely in a learned latent space using the Recurrent State Space Model (RSSM):

**RSSM architecture:**
- Recurrent state h_t (deterministic): h_t = f(h_{t-1}, z_{t-1}, a_{t-1})   ← GRU
- Stochastic state z_t: z_t ~ q(z_t | h_t, x_t)   ← posterior (uses real observation)
- Prior: z_t ~ p(z_t | h_t)   ← prior (used for rollouts without real obs)

The joint state (h_t, z_t) is the latent state passed to actor-critic.

**Three training objectives:**

| Component | Objective |
| :--- | :--- |
| World model | Predict observations, rewards, episode ends from latent; minimize KL between prior and posterior (ELBO) |
| Actor | Maximize imagined returns in latent space; gradients backprop through latent rollouts |
| Critic | Fit value function to λ-returns from imagined trajectories |

**DreamerV3 key innovations over V1/V2:**
- Symlog predictions (log-scale targets) for reward normalization across scales
- Free bits KL: don't penalize if KL < threshold, preventing posterior collapse
- Percentile return normalization for stable gradients

### MuZero

**The problem:** Dreamer requires reconstructing observations to train the world model. MuZero asks: do we even need that?

**The core insight:** The world model only needs to be accurate *for planning*, not for observation reconstruction. Learn a model that supports MCTS directly — no decoder required.

**Three learned functions:**
- Representation: h(o) → s (encode observation to latent state)
- Dynamics: g(s,a) → (r, s') (predict reward and next latent state)
- Prediction: f(s) → (p, v) (predict policy prior and value)

**Training:** Play using MCTS + current model. Store (observation, action, reward, search policy, outcome). Unroll the model k steps through real actions; minimize:

$$\mathcal{L} = \sum_{t}^{t+k} \ell^r(r_\tau, \hat{r}_\tau) + \ell^v(z_\tau, \hat{v}_\tau) + \ell^p(\pi_\tau, \hat{p}_\tau)$$

**What this buys:** MuZero works on games with unknown rules — it never has access to the true dynamics, only experiences. It matched AlphaZero on Chess/Go/Shogi and surpassed DQN on Atari without knowing the rules.

| | DreamerV3 | MuZero |
| :--- | :--- | :--- |
| World model trained by | Observation reconstruction + KL | Planning consistency (no reconstruction) |
| Planning method | Latent actor-critic (gradient backprop) | MCTS |
| Known rules required | No | No |
| Suited for | Continuous control, visual tasks | Board games, discrete planning |
| Reconstruction overhead | Yes | No |

---

# 6. Offline RL

> Full coverage: [`advanced-rl.md` — Offline / Batch RL section](#)

**The core problem:** Online RL cannot be used when environment interaction is dangerous (medical devices, autonomous vehicles) or prohibitively expensive. Offline RL learns from a fixed dataset of pre-collected transitions. The fundamental challenge: the learned policy will query Q(s,a) for actions never seen in the dataset — bootstrapped Q-learning extrapolates these optimistically, causing policy collapse.

**The three main algorithms:**

| Algorithm | Conservatism mechanism | Can stitch suboptimal trajectories? |
| :--- | :--- | :--- |
| **CQL** | Penalize OOD Q-values (push down); push up in-dataset Q | Yes |
| **IQL** | Never query OOD actions; learn V(s) via expectile regression | Partially |
| **Decision Transformer** | Reframe as sequence modeling; condition on Return-to-Go | No — limited to imitating best trajectories in data |

**CQL key loss:**
$$\mathcal{L}_{CQL} = \mathcal{L}_{\text{Bellman}} + \alpha \left[\mathbb{E}_{a \sim \mu}[Q(s,a)] - \mathbb{E}_{(s,a) \sim \mathcal{D}}[Q(s,a)]\right]$$

**Decision Transformer:** Input `(R_1, s_1, a_1, R_2, s_2, ...)`, condition on high RTG at test time. No Bellman backups; fails to stitch suboptimal trajectories.

**IQL:** Expectile regression with τ > 0.5 approximates max over in-distribution actions without OOD queries.

**What breaks (all offline RL):** Dataset coverage is the ceiling. If the optimal policy requires actions never taken in the logged data, no offline method can discover them.

---

# 7. Inverse RL & Reward Learning

> Full coverage: [`advanced-rl.md` — Imitation Learning and Inverse RL sections](#)

**The core problem:** Reward functions are hard to specify for complex tasks (driving, surgery, content moderation). IRL recovers a reward function R(s,a) under which the expert's demonstrations are optimal.

**MaxEntropy IRL (Ziebart et al., 2008):** Among all reward functions consistent with expert behavior, choose the one that maximizes entropy of the resulting policy — the least-committed explanation. Feature expectations of the learned policy must match the expert's.

$$\max_{R_\theta} \mathcal{H}(\pi_\theta) \quad \text{s.t.} \quad \mathbb{E}_{\pi_\theta}[\phi(s,a)] = \mathbb{E}_{\pi_E}[\phi(s,a)]$$

**GAIL:** Skip explicit reward recovery. Match state-action occupancy measure between expert and policy via a GAN discriminator. Discriminator output becomes the RL reward signal; PPO is the "generator."

**Connection to RLHF:** RLHF is IRL with human preference comparisons as the reward signal source. The KL penalty vs. reference policy serves the same role as entropy regularization in MaxEntIRL — preventing reward hacking.

| Method | Reward signal source | Requires forward RL? | Key issue |
| :--- | :--- | :--- | :--- |
| MaxEntropy IRL | Demonstrations | Yes | Expensive inner-loop RL |
| GAIL | Demonstrations (adversarial) | Yes (PPO) | Training instability |
| RLHF | Human preference comparisons | Yes (PPO) | Reward model overfitting |

---

# 8. Hierarchical RL

> Full coverage: [`advanced-rl.md` — Hierarchical Reinforcement Learning section](#)

**The core problem:** Long-horizon sparse reward tasks defeat flat RL. Temporal abstraction — decomposing tasks into a hierarchy of subtasks — makes credit assignment tractable.

**Options framework:** An option ω = (I_ω, π_ω, β_ω) is a temporally extended action. I_ω = initiation set; π_ω = option policy; β_ω = termination condition. The high-level policy selects options; the low-level executes them.

**HER (Hindsight Experience Replay):** Relabel failed trajectories with the state actually reached as a hindsight goal. Converts sparse reward environments to dense by generating a positive training signal from every trajectory.

**Feudal RL:** Stack multiple hierarchy levels. Each level operates at a dilated timescale and sets subgoals for the level below.

**Skill discovery — DIAYN:**
$$\max_\theta \mathcal{I}(s; z) = \mathcal{H}(z) - \mathcal{H}(z | s)$$
Train skills to produce distinguishable state distributions; discriminator identifies which skill produced which states.

**HIRO:** Manager sets subgoals in state space every k steps; worker receives intrinsic reward for reaching them; off-policy correction relabels historical subgoals for consistency.

**What breaks:** The goal representation problem — the manager must set goals in a space the worker can navigate. End-to-end training is unstable; levels have a moving target problem as both are learning simultaneously.

---

# 9. Multi-Agent RL

> Full coverage: [`advanced-rl.md` — Multi-Agent Reinforcement Learning section](#)

**The core problem:** As multiple agents learn simultaneously, the environment is non-stationary from each agent's perspective — standard Q-learning convergence proofs break.

**CTDE (Centralized Training, Decentralized Execution):** Train with global state and joint observations; execute with local observations only. Resolves non-stationarity during training while keeping deployment scalable.

**Nash Equilibrium:** No agent can improve expected reward by unilaterally changing policy. Guaranteed to exist in finite games; convergence not guaranteed for general-sum games in MARL.

$$\forall i: \pi_i^* \in \arg\max_{\pi_i} \mathbb{E}\left[R_i \mid \pi_i, \pi_{-i}^*\right]$$

**Key algorithms:**

| Algorithm | Setting | Key idea |
| :--- | :--- | :--- |
| MADDPG | Cooperative/competitive, continuous | CTDE with per-agent centralized critics |
| QMIX | Cooperative, discrete | Monotonic mixing network for factored joint Q — enables IGM |
| MAPPO | Cooperative | PPO with shared centralized value function |
| COMA | Cooperative | Counterfactual baseline: marginalizes out one agent's action to isolate individual credit |

**MADDPG:** Per-agent actor π_i(o_i) uses only local observation; centralized critic Q_i(o_1,...,o_N, a_1,...,a_N) sees everything during training.

**Self-play (competitive):** Train agent against copy of current policy. Drives emergent strategy. Converges toward Nash in two-player zero-sum; can cycle in multi-player settings.

---

# 10. Connection to LLMs

**The problem:** Language models generate text autoregressively — one token at a time, committed and irrevocable. For complex reasoning tasks (math, code, planning), greedy token generation misses solutions that require backtracking or multi-step deliberation.

**The core insight:** If you model token generation as a sequential decision process, the machinery of model-based RL becomes applicable. The LLM is the *policy*; the token sequence is the *trajectory*; correct answers are *terminal rewards*. A world model predicts the value of intermediate reasoning steps.

### MCTS + LLMs (o1-Style Reasoning)

**Process Reward Models (PRMs):** Train a separate model to score intermediate reasoning steps (not just final answers). This is the "value network" in MCTS — it allows early termination of unpromising reasoning branches without completing them.

```
MCTS on reasoning:
- State:  current chain-of-thought prefix
- Action: next reasoning step (sentence or token chunk)
- Policy prior:  base LLM p(next_step | prefix)
- Value:  PRM(prefix) → estimate of eventual correctness
- Backprop: propagate outcome (correct/wrong) up the tree
```

**Analogy to AlphaZero:**

| AlphaZero | LLM Reasoning (o1-style) |
| :--- | :--- |
| Board state | Chain-of-thought prefix |
| Legal moves | Candidate next reasoning steps |
| Policy network | Base LLM prior over steps |
| Value network | Process Reward Model (PRM) |
| Terminal reward | Answer correctness |
| MCTS | Tree search over reasoning paths |

### World Models and Chain-of-Thought

Chain-of-thought prompting can be viewed as the LLM constructing an *implicit world model* in token space — simulating intermediate states (calculations, logical deductions) before committing to an action (final answer). DreamerV3's latent rollouts and CoT both trade compute for sample efficiency: more thinking reduces the number of real queries needed.

**Inference-time compute scaling** (the o1/o3 paradigm): allocating more MCTS simulations at inference is analogous to increasing rollout depth in Dyna — more planning per real step, same underlying model.

---

# 11. Interview Questions

**Q1: Why does naive Q-learning fail in offline RL, and how does CQL fix it?**

Offline Q-learning bootstraps on max_a Q(s',a) — but the maximizing action a may be out-of-distribution (never appeared in the offline dataset). The function approximator extrapolates optimistically for OOD actions, producing inflated Q-values. The policy then greedily selects these OOD actions, which were never validated by real experience, causing policy collapse.

CQL adds a penalty that explicitly pushes Q-values *down* for actions sampled from a broad distribution (approximating OOD) and *up* for actions in the dataset. The resulting Q-function is a lower bound on the true value — conservative by construction. The policy, being greedy w.r.t. a conservative Q, selects actions that are *actually* supported by data.

---

**Q2: What is the role of the process reward model in o1-style reasoning, and how does it connect to RL value functions?**

A PRM is trained to estimate the probability that a partial reasoning trajectory will lead to a correct final answer, given only the steps so far. It is the *value function* in the MCTS formulation of reasoning: V(prefix) ≈ P(correct | current reasoning state). Like a learned value function in AlphaZero, it allows the search to evaluate non-terminal nodes without rolling out to the answer — making deep tree search tractable.

---

**Q3: DreamerV3 trains actor-critic entirely in latent space. What prevents the actor from exploiting model errors?**

Several mechanisms: (1) The RSSM's stochastic state z_t captures uncertainty — high variance in z implies the model is uncertain, which propagates into imagined rewards and values. (2) Free bits KL prevents the posterior from collapsing to the prior, keeping the stochastic state informative. (3) Symlog reward normalization and percentile return normalization prevent the actor from chasing extremely large imagined rewards that are likely artifacts of model extrapolation. (4) In practice, Dreamer uses finite-horizon imagined rollouts (typically 15 steps), limiting error accumulation.

---

**Q4: In MCTS, why use visit counts N(s,a) for the final action selection rather than the Q-value directly?**

Q(s,a) is an average over all simulations that passed through (s,a) and may have high variance, especially for rarely-visited nodes. N(s,a) is proportional to how often MCTS chose that action after evaluating all alternatives — it represents the *search's accumulated vote* after exploration. Highly-visited nodes are likely to be good both because they had high Q estimates *and* because they remained competitive as the search ran more simulations. This makes the visit count a more robust, low-variance action signal than Q alone.

---

**Q5: What is the options framework and why does it help with long-horizon sparse reward?**

An option is a temporally extended action: a (initiation set, policy, termination condition) triple. The high-level policy selects options rather than primitive actions. This addresses sparse reward in two ways: (1) Temporal abstraction — the high-level policy sees one reward signal per option execution rather than per primitive step, shortening the effective horizon and improving credit assignment. (2) Reuse — a learned locomotion option can be shared across many high-level goals, amortizing exploration cost.

---

**Q6: What breaks in standard single-agent RL when applied naively to multi-agent settings?**

The Markov assumption breaks: from agent i's perspective, the environment's transition dynamics depend on other agents' policies, which are changing during training. This makes the environment non-stationary — Q-values estimated at one point in training are invalidated as other agents' policies update. Additionally, credit assignment becomes ambiguous: in cooperative settings, a shared reward does not indicate which agent contributed to it. CTDE addresses non-stationarity by conditioning the critic on all agents' information during training; counterfactual baselines (COMA) address credit assignment.

---

**Q7: MuZero learns a world model without ever predicting observations. What does it predict instead, and why is this sufficient?**

MuZero's dynamics model predicts: (1) the reward r for the action taken, and (2) a latent next state s' (abstract, not tied to pixel reconstruction). Its prediction network then maps s' to a policy prior p and value v — the two quantities needed for MCTS. The model never reconstructs the raw observation.

This is sufficient because the world model is trained to be accurate *for planning*, not perceptually faithful. The latent states are supervised by consistency with downstream value and policy targets (through MCTS-derived π and eventual game outcomes z), forcing them to capture decision-relevant information. Pixel-level details (shadows, textures) that are irrelevant to the game outcome are simply not learned — which is exactly what you want.

## Flashcards

**p = policy prior over actions (used in UCT)?** #flashcard
p = policy prior over actions (used in UCT)

**v = value estimate (used instead of rollout)?** #flashcard
v = value estimate (used instead of rollout)

**Recurrent state h_t (deterministic)?** #flashcard
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})   ← GRU

**Stochastic state z_t?** #flashcard
z_t ~ q(z_t | h_t, x_t)   ← posterior (uses real observation)

**Prior?** #flashcard
z_t ~ p(z_t | h_t)   ← prior (used for rollouts without real obs)

**Symlog predictions (log-scale targets) for reward normalization across scales?** #flashcard
Symlog predictions (log-scale targets) for reward normalization across scales

**Free bits KL?** #flashcard
don't penalize if KL < threshold, preventing posterior collapse

**Percentile return normalization for stable gradients?** #flashcard
Percentile return normalization for stable gradients

**Representation?** #flashcard
h(o) → s (encode observation to latent state)

**Dynamics?** #flashcard
g(s,a) → (r, s') (predict reward and next latent state)

**Prediction?** #flashcard
f(s) → (p, v) (predict policy prior and value)

**I_ω: initiation set?** #flashcard
states where the option can be started

**π_ω: option policy?** #flashcard
a policy for the duration of the option

**β_ω: termination condition?** #flashcard
probability of ending the option at each state

**Manager sets a subgoal g every k steps (in state space).?** #flashcard
Manager sets a subgoal g every k steps (in state space).

**Worker receives intrinsic reward for reaching g.?** #flashcard
Worker receives intrinsic reward for reaching g.

**Off-policy correction?** #flashcard
relabel historical subgoals to make manager data consistent.

**Each agent i has its own actor π_i(o_i)?** #flashcard
takes only local observation.

**Each agent has a centralized critic Q_i(o_1,...,o_N, a_1,...,a_N)?** #flashcard
sees all agents' observations and actions during training.

**At execution, only the actor is used.?** #flashcard
At execution, only the actor is used.

**State?** #flashcard
current chain-of-thought prefix

**Action?** #flashcard
next reasoning step (sentence or token chunk)

**Policy prior?** #flashcard
base LLM p(next_step | prefix)

**Value?** #flashcard
PRM(prefix) → estimate of eventual correctness

**Backprop?** #flashcard
propagate outcome (correct/wrong) up the tree
