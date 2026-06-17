---
module: Specialized Domains
topic: Flashcards
status: unread
tags: [flashcards, specializeddomains, ml]
---
# Specialized Domains Flashcards


**I ⊆ S?** #flashcard
initiation set (when option can start)

**π_ω?** #flashcard
option policy (what to do during option)

**β?** #flashcard
S → [0,1]: termination condition

**BC is simple but suffers from compounding error; DAgger fixes this by iteratively querying the expert on states the current policy actually visits.?** #flashcard
BC is simple but suffers from compounding error; DAgger fixes this by iteratively querying the expert on states the current policy actually visits.

**GAIL?** #flashcard
discriminator distinguishes expert vs policy trajectories; PPO maximizes discriminator "fooling" → implicit imitation without reward design.

**IRL?** #flashcard
learn a reward function that makes the expert's behavior optimal. MaxEntIRL is the classic formulation; RLHF preference learning is its modern application.

**MARL: non-stationarity is the core challenge?** #flashcard
as agents learn, the environment each agent faces changes. CTDE (centralized training, decentralized execution) is the standard paradigm.

**HER: reuse failed trajectories by treating the final state as a hindsight goal?** #flashcard
key for sparse reward robotics. Generates positive reward signal from every trajectory regardless of outcome.

**Domain randomization?** #flashcard
vary sim parameters so the policy generalizes; the real world is "just another sample." The key assumption is that the real world falls within the randomization distribution.

**Offline RL needs conservative Q-estimates (CQL) or avoidance of OOD actions entirely (IQL, Decision Transformer) because the dataset is fixed and errors cannot be corrected by collecting new data.?** #flashcard
Offline RL needs conservative Q-estimates (CQL) or avoidance of OOD actions entirely (IQL, Decision Transformer) because the dataset is fixed and errors cannot be corrected by collecting new data.

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
