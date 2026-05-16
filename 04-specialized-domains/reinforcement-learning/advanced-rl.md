# Advanced Reinforcement Learning

Extends core RL (MDPs, Q-learning, PPO) with multi-agent, imitation learning, inverse RL, hierarchical RL, and sim-to-real transfer.

---

## Imitation Learning

Learn a policy from expert demonstrations without a reward function.

### Behavioral Cloning (BC)

Supervised learning on (state, action) pairs from expert trajectories.

```python
# Dataset: state-action pairs from expert
class BC(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, action_dim))
    def forward(self, s): return self.net(s)

# Train: minimize MSE (continuous) or cross-entropy (discrete)
loss = F.mse_loss(policy(states), expert_actions)
```

**Compounding error problem:** Small errors accumulate — the agent visits states the expert never did, making predictions unreliable.

**DAgger (Dataset Aggregation):** Iteratively query expert on states visited by the current policy, aggregate into training set. Converges to expert performance.

```
Initialize D = expert_demonstrations
For each iteration:
    Run policy π on environment → states S_i
    Query expert for actions: A_i = π_expert(S_i)
    Aggregate: D = D ∪ {(S_i, A_i)}
    Train π on D
```

### GAIL — Generative Adversarial Imitation Learning

Frame imitation as matching the state-action occupancy measure between expert and policy. Use a discriminator to distinguish expert from policy trajectories; use PPO as the "generator."

```
Discriminator: D(s, a) = P(expert | s, a)
Reward for RL: r(s, a) = -log(1 - D(s, a))   (encourage expert-like behavior)
Policy update: PPO with reward r(s, a)
```

**Advantages over BC:** No compounding errors; learns reward function implicitly; handles stochastic experts.

---

## Inverse Reinforcement Learning (IRL)

Learn a reward function from expert demonstrations. Then solve the MDP under the learned reward.

**MaxEntropy IRL (Ziebart et al., 2008):** Learn reward `r(s, a)` such that the expert's behavior is the maximum entropy distribution consistent with the demonstrated feature expectations.

```
Objective: maximize H(π) - Σ_{s,a} r(s,a)(d_expert(s,a) - d_policy(s,a))
```

where `d(s,a)` is the state-action occupancy measure.

**Applications:** RLHF preference learning, autonomous driving reward design, robot manipulation.

---

## Multi-Agent Reinforcement Learning (MARL)

Multiple agents act in shared or separate environments, potentially with competing or cooperative objectives.

### Cooperative (Same reward)

All agents share a team reward. Challenge: credit assignment — which agent's action caused the good outcome?

**QMIX:** Factorize the joint Q-function as a monotone mixing of individual Q-functions.

`Q_total(s, a_1, ..., a_n) = f_mix(Q_1(s_1, a_1), ..., Q_n(s_n, a_n))`

`f_mix` is a monotone hypernetwork — ensures global argmax ↔ each agent's local argmax.

**MAPPO:** Multi-agent PPO with centralized critic (sees global state), decentralized actors (each sees local observation).

### Competitive (Zero-sum)

Agents have opposing goals. Classic setting: two-player zero-sum games.

**Self-play:** Each agent trains against a copy of itself. Drives emergent complex strategies (AlphaGo, OpenAI Five, AlphaStar).

```
Copy current policy → opponent
Run episodes: agent vs opponent
Update agent with RL
Periodically update opponent copy
```

**Nash Equilibrium:** Solution concept for competitive games — no agent can improve by unilaterally changing strategy.

### Mixed (Cooperative + Competitive)

Real-world settings: traffic (cooperate with nearby cars, compete for space), trading.

### Non-Stationarity Problem

As one agent learns, the environment (from others' perspective) changes → non-Markovian from any single agent's view. Standard RL convergence guarantees break down.

**Centralized Training, Decentralized Execution (CTDE):** Train with global state access, execute with local observations only.

---

## Hierarchical Reinforcement Learning

Decompose complex tasks into sub-goals and primitives. High-level policy sets sub-goals; low-level policy achieves them.

### Options Framework

An **option** `ω = (I, π_ω, β)`:
- `I ⊆ S`: initiation set (when option can start)
- `π_ω`: option policy (what to do during option)
- `β: S → [0,1]`: termination condition

```
High-level policy: select option ω at each sub-goal timestep
Low-level policy: execute π_ω until β terminates it
```

**Option-Critic:** Learn options end-to-end with gradient through both levels.

### HER — Hindsight Experience Replay

For sparse reward environments (reward only at goal). Re-label failed trajectories with the actual outcome as the "goal" — creates positive reward signal.

```python
# Agent tried to reach goal g, ended at state s_T
# Re-label: pretend g_hindsight = s_T
# Now this trajectory was "successful" for goal g_hindsight
buffer.store(s, a, r=-1, s', done, goal=g)
buffer.store(s, a, r=0,  s', done, goal=s_T)  # hindsight goal
```

**Used in:** Robot manipulation, goal-conditioned RL with sparse rewards.

### Feudal RL

Manager sets goals for sub-managers; sub-managers set goals for workers. Dilated temporal abstractions across levels.

---

## Meta-RL (Learning to Learn in RL)

Learn across a distribution of tasks such that adapting to a new task requires very few interactions.

### MAML for RL

Apply MAML (see `transfer-learning.md`) to policy gradient: learn initialization θ that can be fine-tuned to any new task in K rollouts.

### RL² (RL Squared)

Use a recurrent policy that treats the entire trajectory (including rewards) as input. The RNN's hidden state acts as a "memory" that accumulates task-relevant information across episodes.

```
At each step: input = (s_t, a_{t-1}, r_{t-1}, done_{t-1})
RNN hidden state h_t encodes inferred task identity
After K episodes: policy has "adapted" via hidden state — no gradient update needed
```

---

## Curriculum Learning in RL

Present tasks in increasing order of difficulty. Prevents the agent from learning bad habits on easy tasks or getting stuck on hard ones.

### Automatic Curriculum

**Self-play Curriculum:** The opponent's skill defines the curriculum naturally (as agent improves, so does opponent).

**ALP-GMM (Absolute Learning Progress with Gaussian Mixture Models):** Track which task regions produce the fastest learning progress; sample from there.

```
For each task parameter τ:
    Track learning progress LP(τ) = |improvement in performance over recent trials|
    Sample τ proportional to LP(τ)
```

---

## Sim-to-Real Transfer

Train in simulation (cheap, safe, parallelizable), deploy in reality. The gap between sim and real is the key challenge.

### Domain Randomization

Vary simulation parameters (friction, mass, lighting, camera noise) randomly during training. Forces the policy to work across a wide range → generalizes to the real domain.

```python
# At each episode reset, randomize sim parameters
friction = np.random.uniform(0.5, 1.5)
mass = np.random.uniform(0.8, 1.2) * nominal_mass
lighting = np.random.uniform(0.5, 1.5)
env.reset(friction=friction, mass=mass, lighting=lighting)
```

**Visual domain randomization:** Randomize textures, colors, camera positions — forces visual policies to ignore non-physical visual details.

### System Identification

Estimate real-world physics parameters from real data; use those in simulation.

```
Collect real trajectories with random actions
Fit sim parameters to minimize trajectory MSE
Retrain policy in calibrated simulation
```

### Adaptive Methods (RMA — Rapid Motor Adaptation)

Train a base policy in sim with privileged access to env parameters. Then train an adaptation module that infers those parameters from proprioceptive history (no privileged access). The adaptation module works in the real world.

**Used in:** ANYmal (quadruped locomotion), Spot (Boston Dynamics), dexterous manipulation.

### Sim-to-Real with Real Fine-Tuning

Deploy sim-trained policy in real, collect real data, fine-tune with RL. Requires safe initial policy and careful reward shaping in real environment.

---

## Offline / Batch RL

Learn from a fixed dataset of pre-collected transitions without interacting with the environment. Critical for healthcare, robotics, autonomous driving (can't run online RL).

**Challenge:** Out-of-distribution (OOD) actions — the policy may take actions not seen in the dataset, where Q-values are extrapolated poorly.

### CQL — Conservative Q-Learning

Add a penalty that minimizes Q-values for OOD actions while maximizing Q-values for in-distribution actions.

`L_CQL = L_Bellman + α (E_{a~π}[Q(s,a)] - E_{a~D}[Q(s,a)])`

### IQL — Implicit Q-Learning

Avoid querying OOD actions entirely. Learn a value function V(s) via expectile regression; derive policy via advantage-weighted regression.

### Decision Transformer

Reframe RL as sequence modeling: `(R_1, s_1, a_1, R_2, s_2, a_2, ...)`. Given a desired return-to-go, predict actions autoregressively with a Transformer.

---

## Key Interview Points

- BC is simple but suffers from compounding error; DAgger fixes this by iteratively querying the expert.
- GAIL: discriminator distinguishes expert vs policy trajectories; PPO maximizes discriminator "fooling" → implicit imitation.
- IRL: learn a reward function that makes the expert's behavior optimal. MaxEntIRL is the classic formulation.
- MARL: non-stationarity is the core challenge. CTDE (centralized training, decentralized execution) is the standard paradigm.
- HER: reuse failed trajectories by treating the final state as a hindsight goal — key for sparse reward robotics.
- Domain randomization: vary sim parameters so the policy generalizes; the real world is "just another sample."
- Offline RL needs conservative Q-estimates (CQL) or avoidance of OOD actions (IQL) because we can't collect new data to correct errors.
