---
module: Specialized Domains
topic: Reinforcement Learning
subtopic: Advanced Rl
status: unread
tags: [specializeddomains, ml, reinforcement-learning-advance]
---
# Advanced Reinforcement Learning

Extends core RL (MDPs, Q-learning, PPO) with multi-agent, imitation learning, inverse RL, hierarchical RL, and sim-to-real transfer.

---

## Imitation Learning

**The problem:** Designing a reward function for complex tasks is hard. For robot surgery, autonomous driving, or language generation, specifying exactly what "good" means numerically is often impossible. But experts exist who can demonstrate good behavior. The question: can we learn a policy from demonstrations alone, without ever defining a reward?

**The core insight:** If you have state-action pairs from an expert, policy learning reduces to supervised learning — predict the expert's action given the current state. No reward required.

**The mechanics:** Train a policy network π_θ(s) to minimize prediction error on expert data.

### Behavioral Cloning (BC)

**The problem:** You have expert demonstrations as (state, action) pairs. How do you turn these into a policy?

**The core insight:** This is supervised learning. Treat expert actions as labels. Train the policy to predict what the expert would do in each observed state.

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

**What breaks:** Compounding error. Small prediction mistakes move the agent to states the expert never visited. The policy has no training signal for those states, so it makes larger errors, drifting further still. A 1% per-step error rate compounds to near-random behavior over 100 steps. The policy is only trained on the expert's state distribution — not on its own mistakes.

### DAgger (Dataset Aggregation)

**The problem:** BC fails because at test time the agent encounters states the expert never visited — and has no training signal for those states. The training distribution does not match the deployment distribution.

**The core insight:** Close the distribution mismatch loop by iteratively generating training data from the states the *current policy* actually visits, then querying the expert for the correct action there. Over iterations, the training distribution expands to cover states the policy actually encounters.

```
Initialize D = expert_demonstrations
For each iteration:
    Run policy π on environment → states S_i
    Query expert for actions: A_i = π_expert(S_i)
    Aggregate: D = D ∪ {(S_i, A_i)}
    Train π on D
```

**What breaks:** DAgger requires querying the expert online at every iteration. For many real-world tasks, the expert cannot be queried interactively — a human surgeon cannot provide action labels for millions of states on demand. This motivates GAIL.

### GAIL — Generative Adversarial Imitation Learning

**The problem:** DAgger still requires querying the expert online, which is expensive or impossible for many real-world tasks. BC fixes the covariate shift but not the reward design problem. What if we want to learn reward-free imitation without ongoing expert access?

**The core insight:** An expert policy and a learned policy produce different distributions of (state, action) pairs. Train a discriminator to distinguish them — then use the discriminator's signal as a reward to drive the learned policy toward the expert's distribution. The discriminator is the reward function.

**The mechanics:** Frame imitation as matching the state-action occupancy measure between expert and policy. Use a discriminator to distinguish expert from policy trajectories; use PPO as the "generator."

```
Discriminator: D(s, a) = P(expert | s, a)
Reward for RL: r(s, a) = -log(1 - D(s, a))   (encourage expert-like behavior)
Policy update: PPO with reward r(s, a)
```

**Advantages over BC:** No compounding errors; learns reward function implicitly; handles stochastic experts.

**What breaks:** Training instability — same GAN failure modes apply (discriminator collapse, mode dropping). Sensitive to hyperparameters. Requires many environment interactions even though no reward is defined. Does not produce an interpretable reward function — you cannot inspect it to understand what the expert was optimizing.

---

## Inverse Reinforcement Learning (IRL)

**The problem:** GAIL learns to imitate but does not recover an interpretable reward function. For safety-critical applications — autonomous driving, medical treatment planning — you want to understand *why* the expert acts as it does, not just copy the behavior. This requires learning a reward function, not just a policy.

**The core insight:** If an expert is acting optimally under some hidden reward, the reward must make the expert's behavior the highest-value option at every state. Invert this: find the reward function that makes the observed behavior optimal.

**The mechanics:** Learn reward `r(s, a)` such that the expert's behavior is the maximum entropy distribution consistent with the demonstrated feature expectations.

**MaxEntropy IRL (Ziebart et al., 2008):**

**The problem:** Many reward functions can rationalize the same expert behavior — the optimal reward is not unique. You need a principled way to select among them.

**The core insight:** Apply the maximum entropy principle: among all reward functions that are consistent with the demonstrated behavior, prefer the one that induces the least certain (most entropy) policy. This prevents overfitting to the specific demonstrations seen.

```
Objective: maximize H(π) - Σ_{s,a} r(s,a)(d_expert(s,a) - d_policy(s,a))
```

where `d(s,a)` is the state-action occupancy measure.

**What breaks:** Requires solving the full MDP (computing the policy) in the inner loop — expensive for large state spaces. Reward function is not unique: many rewards can rationalize the same behavior. Sensitive to the feature representation chosen. For neural-network reward functions, the MaxEnt objective becomes intractable and requires sampling-based approximations.

**Applications:** RLHF preference learning, autonomous driving reward design, robot manipulation.

---

## Multi-Agent Reinforcement Learning (MARL)

**The problem:** Single-agent RL assumes the environment is stationary — transition dynamics and reward do not change during training. But in the real world, other agents are present: other cars, other traders, teammates, opponents. As each agent learns, the environment experienced by every other agent changes. Standard convergence guarantees break down entirely.

**The core insight:** The stationarity assumption is violated because agents are part of each other's environments. The solution depends on the relationship between agents: cooperative agents can share information and credit; competitive agents need equilibrium concepts; all agents benefit from centralized training even if execution must be decentralized.

**The mechanics:** Structure training around agent relationships.

### Cooperative (Same reward)

**The problem:** All agents share a team reward, but individual agents only observe local state and take local actions. Which agent's action caused the team to succeed? This is the credit assignment problem in multi-agent settings.

**The core insight:** Factorize the joint Q-function in a way that makes each agent's locally-optimal action also globally optimal — Individual Global Max (IGM) property.

**QMIX:** Factorize the joint Q-function as a monotone mixing of individual Q-functions.

`Q_total(s, a_1, ..., a_n) = f_mix(Q_1(s_1, a_1), ..., Q_n(s_n, a_n))`

`f_mix` is a monotone hypernetwork — ensures global argmax ↔ each agent's local argmax.

**MAPPO:** Multi-agent PPO with centralized critic (sees global state), decentralized actors (each sees local observation).

**What breaks:** QMIX's monotone mixing is restrictive — it cannot represent all cooperative Q-functions. Tasks where the optimal joint action requires non-monotone dependencies between agents cannot be captured. MAPPO requires a centralized critic that sees all agents' states, which scales poorly as the number of agents grows.

### Competitive (Zero-sum)

**The problem:** Agents have opposing goals. Training one agent against a fixed opponent causes it to overfit to that opponent's weaknesses, not to learn a generally strong strategy.

**The core insight:** The best opponent for training is a copy of the current policy. As the policy improves, so does the opponent, maintaining an appropriately challenging training signal.

**Self-play:** Each agent trains against a copy of itself. Drives emergent complex strategies (AlphaGo, OpenAI Five, AlphaStar).

```
Copy current policy → opponent
Run episodes: agent vs opponent
Update agent with RL
Periodically update opponent copy
```

**Nash Equilibrium:** Solution concept for competitive games — no agent can improve by unilaterally changing strategy. Self-play converges toward Nash equilibrium in two-player zero-sum games.

**What breaks:** In non-zero-sum or multi-player games, self-play can cycle — strategy A beats B beats C beats A — without converging to any stable equilibrium. Population-based training (maintaining a diverse pool of past policies as opponents) partially addresses this.

### Mixed (Cooperative + Competitive)

**The problem:** Real-world settings combine both cooperative and competitive dynamics. Agents may cooperate with teammates while competing against opponents. Standard cooperative or competitive algorithms handle only one regime.

**The core insight:** Decompose agent relationships and apply appropriate training regimes: shared rewards and CTDE for teammates, adversarial self-play for competitors.

Real-world settings: traffic (cooperate with nearby cars, compete for space), trading, team sports.

### Non-Stationarity Problem

**The problem:** As one agent learns, the environment (from others' perspective) changes — it is non-Markovian from any single agent's view. The standard Q-learning convergence proof assumes a stationary environment and breaks down when other agents are updating simultaneously.

**The core insight:** Separate what agents know during training from what they can observe during execution. During training, share global information to make the joint problem stationary. During execution, act on local observations only.

**Centralized Training, Decentralized Execution (CTDE):** Train with global state access, execute with local observations only.

**What breaks:** Even CTDE does not fully resolve non-stationarity — other agents continue to update during training. Scalability collapses as the joint action space grows exponentially with the number of agents. Reward shaping and credit assignment remain open problems at scale.

---

## Hierarchical Reinforcement Learning

**The problem:** Long-horizon sparse reward tasks defeat flat RL. Consider a robot that must open a drawer, pick up a key, walk to a door, unlock it, and exit. The reward arrives only at the end. The probability of stumbling into that reward by random exploration is vanishingly small — standard RL cannot learn this in any reasonable time.

**The core insight:** Humans solve long-horizon tasks by decomposing them: "go to the kitchen" is a high-level goal that the body (a lower-level controller) achieves automatically. A high-level policy that sets subgoals never needs to worry about joint-level motor control; a low-level policy that tracks subgoals never needs to reason about the full task. Temporal abstraction separates concerns and makes sparse rewards tractable.

**The mechanics:** A high-level policy selects subgoals or options; a low-level policy executes them. Reward flows to the high-level policy on subgoal completion and to the low-level policy on primitive step execution.

### Options Framework

**The problem:** Standard RL operates at the level of primitive actions (one per timestep). For long-horizon tasks, this granularity is too fine — the agent needs to reason about multi-step behaviors as units.

**The core insight:** Define temporally extended actions called options. Each option is a sub-policy that runs until a termination condition triggers, then hands control back to the high-level policy. The high-level policy reasons over options, not primitive actions.

An **option** `ω = (I, π_ω, β)`:
- `I ⊆ S`: initiation set (when option can start)
- `π_ω`: option policy (what to do during option)
- `β: S → [0,1]`: termination condition

```
High-level policy: select option ω at each sub-goal timestep
Low-level policy: execute π_ω until β terminates it
```

**Option-Critic:** Learn options end-to-end with gradient through both levels.

**What breaks:** Options must be hand-designed unless learned end-to-end. Learned options often degenerate: the high-level policy learns to use one option exclusively, collapsing the hierarchy. The termination condition β is difficult to learn and tends toward never terminating (options do not hand back control).

### HER — Hindsight Experience Replay

**The problem:** Even with hierarchical structure, sparse reward environments produce almost no positive signal. If the robot reaches the wrong location, every step of that trajectory has reward -1, providing no gradient information about what would have been good. The agent cannot learn from any of these failed trajectories.

**The core insight:** A failed trajectory is only a failure relative to the *intended* goal. Relative to the state the robot *actually* reached, that same trajectory was a complete success. Relabel the goal retroactively — the trajectory now carries a positive reward signal.

**The mechanics:**

```python
# Agent tried to reach goal g, ended at state s_T
# Re-label: pretend g_hindsight = s_T
# Now this trajectory was "successful" for goal g_hindsight
buffer.store(s, a, r=-1, s', done, goal=g)
buffer.store(s, a, r=0,  s', done, goal=s_T)  # hindsight goal
```

Every trajectory, regardless of whether it achieved its intended goal, generates at least one successful training example. This converts an environment with almost zero positive reward into one with dense positive signal.

**Used in:** Robot manipulation, goal-conditioned RL with sparse rewards.

**What breaks:** HER requires goal-conditioned policies and a goal space where arbitrary achieved states can serve as goals. For tasks where the goal space is not easily parameterizable (e.g., generate a useful response), HER does not apply. Also, hindsight relabeling creates off-policy data that must be handled correctly.

### Feudal RL

**The problem:** Two-level hierarchies (manager + worker) still struggle with very long-horizon tasks that require multiple levels of abstraction.

**The core insight:** Stack multiple levels of hierarchy. Each level sets goals for the level below it and operates on a dilated timescale — the top level acts once every K steps, the middle level once every k steps, the bottom level every step. Temporal abstraction is recursive.

Manager sets goals for sub-managers; sub-managers set goals for workers. Dilated temporal abstractions across levels.

**What breaks (Hierarchical RL broadly):** The high-level policy must set subgoals that are achievable by the low-level policy — but the low-level policy is still learning. This creates a moving target problem: subgoals that were achievable early may become trivially easy or impossible as the low-level policy changes. End-to-end training is unstable; pre-training levels separately requires manual task decomposition.

---

## Meta-RL (Learning to Learn in RL)

**The problem:** Standard RL agents require millions of environment interactions to learn a single task. This is acceptable for simulated games but not for physical robots that wear out, or medical trials that are expensive and ethically constrained. The question: can an agent learn how to learn, so that adapting to a new task requires only a handful of interactions?

**The core insight:** If an agent has been trained across many related tasks, it should have already learned the *structure* of the task family — which dimensions of variation matter, how to infer task identity from a few observations, and what kind of exploration is informative. Fast adaptation is then not learning from scratch; it is inference over a known task distribution.

**The mechanics:** Train across a distribution of tasks. At test time, the agent adapts in a small number of rollouts.

### MAML for RL

**The problem:** Standard gradient descent initializations are not designed for fast adaptation. A good meta-initialization should be one step of gradient descent away from optimal on any task in the distribution.

**The core insight:** Apply MAML (see `transfer-learning.md`) to policy gradient: learn initialization θ that can be fine-tuned to any new task in K rollouts. The meta-objective optimizes for performance *after* gradient steps, not before.

**What breaks:** MAML requires second-order gradients — the gradient of a gradient — which is computationally expensive. First-order approximations (FOMAML) are cheaper but less accurate. Meta-gradient estimation is very high-variance; large batch sizes are needed for stable training.

### RL² (RL Squared)

**The problem:** MAML adapts via gradient descent, which requires multiple rollouts and gradient computation at test time. Can adaptation happen without gradient steps?

**The core insight:** Use a recurrent policy that treats the entire trajectory (including rewards) as input. The RNN's hidden state acts as a "memory" that accumulates task-relevant information across episodes. No gradient update is needed at test time — adaptation is implicit in the hidden state dynamics.

```
At each step: input = (s_t, a_{t-1}, r_{t-1}, done_{t-1})
RNN hidden state h_t encodes inferred task identity
After K episodes: policy has "adapted" via hidden state — no gradient update needed
```

**What breaks:** Meta-RL requires a well-specified task distribution at training time — if the test tasks lie outside this distribution, adaptation fails. MAML requires second-order gradients (expensive). RL² requires very long context to capture task identity; RNNs forget over long horizons. Both methods fail if the test task is structurally different from the training task distribution.

---

## Curriculum Learning in RL

**The problem:** An agent learning to play chess cannot learn from grandmaster games immediately — it has no context to extract signal from such distant-from-current-ability trajectories. But training only on trivially easy games produces an agent that cannot generalize. The agent needs to face tasks that are hard enough to force improvement but easy enough to produce a learning signal.

**The core insight:** Learning progress is the signal: a task that is being mastered produces rapid improvement; a task already mastered or completely unsolvable produces none. Sample training tasks proportionally to where improvement is currently happening.

**The mechanics:** Present tasks in increasing order of difficulty. Prevents the agent from learning bad habits on easy tasks or getting stuck on hard ones.

### Automatic Curriculum

**The problem:** Manually specifying the difficulty ordering of tasks requires domain expertise and becomes infeasible for large task spaces.

**The core insight:** Measure improvement directly and let it drive task selection. Tasks producing the most learning signal are the most valuable — sample them more.

**Self-play Curriculum:** The opponent's skill defines the curriculum naturally (as agent improves, so does opponent). No manual difficulty specification needed.

**ALP-GMM (Absolute Learning Progress with Gaussian Mixture Models):** Track which task regions produce the fastest learning progress; sample from there.

```
For each task parameter τ:
    Track learning progress LP(τ) = |improvement in performance over recent trials|
    Sample τ proportional to LP(τ)
```

**What breaks:** Measuring learning progress requires comparing performance across time — noisy in RL. If the curriculum advances too quickly, the agent never consolidates skills. If it advances too slowly, the agent over-specializes on easy tasks. Automatic curriculum methods assume smooth task difficulty, which may not hold — some tasks have sudden difficulty cliffs where improvement jumps discontinuously.

---

## Sim-to-Real Transfer

**The problem:** Real-world RL is expensive, slow, and unsafe. Training a robot arm in the real world requires the arm to physically move millions of times — wearing down hardware and risking damage. But simulation is imperfect: a policy trained in simulation performs poorly in reality because the simulation does not match the real world exactly (the "reality gap").

**The core insight:** If the policy cannot overfit to any single simulated world because the world keeps changing, it must learn behavior that works across many simulated worlds. If the real world is just one more sample from that distribution, the policy generalizes to it.

**The mechanics:** Randomize simulation parameters during training.

### Domain Randomization

**The problem:** Simulation parameters (friction, mass, lighting) are fixed in a single simulation but differ from the real world in unknown ways. A policy trained on one set of parameters will fail when those parameters change.

**The core insight:** Randomly vary simulation parameters during training. The policy must learn behavior that works across the full range. If the real world falls within the randomization range, the policy generalizes without any real-world data.

```python
# At each episode reset, randomize sim parameters
friction = np.random.uniform(0.5, 1.5)
mass = np.random.uniform(0.8, 1.2) * nominal_mass
lighting = np.random.uniform(0.5, 1.5)
env.reset(friction=friction, mass=mass, lighting=lighting)
```

**Visual domain randomization:** Randomize textures, colors, camera positions — forces visual policies to ignore non-physical visual details.

**What breaks:** Domain randomization requires knowing which parameters to randomize and their plausible ranges. Wrong ranges produce policies that are over-conservative (trying to work in physically implausible conditions) or under-conservative (real-world parameters fall outside the randomization range). This motivates system identification.

### System Identification

**The problem:** Domain randomization is conservative — it forces the policy to work everywhere, including physically implausible parameter combinations. This over-constrains the policy and reduces peak performance. Can we do better by measuring what the real world actually looks like?

**The core insight:** Collect real data with a simple exploratory policy, then fit simulation parameters to minimize prediction error. Now the simulation *is* the real world, to within measurement error. The policy is trained on a calibrated simulation, not a generic randomized one.

**The mechanics:**

```
Collect real trajectories with random actions
Fit sim parameters to minimize trajectory MSE
Retrain policy in calibrated simulation
```

**What breaks:** System identification requires a safe exploratory policy before the final policy is trained — a chicken-and-egg problem. The calibrated simulation only captures the real world at the time of identification; mechanical wear or environmental changes make it stale.

### Adaptive Methods (RMA — Rapid Motor Adaptation)

**The problem:** Both domain randomization and system identification produce a fixed policy. When real-world conditions change during deployment (different terrain, worn motors), the policy cannot adapt.

**The core insight:** Train a base policy in sim with privileged access to env parameters. Then train an adaptation module that infers those parameters from proprioceptive history (no privileged access). The adaptation module works in the real world because it only needs sensory history, not direct parameter access.

**Two-phase training:**
1. Train base policy π(a | s, e) where e is the privileged environment parameter vector.
2. Train adaptation module f(h_t) ≈ e where h_t is the recent proprioceptive history, without privileged access.

**Used in:** ANYmal (quadruped locomotion), Spot (Boston Dynamics), dexterous manipulation.

**What breaks:** RMA's adaptation module only works for parameter variations it saw during training; novel real-world failures (broken motor, unexpected surface) are out-of-distribution. The adaptation module introduces a lag — it requires a history of steps to infer environment parameters, so it adapts slowly to sudden changes.

### Sim-to-Real with Real Fine-Tuning

**The problem:** Even the best sim-to-real transfer leaves a gap that no amount of simulation training can close.

**The core insight:** Deploy sim-trained policy in real, collect real data, fine-tune with RL. The sim-trained policy provides a safe starting point; real fine-tuning closes the remaining gap.

Requires safe initial policy and careful reward shaping in real environment.

**What breaks (Sim-to-Real broadly):** Domain randomization requires knowing which parameters to randomize and their plausible ranges — wrong ranges produce policies that fail at real-world physics. System identification requires a safe exploratory policy before the final policy is trained. RMA's adaptation module only works for parameter variations it saw during training; novel real-world failures (broken motor, unexpected surface) are out-of-distribution.

---

## Offline / Batch RL

**The problem:** Online RL requires interacting with an environment to collect data. In healthcare, autonomous driving, and industrial control, online data collection is either too dangerous or too expensive. Hospitals have historical records of treatments and outcomes — can a policy be learned from that fixed dataset without any additional interactions?

**The core insight:** Q-learning only requires (s, a, r, s') transitions, not necessarily online collection. But there is a critical failure mode: the learned policy will take actions not seen in the historical data, and the Q-function has no reliable estimate for those out-of-distribution (OOD) actions. Extrapolation error causes Q-values for OOD actions to be wildly overestimated, and the policy exploits these phantom high-value actions.

**The mechanics:** Learn from a fixed dataset of pre-collected transitions. The key design choice is suppressing or avoiding OOD action queries.

### CQL — Conservative Q-Learning

**The problem:** Standard Q-learning on offline data overestimates Q-values for actions not in the dataset. The policy greedily selects these phantom high-value actions, which are never corrected by real experience.

**The core insight:** Add a penalty that minimizes Q-values for OOD actions while maximizing Q-values for in-distribution actions. Force the Q-function to be pessimistic about actions the dataset does not support.

`L_CQL = L_Bellman + α (E_{a~π}[Q(s,a)] - E_{a~D}[Q(s,a)])`

**What breaks:** Too large α makes the policy overly conservative — it performs no better than the behavior policy because it avoids all actions even slightly outside the dataset. Too small α allows OOD extrapolation. The right α is task-dependent and hard to set without real environment interaction.

### IQL — Implicit Q-Learning

**The problem:** CQL must evaluate the current policy's Q-values during training, which requires querying out-of-distribution actions. Even the penalty computation requires evaluating Q at OOD points.

**The core insight:** Avoid querying OOD actions entirely. Learn a value function V(s) via expectile regression that approximates the maximum Q-value over in-distribution actions — without ever querying the Q-function at OOD points. Derive the policy via advantage-weighted regression.

**What breaks:** Expectile regression approximates the maximum in a distributional sense; it does not guarantee the learned V matches the optimal value function. The implicit approximation introduces bias that can limit performance on tasks requiring significant policy improvement over the behavior policy.

### Decision Transformer

**The problem:** Even conservative Q-learning must query the Q-function for many candidate actions during policy extraction — still risky if the Q-function is imprecise. What if we avoid Q-learning entirely and treat the problem as sequence modeling?

**The core insight:** RL can be reframed as sequence prediction: given a desired return-to-go, predict the sequence of actions that achieves it. A Transformer trained on offline trajectories conditioned on return-to-go can produce high-return behavior by conditioning on a high desired return at test time. No Q-function, no Bellman backups.

**The mechanics:** Reframe RL as sequence modeling: `(R_1, s_1, a_1, R_2, s_2, a_2, ...)`. Given a desired return-to-go, predict actions autoregressively with a Transformer.

**What breaks:** Decision Transformer requires the dataset to contain high-return trajectories; it cannot stitch together suboptimal trajectories into a better policy the way Q-learning can — it is limited to imitating the best behavior in the dataset. It also fails to generalize to desired returns higher than anything seen in training.

**What breaks (Offline RL broadly):** All offline methods are limited by dataset coverage — if the optimal policy requires actions never taken in the data, no offline method can find them. CQL's conservatism trades off coverage against pessimism: too conservative and performance degrades toward the behavior policy; too permissive and OOD extrapolation fails. Decision Transformer requires the dataset to contain high-return trajectories; it cannot stitch together suboptimal trajectories into a better policy the way Q-learning can.

---

## Key Interview Points

- BC is simple but suffers from compounding error; DAgger fixes this by iteratively querying the expert on states the current policy actually visits.
- GAIL: discriminator distinguishes expert vs policy trajectories; PPO maximizes discriminator "fooling" → implicit imitation without reward design.
- IRL: learn a reward function that makes the expert's behavior optimal. MaxEntIRL is the classic formulation; RLHF preference learning is its modern application.
- MARL: non-stationarity is the core challenge — as agents learn, the environment each agent faces changes. CTDE (centralized training, decentralized execution) is the standard paradigm.
- HER: reuse failed trajectories by treating the final state as a hindsight goal — key for sparse reward robotics. Generates positive reward signal from every trajectory regardless of outcome.
- Domain randomization: vary sim parameters so the policy generalizes; the real world is "just another sample." The key assumption is that the real world falls within the randomization distribution.
- Offline RL needs conservative Q-estimates (CQL) or avoidance of OOD actions entirely (IQL, Decision Transformer) because the dataset is fixed and errors cannot be corrected by collecting new data.

